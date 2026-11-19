// SPDX-License-Identifier: Apache-2.0

//! [`RelationPlanner`] implementation lowering `sqlparser::ast::TableFactor::MatchRecognize`
//! into a [`MatchRecognizeNode`].

use std::collections::HashSet;

use datafusion_common::{DFSchema, Result, plan_datafusion_err, plan_err};
use datafusion_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use sqlparser::ast::{
    self, AfterMatchSkip as SqlSkip, BinaryOperator, EmptyMatchesMode as SqlEmptyMode,
    Expr as SqlExpr, FunctionArg, FunctionArgExpr, FunctionArguments, Ident,
    MatchRecognizePattern as SqlPattern, MatchRecognizeSymbol as SqlSymbol,
    RepetitionQuantifier as SqlQuant, RowsPerMatch as SqlRowsPerMatch, TableFactor, UnaryOperator,
};

use crate::ast::{
    AfterMatchSkip, AggKind, EvalSemantics, LogicalNavKind, MatchRecognizeConfig, MeasureSlot,
    MeasureSlotKind, MeasureSpec, OrderBySpec, Pattern, PhysicalNavKind, RowsPerMatch, SymbolDef,
    SymbolRef,
};
use crate::logical::{MatchRecognizeNode, build_slot_schema};
use crate::pattern::Program;

/// DataFusion [`RelationPlanner`] handling SQL:2016 `MATCH_RECOGNIZE`.
#[derive(Debug, Clone, Default)]
pub struct MatchRecognizeRelationPlanner {
    pub config: MatchRecognizeConfig,
}

impl MatchRecognizeRelationPlanner {
    /// Creates a relation planner with custom compilation limits.
    pub fn new(config: MatchRecognizeConfig) -> Self {
        Self { config }
    }
}

impl RelationPlanner for MatchRecognizeRelationPlanner {
    fn plan_relation(
        &self,
        relation: TableFactor,
        ctx: &mut dyn RelationPlannerContext,
    ) -> Result<RelationPlanning> {
        let TableFactor::MatchRecognize {
            table,
            partition_by,
            order_by,
            measures,
            rows_per_match,
            after_match_skip,
            pattern,
            symbols,
            alias,
        } = relation
        else {
            return Ok(RelationPlanning::Original(Box::new(relation)));
        };

        let input = ctx.plan(*table)?;
        let input_schema = input.schema().clone();

        let planned_partition_by = partition_by
            .into_iter()
            .map(|e| ctx.sql_to_expr(e, input_schema.as_ref()))
            .collect::<Result<Vec<_>>>()?;

        let planned_order_by = order_by
            .into_iter()
            .map(|ob| {
                let expr = ctx.sql_to_expr(ob.expr, input_schema.as_ref())?;
                let asc = match ob.options.sort {
                    None | Some(ast::OrderBySort::Asc) => true,
                    Some(ast::OrderBySort::Desc) => false,
                    Some(ast::OrderBySort::Using(op)) => {
                        return plan_err!(
                            "ORDER BY USING ({op}) is not supported in MATCH_RECOGNIZE"
                        );
                    }
                };
                let nulls_first = ob.options.nulls_first.unwrap_or(!asc);
                Ok(OrderBySpec {
                    expr,
                    asc,
                    nulls_first,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let planned_rows_per_match = lower_rows_per_match(rows_per_match);
        let planned_skip = lower_after_match_skip(after_match_skip, ctx);
        let planned_pattern = lower_pattern(pattern, ctx)?;

        // SQL:2016 forbids exclusions with ALL ROWS PER MATCH WITH UNMATCHED ROWS.
        if planned_rows_per_match.include_unmatched_rows() && planned_pattern.has_exclusion() {
            return plan_err!(
                "MATCH_RECOGNIZE pattern exclusion {{- ... -}} is not allowed with ALL ROWS PER MATCH WITH UNMATCHED ROWS"
            );
        }

        // Collect all known pattern variable names (from DEFINE + PATTERN).
        let mut known_symbols: HashSet<String> = HashSet::new();
        let mut pattern_symbols = Vec::new();
        planned_pattern.collect_named_symbols(&mut pattern_symbols);
        for s in &pattern_symbols {
            known_symbols.insert(s.clone());
        }

        // Validate DEFINE definitions (Tier 1 causal predicates).
        // Per SQL:2016, every variable in DEFINE must be a primary pattern variable in PATTERN.
        let mut planned_definitions = Vec::with_capacity(symbols.len());
        let mut seen_defs: HashSet<String> = HashSet::new();
        for sym_def in symbols {
            let name = ctx.normalize_ident(sym_def.symbol);
            if !known_symbols.contains(&name) {
                return plan_err!(
                    "Defined variable '{name}' is not a primary pattern variable in PATTERN"
                );
            }
            if !seen_defs.insert(name.clone()) {
                return plan_err!("Duplicate pattern variable '{name}' in DEFINE clause");
            }
            let rewritten =
                strip_self_symbol_qualifier(sym_def.definition, &name, &known_symbols, ctx)?;
            let predicate = ctx.sql_to_expr(rewritten, input_schema.as_ref())?;
            planned_definitions.push(SymbolDef { name, predicate });
        }

        // Validate AFTER MATCH SKIP target variable exists.
        if let AfterMatchSkip::ToFirst(target) | AfterMatchSkip::ToLast(target) = &planned_skip
            && !known_symbols.contains(target)
        {
            return plan_err!("AFTER MATCH SKIP references undefined pattern variable '{target}'");
        }

        // Validate pattern compiles cleanly within limits at plan time.
        let declared_names: Vec<String> =
            planned_definitions.iter().map(|d| d.name.clone()).collect();
        Program::compile(&planned_pattern, &declared_names, self.config)?;

        // Plan MEASURES via slot extraction + outer standard DataFusion Expr.
        let default_semantics = if planned_rows_per_match.is_one_row() {
            EvalSemantics::Final
        } else {
            EvalSemantics::Running
        };

        let mut planned_measures = Vec::with_capacity(measures.len());
        for m in measures {
            let measure_alias = ctx.normalize_ident(m.alias);
            let mut extractor = MeasureExtractor {
                ctx,
                input_schema: input_schema.as_ref(),
                known_symbols: &known_symbols,
                rows_per_match: planned_rows_per_match,
                default_semantics,
                slots: Vec::new(),
            };
            let rewritten_sql = extractor.rewrite_outer(m.expr)?;
            let slots = extractor.slots;
            let slot_schema =
                build_slot_schema(input_schema.as_ref(), &slots, planned_rows_per_match)?;
            let outer_expr = ctx.sql_to_expr(rewritten_sql, &slot_schema)?;
            planned_measures.push(MeasureSpec {
                alias: measure_alias,
                outer_expr,
                slots,
            });
        }

        let node = MatchRecognizeNode::try_new(
            input,
            planned_partition_by,
            planned_order_by,
            planned_measures,
            planned_rows_per_match,
            planned_skip,
            planned_pattern,
            planned_definitions,
            self.config,
        )?;

        Ok(RelationPlanning::Planned(Box::new(PlannedRelation::new(
            node.into_plan(),
            alias,
        ))))
    }
}

fn lower_rows_per_match(rpm: Option<SqlRowsPerMatch>) -> RowsPerMatch {
    match rpm {
        None | Some(SqlRowsPerMatch::OneRow) => RowsPerMatch::OneRow,
        Some(SqlRowsPerMatch::AllRows(None | Some(SqlEmptyMode::Show))) => {
            RowsPerMatch::AllRowsShowEmpty
        }
        Some(SqlRowsPerMatch::AllRows(Some(SqlEmptyMode::Omit))) => RowsPerMatch::AllRowsOmitEmpty,
        Some(SqlRowsPerMatch::AllRows(Some(SqlEmptyMode::WithUnmatched))) => {
            RowsPerMatch::AllRowsWithUnmatched
        }
    }
}

fn lower_after_match_skip(
    skip: Option<SqlSkip>,
    ctx: &dyn RelationPlannerContext,
) -> AfterMatchSkip {
    match skip {
        None | Some(SqlSkip::PastLastRow) => AfterMatchSkip::PastLastRow,
        Some(SqlSkip::ToNextRow) => AfterMatchSkip::ToNextRow,
        Some(SqlSkip::ToFirst(id)) => AfterMatchSkip::ToFirst(ctx.normalize_ident(id)),
        Some(SqlSkip::ToLast(id)) => AfterMatchSkip::ToLast(ctx.normalize_ident(id)),
    }
}

fn lower_pattern(pattern: SqlPattern, ctx: &dyn RelationPlannerContext) -> Result<Pattern> {
    match pattern {
        SqlPattern::Symbol(sym) => Ok(Pattern::Symbol(lower_symbol(sym, ctx))),
        SqlPattern::Exclude(sym) => Ok(Pattern::Exclude(Box::new(Pattern::Symbol(lower_symbol(
            sym, ctx,
        ))))),
        SqlPattern::Permute(syms) => Ok(Pattern::Permute(
            syms.into_iter()
                .map(|s| Pattern::Symbol(lower_symbol(s, ctx)))
                .collect(),
        )),
        SqlPattern::Concat(items) => {
            let mut lowered = Vec::with_capacity(items.len());
            for item in items {
                lowered.push(lower_pattern(item, ctx)?);
            }
            Ok(Pattern::Concat(lowered))
        }
        SqlPattern::Group(inner) => lower_pattern(*inner, ctx),
        SqlPattern::Alternation(items) => {
            let mut lowered = Vec::with_capacity(items.len());
            for item in items {
                lowered.push(lower_pattern(item, ctx)?);
            }
            Ok(Pattern::Alternation(lowered))
        }
        SqlPattern::Repetition(inner, quant) => {
            let (min, max, reluctant) = lower_quantifier(quant);
            Ok(Pattern::Repetition {
                inner: Box::new(lower_pattern(*inner, ctx)?),
                min,
                max,
                reluctant,
            })
        }
    }
}

fn lower_symbol(sym: SqlSymbol, ctx: &dyn RelationPlannerContext) -> SymbolRef {
    match sym {
        SqlSymbol::Named(id) => SymbolRef::Named(ctx.normalize_ident(id)),
        SqlSymbol::Start => SymbolRef::Start,
        SqlSymbol::End => SymbolRef::End,
    }
}

fn lower_quantifier(quant: SqlQuant) -> (u32, Option<u32>, bool) {
    match quant {
        SqlQuant::ZeroOrMore => (0, None, false),
        SqlQuant::OneOrMore => (1, None, false),
        SqlQuant::AtMostOne => (0, Some(1), false),
        SqlQuant::Exactly(n) => (n, Some(n), false),
        SqlQuant::AtLeast(n) => (n, None, false),
        SqlQuant::AtMost(m) => (0, Some(m), false),
        SqlQuant::Range(n, m) => (n, Some(m), false),
    }
}

/// Rewrites `A.col` inside `DEFINE A AS ...` to bare `col`, and rejects cross-symbol or
/// match-state references (`PREV`, `NEXT`, `FIRST`, `LAST`, `CLASSIFIER`, `MATCH_NUMBER`)
/// in Tier 1 causal DEFINE mode with a clear error message.
fn strip_self_symbol_qualifier(
    expr: SqlExpr,
    current_symbol: &str,
    known_symbols: &HashSet<String>,
    ctx: &dyn RelationPlannerContext,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::CompoundIdentifier(mut idents) => {
            if idents.len() == 2 {
                let prefix = ctx.normalize_ident(idents[0].clone());
                if prefix == current_symbol {
                    return Ok(SqlExpr::Identifier(idents.pop().expect("len 2")));
                }
                if known_symbols.contains(&prefix) {
                    return plan_err!(
                        "Cross-symbol reference '{prefix}.{}' in DEFINE '{current_symbol}' requires stateful DEFINE (Phase 2)",
                        idents[1]
                    );
                }
            }
            Ok(SqlExpr::CompoundIdentifier(idents))
        }
        SqlExpr::Function(func) => {
            let fname = func.name.to_string().to_ascii_lowercase();
            if matches!(
                fname.as_str(),
                "prev" | "next" | "first" | "last" | "classifier" | "match_number"
            ) {
                return plan_err!(
                    "Navigation/pattern function '{fname}()' inside DEFINE '{current_symbol}' requires stateful DEFINE (Phase 2)"
                );
            }
            let args = match func.args {
                FunctionArguments::List(mut list) => {
                    for arg in &mut list.args {
                        if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                            let owned =
                                std::mem::replace(e, SqlExpr::Value(ast::Value::Null.into()));
                            *e = strip_self_symbol_qualifier(
                                owned,
                                current_symbol,
                                known_symbols,
                                ctx,
                            )?;
                        }
                    }
                    FunctionArguments::List(list)
                }
                other => other,
            };
            Ok(SqlExpr::Function(ast::Function { args, ..func }))
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(strip_self_symbol_qualifier(
                *left,
                current_symbol,
                known_symbols,
                ctx,
            )?),
            op,
            right: Box::new(strip_self_symbol_qualifier(
                *right,
                current_symbol,
                known_symbols,
                ctx,
            )?),
        }),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op,
            expr: Box::new(strip_self_symbol_qualifier(
                *expr,
                current_symbol,
                known_symbols,
                ctx,
            )?),
        }),
        SqlExpr::Nested(inner) => Ok(SqlExpr::Nested(Box::new(strip_self_symbol_qualifier(
            *inner,
            current_symbol,
            known_symbols,
            ctx,
        )?))),
        SqlExpr::IsNull(inner) => Ok(SqlExpr::IsNull(Box::new(strip_self_symbol_qualifier(
            *inner,
            current_symbol,
            known_symbols,
            ctx,
        )?))),
        SqlExpr::IsNotNull(inner) => Ok(SqlExpr::IsNotNull(Box::new(strip_self_symbol_qualifier(
            *inner,
            current_symbol,
            known_symbols,
            ctx,
        )?))),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(SqlExpr::Between {
            expr: Box::new(strip_self_symbol_qualifier(
                *expr,
                current_symbol,
                known_symbols,
                ctx,
            )?),
            negated,
            low: Box::new(strip_self_symbol_qualifier(
                *low,
                current_symbol,
                known_symbols,
                ctx,
            )?),
            high: Box::new(strip_self_symbol_qualifier(
                *high,
                current_symbol,
                known_symbols,
                ctx,
            )?),
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => {
            let expr = Box::new(strip_self_symbol_qualifier(
                *expr,
                current_symbol,
                known_symbols,
                ctx,
            )?);
            let list = list
                .into_iter()
                .map(|e| strip_self_symbol_qualifier(e, current_symbol, known_symbols, ctx))
                .collect::<Result<Vec<_>>>()?;
            Ok(SqlExpr::InList {
                expr,
                list,
                negated,
            })
        }
        SqlExpr::Case {
            case_token,
            end_token,
            operand,
            conditions,
            else_result,
        } => {
            let operand = operand
                .map(|e| strip_self_symbol_qualifier(*e, current_symbol, known_symbols, ctx))
                .transpose()?
                .map(Box::new);
            let mut new_conditions = Vec::with_capacity(conditions.len());
            for c in conditions {
                new_conditions.push(ast::CaseWhen {
                    condition: strip_self_symbol_qualifier(
                        c.condition,
                        current_symbol,
                        known_symbols,
                        ctx,
                    )?,
                    result: strip_self_symbol_qualifier(
                        c.result,
                        current_symbol,
                        known_symbols,
                        ctx,
                    )?,
                });
            }
            let else_result = else_result
                .map(|e| strip_self_symbol_qualifier(*e, current_symbol, known_symbols, ctx))
                .transpose()?
                .map(Box::new);
            Ok(SqlExpr::Case {
                case_token,
                end_token,
                operand,
                conditions: new_conditions,
                else_result,
            })
        }
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => Ok(SqlExpr::Cast {
            kind,
            expr: Box::new(strip_self_symbol_qualifier(
                *expr,
                current_symbol,
                known_symbols,
                ctx,
            )?),
            data_type,
            format,
        }),
        other => Ok(other),
    }
}

struct MeasureExtractor<'a, 'b> {
    ctx: &'a mut dyn RelationPlannerContext,
    input_schema: &'b DFSchema,
    known_symbols: &'b HashSet<String>,
    rows_per_match: RowsPerMatch,
    default_semantics: EvalSemantics,
    slots: Vec<MeasureSlot>,
}

impl<'a, 'b> MeasureExtractor<'a, 'b> {
    fn next_slot_name(&self) -> String {
        format!("__mr_slot_{}", self.slots.len())
    }

    fn push_slot(&mut self, kind: MeasureSlotKind) -> SqlExpr {
        let slot_name = self.next_slot_name();
        self.slots.push(MeasureSlot {
            slot_name: slot_name.clone(),
            semantics: self.default_semantics,
            kind,
        });
        SqlExpr::Identifier(Ident::new(slot_name))
    }

    /// Rewrites a SQL measure expression by replacing pattern-recognition leaves with synthetic
    /// slot identifiers while preserving arbitrary outer scalar expressions.
    fn rewrite_outer(&mut self, expr: SqlExpr) -> Result<SqlExpr> {
        match expr {
            SqlExpr::Function(func) => {
                let fname = func.name.to_string().to_ascii_lowercase();
                match fname.as_str() {
                    "match_number" => {
                        let args = extract_unnamed_args(&func.args)?;
                        if !args.is_empty() {
                            return plan_err!("MATCH_NUMBER() takes no arguments");
                        }
                        Ok(self.push_slot(MeasureSlotKind::MatchNumber))
                    }
                    "classifier" => {
                        let args = extract_unnamed_args(&func.args)?;
                        let symbol = match args.len() {
                            0 => None,
                            1 => match &args[0] {
                                FunctionArgExpr::Expr(SqlExpr::Identifier(id)) => {
                                    let sym = self.ctx.normalize_ident(id.clone());
                                    if !self.known_symbols.contains(&sym) {
                                        return plan_err!(
                                            "CLASSIFIER({sym}) references unknown pattern variable"
                                        );
                                    }
                                    Some(sym)
                                }
                                _ => {
                                    return plan_err!(
                                        "CLASSIFIER() argument must be a pattern variable identifier"
                                    );
                                }
                            },
                            _ => return plan_err!("CLASSIFIER() takes at most 1 argument"),
                        };
                        Ok(self.push_slot(MeasureSlotKind::Classifier { symbol }))
                    }
                    "first" | "last" | "prev" | "next" => {
                        let slot_kind = self.lower_nav_function(&fname, &func.args)?;
                        Ok(self.push_slot(slot_kind))
                    }
                    "count" | "sum" | "avg" | "min" | "max" => {
                        let slot_kind = self.lower_agg_function(&fname, &func.args)?;
                        Ok(self.push_slot(slot_kind))
                    }
                    _ => {
                        // Ordinary scalar function (COALESCE, ABS, ROUND, CONCAT, UDFs, etc.):
                        // recursively rewrite its arguments.
                        let args = match func.args {
                            FunctionArguments::List(mut list) => {
                                for arg in &mut list.args {
                                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                                        let owned = std::mem::replace(
                                            e,
                                            SqlExpr::Value(ast::Value::Null.into()),
                                        );
                                        *e = self.rewrite_outer(owned)?;
                                    }
                                }
                                FunctionArguments::List(list)
                            }
                            other => other,
                        };
                        Ok(SqlExpr::Function(ast::Function { args, ..func }))
                    }
                }
            }
            SqlExpr::CompoundIdentifier(idents) => {
                if idents.len() == 2 {
                    let prefix = self.ctx.normalize_ident(idents[0].clone());
                    if self.known_symbols.contains(&prefix) {
                        let col_ident = idents[1].clone();
                        let inner_expr = self
                            .ctx
                            .sql_to_expr(SqlExpr::Identifier(col_ident), self.input_schema)?;
                        return Ok(self.push_slot(MeasureSlotKind::Nav {
                            logical_nav: Some(LogicalNavKind::Last),
                            logical_offset: 0,
                            physical_nav: None,
                            physical_offset: 0,
                            symbol: Some(prefix),
                            inner_expr,
                        }));
                    }
                }
                // Table-qualified input column reference
                let inner_expr = self
                    .ctx
                    .sql_to_expr(SqlExpr::CompoundIdentifier(idents), self.input_schema)?;
                Ok(self.push_bare_column_slot(inner_expr))
            }
            SqlExpr::Identifier(ident) => {
                let inner_expr = self
                    .ctx
                    .sql_to_expr(SqlExpr::Identifier(ident), self.input_schema)?;
                Ok(self.push_bare_column_slot(inner_expr))
            }
            SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
                left: Box::new(self.rewrite_outer(*left)?),
                op,
                right: Box::new(self.rewrite_outer(*right)?),
            }),
            SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
                op,
                expr: Box::new(self.rewrite_outer(*expr)?),
            }),
            SqlExpr::Nested(inner) => Ok(SqlExpr::Nested(Box::new(self.rewrite_outer(*inner)?))),
            SqlExpr::IsNull(inner) => Ok(SqlExpr::IsNull(Box::new(self.rewrite_outer(*inner)?))),
            SqlExpr::IsNotNull(inner) => {
                Ok(SqlExpr::IsNotNull(Box::new(self.rewrite_outer(*inner)?)))
            }
            SqlExpr::Cast {
                kind,
                expr,
                data_type,
                format,
            } => Ok(SqlExpr::Cast {
                kind,
                expr: Box::new(self.rewrite_outer(*expr)?),
                data_type,
                format,
            }),
            SqlExpr::Case {
                case_token,
                end_token,
                operand,
                conditions,
                else_result,
            } => {
                let operand = operand
                    .map(|e| self.rewrite_outer(*e))
                    .transpose()?
                    .map(Box::new);
                let mut new_conditions = Vec::with_capacity(conditions.len());
                for c in conditions {
                    new_conditions.push(ast::CaseWhen {
                        condition: self.rewrite_outer(c.condition)?,
                        result: self.rewrite_outer(c.result)?,
                    });
                }
                let else_result = else_result
                    .map(|e| self.rewrite_outer(*e))
                    .transpose()?
                    .map(Box::new);
                Ok(SqlExpr::Case {
                    case_token,
                    end_token,
                    operand,
                    conditions: new_conditions,
                    else_result,
                })
            }
            SqlExpr::Between {
                expr,
                negated,
                low,
                high,
            } => Ok(SqlExpr::Between {
                expr: Box::new(self.rewrite_outer(*expr)?),
                negated,
                low: Box::new(self.rewrite_outer(*low)?),
                high: Box::new(self.rewrite_outer(*high)?),
            }),
            SqlExpr::InList {
                expr,
                list,
                negated,
            } => {
                let expr = Box::new(self.rewrite_outer(*expr)?);
                let list = list
                    .into_iter()
                    .map(|e| self.rewrite_outer(e))
                    .collect::<Result<Vec<_>>>()?;
                Ok(SqlExpr::InList {
                    expr,
                    list,
                    negated,
                })
            }
            // Literals and other leaf constants pass straight through.
            other => Ok(other),
        }
    }

    fn push_bare_column_slot(&mut self, inner_expr: datafusion_expr::Expr) -> SqlExpr {
        if self.rows_per_match.is_one_row() {
            self.push_slot(MeasureSlotKind::Nav {
                logical_nav: Some(LogicalNavKind::Last),
                logical_offset: 0,
                physical_nav: None,
                physical_offset: 0,
                symbol: None,
                inner_expr,
            })
        } else {
            self.push_slot(MeasureSlotKind::CurrentRow { inner_expr })
        }
    }

    fn lower_nav_function(
        &mut self,
        fname: &str,
        args: &FunctionArguments,
    ) -> Result<MeasureSlotKind> {
        let arg_exprs = extract_unnamed_args(args)?;
        if arg_exprs.is_empty() || arg_exprs.len() > 2 {
            return plan_err!(
                "{fname}() expects 1 or 2 arguments, got {}",
                arg_exprs.len()
            );
        }
        let target_sql = match &arg_exprs[0] {
            FunctionArgExpr::Expr(e) => e.clone(),
            _ => return plan_err!("{fname}() first argument must be an expression"),
        };

        let offset = if arg_exprs.len() == 2 {
            match &arg_exprs[1] {
                FunctionArgExpr::Expr(e) => parse_non_negative_u32(e)?,
                _ => return plan_err!("{fname}() offset must be a non-negative integer literal"),
            }
        } else if matches!(fname, "prev" | "next") {
            1
        } else {
            0
        };

        match fname {
            "first" | "last" => {
                let logical_nav = if fname == "first" {
                    LogicalNavKind::First
                } else {
                    LogicalNavKind::Last
                };
                let (symbol, stripped) =
                    extract_single_symbol_qualifier(target_sql, self.known_symbols, self.ctx)?;
                let inner_expr = self.ctx.sql_to_expr(stripped, self.input_schema)?;
                Ok(MeasureSlotKind::Nav {
                    logical_nav: Some(logical_nav),
                    logical_offset: offset,
                    physical_nav: None,
                    physical_offset: 0,
                    symbol,
                    inner_expr,
                })
            }
            "prev" | "next" => {
                let physical_nav = if fname == "prev" {
                    PhysicalNavKind::Prev
                } else {
                    PhysicalNavKind::Next
                };
                // Support nested physical(logical(expr)), e.g. PREV(FIRST(A.x, 1), 1).
                let (logical_nav, logical_offset, inner_target) =
                    if let SqlExpr::Function(inner_func) = &target_sql {
                        let inner_name = inner_func.name.to_string().to_ascii_lowercase();
                        if matches!(inner_name.as_str(), "first" | "last") {
                            let inner_args = extract_unnamed_args(&inner_func.args)?;
                            if inner_args.is_empty() || inner_args.len() > 2 {
                                return plan_err!("{inner_name}() expects 1 or 2 arguments");
                            }
                            let expr = match &inner_args[0] {
                                FunctionArgExpr::Expr(e) => e.clone(),
                                _ => return plan_err!("{inner_name}() expects an expression"),
                            };
                            let l_off = if inner_args.len() == 2 {
                                match &inner_args[1] {
                                    FunctionArgExpr::Expr(e) => parse_non_negative_u32(e)?,
                                    _ => return plan_err!("{inner_name}() offset must be an int"),
                                }
                            } else {
                                0
                            };
                            let kind = if inner_name == "first" {
                                LogicalNavKind::First
                            } else {
                                LogicalNavKind::Last
                            };
                            (Some(kind), l_off, expr)
                        } else {
                            (Some(LogicalNavKind::Last), 0, target_sql)
                        }
                    } else {
                        (Some(LogicalNavKind::Last), 0, target_sql)
                    };

                let (symbol, stripped) =
                    extract_single_symbol_qualifier(inner_target, self.known_symbols, self.ctx)?;
                let inner_expr = self.ctx.sql_to_expr(stripped, self.input_schema)?;
                Ok(MeasureSlotKind::Nav {
                    logical_nav,
                    logical_offset,
                    physical_nav: Some(physical_nav),
                    physical_offset: offset,
                    symbol,
                    inner_expr,
                })
            }
            _ => unreachable!(),
        }
    }

    fn lower_agg_function(
        &mut self,
        fname: &str,
        args: &FunctionArguments,
    ) -> Result<MeasureSlotKind> {
        let arg_exprs = extract_unnamed_args(args)?;
        if arg_exprs.len() != 1 {
            return plan_err!(
                "{fname}() in MEASURES expects 1 argument, got {}",
                arg_exprs.len()
            );
        }

        match &arg_exprs[0] {
            FunctionArgExpr::Wildcard => {
                if fname != "count" {
                    return plan_err!("{fname}(*) is not valid in MEASURES");
                }
                Ok(MeasureSlotKind::Agg {
                    agg_kind: AggKind::CountStar,
                    symbol: None,
                    inner_expr: None,
                })
            }
            FunctionArgExpr::QualifiedWildcard(obj_name) => {
                if fname != "count" {
                    return plan_err!("{fname}({obj_name}.*) is not valid in MEASURES");
                }
                let ident = obj_name
                    .0
                    .last()
                    .and_then(|part| part.as_ident())
                    .ok_or_else(|| plan_datafusion_err!("Invalid qualified wildcard in COUNT"))?;
                let sym = self.ctx.normalize_ident(ident.clone());
                if !self.known_symbols.contains(&sym) {
                    return plan_err!("COUNT({sym}.*) references unknown pattern variable '{sym}'");
                }
                Ok(MeasureSlotKind::Agg {
                    agg_kind: AggKind::CountSymbol,
                    symbol: Some(sym),
                    inner_expr: None,
                })
            }
            FunctionArgExpr::Expr(sql_expr) => {
                let agg_kind = match fname {
                    "count" => AggKind::Count,
                    "sum" => AggKind::Sum,
                    "avg" => AggKind::Avg,
                    "min" => AggKind::Min,
                    "max" => AggKind::Max,
                    _ => unreachable!(),
                };
                let (symbol, stripped) = extract_single_symbol_qualifier(
                    sql_expr.clone(),
                    self.known_symbols,
                    self.ctx,
                )?;
                let inner_expr = self.ctx.sql_to_expr(stripped, self.input_schema)?;
                Ok(MeasureSlotKind::Agg {
                    agg_kind,
                    symbol,
                    inner_expr: Some(inner_expr),
                })
            }
            FunctionArgExpr::WildcardWithOptions(_) => {
                plan_err!("Wildcard options are not supported in MEASURES aggregates")
            }
        }
    }
}

fn extract_unnamed_args(args: &FunctionArguments) -> Result<Vec<FunctionArgExpr>> {
    match args {
        FunctionArguments::None => Ok(vec![]),
        FunctionArguments::List(list) => {
            if list.duplicate_treatment.is_some() {
                return plan_err!(
                    "DISTINCT/ALL aggregate modifiers are not yet supported in MEASURES"
                );
            }
            let mut out = Vec::with_capacity(list.args.len());
            for arg in &list.args {
                match arg {
                    FunctionArg::Unnamed(expr) => out.push(expr.clone()),
                    _ => {
                        return plan_err!(
                            "Named arguments are not supported in MEASURES functions"
                        );
                    }
                }
            }
            Ok(out)
        }
        FunctionArguments::Subquery(_) => {
            plan_err!("Subquery arguments are not supported in MEASURES functions")
        }
    }
}

fn parse_non_negative_u32(expr: &SqlExpr) -> Result<u32> {
    match expr {
        SqlExpr::Value(v) => match &v.value {
            ast::Value::Number(s, _) => s.parse::<u32>().map_err(|_| {
                plan_datafusion_err!("Navigation offset must be a non-negative integer, got '{s}'")
            }),
            other => plan_err!("Navigation offset must be an integer literal, got {other}"),
        },
        SqlExpr::UnaryOp {
            op: UnaryOperator::Plus,
            expr,
        } => parse_non_negative_u32(expr),
        other => plan_err!("Navigation offset must be a non-negative integer literal, got {other}"),
    }
}

/// Strips pattern-variable prefixes (`A.col` -> `col`) inside a single navigation or aggregate
/// argument and verifies that at most one distinct pattern variable is referenced.
fn extract_single_symbol_qualifier(
    expr: SqlExpr,
    known_symbols: &HashSet<String>,
    ctx: &dyn RelationPlannerContext,
) -> Result<(Option<String>, SqlExpr)> {
    let mut found: Option<String> = None;
    let stripped = strip_symbol_inner(expr, known_symbols, ctx, &mut found)?;
    Ok((found, stripped))
}

fn strip_symbol_inner(
    expr: SqlExpr,
    known_symbols: &HashSet<String>,
    ctx: &dyn RelationPlannerContext,
    found: &mut Option<String>,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::CompoundIdentifier(mut idents) => {
            if idents.len() == 2 {
                let prefix = ctx.normalize_ident(idents[0].clone());
                if known_symbols.contains(&prefix) {
                    if let Some(existing) = found
                        && existing != &prefix
                    {
                        return plan_err!(
                            "Navigation/aggregate expression cannot reference multiple pattern variables ('{existing}' and '{prefix}')"
                        );
                    }
                    *found = Some(prefix);
                    return Ok(SqlExpr::Identifier(idents.pop().expect("len 2")));
                }
            }
            Ok(SqlExpr::CompoundIdentifier(idents))
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(strip_symbol_inner(*left, known_symbols, ctx, found)?),
            op: op_clone(op),
            right: Box::new(strip_symbol_inner(*right, known_symbols, ctx, found)?),
        }),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op,
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, ctx, found)?),
        }),
        SqlExpr::Nested(inner) => Ok(SqlExpr::Nested(Box::new(strip_symbol_inner(
            *inner,
            known_symbols,
            ctx,
            found,
        )?))),
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => Ok(SqlExpr::Cast {
            kind,
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, ctx, found)?),
            data_type,
            format,
        }),
        SqlExpr::Function(func) => {
            let args = match func.args {
                FunctionArguments::List(mut list) => {
                    for arg in &mut list.args {
                        if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                            let owned =
                                std::mem::replace(e, SqlExpr::Value(ast::Value::Null.into()));
                            *e = strip_symbol_inner(owned, known_symbols, ctx, found)?;
                        }
                    }
                    FunctionArguments::List(list)
                }
                other => other,
            };
            Ok(SqlExpr::Function(ast::Function { args, ..func }))
        }
        SqlExpr::Case {
            case_token,
            end_token,
            operand,
            conditions,
            else_result,
        } => {
            let operand = operand
                .map(|e| strip_symbol_inner(*e, known_symbols, ctx, found))
                .transpose()?
                .map(Box::new);
            let mut new_conditions = Vec::with_capacity(conditions.len());
            for c in conditions {
                new_conditions.push(ast::CaseWhen {
                    condition: strip_symbol_inner(c.condition, known_symbols, ctx, found)?,
                    result: strip_symbol_inner(c.result, known_symbols, ctx, found)?,
                });
            }
            let else_result = else_result
                .map(|e| strip_symbol_inner(*e, known_symbols, ctx, found))
                .transpose()?
                .map(Box::new);
            Ok(SqlExpr::Case {
                case_token,
                end_token,
                operand,
                conditions: new_conditions,
                else_result,
            })
        }
        other => Ok(other),
    }
}

#[inline]
fn op_clone(op: BinaryOperator) -> BinaryOperator {
    op
}
