// SPDX-License-Identifier: Apache-2.0

//! [`RelationPlanner`] implementation lowering `sqlparser::ast::TableFactor::MatchRecognize`
//! into a [`MatchRecognizeNode`], supporting Tier 1 stateless `DEFINE`, Tier 2 stateful `DEFINE`,
//! navigation over `CLASSIFIER()`, `PREV`/`NEXT` inside aggregates, and arbitrary registered
//! DataFusion [`AggregateUDF`]s.

use std::collections::HashSet;
use std::sync::Arc;

use datafusion_common::{DFSchema, Result, plan_datafusion_err, plan_err};
use datafusion_expr::AggregateUDF;
use datafusion_expr::planner::{
    PlannedRelation, RelationPlanner, RelationPlannerContext, RelationPlanning,
};
use sqlparser::ast::{
    self, AfterMatchSkip as SqlSkip, BinaryOperator, DuplicateTreatment,
    EmptyMatchesMode as SqlEmptyMode, Expr as SqlExpr, FunctionArg, FunctionArgExpr,
    FunctionArguments, Ident, MatchRecognizePattern as SqlPattern,
    MatchRecognizeSymbol as SqlSymbol, RepetitionQuantifier as SqlQuant,
    RowsPerMatch as SqlRowsPerMatch, TableFactor, UnaryOperator,
};

use crate::ast::{
    AfterMatchSkip, AggKind, EvalSemantics, LogicalNavKind, MatchRecognizeConfig, MeasureSlot,
    MeasureSlotKind, MeasureSpec, OrderBySpec, Pattern, PhysicalNavKind, RowsPerMatch, SymbolDef,
    SymbolDefKind, SymbolRef,
};
use crate::logical::{
    AGG_META_CLASSIFIER_COL, AGG_META_MATCH_NUMBER_COL, MatchRecognizeNode,
    build_agg_metadata_dfschema, build_slot_schema,
};
use crate::pattern::Program;

/// Normalizes a SQL:2016 row pattern variable identifier:
/// - Unquoted identifiers fold to uppercase (`a` -> `"A"`), matching SQL:2016 and Trino `CLASSIFIER()`.
/// - Quoted identifiers preserve exact case (`"b"` -> `"b"`).
fn normalize_pattern_ident(id: Ident) -> String {
    if id.quote_style.is_some() {
        id.value
    } else {
        id.value.to_ascii_uppercase()
    }
}

/// DataFusion [`RelationPlanner`] handling SQL:2016 `MATCH_RECOGNIZE`.
#[derive(Debug, Clone, Default)]
pub struct MatchRecognizeRelationPlanner {
    pub config: MatchRecognizeConfig,
}

impl MatchRecognizeRelationPlanner {
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
        let planned_skip = lower_after_match_skip(after_match_skip);
        let planned_pattern = lower_pattern(pattern)?;

        if planned_rows_per_match.include_unmatched_rows() && planned_pattern.has_exclusion() {
            return plan_err!(
                "MATCH_RECOGNIZE pattern exclusion {{- ... -}} is not allowed with ALL ROWS PER MATCH WITH UNMATCHED ROWS"
            );
        }

        let mut known_symbols: HashSet<String> = HashSet::new();
        let mut pattern_symbols = Vec::new();
        planned_pattern.collect_named_symbols(&mut pattern_symbols);
        for s in &pattern_symbols {
            known_symbols.insert(s.clone());
        }

        let mut planned_definitions = Vec::with_capacity(symbols.len());
        let mut seen_defs: HashSet<String> = HashSet::new();
        for sym_def in symbols {
            let name = normalize_pattern_ident(sym_def.symbol);
            if !known_symbols.contains(&name) {
                return plan_err!(
                    "Defined variable '{name}' is not a primary pattern variable in PATTERN"
                );
            }
            if !seen_defs.insert(name.clone()) {
                return plan_err!("Duplicate pattern variable '{name}' in DEFINE clause");
            }

            if is_stateful_define(&sym_def.definition, &name, &known_symbols, ctx) {
                let mut extractor = SlotExtractor {
                    ctx,
                    input_schema: input_schema.as_ref(),
                    known_symbols: &known_symbols,
                    bare_col_is_current_row: true,
                    default_semantics: EvalSemantics::Running,
                    current_define_symbol: Some(&name),
                    slots: Vec::new(),
                };
                let rewritten_sql = extractor.rewrite_outer(sym_def.definition)?;
                let slots = extractor.slots;
                let slot_schema =
                    build_slot_schema(input_schema.as_ref(), &slots, RowsPerMatch::OneRow)?;
                let outer_expr = ctx.sql_to_expr(rewritten_sql, &slot_schema)?;
                planned_definitions.push(SymbolDef {
                    name,
                    kind: SymbolDefKind::Stateful { outer_expr, slots },
                });
            } else {
                let stripped = strip_self_qualifier(sym_def.definition, &name);
                let predicate = ctx.sql_to_expr(stripped, input_schema.as_ref())?;
                planned_definitions.push(SymbolDef {
                    name,
                    kind: SymbolDefKind::Stateless { predicate },
                });
            }
        }

        if let AfterMatchSkip::ToFirst(target) | AfterMatchSkip::ToLast(target) = &planned_skip
            && !known_symbols.contains(target)
        {
            return plan_err!("AFTER MATCH SKIP references undefined pattern variable '{target}'");
        }

        let declared_names: Vec<String> =
            planned_definitions.iter().map(|d| d.name.clone()).collect();
        Program::compile(&planned_pattern, &declared_names, self.config)?;

        let default_semantics = if planned_rows_per_match.is_one_row() {
            EvalSemantics::Final
        } else {
            EvalSemantics::Running
        };

        let mut planned_measures = Vec::with_capacity(measures.len());
        for m in measures {
            let measure_alias = ctx.normalize_ident(m.alias);
            let mut extractor = SlotExtractor {
                ctx,
                input_schema: input_schema.as_ref(),
                known_symbols: &known_symbols,
                bare_col_is_current_row: !planned_rows_per_match.is_one_row(),
                default_semantics,
                current_define_symbol: None,
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

fn lower_after_match_skip(skip: Option<SqlSkip>) -> AfterMatchSkip {
    match skip {
        None | Some(SqlSkip::PastLastRow) => AfterMatchSkip::PastLastRow,
        Some(SqlSkip::ToNextRow) => AfterMatchSkip::ToNextRow,
        Some(SqlSkip::ToFirst(id)) => AfterMatchSkip::ToFirst(normalize_pattern_ident(id)),
        Some(SqlSkip::ToLast(id)) => AfterMatchSkip::ToLast(normalize_pattern_ident(id)),
    }
}

fn lower_pattern(pattern: SqlPattern) -> Result<Pattern> {
    match pattern {
        SqlPattern::Symbol(sym) => Ok(Pattern::Symbol(lower_symbol(sym))),
        SqlPattern::Exclude(sym) => Ok(Pattern::Exclude(Box::new(Pattern::Symbol(lower_symbol(
            sym,
        ))))),
        SqlPattern::Permute(syms) => Ok(Pattern::Permute(
            syms.into_iter()
                .map(|s| Pattern::Symbol(lower_symbol(s)))
                .collect(),
        )),
        SqlPattern::Concat(items) => {
            let mut lowered = Vec::with_capacity(items.len());
            for item in items {
                lowered.push(lower_pattern(item)?);
            }
            Ok(Pattern::Concat(lowered))
        }
        SqlPattern::Group(inner) => lower_pattern(*inner),
        SqlPattern::Alternation(items) => {
            let mut lowered = Vec::with_capacity(items.len());
            for item in items {
                lowered.push(lower_pattern(item)?);
            }
            Ok(Pattern::Alternation(lowered))
        }
        SqlPattern::Repetition(inner, quant) => {
            let (min, max, reluctant) = lower_quantifier(quant);
            Ok(Pattern::Repetition {
                inner: Box::new(lower_pattern(*inner)?),
                min,
                max,
                reluctant,
            })
        }
    }
}

fn lower_symbol(sym: SqlSymbol) -> SymbolRef {
    match sym {
        SqlSymbol::Named(id) => SymbolRef::Named(normalize_pattern_ident(id)),
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

fn is_builtin_or_registered_agg(fname: &str, ctx: &dyn RelationPlannerContext) -> bool {
    matches!(fname, "count" | "sum" | "avg" | "min" | "max")
        || ctx.context_provider().get_aggregate_meta(fname).is_some()
}

fn is_stateful_define(
    expr: &SqlExpr,
    current_symbol: &str,
    known_symbols: &HashSet<String>,
    ctx: &dyn RelationPlannerContext,
) -> bool {
    match expr {
        SqlExpr::CompoundIdentifier(idents) => {
            if idents.len() == 2 {
                let prefix = normalize_pattern_ident(idents[0].clone());
                if prefix != current_symbol && known_symbols.contains(&prefix) {
                    return true;
                }
            }
            false
        }
        SqlExpr::Function(func) => {
            let fname = func.name.to_string().to_ascii_lowercase();
            if matches!(
                fname.as_str(),
                "prev" | "next" | "first" | "last" | "classifier" | "match_number"
            ) || is_builtin_or_registered_agg(&fname, ctx)
            {
                return true;
            }
            if let FunctionArguments::List(list) = &func.args {
                for arg in &list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg
                        && is_stateful_define(e, current_symbol, known_symbols, ctx)
                    {
                        return true;
                    }
                }
            }
            false
        }
        SqlExpr::BinaryOp { left, right, .. } => {
            is_stateful_define(left, current_symbol, known_symbols, ctx)
                || is_stateful_define(right, current_symbol, known_symbols, ctx)
        }
        SqlExpr::UnaryOp { expr, .. }
        | SqlExpr::Nested(expr)
        | SqlExpr::IsNull(expr)
        | SqlExpr::IsNotNull(expr)
        | SqlExpr::IsTrue(expr)
        | SqlExpr::IsFalse(expr)
        | SqlExpr::IsNotTrue(expr)
        | SqlExpr::IsNotFalse(expr)
        | SqlExpr::IsUnknown(expr)
        | SqlExpr::IsNotUnknown(expr)
        | SqlExpr::Cast { expr, .. }
        | SqlExpr::Extract { expr, .. }
        | SqlExpr::Ceil { expr, .. }
        | SqlExpr::Floor { expr, .. } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
        }
        SqlExpr::IsDistinctFrom(left, right) | SqlExpr::IsNotDistinctFrom(left, right) => {
            is_stateful_define(left, current_symbol, known_symbols, ctx)
                || is_stateful_define(right, current_symbol, known_symbols, ctx)
        }
        SqlExpr::Like { expr, pattern, .. }
        | SqlExpr::ILike { expr, pattern, .. }
        | SqlExpr::SimilarTo { expr, pattern, .. } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
                || is_stateful_define(pattern, current_symbol, known_symbols, ctx)
        }
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            ..
        } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
                || substring_from
                    .as_deref()
                    .is_some_and(|e| is_stateful_define(e, current_symbol, known_symbols, ctx))
                || substring_for
                    .as_deref()
                    .is_some_and(|e| is_stateful_define(e, current_symbol, known_symbols, ctx))
        }
        SqlExpr::Trim {
            expr, trim_what, ..
        } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
                || trim_what
                    .as_deref()
                    .is_some_and(|e| is_stateful_define(e, current_symbol, known_symbols, ctx))
        }
        SqlExpr::Position { expr, r#in } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
                || is_stateful_define(r#in, current_symbol, known_symbols, ctx)
        }
        SqlExpr::Array(arr) => arr
            .elem
            .iter()
            .any(|e| is_stateful_define(e, current_symbol, known_symbols, ctx)),
        SqlExpr::Tuple(elems) => elems
            .iter()
            .any(|e| is_stateful_define(e, current_symbol, known_symbols, ctx)),
        SqlExpr::Between {
            expr, low, high, ..
        } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
                || is_stateful_define(low, current_symbol, known_symbols, ctx)
                || is_stateful_define(high, current_symbol, known_symbols, ctx)
        }
        SqlExpr::InList { expr, list, .. } => {
            is_stateful_define(expr, current_symbol, known_symbols, ctx)
                || list
                    .iter()
                    .any(|e| is_stateful_define(e, current_symbol, known_symbols, ctx))
        }
        SqlExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            operand
                .as_deref()
                .is_some_and(|e| is_stateful_define(e, current_symbol, known_symbols, ctx))
                || conditions.iter().any(|c| {
                    is_stateful_define(&c.condition, current_symbol, known_symbols, ctx)
                        || is_stateful_define(&c.result, current_symbol, known_symbols, ctx)
                })
                || else_result
                    .as_deref()
                    .is_some_and(|e| is_stateful_define(e, current_symbol, known_symbols, ctx))
        }
        _ => false,
    }
}

fn strip_self_qualifier(expr: SqlExpr, current_symbol: &str) -> SqlExpr {
    match expr {
        SqlExpr::CompoundIdentifier(mut idents) => {
            if idents.len() == 2 && normalize_pattern_ident(idents[0].clone()) == current_symbol {
                return SqlExpr::Identifier(idents.pop().expect("len 2"));
            }
            SqlExpr::CompoundIdentifier(idents)
        }
        SqlExpr::Function(func) => {
            let args = match func.args {
                FunctionArguments::List(mut list) => {
                    for arg in &mut list.args {
                        if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                            let owned =
                                std::mem::replace(e, SqlExpr::Value(ast::Value::Null.into()));
                            *e = strip_self_qualifier(owned, current_symbol);
                        }
                    }
                    FunctionArguments::List(list)
                }
                other => other,
            };
            SqlExpr::Function(ast::Function { args, ..func })
        }
        SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
            left: Box::new(strip_self_qualifier(*left, current_symbol)),
            op,
            right: Box::new(strip_self_qualifier(*right, current_symbol)),
        },
        SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
            op,
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
        },
        SqlExpr::Nested(inner) => {
            SqlExpr::Nested(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsNull(inner) => {
            SqlExpr::IsNull(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsNotNull(inner) => {
            SqlExpr::IsNotNull(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsTrue(inner) => {
            SqlExpr::IsTrue(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsFalse(inner) => {
            SqlExpr::IsFalse(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsNotTrue(inner) => {
            SqlExpr::IsNotTrue(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsNotFalse(inner) => {
            SqlExpr::IsNotFalse(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsUnknown(inner) => {
            SqlExpr::IsUnknown(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsNotUnknown(inner) => {
            SqlExpr::IsNotUnknown(Box::new(strip_self_qualifier(*inner, current_symbol)))
        }
        SqlExpr::IsDistinctFrom(l, r) => SqlExpr::IsDistinctFrom(
            Box::new(strip_self_qualifier(*l, current_symbol)),
            Box::new(strip_self_qualifier(*r, current_symbol)),
        ),
        SqlExpr::IsNotDistinctFrom(l, r) => SqlExpr::IsNotDistinctFrom(
            Box::new(strip_self_qualifier(*l, current_symbol)),
            Box::new(strip_self_qualifier(*r, current_symbol)),
        ),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => SqlExpr::Like {
            negated,
            any,
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            pattern: Box::new(strip_self_qualifier(*pattern, current_symbol)),
            escape_char,
        },
        SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => SqlExpr::ILike {
            negated,
            any,
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            pattern: Box::new(strip_self_qualifier(*pattern, current_symbol)),
            escape_char,
        },
        SqlExpr::SimilarTo {
            negated,
            expr,
            pattern,
            escape_char,
        } => SqlExpr::SimilarTo {
            negated,
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            pattern: Box::new(strip_self_qualifier(*pattern, current_symbol)),
            escape_char,
        },
        SqlExpr::Substring {
            expr,
            substring_from,
            substring_for,
            special,
            shorthand,
        } => SqlExpr::Substring {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            substring_from: substring_from
                .map(|e| Box::new(strip_self_qualifier(*e, current_symbol))),
            substring_for: substring_for
                .map(|e| Box::new(strip_self_qualifier(*e, current_symbol))),
            special,
            shorthand,
        },
        SqlExpr::Trim {
            expr,
            trim_where,
            trim_what,
            trim_characters,
        } => SqlExpr::Trim {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            trim_where,
            trim_what: trim_what.map(|e| Box::new(strip_self_qualifier(*e, current_symbol))),
            trim_characters: trim_characters.map(|v| {
                v.into_iter()
                    .map(|e| strip_self_qualifier(e, current_symbol))
                    .collect()
            }),
        },
        SqlExpr::Position { expr, r#in } => SqlExpr::Position {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            r#in: Box::new(strip_self_qualifier(*r#in, current_symbol)),
        },
        SqlExpr::Extract {
            field,
            syntax,
            expr,
        } => SqlExpr::Extract {
            field,
            syntax,
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
        },
        SqlExpr::Ceil { expr, field } => SqlExpr::Ceil {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            field,
        },
        SqlExpr::Floor { expr, field } => SqlExpr::Floor {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            field,
        },
        SqlExpr::Array(ast::Array { elem, named }) => SqlExpr::Array(ast::Array {
            elem: elem
                .into_iter()
                .map(|e| strip_self_qualifier(e, current_symbol))
                .collect(),
            named,
        }),
        SqlExpr::Tuple(elems) => SqlExpr::Tuple(
            elems
                .into_iter()
                .map(|e| strip_self_qualifier(e, current_symbol))
                .collect(),
        ),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => SqlExpr::Between {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            negated,
            low: Box::new(strip_self_qualifier(*low, current_symbol)),
            high: Box::new(strip_self_qualifier(*high, current_symbol)),
        },
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => SqlExpr::InList {
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            list: list
                .into_iter()
                .map(|e| strip_self_qualifier(e, current_symbol))
                .collect(),
            negated,
        },
        SqlExpr::Case {
            case_token,
            end_token,
            operand,
            conditions,
            else_result,
        } => SqlExpr::Case {
            case_token,
            end_token,
            operand: operand.map(|e| Box::new(strip_self_qualifier(*e, current_symbol))),
            conditions: conditions
                .into_iter()
                .map(|c| ast::CaseWhen {
                    condition: strip_self_qualifier(c.condition, current_symbol),
                    result: strip_self_qualifier(c.result, current_symbol),
                })
                .collect(),
            else_result: else_result.map(|e| Box::new(strip_self_qualifier(*e, current_symbol))),
        },
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => SqlExpr::Cast {
            kind,
            expr: Box::new(strip_self_qualifier(*expr, current_symbol)),
            data_type,
            format,
        },
        other => other,
    }
}

struct SlotExtractor<'a, 'b> {
    ctx: &'a mut dyn RelationPlannerContext,
    input_schema: &'b DFSchema,
    known_symbols: &'b HashSet<String>,
    bare_col_is_current_row: bool,
    default_semantics: EvalSemantics,
    current_define_symbol: Option<&'b str>,
    slots: Vec<MeasureSlot>,
}

impl<'a, 'b> SlotExtractor<'a, 'b> {
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

    fn parse_classifier_symbol(&self, args: &FunctionArguments) -> Result<Option<String>> {
        let (arg_exprs, _) = extract_unnamed_args(args)?;
        match arg_exprs.len() {
            0 => Ok(None),
            1 => match &arg_exprs[0] {
                FunctionArgExpr::Expr(SqlExpr::Identifier(id)) => {
                    let sym = normalize_pattern_ident(id.clone());
                    if !self.known_symbols.contains(&sym) {
                        return plan_err!("CLASSIFIER({sym}) references unknown pattern variable");
                    }
                    Ok(Some(sym))
                }
                _ => plan_err!("CLASSIFIER() argument must be a pattern variable identifier"),
            },
            _ => plan_err!("CLASSIFIER() takes at most 1 argument"),
        }
    }

    fn rewrite_outer(&mut self, expr: SqlExpr) -> Result<SqlExpr> {
        match expr {
            SqlExpr::Function(func) => {
                let fname = func.name.to_string().to_ascii_lowercase();
                match fname.as_str() {
                    "match_number" => {
                        let (args, _) = extract_unnamed_args(&func.args)?;
                        if !args.is_empty() {
                            return plan_err!("MATCH_NUMBER() takes no arguments");
                        }
                        Ok(self.push_slot(MeasureSlotKind::MatchNumber))
                    }
                    "classifier" => {
                        let symbol = self.parse_classifier_symbol(&func.args)?;
                        Ok(self.push_slot(MeasureSlotKind::Classifier {
                            symbol,
                            logical_nav: None,
                            logical_offset: 0,
                            physical_nav: None,
                            physical_offset: 0,
                        }))
                    }
                    "first" | "last" | "prev" | "next" => {
                        let slot_kind = self.lower_nav_function(&fname, &func.args)?;
                        Ok(self.push_slot(slot_kind))
                    }
                    "count" | "sum" | "avg" | "min" | "max" => {
                        let slot_kind = self.lower_agg_function(&fname, None, &func.args)?;
                        Ok(self.push_slot(slot_kind))
                    }
                    _ => {
                        // Check if this is any registered DataFusion AggregateUDF (e.g.
                        // ARRAY_AGG, STRING_AGG, STDDEV_SAMP, BOOL_AND, custom UDAFs).
                        if let Some(udaf) = self.ctx.context_provider().get_aggregate_meta(&fname) {
                            let slot_kind =
                                self.lower_agg_function(&fname, Some(udaf), &func.args)?;
                            return Ok(self.push_slot(slot_kind));
                        }
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
                    let prefix = normalize_pattern_ident(idents[0].clone());
                    if self.known_symbols.contains(&prefix) {
                        let col_ident = idents[1].clone();
                        let inner_expr = self
                            .ctx
                            .sql_to_expr(SqlExpr::Identifier(col_ident), self.input_schema)?;
                        if self.current_define_symbol == Some(prefix.as_str()) {
                            return Ok(self.push_slot(MeasureSlotKind::CurrentRow { inner_expr }));
                        }
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
            SqlExpr::IsTrue(inner) => Ok(SqlExpr::IsTrue(Box::new(self.rewrite_outer(*inner)?))),
            SqlExpr::IsFalse(inner) => Ok(SqlExpr::IsFalse(Box::new(self.rewrite_outer(*inner)?))),
            SqlExpr::IsNotTrue(inner) => {
                Ok(SqlExpr::IsNotTrue(Box::new(self.rewrite_outer(*inner)?)))
            }
            SqlExpr::IsNotFalse(inner) => {
                Ok(SqlExpr::IsNotFalse(Box::new(self.rewrite_outer(*inner)?)))
            }
            SqlExpr::IsUnknown(inner) => {
                Ok(SqlExpr::IsUnknown(Box::new(self.rewrite_outer(*inner)?)))
            }
            SqlExpr::IsNotUnknown(inner) => {
                Ok(SqlExpr::IsNotUnknown(Box::new(self.rewrite_outer(*inner)?)))
            }
            SqlExpr::IsDistinctFrom(l, r) => Ok(SqlExpr::IsDistinctFrom(
                Box::new(self.rewrite_outer(*l)?),
                Box::new(self.rewrite_outer(*r)?),
            )),
            SqlExpr::IsNotDistinctFrom(l, r) => Ok(SqlExpr::IsNotDistinctFrom(
                Box::new(self.rewrite_outer(*l)?),
                Box::new(self.rewrite_outer(*r)?),
            )),
            SqlExpr::Like {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            } => Ok(SqlExpr::Like {
                negated,
                any,
                expr: Box::new(self.rewrite_outer(*expr)?),
                pattern: Box::new(self.rewrite_outer(*pattern)?),
                escape_char,
            }),
            SqlExpr::ILike {
                negated,
                any,
                expr,
                pattern,
                escape_char,
            } => Ok(SqlExpr::ILike {
                negated,
                any,
                expr: Box::new(self.rewrite_outer(*expr)?),
                pattern: Box::new(self.rewrite_outer(*pattern)?),
                escape_char,
            }),
            SqlExpr::SimilarTo {
                negated,
                expr,
                pattern,
                escape_char,
            } => Ok(SqlExpr::SimilarTo {
                negated,
                expr: Box::new(self.rewrite_outer(*expr)?),
                pattern: Box::new(self.rewrite_outer(*pattern)?),
                escape_char,
            }),
            SqlExpr::Substring {
                expr,
                substring_from,
                substring_for,
                special,
                shorthand,
            } => Ok(SqlExpr::Substring {
                expr: Box::new(self.rewrite_outer(*expr)?),
                substring_from: substring_from
                    .map(|e| self.rewrite_outer(*e))
                    .transpose()?
                    .map(Box::new),
                substring_for: substring_for
                    .map(|e| self.rewrite_outer(*e))
                    .transpose()?
                    .map(Box::new),
                special,
                shorthand,
            }),
            SqlExpr::Trim {
                expr,
                trim_where,
                trim_what,
                trim_characters,
            } => Ok(SqlExpr::Trim {
                expr: Box::new(self.rewrite_outer(*expr)?),
                trim_where,
                trim_what: trim_what
                    .map(|e| self.rewrite_outer(*e))
                    .transpose()?
                    .map(Box::new),
                trim_characters: trim_characters
                    .map(|v| {
                        v.into_iter()
                            .map(|e| self.rewrite_outer(e))
                            .collect::<Result<Vec<_>>>()
                    })
                    .transpose()?,
            }),
            SqlExpr::Position { expr, r#in } => Ok(SqlExpr::Position {
                expr: Box::new(self.rewrite_outer(*expr)?),
                r#in: Box::new(self.rewrite_outer(*r#in)?),
            }),
            SqlExpr::Extract {
                field,
                syntax,
                expr,
            } => Ok(SqlExpr::Extract {
                field,
                syntax,
                expr: Box::new(self.rewrite_outer(*expr)?),
            }),
            SqlExpr::Ceil { expr, field } => Ok(SqlExpr::Ceil {
                expr: Box::new(self.rewrite_outer(*expr)?),
                field,
            }),
            SqlExpr::Floor { expr, field } => Ok(SqlExpr::Floor {
                expr: Box::new(self.rewrite_outer(*expr)?),
                field,
            }),
            SqlExpr::Array(ast::Array { elem, named }) => Ok(SqlExpr::Array(ast::Array {
                elem: elem
                    .into_iter()
                    .map(|e| self.rewrite_outer(e))
                    .collect::<Result<Vec<_>>>()?,
                named,
            })),
            SqlExpr::Tuple(elems) => Ok(SqlExpr::Tuple(
                elems
                    .into_iter()
                    .map(|e| self.rewrite_outer(e))
                    .collect::<Result<Vec<_>>>()?,
            )),
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
            other => Ok(other),
        }
    }

    fn push_bare_column_slot(&mut self, inner_expr: datafusion_expr::Expr) -> SqlExpr {
        if self.bare_col_is_current_row {
            self.push_slot(MeasureSlotKind::CurrentRow { inner_expr })
        } else {
            self.push_slot(MeasureSlotKind::Nav {
                logical_nav: Some(LogicalNavKind::Last),
                logical_offset: 0,
                physical_nav: None,
                physical_offset: 0,
                symbol: None,
                inner_expr,
            })
        }
    }

    fn lower_nav_function(
        &mut self,
        fname: &str,
        args: &FunctionArguments,
    ) -> Result<MeasureSlotKind> {
        let (arg_exprs, distinct) = extract_unnamed_args(args)?;
        if distinct {
            return plan_err!("DISTINCT is not valid in {fname}()");
        }
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
                // Support FIRST(CLASSIFIER([sym]), n) / LAST(CLASSIFIER([sym]), n)
                if let SqlExpr::Function(inner_fn) = &target_sql
                    && inner_fn.name.to_string().eq_ignore_ascii_case("classifier")
                {
                    let symbol = self.parse_classifier_symbol(&inner_fn.args)?;
                    return Ok(MeasureSlotKind::Classifier {
                        symbol,
                        logical_nav: Some(logical_nav),
                        logical_offset: offset,
                        physical_nav: None,
                        physical_offset: 0,
                    });
                }
                let (symbol, stripped) =
                    extract_single_symbol_qualifier(target_sql, self.known_symbols)?;
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
                // Support PREV(CLASSIFIER([sym]), n) / NEXT(CLASSIFIER([sym]), n)
                if let SqlExpr::Function(inner_fn) = &target_sql
                    && inner_fn.name.to_string().eq_ignore_ascii_case("classifier")
                {
                    let symbol = self.parse_classifier_symbol(&inner_fn.args)?;
                    return Ok(MeasureSlotKind::Classifier {
                        symbol,
                        logical_nav: Some(LogicalNavKind::Last),
                        logical_offset: 0,
                        physical_nav: Some(physical_nav),
                        physical_offset: offset,
                    });
                }

                let (logical_nav, logical_offset, inner_target) =
                    if let SqlExpr::Function(inner_func) = &target_sql {
                        let inner_name = inner_func.name.to_string().to_ascii_lowercase();
                        if matches!(inner_name.as_str(), "first" | "last") {
                            let (inner_args, _) = extract_unnamed_args(&inner_func.args)?;
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

                // Support PREV(FIRST(CLASSIFIER()), n) / NEXT(LAST(CLASSIFIER()), n)
                if let SqlExpr::Function(inner_fn) = &inner_target
                    && inner_fn.name.to_string().eq_ignore_ascii_case("classifier")
                {
                    let symbol = self.parse_classifier_symbol(&inner_fn.args)?;
                    return Ok(MeasureSlotKind::Classifier {
                        symbol,
                        logical_nav,
                        logical_offset,
                        physical_nav: Some(physical_nav),
                        physical_offset: offset,
                    });
                }

                let (symbol, stripped) =
                    extract_single_symbol_qualifier(inner_target, self.known_symbols)?;
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
        udaf: Option<Arc<AggregateUDF>>,
        args: &FunctionArguments,
    ) -> Result<MeasureSlotKind> {
        let (arg_exprs, distinct) = extract_unnamed_args(args)?;
        if udaf.is_none() && fname == "count" && arg_exprs.is_empty() && !distinct {
            return Ok(MeasureSlotKind::Agg {
                agg_kind: AggKind::CountStar,
                symbol: None,
                args: vec![],
                distinct: false,
                arg_physical_nav: None,
                uses_match_metadata: false,
            });
        }
        if udaf.is_none() && arg_exprs.len() != 1 {
            return plan_err!(
                "{fname}() in MATCH_RECOGNIZE expects 1 argument, got {}",
                arg_exprs.len()
            );
        }
        if arg_exprs.is_empty() {
            return plan_err!("{fname}() in MATCH_RECOGNIZE expects at least 1 argument");
        }

        if udaf.is_none() {
            match &arg_exprs[0] {
                FunctionArgExpr::Wildcard => {
                    if fname != "count" || distinct {
                        return plan_err!("{fname}(*) is not valid in MATCH_RECOGNIZE");
                    }
                    return Ok(MeasureSlotKind::Agg {
                        agg_kind: AggKind::CountStar,
                        symbol: None,
                        args: vec![],
                        distinct: false,
                        arg_physical_nav: None,
                        uses_match_metadata: false,
                    });
                }
                FunctionArgExpr::QualifiedWildcard(obj_name) => {
                    if fname != "count" || distinct {
                        return plan_err!("{fname}({obj_name}.*) is not valid in MATCH_RECOGNIZE");
                    }
                    let ident = obj_name
                        .0
                        .last()
                        .and_then(|part| part.as_ident())
                        .ok_or_else(|| {
                            plan_datafusion_err!("Invalid qualified wildcard in COUNT")
                        })?;
                    let sym = normalize_pattern_ident(ident.clone());
                    if !self.known_symbols.contains(&sym) {
                        return plan_err!(
                            "COUNT({sym}.*) references unknown pattern variable '{sym}'"
                        );
                    }
                    return Ok(MeasureSlotKind::Agg {
                        agg_kind: AggKind::CountSymbol,
                        symbol: Some(sym),
                        args: vec![],
                        distinct: false,
                        arg_physical_nav: None,
                        uses_match_metadata: false,
                    });
                }
                _ => {}
            }
        }

        let agg_kind = match udaf {
            Some(u) => AggKind::Udaf(u),
            None => match fname {
                "count" => AggKind::Count,
                "sum" => AggKind::Sum,
                "avg" => AggKind::Avg,
                "min" => AggKind::Min,
                "max" => AggKind::Max,
                _ => unreachable!(),
            },
        };

        let mut shared_symbol: Option<String> = None;
        let mut arg_physical_nav: Option<(PhysicalNavKind, u32)> = None;
        let mut uses_match_metadata = false;
        let mut rewritten_sql_args = Vec::with_capacity(arg_exprs.len());

        for (i, arg) in arg_exprs.into_iter().enumerate() {
            let sql_expr = match arg {
                FunctionArgExpr::Expr(e) => e,
                _ => return plan_err!("Wildcard arguments are not supported in {fname}()"),
            };

            let (phys_nav, unwrapped_sql) = unwrap_arg_physical_nav(sql_expr)?;
            if let Some(pn) = phys_nav {
                if i == 0 {
                    arg_physical_nav = Some(pn);
                } else if arg_physical_nav != Some(pn) {
                    return plan_err!(
                        "Mismatched PREV/NEXT navigation across multiple arguments of {fname}()"
                    );
                }
            }

            let (sym, stripped) =
                extract_single_symbol_qualifier(unwrapped_sql, self.known_symbols)?;
            if let Some(s) = sym {
                if let Some(existing) = &shared_symbol
                    && existing != &s
                {
                    return plan_err!(
                        "Aggregate {fname}() cannot reference multiple pattern variables ('{existing}' and '{s}')"
                    );
                }
                shared_symbol = Some(s);
            }
            let (replaced, has_meta, meta_sym) =
                rewrite_match_metadata_in_agg_arg(stripped, self.known_symbols)?;
            if let Some(s) = meta_sym {
                if let Some(existing) = &shared_symbol
                    && existing != &s
                {
                    return plan_err!(
                        "Aggregate {fname}() cannot reference multiple pattern variables ('{existing}' and '{s}')"
                    );
                }
                shared_symbol = Some(s);
            }
            if has_meta {
                uses_match_metadata = true;
            }
            rewritten_sql_args.push(replaced);
        }

        let meta_schema_owned;
        let eval_schema: &DFSchema = if uses_match_metadata {
            meta_schema_owned = build_agg_metadata_dfschema(self.input_schema)?;
            &meta_schema_owned
        } else {
            self.input_schema
        };

        let planned_args = rewritten_sql_args
            .into_iter()
            .map(|e| self.ctx.sql_to_expr(e, eval_schema))
            .collect::<Result<Vec<_>>>()?;

        Ok(MeasureSlotKind::Agg {
            agg_kind,
            symbol: shared_symbol,
            args: planned_args,
            distinct,
            arg_physical_nav,
            uses_match_metadata,
        })
    }
}

fn unwrap_arg_physical_nav(expr: SqlExpr) -> Result<(Option<(PhysicalNavKind, u32)>, SqlExpr)> {
    if let SqlExpr::Function(ref func) = expr {
        let name = func.name.to_string().to_ascii_lowercase();
        if matches!(name.as_str(), "prev" | "next") {
            let (args, _) = extract_unnamed_args(&func.args)?;
            if args.is_empty() || args.len() > 2 {
                return plan_err!("{name}() expects 1 or 2 arguments");
            }
            let inner = match &args[0] {
                FunctionArgExpr::Expr(e) => e.clone(),
                _ => return plan_err!("{name}() expects an expression argument"),
            };
            let offset = if args.len() == 2 {
                match &args[1] {
                    FunctionArgExpr::Expr(e) => parse_non_negative_u32(e)?,
                    _ => return plan_err!("{name}() offset must be an integer"),
                }
            } else {
                1
            };
            let kind = if name == "prev" {
                PhysicalNavKind::Prev
            } else {
                PhysicalNavKind::Next
            };
            return Ok((Some((kind, offset)), inner));
        }
    }
    Ok((None, expr))
}

fn extract_unnamed_args(args: &FunctionArguments) -> Result<(Vec<FunctionArgExpr>, bool)> {
    match args {
        FunctionArguments::None => Ok((vec![], false)),
        FunctionArguments::List(list) => {
            let distinct = matches!(list.duplicate_treatment, Some(DuplicateTreatment::Distinct));
            let mut out = Vec::with_capacity(list.args.len());
            for arg in &list.args {
                match arg {
                    FunctionArg::Unnamed(expr) => out.push(expr.clone()),
                    _ => {
                        return plan_err!(
                            "Named arguments are not supported in MATCH_RECOGNIZE functions"
                        );
                    }
                }
            }
            Ok((out, distinct))
        }
        FunctionArguments::Subquery(_) => {
            plan_err!("Subquery arguments are not supported in MATCH_RECOGNIZE functions")
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

/// Rewrites `CLASSIFIER([sym])` and `MATCH_NUMBER()` inside an aggregate argument expression to
/// synthetic columns `__mr_agg_classifier` and `__mr_agg_match_number`, returning any pattern
/// variable symbol referenced by `CLASSIFIER(sym)`.
fn rewrite_match_metadata_in_agg_arg(
    expr: SqlExpr,
    known_symbols: &HashSet<String>,
) -> Result<(SqlExpr, bool, Option<String>)> {
    let mut found = false;
    let mut classifier_sym: Option<String> = None;
    let rewritten = rewrite_meta_inner(expr, known_symbols, &mut found, &mut classifier_sym)?;
    Ok((rewritten, found, classifier_sym))
}

fn rewrite_meta_inner(
    expr: SqlExpr,
    known_symbols: &HashSet<String>,
    found: &mut bool,
    classifier_sym: &mut Option<String>,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::Function(func) => {
            let fname = func.name.to_string().to_ascii_lowercase();
            if fname == "classifier" {
                let (args, _) = extract_unnamed_args(&func.args)?;
                match args.len() {
                    0 => {}
                    1 => match &args[0] {
                        FunctionArgExpr::Expr(SqlExpr::Identifier(id)) => {
                            let sym = normalize_pattern_ident(id.clone());
                            if !known_symbols.contains(&sym) {
                                return plan_err!(
                                    "CLASSIFIER({sym}) references unknown pattern variable"
                                );
                            }
                            if let Some(existing) = classifier_sym.as_ref()
                                && existing != &sym
                            {
                                return plan_err!(
                                    "Aggregate argument cannot reference multiple pattern variables ('{existing}' and '{sym}')"
                                );
                            }
                            *classifier_sym = Some(sym);
                        }
                        _ => {
                            return plan_err!(
                                "CLASSIFIER() argument must be a pattern variable identifier"
                            );
                        }
                    },
                    _ => return plan_err!("CLASSIFIER() takes at most 1 argument"),
                }
                *found = true;
                return Ok(SqlExpr::Identifier(Ident::new(AGG_META_CLASSIFIER_COL)));
            }
            if fname == "match_number" {
                let (args, _) = extract_unnamed_args(&func.args)?;
                if !args.is_empty() {
                    return plan_err!("MATCH_NUMBER() takes no arguments");
                }
                *found = true;
                return Ok(SqlExpr::Identifier(Ident::new(AGG_META_MATCH_NUMBER_COL)));
            }
            let args = match func.args {
                FunctionArguments::List(mut list) => {
                    for arg in &mut list.args {
                        if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                            let owned =
                                std::mem::replace(e, SqlExpr::Value(ast::Value::Null.into()));
                            *e = rewrite_meta_inner(owned, known_symbols, found, classifier_sym)?;
                        }
                    }
                    FunctionArguments::List(list)
                }
                other => other,
            };
            Ok(SqlExpr::Function(ast::Function { args, ..func }))
        }
        SqlExpr::BinaryOp { left, op, right } => Ok(SqlExpr::BinaryOp {
            left: Box::new(rewrite_meta_inner(
                *left,
                known_symbols,
                found,
                classifier_sym,
            )?),
            op,
            right: Box::new(rewrite_meta_inner(
                *right,
                known_symbols,
                found,
                classifier_sym,
            )?),
        }),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op,
            expr: Box::new(rewrite_meta_inner(
                *expr,
                known_symbols,
                found,
                classifier_sym,
            )?),
        }),
        SqlExpr::Nested(inner) => Ok(SqlExpr::Nested(Box::new(rewrite_meta_inner(
            *inner,
            known_symbols,
            found,
            classifier_sym,
        )?))),
        SqlExpr::IsNull(inner) => Ok(SqlExpr::IsNull(Box::new(rewrite_meta_inner(
            *inner,
            known_symbols,
            found,
            classifier_sym,
        )?))),
        SqlExpr::IsNotNull(inner) => Ok(SqlExpr::IsNotNull(Box::new(rewrite_meta_inner(
            *inner,
            known_symbols,
            found,
            classifier_sym,
        )?))),
        SqlExpr::IsDistinctFrom(l, r) => Ok(SqlExpr::IsDistinctFrom(
            Box::new(rewrite_meta_inner(
                *l,
                known_symbols,
                found,
                classifier_sym,
            )?),
            Box::new(rewrite_meta_inner(
                *r,
                known_symbols,
                found,
                classifier_sym,
            )?),
        )),
        SqlExpr::IsNotDistinctFrom(l, r) => Ok(SqlExpr::IsNotDistinctFrom(
            Box::new(rewrite_meta_inner(
                *l,
                known_symbols,
                found,
                classifier_sym,
            )?),
            Box::new(rewrite_meta_inner(
                *r,
                known_symbols,
                found,
                classifier_sym,
            )?),
        )),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::Like {
            negated,
            any,
            expr: Box::new(rewrite_meta_inner(
                *expr,
                known_symbols,
                found,
                classifier_sym,
            )?),
            pattern: Box::new(rewrite_meta_inner(
                *pattern,
                known_symbols,
                found,
                classifier_sym,
            )?),
            escape_char,
        }),
        SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::ILike {
            negated,
            any,
            expr: Box::new(rewrite_meta_inner(
                *expr,
                known_symbols,
                found,
                classifier_sym,
            )?),
            pattern: Box::new(rewrite_meta_inner(
                *pattern,
                known_symbols,
                found,
                classifier_sym,
            )?),
            escape_char,
        }),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(SqlExpr::Between {
            expr: Box::new(rewrite_meta_inner(
                *expr,
                known_symbols,
                found,
                classifier_sym,
            )?),
            negated,
            low: Box::new(rewrite_meta_inner(
                *low,
                known_symbols,
                found,
                classifier_sym,
            )?),
            high: Box::new(rewrite_meta_inner(
                *high,
                known_symbols,
                found,
                classifier_sym,
            )?),
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => Ok(SqlExpr::InList {
            expr: Box::new(rewrite_meta_inner(
                *expr,
                known_symbols,
                found,
                classifier_sym,
            )?),
            list: list
                .into_iter()
                .map(|e| rewrite_meta_inner(e, known_symbols, found, classifier_sym))
                .collect::<Result<Vec<_>>>()?,
            negated,
        }),
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => Ok(SqlExpr::Cast {
            kind,
            expr: Box::new(rewrite_meta_inner(
                *expr,
                known_symbols,
                found,
                classifier_sym,
            )?),
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
                .map(|e| rewrite_meta_inner(*e, known_symbols, found, classifier_sym))
                .transpose()?
                .map(Box::new);
            let mut new_conditions = Vec::with_capacity(conditions.len());
            for c in conditions {
                new_conditions.push(ast::CaseWhen {
                    condition: rewrite_meta_inner(
                        c.condition,
                        known_symbols,
                        found,
                        classifier_sym,
                    )?,
                    result: rewrite_meta_inner(c.result, known_symbols, found, classifier_sym)?,
                });
            }
            let else_result = else_result
                .map(|e| rewrite_meta_inner(*e, known_symbols, found, classifier_sym))
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

fn extract_single_symbol_qualifier(
    expr: SqlExpr,
    known_symbols: &HashSet<String>,
) -> Result<(Option<String>, SqlExpr)> {
    let mut found: Option<String> = None;
    let stripped = strip_symbol_inner(expr, known_symbols, &mut found)?;
    Ok((found, stripped))
}

fn strip_symbol_inner(
    expr: SqlExpr,
    known_symbols: &HashSet<String>,
    found: &mut Option<String>,
) -> Result<SqlExpr> {
    match expr {
        SqlExpr::CompoundIdentifier(mut idents) => {
            if idents.len() == 2 {
                let prefix = normalize_pattern_ident(idents[0].clone());
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
            left: Box::new(strip_symbol_inner(*left, known_symbols, found)?),
            op: op_clone(op),
            right: Box::new(strip_symbol_inner(*right, known_symbols, found)?),
        }),
        SqlExpr::UnaryOp { op, expr } => Ok(SqlExpr::UnaryOp {
            op,
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, found)?),
        }),
        SqlExpr::Nested(inner) => Ok(SqlExpr::Nested(Box::new(strip_symbol_inner(
            *inner,
            known_symbols,
            found,
        )?))),
        SqlExpr::IsNull(inner) => Ok(SqlExpr::IsNull(Box::new(strip_symbol_inner(
            *inner,
            known_symbols,
            found,
        )?))),
        SqlExpr::IsNotNull(inner) => Ok(SqlExpr::IsNotNull(Box::new(strip_symbol_inner(
            *inner,
            known_symbols,
            found,
        )?))),
        SqlExpr::IsDistinctFrom(l, r) => Ok(SqlExpr::IsDistinctFrom(
            Box::new(strip_symbol_inner(*l, known_symbols, found)?),
            Box::new(strip_symbol_inner(*r, known_symbols, found)?),
        )),
        SqlExpr::IsNotDistinctFrom(l, r) => Ok(SqlExpr::IsNotDistinctFrom(
            Box::new(strip_symbol_inner(*l, known_symbols, found)?),
            Box::new(strip_symbol_inner(*r, known_symbols, found)?),
        )),
        SqlExpr::Like {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::Like {
            negated,
            any,
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, found)?),
            pattern: Box::new(strip_symbol_inner(*pattern, known_symbols, found)?),
            escape_char,
        }),
        SqlExpr::ILike {
            negated,
            any,
            expr,
            pattern,
            escape_char,
        } => Ok(SqlExpr::ILike {
            negated,
            any,
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, found)?),
            pattern: Box::new(strip_symbol_inner(*pattern, known_symbols, found)?),
            escape_char,
        }),
        SqlExpr::Between {
            expr,
            negated,
            low,
            high,
        } => Ok(SqlExpr::Between {
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, found)?),
            negated,
            low: Box::new(strip_symbol_inner(*low, known_symbols, found)?),
            high: Box::new(strip_symbol_inner(*high, known_symbols, found)?),
        }),
        SqlExpr::InList {
            expr,
            list,
            negated,
        } => Ok(SqlExpr::InList {
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, found)?),
            list: list
                .into_iter()
                .map(|e| strip_symbol_inner(e, known_symbols, found))
                .collect::<Result<Vec<_>>>()?,
            negated,
        }),
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => Ok(SqlExpr::Cast {
            kind,
            expr: Box::new(strip_symbol_inner(*expr, known_symbols, found)?),
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
                            *e = strip_symbol_inner(owned, known_symbols, found)?;
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
                .map(|e| strip_symbol_inner(*e, known_symbols, found))
                .transpose()?
                .map(Box::new);
            let mut new_conditions = Vec::with_capacity(conditions.len());
            for c in conditions {
                new_conditions.push(ast::CaseWhen {
                    condition: strip_symbol_inner(c.condition, known_symbols, found)?,
                    result: strip_symbol_inner(c.result, known_symbols, found)?,
                });
            }
            let else_result = else_result
                .map(|e| strip_symbol_inner(*e, known_symbols, found))
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
