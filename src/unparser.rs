// SPDX-License-Identifier: Apache-2.0

//! [`UserDefinedLogicalNodeUnparser`] implementation converting [`MatchRecognizeNode`] back into
//! SQL (`TableFactor::MatchRecognize`) for plan-to-SQL round-tripping and query federation.

use std::collections::HashMap;

use datafusion_common::{Result, plan_err};
use datafusion_expr::{Expr, UserDefinedLogicalNode};
use datafusion_sql::unparser::Unparser;
use datafusion_sql::unparser::extension_unparser::{
    UnparseToStatementResult, UserDefinedLogicalNodeUnparser,
};
use sqlparser::ast::{
    self, AfterMatchSkip as SqlSkip, DuplicateTreatment, EmptyMatchesMode as SqlEmptyMode,
    Expr as SqlExpr, Function, FunctionArg, FunctionArgExpr, FunctionArgumentList,
    FunctionArguments, Ident, MatchRecognizePattern as SqlPattern,
    MatchRecognizeSymbol as SqlSymbol, Measure as SqlMeasure, ObjectName, ObjectNamePart,
    OrderByExpr, OrderByOptions, OrderBySort, Query, RepetitionQuantifier as SqlQuant,
    RowsPerMatch as SqlRowsPerMatch, Select, SelectItem, SetExpr, SymbolDefinition as SqlSymDef,
    TableFactor, TableWithJoins,
};

use crate::ast::{
    AfterMatchSkip, AggKind, LogicalNavKind, MeasureSlot, MeasureSlotKind, Pattern,
    PhysicalNavKind, RowsPerMatch, SymbolDefKind, SymbolRef,
};
use crate::logical::MatchRecognizeNode;

/// Unparses [`MatchRecognizeNode`] into SQL `SELECT * FROM <input> MATCH_RECOGNIZE(...)`.
///
/// Implements [`UserDefinedLogicalNodeUnparser::unparse_to_statement`] and also provides
/// [`match_recognize_to_table_factor`] for callers embedding `TableFactor::MatchRecognize`
/// inside custom AST builders.
#[derive(Debug, Default)]
pub struct MatchRecognizeUnparser;

impl UserDefinedLogicalNodeUnparser for MatchRecognizeUnparser {
    fn unparse_to_statement(
        &self,
        node: &dyn UserDefinedLogicalNode,
        unparser: &Unparser,
    ) -> Result<UnparseToStatementResult> {
        let Some(mr) = node.as_any().downcast_ref::<MatchRecognizeNode>() else {
            return Ok(UnparseToStatementResult::Unmodified);
        };
        let table_factor = match_recognize_to_table_factor(mr, unparser)?;
        let select = Select {
            select_token: ast::helpers::attached_token::AttachedToken::empty(),
            optimizer_hints: vec![],
            distinct: None,
            select_modifiers: None,
            top: None,
            top_before_distinct: false,
            projection: vec![SelectItem::Wildcard(
                ast::WildcardAdditionalOptions::default(),
            )],
            exclude: None,
            into: None,
            from: vec![TableWithJoins {
                relation: table_factor,
                joins: vec![],
            }],
            lateral_views: vec![],
            prewhere: None,
            selection: None,
            group_by: ast::GroupByExpr::Expressions(vec![], vec![]),
            cluster_by: vec![],
            distribute_by: vec![],
            sort_by: vec![],
            having: None,
            named_window: vec![],
            qualify: None,
            window_before_qualify: false,
            value_table_mode: None,
            connect_by: vec![],
            flavor: ast::SelectFlavor::Standard,
        };
        let query = Query {
            with: None,
            body: Box::new(SetExpr::Select(Box::new(select))),
            order_by: None,
            limit_clause: None,
            fetch: None,
            locks: vec![],
            for_clause: None,
            settings: None,
            format_clause: None,
            pipe_operators: vec![],
        };
        Ok(UnparseToStatementResult::Modified(ast::Statement::Query(
            Box::new(query),
        )))
    }
}

/// Converts a [`MatchRecognizeNode`] directly into a [`TableFactor::MatchRecognize`].
pub fn match_recognize_to_table_factor(
    mr: &MatchRecognizeNode,
    unparser: &Unparser,
) -> Result<TableFactor> {
    let input_table = unparse_input_table_factor(unparser, &mr.input)?;

    let partition_by = mr
        .partition_by
        .iter()
        .map(|e| unparser.expr_to_sql(e))
        .collect::<Result<Vec<_>>>()?;

    let order_by = mr
        .order_by
        .iter()
        .map(|o| {
            Ok(OrderByExpr {
                expr: unparser.expr_to_sql(&o.expr)?,
                options: OrderByOptions {
                    sort: Some(if o.asc {
                        OrderBySort::Asc
                    } else {
                        OrderBySort::Desc
                    }),
                    nulls_first: Some(o.nulls_first),
                },
                with_fill: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut measures = Vec::with_capacity(mr.measures.len());
    for m in &mr.measures {
        let expr = unparse_slotted_expr(unparser, &m.outer_expr, &m.slots)?;
        measures.push(SqlMeasure {
            expr,
            alias: Ident::new(&m.alias),
        });
    }

    let rows_per_match = Some(match mr.rows_per_match {
        RowsPerMatch::OneRow => SqlRowsPerMatch::OneRow,
        RowsPerMatch::AllRowsShowEmpty => SqlRowsPerMatch::AllRows(Some(SqlEmptyMode::Show)),
        RowsPerMatch::AllRowsOmitEmpty => SqlRowsPerMatch::AllRows(Some(SqlEmptyMode::Omit)),
        RowsPerMatch::AllRowsWithUnmatched => {
            SqlRowsPerMatch::AllRows(Some(SqlEmptyMode::WithUnmatched))
        }
    });

    let after_match_skip = Some(match &mr.after_match_skip {
        AfterMatchSkip::PastLastRow => SqlSkip::PastLastRow,
        AfterMatchSkip::ToNextRow => SqlSkip::ToNextRow,
        AfterMatchSkip::ToFirst(s) => SqlSkip::ToFirst(Ident::new(s)),
        AfterMatchSkip::ToLast(s) => SqlSkip::ToLast(Ident::new(s)),
    });

    let pattern = unparse_pattern(&mr.pattern)?;

    let mut symbols = Vec::with_capacity(mr.definitions.len());
    for def in &mr.definitions {
        let definition = match &def.kind {
            SymbolDefKind::Stateless { predicate } => unparser.expr_to_sql(predicate)?,
            SymbolDefKind::Stateful { outer_expr, slots } => {
                unparse_slotted_expr(unparser, outer_expr, slots)?
            }
        };
        symbols.push(SqlSymDef {
            symbol: Ident::new(&def.name),
            definition,
        });
    }

    Ok(TableFactor::MatchRecognize {
        table: Box::new(input_table),
        partition_by,
        order_by,
        measures,
        rows_per_match,
        after_match_skip,
        pattern,
        symbols,
        alias: None,
    })
}

fn unparse_input_table_factor(
    unparser: &Unparser,
    input: &datafusion_expr::LogicalPlan,
) -> Result<TableFactor> {
    match input {
        datafusion_expr::LogicalPlan::TableScan(scan) => Ok(TableFactor::Table {
            name: ObjectName(
                scan.table_name
                    .to_string()
                    .split('.')
                    .map(|p| ObjectNamePart::Identifier(Ident::new(p)))
                    .collect(),
            ),
            alias: None,
            args: None,
            with_hints: vec![],
            version: None,
            with_ordinality: false,
            partitions: vec![],
            json_path: None,
            sample: None,
            index_hints: vec![],
        }),
        other => {
            let stmt = unparser.plan_to_sql(other)?;
            match stmt {
                ast::Statement::Query(q) => Ok(TableFactor::Derived {
                    lateral: false,
                    subquery: q,
                    alias: None,
                    sample: None,
                }),
                _ => plan_err!("Cannot unparse non-query input to MatchRecognize"),
            }
        }
    }
}

fn unparse_slotted_expr(
    unparser: &Unparser,
    outer_expr: &Expr,
    slots: &[MeasureSlot],
) -> Result<SqlExpr> {
    let mut slot_map: HashMap<String, SqlExpr> = HashMap::with_capacity(slots.len());
    for slot in slots {
        slot_map.insert(slot.slot_name.clone(), unparse_slot(unparser, slot)?);
    }
    let outer_sql = unparser.expr_to_sql(outer_expr)?;
    Ok(substitute_slots(outer_sql, &slot_map))
}

fn unparse_slot(unparser: &Unparser, slot: &MeasureSlot) -> Result<SqlExpr> {
    match &slot.kind {
        MeasureSlotKind::MatchNumber => Ok(make_func("MATCH_NUMBER", vec![], false)),
        MeasureSlotKind::Classifier {
            symbol,
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
        } => {
            let args = symbol
                .as_ref()
                .map(|s| {
                    vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(
                        SqlExpr::Identifier(Ident::new(s)),
                    ))]
                })
                .unwrap_or_default();
            let mut expr = make_func("CLASSIFIER", args, false);
            if let Some(ln) = logical_nav {
                expr = wrap_logical_nav(expr, *ln, *logical_offset);
            }
            if let Some(pn) = physical_nav {
                expr = wrap_physical_nav(expr, *pn, *physical_offset);
            }
            Ok(expr)
        }
        MeasureSlotKind::CurrentRow { inner_expr } => unparser.expr_to_sql(inner_expr),
        MeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            symbol,
            inner_expr,
        } => {
            let mut expr = unparser.expr_to_sql(inner_expr)?;
            if let Some(sym) = symbol {
                expr = qualify_identifiers(expr, sym);
            }
            if let Some(ln) = logical_nav {
                expr = wrap_logical_nav(expr, *ln, *logical_offset);
            }
            if let Some(pn) = physical_nav {
                expr = wrap_physical_nav(expr, *pn, *physical_offset);
            }
            Ok(expr)
        }
        MeasureSlotKind::Agg {
            agg_kind,
            symbol,
            args,
            distinct,
            arg_physical_nav,
            uses_match_metadata,
        } => match agg_kind {
            AggKind::CountStar => Ok(make_func(
                "COUNT",
                vec![FunctionArg::Unnamed(FunctionArgExpr::Wildcard)],
                false,
            )),
            AggKind::CountSymbol => {
                let sym = symbol.as_deref().unwrap_or("");
                Ok(make_func(
                    "COUNT",
                    vec![FunctionArg::Unnamed(FunctionArgExpr::QualifiedWildcard(
                        ObjectName(vec![ObjectNamePart::Identifier(Ident::new(sym))]),
                    ))],
                    false,
                ))
            }
            _ => {
                let fname = match agg_kind {
                    AggKind::Count => "COUNT",
                    AggKind::Sum => "SUM",
                    AggKind::Avg => "AVG",
                    AggKind::Min => "MIN",
                    AggKind::Max => "MAX",
                    AggKind::Udaf(u) => u.name(),
                    AggKind::CountStar | AggKind::CountSymbol => unreachable!(),
                };
                let mut meta_map = HashMap::new();
                if *uses_match_metadata {
                    meta_map.insert(
                        crate::logical::AGG_META_CLASSIFIER_COL.to_string(),
                        make_func("CLASSIFIER", vec![], false),
                    );
                    meta_map.insert(
                        crate::logical::AGG_META_MATCH_NUMBER_COL.to_string(),
                        make_func("MATCH_NUMBER", vec![], false),
                    );
                }
                let mut fn_args = Vec::with_capacity(args.len());
                for a in args {
                    let mut sql_a = unparser.expr_to_sql(a)?;
                    if *uses_match_metadata {
                        sql_a = substitute_slots(sql_a, &meta_map);
                    }
                    if let Some(sym) = symbol {
                        sql_a = qualify_identifiers(sql_a, sym);
                    }
                    if let Some((pn, off)) = arg_physical_nav {
                        sql_a = wrap_physical_nav(sql_a, *pn, *off);
                    }
                    fn_args.push(FunctionArg::Unnamed(FunctionArgExpr::Expr(sql_a)));
                }
                Ok(make_func(fname, fn_args, *distinct))
            }
        },
    }
}

fn wrap_logical_nav(inner: SqlExpr, nav: LogicalNavKind, offset: u32) -> SqlExpr {
    let name = match nav {
        LogicalNavKind::First => "FIRST",
        LogicalNavKind::Last => "LAST",
    };
    let mut args = vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(inner))];
    if offset > 0 {
        args.push(FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(
            ast::Value::Number(offset.to_string(), false).into(),
        ))));
    }
    make_func(name, args, false)
}

fn wrap_physical_nav(inner: SqlExpr, nav: PhysicalNavKind, offset: u32) -> SqlExpr {
    let name = match nav {
        PhysicalNavKind::Prev => "PREV",
        PhysicalNavKind::Next => "NEXT",
    };
    let mut args = vec![FunctionArg::Unnamed(FunctionArgExpr::Expr(inner))];
    if offset != 1 {
        args.push(FunctionArg::Unnamed(FunctionArgExpr::Expr(SqlExpr::Value(
            ast::Value::Number(offset.to_string(), false).into(),
        ))));
    }
    make_func(name, args, false)
}

fn make_func(name: &str, args: Vec<FunctionArg>, distinct: bool) -> SqlExpr {
    SqlExpr::Function(Function {
        name: ObjectName(vec![ObjectNamePart::Identifier(Ident::new(name))]),
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(FunctionArgumentList {
            duplicate_treatment: if distinct {
                Some(DuplicateTreatment::Distinct)
            } else {
                None
            },
            args,
            clauses: vec![],
        }),
        filter: None,
        null_treatment: None,
        over: None,
        within_group: vec![],
    })
}

fn qualify_identifiers(expr: SqlExpr, sym: &str) -> SqlExpr {
    match expr {
        SqlExpr::Identifier(id) => SqlExpr::CompoundIdentifier(vec![Ident::new(sym), id]),
        SqlExpr::CompoundIdentifier(mut ids) => {
            let col = ids.pop().expect("non-empty compound identifier");
            SqlExpr::CompoundIdentifier(vec![Ident::new(sym), col])
        }
        SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
            left: Box::new(qualify_identifiers(*left, sym)),
            op,
            right: Box::new(qualify_identifiers(*right, sym)),
        },
        SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
            op,
            expr: Box::new(qualify_identifiers(*expr, sym)),
        },
        SqlExpr::Nested(inner) => SqlExpr::Nested(Box::new(qualify_identifiers(*inner, sym))),
        other => other,
    }
}

fn substitute_slots(expr: SqlExpr, map: &HashMap<String, SqlExpr>) -> SqlExpr {
    match expr {
        SqlExpr::Identifier(ref id) if map.contains_key(&id.value) => map[&id.value].clone(),
        SqlExpr::CompoundIdentifier(ref ids)
            if ids.len() == 1 && map.contains_key(&ids[0].value) =>
        {
            map[&ids[0].value].clone()
        }
        SqlExpr::BinaryOp { left, op, right } => SqlExpr::BinaryOp {
            left: Box::new(substitute_slots(*left, map)),
            op,
            right: Box::new(substitute_slots(*right, map)),
        },
        SqlExpr::UnaryOp { op, expr } => SqlExpr::UnaryOp {
            op,
            expr: Box::new(substitute_slots(*expr, map)),
        },
        SqlExpr::Nested(inner) => SqlExpr::Nested(Box::new(substitute_slots(*inner, map))),
        SqlExpr::IsNull(inner) => SqlExpr::IsNull(Box::new(substitute_slots(*inner, map))),
        SqlExpr::IsNotNull(inner) => SqlExpr::IsNotNull(Box::new(substitute_slots(*inner, map))),
        SqlExpr::Cast {
            kind,
            expr,
            data_type,
            format,
        } => SqlExpr::Cast {
            kind,
            expr: Box::new(substitute_slots(*expr, map)),
            data_type,
            format,
        },
        SqlExpr::Function(func) => {
            let args = match func.args {
                FunctionArguments::List(mut list) => {
                    for arg in &mut list.args {
                        if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e)) = arg {
                            let owned =
                                std::mem::replace(e, SqlExpr::Value(ast::Value::Null.into()));
                            *e = substitute_slots(owned, map);
                        }
                    }
                    FunctionArguments::List(list)
                }
                other => other,
            };
            SqlExpr::Function(Function { args, ..func })
        }
        SqlExpr::Case {
            case_token,
            end_token,
            operand,
            conditions,
            else_result,
        } => SqlExpr::Case {
            case_token,
            end_token,
            operand: operand.map(|e| Box::new(substitute_slots(*e, map))),
            conditions: conditions
                .into_iter()
                .map(|c| ast::CaseWhen {
                    condition: substitute_slots(c.condition, map),
                    result: substitute_slots(c.result, map),
                })
                .collect(),
            else_result: else_result.map(|e| Box::new(substitute_slots(*e, map))),
        },
        other => other,
    }
}

fn unparse_pattern(p: &Pattern) -> Result<SqlPattern> {
    match p {
        Pattern::Symbol(sym) => Ok(SqlPattern::Symbol(match sym {
            SymbolRef::Named(s) => SqlSymbol::Named(Ident::new(s)),
            SymbolRef::Start => SqlSymbol::Start,
            SymbolRef::End => SqlSymbol::End,
        })),
        Pattern::Exclude(inner) => match inner.as_ref() {
            Pattern::Symbol(sym) => Ok(SqlPattern::Group(Box::new(SqlPattern::Exclude(
                match sym {
                    SymbolRef::Named(s) => SqlSymbol::Named(Ident::new(s)),
                    SymbolRef::Start => SqlSymbol::Start,
                    SymbolRef::End => SqlSymbol::End,
                },
            )))),
            _ => plan_err!("sqlparser 0.63.0 only unparses single-symbol exclusions"),
        },
        Pattern::Permute(items) => {
            let mut syms = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Pattern::Symbol(SymbolRef::Named(s)) => {
                        syms.push(SqlSymbol::Named(Ident::new(s)))
                    }
                    _ => return plan_err!("sqlparser 0.63.0 PERMUTE expects named symbols"),
                }
            }
            Ok(SqlPattern::Permute(syms))
        }
        Pattern::Concat(items) => Ok(SqlPattern::Concat(
            items
                .iter()
                .map(unparse_pattern)
                .collect::<Result<Vec<_>>>()?,
        )),
        Pattern::Alternation(items) => Ok(SqlPattern::Group(Box::new(SqlPattern::Alternation(
            items
                .iter()
                .map(unparse_pattern)
                .collect::<Result<Vec<_>>>()?,
        )))),
        Pattern::Repetition {
            inner,
            min,
            max,
            reluctant,
        } => {
            if *reluctant {
                return plan_err!("sqlparser 0.63.0 does not yet unparse reluctant quantifiers");
            }
            let quant = match (*min, *max) {
                (0, None) => SqlQuant::ZeroOrMore,
                (1, None) => SqlQuant::OneOrMore,
                (0, Some(1)) => SqlQuant::AtMostOne,
                (n, Some(m)) if n == m => SqlQuant::Exactly(n),
                (n, None) => SqlQuant::AtLeast(n),
                (0, Some(m)) => SqlQuant::AtMost(m),
                (n, Some(m)) => SqlQuant::Range(n, m),
            };
            Ok(SqlPattern::Repetition(
                Box::new(unparse_pattern(inner)?),
                quant,
            ))
        }
    }
}
