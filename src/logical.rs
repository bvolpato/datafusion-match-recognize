// SPDX-License-Identifier: Apache-2.0

//! Custom [`UserDefinedLogicalNodeCore`] for `MATCH_RECOGNIZE`.

use std::collections::HashSet;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow_schema::{DataType, Field, FieldRef, Schema};
use datafusion::optimizer::analyzer::type_coercion::TypeCoercionRewriter;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::{DFSchema, DFSchemaRef, Result, plan_datafusion_err, plan_err};
use datafusion_expr::expr_rewriter::strip_outer_reference;
use datafusion_expr::logical_plan::{Extension, LogicalPlan};
use datafusion_expr::utils::expr_to_columns;
use datafusion_expr::{Expr, ExprSchemable, UserDefinedLogicalNodeCore};

use crate::ast::{
    AfterMatchSkip, AggKind, MatchRecognizeConfig, MeasureSlot, MeasureSlotKind, MeasureSpec,
    OrderBySpec, Pattern, RowsPerMatch, SubsetDef, SymbolDef, SymbolDefKind,
};

/// Logical plan node representing a SQL `MATCH_RECOGNIZE` relation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatchRecognizeNode {
    pub input: LogicalPlan,
    pub partition_by: Vec<Expr>,
    pub order_by: Vec<OrderBySpec>,
    pub measures: Vec<MeasureSpec>,
    pub rows_per_match: RowsPerMatch,
    pub after_match_skip: AfterMatchSkip,
    pub pattern: Pattern,
    pub subsets: Vec<SubsetDef>,
    pub definitions: Vec<SymbolDef>,
    pub config: MatchRecognizeConfig,
    pub schema: DFSchemaRef,
}

impl PartialOrd for MatchRecognizeNode {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.input
            .partial_cmp(&other.input)
            .and_then(|o| o.then_with(|| self.pattern.cmp(&other.pattern)).into())
    }
}

impl MatchRecognizeNode {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        input: LogicalPlan,
        partition_by: Vec<Expr>,
        order_by: Vec<OrderBySpec>,
        measures: Vec<MeasureSpec>,
        rows_per_match: RowsPerMatch,
        after_match_skip: AfterMatchSkip,
        pattern: Pattern,
        definitions: Vec<SymbolDef>,
        config: MatchRecognizeConfig,
    ) -> Result<Self> {
        Self::try_new_with_subsets(
            input,
            partition_by,
            order_by,
            measures,
            rows_per_match,
            after_match_skip,
            pattern,
            vec![],
            definitions,
            config,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_subsets(
        input: LogicalPlan,
        partition_by: Vec<Expr>,
        order_by: Vec<OrderBySpec>,
        measures: Vec<MeasureSpec>,
        rows_per_match: RowsPerMatch,
        after_match_skip: AfterMatchSkip,
        pattern: Pattern,
        subsets: Vec<SubsetDef>,
        definitions: Vec<SymbolDef>,
        config: MatchRecognizeConfig,
    ) -> Result<Self> {
        let mut coerced_measures = Vec::with_capacity(measures.len());
        for m in measures {
            let slot_schema = build_slot_schema(input.schema(), &m.slots, rows_per_match)?;
            let mut rewriter = TypeCoercionRewriter::new(&slot_schema);
            let outer_expr = m.outer_expr.rewrite(&mut rewriter)?.data;
            coerced_measures.push(MeasureSpec {
                alias: m.alias,
                outer_expr,
                slots: m.slots,
            });
        }

        let mut coerced_defs = Vec::with_capacity(definitions.len());
        for def in definitions {
            let kind = match def.kind {
                SymbolDefKind::Stateless { predicate } => SymbolDefKind::Stateless { predicate },
                SymbolDefKind::Stateful { outer_expr, slots } => {
                    let slot_schema =
                        build_slot_schema(input.schema(), &slots, RowsPerMatch::OneRow)?;
                    let mut rewriter = TypeCoercionRewriter::new(&slot_schema);
                    let outer_expr = outer_expr.rewrite(&mut rewriter)?.data;
                    SymbolDefKind::Stateful { outer_expr, slots }
                }
            };
            coerced_defs.push(SymbolDef {
                name: def.name,
                kind,
            });
        }

        let schema = compute_output_schema(
            input.schema(),
            &partition_by,
            &order_by,
            &coerced_measures,
            rows_per_match,
        )?;
        Ok(Self {
            input,
            partition_by,
            order_by,
            measures: coerced_measures,
            rows_per_match,
            after_match_skip,
            pattern,
            subsets,
            definitions: coerced_defs,
            config,
            schema,
        })
    }

    pub fn into_plan(self) -> LogicalPlan {
        LogicalPlan::Extension(Extension {
            node: Arc::new(self),
        })
    }
}

/// Computes the synthetic `DFSchema` for a measure or stateful define's extracted leaf slots.
pub fn build_slot_schema(
    input_schema: &DFSchema,
    slots: &[MeasureSlot],
    rows_per_match: RowsPerMatch,
) -> Result<DFSchema> {
    let mut fields = Vec::with_capacity(slots.len());
    for slot in slots {
        let (dt, nullable) = slot_data_type_and_nullability(input_schema, slot, rows_per_match)?;
        fields.push(Field::new(&slot.slot_name, dt, nullable));
    }
    DFSchema::try_from(Schema::new(fields))
}

/// Infers the Arrow `(DataType, nullable)` for a single [`MeasureSlot`].
pub fn slot_data_type_and_nullability(
    input_schema: &DFSchema,
    slot: &MeasureSlot,
    rows_per_match: RowsPerMatch,
) -> Result<(DataType, bool)> {
    let unmatched = rows_per_match.include_unmatched_rows();
    match &slot.kind {
        MeasureSlotKind::MatchNumber => Ok((DataType::Int64, unmatched)),
        MeasureSlotKind::Classifier { .. } => Ok((DataType::Utf8, true)),
        MeasureSlotKind::Nav { inner_expr, .. } => {
            let dt = inner_expr.get_type(input_schema)?;
            Ok((dt, true))
        }
        MeasureSlotKind::CurrentRow { inner_expr } => {
            let dt = inner_expr.get_type(input_schema)?;
            Ok((dt, true))
        }
        MeasureSlotKind::Agg { agg_kind, args, .. } => match agg_kind {
            AggKind::CountStar | AggKind::CountSymbol | AggKind::Count => {
                Ok((DataType::Int64, unmatched))
            }
            AggKind::Sum => {
                let expr = args.first().ok_or_else(|| {
                    plan_datafusion_err!("SUM in MEASURES/DEFINE requires an input expression")
                })?;
                let inner_dt = expr.get_type(input_schema)?;
                Ok((sum_result_type(&inner_dt)?, true))
            }
            AggKind::Avg => Ok((DataType::Float64, true)),
            AggKind::Min | AggKind::Max => {
                let expr = args.first().ok_or_else(|| {
                    plan_datafusion_err!("MIN/MAX in MEASURES/DEFINE requires an input expression")
                })?;
                let inner_dt = expr.get_type(input_schema)?;
                Ok((inner_dt, true))
            }
            AggKind::Udaf(udaf) => {
                let raw_fields: Vec<FieldRef> = args
                    .iter()
                    .map(|a| {
                        let (_, f) = a.to_field(input_schema)?;
                        Ok(f)
                    })
                    .collect::<Result<_>>()?;
                let coerced_fields = datafusion_expr::type_coercion::functions::fields_with_udf(
                    &raw_fields,
                    udaf.as_ref(),
                )?;
                let ret = udaf.return_field(&coerced_fields)?;
                Ok((ret.data_type().clone(), true))
            }
        },
    }
}

pub fn compute_output_schema(
    input_schema: &DFSchema,
    partition_by: &[Expr],
    order_by: &[OrderBySpec],
    measures: &[MeasureSpec],
    rows_per_match: RowsPerMatch,
) -> Result<DFSchemaRef> {
    let mut fields: Vec<Field> = Vec::new();
    let mut included_input_indices: HashSet<usize> = HashSet::new();

    for expr in partition_by {
        if let Expr::Column(col) = expr {
            let idx = input_schema.index_of_column(col)?;
            if included_input_indices.insert(idx) {
                let f = input_schema.field(idx);
                fields.push(Field::new(f.name(), f.data_type().clone(), f.is_nullable()));
            }
        } else {
            let (_, field) = expr.to_field(input_schema)?;
            fields.push(Field::new(
                expr.schema_name().to_string(),
                field.data_type().clone(),
                field.is_nullable(),
            ));
        }
    }

    if !rows_per_match.is_one_row() {
        for ob in order_by {
            if let Expr::Column(col) = &ob.expr {
                let idx = input_schema.index_of_column(col)?;
                if included_input_indices.insert(idx) {
                    let f = input_schema.field(idx);
                    fields.push(Field::new(f.name(), f.data_type().clone(), true));
                }
            }
        }
    }

    for m in measures {
        let slot_schema = build_slot_schema(input_schema, &m.slots, rows_per_match)?;
        let (_, field) = m.outer_expr.to_field(&slot_schema)?;
        let nullable = if rows_per_match.include_unmatched_rows() {
            true
        } else {
            field.is_nullable()
        };
        fields.push(Field::new(&m.alias, field.data_type().clone(), nullable));
    }

    if !rows_per_match.is_one_row() {
        for (idx, f) in input_schema.fields().iter().enumerate() {
            if included_input_indices.insert(idx) {
                fields.push(Field::new(f.name(), f.data_type().clone(), true));
            }
        }
    }

    let arrow_schema = Schema::new(fields);
    Ok(Arc::new(DFSchema::try_from(arrow_schema)?))
}

fn sum_result_type(dt: &DataType) -> Result<DataType> {
    match dt {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => Ok(DataType::Int64),
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        DataType::Float16 | DataType::Float32 | DataType::Float64 => Ok(DataType::Float64),
        DataType::Decimal128(p, s) => Ok(DataType::Decimal128((*p + 10).min(38), *s)),
        other => plan_err!("SUM in MATCH_RECOGNIZE does not support type {other}"),
    }
}

fn push_slot_exprs(slots: &[MeasureSlot], exprs: &mut Vec<Expr>) {
    for slot in slots {
        match &slot.kind {
            MeasureSlotKind::Nav { inner_expr, .. }
            | MeasureSlotKind::CurrentRow { inner_expr } => {
                exprs.push(inner_expr.clone());
            }
            MeasureSlotKind::Agg { args, .. } => {
                exprs.extend(args.iter().cloned());
            }
            MeasureSlotKind::MatchNumber | MeasureSlotKind::Classifier { .. } => {}
        }
    }
}

fn rebuild_slots_from_iter(
    orig_slots: &[MeasureSlot],
    iter: &mut impl Iterator<Item = Expr>,
) -> Result<Vec<MeasureSlot>> {
    let mut new_slots = Vec::with_capacity(orig_slots.len());
    for slot in orig_slots {
        let kind = match &slot.kind {
            MeasureSlotKind::Nav {
                logical_nav,
                logical_offset,
                physical_nav,
                physical_offset,
                symbol,
                ..
            } => MeasureSlotKind::Nav {
                logical_nav: *logical_nav,
                logical_offset: *logical_offset,
                physical_nav: *physical_nav,
                physical_offset: *physical_offset,
                symbol: symbol.clone(),
                inner_expr: iter.next().ok_or_else(|| {
                    plan_datafusion_err!("MatchRecognize: missing nav slot expression")
                })?,
            },
            MeasureSlotKind::CurrentRow { .. } => MeasureSlotKind::CurrentRow {
                inner_expr: iter.next().ok_or_else(|| {
                    plan_datafusion_err!("MatchRecognize: missing current_row slot expression")
                })?,
            },
            MeasureSlotKind::Agg {
                agg_kind,
                symbol,
                args,
                distinct,
                arg_physical_nav,
            } => {
                let mut new_args = Vec::with_capacity(args.len());
                for _ in 0..args.len() {
                    new_args.push(iter.next().ok_or_else(|| {
                        plan_datafusion_err!("MatchRecognize: missing agg slot expression")
                    })?);
                }
                MeasureSlotKind::Agg {
                    agg_kind: agg_kind.clone(),
                    symbol: symbol.clone(),
                    args: new_args,
                    distinct: *distinct,
                    arg_physical_nav: *arg_physical_nav,
                }
            }
            other => other.clone(),
        };
        new_slots.push(MeasureSlot {
            slot_name: slot.slot_name.clone(),
            semantics: slot.semantics,
            kind,
        });
    }
    Ok(new_slots)
}

impl UserDefinedLogicalNodeCore for MatchRecognizeNode {
    fn name(&self) -> &str {
        "MatchRecognize"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![&self.input]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        let mut exprs = Vec::new();
        exprs.extend(self.partition_by.iter().cloned());
        exprs.extend(self.order_by.iter().map(|o| o.expr.clone()));
        for def in &self.definitions {
            match &def.kind {
                SymbolDefKind::Stateless { predicate } => exprs.push(predicate.clone()),
                SymbolDefKind::Stateful { slots, .. } => push_slot_exprs(slots, &mut exprs),
            }
        }
        for m in &self.measures {
            push_slot_exprs(&m.slots, &mut exprs);
        }
        exprs
    }

    fn fmt_for_explain(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatchRecognize: pattern=[{}], rows_per_match=[{}], skip=[{}]",
            self.pattern, self.rows_per_match, self.after_match_skip
        )?;
        if !self.partition_by.is_empty() {
            write!(f, ", partition_by=[")?;
            for (i, e) in self.partition_by.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{e}")?;
            }
            write!(f, "]")?;
        }
        if !self.order_by.is_empty() {
            write!(f, ", order_by=[")?;
            for (i, o) in self.order_by.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{} {}", o.expr, if o.asc { "ASC" } else { "DESC" })?;
            }
            write!(f, "]")?;
        }
        if !self.subsets.is_empty() {
            write!(f, ", subsets=[")?;
            for (i, s) in self.subsets.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}=({})", s.name, s.members.join(", "))?;
            }
            write!(f, "]")?;
        }
        if !self.measures.is_empty() {
            write!(f, ", measures=[")?;
            for (i, m) in self.measures.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", m.alias)?;
            }
            write!(f, "]")?;
        }
        Ok(())
    }

    fn with_exprs_and_inputs(
        &self,
        exprs: Vec<Expr>,
        mut inputs: Vec<LogicalPlan>,
    ) -> Result<Self> {
        if inputs.len() != 1 {
            return plan_err!(
                "MatchRecognize expects exactly 1 input, got {}",
                inputs.len()
            );
        }
        let input = inputs.swap_remove(0);
        let mut iter = exprs.into_iter();

        let mut partition_by = Vec::with_capacity(self.partition_by.len());
        for _ in 0..self.partition_by.len() {
            partition_by.push(iter.next().ok_or_else(|| {
                plan_datafusion_err!("MatchRecognize: missing partition_by expression")
            })?);
        }

        let mut order_by = Vec::with_capacity(self.order_by.len());
        for orig in &self.order_by {
            let expr = iter.next().ok_or_else(|| {
                plan_datafusion_err!("MatchRecognize: missing order_by expression")
            })?;
            order_by.push(OrderBySpec {
                expr,
                asc: orig.asc,
                nulls_first: orig.nulls_first,
            });
        }

        let mut definitions = Vec::with_capacity(self.definitions.len());
        for orig in &self.definitions {
            let kind = match &orig.kind {
                SymbolDefKind::Stateless { .. } => {
                    let predicate = iter.next().ok_or_else(|| {
                        plan_datafusion_err!("MatchRecognize: missing stateless define expression")
                    })?;
                    SymbolDefKind::Stateless { predicate }
                }
                SymbolDefKind::Stateful { outer_expr, slots } => {
                    let new_slots = rebuild_slots_from_iter(slots, &mut iter)?;
                    SymbolDefKind::Stateful {
                        outer_expr: outer_expr.clone(),
                        slots: new_slots,
                    }
                }
            };
            definitions.push(SymbolDef {
                name: orig.name.clone(),
                kind,
            });
        }

        let mut measures = Vec::with_capacity(self.measures.len());
        for orig in &self.measures {
            let new_slots = rebuild_slots_from_iter(&orig.slots, &mut iter)?;
            measures.push(MeasureSpec {
                alias: orig.alias.clone(),
                outer_expr: orig.outer_expr.clone(),
                slots: new_slots,
            });
        }

        if iter.next().is_some() {
            return plan_err!("MatchRecognize: unexpected extra expressions");
        }

        Self::try_new_with_subsets(
            input,
            partition_by,
            order_by,
            measures,
            self.rows_per_match,
            self.after_match_skip.clone(),
            self.pattern.clone(),
            self.subsets.clone(),
            definitions,
            self.config,
        )
    }

    fn necessary_children_exprs(&self, output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        if !self.rows_per_match.is_one_row() {
            return None;
        }
        let input_schema = self.input.schema();
        let mut needed: HashSet<usize> = HashSet::new();

        for expr in self.expressions() {
            let mut cols = HashSet::new();
            if expr_to_columns(&strip_outer_reference(expr), &mut cols).is_err() {
                return None;
            }
            for col in cols {
                if let Ok(idx) = input_schema.index_of_column(&col) {
                    needed.insert(idx);
                } else {
                    return None;
                }
            }
        }

        for &out_idx in output_columns {
            if out_idx < self.partition_by.len()
                && let Expr::Column(col) = &self.partition_by[out_idx]
                && let Ok(idx) = input_schema.index_of_column(col)
            {
                needed.insert(idx);
            }
        }

        if needed.is_empty() && !input_schema.fields().is_empty() {
            needed.insert(0);
        }

        let mut sorted: Vec<usize> = needed.into_iter().collect();
        sorted.sort_unstable();
        Some(vec![sorted])
    }
}
