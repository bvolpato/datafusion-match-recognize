// SPDX-License-Identifier: Apache-2.0

//! Physical evaluation of `MATCH_RECOGNIZE` `MEASURES`, stateful `DEFINE` predicates,
//! `SUBSET` union variables, `DISTINCT` aggregates, navigation over `CLASSIFIER()`,
//! `PREV`/`NEXT` inside aggregates, and arbitrary DataFusion [`AggregateUDF`]s via
//! [`datafusion_expr::Accumulator`].

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Builder, Int64Builder, RecordBatch, RecordBatchOptions,
    StringBuilder, UInt32Array, new_null_array,
};
use arrow::compute::{self, kernels::zip};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion_common::{DataFusionError, Result, ScalarValue, exec_err};
use datafusion_expr::function::AccumulatorArgs;
use datafusion_physical_expr::PhysicalExpr;

use crate::ast::{AggKind, EvalSemantics, LogicalNavKind, PhysicalNavKind, RowsPerMatch};
use crate::logical::build_agg_metadata_arrow_schema;
use crate::matcher::{FoundMatch, MatchStep, PartitionEvent, StatefulPredicateEvaluator};
use crate::pattern::{Program, SymbolId, SymbolSet};

/// Physical counterpart of [`crate::ast::MeasureSlotKind`].
#[derive(Debug, Clone)]
pub enum PhysicalMeasureSlotKind {
    MatchNumber,
    Classifier {
        symbol: Option<String>,
        symbol_set: Option<SymbolSet>,
        logical_nav: Option<LogicalNavKind>,
        logical_offset: u32,
        physical_nav: Option<PhysicalNavKind>,
        physical_offset: u32,
    },
    Nav {
        logical_nav: Option<LogicalNavKind>,
        logical_offset: u32,
        physical_nav: Option<PhysicalNavKind>,
        physical_offset: u32,
        symbol: Option<String>,
        symbol_set: Option<SymbolSet>,
        inner_expr: Arc<dyn PhysicalExpr>,
    },
    Agg {
        agg_kind: AggKind,
        symbol: Option<String>,
        symbol_set: Option<SymbolSet>,
        args: Vec<Arc<dyn PhysicalExpr>>,
        distinct: bool,
        arg_physical_nav: Option<(PhysicalNavKind, u32)>,
        uses_match_metadata: bool,
    },
    CurrentRow {
        inner_expr: Arc<dyn PhysicalExpr>,
    },
}

/// A compiled pattern-recognition leaf slot ready for physical execution.
#[derive(Debug, Clone)]
pub struct PhysicalMeasureSlot {
    pub slot_name: String,
    pub semantics: EvalSemantics,
    pub data_type: DataType,
    pub nullable: bool,
    pub kind: PhysicalMeasureSlotKind,
}

/// A compiled `MEASURES` entry: leaf slots + outer vectorized [`PhysicalExpr`].
#[derive(Debug, Clone)]
pub struct PhysicalMeasure {
    pub alias: String,
    pub outer_expr: Arc<dyn PhysicalExpr>,
    pub slot_schema: SchemaRef,
    pub slots: Vec<PhysicalMeasureSlot>,
}

/// Borrowed specification for one stateful `DEFINE` symbol passed to [`PartitionStatefulEvaluator`].
pub struct StatefulDefineInput<'a> {
    pub sym_id: SymbolId,
    pub outer_expr: &'a Arc<dyn PhysicalExpr>,
    pub slot_schema: &'a SchemaRef,
    pub slots: &'a [PhysicalMeasureSlot],
}

struct CompiledStatefulSymbol {
    outer_expr: Arc<dyn PhysicalExpr>,
    slot_schema: SchemaRef,
    slots: Vec<PhysicalMeasureSlot>,
    /// Pre-evaluated inner Arrow arrays (`precomputed_args[slot_idx][arg_idx]`).
    precomputed_args: Vec<Vec<ArrayRef>>,
}

/// Evaluates Tier 2 stateful `DEFINE` predicates against a partition's precomputed slot arrays.
pub struct PartitionStatefulEvaluator<'a> {
    program: &'a Program,
    partition_batch: &'a RecordBatch,
    input_schema: SchemaRef,
    num_part_rows: usize,
    by_symbol: HashMap<SymbolId, CompiledStatefulSymbol>,
}

impl<'a> PartitionStatefulEvaluator<'a> {
    /// Pre-evaluates all non-metadata stateful `DEFINE` inner expressions once for `partition_batch`.
    pub fn try_new(
        partition_batch: &'a RecordBatch,
        program: &'a Program,
        stateful_defs: &[StatefulDefineInput<'_>],
    ) -> Result<Self> {
        let num_part_rows = partition_batch.num_rows();
        let mut by_symbol = HashMap::with_capacity(stateful_defs.len());

        for def in stateful_defs {
            let mut precomputed_args = Vec::with_capacity(def.slots.len());
            for slot in def.slots {
                let arrs = match &slot.kind {
                    PhysicalMeasureSlotKind::Nav { inner_expr, .. }
                    | PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
                        vec![
                            inner_expr
                                .evaluate(partition_batch)?
                                .into_array(num_part_rows)?,
                        ]
                    }
                    PhysicalMeasureSlotKind::Agg {
                        agg_kind,
                        args,
                        uses_match_metadata,
                        ..
                    } => {
                        if *uses_match_metadata {
                            vec![]
                        } else {
                            let mut evaluated = Vec::with_capacity(args.len());
                            for (i, e) in args.iter().enumerate() {
                                let raw = e.evaluate(partition_batch)?.into_array(num_part_rows)?;
                                let pre = if i == 0 {
                                    match agg_kind {
                                        AggKind::Avg => {
                                            compute::cast(raw.as_ref(), &DataType::Float64)?
                                        }
                                        AggKind::Sum if raw.data_type() != &slot.data_type => {
                                            compute::cast(raw.as_ref(), &slot.data_type)?
                                        }
                                        _ => raw,
                                    }
                                } else {
                                    raw
                                };
                                evaluated.push(pre);
                            }
                            evaluated
                        }
                    }
                    PhysicalMeasureSlotKind::MatchNumber
                    | PhysicalMeasureSlotKind::Classifier { .. } => vec![],
                };
                precomputed_args.push(arrs);
            }
            by_symbol.insert(
                def.sym_id,
                CompiledStatefulSymbol {
                    outer_expr: Arc::clone(def.outer_expr),
                    slot_schema: Arc::clone(def.slot_schema),
                    slots: def.slots.to_vec(),
                    precomputed_args,
                },
            );
        }

        Ok(Self {
            program,
            partition_batch,
            input_schema: partition_batch.schema(),
            num_part_rows,
            by_symbol,
        })
    }
}

impl<'a> StatefulPredicateEvaluator for PartitionStatefulEvaluator<'a> {
    #[inline]
    fn is_stateful(&self, sym: SymbolId) -> bool {
        self.by_symbol.contains_key(&sym)
    }

    #[inline]
    fn has_any_stateful(&self) -> bool {
        !self.by_symbol.is_empty()
    }

    fn eval_stateful(
        &mut self,
        sym: SymbolId,
        tentative_steps: &[MatchStep],
        candidate_match_number: i64,
    ) -> Result<bool> {
        let compiled = self
            .by_symbol
            .get(&sym)
            .expect("eval_stateful called only for stateful symbols");
        let current_row = tentative_steps
            .last()
            .expect("tentative_steps always contains candidate row")
            .row_idx;

        let mut slot_arrays: Vec<ArrayRef> = Vec::with_capacity(compiled.slots.len());
        for (slot, pre_args) in compiled.slots.iter().zip(&compiled.precomputed_args) {
            let scalar = eval_single_slot_scalar(
                self.partition_batch,
                tentative_steps,
                tentative_steps,
                candidate_match_number,
                current_row,
                self.num_part_rows,
                self.program,
                self.input_schema.as_ref(),
                slot,
                pre_args,
            )?;
            slot_arrays.push(scalar.to_array_of_size(1)?);
        }

        let slot_batch = RecordBatch::try_new_with_options(
            Arc::clone(&compiled.slot_schema),
            slot_arrays,
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )?;
        let evaluated = compiled.outer_expr.evaluate(&slot_batch)?.into_array(1)?;
        let bool_arr = evaluated
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Execution(format!(
                    "Stateful DEFINE predicate must evaluate to Boolean, got {}",
                    evaluated.data_type()
                ))
            })?;
        Ok(bool_arr.is_valid(0) && bool_arr.value(0))
    }
}

#[allow(clippy::too_many_arguments)]
fn eval_single_slot_scalar(
    partition_batch: &RecordBatch,
    horizon: &[MatchStep],
    full_steps: &[MatchStep],
    match_number: i64,
    current_row: usize,
    num_part_rows: usize,
    program: &Program,
    input_schema: &Schema,
    slot: &PhysicalMeasureSlot,
    precomputed_args: &[ArrayRef],
) -> Result<ScalarValue> {
    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => Ok(ScalarValue::Int64(Some(match_number))),
        PhysicalMeasureSlotKind::Classifier {
            symbol_set,
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            ..
        } => {
            let val = eval_classifier_on_horizon(
                horizon,
                full_steps,
                symbol_set.as_ref(),
                *logical_nav,
                *logical_offset as usize,
                *physical_nav,
                *physical_offset as usize,
                program,
            );
            Ok(ScalarValue::Utf8(val))
        }
        PhysicalMeasureSlotKind::CurrentRow { .. } => {
            let arr = &precomputed_args[0];
            ScalarValue::try_from_array(arr.as_ref(), current_row)
        }
        PhysicalMeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            symbol_set,
            ..
        } => {
            let arr = &precomputed_args[0];
            let Some(base_row) = navigate_logical_row(
                horizon,
                *logical_nav,
                *logical_offset as usize,
                symbol_set.as_ref(),
            ) else {
                return ScalarValue::try_from(arr.data_type());
            };
            let Some(final_row) = navigate_physical(
                base_row,
                *physical_nav,
                *physical_offset as usize,
                num_part_rows,
            ) else {
                return ScalarValue::try_from(arr.data_type());
            };
            ScalarValue::try_from_array(arr.as_ref(), final_row)
        }
        PhysicalMeasureSlotKind::Agg {
            agg_kind,
            symbol_set,
            args,
            distinct,
            arg_physical_nav,
            uses_match_metadata,
            ..
        } => {
            if *uses_match_metadata {
                eval_metadata_agg_scalar(
                    partition_batch,
                    horizon,
                    match_number,
                    program,
                    agg_kind,
                    symbol_set.as_ref(),
                    args,
                    *distinct,
                    *arg_physical_nav,
                    &slot.data_type,
                )
            } else {
                eval_single_agg_scalar(
                    horizon,
                    num_part_rows,
                    input_schema,
                    agg_kind,
                    symbol_set.as_ref(),
                    args,
                    *distinct,
                    *arg_physical_nav,
                    precomputed_args,
                    &slot.data_type,
                )
            }
        }
    }
}

/// Evaluates `CLASSIFIER([sym])` with optional `FIRST`/`LAST` and `PREV`/`NEXT` navigation.
///
/// Under SQL:2016 / Trino:
/// - In `MEASURES`, the match is complete so `NEXT(CLASSIFIER(), n)` can access labels past the
///   current row up to `full_steps.len()`.
/// - In `DEFINE`, `full_steps == tentative_steps`, so `NEXT(CLASSIFIER(), n)` past the candidate
///   row returns `NULL`.
#[allow(clippy::too_many_arguments)]
fn eval_classifier_on_horizon(
    horizon: &[MatchStep],
    full_steps: &[MatchStep],
    symbol_set: Option<&SymbolSet>,
    logical_nav: Option<LogicalNavKind>,
    logical_offset: usize,
    physical_nav: Option<PhysicalNavKind>,
    physical_offset: usize,
    program: &Program,
) -> Option<String> {
    if horizon.is_empty() {
        return None;
    }
    let base_idx = match (symbol_set, logical_nav.unwrap_or(LogicalNavKind::Last)) {
        (Some(set), LogicalNavKind::First) => horizon
            .iter()
            .enumerate()
            .filter(|(_, s)| set.contains(s.sym))
            .nth(logical_offset)
            .map(|(i, _)| i)?,
        (Some(set), LogicalNavKind::Last) => horizon
            .iter()
            .enumerate()
            .rev()
            .filter(|(_, s)| set.contains(s.sym))
            .nth(logical_offset)
            .map(|(i, _)| i)?,
        (None, LogicalNavKind::First) => {
            if logical_offset < horizon.len() {
                logical_offset
            } else {
                return None;
            }
        }
        (None, LogicalNavKind::Last) => {
            if logical_offset < horizon.len() {
                horizon.len() - 1 - logical_offset
            } else {
                return None;
            }
        }
    };

    let final_idx = match physical_nav {
        None => base_idx,
        Some(PhysicalNavKind::Prev) => base_idx.checked_sub(physical_offset)?,
        Some(PhysicalNavKind::Next) => {
            let t = base_idx.checked_add(physical_offset)?;
            if t < full_steps.len() { t } else { return None }
        }
    };

    let step = full_steps.get(final_idx)?;
    Some(program.id_to_symbol[step.sym as usize].clone())
}

/// Evaluates an aggregate whose arguments reference runtime match metadata (`CLASSIFIER()` or
/// `MATCH_NUMBER()`), such as `ARRAY_AGG(CLASSIFIER())` or `SUM(MATCH_NUMBER() * 100)`.
#[allow(clippy::too_many_arguments)]
fn eval_metadata_agg_scalar(
    partition_batch: &RecordBatch,
    horizon: &[MatchStep],
    match_number: i64,
    program: &Program,
    agg_kind: &AggKind,
    symbol_set: Option<&SymbolSet>,
    arg_exprs: &[Arc<dyn PhysicalExpr>],
    distinct: bool,
    arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    result_type: &DataType,
) -> Result<ScalarValue> {
    let num_part_rows = partition_batch.num_rows();
    let matched_steps: Vec<(Option<usize>, SymbolId)> = horizon
        .iter()
        .filter_map(|s| {
            if let Some(set) = symbol_set
                && !set.contains(s.sym)
            {
                return None;
            }
            let row_idx = match arg_physical_nav {
                None => Some(s.row_idx),
                Some((kind, off)) => {
                    navigate_physical(s.row_idx, Some(kind), off as usize, num_part_rows)
                }
            };
            Some((row_idx, s.sym))
        })
        .collect();

    let n = matched_steps.len();
    let ext_schema = Arc::new(build_agg_metadata_arrow_schema(
        partition_batch.schema().as_ref(),
    ));
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(ext_schema.fields().len());
    let take_idx = UInt32Array::from_iter(matched_steps.iter().map(|(r, _)| r.map(|i| i as u32)));
    for col in partition_batch.columns() {
        cols.push(compute::take(col.as_ref(), &take_idx, None)?);
    }

    let mut cls_builder = StringBuilder::with_capacity(n, n * 8);
    let mut mn_builder = Int64Builder::with_capacity(n);
    for (_, sym) in &matched_steps {
        cls_builder.append_value(&program.id_to_symbol[*sym as usize]);
        mn_builder.append_value(match_number);
    }
    cols.push(Arc::new(cls_builder.finish()));
    cols.push(Arc::new(mn_builder.finish()));

    let horizon_batch = RecordBatch::try_new_with_options(
        Arc::clone(&ext_schema),
        cols,
        &RecordBatchOptions::new().with_row_count(Some(n)),
    )?;

    let mut evaluated_args = Vec::with_capacity(arg_exprs.len());
    for (i, expr) in arg_exprs.iter().enumerate() {
        let raw = expr.evaluate(&horizon_batch)?.into_array(n)?;
        let pre = if i == 0 {
            match agg_kind {
                AggKind::Avg => compute::cast(raw.as_ref(), &DataType::Float64)?,
                AggKind::Sum if raw.data_type() != result_type => {
                    compute::cast(raw.as_ref(), result_type)?
                }
                _ => raw,
            }
        } else {
            raw
        };
        evaluated_args.push(pre);
    }

    // Build synthetic 0..n steps so eval_single_agg_scalar indexes directly into `evaluated_args`.
    let local_steps: Vec<MatchStep> = (0..n)
        .map(|i| MatchStep {
            row_idx: i,
            sym: 0,
            excluded: false,
        })
        .collect();
    eval_single_agg_scalar(
        &local_steps,
        n,
        ext_schema.as_ref(),
        agg_kind,
        None,
        arg_exprs,
        distinct,
        None,
        &evaluated_args,
        result_type,
    )
}

#[allow(clippy::too_many_arguments)]
fn eval_single_agg_scalar(
    horizon: &[MatchStep],
    num_part_rows: usize,
    input_schema: &Schema,
    agg_kind: &AggKind,
    symbol_set: Option<&SymbolSet>,
    arg_exprs: &[Arc<dyn PhysicalExpr>],
    distinct: bool,
    arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    precomputed_args: &[ArrayRef],
    result_type: &DataType,
) -> Result<ScalarValue> {
    match agg_kind {
        AggKind::CountStar | AggKind::CountSymbol => {
            let count = matching_row_indices(horizon, symbol_set, None, num_part_rows)
                .filter(|o| o.is_some())
                .count();
            Ok(ScalarValue::Int64(Some(count as i64)))
        }
        AggKind::Count => {
            let arr = &precomputed_args[0];
            if distinct {
                let mut seen: HashSet<ScalarValue> = HashSet::new();
                for maybe_idx in
                    matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows)
                {
                    if let Some(idx) = maybe_idx
                        && arr.is_valid(idx)
                    {
                        seen.insert(ScalarValue::try_from_array(arr.as_ref(), idx)?);
                    }
                }
                Ok(ScalarValue::Int64(Some(seen.len() as i64)))
            } else {
                let count =
                    matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows)
                        .filter(|maybe_idx| maybe_idx.is_some_and(|idx| arr.is_valid(idx)))
                        .count();
                Ok(ScalarValue::Int64(Some(count as i64)))
            }
        }
        AggKind::Avg => {
            let arr = &precomputed_args[0];
            let f64_arr = arr
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!("Expected Float64Array for AVG")
                })?;
            let mut sum = 0.0f64;
            let mut count = 0u64;
            if distinct {
                let mut seen: HashSet<u64> = HashSet::new();
                for maybe_idx in
                    matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows)
                {
                    if let Some(idx) = maybe_idx
                        && f64_arr.is_valid(idx)
                    {
                        let v = f64_arr.value(idx);
                        if seen.insert(v.to_bits()) {
                            sum += v;
                            count += 1;
                        }
                    }
                }
            } else {
                for maybe_idx in
                    matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows)
                {
                    if let Some(idx) = maybe_idx
                        && f64_arr.is_valid(idx)
                    {
                        sum += f64_arr.value(idx);
                        count += 1;
                    }
                }
            }
            if count == 0 {
                Ok(ScalarValue::Float64(None))
            } else {
                Ok(ScalarValue::Float64(Some(sum / count as f64)))
            }
        }
        AggKind::Sum => {
            let arr = &precomputed_args[0];
            let valid_indices =
                matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows)
                    .flatten();
            if distinct {
                sum_indices_distinct(arr, valid_indices, result_type)
            } else {
                sum_indices(arr, valid_indices, result_type)
            }
        }
        AggKind::Min | AggKind::Max => {
            let arr = &precomputed_args[0];
            let is_min = matches!(agg_kind, AggKind::Min);
            let mut best: Option<ScalarValue> = None;
            for idx in
                matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows).flatten()
            {
                if !arr.is_valid(idx) {
                    continue;
                }
                let v = ScalarValue::try_from_array(arr.as_ref(), idx)?;
                best = match best {
                    None => Some(v),
                    Some(curr) => {
                        let ord = v.partial_cmp(&curr);
                        if (is_min && ord == Some(std::cmp::Ordering::Less))
                            || (!is_min && ord == Some(std::cmp::Ordering::Greater))
                        {
                            Some(v)
                        } else {
                            Some(curr)
                        }
                    }
                };
            }
            match best {
                Some(v) => Ok(v),
                None => ScalarValue::try_from(arr.data_type()),
            }
        }
        AggKind::Udaf(udaf) => {
            let raw_fields: Vec<FieldRef> = precomputed_args
                .iter()
                .enumerate()
                .map(|(i, a)| Arc::new(Field::new(format!("arg_{i}"), a.data_type().clone(), true)))
                .collect();
            let coerced_fields = datafusion_expr::type_coercion::functions::fields_with_udf(
                &raw_fields,
                udaf.as_ref(),
            )?;
            let coerced_args: Vec<ArrayRef> = precomputed_args
                .iter()
                .zip(&coerced_fields)
                .map(|(a, f)| {
                    if a.data_type() == f.data_type() {
                        Ok(Arc::clone(a))
                    } else {
                        Ok(compute::cast(a.as_ref(), f.data_type())?)
                    }
                })
                .collect::<Result<_>>()?;

            let take_indices: Vec<Option<u32>> =
                matching_row_indices(horizon, symbol_set, arg_physical_nav, num_part_rows)
                    .map(|opt| opt.map(|i| i as u32))
                    .collect();
            let return_field = Arc::new(Field::new(udaf.name(), result_type.clone(), true));
            let acc_args = AccumulatorArgs {
                return_field,
                schema: input_schema,
                ignore_nulls: false,
                order_bys: &[],
                is_reversed: false,
                name: udaf.name(),
                is_distinct: distinct,
                exprs: arg_exprs,
                expr_fields: &coerced_fields,
            };
            let mut acc = udaf.accumulator(acc_args)?;
            if !take_indices.is_empty() {
                let idx_arr = UInt32Array::from(take_indices);
                let mut taken: Vec<ArrayRef> = Vec::with_capacity(coerced_args.len());
                for a in &coerced_args {
                    taken.push(compute::take(a.as_ref(), &idx_arr, None)?);
                }
                if distinct && taken.len() == 1 {
                    let mut seen = HashSet::new();
                    let mut keep = Vec::new();
                    for r in 0..taken[0].len() {
                        if taken[0].is_valid(r) {
                            let sv = ScalarValue::try_from_array(taken[0].as_ref(), r)?;
                            if seen.insert(sv) {
                                keep.push(r as u32);
                            }
                        }
                    }
                    if !keep.is_empty() {
                        let k_arr = UInt32Array::from(keep);
                        taken[0] = compute::take(taken[0].as_ref(), &k_arr, None)?;
                        acc.update_batch(&taken)?;
                    }
                } else {
                    acc.update_batch(&taken)?;
                }
            }
            acc.evaluate()
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum EmitRow<'a> {
    Matched {
        found: &'a FoundMatch,
        step_pos: Option<usize>,
        source_row: usize,
        is_empty_match: bool,
    },
    Unmatched {
        source_row: usize,
    },
}

#[allow(clippy::too_many_arguments)]
pub fn build_partition_output_batch(
    partition_batch: &RecordBatch,
    events: &[PartitionEvent],
    program: &Program,
    partition_by_exprs: &[Arc<dyn PhysicalExpr>],
    partition_by_input_indices: &[Option<usize>],
    order_by_input_indices: &[Option<usize>],
    measures: &[PhysicalMeasure],
    rows_per_match: RowsPerMatch,
    output_schema: &SchemaRef,
) -> Result<Option<RecordBatch>> {
    let mut emitted: Vec<EmitRow<'_>> = Vec::new();
    for event in events {
        match event {
            PartitionEvent::Match(found) => {
                if rows_per_match.is_one_row() {
                    let is_empty = found.steps.is_empty();
                    let step_pos = if is_empty {
                        None
                    } else {
                        Some(found.steps.len() - 1)
                    };
                    emitted.push(EmitRow::Matched {
                        found,
                        step_pos,
                        source_row: found.start_row,
                        is_empty_match: is_empty,
                    });
                } else if found.steps.is_empty() {
                    if rows_per_match.show_empty_matches() {
                        emitted.push(EmitRow::Matched {
                            found,
                            step_pos: None,
                            source_row: found.start_row,
                            is_empty_match: true,
                        });
                    }
                } else {
                    let mut emitted_any = false;
                    for (idx, step) in found.steps.iter().enumerate() {
                        if !step.excluded {
                            emitted_any = true;
                            emitted.push(EmitRow::Matched {
                                found,
                                step_pos: Some(idx),
                                source_row: step.row_idx,
                                is_empty_match: false,
                            });
                        }
                    }
                    if !emitted_any && rows_per_match.show_empty_matches() {
                        emitted.push(EmitRow::Matched {
                            found,
                            step_pos: None,
                            source_row: found.start_row,
                            is_empty_match: true,
                        });
                    }
                }
            }
            PartitionEvent::UnmatchedRow { row_idx } => {
                emitted.push(EmitRow::Unmatched {
                    source_row: *row_idx,
                });
            }
        }
    }

    if emitted.is_empty() {
        return Ok(None);
    }

    let out_len = emitted.len();
    let mut columns: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    let mut included_input_indices: HashSet<usize> = HashSet::new();

    let source_indices = UInt32Array::from_iter_values(emitted.iter().map(|r| match r {
        EmitRow::Matched { source_row, .. } | EmitRow::Unmatched { source_row } => {
            *source_row as u32
        }
    }));

    for (expr, maybe_idx) in partition_by_exprs.iter().zip(partition_by_input_indices) {
        if let Some(col_idx) = maybe_idx {
            if included_input_indices.insert(*col_idx) {
                let col_arr = partition_batch.column(*col_idx);
                columns.push(compute::take(col_arr.as_ref(), &source_indices, None)?);
            }
        } else {
            let evaluated = expr
                .evaluate(partition_batch)?
                .into_array(partition_batch.num_rows())?;
            columns.push(compute::take(evaluated.as_ref(), &source_indices, None)?);
        }
    }

    if !rows_per_match.is_one_row() {
        for maybe_idx in order_by_input_indices {
            if let Some(col_idx) = maybe_idx
                && included_input_indices.insert(*col_idx)
            {
                let col_arr = partition_batch.column(*col_idx);
                columns.push(compute::take(col_arr.as_ref(), &source_indices, None)?);
            }
        }
    }

    let unmatched_mask = if rows_per_match.include_unmatched_rows() {
        Some(BooleanArray::from_iter(
            emitted
                .iter()
                .map(|r| Some(matches!(r, EmitRow::Unmatched { .. }))),
        ))
    } else {
        None
    };

    for measure in measures {
        let arr = evaluate_measure(
            partition_batch,
            &emitted,
            program,
            measure,
            unmatched_mask.as_ref(),
        )?;
        columns.push(arr);
    }

    if !rows_per_match.is_one_row() {
        for col_idx in 0..partition_batch.num_columns() {
            if included_input_indices.insert(col_idx) {
                let col_arr = partition_batch.column(col_idx);
                columns.push(compute::take(col_arr.as_ref(), &source_indices, None)?);
            }
        }
    }

    let batch = RecordBatch::try_new_with_options(
        Arc::clone(output_schema),
        columns,
        &RecordBatchOptions::new().with_row_count(Some(out_len)),
    )?;
    Ok(Some(batch))
}

fn evaluate_measure(
    partition_batch: &RecordBatch,
    emitted: &[EmitRow<'_>],
    program: &Program,
    measure: &PhysicalMeasure,
    unmatched_mask: Option<&BooleanArray>,
) -> Result<ArrayRef> {
    let out_len = emitted.len();
    let mut slot_arrays = Vec::with_capacity(measure.slots.len());
    for slot in &measure.slots {
        slot_arrays.push(evaluate_slot(partition_batch, emitted, program, slot)?);
    }

    let slot_batch = RecordBatch::try_new_with_options(
        Arc::clone(&measure.slot_schema),
        slot_arrays,
        &RecordBatchOptions::new().with_row_count(Some(out_len)),
    )?;

    let mut result = measure
        .outer_expr
        .evaluate(&slot_batch)?
        .into_array(out_len)?;

    if let Some(mask) = unmatched_mask {
        let nulls = new_null_array(result.data_type(), out_len);
        result = zip::zip(mask, &nulls, &result)?;
    }

    Ok(result)
}

fn evaluate_slot(
    partition_batch: &RecordBatch,
    emitted: &[EmitRow<'_>],
    program: &Program,
    slot: &PhysicalMeasureSlot,
) -> Result<ArrayRef> {
    let out_len = emitted.len();
    let num_part_rows = partition_batch.num_rows();

    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => {
            let mut builder = Int64Builder::with_capacity(out_len);
            for row in emitted {
                match row {
                    EmitRow::Matched { found, .. } => builder.append_value(found.match_number),
                    EmitRow::Unmatched { .. } => builder.append_null(),
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        PhysicalMeasureSlotKind::Classifier {
            symbol_set,
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            ..
        } => {
            let mut builder = StringBuilder::with_capacity(out_len, out_len * 8);
            for row in emitted {
                match row {
                    EmitRow::Unmatched { .. } => builder.append_null(),
                    EmitRow::Matched {
                        found,
                        step_pos,
                        is_empty_match,
                        ..
                    } => {
                        if *is_empty_match {
                            builder.append_null();
                        } else {
                            let horizon = horizon_steps(found, *step_pos, slot.semantics);
                            match eval_classifier_on_horizon(
                                horizon,
                                &found.steps,
                                symbol_set.as_ref(),
                                *logical_nav,
                                *logical_offset as usize,
                                *physical_nav,
                                *physical_offset as usize,
                                program,
                            ) {
                                Some(name) => builder.append_value(name),
                                None => builder.append_null(),
                            }
                        }
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
            let evaluated = inner_expr
                .evaluate(partition_batch)?
                .into_array(num_part_rows)?;
            let indices = UInt32Array::from_iter(emitted.iter().map(|row| match row {
                EmitRow::Matched {
                    source_row,
                    is_empty_match,
                    ..
                } => {
                    if *is_empty_match {
                        None
                    } else {
                        Some(*source_row as u32)
                    }
                }
                EmitRow::Unmatched { .. } => None,
            }));
            Ok(compute::take(evaluated.as_ref(), &indices, None)?)
        }
        PhysicalMeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            symbol_set,
            inner_expr,
            ..
        } => {
            let evaluated = inner_expr
                .evaluate(partition_batch)?
                .into_array(num_part_rows)?;
            let indices = UInt32Array::from_iter(emitted.iter().map(|row| match row {
                EmitRow::Unmatched { .. } => None,
                EmitRow::Matched {
                    found,
                    step_pos,
                    is_empty_match,
                    ..
                } => {
                    if *is_empty_match {
                        return None;
                    }
                    let horizon = horizon_steps(found, *step_pos, slot.semantics);
                    let base_row = navigate_logical_row(
                        horizon,
                        *logical_nav,
                        *logical_offset as usize,
                        symbol_set.as_ref(),
                    )?;
                    let final_row = navigate_physical(
                        base_row,
                        *physical_nav,
                        *physical_offset as usize,
                        num_part_rows,
                    )?;
                    Some(final_row as u32)
                }
            }));
            Ok(compute::take(evaluated.as_ref(), &indices, None)?)
        }
        PhysicalMeasureSlotKind::Agg {
            agg_kind,
            symbol_set,
            args,
            distinct,
            arg_physical_nav,
            uses_match_metadata,
            ..
        } => {
            if *uses_match_metadata {
                let mut scalars = Vec::with_capacity(out_len);
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => {
                            scalars.push(ScalarValue::try_from(&slot.data_type)?);
                        }
                        EmitRow::Matched {
                            found,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                if matches!(agg_kind, AggKind::Count) {
                                    scalars.push(ScalarValue::Int64(Some(0)));
                                } else {
                                    scalars.push(ScalarValue::try_from(&slot.data_type)?);
                                }
                            } else {
                                let horizon = horizon_steps(found, *step_pos, slot.semantics);
                                scalars.push(eval_metadata_agg_scalar(
                                    partition_batch,
                                    horizon,
                                    found.match_number,
                                    program,
                                    agg_kind,
                                    symbol_set.as_ref(),
                                    args,
                                    *distinct,
                                    *arg_physical_nav,
                                    &slot.data_type,
                                )?);
                            }
                        }
                    }
                }
                ScalarValue::iter_to_array(scalars)
            } else {
                evaluate_agg_slot(
                    partition_batch,
                    emitted,
                    slot.semantics,
                    agg_kind,
                    symbol_set.as_ref(),
                    args,
                    *distinct,
                    *arg_physical_nav,
                    &slot.data_type,
                )
            }
        }
    }
}

#[inline]
fn horizon_steps(
    found: &FoundMatch,
    step_pos: Option<usize>,
    semantics: EvalSemantics,
) -> &[MatchStep] {
    match semantics {
        EvalSemantics::Final => &found.steps,
        EvalSemantics::Running => match step_pos {
            Some(idx) => &found.steps[..=idx],
            None => &[],
        },
    }
}

fn navigate_logical_row(
    horizon: &[MatchStep],
    logical_nav: Option<LogicalNavKind>,
    offset: usize,
    symbol_set: Option<&SymbolSet>,
) -> Option<usize> {
    let nav = logical_nav.unwrap_or(LogicalNavKind::Last);
    match symbol_set {
        Some(set) => match nav {
            LogicalNavKind::First => horizon
                .iter()
                .filter(|s| set.contains(s.sym))
                .nth(offset)
                .map(|s| s.row_idx),
            LogicalNavKind::Last => horizon
                .iter()
                .rev()
                .filter(|s| set.contains(s.sym))
                .nth(offset)
                .map(|s| s.row_idx),
        },
        None => match nav {
            LogicalNavKind::First => horizon.get(offset).map(|s| s.row_idx),
            LogicalNavKind::Last => {
                if offset < horizon.len() {
                    Some(horizon[horizon.len() - 1 - offset].row_idx)
                } else {
                    None
                }
            }
        },
    }
}

fn navigate_physical(
    base_row: usize,
    physical_nav: Option<PhysicalNavKind>,
    offset: usize,
    num_part_rows: usize,
) -> Option<usize> {
    match physical_nav {
        None => Some(base_row),
        Some(PhysicalNavKind::Prev) => base_row.checked_sub(offset),
        Some(PhysicalNavKind::Next) => {
            let target = base_row.checked_add(offset)?;
            if target < num_part_rows {
                Some(target)
            } else {
                None
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn evaluate_agg_slot(
    partition_batch: &RecordBatch,
    emitted: &[EmitRow<'_>],
    semantics: EvalSemantics,
    agg_kind: &AggKind,
    symbol_set: Option<&SymbolSet>,
    arg_exprs: &[Arc<dyn PhysicalExpr>],
    distinct: bool,
    arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    result_type: &DataType,
) -> Result<ArrayRef> {
    let out_len = emitted.len();
    let num_part_rows = partition_batch.num_rows();

    match agg_kind {
        AggKind::CountStar | AggKind::CountSymbol => {
            let mut builder = Int64Builder::with_capacity(out_len);
            for row in emitted {
                match row {
                    EmitRow::Unmatched { .. } => builder.append_null(),
                    EmitRow::Matched {
                        found,
                        step_pos,
                        is_empty_match,
                        ..
                    } => {
                        if *is_empty_match {
                            builder.append_value(0);
                        } else {
                            let horizon = horizon_steps(found, *step_pos, semantics);
                            let count =
                                matching_row_indices(horizon, symbol_set, None, num_part_rows)
                                    .count();
                            builder.append_value(count as i64);
                        }
                    }
                }
            }
            return Ok(Arc::new(builder.finish()));
        }
        _ => {}
    }

    let mut prepared_args = Vec::with_capacity(arg_exprs.len());
    for (i, expr) in arg_exprs.iter().enumerate() {
        let raw = expr.evaluate(partition_batch)?.into_array(num_part_rows)?;
        let prepared = if i == 0 {
            match agg_kind {
                AggKind::Avg => compute::cast(raw.as_ref(), &DataType::Float64)?,
                AggKind::Sum if raw.data_type() != result_type => {
                    compute::cast(raw.as_ref(), result_type)?
                }
                _ => raw,
            }
        } else {
            raw
        };
        prepared_args.push(prepared);
    }

    // Fast path for common non-distinct COUNT and AVG without physical shift.
    if !distinct && arg_physical_nav.is_none() {
        match agg_kind {
            AggKind::Count => {
                let evaluated = &prepared_args[0];
                let mut builder = Int64Builder::with_capacity(out_len);
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => builder.append_null(),
                        EmitRow::Matched {
                            found,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                builder.append_value(0);
                            } else {
                                let horizon = horizon_steps(found, *step_pos, semantics);
                                let count =
                                    matching_row_indices(horizon, symbol_set, None, num_part_rows)
                                        .flatten()
                                        .filter(|&idx| evaluated.is_valid(idx))
                                        .count();
                                builder.append_value(count as i64);
                            }
                        }
                    }
                }
                return Ok(Arc::new(builder.finish()));
            }
            AggKind::Avg => {
                let f64_arr = prepared_args[0]
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::exec_datafusion_err!("Expected Float64Array for AVG")
                    })?;
                let mut builder = Float64Builder::with_capacity(out_len);
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => builder.append_null(),
                        EmitRow::Matched {
                            found,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                builder.append_null();
                            } else {
                                let horizon = horizon_steps(found, *step_pos, semantics);
                                let mut sum = 0.0f64;
                                let mut count = 0u64;
                                for idx in
                                    matching_row_indices(horizon, symbol_set, None, num_part_rows)
                                        .flatten()
                                {
                                    if f64_arr.is_valid(idx) {
                                        sum += f64_arr.value(idx);
                                        count += 1;
                                    }
                                }
                                if count == 0 {
                                    builder.append_null();
                                } else {
                                    builder.append_value(sum / count as f64);
                                }
                            }
                        }
                    }
                }
                return Ok(Arc::new(builder.finish()));
            }
            _ => {}
        }
    }

    let scalar_dt = match agg_kind {
        AggKind::Count => &DataType::Int64,
        AggKind::Avg => &DataType::Float64,
        AggKind::Sum | AggKind::Udaf(_) => result_type,
        AggKind::Min | AggKind::Max => prepared_args[0].data_type(),
        AggKind::CountStar | AggKind::CountSymbol => unreachable!(),
    };
    let input_schema = partition_batch.schema();
    let mut scalars = Vec::with_capacity(out_len);
    for row in emitted {
        match row {
            EmitRow::Unmatched { .. } => {
                scalars.push(ScalarValue::try_from(scalar_dt)?);
            }
            EmitRow::Matched {
                found,
                step_pos,
                is_empty_match,
                ..
            } => {
                if *is_empty_match {
                    if matches!(agg_kind, AggKind::Count) {
                        scalars.push(ScalarValue::Int64(Some(0)));
                    } else {
                        scalars.push(ScalarValue::try_from(scalar_dt)?);
                    }
                } else {
                    let horizon = horizon_steps(found, *step_pos, semantics);
                    scalars.push(eval_single_agg_scalar(
                        horizon,
                        num_part_rows,
                        input_schema.as_ref(),
                        agg_kind,
                        symbol_set,
                        arg_exprs,
                        distinct,
                        arg_physical_nav,
                        &prepared_args,
                        result_type,
                    )?);
                }
            }
        }
    }
    ScalarValue::iter_to_array(scalars)
}

fn matching_row_indices<'a>(
    horizon: &'a [MatchStep],
    symbol_set: Option<&'a SymbolSet>,
    arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    num_part_rows: usize,
) -> impl Iterator<Item = Option<usize>> + 'a {
    horizon.iter().filter_map(move |s| {
        if let Some(set) = symbol_set
            && !set.contains(s.sym)
        {
            return None;
        }
        let shifted = match arg_physical_nav {
            None => Some(s.row_idx),
            Some((kind, off)) => {
                navigate_physical(s.row_idx, Some(kind), off as usize, num_part_rows)
            }
        };
        Some(shifted)
    })
}

fn sum_indices(
    arr: &ArrayRef,
    indices: impl Iterator<Item = usize>,
    result_type: &DataType,
) -> Result<ScalarValue> {
    match result_type {
        DataType::Int64 => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected Int64Array"))?;
            let mut acc: Option<i64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    acc = Some(acc.unwrap_or(0).wrapping_add(a.value(idx)));
                }
            }
            Ok(ScalarValue::Int64(acc))
        }
        DataType::UInt64 => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected UInt64Array"))?;
            let mut acc: Option<u64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    acc = Some(acc.unwrap_or(0).wrapping_add(a.value(idx)));
                }
            }
            Ok(ScalarValue::UInt64(acc))
        }
        DataType::Float64 => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected Float64Array"))?;
            let mut acc: Option<f64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    acc = Some(acc.unwrap_or(0.0) + a.value(idx));
                }
            }
            Ok(ScalarValue::Float64(acc))
        }
        DataType::Decimal128(p, s) => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::Decimal128Array>()
                .ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!("Expected Decimal128Array")
                })?;
            let mut acc: Option<i128> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    acc = Some(acc.unwrap_or(0) + a.value(idx));
                }
            }
            Ok(ScalarValue::Decimal128(acc, *p, *s))
        }
        other => exec_err!("Unsupported SUM type in MATCH_RECOGNIZE: {other}"),
    }
}

fn sum_indices_distinct(
    arr: &ArrayRef,
    indices: impl Iterator<Item = usize>,
    result_type: &DataType,
) -> Result<ScalarValue> {
    match result_type {
        DataType::Int64 => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected Int64Array"))?;
            let mut seen = HashSet::new();
            let mut acc: Option<i64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    let v = a.value(idx);
                    if seen.insert(v) {
                        acc = Some(acc.unwrap_or(0).wrapping_add(v));
                    }
                }
            }
            Ok(ScalarValue::Int64(acc))
        }
        DataType::UInt64 => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
                .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected UInt64Array"))?;
            let mut seen = HashSet::new();
            let mut acc: Option<u64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    let v = a.value(idx);
                    if seen.insert(v) {
                        acc = Some(acc.unwrap_or(0).wrapping_add(v));
                    }
                }
            }
            Ok(ScalarValue::UInt64(acc))
        }
        DataType::Float64 => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| datafusion_common::exec_datafusion_err!("Expected Float64Array"))?;
            let mut seen = HashSet::new();
            let mut acc: Option<f64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    let v = a.value(idx);
                    if seen.insert(v.to_bits()) {
                        acc = Some(acc.unwrap_or(0.0) + v);
                    }
                }
            }
            Ok(ScalarValue::Float64(acc))
        }
        DataType::Decimal128(p, s) => {
            let a = arr
                .as_any()
                .downcast_ref::<arrow::array::Decimal128Array>()
                .ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!("Expected Decimal128Array")
                })?;
            let mut seen = HashSet::new();
            let mut acc: Option<i128> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    let v = a.value(idx);
                    if seen.insert(v) {
                        acc = Some(acc.unwrap_or(0) + v);
                    }
                }
            }
            Ok(ScalarValue::Decimal128(acc, *p, *s))
        }
        other => exec_err!("Unsupported SUM(DISTINCT) type in MATCH_RECOGNIZE: {other}"),
    }
}

pub fn build_physical_slot_schema(slots: &[PhysicalMeasureSlot]) -> SchemaRef {
    let fields: Vec<Field> = slots
        .iter()
        .map(|s| Field::new(&s.slot_name, s.data_type.clone(), s.nullable))
        .collect();
    Arc::new(Schema::new(fields))
}
