// SPDX-License-Identifier: Apache-2.0

//! Physical evaluation of `MATCH_RECOGNIZE` `MEASURES`, stateful `DEFINE` predicates,
//! `SUBSET` union variables, `DISTINCT` aggregates, and output batch construction.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Builder, Int64Builder, RecordBatch, RecordBatchOptions,
    StringBuilder, UInt32Array, new_null_array,
};
use arrow::compute::{self, kernels::zip};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{DataFusionError, Result, ScalarValue, exec_err};
use datafusion_physical_expr::PhysicalExpr;

use crate::ast::{AggKind, EvalSemantics, LogicalNavKind, PhysicalNavKind, RowsPerMatch};
use crate::matcher::{FoundMatch, MatchStep, PartitionEvent, StatefulPredicateEvaluator};
use crate::pattern::{Program, SymbolId, SymbolSet};

/// Physical counterpart of [`crate::ast::MeasureSlotKind`].
#[derive(Debug, Clone)]
pub enum PhysicalMeasureSlotKind {
    MatchNumber,
    Classifier {
        symbol: Option<String>,
        symbol_set: Option<SymbolSet>,
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
        inner_expr: Option<Arc<dyn PhysicalExpr>>,
        distinct: bool,
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

/// Precomputed partition-level evaluation state for one stateful `DEFINE` symbol.
struct CompiledStatefulSymbol {
    outer_expr: Arc<dyn PhysicalExpr>,
    slot_schema: SchemaRef,
    slots: Vec<PhysicalMeasureSlot>,
    /// Pre-evaluated inner Arrow arrays (`precomputed_inners[slot_idx]`) evaluated ONCE per
    /// partition in vectorized Arrow so candidate transitions do zero inner-expr work.
    precomputed_inners: Vec<Option<ArrayRef>>,
}

/// Evaluates Tier 2 stateful `DEFINE` predicates against a partition's precomputed slot arrays.
pub struct PartitionStatefulEvaluator<'a> {
    program: &'a Program,
    num_part_rows: usize,
    by_symbol: HashMap<SymbolId, CompiledStatefulSymbol>,
}

/// Borrowed specification for one stateful `DEFINE` symbol passed to [`PartitionStatefulEvaluator`].
pub struct StatefulDefineInput<'a> {
    pub sym_id: SymbolId,
    pub outer_expr: &'a Arc<dyn PhysicalExpr>,
    pub slot_schema: &'a SchemaRef,
    pub slots: &'a [PhysicalMeasureSlot],
}

impl<'a> PartitionStatefulEvaluator<'a> {
    /// Pre-evaluates all stateful `DEFINE` inner expressions once for `partition_batch`.
    pub fn try_new(
        partition_batch: &RecordBatch,
        program: &'a Program,
        stateful_defs: &[StatefulDefineInput<'_>],
    ) -> Result<Self> {
        let num_part_rows = partition_batch.num_rows();
        let mut by_symbol = HashMap::with_capacity(stateful_defs.len());

        for def in stateful_defs {
            let sym_id = def.sym_id;
            let outer_expr = def.outer_expr;
            let slot_schema = def.slot_schema;
            let slots = def.slots;
            let mut precomputed_inners = Vec::with_capacity(slots.len());
            for slot in slots {
                let arr = match &slot.kind {
                    PhysicalMeasureSlotKind::Nav { inner_expr, .. }
                    | PhysicalMeasureSlotKind::CurrentRow { inner_expr } => Some(
                        inner_expr
                            .evaluate(partition_batch)?
                            .into_array(num_part_rows)?,
                    ),
                    PhysicalMeasureSlotKind::Agg {
                        agg_kind,
                        inner_expr: Some(e),
                        ..
                    } => {
                        let raw = e.evaluate(partition_batch)?.into_array(num_part_rows)?;
                        match agg_kind {
                            AggKind::Avg => Some(compute::cast(raw.as_ref(), &DataType::Float64)?),
                            AggKind::Sum if raw.data_type() != &slot.data_type => {
                                Some(compute::cast(raw.as_ref(), &slot.data_type)?)
                            }
                            _ => Some(raw),
                        }
                    }
                    PhysicalMeasureSlotKind::MatchNumber
                    | PhysicalMeasureSlotKind::Classifier { .. }
                    | PhysicalMeasureSlotKind::Agg {
                        inner_expr: None, ..
                    } => None,
                };
                precomputed_inners.push(arr);
            }
            by_symbol.insert(
                sym_id,
                CompiledStatefulSymbol {
                    outer_expr: Arc::clone(outer_expr),
                    slot_schema: Arc::clone(slot_schema),
                    slots: slots.to_vec(),
                    precomputed_inners,
                },
            );
        }

        Ok(Self {
            program,
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
        for (slot, pre_arr) in compiled.slots.iter().zip(&compiled.precomputed_inners) {
            let scalar = eval_single_slot_scalar(
                tentative_steps,
                candidate_match_number,
                current_row,
                self.num_part_rows,
                self.program,
                slot,
                pre_arr.as_ref(),
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

/// Evaluates one slot on a single tentative match horizon (used by Tier 2 stateful `DEFINE`).
fn eval_single_slot_scalar(
    horizon: &[MatchStep],
    match_number: i64,
    current_row: usize,
    num_part_rows: usize,
    program: &Program,
    slot: &PhysicalMeasureSlot,
    precomputed_inner: Option<&ArrayRef>,
) -> Result<ScalarValue> {
    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => Ok(ScalarValue::Int64(Some(match_number))),
        PhysicalMeasureSlotKind::Classifier { symbol_set, .. } => {
            let target_step = match symbol_set {
                Some(set) => horizon.iter().rev().find(|s| set.contains(s.sym)),
                None => horizon.last(),
            };
            Ok(ScalarValue::Utf8(target_step.map(|s| {
                program.id_to_symbol[s.sym as usize].to_ascii_uppercase()
            })))
        }
        PhysicalMeasureSlotKind::CurrentRow { .. } => {
            let arr = precomputed_inner.expect("CurrentRow has precomputed array");
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
            let arr = precomputed_inner.expect("Nav has precomputed array");
            let Some(base_row) = navigate_logical(
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
            distinct,
            ..
        } => eval_single_agg_scalar(
            horizon,
            *agg_kind,
            symbol_set.as_ref(),
            *distinct,
            precomputed_inner,
            &slot.data_type,
        ),
    }
}

/// Evaluates one aggregate slot over a single horizon slice (used by both stateful `DEFINE` and
/// `DISTINCT` measure aggregates).
fn eval_single_agg_scalar(
    horizon: &[MatchStep],
    agg_kind: AggKind,
    symbol_set: Option<&SymbolSet>,
    distinct: bool,
    precomputed_inner: Option<&ArrayRef>,
    result_type: &DataType,
) -> Result<ScalarValue> {
    match agg_kind {
        AggKind::CountStar | AggKind::CountSymbol => {
            let count = matching_row_indices(horizon, symbol_set).count();
            Ok(ScalarValue::Int64(Some(count as i64)))
        }
        AggKind::Count => {
            let arr = precomputed_inner.expect("COUNT(expr) has precomputed array");
            if distinct {
                let mut seen: HashSet<ScalarValue> = HashSet::new();
                for idx in matching_row_indices(horizon, symbol_set) {
                    if arr.is_valid(idx) {
                        seen.insert(ScalarValue::try_from_array(arr.as_ref(), idx)?);
                    }
                }
                Ok(ScalarValue::Int64(Some(seen.len() as i64)))
            } else {
                let count = matching_row_indices(horizon, symbol_set)
                    .filter(|&idx| arr.is_valid(idx))
                    .count();
                Ok(ScalarValue::Int64(Some(count as i64)))
            }
        }
        AggKind::Avg => {
            let arr = precomputed_inner.expect("AVG has precomputed Float64 array");
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
                for idx in matching_row_indices(horizon, symbol_set) {
                    if f64_arr.is_valid(idx) {
                        let v = f64_arr.value(idx);
                        if seen.insert(v.to_bits()) {
                            sum += v;
                            count += 1;
                        }
                    }
                }
            } else {
                for idx in matching_row_indices(horizon, symbol_set) {
                    if f64_arr.is_valid(idx) {
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
            let arr = precomputed_inner.expect("SUM has precomputed casted array");
            if distinct {
                sum_indices_distinct(arr, matching_row_indices(horizon, symbol_set), result_type)
            } else {
                sum_indices(arr, matching_row_indices(horizon, symbol_set), result_type)
            }
        }
        AggKind::Min | AggKind::Max => {
            let arr = precomputed_inner.expect("MIN/MAX has precomputed array");
            let is_min = matches!(agg_kind, AggKind::Min);
            let mut best: Option<ScalarValue> = None;
            for idx in matching_row_indices(horizon, symbol_set) {
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
    }
}

/// Describes one emitted row in the output batch of a partition.
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

/// Evaluates all output columns for one partition given its concatenated `RecordBatch` and
/// matched [`PartitionEvent`]s.
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

    // Under SQL:2016 / Trino, even for empty matches in ALL ROWS SHOW EMPTY MATCHES,
    // PARTITION BY, ORDER BY, and remaining input columns carry the starting row's values
    // (while MEASURES evaluate over the empty match).
    let source_indices = UInt32Array::from_iter_values(emitted.iter().map(|r| match r {
        EmitRow::Matched { source_row, .. } | EmitRow::Unmatched { source_row } => {
            *source_row as u32
        }
    }));

    // 1. PARTITION BY output columns.
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

    // 2. ORDER BY output columns (ALL ROWS only).
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

    // 3. MEASURES output columns.
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

    // 4. Remaining input columns (ALL ROWS only).
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
        PhysicalMeasureSlotKind::Classifier { symbol_set, .. } => {
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
                            let target_step = match symbol_set {
                                Some(set) => horizon.iter().rev().find(|s| set.contains(s.sym)),
                                None => horizon.last(),
                            };
                            match target_step {
                                Some(step) => {
                                    let name = &program.id_to_symbol[step.sym as usize];
                                    builder.append_value(name.to_ascii_uppercase());
                                }
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
                    let base_row = navigate_logical(
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
            inner_expr,
            distinct,
            ..
        } => evaluate_agg_slot(
            partition_batch,
            emitted,
            slot.semantics,
            *agg_kind,
            symbol_set.as_ref(),
            *distinct,
            inner_expr.as_ref(),
            &slot.data_type,
        ),
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

fn navigate_logical(
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
    agg_kind: AggKind,
    symbol_set: Option<&SymbolSet>,
    distinct: bool,
    inner_expr: Option<&Arc<dyn PhysicalExpr>>,
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
                            let count = matching_row_indices(horizon, symbol_set).count();
                            builder.append_value(count as i64);
                        }
                    }
                }
            }
            return Ok(Arc::new(builder.finish()));
        }
        _ => {}
    }

    let expr = inner_expr.ok_or_else(|| {
        datafusion_common::exec_datafusion_err!("Aggregate {agg_kind:?} requires an expression")
    })?;
    let evaluated = expr.evaluate(partition_batch)?.into_array(num_part_rows)?;

    match agg_kind {
        AggKind::Count if !distinct => {
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
                            let count = matching_row_indices(horizon, symbol_set)
                                .filter(|&idx| evaluated.is_valid(idx))
                                .count();
                            builder.append_value(count as i64);
                        }
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        AggKind::Avg if !distinct => {
            let float_arr = compute::cast(evaluated.as_ref(), &DataType::Float64)?;
            let f64_arr = float_arr
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!("Failed to cast AVG input to Float64")
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
                            for idx in matching_row_indices(horizon, symbol_set) {
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
            Ok(Arc::new(builder.finish()))
        }
        _ => {
            let prepared = match agg_kind {
                AggKind::Avg => compute::cast(evaluated.as_ref(), &DataType::Float64)?,
                AggKind::Sum if evaluated.data_type() != result_type => {
                    compute::cast(evaluated.as_ref(), result_type)?
                }
                _ => evaluated,
            };
            let scalar_dt = match agg_kind {
                AggKind::Count => &DataType::Int64,
                AggKind::Avg => &DataType::Float64,
                AggKind::Sum => result_type,
                AggKind::Min | AggKind::Max => prepared.data_type(),
                AggKind::CountStar | AggKind::CountSymbol => unreachable!(),
            };
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
                                agg_kind,
                                symbol_set,
                                distinct,
                                Some(&prepared),
                                result_type,
                            )?);
                        }
                    }
                }
            }
            ScalarValue::iter_to_array(scalars)
        }
    }
}

fn matching_row_indices<'a>(
    horizon: &'a [MatchStep],
    symbol_set: Option<&'a SymbolSet>,
) -> impl Iterator<Item = usize> + 'a {
    horizon.iter().filter_map(move |s| match symbol_set {
        Some(set) if !set.contains(s.sym) => None,
        _ => Some(s.row_idx),
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

/// Builds a [`SchemaRef`] for a measure or stateful define's slots.
pub fn build_physical_slot_schema(slots: &[PhysicalMeasureSlot]) -> SchemaRef {
    let fields: Vec<Field> = slots
        .iter()
        .map(|s| Field::new(&s.slot_name, s.data_type.clone(), s.nullable))
        .collect();
    Arc::new(Schema::new(fields))
}
