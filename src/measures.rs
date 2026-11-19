// SPDX-License-Identifier: Apache-2.0

//! Physical evaluation of `MATCH_RECOGNIZE` `MEASURES` and output batch construction.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Builder, Int64Builder, RecordBatch, RecordBatchOptions,
    StringBuilder, UInt32Array, new_null_array,
};
use arrow::compute::{self, kernels::zip};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use datafusion_common::{Result, ScalarValue, exec_err};
use datafusion_physical_expr::PhysicalExpr;

use crate::ast::{AggKind, EvalSemantics, LogicalNavKind, PhysicalNavKind, RowsPerMatch};
use crate::matcher::{FoundMatch, MatchStep, PartitionEvent};
use crate::pattern::{Program, SymbolId};

/// Physical counterpart of [`crate::ast::MeasureSlotKind`].
#[derive(Debug, Clone)]
pub enum PhysicalMeasureSlotKind {
    MatchNumber,
    Classifier {
        symbol: Option<String>,
        symbol_id: Option<SymbolId>,
    },
    Nav {
        logical_nav: Option<LogicalNavKind>,
        logical_offset: u32,
        physical_nav: Option<PhysicalNavKind>,
        physical_offset: u32,
        symbol: Option<String>,
        symbol_id: Option<SymbolId>,
        inner_expr: Arc<dyn PhysicalExpr>,
    },
    Agg {
        agg_kind: AggKind,
        symbol: Option<String>,
        symbol_id: Option<SymbolId>,
        inner_expr: Option<Arc<dyn PhysicalExpr>>,
    },
    CurrentRow {
        inner_expr: Arc<dyn PhysicalExpr>,
    },
}

/// A compiled measure leaf slot ready for physical execution.
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

/// Describes one emitted row in the output batch of a partition.
#[derive(Debug, Clone, Copy)]
enum EmitRow<'a> {
    /// Emitted from a match (either `ONE ROW PER MATCH` or one non-excluded step of `ALL ROWS`).
    Matched {
        found: &'a FoundMatch,
        /// Inclusive index into `found.steps` for `RUNNING` evaluation (`None` for empty match).
        step_pos: Option<usize>,
        /// Representative partition row index for passthrough / partition columns.
        source_row: usize,
        /// True when this emitted row is an empty match (or all-excluded match).
        is_empty_match: bool,
    },
    /// Emitted only under `ALL ROWS PER MATCH WITH UNMATCHED ROWS`.
    Unmatched { source_row: usize },
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
    // 1. Expand PartitionEvents into concrete emitted row descriptors.
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
                    // If all rows of a non-empty match were excluded via {- ... -}, treat like an
                    // empty output match per SQL:2016 / Trino.
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

    // Precompute source row take indices. Under SQL:2016 / Trino, even for empty matches in
    // ALL ROWS PER MATCH SHOW EMPTY MATCHES, PARTITION BY, ORDER BY, and remaining input columns
    // carry the starting row's values (while MEASURES evaluate over the empty match).
    let source_indices = UInt32Array::from_iter_values(emitted.iter().map(|r| match r {
        EmitRow::Matched { source_row, .. } | EmitRow::Unmatched { source_row } => {
            *source_row as u32
        }
    }));

    // 2. PARTITION BY output columns.
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

    // 3. ORDER BY output columns (ALL ROWS only).
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

    // 4. MEASURES output columns.
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

    // 5. Remaining input columns (ALL ROWS only).
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

    // Under WITH UNMATCHED ROWS, every MEASURE column is NULL on unmatched rows (even constants).
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
        PhysicalMeasureSlotKind::Classifier { symbol_id, .. } => {
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
                            let target_step = match symbol_id {
                                Some(sym) => horizon.iter().rev().find(|s| s.sym == *sym),
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
            symbol_id,
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
                        *symbol_id,
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
            symbol_id,
            inner_expr,
            ..
        } => evaluate_agg_slot(
            partition_batch,
            emitted,
            slot.semantics,
            *agg_kind,
            *symbol_id,
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
    symbol_id: Option<SymbolId>,
) -> Option<usize> {
    let nav = logical_nav.unwrap_or(LogicalNavKind::Last);
    match symbol_id {
        Some(sym) => match nav {
            LogicalNavKind::First => horizon
                .iter()
                .filter(|s| s.sym == sym)
                .nth(offset)
                .map(|s| s.row_idx),
            LogicalNavKind::Last => horizon
                .iter()
                .rev()
                .filter(|s| s.sym == sym)
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

fn evaluate_agg_slot(
    partition_batch: &RecordBatch,
    emitted: &[EmitRow<'_>],
    semantics: EvalSemantics,
    agg_kind: AggKind,
    symbol_id: Option<SymbolId>,
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
                            let count = match symbol_id {
                                Some(sym) => horizon.iter().filter(|s| s.sym == sym).count(),
                                None => horizon.len(),
                            };
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
        AggKind::Count => {
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
                            let count = matching_row_indices(horizon, symbol_id)
                                .filter(|&idx| evaluated.is_valid(idx))
                                .count();
                            builder.append_value(count as i64);
                        }
                    }
                }
            }
            Ok(Arc::new(builder.finish()))
        }
        AggKind::Avg => {
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
                            for idx in matching_row_indices(horizon, symbol_id) {
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
        AggKind::Sum => {
            let mut scalars = Vec::with_capacity(out_len);
            let casted = if evaluated.data_type() != result_type {
                compute::cast(evaluated.as_ref(), result_type)?
            } else {
                evaluated
            };
            for row in emitted {
                match row {
                    EmitRow::Unmatched { .. } => {
                        scalars.push(ScalarValue::try_from(result_type)?);
                    }
                    EmitRow::Matched {
                        found,
                        step_pos,
                        is_empty_match,
                        ..
                    } => {
                        if *is_empty_match {
                            scalars.push(ScalarValue::try_from(result_type)?);
                        } else {
                            let horizon = horizon_steps(found, *step_pos, semantics);
                            let scalar = sum_indices(
                                &casted,
                                matching_row_indices(horizon, symbol_id),
                                result_type,
                            )?;
                            scalars.push(scalar);
                        }
                    }
                }
            }
            ScalarValue::iter_to_array(scalars)
        }
        AggKind::Min | AggKind::Max => {
            let is_min = matches!(agg_kind, AggKind::Min);
            let mut scalars = Vec::with_capacity(out_len);
            for row in emitted {
                match row {
                    EmitRow::Unmatched { .. } => {
                        scalars.push(ScalarValue::try_from(evaluated.data_type())?);
                    }
                    EmitRow::Matched {
                        found,
                        step_pos,
                        is_empty_match,
                        ..
                    } => {
                        if *is_empty_match {
                            scalars.push(ScalarValue::try_from(evaluated.data_type())?);
                        } else {
                            let horizon = horizon_steps(found, *step_pos, semantics);
                            let mut best: Option<ScalarValue> = None;
                            for idx in matching_row_indices(horizon, symbol_id) {
                                if !evaluated.is_valid(idx) {
                                    continue;
                                }
                                let v = ScalarValue::try_from_array(evaluated.as_ref(), idx)?;
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
                                Some(v) => scalars.push(v),
                                None => scalars.push(ScalarValue::try_from(evaluated.data_type())?),
                            }
                        }
                    }
                }
            }
            ScalarValue::iter_to_array(scalars)
        }
        AggKind::CountStar | AggKind::CountSymbol => unreachable!(),
    }
}

fn matching_row_indices<'a>(
    horizon: &'a [MatchStep],
    symbol_id: Option<SymbolId>,
) -> impl Iterator<Item = usize> + 'a {
    horizon.iter().filter_map(move |s| match symbol_id {
        Some(sym) if s.sym != sym => None,
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

/// Builds a [`SchemaRef`] for a measure's slots.
pub fn build_physical_slot_schema(slots: &[PhysicalMeasureSlot]) -> SchemaRef {
    let fields: Vec<Field> = slots
        .iter()
        .map(|s| Field::new(&s.slot_name, s.data_type.clone(), s.nullable))
        .collect();
    Arc::new(Schema::new(fields))
}
