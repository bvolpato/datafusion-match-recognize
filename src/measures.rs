// SPDX-License-Identifier: Apache-2.0

//! Physical evaluation of `MATCH_RECOGNIZE` `MEASURES`, stateful `DEFINE` predicates,
//! `SUBSET` union variables, `DISTINCT` aggregates, navigation over `CLASSIFIER()`,
//! `PREV`/`NEXT` inside aggregates, and arbitrary DataFusion [`datafusion_expr::AggregateUDF`]s
//! via [`datafusion_expr::Accumulator`].

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Float64Builder, Int32Array, Int64Array,
    Int64Builder, RecordBatch, RecordBatchOptions, StringBuilder, UInt32Array,
    new_null_array,
};
use arrow::compute::{self, kernels::zip};
use arrow_schema::{DataType, Field, FieldRef, Schema, SchemaRef};
use datafusion_common::{
    DataFusionError, HashSet, Result, ScalarValue, exec_err, internal_datafusion_err,
};
use datafusion_expr::Operator;
use datafusion_expr::function::AccumulatorArgs;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
};

use crate::ast::{AggKind, EvalSemantics, LogicalNavKind, PhysicalNavKind, RowsPerMatch};
use crate::logical::build_agg_metadata_arrow_schema;
use crate::matcher::{
    FlatMatchPool, FlatPartitionEvent, MatchStep, NO_HISTORY, PartitionEvent,
    StatefulPredicateEvaluator,
};
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
        uses_match_metadata: bool,
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

/// Evaluates `expr` on `batch`, avoiding `PhysicalExpr::evaluate` overhead for direct [`Column`] refs.
#[inline]
pub(crate) fn eval_expr_or_col(
    expr: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
    num_rows: usize,
) -> Result<ArrayRef> {
    if let Some(col) = expr.downcast_ref::<Column>() {
        return Ok(Arc::clone(batch.column(col.index())));
    }
    expr.evaluate(batch)?.into_array(num_rows)
}

/// Returns true when a stateful `DEFINE` symbol depends ONLY on the current row and
/// self-anchored `PREV`/`NEXT` physical shifts (no match history, no aggregates, no metadata).
///
/// Because `RUNNING LAST(self)` on a tentative transition `(pos, self)` is always `pos`,
/// `PREV(self.col, k)` is identically `col[pos - k]` within `[part_start, part_end)` (and `NULL`
/// outside), so the entire predicate can be evaluated once in vectorized Arrow into the Tier 1
/// bitmask!
pub fn is_physical_only_stateful(
    sym_id: SymbolId,
    slots: &[PhysicalMeasureSlot],
) -> bool {
    slots.iter().all(|slot| match &slot.kind {
        PhysicalMeasureSlotKind::CurrentRow { .. } => true,
        PhysicalMeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav: Some(_),
            symbol_set,
            uses_match_metadata: false,
            ..
        } => {
            *logical_offset == 0
                && matches!(logical_nav, None | Some(LogicalNavKind::Last))
                && match symbol_set {
                    None => true,
                    Some(SymbolSet::Single(id)) => *id == sym_id,
                    Some(set) => set.contains(sym_id),
                }
        }
        _ => false,
    })
}

/// Evaluates a physical-only stateful `DEFINE` predicate directly into `bitmasks` (`bit` set on
/// every row where the predicate holds), using a zero-allocation primitive fast path when
/// possible and falling back to vectorized Arrow evaluation for general expressions.
pub fn eval_physical_only_into_bitmask(
    batch: &RecordBatch,
    runs: &[(usize, usize)],
    outer_expr: &Arc<dyn PhysicalExpr>,
    slot_schema: &SchemaRef,
    slots: &[PhysicalMeasureSlot],
    bitmasks: &mut [u64],
    bit: u64,
) -> Result<()> {
    // Zero-allocation fast path: compile outer_expr to FastScalarExpr and evaluate directly
    // over raw primitive Arrow slices without allocating shift index arrays, take buffers,
    // slot RecordBatches, or BooleanArrays.
    if let Some(fast_expr) = try_compile_fast_scalar_expr(outer_expr) {
        let num_rows = batch.num_rows();
        let mut slot_specs: Vec<(
            FastPrecomputedArray,
            Option<(PhysicalNavKind, usize)>,
        )> = Vec::with_capacity(slots.len());
        let mut all_fast = true;
        for slot in slots {
            match &slot.kind {
                PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
                    let arr = eval_expr_or_col(inner_expr, batch, num_rows)?;
                    let fa = FastPrecomputedArray::from_array(&arr);
                    if !fa.is_supported() {
                        all_fast = false;
                        break;
                    }
                    slot_specs.push((fa, None));
                }
                PhysicalMeasureSlotKind::Nav {
                    physical_nav: Some(pnav),
                    physical_offset,
                    inner_expr,
                    ..
                } => {
                    let arr = eval_expr_or_col(inner_expr, batch, num_rows)?;
                    let fa = FastPrecomputedArray::from_array(&arr);
                    if !fa.is_supported() {
                        all_fast = false;
                        break;
                    }
                    slot_specs.push((fa, Some((*pnav, *physical_offset as usize))));
                }
                _ => {
                    all_fast = false;
                    break;
                }
            }
        }
        if all_fast {
            // Ultra-fast specialized loop for common 2-slot `col <op> PREV(col, off)` comparisons
            // (e.g. V-shape `down.price < PREV(down.price)`): zero per-row dynamic dispatch,
            // auto-vectorizes cleanly over raw Arrow primitive slices.
            if let FastScalarExpr::Bin { op, left, right } = &fast_expr
                && let (FastScalarExpr::Slot(l_idx), FastScalarExpr::Slot(r_idx)) =
                    (left.as_ref(), right.as_ref())
                && slot_specs.len() == 2
                && let (FastPrecomputedArray::Int64(la), None) = &slot_specs[*l_idx]
                && let (
                    FastPrecomputedArray::Int64(ra),
                    Some((PhysicalNavKind::Prev, off)),
                ) = &slot_specs[*r_idx]
                && la.null_count() == 0
                && ra.null_count() == 0
            {
                let lv = la.values();
                let rv = ra.values();
                let off = *off;
                macro_rules! cmp_loop {
                    ($op:tt) => {
                        for &(p_start, p_end) in runs {
                            let first = (p_start + off).min(p_end);
                            for r in first..p_end {
                                if lv[r] $op rv[r - off] {
                                    bitmasks[r] |= bit;
                                }
                            }
                        }
                    };
                }
                match op {
                    Operator::Lt => {
                        cmp_loop!(<);
                        return Ok(());
                    }
                    Operator::LtEq => {
                        cmp_loop!(<=);
                        return Ok(());
                    }
                    Operator::Gt => {
                        cmp_loop!(>);
                        return Ok(());
                    }
                    Operator::GtEq => {
                        cmp_loop!(>=);
                        return Ok(());
                    }
                    Operator::Eq => {
                        cmp_loop!(==);
                        return Ok(());
                    }
                    Operator::NotEq => {
                        cmp_loop!(!=);
                        return Ok(());
                    }
                    _ => {}
                }
            }

            let mut slot_vals = vec![FastVal::Null; slot_specs.len()];
            for &(p_start, p_end) in runs {
                for (r, mask) in bitmasks.iter_mut().enumerate().take(p_end).skip(p_start)
                {
                    for (i, (fa, shift)) in slot_specs.iter().enumerate() {
                        slot_vals[i] = match shift {
                            None => fa.get_fast(r),
                            Some((PhysicalNavKind::Prev, off)) => {
                                if r >= p_start + off {
                                    fa.get_fast(r - off)
                                } else {
                                    FastVal::Null
                                }
                            }
                            Some((PhysicalNavKind::Next, off)) => {
                                if r + off < p_end {
                                    fa.get_fast(r + off)
                                } else {
                                    FastVal::Null
                                }
                            }
                        };
                    }
                    if matches!(fast_expr.eval(&slot_vals), FastVal::Bool(true)) {
                        *mask |= bit;
                    }
                }
            }
            return Ok(());
        }
    }

    let bool_arr =
        eval_physical_only_stateful_batch(batch, runs, outer_expr, slot_schema, slots)?;
    if bool_arr.null_count() == 0 {
        for idx in bool_arr.values().set_indices() {
            bitmasks[idx] |= bit;
        }
    } else if let Some(nulls) = bool_arr.nulls() {
        let valid_and_true = bool_arr.values() & nulls.inner();
        for idx in valid_and_true.set_indices() {
            bitmasks[idx] |= bit;
        }
    }
    Ok(())
}

/// Evaluates a physical-only stateful `DEFINE` predicate across all partition runs `(p_start, p_end)`
/// in `batch` using vectorized Arrow operations with exact per-partition boundary clamping.
pub fn eval_physical_only_stateful_batch(
    batch: &RecordBatch,
    runs: &[(usize, usize)],
    outer_expr: &Arc<dyn PhysicalExpr>,
    slot_schema: &SchemaRef,
    slots: &[PhysicalMeasureSlot],
) -> Result<BooleanArray> {
    let num_rows = batch.num_rows();
    let mut slot_arrays: Vec<ArrayRef> = Vec::with_capacity(slots.len());
    for slot in slots {
        match &slot.kind {
            PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
                slot_arrays.push(eval_expr_or_col(inner_expr, batch, num_rows)?);
            }
            PhysicalMeasureSlotKind::Nav {
                physical_nav: Some(pnav),
                physical_offset,
                inner_expr,
                ..
            } => {
                let raw = eval_expr_or_col(inner_expr, batch, num_rows)?;
                let off = *physical_offset as usize;
                let mut indices: Vec<Option<u32>> = vec![None; num_rows];
                match pnav {
                    PhysicalNavKind::Prev => {
                        for &(p_start, p_end) in runs {
                            let first_valid = (p_start + off).min(p_end);
                            for (dst, r) in indices[first_valid..p_end]
                                .iter_mut()
                                .zip(first_valid..p_end)
                            {
                                *dst = Some((r - off) as u32);
                            }
                        }
                    }
                    PhysicalNavKind::Next => {
                        for &(p_start, p_end) in runs {
                            let valid_end = p_end.saturating_sub(off).max(p_start);
                            for (dst, r) in indices[p_start..valid_end]
                                .iter_mut()
                                .zip(p_start..valid_end)
                            {
                                *dst = Some((r + off) as u32);
                            }
                        }
                    }
                }
                let idx_arr = UInt32Array::from(indices);
                slot_arrays.push(compute::take(raw.as_ref(), &idx_arr, None)?);
            }
            _ => {
                return Err(internal_datafusion_err!(
                    "eval_physical_only_stateful_batch called on non-physical-only slot"
                ));
            }
        }
    }
    let slot_batch = RecordBatch::try_new_with_options(
        Arc::clone(slot_schema),
        slot_arrays,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )?;
    let evaluated = outer_expr.evaluate(&slot_batch)?.into_array(num_rows)?;
    let bool_arr = evaluated
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| {
            DataFusionError::Execution(format!(
                "DEFINE predicate must evaluate to Boolean, got {}",
                evaluated.data_type()
            ))
        })?;
    Ok(bool_arr.clone())
}

/// Pre-filter extracted from top-level `AND` conjuncts of a Tier 2 stateful `DEFINE` that
/// reference only `CurrentRow` slots (evaluated once per batch into the symbol bitmask).
#[derive(Debug, Clone)]
pub struct RowLocalPrefilter {
    conjuncts: Vec<Arc<dyn PhysicalExpr>>,
    slot_schema: SchemaRef,
    /// `(slot_idx_in_full_slots, inner_expr)` for all slots referenced by `conjuncts`.
    needed_slots: Vec<(usize, Arc<dyn PhysicalExpr>)>,
}

/// Extracts all top-level `AND` conjuncts of `outer_expr` whose columns are all `CurrentRow`.
pub fn extract_row_local_prefilter(
    outer_expr: &Arc<dyn PhysicalExpr>,
    slot_schema: &SchemaRef,
    slots: &[PhysicalMeasureSlot],
) -> Option<RowLocalPrefilter> {
    let mut parts = Vec::new();
    collect_and_conjuncts(outer_expr, &mut parts);

    let mut row_local_conjuncts = Vec::new();
    let mut referenced_slot_indices = HashSet::new();
    for p in parts {
        let mut cols = Vec::new();
        collect_phys_columns(&p, &mut cols);
        if !cols.is_empty()
            && cols.iter().all(|&idx| {
                matches!(
                    slots.get(idx).map(|s| &s.kind),
                    Some(PhysicalMeasureSlotKind::CurrentRow { .. })
                )
            })
        {
            for idx in cols {
                referenced_slot_indices.insert(idx);
            }
            row_local_conjuncts.push(p);
        }
    }

    if row_local_conjuncts.is_empty() {
        return None;
    }

    let mut needed_slots = Vec::with_capacity(referenced_slot_indices.len());
    for (idx, slot) in slots.iter().enumerate() {
        if referenced_slot_indices.contains(&idx)
            && let PhysicalMeasureSlotKind::CurrentRow { inner_expr } = &slot.kind
        {
            needed_slots.push((idx, Arc::clone(inner_expr)));
        }
    }

    let nullable_fields: Vec<Field> = slot_schema
        .fields()
        .iter()
        .map(|f| f.as_ref().clone().with_nullable(true))
        .collect();
    Some(RowLocalPrefilter {
        conjuncts: row_local_conjuncts,
        slot_schema: Arc::new(Schema::new(nullable_fields)),
        needed_slots,
    })
}

/// Evaluates a [`RowLocalPrefilter`] in vectorized Arrow across the whole `batch`.
pub fn eval_row_local_prefilter_batch(
    batch: &RecordBatch,
    prefilter: &RowLocalPrefilter,
) -> Result<BooleanArray> {
    let num_rows = batch.num_rows();
    let mut slot_arrays: Vec<ArrayRef> = prefilter
        .slot_schema
        .fields()
        .iter()
        .map(|f| new_null_array(f.data_type(), num_rows))
        .collect();
    for (idx, expr) in &prefilter.needed_slots {
        slot_arrays[*idx] = eval_expr_or_col(expr, batch, num_rows)?;
    }
    let slot_batch = RecordBatch::try_new_with_options(
        Arc::clone(&prefilter.slot_schema),
        slot_arrays,
        &RecordBatchOptions::new().with_row_count(Some(num_rows)),
    )?;
    let mut acc: Option<BooleanArray> = None;
    for conj in &prefilter.conjuncts {
        let arr = conj.evaluate(&slot_batch)?.into_array(num_rows)?;
        let b = arr.as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
            DataFusionError::Execution(format!(
                "DEFINE conjunct must evaluate to Boolean, got {}",
                arr.data_type()
            ))
        })?;
        acc = Some(match acc {
            None => b.clone(),
            Some(prev) => compute::and(&prev, b)?,
        });
    }
    acc.ok_or_else(|| internal_datafusion_err!("empty RowLocalPrefilter conjuncts"))
}

fn collect_and_conjuncts(
    expr: &Arc<dyn PhysicalExpr>,
    out: &mut Vec<Arc<dyn PhysicalExpr>>,
) {
    if let Some(bin) = expr.downcast_ref::<BinaryExpr>()
        && *bin.op() == Operator::And
    {
        collect_and_conjuncts(bin.left(), out);
        collect_and_conjuncts(bin.right(), out);
        return;
    }
    out.push(Arc::clone(expr));
}

fn collect_phys_columns(expr: &Arc<dyn PhysicalExpr>, out: &mut Vec<usize>) {
    if let Some(col) = expr.downcast_ref::<Column>() {
        out.push(col.index());
        return;
    }
    for child in expr.children() {
        collect_phys_columns(child, out);
    }
}

/// Compact primitive value for the zero-allocation Tier 2 scalar predicate evaluator.
#[derive(Debug, Clone, Copy)]
enum FastVal {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
}

/// Compiled native scalar expression tree for Tier 2 stateful `DEFINE` predicates, avoiding
/// per-transition 1-row `RecordBatch` allocations and `ScalarValue` boxing.
#[derive(Debug, Clone)]
enum FastScalarExpr {
    Slot(usize),
    Lit(FastVal),
    Not(Box<FastScalarExpr>),
    IsNull(Box<FastScalarExpr>),
    IsNotNull(Box<FastScalarExpr>),
    CastToInt64(Box<FastScalarExpr>),
    CastToFloat64(Box<FastScalarExpr>),
    Bin {
        op: Operator,
        left: Box<FastScalarExpr>,
        right: Box<FastScalarExpr>,
    },
}

fn try_compile_fast_scalar_expr(expr: &Arc<dyn PhysicalExpr>) -> Option<FastScalarExpr> {
    if let Some(col) = expr.downcast_ref::<Column>() {
        return Some(FastScalarExpr::Slot(col.index()));
    }
    if let Some(lit) = expr.downcast_ref::<Literal>() {
        let v = match lit.value() {
            ScalarValue::Boolean(Some(b)) => FastVal::Bool(*b),
            ScalarValue::Int64(Some(n)) => FastVal::Int64(*n),
            ScalarValue::Int32(Some(n)) => FastVal::Int64(i64::from(*n)),
            ScalarValue::Int16(Some(n)) => FastVal::Int64(i64::from(*n)),
            ScalarValue::Int8(Some(n)) => FastVal::Int64(i64::from(*n)),
            ScalarValue::Float64(Some(f)) => FastVal::Float64(*f),
            ScalarValue::Float32(Some(f)) => FastVal::Float64(f64::from(*f)),
            ScalarValue::Null
            | ScalarValue::Boolean(None)
            | ScalarValue::Int64(None)
            | ScalarValue::Int32(None)
            | ScalarValue::Float64(None) => FastVal::Null,
            _ => return None,
        };
        return Some(FastScalarExpr::Lit(v));
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return Some(FastScalarExpr::Not(Box::new(try_compile_fast_scalar_expr(
            not.arg(),
        )?)));
    }
    if let Some(isn) = expr.downcast_ref::<IsNullExpr>() {
        return Some(FastScalarExpr::IsNull(Box::new(
            try_compile_fast_scalar_expr(isn.arg())?,
        )));
    }
    if let Some(isnn) = expr.downcast_ref::<IsNotNullExpr>() {
        return Some(FastScalarExpr::IsNotNull(Box::new(
            try_compile_fast_scalar_expr(isnn.arg())?,
        )));
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let inner = Box::new(try_compile_fast_scalar_expr(cast.expr())?);
        return match cast.cast_type() {
            DataType::Int64 | DataType::Int32 => Some(FastScalarExpr::CastToInt64(inner)),
            DataType::Float64 | DataType::Float32 => {
                Some(FastScalarExpr::CastToFloat64(inner))
            }
            _ => None,
        };
    }
    if let Some(bin) = expr.downcast_ref::<BinaryExpr>() {
        let op = *bin.op();
        if !matches!(
            op,
            Operator::And
                | Operator::Or
                | Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
                | Operator::Plus
                | Operator::Minus
                | Operator::Multiply
        ) {
            return None;
        }
        let left = Box::new(try_compile_fast_scalar_expr(bin.left())?);
        let right = Box::new(try_compile_fast_scalar_expr(bin.right())?);
        return Some(FastScalarExpr::Bin { op, left, right });
    }
    None
}

impl FastScalarExpr {
    #[inline]
    fn eval(&self, slot_vals: &[FastVal]) -> FastVal {
        match self {
            FastScalarExpr::Slot(idx) => slot_vals[*idx],
            FastScalarExpr::Lit(v) => *v,
            FastScalarExpr::Not(inner) => match inner.eval(slot_vals) {
                FastVal::Bool(b) => FastVal::Bool(!b),
                _ => FastVal::Null,
            },
            FastScalarExpr::IsNull(inner) => {
                FastVal::Bool(matches!(inner.eval(slot_vals), FastVal::Null))
            }
            FastScalarExpr::IsNotNull(inner) => {
                FastVal::Bool(!matches!(inner.eval(slot_vals), FastVal::Null))
            }
            FastScalarExpr::CastToInt64(inner) => match inner.eval(slot_vals) {
                FastVal::Int64(v) => FastVal::Int64(v),
                FastVal::Float64(f) => FastVal::Int64(f as i64),
                _ => FastVal::Null,
            },
            FastScalarExpr::CastToFloat64(inner) => match inner.eval(slot_vals) {
                FastVal::Float64(f) => FastVal::Float64(f),
                FastVal::Int64(v) => FastVal::Float64(v as f64),
                _ => FastVal::Null,
            },
            FastScalarExpr::Bin { op, left, right } => {
                let lv = left.eval(slot_vals);
                match op {
                    Operator::And => match lv {
                        FastVal::Bool(false) => FastVal::Bool(false),
                        FastVal::Bool(true) => right.eval(slot_vals),
                        _ => match right.eval(slot_vals) {
                            FastVal::Bool(false) => FastVal::Bool(false),
                            _ => FastVal::Null,
                        },
                    },
                    Operator::Or => match lv {
                        FastVal::Bool(true) => FastVal::Bool(true),
                        FastVal::Bool(false) => right.eval(slot_vals),
                        _ => match right.eval(slot_vals) {
                            FastVal::Bool(true) => FastVal::Bool(true),
                            _ => FastVal::Null,
                        },
                    },
                    _ => {
                        let rv = right.eval(slot_vals);
                        match (lv, rv) {
                            (FastVal::Int64(a), FastVal::Int64(b)) => match op {
                                Operator::Eq => FastVal::Bool(a == b),
                                Operator::NotEq => FastVal::Bool(a != b),
                                Operator::Lt => FastVal::Bool(a < b),
                                Operator::LtEq => FastVal::Bool(a <= b),
                                Operator::Gt => FastVal::Bool(a > b),
                                Operator::GtEq => FastVal::Bool(a >= b),
                                Operator::Plus => FastVal::Int64(a.wrapping_add(b)),
                                Operator::Minus => FastVal::Int64(a.wrapping_sub(b)),
                                Operator::Multiply => FastVal::Int64(a.wrapping_mul(b)),
                                Operator::Divide => {
                                    if b == 0 {
                                        FastVal::Null
                                    } else {
                                        FastVal::Int64(a / b)
                                    }
                                }
                                _ => FastVal::Null,
                            },
                            (FastVal::Float64(a), FastVal::Float64(b)) => match op {
                                Operator::Eq => FastVal::Bool(a == b),
                                Operator::NotEq => FastVal::Bool(a != b),
                                Operator::Lt => FastVal::Bool(a < b),
                                Operator::LtEq => FastVal::Bool(a <= b),
                                Operator::Gt => FastVal::Bool(a > b),
                                Operator::GtEq => FastVal::Bool(a >= b),
                                Operator::Plus => FastVal::Float64(a + b),
                                Operator::Minus => FastVal::Float64(a - b),
                                Operator::Multiply => FastVal::Float64(a * b),
                                Operator::Divide => FastVal::Float64(a / b),
                                _ => FastVal::Null,
                            },
                            (FastVal::Bool(a), FastVal::Bool(b)) => match op {
                                Operator::Eq => FastVal::Bool(a == b),
                                Operator::NotEq => FastVal::Bool(a != b),
                                _ => FastVal::Null,
                            },
                            _ => FastVal::Null,
                        }
                    }
                }
            }
        }
    }
}

/// Typed primitive view over a precomputed slot argument array for zero-boxing Tier 2 reads.
enum FastPrecomputedArray {
    Int64(Int64Array),
    Int32(Int32Array),
    Float64(Float64Array),
    Bool(BooleanArray),
    Other,
}

impl FastPrecomputedArray {
    fn from_array(arr: &ArrayRef) -> Self {
        if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
            Self::Int64(a.clone())
        } else if let Some(a) = arr.as_any().downcast_ref::<Int32Array>() {
            Self::Int32(a.clone())
        } else if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
            Self::Float64(a.clone())
        } else if let Some(a) = arr.as_any().downcast_ref::<BooleanArray>() {
            Self::Bool(a.clone())
        } else {
            Self::Other
        }
    }

    #[inline]
    fn get_fast(&self, idx: usize) -> FastVal {
        match self {
            FastPrecomputedArray::Int64(a) => {
                if a.is_valid(idx) {
                    FastVal::Int64(a.value(idx))
                } else {
                    FastVal::Null
                }
            }
            FastPrecomputedArray::Int32(a) => {
                if a.is_valid(idx) {
                    FastVal::Int64(i64::from(a.value(idx)))
                } else {
                    FastVal::Null
                }
            }
            FastPrecomputedArray::Float64(a) => {
                if a.is_valid(idx) {
                    FastVal::Float64(a.value(idx))
                } else {
                    FastVal::Null
                }
            }
            FastPrecomputedArray::Bool(a) => {
                if a.is_valid(idx) {
                    FastVal::Bool(a.value(idx))
                } else {
                    FastVal::Null
                }
            }
            FastPrecomputedArray::Other => FastVal::Null,
        }
    }

    fn is_supported(&self) -> bool {
        !matches!(self, FastPrecomputedArray::Other)
    }
}

fn slot_supports_fast_eval(
    slot: &PhysicalMeasureSlot,
    fast_args: &[FastPrecomputedArray],
) -> bool {
    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => true,
        PhysicalMeasureSlotKind::CurrentRow { .. } => fast_args
            .first()
            .is_some_and(FastPrecomputedArray::is_supported),
        PhysicalMeasureSlotKind::Nav {
            uses_match_metadata: false,
            ..
        } => fast_args
            .first()
            .is_some_and(FastPrecomputedArray::is_supported),
        PhysicalMeasureSlotKind::Agg {
            agg_kind,
            distinct: false,
            uses_match_metadata: false,
            ..
        } => match agg_kind {
            AggKind::CountStar | AggKind::CountSymbol => true,
            AggKind::Count
            | AggKind::Sum
            | AggKind::Avg
            | AggKind::Min
            | AggKind::Max => fast_args
                .first()
                .is_some_and(FastPrecomputedArray::is_supported),
            AggKind::Udaf(_) => false,
        },
        _ => false,
    }
}

#[inline]
fn eval_slot_fast(
    horizon: &[MatchStep],
    match_number: i64,
    current_row: usize,
    part_start: usize,
    part_end: usize,
    slot: &PhysicalMeasureSlot,
    fast_args: &[FastPrecomputedArray],
) -> FastVal {
    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => FastVal::Int64(match_number),
        PhysicalMeasureSlotKind::CurrentRow { .. } => fast_args[0].get_fast(current_row),
        PhysicalMeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            symbol_set,
            ..
        } => {
            let Some(base_row) = navigate_logical_row(
                horizon,
                *logical_nav,
                *logical_offset as usize,
                symbol_set.as_ref(),
            ) else {
                return FastVal::Null;
            };
            let Some(final_row) = navigate_physical_in_range(
                base_row,
                *physical_nav,
                *physical_offset as usize,
                part_start,
                part_end,
            ) else {
                return FastVal::Null;
            };
            fast_args[0].get_fast(final_row)
        }
        PhysicalMeasureSlotKind::Agg {
            agg_kind,
            symbol_set,
            arg_physical_nav,
            ..
        } => match agg_kind {
            AggKind::CountStar | AggKind::CountSymbol => {
                let c = matching_row_indices_in_range(
                    horizon,
                    symbol_set.as_ref(),
                    None,
                    part_start,
                    part_end,
                )
                .flatten()
                .count();
                FastVal::Int64(c as i64)
            }
            AggKind::Count => {
                let fa = &fast_args[0];
                let mut c = 0i64;
                for idx in matching_row_indices_in_range(
                    horizon,
                    symbol_set.as_ref(),
                    *arg_physical_nav,
                    part_start,
                    part_end,
                )
                .flatten()
                {
                    if !matches!(fa.get_fast(idx), FastVal::Null) {
                        c += 1;
                    }
                }
                FastVal::Int64(c)
            }
            AggKind::Sum => {
                let fa = &fast_args[0];
                match fa {
                    FastPrecomputedArray::Int64(a) => {
                        let mut acc: Option<i64> = None;
                        for idx in matching_row_indices_in_range(
                            horizon,
                            symbol_set.as_ref(),
                            *arg_physical_nav,
                            part_start,
                            part_end,
                        )
                        .flatten()
                        {
                            if a.is_valid(idx) {
                                acc = Some(acc.unwrap_or(0).wrapping_add(a.value(idx)));
                            }
                        }
                        acc.map_or(FastVal::Null, FastVal::Int64)
                    }
                    FastPrecomputedArray::Float64(a) => {
                        let mut acc: Option<f64> = None;
                        for idx in matching_row_indices_in_range(
                            horizon,
                            symbol_set.as_ref(),
                            *arg_physical_nav,
                            part_start,
                            part_end,
                        )
                        .flatten()
                        {
                            if a.is_valid(idx) {
                                acc = Some(acc.unwrap_or(0.0) + a.value(idx));
                            }
                        }
                        acc.map_or(FastVal::Null, FastVal::Float64)
                    }
                    _ => FastVal::Null,
                }
            }
            AggKind::Avg => {
                let FastPrecomputedArray::Float64(a) = &fast_args[0] else {
                    return FastVal::Null;
                };
                let mut sum = 0.0f64;
                let mut count = 0u64;
                for idx in matching_row_indices_in_range(
                    horizon,
                    symbol_set.as_ref(),
                    *arg_physical_nav,
                    part_start,
                    part_end,
                )
                .flatten()
                {
                    if a.is_valid(idx) {
                        sum += a.value(idx);
                        count += 1;
                    }
                }
                if count == 0 {
                    FastVal::Null
                } else {
                    FastVal::Float64(sum / count as f64)
                }
            }
            AggKind::Min | AggKind::Max => {
                let is_min = matches!(agg_kind, AggKind::Min);
                let fa = &fast_args[0];
                match fa {
                    FastPrecomputedArray::Int64(a) => {
                        let mut best: Option<i64> = None;
                        for idx in matching_row_indices_in_range(
                            horizon,
                            symbol_set.as_ref(),
                            *arg_physical_nav,
                            part_start,
                            part_end,
                        )
                        .flatten()
                        {
                            if a.is_valid(idx) {
                                let v = a.value(idx);
                                best = Some(match best {
                                    None => v,
                                    Some(c) => {
                                        if is_min {
                                            c.min(v)
                                        } else {
                                            c.max(v)
                                        }
                                    }
                                });
                            }
                        }
                        best.map_or(FastVal::Null, FastVal::Int64)
                    }
                    FastPrecomputedArray::Int32(a) => {
                        let mut best: Option<i64> = None;
                        for idx in matching_row_indices_in_range(
                            horizon,
                            symbol_set.as_ref(),
                            *arg_physical_nav,
                            part_start,
                            part_end,
                        )
                        .flatten()
                        {
                            if a.is_valid(idx) {
                                let v = i64::from(a.value(idx));
                                best = Some(match best {
                                    None => v,
                                    Some(c) => {
                                        if is_min {
                                            c.min(v)
                                        } else {
                                            c.max(v)
                                        }
                                    }
                                });
                            }
                        }
                        best.map_or(FastVal::Null, FastVal::Int64)
                    }
                    FastPrecomputedArray::Float64(a) => {
                        let mut best: Option<f64> = None;
                        for idx in matching_row_indices_in_range(
                            horizon,
                            symbol_set.as_ref(),
                            *arg_physical_nav,
                            part_start,
                            part_end,
                        )
                        .flatten()
                        {
                            if a.is_valid(idx) {
                                let v = a.value(idx);
                                best = Some(match best {
                                    None => v,
                                    Some(c) => {
                                        if (is_min && v < c) || (!is_min && v > c) {
                                            v
                                        } else {
                                            c
                                        }
                                    }
                                });
                            }
                        }
                        best.map_or(FastVal::Null, FastVal::Float64)
                    }
                    _ => FastVal::Null,
                }
            }
            AggKind::Udaf(_) => FastVal::Null,
        },
        _ => FastVal::Null,
    }
}

/// Borrowed specification for one stateful `DEFINE` symbol passed to [`PartitionStatefulEvaluator`].
pub struct StatefulDefineInput<'a> {
    pub sym_id: SymbolId,
    pub outer_expr: &'a Arc<dyn PhysicalExpr>,
    pub slot_schema: &'a SchemaRef,
    pub slots: &'a [PhysicalMeasureSlot],
    pub has_prefilter: bool,
}

/// Incremental prefix accumulator for one fast-path aggregate slot in Tier 2 stateful `DEFINE`.
#[derive(Debug, Clone, Copy)]
enum PrefixAggAcc {
    None,
    CountStar(i64),
    CountValid(i64),
    SumI64(Option<i64>),
    SumF64(Option<f64>),
    AvgF64 { sum: f64, count: u64 },
    MinI64(Option<i64>),
    MaxI64(Option<i64>),
    MinF64(Option<f64>),
    MaxF64(Option<f64>),
}

impl PrefixAggAcc {
    fn for_slot(slot: &PhysicalMeasureSlot, fast_args: &[FastPrecomputedArray]) -> Self {
        let PhysicalMeasureSlotKind::Agg {
            agg_kind,
            distinct: false,
            arg_physical_nav: None,
            uses_match_metadata: false,
            ..
        } = &slot.kind
        else {
            return Self::None;
        };
        match agg_kind {
            AggKind::CountStar | AggKind::CountSymbol => Self::CountStar(0),
            AggKind::Count => Self::CountValid(0),
            AggKind::Sum => match fast_args.first() {
                Some(FastPrecomputedArray::Int64(_)) => Self::SumI64(None),
                Some(FastPrecomputedArray::Float64(_)) => Self::SumF64(None),
                _ => Self::None,
            },
            AggKind::Avg => match fast_args.first() {
                Some(FastPrecomputedArray::Float64(_)) => {
                    Self::AvgF64 { sum: 0.0, count: 0 }
                }
                _ => Self::None,
            },
            AggKind::Min => match fast_args.first() {
                Some(FastPrecomputedArray::Int64(_) | FastPrecomputedArray::Int32(_)) => {
                    Self::MinI64(None)
                }
                Some(FastPrecomputedArray::Float64(_)) => Self::MinF64(None),
                _ => Self::None,
            },
            AggKind::Max => match fast_args.first() {
                Some(FastPrecomputedArray::Int64(_) | FastPrecomputedArray::Int32(_)) => {
                    Self::MaxI64(None)
                }
                Some(FastPrecomputedArray::Float64(_)) => Self::MaxF64(None),
                _ => Self::None,
            },
            AggKind::Udaf(_) => Self::None,
        }
    }

    #[inline]
    fn reset(&mut self) {
        match self {
            Self::None => {}
            Self::CountStar(c) | Self::CountValid(c) => *c = 0,
            Self::SumI64(v) | Self::MinI64(v) | Self::MaxI64(v) => *v = None,
            Self::SumF64(v) | Self::MinF64(v) | Self::MaxF64(v) => *v = None,
            Self::AvgF64 { sum, count } => {
                *sum = 0.0;
                *count = 0;
            }
        }
    }

    #[inline]
    fn push_step(
        &mut self,
        step: MatchStep,
        symbol_set: Option<&SymbolSet>,
        fast_args: &[FastPrecomputedArray],
    ) {
        if matches!(self, Self::None) {
            return;
        }
        if symbol_set.is_some_and(|set| !set.contains(step.sym)) {
            return;
        }
        let idx = step.row_idx;
        match self {
            Self::None => {}
            Self::CountStar(c) => *c += 1,
            Self::CountValid(c) => {
                if !matches!(fast_args[0].get_fast(idx), FastVal::Null) {
                    *c += 1;
                }
            }
            Self::SumI64(acc) => {
                if let FastVal::Int64(v) = fast_args[0].get_fast(idx) {
                    *acc = Some(acc.unwrap_or(0).wrapping_add(v));
                }
            }
            Self::SumF64(acc) => {
                if let FastVal::Float64(v) = fast_args[0].get_fast(idx) {
                    *acc = Some(acc.unwrap_or(0.0) + v);
                }
            }
            Self::AvgF64 { sum, count } => {
                if let FastVal::Float64(v) = fast_args[0].get_fast(idx) {
                    *sum += v;
                    *count += 1;
                }
            }
            Self::MinI64(best) => {
                if let FastVal::Int64(v) = fast_args[0].get_fast(idx) {
                    *best = Some(best.map_or(v, |c| c.min(v)));
                }
            }
            Self::MaxI64(best) => {
                if let FastVal::Int64(v) = fast_args[0].get_fast(idx) {
                    *best = Some(best.map_or(v, |c| c.max(v)));
                }
            }
            Self::MinF64(best) => {
                if let FastVal::Float64(v) = fast_args[0].get_fast(idx) {
                    *best = Some(best.map_or(v, |c| if v < c { v } else { c }));
                }
            }
            Self::MaxF64(best) => {
                if let FastVal::Float64(v) = fast_args[0].get_fast(idx) {
                    *best = Some(best.map_or(v, |c| if v > c { v } else { c }));
                }
            }
        }
    }

    #[inline]
    fn eval_with_candidate(
        &self,
        cand: MatchStep,
        symbol_set: Option<&SymbolSet>,
        fast_args: &[FastPrecomputedArray],
    ) -> Option<FastVal> {
        if matches!(self, Self::None) {
            return None;
        }
        let mut tmp = *self;
        tmp.push_step(cand, symbol_set, fast_args);
        let val = match tmp {
            Self::None => return None,
            Self::CountStar(c) | Self::CountValid(c) => FastVal::Int64(c),
            Self::SumI64(v) | Self::MinI64(v) | Self::MaxI64(v) => {
                v.map_or(FastVal::Null, FastVal::Int64)
            }
            Self::SumF64(v) | Self::MinF64(v) | Self::MaxF64(v) => {
                v.map_or(FastVal::Null, FastVal::Float64)
            }
            Self::AvgF64 { sum, count } => {
                if count == 0 {
                    FastVal::Null
                } else {
                    FastVal::Float64(sum / count as f64)
                }
            }
        };
        Some(val)
    }
}

struct SymbolPrefixCache {
    attempt_gen: u64,
    valid_head: u32,
    first_row_by_sym: Vec<Option<usize>>,
    last_row_by_sym: Vec<Option<usize>>,
    slot_aggs: Vec<PrefixAggAcc>,
    has_any_agg: bool,
}

impl SymbolPrefixCache {
    fn new(
        num_symbols: usize,
        slots: &[PhysicalMeasureSlot],
        fast_args: &[Vec<FastPrecomputedArray>],
    ) -> Self {
        let slot_aggs: Vec<PrefixAggAcc> = slots
            .iter()
            .zip(fast_args)
            .map(|(s, fa)| PrefixAggAcc::for_slot(s, fa))
            .collect();
        let has_any_agg = slot_aggs.iter().any(|a| !matches!(a, PrefixAggAcc::None));
        Self {
            attempt_gen: 0,
            valid_head: NO_HISTORY,
            first_row_by_sym: vec![None; num_symbols],
            last_row_by_sym: vec![None; num_symbols],
            slot_aggs,
            has_any_agg,
        }
    }

    #[inline]
    fn reset(&mut self, attempt_gen: u64) {
        self.attempt_gen = attempt_gen;
        self.valid_head = NO_HISTORY;
        self.first_row_by_sym.fill(None);
        self.last_row_by_sym.fill(None);
        if self.has_any_agg {
            for acc in &mut self.slot_aggs {
                acc.reset();
            }
        }
    }

    #[inline]
    fn push_committed_step(
        &mut self,
        step: MatchStep,
        slots: &[PhysicalMeasureSlot],
        fast_args: &[Vec<FastPrecomputedArray>],
    ) {
        let sym_idx = step.sym as usize;
        if self.first_row_by_sym[sym_idx].is_none() {
            self.first_row_by_sym[sym_idx] = Some(step.row_idx);
        }
        self.last_row_by_sym[sym_idx] = Some(step.row_idx);
        if self.has_any_agg {
            for ((acc, slot), fargs) in
                self.slot_aggs.iter_mut().zip(slots).zip(fast_args)
            {
                if let PhysicalMeasureSlotKind::Agg { symbol_set, .. } = &slot.kind {
                    acc.push_step(step, symbol_set.as_ref(), fargs);
                }
            }
        }
    }

    #[inline]
    fn sync_to_prefix(
        &mut self,
        attempt_gen: u64,
        prefix: &[MatchStep],
        prefix_head: u32,
        prefix_prev_head: u32,
        slots: &[PhysicalMeasureSlot],
        fast_args: &[Vec<FastPrecomputedArray>],
    ) {
        if self.attempt_gen != attempt_gen {
            self.reset(attempt_gen);
        }
        if prefix_head == self.valid_head {
            return;
        }
        if prefix_head != NO_HISTORY && prefix_prev_head == self.valid_head {
            if let Some(&committed) = prefix.last() {
                self.push_committed_step(committed, slots, fast_args);
            }
            self.valid_head = prefix_head;
            return;
        }
        self.reset(attempt_gen);
        for &step in prefix {
            self.push_committed_step(step, slots, fast_args);
        }
        self.valid_head = prefix_head;
    }
}

#[inline]
#[expect(clippy::too_many_arguments)]
fn eval_slot_fast_cached(
    tentative_steps: &[MatchStep],
    cand: MatchStep,
    match_number: i64,
    part_start: usize,
    part_end: usize,
    slot: &PhysicalMeasureSlot,
    fast_args: &[FastPrecomputedArray],
    cache: &SymbolPrefixCache,
    slot_idx: usize,
) -> FastVal {
    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => FastVal::Int64(match_number),
        PhysicalMeasureSlotKind::CurrentRow { .. } => fast_args[0].get_fast(cand.row_idx),
        PhysicalMeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            symbol_set,
            ..
        } => {
            let base_row = if *logical_offset == 0 {
                match (logical_nav, symbol_set.as_ref()) {
                    (None | Some(LogicalNavKind::Last), None) => Some(cand.row_idx),
                    (Some(LogicalNavKind::First), None) => {
                        tentative_steps.first().map(|s| s.row_idx)
                    }
                    (
                        None | Some(LogicalNavKind::Last),
                        Some(SymbolSet::Single(target_sym)),
                    ) => {
                        if cand.sym == *target_sym {
                            Some(cand.row_idx)
                        } else {
                            cache.last_row_by_sym[*target_sym as usize]
                        }
                    }
                    (
                        Some(LogicalNavKind::First),
                        Some(SymbolSet::Single(target_sym)),
                    ) => cache.first_row_by_sym[*target_sym as usize]
                        .or_else(|| (cand.sym == *target_sym).then_some(cand.row_idx)),
                    _ => navigate_logical_row(
                        tentative_steps,
                        *logical_nav,
                        0,
                        symbol_set.as_ref(),
                    ),
                }
            } else {
                navigate_logical_row(
                    tentative_steps,
                    *logical_nav,
                    *logical_offset as usize,
                    symbol_set.as_ref(),
                )
            };
            let Some(base_row) = base_row else {
                return FastVal::Null;
            };
            let Some(final_row) = navigate_physical_in_range(
                base_row,
                *physical_nav,
                *physical_offset as usize,
                part_start,
                part_end,
            ) else {
                return FastVal::Null;
            };
            fast_args[0].get_fast(final_row)
        }
        PhysicalMeasureSlotKind::Agg { symbol_set, .. } => {
            if let Some(v) = cache.slot_aggs[slot_idx].eval_with_candidate(
                cand,
                symbol_set.as_ref(),
                fast_args,
            ) {
                return v;
            }
            eval_slot_fast(
                tentative_steps,
                match_number,
                cand.row_idx,
                part_start,
                part_end,
                slot,
                fast_args,
            )
        }
        _ => FastVal::Null,
    }
}

struct CompiledStatefulSymbol {
    outer_expr: Arc<dyn PhysicalExpr>,
    slot_schema: SchemaRef,
    slots: Vec<PhysicalMeasureSlot>,
    /// Pre-evaluated inner Arrow arrays (`precomputed_args[slot_idx][arg_idx]`).
    precomputed_args: Vec<Vec<ArrayRef>>,
    /// Typed primitive views over `precomputed_args` for zero-boxing Tier 2 evaluation.
    fast_args: Vec<Vec<FastPrecomputedArray>>,
    /// Optional compiled scalar expression tree (bypasses 1-row `RecordBatch` creation).
    fast_expr: Option<FastScalarExpr>,
    /// Incremental O(1) prefix-history cache keyed by arena `history_head`.
    prefix_cache: SymbolPrefixCache,
    has_prefilter: bool,
}

/// Evaluates Tier 2 stateful `DEFINE` predicates against a batch's precomputed slot arrays.
pub struct PartitionStatefulEvaluator<'a> {
    program: &'a Program,
    partition_batch: &'a RecordBatch,
    input_schema: SchemaRef,
    part_start: usize,
    part_end: usize,
    by_symbol: Vec<Option<CompiledStatefulSymbol>>,
    has_any: bool,
    attempt_gen: u64,
    fast_slot_buf: Vec<FastVal>,
}

impl<'a> PartitionStatefulEvaluator<'a> {
    /// Pre-evaluates all non-metadata stateful `DEFINE` inner expressions once for `partition_batch`.
    pub fn try_new(
        partition_batch: &'a RecordBatch,
        program: &'a Program,
        stateful_defs: &[StatefulDefineInput<'_>],
    ) -> Result<Self> {
        let num_rows = partition_batch.num_rows();
        let num_symbols = program.num_symbols();
        let mut by_symbol: Vec<Option<CompiledStatefulSymbol>> =
            (0..num_symbols).map(|_| None).collect();
        let mut max_slots = 0usize;

        for def in stateful_defs {
            max_slots = max_slots.max(def.slots.len());
            let mut precomputed_args = Vec::with_capacity(def.slots.len());
            let mut fast_args = Vec::with_capacity(def.slots.len());
            for slot in def.slots {
                let arrs = match &slot.kind {
                    PhysicalMeasureSlotKind::Nav {
                        inner_expr,
                        uses_match_metadata,
                        ..
                    } => {
                        if *uses_match_metadata {
                            vec![]
                        } else {
                            vec![eval_expr_or_col(inner_expr, partition_batch, num_rows)?]
                        }
                    }
                    PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
                        vec![eval_expr_or_col(inner_expr, partition_batch, num_rows)?]
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
                                let raw = eval_expr_or_col(e, partition_batch, num_rows)?;
                                let pre = if i == 0 {
                                    match agg_kind {
                                        AggKind::Avg => compute::cast(
                                            raw.as_ref(),
                                            &DataType::Float64,
                                        )?,
                                        AggKind::Sum
                                            if raw.data_type() != &slot.data_type =>
                                        {
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
                let fargs: Vec<FastPrecomputedArray> =
                    arrs.iter().map(FastPrecomputedArray::from_array).collect();
                precomputed_args.push(arrs);
                fast_args.push(fargs);
            }
            let all_slots_fast = def
                .slots
                .iter()
                .zip(&fast_args)
                .all(|(s, fa)| slot_supports_fast_eval(s, fa));
            let fast_expr = if all_slots_fast {
                try_compile_fast_scalar_expr(def.outer_expr)
            } else {
                None
            };
            let slots_vec = def.slots.to_vec();
            let prefix_cache =
                SymbolPrefixCache::new(num_symbols, &slots_vec, &fast_args);
            by_symbol[def.sym_id as usize] = Some(CompiledStatefulSymbol {
                outer_expr: Arc::clone(def.outer_expr),
                slot_schema: Arc::clone(def.slot_schema),
                slots: slots_vec,
                precomputed_args,
                fast_args,
                fast_expr,
                prefix_cache,
                has_prefilter: def.has_prefilter,
            });
        }

        Ok(Self {
            program,
            partition_batch,
            input_schema: partition_batch.schema(),
            part_start: 0,
            part_end: num_rows,
            by_symbol,
            has_any: !stateful_defs.is_empty(),
            attempt_gen: 1,
            fast_slot_buf: vec![FastVal::Null; max_slots],
        })
    }

    /// Updates the active logical partition bounds `[part_start..part_end)` within the batch.
    #[inline]
    pub fn set_partition_bounds(&mut self, part_start: usize, part_end: usize) {
        self.part_start = part_start;
        self.part_end = part_end;
        self.attempt_gen = self.attempt_gen.wrapping_add(1).max(1);
    }
}

impl StatefulPredicateEvaluator for PartitionStatefulEvaluator<'_> {
    #[inline]
    fn is_stateful(&self, sym: SymbolId) -> bool {
        self.by_symbol
            .get(sym as usize)
            .is_some_and(Option::is_some)
    }

    #[inline]
    fn has_any_stateful(&self) -> bool {
        self.has_any
    }

    #[inline]
    fn has_prefilter(&self, sym: SymbolId) -> bool {
        self.by_symbol
            .get(sym as usize)
            .and_then(Option::as_ref)
            .is_some_and(|c| c.has_prefilter)
    }

    #[inline]
    fn on_new_match_attempt(&mut self) {
        self.attempt_gen = self.attempt_gen.wrapping_add(1).max(1);
    }

    fn eval_stateful(
        &mut self,
        sym: SymbolId,
        tentative_steps: &[MatchStep],
        candidate_match_number: i64,
    ) -> Result<bool> {
        // Without history-head tokens, force a clean prefix sync by bumping attempt_gen.
        self.attempt_gen = self.attempt_gen.wrapping_add(1).max(1);
        self.eval_stateful_with_history(
            sym,
            tentative_steps,
            candidate_match_number,
            NO_HISTORY,
            NO_HISTORY,
        )
    }

    fn eval_stateful_with_history(
        &mut self,
        sym: SymbolId,
        tentative_steps: &[MatchStep],
        candidate_match_number: i64,
        prefix_head: u32,
        prefix_prev_head: u32,
    ) -> Result<bool> {
        let compiled = self
            .by_symbol
            .get_mut(sym as usize)
            .and_then(Option::as_mut)
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "eval_stateful called for non-stateful symbol id {sym}"
                )
            })?;
        let cand = *tentative_steps.last().ok_or_else(|| {
            internal_datafusion_err!(
                "tentative_steps missing candidate row in eval_stateful"
            )
        })?;
        let current_row = cand.row_idx;

        if let Some(fast_expr) = &compiled.fast_expr {
            let prefix = &tentative_steps[..tentative_steps.len() - 1];
            compiled.prefix_cache.sync_to_prefix(
                self.attempt_gen,
                prefix,
                prefix_head,
                prefix_prev_head,
                &compiled.slots,
                &compiled.fast_args,
            );
            for (i, (slot, fargs)) in
                compiled.slots.iter().zip(&compiled.fast_args).enumerate()
            {
                self.fast_slot_buf[i] = eval_slot_fast_cached(
                    tentative_steps,
                    cand,
                    candidate_match_number,
                    self.part_start,
                    self.part_end,
                    slot,
                    fargs,
                    &compiled.prefix_cache,
                    i,
                );
            }
            return Ok(matches!(
                fast_expr.eval(&self.fast_slot_buf[..compiled.slots.len()]),
                FastVal::Bool(true)
            ));
        }

        let mut slot_arrays: Vec<ArrayRef> = Vec::with_capacity(compiled.slots.len());
        for (slot, pre_args) in compiled.slots.iter().zip(&compiled.precomputed_args) {
            let scalar = eval_single_slot_scalar(
                self.partition_batch,
                tentative_steps,
                tentative_steps,
                candidate_match_number,
                current_row,
                self.part_start,
                self.part_end,
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
        if evaluated.data_type() == &DataType::Null {
            return Ok(false);
        }
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

#[expect(clippy::too_many_arguments)]
fn eval_single_slot_scalar(
    partition_batch: &RecordBatch,
    horizon: &[MatchStep],
    full_steps: &[MatchStep],
    match_number: i64,
    current_row: usize,
    part_start: usize,
    part_end: usize,
    program: &Program,
    input_schema: &Schema,
    slot: &PhysicalMeasureSlot,
    precomputed_args: &[ArrayRef],
) -> Result<ScalarValue> {
    match &slot.kind {
        PhysicalMeasureSlotKind::MatchNumber => {
            Ok(ScalarValue::Int64(Some(match_number)))
        }
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
            inner_expr,
            uses_match_metadata,
            ..
        } => {
            let Some(base_row) = navigate_logical_row(
                horizon,
                *logical_nav,
                *logical_offset as usize,
                symbol_set.as_ref(),
            ) else {
                return ScalarValue::try_from(&slot.data_type);
            };
            let Some(final_row) = navigate_physical_in_range(
                base_row,
                *physical_nav,
                *physical_offset as usize,
                part_start,
                part_end,
            ) else {
                return ScalarValue::try_from(&slot.data_type);
            };
            if *uses_match_metadata {
                let cls = classifier_at_partition_row(full_steps, final_row, program);
                eval_metadata_nav_scalar(
                    partition_batch,
                    final_row,
                    cls.as_deref(),
                    match_number,
                    inner_expr,
                )
            } else {
                let arr = &precomputed_args[0];
                ScalarValue::try_from_array(arr.as_ref(), final_row)
            }
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
                    part_start,
                    part_end,
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
                    part_start,
                    part_end,
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
#[expect(clippy::too_many_arguments)]
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

/// Returns the classifier label mapped to `partition_row` within `full_steps`, or `None` if
/// `partition_row` lies outside the match.
fn classifier_at_partition_row(
    full_steps: &[MatchStep],
    partition_row: usize,
    program: &Program,
) -> Option<String> {
    let first_row = full_steps.first()?.row_idx;
    let rel = partition_row.checked_sub(first_row)?;
    let step = full_steps.get(rel)?;
    Some(program.id_to_symbol[step.sym as usize].clone())
}

/// Evaluates a navigation expression referencing `CLASSIFIER([sym])` or `MATCH_NUMBER()` on a
/// single navigated partition row.
fn eval_metadata_nav_scalar(
    partition_batch: &RecordBatch,
    row_idx: usize,
    classifier: Option<&str>,
    match_number: i64,
    inner_expr: &Arc<dyn PhysicalExpr>,
) -> Result<ScalarValue> {
    let ext_schema = Arc::new(build_agg_metadata_arrow_schema(
        partition_batch.schema().as_ref(),
    ));
    let take_idx = UInt32Array::from(vec![row_idx as u32]);
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(ext_schema.fields().len());
    for col in partition_batch.columns() {
        cols.push(compute::take(col.as_ref(), &take_idx, None)?);
    }
    let mut cls_builder = StringBuilder::with_capacity(1, 16);
    match classifier {
        Some(s) => cls_builder.append_value(s),
        None => cls_builder.append_null(),
    }
    let mut mn_builder = Int64Builder::with_capacity(1);
    mn_builder.append_value(match_number);
    cols.push(Arc::new(cls_builder.finish()));
    cols.push(Arc::new(mn_builder.finish()));

    let row_batch = RecordBatch::try_new_with_options(
        ext_schema,
        cols,
        &RecordBatchOptions::new().with_row_count(Some(1)),
    )?;
    let arr = inner_expr.evaluate(&row_batch)?.into_array(1)?;
    ScalarValue::try_from_array(arr.as_ref(), 0)
}

/// Evaluates an aggregate whose arguments reference runtime match metadata (`CLASSIFIER()` or
/// `MATCH_NUMBER()`), such as `ARRAY_AGG(CLASSIFIER())` or `SUM(MATCH_NUMBER() * 100)`.
#[expect(clippy::too_many_arguments)]
fn eval_metadata_agg_scalar(
    partition_batch: &RecordBatch,
    horizon: &[MatchStep],
    match_number: i64,
    part_start: usize,
    part_end: usize,
    program: &Program,
    agg_kind: &AggKind,
    symbol_set: Option<&SymbolSet>,
    arg_exprs: &[Arc<dyn PhysicalExpr>],
    distinct: bool,
    arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    result_type: &DataType,
) -> Result<ScalarValue> {
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
                Some((kind, off)) => navigate_physical_in_range(
                    s.row_idx,
                    Some(kind),
                    off as usize,
                    part_start,
                    part_end,
                ),
            };
            Some((row_idx, s.sym))
        })
        .collect();

    let n = matched_steps.len();
    let ext_schema = Arc::new(build_agg_metadata_arrow_schema(
        partition_batch.schema().as_ref(),
    ));
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(ext_schema.fields().len());
    let take_idx =
        UInt32Array::from_iter(matched_steps.iter().map(|(r, _)| r.map(|i| i as u32)));
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
        0,
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

#[expect(clippy::too_many_arguments)]
fn eval_single_agg_scalar(
    horizon: &[MatchStep],
    part_start: usize,
    part_end: usize,
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
            let count = matching_row_indices_in_range(
                horizon, symbol_set, None, part_start, part_end,
            )
            .flatten()
            .count();
            Ok(ScalarValue::Int64(Some(count as i64)))
        }
        AggKind::Count => {
            let arr = &precomputed_args[0];
            if distinct {
                let mut seen: HashSet<ScalarValue> = HashSet::new();
                for maybe_idx in matching_row_indices_in_range(
                    horizon,
                    symbol_set,
                    arg_physical_nav,
                    part_start,
                    part_end,
                ) {
                    if let Some(idx) = maybe_idx
                        && arr.is_valid(idx)
                    {
                        seen.insert(ScalarValue::try_from_array(arr.as_ref(), idx)?);
                    }
                }
                Ok(ScalarValue::Int64(Some(seen.len() as i64)))
            } else {
                let count = matching_row_indices_in_range(
                    horizon,
                    symbol_set,
                    arg_physical_nav,
                    part_start,
                    part_end,
                )
                .filter(|maybe_idx| maybe_idx.is_some_and(|idx| arr.is_valid(idx)))
                .count();
                Ok(ScalarValue::Int64(Some(count as i64)))
            }
        }
        AggKind::Avg => {
            let arr = &precomputed_args[0];
            let f64_arr =
                arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!(
                        "Expected Float64Array for AVG"
                    )
                })?;
            let mut sum = 0.0f64;
            let mut count = 0u64;
            if distinct {
                let mut seen: HashSet<u64> = HashSet::new();
                for maybe_idx in matching_row_indices_in_range(
                    horizon,
                    symbol_set,
                    arg_physical_nav,
                    part_start,
                    part_end,
                ) {
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
                for maybe_idx in matching_row_indices_in_range(
                    horizon,
                    symbol_set,
                    arg_physical_nav,
                    part_start,
                    part_end,
                ) {
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
            let valid_indices = matching_row_indices_in_range(
                horizon,
                symbol_set,
                arg_physical_nav,
                part_start,
                part_end,
            )
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
            for idx in matching_row_indices_in_range(
                horizon,
                symbol_set,
                arg_physical_nav,
                part_start,
                part_end,
            )
            .flatten()
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
                .map(|(i, a)| {
                    Arc::new(Field::new(format!("arg_{i}"), a.data_type().clone(), true))
                })
                .collect();
            let coerced_fields =
                datafusion_expr::type_coercion::functions::fields_with_udf(
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

            let take_indices: Vec<Option<u32>> = matching_row_indices_in_range(
                horizon,
                symbol_set,
                arg_physical_nav,
                part_start,
                part_end,
            )
            .map(|opt| opt.map(|i| i as u32))
            .collect();
            let return_field =
                Arc::new(Field::new(udaf.name(), result_type.clone(), true));
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

/// Matched events belonging to one logical partition `[part_start..part_end)` inside a
/// shared [`RecordBatch`].
pub struct PartitionEventsSlice<'a> {
    pub part_start: usize,
    pub part_end: usize,
    pub events: &'a [PartitionEvent],
}

#[derive(Debug, Clone, Copy)]
enum EmitRow<'a> {
    Matched {
        match_id: usize,
        match_number: i64,
        steps: &'a [MatchStep],
        step_pos: Option<usize>,
        source_row: usize,
        part_start: usize,
        part_end: usize,
        is_empty_match: bool,
    },
    Unmatched {
        source_row: usize,
    },
}

/// Builds a single coalesced output [`RecordBatch`] from a zero-allocation [`FlatMatchPool`].
#[expect(clippy::too_many_arguments)]
pub fn build_flat_pool_output_batch(
    batch: &RecordBatch,
    pool: &FlatMatchPool,
    program: &Program,
    partition_by_exprs: &[Arc<dyn PhysicalExpr>],
    partition_by_input_indices: &[Option<usize>],
    order_by_input_indices: &[Option<usize>],
    measures: &[PhysicalMeasure],
    rows_per_match: RowsPerMatch,
    output_schema: &SchemaRef,
    output_projection: Option<&[String]>,
) -> Result<Option<RecordBatch>> {
    if pool.events.is_empty() {
        return Ok(None);
    }
    let mut emitted: Vec<EmitRow<'_>> = Vec::with_capacity(pool.events.len());
    for (match_id, event) in pool.events.iter().enumerate() {
        match event {
            FlatPartitionEvent::Match(found) => {
                let steps = &pool.steps[found.steps_start..found.steps_end];
                if rows_per_match.is_one_row() {
                    let is_empty = steps.is_empty();
                    let step_pos = if is_empty {
                        None
                    } else {
                        Some(steps.len() - 1)
                    };
                    emitted.push(EmitRow::Matched {
                        match_id,
                        match_number: found.match_number,
                        steps,
                        step_pos,
                        source_row: found.start_row,
                        part_start: found.part_start,
                        part_end: found.part_end,
                        is_empty_match: is_empty,
                    });
                } else if steps.is_empty() {
                    if rows_per_match.show_empty_matches() {
                        emitted.push(EmitRow::Matched {
                            match_id,
                            match_number: found.match_number,
                            steps,
                            step_pos: None,
                            source_row: found.start_row,
                            part_start: found.part_start,
                            part_end: found.part_end,
                            is_empty_match: true,
                        });
                    }
                } else {
                    for (idx, step) in steps.iter().enumerate() {
                        if !step.excluded {
                            emitted.push(EmitRow::Matched {
                                match_id,
                                match_number: found.match_number,
                                steps,
                                step_pos: Some(idx),
                                source_row: step.row_idx,
                                part_start: found.part_start,
                                part_end: found.part_end,
                                is_empty_match: false,
                            });
                        }
                    }
                }
            }
            FlatPartitionEvent::UnmatchedRow { row_idx } => {
                emitted.push(EmitRow::Unmatched {
                    source_row: *row_idx,
                });
            }
        }
    }

    build_output_from_emitted(
        batch,
        &emitted,
        program,
        partition_by_exprs,
        partition_by_input_indices,
        order_by_input_indices,
        measures,
        rows_per_match,
        output_schema,
        output_projection,
    )
}

#[expect(clippy::too_many_arguments)]
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
    let slice = [PartitionEventsSlice {
        part_start: 0,
        part_end: partition_batch.num_rows(),
        events,
    }];
    build_multi_partition_output_batch(
        partition_batch,
        &slice,
        program,
        partition_by_exprs,
        partition_by_input_indices,
        order_by_input_indices,
        measures,
        rows_per_match,
        output_schema,
        None,
    )
}

/// Builds a single coalesced output [`RecordBatch`] covering one or more complete logical
/// partitions that reside inside `batch`, evaluating all projected `MEASURES` and `PARTITION BY`
/// expressions once across the whole batch instead of once per partition.
#[expect(clippy::too_many_arguments)]
pub fn build_multi_partition_output_batch(
    batch: &RecordBatch,
    partitions: &[PartitionEventsSlice<'_>],
    program: &Program,
    partition_by_exprs: &[Arc<dyn PhysicalExpr>],
    partition_by_input_indices: &[Option<usize>],
    order_by_input_indices: &[Option<usize>],
    measures: &[PhysicalMeasure],
    rows_per_match: RowsPerMatch,
    output_schema: &SchemaRef,
    output_projection: Option<&[String]>,
) -> Result<Option<RecordBatch>> {
    let total_events: usize = partitions.iter().map(|p| p.events.len()).sum();
    if total_events == 0 {
        return Ok(None);
    }
    let mut emitted: Vec<EmitRow<'_>> = Vec::with_capacity(total_events);
    let mut match_id = 0usize;
    for part in partitions {
        let part_start = part.part_start;
        let part_end = part.part_end;
        for event in part.events {
            match event {
                PartitionEvent::Match(found) => {
                    let mid = match_id;
                    match_id += 1;
                    let steps = found.steps.as_slice();
                    if rows_per_match.is_one_row() {
                        let is_empty = steps.is_empty();
                        let step_pos = if is_empty {
                            None
                        } else {
                            Some(steps.len() - 1)
                        };
                        emitted.push(EmitRow::Matched {
                            match_id: mid,
                            match_number: found.match_number,
                            steps,
                            step_pos,
                            source_row: found.start_row,
                            part_start,
                            part_end,
                            is_empty_match: is_empty,
                        });
                    } else if steps.is_empty() {
                        if rows_per_match.show_empty_matches() {
                            emitted.push(EmitRow::Matched {
                                match_id: mid,
                                match_number: found.match_number,
                                steps,
                                step_pos: None,
                                source_row: found.start_row,
                                part_start,
                                part_end,
                                is_empty_match: true,
                            });
                        }
                    } else {
                        for (idx, step) in steps.iter().enumerate() {
                            if !step.excluded {
                                emitted.push(EmitRow::Matched {
                                    match_id: mid,
                                    match_number: found.match_number,
                                    steps,
                                    step_pos: Some(idx),
                                    source_row: step.row_idx,
                                    part_start,
                                    part_end,
                                    is_empty_match: false,
                                });
                            }
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
    }

    build_output_from_emitted(
        batch,
        &emitted,
        program,
        partition_by_exprs,
        partition_by_input_indices,
        order_by_input_indices,
        measures,
        rows_per_match,
        output_schema,
        output_projection,
    )
}

#[expect(clippy::too_many_arguments)]
fn build_output_from_emitted(
    batch: &RecordBatch,
    emitted: &[EmitRow<'_>],
    program: &Program,
    partition_by_exprs: &[Arc<dyn PhysicalExpr>],
    partition_by_input_indices: &[Option<usize>],
    order_by_input_indices: &[Option<usize>],
    measures: &[PhysicalMeasure],
    rows_per_match: RowsPerMatch,
    output_schema: &SchemaRef,
    output_projection: Option<&[String]>,
) -> Result<Option<RecordBatch>> {
    if emitted.is_empty() {
        return Ok(None);
    }

    let partition_batch = batch;
    let out_len = emitted.len();
    let unmatched_mask = if rows_per_match.include_unmatched_rows() {
        Some(BooleanArray::from_iter(
            emitted
                .iter()
                .map(|r| Some(matches!(r, EmitRow::Unmatched { .. }))),
        ))
    } else {
        None
    };

    // Fast projected output path: materialize ONLY the columns requested by `output_projection`
    // (skipping unused `compute::take` copies on input columns and unused measure evaluations).
    if let Some(proj_names) = output_projection {
        let mut columns: Vec<ArrayRef> = Vec::with_capacity(proj_names.len());
        let mut source_indices_cache: Option<UInt32Array> = None;
        let input_schema = partition_batch.schema();
        for name in proj_names {
            if let Some(m) = measures.iter().find(|m| m.alias == *name) {
                columns.push(evaluate_measure(
                    partition_batch,
                    emitted,
                    program,
                    m,
                    unmatched_mask.as_ref(),
                )?);
            } else if let Ok(col_idx) = input_schema.index_of(name) {
                let idx_arr = source_indices_cache
                    .get_or_insert_with(|| build_source_indices(emitted));
                let col_arr = partition_batch.column(col_idx);
                columns.push(take_fast(col_arr, idx_arr)?);
            } else {
                // Expression PARTITION BY column
                let mut matched_pb = None;
                for (expr, maybe_idx) in
                    partition_by_exprs.iter().zip(partition_by_input_indices)
                {
                    if maybe_idx.is_none() && expr.to_string() == *name {
                        matched_pb = Some(expr);
                        break;
                    }
                }
                let expr = matched_pb.ok_or_else(|| {
                    internal_datafusion_err!(
                        "Projected MATCH_RECOGNIZE output column '{name}' not found"
                    )
                })?;
                let idx_arr = source_indices_cache
                    .get_or_insert_with(|| build_source_indices(emitted));
                let evaluated = expr
                    .evaluate(partition_batch)?
                    .into_array(partition_batch.num_rows())?;
                columns.push(take_fast(&evaluated, idx_arr)?);
            }
        }
        let out_batch = RecordBatch::try_new_with_options(
            Arc::clone(output_schema),
            columns,
            &RecordBatchOptions::new().with_row_count(Some(out_len)),
        )?;
        return Ok(Some(out_batch));
    }

    let mut columns: Vec<ArrayRef> = Vec::with_capacity(output_schema.fields().len());
    let mut included_input_indices: HashSet<usize> = HashSet::new();
    let source_indices = build_source_indices(emitted);

    for (expr, maybe_idx) in partition_by_exprs.iter().zip(partition_by_input_indices) {
        if let Some(col_idx) = maybe_idx {
            if included_input_indices.insert(*col_idx) {
                let col_arr = partition_batch.column(*col_idx);
                columns.push(take_fast(col_arr, &source_indices)?);
            }
        } else {
            let evaluated = expr
                .evaluate(partition_batch)?
                .into_array(partition_batch.num_rows())?;
            columns.push(take_fast(&evaluated, &source_indices)?);
        }
    }

    if !rows_per_match.is_one_row() {
        for maybe_idx in order_by_input_indices {
            if let Some(col_idx) = maybe_idx
                && included_input_indices.insert(*col_idx)
            {
                let col_arr = partition_batch.column(*col_idx);
                columns.push(take_fast(col_arr, &source_indices)?);
            }
        }
    }

    for measure in measures {
        let arr = evaluate_measure(
            partition_batch,
            emitted,
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
                columns.push(take_fast(col_arr, &source_indices)?);
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

fn build_source_indices(emitted: &[EmitRow<'_>]) -> UInt32Array {
    let mut v = Vec::with_capacity(emitted.len());
    for r in emitted {
        let row = match r {
            EmitRow::Matched { source_row, .. } | EmitRow::Unmatched { source_row } => {
                *source_row as u32
            }
        };
        v.push(row);
    }
    UInt32Array::from(v)
}

/// Fast primitive gather bypassing Arrow's generic `compute::take` dispatch for non-null
/// `Int64Array` / `Float64Array` columns.
#[inline]
fn take_fast(arr: &ArrayRef, indices: &UInt32Array) -> Result<ArrayRef> {
    if arr.null_count() == 0 && indices.null_count() == 0 {
        if let Some(i64_arr) = arr.as_any().downcast_ref::<Int64Array>() {
            let src = i64_arr.values();
            let mut out = Vec::with_capacity(indices.len());
            for &idx in indices.values() {
                out.push(src[idx as usize]);
            }
            return Ok(Arc::new(Int64Array::from(out)));
        }
        if let Some(f64_arr) = arr.as_any().downcast_ref::<Float64Array>() {
            let src = f64_arr.values();
            let mut out = Vec::with_capacity(indices.len());
            for &idx in indices.values() {
                out.push(src[idx as usize]);
            }
            return Ok(Arc::new(Float64Array::from(out)));
        }
    }
    Ok(compute::take(arr.as_ref(), indices, None)?)
}

fn evaluate_measure(
    partition_batch: &RecordBatch,
    emitted: &[EmitRow<'_>],
    program: &Program,
    measure: &PhysicalMeasure,
    unmatched_mask: Option<&BooleanArray>,
) -> Result<ArrayRef> {
    let out_len = emitted.len();
    // Single-slot direct Column(0) shortcut: avoids building a temporary slot RecordBatch
    // and dispatching outer_expr.evaluate() for measures like `SUM(price)` or `LAST(down.price)`.
    let mut result = if measure.slots.len() == 1
        && measure
            .outer_expr
            .downcast_ref::<Column>()
            .is_some_and(|c| c.index() == 0)
    {
        evaluate_slot(partition_batch, emitted, program, &measure.slots[0])?
    } else {
        let mut slot_arrays = Vec::with_capacity(measure.slots.len());
        for slot in &measure.slots {
            slot_arrays.push(evaluate_slot(partition_batch, emitted, program, slot)?);
        }

        let slot_batch = RecordBatch::try_new_with_options(
            Arc::clone(&measure.slot_schema),
            slot_arrays,
            &RecordBatchOptions::new().with_row_count(Some(out_len)),
        )?;

        measure
            .outer_expr
            .evaluate(&slot_batch)?
            .into_array(out_len)?
    };

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
                    EmitRow::Matched { match_number, .. } => {
                        builder.append_value(*match_number)
                    }
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
                        steps,
                        step_pos,
                        is_empty_match,
                        ..
                    } => {
                        if *is_empty_match {
                            builder.append_null();
                        } else {
                            let horizon = horizon_steps(steps, *step_pos, slot.semantics);
                            match eval_classifier_on_horizon(
                                horizon,
                                steps,
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
            let evaluated = eval_expr_or_col(inner_expr, partition_batch, num_part_rows)?;
            let mut idx_vec = Vec::with_capacity(out_len);
            for row in emitted {
                let idx = match row {
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
                };
                idx_vec.push(idx);
            }
            let indices = UInt32Array::from(idx_vec);
            take_fast(&evaluated, &indices)
        }
        PhysicalMeasureSlotKind::Nav {
            logical_nav,
            logical_offset,
            physical_nav,
            physical_offset,
            symbol_set,
            inner_expr,
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
                            match_number,
                            steps,
                            step_pos,
                            part_start,
                            part_end,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                scalars.push(ScalarValue::try_from(&slot.data_type)?);
                            } else {
                                let horizon =
                                    horizon_steps(steps, *step_pos, slot.semantics);
                                let target = navigate_logical_row(
                                    horizon,
                                    *logical_nav,
                                    *logical_offset as usize,
                                    symbol_set.as_ref(),
                                )
                                .and_then(|base| {
                                    navigate_physical_in_range(
                                        base,
                                        *physical_nav,
                                        *physical_offset as usize,
                                        *part_start,
                                        *part_end,
                                    )
                                });
                                match target {
                                    None => scalars
                                        .push(ScalarValue::try_from(&slot.data_type)?),
                                    Some(final_row) => {
                                        let cls = classifier_at_partition_row(
                                            steps, final_row, program,
                                        );
                                        scalars.push(eval_metadata_nav_scalar(
                                            partition_batch,
                                            final_row,
                                            cls.as_deref(),
                                            *match_number,
                                            inner_expr,
                                        )?);
                                    }
                                }
                            }
                        }
                    }
                }
                ScalarValue::iter_to_array(scalars)
            } else {
                let evaluated =
                    eval_expr_or_col(inner_expr, partition_batch, num_part_rows)?;
                let mut idx_vec = Vec::with_capacity(out_len);
                for row in emitted {
                    let idx = match row {
                        EmitRow::Unmatched { .. } => None,
                        EmitRow::Matched {
                            steps,
                            step_pos,
                            part_start,
                            part_end,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                None
                            } else {
                                let horizon =
                                    horizon_steps(steps, *step_pos, slot.semantics);
                                navigate_logical_row(
                                    horizon,
                                    *logical_nav,
                                    *logical_offset as usize,
                                    symbol_set.as_ref(),
                                )
                                .and_then(|base| {
                                    navigate_physical_in_range(
                                        base,
                                        *physical_nav,
                                        *physical_offset as usize,
                                        *part_start,
                                        *part_end,
                                    )
                                })
                                .map(|r| r as u32)
                            }
                        }
                    };
                    idx_vec.push(idx);
                }
                let indices = UInt32Array::from(idx_vec);
                take_fast(&evaluated, &indices)
            }
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
                            match_number,
                            steps,
                            step_pos,
                            part_start,
                            part_end,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                if matches!(agg_kind, AggKind::Count) {
                                    scalars.push(ScalarValue::Int64(Some(0)));
                                } else if matches!(agg_kind, AggKind::Udaf(_)) {
                                    scalars.push(eval_metadata_agg_scalar(
                                        partition_batch,
                                        &[],
                                        *match_number,
                                        *part_start,
                                        *part_end,
                                        program,
                                        agg_kind,
                                        symbol_set.as_ref(),
                                        args,
                                        *distinct,
                                        *arg_physical_nav,
                                        &slot.data_type,
                                    )?);
                                } else {
                                    scalars.push(ScalarValue::try_from(&slot.data_type)?);
                                }
                            } else {
                                let horizon =
                                    horizon_steps(steps, *step_pos, slot.semantics);
                                scalars.push(eval_metadata_agg_scalar(
                                    partition_batch,
                                    horizon,
                                    *match_number,
                                    *part_start,
                                    *part_end,
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
    steps: &[MatchStep],
    step_pos: Option<usize>,
    semantics: EvalSemantics,
) -> &[MatchStep] {
    match semantics {
        EvalSemantics::Final => steps,
        EvalSemantics::Running => match step_pos {
            Some(idx) => &steps[..=idx],
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

#[inline]
fn navigate_physical_in_range(
    base_row: usize,
    physical_nav: Option<PhysicalNavKind>,
    offset: usize,
    part_start: usize,
    part_end: usize,
) -> Option<usize> {
    match physical_nav {
        None => Some(base_row),
        Some(PhysicalNavKind::Prev) => {
            let target = base_row.checked_sub(offset)?;
            if target >= part_start {
                Some(target)
            } else {
                None
            }
        }
        Some(PhysicalNavKind::Next) => {
            let target = base_row.checked_add(offset)?;
            if target < part_end {
                Some(target)
            } else {
                None
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
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
                        steps,
                        step_pos,
                        is_empty_match,
                        ..
                    } => {
                        if *is_empty_match {
                            builder.append_value(0);
                        } else {
                            let horizon = horizon_steps(steps, *step_pos, semantics);
                            let count = match symbol_set {
                                None => horizon.len(),
                                Some(set) => {
                                    horizon.iter().filter(|s| set.contains(s.sym)).count()
                                }
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

    let mut prepared_args = Vec::with_capacity(arg_exprs.len());
    for (i, expr) in arg_exprs.iter().enumerate() {
        let raw = eval_expr_or_col(expr, partition_batch, num_part_rows)?;
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

    // Fast path for non-distinct COUNT, SUM (Int64/Float64), and AVG without physical shift,
    // using O(1) incremental prefix accumulation when RUNNING progresses monotonically within a
    // match (eliminating O(L^2) rescanning and ScalarValue boxing).
    if !distinct && arg_physical_nav.is_none() {
        match agg_kind {
            AggKind::Count => {
                let evaluated = &prepared_args[0];
                let mut builder = Int64Builder::with_capacity(out_len);
                let mut last_match: Option<usize> = None;
                let mut last_step_pos: usize = 0;
                let mut running_count: i64 = 0;
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => {
                            last_match = None;
                            builder.append_null();
                        }
                        EmitRow::Matched {
                            match_id,
                            steps,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                last_match = None;
                                builder.append_value(0);
                            } else if semantics == EvalSemantics::Running
                                && let Some(pos) = *step_pos
                            {
                                let start_idx = if last_match == Some(*match_id)
                                    && pos >= last_step_pos
                                {
                                    last_step_pos + 1
                                } else {
                                    running_count = 0;
                                    0
                                };
                                for s in &steps[start_idx..=pos] {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && evaluated.is_valid(s.row_idx)
                                    {
                                        running_count += 1;
                                    }
                                }
                                last_match = Some(*match_id);
                                last_step_pos = pos;
                                builder.append_value(running_count);
                            } else {
                                last_match = None;
                                let horizon = horizon_steps(steps, *step_pos, semantics);
                                let mut count = 0i64;
                                for s in horizon {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && evaluated.is_valid(s.row_idx)
                                    {
                                        count += 1;
                                    }
                                }
                                builder.append_value(count);
                            }
                        }
                    }
                }
                return Ok(Arc::new(builder.finish()));
            }
            AggKind::Sum if result_type == &DataType::Int64 => {
                let i64_arr = prepared_args[0]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::exec_datafusion_err!("Expected Int64Array")
                    })?;
                let vals = i64_arr.values();
                let has_nulls = i64_arr.null_count() > 0;
                let mut builder = Int64Builder::with_capacity(out_len);
                let mut last_match: Option<usize> = None;
                let mut last_step_pos: usize = 0;
                let mut running_acc: Option<i64> = None;
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => {
                            last_match = None;
                            builder.append_null();
                        }
                        EmitRow::Matched {
                            match_id,
                            steps,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                last_match = None;
                                builder.append_null();
                            } else if semantics == EvalSemantics::Running
                                && let Some(pos) = *step_pos
                            {
                                let start_idx = if last_match == Some(*match_id)
                                    && pos >= last_step_pos
                                {
                                    last_step_pos + 1
                                } else {
                                    running_acc = None;
                                    0
                                };
                                for s in &steps[start_idx..=pos] {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && (!has_nulls || i64_arr.is_valid(s.row_idx))
                                    {
                                        running_acc = Some(
                                            running_acc
                                                .unwrap_or(0)
                                                .wrapping_add(vals[s.row_idx]),
                                        );
                                    }
                                }
                                last_match = Some(*match_id);
                                last_step_pos = pos;
                                builder.append_option(running_acc);
                            } else {
                                last_match = None;
                                let horizon = horizon_steps(steps, *step_pos, semantics);
                                let mut acc: Option<i64> = None;
                                for s in horizon {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && (!has_nulls || i64_arr.is_valid(s.row_idx))
                                    {
                                        acc = Some(
                                            acc.unwrap_or(0)
                                                .wrapping_add(vals[s.row_idx]),
                                        );
                                    }
                                }
                                builder.append_option(acc);
                            }
                        }
                    }
                }
                return Ok(Arc::new(builder.finish()));
            }
            AggKind::Sum if result_type == &DataType::Float64 => {
                let f64_arr = prepared_args[0]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::exec_datafusion_err!("Expected Float64Array")
                    })?;
                let vals = f64_arr.values();
                let has_nulls = f64_arr.null_count() > 0;
                let mut builder = Float64Builder::with_capacity(out_len);
                let mut last_match: Option<usize> = None;
                let mut last_step_pos: usize = 0;
                let mut running_acc: Option<f64> = None;
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => {
                            last_match = None;
                            builder.append_null();
                        }
                        EmitRow::Matched {
                            match_id,
                            steps,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                last_match = None;
                                builder.append_null();
                            } else if semantics == EvalSemantics::Running
                                && let Some(pos) = *step_pos
                            {
                                let start_idx = if last_match == Some(*match_id)
                                    && pos >= last_step_pos
                                {
                                    last_step_pos + 1
                                } else {
                                    running_acc = None;
                                    0
                                };
                                for s in &steps[start_idx..=pos] {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && (!has_nulls || f64_arr.is_valid(s.row_idx))
                                    {
                                        running_acc = Some(
                                            running_acc.unwrap_or(0.0) + vals[s.row_idx],
                                        );
                                    }
                                }
                                last_match = Some(*match_id);
                                last_step_pos = pos;
                                builder.append_option(running_acc);
                            } else {
                                last_match = None;
                                let horizon = horizon_steps(steps, *step_pos, semantics);
                                let mut acc: Option<f64> = None;
                                for s in horizon {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && (!has_nulls || f64_arr.is_valid(s.row_idx))
                                    {
                                        acc = Some(acc.unwrap_or(0.0) + vals[s.row_idx]);
                                    }
                                }
                                builder.append_option(acc);
                            }
                        }
                    }
                }
                return Ok(Arc::new(builder.finish()));
            }
            AggKind::Avg => {
                let f64_arr = prepared_args[0]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::exec_datafusion_err!(
                            "Expected Float64Array for AVG"
                        )
                    })?;
                let vals = f64_arr.values();
                let has_nulls = f64_arr.null_count() > 0;
                let mut builder = Float64Builder::with_capacity(out_len);
                let mut last_match: Option<usize> = None;
                let mut last_step_pos: usize = 0;
                let mut running_sum = 0.0f64;
                let mut running_cnt = 0u64;
                for row in emitted {
                    match row {
                        EmitRow::Unmatched { .. } => {
                            last_match = None;
                            builder.append_null();
                        }
                        EmitRow::Matched {
                            match_id,
                            steps,
                            step_pos,
                            is_empty_match,
                            ..
                        } => {
                            if *is_empty_match {
                                last_match = None;
                                builder.append_null();
                            } else if semantics == EvalSemantics::Running
                                && let Some(pos) = *step_pos
                            {
                                let start_idx = if last_match == Some(*match_id)
                                    && pos >= last_step_pos
                                {
                                    last_step_pos + 1
                                } else {
                                    running_sum = 0.0;
                                    running_cnt = 0;
                                    0
                                };
                                for s in &steps[start_idx..=pos] {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && (!has_nulls || f64_arr.is_valid(s.row_idx))
                                    {
                                        running_sum += vals[s.row_idx];
                                        running_cnt += 1;
                                    }
                                }
                                last_match = Some(*match_id);
                                last_step_pos = pos;
                                if running_cnt == 0 {
                                    builder.append_null();
                                } else {
                                    builder
                                        .append_value(running_sum / running_cnt as f64);
                                }
                            } else {
                                last_match = None;
                                let horizon = horizon_steps(steps, *step_pos, semantics);
                                let mut sum = 0.0f64;
                                let mut count = 0u64;
                                for s in horizon {
                                    if symbol_set.is_none_or(|set| set.contains(s.sym))
                                        && (!has_nulls || f64_arr.is_valid(s.row_idx))
                                    {
                                        sum += vals[s.row_idx];
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
                steps,
                step_pos,
                part_start,
                part_end,
                is_empty_match,
                ..
            } => {
                if *is_empty_match {
                    if matches!(agg_kind, AggKind::Count) {
                        scalars.push(ScalarValue::Int64(Some(0)));
                    } else if matches!(agg_kind, AggKind::Udaf(_)) {
                        scalars.push(eval_single_agg_scalar(
                            &[],
                            *part_start,
                            *part_end,
                            input_schema.as_ref(),
                            agg_kind,
                            symbol_set,
                            arg_exprs,
                            distinct,
                            arg_physical_nav,
                            &prepared_args,
                            result_type,
                        )?);
                    } else {
                        scalars.push(ScalarValue::try_from(scalar_dt)?);
                    }
                } else {
                    let horizon = horizon_steps(steps, *step_pos, semantics);
                    scalars.push(eval_single_agg_scalar(
                        horizon,
                        *part_start,
                        *part_end,
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

fn matching_row_indices_in_range<'a>(
    horizon: &'a [MatchStep],
    symbol_set: Option<&'a SymbolSet>,
    arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    part_start: usize,
    part_end: usize,
) -> impl Iterator<Item = Option<usize>> + 'a {
    horizon.iter().filter_map(move |s| {
        if let Some(set) = symbol_set
            && !set.contains(s.sym)
        {
            return None;
        }
        let shifted = match arg_physical_nav {
            None => Some(s.row_idx),
            Some((kind, off)) => navigate_physical_in_range(
                s.row_idx,
                Some(kind),
                off as usize,
                part_start,
                part_end,
            ),
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
            let a = arr.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                datafusion_common::exec_datafusion_err!("Expected Int64Array")
            })?;
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
                .ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!("Expected UInt64Array")
                })?;
            let mut acc: Option<u64> = None;
            for idx in indices {
                if a.is_valid(idx) {
                    acc = Some(acc.unwrap_or(0).wrapping_add(a.value(idx)));
                }
            }
            Ok(ScalarValue::UInt64(acc))
        }
        DataType::Float64 => {
            let a = arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                datafusion_common::exec_datafusion_err!("Expected Float64Array")
            })?;
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
            let a = arr.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                datafusion_common::exec_datafusion_err!("Expected Int64Array")
            })?;
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
                .ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!("Expected UInt64Array")
                })?;
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
            let a = arr.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                datafusion_common::exec_datafusion_err!("Expected Float64Array")
            })?;
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
