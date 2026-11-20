// SPDX-License-Identifier: Apache-2.0

//! Hybrid priority-ordered PikeVM row pattern matcher.
//!
//! Supports two execution tiers automatically:
//! - **Tier 1 (all symbols stateless):** O(N * |states|) per match attempt via per-position
//!   `visited[pc]` thread deduplication plus O(1) start-symbol bitmask fast skipping.
//! - **Tier 2 (one or more stateful `DEFINE` symbols):** Stateless symbols still resolve in O(1)
//!   from the precomputed bitmask; stateful symbols invoke [`StatefulPredicateEvaluator`] lazily
//!   on candidate transitions with per-thread epsilon-cycle detection and configurable thread
//!   limits (`max_active_threads`).

use datafusion_common::{Result, exec_err, internal_datafusion_err};

use crate::ast::{AfterMatchSkip, RowsPerMatch};
use crate::pattern::{Inst, Program, SymbolId, SymbolSet};

/// Compact per-row representation of which stateless pattern symbols hold on each row.
#[derive(Debug, Clone)]
pub enum PartitionSymbols {
    /// Fast path for up to 64 distinct symbols: bit `sym_id` indicates whether `sym_id` holds.
    Bitmask64 { rows: Vec<u64>, num_symbols: usize },
    /// Fallback for partitions with more than 64 distinct symbols.
    Bitset {
        words_per_row: usize,
        words: Vec<u64>,
        num_rows: usize,
        num_symbols: usize,
    },
}

impl PartitionSymbols {
    /// Creates a `PartitionSymbols` directly from per-row `u64` bitmasks.
    pub fn from_bitmasks(rows: Vec<u64>, num_symbols: usize) -> Self {
        debug_assert!(num_symbols <= 64);
        Self::Bitmask64 { rows, num_symbols }
    }

    /// Creates a `PartitionSymbols` from per-symbol boolean columns (`columns[sym_id][row]`).
    pub fn from_columns(columns: &[Vec<bool>], num_rows: usize) -> Self {
        let num_symbols = columns.len();
        if num_symbols <= 64 {
            let mut rows = vec![0u64; num_rows];
            for (sym, col) in columns.iter().enumerate() {
                let bit = 1u64 << sym;
                for (row_idx, &matched) in col.iter().enumerate() {
                    if matched {
                        rows[row_idx] |= bit;
                    }
                }
            }
            Self::Bitmask64 { rows, num_symbols }
        } else {
            let words_per_row = num_symbols.div_ceil(64);
            let mut words = vec![0u64; num_rows * words_per_row];
            for (sym, col) in columns.iter().enumerate() {
                let word_offset = sym / 64;
                let bit = 1u64 << (sym % 64);
                for (row_idx, &matched) in col.iter().enumerate() {
                    if matched {
                        words[row_idx * words_per_row + word_offset] |= bit;
                    }
                }
            }
            Self::Bitset {
                words_per_row,
                words,
                num_rows,
                num_symbols,
            }
        }
    }

    /// Returns the number of rows in the partition.
    #[inline]
    pub fn num_rows(&self) -> usize {
        match self {
            PartitionSymbols::Bitmask64 { rows, .. } => rows.len(),
            PartitionSymbols::Bitset { num_rows, .. } => *num_rows,
        }
    }

    /// Returns true if symbol `sym` holds at row `row_idx`.
    #[inline]
    pub fn matches(&self, row_idx: usize, sym: SymbolId) -> bool {
        let sym_idx = sym as usize;
        match self {
            PartitionSymbols::Bitmask64 { rows, num_symbols } => {
                debug_assert!(sym_idx < *num_symbols);
                (rows[row_idx] & (1u64 << sym_idx)) != 0
            }
            PartitionSymbols::Bitset {
                words_per_row,
                words,
                num_symbols,
                ..
            } => {
                debug_assert!(sym_idx < *num_symbols);
                let word = words[row_idx * *words_per_row + (sym_idx / 64)];
                (word & (1u64 << (sym_idx % 64))) != 0
            }
        }
    }

    /// Returns the raw `u64` bitmask at `row_idx` when using the `Bitmask64` fast path.
    #[inline]
    pub fn row_bitmask64(&self, row_idx: usize) -> Option<u64> {
        match self {
            PartitionSymbols::Bitmask64 { rows, .. } => Some(rows[row_idx]),
            PartitionSymbols::Bitset { .. } => None,
        }
    }

    /// Estimated heap bytes held by this symbol table.
    pub fn memory_bytes(&self) -> usize {
        match self {
            PartitionSymbols::Bitmask64 { rows, .. } => {
                rows.capacity() * size_of::<u64>()
            }
            PartitionSymbols::Bitset { words, .. } => words.capacity() * size_of::<u64>(),
        }
    }
}

/// Callback trait for evaluating Tier 2 stateful `DEFINE` predicates that depend on the current
/// candidate match history (e.g. `PREV(b.price)`, `FIRST(a.ts)`, `AVG(a.x)`).
pub trait StatefulPredicateEvaluator {
    /// Returns true if `sym` has a stateful `DEFINE` predicate.
    fn is_stateful(&self, sym: SymbolId) -> bool;

    /// Returns true if any symbol in the query is stateful.
    fn has_any_stateful(&self) -> bool;

    /// Returns true if `symbols.matches(pos, sym)` holds a necessary row-local pre-filter bit
    /// for stateful symbol `sym` (allowing O(1) bitmask rejection before calling `eval_stateful`).
    #[inline]
    fn has_prefilter(&self, _sym: SymbolId) -> bool {
        false
    }

    /// Evaluates the stateful `DEFINE` predicate for `sym` on `tentative_steps`, where the last
    /// element of `tentative_steps` is the candidate row `(pos, sym)`.
    fn eval_stateful(
        &mut self,
        sym: SymbolId,
        tentative_steps: &[MatchStep],
        candidate_match_number: i64,
    ) -> Result<bool>;

    /// Notification invoked at the start of each match attempt so evaluators maintaining
    /// history-head-keyed prefix caches can invalidate them.
    #[inline]
    fn on_new_match_attempt(&mut self) {}

    /// Evaluates the stateful `DEFINE` predicate with immutable arena history-head identifiers
    /// for `tentative_steps[..len-1]` (`prefix_head` and its parent `prefix_prev_head`), enabling
    /// O(1) incremental prefix-state maintenance across linear match growth.
    #[inline]
    fn eval_stateful_with_history(
        &mut self,
        sym: SymbolId,
        tentative_steps: &[MatchStep],
        candidate_match_number: i64,
        _prefix_head: u32,
        _prefix_prev_head: u32,
    ) -> Result<bool> {
        self.eval_stateful(sym, tentative_steps, candidate_match_number)
    }
}

/// No-op evaluator used when all `DEFINE` symbols are stateless (Tier 1 fast path).
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopStatefulEvaluator;

impl StatefulPredicateEvaluator for NoopStatefulEvaluator {
    #[inline]
    fn is_stateful(&self, _sym: SymbolId) -> bool {
        false
    }

    #[inline]
    fn has_any_stateful(&self) -> bool {
        false
    }

    fn eval_stateful(
        &mut self,
        _sym: SymbolId,
        _tentative_steps: &[MatchStep],
        _candidate_match_number: i64,
    ) -> Result<bool> {
        unreachable!("NoopStatefulEvaluator should never be called")
    }
}

/// One matched row inside a match attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchStep {
    /// 0-based row index within the partition.
    pub row_idx: usize,
    /// Assigned primary pattern variable id.
    pub sym: SymbolId,
    /// True if matched inside `{- ... -}` exclusion syntax.
    pub excluded: bool,
}

/// A single completed match (possibly empty) found within a partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundMatch {
    /// 1-based match number within the partition (`MATCH_NUMBER()`).
    pub match_number: i64,
    /// Partition row index where matching started.
    pub start_row: usize,
    /// Exclusive end row index (`start_row + steps.len()`).
    pub end_row: usize,
    /// Matched steps in ascending row order (empty if this is an empty match).
    pub steps: Vec<MatchStep>,
}

/// An output event emitted while scanning a partition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PartitionEvent {
    /// A completed match (empty or non-empty).
    Match(FoundMatch),
    /// An unmatched row (emitted only under `ALL ROWS PER MATCH WITH UNMATCHED ROWS`).
    UnmatchedRow { row_idx: usize },
}

/// Zero-allocation flat match record indexing into a shared `steps_pool: Vec<MatchStep>`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlatFoundMatch {
    pub match_number: i64,
    pub start_row: usize,
    pub end_row: usize,
    pub steps_start: usize,
    pub steps_end: usize,
    pub part_start: usize,
    pub part_end: usize,
}

/// Flat partition output event avoiding per-match `Vec<MatchStep>` heap allocations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlatPartitionEvent {
    Match(FlatFoundMatch),
    UnmatchedRow { row_idx: usize },
}

/// Batch-wide flat storage for all matched steps and events across all partitions in a batch.
#[derive(Debug, Default)]
pub struct FlatMatchPool {
    pub steps: Vec<MatchStep>,
    pub events: Vec<FlatPartitionEvent>,
}

impl FlatMatchPool {
    pub fn clear(&mut self) {
        self.steps.clear();
        self.events.clear();
    }
}

/// Diagnostic counters collected during matching (used for complexity tests and metrics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatchStats {
    pub attempts: usize,
    pub fast_skipped_rows: usize,
    pub total_thread_steps: usize,
}

/// Sentinel history head representing an empty match prefix.
pub const NO_HISTORY: u32 = u32::MAX;

/// Row walk threshold before activating lazy cross-attempt Tier 1 dead-state memoization.
const DEAD_CACHE_TRIGGER_ROWS: usize = 64;
/// Upper cap on lazy dead-state bitset size (64M bits = 8 MiB).
const DEAD_CACHE_MAX_BITS: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy)]
struct HistoryEntry {
    step: MatchStep,
    prev: u32,
}

#[derive(Debug, Clone, Copy)]
struct Thread {
    pc: usize,
    exclude_depth: u32,
    history_head: u32,
}

/// Reusable scratch buffers for [`Matcher`] so streams avoid allocating per partition.
#[derive(Debug, Default)]
pub struct MatcherBuffers {
    visited: Vec<u32>,
    visit_gen: u32,
    arena: Vec<HistoryEntry>,
    epsilon_stack: Vec<Thread>,
    scratch_steps: Vec<MatchStep>,
    scratch_head: u32,
    current: Vec<Thread>,
    next: Vec<Thread>,
    /// Lazy per-partition Tier 1 dead-state bitset indexed by `(pos - part_start) * num_match_labels + ord`.
    dead_bits: Vec<u64>,
    dead_active: bool,
    attempt_visited_bits: Vec<usize>,
}

impl MatcherBuffers {
    /// Creates reusable buffers sized for `program`.
    pub fn new(program: &Program) -> Self {
        Self {
            visited: vec![0; program.instructions.len()],
            visit_gen: 0,
            arena: Vec::new(),
            epsilon_stack: Vec::new(),
            scratch_steps: Vec::new(),
            scratch_head: NO_HISTORY,
            current: Vec::new(),
            next: Vec::new(),
            dead_bits: Vec::new(),
            dead_active: false,
            attempt_visited_bits: Vec::new(),
        }
    }
}

/// Reusable hybrid PikeVM matcher for a compiled [`Program`].
#[derive(Debug)]
pub struct Matcher<'a> {
    program: &'a Program,
    buf: MatcherBuffers,
    max_active_threads: usize,
}

impl<'a> Matcher<'a> {
    /// Creates a matcher for `program` with default stateful thread limit.
    pub fn new(program: &'a Program) -> Self {
        Self::with_max_active_threads(
            program,
            crate::ast::DEFAULT_MAX_ACTIVE_THREADS as usize,
        )
    }

    /// Creates a matcher for `program` with an explicit active-thread cap for stateful matching.
    pub fn with_max_active_threads(
        program: &'a Program,
        max_active_threads: usize,
    ) -> Self {
        Self {
            buf: MatcherBuffers::new(program),
            program,
            max_active_threads,
        }
    }

    /// Creates a matcher reusing an existing [`MatcherBuffers`] allocation.
    pub fn from_buffers(
        program: &'a Program,
        max_active_threads: usize,
        mut buf: MatcherBuffers,
    ) -> Self {
        if buf.visited.len() != program.instructions.len() {
            buf.visited.resize(program.instructions.len(), 0);
            buf.visit_gen = 0;
        }
        Self {
            program,
            buf,
            max_active_threads,
        }
    }

    /// Returns the underlying reusable buffers for the next partition.
    pub fn into_buffers(self) -> MatcherBuffers {
        self.buf
    }

    /// Scans an entire partition using the Tier 1 stateless fast path.
    pub fn match_partition(
        &mut self,
        symbols: &PartitionSymbols,
        skip: &AfterMatchSkip,
        rows_per_match: RowsPerMatch,
    ) -> Result<(Vec<PartitionEvent>, MatchStats)> {
        let mut noop = NoopStatefulEvaluator;
        self.match_partition_with_evaluator(symbols, skip, rows_per_match, &mut noop)
    }

    /// Scans an entire partition, using Tier 1 when `evaluator.has_any_stateful()` is false and
    /// Tier 2 hybrid stateful matching when true.
    pub fn match_partition_with_evaluator<E: StatefulPredicateEvaluator>(
        &mut self,
        symbols: &PartitionSymbols,
        skip: &AfterMatchSkip,
        rows_per_match: RowsPerMatch,
        evaluator: &mut E,
    ) -> Result<(Vec<PartitionEvent>, MatchStats)> {
        self.match_partition_range_with_evaluator(
            symbols,
            0,
            symbols.num_rows(),
            skip,
            rows_per_match,
            evaluator,
        )
    }

    /// Scans rows `[part_start..part_end)` of a batch-wide [`PartitionSymbols`] table as a single
    /// logical partition. Emitted `MatchStep.row_idx` and `FoundMatch.start_row`/`end_row` use
    /// batch-global row indices within `part_start..part_end`.
    pub fn match_partition_range_with_evaluator<E: StatefulPredicateEvaluator>(
        &mut self,
        symbols: &PartitionSymbols,
        part_start: usize,
        part_end: usize,
        skip: &AfterMatchSkip,
        rows_per_match: RowsPerMatch,
        evaluator: &mut E,
    ) -> Result<(Vec<PartitionEvent>, MatchStats)> {
        let skip_target_set = match skip {
            AfterMatchSkip::ToFirst(name) | AfterMatchSkip::ToLast(name) => {
                let set = self.program.resolve_symbol_set(name).ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!(
                        "AFTER MATCH SKIP references unknown pattern variable '{name}'"
                    )
                })?;
                Some(set)
            }
            AfterMatchSkip::PastLastRow | AfterMatchSkip::ToNextRow => None,
        };

        let has_stateful = evaluator.has_any_stateful();
        let start_mask = self.program.start_symbols_mask.filter(|&mask| {
            !has_stateful
                || (0..self.program.num_symbols()).all(|s| {
                    let sym = s as SymbolId;
                    (mask & (1u64 << s)) == 0
                        || !evaluator.is_stateful(sym)
                        || evaluator.has_prefilter(sym)
                })
        });
        let second_mask = self.program.second_symbols_mask.filter(|&mask| {
            !has_stateful
                || (0..self.program.num_symbols()).all(|s| {
                    let sym = s as SymbolId;
                    (mask & (1u64 << s)) == 0
                        || !evaluator.is_stateful(sym)
                        || evaluator.has_prefilter(sym)
                })
        });

        let mut events = Vec::new();
        let mut stats = MatchStats::default();
        self.buf.dead_active = false;

        // O(N) whole-partition rejection when a mandatory symbol never appears in the partition.
        if part_end.saturating_sub(part_start) >= DEAD_CACHE_TRIGGER_ROWS
            && self.partition_missing_required_symbol(
                symbols,
                part_start,
                part_end,
                has_stateful,
                evaluator,
            )
        {
            stats.fast_skipped_rows += part_end - part_start;
            if rows_per_match.include_unmatched_rows() {
                events.reserve(part_end - part_start);
                for row_idx in part_start..part_end {
                    events.push(PartitionEvent::UnmatchedRow { row_idx });
                }
            }
            return Ok((events, stats));
        }

        let mut match_number: i64 = 0;
        let mut start_row: usize = part_start;
        let mut max_matched_end: usize = part_start;

        while start_row < part_end {
            // O(1) 1-row and 2-row frontier fast skip.
            if let (Some(mask), Some(row_bits)) =
                (start_mask, symbols.row_bitmask64(start_row))
                && (row_bits & mask) == 0
            {
                stats.fast_skipped_rows += 1;
                if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end
                {
                    events.push(PartitionEvent::UnmatchedRow { row_idx: start_row });
                }
                start_row += 1;
                continue;
            }
            if let Some(mask2) = second_mask
                && (start_row + 1 >= part_end
                    || symbols
                        .row_bitmask64(start_row + 1)
                        .is_some_and(|bits2| (bits2 & mask2) == 0))
            {
                stats.fast_skipped_rows += 1;
                if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end
                {
                    events.push(PartitionEvent::UnmatchedRow { row_idx: start_row });
                }
                start_row += 1;
                continue;
            }

            stats.attempts += 1;
            let candidate_mn = match_number + 1;
            let found = if has_stateful {
                self.find_match_at_stateful_in_range(
                    symbols,
                    start_row,
                    part_start,
                    part_end,
                    candidate_mn,
                    evaluator,
                    &mut stats,
                )?
            } else {
                self.find_match_at_in_range(
                    symbols, start_row, part_start, part_end, &mut stats,
                )
            };

            match found {
                Some(steps) => {
                    let end_row = start_row + steps.len();
                    let is_empty = steps.is_empty();
                    match_number = candidate_mn;
                    if !is_empty {
                        max_matched_end = max_matched_end.max(end_row);
                    }

                    let next_start = if is_empty {
                        start_row + 1
                    } else {
                        compute_next_start(
                            start_row,
                            end_row,
                            &steps,
                            skip,
                            skip_target_set.as_ref(),
                        )?
                    };

                    let should_emit = !is_empty || rows_per_match.show_empty_matches();
                    if should_emit {
                        events.push(PartitionEvent::Match(FoundMatch {
                            match_number,
                            start_row,
                            end_row,
                            steps,
                        }));
                    }

                    start_row = next_start;
                }
                None => {
                    if rows_per_match.include_unmatched_rows()
                        && start_row >= max_matched_end
                    {
                        events.push(PartitionEvent::UnmatchedRow { row_idx: start_row });
                    }
                    start_row += 1;
                }
            }
        }

        Ok((events, stats))
    }

    /// Zero-allocation variant of [`Self::match_partition_range_with_evaluator`] that appends
    /// matched steps and events contiguously into `pool` (avoiding per-match `Vec<MatchStep>`
    /// heap allocations across millions of matches).
    #[expect(clippy::too_many_arguments)]
    pub fn match_partition_range_into_pool<E: StatefulPredicateEvaluator>(
        &mut self,
        symbols: &PartitionSymbols,
        part_start: usize,
        part_end: usize,
        skip: &AfterMatchSkip,
        rows_per_match: RowsPerMatch,
        evaluator: &mut E,
        pool: &mut FlatMatchPool,
    ) -> Result<()> {
        let skip_target_set = match skip {
            AfterMatchSkip::ToFirst(name) | AfterMatchSkip::ToLast(name) => {
                let set = self.program.resolve_symbol_set(name).ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!(
                        "AFTER MATCH SKIP references unknown pattern variable '{name}'"
                    )
                })?;
                Some(set)
            }
            AfterMatchSkip::PastLastRow | AfterMatchSkip::ToNextRow => None,
        };

        let has_stateful = evaluator.has_any_stateful();
        let start_mask = self.program.start_symbols_mask.filter(|&mask| {
            !has_stateful
                || (0..self.program.num_symbols()).all(|s| {
                    let sym = s as SymbolId;
                    (mask & (1u64 << s)) == 0
                        || !evaluator.is_stateful(sym)
                        || evaluator.has_prefilter(sym)
                })
        });
        let second_mask = self.program.second_symbols_mask.filter(|&mask| {
            !has_stateful
                || (0..self.program.num_symbols()).all(|s| {
                    let sym = s as SymbolId;
                    (mask & (1u64 << s)) == 0
                        || !evaluator.is_stateful(sym)
                        || evaluator.has_prefilter(sym)
                })
        });

        let mut stats = MatchStats::default();
        self.buf.dead_active = false;

        // O(N) whole-partition rejection when a mandatory symbol never appears in the partition.
        if part_end.saturating_sub(part_start) >= DEAD_CACHE_TRIGGER_ROWS
            && self.partition_missing_required_symbol(
                symbols,
                part_start,
                part_end,
                has_stateful,
                evaluator,
            )
        {
            if rows_per_match.include_unmatched_rows() {
                pool.events.reserve(part_end - part_start);
                for row_idx in part_start..part_end {
                    pool.events
                        .push(FlatPartitionEvent::UnmatchedRow { row_idx });
                }
            }
            return Ok(());
        }

        let mut match_number: i64 = 0;
        let mut start_row: usize = part_start;
        let mut max_matched_end: usize = part_start;

        while start_row < part_end {
            if let (Some(mask), Some(row_bits)) =
                (start_mask, symbols.row_bitmask64(start_row))
                && (row_bits & mask) == 0
            {
                if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end
                {
                    pool.events
                        .push(FlatPartitionEvent::UnmatchedRow { row_idx: start_row });
                }
                start_row += 1;
                continue;
            }
            if let Some(mask2) = second_mask
                && (start_row + 1 >= part_end
                    || symbols
                        .row_bitmask64(start_row + 1)
                        .is_some_and(|bits2| (bits2 & mask2) == 0))
            {
                if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end
                {
                    pool.events
                        .push(FlatPartitionEvent::UnmatchedRow { row_idx: start_row });
                }
                start_row += 1;
                continue;
            }

            let candidate_mn = match_number + 1;
            let best_head = if has_stateful {
                self.find_match_head_stateful_in_range(
                    symbols,
                    start_row,
                    part_start,
                    part_end,
                    candidate_mn,
                    evaluator,
                    &mut stats,
                )?
            } else {
                self.find_match_head_in_range(
                    symbols, start_row, part_start, part_end, &mut stats,
                )
            };

            match best_head {
                Some(head) => {
                    let steps_start = pool.steps.len();
                    materialize_history_into(&self.buf.arena, head, &mut pool.steps);
                    let steps_end = pool.steps.len();
                    let steps_slice = &pool.steps[steps_start..steps_end];
                    let end_row = start_row + steps_slice.len();
                    let is_empty = steps_slice.is_empty();
                    match_number = candidate_mn;
                    if !is_empty {
                        max_matched_end = max_matched_end.max(end_row);
                    }

                    let next_start = if is_empty {
                        start_row + 1
                    } else {
                        compute_next_start(
                            start_row,
                            end_row,
                            steps_slice,
                            skip,
                            skip_target_set.as_ref(),
                        )?
                    };

                    let should_emit = !is_empty || rows_per_match.show_empty_matches();
                    if should_emit {
                        pool.events.push(FlatPartitionEvent::Match(FlatFoundMatch {
                            match_number,
                            start_row,
                            end_row,
                            steps_start,
                            steps_end,
                            part_start,
                            part_end,
                        }));
                    } else {
                        pool.steps.truncate(steps_start);
                    }

                    start_row = next_start;
                }
                None => {
                    if rows_per_match.include_unmatched_rows()
                        && start_row >= max_matched_end
                    {
                        pool.events.push(FlatPartitionEvent::UnmatchedRow {
                            row_idx: start_row,
                        });
                    }
                    start_row += 1;
                }
            }
        }

        Ok(())
    }

    /// Tier 1 stateless fast path: finds the highest-priority SQL:2016 match starting at
    /// `start_row` in O(N * |states|) time using per-position `visited[pc]` deduplication.
    pub fn find_match_at(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        stats: &mut MatchStats,
    ) -> Option<Vec<MatchStep>> {
        self.find_match_at_in_range(symbols, start_row, 0, symbols.num_rows(), stats)
    }

    fn find_match_at_in_range(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        part_start: usize,
        part_end: usize,
        stats: &mut MatchStats,
    ) -> Option<Vec<MatchStep>> {
        self.find_match_head_in_range(symbols, start_row, part_start, part_end, stats)
            .map(|head| materialize_history(&self.buf.arena, head))
    }

    fn partition_missing_required_symbol<E: StatefulPredicateEvaluator>(
        &self,
        symbols: &PartitionSymbols,
        part_start: usize,
        part_end: usize,
        has_stateful: bool,
        evaluator: &E,
    ) -> bool {
        let req = self.program.required_symbols_mask;
        if req == 0 {
            return false;
        }
        let PartitionSymbols::Bitmask64 { rows, .. } = symbols else {
            return false;
        };
        if has_stateful {
            for s in 0..self.program.num_symbols() {
                if (req & (1u64 << s)) != 0 {
                    let sym = s as SymbolId;
                    if evaluator.is_stateful(sym) && !evaluator.has_prefilter(sym) {
                        return false;
                    }
                }
            }
        }
        let mut present = 0u64;
        for &bits in &rows[part_start..part_end] {
            present |= bits;
            if (present & req) == req {
                return false;
            }
        }
        (present & req) != req
    }

    fn find_match_head_in_range(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        part_start: usize,
        part_end: usize,
        stats: &mut MatchStats,
    ) -> Option<u32> {
        self.buf.arena.clear();
        self.buf.current.clear();
        self.buf.next.clear();
        self.buf.attempt_visited_bits.clear();
        let mut best_match: Option<u32> = None;

        self.bump_gen();
        self.add_thread_into_current(
            Thread {
                pc: 0,
                exclude_depth: 0,
                history_head: NO_HISTORY,
            },
            start_row,
            part_start,
            part_end,
            &mut best_match,
            stats,
        );

        let num_ml = self.program.num_match_labels;
        let can_dead_cache = num_ml > 0
            && (part_end - part_start)
                .checked_mul(num_ml)
                .is_some_and(|total_bits| total_bits <= DEAD_CACHE_MAX_BITS);

        let mut pos = start_row;
        while !self.buf.current.is_empty() && pos < part_end {
            if !self.buf.dead_active
                && can_dead_cache
                && best_match.is_none()
                && pos - start_row >= DEAD_CACHE_TRIGGER_ROWS
            {
                let total_bits = (part_end - part_start) * num_ml;
                let words = total_bits.div_ceil(64);
                self.buf.dead_bits.clear();
                self.buf.dead_bits.resize(words, 0);
                self.buf.dead_active = true;
            }

            let record_dead = self.buf.dead_active && best_match.is_none();
            let row_base = (pos - part_start) * num_ml;

            self.buf.next.clear();
            self.bump_gen();

            let cur_len = self.buf.current.len();
            for idx in 0..cur_len {
                let thread = self.buf.current[idx];
                if self.buf.dead_active {
                    let ord = self.program.pc_to_match_ordinal[thread.pc];
                    if ord != u16::MAX {
                        let bit_idx = row_base + ord as usize;
                        if (self.buf.dead_bits[bit_idx >> 6] & (1u64 << (bit_idx & 63)))
                            != 0
                        {
                            continue;
                        }
                        if record_dead {
                            self.buf.attempt_visited_bits.push(bit_idx);
                        }
                    }
                }

                stats.total_thread_steps += 1;
                match self.program.instructions[thread.pc] {
                    Inst::MatchLabel { sym } => {
                        if symbols.matches(pos, sym) {
                            let new_head = self.buf.arena.len() as u32;
                            self.buf.arena.push(HistoryEntry {
                                step: MatchStep {
                                    row_idx: pos,
                                    sym,
                                    excluded: thread.exclude_depth > 0,
                                },
                                prev: thread.history_head,
                            });
                            let hit_accept = self.add_thread_into_next(
                                Thread {
                                    pc: thread.pc + 1,
                                    exclude_depth: thread.exclude_depth,
                                    history_head: new_head,
                                },
                                pos + 1,
                                part_start,
                                part_end,
                                &mut best_match,
                                stats,
                            );
                            if hit_accept {
                                break;
                            }
                        }
                    }
                    other => {
                        unreachable!(
                            "non-consuming instruction {other:?} should have been resolved in epsilon closure"
                        );
                    }
                }
            }

            std::mem::swap(&mut self.buf.current, &mut self.buf.next);
            pos += 1;
        }

        if best_match.is_none() && self.buf.dead_active {
            for &bit_idx in &self.buf.attempt_visited_bits {
                self.buf.dead_bits[bit_idx >> 6] |= 1u64 << (bit_idx & 63);
            }
        }

        best_match
    }

    /// Tier 2 hybrid stateful path: stateless symbols resolve in O(1) from `symbols`; stateful
    /// symbols call `evaluator.eval_stateful(...)` with the thread's tentative match history.
    ///
    /// Uses per-thread epsilon-cycle detection so threads with distinct histories can explore
    /// the same `(pc, pos)` independently while still enforcing strict SQL:2016 priority order.
    pub fn find_match_at_stateful<E: StatefulPredicateEvaluator>(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        candidate_match_number: i64,
        evaluator: &mut E,
        stats: &mut MatchStats,
    ) -> Result<Option<Vec<MatchStep>>> {
        self.find_match_at_stateful_in_range(
            symbols,
            start_row,
            0,
            symbols.num_rows(),
            candidate_match_number,
            evaluator,
            stats,
        )
    }

    #[expect(clippy::too_many_arguments)]
    fn find_match_at_stateful_in_range<E: StatefulPredicateEvaluator>(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        part_start: usize,
        part_end: usize,
        candidate_match_number: i64,
        evaluator: &mut E,
        stats: &mut MatchStats,
    ) -> Result<Option<Vec<MatchStep>>> {
        Ok(self
            .find_match_head_stateful_in_range(
                symbols,
                start_row,
                part_start,
                part_end,
                candidate_match_number,
                evaluator,
                stats,
            )?
            .map(|head| materialize_history(&self.buf.arena, head)))
    }

    #[expect(clippy::too_many_arguments)]
    fn find_match_head_stateful_in_range<E: StatefulPredicateEvaluator>(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        part_start: usize,
        part_end: usize,
        candidate_match_number: i64,
        evaluator: &mut E,
        stats: &mut MatchStats,
    ) -> Result<Option<u32>> {
        self.buf.arena.clear();
        self.buf.current.clear();
        self.buf.next.clear();
        self.buf.scratch_steps.clear();
        self.buf.scratch_head = NO_HISTORY;
        evaluator.on_new_match_attempt();
        let mut best_match: Option<u32> = None;

        self.bump_gen();
        self.add_thread_into_current(
            Thread {
                pc: 0,
                exclude_depth: 0,
                history_head: NO_HISTORY,
            },
            start_row,
            part_start,
            part_end,
            &mut best_match,
            stats,
        );

        let mut pos = start_row;
        while !self.buf.current.is_empty() && pos < part_end {
            self.buf.next.clear();

            let cur_len = self.buf.current.len();
            for idx in 0..cur_len {
                let thread = self.buf.current[idx];
                stats.total_thread_steps += 1;
                match self.program.instructions[thread.pc] {
                    Inst::MatchLabel { sym } => {
                        let matched = if evaluator.is_stateful(sym) {
                            if evaluator.has_prefilter(sym) && !symbols.matches(pos, sym)
                            {
                                false
                            } else {
                                let step = MatchStep {
                                    row_idx: pos,
                                    sym,
                                    excluded: thread.exclude_depth > 0,
                                };
                                let prefix_head = thread.history_head;
                                let prefix_prev_head = if prefix_head == NO_HISTORY {
                                    NO_HISTORY
                                } else {
                                    self.buf.arena[prefix_head as usize].prev
                                };
                                if prefix_head != self.buf.scratch_head {
                                    if prefix_head != NO_HISTORY
                                        && prefix_prev_head == self.buf.scratch_head
                                    {
                                        self.buf.scratch_steps.push(
                                            self.buf.arena[prefix_head as usize].step,
                                        );
                                    } else {
                                        rebuild_prefix_steps(
                                            &self.buf.arena,
                                            prefix_head,
                                            &mut self.buf.scratch_steps,
                                        );
                                    }
                                    self.buf.scratch_head = prefix_head;
                                }
                                self.buf.scratch_steps.push(step);
                                let eval_res = evaluator.eval_stateful_with_history(
                                    sym,
                                    &self.buf.scratch_steps,
                                    candidate_match_number,
                                    prefix_head,
                                    prefix_prev_head,
                                );
                                self.buf.scratch_steps.pop();
                                eval_res?
                            }
                        } else {
                            symbols.matches(pos, sym)
                        };

                        if matched {
                            let new_head = self.buf.arena.len() as u32;
                            self.buf.arena.push(HistoryEntry {
                                step: MatchStep {
                                    row_idx: pos,
                                    sym,
                                    excluded: thread.exclude_depth > 0,
                                },
                                prev: thread.history_head,
                            });

                            // Per-thread visited generation prevents epsilon cycles without
                            // merging distinct stateful histories across different threads.
                            self.bump_gen();
                            let hit_accept = self.add_thread_into_next(
                                Thread {
                                    pc: thread.pc + 1,
                                    exclude_depth: thread.exclude_depth,
                                    history_head: new_head,
                                },
                                pos + 1,
                                part_start,
                                part_end,
                                &mut best_match,
                                stats,
                            );
                            if hit_accept {
                                break;
                            }
                            if self.buf.next.len() > self.max_active_threads {
                                return exec_err!(
                                    "MATCH_RECOGNIZE stateful matching exceeded maximum active thread limit ({})",
                                    self.max_active_threads
                                );
                            }
                        }
                    }
                    other => {
                        unreachable!(
                            "non-consuming instruction {other:?} should have been resolved in epsilon closure"
                        );
                    }
                }
            }

            std::mem::swap(&mut self.buf.current, &mut self.buf.next);
            pos += 1;
        }

        Ok(best_match)
    }

    #[inline]
    fn bump_gen(&mut self) {
        self.buf.visit_gen = self.buf.visit_gen.wrapping_add(1);
        if self.buf.visit_gen == 0 {
            self.buf.visited.fill(0);
            self.buf.visit_gen = 1;
        }
    }

    fn add_thread_into_current(
        &mut self,
        initial: Thread,
        pos: usize,
        part_start: usize,
        part_end: usize,
        best_match: &mut Option<u32>,
        stats: &mut MatchStats,
    ) -> bool {
        add_thread_impl(
            self.program,
            &mut self.buf.visited,
            self.buf.visit_gen,
            &mut self.buf.epsilon_stack,
            initial,
            pos,
            part_start,
            part_end,
            &mut self.buf.current,
            best_match,
            stats,
        )
    }

    fn add_thread_into_next(
        &mut self,
        initial: Thread,
        pos: usize,
        part_start: usize,
        part_end: usize,
        best_match: &mut Option<u32>,
        stats: &mut MatchStats,
    ) -> bool {
        add_thread_impl(
            self.program,
            &mut self.buf.visited,
            self.buf.visit_gen,
            &mut self.buf.epsilon_stack,
            initial,
            pos,
            part_start,
            part_end,
            &mut self.buf.next,
            best_match,
            stats,
        )
    }
}

#[expect(clippy::too_many_arguments)]
fn add_thread_impl(
    program: &Program,
    visited: &mut [u32],
    visit_gen: u32,
    epsilon_stack: &mut Vec<Thread>,
    initial: Thread,
    pos: usize,
    part_start: usize,
    part_end: usize,
    out: &mut Vec<Thread>,
    best_match: &mut Option<u32>,
    stats: &mut MatchStats,
) -> bool {
    epsilon_stack.clear();
    epsilon_stack.push(initial);

    while let Some(thread) = epsilon_stack.pop() {
        let pc = thread.pc;
        if visited[pc] == visit_gen {
            continue;
        }
        visited[pc] = visit_gen;
        stats.total_thread_steps += 1;

        match program.instructions[pc] {
            Inst::MatchLabel { .. } => {
                out.push(thread);
            }
            Inst::Jump(target) => {
                epsilon_stack.push(Thread {
                    pc: target,
                    ..thread
                });
            }
            Inst::Split { first, second } => {
                epsilon_stack.push(Thread {
                    pc: second,
                    ..thread
                });
                epsilon_stack.push(Thread {
                    pc: first,
                    ..thread
                });
            }
            Inst::AnchorStart => {
                if pos == part_start {
                    epsilon_stack.push(Thread {
                        pc: pc + 1,
                        ..thread
                    });
                }
            }
            Inst::AnchorEnd => {
                if pos == part_end {
                    epsilon_stack.push(Thread {
                        pc: pc + 1,
                        ..thread
                    });
                }
            }
            Inst::ExcludeEnter => {
                epsilon_stack.push(Thread {
                    pc: pc + 1,
                    exclude_depth: thread.exclude_depth + 1,
                    ..thread
                });
            }
            Inst::ExcludeExit => {
                epsilon_stack.push(Thread {
                    pc: pc + 1,
                    exclude_depth: thread.exclude_depth.saturating_sub(1),
                    ..thread
                });
            }
            Inst::Accept => {
                *best_match = Some(thread.history_head);
                epsilon_stack.clear();
                return true;
            }
        }
    }

    false
}

fn materialize_history(arena: &[HistoryEntry], head: u32) -> Vec<MatchStep> {
    let mut steps = Vec::new();
    materialize_history_into(arena, head, &mut steps);
    steps
}

#[inline]
fn materialize_history_into(
    arena: &[HistoryEntry],
    mut head: u32,
    out: &mut Vec<MatchStep>,
) {
    let start = out.len();
    while head != NO_HISTORY {
        let entry = arena[head as usize];
        out.push(entry.step);
        head = entry.prev;
    }
    out[start..].reverse();
}

fn rebuild_prefix_steps(arena: &[HistoryEntry], mut head: u32, out: &mut Vec<MatchStep>) {
    out.clear();
    while head != NO_HISTORY {
        let entry = arena[head as usize];
        out.push(entry.step);
        head = entry.prev;
    }
    out.reverse();
}

fn compute_next_start(
    start_row: usize,
    end_row: usize,
    steps: &[MatchStep],
    skip: &AfterMatchSkip,
    skip_target_set: Option<&SymbolSet>,
) -> Result<usize> {
    match skip {
        AfterMatchSkip::PastLastRow => Ok(end_row),
        AfterMatchSkip::ToNextRow => Ok(start_row + 1),
        AfterMatchSkip::ToFirst(name) => {
            let target = skip_target_set.ok_or_else(|| {
                internal_datafusion_err!("missing validated skip target '{name}'")
            })?;
            let Some(step) = steps.iter().find(|s| target.contains(s.sym)) else {
                return exec_err!(
                    "AFTER MATCH SKIP TO FIRST {name} failed: variable '{name}' was not matched"
                );
            };
            if step.row_idx == start_row {
                return exec_err!(
                    "AFTER MATCH SKIP TO FIRST {name} failed: cannot skip to the first row of the current match"
                );
            }
            Ok(step.row_idx)
        }
        AfterMatchSkip::ToLast(name) => {
            let target = skip_target_set.ok_or_else(|| {
                internal_datafusion_err!("missing validated skip target '{name}'")
            })?;
            let Some(step) = steps.iter().rev().find(|s| target.contains(s.sym)) else {
                return exec_err!(
                    "AFTER MATCH SKIP TO LAST {name} failed: variable '{name}' was not matched"
                );
            };
            if step.row_idx == start_row {
                return exec_err!(
                    "AFTER MATCH SKIP TO LAST {name} failed: cannot skip to the first row of the current match"
                );
            }
            Ok(step.row_idx)
        }
    }
}
