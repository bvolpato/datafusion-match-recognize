// SPDX-License-Identifier: Apache-2.0

//! Priority-ordered PikeVM row pattern matcher.
//!
//! Simulates a compiled [`Program`] over a partition's per-row symbol satisfaction sets,
//! preserving SQL:2016 leftmost-ordered alternation and greedy/reluctant quantifier semantics
//! while guaranteeing linear time per match attempt through per-position `pc` deduplication.
//!
//! Match histories are stored in a flat arena (`Vec<HistoryEntry>`) so thread cloning is `Copy`
//! and dropping long matches (tens of thousands of rows) uses O(1) stack space.

use datafusion_common::{Result, exec_err};

use crate::ast::{AfterMatchSkip, RowsPerMatch};
use crate::pattern::{Inst, Program, SymbolId};

/// Compact per-row representation of which pattern symbols hold on each row in a partition.
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

    /// Estimated heap bytes held by this symbol table.
    pub fn memory_bytes(&self) -> usize {
        match self {
            PartitionSymbols::Bitmask64 { rows, .. } => rows.capacity() * size_of::<u64>(),
            PartitionSymbols::Bitset { words, .. } => words.capacity() * size_of::<u64>(),
        }
    }
}

/// One matched row inside a match attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MatchStep {
    /// 0-based row index within the partition.
    pub row_idx: usize,
    /// Assigned pattern variable id.
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

/// Diagnostic counters collected during matching (used for complexity tests and metrics).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MatchStats {
    pub attempts: usize,
    pub total_thread_steps: usize,
}

const NO_HISTORY: u32 = u32::MAX;

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

/// Reusable PikeVM matcher for a compiled [`Program`].
#[derive(Debug)]
pub struct Matcher<'a> {
    program: &'a Program,
    visited: Vec<u32>,
    visit_gen: u32,
    arena: Vec<HistoryEntry>,
    epsilon_stack: Vec<Thread>,
}

impl<'a> Matcher<'a> {
    /// Creates a matcher for `program`.
    pub fn new(program: &'a Program) -> Self {
        Self {
            visited: vec![0; program.instructions.len()],
            program,
            visit_gen: 0,
            arena: Vec::new(),
            epsilon_stack: Vec::new(),
        }
    }

    /// Scans an entire partition, returning all emitted matches (and unmatched rows if requested).
    pub fn match_partition(
        &mut self,
        symbols: &PartitionSymbols,
        skip: &AfterMatchSkip,
        rows_per_match: RowsPerMatch,
    ) -> Result<(Vec<PartitionEvent>, MatchStats)> {
        let num_rows = symbols.num_rows();
        let skip_target_sym = match skip {
            AfterMatchSkip::ToFirst(name) | AfterMatchSkip::ToLast(name) => {
                let id = self.program.symbol_id(name).ok_or_else(|| {
                    datafusion_common::exec_datafusion_err!(
                        "AFTER MATCH SKIP references unknown pattern variable '{name}'"
                    )
                })?;
                Some(id)
            }
            AfterMatchSkip::PastLastRow | AfterMatchSkip::ToNextRow => None,
        };

        let mut events = Vec::new();
        let mut stats = MatchStats::default();
        let mut match_number: i64 = 0;
        let mut start_row: usize = 0;
        // Exclusive upper bound of the furthest row included in any non-empty match so far.
        // Under SQL:2016, a row is unmatched only if it was never part of any match (matters
        // when AFTER MATCH SKIP TO NEXT ROW / TO FIRST / TO LAST revisits already-matched rows).
        let mut max_matched_end: usize = 0;

        while start_row < num_rows {
            stats.attempts += 1;
            match self.find_match_at(symbols, start_row, &mut stats) {
                Some(steps) => {
                    let end_row = start_row + steps.len();
                    let is_empty = steps.is_empty();
                    match_number += 1;
                    if !is_empty {
                        max_matched_end = max_matched_end.max(end_row);
                    }

                    let next_start = if is_empty {
                        start_row + 1
                    } else {
                        compute_next_start(start_row, end_row, &steps, skip, skip_target_sym)?
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
                    if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end {
                        events.push(PartitionEvent::UnmatchedRow { row_idx: start_row });
                    }
                    start_row += 1;
                }
            }
        }

        Ok((events, stats))
    }

    /// Finds the highest-priority SQL:2016 match starting at `start_row`.
    pub fn find_match_at(
        &mut self,
        symbols: &PartitionSymbols,
        start_row: usize,
        stats: &mut MatchStats,
    ) -> Option<Vec<MatchStep>> {
        let num_rows = symbols.num_rows();
        self.arena.clear();
        let mut current: Vec<Thread> = Vec::new();
        let mut next: Vec<Thread> = Vec::new();
        let mut best_match: Option<u32> = None;

        // Seed initial epsilon closure at `start_row`.
        self.bump_gen();
        self.add_thread(
            Thread {
                pc: 0,
                exclude_depth: 0,
                history_head: NO_HISTORY,
            },
            start_row,
            num_rows,
            &mut current,
            &mut best_match,
            stats,
        );

        let mut pos = start_row;
        while !current.is_empty() && pos < num_rows {
            next.clear();
            self.bump_gen();

            for thread in current.drain(..) {
                stats.total_thread_steps += 1;
                match self.program.instructions[thread.pc] {
                    Inst::MatchLabel { sym } => {
                        if symbols.matches(pos, sym) {
                            let new_head = self.arena.len() as u32;
                            self.arena.push(HistoryEntry {
                                step: MatchStep {
                                    row_idx: pos,
                                    sym,
                                    excluded: thread.exclude_depth > 0,
                                },
                                prev: thread.history_head,
                            });
                            let hit_accept = self.add_thread(
                                Thread {
                                    pc: thread.pc + 1,
                                    exclude_depth: thread.exclude_depth,
                                    history_head: new_head,
                                },
                                pos + 1,
                                num_rows,
                                &mut next,
                                &mut best_match,
                                stats,
                            );
                            if hit_accept {
                                // This thread had higher priority than all remaining threads in
                                // `current` and reached `Accept`, so no lower-priority thread
                                // can ever beat it.
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

            std::mem::swap(&mut current, &mut next);
            pos += 1;
        }

        best_match.map(|head| materialize_history(&self.arena, head))
    }

    #[inline]
    fn bump_gen(&mut self) {
        self.visit_gen = self.visit_gen.wrapping_add(1);
        if self.visit_gen == 0 {
            self.visited.fill(0);
            self.visit_gen = 1;
        }
    }

    /// Iteratively expands epsilon transitions in strict priority order (DFS stack).
    ///
    /// Returns `true` if `Inst::Accept` was reached (cutting off all lower-priority branches).
    fn add_thread(
        &mut self,
        initial: Thread,
        pos: usize,
        num_rows: usize,
        out: &mut Vec<Thread>,
        best_match: &mut Option<u32>,
        stats: &mut MatchStats,
    ) -> bool {
        self.epsilon_stack.clear();
        self.epsilon_stack.push(initial);

        while let Some(thread) = self.epsilon_stack.pop() {
            let pc = thread.pc;
            if self.visited[pc] == self.visit_gen {
                continue;
            }
            self.visited[pc] = self.visit_gen;
            stats.total_thread_steps += 1;

            match self.program.instructions[pc] {
                Inst::MatchLabel { .. } => {
                    out.push(thread);
                }
                Inst::Jump(target) => {
                    self.epsilon_stack.push(Thread {
                        pc: target,
                        ..thread
                    });
                }
                Inst::Split { first, second } => {
                    // Push `second` first so `first` is popped and explored with higher priority.
                    self.epsilon_stack.push(Thread {
                        pc: second,
                        ..thread
                    });
                    self.epsilon_stack.push(Thread {
                        pc: first,
                        ..thread
                    });
                }
                Inst::AnchorStart => {
                    if pos == 0 {
                        self.epsilon_stack.push(Thread {
                            pc: pc + 1,
                            ..thread
                        });
                    }
                }
                Inst::AnchorEnd => {
                    if pos == num_rows {
                        self.epsilon_stack.push(Thread {
                            pc: pc + 1,
                            ..thread
                        });
                    }
                }
                Inst::ExcludeEnter => {
                    self.epsilon_stack.push(Thread {
                        pc: pc + 1,
                        exclude_depth: thread.exclude_depth + 1,
                        ..thread
                    });
                }
                Inst::ExcludeExit => {
                    self.epsilon_stack.push(Thread {
                        pc: pc + 1,
                        exclude_depth: thread.exclude_depth.saturating_sub(1),
                        ..thread
                    });
                }
                Inst::Accept => {
                    *best_match = Some(thread.history_head);
                    self.epsilon_stack.clear();
                    return true;
                }
            }
        }

        false
    }
}

fn materialize_history(arena: &[HistoryEntry], mut head: u32) -> Vec<MatchStep> {
    let mut steps = Vec::new();
    while head != NO_HISTORY {
        let entry = arena[head as usize];
        steps.push(entry.step);
        head = entry.prev;
    }
    steps.reverse();
    steps
}

fn compute_next_start(
    start_row: usize,
    end_row: usize,
    steps: &[MatchStep],
    skip: &AfterMatchSkip,
    skip_target_sym: Option<SymbolId>,
) -> Result<usize> {
    match skip {
        AfterMatchSkip::PastLastRow => Ok(end_row),
        AfterMatchSkip::ToNextRow => Ok(start_row + 1),
        AfterMatchSkip::ToFirst(name) => {
            let target = skip_target_sym.expect("validated symbol");
            let Some(step) = steps.iter().find(|s| s.sym == target) else {
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
            let target = skip_target_sym.expect("validated symbol");
            let Some(step) = steps.iter().rev().find(|s| s.sym == target) else {
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
