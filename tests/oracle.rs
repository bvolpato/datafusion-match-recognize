// SPDX-License-Identifier: Apache-2.0

//! Layer 1 verification: independent recursive backtracking reference oracle + `proptest`
//! differential fuzzing against the compiled PikeVM [`Matcher`].

use datafusion_match_recognize::ast::{
    AfterMatchSkip, MatchRecognizeConfig, Pattern, RowsPerMatch, SymbolRef,
};
use datafusion_match_recognize::matcher::{
    FoundMatch, MatchStep, Matcher, PartitionEvent, PartitionSymbols,
};
use datafusion_match_recognize::pattern::{Program, SymbolId};
use proptest::prelude::*;

// ============================================================================
// Independent Reference Backtracking Oracle (SQL:2016 Operational Semantics)
// ============================================================================

/// Reference recursive evaluator of SQL:2016 row pattern semantics.
///
/// Intentionally written as a direct operational interpreter of the [`Pattern`] AST (with zero
/// shared code with the NFA compiler or PikeVM) so bugs in compilation, split ordering, epsilon
/// deduplication, or loop handling are caught immediately.
///
/// The continuation `cont` models the remainder of the pattern; returning the first `Some(steps)`
/// strictly implements SQL:2016 leftmost-ordered alternation and greedy/reluctant precedence.
type MatchCont<'a> = &'a dyn Fn(usize, &mut Vec<MatchStep>) -> Option<Vec<MatchStep>>;

fn ref_eval(
    pattern: &Pattern,
    pos: usize,
    exclude_depth: u32,
    symbols: &PartitionSymbols,
    program: &Program,
    steps: &mut Vec<MatchStep>,
    cont: MatchCont<'_>,
) -> Option<Vec<MatchStep>> {
    match pattern {
        Pattern::Symbol(SymbolRef::Start) => {
            if pos == 0 {
                cont(pos, steps)
            } else {
                None
            }
        }
        Pattern::Symbol(SymbolRef::End) => {
            if pos == symbols.num_rows() {
                cont(pos, steps)
            } else {
                None
            }
        }
        Pattern::Symbol(SymbolRef::Named(name)) => {
            if pos >= symbols.num_rows() {
                return None;
            }
            let sym = program.symbol_id(name).expect("registered symbol");
            if !symbols.matches(pos, sym) {
                return None;
            }
            steps.push(MatchStep {
                row_idx: pos,
                sym,
                excluded: exclude_depth > 0,
            });
            let res = cont(pos + 1, steps);
            steps.pop();
            res
        }
        Pattern::Exclude(inner) => {
            ref_eval(inner, pos, exclude_depth + 1, symbols, program, steps, cont)
        }
        Pattern::Concat(items) => {
            ref_eval_slice(items, pos, exclude_depth, symbols, program, steps, cont)
        }
        Pattern::Alternation(branches) => {
            for branch in branches {
                if let Some(m) =
                    ref_eval(branch, pos, exclude_depth, symbols, program, steps, cont)
                {
                    return Some(m);
                }
            }
            None
        }
        Pattern::Permute(items) => {
            if items.is_empty() {
                return cont(pos, steps);
            }
            let perms = lexicographic_perms(items.len());
            for order in perms {
                let seq: Vec<Pattern> =
                    order.into_iter().map(|i| items[i].clone()).collect();
                if let Some(m) = ref_eval_slice(
                    &seq,
                    pos,
                    exclude_depth,
                    symbols,
                    program,
                    steps,
                    cont,
                ) {
                    return Some(m);
                }
            }
            None
        }
        Pattern::Repetition {
            inner,
            min,
            max,
            reluctant,
        } => ref_eval_rep(
            inner,
            *min,
            *max,
            *reluctant,
            0,
            None,
            pos,
            exclude_depth,
            symbols,
            program,
            steps,
            cont,
        ),
    }
}

fn ref_eval_slice(
    items: &[Pattern],
    pos: usize,
    exclude_depth: u32,
    symbols: &PartitionSymbols,
    program: &Program,
    steps: &mut Vec<MatchStep>,
    cont: MatchCont<'_>,
) -> Option<Vec<MatchStep>> {
    match items.split_first() {
        None => cont(pos, steps),
        Some((head, tail)) => ref_eval(
            head,
            pos,
            exclude_depth,
            symbols,
            program,
            steps,
            &|p, st| ref_eval_slice(tail, p, exclude_depth, symbols, program, st, cont),
        ),
    }
}

#[expect(clippy::too_many_arguments)]
fn ref_eval_rep(
    inner: &Pattern,
    min: u32,
    max: Option<u32>,
    reluctant: bool,
    count: u32,
    last_iter_start_pos: Option<usize>,
    pos: usize,
    exclude_depth: u32,
    symbols: &PartitionSymbols,
    program: &Program,
    steps: &mut Vec<MatchStep>,
    cont: MatchCont<'_>,
) -> Option<Vec<MatchStep>> {
    // Mandatory iterations (count < min)
    if count < min {
        return ref_eval(
            inner,
            pos,
            exclude_depth,
            symbols,
            program,
            steps,
            &|next_pos, st| {
                ref_eval_rep(
                    inner,
                    min,
                    max,
                    reluctant,
                    count + 1,
                    Some(pos),
                    next_pos,
                    exclude_depth,
                    symbols,
                    program,
                    st,
                    cont,
                )
            },
        );
    }

    // Reached explicit upper bound
    if let Some(max_val) = max
        && count >= max_val
    {
        return cont(pos, steps);
    }

    // Unbounded loop empty-iteration guard: if the previous optional iteration consumed 0 rows,
    // SQL:2016 / Trino bytecode stops looping at the same position and exits immediately.
    if max.is_none() && last_iter_start_pos == Some(pos) {
        return cont(pos, steps);
    }

    let try_more = |st: &mut Vec<MatchStep>| {
        ref_eval(
            inner,
            pos,
            exclude_depth,
            symbols,
            program,
            st,
            &|next_pos, st2| {
                // Once already in a loop-back iteration (`count >= 1`) of an unbounded loop,
                // returning to `loop_split` at the same `pos` without consuming a row hits the
                // epsilon-cycle guard (`visited[loop_split]`), aborting this zero-width branch so
                // remaining alternation branches (e.g. `(c | c? | a)*` on `[c, a]`) can run first.
                // Verified against Trino 484.
                if max.is_none() && count >= 1 && next_pos == pos {
                    return None;
                }
                ref_eval_rep(
                    inner,
                    min,
                    max,
                    reluctant,
                    count + 1,
                    Some(pos),
                    next_pos,
                    exclude_depth,
                    symbols,
                    program,
                    st2,
                    cont,
                )
            },
        )
    };

    if reluctant {
        if let Some(m) = cont(pos, steps) {
            return Some(m);
        }
        try_more(steps)
    } else {
        if let Some(m) = try_more(steps) {
            return Some(m);
        }
        cont(pos, steps)
    }
}

fn ref_find_match_at(
    pattern: &Pattern,
    symbols: &PartitionSymbols,
    program: &Program,
    start_row: usize,
) -> Option<Vec<MatchStep>> {
    let mut steps = Vec::new();
    ref_eval(
        pattern,
        start_row,
        0,
        symbols,
        program,
        &mut steps,
        &|_end_pos, st| Some(st.clone()),
    )
}

fn ref_match_partition(
    pattern: &Pattern,
    symbols: &PartitionSymbols,
    program: &Program,
    skip: &AfterMatchSkip,
    rows_per_match: RowsPerMatch,
) -> Result<Vec<PartitionEvent>, String> {
    let num_rows = symbols.num_rows();
    let skip_sym: Option<SymbolId> = match skip {
        AfterMatchSkip::ToFirst(name) | AfterMatchSkip::ToLast(name) => Some(
            program
                .symbol_id(name)
                .ok_or_else(|| format!("unknown {name}"))?,
        ),
        _ => None,
    };

    let mut events = Vec::new();
    let mut match_number = 0i64;
    let mut start_row = 0usize;
    let mut max_matched_end = 0usize;

    while start_row < num_rows {
        match ref_find_match_at(pattern, symbols, program, start_row) {
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
                    match skip {
                        AfterMatchSkip::PastLastRow => end_row,
                        AfterMatchSkip::ToNextRow => start_row + 1,
                        AfterMatchSkip::ToFirst(_) => {
                            let s = steps
                                .iter()
                                .find(|s| s.sym == skip_sym.unwrap())
                                .ok_or_else(|| "not matched".to_string())?;
                            if s.row_idx == start_row {
                                return Err("infinite loop".to_string());
                            }
                            s.row_idx
                        }
                        AfterMatchSkip::ToLast(_) => {
                            let s = steps
                                .iter()
                                .rev()
                                .find(|s| s.sym == skip_sym.unwrap())
                                .ok_or_else(|| "not matched".to_string())?;
                            if s.row_idx == start_row {
                                return Err("infinite loop".to_string());
                            }
                            s.row_idx
                        }
                    }
                };

                if !is_empty || rows_per_match.show_empty_matches() {
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
                if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end
                {
                    events.push(PartitionEvent::UnmatchedRow { row_idx: start_row });
                }
                start_row += 1;
            }
        }
    }

    Ok(events)
}

fn lexicographic_perms(n: usize) -> Vec<Vec<usize>> {
    let mut cur: Vec<usize> = (0..n).collect();
    let mut out = vec![cur.clone()];
    loop {
        if cur.len() < 2 {
            break;
        }
        let mut i = cur.len() - 2;
        while cur[i] >= cur[i + 1] {
            if i == 0 {
                return out;
            }
            i -= 1;
        }
        let mut j = cur.len() - 1;
        while cur[j] <= cur[i] {
            j -= 1;
        }
        cur.swap(i, j);
        cur[i + 1..].reverse();
        out.push(cur.clone());
    }
    out
}

// ============================================================================
// Proptest Generators
// ============================================================================

fn arb_symbol_ref() -> impl Strategy<Value = SymbolRef> {
    prop_oneof![
        4 => Just(SymbolRef::Named("a".into())),
        4 => Just(SymbolRef::Named("b".into())),
        3 => Just(SymbolRef::Named("c".into())),
        1 => Just(SymbolRef::Start),
        1 => Just(SymbolRef::End),
    ]
}

fn arb_pattern() -> impl Strategy<Value = Pattern> {
    let leaf = arb_symbol_ref().prop_map(Pattern::Symbol);
    leaf.prop_recursive(3, 18, 3, |inner| {
        prop_oneof![
            3 => prop::collection::vec(inner.clone(), 1..=3).prop_map(Pattern::Concat),
            3 => prop::collection::vec(inner.clone(), 2..=3).prop_map(Pattern::Alternation),
            // Greedy and reluctant quantifiers
            3 => (
                inner.clone(),
                prop_oneof![
                    Just((0u32, None)),
                    Just((1u32, None)),
                    Just((0u32, Some(1u32))),
                    Just((1u32, Some(3u32))),
                    Just((2u32, Some(2u32))),
                    Just((0u32, Some(2u32))),
                ],
                any::<bool>(),
            )
                .prop_map(|(p, (min, max), reluctant)| Pattern::Repetition {
                    inner: Box::new(p),
                    min,
                    max,
                    reluctant,
                }),
            1 => inner.clone().prop_map(|p| Pattern::Exclude(Box::new(p))),
            1 => prop::collection::vec(
                prop_oneof![
                    Just(Pattern::Symbol(SymbolRef::Named("a".into()))),
                    Just(Pattern::Symbol(SymbolRef::Named("b".into()))),
                    Just(Pattern::Symbol(SymbolRef::Named("c".into()))),
                ],
                2..=3,
            )
            .prop_map(Pattern::Permute),
        ]
    })
}

fn arb_skip() -> impl Strategy<Value = AfterMatchSkip> {
    prop_oneof![
        3 => Just(AfterMatchSkip::PastLastRow),
        3 => Just(AfterMatchSkip::ToNextRow),
        1 => Just(AfterMatchSkip::ToFirst("a".into())),
        1 => Just(AfterMatchSkip::ToFirst("b".into())),
        1 => Just(AfterMatchSkip::ToLast("a".into())),
        1 => Just(AfterMatchSkip::ToLast("b".into())),
    ]
}

fn arb_rows_per_match() -> impl Strategy<Value = RowsPerMatch> {
    prop_oneof![
        Just(RowsPerMatch::OneRow),
        Just(RowsPerMatch::AllRowsShowEmpty),
        Just(RowsPerMatch::AllRowsOmitEmpty),
        Just(RowsPerMatch::AllRowsWithUnmatched),
    ]
}

// ============================================================================
// Targeted Unit Tests + Proptest Fuzz Suite
// ============================================================================

#[test]
fn test_targeted_sql2016_edge_cases_against_oracle() {
    let declared = vec!["a".into(), "b".into(), "c".into()];
    let cases: Vec<(Pattern, Vec<u64>)> = vec![
        // Ordered alternation: A | A B must pick A even though A B is longer.
        (
            Pattern::Alternation(vec![
                Pattern::Symbol(SymbolRef::Named("a".into())),
                Pattern::Concat(vec![
                    Pattern::Symbol(SymbolRef::Named("a".into())),
                    Pattern::Symbol(SymbolRef::Named("b".into())),
                ]),
            ]),
            vec![0b001, 0b010],
        ),
        // Reluctant inside greedy loop: (A??)* on [A]
        (
            Pattern::Repetition {
                inner: Box::new(Pattern::Repetition {
                    inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
                    min: 0,
                    max: Some(1),
                    reluctant: true,
                }),
                min: 0,
                max: None,
                reluctant: false,
            },
            vec![0b001, 0b001],
        ),
        // Reluctant inside greedy loop followed by B: (A??)* B on [A, B]
        (
            Pattern::Concat(vec![
                Pattern::Repetition {
                    inner: Box::new(Pattern::Repetition {
                        inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
                        min: 0,
                        max: Some(1),
                        reluctant: true,
                    }),
                    min: 0,
                    max: None,
                    reluctant: false,
                },
                Pattern::Symbol(SymbolRef::Named("b".into())),
            ]),
            vec![0b001, 0b010],
        ),
        // PERMUTE(A, B, C) with overlapping symbols
        (
            Pattern::Permute(vec![
                Pattern::Symbol(SymbolRef::Named("a".into())),
                Pattern::Symbol(SymbolRef::Named("b".into())),
                Pattern::Symbol(SymbolRef::Named("c".into())),
            ]),
            vec![0b111, 0b111, 0b111],
        ),
        // Empty-iteration cycle in loop-back vs initial iteration (verified against Trino 484):
        // (c | c? | a)* on [c, a] matches [c, a], whereas (c? | a)* on [a] matches empty [].
        (
            Pattern::Repetition {
                inner: Box::new(Pattern::Alternation(vec![
                    Pattern::Symbol(SymbolRef::Named("c".into())),
                    Pattern::Repetition {
                        inner: Box::new(Pattern::Symbol(SymbolRef::Named("c".into()))),
                        min: 0,
                        max: Some(1),
                        reluctant: false,
                    },
                    Pattern::Symbol(SymbolRef::Named("a".into())),
                ])),
                min: 0,
                max: None,
                reluctant: false,
            },
            vec![0b100, 0b001],
        ),
        (
            Pattern::Repetition {
                inner: Box::new(Pattern::Alternation(vec![
                    Pattern::Repetition {
                        inner: Box::new(Pattern::Symbol(SymbolRef::Named("c".into()))),
                        min: 0,
                        max: Some(1),
                        reluctant: false,
                    },
                    Pattern::Symbol(SymbolRef::Named("a".into())),
                ])),
                min: 0,
                max: None,
                reluctant: false,
            },
            vec![0b001],
        ),
        // Anchors ^ A+ $
        (
            Pattern::Concat(vec![
                Pattern::Symbol(SymbolRef::Start),
                Pattern::Repetition {
                    inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
                    min: 1,
                    max: None,
                    reluctant: false,
                },
                Pattern::Symbol(SymbolRef::End),
            ]),
            vec![0b001, 0b001, 0b001],
        ),
    ];

    for (pattern, bitmasks) in cases {
        let program =
            Program::compile(&pattern, &declared, MatchRecognizeConfig::default())
                .unwrap();
        let symbols = PartitionSymbols::from_bitmasks(bitmasks, 3);
        for skip in [AfterMatchSkip::PastLastRow, AfterMatchSkip::ToNextRow] {
            for rpm in [
                RowsPerMatch::OneRow,
                RowsPerMatch::AllRowsShowEmpty,
                RowsPerMatch::AllRowsOmitEmpty,
                RowsPerMatch::AllRowsWithUnmatched,
            ] {
                let expected =
                    ref_match_partition(&pattern, &symbols, &program, &skip, rpm);
                let mut matcher = Matcher::new(&program);
                let actual = matcher
                    .match_partition(&symbols, &skip, rpm)
                    .map(|(ev, _)| ev)
                    .map_err(|e| e.to_string());
                assert_eq!(
                    actual.is_ok(),
                    expected.is_ok(),
                    "status mismatch on pattern={pattern}, skip={skip}, rpm={rpm}"
                );
                if let (Ok(a), Ok(e)) = (actual, expected) {
                    assert_eq!(
                        a, e,
                        "event mismatch on pattern={pattern}, skip={skip}, rpm={rpm}"
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1500,
        max_shrink_iters: 2000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn fuzz_pikevm_matches_reference_oracle(
        pattern in arb_pattern(),
        bitmasks in prop::collection::vec(0u64..8u64, 0..=10),
        skip in arb_skip(),
        rpm in arb_rows_per_match(),
    ) {
        let declared = vec!["a".into(), "b".into(), "c".into()];
        let program = Program::compile(&pattern, &declared, MatchRecognizeConfig::default())
            .expect("generated pattern compiles");
        let symbols = PartitionSymbols::from_bitmasks(bitmasks, 3);

        let expected = ref_match_partition(&pattern, &symbols, &program, &skip, rpm);
        let mut matcher = Matcher::new(&program);
        let actual = matcher.match_partition(&symbols, &skip, rpm);

        match (actual, expected) {
            (Ok((actual_events, _)), Ok(expected_events)) => {
                prop_assert_eq!(
                    actual_events,
                    expected_events,
                    "Mismatch for pattern={}, skip={}, rpm={}",
                    pattern,
                    skip,
                    rpm
                );
            }
            (Err(_), Err(_)) => {
                // Both rejected an invalid SKIP TO FIRST/LAST target or infinite-loop skip.
            }
            (actual, expected) => {
                prop_assert!(
                    false,
                    "Result/error mismatch for pattern={}, skip={}, rpm={}: actual={:?}, expected={:?}",
                    pattern,
                    skip,
                    rpm,
                    actual.map(|(e, _)| e),
                    expected
                );
            }
        }
    }
}

type StatefulMatchCont<'a> = &'a dyn Fn(
    usize,
    &mut Vec<MatchStep>,
    &mut TestStatefulEval,
) -> Option<Vec<MatchStep>>;

struct TestStatefulEval {
    vals: Vec<i32>,
}

impl datafusion_match_recognize::matcher::StatefulPredicateEvaluator
    for TestStatefulEval
{
    fn is_stateful(&self, sym: SymbolId) -> bool {
        sym == 1 || sym == 2
    }

    fn has_any_stateful(&self) -> bool {
        true
    }

    fn eval_stateful(
        &mut self,
        sym: SymbolId,
        tentative_steps: &[MatchStep],
        _candidate_match_number: i64,
    ) -> datafusion_common::Result<bool> {
        let pos = tentative_steps.last().unwrap().row_idx;
        match sym {
            1 => Ok(pos > 0 && self.vals[pos] < self.vals[pos - 1]),
            2 => {
                if let Some(first_a) = tentative_steps.iter().find(|s| s.sym == 0) {
                    Ok(self.vals[pos] > self.vals[first_a.row_idx])
                } else {
                    Ok(pos > 0 && self.vals[pos] > self.vals[pos - 1])
                }
            }
            _ => unreachable!(),
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn ref_eval_stateful(
    pattern: &Pattern,
    pos: usize,
    exclude_depth: u32,
    symbols: &PartitionSymbols,
    program: &Program,
    eval: &mut TestStatefulEval,
    steps: &mut Vec<MatchStep>,
    cont: StatefulMatchCont<'_>,
) -> Option<Vec<MatchStep>> {
    use datafusion_match_recognize::matcher::StatefulPredicateEvaluator;
    match pattern {
        Pattern::Symbol(SymbolRef::Start) => {
            if pos == 0 {
                cont(pos, steps, eval)
            } else {
                None
            }
        }
        Pattern::Symbol(SymbolRef::End) => {
            if pos == symbols.num_rows() {
                cont(pos, steps, eval)
            } else {
                None
            }
        }
        Pattern::Symbol(SymbolRef::Named(name)) => {
            if pos >= symbols.num_rows() {
                return None;
            }
            let sym = program.symbol_id(name).unwrap();
            steps.push(MatchStep {
                row_idx: pos,
                sym,
                excluded: exclude_depth > 0,
            });
            let ok = if eval.is_stateful(sym) {
                eval.eval_stateful(sym, steps, 1).unwrap()
            } else {
                symbols.matches(pos, sym)
            };
            let res = if ok { cont(pos + 1, steps, eval) } else { None };
            steps.pop();
            res
        }
        Pattern::Exclude(inner) => ref_eval_stateful(
            inner,
            pos,
            exclude_depth + 1,
            symbols,
            program,
            eval,
            steps,
            cont,
        ),
        Pattern::Concat(items) => ref_eval_stateful_slice(
            items,
            pos,
            exclude_depth,
            symbols,
            program,
            eval,
            steps,
            cont,
        ),
        Pattern::Alternation(branches) => {
            for b in branches {
                if let Some(m) = ref_eval_stateful(
                    b,
                    pos,
                    exclude_depth,
                    symbols,
                    program,
                    eval,
                    steps,
                    cont,
                ) {
                    return Some(m);
                }
            }
            None
        }
        Pattern::Permute(items) => {
            for order in lexicographic_perms(items.len()) {
                let seq: Vec<Pattern> =
                    order.into_iter().map(|i| items[i].clone()).collect();
                if let Some(m) = ref_eval_stateful_slice(
                    &seq,
                    pos,
                    exclude_depth,
                    symbols,
                    program,
                    eval,
                    steps,
                    cont,
                ) {
                    return Some(m);
                }
            }
            None
        }
        Pattern::Repetition {
            inner,
            min,
            max,
            reluctant,
        } => ref_eval_stateful_rep(
            inner,
            *min,
            *max,
            *reluctant,
            0,
            None,
            pos,
            exclude_depth,
            symbols,
            program,
            eval,
            steps,
            cont,
        ),
    }
}

#[expect(clippy::too_many_arguments)]
fn ref_eval_stateful_slice(
    items: &[Pattern],
    pos: usize,
    exclude_depth: u32,
    symbols: &PartitionSymbols,
    program: &Program,
    eval: &mut TestStatefulEval,
    steps: &mut Vec<MatchStep>,
    cont: StatefulMatchCont<'_>,
) -> Option<Vec<MatchStep>> {
    match items.split_first() {
        None => cont(pos, steps, eval),
        Some((head, tail)) => ref_eval_stateful(
            head,
            pos,
            exclude_depth,
            symbols,
            program,
            eval,
            steps,
            &|p, st, ev| {
                ref_eval_stateful_slice(
                    tail,
                    p,
                    exclude_depth,
                    symbols,
                    program,
                    ev,
                    st,
                    cont,
                )
            },
        ),
    }
}

#[expect(clippy::too_many_arguments)]
fn ref_eval_stateful_rep(
    inner: &Pattern,
    min: u32,
    max: Option<u32>,
    reluctant: bool,
    count: u32,
    last_iter_start_pos: Option<usize>,
    pos: usize,
    exclude_depth: u32,
    symbols: &PartitionSymbols,
    program: &Program,
    eval: &mut TestStatefulEval,
    steps: &mut Vec<MatchStep>,
    cont: StatefulMatchCont<'_>,
) -> Option<Vec<MatchStep>> {
    if count < min {
        return ref_eval_stateful(
            inner,
            pos,
            exclude_depth,
            symbols,
            program,
            eval,
            steps,
            &|next_pos, st, ev| {
                ref_eval_stateful_rep(
                    inner,
                    min,
                    max,
                    reluctant,
                    count + 1,
                    Some(pos),
                    next_pos,
                    exclude_depth,
                    symbols,
                    program,
                    ev,
                    st,
                    cont,
                )
            },
        );
    }
    if let Some(m) = max
        && count >= m
    {
        return cont(pos, steps, eval);
    }
    if max.is_none() && last_iter_start_pos == Some(pos) {
        return cont(pos, steps, eval);
    }
    let try_more = |st: &mut Vec<MatchStep>, ev: &mut TestStatefulEval| {
        ref_eval_stateful(
            inner,
            pos,
            exclude_depth,
            symbols,
            program,
            ev,
            st,
            &|next_pos, st2, ev2| {
                if max.is_none() && count >= 1 && next_pos == pos {
                    return None;
                }
                ref_eval_stateful_rep(
                    inner,
                    min,
                    max,
                    reluctant,
                    count + 1,
                    Some(pos),
                    next_pos,
                    exclude_depth,
                    symbols,
                    program,
                    ev2,
                    st2,
                    cont,
                )
            },
        )
    };
    if reluctant {
        if let Some(m) = cont(pos, steps, eval) {
            return Some(m);
        }
        try_more(steps, eval)
    } else {
        if let Some(m) = try_more(steps, eval) {
            return Some(m);
        }
        cont(pos, steps, eval)
    }
}

#[test]
fn test_start_symbol_fast_skip_and_subset_skip() {
    use datafusion_match_recognize::ast::SubsetDef;
    let declared = vec!["a".into(), "b".into(), "c".into()];
    let pattern = Pattern::Concat(vec![
        Pattern::Symbol(SymbolRef::Named("a".into())),
        Pattern::Symbol(SymbolRef::Named("b".into())),
        Pattern::Symbol(SymbolRef::Named("c".into())),
    ]);
    let subsets = vec![SubsetDef {
        name: "u".into(),
        members: vec!["b".into(), "c".into()],
    }];
    let program = Program::compile_with_subsets(
        &pattern,
        &declared,
        &subsets,
        MatchRecognizeConfig::default(),
    )
    .unwrap();
    assert_eq!(program.start_symbols_mask, Some(0b001));

    let mut bitmasks = vec![0u64; 100];
    bitmasks[50] = 0b001;
    bitmasks[51] = 0b010;
    bitmasks[52] = 0b100;
    let symbols = PartitionSymbols::from_bitmasks(bitmasks, 3);
    let mut matcher = Matcher::new(&program);
    let (events, stats) = matcher
        .match_partition(
            &symbols,
            &AfterMatchSkip::ToLast("u".into()),
            RowsPerMatch::OneRow,
        )
        .unwrap();
    assert_eq!(events.len(), 1);
    assert!(
        stats.fast_skipped_rows >= 95,
        "expected >= 95 fast-skipped rows, got {}",
        stats.fast_skipped_rows
    );
}

#[test]
fn test_trino_pattern_matching_machine_cases() {
    // Ported directly from Trino's TestPatternMatchingMachine.java:
    // verifies exact matched labels AND exact per-step `excluded` flags across nested exclusions,
    // quantifiers, anchors, permutations, reluctant `?`, and long exponential-trap patterns.
    let declared = vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()];
    let sym = |name: &str| Pattern::Symbol(SymbolRef::Named(name.into()));
    let excl = |p: Pattern| Pattern::Exclude(Box::new(p));
    let cat = |items: Vec<Pattern>| Pattern::Concat(items);
    let star = |p: Pattern, greedy: bool| Pattern::Repetition {
        inner: Box::new(p),
        min: 0,
        max: None,
        reluctant: !greedy,
    };
    let plus = |p: Pattern, greedy: bool| Pattern::Repetition {
        inner: Box::new(p),
        min: 1,
        max: None,
        reluctant: !greedy,
    };
    let opt = |p: Pattern, greedy: bool| Pattern::Repetition {
        inner: Box::new(p),
        min: 0,
        max: Some(1),
        reluctant: !greedy,
    };

    let make_input = |s: &str| -> PartitionSymbols {
        let bitmasks: Vec<u64> = s
            .chars()
            .map(|c| match c {
                'A' => 1 << 0,
                'B' => 1 << 1,
                'C' => 1 << 2,
                'D' => 1 << 3,
                'E' => 1 << 4,
                _ => 0,
            })
            .collect();
        PartitionSymbols::from_bitmasks(bitmasks, 5)
    };

    let check = |pat: Pattern, input: &str, expected: Option<&[(char, bool)]>| {
        let program =
            Program::compile(&pat, &declared, MatchRecognizeConfig::default()).unwrap();
        let symbols = make_input(input);
        let mut matcher = Matcher::new(&program);
        let mut stats = Default::default();
        let actual = matcher.find_match_at(&symbols, 0, &mut stats);
        let ref_actual = ref_find_match_at(&pat, &symbols, &program, 0);
        assert_eq!(actual, ref_actual, "oracle mismatch for {pat} on {input}");
        match (actual, expected) {
            (None, None) => {}
            (Some(steps), Some(exp)) => {
                let got: Vec<(char, bool)> = steps
                    .iter()
                    .map(|st| {
                        (
                            program.id_to_symbol[st.sym as usize]
                                .chars()
                                .next()
                                .unwrap(),
                            st.excluded,
                        )
                    })
                    .collect();
                assert_eq!(got, exp, "mismatch for {pat} on {input}");
            }
            (got, exp) => {
                panic!(
                    "match presence mismatch for {pat} on {input}: got={got:?}, exp={exp:?}"
                )
            }
        }
    };

    // testCaptures (exact excluded row spans)
    check(
        cat(vec![excl(sym("A")), sym("B")]),
        "ABCD",
        Some(&[('A', true), ('B', false)]),
    );
    check(
        excl(cat(vec![sym("A"), sym("B")])),
        "ABCD",
        Some(&[('A', true), ('B', true)]),
    );
    check(
        cat(vec![sym("A"), excl(sym("B"))]),
        "ABCD",
        Some(&[('A', false), ('B', true)]),
    );
    check(
        cat(vec![sym("A"), star(excl(sym("B")), true)]),
        "ABBBCD",
        Some(&[('A', false), ('B', true), ('B', true), ('B', true)]),
    );
    check(
        cat(vec![sym("A"), excl(star(sym("B"), true))]),
        "ABBBCD",
        Some(&[('A', false), ('B', true), ('B', true), ('B', true)]),
    );
    check(
        cat(vec![
            sym("A"),
            star(excl(sym("B")), true),
            sym("C"),
            excl(star(sym("D"), true)),
        ]),
        "ABBCDDDE",
        Some(&[
            ('A', false),
            ('B', true),
            ('B', true),
            ('C', false),
            ('D', true),
            ('D', true),
            ('D', true),
        ]),
    );
    check(
        cat(vec![
            sym("A"),
            star(
                cat(vec![
                    excl(cat(vec![sym("B"), sym("C")])),
                    sym("D"),
                    excl(sym("E")),
                ]),
                true,
            ),
        ]),
        "ABCDEBCDEBCDE",
        Some(&[
            ('A', false),
            ('B', true),
            ('C', true),
            ('D', false),
            ('E', true),
            ('B', true),
            ('C', true),
            ('D', false),
            ('E', true),
            ('B', true),
            ('C', true),
            ('D', false),
            ('E', true),
        ]),
    );

    // testEmptyMatch & testNoMatch
    check(cat(vec![]), "ABCD", Some(&[]));
    check(sym("A"), "B", None);

    // testLabels: greedy/reluctant ?, anchors, PERMUTE
    check(
        cat(vec![sym("A"), opt(sym("B"), true)]),
        "ABCD",
        Some(&[('A', false), ('B', false)]),
    );
    check(
        cat(vec![sym("A"), opt(sym("B"), false)]),
        "ABCD",
        Some(&[('A', false)]),
    );
    check(
        cat(vec![Pattern::Symbol(SymbolRef::Start), sym("A"), sym("B")]),
        "ABCD",
        Some(&[('A', false), ('B', false)]),
    );
    check(
        cat(vec![sym("A"), sym("B"), Pattern::Symbol(SymbolRef::End)]),
        "AB",
        Some(&[('A', false), ('B', false)]),
    );
    check(
        Pattern::Permute(vec![sym("A"), sym("B"), sym("C")]),
        "ABCD",
        Some(&[('A', false), ('B', false), ('C', false)]),
    );

    // testLongMatch: linear-time no-match on ^ (A+)+ $ against 40 A's + B
    let program = Program::compile(
        &cat(vec![
            Pattern::Symbol(SymbolRef::Start),
            plus(plus(sym("A"), true), true),
            Pattern::Symbol(SymbolRef::End),
        ]),
        &declared,
        MatchRecognizeConfig::default(),
    )
    .unwrap();
    let mut matcher = Matcher::new(&program);
    let mut stats = Default::default();
    let long_input = make_input("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAB");
    assert_eq!(matcher.find_match_at(&long_input, 0, &mut stats), None);
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 1000,
        max_shrink_iters: 1000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn fuzz_stateful_pikevm_matches_reference_oracle(
        pattern in arb_pattern(),
        vals in prop::collection::vec(0i32..10i32, 0..=8),
        a_flags in prop::collection::vec(any::<bool>(), 0..=8),
    ) {
        let len = vals.len().min(a_flags.len());
        let vals = vals[..len].to_vec();
        let bitmasks: Vec<u64> = a_flags[..len].iter().map(|&b| u64::from(b)).collect();
        let declared = vec!["a".into(), "b".into(), "c".into()];
        let program = Program::compile(&pattern, &declared, MatchRecognizeConfig::default()).unwrap();
        let symbols = PartitionSymbols::from_bitmasks(bitmasks, 3);

        for start_row in 0..len {
            let mut eval_ref = TestStatefulEval { vals: vals.clone() };
            let mut steps_buf = Vec::new();
            let expected = ref_eval_stateful(
                &pattern,
                start_row,
                0,
                &symbols,
                &program,
                &mut eval_ref,
                &mut steps_buf,
                &|_, st, _| Some(st.clone()),
            );

            let mut eval_vm = TestStatefulEval { vals: vals.clone() };
            let mut matcher = Matcher::new(&program);
            let mut stats = Default::default();
            let actual = matcher
                .find_match_at_stateful(&symbols, start_row, 1, &mut eval_vm, &mut stats)
                .unwrap();

            prop_assert_eq!(
                actual,
                expected,
                "Stateful mismatch for pattern={}, vals={:?}, start_row={}",
                pattern,
                vals,
                start_row
            );
        }
    }
}

// ============================================================================
// Layer 1b: Independent Reference Measure Evaluator + Proptest
// ============================================================================
//
// Cross-checks `build_partition_output_batch` (MEASURES, SUBSET union variables,
// RUNNING vs FINAL semantics, CLASSIFIER() nav, value nav, DISTINCT aggregates,
// SUM(PREV(...)), exclusions, empty matches, and WITH UNMATCHED ROWS) against a
// completely independent scalar reference implementation over randomized patterns
// (including reluctant quantifiers and SUBSETs not yet exposed via sqlparser 0.63).

fn ref_eval_measures_oracle(
    vals: &[Option<i64>],
    events: &[PartitionEvent],
    program: &Program,
    rows_per_match: RowsPerMatch,
) -> Vec<Vec<String>> {
    use datafusion_common::HashSet;

    let a_id = program.symbol_id("a").unwrap();
    let u_set = program.resolve_symbol_set("u").unwrap();
    let sym_name = |id: SymbolId| program.id_to_symbol[id as usize].as_str();
    let fmt_opt_i64 =
        |v: Option<i64>| v.map(|x| x.to_string()).unwrap_or_else(|| "NULL".into());
    let fmt_opt_str =
        |v: Option<&str>| v.map(str::to_string).unwrap_or_else(|| "NULL".into());

    let mut out = Vec::new();
    for event in events {
        match event {
            PartitionEvent::UnmatchedRow { row_idx } => {
                // Output schema for ALL ROWS: id (ORDER BY), 15 measures (all NULL), val
                let mut row = vec![(*row_idx as i64).to_string()];
                for _ in 0..15 {
                    row.push("NULL".into());
                }
                row.push(fmt_opt_i64(vals[*row_idx]));
                out.push(row);
            }
            PartitionEvent::Match(found) => {
                let emit_positions: Vec<(Option<usize>, usize, bool)> =
                    if rows_per_match.is_one_row() {
                        if found.steps.is_empty() {
                            vec![(None, found.start_row, true)]
                        } else {
                            vec![(Some(found.steps.len() - 1), found.start_row, false)]
                        }
                    } else if found.steps.is_empty() {
                        if rows_per_match.show_empty_matches() {
                            vec![(None, found.start_row, true)]
                        } else {
                            vec![]
                        }
                    } else {
                        found
                            .steps
                            .iter()
                            .enumerate()
                            .filter(|(_, s)| !s.excluded)
                            .map(|(i, s)| (Some(i), s.row_idx, false))
                            .collect()
                    };

                for (step_pos, source_row, is_empty) in emit_positions {
                    let run_horizon: &[MatchStep] = if is_empty {
                        &[]
                    } else {
                        match step_pos {
                            Some(idx) => &found.steps[..=idx],
                            None => &[],
                        }
                    };
                    let fin_horizon: &[MatchStep] =
                        if is_empty { &[] } else { &found.steps };

                    let mn = found.match_number;
                    let cls_run = run_horizon.last().map(|s| sym_name(s.sym));
                    let cls_fin = fin_horizon.last().map(|s| sym_name(s.sym));
                    let cls_u_run = run_horizon
                        .iter()
                        .rev()
                        .find(|s| u_set.contains(s.sym))
                        .map(|s| sym_name(s.sym));
                    let next_cls_run = if run_horizon.is_empty() {
                        None
                    } else {
                        found.steps.get(run_horizon.len()).map(|s| sym_name(s.sym))
                    };
                    let first_a_val = run_horizon
                        .iter()
                        .find(|s| s.sym == a_id)
                        .and_then(|s| vals[s.row_idx]);
                    let last_u_val_fin = fin_horizon
                        .iter()
                        .rev()
                        .find(|s| u_set.contains(s.sym))
                        .and_then(|s| vals[s.row_idx]);
                    let prev_val = run_horizon
                        .last()
                        .and_then(|s| s.row_idx.checked_sub(1))
                        .and_then(|r| vals[r]);
                    let cnt_star = run_horizon.len() as i64;
                    let cnt_u_fin =
                        fin_horizon.iter().filter(|s| u_set.contains(s.sym)).count()
                            as i64;
                    let cnt_distinct = {
                        let mut set = HashSet::new();
                        for s in run_horizon {
                            if let Some(v) = vals[s.row_idx] {
                                set.insert(v);
                            }
                        }
                        set.len() as i64
                    };
                    let sum_distinct = {
                        let mut set = HashSet::new();
                        let mut acc: Option<i64> = None;
                        for s in run_horizon {
                            if let Some(v) = vals[s.row_idx]
                                && set.insert(v)
                            {
                                acc = Some(acc.unwrap_or(0) + v);
                            }
                        }
                        acc
                    };
                    let sum_prev = {
                        let mut acc: Option<i64> = None;
                        for s in run_horizon {
                            if let Some(prev_r) = s.row_idx.checked_sub(1)
                                && let Some(v) = vals[prev_r]
                            {
                                acc = Some(acc.unwrap_or(0) + v);
                            }
                        }
                        acc
                    };
                    let max_val =
                        run_horizon.iter().filter_map(|s| vals[s.row_idx]).max();
                    let min_val =
                        run_horizon.iter().filter_map(|s| vals[s.row_idx]).min();

                    let mut row = Vec::new();
                    if !rows_per_match.is_one_row() {
                        row.push((source_row as i64).to_string());
                    }
                    row.push(mn.to_string());
                    row.push(fmt_opt_str(cls_run));
                    row.push(fmt_opt_str(cls_fin));
                    row.push(fmt_opt_str(cls_u_run));
                    row.push(fmt_opt_str(next_cls_run));
                    row.push(fmt_opt_i64(first_a_val));
                    row.push(fmt_opt_i64(last_u_val_fin));
                    row.push(fmt_opt_i64(prev_val));
                    row.push(cnt_star.to_string());
                    row.push(cnt_u_fin.to_string());
                    row.push(cnt_distinct.to_string());
                    row.push(fmt_opt_i64(sum_distinct));
                    row.push(fmt_opt_i64(sum_prev));
                    row.push(fmt_opt_i64(max_val));
                    row.push(fmt_opt_i64(min_val));
                    if !rows_per_match.is_one_row() {
                        row.push(fmt_opt_i64(vals[source_row]));
                    }
                    out.push(row);
                }
            }
        }
    }
    out
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 500,
        max_shrink_iters: 1000,
        .. ProptestConfig::default()
    })]

    #[test]
    fn fuzz_measures_and_subset_against_reference_oracle(
        pattern in arb_pattern(),
        bitmasks in prop::collection::vec(0u64..8u64, 1..=8),
        raw_vals in prop::collection::vec(prop::option::of(-5i64..=5i64), 1..=8),
        skip in arb_skip(),
        rpm in arb_rows_per_match(),
    ) {
        use std::sync::Arc;
        use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch};
        use arrow::util::display::{ArrayFormatter, FormatOptions};
        use arrow_schema::{DataType, Field, Schema};
        use datafusion_match_recognize::ast::{
            AggKind, EvalSemantics, LogicalNavKind, PhysicalNavKind, SubsetDef,
        };
        use datafusion_match_recognize::measures::{
            PhysicalMeasure, PhysicalMeasureSlot, PhysicalMeasureSlotKind,
            build_partition_output_batch, build_physical_slot_schema,
        };
        use datafusion_physical_expr::expressions::Column;

        // Exclusions are forbidden with WITH UNMATCHED ROWS per SQL:2016.
        if rpm.include_unmatched_rows() && pattern.has_exclusion() {
            return Ok(());
        }

        let len = bitmasks.len().min(raw_vals.len());
        let bitmasks = bitmasks[..len].to_vec();
        let vals = raw_vals[..len].to_vec();

        let declared = vec!["a".into(), "b".into(), "c".into()];
        let subsets = vec![SubsetDef {
            name: "u".into(),
            members: vec!["a".into(), "b".into()],
        }];
        let program = Program::compile_with_subsets(
            &pattern,
            &declared,
            &subsets,
            MatchRecognizeConfig::default(),
        )
        .unwrap();
        let symbols = PartitionSymbols::from_bitmasks(bitmasks, 3);
        let mut matcher = Matcher::new(&program);
        let Ok((events, _)) = matcher.match_partition(&symbols, &skip, rpm) else {
            return Ok(());
        };

        let partition_batch = RecordBatch::try_from_iter(vec![
            ("id", Arc::new(Int64Array::from_iter_values(0..len as i64)) as ArrayRef),
            ("v", Arc::new(Int64Array::from(vals.clone())) as ArrayRef),
        ])
        .unwrap();

        let a_set = program.resolve_symbol_set("a");
        let u_set = program.resolve_symbol_set("u");
        let v_col: Arc<dyn datafusion_physical_expr::PhysicalExpr> =
            Arc::new(Column::new("v", 1));
        let slot0_col: Arc<dyn datafusion_physical_expr::PhysicalExpr> =
            Arc::new(Column::new("s0", 0));

        let make_measure = |alias: &str, sem: EvalSemantics, dt: DataType, kind: PhysicalMeasureSlotKind| {
            let slot = PhysicalMeasureSlot {
                slot_name: "s0".into(),
                semantics: sem,
                data_type: dt,
                nullable: true,
                kind,
            };
            let slot_schema = build_physical_slot_schema(std::slice::from_ref(&slot));
            PhysicalMeasure {
                alias: alias.into(),
                outer_expr: Arc::clone(&slot0_col),
                slot_schema,
                slots: vec![slot],
            }
        };

        let measures = vec![
            make_measure("mn", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::MatchNumber),
            make_measure("cls_run", EvalSemantics::Running, DataType::Utf8, PhysicalMeasureSlotKind::Classifier {
                symbol: None, symbol_set: None, logical_nav: None, logical_offset: 0, physical_nav: None, physical_offset: 0,
            }),
            make_measure("cls_fin", EvalSemantics::Final, DataType::Utf8, PhysicalMeasureSlotKind::Classifier {
                symbol: None, symbol_set: None, logical_nav: None, logical_offset: 0, physical_nav: None, physical_offset: 0,
            }),
            make_measure("cls_u_run", EvalSemantics::Running, DataType::Utf8, PhysicalMeasureSlotKind::Classifier {
                symbol: Some("u".into()), symbol_set: u_set.clone(), logical_nav: None, logical_offset: 0, physical_nav: None, physical_offset: 0,
            }),
            make_measure("next_cls_run", EvalSemantics::Running, DataType::Utf8, PhysicalMeasureSlotKind::Classifier {
                symbol: None, symbol_set: None, logical_nav: Some(LogicalNavKind::Last), logical_offset: 0, physical_nav: Some(PhysicalNavKind::Next), physical_offset: 1,
            }),
            make_measure("first_a_v", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Nav {
                logical_nav: Some(LogicalNavKind::First), logical_offset: 0, physical_nav: None, physical_offset: 0, symbol: Some("a".into()), symbol_set: a_set, inner_expr: Arc::clone(&v_col), uses_match_metadata: false,
            }),
            make_measure("last_u_v_fin", EvalSemantics::Final, DataType::Int64, PhysicalMeasureSlotKind::Nav {
                logical_nav: Some(LogicalNavKind::Last), logical_offset: 0, physical_nav: None, physical_offset: 0, symbol: Some("u".into()), symbol_set: u_set.clone(), inner_expr: Arc::clone(&v_col), uses_match_metadata: false,
            }),
            make_measure("prev_v", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Nav {
                logical_nav: Some(LogicalNavKind::Last), logical_offset: 0, physical_nav: Some(PhysicalNavKind::Prev), physical_offset: 1, symbol: None, symbol_set: None, inner_expr: Arc::clone(&v_col), uses_match_metadata: false,
            }),
            make_measure("cnt_star", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::CountStar, symbol: None, symbol_set: None, args: vec![], distinct: false, arg_physical_nav: None, uses_match_metadata: false,
            }),
            make_measure("cnt_u_fin", EvalSemantics::Final, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::CountSymbol, symbol: Some("u".into()), symbol_set: u_set, args: vec![], distinct: false, arg_physical_nav: None, uses_match_metadata: false,
            }),
            make_measure("cnt_dist", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::Count, symbol: None, symbol_set: None, args: vec![Arc::clone(&v_col)], distinct: true, arg_physical_nav: None, uses_match_metadata: false,
            }),
            make_measure("sum_dist", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::Sum, symbol: None, symbol_set: None, args: vec![Arc::clone(&v_col)], distinct: true, arg_physical_nav: None, uses_match_metadata: false,
            }),
            make_measure("sum_prev", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::Sum, symbol: None, symbol_set: None, args: vec![Arc::clone(&v_col)], distinct: false, arg_physical_nav: Some((PhysicalNavKind::Prev, 1)), uses_match_metadata: false,
            }),
            make_measure("max_v", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::Max, symbol: None, symbol_set: None, args: vec![Arc::clone(&v_col)], distinct: false, arg_physical_nav: None, uses_match_metadata: false,
            }),
            make_measure("min_v", EvalSemantics::Running, DataType::Int64, PhysicalMeasureSlotKind::Agg {
                agg_kind: AggKind::Min, symbol: None, symbol_set: None, args: vec![Arc::clone(&v_col)], distinct: false, arg_physical_nav: None, uses_match_metadata: false,
            }),
        ];

        let mut out_fields = Vec::new();
        if !rpm.is_one_row() {
            out_fields.push(Field::new("id", DataType::Int64, true));
        }
        for m in &measures {
            out_fields.push(Field::new(&m.alias, m.slots[0].data_type.clone(), true));
        }
        if !rpm.is_one_row() {
            out_fields.push(Field::new("v", DataType::Int64, true));
        }
        let out_schema = Arc::new(Schema::new(out_fields));

        let actual_batch = build_partition_output_batch(
            &partition_batch,
            &events,
            &program,
            &[],
            &[],
            &[Some(0)],
            &measures,
            rpm,
            &out_schema,
        )
        .unwrap();

        let expected = ref_eval_measures_oracle(&vals, &events, &program, rpm);
        let actual: Vec<Vec<String>> = match actual_batch {
            None => vec![],
            Some(b) => {
                let opts = FormatOptions::default().with_null("NULL");
                let fmts: Vec<ArrayFormatter<'_>> = b
                    .columns()
                    .iter()
                    .map(|c| ArrayFormatter::try_new(c.as_ref(), &opts).unwrap())
                    .collect();
                (0..b.num_rows())
                    .map(|r| {
                        (0..b.num_columns())
                            .map(|c| {
                                if b.column(c).is_null(r) {
                                    "NULL".to_string()
                                } else {
                                    fmts[c].value(r).to_string()
                                }
                            })
                            .collect()
                    })
                    .collect()
            }
        };

        prop_assert_eq!(
            actual,
            expected,
            "Measure oracle mismatch for pattern={}, skip={}, rpm={}, vals={:?}",
            pattern,
            skip,
            rpm,
            vals
        );
    }
}
