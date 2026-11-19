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
                if let Some(m) = ref_eval(branch, pos, exclude_depth, symbols, program, steps, cont)
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
                let seq: Vec<Pattern> = order.into_iter().map(|i| items[i].clone()).collect();
                if let Some(m) =
                    ref_eval_slice(&seq, pos, exclude_depth, symbols, program, steps, cont)
                {
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

#[allow(clippy::too_many_arguments)]
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
                if rows_per_match.include_unmatched_rows() && start_row >= max_matched_end {
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
            Program::compile(&pattern, &declared, MatchRecognizeConfig::default()).unwrap();
        let symbols = PartitionSymbols::from_bitmasks(bitmasks, 3);
        for skip in [AfterMatchSkip::PastLastRow, AfterMatchSkip::ToNextRow] {
            for rpm in [
                RowsPerMatch::OneRow,
                RowsPerMatch::AllRowsShowEmpty,
                RowsPerMatch::AllRowsOmitEmpty,
                RowsPerMatch::AllRowsWithUnmatched,
            ] {
                let expected = ref_match_partition(&pattern, &symbols, &program, &skip, rpm);
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
