// SPDX-License-Identifier: Apache-2.0

//! Layer 4 verification: adversarial anti-backtracking complexity gate proving PikeVM thread
//! deduplication executes in linear time O(N * |states|) even on classic catastrophic patterns.

use std::time::Instant;

use datafusion_match_recognize::ast::{MatchRecognizeConfig, Pattern, SymbolRef};
use datafusion_match_recognize::matcher::{MatchStats, Matcher, PartitionSymbols};
use datafusion_match_recognize::pattern::Program;

fn assert_linear_no_match(name: &str, pattern: Pattern, num_rows: usize) {
    let program = Program::compile(
        &pattern,
        &["a".into(), "b".into()],
        MatchRecognizeConfig::default(),
    )
    .expect("compile");
    // Every row matches A (bit 0), no row matches B (bit 1).
    let symbols = PartitionSymbols::from_bitmasks(vec![0b01; num_rows], 2);
    let mut matcher = Matcher::new(&program);
    let mut stats = MatchStats::default();

    let start = Instant::now();
    let res = matcher.find_match_at(&symbols, 0, &mut stats);
    let elapsed = start.elapsed();

    assert!(res.is_none(), "{name} should not match");
    let max_expected_steps = num_rows * (program.instructions.len() * 4 + 16);
    assert!(
        stats.total_thread_steps <= max_expected_steps,
        "{name}: thread steps {} exceeded linear bound {} (instructions={}, rows={})",
        stats.total_thread_steps,
        max_expected_steps,
        program.instructions.len(),
        num_rows
    );
    assert!(
        elapsed.as_secs_f64() < 2.0,
        "{name}: took {elapsed:?} for {num_rows} rows (expected well under 2s)"
    );
}

#[test]
fn test_catastrophic_backtracking_nested_plus() {
    // (A+)+ B on 50,000 rows of A
    let pattern = Pattern::Concat(vec![
        Pattern::Repetition {
            inner: Box::new(Pattern::Repetition {
                inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
                min: 1,
                max: None,
                reluctant: false,
            }),
            min: 1,
            max: None,
            reluctant: false,
        },
        Pattern::Symbol(SymbolRef::Named("b".into())),
    ]);
    assert_linear_no_match("(A+)+ B", pattern, 50_000);
}

#[test]
fn test_catastrophic_backtracking_nested_star() {
    // (A*)* B on 50,000 rows of A
    let pattern = Pattern::Concat(vec![
        Pattern::Repetition {
            inner: Box::new(Pattern::Repetition {
                inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
                min: 0,
                max: None,
                reluctant: false,
            }),
            min: 0,
            max: None,
            reluctant: false,
        },
        Pattern::Symbol(SymbolRef::Named("b".into())),
    ]);
    assert_linear_no_match("(A*)* B", pattern, 50_000);
}

#[test]
fn test_catastrophic_backtracking_overlapping_alternation() {
    // (A | A)* B on 50,000 rows of A
    let pattern = Pattern::Concat(vec![
        Pattern::Repetition {
            inner: Box::new(Pattern::Alternation(vec![
                Pattern::Symbol(SymbolRef::Named("a".into())),
                Pattern::Symbol(SymbolRef::Named("a".into())),
            ])),
            min: 0,
            max: None,
            reluctant: false,
        },
        Pattern::Symbol(SymbolRef::Named("b".into())),
    ]);
    assert_linear_no_match("(A | A)* B", pattern, 50_000);
}

#[test]
fn test_catastrophic_backtracking_classic_a_opt_n() {
    // (A?){25} A{25} B on 20,000 rows of A
    let pattern = Pattern::Concat(vec![
        Pattern::Repetition {
            inner: Box::new(Pattern::Repetition {
                inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
                min: 0,
                max: Some(1),
                reluctant: false,
            }),
            min: 25,
            max: Some(25),
            reluctant: false,
        },
        Pattern::Repetition {
            inner: Box::new(Pattern::Symbol(SymbolRef::Named("a".into()))),
            min: 25,
            max: Some(25),
            reluctant: false,
        },
        Pattern::Symbol(SymbolRef::Named("b".into())),
    ]);
    assert_linear_no_match("(A?){25} A{25} B", pattern, 20_000);
}
