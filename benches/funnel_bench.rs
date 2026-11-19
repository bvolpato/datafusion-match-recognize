// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::matcher::{Matcher, PartitionSymbols};
use datafusion_match_recognize::pattern::Program;
use datafusion_match_recognize::{
    MatchRecognizeConfig, Pattern, SymbolRef, session_context_with_match_recognize,
};

fn build_synthetic_funnel_batch(num_users: usize, events_per_user: usize) -> RecordBatch {
    let total = num_users * events_per_user;
    let mut users = Vec::with_capacity(total);
    let mut ts = Vec::with_capacity(total);
    let mut events = Vec::with_capacity(total);

    for u in 0..num_users {
        for e in 0..events_per_user {
            users.push(u as i64);
            ts.push((e * 10) as i64);
            let ev = match (u + e) % 7 {
                0 => "view",
                2 => "cart",
                5 => "buy",
                _ => "other",
            };
            events.push(ev);
        }
    }

    RecordBatch::try_from_iter(vec![
        ("user_id", Arc::new(Int64Array::from(users)) as ArrayRef),
        ("ts", Arc::new(Int64Array::from(ts)) as ArrayRef),
        (
            "event_name",
            Arc::new(StringArray::from(events)) as ArrayRef,
        ),
    ])
    .expect("valid batch")
}

fn bench_nfa_linear_backtracking(c: &mut Criterion) {
    let mut group = c.benchmark_group("nfa_adversarial");
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
    let program = Program::compile(
        &pattern,
        &["a".into(), "b".into()],
        MatchRecognizeConfig::default(),
    )
    .expect("compile");

    for &len in &[1_000usize, 10_000] {
        // Every row matches A, none match B (classic catastrophic backtracking trap).
        let symbols = PartitionSymbols::from_bitmasks(vec![0b01; len], 2);
        group.bench_with_input(BenchmarkId::new("(A+)+B_no_match", len), &len, |b, _| {
            let mut matcher = Matcher::new(&program);
            b.iter(|| {
                let mut stats = Default::default();
                let res = matcher.find_match_at(&symbols, 0, &mut stats);
                assert!(res.is_none());
            });
        });
    }
    group.finish();
}

fn bench_sql_funnel(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
    let mut group = c.benchmark_group("sql_3step_funnel");

    for &(users, events_per_user) in &[(100usize, 50usize), (1_000, 50)] {
        let batch = build_synthetic_funnel_batch(users, events_per_user);
        let sql = r#"
            SELECT * FROM events
            MATCH_RECOGNIZE (
                PARTITION BY user_id
                ORDER BY ts
                MEASURES
                    FIRST(a.ts) AS a_ts,
                    FIRST(b.ts) AS b_ts,
                    FIRST(c.ts) AS c_ts,
                    FIRST(c.ts) - FIRST(a.ts) AS funnel_time
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((a any0* b any1* c) | (a any0* b) | a)
                DEFINE
                    a AS event_name = 'view',
                    any0 AS event_name <> 'cart',
                    b AS event_name = 'cart',
                    any1 AS event_name <> 'buy',
                    c AS event_name = 'buy'
            )
        "#;
        group.bench_with_input(
            BenchmarkId::new("rows", users * events_per_user),
            &(users, events_per_user),
            |b, _| {
                b.to_async(&rt).iter(|| async {
                    let ctx = session_context_with_match_recognize(
                        SessionConfig::new().with_target_partitions(4),
                        MatchRecognizeConfig::default(),
                    )
                    .expect("ctx");
                    ctx.register_batch("events", batch.clone()).expect("reg");
                    let batches = ctx
                        .sql(sql)
                        .await
                        .expect("sql")
                        .collect()
                        .await
                        .expect("collect");
                    assert!(!batches.is_empty());
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_nfa_linear_backtracking, bench_sql_funnel);
criterion_main!(benches);
