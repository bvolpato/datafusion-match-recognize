// SPDX-License-Identifier: Apache-2.0

//! Runs DuckDB's own `benchmark/micro/match_recognize/*.benchmark` workloads on DataFusion
//! (and verifies exact result parity against DuckDB 2.1 checksums saved by
//! `scripts/bench_duckdb_suite.sh`).

use std::collections::BTreeMap;

use arrow::array::{Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::{ParquetReadOptions, SessionConfig};
use datafusion_common::instant::Instant;
use datafusion_match_recognize::{
    MatchRecognizeConfig, session_context_with_match_recognize,
};

const WARMUP_ITERS: usize = 1;
const MEASURED_ITERS: usize = 3;

fn format_df_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let options = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for batch in batches {
        let formatters: Vec<ArrayFormatter<'_>> = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &options).unwrap())
            .collect();
        for r in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for (c, fmt) in formatters.iter().enumerate() {
                if batch.column(c).is_null(r) {
                    row.push("NULL".to_string());
                } else {
                    row.push(fmt.value(r).to_string());
                }
            }
            rows.push(row);
        }
    }
    rows
}

fn median(mut vals: Vec<f64>) -> f64 {
    vals.sort_by(|a, b| a.total_cmp(b));
    vals[vals.len() / 2]
}

fn suite() -> Vec<(&'static str, &'static str, &'static str)> {
    // (benchmark_name, table_name, sql)
    vec![
        (
            "sessionize_10m",
            "hits_sessionize",
            r#"
            SELECT count(*), max(len) FROM hits_sessionize MATCH_RECOGNIZE(
                PARTITION BY visitor ORDER BY seq
                MEASURES count(*) AS len
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A B*)
                DEFINE B AS B.ts - PREV(B.ts) <= 100
            )
            "#,
        ),
        (
            "streak_10m",
            "readings_streak",
            r#"
            SELECT max(len) FROM readings_streak MATCH_RECOGNIZE(
                PARTITION BY sensor ORDER BY i
                MEASURES count(*) AS len
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A B*)
                DEFINE B AS B.v > PREV(B.v)
            )
            "#,
        ),
        (
            "funnel_three_step_10m",
            "hits_funnel",
            r#"
            SELECT count(*) FROM hits_funnel MATCH_RECOGNIZE(
                PARTITION BY visitor ORDER BY seq
                MEASURES count(*) AS steps
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN (BROWSE SELECT_IT CHECKOUT)
                DEFINE BROWSE AS BROWSE.action = 'browse', SELECT_IT AS SELECT_IT.action = 'select',
                       CHECKOUT AS CHECKOUT.action = 'checkout'
            )
            "#,
        ),
        (
            "funnel_two_step_10m",
            "hits_funnel",
            r#"
            SELECT count(*) FROM hits_funnel MATCH_RECOGNIZE(
                PARTITION BY visitor ORDER BY seq
                MEASURES count(*) AS steps
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((BROWSE SELECT_IT) | BROWSE)
                DEFINE BROWSE AS BROWSE.action = 'browse', SELECT_IT AS SELECT_IT.action = 'select'
            )
            "#,
        ),
        (
            "selectivity_mid_10m",
            "ticks_sel",
            r#"
            SELECT count(*) FROM ticks_sel MATCH_RECOGNIZE(
                ORDER BY i
                MEASURES count(*) AS len
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A B{2})
                DEFINE A AS A.v < 25, B AS B.v < 50
            )
            "#,
        ),
        (
            "aggregate_define_1m",
            "readings_agg",
            r#"
            SELECT count(*) FROM readings_agg MATCH_RECOGNIZE(
                ORDER BY i
                MEASURES count(*) AS len
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A+)
                DEFINE A AS A.v >= avg(A.v)
            )
            "#,
        ),
        (
            "bin_packing_1m",
            "items_bin",
            r#"
            SELECT count(*), sum(total), sum(packed) FROM items_bin MATCH_RECOGNIZE(
                PARTITION BY bin ORDER BY i
                MEASURES sum(X.w) AS total, count(X.i) AS packed
                ONE ROW PER MATCH
                PATTERN ((X | ({- REST -}))+)
                DEFINE X AS sum(X.w) <= 20
            )
            "#,
        ),
        (
            "merge_ranges_1m",
            "spans_merge",
            r#"
            SELECT count(*), sum(n), max(e - s) FROM spans_merge MATCH_RECOGNIZE(
                ORDER BY s, e
                MEASURES first(s) AS s, max(e) AS e, count(*) AS n
                ONE ROW PER MATCH
                PATTERN (J* L)
                DEFINE J AS NEXT(s) <= max(e)
            )
            "#,
        ),
        (
            "shoulders_1m",
            "ticks_shoulders",
            r#"
            SELECT count(*), sum(len) FROM ticks_shoulders MATCH_RECOGNIZE(
                ORDER BY i
                MEASURES last(i) - first(i) + 1 AS len
                ONE ROW PER MATCH AFTER MATCH SKIP PAST LAST ROW
                PATTERN (S UP+ DOWN+ UP2+ DOWN+ UP+ DOWN2+)
                DEFINE UP AS v > PREV(v), DOWN AS v < PREV(v), UP2 AS v > max(UP.v), DOWN2 AS v < min(DOWN.v)
            )
            "#,
        ),
        (
            "navigation_200k",
            "readings_nav",
            r#"
            SELECT count(*) FROM readings_nav MATCH_RECOGNIZE(
                ORDER BY i
                MEASURES i AS start_i
                ONE ROW PER MATCH
                PATTERN (A B+)
                DEFINE A AS true, B AS v > LAST(A.v)
            )
            "#,
        ),
        (
            "quantifier_500k",
            "readings_quant",
            r#"
            SELECT count(*) FROM readings_quant MATCH_RECOGNIZE(
                ORDER BY i
                MEASURES i AS start_i
                ALL ROWS PER MATCH
                PATTERN (A+ B+ A+)
                DEFINE A AS v < 12, B AS v >= 12
            )
            "#,
        ),
        (
            "no_match_200k",
            "runs_no_match",
            r#"
            SELECT count(*) FROM runs_no_match MATCH_RECOGNIZE(
                ORDER BY seq
                MEASURES count(*) AS steps
                ONE ROW PER MATCH
                PATTERN (BROWSE+ SELECT_IT+ CHECKOUT+ REFUND)
                DEFINE BROWSE AS BROWSE.action = 'browse', SELECT_IT AS SELECT_IT.action = 'select',
                       CHECKOUT AS CHECKOUT.action = 'checkout', REFUND AS REFUND.action = 'refund'
            )
            "#,
        ),
        (
            "nested_quantifier_2m",
            "run_nested",
            r#"
            SELECT count(*) FROM run_nested MATCH_RECOGNIZE(
                ORDER BY i
                MEASURES count(*) AS len
                ONE ROW PER MATCH
                PATTERN ((A+)+ B)
                DEFINE A AS A.c = 'a', B AS B.c = 'b'
            )
            "#,
        ),
    ]
}

fn main() {
    let data_dir = std::env::var("BENCH_DATA_DIR")
        .expect("BENCH_DATA_DIR must point to generated Parquet tables");
    let checksum_file = std::env::var("BENCH_CHECKSUM_FILE").ok();
    let filter = std::env::var("BENCH_FILTER").ok();
    let threads: usize = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async move {
        let ctx = session_context_with_match_recognize(
            SessionConfig::new().with_target_partitions(threads),
            MatchRecognizeConfig::default(),
        )
        .expect("df ctx");

        let saved_checksums: BTreeMap<String, Vec<Vec<String>>> = checksum_file
            .as_deref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();

        let mut registered = datafusion_common::HashSet::new();
        for (name, table, sql) in suite() {
            if let Some(f) = &filter
                && !name.contains(f.as_str())
            {
                continue;
            }
            if registered.insert(table) {
                let path = format!("{data_dir}/{table}");
                ctx.register_parquet(table, &path, ParquetReadOptions::default())
                    .await
                    .unwrap_or_else(|e| panic!("register {table} at {path}: {e}"));
            }

            for _ in 0..WARMUP_ITERS {
                let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
            }
            let mut times = Vec::with_capacity(MEASURED_ITERS);
            let mut out = Vec::new();
            for _ in 0..MEASURED_ITERS {
                let t0 = Instant::now();
                let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
                times.push(t0.elapsed().as_secs_f64() * 1000.0);
                out = format_df_rows(&batches);
            }
            let med = median(times);
            if let Some(expected) = saved_checksums.get(name) {
                assert_eq!(
                    &out, expected,
                    "Result mismatch on DuckDB benchmark '{name}': DF={out:?} vs DuckDB={expected:?}"
                );
            }
            println!(
                "DF     | `{name}` | wall={med:.1} ms | out=({}) | parity={}",
                out[0].join(", "),
                if saved_checksums.contains_key(name) {
                    "verified"
                } else {
                    "unchecked"
                }
            );
        }
    });
}
