// SPDX-License-Identifier: Apache-2.0

//! Reproducible head-to-head benchmark comparing `datafusion-match-recognize` against a live
//! Trino container on 5 canonical SQL:2016 `MATCH_RECOGNIZE` workloads (1,000,000 rows).
//!
//! Fairness controls:
//! - Identical deterministic 1M-row dataset loaded into memory on both engines (DataFusion
//!   `MemTable` vs. Trino `memory` catalog).
//! - DataFusion Tokio runtime and `target_partitions` pinned to `BENCH_THREADS` (default: 10,
//!   matching Colima's vCPU count).
//! - Warmup runs before measured runs on both engines (warms Trino JIT).
//! - Every query is wrapped in `SELECT COUNT(*), SUM(...)` so result transfer is 1 row, and
//!   outputs are asserted equal between DataFusion and Trino.
//!
//! Run:
//!   docker run -d --name mr-trino-bench -p 18080:8080 trinodb/trino:latest
//!   cargo run --release --example bench_vs_trino
//!   docker rm -f mr-trino-bench

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::{MatchRecognizeConfig, session_context_with_match_recognize};
use serde::Deserialize;

const NUM_PARTITIONS: usize = 20_000;
const ROWS_PER_PARTITION: usize = 50; // 20,000 * 50 = 1,000,000 rows
const WARMUP_ITERS: usize = 3;
const MEASURED_ITERS: usize = 5;

#[derive(Debug, Deserialize)]
struct TrinoStats {
    #[serde(rename = "elapsedTimeMillis")]
    elapsed_time_millis: Option<u64>,
    #[serde(rename = "cpuTimeMillis")]
    cpu_time_millis: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct TrinoResponse {
    #[serde(rename = "nextUri")]
    next_uri: Option<String>,
    data: Option<Vec<Vec<serde_json::Value>>>,
    error: Option<serde_json::Value>,
    stats: Option<TrinoStats>,
}

struct TrinoRunResult {
    rows: Vec<Vec<String>>,
    wall_ms: f64,
    server_elapsed_ms: u64,
    server_cpu_ms: u64,
}

async fn run_trino(
    client: &reqwest::Client,
    base_url: &str,
    sql: &str,
) -> Result<TrinoRunResult, String> {
    let start = Instant::now();
    let mut resp: TrinoResponse = client
        .post(format!("{base_url}/v1/statement"))
        .header("X-Trino-User", "bench")
        .header("X-Trino-Catalog", "memory")
        .header("X-Trino-Schema", "default")
        .body(sql.to_string())
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json()
        .await
        .map_err(|e| e.to_string())?;

    let mut rows = Vec::new();
    let mut last_stats: Option<TrinoStats> = None;
    loop {
        if let Some(err) = resp.error {
            return Err(format!("Trino error: {err}"));
        }
        if let Some(st) = resp.stats {
            last_stats = Some(st);
        }
        if let Some(data) = resp.data {
            for r in data {
                rows.push(
                    r.into_iter()
                        .map(|v| match v {
                            serde_json::Value::Null => "NULL".to_string(),
                            serde_json::Value::Number(n) => n.to_string(),
                            serde_json::Value::String(s) => s,
                            other => other.to_string(),
                        })
                        .collect(),
                );
            }
        }
        let Some(next) = resp.next_uri else {
            break;
        };
        resp = client
            .get(&next)
            .header("X-Trino-User", "bench")
            .send()
            .await
            .map_err(|e| e.to_string())?
            .json()
            .await
            .map_err(|e| e.to_string())?;
    }
    let wall_ms = start.elapsed().as_secs_f64() * 1000.0;
    let server_elapsed_ms = last_stats
        .as_ref()
        .and_then(|s| s.elapsed_time_millis)
        .unwrap_or(0);
    let server_cpu_ms = last_stats
        .as_ref()
        .and_then(|s| s.cpu_time_millis)
        .unwrap_or(0);
    Ok(TrinoRunResult {
        rows,
        wall_ms,
        server_elapsed_ms,
        server_cpu_ms,
    })
}

/// Deterministic 1,000,000-row table matching Trino's `UNNEST(sequence(...))` generation formula
/// so both engines evaluate the exact same rows without huge INSERT payloads:
///   p     = 1..=20_000
///   seq   = 1..=50
///   ts    = seq * 10
///   ev    = (p * 31 + seq * 17) % 10
///   price = 100 + ((p * 7 + seq * 13) % 41) - ((seq * 3) % 19)
///   sparse_ev = CASE WHEN p % 100 = 0 AND seq = 5 THEN 0 WHEN seq = 25 THEN 1 WHEN seq = 45 THEN 2 ELSE 9 END
fn build_df_batch() -> RecordBatch {
    let total = NUM_PARTITIONS * ROWS_PER_PARTITION;
    let mut p_col = Vec::with_capacity(total);
    let mut seq_col = Vec::with_capacity(total);
    let mut ts_col = Vec::with_capacity(total);
    let mut ev_col = Vec::with_capacity(total);
    let mut price_col = Vec::with_capacity(total);
    let mut sparse_col = Vec::with_capacity(total);

    for p in 1..=(NUM_PARTITIONS as i64) {
        for seq in 1..=(ROWS_PER_PARTITION as i64) {
            p_col.push(p);
            seq_col.push(seq);
            ts_col.push(seq * 10);
            ev_col.push((p * 31 + seq * 17) % 10);
            price_col.push(100 + ((p * 7 + seq * 13) % 41) - ((seq * 3) % 19));
            let sp = if p % 100 == 0 && seq == 5 {
                0
            } else if seq == 25 {
                1
            } else if seq == 45 {
                2
            } else {
                9
            };
            sparse_col.push(sp);
        }
    }

    RecordBatch::try_from_iter(vec![
        ("p", Arc::new(Int64Array::from(p_col)) as ArrayRef),
        ("seq", Arc::new(Int64Array::from(seq_col))),
        ("ts", Arc::new(Int64Array::from(ts_col))),
        ("ev", Arc::new(Int64Array::from(ev_col))),
        ("price", Arc::new(Int64Array::from(price_col))),
        ("sparse_ev", Arc::new(Int64Array::from(sparse_col))),
    ])
    .expect("valid batch")
}

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

fn main() {
    let threads: usize = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let trino_url =
        std::env::var("TRINO_URL").unwrap_or_else(|_| "http://localhost:18080".to_string());

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async move {
        let client = reqwest::Client::new();

        // Wait for Trino to finish starting up.
        for _ in 0..60 {
            if let Ok(r) = client.get(format!("{trino_url}/v1/info")).send().await
                && let Ok(txt) = r.text().await
                && txt.contains("\"starting\":false")
            {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_secs(1)).await;
        }

        // Load identical 1,000,000-row table into Trino memory catalog.
        let _ = run_trino(&client, &trino_url, "DROP TABLE IF EXISTS memory.default.bench_events")
            .await;
        let create_trino_sql = format!(
            r#"
            CREATE TABLE memory.default.bench_events AS
            WITH parts AS (
                SELECT CAST(g * 1000 + k AS BIGINT) AS p
                FROM UNNEST(sequence(0, {})) AS tg(g)
                CROSS JOIN UNNEST(sequence(1, 1000)) AS tk(k)
            )
            SELECT
                p,
                CAST(seq AS BIGINT) AS seq,
                CAST(seq * 10 AS BIGINT) AS ts,
                CAST((p * 31 + seq * 17) % 10 AS BIGINT) AS ev,
                CAST(100 + ((p * 7 + seq * 13) % 41) - ((seq * 3) % 19) AS BIGINT) AS price,
                CAST(CASE
                    WHEN p % 100 = 0 AND seq = 5 THEN 0
                    WHEN seq = 25 THEN 1
                    WHEN seq = 45 THEN 2
                    ELSE 9
                END AS BIGINT) AS sparse_ev
            FROM parts
            CROSS JOIN UNNEST(sequence(1, {ROWS_PER_PARTITION})) AS t2(seq)
            "#,
            (NUM_PARTITIONS / 1000) - 1
        );
        run_trino(&client, &trino_url, &create_trino_sql)
            .await
            .expect("create Trino bench_events");

        // Load identical 1,000,000-row table into DataFusion MemTable with `threads` partitions.
        let ctx = session_context_with_match_recognize(
            SessionConfig::new().with_target_partitions(threads),
            MatchRecognizeConfig::default(),
        )
        .expect("df ctx");
        let batch = build_df_batch();
        let chunk_rows = batch.num_rows() / threads;
        let mut mem_parts = Vec::with_capacity(threads);
        for i in 0..threads {
            let start = i * chunk_rows;
            let len = if i + 1 == threads {
                batch.num_rows() - start
            } else {
                chunk_rows
            };
            mem_parts.push(vec![batch.slice(start, len)]);
        }
        let table = datafusion::catalog::memory::MemTable::try_new(batch.schema(), mem_parts)
            .expect("memtable");
        ctx.register_table("bench_events", Arc::new(table))
            .expect("register");

        let workloads: Vec<(&str, &str)> = vec![
            (
                "1_funnel_3step_stateless",
                r#"
                SELECT COUNT(*) AS cnt, COALESCE(SUM(dt), 0) AS total_dt
                FROM bench_events
                MATCH_RECOGNIZE (
                    PARTITION BY p
                    ORDER BY seq
                    MEASURES
                        CASE WHEN c.ts IS NOT NULL THEN FIRST(c.ts) - FIRST(a.ts) ELSE 0 END AS dt
                    ONE ROW PER MATCH
                    AFTER MATCH SKIP PAST LAST ROW
                    PATTERN ((a any0* b any1* c) | (a any0* b) | a)
                    DEFINE
                        a AS ev = 0,
                        any0 AS ev <> 1,
                        b AS ev = 1,
                        any1 AS ev <> 2,
                        c AS ev = 2
                )
                "#,
            ),
            (
                "2_funnel_sparse_1pct_fast_skip",
                r#"
                SELECT COUNT(*) AS cnt, COALESCE(SUM(dt), 0) AS total_dt
                FROM bench_events
                MATCH_RECOGNIZE (
                    PARTITION BY p
                    ORDER BY seq
                    MEASURES FIRST(c.ts) - FIRST(a.ts) AS dt
                    ONE ROW PER MATCH
                    AFTER MATCH SKIP PAST LAST ROW
                    PATTERN (a any0* b any1* c)
                    DEFINE
                        a AS sparse_ev = 0,
                        any0 AS sparse_ev <> 1,
                        b AS sparse_ev = 1,
                        any1 AS sparse_ev <> 2,
                        c AS sparse_ev = 2
                )
                "#,
            ),
            (
                "3_overlapping_skip_next_all_rows",
                r#"
                SELECT COUNT(*) AS cnt, COALESCE(SUM(s_price), 0) AS total_s
                FROM bench_events
                MATCH_RECOGNIZE (
                    PARTITION BY p
                    ORDER BY seq
                    MEASURES SUM(price) AS s_price
                    ALL ROWS PER MATCH OMIT EMPTY MATCHES
                    AFTER MATCH SKIP TO NEXT ROW
                    PATTERN (a b+ c)
                    DEFINE
                        a AS price >= 100,
                        b AS price < 115,
                        c AS price >= 115
                )
                "#,
            ),
            (
                "4_stock_vshape_stateful_prev",
                r#"
                SELECT COUNT(*) AS cnt, COALESCE(SUM(bottom_p), 0) AS total_bp
                FROM bench_events
                MATCH_RECOGNIZE (
                    PARTITION BY p
                    ORDER BY seq
                    MEASURES LAST(down.price) AS bottom_p
                    ONE ROW PER MATCH
                    AFTER MATCH SKIP TO LAST up
                    PATTERN (strt down+ up+)
                    DEFINE
                        down AS down.price < PREV(down.price),
                        up AS up.price > PREV(up.price)
                )
                "#,
            ),
            (
                "5_time_window_funnel_stateful",
                r#"
                SELECT COUNT(*) AS cnt, COALESCE(SUM(dt), 0) AS total_dt
                FROM bench_events
                MATCH_RECOGNIZE (
                    PARTITION BY p
                    ORDER BY seq
                    MEASURES LAST(b.ts) - FIRST(a.ts) AS dt
                    ONE ROW PER MATCH
                    AFTER MATCH SKIP PAST LAST ROW
                    PATTERN (a any0* b)
                    DEFINE
                        a AS ev = 0,
                        any0 AS ev <> 2,
                        b AS ev = 2 AND b.ts - FIRST(a.ts) <= 150
                )
                "#,
            ),
        ];

        println!(
            "Benchmark: {} rows ({} partitions x {} rows), DataFusion threads={}, Trino Docker (Colima), warmup={}, runs={}\n",
            NUM_PARTITIONS * ROWS_PER_PARTITION,
            NUM_PARTITIONS,
            ROWS_PER_PARTITION,
            threads,
            WARMUP_ITERS,
            MEASURED_ITERS
        );
        println!(
            "| Workload | DataFusion p50 (ms) | Trino wall p50 (ms) | Trino exec p50 (ms) | Trino CPU p50 (ms) | Speedup (vs Trino exec) | Result Parity |"
        );
        println!("| --- | ---: | ---: | ---: | ---: | ---: | --- |");

        for (name, sql) in workloads {
            // Warmup both engines
            for _ in 0..WARMUP_ITERS {
                let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
                let _ = run_trino(&client, &trino_url, sql).await.unwrap();
            }

            let mut df_times = Vec::with_capacity(MEASURED_ITERS);
            let mut df_out = Vec::new();
            for _ in 0..MEASURED_ITERS {
                let t0 = Instant::now();
                let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
                df_times.push(t0.elapsed().as_secs_f64() * 1000.0);
                df_out = format_df_rows(&batches);
            }

            let mut trino_wall = Vec::with_capacity(MEASURED_ITERS);
            let mut trino_exec = Vec::with_capacity(MEASURED_ITERS);
            let mut trino_cpu = Vec::with_capacity(MEASURED_ITERS);
            let mut trino_out = Vec::new();
            for _ in 0..MEASURED_ITERS {
                let res = run_trino(&client, &trino_url, sql).await.unwrap();
                trino_wall.push(res.wall_ms);
                trino_exec.push(res.server_elapsed_ms as f64);
                trino_cpu.push(res.server_cpu_ms as f64);
                trino_out = res.rows;
            }

            assert_eq!(
                df_out, trino_out,
                "Result mismatch on workload {name}: DF={df_out:?} vs Trino={trino_out:?}"
            );

            let df_med = median(df_times);
            let tw_med = median(trino_wall);
            let te_med = median(trino_exec);
            let tc_med = median(trino_cpu);
            let ratio = te_med / df_med;
            println!(
                "| `{name}` | {df_med:.1} | {tw_med:.1} | {te_med:.1} | {tc_med:.1} | {ratio:.2}x | exact ({}) |",
                df_out[0].join(", ")
            );
        }
    });
}
