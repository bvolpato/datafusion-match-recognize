// SPDX-License-Identifier: Apache-2.0

//! Reproducible head-to-head benchmark comparing `datafusion-match-recognize` against Trino 484
//! on five canonical SQL:2016 `MATCH_RECOGNIZE` workloads (1,000,000 rows).
//!
//! Designed so both engines can run sequentially inside identical Linux ARM64 Docker containers
//! (`--cpus=8 --memory=12g`) on the same VM with zero host-vs-VM asymmetry and zero CPU
//! contention, over either in-memory tables or shared Parquet files written by Trino.
//!
//! Environment variables:
//! - `BENCH_MODE`: `both` (default), `trino`, or `df`
//! - `BENCH_STORAGE`: `memory` (default) or `parquet`
//! - `BENCH_THREADS`: DataFusion Tokio worker threads + `target_partitions` (default: 8)
//! - `BENCH_PARQUET_DIR`: directory of Parquet files for `BENCH_STORAGE=parquet`
//!   (default: `/benchdata/metastore/default/bench_events`)
//! - `BENCH_CHECKSUM_FILE`: optional JSON path to write (in `trino` mode) / verify (in `df` mode)
//!   exact result parity across sequential container runs
//! - `TRINO_URL`: Trino HTTP endpoint (default: `http://localhost:18080`)

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int64Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::{ParquetReadOptions, SessionConfig};
use datafusion_common::instant::Instant;
use datafusion_match_recognize::{
    MatchRecognizeConfig, session_context_with_match_recognize,
};
use serde::Deserialize;

const DEFAULT_PARTITIONS: usize = 200_000;
const ROWS_PER_PARTITION: usize = 50; // 200,000 * 50 = 10,000,000 rows
const WARMUP_ITERS: usize = 2;
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
    catalog: &str,
    sql: &str,
) -> Result<TrinoRunResult, String> {
    let start = Instant::now();
    let mut resp: TrinoResponse = client
        .post(format!("{base_url}/v1/statement"))
        .header("X-Trino-User", "bench")
        .header("X-Trino-Catalog", catalog)
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

fn build_df_batch(num_partitions: usize) -> RecordBatch {
    let total = num_partitions * ROWS_PER_PARTITION;
    let mut p_col = Vec::with_capacity(total);
    let mut seq_col = Vec::with_capacity(total);
    let mut ts_col = Vec::with_capacity(total);
    let mut ev_col = Vec::with_capacity(total);
    let mut price_col = Vec::with_capacity(total);
    let mut sparse_col = Vec::with_capacity(total);

    for p in 1..=(num_partitions as i64) {
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

fn workloads() -> Vec<(&'static str, &'static str)> {
    vec![
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
    ]
}

fn main() {
    let mode = std::env::var("BENCH_MODE").unwrap_or_else(|_| "both".to_string());
    let storage = std::env::var("BENCH_STORAGE").unwrap_or_else(|_| "memory".to_string());
    let num_partitions: usize = std::env::var("BENCH_PARTITIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PARTITIONS);
    assert!(
        num_partitions >= 1000 && num_partitions.is_multiple_of(1000),
        "BENCH_PARTITIONS must be a multiple of 1000"
    );
    let threads: usize = std::env::var("BENCH_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let trino_url = std::env::var("TRINO_URL")
        .unwrap_or_else(|_| "http://localhost:18080".to_string());
    let parquet_dir = std::env::var("BENCH_PARQUET_DIR")
        .unwrap_or_else(|_| "/benchdata/metastore/default/bench_events".to_string());
    let checksum_file = std::env::var("BENCH_CHECKSUM_FILE").ok();

    let run_trino_side = mode == "trino" || mode == "both";
    let run_df_side = mode == "df" || mode == "both";

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .expect("tokio runtime");

    rt.block_on(async move {
        let client = reqwest::Client::new();
        let trino_catalog = if storage == "parquet" { "hive" } else { "memory" };

        if run_trino_side {
            for _ in 0..60 {
                if let Ok(r) = client.get(format!("{trino_url}/v1/info")).send().await
                    && let Ok(txt) = r.text().await
                    && txt.contains("\"starting\":false")
                {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_secs(1)).await;
            }

            if storage == "parquet" {
                let _ = run_trino(
                    &client,
                    &trino_url,
                    "hive",
                    "CREATE SCHEMA IF NOT EXISTS hive.default",
                )
                .await;
            }
            let _ = run_trino(
                &client,
                &trino_url,
                trino_catalog,
                &format!("DROP TABLE IF EXISTS {trino_catalog}.default.bench_events"),
            )
            .await;

            let with_clause = if storage == "parquet" {
                "WITH (format = 'PARQUET')"
            } else {
                ""
            };
            let create_trino_sql = format!(
                r#"
                CREATE TABLE {trino_catalog}.default.bench_events {with_clause} AS
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
                (num_partitions / 1000) - 1
            );
            run_trino(&client, &trino_url, trino_catalog, &create_trino_sql)
                .await
                .expect("create Trino bench_events");
        }

        let ctx = if run_df_side {
            let ctx = session_context_with_match_recognize(
                SessionConfig::new().with_target_partitions(threads),
                MatchRecognizeConfig::default(),
            )
            .expect("df ctx");
            if storage == "parquet" {
                ctx.register_parquet("bench_events", &parquet_dir, ParquetReadOptions::default())
                    .await
                    .expect("register parquet");
            } else {
                let batch = build_df_batch(num_partitions);
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
                let table =
                    datafusion::catalog::memory::MemTable::try_new(batch.schema(), mem_parts)
                        .expect("memtable");
                ctx.register_table("bench_events", Arc::new(table))
                    .expect("register");
            }
            Some(ctx)
        } else {
            None
        };

        let mut saved_checksums: BTreeMap<String, Vec<Vec<String>>> = if mode == "df" {
            checksum_file
                .as_deref()
                .and_then(|p| std::fs::read_to_string(p).ok())
                .and_then(|s| serde_json::from_str(&s).ok())
                .unwrap_or_default()
        } else {
            BTreeMap::new()
        };

        println!(
            "Mode={mode}, Storage={storage}, Rows={}, Threads={threads}, Warmup={WARMUP_ITERS}, Runs={MEASURED_ITERS}\n",
            num_partitions * ROWS_PER_PARTITION
        );

        for (name, sql) in workloads() {
            let mut df_med: Option<f64> = None;
            let mut df_out: Option<Vec<Vec<String>>> = None;
            if let Some(ctx) = &ctx {
                for _ in 0..WARMUP_ITERS {
                    let _ = ctx.sql(sql).await.unwrap().collect().await.unwrap();
                }
                let mut times = Vec::with_capacity(MEASURED_ITERS);
                for _ in 0..MEASURED_ITERS {
                    let t0 = Instant::now();
                    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    df_out = Some(format_df_rows(&batches));
                }
                df_med = Some(median(times));
            }

            let mut trino_wall_med: Option<f64> = None;
            let mut trino_exec_med: Option<f64> = None;
            let mut trino_cpu_med: Option<f64> = None;
            let mut trino_out: Option<Vec<Vec<String>>> = None;
            if run_trino_side {
                for _ in 0..WARMUP_ITERS {
                    let _ = run_trino(&client, &trino_url, trino_catalog, sql)
                        .await
                        .unwrap();
                }
                let mut wall = Vec::with_capacity(MEASURED_ITERS);
                let mut exec = Vec::with_capacity(MEASURED_ITERS);
                let mut cpu = Vec::with_capacity(MEASURED_ITERS);
                for _ in 0..MEASURED_ITERS {
                    let res = run_trino(&client, &trino_url, trino_catalog, sql)
                        .await
                        .unwrap();
                    wall.push(res.wall_ms);
                    exec.push(res.server_elapsed_ms as f64);
                    cpu.push(res.server_cpu_ms as f64);
                    trino_out = Some(res.rows);
                }
                trino_wall_med = Some(median(wall));
                trino_exec_med = Some(median(exec));
                trino_cpu_med = Some(median(cpu));
                saved_checksums.insert(name.to_string(), trino_out.clone().unwrap());
            }

            if let (Some(d), Some(t)) = (&df_out, &trino_out) {
                assert_eq!(d, t, "Result mismatch on {name}: DF={d:?} vs Trino={t:?}");
            } else if let (Some(d), Some(t)) = (&df_out, saved_checksums.get(name)) {
                assert_eq!(
                    d, t,
                    "Result mismatch vs saved Trino checksum on {name}: DF={d:?} vs Trino={t:?}"
                );
            }

            match (df_med, trino_wall_med, trino_exec_med, trino_cpu_med) {
                (Some(d), Some(tw), Some(te), Some(tc)) => {
                    println!(
                        "| `{name}` | {d:.1} ms | {tw:.1} ms | {te:.1} ms | {tc:.1} ms | {:.2}x | exact ({}) |",
                        te / d,
                        df_out.unwrap()[0].join(", ")
                    );
                }
                (None, Some(tw), Some(te), Some(tc)) => {
                    println!(
                        "TRINO | `{name}` | wall={tw:.1} ms | exec={te:.1} ms | cpu={tc:.1} ms | out=({})",
                        trino_out.unwrap()[0].join(", ")
                    );
                }
                (Some(d), None, None, None) => {
                    println!(
                        "DF    | `{name}` | wall={d:.1} ms | out=({}) | parity=verified",
                        df_out.unwrap()[0].join(", ")
                    );
                }
                _ => unreachable!(),
            }
        }

        if mode == "trino"
            && let Some(path) = &checksum_file
        {
            std::fs::write(path, serde_json::to_string_pretty(&saved_checksums).unwrap())
                .expect("write checksums");
        }
    });
}
