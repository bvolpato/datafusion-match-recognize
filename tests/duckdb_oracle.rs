// SPDX-License-Identifier: Apache-2.0

//! Layer 2c verification: live differential SQL oracle against DuckDB 2.1's native C++
//! `MATCH_RECOGNIZE` implementation.
//!
//! Spawns a single persistent Python worker over stdin/stdout NDJSON using either
//! `DUCKDB_PYTHON` (if set to a Python interpreter with `duckdb >= 2.0.1.dev0` installed) or
//! `uv run --prerelease=allow --with duckdb>=2.0.1.dev0 python3`.
//!
//! Run with:
//!   cargo test --test duckdb_oracle -- --ignored --nocapture

use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use arrow::array::{Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::{
    MatchRecognizeConfig, session_context_with_match_recognize,
};
use serde::Deserialize;

const DUCKDB_WORKER_PY: &str = r#"
import datetime, decimal, json, sys, duckdb

con = duckdb.connect()
con.execute("SET threads = 1")

def fmt(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (float, decimal.Decimal)):
        return f"{float(v):.6f}"
    if isinstance(v, list):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, (datetime.date, datetime.datetime)):
        return v.isoformat()
    return str(v)

print(json.dumps({"ready": duckdb.__version__}), flush=True)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    try:
        rows = con.execute(req["sql"]).fetchall()
        out = [[fmt(cell) for cell in row] for row in rows]
        print(json.dumps({"ok": out}), flush=True)
    except Exception as e:
        print(json.dumps({"err": str(e)}), flush=True)
"#;

#[derive(Debug, Deserialize)]
struct ReadyMsg {
    ready: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WorkerReply {
    Ok { ok: Vec<Vec<String>> },
    Err { err: String },
}

struct DuckDbWorker {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    version: String,
}

impl DuckDbWorker {
    fn spawn() -> Self {
        let mut cmd = if let Ok(py) = std::env::var("DUCKDB_PYTHON") {
            let mut c = Command::new(py);
            c.args(["-u", "-c", DUCKDB_WORKER_PY]);
            c
        } else {
            let mut c = Command::new("uv");
            c.args([
                "run",
                "--prerelease=allow",
                "--with",
                "duckdb>=2.0.1.dev0",
                "python3",
                "-u",
                "-c",
                DUCKDB_WORKER_PY,
            ]);
            c
        };

        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect(
                "failed to spawn DuckDB Python worker (requires uv or DUCKDB_PYTHON)",
            );

        let stdin = child.stdin.take().expect("worker stdin");
        let mut stdout = BufReader::new(child.stdout.take().expect("worker stdout"));

        let mut ready_line = String::new();
        stdout
            .read_line(&mut ready_line)
            .expect("failed reading DuckDB ready handshake");
        let ready: ReadyMsg = serde_json::from_str(&ready_line).unwrap_or_else(|e| {
            panic!("invalid DuckDB ready handshake '{ready_line}': {e}")
        });

        Self {
            child,
            stdin,
            stdout,
            version: ready.ready,
        }
    }

    fn run_sql(&mut self, sql: &str) -> Result<Vec<Vec<String>>, String> {
        let payload = serde_json::json!({ "sql": sql }).to_string();
        writeln!(self.stdin, "{payload}").map_err(|e| format!("stdin write: {e}"))?;
        self.stdin
            .flush()
            .map_err(|e| format!("stdin flush: {e}"))?;

        let mut reply_line = String::new();
        self.stdout
            .read_line(&mut reply_line)
            .map_err(|e| format!("stdout read: {e}"))?;
        let reply: WorkerReply = serde_json::from_str(&reply_line)
            .map_err(|e| format!("worker JSON decode error on '{reply_line}': {e}"))?;
        match reply {
            WorkerReply::Ok { ok } => Ok(ok),
            WorkerReply::Err { err } => Err(err),
        }
    }
}

impl Drop for DuckDbWorker {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn format_df_batches(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let options = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for batch in batches {
        let formatters: Vec<ArrayFormatter<'_>> = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &options).unwrap())
            .collect();
        for row_idx in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for (col_idx, fmt) in formatters.iter().enumerate() {
                let col = batch.column(col_idx);
                if col.is_null(row_idx) {
                    row.push("NULL".to_string());
                } else {
                    match col.data_type() {
                        arrow_schema::DataType::Float64 => {
                            let arr = col
                                .as_any()
                                .downcast_ref::<arrow::array::Float64Array>()
                                .unwrap();
                            row.push(format!("{:.6}", arr.value(row_idx)));
                        }
                        arrow_schema::DataType::Float32 => {
                            let arr = col
                                .as_any()
                                .downcast_ref::<arrow::array::Float32Array>()
                                .unwrap();
                            row.push(format!("{:.6}", arr.value(row_idx) as f64));
                        }
                        _ => row.push(fmt.value(row_idx).to_string()),
                    }
                }
            }
            rows.push(row);
        }
    }
    rows
}

async fn run_df_sql(sql: &str) -> Result<Vec<Vec<String>>, String> {
    let ctx = session_context_with_match_recognize(
        SessionConfig::new()
            .with_batch_size(3)
            .with_target_partitions(2),
        MatchRecognizeConfig::default(),
    )
    .map_err(|e| e.to_string())?;
    let batches = ctx
        .sql(sql)
        .await
        .map_err(|e| e.to_string())?
        .collect()
        .await
        .map_err(|e| e.to_string())?;
    Ok(format_df_batches(&batches))
}

async fn assert_matches_duckdb(worker: &mut DuckDbWorker, name: &str, sql: &str) {
    let duck_res = worker.run_sql(sql);
    let df_res = run_df_sql(sql).await;

    match (df_res, duck_res) {
        (Ok(df_rows), Ok(duck_rows)) => {
            assert_eq!(
                df_rows, duck_rows,
                "Differential mismatch vs DuckDB on case '{name}'!\nSQL:\n{sql}\nDataFusion: {df_rows:?}\nDuckDB:     {duck_rows:?}"
            );
        }
        (Err(df_err), Err(duck_err)) => {
            let duck_first = duck_err.lines().next().unwrap_or("");
            eprintln!(
                "[{name}] Both errored as expected (DF: {df_err} | DuckDB: {duck_first})"
            );
        }
        (df_res, duck_res) => {
            panic!(
                "Success/Error mismatch vs DuckDB on case '{name}'!\nSQL:\n{sql}\nDataFusion: {df_res:?}\nDuckDB:     {duck_res:?}"
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires uv + DuckDB 2.1 nightly (or DUCKDB_PYTHON)"]
async fn test_live_duckdb_differential_oracle() {
    let mut worker = DuckDbWorker::spawn();
    eprintln!("Connected to live DuckDB {}", worker.version);

    // 1. Hand-crafted SQL:2016 + DuckDB + Trino + Flink + RisingWave + Snowflake cases.
    //    Unquoted pattern variables are written in UPPERCASE so CLASSIFIER() matches identically
    //    across DataFusion/SQL:2016 (uppercase fold) and DuckDB (written-case preservation).
    let hand_cases: Vec<(&str, &str)> = vec![
        (
            "ordered_alternation_leftmost",
            r#"
            WITH t(id, tag) AS (VALUES (1, 'A'), (2, 'B'), (3, 'A'), (4, 'C'))
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, FIRST(id) AS s_id, LAST(id) AS e_id, CLASSIFIER() AS cls
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A | A B)
                DEFINE A AS tag = 'A', B AS tag = 'B'
            )
            ORDER BY mn
            "#,
        ),
        (
            "multi_partition_aggs_and_nav",
            r#"
            WITH t(p, id, tag, v) AS (
                VALUES
                    ('p1', 1, 'A', 10),
                    ('p1', 2, 'B', 20),
                    ('p1', 3, 'B', CAST(NULL AS INTEGER)),
                    ('p1', 4, 'B', 40),
                    ('p1', 5, 'C', 50),
                    ('p2', 1, 'A', 100),
                    ('p2', 2, 'C', 200)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                MEASURES
                    MATCH_NUMBER() AS mn,
                    FIRST(A.v) AS fa,
                    LAST(B.v) AS lb,
                    PREV(FIRST(B.v), 1) AS prev_fb,
                    NEXT(LAST(B.v), 1) AS next_lb,
                    COUNT(*) AS cnt_all,
                    COUNT(B.v) AS cnt_b_v,
                    SUM(B.v) AS sum_b,
                    AVG(B.v) AS avg_b,
                    MIN(v) AS min_v,
                    MAX(v) AS max_v
                ONE ROW PER MATCH
                PATTERN (A B* C)
                DEFINE A AS tag = 'A', B AS tag = 'B', C AS tag = 'C'
            )
            ORDER BY p, mn
            "#,
        ),
        (
            "all_rows_running_with_unmatched",
            r#"
            WITH t(p, id, tag, v) AS (
                VALUES
                    ('p1', 1, 'X', 5),
                    ('p1', 2, 'A', 10),
                    ('p1', 3, 'B', 20),
                    ('p1', 4, 'B', 30),
                    ('p1', 5, 'Y', 40),
                    ('p2', 1, 'A', 7),
                    ('p2', 2, 'B', 8)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                MEASURES
                    MATCH_NUMBER() AS mn,
                    CLASSIFIER() AS cls,
                    FIRST(A.v) AS fa,
                    COUNT(*) AS r_cnt,
                    SUM(v) AS r_sum,
                    B.v AS cur_b
                ALL ROWS PER MATCH WITH UNMATCHED ROWS
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A B+)
                DEFINE A AS tag = 'A', B AS tag = 'B'
            )
            ORDER BY p, id
            "#,
        ),
        (
            "empty_matches_show",
            r#"
            WITH t(p, id, tag) AS (
                VALUES ('p1', 1, 'X'), ('p1', 2, 'A'), ('p1', 3, 'A'), ('p1', 4, 'X')
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls, COUNT(*) AS cnt
                ALL ROWS PER MATCH SHOW EMPTY MATCHES
                PATTERN (A*)
                DEFINE A AS tag = 'A'
            )
            ORDER BY p, mn, id NULLS FIRST
            "#,
        ),
        (
            "empty_matches_omit",
            r#"
            WITH t(p, id, tag) AS (
                VALUES ('p1', 1, 'X'), ('p1', 2, 'A'), ('p1', 3, 'A'), ('p1', 4, 'X')
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls, COUNT(*) AS cnt
                ALL ROWS PER MATCH OMIT EMPTY MATCHES
                PATTERN (A*)
                DEFINE A AS tag = 'A'
            )
            ORDER BY p, mn, id
            "#,
        ),
        (
            "exclusion_all_rows",
            r#"
            WITH t(id, tag, v) AS (
                VALUES (1, 'A', 10), (2, 'B', 20), (3, 'B', 30), (4, 'C', 40)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls, SUM(v) AS r_sum
                ALL ROWS PER MATCH
                PATTERN (A ({- B -})* C)
                DEFINE A AS tag = 'A', B AS tag = 'B', C AS tag = 'C'
            )
            ORDER BY id
            "#,
        ),
        (
            "permute_and_anchors",
            r#"
            WITH t(p, id, tag) AS (
                VALUES
                    ('p1', 1, 'B'), ('p1', 2, 'A'), ('p1', 3, 'C'),
                    ('p2', 1, 'C'), ('p2', 2, 'B'), ('p2', 3, 'A')
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls
                ALL ROWS PER MATCH
                PATTERN (^ PERMUTE(A, B, C) $)
                DEFINE A AS tag = 'A', B AS tag = 'B', C AS tag = 'C'
            )
            ORDER BY p, id
            "#,
        ),
        (
            "skip_to_next_row_overlapping",
            r#"
            WITH t(id, tag, v) AS (
                VALUES (1, 'A', 10), (2, 'A', 20), (3, 'A', 30), (4, 'A', 40)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, FIRST(id) AS s, LAST(id) AS e, SUM(v) AS sv
                ONE ROW PER MATCH
                AFTER MATCH SKIP TO NEXT ROW
                PATTERN (A{2,3})
                DEFINE A AS tag = 'A'
            )
            ORDER BY mn
            "#,
        ),
        (
            "rum_funnel_3step_ladder",
            r#"
            WITH t(session_id, event_number, ts, event_name, flag0) AS (
                VALUES
                    ('s1', 1, 1000, 'view', 1),
                    ('s1', 2, 1100, 'scroll', 0),
                    ('s1', 3, 1500, 'cart', 0),
                    ('s1', 4, 1800, 'scroll', 0),
                    ('s1', 5, 2200, 'buy', 0),
                    ('s2', 1, 100, 'view', 1),
                    ('s2', 2, 350, 'cart', 0),
                    ('s2', 3, 500, 'exit', 0),
                    ('s3', 1, 50, 'view', 1),
                    ('s3', 2, 80, 'exit', 0),
                    ('s4', 1, 10, 'view', 0),
                    ('s4', 2, 20, 'cart', 0),
                    ('s4', 3, 30, 'buy', 0)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY session_id
                ORDER BY event_number
                MEASURES
                    S0.session_id AS s0_m,
                    S1.session_id AS s1_m,
                    S2.session_id AS s2_m,
                    CASE WHEN S1.ts IS NOT NULL AND (FIRST(S1.ts) - FIRST(S0.ts)) <= 1000
                         THEN FIRST(S1.ts) - FIRST(S0.ts) ELSE NULL END AS dt01,
                    CASE WHEN S2.ts IS NOT NULL AND (FIRST(S2.ts) - FIRST(S0.ts)) <= 2000
                         THEN FIRST(S2.ts) - FIRST(S0.ts) ELSE NULL END AS dt_all
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((S0 ANY0* S1 ANY1* S2) | (S0 ANY0* S1) | (S0))
                DEFINE
                    S0 AS event_name = 'view' AND flag0 = 1,
                    ANY0 AS event_name <> 'cart',
                    S1 AS event_name = 'cart',
                    ANY1 AS event_name <> 'buy',
                    S2 AS event_name = 'buy'
            )
            ORDER BY session_id
            "#,
        ),
        (
            "null_partition_keys_grouped",
            r#"
            WITH t(p, id, tag) AS (
                VALUES
                    (CAST(NULL AS VARCHAR), 1, 'A'),
                    (CAST(NULL AS VARCHAR), 2, 'B'),
                    ('p1', 1, 'A'),
                    ('p1', 2, 'B')
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, FIRST(id) AS s, LAST(id) AS e
                ONE ROW PER MATCH
                PATTERN (A B)
                DEFINE A AS tag = 'A', B AS tag = 'B'
            )
            ORDER BY p NULLS FIRST, mn
            "#,
        ),
        (
            "unmatched_rows_with_skip_to_next_row",
            r#"
            WITH t(id, tag) AS (
                VALUES (1, 'A'), (2, 'A'), (3, 'X'), (4, 'A'), (5, 'A'), (6, 'A')
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls
                ALL ROWS PER MATCH WITH UNMATCHED ROWS
                AFTER MATCH SKIP TO NEXT ROW
                PATTERN (A{2})
                DEFINE A AS tag = 'A'
            )
            ORDER BY id, mn NULLS FIRST
            "#,
        ),
        (
            "stateful_v_shape_one_row",
            r#"
            WITH ticker(sym, t, price) AS (
                VALUES
                    ('ACME', 1, 100), ('ACME', 2, 90), ('ACME', 3, 80),
                    ('ACME', 4,  85), ('ACME', 5, 95), ('ACME', 6, 95),
                    ('ACME', 7,  70), ('ACME', 8, 60), ('ACME', 9, 110),
                    ('BETA', 1,  50), ('BETA', 2, 40), ('BETA', 3, 60), ('BETA', 4, 30), ('BETA', 5, 80)
            )
            SELECT * FROM ticker
            MATCH_RECOGNIZE (
                PARTITION BY sym
                ORDER BY t
                MEASURES
                    MATCH_NUMBER() AS mn,
                    FIRST(STRT.t) AS start_t,
                    LAST(DOWN.t) AS bottom_t,
                    LAST(UP.t) AS end_t,
                    LAST(DOWN.price) AS bottom_price
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (STRT DOWN+ UP+)
                DEFINE
                    DOWN AS DOWN.price < PREV(DOWN.price),
                    UP AS UP.price > PREV(UP.price)
            )
            ORDER BY sym, mn
            "#,
        ),
        (
            "stateful_v_shape_skip_to_last_up_all_rows",
            r#"
            WITH ticker(sym, t, price) AS (
                VALUES
                    ('ACME', 1, 100), ('ACME', 2, 80), ('ACME', 3, 95),
                    ('ACME', 4,  70), ('ACME', 5, 90)
            )
            SELECT * FROM ticker
            MATCH_RECOGNIZE (
                PARTITION BY sym
                ORDER BY t
                MEASURES
                    MATCH_NUMBER() AS mn,
                    CLASSIFIER() AS cls,
                    FIRST(STRT.price) AS sp,
                    LAST(DOWN.price) AS bp
                ALL ROWS PER MATCH
                AFTER MATCH SKIP TO LAST UP
                PATTERN (STRT DOWN+ UP+)
                DEFINE
                    DOWN AS DOWN.price < PREV(DOWN.price),
                    UP AS UP.price > PREV(UP.price)
            )
            ORDER BY sym, mn, t
            "#,
        ),
        (
            "stateful_time_window_funnel",
            r#"
            WITH timed(u, ts, ev) AS (
                VALUES
                    ('u1', 100, 'view'), ('u1', 300, 'scroll'), ('u1', 450, 'buy'),
                    ('u2', 100, 'view'), ('u2', 700, 'buy'),
                    ('u3', 100, 'view'), ('u3', 200, 'view'), ('u3', 650, 'buy')
            )
            SELECT * FROM timed
            MATCH_RECOGNIZE (
                PARTITION BY u
                ORDER BY ts
                MEASURES
                    FIRST(A.ts) AS a_ts,
                    LAST(B.ts) AS b_ts,
                    LAST(B.ts) - FIRST(A.ts) AS dt
                ONE ROW PER MATCH
                PATTERN (A ANY_R* B)
                DEFINE
                    A AS ev = 'view',
                    B AS ev = 'buy' AND B.ts - FIRST(A.ts) <= 500
            )
            ORDER BY u
            "#,
        ),
        (
            "duckdb_spec_nest_prev_last",
            r#"
            WITH t(id, price, tax) AS (
                VALUES (1, 10, 1), (2, 12, 2), (3, 14, 3), (4, 16, 4), (5, 18, 5), (6, 20, 6)
            )
            SELECT id, mn, p FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, PREV(LAST(A.price + A.tax, 1), 3) AS p
                ALL ROWS PER MATCH
                PATTERN (A+)
                DEFINE A AS id >= 4
            )
            ORDER BY id
            "#,
        ),
        (
            "duckdb_spec_match_number_parity",
            r#"
            WITH t(rn, price) AS (VALUES (1, 10), (2, 20), (3, 15), (4, 10), (5, 12), (6, 25))
            SELECT rn, mn, price, cls FROM t
            MATCH_RECOGNIZE (
                ORDER BY rn
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls
                ALL ROWS PER MATCH
                PATTERN (A+ | B+)
                DEFINE
                    A AS MATCH_NUMBER() % 2 = 1 AND (PREV(A.price) IS NULL OR A.price >= PREV(A.price)),
                    B AS MATCH_NUMBER() % 2 = 0 AND (PREV(B.price) IS NULL OR B.price <= PREV(B.price))
            )
            ORDER BY rn
            "#,
        ),
        (
            "duckdb_head_and_shoulders_recurring_vars",
            r#"
            WITH t(i, v) AS (
                VALUES
                    (1, 10), (2, 15), (3, 20), (4, 14), (5, 12),
                    (6, 25), (7, 30), (8, 18), (9, 22), (10, 10), (11, 8)
            )
            SELECT i, mn, v, cls FROM t
            MATCH_RECOGNIZE (
                ORDER BY i
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (S UP+ DOWN+ UP2+ DOWN+ UP+ DOWN2+)
                DEFINE
                    UP AS v > PREV(v),
                    DOWN AS v < PREV(v),
                    UP2 AS v > max(UP.v),
                    DOWN2 AS v < min(DOWN.v)
            )
            ORDER BY i
            "#,
        ),
        (
            "duckdb_bin_packing_exclusion",
            r#"
            WITH t(bin, i, w) AS (
                VALUES (1, 1, 6), (1, 2, 7), (1, 3, 4), (1, 4, 5), (1, 5, 1),
                       (2, 1, 8), (2, 2, 3), (2, 3, 2)
            )
            SELECT bin, total_w, packed_cnt FROM t
            MATCH_RECOGNIZE (
                PARTITION BY bin
                ORDER BY i
                MEASURES sum(X.w) AS total_w, count(X.i) AS packed_cnt
                ONE ROW PER MATCH
                PATTERN ((X | ({- REST -}))+)
                DEFINE X AS sum(X.w) <= 10
            )
            ORDER BY bin
            "#,
        ),
        (
            "duckdb_merge_overlapping_intervals",
            r#"
            WITH t(s0, e0) AS (
                VALUES (1, 4), (2, 6), (5, 7), (10, 12), (11, 15), (20, 22)
            )
            SELECT merged_s, merged_e, n FROM t
            MATCH_RECOGNIZE (
                ORDER BY s0, e0
                MEASURES FIRST(s0) AS merged_s, max(e0) AS merged_e, count(*) AS n
                ONE ROW PER MATCH
                PATTERN (J* L)
                DEFINE J AS NEXT(s0) <= max(e0)
            )
            ORDER BY merged_s
            "#,
        ),
        (
            "duckdb_word_wrap_string_agg",
            r#"
            WITH t(rn, w) AS (
                VALUES (1, 'cat'), (2, 'dog'), (3, 'bird'), (4, 'elephant'), (5, 'ant'), (6, 'fox')
            )
            SELECT mn, cnt, line FROM t
            MATCH_RECOGNIZE (
                ORDER BY rn
                MEASURES MATCH_NUMBER() AS mn, count(*) AS cnt, string_agg(w, ', ') AS line
                ONE ROW PER MATCH
                PATTERN (I L*)
                DEFINE L AS length(I.w) + sum(length(L.w) + 2) <= 12
            )
            ORDER BY mn
            "#,
        ),
        (
            "duckdb_supply_depletion",
            r#"
            WITH t(rn, days) AS (
                VALUES (1, 2), (2, 1), (3, 0), (4, 0), (5, 1), (6, 0), (7, 0)
            )
            SELECT mn, start_rn, end_rn FROM t
            MATCH_RECOGNIZE (
                ORDER BY rn
                MEASURES MATCH_NUMBER() AS mn, FIRST(rn) AS start_rn, LAST(rn) AS end_rn
                ONE ROW PER MATCH
                PATTERN (A B*)
                DEFINE B AS rn <= A.rn + sum(days) - days
            )
            ORDER BY mn
            "#,
        ),
        (
            "duckdb_cross_var_next_and_forward_ref",
            r#"
            WITH t(id, tier) AS (VALUES (1, 'low'), (2, 'mid'), (3, 'high'))
            SELECT id, cls, tier FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES CLASSIFIER() AS cls
                ALL ROWS PER MATCH
                PATTERN (P Q R)
                DEFINE P AS tier = 'low' AND LAST(R.tier) IS NULL, Q AS NEXT(P.tier) = 'mid', R AS true
            )
            ORDER BY id
            "#,
        ),
        (
            "duckdb_classifier_alternation_backtracking",
            r#"
            WITH v(rn, x) AS (VALUES (1, 1), (2, 2))
            SELECT rn, c FROM v
            MATCH_RECOGNIZE (
                ORDER BY rn
                MEASURES CLASSIFIER() AS c
                ALL ROWS PER MATCH
                PATTERN ((A | B) C)
                DEFINE A AS true, B AS true, C AS PREV(CLASSIFIER()) = 'B'
            )
            ORDER BY rn
            "#,
        ),
        (
            "duckdb_expr_partition_and_order_by",
            r#"
            WITH t(g, x) AS (VALUES ('a', 1), ('A', 2), ('b', 3))
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY upper(g)
                ORDER BY -x
                MEASURES count(*) AS n
                ONE ROW PER MATCH
                PATTERN (A+)
                DEFINE A AS true
            )
            ORDER BY n
            "#,
        ),
        (
            "duckdb_stacked_match_recognize",
            r#"
            WITH src(p, i, v) AS (
                VALUES (1, 1, 10), (1, 2, 20), (1, 3, 5), (1, 4, 15),
                       (2, 1, 30), (2, 2, 40), (2, 3, 25)
            ),
            runs AS (
                SELECT p, mn, last_v FROM src
                MATCH_RECOGNIZE (
                    PARTITION BY p
                    ORDER BY i
                    MEASURES MATCH_NUMBER() AS mn, LAST(v) AS last_v
                    ONE ROW PER MATCH
                    PATTERN (A B*)
                    DEFINE B AS v > PREV(v)
                )
            )
            SELECT p, pairs FROM runs
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY mn
                MEASURES count(*) AS pairs
                ONE ROW PER MATCH
                PATTERN (X Y)
                DEFINE X AS true, Y AS Y.last_v < X.last_v
            )
            ORDER BY p
            "#,
        ),
        (
            "trino_balancing_sums",
            r#"
            WITH t(id, value) AS (
                VALUES (1, 4), (2, 6), (3, 10), (4, 1), (5, 1), (6, 1), (7, 10), (8, 5), (9, 1)
            )
            SELECT m.id, m.classy, m.running_sum_a, m.running_sum_b FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    sum(A.value) AS running_sum_a,
                    sum(B.value) AS running_sum_b,
                    CLASSIFIER() AS classy
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((A | B)*)
                DEFINE A AS sum(A.value) - A.value <= sum(B.value)
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_period_length",
            r#"
            WITH t(user_id, minute_of_the_day) AS (
                VALUES (1, 3), (1, 4), (1, 5), (1, 8), (1, 9), (2, 2), (2, 3), (2, 4)
            )
            SELECT user_id, CAST(periods_total AS INTEGER) AS pt FROM t
            MATCH_RECOGNIZE (
                PARTITION BY user_id
                ORDER BY minute_of_the_day
                MEASURES COALESCE(sum(C.minute_of_the_day) - sum(A.minute_of_the_day), 0) AS periods_total
                ONE ROW PER MATCH
                PATTERN ((A B* C | D)*)
                DEFINE
                    B AS minute_of_the_day = PREV(minute_of_the_day) + 1,
                    C AS minute_of_the_day = PREV(minute_of_the_day) + 1
            )
            ORDER BY user_id
            "#,
        ),
        (
            "trino_set_partitioning_array_agg_classifier",
            r#"
            WITH t(id) AS (VALUES (1), (2), (3), (4), (5), (6), (7), (8))
            SELECT m.id, m.running_labels FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES array_agg(CLASSIFIER()) AS running_labels
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (^ (A | B)* (LAST_A | LAST_B) $)
                DEFINE
                    LAST_A AS sum(A.id) + id = sum(B.id),
                    LAST_B AS sum(B.id) + id = sum(A.id)
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_all_excluded_all_rows_show_empty",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT m.id, m.mn, m.v FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn
                ALL ROWS PER MATCH SHOW EMPTY MATCHES
                PATTERN (({- A -}) ({- B -}))
                DEFINE A AS true, B AS true
            ) AS m
            "#,
        ),
        (
            "trino_all_excluded_one_row",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT m.mn, m.cls, m.s FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls, SUM(v) AS s
                ONE ROW PER MATCH
                PATTERN (({- A -}) ({- B -}))
                DEFINE A AS true, B AS true
            ) AS m
            "#,
        ),
        (
            "trino_mixed_excluded_match_number",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20), (3, 30))
            SELECT m.id, m.mn, m.cls FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls
                ALL ROWS PER MATCH SHOW EMPTY MATCHES
                PATTERN (({- A -}) | B)
                DEFINE A AS v = 10, B AS v > 10
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_tentative_label_avg_b_and_a",
            r#"
            WITH t(id, value) AS (VALUES (1, 4), (2, 6), (3, 0), (4, 5))
            SELECT m.id, m.classy, m.running_avg_a FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES avg(A.value) AS running_avg_a, CLASSIFIER() AS classy
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((A | B)*)
                DEFINE A AS avg(A.value) = 5
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_empty_iter_in_unbounded_loop",
            r#"
            WITH t(id, tag) AS (VALUES (1, 'C'), (2, 'A'))
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, FIRST(id) AS s, LAST(id) AS e, CLASSIFIER() AS cls
                ONE ROW PER MATCH
                PATTERN ((C | C? | A)*)
                DEFINE C AS tag = 'C', A AS tag = 'A'
            )
            ORDER BY mn
            "#,
        ),
        (
            "flink_logical_offsets_backtracking",
            r#"
            WITH ticker(symbol, tstamp, price, tax) AS (
                VALUES
                    ('ACME', 1, 19, 1), ('ACME', 2, 17, 2), ('ACME', 3, 13, 3), ('ACME', 4, 20, 4),
                    ('ACME', 5, 20, 5), ('ACME', 6, 26, 6), ('ACME', 7, 20, 7), ('ACME', 8, 25, 8)
            )
            SELECT * FROM ticker
            MATCH_RECOGNIZE (
                ORDER BY tstamp
                MEASURES
                    FIRST(DOWN.tstamp) AS start_tstamp,
                    LAST(DOWN.tstamp) AS bottom_tstamp,
                    UP.tstamp AS end_tstamp,
                    FIRST(DOWN.price + DOWN.tax + 1) AS bottom_total,
                    UP.price + UP.tax AS end_total
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (DOWN{2,} UP)
                DEFINE
                    DOWN AS price < LAST(DOWN.price, 1) OR LAST(DOWN.price, 1) IS NULL,
                    UP AS price < FIRST(DOWN.price)
            ) AS T
            ORDER BY start_tstamp
            "#,
        ),
        (
            "snowflake_stock_vshape_dates",
            r#"
            WITH sf_stocks(company, price_date, price) AS (
                VALUES
                    ('ABCD', DATE '2020-10-01', 50), ('XYZ', DATE '2020-10-01', 89),
                    ('ABCD', DATE '2020-10-02', 36), ('XYZ', DATE '2020-10-02', 24),
                    ('ABCD', DATE '2020-10-03', 39), ('XYZ', DATE '2020-10-03', 37),
                    ('ABCD', DATE '2020-10-04', 42), ('XYZ', DATE '2020-10-04', 63),
                    ('ABCD', DATE '2020-10-05', 30), ('XYZ', DATE '2020-10-05', 65),
                    ('ABCD', DATE '2020-10-06', 47), ('XYZ', DATE '2020-10-06', 56),
                    ('ABCD', DATE '2020-10-07', 71), ('XYZ', DATE '2020-10-07', 50),
                    ('ABCD', DATE '2020-10-08', 80), ('XYZ', DATE '2020-10-08', 54),
                    ('ABCD', DATE '2020-10-09', 75), ('XYZ', DATE '2020-10-09', 30),
                    ('ABCD', DATE '2020-10-10', 63), ('XYZ', DATE '2020-10-10', 32)
            )
            SELECT * FROM sf_stocks
            MATCH_RECOGNIZE (
                PARTITION BY company
                ORDER BY price_date
                MEASURES
                    MATCH_NUMBER() AS match_number,
                    FIRST(price_date) AS start_date,
                    LAST(price_date) AS end_date,
                    COUNT(*) AS rows_in_sequence,
                    COUNT(DEC_ROW.price) AS num_decreases,
                    COUNT(INC_ROW.price) AS num_increases
                ONE ROW PER MATCH
                AFTER MATCH SKIP TO LAST INC_ROW
                PATTERN (BASE_ROW DEC_ROW+ INC_ROW+)
                DEFINE
                    DEC_ROW AS price < PREV(price),
                    INC_ROW AS price > PREV(price)
            )
            WHERE company = 'ABCD'
            ORDER BY company, match_number
            "#,
        ),
        // Negative error-parity checks against DuckDB 2.1
        (
            "err_parity_skip_infinite_loop",
            r#"
            WITH t(id, tag) AS (VALUES (1, 'A'), (2, 'B'))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES MATCH_NUMBER() AS mn
                AFTER MATCH SKIP TO FIRST A PATTERN (A B) DEFINE A AS tag = 'A', B AS tag = 'B'
            )
            "#,
        ),
        (
            "err_parity_unused_define",
            r#"
            WITH t(id, tag) AS (VALUES (1, 'A'))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES MATCH_NUMBER() AS mn
                PATTERN (A) DEFINE A AS tag = 'A', UNUSED_SYM AS tag = 'B'
            )
            "#,
        ),
        (
            "err_parity_dup_measure_alias",
            r#"
            WITH t(id) AS (VALUES (1), (2))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES count(*) AS n, sum(id) AS n PATTERN (A+) DEFINE A AS true
            )
            "#,
        ),
        (
            "err_parity_order_by_match_number",
            r#"
            WITH t(id) AS (VALUES (1), (2))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY MATCH_NUMBER() MEASURES count(*) AS n PATTERN (A+) DEFINE A AS true
            )
            "#,
        ),
        (
            "err_parity_order_by_first",
            r#"
            WITH t(id) AS (VALUES (1), (2))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY FIRST(id) MEASURES count(*) AS n PATTERN (A+) DEFINE A AS true
            )
            "#,
        ),
        (
            "err_parity_zero_quantifier",
            r#"
            WITH t(id) AS (VALUES (1))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES MATCH_NUMBER() AS mn PATTERN (A{0}) DEFINE A AS true
            )
            "#,
        ),
        (
            "err_parity_no_output_columns",
            r#"
            WITH t(id) AS (VALUES (1))
            SELECT * FROM t MATCH_RECOGNIZE (ORDER BY id ONE ROW PER MATCH PATTERN (A) DEFINE A AS true)
            "#,
        ),
        (
            "err_parity_mixed_labeled_unlabeled_nav",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES PREV(LAST(A.v + v)) AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_nested_nav_last_first_define",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES id AS x PATTERN (A B) DEFINE A AS true, B AS v > LAST(FIRST(v))
            )
            "#,
        ),
        (
            "err_parity_nested_agg",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES max(1 + min(v)) AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_logical_nav_in_agg",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES max(1 + LAST(v)) AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_null_nav_offset",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES FIRST(v, NULL) AS x PATTERN (A+) DEFINE A AS true
            )
            "#,
        ),
        (
            "err_parity_skip_unmatched_optional",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES MATCH_NUMBER() AS mn AFTER MATCH SKIP TO LAST D PATTERN (A B+ D?) DEFINE B AS true, D AS false
            )
            "#,
        ),
    ];

    for (name, sql) in &hand_cases {
        assert_matches_duckdb(&mut worker, name, sql).await;
    }

    // 2. Stateless fuzz matrix (48 queries)
    let patterns: &[(&str, &str)] = &[
        (
            "A B* C",
            "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7, C AS v >= 6",
        ),
        (
            "(A | A B)",
            "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7",
        ),
        (
            "A{1,3} B?",
            "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7",
        ),
        (
            "PERMUTE(A, B) C?",
            "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7, C AS v >= 6",
        ),
        ("^ A+ B*", "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7"),
        ("A* B+ $", "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7"),
        (
            "(A B | B A)+",
            "A AS v >= 2 AND v <= 4, B AS v >= 4 AND v <= 7",
        ),
        ("A ANY_R* C", "A AS v >= 2 AND v <= 4, C AS v >= 6"),
    ];
    let skips = [
        "AFTER MATCH SKIP PAST LAST ROW",
        "AFTER MATCH SKIP TO NEXT ROW",
    ];
    let modes = [
        ("ONE ROW PER MATCH", "ORDER BY p, mn"),
        (
            "ALL ROWS PER MATCH OMIT EMPTY MATCHES",
            "ORDER BY p, mn, id",
        ),
        (
            "ALL ROWS PER MATCH WITH UNMATCHED ROWS",
            "ORDER BY p, id, mn NULLS FIRST",
        ),
    ];
    let base_cte = r#"
        WITH t(p, id, v) AS (
            VALUES
                (1, 1, 2), (1, 2, 4), (1, 3, 5), (1, 4, 6), (1, 5, 1), (1, 6, 8),
                (2, 1, 1), (2, 2, 2), (2, 3, 3), (2, 4, 6), (2, 5, 7),
                (3, 1, CAST(NULL AS INTEGER)), (3, 2, 4), (3, 3, 6)
        )
    "#;

    let mut stateless_checked = 0usize;
    for (pi, (pat, defines)) in patterns.iter().enumerate() {
        for (si, skip) in skips.iter().enumerate() {
            for (mi, (mode, order_clause)) in modes.iter().enumerate() {
                let sql = format!(
                    r#"
                    {base_cte}
                    SELECT * FROM t
                    MATCH_RECOGNIZE (
                        PARTITION BY p
                        ORDER BY id
                        MEASURES
                            MATCH_NUMBER() AS mn,
                            CLASSIFIER() AS cls,
                            FIRST(v) AS fv,
                            LAST(v) AS lv,
                            COUNT(*) AS cnt,
                            SUM(v) AS sv
                        {mode}
                        {skip}
                        PATTERN ({pat})
                        DEFINE {defines}
                    )
                    {order_clause}
                    "#
                );
                assert_matches_duckdb(
                    &mut worker,
                    &format!("stateless_fuzz_p{pi}_s{si}_m{mi}"),
                    &sql,
                )
                .await;
                stateless_checked += 1;
            }
        }
    }

    // 3. Stateful fuzz matrix (16 queries)
    let stateful_patterns: &[(&str, &str)] = &[
        ("S D+ U+", "D AS D.v < PREV(D.v), U AS U.v > PREV(U.v)"),
        (
            "A B+",
            "A AS v <= 5, B AS B.v > FIRST(A.v) AND COUNT(B.id) <= 2",
        ),
        (
            "A B* C",
            "A AS v >= 2, B AS B.v >= PREV(B.v), C AS C.v < PREV(C.v)",
        ),
        ("A+ B+", "A AS A.v <= 5, B AS B.v > AVG(A.v)"),
    ];
    let mut stateful_checked = 0usize;
    for (pi, (pat, defines)) in stateful_patterns.iter().enumerate() {
        for (si, skip) in skips.iter().enumerate() {
            for (mi, (mode, order_clause)) in modes[..2].iter().enumerate() {
                let sql = format!(
                    r#"
                    {base_cte}
                    SELECT * FROM t
                    MATCH_RECOGNIZE (
                        PARTITION BY p
                        ORDER BY id
                        MEASURES
                            MATCH_NUMBER() AS mn,
                            CLASSIFIER() AS cls,
                            FIRST(v) AS fv,
                            LAST(v) AS lv,
                            COUNT(*) AS cnt,
                            SUM(v) AS sv
                        {mode}
                        {skip}
                        PATTERN ({pat})
                        DEFINE {defines}
                    )
                    {order_clause}
                    "#
                );
                assert_matches_duckdb(
                    &mut worker,
                    &format!("stateful_fuzz_p{pi}_s{si}_m{mi}"),
                    &sql,
                )
                .await;
                stateful_checked += 1;
            }
        }
    }

    // 4. 200-query seeded PRNG randomized SQL + randomized table fuzzer vs live DuckDB 2.1
    let mut rng_state: u64 = 0xD0C4_DB21_9876_5432;
    let mut next_u64 = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // (pattern, defines, has_exclusion, can_skip_to_last_b)
    let rand_patterns: &[(&str, &str, bool, bool)] = &[
        (
            "A B* C?",
            "A AS v >= 1 AND v <= 4, B AS v >= 3 AND v <= 6, C AS v >= 5",
            false,
            false,
        ),
        ("(A | A B)", "A AS v <= 4, B AS v >= 3", false, false),
        (
            "S D+ U+",
            "D AS D.v < PREV(D.v), U AS U.v > PREV(U.v)",
            false,
            false,
        ),
        ("A+ B+", "A AS A.v <= 5, B AS B.v > AVG(A.v)", false, true),
        (
            "A B+",
            "A AS v >= 2, B AS B.v - FIRST(A.v) <= 3 AND COUNT(B.id) <= 3",
            false,
            true,
        ),
        ("PERMUTE(A, B)", "A AS v <= 4, B AS v >= 4", false, false),
        ("A*", "A AS v >= 2 AND v <= 6", false, false),
        ("({- A -}) B+", "A AS v <= 4, B AS v >= 3", true, true),
        ("({- A -}) | B", "A AS v <= 3, B AS v >= 5", true, false),
        (
            "A{0,2} B+",
            "A AS v <= 5 AND MATCH_NUMBER() <= 3, B AS v >= 4",
            false,
            false,
        ),
        (
            "^ A+ B*",
            "A AS v >= 2, B AS B.v >= PREV(B.v)",
            false,
            false,
        ),
        (
            "A+ B $",
            "A AS v <= 6, B AS PREV(CLASSIFIER()) = 'A' AND B.v >= 2",
            false,
            true,
        ),
        ("A+", "A AS FIRST(A.v) = A.v", false, false),
        ("(C | A)*", "C AS v >= 5, A AS v <= 3", false, false),
        ("A B B", "A AS v >= 2, B AS v >= 2", false, true),
    ];
    let rand_orders = [
        "ORDER BY id ASC NULLS LAST",
        "ORDER BY id DESC NULLS FIRST",
        "ORDER BY id ASC NULLS FIRST",
        "ORDER BY id DESC NULLS LAST",
    ];
    let rand_skips = [
        "AFTER MATCH SKIP PAST LAST ROW",
        "AFTER MATCH SKIP TO NEXT ROW",
        "AFTER MATCH SKIP TO LAST B",
    ];

    let mut random_checked = 0usize;
    for q in 0..200usize {
        let num_parts = (next_u64() % 3 + 1) as usize;
        let mut value_rows = Vec::new();
        for p in 1..=num_parts {
            let nrows = (next_u64() % 7 + 1) as usize;
            for r in 1..=nrows {
                let is_null = (next_u64() % 6) == 0;
                let v_str = if is_null {
                    "CAST(NULL AS INTEGER)".to_string()
                } else {
                    format!("{}", (next_u64() % 9) as i64)
                };
                value_rows.push(format!("({p}, {r}, {v_str})"));
            }
        }
        let values_sql = value_rows.join(", ");
        let (pat, defs, has_excl, can_skip_b) =
            rand_patterns[(next_u64() as usize) % rand_patterns.len()];
        let ord = rand_orders[(next_u64() as usize) % rand_orders.len()];
        let skip_idx = (next_u64() as usize) % if can_skip_b { 3 } else { 2 };
        let skip = rand_skips[skip_idx];
        let mode_choice = (next_u64() % 4) as usize;
        let (mode, outer_ord) = match mode_choice {
            0 => ("ONE ROW PER MATCH", "ORDER BY m.p, m.mn"),
            1 => (
                "ALL ROWS PER MATCH OMIT EMPTY MATCHES",
                "ORDER BY m.p, m.mn, m.id",
            ),
            2 => (
                "ALL ROWS PER MATCH SHOW EMPTY MATCHES",
                "ORDER BY m.p, m.mn, m.id NULLS FIRST",
            ),
            _ if !has_excl => (
                "ALL ROWS PER MATCH WITH UNMATCHED ROWS",
                "ORDER BY m.p, m.id, m.mn NULLS FIRST",
            ),
            _ => (
                "ALL ROWS PER MATCH SHOW EMPTY MATCHES",
                "ORDER BY m.p, m.mn, m.id NULLS FIRST",
            ),
        };
        let where_clause = if (next_u64() % 4) == 0 {
            "WHERE m.p >= 1"
        } else {
            ""
        };

        let sql = format!(
            r#"
            WITH t(p, id, v) AS (VALUES {values_sql})
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                {ord}
                MEASURES
                    MATCH_NUMBER() AS mn,
                    FIRST(CLASSIFIER()) AS f_cls,
                    LAST(CLASSIFIER()) AS l_cls,
                    NEXT(CLASSIFIER()) AS n_cls,
                    ARRAY_AGG(CLASSIFIER()) AS cls_arr,
                    FIRST(v) AS fv,
                    LAST(v) AS lv,
                    COUNT(*) AS cnt,
                    COUNT() AS cnt0,
                    SUM(v) AS sv,
                    MIN(v) AS min_v,
                    MAX(v) AS max_v,
                    BOOL_OR(v > 4) AS any_gt4
                {mode}
                {skip}
                PATTERN ({pat})
                DEFINE {defs}
            ) AS m
            {where_clause}
            {outer_ord}
            "#
        );
        assert_matches_duckdb(&mut worker, &format!("prng_q{q}"), &sql).await;
        random_checked += 1;
    }

    eprintln!(
        "Live DuckDB {} differential oracle passed all {} hand cases + {} stateless fuzz + {} stateful fuzz + {} PRNG random SQL queries!",
        worker.version,
        hand_cases.len(),
        stateless_checked,
        stateful_checked,
        random_checked
    );
}
