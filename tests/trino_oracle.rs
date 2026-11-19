// SPDX-License-Identifier: Apache-2.0

//! Layer 2b verification: live differential SQL oracle against a real Trino (`trinodb/trino`)
//! Docker container on `TRINO_URL` (default `http://localhost:18080`).
//!
//! Run with:
//!   cargo test --test trino_oracle -- --ignored --nocapture

use arrow::array::{Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::{MatchRecognizeConfig, session_context_with_match_recognize};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TrinoResponse {
    #[serde(rename = "nextUri")]
    next_uri: Option<String>,
    data: Option<Vec<Vec<serde_json::Value>>>,
    error: Option<serde_json::Value>,
}

async fn run_trino_sql(
    client: &reqwest::Client,
    base_url: &str,
    sql: &str,
) -> Result<Vec<Vec<String>>, String> {
    let mut resp: TrinoResponse = client
        .post(format!("{base_url}/v1/statement"))
        .header("X-Trino-User", "oracle")
        .header("X-Trino-Catalog", "memory")
        .header("X-Trino-Schema", "default")
        .body(sql.to_string())
        .send()
        .await
        .map_err(|e| format!("HTTP POST error: {e}"))?
        .json()
        .await
        .map_err(|e| format!("JSON decode error: {e}"))?;

    let mut rows: Vec<Vec<String>> = Vec::new();
    loop {
        if let Some(err) = resp.error {
            return Err(format!("Trino SQL error: {err}"));
        }
        if let Some(data) = resp.data {
            for r in data {
                rows.push(r.into_iter().map(format_json_cell).collect());
            }
        }
        let Some(next) = resp.next_uri else {
            break;
        };
        resp = client
            .get(&next)
            .header("X-Trino-User", "oracle")
            .send()
            .await
            .map_err(|e| format!("HTTP GET error: {e}"))?
            .json()
            .await
            .map_err(|e| format!("JSON decode error: {e}"))?;
    }
    Ok(rows)
}

fn format_json_cell(v: serde_json::Value) -> String {
    match v {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                format!("{f:.6}")
            } else {
                n.to_string()
            }
        }
        serde_json::Value::String(s) => s,
        other => other.to_string(),
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

async fn assert_matches_trino(client: &reqwest::Client, base_url: &str, name: &str, sql: &str) {
    let trino_res = run_trino_sql(client, base_url, sql).await;
    let df_res = run_df_sql(sql).await;

    match (df_res, trino_res) {
        (Ok(df_rows), Ok(trino_rows)) => {
            assert_eq!(
                df_rows, trino_rows,
                "Differential mismatch vs Trino on case '{name}'!\nSQL:\n{sql}\nDataFusion: {df_rows:?}\nTrino:      {trino_rows:?}"
            );
        }
        (Err(df_err), Err(trino_err)) => {
            eprintln!("[{name}] Both errored as expected (DF: {df_err} | Trino: {trino_err})");
        }
        (df_res, trino_res) => {
            panic!(
                "Success/Error mismatch vs Trino on case '{name}'!\nSQL:\n{sql}\nDataFusion: {df_res:?}\nTrino:      {trino_res:?}"
            );
        }
    }
}

#[tokio::test]
#[ignore = "requires live Trino container at TRINO_URL (default http://localhost:18080)"]
async fn test_live_trino_differential_oracle() {
    let base_url =
        std::env::var("TRINO_URL").unwrap_or_else(|_| "http://localhost:18080".to_string());
    let client = reqwest::Client::new();

    // Verify Trino is reachable.
    let info = client
        .get(format!("{base_url}/v1/info"))
        .send()
        .await
        .expect("Trino must be reachable at TRINO_URL");
    assert!(
        info.status().is_success(),
        "Trino /v1/info returned {}",
        info.status()
    );

    // 1. Hand-crafted SQL:2016 corner cases covering Trino semantics.
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
                PATTERN (a | a b)
                DEFINE a AS tag = 'A', b AS tag = 'B'
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
                    FIRST(a.v) AS fa,
                    LAST(b.v) AS lb,
                    PREV(FIRST(b.v), 1) AS prev_fb,
                    NEXT(LAST(b.v), 1) AS next_lb,
                    COUNT(*) AS cnt_all,
                    COUNT(b.*) AS cnt_b_star,
                    COUNT(b.v) AS cnt_b_v,
                    SUM(b.v) AS sum_b,
                    AVG(b.v) AS avg_b,
                    MIN(v) AS min_v,
                    MAX(v) AS max_v
                ONE ROW PER MATCH
                PATTERN (a b* c)
                DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
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
                    FIRST(a.v) AS fa,
                    COUNT(*) AS r_cnt,
                    SUM(v) AS r_sum,
                    b.v AS cur_b
                ALL ROWS PER MATCH WITH UNMATCHED ROWS
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (a b+)
                DEFINE a AS tag = 'A', b AS tag = 'B'
            )
            ORDER BY p, id
            "#,
        ),
        (
            "empty_matches_show_vs_omit",
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
                PATTERN (a*)
                DEFINE a AS tag = 'A'
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
                PATTERN (a*)
                DEFINE a AS tag = 'A'
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
                PATTERN (a ({- b -})* c)
                DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
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
                PATTERN (^ PERMUTE(a, b, c) $)
                DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
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
                PATTERN (a{2,3})
                DEFINE a AS tag = 'A'
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
                    s0.session_id AS s0_m,
                    s1.session_id AS s1_m,
                    s2.session_id AS s2_m,
                    CASE WHEN s1.ts IS NOT NULL AND (FIRST(s1.ts) - FIRST(s0.ts)) <= 1000
                         THEN FIRST(s1.ts) - FIRST(s0.ts) ELSE NULL END AS dt01,
                    CASE WHEN s2.ts IS NOT NULL AND (FIRST(s2.ts) - FIRST(s0.ts)) <= 2000
                         THEN FIRST(s2.ts) - FIRST(s0.ts) ELSE NULL END AS dt_all
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((s0 any0* s1 any1* s2) | (s0 any0* s1) | (s0))
                DEFINE
                    s0 AS event_name = 'view' AND flag0 = 1,
                    any0 AS event_name <> 'cart',
                    s1 AS event_name = 'cart',
                    any1 AS event_name <> 'buy',
                    s2 AS event_name = 'buy'
            )
            ORDER BY session_id
            "#,
        ),
        (
            "skip_infinite_loop_error_parity",
            r#"
            WITH t(id, tag) AS (VALUES (1, 'A'), (2, 'B'))
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn
                ONE ROW PER MATCH
                AFTER MATCH SKIP TO FIRST a
                PATTERN (a b)
                DEFINE a AS tag = 'A', b AS tag = 'B'
            )
            "#,
        ),
        (
            "unused_define_error_parity",
            r#"
            WITH t(id, tag) AS (VALUES (1, 'A'))
            SELECT * FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn
                ONE ROW PER MATCH
                PATTERN (a)
                DEFINE a AS tag = 'A', unused_sym AS tag = 'B'
            )
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
                PATTERN (a b)
                DEFINE a AS tag = 'A', b AS tag = 'B'
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
                PATTERN (a{2})
                DEFINE a AS tag = 'A'
            )
            ORDER BY id, mn NULLS FIRST
            "#,
        ),
    ];

    for (name, sql) in hand_cases {
        assert_matches_trino(&client, &base_url, name, sql).await;
    }

    // 2. Deterministic pseudo-random differential fuzzing: 48 generated SQL queries across
    //    varied patterns, skip modes (including WITH UNMATCHED ROWS + SKIP TO NEXT ROW!),
    //    output modes, and overlapping DEFINE predicates.
    let patterns: &[(&str, &str)] = &[
        (
            "a b* c",
            "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7, c AS v >= 6",
        ),
        (
            "(a | a b)",
            "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7",
        ),
        (
            "a{1,3} b?",
            "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7",
        ),
        (
            "PERMUTE(a, b) c?",
            "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7, c AS v >= 6",
        ),
        ("^ a+ b*", "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7"),
        ("a* b+ $", "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7"),
        (
            "(a b | b a)+",
            "a AS v >= 2 AND v <= 4, b AS v >= 4 AND v <= 7",
        ),
        ("a any_r* c", "a AS v >= 2 AND v <= 4, c AS v >= 6"),
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

    // Table with overlapping DEFINE predicates (e.g., v % 2 = 0 overlaps with v > 3).
    let base_cte = r#"
        WITH t(p, id, v) AS (
            VALUES
                (1, 1, 2), (1, 2, 4), (1, 3, 5), (1, 4, 6), (1, 5, 1), (1, 6, 8),
                (2, 1, 1), (2, 2, 2), (2, 3, 3), (2, 4, 6), (2, 5, 7),
                (3, 1, CAST(NULL AS INTEGER)), (3, 2, 4), (3, 3, 6)
        )
    "#;

    let mut checked = 0usize;
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
                let name = format!("fuzz_p{pi}_s{si}_m{mi}");
                assert_matches_trino(&client, &base_url, &name, &sql).await;
                checked += 1;
            }
        }
    }
    // 3. Phase 2 stateful DEFINE & DISTINCT aggregate differential tests vs live Trino 484!
    let stateful_cases: Vec<(&str, &str)> = vec![
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
                    FIRST(strt.t) AS start_t,
                    LAST(down.t) AS bottom_t,
                    LAST(up.t) AS end_t,
                    LAST(down.price) AS bottom_price
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (strt down+ up+)
                DEFINE
                    down AS down.price < PREV(down.price),
                    up AS up.price > PREV(up.price)
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
                    FIRST(strt.price) AS sp,
                    LAST(down.price) AS bp
                ALL ROWS PER MATCH
                AFTER MATCH SKIP TO LAST up
                PATTERN (strt down+ up+)
                DEFINE
                    down AS down.price < PREV(down.price),
                    up AS up.price > PREV(up.price)
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
                    FIRST(a.ts) AS a_ts,
                    LAST(b.ts) AS b_ts,
                    LAST(b.ts) - FIRST(a.ts) AS dt
                ONE ROW PER MATCH
                PATTERN (a any_r* b)
                DEFINE
                    a AS ev = 'view',
                    b AS ev = 'buy' AND b.ts - FIRST(a.ts) <= 500
            )
            ORDER BY u
            "#,
        ),
        (
            "stateful_running_avg_and_count_in_define",
            r#"
            WITH nums(id, v) AS (
                VALUES (1, 10), (2, 20), (3, 18), (4, 25), (5, 30), (6, 5)
            )
            SELECT * FROM nums
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    MATCH_NUMBER() AS mn,
                    FIRST(a.id) AS first_a,
                    LAST(b.id) AS last_b,
                    COUNT(b.*) AS cnt_b
                ONE ROW PER MATCH
                PATTERN (a+ b+)
                DEFINE
                    a AS v <= 20,
                    b AS b.v > AVG(a.v) AND COUNT(b.*) <= 2
            )
            ORDER BY mn
            "#,
        ),
    ];

    for (name, sql) in &stateful_cases {
        assert_matches_trino(&client, &base_url, name, sql).await;
    }

    // 4. Stateful DEFINE randomized fuzz matrix vs live Trino 484 (16 queries)
    let stateful_patterns: &[(&str, &str)] = &[
        ("s d+ u+", "d AS d.v < PREV(d.v), u AS u.v > PREV(u.v)"),
        (
            "a b+",
            "a AS v <= 5, b AS b.v > FIRST(a.v) AND COUNT(b.*) <= 2",
        ),
        (
            "a b* c",
            "a AS v >= 2, b AS b.v >= PREV(b.v), c AS c.v < PREV(c.v)",
        ),
        ("a+ b+", "a AS a.v <= 5, b AS b.v > AVG(a.v)"),
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
                let name = format!("stateful_fuzz_p{pi}_s{si}_m{mi}");
                assert_matches_trino(&client, &base_url, &name, &sql).await;
                stateful_checked += 1;
            }
        }
    }

    // 5. Edge-case & UDAF / CLASSIFIER nav / SUM(PREV) hand cases vs live Trino 484!
    let edge_cases: Vec<(&str, &str)> = vec![
        (
            "classifier_nav_and_sum_prev",
            r#"
            WITH seq(id, tag, price) AS (
                VALUES (1, 'A', 10), (2, 'B', 20), (3, 'B', 30), (4, 'C', 40)
            )
            SELECT * FROM seq
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    MATCH_NUMBER() AS mn,
                    FIRST(CLASSIFIER()) AS first_cls,
                    LAST(CLASSIFIER()) AS last_cls,
                    LAST(CLASSIFIER(), 1) AS prev_last_cls,
                    PREV(CLASSIFIER(), 1) AS prev_cls,
                    NEXT(CLASSIFIER(), 1) AS next_cls,
                    PREV(a.price, 0) AS prev0_a,
                    FIRST(b.price, 100) AS oob_first
                ONE ROW PER MATCH
                PATTERN (a b+ c)
                DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
            )
            "#,
        ),
        (
            "udaf_stddev_and_bool_and",
            r#"
            WITH seq(id, tag, price) AS (
                VALUES (1, 'A', 10), (2, 'B', 20), (3, 'B', 30), (4, 'C', 40)
            )
            SELECT * FROM seq
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    ROUND(STDDEV_SAMP(price), 4) AS sd_all,
                    ROUND(STDDEV_POP(b.price), 4) AS sd_b,
                    BOOL_AND(price > 0) AS all_pos,
                    BOOL_OR(price > 35) AS any_hi
                ONE ROW PER MATCH
                PATTERN (a b+ c)
                DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
            )
            "#,
        ),
        (
            "multi_col_partition_desc_nulls_first_and_null_define",
            r#"
            WITH t(p1, p2, id, tag, v) AS (
                VALUES
                    ('x', 1, 1, 'A', 10),
                    ('x', 2, 1, 'A', CAST(NULL AS INTEGER)),
                    ('x', 2, 2, 'B', 20),
                    ('y', 1, 3, 'A', 30),
                    ('y', 1, 2, 'B', 25),
                    ('y', 1, 1, 'B', 15),
                    ('y', 1, 0, 'C', 5)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p1, p2
                ORDER BY id DESC NULLS FIRST
                MEASURES MATCH_NUMBER() AS mn, FIRST(id) AS first_id, LAST(id) AS last_id
                ONE ROW PER MATCH
                PATTERN (a b* c?)
                DEFINE a AS v > 5, b AS tag = 'B', c AS tag = 'C'
            )
            ORDER BY p1, p2, mn
            "#,
        ),
        (
            "omitted_measures_all_rows",
            r#"
            WITH t(p, id, tag) AS (
                VALUES ('p1', 1, 'A'), ('p1', 2, 'B'), ('p1', 3, 'C')
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id
                ALL ROWS PER MATCH
                PATTERN (a b+ c)
                DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
            )
            ORDER BY p, id
            "#,
        ),
        (
            "joined_match_recognize_ctes",
            r#"
            WITH t(p, id, tag) AS (
                VALUES ('p1', 1, 'A'), ('p1', 2, 'B'), ('p1', 3, 'C'), ('p2', 1, 'A'), ('p2', 2, 'C')
            ),
            m1 AS (
                SELECT * FROM t
                MATCH_RECOGNIZE (
                    PARTITION BY p ORDER BY id
                    MEASURES FIRST(id) AS s_id
                    ONE ROW PER MATCH
                    PATTERN (a b* c)
                    DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
                )
            ),
            m2 AS (
                SELECT * FROM t
                MATCH_RECOGNIZE (
                    PARTITION BY p ORDER BY id
                    MEASURES LAST(id) AS e_id, COUNT(*) AS cnt
                    ONE ROW PER MATCH
                    PATTERN (a b* c)
                    DEFINE a AS tag = 'A', b AS tag = 'B', c AS tag = 'C'
                )
            )
            SELECT m1.p, m1.s_id, m2.e_id, m2.cnt
            FROM m1 JOIN m2 ON m1.p = m2.p
            ORDER BY m1.p
            "#,
        ),
    ];

    for (name, sql) in &edge_cases {
        assert_matches_trino(&client, &base_url, name, sql).await;
    }

    // 6. Seeded PRNG randomized SQL + randomized table fuzzer vs live Trino 484 (50 queries)
    let mut rng_state: u64 = 0x5EED_CAFE_1234_ABCD;
    let mut next_u64 = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    let rand_patterns: &[(&str, &str)] = &[
        (
            "a b* c?",
            "a AS v >= 1 AND v <= 4, b AS v >= 3 AND v <= 6, c AS v >= 5",
        ),
        ("(a | a b)", "a AS v <= 4, b AS v >= 3"),
        ("s d+ u+", "d AS d.v < PREV(d.v), u AS u.v > PREV(u.v)"),
        ("a+ b+", "a AS a.v <= 5, b AS b.v > AVG(a.v)"),
        (
            "a b+",
            "a AS v >= 2, b AS b.v - FIRST(a.v) <= 3 AND COUNT(b.*) <= 3",
        ),
        ("PERMUTE(a, b)", "a AS v <= 4, b AS v >= 4"),
    ];
    let rand_orders = ["ORDER BY id ASC", "ORDER BY id DESC NULLS FIRST"];
    let rand_skips = [
        "AFTER MATCH SKIP PAST LAST ROW",
        "AFTER MATCH SKIP TO NEXT ROW",
    ];

    let mut random_checked = 0usize;
    for q in 0..50usize {
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
        let (pat, defs) = rand_patterns[(next_u64() as usize) % rand_patterns.len()];
        let ord = rand_orders[(next_u64() as usize) % rand_orders.len()];
        let skip = rand_skips[(next_u64() as usize) % rand_skips.len()];
        let (mode, outer_ord) = if (next_u64() % 2) == 0 {
            ("ONE ROW PER MATCH", "ORDER BY p, mn")
        } else {
            (
                "ALL ROWS PER MATCH OMIT EMPTY MATCHES",
                "ORDER BY p, mn, id",
            )
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
                    FIRST(v) AS fv,
                    LAST(v) AS lv,
                    COUNT(*) AS cnt,
                    SUM(v) AS sv
                {mode}
                {skip}
                PATTERN ({pat})
                DEFINE {defs}
            )
            {outer_ord}
            "#
        );
        assert_matches_trino(&client, &base_url, &format!("prng_q{q}"), &sql).await;
        random_checked += 1;
    }

    eprintln!(
        "Live Trino differential oracle passed all 13 stateless + {} stateful + {} edge hand cases + {} stateless fuzz + {} stateful fuzz + {} PRNG random SQL queries!",
        stateful_cases.len(),
        edge_cases.len(),
        checked,
        stateful_checked,
        random_checked
    );
}
