// SPDX-License-Identifier: Apache-2.0

//! Layer 2b verification: live differential SQL oracle against a real Trino (`trinodb/trino`)
//! Docker container on `TRINO_URL` (default `http://localhost:18080`).
//!
//! Run with:
//!   cargo test --test trino_oracle -- --ignored --nocapture

use arrow::array::{Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::{
    MatchRecognizeConfig, session_context_with_match_recognize,
};
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
        serde_json::Value::Array(items) => {
            let formatted: Vec<String> =
                items.into_iter().map(format_json_cell).collect();
            format!("[{}]", formatted.join(", "))
        }
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

async fn assert_matches_trino(
    client: &reqwest::Client,
    base_url: &str,
    name: &str,
    sql: &str,
) {
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
            eprintln!(
                "[{name}] Both errored as expected (DF: {df_err} | Trino: {trino_err})"
            );
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
    let base_url = std::env::var("TRINO_URL")
        .unwrap_or_else(|_| "http://localhost:18080".to_string());
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
        (
            "trino_back_reference_multi_symbol",
            r#"
            WITH t(value) AS (VALUES (1), (2), (3), (4), (5), (6))
            SELECT value, classy, defining_condition FROM t
            MATCH_RECOGNIZE (
                ORDER BY value
                MEASURES
                    CLASSIFIER() AS classy,
                    PREV(LAST(A.value), 3) + FIRST(A.value) + PREV(LAST(B.value), 2) AS defining_condition
                ALL ROWS PER MATCH
                PATTERN ((A | B)* X)
                DEFINE X AS value = PREV(LAST(A.value), 3) + FIRST(A.value) + PREV(LAST(B.value), 2)
            )
            ORDER BY value
            "#,
        ),
        (
            "trino_classifier_past_current_row_measures",
            r#"
            WITH t(id, value) AS (VALUES (1, 90), (2, 80), (3, 70), (4, 80))
            SELECT m.id, m.value, m.label, m.next_label FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    CLASSIFIER() AS label,
                    NEXT(CLASSIFIER()) AS next_label
                ALL ROWS PER MATCH
                PATTERN (A B+ C+)
                DEFINE
                    B AS B.value < PREV(B.value),
                    C AS C.value > PREV(C.value)
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_classifier_past_current_row_define",
            r#"
            WITH t(id, value) AS (VALUES (1, 90), (2, 80), (3, 70), (4, 80))
            SELECT m.id, m.val FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES value AS val
                ALL ROWS PER MATCH
                PATTERN (A+)
                DEFINE A AS NEXT(CLASSIFIER()) = 'A'
            ) AS m
            "#,
        ),
        (
            "trino_case_sensitive_labels",
            r#"
            WITH t(id, value) AS (VALUES (1, 90), (2, 80), (3, 70), (4, 80))
            SELECT m.id, m.label FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES CLASSIFIER() AS label
                ALL ROWS PER MATCH
                PATTERN (a "b"+ C+)
                DEFINE
                    "b" AS "b".value < PREV("b".value),
                    C AS C.value > PREV(C.value)
            ) AS m
            ORDER BY m.id
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
            "trino_multiple_aggs_in_define",
            r#"
            WITH t(id) AS (VALUES (1), (2), (3), (4), (5), (6), (7), (8))
            SELECT m.match_no, m.labels FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    MATCH_NUMBER() AS match_no,
                    array_agg(CLASSIFIER()) AS labels
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((A | B){4})
                DEFINE
                    A AS max(id - 2 * MATCH_NUMBER()) > 1 AND max(CLASSIFIER()) = 'B',
                    B AS min(lower(CLASSIFIER())) = 'b' OR min(MATCH_NUMBER() + 100) < 0
            ) AS m
            ORDER BY m.match_no
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
            "trino_exclusions_array_agg_classifier",
            r#"
            WITH t(part, id, value) AS (
                VALUES
                    ('p1', 1, '1a'), ('p1', 2, '1b'), ('p1', 3, '1c'), ('p1', 4, '1d'), ('p1', 5, '1e'),
                    ('p2', 1, '2a'), ('p2', 2, '2b'), ('p2', 3, '2c'), ('p2', 4, '2d'), ('p2', 5, '2e')
            )
            SELECT m.part, m.measure_1, m.measure_2 FROM t
            MATCH_RECOGNIZE (
                PARTITION BY part
                ORDER BY id
                MEASURES
                    array_agg(value) AS measure_1,
                    array_agg(value || CLASSIFIER()) AS measure_2
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (P ({- Q -}) ({- R -}) S)
                DEFINE P AS id > 1
            ) AS m
            ORDER BY m.part, m.id
            "#,
        ),
        (
            "trino_tentative_label_avg_b",
            r#"
            WITH t(id, value) AS (VALUES (1, 4), (2, 6), (3, 0))
            SELECT m.id, m.classy, m.running_avg_b FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES avg(B.value) AS running_avg_b, CLASSIFIER() AS classy
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((A | B)*)
                DEFINE A AS avg(B.value) = 5
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_tentative_label_avg_a",
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
            "trino_tentative_label_runtime_match_number",
            r#"
            WITH t(id, part, val) AS (
                VALUES
                    (1, 'p1', CAST(1 AS INTEGER)), (2, 'p1', 2), (3, 'p1', 3), (4, 'p1', 4), (5, 'p1', 5),
                    (1, 'p2', 2), (2, 'p2', 4), (3, 'p2', 6), (4, 'p2', 8), (5, 'p2', 10)
            )
            SELECT m.part, m.id, m.classy, m.running_max FROM t
            MATCH_RECOGNIZE (
                PARTITION BY part
                ORDER BY id
                MEASURES max(val) AS running_max, CLASSIFIER() AS classy
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((A | B)*)
                DEFINE A AS max(val + MATCH_NUMBER()) > 5
            ) AS m
            ORDER BY m.part, m.id
            "#,
        ),
        (
            "trino_agg_args_duplicate_col_and_qualified_classifier",
            r#"
            WITH t(id, value) AS (VALUES (1, 'a'), (2, 'b'), (3, 'c'))
            SELECT m.id, m.dup, m.a_cls, m.b_tagged FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    array_agg(value || value || CLASSIFIER()) AS dup,
                    array_agg(CLASSIFIER(A)) AS a_cls,
                    array_agg(B.value || CLASSIFIER(B)) AS b_tagged
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A B A)
                DEFINE A AS true
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_count_empty_and_label_column_collision",
            r#"
            WITH t(id, a) AS (VALUES (1, 'p'), (2, 'q'), (3, CAST(NULL AS VARCHAR)), (4, 's'))
            SELECT m.id, m.classy, m.m1, m.m2, m.m3, m.m4 FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    CLASSIFIER() AS classy,
                    count(A.a) AS m1,
                    count(a) AS m2,
                    count(A.*) AS m3,
                    count() AS m4
                ALL ROWS PER MATCH
                PATTERN (A B A A)
                DEFINE A AS true
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_set_partitioning_3way",
            r#"
            WITH t(id) AS (VALUES (1), (2), (3), (4), (5), (6))
            SELECT m.id, m.running_labels FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES array_agg(CLASSIFIER()) AS running_labels
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (^ (A | B | C)* (LAST_A | LAST_B | LAST_C) $)
                DEFINE
                    LAST_A AS sum(A.id) + id = sum(B.id) AND sum(B.id) = sum(C.id),
                    LAST_B AS sum(B.id) + id = sum(A.id) AND sum(A.id) = sum(C.id),
                    LAST_C AS sum(C.id) + id = sum(A.id) AND sum(A.id) = sum(B.id)
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_scalar_in_and_like_predicates",
            r#"
            WITH t(id, tag, value) AS (
                VALUES (1, 'alpha_1', 100), (2, 'alpha_2', 200), (3, 'beta_3', 300), (4, 'beta_4', 400)
            )
            SELECT m.id, m.v1, m.v2, m.v3, m.is_a, m.diff_prev FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    FIRST(A.value) IN (100, LAST(A.value)) AS v1,
                    CLASSIFIER() IN ('X', lower(CLASSIFIER())) AS v2,
                    MATCH_NUMBER() IN (0, MATCH_NUMBER()) AS v3,
                    CLASSIFIER() LIKE 'A%' AS is_a,
                    value IS DISTINCT FROM PREV(value) AS diff_prev
                ALL ROWS PER MATCH
                PATTERN (A+ B+)
                DEFINE
                    A AS A.tag LIKE 'alpha%',
                    B AS LAST(A.tag) LIKE 'alpha%' AND B.value IS DISTINCT FROM LAST(A.value)
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_exponential_match_and_kill_thread",
            r#"
            WITH t(value) AS (VALUES (1), (2), (3), (4), (5), (6), (7), (8), (9))
            SELECT m.classy FROM t
            MATCH_RECOGNIZE (
                ORDER BY value
                MEASURES CLASSIFIER() AS classy
                ALL ROWS PER MATCH
                PATTERN ((A | B)+ LAST)
                DEFINE LAST AS
                    FIRST(CLASSIFIER()) = 'B' AND
                    FIRST(CLASSIFIER(), 1) = 'B' AND
                    FIRST(CLASSIFIER(), 2) = 'B' AND
                    FIRST(CLASSIFIER(), 3) = 'B' AND
                    FIRST(CLASSIFIER(), 4) = 'B' AND
                    FIRST(CLASSIFIER(), 5) = 'B' AND
                    FIRST(CLASSIFIER(), 6) = 'B' AND
                    FIRST(CLASSIFIER(), 7) = 'B'
            ) AS m
            ORDER BY m.value
            "#,
        ),
        (
            "trino_kill_thread_nullable_loop",
            r#"
            WITH t(x) AS (VALUES (1), (2), (3), (4), (5))
            SELECT * FROM t
            MATCH_RECOGNIZE (
                MEASURES 'foo' AS foo
                PATTERN ((Y?){2,})
                DEFINE Y AS true
            )
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
            "flink_logical_offsets_and_star_variable",
            r#"
            WITH ticker(id, symbol, tstamp, price) AS (
                VALUES
                    (1, 'ACME', 1, 20), (2, 'ACME', 2, 19), (3, 'ACME', 3, 18), (4, 'ACME', 4, 17),
                    (5, 'ACME', 5, 16), (6, 'ACME', 6, 15), (7, 'ACME', 7, 14), (8, 'ACME', 8, 20)
            )
            SELECT * FROM ticker
            MATCH_RECOGNIZE (
                ORDER BY tstamp
                MEASURES
                    FIRST(id, 0) AS id0, FIRST(id, 3) AS id3, FIRST(id, 7) AS id7,
                    LAST(id, 0) AS id8, LAST(id, 3) AS id11, LAST(id, 7) AS id15
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ("DOWN_Q"{2,} UP)
                DEFINE
                    "DOWN_Q" AS price < LAST(price, 1) OR LAST(price, 1) IS NULL,
                    UP AS price = FIRST(price) AND price > FIRST(price, 3) AND price = LAST(price, 7)
            ) AS T
            "#,
        ),
        (
            "flink_aggregates_and_null_inputs",
            r#"
            WITH my_table(id, name, price) AS (
                VALUES
                    (1, 'a', 10), (2, 'z', 10),
                    (3, 'b', CAST(NULL AS INTEGER)), (4, 'c', CAST(NULL AS INTEGER)),
                    (5, 'd', 3), (6, 'c', 3), (7, 'c', 3), (8, 'c', 3), (9, 'c', 2)
            )
            SELECT * FROM my_table
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    SUM(A.price) AS sum_a,
                    COUNT(A.id) AS count_a_id,
                    COUNT(A.price) AS count_a_price,
                    COUNT(*) AS count_all,
                    COUNT(price) AS count_all_price,
                    LAST(id) AS end_id
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (A+ C)
                DEFINE
                    A AS SUM(A.price) < 30,
                    C AS C.name = 'c'
            ) AS T
            "#,
        ),
        (
            "flink_multi_column_order_by",
            r#"
            WITH my_table(ts, secondary_order, ternary_order, name, id) AS (
                VALUES
                    (2, 12, 1, 'a', 1), (1, 11, 2, 'b', 2), (3, 10, 3, 'c', 3),
                    (4,  8, 4, 'a', 4), (4,  9, 5, 'b', 5), (5,  7, 6, 'c', 6),
                    (6,  6, 8, 'a', 7), (6,  6, 7, 'b', 8), (8,  4, 9, 'c', 9),
                    (9,  3, 10, 'a', 10), (10, 2, 11, 'b', 11), (11, 1, 12, 'c', 12)
            )
            SELECT T.aid, T.bid, T.cid FROM my_table
            MATCH_RECOGNIZE (
                ORDER BY ts, secondary_order DESC, ternary_order ASC
                MEASURES A.id AS aid, B.id AS bid, C.id AS cid
                PATTERN (A B C)
                DEFINE A AS name = 'a', B AS name = 'b', C AS name = 'c'
            ) AS T
            "#,
        ),
        (
            "trino_desugar_like_in_aggs",
            r#"
            WITH t(id) AS (VALUES (1), (2), (3))
            SELECT m.id, m.classy, m.m1, m.m2 FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    CLASSIFIER() AS classy,
                    CLASSIFIER() LIKE '%X' AS m1,
                    bool_or(CLASSIFIER() LIKE '%B') AS m2
                ALL ROWS PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((A | B)* X)
                DEFINE X AS PREV(CLASSIFIER()) LIKE '%B' AND bool_or(CLASSIFIER() LIKE '%A')
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_decimal_sum_in_define",
            r#"
            WITH t(id) AS (VALUES (1), (2), (3))
            SELECT m.id, m.classy FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES CLASSIFIER() AS classy
                ALL ROWS PER MATCH
                PATTERN ((A | B)* X)
                DEFINE X AS sum(CAST(B.id AS DECIMAL(10, 2))) = 1.0
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_classifier_inside_nav_scalar_expr",
            r#"
            WITH t(id, val) AS (VALUES (1, 10), (2, 20), (3, 30))
            SELECT m.id, m.cls, m.prev_lower_cls, m.a_len_plus_id FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES
                    CLASSIFIER() AS cls,
                    PREV(LAST(lower(CLASSIFIER()))) AS prev_lower_cls,
                    LAST(length(CLASSIFIER(A)) + A.id + 1) AS a_len_plus_id
                ALL ROWS PER MATCH
                PATTERN (A B+)
                DEFINE A AS id = 1, B AS id > 1
            ) AS m
            ORDER BY m.id
            "#,
        ),
        (
            "trino_direct_match_number_in_define",
            r#"
            WITH t(id) AS (VALUES (1), (2), (3))
            SELECT m.id, m.mn, m.cls FROM t
            MATCH_RECOGNIZE (
                ORDER BY id
                MEASURES MATCH_NUMBER() AS mn, CLASSIFIER() AS cls
                ALL ROWS PER MATCH
                PATTERN (A | B)
                DEFINE A AS MATCH_NUMBER() = 1, B AS MATCH_NUMBER() = 2
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
            "trino_aliased_filter_pushdown_and_null_order_by",
            r#"
            WITH t(p, id, v) AS (
                VALUES
                    ('p1', CAST(NULL AS INTEGER), 100),
                    ('p1', 1, 10),
                    ('p1', 2, 20),
                    ('p2', 1, 5),
                    ('p2', 2, 6)
            )
            SELECT m.p, m.mn, m.k, m.fv FROM t
            MATCH_RECOGNIZE (
                PARTITION BY p
                ORDER BY id ASC NULLS FIRST
                MEASURES MATCH_NUMBER() AS mn, 42 AS k, FIRST(v) AS fv
                ONE ROW PER MATCH
                PATTERN (A B)
                DEFINE A AS v >= 5, B AS v >= 6
            ) AS m
            WHERE m.p = 'p1' AND m.fv >= 50
            "#,
        ),
        // Negative semantic & runtime error parity checks from Trino TestAnalyzer.java
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
            "err_parity_mixed_labeled_unlabeled_agg",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES count(B.v < 5 OR v > 5) AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_nested_nav_last_next",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES LAST(NEXT(v, 2)) AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_non_immediate_nav_nesting",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES PREV(LAST(v, 2) + 5) AS x PATTERN (A B+) DEFINE B AS true
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
            "err_parity_window_in_measures",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES max(v) OVER () AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_filter_in_agg",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES max(v) FILTER (WHERE true) AS x PATTERN (A B+) DEFINE B AS true
            )
            "#,
        ),
        (
            "err_parity_const_only_nav",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES PREV(LAST('no_column')) AS x PATTERN (A B+) DEFINE B AS true
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
        (
            "risingwave_self_ref_define_tautology_and_equal_runs",
            r#"
            WITH t(id, ts, v) AS (
                VALUES (1, 1, 7), (1, 2, 7), (1, 3, 9), (1, 4, 11), (1, 5, 11), (1, 6, 11)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY id
                ORDER BY ts
                MEASURES first(a.v) AS v_first, last(a.v) AS v_last, count(*) AS n
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (a+)
                DEFINE a AS first(a.v) = a.v AND last(a.v) = a.v
            )
            ORDER BY v_first
            "#,
        ),
        (
            "risingwave_nested_nav_case_in_distinct_array_subscript",
            r#"
            WITH t(id, ts, v) AS (
                VALUES (1, 1, 5), (1, 2, 10), (1, 3, 2), (1, 4, 20)
            )
            SELECT * FROM t
            MATCH_RECOGNIZE (
                PARTITION BY id
                ORDER BY ts
                MEASURES a.v AS av, b.v AS bv
                ONE ROW PER MATCH
                PATTERN (a b)
                DEFINE
                    a AS a.v > 0,
                    b AS b.v > CASE WHEN prev(b.v) > 0 THEN prev(b.v) ELSE 0 END
                        AND b.v NOT IN (prev(b.v), last(a.v))
                        AND (prev(b.v) IS DISTINCT FROM b.v)
                        AND ((ARRAY[prev(b.v), last(a.v)])[1] < b.v)
            )
            ORDER BY av
            "#,
        ),
        (
            "snowflake_stock_vshape_date_columns_and_skip_to_last",
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
                    COUNT(row_with_price_decrease.*) AS num_decreases,
                    COUNT(row_with_price_increase.*) AS num_increases
                ONE ROW PER MATCH
                AFTER MATCH SKIP TO LAST row_with_price_increase
                PATTERN (row_before_decrease row_with_price_decrease+ row_with_price_increase+)
                DEFINE
                    row_with_price_decrease AS price < PREV(price),
                    row_with_price_increase AS price > PREV(price)
            )
            WHERE company = 'ABCD'
            ORDER BY company, match_number
            "#,
        ),
        (
            "err_parity_last_distinct_in_define",
            r#"
            WITH t(id, v) AS (VALUES (1, 10), (2, 20))
            SELECT * FROM t MATCH_RECOGNIZE (
                ORDER BY id MEASURES a.v AS p PATTERN (a b)
                DEFINE a AS a.v > 0, b AS b.v < last(DISTINCT a.v)
            )
            "#,
        ),
    ];

    for (name, sql) in &edge_cases {
        assert_matches_trino(&client, &base_url, name, sql).await;
    }

    // 6. Augmented seeded PRNG randomized SQL + randomized table fuzzer vs live Trino 484 (200 queries)
    let mut rng_state: u64 = 0x5EED_CAFE_1234_ABCD;
    let mut next_u64 = || -> u64 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        rng_state
    };

    // (pattern, defines, has_exclusion, can_skip_to_last_b)
    let rand_patterns: &[(&str, &str, bool, bool)] = &[
        (
            "a b* c?",
            "a AS v >= 1 AND v <= 4, b AS v >= 3 AND v <= 6, c AS v >= 5",
            false,
            false,
        ),
        ("(a | a b)", "a AS v <= 4, b AS v >= 3", false, false),
        (
            "s d+ u+",
            "d AS d.v < PREV(d.v), u AS u.v > PREV(u.v)",
            false,
            false,
        ),
        ("a+ b+", "a AS a.v <= 5, b AS b.v > AVG(a.v)", false, true),
        (
            "a b+",
            "a AS v >= 2, b AS b.v - FIRST(a.v) <= 3 AND COUNT(b.*) <= 3",
            false,
            true,
        ),
        ("PERMUTE(a, b)", "a AS v <= 4, b AS v >= 4", false, false),
        ("a*", "a AS v >= 2 AND v <= 6", false, false),
        ("({- a -}) b+", "a AS v <= 4, b AS v >= 3", true, true),
        ("({- a -}) | b", "a AS v <= 3, b AS v >= 5", true, false),
        (
            "a{0,2} b+",
            "a AS v <= 5 AND MATCH_NUMBER() <= 3, b AS v >= 4",
            false,
            false,
        ),
        (
            "^ a+ b*",
            "a AS v >= 2, b AS b.v >= PREV(b.v)",
            false,
            false,
        ),
        (
            "a+ b $",
            "a AS v <= 6, b AS PREV(CLASSIFIER()) = 'A' AND b.v >= 2",
            false,
            true,
        ),
        ("a+", "a AS FIRST(a.v) = a.v", false, false),
        ("(c | c? | a)*", "c AS v >= 5, a AS v <= 3", false, false),
        ("a b b", "a AS v >= 2, b AS v >= 2", false, true),
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
        "AFTER MATCH SKIP TO LAST b",
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
                    PREV(LAST(lower(CLASSIFIER()))) AS p_lcls,
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
