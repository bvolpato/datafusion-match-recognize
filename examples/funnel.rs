// SPDX-License-Identifier: Apache-2.0

//! Multi-step product funnel drop-off analysis using `MATCH_RECOGNIZE`.
//!
//! Run with: `cargo run --example funnel`

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::{
    MatchRecognizeConfig, session_context_with_match_recognize,
};

#[tokio::main]
async fn main() -> datafusion_common::Result<()> {
    let ctx = session_context_with_match_recognize(
        SessionConfig::new(),
        MatchRecognizeConfig::default(),
    )?;

    let batch = RecordBatch::try_from_iter(vec![
        (
            "session_id",
            Arc::new(StringArray::from(vec![
                "s1", "s1", "s1", "s1", "s2", "s2", "s3",
            ])) as ArrayRef,
        ),
        (
            "ts",
            Arc::new(Int64Array::from(vec![10, 20, 45, 80, 10, 35, 10])),
        ),
        (
            "event_name",
            Arc::new(StringArray::from(vec![
                "view", "scroll", "cart", "buy", "view", "cart", "view",
            ])),
        ),
    ])?;
    ctx.register_batch("events", batch)?;

    let sql = r#"
        SELECT * FROM events
        MATCH_RECOGNIZE (
            PARTITION BY session_id
            ORDER BY ts
            MEASURES
                FIRST(a.ts) AS view_ts,
                FIRST(b.ts) AS cart_ts,
                FIRST(c.ts) AS buy_ts,
                CASE
                    WHEN c.ts IS NOT NULL THEN FIRST(c.ts) - FIRST(a.ts)
                    ELSE NULL
                END AS funnel_duration
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
        ORDER BY session_id
    "#;

    ctx.sql(sql).await?.show().await?;
    Ok(())
}
