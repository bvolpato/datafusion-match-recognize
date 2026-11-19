// SPDX-License-Identifier: Apache-2.0

//! Classic SQL:2016 stock-price V-shape detection using stateful `PREV()` in `DEFINE`.
//!
//! Run with: `cargo run --example stock_vshape`

use std::sync::Arc;

use arrow::array::{ArrayRef, Int64Array, RecordBatch, StringArray};
use datafusion::prelude::SessionConfig;
use datafusion_match_recognize::{MatchRecognizeConfig, session_context_with_match_recognize};

#[tokio::main]
async fn main() -> datafusion_common::Result<()> {
    let ctx = session_context_with_match_recognize(
        SessionConfig::new(),
        MatchRecognizeConfig::default(),
    )?;

    let batch = RecordBatch::try_from_iter(vec![
        (
            "symbol",
            Arc::new(StringArray::from(vec![
                "ACME", "ACME", "ACME", "ACME", "ACME", "ACME", "ACME", "ACME",
            ])) as ArrayRef,
        ),
        (
            "t",
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8])),
        ),
        (
            "price",
            Arc::new(Int64Array::from(vec![100, 90, 80, 85, 95, 95, 70, 110])),
        ),
    ])?;
    ctx.register_batch("ticker", batch)?;

    let sql = r#"
        SELECT * FROM ticker
        MATCH_RECOGNIZE (
            PARTITION BY symbol
            ORDER BY t
            MEASURES
                MATCH_NUMBER() AS match_num,
                FIRST(strt.t) AS start_t,
                LAST(down.t) AS bottom_t,
                LAST(up.t) AS peak_t,
                LAST(down.price) AS bottom_price,
                LAST(up.price) AS peak_price
            ONE ROW PER MATCH
            AFTER MATCH SKIP TO LAST up
            PATTERN (strt down+ up+)
            DEFINE
                down AS down.price < PREV(down.price),
                up AS up.price > PREV(up.price)
        )
        ORDER BY symbol, match_num
    "#;

    ctx.sql(sql).await?.show().await?;
    Ok(())
}
