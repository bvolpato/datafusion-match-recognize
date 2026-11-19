// SPDX-License-Identifier: Apache-2.0

//! Layer 2a verification: SQL Logic Tests (`.slt`) ported from Trino's `TestRowPatternMatching`,
//! Apache Calcite's `match.iq`, and real-world funnel SQL queries.

use std::path::Path;

use arrow::array::{Array, RecordBatch};
use arrow::util::display::{ArrayFormatter, FormatOptions};
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_match_recognize::{MatchRecognizeConfig, session_context_with_match_recognize};
use sqllogictest::{AsyncDB, DBOutput, DefaultColumnType};

struct MatchRecognizeDb {
    ctx: SessionContext,
}

impl MatchRecognizeDb {
    fn new(batch_size: usize, target_partitions: usize) -> Self {
        let cfg = SessionConfig::new()
            .with_batch_size(batch_size)
            .with_target_partitions(target_partitions);
        let ctx = session_context_with_match_recognize(cfg, MatchRecognizeConfig::default())
            .expect("session context");
        Self { ctx }
    }
}

fn format_batch_rows(batches: &[RecordBatch]) -> Vec<Vec<String>> {
    let options = FormatOptions::default().with_null("NULL");
    let mut rows = Vec::new();
    for batch in batches {
        let formatters: Vec<ArrayFormatter<'_>> = batch
            .columns()
            .iter()
            .map(|c| ArrayFormatter::try_new(c.as_ref(), &options).expect("formatter"))
            .collect();
        for row_idx in 0..batch.num_rows() {
            let mut row = Vec::with_capacity(batch.num_columns());
            for (col_idx, fmt) in formatters.iter().enumerate() {
                if batch.column(col_idx).is_null(row_idx) {
                    row.push("NULL".to_string());
                } else {
                    row.push(fmt.value(row_idx).to_string());
                }
            }
            rows.push(row);
        }
    }
    rows
}

fn arrow_type_to_slt(dt: &arrow_schema::DataType) -> DefaultColumnType {
    use arrow_schema::DataType::*;
    match dt {
        Int8 | Int16 | Int32 | Int64 | UInt8 | UInt16 | UInt32 | UInt64 => {
            DefaultColumnType::Integer
        }
        Float16 | Float32 | Float64 | Decimal128(_, _) | Decimal256(_, _) => {
            DefaultColumnType::FloatingPoint
        }
        Utf8 | LargeUtf8 | Utf8View => DefaultColumnType::Text,
        _ => DefaultColumnType::Any,
    }
}

#[async_trait::async_trait]
impl AsyncDB for MatchRecognizeDb {
    type Error = datafusion_common::DataFusionError;
    type ColumnType = DefaultColumnType;

    async fn run(&mut self, sql: &str) -> Result<DBOutput<Self::ColumnType>, Self::Error> {
        let df = self.ctx.sql(sql).await?;
        let schema = df.schema().clone();
        let batches = df.collect().await?;
        if schema.fields().is_empty() {
            return Ok(DBOutput::StatementComplete(0));
        }
        let types: Vec<DefaultColumnType> = schema
            .fields()
            .iter()
            .map(|f| arrow_type_to_slt(f.data_type()))
            .collect();
        let rows = format_batch_rows(&batches);
        Ok(DBOutput::Rows { types, rows })
    }

    fn engine_name(&self) -> &str {
        "datafusion-match-recognize"
    }

    async fn shutdown(&mut self) {}
}

async fn run_slt_file(path: &str, batch_size: usize, target_partitions: usize) {
    let full_path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    let mut runner = sqllogictest::Runner::new(|| async {
        Ok(MatchRecognizeDb::new(batch_size, target_partitions))
    });
    runner
        .run_file_async(&full_path)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "SLT failure in {path} (batch_size={batch_size}, target_partitions={target_partitions}):\n{e}"
            )
        });
}

#[tokio::test]
async fn test_slt_basic_patterns() {
    for &(bs, tp) in &[(8192, 1), (2, 4)] {
        run_slt_file("tests/slt/basic_patterns.slt", bs, tp).await;
    }
}

#[tokio::test]
async fn test_slt_measures_and_aggs() {
    for &(bs, tp) in &[(8192, 1), (2, 4)] {
        run_slt_file("tests/slt/measures_and_aggs.slt", bs, tp).await;
    }
}

#[tokio::test]
async fn test_slt_calcite_and_funnels() {
    for &(bs, tp) in &[(8192, 1), (2, 4)] {
        run_slt_file("tests/slt/calcite_and_funnels.slt", bs, tp).await;
    }
}

#[tokio::test]
async fn test_slt_stateful_and_distinct() {
    for &(bs, tp) in &[(8192, 1), (2, 4)] {
        run_slt_file("tests/slt/stateful_and_distinct.slt", bs, tp).await;
    }
}

#[tokio::test]
async fn test_slt_edge_cases() {
    for &(bs, tp) in &[(8192, 1), (2, 4)] {
        run_slt_file("tests/slt/edge_cases.slt", bs, tp).await;
    }
}

#[tokio::test]
async fn test_slt_trino_ported_suite() {
    for &(bs, tp) in &[(8192, 1), (2, 4)] {
        run_slt_file("tests/slt/trino_ported_suite.slt", bs, tp).await;
    }
}
