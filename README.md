# datafusion-match-recognize

SQL:2016 `MATCH_RECOGNIZE` (Row Pattern Recognition) extension crate for [Apache DataFusion](https://datafusion.apache.org/).

This crate builds `MATCH_RECOGNIZE` purely on top of DataFusion public extension APIs (`RelationPlanner`, `UserDefinedLogicalNode`, `ExtensionPlanner`, `ExecutionPlan`, `OptimizerRule`, `LogicalExtensionCodec`, and `PhysicalExtensionCodec`) without modifying DataFusion core. It targets `datafusion-contrib` under the `Apache-2.0` license.

## Architecture

- **SQL Planning (`MatchRecognizeRelationPlanner`):** Intercepts `sqlparser::ast::TableFactor::MatchRecognize` via DataFusion `RelationPlanner`, validates SQL:2016 variable rules, and builds a `MatchRecognizeNode`.
- **Measure Slot Decomposition:** Extracts pattern-recognition leaves (`MATCH_NUMBER()`, `CLASSIFIER()`, `FIRST`, `LAST`, `PREV`, `NEXT`, symbol-qualified columns, and scoped aggregates) into synthetic slots while preserving arbitrary outer DataFusion scalar expressions (`CASE WHEN`, arithmetic, `COALESCE`, `CAST`, and UDFs) with automatic type coercion.
- **Priority-Ordered PikeVM Matcher:** Compiles patterns into NFA bytecode preserving SQL:2016 leftmost-ordered alternation and greedy vs. reluctant quantifier priority. Uses per-position `pc` deduplication and a flat arena for match histories, providing linear worst-case matching time and O(1) drop stack usage.
- **Partition-Streaming Execution (`MatchRecognizeExec`):** Declares `Distribution::KeyPartitioned` (or `SinglePartition`) and `[PARTITION BY..., ORDER BY...]` ordering requirements so DataFusion inserts or elides `RepartitionExec` and `SortExec`. Streams one partition at a time with `MemoryReservation` accounting and vectorized Arrow `DEFINE` evaluation packed into per-row `u64` bitmasks.
- **Optimizer and Serialization:** Includes `MatchRecognizeFilterPushdown` (pushes partition-only predicates below `MatchRecognizeNode`), `necessary_children_exprs` column pruning, and optional `proto` codecs (`MatchRecognizeLogicalCodec` and `MatchRecognizePhysicalCodec`) for distributed execution.

## Quick Start

```rust
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

    let sql = r#"
        WITH events(user_id, ts, event_name) AS (
            VALUES
                ('u1', 1, 'view'),
                ('u1', 2, 'scroll'),
                ('u1', 3, 'buy')
        )
        SELECT * FROM events
        MATCH_RECOGNIZE (
            PARTITION BY user_id
            ORDER BY ts
            MEASURES
                FIRST(a.ts) AS start_ts,
                LAST(c.ts) AS end_ts,
                LAST(c.ts) - FIRST(a.ts) AS elapsed
            ONE ROW PER MATCH
            AFTER MATCH SKIP PAST LAST ROW
            PATTERN (a any_row* c)
            DEFINE
                a AS event_name = 'view',
                c AS event_name = 'buy'
        )
    "#;

    ctx.sql(sql).await?.show().await?;
    Ok(())
}
```

## Verification Suite

The test suite uses a 4-layer oracle strategy:

1. **Layer 1 (`tests/oracle.rs`):** Independent recursive backtracking reference implementation of SQL:2016 row pattern semantics fuzz-tested against the PikeVM matcher across 1,500 random patterns, skip modes, and output modes.
2. **Layer 2a (`tests/sql_tests.rs`, `tests/slt/`):** SQL Logic Tests ported from Trino and Calcite plus multi-step funnel queries, executed across multiple batch sizes and partition counts.
3. **Layer 2b (`tests/trino_oracle.rs`):** Live differential SQL oracle executing hand-crafted corner cases and randomized multi-partition queries against a real Trino Docker container (`trinodb/trino`).
4. **Layer 3 (`tests/invariance.rs`):** Matrix tests across batch sizes (`1, 2, 3, 7, 8192`), partition counts (`1, 4`), sliced Arrow batches, diverse Arrow types, optimizer pushdown equivalence, `GreedyMemoryPool` limits, and logical/physical protobuf round-trips.
5. **Layer 4 (`tests/complexity.rs`, `benches/funnel_bench.rs`):** Adversarial anti-backtracking complexity tests (`(A+)+ B`, `(A*)* B`, `(A | A)* B`, `(A?){25} A{25} B`) and Criterion throughput benchmarks.

### Running Tests

```sh
# Run Layers 1, 2a, 3, and 4 (no external services required)
cargo test

# Run Layer 2b live differential oracle against Trino in Docker
docker run -d --name mr-trino-oracle -p 18080:8080 trinodb/trino:latest
cargo test --test trino_oracle -- --ignored --nocapture
docker rm -f mr-trino-oracle
```

## Current Scope and Known Gaps

See [tests/divergences.md](tests/divergences.md) for documented differences from Trino and upstream `sqlparser-rs` parser gaps.

## Acknowledgments

This project builds on prior work and design discussions across the Apache DataFusion, Trino, and Apache Calcite communities:

- **Geoffrey Claude (`geoffreyclaude`)** for designing and contributing DataFusion's `RelationPlanner` extension API ([apache/datafusion#18078](https://github.com/apache/datafusion/issues/18078)) and exploring the initial `MATCH_RECOGNIZE` prototype ([apache/datafusion#16685](https://github.com/apache/datafusion/pull/16685)).
- **Andrew Lamb (`alamb`) and the Apache DataFusion maintainers** for guiding SQL extensions toward modular `datafusion-contrib` crates built on public extension APIs.
- **Trino (`trinodb/trino`)** for its SQL:2016 row pattern recognition bytecode compiler (`IrRowPatternToProgramRewriter`), documentation, and `TestRowPatternMatching` test suite, which informed our NFA compilation strategy and live differential oracle.
- **Apache Calcite (`apache/calcite`)** for its `match.iq` SQL:2016 test cases.

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
