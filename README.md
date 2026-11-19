# DataFusion Match Recognize

[![CI][ci-badge]][ci-url]
[![Apache licensed][license-badge]][license-url]

SQL:2016 `MATCH_RECOGNIZE` (Row Pattern Recognition) for [Apache DataFusion](https://github.com/apache/datafusion), built purely on DataFusion public extension APIs.

> [!NOTE]
> This project is not part of Apache DataFusion.

`datafusion-match-recognize` is a library for adding SQL:2016 row pattern recognition to any DataFusion application without forking DataFusion core. Register it on a `SessionContext`, and `MATCH_RECOGNIZE` queries plan through `RelationPlanner`, optimize alongside normal DataFusion rules, and run as a regular `ExecutionPlan` with Arrow memory accounting, disk spilling, and optional protobuf serialization for distributed engines.

## Getting started

Enabling `MATCH_RECOGNIZE` takes two steps:

1. Wire the query planner and partition-filter pushdown optimizer onto your `SessionStateBuilder` with `configure_session_state`.
2. Register `MatchRecognizeRelationPlanner` on your `SessionContext` with `register_relation_planner` (or call `session_context_with_match_recognize` to do both at once).

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

Runnable examples live in [`examples/`](./examples):

```sh
cargo run --example funnel
cargo run --example stock_vshape
```

## Benchmarks

The tables below compare `datafusion-match-recognize` against Trino 484 on five 10,000,000-row Parquet workloads (200,000 partitions x 50 rows) running sequentially in identical Linux ARM64 Docker containers, plus Criterion microbenchmarks for adversarial backtracking patterns.

### Single-node Parquet comparison against Trino 484 (10,000,000 rows)

| Workload | Matched output rows | DataFusion p50 | Trino server exec p50 | Trino CPU p50 | Relative (vs Trino exec) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1_funnel_3step_stateless` | 1,000,000 | **0.85 s** (850 ms) | 8.63 s (8,631 ms) | 21.97 s | 10.2x faster |
| `2_funnel_sparse_1pct_fast_skip` | 2,000 | **0.43 s** (427 ms) | 3.30 s (3,300 ms) | 9.48 s | 7.7x faster |
| `3_overlapping_skip_next_all_rows` | 15,092,674 | **2.16 s** (2,161 ms) | 30.89 s (30,892 ms) | 50.81 s | 14.3x faster |
| `4_stock_vshape_stateful_prev` | 3,043,902 | **2.84 s** (2,845 ms) | 18.49 s (18,486 ms) | 38.33 s | 6.5x faster |
| `5_time_window_funnel_stateful` | 880,000 | **2.11 s** (2,113 ms) | 12.04 s (12,038 ms) | 26.04 s | 5.7x faster |
| **Total (all 5 workloads)** | **20,018,576** | **8.40 s** | **73.35 s** | **146.64 s** | **8.7x faster** |

**Conditions.** Both engines ran sequentially (never concurrently) inside identical Linux ARM64 Docker containers (`--cpus=8 --memory=12g`) on the same Colima VM, scanning the exact same 10,000,000-row Parquet files written by Trino's Hive file metastore. DataFusion was pinned to `target_partitions = 8` and 8 Tokio worker threads. Each workload ran 2 warmup iterations (to warm Trino's JIT) followed by 5 measured runs (median p50 reported), wrapped in `SELECT COUNT(*), SUM(...)` so result transfer is 1 row, with exact checksum parity verified across containers on every query. Trino's server-side `elapsedTimeMillis` is reported so HTTP client round-trip time is excluded.

This is a single-node local Parquet snapshot, not a cluster benchmark: it measures Parquet scan, hash repartitioning, sorting, and `MATCH_RECOGNIZE` execution on one node, not S3 latency, distributed network shuffles, or multi-tenant cluster scheduling (for cluster scale, pair this crate with `datafusion-distributed`). Note also that Tier 2 stateful `DEFINE` workloads (`4_stock_vshape_stateful_prev`, `5_time_window_funnel_stateful`) evaluate candidate transitions via DataFusion `PhysicalExpr` on 1-row slot batches rather than compiled scalar bytecode, leaving room for further optimization.

Reproduce the exact containerized Parquet comparison locally:

```sh
./scripts/bench_vs_trino_docker.sh
```

### Adversarial backtracking microbenchmarks (Criterion)

Measured with `cargo bench --bench funnel_bench -- --quick` on `(A+)+ B` against non-matching rows of `A` (confirming linear scaling from Tier 1 per-position `pc` deduplication):

| Benchmark | Rows | Median time | Throughput |
| --- | ---: | ---: | ---: |
| `nfa_adversarial/(A+)+B_no_match` | 1,000 | 18.3 us | ~54.6M rows/s |
| `nfa_adversarial/(A+)+B_no_match` | 10,000 | 192.3 us | ~52.0M rows/s |

## Core tenets of the project

- Stay on DataFusion public extension APIs (`RelationPlanner`, `UserDefinedLogicalNodeCore`, `ExtensionPlanner`, `ExecutionPlan`, `OptimizerRule`, `LogicalExtensionCodec`, `PhysicalExtensionCodec`, and `UserDefinedLogicalNodeUnparser`) with zero core fork.
- Keep simple row-local `DEFINE` patterns fast. Row-local predicates evaluate once per partition in vectorized Arrow into per-row `u64` bitmasks with O(1) start-symbol fast skipping and linear-time PikeVM thread deduplication. Stateful `DEFINE` predicates (`PREV`, `NEXT`, `FIRST`, `LAST`, running aggregates, cross-symbol references) precompute inner expressions once per partition and evaluate lazily on candidate transitions.
- Reuse DataFusion instead of rebuilding it. Partitioning and sorting go through `EnforceDistribution` and `EnforceSorting`, `MEASURES` outer expressions run as standard DataFusion `PhysicalExpr`s with native type coercion, custom aggregates bridge to DataFusion's `Accumulator` trait, and large partitions spill through `SpillManager`.
- Verify against independent oracles, not just hand-written tests. Every commit runs a recursive backtracking reference oracle with `proptest`, multi-batch/multi-partition Arrow invariance tests, disk-spill and memory-pool tests, protobuf round-trips, and a live differential SQL oracle against Trino 484 in CI.

## Verification

```sh
# Offline suites: reference oracle + proptest, SLT suites, Arrow/proto/spill invariance, complexity
cargo test

# Live differential SQL oracle against Trino in Docker
docker run -d --name mr-trino-oracle -p 18080:8080 trinodb/trino:latest
cargo test --test trino_oracle -- --ignored --nocapture
docker rm -f mr-trino-oracle
```

## Documentation and upstream tracking

- Coverage matrix, intentional supersets over Trino, and upstream parser/API blockers: [tests/divergences.md](./tests/divergences.md)
- Contributing guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Agent and repository map: [AGENTS.md](./AGENTS.md)

Upstream SQL syntax PRs tracked for full SQL:2016 parsing in DataFusion:

- Reluctant quantifiers (`*?`, `+?`, `??`, `{n,m}?`): [apache/datafusion-sqlparser-rs#2510](https://github.com/apache/datafusion-sqlparser-rs/pull/2510) (merged in `0.64.0`), DataFusion adoption in [apache/datafusion#27063](https://github.com/apache/datafusion/pull/27063).
- Derived tables, unparenthesized `a {- b -}` exclusions, and lowercase `permute`: [apache/datafusion-sqlparser-rs#3017](https://github.com/apache/datafusion-sqlparser-rs/pull/3017) (draft).
- SQL:2016 `SUBSET` union variables: [apache/datafusion-sqlparser-rs#3018](https://github.com/apache/datafusion-sqlparser-rs/pull/3018) (draft).
- SQL:2016 `RUNNING` and `FINAL` measure modifiers: [apache/datafusion-sqlparser-rs#3019](https://github.com/apache/datafusion-sqlparser-rs/pull/3019) (draft).

## Acknowledgments

This crate builds on work across the Apache DataFusion, Trino, and Apache Calcite communities:

- Geoffrey Claude (`geoffreyclaude`) for designing DataFusion's `RelationPlanner` extension API ([apache/datafusion#18078](https://github.com/apache/datafusion/issues/18078)) and exploring the earlier `MATCH_RECOGNIZE` prototype ([apache/datafusion#16685](https://github.com/apache/datafusion/pull/16685)).
- Andrew Lamb (`alamb`) and the Apache DataFusion maintainers for steering SQL extensions toward modular `datafusion-contrib` crates.
- Trino (`trinodb/trino`) for its SQL:2016 row pattern bytecode compiler (`IrRowPatternToProgramRewriter`), documentation, and `TestRowPatternMatching` test suite, which shaped our NFA compilation and live differential oracle.
- Apache Calcite (`apache/calcite`) for its `match.iq` SQL:2016 test cases.

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).

[ci-badge]: https://github.com/bvolpato/datafusion-match-recognize/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/bvolpato/datafusion-match-recognize/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license-url]: https://github.com/bvolpato/datafusion-match-recognize/blob/main/LICENSE
