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

The tables below compare `datafusion-match-recognize` against both DuckDB 2.1 (`duckdb>=2.0.1.dev0`, native C++ `MATCH_RECOGNIZE`) and Trino 484 on five 10,000,000-row Parquet workloads (200,000 partitions x 50 rows), plus Criterion microbenchmarks for adversarial backtracking patterns.

### Single-node Parquet comparison against DuckDB 2.1 (10,000,000 rows)

| Workload | Matched output rows | DataFusion p50 | DuckDB 2.1 p50 | Relative |
| --- | ---: | ---: | ---: | ---: |
| `1_funnel_3step_stateless` | 1,000,000 | **0.170 s** | 0.571 s | 3.4x faster |
| `2_funnel_sparse_1pct_fast_skip` | 2,000 | **0.154 s** | 0.332 s | 2.2x faster |
| `3_overlapping_skip_next_all_rows` | 15,092,674 | **0.205 s** | 0.920 s | 4.5x faster |
| `4_stock_vshape_stateful_prev` | 3,043,902 | **0.198 s** | 0.812 s | 4.1x faster |
| `5_time_window_funnel_stateful` | 880,000 | **0.182 s** | 0.648 s | 3.6x faster |
| **Total (all 5 workloads)** | **20,018,576** | **0.909 s** | **3.283 s** | **3.6x faster** |

**Conditions.** Both engines ran sequentially (never concurrently) natively on Apple Silicon ARM64 with 8 threads (`SET threads = 8` for DuckDB 2.1; `target_partitions = 8` and 8 Tokio worker threads for DataFusion), scanning the exact same 10,000,000-row Parquet files written by DuckDB (`PER_THREAD_OUTPUT true`). Each workload ran 2 warmup iterations followed by 5 measured runs (median p50 reported), wrapped in `SELECT COUNT(*), SUM(...)` with exact checksum parity verified across engines on every query. (On in-memory tables on the same host, DataFusion totaled 0.640 s vs DuckDB's 2.645 s, 4.1x faster.)

Reproduce the DuckDB 2.1 Parquet comparison locally:

```sh
./scripts/bench_vs_duckdb.sh
```

### DuckDB's own `MATCH_RECOGNIZE` benchmark suite (3-way comparison)

To test on an independent workload suite authored by another engine, we ported all 13 self-contained SQL:2016 benchmarks from DuckDB's `benchmark/micro/match_recognize/*.benchmark` directory (200k to 10M rows, covering sessionization, streaks, funnels, running aggregates in `DEFINE`, bin packing with exclusions, interval merging, 7-variable head-and-shoulders backtracking, cross-variable navigation, and adversarial non-matching patterns) and ran all three engines over the exact same Parquet files:

| DuckDB Benchmark | Rows | DataFusion p50 | DuckDB 2.1 p50 | Trino 484 exec p50 | vs DuckDB | vs Trino |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sessionize_10m` | 10,000,000 | **0.165 s** | 0.437 s | 7.280 s | 2.7x faster | 44.1x faster |
| `streak_10m` | 10,000,000 | **0.169 s** | 0.547 s | 9.465 s | 3.2x faster | 55.9x faster |
| `funnel_three_step_10m` | 10,000,000 | **0.151 s** | 0.281 s | 7.043 s | 1.9x faster | 46.7x faster |
| `funnel_two_step_10m` | 10,000,000 | **0.134 s** | 0.276 s | 5.986 s | 2.1x faster | 44.6x faster |
| `selectivity_mid_10m` | 10,000,000 | **0.257 s** | 0.264 s | 5.149 s | 1.0x | 20.0x faster |
| `aggregate_define_1m` | 1,000,000 | **0.098 s** | 0.284 s | 1.132 s | 2.9x faster | 11.5x faster |
| `bin_packing_1m` | 1,000,000 | **0.020 s** | 0.137 s | 0.634 s | 7.0x faster | 32.5x faster |
| `merge_ranges_1m` | 1,000,000 | **0.108 s** | 0.199 s | 1.313 s | 1.8x faster | 12.1x faster |
| `shoulders_1m` | 1,000,000 | **0.172 s** | 0.374 s | 4.508 s | 2.2x faster | 26.2x faster |
| `navigation_200k` | 200,000 | **0.009 s** | 0.024 s | 0.524 s | 2.6x faster | 57.6x faster |
| `quantifier_500k` | 500,000 | 0.025 s | **0.022 s** | 0.354 s | 0.9x | 14.3x faster |
| `no_match_200k` | 200,000 | **0.004 s** | 0.012 s | >35.000 s (timeout) | 3.1x faster | >9,200x faster |
| `nested_quantifier_2m` | 2,000,000 | **0.040 s** | 0.145 s | >35.000 s (timeout) | 3.6x faster | >868x faster |
| **Total (11 completed by all 3)** | **54,500,000** | **1.309 s** | **2.844 s** | **43.388 s** | **2.2x faster** | **33.2x faster** |
| **Total (all 13, DF vs DuckDB)** | **56,700,000** | **1.353 s** | **3.001 s** | -- | **2.2x faster** | -- |

**Conditions.** DataFusion and DuckDB 2.1 ran sequentially natively on Apple Silicon ARM64 with 8 threads over the same Parquet tables (1 warmup + 3 measured runs, median p50). Trino 484 ran sequentially in an `--cpus=8 --memory=12g` Linux ARM64 Docker container on the same host scanning those exact Parquet tables (server-side `elapsedTimeMillis` p50). Exact 3-way result parity was verified on all 11 queries Trino completed within 35 s, and 2-way exact parity with DuckDB 2.1 on all 13 queries. Running DuckDB's suite directly uncovered and drove three general optimizations in our engine: `required_symbols_mask` whole-partition rejection, lazy Tier 1 cross-attempt dead-state memoization (keeping `no_match_200k` and `nested_quantifier_2m` linear), and `history_head`-keyed O(1) incremental prefix caching for Tier 2 stateful `DEFINE` predicates (driving a 24.5x speedup on `navigation_200k` and a 7.7x speedup on `bin_packing_1m`).

Reproduce DuckDB's benchmark suite locally:

```sh
./scripts/bench_duckdb_suite.sh
```

### Single-node Parquet comparison against Trino 484 (10,000,000 rows)

| Workload | Matched output rows | DataFusion p50 | Trino server exec p50 | Trino CPU p50 | Relative (vs Trino exec) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1_funnel_3step_stateless` | 1,000,000 | **0.188 s** | 6.884 s | 17.15 s | 36.6x faster |
| `2_funnel_sparse_1pct_fast_skip` | 2,000 | **0.165 s** | 1.101 s | 5.33 s | 6.7x faster |
| `3_overlapping_skip_next_all_rows` | 15,092,674 | **0.192 s** | 16.278 s | 37.63 s | 84.8x faster |
| `4_stock_vshape_stateful_prev` | 3,043,902 | **0.172 s** | 9.289 s | 20.51 s | 53.9x faster |
| `5_time_window_funnel_stateful` | 880,000 | **0.170 s** | 5.707 s | 13.34 s | 33.5x faster |
| **Total (all 5 workloads)** | **20,018,576** | **0.888 s** | **39.259 s** | **93.96 s** | **44.2x faster** |

**Conditions.** Both engines ran sequentially (never concurrently) inside identical Linux ARM64 Docker containers (`--cpus=8 --memory=12g`) on the same Colima VM, scanning the exact same 10,000,000-row Parquet files written by Trino's Hive file metastore. DataFusion was pinned to `target_partitions = 8` and 8 Tokio worker threads. Each workload ran 2 warmup iterations (to warm Trino's JIT) followed by 5 measured runs (median p50 reported), wrapped in `SELECT COUNT(*), SUM(...)` so result transfer is 1 row, with exact checksum parity verified across containers on every query. Trino's server-side `elapsedTimeMillis` is reported so HTTP client round-trip time is excluded.

This is a single-node local Parquet snapshot, not a cluster benchmark: it measures Parquet scan, hash repartitioning, sorting, and `MATCH_RECOGNIZE` execution on one node, not S3 latency, distributed network shuffles, or multi-tenant cluster scheduling (for cluster scale, pair this crate with `datafusion-distributed`). Across all five workloads, `MatchRecognizeExec` itself takes less CPU time than DataFusion's upstream `SortExec` thanks to batch-level interior-partition vectorization, zero-allocation flat step pooling, output projection pushdown, Tier 1b promotion for self-anchored `PREV`/`NEXT` predicates, and a compiled scalar evaluator with row-local pre-filters for Tier 2 stateful `DEFINE`s.

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
- Verify against independent oracles, not just hand-written tests. Every commit runs a recursive backtracking reference oracle with `proptest`, multi-batch/multi-partition Arrow invariance tests, disk-spill and memory-pool tests, protobuf round-trips, and live differential SQL oracles against both Trino 484 and DuckDB 2.1 in CI.

## Verification

```sh
# Offline suites: reference oracle + proptest, SLT suites, Arrow/proto/spill invariance, complexity
cargo test

# Live differential SQL oracle against DuckDB 2.1 (native C++ MATCH_RECOGNIZE, requires uv)
cargo test --test duckdb_oracle -- --ignored --nocapture

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

This crate builds on work across the Apache DataFusion, Trino, DuckDB, Apache Calcite, Apache Flink, and RisingWave communities:

- Geoffrey Claude (`geoffreyclaude`) for designing DataFusion's `RelationPlanner` extension API ([apache/datafusion#18078](https://github.com/apache/datafusion/issues/18078)) and exploring the earlier `MATCH_RECOGNIZE` prototype ([apache/datafusion#16685](https://github.com/apache/datafusion/pull/16685)).
- Andrew Lamb (`alamb`) and the Apache DataFusion maintainers for steering SQL extensions toward modular `datafusion-contrib` crates.
- Trino (`trinodb/trino`) for its SQL:2016 row pattern bytecode compiler (`IrRowPatternToProgramRewriter`), documentation, and `TestRowPatternMatching` / `TestAggregationsInRowPatternMatching` / `TestPatternMatchingMachine` test suites, which shaped our NFA compilation and live differential oracle.
- DuckDB (`duckdb/duckdb`) for its native C++ `MATCH_RECOGNIZE` implementation, ISO/IEC 19075-5 `test/sql/match_recognize/*.test` suites, and microbenchmarks, which serve as our second live differential oracle and native benchmark baseline.
- Apache Calcite (`apache/calcite`) for its `match.iq` SQL:2016 test cases.
- Apache Flink (`apache/flink`) for its `MatchRecognizeITCase` batch and streaming SQL test cases.
- RisingWave (`risingwavelabs/risingwave`) for its `e2e_test/streaming/match_recognize*.slt` SQL logic test suites.

## License

Licensed under the [Apache License, Version 2.0](./LICENSE).

[ci-badge]: https://github.com/bvolpato/datafusion-match-recognize/actions/workflows/ci.yml/badge.svg
[ci-url]: https://github.com/bvolpato/datafusion-match-recognize/actions/workflows/ci.yml
[license-badge]: https://img.shields.io/badge/license-Apache%202.0-blue.svg
[license-url]: https://github.com/bvolpato/datafusion-match-recognize/blob/main/LICENSE
