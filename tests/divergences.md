# SQL:2016 coverage, divergences, and upstream tracking

This file records what the crate implements, where it intentionally goes beyond Trino 484, and which remaining syntax or extension-point gaps live upstream.

## Implemented engine capabilities

Verified against Trino 484 (and the Layer 1 recursive reference oracle):

- Pattern engine supporting named primary variables, implicit-true variables omitted from `DEFINE`, `^` and `$` partition anchors, concatenation, ordered alternation (`|`), groups, greedy and reluctant quantifiers (`*`, `+`, `?`, `{n}`, `{n,}`, `{,m}`, `{n,m}`, `*?`, `+?`, `??`, `{n,m}?`), `{- exclude -}`, and `PERMUTE(...)`.
- Hybrid two-tier `DEFINE` evaluation. Row-local predicates evaluate once per partition in vectorized Arrow into per-row `u64` bitmasks with O(1) `start_symbols_mask` fast skipping, while stateful predicates (`PREV`, `NEXT`, `FIRST`, `LAST`, running aggregates, `CLASSIFIER()`, `MATCH_NUMBER()`, cross-symbol references) precompute inner expressions once per partition and evaluate lazily on candidate transitions with configurable `max_active_threads` limits.
- Output and skip modes covering `ONE ROW PER MATCH`, `ALL ROWS PER MATCH` (`SHOW EMPTY MATCHES`, `OMIT EMPTY MATCHES`, `WITH UNMATCHED ROWS`), and `AFTER MATCH SKIP` (`PAST LAST ROW`, `TO NEXT ROW`, `TO FIRST <var>`, `TO LAST <var>`, including `SUBSET` union variables).
- Measure evaluation for `MATCH_NUMBER()`, `CLASSIFIER([var])` (including `FIRST`/`LAST`/`PREV`/`NEXT` over `CLASSIFIER()` and runtime `CLASSIFIER([var])`/`MATCH_NUMBER()` inside aggregate arguments), `FIRST`/`LAST`/`PREV`/`NEXT` with offsets and nesting, symbol-qualified and `SUBSET`-qualified columns, case-sensitive quoted labels, omitted `MEASURES`, built-in aggregates (`COUNT()`, `COUNT(*)`, `COUNT(var.*)`, `COUNT`, `SUM`, `AVG`, `MIN`, `MAX`), `DISTINCT` aggregates, and arbitrary registered DataFusion `AggregateUDF`s via DataFusion's `Accumulator` trait (`ARRAY_AGG`, `STRING_AGG`, `STDDEV_SAMP`, `STDDEV_POP`, `BOOL_AND`, `BOOL_OR`, and custom UDAFs) with `fields_with_udf` type coercion.
- Memory accounting and disk spilling. Partition buffers register against `TaskContext::memory_pool()`, spill automatically to Arrow IPC temp files through DataFusion's `SpillManager` (`SpillMetrics`) when `DiskManager` is enabled, and return clean `ResourcesExhausted` errors when disk spilling is disabled.
- Optimizer, serialization, and unparsing support via `MatchRecognizeFilterPushdown`, `necessary_children_exprs` column pruning, composable `MatchRecognizeLogicalCodec` and `MatchRecognizePhysicalCodec` (`with_delegate`), and `MatchRecognizeUnparser` (`UserDefinedLogicalNodeUnparser::unparse_to_statement` and `match_recognize_to_table_factor`).

## Intentional supersets over Trino 484

1. `DISTINCT` inside `MATCH_RECOGNIZE` aggregates. Trino 484 rejects `COUNT(DISTINCT ...)`, `SUM(DISTINCT ...)`, and `AVG(DISTINCT ...)` in pattern recognition context (`NOT_SUPPORTED`). This crate supports `DISTINCT` across `COUNT`, `SUM`, `AVG`, and single-argument UDAFs (verified in `tests/slt/stateful_and_distinct.slt` and `tests/invariance.rs`).
2. Physical navigation inside aggregates (`SUM(PREV(b.price))`, `SUM(NEXT(b.price))`). Trino 484 rejects nesting `PREV` or `NEXT` inside aggregates (`INVALID_NAVIGATION_NESTING`). This crate supports shifting aggregate input rows by `PREV` or `NEXT` (verified in `tests/slt/edge_cases.slt` and `tests/invariance.rs`).

## Upstream parser and extension-point blockers

Every remaining gap lives outside this crate and is tracked by an upstream PR or explicit DataFusion API boundary:

1. Reluctant quantifiers (`*?`, `+?`, `??`, `{n,m}?`) in SQL are implemented in the engine (verified in `tests/oracle.rs`), merged in [apache/datafusion-sqlparser-rs#2510](https://github.com/apache/datafusion-sqlparser-rs/pull/2510) (`0.64.0`), and tracked for DataFusion adoption in [apache/datafusion#27063](https://github.com/apache/datafusion/pull/27063).
2. Derived tables (`FROM (SELECT/VALUES ...) AS t MATCH_RECOGNIZE`), unparenthesized exclusions (`a {- b -}* c`), and lowercase `permute(...)` are tracked in draft PR [apache/datafusion-sqlparser-rs#3017](https://github.com/apache/datafusion-sqlparser-rs/pull/3017). Until adopted by DataFusion, wrap exclusions in parentheses (`a ({- b -})* c`) and use a CTE (`WITH t AS (...) SELECT * FROM t MATCH_RECOGNIZE (...)`).
3. SQL syntax for `SUBSET u = (a, b)` union variables is implemented in our IR, engine, and protobuf codecs (verified in `tests/invariance.rs` and `tests/oracle.rs`) and tracked in draft PR [apache/datafusion-sqlparser-rs#3018](https://github.com/apache/datafusion-sqlparser-rs/pull/3018).
4. SQL syntax for explicit `RUNNING` and `FINAL` measure modifiers is implemented in our IR (`EvalSemantics`), evaluator, and protobuf codecs (verified in `tests/invariance.rs`) and tracked in draft PR [apache/datafusion-sqlparser-rs#3019](https://github.com/apache/datafusion-sqlparser-rs/pull/3019).
5. Inline `RelationBuilder` unparsing inside outer `SELECT` queries. `MatchRecognizeUnparser::unparse_to_statement` unparses standalone `MatchRecognizeNode` plans back to SQL, but DataFusion's `RelationBuilder` (`datafusion/sql/src/unparser/ast.rs`) wraps a private `TableFactorBuilder` enum with no custom `TableFactor` variant. Unparsing a `MatchRecognizeNode` nested inside an outer `SELECT` requires adding a `RelationBuilder::table_factor(ast::TableFactor)` method upstream in `apache/datafusion`.
6. Row pattern recognition inside window clauses (`OVER (... MEASURES ... PATTERN ...)`) is blocked upstream because `apache/datafusion-sqlparser-rs` does not parse RPR inside `WindowSpec` and DataFusion's `RelationPlanner` only intercepts `TableFactor` relations (no `WindowPlanner` extension API exists yet).
