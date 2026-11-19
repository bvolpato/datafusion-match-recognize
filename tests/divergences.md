# Documented Divergences and Upstream Parser Gaps

This file tracks intentional differences from Trino and remaining upstream parser boundaries verified against Trino 484.

## Engine Parity Status

All core SQL:2016 row pattern recognition features are implemented and verified against Trino 484:

- Stateless (`u64` bitmask fast path) and stateful (`PREV`, `NEXT`, `FIRST`, `LAST`, running aggregates, `CLASSIFIER()`, `MATCH_NUMBER()`, cross-symbol references) `DEFINE` predicates.
- `ONE ROW PER MATCH` and `ALL ROWS PER MATCH` (`SHOW EMPTY MATCHES`, `OMIT EMPTY MATCHES`, `WITH UNMATCHED ROWS`).
- `AFTER MATCH SKIP` (`PAST LAST ROW`, `TO NEXT ROW`, `TO FIRST <var>`, `TO LAST <var>`, including `SUBSET` union variables).
- `COUNT(*)`, `COUNT(var.*)`, `COUNT([DISTINCT] expr)`, `SUM([DISTINCT] expr)`, `AVG([DISTINCT] expr)`, `MIN(expr)`, and `MAX(expr)` in both `MEASURES` and stateful `DEFINE`.
- `SUBSET` union variables and `RUNNING` / `FINAL` evaluation semantics in the IR, matcher, measure evaluator, and protobuf codecs.
- Reluctant quantifiers (`*?`, `+?`, `??`, `{n,m}?`) in the IR, bytecode compiler, PikeVM matcher, and protobuf codecs.

## Intentional Supersets Over Trino 484

- **`DISTINCT` inside `MATCH_RECOGNIZE` aggregates:** Trino 484 rejects `COUNT(DISTINCT ...)`, `SUM(DISTINCT ...)`, and `AVG(DISTINCT ...)` in pattern recognition context (`NOT_SUPPORTED`). `datafusion-match-recognize` supports `DISTINCT` on `COUNT`, `SUM`, and `AVG` in both `MEASURES` and stateful `DEFINE` (verified in `tests/slt/stateful_and_distinct.slt` and `tests/invariance.rs`).

## Remaining Upstream `sqlparser 0.63.0` Parser Gaps

DataFusion `57.0.0-dev` currently pins `sqlparser = 0.63.0`, which has four SQL syntax limitations in `TableFactor::MatchRecognize`:

1. **Reluctant quantifiers (`*?`, `+?`, `??`, `{n,m}?`):** Supported in our engine and verified in `tests/oracle.rs`. Already added in `sqlparser 0.64.0` (`RepetitionQuantifier::Reluctant`), so bumping DataFusion's `sqlparser` dependency enables them directly in SQL.
2. **Unparenthesized exclusion after a symbol (`a {- b -}`):** `sqlparser 0.63.0` treats `{` immediately following a pattern term as the start of a `{n,m}` quantifier rather than checking for `{-`. Wrap exclusions in parentheses (`a ({- b -})* c`) until fixed upstream in `apache/datafusion-sqlparser-rs`.
3. **`MATCH_RECOGNIZE` directly after an aliased derived table:** `FROM (VALUES ...) AS t(id, tag) MATCH_RECOGNIZE (...)` fails to parse in `sqlparser 0.63.0`. Use a CTE (`WITH t(id, tag) AS (VALUES ...) SELECT * FROM t MATCH_RECOGNIZE (...)`) until fixed upstream.
4. **SQL syntax for `SUBSET` and explicit `RUNNING` / `FINAL` modifiers:** Supported in our IR, execution engine, and protobuf codecs (verified in `tests/invariance.rs` and `tests/oracle.rs`), awaiting AST support in `apache/datafusion-sqlparser-rs`.
