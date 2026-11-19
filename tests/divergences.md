# Documented Divergences and Upstream Parser Gaps

This file tracks intentional differences from Trino and current Phase 1 scope boundaries verified against Trino 484.

## Upstream `sqlparser 0.63.0` Parser Gaps

DataFusion `57.0.0-dev` currently pins `sqlparser = 0.63.0`, which has four parser limitations in `TableFactor::MatchRecognize`:

1. **Reluctant quantifiers (`*?`, `+?`, `??`, `{n,m}?`):** The PikeVM engine and IR in `datafusion-match-recognize` support reluctant quantifiers (verified in `tests/oracle.rs`), and `sqlparser 0.64.0` adds `RepetitionQuantifier::Reluctant`. Bumping `sqlparser` in DataFusion enables SQL parsing of reluctant quantifiers.
2. **Unparenthesized exclusion after a symbol (`a {- b -}`):** `sqlparser 0.63.0` treats `{` immediately following a pattern term as the start of a `{n,m}` quantifier rather than checking for `{-`. Wrap exclusions in parentheses (`a ({- b -})* c`) until fixed upstream in `apache/datafusion-sqlparser-rs`.
3. **`MATCH_RECOGNIZE` directly after an aliased derived table:** `FROM (VALUES ...) AS t(id, tag) MATCH_RECOGNIZE (...)` fails to parse in `sqlparser 0.63.0` because `AS t(...)` is parsed before checking for `MATCH_RECOGNIZE`. Use a CTE (`WITH t(id, tag) AS (VALUES ...) SELECT * FROM t MATCH_RECOGNIZE (...)`) until fixed upstream.
4. **`SUBSET` and `RUNNING` / `FINAL` modifiers:** Not yet present in `sqlparser 0.63.0` AST. Planned for Phase 2 alongside an upstream `apache/datafusion-sqlparser-rs` PR.

## Phase 1 Scope Boundaries (Planned for Phase 2)

1. **Stateful `DEFINE` predicates:** Cross-symbol references (`DEFINE b AS b.price > PREV(b.price)`) and navigation/aggregate functions inside `DEFINE` currently return a clear planning error. Phase 1 evaluates row-local causal `DEFINE` predicates in vectorized Arrow.
2. **`SUBSET` union variables:** Planned for Phase 2 once parsed.
3. **Explicit `RUNNING` / `FINAL` measure modifiers:** Phase 1 applies SQL:2016 defaults (`FINAL` for `ONE ROW PER MATCH`, `RUNNING` for `ALL ROWS PER MATCH`). The internal `EvalSemantics` IR and evaluator already support both modes.
4. **`DISTINCT` inside measure aggregates:** `COUNT(DISTINCT a.col)` is not yet supported in Phase 1.
