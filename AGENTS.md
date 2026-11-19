# DataFusion Match Recognize agent guide

## Project purpose

`datafusion-match-recognize` is a Rust library that adds SQL:2016 `MATCH_RECOGNIZE` (Row Pattern Recognition) to Apache DataFusion using only public extension APIs. Keep it decoupled from DataFusion internals so downstream engines (including `datafusion-distributed`) can embed it without carrying a fork.

## Repository map

- `proto/match_recognize.proto`: Protobuf contract for logical and physical plan serialization (compiled by `build.rs` behind the default `proto` feature).
- `src/`: Library implementation.
  - `ast.rs`: Validated internal IR for patterns, quantifiers, `SUBSET`, skip modes, output modes, measure slots, and `DEFINE` definitions.
  - `pattern/mod.rs`: Pattern AST to priority-ordered NFA bytecode compiler (`Program`, `Inst`, `SymbolSet`, and O(1) `start_symbols_mask`).
  - `matcher/mod.rs`: Hybrid two-tier PikeVM matcher (Tier 1 stateless O(N * |states|) deduplication + fast skip, Tier 2 stateful per-thread epsilon-cycle detection + thread cap, flat arena history).
  - `planner.rs`: `MatchRecognizeRelationPlanner` (`RelationPlanner`), `DEFINE` tier classifier, `MEASURES` slot extraction, `CLASSIFIER()` navigation, and `AggregateUDF` resolution.
  - `logical.rs`: `MatchRecognizeNode` (`UserDefinedLogicalNodeCore`), SQL:2016 output schema computation, `TypeCoercionRewriter` on slot schemas, and `necessary_children_exprs` column pruning.
  - `optimizer.rs`: `MatchRecognizeFilterPushdown` pushing partition-only predicates below `MatchRecognizeNode`.
  - `measures.rs`: Slot evaluation for `MEASURES`, `PartitionStatefulEvaluator` for Tier 2 stateful `DEFINE`, `DISTINCT` aggregates, and `Accumulator` bridge for arbitrary DataFusion `AggregateUDF`s.
  - `physical.rs`: `MatchRecognizeExec` (`ExecutionPlan`), `MatchRecognizeExtensionPlanner`, `MatchRecognizeQueryPlanner`, partition-streaming state machine, and `SpillManager` disk spilling.
  - `proto.rs`: Composable `MatchRecognizeLogicalCodec` and `MatchRecognizePhysicalCodec` (`with_delegate`).
  - `unparser.rs`: `MatchRecognizeUnparser` (`UserDefinedLogicalNodeUnparser`) and `match_recognize_to_table_factor`.
- `examples/`: Runnable usage examples (`funnel.rs`, `stock_vshape.rs`).
- `tests/`: 4-layer verification suite.
  - `oracle.rs`: Layer 1 independent recursive backtracking oracle + `proptest` fuzzer (stateless and stateful).
  - `sql_tests.rs` and `tests/slt/`: Layer 2a SQL Logic Tests (Trino, Calcite, funnel, stateful, edge cases).
  - `trino_oracle.rs`: Layer 2b live differential SQL oracle against Trino 484 in Docker (hand cases + combinatorial fuzz + PRNG random queries).
  - `invariance.rs`: Layer 3 batch-size, partition-count, Arrow slicing/types, optimizer, memory-pool, disk-spill, protobuf round-trip, and unparser tests.
  - `complexity.rs`: Layer 4 adversarial anti-backtracking linear-time tests.
  - `divergences.md`: Documented parity matrix, supersets over Trino, and upstream tracking PRs.
- `benches/funnel_bench.rs`: Criterion throughput and anti-backtracking benchmarks.

## Key invariants to preserve

1. **Public extension APIs only:** Do not rely on private DataFusion internals.
2. **Tier 1 linear time:** Stateless `DEFINE` matching must keep per-position `visited[pc]` deduplication and pass `tests/complexity.rs`.
3. **Flat arena history:** Do not reintroduce recursive heap pointers (`Rc`/`Box` chains) for match history (50k-row matches will overflow the stack on `Drop`).
4. **Proto parity:** Every new field in `MatchRecognizeNode` or `MatchRecognizeExec` must round-trip through `MatchRecognizeLogicalCodec` and `MatchRecognizePhysicalCodec`.
5. **Oracle verification:** Any NFA or SQL semantic change must pass `cargo test` and `cargo test --test trino_oracle -- --ignored --nocapture` against Trino 484.
