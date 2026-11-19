// SPDX-License-Identifier: Apache-2.0

//! Layer 3 verification: DataFusion & Arrow invariance matrix (batch sizes, partition counts,
//! sliced Arrow arrays, diverse Arrow types, optimizer equivalence, memory pool enforcement,
//! and protobuf logical/physical codec round-trip).

use std::sync::Arc;

use arrow::array::{
    ArrayRef, Decimal128Array, DictionaryArray, Float64Array, Int32Array, Int64Array,
    LargeStringArray, RecordBatch, StringArray, StringViewArray, TimestampNanosecondArray,
};
use arrow::datatypes::Int32Type;
use arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::memory::MemTable;
use datafusion::execution::SessionStateBuilder;
use datafusion::execution::memory_pool::GreedyMemoryPool;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::collect;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::DataFusionError;
use datafusion_match_recognize::{
    MatchRecognizeConfig, MatchRecognizeFilterPushdown, MatchRecognizeLogicalCodec,
    MatchRecognizePhysicalCodec, MatchRecognizeQueryPlanner, register_relation_planner,
    session_context_with_match_recognize,
};
use datafusion_proto::bytes::{
    logical_plan_from_bytes_with_extension_codec, logical_plan_to_bytes_with_extension_codec,
    physical_plan_from_bytes_with_extension_codec, physical_plan_to_bytes_with_extension_codec,
};

fn make_multi_partition_events() -> RecordBatch {
    // 3 partitions (u1, u2, u3) interleaved out of order to exercise RepartitionExec + SortExec.
    RecordBatch::try_from_iter(vec![
        (
            "user_id",
            Arc::new(StringArray::from(vec![
                "u2", "u1", "u3", "u1", "u2", "u1", "u3", "u2", "u1", "u3", "u2", "u1",
            ])) as ArrayRef,
        ),
        (
            "ts",
            Arc::new(Int64Array::from(vec![
                10, 1, 100, 2, 20, 3, 200, 30, 4, 300, 40, 5,
            ])),
        ),
        (
            "event",
            Arc::new(StringArray::from(vec![
                "view", "view", "view", "click", "click", "buy", "other", "buy", "view", "click",
                "view", "buy",
            ])),
        ),
        (
            "amount",
            Arc::new(Float64Array::from(vec![
                Some(1.0),
                Some(10.0),
                Some(5.0),
                None,
                Some(2.0),
                Some(50.0),
                Some(3.0),
                Some(20.0),
                Some(7.0),
                Some(4.0),
                Some(8.0),
                Some(15.0),
            ])),
        ),
    ])
    .unwrap()
}

async fn run_and_format(ctx: &SessionContext, sql: &str) -> String {
    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    pretty_format_batches(&batches).unwrap().to_string()
}

#[tokio::test]
async fn test_batch_size_and_partition_count_invariance() {
    let base_batch = make_multi_partition_events();
    let queries = [
        // ONE ROW PER MATCH with aggregates, navigation, and CASE
        r#"
            SELECT * FROM events
            MATCH_RECOGNIZE (
                PARTITION BY user_id
                ORDER BY ts
                MEASURES
                    MATCH_NUMBER() AS mn,
                    FIRST(a.ts) AS start_ts,
                    LAST(c.ts) AS end_ts,
                    COUNT(*) AS total_rows,
                    COUNT(b.*) AS click_rows,
                    SUM(amount) AS total_amt,
                    AVG(amount) AS avg_amt,
                    CASE WHEN c.ts IS NOT NULL THEN FIRST(c.ts) - FIRST(a.ts) ELSE NULL END AS dt
                ONE ROW PER MATCH
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN ((a b* c) | (a b+))
                DEFINE
                    a AS event = 'view',
                    b AS event = 'click',
                    c AS event = 'buy'
            )
            ORDER BY user_id, mn
        "#,
        // ALL ROWS PER MATCH WITH UNMATCHED ROWS + RUNNING measures
        r#"
            SELECT * FROM events
            MATCH_RECOGNIZE (
                PARTITION BY user_id
                ORDER BY ts
                MEASURES
                    MATCH_NUMBER() AS mn,
                    CLASSIFIER() AS cls,
                    FIRST(a.ts) AS first_a_ts,
                    PREV(ts, 1) AS prev_ts,
                    NEXT(ts, 1) AS next_ts,
                    COUNT(*) AS running_cnt
                ALL ROWS PER MATCH WITH UNMATCHED ROWS
                AFTER MATCH SKIP PAST LAST ROW
                PATTERN (a b* c)
                DEFINE
                    a AS event = 'view',
                    b AS event = 'click',
                    c AS event = 'buy'
            )
            ORDER BY user_id, ts
        "#,
        // ALL ROWS PER MATCH with exclusion {- b -} and SKIP TO NEXT ROW
        r#"
            SELECT * FROM events
            MATCH_RECOGNIZE (
                PARTITION BY user_id
                ORDER BY ts
                MEASURES
                    MATCH_NUMBER() AS mn,
                    CLASSIFIER() AS cls,
                    COUNT(*) AS running_cnt
                ALL ROWS PER MATCH OMIT EMPTY MATCHES
                AFTER MATCH SKIP TO NEXT ROW
                PATTERN (a ({- b -})* c)
                DEFINE
                    a AS event = 'view',
                    b AS event = 'click',
                    c AS event = 'buy'
            )
            ORDER BY user_id, mn, ts
        "#,
    ];

    for sql in queries {
        let mut baseline: Option<String> = None;
        for &batch_size in &[1usize, 2, 3, 7, 8192] {
            for &target_partitions in &[1usize, 4] {
                let cfg = SessionConfig::new()
                    .with_batch_size(batch_size)
                    .with_target_partitions(target_partitions);
                let ctx =
                    session_context_with_match_recognize(cfg, MatchRecognizeConfig::default())
                        .unwrap();

                // Split input across multiple small batches to force multi-batch partitions.
                let mut chunks = Vec::new();
                let mut offset = 0;
                while offset < base_batch.num_rows() {
                    let len = batch_size.min(base_batch.num_rows() - offset);
                    chunks.push(base_batch.slice(offset, len));
                    offset += len;
                }
                let table = MemTable::try_new(base_batch.schema(), vec![chunks]).unwrap();
                ctx.register_table("events", Arc::new(table)).unwrap();

                let out = run_and_format(&ctx, sql).await;
                match &baseline {
                    None => baseline = Some(out),
                    Some(expected) => {
                        assert_eq!(
                            &out, expected,
                            "Invariance failed for batch_size={batch_size}, target_partitions={target_partitions}"
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn test_sliced_arrow_batches_and_diverse_types() {
    // Construct arrays and slice with non-zero offset (offset = 2) to test Arrow offset safety.
    let dict_vals: DictionaryArray<Int32Type> =
        vec!["pad", "pad", "p1", "p1", "p1", "p2", "p2", "pad"]
            .into_iter()
            .collect();
    let full = RecordBatch::try_from_iter(vec![
        ("part", Arc::new(dict_vals) as ArrayRef),
        (
            "ts",
            Arc::new(TimestampNanosecondArray::from(vec![
                0, 1, 100, 200, 300, 400, 500, 600,
            ])),
        ),
        (
            "ev_view",
            Arc::new(StringViewArray::from(vec![
                "x", "x", "a", "b", "c", "a", "c", "x",
            ])),
        ),
        (
            "ev_large",
            Arc::new(LargeStringArray::from(vec![
                "x", "x", "L1", "L2", "L3", "L4", "L5", "x",
            ])),
        ),
        (
            "dec",
            Arc::new(
                Decimal128Array::from(vec![
                    Some(0),
                    Some(0),
                    Some(100),
                    Some(250),
                    Some(350),
                    Some(500),
                    Some(100),
                    Some(0),
                ])
                .with_precision_and_scale(10, 2)
                .unwrap(),
            ),
        ),
        (
            "i32_col",
            Arc::new(Int32Array::from(vec![0, 0, 1, 2, 3, 4, 5, 0])),
        ),
    ])
    .unwrap();

    let sliced = full.slice(2, 5);
    let ctx = session_context_with_match_recognize(
        SessionConfig::new().with_batch_size(2),
        MatchRecognizeConfig::default(),
    )
    .unwrap();
    ctx.register_batch("t", sliced).unwrap();

    let sql = r#"
        SELECT * FROM t
        MATCH_RECOGNIZE (
            PARTITION BY part
            ORDER BY ts
            MEASURES
                FIRST(a.ev_large) AS first_l,
                LAST(c.ev_large) AS last_l,
                SUM(dec) AS sum_dec,
                MIN(i32_col) AS min_i32,
                MAX(i32_col) AS max_i32
            ONE ROW PER MATCH
            PATTERN (a b* c)
            DEFINE
                a AS ev_view = 'a',
                b AS ev_view = 'b',
                c AS ev_view = 'c'
        )
        ORDER BY part
    "#;

    let batches = ctx.sql(sql).await.unwrap().collect().await.unwrap();
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "expected 2 matches (one for p1, one for p2)");
}

#[tokio::test]
async fn test_optimizer_filter_pushdown_and_projection_pruning() {
    let batch = make_multi_partition_events();
    let sql = r#"
        SELECT user_id, mn, total_amt FROM events
        MATCH_RECOGNIZE (
            PARTITION BY user_id
            ORDER BY ts
            MEASURES
                MATCH_NUMBER() AS mn,
                SUM(amount) AS total_amt
            ONE ROW PER MATCH
            PATTERN (a b* c)
            DEFINE
                a AS event = 'view',
                b AS event = 'click',
                c AS event = 'buy'
        )
        WHERE user_id = 'u1'
        ORDER BY mn
    "#;

    // With MatchRecognizeFilterPushdown enabled
    let ctx_opt =
        session_context_with_match_recognize(SessionConfig::new(), MatchRecognizeConfig::default())
            .unwrap();
    ctx_opt.register_batch("events", batch.clone()).unwrap();

    let df_opt = ctx_opt.sql(sql).await.unwrap();
    let optimized_plan = df_opt.clone().into_optimized_plan().unwrap();
    let explain = optimized_plan.display_indent().to_string();
    // Filter on partition key should be pushed BELOW MatchRecognize.
    let filter_pos = explain.find("Filter:").expect("Filter present in plan");
    let mr_pos = explain
        .find("MatchRecognize:")
        .expect("MatchRecognize present in plan");
    assert!(
        filter_pos > mr_pos,
        "Expected partition Filter below MatchRecognize in explain:\n{explain}"
    );
    let out_opt = pretty_format_batches(&df_opt.collect().await.unwrap())
        .unwrap()
        .to_string();

    // Without MatchRecognizeFilterPushdown
    let state_no_opt = SessionStateBuilder::new()
        .with_default_features()
        .with_query_planner(Arc::new(MatchRecognizeQueryPlanner::new()))
        .build();
    let ctx_no_opt = SessionContext::new_with_state(state_no_opt);
    register_relation_planner(&ctx_no_opt, MatchRecognizeConfig::default()).unwrap();
    ctx_no_opt.register_batch("events", batch).unwrap();
    let out_no_opt = run_and_format(&ctx_no_opt, sql).await;

    assert_eq!(out_opt, out_no_opt);
}

#[tokio::test]
async fn test_memory_pool_enforcement_when_disk_disabled() {
    use datafusion::execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};

    // Create a 50,000-row single partition under a 64 KB GreedyMemoryPool with disk spilling
    // explicitly disabled -> must return clean ResourcesExhausted error.
    let n = 50_000usize;
    let batch = RecordBatch::try_from_iter(vec![
        ("k", Arc::new(Int64Array::from(vec![1i64; n])) as ArrayRef),
        ("ts", Arc::new(Int64Array::from_iter_values(0..n as i64))),
        (
            "v",
            Arc::new(StringArray::from(vec!["payload_string_value"; n])),
        ),
    ])
    .unwrap();

    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(64 * 1024)))
        .with_disk_manager_builder(
            DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
        )
        .build_arc()
        .unwrap();
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_runtime_env(runtime)
        .with_query_planner(Arc::new(MatchRecognizeQueryPlanner::new()))
        .with_optimizer_rule(Arc::new(MatchRecognizeFilterPushdown))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_relation_planner(&ctx, MatchRecognizeConfig::default()).unwrap();
    ctx.register_batch("big", batch).unwrap();

    let sql = r#"
        SELECT * FROM big
        MATCH_RECOGNIZE (
            PARTITION BY k
            ORDER BY ts
            MEASURES COUNT(*) AS c
            ONE ROW PER MATCH
            PATTERN (a+)
            DEFINE a AS ts >= 0
        )
    "#;
    let res = ctx.sql(sql).await.unwrap().collect().await;
    match res {
        Err(DataFusionError::ResourcesExhausted(_)) => {}
        Err(other) if other.to_string().contains("Resources exhausted") => {}
        other => panic!("Expected ResourcesExhausted error under 64KB memory pool, got: {other:?}"),
    }
}

#[tokio::test]
async fn test_disk_spill_large_partition() {
    use datafusion::execution::memory_pool::FairSpillPool;
    use datafusion::physical_plan::ExecutionPlan;

    // 20,000 rows split across small batches under a 64 KB FairSpillPool with disk spilling
    // enabled: MatchRecognizeExec must spill buffered batches to disk and still produce exact
    // output with spill_count > 0.
    let n = 20_000usize;
    let full = RecordBatch::try_from_iter(vec![
        ("k", Arc::new(Int64Array::from(vec![1i64; n])) as ArrayRef),
        ("ts", Arc::new(Int64Array::from_iter_values(0..n as i64))),
        (
            "v",
            Arc::new(StringArray::from(vec!["payload_string_value_for_spill"; n])),
        ),
    ])
    .unwrap();
    let mut chunks = Vec::new();
    let mut off = 0;
    while off < n {
        let len = 1000.min(n - off);
        chunks.push(full.slice(off, len));
        off += len;
    }

    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(FairSpillPool::new(64 * 1024)))
        .build_arc()
        .unwrap();
    let state = SessionStateBuilder::new()
        .with_default_features()
        .with_config(
            SessionConfig::new()
                .with_batch_size(1000)
                .with_target_partitions(1),
        )
        .with_runtime_env(runtime)
        .with_query_planner(Arc::new(MatchRecognizeQueryPlanner::new()))
        .build();
    let ctx = SessionContext::new_with_state(state);
    register_relation_planner(&ctx, MatchRecognizeConfig::default()).unwrap();
    let table = MemTable::try_new(full.schema(), vec![chunks]).unwrap();
    ctx.register_table("big_spill", Arc::new(table)).unwrap();

    // No ORDER BY / PARTITION BY so SortExec doesn't reject the tight pool before MatchRecognizeExec.
    let sql = r#"
        SELECT * FROM big_spill
        MATCH_RECOGNIZE (
            MEASURES COUNT(*) AS c
            ONE ROW PER MATCH
            PATTERN (a+)
            DEFINE a AS ts >= 0
        )
    "#;
    let df = ctx.sql(sql).await.unwrap();
    let plan = df.create_physical_plan().await.unwrap();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let formatted = pretty_format_batches(&batches).unwrap().to_string();
    assert!(
        formatted.contains("20000"),
        "expected count=20000, got:\n{formatted}"
    );

    fn find_mr_metrics(plan: &Arc<dyn ExecutionPlan>) -> usize {
        if plan.name() == "MatchRecognizeExec" {
            return plan.metrics().and_then(|m| m.spill_count()).unwrap_or(0);
        }
        plan.children().iter().map(|c| find_mr_metrics(c)).sum()
    }
    let spills = find_mr_metrics(&plan);
    assert!(
        spills > 0,
        "expected MatchRecognizeExec spill_count > 0, got {spills}"
    );
}

#[tokio::test]
async fn test_logical_and_physical_proto_roundtrip() {
    let ctx = session_context_with_match_recognize(
        SessionConfig::new().with_target_partitions(2),
        MatchRecognizeConfig::default(),
    )
    .unwrap();

    // Use inline VALUES so logical plan serialization exercises MatchRecognizeLogicalCodec
    // end-to-end without requiring a custom MemTable TableProvider serializer.
    let sql = r#"
        WITH events(user_id, ts, event, amount) AS (
            VALUES
                ('u2', 10, 'view',  1.0),
                ('u1',  1, 'view', 10.0),
                ('u3', 100, 'view', 5.0),
                ('u1',  2, 'click', NULL),
                ('u2', 20, 'click', 2.0),
                ('u1',  3, 'buy',  50.0),
                ('u3', 200, 'other', 3.0),
                ('u2', 30, 'buy',  20.0),
                ('u1',  4, 'view',  7.0),
                ('u3', 300, 'click', 4.0),
                ('u2', 40, 'view',  8.0),
                ('u1',  5, 'buy',  15.0)
        )
        SELECT * FROM events
        MATCH_RECOGNIZE (
            PARTITION BY user_id
            ORDER BY ts
            MEASURES
                MATCH_NUMBER() AS mn,
                CLASSIFIER() AS cls,
                FIRST(a.ts) AS a_ts,
                LAST(c.ts) AS c_ts,
                PREV(ts, 1) AS prev_ts,
                COUNT(*) AS cnt,
                SUM(amount) AS sum_amt,
                CASE WHEN c.ts IS NOT NULL THEN FIRST(c.ts) - FIRST(a.ts) ELSE -1 END AS dt
            ALL ROWS PER MATCH WITH UNMATCHED ROWS
            AFTER MATCH SKIP PAST LAST ROW
            PATTERN ((a b* c) | (a b+))
            DEFINE
                a AS event = 'view',
                b AS event = 'click',
                c AS event = 'buy'
        )
        ORDER BY user_id, ts
    "#;

    let df = ctx.sql(sql).await.unwrap();
    let expected = pretty_format_batches(&df.clone().collect().await.unwrap())
        .unwrap()
        .to_string();

    // 1. Logical plan protobuf round-trip
    let logical_plan = df.clone().into_optimized_plan().unwrap();
    let logical_codec = MatchRecognizeLogicalCodec::new();
    let logical_bytes =
        logical_plan_to_bytes_with_extension_codec(&logical_plan, &logical_codec).unwrap();
    let decoded_logical = logical_plan_from_bytes_with_extension_codec(
        &logical_bytes,
        &ctx.task_ctx(),
        &logical_codec,
    )
    .unwrap();
    let logical_out = pretty_format_batches(
        &ctx.execute_logical_plan(decoded_logical)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();
    assert_eq!(
        logical_out, expected,
        "Logical proto round-trip output mismatch"
    );

    // 2. Physical plan protobuf round-trip
    let physical_plan = df.create_physical_plan().await.unwrap();
    let phys_codec = MatchRecognizePhysicalCodec::new();
    let phys_bytes =
        physical_plan_to_bytes_with_extension_codec(Arc::clone(&physical_plan), &phys_codec)
            .unwrap();
    let decoded_phys =
        physical_plan_from_bytes_with_extension_codec(&phys_bytes, &ctx.task_ctx(), &phys_codec)
            .unwrap();
    let phys_batches = collect(decoded_phys, ctx.task_ctx()).await.unwrap();
    let phys_out = pretty_format_batches(&phys_batches).unwrap().to_string();
    assert_eq!(
        phys_out, expected,
        "Physical proto round-trip output mismatch"
    );
}

#[tokio::test]
async fn test_stateful_define_and_distinct_proto_roundtrip() {
    let ctx = session_context_with_match_recognize(
        SessionConfig::new()
            .with_batch_size(2)
            .with_target_partitions(2),
        MatchRecognizeConfig::default(),
    )
    .unwrap();

    let sql = r#"
        WITH ticker(sym, t, price) AS (
            VALUES
                ('ACME', 1, 100),
                ('ACME', 2,  90),
                ('ACME', 3,  80),
                ('ACME', 4,  85),
                ('ACME', 5,  95),
                ('ACME', 6,  95),
                ('ACME', 7,  70),
                ('ACME', 8, 110)
        )
        SELECT * FROM ticker
        MATCH_RECOGNIZE (
            PARTITION BY sym
            ORDER BY t
            MEASURES
                MATCH_NUMBER() AS mn,
                FIRST(strt.price) AS start_p,
                LAST(down.price) AS bottom_p,
                LAST(up.price) AS peak_p,
                COUNT(DISTINCT price) AS distinct_prices,
                SUM(DISTINCT price) AS sum_distinct_prices
            ONE ROW PER MATCH
            AFTER MATCH SKIP TO LAST up
            PATTERN (strt down+ up+)
            DEFINE
                down AS down.price < PREV(down.price),
                up AS up.price > PREV(up.price)
        )
        ORDER BY sym, mn
    "#;

    let df = ctx.sql(sql).await.unwrap();
    let expected = pretty_format_batches(&df.clone().collect().await.unwrap())
        .unwrap()
        .to_string();

    let logical_plan = df.clone().into_optimized_plan().unwrap();
    let logical_codec = MatchRecognizeLogicalCodec::with_delegate(Arc::new(
        datafusion_proto::logical_plan::DefaultLogicalExtensionCodec {},
    ));
    let logical_bytes =
        logical_plan_to_bytes_with_extension_codec(&logical_plan, &logical_codec).unwrap();
    let decoded_logical = logical_plan_from_bytes_with_extension_codec(
        &logical_bytes,
        &ctx.task_ctx(),
        &logical_codec,
    )
    .unwrap();
    let logical_out = pretty_format_batches(
        &ctx.execute_logical_plan(decoded_logical)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();
    assert_eq!(logical_out, expected);

    let physical_plan = df.create_physical_plan().await.unwrap();
    let phys_codec = MatchRecognizePhysicalCodec::with_delegate(Arc::new(
        datafusion_proto::physical_plan::DefaultPhysicalExtensionCodec {},
    ));
    let phys_bytes =
        physical_plan_to_bytes_with_extension_codec(Arc::clone(&physical_plan), &phys_codec)
            .unwrap();
    let decoded_phys =
        physical_plan_from_bytes_with_extension_codec(&phys_bytes, &ctx.task_ctx(), &phys_codec)
            .unwrap();
    let phys_batches = collect(decoded_phys, ctx.task_ctx()).await.unwrap();
    let phys_out = pretty_format_batches(&phys_batches).unwrap().to_string();
    assert_eq!(phys_out, expected);
}

#[tokio::test]
async fn test_subset_union_variables_and_final_semantics_roundtrip() {
    use datafusion_expr::{col, lit};
    use datafusion_match_recognize::{
        AfterMatchSkip, AggKind, EvalSemantics, LogicalNavKind, MatchRecognizeNode, MeasureSlot,
        MeasureSlotKind, MeasureSpec, OrderBySpec, Pattern, RowsPerMatch, SubsetDef, SymbolDef,
        SymbolRef,
    };

    let ctx = session_context_with_match_recognize(
        SessionConfig::new()
            .with_batch_size(2)
            .with_target_partitions(2),
        MatchRecognizeConfig::default(),
    )
    .unwrap();

    let base_plan = ctx
        .sql(
            "SELECT column1 AS id, column2 AS tag, column3 AS val FROM (VALUES \
             (1, 'A', 10), (2, 'B', 20), (3, 'C', 30), (4, 'B', 40))",
        )
        .await
        .unwrap()
        .into_unoptimized_plan();

    let pattern = Pattern::Concat(vec![
        Pattern::Symbol(SymbolRef::Named("a".into())),
        Pattern::Repetition {
            inner: Box::new(Pattern::Symbol(SymbolRef::Named("b".into()))),
            min: 1,
            max: None,
            reluctant: false,
        },
        Pattern::Repetition {
            inner: Box::new(Pattern::Symbol(SymbolRef::Named("c".into()))),
            min: 1,
            max: None,
            reluctant: false,
        },
    ]);
    let subsets = vec![SubsetDef {
        name: "u".into(),
        members: vec!["b".into(), "c".into()],
    }];
    let definitions = vec![
        SymbolDef::stateless("a", col("tag").eq(lit("A"))),
        SymbolDef::stateless("b", col("tag").eq(lit("B"))),
        SymbolDef::stateless("c", col("tag").eq(lit("C"))),
    ];
    let measures = vec![
        MeasureSpec {
            alias: "u_first".into(),
            outer_expr: col("__mr_slot_0"),
            slots: vec![MeasureSlot {
                slot_name: "__mr_slot_0".into(),
                semantics: EvalSemantics::Final,
                kind: MeasureSlotKind::Nav {
                    logical_nav: Some(LogicalNavKind::First),
                    logical_offset: 0,
                    physical_nav: None,
                    physical_offset: 0,
                    symbol: Some("u".into()),
                    inner_expr: col("val"),
                },
            }],
        },
        MeasureSpec {
            alias: "u_last_final".into(),
            outer_expr: col("__mr_slot_0"),
            slots: vec![MeasureSlot {
                slot_name: "__mr_slot_0".into(),
                semantics: EvalSemantics::Final,
                kind: MeasureSlotKind::Nav {
                    logical_nav: Some(LogicalNavKind::Last),
                    logical_offset: 0,
                    physical_nav: None,
                    physical_offset: 0,
                    symbol: Some("u".into()),
                    inner_expr: col("val"),
                },
            }],
        },
        MeasureSpec {
            alias: "u_cnt_final".into(),
            outer_expr: col("__mr_slot_0"),
            slots: vec![MeasureSlot {
                slot_name: "__mr_slot_0".into(),
                semantics: EvalSemantics::Final,
                kind: MeasureSlotKind::Agg {
                    agg_kind: AggKind::CountSymbol,
                    symbol: Some("u".into()),
                    args: vec![],
                    distinct: false,
                    arg_physical_nav: None,
                    uses_match_metadata: false,
                },
            }],
        },
        MeasureSpec {
            alias: "u_cls_running".into(),
            outer_expr: col("__mr_slot_0"),
            slots: vec![MeasureSlot {
                slot_name: "__mr_slot_0".into(),
                semantics: EvalSemantics::Running,
                kind: MeasureSlotKind::Classifier {
                    symbol: Some("u".into()),
                    logical_nav: None,
                    logical_offset: 0,
                    physical_nav: None,
                    physical_offset: 0,
                },
            }],
        },
    ];

    let mr_plan = MatchRecognizeNode::try_new_with_subsets(
        base_plan,
        vec![],
        vec![OrderBySpec {
            expr: col("id"),
            asc: true,
            nulls_first: false,
        }],
        measures,
        RowsPerMatch::AllRowsShowEmpty,
        AfterMatchSkip::PastLastRow,
        pattern,
        subsets,
        definitions,
        MatchRecognizeConfig::default(),
    )
    .unwrap()
    .into_plan();

    let df = ctx.execute_logical_plan(mr_plan.clone()).await.unwrap();
    let batches = df.clone().collect().await.unwrap();
    let formatted = pretty_format_batches(&batches).unwrap().to_string();
    assert!(
        formatted
            .contains("| 1  | 20      | 30           | 2           |               | A   | 10  |"),
        "got:\n{formatted}"
    );
    assert!(
        formatted
            .contains("| 2  | 20      | 30           | 2           | b             | B   | 20  |"),
        "got:\n{formatted}"
    );
    assert!(
        formatted
            .contains("| 3  | 20      | 30           | 2           | c             | C   | 30  |"),
        "got:\n{formatted}"
    );

    let lcodec = MatchRecognizeLogicalCodec::new();
    let lbytes = logical_plan_to_bytes_with_extension_codec(&mr_plan, &lcodec).unwrap();
    let ldecoded =
        logical_plan_from_bytes_with_extension_codec(&lbytes, &ctx.task_ctx(), &lcodec).unwrap();
    let lout = pretty_format_batches(
        &ctx.execute_logical_plan(ldecoded)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();
    assert_eq!(lout, formatted);

    let phys = df.create_physical_plan().await.unwrap();
    let pcodec = MatchRecognizePhysicalCodec::new();
    let pbytes = physical_plan_to_bytes_with_extension_codec(phys, &pcodec).unwrap();
    let pdecoded =
        physical_plan_from_bytes_with_extension_codec(&pbytes, &ctx.task_ctx(), &pcodec).unwrap();
    let pout = pretty_format_batches(&collect(pdecoded, ctx.task_ctx()).await.unwrap())
        .unwrap()
        .to_string();
    assert_eq!(pout, formatted);
}

#[tokio::test]
async fn test_udaf_and_classifier_nav_proto_roundtrip() {
    let ctx = session_context_with_match_recognize(
        SessionConfig::new()
            .with_batch_size(2)
            .with_target_partitions(2),
        MatchRecognizeConfig::default(),
    )
    .unwrap();

    let sql = r#"
        WITH seq(id, tag, price) AS (
            VALUES (1, 'A', 10), (2, 'B', 20), (3, 'B', 30), (4, 'C', 40)
        )
        SELECT * FROM seq
        MATCH_RECOGNIZE (
            ORDER BY id
            MEASURES
                FIRST(CLASSIFIER()) AS first_cls,
                LAST(CLASSIFIER(), 1) AS prev_last_cls,
                SUM(PREV(b.price)) AS sum_prev_b,
                ARRAY_AGG(b.price) AS b_arr,
                STRING_AGG(tag, '-') AS path_str,
                ROUND(STDDEV_SAMP(price), 4) AS sd
            ONE ROW PER MATCH
            PATTERN (a b+ c)
            DEFINE
                a AS tag = 'A',
                b AS tag = 'B',
                c AS tag = 'C'
        )
    "#;

    let df = ctx.sql(sql).await.unwrap();
    let expected = pretty_format_batches(&df.clone().collect().await.unwrap())
        .unwrap()
        .to_string();

    let lplan = df.clone().into_optimized_plan().unwrap();
    let lcodec = MatchRecognizeLogicalCodec::new();
    let lbytes = logical_plan_to_bytes_with_extension_codec(&lplan, &lcodec).unwrap();
    let ldecoded =
        logical_plan_from_bytes_with_extension_codec(&lbytes, &ctx.task_ctx(), &lcodec).unwrap();
    let lout = pretty_format_batches(
        &ctx.execute_logical_plan(ldecoded)
            .await
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();
    assert_eq!(lout, expected);

    let pplan = df.create_physical_plan().await.unwrap();
    let pcodec = MatchRecognizePhysicalCodec::new();
    let pbytes = physical_plan_to_bytes_with_extension_codec(pplan, &pcodec).unwrap();
    let pdecoded =
        physical_plan_from_bytes_with_extension_codec(&pbytes, &ctx.task_ctx(), &pcodec).unwrap();
    let pout = pretty_format_batches(&collect(pdecoded, ctx.task_ctx()).await.unwrap())
        .unwrap()
        .to_string();
    assert_eq!(pout, expected);
}

#[tokio::test]
async fn test_sql_unparser_roundtrip() {
    use datafusion_expr::logical_plan::LogicalPlan;
    use datafusion_match_recognize::MatchRecognizeUnparser;
    use datafusion_sql::unparser::Unparser;

    let ctx =
        session_context_with_match_recognize(SessionConfig::new(), MatchRecognizeConfig::default())
            .unwrap();
    ctx.register_batch(
        "ticks",
        RecordBatch::try_from_iter(vec![
            (
                "id",
                Arc::new(Int64Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            ),
            ("tag", Arc::new(StringArray::from(vec!["A", "B", "B", "C"]))),
            ("price", Arc::new(Int64Array::from(vec![10, 20, 30, 40]))),
        ])
        .unwrap(),
    )
    .unwrap();

    let sql = r#"
        SELECT * FROM ticks
        MATCH_RECOGNIZE (
            ORDER BY id
            MEASURES
                MATCH_NUMBER() AS mn,
                FIRST(a.price) AS start_p,
                LAST(c.price) AS end_p,
                SUM(b.price) AS sum_b
            ONE ROW PER MATCH
            AFTER MATCH SKIP PAST LAST ROW
            PATTERN (a b+ c)
            DEFINE
                a AS tag = 'A',
                b AS tag = 'B',
                c AS tag = 'C'
        )
    "#;

    let df = ctx.sql(sql).await.unwrap();
    let expected = pretty_format_batches(&df.clone().collect().await.unwrap())
        .unwrap()
        .to_string();

    // Locate the MatchRecognize Extension node and unparse it back to SQL via MatchRecognizeUnparser.
    fn find_extension(plan: &LogicalPlan) -> Option<LogicalPlan> {
        if matches!(plan, LogicalPlan::Extension(_)) {
            return Some(plan.clone());
        }
        for child in plan.inputs() {
            if let Some(found) = find_extension(child) {
                return Some(found);
            }
        }
        None
    }

    let mr_ext = find_extension(&df.into_unoptimized_plan()).expect("MatchRecognize Extension");
    let unparser =
        Unparser::default().with_extension_unparsers(vec![Arc::new(MatchRecognizeUnparser)]);
    let unparsed_sql = unparser.plan_to_sql(&mr_ext).unwrap().to_string();

    let roundtrip = run_and_format(&ctx, &unparsed_sql).await;
    assert_eq!(
        roundtrip, expected,
        "Unparsed SQL roundtrip mismatch!\nUnparsed SQL:\n{unparsed_sql}"
    );
}
