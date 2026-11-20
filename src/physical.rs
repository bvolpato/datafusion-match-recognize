// SPDX-License-Identifier: Apache-2.0

//! Physical execution plan (`MatchRecognizeExec`), [`ExtensionPlanner`], and [`QueryPlanner`]
//! for `MATCH_RECOGNIZE`, supporting Tier 1 stateless bitmask matching, Tier 2 hybrid stateful
//! `DEFINE` evaluation, `SUBSET` union variables, `DISTINCT` aggregates, arbitrary DataFusion
//! `AggregateUDF`s, and automatic disk spilling via DataFusion's [`SpillManager`].

use std::collections::HashMap;
use std::fmt::{self, Formatter};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, BooleanArray, RecordBatch};
use arrow::compute::{self, SortOptions};
use arrow::ipc::reader::StreamReader;
use arrow::row::{RowConverter, SortField};
use arrow_schema::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::Session;
use datafusion::execution::context::QueryPlanner;
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::spill_file::SpillFile;
use datafusion::execution::{RecordBatchStream, SendableRecordBatchStream, TaskContext};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, SpillMetrics,
};
use datafusion::physical_plan::spill::SpillManager;
use datafusion::physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, Distribution, ExecutionPlan,
    ExecutionPlanProperties, InputDistributionRequirements, PlanProperties, ReplaceChildrenOptions,
};
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{DFSchema, DataFusionError, Result, exec_datafusion_err, internal_err};
use datafusion_expr::logical_plan::LogicalPlan;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{Expr, UserDefinedLogicalNode};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::{
    LexRequirement, OrderingRequirements, PhysicalExpr, PhysicalSortExpr, PhysicalSortRequirement,
    create_physical_expr,
};
use futures::stream::Stream;

use crate::ast::{
    AfterMatchSkip, MatchRecognizeConfig, MeasureSlot, MeasureSlotKind, Pattern, RowsPerMatch,
    SubsetDef, SymbolDefKind,
};
use crate::logical::{
    MatchRecognizeNode, build_agg_metadata_dfschema, slot_data_type_and_nullability,
};
use crate::matcher::{Matcher, PartitionSymbols};
use crate::measures::{
    PartitionStatefulEvaluator, PhysicalMeasure, PhysicalMeasureSlot, PhysicalMeasureSlotKind,
    StatefulDefineInput, build_partition_output_batch, build_physical_slot_schema,
};
use crate::pattern::Program;

#[derive(Debug, Clone)]
pub enum PhysicalSymbolDefKind {
    Stateless {
        predicate: Arc<dyn PhysicalExpr>,
    },
    Stateful {
        outer_expr: Arc<dyn PhysicalExpr>,
        slot_schema: SchemaRef,
        slots: Vec<PhysicalMeasureSlot>,
    },
}

#[derive(Debug, Clone)]
pub struct PhysicalSymbolDef {
    pub name: String,
    pub kind: PhysicalSymbolDefKind,
}

#[derive(Debug, Clone)]
pub struct PhysicalOrderBySpec {
    pub expr: Arc<dyn PhysicalExpr>,
    pub options: SortOptions,
    pub input_col_idx: Option<usize>,
}

#[derive(Default)]
pub struct MatchRecognizeQueryPlanner {
    additional_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
}

impl fmt::Debug for MatchRecognizeQueryPlanner {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MatchRecognizeQueryPlanner")
            .field("additional_planners_len", &self.additional_planners.len())
            .finish()
    }
}

impl MatchRecognizeQueryPlanner {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_extension_planners(
        additional_planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>>,
    ) -> Self {
        Self {
            additional_planners,
        }
    }
}

#[async_trait]
impl QueryPlanner for MatchRecognizeQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let mut planners: Vec<Arc<dyn ExtensionPlanner + Send + Sync>> =
            Vec::with_capacity(self.additional_planners.len() + 1);
        planners.push(Arc::new(MatchRecognizeExtensionPlanner));
        planners.extend(self.additional_planners.iter().cloned());
        let planner = DefaultPhysicalPlanner::with_extension_planners(planners);
        planner
            .create_physical_plan(logical_plan, session_state)
            .await
    }
}

#[derive(Debug, Default)]
pub struct MatchRecognizeExtensionPlanner;

fn compile_physical_slots(
    slots: &[MeasureSlot],
    input_dfschema: &DFSchema,
    program: &Program,
    rows_per_match: RowsPerMatch,
    session_state: &dyn Session,
    planning_ctx: &PhysicalPlanningContext,
) -> Result<Vec<PhysicalMeasureSlot>> {
    let exec_props = session_state.execution_props();
    let mut phys_slots = Vec::with_capacity(slots.len());
    for slot in slots {
        let (data_type, nullable) =
            slot_data_type_and_nullability(input_dfschema, slot, rows_per_match)?;
        let kind = match &slot.kind {
            MeasureSlotKind::MatchNumber => PhysicalMeasureSlotKind::MatchNumber,
            MeasureSlotKind::Classifier {
                symbol,
                logical_nav,
                logical_offset,
                physical_nav,
                physical_offset,
            } => PhysicalMeasureSlotKind::Classifier {
                symbol_set: symbol.as_ref().and_then(|s| program.resolve_symbol_set(s)),
                symbol: symbol.clone(),
                logical_nav: *logical_nav,
                logical_offset: *logical_offset,
                physical_nav: *physical_nav,
                physical_offset: *physical_offset,
            },
            MeasureSlotKind::Nav {
                logical_nav,
                logical_offset,
                physical_nav,
                physical_offset,
                symbol,
                inner_expr,
                uses_match_metadata,
            } => {
                let meta_schema_owned;
                let eval_schema: &DFSchema = if *uses_match_metadata {
                    meta_schema_owned = build_agg_metadata_dfschema(input_dfschema)?;
                    &meta_schema_owned
                } else {
                    input_dfschema
                };
                let phys_inner =
                    create_physical_expr(inner_expr, eval_schema, exec_props, planning_ctx)?;
                PhysicalMeasureSlotKind::Nav {
                    logical_nav: *logical_nav,
                    logical_offset: *logical_offset,
                    physical_nav: *physical_nav,
                    physical_offset: *physical_offset,
                    symbol_set: symbol.as_ref().and_then(|s| program.resolve_symbol_set(s)),
                    symbol: symbol.clone(),
                    inner_expr: phys_inner,
                    uses_match_metadata: *uses_match_metadata,
                }
            }
            MeasureSlotKind::Agg {
                agg_kind,
                symbol,
                args,
                distinct,
                arg_physical_nav,
                uses_match_metadata,
            } => {
                let meta_schema_owned;
                let eval_schema: &DFSchema = if *uses_match_metadata {
                    meta_schema_owned = build_agg_metadata_dfschema(input_dfschema)?;
                    &meta_schema_owned
                } else {
                    input_dfschema
                };
                let phys_args = args
                    .iter()
                    .map(|e| create_physical_expr(e, eval_schema, exec_props, planning_ctx))
                    .collect::<Result<Vec<_>>>()?;
                PhysicalMeasureSlotKind::Agg {
                    agg_kind: agg_kind.clone(),
                    symbol_set: symbol.as_ref().and_then(|s| program.resolve_symbol_set(s)),
                    symbol: symbol.clone(),
                    args: phys_args,
                    distinct: *distinct,
                    arg_physical_nav: *arg_physical_nav,
                    uses_match_metadata: *uses_match_metadata,
                }
            }
            MeasureSlotKind::CurrentRow { inner_expr } => {
                let phys_inner =
                    create_physical_expr(inner_expr, input_dfschema, exec_props, planning_ctx)?;
                PhysicalMeasureSlotKind::CurrentRow {
                    inner_expr: phys_inner,
                }
            }
        };
        phys_slots.push(PhysicalMeasureSlot {
            slot_name: slot.slot_name.clone(),
            semantics: slot.semantics,
            data_type,
            nullable,
            kind,
        });
    }
    Ok(phys_slots)
}

#[async_trait]
impl ExtensionPlanner for MatchRecognizeExtensionPlanner {
    async fn plan_extension(
        &self,
        _planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        logical_inputs: &[&LogicalPlan],
        physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &dyn Session,
        planning_ctx: &PhysicalPlanningContext,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(mr_node) = node.as_any().downcast_ref::<MatchRecognizeNode>() else {
            return Ok(None);
        };

        let input_exec = Arc::clone(&physical_inputs[0]);
        let input_dfschema = logical_inputs[0].schema();
        let exec_props = session_state.execution_props();

        let partition_by: Vec<Arc<dyn PhysicalExpr>> = mr_node
            .partition_by
            .iter()
            .map(|e| create_physical_expr(e, input_dfschema, exec_props, planning_ctx))
            .collect::<Result<_>>()?;

        let order_by: Vec<PhysicalOrderBySpec> = mr_node
            .order_by
            .iter()
            .map(|ob| {
                let expr =
                    create_physical_expr(&ob.expr, input_dfschema, exec_props, planning_ctx)?;
                let input_col_idx = expr.downcast_ref::<Column>().map(|c| c.index());
                Ok(PhysicalOrderBySpec {
                    expr,
                    options: SortOptions {
                        descending: !ob.asc,
                        nulls_first: ob.nulls_first,
                    },
                    input_col_idx,
                })
            })
            .collect::<Result<_>>()?;

        let declared_names: Vec<String> =
            mr_node.definitions.iter().map(|d| d.name.clone()).collect();
        let program = Program::compile_with_subsets(
            &mr_node.pattern,
            &declared_names,
            &mr_node.subsets,
            mr_node.config,
        )?;

        let mut definitions = Vec::with_capacity(mr_node.definitions.len());
        for d in &mr_node.definitions {
            let kind = match &d.kind {
                SymbolDefKind::Stateless { predicate } => {
                    let phys_pred =
                        create_physical_expr(predicate, input_dfschema, exec_props, planning_ctx)?;
                    PhysicalSymbolDefKind::Stateless {
                        predicate: phys_pred,
                    }
                }
                SymbolDefKind::Stateful { outer_expr, slots } => {
                    let phys_slots = compile_physical_slots(
                        slots,
                        input_dfschema,
                        &program,
                        RowsPerMatch::OneRow,
                        session_state,
                        planning_ctx,
                    )?;
                    let slot_schema = build_physical_slot_schema(&phys_slots);
                    let slot_dfschema = DFSchema::try_from(slot_schema.as_ref().clone())?;
                    let phys_outer =
                        create_physical_expr(outer_expr, &slot_dfschema, exec_props, planning_ctx)?;
                    PhysicalSymbolDefKind::Stateful {
                        outer_expr: phys_outer,
                        slot_schema,
                        slots: phys_slots,
                    }
                }
            };
            definitions.push(PhysicalSymbolDef {
                name: d.name.clone(),
                kind,
            });
        }

        let mut physical_measures = Vec::with_capacity(mr_node.measures.len());
        for m in &mr_node.measures {
            let phys_slots = compile_physical_slots(
                &m.slots,
                input_dfschema,
                &program,
                mr_node.rows_per_match,
                session_state,
                planning_ctx,
            )?;
            let slot_schema = build_physical_slot_schema(&phys_slots);
            let slot_dfschema = DFSchema::try_from(slot_schema.as_ref().clone())?;
            let outer_expr =
                create_physical_expr(&m.outer_expr, &slot_dfschema, exec_props, planning_ctx)?;

            physical_measures.push(PhysicalMeasure {
                alias: m.alias.clone(),
                outer_expr,
                slot_schema,
                slots: phys_slots,
            });
        }

        let output_schema = Arc::clone(mr_node.schema.inner());

        let exec = MatchRecognizeExec::try_new_with_subsets(
            input_exec,
            partition_by,
            order_by,
            physical_measures,
            mr_node.rows_per_match,
            mr_node.after_match_skip.clone(),
            mr_node.pattern.clone(),
            mr_node.subsets.clone(),
            definitions,
            mr_node.config,
            output_schema,
        )?;
        Ok(Some(Arc::new(exec)))
    }
}

/// Physical [`ExecutionPlan`] executing SQL:2016 `MATCH_RECOGNIZE`.
#[derive(Debug, Clone)]
pub struct MatchRecognizeExec {
    input: Arc<dyn ExecutionPlan>,
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    partition_by_input_indices: Vec<Option<usize>>,
    order_by: Vec<PhysicalOrderBySpec>,
    measures: Vec<PhysicalMeasure>,
    rows_per_match: RowsPerMatch,
    after_match_skip: AfterMatchSkip,
    pattern: Pattern,
    subsets: Vec<SubsetDef>,
    definitions: Vec<PhysicalSymbolDef>,
    config: MatchRecognizeConfig,
    program: Arc<Program>,
    output_schema: SchemaRef,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl MatchRecognizeExec {
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        partition_by: Vec<Arc<dyn PhysicalExpr>>,
        order_by: Vec<PhysicalOrderBySpec>,
        measures: Vec<PhysicalMeasure>,
        rows_per_match: RowsPerMatch,
        after_match_skip: AfterMatchSkip,
        pattern: Pattern,
        definitions: Vec<PhysicalSymbolDef>,
        config: MatchRecognizeConfig,
        output_schema: SchemaRef,
    ) -> Result<Self> {
        Self::try_new_with_subsets(
            input,
            partition_by,
            order_by,
            measures,
            rows_per_match,
            after_match_skip,
            pattern,
            vec![],
            definitions,
            config,
            output_schema,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn try_new_with_subsets(
        input: Arc<dyn ExecutionPlan>,
        partition_by: Vec<Arc<dyn PhysicalExpr>>,
        order_by: Vec<PhysicalOrderBySpec>,
        measures: Vec<PhysicalMeasure>,
        rows_per_match: RowsPerMatch,
        after_match_skip: AfterMatchSkip,
        pattern: Pattern,
        subsets: Vec<SubsetDef>,
        definitions: Vec<PhysicalSymbolDef>,
        config: MatchRecognizeConfig,
        output_schema: SchemaRef,
    ) -> Result<Self> {
        let declared_names: Vec<String> = definitions.iter().map(|d| d.name.clone()).collect();
        let program = Arc::new(Program::compile_with_subsets(
            &pattern,
            &declared_names,
            &subsets,
            config,
        )?);

        let partition_by_input_indices: Vec<Option<usize>> = partition_by
            .iter()
            .map(|e| e.downcast_ref::<Column>().map(|c| c.index()))
            .collect();

        let eq_properties = EquivalenceProperties::new(Arc::clone(&output_schema));
        let output_partitioning = input.output_partitioning().clone();
        let cache = Arc::new(PlanProperties::new(
            eq_properties,
            output_partitioning,
            input.pipeline_behavior(),
            input.boundedness(),
        ));

        Ok(Self {
            input,
            partition_by,
            partition_by_input_indices,
            order_by,
            measures,
            rows_per_match,
            after_match_skip,
            pattern,
            subsets,
            definitions,
            config,
            program,
            output_schema,
            cache,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }

    pub fn input(&self) -> &Arc<dyn ExecutionPlan> {
        &self.input
    }

    pub fn partition_by(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.partition_by
    }

    pub fn order_by(&self) -> &[PhysicalOrderBySpec] {
        &self.order_by
    }

    pub fn measures(&self) -> &[PhysicalMeasure] {
        &self.measures
    }

    pub fn rows_per_match(&self) -> RowsPerMatch {
        self.rows_per_match
    }

    pub fn after_match_skip(&self) -> &AfterMatchSkip {
        &self.after_match_skip
    }

    pub fn pattern(&self) -> &Pattern {
        &self.pattern
    }

    pub fn subsets(&self) -> &[SubsetDef] {
        &self.subsets
    }

    pub fn definitions(&self) -> &[PhysicalSymbolDef] {
        &self.definitions
    }

    pub fn config(&self) -> MatchRecognizeConfig {
        self.config
    }

    pub fn output_schema(&self) -> &SchemaRef {
        &self.output_schema
    }
}

impl DisplayAs for MatchRecognizeExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "MatchRecognizeExec: pattern=[{}], rows_per_match=[{}], skip=[{}]",
            self.pattern, self.rows_per_match, self.after_match_skip
        )
    }
}

fn apply_slot_exprs(
    slots: &[PhysicalMeasureSlot],
    f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
) -> Result<TreeNodeRecursion> {
    for slot in slots {
        match &slot.kind {
            PhysicalMeasureSlotKind::Nav {
                inner_expr,
                uses_match_metadata,
                ..
            } => {
                if !uses_match_metadata && f(inner_expr)? == TreeNodeRecursion::Stop {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
                if f(inner_expr)? == TreeNodeRecursion::Stop {
                    return Ok(TreeNodeRecursion::Stop);
                }
            }
            PhysicalMeasureSlotKind::Agg {
                args,
                uses_match_metadata,
                ..
            } => {
                if !uses_match_metadata {
                    for e in args {
                        if f(e)? == TreeNodeRecursion::Stop {
                            return Ok(TreeNodeRecursion::Stop);
                        }
                    }
                }
            }
            PhysicalMeasureSlotKind::MatchNumber | PhysicalMeasureSlotKind::Classifier { .. } => {}
        }
    }
    Ok(TreeNodeRecursion::Continue)
}

impl ExecutionPlan for MatchRecognizeExec {
    fn name(&self) -> &'static str {
        "MatchRecognizeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        for expr in &self.partition_by {
            if f(expr)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        for ob in &self.order_by {
            if f(&ob.expr)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        for def in &self.definitions {
            match &def.kind {
                PhysicalSymbolDefKind::Stateless { predicate } => {
                    if f(predicate)? == TreeNodeRecursion::Stop {
                        return Ok(TreeNodeRecursion::Stop);
                    }
                }
                PhysicalSymbolDefKind::Stateful { slots, .. } => {
                    if apply_slot_exprs(slots, f)? == TreeNodeRecursion::Stop {
                        return Ok(TreeNodeRecursion::Stop);
                    }
                }
            }
        }
        for m in &self.measures {
            if apply_slot_exprs(&m.slots, f)? == TreeNodeRecursion::Stop {
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        self.input_distribution_requirements().into_per_child()
    }

    fn input_distribution_requirements(&self) -> InputDistributionRequirements {
        if self.partition_by.is_empty() {
            InputDistributionRequirements::new(vec![Distribution::SinglePartition])
        } else {
            InputDistributionRequirements::new(vec![Distribution::KeyPartitioned(
                self.partition_by.clone(),
            )])
        }
    }

    fn required_input_ordering(&self) -> Vec<Option<OrderingRequirements>> {
        let mut reqs: Vec<PhysicalSortRequirement> = Vec::new();
        for p in &self.partition_by {
            reqs.push(PhysicalSortRequirement::new(Arc::clone(p), None));
        }
        for ob in &self.order_by {
            reqs.push(PhysicalSortRequirement::from(PhysicalSortExpr {
                expr: Arc::clone(&ob.expr),
                options: ob.options,
            }));
        }
        if reqs.is_empty() {
            vec![None]
        } else {
            vec![LexRequirement::new(reqs).map(OrderingRequirements::from)]
        }
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }

    fn replace_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("MatchRecognizeExec expects 1 child, got {}", children.len());
        }
        let child = children.swap_remove(0);
        Ok(Arc::new(Self::try_new_with_subsets(
            child,
            self.partition_by.clone(),
            self.order_by.clone(),
            self.measures.clone(),
            self.rows_per_match,
            self.after_match_skip.clone(),
            self.pattern.clone(),
            self.subsets.clone(),
            self.definitions.clone(),
            self.config,
            Arc::clone(&self.output_schema),
        )?))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input_stream = self.input.execute(partition, Arc::clone(&context))?;
        let runtime_env = context.runtime_env();
        let disk_spill_enabled = runtime_env.disk_manager.tmp_files_enabled();
        let consumer = MemoryConsumer::new(format!("MatchRecognizeExec[{partition}]"))
            .with_can_spill(disk_spill_enabled);
        let reservation = consumer.register(context.memory_pool());
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let input_schema = self.input.schema();
        let spill_manager = if disk_spill_enabled {
            let spill_metrics = SpillMetrics::new(&self.metrics, partition);
            Some(SpillManager::new(
                runtime_env,
                spill_metrics,
                Arc::clone(&input_schema),
            ))
        } else {
            None
        };

        let order_by_input_indices: Vec<Option<usize>> =
            self.order_by.iter().map(|o| o.input_col_idx).collect();

        let row_converter = if self.partition_by.is_empty() {
            None
        } else {
            let sort_fields: Vec<SortField> = self
                .partition_by
                .iter()
                .map(|e| Ok(SortField::new(e.data_type(input_schema.as_ref())?)))
                .collect::<Result<_>>()?;
            Some(RowConverter::new(sort_fields)?)
        };

        Ok(Box::pin(MatchRecognizeStream {
            input: input_stream,
            input_schema,
            output_schema: Arc::clone(&self.output_schema),
            partition_by: self.partition_by.clone(),
            partition_by_input_indices: self.partition_by_input_indices.clone(),
            order_by_input_indices,
            measures: self.measures.clone(),
            rows_per_match: self.rows_per_match,
            after_match_skip: self.after_match_skip.clone(),
            definitions: self.definitions.clone(),
            program: Arc::clone(&self.program),
            max_active_threads: self.config.max_active_threads as usize,
            row_converter,
            current_partition_key: None,
            buffered_batches: Vec::new(),
            spilled_files: Vec::new(),
            spill_manager,
            pending_outputs: std::collections::VecDeque::new(),
            input_done: false,
            reservation,
            metrics,
        }))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
}

struct MatchRecognizeStream {
    input: SendableRecordBatchStream,
    input_schema: SchemaRef,
    output_schema: SchemaRef,
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    partition_by_input_indices: Vec<Option<usize>>,
    order_by_input_indices: Vec<Option<usize>>,
    measures: Vec<PhysicalMeasure>,
    rows_per_match: RowsPerMatch,
    after_match_skip: AfterMatchSkip,
    definitions: Vec<PhysicalSymbolDef>,
    program: Arc<Program>,
    max_active_threads: usize,
    row_converter: Option<RowConverter>,
    current_partition_key: Option<Vec<u8>>,
    buffered_batches: Vec<RecordBatch>,
    spilled_files: Vec<Arc<dyn SpillFile>>,
    spill_manager: Option<SpillManager>,
    pending_outputs: std::collections::VecDeque<RecordBatch>,
    input_done: bool,
    reservation: MemoryReservation,
    metrics: BaselineMetrics,
}

impl MatchRecognizeStream {
    /// Buffers a batch for the active partition, automatically spilling to disk via
    /// [`SpillManager`] if memory reservation fails and disk spilling is enabled.
    fn push_partition_batch(&mut self, batch: RecordBatch) -> Result<()> {
        let mem = batch.get_array_memory_size();
        if let Err(err) = self.reservation.try_grow(mem) {
            if let Some(spill_mgr) = &self.spill_manager {
                let mut to_spill = std::mem::take(&mut self.buffered_batches);
                to_spill.push(batch);
                if let Some(file) = spill_mgr.spill_record_batch_and_finish(
                    &to_spill,
                    "MatchRecognizeExec partition buffer spill",
                )? {
                    self.spilled_files.push(file);
                }
                self.reservation.free();
                return Ok(());
            }
            return Err(err);
        }
        self.buffered_batches.push(batch);
        Ok(())
    }

    fn ingest_batch(&mut self, batch: RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }

        if self.partition_by.is_empty() {
            return self.push_partition_batch(batch);
        }

        let key_arrays: Vec<_> = self
            .partition_by
            .iter()
            .map(|e| e.evaluate(&batch)?.into_array(batch.num_rows()))
            .collect::<Result<_>>()?;
        let converter = self
            .row_converter
            .as_ref()
            .expect("RowConverter present when partition_by is non-empty");
        let rows = converter.convert_columns(&key_arrays)?;

        let num_rows = batch.num_rows();
        let mut slice_start = 0;

        for row_idx in 0..num_rows {
            let row = rows.row(row_idx);
            let key_bytes = row.as_ref();
            let same_partition = match &self.current_partition_key {
                None => {
                    self.current_partition_key = Some(key_bytes.to_vec());
                    true
                }
                Some(curr) => curr.as_slice() == key_bytes,
            };

            if !same_partition {
                if row_idx > slice_start {
                    let sub = batch.slice(slice_start, row_idx - slice_start);
                    self.push_partition_batch(sub)?;
                }
                self.flush_current_partition()?;
                self.current_partition_key = Some(key_bytes.to_vec());
                slice_start = row_idx;
            }
        }

        if slice_start < num_rows {
            let sub = batch.slice(slice_start, num_rows - slice_start);
            self.push_partition_batch(sub)?;
        }

        Ok(())
    }

    fn flush_current_partition(&mut self) -> Result<()> {
        if self.buffered_batches.is_empty() && self.spilled_files.is_empty() {
            return Ok(());
        }

        let mut all_batches = Vec::new();
        let spilled = std::mem::take(&mut self.spilled_files);
        let had_spills = !spilled.is_empty();
        for spill_file in spilled {
            let path = spill_file.path().ok_or_else(|| {
                exec_datafusion_err!("MatchRecognizeExec spill file does not expose a local path")
            })?;
            let file = std::fs::File::open(path).map_err(DataFusionError::IoError)?;
            let reader = StreamReader::try_new(file, None)?;
            for batch_res in reader {
                all_batches.push(batch_res?);
            }
        }
        all_batches.extend(std::mem::take(&mut self.buffered_batches));

        if all_batches.is_empty() {
            self.reservation.free();
            return Ok(());
        }

        let partition_batch = if all_batches.len() == 1 {
            all_batches.into_iter().next().expect("len 1")
        } else {
            compute::concat_batches(&self.input_schema, &all_batches)?
        };

        // When no disk spill occurred, enforce tight memory accounting on the active partition;
        // when spilled from disk, allow processing the reloaded partition and release immediately.
        if !had_spills {
            let work_mem = partition_batch.get_array_memory_size();
            self.reservation.try_resize(work_mem)?;
        }

        let symbols =
            evaluate_stateless_symbols(&partition_batch, &self.program, &self.definitions)?;
        if !had_spills {
            self.reservation.try_grow(symbols.memory_bytes())?;
        }

        let stateful_defs: Vec<StatefulDefineInput<'_>> = self
            .definitions
            .iter()
            .filter_map(|d| match &d.kind {
                PhysicalSymbolDefKind::Stateless { .. } => None,
                PhysicalSymbolDefKind::Stateful {
                    outer_expr,
                    slot_schema,
                    slots,
                } => self
                    .program
                    .symbol_id(&d.name)
                    .map(|sym_id| StatefulDefineInput {
                        sym_id,
                        outer_expr,
                        slot_schema,
                        slots: slots.as_slice(),
                    }),
            })
            .collect();

        let mut matcher = Matcher::with_max_active_threads(&self.program, self.max_active_threads);
        let (events, _stats) = if stateful_defs.is_empty() {
            matcher.match_partition(&symbols, &self.after_match_skip, self.rows_per_match)?
        } else {
            let mut stateful_eval = PartitionStatefulEvaluator::try_new(
                &partition_batch,
                &self.program,
                &stateful_defs,
            )?;
            matcher.match_partition_with_evaluator(
                &symbols,
                &self.after_match_skip,
                self.rows_per_match,
                &mut stateful_eval,
            )?
        };

        if let Some(out_batch) = build_partition_output_batch(
            &partition_batch,
            &events,
            &self.program,
            &self.partition_by,
            &self.partition_by_input_indices,
            &self.order_by_input_indices,
            &self.measures,
            self.rows_per_match,
            &self.output_schema,
        )? {
            self.metrics.record_output(out_batch.num_rows());
            self.pending_outputs.push_back(out_batch);
        }

        self.reservation.free();
        Ok(())
    }
}

fn evaluate_stateless_symbols(
    partition_batch: &RecordBatch,
    program: &Program,
    definitions: &[PhysicalSymbolDef],
) -> Result<PartitionSymbols> {
    let num_rows = partition_batch.num_rows();
    let num_symbols = program.num_symbols();
    let def_map: HashMap<&str, &PhysicalSymbolDefKind> = definitions
        .iter()
        .map(|d| (d.name.as_str(), &d.kind))
        .collect();

    let mut columns: Vec<Vec<bool>> = Vec::with_capacity(num_symbols);
    for sym_name in &program.id_to_symbol {
        match def_map.get(sym_name.as_str()) {
            Some(PhysicalSymbolDefKind::Stateless { predicate }) => {
                let evaluated = predicate.evaluate(partition_batch)?.into_array(num_rows)?;
                let bool_arr = evaluated
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "DEFINE predicate for '{sym_name}' must evaluate to Boolean, got {}",
                            evaluated.data_type()
                        ))
                    })?;
                let mut flags = Vec::with_capacity(num_rows);
                for i in 0..num_rows {
                    flags.push(bool_arr.is_valid(i) && bool_arr.value(i));
                }
                columns.push(flags);
            }
            Some(PhysicalSymbolDefKind::Stateful { .. }) => {
                columns.push(vec![false; num_rows]);
            }
            None => {
                columns.push(vec![true; num_rows]);
            }
        }
    }

    Ok(PartitionSymbols::from_columns(&columns, num_rows))
}

impl Stream for MatchRecognizeStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.as_mut().get_mut();
        loop {
            if let Some(batch) = this.pending_outputs.pop_front() {
                return Poll::Ready(Some(Ok(batch)));
            }

            if this.input_done {
                return Poll::Ready(None);
            }

            match Pin::new(&mut this.input).poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(Some(Ok(batch))) => {
                    let timer = this.metrics.elapsed_compute().clone();
                    let _guard = timer.timer();
                    let res = this.ingest_batch(batch);
                    drop(_guard);
                    if let Err(e) = res {
                        this.input_done = true;
                        return Poll::Ready(Some(Err(e)));
                    }
                }
                Poll::Ready(Some(Err(e))) => {
                    this.input_done = true;
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Ready(None) => {
                    this.input_done = true;
                    let timer = this.metrics.elapsed_compute().clone();
                    let _guard = timer.timer();
                    let res = this.flush_current_partition();
                    drop(_guard);
                    if let Err(e) = res {
                        return Poll::Ready(Some(Err(e)));
                    }
                }
            }
        }
    }
}

impl RecordBatchStream for MatchRecognizeStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.output_schema)
    }
}

pub fn partition_expr_is_column(expr: &Expr) -> bool {
    matches!(expr, Expr::Column(_))
}
