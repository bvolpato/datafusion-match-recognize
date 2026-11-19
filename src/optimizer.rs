// SPDX-License-Identifier: Apache-2.0

//! Optional optimizer rule pushing filters that reference only `PARTITION BY` columns below
//! [`MatchRecognizeNode`].

use std::collections::HashSet;
use std::sync::Arc;

use datafusion::optimizer::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::Transformed;
use datafusion_common::{Column, Result};
use datafusion_expr::Expr;
use datafusion_expr::logical_plan::{Extension, Filter, LogicalPlan};
use datafusion_expr::utils::{conjunction, expr_to_columns, split_conjunction};

use crate::logical::MatchRecognizeNode;

/// Pushes filter conjuncts referencing exclusively `PARTITION BY` columns below
/// [`MatchRecognizeNode`].
///
/// Filtering entire partitions before sorting and pattern matching never alters match boundaries
/// or `MATCH_NUMBER()` within surviving partitions.
#[derive(Debug, Default)]
pub struct MatchRecognizeFilterPushdown;

impl OptimizerRule for MatchRecognizeFilterPushdown {
    fn name(&self) -> &str {
        "match_recognize_filter_pushdown"
    }

    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::TopDown)
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        let LogicalPlan::Filter(filter) = plan else {
            return Ok(Transformed::no(plan));
        };

        let LogicalPlan::Extension(Extension { node }) = filter.input.as_ref() else {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        };

        let Some(mr_node) = node.as_any().downcast_ref::<MatchRecognizeNode>() else {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        };

        if mr_node.partition_by.is_empty() {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }

        // Collect normalized partition column names exposed by MatchRecognizeNode.
        let mut partition_col_names: HashSet<String> = HashSet::new();
        for p in &mr_node.partition_by {
            if let Expr::Column(c) = p {
                partition_col_names.insert(c.name.clone());
            }
        }
        if partition_col_names.is_empty() {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }

        let conjuncts = split_conjunction(&filter.predicate);
        let mut pushable: Vec<Expr> = Vec::new();
        let mut remaining: Vec<Expr> = Vec::new();

        for pred in conjuncts {
            let mut cols: HashSet<Column> = HashSet::new();
            expr_to_columns(pred, &mut cols)?;
            if !cols.is_empty() && cols.iter().all(|c| partition_col_names.contains(&c.name)) {
                // Rewrite column qualifiers to match input schema.
                let mut valid_on_input = true;
                let input_schema = mr_node.input.schema();
                for c in &cols {
                    if input_schema.field_with_unqualified_name(&c.name).is_err() {
                        valid_on_input = false;
                        break;
                    }
                }
                if valid_on_input {
                    pushable.push(pred.clone());
                    continue;
                }
            }
            remaining.push(pred.clone());
        }

        if pushable.is_empty() {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }

        let pushed_pred = conjunction(pushable).expect("non-empty pushable");
        let new_input = LogicalPlan::Filter(Filter::try_new(
            pushed_pred,
            Arc::new(mr_node.input.clone()),
        )?);

        let new_mr = MatchRecognizeNode::try_new(
            new_input,
            mr_node.partition_by.clone(),
            mr_node.order_by.clone(),
            mr_node.measures.clone(),
            mr_node.rows_per_match,
            mr_node.after_match_skip.clone(),
            mr_node.pattern.clone(),
            mr_node.definitions.clone(),
            mr_node.config,
        )?
        .into_plan();

        let final_plan = if let Some(rem) = conjunction(remaining) {
            LogicalPlan::Filter(Filter::try_new(rem, Arc::new(new_mr))?)
        } else {
            new_mr
        };

        Ok(Transformed::yes(final_plan))
    }
}
