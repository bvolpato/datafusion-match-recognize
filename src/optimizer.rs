// SPDX-License-Identifier: Apache-2.0

//! Optional optimizer rule pushing filters that reference only `PARTITION BY` columns below
//! [`MatchRecognizeNode`].

use std::sync::Arc;

use datafusion::optimizer::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{Column, HashSet, Result};
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
        // Push pure column projections into MatchRecognizeNode so unused output columns
        // (especially in ALL ROWS PER MATCH) skip materialization and allow upstream column
        // pruning into Parquet/Repartition/Sort.
        if let LogicalPlan::Projection(proj) = &plan
            && let LogicalPlan::Extension(Extension { node }) = proj.input.as_ref()
            && let Some(mr_node) = node.as_any().downcast_ref::<MatchRecognizeNode>()
            && mr_node.output_projection.is_none()
            && !proj.expr.is_empty()
            && proj.expr.len() < mr_node.schema.fields().len()
        {
            let mut col_names: Vec<String> = Vec::with_capacity(proj.expr.len());
            let mut seen = HashSet::new();
            let mut all_pure_unaliased_cols = true;
            for (expr, out_field) in proj.expr.iter().zip(proj.schema.fields()) {
                if let Expr::Column(c) = expr
                    && c.name == *out_field.name()
                    && seen.insert(c.name.clone())
                    && mr_node.schema.field_with_unqualified_name(&c.name).is_ok()
                    && (mr_node.measures.iter().any(|m| m.alias == c.name)
                        || mr_node
                            .input
                            .schema()
                            .field_with_unqualified_name(&c.name)
                            .is_ok())
                {
                    col_names.push(c.name.clone());
                } else {
                    all_pure_unaliased_cols = false;
                    break;
                }
            }
            if all_pure_unaliased_cols {
                let projected_mr = MatchRecognizeNode::try_new_with_projection(
                    mr_node.input.clone(),
                    mr_node.partition_by.clone(),
                    mr_node.order_by.clone(),
                    mr_node.measures.clone(),
                    mr_node.rows_per_match,
                    mr_node.after_match_skip.clone(),
                    mr_node.pattern.clone(),
                    mr_node.subsets.clone(),
                    mr_node.definitions.clone(),
                    mr_node.config,
                    Some(col_names),
                )?
                .into_plan();
                // Keep outer schema qualifiers (if any) intact only if needed.
                if projected_mr.schema() == &proj.schema {
                    return Ok(Transformed::yes(projected_mr));
                }
            }
        }

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
            #[expect(
                clippy::disallowed_types,
                reason = "datafusion_expr::utils::expr_to_columns requires std::collections::HashSet"
            )]
            let mut cols: std::collections::HashSet<Column> =
                std::collections::HashSet::new();
            expr_to_columns(pred, &mut cols)?;
            if !cols.is_empty()
                && cols.iter().all(|c| partition_col_names.contains(&c.name))
            {
                // Rewrite column qualifiers to match input schema (dropping any MATCH_RECOGNIZE alias).
                let input_schema = mr_node.input.schema();
                let mut valid_on_input = true;
                for c in &cols {
                    if input_schema
                        .qualified_field_with_unqualified_name(&c.name)
                        .is_err()
                    {
                        valid_on_input = false;
                        break;
                    }
                }
                if valid_on_input {
                    let rewritten = pred
                        .clone()
                        .transform(|e| match e {
                            Expr::Column(c) => {
                                let (qual, field) = input_schema
                                    .qualified_field_with_unqualified_name(&c.name)?;
                                Ok(Transformed::yes(Expr::Column(Column::new(
                                    qual.cloned(),
                                    field.name(),
                                ))))
                            }
                            other => Ok(Transformed::no(other)),
                        })?
                        .data;
                    pushable.push(rewritten);
                    continue;
                }
            }
            remaining.push(pred.clone());
        }

        if pushable.is_empty() {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        }

        let Some(pushed_pred) = conjunction(pushable) else {
            return Ok(Transformed::no(LogicalPlan::Filter(filter)));
        };
        let new_input = LogicalPlan::Filter(Filter::try_new(
            pushed_pred,
            Arc::new(mr_node.input.clone()),
        )?);

        let new_mr = MatchRecognizeNode::try_new_with_projection(
            new_input,
            mr_node.partition_by.clone(),
            mr_node.order_by.clone(),
            mr_node.measures.clone(),
            mr_node.rows_per_match,
            mr_node.after_match_skip.clone(),
            mr_node.pattern.clone(),
            mr_node.subsets.clone(),
            mr_node.definitions.clone(),
            mr_node.config,
            mr_node.output_projection.clone(),
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
