// SPDX-License-Identifier: Apache-2.0

//! SQL:2016 `MATCH_RECOGNIZE` (Row Pattern Recognition) extension for Apache DataFusion.
//!
//! Register with a [`datafusion::prelude::SessionContext`] via [`session_context_with_match_recognize`]
//! or [`register_match_recognize`].

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::Result;

pub mod ast;
pub mod logical;
pub mod matcher;
pub mod measures;
pub mod optimizer;
pub mod pattern;
pub mod physical;
pub mod planner;
#[cfg(feature = "proto")]
pub mod proto;

pub use ast::{
    AfterMatchSkip, AggKind, EvalSemantics, LogicalNavKind, MatchRecognizeConfig, MeasureSlot,
    MeasureSlotKind, MeasureSpec, OrderBySpec, Pattern, PhysicalNavKind, RowsPerMatch, SubsetDef,
    SymbolDef, SymbolDefKind, SymbolRef,
};
pub use logical::MatchRecognizeNode;
pub use optimizer::MatchRecognizeFilterPushdown;
pub use physical::{
    MatchRecognizeExec, MatchRecognizeExtensionPlanner, MatchRecognizeQueryPlanner,
};
pub use planner::MatchRecognizeRelationPlanner;
#[cfg(feature = "proto")]
pub use proto::{MatchRecognizeLogicalCodec, MatchRecognizePhysicalCodec};

/// Configures a [`SessionStateBuilder`] with `MATCH_RECOGNIZE` physical planning and
/// partition-filter pushdown optimization.
///
/// After calling `.build()` and creating a [`SessionContext`], register the SQL relation planner
/// via [`register_relation_planner`] (or use [`session_context_with_match_recognize`]).
pub fn configure_session_state(builder: SessionStateBuilder) -> SessionStateBuilder {
    builder
        .with_query_planner(Arc::new(MatchRecognizeQueryPlanner::new()))
        .with_optimizer_rule(Arc::new(MatchRecognizeFilterPushdown))
}

/// Registers [`MatchRecognizeRelationPlanner`] on an existing [`SessionContext`].
pub fn register_relation_planner(ctx: &SessionContext, config: MatchRecognizeConfig) -> Result<()> {
    ctx.register_relation_planner(Arc::new(MatchRecognizeRelationPlanner::new(config)))
}

/// Convenience helper constructing a [`SessionContext`] with full `MATCH_RECOGNIZE` support
/// (SQL relation planner, partition-filter pushdown optimizer, and physical execution planner).
pub fn session_context_with_match_recognize(
    session_config: SessionConfig,
    mr_config: MatchRecognizeConfig,
) -> Result<SessionContext> {
    let state = configure_session_state(
        SessionStateBuilder::new()
            .with_default_features()
            .with_config(session_config),
    )
    .build();
    let ctx = SessionContext::new_with_state(state);
    register_relation_planner(&ctx, mr_config)?;
    Ok(ctx)
}
