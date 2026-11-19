// SPDX-License-Identifier: Apache-2.0

//! Protobuf serialization codecs ([`LogicalExtensionCodec`] and [`PhysicalExtensionCodec`])
//! for distributed execution (e.g., `datafusion-distributed`) and plan persistence.
//!
//! Both codecs support optional delegation (`with_delegate(...)`) so downstream engines can
//! chain `MATCH_RECOGNIZE` serialization with custom `TableProvider`, `FileFormat`, or other
//! extension plan codecs.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::compute::SortOptions;
use arrow_schema::{DataType, Field, Schema};
use datafusion::catalog::TableProvider;
use datafusion::datasource::file_format::FileFormatFactory;
use datafusion::execution::TaskContext;
use datafusion::physical_plan::ExecutionPlan;
use datafusion_common::{
    DFSchema, DataFusionError, Result, TableReference, internal_err, not_impl_err,
};
use datafusion_expr::logical_plan::{Extension, LogicalPlan};
use datafusion_physical_expr::expressions::Column;
use datafusion_proto::logical_plan::{
    DefaultLogicalExtensionCodec, LogicalExtensionCodec, from_proto as logical_from_proto,
    to_proto as logical_to_proto,
};
use datafusion_proto::physical_plan::{
    DefaultPhysicalExtensionCodec, PhysicalExtensionCodec, PhysicalPlanDecodeContext,
    PhysicalProtoConverterExtension, from_proto as phys_from_proto, to_proto as phys_to_proto,
};
use datafusion_proto::protobuf::{LogicalExprNode, PhysicalExprNode};
use prost::Message;

use crate::ast::{
    AfterMatchSkip, AggKind, DEFAULT_MAX_ACTIVE_THREADS, EvalSemantics, LogicalNavKind,
    MatchRecognizeConfig, MeasureSlot, MeasureSlotKind, MeasureSpec, OrderBySpec, Pattern,
    PhysicalNavKind, RowsPerMatch, SubsetDef, SymbolDef, SymbolDefKind, SymbolRef,
};
use crate::logical::MatchRecognizeNode;
use crate::measures::{
    PhysicalMeasure, PhysicalMeasureSlot, PhysicalMeasureSlotKind, build_physical_slot_schema,
};
use crate::pattern::Program;
use crate::physical::{
    MatchRecognizeExec, PhysicalOrderBySpec, PhysicalSymbolDef, PhysicalSymbolDefKind,
};

pub mod pb {
    include!(concat!(
        env!("OUT_DIR"),
        "/datafusion_match_recognize.v1.rs"
    ));
}

/// [`LogicalExtensionCodec`] for serializing and deserializing [`MatchRecognizeNode`].
#[derive(Debug, Default)]
pub struct MatchRecognizeLogicalCodec {
    delegate: Option<Arc<dyn LogicalExtensionCodec>>,
}

impl MatchRecognizeLogicalCodec {
    /// Creates a codec handling [`MatchRecognizeNode`] without a fallback delegate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a codec that delegates non-`MatchRecognize` extensions, table providers, and
    /// file formats to `delegate`.
    pub fn with_delegate(delegate: Arc<dyn LogicalExtensionCodec>) -> Self {
        Self {
            delegate: Some(delegate),
        }
    }
}

fn decode_logical_slots(
    proto_slots: &[pb::MeasureSlotProto],
    decode_expr: &dyn Fn(&[u8]) -> Result<datafusion_expr::Expr>,
) -> Result<Vec<MeasureSlot>> {
    let mut slots = Vec::with_capacity(proto_slots.len());
    for s in proto_slots {
        let semantics = decode_eval_semantics(s.semantics);
        let kind =
            match s.kind.as_ref().ok_or_else(|| {
                DataFusionError::Internal("Missing MeasureSlotProto.kind".to_string())
            })? {
                pb::measure_slot_proto::Kind::MatchNumber(_) => MeasureSlotKind::MatchNumber,
                pb::measure_slot_proto::Kind::Classifier(c) => MeasureSlotKind::Classifier {
                    symbol: c.symbol.clone(),
                },
                pb::measure_slot_proto::Kind::Nav(n) => MeasureSlotKind::Nav {
                    logical_nav: n.logical_nav.map(decode_logical_nav),
                    logical_offset: n.logical_offset,
                    physical_nav: n.physical_nav.map(decode_physical_nav),
                    physical_offset: n.physical_offset,
                    symbol: n.symbol.clone(),
                    inner_expr: decode_expr(&n.inner_expr)?,
                },
                pb::measure_slot_proto::Kind::Agg(a) => MeasureSlotKind::Agg {
                    agg_kind: decode_agg_kind(a.agg_kind)?,
                    symbol: a.symbol.clone(),
                    inner_expr: a.inner_expr.as_deref().map(decode_expr).transpose()?,
                    distinct: a.distinct,
                },
                pb::measure_slot_proto::Kind::CurrentRow(cr) => MeasureSlotKind::CurrentRow {
                    inner_expr: decode_expr(&cr.inner_expr)?,
                },
            };
        slots.push(MeasureSlot {
            slot_name: s.slot_name.clone(),
            semantics,
            kind,
        });
    }
    Ok(slots)
}

fn encode_logical_slots(
    slots: &[MeasureSlot],
    encode_expr: &dyn Fn(&datafusion_expr::Expr) -> Result<Vec<u8>>,
) -> Result<Vec<pb::MeasureSlotProto>> {
    let mut out = Vec::with_capacity(slots.len());
    for s in slots {
        let kind = match &s.kind {
            MeasureSlotKind::MatchNumber => pb::measure_slot_proto::Kind::MatchNumber(true),
            MeasureSlotKind::Classifier { symbol } => {
                pb::measure_slot_proto::Kind::Classifier(pb::ClassifierSlotProto {
                    symbol: symbol.clone(),
                })
            }
            MeasureSlotKind::Nav {
                logical_nav,
                logical_offset,
                physical_nav,
                physical_offset,
                symbol,
                inner_expr,
            } => pb::measure_slot_proto::Kind::Nav(pb::NavSlotProto {
                logical_nav: logical_nav.map(encode_logical_nav),
                logical_offset: *logical_offset,
                physical_nav: physical_nav.map(encode_physical_nav),
                physical_offset: *physical_offset,
                symbol: symbol.clone(),
                inner_expr: encode_expr(inner_expr)?,
            }),
            MeasureSlotKind::Agg {
                agg_kind,
                symbol,
                inner_expr,
                distinct,
            } => pb::measure_slot_proto::Kind::Agg(pb::AggSlotProto {
                agg_kind: encode_agg_kind(*agg_kind),
                symbol: symbol.clone(),
                inner_expr: inner_expr.as_ref().map(encode_expr).transpose()?,
                distinct: *distinct,
            }),
            MeasureSlotKind::CurrentRow { inner_expr } => {
                pb::measure_slot_proto::Kind::CurrentRow(pb::CurrentRowSlotProto {
                    inner_expr: encode_expr(inner_expr)?,
                })
            }
        };
        out.push(pb::MeasureSlotProto {
            slot_name: s.slot_name.clone(),
            semantics: encode_eval_semantics(s.semantics),
            kind: Some(kind),
        });
    }
    Ok(out)
}

impl LogicalExtensionCodec for MatchRecognizeLogicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[LogicalPlan],
        ctx: &TaskContext,
    ) -> Result<Extension> {
        let msg = match pb::MatchRecognizeLogicalNodeProto::decode(buf) {
            Ok(m) if m.pattern.is_some() => m,
            _ => {
                if let Some(delegate) = &self.delegate {
                    return delegate.try_decode(buf, inputs, ctx);
                }
                return internal_err!("Failed to decode MatchRecognizeLogicalNodeProto");
            }
        };
        if inputs.len() != 1 {
            return internal_err!(
                "MatchRecognizeLogicalCodec expects 1 input, got {}",
                inputs.len()
            );
        }
        let default_codec = DefaultLogicalExtensionCodec {};
        let decode_expr = |bytes: &[u8]| -> Result<datafusion_expr::Expr> {
            let node = LogicalExprNode::decode(bytes).map_err(|e| {
                DataFusionError::Internal(format!("Failed to decode LogicalExprNode: {e}"))
            })?;
            logical_from_proto::parse_expr(&node, ctx, &default_codec).map_err(|e| {
                DataFusionError::Internal(format!("Failed to parse logical expr: {e}"))
            })
        };

        let partition_by = msg
            .partition_by
            .iter()
            .map(|b| decode_expr(b))
            .collect::<Result<Vec<_>>>()?;

        let order_by = msg
            .order_by
            .iter()
            .map(|ob| {
                Ok(OrderBySpec {
                    expr: decode_expr(&ob.expr)?,
                    asc: ob.asc,
                    nulls_first: ob.nulls_first,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let subsets: Vec<SubsetDef> = msg
            .subsets
            .iter()
            .map(|s| SubsetDef {
                name: s.name.clone(),
                members: s.members.clone(),
            })
            .collect();

        let mut definitions = Vec::with_capacity(msg.definitions.len());
        for d in &msg.definitions {
            let kind = if let Some(st) = &d.stateful {
                SymbolDefKind::Stateful {
                    outer_expr: decode_expr(&st.outer_expr)?,
                    slots: decode_logical_slots(&st.slots, &decode_expr)?,
                }
            } else {
                SymbolDefKind::Stateless {
                    predicate: decode_expr(&d.predicate)?,
                }
            };
            definitions.push(SymbolDef {
                name: d.name.clone(),
                kind,
            });
        }

        let mut measures = Vec::with_capacity(msg.measures.len());
        for m in &msg.measures {
            measures.push(MeasureSpec {
                alias: m.alias.clone(),
                outer_expr: decode_expr(&m.outer_expr)?,
                slots: decode_logical_slots(&m.slots, &decode_expr)?,
            });
        }

        let pattern = decode_pattern(msg.pattern.as_ref().expect("checked is_some"))?;
        let rows_per_match = decode_rows_per_match(msg.rows_per_match)?;
        let after_match_skip = decode_skip(msg.after_match_skip.as_ref())?;
        let config = MatchRecognizeConfig {
            max_permute_elements: msg.max_permute_elements,
            max_nfa_states: msg.max_nfa_states,
            max_active_threads: if msg.max_active_threads == 0 {
                DEFAULT_MAX_ACTIVE_THREADS
            } else {
                msg.max_active_threads
            },
        };

        let node = MatchRecognizeNode::try_new_with_subsets(
            inputs[0].clone(),
            partition_by,
            order_by,
            measures,
            rows_per_match,
            after_match_skip,
            pattern,
            subsets,
            definitions,
            config,
        )?;
        Ok(Extension {
            node: Arc::new(node),
        })
    }

    fn try_encode(&self, node: &Extension, buf: &mut Vec<u8>) -> Result<()> {
        let Some(mr) = node.node.as_any().downcast_ref::<MatchRecognizeNode>() else {
            if let Some(delegate) = &self.delegate {
                return delegate.try_encode(node, buf);
            }
            return not_impl_err!("Unsupported logical extension node: {}", node.node.name());
        };
        let default_codec = DefaultLogicalExtensionCodec {};
        let encode_expr = |expr: &datafusion_expr::Expr| -> Result<Vec<u8>> {
            let proto_expr =
                logical_to_proto::serialize_expr(expr, &default_codec).map_err(|e| {
                    DataFusionError::Internal(format!("Failed to serialize logical expr: {e}"))
                })?;
            Ok(proto_expr.encode_to_vec())
        };

        let partition_by = mr
            .partition_by
            .iter()
            .map(encode_expr)
            .collect::<Result<Vec<_>>>()?;
        let order_by = mr
            .order_by
            .iter()
            .map(|o| {
                Ok(pb::OrderBySpecProto {
                    expr: encode_expr(&o.expr)?,
                    asc: o.asc,
                    nulls_first: o.nulls_first,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let subsets: Vec<pb::SubsetDefProto> = mr
            .subsets
            .iter()
            .map(|s| pb::SubsetDefProto {
                name: s.name.clone(),
                members: s.members.clone(),
            })
            .collect();

        let mut definitions = Vec::with_capacity(mr.definitions.len());
        for d in &mr.definitions {
            match &d.kind {
                SymbolDefKind::Stateless { predicate } => {
                    definitions.push(pb::SymbolDefProto {
                        name: d.name.clone(),
                        predicate: encode_expr(predicate)?,
                        stateful: None,
                    });
                }
                SymbolDefKind::Stateful { outer_expr, slots } => {
                    definitions.push(pb::SymbolDefProto {
                        name: d.name.clone(),
                        predicate: Vec::new(),
                        stateful: Some(pb::StatefulSymbolDefProto {
                            outer_expr: encode_expr(outer_expr)?,
                            slots: encode_logical_slots(slots, &encode_expr)?,
                        }),
                    });
                }
            }
        }

        let mut measures = Vec::with_capacity(mr.measures.len());
        for m in &mr.measures {
            measures.push(pb::MeasureSpecProto {
                alias: m.alias.clone(),
                outer_expr: encode_expr(&m.outer_expr)?,
                slots: encode_logical_slots(&m.slots, &encode_expr)?,
            });
        }

        let msg = pb::MatchRecognizeLogicalNodeProto {
            partition_by,
            order_by,
            measures,
            rows_per_match: encode_rows_per_match(mr.rows_per_match),
            after_match_skip: Some(encode_skip(&mr.after_match_skip)),
            pattern: Some(encode_pattern(&mr.pattern)),
            definitions,
            max_permute_elements: mr.config.max_permute_elements,
            max_nfa_states: mr.config.max_nfa_states,
            max_active_threads: mr.config.max_active_threads,
            subsets,
        };
        msg.encode(buf).map_err(|e| {
            DataFusionError::Internal(format!(
                "Failed to encode MatchRecognizeLogicalNodeProto: {e}"
            ))
        })
    }

    fn try_decode_table_provider(
        &self,
        buf: &[u8],
        table_ref: &TableReference,
        schema: Arc<Schema>,
        ctx: &TaskContext,
    ) -> Result<Arc<dyn TableProvider>> {
        if let Some(delegate) = &self.delegate {
            return delegate.try_decode_table_provider(buf, table_ref, schema, ctx);
        }
        not_impl_err!("MatchRecognizeLogicalCodec does not handle custom TableProvider")
    }

    fn try_encode_table_provider(
        &self,
        table_ref: &TableReference,
        node: Arc<dyn TableProvider>,
        buf: &mut Vec<u8>,
    ) -> Result<()> {
        if let Some(delegate) = &self.delegate {
            return delegate.try_encode_table_provider(table_ref, node, buf);
        }
        not_impl_err!("MatchRecognizeLogicalCodec does not handle custom TableProvider")
    }

    fn try_decode_file_format(
        &self,
        buf: &[u8],
        ctx: &TaskContext,
    ) -> Result<Arc<dyn FileFormatFactory>> {
        if let Some(delegate) = &self.delegate {
            return delegate.try_decode_file_format(buf, ctx);
        }
        not_impl_err!("MatchRecognizeLogicalCodec does not handle custom FileFormatFactory")
    }
}

/// [`PhysicalExtensionCodec`] for serializing and deserializing [`MatchRecognizeExec`].
#[derive(Debug, Default)]
pub struct MatchRecognizePhysicalCodec {
    delegate: Option<Arc<dyn PhysicalExtensionCodec>>,
}

impl MatchRecognizePhysicalCodec {
    /// Creates a codec handling [`MatchRecognizeExec`] without a fallback delegate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a codec that delegates non-`MatchRecognizeExec` physical nodes to `delegate`.
    pub fn with_delegate(delegate: Arc<dyn PhysicalExtensionCodec>) -> Self {
        Self {
            delegate: Some(delegate),
        }
    }
}

type PhysExprDecoder<'a> =
    &'a dyn Fn(&[u8], &Schema) -> Result<Arc<dyn datafusion_physical_expr::PhysicalExpr>>;
type PhysExprEncoder<'a> =
    &'a dyn Fn(&Arc<dyn datafusion_physical_expr::PhysicalExpr>) -> Result<Vec<u8>>;

fn decode_physical_slots(
    proto_slots: &[pb::MeasureSlotProto],
    input_schema: &Schema,
    program: &Program,
    unmatched: bool,
    decode_phys: PhysExprDecoder<'_>,
) -> Result<Vec<PhysicalMeasureSlot>> {
    let mut phys_slots = Vec::with_capacity(proto_slots.len());
    for s in proto_slots {
        let semantics = decode_eval_semantics(s.semantics);
        let (kind, data_type, nullable) = match s
            .kind
            .as_ref()
            .ok_or_else(|| DataFusionError::Internal("Missing MeasureSlotProto.kind".to_string()))?
        {
            pb::measure_slot_proto::Kind::MatchNumber(_) => (
                PhysicalMeasureSlotKind::MatchNumber,
                DataType::Int64,
                unmatched,
            ),
            pb::measure_slot_proto::Kind::Classifier(c) => (
                PhysicalMeasureSlotKind::Classifier {
                    symbol_set: c
                        .symbol
                        .as_ref()
                        .and_then(|sym| program.resolve_symbol_set(sym)),
                    symbol: c.symbol.clone(),
                },
                DataType::Utf8,
                true,
            ),
            pb::measure_slot_proto::Kind::Nav(n) => {
                let inner_expr = decode_phys(&n.inner_expr, input_schema)?;
                let dt = inner_expr.data_type(input_schema)?;
                (
                    PhysicalMeasureSlotKind::Nav {
                        logical_nav: n.logical_nav.map(decode_logical_nav),
                        logical_offset: n.logical_offset,
                        physical_nav: n.physical_nav.map(decode_physical_nav),
                        physical_offset: n.physical_offset,
                        symbol_set: n
                            .symbol
                            .as_ref()
                            .and_then(|sym| program.resolve_symbol_set(sym)),
                        symbol: n.symbol.clone(),
                        inner_expr,
                    },
                    dt,
                    true,
                )
            }
            pb::measure_slot_proto::Kind::Agg(a) => {
                let agg_kind = decode_agg_kind(a.agg_kind)?;
                let inner_expr = a
                    .inner_expr
                    .as_deref()
                    .map(|b| decode_phys(b, input_schema))
                    .transpose()?;
                let (dt, null) = match agg_kind {
                    AggKind::CountStar | AggKind::CountSymbol | AggKind::Count => {
                        (DataType::Int64, unmatched)
                    }
                    AggKind::Avg => (DataType::Float64, true),
                    AggKind::Sum => {
                        let inner_dt = inner_expr
                            .as_ref()
                            .expect("SUM has inner_expr")
                            .data_type(input_schema)?;
                        let sum_dt = match inner_dt {
                            DataType::Int8
                            | DataType::Int16
                            | DataType::Int32
                            | DataType::Int64 => DataType::Int64,
                            DataType::UInt8
                            | DataType::UInt16
                            | DataType::UInt32
                            | DataType::UInt64 => DataType::UInt64,
                            DataType::Float16 | DataType::Float32 | DataType::Float64 => {
                                DataType::Float64
                            }
                            DataType::Decimal128(p, s) => DataType::Decimal128((p + 10).min(38), s),
                            other => other,
                        };
                        (sum_dt, true)
                    }
                    AggKind::Min | AggKind::Max => {
                        let inner_dt = inner_expr
                            .as_ref()
                            .expect("MIN/MAX has inner_expr")
                            .data_type(input_schema)?;
                        (inner_dt, true)
                    }
                };
                (
                    PhysicalMeasureSlotKind::Agg {
                        agg_kind,
                        symbol_set: a
                            .symbol
                            .as_ref()
                            .and_then(|sym| program.resolve_symbol_set(sym)),
                        symbol: a.symbol.clone(),
                        inner_expr,
                        distinct: a.distinct,
                    },
                    dt,
                    null,
                )
            }
            pb::measure_slot_proto::Kind::CurrentRow(cr) => {
                let inner_expr = decode_phys(&cr.inner_expr, input_schema)?;
                let dt = inner_expr.data_type(input_schema)?;
                (PhysicalMeasureSlotKind::CurrentRow { inner_expr }, dt, true)
            }
        };
        phys_slots.push(PhysicalMeasureSlot {
            slot_name: s.slot_name.clone(),
            semantics,
            data_type,
            nullable,
            kind,
        });
    }
    Ok(phys_slots)
}

fn encode_physical_slots(
    slots: &[PhysicalMeasureSlot],
    encode_phys: PhysExprEncoder<'_>,
) -> Result<Vec<pb::MeasureSlotProto>> {
    let mut out = Vec::with_capacity(slots.len());
    for s in slots {
        let kind = match &s.kind {
            PhysicalMeasureSlotKind::MatchNumber => pb::measure_slot_proto::Kind::MatchNumber(true),
            PhysicalMeasureSlotKind::Classifier { symbol, .. } => {
                pb::measure_slot_proto::Kind::Classifier(pb::ClassifierSlotProto {
                    symbol: symbol.clone(),
                })
            }
            PhysicalMeasureSlotKind::Nav {
                logical_nav,
                logical_offset,
                physical_nav,
                physical_offset,
                symbol,
                inner_expr,
                ..
            } => pb::measure_slot_proto::Kind::Nav(pb::NavSlotProto {
                logical_nav: logical_nav.map(encode_logical_nav),
                logical_offset: *logical_offset,
                physical_nav: physical_nav.map(encode_physical_nav),
                physical_offset: *physical_offset,
                symbol: symbol.clone(),
                inner_expr: encode_phys(inner_expr)?,
            }),
            PhysicalMeasureSlotKind::Agg {
                agg_kind,
                symbol,
                inner_expr,
                distinct,
                ..
            } => pb::measure_slot_proto::Kind::Agg(pb::AggSlotProto {
                agg_kind: encode_agg_kind(*agg_kind),
                symbol: symbol.clone(),
                inner_expr: inner_expr.as_ref().map(encode_phys).transpose()?,
                distinct: *distinct,
            }),
            PhysicalMeasureSlotKind::CurrentRow { inner_expr } => {
                pb::measure_slot_proto::Kind::CurrentRow(pb::CurrentRowSlotProto {
                    inner_expr: encode_phys(inner_expr)?,
                })
            }
        };
        out.push(pb::MeasureSlotProto {
            slot_name: s.slot_name.clone(),
            semantics: encode_eval_semantics(s.semantics),
            kind: Some(kind),
        });
    }
    Ok(out)
}

impl PhysicalExtensionCodec for MatchRecognizePhysicalCodec {
    fn try_decode(
        &self,
        buf: &[u8],
        inputs: &[Arc<dyn ExecutionPlan>],
        ctx: &TaskContext,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let msg = match pb::MatchRecognizePhysicalNodeProto::decode(buf) {
            Ok(m) if m.pattern.is_some() => m,
            _ => {
                if let Some(delegate) = &self.delegate {
                    return delegate.try_decode(buf, inputs, ctx, proto_converter);
                }
                return internal_err!("Failed to decode MatchRecognizePhysicalNodeProto");
            }
        };
        if inputs.len() != 1 {
            return internal_err!(
                "MatchRecognizePhysicalCodec expects 1 input, got {}",
                inputs.len()
            );
        }
        let input = Arc::clone(&inputs[0]);
        let input_schema = input.schema();
        let default_codec = DefaultPhysicalExtensionCodec {};
        let decode_ctx = PhysicalPlanDecodeContext::new(ctx, &default_codec);

        let decode_phys = |bytes: &[u8],
                           schema: &Schema|
         -> Result<Arc<dyn datafusion_physical_expr::PhysicalExpr>> {
            let node = PhysicalExprNode::decode(bytes).map_err(|e| {
                DataFusionError::Internal(format!("Failed to decode PhysicalExprNode: {e}"))
            })?;
            phys_from_proto::parse_physical_expr_with_converter(
                &node,
                schema,
                &decode_ctx,
                proto_converter,
            )
        };

        let partition_by = msg
            .partition_by
            .iter()
            .map(|b| decode_phys(b, input_schema.as_ref()))
            .collect::<Result<Vec<_>>>()?;

        let order_by = msg
            .order_by
            .iter()
            .map(|ob| {
                let expr = decode_phys(&ob.expr, input_schema.as_ref())?;
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
            .collect::<Result<Vec<_>>>()?;

        let pattern = decode_pattern(msg.pattern.as_ref().expect("checked is_some"))?;
        let rows_per_match = decode_rows_per_match(msg.rows_per_match)?;
        let after_match_skip = decode_skip(msg.after_match_skip.as_ref())?;
        let subsets: Vec<SubsetDef> = msg
            .subsets
            .iter()
            .map(|s| SubsetDef {
                name: s.name.clone(),
                members: s.members.clone(),
            })
            .collect();
        let config = MatchRecognizeConfig {
            max_permute_elements: msg.max_permute_elements,
            max_nfa_states: msg.max_nfa_states,
            max_active_threads: if msg.max_active_threads == 0 {
                DEFAULT_MAX_ACTIVE_THREADS
            } else {
                msg.max_active_threads
            },
        };

        let declared_names: Vec<String> = msg.definitions.iter().map(|d| d.name.clone()).collect();
        let program = Program::compile_with_subsets(&pattern, &declared_names, &subsets, config)?;

        let mut definitions = Vec::with_capacity(msg.definitions.len());
        for d in &msg.definitions {
            let kind = if let Some(st) = &d.stateful {
                let slots = decode_physical_slots(
                    &st.slots,
                    input_schema.as_ref(),
                    &program,
                    false,
                    &decode_phys,
                )?;
                let slot_schema = build_physical_slot_schema(&slots);
                let outer_expr = decode_phys(&st.outer_expr, slot_schema.as_ref())?;
                PhysicalSymbolDefKind::Stateful {
                    outer_expr,
                    slot_schema,
                    slots,
                }
            } else {
                PhysicalSymbolDefKind::Stateless {
                    predicate: decode_phys(&d.predicate, input_schema.as_ref())?,
                }
            };
            definitions.push(PhysicalSymbolDef {
                name: d.name.clone(),
                kind,
            });
        }

        let unmatched = rows_per_match.include_unmatched_rows();
        let mut measures = Vec::with_capacity(msg.measures.len());
        for m in &msg.measures {
            let phys_slots = decode_physical_slots(
                &m.slots,
                input_schema.as_ref(),
                &program,
                unmatched,
                &decode_phys,
            )?;
            let slot_schema = build_physical_slot_schema(&phys_slots);
            let outer_expr = decode_phys(&m.outer_expr, slot_schema.as_ref())?;
            measures.push(PhysicalMeasure {
                alias: m.alias.clone(),
                outer_expr,
                slot_schema,
                slots: phys_slots,
            });
        }

        let output_schema = compute_physical_output_schema(
            input_schema.as_ref(),
            &partition_by,
            &order_by,
            &measures,
            rows_per_match,
        )?;

        Ok(Arc::new(MatchRecognizeExec::try_new_with_subsets(
            input,
            partition_by,
            order_by,
            measures,
            rows_per_match,
            after_match_skip,
            pattern,
            subsets,
            definitions,
            config,
            output_schema,
        )?))
    }

    fn try_encode(
        &self,
        node: Arc<dyn ExecutionPlan>,
        buf: &mut Vec<u8>,
        proto_converter: &dyn PhysicalProtoConverterExtension,
    ) -> Result<()> {
        let Some(exec) = node.downcast_ref::<MatchRecognizeExec>() else {
            if let Some(delegate) = &self.delegate {
                return delegate.try_encode(node, buf, proto_converter);
            }
            return not_impl_err!("Unsupported physical extension node: {}", node.name());
        };
        let default_codec = DefaultPhysicalExtensionCodec {};
        let encode_phys =
            |expr: &Arc<dyn datafusion_physical_expr::PhysicalExpr>| -> Result<Vec<u8>> {
                let proto_expr = phys_to_proto::serialize_physical_expr_with_converter(
                    expr,
                    &default_codec,
                    proto_converter,
                )?;
                Ok(proto_expr.encode_to_vec())
            };

        let partition_by = exec
            .partition_by()
            .iter()
            .map(encode_phys)
            .collect::<Result<Vec<_>>>()?;

        let order_by = exec
            .order_by()
            .iter()
            .map(|o| {
                Ok(pb::OrderBySpecProto {
                    expr: encode_phys(&o.expr)?,
                    asc: !o.options.descending,
                    nulls_first: o.options.nulls_first,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let subsets: Vec<pb::SubsetDefProto> = exec
            .subsets()
            .iter()
            .map(|s| pb::SubsetDefProto {
                name: s.name.clone(),
                members: s.members.clone(),
            })
            .collect();

        let mut definitions = Vec::with_capacity(exec.definitions().len());
        for d in exec.definitions() {
            match &d.kind {
                PhysicalSymbolDefKind::Stateless { predicate } => {
                    definitions.push(pb::SymbolDefProto {
                        name: d.name.clone(),
                        predicate: encode_phys(predicate)?,
                        stateful: None,
                    });
                }
                PhysicalSymbolDefKind::Stateful {
                    outer_expr, slots, ..
                } => {
                    definitions.push(pb::SymbolDefProto {
                        name: d.name.clone(),
                        predicate: Vec::new(),
                        stateful: Some(pb::StatefulSymbolDefProto {
                            outer_expr: encode_phys(outer_expr)?,
                            slots: encode_physical_slots(slots, &encode_phys)?,
                        }),
                    });
                }
            }
        }

        let mut measures = Vec::with_capacity(exec.measures().len());
        for m in exec.measures() {
            measures.push(pb::MeasureSpecProto {
                alias: m.alias.clone(),
                outer_expr: encode_phys(&m.outer_expr)?,
                slots: encode_physical_slots(&m.slots, &encode_phys)?,
            });
        }

        let msg = pb::MatchRecognizePhysicalNodeProto {
            partition_by,
            order_by,
            measures,
            rows_per_match: encode_rows_per_match(exec.rows_per_match()),
            after_match_skip: Some(encode_skip(exec.after_match_skip())),
            pattern: Some(encode_pattern(exec.pattern())),
            definitions,
            max_permute_elements: exec.config().max_permute_elements,
            max_nfa_states: exec.config().max_nfa_states,
            max_active_threads: exec.config().max_active_threads,
            subsets,
        };
        msg.encode(buf).map_err(|e| {
            DataFusionError::Internal(format!(
                "Failed to encode MatchRecognizePhysicalNodeProto: {e}"
            ))
        })
    }
}

fn compute_physical_output_schema(
    input_schema: &Schema,
    partition_by: &[Arc<dyn datafusion_physical_expr::PhysicalExpr>],
    order_by: &[PhysicalOrderBySpec],
    measures: &[PhysicalMeasure],
    rows_per_match: RowsPerMatch,
) -> Result<Arc<Schema>> {
    let mut fields = Vec::new();
    let mut included = HashSet::new();

    for p in partition_by {
        if let Some(col) = p.downcast_ref::<Column>() {
            if included.insert(col.index()) {
                let f = input_schema.field(col.index());
                fields.push(Field::new(f.name(), f.data_type().clone(), f.is_nullable()));
            }
        } else {
            let dt = p.data_type(input_schema)?;
            let nullable = p.nullable(input_schema)?;
            fields.push(Field::new(p.to_string(), dt, nullable));
        }
    }

    if !rows_per_match.is_one_row() {
        for ob in order_by {
            if let Some(idx) = ob.input_col_idx
                && included.insert(idx)
            {
                let f = input_schema.field(idx);
                fields.push(Field::new(f.name(), f.data_type().clone(), true));
            }
        }
    }

    for m in measures {
        let dt = m.outer_expr.data_type(m.slot_schema.as_ref())?;
        let mut nullable = m.outer_expr.nullable(m.slot_schema.as_ref())?;
        if rows_per_match.include_unmatched_rows() {
            nullable = true;
        }
        fields.push(Field::new(&m.alias, dt, nullable));
    }

    if !rows_per_match.is_one_row() {
        for (idx, f) in input_schema.fields().iter().enumerate() {
            if included.insert(idx) {
                fields.push(Field::new(f.name(), f.data_type().clone(), true));
            }
        }
    }

    let schema = Arc::new(Schema::new(fields));
    let _ = DFSchema::try_from(schema.as_ref().clone())?;
    Ok(schema)
}

fn encode_pattern(p: &Pattern) -> pb::PatternProto {
    let kind = match p {
        Pattern::Symbol(sym) => {
            let s_kind = match sym {
                SymbolRef::Named(name) => pb::symbol_ref_proto::Kind::Named(name.clone()),
                SymbolRef::Start => pb::symbol_ref_proto::Kind::Start(true),
                SymbolRef::End => pb::symbol_ref_proto::Kind::End(true),
            };
            pb::pattern_proto::Kind::Symbol(pb::SymbolRefProto { kind: Some(s_kind) })
        }
        Pattern::Exclude(inner) => {
            pb::pattern_proto::Kind::Exclude(Box::new(encode_pattern(inner)))
        }
        Pattern::Permute(items) => pb::pattern_proto::Kind::Permute(pb::PatternListProto {
            items: items.iter().map(encode_pattern).collect(),
        }),
        Pattern::Concat(items) => pb::pattern_proto::Kind::Concat(pb::PatternListProto {
            items: items.iter().map(encode_pattern).collect(),
        }),
        Pattern::Alternation(items) => pb::pattern_proto::Kind::Alternation(pb::PatternListProto {
            items: items.iter().map(encode_pattern).collect(),
        }),
        Pattern::Repetition {
            inner,
            min,
            max,
            reluctant,
        } => pb::pattern_proto::Kind::Repetition(Box::new(pb::RepetitionProto {
            inner: Some(Box::new(encode_pattern(inner))),
            min: *min,
            max: *max,
            reluctant: *reluctant,
        })),
    };
    pb::PatternProto { kind: Some(kind) }
}

fn decode_pattern(p: &pb::PatternProto) -> Result<Pattern> {
    let kind = p
        .kind
        .as_ref()
        .ok_or_else(|| DataFusionError::Internal("Missing PatternProto.kind".into()))?;
    match kind {
        pb::pattern_proto::Kind::Symbol(s) => {
            let sym =
                match s.kind.as_ref().ok_or_else(|| {
                    DataFusionError::Internal("Missing SymbolRefProto.kind".into())
                })? {
                    pb::symbol_ref_proto::Kind::Named(n) => SymbolRef::Named(n.clone()),
                    pb::symbol_ref_proto::Kind::Start(_) => SymbolRef::Start,
                    pb::symbol_ref_proto::Kind::End(_) => SymbolRef::End,
                };
            Ok(Pattern::Symbol(sym))
        }
        pb::pattern_proto::Kind::Exclude(inner) => {
            Ok(Pattern::Exclude(Box::new(decode_pattern(inner)?)))
        }
        pb::pattern_proto::Kind::Permute(list) => Ok(Pattern::Permute(
            list.items
                .iter()
                .map(decode_pattern)
                .collect::<Result<_>>()?,
        )),
        pb::pattern_proto::Kind::Concat(list) => Ok(Pattern::Concat(
            list.items
                .iter()
                .map(decode_pattern)
                .collect::<Result<_>>()?,
        )),
        pb::pattern_proto::Kind::Alternation(list) => Ok(Pattern::Alternation(
            list.items
                .iter()
                .map(decode_pattern)
                .collect::<Result<_>>()?,
        )),
        pb::pattern_proto::Kind::Repetition(rep) => {
            let inner = decode_pattern(rep.inner.as_deref().ok_or_else(|| {
                DataFusionError::Internal("Missing RepetitionProto.inner".into())
            })?)?;
            Ok(Pattern::Repetition {
                inner: Box::new(inner),
                min: rep.min,
                max: rep.max,
                reluctant: rep.reluctant,
            })
        }
    }
}

fn encode_rows_per_match(r: RowsPerMatch) -> i32 {
    match r {
        RowsPerMatch::OneRow => pb::RowsPerMatchProto::RowsPerMatchOneRow as i32,
        RowsPerMatch::AllRowsShowEmpty => {
            pb::RowsPerMatchProto::RowsPerMatchAllRowsShowEmpty as i32
        }
        RowsPerMatch::AllRowsOmitEmpty => {
            pb::RowsPerMatchProto::RowsPerMatchAllRowsOmitEmpty as i32
        }
        RowsPerMatch::AllRowsWithUnmatched => {
            pb::RowsPerMatchProto::RowsPerMatchAllRowsWithUnmatched as i32
        }
    }
}

fn decode_rows_per_match(v: i32) -> Result<RowsPerMatch> {
    match pb::RowsPerMatchProto::try_from(v) {
        Ok(pb::RowsPerMatchProto::RowsPerMatchOneRow) => Ok(RowsPerMatch::OneRow),
        Ok(pb::RowsPerMatchProto::RowsPerMatchAllRowsShowEmpty) => {
            Ok(RowsPerMatch::AllRowsShowEmpty)
        }
        Ok(pb::RowsPerMatchProto::RowsPerMatchAllRowsOmitEmpty) => {
            Ok(RowsPerMatch::AllRowsOmitEmpty)
        }
        Ok(pb::RowsPerMatchProto::RowsPerMatchAllRowsWithUnmatched) => {
            Ok(RowsPerMatch::AllRowsWithUnmatched)
        }
        Err(_) => internal_err!("Invalid RowsPerMatchProto enum value: {v}"),
    }
}

fn encode_skip(s: &AfterMatchSkip) -> pb::AfterMatchSkipProto {
    let kind = match s {
        AfterMatchSkip::PastLastRow => pb::after_match_skip_proto::Kind::PastLastRow(true),
        AfterMatchSkip::ToNextRow => pb::after_match_skip_proto::Kind::ToNextRow(true),
        AfterMatchSkip::ToFirst(v) => pb::after_match_skip_proto::Kind::ToFirst(v.clone()),
        AfterMatchSkip::ToLast(v) => pb::after_match_skip_proto::Kind::ToLast(v.clone()),
    };
    pb::AfterMatchSkipProto { kind: Some(kind) }
}

fn decode_skip(s: Option<&pb::AfterMatchSkipProto>) -> Result<AfterMatchSkip> {
    let Some(skip) = s else {
        return Ok(AfterMatchSkip::PastLastRow);
    };
    match skip.kind.as_ref() {
        None | Some(pb::after_match_skip_proto::Kind::PastLastRow(_)) => {
            Ok(AfterMatchSkip::PastLastRow)
        }
        Some(pb::after_match_skip_proto::Kind::ToNextRow(_)) => Ok(AfterMatchSkip::ToNextRow),
        Some(pb::after_match_skip_proto::Kind::ToFirst(v)) => {
            Ok(AfterMatchSkip::ToFirst(v.clone()))
        }
        Some(pb::after_match_skip_proto::Kind::ToLast(v)) => Ok(AfterMatchSkip::ToLast(v.clone())),
    }
}

fn encode_logical_nav(n: LogicalNavKind) -> i32 {
    match n {
        LogicalNavKind::First => pb::LogicalNavKindProto::LogicalNavFirst as i32,
        LogicalNavKind::Last => pb::LogicalNavKindProto::LogicalNavLast as i32,
    }
}

fn decode_logical_nav(v: i32) -> LogicalNavKind {
    match pb::LogicalNavKindProto::try_from(v) {
        Ok(pb::LogicalNavKindProto::LogicalNavFirst) => LogicalNavKind::First,
        _ => LogicalNavKind::Last,
    }
}

fn encode_physical_nav(n: PhysicalNavKind) -> i32 {
    match n {
        PhysicalNavKind::Prev => pb::PhysicalNavKindProto::PhysicalNavPrev as i32,
        PhysicalNavKind::Next => pb::PhysicalNavKindProto::PhysicalNavNext as i32,
    }
}

fn decode_physical_nav(v: i32) -> PhysicalNavKind {
    match pb::PhysicalNavKindProto::try_from(v) {
        Ok(pb::PhysicalNavKindProto::PhysicalNavPrev) => PhysicalNavKind::Prev,
        _ => PhysicalNavKind::Next,
    }
}

fn encode_agg_kind(a: AggKind) -> i32 {
    match a {
        AggKind::CountStar => pb::AggKindProto::AggCountStar as i32,
        AggKind::CountSymbol => pb::AggKindProto::AggCountSymbol as i32,
        AggKind::Count => pb::AggKindProto::AggCount as i32,
        AggKind::Sum => pb::AggKindProto::AggSum as i32,
        AggKind::Avg => pb::AggKindProto::AggAvg as i32,
        AggKind::Min => pb::AggKindProto::AggMin as i32,
        AggKind::Max => pb::AggKindProto::AggMax as i32,
    }
}

fn decode_agg_kind(v: i32) -> Result<AggKind> {
    match pb::AggKindProto::try_from(v) {
        Ok(pb::AggKindProto::AggCountStar) => Ok(AggKind::CountStar),
        Ok(pb::AggKindProto::AggCountSymbol) => Ok(AggKind::CountSymbol),
        Ok(pb::AggKindProto::AggCount) => Ok(AggKind::Count),
        Ok(pb::AggKindProto::AggSum) => Ok(AggKind::Sum),
        Ok(pb::AggKindProto::AggAvg) => Ok(AggKind::Avg),
        Ok(pb::AggKindProto::AggMin) => Ok(AggKind::Min),
        Ok(pb::AggKindProto::AggMax) => Ok(AggKind::Max),
        Err(_) => internal_err!("Invalid AggKindProto enum value: {v}"),
    }
}

fn encode_eval_semantics(s: EvalSemantics) -> i32 {
    match s {
        EvalSemantics::Running => pb::EvalSemanticsProto::EvalSemanticsRunning as i32,
        EvalSemantics::Final => pb::EvalSemanticsProto::EvalSemanticsFinal as i32,
    }
}

fn decode_eval_semantics(v: i32) -> EvalSemantics {
    match pb::EvalSemanticsProto::try_from(v) {
        Ok(pb::EvalSemanticsProto::EvalSemanticsFinal) => EvalSemantics::Final,
        _ => EvalSemantics::Running,
    }
}
