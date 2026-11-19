// SPDX-License-Identifier: Apache-2.0

//! Validated internal IR for SQL:2016 `MATCH_RECOGNIZE`.

use std::fmt;
use std::sync::Arc;

use datafusion_expr::{AggregateUDF, Expr};

/// Default safety limit on the number of elements inside `PERMUTE(...)`.
pub const DEFAULT_MAX_PERMUTE_ELEMENTS: u32 = 8;

/// Default safety limit on the number of compiled NFA bytecode instructions.
pub const DEFAULT_MAX_NFA_STATES: u32 = 16_384;

/// Default safety limit on concurrent active NFA threads during stateful `DEFINE` matching.
pub const DEFAULT_MAX_ACTIVE_THREADS: u32 = 4_096;

/// Configuration knobs for `MATCH_RECOGNIZE` compilation and execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatchRecognizeConfig {
    /// Maximum number of elements allowed in `PERMUTE(...)` (factorial expansion).
    pub max_permute_elements: u32,
    /// Maximum number of bytecode instructions allowed in a compiled NFA program.
    pub max_nfa_states: u32,
    /// Maximum concurrent NFA threads allowed in stateful `DEFINE` mode before erroring.
    pub max_active_threads: u32,
}

impl Default for MatchRecognizeConfig {
    fn default() -> Self {
        Self {
            max_permute_elements: DEFAULT_MAX_PERMUTE_ELEMENTS,
            max_nfa_states: DEFAULT_MAX_NFA_STATES,
            max_active_threads: DEFAULT_MAX_ACTIVE_THREADS,
        }
    }
}

/// A pattern symbol reference (named variable or partition anchor).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum SymbolRef {
    Named(String),
    Start,
    End,
}

impl fmt::Display for SymbolRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymbolRef::Named(name) => write!(f, "{name}"),
            SymbolRef::Start => write!(f, "^"),
            SymbolRef::End => write!(f, "$"),
        }
    }
}

/// Internal row-pattern AST.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Pattern {
    Symbol(SymbolRef),
    Exclude(Box<Pattern>),
    Permute(Vec<Pattern>),
    Concat(Vec<Pattern>),
    Alternation(Vec<Pattern>),
    Repetition {
        inner: Box<Pattern>,
        min: u32,
        max: Option<u32>,
        reluctant: bool,
    },
}

impl Pattern {
    /// Collects all unique named symbols referenced by this pattern in declaration order.
    pub fn collect_named_symbols(&self, out: &mut Vec<String>) {
        match self {
            Pattern::Symbol(SymbolRef::Named(name)) => {
                if !out.contains(name) {
                    out.push(name.clone());
                }
            }
            Pattern::Symbol(SymbolRef::Start | SymbolRef::End) => {}
            Pattern::Exclude(inner) => inner.collect_named_symbols(out),
            Pattern::Permute(items) | Pattern::Concat(items) | Pattern::Alternation(items) => {
                for item in items {
                    item.collect_named_symbols(out);
                }
            }
            Pattern::Repetition { inner, .. } => inner.collect_named_symbols(out),
        }
    }

    /// Returns true if this pattern contains any `{- ... -}` exclusion.
    pub fn has_exclusion(&self) -> bool {
        match self {
            Pattern::Symbol(_) => false,
            Pattern::Exclude(_) => true,
            Pattern::Permute(items) | Pattern::Concat(items) | Pattern::Alternation(items) => {
                items.iter().any(Pattern::has_exclusion)
            }
            Pattern::Repetition { inner, .. } => inner.has_exclusion(),
        }
    }
}

impl fmt::Display for Pattern {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Pattern::Symbol(sym) => write!(f, "{sym}"),
            Pattern::Exclude(inner) => write!(f, "{{- {inner} -}}"),
            Pattern::Permute(items) => {
                write!(f, "PERMUTE(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
            Pattern::Concat(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
            Pattern::Alternation(items) => {
                write!(f, "(")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, " | ")?;
                    }
                    write!(f, "{item}")?;
                }
                write!(f, ")")
            }
            Pattern::Repetition {
                inner,
                min,
                max,
                reluctant,
            } => {
                write!(f, "({inner})")?;
                match (min, max) {
                    (0, None) => write!(f, "*")?,
                    (1, None) => write!(f, "+")?,
                    (0, Some(1)) => write!(f, "?")?,
                    (n, Some(m)) if n == m => write!(f, "{{{n}}}")?,
                    (n, None) => write!(f, "{{{n},}}")?,
                    (0, Some(m)) => write!(f, "{{,{m}}}")?,
                    (n, Some(m)) => write!(f, "{{{n},{m}}}")?,
                }
                if *reluctant {
                    write!(f, "?")?;
                }
                Ok(())
            }
        }
    }
}

/// Output row mode for `MATCH_RECOGNIZE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RowsPerMatch {
    #[default]
    OneRow,
    AllRowsShowEmpty,
    AllRowsOmitEmpty,
    AllRowsWithUnmatched,
}

impl RowsPerMatch {
    pub fn is_one_row(self) -> bool {
        matches!(self, RowsPerMatch::OneRow)
    }

    pub fn show_empty_matches(self) -> bool {
        !matches!(self, RowsPerMatch::AllRowsOmitEmpty)
    }

    pub fn include_unmatched_rows(self) -> bool {
        matches!(self, RowsPerMatch::AllRowsWithUnmatched)
    }
}

impl fmt::Display for RowsPerMatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RowsPerMatch::OneRow => write!(f, "ONE ROW PER MATCH"),
            RowsPerMatch::AllRowsShowEmpty => write!(f, "ALL ROWS PER MATCH SHOW EMPTY MATCHES"),
            RowsPerMatch::AllRowsOmitEmpty => write!(f, "ALL ROWS PER MATCH OMIT EMPTY MATCHES"),
            RowsPerMatch::AllRowsWithUnmatched => {
                write!(f, "ALL ROWS PER MATCH WITH UNMATCHED ROWS")
            }
        }
    }
}

/// `AFTER MATCH SKIP` resumption policy.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum AfterMatchSkip {
    #[default]
    PastLastRow,
    ToNextRow,
    ToFirst(String),
    ToLast(String),
}

impl fmt::Display for AfterMatchSkip {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AfterMatchSkip::PastLastRow => write!(f, "AFTER MATCH SKIP PAST LAST ROW"),
            AfterMatchSkip::ToNextRow => write!(f, "AFTER MATCH SKIP TO NEXT ROW"),
            AfterMatchSkip::ToFirst(s) => write!(f, "AFTER MATCH SKIP TO FIRST {s}"),
            AfterMatchSkip::ToLast(s) => write!(f, "AFTER MATCH SKIP TO LAST {s}"),
        }
    }
}

/// SQL:2016 `SUBSET <name> = (<member1>, <member2>, ...)` union variable definition.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SubsetDef {
    pub name: String,
    pub members: Vec<String>,
}

/// Logical navigation function within a match (`FIRST` or `LAST`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum LogicalNavKind {
    First,
    Last,
}

/// Physical offset navigation within a partition (`PREV` or `NEXT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PhysicalNavKind {
    Prev,
    Next,
}

/// Aggregate functions supported inside `MEASURES` and stateful `DEFINE`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum AggKind {
    CountStar,
    CountSymbol,
    Count,
    Sum,
    Avg,
    Min,
    Max,
    /// Any registered DataFusion [`AggregateUDF`] evaluated via its [`datafusion_expr::Accumulator`].
    Udaf(Arc<AggregateUDF>),
}

/// Evaluation horizon for a measure or define leaf (`RUNNING` vs `FINAL`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EvalSemantics {
    Running,
    Final,
}

/// A leaf pattern-recognition sub-expression extracted from a `MEASURES` or stateful `DEFINE` expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum MeasureSlotKind {
    /// `MATCH_NUMBER()` returning 1-based `Int64` match index per partition.
    MatchNumber,
    /// `CLASSIFIER([symbol_or_subset])`, optionally wrapped in `FIRST`/`LAST`/`PREV`/`NEXT`.
    Classifier {
        symbol: Option<String>,
        logical_nav: Option<LogicalNavKind>,
        logical_offset: u32,
        physical_nav: Option<PhysicalNavKind>,
        physical_offset: u32,
    },
    /// Pattern navigation (`FIRST`/`LAST`, `PREV`/`NEXT`, or bare `A.col`).
    Nav {
        logical_nav: Option<LogicalNavKind>,
        logical_offset: u32,
        physical_nav: Option<PhysicalNavKind>,
        physical_offset: u32,
        symbol: Option<String>,
        inner_expr: Expr,
    },
    /// Aggregate over matched rows (built-in fast path or any DataFusion `AggregateUDF`),
    /// optionally scoped to a pattern variable/subset and optionally with `PREV`/`NEXT` shifted args.
    Agg {
        agg_kind: AggKind,
        symbol: Option<String>,
        args: Vec<Expr>,
        distinct: bool,
        arg_physical_nav: Option<(PhysicalNavKind, u32)>,
    },
    /// Bare input-column reference evaluated on the current row.
    CurrentRow { inner_expr: Expr },
}

/// Named synthetic slot for a pattern-recognition leaf expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct MeasureSlot {
    pub slot_name: String,
    pub semantics: EvalSemantics,
    pub kind: MeasureSlotKind,
}

/// A single `MEASURES <expr> AS <alias>` definition decomposed into slots + outer scalar `Expr`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct MeasureSpec {
    pub alias: String,
    pub outer_expr: Expr,
    pub slots: Vec<MeasureSlot>,
}

/// Classification of a `DEFINE <name> AS <predicate>` expression.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub enum SymbolDefKind {
    Stateless {
        predicate: Expr,
    },
    Stateful {
        outer_expr: Expr,
        slots: Vec<MeasureSlot>,
    },
}

/// A single `DEFINE <name> AS <predicate>` entry.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct SymbolDef {
    pub name: String,
    pub kind: SymbolDefKind,
}

impl SymbolDef {
    pub fn stateless(name: impl Into<String>, predicate: Expr) -> Self {
        Self {
            name: name.into(),
            kind: SymbolDefKind::Stateless { predicate },
        }
    }

    pub fn stateful(name: impl Into<String>, outer_expr: Expr, slots: Vec<MeasureSlot>) -> Self {
        Self {
            name: name.into(),
            kind: SymbolDefKind::Stateful { outer_expr, slots },
        }
    }

    pub fn is_stateful(&self) -> bool {
        matches!(self.kind, SymbolDefKind::Stateful { .. })
    }
}

/// A single `ORDER BY` specification.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct OrderBySpec {
    pub expr: Expr,
    pub asc: bool,
    pub nulls_first: bool,
}
