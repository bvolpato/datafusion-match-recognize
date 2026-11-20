// SPDX-License-Identifier: Apache-2.0

//! Compiles a [`Pattern`] AST into priority-ordered NFA bytecode, resolves `SUBSET` union
//! variables into dense [`SymbolSet`] bitmasks, and precomputes start-symbol masks for
//! O(1) row fast-skipping.

use datafusion_common::{HashMap, Result, plan_err};

use crate::ast::{MatchRecognizeConfig, Pattern, SubsetDef, SymbolRef};

/// Dense 0-based runtime identifier for a primary pattern variable.
pub type SymbolId = u16;

/// Set of primary [`SymbolId`]s matched by either a single primary variable or a `SUBSET` union.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolSet {
    /// Single primary variable (fast equality check).
    Single(SymbolId),
    /// Bitmask for up to 64 primary variables (`SUBSET` fast path).
    Mask64(u64),
    /// General sorted list for > 64 primary variables.
    List(Vec<SymbolId>),
}

impl SymbolSet {
    /// Creates a [`SymbolSet`] from a slice of member [`SymbolId`]s.
    pub fn from_ids(ids: &[SymbolId]) -> Self {
        if ids.len() == 1 {
            return Self::Single(ids[0]);
        }
        if ids.iter().all(|&id| (id as usize) < 64) {
            let mut mask = 0u64;
            for &id in ids {
                mask |= 1u64 << id;
            }
            Self::Mask64(mask)
        } else {
            let mut sorted = ids.to_vec();
            sorted.sort_unstable();
            sorted.dedup();
            Self::List(sorted)
        }
    }

    /// Returns true if `sym` belongs to this primary variable or subset.
    #[inline]
    pub fn contains(&self, sym: SymbolId) -> bool {
        match self {
            SymbolSet::Single(id) => *id == sym,
            SymbolSet::Mask64(mask) => {
                (sym as usize) < 64 && (*mask & (1u64 << sym)) != 0
            }
            SymbolSet::List(ids) => ids.binary_search(&sym).is_ok(),
        }
    }
}

/// A single NFA bytecode instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Inst {
    /// Match the current row against `sym` and advance to `pc + 1`.
    MatchLabel { sym: SymbolId },
    /// Priority-ordered epsilon fork: explore `first` before `second`.
    Split { first: usize, second: usize },
    /// Unconditional epsilon jump.
    Jump(usize),
    /// Succeed via epsilon only at row index `0` (partition start, `^`).
    AnchorStart,
    /// Succeed via epsilon only at row index `partition_len` (partition end, `$`).
    AnchorEnd,
    /// Enter a `{- ... -}` exclusion scope.
    ExcludeEnter,
    /// Exit a `{- ... -}` exclusion scope.
    ExcludeExit,
    /// Match completed.
    Accept,
}

/// Compiled immutable NFA program ready for execution by [`crate::matcher::Matcher`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Program {
    /// Bytecode instruction stream.
    pub instructions: Vec<Inst>,
    /// Index-to-name primary symbol table (`id_to_symbol[sym as usize]`).
    pub id_to_symbol: Vec<String>,
    /// Name-to-index primary symbol lookup.
    pub symbol_to_id: HashMap<String, SymbolId>,
    /// Resolved `SUBSET` union variables (`subset_name -> SymbolSet`).
    pub subsets: HashMap<String, SymbolSet>,
    /// Optional bitmask of all symbols that can match the very first row of a match.
    ///
    /// Populated only when the pattern cannot match empty, contains no `^`/`$` anchors on the
    /// start frontier, and has at most 64 primary symbols. Enables O(1) row fast-skipping when
    /// all symbols are stateless.
    pub start_symbols_mask: Option<u64>,
    /// Optional bitmask of symbols reachable at `start_row + 1` when `instructions[0]` is an
    /// unconditional `MatchLabel` and `pc = 1` cannot accept or hit an anchor without consuming
    /// a second row (e.g. `A B+ C` or `STRT DOWN+ UP+`). Enables O(1) two-row lookahead skip.
    pub second_symbols_mask: Option<u64>,
    /// Bitmask of primary symbols (when `num_symbols <= 64`) that must appear at least once in
    /// any valid match. Enables O(N) whole-partition rejection when a mandatory symbol is absent.
    pub required_symbols_mask: u64,
    /// Number of `Inst::MatchLabel` instructions in `instructions`.
    pub num_match_labels: usize,
    /// Maps each instruction `pc` to its dense `0..num_match_labels` ordinal (`u16::MAX` for
    /// non-`MatchLabel` instructions). Used by lazy cross-attempt dead-state memoization.
    pub pc_to_match_ordinal: Vec<u16>,
}

impl Program {
    /// Compiles a [`Pattern`] with declared symbols and no `SUBSET` union variables.
    pub fn compile(
        pattern: &Pattern,
        declared_symbols: &[String],
        config: MatchRecognizeConfig,
    ) -> Result<Self> {
        Self::compile_with_subsets(pattern, declared_symbols, &[], config)
    }

    /// Compiles a [`Pattern`] with declared symbols and optional `SUBSET` union variables.
    ///
    /// Any pattern variable referenced in `pattern` that is not in `declared_symbols`
    /// is automatically registered (implicit `TRUE` definition per SQL:2016).
    pub fn compile_with_subsets(
        pattern: &Pattern,
        declared_symbols: &[String],
        subset_defs: &[SubsetDef],
        config: MatchRecognizeConfig,
    ) -> Result<Self> {
        let mut id_to_symbol = Vec::new();
        let mut symbol_to_id = HashMap::new();

        for name in declared_symbols {
            register_symbol(name, &mut id_to_symbol, &mut symbol_to_id)?;
        }
        let mut referenced = Vec::new();
        pattern.collect_named_symbols(&mut referenced);
        for name in &referenced {
            register_symbol(name, &mut id_to_symbol, &mut symbol_to_id)?;
        }

        // Validate and resolve SUBSET union variables.
        let mut subsets: HashMap<String, SymbolSet> =
            HashMap::with_capacity(subset_defs.len());
        for sdef in subset_defs {
            if symbol_to_id.contains_key(&sdef.name) {
                return plan_err!(
                    "SUBSET variable '{}' conflicts with primary pattern variable",
                    sdef.name
                );
            }
            if subsets.contains_key(&sdef.name) {
                return plan_err!("Duplicate SUBSET variable '{}'", sdef.name);
            }
            if sdef.members.is_empty() {
                return plan_err!(
                    "SUBSET variable '{}' must reference at least one primary variable",
                    sdef.name
                );
            }
            let mut member_ids = Vec::with_capacity(sdef.members.len());
            for member in &sdef.members {
                let id = *symbol_to_id.get(member).ok_or_else(|| {
                    datafusion_common::plan_datafusion_err!(
                        "SUBSET '{}' references unknown primary pattern variable '{member}'",
                        sdef.name
                    )
                })?;
                member_ids.push(id);
            }
            subsets.insert(sdef.name.clone(), SymbolSet::from_ids(&member_ids));
        }

        let mut compiler = Compiler {
            instructions: Vec::new(),
            symbol_to_id: &symbol_to_id,
            max_permute_elements: config.max_permute_elements as usize,
            max_nfa_states: config.max_nfa_states as usize,
        };
        compiler.compile_pattern(pattern)?;
        compiler.push(Inst::Accept)?;

        let start_symbols_mask =
            compute_frontier_mask(&compiler.instructions, 0, id_to_symbol.len());
        let second_symbols_mask =
            if matches!(compiler.instructions.first(), Some(Inst::MatchLabel { .. })) {
                compute_frontier_mask(&compiler.instructions, 1, id_to_symbol.len())
            } else {
                None
            };
        let required_symbols_mask = if id_to_symbol.len() <= 64 {
            compute_required_symbols_mask(pattern, &symbol_to_id)
        } else {
            0
        };
        let mut pc_to_match_ordinal = vec![u16::MAX; compiler.instructions.len()];
        let mut num_match_labels = 0usize;
        for (pc, inst) in compiler.instructions.iter().enumerate() {
            if matches!(inst, Inst::MatchLabel { .. }) {
                if let Ok(ord) = u16::try_from(num_match_labels) {
                    pc_to_match_ordinal[pc] = ord;
                }
                num_match_labels += 1;
            }
        }

        Ok(Program {
            instructions: compiler.instructions,
            id_to_symbol,
            symbol_to_id,
            subsets,
            start_symbols_mask,
            second_symbols_mask,
            required_symbols_mask,
            num_match_labels,
            pc_to_match_ordinal,
        })
    }

    /// Returns the number of registered primary pattern symbols.
    pub fn num_symbols(&self) -> usize {
        self.id_to_symbol.len()
    }

    /// Looks up a primary symbol's dense runtime id by normalized name.
    pub fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbol_to_id.get(name).copied()
    }

    /// Resolves either a primary symbol name or a `SUBSET` union variable name to a [`SymbolSet`].
    pub fn resolve_symbol_set(&self, name: &str) -> Option<SymbolSet> {
        if let Some(id) = self.symbol_to_id.get(name).copied() {
            return Some(SymbolSet::Single(id));
        }
        self.subsets.get(name).cloned()
    }
}

/// Computes the bitmask of primary symbols reachable on the next consuming step from `entry_pc`.
///
/// Returns `None` if `Accept` is reachable without consuming a row (nullable), if an anchor
/// (`^`/`$`) lies on the frontier, or if there are more than 64 symbols.
fn compute_frontier_mask(
    instructions: &[Inst],
    entry_pc: usize,
    num_symbols: usize,
) -> Option<u64> {
    if num_symbols == 0 || num_symbols > 64 || entry_pc >= instructions.len() {
        return None;
    }
    let mut visited = vec![false; instructions.len()];
    let mut stack = vec![entry_pc];
    let mut mask = 0u64;

    while let Some(pc) = stack.pop() {
        if visited[pc] {
            continue;
        }
        visited[pc] = true;
        match instructions[pc] {
            Inst::MatchLabel { sym } => {
                mask |= 1u64 << sym;
            }
            Inst::Jump(target) => stack.push(target),
            Inst::Split { first, second } => {
                stack.push(second);
                stack.push(first);
            }
            Inst::ExcludeEnter | Inst::ExcludeExit => stack.push(pc + 1),
            // Anchors or immediate Accept can succeed/fail without matching a symbol mask.
            Inst::AnchorStart | Inst::AnchorEnd | Inst::Accept => return None,
        }
    }

    Some(mask)
}

/// Computes the bitmask of primary symbols that MUST appear at least once in any match of `pattern`.
fn compute_required_symbols_mask(
    pattern: &Pattern,
    symbol_to_id: &HashMap<String, SymbolId>,
) -> u64 {
    match pattern {
        Pattern::Symbol(SymbolRef::Named(name)) => symbol_to_id
            .get(name)
            .map_or(0, |&id| 1u64 << (id as usize)),
        Pattern::Symbol(SymbolRef::Start | SymbolRef::End) => 0,
        Pattern::Exclude(inner) => compute_required_symbols_mask(inner, symbol_to_id),
        Pattern::Concat(items) | Pattern::Permute(items) => {
            let mut mask = 0u64;
            for item in items {
                mask |= compute_required_symbols_mask(item, symbol_to_id);
            }
            mask
        }
        Pattern::Alternation(branches) => {
            let Some((first, rest)) = branches.split_first() else {
                return 0;
            };
            let mut mask = compute_required_symbols_mask(first, symbol_to_id);
            for b in rest {
                mask &= compute_required_symbols_mask(b, symbol_to_id);
            }
            mask
        }
        Pattern::Repetition { inner, min, .. } => {
            if *min >= 1 {
                compute_required_symbols_mask(inner, symbol_to_id)
            } else {
                0
            }
        }
    }
}

fn register_symbol(
    name: &str,
    id_to_symbol: &mut Vec<String>,
    symbol_to_id: &mut HashMap<String, SymbolId>,
) -> Result<()> {
    if symbol_to_id.contains_key(name) {
        return Ok(());
    }
    let id = SymbolId::try_from(id_to_symbol.len()).map_err(|_| {
        datafusion_common::plan_datafusion_err!("Too many pattern variables")
    })?;
    id_to_symbol.push(name.to_string());
    symbol_to_id.insert(name.to_string(), id);
    Ok(())
}

struct Compiler<'a> {
    instructions: Vec<Inst>,
    symbol_to_id: &'a HashMap<String, SymbolId>,
    max_permute_elements: usize,
    max_nfa_states: usize,
}

impl Compiler<'_> {
    fn push(&mut self, inst: Inst) -> Result<usize> {
        if self.instructions.len() >= self.max_nfa_states {
            return plan_err!(
                "MATCH_RECOGNIZE pattern exceeds maximum NFA instruction limit ({})",
                self.max_nfa_states
            );
        }
        let pc = self.instructions.len();
        self.instructions.push(inst);
        Ok(pc)
    }

    fn compile_pattern(&mut self, pattern: &Pattern) -> Result<()> {
        match pattern {
            Pattern::Symbol(SymbolRef::Named(name)) => {
                let sym = *self.symbol_to_id.get(name).ok_or_else(|| {
                    datafusion_common::plan_datafusion_err!(
                        "Unknown pattern variable '{name}'"
                    )
                })?;
                self.push(Inst::MatchLabel { sym })?;
            }
            Pattern::Symbol(SymbolRef::Start) => {
                self.push(Inst::AnchorStart)?;
            }
            Pattern::Symbol(SymbolRef::End) => {
                self.push(Inst::AnchorEnd)?;
            }
            Pattern::Exclude(inner) => {
                self.push(Inst::ExcludeEnter)?;
                self.compile_pattern(inner)?;
                self.push(Inst::ExcludeExit)?;
            }
            Pattern::Concat(items) => {
                for item in items {
                    self.compile_pattern(item)?;
                }
            }
            Pattern::Alternation(branches) => {
                self.compile_alternation(branches)?;
            }
            Pattern::Permute(items) => {
                self.compile_permute(items)?;
            }
            Pattern::Repetition {
                inner,
                min,
                max,
                reluctant,
            } => {
                self.compile_repetition(inner, *min, *max, *reluctant)?;
            }
        }
        Ok(())
    }

    fn compile_alternation(&mut self, branches: &[Pattern]) -> Result<()> {
        if branches.is_empty() {
            return Ok(());
        }
        if branches.len() == 1 {
            return self.compile_pattern(&branches[0]);
        }

        let mut jump_pcs = Vec::with_capacity(branches.len() - 1);
        for branch in &branches[..branches.len() - 1] {
            let split_pc = self.push(Inst::Jump(0))?;
            let branch_start = self.instructions.len();
            self.compile_pattern(branch)?;
            let jump_pc = self.push(Inst::Jump(0))?;
            jump_pcs.push(jump_pc);
            let next_branch = self.instructions.len();
            self.instructions[split_pc] = Inst::Split {
                first: branch_start,
                second: next_branch,
            };
        }

        if let Some(last) = branches.last() {
            self.compile_pattern(last)?;
        }
        let end_pc = self.instructions.len();
        for jump_pc in jump_pcs {
            self.instructions[jump_pc] = Inst::Jump(end_pc);
        }
        Ok(())
    }

    fn compile_permute(&mut self, items: &[Pattern]) -> Result<()> {
        if items.is_empty() {
            return Ok(());
        }
        if items.len() == 1 {
            return self.compile_pattern(&items[0]);
        }
        if items.len() > self.max_permute_elements {
            return plan_err!(
                "PERMUTE supports at most {} elements, got {}",
                self.max_permute_elements,
                items.len()
            );
        }

        let perms = lexicographic_permutations(items.len());
        let branches: Vec<Pattern> = perms
            .into_iter()
            .map(|order| {
                Pattern::Concat(order.into_iter().map(|i| items[i].clone()).collect())
            })
            .collect();
        self.compile_alternation(&branches)
    }

    fn compile_repetition(
        &mut self,
        inner: &Pattern,
        min: u32,
        max: Option<u32>,
        reluctant: bool,
    ) -> Result<()> {
        if let Some(max_val) = max {
            if max_val == 0 {
                return plan_err!(
                    "Pattern quantifier upper bound must be greater than or equal to 1"
                );
            }
            if min > max_val {
                return plan_err!(
                    "Invalid quantifier bounds: min ({min}) exceeds max ({max_val})"
                );
            }
            for _ in 0..min {
                self.compile_pattern(inner)?;
            }
            if min == max_val {
                return Ok(());
            }

            let mut split_pcs = Vec::with_capacity((max_val - min) as usize);
            let mut body_pcs = Vec::with_capacity((max_val - min) as usize);
            for _ in min..max_val {
                let split_pc = self.push(Inst::Jump(0))?;
                let body_pc = self.instructions.len();
                self.compile_pattern(inner)?;
                split_pcs.push(split_pc);
                body_pcs.push(body_pc);
            }
            let exit_pc = self.instructions.len();
            for (split_pc, body_pc) in split_pcs.into_iter().zip(body_pcs) {
                self.instructions[split_pc] = if reluctant {
                    Inst::Split {
                        first: exit_pc,
                        second: body_pc,
                    }
                } else {
                    Inst::Split {
                        first: body_pc,
                        second: exit_pc,
                    }
                };
            }
            return Ok(());
        }

        if min == 0 {
            let entry_split = self.push(Inst::Jump(0))?;
            let body_start = self.instructions.len();
            self.compile_pattern(inner)?;
            let exit_pc = self.instructions.len() + 1;
            let loop_split = if reluctant {
                Inst::Split {
                    first: exit_pc,
                    second: body_start,
                }
            } else {
                Inst::Split {
                    first: body_start,
                    second: exit_pc,
                }
            };
            self.push(loop_split)?;

            self.instructions[entry_split] = if reluctant {
                Inst::Split {
                    first: exit_pc,
                    second: body_start,
                }
            } else {
                Inst::Split {
                    first: body_start,
                    second: exit_pc,
                }
            };
            return Ok(());
        }

        let mut last_body_start = self.instructions.len();
        for _ in 0..min {
            last_body_start = self.instructions.len();
            self.compile_pattern(inner)?;
        }
        let exit_pc = self.instructions.len() + 1;
        let loop_split = if reluctant {
            Inst::Split {
                first: exit_pc,
                second: last_body_start,
            }
        } else {
            Inst::Split {
                first: last_body_start,
                second: exit_pc,
            }
        };
        self.push(loop_split)?;
        Ok(())
    }
}

fn lexicographic_permutations(n: usize) -> Vec<Vec<usize>> {
    let mut current: Vec<usize> = (0..n).collect();
    let mut result = vec![current.clone()];
    while next_permutation(&mut current) {
        result.push(current.clone());
    }
    result
}

fn next_permutation(a: &mut [usize]) -> bool {
    if a.len() < 2 {
        return false;
    }
    let mut i = a.len() - 2;
    while a[i] >= a[i + 1] {
        if i == 0 {
            return false;
        }
        i -= 1;
    }
    let mut j = a.len() - 1;
    while a[j] <= a[i] {
        j -= 1;
    }
    a.swap(i, j);
    a[i + 1..].reverse();
    true
}
