// SPDX-License-Identifier: Apache-2.0

//! Compiles a [`Pattern`] AST into priority-ordered NFA bytecode.
//!
//! The bytecode preserves SQL:2016 leftmost-ordered alternation and greedy vs. reluctant
//! quantifier priorities. Placing the loop-back `Split` at the end of unbounded repetitions
//! ensures standard per-position `visited[pc]` epsilon-closure deduplication handles empty
//! iterations (such as `(A??)*`) without dropping higher-priority exits.

use std::collections::HashMap;

use datafusion_common::{Result, plan_err};

use crate::ast::{MatchRecognizeConfig, Pattern, SymbolRef};

/// Dense 0-based runtime identifier for a pattern variable.
pub type SymbolId = u16;

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
    /// Index-to-name symbol table (`id_to_symbol[sym as usize]`).
    pub id_to_symbol: Vec<String>,
    /// Name-to-index symbol lookup.
    pub symbol_to_id: HashMap<String, SymbolId>,
}

impl Program {
    /// Compiles a [`Pattern`] and explicit symbol list into a [`Program`].
    ///
    /// Any pattern variable referenced in `pattern` that is not in `declared_symbols`
    /// is automatically registered (implicit `TRUE` definition per SQL:2016).
    pub fn compile(
        pattern: &Pattern,
        declared_symbols: &[String],
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

        let mut compiler = Compiler {
            instructions: Vec::new(),
            symbol_to_id: &symbol_to_id,
            max_permute_elements: config.max_permute_elements as usize,
            max_nfa_states: config.max_nfa_states as usize,
        };
        compiler.compile_pattern(pattern)?;
        compiler.push(Inst::Accept)?;

        Ok(Program {
            instructions: compiler.instructions,
            id_to_symbol,
            symbol_to_id,
        })
    }

    /// Returns the number of registered pattern symbols.
    pub fn num_symbols(&self) -> usize {
        self.id_to_symbol.len()
    }

    /// Looks up a symbol's dense runtime id by normalized name.
    pub fn symbol_id(&self, name: &str) -> Option<SymbolId> {
        self.symbol_to_id.get(name).copied()
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
    let id = SymbolId::try_from(id_to_symbol.len())
        .map_err(|_| datafusion_common::plan_datafusion_err!("Too many pattern variables"))?;
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

impl<'a> Compiler<'a> {
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
                    datafusion_common::plan_datafusion_err!("Unknown pattern variable '{name}'")
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

        self.compile_pattern(branches.last().expect("non-empty branches"))?;
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
            .map(|order| Pattern::Concat(order.into_iter().map(|i| items[i].clone()).collect()))
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
            if min > max_val {
                return plan_err!("Invalid quantifier bounds: min ({min}) exceeds max ({max_val})");
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

        // Unbounded repetition: {min,}
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

/// Generates all permutations of `0..n` in strict lexicographical order.
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
