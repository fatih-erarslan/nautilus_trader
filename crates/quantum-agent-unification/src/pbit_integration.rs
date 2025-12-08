//! pBit Integration Module
//!
//! Bridges the quantum agent optimization system with the pBit probabilistic computing
//! infrastructure from quantum-core. This enables quantum algorithms to run on the
//! Ising-model-based pBit lattice with STDP learning.
//!
//! ## Architecture
//!
//! ```text
//! QuantumBit (local) ←→ PBit (quantum-core)
//! QuantumState (local) ←→ PBitState (quantum-core)
//! QuantumOptimizer ←→ PBitOptimizer
//! ```

use crate::quantum_state::{QuantumBit, QuantumState, BlochSphere};
use crate::{QuantumResult, QuantumError, QuantumOptimizationResult, QuantumMetrics};
use quantum_core::{
    PBitState, PBitConfig, PBitCircuit, PBitBackend, PBitBackendConfig,
    LatticeState, LatticeBridgeConfig,
};
use nalgebra::DVector;
use rand::{thread_rng, Rng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::Arc;

/// Configuration for pBit-based optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitOptimizerConfig {
    /// Number of pBits per dimension
    pub pbits_per_dimension: usize,
    /// Initial temperature for annealing
    pub initial_temperature: f64,
    /// Final temperature after annealing
    pub final_temperature: f64,
    /// Annealing schedule steps
    pub annealing_steps: usize,
    /// Coupling strength for entanglement
    pub coupling_strength: f64,
    /// STDP learning rate
    pub stdp_learning_rate: f64,
    /// Number of sweeps per iteration
    pub sweeps_per_iteration: usize,
    /// Enable parallel execution
    pub parallel: bool,
}

impl Default for PBitOptimizerConfig {
    fn default() -> Self {
        Self {
            pbits_per_dimension: 8,
            initial_temperature: 10.0,
            final_temperature: 0.01,
            annealing_steps: 100,
            coupling_strength: 1.0,
            stdp_learning_rate: 0.01,
            sweeps_per_iteration: 10,
            parallel: true,
        }
    }
}

/// pBit-based optimizer using Ising lattice dynamics
#[derive(Debug)]
pub struct PBitOptimizer {
    config: PBitOptimizerConfig,
    /// pBit states for each candidate solution
    population: Vec<PBitState>,
    /// Best solution found
    best_solution: Option<(Vec<f64>, f64)>,
    /// Lattice for global coordination
    coordination_lattice: Option<LatticeState>,
    /// Temperature schedule
    current_temperature: f64,
    /// Iteration counter
    iteration: usize,
}

impl PBitOptimizer {
    /// Create a new pBit optimizer
    pub fn new(config: PBitOptimizerConfig) -> Self {
        Self {
            config,
            population: Vec::new(),
            best_solution: None,
            coordination_lattice: None,
            current_temperature: 0.0,
            iteration: 0,
        }
    }

    /// Create with default configuration
    pub fn default_optimizer() -> Self {
        Self::new(PBitOptimizerConfig::default())
    }

    /// Initialize population for optimization problem
    pub fn initialize(&mut self, dimensions: usize, population_size: usize) -> QuantumResult<()> {
        let total_pbits = dimensions * self.config.pbits_per_dimension;
        
        // Limit to 32 qubits max
        let actual_pbits = total_pbits.min(32);
        
        self.population.clear();
        self.current_temperature = self.config.initial_temperature;
        self.iteration = 0;

        let pbit_config = PBitConfig {
            temperature: self.config.initial_temperature,
            coupling_strength: self.config.coupling_strength,
            external_field: 0.0,
            seed: None,
        };

        for _ in 0..population_size {
            let mut state = PBitState::with_config(actual_pbits, pbit_config.clone())
                .map_err(|e| QuantumError::GateError(e.to_string()))?;
            
            // Initialize in superposition
            for i in 0..actual_pbits {
                if let Some(pbit) = state.get_pbit_mut(i) {
                    pbit.probability_up = 0.5;
                    pbit.spin = if rand::random::<f64>() < 0.5 { 1.0 } else { -1.0 };
                }
            }
            
            self.population.push(state);
        }

        // Create coordination lattice
        let lattice_config = LatticeBridgeConfig::for_qubits(population_size.min(32));
        self.coordination_lattice = Some(
            LatticeState::new(lattice_config)
                .map_err(|e| QuantumError::GateError(e.to_string()))?
        );

        Ok(())
    }

    /// Decode pBit state to solution vector
    fn decode_solution(&self, state: &PBitState, bounds: &[(f64, f64)]) -> Vec<f64> {
        let dimensions = bounds.len();
        let pbits_per_dim = self.config.pbits_per_dimension.min(state.num_qubits() / dimensions.max(1));
        
        let mut solution = Vec::with_capacity(dimensions);
        
        for (dim, &(min, max)) in bounds.iter().enumerate() {
            // Gray-code-like decoding from pBit probabilities
            let start_idx = dim * pbits_per_dim;
            let end_idx = (start_idx + pbits_per_dim).min(state.num_qubits());
            
            let mut value = 0.0;
            let mut weight = 0.5;
            
            for i in start_idx..end_idx {
                if let Some(pbit) = state.get_pbit(i) {
                    value += pbit.probability_up * weight;
                }
                weight /= 2.0;
            }
            
            // Scale to bounds
            solution.push(min + value * (max - min) * 2.0);
        }
        
        solution
    }

    /// Encode solution into pBit biases
    fn encode_solution(&self, state: &mut PBitState, solution: &[f64], bounds: &[(f64, f64)]) {
        let dimensions = bounds.len();
        let pbits_per_dim = self.config.pbits_per_dimension.min(state.num_qubits() / dimensions.max(1));
        
        for (dim, (&val, &(min, max))) in solution.iter().zip(bounds.iter()).enumerate() {
            let normalized = (val - min) / (max - min);
            let start_idx = dim * pbits_per_dim;
            let end_idx = (start_idx + pbits_per_dim).min(state.num_qubits());
            
            let mut remaining = normalized;
            let mut weight = 0.5;
            
            for i in start_idx..end_idx {
                if let Some(pbit) = state.get_pbit_mut(i) {
                    if remaining >= weight {
                        pbit.bias = self.config.coupling_strength;
                        remaining -= weight;
                    } else {
                        pbit.bias = -self.config.coupling_strength * (1.0 - remaining / weight);
                    }
                }
                weight /= 2.0;
            }
        }
    }

    /// Perform one optimization step
    pub fn step<F>(&mut self, objective: F, bounds: &[(f64, f64)]) -> QuantumResult<f64>
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        // Update temperature (annealing schedule)
        let progress = self.iteration as f64 / self.config.annealing_steps as f64;
        self.current_temperature = self.config.initial_temperature 
            * (self.config.final_temperature / self.config.initial_temperature).powf(progress.min(1.0));

        // Evaluate and update population
        let mut best_fitness = f64::INFINITY;
        let mut best_idx = 0;

        // Parallel or sequential evaluation
        let evaluations: Vec<(Vec<f64>, f64)> = if self.config.parallel {
            self.population
                .par_iter()
                .map(|state| {
                    let solution = self.decode_solution(state, bounds);
                    let fitness = objective(&solution);
                    (solution, fitness)
                })
                .collect()
        } else {
            self.population
                .iter()
                .map(|state| {
                    let solution = self.decode_solution(state, bounds);
                    let fitness = objective(&solution);
                    (solution, fitness)
                })
                .collect()
        };

        // Find best
        for (i, (solution, fitness)) in evaluations.iter().enumerate() {
            if *fitness < best_fitness {
                best_fitness = *fitness;
                best_idx = i;
            }
        }

        // Update best solution
        let (best_sol, best_fit) = &evaluations[best_idx];
        if self.best_solution.is_none() || best_fit < &self.best_solution.as_ref().unwrap().1 {
            self.best_solution = Some((best_sol.clone(), *best_fit));
        }

        // pBit dynamics: sweep and anneal
        for state in &mut self.population {
            // Update temperature
            if let Some(pbit) = state.get_pbit_mut(0) {
                // Access config through state
            }
            
            // Perform Monte Carlo sweeps
            for _ in 0..self.config.sweeps_per_iteration {
                state.sweep();
            }
        }

        // Apply STDP-like learning: reinforce good solutions
        if let Some((best_solution, _)) = &self.best_solution {
            for state in &mut self.population {
                self.encode_solution(state, best_solution, bounds);
            }
        }

        // Coordination via lattice
        if let Some(ref mut lattice) = self.coordination_lattice {
            lattice.sweep();
        }

        self.iteration += 1;
        Ok(best_fitness)
    }

    /// Run full optimization
    pub fn optimize<F>(
        &mut self,
        objective: F,
        bounds: &[(f64, f64)],
        max_iterations: usize,
        population_size: usize,
    ) -> QuantumResult<QuantumOptimizationResult>
    where
        F: Fn(&[f64]) -> f64 + Sync,
    {
        self.initialize(bounds.len(), population_size)?;

        let mut convergence_history = Vec::with_capacity(max_iterations);

        for _ in 0..max_iterations {
            let fitness = self.step(&objective, bounds)?;
            convergence_history.push(fitness);

            // Early termination if converged
            if convergence_history.len() > 50 {
                let recent: Vec<_> = convergence_history.iter().rev().take(50).collect();
                let variance: f64 = recent.iter().map(|&&x| x * x).sum::<f64>() / 50.0
                    - (recent.iter().map(|&&x| x).sum::<f64>() / 50.0).powi(2);
                if variance < 1e-10 {
                    break;
                }
            }
        }

        let (best_solution, best_fitness) = self.best_solution.clone()
            .ok_or_else(|| QuantumError::MeasurementError("No solution found".into()))?;

        // Convert population to QuantumStates for compatibility
        let quantum_states: Vec<crate::quantum_state::QuantumState> = self.population
            .iter()
            .map(|pbit_state| {
                let mut qs = crate::quantum_state::QuantumState::new(pbit_state.num_qubits());
                // Map pBit probabilities to qubit states
                for i in 0..pbit_state.num_qubits().min(qs.qubits.len()) {
                    if let Some(pbit) = pbit_state.get_pbit(i) {
                        let theta = 2.0 * pbit.probability_up.acos().max(0.0).min(PI);
                        qs.qubits[i] = QuantumBit::from_angles(theta, 0.0);
                    }
                }
                qs
            })
            .collect();

        Ok(QuantumOptimizationResult {
            best_solution,
            best_fitness,
            iterations: self.iteration,
            quantum_metrics: QuantumMetrics::default(),
            convergence_history,
            quantum_states,
        })
    }

    /// Get current best solution
    pub fn best(&self) -> Option<&(Vec<f64>, f64)> {
        self.best_solution.as_ref()
    }

    /// Get population size
    pub fn population_size(&self) -> usize {
        self.population.len()
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.current_temperature
    }
}

/// Convert local QuantumBit to pBit probability
pub fn quantum_bit_to_pbit_prob(qubit: &QuantumBit) -> f64 {
    qubit.prob_one() // P(|1⟩) maps to P(↑)
}

/// Convert pBit probability to local QuantumBit
pub fn pbit_prob_to_quantum_bit(prob_up: f64) -> QuantumBit {
    let theta = 2.0 * prob_up.acos().max(0.0).min(PI);
    QuantumBit::from_angles(theta, 0.0)
}

/// Convert local QuantumState to PBitState
pub fn quantum_state_to_pbit(qs: &QuantumState) -> Result<PBitState, QuantumError> {
    let config = PBitConfig::default();
    let mut pbit_state = PBitState::with_config(qs.qubits.len(), config)
        .map_err(|e| QuantumError::GateError(e.to_string()))?;

    for (i, qubit) in qs.qubits.iter().enumerate() {
        if let Some(pbit) = pbit_state.get_pbit_mut(i) {
            pbit.probability_up = qubit.prob_one();
            pbit.spin = if qubit.prob_one() > 0.5 { 1.0 } else { -1.0 };
        }
    }

    // Add couplings for entangled pairs
    for &(i, j) in &qs.entangled_pairs {
        pbit_state.add_coupling(quantum_core::PBitCoupling::bell_coupling(i, j, 1.0));
    }

    Ok(pbit_state)
}

/// Convert PBitState to local QuantumState
pub fn pbit_to_quantum_state(pbit: &PBitState) -> QuantumState {
    let mut qs = QuantumState::new(pbit.num_qubits());

    for i in 0..pbit.num_qubits() {
        if let Some(pbit_node) = pbit.get_pbit(i) {
            qs.qubits[i] = pbit_prob_to_quantum_bit(pbit_node.probability_up);
        }
    }

    qs
}

/// Trait for types that can be optimized via pBit
pub trait PBitOptimizable {
    /// Convert to pBit representation
    fn to_pbit_state(&self) -> Result<PBitState, QuantumError>;
    
    /// Update from pBit state
    fn from_pbit_state(&mut self, pbit: &PBitState) -> Result<(), QuantumError>;
}

impl PBitOptimizable for QuantumState {
    fn to_pbit_state(&self) -> Result<PBitState, QuantumError> {
        quantum_state_to_pbit(self)
    }

    fn from_pbit_state(&mut self, pbit: &PBitState) -> Result<(), QuantumError> {
        *self = pbit_to_quantum_state(pbit);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbit_optimizer_creation() {
        let optimizer = PBitOptimizer::default_optimizer();
        assert!(optimizer.population.is_empty());
    }

    #[test]
    fn test_pbit_optimizer_initialize() {
        let mut optimizer = PBitOptimizer::default_optimizer();
        optimizer.initialize(3, 10).unwrap();
        assert_eq!(optimizer.population_size(), 10);
    }

    #[test]
    fn test_sphere_optimization() {
        let mut optimizer = PBitOptimizer::new(PBitOptimizerConfig {
            annealing_steps: 50,
            ..Default::default()
        });

        let sphere = |x: &[f64]| -> f64 {
            x.iter().map(|xi| xi * xi).sum()
        };

        let bounds = vec![(-5.0, 5.0); 3];
        
        let result = optimizer.optimize(sphere, &bounds, 100, 20).unwrap();
        
        // Should find a solution reasonably close to origin
        assert!(result.best_fitness < 10.0);
    }

    #[test]
    fn test_quantum_pbit_conversion() {
        let mut qs = QuantumState::new(4);
        qs.create_superposition();
        qs.entangle(0, 1).unwrap();

        let pbit = quantum_state_to_pbit(&qs).unwrap();
        assert_eq!(pbit.num_qubits(), 4);

        let qs_back = pbit_to_quantum_state(&pbit);
        assert_eq!(qs_back.qubits.len(), 4);
    }
}
