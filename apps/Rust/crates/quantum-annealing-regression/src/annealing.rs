//! Quantum Annealing Implementation
//!
//! This module provides quantum annealing algorithms for optimization problems.
//! It implements simulated quantum annealing with multiple temperature schedules
//! and parallel chain execution.

use crate::core::*;
use crate::error::*;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};
use rayon::prelude::*;
use std::time::Instant;

/// Quantum annealing optimizer
#[derive(Debug, Clone)]
pub struct QuantumAnnealer {
    config: QuantumAnnealingConfig,
    rng: rand::rngs::StdRng,
}

impl QuantumAnnealer {
    /// Create a new quantum annealer
    pub fn new(config: QuantumAnnealingConfig) -> QarResult<Self> {
        config.validate()?;
        
        let rng = match config.seed {
            Some(seed) => rand::rngs::StdRng::seed_from_u64(seed),
            None => rand::rngs::StdRng::from_entropy(),
        };
        
        Ok(Self { config, rng })
    }
    
    /// Optimize using quantum annealing
    pub fn optimize<F>(&mut self, cost_function: F, initial_parameters: Vec<f64>) -> QarResult<AnnealingResult>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let start_time = Instant::now();
        
        // Run multiple parallel chains
        let chain_results: Vec<AnnealingResult> = (0..self.config.num_chains)
            .into_par_iter()
            .map(|chain_id| {
                let mut chain_rng = rand::rngs::StdRng::seed_from_u64(
                    self.config.seed.unwrap_or(0) + chain_id as u64
                );
                self.run_single_chain(&cost_function, &initial_parameters, &mut chain_rng)
            })
            .collect::<Result<Vec<_>>>()?;
        
        // Select best result
        let best_result = chain_results
            .into_iter()
            .min_by(|a, b| a.optimal_energy.partial_cmp(&b.optimal_energy).unwrap())
            .unwrap();
        
        let computation_time = start_time.elapsed().as_secs_f64();
        
        Ok(AnnealingResult::new(
            best_result.optimal_parameters,
            best_result.optimal_energy,
            best_result.iterations,
            best_result.converged,
            best_result.acceptance_rate,
            computation_time,
        ))
    }
    
    /// Run a single annealing chain
    fn run_single_chain<F>(
        &self,
        cost_function: &F,
        initial_parameters: &[f64],
        rng: &mut rand::rngs::StdRng,
    ) -> QarResult<AnnealingResult>
    where
        F: Fn(&[f64]) -> f64 + Send + Sync,
    {
        let mut state = AnnealingState::new(
            initial_parameters.to_vec(),
            cost_function(initial_parameters),
            self.config.initial_temperature,
        );
        
        let mut best_parameters = state.parameters.clone();
        let mut best_energy = state.energy;
        
        for step in 0..self.config.num_steps {
            // Update temperature
            state.temperature = self.calculate_temperature(step);
            
            // Generate new candidate solution
            let candidate_parameters = self.generate_candidate(&state.parameters, state.temperature, rng);
            let candidate_energy = cost_function(&candidate_parameters);
            
            // Metropolis acceptance criterion
            let accepted = self.accept_candidate(
                state.energy,
                candidate_energy,
                state.temperature,
                rng,
            );
            
            // Update state
            state.update(candidate_parameters.clone(), candidate_energy, accepted);
            
            // Track best solution
            if candidate_energy < best_energy {
                best_parameters = candidate_parameters;
                best_energy = candidate_energy;
            }
            
            // Check convergence
            if step > 100 && self.check_convergence(&state) {
                break;
            }
        }
        
        let converged = self.check_convergence(&state);
        
        Ok(AnnealingResult::new(
            best_parameters,
            best_energy,
            state.iteration,
            converged,
            state.acceptance_rate(),
            0.0, // Will be set by caller
        ))
    }
    
    /// Calculate temperature for current step
    fn calculate_temperature(&self, step: usize) -> f64 {
        let progress = step as f64 / self.config.num_steps as f64;
        
        // Exponential cooling schedule
        self.config.initial_temperature * 
            (self.config.final_temperature / self.config.initial_temperature).powf(progress)
    }
    
    /// Generate candidate solution using quantum tunneling
    fn generate_candidate(
        &self,
        current_parameters: &[f64],
        temperature: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> Vec<f64> {
        let step_size = temperature.sqrt();
        let normal = Normal::new(0.0, step_size).unwrap();
        
        current_parameters
            .iter()
            .map(|&param| param + normal.sample(rng))
            .collect()
    }
    
    /// Metropolis acceptance criterion with quantum effects
    fn accept_candidate(
        &self,
        current_energy: f64,
        candidate_energy: f64,
        temperature: f64,
        rng: &mut rand::rngs::StdRng,
    ) -> bool {
        if candidate_energy < current_energy {
            true // Always accept better solutions
        } else {
            let delta_energy = candidate_energy - current_energy;
            let acceptance_probability = (-delta_energy / temperature).exp();
            
            // Add quantum tunneling effect
            let tunneling_probability = 0.1 * (-delta_energy.abs() / temperature).exp();
            let total_probability = acceptance_probability + tunneling_probability;
            
            rng.gen::<f64>() < total_probability
        }
    }
    
    /// Check convergence based on energy stability
    fn check_convergence(&self, state: &AnnealingState) -> bool {
        if state.acceptance_history.len() < 100 {
            return false;
        }
        
        // Check if acceptance rate is too low
        if state.acceptance_rate() < 0.01 {
            return true;
        }
        
        // Check temperature
        if state.temperature < self.config.final_temperature {
            return true;
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantum_annealer_creation() {
        let config = QuantumAnnealingConfig::default();
        let annealer = QuantumAnnealer::new(config);
        assert!(annealer.is_ok());
    }

    #[test]
    fn test_temperature_calculation() {
        let config = QuantumAnnealingConfig {
            initial_temperature: 10.0,
            final_temperature: 0.1,
            num_steps: 100,
            ..Default::default()
        };
        let annealer = QuantumAnnealer::new(config).unwrap();
        
        let temp_start = annealer.calculate_temperature(0);
        let temp_end = annealer.calculate_temperature(100);
        
        assert_relative_eq!(temp_start, 10.0, epsilon = 1e-6);
        assert!(temp_end < temp_start);
    }

    #[test]
    fn test_simple_optimization() {
        let config = QuantumAnnealingConfig {
            num_steps: 100,
            num_chains: 2,
            seed: Some(42),
            ..Default::default()
        };
        let mut annealer = QuantumAnnealer::new(config).unwrap();
        
        // Simple quadratic function: f(x) = (x-2)^2
        let cost_function = |params: &[f64]| (params[0] - 2.0).powi(2);
        
        let initial_params = vec![0.0];
        let result = annealer.optimize(cost_function, initial_params);
        
        assert!(result.is_ok());
        let result = result.unwrap();
        
        // Should converge close to x = 2
        assert!(result.optimal_parameters[0] > 1.5);
        assert!(result.optimal_parameters[0] < 2.5);
        assert!(result.optimal_energy < 0.5);
    }

    #[test]
    fn test_candidate_generation() {
        let config = QuantumAnnealingConfig::default();
        let annealer = QuantumAnnealer::new(config).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        let current = vec![1.0, 2.0];
        let candidate = annealer.generate_candidate(&current, 1.0, &mut rng);
        
        assert_eq!(candidate.len(), current.len());
        // Candidates should be different from current
        assert!(candidate != current);
    }

    #[test]
    fn test_acceptance_criterion() {
        let config = QuantumAnnealingConfig::default();
        let annealer = QuantumAnnealer::new(config).unwrap();
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        
        // Better solution should always be accepted
        assert!(annealer.accept_candidate(5.0, 3.0, 1.0, &mut rng));
        
        // Worse solution acceptance depends on temperature and probability
        let accepted = annealer.accept_candidate(3.0, 5.0, 10.0, &mut rng);
        // With high temperature, some worse solutions should be accepted
        // (This is probabilistic, so we can't assert the exact result)
    }
}