//! Ant Colony Optimization (ACO) for continuous domains.
//!
//! # References
//! - Socha & Dorigo (2008): "Ant Colony Optimization for Continuous Domains"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

/// ACO configuration for continuous optimization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ACOConfig {
    /// Archive size (number of solutions)
    pub archive_size: usize,
    /// Locality of search parameter q
    pub q: f64,
    /// Convergence speed parameter ξ (xi)
    pub xi: f64,
}

impl Default for ACOConfig {
    fn default() -> Self {
        Self { archive_size: 50, q: 0.5, xi: 0.85 }
    }
}

impl AlgorithmConfig for ACOConfig {
    fn validate(&self) -> Result<(), String> {
        if self.q <= 0.0 || self.q > 1.0 {
            return Err("q must be in (0, 1]".to_string());
        }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { archive_size: 30, q: 0.3, xi: 0.8 } }
    fn high_accuracy() -> Self { Self { archive_size: 100, q: 0.1, xi: 0.9 } }
}

/// Ant Colony Optimizer for continuous domains.
pub struct AntColonyOptimizer {
    config: ACOConfig,
    opt_config: OptimizationConfig,
    population: Population,
    archive: Vec<Individual>,
    weights: Vec<f64>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl AntColonyOptimizer {
    pub fn new(config: ACOConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let archive_size = config.archive_size;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            archive: Vec::with_capacity(archive_size),
            weights: Vec::new(),
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.config.archive_size);
    }

    fn calculate_weights(&mut self) {
        let k = self.config.archive_size;
        let q = self.config.q;
        self.weights = (1..=k).map(|l| {
            let l = l as f64;
            let k = k as f64;
            (1.0 / (q * k * (2.0 * std::f64::consts::PI).sqrt())) *
                (-((l - 1.0).powi(2)) / (2.0 * q.powi(2) * k.powi(2))).exp()
        }).collect();
        let sum: f64 = self.weights.iter().sum();
        for w in &mut self.weights { *w /= sum; }
    }

    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Update archive
        let mut all: Vec<Individual> = self.archive.iter().cloned()
            .chain(self.population.individuals().iter().cloned())
            .collect();
        all.sort();
        self.archive = all.into_iter().take(self.config.archive_size).collect();
        self.calculate_weights();

        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let mut rng = rand::thread_rng();

        // Generate new solutions
        let mut new_solutions: Vec<Individual> = Vec::new();
        let archive_len = self.archive.len();
        for _ in 0..self.opt_config.population_size {
            // Select solution from archive based on weights
            let r: f64 = rng.gen();
            let mut cumulative = 0.0;
            let mut selected_idx = 0;
            for (i, w) in self.weights.iter().take(archive_len).enumerate() {
                cumulative += w;
                if r <= cumulative {
                    selected_idx = i;
                    break;
                }
            }
            // Ensure selected_idx is valid for the current archive size
            selected_idx = selected_idx.min(archive_len.saturating_sub(1));

            let selected = &self.archive[selected_idx];
            let mut new_pos = Array1::zeros(dim);

            for j in 0..dim {
                // Calculate standard deviation
                let sigma: f64 = self.archive.iter().enumerate()
                    .map(|(e, sol)| self.config.xi * (sol.position[j] - selected.position[j]).abs() / (self.archive.len() - 1) as f64)
                    .sum();
                let sigma = sigma.max(1e-10);

                // Sample from Gaussian
                let normal = Normal::new(selected.position[j], sigma).unwrap_or(Normal::new(0.0, 1.0).unwrap());
                new_pos[j] = normal.sample(&mut rng);
            }

            let mut ind = Individual::new(bounds.repair(new_pos.view()));
            ind.fitness = Some(objective.evaluate(ind.position.view()));
            new_solutions.push(ind);
        }

        {
            let mut pop = self.population.individuals_mut();
            *pop = new_solutions;
        }

        if let Some(f) = self.archive.first().and_then(|a| a.fitness) {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        self.archive = self.population.individuals().clone();
        self.archive.sort();
        self.archive.truncate(self.config.archive_size);
        self.calculate_weights();

        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.archive.first().cloned().ok_or_else(|| OptimizationError::NoSolution("ACO failed".to_string()))
    }
}

impl Algorithm for AntColonyOptimizer {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::AntColony }
    fn name(&self) -> &str { "Ant Colony Optimization" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.archive.first().and_then(|a| a.fitness) }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_aco_optimization() {
        let config = ACOConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(50).with_population_size(20);
        let bounds = Bounds::symmetric(3, 5.12);
        let mut aco = AntColonyOptimizer::new(config, opt_config, bounds).unwrap();
        let solution = aco.optimize(&SphereFunction::new(3)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
