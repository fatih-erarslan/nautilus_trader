//! Whale Optimization Algorithm (WOA) - Humpback whale bubble-net feeding.
//!
//! # References
//! - Mirjalili & Lewis (2016): "The Whale Optimization Algorithm"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// WOA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WOAConfig {
    /// Spiral shape constant
    pub b: f64,
    /// Probability of choosing spiral vs encircling
    pub spiral_prob: f64,
}

impl Default for WOAConfig {
    fn default() -> Self {
        Self { b: 1.0, spiral_prob: 0.5 }
    }
}

impl AlgorithmConfig for WOAConfig {
    fn validate(&self) -> Result<(), String> {
        if self.spiral_prob < 0.0 || self.spiral_prob > 1.0 {
            return Err("spiral_prob must be in [0, 1]".to_string());
        }
        Ok(())
    }
    fn hft_optimized() -> Self { Self::default() }
    fn high_accuracy() -> Self { Self { b: 1.5, spiral_prob: 0.5 } }
}

/// Whale Optimization Algorithm.
pub struct WhaleOptimizer {
    config: WOAConfig,
    opt_config: OptimizationConfig,
    population: Population,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl WhaleOptimizer {
    /// Create a new Whale Optimization Algorithm optimizer
    pub fn new(config: WOAConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize population using Latin Hypercube Sampling
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    /// Execute one iteration of the whale optimization algorithm
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let best = self.population.best().unwrap();
        let progress = self.iteration as f64 / self.opt_config.max_iterations as f64;
        let a = 2.0 - 2.0 * progress; // Decreases from 2 to 0

        let mut rng = rand::thread_rng();
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let pop_snapshot: Vec<Individual> = self.population.individuals().clone();

        let mut new_positions: Vec<Array1<f64>> = Vec::new();

        for whale in pop_snapshot.iter() {
            let mut new_pos = Array1::zeros(dim);
            let p: f64 = rng.gen();

            for j in 0..dim {
                let r: f64 = rng.gen();
                let a_coef = 2.0 * a * r - a;
                let c = 2.0 * rng.gen::<f64>();

                if p < self.config.spiral_prob {
                    // Shrinking encircling mechanism
                    if a_coef.abs() < 1.0 {
                        let d = (c * best.position[j] - whale.position[j]).abs();
                        new_pos[j] = best.position[j] - a_coef * d;
                    } else {
                        // Search for prey (exploration)
                        let rand_idx = rng.gen_range(0..pop_snapshot.len());
                        let rand_whale = &pop_snapshot[rand_idx];
                        let d = (c * rand_whale.position[j] - whale.position[j]).abs();
                        new_pos[j] = rand_whale.position[j] - a_coef * d;
                    }
                } else {
                    // Spiral updating position (bubble-net attack)
                    let d_prime = (best.position[j] - whale.position[j]).abs();
                    let l: f64 = rng.gen_range(-1.0..1.0);
                    new_pos[j] = d_prime * (self.config.b * l * std::f64::consts::PI).exp() * (2.0 * std::f64::consts::PI * l).cos() + best.position[j];
                }
            }
            new_positions.push(bounds.repair(new_pos.view()));
        }

        {
            let mut pop = self.population.individuals_mut();
            for (i, ind) in pop.iter_mut().enumerate() {
                ind.position = new_positions[i].clone();
                ind.fitness = None;
            }
        }

        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    /// Run the full optimization until convergence or max iterations
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("WOA failed".to_string()))
    }
}

impl Algorithm for WhaleOptimizer {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::WhaleOptimization }
    fn name(&self) -> &str { "Whale Optimization Algorithm" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_woa_optimization() {
        let config = WOAConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(100).with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut woa = WhaleOptimizer::new(config, opt_config, bounds).unwrap();
        let solution = woa.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.unwrap() < 50.0);
    }
}
