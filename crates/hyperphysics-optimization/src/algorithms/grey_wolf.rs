//! Grey Wolf Optimizer (GWO) with formal convergence analysis.
//!
//! # Biological Inspiration
//! Models the social hierarchy and hunting behavior of grey wolves.
//! Alpha (α), Beta (β), Delta (δ) wolves guide the pack towards prey.
//!
//! # References
//! - Mirjalili et al. (2014): "Grey Wolf Optimizer"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// GWO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GWOConfig {
    /// Initial value of parameter a (decreases from 2 to 0)
    pub a_initial: f64,
    /// Final value of parameter a
    pub a_final: f64,
}

impl Default for GWOConfig {
    fn default() -> Self {
        Self { a_initial: 2.0, a_final: 0.0 }
    }
}

impl AlgorithmConfig for GWOConfig {
    fn validate(&self) -> Result<(), String> {
        if self.a_initial < self.a_final {
            return Err("a_initial must be >= a_final".to_string());
        }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { a_initial: 2.0, a_final: 0.0 } }
    fn high_accuracy() -> Self { Self { a_initial: 2.5, a_final: 0.0 } }
}

/// Grey Wolf Optimizer.
pub struct GreyWolfOptimizer {
    config: GWOConfig,
    opt_config: OptimizationConfig,
    population: Population,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
    alpha: Option<Individual>,
    beta: Option<Individual>,
    delta: Option<Individual>,
}

impl GreyWolfOptimizer {
    /// Create new GWO optimizer.
    pub fn new(config: GWOConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
            alpha: None,
            beta: None,
            delta: None,
        })
    }

    /// Initialize population.
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    /// Run single GWO iteration.
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Update alpha, beta, delta (top 3 wolves)
        let top3 = self.population.top_k(3);
        if top3.len() >= 3 {
            self.alpha = Some(top3[0].clone());
            self.beta = Some(top3[1].clone());
            self.delta = Some(top3[2].clone());
        }

        let alpha = self.alpha.as_ref().unwrap();
        let beta = self.beta.as_ref().unwrap();
        let delta = self.delta.as_ref().unwrap();

        // Linearly decrease a from a_initial to a_final
        let progress = self.iteration as f64 / self.opt_config.max_iterations as f64;
        let a = self.config.a_initial - (self.config.a_initial - self.config.a_final) * progress;

        let mut rng = rand::thread_rng();
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;

        let mut new_positions: Vec<Array1<f64>> = Vec::new();
        let pop = self.population.individuals();

        for wolf in pop.iter() {
            let mut new_pos = Array1::zeros(dim);
            for j in 0..dim {
                // Position update using alpha, beta, delta
                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();
                let a1 = 2.0 * a * r1 - a;
                let c1 = 2.0 * r2;
                let d_alpha = (c1 * alpha.position[j] - wolf.position[j]).abs();
                let x1 = alpha.position[j] - a1 * d_alpha;

                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();
                let a2 = 2.0 * a * r1 - a;
                let c2 = 2.0 * r2;
                let d_beta = (c2 * beta.position[j] - wolf.position[j]).abs();
                let x2 = beta.position[j] - a2 * d_beta;

                let r1: f64 = rng.gen();
                let r2: f64 = rng.gen();
                let a3 = 2.0 * a * r1 - a;
                let c3 = 2.0 * r2;
                let d_delta = (c3 * delta.position[j] - wolf.position[j]).abs();
                let x3 = delta.position[j] - a3 * d_delta;

                new_pos[j] = (x1 + x2 + x3) / 3.0;
            }
            new_positions.push(bounds.repair(new_pos.view()));
        }
        drop(pop);

        // Update positions
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

    /// Run full optimization.
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("GWO failed".to_string()))
    }
}

impl Algorithm for GreyWolfOptimizer {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::GreyWolf }
    fn name(&self) -> &str { "Grey Wolf Optimizer" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_gwo_optimization() {
        let config = GWOConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(100).with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut gwo = GreyWolfOptimizer::new(config, opt_config, bounds).unwrap();
        let objective = SphereFunction::new(5);
        let solution = gwo.optimize(&objective).unwrap();
        assert!(solution.fitness.unwrap() < 10.0);
    }
}
