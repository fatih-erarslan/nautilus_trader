//! Firefly Algorithm (FA) - Bioluminescent attraction behavior.
//!
//! # References
//! - Yang (2008): "Nature-Inspired Metaheuristic Algorithms"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Firefly Algorithm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireflyConfig {
    /// Light absorption coefficient
    pub gamma: f64,
    /// Attraction coefficient at r=0
    pub beta0: f64,
    /// Randomization parameter
    pub alpha: f64,
    /// Alpha reduction rate
    pub alpha_decay: f64,
}

impl Default for FireflyConfig {
    fn default() -> Self {
        Self { gamma: 1.0, beta0: 1.0, alpha: 0.5, alpha_decay: 0.97 }
    }
}

impl AlgorithmConfig for FireflyConfig {
    fn validate(&self) -> Result<(), String> {
        if self.gamma < 0.0 { return Err("gamma must be non-negative".to_string()); }
        if self.beta0 < 0.0 { return Err("beta0 must be non-negative".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { gamma: 1.5, beta0: 1.0, alpha: 0.3, alpha_decay: 0.95 } }
    fn high_accuracy() -> Self { Self { gamma: 0.5, beta0: 1.0, alpha: 0.2, alpha_decay: 0.99 } }
}

/// Firefly Algorithm optimizer.
pub struct FireflyOptimizer {
    config: FireflyConfig,
    opt_config: OptimizationConfig,
    population: Population,
    alpha: f64,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl FireflyOptimizer {
    /// Create a new Firefly Algorithm optimizer
    pub fn new(config: FireflyConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let alpha = config.alpha;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            alpha,
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize firefly population using Latin Hypercube Sampling
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    /// Execute one iteration of firefly movement and attraction
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let ranges = bounds.ranges();
        let mut rng = rand::thread_rng();

        let pop_snapshot: Vec<Individual> = self.population.individuals().clone();
        let n = pop_snapshot.len();
        let mut new_positions: Vec<Array1<f64>> = pop_snapshot.iter().map(|p| p.position.clone()).collect();

        // Move fireflies towards brighter ones (O(n²) complexity)
        for i in 0..n {
            for j in 0..n {
                let fi = pop_snapshot[i].fitness.unwrap_or(f64::INFINITY);
                let fj = pop_snapshot[j].fitness.unwrap_or(f64::INFINITY);

                if fj < fi {
                    // Calculate Euclidean distance
                    let r_sq: f64 = (0..dim).map(|k| (pop_snapshot[i].position[k] - pop_snapshot[j].position[k]).powi(2)).sum();
                    let _r = r_sq.sqrt();

                    // Attractiveness decreases with distance
                    let beta = self.config.beta0 * (-self.config.gamma * r_sq).exp();

                    // Move firefly i towards j
                    for k in 0..dim {
                        let rand_term = self.alpha * (rng.gen::<f64>() - 0.5) * ranges[k];
                        new_positions[i][k] += beta * (pop_snapshot[j].position[k] - pop_snapshot[i].position[k]) + rand_term;
                    }
                }
            }
        }

        // Repair bounds and update
        {
            let mut pop = self.population.individuals_mut();
            for (i, ind) in pop.iter_mut().enumerate() {
                ind.position = bounds.repair(new_positions[i].view());
                ind.fitness = None;
            }
        }

        // Reduce alpha
        self.alpha *= self.config.alpha_decay;

        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    /// Run the full firefly optimization until convergence or max iterations
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("FA failed".to_string()))
    }
}

impl Algorithm for FireflyOptimizer {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::FireflyAlgorithm }
    fn name(&self) -> &str { "Firefly Algorithm" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_firefly_optimization() {
        let config = FireflyConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(50).with_population_size(20);
        let bounds = Bounds::symmetric(3, 5.12);
        let mut fa = FireflyOptimizer::new(config, opt_config, bounds).unwrap();
        let solution = fa.optimize(&SphereFunction::new(3)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
