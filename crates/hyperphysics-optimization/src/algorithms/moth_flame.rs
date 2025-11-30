//! Moth-Flame Optimization (MFO) - Moth navigation by moonlight.
//!
//! # References
//! - Mirjalili (2015): "Moth-flame optimization algorithm: A novel nature-inspired heuristic paradigm"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// MFO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MFOConfig {
    /// Spiral shape constant (typically 1)
    pub b: f64,
}

impl Default for MFOConfig {
    fn default() -> Self {
        Self { b: 1.0 }
    }
}

impl AlgorithmConfig for MFOConfig {
    fn validate(&self) -> Result<(), String> {
        if self.b <= 0.0 { return Err("b must be positive".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { b: 0.5 } }
    fn high_accuracy() -> Self { Self { b: 1.5 } }
}

/// Moth-Flame Optimizer.
pub struct MothFlame {
    config: MFOConfig,
    opt_config: OptimizationConfig,
    population: Population,
    flames: Vec<Individual>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl MothFlame {
    /// Create a new Moth-Flame Optimization optimizer
    pub fn new(config: MFOConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            flames: Vec::new(),
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize moth population and flames
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
        self.flames = Vec::new();
    }

    /// Execute one iteration of moth-flame spiral movement
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let n = self.population.len();
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let max_iter = self.opt_config.max_iterations as f64;
        let current_iter = self.iteration as f64;
        let mut rng = rand::thread_rng();

        // Number of flames decreases over iterations
        let flame_no = (n as f64 - current_iter * (n as f64 - 1.0) / max_iter).round() as usize;
        let flame_no = flame_no.max(1);

        // Merge moths and flames, sort by fitness
        let moths: Vec<Individual> = self.population.individuals().clone();
        let mut all_solutions: Vec<Individual> = moths.clone();
        all_solutions.extend(self.flames.clone());

        // Sort by fitness (ascending for minimization)
        all_solutions.sort_by(|a, b| {
            a.fitness.unwrap_or(f64::INFINITY)
                .partial_cmp(&b.fitness.unwrap_or(f64::INFINITY))
                .unwrap()
        });

        // Select best solutions as flames
        self.flames = all_solutions.into_iter().take(flame_no).collect();

        // Linearly decreasing convergence constant
        let a = -1.0 - current_iter / max_iter; // Decreases from -1 to -2

        let mut new_positions: Vec<Array1<f64>> = Vec::new();

        for i in 0..n {
            // Each moth is associated with a flame (cycling if needed)
            let flame_idx = if i < flame_no { i } else { flame_no - 1 };
            let flame = &self.flames[flame_idx];
            let moth = &moths[i];

            let mut new_pos = Array1::zeros(dim);

            for d in 0..dim {
                // Distance to flame
                let distance = (flame.position[d] - moth.position[d]).abs();

                // Random parameter t in [a, 1] for spiral
                let t = (a - 1.0) * rng.gen::<f64>() + 1.0; // t in [a, 1]

                // Logarithmic spiral
                new_pos[d] = distance * (self.config.b * t * std::f64::consts::PI).exp()
                    * (2.0 * std::f64::consts::PI * t).cos()
                    + flame.position[d];
            }

            new_positions.push(bounds.repair(new_pos.view()));
        }

        // Update population
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

    /// Run the full moth-flame optimization until convergence or max iterations
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();

        // Initial evaluation to set up flames
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        self.flames = self.population.individuals().clone();
        self.flames.sort_by(|a, b| {
            a.fitness.unwrap_or(f64::INFINITY)
                .partial_cmp(&b.fitness.unwrap_or(f64::INFINITY))
                .unwrap()
        });

        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }

        // Return best from flames
        self.flames.first()
            .cloned()
            .ok_or_else(|| OptimizationError::NoSolution("MFO failed".to_string()))
    }
}

impl Algorithm for MothFlame {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::MothFlame }
    fn name(&self) -> &str { "Moth-Flame Optimization" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> {
        self.flames.first().and_then(|f| f.fitness)
            .or_else(|| self.population.best_fitness())
    }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_mfo_optimization() {
        let config = MFOConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(50).with_population_size(25);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut mfo = MothFlame::new(config, opt_config, bounds).unwrap();
        let solution = mfo.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
