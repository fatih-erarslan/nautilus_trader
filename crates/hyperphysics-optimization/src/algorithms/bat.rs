//! Bat Algorithm (BA) - Echolocation behavior of bats.
//!
//! # References
//! - Yang (2010): "A New Metaheuristic Bat-Inspired Algorithm"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Bat Algorithm configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatConfig {
    /// Minimum frequency
    pub f_min: f64,
    /// Maximum frequency
    pub f_max: f64,
    /// Initial loudness
    pub loudness: f64,
    /// Initial pulse rate
    pub pulse_rate: f64,
    /// Loudness decrease rate
    pub alpha: f64,
    /// Pulse rate increase rate
    pub gamma: f64,
}

impl Default for BatConfig {
    fn default() -> Self {
        Self { f_min: 0.0, f_max: 2.0, loudness: 0.5, pulse_rate: 0.5, alpha: 0.9, gamma: 0.9 }
    }
}

impl AlgorithmConfig for BatConfig {
    fn validate(&self) -> Result<(), String> {
        if self.f_min > self.f_max { return Err("f_min must be <= f_max".to_string()); }
        if self.loudness < 0.0 || self.loudness > 1.0 { return Err("loudness must be in [0, 1]".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { f_min: 0.0, f_max: 1.5, loudness: 0.7, pulse_rate: 0.3, alpha: 0.95, gamma: 0.9 } }
    fn high_accuracy() -> Self { Self::default() }
}

/// Bat Algorithm optimizer.
pub struct BatOptimizer {
    config: BatConfig,
    opt_config: OptimizationConfig,
    population: Population,
    velocities: Vec<Array1<f64>>,
    frequencies: Vec<f64>,
    loudnesses: Vec<f64>,
    pulse_rates: Vec<f64>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl BatOptimizer {
    pub fn new(config: BatConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let pop_size = opt_config.population_size;
        let dim = bounds.dimension();
        Ok(Self {
            config: config.clone(),
            opt_config: opt_config.clone(),
            population: Population::new(pop_size, bounds),
            velocities: vec![Array1::zeros(dim); pop_size],
            frequencies: vec![0.0; pop_size],
            loudnesses: vec![config.loudness; pop_size],
            pulse_rates: vec![config.pulse_rate; pop_size],
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let best = self.population.best().unwrap();
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let mut rng = rand::thread_rng();

        let pop_snapshot: Vec<Individual> = self.population.individuals().clone();
        let mut new_positions: Vec<Array1<f64>> = Vec::new();

        for (i, bat) in pop_snapshot.iter().enumerate() {
            // Update frequency
            let beta: f64 = rng.gen();
            self.frequencies[i] = self.config.f_min + (self.config.f_max - self.config.f_min) * beta;

            // Update velocity and position
            let mut new_vel = Array1::zeros(dim);
            let mut new_pos = Array1::zeros(dim);

            for j in 0..dim {
                new_vel[j] = self.velocities[i][j] + (bat.position[j] - best.position[j]) * self.frequencies[i];
                new_pos[j] = bat.position[j] + new_vel[j];
            }

            // Local search
            if rng.gen::<f64>() > self.pulse_rates[i] {
                let avg_loudness: f64 = self.loudnesses.iter().sum::<f64>() / self.loudnesses.len() as f64;
                for j in 0..dim {
                    new_pos[j] = best.position[j] + avg_loudness * (rng.gen::<f64>() * 2.0 - 1.0);
                }
            }

            let repaired = bounds.repair(new_pos.view());
            let new_fitness = objective.evaluate(repaired.view());

            // Accept if improved and random < loudness
            if rng.gen::<f64>() < self.loudnesses[i] && new_fitness < bat.fitness.unwrap_or(f64::INFINITY) {
                self.velocities[i] = new_vel;
                self.loudnesses[i] *= self.config.alpha;
                self.pulse_rates[i] = self.config.pulse_rate * (1.0 - (-self.config.gamma * self.iteration as f64).exp());
                new_positions.push(repaired);
            } else {
                new_positions.push(bat.position.clone());
            }
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

    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("BA failed".to_string()))
    }
}

impl Algorithm for BatOptimizer {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::BatAlgorithm }
    fn name(&self) -> &str { "Bat Algorithm" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_bat_optimization() {
        let config = BatConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(100).with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut ba = BatOptimizer::new(config, opt_config, bounds).unwrap();
        let solution = ba.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
