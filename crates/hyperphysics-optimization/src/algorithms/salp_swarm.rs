//! Salp Swarm Algorithm (SSA) - Salp chain swarming behavior.
//!
//! # References
//! - Mirjalili et al. (2017): "Salp Swarm Algorithm: A bio-inspired optimizer for engineering design problems"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// SSA configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SSAConfig {
    /// Food source update weight
    pub c2_weight: f64,
    /// Chain following weight
    pub c3_weight: f64,
}

impl Default for SSAConfig {
    fn default() -> Self {
        Self { c2_weight: 1.0, c3_weight: 0.5 }
    }
}

impl AlgorithmConfig for SSAConfig {
    fn validate(&self) -> Result<(), String> {
        if self.c2_weight < 0.0 { return Err("c2_weight must be non-negative".to_string()); }
        if self.c3_weight < 0.0 { return Err("c3_weight must be non-negative".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { c2_weight: 0.8, c3_weight: 0.4 } }
    fn high_accuracy() -> Self { Self { c2_weight: 1.0, c3_weight: 0.5 } }
}

/// Salp Swarm Optimizer.
pub struct SalpSwarm {
    config: SSAConfig,
    opt_config: OptimizationConfig,
    population: Population,
    food_source: Option<Individual>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl SalpSwarm {
    pub fn new(config: SSAConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(opt_config.population_size, bounds),
            food_source: None,
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
        self.food_source = None;
    }

    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Update food source (best solution found so far)
        if let Some(best) = self.population.best() {
            match &self.food_source {
                Some(food) => {
                    if best.fitness.unwrap_or(f64::INFINITY) < food.fitness.unwrap_or(f64::INFINITY) {
                        self.food_source = Some(best);
                    }
                }
                None => {
                    self.food_source = Some(best);
                }
            }
        }

        let food = match &self.food_source {
            Some(f) => f.clone(),
            None => return Ok(()),
        };

        let n = self.population.len();
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let max_iter = self.opt_config.max_iterations as f64;
        let current_iter = self.iteration as f64;
        let mut rng = rand::thread_rng();

        // c1 decreases exponentially from 2 to 0
        let c1 = 2.0 * (-4.0 * current_iter / max_iter).exp();

        let salps: Vec<Individual> = self.population.individuals().clone();
        let mut new_positions: Vec<Array1<f64>> = Vec::new();

        for i in 0..n {
            let mut new_pos = Array1::zeros(dim);

            if i == 0 {
                // Leader salp - moves towards food source
                for d in 0..dim {
                    let c2: f64 = rng.gen();
                    let c3: f64 = rng.gen();
                    let (lb, ub) = bounds.box_bounds[d];

                    if c3 < 0.5 {
                        new_pos[d] = food.position[d] + c1 * ((ub - lb) * c2 + lb);
                    } else {
                        new_pos[d] = food.position[d] - c1 * ((ub - lb) * c2 + lb);
                    }
                }
            } else {
                // Follower salps - follow the salp ahead in the chain
                // Uses Newton's equation of motion: x_new = 0.5 * (x_current + x_previous)
                for d in 0..dim {
                    new_pos[d] = 0.5 * (salps[i].position[d] + salps[i - 1].position[d]);
                }
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

        // Sort population by fitness to form proper chain
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        {
            let mut pop = self.population.individuals_mut();
            pop.sort_by(|a, b| {
                a.fitness.unwrap_or(f64::INFINITY)
                    .partial_cmp(&b.fitness.unwrap_or(f64::INFINITY))
                    .unwrap()
            });
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

        // Initial evaluation
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Sort to form initial chain
        {
            let mut pop = self.population.individuals_mut();
            pop.sort_by(|a, b| {
                a.fitness.unwrap_or(f64::INFINITY)
                    .partial_cmp(&b.fitness.unwrap_or(f64::INFINITY))
                    .unwrap()
            });
        }

        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }

        self.food_source.clone()
            .or_else(|| self.population.best())
            .ok_or_else(|| OptimizationError::NoSolution("SSA failed".to_string()))
    }
}

impl Algorithm for SalpSwarm {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::SalpSwarm }
    fn name(&self) -> &str { "Salp Swarm Algorithm" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> {
        self.food_source.as_ref().and_then(|f| f.fitness)
            .or_else(|| self.population.best_fitness())
    }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_ssa_optimization() {
        let config = SSAConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(50).with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut ssa = SalpSwarm::new(config, opt_config, bounds).unwrap();
        let solution = ssa.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
