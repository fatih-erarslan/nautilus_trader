//! Artificial Bee Colony (ABC) - Honeybee foraging behavior.
//!
//! # References
//! - Karaboga (2005): "An Idea Based On Honey Bee Swarm for Numerical Optimization"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// ABC configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ABCConfig {
    /// Limit for scout bee phase (trials before abandonment)
    pub limit: u32,
    /// Number of employed bees (half of colony)
    pub employed_ratio: f64,
}

impl Default for ABCConfig {
    fn default() -> Self {
        Self { limit: 100, employed_ratio: 0.5 }
    }
}

impl AlgorithmConfig for ABCConfig {
    fn validate(&self) -> Result<(), String> {
        if self.employed_ratio <= 0.0 || self.employed_ratio >= 1.0 {
            return Err("employed_ratio must be in (0, 1)".to_string());
        }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { limit: 50, employed_ratio: 0.5 } }
    fn high_accuracy() -> Self { Self { limit: 200, employed_ratio: 0.5 } }
}

/// Artificial Bee Colony optimizer.
pub struct ArtificialBeeColony {
    config: ABCConfig,
    opt_config: OptimizationConfig,
    population: Population,
    trials: Vec<u32>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl ArtificialBeeColony {
    /// Create a new Artificial Bee Colony optimizer
    pub fn new(config: ABCConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let pop_size = opt_config.population_size;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(pop_size, bounds),
            trials: vec![0; pop_size],
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize bee colony with food sources
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    fn fitness_to_probability(fitness: f64) -> f64 {
        if fitness >= 0.0 { 1.0 / (1.0 + fitness) } else { 1.0 + fitness.abs() }
    }

    /// Execute one iteration with employed, onlooker, and scout bee phases
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let mut rng = rand::thread_rng();
        let n = self.population.len();

        // Employed bee phase
        let pop_snapshot: Vec<Individual> = self.population.individuals().clone();
        let mut new_pop = pop_snapshot.clone();

        for i in 0..n {
            let j = rng.gen_range(0..dim);
            let k = loop {
                let candidate = rng.gen_range(0..n);
                if candidate != i { break candidate; }
            };

            let phi: f64 = rng.gen_range(-1.0..1.0);
            let mut new_pos = pop_snapshot[i].position.clone();
            new_pos[j] = pop_snapshot[i].position[j] + phi * (pop_snapshot[i].position[j] - pop_snapshot[k].position[j]);
            let repaired = bounds.repair(new_pos.view());
            let new_fitness = objective.evaluate(repaired.view());

            if new_fitness < pop_snapshot[i].fitness.unwrap_or(f64::INFINITY) {
                new_pop[i] = Individual::new(repaired);
                new_pop[i].fitness = Some(new_fitness);
                self.trials[i] = 0;
            } else {
                self.trials[i] += 1;
            }
        }

        // Onlooker bee phase
        let probs: Vec<f64> = new_pop.iter()
            .map(|ind| Self::fitness_to_probability(ind.fitness.unwrap_or(f64::INFINITY)))
            .collect();
        let total_prob: f64 = probs.iter().sum();

        for _ in 0..n {
            let r: f64 = rng.gen::<f64>() * total_prob;
            let mut cumulative = 0.0;
            let mut selected = 0;
            for (idx, p) in probs.iter().enumerate() {
                cumulative += p;
                if cumulative >= r {
                    selected = idx;
                    break;
                }
            }

            let j = rng.gen_range(0..dim);
            let k = loop {
                let candidate = rng.gen_range(0..n);
                if candidate != selected { break candidate; }
            };

            let phi: f64 = rng.gen_range(-1.0..1.0);
            let mut new_pos = new_pop[selected].position.clone();
            new_pos[j] = new_pop[selected].position[j] + phi * (new_pop[selected].position[j] - new_pop[k].position[j]);
            let repaired = bounds.repair(new_pos.view());
            let new_fitness = objective.evaluate(repaired.view());

            if new_fitness < new_pop[selected].fitness.unwrap_or(f64::INFINITY) {
                new_pop[selected] = Individual::new(repaired);
                new_pop[selected].fitness = Some(new_fitness);
                self.trials[selected] = 0;
            } else {
                self.trials[selected] += 1;
            }
        }

        // Scout bee phase
        for i in 0..n {
            if self.trials[i] > self.config.limit {
                let new_pos = Array1::from_iter(bounds.box_bounds.iter().map(|(min, max)| rng.gen_range(*min..*max)));
                new_pop[i] = Individual::new(new_pos);
                new_pop[i].fitness = Some(objective.evaluate(new_pop[i].position.view()));
                self.trials[i] = 0;
            }
        }

        {
            let mut pop = self.population.individuals_mut();
            *pop = new_pop;
        }

        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    /// Run the full bee colony optimization until convergence or max iterations
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("ABC failed".to_string()))
    }
}

impl Algorithm for ArtificialBeeColony {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::ArtificialBeeColony }
    fn name(&self) -> &str { "Artificial Bee Colony" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_abc_optimization() {
        let config = ABCConfig::default();
        let opt_config = OptimizationConfig::default().with_max_iterations(100).with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);
        let mut abc = ArtificialBeeColony::new(config, opt_config, bounds).unwrap();
        let solution = abc.optimize(&SphereFunction::new(5)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
