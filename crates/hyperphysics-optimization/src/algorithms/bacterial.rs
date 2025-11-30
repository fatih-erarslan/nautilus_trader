//! Bacterial Foraging Optimization (BFO) - E. coli chemotaxis.
//!
//! # References
//! - Passino (2002): "Biomimicry of Bacterial Foraging for Distributed Optimization"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// BFO configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BFOConfig {
    /// Number of chemotactic steps
    pub nc: u32,
    /// Number of reproduction steps
    pub nre: u32,
    /// Number of elimination-dispersal events
    pub ned: u32,
    /// Swim length (steps in same direction)
    pub ns: u32,
    /// Chemotactic step size
    pub step_size: f64,
    /// Probability of elimination-dispersal
    pub ped: f64,
}

impl Default for BFOConfig {
    fn default() -> Self {
        Self { nc: 25, nre: 4, ned: 2, ns: 4, step_size: 0.1, ped: 0.25 }
    }
}

impl AlgorithmConfig for BFOConfig {
    fn validate(&self) -> Result<(), String> {
        if self.ped < 0.0 || self.ped > 1.0 { return Err("ped must be in [0, 1]".to_string()); }
        Ok(())
    }
    fn hft_optimized() -> Self { Self { nc: 10, nre: 2, ned: 1, ns: 2, step_size: 0.2, ped: 0.2 } }
    fn high_accuracy() -> Self { Self { nc: 50, nre: 6, ned: 3, ns: 6, step_size: 0.05, ped: 0.15 } }
}

/// Bacterial Foraging optimizer.
pub struct BacterialForaging {
    config: BFOConfig,
    opt_config: OptimizationConfig,
    population: Population,
    health: Vec<f64>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl BacterialForaging {
    /// Create a new Bacterial Foraging Optimization optimizer
    pub fn new(config: BFOConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let pop_size = opt_config.population_size;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(pop_size, bounds),
            health: vec![0.0; pop_size],
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize bacterial population
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    fn generate_direction(dim: usize, rng: &mut impl Rng) -> Array1<f64> {
        let dir: Array1<f64> = Array1::from_iter((0..dim).map(|_| rng.gen::<f64>() * 2.0 - 1.0));
        let norm = dir.iter().map(|x| x.powi(2)).sum::<f64>().sqrt();
        if norm > 1e-10 { dir / norm } else { Array1::ones(dim) / (dim as f64).sqrt() }
    }

    /// Execute one chemotaxis-reproduction-elimination cycle
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        let dim = self.population.bounds.dimension();
        let bounds = &self.population.bounds;
        let ranges = bounds.ranges();
        let mut rng = rand::thread_rng();

        // Reset health
        self.health.iter_mut().for_each(|h| *h = 0.0);

        // Chemotaxis loop
        for _j in 0..self.config.nc {
            #[cfg(feature = "parallel")]
            self.population.evaluate_parallel(objective)?;
            #[cfg(not(feature = "parallel"))]
            self.population.evaluate_sequential(objective)?;

            let mut pop = self.population.individuals_mut();
            for (i, bacterium) in pop.iter_mut().enumerate() {
                let cost = bacterium.fitness.unwrap_or(f64::INFINITY);
                self.health[i] += cost;

                // Tumble - generate random direction
                let direction = Self::generate_direction(dim, &mut rng);

                // Swim in that direction
                for _m in 0..self.config.ns {
                    let mut new_pos = Array1::zeros(dim);
                    for d in 0..dim {
                        new_pos[d] = bacterium.position[d] + self.config.step_size * ranges[d] * direction[d];
                    }
                    let repaired = bounds.repair(new_pos.view());
                    let new_cost = objective.evaluate(repaired.view());

                    if new_cost < cost {
                        bacterium.position = repaired;
                        bacterium.fitness = Some(new_cost);
                        self.health[i] += new_cost;
                    } else {
                        break;
                    }
                }
            }
        }

        // Reproduction - keep healthier half
        let n = self.population.len();
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| self.health[a].partial_cmp(&self.health[b]).unwrap());

        let mut pop = self.population.individuals_mut();
        let survivors: Vec<Individual> = indices.iter().take(n / 2).map(|&i| pop[i].clone()).collect();
        let mut new_pop: Vec<Individual> = Vec::with_capacity(n);
        for survivor in survivors {
            new_pop.push(survivor.clone());
            new_pop.push(survivor);
        }
        while new_pop.len() < n {
            new_pop.push(pop[0].clone());
        }
        *pop = new_pop;
        drop(pop);

        // Reset health for new population
        self.health = vec![0.0; n];

        // Elimination-dispersal
        let mut pop = self.population.individuals_mut();
        for bacterium in pop.iter_mut() {
            if rng.gen::<f64>() < self.config.ped {
                let new_pos = Array1::from_iter(bounds.box_bounds.iter().map(|(min, max)| rng.gen_range(*min..*max)));
                bacterium.position = new_pos;
                bacterium.fitness = None;
            }
        }
        drop(pop);

        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }
        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    /// Run the full bacterial foraging optimization with all cycles
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();
        let total_iters = self.config.ned * self.config.nre;
        for _ in 0..total_iters {
            self.step(objective)?;
        }
        self.population.best().ok_or_else(|| OptimizationError::NoSolution("BFO failed".to_string()))
    }
}

impl Algorithm for BacterialForaging {
    fn algorithm_type(&self) -> AlgorithmType { AlgorithmType::BacterialForaging }
    fn name(&self) -> &str { "Bacterial Foraging Optimization" }
    fn is_converged(&self) -> bool { self.converged }
    fn best_fitness(&self) -> Option<f64> { self.population.best_fitness() }
    fn iteration(&self) -> u32 { self.iteration }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_bfo_optimization() {
        let config = BFOConfig { nc: 10, nre: 2, ned: 1, ns: 2, step_size: 0.1, ped: 0.2 };
        let opt_config = OptimizationConfig::default().with_max_iterations(100).with_population_size(20);
        let bounds = Bounds::symmetric(3, 5.12);
        let mut bfo = BacterialForaging::new(config, opt_config, bounds).unwrap();
        let solution = bfo.optimize(&SphereFunction::new(3)).unwrap();
        assert!(solution.fitness.is_some());
    }
}
