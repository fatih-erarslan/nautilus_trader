//! Wasp Swarm Optimization (WSO) / Wasp Colony Optimization.
//!
//! # References
//! - Pinto et al. (2007): "Wasp Swarm Optimization of Logistic Systems"
//! - Pham & Castellani (2009): "The Bees Algorithm and the Wasp Swarm"
//!
//! Models the foraging behavior of social wasps including:
//! - Scouting behavior (exploration)
//! - Recruitment via dancing (exploitation)
//! - Dominance hierarchy for resource allocation

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// Wasp Swarm Optimization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaspConfig {
    /// Forager ratio (wasps assigned to known food sources)
    pub forager_ratio: f64,
    /// Scout ratio (wasps exploring new areas)
    pub scout_ratio: f64,
    /// Dance threshold (minimum fitness improvement to recruit)
    pub dance_threshold: f64,
    /// Dominance factor (strength of hierarchy influence)
    pub dominance_factor: f64,
    /// Local search radius
    pub local_search_radius: f64,
    /// Abandonment limit (iterations before abandoning food source)
    pub abandonment_limit: u32,
}

impl Default for WaspConfig {
    fn default() -> Self {
        Self {
            forager_ratio: 0.5,
            scout_ratio: 0.2,
            dance_threshold: 0.1,
            dominance_factor: 0.8,
            local_search_radius: 0.1,
            abandonment_limit: 20,
        }
    }
}

impl AlgorithmConfig for WaspConfig {
    fn validate(&self) -> Result<(), String> {
        if self.forager_ratio < 0.0 || self.forager_ratio > 1.0 {
            return Err("forager_ratio must be in [0, 1]".to_string());
        }
        if self.scout_ratio < 0.0 || self.scout_ratio > 1.0 {
            return Err("scout_ratio must be in [0, 1]".to_string());
        }
        if self.forager_ratio + self.scout_ratio > 1.0 {
            return Err("forager_ratio + scout_ratio must not exceed 1.0".to_string());
        }
        if self.dominance_factor < 0.0 || self.dominance_factor > 1.0 {
            return Err("dominance_factor must be in [0, 1]".to_string());
        }
        Ok(())
    }
    fn hft_optimized() -> Self {
        Self {
            forager_ratio: 0.6,
            scout_ratio: 0.15,
            dance_threshold: 0.05,
            dominance_factor: 0.9,
            local_search_radius: 0.15,
            abandonment_limit: 10,
        }
    }
    fn high_accuracy() -> Self {
        Self {
            forager_ratio: 0.4,
            scout_ratio: 0.25,
            dance_threshold: 0.15,
            dominance_factor: 0.7,
            local_search_radius: 0.05,
            abandonment_limit: 30,
        }
    }
}

/// Wasp role in the colony.
#[derive(Debug, Clone, Copy, PartialEq)]
enum WaspRole {
    /// Forages at known food sources
    Forager,
    /// Explores for new food sources
    Scout,
    /// Waits and follows recruiters
    Onlooker,
}

/// Food source tracked by the colony.
#[derive(Clone)]
struct FoodSource {
    position: Array1<f64>,
    fitness: f64,
    trials: u32,
    visitors: u32,
}

/// Wasp Swarm Optimizer.
pub struct WaspSwarm {
    config: WaspConfig,
    opt_config: OptimizationConfig,
    population: Population,
    food_sources: Vec<FoodSource>,
    dominance_scores: Vec<f64>,
    iteration: u32,
    converged: bool,
    fitness_history: Vec<f64>,
}

impl WaspSwarm {
    /// Create a new Wasp Swarm optimizer.
    pub fn new(config: WaspConfig, opt_config: OptimizationConfig, bounds: Bounds) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;
        let pop_size = opt_config.population_size;
        Ok(Self {
            config,
            opt_config: opt_config.clone(),
            population: Population::new(pop_size, bounds),
            food_sources: Vec::new(),
            dominance_scores: vec![0.0; pop_size],
            iteration: 0,
            converged: false,
            fitness_history: Vec::new(),
        })
    }

    /// Initialize the swarm.
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
        self.food_sources.clear();
        self.dominance_scores = vec![0.0; self.opt_config.population_size];
    }

    /// Assign roles to wasps based on dominance hierarchy.
    fn assign_roles(&self, n: usize) -> Vec<WaspRole> {
        let forager_count = (n as f64 * self.config.forager_ratio) as usize;
        let scout_count = (n as f64 * self.config.scout_ratio) as usize;

        // Sort by dominance score (higher = more dominant)
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.dominance_scores[b]
                .partial_cmp(&self.dominance_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut roles = vec![WaspRole::Onlooker; n];
        for (rank, &idx) in indices.iter().enumerate() {
            if rank < forager_count {
                roles[idx] = WaspRole::Forager;
            } else if rank < forager_count + scout_count {
                roles[idx] = WaspRole::Scout;
            }
        }
        roles
    }

    /// Update dominance scores based on fitness.
    fn update_dominance(&mut self) {
        let n = self.population.len();
        let individuals = self.population.individuals();

        // Get fitness values, handling None
        let fitnesses: Vec<f64> = individuals
            .iter()
            .map(|ind| ind.fitness.unwrap_or(f64::INFINITY))
            .collect();

        let best_fit = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let worst_fit = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = (worst_fit - best_fit).max(1e-10);

        // Update dominance scores based on relative fitness
        for i in 0..n {
            let normalized = (worst_fit - fitnesses[i]) / range;
            self.dominance_scores[i] = self.config.dominance_factor * self.dominance_scores[i]
                + (1.0 - self.config.dominance_factor) * normalized;
        }
    }

    /// Perform forager behavior (exploit known food sources).
    fn forager_step(
        &self,
        wasp: &Individual,
        bounds: &Bounds,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let dim = bounds.dimension();
        let ranges = bounds.ranges();
        let mut new_pos = wasp.position.clone();

        // Move towards best known food source with local perturbation
        if let Some(best_food) = self.food_sources.first() {
            for d in 0..dim {
                let attraction = rng.gen::<f64>() * self.config.dominance_factor;
                let perturbation = (rng.gen::<f64>() - 0.5) * self.config.local_search_radius * ranges[d];
                new_pos[d] = wasp.position[d]
                    + attraction * (best_food.position[d] - wasp.position[d])
                    + perturbation;
            }
        } else {
            // No food sources, perform local search
            for d in 0..dim {
                let perturbation = (rng.gen::<f64>() - 0.5) * self.config.local_search_radius * ranges[d];
                new_pos[d] = wasp.position[d] + perturbation;
            }
        }

        bounds.repair(new_pos.view())
    }

    /// Perform scout behavior (explore new areas).
    fn scout_step(
        &self,
        _wasp: &Individual,
        bounds: &Bounds,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        // Scouts explore randomly within bounds (Lévy-like exploration)
        let dim = bounds.dimension();
        let mut new_pos = Array1::zeros(dim);

        for d in 0..dim {
            let (min, max) = bounds.box_bounds[d];
            new_pos[d] = rng.gen_range(min..max);
        }

        new_pos
    }

    /// Perform onlooker behavior (follow dance recruitment).
    fn onlooker_step(
        &self,
        wasp: &Individual,
        bounds: &Bounds,
        rng: &mut impl Rng,
    ) -> Array1<f64> {
        let dim = bounds.dimension();
        let ranges = bounds.ranges();

        // Follow recruited food source based on quality (roulette wheel selection)
        if !self.food_sources.is_empty() {
            let total_quality: f64 = self.food_sources
                .iter()
                .map(|f| 1.0 / (1.0 + f.fitness))
                .sum();

            let r = rng.gen::<f64>() * total_quality;
            let mut cumulative = 0.0;
            let mut selected_food = &self.food_sources[0];

            for food in &self.food_sources {
                cumulative += 1.0 / (1.0 + food.fitness);
                if cumulative >= r {
                    selected_food = food;
                    break;
                }
            }

            // Move towards selected food source
            let mut new_pos = wasp.position.clone();
            for d in 0..dim {
                let step = rng.gen::<f64>() * (selected_food.position[d] - wasp.position[d]);
                new_pos[d] = wasp.position[d] + step;
            }
            bounds.repair(new_pos.view())
        } else {
            // No food sources, stay put with small perturbation
            let mut new_pos = wasp.position.clone();
            for d in 0..dim {
                let perturbation = (rng.gen::<f64>() - 0.5) * 0.01 * ranges[d];
                new_pos[d] = wasp.position[d] + perturbation;
            }
            bounds.repair(new_pos.view())
        }
    }

    /// Perform one optimization step.
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Update dominance hierarchy
        self.update_dominance();

        let n = self.population.len();
        let bounds = &self.population.bounds;
        let roles = self.assign_roles(n);
        let mut rng = rand::thread_rng();

        let individuals: Vec<Individual> = self.population.individuals().clone();
        let mut new_positions: Vec<Array1<f64>> = Vec::with_capacity(n);

        // Each wasp acts according to its role
        for (i, wasp) in individuals.iter().enumerate() {
            let new_pos = match roles[i] {
                WaspRole::Forager => self.forager_step(wasp, bounds, &mut rng),
                WaspRole::Scout => self.scout_step(wasp, bounds, &mut rng),
                WaspRole::Onlooker => self.onlooker_step(wasp, bounds, &mut rng),
            };
            new_positions.push(new_pos);
        }

        // Evaluate new positions and update if improved
        let mut improved_sources: Vec<FoodSource> = Vec::new();
        {
            let mut pop = self.population.individuals_mut();
            for (i, ind) in pop.iter_mut().enumerate() {
                let new_fitness = objective.evaluate(new_positions[i].view());
                let old_fitness = ind.fitness.unwrap_or(f64::INFINITY);

                if new_fitness < old_fitness {
                    ind.position = new_positions[i].clone();
                    ind.fitness = Some(new_fitness);

                    // Register as new food source if improvement is significant
                    let improvement = (old_fitness - new_fitness) / old_fitness.abs().max(1e-10);
                    if improvement > self.config.dance_threshold {
                        improved_sources.push(FoodSource {
                            position: ind.position.clone(),
                            fitness: new_fitness,
                            trials: 0,
                            visitors: 1,
                        });
                    }
                }
            }
        }

        // Update food sources
        self.update_food_sources(improved_sources);

        // Track best fitness
        if let Some(f) = self.population.best_fitness() {
            self.fitness_history.push(f);
        }

        self.iteration += 1;
        self.population.next_generation();
        Ok(())
    }

    /// Update and maintain food sources.
    fn update_food_sources(&mut self, new_sources: Vec<FoodSource>) {
        // Add new sources
        for source in new_sources {
            // Check if similar source exists
            let exists = self.food_sources.iter().any(|f| {
                f.position
                    .iter()
                    .zip(source.position.iter())
                    .all(|(a, b)| (a - b).abs() < 0.01)
            });

            if !exists {
                self.food_sources.push(source);
            }
        }

        // Increment trials for existing sources
        for source in &mut self.food_sources {
            source.trials += 1;
        }

        // Remove abandoned sources
        self.food_sources
            .retain(|s| s.trials < self.config.abandonment_limit);

        // Sort by fitness (best first)
        self.food_sources
            .sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        // Keep only top sources (limit memory usage)
        let max_sources = self.opt_config.population_size / 2;
        self.food_sources.truncate(max_sources);
    }

    /// Run the full optimization process.
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();

        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }

        self.population
            .best()
            .ok_or_else(|| OptimizationError::NoSolution("Wasp Swarm failed".to_string()))
    }
}

impl Algorithm for WaspSwarm {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::WaspSwarm
    }

    fn name(&self) -> &str {
        "Wasp Swarm Optimization"
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn best_fitness(&self) -> Option<f64> {
        self.population.best_fitness()
    }

    fn iteration(&self) -> u32 {
        self.iteration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_wasp_config_validation() {
        let config = WaspConfig::default();
        assert!(config.validate().is_ok());

        let invalid = WaspConfig {
            forager_ratio: 0.8,
            scout_ratio: 0.5,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_wasp_optimization() {
        let config = WaspConfig::default();
        let opt_config = OptimizationConfig::default()
            .with_max_iterations(50)
            .with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);

        let mut wso = WaspSwarm::new(config, opt_config, bounds).unwrap();
        let solution = wso.optimize(&SphereFunction::new(5)).unwrap();

        assert!(solution.fitness.is_some());
        assert!(solution.fitness.unwrap() < 10.0); // Should find reasonable solution
    }

    #[test]
    fn test_wasp_hft_config() {
        let config = WaspConfig::hft_optimized();
        assert!(config.validate().is_ok());
        assert!(config.abandonment_limit < WaspConfig::default().abandonment_limit);
    }
}
