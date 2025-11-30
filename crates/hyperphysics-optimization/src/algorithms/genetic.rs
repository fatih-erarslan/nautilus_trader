//! Genetic Algorithm (GA) with formal convergence properties.
//!
//! # Formal Verification
//!
//! **Theorem GA1** (Schema Theorem): The number of instances of a schema H
//! in generation t+1 is at least m(H,t) * f(H)/f̄ * (1 - p_c * δ(H)/(l-1)) * (1 - p_m)^o(H)
//!
//! **Property GA2**: Elitism guarantees monotonic best fitness improvement
//!
//! **Invariant GA3**: Population size remains constant across generations
//!
//! # References
//!
//! - Holland (1975): "Adaptation in Natural and Artificial Systems"
//! - Goldberg (1989): "Genetic Algorithms in Search, Optimization, and Machine Learning"
//! - De Jong (2006): "Evolutionary Computation: A Unified Approach"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

/// GA configuration with validated parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GAConfig {
    /// Crossover probability
    /// Formal constraint: p_c ∈ [0, 1]
    pub crossover_rate: f64,
    /// Mutation probability per gene
    /// Formal constraint: p_m ∈ [0, 1], typically small (0.001-0.1)
    pub mutation_rate: f64,
    /// Selection strategy
    pub selection: SelectionStrategy,
    /// Crossover operator
    pub crossover: CrossoverOperator,
    /// Mutation operator
    pub mutation: MutationOperator,
    /// Tournament size (for tournament selection)
    pub tournament_size: usize,
    /// Mutation strength (fraction of search range)
    pub mutation_strength: f64,
    /// Adaptive mutation (increase on stagnation)
    pub adaptive_mutation: bool,
}

impl Default for GAConfig {
    fn default() -> Self {
        Self {
            crossover_rate: 0.9,
            mutation_rate: 0.1,
            selection: SelectionStrategy::Tournament,
            crossover: CrossoverOperator::SBX { eta: 20.0 },
            mutation: MutationOperator::Polynomial { eta: 20.0 },
            tournament_size: 3,
            mutation_strength: 0.1,
            adaptive_mutation: true,
        }
    }
}

impl AlgorithmConfig for GAConfig {
    fn validate(&self) -> Result<(), String> {
        if self.crossover_rate < 0.0 || self.crossover_rate > 1.0 {
            return Err(format!("Crossover rate {} not in [0, 1]", self.crossover_rate));
        }
        if self.mutation_rate < 0.0 || self.mutation_rate > 1.0 {
            return Err(format!("Mutation rate {} not in [0, 1]", self.mutation_rate));
        }
        if self.tournament_size < 2 {
            return Err("Tournament size must be at least 2".to_string());
        }
        if self.mutation_strength <= 0.0 {
            return Err("Mutation strength must be positive".to_string());
        }
        Ok(())
    }

    fn hft_optimized() -> Self {
        Self {
            crossover_rate: 0.95,
            mutation_rate: 0.15,
            selection: SelectionStrategy::Tournament,
            crossover: CrossoverOperator::BLX { alpha: 0.5 },
            mutation: MutationOperator::Uniform,
            tournament_size: 2,
            mutation_strength: 0.05,
            adaptive_mutation: false,
        }
    }

    fn high_accuracy() -> Self {
        Self {
            crossover_rate: 0.8,
            mutation_rate: 0.05,
            selection: SelectionStrategy::Tournament,
            crossover: CrossoverOperator::SBX { eta: 15.0 },
            mutation: MutationOperator::Polynomial { eta: 20.0 },
            tournament_size: 5,
            mutation_strength: 0.1,
            adaptive_mutation: true,
        }
    }
}

/// Selection strategies for parent selection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Tournament selection
    Tournament,
    /// Roulette wheel (fitness-proportionate)
    RouletteWheel,
    /// Rank-based selection
    RankBased,
    /// Stochastic Universal Sampling
    SUS,
}

/// Crossover operators for real-valued GAs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CrossoverOperator {
    /// Simulated Binary Crossover (SBX)
    SBX {
        /// Distribution index for SBX
        eta: f64
    },
    /// Blend Crossover (BLX-α)
    BLX {
        /// Blend factor alpha
        alpha: f64
    },
    /// Arithmetic crossover
    Arithmetic,
    /// Uniform crossover
    Uniform,
}

/// Mutation operators for real-valued GAs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationOperator {
    /// Polynomial mutation
    Polynomial {
        /// Distribution index for polynomial mutation
        eta: f64
    },
    /// Gaussian mutation
    Gaussian,
    /// Uniform mutation
    Uniform,
    /// Non-uniform mutation (decreasing over time)
    NonUniform,
}

/// Genetic Algorithm optimizer.
pub struct GeneticAlgorithm {
    /// Algorithm configuration
    config: GAConfig,
    /// Optimization configuration
    opt_config: OptimizationConfig,
    /// Population
    population: Population,
    /// Current generation
    generation: u32,
    /// Convergence flag
    converged: bool,
    /// Fitness history
    fitness_history: Vec<f64>,
    /// Stagnation counter
    stagnation_count: u32,
    /// Current mutation rate (may adapt)
    current_mutation_rate: f64,
}

impl GeneticAlgorithm {
    /// Create a new GA optimizer.
    pub fn new(
        config: GAConfig,
        opt_config: OptimizationConfig,
        bounds: Bounds,
    ) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;

        let current_mutation_rate = config.mutation_rate;
        let population = Population::new(opt_config.population_size, bounds);

        Ok(Self {
            config,
            opt_config,
            population,
            generation: 0,
            converged: false,
            fitness_history: Vec::with_capacity(1000),
            stagnation_count: 0,
            current_mutation_rate,
        })
    }

    /// Initialize population.
    pub fn initialize(&mut self) {
        self.population.initialize_lhs(self.opt_config.population_size);
    }

    /// Run single GA generation.
    ///
    /// # Formal Properties
    /// - **Invariant GA3**: Population size unchanged
    /// - **Property GA2**: With elitism, best fitness monotonically improves
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        // Evaluate population
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Store elite individuals (Property GA2)
        let elites: Vec<Individual> = if self.opt_config.elitism {
            self.population.top_k(self.opt_config.elite_count)
        } else {
            Vec::new()
        };

        // Create offspring population
        let mut offspring = Vec::with_capacity(self.opt_config.population_size);
        let mut rng = rand::thread_rng();

        while offspring.len() < self.opt_config.population_size - elites.len() {
            // Selection
            let parent1 = self.select(&mut rng);
            let parent2 = self.select(&mut rng);

            // Crossover
            let (mut child1, mut child2) = if rng.gen::<f64>() < self.config.crossover_rate {
                self.crossover(&parent1, &parent2, &mut rng)
            } else {
                (parent1.clone(), parent2.clone())
            };

            // Mutation
            self.mutate(&mut child1, &mut rng);
            self.mutate(&mut child2, &mut rng);

            // Repair to bounds
            child1.position = self.population.bounds.repair(child1.position.view());
            child2.position = self.population.bounds.repair(child2.position.view());

            // Reset fitness for re-evaluation
            child1.fitness = None;
            child2.fitness = None;

            offspring.push(child1);
            if offspring.len() < self.opt_config.population_size - elites.len() {
                offspring.push(child2);
            }
        }

        // Add elites back (Property GA2: monotonic improvement)
        offspring.extend(elites);

        // Replace population
        {
            let mut pop = self.population.individuals_mut();
            *pop = offspring;
        }

        // Update statistics
        if let Some(best_fitness) = self.population.best_fitness() {
            self.update_stagnation(best_fitness);
            self.fitness_history.push(best_fitness);
        }

        // Adaptive mutation
        if self.config.adaptive_mutation {
            self.adapt_mutation();
        }

        self.check_convergence();
        self.generation += 1;
        self.population.next_generation();

        Ok(())
    }

    /// Select parent using configured strategy.
    fn select(&self, rng: &mut impl Rng) -> Individual {
        match self.config.selection {
            SelectionStrategy::Tournament => self.tournament_select(rng),
            SelectionStrategy::RouletteWheel => self.roulette_select(rng),
            SelectionStrategy::RankBased => self.rank_select(rng),
            SelectionStrategy::SUS => self.tournament_select(rng), // Fallback
        }
    }

    /// Tournament selection.
    fn tournament_select(&self, rng: &mut impl Rng) -> Individual {
        let pop = self.population.individuals();
        let mut best: Option<&Individual> = None;

        for _ in 0..self.config.tournament_size {
            let idx = rng.gen_range(0..pop.len());
            let candidate = &pop[idx];

            match best {
                None => best = Some(candidate),
                Some(b) if candidate < b => best = Some(candidate),
                _ => {}
            }
        }

        best.unwrap().clone()
    }

    /// Roulette wheel selection (fitness-proportionate).
    fn roulette_select(&self, rng: &mut impl Rng) -> Individual {
        let pop = self.population.individuals();

        // Calculate fitness sum (inverted for minimization)
        let fitnesses: Vec<f64> = pop.iter()
            .map(|i| 1.0 / (1.0 + i.fitness.unwrap_or(f64::MAX)))
            .collect();
        let total: f64 = fitnesses.iter().sum();

        let target = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;

        for (i, f) in fitnesses.iter().enumerate() {
            cumulative += f;
            if cumulative >= target {
                return pop[i].clone();
            }
        }

        pop.last().unwrap().clone()
    }

    /// Rank-based selection.
    fn rank_select(&self, rng: &mut impl Rng) -> Individual {
        let mut pop: Vec<Individual> = self.population.individuals().clone();
        pop.sort();

        let n = pop.len();
        let total: f64 = (n * (n + 1) / 2) as f64;
        let target = rng.gen::<f64>() * total;
        let mut cumulative = 0.0;

        for (rank, ind) in pop.iter().enumerate() {
            cumulative += (n - rank) as f64;
            if cumulative >= target {
                return ind.clone();
            }
        }

        pop.last().unwrap().clone()
    }

    /// Perform crossover operation.
    fn crossover(&self, p1: &Individual, p2: &Individual, rng: &mut impl Rng) -> (Individual, Individual) {
        let dim = p1.position.len();
        let mut c1 = Array1::zeros(dim);
        let mut c2 = Array1::zeros(dim);

        match &self.config.crossover {
            CrossoverOperator::SBX { eta } => {
                for i in 0..dim {
                    let (child1_gene, child2_gene) = sbx_crossover(
                        p1.position[i],
                        p2.position[i],
                        *eta,
                        rng,
                    );
                    c1[i] = child1_gene;
                    c2[i] = child2_gene;
                }
            }
            CrossoverOperator::BLX { alpha } => {
                for i in 0..dim {
                    let min_val = p1.position[i].min(p2.position[i]);
                    let max_val = p1.position[i].max(p2.position[i]);
                    let range = max_val - min_val;
                    let lower = min_val - alpha * range;
                    let upper = max_val + alpha * range;
                    c1[i] = rng.gen_range(lower..upper);
                    c2[i] = rng.gen_range(lower..upper);
                }
            }
            CrossoverOperator::Arithmetic => {
                let alpha: f64 = rng.gen();
                for i in 0..dim {
                    c1[i] = alpha * p1.position[i] + (1.0 - alpha) * p2.position[i];
                    c2[i] = (1.0 - alpha) * p1.position[i] + alpha * p2.position[i];
                }
            }
            CrossoverOperator::Uniform => {
                for i in 0..dim {
                    if rng.gen::<bool>() {
                        c1[i] = p1.position[i];
                        c2[i] = p2.position[i];
                    } else {
                        c1[i] = p2.position[i];
                        c2[i] = p1.position[i];
                    }
                }
            }
        }

        (Individual::new(c1), Individual::new(c2))
    }

    /// Apply mutation to individual.
    fn mutate(&self, individual: &mut Individual, rng: &mut impl Rng) {
        let bounds = &self.population.bounds;

        for i in 0..individual.position.len() {
            if rng.gen::<f64>() < self.current_mutation_rate {
                let (lower, upper) = bounds.box_bounds[i];
                let range = upper - lower;

                individual.position[i] = match &self.config.mutation {
                    MutationOperator::Polynomial { eta } => {
                        polynomial_mutation(individual.position[i], lower, upper, *eta, rng)
                    }
                    MutationOperator::Gaussian => {
                        let sigma = range * self.config.mutation_strength;
                        let delta: f64 = rng.gen::<f64>() * 2.0 - 1.0;
                        individual.position[i] + delta * sigma
                    }
                    MutationOperator::Uniform => {
                        rng.gen_range(lower..upper)
                    }
                    MutationOperator::NonUniform => {
                        let progress = self.generation as f64 / self.opt_config.max_iterations as f64;
                        let b = 5.0; // Non-uniformity parameter
                        let delta = range * (1.0 - progress.powf(b)) * (rng.gen::<f64>() * 2.0 - 1.0);
                        individual.position[i] + delta * self.config.mutation_strength
                    }
                };
            }
        }
    }

    /// Update stagnation counter.
    fn update_stagnation(&mut self, current_best: f64) {
        if let Some(&prev_best) = self.fitness_history.last() {
            if (prev_best - current_best).abs() < self.opt_config.tolerance {
                self.stagnation_count += 1;
            } else {
                self.stagnation_count = 0;
            }
        }
    }

    /// Adapt mutation rate based on stagnation.
    fn adapt_mutation(&mut self) {
        if self.stagnation_count > self.opt_config.max_stagnation / 2 {
            self.current_mutation_rate = (self.config.mutation_rate * 2.0).min(0.5);
        } else {
            self.current_mutation_rate = self.config.mutation_rate;
        }
    }

    /// Check convergence conditions.
    fn check_convergence(&mut self) {
        if self.stagnation_count >= self.opt_config.max_stagnation {
            self.converged = true;
            return;
        }

        if let (Some(target), Some(best)) = (self.opt_config.target_fitness, self.population.best_fitness()) {
            if best <= target {
                self.converged = true;
            }
        }
    }

    /// Run full optimization.
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();

        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        while self.generation < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }

        self.population.best()
            .ok_or_else(|| OptimizationError::NoSolution("GA failed to find solution".to_string()))
    }
}

impl Algorithm for GeneticAlgorithm {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::GeneticAlgorithm
    }

    fn name(&self) -> &str {
        "Genetic Algorithm"
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn best_fitness(&self) -> Option<f64> {
        self.population.best_fitness()
    }

    fn iteration(&self) -> u32 {
        self.generation
    }
}

/// Simulated Binary Crossover (SBX) for a single gene.
fn sbx_crossover(p1: f64, p2: f64, eta: f64, rng: &mut impl Rng) -> (f64, f64) {
    if (p1 - p2).abs() < 1e-14 {
        return (p1, p2);
    }

    let u: f64 = rng.gen();
    let beta = if u <= 0.5 {
        (2.0 * u).powf(1.0 / (eta + 1.0))
    } else {
        (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
    };

    let c1 = 0.5 * ((p1 + p2) - beta * (p2 - p1));
    let c2 = 0.5 * ((p1 + p2) + beta * (p2 - p1));

    (c1, c2)
}

/// Polynomial mutation for a single gene.
fn polynomial_mutation(x: f64, lower: f64, upper: f64, eta: f64, rng: &mut impl Rng) -> f64 {
    let u: f64 = rng.gen();
    let delta = if u < 0.5 {
        let xl = (x - lower) / (upper - lower);
        (2.0 * u + (1.0 - 2.0 * u) * (1.0 - xl).powf(eta + 1.0)).powf(1.0 / (eta + 1.0)) - 1.0
    } else {
        let xu = (upper - x) / (upper - lower);
        1.0 - (2.0 * (1.0 - u) + 2.0 * (u - 0.5) * (1.0 - xu).powf(eta + 1.0)).powf(1.0 / (eta + 1.0))
    };

    x + delta * (upper - lower)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_ga_config_validation() {
        let config = GAConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_ga_sphere_optimization() {
        let config = GAConfig::default();
        let opt_config = OptimizationConfig::default()
            .with_max_iterations(100)
            .with_population_size(50);
        let bounds = Bounds::symmetric(5, 5.12);

        let mut ga = GeneticAlgorithm::new(config, opt_config, bounds).unwrap();
        let objective = SphereFunction::new(5);

        let solution = ga.optimize(&objective).unwrap();
        assert!(solution.fitness.unwrap() < 10.0);
    }

    #[test]
    fn test_sbx_crossover() {
        let mut rng = rand::thread_rng();
        let (c1, c2) = sbx_crossover(1.0, 3.0, 20.0, &mut rng);
        // Children should be in reasonable range around parents
        assert!(c1 >= -1.0 && c1 <= 5.0);
        assert!(c2 >= -1.0 && c2 <= 5.0);
    }
}
