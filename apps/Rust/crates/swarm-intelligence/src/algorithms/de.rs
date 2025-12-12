//! Differential Evolution (DE) implementation
//!
//! DE is a population-based optimization algorithm that uses difference vectors
//! between population members to guide the search process. It's particularly
//! effective for continuous optimization problems.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rayon::prelude::*;
use rand::prelude::*;

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Population, Individual, BasicIndividual, Position, AlgorithmMetrics,
    AdaptiveAlgorithm, ParallelAlgorithm, AdaptationStrategy
};
use crate::validate_parameter;

/// Differential Evolution mutation strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DeStrategy {
    /// DE/rand/1: v = x_r1 + F * (x_r2 - x_r3)
    Rand1,
    /// DE/best/1: v = x_best + F * (x_r1 - x_r2)
    Best1,
    /// DE/current-to-best/1: v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
    CurrentToBest1,
    /// DE/rand/2: v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
    Rand2,
    /// DE/best/2: v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
    Best2,
    /// DE/rand-to-best/1: v = x_r1 + F * (x_best - x_r1) + F * (x_r2 - x_r3)
    RandToBest1,
}

/// Crossover strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CrossoverStrategy {
    /// Binomial crossover
    Binomial,
    /// Exponential crossover
    Exponential,
}

/// DE algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeParameters {
    /// Population size
    pub population_size: usize,
    
    /// Differential weight factor (F)
    pub differential_weight: f64,
    
    /// Crossover probability (CR)
    pub crossover_probability: f64,
    
    /// Mutation strategy
    pub strategy: DeStrategy,
    
    /// Crossover strategy
    pub crossover_strategy: CrossoverStrategy,
    
    /// Enable self-adaptive parameters
    pub self_adaptive: bool,
    
    /// Enable jittering of F parameter
    pub jitter_factor: f64,
    
    /// Minimum differential weight
    pub min_differential_weight: f64,
    
    /// Maximum differential weight
    pub max_differential_weight: f64,
    
    /// Enable opposition-based learning
    pub opposition_based: bool,
    
    /// Enable bounds handling
    pub bounds_handling: BoundsHandling,
    
    /// Enable archive for diversity
    pub use_archive: bool,
    
    /// Archive size factor
    pub archive_size_factor: f64,
}

/// Bounds handling strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BoundsHandling {
    /// Clip to bounds
    Clip,
    /// Reflect from bounds
    Reflect,
    /// Wrap around bounds
    Wrap,
    /// Reinitialize violating components
    Reinitialize,
    /// Midpoint between target and bound
    Midpoint,
}

impl Default for DeParameters {
    fn default() -> Self {
        Self {
            population_size: 30,
            differential_weight: 0.8,
            crossover_probability: 0.9,
            strategy: DeStrategy::Rand1,
            crossover_strategy: CrossoverStrategy::Binomial,
            self_adaptive: false,
            jitter_factor: 0.1,
            min_differential_weight: 0.1,
            max_differential_weight: 2.0,
            opposition_based: false,
            bounds_handling: BoundsHandling::Clip,
            use_archive: false,
            archive_size_factor: 1.0,
        }
    }
}

impl DeParameters {
    /// Validate parameters
    pub fn validate(&self) -> Result<(), SwarmError> {
        validate_parameter!(self.population_size, "population_size", 4, 10000);
        validate_parameter!(self.differential_weight, "differential_weight", 0.0, 3.0);
        validate_parameter!(self.crossover_probability, "crossover_probability", 0.0, 1.0);
        validate_parameter!(self.jitter_factor, "jitter_factor", 0.0, 1.0);
        validate_parameter!(self.min_differential_weight, "min_differential_weight", 0.0, 2.0);
        validate_parameter!(self.max_differential_weight, "max_differential_weight", 
                          self.min_differential_weight, 3.0);
        validate_parameter!(self.archive_size_factor, "archive_size_factor", 0.0, 10.0);
        
        Ok(())
    }
    
    /// Create builder for DE parameters
    pub fn builder() -> DeParametersBuilder {
        DeParametersBuilder::new()
    }
}

/// Builder for DE parameters
pub struct DeParametersBuilder {
    params: DeParameters,
}

impl DeParametersBuilder {
    pub fn new() -> Self {
        Self {
            params: DeParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.params.population_size = size;
        self
    }
    
    pub fn differential_weight(mut self, weight: f64) -> Self {
        self.params.differential_weight = weight;
        self
    }
    
    pub fn crossover_probability(mut self, prob: f64) -> Self {
        self.params.crossover_probability = prob;
        self
    }
    
    pub fn strategy(mut self, strategy: DeStrategy) -> Self {
        self.params.strategy = strategy;
        self
    }
    
    pub fn crossover_strategy(mut self, strategy: CrossoverStrategy) -> Self {
        self.params.crossover_strategy = strategy;
        self
    }
    
    pub fn self_adaptive(mut self, adaptive: bool) -> Self {
        self.params.self_adaptive = adaptive;
        self
    }
    
    pub fn opposition_based(mut self, opposition: bool) -> Self {
        self.params.opposition_based = opposition;
        self
    }
    
    pub fn bounds_handling(mut self, handling: BoundsHandling) -> Self {
        self.params.bounds_handling = handling;
        self
    }
    
    pub fn build(self) -> Result<DeParameters, SwarmError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

/// DE individual with self-adaptive parameters
#[derive(Debug, Clone)]
pub struct DeIndividual {
    /// Current position
    position: Position,
    
    /// Fitness value
    fitness: f64,
    
    /// Trial vector (candidate solution)
    trial_vector: Option<Position>,
    
    /// Trial fitness
    trial_fitness: f64,
    
    /// Self-adaptive differential weight
    self_f: f64,
    
    /// Self-adaptive crossover probability
    self_cr: f64,
    
    /// Success count for adaptation
    success_count: usize,
    
    /// Total attempts for adaptation
    total_attempts: usize,
}

impl DeIndividual {
    /// Create a new DE individual
    pub fn new(position: Position) -> Self {
        Self {
            position,
            fitness: f64::INFINITY,
            trial_vector: None,
            trial_fitness: f64::INFINITY,
            self_f: 0.8,
            self_cr: 0.9,
            success_count: 0,
            total_attempts: 0,
        }
    }
    
    /// Create random individual within bounds
    pub fn random(dimensions: usize, lower_bound: f64, upper_bound: f64) -> Self {
        let mut rng = thread_rng();
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(lower_bound..=upper_bound)
        });
        Self::new(position)
    }
    
    /// Get trial vector
    pub fn trial_vector(&self) -> Option<&Position> {
        self.trial_vector.as_ref()
    }
    
    /// Set trial vector
    pub fn set_trial_vector(&mut self, trial: Position) {
        self.trial_vector = Some(trial);
    }
    
    /// Get trial fitness
    pub fn trial_fitness(&self) -> f64 {
        self.trial_fitness
    }
    
    /// Set trial fitness
    pub fn set_trial_fitness(&mut self, fitness: f64) {
        self.trial_fitness = fitness;
    }
    
    /// Accept trial if better
    pub fn accept_trial_if_better(&mut self) -> bool {
        if self.trial_fitness < self.fitness {
            if let Some(trial) = self.trial_vector.take() {
                self.position = trial;
                self.fitness = self.trial_fitness;
                self.success_count += 1;
                self.total_attempts += 1;
                return true;
            }
        }
        self.total_attempts += 1;
        false
    }
    
    /// Update self-adaptive parameters
    pub fn update_adaptive_parameters(&mut self, tau1: f64, tau2: f64) {
        let mut rng = thread_rng();
        
        // Self-adaptive F
        if rng.gen::<f64>() < tau1 {
            self.self_f = 0.1 + rng.gen::<f64>() * 0.9;
        }
        
        // Self-adaptive CR
        if rng.gen::<f64>() < tau2 {
            self.self_cr = rng.gen::<f64>();
        }
    }
    
    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_attempts > 0 {
            self.success_count as f64 / self.total_attempts as f64
        } else {
            0.0
        }
    }
    
    /// Get self-adaptive F parameter
    pub fn self_f(&self) -> f64 {
        self.self_f
    }
    
    /// Get self-adaptive CR parameter
    pub fn self_cr(&self) -> f64 {
        self.self_cr
    }
}

impl Individual for DeIndividual {
    fn position(&self) -> &Position {
        &self.position
    }
    
    fn position_mut(&mut self) -> &mut Position {
        &mut self.position
    }
    
    fn fitness(&self) -> &f64 {
        &self.fitness
    }
    
    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// Differential Evolution algorithm
#[derive(Debug, Clone)]
pub struct DifferentialEvolution {
    /// Algorithm parameters
    parameters: DeParameters,
    
    /// Current population
    population: Population<DeIndividual>,
    
    /// Archive for diversity (optional)
    archive: Vec<DeIndividual>,
    
    /// Best individual found
    best_individual: Option<Arc<DeIndividual>>,
    
    /// Best fitness
    best_fitness: f64,
    
    /// Current iteration
    iteration: usize,
    
    /// Optimization problem
    problem: Option<Arc<OptimizationProblem>>,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
    
    /// Success history for adaptation
    success_history: Vec<bool>,
    
    /// F values history for adaptation
    f_history: Vec<f64>,
    
    /// CR values history for adaptation
    cr_history: Vec<f64>,
}

impl DifferentialEvolution {
    /// Create a new DE algorithm with default parameters
    pub fn new() -> Self {
        Self::with_parameters(DeParameters::default())
    }
    
    /// Create DE with specific parameters
    pub fn with_parameters(parameters: DeParameters) -> Self {
        Self {
            parameters,
            population: Population::new(),
            archive: Vec::new(),
            best_individual: None,
            best_fitness: f64::INFINITY,
            iteration: 0,
            problem: None,
            metrics: AlgorithmMetrics::default(),
            success_history: Vec::new(),
            f_history: Vec::new(),
            cr_history: Vec::new(),
        }
    }
    
    /// Builder pattern for DE construction
    pub fn builder() -> DeBuilder {
        DeBuilder::new()
    }
    
    /// Generate mutant vector using specified strategy
    fn generate_mutant(&self, target_idx: usize, rng: &mut ThreadRng) -> Result<Position, SwarmError> {
        let pop_size = self.population.size();
        let dimensions = self.population.individuals[target_idx].position().len();
        
        // Select random indices different from target
        let mut indices = Vec::new();
        while indices.len() < 5 {
            let idx = rng.gen_range(0..pop_size);
            if idx != target_idx && !indices.contains(&idx) {
                indices.push(idx);
            }
        }
        
        let best_idx = self.population.individuals.iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.fitness().partial_cmp(b.fitness()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        // Get differential weight (with jittering if enabled)
        let f = if self.parameters.jitter_factor > 0.0 {
            let jitter = rng.gen_range(-self.parameters.jitter_factor..self.parameters.jitter_factor);
            (self.parameters.differential_weight + jitter).clamp(0.1, 2.0)
        } else {
            self.parameters.differential_weight
        };
        
        // Apply mutation strategy
        let mutant = match self.parameters.strategy {
            DeStrategy::Rand1 => {
                // v = x_r1 + F * (x_r2 - x_r3)
                let x_r1 = self.population.individuals[indices[0]].position();
                let x_r2 = self.population.individuals[indices[1]].position();
                let x_r3 = self.population.individuals[indices[2]].position();
                x_r1 + f * (x_r2 - x_r3)
            }
            DeStrategy::Best1 => {
                // v = x_best + F * (x_r1 - x_r2)
                let x_best = self.population.individuals[best_idx].position();
                let x_r1 = self.population.individuals[indices[0]].position();
                let x_r2 = self.population.individuals[indices[1]].position();
                x_best + f * (x_r1 - x_r2)
            }
            DeStrategy::CurrentToBest1 => {
                // v = x_i + F * (x_best - x_i) + F * (x_r1 - x_r2)
                let x_i = self.population.individuals[target_idx].position();
                let x_best = self.population.individuals[best_idx].position();
                let x_r1 = self.population.individuals[indices[0]].position();
                let x_r2 = self.population.individuals[indices[1]].position();
                x_i + f * (x_best - x_i) + f * (x_r1 - x_r2)
            }
            DeStrategy::Rand2 => {
                // v = x_r1 + F * (x_r2 - x_r3) + F * (x_r4 - x_r5)
                let x_r1 = self.population.individuals[indices[0]].position();
                let x_r2 = self.population.individuals[indices[1]].position();
                let x_r3 = self.population.individuals[indices[2]].position();
                let x_r4 = self.population.individuals[indices[3]].position();
                let x_r5 = self.population.individuals[indices[4]].position();
                x_r1 + f * (x_r2 - x_r3) + f * (x_r4 - x_r5)
            }
            DeStrategy::Best2 => {
                // v = x_best + F * (x_r1 - x_r2) + F * (x_r3 - x_r4)
                let x_best = self.population.individuals[best_idx].position();
                let x_r1 = self.population.individuals[indices[0]].position();
                let x_r2 = self.population.individuals[indices[1]].position();
                let x_r3 = self.population.individuals[indices[2]].position();
                let x_r4 = self.population.individuals[indices[3]].position();
                x_best + f * (x_r1 - x_r2) + f * (x_r3 - x_r4)
            }
            DeStrategy::RandToBest1 => {
                // v = x_r1 + F * (x_best - x_r1) + F * (x_r2 - x_r3)
                let x_r1 = self.population.individuals[indices[0]].position();
                let x_best = self.population.individuals[best_idx].position();
                let x_r2 = self.population.individuals[indices[1]].position();
                let x_r3 = self.population.individuals[indices[2]].position();
                x_r1 + f * (x_best - x_r1) + f * (x_r2 - x_r3)
            }
        };
        
        Ok(mutant)
    }
    
    /// Perform crossover between target and mutant
    fn crossover(
        &self,
        target: &Position,
        mutant: &Position,
        cr: f64,
        rng: &mut ThreadRng,
    ) -> Position {
        let dimensions = target.len();
        let mut trial = target.clone();
        
        match self.parameters.crossover_strategy {
            CrossoverStrategy::Binomial => {
                // Ensure at least one dimension is taken from mutant
                let j_rand = rng.gen_range(0..dimensions);
                
                for j in 0..dimensions {
                    if rng.gen::<f64>() < cr || j == j_rand {
                        trial[j] = mutant[j];
                    }
                }
            }
            CrossoverStrategy::Exponential => {
                let j_start = rng.gen_range(0..dimensions);
                let mut j = j_start;
                let mut count = 0;
                
                loop {
                    trial[j] = mutant[j];
                    count += 1;
                    j = (j + 1) % dimensions;
                    
                    if count >= dimensions || rng.gen::<f64>() >= cr {
                        break;
                    }
                }
            }
        }
        
        trial
    }
    
    /// Apply bounds handling
    fn handle_bounds(&self, position: &mut Position) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::optimization("Problem not set"))?;
        
        match self.parameters.bounds_handling {
            BoundsHandling::Clip => {
                for i in 0..position.len() {
                    position[i] = position[i].clamp(
                        problem.lower_bounds[i],
                        problem.upper_bounds[i],
                    );
                }
            }
            BoundsHandling::Reflect => {
                for i in 0..position.len() {
                    let lower = problem.lower_bounds[i];
                    let upper = problem.upper_bounds[i];
                    
                    if position[i] < lower {
                        position[i] = 2.0 * lower - position[i];
                    } else if position[i] > upper {
                        position[i] = 2.0 * upper - position[i];
                    }
                    
                    // Clip if still out of bounds
                    position[i] = position[i].clamp(lower, upper);
                }
            }
            BoundsHandling::Wrap => {
                for i in 0..position.len() {
                    let lower = problem.lower_bounds[i];
                    let upper = problem.upper_bounds[i];
                    let range = upper - lower;
                    
                    if position[i] < lower {
                        position[i] = upper - (lower - position[i]) % range;
                    } else if position[i] > upper {
                        position[i] = lower + (position[i] - upper) % range;
                    }
                }
            }
            BoundsHandling::Reinitialize => {
                let mut rng = thread_rng();
                for i in 0..position.len() {
                    if position[i] < problem.lower_bounds[i] || position[i] > problem.upper_bounds[i] {
                        position[i] = rng.gen_range(
                            problem.lower_bounds[i]..=problem.upper_bounds[i]
                        );
                    }
                }
            }
            BoundsHandling::Midpoint => {
                for i in 0..position.len() {
                    let lower = problem.lower_bounds[i];
                    let upper = problem.upper_bounds[i];
                    
                    if position[i] < lower {
                        position[i] = (position[i] + lower) / 2.0;
                    } else if position[i] > upper {
                        position[i] = (position[i] + upper) / 2.0;
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Generate opposition-based population
    fn generate_opposition_based_population(&self) -> Result<Vec<DeIndividual>, SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::optimization("Problem not set"))?;
        
        let mut opposition_pop = Vec::new();
        
        for individual in self.population.iter() {
            let mut opposition_position = Position::zeros(individual.position().len());
            
            for i in 0..opposition_position.len() {
                opposition_position[i] = problem.lower_bounds[i] + problem.upper_bounds[i] - individual.position()[i];
            }
            
            let opposition_individual = DeIndividual::new(opposition_position);
            opposition_pop.push(opposition_individual);
        }
        
        Ok(opposition_pop)
    }
    
    /// Update success history for adaptive DE
    fn update_success_history(&mut self, success: bool, f: f64, cr: f64) {
        self.success_history.push(success);
        self.f_history.push(f);
        self.cr_history.push(cr);
        
        // Keep only recent history
        let max_history = 100;
        if self.success_history.len() > max_history {
            self.success_history.remove(0);
            self.f_history.remove(0);
            self.cr_history.remove(0);
        }
    }
    
    /// Calculate adaptive parameters based on success history
    fn calculate_adaptive_parameters(&self) -> (f64, f64) {
        if self.success_history.is_empty() {
            return (self.parameters.differential_weight, self.parameters.crossover_probability);
        }
        
        let mut successful_f = Vec::new();
        let mut successful_cr = Vec::new();
        
        for (i, &success) in self.success_history.iter().enumerate() {
            if success {
                successful_f.push(self.f_history[i]);
                successful_cr.push(self.cr_history[i]);
            }
        }
        
        let adaptive_f = if !successful_f.is_empty() {
            successful_f.iter().sum::<f64>() / successful_f.len() as f64
        } else {
            self.parameters.differential_weight
        };
        
        let adaptive_cr = if !successful_cr.is_empty() {
            successful_cr.iter().sum::<f64>() / successful_cr.len() as f64
        } else {
            self.parameters.crossover_probability
        };
        
        (adaptive_f, adaptive_cr)
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        self.population.diversity().unwrap_or(0.0)
    }
    
    /// Check convergence criteria
    fn has_converged(&self) -> bool {
        let diversity = self.calculate_diversity();
        diversity < 1e-6 || self.iteration > 2000
    }
}

#[async_trait]
impl SwarmAlgorithm for DifferentialEvolution {
    type Individual = DeIndividual;
    type Fitness = f64;
    type Parameters = DeParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.parameters.validate()?;
        
        // Store problem
        self.problem = Some(Arc::new(problem));
        let problem_ref = self.problem.as_ref().unwrap();
        
        // Initialize population
        self.population = Population::with_capacity(self.parameters.population_size);
        
        for _ in 0..self.parameters.population_size {
            let individual = DeIndividual::random(
                problem_ref.dimensions,
                problem_ref.lower_bounds.min(),
                problem_ref.upper_bounds.max(),
            );
            self.population.add(individual);
        }
        
        // Opposition-based initialization
        if self.parameters.opposition_based {
            let opposition_pop = self.generate_opposition_based_population()?;
            
            // Evaluate both populations and select best
            let all_individuals = self.population.individuals.iter()
                .chain(opposition_pop.iter())
                .cloned()
                .collect::<Vec<_>>();
            
            // Keep best half
            let mut evaluated: Vec<_> = all_individuals.into_iter()
                .map(|mut ind| {
                    let fitness = problem_ref.evaluate(ind.position());
                    ind.set_fitness(fitness);
                    ind
                })
                .collect();
            
            evaluated.sort_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap());
            
            self.population.individuals = evaluated.into_iter()
                .take(self.parameters.population_size)
                .collect();
        } else {
            // Standard evaluation
            for individual in self.population.iter_mut() {
                let fitness = problem_ref.evaluate(individual.position());
                individual.set_fitness(fitness);
            }
        }
        
        // Find best individual
        if let Some(best) = self.population.best() {
            self.best_fitness = *best.fitness();
            self.best_individual = Some(Arc::new(best.clone()));
        }
        
        // Initialize archive if enabled
        if self.parameters.use_archive {
            let archive_size = (self.parameters.population_size as f64 * self.parameters.archive_size_factor) as usize;
            self.archive = Vec::with_capacity(archive_size);
        }
        
        // Initialize metrics
        self.metrics = AlgorithmMetrics {
            iteration: 0,
            best_fitness: Some(self.best_fitness),
            average_fitness: self.population.average_fitness(),
            diversity: Some(self.calculate_diversity()),
            convergence_rate: None,
            evaluations: self.parameters.population_size,
            time_per_iteration: None,
            memory_usage: None,
        };
        
        tracing::info!(
            "DE initialized with {} individuals, strategy: {:?}",
            self.parameters.population_size,
            self.parameters.strategy
        );
        
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        self.iteration += 1;
        let mut rng = thread_rng();
        
        // Get adaptive parameters if enabled
        let (adaptive_f, adaptive_cr) = if self.parameters.self_adaptive {
            self.calculate_adaptive_parameters()
        } else {
            (self.parameters.differential_weight, self.parameters.crossover_probability)
        };
        
        // Process each individual
        for i in 0..self.population.size() {
            // Generate mutant vector
            let mutant = self.generate_mutant(i, &mut rng)?;
            
            // Get individual-specific parameters if self-adaptive
            let (f, cr) = if self.parameters.self_adaptive {
                let individual = &self.population.individuals[i];
                (individual.self_f(), individual.self_cr())
            } else {
                (adaptive_f, adaptive_cr)
            };
            
            // Perform crossover
            let target = self.population.individuals[i].position();
            let mut trial = self.crossover(target, &mutant, cr, &mut rng);
            
            // Handle bounds
            self.handle_bounds(&mut trial)?;
            
            // Set trial vector
            self.population.individuals[i].set_trial_vector(trial);
        }
        
        // Evaluate trial vectors
        let trial_positions: Vec<Position> = self.population.iter()
            .filter_map(|ind| ind.trial_vector())
            .cloned()
            .collect();
        
        if let Some(ref problem) = self.problem {
            let trial_fitnesses = problem.evaluate_parallel(&trial_positions);
            
            for (individual, &trial_fitness) in self.population.iter_mut().zip(trial_fitnesses.iter()) {
                individual.set_trial_fitness(trial_fitness);
            }
            
            self.metrics.evaluations += trial_fitnesses.len();
        }
        
        // Selection phase
        let mut improved_count = 0;
        for individual in self.population.iter_mut() {
            let improved = individual.accept_trial_if_better();
            if improved {
                improved_count += 1;
                
                // Update global best
                if *individual.fitness() < self.best_fitness {
                    self.best_fitness = *individual.fitness();
                    self.best_individual = Some(Arc::new(individual.clone()));
                }
            }
            
            // Update self-adaptive parameters
            if self.parameters.self_adaptive {
                individual.update_adaptive_parameters(0.1, 0.1);
                
                // Record success for adaptation
                self.update_success_history(
                    improved,
                    individual.self_f(),
                    individual.self_cr(),
                );
            }
        }
        
        // Archive management
        if self.parameters.use_archive && improved_count > 0 {
            // Add some improved solutions to archive
            let archive_capacity = (self.parameters.population_size as f64 * self.parameters.archive_size_factor) as usize;
            
            if self.archive.len() < archive_capacity {
                for individual in self.population.iter() {
                    if self.archive.len() < archive_capacity {
                        self.archive.push(individual.clone());
                    }
                }
            }
        }
        
        // Update metrics
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.average_fitness = self.population.average_fitness();
        self.metrics.diversity = Some(self.calculate_diversity());
        self.metrics.time_per_iteration = Some(start_time.elapsed().as_micros() as u64);
        
        Ok(())
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.best_individual.as_ref().map(|arc| arc.as_ref())
    }
    
    fn get_population(&self) -> &Population<Self::Individual> {
        &self.population
    }
    
    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        &mut self.population
    }
    
    fn has_converged(&self) -> bool {
        self.has_converged()
    }
    
    fn name(&self) -> &'static str {
        "DifferentialEvolution"
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }
    
    fn update_parameters(&mut self, params: Self::Parameters) {
        self.parameters = params;
    }
    
    fn metrics(&self) -> AlgorithmMetrics {
        self.metrics.clone()
    }
    
    async fn reset(&mut self) -> Result<(), SwarmError> {
        self.population = Population::new();
        self.archive.clear();
        self.best_individual = None;
        self.best_fitness = f64::INFINITY;
        self.iteration = 0;
        self.metrics = AlgorithmMetrics::default();
        self.success_history.clear();
        self.f_history.clear();
        self.cr_history.clear();
        Ok(())
    }
    
    fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<
        Individual = Self::Individual,
        Fitness = Self::Fitness,
        Parameters = Self::Parameters
    >> {
        Box::new(self.clone())
    }
}

impl AdaptiveAlgorithm for DifferentialEvolution {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        if !self.parameters.self_adaptive {
            return;
        }
        
        // Adapt based on success rate and diversity
        if let Some(diversity) = performance_metrics.diversity {
            if diversity < 0.1 {
                // Low diversity - increase F for more exploration
                self.parameters.differential_weight = 
                    (self.parameters.differential_weight + 0.1)
                    .min(self.parameters.max_differential_weight);
            } else if diversity > 0.5 {
                // High diversity - decrease F for more exploitation
                self.parameters.differential_weight = 
                    (self.parameters.differential_weight - 0.1)
                    .max(self.parameters.min_differential_weight);
            }
        }
        
        // Adapt crossover probability based on success rate
        let success_rate = if !self.success_history.is_empty() {
            self.success_history.iter().filter(|&&x| x).count() as f64 / self.success_history.len() as f64
        } else {
            0.5
        };
        
        if success_rate < 0.2 {
            self.parameters.crossover_probability = 
                (self.parameters.crossover_probability + 0.1).min(1.0);
        } else if success_rate > 0.8 {
            self.parameters.crossover_probability = 
                (self.parameters.crossover_probability - 0.1).max(0.1);
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::SuccessBased {
            increase_factor: 1.1,
            decrease_factor: 0.9,
        }
    }
}

impl ParallelAlgorithm for DifferentialEvolution {
    async fn parallel_step(&mut self, thread_count: usize) -> Result<(), SwarmError> {
        // DE can benefit from parallel mutation and crossover
        // For now, use the standard step implementation with parallel evaluation
        self.step().await
    }
    
    fn optimal_thread_count(&self) -> usize {
        (self.parameters.population_size / 5).max(1).min(num_cpus::get())
    }
}

/// Builder for DE algorithm
pub struct DeBuilder {
    parameters: DeParameters,
}

impl DeBuilder {
    pub fn new() -> Self {
        Self {
            parameters: DeParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.parameters.population_size = size;
        self
    }
    
    pub fn differential_weight(mut self, weight: f64) -> Self {
        self.parameters.differential_weight = weight;
        self
    }
    
    pub fn crossover_probability(mut self, prob: f64) -> Self {
        self.parameters.crossover_probability = prob;
        self
    }
    
    pub fn strategy(mut self, strategy: DeStrategy) -> Self {
        self.parameters.strategy = strategy;
        self
    }
    
    pub fn self_adaptive(mut self, adaptive: bool) -> Self {
        self.parameters.self_adaptive = adaptive;
        self
    }
    
    pub fn opposition_based(mut self, opposition: bool) -> Self {
        self.parameters.opposition_based = opposition;
        self
    }
    
    pub fn build(self) -> Result<DifferentialEvolution, SwarmError> {
        self.parameters.validate()?;
        Ok(DifferentialEvolution::with_parameters(self.parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_de_initialization() {
        let mut de = DifferentialEvolution::new();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-10.0, 10.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert!(de.initialize(problem).await.is_ok());
        assert_eq!(de.population.size(), 30);
        assert!(de.best_individual.is_some());
    }
    
    #[tokio::test]
    async fn test_de_optimization() {
        let mut de = DifferentialEvolution::builder()
            .population_size(20)
            .strategy(DeStrategy::Best1)
            .build()
            .unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let result = de.optimize(50).await.unwrap();
        
        // Should find good solution close to [0, 0]
        assert!(result.best_fitness < 1.0);
        assert!(result.iterations <= 50);
        assert_eq!(result.algorithm_name, "DifferentialEvolution");
    }
    
    #[test]
    fn test_de_parameters() {
        let params = DeParameters::builder()
            .population_size(100)
            .differential_weight(0.7)
            .crossover_probability(0.8)
            .strategy(DeStrategy::CurrentToBest1)
            .self_adaptive(true)
            .opposition_based(true)
            .build()
            .unwrap();
        
        assert_eq!(params.population_size, 100);
        assert_relative_eq!(params.differential_weight, 0.7);
        assert_relative_eq!(params.crossover_probability, 0.8);
        assert!(params.self_adaptive);
        assert!(params.opposition_based);
        assert!(matches!(params.strategy, DeStrategy::CurrentToBest1));
    }
    
    #[test]
    fn test_de_individual() {
        let position = Position::from_vec(vec![1.0, 2.0, 3.0]);
        let mut individual = DeIndividual::new(position.clone());
        
        assert_eq!(individual.position(), &position);
        assert_eq!(*individual.fitness(), f64::INFINITY);
        assert_eq!(individual.trial_fitness(), f64::INFINITY);
        assert_eq!(individual.success_rate(), 0.0);
        
        // Test trial acceptance
        let trial = Position::from_vec(vec![0.5, 1.0, 1.5]);
        individual.set_trial_vector(trial.clone());
        individual.set_trial_fitness(5.0);
        individual.set_fitness(10.0);
        
        assert!(individual.accept_trial_if_better());
        assert_eq!(individual.position(), &trial);
        assert_eq!(*individual.fitness(), 5.0);
        assert!(individual.success_rate() > 0.0);
    }
}