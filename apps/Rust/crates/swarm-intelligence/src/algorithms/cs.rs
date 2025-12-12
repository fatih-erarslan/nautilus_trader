//! Cuckoo Search (CS) optimization implementation
//!
//! CS is inspired by the brood parasitism of cuckoo species. It uses
//! Lévy flights for global exploration and local search for exploitation,
//! making it effective for continuous optimization problems.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal};

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Population, Individual, BasicIndividual, Position, AlgorithmMetrics,
    AdaptiveAlgorithm, ParallelAlgorithm, AdaptationStrategy
};
use crate::validate_parameter;

/// CS algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CsVariant {
    /// Standard Cuckoo Search
    Standard,
    /// Improved Cuckoo Search (ICS)
    Improved,
    /// Modified Cuckoo Search (MCS)
    Modified,
    /// Enhanced Cuckoo Search (ECS)
    Enhanced,
    /// Hybrid Cuckoo Search (HCS)
    Hybrid,
    /// Quantum-inspired Cuckoo Search (QCS)
    Quantum,
}

/// Lévy flight distribution types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LevyDistribution {
    /// Standard Lévy distribution
    Standard,
    /// Mantegna algorithm
    Mantegna,
    /// Chambers-Mallows-Stuck method
    ChambersMallowsStuck,
    /// Symmetric stable distribution
    SymmetricStable,
}

/// CS algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CsParameters {
    /// Population size (number of nests)
    pub population_size: usize,
    
    /// Discovery probability (Pa)
    pub discovery_probability: f64,
    
    /// Lévy flight step size control parameter
    pub levy_lambda: f64,
    
    /// Lévy flight distribution type
    pub levy_distribution: LevyDistribution,
    
    /// CS variant to use
    pub variant: CsVariant,
    
    /// Enable adaptive discovery probability
    pub adaptive_pa: bool,
    
    /// Minimum discovery probability
    pub min_discovery_probability: f64,
    
    /// Maximum discovery probability
    pub max_discovery_probability: f64,
    
    /// Elite preservation ratio
    pub elite_ratio: f64,
    
    /// Step size scaling factor
    pub step_size_scaling: f64,
    
    /// Enable local search enhancement
    pub local_search: bool,
    
    /// Local search probability
    pub local_search_probability: f64,
    
    /// Local search radius
    pub local_search_radius: f64,
    
    /// Enable opposition-based learning
    pub opposition_based: bool,
    
    /// Opposition probability
    pub opposition_probability: f64,
    
    /// Enable chaotic maps
    pub chaotic_maps: bool,
    
    /// Chaotic map type parameter
    pub chaotic_parameter: f64,
    
    /// Quantum rotation angle (for quantum variant)
    pub quantum_rotation_angle: f64,
    
    /// Enable dynamic switching between exploration/exploitation
    pub dynamic_switching: bool,
    
    /// Switching threshold
    pub switching_threshold: f64,
}

impl Default for CsParameters {
    fn default() -> Self {
        Self {
            population_size: 25,
            discovery_probability: 0.25,
            levy_lambda: 1.5,
            levy_distribution: LevyDistribution::Mantegna,
            variant: CsVariant::Standard,
            adaptive_pa: false,
            min_discovery_probability: 0.1,
            max_discovery_probability: 0.5,
            elite_ratio: 0.1,
            step_size_scaling: 1.0,
            local_search: false,
            local_search_probability: 0.1,
            local_search_radius: 0.1,
            opposition_based: false,
            opposition_probability: 0.1,
            chaotic_maps: false,
            chaotic_parameter: 4.0,
            quantum_rotation_angle: 0.05,
            dynamic_switching: false,
            switching_threshold: 0.5,
        }
    }
}

impl CsParameters {
    /// Validate parameters
    pub fn validate(&self) -> Result<(), SwarmError> {
        validate_parameter!(self.population_size, "population_size", 4, 1000);
        validate_parameter!(self.discovery_probability, "discovery_probability", 0.0, 1.0);
        validate_parameter!(self.levy_lambda, "levy_lambda", 0.3, 2.0);
        validate_parameter!(self.elite_ratio, "elite_ratio", 0.0, 0.5);
        validate_parameter!(self.step_size_scaling, "step_size_scaling", 0.1, 10.0);
        validate_parameter!(self.local_search_probability, "local_search_probability", 0.0, 1.0);
        validate_parameter!(self.local_search_radius, "local_search_radius", 0.01, 1.0);
        
        if self.min_discovery_probability >= self.max_discovery_probability {
            return Err(SwarmError::parameter("min_discovery_probability must be < max_discovery_probability"));
        }
        
        Ok(())
    }
    
    /// Create builder for CS parameters
    pub fn builder() -> CsParametersBuilder {
        CsParametersBuilder::new()
    }
}

/// Builder for CS parameters
pub struct CsParametersBuilder {
    params: CsParameters,
}

impl CsParametersBuilder {
    pub fn new() -> Self {
        Self {
            params: CsParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.params.population_size = size;
        self
    }
    
    pub fn discovery_probability(mut self, pa: f64) -> Self {
        self.params.discovery_probability = pa;
        self
    }
    
    pub fn levy_lambda(mut self, lambda: f64) -> Self {
        self.params.levy_lambda = lambda;
        self
    }
    
    pub fn variant(mut self, variant: CsVariant) -> Self {
        self.params.variant = variant;
        self
    }
    
    pub fn adaptive_pa(mut self, adaptive: bool) -> Self {
        self.params.adaptive_pa = adaptive;
        self
    }
    
    pub fn local_search(mut self, enabled: bool) -> Self {
        self.params.local_search = enabled;
        self
    }
    
    pub fn opposition_based(mut self, enabled: bool) -> Self {
        self.params.opposition_based = enabled;
        self
    }
    
    pub fn build(self) -> Result<CsParameters, SwarmError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

/// Cuckoo nest representation
#[derive(Debug, Clone)]
pub struct Nest {
    /// Current position
    position: Position,
    
    /// Fitness value
    fitness: f64,
    
    /// Number of times discovered and replaced
    discovery_count: usize,
    
    /// Local search history
    local_search_history: Vec<Position>,
    
    /// Quantum state (for quantum variant)
    quantum_state: Option<Vec<f64>>,
    
    /// Chaotic sequence value
    chaotic_value: f64,
    
    /// Age of the nest
    age: usize,
}

impl Nest {
    /// Create a new nest at random position
    pub fn new(dimensions: usize, bounds: (f64, f64)) -> Self {
        let mut rng = thread_rng();
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(bounds.0..=bounds.1)
        });
        
        Self {
            position,
            fitness: f64::INFINITY,
            discovery_count: 0,
            local_search_history: Vec::new(),
            quantum_state: None,
            chaotic_value: rng.gen(),
            age: 0,
        }
    }
    
    /// Initialize quantum state
    pub fn initialize_quantum_state(&mut self, dimensions: usize) {
        let mut rng = thread_rng();
        self.quantum_state = Some((0..dimensions).map(|_| rng.gen_range(0.0..2.0 * std::f64::consts::PI)).collect());
    }
    
    /// Generate Lévy flight step
    pub fn levy_flight_step(&self, lambda: f64, distribution: LevyDistribution, step_size: f64) -> Position {
        match distribution {
            LevyDistribution::Mantegna => self.levy_mantegna(lambda, step_size),
            LevyDistribution::Standard => self.levy_standard(lambda, step_size),
            LevyDistribution::ChambersMallowsStuck => self.levy_cms(lambda, step_size),
            LevyDistribution::SymmetricStable => self.levy_symmetric_stable(lambda, step_size),
        }
    }
    
    /// Mantegna algorithm for Lévy flight
    fn levy_mantegna(&self, lambda: f64, step_size: f64) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        let sigma_u = (
            gamma_function((1.0 + lambda) / 2.0) * 
            (lambda * std::f64::consts::PI / 2.0).sin() /
            (gamma_function((1.0 + lambda) / 2.0) * 2.0_f64.powf((lambda - 1.0) / 2.0))
        ).powf(1.0 / lambda);
        
        Position::from_fn(dimensions, |_, _| {
            let u: f64 = Normal::new(0.0, sigma_u.powi(2)).unwrap().sample(&mut rng);
            let v: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut rng);
            step_size * u / v.abs().powf(1.0 / lambda)
        })
    }
    
    /// Standard Lévy flight
    fn levy_standard(&self, lambda: f64, step_size: f64) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        Position::from_fn(dimensions, |_, _| {
            let u: f64 = rng.sample(StandardNormal);
            step_size * u.signum() * u.abs().powf(1.0 / lambda)
        })
    }
    
    /// Chambers-Mallows-Stuck method
    fn levy_cms(&self, lambda: f64, step_size: f64) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        Position::from_fn(dimensions, |_, _| {
            let theta = rng.gen_range(-std::f64::consts::PI/2.0..std::f64::consts::PI/2.0);
            let w = rng.gen_range(0.0..1.0);
            
            let s = ((1.0 + lambda) * theta).sin() / 
                   ((theta).cos().powf(1.0 / (1.0 + lambda))) *
                   (((1.0 - lambda) * theta).cos() / w.ln()).powf((1.0 - lambda) / (1.0 + lambda));
            
            step_size * s
        })
    }
    
    /// Symmetric stable distribution
    fn levy_symmetric_stable(&self, lambda: f64, step_size: f64) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        Position::from_fn(dimensions, |_, _| {
            let u = rng.gen_range(-std::f64::consts::PI/2.0..std::f64::consts::PI/2.0);
            let w = -w.ln();
            
            let x = if (lambda - 1.0).abs() < 1e-7 {
                (2.0 / std::f64::consts::PI) * 
                ((std::f64::consts::PI / 2.0 * lambda) * u).tan() * 
                w.ln() - (1.0 / lambda) * 
                w.ln() * (std::f64::consts::PI / 2.0 * lambda * u).cos()
            } else {
                (u + std::f64::consts::PI / 2.0).sin() * 
                (lambda * u).cos().powf(1.0 / lambda) /
                u.cos() * 
                (w * u.cos() / (u + std::f64::consts::PI / 2.0).sin()).powf((1.0 - lambda) / lambda)
            };
            
            step_size * x
        })
    }
    
    /// Update chaotic value using logistic map
    pub fn update_chaotic_value(&mut self, parameter: f64) {
        self.chaotic_value = parameter * self.chaotic_value * (1.0 - self.chaotic_value);
    }
    
    /// Apply quantum rotation (for quantum variant)
    pub fn apply_quantum_rotation(&mut self, angle: f64, bounds: (f64, f64)) {
        if let Some(ref mut quantum_state) = self.quantum_state {
            let mut rng = thread_rng();
            
            for (i, state) in quantum_state.iter_mut().enumerate() {
                *state += angle;
                
                // Update position based on quantum state
                let new_value = bounds.0 + (bounds.1 - bounds.0) * 
                               (0.5 + 0.5 * state.cos());
                
                // Add quantum interference
                let interference = 0.1 * (state * 2.0).sin() * rng.gen_range(-1.0..1.0);
                self.position[i] = (new_value + interference).clamp(bounds.0, bounds.1);
            }
        }
    }
    
    /// Perform local search around current position
    pub fn local_search(&mut self, radius: f64, bounds: (f64, f64)) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        let search_position = Position::from_fn(dimensions, |i, _| {
            let perturbation = rng.gen_range(-radius..radius);
            (self.position[i] + perturbation).clamp(bounds.0, bounds.1)
        });
        
        self.local_search_history.push(search_position.clone());
        if self.local_search_history.len() > 10 {
            self.local_search_history.remove(0);
        }
        
        search_position
    }
    
    /// Get discovery count
    pub fn discovery_count(&self) -> usize {
        self.discovery_count
    }
    
    /// Increment discovery count
    pub fn discovered(&mut self) {
        self.discovery_count += 1;
        self.age += 1;
    }
    
    /// Age the nest
    pub fn age_nest(&mut self) {
        self.age += 1;
    }
    
    /// Get nest age
    pub fn age(&self) -> usize {
        self.age
    }
}

impl Individual for Nest {
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

/// Cuckoo Search algorithm
#[derive(Debug, Clone)]
pub struct CuckooSearch {
    /// Algorithm parameters
    parameters: CsParameters,
    
    /// Population of nests
    population: Population<Nest>,
    
    /// Best solution found
    best_solution: Option<Arc<Nest>>,
    
    /// Best fitness
    best_fitness: f64,
    
    /// Current iteration
    iteration: usize,
    
    /// Optimization problem
    problem: Option<Arc<OptimizationProblem>>,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
    
    /// Current discovery probability (adaptive)
    current_discovery_probability: f64,
    
    /// Elite solutions
    elite_solutions: Vec<Nest>,
    
    /// Diversity measure
    diversity: f64,
}

impl CuckooSearch {
    /// Create a new CS algorithm with default parameters
    pub fn new() -> Self {
        Self::with_parameters(CsParameters::default())
    }
    
    /// Create CS with specific parameters
    pub fn with_parameters(parameters: CsParameters) -> Self {
        Self {
            current_discovery_probability: parameters.discovery_probability,
            parameters,
            population: Population::new(),
            best_solution: None,
            best_fitness: f64::INFINITY,
            iteration: 0,
            problem: None,
            metrics: AlgorithmMetrics::default(),
            elite_solutions: Vec::new(),
            diversity: 0.0,
        }
    }
    
    /// Builder pattern for CS construction
    pub fn builder() -> CsBuilder {
        CsBuilder::new()
    }
    
    /// Generate new cuckoo using Lévy flight
    async fn generate_new_cuckoo(&self, nest_index: usize) -> Result<Nest, SwarmError> {
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        
        let current_nest = &self.population[nest_index];
        let mut new_nest = current_nest.clone();
        
        // Generate Lévy flight step
        let step_size = self.parameters.step_size_scaling * 
                       self.calculate_adaptive_step_size();
        
        let levy_step = current_nest.levy_flight_step(
            self.parameters.levy_lambda,
            self.parameters.levy_distribution,
            step_size,
        );
        
        // Apply variant-specific modifications
        let new_position = match self.parameters.variant {
            CsVariant::Standard => &current_nest.position + levy_step,
            CsVariant::Improved => self.improved_cuckoo_generation(current_nest, levy_step, bounds)?,
            CsVariant::Modified => self.modified_cuckoo_generation(current_nest, levy_step, bounds)?,
            CsVariant::Enhanced => self.enhanced_cuckoo_generation(current_nest, levy_step, bounds)?,
            CsVariant::Hybrid => self.hybrid_cuckoo_generation(current_nest, levy_step, bounds)?,
            CsVariant::Quantum => self.quantum_cuckoo_generation(current_nest, levy_step, bounds)?,
        };
        
        // Bound the position
        let bounded_position = Position::from_fn(new_position.len(), |i, _| {
            new_position[i].clamp(bounds.0, bounds.1)
        });
        
        new_nest.update_position(bounded_position);
        new_nest.age_nest();
        
        Ok(new_nest)
    }
    
    /// Calculate adaptive step size
    fn calculate_adaptive_step_size(&self) -> f64 {
        if self.parameters.dynamic_switching {
            let progress = self.iteration as f64 / 1000.0; // Assume max 1000 iterations
            let exploration_factor = (-2.0 * progress).exp();
            let exploitation_factor = 1.0 - exploration_factor;
            
            if self.diversity > self.parameters.switching_threshold {
                exploration_factor // Focus on exploration
            } else {
                exploitation_factor // Focus on exploitation
            }
        } else {
            1.0
        }
    }
    
    /// Improved Cuckoo Search generation
    fn improved_cuckoo_generation(&self, current_nest: &Nest, levy_step: Position, bounds: (f64, f64)) -> Result<Position, SwarmError> {
        // Use information from best solution
        if let Some(ref best_nest) = self.best_solution {
            let direction_to_best = best_nest.position() - current_nest.position();
            let new_position = current_nest.position() + 0.5 * levy_step + 0.3 * direction_to_best;
            Ok(new_position)
        } else {
            Ok(current_nest.position() + levy_step)
        }
    }
    
    /// Modified Cuckoo Search generation
    fn modified_cuckoo_generation(&self, current_nest: &Nest, levy_step: Position, bounds: (f64, f64)) -> Result<Position, SwarmError> {
        let mut rng = thread_rng();
        
        // Use chaotic sequences for better exploration
        let chaotic_factor = current_nest.chaotic_value;
        let scaled_levy = levy_step * chaotic_factor;
        
        let new_position = current_nest.position() + scaled_levy;
        Ok(new_position)
    }
    
    /// Enhanced Cuckoo Search generation
    fn enhanced_cuckoo_generation(&self, current_nest: &Nest, levy_step: Position, bounds: (f64, f64)) -> Result<Position, SwarmError> {
        let mut rng = thread_rng();
        
        // Combine multiple random nests
        let mut combined_influence = Position::zeros(current_nest.position().len());
        let num_influences = (self.population.size() / 4).max(2);
        
        for _ in 0..num_influences {
            let random_index = rng.gen_range(0..self.population.size());
            let random_nest = &self.population[random_index];
            combined_influence = combined_influence + (random_nest.position() - current_nest.position()) / num_influences as f64;
        }
        
        let new_position = current_nest.position() + levy_step + 0.2 * combined_influence;
        Ok(new_position)
    }
    
    /// Hybrid Cuckoo Search generation
    fn hybrid_cuckoo_generation(&self, current_nest: &Nest, levy_step: Position, bounds: (f64, f64)) -> Result<Position, SwarmError> {
        let mut rng = thread_rng();
        
        if rng.gen::<f64>() < 0.5 {
            // Use Lévy flight
            Ok(current_nest.position() + levy_step)
        } else {
            // Use random walk
            let random_step = Position::from_fn(current_nest.position().len(), |_, _| {
                rng.gen_range(-0.1..0.1)
            });
            Ok(current_nest.position() + random_step)
        }
    }
    
    /// Quantum-inspired Cuckoo Search generation
    fn quantum_cuckoo_generation(&self, current_nest: &Nest, levy_step: Position, bounds: (f64, f64)) -> Result<Position, SwarmError> {
        let mut new_nest = current_nest.clone();
        
        if new_nest.quantum_state.is_none() {
            new_nest.initialize_quantum_state(current_nest.position().len());
        }
        
        new_nest.apply_quantum_rotation(self.parameters.quantum_rotation_angle, bounds);
        
        // Combine quantum position with Lévy flight
        let quantum_position = new_nest.position().clone();
        let combined_position = 0.7 * quantum_position + 0.3 * (current_nest.position() + levy_step);
        
        Ok(combined_position)
    }
    
    /// Abandon worse nests and replace with new random solutions
    async fn abandon_nests(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        
        let mut rng = thread_rng();
        let num_to_abandon = (self.population.size() as f64 * self.current_discovery_probability).round() as usize;
        
        // Sort nests by fitness to identify worst ones
        let mut indexed_nests: Vec<(usize, f64)> = self.population.iter()
            .enumerate()
            .map(|(i, nest)| (i, *nest.fitness()))
            .collect();
        indexed_nests.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Replace worst nests
        for i in 0..num_to_abandon {
            if i < indexed_nests.len() {
                let nest_index = indexed_nests[i].0;
                let mut new_nest = Nest::new(problem.dimensions, bounds);
                
                // Initialize quantum state if using quantum variant
                if matches!(self.parameters.variant, CsVariant::Quantum) {
                    new_nest.initialize_quantum_state(problem.dimensions);
                }
                
                // Apply opposition-based learning
                if self.parameters.opposition_based && rng.gen::<f64>() < self.parameters.opposition_probability {
                    let opposition_position = Position::from_fn(problem.dimensions, |i, _| {
                        bounds.0 + bounds.1 - new_nest.position()[i]
                    });
                    new_nest.update_position(opposition_position);
                }
                
                self.population[nest_index] = new_nest;
                self.population[nest_index].discovered();
            }
        }
        
        Ok(())
    }
    
    /// Perform local search on elite solutions
    async fn elite_local_search(&mut self) -> Result<(), SwarmError> {
        if !self.parameters.local_search {
            return Ok();
        }
        
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        let mut rng = thread_rng();
        
        let num_elite = (self.population.size() as f64 * self.parameters.elite_ratio).round() as usize;
        
        // Sort to get elite solutions
        let mut indexed_nests: Vec<(usize, f64)> = self.population.iter()
            .enumerate()
            .map(|(i, nest)| (i, *nest.fitness()))
            .collect();
        indexed_nests.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        for i in 0..num_elite.min(indexed_nests.len()) {
            if rng.gen::<f64>() < self.parameters.local_search_probability {
                let nest_index = indexed_nests[i].0;
                let local_position = self.population[nest_index].local_search(
                    self.parameters.local_search_radius,
                    bounds
                );
                
                // Evaluate local search position
                let local_fitness = problem.evaluate(&local_position);
                
                if local_fitness < *self.population[nest_index].fitness() {
                    self.population[nest_index].update_position(local_position);
                    self.population[nest_index].set_fitness(local_fitness);
                    
                    // Update global best if needed
                    if local_fitness < self.best_fitness {
                        self.best_fitness = local_fitness;
                        self.best_solution = Some(Arc::new(self.population[nest_index].clone()));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Update adaptive parameters
    fn update_adaptive_parameters(&mut self) {
        if self.parameters.adaptive_pa {
            // Adapt discovery probability based on convergence
            let convergence_rate = if self.iteration > 10 {
                let recent_improvements = self.metrics.convergence_rate.unwrap_or(0.0);
                if recent_improvements < 0.01 {
                    // Slow convergence - increase exploration
                    self.current_discovery_probability = 
                        (self.current_discovery_probability + 0.01)
                        .min(self.parameters.max_discovery_probability);
                } else {
                    // Fast convergence - increase exploitation
                    self.current_discovery_probability = 
                        (self.current_discovery_probability - 0.01)
                        .max(self.parameters.min_discovery_probability);
                }
            };
        }
        
        // Update chaotic values for all nests
        if self.parameters.chaotic_maps {
            for nest in self.population.iter_mut() {
                nest.update_chaotic_value(self.parameters.chaotic_parameter);
            }
        }
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&mut self) -> f64 {
        self.diversity = self.population.diversity().unwrap_or(0.0);
        self.diversity
    }
    
    /// Check convergence
    fn has_converged(&self) -> bool {
        self.diversity < 1e-6 || self.iteration > 1000
    }
}

#[async_trait]
impl SwarmAlgorithm for CuckooSearch {
    type Individual = Nest;
    type Fitness = f64;
    type Parameters = CsParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.parameters.validate()?;
        
        // Store problem
        self.problem = Some(Arc::new(problem));
        let problem_ref = self.problem.as_ref().unwrap();
        
        // Initialize population
        self.population = Population::with_capacity(self.parameters.population_size);
        let bounds = (problem_ref.lower_bounds.min(), problem_ref.upper_bounds.max());
        
        for _ in 0..self.parameters.population_size {
            let mut nest = Nest::new(problem_ref.dimensions, bounds);
            
            // Initialize quantum state if using quantum variant
            if matches!(self.parameters.variant, CsVariant::Quantum) {
                nest.initialize_quantum_state(problem_ref.dimensions);
            }
            
            self.population.add(nest);
        }
        
        // Initial evaluation
        let positions: Vec<Position> = self.population.iter().map(|nest| nest.position().clone()).collect();
        let fitnesses = problem_ref.evaluate_parallel(&positions);
        
        for (nest, fitness) in self.population.iter_mut().zip(fitnesses.iter()) {
            nest.set_fitness(*fitness);
            
            if *fitness < self.best_fitness {
                self.best_fitness = *fitness;
                self.best_solution = Some(Arc::new(nest.clone()));
            }
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
            "CS initialized with {} nests, variant: {:?}",
            self.parameters.population_size,
            self.parameters.variant
        );
        
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        self.iteration += 1;
        let problem = self.problem.as_ref().unwrap();
        
        // Generate new cuckoos via Lévy flights
        for i in 0..self.population.size() {
            let new_cuckoo = self.generate_new_cuckoo(i).await?;
            let new_fitness = problem.evaluate(new_cuckoo.position());
            
            // Replace if better
            if new_fitness < *self.population[i].fitness() {
                let mut updated_cuckoo = new_cuckoo;
                updated_cuckoo.set_fitness(new_fitness);
                self.population[i] = updated_cuckoo;
                
                // Update global best
                if new_fitness < self.best_fitness {
                    self.best_fitness = new_fitness;
                    self.best_solution = Some(Arc::new(self.population[i].clone()));
                }
            }
        }
        
        // Abandon worse nests
        self.abandon_nests().await?;
        
        // Re-evaluate abandoned nests
        let positions: Vec<Position> = self.population.iter().map(|nest| nest.position().clone()).collect();
        let fitnesses = problem.evaluate_parallel(&positions);
        
        for (nest, fitness) in self.population.iter_mut().zip(fitnesses.iter()) {
            if *fitness < *nest.fitness() {
                nest.set_fitness(*fitness);
                
                if *fitness < self.best_fitness {
                    self.best_fitness = *fitness;
                    self.best_solution = Some(Arc::new(nest.clone()));
                }
            }
        }
        
        // Perform elite local search
        self.elite_local_search().await?;
        
        // Update adaptive parameters
        self.update_adaptive_parameters();
        
        // Update metrics
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.average_fitness = self.population.average_fitness();
        self.metrics.diversity = Some(self.calculate_diversity());
        self.metrics.time_per_iteration = Some(start_time.elapsed().as_micros() as u64);
        self.metrics.evaluations += self.parameters.population_size * 2; // Cuckoo generation + abandonment
        
        Ok(())
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.best_solution.as_ref().map(|arc| arc.as_ref())
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
        "CuckooSearch"
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }
    
    fn update_parameters(&mut self, params: Self::Parameters) {
        self.parameters = params;
        self.current_discovery_probability = params.discovery_probability;
    }
    
    fn metrics(&self) -> AlgorithmMetrics {
        self.metrics.clone()
    }
    
    async fn reset(&mut self) -> Result<(), SwarmError> {
        self.population = Population::new();
        self.best_solution = None;
        self.best_fitness = f64::INFINITY;
        self.iteration = 0;
        self.metrics = AlgorithmMetrics::default();
        self.current_discovery_probability = self.parameters.discovery_probability;
        self.elite_solutions.clear();
        self.diversity = 0.0;
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

impl AdaptiveAlgorithm for CuckooSearch {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        if let Some(diversity) = performance_metrics.diversity {
            if diversity < 0.1 {
                // Low diversity - increase exploration
                self.current_discovery_probability = 
                    (self.current_discovery_probability + 0.05)
                    .min(self.parameters.max_discovery_probability);
                self.parameters.step_size_scaling = 
                    (self.parameters.step_size_scaling * 1.1).min(2.0);
            } else if diversity > 0.5 {
                // High diversity - increase exploitation
                self.current_discovery_probability = 
                    (self.current_discovery_probability - 0.05)
                    .max(self.parameters.min_discovery_probability);
                self.parameters.step_size_scaling = 
                    (self.parameters.step_size_scaling * 0.9).max(0.1);
            }
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::Feedback { sensitivity: 0.1 }
    }
}

impl ParallelAlgorithm for CuckooSearch {
    async fn parallel_step(&mut self, thread_count: usize) -> Result<(), SwarmError> {
        // CS can be parallelized at the cuckoo generation level
        // For now, use the standard step implementation
        self.step().await
    }
    
    fn optimal_thread_count(&self) -> usize {
        (self.parameters.population_size / 5).max(1).min(num_cpus::get())
    }
}

/// Builder for CS algorithm
pub struct CsBuilder {
    parameters: CsParameters,
}

impl CsBuilder {
    pub fn new() -> Self {
        Self {
            parameters: CsParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.parameters.population_size = size;
        self
    }
    
    pub fn discovery_probability(mut self, pa: f64) -> Self {
        self.parameters.discovery_probability = pa;
        self
    }
    
    pub fn levy_lambda(mut self, lambda: f64) -> Self {
        self.parameters.levy_lambda = lambda;
        self
    }
    
    pub fn variant(mut self, variant: CsVariant) -> Self {
        self.parameters.variant = variant;
        self
    }
    
    pub fn adaptive_pa(mut self, adaptive: bool) -> Self {
        self.parameters.adaptive_pa = adaptive;
        self
    }
    
    pub fn build(self) -> Result<CuckooSearch, SwarmError> {
        self.parameters.validate()?;
        Ok(CuckooSearch::with_parameters(self.parameters))
    }
}

/// Gamma function approximation
fn gamma_function(x: f64) -> f64 {
    // Stirling's approximation for gamma function
    if x < 1.0 {
        std::f64::consts::PI / (x * gamma_function(1.0 - x) * (std::f64::consts::PI * x).sin())
    } else {
        (2.0 * std::f64::consts::PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_cs_initialization() {
        let mut cs = CuckooSearch::new();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-10.0, 10.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert!(cs.initialize(problem).await.is_ok());
        assert_eq!(cs.population.size(), 25);
        assert!(cs.best_solution.is_some());
    }
    
    #[tokio::test]
    async fn test_cs_optimization() {
        let mut cs = CuckooSearch::builder()
            .population_size(20)
            .discovery_probability(0.25)
            .variant(CsVariant::Improved)
            .build()
            .unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let result = cs.optimize(50).await.unwrap();
        
        // Should find reasonable solution
        assert!(result.best_fitness < 25.0);
        assert!(result.iterations <= 50);
        assert_eq!(result.algorithm_name, "CuckooSearch");
    }
    
    #[test]
    fn test_cs_parameters() {
        let params = CsParameters::builder()
            .population_size(30)
            .discovery_probability(0.3)
            .levy_lambda(1.5)
            .variant(CsVariant::Enhanced)
            .adaptive_pa(true)
            .build()
            .unwrap();
        
        assert_eq!(params.population_size, 30);
        assert_relative_eq!(params.discovery_probability, 0.3);
        assert_relative_eq!(params.levy_lambda, 1.5);
        assert!(params.adaptive_pa);
        assert!(matches!(params.variant, CsVariant::Enhanced));
    }
    
    #[test]
    fn test_levy_flight() {
        let nest = Nest::new(3, (-5.0, 5.0));
        
        let levy_step = nest.levy_flight_step(1.5, LevyDistribution::Mantegna, 1.0);
        assert_eq!(levy_step.len(), 3);
        
        // Check that Lévy flight produces reasonable steps
        let norm = levy_step.norm();
        assert!(norm > 0.0);
        assert!(norm < 100.0); // Should not be extremely large
    }
    
    #[test]
    fn test_parameter_validation() {
        let result = CsParameters::builder()
            .population_size(0) // Invalid
            .build();
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_nest_creation() {
        let nest = Nest::new(3, (-5.0, 5.0));
        
        assert_eq!(nest.position().len(), 3);
        assert_eq!(*nest.fitness(), f64::INFINITY);
        assert_eq!(nest.discovery_count(), 0);
        assert_eq!(nest.age(), 0);
    }
    
    #[test]
    fn test_quantum_features() {
        let mut nest = Nest::new(3, (-5.0, 5.0));
        nest.initialize_quantum_state(3);
        
        assert!(nest.quantum_state.is_some());
        assert_eq!(nest.quantum_state.as_ref().unwrap().len(), 3);
        
        nest.apply_quantum_rotation(0.1, (-5.0, 5.0));
        // Position should be updated based on quantum state
    }
}