//! Artificial Bee Colony (ABC) optimization implementation
//!
//! ABC is inspired by the foraging behavior of honey bees. The algorithm
//! consists of three types of bees: employed bees, onlooker bees, and scout bees,
//! each with different roles in the optimization process.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rayon::prelude::*;
use rand::prelude::*;

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Population, Individual, Position, AlgorithmMetrics,
    AdaptiveAlgorithm, ParallelAlgorithm, AdaptationStrategy
};
use crate::validate_parameter;

/// ABC algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbcParameters {
    /// Total number of bees in the colony
    pub colony_size: usize,
    
    /// Number of employed bees (usually colony_size / 2)
    pub num_employed_bees: usize,
    
    /// Number of onlooker bees (usually colony_size / 2)
    pub num_onlooker_bees: usize,
    
    /// Maximum number of trials before abandoning a food source
    pub limit: usize,
    
    /// Enable modified ABC variant
    pub modified_abc: bool,
    
    /// Enable best-so-far ABC variant
    pub best_so_far_abc: bool,
    
    /// Enable gbest-guided ABC variant
    pub gbest_guided_abc: bool,
    
    /// Guide factor for gbest-guided ABC
    pub guide_factor: f64,
    
    /// Acceleration coefficient for modified ABC
    pub acceleration_coefficient: f64,
    
    /// Enable adaptive limit
    pub adaptive_limit: bool,
    
    /// Minimum limit for adaptive version
    pub min_limit: usize,
    
    /// Maximum limit for adaptive version
    pub max_limit: usize,
    
    /// Enable opposition-based learning
    pub opposition_based: bool,
    
    /// Probability of applying Levy flight
    pub levy_flight_probability: f64,
    
    /// Enable chaotic initialization
    pub chaotic_initialization: bool,
    
    /// Chaotic map parameter
    pub chaotic_parameter: f64,
}

impl Default for AbcParameters {
    fn default() -> Self {
        Self {
            colony_size: 40,
            num_employed_bees: 20,
            num_onlooker_bees: 20,
            limit: 100,
            modified_abc: false,
            best_so_far_abc: false,
            gbest_guided_abc: false,
            guide_factor: 1.0,
            acceleration_coefficient: 1.0,
            adaptive_limit: false,
            min_limit: 10,
            max_limit: 200,
            opposition_based: false,
            levy_flight_probability: 0.1,
            chaotic_initialization: false,
            chaotic_parameter: 4.0,
        }
    }
}

impl AbcParameters {
    /// Validate parameters
    pub fn validate(&self) -> Result<(), SwarmError> {
        validate_parameter!(self.colony_size, "colony_size", 4, 10000);
        validate_parameter!(self.num_employed_bees, "num_employed_bees", 1, self.colony_size);
        validate_parameter!(self.num_onlooker_bees, "num_onlooker_bees", 1, self.colony_size);
        validate_parameter!(self.limit, "limit", 1, 1000);
        validate_parameter!(self.guide_factor, "guide_factor", 0.0, 10.0);
        validate_parameter!(self.acceleration_coefficient, "acceleration_coefficient", 0.0, 10.0);
        validate_parameter!(self.min_limit, "min_limit", 1, self.max_limit);
        validate_parameter!(self.levy_flight_probability, "levy_flight_probability", 0.0, 1.0);
        validate_parameter!(self.chaotic_parameter, "chaotic_parameter", 0.0, 4.0);
        
        if self.num_employed_bees + self.num_onlooker_bees > self.colony_size {
            return Err(SwarmError::parameter(
                "Sum of employed and onlooker bees exceeds colony size"
            ));
        }
        
        Ok(())
    }
    
    /// Create builder for ABC parameters
    pub fn builder() -> AbcParametersBuilder {
        AbcParametersBuilder::new()
    }
}

/// Builder for ABC parameters
pub struct AbcParametersBuilder {
    params: AbcParameters,
}

impl AbcParametersBuilder {
    pub fn new() -> Self {
        Self {
            params: AbcParameters::default(),
        }
    }
    
    pub fn colony_size(mut self, size: usize) -> Self {
        self.params.colony_size = size;
        // Auto-adjust bee counts
        self.params.num_employed_bees = size / 2;
        self.params.num_onlooker_bees = size / 2;
        self
    }
    
    pub fn employed_bees(mut self, count: usize) -> Self {
        self.params.num_employed_bees = count;
        self
    }
    
    pub fn onlooker_bees(mut self, count: usize) -> Self {
        self.params.num_onlooker_bees = count;
        self
    }
    
    pub fn limit(mut self, limit: usize) -> Self {
        self.params.limit = limit;
        self
    }
    
    pub fn modified_abc(mut self, enabled: bool) -> Self {
        self.params.modified_abc = enabled;
        self
    }
    
    pub fn gbest_guided_abc(mut self, enabled: bool) -> Self {
        self.params.gbest_guided_abc = enabled;
        self
    }
    
    pub fn opposition_based(mut self, enabled: bool) -> Self {
        self.params.opposition_based = enabled;
        self
    }
    
    pub fn chaotic_initialization(mut self, enabled: bool) -> Self {
        self.params.chaotic_initialization = enabled;
        self
    }
    
    pub fn build(self) -> Result<AbcParameters, SwarmError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

/// Bee types in the colony
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BeeType {
    Employed,
    Onlooker,
    Scout,
}

/// Artificial bee representation
#[derive(Debug, Clone)]
pub struct Bee {
    /// Current position (food source)
    position: Position,
    
    /// Fitness value (nectar amount)
    fitness: f64,
    
    /// Trial counter for abandonment
    trial_count: usize,
    
    /// Bee type
    bee_type: BeeType,
    
    /// Probability for onlooker selection
    selection_probability: f64,
    
    /// Improvement count for adaptive mechanisms
    improvement_count: usize,
    
    /// Total evaluations for this bee
    total_evaluations: usize,
}

impl Bee {
    /// Create a new bee
    pub fn new(position: Position, bee_type: BeeType) -> Self {
        Self {
            position,
            fitness: f64::INFINITY,
            trial_count: 0,
            bee_type,
            selection_probability: 0.0,
            improvement_count: 0,
            total_evaluations: 0,
        }
    }
    
    /// Create random bee within bounds
    pub fn random(dimensions: usize, bounds: (f64, f64), bee_type: BeeType) -> Self {
        let mut rng = thread_rng();
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(bounds.0..=bounds.1)
        });
        Self::new(position, bee_type)
    }
    
    /// Create bee with chaotic initialization
    pub fn chaotic(dimensions: usize, bounds: (f64, f64), bee_type: BeeType, chaotic_param: f64) -> Self {
        let mut rng = thread_rng();
        let mut chaotic_value = rng.gen::<f64>();
        
        let position = Position::from_fn(dimensions, |_, _| {
            // Logistic map for chaotic initialization
            chaotic_value = chaotic_param * chaotic_value * (1.0 - chaotic_value);
            bounds.0 + chaotic_value * (bounds.1 - bounds.0)
        });
        
        Self::new(position, bee_type)
    }
    
    /// Search for new food source (employed bee phase)
    pub fn employed_bee_search(
        &mut self,
        colony: &[Bee],
        problem: &OptimizationProblem,
        params: &AbcParameters,
    ) -> Result<(), SwarmError> {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        // Select a random partner bee
        let partner_idx = loop {
            let idx = rng.gen_range(0..colony.len());
            if idx != self.get_index_in_colony(colony) {
                break idx;
            }
        };
        
        let partner = &colony[partner_idx];
        let mut new_position = self.position.clone();
        
        // Select random dimension to modify
        let dim_to_modify = rng.gen_range(0..dimensions);
        
        if params.modified_abc {
            // Modified ABC: use best solution information
            if let Some(best_bee) = colony.iter().min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()) {
                let phi = rng.gen_range(-1.0..1.0);
                new_position[dim_to_modify] = self.position[dim_to_modify] + 
                    phi * (best_bee.position[dim_to_modify] - partner.position[dim_to_modify]);
            }
        } else if params.gbest_guided_abc {
            // Gbest-guided ABC
            if let Some(best_bee) = colony.iter().min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap()) {
                let phi = rng.gen_range(-1.0..1.0);
                let c = params.guide_factor;
                
                new_position[dim_to_modify] = self.position[dim_to_modify] + 
                    phi * (partner.position[dim_to_modify] - self.position[dim_to_modify]) +
                    c * rng.gen_range(-1.0..1.0) * (best_bee.position[dim_to_modify] - self.position[dim_to_modify]);
            }
        } else {
            // Standard ABC
            let phi = rng.gen_range(-1.0..1.0);
            new_position[dim_to_modify] = self.position[dim_to_modify] + 
                phi * (self.position[dim_to_modify] - partner.position[dim_to_modify]);
        }
        
        // Apply Levy flight if enabled
        if rng.gen::<f64>() < params.levy_flight_probability {
            let levy_step = self.generate_levy_flight();
            new_position[dim_to_modify] += levy_step;
        }
        
        // Ensure bounds
        new_position[dim_to_modify] = new_position[dim_to_modify].clamp(
            problem.lower_bounds[dim_to_modify],
            problem.upper_bounds[dim_to_modify],
        );
        
        // Evaluate new position
        let new_fitness = problem.evaluate(&new_position);
        self.total_evaluations += 1;
        
        // Greedy selection
        if new_fitness < self.fitness {
            self.position = new_position;
            self.fitness = new_fitness;
            self.trial_count = 0;
            self.improvement_count += 1;
        } else {
            self.trial_count += 1;
        }
        
        Ok(())
    }
    
    /// Generate Levy flight step
    fn generate_levy_flight(&self) -> f64 {
        let mut rng = thread_rng();
        
        // Mantegna's algorithm for Levy flight
        let beta = 1.5;
        let sigma_u = (
            (1.0 + beta).sin() * (1.0 + beta) / 2.0 / 
            ((1.0 + beta) / 2.0).exp() / 
            beta / 
            2_f64.powf((beta - 1.0) / 2.0)
        ).powf(1.0 / beta);
        
        let u: f64 = rng.sample(rand_distr::Normal::new(0.0, sigma_u).unwrap());
        let v: f64 = rng.sample(rand_distr::Normal::new(0.0, 1.0).unwrap());
        
        0.01 * u / v.abs().powf(1.0 / beta)
    }
    
    /// Calculate selection probability for onlooker bees
    pub fn calculate_selection_probability(&mut self, max_fitness: f64) {
        // Use fitness proportionate selection
        if max_fitness > 0.0 {
            // Convert to minimization: higher fitness = lower probability
            let normalized_fitness = if self.fitness.is_finite() {
                1.0 / (1.0 + self.fitness)
            } else {
                0.0
            };
            self.selection_probability = normalized_fitness;
        } else {
            self.selection_probability = 0.0;
        }
    }
    
    /// Check if this bee should become a scout
    pub fn should_become_scout(&self, limit: usize) -> bool {
        self.trial_count >= limit
    }
    
    /// Convert to scout bee (random initialization)
    pub fn become_scout(&mut self, problem: &OptimizationProblem, params: &AbcParameters) {
        let mut rng = thread_rng();
        
        if params.chaotic_initialization {
            *self = Self::chaotic(
                problem.dimensions,
                (problem.lower_bounds.min(), problem.upper_bounds.max()),
                BeeType::Scout,
                params.chaotic_parameter,
            );
        } else {
            for i in 0..self.position.len() {
                self.position[i] = rng.gen_range(
                    problem.lower_bounds[i]..=problem.upper_bounds[i]
                );
            }
        }
        
        self.trial_count = 0;
        self.bee_type = BeeType::Scout;
        self.fitness = problem.evaluate(&self.position);
        self.total_evaluations += 1;
    }
    
    /// Get index of this bee in the colony
    fn get_index_in_colony(&self, colony: &[Bee]) -> usize {
        // Simple approach: use memory address comparison
        colony.iter().position(|b| std::ptr::eq(b, self)).unwrap_or(0)
    }
    
    /// Get bee type
    pub fn bee_type(&self) -> BeeType {
        self.bee_type
    }
    
    /// Set bee type
    pub fn set_bee_type(&mut self, bee_type: BeeType) {
        self.bee_type = bee_type;
    }
    
    /// Get selection probability
    pub fn selection_probability(&self) -> f64 {
        self.selection_probability
    }
    
    /// Get trial count
    pub fn trial_count(&self) -> usize {
        self.trial_count
    }
    
    /// Get improvement rate
    pub fn improvement_rate(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.improvement_count as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }
}

impl Individual for Bee {
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

/// Artificial Bee Colony optimization algorithm
#[derive(Debug, Clone)]
pub struct ArtificialBeeColony {
    /// Algorithm parameters
    parameters: AbcParameters,
    
    /// Colony of bees
    colony: Population<Bee>,
    
    /// Best bee found so far
    best_bee: Option<Arc<Bee>>,
    
    /// Best fitness
    best_fitness: f64,
    
    /// Current iteration
    iteration: usize,
    
    /// Optimization problem
    problem: Option<Arc<OptimizationProblem>>,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
    
    /// Adaptive limit for scout bee conversion
    adaptive_limit: usize,
    
    /// Success rate history for adaptation
    success_history: Vec<f64>,
}

impl ArtificialBeeColony {
    /// Create a new ABC algorithm with default parameters
    pub fn new() -> Self {
        Self::with_parameters(AbcParameters::default())
    }
    
    /// Create ABC with specific parameters
    pub fn with_parameters(parameters: AbcParameters) -> Self {
        Self {
            adaptive_limit: parameters.limit,
            parameters,
            colony: Population::new(),
            best_bee: None,
            best_fitness: f64::INFINITY,
            iteration: 0,
            problem: None,
            metrics: AlgorithmMetrics::default(),
            success_history: Vec::new(),
        }
    }
    
    /// Builder pattern for ABC construction
    pub fn builder() -> AbcBuilder {
        AbcBuilder::new()
    }
    
    /// Employed bees phase
    async fn employed_bees_phase(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::optimization("Problem not set"))?;
        
        for i in 0..self.parameters.num_employed_bees {
            if i < self.colony.size() {
                let colony_clone = self.colony.individuals.clone();
                self.colony.individuals[i].employed_bee_search(
                    &colony_clone,
                    problem,
                    &self.parameters,
                )?;
                
                // Update global best
                if self.colony.individuals[i].fitness < self.best_fitness {
                    self.best_fitness = self.colony.individuals[i].fitness;
                    self.best_bee = Some(Arc::new(self.colony.individuals[i].clone()));
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate selection probabilities for onlooker bees
    fn calculate_selection_probabilities(&mut self) {
        let max_fitness = self.colony.individuals[..self.parameters.num_employed_bees]
            .iter()
            .map(|bee| if bee.fitness.is_finite() { 1.0 / (1.0 + bee.fitness) } else { 0.0 })
            .fold(0.0, f64::max);
        
        for i in 0..self.parameters.num_employed_bees {
            self.colony.individuals[i].calculate_selection_probability(max_fitness);
        }
    }
    
    /// Onlooker bees phase
    async fn onlooker_bees_phase(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::optimization("Problem not set"))?;
        
        // Calculate selection probabilities
        self.calculate_selection_probabilities();
        
        // Roulette wheel selection for onlooker bees
        let mut rng = thread_rng();
        let mut onlooker_count = 0;
        
        while onlooker_count < self.parameters.num_onlooker_bees {
            let total_probability: f64 = self.colony.individuals[..self.parameters.num_employed_bees]
                .iter()
                .map(|bee| bee.selection_probability())
                .sum();
            
            if total_probability > 0.0 {
                let r: f64 = rng.gen::<f64>() * total_probability;
                let mut cumulative_prob = 0.0;
                
                for i in 0..self.parameters.num_employed_bees {
                    cumulative_prob += self.colony.individuals[i].selection_probability();
                    
                    if r <= cumulative_prob {
                        // Selected food source for onlooker bee
                        let colony_clone = self.colony.individuals.clone();
                        self.colony.individuals[i].employed_bee_search(
                            &colony_clone,
                            problem,
                            &self.parameters,
                        )?;
                        
                        // Update global best
                        if self.colony.individuals[i].fitness < self.best_fitness {
                            self.best_fitness = self.colony.individuals[i].fitness;
                            self.best_bee = Some(Arc::new(self.colony.individuals[i].clone()));
                        }
                        
                        onlooker_count += 1;
                        break;
                    }
                }
            } else {
                // If no valid probabilities, select randomly
                let selected_idx = rng.gen_range(0..self.parameters.num_employed_bees);
                let colony_clone = self.colony.individuals.clone();
                self.colony.individuals[selected_idx].employed_bee_search(
                    &colony_clone,
                    problem,
                    &self.parameters,
                )?;
                onlooker_count += 1;
            }
        }
        
        Ok(())
    }
    
    /// Scout bees phase
    async fn scout_bees_phase(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::optimization("Problem not set"))?;
        
        let current_limit = if self.parameters.adaptive_limit {
            self.adaptive_limit
        } else {
            self.parameters.limit
        };
        
        let mut scout_count = 0;
        
        for i in 0..self.parameters.num_employed_bees {
            if self.colony.individuals[i].should_become_scout(current_limit) {
                self.colony.individuals[i].become_scout(problem, &self.parameters);
                scout_count += 1;
                
                // Update global best if scout found better solution
                if self.colony.individuals[i].fitness < self.best_fitness {
                    self.best_fitness = self.colony.individuals[i].fitness;
                    self.best_bee = Some(Arc::new(self.colony.individuals[i].clone()));
                }
            }
        }
        
        // Update success history for adaptive limit
        if self.parameters.adaptive_limit {
            let success_rate = if scout_count > 0 {
                scout_count as f64 / self.parameters.num_employed_bees as f64
            } else {
                0.0
            };
            
            self.success_history.push(success_rate);
            
            // Keep only recent history
            if self.success_history.len() > 50 {
                self.success_history.remove(0);
            }
            
            // Adapt limit based on success rate
            self.adapt_limit();
        }
        
        Ok(())
    }
    
    /// Adapt the limit parameter based on performance
    fn adapt_limit(&mut self) {
        if self.success_history.is_empty() {
            return;
        }
        
        let avg_success_rate = self.success_history.iter().sum::<f64>() / self.success_history.len() as f64;
        
        if avg_success_rate > 0.3 {
            // High scout conversion rate - increase limit to give more chances
            self.adaptive_limit = (self.adaptive_limit + 5).min(self.parameters.max_limit);
        } else if avg_success_rate < 0.1 {
            // Low scout conversion rate - decrease limit to promote exploration
            self.adaptive_limit = (self.adaptive_limit.saturating_sub(5)).max(self.parameters.min_limit);
        }
    }
    
    /// Generate opposition-based population
    fn generate_opposition_based_population(&self) -> Result<Vec<Bee>, SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::optimization("Problem not set"))?;
        
        let mut opposition_bees = Vec::new();
        
        for bee in self.colony.iter() {
            let mut opposition_position = Position::zeros(bee.position().len());
            
            for i in 0..opposition_position.len() {
                opposition_position[i] = problem.lower_bounds[i] + problem.upper_bounds[i] - bee.position()[i];
            }
            
            let mut opposition_bee = Bee::new(opposition_position, bee.bee_type());
            opposition_bee.fitness = problem.evaluate(opposition_bee.position());
            
            opposition_bees.push(opposition_bee);
        }
        
        Ok(opposition_bees)
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        self.colony.diversity().unwrap_or(0.0)
    }
    
    /// Check convergence
    fn has_converged(&self) -> bool {
        let diversity = self.calculate_diversity();
        diversity < 1e-6 || self.iteration > 1500
    }
}

#[async_trait]
impl SwarmAlgorithm for ArtificialBeeColony {
    type Individual = Bee;
    type Fitness = f64;
    type Parameters = AbcParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.parameters.validate()?;
        
        // Store problem
        self.problem = Some(Arc::new(problem));
        let problem_ref = self.problem.as_ref().unwrap();
        
        // Initialize colony
        self.colony = Population::with_capacity(self.parameters.colony_size);
        let bounds = (problem_ref.lower_bounds.min(), problem_ref.upper_bounds.max());
        
        // Create employed bees
        for _ in 0..self.parameters.num_employed_bees {
            let bee = if self.parameters.chaotic_initialization {
                Bee::chaotic(problem_ref.dimensions, bounds, BeeType::Employed, self.parameters.chaotic_parameter)
            } else {
                Bee::random(problem_ref.dimensions, bounds, BeeType::Employed)
            };
            self.colony.add(bee);
        }
        
        // Fill remaining with onlooker bees initially
        for _ in self.parameters.num_employed_bees..self.parameters.colony_size {
            let bee = if self.parameters.chaotic_initialization {
                Bee::chaotic(problem_ref.dimensions, bounds, BeeType::Onlooker, self.parameters.chaotic_parameter)
            } else {
                Bee::random(problem_ref.dimensions, bounds, BeeType::Onlooker)
            };
            self.colony.add(bee);
        }
        
        // Initial evaluation
        for bee in self.colony.iter_mut() {
            let fitness = problem_ref.evaluate(bee.position());
            bee.set_fitness(fitness);
            
            if fitness < self.best_fitness {
                self.best_fitness = fitness;
                self.best_bee = Some(Arc::new(bee.clone()));
            }
        }
        
        // Opposition-based initialization
        if self.parameters.opposition_based {
            let opposition_bees = self.generate_opposition_based_population()?;
            
            // Combine original and opposition populations
            let mut all_bees = self.colony.individuals.clone();
            all_bees.extend(opposition_bees);
            
            // Select best bees
            all_bees.sort_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap());
            self.colony.individuals = all_bees.into_iter()
                .take(self.parameters.colony_size)
                .collect();
            
            // Update best
            if let Some(best) = self.colony.best() {
                self.best_fitness = *best.fitness();
                self.best_bee = Some(Arc::new(best.clone()));
            }
        }
        
        // Initialize metrics
        self.metrics = AlgorithmMetrics {
            iteration: 0,
            best_fitness: Some(self.best_fitness),
            average_fitness: self.colony.average_fitness(),
            diversity: Some(self.calculate_diversity()),
            convergence_rate: None,
            evaluations: self.parameters.colony_size,
            time_per_iteration: None,
            memory_usage: None,
        };
        
        tracing::info!(
            "ABC initialized with {} bees ({} employed, {} onlooker)",
            self.parameters.colony_size,
            self.parameters.num_employed_bees,
            self.parameters.num_onlooker_bees
        );
        
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        self.iteration += 1;
        
        // Employed bees phase
        self.employed_bees_phase().await?;
        
        // Onlooker bees phase
        self.onlooker_bees_phase().await?;
        
        // Scout bees phase
        self.scout_bees_phase().await?;
        
        // Update metrics
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.average_fitness = self.colony.average_fitness();
        self.metrics.diversity = Some(self.calculate_diversity());
        self.metrics.time_per_iteration = Some(start_time.elapsed().as_micros() as u64);
        
        Ok(())
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.best_bee.as_ref().map(|arc| arc.as_ref())
    }
    
    fn get_population(&self) -> &Population<Self::Individual> {
        &self.colony
    }
    
    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        &mut self.colony
    }
    
    fn has_converged(&self) -> bool {
        self.has_converged()
    }
    
    fn name(&self) -> &'static str {
        "ArtificialBeeColony"
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
        self.colony = Population::new();
        self.best_bee = None;
        self.best_fitness = f64::INFINITY;
        self.iteration = 0;
        self.metrics = AlgorithmMetrics::default();
        self.adaptive_limit = self.parameters.limit;
        self.success_history.clear();
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

impl AdaptiveAlgorithm for ArtificialBeeColony {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        // Adapt limit based on diversity and convergence
        if let Some(diversity) = performance_metrics.diversity {
            if diversity < 0.1 {
                // Low diversity - decrease limit to promote more exploration
                self.adaptive_limit = (self.adaptive_limit.saturating_sub(10))
                    .max(self.parameters.min_limit);
            } else if diversity > 0.5 {
                // High diversity - increase limit to allow more exploitation
                self.adaptive_limit = (self.adaptive_limit + 10)
                    .min(self.parameters.max_limit);
            }
        }
        
        // Adaptive guide factor for gbest-guided ABC
        if self.parameters.gbest_guided_abc {
            let progress = self.iteration as f64 / 1000.0; // Assume max 1000 iterations
            self.parameters.guide_factor = 2.0 * (1.0 - progress);
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::Linear {
            start: self.parameters.max_limit as f64,
            end: self.parameters.min_limit as f64,
        }
    }
}

impl ParallelAlgorithm for ArtificialBeeColony {
    async fn parallel_step(&mut self, _thread_count: usize) -> Result<(), SwarmError> {
        // ABC can be parallelized at the bee level
        // For now, use the standard step implementation
        self.step().await
    }
    
    fn optimal_thread_count(&self) -> usize {
        (self.parameters.colony_size / 8).max(1).min(num_cpus::get())
    }
}

/// Builder for ABC algorithm
pub struct AbcBuilder {
    parameters: AbcParameters,
}

impl AbcBuilder {
    pub fn new() -> Self {
        Self {
            parameters: AbcParameters::default(),
        }
    }
    
    pub fn colony_size(mut self, size: usize) -> Self {
        self.parameters.colony_size = size;
        self.parameters.num_employed_bees = size / 2;
        self.parameters.num_onlooker_bees = size / 2;
        self
    }
    
    pub fn limit(mut self, limit: usize) -> Self {
        self.parameters.limit = limit;
        self
    }
    
    pub fn modified_abc(mut self, enabled: bool) -> Self {
        self.parameters.modified_abc = enabled;
        self
    }
    
    pub fn gbest_guided_abc(mut self, enabled: bool) -> Self {
        self.parameters.gbest_guided_abc = enabled;
        self
    }
    
    pub fn opposition_based(mut self, enabled: bool) -> Self {
        self.parameters.opposition_based = enabled;
        self
    }
    
    pub fn build(self) -> Result<ArtificialBeeColony, SwarmError> {
        self.parameters.validate()?;
        Ok(ArtificialBeeColony::with_parameters(self.parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_abc_initialization() {
        let mut abc = ArtificialBeeColony::new();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-10.0, 10.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert!(abc.initialize(problem).await.is_ok());
        assert_eq!(abc.colony.size(), 40);
        assert!(abc.best_bee.is_some());
    }
    
    #[tokio::test]
    async fn test_abc_optimization() {
        let mut abc = ArtificialBeeColony::builder()
            .colony_size(20)
            .limit(50)
            .build()
            .unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let result = abc.optimize(50).await.unwrap();
        
        // Should find reasonable solution
        assert!(result.best_fitness < 10.0);
        assert!(result.iterations <= 50);
        assert_eq!(result.algorithm_name, "ArtificialBeeColony");
    }
    
    #[test]
    fn test_abc_parameters() {
        let params = AbcParameters::builder()
            .colony_size(60)
            .limit(80)
            .modified_abc(true)
            .gbest_guided_abc(true)
            .opposition_based(true)
            .build()
            .unwrap();
        
        assert_eq!(params.colony_size, 60);
        assert_eq!(params.num_employed_bees, 30);
        assert_eq!(params.num_onlooker_bees, 30);
        assert_eq!(params.limit, 80);
        assert!(params.modified_abc);
        assert!(params.gbest_guided_abc);
        assert!(params.opposition_based);
    }
    
    #[test]
    fn test_bee_creation() {
        let position = Position::from_vec(vec![1.0, 2.0, 3.0]);
        let bee = Bee::new(position.clone(), BeeType::Employed);
        
        assert_eq!(bee.position(), &position);
        assert_eq!(*bee.fitness(), f64::INFINITY);
        assert_eq!(bee.bee_type(), BeeType::Employed);
        assert_eq!(bee.trial_count(), 0);
        assert_eq!(bee.selection_probability(), 0.0);
    }
    
    #[test]
    fn test_scout_conversion() {
        let position = Position::from_vec(vec![1.0, 2.0]);
        let mut bee = Bee::new(position, BeeType::Employed);
        
        // Simulate failed trials
        for _ in 0..100 {
            bee.trial_count += 1;
        }
        
        assert!(bee.should_become_scout(50));
        assert!(!bee.should_become_scout(150));
    }
}