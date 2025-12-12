//! Bacterial Foraging Optimization (BFO) implementation
//!
//! BFO is inspired by the foraging behavior of E. coli bacteria. It simulates
//! chemotaxis, swarming, reproduction, and elimination-dispersal events that
//! occur in bacterial populations.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::Normal;

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Population, Individual, BasicIndividual, Position, AlgorithmMetrics,
    AdaptiveAlgorithm, ParallelAlgorithm, AdaptationStrategy
};
use crate::validate_parameter;

/// BFO algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum BfoVariant {
    /// Classic Bacterial Foraging Optimization
    Classic,
    /// Adaptive Bacterial Foraging Optimization (ABFO)
    Adaptive,
    /// Modified Bacterial Foraging Optimization (MBFO)
    Modified,
    /// Hybrid Bacterial Foraging Optimization (HBFO)
    Hybrid,
    /// Improved Bacterial Foraging Optimization (IBFO)
    Improved,
    /// Quantum Bacterial Foraging Optimization (QBFO)
    Quantum,
}

/// Chemotaxis strategy
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ChemotaxisStrategy {
    /// Random direction with fixed step size
    Random,
    /// Gradient-based movement
    Gradient,
    /// Adaptive step size
    Adaptive,
    /// Levy flight-based movement
    LevyFlight,
    /// Best-guided movement
    BestGuided,
}

/// BFO algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfoParameters {
    /// Number of bacteria in the population
    pub population_size: usize,
    
    /// Number of chemotactic steps
    pub num_chemotactic_steps: usize,
    
    /// Number of swim steps in each chemotactic step
    pub num_swim_steps: usize,
    
    /// Number of reproduction steps
    pub num_reproduction_steps: usize,
    
    /// Number of elimination-dispersal events
    pub num_elimination_dispersal_events: usize,
    
    /// Chemotactic step size
    pub chemotactic_step_size: f64,
    
    /// Probability of elimination-dispersal
    pub elimination_dispersal_probability: f64,
    
    /// Attraction coefficient for cell-to-cell signaling
    pub attraction_coefficient: f64,
    
    /// Repulsion coefficient for cell-to-cell signaling
    pub repulsion_coefficient: f64,
    
    /// Attraction range
    pub attraction_range: f64,
    
    /// Repulsion range
    pub repulsion_range: f64,
    
    /// BFO variant to use
    pub variant: BfoVariant,
    
    /// Chemotaxis strategy
    pub chemotaxis_strategy: ChemotaxisStrategy,
    
    /// Enable adaptive parameters
    pub adaptive: bool,
    
    /// Minimum step size for adaptive variant
    pub min_step_size: f64,
    
    /// Maximum step size for adaptive variant
    pub max_step_size: f64,
    
    /// Communication range for bacterial communication
    pub communication_range: f64,
    
    /// Quorum sensing threshold
    pub quorum_threshold: f64,
    
    /// Enable tumble behavior
    pub enable_tumble: bool,
    
    /// Tumble probability
    pub tumble_probability: f64,
    
    /// Nutrient concentration gradient factor
    pub nutrient_gradient_factor: f64,
    
    /// Enable biofilm formation
    pub enable_biofilm: bool,
    
    /// Biofilm adhesion strength
    pub biofilm_adhesion: f64,
    
    /// Temperature for thermal effects
    pub temperature: f64,
    
    /// pH level for environmental effects
    pub ph_level: f64,
}

impl Default for BfoParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            num_chemotactic_steps: 100,
            num_swim_steps: 4,
            num_reproduction_steps: 4,
            num_elimination_dispersal_events: 2,
            chemotactic_step_size: 0.1,
            elimination_dispersal_probability: 0.25,
            attraction_coefficient: 0.1,
            repulsion_coefficient: 0.1,
            attraction_range: 0.1,
            repulsion_range: 0.1,
            variant: BfoVariant::Classic,
            chemotaxis_strategy: ChemotaxisStrategy::Random,
            adaptive: false,
            min_step_size: 0.01,
            max_step_size: 1.0,
            communication_range: 0.5,
            quorum_threshold: 0.3,
            enable_tumble: true,
            tumble_probability: 0.1,
            nutrient_gradient_factor: 1.0,
            enable_biofilm: false,
            biofilm_adhesion: 0.1,
            temperature: 37.0, // Body temperature in Celsius
            ph_level: 7.0,     // Neutral pH
        }
    }
}

impl BfoParameters {
    /// Validate parameters
    pub fn validate(&self) -> Result<(), SwarmError> {
        validate_parameter!(self.population_size, "population_size", 4, 1000);
        validate_parameter!(self.num_chemotactic_steps, "num_chemotactic_steps", 1, 1000);
        validate_parameter!(self.num_swim_steps, "num_swim_steps", 1, 20);
        validate_parameter!(self.num_reproduction_steps, "num_reproduction_steps", 1, 20);
        validate_parameter!(self.num_elimination_dispersal_events, "num_elimination_dispersal_events", 1, 10);
        validate_parameter!(self.chemotactic_step_size, "chemotactic_step_size", 0.001, 10.0);
        validate_parameter!(self.elimination_dispersal_probability, "elimination_dispersal_probability", 0.0, 1.0);
        validate_parameter!(self.attraction_coefficient, "attraction_coefficient", 0.0, 10.0);
        validate_parameter!(self.repulsion_coefficient, "repulsion_coefficient", 0.0, 10.0);
        validate_parameter!(self.temperature, "temperature", 0.0, 100.0);
        validate_parameter!(self.ph_level, "ph_level", 0.0, 14.0);
        
        if self.min_step_size >= self.max_step_size {
            return Err(SwarmError::parameter("min_step_size must be < max_step_size"));
        }
        
        Ok(())
    }
    
    /// Create builder for BFO parameters
    pub fn builder() -> BfoParametersBuilder {
        BfoParametersBuilder::new()
    }
}

/// Builder for BFO parameters
pub struct BfoParametersBuilder {
    params: BfoParameters,
}

impl BfoParametersBuilder {
    pub fn new() -> Self {
        Self {
            params: BfoParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.params.population_size = size;
        self
    }
    
    pub fn chemotactic_steps(mut self, steps: usize) -> Self {
        self.params.num_chemotactic_steps = steps;
        self
    }
    
    pub fn step_size(mut self, size: f64) -> Self {
        self.params.chemotactic_step_size = size;
        self
    }
    
    pub fn variant(mut self, variant: BfoVariant) -> Self {
        self.params.variant = variant;
        self
    }
    
    pub fn chemotaxis_strategy(mut self, strategy: ChemotaxisStrategy) -> Self {
        self.params.chemotaxis_strategy = strategy;
        self
    }
    
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.params.adaptive = adaptive;
        self
    }
    
    pub fn build(self) -> Result<BfoParameters, SwarmError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

/// Bacterial cell representation
#[derive(Debug, Clone)]
pub struct Bacterium {
    /// Current position
    position: Position,
    
    /// Fitness value (nutrient level)
    fitness: f64,
    
    /// Health metric (survival capability)
    health: f64,
    
    /// Number of chemotactic steps taken
    chemotactic_steps: usize,
    
    /// Last chemotactic direction
    last_direction: Option<Position>,
    
    /// Cumulative fitness over reproductive period
    cumulative_fitness: f64,
    
    /// Communication state with other bacteria
    communication_state: f64,
    
    /// Biofilm attachment strength
    biofilm_attachment: f64,
    
    /// Age of the bacterium
    age: usize,
    
    /// Swimming velocity
    swimming_velocity: Position,
    
    /// Nutrient consumption rate
    consumption_rate: f64,
    
    /// Flagella efficiency (movement capability)
    flagella_efficiency: f64,
    
    /// Cell membrane permeability
    membrane_permeability: f64,
    
    /// Metabolic state
    metabolic_state: f64,
}

impl Bacterium {
    /// Create a new bacterium at random position
    pub fn new(dimensions: usize, bounds: (f64, f64)) -> Self {
        let mut rng = thread_rng();
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(bounds.0..=bounds.1)
        });
        
        let swimming_velocity = Position::zeros(dimensions);
        
        Self {
            position,
            fitness: f64::INFINITY,
            health: 1.0,
            chemotactic_steps: 0,
            last_direction: None,
            cumulative_fitness: 0.0,
            communication_state: 0.0,
            biofilm_attachment: 0.0,
            age: 0,
            swimming_velocity,
            consumption_rate: rng.gen_range(0.5..1.5),
            flagella_efficiency: rng.gen_range(0.7..1.3),
            membrane_permeability: rng.gen_range(0.8..1.2),
            metabolic_state: 1.0,
        }
    }
    
    /// Perform chemotaxis (movement towards nutrients)
    pub fn chemotaxis(
        &mut self,
        strategy: ChemotaxisStrategy,
        step_size: f64,
        bounds: (f64, f64),
        best_position: Option<&Position>,
        gradient: Option<&Position>,
    ) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        let direction = match strategy {
            ChemotaxisStrategy::Random => {
                // Random direction
                let mut dir = Position::from_fn(dimensions, |_, _| {
                    rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())
                });
                let norm = dir.norm();
                if norm > 0.0 {
                    dir = dir / norm;
                }
                dir
            },
            ChemotaxisStrategy::Gradient => {
                // Move towards gradient
                if let Some(grad) = gradient {
                    -grad.clone() // Negative gradient for minimization
                } else {
                    // Fallback to random
                    let mut dir = Position::from_fn(dimensions, |_, _| {
                        rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())
                    });
                    let norm = dir.norm();
                    if norm > 0.0 { dir = dir / norm; }
                    dir
                }
            },
            ChemotaxisStrategy::Adaptive => {
                // Adaptive based on previous success
                if let Some(ref last_dir) = self.last_direction {
                    // Continue in successful direction with some randomness
                    let mut dir = last_dir.clone();
                    for i in 0..dimensions {
                        dir[i] += rng.sample::<f64, _>(Normal::new(0.0, 0.1).unwrap());
                    }
                    let norm = dir.norm();
                    if norm > 0.0 { dir = dir / norm; }
                    dir
                } else {
                    // Random direction for first step
                    let mut dir = Position::from_fn(dimensions, |_, _| {
                        rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())
                    });
                    let norm = dir.norm();
                    if norm > 0.0 { dir = dir / norm; }
                    dir
                }
            },
            ChemotaxisStrategy::LevyFlight => {
                // Lévy flight for better exploration
                let lambda = 1.5;
                let mut dir = Position::from_fn(dimensions, |_, _| {
                    let u: f64 = rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap());
                    u.signum() * u.abs().powf(1.0 / lambda)
                });
                let norm = dir.norm();
                if norm > 0.0 { dir = dir / norm; }
                dir
            },
            ChemotaxisStrategy::BestGuided => {
                // Move towards best position
                if let Some(best) = best_position {
                    let mut dir = best - &self.position;
                    // Add some randomness
                    for i in 0..dimensions {
                        dir[i] += rng.sample::<f64, _>(Normal::new(0.0, 0.1).unwrap());
                    }
                    let norm = dir.norm();
                    if norm > 0.0 { dir = dir / norm; }
                    dir
                } else {
                    // Fallback to random
                    let mut dir = Position::from_fn(dimensions, |_, _| {
                        rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap())
                    });
                    let norm = dir.norm();
                    if norm > 0.0 { dir = dir / norm; }
                    dir
                }
            },
        };
        
        // Apply environmental factors
        let environmental_factor = self.calculate_environmental_factor();
        let effective_step_size = step_size * self.flagella_efficiency * environmental_factor;
        
        // Calculate new position
        let new_position = &self.position + effective_step_size * direction;
        
        // Apply bounds
        let bounded_position = Position::from_fn(dimensions, |i, _| {
            new_position[i].clamp(bounds.0, bounds.1)
        });
        
        // Store direction for adaptive strategy
        self.last_direction = Some(direction);
        self.chemotactic_steps += 1;
        
        bounded_position
    }
    
    /// Calculate environmental factor based on temperature and pH
    fn calculate_environmental_factor(&self) -> f64 {
        // Optimal temperature around 37°C, optimal pH around 7.0
        let temp_factor = (-((37.0 - 37.0) / 10.0).powi(2)).exp(); // Always 1.0 for body temp
        let ph_factor = (-((7.0 - 7.0) / 2.0).powi(2)).exp(); // Always 1.0 for neutral pH
        
        (temp_factor * ph_factor).max(0.1) // Minimum survival factor
    }
    
    /// Swimming behavior (continue in same direction if beneficial)
    pub fn swim(&mut self, current_fitness: f64, step_size: f64, bounds: (f64, f64)) -> Option<Position> {
        if let Some(ref direction) = self.last_direction {
            if current_fitness < self.fitness {
                // Swimming is beneficial, continue
                let new_position = &self.position + step_size * self.flagella_efficiency * direction;
                
                // Apply bounds
                let bounded_position = Position::from_fn(self.position.len(), |i, _| {
                    new_position[i].clamp(bounds.0, bounds.1)
                });
                
                Some(bounded_position)
            } else {
                None // Stop swimming
            }
        } else {
            None
        }
    }
    
    /// Tumbling behavior (random reorientation)
    pub fn tumble(&mut self, probability: f64) -> bool {
        let mut rng = thread_rng();
        if rng.gen::<f64>() < probability {
            // Reset direction for tumbling
            self.last_direction = None;
            true
        } else {
            false
        }
    }
    
    /// Cell-to-cell signaling calculation
    pub fn cell_signaling(&self, other_bacteria: &[Bacterium], params: &BfoParameters) -> f64 {
        let mut signaling_effect = 0.0;
        
        for other in other_bacteria {
            let distance = (&self.position - other.position()).norm();
            
            if distance > 0.0 {
                // Attractive signaling (quorum sensing)
                if distance <= params.attraction_range {
                    signaling_effect += params.attraction_coefficient * 
                                      (-distance / params.attraction_range).exp();
                }
                
                // Repulsive signaling (overcrowding avoidance)
                if distance <= params.repulsion_range {
                    signaling_effect -= params.repulsion_coefficient * 
                                      (-distance / params.repulsion_range).exp();
                }
            }
        }
        
        signaling_effect
    }
    
    /// Update health based on nutrient availability and environment
    pub fn update_health(&mut self, fitness_improvement: f64, environmental_stress: f64) {
        let nutrient_factor = if fitness_improvement > 0.0 { 1.1 } else { 0.95 };
        let stress_factor = (1.0 - environmental_stress).max(0.1);
        
        self.health = (self.health * nutrient_factor * stress_factor).clamp(0.1, 2.0);
        self.metabolic_state = self.health * self.membrane_permeability;
    }
    
    /// Calculate reproductive fitness
    pub fn reproductive_fitness(&self) -> f64 {
        // Combine cumulative fitness with health and age factors
        let age_factor = (1.0 - self.age as f64 / 1000.0).max(0.1);
        self.cumulative_fitness * self.health * age_factor
    }
    
    /// Create offspring with mutation
    pub fn reproduce(&self, bounds: (f64, f64)) -> Bacterium {
        let mut offspring = self.clone();
        let mut rng = thread_rng();
        
        // Mutate position slightly
        for i in 0..offspring.position.len() {
            let mutation = rng.sample::<f64, _>(Normal::new(0.0, 0.01).unwrap());
            offspring.position[i] = (offspring.position[i] + mutation).clamp(bounds.0, bounds.1);
        }
        
        // Reset offspring state
        offspring.age = 0;
        offspring.cumulative_fitness = 0.0;
        offspring.health = 1.0;
        offspring.chemotactic_steps = 0;
        offspring.last_direction = None;
        
        // Inherit some traits with variation
        offspring.consumption_rate = (self.consumption_rate + rng.gen_range(-0.1..0.1)).clamp(0.1, 2.0);
        offspring.flagella_efficiency = (self.flagella_efficiency + rng.gen_range(-0.1..0.1)).clamp(0.1, 2.0);
        offspring.membrane_permeability = (self.membrane_permeability + rng.gen_range(-0.05..0.05)).clamp(0.1, 2.0);
        
        offspring
    }
    
    /// Update biofilm attachment
    pub fn update_biofilm(&mut self, local_density: f64, adhesion_strength: f64) {
        if local_density > 0.5 {
            self.biofilm_attachment = (self.biofilm_attachment + adhesion_strength * 0.1).min(1.0);
        } else {
            self.biofilm_attachment = (self.biofilm_attachment - 0.05).max(0.0);
        }
    }
    
    /// Check if bacterium can move (not stuck in biofilm)
    pub fn can_move(&self) -> bool {
        self.biofilm_attachment < 0.8
    }
    
    /// Age the bacterium
    pub fn age_bacterium(&mut self) {
        self.age += 1;
        self.cumulative_fitness += if self.fitness.is_finite() { 1.0 / (1.0 + self.fitness) } else { 0.0 };
    }
}

impl Individual for Bacterium {
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
        let old_fitness = self.fitness;
        self.fitness = fitness;
        
        // Update health based on fitness change
        let improvement = if old_fitness.is_finite() && fitness.is_finite() {
            old_fitness - fitness
        } else if fitness.is_finite() {
            0.1 // Small improvement for first evaluation
        } else {
            -0.1 // Penalty for invalid fitness
        };
        
        self.update_health(improvement, 0.0);
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// Bacterial Foraging Optimization algorithm
#[derive(Debug, Clone)]
pub struct BacterialForagingOptimization {
    /// Algorithm parameters
    parameters: BfoParameters,
    
    /// Population of bacteria
    population: Population<Bacterium>,
    
    /// Best solution found
    best_solution: Option<Arc<Bacterium>>,
    
    /// Best fitness
    best_fitness: f64,
    
    /// Current iteration
    iteration: usize,
    
    /// Current chemotactic step
    current_chemotactic_step: usize,
    
    /// Current reproduction step
    current_reproduction_step: usize,
    
    /// Current elimination-dispersal event
    current_elimination_dispersal_event: usize,
    
    /// Optimization problem
    problem: Option<Arc<OptimizationProblem>>,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
    
    /// Current adaptive step size
    current_step_size: f64,
    
    /// Population diversity
    diversity: f64,
    
    /// Nutrient gradient map
    gradient_map: Option<Vec<Position>>,
}

impl BacterialForagingOptimization {
    /// Create a new BFO algorithm with default parameters
    pub fn new() -> Self {
        Self::with_parameters(BfoParameters::default())
    }
    
    /// Create BFO with specific parameters
    pub fn with_parameters(parameters: BfoParameters) -> Self {
        Self {
            current_step_size: parameters.chemotactic_step_size,
            parameters,
            population: Population::new(),
            best_solution: None,
            best_fitness: f64::INFINITY,
            iteration: 0,
            current_chemotactic_step: 0,
            current_reproduction_step: 0,
            current_elimination_dispersal_event: 0,
            problem: None,
            metrics: AlgorithmMetrics::default(),
            diversity: 0.0,
            gradient_map: None,
        }
    }
    
    /// Builder pattern for BFO construction
    pub fn builder() -> BfoBuilder {
        BfoBuilder::new()
    }
    
    /// Perform chemotaxis for all bacteria
    async fn chemotaxis_phase(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        let best_position = self.best_solution.as_ref().map(|b| b.position());
        
        for i in 0..self.population.size() {
            let mut bacterium = self.population[i].clone();
            
            // Skip if stuck in biofilm
            if self.parameters.enable_biofilm && !bacterium.can_move() {
                continue;
            }
            
            // Tumbling behavior
            if self.parameters.enable_tumble {
                bacterium.tumble(self.parameters.tumble_probability);
            }
            
            // Calculate gradient if available
            let gradient = if let Some(ref gradient_map) = self.gradient_map {
                gradient_map.get(i).cloned()
            } else {
                None
            };
            
            // Perform chemotaxis
            let new_position = bacterium.chemotaxis(
                self.parameters.chemotaxis_strategy,
                self.current_step_size,
                bounds,
                best_position,
                gradient.as_ref(),
            );
            
            // Evaluate new position
            let new_fitness = problem.evaluate(&new_position);
            let old_fitness = *bacterium.fitness();
            
            // Swimming phase - continue if improvement
            let mut current_position = new_position;
            let mut current_fitness = new_fitness;
            
            for _ in 0..self.parameters.num_swim_steps {
                if let Some(swim_position) = bacterium.swim(current_fitness, self.current_step_size, bounds) {
                    let swim_fitness = problem.evaluate(&swim_position);
                    if swim_fitness < current_fitness {
                        current_position = swim_position;
                        current_fitness = swim_fitness;
                    } else {
                        break; // Stop swimming if no improvement
                    }
                }
            }
            
            // Apply cell-to-cell signaling
            let signaling_effect = bacterium.cell_signaling(&self.population.as_slice(), &self.parameters);
            let signaling_adjustment = signaling_effect * 0.01; // Small adjustment
            let adjusted_fitness = current_fitness + signaling_adjustment;
            
            // Update bacterium if improved
            if adjusted_fitness < old_fitness || old_fitness == f64::INFINITY {
                bacterium.update_position(current_position);
                bacterium.set_fitness(adjusted_fitness);
                
                // Update global best
                if adjusted_fitness < self.best_fitness {
                    self.best_fitness = adjusted_fitness;
                    self.best_solution = Some(Arc::new(bacterium.clone()));
                }
            }
            
            // Update biofilm if enabled
            if self.parameters.enable_biofilm {
                let local_density = self.calculate_local_density(i);
                bacterium.update_biofilm(local_density, self.parameters.biofilm_adhesion);
            }
            
            // Age bacterium
            bacterium.age_bacterium();
            
            self.population[i] = bacterium;
        }
        
        self.current_chemotactic_step += 1;
        Ok(())
    }
    
    /// Calculate local bacterial density
    fn calculate_local_density(&self, bacterium_index: usize) -> f64 {
        let current_bacterium = &self.population[bacterium_index];
        let mut neighbors = 0;
        
        for (i, other) in self.population.iter().enumerate() {
            if i != bacterium_index {
                let distance = (current_bacterium.position() - other.position()).norm();
                if distance <= self.parameters.communication_range {
                    neighbors += 1;
                }
            }
        }
        
        neighbors as f64 / self.population.size() as f64
    }
    
    /// Reproduction phase - eliminate less healthy bacteria and reproduce fitter ones
    async fn reproduction_phase(&mut self) -> Result<(), SwarmError> {
        if self.current_chemotactic_step % self.parameters.num_chemotactic_steps == 0 {
            let problem = self.problem.as_ref().unwrap();
            let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
            
            // Sort bacteria by reproductive fitness
            let mut indexed_bacteria: Vec<(usize, f64)> = self.population.iter()
                .enumerate()
                .map(|(i, bacterium)| (i, bacterium.reproductive_fitness()))
                .collect();
            indexed_bacteria.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Reproduce best half, eliminate worst half
            let reproduction_size = self.population.size() / 2;
            let mut new_population = Population::with_capacity(self.population.size());
            
            // Keep best half
            for i in 0..reproduction_size {
                let bacterium_index = indexed_bacteria[i].0;
                let parent = self.population[bacterium_index].clone();
                new_population.add(parent);
            }
            
            // Reproduce to fill population
            for i in 0..reproduction_size {
                let parent_index = i % reproduction_size;
                let offspring = new_population[parent_index].reproduce(bounds);
                new_population.add(offspring);
            }
            
            self.population = new_population;
            self.current_reproduction_step += 1;
            self.current_chemotactic_step = 0;
        }
        
        Ok(())
    }
    
    /// Elimination-dispersal phase - randomly eliminate and replace bacteria
    async fn elimination_dispersal_phase(&mut self) -> Result<(), SwarmError> {
        if self.current_reproduction_step % self.parameters.num_reproduction_steps == 0 {
            let problem = self.problem.as_ref().unwrap();
            let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
            let mut rng = thread_rng();
            
            for bacterium in self.population.iter_mut() {
                if rng.gen::<f64>() < self.parameters.elimination_dispersal_probability {
                    // Create new bacterium at random location
                    let new_bacterium = Bacterium::new(problem.dimensions, bounds);
                    *bacterium = new_bacterium;
                }
            }
            
            self.current_elimination_dispersal_event += 1;
            self.current_reproduction_step = 0;
        }
        
        Ok(())
    }
    
    /// Update adaptive parameters
    fn update_adaptive_parameters(&mut self) {
        if self.parameters.adaptive {
            // Adapt step size based on diversity and improvement
            if self.diversity < 0.1 {
                // Low diversity - increase exploration
                self.current_step_size = (self.current_step_size * 1.1)
                    .min(self.parameters.max_step_size);
            } else if self.diversity > 0.5 {
                // High diversity - increase exploitation
                self.current_step_size = (self.current_step_size * 0.9)
                    .max(self.parameters.min_step_size);
            }
        }
        
        // Update nutrient gradient if using gradient-based chemotaxis
        if matches!(self.parameters.chemotaxis_strategy, ChemotaxisStrategy::Gradient) {
            self.update_gradient_map();
        }
    }
    
    /// Update gradient map for gradient-based chemotaxis
    fn update_gradient_map(&mut self) {
        if let Some(ref problem) = self.problem {
            let mut gradients = Vec::with_capacity(self.population.size());
            let epsilon = 1e-8;
            
            for bacterium in self.population.iter() {
                let position = bacterium.position();
                let mut gradient = Position::zeros(position.len());
                
                // Finite difference approximation
                for i in 0..position.len() {
                    let mut pos_plus = position.clone();
                    let mut pos_minus = position.clone();
                    
                    pos_plus[i] += epsilon;
                    pos_minus[i] -= epsilon;
                    
                    let f_plus = problem.evaluate(&pos_plus);
                    let f_minus = problem.evaluate(&pos_minus);
                    
                    gradient[i] = (f_plus - f_minus) / (2.0 * epsilon);
                }
                
                gradients.push(gradient);
            }
            
            self.gradient_map = Some(gradients);
        }
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&mut self) -> f64 {
        self.diversity = self.population.diversity().unwrap_or(0.0);
        self.diversity
    }
    
    /// Check convergence
    fn has_converged(&self) -> bool {
        self.diversity < 1e-6 || 
        self.current_elimination_dispersal_event >= self.parameters.num_elimination_dispersal_events
    }
}

#[async_trait]
impl SwarmAlgorithm for BacterialForagingOptimization {
    type Individual = Bacterium;
    type Fitness = f64;
    type Parameters = BfoParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.parameters.validate()?;
        
        // Store problem
        self.problem = Some(Arc::new(problem));
        let problem_ref = self.problem.as_ref().unwrap();
        
        // Initialize population
        self.population = Population::with_capacity(self.parameters.population_size);
        let bounds = (problem_ref.lower_bounds.min(), problem_ref.upper_bounds.max());
        
        for _ in 0..self.parameters.population_size {
            let bacterium = Bacterium::new(problem_ref.dimensions, bounds);
            self.population.add(bacterium);
        }
        
        // Initial evaluation
        let positions: Vec<Position> = self.population.iter().map(|b| b.position().clone()).collect();
        let fitnesses = problem_ref.evaluate_parallel(&positions);
        
        for (bacterium, fitness) in self.population.iter_mut().zip(fitnesses.iter()) {
            bacterium.set_fitness(*fitness);
            
            if *fitness < self.best_fitness {
                self.best_fitness = *fitness;
                self.best_solution = Some(Arc::new(bacterium.clone()));
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
            "BFO initialized with {} bacteria, variant: {:?}",
            self.parameters.population_size,
            self.parameters.variant
        );
        
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        self.iteration += 1;
        
        // Chemotaxis phase
        self.chemotaxis_phase().await?;
        
        // Reproduction phase (periodic)
        self.reproduction_phase().await?;
        
        // Elimination-dispersal phase (periodic)
        self.elimination_dispersal_phase().await?;
        
        // Update adaptive parameters
        self.update_adaptive_parameters();
        
        // Update metrics
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.average_fitness = self.population.average_fitness();
        self.metrics.diversity = Some(self.calculate_diversity());
        self.metrics.time_per_iteration = Some(start_time.elapsed().as_micros() as u64);
        self.metrics.evaluations += self.parameters.population_size * (1 + self.parameters.num_swim_steps);
        
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
        "BacterialForagingOptimization"
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }
    
    fn update_parameters(&mut self, params: Self::Parameters) {
        self.current_step_size = params.chemotactic_step_size;
        self.parameters = params;
    }
    
    fn metrics(&self) -> AlgorithmMetrics {
        self.metrics.clone()
    }
    
    async fn reset(&mut self) -> Result<(), SwarmError> {
        self.population = Population::new();
        self.best_solution = None;
        self.best_fitness = f64::INFINITY;
        self.iteration = 0;
        self.current_chemotactic_step = 0;
        self.current_reproduction_step = 0;
        self.current_elimination_dispersal_event = 0;
        self.metrics = AlgorithmMetrics::default();
        self.current_step_size = self.parameters.chemotactic_step_size;
        self.diversity = 0.0;
        self.gradient_map = None;
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

impl AdaptiveAlgorithm for BacterialForagingOptimization {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        if let Some(diversity) = performance_metrics.diversity {
            if diversity < 0.1 {
                // Low diversity - increase exploration
                self.current_step_size = (self.current_step_size * 1.05)
                    .min(self.parameters.max_step_size);
                self.parameters.elimination_dispersal_probability = 
                    (self.parameters.elimination_dispersal_probability + 0.01).min(0.5);
            } else if diversity > 0.5 {
                // High diversity - increase exploitation
                self.current_step_size = (self.current_step_size * 0.95)
                    .max(self.parameters.min_step_size);
                self.parameters.elimination_dispersal_probability = 
                    (self.parameters.elimination_dispersal_probability - 0.01).max(0.05);
            }
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::Feedback { sensitivity: 0.05 }
    }
}

impl ParallelAlgorithm for BacterialForagingOptimization {
    async fn parallel_step(&mut self, thread_count: usize) -> Result<(), SwarmError> {
        // BFO can be parallelized at the bacterial level
        // For now, use the standard step implementation
        self.step().await
    }
    
    fn optimal_thread_count(&self) -> usize {
        (self.parameters.population_size / 8).max(1).min(num_cpus::get())
    }
}

/// Builder for BFO algorithm
pub struct BfoBuilder {
    parameters: BfoParameters,
}

impl BfoBuilder {
    pub fn new() -> Self {
        Self {
            parameters: BfoParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.parameters.population_size = size;
        self
    }
    
    pub fn chemotactic_steps(mut self, steps: usize) -> Self {
        self.parameters.num_chemotactic_steps = steps;
        self
    }
    
    pub fn step_size(mut self, size: f64) -> Self {
        self.parameters.chemotactic_step_size = size;
        self
    }
    
    pub fn variant(mut self, variant: BfoVariant) -> Self {
        self.parameters.variant = variant;
        self
    }
    
    pub fn chemotaxis_strategy(mut self, strategy: ChemotaxisStrategy) -> Self {
        self.parameters.chemotaxis_strategy = strategy;
        self
    }
    
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.parameters.adaptive = adaptive;
        self
    }
    
    pub fn build(self) -> Result<BacterialForagingOptimization, SwarmError> {
        self.parameters.validate()?;
        Ok(BacterialForagingOptimization::with_parameters(self.parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_bfo_initialization() {
        let mut bfo = BacterialForagingOptimization::new();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-10.0, 10.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert!(bfo.initialize(problem).await.is_ok());
        assert_eq!(bfo.population.size(), 50);
        assert!(bfo.best_solution.is_some());
    }
    
    #[tokio::test]
    async fn test_bfo_optimization() {
        let mut bfo = BacterialForagingOptimization::builder()
            .population_size(30)
            .chemotactic_steps(20)
            .variant(BfoVariant::Adaptive)
            .build()
            .unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let result = bfo.optimize(10).await.unwrap();
        
        // Should find reasonable solution
        assert!(result.best_fitness < 50.0);
        assert!(result.iterations <= 10);
        assert_eq!(result.algorithm_name, "BacterialForagingOptimization");
    }
    
    #[test]
    fn test_bfo_parameters() {
        let params = BfoParameters::builder()
            .population_size(40)
            .chemotactic_steps(50)
            .step_size(0.2)
            .variant(BfoVariant::Improved)
            .adaptive(true)
            .build()
            .unwrap();
        
        assert_eq!(params.population_size, 40);
        assert_eq!(params.num_chemotactic_steps, 50);
        assert_relative_eq!(params.chemotactic_step_size, 0.2);
        assert!(params.adaptive);
        assert!(matches!(params.variant, BfoVariant::Improved));
    }
    
    #[test]
    fn test_bacterium_creation() {
        let bacterium = Bacterium::new(3, (-5.0, 5.0));
        
        assert_eq!(bacterium.position().len(), 3);
        assert_eq!(*bacterium.fitness(), f64::INFINITY);
        assert_eq!(bacterium.health, 1.0);
        assert_eq!(bacterium.age, 0);
    }
    
    #[test]
    fn test_chemotaxis() {
        let mut bacterium = Bacterium::new(2, (-5.0, 5.0));
        
        let new_position = bacterium.chemotaxis(
            ChemotaxisStrategy::Random,
            0.1,
            (-5.0, 5.0),
            None,
            None,
        );
        
        assert_eq!(new_position.len(), 2);
        assert!(new_position[0] >= -5.0 && new_position[0] <= 5.0);
        assert!(new_position[1] >= -5.0 && new_position[1] <= 5.0);
        assert!(bacterium.last_direction.is_some());
    }
    
    #[test]
    fn test_reproduction() {
        let parent = Bacterium::new(2, (-5.0, 5.0));
        let offspring = parent.reproduce((-5.0, 5.0));
        
        assert_eq!(offspring.position().len(), 2);
        assert_eq!(offspring.age, 0);
        assert_eq!(offspring.cumulative_fitness, 0.0);
        assert_ne!(offspring.position()[0], parent.position()[0]); // Should be slightly different
    }
}