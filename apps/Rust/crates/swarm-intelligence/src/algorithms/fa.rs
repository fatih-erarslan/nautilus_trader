//! Firefly Algorithm (FA) optimization implementation
//!
//! FA is inspired by the flashing behavior of fireflies. Fireflies are attracted
//! to other fireflies with higher brightness (better fitness), creating a
//! natural optimization mechanism through bioluminescent communication.

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

/// FA algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum FaVariant {
    /// Standard Firefly Algorithm
    Standard,
    /// Discrete Firefly Algorithm (DFA)
    Discrete,
    /// Lévy-Flight Firefly Algorithm (LFA)
    LevyFlight,
    /// Chaotic Firefly Algorithm (CFA)
    Chaotic,
    /// Quantum Firefly Algorithm (QFA)
    Quantum,
    /// Hybrid Firefly Algorithm (HFA)
    Hybrid,
}

/// Light absorption models
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LightAbsorption {
    /// Exponential decay model
    Exponential,
    /// Gaussian model
    Gaussian,
    /// Inverse square law
    InverseSquare,
    /// Linear decay
    Linear,
    /// Sigmoid model
    Sigmoid,
}

/// Movement strategies for fireflies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MovementStrategy {
    /// Standard attraction-based movement
    Standard,
    /// Lévy flight enhanced movement
    LevyFlight,
    /// Random walk fallback
    RandomWalk,
    /// Best-guided movement
    BestGuided,
    /// Opposition-based movement
    OppositionBased,
}

/// FA algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaParameters {
    /// Population size (number of fireflies)
    pub population_size: usize,
    
    /// Light absorption coefficient (gamma)
    pub absorption_coefficient: f64,
    
    /// Initial attractiveness (beta_0)
    pub initial_attractiveness: f64,
    
    /// Randomization parameter (alpha)
    pub randomization_parameter: f64,
    
    /// Minimum randomization
    pub min_randomization: f64,
    
    /// Maximum randomization
    pub max_randomization: f64,
    
    /// FA variant to use
    pub variant: FaVariant,
    
    /// Light absorption model
    pub absorption_model: LightAbsorption,
    
    /// Movement strategy
    pub movement_strategy: MovementStrategy,
    
    /// Enable adaptive parameters
    pub adaptive: bool,
    
    /// Lévy flight lambda parameter
    pub levy_lambda: f64,
    
    /// Communication range for fireflies
    pub communication_range: f64,
    
    /// Enable flickering behavior
    pub enable_flickering: bool,
    
    /// Flickering probability
    pub flickering_probability: f64,
    
    /// Flash intensity scaling factor
    pub flash_intensity_scaling: f64,
    
    /// Minimum brightness threshold
    pub min_brightness_threshold: f64,
    
    /// Enable swarming behavior
    pub enable_swarming: bool,
    
    /// Swarm cohesion factor
    pub swarm_cohesion: f64,
    
    /// Enable opposition-based learning
    pub opposition_based: bool,
    
    /// Opposition probability
    pub opposition_probability: f64,
    
    /// Chaotic map parameter (for chaotic variant)
    pub chaotic_parameter: f64,
    
    /// Quantum rotation angle (for quantum variant)
    pub quantum_rotation_angle: f64,
    
    /// Temperature for thermal effects
    pub temperature: f64,
    
    /// Humidity factor
    pub humidity: f64,
    
    /// Night time factor (affects visibility)
    pub night_time_factor: f64,
}

impl Default for FaParameters {
    fn default() -> Self {
        Self {
            population_size: 30,
            absorption_coefficient: 1.0,
            initial_attractiveness: 1.0,
            randomization_parameter: 0.2,
            min_randomization: 0.1,
            max_randomization: 0.5,
            variant: FaVariant::Standard,
            absorption_model: LightAbsorption::Exponential,
            movement_strategy: MovementStrategy::Standard,
            adaptive: false,
            levy_lambda: 1.5,
            communication_range: 1.0,
            enable_flickering: false,
            flickering_probability: 0.1,
            flash_intensity_scaling: 1.0,
            min_brightness_threshold: 0.01,
            enable_swarming: false,
            swarm_cohesion: 0.1,
            opposition_based: false,
            opposition_probability: 0.1,
            chaotic_parameter: 4.0,
            quantum_rotation_angle: 0.05,
            temperature: 20.0, // Celsius
            humidity: 0.6,     // 60%
            night_time_factor: 1.0, // Full night
        }
    }
}

impl FaParameters {
    /// Validate parameters
    pub fn validate(&self) -> Result<(), SwarmError> {
        validate_parameter!(self.population_size, "population_size", 4, 1000);
        validate_parameter!(self.absorption_coefficient, "absorption_coefficient", 0.01, 10.0);
        validate_parameter!(self.initial_attractiveness, "initial_attractiveness", 0.1, 10.0);
        validate_parameter!(self.randomization_parameter, "randomization_parameter", 0.0, 1.0);
        validate_parameter!(self.levy_lambda, "levy_lambda", 0.3, 2.0);
        validate_parameter!(self.communication_range, "communication_range", 0.1, 10.0);
        validate_parameter!(self.temperature, "temperature", -40.0, 50.0);
        validate_parameter!(self.humidity, "humidity", 0.0, 1.0);
        
        if self.min_randomization >= self.max_randomization {
            return Err(SwarmError::parameter("min_randomization must be < max_randomization"));
        }
        
        Ok(())
    }
    
    /// Create builder for FA parameters
    pub fn builder() -> FaParametersBuilder {
        FaParametersBuilder::new()
    }
}

/// Builder for FA parameters
pub struct FaParametersBuilder {
    params: FaParameters,
}

impl FaParametersBuilder {
    pub fn new() -> Self {
        Self {
            params: FaParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.params.population_size = size;
        self
    }
    
    pub fn absorption_coefficient(mut self, gamma: f64) -> Self {
        self.params.absorption_coefficient = gamma;
        self
    }
    
    pub fn initial_attractiveness(mut self, beta_0: f64) -> Self {
        self.params.initial_attractiveness = beta_0;
        self
    }
    
    pub fn randomization_parameter(mut self, alpha: f64) -> Self {
        self.params.randomization_parameter = alpha;
        self
    }
    
    pub fn variant(mut self, variant: FaVariant) -> Self {
        self.params.variant = variant;
        self
    }
    
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.params.adaptive = adaptive;
        self
    }
    
    pub fn build(self) -> Result<FaParameters, SwarmError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

/// Firefly representation
#[derive(Debug, Clone)]
pub struct Firefly {
    /// Current position
    position: Position,
    
    /// Fitness value (brightness)
    fitness: f64,
    
    /// Light intensity
    light_intensity: f64,
    
    /// Flash frequency
    flash_frequency: f64,
    
    /// Last flash time
    last_flash_time: usize,
    
    /// Attraction history
    attraction_history: Vec<usize>,
    
    /// Movement velocity
    velocity: Position,
    
    /// Flickering state
    is_flickering: bool,
    
    /// Communication state with other fireflies
    communication_state: Vec<f64>,
    
    /// Age of the firefly
    age: usize,
    
    /// Energy level (affects flash intensity)
    energy_level: f64,
    
    /// Mating readiness (for reproduction)
    mating_readiness: f64,
    
    /// Territory center
    territory_center: Option<Position>,
    
    /// Quantum state (for quantum variant)
    quantum_state: Option<Vec<f64>>,
    
    /// Chaotic sequence value
    chaotic_value: f64,
}

impl Firefly {
    /// Create a new firefly at random position
    pub fn new(dimensions: usize, bounds: (f64, f64)) -> Self {
        let mut rng = thread_rng();
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(bounds.0..=bounds.1)
        });
        
        let velocity = Position::zeros(dimensions);
        let communication_state = vec![0.0; dimensions];
        
        Self {
            position,
            fitness: f64::INFINITY,
            light_intensity: 1.0,
            flash_frequency: rng.gen_range(0.5..2.0),
            last_flash_time: 0,
            attraction_history: Vec::new(),
            velocity,
            is_flickering: false,
            communication_state,
            age: 0,
            energy_level: 1.0,
            mating_readiness: rng.gen(),
            territory_center: None,
            quantum_state: None,
            chaotic_value: rng.gen(),
        }
    }
    
    /// Calculate attractiveness based on distance and parameters
    pub fn attractiveness(
        &self,
        other: &Firefly,
        absorption_coefficient: f64,
        initial_attractiveness: f64,
        absorption_model: LightAbsorption,
    ) -> f64 {
        let distance = (&self.position - other.position()).norm();
        
        match absorption_model {
            LightAbsorption::Exponential => {
                initial_attractiveness * (-absorption_coefficient * distance.powi(2)).exp()
            },
            LightAbsorption::Gaussian => {
                initial_attractiveness * (-(distance.powi(2)) / (2.0 * absorption_coefficient.powi(2))).exp()
            },
            LightAbsorption::InverseSquare => {
                initial_attractiveness / (1.0 + absorption_coefficient * distance.powi(2))
            },
            LightAbsorption::Linear => {
                (initial_attractiveness - absorption_coefficient * distance).max(0.0)
            },
            LightAbsorption::Sigmoid => {
                initial_attractiveness / (1.0 + (absorption_coefficient * distance).exp())
            },
        }
    }
    
    /// Move towards another firefly
    pub fn move_towards(
        &mut self,
        target: &Firefly,
        attractiveness: f64,
        randomization: f64,
        strategy: MovementStrategy,
        bounds: (f64, f64),
        environmental_factors: (f64, f64, f64), // temperature, humidity, night_time
    ) -> Position {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        // Calculate base movement
        let direction = target.position() - &self.position;
        let base_movement = attractiveness * direction;
        
        // Apply movement strategy
        let strategic_movement = match strategy {
            MovementStrategy::Standard => base_movement,
            MovementStrategy::LevyFlight => {
                let levy_step = self.generate_levy_flight_step(1.5, dimensions);
                base_movement + 0.3 * levy_step
            },
            MovementStrategy::RandomWalk => {
                let random_step = Position::from_fn(dimensions, |_, _| {
                    rng.sample::<f64, _>(Normal::new(0.0, 0.1).unwrap())
                });
                base_movement + random_step
            },
            MovementStrategy::BestGuided => {
                // Enhanced movement towards target
                let enhanced_direction = direction * 1.5;
                base_movement + 0.2 * enhanced_direction
            },
            MovementStrategy::OppositionBased => {
                if rng.gen::<f64>() < 0.1 {
                    // Opposite movement
                    -base_movement
                } else {
                    base_movement
                }
            },
        };
        
        // Add randomization
        let random_component = Position::from_fn(dimensions, |_, _| {
            rng.sample::<f64, _>(Normal::new(0.0, randomization).unwrap())
        });
        
        // Apply environmental factors
        let (temperature, humidity, night_time) = environmental_factors;
        let temp_factor = 1.0 + (temperature - 20.0) / 100.0; // Optimal at 20°C
        let humidity_factor = 1.0 - (humidity - 0.6).abs(); // Optimal at 60%
        let visibility_factor = night_time; // Better visibility at night
        
        let environmental_scaling = temp_factor * humidity_factor * visibility_factor;
        
        // Calculate new position
        let movement = environmental_scaling * (strategic_movement + random_component);
        let new_position = &self.position + movement;
        
        // Apply bounds
        let bounded_position = Position::from_fn(dimensions, |i, _| {
            new_position[i].clamp(bounds.0, bounds.1)
        });
        
        // Update velocity for momentum
        self.velocity = movement * 0.9 + self.velocity * 0.1;
        
        // Record attraction
        if let Some(target_id) = self.attraction_history.last() {
            if *target_id != target.age {
                self.attraction_history.push(target.age);
                if self.attraction_history.len() > 10 {
                    self.attraction_history.remove(0);
                }
            }
        } else {
            self.attraction_history.push(target.age);
        }
        
        bounded_position
    }
    
    /// Generate Lévy flight step
    fn generate_levy_flight_step(&self, lambda: f64, dimensions: usize) -> Position {
        let mut rng = thread_rng();
        
        Position::from_fn(dimensions, |_, _| {
            let u: f64 = rng.sample::<f64, _>(Normal::new(0.0, 1.0).unwrap());
            u.signum() * u.abs().powf(1.0 / lambda)
        })
    }
    
    /// Calculate brightness based on fitness
    pub fn calculate_brightness(&mut self, fitness: f64) {
        self.fitness = fitness;
        
        // Brightness is inversely related to fitness (for minimization)
        self.light_intensity = if fitness.is_finite() && fitness > 0.0 {
            self.energy_level / (1.0 + fitness)
        } else if fitness == 0.0 {
            self.energy_level * 1000.0 // Very bright for perfect solution
        } else {
            0.0 // No light for invalid solutions
        };
        
        // Apply flash intensity scaling
        self.light_intensity *= self.energy_level;
    }
    
    /// Flash behavior
    pub fn flash(&mut self, current_time: usize) -> bool {
        if current_time - self.last_flash_time >= (1.0 / self.flash_frequency) as usize {
            self.last_flash_time = current_time;
            self.energy_level = (self.energy_level - 0.01).max(0.1); // Flashing costs energy
            true
        } else {
            false
        }
    }
    
    /// Flickering behavior
    pub fn flicker(&mut self, probability: f64) -> bool {
        let mut rng = thread_rng();
        if rng.gen::<f64>() < probability {
            self.is_flickering = !self.is_flickering;
            true
        } else {
            false
        }
    }
    
    /// Update energy level based on environment
    pub fn update_energy(&mut self, nutrition_level: f64, temperature: f64) {
        let nutrition_factor = nutrition_level.clamp(0.1, 2.0);
        let temp_factor = 1.0 - (temperature - 20.0).abs() / 50.0; // Optimal at 20°C
        
        self.energy_level = (self.energy_level * nutrition_factor * temp_factor.max(0.1))
            .clamp(0.1, 2.0);
        
        // Regenerate some energy over time
        self.energy_level = (self.energy_level + 0.001).min(2.0);
    }
    
    /// Initialize quantum state
    pub fn initialize_quantum_state(&mut self, dimensions: usize) {
        let mut rng = thread_rng();
        self.quantum_state = Some((0..dimensions).map(|_| {
            rng.gen_range(0.0..2.0 * std::f64::consts::PI)
        }).collect());
    }
    
    /// Apply quantum rotation
    pub fn apply_quantum_rotation(&mut self, angle: f64, bounds: (f64, f64)) {
        if let Some(ref mut quantum_state) = self.quantum_state {
            for (i, state) in quantum_state.iter_mut().enumerate() {
                *state += angle;
                
                // Update position based on quantum state
                let new_value = bounds.0 + (bounds.1 - bounds.0) * 
                               (0.5 + 0.5 * state.cos());
                
                self.position[i] = new_value.clamp(bounds.0, bounds.1);
            }
        }
    }
    
    /// Update chaotic value
    pub fn update_chaotic_value(&mut self, parameter: f64) {
        self.chaotic_value = parameter * self.chaotic_value * (1.0 - self.chaotic_value);
    }
    
    /// Calculate territory attractiveness
    pub fn territory_attractiveness(&self, position: &Position) -> f64 {
        if let Some(ref center) = self.territory_center {
            let distance = (position - center).norm();
            1.0 / (1.0 + distance)
        } else {
            0.0
        }
    }
    
    /// Establish territory
    pub fn establish_territory(&mut self) {
        self.territory_center = Some(self.position.clone());
    }
    
    /// Check if can reproduce
    pub fn can_reproduce(&self, other: &Firefly) -> bool {
        self.mating_readiness > 0.8 && 
        other.mating_readiness > 0.8 &&
        self.energy_level > 0.5 &&
        other.energy_level > 0.5
    }
    
    /// Reproduce with another firefly
    pub fn reproduce(&self, other: &Firefly, bounds: (f64, f64)) -> Firefly {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        // Crossover positions
        let offspring_position = Position::from_fn(dimensions, |i, _| {
            if rng.gen::<f64>() < 0.5 {
                self.position[i]
            } else {
                other.position()[i]
            }
        });
        
        let mut offspring = Firefly::new(dimensions, bounds);
        offspring.update_position(offspring_position);
        
        // Inherit traits
        offspring.flash_frequency = (self.flash_frequency + other.flash_frequency) / 2.0;
        offspring.energy_level = (self.energy_level + other.energy_level) / 2.0;
        
        offspring
    }
    
    /// Age the firefly
    pub fn age_firefly(&mut self) {
        self.age += 1;
        
        // Natural aging effects
        if self.age > 1000 {
            self.energy_level = (self.energy_level * 0.999).max(0.1);
            self.flash_frequency = (self.flash_frequency * 0.999).max(0.1);
        }
    }
    
    /// Get visible fireflies within communication range
    pub fn get_visible_fireflies<'a>(
        &self,
        fireflies: &'a [Firefly],
        communication_range: f64,
        night_time_factor: f64,
    ) -> Vec<&'a Firefly> {
        fireflies.iter()
            .filter(|other| {
                let distance = (&self.position - other.position()).norm();
                distance <= communication_range * night_time_factor &&
                other.light_intensity > 0.01 &&
                !other.is_flickering
            })
            .collect()
    }
}

impl Individual for Firefly {
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
        self.calculate_brightness(fitness);
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// Firefly Algorithm
#[derive(Debug, Clone)]
pub struct FireflyAlgorithm {
    /// Algorithm parameters
    parameters: FaParameters,
    
    /// Population of fireflies
    population: Population<Firefly>,
    
    /// Best solution found
    best_solution: Option<Arc<Firefly>>,
    
    /// Best fitness
    best_fitness: f64,
    
    /// Current iteration
    iteration: usize,
    
    /// Optimization problem
    problem: Option<Arc<OptimizationProblem>>,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
    
    /// Current randomization parameter (adaptive)
    current_randomization: f64,
    
    /// Population diversity
    diversity: f64,
    
    /// Environmental conditions
    environmental_conditions: (f64, f64, f64), // temperature, humidity, night_time
}

impl FireflyAlgorithm {
    /// Create a new FA algorithm with default parameters
    pub fn new() -> Self {
        Self::with_parameters(FaParameters::default())
    }
    
    /// Create FA with specific parameters
    pub fn with_parameters(parameters: FaParameters) -> Self {
        Self {
            current_randomization: parameters.randomization_parameter,
            environmental_conditions: (parameters.temperature, parameters.humidity, parameters.night_time_factor),
            parameters,
            population: Population::new(),
            best_solution: None,
            best_fitness: f64::INFINITY,
            iteration: 0,
            problem: None,
            metrics: AlgorithmMetrics::default(),
            diversity: 0.0,
        }
    }
    
    /// Builder pattern for FA construction
    pub fn builder() -> FaBuilder {
        FaBuilder::new()
    }
    
    /// Perform firefly interactions and movements
    async fn firefly_interactions(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        
        let population_size = self.population.size();
        let mut new_positions = vec![None; population_size];
        
        // Process firefly interactions in parallel
        for i in 0..population_size {
            let current_firefly = &self.population[i];
            let mut best_attractiveness = 0.0;
            let mut best_target_idx = None;
            
            // Find most attractive firefly
            for j in 0..population_size {
                if i != j {
                    let other_firefly = &self.population[j];
                    
                    // Only move towards brighter fireflies
                    if other_firefly.light_intensity > current_firefly.light_intensity {
                        let attractiveness = current_firefly.attractiveness(
                            other_firefly,
                            self.parameters.absorption_coefficient,
                            self.parameters.initial_attractiveness,
                            self.parameters.absorption_model,
                        );
                        
                        if attractiveness > best_attractiveness {
                            best_attractiveness = attractiveness;
                            best_target_idx = Some(j);
                        }
                    }
                }
            }
            
            // Move towards best target or random walk
            let new_position = if let Some(target_idx) = best_target_idx {
                let target = &self.population[target_idx];
                let mut firefly_copy = current_firefly.clone();
                
                firefly_copy.move_towards(
                    target,
                    best_attractiveness,
                    self.current_randomization,
                    self.parameters.movement_strategy,
                    bounds,
                    self.environmental_conditions,
                )
            } else {
                // Random movement for least attractive fireflies
                self.random_movement(current_firefly, bounds)
            };
            
            new_positions[i] = Some(new_position);
        }
        
        // Update positions and evaluate
        for (i, new_pos) in new_positions.into_iter().enumerate() {
            if let Some(position) = new_pos {
                self.population[i].update_position(position);
                
                let fitness = problem.evaluate(self.population[i].position());
                self.population[i].set_fitness(fitness);
                
                // Update global best
                if fitness < self.best_fitness {
                    self.best_fitness = fitness;
                    self.best_solution = Some(Arc::new(self.population[i].clone()));
                }
            }
        }
        
        Ok(())
    }
    
    /// Random movement for fireflies with no attractive targets
    fn random_movement(&self, firefly: &Firefly, bounds: (f64, f64)) -> Position {
        let mut rng = thread_rng();
        let dimensions = firefly.position().len();
        
        let random_step = Position::from_fn(dimensions, |_, _| {
            rng.sample::<f64, _>(Normal::new(0.0, self.current_randomization).unwrap())
        });
        
        let new_position = firefly.position() + random_step;
        
        Position::from_fn(dimensions, |i, _| {
            new_position[i].clamp(bounds.0, bounds.1)
        })
    }
    
    /// Apply variant-specific behaviors
    async fn apply_variant_behaviors(&mut self) -> Result<(), SwarmError> {
        let bounds = self.problem.as_ref().unwrap().lower_bounds.min();
        let upper_bounds = self.problem.as_ref().unwrap().upper_bounds.max();
        let bounds_tuple = (bounds, upper_bounds);
        
        match self.parameters.variant {
            FaVariant::Standard => {
                // Standard FA - no additional behaviors
            },
            FaVariant::Discrete => {
                // Discretize positions for discrete problems
                for firefly in self.population.iter_mut() {
                    let position = firefly.position_mut();
                    for value in position.iter_mut() {
                        *value = value.round();
                    }
                }
            },
            FaVariant::LevyFlight => {
                // Apply Lévy flight to some fireflies
                let mut rng = thread_rng();
                for firefly in self.population.iter_mut() {
                    if rng.gen::<f64>() < 0.1 {
                        let levy_step = firefly.generate_levy_flight_step(
                            self.parameters.levy_lambda,
                            firefly.position().len()
                        );
                        let new_position = firefly.position() + 0.1 * levy_step;
                        let bounded_position = Position::from_fn(new_position.len(), |i, _| {
                            new_position[i].clamp(bounds_tuple.0, bounds_tuple.1)
                        });
                        firefly.update_position(bounded_position);
                    }
                }
            },
            FaVariant::Chaotic => {
                // Update chaotic values and apply chaotic movements
                for firefly in self.population.iter_mut() {
                    firefly.update_chaotic_value(self.parameters.chaotic_parameter);
                    
                    // Apply chaotic perturbation
                    let chaotic_factor = firefly.chaotic_value;
                    let perturbation = Position::from_fn(firefly.position().len(), |i, _| {
                        (chaotic_factor - 0.5) * 0.1
                    });
                    
                    let new_position = firefly.position() + perturbation;
                    let bounded_position = Position::from_fn(new_position.len(), |i, _| {
                        new_position[i].clamp(bounds_tuple.0, bounds_tuple.1)
                    });
                    firefly.update_position(bounded_position);
                }
            },
            FaVariant::Quantum => {
                // Apply quantum rotations
                for firefly in self.population.iter_mut() {
                    if firefly.quantum_state.is_none() {
                        firefly.initialize_quantum_state(firefly.position().len());
                    }
                    firefly.apply_quantum_rotation(
                        self.parameters.quantum_rotation_angle,
                        bounds_tuple
                    );
                }
            },
            FaVariant::Hybrid => {
                // Combine multiple strategies
                let mut rng = thread_rng();
                for firefly in self.population.iter_mut() {
                    match rng.gen_range(0..3) {
                        0 => {
                            // Lévy flight
                            let levy_step = firefly.generate_levy_flight_step(1.5, firefly.position().len());
                            let new_position = firefly.position() + 0.05 * levy_step;
                            let bounded_position = Position::from_fn(new_position.len(), |i, _| {
                                new_position[i].clamp(bounds_tuple.0, bounds_tuple.1)
                            });
                            firefly.update_position(bounded_position);
                        },
                        1 => {
                            // Chaotic movement
                            firefly.update_chaotic_value(4.0);
                            let chaotic_perturbation = (firefly.chaotic_value - 0.5) * 0.05;
                            let perturbation = Position::from_fn(firefly.position().len(), |_, _| chaotic_perturbation);
                            let new_position = firefly.position() + perturbation;
                            let bounded_position = Position::from_fn(new_position.len(), |i, _| {
                                new_position[i].clamp(bounds_tuple.0, bounds_tuple.1)
                            });
                            firefly.update_position(bounded_position);
                        },
                        _ => {
                            // Standard movement (already handled)
                        }
                    }
                }
            },
        }
        
        Ok(())
    }
    
    /// Handle flickering behavior
    fn handle_flickering(&mut self) {
        if self.parameters.enable_flickering {
            for firefly in self.population.iter_mut() {
                firefly.flicker(self.parameters.flickering_probability);
            }
        }
    }
    
    /// Update environmental conditions
    fn update_environment(&mut self) {
        // Simulate natural environmental changes
        let (mut temp, mut humidity, mut night_time) = self.environmental_conditions;
        
        // Temperature variation (day/night cycle)
        temp += (self.iteration as f64 * 0.1).sin() * 5.0;
        
        // Humidity changes
        humidity += (self.iteration as f64 * 0.05).cos() * 0.1;
        humidity = humidity.clamp(0.0, 1.0);
        
        // Night time factor (fireflies more active at night)
        night_time = (0.8 + 0.2 * (self.iteration as f64 * 0.2).sin()).max(0.5);
        
        self.environmental_conditions = (temp, humidity, night_time);
        
        // Update firefly energy based on environment
        for firefly in self.population.iter_mut() {
            let nutrition_level = 1.0 - self.diversity * 0.5; // Less nutrition in crowded areas
            firefly.update_energy(nutrition_level, temp);
        }
    }
    
    /// Update adaptive parameters
    fn update_adaptive_parameters(&mut self) {
        if self.parameters.adaptive {
            // Adapt randomization based on diversity
            if self.diversity < 0.1 {
                // Low diversity - increase exploration
                self.current_randomization = (self.current_randomization * 1.1)
                    .min(self.parameters.max_randomization);
            } else if self.diversity > 0.5 {
                // High diversity - increase exploitation
                self.current_randomization = (self.current_randomization * 0.9)
                    .max(self.parameters.min_randomization);
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
impl SwarmAlgorithm for FireflyAlgorithm {
    type Individual = Firefly;
    type Fitness = f64;
    type Parameters = FaParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.parameters.validate()?;
        
        // Store problem
        self.problem = Some(Arc::new(problem));
        let problem_ref = self.problem.as_ref().unwrap();
        
        // Initialize population
        self.population = Population::with_capacity(self.parameters.population_size);
        let bounds = (problem_ref.lower_bounds.min(), problem_ref.upper_bounds.max());
        
        for _ in 0..self.parameters.population_size {
            let mut firefly = Firefly::new(problem_ref.dimensions, bounds);
            
            // Initialize quantum state if using quantum variant
            if matches!(self.parameters.variant, FaVariant::Quantum) {
                firefly.initialize_quantum_state(problem_ref.dimensions);
            }
            
            self.population.add(firefly);
        }
        
        // Initial evaluation
        let positions: Vec<Position> = self.population.iter().map(|f| f.position().clone()).collect();
        let fitnesses = problem_ref.evaluate_parallel(&positions);
        
        for (firefly, fitness) in self.population.iter_mut().zip(fitnesses.iter()) {
            firefly.set_fitness(*fitness);
            
            if *fitness < self.best_fitness {
                self.best_fitness = *fitness;
                self.best_solution = Some(Arc::new(firefly.clone()));
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
            "FA initialized with {} fireflies, variant: {:?}",
            self.parameters.population_size,
            self.parameters.variant
        );
        
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        self.iteration += 1;
        
        // Handle flickering behavior
        self.handle_flickering();
        
        // Update environmental conditions
        self.update_environment();
        
        // Perform firefly interactions and movements
        self.firefly_interactions().await?;
        
        // Apply variant-specific behaviors
        self.apply_variant_behaviors().await?;
        
        // Age fireflies
        for firefly in self.population.iter_mut() {
            firefly.age_firefly();
        }
        
        // Update adaptive parameters
        self.update_adaptive_parameters();
        
        // Update metrics
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.average_fitness = self.population.average_fitness();
        self.metrics.diversity = Some(self.calculate_diversity());
        self.metrics.time_per_iteration = Some(start_time.elapsed().as_micros() as u64);
        self.metrics.evaluations += self.parameters.population_size;
        
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
        "FireflyAlgorithm"
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }
    
    fn update_parameters(&mut self, params: Self::Parameters) {
        self.current_randomization = params.randomization_parameter;
        self.environmental_conditions = (params.temperature, params.humidity, params.night_time_factor);
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
        self.metrics = AlgorithmMetrics::default();
        self.current_randomization = self.parameters.randomization_parameter;
        self.diversity = 0.0;
        self.environmental_conditions = (self.parameters.temperature, self.parameters.humidity, self.parameters.night_time_factor);
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

impl AdaptiveAlgorithm for FireflyAlgorithm {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        if let Some(diversity) = performance_metrics.diversity {
            if diversity < 0.1 {
                // Low diversity - increase exploration
                self.current_randomization = (self.current_randomization * 1.05)
                    .min(self.parameters.max_randomization);
                self.parameters.absorption_coefficient = 
                    (self.parameters.absorption_coefficient * 0.95).max(0.1);
            } else if diversity > 0.5 {
                // High diversity - increase exploitation
                self.current_randomization = (self.current_randomization * 0.95)
                    .max(self.parameters.min_randomization);
                self.parameters.absorption_coefficient = 
                    (self.parameters.absorption_coefficient * 1.05).min(5.0);
            }
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::Feedback { sensitivity: 0.05 }
    }
}

impl ParallelAlgorithm for FireflyAlgorithm {
    async fn parallel_step(&mut self, thread_count: usize) -> Result<(), SwarmError> {
        // FA can be parallelized at the firefly interaction level
        // For now, use the standard step implementation
        self.step().await
    }
    
    fn optimal_thread_count(&self) -> usize {
        (self.parameters.population_size / 6).max(1).min(num_cpus::get())
    }
}

/// Builder for FA algorithm
pub struct FaBuilder {
    parameters: FaParameters,
}

impl FaBuilder {
    pub fn new() -> Self {
        Self {
            parameters: FaParameters::default(),
        }
    }
    
    pub fn population_size(mut self, size: usize) -> Self {
        self.parameters.population_size = size;
        self
    }
    
    pub fn absorption_coefficient(mut self, gamma: f64) -> Self {
        self.parameters.absorption_coefficient = gamma;
        self
    }
    
    pub fn initial_attractiveness(mut self, beta_0: f64) -> Self {
        self.parameters.initial_attractiveness = beta_0;
        self
    }
    
    pub fn randomization_parameter(mut self, alpha: f64) -> Self {
        self.parameters.randomization_parameter = alpha;
        self
    }
    
    pub fn variant(mut self, variant: FaVariant) -> Self {
        self.parameters.variant = variant;
        self
    }
    
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.parameters.adaptive = adaptive;
        self
    }
    
    pub fn build(self) -> Result<FireflyAlgorithm, SwarmError> {
        self.parameters.validate()?;
        Ok(FireflyAlgorithm::with_parameters(self.parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_fa_initialization() {
        let mut fa = FireflyAlgorithm::new();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-10.0, 10.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert!(fa.initialize(problem).await.is_ok());
        assert_eq!(fa.population.size(), 30);
        assert!(fa.best_solution.is_some());
    }
    
    #[tokio::test]
    async fn test_fa_optimization() {
        let mut fa = FireflyAlgorithm::builder()
            .population_size(20)
            .absorption_coefficient(1.0)
            .variant(FaVariant::LevyFlight)
            .build()
            .unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let result = fa.optimize(30).await.unwrap();
        
        // Should find reasonable solution
        assert!(result.best_fitness < 25.0);
        assert!(result.iterations <= 30);
        assert_eq!(result.algorithm_name, "FireflyAlgorithm");
    }
    
    #[test]
    fn test_fa_parameters() {
        let params = FaParameters::builder()
            .population_size(40)
            .absorption_coefficient(1.5)
            .initial_attractiveness(2.0)
            .randomization_parameter(0.3)
            .variant(FaVariant::Quantum)
            .adaptive(true)
            .build()
            .unwrap();
        
        assert_eq!(params.population_size, 40);
        assert_relative_eq!(params.absorption_coefficient, 1.5);
        assert_relative_eq!(params.initial_attractiveness, 2.0);
        assert_relative_eq!(params.randomization_parameter, 0.3);
        assert!(params.adaptive);
        assert!(matches!(params.variant, FaVariant::Quantum));
    }
    
    #[test]
    fn test_firefly_creation() {
        let firefly = Firefly::new(3, (-5.0, 5.0));
        
        assert_eq!(firefly.position().len(), 3);
        assert_eq!(*firefly.fitness(), f64::INFINITY);
        assert_eq!(firefly.light_intensity, 1.0);
        assert_eq!(firefly.age, 0);
    }
    
    #[test]
    fn test_attractiveness_calculation() {
        let firefly1 = Firefly::new(2, (-5.0, 5.0));
        let mut firefly2 = Firefly::new(2, (-5.0, 5.0));
        
        // Set different positions
        firefly2.update_position(Position::from_vec(vec![1.0, 1.0]));
        
        let attractiveness = firefly1.attractiveness(
            &firefly2,
            1.0, // absorption_coefficient
            1.0, // initial_attractiveness
            LightAbsorption::Exponential,
        );
        
        assert!(attractiveness > 0.0);
        assert!(attractiveness <= 1.0);
    }
    
    #[test]
    fn test_brightness_calculation() {
        let mut firefly = Firefly::new(2, (-5.0, 5.0));
        
        firefly.calculate_brightness(10.0);
        let brightness1 = firefly.light_intensity;
        
        firefly.calculate_brightness(5.0);
        let brightness2 = firefly.light_intensity;
        
        // Better fitness should result in higher brightness
        assert!(brightness2 > brightness1);
    }
    
    #[test]
    fn test_movement() {
        let mut firefly1 = Firefly::new(2, (-5.0, 5.0));
        let mut firefly2 = Firefly::new(2, (-5.0, 5.0));
        
        firefly2.update_position(Position::from_vec(vec![2.0, 2.0]));
        firefly2.calculate_brightness(1.0); // Make it attractive
        
        let new_position = firefly1.move_towards(
            &firefly2,
            0.5, // attractiveness
            0.1, // randomization
            MovementStrategy::Standard,
            (-5.0, 5.0),
            (20.0, 0.6, 1.0), // environmental factors
        );
        
        // Should move towards the target
        let old_distance = (&firefly1.position() - firefly2.position()).norm();
        let new_distance = (&new_position - firefly2.position()).norm();
        assert!(new_distance <= old_distance + 0.5); // Allow for randomization
    }
    
    #[test]
    fn test_quantum_features() {
        let mut firefly = Firefly::new(3, (-5.0, 5.0));
        firefly.initialize_quantum_state(3);
        
        assert!(firefly.quantum_state.is_some());
        assert_eq!(firefly.quantum_state.as_ref().unwrap().len(), 3);
        
        firefly.apply_quantum_rotation(0.1, (-5.0, 5.0));
        // Position should be updated based on quantum state
    }
}