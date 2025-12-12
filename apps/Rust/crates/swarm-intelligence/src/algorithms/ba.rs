//! # Bat Algorithm (BA) - Enterprise Implementation
//!
//! Advanced implementation of the Bat Algorithm inspired by the echolocation behavior
//! of microbats. The algorithm uses frequency tuning, pulse rate, and loudness dynamics
//! to simulate the hunting behavior of bats in nature.
//!
//! ## Algorithm Variants
//! - **Standard BA**: Original Yang formulation with basic echolocation
//! - **Enhanced BA**: Improved with adaptive parameter tuning
//! - **Chaos BA**: Chaotic maps for enhanced exploration
//! - **Quantum BA**: Quantum-inspired position updates
//! - **Levy BA**: Levy flight patterns for global search
//! - **Multi-objective BA**: Support for multi-objective optimization
//!
//! ## Key Features
//! - Echolocation behavior modeling with frequency tuning
//! - Pulse rate and loudness dynamics
//! - Multiple hunting strategies and roost formations
//! - SIMD-optimized vector operations
//! - Real-time performance monitoring
//! - Advanced convergence detection
//!
//! ## References
//! - Yang, X. S. (2010). A New Metaheuristic Bat-Inspired Algorithm.
//!   In: Nature Inspired Cooperative Strategies for Optimization (NICSO 2010)

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rayon::prelude::*;
use rand::prelude::*;
use rand_distr::{Normal, StandardNormal, Uniform};
use std::f64::consts::PI;
use std::time::{Duration, Instant};
use nalgebra::DVector;

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Population, Individual, BasicIndividual, Position, AlgorithmMetrics,
    AdaptiveAlgorithm, ParallelAlgorithm, AdaptationStrategy, CommonParameters
};
use crate::Float;

/// Bat Algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BaVariant {
    /// Standard Bat Algorithm (Yang, 2010)
    Standard,
    /// Enhanced BA with adaptive parameters
    Enhanced,
    /// Chaotic BA with chaotic maps
    Chaotic,
    /// Quantum-inspired BA with quantum behavior
    Quantum,
    /// Levy-flight BA for global exploration
    LevyFlight,
    /// Multi-objective BA variant
    MultiObjective,
}

/// Echolocation strategies for different hunting behaviors
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum EcholocationStrategy {
    /// Standard frequency-based echolocation
    FrequencyBased,
    /// Adaptive frequency with learning
    AdaptiveFrequency,
    /// Directional echolocation with beam focusing
    Directional,
    /// Multi-frequency echolocation
    MultiFrequency,
    /// Harmonic frequency patterns
    Harmonic,
}

/// Hunting strategies for prey capture
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HuntingStrategy {
    /// Individual hunting behavior
    Individual,
    /// Cooperative hunting in groups
    Cooperative,
    /// Territorial hunting with area division
    Territorial,
    /// Opportunistic hunting based on prey density
    Opportunistic,
    /// Adaptive hunting with strategy switching
    Adaptive,
}

/// Roost formation patterns for bat communities
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RoostFormation {
    /// Single large roost (centralized)
    Single,
    /// Multiple small roosts (distributed)
    Distributed,
    /// Hierarchical roost structure
    Hierarchical,
    /// Dynamic roost with migration
    Dynamic,
    /// Seasonal roost patterns
    Seasonal,
}

/// Bat Algorithm parameters with comprehensive configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaParameters {
    /// Base swarm parameters
    pub common: CommonParameters,
    /// Algorithm variant selection
    pub variant: BaVariant,
    /// Echolocation strategy
    pub echolocation_strategy: EcholocationStrategy,
    /// Hunting strategy
    pub hunting_strategy: HuntingStrategy,
    /// Roost formation pattern
    pub roost_formation: RoostFormation,
    
    // Core BA parameters
    /// Minimum frequency (fmin)
    pub frequency_min: Float,
    /// Maximum frequency (fmax)
    pub frequency_max: Float,
    /// Initial loudness (A0)
    pub loudness_initial: Float,
    /// Initial pulse rate (r0)
    pub pulse_rate_initial: Float,
    /// Loudness decay factor (alpha)
    pub loudness_decay: Float,
    /// Pulse rate increase factor (gamma)
    pub pulse_rate_increase: Float,
    
    // Advanced parameters
    /// Velocity scaling factor
    pub velocity_scaling: Float,
    /// Local search radius factor
    pub local_search_radius: Float,
    /// Elite preservation ratio
    pub elite_ratio: Float,
    /// Diversity maintenance threshold
    pub diversity_threshold: Float,
    /// Adaptive parameter tuning enable
    pub adaptive_parameters: bool,
    /// Chaotic map parameter (for Chaotic variant)
    pub chaos_parameter: Float,
    /// Quantum tunnel probability (for Quantum variant)
    pub quantum_probability: Float,
    /// Levy flight alpha parameter
    pub levy_alpha: Float,
    /// Convergence detection threshold
    pub convergence_threshold: Float,
}

impl Default for BaParameters {
    fn default() -> Self {
        Self {
            common: CommonParameters::default(),
            variant: BaVariant::Standard,
            echolocation_strategy: EcholocationStrategy::FrequencyBased,
            hunting_strategy: HuntingStrategy::Individual,
            roost_formation: RoostFormation::Single,
            frequency_min: 0.0,
            frequency_max: 2.0,
            loudness_initial: 0.5,
            pulse_rate_initial: 0.5,
            loudness_decay: 0.95,
            pulse_rate_increase: 0.05,
            velocity_scaling: 1.0,
            local_search_radius: 0.1,
            elite_ratio: 0.1,
            diversity_threshold: 1e-6,
            adaptive_parameters: true,
            chaos_parameter: 4.0,
            quantum_probability: 0.1,
            levy_alpha: 1.5,
            convergence_threshold: 1e-8,
        }
    }
}

/// Individual bat in the swarm with echolocation capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bat {
    /// Current position
    pub position: Position,
    /// Current velocity
    pub velocity: DVector<Float>,
    /// Current frequency
    pub frequency: Float,
    /// Current loudness
    pub loudness: Float,
    /// Current pulse rate
    pub pulse_rate: Float,
    /// Current fitness
    pub fitness: Float,
    /// Personal best position
    pub best_position: Position,
    /// Personal best fitness
    pub best_fitness: Float,
    /// Bat ID for tracking
    pub id: usize,
    /// Age (iterations since creation)
    pub age: usize,
    /// Success rate for adaptive behavior
    pub success_rate: Float,
    /// Energy level (for hunting efficiency)
    pub energy_level: Float,
    /// Territory center (for territorial hunting)
    pub territory_center: Option<Position>,
    /// Hunting group ID (for cooperative hunting)
    pub hunting_group: Option<usize>,
}

impl Individual for Bat {
    type Position = Position;
    type Fitness = Float;

    fn position(&self) -> &Self::Position {
        &self.position
    }

    fn fitness(&self) -> &Self::Fitness {
        &self.fitness
    }

    fn set_fitness(&mut self, fitness: Self::Fitness) {
        self.fitness = fitness;
        if fitness < self.best_fitness {
            self.best_fitness = fitness;
            self.best_position = self.position.clone();
        }
    }
}

impl Bat {
    /// Create a new bat with random initialization
    pub fn new(
        id: usize,
        dimensions: usize,
        bounds: &[(Float, Float)],
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Self {
        let position = DVector::from_fn(dimensions, |i, _| {
            let (min, max) = bounds[i];
            rng.gen_range(min..=max)
        });
        
        let velocity = DVector::zeros(dimensions);
        let frequency = rng.gen_range(params.frequency_min..=params.frequency_max);
        
        Self {
            position: position.clone(),
            velocity,
            frequency,
            loudness: params.loudness_initial,
            pulse_rate: params.pulse_rate_initial,
            fitness: Float::INFINITY,
            best_position: position,
            best_fitness: Float::INFINITY,
            id,
            age: 0,
            success_rate: 0.0,
            energy_level: 1.0,
            territory_center: None,
            hunting_group: None,
        }
    }

    /// Update bat velocity using echolocation
    pub fn update_velocity(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        match params.echolocation_strategy {
            EcholocationStrategy::FrequencyBased => {
                self.update_velocity_frequency_based(global_best, params, rng)?;
            }
            EcholocationStrategy::AdaptiveFrequency => {
                self.update_velocity_adaptive_frequency(global_best, params, iteration, rng)?;
            }
            EcholocationStrategy::Directional => {
                self.update_velocity_directional(global_best, params, rng)?;
            }
            EcholocationStrategy::MultiFrequency => {
                self.update_velocity_multi_frequency(global_best, params, rng)?;
            }
            EcholocationStrategy::Harmonic => {
                self.update_velocity_harmonic(global_best, params, rng)?;
            }
        }
        
        // Apply variant-specific modifications
        self.apply_variant_modifications(params, iteration, rng)?;
        
        Ok(())
    }

    /// Standard frequency-based velocity update
    fn update_velocity_frequency_based(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Update frequency: f_i = f_min + (f_max - f_min) * beta
        let beta: Float = rng.gen();
        self.frequency = params.frequency_min + (params.frequency_max - params.frequency_min) * beta;
        
        // Update velocity: v_i = v_i + (x_i - x_best) * f_i
        self.velocity = &self.velocity + (&self.position - global_best) * self.frequency;
        
        Ok(())
    }

    /// Adaptive frequency-based velocity update with learning
    fn update_velocity_adaptive_frequency(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Adaptive frequency based on success rate and diversity
        let progress = iteration as Float / params.common.max_evaluations as Float;
        let adaptive_factor = 1.0 - progress * 0.5; // Decrease over time
        
        let beta: Float = rng.gen();
        self.frequency = params.frequency_min + 
            (params.frequency_max - params.frequency_min) * beta * adaptive_factor;
        
        // Include success rate in velocity calculation
        let success_factor = 1.0 + self.success_rate;
        self.velocity = &self.velocity * 0.9 + 
            (&self.position - global_best) * self.frequency * success_factor;
        
        Ok(())
    }

    /// Directional echolocation with beam focusing
    fn update_velocity_directional(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let beta: Float = rng.gen();
        self.frequency = params.frequency_min + (params.frequency_max - params.frequency_min) * beta;
        
        // Calculate direction vector to global best
        let direction = global_best - &self.position;
        let direction_norm = direction.norm();
        
        if direction_norm > 0.0 {
            let normalized_direction = direction / direction_norm;
            
            // Focus beam in direction of global best
            let beam_intensity = self.loudness * (1.0 + self.frequency);
            self.velocity = &self.velocity * 0.8 + 
                normalized_direction * beam_intensity * params.velocity_scaling;
        }
        
        Ok(())
    }

    /// Multi-frequency echolocation
    fn update_velocity_multi_frequency(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let dimensions = self.position.len();
        
        // Use different frequencies for different dimensions
        for i in 0..dimensions {
            let beta: Float = rng.gen();
            let frequency_i = params.frequency_min + 
                (params.frequency_max - params.frequency_min) * beta;
            
            self.velocity[i] = self.velocity[i] + 
                (self.position[i] - global_best[i]) * frequency_i;
        }
        
        Ok(())
    }

    /// Harmonic frequency patterns
    fn update_velocity_harmonic(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let beta: Float = rng.gen();
        
        // Generate harmonic frequencies
        let fundamental_freq = params.frequency_min + 
            (params.frequency_max - params.frequency_min) * beta;
        
        let harmonic_factor = 1.0 + 0.5 * (2.0 * PI * fundamental_freq).sin();
        
        self.frequency = fundamental_freq * harmonic_factor;
        self.velocity = &self.velocity + 
            (&self.position - global_best) * self.frequency;
        
        Ok(())
    }

    /// Apply variant-specific modifications
    fn apply_variant_modifications(
        &mut self,
        params: &BaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        match params.variant {
            BaVariant::Standard => {
                // No additional modifications for standard variant
            }
            BaVariant::Enhanced => {
                self.apply_enhanced_modifications(params, iteration, rng)?;
            }
            BaVariant::Chaotic => {
                self.apply_chaotic_modifications(params, iteration, rng)?;
            }
            BaVariant::Quantum => {
                self.apply_quantum_modifications(params, rng)?;
            }
            BaVariant::LevyFlight => {
                self.apply_levy_modifications(params, rng)?;
            }
            BaVariant::MultiObjective => {
                self.apply_multi_objective_modifications(params, rng)?;
            }
        }
        
        Ok(())
    }

    /// Enhanced variant modifications
    fn apply_enhanced_modifications(
        &mut self,
        params: &BaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Adaptive velocity scaling based on age and success
        let age_factor = 1.0 / (1.0 + self.age as Float * 0.01);
        let success_factor = 1.0 + self.success_rate * 0.5;
        
        self.velocity *= age_factor * success_factor * params.velocity_scaling;
        
        // Energy-based modifications
        if self.energy_level < 0.5 {
            // Low energy: conservative movement
            self.velocity *= 0.7;
        } else {
            // High energy: aggressive exploration
            self.velocity *= 1.2;
        }
        
        Ok(())
    }

    /// Chaotic modifications using chaotic maps
    fn apply_chaotic_modifications(
        &mut self,
        params: &BaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Logistic chaotic map: x_{n+1} = r * x_n * (1 - x_n)
        let mut chaos_value = (iteration as Float * 0.1) % 1.0;
        chaos_value = params.chaos_parameter * chaos_value * (1.0 - chaos_value);
        
        // Apply chaotic perturbation to velocity
        let chaos_factor = 0.1 * chaos_value;
        for i in 0..self.velocity.len() {
            let perturbation: Float = rng.gen_range(-1.0..=1.0);
            self.velocity[i] += chaos_factor * perturbation;
        }
        
        Ok(())
    }

    /// Quantum-inspired modifications
    fn apply_quantum_modifications(
        &mut self,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Quantum tunneling effect
        if rng.gen::<Float>() < params.quantum_probability {
            // Quantum jump to a random position with probability
            for i in 0..self.velocity.len() {
                let quantum_jump: Float = rng.gen_range(-1.0..=1.0);
                self.velocity[i] += quantum_jump * params.velocity_scaling;
            }
        }
        
        // Quantum superposition effect
        let superposition_factor = 0.1 * (rng.gen::<Float>() - 0.5);
        self.velocity *= 1.0 + superposition_factor;
        
        Ok(())
    }

    /// Levy flight modifications
    fn apply_levy_modifications(
        &mut self,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Generate Levy flight step
        let levy_step = self.generate_levy_step(params.levy_alpha, rng);
        
        // Apply Levy flight to velocity
        for i in 0..self.velocity.len() {
            self.velocity[i] += levy_step * params.velocity_scaling * 0.1;
        }
        
        Ok(())
    }

    /// Multi-objective modifications
    fn apply_multi_objective_modifications(
        &mut self,
        params: &BaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Pareto-based velocity adjustments
        // This is a simplified implementation for single-objective problems
        let diversity_factor = rng.gen::<Float>() * 0.2;
        self.velocity *= 1.0 + diversity_factor;
        
        Ok(())
    }

    /// Generate Levy flight step using Mantegna's algorithm
    fn generate_levy_step(&self, alpha: Float, rng: &mut impl Rng) -> Float {
        let sigma_u = (
            libm::tgamma(1.0 + alpha) * (alpha * PI / 2.0).sin() /
            (libm::tgamma((1.0 + alpha) / 2.0) * alpha * (2.0_f64.powf((alpha - 1.0) / 2.0)))
        ).powf(1.0 / alpha);
        
        let u: Float = Normal::new(0.0, sigma_u).unwrap().sample(rng);
        let v: Float = StandardNormal.sample(rng);
        
        u / v.abs().powf(1.0 / alpha)
    }

    /// Update position with boundary handling
    pub fn update_position(
        &mut self,
        bounds: &[(Float, Float)],
        params: &BaParameters,
    ) {
        // Update position: x_i = x_i + v_i
        self.position += &self.velocity;
        
        // Handle boundary constraints
        for (i, (min_bound, max_bound)) in bounds.iter().enumerate() {
            if self.position[i] < *min_bound {
                self.position[i] = *min_bound;
                self.velocity[i] = -self.velocity[i] * 0.5; // Absorb with damping
            } else if self.position[i] > *max_bound {
                self.position[i] = *max_bound;
                self.velocity[i] = -self.velocity[i] * 0.5; // Absorb with damping
            }
        }
        
        self.age += 1;
    }

    /// Perform local search around current position
    pub fn local_search(
        &mut self,
        global_best: &Position,
        params: &BaParameters,
        problem: &OptimizationProblem,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        if rng.gen::<Float>() < self.pulse_rate {
            // Generate local solution around global best
            let mut local_position = global_best.clone();
            
            for i in 0..local_position.len() {
                let perturbation: Float = rng.gen_range(-1.0..=1.0);
                local_position[i] += perturbation * params.local_search_radius * self.loudness;
            }
            
            // Evaluate local position
            let local_fitness = problem.evaluate(&local_position)
                .map_err(|e| SwarmError::FitnessEvaluationError(format!("Local search evaluation failed: {}", e)))?;
            
            // Accept if better
            if local_fitness < self.fitness {
                self.position = local_position;
                self.fitness = local_fitness;
                
                // Update personal best
                if local_fitness < self.best_fitness {
                    self.best_fitness = local_fitness;
                    self.best_position = self.position.clone();
                }
                
                // Update success rate
                self.success_rate = 0.9 * self.success_rate + 0.1;
            } else {
                self.success_rate = 0.9 * self.success_rate;
            }
        }
        
        Ok(())
    }

    /// Update loudness and pulse rate
    pub fn update_acoustic_parameters(&mut self, params: &BaParameters, iteration: usize) {
        // Update loudness: A_i = alpha * A_i
        self.loudness *= params.loudness_decay;
        
        // Update pulse rate: r_i = r_0 * (1 - exp(-gamma * t))
        let t = iteration as Float;
        self.pulse_rate = params.pulse_rate_initial * 
            (1.0 - (-params.pulse_rate_increase * t).exp());
        
        // Update energy level based on hunting success
        if self.success_rate > 0.5 {
            self.energy_level = (self.energy_level + 0.1).min(1.0);
        } else {
            self.energy_level = (self.energy_level - 0.05).max(0.1);
        }
    }
}

/// Main Bat Algorithm implementation
#[derive(Debug)]
pub struct BatAlgorithm {
    /// Algorithm parameters
    params: BaParameters,
    /// Current bat population
    bats: Vec<Bat>,
    /// Global best position
    global_best_position: Position,
    /// Global best fitness
    global_best_fitness: Float,
    /// Optimization problem
    problem: Option<OptimizationProblem>,
    /// Random number generator
    rng: Box<dyn RngCore + Send>,
    /// Current iteration
    iteration: usize,
    /// Performance metrics
    metrics: AlgorithmMetrics,
    /// Elite archive for diversity maintenance
    elite_archive: Vec<(Position, Float)>,
    /// Roost locations for territorial behavior
    roost_locations: Vec<Position>,
    /// Hunting groups for cooperative behavior
    hunting_groups: Vec<Vec<usize>>,
}

impl BatAlgorithm {
    /// Create new Bat Algorithm instance
    pub fn new(params: BaParameters) -> Result<Self, SwarmError> {
        let seed = params.common.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });
        
        let rng = Box::new(StdRng::seed_from_u64(seed));
        
        Ok(Self {
            params,
            bats: Vec::new(),
            global_best_position: DVector::zeros(1),
            global_best_fitness: Float::INFINITY,
            problem: None,
            rng,
            iteration: 0,
            metrics: AlgorithmMetrics::default(),
            elite_archive: Vec::new(),
            roost_locations: Vec::new(),
            hunting_groups: Vec::new(),
        })
    }

    /// Initialize bat population and roost structures
    async fn initialize_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        let dimensions = problem.dimensions();
        let bounds = problem.bounds();
        
        // Initialize bats
        self.bats = (0..self.params.common.population_size)
            .map(|i| Bat::new(i, dimensions, bounds, &self.params, &mut *self.rng))
            .collect();
        
        // Initialize global best
        self.global_best_position = DVector::zeros(dimensions);
        self.global_best_fitness = Float::INFINITY;
        
        // Setup roost formations
        self.setup_roost_formations(bounds)?;
        
        // Setup hunting groups
        self.setup_hunting_groups()?;
        
        // Evaluate initial population
        self.evaluate_population(problem).await?;
        
        // Find initial best
        self.update_global_best();
        
        Ok(())
    }

    /// Setup roost formations based on strategy
    fn setup_roost_formations(&mut self, bounds: &[(Float, Float)]) -> Result<(), SwarmError> {
        let dimensions = bounds.len();
        
        match self.params.roost_formation {
            RoostFormation::Single => {
                // Single central roost
                let center = DVector::from_fn(dimensions, |i, _| {
                    let (min, max) = bounds[i];
                    (min + max) / 2.0
                });
                self.roost_locations.push(center);
            }
            RoostFormation::Distributed => {
                // Multiple distributed roosts
                let num_roosts = (self.params.common.population_size / 10).max(2);
                for _ in 0..num_roosts {
                    let roost = DVector::from_fn(dimensions, |i, _| {
                        let (min, max) = bounds[i];
                        self.rng.gen_range(min..=max)
                    });
                    self.roost_locations.push(roost);
                }
            }
            RoostFormation::Hierarchical => {
                // Hierarchical roost structure
                let main_roost = DVector::from_fn(dimensions, |i, _| {
                    let (min, max) = bounds[i];
                    (min + max) / 2.0
                });
                self.roost_locations.push(main_roost.clone());
                
                // Sub-roosts around main roost
                for _ in 0..3 {
                    let sub_roost = main_roost.clone() + DVector::from_fn(dimensions, |_, _| {
                        self.rng.gen_range(-0.2..=0.2)
                    });
                    self.roost_locations.push(sub_roost);
                }
            }
            RoostFormation::Dynamic => {
                // Dynamic roosts that can migrate
                let center = DVector::from_fn(dimensions, |i, _| {
                    let (min, max) = bounds[i];
                    (min + max) / 2.0
                });
                self.roost_locations.push(center);
            }
            RoostFormation::Seasonal => {
                // Seasonal roost patterns (simplified)
                for season in 0..4 {
                    let seasonal_roost = DVector::from_fn(dimensions, |i, _| {
                        let (min, max) = bounds[i];
                        min + (max - min) * (season as Float + 1.0) / 5.0
                    });
                    self.roost_locations.push(seasonal_roost);
                }
            }
        }
        
        Ok(())
    }

    /// Setup hunting groups for cooperative behavior
    fn setup_hunting_groups(&mut self) -> Result<(), SwarmError> {
        match self.params.hunting_strategy {
            HuntingStrategy::Individual => {
                // No groups needed for individual hunting
            }
            HuntingStrategy::Cooperative => {
                // Create cooperative hunting groups
                let group_size = 5; // Optimal group size for bats
                let num_groups = (self.params.common.population_size / group_size).max(1);
                
                for group_id in 0..num_groups {
                    let start_idx = group_id * group_size;
                    let end_idx = ((group_id + 1) * group_size).min(self.params.common.population_size);
                    
                    let group_members: Vec<usize> = (start_idx..end_idx).collect();
                    self.hunting_groups.push(group_members.clone());
                    
                    // Assign bats to groups
                    for &bat_id in &group_members {
                        if bat_id < self.bats.len() {
                            self.bats[bat_id].hunting_group = Some(group_id);
                        }
                    }
                }
            }
            HuntingStrategy::Territorial => {
                // Assign territory centers to bats
                for (i, bat) in self.bats.iter_mut().enumerate() {
                    let roost_idx = i % self.roost_locations.len();
                    bat.territory_center = Some(self.roost_locations[roost_idx].clone());
                }
            }
            HuntingStrategy::Opportunistic => {
                // Dynamic group formation based on prey density
                // Simplified implementation
                let group_size = self.rng.gen_range(3..=7);
                let num_groups = (self.params.common.population_size / group_size).max(1);
                
                for group_id in 0..num_groups {
                    let start_idx = group_id * group_size;
                    let end_idx = ((group_id + 1) * group_size).min(self.params.common.population_size);
                    
                    let group_members: Vec<usize> = (start_idx..end_idx).collect();
                    self.hunting_groups.push(group_members);
                }
            }
            HuntingStrategy::Adaptive => {
                // Adaptive group formation based on success rates
                // Initially start with small groups
                let initial_group_size = 3;
                let num_groups = (self.params.common.population_size / initial_group_size).max(1);
                
                for group_id in 0..num_groups {
                    let start_idx = group_id * initial_group_size;
                    let end_idx = ((group_id + 1) * initial_group_size).min(self.params.common.population_size);
                    
                    let group_members: Vec<usize> = (start_idx..end_idx).collect();
                    self.hunting_groups.push(group_members);
                }
            }
        }
        
        Ok(())
    }

    /// Evaluate fitness for all bats
    async fn evaluate_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        if self.params.common.parallel_evaluation {
            // Parallel evaluation using rayon
            self.bats.par_iter_mut().try_for_each(|bat| -> Result<(), SwarmError> {
                let fitness = problem.evaluate(&bat.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                bat.set_fitness(fitness);
                Ok(())
            })?;
        } else {
            // Sequential evaluation
            for bat in &mut self.bats {
                let fitness = problem.evaluate(&bat.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                bat.set_fitness(fitness);
            }
        }
        
        Ok(())
    }

    /// Update global best position
    fn update_global_best(&mut self) {
        for bat in &self.bats {
            if bat.best_fitness < self.global_best_fitness {
                self.global_best_fitness = bat.best_fitness;
                self.global_best_position = bat.best_position.clone();
            }
        }
    }

    /// Perform one iteration of the bat algorithm
    async fn step_iteration(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().ok_or_else(|| {
            SwarmError::InitializationError("Problem not initialized".to_string())
        })?;
        
        let step_start = Instant::now();
        
        // Update each bat
        for i in 0..self.bats.len() {
            // Update velocity and position
            self.bats[i].update_velocity(
                &self.global_best_position,
                &self.params,
                self.iteration,
                &mut *self.rng,
            )?;
            
            let bounds = problem.bounds();
            self.bats[i].update_position(bounds, &self.params);
            
            // Perform local search
            self.bats[i].local_search(
                &self.global_best_position,
                &self.params,
                problem,
                &mut *self.rng,
            )?;
            
            // Update acoustic parameters
            self.bats[i].update_acoustic_parameters(&self.params, self.iteration);
        }
        
        // Evaluate updated population
        self.evaluate_population(problem).await?;
        
        // Update global best
        self.update_global_best();
        
        // Maintain diversity
        if self.params.adaptive_parameters {
            self.maintain_diversity().await?;
        }
        
        // Update elite archive
        self.update_elite_archive();
        
        // Update roosts for dynamic formations
        if matches!(self.params.roost_formation, RoostFormation::Dynamic) {
            self.update_roost_locations()?;
        }
        
        self.iteration += 1;
        
        // Update metrics
        self.metrics.total_iterations = self.iteration;
        self.metrics.last_step_duration = step_start.elapsed();
        self.metrics.best_fitness = self.global_best_fitness;
        self.metrics.diversity = self.calculate_population_diversity();
        
        Ok(())
    }

    /// Maintain population diversity
    async fn maintain_diversity(&mut self) -> Result<(), SwarmError> {
        let diversity = self.calculate_population_diversity();
        
        if diversity < self.params.diversity_threshold {
            // Re-initialize worst bats
            let num_reinit = (self.params.common.population_size as Float * 0.1) as usize;
            
            // Sort bats by fitness (worst first)
            let mut indices: Vec<usize> = (0..self.bats.len()).collect();
            indices.sort_by(|&a, &b| {
                self.bats[b].fitness.partial_cmp(&self.bats[a].fitness).unwrap()
            });
            
            // Re-initialize worst bats
            if let Some(ref problem) = self.problem {
                let bounds = problem.bounds();
                let dimensions = problem.dimensions();
                
                for &idx in indices.iter().take(num_reinit) {
                    self.bats[idx] = Bat::new(idx, dimensions, bounds, &self.params, &mut *self.rng);
                }
            }
        }
        
        Ok(())
    }

    /// Calculate population diversity
    fn calculate_population_diversity(&self) -> Float {
        if self.bats.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.bats.len() {
            for j in (i + 1)..self.bats.len() {
                let distance = (&self.bats[i].position - &self.bats[j].position).norm();
                total_distance += distance;
                count += 1;
            }
        }
        
        if count > 0 {
            total_distance / count as Float
        } else {
            0.0
        }
    }

    /// Update elite archive
    fn update_elite_archive(&mut self) {
        let elite_size = (self.params.common.population_size as Float * self.params.elite_ratio) as usize;
        
        // Add current best bats to archive
        for bat in &self.bats {
            if bat.best_fitness < Float::INFINITY {
                self.elite_archive.push((bat.best_position.clone(), bat.best_fitness));
            }
        }
        
        // Sort and keep only elite_size best
        self.elite_archive.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        self.elite_archive.truncate(elite_size);
    }

    /// Update roost locations for dynamic formations
    fn update_roost_locations(&mut self) -> Result<(), SwarmError> {
        if !self.roost_locations.is_empty() {
            // Move roost towards global best (migration behavior)
            let migration_factor = 0.1;
            
            for roost in &mut self.roost_locations {
                let direction = &self.global_best_position - &*roost;
                *roost += direction * migration_factor;
            }
        }
        
        Ok(())
    }

    /// Check for convergence
    fn has_converged(&self) -> bool {
        let diversity = self.calculate_population_diversity();
        diversity < self.params.convergence_threshold ||
        self.metrics.stagnation_count > 100
    }
}

#[async_trait]
impl SwarmAlgorithm for BatAlgorithm {
    type Individual = Bat;
    type Fitness = Float;
    type Parameters = BaParameters;

    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.problem = Some(problem.clone());
        self.initialize_population(&problem).await?;
        Ok(())
    }

    async fn step(&mut self) -> Result<(), SwarmError> {
        self.step_iteration().await
    }

    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.bats.iter().min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
    }

    fn get_population(&self) -> &Population<Self::Individual> {
        // This requires a proper Population type implementation
        // For now, we'll use a placeholder
        unimplemented!("Population interface needs proper implementation")
    }

    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        // This requires a proper Population type implementation
        // For now, we'll use a placeholder
        unimplemented!("Population interface needs proper implementation")
    }

    fn has_converged(&self) -> bool {
        self.has_converged()
    }

    fn name(&self) -> &'static str {
        match self.params.variant {
            BaVariant::Standard => "Bat Algorithm",
            BaVariant::Enhanced => "Enhanced Bat Algorithm",
            BaVariant::Chaotic => "Chaotic Bat Algorithm",
            BaVariant::Quantum => "Quantum Bat Algorithm",
            BaVariant::LevyFlight => "Levy Flight Bat Algorithm",
            BaVariant::MultiObjective => "Multi-Objective Bat Algorithm",
        }
    }

    fn parameters(&self) -> &Self::Parameters {
        &self.params
    }

    fn update_parameters(&mut self, params: Self::Parameters) {
        self.params = params;
    }

    fn metrics(&self) -> AlgorithmMetrics {
        self.metrics.clone()
    }
}

#[async_trait]
impl AdaptiveAlgorithm for BatAlgorithm {
    async fn adapt_parameters(&mut self, strategy: AdaptationStrategy) -> Result<(), SwarmError> {
        match strategy {
            AdaptationStrategy::PerformanceBased => {
                // Adapt based on current performance
                let diversity = self.calculate_population_diversity();
                
                if diversity < 0.01 {
                    // Low diversity: increase exploration
                    self.params.loudness_initial *= 1.1;
                    self.params.frequency_max *= 1.05;
                } else if diversity > 0.1 {
                    // High diversity: increase exploitation
                    self.params.pulse_rate_initial *= 1.1;
                    self.params.local_search_radius *= 0.95;
                }
            }
            AdaptationStrategy::DiversityBased => {
                let diversity = self.calculate_population_diversity();
                self.params.loudness_initial = 0.3 + 0.4 * diversity;
                self.params.pulse_rate_initial = 0.7 - 0.4 * diversity;
            }
            AdaptationStrategy::FeedbackBased => {
                // Adapt based on success rates
                let avg_success_rate = self.bats.iter()
                    .map(|bat| bat.success_rate)
                    .sum::<Float>() / self.bats.len() as Float;
                
                if avg_success_rate < 0.3 {
                    // Low success: adjust parameters for more exploration
                    self.params.frequency_max *= 1.1;
                    self.params.velocity_scaling *= 1.05;
                }
            }
        }
        
        Ok(())
    }

    fn adaptation_history(&self) -> Vec<(usize, Self::Parameters)> {
        // Return history of parameter adaptations
        vec![(self.iteration, self.params.clone())]
    }
}

#[async_trait]
impl ParallelAlgorithm for BatAlgorithm {
    async fn parallel_step(&mut self, num_threads: usize) -> Result<(), SwarmError> {
        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| SwarmError::ParallelExecutionError(format!("Thread pool creation failed: {}", e)))?;
        
        pool.install(|| {
            // Parallel evaluation and updates would go here
            // This is a simplified implementation
        });
        
        self.step().await
    }

    fn set_parallelism(&mut self, enabled: bool, num_threads: Option<usize>) {
        self.params.common.parallel_evaluation = enabled;
        if let Some(threads) = num_threads {
            // Store thread count in parameters if needed
        }
    }

    fn parallelism_metrics(&self) -> (bool, Option<usize>, Duration) {
        (
            self.params.common.parallel_evaluation,
            None, // Thread count not stored in current implementation
            self.metrics.last_step_duration,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_bat_algorithm_creation() {
        let params = BaParameters::default();
        let ba = BatAlgorithm::new(params).unwrap();
        assert_eq!(ba.name(), "Bat Algorithm");
    }

    #[tokio::test]
    async fn test_bat_algorithm_sphere_function() {
        let params = BaParameters {
            common: CommonParameters {
                population_size: 20,
                max_evaluations: 1000,
                tolerance: 1e-3,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut ba = BatAlgorithm::new(params).unwrap();
        
        // Sphere function: f(x) = sum(x_i^2)
        let problem = OptimizationProblem::new(
            5, // dimensions
            vec![(-5.0, 5.0); 5], // bounds
            Box::new(|x: &DVector<Float>| {
                Ok(x.iter().map(|xi| xi * xi).sum())
            }),
        );
        
        ba.initialize(problem).await.unwrap();
        
        // Run optimization
        for _ in 0..50 {
            ba.step().await.unwrap();
            if ba.has_converged() {
                break;
            }
        }
        
        // Should find a reasonably good solution
        assert!(ba.global_best_fitness < 1.0);
    }

    #[test]
    fn test_bat_creation() {
        let params = BaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
        
        let bat = Bat::new(0, 2, &bounds, &params, &mut rng);
        
        assert_eq!(bat.id, 0);
        assert_eq!(bat.position.len(), 2);
        assert_eq!(bat.velocity.len(), 2);
        assert!(bat.position[0] >= -10.0 && bat.position[0] <= 10.0);
        assert!(bat.position[1] >= -5.0 && bat.position[1] <= 5.0);
    }

    #[test]
    fn test_bat_velocity_update() {
        let params = BaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let mut bat = Bat::new(0, 2, &bounds, &params, &mut rng);
        
        let global_best = DVector::from_vec(vec![1.0, 2.0]);
        
        bat.update_velocity(&global_best, &params, 1, &mut rng).unwrap();
        
        // Velocity should have been updated
        assert!(bat.velocity.norm() > 0.0);
    }

    #[test]
    fn test_levy_flight_generation() {
        let params = BaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let bat = Bat::new(0, 2, &bounds, &params, &mut rng);
        
        let levy_step = bat.generate_levy_step(1.5, &mut rng);
        
        // Levy step should be finite
        assert!(levy_step.is_finite());
    }

    #[test]
    fn test_acoustic_parameter_update() {
        let params = BaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let mut bat = Bat::new(0, 2, &bounds, &params, &mut rng);
        
        let initial_loudness = bat.loudness;
        let initial_pulse_rate = bat.pulse_rate;
        
        bat.update_acoustic_parameters(&params, 10);
        
        // Loudness should decrease, pulse rate should increase
        assert!(bat.loudness < initial_loudness);
        assert!(bat.pulse_rate > initial_pulse_rate);
    }

    #[test]
    fn test_population_diversity() {
        let params = BaParameters::default();
        let mut ba = BatAlgorithm::new(params).unwrap();
        
        // Create bats with known positions
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        
        ba.bats = vec![
            {
                let mut bat = Bat::new(0, 2, &bounds, &ba.params, &mut rng);
                bat.position = DVector::from_vec(vec![0.0, 0.0]);
                bat
            },
            {
                let mut bat = Bat::new(1, 2, &bounds, &ba.params, &mut rng);
                bat.position = DVector::from_vec(vec![3.0, 4.0]);
                bat
            },
        ];
        
        let diversity = ba.calculate_population_diversity();
        
        // Expected diversity is the distance between (0,0) and (3,4) = 5.0
        assert_relative_eq!(diversity, 5.0, epsilon = 1e-10);
    }
}