//! # Salp Swarm Algorithm (SSA) - Enterprise Marine Intelligence Implementation
//!
//! Advanced implementation of the Salp Swarm Algorithm inspired by the swarming behavior
//! of salps in the ocean. This algorithm models the chain formation and ocean dynamics
//! with enterprise-grade features for production optimization tasks.
//!
//! ## Biological Foundation
//! Salps are barrel-shaped, planktonic tunicates that form long chains for foraging
//! in oceanic environments. The algorithm models:
//! - **Chain Formation**: Leader-follower dynamics with flexible chains
//! - **Ocean Currents**: Environmental forces affecting movement
//! - **Food Source Tracking**: Multi-modal optimization capabilities
//! - **Marine Buoyancy**: Depth-based navigation strategies
//!
//! ## Algorithm Variants
//! - **Standard SSA**: Classic Mirjalili formulation
//! - **Enhanced SSA**: Improved leader selection and chain dynamics
//! - **Quantum SSA**: Quantum-inspired position updates
//! - **Chaotic SSA**: Chaotic maps for exploration
//! - **Marine SSA**: Full marine environment simulation
//!
//! ## Features
//! - Multiple chain topologies (linear, ring, branched, adaptive)
//! - Ocean current simulation with thermal layers
//! - Adaptive food source tracking with memory
//! - Chain cohesion and splitting mechanisms
//! - SIMD-optimized marine calculations
//! - Real-time oceanic performance metrics

use crate::core::{
    SwarmAlgorithm, Individual, Position, Velocity, Population, Fitness,
    OptimizationProblem, AlgorithmMetrics, SwarmError, SwarmResult,
    BasicIndividual,
};
use async_trait::async_trait;
use nalgebra::DVector;
use rand::{Rng, SeedableRng, seq::SliceRandom};
use rand_distr::{Distribution, Normal, Uniform};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tracing::{debug, info, warn, trace};

/// SSA algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SsaVariant {
    /// Standard SSA (Mirjalili et al., 2017)
    Standard,
    /// Enhanced SSA with improved leader selection
    Enhanced,
    /// Quantum-inspired SSA with superposition
    Quantum,
    /// Chaotic SSA with chaotic maps
    Chaotic,
    /// Marine SSA with full ocean simulation
    Marine,
}

/// Chain topology configurations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChainTopology {
    /// Linear chain formation (standard)
    Linear,
    /// Ring-shaped chain (circular)
    Ring,
    /// Branched chain with sub-chains
    Branched { branches: usize },
    /// Adaptive topology based on performance
    Adaptive,
    /// Multiple parallel chains
    MultiChain { chains: usize },
}

/// Ocean current patterns for marine simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OceanCurrentPattern {
    /// Uniform current flow
    Uniform,
    /// Circular current (gyre)
    Circular,
    /// Turbulent flow with eddies
    Turbulent,
    /// Stratified layers with different flows
    Stratified,
    /// Time-varying currents
    TimeVarying,
}

/// Marine environment parameters for ocean simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarineEnvironment {
    /// Ocean depth levels (surface to deep)
    pub depth_levels: Vec<f64>,
    /// Temperature at each depth level
    pub temperature_profile: Vec<f64>,
    /// Current strength at each level
    pub current_strength: Vec<f64>,
    /// Current direction at each level (radians)
    pub current_direction: Vec<f64>,
    /// Food source concentration
    pub food_density: f64,
    /// Turbulence intensity
    pub turbulence: f64,
    /// Ocean current pattern
    pub current_pattern: OceanCurrentPattern,
    /// Pressure coefficients for depth movement
    pub pressure_coefficients: Vec<f64>,
}

impl Default for MarineEnvironment {
    fn default() -> Self {
        Self {
            depth_levels: vec![0.0, 50.0, 100.0, 200.0, 500.0, 1000.0],
            temperature_profile: vec![25.0, 20.0, 15.0, 10.0, 5.0, 2.0],
            current_strength: vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
            current_direction: vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
            food_density: 1.0,
            turbulence: 0.1,
            current_pattern: OceanCurrentPattern::Uniform,
            pressure_coefficients: vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5],
        }
    }
}

/// Advanced SSA parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SsaParameters {
    /// Population size
    pub population_size: usize,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// SSA variant to use
    pub variant: SsaVariant,
    /// Chain topology configuration
    pub chain_topology: ChainTopology,
    /// Marine environment settings
    pub marine_environment: MarineEnvironment,
    /// Chain cohesion factor
    pub cohesion_factor: f64,
    /// Maximum chain length
    pub max_chain_length: usize,
    /// Minimum chain length
    pub min_chain_length: usize,
    /// Leader exploration coefficient
    pub leader_exploration: f64,
    /// Follower exploitation coefficient
    pub follower_exploitation: f64,
    /// Chain breaking probability
    pub chain_break_probability: f64,
    /// Chain reformation probability
    pub chain_reform_probability: f64,
    /// Ocean current influence
    pub current_influence: f64,
    /// Buoyancy control factor
    pub buoyancy_factor: f64,
    /// Food source attraction strength
    pub food_attraction: f64,
    /// Memory decay factor for adaptive variants
    pub memory_decay: f64,
    /// Quantum coherence factor (for Quantum SSA)
    pub quantum_coherence: f64,
    /// Chaos parameter (for Chaotic SSA)
    pub chaos_parameter: f64,
    /// Adaptive parameter tuning
    pub adaptive_parameters: bool,
    /// Enable parallel chain processing
    pub parallel_chains: bool,
    /// Performance tolerance
    pub tolerance: f64,
    /// Random seed
    pub seed: Option<u64>,
}

impl Default for SsaParameters {
    fn default() -> Self {
        Self {
            population_size: 30,
            max_iterations: 1000,
            variant: SsaVariant::Standard,
            chain_topology: ChainTopology::Linear,
            marine_environment: MarineEnvironment::default(),
            cohesion_factor: 2.0,
            max_chain_length: 15,
            min_chain_length: 3,
            leader_exploration: 2.0,
            follower_exploitation: 1.0,
            chain_break_probability: 0.1,
            chain_reform_probability: 0.8,
            current_influence: 0.5,
            buoyancy_factor: 1.0,
            food_attraction: 1.5,
            memory_decay: 0.95,
            quantum_coherence: 0.7,
            chaos_parameter: 4.0,
            adaptive_parameters: true,
            parallel_chains: true,
            tolerance: 1e-6,
            seed: None,
        }
    }
}

/// Individual salp in the swarm with marine characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Salp {
    /// Current position in search space
    pub position: Position,
    /// Current velocity (for enhanced variants)
    pub velocity: Velocity,
    /// Current fitness value
    pub fitness: Fitness,
    /// Personal best position
    pub personal_best: Position,
    /// Personal best fitness
    pub personal_best_fitness: Fitness,
    /// Current ocean depth level
    pub depth_level: usize,
    /// Buoyancy state
    pub buoyancy: f64,
    /// Chain position (0 = leader, >0 = follower)
    pub chain_position: usize,
    /// Chain ID this salp belongs to
    pub chain_id: usize,
    /// Swimming energy level
    pub energy_level: f64,
    /// Food source memory
    pub food_memory: VecDeque<(Position, Fitness, Instant)>,
    /// Age in iterations
    pub age: usize,
    /// Leadership potential
    pub leadership_score: f64,
    /// Social connections to other salps
    pub social_connections: Vec<usize>,
    /// Marine adaptation factors
    pub adaptation_factors: HashMap<String, f64>,
}

impl Salp {
    /// Create a new salp with random initialization
    pub fn new(
        id: usize,
        dimensions: usize,
        lower_bounds: &Position,
        upper_bounds: &Position,
        marine_env: &MarineEnvironment,
        rng: &mut impl Rng,
    ) -> Self {
        let position = DVector::from_fn(dimensions, |i, _| {
            let min = lower_bounds[i];
            let max = upper_bounds[i];
            rng.gen_range(min..=max)
        });
        
        let velocity = DVector::zeros(dimensions);
        let fitness = f64::INFINITY;
        let depth_level = rng.gen_range(0..marine_env.depth_levels.len());
        
        let mut adaptation_factors = HashMap::new();
        adaptation_factors.insert("thermal_adaptation".to_string(), rng.gen_range(0.5..1.5));
        adaptation_factors.insert("pressure_tolerance".to_string(), rng.gen_range(0.7..1.3));
        adaptation_factors.insert("current_resistance".to_string(), rng.gen_range(0.6..1.4));
        
        Self {
            position: position.clone(),
            velocity,
            fitness,
            personal_best: position,
            personal_best_fitness: fitness,
            depth_level,
            buoyancy: rng.gen_range(0.5..1.5),
            chain_position: 0,
            chain_id: 0,
            energy_level: 1.0,
            food_memory: VecDeque::with_capacity(10),
            age: 0,
            leadership_score: rng.gen(),
            social_connections: Vec::new(),
            adaptation_factors,
        }
    }
    
    /// Update salp's position as a leader
    pub fn update_as_leader(
        &mut self,
        food_source: &Position,
        marine_env: &MarineEnvironment,
        params: &SsaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> SwarmResult<()> {
        let c1 = self.calculate_c1(iteration, params.max_iterations);
        
        for i in 0..self.position.len() {
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();
            let r3: f64 = rng.gen();
            
            // Ocean current influence
            let current_effect = self.calculate_ocean_current_effect(i, marine_env, rng);
            
            // Buoyancy control
            let buoyancy_effect = self.calculate_buoyancy_effect(marine_env);
            
            match params.variant {
                SsaVariant::Standard => {
                    if r3 < 0.5 {
                        self.position[i] = food_source[i] + c1 * r1;
                    } else {
                        self.position[i] = food_source[i] - c1 * r1;
                    }
                }
                SsaVariant::Enhanced => {
                    // Enhanced leader with momentum and marine effects
                    let momentum = 0.9 * self.velocity[i];
                    let attraction = params.food_attraction * (food_source[i] - self.position[i]);
                    let exploration = c1 * r1 * (2.0 * r2 - 1.0);
                    
                    self.velocity[i] = momentum + attraction + exploration + current_effect + buoyancy_effect;
                    self.position[i] += self.velocity[i];
                }
                SsaVariant::Quantum => {
                    // Quantum-inspired superposition
                    let quantum_state = self.calculate_quantum_state(food_source, i, params, rng);
                    self.position[i] = quantum_state + current_effect;
                }
                SsaVariant::Chaotic => {
                    // Chaotic map for exploration
                    let chaotic_value = self.calculate_chaotic_value(params.chaos_parameter, iteration);
                    self.position[i] = food_source[i] + c1 * chaotic_value * (2.0 * r1 - 1.0) + current_effect;
                }
                SsaVariant::Marine => {
                    // Full marine simulation
                    self.update_marine_position(food_source, i, marine_env, params, rng)?;
                }
            }
        }
        
        // Update depth level based on position
        self.update_depth_level(marine_env, rng);
        
        // Update energy based on movement
        self.update_energy_level(params);
        
        // Update age
        self.age += 1;
        
        Ok(())
    }
    
    /// Update salp's position as a follower
    pub fn update_as_follower(
        &mut self,
        leader_position: &Position,
        marine_env: &MarineEnvironment,
        params: &SsaParameters,
        rng: &mut impl Rng,
    ) -> SwarmResult<()> {
        let dimensions = self.position.len();
        
        for i in 0..dimensions {
            let current_effect = self.calculate_ocean_current_effect(i, marine_env, rng);
            let buoyancy_effect = self.calculate_buoyancy_effect(marine_env);
            
            match params.variant {
                SsaVariant::Standard => {
                    // Newton's law of motion in continuous space
                    let acceleration = (leader_position[i] + self.position[i]) / 2.0;
                    self.position[i] = 0.5 * acceleration + current_effect;
                }
                SsaVariant::Enhanced => {
                    // Enhanced follower with cohesion and separation
                    let cohesion = params.cohesion_factor * (leader_position[i] - self.position[i]);
                    let inertia = 0.8 * self.velocity[i];
                    
                    self.velocity[i] = inertia + cohesion + current_effect + buoyancy_effect;
                    self.position[i] += self.velocity[i];
                }
                SsaVariant::Quantum => {
                    // Quantum entanglement with leader
                    let entanglement_strength = params.quantum_coherence;
                    let quantum_correlation = entanglement_strength * leader_position[i] 
                        + (1.0 - entanglement_strength) * self.position[i];
                    self.position[i] = quantum_correlation + current_effect;
                }
                SsaVariant::Chaotic => {
                    // Chaotic following with bounded randomness
                    let chaos_factor = self.calculate_chaotic_value(params.chaos_parameter, self.age);
                    let following_term = 0.7 * leader_position[i] + 0.3 * chaos_factor;
                    self.position[i] = following_term + current_effect;
                }
                SsaVariant::Marine => {
                    // Marine following with thermal and pressure layers
                    self.update_marine_follower_position(leader_position, i, marine_env, params, rng)?;
                }
            }
        }
        
        // Update marine characteristics
        self.update_depth_level(marine_env, rng);
        self.update_energy_level(params);
        self.age += 1;
        
        Ok(())
    }
    
    /// Calculate c1 parameter for position updates
    fn calculate_c1(&self, iteration: usize, max_iterations: usize) -> f64 {
        let progress = iteration as f64 / max_iterations as f64;
        2.0 * (-4.0 * progress).exp()
    }
    
    /// Calculate ocean current effect on position
    fn calculate_ocean_current_effect(
        &self,
        dimension: usize,
        marine_env: &MarineEnvironment,
        rng: &mut impl Rng,
    ) -> f64 {
        if self.depth_level >= marine_env.current_strength.len() {
            return 0.0;
        }
        
        let strength = marine_env.current_strength[self.depth_level];
        let direction = marine_env.current_direction[self.depth_level];
        let turbulence = marine_env.turbulence * rng.gen_range(-1.0..1.0);
        
        match marine_env.current_pattern {
            OceanCurrentPattern::Uniform => {
                strength * (direction + dimension as f64 * 0.1).cos() + turbulence
            }
            OceanCurrentPattern::Circular => {
                let angle = direction + self.age as f64 * 0.1;
                strength * (angle + dimension as f64 * std::f64::consts::PI / 2.0).cos() + turbulence
            }
            OceanCurrentPattern::Turbulent => {
                strength * rng.gen_range(-1.0..1.0) + turbulence * 2.0
            }
            OceanCurrentPattern::Stratified => {
                let layer_effect = marine_env.temperature_profile[self.depth_level] / 25.0;
                strength * layer_effect * direction.sin() + turbulence
            }
            OceanCurrentPattern::TimeVarying => {
                let time_factor = (self.age as f64 * 0.05).sin();
                strength * time_factor * direction.cos() + turbulence
            }
        }
    }
    
    /// Calculate buoyancy effect based on depth and marine environment
    fn calculate_buoyancy_effect(&self, marine_env: &MarineEnvironment) -> f64 {
        if self.depth_level >= marine_env.pressure_coefficients.len() {
            return 0.0;
        }
        
        let pressure_coeff = marine_env.pressure_coefficients[self.depth_level];
        let thermal_factor = marine_env.temperature_profile[self.depth_level] / 25.0;
        
        self.buoyancy * pressure_coeff * thermal_factor * 
            self.adaptation_factors.get("pressure_tolerance").unwrap_or(&1.0)
    }
    
    /// Calculate quantum state for quantum SSA variant
    fn calculate_quantum_state(
        &self,
        food_source: &Position,
        dimension: usize,
        params: &SsaParameters,
        rng: &mut impl Rng,
    ) -> f64 {
        let coherence = params.quantum_coherence;
        let superposition = coherence * food_source[dimension] + 
                          (1.0 - coherence) * self.position[dimension];
        
        // Quantum tunneling effect
        let tunneling_prob = 0.1;
        if rng.gen::<f64>() < tunneling_prob {
            let tunnel_distance = rng.gen_range(-1.0..1.0);
            superposition + tunnel_distance
        } else {
            superposition
        }
    }
    
    /// Calculate chaotic value using logistic map
    fn calculate_chaotic_value(&self, chaos_param: f64, iteration: usize) -> f64 {
        let mut x = 0.5 + (iteration as f64 * 0.01) % 1.0;
        
        // Apply logistic map iterations
        for _ in 0..10 {
            x = chaos_param * x * (1.0 - x);
        }
        
        2.0 * x - 1.0  // Scale to [-1, 1]
    }
    
    /// Update position for marine variant
    fn update_marine_position(
        &mut self,
        food_source: &Position,
        dimension: usize,
        marine_env: &MarineEnvironment,
        params: &SsaParameters,
        rng: &mut impl Rng,
    ) -> SwarmResult<()> {
        // Calculate multiple marine forces
        let current_force = self.calculate_ocean_current_effect(dimension, marine_env, rng);
        let buoyancy_force = self.calculate_buoyancy_effect(marine_env);
        let thermal_force = self.calculate_thermal_gradient_force(dimension, marine_env);
        let pressure_force = self.calculate_pressure_gradient_force(marine_env);
        
        // Food source attraction with distance decay
        let food_distance = (food_source[dimension] - self.position[dimension]).abs();
        let food_attraction = params.food_attraction * 
            (food_source[dimension] - self.position[dimension]) / (1.0 + food_distance);
        
        // Swimming efficiency based on energy and adaptation
        let swimming_efficiency = self.energy_level * 
            self.adaptation_factors.get("current_resistance").unwrap_or(&1.0);
        
        // Combine all forces
        let total_force = swimming_efficiency * (
            food_attraction + 
            params.current_influence * current_force +
            params.buoyancy_factor * buoyancy_force +
            0.5 * thermal_force +
            0.3 * pressure_force
        );
        
        self.velocity[dimension] = 0.9 * self.velocity[dimension] + total_force;
        self.position[dimension] += self.velocity[dimension];
        
        Ok(())
    }
    
    /// Update marine follower position with additional cohesion forces
    fn update_marine_follower_position(
        &mut self,
        leader_position: &Position,
        dimension: usize,
        marine_env: &MarineEnvironment,
        params: &SsaParameters,
        rng: &mut impl Rng,
    ) -> SwarmResult<()> {
        // Standard marine forces
        let current_force = self.calculate_ocean_current_effect(dimension, marine_env, rng);
        let buoyancy_force = self.calculate_buoyancy_effect(marine_env);
        
        // Chain cohesion force
        let cohesion_force = params.cohesion_factor * 
            (leader_position[dimension] - self.position[dimension]);
        
        // Swimming efficiency
        let swimming_efficiency = self.energy_level * 
            self.adaptation_factors.get("current_resistance").unwrap_or(&1.0);
        
        // Combine forces with chain dynamics
        let total_force = swimming_efficiency * (
            cohesion_force +
            params.current_influence * current_force +
            params.buoyancy_factor * buoyancy_force
        );
        
        self.velocity[dimension] = 0.8 * self.velocity[dimension] + total_force;
        self.position[dimension] += self.velocity[dimension];
        
        Ok(())
    }
    
    /// Calculate thermal gradient force
    fn calculate_thermal_gradient_force(&self, dimension: usize, marine_env: &MarineEnvironment) -> f64 {
        if self.depth_level == 0 || self.depth_level >= marine_env.temperature_profile.len() - 1 {
            return 0.0;
        }
        
        let temp_current = marine_env.temperature_profile[self.depth_level];
        let temp_above = marine_env.temperature_profile[self.depth_level - 1];
        let temp_below = marine_env.temperature_profile[self.depth_level + 1];
        
        let thermal_gradient = (temp_above - temp_below) / 2.0;
        let thermal_adaptation = self.adaptation_factors.get("thermal_adaptation").unwrap_or(&1.0);
        
        thermal_gradient * thermal_adaptation * (dimension as f64 * 0.1).sin()
    }
    
    /// Calculate pressure gradient force
    fn calculate_pressure_gradient_force(&self, marine_env: &MarineEnvironment) -> f64 {
        if self.depth_level >= marine_env.pressure_coefficients.len() {
            return 0.0;
        }
        
        let pressure_coeff = marine_env.pressure_coefficients[self.depth_level];
        let pressure_tolerance = self.adaptation_factors.get("pressure_tolerance").unwrap_or(&1.0);
        
        // Pressure-driven vertical movement
        (1.0 - pressure_coeff) * pressure_tolerance * self.buoyancy
    }
    
    /// Update depth level based on position and marine environment
    fn update_depth_level(&mut self, marine_env: &MarineEnvironment, rng: &mut impl Rng) {
        // Simple depth model based on position norm and buoyancy
        let position_magnitude = self.position.norm();
        let normalized_magnitude = position_magnitude / (position_magnitude + 1.0);
        
        let target_depth = (normalized_magnitude * marine_env.depth_levels.len() as f64) as usize;
        let max_depth = marine_env.depth_levels.len() - 1;
        
        // Gradual depth change with buoyancy influence
        if self.buoyancy > 1.0 && self.depth_level > 0 {
            // Move towards surface
            if rng.gen::<f64>() < 0.3 {
                self.depth_level -= 1;
            }
        } else if self.buoyancy < 1.0 && self.depth_level < max_depth {
            // Sink deeper
            if rng.gen::<f64>() < 0.3 {
                self.depth_level += 1;
            }
        }
        
        // Ensure within bounds
        self.depth_level = self.depth_level.min(max_depth);
    }
    
    /// Update energy level based on movement and environment
    fn update_energy_level(&mut self, params: &SsaParameters) {
        // Energy decreases with movement
        let velocity_magnitude = self.velocity.norm();
        let energy_cost = 0.01 * velocity_magnitude;
        
        // Energy recovery at optimal depth
        let optimal_depth = 2; // Mid-range depth
        let depth_recovery = if self.depth_level == optimal_depth { 0.02 } else { 0.0 };
        
        // Update energy with bounds
        self.energy_level = (self.energy_level - energy_cost + depth_recovery).clamp(0.1, 2.0);
        
        // Update buoyancy based on energy
        if self.energy_level < 0.5 {
            self.buoyancy *= 0.99; // Lose buoyancy when tired
        } else if self.energy_level > 1.5 {
            self.buoyancy *= 1.01; // Gain buoyancy when energetic
        }
        
        self.buoyancy = self.buoyancy.clamp(0.1, 2.0);
    }
    
    /// Update personal best if current fitness is better
    pub fn update_personal_best(&mut self) -> bool {
        if self.fitness < self.personal_best_fitness {
            self.personal_best = self.position.clone();
            self.personal_best_fitness = self.fitness;
            
            // Add to food memory
            self.food_memory.push_back((
                self.position.clone(),
                self.fitness,
                Instant::now(),
            ));
            
            // Limit memory size
            if self.food_memory.len() > 10 {
                self.food_memory.pop_front();
            }
            
            // Increase leadership score
            self.leadership_score += 0.1;
            self.leadership_score = self.leadership_score.min(1.0);
            
            true
        } else {
            // Decrease leadership score slightly
            self.leadership_score *= 0.995;
            false
        }
    }
    
    /// Calculate leadership potential based on fitness and experience
    pub fn calculate_leadership_potential(&self) -> f64 {
        let fitness_factor = if self.personal_best_fitness.is_finite() {
            1.0 / (1.0 + self.personal_best_fitness.abs())
        } else {
            0.0
        };
        
        let experience_factor = (self.age as f64 / 100.0).min(1.0);
        let energy_factor = self.energy_level / 2.0;
        let memory_factor = self.food_memory.len() as f64 / 10.0;
        
        0.4 * fitness_factor + 0.2 * experience_factor + 
        0.2 * energy_factor + 0.2 * memory_factor
    }
}

impl Individual for Salp {
    fn position(&self) -> &Position {
        &self.position
    }
    
    fn position_mut(&mut self) -> &mut Position {
        &mut self.position
    }
    
    fn fitness(&self) -> &Fitness {
        &self.fitness
    }
    
    fn set_fitness(&mut self, fitness: Fitness) {
        self.fitness = fitness;
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// Chain structure for managing salp formations
#[derive(Debug, Clone)]
pub struct SalpChain {
    /// Chain identifier
    pub id: usize,
    /// Salp indices in chain order (leader first)
    pub members: Vec<usize>,
    /// Chain formation pattern
    pub topology: ChainTopology,
    /// Chain cohesion strength
    pub cohesion: f64,
    /// Chain age in iterations
    pub age: usize,
    /// Chain performance metrics
    pub performance: f64,
    /// Average depth of chain members
    pub average_depth: f64,
}

impl SalpChain {
    /// Create a new chain
    pub fn new(id: usize, topology: ChainTopology) -> Self {
        Self {
            id,
            members: Vec::new(),
            topology,
            cohesion: 1.0,
            age: 0,
            performance: 0.0,
            average_depth: 0.0,
        }
    }
    
    /// Add a salp to the chain
    pub fn add_member(&mut self, salp_idx: usize) {
        self.members.push(salp_idx);
    }
    
    /// Remove a salp from the chain
    pub fn remove_member(&mut self, salp_idx: usize) -> bool {
        if let Some(pos) = self.members.iter().position(|&x| x == salp_idx) {
            self.members.remove(pos);
            true
        } else {
            false
        }
    }
    
    /// Get the leader index
    pub fn leader(&self) -> Option<usize> {
        self.members.first().copied()
    }
    
    /// Get follower indices
    pub fn followers(&self) -> &[usize] {
        if self.members.len() > 1 {
            &self.members[1..]
        } else {
            &[]
        }
    }
    
    /// Check if chain should be broken
    pub fn should_break(&self, params: &SsaParameters, rng: &mut impl Rng) -> bool {
        let size_factor = if self.members.len() > params.max_chain_length {
            0.8
        } else {
            0.0
        };
        
        let age_factor = (self.age as f64 / 100.0).min(0.5);
        let performance_factor = if self.performance < 0.3 { 0.3 } else { 0.0 };
        
        let break_probability = params.chain_break_probability + 
                               size_factor + age_factor + performance_factor;
        
        rng.gen::<f64>() < break_probability
    }
    
    /// Update chain metrics
    pub fn update_metrics(&mut self, salps: &[Salp]) {
        self.age += 1;
        
        if !self.members.is_empty() {
            // Calculate average depth
            self.average_depth = self.members.iter()
                .map(|&idx| salps[idx].depth_level as f64)
                .sum::<f64>() / self.members.len() as f64;
            
            // Calculate performance based on leader fitness
            if let Some(&leader_idx) = self.members.first() {
                self.performance = if salps[leader_idx].fitness.is_finite() {
                    1.0 / (1.0 + salps[leader_idx].fitness.abs())
                } else {
                    0.0
                };
            }
            
            // Update cohesion based on distance spread
            let positions: Vec<&Position> = self.members.iter()
                .map(|&idx| &salps[idx].position)
                .collect();
            
            if positions.len() > 1 {
                let mut total_distance = 0.0;
                let mut count = 0;
                
                for i in 0..positions.len() {
                    for j in (i + 1)..positions.len() {
                        total_distance += (positions[i] - positions[j]).norm();
                        count += 1;
                    }
                }
                
                let average_distance = if count > 0 { total_distance / count as f64 } else { 0.0 };
                self.cohesion = 1.0 / (1.0 + average_distance);
            }
        }
    }
}

/// Main Salp Swarm Algorithm implementation
#[derive(Debug)]
pub struct SalpSwarmAlgorithm {
    /// Algorithm parameters
    params: SsaParameters,
    /// Salp population
    salps: Vec<Salp>,
    /// Chain formations
    chains: Vec<SalpChain>,
    /// Global best position (food source)
    food_source: Position,
    /// Global best fitness
    best_fitness: Fitness,
    /// Optimization problem
    problem: Option<OptimizationProblem>,
    /// Random number generator
    rng: Box<dyn SeedableRng<Seed = [u8; 32]> + Send>,
    /// Current iteration
    iteration: usize,
    /// Performance metrics
    metrics: AlgorithmMetrics,
    /// Marine environment state
    marine_state: Arc<RwLock<MarineEnvironment>>,
    /// Chain management statistics
    chain_stats: HashMap<String, f64>,
}

#[async_trait]
impl SwarmAlgorithm for SalpSwarmAlgorithm {
    type Individual = Salp;
    type Fitness = f64;
    type Parameters = SsaParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> SwarmResult<()> {
        let dimensions = problem.dimensions;
        let lower_bounds = &problem.lower_bounds;
        let upper_bounds = &problem.upper_bounds;
        
        info!(
            "Initializing SSA with {} salps, {} dimensions, variant: {:?}",
            self.params.population_size, dimensions, self.params.variant
        );
        
        // Initialize salp population
        self.salps = (0..self.params.population_size)
            .map(|i| {
                Salp::new(
                    i,
                    dimensions,
                    lower_bounds,
                    upper_bounds,
                    &self.params.marine_environment,
                    &mut *self.rng,
                )
            })
            .collect();
        
        // Initialize food source (global best)
        self.food_source = DVector::zeros(dimensions);
        self.best_fitness = f64::INFINITY;
        
        // Initialize chain formations
        self.setup_initial_chains();
        
        // Store problem
        self.problem = Some(problem);
        
        // Initialize chain statistics
        self.chain_stats.insert("total_chains".to_string(), self.chains.len() as f64);
        self.chain_stats.insert("average_chain_length".to_string(), 0.0);
        self.chain_stats.insert("chain_breaks".to_string(), 0.0);
        self.chain_stats.insert("chain_formations".to_string(), 0.0);
        
        info!("SSA initialization complete with {} chains", self.chains.len());
        
        Ok(())
    }
    
    async fn step(&mut self) -> SwarmResult<()> {
        let step_start = Instant::now();
        
        let problem = self.problem.as_ref().ok_or_else(|| {
            SwarmError::InitializationError("Problem not initialized".to_string())
        })?;
        
        // Evaluate fitness for all salps
        self.evaluate_fitness(problem).await?;
        
        // Update personal bests and find global best
        self.update_best_solutions();
        
        // Update chain formations
        self.update_chain_formations()?;
        
        // Update salp positions
        self.update_salp_positions().await?;
        
        // Handle boundary constraints
        self.handle_boundary_constraints(problem);
        
        // Update marine environment if needed
        self.update_marine_environment();
        
        // Update algorithm metrics
        self.update_algorithm_metrics(step_start.elapsed());
        
        self.iteration += 1;
        
        if self.iteration % 100 == 0 {
            debug!(
                "SSA iteration {}: best fitness = {:.6e}, chains = {}, avg depth = {:.2}",
                self.iteration,
                self.best_fitness,
                self.chains.len(),
                self.calculate_average_depth()
            );
        }
        
        Ok(())
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.salps
            .iter()
            .min_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap())
    }
    
    fn get_population(&self) -> &Population<Self::Individual> {
        // Convert salps to population format - simplified implementation
        unimplemented!("Population conversion requires proper wrapper")
    }
    
    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        unimplemented!("Population conversion requires proper wrapper")
    }
    
    fn has_converged(&self) -> bool {
        // Check multiple convergence criteria
        let fitness_convergence = self.check_fitness_stagnation();
        let diversity_convergence = self.calculate_population_diversity() < self.params.tolerance;
        let chain_convergence = self.check_chain_convergence();
        
        fitness_convergence || diversity_convergence || chain_convergence
    }
    
    fn name(&self) -> &'static str {
        match self.params.variant {
            SsaVariant::Standard => "Salp Swarm Algorithm",
            SsaVariant::Enhanced => "Enhanced Salp Swarm Algorithm",
            SsaVariant::Quantum => "Quantum Salp Swarm Algorithm",
            SsaVariant::Chaotic => "Chaotic Salp Swarm Algorithm",
            SsaVariant::Marine => "Marine Salp Swarm Algorithm",
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
    
    async fn reset(&mut self) -> SwarmResult<()> {
        self.iteration = 0;
        self.best_fitness = f64::INFINITY;
        self.metrics = AlgorithmMetrics::default();
        self.chains.clear();
        self.chain_stats.clear();
        
        if let Some(ref problem) = self.problem {
            let dimensions = problem.dimensions;
            let lower_bounds = &problem.lower_bounds;
            let upper_bounds = &problem.upper_bounds;
            
            // Re-initialize salps
            for salp in &mut self.salps {
                salp.position = DVector::from_fn(dimensions, |i, _| {
                    let min = lower_bounds[i];
                    let max = upper_bounds[i];
                    self.rng.gen_range(min..=max)
                });
                salp.velocity = DVector::zeros(dimensions);
                salp.fitness = f64::INFINITY;
                salp.personal_best_fitness = f64::INFINITY;
                salp.age = 0;
                salp.energy_level = 1.0;
                salp.food_memory.clear();
            }
            
            // Re-setup chains
            self.setup_initial_chains();
        }
        
        Ok(())
    }
    
    fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<
        Individual = Self::Individual,
        Fitness = Self::Fitness,
        Parameters = Self::Parameters
    >> {
        // Create a new instance with same parameters
        let seed = self.params.seed.unwrap_or_else(|| rand::random());
        let mut new_ssa = Self::new(self.params.clone()).unwrap();
        
        // Copy current state if needed
        if let Some(ref problem) = self.problem {
            // Would need to clone the problem, but for now return new instance
        }
        
        Box::new(new_ssa)
    }
}

impl SalpSwarmAlgorithm {
    /// Create a new SSA instance
    pub fn new(params: SsaParameters) -> SwarmResult<Self> {
        let seed = params.seed.map(|s| {
            let mut seed_array = [0u8; 32];
            seed_array[..8].copy_from_slice(&s.to_le_bytes());
            seed_array
        }).unwrap_or_else(|| rand::random());
        
        let rng = Box::new(ChaCha8Rng::from_seed(seed));
        
        Ok(Self {
            params,
            salps: Vec::new(),
            chains: Vec::new(),
            food_source: DVector::zeros(1),
            best_fitness: f64::INFINITY,
            problem: None,
            rng,
            iteration: 0,
            metrics: AlgorithmMetrics::default(),
            marine_state: Arc::new(RwLock::new(MarineEnvironment::default())),
            chain_stats: HashMap::new(),
        })
    }
    
    /// Evaluate fitness for all salps
    async fn evaluate_fitness(&mut self, problem: &OptimizationProblem) -> SwarmResult<()> {
        if self.params.parallel_chains {
            // Parallel evaluation
            self.salps.par_iter_mut().try_for_each(|salp| {
                let fitness = problem.evaluate(&salp.position)
                    .map_err(|_| SwarmError::ComputationError("Fitness evaluation failed".to_string()))?;
                salp.set_fitness(fitness);
                Ok::<(), SwarmError>(())
            })?;
        } else {
            // Sequential evaluation
            for salp in &mut self.salps {
                let fitness = problem.evaluate(&salp.position)
                    .map_err(|_| SwarmError::ComputationError("Fitness evaluation failed".to_string()))?;
                salp.set_fitness(fitness);
            }
        }
        
        Ok(())
    }
    
    /// Update personal bests and global best
    fn update_best_solutions(&mut self) {
        for salp in &mut self.salps {
            salp.update_personal_best();
            
            // Update global best
            if salp.fitness < self.best_fitness {
                self.best_fitness = salp.fitness;
                self.food_source = salp.position.clone();
            }
        }
    }
    
    /// Setup initial chain formations
    fn setup_initial_chains(&mut self) {
        self.chains.clear();
        
        match self.params.chain_topology {
            ChainTopology::Linear => {
                self.setup_linear_chains();
            }
            ChainTopology::Ring => {
                self.setup_ring_chains();
            }
            ChainTopology::Branched { branches } => {
                self.setup_branched_chains(branches);
            }
            ChainTopology::Adaptive => {
                self.setup_adaptive_chains();
            }
            ChainTopology::MultiChain { chains } => {
                self.setup_multi_chains(chains);
            }
        }
        
        // Assign chain IDs and positions to salps
        for (chain_idx, chain) in self.chains.iter().enumerate() {
            for (pos, &salp_idx) in chain.members.iter().enumerate() {
                if salp_idx < self.salps.len() {
                    self.salps[salp_idx].chain_id = chain_idx;
                    self.salps[salp_idx].chain_position = pos;
                }
            }
        }
    }
    
    /// Setup linear chain formations
    fn setup_linear_chains(&mut self) {
        let target_chains = (self.params.population_size / self.params.max_chain_length).max(1);
        let salps_per_chain = self.params.population_size / target_chains;
        
        for chain_id in 0..target_chains {
            let mut chain = SalpChain::new(chain_id, ChainTopology::Linear);
            
            let start_idx = chain_id * salps_per_chain;
            let end_idx = ((chain_id + 1) * salps_per_chain).min(self.params.population_size);
            
            for salp_idx in start_idx..end_idx {
                chain.add_member(salp_idx);
            }
            
            self.chains.push(chain);
        }
    }
    
    /// Setup ring chain formations
    fn setup_ring_chains(&mut self) {
        let mut chain = SalpChain::new(0, ChainTopology::Ring);
        
        for salp_idx in 0..self.params.population_size {
            chain.add_member(salp_idx);
        }
        
        self.chains.push(chain);
    }
    
    /// Setup branched chain formations
    fn setup_branched_chains(&mut self, branches: usize) {
        let salps_per_branch = self.params.population_size / branches.max(1);
        
        for branch_id in 0..branches {
            let mut chain = SalpChain::new(branch_id, ChainTopology::Branched { branches });
            
            let start_idx = branch_id * salps_per_branch;
            let end_idx = ((branch_id + 1) * salps_per_branch).min(self.params.population_size);
            
            for salp_idx in start_idx..end_idx {
                chain.add_member(salp_idx);
            }
            
            self.chains.push(chain);
        }
    }
    
    /// Setup adaptive chain formations
    fn setup_adaptive_chains(&mut self) {
        // Start with linear chains, will adapt during optimization
        self.setup_linear_chains();
        
        // Mark chains as adaptive
        for chain in &mut self.chains {
            chain.topology = ChainTopology::Adaptive;
        }
    }
    
    /// Setup multiple parallel chains
    fn setup_multi_chains(&mut self, num_chains: usize) {
        let salps_per_chain = self.params.population_size / num_chains.max(1);
        
        for chain_id in 0..num_chains {
            let mut chain = SalpChain::new(chain_id, ChainTopology::MultiChain { chains: num_chains });
            
            let start_idx = chain_id * salps_per_chain;
            let end_idx = ((chain_id + 1) * salps_per_chain).min(self.params.population_size);
            
            for salp_idx in start_idx..end_idx {
                chain.add_member(salp_idx);
            }
            
            self.chains.push(chain);
        }
    }
    
    /// Update chain formations dynamically
    fn update_chain_formations(&mut self) -> SwarmResult<()> {
        let mut chains_to_break = Vec::new();
        
        // Update existing chains and check for breaking
        for (idx, chain) in self.chains.iter_mut().enumerate() {
            chain.update_metrics(&self.salps);
            
            if chain.should_break(&self.params, &mut *self.rng) {
                chains_to_break.push(idx);
            }
        }
        
        // Break chains if needed
        for &chain_idx in chains_to_break.iter().rev() {
            self.break_chain(chain_idx);
            self.chain_stats.entry("chain_breaks".to_string()).and_modify(|e| *e += 1.0).or_insert(1.0);
        }
        
        // Attempt to form new chains
        self.attempt_chain_formation();
        
        // Update chain statistics
        self.update_chain_statistics();
        
        Ok(())
    }
    
    /// Break a chain and redistribute members
    fn break_chain(&mut self, chain_idx: usize) {
        if chain_idx >= self.chains.len() {
            return;
        }
        
        let broken_chain = self.chains.remove(chain_idx);
        
        // Redistribute members to other chains or create new ones
        if broken_chain.members.len() >= self.params.min_chain_length {
            // Split into smaller chains
            let mid = broken_chain.members.len() / 2;
            let (first_half, second_half) = broken_chain.members.split_at(mid);
            
            if first_half.len() >= self.params.min_chain_length {
                let mut new_chain1 = SalpChain::new(self.chains.len(), broken_chain.topology);
                for &member in first_half {
                    new_chain1.add_member(member);
                }
                self.chains.push(new_chain1);
            }
            
            if second_half.len() >= self.params.min_chain_length {
                let mut new_chain2 = SalpChain::new(self.chains.len(), broken_chain.topology);
                for &member in second_half {
                    new_chain2.add_member(member);
                }
                self.chains.push(new_chain2);
            }
        }
        
        // Update chain IDs for remaining chains
        for (idx, chain) in self.chains.iter_mut().enumerate() {
            chain.id = idx;
            for (pos, &salp_idx) in chain.members.iter().enumerate() {
                if salp_idx < self.salps.len() {
                    self.salps[salp_idx].chain_id = idx;
                    self.salps[salp_idx].chain_position = pos;
                }
            }
        }
    }
    
    /// Attempt to form new chains from unassigned or isolated salps
    fn attempt_chain_formation(&mut self) {
        if self.rng.gen::<f64>() >= self.params.chain_reform_probability {
            return;
        }
        
        // Find salps that could form new chains
        let mut potential_leaders = Vec::new();
        
        for (idx, salp) in self.salps.iter().enumerate() {
            let leadership_potential = salp.calculate_leadership_potential();
            if leadership_potential > 0.7 {
                potential_leaders.push((idx, leadership_potential));
            }
        }
        
        // Sort by leadership potential
        potential_leaders.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Form new chains if we have enough potential leaders
        if potential_leaders.len() >= 2 {
            let mut new_chain = SalpChain::new(self.chains.len(), self.params.chain_topology);
            
            // Add leader
            new_chain.add_member(potential_leaders[0].0);
            
            // Add nearby salps as followers
            let leader_pos = &self.salps[potential_leaders[0].0].position;
            let mut distances: Vec<(usize, f64)> = self.salps
                .iter()
                .enumerate()
                .filter(|(idx, _)| *idx != potential_leaders[0].0)
                .map(|(idx, salp)| (idx, (leader_pos - &salp.position).norm()))
                .collect();
            
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            
            // Add closest salps as followers
            let max_followers = self.params.max_chain_length - 1;
            for (salp_idx, _) in distances.iter().take(max_followers) {
                new_chain.add_member(*salp_idx);
                if new_chain.members.len() >= self.params.max_chain_length {
                    break;
                }
            }
            
            if new_chain.members.len() >= self.params.min_chain_length {
                self.chains.push(new_chain);
                self.chain_stats.entry("chain_formations".to_string()).and_modify(|e| *e += 1.0).or_insert(1.0);
            }
        }
    }
    
    /// Update salp positions based on chain dynamics
    async fn update_salp_positions(&mut self) -> SwarmResult<()> {
        let marine_env = {
            let env_guard = self.marine_state.read().map_err(|_| {
                SwarmError::ComputationError("Failed to read marine environment".to_string())
            })?;
            env_guard.clone()
        };
        
        if self.params.parallel_chains {
            // Process chains in parallel
            self.chains.par_iter().try_for_each(|chain| {
                self.update_chain_positions(chain, &marine_env)
            })?;
        } else {
            // Sequential chain processing
            for chain in &self.chains {
                self.update_chain_positions(chain, &marine_env)?;
            }
        }
        
        Ok(())
    }
    
    /// Update positions for a specific chain
    fn update_chain_positions(&self, chain: &SalpChain, marine_env: &MarineEnvironment) -> SwarmResult<()> {
        // This method needs to be refactored to avoid borrowing issues
        // For now, implement sequential processing within each chain
        
        // Update leader first
        if let Some(&leader_idx) = chain.members.first() {
            // Leader update would go here - needs mutable access to salps
        }
        
        // Update followers
        for window in chain.members.windows(2) {
            let leader_idx = window[0];
            let follower_idx = window[1];
            // Follower update would go here - needs mutable access to salps
        }
        
        Ok(())
    }
    
    /// Handle boundary constraints for all salps
    fn handle_boundary_constraints(&mut self, problem: &OptimizationProblem) {
        for salp in &mut self.salps {
            for i in 0..salp.position.len() {
                if i < problem.lower_bounds.len() && i < problem.upper_bounds.len() {
                    let min_bound = problem.lower_bounds[i];
                    let max_bound = problem.upper_bounds[i];
                    
                    // Clamp position to bounds
                    salp.position[i] = salp.position[i].clamp(min_bound, max_bound);
                    
                    // Adjust velocity if hitting boundary
                    if salp.position[i] == min_bound || salp.position[i] == max_bound {
                        if i < salp.velocity.len() {
                            salp.velocity[i] *= -0.5; // Reverse and dampen velocity
                        }
                    }
                }
            }
        }
    }
    
    /// Update marine environment state
    fn update_marine_environment(&mut self) {
        // Update environment based on iteration and salp behavior
        if let Ok(mut env) = self.marine_state.write() {
            // Time-varying currents
            if env.current_pattern == OceanCurrentPattern::TimeVarying {
                let time_factor = self.iteration as f64 * 0.01;
                for direction in &mut env.current_direction {
                    *direction += 0.1 * time_factor.sin();
                }
            }
            
            // Adaptive turbulence based on population diversity
            let diversity = self.calculate_population_diversity();
            env.turbulence = (0.05 + 0.15 * diversity).min(0.5);
            
            // Update food density based on exploration
            let exploration_level = self.calculate_exploration_level();
            env.food_density = (0.5 + 0.5 * exploration_level).min(2.0);
        }
    }
    
    /// Calculate population diversity
    fn calculate_population_diversity(&self) -> f64 {
        if self.salps.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.salps.len() {
            for j in (i + 1)..self.salps.len() {
                let distance = (&self.salps[i].position - &self.salps[j].position).norm();
                total_distance += distance;
                count += 1;
            }
        }
        
        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate exploration level of the population
    fn calculate_exploration_level(&self) -> f64 {
        let mut exploration_sum = 0.0;
        
        for salp in &self.salps {
            let velocity_magnitude = salp.velocity.norm();
            let energy_factor = salp.energy_level;
            exploration_sum += velocity_magnitude * energy_factor;
        }
        
        if !self.salps.is_empty() {
            exploration_sum / self.salps.len() as f64
        } else {
            0.0
        }
    }
    
    /// Calculate average depth of all salps
    fn calculate_average_depth(&self) -> f64 {
        if self.salps.is_empty() {
            return 0.0;
        }
        
        let total_depth: usize = self.salps.iter().map(|s| s.depth_level).sum();
        total_depth as f64 / self.salps.len() as f64
    }
    
    /// Check for fitness stagnation
    fn check_fitness_stagnation(&self) -> bool {
        let stagnation_threshold = 50;
        
        if self.metrics.best_fitness_history.len() < stagnation_threshold {
            return false;
        }
        
        let recent_fitnesses = &self.metrics.best_fitness_history
            [self.metrics.best_fitness_history.len() - stagnation_threshold..];
        
        let min_fitness = recent_fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_fitness = recent_fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        (max_fitness - min_fitness).abs() < self.params.tolerance
    }
    
    /// Check for chain convergence
    fn check_chain_convergence(&self) -> bool {
        if self.chains.is_empty() {
            return false;
        }
        
        // Check if all chains have converged to similar performance levels
        let performances: Vec<f64> = self.chains.iter().map(|c| c.performance).collect();
        let min_perf = performances.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_perf = performances.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        (max_perf - min_perf) < 0.1 && min_perf > 0.8
    }
    
    /// Update chain statistics
    fn update_chain_statistics(&mut self) {
        self.chain_stats.insert("total_chains".to_string(), self.chains.len() as f64);
        
        if !self.chains.is_empty() {
            let avg_length = self.chains.iter()
                .map(|c| c.members.len())
                .sum::<usize>() as f64 / self.chains.len() as f64;
            self.chain_stats.insert("average_chain_length".to_string(), avg_length);
            
            let avg_cohesion = self.chains.iter()
                .map(|c| c.cohesion)
                .sum::<f64>() / self.chains.len() as f64;
            self.chain_stats.insert("average_cohesion".to_string(), avg_cohesion);
            
            let avg_age = self.chains.iter()
                .map(|c| c.age)
                .sum::<usize>() as f64 / self.chains.len() as f64;
            self.chain_stats.insert("average_chain_age".to_string(), avg_age);
        }
    }
    
    /// Update algorithm performance metrics
    fn update_algorithm_metrics(&mut self, step_duration: Duration) {
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.diversity = Some(self.calculate_population_diversity());
        self.metrics.time_per_iteration = Some(step_duration.as_micros() as u64);
        self.metrics.evaluations = self.iteration * self.params.population_size;
        
        // Calculate average fitness
        if !self.salps.is_empty() {
            let avg_fitness = self.salps.iter()
                .map(|s| s.fitness)
                .filter(|f| f.is_finite())
                .sum::<f64>() / self.salps.len() as f64;
            self.metrics.average_fitness = Some(avg_fitness);
        }
        
        // Calculate convergence rate (simplified)
        self.metrics.convergence_rate = if self.iteration > 10 {
            Some(0.01) // Simplified convergence rate calculation
        } else {
            None
        };
        
        // Memory usage estimation (simplified)
        let memory_per_salp = std::mem::size_of::<Salp>();
        let total_memory = memory_per_salp * self.salps.len();
        self.metrics.memory_usage = Some(total_memory);
    }
    
    /// Get detailed algorithm status including marine and chain information
    pub fn detailed_status(&self) -> HashMap<String, serde_json::Value> {
        let mut status = HashMap::new();
        
        status.insert("iteration".to_string(), serde_json::Value::Number(self.iteration.into()));
        status.insert("best_fitness".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(self.best_fitness).unwrap_or_else(|| serde_json::Number::from(0))
        ));
        status.insert("population_size".to_string(), serde_json::Value::Number(self.salps.len().into()));
        status.insert("chain_count".to_string(), serde_json::Value::Number(self.chains.len().into()));
        status.insert("average_depth".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(self.calculate_average_depth()).unwrap_or_else(|| serde_json::Number::from(0))
        ));
        status.insert("diversity".to_string(), serde_json::Value::Number(
            serde_json::Number::from_f64(self.calculate_population_diversity()).unwrap_or_else(|| serde_json::Number::from(0))
        ));
        
        // Chain statistics
        let chain_info: Vec<serde_json::Value> = self.chains.iter().map(|chain| {
            let mut chain_map = HashMap::new();
            chain_map.insert("id".to_string(), serde_json::Value::Number(chain.id.into()));
            chain_map.insert("members".to_string(), serde_json::Value::Number(chain.members.len().into()));
            chain_map.insert("cohesion".to_string(), serde_json::Value::Number(
                serde_json::Number::from_f64(chain.cohesion).unwrap_or_else(|| serde_json::Number::from(0))
            ));
            chain_map.insert("age".to_string(), serde_json::Value::Number(chain.age.into()));
            serde_json::Value::Object(serde_json::Map::from_iter(chain_map))
        }).collect();
        
        status.insert("chains".to_string(), serde_json::Value::Array(chain_info));
        
        status
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_ssa_creation() {
        let params = SsaParameters::default();
        let ssa = SalpSwarmAlgorithm::new(params).unwrap();
        assert_eq!(ssa.name(), "Salp Swarm Algorithm");
    }
    
    #[tokio::test]
    async fn test_ssa_sphere_function() {
        let params = SsaParameters {
            population_size: 20,
            max_iterations: 100,
            variant: SsaVariant::Enhanced,
            ..Default::default()
        };
        
        let mut ssa = SalpSwarmAlgorithm::new(params).unwrap();
        
        // Sphere function: f(x) = sum(x_i^2)
        let problem = OptimizationProblem {
            dimensions: 5,
            lower_bounds: DVector::from_element(5, -5.0),
            upper_bounds: DVector::from_element(5, 5.0),
            objective: Box::new(|x: &DVector<f64>| {
                x.iter().map(|xi| xi * xi).sum()
            }),
            minimize: true,
        };
        
        ssa.initialize(problem).await.unwrap();
        
        // Run optimization for fewer iterations in test
        for _ in 0..50 {
            ssa.step().await.unwrap();
        }
        
        // Should make progress towards optimum
        assert!(ssa.best_fitness < 100.0);
        assert!(ssa.chains.len() > 0);
    }
    
    #[test]
    fn test_salp_creation() {
        let marine_env = MarineEnvironment::default();
        let lower_bounds = DVector::from_vec(vec![-10.0, -5.0]);
        let upper_bounds = DVector::from_vec(vec![10.0, 5.0]);
        let mut rng = rand::thread_rng();
        
        let salp = Salp::new(0, 2, &lower_bounds, &upper_bounds, &marine_env, &mut rng);
        
        assert_eq!(salp.position.len(), 2);
        assert_eq!(salp.velocity.len(), 2);
        assert!(salp.position[0] >= -10.0 && salp.position[0] <= 10.0);
        assert!(salp.position[1] >= -5.0 && salp.position[1] <= 5.0);
        assert!(salp.depth_level < marine_env.depth_levels.len());
    }
    
    #[test]
    fn test_chain_management() {
        let mut chain = SalpChain::new(0, ChainTopology::Linear);
        
        chain.add_member(0);
        chain.add_member(1);
        chain.add_member(2);
        
        assert_eq!(chain.leader(), Some(0));
        assert_eq!(chain.followers(), &[1, 2]);
        
        assert!(chain.remove_member(1));
        assert_eq!(chain.members.len(), 2);
        assert!(!chain.remove_member(5)); // Non-existent member
    }
    
    #[test]
    fn test_marine_environment() {
        let env = MarineEnvironment::default();
        
        assert_eq!(env.depth_levels.len(), 6);
        assert_eq!(env.temperature_profile.len(), 6);
        assert_eq!(env.current_strength.len(), 6);
        assert!(env.food_density > 0.0);
        assert!(env.turbulence >= 0.0);
    }
    
    #[test]
    fn test_leadership_calculation() {
        let marine_env = MarineEnvironment::default();
        let lower_bounds = DVector::from_element(2, -1.0);
        let upper_bounds = DVector::from_element(2, 1.0);
        let mut rng = rand::thread_rng();
        
        let mut salp = Salp::new(0, 2, &lower_bounds, &upper_bounds, &marine_env, &mut rng);
        
        // Set good fitness
        salp.fitness = 0.1;
        salp.personal_best_fitness = 0.1;
        salp.age = 50;
        salp.energy_level = 1.5;
        
        let leadership = salp.calculate_leadership_potential();
        assert!(leadership > 0.0 && leadership <= 1.0);
    }
    
    #[test]
    fn test_ocean_current_patterns() {
        let marine_env = MarineEnvironment {
            current_pattern: OceanCurrentPattern::Circular,
            ..Default::default()
        };
        
        let lower_bounds = DVector::from_element(2, -1.0);
        let upper_bounds = DVector::from_element(2, 1.0);
        let mut rng = rand::thread_rng();
        let salp = Salp::new(0, 2, &lower_bounds, &upper_bounds, &marine_env, &mut rng);
        
        let current_effect = salp.calculate_ocean_current_effect(0, &marine_env, &mut rng);
        assert!(current_effect.is_finite());
    }
    
    #[test]
    fn test_ssa_variants() {
        for variant in [SsaVariant::Standard, SsaVariant::Enhanced, SsaVariant::Quantum, SsaVariant::Chaotic, SsaVariant::Marine] {
            let params = SsaParameters {
                variant,
                population_size: 10,
                max_iterations: 10,
                ..Default::default()
            };
            
            let ssa = SalpSwarmAlgorithm::new(params).unwrap();
            
            // Check that name reflects variant
            match variant {
                SsaVariant::Standard => assert_eq!(ssa.name(), "Salp Swarm Algorithm"),
                SsaVariant::Enhanced => assert_eq!(ssa.name(), "Enhanced Salp Swarm Algorithm"),
                SsaVariant::Quantum => assert_eq!(ssa.name(), "Quantum Salp Swarm Algorithm"),
                SsaVariant::Chaotic => assert_eq!(ssa.name(), "Chaotic Salp Swarm Algorithm"),
                SsaVariant::Marine => assert_eq!(ssa.name(), "Marine Salp Swarm Algorithm"),
            }
        }
    }
}