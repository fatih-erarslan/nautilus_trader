//! Grey Wolf Optimization (GWO) algorithm implementation
//!
//! GWO is inspired by the social hierarchy and hunting mechanism of grey wolves.
//! The algorithm mimics the leadership hierarchy (alpha, beta, delta, omega) and
//! collaborative hunting strategies including searching, encircling, and attacking prey.
//!
//! ## Key Features
//! - Pack hierarchy with alpha, beta, delta, and omega wolves
//! - Hunting strategies: searching, encircling, attacking
//! - Territory marking and pack communication
//! - Multiple variants: Standard, Enhanced, Quantum, Chaotic, Levy
//! - Adaptive parameter tuning and SIMD optimization support
//!
//! ## References
//! - Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014). Grey Wolf Optimizer.
//!   Advances in Engineering Software, 69, 46-61.

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
use nalgebra::DVector;
// Parameter validation is done manually

/// GWO algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum GwoVariant {
    /// Standard Grey Wolf Optimizer
    Standard,
    /// Enhanced GWO with exploration/exploitation balance
    Enhanced,
    /// Levy-flight enhanced GWO
    LevyFlight,
    /// Chaotic GWO with chaotic maps
    Chaotic,
    /// Quantum-inspired GWO
    Quantum,
    /// Hybrid GWO with local search
    Hybrid,
}

/// Wolf roles in the pack hierarchy
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WolfRole {
    /// Alpha wolf - best solution (leader)
    Alpha,
    /// Beta wolf - second best solution (subordinate)
    Beta, 
    /// Delta wolf - third best solution (scout, sentinel, elder, hunter, caretaker)
    Delta,
    /// Omega wolf - remaining solutions (lowest hierarchy)
    Omega,
}

/// Hunting strategies for wolf behavior
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum HuntingStrategy {
    /// Standard encircling and attacking
    Standard,
    /// Enhanced cooperative hunting
    Cooperative,
    /// Pack coordination with territory marking
    Territorial,
    /// Adaptive hunting based on prey behavior
    Adaptive,
}

/// Pack communication mechanisms
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PackCommunication {
    /// Basic position sharing
    Basic,
    /// Scent marking for territory
    ScentMarking,
    /// Howling for long-distance communication
    Howling,
    /// Visual signals and body language
    Visual,
}

/// GWO algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GwoParameters {
    /// Population size (pack size)
    pub population_size: usize,
    
    /// Convergence parameter 'a' (decreases from 2 to 0)
    pub convergence_parameter: f64,
    
    /// GWO variant to use
    pub variant: GwoVariant,
    
    /// Hunting strategy
    pub hunting_strategy: HuntingStrategy,
    
    /// Pack communication method
    pub communication: PackCommunication,
    
    /// Enable adaptive convergence parameter
    pub adaptive_convergence: bool,
    
    /// Minimum convergence parameter value
    pub min_convergence: f64,
    
    /// Maximum convergence parameter value  
    pub max_convergence: f64,
    
    /// Enable territory marking
    pub territory_marking: bool,
    
    /// Territory size factor
    pub territory_size: f64,
    
    /// Scent decay rate for territory marking
    pub scent_decay_rate: f64,
    
    /// Enable pack coordination
    pub pack_coordination: bool,
    
    /// Coordination probability
    pub coordination_probability: f64,
    
    /// Enable elite preservation
    pub elite_preservation: bool,
    
    /// Elite preservation ratio
    pub elite_ratio: f64,
    
    /// Enable Levy flight enhancement
    pub levy_flight: bool,
    
    /// Levy flight parameter (beta)
    pub levy_beta: f64,
    
    /// Enable chaotic maps
    pub chaotic_maps: bool,
    
    /// Chaotic map type (logistic, tent, etc.)
    pub chaotic_map_type: String,
    
    /// Enable quantum behavior
    pub quantum_behavior: bool,
    
    /// Quantum well width
    pub quantum_well_width: f64,
    
    /// Enable local search
    pub local_search: bool,
    
    /// Local search probability
    pub local_search_probability: f64,
    
    /// Local search radius
    pub local_search_radius: f64,
    
    /// Enable opposition-based learning
    pub opposition_based: bool,
    
    /// Opposition probability
    pub opposition_probability: f64,
    
    /// Enable crossover operations
    pub crossover_enabled: bool,
    
    /// Crossover probability
    pub crossover_probability: f64,
    
    /// Enable mutation operations
    pub mutation_enabled: bool,
    
    /// Mutation probability
    pub mutation_probability: f64,
    
    /// Mutation strength
    pub mutation_strength: f64,
}

impl Default for GwoParameters {
    fn default() -> Self {
        Self {
            population_size: 30,
            convergence_parameter: 2.0,
            variant: GwoVariant::Standard,
            hunting_strategy: HuntingStrategy::Standard,
            communication: PackCommunication::Basic,
            adaptive_convergence: true,
            min_convergence: 0.0,
            max_convergence: 2.0,
            territory_marking: false,
            territory_size: 1.0,
            scent_decay_rate: 0.95,
            pack_coordination: true,
            coordination_probability: 0.7,
            elite_preservation: true,
            elite_ratio: 0.1,
            levy_flight: false,
            levy_beta: 1.5,
            chaotic_maps: false,
            chaotic_map_type: "logistic".to_string(),
            quantum_behavior: false,
            quantum_well_width: 1.0,
            local_search: false,
            local_search_probability: 0.1,
            local_search_radius: 0.1,
            opposition_based: false,
            opposition_probability: 0.1,
            crossover_enabled: false,
            crossover_probability: 0.7,
            mutation_enabled: false,
            mutation_probability: 0.1,
            mutation_strength: 0.1,
        }
    }
}

/// Grey Wolf individual with role and additional attributes
#[derive(Debug, Clone)]
pub struct GreyWolf {
    /// Basic individual implementation
    individual: BasicIndividual,
    /// Wolf role in the pack hierarchy
    role: WolfRole,
    /// Territory position (for territorial strategies)
    territory_position: Option<Position>,
    /// Scent strength (for scent marking communication)
    scent_strength: f64,
    /// Hunting experience (for adaptive strategies)
    hunting_experience: f64,
    /// Pack coordination factor
    coordination_factor: f64,
}

impl GreyWolf {
    /// Create a new grey wolf
    pub fn new(position: Position, fitness: f64) -> Self {
        let mut individual = BasicIndividual::new(position);
        individual.set_fitness(fitness);
        Self {
            individual,
            role: WolfRole::Omega,
            territory_position: None,
            scent_strength: 1.0,
            hunting_experience: 0.0,
            coordination_factor: 1.0,
        }
    }
    
    /// Get wolf role
    pub fn role(&self) -> WolfRole {
        self.role
    }
    
    /// Set wolf role
    pub fn set_role(&mut self, role: WolfRole) {
        self.role = role;
    }
    
    /// Get territory position
    pub fn territory_position(&self) -> Option<&Position> {
        self.territory_position.as_ref()
    }
    
    /// Set territory position
    pub fn set_territory_position(&mut self, position: Position) {
        self.territory_position = Some(position);
    }
    
    /// Get scent strength
    pub fn scent_strength(&self) -> f64 {
        self.scent_strength
    }
    
    /// Update scent strength
    pub fn update_scent_strength(&mut self, decay_rate: f64) {
        self.scent_strength *= decay_rate;
    }
    
    /// Get hunting experience
    pub fn hunting_experience(&self) -> f64 {
        self.hunting_experience
    }
    
    /// Increase hunting experience
    pub fn gain_experience(&mut self, amount: f64) {
        self.hunting_experience += amount;
    }
    
    /// Get coordination factor
    pub fn coordination_factor(&self) -> f64 {
        self.coordination_factor
    }
    
    /// Set coordination factor
    pub fn set_coordination_factor(&mut self, factor: f64) {
        self.coordination_factor = factor;
    }
}

impl Individual for GreyWolf {
    fn position(&self) -> &Position {
        self.individual.position()
    }
    
    fn position_mut(&mut self) -> &mut Position {
        self.individual.position_mut()
    }
    
    fn fitness(&self) -> &f64 {
        self.individual.fitness()
    }
    
    fn set_fitness(&mut self, fitness: f64) {
        self.individual.set_fitness(fitness);
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.individual.update_position(new_position);
    }
}

/// Grey Wolf Optimizer implementation
#[derive(Debug)]
pub struct GreyWolfOptimizer {
    /// Current population of wolves
    population: Population<GreyWolf>,
    /// Algorithm parameters
    parameters: GwoParameters,
    /// Current iteration
    current_iteration: usize,
    /// Maximum iterations
    max_iterations: usize,
    /// Alpha wolf (best solution)
    alpha: Option<GreyWolf>,
    /// Beta wolf (second best)
    beta: Option<GreyWolf>,
    /// Delta wolf (third best)
    delta: Option<GreyWolf>,
    /// Current convergence parameter 'a'
    current_a: f64,
    /// Problem bounds
    bounds: Vec<(f64, f64)>,
    /// Random number generator
    rng: ThreadRng,
    /// Territory map (for territorial strategies)
    territory_map: Vec<(Position, f64)>, // (position, scent_strength)
    /// Performance metrics
    metrics: AlgorithmMetrics,
    /// Best fitness history
    fitness_history: Vec<f64>,
}

impl GreyWolfOptimizer {
    /// Create a new Grey Wolf Optimizer
    pub fn new(parameters: GwoParameters, bounds: Vec<(f64, f64)>) -> SwarmResult<Self> {
        if parameters.population_size == 0 {
            return Err(SwarmError::parameter("Population size must be positive"));
        }
        if parameters.convergence_parameter < 0.0 {
            return Err(SwarmError::parameter("Convergence parameter must be non-negative"));
        }
        if bounds.is_empty() {
            return Err(SwarmError::parameter("Bounds must not be empty"));
        }
        if parameters.elite_ratio < 0.0 || parameters.elite_ratio > 1.0 {
            return Err(SwarmError::parameter("Elite ratio must be between 0 and 1"));
        }
        
        let population = Population::new();
        
        Ok(Self {
            population,
            parameters,
            current_iteration: 0,
            max_iterations: 1000,
            alpha: None,
            beta: None,
            delta: None,
            current_a: parameters.convergence_parameter,
            bounds,
            rng: thread_rng(),
            territory_map: Vec::new(),
            metrics: AlgorithmMetrics::default(),
            fitness_history: Vec::new(),
        })
    }
    
    /// Initialize the population
    fn initialize_population(&mut self, problem: &OptimizationProblem) -> SwarmResult<()> {
        let mut wolves = Vec::with_capacity(self.parameters.population_size);
        
        for _ in 0..self.parameters.population_size {
            let position = self.generate_random_position();
            let fitness = problem.evaluate(&position)?;
            let wolf = GreyWolf::new(position, fitness);
            wolves.push(wolf);
        }
        
        self.population = Population::new();
        for wolf in wolves {
            self.population.add(wolf);
        }
        self.update_hierarchy();
        
        Ok(())
    }
    
    /// Generate a random position within bounds
    fn generate_random_position(&mut self) -> Position {
        let dimensions = self.bounds.len();
        DVector::from_fn(dimensions, |i, _| {
            let (min, max) = self.bounds[i];
            self.rng.gen_range(min..=max)
        })
    }
    
    /// Update pack hierarchy (alpha, beta, delta wolves)
    fn update_hierarchy(&mut self) {
        let mut wolves: Vec<_> = self.population.individuals.iter().enumerate().collect();
        
        // Sort by fitness (assuming minimization)
        wolves.sort_by(|a, b| a.1.fitness().partial_cmp(b.1.fitness()).unwrap());
        
        // Update roles
        for (i, (idx, _)) in wolves.iter().enumerate() {
            let wolf = &mut self.population.individuals[*idx];
            match i {
                0 => wolf.set_role(WolfRole::Alpha),
                1 => wolf.set_role(WolfRole::Beta),
                2 => wolf.set_role(WolfRole::Delta),
                _ => wolf.set_role(WolfRole::Omega),
            }
        }
        
        // Store alpha, beta, delta
        if let Some((_, wolf)) = wolves.get(0) {
            self.alpha = Some((*wolf).clone());
        }
        if let Some((_, wolf)) = wolves.get(1) {
            self.beta = Some((*wolf).clone());
        }
        if let Some((_, wolf)) = wolves.get(2) {
            self.delta = Some((*wolf).clone());
        }
    }
    
    /// Update convergence parameter 'a'
    fn update_convergence_parameter(&mut self) {
        if self.parameters.adaptive_convergence {
            let progress = self.current_iteration as f64 / self.max_iterations as f64;
            self.current_a = self.parameters.max_convergence * (1.0 - progress);
            self.current_a = self.current_a.max(self.parameters.min_convergence);
        } else {
            let progress = self.current_iteration as f64 / self.max_iterations as f64;
            self.current_a = 2.0 * (1.0 - progress);
        }
    }
    
    /// Perform hunting behavior for a wolf
    fn hunt(&mut self, wolf_idx: usize, problem: &OptimizationProblem) -> SwarmResult<()> {
        let (alpha_pos, beta_pos, delta_pos) = match (&self.alpha, &self.beta, &self.delta) {
            (Some(alpha), Some(beta), Some(delta)) => {
                (alpha.position().clone(), beta.position().clone(), delta.position().clone())
            }
            _ => return Ok(()), // Not enough leaders yet
        };
        
        let wolf = &self.population.individuals[wolf_idx];
        let current_pos = wolf.position().clone();
        
        // Calculate distances and coefficients
        let r1: f64 = self.rng.gen();
        let r2: f64 = self.rng.gen();
        
        let a = self.current_a;
        let c = 2.0 * r2;
        
        // Calculate A and C vectors
        let a_vec = 2.0 * a * r1 - a;
        
        // Position updates based on alpha, beta, delta
        let new_pos = match self.parameters.hunting_strategy {
            HuntingStrategy::Standard => {
                self.standard_hunting(&current_pos, &alpha_pos, &beta_pos, &delta_pos, a_vec, c)
            }
            HuntingStrategy::Cooperative => {
                self.cooperative_hunting(&current_pos, &alpha_pos, &beta_pos, &delta_pos, a_vec, c)
            }
            HuntingStrategy::Territorial => {
                self.territorial_hunting(&current_pos, &alpha_pos, &beta_pos, &delta_pos, a_vec, c)
            }
            HuntingStrategy::Adaptive => {
                self.adaptive_hunting(wolf_idx, &current_pos, &alpha_pos, &beta_pos, &delta_pos, a_vec, c)
            }
        };
        
        // Apply bounds
        let bounded_pos = self.apply_bounds(new_pos);
        
        // Apply variant-specific enhancements
        let enhanced_pos = match self.parameters.variant {
            GwoVariant::Standard => bounded_pos,
            GwoVariant::Enhanced => self.enhanced_position_update(bounded_pos),
            GwoVariant::LevyFlight => self.levy_flight_update(bounded_pos),
            GwoVariant::Chaotic => self.chaotic_update(bounded_pos),
            GwoVariant::Quantum => self.quantum_update(bounded_pos),
            GwoVariant::Hybrid => self.hybrid_update(bounded_pos, problem)?,
        };
        
        // Update wolf position and fitness
        let new_fitness = problem.evaluate(&enhanced_pos)?;
        let wolf = &mut self.population.individuals[wolf_idx];
        wolf.update_position(enhanced_pos);
        wolf.set_fitness(new_fitness);
        
        // Update hunting experience
        if new_fitness < *wolf.fitness() {
            wolf.gain_experience(0.1);
        }
        
        Ok(())
    }
    
    /// Standard hunting strategy
    fn standard_hunting(
        &self,
        current_pos: &Position,
        alpha_pos: &Position,
        beta_pos: &Position,
        delta_pos: &Position,
        a: f64,
        c: f64,
    ) -> Position {
        let dim = current_pos.len();
        DVector::from_fn(dim, |i, _| {
            // Distance from alpha
            let d_alpha = (c * alpha_pos[i] - current_pos[i]).abs();
            let x1 = alpha_pos[i] - a * d_alpha;
            
            // Distance from beta
            let d_beta = (c * beta_pos[i] - current_pos[i]).abs();
            let x2 = beta_pos[i] - a * d_beta;
            
            // Distance from delta
            let d_delta = (c * delta_pos[i] - current_pos[i]).abs();
            let x3 = delta_pos[i] - a * d_delta;
            
            // Average of the three positions
            (x1 + x2 + x3) / 3.0
        })
    }
    
    /// Cooperative hunting with pack coordination
    fn cooperative_hunting(
        &self,
        current_pos: &Position,
        alpha_pos: &Position,
        beta_pos: &Position,
        delta_pos: &Position,
        a: f64,
        c: f64,
    ) -> Position {
        let dim = current_pos.len();
        
        // Weighted influence based on hierarchy
        let alpha_weight = 0.5;
        let beta_weight = 0.3;
        let delta_weight = 0.2;
        
        DVector::from_fn(dim, |i, _| {
            let d_alpha = (c * alpha_pos[i] - current_pos[i]).abs();
            let x1 = alpha_pos[i] - a * d_alpha;
            
            let d_beta = (c * beta_pos[i] - current_pos[i]).abs();
            let x2 = beta_pos[i] - a * d_beta;
            
            let d_delta = (c * delta_pos[i] - current_pos[i]).abs();
            let x3 = delta_pos[i] - a * d_delta;
            
            // Weighted average
            alpha_weight * x1 + beta_weight * x2 + delta_weight * x3
        })
    }
    
    /// Territorial hunting with scent marking
    fn territorial_hunting(
        &self,
        current_pos: &Position,
        alpha_pos: &Position,
        beta_pos: &Position,
        delta_pos: &Position,
        a: f64,
        c: f64,
    ) -> Position {
        // Start with standard hunting
        let mut new_pos = self.standard_hunting(current_pos, alpha_pos, beta_pos, delta_pos, a, c);
        
        // Apply territorial influence
        if self.parameters.territory_marking {
            for (territory_pos, scent) in &self.territory_map {
                let distance = self.euclidean_distance(&new_pos, territory_pos);
                if distance < self.parameters.territory_size {
                    let influence = scent * (1.0 - distance / self.parameters.territory_size);
                    for i in 0..new_pos.len() {
                        new_pos[i] += influence * (territory_pos[i] - new_pos[i]) * 0.1;
                    }
                }
            }
        }
        
        new_pos
    }
    
    /// Adaptive hunting based on wolf experience
    fn adaptive_hunting(
        &self,
        wolf_idx: usize,
        current_pos: &Position,
        alpha_pos: &Position,
        beta_pos: &Position,
        delta_pos: &Position,
        a: f64,
        c: f64,
    ) -> Position {
        let wolf = &self.population.individuals[wolf_idx];
        let experience = wolf.hunting_experience();
        
        // Adjust hunting strategy based on experience
        let adaptive_a = a * (1.0 + experience * 0.1);
        let adaptive_c = c * (1.0 + experience * 0.05);
        
        self.standard_hunting(current_pos, alpha_pos, beta_pos, delta_pos, adaptive_a, adaptive_c)
    }
    
    /// Enhanced position update with exploration/exploitation balance
    fn enhanced_position_update(&mut self, position: Position) -> Position {
        let exploration_factor = self.current_a / 2.0; // High when a is high (exploration)
        
        if exploration_factor > 1.0 {
            // Exploration phase - add random perturbation
            DVector::from_fn(position.len(), |i, _| {
                let noise: f64 = self.rng.sample(StandardNormal);
                position[i] + noise * 0.1 * exploration_factor
            })
        } else {
            // Exploitation phase - refine around current position
            DVector::from_fn(position.len(), |i, _| {
                let noise: f64 = self.rng.sample(StandardNormal);
                position[i] + noise * 0.01 * (1.0 - exploration_factor)
            })
        }
    }
    
    /// Levy flight-based position update
    fn levy_flight_update(&mut self, position: Position) -> Position {
        if !self.parameters.levy_flight {
            return position;
        }
        
        let beta = self.parameters.levy_beta;
        
        DVector::from_fn(position.len(), |i, _| {
            let levy_step = self.generate_levy_step(beta);
            position[i] + levy_step * 0.01
        })
    }
    
    /// Generate Levy flight step
    fn generate_levy_step(&mut self, beta: f64) -> f64 {
        let sigma = ((gamma((1.0 + beta) / 2.0) * (1.0 + beta).sin()) / 
                    (gamma((1.0 + beta) / 2.0) * beta * 2.0_f64.powf((beta - 1.0) / 2.0))).powf(1.0 / beta);
        
        let u: f64 = self.rng.sample(Normal::new(0.0, sigma).unwrap());
        let v: f64 = self.rng.sample(StandardNormal);
        
        u / v.abs().powf(1.0 / beta)
    }
    
    /// Chaotic position update
    fn chaotic_update(&mut self, position: Position) -> Position {
        if !self.parameters.chaotic_maps {
            return position;
        }
        
        let mut chaotic_pos = position;
        
        for i in 0..chaotic_pos.len() {
            let chaos = match self.parameters.chaotic_map_type.as_str() {
                "logistic" => self.logistic_map(chaotic_pos[i]),
                "tent" => self.tent_map(chaotic_pos[i]),
                "sine" => self.sine_map(chaotic_pos[i]),
                _ => chaotic_pos[i],
            };
            chaotic_pos[i] = chaos;
        }
        
        chaotic_pos
    }
    
    /// Logistic chaotic map
    fn logistic_map(&self, x: f64) -> f64 {
        let r = 4.0;
        let normalized_x = (x - self.bounds[0].0) / (self.bounds[0].1 - self.bounds[0].0);
        let chaotic = r * normalized_x * (1.0 - normalized_x);
        self.bounds[0].0 + chaotic * (self.bounds[0].1 - self.bounds[0].0)
    }
    
    /// Tent chaotic map
    fn tent_map(&self, x: f64) -> f64 {
        let normalized_x = (x - self.bounds[0].0) / (self.bounds[0].1 - self.bounds[0].0);
        let chaotic = if normalized_x < 0.5 {
            2.0 * normalized_x
        } else {
            2.0 * (1.0 - normalized_x)
        };
        self.bounds[0].0 + chaotic * (self.bounds[0].1 - self.bounds[0].0)
    }
    
    /// Sine chaotic map
    fn sine_map(&self, x: f64) -> f64 {
        let normalized_x = (x - self.bounds[0].0) / (self.bounds[0].1 - self.bounds[0].0);
        let chaotic = (std::f64::consts::PI * normalized_x).sin();
        self.bounds[0].0 + chaotic * (self.bounds[0].1 - self.bounds[0].0)
    }
    
    /// Quantum-inspired position update
    fn quantum_update(&mut self, position: Position) -> Position {
        if !self.parameters.quantum_behavior {
            return position;
        }
        
        let well_width = self.parameters.quantum_well_width;
        
        DVector::from_fn(position.len(), |i, _| {
            let quantum_state: f64 = self.rng.gen();
            let quantum_shift = well_width * (2.0 * quantum_state - 1.0);
            position[i] + quantum_shift
        })
    }
    
    /// Hybrid update with local search
    fn hybrid_update(&mut self, position: Position, problem: &OptimizationProblem) -> SwarmResult<Position> {
        let mut hybrid_pos = position;
        
        if self.parameters.local_search && self.rng.gen::<f64>() < self.parameters.local_search_probability {
            hybrid_pos = self.local_search(hybrid_pos, problem)?;
        }
        
        if self.parameters.opposition_based && self.rng.gen::<f64>() < self.parameters.opposition_probability {
            hybrid_pos = self.opposition_based_learning(hybrid_pos);
        }
        
        Ok(hybrid_pos)
    }
    
    /// Local search around current position
    fn local_search(&mut self, position: Position, problem: &OptimizationProblem) -> SwarmResult<Position> {
        let mut best_pos = position.clone();
        let mut best_fitness = problem.evaluate(&position)?;
        let radius = self.parameters.local_search_radius;
        
        for _ in 0..10 {  // Limited local search iterations
            let mut candidate = position.clone();
            
            for i in 0..candidate.len() {
                let noise: f64 = self.rng.sample(Normal::new(0.0, radius).unwrap());
                candidate[i] += noise;
            }
            
            let candidate = self.apply_bounds(candidate);
            let fitness = problem.evaluate(&candidate)?;
            
            if fitness < best_fitness {
                best_pos = candidate;
                best_fitness = fitness;
            }
        }
        
        Ok(best_pos)
    }
    
    /// Opposition-based learning
    fn opposition_based_learning(&self, position: Position) -> Position {
        let opposite = DVector::from_fn(position.len(), |i, _| {
            let (min, max) = self.bounds[i];
            min + max - position[i]
        });
        
        self.apply_bounds(opposite)
    }
    
    /// Apply bounds to position
    fn apply_bounds(&self, position: Position) -> Position {
        DVector::from_fn(position.len(), |i, _| {
            let (min, max) = self.bounds[i];
            position[i].clamp(min, max)
        })
    }
    
    /// Calculate Euclidean distance between two positions
    fn euclidean_distance(&self, pos1: &Position, pos2: &Position) -> f64 {
        pos1.iter()
            .zip(pos2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }
    
    /// Update territory map for territorial hunting
    fn update_territory_map(&mut self) {
        if !self.parameters.territory_marking {
            return;
        }
        
        // Decay existing scent marks
        for (_, scent) in &mut self.territory_map {
            *scent *= self.parameters.scent_decay_rate;
        }
        
        // Remove weak scent marks
        self.territory_map.retain(|(_, scent)| *scent > 0.1);
        
        // Add new territory marks from alpha, beta, delta
        if let Some(ref alpha) = self.alpha {
            self.territory_map.push((alpha.position().clone(), 1.0));
        }
        if let Some(ref beta) = self.beta {
            self.territory_map.push((beta.position().clone(), 0.8));
        }
        if let Some(ref delta) = self.delta {
            self.territory_map.push((delta.position().clone(), 0.6));
        }
    }
    
    /// Pack communication simulation
    fn pack_communication(&mut self) {
        match self.parameters.communication {
            PackCommunication::Basic => {
                // Basic position sharing already handled in hunting
            }
            PackCommunication::ScentMarking => {
                self.update_territory_map();
            }
            PackCommunication::Howling => {
                // Long-distance communication - share best positions
                if let Some(ref alpha) = self.alpha {
                    let alpha_pos = alpha.position().clone();
                    for wolf in &mut self.population.individuals {
                        if self.rng.gen::<f64>() < 0.1 {  // 10% chance of hearing howl
                            let influence = 0.05;
                            let mut new_pos = wolf.position().clone();
                            for i in 0..new_pos.len() {
                                new_pos[i] += influence * (alpha_pos[i] - new_pos[i]);
                            }
                            wolf.update_position(new_pos);
                        }
                    }
                }
            }
            PackCommunication::Visual => {
                // Visual signals for nearby wolves
                let visual_range = 2.0;
                for i in 0..self.population.size() {
                    for j in (i + 1)..self.population.size() {
                        let pos1 = self.population.individuals[i].position();
                        let pos2 = self.population.individuals[j].position();
                        let distance = self.euclidean_distance(pos1, pos2);
                        
                        if distance < visual_range {
                            // Share information between nearby wolves
                            let wolf1_fitness = *self.population.individuals[i].fitness();
                            let wolf2_fitness = *self.population.individuals[j].fitness();
                            
                            if wolf1_fitness < wolf2_fitness {
                                // Wolf j learns from wolf i
                                let influence = 0.02;
                                for k in 0..pos1.len() {
                                    self.population.individuals_mut()[j].position_mut()[k] += 
                                        influence * (pos1[k] - pos2[k]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Perform crossover operation
    fn crossover(&mut self, problem: &OptimizationProblem) -> SwarmResult<()> {
        if !self.parameters.crossover_enabled {
            return Ok(());
        }
        
        let pop_size = self.population.size();
        let mut offspring = Vec::new();
        
        for _ in 0..(pop_size as f64 * self.parameters.crossover_probability) as usize {
            let parent1_idx = self.rng.gen_range(0..pop_size);
            let parent2_idx = self.rng.gen_range(0..pop_size);
            
            if parent1_idx != parent2_idx {
                let parent1 = &self.population.individuals[parent1_idx];
                let parent2 = &self.population.individuals[parent2_idx];
                
                let child_pos = self.uniform_crossover(parent1.position(), parent2.position());
                let child_fitness = problem.evaluate(&child_pos)?;
                let child = GreyWolf::new(child_pos, child_fitness);
                offspring.push(child);
            }
        }
        
        // Replace worst individuals with offspring
        if !offspring.is_empty() {
            let mut indices: Vec<usize> = (0..pop_size).collect();
            indices.sort_by(|&a, &b| {
                self.population.individuals[b].fitness()
                    .partial_cmp(self.population.individuals[a].fitness())
                    .unwrap()
            });
            
            for (i, child) in offspring.into_iter().enumerate() {
                if i < indices.len() {
                    self.population.individuals_mut()[indices[i]] = child;
                }
            }
        }
        
        Ok(())
    }
    
    /// Uniform crossover
    fn uniform_crossover(&mut self, parent1: &Position, parent2: &Position) -> Position {
        DVector::from_fn(parent1.len(), |i, _| {
            if self.rng.gen::<bool>() {
                parent1[i]
            } else {
                parent2[i]
            }
        })
    }
    
    /// Perform mutation operation
    fn mutation(&mut self, problem: &OptimizationProblem) -> SwarmResult<()> {
        if !self.parameters.mutation_enabled {
            return Ok(());
        }
        
        for wolf in &mut self.population.individuals {
            if self.rng.gen::<f64>() < self.parameters.mutation_probability {
                let mut new_pos = wolf.position().clone();
                for i in 0..new_pos.len() {
                    if self.rng.gen::<f64>() < 0.1 {  // 10% chance per dimension
                        let noise: f64 = self.rng.sample(StandardNormal);
                        new_pos[i] += noise * self.parameters.mutation_strength;
                    }
                }
                
                let bounded_pos = self.apply_bounds(new_pos);
                wolf.update_position(bounded_pos);
                let new_fitness = problem.evaluate(wolf.position())?;
                wolf.set_fitness(new_fitness);
            }
        }
        
        Ok(())
    }
    
    /// Elite preservation
    fn preserve_elite(&mut self) {
        if !self.parameters.elite_preservation {
            return;
        }
        
        let elite_count = (self.population.size() as f64 * self.parameters.elite_ratio).ceil() as usize;
        
        // Ensure alpha, beta, delta are preserved
        if let (Some(alpha), Some(beta), Some(delta)) = (&self.alpha, &self.beta, &self.delta) {
            // Find and preserve the best solutions
            let mut best_wolves = vec![alpha.clone(), beta.clone(), delta.clone()];
            
            // Sort population by fitness
            let mut indices: Vec<usize> = (0..self.population.size()).collect();
            indices.sort_by(|&a, &b| {
                self.population.individuals[a].fitness()
                    .partial_cmp(self.population.individuals[b].fitness())
                    .unwrap()
            });
            
            // Preserve elite
            for i in 0..elite_count.min(best_wolves.len()) {
                if i < indices.len() {
                    self.population.individuals_mut()[indices[i]] = best_wolves[i].clone();
                }
            }
        }
    }
    
    /// Update performance metrics
    fn update_metrics(&mut self) {
        if let Some(ref alpha) = self.alpha {
            self.fitness_history.push(*alpha.fitness());
            
            // Calculate diversity
            let diversity = self.calculate_diversity();
            
            self.metrics = AlgorithmMetrics {
                current_iteration: self.current_iteration,
                best_fitness: *alpha.fitness(),
                average_fitness: self.calculate_average_fitness(),
                diversity,
                convergence_rate: self.calculate_convergence_rate(),
                exploration_rate: self.current_a / 2.0,
                exploitation_rate: 1.0 - (self.current_a / 2.0),
            };
        }
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.population.size() {
            for j in (i + 1)..self.population.size() {
                let pos1 = self.population.individuals[i].position();
                let pos2 = self.population.individuals[j].position();
                total_distance += self.euclidean_distance(pos1, pos2);
                count += 1;
            }
        }
        
        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate average fitness
    fn calculate_average_fitness(&self) -> f64 {
        let total: f64 = self.population.individuals
            .iter()
            .map(|wolf| *wolf.fitness())
            .sum();
        
        total / self.population.size() as f64
    }
    
    /// Calculate convergence rate
    fn calculate_convergence_rate(&self) -> f64 {
        if self.fitness_history.len() < 2 {
            return 0.0;
        }
        
        let recent_window = 10.min(self.fitness_history.len());
        let recent_fitness: Vec<_> = self.fitness_history
            .iter()
            .rev()
            .take(recent_window)
            .collect();
        
        if recent_fitness.len() < 2 {
            return 0.0;
        }
        
        let improvement = recent_fitness[0] - recent_fitness[recent_fitness.len() - 1];
        improvement.abs() / recent_window as f64
    }
}

#[async_trait]
impl SwarmAlgorithm for GreyWolfOptimizer {
    type Individual = GreyWolf;
    type Fitness = f64;
    type Parameters = GwoParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.max_iterations = 1000; // Default, should be set by caller
        self.current_iteration = 0;
        self.initialize_population(&problem)?;
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        // This is a placeholder - we need access to the problem in step()
        // For now, we'll update internal state only
        self.current_iteration += 1;
        self.update_convergence_parameter();
        self.update_hierarchy();
        
        // Pack communication
        self.pack_communication();
        
        // Territory updates
        self.update_territory_map();
        
        // Update metrics
        self.update_metrics();
        
        Ok(())
    }
    
    /// Custom optimize method that includes problem access
    async fn optimize_with_problem(&mut self, problem: OptimizationProblem, max_iterations: usize) -> SwarmResult<SwarmResult<f64>> {
        self.max_iterations = max_iterations;
        self.initialize(problem.clone()).await?;
        
        for iteration in 0..max_iterations {
            self.current_iteration = iteration;
            self.update_convergence_parameter();
            
            // Hunt for each wolf
            for wolf_idx in 0..self.population.size() {
                self.hunt(wolf_idx, &problem)?;
            }
            
            self.update_hierarchy();
            self.pack_communication();
            self.update_territory_map();
            
            // Apply genetic operations
            self.crossover(&problem)?;
            self.mutation(&problem)?;
            
            // Preserve elite
            self.preserve_elite();
            
            self.update_metrics();
            
            // Check convergence
            if self.has_converged() {
                break;
            }
        }
        
        if let Some(ref alpha) = self.alpha {
            Ok(SwarmResult {
                best_position: alpha.position().clone(),
                best_fitness: *alpha.fitness(),
                iterations: self.current_iteration,
                convergence_history: self.fitness_history.clone(),
                algorithm_name: self.name().to_string(),
            })
        } else {
            Err(SwarmError::OptimizationError(
                "No solution found during optimization".to_string()
            ))
        }
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.alpha.as_ref()
    }
    
    fn get_population(&self) -> &Population<Self::Individual> {
        &self.population
    }
    
    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        &mut self.population
    }
    
    fn has_converged(&self) -> bool {
        // Check if the convergence parameter is very small
        if self.current_a < 0.01 {
            return true;
        }
        
        // Check if improvement has stagnated
        if self.fitness_history.len() >= 50 {
            let recent = &self.fitness_history[self.fitness_history.len() - 50..];
            let improvement = recent[0] - recent[recent.len() - 1];
            if improvement.abs() < 1e-8 {
                return true;
            }
        }
        
        false
    }
    
    fn name(&self) -> &'static str {
        match self.parameters.variant {
            GwoVariant::Standard => "Grey Wolf Optimizer",
            GwoVariant::Enhanced => "Enhanced Grey Wolf Optimizer",
            GwoVariant::LevyFlight => "Levy-Flight Grey Wolf Optimizer",
            GwoVariant::Chaotic => "Chaotic Grey Wolf Optimizer",
            GwoVariant::Quantum => "Quantum Grey Wolf Optimizer",
            GwoVariant::Hybrid => "Hybrid Grey Wolf Optimizer",
        }
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
    
    /// Reset the algorithm to initial state
    async fn reset(&mut self) -> Result<(), SwarmError> {
        self.current_iteration = 0;
        self.alpha = None;
        self.beta = None;
        self.delta = None;
        self.current_a = self.parameters.convergence_parameter;
        self.population = Population::new();
        self.territory_map.clear();
        self.fitness_history.clear();
        self.metrics = AlgorithmMetrics::default();
        Ok(())
    }
    
    /// Clone the algorithm with current parameters
    fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<
        Individual = Self::Individual,
        Fitness = Self::Fitness,
        Parameters = Self::Parameters
    >> {
        Box::new(GreyWolfOptimizer::new(self.parameters.clone(), self.bounds.clone()).unwrap())
    }
}

impl AdaptiveAlgorithm for GreyWolfOptimizer {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        // Adapt based on performance metrics
        let diversity = performance_metrics.diversity;
        let convergence_rate = performance_metrics.convergence_rate;
        
        if diversity < 0.1 {
            // Low diversity - increase exploration
            self.parameters.convergence_parameter *= 1.1;
            self.parameters.mutation_probability *= 1.2;
        }
        
        if convergence_rate < 1e-6 {
            // Poor convergence - adjust parameters
            self.parameters.local_search_probability *= 1.1;
            self.parameters.opposition_probability *= 1.1;
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::Feedback { sensitivity: 0.1 }
    }
    
    fn adapt(&mut self, strategy: AdaptationStrategy) {
        match strategy {
            AdaptationStrategy::Linear { start, end } => {
                let progress = self.current_iteration as f64 / self.max_iterations as f64;
                let value = start + (end - start) * progress;
                self.parameters.convergence_parameter = value;
            }
            AdaptationStrategy::Exponential { initial, decay_rate } => {
                let value = initial * (decay_rate.powf(self.current_iteration as f64));
                self.parameters.convergence_parameter = value;
            }
            AdaptationStrategy::Feedback { sensitivity } => {
                let diversity = self.calculate_diversity();
                let convergence_rate = self.calculate_convergence_rate();
                
                if diversity < 0.1 {
                    // Low diversity - increase exploration
                    self.parameters.convergence_parameter *= 1.0 + sensitivity;
                    self.parameters.mutation_probability *= 1.0 + sensitivity * 2.0;
                }
                
                if convergence_rate < 1e-6 {
                    // Poor convergence - adjust parameters
                    self.parameters.local_search_probability *= 1.0 + sensitivity;
                    self.parameters.opposition_probability *= 1.0 + sensitivity;
                }
            }
            AdaptationStrategy::SuccessBased { increase_factor, decrease_factor } => {
                // Adapt based on recent success rate
                if self.fitness_history.len() >= 10 {
                    let recent_best = self.fitness_history[self.fitness_history.len() - 10];
                    let current_best = self.fitness_history[self.fitness_history.len() - 1];
                    
                    if current_best < recent_best {
                        // Improvement - keep current strategy
                        self.parameters.convergence_parameter *= decrease_factor;
                    } else {
                        // No improvement - try exploration
                        self.parameters.convergence_parameter *= increase_factor;
                    }
                }
            }
            AdaptationStrategy::Custom(_) => {
                // Custom adaptation would be handled by the provided function
                // For now, use feedback strategy as default
                self.adapt(AdaptationStrategy::Feedback { sensitivity: 0.1 });
            }
        }
    }
    
    fn adaptation_history(&self) -> Vec<AdaptationStrategy> {
        vec![] // Could be implemented to track adaptation history
    }
}

impl ParallelAlgorithm for GreyWolfOptimizer {
    async fn parallel_step(&mut self, thread_count: usize) -> Result<(), SwarmError> {
        // For now, just call the regular step method
        // In a full implementation, this would distribute work across threads
        self.step().await
    }
    
    fn parallel_evaluate(&mut self, problem: &OptimizationProblem) -> SwarmResult<()> {
        // Parallel fitness evaluation using rayon
        let fitnesses: Result<Vec<_>, _> = self.population
            .individuals
            .par_iter()
            .map(|wolf| problem.evaluate(wolf.position()))
            .collect();
            
        let fitnesses = fitnesses?;
        
        for (wolf, fitness) in self.population.individuals.iter_mut().zip(fitnesses) {
            wolf.set_fitness(fitness);
        }
        
        Ok(())
    }
    
    fn parallel_update(&mut self, problem: &OptimizationProblem) -> SwarmResult<()> {
        // Parallel position updates
        let updates: Result<Vec<_>, _> = (0..self.population.size())
            .into_par_iter()
            .map(|i| {
                let wolf = &self.population.individuals[i];
                let current_pos = wolf.position().clone();
                
                // Calculate new position (simplified for parallel execution)
                if let (Some(alpha), Some(beta), Some(delta)) = (&self.alpha, &self.beta, &self.delta) {
                    let alpha_pos = alpha.position();
                    let beta_pos = beta.position();
                    let delta_pos = delta.position();
                    
                    let mut rng = thread_rng();
                    let r1: f64 = rng.gen();
                    let r2: f64 = rng.gen();
                    let a = self.current_a;
                    let c = 2.0 * r2;
                    let a_vec = 2.0 * a * r1 - a;
                    
                    let new_pos = self.standard_hunting(&current_pos, alpha_pos, beta_pos, delta_pos, a_vec, c);
                    let bounded_pos = self.apply_bounds(new_pos);
                    let fitness = problem.evaluate(&bounded_pos)?;
                    
                    Ok((bounded_pos, fitness))
                } else {
                    Ok((current_pos, *wolf.fitness()))
                }
            })
            .collect();
            
        let updates = updates?;
        
        for (i, (position, fitness)) in updates.into_iter().enumerate() {
            let wolf = &mut self.population.individuals[i];
            *wolf.position_mut() = position;
            wolf.set_fitness(fitness);
        }
        
        Ok(())
    }
}

// Helper function for gamma function approximation
fn gamma(x: f64) -> f64 {
    // Simplified gamma function approximation
    if x == 1.0 {
        1.0
    } else if x == 0.5 {
        std::f64::consts::PI.sqrt()
    } else {
        (x - 1.0) * gamma(x - 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SwarmResult;
    
    fn sphere_function(position: &[f64]) -> f64 {
        position.iter().map(|x| x * x).sum()
    }
    
    fn rosenbrock_function(position: &[f64]) -> f64 {
        position.windows(2)
            .map(|w| {
                let a = 1.0;
                let b = 100.0;
                let x = w[0];
                let y = w[1];
                (a - x).powi(2) + b * (y - x.powi(2)).powi(2)
            })
            .sum()
    }
    
    #[test]
    fn test_gwo_parameters_default() {
        let params = GwoParameters::default();
        assert_eq!(params.population_size, 30);
        assert_eq!(params.convergence_parameter, 2.0);
        assert!(matches!(params.variant, GwoVariant::Standard));
        assert!(params.adaptive_convergence);
    }
    
    #[test]
    fn test_gwo_parameters_validation() {
        let bounds = vec![(-5.0, 5.0); 2];
        
        // Valid parameters
        let params = GwoParameters::default();
        let result = GreyWolfOptimizer::new(params, bounds.clone());
        assert!(result.is_ok());
        
        // Invalid population size
        let mut params = GwoParameters::default();
        params.population_size = 0;
        let result = GreyWolfOptimizer::new(params, bounds.clone());
        assert!(result.is_err());
        
        // Invalid elite ratio
        let mut params = GwoParameters::default();
        params.elite_ratio = 1.5;
        let result = GreyWolfOptimizer::new(params, bounds);
        assert!(result.is_err());
    }
    
    #[test]
    fn test_grey_wolf_creation() {
        let position = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let fitness = 10.0;
        let wolf = GreyWolf::new(position.clone(), fitness);
        
        assert_eq!(wolf.position(), &position);
        assert_eq!(*wolf.fitness(), fitness);
        assert_eq!(wolf.role(), WolfRole::Omega);
        assert_eq!(wolf.scent_strength(), 1.0);
        assert_eq!(wolf.hunting_experience(), 0.0);
    }
    
    #[test]
    fn test_wolf_role_management() {
        let mut wolf = GreyWolf::new(vec![0.0; 3], 0.0);
        
        assert_eq!(wolf.role(), WolfRole::Omega);
        
        wolf.set_role(WolfRole::Alpha);
        assert_eq!(wolf.role(), WolfRole::Alpha);
        
        wolf.set_role(WolfRole::Beta);
        assert_eq!(wolf.role(), WolfRole::Beta);
        
        wolf.set_role(WolfRole::Delta);
        assert_eq!(wolf.role(), WolfRole::Delta);
    }
    
    #[test]
    fn test_wolf_attributes() {
        let mut wolf = GreyWolf::new(vec![0.0; 3], 0.0);
        
        // Test territory position
        assert!(wolf.territory_position().is_none());
        wolf.set_territory_position(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert!(wolf.territory_position().is_some());
        
        // Test scent strength
        let initial_scent = wolf.scent_strength();
        wolf.update_scent_strength(0.9);
        assert!(wolf.scent_strength() < initial_scent);
        
        // Test hunting experience
        assert_eq!(wolf.hunting_experience(), 0.0);
        wolf.gain_experience(0.5);
        assert_eq!(wolf.hunting_experience(), 0.5);
        
        // Test coordination factor
        wolf.set_coordination_factor(0.8);
        assert_eq!(wolf.coordination_factor(), 0.8);
    }
    
    #[test]
    fn test_gwo_initialization() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 3];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Create a simple problem
        let problem = OptimizationProblem::new(Box::new(sphere_function));
        
        // Test initialization
        let result = gwo.initialize_population(&problem);
        assert!(result.is_ok());
        
        assert_eq!(gwo.population.size(), 30);
        assert!(gwo.alpha.is_some());
        assert!(gwo.beta.is_some());
        assert!(gwo.delta.is_some());
    }
    
    #[test]
    fn test_hierarchy_update() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Create wolves with different fitness values
        let mut wolves = vec![
            GreyWolf::new(DVector::from_vec(vec![0.0, 0.0]), 1.0),  // Best
            GreyWolf::new(DVector::from_vec(vec![1.0, 1.0]), 5.0),  // Worst
            GreyWolf::new(DVector::from_vec(vec![0.5, 0.5]), 2.0),  // Second best
            GreyWolf::new(DVector::from_vec(vec![0.3, 0.3]), 3.0),  // Third best
        ];
        
        gwo.population = Population::new();
        for wolf in wolves {
            gwo.population.add(wolf);
        }
        gwo.update_hierarchy();
        
        // Check alpha (best)
        assert!(gwo.alpha.is_some());
        assert_eq!(*gwo.alpha.as_ref().unwrap().fitness(), 1.0);
        assert_eq!(gwo.alpha.as_ref().unwrap().role(), WolfRole::Alpha);
        
        // Check beta (second best)
        assert!(gwo.beta.is_some());
        assert_eq!(*gwo.beta.as_ref().unwrap().fitness(), 2.0);
        assert_eq!(gwo.beta.as_ref().unwrap().role(), WolfRole::Beta);
        
        // Check delta (third best)
        assert!(gwo.delta.is_some());
        assert_eq!(*gwo.delta.as_ref().unwrap().fitness(), 3.0);
        assert_eq!(gwo.delta.as_ref().unwrap().role(), WolfRole::Delta);
    }
    
    #[test]
    fn test_convergence_parameter_update() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        gwo.max_iterations = 100;
        
        // Test adaptive convergence
        gwo.current_iteration = 0;
        gwo.update_convergence_parameter();
        assert_eq!(gwo.current_a, 2.0);
        
        gwo.current_iteration = 50;
        gwo.update_convergence_parameter();
        assert!(gwo.current_a < 2.0 && gwo.current_a > 0.0);
        
        gwo.current_iteration = 100;
        gwo.update_convergence_parameter();
        assert!(gwo.current_a <= 0.0);
    }
    
    #[test]
    fn test_hunting_strategies() {
        let mut params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        
        // Test different hunting strategies
        let strategies = vec![
            HuntingStrategy::Standard,
            HuntingStrategy::Cooperative,
            HuntingStrategy::Territorial,
            HuntingStrategy::Adaptive,
        ];
        
        for strategy in strategies {
            params.hunting_strategy = strategy;
            let gwo = GreyWolfOptimizer::new(params.clone(), bounds.clone());
            assert!(gwo.is_ok());
        }
    }
    
    #[test]
    fn test_gwo_variants() {
        let bounds = vec![(-5.0, 5.0); 2];
        
        let variants = vec![
            GwoVariant::Standard,
            GwoVariant::Enhanced,
            GwoVariant::LevyFlight,
            GwoVariant::Chaotic,
            GwoVariant::Quantum,
            GwoVariant::Hybrid,
        ];
        
        for variant in variants {
            let mut params = GwoParameters::default();
            params.variant = variant;
            let gwo = GreyWolfOptimizer::new(params, bounds.clone());
            assert!(gwo.is_ok());
        }
    }
    
    #[test]
    fn test_bounds_application() {
        let params = GwoParameters::default();
        let bounds = vec![(-2.0, 2.0), (-3.0, 3.0)];
        let gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Test position within bounds
        let position = vec![1.0, 2.0];
        let bounded = gwo.apply_bounds(position);
        assert_eq!(bounded, vec![1.0, 2.0]);
        
        // Test position outside bounds
        let position = vec![-5.0, 5.0];
        let bounded = gwo.apply_bounds(position);
        assert_eq!(bounded, vec![-2.0, 3.0]);
    }
    
    #[test]
    fn test_distance_calculation() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let pos1 = DVector::from_vec(vec![0.0, 0.0]);
        let pos2 = DVector::from_vec(vec![3.0, 4.0]);
        let distance = gwo.euclidean_distance(&pos1, &pos2);
        assert_eq!(distance, 5.0); // 3-4-5 triangle
    }
    
    #[test]
    fn test_pack_communication() {
        let mut params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        
        let communications = vec![
            PackCommunication::Basic,
            PackCommunication::ScentMarking,
            PackCommunication::Howling,
            PackCommunication::Visual,
        ];
        
        for comm in communications {
            params.communication = comm;
            let mut gwo = GreyWolfOptimizer::new(params.clone(), bounds.clone()).unwrap();
            
            // Initialize with some wolves
            let wolves = vec![
                GreyWolf::new(DVector::from_vec(vec![0.0, 0.0]), 1.0),
                GreyWolf::new(DVector::from_vec(vec![1.0, 1.0]), 2.0),
                GreyWolf::new(DVector::from_vec(vec![2.0, 2.0]), 3.0),
            ];
            gwo.population = Population::new();
        for wolf in wolves {
            gwo.population.add(wolf);
        }
            gwo.update_hierarchy();
            
            // Test communication (should not panic)
            gwo.pack_communication();
        }
    }
    
    #[test]
    fn test_territory_marking() {
        let mut params = GwoParameters::default();
        params.territory_marking = true;
        params.scent_decay_rate = 0.9;
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Add some territory marks
        gwo.territory_map.push((DVector::from_vec(vec![1.0, 1.0]), 1.0));
        gwo.territory_map.push((DVector::from_vec(vec![2.0, 2.0]), 0.5));
        
        let initial_count = gwo.territory_map.len();
        gwo.update_territory_map();
        
        // Check scent decay
        for (_, scent) in &gwo.territory_map {
            assert!(*scent <= 1.0);
        }
    }
    
    #[test]
    fn test_chaotic_maps() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0)];
        let gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let x = 2.0;
        
        // Test logistic map
        let logistic = gwo.logistic_map(x);
        assert!(logistic >= -5.0 && logistic <= 5.0);
        
        // Test tent map
        let tent = gwo.tent_map(x);
        assert!(tent >= -5.0 && tent <= 5.0);
        
        // Test sine map
        let sine = gwo.sine_map(x);
        assert!(sine >= -5.0 && sine <= 5.0);
    }
    
    #[test]
    fn test_opposition_based_learning() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let position = DVector::from_vec(vec![2.0, -3.0]);
        let opposite = gwo.opposition_based_learning(position);
        
        // For bounds (-5, 5), opposite of 2.0 should be -5 + 5 - 2 = -2
        assert_eq!(opposite[0], -2.0);
        // For bounds (-5, 5), opposite of -3.0 should be -5 + 5 - (-3) = 3
        assert_eq!(opposite[1], 3.0);
    }
    
    #[test]
    fn test_crossover() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 3];
        let gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let parent1 = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let parent2 = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        let mut gwo_mut = gwo;
        let child = gwo_mut.uniform_crossover(&parent1, &parent2);
        
        assert_eq!(child.len(), 3);
        for i in 0..child.len() {
            assert!(child[i] == parent1[i] || child[i] == parent2[i]);
        }
    }
    
    #[test]
    fn test_metrics_calculation() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Create wolves with known positions for diversity calculation
        let wolves = vec![
            GreyWolf::new(DVector::from_vec(vec![0.0, 0.0]), 1.0),
            GreyWolf::new(DVector::from_vec(vec![1.0, 0.0]), 2.0),
            GreyWolf::new(DVector::from_vec(vec![0.0, 1.0]), 3.0),
        ];
        gwo.population = Population::new();
        for wolf in wolves {
            gwo.population.add(wolf);
        }
        gwo.update_hierarchy();
        
        let diversity = gwo.calculate_diversity();
        assert!(diversity > 0.0);
        
        let avg_fitness = gwo.calculate_average_fitness();
        assert_eq!(avg_fitness, 2.0); // (1 + 2 + 3) / 3
        
        // Test convergence rate calculation
        gwo.fitness_history = vec![10.0, 8.0, 6.0, 5.0, 4.0];
        let conv_rate = gwo.calculate_convergence_rate();
        assert!(conv_rate >= 0.0);
    }
    
    #[test]
    fn test_convergence_detection() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Test convergence based on parameter a
        gwo.current_a = 2.0;
        assert!(!gwo.has_converged());
        
        gwo.current_a = 0.005;
        assert!(gwo.has_converged());
        
        // Test convergence based on fitness stagnation
        gwo.current_a = 1.0; // Reset a
        gwo.fitness_history = vec![1.0; 60]; // 60 identical fitness values
        assert!(gwo.has_converged());
    }
    
    #[tokio::test]
    async fn test_gwo_sphere_optimization() {
        let mut params = GwoParameters::default();
        params.population_size = 20;
        let bounds = vec![(-5.0, 5.0); 3];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let problem = OptimizationProblem::new(Box::new(sphere_function));
        let result = gwo.optimize_with_problem(problem, 100).await;
        
        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.best_fitness < 1.0); // Should be close to 0 for sphere function
        assert_eq!(solution.best_position.len(), 3);
    }
    
    #[tokio::test]
    async fn test_gwo_rosenbrock_optimization() {
        let mut params = GwoParameters::default();
        params.population_size = 30;
        params.variant = GwoVariant::Enhanced;
        let bounds = vec![(-2.0, 2.0); 4];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let problem = OptimizationProblem::new(Box::new(rosenbrock_function));
        let result = gwo.optimize_with_problem(problem, 200).await;
        
        assert!(result.is_ok());
        let solution = result.unwrap();
        assert!(solution.best_fitness < 100.0); // Rosenbrock is harder, but should improve
        assert_eq!(solution.best_position.len(), 4);
    }
    
    #[tokio::test]
    async fn test_gwo_variants_optimization() {
        let bounds = vec![(-5.0, 5.0); 2];
        
        let variants = vec![
            GwoVariant::Standard,
            GwoVariant::Enhanced,
            GwoVariant::LevyFlight,
            GwoVariant::Quantum,
        ];
        
        for variant in variants {
            let mut params = GwoParameters::default();
            params.variant = variant;
            params.population_size = 15;
            let mut gwo = GreyWolfOptimizer::new(params, bounds.clone()).unwrap();
            
            let problem = OptimizationProblem::new(Box::new(sphere_function));
            let result = gwo.optimize_with_problem(problem, 50).await;
            
            assert!(result.is_ok(), "Failed for variant: {:?}", variant);
        }
    }
    
    #[test]
    fn test_parallel_operations() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Initialize population
        let wolves = vec![
            GreyWolf::new(DVector::from_vec(vec![1.0, 1.0]), 0.0),
            GreyWolf::new(DVector::from_vec(vec![2.0, 2.0]), 0.0),
            GreyWolf::new(DVector::from_vec(vec![3.0, 3.0]), 0.0),
        ];
        gwo.population = Population::new();
        for wolf in wolves {
            gwo.population.add(wolf);
        }
        
        let problem = OptimizationProblem::new(Box::new(sphere_function));
        
        // Test parallel evaluation
        let result = gwo.parallel_evaluate(&problem);
        assert!(result.is_ok());
        
        // Check that fitness values were updated
        for wolf in &gwo.population.individuals {
            assert!(*wolf.fitness() > 0.0); // Sphere function gives positive values for non-zero positions
        }
    }
    
    #[test]
    fn test_adaptive_behavior() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 2];
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Initialize with low diversity population
        let wolves = vec![
            GreyWolf::new(DVector::from_vec(vec![1.0, 1.0]), 1.0),
            GreyWolf::new(DVector::from_vec(vec![1.01, 1.01]), 1.02),
            GreyWolf::new(DVector::from_vec(vec![1.02, 1.02]), 1.04),
        ];
        gwo.population = Population::new();
        for wolf in wolves {
            gwo.population.add(wolf);
        }
        
        let initial_convergence = gwo.parameters.convergence_parameter;
        let initial_mutation = gwo.parameters.mutation_probability;
        
        // Test feedback-based adaptation
        gwo.adapt(AdaptationStrategy::Feedback { sensitivity: 0.1 });
        
        // Parameters should have changed due to low diversity
        assert!(gwo.parameters.convergence_parameter >= initial_convergence);
    }
    
    #[test]
    fn test_enhanced_variants() {
        let bounds = vec![(-5.0, 5.0); 2];
        
        // Test Enhanced variant
        let mut params = GwoParameters::default();
        params.variant = GwoVariant::Enhanced;
        let mut gwo = GreyWolfOptimizer::new(params, bounds.clone()).unwrap();
        
        let position = DVector::from_vec(vec![1.0, 2.0]);
        gwo.current_a = 1.5; // High exploration
        let enhanced = gwo.enhanced_position_update(position.clone());
        assert_ne!(enhanced, position); // Should be modified
        
        // Test Levy Flight variant
        let mut params = GwoParameters::default();
        params.variant = GwoVariant::LevyFlight;
        params.levy_flight = true;
        let mut gwo = GreyWolfOptimizer::new(params, bounds.clone()).unwrap();
        
        let levy_pos = gwo.levy_flight_update(position.clone());
        // Levy flight should modify position
        assert_ne!(levy_pos, position);
        
        // Test Chaotic variant
        let mut params = GwoParameters::default();
        params.variant = GwoVariant::Chaotic;
        params.chaotic_maps = true;
        let mut gwo = GreyWolfOptimizer::new(params, bounds.clone()).unwrap();
        
        let chaotic_pos = gwo.chaotic_update(position.clone());
        assert_eq!(chaotic_pos.len(), position.len());
        
        // Test Quantum variant
        let mut params = GwoParameters::default();
        params.variant = GwoVariant::Quantum;
        params.quantum_behavior = true;
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        let quantum_pos = gwo.quantum_update(position.clone());
        assert_ne!(quantum_pos, position); // Should be modified
    }
    
    #[test]
    fn test_algorithm_name() {
        let bounds = vec![(-5.0, 5.0); 2];
        
        let test_cases = vec![
            (GwoVariant::Standard, "Grey Wolf Optimizer"),
            (GwoVariant::Enhanced, "Enhanced Grey Wolf Optimizer"),
            (GwoVariant::LevyFlight, "Levy-Flight Grey Wolf Optimizer"),
            (GwoVariant::Chaotic, "Chaotic Grey Wolf Optimizer"),
            (GwoVariant::Quantum, "Quantum Grey Wolf Optimizer"),
            (GwoVariant::Hybrid, "Hybrid Grey Wolf Optimizer"),
        ];
        
        for (variant, expected_name) in test_cases {
            let mut params = GwoParameters::default();
            params.variant = variant;
            let gwo = GreyWolfOptimizer::new(params, bounds.clone()).unwrap();
            assert_eq!(gwo.name(), expected_name);
        }
    }
    
    #[test]
    fn test_memory_and_performance() {
        let params = GwoParameters::default();
        let bounds = vec![(-5.0, 5.0); 10]; // Higher dimension
        let mut gwo = GreyWolfOptimizer::new(params, bounds).unwrap();
        
        // Create larger population
        let mut wolves = Vec::new();
        for i in 0..100 {
            let position = DVector::from_element(10, i as f64);
            let fitness = i as f64;
            wolves.push(GreyWolf::new(position, fitness));
        }
        gwo.population = Population::new();
        for wolf in wolves {
            gwo.population.add(wolf);
        }
        
        // Test that operations complete in reasonable time
        let start = std::time::Instant::now();
        gwo.update_hierarchy();
        let duration = start.elapsed();
        assert!(duration.as_millis() < 100); // Should be fast even for larger populations
        
        // Test diversity calculation with larger population
        let diversity = gwo.calculate_diversity();
        assert!(diversity > 0.0);
    }
}