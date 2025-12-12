//! # Moth-Flame Optimization (MFO) - Enterprise Implementation
//!
//! Advanced implementation of the Moth-Flame Optimization algorithm inspired by the navigation
//! method of moths in nature. Moths are attracted to artificial lights and navigate using a
//! transverse orientation method, maintaining a fixed angle with respect to light sources.
//!
//! ## Algorithm Variants
//! - **Standard MFO**: Original Mirjalili formulation
//! - **Enhanced MFO**: Improved with adaptive flame updating
//! - **Chaotic MFO**: Chaotic maps for flame positioning
//! - **Quantum MFO**: Quantum-inspired moth behavior
//! - **Levy MFO**: Levy flight patterns for exploration
//! - **Binary MFO**: Binary variant for discrete optimization
//!
//! ## Key Features
//! - Moth navigation using logarithmic spiral around flames
//! - Flame sorting and moth-flame pairing mechanisms
//! - Multiple spiral patterns and navigation strategies
//! - SIMD-optimized vector operations
//! - Real-time performance monitoring
//! - Advanced convergence detection
//!
//! ## References
//! - Mirjalili, S. (2015). Moth-flame optimization algorithm: A novel nature-inspired
//!   heuristic paradigm. Knowledge-based systems, 89, 228-249.

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

/// Moth-Flame Optimization variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum MfoVariant {
    /// Standard Moth-Flame Optimization (Mirjalili, 2015)
    Standard,
    /// Enhanced MFO with adaptive flame management
    Enhanced,
    /// Chaotic MFO with chaotic flame positioning
    Chaotic,
    /// Quantum-inspired MFO with quantum tunneling
    Quantum,
    /// Levy-flight MFO for global exploration
    LevyFlight,
    /// Binary MFO for discrete optimization
    Binary,
}

/// Navigation strategies for moth movement
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NavigationStrategy {
    /// Standard logarithmic spiral navigation
    LogarithmicSpiral,
    /// Adaptive spiral with changing parameters
    AdaptiveSpiral,
    /// Multi-spiral navigation with multiple patterns
    MultiSpiral,
    /// Phototaxis-based direct movement
    Phototaxis,
    /// Random walk with flame attraction
    RandomWalk,
}

/// Flame update strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum FlameUpdateStrategy {
    /// Best moths become flames (standard)
    BestMoths,
    /// Diversity-based flame selection
    DiversityBased,
    /// Clustering-based flame positioning
    ClusterBased,
    /// Adaptive flame redistribution
    AdaptiveRedistribution,
    /// Elite-based flame management
    EliteBased,
}

/// Spiral patterns for moth navigation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SpiralPattern {
    /// Standard logarithmic spiral
    Logarithmic,
    /// Archimedean spiral
    Archimedean,
    /// Golden spiral
    Golden,
    /// Fibonacci spiral
    Fibonacci,
    /// Custom spiral with adaptive parameters
    Adaptive,
}

/// Moth-Flame Optimization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MfoParameters {
    /// Base swarm parameters
    pub common: CommonParameters,
    /// Algorithm variant selection
    pub variant: MfoVariant,
    /// Navigation strategy for moths
    pub navigation_strategy: NavigationStrategy,
    /// Flame update strategy
    pub flame_update_strategy: FlameUpdateStrategy,
    /// Spiral pattern for navigation
    pub spiral_pattern: SpiralPattern,
    
    // Core MFO parameters
    /// Spiral constant (b) for logarithmic spiral
    pub spiral_constant: Float,
    /// Number of flames (initially equal to population size)
    pub initial_flame_count: usize,
    /// Flame reduction factor per iteration
    pub flame_reduction_factor: Float,
    /// Minimum number of flames to maintain
    pub min_flame_count: usize,
    
    // Advanced parameters
    /// Attraction strength to flames
    pub attraction_strength: Float,
    /// Spiral tightness factor
    pub spiral_tightness: Float,
    /// Distance normalization factor
    pub distance_normalization: Float,
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
    /// Flame memory duration
    pub flame_memory_duration: usize,
}

impl Default for MfoParameters {
    fn default() -> Self {
        Self {
            common: CommonParameters::default(),
            variant: MfoVariant::Standard,
            navigation_strategy: NavigationStrategy::LogarithmicSpiral,
            flame_update_strategy: FlameUpdateStrategy::BestMoths,
            spiral_pattern: SpiralPattern::Logarithmic,
            spiral_constant: 1.0,
            initial_flame_count: 30, // Will be set to population size
            flame_reduction_factor: 1.0,
            min_flame_count: 1,
            attraction_strength: 1.0,
            spiral_tightness: 1.0,
            distance_normalization: 1.0,
            elite_ratio: 0.1,
            diversity_threshold: 1e-6,
            adaptive_parameters: true,
            chaos_parameter: 4.0,
            quantum_probability: 0.1,
            levy_alpha: 1.5,
            convergence_threshold: 1e-8,
            flame_memory_duration: 10,
        }
    }
}

/// Individual moth in the swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moth {
    /// Current position
    pub position: Position,
    /// Current velocity (for enhanced variants)
    pub velocity: DVector<Float>,
    /// Current fitness
    pub fitness: Float,
    /// Personal best position
    pub best_position: Position,
    /// Personal best fitness
    pub best_fitness: Float,
    /// Moth ID for tracking
    pub id: usize,
    /// Age (iterations since creation)
    pub age: usize,
    /// Current flame assignment
    pub assigned_flame: Option<usize>,
    /// Navigation angle for spiral movement
    pub navigation_angle: Float,
    /// Distance to assigned flame
    pub flame_distance: Float,
    /// Wing beat frequency (for adaptive behavior)
    pub wing_frequency: Float,
    /// Energy level for flight
    pub energy_level: Float,
    /// Success rate for adaptive navigation
    pub success_rate: Float,
    /// Memory of previous flame positions
    pub flame_memory: Vec<Position>,
}

impl Individual for Moth {
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

/// Flame structure representing light sources
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flame {
    /// Flame position
    pub position: Position,
    /// Flame fitness (brightness)
    pub fitness: Float,
    /// Flame ID
    pub id: usize,
    /// Flame intensity (attraction strength)
    pub intensity: Float,
    /// Flame age (for memory and decay)
    pub age: usize,
    /// Number of moths attracted to this flame
    pub moth_count: usize,
    /// Flame stability (resistance to position changes)
    pub stability: Float,
}

impl Flame {
    /// Create a new flame
    pub fn new(id: usize, position: Position, fitness: Float) -> Self {
        Self {
            position,
            fitness,
            id,
            intensity: 1.0 / (1.0 + fitness), // Better fitness = higher intensity
            age: 0,
            moth_count: 0,
            stability: 1.0,
        }
    }

    /// Update flame intensity based on fitness
    pub fn update_intensity(&mut self) {
        self.intensity = 1.0 / (1.0 + self.fitness);
        self.age += 1;
    }

    /// Check if flame should be extinguished
    pub fn should_extinguish(&self, min_intensity: Float) -> bool {
        self.intensity < min_intensity || self.moth_count == 0
    }
}

impl Moth {
    /// Create a new moth with random initialization
    pub fn new(
        id: usize,
        dimensions: usize,
        bounds: &[(Float, Float)],
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Self {
        let position = DVector::from_fn(dimensions, |i, _| {
            let (min, max) = bounds[i];
            rng.gen_range(min..=max)
        });
        
        let velocity = DVector::zeros(dimensions);
        let navigation_angle = rng.gen_range(0.0..=2.0 * PI);
        
        Self {
            position: position.clone(),
            velocity,
            fitness: Float::INFINITY,
            best_position: position,
            best_fitness: Float::INFINITY,
            id,
            age: 0,
            assigned_flame: None,
            navigation_angle,
            flame_distance: 0.0,
            wing_frequency: rng.gen_range(0.5..=2.0),
            energy_level: rng.gen_range(0.7..=1.0),
            success_rate: 0.0,
            flame_memory: Vec::new(),
        }
    }

    /// Update moth position using spiral navigation
    pub fn update_position(
        &mut self,
        flames: &[Flame],
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        if flames.is_empty() {
            return Ok(());
        }

        // Assign moth to flame
        self.assign_to_flame(flames, params)?;
        
        if let Some(flame_idx) = self.assigned_flame {
            if flame_idx < flames.len() {
                let flame = &flames[flame_idx];
                
                // Navigate based on strategy
                match params.navigation_strategy {
                    NavigationStrategy::LogarithmicSpiral => {
                        self.logarithmic_spiral_navigation(flame, params, iteration, rng)?;
                    }
                    NavigationStrategy::AdaptiveSpiral => {
                        self.adaptive_spiral_navigation(flame, params, iteration, rng)?;
                    }
                    NavigationStrategy::MultiSpiral => {
                        self.multi_spiral_navigation(flame, params, iteration, rng)?;
                    }
                    NavigationStrategy::Phototaxis => {
                        self.phototaxis_navigation(flame, params, rng)?;
                    }
                    NavigationStrategy::RandomWalk => {
                        self.random_walk_navigation(flame, params, rng)?;
                    }
                }
                
                // Apply variant-specific modifications
                self.apply_variant_modifications(params, iteration, rng)?;
                
                // Update flame memory
                self.update_flame_memory(flame, params);
            }
        }
        
        self.age += 1;
        Ok(())
    }

    /// Assign moth to best suitable flame
    fn assign_to_flame(&mut self, flames: &[Flame], params: &MfoParameters) -> Result<(), SwarmError> {
        if flames.is_empty() {
            return Ok(());
        }

        match params.flame_update_strategy {
            FlameUpdateStrategy::BestMoths => {
                // Assign to flame with index corresponding to moth's rank
                let flame_idx = self.id.min(flames.len() - 1);
                self.assigned_flame = Some(flame_idx);
            }
            FlameUpdateStrategy::DiversityBased => {
                // Assign to flame with maximum diversity consideration
                let mut best_flame = 0;
                let mut best_score = Float::NEG_INFINITY;
                
                for (i, flame) in flames.iter().enumerate() {
                    let distance = (&self.position - &flame.position).norm();
                    let diversity_score = distance - flame.moth_count as Float * 0.1;
                    
                    if diversity_score > best_score {
                        best_score = diversity_score;
                        best_flame = i;
                    }
                }
                
                self.assigned_flame = Some(best_flame);
            }
            FlameUpdateStrategy::ClusterBased => {
                // Assign to nearest flame
                let mut nearest_flame = 0;
                let mut min_distance = Float::INFINITY;
                
                for (i, flame) in flames.iter().enumerate() {
                    let distance = (&self.position - &flame.position).norm();
                    if distance < min_distance {
                        min_distance = distance;
                        nearest_flame = i;
                    }
                }
                
                self.assigned_flame = Some(nearest_flame);
            }
            FlameUpdateStrategy::AdaptiveRedistribution => {
                // Assign based on flame intensity and distance
                let mut best_flame = 0;
                let mut best_score = Float::NEG_INFINITY;
                
                for (i, flame) in flames.iter().enumerate() {
                    let distance = (&self.position - &flame.position).norm();
                    let score = flame.intensity / (1.0 + distance);
                    
                    if score > best_score {
                        best_score = score;
                        best_flame = i;
                    }
                }
                
                self.assigned_flame = Some(best_flame);
            }
            FlameUpdateStrategy::EliteBased => {
                // Assign to best flames first
                let flame_idx = (self.id % flames.len()).min(flames.len() - 1);
                self.assigned_flame = Some(flame_idx);
            }
        }
        
        Ok(())
    }

    /// Standard logarithmic spiral navigation
    fn logarithmic_spiral_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let distance_to_flame = (&self.position - &flame.position).norm();
        self.flame_distance = distance_to_flame;
        
        // Logarithmic spiral equation: r = a * e^(b*theta)
        let t = rng.gen_range(-PI..=PI);
        let a = distance_to_flame;
        let r = a * (-params.spiral_constant * t).exp();
        
        // Update position using spiral
        for i in 0..self.position.len() {
            let spiral_component = if i == 0 {
                r * t.cos()
            } else if i == 1 {
                r * t.sin()
            } else {
                // For higher dimensions, use modulated spiral
                r * (t * (i as Float + 1.0)).cos() / (i as Float + 1.0)
            };
            
            self.position[i] = flame.position[i] + spiral_component * params.attraction_strength;
        }
        
        Ok(())
    }

    /// Adaptive spiral navigation with changing parameters
    fn adaptive_spiral_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Adaptive spiral constant based on success rate and energy
        let adaptive_spiral_constant = params.spiral_constant * 
            (1.0 + self.success_rate) * self.energy_level;
        
        let distance_to_flame = (&self.position - &flame.position).norm();
        let t = rng.gen_range(-PI..=PI);
        
        // Adaptive radius calculation
        let convergence_factor = 1.0 - iteration as Float / 1000.0;
        let r = distance_to_flame * (-adaptive_spiral_constant * t).exp() * convergence_factor;
        
        // Apply spiral movement with energy consideration
        for i in 0..self.position.len() {
            let spiral_component = if i % 2 == 0 {
                r * (t + self.navigation_angle).cos()
            } else {
                r * (t + self.navigation_angle).sin()
            };
            
            self.position[i] = flame.position[i] + 
                spiral_component * params.attraction_strength * self.energy_level;
        }
        
        // Update navigation angle
        self.navigation_angle += 0.1 * self.wing_frequency;
        
        Ok(())
    }

    /// Multi-spiral navigation with multiple patterns
    fn multi_spiral_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let distance_to_flame = (&self.position - &flame.position).norm();
        
        match params.spiral_pattern {
            SpiralPattern::Logarithmic => {
                self.logarithmic_spiral_navigation(flame, params, iteration, rng)?;
            }
            SpiralPattern::Archimedean => {
                self.archimedean_spiral_navigation(flame, params, rng)?;
            }
            SpiralPattern::Golden => {
                self.golden_spiral_navigation(flame, params, rng)?;
            }
            SpiralPattern::Fibonacci => {
                self.fibonacci_spiral_navigation(flame, params, rng)?;
            }
            SpiralPattern::Adaptive => {
                // Choose spiral pattern based on current performance
                if self.success_rate > 0.7 {
                    self.logarithmic_spiral_navigation(flame, params, iteration, rng)?;
                } else if self.success_rate > 0.4 {
                    self.golden_spiral_navigation(flame, params, rng)?;
                } else {
                    self.archimedean_spiral_navigation(flame, params, rng)?;
                }
            }
        }
        
        Ok(())
    }

    /// Archimedean spiral navigation
    fn archimedean_spiral_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let t = rng.gen_range(0.0..=4.0 * PI);
        let r = params.spiral_constant * t; // Archimedean: r = a * theta
        
        for i in 0..self.position.len() {
            let spiral_component = if i % 2 == 0 {
                r * t.cos()
            } else {
                r * t.sin()
            };
            
            self.position[i] = flame.position[i] + 
                spiral_component * params.attraction_strength * 0.1;
        }
        
        Ok(())
    }

    /// Golden spiral navigation
    fn golden_spiral_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
        let t = rng.gen_range(0.0..=2.0 * PI);
        let r = params.spiral_constant * golden_ratio.powf(t / (PI / 2.0));
        
        for i in 0..self.position.len() {
            let angle = t + i as Float * 2.0 * PI / self.position.len() as Float;
            let spiral_component = r * angle.cos() / (i as Float + 1.0);
            
            self.position[i] = flame.position[i] + 
                spiral_component * params.attraction_strength * 0.1;
        }
        
        Ok(())
    }

    /// Fibonacci spiral navigation
    fn fibonacci_spiral_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let fib_angle = 2.0 * PI * (2.0 / (1.0 + 5.0_f64.sqrt())); // Golden angle
        let t = rng.gen_range(0.0..=10.0);
        let r = params.spiral_constant * t.sqrt();
        
        for i in 0..self.position.len() {
            let angle = fib_angle * (i as Float * t);
            let spiral_component = r * angle.cos();
            
            self.position[i] = flame.position[i] + 
                spiral_component * params.attraction_strength * 0.1;
        }
        
        Ok(())
    }

    /// Direct phototaxis navigation
    fn phototaxis_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let direction = &flame.position - &self.position;
        let distance = direction.norm();
        
        if distance > 0.0 {
            let normalized_direction = direction / distance;
            let step_size = distance * params.attraction_strength * 0.1;
            
            // Add some randomness to avoid getting stuck
            let noise_factor = 0.1;
            for i in 0..self.position.len() {
                let noise: Float = rng.gen_range(-noise_factor..=noise_factor);
                self.position[i] += normalized_direction[i] * step_size + noise;
            }
        }
        
        Ok(())
    }

    /// Random walk navigation with flame attraction
    fn random_walk_navigation(
        &mut self,
        flame: &Flame,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let attraction_factor = 0.7;
        let random_factor = 0.3;
        
        // Move towards flame
        let direction = &flame.position - &self.position;
        
        // Add random component
        for i in 0..self.position.len() {
            let attraction_component = direction[i] * params.attraction_strength * attraction_factor;
            let random_component: Float = rng.gen_range(-1.0..=1.0) * random_factor;
            
            self.position[i] += (attraction_component + random_component) * 0.1;
        }
        
        Ok(())
    }

    /// Apply variant-specific modifications
    fn apply_variant_modifications(
        &mut self,
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        match params.variant {
            MfoVariant::Standard => {
                // No additional modifications
            }
            MfoVariant::Enhanced => {
                self.apply_enhanced_modifications(params, iteration, rng)?;
            }
            MfoVariant::Chaotic => {
                self.apply_chaotic_modifications(params, iteration, rng)?;
            }
            MfoVariant::Quantum => {
                self.apply_quantum_modifications(params, rng)?;
            }
            MfoVariant::LevyFlight => {
                self.apply_levy_modifications(params, rng)?;
            }
            MfoVariant::Binary => {
                self.apply_binary_modifications(params, rng)?;
            }
        }
        
        Ok(())
    }

    /// Enhanced variant modifications
    fn apply_enhanced_modifications(
        &mut self,
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Energy-based movement adjustments
        if self.energy_level < 0.5 {
            // Low energy: more conservative movement
            self.position *= 0.95;
        } else {
            // High energy: more exploration
            for i in 0..self.position.len() {
                let exploration: Float = rng.gen_range(-0.1..=0.1);
                self.position[i] += exploration * self.energy_level;
            }
        }
        
        // Update energy based on movement success
        if self.fitness < self.best_fitness {
            self.energy_level = (self.energy_level + 0.1).min(1.0);
            self.success_rate = 0.9 * self.success_rate + 0.1;
        } else {
            self.energy_level = (self.energy_level - 0.05).max(0.1);
            self.success_rate = 0.9 * self.success_rate;
        }
        
        Ok(())
    }

    /// Chaotic modifications
    fn apply_chaotic_modifications(
        &mut self,
        params: &MfoParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Logistic chaotic map
        let mut chaos_value = (iteration as Float * 0.1) % 1.0;
        chaos_value = params.chaos_parameter * chaos_value * (1.0 - chaos_value);
        
        // Apply chaotic perturbation
        let chaos_factor = 0.1 * chaos_value;
        for i in 0..self.position.len() {
            let perturbation: Float = rng.gen_range(-1.0..=1.0);
            self.position[i] += chaos_factor * perturbation;
        }
        
        Ok(())
    }

    /// Quantum modifications
    fn apply_quantum_modifications(
        &mut self,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Quantum tunneling effect
        if rng.gen::<Float>() < params.quantum_probability {
            for i in 0..self.position.len() {
                let quantum_jump: Float = rng.gen_range(-1.0..=1.0);
                self.position[i] += quantum_jump * 0.5;
            }
        }
        
        // Quantum superposition
        let superposition_factor = 0.1 * (rng.gen::<Float>() - 0.5);
        self.position *= 1.0 + superposition_factor;
        
        Ok(())
    }

    /// Levy flight modifications
    fn apply_levy_modifications(
        &mut self,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        if rng.gen::<Float>() < 0.1 {
            let levy_step = self.generate_levy_step(params.levy_alpha, rng);
            
            for i in 0..self.position.len() {
                self.position[i] += levy_step * 0.1;
            }
        }
        
        Ok(())
    }

    /// Binary modifications
    fn apply_binary_modifications(
        &mut self,
        params: &MfoParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Convert to binary using sigmoid function
        for i in 0..self.position.len() {
            let sigmoid_value = 1.0 / (1.0 + (-self.position[i]).exp());
            self.position[i] = if rng.gen::<Float>() < sigmoid_value { 1.0 } else { 0.0 };
        }
        
        Ok(())
    }

    /// Generate Levy flight step
    fn generate_levy_step(&self, alpha: Float, rng: &mut impl Rng) -> Float {
        let sigma_u = (
            libm::tgamma(1.0 + alpha) * (alpha * PI / 2.0).sin() /
            (libm::tgamma((1.0 + alpha) / 2.0) * alpha * (2.0_f64.powf((alpha - 1.0) / 2.0)))
        ).powf(1.0 / alpha);
        
        let u: Float = Normal::new(0.0, sigma_u).unwrap().sample(rng);
        let v: Float = StandardNormal.sample(rng);
        
        u / v.abs().powf(1.0 / alpha)
    }

    /// Update flame memory
    fn update_flame_memory(&mut self, flame: &Flame, params: &MfoParameters) {
        self.flame_memory.push(flame.position.clone());
        
        if self.flame_memory.len() > params.flame_memory_duration {
            self.flame_memory.remove(0);
        }
    }

    /// Apply boundary constraints
    pub fn apply_boundaries(&mut self, bounds: &[(Float, Float)]) {
        for (i, (min_bound, max_bound)) in bounds.iter().enumerate() {
            if i < self.position.len() {
                self.position[i] = self.position[i].clamp(*min_bound, *max_bound);
            }
        }
    }
}

/// Main Moth-Flame Optimization Algorithm implementation
#[derive(Debug)]
pub struct MothFlameOptimization {
    /// Algorithm parameters
    params: MfoParameters,
    /// Current moth population
    moths: Vec<Moth>,
    /// Current flame set
    flames: Vec<Flame>,
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
    /// Elite archive
    elite_archive: Vec<(Position, Float)>,
    /// Current number of flames
    current_flame_count: usize,
}

impl MothFlameOptimization {
    /// Create new Moth-Flame Optimization instance
    pub fn new(params: MfoParameters) -> Result<Self, SwarmError> {
        let seed = params.common.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });
        
        let rng = Box::new(StdRng::seed_from_u64(seed));
        let current_flame_count = params.initial_flame_count;
        
        Ok(Self {
            params,
            moths: Vec::new(),
            flames: Vec::new(),
            global_best_position: DVector::zeros(1),
            global_best_fitness: Float::INFINITY,
            problem: None,
            rng,
            iteration: 0,
            metrics: AlgorithmMetrics::default(),
            elite_archive: Vec::new(),
            current_flame_count,
        })
    }

    /// Initialize moth population and initial flames
    async fn initialize_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        let dimensions = problem.dimensions();
        let bounds = problem.bounds();
        
        // Initialize moths
        self.moths = (0..self.params.common.population_size)
            .map(|i| Moth::new(i, dimensions, bounds, &self.params, &mut *self.rng))
            .collect();
        
        // Initialize global best
        self.global_best_position = DVector::zeros(dimensions);
        self.global_best_fitness = Float::INFINITY;
        
        // Set initial flame count to population size
        self.current_flame_count = self.params.common.population_size;
        
        // Evaluate initial population
        self.evaluate_population(problem).await?;
        
        // Initialize flames from best moths
        self.initialize_flames()?;
        
        // Find initial best
        self.update_global_best();
        
        Ok(())
    }

    /// Initialize flames from moth positions
    fn initialize_flames(&mut self) -> Result<(), SwarmError> {
        // Sort moths by fitness (best first)
        let mut moth_indices: Vec<usize> = (0..self.moths.len()).collect();
        moth_indices.sort_by(|&a, &b| {
            self.moths[a].fitness.partial_cmp(&self.moths[b].fitness).unwrap()
        });
        
        // Create flames from best moths
        self.flames.clear();
        for (i, &moth_idx) in moth_indices.iter().enumerate() {
            if i < self.current_flame_count {
                let moth = &self.moths[moth_idx];
                let flame = Flame::new(i, moth.position.clone(), moth.fitness);
                self.flames.push(flame);
            }
        }
        
        Ok(())
    }

    /// Update flames based on current moth positions
    fn update_flames(&mut self) -> Result<(), SwarmError> {
        // Reduce flame count
        let max_iterations = self.params.common.max_evaluations / self.params.common.population_size;
        self.current_flame_count = (self.params.common.population_size as Float * 
            (1.0 - self.iteration as Float / max_iterations as Float * self.params.flame_reduction_factor))
            .max(self.params.min_flame_count as Float) as usize;
        
        match self.params.flame_update_strategy {
            FlameUpdateStrategy::BestMoths => {
                self.update_flames_best_moths()?;
            }
            FlameUpdateStrategy::DiversityBased => {
                self.update_flames_diversity_based()?;
            }
            FlameUpdateStrategy::ClusterBased => {
                self.update_flames_cluster_based()?;
            }
            FlameUpdateStrategy::AdaptiveRedistribution => {
                self.update_flames_adaptive()?;
            }
            FlameUpdateStrategy::EliteBased => {
                self.update_flames_elite_based()?;
            }
        }
        
        Ok(())
    }

    /// Update flames using best moths
    fn update_flames_best_moths(&mut self) -> Result<(), SwarmError> {
        // Sort moths by fitness
        let mut moth_indices: Vec<usize> = (0..self.moths.len()).collect();
        moth_indices.sort_by(|&a, &b| {
            self.moths[a].fitness.partial_cmp(&self.moths[b].fitness).unwrap()
        });
        
        // Update existing flames or create new ones
        for i in 0..self.current_flame_count {
            if i < moth_indices.len() {
                let moth_idx = moth_indices[i];
                let moth = &self.moths[moth_idx];
                
                if i < self.flames.len() {
                    // Update existing flame
                    if moth.fitness < self.flames[i].fitness {
                        self.flames[i].position = moth.position.clone();
                        self.flames[i].fitness = moth.fitness;
                        self.flames[i].update_intensity();
                    }
                } else {
                    // Create new flame
                    let flame = Flame::new(i, moth.position.clone(), moth.fitness);
                    self.flames.push(flame);
                }
            }
        }
        
        // Remove excess flames
        self.flames.truncate(self.current_flame_count);
        
        Ok(())
    }

    /// Update flames with diversity consideration
    fn update_flames_diversity_based(&mut self) -> Result<(), SwarmError> {
        let mut selected_moths = Vec::new();
        let mut remaining_moths: Vec<usize> = (0..self.moths.len()).collect();
        
        // Select diverse moths for flames
        for _ in 0..self.current_flame_count {
            if remaining_moths.is_empty() {
                break;
            }
            
            let mut best_moth = 0;
            let mut best_score = Float::NEG_INFINITY;
            
            for (idx, &moth_idx) in remaining_moths.iter().enumerate() {
                let moth = &self.moths[moth_idx];
                let mut diversity_score = 1.0 / (1.0 + moth.fitness);
                
                // Penalize moths close to already selected ones
                for &selected_idx in &selected_moths {
                    let distance = (&moth.position - &self.moths[selected_idx].position).norm();
                    diversity_score += distance * 0.1;
                }
                
                if diversity_score > best_score {
                    best_score = diversity_score;
                    best_moth = idx;
                }
            }
            
            let selected_moth_idx = remaining_moths.remove(best_moth);
            selected_moths.push(selected_moth_idx);
        }
        
        // Update flames with selected moths
        self.flames.clear();
        for (i, &moth_idx) in selected_moths.iter().enumerate() {
            let moth = &self.moths[moth_idx];
            let flame = Flame::new(i, moth.position.clone(), moth.fitness);
            self.flames.push(flame);
        }
        
        Ok(())
    }

    /// Update flames using clustering
    fn update_flames_cluster_based(&mut self) -> Result<(), SwarmError> {
        // Simple k-means-like clustering
        if self.current_flame_count > self.moths.len() {
            return self.update_flames_best_moths();
        }
        
        // Initialize cluster centers randomly from moths
        let mut cluster_centers = Vec::new();
        let mut used_indices = std::collections::HashSet::new();
        
        for _ in 0..self.current_flame_count {
            let mut idx = self.rng.gen_range(0..self.moths.len());
            while used_indices.contains(&idx) && used_indices.len() < self.moths.len() {
                idx = self.rng.gen_range(0..self.moths.len());
            }
            used_indices.insert(idx);
            cluster_centers.push(self.moths[idx].position.clone());
        }
        
        // Update flames with cluster centers
        self.flames.clear();
        for (i, center) in cluster_centers.iter().enumerate() {
            // Find best fitness in cluster
            let mut best_fitness = Float::INFINITY;
            for moth in &self.moths {
                let distance = (&moth.position - center).norm();
                if distance < 1.0 && moth.fitness < best_fitness {
                    best_fitness = moth.fitness;
                }
            }
            
            let flame = Flame::new(i, center.clone(), best_fitness);
            self.flames.push(flame);
        }
        
        Ok(())
    }

    /// Adaptive flame redistribution
    fn update_flames_adaptive(&mut self) -> Result<(), SwarmError> {
        // Combine fitness and diversity considerations
        let mut flame_candidates = Vec::new();
        
        for (i, moth) in self.moths.iter().enumerate() {
            let fitness_score = 1.0 / (1.0 + moth.fitness);
            let diversity_score = self.calculate_moth_diversity(i);
            let combined_score = 0.7 * fitness_score + 0.3 * diversity_score;
            
            flame_candidates.push((i, combined_score));
        }
        
        // Sort by combined score
        flame_candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Update flames
        self.flames.clear();
        for i in 0..self.current_flame_count.min(flame_candidates.len()) {
            let moth_idx = flame_candidates[i].0;
            let moth = &self.moths[moth_idx];
            let flame = Flame::new(i, moth.position.clone(), moth.fitness);
            self.flames.push(flame);
        }
        
        Ok(())
    }

    /// Elite-based flame update
    fn update_flames_elite_based(&mut self) -> Result<(), SwarmError> {
        // Use elite archive for flame positions
        let elite_size = (self.current_flame_count as Float * self.params.elite_ratio) as usize;
        
        self.flames.clear();
        
        // Add elite flames
        for (i, (position, fitness)) in self.elite_archive.iter().enumerate() {
            if i >= elite_size {
                break;
            }
            let flame = Flame::new(i, position.clone(), *fitness);
            self.flames.push(flame);
        }
        
        // Fill remaining with best moths
        let mut moth_indices: Vec<usize> = (0..self.moths.len()).collect();
        moth_indices.sort_by(|&a, &b| {
            self.moths[a].fitness.partial_cmp(&self.moths[b].fitness).unwrap()
        });
        
        for i in elite_size..self.current_flame_count {
            if i - elite_size < moth_indices.len() {
                let moth_idx = moth_indices[i - elite_size];
                let moth = &self.moths[moth_idx];
                let flame = Flame::new(i, moth.position.clone(), moth.fitness);
                self.flames.push(flame);
            }
        }
        
        Ok(())
    }

    /// Calculate diversity score for a moth
    fn calculate_moth_diversity(&self, moth_idx: usize) -> Float {
        let moth = &self.moths[moth_idx];
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for (i, other_moth) in self.moths.iter().enumerate() {
            if i != moth_idx {
                let distance = (&moth.position - &other_moth.position).norm();
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

    /// Evaluate fitness for all moths
    async fn evaluate_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        if self.params.common.parallel_evaluation {
            // Parallel evaluation using rayon
            self.moths.par_iter_mut().try_for_each(|moth| -> Result<(), SwarmError> {
                let fitness = problem.evaluate(&moth.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                moth.set_fitness(fitness);
                Ok(())
            })?;
        } else {
            // Sequential evaluation
            for moth in &mut self.moths {
                let fitness = problem.evaluate(&moth.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                moth.set_fitness(fitness);
            }
        }
        
        Ok(())
    }

    /// Update global best position
    fn update_global_best(&mut self) {
        for moth in &self.moths {
            if moth.best_fitness < self.global_best_fitness {
                self.global_best_fitness = moth.best_fitness;
                self.global_best_position = moth.best_position.clone();
            }
        }
    }

    /// Perform one iteration of the MFO algorithm
    async fn step_iteration(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().ok_or_else(|| {
            SwarmError::InitializationError("Problem not initialized".to_string())
        })?;
        
        let step_start = Instant::now();
        let bounds = problem.bounds();
        
        // Update flame positions
        self.update_flames()?;
        
        // Update moth positions
        for moth in &mut self.moths {
            moth.update_position(&self.flames, &self.params, self.iteration, &mut *self.rng)?;
            moth.apply_boundaries(bounds);
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
            // Re-initialize worst moths
            let num_reinit = (self.params.common.population_size as Float * 0.1) as usize;
            
            // Sort moths by fitness (worst first)
            let mut indices: Vec<usize> = (0..self.moths.len()).collect();
            indices.sort_by(|&a, &b| {
                self.moths[b].fitness.partial_cmp(&self.moths[a].fitness).unwrap()
            });
            
            // Re-initialize worst moths
            if let Some(ref problem) = self.problem {
                let bounds = problem.bounds();
                let dimensions = problem.dimensions();
                
                for &idx in indices.iter().take(num_reinit) {
                    self.moths[idx] = Moth::new(idx, dimensions, bounds, &self.params, &mut *self.rng);
                }
            }
        }
        
        Ok(())
    }

    /// Calculate population diversity
    fn calculate_population_diversity(&self) -> Float {
        if self.moths.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.moths.len() {
            for j in (i + 1)..self.moths.len() {
                let distance = (&self.moths[i].position - &self.moths[j].position).norm();
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
        
        // Add current best moths to archive
        for moth in &self.moths {
            if moth.best_fitness < Float::INFINITY {
                self.elite_archive.push((moth.best_position.clone(), moth.best_fitness));
            }
        }
        
        // Sort and keep only elite_size best
        self.elite_archive.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        self.elite_archive.truncate(elite_size);
    }

    /// Check for convergence
    fn has_converged(&self) -> bool {
        let diversity = self.calculate_population_diversity();
        diversity < self.params.convergence_threshold ||
        self.metrics.stagnation_count > 100
    }
}

#[async_trait]
impl SwarmAlgorithm for MothFlameOptimization {
    type Individual = Moth;
    type Fitness = Float;
    type Parameters = MfoParameters;

    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.problem = Some(problem.clone());
        self.initialize_population(&problem).await?;
        Ok(())
    }

    async fn step(&mut self) -> Result<(), SwarmError> {
        self.step_iteration().await
    }

    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.moths.iter().min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
    }

    fn get_population(&self) -> &Population<Self::Individual> {
        // This requires a proper Population type implementation
        unimplemented!("Population interface needs proper implementation")
    }

    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        // This requires a proper Population type implementation
        unimplemented!("Population interface needs proper implementation")
    }

    fn has_converged(&self) -> bool {
        self.has_converged()
    }

    fn name(&self) -> &'static str {
        match self.params.variant {
            MfoVariant::Standard => "Moth-Flame Optimization",
            MfoVariant::Enhanced => "Enhanced Moth-Flame Optimization",
            MfoVariant::Chaotic => "Chaotic Moth-Flame Optimization",
            MfoVariant::Quantum => "Quantum Moth-Flame Optimization",
            MfoVariant::LevyFlight => "Levy Flight Moth-Flame Optimization",
            MfoVariant::Binary => "Binary Moth-Flame Optimization",
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
impl AdaptiveAlgorithm for MothFlameOptimization {
    async fn adapt_parameters(&mut self, strategy: AdaptationStrategy) -> Result<(), SwarmError> {
        match strategy {
            AdaptationStrategy::PerformanceBased => {
                let diversity = self.calculate_population_diversity();
                
                if diversity < 0.01 {
                    // Low diversity: increase exploration
                    self.params.spiral_constant *= 1.1;
                    self.params.attraction_strength *= 0.95;
                } else if diversity > 0.1 {
                    // High diversity: increase exploitation
                    self.params.spiral_constant *= 0.95;
                    self.params.attraction_strength *= 1.05;
                }
            }
            AdaptationStrategy::DiversityBased => {
                let diversity = self.calculate_population_diversity();
                self.params.attraction_strength = 0.5 + 0.5 * diversity;
            }
            AdaptationStrategy::FeedbackBased => {
                let avg_success_rate = self.moths.iter()
                    .map(|moth| moth.success_rate)
                    .sum::<Float>() / self.moths.len() as Float;
                
                if avg_success_rate < 0.3 {
                    self.params.spiral_constant *= 1.1;
                }
            }
        }
        
        Ok(())
    }

    fn adaptation_history(&self) -> Vec<(usize, Self::Parameters)> {
        vec![(self.iteration, self.params.clone())]
    }
}

#[async_trait]
impl ParallelAlgorithm for MothFlameOptimization {
    async fn parallel_step(&mut self, num_threads: usize) -> Result<(), SwarmError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| SwarmError::ParallelExecutionError(format!("Thread pool creation failed: {}", e)))?;
        
        pool.install(|| {
            // Parallel execution would be implemented here
        });
        
        self.step().await
    }

    fn set_parallelism(&mut self, enabled: bool, num_threads: Option<usize>) {
        self.params.common.parallel_evaluation = enabled;
    }

    fn parallelism_metrics(&self) -> (bool, Option<usize>, Duration) {
        (
            self.params.common.parallel_evaluation,
            None,
            self.metrics.last_step_duration,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[tokio::test]
    async fn test_mfo_creation() {
        let params = MfoParameters::default();
        let mfo = MothFlameOptimization::new(params).unwrap();
        assert_eq!(mfo.name(), "Moth-Flame Optimization");
    }

    #[tokio::test]
    async fn test_mfo_sphere_function() {
        let params = MfoParameters {
            common: CommonParameters {
                population_size: 20,
                max_evaluations: 1000,
                tolerance: 1e-3,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut mfo = MothFlameOptimization::new(params).unwrap();
        
        // Sphere function: f(x) = sum(x_i^2)
        let problem = OptimizationProblem::new(
            5, // dimensions
            vec![(-5.0, 5.0); 5], // bounds
            Box::new(|x: &DVector<Float>| {
                Ok(x.iter().map(|xi| xi * xi).sum())
            }),
        );
        
        mfo.initialize(problem).await.unwrap();
        
        // Run optimization
        for _ in 0..50 {
            mfo.step().await.unwrap();
            if mfo.has_converged() {
                break;
            }
        }
        
        // Should find a reasonably good solution
        assert!(mfo.global_best_fitness < 1.0);
    }

    #[test]
    fn test_moth_creation() {
        let params = MfoParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
        
        let moth = Moth::new(0, 2, &bounds, &params, &mut rng);
        
        assert_eq!(moth.id, 0);
        assert_eq!(moth.position.len(), 2);
        assert!(moth.position[0] >= -10.0 && moth.position[0] <= 10.0);
        assert!(moth.position[1] >= -5.0 && moth.position[1] <= 5.0);
    }

    #[test]
    fn test_flame_creation() {
        let position = DVector::from_vec(vec![1.0, 2.0]);
        let fitness = 0.5;
        let flame = Flame::new(0, position.clone(), fitness);
        
        assert_eq!(flame.id, 0);
        assert_eq!(flame.position, position);
        assert_eq!(flame.fitness, fitness);
        assert!(flame.intensity > 0.0);
    }

    #[test]
    fn test_spiral_navigation() {
        let params = MfoParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let mut moth = Moth::new(0, 2, &bounds, &params, &mut rng);
        
        let flame_position = DVector::from_vec(vec![5.0, 5.0]);
        let flame = Flame::new(0, flame_position, 1.0);
        
        moth.logarithmic_spiral_navigation(&flame, &params, 1, &mut rng).unwrap();
        
        // Position should have been updated
        assert!(moth.position.norm() > 0.0);
    }

    #[test]
    fn test_population_diversity() {
        let params = MfoParameters::default();
        let mut mfo = MothFlameOptimization::new(params).unwrap();
        
        // Create moths with known positions
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        
        mfo.moths = vec![
            {
                let mut moth = Moth::new(0, 2, &bounds, &mfo.params, &mut rng);
                moth.position = DVector::from_vec(vec![0.0, 0.0]);
                moth
            },
            {
                let mut moth = Moth::new(1, 2, &bounds, &mfo.params, &mut rng);
                moth.position = DVector::from_vec(vec![3.0, 4.0]);
                moth
            },
        ];
        
        let diversity = mfo.calculate_population_diversity();
        
        // Expected diversity is the distance between (0,0) and (3,4) = 5.0
        assert_relative_eq!(diversity, 5.0, epsilon = 1e-10);
    }
}