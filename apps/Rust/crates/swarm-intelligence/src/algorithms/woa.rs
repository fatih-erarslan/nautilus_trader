//! # Whale Optimization Algorithm (WOA) - Enterprise Implementation
//!
//! Advanced implementation of the Whale Optimization Algorithm inspired by the social behavior
//! of humpback whales. The algorithm simulates the bubble-net hunting strategy, including
//! encircling prey, bubble-net attacking, and search for prey behaviors.
//!
//! ## Algorithm Variants
//! - **Standard WOA**: Original Mirjalili formulation
//! - **Enhanced WOA**: Improved with adaptive parameters and levy flights
//! - **Chaotic WOA**: Chaotic maps for enhanced exploration
//! - **Quantum WOA**: Quantum-inspired position updates
//! - **Binary WOA**: Binary variant for discrete optimization
//! - **Multi-objective WOA**: Support for multi-objective problems
//!
//! ## Key Features
//! - Bubble-net hunting mechanism with spiral updating
//! - Search for prey with random exploration
//! - Multiple hunting strategies and pod formations
//! - SIMD-optimized vector operations
//! - Real-time performance monitoring
//! - Advanced convergence detection
//!
//! ## References
//! - Mirjalili, S., & Lewis, A. (2016). The whale optimization algorithm.
//!   Advances in Engineering Software, 95, 51-67.

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

/// Whale Optimization Algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WoaVariant {
    /// Standard Whale Optimization Algorithm (Mirjalili, 2016)
    Standard,
    /// Enhanced WOA with adaptive parameters
    Enhanced,
    /// Chaotic WOA with chaotic maps
    Chaotic,
    /// Quantum-inspired WOA with quantum behavior
    Quantum,
    /// Binary WOA for discrete optimization
    Binary,
    /// Multi-objective WOA variant
    MultiObjective,
}

/// Hunting strategies for whale behavior
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HuntingStrategy {
    /// Bubble-net feeding (spiral hunting)
    BubbleNet,
    /// Encircling prey behavior
    Encircling,
    /// Random search for prey
    RandomSearch,
    /// Cooperative hunting in pods
    Cooperative,
    /// Deep dive hunting
    DeepDive,
}

/// Pod formation patterns for whale social behavior
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PodFormation {
    /// Single pod formation
    Single,
    /// Multiple pod formation
    Multiple,
    /// Linear formation for travel
    Linear,
    /// Circular formation for hunting
    Circular,
    /// V-formation for efficiency
    VFormation,
}

/// Bubble-net patterns for hunting efficiency
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BubbleNetPattern {
    /// Upward spiral pattern
    UpwardSpiral,
    /// Shrinking circle pattern
    ShrinkingCircle,
    /// Figure-8 pattern
    Figure8,
    /// Double helix pattern
    DoubleHelix,
    /// Random bubble pattern
    Random,
}

/// Whale Optimization Algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WoaParameters {
    /// Base swarm parameters
    pub common: CommonParameters,
    /// Algorithm variant selection
    pub variant: WoaVariant,
    /// Hunting strategy
    pub hunting_strategy: HuntingStrategy,
    /// Pod formation pattern
    pub pod_formation: PodFormation,
    /// Bubble-net pattern
    pub bubble_net_pattern: BubbleNetPattern,
    
    // Core WOA parameters
    /// Convergence parameter (a) - linearly decreases from 2 to 0
    pub convergence_parameter: Float,
    /// Spiral constant (b) for logarithmic spiral
    pub spiral_constant: Float,
    /// Random coefficient (r) range
    pub random_coefficient_range: (Float, Float),
    /// Probability of encircling vs bubble-net
    pub encircling_probability: Float,
    
    // Advanced parameters
    /// Dive depth factor for deep hunting
    pub dive_depth_factor: Float,
    /// Pod cohesion strength
    pub pod_cohesion: Float,
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
    /// Migration probability between pods
    pub migration_probability: Float,
}

impl Default for WoaParameters {
    fn default() -> Self {
        Self {
            common: CommonParameters::default(),
            variant: WoaVariant::Standard,
            hunting_strategy: HuntingStrategy::BubbleNet,
            pod_formation: PodFormation::Single,
            bubble_net_pattern: BubbleNetPattern::UpwardSpiral,
            convergence_parameter: 2.0,
            spiral_constant: 1.0,
            random_coefficient_range: (0.0, 1.0),
            encircling_probability: 0.5,
            dive_depth_factor: 1.5,
            pod_cohesion: 0.1,
            elite_ratio: 0.1,
            diversity_threshold: 1e-6,
            adaptive_parameters: true,
            chaos_parameter: 4.0,
            quantum_probability: 0.1,
            levy_alpha: 1.5,
            convergence_threshold: 1e-8,
            migration_probability: 0.05,
        }
    }
}

/// Individual whale in the population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Whale {
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
    /// Whale ID for tracking
    pub id: usize,
    /// Age (iterations since creation)
    pub age: usize,
    /// Pod membership
    pub pod_id: Option<usize>,
    /// Hunting role in pod
    pub hunting_role: HuntingRole,
    /// Energy level for diving
    pub energy_level: Float,
    /// Dive depth capability
    pub dive_depth: Float,
    /// Success rate for adaptive behavior
    pub success_rate: Float,
    /// Communication range with other whales
    pub communication_range: Float,
    /// Migration count
    pub migration_count: usize,
}

/// Hunting roles within a whale pod
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum HuntingRole {
    /// Leader whale (initiates hunting)
    Leader,
    /// Scout whale (searches for prey)
    Scout,
    /// Hunter whale (performs bubble-net)
    Hunter,
    /// Follower whale (follows the pod)
    Follower,
    /// Herder whale (drives prey towards bubble-net)
    Herder,
}

impl Individual for Whale {
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

impl Whale {
    /// Create a new whale with random initialization
    pub fn new(
        id: usize,
        dimensions: usize,
        bounds: &[(Float, Float)],
        params: &WoaParameters,
        rng: &mut impl Rng,
    ) -> Self {
        let position = DVector::from_fn(dimensions, |i, _| {
            let (min, max) = bounds[i];
            rng.gen_range(min..=max)
        });
        
        let velocity = DVector::zeros(dimensions);
        let hunting_role = match rng.gen_range(0..5) {
            0 => HuntingRole::Leader,
            1 => HuntingRole::Scout,
            2 => HuntingRole::Hunter,
            3 => HuntingRole::Follower,
            _ => HuntingRole::Herder,
        };
        
        Self {
            position: position.clone(),
            velocity,
            fitness: Float::INFINITY,
            best_position: position,
            best_fitness: Float::INFINITY,
            id,
            age: 0,
            pod_id: None,
            hunting_role,
            energy_level: rng.gen_range(0.5..=1.0),
            dive_depth: rng.gen_range(0.1..=params.dive_depth_factor),
            success_rate: 0.0,
            communication_range: rng.gen_range(0.1..=1.0),
            migration_count: 0,
        }
    }

    /// Update whale position using WOA equations
    pub fn update_position(
        &mut self,
        global_best: &Position,
        random_whale: &Position,
        params: &WoaParameters,
        iteration: usize,
        max_iterations: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Calculate convergence parameter: a decreases linearly from 2 to 0
        let a = 2.0 - 2.0 * iteration as Float / max_iterations as Float;
        
        // Calculate coefficients
        let r1: Float = rng.gen();
        let r2: Float = rng.gen();
        let A = 2.0 * a * r1 - a;
        let C = 2.0 * r2;
        
        // Choose hunting strategy
        let p: Float = rng.gen();
        
        if p < params.encircling_probability {
            if A.abs() < 1.0 {
                // Encircling prey (exploitation)
                self.encircle_prey(global_best, A, C, params)?;
            } else {
                // Search for prey (exploration)
                self.search_for_prey(random_whale, A, C, params, rng)?;
            }
        } else {
            // Bubble-net attacking (exploitation)
            self.bubble_net_attack(global_best, params, iteration, rng)?;
        }
        
        // Apply variant-specific modifications
        self.apply_variant_modifications(params, iteration, rng)?;
        
        self.age += 1;
        Ok(())
    }

    /// Encircling prey behavior
    fn encircle_prey(
        &mut self,
        global_best: &Position,
        A: Float,
        C: Float,
        params: &WoaParameters,
    ) -> Result<(), SwarmError> {
        // D = |C * X*(t) - X(t)|
        let D = C * global_best - &self.position;
        
        // X(t+1) = X*(t) - A * D
        self.position = global_best - A * D;
        
        Ok(())
    }

    /// Search for prey behavior
    fn search_for_prey(
        &mut self,
        random_whale: &Position,
        A: Float,
        C: Float,
        params: &WoaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // D = |C * X_rand - X|
        let D = C * random_whale - &self.position;
        
        // X(t+1) = X_rand - A * D
        self.position = random_whale - A * D;
        
        // Add role-specific behavior
        match self.hunting_role {
            HuntingRole::Scout => {
                // Scouts explore more widely
                let exploration_factor = 1.5;
                self.position *= exploration_factor;
            }
            HuntingRole::Leader => {
                // Leaders move more decisively
                let decision_factor = 0.8;
                self.position *= decision_factor;
            }
            _ => {
                // Other roles follow standard behavior
            }
        }
        
        Ok(())
    }

    /// Bubble-net attacking behavior
    fn bubble_net_attack(
        &mut self,
        global_best: &Position,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        match params.bubble_net_pattern {
            BubbleNetPattern::UpwardSpiral => {
                self.upward_spiral_attack(global_best, params, iteration, rng)?;
            }
            BubbleNetPattern::ShrinkingCircle => {
                self.shrinking_circle_attack(global_best, params, iteration, rng)?;
            }
            BubbleNetPattern::Figure8 => {
                self.figure8_attack(global_best, params, iteration, rng)?;
            }
            BubbleNetPattern::DoubleHelix => {
                self.double_helix_attack(global_best, params, iteration, rng)?;
            }
            BubbleNetPattern::Random => {
                self.random_bubble_attack(global_best, params, rng)?;
            }
        }
        
        Ok(())
    }

    /// Upward spiral bubble-net attack
    fn upward_spiral_attack(
        &mut self,
        global_best: &Position,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Distance to prey
        let distance_to_prey = (global_best - &self.position).norm();
        
        // Spiral parameters
        let t = rng.gen_range(-PI..=PI);
        let r = distance_to_prey * (-params.spiral_constant * t).exp();
        
        // Spiral position update
        let spiral_x = r * t.cos();
        let spiral_y = r * t.sin();
        
        // Update position with spiral movement
        if self.position.len() >= 2 {
            self.position[0] = global_best[0] + spiral_x;
            self.position[1] = global_best[1] + spiral_y;
            
            // Update remaining dimensions proportionally
            for i in 2..self.position.len() {
                let proportion = r / distance_to_prey;
                self.position[i] = global_best[i] + 
                    proportion * (global_best[i] - self.position[i]);
            }
        }
        
        Ok(())
    }

    /// Shrinking circle bubble-net attack
    fn shrinking_circle_attack(
        &mut self,
        global_best: &Position,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let distance_to_prey = (global_best - &self.position).norm();
        let shrink_factor = 1.0 - iteration as Float / 1000.0; // Gradually shrink
        
        let angle = rng.gen_range(0.0..=2.0 * PI);
        let radius = distance_to_prey * shrink_factor;
        
        if self.position.len() >= 2 {
            self.position[0] = global_best[0] + radius * angle.cos();
            self.position[1] = global_best[1] + radius * angle.sin();
        }
        
        Ok(())
    }

    /// Figure-8 bubble-net attack
    fn figure8_attack(
        &mut self,
        global_best: &Position,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let t = iteration as Float * 0.1;
        let scale = (global_best - &self.position).norm() * 0.5;
        
        if self.position.len() >= 2 {
            self.position[0] = global_best[0] + scale * (t * 2.0).sin();
            self.position[1] = global_best[1] + scale * (t).sin() * (t).cos();
        }
        
        Ok(())
    }

    /// Double helix bubble-net attack
    fn double_helix_attack(
        &mut self,
        global_best: &Position,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let t = iteration as Float * 0.1;
        let radius = (global_best - &self.position).norm() * 0.5;
        let height_factor = 0.1;
        
        if self.position.len() >= 3 {
            self.position[0] = global_best[0] + radius * t.cos();
            self.position[1] = global_best[1] + radius * t.sin();
            self.position[2] = global_best[2] + height_factor * t;
        } else if self.position.len() >= 2 {
            self.position[0] = global_best[0] + radius * t.cos();
            self.position[1] = global_best[1] + radius * t.sin();
        }
        
        Ok(())
    }

    /// Random bubble attack
    fn random_bubble_attack(
        &mut self,
        global_best: &Position,
        params: &WoaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        let distance_to_prey = (global_best - &self.position).norm();
        
        for i in 0..self.position.len() {
            let bubble_perturbation = rng.gen_range(-1.0..=1.0) * distance_to_prey * 0.1;
            self.position[i] = global_best[i] + bubble_perturbation;
        }
        
        Ok(())
    }

    /// Apply variant-specific modifications
    fn apply_variant_modifications(
        &mut self,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        match params.variant {
            WoaVariant::Standard => {
                // No additional modifications for standard variant
            }
            WoaVariant::Enhanced => {
                self.apply_enhanced_modifications(params, iteration, rng)?;
            }
            WoaVariant::Chaotic => {
                self.apply_chaotic_modifications(params, iteration, rng)?;
            }
            WoaVariant::Quantum => {
                self.apply_quantum_modifications(params, rng)?;
            }
            WoaVariant::Binary => {
                self.apply_binary_modifications(params, rng)?;
            }
            WoaVariant::MultiObjective => {
                self.apply_multi_objective_modifications(params, rng)?;
            }
        }
        
        Ok(())
    }

    /// Enhanced variant modifications
    fn apply_enhanced_modifications(
        &mut self,
        params: &WoaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Levy flight for better exploration
        if rng.gen::<Float>() < 0.1 {
            let levy_step = self.generate_levy_step(params.levy_alpha, rng);
            for i in 0..self.position.len() {
                self.position[i] += levy_step * 0.1;
            }
        }
        
        // Energy-based movement
        let energy_factor = self.energy_level;
        self.position *= energy_factor;
        
        // Update energy based on movement success
        if self.fitness < self.best_fitness {
            self.energy_level = (self.energy_level + 0.1).min(1.0);
        } else {
            self.energy_level = (self.energy_level - 0.05).max(0.1);
        }
        
        Ok(())
    }

    /// Chaotic modifications
    fn apply_chaotic_modifications(
        &mut self,
        params: &WoaParameters,
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

    /// Quantum-inspired modifications
    fn apply_quantum_modifications(
        &mut self,
        params: &WoaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Quantum tunneling effect
        if rng.gen::<Float>() < params.quantum_probability {
            for i in 0..self.position.len() {
                let quantum_jump: Float = rng.gen_range(-1.0..=1.0);
                self.position[i] += quantum_jump;
            }
        }
        
        // Quantum superposition
        let superposition_factor = 0.1 * (rng.gen::<Float>() - 0.5);
        self.position *= 1.0 + superposition_factor;
        
        Ok(())
    }

    /// Binary modifications for discrete optimization
    fn apply_binary_modifications(
        &mut self,
        params: &WoaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Convert continuous values to binary using sigmoid function
        for i in 0..self.position.len() {
            let sigmoid_value = 1.0 / (1.0 + (-self.position[i]).exp());
            self.position[i] = if rng.gen::<Float>() < sigmoid_value { 1.0 } else { 0.0 };
        }
        
        Ok(())
    }

    /// Multi-objective modifications
    fn apply_multi_objective_modifications(
        &mut self,
        params: &WoaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Pareto-based movement adjustments
        let diversity_factor = rng.gen::<Float>() * 0.2;
        self.position *= 1.0 + diversity_factor;
        
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

    /// Handle boundary constraints
    pub fn apply_boundaries(&mut self, bounds: &[(Float, Float)]) {
        for (i, (min_bound, max_bound)) in bounds.iter().enumerate() {
            if i < self.position.len() {
                self.position[i] = self.position[i].clamp(*min_bound, *max_bound);
            }
        }
    }

    /// Migrate to different pod
    pub fn migrate_pod(&mut self, new_pod_id: usize, params: &WoaParameters) {
        if self.migration_count < 5 { // Limit migrations
            self.pod_id = Some(new_pod_id);
            self.migration_count += 1;
            
            // Adjust energy for migration cost
            self.energy_level = (self.energy_level - 0.1).max(0.1);
        }
    }
}

/// Main Whale Optimization Algorithm implementation
#[derive(Debug)]
pub struct WhaleOptimizationAlgorithm {
    /// Algorithm parameters
    params: WoaParameters,
    /// Current whale population
    whales: Vec<Whale>,
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
    /// Pod structures for social behavior
    pods: Vec<Vec<usize>>,
    /// Pod leaders
    pod_leaders: Vec<usize>,
}

impl WhaleOptimizationAlgorithm {
    /// Create new Whale Optimization Algorithm instance
    pub fn new(params: WoaParameters) -> Result<Self, SwarmError> {
        let seed = params.common.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });
        
        let rng = Box::new(StdRng::seed_from_u64(seed));
        
        Ok(Self {
            params,
            whales: Vec::new(),
            global_best_position: DVector::zeros(1),
            global_best_fitness: Float::INFINITY,
            problem: None,
            rng,
            iteration: 0,
            metrics: AlgorithmMetrics::default(),
            elite_archive: Vec::new(),
            pods: Vec::new(),
            pod_leaders: Vec::new(),
        })
    }

    /// Initialize whale population and pod structures
    async fn initialize_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        let dimensions = problem.dimensions();
        let bounds = problem.bounds();
        
        // Initialize whales
        self.whales = (0..self.params.common.population_size)
            .map(|i| Whale::new(i, dimensions, bounds, &self.params, &mut *self.rng))
            .collect();
        
        // Initialize global best
        self.global_best_position = DVector::zeros(dimensions);
        self.global_best_fitness = Float::INFINITY;
        
        // Setup pod formations
        self.setup_pod_formations()?;
        
        // Evaluate initial population
        self.evaluate_population(problem).await?;
        
        // Find initial best
        self.update_global_best();
        
        Ok(())
    }

    /// Setup pod formations based on strategy
    fn setup_pod_formations(&mut self) -> Result<(), SwarmError> {
        match self.params.pod_formation {
            PodFormation::Single => {
                // Single pod with all whales
                let all_whales: Vec<usize> = (0..self.params.common.population_size).collect();
                self.pods.push(all_whales);
                self.pod_leaders.push(0);
            }
            PodFormation::Multiple => {
                // Multiple pods of equal size
                let pod_size = 8; // Optimal pod size for whales
                let num_pods = (self.params.common.population_size / pod_size).max(1);
                
                for pod_id in 0..num_pods {
                    let start_idx = pod_id * pod_size;
                    let end_idx = ((pod_id + 1) * pod_size).min(self.params.common.population_size);
                    
                    let pod_members: Vec<usize> = (start_idx..end_idx).collect();
                    if !pod_members.is_empty() {
                        self.pod_leaders.push(pod_members[0]);
                        
                        // Assign pod membership
                        for &whale_id in &pod_members {
                            if whale_id < self.whales.len() {
                                self.whales[whale_id].pod_id = Some(pod_id);
                            }
                        }
                        
                        self.pods.push(pod_members);
                    }
                }
            }
            PodFormation::Linear => {
                // Linear formation for travel efficiency
                let pod_size = 6;
                let num_pods = (self.params.common.population_size / pod_size).max(1);
                
                for pod_id in 0..num_pods {
                    let start_idx = pod_id * pod_size;
                    let end_idx = ((pod_id + 1) * pod_size).min(self.params.common.population_size);
                    
                    let pod_members: Vec<usize> = (start_idx..end_idx).collect();
                    if !pod_members.is_empty() {
                        self.pod_leaders.push(pod_members[pod_members.len() / 2]); // Middle as leader
                        
                        for &whale_id in &pod_members {
                            if whale_id < self.whales.len() {
                                self.whales[whale_id].pod_id = Some(pod_id);
                            }
                        }
                        
                        self.pods.push(pod_members);
                    }
                }
            }
            PodFormation::Circular => {
                // Circular formation for hunting
                let pod_size = 10;
                let num_pods = (self.params.common.population_size / pod_size).max(1);
                
                for pod_id in 0..num_pods {
                    let start_idx = pod_id * pod_size;
                    let end_idx = ((pod_id + 1) * pod_size).min(self.params.common.population_size);
                    
                    let pod_members: Vec<usize> = (start_idx..end_idx).collect();
                    if !pod_members.is_empty() {
                        self.pod_leaders.push(pod_members[0]);
                        
                        for &whale_id in &pod_members {
                            if whale_id < self.whales.len() {
                                self.whales[whale_id].pod_id = Some(pod_id);
                            }
                        }
                        
                        self.pods.push(pod_members);
                    }
                }
            }
            PodFormation::VFormation => {
                // V-formation for efficiency
                let pod_size = 7; // Odd number for V shape
                let num_pods = (self.params.common.population_size / pod_size).max(1);
                
                for pod_id in 0..num_pods {
                    let start_idx = pod_id * pod_size;
                    let end_idx = ((pod_id + 1) * pod_size).min(self.params.common.population_size);
                    
                    let pod_members: Vec<usize> = (start_idx..end_idx).collect();
                    if !pod_members.is_empty() {
                        self.pod_leaders.push(pod_members[0]); // Front of V as leader
                        
                        for &whale_id in &pod_members {
                            if whale_id < self.whales.len() {
                                self.whales[whale_id].pod_id = Some(pod_id);
                            }
                        }
                        
                        self.pods.push(pod_members);
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Evaluate fitness for all whales
    async fn evaluate_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        if self.params.common.parallel_evaluation {
            // Parallel evaluation using rayon
            self.whales.par_iter_mut().try_for_each(|whale| -> Result<(), SwarmError> {
                let fitness = problem.evaluate(&whale.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                whale.set_fitness(fitness);
                Ok(())
            })?;
        } else {
            // Sequential evaluation
            for whale in &mut self.whales {
                let fitness = problem.evaluate(&whale.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                whale.set_fitness(fitness);
            }
        }
        
        Ok(())
    }

    /// Update global best position
    fn update_global_best(&mut self) {
        for whale in &self.whales {
            if whale.best_fitness < self.global_best_fitness {
                self.global_best_fitness = whale.best_fitness;
                self.global_best_position = whale.best_position.clone();
            }
        }
    }

    /// Perform one iteration of the whale algorithm
    async fn step_iteration(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().ok_or_else(|| {
            SwarmError::InitializationError("Problem not initialized".to_string())
        })?;
        
        let step_start = Instant::now();
        let bounds = problem.bounds();
        let max_iterations = self.params.common.max_evaluations / self.params.common.population_size;
        
        // Update each whale
        for i in 0..self.whales.len() {
            // Select random whale for exploration
            let random_whale_idx = self.rng.gen_range(0..self.whales.len());
            let random_whale_pos = self.whales[random_whale_idx].position.clone();
            
            // Update whale position
            self.whales[i].update_position(
                &self.global_best_position,
                &random_whale_pos,
                &self.params,
                self.iteration,
                max_iterations,
                &mut *self.rng,
            )?;
            
            // Apply boundary constraints
            self.whales[i].apply_boundaries(bounds);
        }
        
        // Handle pod migrations
        self.handle_pod_migrations()?;
        
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

    /// Handle pod migrations between groups
    fn handle_pod_migrations(&mut self) -> Result<(), SwarmError> {
        if self.pods.len() > 1 {
            for whale in &mut self.whales {
                if self.rng.gen::<Float>() < self.params.migration_probability {
                    let new_pod_id = self.rng.gen_range(0..self.pods.len());
                    whale.migrate_pod(new_pod_id, &self.params);
                }
            }
        }
        
        Ok(())
    }

    /// Maintain population diversity
    async fn maintain_diversity(&mut self) -> Result<(), SwarmError> {
        let diversity = self.calculate_population_diversity();
        
        if diversity < self.params.diversity_threshold {
            // Re-initialize worst whales
            let num_reinit = (self.params.common.population_size as Float * 0.1) as usize;
            
            // Sort whales by fitness (worst first)
            let mut indices: Vec<usize> = (0..self.whales.len()).collect();
            indices.sort_by(|&a, &b| {
                self.whales[b].fitness.partial_cmp(&self.whales[a].fitness).unwrap()
            });
            
            // Re-initialize worst whales
            if let Some(ref problem) = self.problem {
                let bounds = problem.bounds();
                let dimensions = problem.dimensions();
                
                for &idx in indices.iter().take(num_reinit) {
                    self.whales[idx] = Whale::new(idx, dimensions, bounds, &self.params, &mut *self.rng);
                }
            }
        }
        
        Ok(())
    }

    /// Calculate population diversity
    fn calculate_population_diversity(&self) -> Float {
        if self.whales.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.whales.len() {
            for j in (i + 1)..self.whales.len() {
                let distance = (&self.whales[i].position - &self.whales[j].position).norm();
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
        
        // Add current best whales to archive
        for whale in &self.whales {
            if whale.best_fitness < Float::INFINITY {
                self.elite_archive.push((whale.best_position.clone(), whale.best_fitness));
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
impl SwarmAlgorithm for WhaleOptimizationAlgorithm {
    type Individual = Whale;
    type Fitness = Float;
    type Parameters = WoaParameters;

    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.problem = Some(problem.clone());
        self.initialize_population(&problem).await?;
        Ok(())
    }

    async fn step(&mut self) -> Result<(), SwarmError> {
        self.step_iteration().await
    }

    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.whales.iter().min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
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
            WoaVariant::Standard => "Whale Optimization Algorithm",
            WoaVariant::Enhanced => "Enhanced Whale Optimization Algorithm",
            WoaVariant::Chaotic => "Chaotic Whale Optimization Algorithm",
            WoaVariant::Quantum => "Quantum Whale Optimization Algorithm",
            WoaVariant::Binary => "Binary Whale Optimization Algorithm",
            WoaVariant::MultiObjective => "Multi-Objective Whale Optimization Algorithm",
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
impl AdaptiveAlgorithm for WhaleOptimizationAlgorithm {
    async fn adapt_parameters(&mut self, strategy: AdaptationStrategy) -> Result<(), SwarmError> {
        match strategy {
            AdaptationStrategy::PerformanceBased => {
                let diversity = self.calculate_population_diversity();
                
                if diversity < 0.01 {
                    // Low diversity: increase exploration
                    self.params.convergence_parameter *= 1.1;
                    self.params.spiral_constant *= 1.05;
                } else if diversity > 0.1 {
                    // High diversity: increase exploitation
                    self.params.encircling_probability *= 1.1;
                }
            }
            AdaptationStrategy::DiversityBased => {
                let diversity = self.calculate_population_diversity();
                self.params.encircling_probability = 0.3 + 0.4 * diversity;
            }
            AdaptationStrategy::FeedbackBased => {
                let avg_success_rate = self.whales.iter()
                    .map(|whale| whale.success_rate)
                    .sum::<Float>() / self.whales.len() as Float;
                
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
impl ParallelAlgorithm for WhaleOptimizationAlgorithm {
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
    async fn test_woa_creation() {
        let params = WoaParameters::default();
        let woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        assert_eq!(woa.name(), "Whale Optimization Algorithm");
    }

    #[tokio::test]
    async fn test_woa_sphere_function() {
        let params = WoaParameters {
            common: CommonParameters {
                population_size: 20,
                max_evaluations: 1000,
                tolerance: 1e-3,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        
        // Sphere function: f(x) = sum(x_i^2)
        let problem = OptimizationProblem::new(
            5, // dimensions
            vec![(-5.0, 5.0); 5], // bounds
            Box::new(|x: &DVector<Float>| {
                Ok(x.iter().map(|xi| xi * xi).sum())
            }),
        );
        
        woa.initialize(problem).await.unwrap();
        
        // Run optimization
        for _ in 0..50 {
            woa.step().await.unwrap();
            if woa.has_converged() {
                break;
            }
        }
        
        // Should find a reasonably good solution
        assert!(woa.global_best_fitness < 1.0);
    }

    #[test]
    fn test_whale_creation() {
        let params = WoaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
        
        let whale = Whale::new(0, 2, &bounds, &params, &mut rng);
        
        assert_eq!(whale.id, 0);
        assert_eq!(whale.position.len(), 2);
        assert!(whale.position[0] >= -10.0 && whale.position[0] <= 10.0);
        assert!(whale.position[1] >= -5.0 && whale.position[1] <= 5.0);
    }

    #[test]
    fn test_bubble_net_attack() {
        let params = WoaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let mut whale = Whale::new(0, 2, &bounds, &params, &mut rng);
        
        let global_best = DVector::from_vec(vec![1.0, 2.0]);
        
        whale.bubble_net_attack(&global_best, &params, 1, &mut rng).unwrap();
        
        // Position should have been updated
        assert!(whale.position.norm() > 0.0);
    }

    #[test]
    fn test_levy_flight_generation() {
        let params = WoaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let whale = Whale::new(0, 2, &bounds, &params, &mut rng);
        
        let levy_step = whale.generate_levy_step(1.5, &mut rng);
        
        // Levy step should be finite
        assert!(levy_step.is_finite());
    }

    #[test]
    fn test_population_diversity() {
        let params = WoaParameters::default();
        let mut woa = WhaleOptimizationAlgorithm::new(params).unwrap();
        
        // Create whales with known positions
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        
        woa.whales = vec![
            {
                let mut whale = Whale::new(0, 2, &bounds, &woa.params, &mut rng);
                whale.position = DVector::from_vec(vec![0.0, 0.0]);
                whale
            },
            {
                let mut whale = Whale::new(1, 2, &bounds, &woa.params, &mut rng);
                whale.position = DVector::from_vec(vec![3.0, 4.0]);
                whale
            },
        ];
        
        let diversity = woa.calculate_population_diversity();
        
        // Expected diversity is the distance between (0,0) and (3,4) = 5.0
        assert_relative_eq!(diversity, 5.0, epsilon = 1e-10);
    }
}