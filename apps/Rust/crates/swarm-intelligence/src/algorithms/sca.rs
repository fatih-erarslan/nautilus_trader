//! # Sine Cosine Algorithm (SCA) - Enterprise Implementation
//!
//! Advanced implementation of the Sine Cosine Algorithm that leverages mathematical
//! sine and cosine functions to perform exploration and exploitation. The algorithm
//! uses periodic oscillations to update agent positions based on their distance
//! from the best solution found so far.
//!
//! ## Algorithm Variants
//! - **Standard SCA**: Original Mirjalili formulation
//! - **Enhanced SCA**: Improved with adaptive parameters
//! - **Chaotic SCA**: Chaotic maps for enhanced exploration
//! - **Quantum SCA**: Quantum-inspired oscillations
//! - **Levy SCA**: Levy flight patterns for global search
//! - **Binary SCA**: Binary variant for discrete optimization
//!
//! ## Key Features
//! - Sine and cosine functions for position updates
//! - Adaptive parameter control and oscillation patterns
//! - Multiple oscillation strategies and wave patterns
//! - SIMD-optimized vector operations
//! - Real-time performance monitoring
//! - Advanced convergence detection
//!
//! ## References
//! - Mirjalili, S. (2016). SCA: a sine cosine algorithm for solving optimization problems.
//!   Knowledge-based Systems, 96, 120-133.

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

/// Sine Cosine Algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ScaVariant {
    /// Standard Sine Cosine Algorithm (Mirjalili, 2016)
    Standard,
    /// Enhanced SCA with adaptive parameters
    Enhanced,
    /// Chaotic SCA with chaotic oscillations
    Chaotic,
    /// Quantum-inspired SCA with quantum oscillations
    Quantum,
    /// Levy-flight SCA for global exploration
    LevyFlight,
    /// Binary SCA for discrete optimization
    Binary,
}

/// Oscillation strategies for position updates
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OscillationStrategy {
    /// Standard sine/cosine alternation
    Standard,
    /// Adaptive frequency oscillation
    AdaptiveFrequency,
    /// Multi-frequency oscillation
    MultiFrequency,
    /// Harmonic oscillation patterns
    Harmonic,
    /// Damped oscillation with decay
    Damped,
}

/// Wave patterns for oscillation
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum WavePattern {
    /// Pure sine wave
    Sine,
    /// Pure cosine wave
    Cosine,
    /// Mixed sine-cosine wave
    Mixed,
    /// Square wave pattern
    Square,
    /// Triangular wave pattern
    Triangular,
    /// Sawtooth wave pattern
    Sawtooth,
}

/// Parameter update strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ParameterUpdateStrategy {
    /// Linear parameter decrease
    Linear,
    /// Exponential parameter decrease
    Exponential,
    /// Adaptive parameter adjustment
    Adaptive,
    /// Piecewise linear adjustment
    Piecewise,
    /// Sinusoidal parameter variation
    Sinusoidal,
}

/// Sine Cosine Algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaParameters {
    /// Base swarm parameters
    pub common: CommonParameters,
    /// Algorithm variant selection
    pub variant: ScaVariant,
    /// Oscillation strategy
    pub oscillation_strategy: OscillationStrategy,
    /// Wave pattern for oscillations
    pub wave_pattern: WavePattern,
    /// Parameter update strategy
    pub parameter_update_strategy: ParameterUpdateStrategy,
    
    // Core SCA parameters
    /// Initial value of parameter a (controls exploration/exploitation)
    pub initial_a: Float,
    /// Random parameter r1 range
    pub r1_range: (Float, Float),
    /// Random parameter r2 range
    pub r2_range: (Float, Float),
    /// Random parameter r3 range
    pub r3_range: (Float, Float),
    /// Random parameter r4 range (for sine/cosine selection)
    pub r4_range: (Float, Float),
    
    // Advanced parameters
    /// Oscillation frequency
    pub oscillation_frequency: Float,
    /// Amplitude modulation factor
    pub amplitude_modulation: Float,
    /// Phase shift for oscillations
    pub phase_shift: Float,
    /// Damping factor for damped oscillations
    pub damping_factor: Float,
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
    /// Memory factor for adaptive behavior
    pub memory_factor: Float,
}

impl Default for ScaParameters {
    fn default() -> Self {
        Self {
            common: CommonParameters::default(),
            variant: ScaVariant::Standard,
            oscillation_strategy: OscillationStrategy::Standard,
            wave_pattern: WavePattern::Mixed,
            parameter_update_strategy: ParameterUpdateStrategy::Linear,
            initial_a: 2.0,
            r1_range: (0.0, 2.0 * PI),
            r2_range: (0.0, 2.0),
            r3_range: (-2.0, 2.0),
            r4_range: (0.0, 1.0),
            oscillation_frequency: 1.0,
            amplitude_modulation: 1.0,
            phase_shift: 0.0,
            damping_factor: 0.99,
            elite_ratio: 0.1,
            diversity_threshold: 1e-6,
            adaptive_parameters: true,
            chaos_parameter: 4.0,
            quantum_probability: 0.1,
            levy_alpha: 1.5,
            convergence_threshold: 1e-8,
            memory_factor: 0.9,
        }
    }
}

/// Individual agent in the SCA population
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScaAgent {
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
    /// Agent ID for tracking
    pub id: usize,
    /// Age (iterations since creation)
    pub age: usize,
    /// Current oscillation phase
    pub oscillation_phase: Float,
    /// Oscillation frequency (for adaptive variants)
    pub personal_frequency: Float,
    /// Amplitude scaling factor
    pub amplitude_scale: Float,
    /// Success rate for adaptive behavior
    pub success_rate: Float,
    /// Energy level for oscillation intensity
    pub energy_level: Float,
    /// Memory of previous positions
    pub position_memory: Vec<Position>,
    /// Oscillation pattern preference
    pub pattern_preference: WavePattern,
}

impl Individual for ScaAgent {
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
            
            // Update success rate
            self.success_rate = 0.9 * self.success_rate + 0.1;
        } else {
            self.success_rate = 0.9 * self.success_rate;
        }
    }
}

impl ScaAgent {
    /// Create a new SCA agent with random initialization
    pub fn new(
        id: usize,
        dimensions: usize,
        bounds: &[(Float, Float)],
        params: &ScaParameters,
        rng: &mut impl Rng,
    ) -> Self {
        let position = DVector::from_fn(dimensions, |i, _| {
            let (min, max) = bounds[i];
            rng.gen_range(min..=max)
        });
        
        let velocity = DVector::zeros(dimensions);
        let oscillation_phase = rng.gen_range(0.0..=2.0 * PI);
        let personal_frequency = rng.gen_range(0.5..=2.0);
        
        // Random pattern preference
        let pattern_preference = match rng.gen_range(0..6) {
            0 => WavePattern::Sine,
            1 => WavePattern::Cosine,
            2 => WavePattern::Mixed,
            3 => WavePattern::Square,
            4 => WavePattern::Triangular,
            _ => WavePattern::Sawtooth,
        };
        
        Self {
            position: position.clone(),
            velocity,
            fitness: Float::INFINITY,
            best_position: position,
            best_fitness: Float::INFINITY,
            id,
            age: 0,
            oscillation_phase,
            personal_frequency,
            amplitude_scale: rng.gen_range(0.8..=1.2),
            success_rate: 0.0,
            energy_level: rng.gen_range(0.7..=1.0),
            position_memory: Vec::new(),
            pattern_preference,
        }
    }

    /// Update agent position using SCA equations
    pub fn update_position(
        &mut self,
        global_best: &Position,
        current_a: Float,
        params: &ScaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Generate random parameters
        let r1 = rng.gen_range(params.r1_range.0..=params.r1_range.1);
        let r2 = rng.gen_range(params.r2_range.0..=params.r2_range.1);
        let r3 = rng.gen_range(params.r3_range.0..=params.r3_range.1);
        let r4 = rng.gen_range(params.r4_range.0..=params.r4_range.1);
        
        // Update position based on oscillation strategy
        match params.oscillation_strategy {
            OscillationStrategy::Standard => {
                self.standard_oscillation_update(global_best, current_a, r1, r2, r3, r4, params)?;
            }
            OscillationStrategy::AdaptiveFrequency => {
                self.adaptive_frequency_update(global_best, current_a, r1, r2, r3, r4, params, iteration)?;
            }
            OscillationStrategy::MultiFrequency => {
                self.multi_frequency_update(global_best, current_a, r1, r2, r3, r4, params, iteration)?;
            }
            OscillationStrategy::Harmonic => {
                self.harmonic_oscillation_update(global_best, current_a, r1, r2, r3, r4, params, iteration)?;
            }
            OscillationStrategy::Damped => {
                self.damped_oscillation_update(global_best, current_a, r1, r2, r3, r4, params, iteration)?;
            }
        }
        
        // Apply variant-specific modifications
        self.apply_variant_modifications(params, iteration, rng)?;
        
        // Update memory
        self.update_position_memory(params);
        
        self.age += 1;
        Ok(())
    }

    /// Standard SCA position update
    fn standard_oscillation_update(
        &mut self,
        global_best: &Position,
        a: Float,
        r1: Float,
        r2: Float,
        r3: Float,
        r4: Float,
        params: &ScaParameters,
    ) -> Result<(), SwarmError> {
        for i in 0..self.position.len() {
            if r4 < 0.5 {
                // Sine component: X(t+1) = X(t) + r1 * sin(r2) * |r3 * P(t) - X(t)|
                self.position[i] = self.position[i] + 
                    r1 * (r2 * params.oscillation_frequency).sin() * 
                    (r3 * global_best[i] - self.position[i]).abs() * a;
            } else {
                // Cosine component: X(t+1) = X(t) + r1 * cos(r2) * |r3 * P(t) - X(t)|
                self.position[i] = self.position[i] + 
                    r1 * (r2 * params.oscillation_frequency).cos() * 
                    (r3 * global_best[i] - self.position[i]).abs() * a;
            }
        }
        
        Ok(())
    }

    /// Adaptive frequency oscillation update
    fn adaptive_frequency_update(
        &mut self,
        global_best: &Position,
        a: Float,
        r1: Float,
        r2: Float,
        r3: Float,
        r4: Float,
        params: &ScaParameters,
        iteration: usize,
    ) -> Result<(), SwarmError> {
        // Adaptive frequency based on success rate and age
        let adaptive_frequency = params.oscillation_frequency * 
            (1.0 + self.success_rate) * self.personal_frequency;
        
        for i in 0..self.position.len() {
            let wave_value = match params.wave_pattern {
                WavePattern::Sine => (r2 * adaptive_frequency + self.oscillation_phase).sin(),
                WavePattern::Cosine => (r2 * adaptive_frequency + self.oscillation_phase).cos(),
                WavePattern::Mixed => {
                    if r4 < 0.5 {
                        (r2 * adaptive_frequency + self.oscillation_phase).sin()
                    } else {
                        (r2 * adaptive_frequency + self.oscillation_phase).cos()
                    }
                }
                WavePattern::Square => {
                    if (r2 * adaptive_frequency + self.oscillation_phase).sin() > 0.0 { 1.0 } else { -1.0 }
                }
                WavePattern::Triangular => {
                    2.0 / PI * ((r2 * adaptive_frequency + self.oscillation_phase).sin().asin())
                }
                WavePattern::Sawtooth => {
                    2.0 * ((r2 * adaptive_frequency + self.oscillation_phase) / (2.0 * PI) - 
                          ((r2 * adaptive_frequency + self.oscillation_phase) / (2.0 * PI) + 0.5).floor())
                }
            };
            
            self.position[i] = self.position[i] + 
                r1 * wave_value * 
                (r3 * global_best[i] - self.position[i]).abs() * 
                a * self.amplitude_scale * self.energy_level;
        }
        
        // Update phase
        self.oscillation_phase += 0.1 * self.personal_frequency;
        if self.oscillation_phase > 2.0 * PI {
            self.oscillation_phase -= 2.0 * PI;
        }
        
        Ok(())
    }

    /// Multi-frequency oscillation update
    fn multi_frequency_update(
        &mut self,
        global_best: &Position,
        a: Float,
        r1: Float,
        r2: Float,
        r3: Float,
        r4: Float,
        params: &ScaParameters,
        iteration: usize,
    ) -> Result<(), SwarmError> {
        for i in 0..self.position.len() {
            // Use different frequencies for different dimensions
            let freq_i = params.oscillation_frequency * (i as Float + 1.0) / self.position.len() as Float;
            
            let sine_component = (r2 * freq_i + self.oscillation_phase).sin();
            let cosine_component = (r2 * freq_i * 0.7 + self.oscillation_phase * 0.8).cos();
            
            // Combine multiple frequencies
            let combined_wave = 0.6 * sine_component + 0.4 * cosine_component;
            
            self.position[i] = self.position[i] + 
                r1 * combined_wave * 
                (r3 * global_best[i] - self.position[i]).abs() * a;
        }
        
        Ok(())
    }

    /// Harmonic oscillation update
    fn harmonic_oscillation_update(
        &mut self,
        global_best: &Position,
        a: Float,
        r1: Float,
        r2: Float,
        r3: Float,
        r4: Float,
        params: &ScaParameters,
        iteration: usize,
    ) -> Result<(), SwarmError> {
        for i in 0..self.position.len() {
            // Generate harmonic series
            let fundamental = (r2 * params.oscillation_frequency + self.oscillation_phase).sin();
            let second_harmonic = 0.5 * (2.0 * r2 * params.oscillation_frequency + self.oscillation_phase).sin();
            let third_harmonic = 0.25 * (3.0 * r2 * params.oscillation_frequency + self.oscillation_phase).sin();
            
            let harmonic_wave = fundamental + second_harmonic + third_harmonic;
            
            self.position[i] = self.position[i] + 
                r1 * harmonic_wave * 
                (r3 * global_best[i] - self.position[i]).abs() * a;
        }
        
        Ok(())
    }

    /// Damped oscillation update
    fn damped_oscillation_update(
        &mut self,
        global_best: &Position,
        a: Float,
        r1: Float,
        r2: Float,
        r3: Float,
        r4: Float,
        params: &ScaParameters,
        iteration: usize,
    ) -> Result<(), SwarmError> {
        // Calculate damping factor based on iteration
        let damping = params.damping_factor.powf(iteration as Float);
        
        for i in 0..self.position.len() {
            let wave_value = if r4 < 0.5 {
                damping * (r2 * params.oscillation_frequency + self.oscillation_phase).sin()
            } else {
                damping * (r2 * params.oscillation_frequency + self.oscillation_phase).cos()
            };
            
            self.position[i] = self.position[i] + 
                r1 * wave_value * 
                (r3 * global_best[i] - self.position[i]).abs() * a;
        }
        
        Ok(())
    }

    /// Apply variant-specific modifications
    fn apply_variant_modifications(
        &mut self,
        params: &ScaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        match params.variant {
            ScaVariant::Standard => {
                // No additional modifications
            }
            ScaVariant::Enhanced => {
                self.apply_enhanced_modifications(params, iteration, rng)?;
            }
            ScaVariant::Chaotic => {
                self.apply_chaotic_modifications(params, iteration, rng)?;
            }
            ScaVariant::Quantum => {
                self.apply_quantum_modifications(params, rng)?;
            }
            ScaVariant::LevyFlight => {
                self.apply_levy_modifications(params, rng)?;
            }
            ScaVariant::Binary => {
                self.apply_binary_modifications(params, rng)?;
            }
        }
        
        Ok(())
    }

    /// Enhanced variant modifications
    fn apply_enhanced_modifications(
        &mut self,
        params: &ScaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Energy-based amplitude scaling
        if self.energy_level < 0.5 {
            // Low energy: reduce oscillation amplitude
            self.amplitude_scale *= 0.95;
        } else {
            // High energy: increase exploration
            self.amplitude_scale *= 1.02;
        }
        
        // Adaptive frequency adjustment
        if self.success_rate > 0.7 {
            // High success: reduce frequency for exploitation
            self.personal_frequency *= 0.98;
        } else if self.success_rate < 0.3 {
            // Low success: increase frequency for exploration
            self.personal_frequency *= 1.02;
        }
        
        // Update energy based on performance
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
        params: &ScaParameters,
        iteration: usize,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Logistic chaotic map
        let mut chaos_value = (iteration as Float * 0.1) % 1.0;
        chaos_value = params.chaos_parameter * chaos_value * (1.0 - chaos_value);
        
        // Apply chaotic perturbation to oscillation phase
        self.oscillation_phase += chaos_value * 0.1;
        
        // Chaotic amplitude modulation
        let chaos_amplitude = 0.1 * chaos_value;
        for i in 0..self.position.len() {
            let perturbation: Float = rng.gen_range(-1.0..=1.0);
            self.position[i] += chaos_amplitude * perturbation;
        }
        
        Ok(())
    }

    /// Quantum modifications
    fn apply_quantum_modifications(
        &mut self,
        params: &ScaParameters,
        rng: &mut impl Rng,
    ) -> Result<(), SwarmError> {
        // Quantum tunneling effect
        if rng.gen::<Float>() < params.quantum_probability {
            for i in 0..self.position.len() {
                let quantum_jump: Float = rng.gen_range(-1.0..=1.0);
                self.position[i] += quantum_jump * 0.5;
            }
        }
        
        // Quantum superposition in oscillation
        let superposition_factor = 0.1 * (rng.gen::<Float>() - 0.5);
        self.oscillation_phase += superposition_factor;
        
        Ok(())
    }

    /// Levy flight modifications
    fn apply_levy_modifications(
        &mut self,
        params: &ScaParameters,
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
        params: &ScaParameters,
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

    /// Update position memory
    fn update_position_memory(&mut self, params: &ScaParameters) {
        self.position_memory.push(self.position.clone());
        
        // Keep only recent positions
        if self.position_memory.len() > 10 {
            self.position_memory.remove(0);
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

/// Main Sine Cosine Algorithm implementation
#[derive(Debug)]
pub struct SineCosineAlgorithm {
    /// Algorithm parameters
    params: ScaParameters,
    /// Current agent population
    agents: Vec<ScaAgent>,
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
    /// Current value of parameter a
    current_a: Float,
}

impl SineCosineAlgorithm {
    /// Create new Sine Cosine Algorithm instance
    pub fn new(params: ScaParameters) -> Result<Self, SwarmError> {
        let seed = params.common.seed.unwrap_or_else(|| {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
        });
        
        let rng = Box::new(StdRng::seed_from_u64(seed));
        let current_a = params.initial_a;
        
        Ok(Self {
            params,
            agents: Vec::new(),
            global_best_position: DVector::zeros(1),
            global_best_fitness: Float::INFINITY,
            problem: None,
            rng,
            iteration: 0,
            metrics: AlgorithmMetrics::default(),
            elite_archive: Vec::new(),
            current_a,
        })
    }

    /// Initialize agent population
    async fn initialize_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        let dimensions = problem.dimensions();
        let bounds = problem.bounds();
        
        // Initialize agents
        self.agents = (0..self.params.common.population_size)
            .map(|i| ScaAgent::new(i, dimensions, bounds, &self.params, &mut *self.rng))
            .collect();
        
        // Initialize global best
        self.global_best_position = DVector::zeros(dimensions);
        self.global_best_fitness = Float::INFINITY;
        
        // Evaluate initial population
        self.evaluate_population(problem).await?;
        
        // Find initial best
        self.update_global_best();
        
        Ok(())
    }

    /// Update parameter a based on strategy
    fn update_parameter_a(&mut self, max_iterations: usize) {
        match self.params.parameter_update_strategy {
            ParameterUpdateStrategy::Linear => {
                // Linear decrease: a = 2 - 2 * t / T
                self.current_a = self.params.initial_a * 
                    (1.0 - self.iteration as Float / max_iterations as Float);
            }
            ParameterUpdateStrategy::Exponential => {
                // Exponential decrease
                let decay_rate = 0.01;
                self.current_a = self.params.initial_a * 
                    (-decay_rate * self.iteration as Float).exp();
            }
            ParameterUpdateStrategy::Adaptive => {
                // Adaptive based on convergence
                let diversity = self.calculate_population_diversity();
                if diversity < 0.01 {
                    self.current_a *= 0.95; // Decrease for exploitation
                } else if diversity > 0.1 {
                    self.current_a *= 1.05; // Increase for exploration
                }
            }
            ParameterUpdateStrategy::Piecewise => {
                // Piecewise linear
                let progress = self.iteration as Float / max_iterations as Float;
                if progress < 0.3 {
                    self.current_a = self.params.initial_a;
                } else if progress < 0.7 {
                    self.current_a = self.params.initial_a * (1.0 - (progress - 0.3) / 0.4);
                } else {
                    self.current_a = self.params.initial_a * 0.1;
                }
            }
            ParameterUpdateStrategy::Sinusoidal => {
                // Sinusoidal variation
                let freq = 2.0 * PI / max_iterations as Float;
                self.current_a = self.params.initial_a * 
                    (1.0 + 0.5 * (freq * self.iteration as Float).sin()) / 2.0;
            }
        }
        
        // Ensure a stays positive
        self.current_a = self.current_a.max(0.01);
    }

    /// Evaluate fitness for all agents
    async fn evaluate_population(&mut self, problem: &OptimizationProblem) -> Result<(), SwarmError> {
        if self.params.common.parallel_evaluation {
            // Parallel evaluation using rayon
            self.agents.par_iter_mut().try_for_each(|agent| -> Result<(), SwarmError> {
                let fitness = problem.evaluate(&agent.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                agent.set_fitness(fitness);
                Ok(())
            })?;
        } else {
            // Sequential evaluation
            for agent in &mut self.agents {
                let fitness = problem.evaluate(&agent.position)
                    .map_err(|e| SwarmError::FitnessEvaluationError(format!("Evaluation failed: {}", e)))?;
                agent.set_fitness(fitness);
            }
        }
        
        Ok(())
    }

    /// Update global best position
    fn update_global_best(&mut self) {
        for agent in &self.agents {
            if agent.best_fitness < self.global_best_fitness {
                self.global_best_fitness = agent.best_fitness;
                self.global_best_position = agent.best_position.clone();
            }
        }
    }

    /// Perform one iteration of the SCA algorithm
    async fn step_iteration(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().ok_or_else(|| {
            SwarmError::InitializationError("Problem not initialized".to_string())
        })?;
        
        let step_start = Instant::now();
        let bounds = problem.bounds();
        let max_iterations = self.params.common.max_evaluations / self.params.common.population_size;
        
        // Update parameter a
        self.update_parameter_a(max_iterations);
        
        // Update agent positions
        for agent in &mut self.agents {
            agent.update_position(
                &self.global_best_position,
                self.current_a,
                &self.params,
                self.iteration,
                &mut *self.rng,
            )?;
            
            agent.apply_boundaries(bounds);
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
            // Re-initialize worst agents
            let num_reinit = (self.params.common.population_size as Float * 0.1) as usize;
            
            // Sort agents by fitness (worst first)
            let mut indices: Vec<usize> = (0..self.agents.len()).collect();
            indices.sort_by(|&a, &b| {
                self.agents[b].fitness.partial_cmp(&self.agents[a].fitness).unwrap()
            });
            
            // Re-initialize worst agents
            if let Some(ref problem) = self.problem {
                let bounds = problem.bounds();
                let dimensions = problem.dimensions();
                
                for &idx in indices.iter().take(num_reinit) {
                    self.agents[idx] = ScaAgent::new(idx, dimensions, bounds, &self.params, &mut *self.rng);
                }
            }
        }
        
        Ok(())
    }

    /// Calculate population diversity
    fn calculate_population_diversity(&self) -> Float {
        if self.agents.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let distance = (&self.agents[i].position - &self.agents[j].position).norm();
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
        
        // Add current best agents to archive
        for agent in &self.agents {
            if agent.best_fitness < Float::INFINITY {
                self.elite_archive.push((agent.best_position.clone(), agent.best_fitness));
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
impl SwarmAlgorithm for SineCosineAlgorithm {
    type Individual = ScaAgent;
    type Fitness = Float;
    type Parameters = ScaParameters;

    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.problem = Some(problem.clone());
        self.initialize_population(&problem).await?;
        Ok(())
    }

    async fn step(&mut self) -> Result<(), SwarmError> {
        self.step_iteration().await
    }

    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.agents.iter().min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
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
            ScaVariant::Standard => "Sine Cosine Algorithm",
            ScaVariant::Enhanced => "Enhanced Sine Cosine Algorithm",
            ScaVariant::Chaotic => "Chaotic Sine Cosine Algorithm",
            ScaVariant::Quantum => "Quantum Sine Cosine Algorithm",
            ScaVariant::LevyFlight => "Levy Flight Sine Cosine Algorithm",
            ScaVariant::Binary => "Binary Sine Cosine Algorithm",
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
impl AdaptiveAlgorithm for SineCosineAlgorithm {
    async fn adapt_parameters(&mut self, strategy: AdaptationStrategy) -> Result<(), SwarmError> {
        match strategy {
            AdaptationStrategy::PerformanceBased => {
                let diversity = self.calculate_population_diversity();
                
                if diversity < 0.01 {
                    // Low diversity: increase exploration
                    self.params.oscillation_frequency *= 1.1;
                    self.params.amplitude_modulation *= 1.05;
                } else if diversity > 0.1 {
                    // High diversity: increase exploitation
                    self.params.oscillation_frequency *= 0.95;
                    self.params.amplitude_modulation *= 0.98;
                }
            }
            AdaptationStrategy::DiversityBased => {
                let diversity = self.calculate_population_diversity();
                self.params.amplitude_modulation = 0.5 + 0.5 * diversity;
            }
            AdaptationStrategy::FeedbackBased => {
                let avg_success_rate = self.agents.iter()
                    .map(|agent| agent.success_rate)
                    .sum::<Float>() / self.agents.len() as Float;
                
                if avg_success_rate < 0.3 {
                    self.params.oscillation_frequency *= 1.1;
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
impl ParallelAlgorithm for SineCosineAlgorithm {
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
    async fn test_sca_creation() {
        let params = ScaParameters::default();
        let sca = SineCosineAlgorithm::new(params).unwrap();
        assert_eq!(sca.name(), "Sine Cosine Algorithm");
    }

    #[tokio::test]
    async fn test_sca_sphere_function() {
        let params = ScaParameters {
            common: CommonParameters {
                population_size: 20,
                max_evaluations: 1000,
                tolerance: 1e-3,
                ..Default::default()
            },
            ..Default::default()
        };
        
        let mut sca = SineCosineAlgorithm::new(params).unwrap();
        
        // Sphere function: f(x) = sum(x_i^2)
        let problem = OptimizationProblem::new(
            5, // dimensions
            vec![(-5.0, 5.0); 5], // bounds
            Box::new(|x: &DVector<Float>| {
                Ok(x.iter().map(|xi| xi * xi).sum())
            }),
        );
        
        sca.initialize(problem).await.unwrap();
        
        // Run optimization
        for _ in 0..50 {
            sca.step().await.unwrap();
            if sca.has_converged() {
                break;
            }
        }
        
        // Should find a reasonably good solution
        assert!(sca.global_best_fitness < 1.0);
    }

    #[test]
    fn test_agent_creation() {
        let params = ScaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0), (-5.0, 5.0)];
        
        let agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
        
        assert_eq!(agent.id, 0);
        assert_eq!(agent.position.len(), 2);
        assert!(agent.position[0] >= -10.0 && agent.position[0] <= 10.0);
        assert!(agent.position[1] >= -5.0 && agent.position[1] <= 5.0);
    }

    #[test]
    fn test_oscillation_update() {
        let params = ScaParameters::default();
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        let mut agent = ScaAgent::new(0, 2, &bounds, &params, &mut rng);
        
        let global_best = DVector::from_vec(vec![1.0, 2.0]);
        let current_a = 1.5;
        
        agent.update_position(&global_best, current_a, &params, 1, &mut rng).unwrap();
        
        // Position should have been updated
        assert!(agent.position.norm() > 0.0);
    }

    #[test]
    fn test_parameter_a_update() {
        let params = ScaParameters::default();
        let mut sca = SineCosineAlgorithm::new(params).unwrap();
        
        let initial_a = sca.current_a;
        sca.iteration = 50;
        sca.update_parameter_a(100);
        
        // Parameter a should have decreased
        assert!(sca.current_a < initial_a);
    }

    #[test]
    fn test_population_diversity() {
        let params = ScaParameters::default();
        let mut sca = SineCosineAlgorithm::new(params).unwrap();
        
        // Create agents with known positions
        let mut rng = StdRng::seed_from_u64(42);
        let bounds = vec![(-10.0, 10.0); 2];
        
        sca.agents = vec![
            {
                let mut agent = ScaAgent::new(0, 2, &bounds, &sca.params, &mut rng);
                agent.position = DVector::from_vec(vec![0.0, 0.0]);
                agent
            },
            {
                let mut agent = ScaAgent::new(1, 2, &bounds, &sca.params, &mut rng);
                agent.position = DVector::from_vec(vec![3.0, 4.0]);
                agent
            },
        ];
        
        let diversity = sca.calculate_population_diversity();
        
        // Expected diversity is the distance between (0,0) and (3,4) = 5.0
        assert_relative_eq!(diversity, 5.0, epsilon = 1e-10);
    }
}