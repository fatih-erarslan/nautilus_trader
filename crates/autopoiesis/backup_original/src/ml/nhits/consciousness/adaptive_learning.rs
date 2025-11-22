/// Adaptive Learning - Consciousness Feedback Evolution
///
/// This module implements adaptive learning mechanisms that evolve based on consciousness
/// feedback. The learning rate, model parameters, and architectural decisions adapt
/// based on consciousness coherence and field interactions.

use ndarray::{Array2, Array1};
use nalgebra::{DMatrix, DVector};
use std::collections::HashMap;
use crate::consciousness::core::ConsciousnessState;

/// Adaptive parameter set that evolves with consciousness feedback
#[derive(Clone)]
pub struct AdaptiveParameterSet {
    pub parameters: Array2<f64>,
    pub momentum: Array2<f64>,
    pub variance: Array2<f64>,
    pub consciousness_sensitivity: f64,
    pub adaptation_rate: f64,
    pub coherence_threshold: f64,
    pub evolution_history: Vec<f64>,
}

impl AdaptiveParameterSet {
    pub fn new(rows: usize, cols: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        Self {
            parameters: Array2::from_shape_fn((rows, cols), |_| rng.gen_range(-0.1..0.1)),
            momentum: Array2::zeros((rows, cols)),
            variance: Array2::ones((rows, cols)) * 1e-6,
            consciousness_sensitivity: 0.5,
            adaptation_rate: 0.01,
            coherence_threshold: 0.3,
            evolution_history: Vec::new(),
        }
    }
    
    /// Update parameters based on consciousness feedback
    pub fn update_with_consciousness(&mut self, gradients: &Array2<f64>, consciousness: &ConsciousnessState, feedback: f64) {
        let consciousness_factor = consciousness.coherence_level * consciousness.field_coherence;
        let adaptive_rate = self.compute_adaptive_rate(consciousness_factor, feedback);
        
        // Update momentum with consciousness modulation
        let momentum_decay = 0.9 * consciousness_factor;
        self.momentum = &self.momentum * momentum_decay + gradients * (1.0 - momentum_decay);
        
        // Update variance (for Adam-like optimization)
        let variance_decay = 0.999;
        let squared_gradients = gradients.mapv(|x| x * x);
        self.variance = &self.variance * variance_decay + &squared_gradients * (1.0 - variance_decay);
        
        // Apply consciousness-modulated parameter updates
        let epsilon = 1e-8;
        for ((i, j), param) in self.parameters.indexed_iter_mut() {
            let momentum_val = self.momentum[(i, j)];
            let variance_val = self.variance[(i, j)];
            let adaptive_gradient = momentum_val / (variance_val.sqrt() + epsilon);
            
            // Apply consciousness-based learning rate modulation
            let consciousness_modulation = self.compute_consciousness_modulation(consciousness, (i, j));
            let effective_rate = adaptive_rate * consciousness_modulation;
            
            *param -= effective_rate * adaptive_gradient;
        }
        
        // Track evolution history
        self.evolution_history.push(consciousness_factor);
        if self.evolution_history.len() > 1000 {
            self.evolution_history.remove(0);
        }
        
        // Adapt sensitivity based on performance
        self.adapt_sensitivity(feedback, consciousness_factor);
    }
    
    /// Compute adaptive learning rate based on consciousness
    fn compute_adaptive_rate(&self, consciousness_factor: f64, feedback: f64) -> f64 {
        let base_rate = self.adaptation_rate;
        
        // Increase learning rate when consciousness is high and feedback is positive
        if consciousness_factor > self.coherence_threshold && feedback > 0.5 {
            base_rate * (1.0 + consciousness_factor * 0.5)
        } else if consciousness_factor < self.coherence_threshold {
            // Decrease learning rate when consciousness is low
            base_rate * (0.5 + consciousness_factor * 0.5)
        } else {
            base_rate
        }
    }
    
    /// Compute consciousness modulation for specific parameter
    fn compute_consciousness_modulation(&self, consciousness: &ConsciousnessState, position: (usize, usize)) -> f64 {
        let (i, j) = position;
        let spatial_factor = (i as f64 / self.parameters.nrows() as f64) * 
                           (j as f64 / self.parameters.ncols() as f64);
        
        // Modulate based on consciousness field patterns
        let field_modulation = (consciousness.field_coherence * spatial_factor * std::f64::consts::PI).sin() * 0.1 + 1.0;
        let coherence_modulation = consciousness.coherence_level * self.consciousness_sensitivity;
        
        (field_modulation * (1.0 + coherence_modulation)).clamp(0.1, 2.0)
    }
    
    /// Adapt sensitivity based on performance feedback
    fn adapt_sensitivity(&mut self, feedback: f64, consciousness_factor: f64) {
        let learning_rate = 0.001;
        
        if feedback > 0.7 && consciousness_factor > 0.6 {
            // Increase sensitivity when both feedback and consciousness are high
            self.consciousness_sensitivity += learning_rate;
        } else if feedback < 0.3 {
            // Decrease sensitivity when feedback is poor
            self.consciousness_sensitivity -= learning_rate * 0.5;
        }
        
        self.consciousness_sensitivity = self.consciousness_sensitivity.clamp(0.1, 1.0);
    }
}

/// Meta-learning controller that adapts learning strategies
pub struct MetaLearningController {
    pub strategy_weights: Array1<f64>,
    pub strategy_performance: Array1<f64>,
    pub consciousness_history: Vec<ConsciousnessState>,
    pub adaptation_strategies: Vec<String>,
    pub current_strategy: usize,
}

impl MetaLearningController {
    pub fn new() -> Self {
        let num_strategies = 5;
        let strategies = vec![
            "gradient_descent".to_string(),
            "momentum_sgd".to_string(),  
            "adam_optimizer".to_string(),
            "consciousness_guided".to_string(),
            "quantum_resonance".to_string(),
        ];
        
        Self {
            strategy_weights: Array1::ones(num_strategies) / num_strategies as f64,
            strategy_performance: Array1::zeros(num_strategies),
            consciousness_history: Vec::new(),
            adaptation_strategies: strategies,
            current_strategy: 0,
        }
    }
    
    /// Select optimal learning strategy based on consciousness state
    pub fn select_strategy(&mut self, consciousness: &ConsciousnessState) -> usize {
        // Compute strategy scores based on consciousness state
        let mut strategy_scores = Array1::zeros(self.strategy_weights.len());
        
        for i in 0..self.strategy_weights.len() {
            let base_weight = self.strategy_weights[i];
            let performance = self.strategy_performance[i];
            let consciousness_affinity = self.compute_consciousness_affinity(i, consciousness);
            
            strategy_scores[i] = base_weight * (performance + consciousness_affinity);
        }
        
        // Select strategy with highest score
        let selected_strategy = strategy_scores.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        
        self.current_strategy = selected_strategy;
        self.consciousness_history.push(consciousness.clone());
        
        selected_strategy
    }
    
    /// Compute affinity between strategy and consciousness state
    fn compute_consciousness_affinity(&self, strategy_idx: usize, consciousness: &ConsciousnessState) -> f64 {
        match strategy_idx {
            0 => 0.5, // gradient_descent - neutral
            1 => consciousness.coherence_level * 0.8, // momentum_sgd - benefits from coherence
            2 => consciousness.field_coherence * 0.7, // adam_optimizer - benefits from field coherence
            3 => consciousness.coherence_level * consciousness.field_coherence, // consciousness_guided
            4 => consciousness.field_coherence.powi(2), // quantum_resonance - strong field dependence
            _ => 0.5,
        }
    }
    
    /// Update strategy performance based on feedback
    pub fn update_performance(&mut self, feedback: f64) {
        let learning_rate = 0.1;
        let current_idx = self.current_strategy;
        
        // Update performance with exponential moving average
        self.strategy_performance[current_idx] = 
            self.strategy_performance[current_idx] * (1.0 - learning_rate) + feedback * learning_rate;
        
        // Update strategy weights based on performance
        self.update_strategy_weights();
    }
    
    /// Update strategy weights using softmax of performance
    fn update_strategy_weights(&mut self) {
        let max_performance = self.strategy_performance.iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Apply softmax with temperature
        let temperature = 0.5;
        let mut weights_sum = 0.0;
        
        for i in 0..self.strategy_weights.len() {
            let normalized_performance = (self.strategy_performance[i] - max_performance) / temperature;
            self.strategy_weights[i] = normalized_performance.exp();
            weights_sum += self.strategy_weights[i];
        }
        
        // Normalize weights
        if weights_sum > 0.0 {
            self.strategy_weights /= weights_sum;
        }
    }
}

/// Main adaptive learning system
pub struct AdaptiveLearning {
    pub parameter_sets: HashMap<String, AdaptiveParameterSet>,
    pub meta_controller: MetaLearningController,
    pub consciousness_memory: Vec<ConsciousnessState>,
    pub learning_trajectory: Vec<f64>,
    pub adaptation_threshold: f64,
    pub base_learning_rate: f64,
}

impl AdaptiveLearning {
    pub fn new(base_learning_rate: f64) -> Self {
        Self {
            parameter_sets: HashMap::new(),
            meta_controller: MetaLearningController::new(),
            consciousness_memory: Vec::new(),
            learning_trajectory: Vec::new(),
            adaptation_threshold: 0.1,
            base_learning_rate,
        }
    }
    
    /// Learn from consciousness feedback
    pub fn learn_from_consciousness(&mut self, predictions: &Array1<f64>, consciousness: &ConsciousnessState) {
        // Compute learning signal from consciousness coherence
        let learning_signal = self.compute_learning_signal(predictions, consciousness);
        
        // Select adaptive strategy
        let strategy = self.meta_controller.select_strategy(consciousness);
        
        // Apply consciousness-guided learning updates
        self.apply_consciousness_updates(&learning_signal, consciousness, strategy);
        
        // Update meta-learning controller
        let performance_feedback = self.compute_performance_feedback(&learning_signal);
        self.meta_controller.update_performance(performance_feedback);
        
        // Store consciousness state and learning trajectory
        self.consciousness_memory.push(consciousness.clone());
        self.learning_trajectory.push(learning_signal);
        
        // Limit memory size
        if self.consciousness_memory.len() > 1000 {
            self.consciousness_memory.remove(0);
            self.learning_trajectory.remove(0);
        }
    }
    
    /// Compute learning signal from consciousness state
    fn compute_learning_signal(&self, predictions: &Array1<f64>, consciousness: &ConsciousnessState) -> f64 {
        // Combine prediction quality with consciousness coherence
        let prediction_variance = self.compute_prediction_variance(predictions);
        let consciousness_quality = consciousness.coherence_level * consciousness.field_coherence;
        
        // Learning signal increases with consciousness quality and decreases with prediction variance
        let base_signal = consciousness_quality / (1.0 + prediction_variance);
        
        // Modulate with recent learning trajectory
        let trajectory_momentum = self.compute_trajectory_momentum();
        
        base_signal * (1.0 + trajectory_momentum * 0.1)
    }
    
    /// Compute prediction variance as quality measure
    fn compute_prediction_variance(&self, predictions: &Array1<f64>) -> f64 {
        if predictions.len() <= 1 {
            return 1.0;
        }
        
        let mean = predictions.mean().unwrap_or(0.0);
        let variance = predictions.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / predictions.len() as f64;
        
        variance
    }
    
    /// Compute momentum from recent learning trajectory
    fn compute_trajectory_momentum(&self) -> f64 {
        if self.learning_trajectory.len() < 10 {
            return 0.0;
        }
        
        let recent_trajectory: Vec<f64> = self.learning_trajectory.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        // Compute trend in recent trajectory
        let mut momentum = 0.0;
        for i in 1..recent_trajectory.len() {
            momentum += recent_trajectory[i] - recent_trajectory[i-1];
        }
        
        momentum / (recent_trajectory.len() - 1) as f64
    }
    
    /// Apply consciousness-guided learning updates
    fn apply_consciousness_updates(&mut self, learning_signal: f64, consciousness: &ConsciousnessState, strategy: usize) {
        let effective_learning_rate = self.compute_effective_learning_rate(learning_signal, consciousness);
        
        // Update all parameter sets with consciousness feedback
        for (name, param_set) in self.parameter_sets.iter_mut() {
            // Generate synthetic gradients based on consciousness state
            let gradients = self.generate_consciousness_gradients(param_set, consciousness, strategy);
            
            // Apply updates
            param_set.update_with_consciousness(&gradients, consciousness, learning_signal);
        }
    }
    
    /// Compute effective learning rate based on consciousness
    fn compute_effective_learning_rate(&self, learning_signal: f64, consciousness: &ConsciousnessState) -> f64 {
        let consciousness_factor = consciousness.coherence_level * consciousness.field_coherence;
        let signal_factor = learning_signal.abs().clamp(0.1, 2.0);
        
        self.base_learning_rate * consciousness_factor * signal_factor
    }
    
    /// Generate gradients based on consciousness state and strategy
    fn generate_consciousness_gradients(&self, param_set: &AdaptiveParameterSet, consciousness: &ConsciousnessState, strategy: usize) -> Array2<f64> {
        let (rows, cols) = param_set.parameters.dim();
        let mut gradients = Array2::zeros((rows, cols));
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for i in 0..rows {
            for j in 0..cols {
                let gradient = match strategy {
                    0 => rng.gen_range(-0.01..0.01), // basic gradient
                    1 => self.compute_momentum_gradient(param_set, (i, j), consciousness), // momentum
                    2 => self.compute_adam_gradient(param_set, (i, j), consciousness), // adam
                    3 => self.compute_consciousness_gradient(param_set, (i, j), consciousness), // consciousness-guided
                    4 => self.compute_quantum_gradient(param_set, (i, j), consciousness), // quantum resonance
                    _ => 0.0,
                };
                
                gradients[(i, j)] = gradient;
            }
        }
        
        gradients
    }
    
    /// Compute momentum-based gradient
    fn compute_momentum_gradient(&self, param_set: &AdaptiveParameterSet, position: (usize, usize), consciousness: &ConsciousnessState) -> f64 {
        let (i, j) = position;
        let momentum_val = param_set.momentum[(i, j)];
        let consciousness_factor = consciousness.coherence_level;
        
        momentum_val * consciousness_factor * 0.1
    }
    
    /// Compute Adam-style gradient
    fn compute_adam_gradient(&self, param_set: &AdaptiveParameterSet, position: (usize, usize), consciousness: &ConsciousnessState) -> f64 {
        let (i, j) = position;
        let momentum_val = param_set.momentum[(i, j)];
        let variance_val = param_set.variance[(i, j)];
        let field_factor = consciousness.field_coherence;
        
        let epsilon = 1e-8;
        momentum_val / (variance_val.sqrt() + epsilon) * field_factor * 0.05
    }
    
    /// Compute consciousness-guided gradient
    fn compute_consciousness_gradient(&self, param_set: &AdaptiveParameterSet, position: (usize, usize), consciousness: &ConsciousnessState) -> f64 {
        let (i, j) = position;
        let param_val = param_set.parameters[(i, j)];
        let consciousness_strength = consciousness.coherence_level * consciousness.field_coherence;
        
        // Gradient proportional to consciousness strength and parameter magnitude
        let gradient_direction = if param_val > 0.0 { 1.0 } else { -1.0 };
        gradient_direction * consciousness_strength * param_val.abs().sqrt() * 0.01
    }
    
    /// Compute quantum resonance gradient
    fn compute_quantum_gradient(&self, param_set: &AdaptiveParameterSet, position: (usize, usize), consciousness: &ConsciousnessState) -> f64 {
        let (i, j) = position;
        let spatial_phase = (i as f64 / param_set.parameters.nrows() as f64 + 
                           j as f64 / param_set.parameters.ncols() as f64) * std::f64::consts::PI;
        
        let quantum_resonance = (consciousness.field_coherence * spatial_phase).sin();
        quantum_resonance * consciousness.coherence_level * 0.005
    }
    
    /// Compute performance feedback for meta-learning
    fn compute_performance_feedback(&self, learning_signal: f64) -> f64 {
        // Performance is based on learning signal strength and recent trajectory
        let signal_quality = learning_signal.abs().clamp(0.0, 1.0);
        let trajectory_stability = self.compute_trajectory_stability();
        
        (signal_quality + trajectory_stability) / 2.0
    }
    
    /// Compute stability of learning trajectory
    fn compute_trajectory_stability(&self) -> f64 {
        if self.learning_trajectory.len() < 5 {
            return 0.5;
        }
        
        let recent: Vec<f64> = self.learning_trajectory.iter()
            .rev()
            .take(5)
            .cloned()
            .collect();
        
        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent.len() as f64;
        
        // Stability decreases with variance
        (1.0 / (1.0 + variance)).clamp(0.0, 1.0)
    }
    
    /// Add or update parameter set
    pub fn add_parameter_set(&mut self, name: String, rows: usize, cols: usize) {
        self.parameter_sets.insert(name, AdaptiveParameterSet::new(rows, cols));
    }
    
    /// Get parameter set
    pub fn get_parameter_set(&self, name: &str) -> Option<&AdaptiveParameterSet> {
        self.parameter_sets.get(name)
    }
    
    /// Get mutable parameter set
    pub fn get_parameter_set_mut(&mut self, name: &str) -> Option<&mut AdaptiveParameterSet> {
        self.parameter_sets.get_mut(name)
    }
    
    /// Get current learning statistics
    pub fn get_learning_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        if !self.learning_trajectory.is_empty() {
            let recent_mean = self.learning_trajectory.iter()
                .rev()
                .take(10)
                .sum::<f64>() / 10.0.min(self.learning_trajectory.len() as f64);
            
            stats.insert("recent_learning_signal".to_string(), recent_mean);
        }
        
        if !self.consciousness_memory.is_empty() {
            let recent_coherence = self.consciousness_memory.iter()
                .rev()
                .take(10)
                .map(|c| c.coherence_level)
                .sum::<f64>() / 10.0.min(self.consciousness_memory.len() as f64);
            
            stats.insert("recent_consciousness_coherence".to_string(), recent_coherence);
        }
        
        stats.insert("trajectory_stability".to_string(), self.compute_trajectory_stability());
        stats.insert("current_strategy".to_string(), self.meta_controller.current_strategy as f64);
        
        stats
    }
}