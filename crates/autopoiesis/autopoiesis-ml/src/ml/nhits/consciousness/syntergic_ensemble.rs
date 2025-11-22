/// Syntergic Ensemble - Consciousness Coherence Ensembles
///
/// This module implements ensemble forecasting with consciousness coherence weighting.
/// Multiple forecasting models are combined based on their consciousness coherence
/// levels and field resonance patterns for enhanced prediction accuracy.

use ndarray::{Array2, Array1};
use nalgebra::{DMatrix, DVector};
use std::collections::{HashMap, VecDeque};
use crate::consciousness::core::ConsciousnessState;

/// Individual ensemble member with consciousness awareness
#[derive(Clone)]
pub struct ConsciousnessEnsembleMember {
    pub member_id: String,
    pub model_weights: Array2<f64>,
    pub bias: Array1<f64>,
    pub consciousness_affinity: f64,
    pub coherence_history: VecDeque<f64>,
    pub performance_history: VecDeque<f64>,
    pub specialization_domain: SpecializationDomain,
    pub field_resonance_pattern: Array1<f64>,
    pub adaptation_rate: f64,
}

#[derive(Clone, Copy, Debug)]
pub enum SpecializationDomain {
    ShortTerm,    // Optimized for short-term patterns
    LongTerm,     // Optimized for long-term trends
    Volatility,   // Specialized in volatility prediction
    Momentum,     // Momentum-based predictions
    Reversal,     // Mean reversion patterns
    Anomaly,      // Anomaly detection and handling
    Hybrid,       // General-purpose ensemble member
}

impl ConsciousnessEnsembleMember {
    pub fn new(member_id: String, input_dim: usize, output_dim: usize, domain: SpecializationDomain) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Initialize weights based on specialization
        let weight_scale = match domain {
            SpecializationDomain::ShortTerm => 0.2,
            SpecializationDomain::LongTerm => 0.1,
            SpecializationDomain::Volatility => 0.3,
            SpecializationDomain::Momentum => 0.25,
            SpecializationDomain::Reversal => 0.15,
            SpecializationDomain::Anomaly => 0.4,
            SpecializationDomain::Hybrid => 0.2,
        };
        
        let model_weights = Array2::from_shape_fn((output_dim, input_dim), |_| {
            rng.gen_range(-weight_scale..weight_scale)
        });
        
        let bias = Array1::from_shape_fn(output_dim, |_| {
            rng.gen_range(-0.01..0.01)
        });
        
        // Initialize field resonance pattern based on domain
        let field_resonance_pattern = Array1::from_shape_fn(input_dim, |i| {
            let phase = (i as f64 / input_dim as f64) * 2.0 * std::f64::consts::PI;
            match domain {
                SpecializationDomain::ShortTerm => (phase * 5.0).sin() * 0.1,
                SpecializationDomain::LongTerm => (phase * 0.5).sin() * 0.1,
                SpecializationDomain::Volatility => (phase * 10.0).cos() * 0.1,
                SpecializationDomain::Momentum => phase.sin() * 0.1,
                SpecializationDomain::Reversal => (-phase).sin() * 0.1,
                SpecializationDomain::Anomaly => (phase * 3.0 + std::f64::consts::PI/4.0).sin() * 0.1,
                SpecializationDomain::Hybrid => 0.0,
            }
        });
        
        Self {
            member_id,
            model_weights,
            bias,
            consciousness_affinity: 0.5,
            coherence_history: VecDeque::with_capacity(100),
            performance_history: VecDeque::with_capacity(100),
            specialization_domain: domain,
            field_resonance_pattern,
            adaptation_rate: 0.01,
        }
    }
    
    /// Generate prediction from ensemble member
    pub fn predict(&self, input: &Array1<f64>, consciousness: &ConsciousnessState) -> Array1<f64> {
        // Apply consciousness modulation to input
        let consciousness_factor = consciousness.coherence_level * self.consciousness_affinity;
        let modulated_input = input.mapv(|x| x * (1.0 + consciousness_factor * 0.1));
        
        // Apply field resonance
        let resonance_input = &modulated_input + &(&self.field_resonance_pattern * consciousness.field_coherence);
        
        // Compute base prediction
        let prediction = self.model_weights.dot(&resonance_input) + &self.bias;
        
        // Apply domain-specific transformation
        self.apply_domain_transformation(&prediction, consciousness)
    }
    
    /// Apply domain-specific transformation to prediction
    fn apply_domain_transformation(&self, prediction: &Array1<f64>, consciousness: &ConsciousnessState) -> Array1<f64> {
        let consciousness_strength = consciousness.coherence_level * consciousness.field_coherence;
        
        match self.specialization_domain {
            SpecializationDomain::ShortTerm => {
                // High-frequency components emphasized
                prediction.mapv(|x| x.tanh() * (1.0 + consciousness_strength * 0.2))
            },
            SpecializationDomain::LongTerm => {
                // Smooth, trend-following transformation
                prediction.mapv(|x| x * 0.8 + (x * consciousness_strength).tanh() * 0.2)
            },
            SpecializationDomain::Volatility => {
                // Emphasize magnitude changes
                prediction.mapv(|x| x.signum() * (x.abs() + consciousness_strength * 0.1))
            },
            SpecializationDomain::Momentum => {
                // Momentum amplification
                prediction.mapv(|x| x * (1.0 + consciousness_strength * x.signum() * 0.1))
            },
            SpecializationDomain::Reversal => {
                // Mean reversion bias
                prediction.mapv(|x| x * (1.0 - consciousness_strength * 0.1))
            },
            SpecializationDomain::Anomaly => {
                // Non-linear anomaly handling
                prediction.mapv(|x| {
                    if x.abs() > 1.0 {
                        x.signum() * (1.0 + (x.abs() - 1.0) * consciousness_strength)
                    } else {
                        x
                    }
                })
            },
            SpecializationDomain::Hybrid => {
                // Balanced transformation
                prediction.mapv(|x| x.tanh() * (1.0 + consciousness_strength * 0.05))
            },
        }
    }
    
    /// Update member based on performance feedback
    pub fn update(&mut self, performance_score: f64, consciousness: &ConsciousnessState) {
        // Update performance history
        self.performance_history.push_back(performance_score);
        if self.performance_history.len() > 100 {
            self.performance_history.pop_front();
        }
        
        // Update coherence history
        let current_coherence = consciousness.coherence_level * consciousness.field_coherence;
        self.coherence_history.push_back(current_coherence);
        if self.coherence_history.len() > 100 {
            self.coherence_history.pop_front();
        }
        
        // Update consciousness affinity based on performance
        let performance_influence = (performance_score - 0.5) * self.adaptation_rate;
        self.consciousness_affinity += performance_influence * current_coherence;
        self.consciousness_affinity = self.consciousness_affinity.clamp(0.0, 1.0);
        
        // Adapt model weights based on consciousness feedback
        self.adapt_weights(performance_score, consciousness);
    }
    
    /// Adapt model weights based on consciousness feedback
    fn adapt_weights(&mut self, performance_score: f64, consciousness: &ConsciousnessState) {
        let learning_rate = self.adaptation_rate * consciousness.coherence_level;
        let weight_update_strength = (performance_score - 0.5) * learning_rate;
        
        // Update weights with consciousness-guided gradients
        for ((i, j), weight) in self.model_weights.indexed_iter_mut() {
            let spatial_phase = (i as f64 / self.model_weights.nrows() as f64 + 
                               j as f64 / self.model_weights.ncols() as f64) * std::f64::consts::PI;
            let consciousness_gradient = (spatial_phase * consciousness.field_coherence).sin();
            
            *weight += weight_update_strength * consciousness_gradient * 0.01;
        }
        
        // Update bias terms
        for (i, bias_val) in self.bias.iter_mut().enumerate() {
            let bias_phase = (i as f64 / self.bias.len() as f64) * 2.0 * std::f64::consts::PI;
            let consciousness_bias_gradient = (bias_phase * consciousness.coherence_level).cos();
            
            *bias_val += weight_update_strength * consciousness_bias_gradient * 0.001;
        }
        
        // Update field resonance pattern
        for (i, resonance) in self.field_resonance_pattern.iter_mut().enumerate() {
            let resonance_phase = (i as f64 / self.field_resonance_pattern.len() as f64) * 
                                 4.0 * std::f64::consts::PI;
            let consciousness_resonance = (resonance_phase * consciousness.field_coherence).sin();
            
            *resonance += performance_score * consciousness_resonance * learning_rate * 0.01;
        }
    }
    
    /// Get member's current confidence based on recent performance
    pub fn get_confidence(&self) -> f64 {
        if self.performance_history.is_empty() {
            return 0.5;
        }
        
        let recent_performance: f64 = self.performance_history.iter()
            .rev()
            .take(10)
            .sum::<f64>() / 10.0.min(self.performance_history.len() as f64);
        
        let performance_variance = if self.performance_history.len() > 1 {
            let mean = recent_performance;
            let variance = self.performance_history.iter()
                .rev()
                .take(10)
                .map(|p| (p - mean).powi(2))
                .sum::<f64>() / 10.0.min(self.performance_history.len() as f64);
            variance
        } else {
            0.1
        };
        
        // Confidence is high performance with low variance
        recent_performance * (1.0 / (1.0 + performance_variance))
    }
}

/// Syntergic ensemble system combining consciousness-aware models
pub struct SyntergicEnsemble {
    pub ensemble_members: Vec<ConsciousnessEnsembleMember>,
    pub ensemble_weights: Array1<f64>,
    pub consciousness_coherence_matrix: Array2<f64>,
    pub prediction_history: VecDeque<Array1<f64>>,
    pub performance_history: VecDeque<f64>,
    pub diversity_bonus: f64,
    pub coherence_threshold: f64,
    pub rebalancing_frequency: usize,
    pub step_count: usize,
}

impl SyntergicEnsemble {
    pub fn new(num_members: usize) -> Self {
        let mut ensemble_members = Vec::with_capacity(num_members);
        
        // Create diverse ensemble members with different specializations
        let domains = [
            SpecializationDomain::ShortTerm,
            SpecializationDomain::LongTerm,
            SpecializationDomain::Volatility,
            SpecializationDomain::Momentum,
            SpecializationDomain::Reversal,
            SpecializationDomain::Anomaly,
            SpecializationDomain::Hybrid,
        ];
        
        for i in 0..num_members {
            let domain = domains[i % domains.len()];
            let member_id = format!("member_{}_{:?}", i, domain);
            let member = ConsciousnessEnsembleMember::new(member_id, 64, 64, domain);
            ensemble_members.push(member);
        }
        
        let ensemble_weights = Array1::ones(num_members) / num_members as f64;
        let consciousness_coherence_matrix = Array2::eye(num_members);
        
        Self {
            ensemble_members,
            ensemble_weights,
            consciousness_coherence_matrix,
            prediction_history: VecDeque::with_capacity(1000),
            performance_history: VecDeque::with_capacity(1000),
            diversity_bonus: 0.1,
            coherence_threshold: 0.3,
            rebalancing_frequency: 10,
            step_count: 0,
        }
    }
    
    /// Combine forecasts from ensemble members with consciousness coherence weighting
    pub fn combine_forecasts(&mut self, individual_forecasts: &[Array1<f64>], consciousness: &ConsciousnessState) -> Array1<f64> {
        self.step_count += 1;
        
        // Update ensemble members based on consciousness
        self.update_consciousness_coherence(consciousness);
        
        // Generate ensemble member predictions
        let member_predictions = self.generate_member_predictions(individual_forecasts, consciousness);
        
        // Compute consciousness-weighted combination
        let weighted_forecast = self.compute_consciousness_weighted_combination(&member_predictions, consciousness);
        
        // Apply diversity bonus
        let diversity_enhanced_forecast = self.apply_diversity_enhancement(&weighted_forecast, &member_predictions, consciousness);
        
        // Store prediction history
        self.prediction_history.push_back(diversity_enhanced_forecast.clone());
        if self.prediction_history.len() > 1000 {
            self.prediction_history.pop_front();
        }
        
        // Rebalance ensemble if needed
        if self.step_count % self.rebalancing_frequency == 0 {
            self.rebalance_ensemble(consciousness);
        }
        
        diversity_enhanced_forecast
    }
    
    /// Update consciousness coherence matrix between ensemble members
    fn update_consciousness_coherence(&mut self, consciousness: &ConsciousnessState) {
        let num_members = self.ensemble_members.len();
        let consciousness_strength = consciousness.coherence_level * consciousness.field_coherence;
        
        for i in 0..num_members {
            for j in 0..num_members {
                if i != j {
                    // Compute coherence between members based on their consciousness affinity
                    let member_i_affinity = self.ensemble_members[i].consciousness_affinity;
                    let member_j_affinity = self.ensemble_members[j].consciousness_affinity;
                    
                    let coherence = (member_i_affinity * member_j_affinity).sqrt() * consciousness_strength;
                    
                    // Update coherence matrix with exponential smoothing
                    let smoothing = 0.1;
                    self.consciousness_coherence_matrix[(i, j)] = 
                        self.consciousness_coherence_matrix[(i, j)] * (1.0 - smoothing) + coherence * smoothing;
                } else {
                    self.consciousness_coherence_matrix[(i, j)] = 1.0;
                }
            }
        }
    }
    
    /// Generate predictions from ensemble members
    fn generate_member_predictions(&self, base_forecasts: &[Array1<f64>], consciousness: &ConsciousnessState) -> Vec<Array1<f64>> {
        let mut member_predictions = Vec::new();
        
        for (i, member) in self.ensemble_members.iter().enumerate() {
            // Use base forecast as input if available, otherwise use zeros
            let input = if i < base_forecasts.len() {
                &base_forecasts[i]
            } else if !base_forecasts.is_empty() {
                &base_forecasts[0] // Use first forecast as fallback
            } else {
                // Create default input if no base forecasts available
                let default_input = Array1::zeros(64);
                member_predictions.push(member.predict(&default_input, consciousness));
                continue;
            };
            
            let prediction = member.predict(input, consciousness);
            member_predictions.push(prediction);
        }
        
        member_predictions
    }
    
    /// Compute consciousness-weighted combination of member predictions
    fn compute_consciousness_weighted_combination(&self, member_predictions: &[Array1<f64>], consciousness: &ConsciousnessState) -> Array1<f64> {
        if member_predictions.is_empty() {
            return Array1::zeros(64); // Default dimension
        }
        
        let prediction_dim = member_predictions[0].len();
        let mut weighted_forecast = Array1::zeros(prediction_dim);
        let mut total_weight = 0.0;
        
        for (i, prediction) in member_predictions.iter().enumerate() {
            // Base weight from ensemble weights
            let base_weight = self.ensemble_weights[i];
            
            // Consciousness coherence weight
            let member_coherence = self.ensemble_members[i].consciousness_affinity * consciousness.coherence_level;
            let coherence_weight = member_coherence * consciousness.field_coherence;
            
            // Performance-based weight
            let performance_weight = self.ensemble_members[i].get_confidence();
            
            // Combine weights
            let combined_weight = base_weight * (1.0 + coherence_weight + performance_weight * 0.5);
            
            weighted_forecast = &weighted_forecast + &(prediction * combined_weight);
            total_weight += combined_weight;
        }
        
        // Normalize by total weight
        if total_weight > 0.0 {
            weighted_forecast = weighted_forecast / total_weight;
        }
        
        weighted_forecast
    }
    
    /// Apply diversity enhancement to ensemble prediction
    fn apply_diversity_enhancement(&self, base_forecast: &Array1<f64>, member_predictions: &[Array1<f64>], consciousness: &ConsciousnessState) -> Array1<f64> {
        let diversity_score = self.compute_prediction_diversity(member_predictions);
        let diversity_strength = self.diversity_bonus * diversity_score * consciousness.field_coherence;
        
        let mut enhanced_forecast = base_forecast.clone();
        
        // Add diversity-based corrections
        for (i, prediction) in member_predictions.iter().enumerate() {
            let member_deviation = prediction - base_forecast;
            let member_weight = self.ensemble_members[i].get_confidence();
            let diversity_contribution = &member_deviation * (diversity_strength * member_weight * 0.1);
            
            enhanced_forecast = &enhanced_forecast + &diversity_contribution;
        }
        
        enhanced_forecast
    }
    
    /// Compute diversity score of member predictions
    fn compute_prediction_diversity(&self, predictions: &[Array1<f64>]) -> f64 {
        if predictions.len() < 2 {
            return 0.0;
        }
        
        let mut total_variance = 0.0;
        let prediction_dim = predictions[0].len();
        
        for dim in 0..prediction_dim {
            let values: Vec<f64> = predictions.iter().map(|p| p[dim]).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            total_variance += variance;
        }
        
        let avg_variance = total_variance / prediction_dim as f64;
        
        // Normalize diversity score
        (avg_variance / (1.0 + avg_variance)).clamp(0.0, 1.0)
    }
    
    /// Rebalance ensemble weights based on performance and consciousness
    fn rebalance_ensemble(&mut self, consciousness: &ConsciousnessState) {
        let num_members = self.ensemble_members.len();
        let mut new_weights = Array1::zeros(num_members);
        
        // Compute weights based on performance and consciousness coherence
        for i in 0..num_members {
            let member = &self.ensemble_members[i];
            let performance_score = member.get_confidence();
            let consciousness_alignment = member.consciousness_affinity * consciousness.coherence_level;
            
            // Base weight combines performance and consciousness alignment
            let base_weight = performance_score * (1.0 + consciousness_alignment);
            
            // Add coherence bonus from other members
            let mut coherence_bonus = 0.0;
            for j in 0..num_members {
                if i != j {
                    coherence_bonus += self.consciousness_coherence_matrix[(i, j)];
                }
            }
            coherence_bonus /= (num_members - 1) as f64;
            
            new_weights[i] = base_weight * (1.0 + coherence_bonus * 0.1);
        }
        
        // Normalize weights
        let total_weight = new_weights.sum();
        if total_weight > 0.0 {
            new_weights = new_weights / total_weight;
        } else {
            new_weights = Array1::ones(num_members) / num_members as f64;
        }
        
        // Apply exponential smoothing to prevent sudden changes
        let smoothing = 0.2;
        self.ensemble_weights = &self.ensemble_weights * (1.0 - smoothing) + &new_weights * smoothing;
    }
    
    /// Update ensemble based on prediction performance
    pub fn update_performance(&mut self, actual_values: &Array1<f64>, consciousness: &ConsciousnessState) {
        if self.prediction_history.is_empty() {
            return;
        }
        
        let last_prediction = &self.prediction_history[self.prediction_history.len() - 1];
        
        // Compute overall performance score
        let performance_score = self.compute_performance_score(actual_values, last_prediction);
        
        // Store performance history
        self.performance_history.push_back(performance_score);
        if self.performance_history.len() > 1000 {
            self.performance_history.pop_front();
        }
        
        // Update individual ensemble members
        for member in &mut self.ensemble_members {
            // Compute member-specific performance (simplified)
            let member_performance = performance_score; // Could be more sophisticated
            member.update(member_performance, consciousness);
        }
        
        // Update ensemble parameters based on performance
        self.update_ensemble_parameters(performance_score, consciousness);
    }
    
    /// Compute performance score from actual vs predicted values
    fn compute_performance_score(&self, actual: &Array1<f64>, predicted: &Array1<f64>) -> f64 {
        if actual.len() != predicted.len() || actual.is_empty() {
            return 0.0;
        }
        
        let mse = actual.iter()
            .zip(predicted.iter())
            .map(|(a, p)| (a - p).powi(2))
            .sum::<f64>() / actual.len() as f64;
        
        // Convert MSE to performance score (0-1 range)
        1.0 / (1.0 + mse)
    }
    
    /// Update ensemble parameters based on performance feedback
    fn update_ensemble_parameters(&mut self, performance_score: f64, consciousness: &ConsciousnessState) {
        let learning_rate = 0.01;
        
        // Update diversity bonus
        if performance_score > 0.7 {
            self.diversity_bonus += learning_rate;
        } else if performance_score < 0.3 {
            self.diversity_bonus -= learning_rate * 0.5;
        }
        self.diversity_bonus = self.diversity_bonus.clamp(0.0, 0.5);
        
        // Update coherence threshold
        let consciousness_strength = consciousness.coherence_level * consciousness.field_coherence;
        if performance_score > 0.8 && consciousness_strength > 0.7 {
            self.coherence_threshold += learning_rate * 0.1;
        } else if performance_score < 0.2 {
            self.coherence_threshold -= learning_rate * 0.05;
        }
        self.coherence_threshold = self.coherence_threshold.clamp(0.1, 0.8);
        
        // Adjust rebalancing frequency based on performance stability
        let performance_stability = self.compute_performance_stability();
        if performance_stability > 0.8 {
            self.rebalancing_frequency = (self.rebalancing_frequency + 1).min(50);
        } else if performance_stability < 0.3 {
            self.rebalancing_frequency = (self.rebalancing_frequency.saturating_sub(1)).max(5);
        }
    }
    
    /// Compute stability of recent performance
    fn compute_performance_stability(&self) -> f64 {
        if self.performance_history.len() < 10 {
            return 0.5;
        }
        
        let recent_performance: Vec<f64> = self.performance_history.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        let mean = recent_performance.iter().sum::<f64>() / recent_performance.len() as f64;
        let variance = recent_performance.iter()
            .map(|p| (p - mean).powi(2))
            .sum::<f64>() / recent_performance.len() as f64;
        
        // Stability is inverse of variance
        1.0 / (1.0 + variance)
    }
    
    /// Get ensemble statistics for monitoring
    pub fn get_ensemble_stats(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("num_members".to_string(), self.ensemble_members.len() as f64);
        stats.insert("diversity_bonus".to_string(), self.diversity_bonus);
        stats.insert("coherence_threshold".to_string(), self.coherence_threshold);
        stats.insert("rebalancing_frequency".to_string(), self.rebalancing_frequency as f64);
        stats.insert("step_count".to_string(), self.step_count as f64);
        
        // Average member statistics
        if !self.ensemble_members.is_empty() {
            let avg_consciousness_affinity: f64 = self.ensemble_members.iter()
                .map(|m| m.consciousness_affinity)
                .sum::<f64>() / self.ensemble_members.len() as f64;
            
            let avg_confidence: f64 = self.ensemble_members.iter()
                .map(|m| m.get_confidence())
                .sum::<f64>() / self.ensemble_members.len() as f64;
            
            stats.insert("avg_consciousness_affinity".to_string(), avg_consciousness_affinity);
            stats.insert("avg_member_confidence".to_string(), avg_confidence);
        }
        
        // Performance statistics
        if !self.performance_history.is_empty() {
            let recent_performance: f64 = self.performance_history.iter()
                .rev()
                .take(10)
                .sum::<f64>() / 10.0.min(self.performance_history.len() as f64);
            
            stats.insert("recent_performance".to_string(), recent_performance);
            stats.insert("performance_stability".to_string(), self.compute_performance_stability());
        }
        
        // Weight distribution entropy
        let weight_entropy = self.compute_weight_entropy();
        stats.insert("weight_entropy".to_string(), weight_entropy);
        
        stats
    }
    
    /// Compute entropy of ensemble weight distribution
    fn compute_weight_entropy(&self) -> f64 {
        let mut entropy = 0.0;
        
        for &weight in self.ensemble_weights.iter() {
            if weight > 1e-10 {
                entropy -= weight * weight.ln();
            }
        }
        
        entropy
    }
}