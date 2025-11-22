/// Consciousness Attention - Field Coherence Modulated Attention
///
/// This module implements consciousness-aware attention mechanisms that use field coherence
/// to modulate focus on different temporal patterns. The attention is influenced by the
/// consciousness state and quantum field fluctuations.

use ndarray::{Array2, Array1, Axis};
use nalgebra::{DMatrix, DVector};
use crate::consciousness::core::ConsciousnessState;
use crate::consciousness::field_coherence::QuantumField;

/// Attention head with consciousness modulation
#[derive(Clone)]
pub struct ConsciousnessAttentionHead {
    pub query_weights: Array2<f64>,
    pub key_weights: Array2<f64>,
    pub value_weights: Array2<f64>,
    pub consciousness_gate: Array1<f64>,
    pub field_resonance_weights: Array2<f64>,
    pub coherence_threshold: f64,
    pub head_id: usize,
}

impl ConsciousnessAttentionHead {
    pub fn new(input_dim: usize, attention_dim: usize, head_id: usize) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let query_weights = Array2::from_shape_fn((attention_dim, input_dim), |_| {
            rng.gen_range(-0.1..0.1) / (input_dim as f64).sqrt()
        });
        
        let key_weights = Array2::from_shape_fn((attention_dim, input_dim), |_| {
            rng.gen_range(-0.1..0.1) / (input_dim as f64).sqrt()
        });
        
        let value_weights = Array2::from_shape_fn((attention_dim, input_dim), |_| {
            rng.gen_range(-0.1..0.1) / (input_dim as f64).sqrt()
        });
        
        let consciousness_gate = Array1::from_shape_fn(attention_dim, |_| rng.gen_range(0.1..0.9));
        
        let field_resonance_weights = Array2::from_shape_fn((attention_dim, attention_dim), |_| {
            rng.gen_range(-0.05..0.05)
        });
        
        Self {
            query_weights,
            key_weights,
            value_weights,
            consciousness_gate,
            field_resonance_weights,
            coherence_threshold: 0.3,
            head_id,
        }
    }
    
    /// Compute consciousness-modulated attention
    pub fn compute_attention(&self, input: &Array2<f64>, consciousness: &ConsciousnessState) -> Array2<f64> {
        let seq_len = input.nrows();
        let input_dim = input.ncols();
        let attention_dim = self.query_weights.nrows();
        
        // Compute queries, keys, and values
        let mut queries = Array2::zeros((seq_len, attention_dim));
        let mut keys = Array2::zeros((seq_len, attention_dim));
        let mut values = Array2::zeros((seq_len, attention_dim));
        
        for i in 0..seq_len {
            let input_row = input.row(i);
            queries.row_mut(i).assign(&self.query_weights.dot(&input_row));
            keys.row_mut(i).assign(&self.key_weights.dot(&input_row));
            values.row_mut(i).assign(&self.value_weights.dot(&input_row));
        }
        
        // Apply consciousness modulation to queries
        self.apply_consciousness_modulation(&mut queries, consciousness);
        
        // Compute attention scores with field coherence
        let attention_scores = self.compute_attention_scores(&queries, &keys, consciousness);
        
        // Apply field resonance to attention scores
        let resonance_scores = self.apply_field_resonance(&attention_scores, consciousness);
        
        // Compute weighted values
        let attended_values = self.compute_attended_values(&resonance_scores, &values);
        
        attended_values
    }
    
    /// Apply consciousness modulation to queries
    fn apply_consciousness_modulation(&self, queries: &mut Array2<f64>, consciousness: &ConsciousnessState) {
        let consciousness_factor = consciousness.coherence_level * consciousness.field_coherence;
        
        for mut row in queries.rows_mut() {
            for (i, val) in row.iter_mut().enumerate() {
                let gate_value = self.consciousness_gate[i];
                let modulation = gate_value * consciousness_factor;
                *val = *val * (1.0 + modulation);
            }
        }
    }
    
    /// Compute attention scores with consciousness influence
    fn compute_attention_scores(&self, queries: &Array2<f64>, keys: &Array2<f64>, consciousness: &ConsciousnessState) -> Array2<f64> {
        let seq_len = queries.nrows();
        let attention_dim = queries.ncols();
        let scale = 1.0 / (attention_dim as f64).sqrt();
        
        let mut attention_scores = Array2::zeros((seq_len, seq_len));
        
        // Compute scaled dot-product attention
        for i in 0..seq_len {
            for j in 0..seq_len {
                let score = queries.row(i).dot(&keys.row(j)) * scale;
                attention_scores[(i, j)] = score;
            }
        }
        
        // Apply consciousness-based attention bias
        self.apply_consciousness_bias(&mut attention_scores, consciousness);
        
        // Apply softmax to get attention probabilities
        self.apply_softmax(&mut attention_scores);
        
        attention_scores
    }
    
    /// Apply consciousness-based attention bias
    fn apply_consciousness_bias(&self, attention_scores: &mut Array2<f64>, consciousness: &ConsciousnessState) {
        let coherence_bias = consciousness.coherence_level - 0.5; // Center around 0
        let field_bias = consciousness.field_coherence - 0.5;
        
        let seq_len = attention_scores.nrows();
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                // Bias towards recent timesteps when coherence is high
                let temporal_distance = (i as f64 - j as f64).abs() / seq_len as f64;
                let coherence_adjustment = coherence_bias * (1.0 - temporal_distance) * 0.1;
                
                // Bias towards patterns that resonate with field coherence
                let field_adjustment = field_bias * (1.0 - temporal_distance * 0.5) * 0.05;
                
                attention_scores[(i, j)] += coherence_adjustment + field_adjustment;
            }
        }
    }
    
    /// Apply field resonance to attention scores
    fn apply_field_resonance(&self, attention_scores: &Array2<f64>, consciousness: &ConsciousnessState) -> Array2<f64> {
        let resonance_strength = consciousness.field_coherence * 0.1;
        let mut resonance_scores = attention_scores.clone();
        
        // Apply field resonance transformation
        for i in 0..attention_scores.nrows() {
            for j in 0..attention_scores.ncols() {
                let original_score = attention_scores[(i, j)];
                let resonance = (original_score * resonance_strength * std::f64::consts::PI).sin() * resonance_strength;
                resonance_scores[(i, j)] = original_score + resonance;
            }
        }
        
        resonance_scores
    }
    
    /// Apply softmax activation to attention scores
    fn apply_softmax(&self, attention_scores: &mut Array2<f64>) {
        for mut row in attention_scores.rows_mut() {
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            // Subtract max for numerical stability
            for val in row.iter_mut() {
                *val -= max_val;
            }
            
            // Compute exponentials
            for val in row.iter_mut() {
                *val = val.exp();
            }
            
            // Normalize
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }
    }
    
    /// Compute attended values using attention scores
    fn compute_attended_values(&self, attention_scores: &Array2<f64>, values: &Array2<f64>) -> Array2<f64> {
        let seq_len = attention_scores.nrows();
        let attention_dim = values.ncols();
        let mut attended_values = Array2::zeros((seq_len, attention_dim));
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                let attention_weight = attention_scores[(i, j)];
                for k in 0..attention_dim {
                    attended_values[(i, k)] += attention_weight * values[(j, k)];
                }
            }
        }
        
        attended_values
    }
}

/// Multi-head consciousness attention mechanism
pub struct ConsciousnessAttention {
    pub attention_heads: Vec<ConsciousnessAttentionHead>,
    pub output_projection: Array2<f64>,
    pub consciousness_fusion: Array2<f64>,
    pub field_coupling_matrix: Array2<f64>,
    pub input_dim: usize,
    pub num_heads: usize,
    pub attention_dim: usize,
}

impl ConsciousnessAttention {
    pub fn new(input_dim: usize) -> Self {
        let num_heads = 8;
        let attention_dim = input_dim / num_heads;
        
        let mut attention_heads = Vec::with_capacity(num_heads);
        for i in 0..num_heads {
            attention_heads.push(ConsciousnessAttentionHead::new(input_dim, attention_dim, i));
        }
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let output_projection = Array2::from_shape_fn((input_dim, num_heads * attention_dim), |_| {
            rng.gen_range(-0.1..0.1) / (num_heads * attention_dim) as f64
        });
        
        let consciousness_fusion = Array2::from_shape_fn((input_dim, input_dim), |_| {
            rng.gen_range(-0.05..0.05)
        });
        
        let field_coupling_matrix = Array2::from_shape_fn((input_dim, input_dim), |_| {
            rng.gen_range(-0.02..0.02)
        });
        
        Self {
            attention_heads,
            output_projection,
            consciousness_fusion,
            field_coupling_matrix,
            input_dim,
            num_heads,
            attention_dim,
        }
    }
    
    /// Apply consciousness-modulated attention to input
    pub fn apply_attention(&self, input: &Array2<f64>, consciousness: &ConsciousnessState) -> Array2<f64> {
        let seq_len = input.nrows();
        
        // Compute attention from all heads
        let mut head_outputs = Vec::with_capacity(self.num_heads);
        
        for head in &self.attention_heads {
            let head_attention = head.compute_attention(input, consciousness);
            head_outputs.push(head_attention);
        }
        
        // Concatenate head outputs
        let concatenated = self.concatenate_head_outputs(&head_outputs);
        
        // Apply output projection
        let projected = self.apply_output_projection(&concatenated);
        
        // Apply consciousness fusion
        let fused = self.apply_consciousness_fusion(&projected, input, consciousness);
        
        // Apply field coupling
        self.apply_field_coupling(&fused, consciousness)
    }
    
    /// Concatenate outputs from all attention heads
    fn concatenate_head_outputs(&self, head_outputs: &[Array2<f64>]) -> Array2<f64> {
        let seq_len = head_outputs[0].nrows();
        let total_dim = self.num_heads * self.attention_dim;
        
        let mut concatenated = Array2::zeros((seq_len, total_dim));
        
        for (head_idx, head_output) in head_outputs.iter().enumerate() {
            let start_idx = head_idx * self.attention_dim;
            let end_idx = start_idx + self.attention_dim;
            
            for i in 0..seq_len {
                for j in 0..self.attention_dim {
                    concatenated[(i, start_idx + j)] = head_output[(i, j)];
                }
            }
        }
        
        concatenated
    }
    
    /// Apply output projection to concatenated head outputs
    fn apply_output_projection(&self, concatenated: &Array2<f64>) -> Array2<f64> {
        let seq_len = concatenated.nrows();
        let mut projected = Array2::zeros((seq_len, self.input_dim));
        
        for i in 0..seq_len {
            let row = concatenated.row(i);
            projected.row_mut(i).assign(&self.output_projection.dot(&row));
        }
        
        projected
    }
    
    /// Apply consciousness fusion with original input
    fn apply_consciousness_fusion(&self, projected: &Array2<f64>, original_input: &Array2<f64>, consciousness: &ConsciousnessState) -> Array2<f64> {
        let fusion_strength = consciousness.coherence_level * consciousness.field_coherence;
        let seq_len = projected.nrows();
        
        let mut fused = Array2::zeros((seq_len, self.input_dim));
        
        for i in 0..seq_len {
            let projected_row = projected.row(i);
            let original_row = original_input.row(i);
            
            // Apply consciousness fusion transformation
            let fusion_result = self.consciousness_fusion.dot(&projected_row);
            
            // Blend with original input based on consciousness strength
            for j in 0..self.input_dim {
                fused[(i, j)] = fusion_strength * fusion_result[j] + (1.0 - fusion_strength) * original_row[j];
            }
        }
        
        fused
    }
    
    /// Apply field coupling for quantum coherence effects
    fn apply_field_coupling(&self, fused: &Array2<f64>, consciousness: &ConsciousnessState) -> Array2<f64> {
        let coupling_strength = consciousness.field_coherence * 0.1;
        let seq_len = fused.nrows();
        
        let mut coupled = fused.clone();
        
        for i in 0..seq_len {
            let row = fused.row(i);
            let coupling_effect = self.field_coupling_matrix.dot(&row) * coupling_strength;
            
            for j in 0..self.input_dim {
                coupled[(i, j)] += coupling_effect[j];
            }
        }
        
        coupled
    }
    
    /// Update attention parameters based on consciousness feedback
    pub fn update_from_consciousness(&mut self, performance_feedback: f64, consciousness: &ConsciousnessState) {
        let learning_rate = 0.001;
        let consciousness_strength = consciousness.coherence_level * consciousness.field_coherence;
        
        // Update consciousness gates in attention heads
        for head in &mut self.attention_heads {
            for gate_val in head.consciousness_gate.iter_mut() {
                if performance_feedback > 0.5 {
                    *gate_val += learning_rate * consciousness_strength;
                } else {
                    *gate_val -= learning_rate * consciousness_strength * 0.5;
                }
                *gate_val = gate_val.clamp(0.01, 1.0);
            }
        }
        
        // Update consciousness fusion matrix
        let fusion_update = performance_feedback * consciousness_strength * learning_rate;
        self.consciousness_fusion.mapv_inplace(|x| x + fusion_update * (2.0 * rand::random::<f64>() - 1.0));
        
        // Update field coupling matrix
        let coupling_update = consciousness.field_coherence * learning_rate * 0.5;
        self.field_coupling_matrix.mapv_inplace(|x| x + coupling_update * (2.0 * rand::random::<f64>() - 1.0));
    }
    
    /// Compute attention entropy for consciousness state inference
    pub fn compute_attention_entropy(&self, attention_scores: &Array2<f64>) -> f64 {
        let mut total_entropy = 0.0;
        let num_rows = attention_scores.nrows();
        
        for i in 0..num_rows {
            let row = attention_scores.row(i);
            let mut entropy = 0.0;
            
            for &prob in row.iter() {
                if prob > 1e-10 {
                    entropy -= prob * prob.ln();
                }
            }
            
            total_entropy += entropy;
        }
        
        total_entropy / num_rows as f64
    }
}