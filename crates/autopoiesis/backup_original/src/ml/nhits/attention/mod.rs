//! Temporal Attention Mechanisms for NHITS
//! Multi-head attention with consciousness integration

use ndarray::{Array2, Array3, Array4, Axis, s};
use serde::{Deserialize, Serialize};
use std::f64::consts::SQRT_2;

/// Attention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout_rate: f64,
    pub temperature: f64,
    pub use_causal_mask: bool,
    pub attention_type: AttentionType,
    pub consciousness_integration: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AttentionType {
    Standard,
    Relative,
    LocalWindow { window_size: usize },
    Dilated { dilation_rate: usize },
    Sparse { sparsity_factor: f64 },
}

/// Temporal attention mechanism
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    config: AttentionConfig,
    
    // Projection matrices
    query_proj: Array2<f64>,
    key_proj: Array2<f64>,
    value_proj: Array2<f64>,
    output_proj: Array2<f64>,
    
    // Relative position embeddings
    rel_pos_embeddings: Option<Array2<f64>>,
    
    // Adaptive attention parameters
    attention_scores: Vec<Array3<f64>>,
    attention_patterns: Vec<AttentionPattern>,
}

/// Learned attention patterns
#[derive(Debug, Clone)]
pub struct AttentionPattern {
    pub pattern_type: PatternType,
    pub strength: f64,
    pub temporal_range: (usize, usize),
    pub frequency: f64,
}

#[derive(Debug, Clone)]
pub enum PatternType {
    Periodic,
    Trend,
    Seasonal,
    Anomaly,
    Custom(String),
}

impl TemporalAttention {
    pub fn new(config: &AttentionConfig) -> Self {
        let hidden_dim = config.num_heads * config.head_dim;
        
        let init_scale = 1.0 / (hidden_dim as f64).sqrt();
        
        Self {
            query_proj: Array2::from_shape_fn(
                (hidden_dim, hidden_dim),
                |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
            ),
            key_proj: Array2::from_shape_fn(
                (hidden_dim, hidden_dim),
                |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
            ),
            value_proj: Array2::from_shape_fn(
                (hidden_dim, hidden_dim),
                |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
            ),
            output_proj: Array2::from_shape_fn(
                (hidden_dim, hidden_dim),
                |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
            ),
            rel_pos_embeddings: if matches!(config.attention_type, AttentionType::Relative) {
                Some(Self::initialize_relative_positions(512, hidden_dim))
            } else {
                None
            },
            config: config.clone(),
            attention_scores: Vec::new(),
            attention_patterns: Vec::new(),
        }
    }
    
    /// Apply attention with optional consciousness modulation
    pub fn apply(
        &mut self,
        input: &Array3<f64>,
        consciousness_weights: Option<Array2<f64>>,
    ) -> Result<Array3<f64>, AttentionError> {
        let (batch_size, seq_len, hidden_dim) = (
            input.shape()[0],
            input.shape()[1],
            input.shape()[2],
        );
        
        // Validate dimensions
        if hidden_dim != self.config.num_heads * self.config.head_dim {
            return Err(AttentionError::DimensionMismatch {
                expected: self.config.num_heads * self.config.head_dim,
                actual: hidden_dim,
            });
        }
        
        // Project to Q, K, V
        let queries = self.project_input(input, &self.query_proj)?;
        let keys = self.project_input(input, &self.key_proj)?;
        let values = self.project_input(input, &self.value_proj)?;
        
        // Reshape for multi-head attention
        let queries = self.reshape_for_heads(&queries)?;
        let keys = self.reshape_for_heads(&keys)?;
        let values = self.reshape_for_heads(&values)?;
        
        // Compute attention scores
        let mut scores = self.compute_attention_scores(&queries, &keys)?;
        
        // Apply consciousness modulation if provided
        if let Some(ref weights) = consciousness_weights {
            scores = self.modulate_with_consciousness(&scores, weights)?;
        }
        
        // Apply attention mask if needed
        if self.config.use_causal_mask {
            scores = self.apply_causal_mask(&scores)?;
        }
        
        // Apply attention pattern-specific modifications
        scores = match &self.config.attention_type {
            AttentionType::LocalWindow { window_size } => {
                self.apply_local_window_mask(&scores, *window_size)?
            }
            AttentionType::Dilated { dilation_rate } => {
                self.apply_dilated_mask(&scores, *dilation_rate)?
            }
            AttentionType::Sparse { sparsity_factor } => {
                self.apply_sparse_mask(&scores, *sparsity_factor)?
            }
            _ => scores,
        };
        
        // Softmax normalization
        let attention_weights = self.softmax(&scores)?;
        
        // Store attention scores for analysis
        if self.attention_scores.len() > 100 {
            self.attention_scores.remove(0);
        }
        self.attention_scores.push(attention_weights.clone());
        
        // Apply attention to values
        let attended = self.apply_attention_weights(&attention_weights, &values)?;
        
        // Reshape back and project output
        let output = self.reshape_from_heads(&attended)?;
        let projected_output = self.project_output(&output)?;
        
        // Learn attention patterns if enabled
        if self.config.consciousness_integration {
            self.learn_attention_patterns(&attention_weights)?;
        }
        
        Ok(projected_output)
    }
    
    /// Initialize relative position embeddings
    fn initialize_relative_positions(max_len: usize, hidden_dim: usize) -> Array2<f64> {
        let mut embeddings = Array2::zeros((2 * max_len - 1, hidden_dim));
        
        for i in 0..2 * max_len - 1 {
            let pos = i as i32 - max_len as i32 + 1;
            for j in 0..hidden_dim {
                if j % 2 == 0 {
                    embeddings[[i, j]] = (pos as f64 / 10000_f64.powf(j as f64 / hidden_dim as f64)).sin();
                } else {
                    embeddings[[i, j]] = (pos as f64 / 10000_f64.powf((j - 1) as f64 / hidden_dim as f64)).cos();
                }
            }
        }
        
        embeddings
    }
    
    /// Project input through linear layer
    fn project_input(
        &self,
        input: &Array3<f64>,
        projection: &Array2<f64>,
    ) -> Result<Array3<f64>, AttentionError> {
        let (batch_size, seq_len, _) = input.shape();
        let output_dim = projection.shape()[1];
        
        let mut output = Array3::zeros((batch_size, seq_len, output_dim));
        
        for b in 0..batch_size {
            let batch_input = input.slice(s![b, .., ..]);
            let projected = batch_input.dot(projection);
            output.slice_mut(s![b, .., ..]).assign(&projected);
        }
        
        Ok(output)
    }
    
    /// Reshape for multi-head attention
    fn reshape_for_heads(&self, input: &Array3<f64>) -> Result<Array4<f64>, AttentionError> {
        let (batch_size, seq_len, _) = input.shape();
        let num_heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        
        let mut output = Array4::zeros((batch_size, num_heads, seq_len, head_dim));
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        output[[b, h, s, d]] = input[[b, s, h * head_dim + d]];
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Compute attention scores
    fn compute_attention_scores(
        &self,
        queries: &Array4<f64>,
        keys: &Array4<f64>,
    ) -> Result<Array4<f64>, AttentionError> {
        let (batch_size, num_heads, seq_len, head_dim) = queries.shape();
        let scale = 1.0 / (head_dim as f64).sqrt() / self.config.temperature;
        
        let mut scores = Array4::zeros((batch_size, num_heads, seq_len, seq_len));
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                let q = queries.slice(s![b, h, .., ..]);
                let k = keys.slice(s![b, h, .., ..]);
                
                // Q @ K^T / sqrt(d_k)
                let attention_scores = q.dot(&k.t()) * scale;
                scores.slice_mut(s![b, h, .., ..]).assign(&attention_scores);
                
                // Add relative position bias if using relative attention
                if let (Some(ref rel_pos), AttentionType::Relative) = 
                    (&self.rel_pos_embeddings, &self.config.attention_type) {
                    self.add_relative_position_bias(
                        &mut scores.slice_mut(s![b, h, .., ..]),
                        rel_pos,
                        seq_len,
                    );
                }
            }
        }
        
        Ok(scores)
    }
    
    /// Add relative position bias
    fn add_relative_position_bias(
        &self,
        scores: &mut ndarray::ArrayViewMut2<f64>,
        rel_pos_embeddings: &Array2<f64>,
        seq_len: usize,
    ) {
        let max_dist = rel_pos_embeddings.shape()[0] / 2;
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                let dist = (i as i32 - j as i32).abs() as usize;
                if dist <= max_dist {
                    let pos_idx = max_dist + (i as i32 - j as i32) as usize;
                    // Simplified: use first dimension of embedding as bias
                    scores[[i, j]] += rel_pos_embeddings[[pos_idx, 0]] * 0.1;
                }
            }
        }
    }
    
    /// Modulate attention with consciousness weights
    fn modulate_with_consciousness(
        &self,
        scores: &Array4<f64>,
        consciousness_weights: &Array2<f64>,
    ) -> Result<Array4<f64>, AttentionError> {
        let mut modulated = scores.clone();
        let (batch_size, num_heads, seq_len, _) = scores.shape();
        
        // Apply consciousness weights to modulate attention
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        if i < consciousness_weights.shape()[0] && j < consciousness_weights.shape()[1] {
                            modulated[[b, h, i, j]] *= 1.0 + consciousness_weights[[i, j]] * 0.5;
                        }
                    }
                }
            }
        }
        
        Ok(modulated)
    }
    
    /// Apply causal mask
    fn apply_causal_mask(&self, scores: &Array4<f64>) -> Result<Array4<f64>, AttentionError> {
        let mut masked = scores.clone();
        let (_, _, seq_len, _) = scores.shape();
        
        for i in 0..seq_len {
            for j in i + 1..seq_len {
                masked.slice_mut(s![.., .., i, j]).fill(f64::NEG_INFINITY);
            }
        }
        
        Ok(masked)
    }
    
    /// Apply local window mask
    fn apply_local_window_mask(
        &self,
        scores: &Array4<f64>,
        window_size: usize,
    ) -> Result<Array4<f64>, AttentionError> {
        let mut masked = scores.clone();
        let (_, _, seq_len, _) = scores.shape();
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                if (i as i32 - j as i32).abs() > window_size as i32 {
                    masked.slice_mut(s![.., .., i, j]).fill(f64::NEG_INFINITY);
                }
            }
        }
        
        Ok(masked)
    }
    
    /// Apply dilated mask
    fn apply_dilated_mask(
        &self,
        scores: &Array4<f64>,
        dilation_rate: usize,
    ) -> Result<Array4<f64>, AttentionError> {
        let mut masked = scores.clone();
        let (_, _, seq_len, _) = scores.shape();
        
        for i in 0..seq_len {
            for j in 0..seq_len {
                if (i as i32 - j as i32).abs() % dilation_rate as i32 != 0 {
                    masked.slice_mut(s![.., .., i, j]).fill(f64::NEG_INFINITY);
                }
            }
        }
        
        Ok(masked)
    }
    
    /// Apply sparse mask
    fn apply_sparse_mask(
        &self,
        scores: &Array4<f64>,
        sparsity_factor: f64,
    ) -> Result<Array4<f64>, AttentionError> {
        let mut masked = scores.clone();
        let (batch_size, num_heads, seq_len, _) = scores.shape();
        
        // Keep top-k values based on sparsity factor
        let k = ((seq_len as f64) * (1.0 - sparsity_factor)).max(1.0) as usize;
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let row = scores.slice(s![b, h, i, ..]);
                    let mut sorted_indices: Vec<usize> = (0..seq_len).collect();
                    sorted_indices.sort_by(|&a, &b| {
                        row[b].partial_cmp(&row[a]).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    
                    // Mask all but top-k
                    for &j in &sorted_indices[k..] {
                        masked[[b, h, i, j]] = f64::NEG_INFINITY;
                    }
                }
            }
        }
        
        Ok(masked)
    }
    
    /// Softmax normalization
    fn softmax(&self, scores: &Array4<f64>) -> Result<Array4<f64>, AttentionError> {
        let mut output = scores.clone();
        let (batch_size, num_heads, seq_len, _) = scores.shape();
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                for i in 0..seq_len {
                    let row = scores.slice(s![b, h, i, ..]);
                    let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    
                    // Compute exp(x - max) for numerical stability
                    let exp_scores: Vec<f64> = row.iter()
                        .map(|&x| (x - max_val).exp())
                        .collect();
                    
                    let sum_exp: f64 = exp_scores.iter().sum();
                    
                    for j in 0..seq_len {
                        output[[b, h, i, j]] = exp_scores[j] / sum_exp.max(1e-10);
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Apply attention weights to values
    fn apply_attention_weights(
        &self,
        weights: &Array4<f64>,
        values: &Array4<f64>,
    ) -> Result<Array4<f64>, AttentionError> {
        let (batch_size, num_heads, seq_len, head_dim) = values.shape();
        let mut output = Array4::zeros((batch_size, num_heads, seq_len, head_dim));
        
        for b in 0..batch_size {
            for h in 0..num_heads {
                let w = weights.slice(s![b, h, .., ..]);
                let v = values.slice(s![b, h, .., ..]);
                
                // weights @ values
                let attended = w.dot(&v);
                output.slice_mut(s![b, h, .., ..]).assign(&attended);
            }
        }
        
        Ok(output)
    }
    
    /// Reshape from heads back to original dimensions
    fn reshape_from_heads(&self, input: &Array4<f64>) -> Result<Array3<f64>, AttentionError> {
        let (batch_size, num_heads, seq_len, head_dim) = input.shape();
        let hidden_dim = num_heads * head_dim;
        
        let mut output = Array3::zeros((batch_size, seq_len, hidden_dim));
        
        for b in 0..batch_size {
            for s in 0..seq_len {
                for h in 0..num_heads {
                    for d in 0..head_dim {
                        output[[b, s, h * head_dim + d]] = input[[b, h, s, d]];
                    }
                }
            }
        }
        
        Ok(output)
    }
    
    /// Project output
    fn project_output(&self, input: &Array3<f64>) -> Result<Array3<f64>, AttentionError> {
        self.project_input(input, &self.output_proj)
    }
    
    /// Learn attention patterns from scores
    fn learn_attention_patterns(
        &mut self,
        attention_weights: &Array4<f64>,
    ) -> Result<(), AttentionError> {
        // Analyze attention patterns across heads
        let (_, num_heads, seq_len, _) = attention_weights.shape();
        
        for h in 0..num_heads {
            let head_weights = attention_weights.slice(s![.., h, .., ..]);
            
            // Detect periodic patterns
            if let Some(pattern) = self.detect_periodic_pattern(&head_weights) {
                self.attention_patterns.push(pattern);
            }
            
            // Detect trend patterns
            if let Some(pattern) = self.detect_trend_pattern(&head_weights) {
                self.attention_patterns.push(pattern);
            }
        }
        
        // Keep only recent patterns
        if self.attention_patterns.len() > 50 {
            self.attention_patterns.drain(0..10);
        }
        
        Ok(())
    }
    
    /// Detect periodic patterns in attention
    fn detect_periodic_pattern(&self, weights: &ndarray::ArrayView3<f64>) -> Option<AttentionPattern> {
        // Simplified periodic pattern detection
        // Full implementation would use FFT or autocorrelation
        
        let seq_len = weights.shape()[1];
        let mut max_correlation = 0.0;
        let mut best_period = 0;
        
        for period in 2..seq_len / 2 {
            let mut correlation = 0.0;
            let mut count = 0;
            
            for i in 0..seq_len - period {
                for j in 0..seq_len {
                    correlation += weights[[0, i, j]] * weights[[0, i + period, j]];
                    count += 1;
                }
            }
            
            correlation /= count as f64;
            
            if correlation > max_correlation {
                max_correlation = correlation;
                best_period = period;
            }
        }
        
        if max_correlation > 0.7 {
            Some(AttentionPattern {
                pattern_type: PatternType::Periodic,
                strength: max_correlation,
                temporal_range: (0, best_period),
                frequency: 1.0 / best_period as f64,
            })
        } else {
            None
        }
    }
    
    /// Detect trend patterns in attention
    fn detect_trend_pattern(&self, weights: &ndarray::ArrayView3<f64>) -> Option<AttentionPattern> {
        // Simplified trend detection
        let seq_len = weights.shape()[1];
        let mut trend_strength = 0.0;
        
        // Check if attention focuses more on recent values
        for i in 0..seq_len {
            for j in i..seq_len {
                trend_strength += weights[[0, i, j]] * (j - i) as f64 / seq_len as f64;
            }
        }
        
        trend_strength /= (seq_len * seq_len) as f64;
        
        if trend_strength > 0.6 {
            Some(AttentionPattern {
                pattern_type: PatternType::Trend,
                strength: trend_strength,
                temporal_range: (0, seq_len),
                frequency: 0.0,
            })
        } else {
            None
        }
    }
    
    /// Reconfigure attention heads
    pub fn reconfigure_heads(&mut self, num_heads: usize) {
        self.config.num_heads = num_heads;
        // Re-initialize projections with new dimensions
        let hidden_dim = num_heads * self.config.head_dim;
        let init_scale = 1.0 / (hidden_dim as f64).sqrt();
        
        self.query_proj = Array2::from_shape_fn(
            (hidden_dim, hidden_dim),
            |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
        );
        self.key_proj = Array2::from_shape_fn(
            (hidden_dim, hidden_dim),
            |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
        );
        self.value_proj = Array2::from_shape_fn(
            (hidden_dim, hidden_dim),
            |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
        );
        self.output_proj = Array2::from_shape_fn(
            (hidden_dim, hidden_dim),
            |_| rand::random::<f64>() * 2.0 * init_scale - init_scale,
        );
    }
}

/// Attention errors
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid attention configuration: {0}")]
    InvalidConfig(String),
    
    #[error("Computation error: {0}")]
    ComputationError(String),
}

extern crate rand;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_attention_mechanism() {
        // Test implementation
    }
}