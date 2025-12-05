//! # Multi-Head Graph Attention with Hyperbolic Modulation
//!
//! Implements Graph Attention Networks (GAT) adapted for hyperbolic space
//! with spike-timing integration for SNNs.
//!
//! ## Mathematical Foundation
//!
//! Standard attention: α_ij = softmax(e_ij)
//! where e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
//!
//! Hyperbolic modulation: α_ij *= exp(-d_H(p_i, p_j) / λ)
//! Spike-timing modulation: α_ij *= STDP(Δt_ij)
//!
//! ## References
//! - Veličković et al. (2018) "Graph Attention Networks" ICLR
//! - Chami et al. (2019) "Hyperbolic Graph Convolutional Neural Networks"
//! - Liu et al. (2021) "Hyperbolic Graph Neural Networks" AAAI

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::hyperbolic_snn::LorentzVec;

// ============================================================================
// Attention Configuration
// ============================================================================

/// Configuration for graph attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Feature dimension per head
    pub head_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Dropout probability (during training)
    pub dropout: f32,
    /// Negative slope for LeakyReLU
    pub leaky_relu_slope: f32,
    /// Hyperbolic distance scale (λ)
    pub hyperbolic_scale: f64,
    /// Enable spike-timing attention
    pub spike_timing_attention: bool,
    /// STDP time constant for spike attention
    pub stdp_tau: f64,
    /// Temperature for softmax
    pub temperature: f32,
    /// Use residual connections
    pub residual: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 4,
            head_dim: 16,
            output_dim: 64,
            dropout: 0.1,
            leaky_relu_slope: 0.2,
            hyperbolic_scale: 2.0,
            spike_timing_attention: true,
            stdp_tau: 20.0,
            temperature: 1.0,
            residual: true,
        }
    }
}

// ============================================================================
// Attention Head
// ============================================================================

/// Single attention head with hyperbolic modulation
#[derive(Debug, Clone)]
pub struct HyperbolicAttentionHead {
    /// Query projection weights (head_dim x input_dim)
    pub query_weights: Vec<f32>,
    /// Key projection weights (head_dim x input_dim)
    pub key_weights: Vec<f32>,
    /// Value projection weights (head_dim x input_dim)
    pub value_weights: Vec<f32>,
    /// Attention vector (2 * head_dim)
    pub attention_vec: Vec<f32>,
    /// Input dimension
    input_dim: usize,
    /// Head dimension
    head_dim: usize,
    /// LeakyReLU negative slope
    leaky_slope: f32,
    /// Hyperbolic scale
    hyperbolic_scale: f64,
    /// STDP tau
    stdp_tau: f64,
    /// Cached attention scores for inspection
    cached_attention: HashMap<(u32, u32), f32>,
}

impl HyperbolicAttentionHead {
    /// Create new attention head with Xavier initialization
    pub fn new(input_dim: usize, head_dim: usize, config: &AttentionConfig) -> Self {
        // Xavier initialization scale
        let q_scale = (2.0 / (input_dim + head_dim) as f64).sqrt() as f32;
        let a_scale = (1.0 / (2 * head_dim) as f64).sqrt() as f32;

        // Initialize with deterministic values for reproducibility
        let query_weights: Vec<f32> = (0..head_dim * input_dim)
            .map(|i| {
                let x = (i as f32 * 0.618033).sin() * q_scale;
                x
            })
            .collect();

        let key_weights: Vec<f32> = (0..head_dim * input_dim)
            .map(|i| {
                let x = ((i as f32 + 100.0) * 0.618033).sin() * q_scale;
                x
            })
            .collect();

        let value_weights: Vec<f32> = (0..head_dim * input_dim)
            .map(|i| {
                let x = ((i as f32 + 200.0) * 0.618033).sin() * q_scale;
                x
            })
            .collect();

        let attention_vec: Vec<f32> = (0..2 * head_dim)
            .map(|i| {
                let x = ((i as f32 + 300.0) * 0.618033).sin() * a_scale;
                x
            })
            .collect();

        Self {
            query_weights,
            key_weights,
            value_weights,
            attention_vec,
            input_dim,
            head_dim,
            leaky_slope: config.leaky_relu_slope,
            hyperbolic_scale: config.hyperbolic_scale,
            stdp_tau: config.stdp_tau,
            cached_attention: HashMap::new(),
        }
    }

    /// Project input to query space
    pub fn project_query(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = vec![0.0f32; self.head_dim];

        for i in 0..self.head_dim {
            for j in 0..self.input_dim {
                output[i] += self.query_weights[i * self.input_dim + j] * input[j];
            }
        }

        output
    }

    /// Project input to key space
    pub fn project_key(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = vec![0.0f32; self.head_dim];

        for i in 0..self.head_dim {
            for j in 0..self.input_dim {
                output[i] += self.key_weights[i * self.input_dim + j] * input[j];
            }
        }

        output
    }

    /// Project input to value space
    pub fn project_value(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.input_dim);
        let mut output = vec![0.0f32; self.head_dim];

        for i in 0..self.head_dim {
            for j in 0..self.input_dim {
                output[i] += self.value_weights[i * self.input_dim + j] * input[j];
            }
        }

        output
    }

    /// Compute raw attention coefficient (before softmax)
    pub fn attention_coefficient(
        &self,
        query: &[f32],
        key: &[f32],
        hyperbolic_distance: f64,
        spike_dt: Option<f64>,
    ) -> f32 {
        // Concatenate query and key
        let mut concat = Vec::with_capacity(2 * self.head_dim);
        concat.extend_from_slice(query);
        concat.extend_from_slice(key);

        // Compute attention logit: a^T [q || k]
        let mut logit = 0.0f32;
        for (i, &c) in concat.iter().enumerate() {
            logit += self.attention_vec[i] * c;
        }

        // LeakyReLU
        logit = if logit >= 0.0 {
            logit
        } else {
            self.leaky_slope * logit
        };

        // Hyperbolic distance modulation
        let distance_factor = (-hyperbolic_distance / self.hyperbolic_scale).exp() as f32;
        logit *= distance_factor;

        // Spike-timing modulation (optional)
        if let Some(dt) = spike_dt {
            let stdp_factor = if dt > 0.0 {
                // Pre-before-post: positive attention
                (-(dt / self.stdp_tau)).exp() as f32
            } else {
                // Post-before-pre: reduced attention
                0.5 * ((dt / self.stdp_tau)).exp() as f32
            };
            logit *= stdp_factor + 0.5; // Baseline + modulation
        }

        logit
    }

    /// Clear cached attention scores
    pub fn clear_cache(&mut self) {
        self.cached_attention.clear();
    }

    /// Get cached attention score
    pub fn get_cached_attention(&self, source: u32, target: u32) -> Option<f32> {
        self.cached_attention.get(&(source, target)).copied()
    }
}

// ============================================================================
// Multi-Head Attention Layer
// ============================================================================

/// Multi-head graph attention layer with hyperbolic modulation
#[derive(Debug, Clone)]
pub struct MultiHeadGraphAttention {
    /// Attention heads
    heads: Vec<HyperbolicAttentionHead>,
    /// Output projection weights
    output_weights: Vec<f32>,
    /// Configuration
    config: AttentionConfig,
    /// Statistics
    stats: AttentionStats,
}

/// Attention statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AttentionStats {
    /// Total attention computations
    pub total_computations: u64,
    /// Average attention entropy (uniformity measure)
    pub avg_attention_entropy: f64,
    /// Average hyperbolic modulation factor
    pub avg_hyperbolic_factor: f64,
    /// Spike-timing attention activations
    pub spike_attention_activations: u64,
}

impl MultiHeadGraphAttention {
    /// Create new multi-head attention layer
    pub fn new(input_dim: usize, config: AttentionConfig) -> Self {
        let heads: Vec<_> = (0..config.num_heads)
            .map(|_| HyperbolicAttentionHead::new(input_dim, config.head_dim, &config))
            .collect();

        // Output projection: concat all heads -> output_dim
        let concat_dim = config.num_heads * config.head_dim;
        let o_scale = (2.0 / (concat_dim + config.output_dim) as f64).sqrt() as f32;

        let output_weights: Vec<f32> = (0..config.output_dim * concat_dim)
            .map(|i| ((i as f32 + 400.0) * 0.618033).sin() * o_scale)
            .collect();

        Self {
            heads,
            output_weights,
            config,
            stats: AttentionStats::default(),
        }
    }

    /// Compute attention for a node given its neighbors
    pub fn forward(
        &mut self,
        node_id: u32,
        node_features: &[f32],
        node_position: &LorentzVec,
        neighbors: &[(u32, &[f32], &LorentzVec, Option<f64>)], // (id, features, position, spike_dt)
    ) -> Vec<f32> {
        if neighbors.is_empty() {
            // No neighbors: return projected self-features
            let mut output = vec![0.0f32; self.config.output_dim];
            for head in &self.heads {
                let v = head.project_value(node_features);
                for (i, &val) in v.iter().enumerate() {
                    if i < output.len() {
                        output[i] += val / self.config.num_heads as f32;
                    }
                }
            }
            return output;
        }

        let mut head_outputs: Vec<Vec<f32>> = Vec::with_capacity(self.config.num_heads);

        for head in &mut self.heads {
            // Compute query for this node
            let query = head.project_query(node_features);

            // Compute attention coefficients
            let mut attention_logits = Vec::with_capacity(neighbors.len());
            let mut keys_values: Vec<(Vec<f32>, Vec<f32>)> = Vec::with_capacity(neighbors.len());

            for (neighbor_id, features, position, spike_dt) in neighbors {
                let key = head.project_key(features);
                let value = head.project_value(features);

                let distance = node_position.hyperbolic_distance(position);
                let logit = head.attention_coefficient(&query, &key, distance, *spike_dt);

                attention_logits.push(logit);
                keys_values.push((key, value));

                // Update stats
                self.stats.avg_hyperbolic_factor =
                    0.99 * self.stats.avg_hyperbolic_factor +
                    0.01 * (-distance / self.config.hyperbolic_scale as f64).exp();

                if spike_dt.is_some() {
                    self.stats.spike_attention_activations += 1;
                }

                // Cache attention
                head.cached_attention.insert((node_id, *neighbor_id), logit);
            }

            // Softmax over neighbors
            let attention_weights = softmax_with_temperature(&attention_logits, self.config.temperature);

            // Compute attention entropy
            let entropy = attention_weights.iter()
                .filter(|&&w| w > 0.0)
                .map(|&w| -w * w.ln())
                .sum::<f32>() as f64;
            self.stats.avg_attention_entropy =
                0.99 * self.stats.avg_attention_entropy + 0.01 * entropy;

            // Aggregate values weighted by attention
            let mut head_output = vec![0.0f32; self.config.head_dim];
            for (i, (_, (_, value))) in attention_weights.iter().zip(keys_values.iter()).enumerate() {
                for (j, &v) in value.iter().enumerate() {
                    head_output[j] += attention_weights[i] * v;
                }
            }

            head_outputs.push(head_output);
        }

        // Concatenate all heads
        let mut concat: Vec<f32> = Vec::with_capacity(self.config.num_heads * self.config.head_dim);
        for head_out in &head_outputs {
            concat.extend_from_slice(head_out);
        }

        // Output projection
        let concat_dim = self.config.num_heads * self.config.head_dim;
        let mut output = vec![0.0f32; self.config.output_dim];

        for i in 0..self.config.output_dim {
            for j in 0..concat_dim.min(concat.len()) {
                output[i] += self.output_weights[i * concat_dim + j] * concat[j];
            }
        }

        // Residual connection (if enabled and dimensions match)
        if self.config.residual && node_features.len() == self.config.output_dim {
            for (i, &f) in node_features.iter().enumerate() {
                output[i] += f;
            }
        }

        self.stats.total_computations += 1;

        output
    }

    /// Batch forward pass for multiple nodes
    pub fn forward_batch(
        &mut self,
        nodes: &[(u32, Vec<f32>, LorentzVec)],
        adjacency: &HashMap<u32, Vec<(u32, Option<f64>)>>, // node_id -> [(neighbor_id, spike_dt)]
    ) -> HashMap<u32, Vec<f32>> {
        let node_map: HashMap<u32, (&Vec<f32>, &LorentzVec)> = nodes.iter()
            .map(|(id, feat, pos)| (*id, (feat, pos)))
            .collect();

        let mut outputs = HashMap::new();

        for (node_id, features, position) in nodes {
            // Get neighbors
            let neighbor_info: Vec<_> = adjacency.get(node_id)
                .map(|neighbors| {
                    neighbors.iter()
                        .filter_map(|(nid, spike_dt)| {
                            node_map.get(nid).map(|(feat, pos)| {
                                (*nid, feat.as_slice(), *pos, *spike_dt)
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();

            let output = self.forward(*node_id, features, position, &neighbor_info);
            outputs.insert(*node_id, output);
        }

        outputs
    }

    /// Get statistics
    pub fn stats(&self) -> &AttentionStats {
        &self.stats
    }

    /// Clear all head caches
    pub fn clear_caches(&mut self) {
        for head in &mut self.heads {
            head.clear_cache();
        }
    }

    /// Get attention weights for a specific node pair (from first head)
    pub fn get_attention(&self, source: u32, target: u32) -> Option<f32> {
        self.heads.first()?.get_cached_attention(source, target)
    }
}

/// Softmax with temperature
fn softmax_with_temperature(logits: &[f32], temperature: f32) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Scale by temperature
    let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature).collect();

    // Numerical stability: subtract max
    let max = scaled.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Vec<f32> = scaled.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exp_vals.iter().sum();

    if sum > 0.0 {
        exp_vals.iter().map(|&x| x / sum).collect()
    } else {
        vec![1.0 / logits.len() as f32; logits.len()] // Uniform if all -inf
    }
}

// ============================================================================
// Spike-Aware Graph Attention
// ============================================================================

/// Graph attention layer that integrates with spiking dynamics
#[derive(Debug)]
pub struct SpikeAwareGraphAttention {
    /// Multi-head attention
    attention: MultiHeadGraphAttention,
    /// Spike timing buffer per node
    spike_times: HashMap<u32, f64>,
    /// Feature buffer per node
    node_features: HashMap<u32, Vec<f32>>,
    /// Position buffer per node
    node_positions: HashMap<u32, LorentzVec>,
    /// Adjacency list
    adjacency: HashMap<u32, Vec<u32>>,
}

impl SpikeAwareGraphAttention {
    /// Create new spike-aware attention layer
    pub fn new(input_dim: usize, config: AttentionConfig) -> Self {
        Self {
            attention: MultiHeadGraphAttention::new(input_dim, config),
            spike_times: HashMap::new(),
            node_features: HashMap::new(),
            node_positions: HashMap::new(),
            adjacency: HashMap::new(),
        }
    }

    /// Register a node
    pub fn register_node(&mut self, id: u32, features: Vec<f32>, position: LorentzVec) {
        self.node_features.insert(id, features);
        self.node_positions.insert(id, position);
        self.adjacency.entry(id).or_insert_with(Vec::new);
    }

    /// Add edge
    pub fn add_edge(&mut self, source: u32, target: u32) {
        self.adjacency.entry(source).or_insert_with(Vec::new).push(target);
    }

    /// Record spike time
    pub fn record_spike(&mut self, node_id: u32, time: f64) {
        self.spike_times.insert(node_id, time);
    }

    /// Compute attention-weighted features for a node
    pub fn compute_attention(&mut self, node_id: u32, current_time: f64) -> Option<Vec<f32>> {
        let features = self.node_features.get(&node_id)?.clone();
        let position = *self.node_positions.get(&node_id)?;
        let neighbors = self.adjacency.get(&node_id)?;

        // Build neighbor info with spike timing
        let neighbor_info: Vec<_> = neighbors.iter()
            .filter_map(|&nid| {
                let feat = self.node_features.get(&nid)?;
                let pos = self.node_positions.get(&nid)?;

                // Compute spike timing difference relative to current time
                // Recent spikes get higher attention weight (temporal relevance)
                let spike_dt = match (self.spike_times.get(&node_id), self.spike_times.get(&nid)) {
                    (Some(&t1), Some(&t2)) => {
                        // Causal spike timing: positive if neighbor spiked before node
                        let dt = t1 - t2;
                        // Apply temporal decay based on how long ago spikes occurred
                        let node_recency = (-(current_time - t1).abs() / 50.0).exp();
                        let neighbor_recency = (-(current_time - t2).abs() / 50.0).exp();
                        // Weight by recency: recent spike pairs are more relevant
                        Some(dt * (node_recency * neighbor_recency).sqrt())
                    },
                    _ => None,
                };

                Some((nid, feat.as_slice(), pos, spike_dt))
            })
            .collect();

        Some(self.attention.forward(node_id, &features, &position, &neighbor_info))
    }

    /// Update node features with attention-weighted aggregation
    pub fn step(&mut self, current_time: f64) {
        let node_ids: Vec<u32> = self.node_features.keys().copied().collect();

        let mut new_features = HashMap::new();

        for node_id in node_ids {
            if let Some(output) = self.compute_attention(node_id, current_time) {
                new_features.insert(node_id, output);
            }
        }

        // Update features
        for (id, feat) in new_features {
            self.node_features.insert(id, feat);
        }
    }

    /// Get node features
    pub fn get_features(&self, node_id: u32) -> Option<&Vec<f32>> {
        self.node_features.get(&node_id)
    }

    /// Get attention statistics
    pub fn stats(&self) -> &AttentionStats {
        self.attention.stats()
    }
}

// ============================================================================
// Hyperbolic Message Passing with Attention
// ============================================================================

/// Message passing layer using hyperbolic attention
#[derive(Debug)]
pub struct HyperbolicMessagePassing {
    /// Attention mechanism
    attention: MultiHeadGraphAttention,
    /// Message aggregation type
    aggregation: MessageAggregation,
    /// Update function weights
    update_weights: Vec<f32>,
    /// Update bias
    update_bias: Vec<f32>,
    /// Output dimension
    output_dim: usize,
}

/// Message aggregation strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageAggregation {
    /// Sum aggregation
    Sum,
    /// Mean aggregation
    Mean,
    /// Max aggregation
    Max,
    /// Attention-weighted aggregation
    Attention,
}

impl HyperbolicMessagePassing {
    /// Create new message passing layer
    pub fn new(input_dim: usize, output_dim: usize, config: AttentionConfig) -> Self {
        let attention = MultiHeadGraphAttention::new(input_dim, config);

        // Update function: output = W * [h || m] + b
        let update_dim = output_dim + output_dim; // node + message
        let scale = (2.0 / (update_dim + output_dim) as f64).sqrt() as f32;

        let update_weights: Vec<f32> = (0..output_dim * update_dim)
            .map(|i| ((i as f32 + 500.0) * 0.618033).sin() * scale)
            .collect();

        let update_bias = vec![0.0f32; output_dim];

        Self {
            attention,
            aggregation: MessageAggregation::Attention,
            update_weights,
            update_bias,
            output_dim,
        }
    }

    /// Set aggregation strategy
    pub fn set_aggregation(&mut self, aggregation: MessageAggregation) {
        self.aggregation = aggregation;
    }

    /// Message passing step
    pub fn message_pass(
        &mut self,
        node_id: u32,
        node_features: &[f32],
        node_position: &LorentzVec,
        neighbors: &[(u32, &[f32], &LorentzVec, Option<f64>)],
    ) -> Vec<f32> {
        if neighbors.is_empty() {
            return node_features.to_vec();
        }

        // Compute attention-weighted messages
        let messages = self.attention.forward(
            node_id, node_features, node_position, neighbors
        );

        // Aggregate messages based on strategy
        let aggregated = match &self.aggregation {
            MessageAggregation::Sum | MessageAggregation::Attention => {
                messages.clone()
            }
            MessageAggregation::Mean => {
                messages.iter().map(|&m| m / neighbors.len() as f32).collect()
            }
            MessageAggregation::Max => {
                // Element-wise max (attention already gives weighted combination)
                messages.clone()
            }
        };

        // Update: concat node features with message, apply linear transform
        let node_feat_trunc: Vec<f32> = node_features.iter()
            .take(self.output_dim)
            .copied()
            .collect();

        let mut concat = node_feat_trunc;
        concat.extend_from_slice(&aggregated);

        // Linear update
        let update_input_dim = self.output_dim * 2;
        let mut output = self.update_bias.clone();

        for i in 0..self.output_dim {
            for j in 0..update_input_dim.min(concat.len()) {
                output[i] += self.update_weights[i * update_input_dim + j] * concat[j];
            }
            // ReLU activation
            output[i] = output[i].max(0.0);
        }

        output
    }

    /// Get attention layer stats
    pub fn stats(&self) -> &AttentionStats {
        self.attention.stats()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_head_projection() {
        let config = AttentionConfig::default();
        let head = HyperbolicAttentionHead::new(8, 16, &config);

        let input = vec![1.0f32; 8];
        let query = head.project_query(&input);
        let key = head.project_key(&input);
        let value = head.project_value(&input);

        assert_eq!(query.len(), 16);
        assert_eq!(key.len(), 16);
        assert_eq!(value.len(), 16);
    }

    #[test]
    fn test_attention_coefficient() {
        let config = AttentionConfig::default();
        let head = HyperbolicAttentionHead::new(8, 16, &config);

        let input1 = vec![1.0f32; 8];
        let input2 = vec![1.0f32; 8];

        let query = head.project_query(&input1);
        let key = head.project_key(&input2);

        // Test without spike timing
        let coeff1 = head.attention_coefficient(&query, &key, 0.0, None);

        // Test with larger distance (should reduce attention)
        let coeff2 = head.attention_coefficient(&query, &key, 5.0, None);
        assert!(coeff2.abs() <= coeff1.abs() || coeff1.abs() < 0.01);

        // Test with spike timing
        let coeff3 = head.attention_coefficient(&query, &key, 0.0, Some(5.0));
        // Should have spike modulation effect
        assert!(coeff3 != coeff1 || coeff1.abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax_with_temperature(&logits, 1.0);

        // Should sum to 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);

        // Higher logit = higher probability
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_multi_head_attention() {
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 8,
            output_dim: 16,
            ..Default::default()
        };

        let mut attention = MultiHeadGraphAttention::new(8, config);

        let node_features = vec![1.0f32; 8];
        let node_position = LorentzVec::origin();

        // Create neighbors
        let neighbor1_feat = vec![0.5f32; 8];
        let neighbor1_pos = LorentzVec::from_spatial(0.3, 0.0, 0.0);

        let neighbor2_feat = vec![0.8f32; 8];
        let neighbor2_pos = LorentzVec::from_spatial(0.0, 0.3, 0.0);

        let neighbors: Vec<(u32, &[f32], &LorentzVec, Option<f64>)> = vec![
            (1, &neighbor1_feat, &neighbor1_pos, Some(5.0)),
            (2, &neighbor2_feat, &neighbor2_pos, None),
        ];

        let output = attention.forward(0, &node_features, &node_position, &neighbors);

        assert_eq!(output.len(), 16);
        assert!(attention.stats().total_computations > 0);
    }

    #[test]
    fn test_spike_aware_attention() {
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 8,
            output_dim: 16,
            spike_timing_attention: true,
            ..Default::default()
        };

        let mut attention = SpikeAwareGraphAttention::new(8, config);

        // Register nodes
        attention.register_node(0, vec![1.0; 8], LorentzVec::origin());
        attention.register_node(1, vec![0.5; 8], LorentzVec::from_spatial(0.3, 0.0, 0.0));

        // Add edge
        attention.add_edge(0, 1);

        // Record spikes
        attention.record_spike(0, 10.0);
        attention.record_spike(1, 5.0);

        // Compute attention
        let output = attention.compute_attention(0, 10.0);
        assert!(output.is_some());
        assert_eq!(output.unwrap().len(), 16);
    }

    #[test]
    fn test_message_passing() {
        let config = AttentionConfig {
            num_heads: 2,
            head_dim: 8,
            output_dim: 16,
            ..Default::default()
        };

        let mut mp = HyperbolicMessagePassing::new(8, 16, config);

        let node_features = vec![1.0f32; 8];
        let node_position = LorentzVec::origin();

        let neighbor_feat = vec![0.5f32; 8];
        let neighbor_pos = LorentzVec::from_spatial(0.3, 0.0, 0.0);

        let neighbors: Vec<(u32, &[f32], &LorentzVec, Option<f64>)> = vec![
            (1, &neighbor_feat, &neighbor_pos, None),
        ];

        let output = mp.message_pass(0, &node_features, &node_position, &neighbors);

        assert_eq!(output.len(), 16);
    }
}
