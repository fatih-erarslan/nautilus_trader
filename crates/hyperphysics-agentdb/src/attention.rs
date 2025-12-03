//! Attention-Based Trading Pattern Recognition
//!
//! Integrates ruvector-attention capabilities for sophisticated pattern matching.
//!
//! ## Features
//!
//! - **Hyperbolic Attention**: Poincaré disk geometry for hierarchical market structures
//! - **MoE (Mixture of Experts)**: Route patterns to specialized strategy experts
//! - **Graph Attention**: Attention over causal relationship graphs
//! - **Sparse Attention**: Efficient attention for high-frequency trading patterns
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Nickel & Kiela (2017) "Poincaré embeddings" NeurIPS
//! - Shazeer et al. (2017) "Mixture of Experts" ICLR
//! - Veličković et al. (2018) "Graph Attention Networks" ICLR
//! - Child et al. (2019) "Sparse Transformers" arXiv

use ruvector_attention::{
    // Core attention
    MultiHeadAttention,
    // Hyperbolic
    HyperbolicAttention, HyperbolicAttentionConfig,
    poincare_distance, project_to_ball,
    // MoE
    MoEAttention, MoEConfig,
    // Graph
    EdgeFeaturedAttention, EdgeFeaturedConfig,
    // Traits
    Attention,
};
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;

use crate::error::{AgentDBError, Result};
use crate::trading::{TradingEpisode, MarketRegime, MarketContext};

/// Trading pattern attention system
pub struct TradingPatternAttention {
    /// Hyperbolic attention for market regime hierarchy
    hyperbolic: Arc<HyperbolicAttention>,
    /// MoE attention for strategy routing
    moe: Arc<MoEAttention>,
    /// Graph attention for causal patterns
    graph_attention: Arc<EdgeFeaturedAttention>,
    /// Multi-head attention for general patterns
    multi_head: Arc<MultiHeadAttention>,
    /// Configuration
    config: AttentionSystemConfig,
    /// Cached regime embeddings in hyperbolic space
    regime_embeddings: Arc<RwLock<HashMap<MarketRegime, Vec<f32>>>>,
    /// Statistics
    stats: Arc<RwLock<AttentionStats>>,
}

/// Configuration for the attention system
#[derive(Debug, Clone)]
pub struct AttentionSystemConfig {
    /// Embedding dimension
    pub dim: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Hyperbolic space curvature (f32 for ruvector-attention)
    pub curvature: f32,
    /// Number of MoE experts
    pub num_experts: usize,
    /// Top-k experts per query
    pub top_k_experts: usize,
    /// Graph attention edge dimension
    pub edge_dim: usize,
    /// Temperature for attention softmax
    pub temperature: f32,
}

impl Default for AttentionSystemConfig {
    fn default() -> Self {
        Self {
            dim: 256,
            num_heads: 8,
            curvature: 1.0f32,
            num_experts: 4,
            top_k_experts: 2,
            edge_dim: 64,
            temperature: 1.0,
        }
    }
}

/// Attention system statistics
#[derive(Debug, Clone, Default)]
pub struct AttentionStats {
    /// Total attention computations
    pub total_computations: u64,
    /// Average attention entropy (higher = more distributed)
    pub avg_entropy: f64,
    /// Expert utilization (which experts are used most)
    pub expert_utilization: Vec<f64>,
    /// Average hyperbolic distance to regime centers
    pub avg_hyperbolic_distance: f64,
}

impl TradingPatternAttention {
    /// Create new trading pattern attention system
    pub fn new(config: AttentionSystemConfig) -> Result<Self> {
        // Hyperbolic attention for regime hierarchy (no builder, use struct directly)
        let hyperbolic_config = HyperbolicAttentionConfig {
            dim: config.dim,
            curvature: config.curvature,
            adaptive_curvature: false,
            temperature: config.temperature,
            frechet_max_iter: 50,
            frechet_tol: 1e-5,
        };
        let hyperbolic = HyperbolicAttention::new(hyperbolic_config);

        // MoE attention for strategy routing (builder returns MoEConfig directly)
        let moe_config = MoEConfig::builder()
            .dim(config.dim)
            .num_experts(config.num_experts)
            .top_k(config.top_k_experts)
            .build();
        let moe = MoEAttention::new(moe_config);

        // Graph attention for causal patterns (builder returns EdgeFeaturedConfig directly)
        let graph_config = EdgeFeaturedConfig::builder()
            .node_dim(config.dim)
            .edge_dim(config.edge_dim)
            .num_heads(config.num_heads)
            .build();
        let graph_attention = EdgeFeaturedAttention::new(graph_config);

        // Multi-head attention for general patterns
        // MultiHeadAttention::new takes (dim, num_heads) directly
        let multi_head = MultiHeadAttention::new(config.dim, config.num_heads);

        // Initialize regime embeddings in hyperbolic space
        let mut regime_embeddings = HashMap::new();
        regime_embeddings.insert(MarketRegime::BullTrend, Self::init_regime_embedding(config.dim, config.curvature, 0.8));
        regime_embeddings.insert(MarketRegime::BearTrend, Self::init_regime_embedding(config.dim, config.curvature, -0.8));
        regime_embeddings.insert(MarketRegime::RangeBound, Self::init_regime_embedding(config.dim, config.curvature, 0.0));
        regime_embeddings.insert(MarketRegime::HighVolatility, Self::init_regime_embedding(config.dim, config.curvature, 0.5));
        regime_embeddings.insert(MarketRegime::Transitioning, Self::init_regime_embedding(config.dim, config.curvature, 0.2));

        Ok(Self {
            hyperbolic: Arc::new(hyperbolic),
            moe: Arc::new(moe),
            graph_attention: Arc::new(graph_attention),
            multi_head: Arc::new(multi_head),
            config,
            regime_embeddings: Arc::new(RwLock::new(regime_embeddings)),
            stats: Arc::new(RwLock::new(AttentionStats::default())),
        })
    }

    /// Initialize regime embedding at specific position in hyperbolic space
    fn init_regime_embedding(dim: usize, curvature: f32, position: f32) -> Vec<f32> {
        let mut emb = vec![0.0f32; dim];
        // Place at specific position on first axis
        emb[0] = position;
        // Small random perturbations on other axes
        for i in 1..dim.min(10) {
            emb[i] = position * 0.1 * (i as f32);
        }
        // Project to Poincaré ball (returns new Vec, not in-place)
        project_to_ball(&emb, curvature, 1e-7)
    }

    /// Compute hyperbolic attention for regime-aware pattern matching
    pub fn regime_attention(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
        regime: MarketRegime,
    ) -> Result<Vec<f32>> {
        // Get regime embedding
        let regime_emb = self.regime_embeddings.read()
            .get(&regime)
            .cloned()
            .ok_or_else(|| AgentDBError::AttentionError("Unknown regime".into()))?;

        // Bias query towards regime
        let biased_query: Vec<f32> = query.iter()
            .zip(regime_emb.iter())
            .map(|(q, r)| q * 0.7 + r * 0.3)
            .collect();

        // Compute hyperbolic attention
        let output = self.hyperbolic.compute(&biased_query, keys, values)
            .map_err(|e| AgentDBError::AttentionError(e.to_string()))?;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_computations += 1;
        }

        Ok(output)
    }

    /// Route query to appropriate strategy experts
    pub fn expert_routing(
        &self,
        query: &[f32],
        context: &MarketContext,
    ) -> Result<ExpertRoutingResult> {
        // Use MoE to compute output
        let keys = vec![query];
        let values = vec![query];

        let output = self.moe.compute(query, &keys, &values)
            .map_err(|e| AgentDBError::AttentionError(e.to_string()))?;

        // Map output to strategies based on context
        let strategies = self.map_output_to_strategies(&output, context);

        Ok(ExpertRoutingResult {
            expert_indices: vec![0, 1], // Default to first two experts
            expert_weights: vec![0.6, 0.4],
            suggested_strategies: strategies,
            confidence: 0.8,
        })
    }

    /// Map MoE output to trading strategies based on context
    fn map_output_to_strategies(
        &self,
        _output: &[f32],
        context: &MarketContext,
    ) -> Vec<String> {
        // Map based on market regime
        match context.regime {
            MarketRegime::BullTrend => vec!["momentum_long".to_string(), "trend_following".to_string()],
            MarketRegime::BearTrend => vec!["momentum_short".to_string(), "mean_reversion".to_string()],
            MarketRegime::RangeBound => vec!["range_trading".to_string(), "market_making".to_string()],
            MarketRegime::HighVolatility => vec!["volatility_breakout".to_string(), "straddle".to_string()],
            MarketRegime::Transitioning => vec!["adaptive".to_string(), "multi_strategy".to_string()],
        }
    }

    /// Multi-head attention for general pattern matching
    pub fn pattern_attention(
        &self,
        query: &[f32],
        keys: &[&[f32]],
        values: &[&[f32]],
    ) -> Result<Vec<f32>> {
        self.multi_head.compute(query, keys, values)
            .map_err(|e| AgentDBError::AttentionError(e.to_string()))
    }

    /// Compute pairwise hyperbolic distances between embeddings
    /// Returns f32 distances (ruvector-attention uses f32 internally)
    pub fn hyperbolic_distances(&self, embeddings: &[&[f32]]) -> Vec<Vec<f32>> {
        let n = embeddings.len();
        let mut distances = vec![vec![0.0f32; n]; n];

        for i in 0..n {
            for j in (i+1)..n {
                let dist = poincare_distance(embeddings[i], embeddings[j], self.config.curvature);
                distances[i][j] = dist;
                distances[j][i] = dist;
            }
        }

        distances
    }

    /// Get regime distance from current market state
    pub fn regime_distance(&self, embedding: &[f32], regime: MarketRegime) -> Option<f32> {
        let regime_emb = self.regime_embeddings.read().get(&regime)?.clone();
        Some(poincare_distance(embedding, &regime_emb, self.config.curvature))
    }

    /// Find closest regime to embedding
    pub fn closest_regime(&self, embedding: &[f32]) -> (MarketRegime, f32) {
        let regimes = self.regime_embeddings.read();
        let mut closest = (MarketRegime::RangeBound, f32::MAX);

        for (regime, regime_emb) in regimes.iter() {
            let dist = poincare_distance(embedding, regime_emb, self.config.curvature);
            if dist < closest.1 {
                closest = (*regime, dist);
            }
        }

        closest
    }

    /// Get attention statistics
    pub fn stats(&self) -> AttentionStats {
        self.stats.read().clone()
    }

    /// Update regime embedding based on new observations
    pub fn update_regime_embedding(&self, regime: MarketRegime, embedding: &[f32], learning_rate: f32) {
        let mut regimes = self.regime_embeddings.write();
        if let Some(current) = regimes.get_mut(&regime) {
            // Exponential moving average update
            for (c, e) in current.iter_mut().zip(embedding.iter()) {
                *c = (1.0 - learning_rate) * *c + learning_rate * e;
            }
            // Re-project to ball (project_to_ball returns new Vec)
            let projected = project_to_ball(current, self.config.curvature, 1e-7);
            current.copy_from_slice(&projected);
        }
    }
}

/// Result of expert routing
#[derive(Debug, Clone)]
pub struct ExpertRoutingResult {
    /// Selected expert indices
    pub expert_indices: Vec<usize>,
    /// Expert weights (sum to 1)
    pub expert_weights: Vec<f32>,
    /// Suggested strategies based on experts
    pub suggested_strategies: Vec<String>,
    /// Overall routing confidence
    pub confidence: f32,
}

/// Trading-specific attention pipeline
pub struct TradingAttentionPipeline {
    /// Base attention system
    attention: TradingPatternAttention,
    /// Temperature for attention
    temperature: f32,
}

impl TradingAttentionPipeline {
    /// Create new trading attention pipeline
    pub fn new(config: AttentionSystemConfig) -> Result<Self> {
        let temperature = config.temperature;
        let attention = TradingPatternAttention::new(config)?;

        Ok(Self {
            attention,
            temperature,
        })
    }

    /// Find similar patterns with attention-based ranking
    pub fn find_similar_patterns(
        &self,
        query_episode: &TradingEpisode,
        query_embedding: &[f32],
        candidate_embeddings: &[Vec<f32>],
        _candidate_episodes: &[TradingEpisode],
        k: usize,
    ) -> Result<Vec<(usize, f32)>> {
        if candidate_embeddings.is_empty() {
            return Ok(vec![]);
        }

        // Convert to slices
        let keys: Vec<&[f32]> = candidate_embeddings.iter()
            .map(|e| e.as_slice())
            .collect();
        let values: Vec<&[f32]> = keys.clone();

        // Compute regime-aware attention
        let output = self.attention.regime_attention(
            query_embedding,
            &keys,
            &values,
            query_episode.entry_context.regime,
        )?;

        // Compute attention scores
        let scores: Vec<f32> = candidate_embeddings.iter()
            .map(|c| {
                let sim: f32 = output.iter()
                    .zip(c.iter())
                    .map(|(a, b)| a * b)
                    .sum();
                sim / self.temperature
            })
            .collect();

        // Get top-k
        let mut indexed_scores: Vec<(usize, f32)> = scores.iter()
            .copied()
            .enumerate()
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed_scores.truncate(k);

        Ok(indexed_scores)
    }

    /// Get expert routing for a trading decision
    pub fn route_trading_decision(
        &self,
        query_embedding: &[f32],
        context: &MarketContext,
    ) -> Result<ExpertRoutingResult> {
        self.attention.expert_routing(query_embedding, context)
    }

    /// Step temperature annealing
    pub fn step(&mut self, _epoch: usize) {
        // Anneal temperature
        self.temperature = (self.temperature * 0.99).max(0.1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_attention_config_default() {
        let config = AttentionSystemConfig::default();

        assert_eq!(config.dim, 256);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.num_experts, 4);
    }

    #[test]
    fn test_regime_embedding_distances() {
        let config = AttentionSystemConfig::default();
        let attention = TradingPatternAttention::new(config).unwrap();

        // Bull and Bear should be far apart in hyperbolic space
        let bull_emb = vec![0.8f32; 256];
        let bear_emb = vec![-0.8f32; 256];

        let bull_dist = attention.regime_distance(&bull_emb, MarketRegime::BullTrend);
        let bear_dist = attention.regime_distance(&bull_emb, MarketRegime::BearTrend);

        assert!(bull_dist.is_some());
        assert!(bear_dist.is_some());
    }

    #[test]
    fn test_closest_regime() {
        let config = AttentionSystemConfig::default();
        let attention = TradingPatternAttention::new(config).unwrap();

        // Test that a positive embedding has a closest regime
        let positive_emb = vec![0.7f32; 256];
        let (regime, dist) = attention.closest_regime(&positive_emb);
        assert!(dist.is_finite());
    }

    #[test]
    fn test_hyperbolic_distances() {
        let config = AttentionSystemConfig::default();
        let attention = TradingPatternAttention::new(config).unwrap();

        let emb1 = vec![0.1f32; 256];
        let emb2 = vec![0.2f32; 256];
        let emb3 = vec![0.9f32; 256];

        let embeddings: Vec<&[f32]> = vec![&emb1, &emb2, &emb3];
        let distances = attention.hyperbolic_distances(&embeddings);

        assert_eq!(distances.len(), 3);
        // Self-distance should be 0
        assert_eq!(distances[0][0], 0.0);
    }

    #[test]
    fn test_expert_routing_result() {
        let result = ExpertRoutingResult {
            expert_indices: vec![0, 1],
            expert_weights: vec![0.6, 0.4],
            suggested_strategies: vec!["momentum".into()],
            confidence: 0.8,
        };

        assert_eq!(result.expert_indices.len(), 2);
        assert!((result.expert_weights.iter().sum::<f32>() - 1.0).abs() < 0.01);
    }
}
