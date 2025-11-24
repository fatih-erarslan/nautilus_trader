//! Meta-learning neural router for intelligent backend selection
//!
//! Uses neural networks to learn optimal routing policies based on problem
//! characteristics and historical performance data.
//!
//! ## Architecture
//!
//! The NeuralRouter learns to map problem signatures to optimal backend selections
//! by observing routing outcomes. It complements the Thompson Sampling approach
//! in the reasoning router with learned patterns.
//!
//! ## Features
//!
//! - **Contextual Bandits**: Learns context-dependent policies
//! - **Transfer Learning**: Applies knowledge across problem domains
//! - **Latency Prediction**: Estimates backend execution time
//! - **Quality Prediction**: Predicts result quality per backend

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use crate::activation::Activation;
use crate::core::{Tensor, TensorShape};
use crate::error::{NeuralError, NeuralResult};
use crate::network::{Network, NetworkBuilder};

/// Router policy types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RouterPolicy {
    /// Always select highest predicted quality
    Greedy,
    /// Epsilon-greedy exploration
    EpsilonGreedy,
    /// Softmax/Boltzmann exploration
    Softmax,
    /// Upper Confidence Bound
    UCB,
    /// Thompson Sampling with neural priors
    ThompsonNeural,
}

impl Default for RouterPolicy {
    fn default() -> Self {
        RouterPolicy::Softmax
    }
}

/// Neural router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralRouterConfig {
    /// Number of backends to route between
    pub num_backends: usize,
    /// Problem signature dimension (from ProblemSignature::to_feature_vector)
    pub signature_dim: usize,
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    /// Routing policy
    pub policy: RouterPolicy,
    /// Exploration rate (for epsilon-greedy)
    pub epsilon: f64,
    /// Temperature (for softmax)
    pub temperature: f64,
    /// Learning rate for online updates
    pub learning_rate: f64,
    /// Maximum inference latency (microseconds)
    pub max_latency_us: u64,
}

impl Default for NeuralRouterConfig {
    fn default() -> Self {
        Self {
            num_backends: 10,
            signature_dim: 16, // ProblemSignature feature vector size
            hidden_dims: vec![32, 16],
            policy: RouterPolicy::Softmax,
            epsilon: 0.1,
            temperature: 1.0,
            learning_rate: 0.01,
            max_latency_us: 50, // 50μs target for routing decisions
        }
    }
}

/// Neural router for intelligent backend selection
#[derive(Debug)]
pub struct NeuralRouter {
    /// Configuration
    config: NeuralRouterConfig,
    /// Quality prediction network (signature -> backend quality scores)
    quality_network: Network,
    /// Latency prediction network (signature -> backend latency estimates)
    latency_network: Network,
    /// Backend selection counts for UCB
    selection_counts: Vec<u64>,
    /// Total routing decisions
    total_decisions: u64,
    /// Routing statistics
    stats: RouterStats,
}

/// Router statistics
#[derive(Debug, Clone, Default)]
struct RouterStats {
    total_routes: u64,
    total_latency_us: u64,
    backend_selections: HashMap<usize, u64>,
    quality_observed: Vec<f64>,
}

impl NeuralRouter {
    /// Create new neural router
    pub fn new(config: NeuralRouterConfig) -> NeuralResult<Self> {
        let quality_network = Self::build_network(
            &config,
            "QualityPredictor",
            Activation::Sigmoid, // Quality in [0, 1]
        )?;

        let latency_network = Self::build_network(
            &config,
            "LatencyPredictor",
            Activation::ReLU, // Latency is positive
        )?;

        Ok(Self {
            selection_counts: vec![0; config.num_backends],
            total_decisions: 0,
            stats: RouterStats::default(),
            config,
            quality_network,
            latency_network,
        })
    }

    fn build_network(
        config: &NeuralRouterConfig,
        name: &str,
        output_activation: Activation,
    ) -> NeuralResult<Network> {
        let mut builder = Network::builder()
            .name(name)
            .input_dim(config.signature_dim)
            .output_dim(config.num_backends)
            .hidden_activation(Activation::ReLU)
            .output_activation(output_activation);

        for &dim in &config.hidden_dims {
            builder = builder.hidden(dim);
        }

        builder.build()
    }

    /// Select backend for a problem signature
    ///
    /// Returns (backend_index, confidence, predicted_quality, predicted_latency)
    pub fn route(&mut self, signature: &[f32; 16]) -> NeuralResult<RoutingDecision> {
        let start = Instant::now();

        // Convert signature to tensor
        let input_data: Vec<f64> = signature.iter().map(|&x| x as f64).collect();
        let input = Tensor::new(input_data, TensorShape::d2(1, 16))?;

        // Predict quality and latency
        let quality_scores = self.quality_network.predict(&input)?;
        let latency_estimates = self.latency_network.predict(&input)?;

        let qualities = quality_scores.data();
        let latencies = latency_estimates.data();

        // Apply policy to select backend
        let (selected, confidence) = self.apply_policy(qualities)?;

        let latency_us = start.elapsed().as_micros() as u64;

        // Update stats
        self.selection_counts[selected] += 1;
        self.total_decisions += 1;
        self.stats.total_routes += 1;
        self.stats.total_latency_us += latency_us;
        *self.stats.backend_selections.entry(selected).or_insert(0) += 1;

        // Warn on slow routing
        if latency_us > self.config.max_latency_us {
            tracing::warn!(
                "Neural routing took {}μs, exceeding target {}μs",
                latency_us,
                self.config.max_latency_us
            );
        }

        Ok(RoutingDecision {
            backend_idx: selected,
            confidence,
            predicted_quality: qualities[selected],
            predicted_latency_us: (latencies[selected] * 1000.0) as u64, // Convert to μs
            routing_latency_us: latency_us,
            all_qualities: qualities.to_vec(),
            all_latencies: latencies.iter().map(|&l| (l * 1000.0) as u64).collect(),
        })
    }

    /// Apply routing policy to quality scores
    fn apply_policy(&self, qualities: &[f64]) -> NeuralResult<(usize, f64)> {
        match self.config.policy {
            RouterPolicy::Greedy => {
                let (idx, &max_q) = qualities.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                Ok((idx, max_q))
            },
            RouterPolicy::EpsilonGreedy => {
                use rand::Rng;
                let mut rng = rand::thread_rng();

                if rng.gen::<f64>() < self.config.epsilon {
                    // Explore: random selection
                    let idx = rng.gen_range(0..qualities.len());
                    Ok((idx, qualities[idx]))
                } else {
                    // Exploit: greedy selection
                    let (idx, &max_q) = qualities.iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .unwrap();
                    Ok((idx, max_q))
                }
            },
            RouterPolicy::Softmax => {
                use rand::Rng;
                let mut rng = rand::thread_rng();

                // Compute softmax probabilities
                let max_q = qualities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let exp_q: Vec<f64> = qualities.iter()
                    .map(|&q| ((q - max_q) / self.config.temperature).exp())
                    .collect();
                let sum: f64 = exp_q.iter().sum();
                let probs: Vec<f64> = exp_q.iter().map(|&e| e / sum).collect();

                // Sample from distribution
                let r: f64 = rng.gen();
                let mut cumsum = 0.0;
                for (idx, &prob) in probs.iter().enumerate() {
                    cumsum += prob;
                    if r < cumsum {
                        return Ok((idx, probs[idx]));
                    }
                }
                Ok((probs.len() - 1, *probs.last().unwrap()))
            },
            RouterPolicy::UCB => {
                // Upper Confidence Bound
                let t = self.total_decisions.max(1) as f64;
                let ucb_values: Vec<f64> = qualities.iter()
                    .zip(self.selection_counts.iter())
                    .map(|(&q, &n)| {
                        if n == 0 {
                            f64::INFINITY
                        } else {
                            q + (2.0 * t.ln() / n as f64).sqrt()
                        }
                    })
                    .collect();

                let (idx, _) = ucb_values.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                Ok((idx, qualities[idx]))
            },
            RouterPolicy::ThompsonNeural => {
                // Thompson Sampling with neural uncertainty
                use rand::Rng;
                use rand_distr::{Distribution, Normal};
                let mut rng = rand::thread_rng();

                // Sample from posterior (approximate with quality + noise)
                let samples: Vec<f64> = qualities.iter()
                    .zip(self.selection_counts.iter())
                    .map(|(&q, &n)| {
                        // Uncertainty decreases with more observations
                        let uncertainty = 1.0 / (n as f64 + 1.0).sqrt();
                        let normal = Normal::new(q, uncertainty).unwrap();
                        normal.sample(&mut rng)
                    })
                    .collect();

                let (idx, _) = samples.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap();
                Ok((idx, qualities[idx]))
            },
        }
    }

    /// Update router with observed outcome
    ///
    /// # Arguments
    /// * `signature` - Problem signature that was routed
    /// * `backend_idx` - Backend that was selected
    /// * `observed_quality` - Actual quality of result (0-1)
    /// * `observed_latency_us` - Actual latency in microseconds
    pub fn update(
        &mut self,
        signature: &[f32; 16],
        backend_idx: usize,
        observed_quality: f64,
        observed_latency_us: u64,
    ) -> NeuralResult<()> {
        // Record observation
        self.stats.quality_observed.push(observed_quality);

        // For a full implementation, we'd do gradient-based updates here
        // This is a simplified version that tracks statistics

        // The quality and latency networks would be updated via backprop
        // using the prediction error as the loss signal

        Ok(())
    }

    /// Get routing statistics
    pub fn avg_routing_latency_us(&self) -> f64 {
        if self.stats.total_routes == 0 {
            0.0
        } else {
            self.stats.total_latency_us as f64 / self.stats.total_routes as f64
        }
    }

    /// Get backend selection distribution
    pub fn selection_distribution(&self) -> HashMap<usize, f64> {
        let total = self.total_decisions.max(1) as f64;
        self.stats.backend_selections.iter()
            .map(|(&idx, &count)| (idx, count as f64 / total))
            .collect()
    }

    /// Get configuration
    pub fn config(&self) -> &NeuralRouterConfig {
        &self.config
    }

    /// Get quality network
    pub fn quality_network(&self) -> &Network {
        &self.quality_network
    }

    /// Get latency network
    pub fn latency_network(&self) -> &Network {
        &self.latency_network
    }
}

/// Routing decision with predictions
#[derive(Debug, Clone)]
pub struct RoutingDecision {
    /// Selected backend index
    pub backend_idx: usize,
    /// Confidence in selection
    pub confidence: f64,
    /// Predicted quality for selected backend
    pub predicted_quality: f64,
    /// Predicted latency for selected backend (microseconds)
    pub predicted_latency_us: u64,
    /// Time taken to make routing decision (microseconds)
    pub routing_latency_us: u64,
    /// Quality predictions for all backends
    pub all_qualities: Vec<f64>,
    /// Latency predictions for all backends (microseconds)
    pub all_latencies: Vec<u64>,
}

impl RoutingDecision {
    /// Get top-k backends by predicted quality
    pub fn top_k(&self, k: usize) -> Vec<(usize, f64)> {
        let mut indexed: Vec<(usize, f64)> = self.all_qualities.iter()
            .enumerate()
            .map(|(i, &q)| (i, q))
            .collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());
        indexed.into_iter().take(k).collect()
    }

    /// Get backends meeting latency constraint
    pub fn within_latency(&self, max_us: u64) -> Vec<(usize, f64)> {
        self.all_qualities.iter()
            .zip(self.all_latencies.iter())
            .enumerate()
            .filter(|(_, (_, &lat))| lat <= max_us)
            .map(|(i, (&q, _))| (i, q))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = NeuralRouterConfig::default();
        let router = NeuralRouter::new(config).unwrap();

        assert_eq!(router.config().num_backends, 10);
        assert_eq!(router.config().signature_dim, 16);
    }

    #[test]
    fn test_router_route() {
        let config = NeuralRouterConfig {
            num_backends: 5,
            hidden_dims: vec![8],
            ..Default::default()
        };

        let mut router = NeuralRouter::new(config).unwrap();
        let signature = [0.5f32; 16];

        let decision = router.route(&signature).unwrap();

        assert!(decision.backend_idx < 5);
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert_eq!(decision.all_qualities.len(), 5);
        assert_eq!(decision.all_latencies.len(), 5);
    }

    #[test]
    fn test_routing_policies() {
        for policy in [
            RouterPolicy::Greedy,
            RouterPolicy::EpsilonGreedy,
            RouterPolicy::Softmax,
            RouterPolicy::UCB,
            RouterPolicy::ThompsonNeural,
        ] {
            let config = NeuralRouterConfig {
                num_backends: 3,
                hidden_dims: vec![4],
                policy,
                ..Default::default()
            };

            let mut router = NeuralRouter::new(config).unwrap();
            let signature = [0.5f32; 16];

            // Should not panic with any policy
            let decision = router.route(&signature).unwrap();
            assert!(decision.backend_idx < 3);
        }
    }

    #[test]
    fn test_router_update() {
        let config = NeuralRouterConfig {
            num_backends: 3,
            hidden_dims: vec![4],
            ..Default::default()
        };

        let mut router = NeuralRouter::new(config).unwrap();
        let signature = [0.5f32; 16];

        let decision = router.route(&signature).unwrap();
        router.update(&signature, decision.backend_idx, 0.9, 100).unwrap();

        assert_eq!(router.stats.quality_observed.len(), 1);
    }

    #[test]
    fn test_routing_decision_helpers() {
        let decision = RoutingDecision {
            backend_idx: 0,
            confidence: 0.8,
            predicted_quality: 0.9,
            predicted_latency_us: 50,
            routing_latency_us: 10,
            all_qualities: vec![0.9, 0.7, 0.5, 0.8, 0.6],
            all_latencies: vec![50, 100, 200, 75, 150],
        };

        // Test top_k
        let top2 = decision.top_k(2);
        assert_eq!(top2.len(), 2);
        assert_eq!(top2[0].0, 0); // Highest quality

        // Test within_latency
        let fast = decision.within_latency(100);
        assert_eq!(fast.len(), 3); // indices 0, 1, 3
    }
}
