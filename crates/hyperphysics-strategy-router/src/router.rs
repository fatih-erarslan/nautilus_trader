//! Main strategy router implementation
//!
//! Combines gating, experts, load balancing, and pBit noise for
//! adaptive trading strategy selection.

use crate::{
    RouterError, Result,
    expert::{Expert, ExpertConfig, StandardExpert, HyperbolicExpert, LinearExpert},
    gating::{GatingNetwork, GatingConfig},
    pbit_noise::{PBitNoiseGenerator, NoiseConfig},
    market_regime::{RegimeDetector, RegimeConfig, MarketRegime},
    load_balancer::{LoadBalancer, LoadBalancerConfig},
};
use serde::{Deserialize, Serialize};

/// Configuration for the strategy router
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Input dimension
    pub input_dim: usize,
    /// Number of experts
    pub num_experts: usize,
    /// Number of experts to route to (top-K)
    pub top_k: usize,
    /// Expert configuration
    pub expert_config: ExpertConfig,
    /// Temperature for routing
    pub temperature: f64,
    /// Whether to use pBit noise
    pub use_pbit_noise: bool,
    /// pBit noise temperature
    pub pbit_temperature: f64,
    /// Whether to use market regime detection
    pub use_regime_detection: bool,
    /// Capacity factor for load balancing
    pub capacity_factor: f64,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            input_dim: 64,
            num_experts: 8,
            top_k: 2,
            expert_config: ExpertConfig::default(),
            temperature: 1.0,
            use_pbit_noise: true,
            pbit_temperature: 1.0,
            use_regime_detection: true,
            capacity_factor: 1.25,
        }
    }
}

/// Result of routing decision
#[derive(Debug, Clone)]
pub struct RoutingResult {
    /// Selected experts with their weights
    pub expert_routing: Vec<(usize, f64)>,
    /// Combined output from experts
    pub output: Vec<f64>,
    /// Current market regime
    pub regime: MarketRegime,
    /// Load balancing loss
    pub auxiliary_loss: f64,
}

/// Main strategy router
pub struct StrategyRouter {
    config: RouterConfig,
    /// Expert models
    experts: Vec<Box<dyn Expert>>,
    /// Gating network
    gating: GatingNetwork,
    /// pBit noise generator
    noise_gen: Option<PBitNoiseGenerator>,
    /// Market regime detector
    regime_detector: Option<RegimeDetector>,
    /// Load balancer
    load_balancer: LoadBalancer,
    /// Routing probability history (for auxiliary loss)
    prob_history: Vec<Vec<f64>>,
    /// Logits history (for z-loss)
    logits_history: Vec<Vec<f64>>,
}

impl StrategyRouter {
    /// Create new strategy router
    pub fn new(config: RouterConfig) -> Result<Self> {
        // Validate config
        if config.top_k > config.num_experts {
            return Err(RouterError::TopKExceedsExperts {
                k: config.top_k,
                n: config.num_experts,
            });
        }

        if config.temperature <= 0.0 {
            return Err(RouterError::InvalidTemperature(config.temperature));
        }

        // Create experts
        let mut experts: Vec<Box<dyn Expert>> = Vec::with_capacity(config.num_experts);
        for i in 0..config.num_experts {
            let expert: Box<dyn Expert> = match i % 3 {
                0 => Box::new(StandardExpert::new(
                    config.expert_config.clone(),
                    format!("standard_{}", i),
                )),
                1 => Box::new(HyperbolicExpert::new(
                    config.expert_config.clone(),
                    format!("hyperbolic_{}", i),
                )),
                _ => Box::new(LinearExpert::new(
                    config.expert_config.clone(),
                    format!("linear_{}", i),
                    i as u64,
                )),
            };
            experts.push(expert);
        }

        // Create gating network
        let gating_config = GatingConfig {
            input_dim: config.input_dim,
            num_experts: config.num_experts,
            top_k: config.top_k,
            temperature: config.temperature,
            noisy_gating: config.use_pbit_noise,
            ..Default::default()
        };
        let gating = GatingNetwork::new(gating_config)?;

        // Create pBit noise generator
        let noise_gen = if config.use_pbit_noise {
            Some(PBitNoiseGenerator::new(NoiseConfig {
                num_pbits: config.num_experts,
                temperature: config.pbit_temperature,
                ..Default::default()
            })?)
        } else {
            None
        };

        // Create regime detector
        let regime_detector = if config.use_regime_detection {
            Some(RegimeDetector::new(RegimeConfig::default()))
        } else {
            None
        };

        // Create load balancer
        let load_balancer = LoadBalancer::new(LoadBalancerConfig {
            num_experts: config.num_experts,
            capacity_factor: config.capacity_factor,
            ..Default::default()
        });

        Ok(Self {
            config,
            experts,
            gating,
            noise_gen,
            regime_detector,
            load_balancer,
            prob_history: Vec::new(),
            logits_history: Vec::new(),
        })
    }

    /// Route input through selected experts
    pub fn route(&mut self, input: &[f64]) -> Result<RoutingResult> {
        if input.len() != self.config.input_dim {
            return Err(RouterError::DimensionMismatch {
                expected: self.config.input_dim,
                actual: input.len(),
            });
        }

        // Generate pBit noise
        let noise = self.noise_gen.as_mut().map(|gen| gen.generate().to_vec());

        // Get routing decision
        let routing = self.gating.route(input, noise.as_deref())?;

        // Update load balancer
        self.load_balancer.update_loads(&routing);

        // Store probabilities for auxiliary loss
        let probs = self.gating.all_probabilities(input)?;
        self.prob_history.push(probs);

        // Compute expert outputs and combine
        let mut combined_output = vec![0.0; self.config.input_dim];

        for &(expert_id, weight) in &routing {
            if expert_id < self.experts.len() {
                // Skip over-capacity experts (optional)
                if self.load_balancer.is_over_capacity(expert_id) {
                    continue;
                }

                let expert_output = self.experts[expert_id].compute(input)?;
                for (i, &val) in expert_output.iter().enumerate() {
                    if i < combined_output.len() {
                        combined_output[i] += weight * val;
                    }
                }
            }
        }

        // Get current regime
        let regime = self.regime_detector
            .as_ref()
            .map(|d| d.current_regime())
            .unwrap_or(MarketRegime::Transitional);

        // Compute auxiliary loss
        let auxiliary_loss = self.load_balancer.compute_balance_loss(&self.prob_history);

        // Limit history size
        if self.prob_history.len() > 1000 {
            self.prob_history.drain(0..500);
        }

        Ok(RoutingResult {
            expert_routing: routing,
            output: combined_output,
            regime,
            auxiliary_loss,
        })
    }

    /// Update regime detector with new return observation
    pub fn update_regime(&mut self, return_: f64) {
        if let Some(detector) = &mut self.regime_detector {
            detector.update(return_);
        }
    }

    /// Get current market regime
    pub fn current_regime(&self) -> MarketRegime {
        self.regime_detector
            .as_ref()
            .map(|d| d.current_regime())
            .unwrap_or(MarketRegime::Transitional)
    }

    /// Get load balancer summary
    pub fn load_summary(&self) -> crate::load_balancer::LoadSummary {
        self.load_balancer.summary()
    }

    /// Reset internal state
    pub fn reset(&mut self) {
        self.load_balancer.reset();
        self.prob_history.clear();
        self.logits_history.clear();
        if let Some(detector) = &mut self.regime_detector {
            detector.reset();
        }
    }

    /// Get number of experts
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Get expert by index
    pub fn expert(&self, index: usize) -> Option<&dyn Expert> {
        self.experts.get(index).map(|e| e.as_ref())
    }

    /// Anneal pBit temperature
    pub fn anneal_noise(&mut self, factor: f64) {
        if let Some(gen) = &mut self.noise_gen {
            gen.anneal(factor);
        }
    }

    /// Set pBit temperature
    pub fn set_noise_temperature(&mut self, temperature: f64) -> Result<()> {
        if let Some(gen) = &mut self.noise_gen {
            gen.set_temperature(temperature)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_creation() {
        let config = RouterConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            expert_config: ExpertConfig {
                dim: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let router = StrategyRouter::new(config).unwrap();

        assert_eq!(router.num_experts(), 4);
    }

    #[test]
    fn test_routing() {
        let config = RouterConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            expert_config: ExpertConfig {
                dim: 4,
                ..Default::default()
            },
            use_pbit_noise: false,
            use_regime_detection: false,
            ..Default::default()
        };
        let mut router = StrategyRouter::new(config).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = router.route(&input).unwrap();

        assert_eq!(result.expert_routing.len(), 2);
        assert_eq!(result.output.len(), 4);

        // Weights should sum to 1
        let weight_sum: f64 = result.expert_routing.iter().map(|(_, w)| w).sum();
        assert!((weight_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_regime_update() {
        let config = RouterConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            expert_config: ExpertConfig {
                dim: 4,
                ..Default::default()
            },
            use_regime_detection: true,
            ..Default::default()
        };
        let mut router = StrategyRouter::new(config).unwrap();

        // Update with returns
        for _ in 0..30 {
            router.update_regime(0.01);
        }

        let regime = router.current_regime();
        // After consistent positive returns, should detect trending
        assert_eq!(regime, MarketRegime::Trending);
    }

    #[test]
    fn test_noise_annealing() {
        let config = RouterConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            expert_config: ExpertConfig {
                dim: 4,
                ..Default::default()
            },
            use_pbit_noise: true,
            pbit_temperature: 10.0,
            ..Default::default()
        };
        let mut router = StrategyRouter::new(config).unwrap();

        router.anneal_noise(0.9);
        router.anneal_noise(0.9);

        // Should not panic
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let _ = router.route(&input).unwrap();
    }

    #[test]
    fn test_invalid_config() {
        let config = RouterConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 10, // Invalid
            ..Default::default()
        };
        let result = StrategyRouter::new(config);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let config = RouterConfig {
            input_dim: 4,
            num_experts: 4,
            top_k: 2,
            expert_config: ExpertConfig {
                dim: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut router = StrategyRouter::new(config).unwrap();

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let _ = router.route(&input).unwrap();
        let _ = router.route(&input).unwrap();

        router.reset();

        let summary = router.load_summary();
        assert_eq!(summary.total_tokens, 0);
    }
}
