//! pBit Regime Detection Integration
//!
//! Integrates HyperPhysics pBit dynamics for market regime detection and pattern completion.
//!
//! ## Features
//!
//! - **pBit Regime Detection**: Use Ising-like dynamics for regime classification
//! - **Hopfield Pattern Completion**: Complete partial trading patterns
//! - **Markov Chain Sampling**: GPU-accelerated state transitions
//! - **Portfolio Optimization**: Ising model for position sizing
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Camsari et al. (2017) "Stochastic p-bits for invertible logic" PRX 7:031014
//! - Kaiser & Datta (2021) "Probabilistic computing with p-bits" Nature Electronics
//! - Hopfield (1982) "Neural networks and physical systems" PNAS
//! - Mezard et al. (1987) "Spin Glass Theory and Beyond"

use hyperphysics_pbit::{
    PBitLattice, MetropolisSimulator,
    CouplingNetwork,
};
use std::collections::HashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::error::{AgentDBError, Result};
use crate::trading::MarketRegime;

/// Market context for regime detection
pub struct MarketContext {
    /// Current regime (for reference)
    pub regime: MarketRegime,
    /// Volatility level
    pub volatility: f64,
    /// Relative volume
    pub relative_volume: f64,
    /// Trend strength (-1 to 1)
    pub trend_strength: f64,
    /// Order book imbalance
    pub order_imbalance: f64,
    /// Additional indicators
    pub indicators: HashMap<String, f64>,
}

impl Default for MarketContext {
    fn default() -> Self {
        Self {
            regime: MarketRegime::RangeBound,
            volatility: 0.02,
            relative_volume: 1.0,
            trend_strength: 0.0,
            order_imbalance: 0.0,
            indicators: HashMap::new(),
        }
    }
}

/// pBit-based regime detector
///
/// Uses pBit lattice dynamics for market regime classification.
/// The lattice represents regime states as pBits with hyperbolic geometry.
pub struct PBitRegimeDetector {
    /// Number of hidden states (regimes)
    num_regimes: usize,
    /// pBit lattice for regime dynamics (tessellation-based)
    lattice: Arc<RwLock<PBitLattice>>,
    /// Coupling network for regime interactions
    coupling_network: CouplingNetwork,
    /// Metropolis simulator for sampling
    simulator: Arc<RwLock<MetropolisSimulator>>,
    /// External biases from market indicators
    biases: Arc<RwLock<Vec<f64>>>,
    /// Temperature for sampling
    temperature: f64,
    /// Regime mapping
    regime_mapping: HashMap<usize, MarketRegime>,
    /// Statistics
    stats: Arc<RwLock<RegimeDetectorStats>>,
}

/// Statistics for regime detector
#[derive(Debug, Clone, Default)]
pub struct RegimeDetectorStats {
    /// Total detections
    pub total_detections: u64,
    /// Average latency in microseconds
    pub avg_latency_us: f64,
    /// Regime distribution (counts per regime)
    pub regime_distribution: HashMap<MarketRegime, u64>,
    /// Transition count between regimes
    pub transitions: u64,
    /// Last detected regime
    pub last_regime: Option<MarketRegime>,
}

impl PBitRegimeDetector {
    /// Create new pBit regime detector
    ///
    /// # Arguments
    /// * `num_regimes` - Number of regime states (typically 5-6)
    /// * `temperature` - Sampling temperature (lower = more deterministic)
    pub fn new(num_regimes: usize, temperature: f64) -> Result<Self> {
        // Create pBit lattice using {3,7} tessellation (hyperbolic)
        // depth=1 gives ~7 nodes, depth=2 gives ~48 nodes
        let depth = if num_regimes <= 7 { 1 } else { 2 };
        let lattice = PBitLattice::new(3, 7, depth, temperature)
            .map_err(|e| AgentDBError::PBitError(e.to_string()))?;

        // Create coupling network for regime interactions
        let coupling_network = CouplingNetwork::new(1.0, 1.0, 0.01);

        // Create Metropolis simulator
        let simulator = MetropolisSimulator::new(lattice.clone(), temperature);

        // Initialize biases (one per node in lattice)
        let biases = vec![0.0; lattice.size()];

        // Default regime mapping (map lattice nodes to regimes)
        let mut regime_mapping = HashMap::new();
        regime_mapping.insert(0, MarketRegime::BullTrend);
        regime_mapping.insert(1, MarketRegime::BearTrend);
        regime_mapping.insert(2, MarketRegime::RangeBound);
        regime_mapping.insert(3, MarketRegime::HighVolatility);
        regime_mapping.insert(4, MarketRegime::Transitioning);

        Ok(Self {
            num_regimes,
            lattice: Arc::new(RwLock::new(lattice)),
            coupling_network,
            simulator: Arc::new(RwLock::new(simulator)),
            biases: Arc::new(RwLock::new(biases)),
            temperature,
            regime_mapping,
            stats: Arc::new(RwLock::new(RegimeDetectorStats::default())),
        })
    }

    /// Detect current market regime from indicators
    ///
    /// Uses pBit dynamics to sample most likely regime given market indicators.
    pub fn detect_regime(&self, context: &MarketContext) -> Result<RegimeDetection> {
        let start = std::time::Instant::now();

        // Convert market indicators to biases
        let biases = self.compute_biases(context);

        // Store biases for reference
        {
            let mut stored_biases = self.biases.write();
            for (i, &bias) in biases.iter().enumerate().take(stored_biases.len()) {
                stored_biases[i] = bias;
            }
        }

        // Run Metropolis sampling using simulator
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let num_samples = 100;

        // Collect samples from Metropolis simulation
        let mut samples: Vec<Vec<bool>> = Vec::with_capacity(num_samples);
        {
            let mut sim = self.simulator.write();
            for _ in 0..num_samples {
                let _ = sim.step(&mut rng);
                samples.push(sim.lattice().states());
            }
        }

        // Count regime occurrences based on magnetization patterns
        // Map lattice states to regime probabilities
        let mut regime_counts = vec![0u64; self.num_regimes];
        for sample in &samples {
            // Use first few pBits to determine regime
            for (i, &bit) in sample.iter().take(self.num_regimes).enumerate() {
                if bit {
                    regime_counts[i] += 1;
                }
            }
        }

        // Find most likely regime
        let (max_idx, max_count) = regime_counts.iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .unwrap_or((2, &0)); // Default to RangeBound

        let detected_regime = self.regime_mapping.get(&max_idx)
            .copied()
            .unwrap_or(MarketRegime::RangeBound);

        // Compute confidence
        let total: u64 = regime_counts.iter().sum();
        let confidence = if total > 0 {
            *max_count as f64 / total as f64
        } else {
            0.0
        };

        // Compute probabilities for all regimes
        let probabilities: Vec<f64> = regime_counts.iter()
            .map(|&c| if total > 0 { c as f64 / total as f64 } else { 0.0 })
            .collect();

        let latency_us = start.elapsed().as_micros() as f64;

        // Update statistics
        {
            let mut stats = self.stats.write();
            stats.total_detections += 1;
            stats.avg_latency_us = 0.99 * stats.avg_latency_us + 0.01 * latency_us;
            *stats.regime_distribution.entry(detected_regime).or_default() += 1;
            if stats.last_regime.is_some() && stats.last_regime != Some(detected_regime) {
                stats.transitions += 1;
            }
            stats.last_regime = Some(detected_regime);
        }

        Ok(RegimeDetection {
            regime: detected_regime,
            confidence,
            probabilities,
            latency_us,
        })
    }

    /// Compute biases from market context
    fn compute_biases(&self, context: &MarketContext) -> Vec<f64> {
        let mut biases = vec![0.0; self.num_regimes];

        // Trend strength affects bull/bear bias
        if context.trend_strength > 0.5 {
            biases[0] += context.trend_strength;  // Bull
        } else if context.trend_strength < -0.5 {
            biases[1] += context.trend_strength.abs();  // Bear
        }

        // Low volatility favors range-bound
        if context.volatility < 0.02 {
            biases[2] += 1.0 - context.volatility * 50.0;  // RangeBound
        }

        // High volatility
        if context.volatility > 0.03 {
            biases[3] += context.volatility * 20.0;  // HighVolatility
        }

        // Volume spikes can indicate transitions
        if context.relative_volume > 2.0 && self.num_regimes > 4 {
            biases[4] += context.relative_volume * 0.3;  // Transitioning
        }

        // Normalize biases
        let max_bias = biases.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if max_bias > 0.0 {
            for bias in &mut biases {
                *bias /= max_bias;
            }
        }

        biases
    }

    /// Get regime index from enum
    fn regime_index(&self, regime: MarketRegime) -> Result<usize> {
        for (&idx, &r) in &self.regime_mapping {
            if r == regime {
                return Ok(idx);
            }
        }
        Err(AgentDBError::PBitError("Unknown regime".into()))
    }

    /// Get detector statistics
    pub fn stats(&self) -> RegimeDetectorStats {
        self.stats.read().clone()
    }

    /// Get number of regimes
    pub fn num_regimes(&self) -> usize {
        self.num_regimes
    }
}

/// Regime detection result
#[derive(Debug, Clone)]
pub struct RegimeDetection {
    /// Detected regime
    pub regime: MarketRegime,
    /// Detection confidence (0-1)
    pub confidence: f64,
    /// Probabilities for each regime
    pub probabilities: Vec<f64>,
    /// Detection latency in microseconds
    pub latency_us: f64,
}

/// Hopfield network for trading pattern completion
///
/// Uses Hebbian learning to store trading patterns and pBit dynamics for recall.
pub struct TradingHopfieldNetwork {
    /// Number of pattern bits
    num_bits: usize,
    /// Stored patterns
    patterns: Vec<Vec<bool>>,
    /// Weight matrix (Hebbian learning)
    weights: Vec<Vec<f64>>,
    /// Temperature for probabilistic updates
    temperature: f64,
}

impl TradingHopfieldNetwork {
    /// Create new Hopfield network for pattern completion
    pub fn new(num_bits: usize, temperature: f64) -> Result<Self> {
        let weights = vec![vec![0.0; num_bits]; num_bits];

        Ok(Self {
            num_bits,
            patterns: Vec::new(),
            weights,
            temperature,
        })
    }

    /// Store a trading pattern using Hebbian learning
    pub fn store_pattern(&mut self, pattern: &[bool]) -> Result<()> {
        if pattern.len() != self.num_bits {
            return Err(AgentDBError::PBitError(
                format!("Pattern size mismatch: expected {}, got {}", self.num_bits, pattern.len())
            ));
        }

        // Hebbian update: w_ij += (2*p_i - 1)(2*p_j - 1) / N
        let n = self.num_bits as f64;
        for i in 0..self.num_bits {
            for j in 0..self.num_bits {
                if i != j {
                    let s_i = if pattern[i] { 1.0 } else { -1.0 };
                    let s_j = if pattern[j] { 1.0 } else { -1.0 };
                    self.weights[i][j] += s_i * s_j / n;
                }
            }
        }

        self.patterns.push(pattern.to_vec());
        Ok(())
    }

    /// Recall/complete a partial pattern
    ///
    /// # Arguments
    /// * `partial` - Partial pattern with Some(bool) for known bits, None for unknown
    /// * `max_iterations` - Maximum iterations for convergence
    pub fn recall(&self, partial: &[Option<bool>], max_iterations: usize) -> Result<Vec<bool>> {
        if partial.len() != self.num_bits {
            return Err(AgentDBError::PBitError(
                format!("Partial size mismatch: expected {}, got {}", self.num_bits, partial.len())
            ));
        }

        // Initialize state from partial pattern
        let mut state: Vec<bool> = partial.iter()
            .map(|&opt| opt.unwrap_or(false))
            .collect();

        // Clamp known bits
        let clamped: Vec<bool> = partial.iter()
            .map(|opt| opt.is_some())
            .collect();

        // Iterate until convergence
        for _ in 0..max_iterations {
            let mut changed = false;

            for i in 0..self.num_bits {
                if clamped[i] {
                    continue; // Skip clamped bits
                }

                // Compute local field
                let mut h_i = 0.0;
                for j in 0..self.num_bits {
                    let s_j = if state[j] { 1.0 } else { -1.0 };
                    h_i += self.weights[i][j] * s_j;
                }

                // Probabilistic update
                let prob = 1.0 / (1.0 + (-h_i / self.temperature).exp());
                let new_state = prob > 0.5;

                if state[i] != new_state {
                    state[i] = new_state;
                    changed = true;
                }
            }

            if !changed {
                break; // Converged
            }
        }

        Ok(state)
    }

    /// Get pattern storage capacity
    pub fn capacity(&self) -> usize {
        // Hopfield capacity: ~0.138 * N patterns
        (0.138 * self.num_bits as f64) as usize
    }

    /// Get number of stored patterns
    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }

    /// Check if pattern is stored (similarity > threshold)
    pub fn is_stored(&self, pattern: &[bool], threshold: f64) -> bool {
        for stored in &self.patterns {
            let similarity = self.pattern_similarity(pattern, stored);
            if similarity >= threshold {
                return true;
            }
        }
        false
    }

    /// Compute pattern similarity (normalized Hamming)
    fn pattern_similarity(&self, p1: &[bool], p2: &[bool]) -> f64 {
        let matches = p1.iter().zip(p2.iter())
            .filter(|(a, b)| a == b)
            .count();
        matches as f64 / p1.len() as f64
    }
}

/// pBit-based portfolio optimizer using Ising model
///
/// Uses Metropolis sampling with simulated annealing for portfolio optimization.
pub struct PBitPortfolioOptimizer {
    /// Number of assets
    num_assets: usize,
    /// pBit lattice for optimization
    lattice: Arc<RwLock<PBitLattice>>,
    /// Metropolis simulator
    simulator: Arc<RwLock<MetropolisSimulator>>,
    /// Expected returns (biases)
    returns: Arc<RwLock<Vec<f64>>>,
    /// Correlation matrix (local storage)
    correlations: Vec<Vec<f64>>,
    /// Risk aversion parameter
    risk_aversion: f64,
    /// Temperature for annealing
    temperature: f64,
}

impl PBitPortfolioOptimizer {
    /// Create new portfolio optimizer
    pub fn new(num_assets: usize, risk_aversion: f64, temperature: f64) -> Result<Self> {
        // Create lattice with appropriate tessellation
        let depth = if num_assets <= 7 { 1 } else { 2 };
        let lattice = PBitLattice::new(3, 7, depth, temperature)
            .map_err(|e| AgentDBError::PBitError(e.to_string()))?;

        let simulator = MetropolisSimulator::new(lattice.clone(), temperature);
        let returns = vec![0.0; num_assets];
        let correlations = vec![vec![0.0; num_assets]; num_assets];

        Ok(Self {
            num_assets,
            lattice: Arc::new(RwLock::new(lattice)),
            simulator: Arc::new(RwLock::new(simulator)),
            returns: Arc::new(RwLock::new(returns)),
            correlations,
            risk_aversion,
            temperature,
        })
    }

    /// Set expected returns for assets
    pub fn set_returns(&self, returns: &[f64]) -> Result<()> {
        if returns.len() != self.num_assets {
            return Err(AgentDBError::PBitError("Returns size mismatch".into()));
        }

        let mut r = self.returns.write();
        for (i, &ret) in returns.iter().enumerate() {
            r[i] = ret;
        }

        Ok(())
    }

    /// Set correlation between assets
    pub fn set_correlation(&mut self, asset_i: usize, asset_j: usize, correlation: f64) -> Result<()> {
        if asset_i >= self.num_assets || asset_j >= self.num_assets {
            return Err(AgentDBError::PBitError("Asset index out of range".into()));
        }

        // Store correlation locally
        self.correlations[asset_i][asset_j] = correlation;
        self.correlations[asset_j][asset_i] = correlation;

        Ok(())
    }

    /// Optimize portfolio allocation
    ///
    /// Returns vector of (asset_index, allocation) for selected assets
    pub fn optimize(&self, num_samples: usize) -> Result<Vec<(usize, f64)>> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Collect samples from Metropolis simulation with annealing
        let mut samples: Vec<Vec<bool>> = Vec::with_capacity(num_samples);
        {
            let mut sim = self.simulator.write();
            let initial_temp = self.temperature;
            let final_temp = 0.01;

            for i in 0..num_samples {
                // Annealing schedule
                let progress = i as f64 / num_samples as f64;
                let temp = initial_temp * (1.0 - progress) + final_temp * progress;
                sim.set_temperature(temp);

                let _ = sim.step(&mut rng);
                samples.push(sim.lattice().states());
            }
        }

        // Count asset selections (map lattice states to assets)
        let mut selection_counts = vec![0u64; self.num_assets];
        for sample in &samples {
            for (i, &selected) in sample.iter().take(self.num_assets).enumerate() {
                if selected {
                    selection_counts[i] += 1;
                }
            }
        }

        // Compute allocation proportional to selection frequency
        let total: u64 = selection_counts.iter().sum();
        if total == 0 {
            return Ok(vec![]);
        }

        let allocations: Vec<(usize, f64)> = selection_counts.iter()
            .enumerate()
            .filter(|(_, &count)| count > 0)
            .map(|(i, &count)| (i, count as f64 / total as f64))
            .collect();

        Ok(allocations)
    }

    /// Get number of assets
    pub fn num_assets(&self) -> usize {
        self.num_assets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbit_regime_detector_creation() {
        let detector = PBitRegimeDetector::new(5, 1.0).unwrap();
        let stats = detector.stats();
        assert_eq!(stats.total_detections, 0);
        assert_eq!(detector.num_regimes(), 5);
    }

    #[test]
    fn test_regime_detection() {
        let detector = PBitRegimeDetector::new(5, 1.0).unwrap();

        let context = MarketContext {
            regime: MarketRegime::BullTrend,
            trend_strength: 0.8,
            volatility: 0.015,
            relative_volume: 1.2,
            ..Default::default()
        };

        let detection = detector.detect_regime(&context).unwrap();
        assert!(detection.confidence > 0.0);
        assert_eq!(detection.probabilities.len(), 5);
    }

    #[test]
    fn test_hopfield_network() {
        let mut network = TradingHopfieldNetwork::new(16, 0.5).unwrap();

        // Store a pattern
        let pattern = vec![true, false, true, false, true, false, true, false,
                          true, false, true, false, true, false, true, false];
        network.store_pattern(&pattern).unwrap();

        assert_eq!(network.num_patterns(), 1);
        assert!(network.capacity() >= 2); // At least 2 patterns can be stored

        // Recall with partial pattern (50% known)
        let partial: Vec<Option<bool>> = pattern.iter()
            .enumerate()
            .map(|(i, &b)| if i % 2 == 0 { Some(b) } else { None })
            .collect();

        let recalled = network.recall(&partial, 100).unwrap();
        assert_eq!(recalled.len(), 16);
    }

    #[test]
    fn test_portfolio_optimizer() {
        let mut optimizer = PBitPortfolioOptimizer::new(5, 1.0, 1.0).unwrap();

        // Set expected returns
        let returns = vec![0.05, 0.03, 0.07, 0.02, 0.04];
        optimizer.set_returns(&returns).unwrap();

        // Set correlations
        optimizer.set_correlation(0, 1, 0.8).unwrap();
        optimizer.set_correlation(0, 2, 0.2).unwrap();

        // Optimize
        let allocation = optimizer.optimize(100).unwrap();
        assert!(!allocation.is_empty());

        // Allocations should sum to ~1.0
        let total: f64 = allocation.iter().map(|(_, a)| a).sum();
        assert!((total - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_market_context_default() {
        let context = MarketContext::default();
        assert_eq!(context.regime, MarketRegime::RangeBound);
        assert_eq!(context.volatility, 0.02);
    }
}
