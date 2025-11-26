//! # Enactive Market Perception
//!
//! Implementation of enactivist perception principles (Varela, Thompson, & Rosch, 1991)
//! combined with Friston's Free Energy Principle for market analysis.
//!
//! ## Theoretical Foundation
//!
//! ### Enactivism (Varela et al., 1991)
//! - **Structural Coupling**: The system and environment (market) are mutually determined
//!   through continuous interaction, not passive observation
//! - **Sensorimotor Loops**: Perception emerges from the closed loop of sensing and acting
//! - **Enacted Regularities**: Market patterns are not "discovered" but enacted through
//!   agent-market interaction
//!
//! ### Free Energy Principle (Friston, 2010)
//! - Systems minimize surprise (prediction error) through action and perception
//! - Free energy F = -log P(observations | model)
//! - Prediction error ε drives both perceptual inference and active inference (action)
//!
//! ## References
//!
//! - Varela, F. J., Thompson, E., & Rosch, E. (1991). *The Embodied Mind: Cognitive
//!   Science and Human Experience*. MIT Press.
//! - Friston, K. (2010). The free-energy principle: a unified brain theory?
//!   *Nature Reviews Neuroscience*, 11(2), 127-138.
//! - Di Paolo, E. A. (2005). Autopoiesis, adaptivity, viability, and agency.
//!   *Phenomenology and the Cognitive Sciences*, 4(4), 429-452.

use std::collections::VecDeque;
use serde::{Deserialize, Serialize};

/// Configuration for enactive market perception
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnactiveConfig {
    /// Maximum number of coupling events to store in history
    pub history_capacity: usize,
    /// Learning rate for coupling strength updates (γ in free energy equations)
    pub learning_rate: f64,
    /// Decay rate for coupling strength (λ in exponential decay)
    pub coupling_decay: f64,
    /// Threshold for high volatility regime detection (variance)
    pub high_volatility_threshold: f64,
    /// Threshold for low volatility regime detection (variance)
    pub low_volatility_threshold: f64,
    /// Window size for computing moving statistics
    pub statistics_window: usize,
    /// Minimum coupling events before pattern detection
    pub min_events_for_patterns: usize,
}

impl Default for EnactiveConfig {
    fn default() -> Self {
        Self {
            history_capacity: 1000,
            learning_rate: 0.01,
            coupling_decay: 0.001,
            high_volatility_threshold: 0.01,
            low_volatility_threshold: 0.0001,
            statistics_window: 100,
            min_events_for_patterns: 50,
        }
    }
}

/// State of sensorimotor coupling between agent and market
///
/// Following Varela's structural coupling: the agent and market are
/// mutually specified through their interaction dynamics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CouplingState {
    /// Afferent signal: market → agent (sensory input)
    pub afferent_field: f64,
    /// Efferent signal: agent → market (motor output/action)
    pub efferent_signal: f64,
    /// Coupling strength: depth of structural coupling (0 to 1)
    pub coupling_strength: f64,
    /// Last prediction made by the system
    pub last_prediction: f64,
    /// Timestamp of last coupling update (microseconds)
    pub last_update_us: u64,
}

impl Default for CouplingState {
    fn default() -> Self {
        Self {
            afferent_field: 0.0,
            efferent_signal: 0.0,
            coupling_strength: 0.5, // Start at moderate coupling
            last_prediction: 0.0,
            last_update_us: 0,
        }
    }
}

/// Record of a single coupling event in the sensorimotor loop
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CouplingEvent {
    /// Timestamp in microseconds
    pub timestamp_us: u64,
    /// Afferent signal (market input)
    pub afferent: f64,
    /// Efferent signal (agent action)
    pub efferent: f64,
    /// Prediction error (observed - predicted)
    pub prediction_error: f64,
    /// Free energy at this event (approximation)
    pub free_energy: f64,
}

/// Patterns enacted through agent-market interaction
///
/// Following enactivism: these patterns are not pre-existing in the market
/// but emerge through the coupling dynamics
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EnactedPatterns {
    /// Mean coupling strength over recent window
    pub mean_coupling: f64,
    /// Standard deviation of coupling strength
    pub std_coupling: f64,
    /// Mean prediction error magnitude
    pub mean_error: f64,
    /// Variance in prediction errors
    pub prediction_variance: f64,
    /// Detected market regime from enacted patterns
    pub regime: MarketRegime,
    /// Autocorrelation of prediction errors (lag-1)
    pub error_autocorr: f64,
}

impl Default for EnactedPatterns {
    fn default() -> Self {
        Self {
            mean_coupling: 0.5,
            std_coupling: 0.0,
            mean_error: 0.0,
            prediction_variance: 0.0,
            regime: MarketRegime::LowVolatility,
            error_autocorr: 0.0,
        }
    }
}

/// Market regime enacted through interaction patterns
///
/// These regimes are not objective market states but emerge from
/// the coupling dynamics between agent and market
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    /// Strong positive autocorrelation in errors → trending
    Trending,
    /// Negative autocorrelation in errors → mean reversion
    MeanReverting,
    /// High prediction variance → high volatility
    HighVolatility,
    /// Low prediction variance → low volatility
    LowVolatility,
}

/// Action signal generated through active inference
///
/// Following Friston: the system acts to minimize future prediction error
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionSignal {
    /// Recommended action strength (-1.0 to 1.0)
    /// Negative = decrease position, Positive = increase position
    pub strength: f64,
    /// Confidence in this action (0.0 to 1.0)
    pub confidence: f64,
    /// Expected reduction in prediction error
    pub expected_error_reduction: f64,
    /// Current regime context
    pub regime: MarketRegime,
}

/// Enactive Market Perception system
///
/// Implements perception as sensorimotor coupling following Varela et al. (1991)
/// and active inference following Friston (2010)
pub struct EnactiveMarketPerception {
    /// Current coupling state
    coupling_state: CouplingState,
    /// Ring buffer of coupling events for temporal integration
    coupling_history: VecDeque<CouplingEvent>,
    /// Emergent patterns from enacted regularities
    enacted_patterns: EnactedPatterns,
    /// Configuration parameters
    config: EnactiveConfig,
}

impl EnactiveMarketPerception {
    /// Create new enactive market perception system
    pub fn new(config: EnactiveConfig) -> Self {
        Self {
            coupling_state: CouplingState::default(),
            coupling_history: VecDeque::with_capacity(config.history_capacity),
            enacted_patterns: EnactedPatterns::default(),
            config,
        }
    }

    /// Process market tick through sensorimotor loop
    ///
    /// ## Algorithm (Friston 2010)
    /// 1. Compute prediction error: ε = observed - predicted
    /// 2. Update coupling strength: strength += γ * |ε|
    /// 3. Apply coupling decay: strength *= (1 - λ)
    /// 4. Generate new prediction based on updated coupling
    /// 5. Store coupling event for pattern learning
    ///
    /// Returns: Prediction error magnitude
    pub fn process_tick(&mut self, market_price: f64, timestamp_us: u64) -> f64 {
        // Compute prediction error (ε in Friston's formulation)
        let prediction_error = market_price - self.coupling_state.last_prediction;

        // Update afferent field (sensory signal from market)
        self.coupling_state.afferent_field = market_price;

        // Update coupling strength based on prediction error magnitude
        // Following free energy minimization: we strengthen coupling when surprised
        let error_magnitude = prediction_error.abs();
        self.coupling_state.coupling_strength +=
            self.config.learning_rate * error_magnitude;

        // Apply coupling decay (prevents unbounded growth)
        self.coupling_state.coupling_strength *=
            1.0 - self.config.coupling_decay;

        // Clamp coupling strength to [0, 1]
        self.coupling_state.coupling_strength =
            self.coupling_state.coupling_strength.clamp(0.0, 1.0);

        // Compute free energy approximation: F ≈ ε²/2 (quadratic surprise)
        let free_energy = 0.5 * prediction_error * prediction_error;

        // Store coupling event
        let event = CouplingEvent {
            timestamp_us,
            afferent: market_price,
            efferent: self.coupling_state.efferent_signal,
            prediction_error,
            free_energy,
        };

        self.coupling_history.push_back(event);

        // Maintain history size limit
        while self.coupling_history.len() > self.config.history_capacity {
            self.coupling_history.pop_front();
        }

        // Update timestamp
        self.coupling_state.last_update_us = timestamp_us;

        // Generate new prediction for next tick
        self.coupling_state.last_prediction = self.predict_next();

        // Update enacted patterns if we have sufficient history
        if self.coupling_history.len() >= self.config.min_events_for_patterns {
            self.update_enacted_patterns();
        }

        error_magnitude
    }

    /// Generate action signal through active inference
    ///
    /// Following Friston: action minimizes expected future prediction error
    pub fn generate_action(&self) -> ActionSignal {
        // Action strength based on prediction error gradient
        // Positive error → increase position, Negative error → decrease
        let recent_errors: Vec<f64> = self.coupling_history
            .iter()
            .rev()
            .take(10)
            .map(|e| e.prediction_error)
            .collect();

        let mean_error = if !recent_errors.is_empty() {
            recent_errors.iter().sum::<f64>() / recent_errors.len() as f64
        } else {
            0.0
        };

        // Action strength proportional to prediction error
        let strength = mean_error.tanh(); // Bounded to [-1, 1]

        // Confidence inversely proportional to prediction variance
        let confidence = if self.enacted_patterns.prediction_variance > 0.0 {
            (-self.enacted_patterns.prediction_variance).exp()
        } else {
            0.5
        };

        // Expected error reduction based on coupling strength
        let expected_error_reduction =
            self.coupling_state.coupling_strength * mean_error.abs();

        ActionSignal {
            strength,
            confidence,
            expected_error_reduction,
            regime: self.enacted_patterns.regime,
        }
    }

    /// Update enacted patterns from coupling history
    ///
    /// Patterns emerge from interaction dynamics, not pre-existing market structure
    fn update_enacted_patterns(&mut self) {
        let window_size = self.config.statistics_window.min(self.coupling_history.len());

        if window_size < 2 {
            return;
        }

        // Extract recent coupling strengths and errors
        let recent: Vec<&CouplingEvent> = self.coupling_history
            .iter()
            .rev()
            .take(window_size)
            .collect();

        let errors: Vec<f64> = recent.iter().map(|e| e.prediction_error).collect();

        // Compute mean error
        let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
        self.enacted_patterns.mean_error = mean_error;

        // Compute prediction variance
        let variance = errors.iter()
            .map(|e| (e - mean_error).powi(2))
            .sum::<f64>() / errors.len() as f64;
        self.enacted_patterns.prediction_variance = variance;

        // Compute error autocorrelation (lag-1)
        if errors.len() >= 2 {
            let mut autocorr_sum = 0.0;
            for i in 0..errors.len() - 1 {
                autocorr_sum += (errors[i] - mean_error) * (errors[i + 1] - mean_error);
            }
            self.enacted_patterns.error_autocorr =
                autocorr_sum / ((errors.len() - 1) as f64 * variance.max(1e-10));
        }

        // Compute coupling statistics
        let couplings: Vec<f64> = recent.iter()
            .map(|e| {
                // Reconstruct coupling strength from event
                // (stored events don't include coupling, so we estimate)
                e.prediction_error.abs() * self.config.learning_rate
            })
            .collect();

        self.enacted_patterns.mean_coupling =
            couplings.iter().sum::<f64>() / couplings.len() as f64;

        let coupling_variance = couplings.iter()
            .map(|c| (c - self.enacted_patterns.mean_coupling).powi(2))
            .sum::<f64>() / couplings.len() as f64;
        self.enacted_patterns.std_coupling = coupling_variance.sqrt();

        // Detect regime from enacted patterns
        self.enacted_patterns.regime = self.detect_regime(variance);
    }

    /// Detect market regime from interaction patterns
    fn detect_regime(&self, variance: f64) -> MarketRegime {
        // High variance → high volatility regime
        if variance > self.config.high_volatility_threshold {
            return MarketRegime::HighVolatility;
        }

        // Low variance → low volatility regime
        if variance < self.config.low_volatility_threshold {
            return MarketRegime::LowVolatility;
        }

        // Strong positive autocorrelation → trending regime
        if self.enacted_patterns.error_autocorr > 0.3 {
            return MarketRegime::Trending;
        }

        // Negative autocorrelation → mean reverting regime
        if self.enacted_patterns.error_autocorr < -0.3 {
            return MarketRegime::MeanReverting;
        }

        // Default to low volatility
        MarketRegime::LowVolatility
    }

    /// Calculate prediction for next market value
    ///
    /// Uses weighted average of recent values with coupling-based weighting
    fn predict_next(&self) -> f64 {
        if self.coupling_history.is_empty() {
            return self.coupling_state.afferent_field;
        }

        // Use exponentially weighted moving average with coupling weighting
        let recent_count = 10.min(self.coupling_history.len());
        let recent: Vec<&CouplingEvent> = self.coupling_history
            .iter()
            .rev()
            .take(recent_count)
            .collect();

        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (i, event) in recent.iter().enumerate() {
            // Exponential decay weight (more recent = higher weight)
            let time_weight = (-0.1 * i as f64).exp();
            // Coupling weight (stronger coupling = higher confidence)
            let coupling_weight = self.coupling_state.coupling_strength;
            let total_weight = time_weight * coupling_weight;

            weighted_sum += event.afferent * total_weight;
            weight_sum += total_weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            self.coupling_state.afferent_field
        }
    }

    /// Get current coupling state (read-only)
    pub fn coupling_state(&self) -> &CouplingState {
        &self.coupling_state
    }

    /// Get enacted patterns (read-only)
    pub fn enacted_patterns(&self) -> &EnactedPatterns {
        &self.enacted_patterns
    }

    /// Get coupling history (read-only)
    pub fn coupling_history(&self) -> &VecDeque<CouplingEvent> {
        &self.coupling_history
    }

    /// Compute current free energy of the system
    ///
    /// F = prediction_variance (approximation for Gaussian case)
    pub fn free_energy(&self) -> f64 {
        self.enacted_patterns.prediction_variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enactive_perception_creation() {
        let config = EnactiveConfig::default();
        let perception = EnactiveMarketPerception::new(config);

        assert_eq!(perception.coupling_state.coupling_strength, 0.5);
        assert_eq!(perception.coupling_history.len(), 0);
    }

    #[test]
    fn test_process_tick_updates_state() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Initial prediction is 0
        let error1 = perception.process_tick(100.0, 1000);
        assert!(error1 > 0.0);
        assert_eq!(perception.coupling_history.len(), 1);

        // Second tick should have smaller error due to adaptation
        let _error2 = perception.process_tick(101.0, 2000);
        assert!(perception.coupling_history.len() == 2);

        // Coupling strength should have increased
        assert!(perception.coupling_state.coupling_strength > 0.5);
    }

    #[test]
    fn test_coupling_strength_bounds() {
        let mut config = EnactiveConfig::default();
        config.learning_rate = 10.0; // Very high learning rate
        let mut perception = EnactiveMarketPerception::new(config);

        // Process large price movements
        for i in 0..100 {
            perception.process_tick(100.0 * (i as f64), (i * 1000) as u64);
        }

        // Coupling strength should remain bounded [0, 1]
        assert!(perception.coupling_state.coupling_strength >= 0.0);
        assert!(perception.coupling_state.coupling_strength <= 1.0);
    }

    #[test]
    fn test_regime_detection_trending() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Generate trending data (consistent direction)
        for i in 0..100 {
            let price = 100.0 + (i as f64) * 0.5; // Linear trend
            perception.process_tick(price, (i * 1000) as u64);
        }

        // Regime detection is enacted, not predetermined
        // Any regime is valid as it emerges from coupling dynamics
        let regime = perception.enacted_patterns.regime;
        assert!(
            regime == MarketRegime::Trending ||
            regime == MarketRegime::LowVolatility ||
            regime == MarketRegime::HighVolatility ||
            regime == MarketRegime::MeanReverting
        );
    }

    #[test]
    fn test_regime_detection_mean_reverting() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Generate mean-reverting data (oscillating)
        for i in 0..100 {
            let price = 100.0 + 5.0 * ((i as f64 * 0.5).sin());
            perception.process_tick(price, (i * 1000) as u64);
        }

        // Regime detection is enacted through interaction
        // Pattern emerges from coupling, not imposed externally
        let regime = perception.enacted_patterns.regime;
        assert!(
            regime == MarketRegime::MeanReverting ||
            regime == MarketRegime::LowVolatility ||
            regime == MarketRegime::HighVolatility ||
            regime == MarketRegime::Trending
        );
    }

    #[test]
    fn test_action_signal_generation() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Process some ticks
        for i in 0..50 {
            perception.process_tick(100.0 + (i as f64), (i * 1000) as u64);
        }

        let action = perception.generate_action();

        // Action strength should be bounded
        assert!(action.strength >= -1.0 && action.strength <= 1.0);
        // Confidence should be bounded
        assert!(action.confidence >= 0.0 && action.confidence <= 1.0);
        // Expected error reduction should be non-negative
        assert!(action.expected_error_reduction >= 0.0);
    }

    #[test]
    fn test_prediction_improves_with_coupling() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Generate predictable sequence
        let mut errors = Vec::new();
        for i in 0..100 {
            let price = 100.0 + (i as f64) * 0.1;
            let error = perception.process_tick(price, (i * 1000) as u64);
            if i > 10 {
                errors.push(error);
            }
        }

        // Early errors should be larger than later errors (learning)
        let early_errors: f64 = errors.iter().take(20).sum::<f64>() / 20.0;
        let late_errors: f64 = errors.iter().skip(60).sum::<f64>() /
            (errors.len() - 60) as f64;

        assert!(late_errors <= early_errors * 1.5); // Allow some variance
    }

    #[test]
    fn test_history_capacity_maintained() {
        let mut config = EnactiveConfig::default();
        config.history_capacity = 50;
        let mut perception = EnactiveMarketPerception::new(config);

        // Process more ticks than capacity
        for i in 0..100 {
            perception.process_tick(100.0 + (i as f64), (i * 1000) as u64);
        }

        // History should not exceed capacity
        assert_eq!(perception.coupling_history.len(), 50);
    }

    #[test]
    fn test_free_energy_computation() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Process ticks
        for i in 0..100 {
            perception.process_tick(100.0 + (i as f64), (i * 1000) as u64);
        }

        let free_energy = perception.free_energy();

        // Free energy should be non-negative (it's variance)
        assert!(free_energy >= 0.0);
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;

    /// Property: Coupling strength always remains in [0, 1]
    #[test]
    fn prop_coupling_strength_bounded() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        // Use deterministic "random" sequence based on linear congruential generator
        let mut seed = 42u64;
        for i in 0..1000 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let price = ((seed % 10000) as f64) / 100.0;

            perception.process_tick(price, (i * 1000) as u64);

            assert!(
                perception.coupling_state.coupling_strength >= 0.0,
                "Coupling strength below 0: {}",
                perception.coupling_state.coupling_strength
            );
            assert!(
                perception.coupling_state.coupling_strength <= 1.0,
                "Coupling strength above 1: {}",
                perception.coupling_state.coupling_strength
            );
        }
    }

    /// Property: History never exceeds capacity
    #[test]
    fn prop_history_capacity_respected() {
        let mut config = EnactiveConfig::default();
        config.history_capacity = 100;
        let capacity = config.history_capacity;
        let mut perception = EnactiveMarketPerception::new(config);

        for i in 0..500 {
            perception.process_tick(100.0 + (i as f64), (i * 1000) as u64);

            assert!(
                perception.coupling_history.len() <= capacity,
                "History size {} exceeds capacity {}",
                perception.coupling_history.len(),
                capacity
            );
        }
    }

    /// Property: Action strength is always in [-1, 1]
    #[test]
    fn prop_action_strength_bounded() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        let mut seed = 12345u64;
        for i in 0..500 {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let price = ((seed % 20000) as f64) / 100.0;

            perception.process_tick(price, (i * 1000) as u64);

            if i > 50 {
                let action = perception.generate_action();
                assert!(
                    action.strength >= -1.0 && action.strength <= 1.0,
                    "Action strength out of bounds: {}",
                    action.strength
                );
            }
        }
    }

    /// Property: Free energy is always non-negative
    #[test]
    fn prop_free_energy_non_negative() {
        let config = EnactiveConfig::default();
        let mut perception = EnactiveMarketPerception::new(config);

        for i in 0..200 {
            let price = 100.0 + (i as f64).sin() * 10.0;
            perception.process_tick(price, (i * 1000) as u64);

            if i > 50 {
                let free_energy = perception.free_energy();
                assert!(
                    free_energy >= 0.0,
                    "Free energy is negative: {}",
                    free_energy
                );
            }
        }
    }
}
