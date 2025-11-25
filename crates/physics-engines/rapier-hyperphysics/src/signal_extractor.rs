//! Trading signal extraction from physics simulation
//!
//! Analyzes physics simulation results to derive actionable trading signals

use crate::{PhysicsMapping, RapierHyperPhysicsAdapter, Result};
use nalgebra::Vector3;
use rapier3d::prelude::*;
use serde::{Deserialize, Serialize};

/// Trading signal derived from physics
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum TradingSignal {
    StrongBuy,
    Buy,
    Hold,
    Sell,
    StrongSell,
}

/// Signal with confidence and metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignalResult {
    /// Primary trading signal
    pub signal: TradingSignal,

    /// Confidence (0.0 - 1.0)
    pub confidence: f64,

    /// Predicted price movement (%)
    pub price_movement: f64,

    /// Market momentum strength
    pub momentum_strength: f64,

    /// Volatility estimate
    pub volatility: f64,

    /// Regime classification
    pub regime: MarketRegime,
}

/// Market regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketRegime {
    HighVolatility,
    LowVolatility,
    Trending,
    Ranging,
    Breakout,
}

/// Signal extractor
pub struct SignalExtractor {
    /// Momentum threshold for buy/sell
    momentum_threshold: f32,

    /// Energy threshold for volatility classification
    energy_threshold: f32,
}

impl SignalExtractor {
    /// Create a new signal extractor with default thresholds
    pub fn new() -> Self {
        Self {
            momentum_threshold: 0.5,
            energy_threshold: 10.0,
        }
    }

    /// Create extractor with custom thresholds
    pub fn with_thresholds(momentum_threshold: f32, energy_threshold: f32) -> Self {
        Self {
            momentum_threshold,
            energy_threshold,
        }
    }

    /// Extract trading signal from physics simulation
    pub fn extract_signal(
        &self,
        adapter: &RapierHyperPhysicsAdapter,
        mapping: &PhysicsMapping,
    ) -> Result<SignalResult> {
        // Analyze bid-ask dynamics
        let bid_momentum = self.calculate_group_momentum(adapter, &mapping.bid_bodies);
        let ask_momentum = self.calculate_group_momentum(adapter, &mapping.ask_bodies);

        // Net momentum: positive = bullish, negative = bearish
        let net_momentum = bid_momentum.y - ask_momentum.y;

        // Calculate total system energy (volatility proxy)
        let total_energy = self.calculate_total_energy(adapter);

        // Determine signal based on momentum
        let signal = self.classify_signal(net_momentum);

        // Calculate confidence based on momentum magnitude and energy
        let confidence = self.calculate_confidence(net_momentum, total_energy);

        // Estimate price movement
        let price_movement = self.estimate_price_movement(net_momentum);

        // Classify market regime
        let regime = self.classify_regime(total_energy, net_momentum);

        // Momentum strength
        let momentum_strength = net_momentum.abs() as f64;

        // Volatility estimate
        let volatility = (total_energy / self.energy_threshold) as f64;

        Ok(SignalResult {
            signal,
            confidence,
            price_movement,
            momentum_strength,
            volatility,
            regime,
        })
    }

    /// Calculate momentum for a group of rigid bodies
    fn calculate_group_momentum(
        &self,
        adapter: &RapierHyperPhysicsAdapter,
        handles: &[RigidBodyHandle],
    ) -> Vector3<f32> {
        let mut total_momentum = Vector3::zeros();

        for handle in handles {
            if let Some(rb) = adapter.rigid_bodies().get(*handle) {
                if rb.is_dynamic() {
                    let velocity = rb.linvel();
                    let mass = rb.mass();
                    total_momentum += velocity * mass;
                }
            }
        }

        total_momentum
    }

    /// Calculate total kinetic energy
    fn calculate_total_energy(&self, adapter: &RapierHyperPhysicsAdapter) -> f32 {
        let mut total_energy = 0.0;

        for (_handle, rb) in adapter.rigid_bodies().iter() {
            if rb.is_dynamic() {
                let velocity = rb.linvel();
                let mass = rb.mass();
                total_energy += 0.5 * mass * velocity.norm_squared();
            }
        }

        total_energy
    }

    /// Classify signal based on net momentum
    fn classify_signal(&self, net_momentum: f32) -> TradingSignal {
        if net_momentum > self.momentum_threshold * 2.0 {
            TradingSignal::StrongBuy
        } else if net_momentum > self.momentum_threshold {
            TradingSignal::Buy
        } else if net_momentum < -self.momentum_threshold * 2.0 {
            TradingSignal::StrongSell
        } else if net_momentum < -self.momentum_threshold {
            TradingSignal::Sell
        } else {
            TradingSignal::Hold
        }
    }

    /// Calculate confidence based on momentum and energy
    fn calculate_confidence(&self, net_momentum: f32, total_energy: f32) -> f64 {
        let momentum_component = (net_momentum.abs() / (self.momentum_threshold * 3.0)).min(1.0);
        let energy_component = (total_energy / (self.energy_threshold * 2.0)).min(1.0);

        // High energy with high momentum = high confidence
        // Low energy with low momentum = low confidence
        let confidence = (momentum_component + energy_component) / 2.0;

        confidence.clamp(0.0, 1.0) as f64
    }

    /// Estimate price movement percentage
    fn estimate_price_movement(&self, net_momentum: f32) -> f64 {
        // Simple linear mapping: momentum -> price movement %
        let movement = (net_momentum / self.momentum_threshold) * 0.5; // 0.5% per threshold unit
        movement.clamp(-5.0, 5.0) as f64 // Cap at ±5%
    }

    /// Classify market regime
    fn classify_regime(&self, total_energy: f32, net_momentum: f32) -> MarketRegime {
        let high_energy = total_energy > self.energy_threshold;
        let strong_momentum = net_momentum.abs() > self.momentum_threshold * 1.5;

        match (high_energy, strong_momentum) {
            (true, true) => MarketRegime::Breakout,
            (true, false) => MarketRegime::HighVolatility,
            (false, true) => MarketRegime::Trending,
            (false, false) => {
                if total_energy < self.energy_threshold * 0.3 {
                    MarketRegime::LowVolatility
                } else {
                    MarketRegime::Ranging
                }
            }
        }
    }
}

impl Default for SignalExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signal_extractor_creation() {
        let extractor = SignalExtractor::new();
        assert_eq!(extractor.momentum_threshold, 0.5);
        assert_eq!(extractor.energy_threshold, 10.0);
    }

    #[test]
    fn test_signal_classification() {
        let extractor = SignalExtractor::new();

        assert_eq!(extractor.classify_signal(0.0), TradingSignal::Hold);
        assert_eq!(extractor.classify_signal(0.6), TradingSignal::Buy);
        assert_eq!(extractor.classify_signal(1.5), TradingSignal::StrongBuy);
        assert_eq!(extractor.classify_signal(-0.6), TradingSignal::Sell);
        assert_eq!(extractor.classify_signal(-1.5), TradingSignal::StrongSell);
    }

    #[test]
    fn test_regime_classification() {
        let extractor = SignalExtractor::new();

        // High energy, strong momentum = Breakout
        assert_eq!(extractor.classify_regime(15.0, 1.0), MarketRegime::Breakout);

        // High energy, weak momentum = High volatility
        assert_eq!(
            extractor.classify_regime(15.0, 0.2),
            MarketRegime::HighVolatility
        );

        // Low energy, strong momentum = Trending
        assert_eq!(extractor.classify_regime(5.0, 1.0), MarketRegime::Trending);

        // Low energy, weak momentum = Low volatility or Ranging
        assert_eq!(
            extractor.classify_regime(2.0, 0.2),
            MarketRegime::LowVolatility
        );
    }

    #[test]
    fn test_confidence_calculation() {
        let extractor = SignalExtractor::new();

        // High momentum, high energy = high confidence
        let confidence1 = extractor.calculate_confidence(1.5, 20.0);
        assert!(confidence1 > 0.7);

        // Low momentum, low energy = low confidence
        let confidence2 = extractor.calculate_confidence(0.1, 2.0);
        assert!(confidence2 < 0.3);
    }
}
