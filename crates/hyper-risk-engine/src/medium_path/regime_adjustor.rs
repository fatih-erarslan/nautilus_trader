//! Regime-aware risk adjustment.
//!
//! Adjusts risk limits and position sizes based on detected market regime.

use crate::core::types::MarketRegime;

/// Regime adjustment output.
#[derive(Debug, Clone)]
pub struct RegimeAdjustment {
    /// Risk multiplier (0.0 - 2.0).
    pub risk_multiplier: f64,
    /// Position size multiplier (0.0 - 2.0).
    pub size_multiplier: f64,
    /// Stop loss tightening factor.
    pub stop_tightening: f64,
    /// Suggested VaR limit adjustment.
    pub var_limit_mult: f64,
    /// Should hedge delta?
    pub hedge_delta: bool,
    /// Should reduce beta?
    pub reduce_beta: bool,
    /// Regime description.
    pub regime_description: String,
}

impl Default for RegimeAdjustment {
    fn default() -> Self {
        Self {
            risk_multiplier: 1.0,
            size_multiplier: 1.0,
            stop_tightening: 1.0,
            var_limit_mult: 1.0,
            hedge_delta: false,
            reduce_beta: false,
            regime_description: "Normal".to_string(),
        }
    }
}

/// Regime adjustment configuration.
#[derive(Debug, Clone)]
pub struct RegimeConfig {
    /// Crisis risk multiplier.
    pub crisis_risk_mult: f64,
    /// Crisis size multiplier.
    pub crisis_size_mult: f64,
    /// Bull market risk multiplier.
    pub bull_risk_mult: f64,
    /// Bull market size multiplier.
    pub bull_size_mult: f64,
    /// High volatility risk multiplier.
    pub high_vol_risk_mult: f64,
    /// Enable automatic hedging in crisis.
    pub auto_hedge_crisis: bool,
}

impl Default for RegimeConfig {
    fn default() -> Self {
        Self {
            crisis_risk_mult: 0.2,
            crisis_size_mult: 0.1,
            bull_risk_mult: 1.2,
            bull_size_mult: 1.1,
            high_vol_risk_mult: 0.6,
            auto_hedge_crisis: true,
        }
    }
}

/// Regime adjustor.
#[derive(Debug)]
pub struct RegimeAdjustor {
    /// Configuration.
    config: RegimeConfig,
    /// Current regime.
    current_regime: MarketRegime,
    /// Regime confidence.
    confidence: f64,
    /// Last adjustment.
    last_adjustment: RegimeAdjustment,
}

impl RegimeAdjustor {
    /// Create new regime adjustor.
    pub fn new(config: RegimeConfig) -> Self {
        Self {
            config,
            current_regime: MarketRegime::Unknown,
            confidence: 0.0,
            last_adjustment: RegimeAdjustment::default(),
        }
    }

    /// Update with new regime detection.
    pub fn update(&mut self, regime: MarketRegime, confidence: f64) {
        self.current_regime = regime;
        self.confidence = confidence;
        self.last_adjustment = self.calculate_adjustment();
    }

    /// Calculate adjustment for current regime.
    fn calculate_adjustment(&self) -> RegimeAdjustment {
        // Weight adjustment by confidence
        let conf = self.confidence.clamp(0.0, 1.0);

        let (base_risk, base_size, stop_tight, var_mult, hedge, reduce_beta, desc) =
            match self.current_regime {
                MarketRegime::BullTrending => (
                    self.config.bull_risk_mult,
                    self.config.bull_size_mult,
                    0.9,  // Looser stops
                    1.1,  // Higher VaR limit
                    false,
                    false,
                    "Bull Trend - Favorable conditions",
                ),
                MarketRegime::BearTrending => (
                    0.7,
                    0.6,
                    1.2,  // Tighter stops
                    0.8,  // Lower VaR limit
                    true,
                    true,
                    "Bear Trend - Defensive posture",
                ),
                MarketRegime::SidewaysLow => (
                    0.9,
                    0.9,
                    1.0,
                    1.0,
                    false,
                    false,
                    "Sideways Low Vol - Range trading",
                ),
                MarketRegime::SidewaysHigh => (
                    self.config.high_vol_risk_mult,
                    0.5,
                    1.3,  // Tighter stops
                    0.7,  // Lower VaR limit
                    true,
                    false,
                    "Sideways High Vol - Choppy conditions",
                ),
                MarketRegime::Crisis => (
                    self.config.crisis_risk_mult,
                    self.config.crisis_size_mult,
                    2.0,  // Very tight stops
                    0.3,  // Minimal VaR limit
                    self.config.auto_hedge_crisis,
                    true,
                    "CRISIS - Maximum protection mode",
                ),
                MarketRegime::Recovery => (
                    0.8,
                    0.7,
                    1.1,
                    0.9,
                    false,
                    false,
                    "Recovery - Cautious optimism",
                ),
                MarketRegime::Unknown => (
                    0.5,
                    0.5,
                    1.5,
                    0.5,
                    false,
                    true,
                    "Unknown - Conservative default",
                ),
            };

        // Blend with neutral based on confidence
        let blend = |val: f64| conf * val + (1.0 - conf) * 1.0;

        RegimeAdjustment {
            risk_multiplier: blend(base_risk),
            size_multiplier: blend(base_size),
            stop_tightening: stop_tight,
            var_limit_mult: blend(var_mult),
            hedge_delta: hedge && conf > 0.7,
            reduce_beta: reduce_beta && conf > 0.6,
            regime_description: desc.to_string(),
        }
    }

    /// Get current adjustment.
    pub fn get_adjustment(&self) -> &RegimeAdjustment {
        &self.last_adjustment
    }

    /// Get current regime.
    pub fn current_regime(&self) -> MarketRegime {
        self.current_regime
    }

    /// Get regime confidence.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Apply adjustment to a raw position size.
    pub fn adjust_size(&self, raw_size: f64) -> f64 {
        raw_size * self.last_adjustment.size_multiplier
    }

    /// Apply adjustment to a VaR limit.
    pub fn adjust_var_limit(&self, raw_limit: f64) -> f64 {
        raw_limit * self.last_adjustment.var_limit_mult
    }

    /// Check if should hedge.
    pub fn should_hedge(&self) -> bool {
        self.last_adjustment.hedge_delta
    }

    /// Check if should reduce beta.
    pub fn should_reduce_beta(&self) -> bool {
        self.last_adjustment.reduce_beta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_regime_adjustor_creation() {
        let config = RegimeConfig::default();
        let adjustor = RegimeAdjustor::new(config);
        assert_eq!(adjustor.current_regime(), MarketRegime::Unknown);
    }

    #[test]
    fn test_crisis_adjustment() {
        let config = RegimeConfig::default();
        let mut adjustor = RegimeAdjustor::new(config);

        adjustor.update(MarketRegime::Crisis, 0.9);

        let adj = adjustor.get_adjustment();
        assert!(adj.risk_multiplier < 0.3);
        assert!(adj.size_multiplier < 0.2);
        assert!(adj.hedge_delta);
        assert!(adj.reduce_beta);
    }

    #[test]
    fn test_bull_adjustment() {
        let config = RegimeConfig::default();
        let mut adjustor = RegimeAdjustor::new(config);

        adjustor.update(MarketRegime::BullTrending, 0.8);

        let adj = adjustor.get_adjustment();
        assert!(adj.risk_multiplier > 1.0);
        assert!(adj.size_multiplier >= 1.0);
        assert!(!adj.hedge_delta);
    }

    #[test]
    fn test_confidence_blending() {
        let config = RegimeConfig::default();
        let mut adjustor = RegimeAdjustor::new(config);

        // Low confidence should blend toward neutral
        adjustor.update(MarketRegime::Crisis, 0.2);
        let low_conf = adjustor.get_adjustment().risk_multiplier;

        adjustor.update(MarketRegime::Crisis, 0.9);
        let high_conf = adjustor.get_adjustment().risk_multiplier;

        // Low confidence should be closer to 1.0
        assert!((low_conf - 1.0).abs() < (high_conf - 1.0).abs());
    }

    #[test]
    fn test_size_adjustment() {
        let config = RegimeConfig::default();
        let mut adjustor = RegimeAdjustor::new(config);

        adjustor.update(MarketRegime::Crisis, 0.9);

        let raw = 1000.0;
        let adjusted = adjustor.adjust_size(raw);

        assert!(adjusted < raw);
    }
}
