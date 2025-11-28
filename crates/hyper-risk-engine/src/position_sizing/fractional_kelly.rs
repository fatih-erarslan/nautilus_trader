//! Fractional Kelly for practical position sizing.
//!
//! Full Kelly is optimal for growth but volatile. In practice:
//! - Half Kelly (0.5) achieves ~75% of growth with ~50% variance
//! - Quarter Kelly (0.25) is more conservative
//!
//! ## Trade-offs
//!
//! | Fraction | Growth Rate | Drawdown | Variance |
//! |----------|-------------|----------|----------|
//! | 1.0 (Full) | 100% | High | High |
//! | 0.5 (Half) | 75% | Medium | Medium |
//! | 0.25 (Quarter) | 50% | Low | Low |
//!
//! ## Scientific References
//! - MacLean, Ziemba & Blazenko (1992): "Growth Versus Security in Dynamic Investment Analysis"

use serde::{Deserialize, Serialize};
use super::kelly::{KellyCriterion, KellyConfig, KellyResult};

/// Risk tolerance levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskTolerance {
    /// Very conservative (0.1x Kelly).
    VeryConservative,
    /// Conservative (0.25x Kelly).
    Conservative,
    /// Moderate (0.5x Kelly).
    Moderate,
    /// Aggressive (0.75x Kelly).
    Aggressive,
    /// Full Kelly (1.0x).
    Full,
}

impl RiskTolerance {
    /// Get Kelly fraction for this risk level.
    pub fn fraction(&self) -> f64 {
        match self {
            Self::VeryConservative => 0.10,
            Self::Conservative => 0.25,
            Self::Moderate => 0.50,
            Self::Aggressive => 0.75,
            Self::Full => 1.00,
        }
    }

    /// Get approximate growth rate relative to full Kelly.
    pub fn relative_growth(&self) -> f64 {
        let f = self.fraction();
        // g(f) / g(1) ≈ 2f - f²
        2.0 * f - f * f
    }

    /// Get approximate variance relative to full Kelly.
    pub fn relative_variance(&self) -> f64 {
        let f = self.fraction();
        // Variance proportional to f²
        f * f
    }
}

/// Fractional Kelly position sizer.
#[derive(Debug)]
pub struct FractionalKelly {
    /// Inner Kelly calculator.
    kelly: KellyCriterion,
    /// Risk tolerance level.
    risk_tolerance: RiskTolerance,
    /// Custom fraction override.
    custom_fraction: Option<f64>,
}

impl FractionalKelly {
    /// Create with risk tolerance level.
    pub fn new(config: KellyConfig, risk_tolerance: RiskTolerance) -> Self {
        Self {
            kelly: KellyCriterion::new(config),
            risk_tolerance,
            custom_fraction: None,
        }
    }

    /// Create with custom Kelly fraction.
    pub fn with_fraction(config: KellyConfig, fraction: f64) -> Self {
        assert!(fraction > 0.0 && fraction <= 1.0, "Fraction must be in (0, 1]");
        Self {
            kelly: KellyCriterion::new(config),
            risk_tolerance: RiskTolerance::Moderate, // Unused
            custom_fraction: Some(fraction),
        }
    }

    /// Get the Kelly fraction being used.
    pub fn fraction(&self) -> f64 {
        self.custom_fraction.unwrap_or_else(|| self.risk_tolerance.fraction())
    }

    /// Calculate fractional Kelly position.
    pub fn calculate_simple(&self, win_prob: f64, win_loss_ratio: f64) -> KellyResult {
        let mut result = self.kelly.calculate_simple(win_prob, win_loss_ratio);
        let frac = self.fraction();

        // Scale by fraction
        result.optimal_fraction *= frac;
        result.constrained_fraction *= frac;

        // Adjust growth rate for fractional Kelly
        // g(f*λ) ≈ g(f*) * (2λ - λ²)
        result.expected_growth_rate *= 2.0 * frac - frac * frac;

        result
    }

    /// Calculate from return distribution.
    pub fn calculate_from_returns(&self, expected_return: f64, variance: f64) -> KellyResult {
        let mut result = self.kelly.calculate_from_returns(expected_return, variance);
        let frac = self.fraction();

        result.optimal_fraction *= frac;
        result.constrained_fraction *= frac;
        result.expected_growth_rate *= 2.0 * frac - frac * frac;

        result
    }

    /// Calculate portfolio weights.
    pub fn calculate_portfolio(
        &self,
        expected_returns: &[f64],
        covariance: &[f64],
        n: usize,
    ) -> Vec<f64> {
        let weights = self.kelly.calculate_portfolio(expected_returns, covariance, n);
        let frac = self.fraction();

        weights.iter().map(|w| w * frac).collect()
    }

    /// Get risk tolerance.
    pub fn risk_tolerance(&self) -> RiskTolerance {
        self.risk_tolerance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_tolerance_fractions() {
        assert!((RiskTolerance::Moderate.fraction() - 0.5).abs() < 0.01);
        assert!((RiskTolerance::Conservative.fraction() - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_relative_growth() {
        // Half Kelly should give ~75% growth
        let moderate_growth = RiskTolerance::Moderate.relative_growth();
        assert!((moderate_growth - 0.75).abs() < 0.01);

        // Quarter Kelly should give ~44% growth
        let conservative_growth = RiskTolerance::Conservative.relative_growth();
        assert!(conservative_growth > 0.4 && conservative_growth < 0.5);
    }

    #[test]
    fn test_fractional_kelly_scaling() {
        let config = KellyConfig::default();
        let full = KellyCriterion::new(config.clone());
        let half = FractionalKelly::new(config, RiskTolerance::Moderate);

        let full_result = full.calculate_simple(0.6, 2.0);
        let half_result = half.calculate_simple(0.6, 2.0);

        // Half Kelly should be approximately half of full
        assert!((half_result.optimal_fraction - full_result.optimal_fraction * 0.5).abs() < 0.01);
    }

    #[test]
    fn test_custom_fraction() {
        let config = KellyConfig::default();
        let sizer = FractionalKelly::with_fraction(config, 0.33);

        assert!((sizer.fraction() - 0.33).abs() < 0.01);
    }
}
