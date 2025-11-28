//! Kelly Criterion for optimal bet sizing.
//!
//! The Kelly Criterion maximizes long-term geometric growth rate.
//!
//! ## Formula
//!
//! For a single asset with win probability p and odds b:
//! f* = (p*b - (1-p)) / b = (p*b - q) / b
//!
//! where:
//! - f* = optimal fraction of capital
//! - p = probability of winning
//! - q = 1 - p = probability of losing
//! - b = win/loss ratio (odds)
//!
//! ## Multi-Asset Extension
//!
//! For multiple correlated assets, use the matrix form:
//! f* = Σ^(-1) * μ / γ
//!
//! where:
//! - Σ = covariance matrix
//! - μ = expected return vector
//! - γ = risk aversion parameter

use serde::{Deserialize, Serialize};

/// Kelly criterion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KellyConfig {
    /// Maximum leverage allowed.
    pub max_leverage: f64,
    /// Minimum position size (fraction).
    pub min_position: f64,
    /// Maximum single position size.
    pub max_position: f64,
    /// Risk-free rate for excess return calculation.
    pub risk_free_rate: f64,
    /// Use geometric or arithmetic mean for returns.
    pub use_geometric_mean: bool,
}

impl Default for KellyConfig {
    fn default() -> Self {
        Self {
            max_leverage: 2.0,
            min_position: 0.01,  // 1% minimum
            max_position: 0.25,  // 25% maximum
            risk_free_rate: 0.04, // 4% annual
            use_geometric_mean: true,
        }
    }
}

/// Kelly calculation result.
#[derive(Debug, Clone)]
pub struct KellyResult {
    /// Optimal fraction (raw Kelly).
    pub optimal_fraction: f64,
    /// Constrained fraction (after limits).
    pub constrained_fraction: f64,
    /// Expected growth rate at optimal.
    pub expected_growth_rate: f64,
    /// Edge (expected return / risk).
    pub edge: f64,
    /// Sharpe ratio estimate.
    pub sharpe_ratio: f64,
}

/// Kelly Criterion calculator.
#[derive(Debug)]
pub struct KellyCriterion {
    /// Configuration.
    config: KellyConfig,
}

impl KellyCriterion {
    /// Create new Kelly calculator.
    pub fn new(config: KellyConfig) -> Self {
        Self { config }
    }

    /// Calculate optimal position size for simple bet.
    ///
    /// # Arguments
    /// * `win_prob` - Probability of winning (0 < p < 1)
    /// * `win_loss_ratio` - Ratio of win to loss magnitude
    ///
    /// # Returns
    /// Optimal fraction of capital to bet
    pub fn calculate_simple(&self, win_prob: f64, win_loss_ratio: f64) -> KellyResult {
        assert!(win_prob > 0.0 && win_prob < 1.0, "Win probability must be in (0, 1)");
        assert!(win_loss_ratio > 0.0, "Win/loss ratio must be positive");

        let p = win_prob;
        let q = 1.0 - p;
        let b = win_loss_ratio;

        // Kelly formula: f* = (p*b - q) / b
        let optimal = (p * b - q) / b;

        // Edge = expected return = p*b - q
        let edge = p * b - q;

        // Sharpe-like metric
        let sharpe = if b > 0.0 {
            edge / b.sqrt() // Simplified
        } else {
            0.0
        };

        // Expected growth rate: g = p*ln(1 + f*b) + q*ln(1 - f)
        let growth_rate = if optimal > 0.0 && optimal < 1.0 {
            p * (1.0 + optimal * b).ln() + q * (1.0 - optimal).ln()
        } else {
            0.0
        };

        // Apply constraints
        let constrained = self.apply_constraints(optimal);

        KellyResult {
            optimal_fraction: optimal,
            constrained_fraction: constrained,
            expected_growth_rate: growth_rate,
            edge,
            sharpe_ratio: sharpe,
        }
    }

    /// Calculate optimal position from returns distribution.
    ///
    /// # Arguments
    /// * `expected_return` - Expected return (excess over risk-free)
    /// * `variance` - Return variance
    pub fn calculate_from_returns(&self, expected_return: f64, variance: f64) -> KellyResult {
        if variance <= 0.0 {
            return KellyResult {
                optimal_fraction: 0.0,
                constrained_fraction: 0.0,
                expected_growth_rate: 0.0,
                edge: 0.0,
                sharpe_ratio: 0.0,
            };
        }

        // Excess return
        let mu = expected_return - self.config.risk_free_rate;

        // Kelly for continuous returns: f* = μ / σ²
        let optimal = mu / variance;

        let sharpe = mu / variance.sqrt();
        let edge = mu;

        // Approximate growth rate: g ≈ μ*f - σ²*f²/2
        let growth_rate = mu * optimal - variance * optimal * optimal / 2.0;

        let constrained = self.apply_constraints(optimal);

        KellyResult {
            optimal_fraction: optimal,
            constrained_fraction: constrained,
            expected_growth_rate: growth_rate,
            edge,
            sharpe_ratio: sharpe,
        }
    }

    /// Calculate multi-asset Kelly weights.
    ///
    /// Uses the matrix form: f* = Σ^(-1) * μ
    ///
    /// # Arguments
    /// * `expected_returns` - Vector of expected excess returns
    /// * `covariance` - Covariance matrix (flattened row-major)
    /// * `n` - Number of assets
    pub fn calculate_portfolio(
        &self,
        expected_returns: &[f64],
        covariance: &[f64],
        n: usize,
    ) -> Vec<f64> {
        assert_eq!(expected_returns.len(), n);
        assert_eq!(covariance.len(), n * n);

        // Simple implementation using Gauss-Jordan elimination
        // In production, use nalgebra or ndarray for proper linear algebra

        // Create augmented matrix [Σ | μ]
        let mut augmented = vec![0.0; n * (n + 1)];

        for i in 0..n {
            for j in 0..n {
                augmented[i * (n + 1) + j] = covariance[i * n + j];
            }
            augmented[i * (n + 1) + n] = expected_returns[i] - self.config.risk_free_rate;
        }

        // Gauss-Jordan elimination
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if augmented[k * (n + 1) + i].abs() > augmented[max_row * (n + 1) + i].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            for k in 0..(n + 1) {
                let tmp = augmented[i * (n + 1) + k];
                augmented[i * (n + 1) + k] = augmented[max_row * (n + 1) + k];
                augmented[max_row * (n + 1) + k] = tmp;
            }

            // Make diagonal 1
            let diag = augmented[i * (n + 1) + i];
            if diag.abs() < 1e-10 {
                // Singular matrix
                return vec![0.0; n];
            }

            for k in 0..(n + 1) {
                augmented[i * (n + 1) + k] /= diag;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[k * (n + 1) + i];
                    for j in 0..(n + 1) {
                        augmented[k * (n + 1) + j] -= factor * augmented[i * (n + 1) + j];
                    }
                }
            }
        }

        // Extract solution and apply constraints
        let mut weights: Vec<f64> = (0..n)
            .map(|i| augmented[i * (n + 1) + n])
            .collect();

        // Apply per-asset constraints
        for w in weights.iter_mut() {
            *w = self.apply_constraints(*w);
        }

        // Normalize if total leverage exceeded
        let total: f64 = weights.iter().map(|w| w.abs()).sum();
        if total > self.config.max_leverage {
            let scale = self.config.max_leverage / total;
            for w in weights.iter_mut() {
                *w *= scale;
            }
        }

        weights
    }

    /// Apply position constraints.
    fn apply_constraints(&self, fraction: f64) -> f64 {
        let mut f = fraction;

        // No shorting beyond limit
        if f < -self.config.max_position {
            f = -self.config.max_position;
        }

        // No excessive long
        if f > self.config.max_position {
            f = self.config.max_position;
        }

        // Minimum position threshold
        if f.abs() < self.config.min_position {
            f = 0.0;
        }

        // Leverage limit
        if f > self.config.max_leverage {
            f = self.config.max_leverage;
        }

        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kelly_simple() {
        let config = KellyConfig::default();
        let kelly = KellyCriterion::new(config);

        // Fair coin with 2:1 payout
        let result = kelly.calculate_simple(0.5, 2.0);

        // Kelly should be (0.5 * 2 - 0.5) / 2 = 0.25
        assert!((result.optimal_fraction - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_kelly_no_edge() {
        let config = KellyConfig::default();
        let kelly = KellyCriterion::new(config);

        // Fair coin with even payout = no edge
        let result = kelly.calculate_simple(0.5, 1.0);

        // Should not bet (no edge)
        assert!(result.optimal_fraction.abs() < 0.01);
    }

    #[test]
    fn test_kelly_from_returns() {
        let config = KellyConfig {
            risk_free_rate: 0.0,
            ..Default::default()
        };
        let kelly = KellyCriterion::new(config);

        // 10% expected return, 20% volatility (4% variance)
        let result = kelly.calculate_from_returns(0.10, 0.04);

        // f* = 0.10 / 0.04 = 2.5, but constrained
        assert!(result.constrained_fraction <= 0.25); // Max position limit
    }

    #[test]
    fn test_kelly_constraints() {
        let config = KellyConfig {
            max_position: 0.1,
            min_position: 0.02,
            ..Default::default()
        };
        let kelly = KellyCriterion::new(config);

        // Large edge should be capped
        let result = kelly.calculate_simple(0.8, 3.0);
        assert!(result.constrained_fraction <= 0.1);
    }
}
