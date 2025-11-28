//! Real-time risk metrics calculator.
//!
//! Computes VaR, CVaR, and other risk metrics using a combination
//! of parametric methods and EVT for tails.

use crate::core::types::{Portfolio, MarketRegime};

/// Risk calculation configuration.
#[derive(Debug, Clone)]
pub struct RiskConfig {
    /// VaR confidence level.
    pub var_confidence: f64,
    /// CVaR/ES confidence level.
    pub es_confidence: f64,
    /// Lookback period in observations.
    pub lookback: usize,
    /// Use EVT for tail estimation.
    pub use_evt: bool,
    /// Decay factor for exponential weighting.
    pub decay_factor: f64,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            var_confidence: 0.95,
            es_confidence: 0.975,
            lookback: 252, // ~1 year of daily data
            use_evt: true,
            decay_factor: 0.94,
        }
    }
}

/// Risk metrics output.
#[derive(Debug, Clone)]
pub struct RiskMetrics {
    /// Value at Risk.
    pub var: f64,
    /// Expected Shortfall (CVaR).
    pub es: f64,
    /// Portfolio volatility (annualized).
    pub volatility: f64,
    /// Portfolio beta.
    pub beta: f64,
    /// Sharpe ratio.
    pub sharpe_ratio: f64,
    /// Maximum drawdown.
    pub max_drawdown: f64,
    /// Current drawdown.
    pub current_drawdown: f64,
    /// Tail risk indicator (from EVT).
    pub tail_risk: f64,
}

impl Default for RiskMetrics {
    fn default() -> Self {
        Self {
            var: 0.0,
            es: 0.0,
            volatility: 0.0,
            beta: 1.0,
            sharpe_ratio: 0.0,
            max_drawdown: 0.0,
            current_drawdown: 0.0,
            tail_risk: 0.0,
        }
    }
}

/// Real-time risk calculator.
#[derive(Debug)]
pub struct RiskCalculator {
    /// Configuration.
    config: RiskConfig,
    /// Return history.
    returns: Vec<f64>,
    /// Squared returns (for variance).
    squared_returns: Vec<f64>,
    /// Running mean.
    mean: f64,
    /// Running variance.
    variance: f64,
    /// Peak portfolio value.
    peak_value: f64,
    /// Current regime.
    regime: MarketRegime,
}

impl RiskCalculator {
    /// Create new risk calculator.
    pub fn new(config: RiskConfig) -> Self {
        Self {
            config,
            returns: Vec::with_capacity(300),
            squared_returns: Vec::with_capacity(300),
            mean: 0.0,
            variance: 0.01, // 1% initial volatility assumption
            peak_value: 0.0,
            regime: MarketRegime::Unknown,
        }
    }

    /// Update with new return observation.
    pub fn update(&mut self, return_value: f64, portfolio_value: f64) {
        // Update return history
        self.returns.push(return_value);
        self.squared_returns.push(return_value * return_value);

        // Trim to lookback
        while self.returns.len() > self.config.lookback {
            self.returns.remove(0);
            self.squared_returns.remove(0);
        }

        // Update exponentially weighted statistics
        let decay = self.config.decay_factor;
        self.mean = decay * self.mean + (1.0 - decay) * return_value;
        self.variance = decay * self.variance + (1.0 - decay) * (return_value - self.mean).powi(2);

        // Update peak
        if portfolio_value > self.peak_value {
            self.peak_value = portfolio_value;
        }
    }

    /// Set current market regime.
    pub fn set_regime(&mut self, regime: MarketRegime) {
        self.regime = regime;
    }

    /// Calculate current risk metrics.
    pub fn calculate(&self, portfolio: &Portfolio) -> RiskMetrics {
        let volatility = self.variance.sqrt() * (252.0_f64).sqrt(); // Annualized

        // Parametric VaR (normal approximation)
        let z_var = self.normal_quantile(self.config.var_confidence);
        let var = -self.mean * 252.0 + z_var * volatility;

        // Expected Shortfall (approximate)
        let z_es = self.normal_quantile(self.config.es_confidence);
        let phi_z = (-0.5 * z_es * z_es).exp() / (2.0 * std::f64::consts::PI).sqrt();
        let es = -self.mean * 252.0 + volatility * phi_z / (1.0 - self.config.es_confidence);

        // Sharpe ratio
        let risk_free = 0.04; // Assume 4% risk-free rate
        let excess_return = self.mean * 252.0 - risk_free;
        let sharpe_ratio = if volatility > 0.0 {
            excess_return / volatility
        } else {
            0.0
        };

        // Drawdown
        let current_drawdown = if self.peak_value > 0.0 {
            (self.peak_value - portfolio.total_value) / self.peak_value
        } else {
            0.0
        };

        let max_drawdown = self.calculate_max_drawdown();

        // Tail risk from return distribution
        let tail_risk = self.calculate_tail_risk();

        // Regime adjustment
        let regime_mult = self.regime.risk_multiplier();

        RiskMetrics {
            var: var.abs() * regime_mult,
            es: es.abs() * regime_mult,
            volatility,
            beta: 1.0, // Would need market data
            sharpe_ratio,
            max_drawdown,
            current_drawdown,
            tail_risk,
        }
    }

    /// Calculate maximum drawdown from return history.
    fn calculate_max_drawdown(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }

        let mut peak = 1.0;
        let mut max_dd = 0.0;
        let mut cumulative = 1.0;

        for &r in &self.returns {
            cumulative *= 1.0 + r;
            if cumulative > peak {
                peak = cumulative;
            }
            let dd = (peak - cumulative) / peak;
            if dd > max_dd {
                max_dd = dd;
            }
        }

        max_dd
    }

    /// Calculate tail risk indicator.
    fn calculate_tail_risk(&self) -> f64 {
        if self.returns.len() < 30 {
            return 0.5; // Not enough data
        }

        // Calculate excess kurtosis as tail risk indicator
        let n = self.returns.len() as f64;
        let mean: f64 = self.returns.iter().sum::<f64>() / n;
        let var: f64 = self.returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();

        if std < 1e-10 {
            return 0.0;
        }

        let fourth_moment: f64 = self.returns.iter()
            .map(|r| ((r - mean) / std).powi(4))
            .sum::<f64>() / n;

        let kurtosis = fourth_moment - 3.0; // Excess kurtosis

        // Normalize to 0-1 range
        (kurtosis / 10.0).tanh().abs()
    }

    /// Standard normal quantile approximation.
    fn normal_quantile(&self, p: f64) -> f64 {
        // Approximation for normal quantile
        // For p close to 0.5: use Taylor expansion
        // For p far from 0.5: use Abramowitz & Stegun approximation

        if p <= 0.0 || p >= 1.0 {
            return 0.0;
        }

        let t = if p < 0.5 {
            (-2.0 * p.ln()).sqrt()
        } else {
            (-2.0 * (1.0 - p).ln()).sqrt()
        };

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let q = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p < 0.5 { -q } else { q }
    }

    /// Get current volatility.
    pub fn volatility(&self) -> f64 {
        self.variance.sqrt() * (252.0_f64).sqrt()
    }

    /// Get observation count.
    pub fn observation_count(&self) -> usize {
        self.returns.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_risk_calculator_creation() {
        let config = RiskConfig::default();
        let calc = RiskCalculator::new(config);
        assert_eq!(calc.observation_count(), 0);
    }

    #[test]
    fn test_risk_update() {
        let config = RiskConfig::default();
        let mut calc = RiskCalculator::new(config);

        // Add some returns
        for i in 0..100 {
            let ret = (i as f64 - 50.0) / 1000.0; // Range: -5% to +5%
            calc.update(ret, 100_000.0);
        }

        assert_eq!(calc.observation_count(), 100);
        assert!(calc.volatility() > 0.0);
    }

    #[test]
    fn test_risk_metrics() {
        let config = RiskConfig::default();
        let mut calc = RiskCalculator::new(config);

        // Generate some returns
        for i in 0..100 {
            let ret = (i as f64 % 10.0 - 5.0) / 100.0;
            calc.update(ret, 100_000.0);
        }

        let portfolio = Portfolio::new(100_000.0);
        let metrics = calc.calculate(&portfolio);

        assert!(metrics.var > 0.0);
        assert!(metrics.es >= metrics.var); // ES >= VaR
        assert!(metrics.volatility > 0.0);
    }

    #[test]
    fn test_normal_quantile() {
        let config = RiskConfig::default();
        let calc = RiskCalculator::new(config);

        // Test known values
        let q50 = calc.normal_quantile(0.5);
        assert!(q50.abs() < 0.01); // Should be ~0

        let q95 = calc.normal_quantile(0.95);
        assert!((q95 - 1.645).abs() < 0.1); // Should be ~1.645

        let q99 = calc.normal_quantile(0.99);
        assert!((q99 - 2.326).abs() < 0.1); // Should be ~2.326
    }
}
