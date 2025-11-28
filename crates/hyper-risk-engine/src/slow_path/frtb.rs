//! FRTB (Fundamental Review of the Trading Book) calculations.
//!
//! Implements Basel III/IV Expected Shortfall methodology.
//!
//! ## FRTB Key Requirements
//!
//! - 97.5% ES instead of 99% VaR
//! - Liquidity horizons (10, 20, 40, 60, 120 days)
//! - Stressed ES calibration
//! - P&L attribution test

use crate::core::types::Portfolio;

/// FRTB configuration.
#[derive(Debug, Clone)]
pub struct FRTBConfig {
    /// ES confidence level (97.5% per Basel).
    pub es_confidence: f64,
    /// Use stressed calibration.
    pub use_stressed: bool,
    /// Stressed period multiplier.
    pub stressed_multiplier: f64,
    /// Liquidity horizons to calculate.
    pub liquidity_horizons: Vec<usize>,
    /// Monte Carlo paths.
    pub mc_paths: usize,
}

impl Default for FRTBConfig {
    fn default() -> Self {
        Self {
            es_confidence: 0.975,
            use_stressed: true,
            stressed_multiplier: 1.5,
            liquidity_horizons: vec![10, 20, 40, 60, 120],
            mc_paths: 10_000,
        }
    }
}

/// FRTB result.
#[derive(Debug, Clone)]
pub struct FRTBResult {
    /// Expected Shortfall (base).
    pub es_base: f64,
    /// Expected Shortfall (stressed).
    pub es_stressed: f64,
    /// ES by liquidity horizon.
    pub es_by_horizon: Vec<(usize, f64)>,
    /// Incremental risk charge.
    pub irc: f64,
    /// Default risk charge.
    pub drc: f64,
    /// Total FRTB capital.
    pub total_capital: f64,
    /// P&L attribution test passed.
    pub pla_passed: bool,
}

/// FRTB calculator.
#[derive(Debug)]
pub struct FRTBCalculator {
    /// Configuration.
    config: FRTBConfig,
}

impl FRTBCalculator {
    /// Create new FRTB calculator.
    pub fn new(config: FRTBConfig) -> Self {
        Self { config }
    }

    /// Calculate FRTB capital requirements.
    pub fn calculate(&self, portfolio: &Portfolio, volatility: f64) -> FRTBResult {
        let portfolio_value = portfolio.total_value;

        // Base ES at 10-day horizon
        let es_10d = self.calculate_es(portfolio_value, volatility, 10);

        // Stressed ES (higher volatility assumption)
        let stressed_vol = volatility * self.config.stressed_multiplier;
        let es_stressed = self.calculate_es(portfolio_value, stressed_vol, 10);

        // ES by liquidity horizon
        let es_by_horizon: Vec<(usize, f64)> = self.config.liquidity_horizons
            .iter()
            .map(|&h| (h, self.calculate_es(portfolio_value, volatility, h)))
            .collect();

        // Aggregate ES across horizons (simplified)
        let aggregate_es = es_by_horizon.iter()
            .map(|(_, es)| es)
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .cloned()
            .unwrap_or(es_10d);

        // IRC (placeholder - would need credit spreads)
        let irc = portfolio_value * 0.005; // 0.5% as placeholder

        // DRC (placeholder - would need default probabilities)
        let drc = portfolio_value * 0.002; // 0.2% as placeholder

        // Total capital
        let es_capital = if self.config.use_stressed {
            es_stressed.max(aggregate_es)
        } else {
            aggregate_es
        };

        let total_capital = es_capital + irc + drc;

        FRTBResult {
            es_base: es_10d,
            es_stressed,
            es_by_horizon,
            irc,
            drc,
            total_capital,
            pla_passed: true, // Placeholder
        }
    }

    /// Calculate ES for given horizon.
    fn calculate_es(&self, value: f64, volatility: f64, horizon_days: usize) -> f64 {
        // Parametric ES approximation
        // ES = value * sigma * sqrt(T) * (phi(z) / (1-alpha))

        let sqrt_t = (horizon_days as f64).sqrt();
        let alpha = self.config.es_confidence;

        // Standard normal density at quantile
        let z = self.normal_quantile(alpha);
        let phi_z = (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt();

        let es = value * volatility * sqrt_t * phi_z / (1.0 - alpha);

        es.abs()
    }

    /// Normal quantile approximation.
    fn normal_quantile(&self, p: f64) -> f64 {
        let t = (-2.0 * (1.0 - p).ln()).sqrt();
        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frtb_creation() {
        let config = FRTBConfig::default();
        let _calc = FRTBCalculator::new(config);
    }

    #[test]
    fn test_frtb_calculation() {
        let config = FRTBConfig::default();
        let calc = FRTBCalculator::new(config);

        let portfolio = Portfolio::new(1_000_000.0);
        let volatility = 0.16 / (252.0_f64).sqrt(); // 16% annual -> daily

        let result = calc.calculate(&portfolio, volatility);

        // ES should be positive
        assert!(result.es_base > 0.0);
        // Stressed should be higher
        assert!(result.es_stressed >= result.es_base);
        // Should have results for all horizons
        assert_eq!(result.es_by_horizon.len(), 5);
        // Total capital should be positive
        assert!(result.total_capital > 0.0);
    }

    #[test]
    fn test_es_increases_with_horizon() {
        let config = FRTBConfig::default();
        let calc = FRTBCalculator::new(config);

        let portfolio = Portfolio::new(1_000_000.0);
        let volatility = 0.01;

        let result = calc.calculate(&portfolio, volatility);

        // ES should increase with horizon (sqrt(T) scaling)
        for i in 1..result.es_by_horizon.len() {
            let (h1, es1) = result.es_by_horizon[i - 1];
            let (h2, es2) = result.es_by_horizon[i];
            assert!(h2 > h1);
            assert!(es2 > es1);
        }
    }
}
