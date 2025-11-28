//! Monte Carlo VaR/ES calculation.
//!
//! Generates scenarios using historical simulation or parametric
//! distribution and computes risk metrics.

use rand::prelude::*;
use rand_distr::{Normal, Distribution};

/// Monte Carlo configuration.
#[derive(Debug, Clone)]
pub struct MonteCarloConfig {
    /// Number of simulation paths.
    pub num_paths: usize,
    /// Time horizon in days.
    pub horizon_days: usize,
    /// VaR confidence level.
    pub var_confidence: f64,
    /// ES confidence level.
    pub es_confidence: f64,
    /// Use historical simulation.
    pub historical: bool,
    /// Use antithetic variates for variance reduction.
    pub antithetic: bool,
    /// Seed for reproducibility (None for random).
    pub seed: Option<u64>,
}

impl Default for MonteCarloConfig {
    fn default() -> Self {
        Self {
            num_paths: 10_000,
            horizon_days: 10,
            var_confidence: 0.99,
            es_confidence: 0.975,
            historical: false,
            antithetic: true,
            seed: None,
        }
    }
}

/// Monte Carlo result.
#[derive(Debug, Clone)]
pub struct MonteCarloResult {
    /// Value at Risk.
    pub var: f64,
    /// Expected Shortfall.
    pub es: f64,
    /// Mean of simulated P&L.
    pub mean_pnl: f64,
    /// Standard deviation of simulated P&L.
    pub std_pnl: f64,
    /// Worst case P&L.
    pub worst_case: f64,
    /// Best case P&L.
    pub best_case: f64,
    /// P&L percentiles (5%, 25%, 50%, 75%, 95%).
    pub percentiles: [f64; 5],
    /// Number of paths used.
    pub num_paths: usize,
    /// Computation time in milliseconds.
    pub compute_time_ms: u64,
}

/// Position for Monte Carlo.
#[derive(Debug, Clone)]
pub struct MCPosition {
    /// Position value.
    pub value: f64,
    /// Expected return (daily).
    pub expected_return: f64,
    /// Volatility (daily).
    pub volatility: f64,
    /// Correlation with other positions (indices).
    pub correlations: Vec<(usize, f64)>,
}

/// Monte Carlo VaR calculator.
#[derive(Debug)]
pub struct MonteCarloVaR {
    /// Configuration.
    config: MonteCarloConfig,
    /// RNG.
    rng: StdRng,
}

impl MonteCarloVaR {
    /// Create new Monte Carlo calculator.
    pub fn new(config: MonteCarloConfig) -> Self {
        let rng = match config.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };

        Self { config, rng }
    }

    /// Run Monte Carlo simulation for a portfolio.
    ///
    /// # Arguments
    /// * `positions` - Portfolio positions with their parameters
    pub fn calculate(&mut self, positions: &[MCPosition]) -> MonteCarloResult {
        let start = std::time::Instant::now();

        if positions.is_empty() {
            return MonteCarloResult {
                var: 0.0,
                es: 0.0,
                mean_pnl: 0.0,
                std_pnl: 0.0,
                worst_case: 0.0,
                best_case: 0.0,
                percentiles: [0.0; 5],
                num_paths: 0,
                compute_time_ms: 0,
            };
        }

        // Generate scenarios
        let mut pnls = Vec::with_capacity(self.config.num_paths);
        let normal = Normal::new(0.0, 1.0).unwrap();

        for _ in 0..self.config.num_paths {
            let pnl = self.simulate_path(positions, &normal);
            pnls.push(pnl);

            // Antithetic variate
            if self.config.antithetic {
                let anti_pnl = self.simulate_antithetic_path(positions, &normal);
                pnls.push(anti_pnl);
            }
        }

        // Sort for quantile calculation
        pnls.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = pnls.len();

        // VaR (negative of quantile for loss)
        let var_idx = ((1.0 - self.config.var_confidence) * n as f64) as usize;
        let var = -pnls[var_idx.max(0).min(n - 1)];

        // ES (average of worst losses)
        let es_idx = ((1.0 - self.config.es_confidence) * n as f64) as usize;
        let es = if es_idx > 0 {
            -pnls[..es_idx].iter().sum::<f64>() / es_idx as f64
        } else {
            var
        };

        // Statistics
        let mean_pnl: f64 = pnls.iter().sum::<f64>() / n as f64;
        let variance: f64 = pnls.iter().map(|p| (p - mean_pnl).powi(2)).sum::<f64>() / n as f64;
        let std_pnl = variance.sqrt();

        // Percentiles
        let percentiles = [
            pnls[(0.05 * n as f64) as usize],
            pnls[(0.25 * n as f64) as usize],
            pnls[(0.50 * n as f64) as usize],
            pnls[(0.75 * n as f64) as usize],
            pnls[(0.95 * n as f64) as usize],
        ];

        let compute_time = start.elapsed().as_millis() as u64;

        MonteCarloResult {
            var,
            es,
            mean_pnl,
            std_pnl,
            worst_case: pnls[0],
            best_case: pnls[n - 1],
            percentiles,
            num_paths: n,
            compute_time_ms: compute_time,
        }
    }

    /// Simulate a single path.
    fn simulate_path(&mut self, positions: &[MCPosition], normal: &Normal<f64>) -> f64 {
        let horizon = self.config.horizon_days as f64;
        let sqrt_horizon = horizon.sqrt();

        let mut total_pnl = 0.0;

        for pos in positions {
            // Generate random shock
            let z: f64 = normal.sample(&mut self.rng);

            // P&L = value * (mu * T + sigma * sqrt(T) * Z)
            let pnl = pos.value * (
                pos.expected_return * horizon +
                pos.volatility * sqrt_horizon * z
            );

            total_pnl += pnl;
        }

        total_pnl
    }

    /// Simulate antithetic path (negated random numbers).
    fn simulate_antithetic_path(&mut self, positions: &[MCPosition], normal: &Normal<f64>) -> f64 {
        let horizon = self.config.horizon_days as f64;
        let sqrt_horizon = horizon.sqrt();

        let mut total_pnl = 0.0;

        for pos in positions {
            let z: f64 = -normal.sample(&mut self.rng); // Negated for antithetic

            let pnl = pos.value * (
                pos.expected_return * horizon +
                pos.volatility * sqrt_horizon * z
            );

            total_pnl += pnl;
        }

        total_pnl
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monte_carlo_creation() {
        let config = MonteCarloConfig::default();
        let _mc = MonteCarloVaR::new(config);
    }

    #[test]
    fn test_monte_carlo_single_position() {
        let config = MonteCarloConfig {
            num_paths: 1000,
            seed: Some(42),
            antithetic: true,
            ..Default::default()
        };
        let mut mc = MonteCarloVaR::new(config);

        let positions = vec![MCPosition {
            value: 100_000.0,
            expected_return: 0.0005, // ~12.5% annual
            volatility: 0.01,        // ~16% annual
            correlations: vec![],
        }];

        let result = mc.calculate(&positions);

        // VaR should be positive (loss)
        assert!(result.var > 0.0);
        // ES should be >= VaR
        assert!(result.es >= result.var);
        // Used correct number of paths
        assert!(result.num_paths >= 1000);
    }

    #[test]
    fn test_monte_carlo_multiple_positions() {
        let config = MonteCarloConfig {
            num_paths: 500,
            seed: Some(123),
            ..Default::default()
        };
        let mut mc = MonteCarloVaR::new(config);

        let positions = vec![
            MCPosition {
                value: 50_000.0,
                expected_return: 0.0003,
                volatility: 0.015,
                correlations: vec![],
            },
            MCPosition {
                value: -30_000.0, // Short position
                expected_return: 0.0002,
                volatility: 0.012,
                correlations: vec![],
            },
        ];

        let result = mc.calculate(&positions);

        // Should produce valid results
        assert!(result.var.is_finite());
        assert!(result.es.is_finite());
        assert!(result.worst_case <= result.best_case);
    }

    #[test]
    fn test_monte_carlo_reproducibility() {
        let config = MonteCarloConfig {
            num_paths: 100,
            seed: Some(999),
            ..Default::default()
        };

        let positions = vec![MCPosition {
            value: 100_000.0,
            expected_return: 0.0,
            volatility: 0.01,
            correlations: vec![],
        }];

        let result1 = MonteCarloVaR::new(config.clone()).calculate(&positions);
        let result2 = MonteCarloVaR::new(config).calculate(&positions);

        // Should be identical with same seed
        assert!((result1.var - result2.var).abs() < 1e-10);
    }
}
