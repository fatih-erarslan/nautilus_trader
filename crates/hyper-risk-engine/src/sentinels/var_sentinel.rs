//! Value-at-Risk (VaR) Sentinel.
//!
//! Real-time VaR monitoring using pre-computed quantiles.
//! Target latency: <20μs
//!
//! ## Scientific References
//!
//! - McNeil, Frey, Embrechts (2005): "Quantitative Risk Management"
//! - Jorion (2006): "Value at Risk: The New Benchmark for Managing Financial Risk"

use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

/// VaR calculation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VaRMethod {
    /// Historical simulation.
    Historical,
    /// Parametric (variance-covariance).
    Parametric,
    /// Monte Carlo simulation (pre-computed).
    MonteCarlo,
    /// Cornish-Fisher expansion for fat tails.
    CornishFisher,
}

/// VaR sentinel configuration.
#[derive(Debug, Clone)]
pub struct VaRConfig {
    /// Confidence level (e.g., 0.95 or 0.99).
    pub confidence: f64,
    /// Time horizon in days (typically 1 or 10).
    pub horizon_days: u32,
    /// VaR limit as fraction of portfolio.
    pub var_limit_pct: f64,
    /// CVaR (Expected Shortfall) limit as fraction.
    pub cvar_limit_pct: f64,
    /// VaR calculation method.
    pub method: VaRMethod,
    /// Stress VaR multiplier.
    pub stress_multiplier: f64,
}

impl Default for VaRConfig {
    fn default() -> Self {
        Self {
            confidence: 0.95,
            horizon_days: 1,
            var_limit_pct: 0.02,      // 2% daily VaR limit
            cvar_limit_pct: 0.03,     // 3% daily CVaR limit
            method: VaRMethod::Historical,
            stress_multiplier: 1.5,
        }
    }
}

impl VaRConfig {
    /// Basel III regulatory configuration.
    pub fn basel3() -> Self {
        Self {
            confidence: 0.99,
            horizon_days: 10,
            var_limit_pct: 0.05,
            cvar_limit_pct: 0.075,
            method: VaRMethod::Historical,
            stress_multiplier: 3.0,
        }
    }

    /// High-frequency trading configuration.
    pub fn hft() -> Self {
        Self {
            confidence: 0.999,
            horizon_days: 1,
            var_limit_pct: 0.005,
            cvar_limit_pct: 0.01,
            method: VaRMethod::Parametric,
            stress_multiplier: 2.0,
        }
    }
}

/// Pre-computed VaR quantiles for fast lookup.
///
/// Quantiles are computed offline and stored for O(1) lookup.
#[derive(Debug, Clone)]
pub struct VaRQuantiles {
    /// Quantile values indexed by confidence level.
    /// Index = (confidence - 0.90) * 1000 for 0.90 to 0.999
    quantiles: [f64; 100],
    /// CVaR values (expected shortfall beyond VaR).
    cvar: [f64; 100],
    /// Timestamp of last update.
    last_update: u64,
}

impl Default for VaRQuantiles {
    fn default() -> Self {
        // Initialize with standard normal quantiles
        // These should be replaced with actual computed quantiles
        let mut quantiles = [0.0f64; 100];
        let mut cvar = [0.0f64; 100];

        // Standard normal quantiles from 0.90 to 0.999
        // z-scores for common confidence levels
        quantiles[0] = 1.282;   // 0.90
        quantiles[50] = 1.645;  // 0.95
        quantiles[90] = 2.326;  // 0.99
        quantiles[99] = 3.090;  // 0.999

        // Linear interpolation for other values
        for i in 1..100 {
            if quantiles[i] == 0.0 {
                let prev_idx = (0..i).rev().find(|&j| quantiles[j] != 0.0).unwrap_or(0);
                let next_idx = (i..100).find(|&j| quantiles[j] != 0.0).unwrap_or(99);

                if prev_idx != next_idx {
                    let t = (i - prev_idx) as f64 / (next_idx - prev_idx) as f64;
                    quantiles[i] = quantiles[prev_idx] + t * (quantiles[next_idx] - quantiles[prev_idx]);
                }
            }
            // CVaR is approximately VaR * 1.15 for normal distribution
            cvar[i] = quantiles[i] * 1.15;
        }

        Self {
            quantiles,
            cvar,
            last_update: 0,
        }
    }
}

impl VaRQuantiles {
    /// Get quantile for confidence level.
    #[inline]
    pub fn get_quantile(&self, confidence: f64) -> f64 {
        let idx = ((confidence - 0.90) * 1000.0).clamp(0.0, 99.0) as usize;
        self.quantiles[idx]
    }

    /// Get CVaR for confidence level.
    #[inline]
    pub fn get_cvar(&self, confidence: f64) -> f64 {
        let idx = ((confidence - 0.90) * 1000.0).clamp(0.0, 99.0) as usize;
        self.cvar[idx]
    }

    /// Update quantiles from new data.
    pub fn update(&mut self, quantiles: [f64; 100], cvar: [f64; 100], timestamp: u64) {
        self.quantiles = quantiles;
        self.cvar = cvar;
        self.last_update = timestamp;
    }
}

/// VaR sentinel.
///
/// Monitors portfolio VaR against limits using pre-computed quantiles.
#[derive(Debug)]
pub struct VaRSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Configuration.
    config: VaRConfig,
    /// Current status.
    status: AtomicU8,
    /// Pre-computed quantiles.
    quantiles: VaRQuantiles,
    /// Current portfolio volatility (annualized, scaled by 1M).
    portfolio_vol_scaled: AtomicU64,
    /// Current VaR (scaled).
    current_var_scaled: AtomicU64,
    /// Current CVaR (scaled).
    current_cvar_scaled: AtomicU64,
    /// Statistics.
    stats: SentinelStats,
}

impl VaRSentinel {
    const SCALE: f64 = 1_000_000.0;

    /// Create new VaR sentinel.
    pub fn new(config: VaRConfig) -> Self {
        Self {
            id: SentinelId::new("var"),
            config,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            quantiles: VaRQuantiles::default(),
            portfolio_vol_scaled: AtomicU64::new(0),
            current_var_scaled: AtomicU64::new(0),
            current_cvar_scaled: AtomicU64::new(0),
            stats: SentinelStats::new(),
        }
    }

    /// Update portfolio volatility.
    pub fn update_volatility(&self, annualized_vol: f64) {
        let scaled = (annualized_vol * Self::SCALE) as u64;
        self.portfolio_vol_scaled.store(scaled, Ordering::Relaxed);
    }

    /// Update pre-computed quantiles.
    pub fn update_quantiles(&mut self, quantiles: VaRQuantiles) {
        self.quantiles = quantiles;
    }

    /// Calculate VaR using parametric method.
    ///
    /// VaR = portfolio_value * z_alpha * sigma * sqrt(horizon)
    #[inline]
    fn calculate_parametric_var(&self, portfolio_value: f64) -> (f64, f64) {
        let vol = self.portfolio_vol_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE;
        let z_alpha = self.quantiles.get_quantile(self.config.confidence);
        let z_cvar = self.quantiles.get_cvar(self.config.confidence);

        // Convert annual vol to horizon vol
        let horizon_factor = (self.config.horizon_days as f64 / 252.0).sqrt();
        let horizon_vol = vol * horizon_factor;

        let var = portfolio_value * z_alpha * horizon_vol;
        let cvar = portfolio_value * z_cvar * horizon_vol;

        (var, cvar)
    }

    /// Get current VaR as percentage.
    pub fn current_var_pct(&self) -> f64 {
        self.current_var_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE
    }

    /// Get current CVaR as percentage.
    pub fn current_cvar_pct(&self) -> f64 {
        self.current_cvar_scaled.load(Ordering::Relaxed) as f64 / Self::SCALE
    }
}

impl Default for VaRSentinel {
    fn default() -> Self {
        Self::new(VaRConfig::default())
    }
}

impl Sentinel for VaRSentinel {
    fn id(&self) -> SentinelId {
        self.id.clone()
    }

    fn status(&self) -> SentinelStatus {
        match self.status.load(Ordering::Relaxed) {
            0 => SentinelStatus::Active,
            1 => SentinelStatus::Disabled,
            2 => SentinelStatus::Triggered,
            _ => SentinelStatus::Error,
        }
    }

    /// Check VaR limits.
    ///
    /// Target: <20μs
    #[inline]
    fn check(&self, _order: &Order, portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();
        let portfolio_value = portfolio.total_value;

        // Calculate VaR and CVaR
        let (var, cvar) = self.calculate_parametric_var(portfolio_value);

        // Convert to percentages
        let var_pct = if portfolio_value > 0.0 {
            var / portfolio_value
        } else {
            0.0
        };
        let cvar_pct = if portfolio_value > 0.0 {
            cvar / portfolio_value
        } else {
            0.0
        };

        // Store current values
        self.current_var_scaled.store((var_pct * Self::SCALE) as u64, Ordering::Relaxed);
        self.current_cvar_scaled.store((cvar_pct * Self::SCALE) as u64, Ordering::Relaxed);

        // Check VaR limit
        if var_pct > self.config.var_limit_pct {
            self.stats.record_trigger();
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::VaRLimitExceeded {
                var: var_pct,
                limit: self.config.var_limit_pct,
                confidence: self.config.confidence,
            });
        }

        // Check CVaR limit
        if cvar_pct > self.config.cvar_limit_pct {
            self.stats.record_trigger();
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::CVaRLimitExceeded {
                cvar: cvar_pct,
                limit: self.config.cvar_limit_pct,
                confidence: self.config.confidence,
            });
        }

        // Check stress VaR
        let stress_var_pct = var_pct * self.config.stress_multiplier;
        if stress_var_pct > self.config.var_limit_pct * 2.0 {
            self.stats.record_check(start.elapsed().as_nanos() as u64);
            return Err(RiskError::StressVaRExceeded {
                stress_var: stress_var_pct,
                limit: self.config.var_limit_pct * 2.0,
            });
        }

        self.stats.record_check(start.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn reset(&self) {
        self.current_var_scaled.store(0, Ordering::SeqCst);
        self.current_cvar_scaled.store(0, Ordering::SeqCst);
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
        self.stats.reset();
    }

    fn enable(&self) {
        self.status.store(SentinelStatus::Active as u8, Ordering::SeqCst);
    }

    fn disable(&self) {
        self.status.store(SentinelStatus::Disabled as u8, Ordering::SeqCst);
    }

    fn check_count(&self) -> u64 {
        self.stats.checks.load(Ordering::Relaxed)
    }

    fn trigger_count(&self) -> u64 {
        self.stats.triggers.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        self.stats.avg_latency_ns()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{OrderSide, Price, Quantity, Symbol, Timestamp};

    fn test_order() -> Order {
        Order {
            symbol: Symbol::new("AAPL"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: Some(Price::from_f64(150.0)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_var_below_limit() {
        let sentinel = VaRSentinel::default();
        sentinel.update_volatility(0.15); // 15% annual vol

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_var_above_limit() {
        let sentinel = VaRSentinel::new(VaRConfig {
            var_limit_pct: 0.01, // 1% limit
            ..Default::default()
        });
        sentinel.update_volatility(0.50); // 50% annual vol

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
    }

    #[test]
    fn test_quantile_lookup() {
        let quantiles = VaRQuantiles::default();

        // 95% confidence should give ~1.645
        let q95 = quantiles.get_quantile(0.95);
        assert!((q95 - 1.645).abs() < 0.1);

        // 99% confidence should give ~2.326
        let q99 = quantiles.get_quantile(0.99);
        assert!((q99 - 2.326).abs() < 0.1);
    }

    #[test]
    fn test_latency() {
        let sentinel = VaRSentinel::default();
        sentinel.update_volatility(0.20);

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        // Warm up
        for _ in 0..1000 {
            let _ = sentinel.check(&order, &portfolio);
        }

        // Measure
        let start = std::time::Instant::now();
        for _ in 0..10000 {
            let _ = sentinel.check(&order, &portfolio);
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / 10000;

        assert!(avg_ns < 20000, "VaR check too slow: {}ns average", avg_ns);
    }
}
