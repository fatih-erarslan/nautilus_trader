//! Greeks Sentinel for Derivatives Portfolio Sensitivity Monitoring.
//!
//! Monitors portfolio Greeks (Delta, Gamma, Vega, Theta, Rho) for options
//! and derivatives portfolios. Enforces risk limits on directional exposure,
//! convexity, volatility risk, and time decay.
//!
//! Target latency: <25μs
//!
//! ## Scientific References
//!
//! - Hull, John C. (2017): "Options, Futures, and Other Derivatives" (10th ed.)
//! - Black, Fischer & Scholes, Myron (1973): "The Pricing of Options and Corporate Liabilities"
//! - Taleb, Nassim N. (1997): "Dynamic Hedging: Managing Vanilla and Exotic Options"
//! - Haug, Espen G. (2007): "The Complete Guide to Option Pricing Formulas"

use std::sync::atomic::{AtomicI64, AtomicU8, Ordering};

use crate::core::error::{Result, RiskError};
use crate::core::types::{Order, Portfolio, Symbol};
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStats, SentinelStatus};

// ============================================================================
// Constants (from Black-Scholes-Merton framework)
// ============================================================================

/// Scaling factor for atomic storage (6 decimal places).
const SCALE: f64 = 1_000_000.0;

/// Standard normal cumulative distribution approximation accuracy.
#[allow(dead_code)]
const NORM_CDF_EPSILON: f64 = 1e-8;

// ============================================================================
// Configuration Types
// ============================================================================

/// Portfolio Greeks aggregation.
#[derive(Debug, Clone, Copy, Default)]
pub struct PortfolioGreeks {
    /// Net delta: ∂V/∂S (directional exposure, $ per $1 move in underlying).
    pub delta: f64,
    /// Net gamma: ∂²V/∂S² (convexity, delta sensitivity to underlying move).
    pub gamma: f64,
    /// Net vega: ∂V/∂σ (volatility exposure, $ per 1% vol change).
    pub vega: f64,
    /// Net theta: ∂V/∂t (time decay, $ per day).
    pub theta: f64,
    /// Net rho: ∂V/∂r (interest rate sensitivity, $ per 1% rate change).
    pub rho: f64,
}

impl PortfolioGreeks {
    /// Create zero Greeks.
    pub const fn zero() -> Self {
        Self {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        }
    }

    /// Add Greeks from a position.
    pub fn add(&mut self, other: &Self) {
        self.delta += other.delta;
        self.gamma += other.gamma;
        self.vega += other.vega;
        self.theta += other.theta;
        self.rho += other.rho;
    }

    /// Check if any Greek exceeds its limit.
    pub fn exceeds_limits(&self, limits: &GreeksLimits) -> Option<GreekType> {
        if self.delta.abs() > limits.max_delta {
            return Some(GreekType::Delta);
        }
        if self.gamma.abs() > limits.max_gamma {
            return Some(GreekType::Gamma);
        }
        if self.vega.abs() > limits.max_vega {
            return Some(GreekType::Vega);
        }
        if self.theta.abs() > limits.max_theta {
            return Some(GreekType::Theta);
        }
        if self.rho.abs() > limits.max_rho {
            return Some(GreekType::Rho);
        }
        None
    }
}

/// Greeks risk limits configuration.
#[derive(Debug, Clone, Copy)]
pub struct GreeksLimits {
    /// Maximum absolute delta ($ exposure).
    pub max_delta: f64,
    /// Maximum absolute gamma (convexity limit).
    pub max_gamma: f64,
    /// Maximum absolute vega (vol risk limit).
    pub max_vega: f64,
    /// Maximum absolute theta (time decay limit).
    pub max_theta: f64,
    /// Maximum absolute rho (rate sensitivity limit).
    pub max_rho: f64,
    /// Alert threshold (fraction of limit, e.g., 0.8 = 80%).
    pub alert_threshold: f64,
}

impl Default for GreeksLimits {
    fn default() -> Self {
        Self {
            max_delta: 100_000.0,      // $100k delta
            max_gamma: 10_000.0,        // $10k gamma
            max_vega: 50_000.0,         // $50k vega
            max_theta: 5_000.0,         // $5k theta decay per day
            max_rho: 20_000.0,          // $20k rho
            alert_threshold: 0.8,       // Alert at 80% of limit
        }
    }
}

impl GreeksLimits {
    /// High-frequency trading limits (tighter).
    pub fn hft() -> Self {
        Self {
            max_delta: 25_000.0,
            max_gamma: 2_500.0,
            max_vega: 10_000.0,
            max_theta: 1_000.0,
            max_rho: 5_000.0,
            alert_threshold: 0.75,
        }
    }

    /// Market maker limits (wider, for liquidity provision).
    pub fn market_maker() -> Self {
        Self {
            max_delta: 500_000.0,
            max_gamma: 50_000.0,
            max_vega: 250_000.0,
            max_theta: 25_000.0,
            max_rho: 100_000.0,
            alert_threshold: 0.9,
        }
    }

    /// Get limit for specific Greek.
    pub fn get_limit(&self, greek_type: GreekType) -> f64 {
        match greek_type {
            GreekType::Delta => self.max_delta,
            GreekType::Gamma => self.max_gamma,
            GreekType::Vega => self.max_vega,
            GreekType::Theta => self.max_theta,
            GreekType::Rho => self.max_rho,
        }
    }
}

/// Greek type enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GreekType {
    /// Delta: directional exposure.
    Delta,
    /// Gamma: convexity/delta acceleration.
    Gamma,
    /// Vega: volatility exposure.
    Vega,
    /// Theta: time decay.
    Theta,
    /// Rho: interest rate sensitivity.
    Rho,
}

impl GreekType {
    /// Get Greek name as string.
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Delta => "Delta",
            Self::Gamma => "Gamma",
            Self::Vega => "Vega",
            Self::Theta => "Theta",
            Self::Rho => "Rho",
        }
    }
}

/// Hedging recommendation.
#[derive(Debug, Clone)]
pub struct HedgeRecommendation {
    /// Instrument to hedge with.
    pub instrument: Symbol,
    /// Recommended quantity (negative = sell, positive = buy).
    pub quantity: f64,
    /// Target Greek to neutralize.
    pub target_greek: GreekType,
    /// Current exposure.
    pub current_exposure: f64,
    /// Target exposure after hedge.
    pub target_exposure: f64,
    /// Urgency level (0.0 = low, 1.0 = critical).
    pub urgency: f64,
}

/// Alert severity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational (below alert threshold).
    Info = 0,
    /// Warning (approaching limit, 80-95%).
    Warning = 1,
    /// Critical (near limit, 95-100%).
    Critical = 2,
    /// Emergency (limit breached).
    Emergency = 3,
}

/// Greeks alert.
#[derive(Debug, Clone)]
pub struct GreeksAlert {
    /// Which Greek triggered alert.
    pub greek_type: GreekType,
    /// Current value.
    pub current: f64,
    /// Configured limit.
    pub limit: f64,
    /// Utilization percentage (current/limit * 100).
    pub utilization_pct: f64,
    /// Alert severity.
    pub severity: AlertSeverity,
}

// ============================================================================
// Greeks Sentinel Implementation
// ============================================================================

/// Greeks monitoring sentinel.
///
/// Monitors portfolio sensitivity to market factors and enforces Greeks limits.
/// Uses lock-free atomics for fast-path checking (<25μs target).
#[derive(Debug)]
pub struct GreeksSentinel {
    /// Sentinel ID.
    id: SentinelId,
    /// Risk limits.
    limits: GreeksLimits,
    /// Current sentinel status.
    status: AtomicU8,
    /// Current portfolio delta (scaled by SCALE).
    delta_scaled: AtomicI64,
    /// Current portfolio gamma (scaled by SCALE).
    gamma_scaled: AtomicI64,
    /// Current portfolio vega (scaled by SCALE).
    vega_scaled: AtomicI64,
    /// Current portfolio theta (scaled by SCALE).
    theta_scaled: AtomicI64,
    /// Current portfolio rho (scaled by SCALE).
    rho_scaled: AtomicI64,
    /// Statistics tracker.
    stats: SentinelStats,
}

impl GreeksSentinel {
    /// Create new Greeks sentinel with custom limits.
    pub fn new(limits: GreeksLimits) -> Self {
        Self {
            id: SentinelId::new("greeks"),
            limits,
            status: AtomicU8::new(SentinelStatus::Active as u8),
            delta_scaled: AtomicI64::new(0),
            gamma_scaled: AtomicI64::new(0),
            vega_scaled: AtomicI64::new(0),
            theta_scaled: AtomicI64::new(0),
            rho_scaled: AtomicI64::new(0),
            stats: SentinelStats::new(),
        }
    }

    /// Update portfolio Greeks atomically.
    ///
    /// Should be called periodically (e.g., every second) by a background
    /// process that computes Greeks from current positions and market data.
    pub fn update_greeks(&self, greeks: &PortfolioGreeks) {
        self.delta_scaled.store((greeks.delta * SCALE) as i64, Ordering::Relaxed);
        self.gamma_scaled.store((greeks.gamma * SCALE) as i64, Ordering::Relaxed);
        self.vega_scaled.store((greeks.vega * SCALE) as i64, Ordering::Relaxed);
        self.theta_scaled.store((greeks.theta * SCALE) as i64, Ordering::Relaxed);
        self.rho_scaled.store((greeks.rho * SCALE) as i64, Ordering::Relaxed);
    }

    /// Get current portfolio Greeks.
    pub fn get_current_greeks(&self) -> PortfolioGreeks {
        PortfolioGreeks {
            delta: self.delta_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            gamma: self.gamma_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            vega: self.vega_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            theta: self.theta_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            rho: self.rho_scaled.load(Ordering::Relaxed) as f64 / SCALE,
        }
    }

    /// Check if any Greek is approaching its limit.
    ///
    /// Returns alerts sorted by severity (highest severity first).
    pub fn check_alerts(&self) -> Vec<GreeksAlert> {
        let greeks = self.get_current_greeks();
        let mut alerts = Vec::new();

        // Check each Greek against limits
        self.check_greek_alert(GreekType::Delta, greeks.delta, &mut alerts);
        self.check_greek_alert(GreekType::Gamma, greeks.gamma, &mut alerts);
        self.check_greek_alert(GreekType::Vega, greeks.vega, &mut alerts);
        self.check_greek_alert(GreekType::Theta, greeks.theta, &mut alerts);
        self.check_greek_alert(GreekType::Rho, greeks.rho, &mut alerts);

        // Sort by severity (highest first)
        alerts.sort_by(|a, b| b.severity.cmp(&a.severity));

        alerts
    }

    /// Check single Greek for alerts.
    fn check_greek_alert(&self, greek_type: GreekType, current: f64, alerts: &mut Vec<GreeksAlert>) {
        let limit = self.limits.get_limit(greek_type);
        let utilization_pct = (current.abs() / limit) * 100.0;

        let severity = if utilization_pct >= 100.0 {
            AlertSeverity::Emergency
        } else if utilization_pct >= 95.0 {
            AlertSeverity::Critical
        } else if utilization_pct >= self.limits.alert_threshold * 100.0 {
            AlertSeverity::Warning
        } else {
            AlertSeverity::Info
        };

        // Only add alerts for Warning and above
        if severity > AlertSeverity::Info {
            alerts.push(GreeksAlert {
                greek_type,
                current,
                limit,
                utilization_pct,
                severity,
            });
        }
    }

    /// Calculate hedge ratio for delta neutralization.
    ///
    /// Returns quantity of underlying needed to neutralize delta.
    /// Negative = sell underlying, Positive = buy underlying.
    pub fn calculate_delta_hedge(&self, target_delta: f64) -> f64 {
        let current_delta = self.delta_scaled.load(Ordering::Relaxed) as f64 / SCALE;
        -(current_delta - target_delta)
    }

    /// Calculate gamma scalping opportunity.
    ///
    /// High gamma means large delta changes, enabling scalping profits.
    /// Returns recommended rehedge size if gamma exceeds threshold.
    pub fn gamma_scalping_signal(&self, gamma_threshold: f64) -> Option<f64> {
        let gamma = self.gamma_scaled.load(Ordering::Relaxed) as f64 / SCALE;
        let delta = self.delta_scaled.load(Ordering::Relaxed) as f64 / SCALE;

        if gamma.abs() > gamma_threshold {
            // Recommend rehedge to capture gamma P&L
            Some(-delta)
        } else {
            None
        }
    }

    /// Generate vega neutralization recommendations.
    ///
    /// Suggests option trades to neutralize volatility exposure.
    pub fn vega_neutralization_hedge(&self, target_vega: f64, available_vega_per_contract: f64) -> Option<HedgeRecommendation> {
        let current_vega = self.vega_scaled.load(Ordering::Relaxed) as f64 / SCALE;
        let vega_diff = current_vega - target_vega;

        if vega_diff.abs() > 1000.0 && available_vega_per_contract.abs() > 0.01 {
            // Number of contracts needed
            let contracts = -vega_diff / available_vega_per_contract;

            let urgency = (vega_diff.abs() / self.limits.max_vega).min(1.0);

            Some(HedgeRecommendation {
                instrument: Symbol::new("OPTION"), // Placeholder - should be specific option
                quantity: contracts,
                target_greek: GreekType::Vega,
                current_exposure: current_vega,
                target_exposure: target_vega,
                urgency,
            })
        } else {
            None
        }
    }

    /// Get utilization percentage for a Greek.
    pub fn get_utilization_pct(&self, greek_type: GreekType) -> f64 {
        let current = match greek_type {
            GreekType::Delta => self.delta_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            GreekType::Gamma => self.gamma_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            GreekType::Vega => self.vega_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            GreekType::Theta => self.theta_scaled.load(Ordering::Relaxed) as f64 / SCALE,
            GreekType::Rho => self.rho_scaled.load(Ordering::Relaxed) as f64 / SCALE,
        };
        let limit = self.limits.get_limit(greek_type);
        (current.abs() / limit) * 100.0
    }
}

impl Default for GreeksSentinel {
    fn default() -> Self {
        Self::new(GreeksLimits::default())
    }
}

impl Sentinel for GreeksSentinel {
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

    /// Check Greeks limits.
    ///
    /// Target latency: <25μs
    ///
    /// This is a fast-path check that reads pre-computed Greeks from atomics
    /// and compares against limits. The actual Greeks calculation should be
    /// done asynchronously by a background process.
    #[inline]
    fn check(&self, _order: &Order, _portfolio: &Portfolio) -> Result<()> {
        let start = std::time::Instant::now();

        // Load current Greeks from atomics (very fast, no computation)
        let greeks = self.get_current_greeks();

        // Check each Greek against limits
        if let Some(violated_greek) = greeks.exceeds_limits(&self.limits) {
            self.stats.record_trigger();
            self.stats.record_check(start.elapsed().as_nanos() as u64);

            let current = match violated_greek {
                GreekType::Delta => greeks.delta,
                GreekType::Gamma => greeks.gamma,
                GreekType::Vega => greeks.vega,
                GreekType::Theta => greeks.theta,
                GreekType::Rho => greeks.rho,
            };
            let _limit = self.limits.get_limit(violated_greek);

            // Trigger sentinel
            self.status.store(SentinelStatus::Triggered as u8, Ordering::SeqCst);

            return Err(RiskError::InvalidParameter {
                param: violated_greek.name(),
                value: format!("{:.2}", current),
                constraint: "absolute value exceeds limit",
            });
        }

        self.stats.record_check(start.elapsed().as_nanos() as u64);
        Ok(())
    }

    fn reset(&self) {
        self.delta_scaled.store(0, Ordering::SeqCst);
        self.gamma_scaled.store(0, Ordering::SeqCst);
        self.vega_scaled.store(0, Ordering::SeqCst);
        self.theta_scaled.store(0, Ordering::SeqCst);
        self.rho_scaled.store(0, Ordering::SeqCst);
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

// ============================================================================
// Black-Scholes Greeks Calculator (for reference/testing)
// ============================================================================

/// Black-Scholes Greeks calculator.
///
/// This is provided for reference and testing. In production, Greeks should
/// be calculated by a separate process (medium/slow path) and fed to the
/// sentinel via `update_greeks()`.
#[derive(Debug, Clone, Copy)]
pub struct BlackScholesGreeks {
    /// Spot price of underlying.
    pub spot: f64,
    /// Strike price.
    pub strike: f64,
    /// Time to expiration (years).
    pub time_to_expiry: f64,
    /// Risk-free rate (annualized).
    pub risk_free_rate: f64,
    /// Volatility (annualized).
    pub volatility: f64,
    /// Call or put.
    pub is_call: bool,
}

impl BlackScholesGreeks {
    /// Calculate all Greeks for a European option.
    ///
    /// Based on Black-Scholes-Merton model (Black & Scholes, 1973).
    pub fn calculate(&self) -> PortfolioGreeks {
        let s = self.spot;
        let k = self.strike;
        let t = self.time_to_expiry;
        let r = self.risk_free_rate;
        let sigma = self.volatility;

        // Avoid division by zero
        if t < 1e-10 || sigma < 1e-10 {
            return PortfolioGreeks::zero();
        }

        let sqrt_t = t.sqrt();
        let sigma_sqrt_t = sigma * sqrt_t;

        // d1 = [ln(S/K) + (r + σ²/2)T] / (σ√T)
        let d1 = ((s / k).ln() + (r + 0.5 * sigma * sigma) * t) / sigma_sqrt_t;

        // d2 = d1 - σ√T
        let d2 = d1 - sigma_sqrt_t;

        let n_d1 = Self::norm_cdf(d1);
        let n_d2 = Self::norm_cdf(d2);
        let n_prime_d1 = Self::norm_pdf(d1);

        // Delta: ∂V/∂S
        let delta = if self.is_call {
            n_d1
        } else {
            n_d1 - 1.0
        };

        // Gamma: ∂²V/∂S² (same for call and put)
        let gamma = n_prime_d1 / (s * sigma_sqrt_t);

        // Vega: ∂V/∂σ (same for call and put, in $ per 1% vol change)
        let vega = s * n_prime_d1 * sqrt_t / 100.0;

        // Theta: ∂V/∂t (negative for long options due to time decay)
        let theta_call = -(s * n_prime_d1 * sigma) / (2.0 * sqrt_t)
            - r * k * (-r * t).exp() * n_d2;
        let theta_put = -(s * n_prime_d1 * sigma) / (2.0 * sqrt_t)
            + r * k * (-r * t).exp() * (1.0 - n_d2);
        let theta = if self.is_call {
            theta_call / 365.0 // Convert to daily theta
        } else {
            theta_put / 365.0
        };

        // Rho: ∂V/∂r (in $ per 1% rate change)
        let rho = if self.is_call {
            k * t * (-r * t).exp() * n_d2 / 100.0
        } else {
            -k * t * (-r * t).exp() * (1.0 - n_d2) / 100.0
        };

        PortfolioGreeks {
            delta: delta * s, // Convert to $ delta
            gamma: gamma * s, // Convert to $ gamma
            vega,
            theta,
            rho,
        }
    }

    /// Standard normal cumulative distribution function (CDF).
    ///
    /// Approximation using Hart (1968) algorithm with error < 7.5e-8.
    fn norm_cdf(x: f64) -> f64 {
        const A1: f64 = 0.319_381_530;
        const A2: f64 = -0.356_563_782;
        const A3: f64 = 1.781_477_937;
        const A4: f64 = -1.821_255_978;
        const A5: f64 = 1.330_274_429;
        const RSQRT2PI: f64 = 0.398_942_280_401_432_7;

        let k = 1.0 / (1.0 + 0.2316419 * x.abs());
        let k2 = k * k;
        let k3 = k2 * k;
        let k4 = k3 * k;
        let k5 = k4 * k;

        let cdf = RSQRT2PI * (-0.5 * x * x).exp() *
            (A1 * k + A2 * k2 + A3 * k3 + A4 * k4 + A5 * k5);

        if x >= 0.0 {
            1.0 - cdf
        } else {
            cdf
        }
    }

    /// Standard normal probability density function (PDF).
    ///
    /// φ(x) = (1/√(2π)) * exp(-x²/2)
    fn norm_pdf(x: f64) -> f64 {
        const SQRT2PI: f64 = 2.506_628_274_631_000_5;
        (-0.5 * x * x).exp() / SQRT2PI
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::types::{Order, OrderSide, Portfolio, Price, Quantity, Timestamp};

    fn test_order() -> Order {
        Order {
            symbol: Symbol::new("SPY"),
            side: OrderSide::Buy,
            quantity: Quantity::from_f64(100.0),
            limit_price: Some(Price::from_f64(450.0)),
            strategy_id: 1,
            timestamp: Timestamp::now(),
        }
    }

    #[test]
    fn test_greeks_below_limits() {
        let sentinel = GreeksSentinel::default();

        // Update with safe Greeks
        let greeks = PortfolioGreeks {
            delta: 50_000.0,   // Below 100k limit
            gamma: 5_000.0,    // Below 10k limit
            vega: 25_000.0,    // Below 50k limit
            theta: -2_000.0,   // Below 5k limit
            rho: 10_000.0,     // Below 20k limit
        };
        sentinel.update_greeks(&greeks);

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        assert!(sentinel.check(&order, &portfolio).is_ok());
    }

    #[test]
    fn test_delta_limit_exceeded() {
        let sentinel = GreeksSentinel::new(GreeksLimits {
            max_delta: 50_000.0,
            ..Default::default()
        });

        let greeks = PortfolioGreeks {
            delta: 75_000.0, // Exceeds 50k limit
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        let result = sentinel.check(&order, &portfolio);
        assert!(result.is_err());
        assert_eq!(sentinel.trigger_count(), 1);
    }

    #[test]
    fn test_gamma_limit_exceeded() {
        let sentinel = GreeksSentinel::new(GreeksLimits {
            max_gamma: 5_000.0,
            ..Default::default()
        });

        let greeks = PortfolioGreeks {
            gamma: 8_000.0, // Exceeds 5k limit
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        assert!(sentinel.check(&order, &portfolio).is_err());
    }

    #[test]
    fn test_vega_limit_exceeded() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            vega: 60_000.0, // Exceeds 50k limit
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        assert!(sentinel.check(&order, &portfolio).is_err());
    }

    #[test]
    fn test_alerts() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            delta: 85_000.0,   // 85% of 100k limit -> Warning
            gamma: 9_500.0,    // 95% of 10k limit -> Critical
            vega: 30_000.0,    // 60% of 50k limit -> Info (no alert)
            theta: -1_000.0,   // 20% of 5k limit -> Info
            rho: 5_000.0,      // 25% of 20k limit -> Info
        };
        sentinel.update_greeks(&greeks);

        let alerts = sentinel.check_alerts();

        // Should have 2 alerts (delta warning, gamma critical)
        assert_eq!(alerts.len(), 2);

        // Check gamma alert (should be first due to higher severity)
        assert_eq!(alerts[0].greek_type, GreekType::Gamma);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);

        // Check delta alert
        assert_eq!(alerts[1].greek_type, GreekType::Delta);
        assert_eq!(alerts[1].severity, AlertSeverity::Warning);
    }

    #[test]
    fn test_delta_hedge_calculation() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            delta: 50_000.0,
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        // Hedge to delta-neutral (0)
        let hedge = sentinel.calculate_delta_hedge(0.0);
        assert!((hedge + 50_000.0).abs() < 0.01, "Hedge should be -50000, got {}", hedge);

        // Hedge to 25k delta
        let hedge = sentinel.calculate_delta_hedge(25_000.0);
        assert!((hedge + 25_000.0).abs() < 0.01, "Hedge should be -25000, got {}", hedge);
    }

    #[test]
    fn test_gamma_scalping_signal() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            delta: 10_000.0,
            gamma: 15_000.0, // High gamma
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        // Should recommend rehedge when gamma high
        let signal = sentinel.gamma_scalping_signal(5_000.0);
        assert!(signal.is_some());
        assert!((signal.unwrap() + 10_000.0).abs() < 0.01);

        // No signal when gamma low
        let signal = sentinel.gamma_scalping_signal(20_000.0);
        assert!(signal.is_none());
    }

    #[test]
    fn test_vega_neutralization() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            vega: 20_000.0,
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        // 100 vega per option contract
        let hedge = sentinel.vega_neutralization_hedge(0.0, 100.0);

        assert!(hedge.is_some());
        let rec = hedge.unwrap();
        assert!((rec.quantity + 200.0).abs() < 0.01); // Should sell 200 contracts
        assert_eq!(rec.target_greek, GreekType::Vega);
    }

    #[test]
    fn test_utilization_percentage() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            delta: 50_000.0,  // 50% of 100k
            gamma: 7_500.0,   // 75% of 10k
            vega: 25_000.0,   // 50% of 50k
            ..PortfolioGreeks::zero()
        };
        sentinel.update_greeks(&greeks);

        assert!((sentinel.get_utilization_pct(GreekType::Delta) - 50.0).abs() < 0.1);
        assert!((sentinel.get_utilization_pct(GreekType::Gamma) - 75.0).abs() < 0.1);
        assert!((sentinel.get_utilization_pct(GreekType::Vega) - 50.0).abs() < 0.1);
    }

    #[test]
    fn test_black_scholes_greeks() {
        // ATM call option on SPY
        let bs = BlackScholesGreeks {
            spot: 450.0,
            strike: 450.0,
            time_to_expiry: 30.0 / 365.0, // 30 days
            risk_free_rate: 0.05,
            volatility: 0.20, // 20% vol
            is_call: true,
        };

        let greeks = bs.calculate();

        // ATM option delta should be ~0.5
        let delta_per_share = greeks.delta / bs.spot;
        assert!(delta_per_share > 0.4 && delta_per_share < 0.6,
            "ATM call delta should be ~0.5, got {}", delta_per_share);

        // Gamma should be positive
        assert!(greeks.gamma > 0.0, "Gamma should be positive");

        // Vega should be positive
        assert!(greeks.vega > 0.0, "Vega should be positive");

        // Theta should be negative (time decay)
        assert!(greeks.theta < 0.0, "Theta should be negative for long option");
    }

    #[test]
    fn test_normal_cdf() {
        // Test standard normal CDF values
        let cdf_0 = BlackScholesGreeks::norm_cdf(0.0);
        assert!((cdf_0 - 0.5).abs() < NORM_CDF_EPSILON, "N(0) should be 0.5");

        let cdf_1_645 = BlackScholesGreeks::norm_cdf(1.645);
        assert!((cdf_1_645 - 0.95).abs() < 0.001, "N(1.645) should be ~0.95");

        let cdf_neg = BlackScholesGreeks::norm_cdf(-1.96);
        assert!((cdf_neg - 0.025).abs() < 0.001, "N(-1.96) should be ~0.025");
    }

    #[test]
    fn test_latency() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            delta: 50_000.0,
            gamma: 5_000.0,
            vega: 25_000.0,
            theta: -2_000.0,
            rho: 10_000.0,
        };
        sentinel.update_greeks(&greeks);

        let order = test_order();
        let portfolio = Portfolio::new(1_000_000.0);

        // Warm up
        for _ in 0..1000 {
            let _ = sentinel.check(&order, &portfolio);
        }

        // Measure latency
        let start = std::time::Instant::now();
        for _ in 0..10_000 {
            let _ = sentinel.check(&order, &portfolio);
        }
        let elapsed = start.elapsed();
        let avg_ns = elapsed.as_nanos() / 10_000;

        println!("Average Greeks check latency: {}ns", avg_ns);
        assert!(avg_ns < 25_000, "Greeks check too slow: {}ns average (target: <25μs)", avg_ns);
    }

    #[test]
    fn test_reset() {
        let sentinel = GreeksSentinel::default();

        let greeks = PortfolioGreeks {
            delta: 50_000.0,
            gamma: 5_000.0,
            vega: 25_000.0,
            theta: -2_000.0,
            rho: 10_000.0,
        };
        sentinel.update_greeks(&greeks);

        sentinel.reset();

        let current = sentinel.get_current_greeks();
        assert!(current.delta.abs() < 0.01);
        assert!(current.gamma.abs() < 0.01);
        assert!(current.vega.abs() < 0.01);
        assert!(current.theta.abs() < 0.01);
        assert!(current.rho.abs() < 0.01);
        assert_eq!(sentinel.status(), SentinelStatus::Active);
    }

    #[test]
    fn test_enable_disable() {
        let sentinel = GreeksSentinel::default();

        assert_eq!(sentinel.status(), SentinelStatus::Active);

        sentinel.disable();
        assert_eq!(sentinel.status(), SentinelStatus::Disabled);

        sentinel.enable();
        assert_eq!(sentinel.status(), SentinelStatus::Active);
    }
}
