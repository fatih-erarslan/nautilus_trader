//! Complex Systems Sentinel - Self-Organized Criticality & Black Swan Detection.
//!
//! Implements complex adaptive systems analysis for market regime detection
//! and tail risk monitoring based on Self-Organized Criticality (SOC) theory.
//!
//! ## Scientific Foundation
//!
//! Based on complex systems theory:
//! - Bak et al. (1987): Self-Organized Criticality (sandpile model)
//! - Taleb (2007): Black Swan events and fat-tailed distributions
//! - Sornette (2003): Critical market dynamics and log-periodic oscillations
//!
//! ## SOC Market Regime Detection
//!
//! Markets exhibit SOC behavior near critical points:
//! - Power-law distributed returns (fat tails)
//! - Long-range correlations (Hurst exponent H > 0.5)
//! - Avalanche dynamics (cascade failures)
//!
//! ## References
//! - Mandelbrot (1963): "Variation of certain speculative prices"
//! - Cont (2001): "Empirical properties of asset returns"
//! - Gabaix et al. (2003): "Theory of power-law distributions"

use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use std::time::Instant;
use std::collections::VecDeque;

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::core::types::{MarketRegime, RiskDecision, RiskLevel, Symbol, Timestamp, Order, Portfolio};
use crate::core::error::Result;
use crate::sentinels::base::{Sentinel, SentinelId, SentinelStatus, SentinelConfig};

/// Configuration for the Complex Systems Sentinel.
#[derive(Debug, Clone)]
pub struct ComplexSystemsConfig {
    /// Base sentinel config.
    pub base: SentinelConfig,

    // === SOC Analysis Parameters ===

    /// Window size for Hurst exponent calculation.
    pub hurst_window: usize,
    /// Minimum window for R/S analysis.
    pub rs_min_window: usize,
    /// Power-law tail exponent threshold for SOC detection.
    pub tail_exponent_threshold: f64,
    /// Hurst exponent threshold for long-range dependence.
    pub hurst_threshold: f64,

    // === Black Swan Detection ===

    /// Number of standard deviations for extreme event classification.
    pub sigma_threshold: f64,
    /// Expected black swan frequency (per 1000 observations).
    pub black_swan_frequency_threshold: f64,
    /// Kurtosis threshold for fat-tail detection.
    pub kurtosis_threshold: f64,
    /// Lookback period for tail analysis.
    pub tail_lookback: usize,

    // === Critical State Detection ===

    /// Autocorrelation decay rate threshold.
    pub autocorr_decay_threshold: f64,
    /// Variance amplification threshold.
    pub variance_amplification_threshold: f64,
    /// Log-periodic oscillation detection enabled.
    pub detect_log_periodic: bool,
}

impl Default for ComplexSystemsConfig {
    fn default() -> Self {
        Self {
            base: SentinelConfig {
                name: "ComplexSystemsSentinel".to_string(),
                ..Default::default()
            },
            // SOC Analysis
            hurst_window: 100,
            rs_min_window: 10,
            tail_exponent_threshold: 3.0,  // α < 3 indicates fat tails
            hurst_threshold: 0.6,          // H > 0.6 indicates persistence

            // Black Swan Detection
            sigma_threshold: 6.0,          // 6-sigma events
            black_swan_frequency_threshold: 0.5,  // 0.5 per 1000 obs
            kurtosis_threshold: 6.0,       // Excess kurtosis > 6
            tail_lookback: 252,            // 1 year of daily data

            // Critical State
            autocorr_decay_threshold: 0.1,
            variance_amplification_threshold: 2.0,
            detect_log_periodic: true,
        }
    }
}

/// SOC regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SOCRegime {
    /// Subcritical - market is stable, mean-reverting.
    Subcritical,
    /// Critical - market near phase transition, high sensitivity.
    Critical,
    /// Supercritical - market in crisis mode, cascade dynamics.
    Supercritical,
    /// Unknown - insufficient data for classification.
    Unknown,
}

impl SOCRegime {
    /// Convert to market regime.
    pub fn to_market_regime(&self) -> MarketRegime {
        match self {
            SOCRegime::Subcritical => MarketRegime::SidewaysLow,
            SOCRegime::Critical => MarketRegime::SidewaysHigh,
            SOCRegime::Supercritical => MarketRegime::Crisis,
            SOCRegime::Unknown => MarketRegime::Unknown,
        }
    }

    /// Get risk level for this regime.
    pub fn risk_level(&self) -> RiskLevel {
        match self {
            SOCRegime::Subcritical => RiskLevel::Normal,
            SOCRegime::Critical => RiskLevel::Elevated,
            SOCRegime::Supercritical => RiskLevel::Critical,
            SOCRegime::Unknown => RiskLevel::Elevated,
        }
    }
}

/// Black Swan event detection result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanEvent {
    /// Symbol where event was detected.
    pub symbol: Symbol,
    /// Magnitude in standard deviations.
    pub sigma_magnitude: f64,
    /// Return value.
    pub return_value: f64,
    /// Probability under normal distribution.
    pub normal_probability: f64,
    /// Probability under fat-tailed distribution.
    pub fat_tail_probability: f64,
    /// Timestamp of detection.
    pub detected_at: Timestamp,
}

/// SOC analysis metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCMetrics {
    /// Hurst exponent (H).
    pub hurst_exponent: f64,
    /// Power-law tail exponent (α).
    pub tail_exponent: f64,
    /// Excess kurtosis.
    pub excess_kurtosis: f64,
    /// Skewness.
    pub skewness: f64,
    /// Autocorrelation at lag 1.
    pub autocorr_lag1: f64,
    /// Variance ratio test statistic.
    pub variance_ratio: f64,
    /// Current SOC regime.
    pub regime: SOCRegime,
    /// Criticality score (0-1).
    pub criticality_score: f64,
    /// Black swan frequency (observed vs expected).
    pub black_swan_ratio: f64,
}

/// Complex Systems Sentinel implementation.
#[derive(Debug)]
pub struct ComplexSystemsSentinel {
    /// Configuration.
    config: ComplexSystemsConfig,
    /// Whether sentinel is active.
    active: AtomicBool,
    /// Check counter.
    checks: AtomicU64,
    /// Trigger counter.
    triggers: AtomicU64,

    /// Return series buffer.
    returns: RwLock<VecDeque<f64>>,
    /// Current SOC metrics.
    soc_metrics: RwLock<Option<SOCMetrics>>,
    /// Detected black swan events.
    black_swans: RwLock<Vec<BlackSwanEvent>>,
    /// Running statistics.
    stats: RwLock<RunningStats>,
}

/// Running statistics for online computation.
#[derive(Debug, Clone, Default)]
struct RunningStats {
    count: u64,
    mean: f64,
    m2: f64,    // For variance
    m3: f64,    // For skewness
    m4: f64,    // For kurtosis
}

impl RunningStats {
    fn update(&mut self, x: f64) {
        self.count += 1;
        let n = self.count as f64;
        let delta = x - self.mean;
        let delta_n = delta / n;
        let delta_n2 = delta_n * delta_n;
        let term1 = delta * delta_n * (n - 1.0);

        self.mean += delta_n;
        self.m4 += term1 * delta_n2 * (n * n - 3.0 * n + 3.0)
            + 6.0 * delta_n2 * self.m2
            - 4.0 * delta_n * self.m3;
        self.m3 += term1 * delta_n * (n - 2.0) - 3.0 * delta_n * self.m2;
        self.m2 += term1;
    }

    fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / (self.count as f64 - 1.0)
    }

    fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    fn skewness(&self) -> f64 {
        if self.count < 3 || self.m2 < 1e-10 {
            return 0.0;
        }
        let n = self.count as f64;
        (n.sqrt() * self.m3) / (self.m2.powf(1.5))
    }

    fn kurtosis(&self) -> f64 {
        if self.count < 4 || self.m2 < 1e-10 {
            return 0.0;
        }
        let n = self.count as f64;
        (n * self.m4) / (self.m2 * self.m2) - 3.0  // Excess kurtosis
    }
}

impl ComplexSystemsSentinel {
    /// Create new Complex Systems Sentinel.
    pub fn new(config: ComplexSystemsConfig) -> Self {
        Self {
            config,
            active: AtomicBool::new(true),
            checks: AtomicU64::new(0),
            triggers: AtomicU64::new(0),
            returns: RwLock::new(VecDeque::with_capacity(500)),
            soc_metrics: RwLock::new(None),
            black_swans: RwLock::new(Vec::new()),
            stats: RwLock::new(RunningStats::default()),
        }
    }

    /// Ingest a new return observation.
    pub fn ingest_return(&self, ret: f64) {
        // Update running statistics
        self.stats.write().update(ret);

        // Add to return buffer
        let mut returns = self.returns.write();
        returns.push_back(ret);

        // Maintain buffer size
        let max_len = self.config.tail_lookback.max(self.config.hurst_window);
        while returns.len() > max_len {
            returns.pop_front();
        }

        // Check for black swan
        self.check_black_swan(ret);
    }

    /// Check if observation is a black swan event.
    fn check_black_swan(&self, ret: f64) {
        let stats = self.stats.read();
        if stats.count < 30 {
            return; // Need minimum data
        }

        let std_dev = stats.std_dev();
        if std_dev < 1e-10 {
            return;
        }

        let z_score = (ret - stats.mean).abs() / std_dev;

        if z_score >= self.config.sigma_threshold {
            // Calculate probabilities
            let normal_prob = normal_tail_probability(z_score);
            let fat_tail_prob = fat_tail_probability(z_score, stats.kurtosis());

            let event = BlackSwanEvent {
                symbol: Symbol::new("PORTFOLIO"),
                sigma_magnitude: z_score,
                return_value: ret,
                normal_probability: normal_prob,
                fat_tail_probability: fat_tail_prob,
                detected_at: Timestamp::now(),
            };

            self.black_swans.write().push(event);
            self.triggers.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Compute Hurst exponent using R/S analysis.
    ///
    /// H < 0.5: Mean-reverting (anti-persistent)
    /// H = 0.5: Random walk (no memory)
    /// H > 0.5: Trending (persistent)
    pub fn compute_hurst_exponent(&self) -> f64 {
        let returns = self.returns.read();
        let data: Vec<f64> = returns.iter().cloned().collect();

        if data.len() < self.config.hurst_window {
            return 0.5; // Default to random walk
        }

        // R/S analysis at multiple scales
        let mut log_n = Vec::new();
        let mut log_rs = Vec::new();

        let min_n = self.config.rs_min_window;
        let max_n = data.len() / 2;

        let mut n = min_n;
        while n <= max_n {
            let rs = compute_rs_statistic(&data, n);
            if rs > 0.0 {
                log_n.push((n as f64).ln());
                log_rs.push(rs.ln());
            }
            n = (n as f64 * 1.5).ceil() as usize;
        }

        if log_n.len() < 3 {
            return 0.5;
        }

        // Linear regression: log(R/S) = H * log(n) + c
        linear_regression_slope(&log_n, &log_rs)
    }

    /// Estimate tail exponent using Hill estimator.
    ///
    /// α < 2: Infinite variance (very fat tails)
    /// α ∈ [2, 3]: Finite variance, infinite kurtosis
    /// α > 3: All moments finite (thin tails)
    pub fn compute_tail_exponent(&self) -> f64 {
        let returns = self.returns.read();
        let mut abs_returns: Vec<f64> = returns.iter().map(|&x| x.abs()).collect();

        if abs_returns.len() < 50 {
            return 3.0; // Default to thin tails
        }

        // Sort in descending order
        abs_returns.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Use top k observations (k = n^0.5 is optimal)
        let k = ((abs_returns.len() as f64).sqrt().ceil() as usize).max(10);
        let threshold = abs_returns[k - 1];

        if threshold < 1e-10 {
            return 3.0;
        }

        // Hill estimator
        let mut sum_log = 0.0;
        for i in 0..k {
            sum_log += (abs_returns[i] / threshold).ln();
        }

        let alpha = k as f64 / sum_log;
        alpha.max(1.0).min(10.0) // Bound to reasonable range
    }

    /// Compute SOC metrics and determine regime.
    pub fn analyze_soc(&self) -> SOCMetrics {
        let stats = self.stats.read();
        let returns = self.returns.read();

        let hurst = self.compute_hurst_exponent();
        let tail_exp = self.compute_tail_exponent();
        let kurtosis = stats.kurtosis();
        let skewness = stats.skewness();

        // Compute autocorrelation at lag 1
        let autocorr = compute_autocorrelation(&returns.iter().cloned().collect::<Vec<_>>(), 1);

        // Compute variance ratio
        let variance_ratio = compute_variance_ratio(&returns.iter().cloned().collect::<Vec<_>>(), 5);

        // Calculate criticality score
        let criticality = self.compute_criticality_score(hurst, tail_exp, kurtosis, autocorr);

        // Determine regime
        let regime = self.classify_regime(hurst, tail_exp, criticality);

        // Black swan ratio
        let black_swan_count = self.black_swans.read().len();
        let expected_black_swans = (stats.count as f64 / 1000.0) * self.config.black_swan_frequency_threshold;
        let black_swan_ratio = if expected_black_swans > 0.0 {
            black_swan_count as f64 / expected_black_swans
        } else {
            0.0
        };

        let metrics = SOCMetrics {
            hurst_exponent: hurst,
            tail_exponent: tail_exp,
            excess_kurtosis: kurtosis,
            skewness,
            autocorr_lag1: autocorr,
            variance_ratio,
            regime,
            criticality_score: criticality,
            black_swan_ratio,
        };

        *self.soc_metrics.write() = Some(metrics.clone());
        metrics
    }

    /// Compute criticality score (0-1) indicating proximity to phase transition.
    fn compute_criticality_score(&self, hurst: f64, tail_exp: f64, kurtosis: f64, autocorr: f64) -> f64 {
        // Criticality indicators:
        // 1. Hurst exponent deviating from 0.5
        let hurst_score = (hurst - 0.5).abs().min(0.5) * 2.0;

        // 2. Fat tails (low tail exponent)
        let tail_score = ((self.config.tail_exponent_threshold - tail_exp) / self.config.tail_exponent_threshold)
            .max(0.0).min(1.0);

        // 3. High kurtosis
        let kurtosis_score = (kurtosis / self.config.kurtosis_threshold).min(1.0).max(0.0);

        // 4. High autocorrelation (slow decay)
        let autocorr_score = autocorr.abs().min(1.0);

        // Weighted average
        (0.3 * hurst_score + 0.3 * tail_score + 0.2 * kurtosis_score + 0.2 * autocorr_score)
    }

    /// Classify SOC regime based on metrics.
    fn classify_regime(&self, hurst: f64, tail_exp: f64, criticality: f64) -> SOCRegime {
        // Supercritical: Clear crisis indicators
        if criticality > 0.7 || tail_exp < 2.0 {
            return SOCRegime::Supercritical;
        }

        // Critical: Near phase transition
        if criticality > 0.4 || (hurst > self.config.hurst_threshold && tail_exp < self.config.tail_exponent_threshold) {
            return SOCRegime::Critical;
        }

        // Subcritical: Stable market
        if criticality < 0.2 && tail_exp > 3.5 && hurst < 0.55 {
            return SOCRegime::Subcritical;
        }

        SOCRegime::Unknown
    }

    /// Get recent black swan events.
    pub fn recent_black_swans(&self, limit: usize) -> Vec<BlackSwanEvent> {
        let events = self.black_swans.read();
        events.iter().rev().take(limit).cloned().collect()
    }

    /// Get current SOC metrics.
    pub fn current_metrics(&self) -> Option<SOCMetrics> {
        self.soc_metrics.read().clone()
    }

    /// Process check and return risk decision if needed.
    pub fn check(&self) -> Option<RiskDecision> {
        let start = Instant::now();
        self.checks.fetch_add(1, Ordering::Relaxed);

        let metrics = self.analyze_soc();
        let latency = start.elapsed().as_nanos() as u64;

        match metrics.regime {
            SOCRegime::Supercritical => Some(RiskDecision {
                allowed: true,
                risk_level: RiskLevel::Critical,
                reason: format!(
                    "[SOC] Supercritical regime detected: criticality={:.2}, H={:.2}, α={:.2}",
                    metrics.criticality_score, metrics.hurst_exponent, metrics.tail_exponent
                ),
                size_adjustment: 0.25, // 75% reduction
                timestamp: Timestamp::now(),
                latency_ns: latency,
            }),
            SOCRegime::Critical => Some(RiskDecision {
                allowed: true,
                risk_level: RiskLevel::Elevated,
                reason: format!(
                    "[SOC] Critical regime: criticality={:.2}, black_swan_ratio={:.2}",
                    metrics.criticality_score, metrics.black_swan_ratio
                ),
                size_adjustment: 0.6, // 40% reduction
                timestamp: Timestamp::now(),
                latency_ns: latency,
            }),
            _ => None,
        }
    }
}

impl Sentinel for ComplexSystemsSentinel {
    fn id(&self) -> SentinelId {
        SentinelId::new(&self.config.base.name)
    }

    fn status(&self) -> SentinelStatus {
        if self.active.load(Ordering::Relaxed) {
            SentinelStatus::Active
        } else {
            SentinelStatus::Disabled
        }
    }

    fn check(&self, _order: &Order, _portfolio: &Portfolio) -> Result<()> {
        // The complex systems sentinel uses a different check pattern (SOC analysis)
        // Orders pass through - this sentinel is for market regime monitoring
        let _ = self.check(); // Run SOC analysis
        Ok(())
    }

    fn reset(&self) {
        *self.returns.write() = VecDeque::with_capacity(500);
        *self.soc_metrics.write() = None;
        *self.black_swans.write() = Vec::new();
        *self.stats.write() = RunningStats::default();
        self.checks.store(0, Ordering::Relaxed);
        self.triggers.store(0, Ordering::Relaxed);
    }

    fn enable(&self) {
        self.active.store(true, Ordering::Relaxed);
    }

    fn disable(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    fn check_count(&self) -> u64 {
        self.checks.load(Ordering::Relaxed)
    }

    fn trigger_count(&self) -> u64 {
        self.triggers.load(Ordering::Relaxed)
    }

    fn avg_latency_ns(&self) -> u64 {
        // Estimate ~30μs per SOC analysis
        30_000
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute R/S statistic for a given window size.
fn compute_rs_statistic(data: &[f64], n: usize) -> f64 {
    if data.len() < n {
        return 0.0;
    }

    let mut rs_values = Vec::new();

    // Divide data into non-overlapping blocks of size n
    for chunk in data.chunks(n) {
        if chunk.len() < n {
            continue;
        }

        // Mean of chunk
        let mean: f64 = chunk.iter().sum::<f64>() / n as f64;

        // Cumulative deviation from mean
        let mut cumsum = Vec::with_capacity(n);
        let mut acc = 0.0;
        for &x in chunk {
            acc += x - mean;
            cumsum.push(acc);
        }

        // Range
        let max = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min = cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
        let range = max - min;

        // Standard deviation
        let variance: f64 = chunk.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std = variance.sqrt();

        if std > 1e-10 {
            rs_values.push(range / std);
        }
    }

    if rs_values.is_empty() {
        return 0.0;
    }

    // Return mean R/S
    rs_values.iter().sum::<f64>() / rs_values.len() as f64
}

/// Linear regression slope using ordinary least squares.
fn linear_regression_slope(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(xi, yi)| xi * yi).sum();
    let sum_xx: f64 = x.iter().map(|xi| xi * xi).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-10 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denom
}

/// Compute autocorrelation at given lag.
fn compute_autocorrelation(data: &[f64], lag: usize) -> f64 {
    if data.len() <= lag {
        return 0.0;
    }

    let n = data.len();
    let mean: f64 = data.iter().sum::<f64>() / n as f64;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &x) in data.iter().enumerate() {
        let diff = x - mean;
        denominator += diff * diff;
        if i >= lag {
            numerator += diff * (data[i - lag] - mean);
        }
    }

    if denominator < 1e-10 {
        return 0.0;
    }

    numerator / denominator
}

/// Compute variance ratio test statistic.
fn compute_variance_ratio(data: &[f64], q: usize) -> f64 {
    if data.len() < q * 2 {
        return 1.0;
    }

    // Variance of 1-period returns
    let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
    let var1: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

    // Variance of q-period returns
    let q_returns: Vec<f64> = data.windows(q).map(|w| w.iter().sum::<f64>()).collect();
    if q_returns.is_empty() {
        return 1.0;
    }

    let q_mean: f64 = q_returns.iter().sum::<f64>() / q_returns.len() as f64;
    let varq: f64 = q_returns.iter().map(|&x| (x - q_mean).powi(2)).sum::<f64>() / (q_returns.len() - 1) as f64;

    if var1 < 1e-10 {
        return 1.0;
    }

    varq / (q as f64 * var1)
}

/// Normal distribution tail probability (two-sided).
fn normal_tail_probability(z: f64) -> f64 {
    // Approximation using complementary error function
    let x = z / std::f64::consts::SQRT_2;
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    let prob = 0.5 * (1.0 + sign * y);
    2.0 * (1.0 - prob) // Two-sided
}

/// Fat-tailed (Student-t) probability approximation.
fn fat_tail_probability(z: f64, kurtosis: f64) -> f64 {
    // Estimate degrees of freedom from excess kurtosis: df = 4 + 6/kurtosis
    let df = if kurtosis > 0.5 { 4.0 + 6.0 / kurtosis } else { 30.0 };
    let df = df.max(3.0).min(100.0);

    // Tail probability for Student-t
    // Approximation: P(|T| > z) ≈ 2 * beta(df/2, 1/2) * (1 + z²/df)^(-(df+1)/2) / sqrt(df*π)
    let t_squared = z * z;
    let factor = (1.0 + t_squared / df).powf(-(df + 1.0) / 2.0);

    // Rough approximation
    factor * 2.0 / (std::f64::consts::PI * df).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentinel_creation() {
        let config = ComplexSystemsConfig::default();
        let sentinel = ComplexSystemsSentinel::new(config);
        assert_eq!(sentinel.status(), SentinelStatus::Active);
    }

    #[test]
    fn test_running_stats() {
        let mut stats = RunningStats::default();

        // Add some data
        for x in [1.0, 2.0, 3.0, 4.0, 5.0] {
            stats.update(x);
        }

        assert_eq!(stats.count, 5);
        assert!((stats.mean - 3.0).abs() < 1e-10);
        assert!(stats.variance() > 0.0);
    }

    #[test]
    fn test_hurst_exponent_random_walk() {
        let config = ComplexSystemsConfig {
            hurst_window: 50,
            rs_min_window: 5,
            ..Default::default()
        };
        let sentinel = ComplexSystemsSentinel::new(config);

        // Generate approximate random walk
        let mut rng = rand::thread_rng();
        use rand::Rng;
        for _ in 0..100 {
            let ret = rng.gen_range(-0.05..0.05);
            sentinel.ingest_return(ret);
        }

        let h = sentinel.compute_hurst_exponent();
        // Random walk should have H ≈ 0.5 (with some tolerance)
        assert!(h > 0.3 && h < 0.7, "Hurst exponent was {}", h);
    }

    #[test]
    fn test_black_swan_detection() {
        let config = ComplexSystemsConfig {
            sigma_threshold: 3.0,
            ..Default::default()
        };
        let sentinel = ComplexSystemsSentinel::new(config);

        // Add normal returns first
        for _ in 0..50 {
            sentinel.ingest_return(0.01);
        }

        // Add extreme return
        sentinel.ingest_return(0.5); // Should be many sigmas away

        let black_swans = sentinel.recent_black_swans(10);
        assert!(!black_swans.is_empty(), "Should detect black swan event");
    }

    #[test]
    fn test_soc_regime_classification() {
        let config = ComplexSystemsConfig::default();
        let sentinel = ComplexSystemsSentinel::new(config);

        // Add some data
        for _ in 0..100 {
            sentinel.ingest_return(0.01);
        }

        let metrics = sentinel.analyze_soc();
        assert!(metrics.hurst_exponent > 0.0);
        assert!(metrics.tail_exponent > 0.0);
    }

    #[test]
    fn test_linear_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x

        let slope = linear_regression_slope(&x, &y);
        assert!((slope - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_tail_probability() {
        let p_3sigma = normal_tail_probability(3.0);
        // Should be approximately 0.0027 (0.27%)
        assert!(p_3sigma > 0.001 && p_3sigma < 0.01);

        let p_6sigma = normal_tail_probability(6.0);
        // Should be very small
        assert!(p_6sigma < 1e-6);
    }
}
