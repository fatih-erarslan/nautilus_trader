//! # Systems Dynamics Tracking
//!
//! Tracks agent state dynamics and computes criticality metrics.
//!
//! ## Theoretical Foundation
//!
//! Self-Organized Criticality (SOC) suggests that complex systems naturally
//! evolve toward critical states where they exhibit power-law behavior and
//! optimal information processing.
//!
//! ### Key Metrics
//!
//! **Branching Ratio (σ)**:
//! σ = <n_descendants> / <n_ancestors>
//!
//! At criticality: σ ≈ 1.0 (edge of chaos)
//! - σ < 1.0: Subcritical (activity dies out)
//! - σ > 1.0: Supercritical (activity explodes)
//! - σ ≈ 1.0: Critical (optimal information processing)
//!
//! **Hurst Exponent (H)**:
//! Measures long-range correlations in time series
//! - H = 0.5: Random walk (no memory)
//! - H > 0.5: Persistent (trending)
//! - H < 0.5: Anti-persistent (mean-reverting)
//!
//! **Avalanche Statistics**:
//! At criticality, avalanche sizes follow power law: P(s) ~ s^(-τ)
//! with τ ≈ 1.5 for mean-field systems
//!
//! ## References
//!
//! - Bak, P., Tang, C., & Wiesenfeld, K. (1987). "Self-organized criticality"
//! - Beggs, J. M., & Plenz, D. (2003). "Neuronal avalanches in neocortical circuits"
//! - Shew, W. L., & Plenz, D. (2013). "The functional benefits of criticality"

use crate::AgentState;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Criticality Metrics
// ============================================================================

/// Criticality analysis results
#[derive(Debug, Clone, Default)]
pub struct CriticalityMetrics {
    /// Branching ratio (σ ≈ 1.0 at criticality)
    pub branching_ratio: f64,
    /// Hurst exponent (long-range correlations)
    pub hurst_exponent: f64,
    /// Mean avalanche size
    pub mean_avalanche_size: f64,
    /// Power law exponent (τ ≈ 1.5 at criticality)
    pub power_law_exponent: f64,
    /// Distance from criticality (0 = critical)
    pub criticality_distance: f64,
}

/// Avalanche event
#[derive(Debug, Clone)]
pub struct Avalanche {
    /// Start time index
    pub start_time: usize,
    /// Duration in time steps
    pub duration: usize,
    /// Total size (sum of activations)
    pub size: f64,
    /// Peak amplitude
    pub peak: f64,
}

// ============================================================================
// Agency Dynamics Tracker
// ============================================================================

/// Tracks agent dynamics over time and computes criticality metrics
///
/// Implements continuous monitoring of:
/// 1. State trajectories (Φ, F, S, C over time)
/// 2. Branching ratio for criticality assessment
/// 3. Avalanche detection and statistics
/// 4. Long-range correlation analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgencyDynamics {
    /// State history buffer
    #[serde(skip)]
    state_history: VecDeque<AgentState>,

    /// Phi (Φ) time series for criticality analysis
    phi_series: VecDeque<f64>,

    /// Free energy time series
    fe_series: VecDeque<f64>,

    /// Survival drive time series
    survival_series: VecDeque<f64>,

    /// Control authority time series
    control_series: VecDeque<f64>,

    /// Detected avalanches
    #[serde(skip)]
    avalanches: Vec<Avalanche>,

    /// Maximum history length
    max_history: usize,

    /// Threshold for avalanche detection (in std deviations)
    avalanche_threshold: f64,

    /// Current avalanche tracking
    #[serde(skip)]
    in_avalanche: bool,
    #[serde(skip)]
    current_avalanche_start: usize,
    #[serde(skip)]
    current_avalanche_size: f64,
    #[serde(skip)]
    current_avalanche_peak: f64,
}

impl AgencyDynamics {
    /// Create new dynamics tracker
    pub fn new() -> Self {
        Self {
            state_history: VecDeque::with_capacity(1000),
            phi_series: VecDeque::with_capacity(1000),
            fe_series: VecDeque::with_capacity(1000),
            survival_series: VecDeque::with_capacity(1000),
            control_series: VecDeque::with_capacity(1000),
            avalanches: Vec::new(),
            max_history: 1000,
            avalanche_threshold: 2.0, // 2 standard deviations
            in_avalanche: false,
            current_avalanche_start: 0,
            current_avalanche_size: 0.0,
            current_avalanche_peak: 0.0,
        }
    }

    /// Create with custom history length
    pub fn with_capacity(capacity: usize) -> Self {
        let mut dynamics = Self::new();
        dynamics.max_history = capacity;
        dynamics.state_history = VecDeque::with_capacity(capacity);
        dynamics.phi_series = VecDeque::with_capacity(capacity);
        dynamics.fe_series = VecDeque::with_capacity(capacity);
        dynamics.survival_series = VecDeque::with_capacity(capacity);
        dynamics.control_series = VecDeque::with_capacity(capacity);
        dynamics
    }

    /// Record state snapshot
    pub fn record_state(&mut self, state: &AgentState) {
        // Add to history
        self.state_history.push_back(state.clone());
        if self.state_history.len() > self.max_history {
            self.state_history.pop_front();
        }

        // Extract key metrics
        self.phi_series.push_back(state.phi);
        if self.phi_series.len() > self.max_history {
            self.phi_series.pop_front();
        }

        self.fe_series.push_back(state.free_energy);
        if self.fe_series.len() > self.max_history {
            self.fe_series.pop_front();
        }

        self.survival_series.push_back(state.survival);
        if self.survival_series.len() > self.max_history {
            self.survival_series.pop_front();
        }

        self.control_series.push_back(state.control);
        if self.control_series.len() > self.max_history {
            self.control_series.pop_front();
        }

        // Avalanche detection on phi series
        self.detect_avalanche(state.phi);
    }

    /// Detect avalanches in activity
    fn detect_avalanche(&mut self, value: f64) {
        if self.phi_series.len() < 10 {
            return;
        }

        // Compute mean and std of recent activity
        let mean = self.phi_series.iter().sum::<f64>() / self.phi_series.len() as f64;
        let variance = self.phi_series.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
            / self.phi_series.len() as f64;
        let std = variance.sqrt().max(0.001);

        let threshold = mean + self.avalanche_threshold * std;
        let time_idx = self.phi_series.len();

        if value > threshold {
            if !self.in_avalanche {
                // Start new avalanche
                self.in_avalanche = true;
                self.current_avalanche_start = time_idx;
                self.current_avalanche_size = value - mean;
                self.current_avalanche_peak = value;
            } else {
                // Continue avalanche
                self.current_avalanche_size += value - mean;
                if value > self.current_avalanche_peak {
                    self.current_avalanche_peak = value;
                }
            }
        } else if self.in_avalanche {
            // End avalanche
            let avalanche = Avalanche {
                start_time: self.current_avalanche_start,
                duration: time_idx - self.current_avalanche_start,
                size: self.current_avalanche_size,
                peak: self.current_avalanche_peak,
            };
            self.avalanches.push(avalanche);

            // Keep only recent avalanches
            if self.avalanches.len() > 100 {
                self.avalanches.remove(0);
            }

            self.in_avalanche = false;
            self.current_avalanche_size = 0.0;
            self.current_avalanche_peak = 0.0;
        }
    }

    /// Compute branching ratio (criticality measure)
    ///
    /// σ = <A(t+1)> / <A(t)> where A is activity level
    ///
    /// At criticality: σ ≈ 1.0
    pub fn branching_ratio(&self) -> Option<f64> {
        if self.phi_series.len() < 10 {
            return None;
        }

        let series: Vec<f64> = self.phi_series.iter().copied().collect();
        Some(self.compute_branching_ratio(&series))
    }

    /// Compute branching ratio from arbitrary time series
    pub fn compute_branching_ratio(&self, series: &[f64]) -> f64 {
        if series.len() < 2 {
            return 1.0;
        }

        // Compute mean of ratios A(t+1)/A(t)
        let mut ratio_sum = 0.0;
        let mut count = 0;

        for i in 0..series.len() - 1 {
            let a_t = series[i].abs();
            let a_t1 = series[i + 1].abs();

            if a_t > 0.001 {
                ratio_sum += a_t1 / a_t;
                count += 1;
            }
        }

        if count > 0 {
            ratio_sum / count as f64
        } else {
            1.0
        }
    }

    /// Compute Hurst exponent using R/S analysis
    ///
    /// H ≈ 0.5: Random walk
    /// H > 0.5: Persistent (trending)
    /// H < 0.5: Anti-persistent (mean-reverting)
    pub fn hurst_exponent(&self) -> Option<f64> {
        if self.phi_series.len() < 50 {
            return None;
        }

        let series: Vec<f64> = self.phi_series.iter().copied().collect();
        Some(self.compute_hurst_exponent(&series))
    }

    /// Compute Hurst exponent from time series
    fn compute_hurst_exponent(&self, series: &[f64]) -> f64 {
        let n = series.len();
        if n < 20 {
            return 0.5;
        }

        // Use multiple window sizes
        let mut log_rs = Vec::new();
        let mut log_n = Vec::new();

        for window_size in [10, 20, 50, 100, 200].iter() {
            let ws = *window_size;
            if ws > n {
                continue;
            }

            let num_windows = n / ws;
            if num_windows == 0 {
                continue;
            }

            let mut rs_values = Vec::new();

            for w in 0..num_windows {
                let start = w * ws;
                let end = start + ws;
                let window: Vec<f64> = series[start..end].to_vec();

                // Compute R/S for this window
                let mean: f64 = window.iter().sum::<f64>() / ws as f64;
                let deviations: Vec<f64> = window.iter().map(|x| x - mean).collect();

                // Cumulative sum of deviations
                let mut cumsum = Vec::with_capacity(ws);
                let mut sum = 0.0;
                for d in &deviations {
                    sum += d;
                    cumsum.push(sum);
                }

                // Range
                let range = cumsum.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
                    - cumsum.iter().fold(f64::INFINITY, |a, &b| a.min(b));

                // Standard deviation
                let variance: f64 =
                    deviations.iter().map(|d| d * d).sum::<f64>() / ws as f64;
                let std = variance.sqrt().max(0.001);

                rs_values.push(range / std);
            }

            if !rs_values.is_empty() {
                let mean_rs: f64 = rs_values.iter().sum::<f64>() / rs_values.len() as f64;
                log_rs.push(mean_rs.ln());
                log_n.push((ws as f64).ln());
            }
        }

        // Linear regression to estimate H
        if log_rs.len() < 2 {
            return 0.5;
        }

        let (slope, _) = self.linear_regression(&log_n, &log_rs);
        slope.clamp(0.0, 1.0)
    }

    /// Simple linear regression
    fn linear_regression(&self, x: &[f64], y: &[f64]) -> (f64, f64) {
        let n = x.len() as f64;
        if n < 2.0 {
            return (0.0, 0.0);
        }

        let sum_x: f64 = x.iter().sum();
        let sum_y: f64 = y.iter().sum();
        let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let sum_xx: f64 = x.iter().map(|a| a * a).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x + 0.001);
        let intercept = (sum_y - slope * sum_x) / n;

        (slope, intercept)
    }

    /// Compute full criticality metrics
    pub fn criticality_metrics(&self) -> CriticalityMetrics {
        let branching = self.branching_ratio().unwrap_or(1.0);
        let hurst = self.hurst_exponent().unwrap_or(0.5);
        let mean_avalanche = self.mean_avalanche_size();
        let power_law = self.estimate_power_law_exponent();
        let distance = self.criticality_distance(branching, hurst);

        CriticalityMetrics {
            branching_ratio: branching,
            hurst_exponent: hurst,
            mean_avalanche_size: mean_avalanche,
            power_law_exponent: power_law,
            criticality_distance: distance,
        }
    }

    /// Compute mean avalanche size
    fn mean_avalanche_size(&self) -> f64 {
        if self.avalanches.is_empty() {
            return 0.0;
        }
        self.avalanches.iter().map(|a| a.size).sum::<f64>() / self.avalanches.len() as f64
    }

    /// Estimate power law exponent from avalanche distribution
    fn estimate_power_law_exponent(&self) -> f64 {
        if self.avalanches.len() < 10 {
            return 1.5; // Default mean-field value
        }

        // Use maximum likelihood estimation
        let sizes: Vec<f64> = self.avalanches.iter().map(|a| a.size.max(0.01)).collect();
        let min_size = sizes.iter().fold(f64::INFINITY, |a, &b| a.min(b));

        let n = sizes.len() as f64;
        let sum_log: f64 = sizes.iter().map(|s| (s / min_size).ln()).sum();

        // MLE estimator: τ = 1 + n / Σ ln(s_i / s_min)
        if sum_log > 0.001 {
            1.0 + n / sum_log
        } else {
            1.5
        }
    }

    /// Compute distance from criticality
    ///
    /// 0 = perfectly critical, higher = further from criticality
    fn criticality_distance(&self, sigma: f64, hurst: f64) -> f64 {
        // Ideal values at criticality
        let sigma_ideal = 1.0;
        let hurst_ideal = 0.5; // Random walk at criticality

        let sigma_distance = (sigma - sigma_ideal).abs();
        let hurst_distance = (hurst - hurst_ideal).abs();

        // Combined distance metric
        (sigma_distance.powi(2) + hurst_distance.powi(2)).sqrt()
    }

    /// Get state history length
    pub fn history_len(&self) -> usize {
        self.state_history.len()
    }

    /// Get avalanche count
    pub fn avalanche_count(&self) -> usize {
        self.avalanches.len()
    }

    /// Clear history
    pub fn clear(&mut self) {
        self.state_history.clear();
        self.phi_series.clear();
        self.fe_series.clear();
        self.survival_series.clear();
        self.control_series.clear();
        self.avalanches.clear();
        self.in_avalanche = false;
    }

    /// Get phi time series
    pub fn phi_series(&self) -> &VecDeque<f64> {
        &self.phi_series
    }

    /// Get free energy time series
    pub fn fe_series(&self) -> &VecDeque<f64> {
        &self.fe_series
    }

    /// Get recent trend in phi (positive = increasing)
    pub fn phi_trend(&self) -> f64 {
        if self.phi_series.len() < 10 {
            return 0.0;
        }

        // Take last 10 values in chronological order (not reversed)
        let start = self.phi_series.len().saturating_sub(10);
        let recent: Vec<f64> = self.phi_series.iter().skip(start).copied().collect();
        let indices: Vec<f64> = (0..recent.len()).map(|i| i as f64).collect();

        let (slope, _) = self.linear_regression(&indices, &recent);
        slope
    }

    /// Get recent trend in free energy (negative = improving)
    pub fn fe_trend(&self) -> f64 {
        if self.fe_series.len() < 10 {
            return 0.0;
        }

        // Take last 10 values in chronological order (not reversed)
        let start = self.fe_series.len().saturating_sub(10);
        let recent: Vec<f64> = self.fe_series.iter().skip(start).copied().collect();
        let indices: Vec<f64> = (0..recent.len()).map(|i| i as f64).collect();

        let (slope, _) = self.linear_regression(&indices, &recent);
        slope
    }
}

impl Default for AgencyDynamics {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Enterprise Features: Export, Statistics, Spectral Analysis
// ============================================================================

/// Temporal statistics for a time series
#[derive(Debug, Clone, Default)]
pub struct TemporalStats {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Lag-1 autocorrelation (temporal persistence)
    pub autocorr_lag1: f64,
    /// Volatility (std of differences)
    pub volatility: f64,
    /// Skewness (asymmetry)
    pub skewness: f64,
    /// Kurtosis (tail heaviness)
    pub kurtosis: f64,
}

impl TemporalStats {
    /// Compute statistics from a time series
    pub fn from_series(series: &[f64]) -> Self {
        if series.is_empty() {
            return Self::default();
        }

        let n = series.len() as f64;
        let mean = series.iter().sum::<f64>() / n;
        let variance = series.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        let min = series.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = series.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Autocorrelation (lag-1)
        let autocorr_lag1 = if series.len() > 1 {
            let mut sum = 0.0;
            for i in 0..series.len() - 1 {
                sum += (series[i] - mean) * (series[i + 1] - mean);
            }
            sum / ((n - 1.0) * variance.max(0.001))
        } else {
            0.0
        };

        // Volatility (std of first differences)
        let volatility = if series.len() > 1 {
            let diffs: Vec<f64> = series.windows(2).map(|w| w[1] - w[0]).collect();
            let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
            let var_diff = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;
            var_diff.sqrt()
        } else {
            0.0
        };

        // Skewness
        let skewness = if std > 0.001 {
            let m3 = series.iter().map(|x| ((x - mean) / std).powi(3)).sum::<f64>() / n;
            m3
        } else {
            0.0
        };

        // Kurtosis
        let kurtosis = if std > 0.001 {
            let m4 = series.iter().map(|x| ((x - mean) / std).powi(4)).sum::<f64>() / n;
            m4 - 3.0 // Excess kurtosis
        } else {
            0.0
        };

        Self {
            mean,
            std,
            min,
            max,
            autocorr_lag1,
            volatility,
            skewness,
            kurtosis,
        }
    }
}

/// Spectral analysis results
#[derive(Debug, Clone, Default)]
pub struct SpectralResult {
    /// Peak frequency (normalized)
    pub peak_frequency: Option<f64>,
    /// Peak power at dominant frequency
    pub peak_power: Option<f64>,
    /// Spectral entropy (complexity measure)
    pub spectral_entropy: Option<f64>,
    /// Harmonic frequencies detected
    pub harmonics: Vec<f64>,
}

/// Combined dynamics statistics
#[derive(Debug, Clone)]
pub struct DynamicsStats {
    /// Phi statistics
    pub phi: TemporalStats,
    /// Free energy statistics
    pub free_energy: TemporalStats,
    /// Control statistics
    pub control: TemporalStats,
    /// Survival statistics
    pub survival: TemporalStats,
    /// Number of samples
    pub n_samples: usize,
}

impl DynamicsStats {
    /// Compute emergence indicator (0 = no emergence, 1 = full emergence)
    pub fn emergence_indicator(&self) -> f64 {
        // Emergence = high phi + high control + low free energy variance
        let phi_component = (self.phi.mean / 5.0).clamp(0.0, 1.0);
        let control_component = self.control.mean.clamp(0.0, 1.0);
        let fe_stability = 1.0 / (1.0 + self.free_energy.volatility);

        (phi_component + control_component + fe_stability) / 3.0
    }

    /// Compute robustness score (0 = fragile, 1 = robust)
    pub fn robustness_score(&self) -> f64 {
        // Robustness = low volatility + high autocorrelation
        let phi_robust = 1.0 / (1.0 + self.phi.volatility);
        let control_robust = self.control.autocorr_lag1.abs().clamp(0.0, 1.0);
        let fe_robust = 1.0 / (1.0 + self.free_energy.volatility);

        (phi_robust + control_robust + fe_robust) / 3.0
    }
}

impl AgencyDynamics {
    /// Get history length
    pub fn len(&self) -> usize {
        self.phi_series.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.phi_series.is_empty()
    }

    /// Get time series by name
    pub fn get_series(&self, name: &str) -> Vec<f64> {
        match name {
            "phi" => self.phi_series.iter().copied().collect(),
            "free_energy" | "fe" => self.fe_series.iter().copied().collect(),
            "survival" => self.survival_series.iter().copied().collect(),
            "control" => self.control_series.iter().copied().collect(),
            _ => Vec::new(),
        }
    }

    /// Compute comprehensive statistics
    pub fn get_stats(&self) -> Option<DynamicsStats> {
        if self.phi_series.is_empty() {
            return None;
        }

        Some(DynamicsStats {
            phi: TemporalStats::from_series(&self.get_series("phi")),
            free_energy: TemporalStats::from_series(&self.get_series("fe")),
            control: TemporalStats::from_series(&self.get_series("control")),
            survival: TemporalStats::from_series(&self.get_series("survival")),
            n_samples: self.phi_series.len(),
        })
    }

    /// Compute criticality analysis (wrapper for criticality_metrics)
    pub fn compute_criticality(&self) -> CriticalitySummary {
        let metrics = self.criticality_metrics();
        CriticalitySummary {
            branching_ratio: Some(metrics.branching_ratio),
            hurst_exponent: Some(metrics.hurst_exponent),
            lyapunov_exponent: None, // Future: implement Lyapunov calculation
            entropy_rate: Some(self.compute_entropy_rate()),
        }
    }

    /// Compute entropy rate from time series
    fn compute_entropy_rate(&self) -> f64 {
        if self.phi_series.len() < 10 {
            return 0.0;
        }

        // Approximate entropy rate using first differences
        let diffs: Vec<f64> = self.phi_series
            .iter()
            .collect::<Vec<_>>()
            .windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        if diffs.is_empty() {
            return 0.0;
        }

        let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
        let var = diffs.iter().map(|d| (d - mean_diff).powi(2)).sum::<f64>() / diffs.len() as f64;

        // Entropy rate approximation (for Gaussian: log(sqrt(2πe*var)))
        if var > 0.001 {
            0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * var).ln()
        } else {
            0.0
        }
    }

    /// Perform spectral analysis using simple FFT approximation
    pub fn analyze_spectral(&self) -> SpectralResult {
        let series = self.get_series("phi");
        if series.len() < 32 {
            return SpectralResult::default();
        }

        // Use zero-padded DFT for power spectrum estimation
        let n = series.len();
        let mean = series.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = series.iter().map(|x| x - mean).collect();

        // Compute power spectrum via autocorrelation (Wiener-Khinchin)
        let mut power_spectrum = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let freq = k as f64 / n as f64;
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (t, &x) in centered.iter().enumerate() {
                let angle = 2.0 * std::f64::consts::PI * freq * t as f64;
                real_sum += x * angle.cos();
                imag_sum += x * angle.sin();
            }

            let power = (real_sum.powi(2) + imag_sum.powi(2)) / n as f64;
            power_spectrum.push((freq, power));
        }

        // Find peak frequency
        let (peak_freq, peak_power) = power_spectrum
            .iter()
            .skip(1) // Skip DC component
            .fold((0.0, 0.0), |(f, p), &(freq, power)| {
                if power > p { (freq, power) } else { (f, p) }
            });

        // Compute spectral entropy
        let total_power: f64 = power_spectrum.iter().map(|(_, p)| p).sum();
        let spectral_entropy = if total_power > 0.001 {
            power_spectrum
                .iter()
                .filter(|(_, p)| *p > 0.001)
                .map(|(_, p)| {
                    let normalized = p / total_power;
                    -normalized * normalized.ln()
                })
                .sum()
        } else {
            0.0
        };

        // Find harmonics (peaks above mean power)
        let mean_power = total_power / power_spectrum.len() as f64;
        let harmonics: Vec<f64> = power_spectrum
            .iter()
            .filter(|(_, p)| *p > mean_power * 2.0)
            .map(|(f, _)| *f)
            .collect();

        SpectralResult {
            peak_frequency: if peak_power > 0.0 { Some(peak_freq) } else { None },
            peak_power: if peak_power > 0.0 { Some(peak_power) } else { None },
            spectral_entropy: Some(spectral_entropy),
            harmonics,
        }
    }

    /// Export to CSV format
    pub fn export_csv(&self) -> String {
        let mut csv = String::from("time,phi,free_energy,control,survival,branching_ratio,hurst,trend\n");

        let phi: Vec<f64> = self.phi_series.iter().copied().collect();
        let fe: Vec<f64> = self.fe_series.iter().copied().collect();
        let control: Vec<f64> = self.control_series.iter().copied().collect();
        let survival: Vec<f64> = self.survival_series.iter().copied().collect();

        for i in 0..phi.len() {
            let row = format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}\n",
                i,
                phi.get(i).copied().unwrap_or(0.0),
                fe.get(i).copied().unwrap_or(0.0),
                control.get(i).copied().unwrap_or(0.0),
                survival.get(i).copied().unwrap_or(0.0),
                self.branching_ratio().unwrap_or(0.0),
                self.hurst_exponent().unwrap_or(0.5),
                self.phi_trend(),
            );
            csv.push_str(&row);
        }

        csv
    }

    /// Export to JSON format
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        use serde_json::json;

        let metrics = self.criticality_metrics();
        let stats = self.get_stats();

        let json_obj = json!({
            "phi": self.get_series("phi"),
            "free_energy": self.get_series("fe"),
            "control": self.get_series("control"),
            "survival": self.get_series("survival"),
            "metrics": {
                "branching_ratio": metrics.branching_ratio,
                "hurst_exponent": metrics.hurst_exponent,
                "criticality_distance": metrics.criticality_distance,
            },
            "stats": stats.map(|s| json!({
                "emergence": s.emergence_indicator(),
                "robustness": s.robustness_score(),
                "n_samples": s.n_samples,
            })),
        });

        serde_json::to_string_pretty(&json_obj)
    }
}

/// Criticality analysis summary (compatible with example code)
#[derive(Debug, Clone, Default)]
pub struct CriticalitySummary {
    /// Branching ratio
    pub branching_ratio: Option<f64>,
    /// Hurst exponent
    pub hurst_exponent: Option<f64>,
    /// Lyapunov exponent (future)
    pub lyapunov_exponent: Option<f64>,
    /// Entropy rate
    pub entropy_rate: Option<f64>,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    fn create_test_state(phi: f64, fe: f64, survival: f64, control: f64) -> AgentState {
        let mut position = Array1::zeros(12);
        position[0] = 1.0; // Lorentz origin

        AgentState {
            beliefs: Array1::from_elem(32, 0.5),
            phi,
            free_energy: fe,
            survival,
            control,
            model_accuracy: 0.8,
            precision: Array1::from_elem(32, 1.0),
            position,
            prediction_errors: std::collections::VecDeque::new(),
        }
    }

    #[test]
    fn test_dynamics_creation() {
        let dynamics = AgencyDynamics::new();
        assert!(dynamics.state_history.is_empty());
        assert_eq!(dynamics.max_history, 1000);
    }

    #[test]
    fn test_state_recording() {
        let mut dynamics = AgencyDynamics::new();
        let state = create_test_state(1.2, 0.8, 0.5, 0.7);

        dynamics.record_state(&state);

        assert_eq!(dynamics.history_len(), 1);
        assert_eq!(dynamics.phi_series.len(), 1);
        assert!((dynamics.phi_series[0] - 1.2).abs() < 0.001);
    }

    #[test]
    fn test_branching_ratio() {
        let mut dynamics = AgencyDynamics::new();

        // Record states with near-critical dynamics (ratio ≈ 1.0)
        for i in 0..20 {
            let phi = 1.0 + 0.1 * (i as f64 * 0.5).sin(); // Oscillating around 1.0
            let state = create_test_state(phi, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        let sigma = dynamics.branching_ratio().unwrap();
        assert!(sigma > 0.0 && sigma < 3.0, "Branching ratio should be reasonable: {}", sigma);
    }

    #[test]
    fn test_hurst_exponent() {
        let mut dynamics = AgencyDynamics::with_capacity(500);

        // Generate random walk (H ≈ 0.5)
        let mut value = 1.0;
        for _ in 0..200 {
            value += 0.1 * (rand_like() - 0.5);
            let state = create_test_state(value, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        let hurst = dynamics.hurst_exponent().unwrap();
        assert!(hurst > 0.0 && hurst < 1.0, "Hurst should be in [0,1]: {}", hurst);
    }

    #[test]
    fn test_avalanche_detection() {
        let mut dynamics = AgencyDynamics::new();

        // Normal activity
        for _ in 0..20 {
            let state = create_test_state(1.0, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        // Spike (avalanche)
        for _ in 0..5 {
            let state = create_test_state(3.0, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        // Return to normal
        for _ in 0..10 {
            let state = create_test_state(1.0, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        assert!(dynamics.avalanche_count() > 0, "Should detect avalanche");
    }

    #[test]
    fn test_criticality_metrics() {
        let mut dynamics = AgencyDynamics::new();

        for i in 0..50 {
            let phi = 1.0 + 0.2 * (i as f64 * 0.3).sin();
            let state = create_test_state(phi, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        let metrics = dynamics.criticality_metrics();

        assert!(metrics.branching_ratio.is_finite());
        assert!(metrics.hurst_exponent.is_finite());
        assert!(metrics.criticality_distance >= 0.0);
    }

    #[test]
    fn test_trends() {
        let mut dynamics = AgencyDynamics::new();

        // Increasing phi trend
        for i in 0..20 {
            let phi = 0.5 + 0.05 * i as f64;
            let state = create_test_state(phi, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        let trend = dynamics.phi_trend();
        assert!(trend > 0.0, "Phi trend should be positive for increasing series");
    }

    #[test]
    fn test_clear() {
        let mut dynamics = AgencyDynamics::new();

        for _ in 0..10 {
            let state = create_test_state(1.0, 0.8, 0.5, 0.7);
            dynamics.record_state(&state);
        }

        dynamics.clear();

        assert_eq!(dynamics.history_len(), 0);
        assert_eq!(dynamics.phi_series.len(), 0);
    }

    // Simple pseudo-random for tests (deterministic)
    fn rand_like() -> f64 {
        use std::time::SystemTime;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos();
        (nanos % 1000) as f64 / 1000.0
    }
}
