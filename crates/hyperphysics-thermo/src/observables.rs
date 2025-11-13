//! Physical observables and measurement operators
//!
//! Implements quantum and statistical mechanical observables including
//! expectation values, correlations, and time-dependent measurements.

// Observables module doesn't use Result type
use std::collections::HashMap;

/// Observable measurement result
#[derive(Debug, Clone)]
pub struct Observable {
    /// Expectation value ⟨O⟩
    pub expectation: f64,

    /// Variance ⟨O²⟩ - ⟨O⟩²
    pub variance: f64,

    /// Standard deviation σ = √variance
    pub std_dev: f64,

    /// Number of samples
    pub n_samples: usize,

    /// Confidence interval (95%)
    pub confidence_interval: (f64, f64),
}

impl Observable {
    /// Create observable from samples
    pub fn from_samples(samples: &[f64]) -> Self {
        let n = samples.len();
        if n == 0 {
            return Self::zero();
        }

        let expectation = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter()
            .map(|x| (x - expectation).powi(2))
            .sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // 95% confidence interval (±1.96 σ/√n)
        let margin = 1.96 * std_dev / (n as f64).sqrt();
        let confidence_interval = (expectation - margin, expectation + margin);

        Self {
            expectation,
            variance,
            std_dev,
            n_samples: n,
            confidence_interval,
        }
    }

    /// Zero observable
    pub fn zero() -> Self {
        Self {
            expectation: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            n_samples: 0,
            confidence_interval: (0.0, 0.0),
        }
    }

    /// Relative error (σ / |⟨O⟩|)
    pub fn relative_error(&self) -> f64 {
        if self.expectation.abs() < 1e-10 {
            return f64::INFINITY;
        }
        self.std_dev / self.expectation.abs()
    }

    /// Check if measurement is converged (relative error < threshold)
    pub fn is_converged(&self, threshold: f64) -> bool {
        self.relative_error() < threshold
    }

    /// Signal-to-noise ratio
    pub fn signal_to_noise(&self) -> f64 {
        if self.std_dev < 1e-10 {
            return f64::INFINITY;
        }
        self.expectation.abs() / self.std_dev
    }
}

/// Two-point correlation function ⟨O(t)O(0)⟩
#[derive(Debug, Clone)]
pub struct Correlation {
    /// Time lags
    pub time_lags: Vec<f64>,

    /// Correlation values
    pub values: Vec<f64>,

    /// Correlation length (exponential decay constant)
    pub correlation_length: Option<f64>,
}

impl Correlation {
    /// Compute correlation function from time series
    pub fn from_time_series(times: &[f64], values: &[f64]) -> Self {
        assert_eq!(times.len(), values.len());

        let n = values.len();
        let mean = values.iter().sum::<f64>() / n as f64;

        // Calculate variance for constant signal detection
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / n as f64;

        // Compute autocorrelation for different lags
        let max_lag = n / 4; // Only compute for first quarter
        let mut time_lags = Vec::new();
        let mut corr_values = Vec::new();

        // Special case: constant signal (zero variance)
        if variance < 1e-15 {
            // Constant signal has perfect correlation at all lags
            for lag in 0..max_lag {
                time_lags.push(times[lag] - times[0]);
                corr_values.push(1.0);
            }
        } else {
            // Normal autocorrelation calculation
            for lag in 0..max_lag {
                let mut sum = 0.0;
                let mut count = 0;

                for i in 0..n-lag {
                    sum += (values[i] - mean) * (values[i + lag] - mean);
                    count += 1;
                }

                if count > 0 {
                    let corr = sum / (count as f64);
                    time_lags.push(times[lag] - times[0]);
                    corr_values.push(corr);
                }
            }

            // Normalize by C(0)
            if !corr_values.is_empty() && corr_values[0].abs() > 1e-10 {
                let c0 = corr_values[0];
                for c in &mut corr_values {
                    *c /= c0;
                }
            }
        }

        // Estimate correlation length from exponential fit
        let correlation_length = Self::estimate_correlation_length(&time_lags, &corr_values);

        Self {
            time_lags,
            values: corr_values,
            correlation_length,
        }
    }

    /// Estimate correlation length from exponential decay
    fn estimate_correlation_length(times: &[f64], values: &[f64]) -> Option<f64> {
        // Find where correlation drops to 1/e
        let threshold = 1.0 / std::f64::consts::E;

        for i in 1..values.len() {
            if values[i] < threshold {
                // Linear interpolation
                let t0 = times[i-1];
                let t1 = times[i];
                let c0 = values[i-1];
                let c1 = values[i];

                let alpha = (threshold - c0) / (c1 - c0);
                let tau = t0 + alpha * (t1 - t0);

                return Some(tau);
            }
        }

        None
    }

    /// Check if system has long-range correlations
    pub fn has_long_range_correlations(&self) -> bool {
        if let Some(tau) = self.correlation_length {
            let total_time = self.time_lags.last().copied().unwrap_or(0.0);
            tau > 0.5 * total_time
        } else {
            // Didn't decay to 1/e in measured range
            true
        }
    }
}

/// Collection of observables over time
#[derive(Debug, Clone)]
pub struct ObservableTimeSeries {
    /// Time points
    pub times: Vec<f64>,

    /// Observable values at each time
    pub values: HashMap<String, Vec<f64>>,
}

impl ObservableTimeSeries {
    /// Create new time series
    pub fn new() -> Self {
        Self {
            times: Vec::new(),
            values: HashMap::new(),
        }
    }

    /// Add time point with observable values
    pub fn add_point(&mut self, time: f64, observables: HashMap<String, f64>) {
        self.times.push(time);

        for (name, value) in observables {
            self.values.entry(name)
                .or_insert_with(Vec::new)
                .push(value);
        }
    }

    /// Get observable statistics
    pub fn statistics(&self, name: &str) -> Option<Observable> {
        self.values.get(name)
            .map(|vals| Observable::from_samples(vals))
    }

    /// Compute correlation function for observable
    pub fn correlation(&self, name: &str) -> Option<Correlation> {
        self.values.get(name)
            .map(|vals| Correlation::from_time_series(&self.times, vals))
    }

    /// Compute cross-correlation between two observables
    pub fn cross_correlation(&self, name1: &str, name2: &str) -> Option<Vec<f64>> {
        let vals1 = self.values.get(name1)?;
        let vals2 = self.values.get(name2)?;

        if vals1.len() != vals2.len() {
            return None;
        }

        let n = vals1.len();
        let mean1 = vals1.iter().sum::<f64>() / n as f64;
        let mean2 = vals2.iter().sum::<f64>() / n as f64;

        let max_lag = n / 4;
        let mut cross_corr = Vec::new();

        for lag in 0..max_lag {
            let mut sum = 0.0;
            let mut count = 0;

            for i in 0..n-lag {
                sum += (vals1[i] - mean1) * (vals2[i + lag] - mean2);
                count += 1;
            }

            if count > 0 {
                cross_corr.push(sum / count as f64);
            }
        }

        Some(cross_corr)
    }

    /// Check if system has reached steady state
    pub fn is_steady_state(&self, name: &str, window: usize) -> bool {
        if let Some(vals) = self.values.get(name) {
            if vals.len() < 2 * window {
                return false;
            }

            // Compare last window to previous window
            let recent: Vec<f64> = vals.iter().rev().take(window).copied().collect();
            let previous: Vec<f64> = vals.iter().rev().skip(window).take(window).copied().collect();

            let recent_mean = recent.iter().sum::<f64>() / window as f64;
            let prev_mean = previous.iter().sum::<f64>() / window as f64;

            // Check if means are similar (within 1%)
            (recent_mean - prev_mean).abs() / recent_mean.abs().max(1e-10) < 0.01
        } else {
            false
        }
    }
}

impl Default for ObservableTimeSeries {
    fn default() -> Self {
        Self::new()
    }
}

/// Spin correlation functions
pub mod spin {
    use super::Observable;

    /// Compute spin-spin correlation ⟨s_i s_j⟩
    pub fn spin_correlation(states: &[u32]) -> Vec<f64> {
        let n = states.len();
        let mut correlations = vec![0.0; n];

        // Convert {0,1} to {-1,+1}
        let spins: Vec<f64> = states.iter()
            .map(|&s| 2.0 * s as f64 - 1.0)
            .collect();

        // Compute correlation with first spin
        let s0 = spins[0];
        for (i, corr) in correlations.iter_mut().enumerate() {
            *corr = s0 * spins[i];
        }

        correlations
    }

    /// Compute magnetization M = Σ s_i / N
    pub fn magnetization(states: &[u32]) -> f64 {
        let sum: i32 = states.iter().map(|&s| 2 * s as i32 - 1).sum();
        sum as f64 / states.len() as f64
    }

    /// Compute susceptibility χ = β (⟨M²⟩ - ⟨M⟩²)
    pub fn susceptibility(magnetizations: &[f64], beta: f64) -> f64 {
        let m_avg = magnetizations.iter().sum::<f64>() / magnetizations.len() as f64;
        let m2_avg = magnetizations.iter().map(|m| m * m).sum::<f64>() / magnetizations.len() as f64;
        beta * (m2_avg - m_avg * m_avg)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    #[test]
    fn test_observable_from_samples() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let obs = Observable::from_samples(&samples);

        assert!((obs.expectation - 3.0).abs() < 1e-10);
        assert!(obs.variance > 0.0);
        assert_eq!(obs.n_samples, 5);
    }

    #[test]
    fn test_relative_error() {
        let samples = vec![10.0, 11.0, 9.0, 10.5, 9.5];
        let obs = Observable::from_samples(&samples);

        let rel_err = obs.relative_error();
        assert!(rel_err < 0.1); // Should be small for these samples
    }

    #[test]
    fn test_correlation_from_constant() {
        let times: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let values = vec![1.0; 100];

        let corr = Correlation::from_time_series(&times, &values);

        // Constant signal should have perfect correlation
        assert!((corr.values[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_time_series() {
        let mut ts = ObservableTimeSeries::new();

        for i in 0..10 {
            let mut obs = HashMap::new();
            obs.insert("energy".to_string(), i as f64);
            obs.insert("entropy".to_string(), (i as f64).sqrt());
            ts.add_point(i as f64, obs);
        }

        let energy_stats = ts.statistics("energy").unwrap();
        assert!((energy_stats.expectation - 4.5).abs() < 0.1);
    }

    #[test]
    fn test_spin_magnetization() {
        let all_up = vec![1; 10];
        let m = spin::magnetization(&all_up);
        assert!((m - 1.0).abs() < 1e-10);

        let all_down = vec![0; 10];
        let m = spin::magnetization(&all_down);
        assert!((m + 1.0).abs() < 1e-10);

        let balanced = vec![1, 0, 1, 0, 1, 0];
        let m = spin::magnetization(&balanced);
        assert!(m.abs() < 1e-10);
    }
}
