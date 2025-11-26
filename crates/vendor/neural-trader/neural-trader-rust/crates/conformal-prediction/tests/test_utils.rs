//! Test Utilities and Data Generators
//!
//! Provides helper functions for generating test data and performing statistical tests

use std::f64::consts::PI;

/// Data distribution types for synthetic data generation
#[derive(Debug, Clone, Copy)]
pub enum DataDistribution {
    /// Normal distribution with mean and std dev
    Normal(f64, f64),

    /// Linear relationship: y = slope * x + intercept
    Linear(f64, f64),

    /// Uniform distribution in range [min, max]
    Uniform(f64, f64),
}

/// Market regime for market data simulation
#[derive(Debug, Clone, Copy)]
pub enum MarketRegime {
    Bull,
    Bear,
    Sideways,
}

/// Generate synthetic data with specified distribution
pub fn generate_synthetic_data(
    n: usize,
    dist: DataDistribution,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    match dist {
        DataDistribution::Normal(mean, std_dev) => {
            let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    // Box-Muller transform for normal distribution
                    let u1: f64 = (i as f64 + 1.0) / (n as f64 + 1.0);
                    let u2: f64 = ((i * 7 + 3) % n) as f64 / n as f64;
                    mean + std_dev * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
                })
                .collect();
            (x, y)
        }

        DataDistribution::Linear(slope, intercept) => {
            let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| slope * (i as f64) + intercept)
                .collect();
            (x, y)
        }

        DataDistribution::Uniform(min, max) => {
            let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let y: Vec<f64> = (0..n)
                .map(|i| {
                    let t = (i as f64) / (n as f64);
                    min + (max - min) * t
                })
                .collect();
            (x, y)
        }
    }
}

/// Generate bimodal data (two clusters)
pub fn generate_bimodal_data(
    n: usize,
    mode1: (f64, f64), // (mean, std_dev)
    mode2: (f64, f64), // (mean, std_dev)
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();

    let n1 = n / 2;
    let n2 = n - n1;

    // First mode
    for i in 0..n1 {
        let u1: f64 = (i as f64 + 1.0) / (n1 as f64 + 1.0);
        let u2: f64 = ((i * 7 + 3) % n1) as f64 / n1 as f64;
        let val = mode1.0 + mode1.1 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        x.push(vec![val]);
        y.push(val);
    }

    // Second mode
    for i in 0..n2 {
        let u1: f64 = (i as f64 + 1.0) / (n2 as f64 + 1.0);
        let u2: f64 = ((i * 11 + 5) % n2) as f64 / n2 as f64;
        let val = mode2.0 + mode2.1 * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

        x.push(vec![val]);
        y.push(val);
    }

    (x, y)
}

/// Generate trimodal data (three clusters)
pub fn generate_trimodal_data(
    n: usize,
    mode1: (f64, f64),
    mode2: (f64, f64),
    mode3: (f64, f64),
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();

    let n1 = n / 3;
    let n2 = n / 3;
    let n3 = n - n1 - n2;

    // Helper to generate normal samples
    let mut generate_mode = |count: usize, mean: f64, std_dev: f64, seed: usize| {
        for i in 0..count {
            let u1: f64 = (i as f64 + 1.0) / (count as f64 + 1.0);
            let u2: f64 = ((i * seed + 3) % count) as f64 / count as f64;
            let val = mean + std_dev * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

            x.push(vec![val]);
            y.push(val);
        }
    };

    generate_mode(n1, mode1.0, mode1.1, 7);
    generate_mode(n2, mode2.0, mode2.1, 11);
    generate_mode(n3, mode3.0, mode3.1, 13);

    (x, y)
}

/// Generate market-like data with regime-specific characteristics
pub fn generate_market_data(
    n: usize,
    regime: MarketRegime,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::new();
    let mut y = Vec::new();

    let mut price = 100.0;

    match regime {
        MarketRegime::Bull => {
            // Bullish trend: upward drift with moderate volatility
            for i in 0..n {
                let t = i as f64 / n as f64;
                let drift = 0.05; // 5% daily drift
                let vol = 0.02;   // 2% volatility

                let u1: f64 = (i as f64 + 1.0) / (n as f64 + 1.0);
                let u2: f64 = ((i * 7 + 3) % n) as f64 / n as f64;
                let shock = vol * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

                price *= 1.0 + drift + shock;
                x.push(vec![t]);
                y.push(price);
            }
        }

        MarketRegime::Bear => {
            // Bearish trend: downward drift with high volatility
            for i in 0..n {
                let t = i as f64 / n as f64;
                let drift = -0.03; // -3% daily drift
                let vol = 0.03;    // 3% volatility

                let u1: f64 = (i as f64 + 1.0) / (n as f64 + 1.0);
                let u2: f64 = ((i * 11 + 5) % n) as f64 / n as f64;
                let shock = vol * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

                price *= 1.0 + drift + shock;
                x.push(vec![t]);
                y.push(price);
            }
        }

        MarketRegime::Sideways => {
            // Sideways: mean-reverting with low drift
            for i in 0..n {
                let t = i as f64 / n as f64;
                let mean_reversion = -0.1 * (price - 100.0) / 100.0;
                let vol = 0.015;

                let u1: f64 = (i as f64 + 1.0) / (n as f64 + 1.0);
                let u2: f64 = ((i * 13 + 7) % n) as f64 / n as f64;
                let shock = vol * (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();

                price *= 1.0 + mean_reversion + shock;
                x.push(vec![t]);
                y.push(price);
            }
        }
    }

    (x, y)
}

/// Kolmogorov-Smirnov test for uniformity
///
/// Tests if the empirical CDF of data matches Uniform(0,1)
/// Returns the KS statistic: max|F_empirical(x) - F_uniform(x)|
pub fn kolmogorov_smirnov_uniform(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    // Sort data
    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted_data.len() as f64;
    let mut max_diff: f64 = 0.0;

    for (i, &value) in sorted_data.iter().enumerate() {
        // Empirical CDF at this point
        let empirical_cdf = (i + 1) as f64 / n;

        // Uniform CDF is just the value itself (for [0,1])
        let uniform_cdf = value.clamp(0.0, 1.0);

        // Compute difference
        let diff = (empirical_cdf - uniform_cdf).abs();

        // Also check at the previous step
        let empirical_cdf_prev = i as f64 / n;
        let diff_prev = (empirical_cdf_prev - uniform_cdf).abs();

        max_diff = max_diff.max(diff).max(diff_prev);
    }

    max_diff
}

/// Compute empirical coverage
pub fn compute_coverage(
    intervals: &[(f64, f64)],
    true_values: &[f64],
) -> f64 {
    if intervals.len() != true_values.len() {
        return 0.0;
    }

    let covered = intervals
        .iter()
        .zip(true_values.iter())
        .filter(|((lower, upper), &y)| *lower <= y && y <= *upper)
        .count();

    covered as f64 / intervals.len() as f64
}

/// Compute mean absolute error
pub fn mean_absolute_error(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() || predictions.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| (pred - actual).abs())
        .sum();

    sum / predictions.len() as f64
}

/// Compute mean squared error
pub fn mean_squared_error(predictions: &[f64], actuals: &[f64]) -> f64 {
    if predictions.len() != actuals.len() || predictions.is_empty() {
        return f64::INFINITY;
    }

    let sum: f64 = predictions
        .iter()
        .zip(actuals.iter())
        .map(|(pred, actual)| (pred - actual).powi(2))
        .sum();

    sum / predictions.len() as f64
}

/// Compute average interval width
pub fn average_interval_width(intervals: &[(f64, f64)]) -> f64 {
    if intervals.is_empty() {
        return 0.0;
    }

    let sum: f64 = intervals.iter().map(|(lower, upper)| upper - lower).sum();
    sum / intervals.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_synthetic_data() {
        let (x, y) = generate_synthetic_data(100, DataDistribution::Linear(2.0, 1.0));

        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);

        // Check linear relationship
        assert!((y[0] - 1.0).abs() < 0.1);
        assert!((y[50] - 101.0).abs() < 0.1);
    }

    #[test]
    fn test_generate_bimodal_data() {
        let (x, y) = generate_bimodal_data(100, (-5.0, 0.5), (5.0, 0.5));

        assert_eq!(x.len(), 100);
        assert_eq!(y.len(), 100);
    }

    #[test]
    fn test_kolmogorov_smirnov_uniform() {
        // Perfect uniform data
        let uniform_data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let ks = kolmogorov_smirnov_uniform(&uniform_data);

        // Should be very close to 0 for perfect uniform
        assert!(ks < 0.05);
    }

    #[test]
    fn test_compute_coverage() {
        let intervals = vec![(0.0, 2.0), (1.0, 3.0), (2.0, 4.0)];
        let true_values = vec![1.0, 2.0, 5.0]; // Third one is not covered

        let coverage = compute_coverage(&intervals, &true_values);
        assert!((coverage - 2.0/3.0).abs() < 0.01);
    }
}
