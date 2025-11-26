//! Test helper utilities for neural model testing

use ndarray::{Array1, Array2};
use rand::Rng;

/// Generate synthetic time series data with various patterns
pub mod synthetic {
    use super::*;

    /// Generate sine wave with configurable amplitude and frequency
    pub fn sine_wave(length: usize, frequency: f64, amplitude: f64, offset: f64) -> Vec<f64> {
        (0..length)
            .map(|i| {
                let t = i as f64;
                amplitude * (2.0 * std::f64::consts::PI * t * frequency).sin() + offset
            })
            .collect()
    }

    /// Generate linear trend
    pub fn linear_trend(length: usize, slope: f64, intercept: f64) -> Vec<f64> {
        (0..length)
            .map(|i| slope * i as f64 + intercept)
            .collect()
    }

    /// Generate random walk
    pub fn random_walk(length: usize, step_size: f64, start: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut value = start;
        let mut series = Vec::with_capacity(length);

        for _ in 0..length {
            series.push(value);
            value += rng.gen_range(-step_size..step_size);
        }

        series
    }

    /// Generate complex series with trend + seasonality + noise
    pub fn complex_series(
        length: usize,
        trend: f64,
        seasonality_period: usize,
        noise_level: f64,
    ) -> Vec<f64> {
        let mut rng = rand::thread_rng();

        (0..length)
            .map(|i| {
                let t = i as f64;
                let trend_component = trend * t;
                let seasonal_component = 10.0 * (2.0 * std::f64::consts::PI * t / seasonality_period as f64).sin();
                let noise = rng.gen::<f64>() * noise_level;

                trend_component + seasonal_component + noise
            })
            .collect()
    }

    /// Generate autoregressive series AR(1)
    pub fn ar1_series(length: usize, phi: f64, sigma: f64, start: f64) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        let mut series = Vec::with_capacity(length);
        let mut current = start;

        for _ in 0..length {
            series.push(current);
            let noise: f64 = rng.gen_range(-sigma..sigma);
            current = phi * current + noise;
        }

        series
    }
}

/// Numerical gradient checking utilities
pub mod gradient_check {
    use super::*;

    /// Compute numerical gradient using finite differences
    pub fn numerical_gradient<F>(
        f: F,
        x: &Array1<f64>,
        epsilon: f64,
    ) -> Array1<f64>
    where
        F: Fn(&Array1<f64>) -> f64,
    {
        let mut grad = Array1::zeros(x.len());

        for i in 0..x.len() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();

            x_plus[i] += epsilon;
            x_minus[i] -= epsilon;

            let f_plus = f(&x_plus);
            let f_minus = f(&x_minus);

            grad[i] = (f_plus - f_minus) / (2.0 * epsilon);
        }

        grad
    }

    /// Check if analytical and numerical gradients match within tolerance
    pub fn gradients_match(
        analytical: &Array1<f64>,
        numerical: &Array1<f64>,
        rtol: f64,
        atol: f64,
    ) -> bool {
        if analytical.len() != numerical.len() {
            return false;
        }

        for i in 0..analytical.len() {
            let diff = (analytical[i] - numerical[i]).abs();
            let threshold = atol + rtol * numerical[i].abs();

            if diff > threshold {
                eprintln!(
                    "Gradient mismatch at index {}: analytical={}, numerical={}, diff={}",
                    i, analytical[i], numerical[i], diff
                );
                return false;
            }
        }

        true
    }
}

/// Model testing utilities
pub mod model_testing {
    use super::*;

    /// Test if model can overfit a small dataset (sanity check)
    pub fn can_overfit(initial_loss: f64, final_loss: f64, threshold: f64) -> bool {
        initial_loss - final_loss > threshold
    }

    /// Test if loss is decreasing monotonically
    pub fn loss_decreasing(history: &[f64]) -> bool {
        if history.len() < 2 {
            return true;
        }

        for i in 1..history.len() {
            if history[i] > history[i - 1] {
                return false;
            }
        }

        true
    }

    /// Check if predictions are finite (no NaN or Inf)
    pub fn predictions_finite(predictions: &[f64]) -> bool {
        predictions.iter().all(|&x| x.is_finite())
    }

    /// Check if prediction intervals are properly ordered
    pub fn intervals_ordered(lower: &[f64], mean: &[f64], upper: &[f64]) -> bool {
        if lower.len() != mean.len() || mean.len() != upper.len() {
            return false;
        }

        for i in 0..lower.len() {
            if lower[i] > mean[i] || mean[i] > upper[i] {
                eprintln!(
                    "Interval ordering violated at {}: lower={}, mean={}, upper={}",
                    i, lower[i], mean[i], upper[i]
                );
                return false;
            }
        }

        true
    }

    /// Compute mean absolute percentage error
    pub fn mape(predictions: &[f64], actuals: &[f64]) -> f64 {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return f64::INFINITY;
        }

        let sum: f64 = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(&pred, &actual)| {
                if actual.abs() < 1e-10 {
                    0.0
                } else {
                    ((pred - actual) / actual).abs()
                }
            })
            .sum();

        sum / predictions.len() as f64
    }

    /// Compute root mean squared error
    pub fn rmse(predictions: &[f64], actuals: &[f64]) -> f64 {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return f64::INFINITY;
        }

        let mse: f64 = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<f64>() / predictions.len() as f64;

        mse.sqrt()
    }
}

/// Performance testing utilities
pub mod performance {
    use std::time::Instant;

    /// Measure execution time of a function
    pub fn time_execution<F, R>(f: F) -> (R, std::time::Duration)
    where
        F: FnOnce() -> R,
    {
        let start = Instant::now();
        let result = f();
        let duration = start.elapsed();
        (result, duration)
    }

    /// Check if execution time is within expected bounds
    pub fn within_time_budget(duration: std::time::Duration, budget_ms: u64) -> bool {
        duration.as_millis() as u64 <= budget_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_wave_generation() {
        let series = synthetic::sine_wave(100, 0.1, 1.0, 0.0);
        assert_eq!(series.len(), 100);
        assert!(series.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_gradient_check() {
        use ndarray::arr1;

        // Simple quadratic function: f(x) = x^2
        let f = |x: &Array1<f64>| x[0].powi(2);
        let x = arr1(&[2.0]);

        let numerical = gradient_check::numerical_gradient(f, &x, 1e-5);
        let analytical = arr1(&[4.0]); // Derivative of x^2 is 2x

        assert!(gradient_check::gradients_match(
            &analytical,
            &numerical,
            1e-3,
            1e-5
        ));
    }

    #[test]
    fn test_mape_calculation() {
        let predictions = vec![10.0, 20.0, 30.0];
        let actuals = vec![11.0, 19.0, 31.0];

        let error = model_testing::mape(&predictions, &actuals);
        assert!(error > 0.0 && error < 0.1);
    }
}
