//! Inference and prediction utilities

use serde::{Deserialize, Serialize};
use crate::Result;

/// Prediction intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionIntervals {
    /// Point predictions
    pub point_forecast: Vec<f64>,

    /// Lower bounds for each confidence level
    pub lower_bounds: Vec<Vec<f64>>,

    /// Upper bounds for each confidence level
    pub upper_bounds: Vec<Vec<f64>>,

    /// Confidence levels (e.g., [0.5, 0.8, 0.95])
    pub levels: Vec<f64>,
}

impl PredictionIntervals {
    pub fn new(
        point_forecast: Vec<f64>,
        lower_bounds: Vec<Vec<f64>>,
        upper_bounds: Vec<Vec<f64>>,
        levels: Vec<f64>,
    ) -> Self {
        Self {
            point_forecast,
            lower_bounds,
            upper_bounds,
            levels,
        }
    }

    /// Create simple intervals using standard deviation
    pub fn from_std(
        point_forecast: Vec<f64>,
        std_dev: Vec<f64>,
        levels: Vec<f64>,
    ) -> Self {
        let mut lower_bounds = Vec::new();
        let mut upper_bounds = Vec::new();

        for &level in &levels {
            // For 95% CI, z ≈ 1.96, for 80% CI, z ≈ 1.28, etc.
            let z = normal_ppf((1.0 + level) / 2.0);

            let lower: Vec<f64> = point_forecast.iter()
                .zip(&std_dev)
                .map(|(&p, &s)| p - z * s)
                .collect();

            let upper: Vec<f64> = point_forecast.iter()
                .zip(&std_dev)
                .map(|(&p, &s)| p + z * s)
                .collect();

            lower_bounds.push(lower);
            upper_bounds.push(upper);
        }

        Self::new(point_forecast, lower_bounds, upper_bounds, levels)
    }
}

/// Approximate inverse of standard normal CDF
fn normal_ppf(p: f64) -> f64 {
    // Rational approximation (Beasley-Springer-Moro algorithm)
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    ];

    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];

    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    ];

    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Lower tail
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Upper tail
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_intervals() {
        let forecast = vec![1.0, 2.0, 3.0];
        let std_dev = vec![0.1, 0.1, 0.1];
        let levels = vec![0.95];

        let intervals = PredictionIntervals::from_std(forecast.clone(), std_dev, levels);

        assert_eq!(intervals.point_forecast, forecast);
        assert_eq!(intervals.lower_bounds.len(), 1);
        assert_eq!(intervals.upper_bounds.len(), 1);
        assert_eq!(intervals.levels, vec![0.95]);
    }

    #[test]
    fn test_normal_ppf() {
        // Test some known values
        let val1 = normal_ppf(0.5);
        let val2 = normal_ppf(0.975);

        assert!((val1 - 0.0).abs() < 0.01, "normal_ppf(0.5) should be ~0.0, got {}", val1);
        assert!((val2 - 1.96).abs() < 0.01, "normal_ppf(0.975) should be ~1.96, got {}", val2);
    }
}
