//! Evaluation metrics for time-series forecasting

use crate::error::MlResult;
use crate::tensor::{Tensor, TensorOps};

/// Forecasting metrics collection
#[derive(Debug, Clone, Default)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f32,
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Mean Absolute Percentage Error
    pub mape: f32,
    /// Symmetric Mean Absolute Percentage Error
    pub smape: f32,
    /// Mean Absolute Scaled Error
    pub mase: Option<f32>,
    /// R-squared (coefficient of determination)
    pub r2: f32,
}

impl ForecastMetrics {
    /// Compute all metrics from predictions and targets
    pub fn compute(
        predictions: &Tensor,
        targets: &Tensor,
        naive_forecast: Option<&Tensor>,
    ) -> MlResult<Self> {
        let mae = mean_absolute_error(predictions, targets)?;
        let mse = mean_squared_error(predictions, targets)?;
        let rmse = mse.sqrt();
        let mape = mean_absolute_percentage_error(predictions, targets)?;
        let smape = symmetric_mape(predictions, targets)?;
        let r2 = r_squared(predictions, targets)?;

        let mase = if let Some(naive) = naive_forecast {
            Some(mean_absolute_scaled_error(predictions, targets, naive)?)
        } else {
            None
        };

        Ok(Self {
            mae,
            mse,
            rmse,
            mape,
            smape,
            mase,
            r2,
        })
    }

    /// Format metrics as string
    pub fn summary(&self) -> String {
        let mut s = format!(
            "MAE: {:.4}, MSE: {:.4}, RMSE: {:.4}, MAPE: {:.2}%, sMAPE: {:.2}%, R²: {:.4}",
            self.mae, self.mse, self.rmse, self.mape * 100.0, self.smape * 100.0, self.r2
        );

        if let Some(mase) = self.mase {
            s.push_str(&format!(", MASE: {:.4}", mase));
        }

        s
    }
}

/// Mean Absolute Error (MAE)
/// MAE = (1/n) * Σ|y_pred - y_true|
pub fn mean_absolute_error(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            let n = pred_data.len() as f32;
            let sum: f32 = pred_data.iter()
                .zip(target_data.iter())
                .map(|(p, t)| (p - t).abs())
                .sum();
            return Ok(sum / n);
        }
    }
    Ok(0.0)
}

/// Mean Squared Error (MSE)
/// MSE = (1/n) * Σ(y_pred - y_true)²
pub fn mean_squared_error(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            let n = pred_data.len() as f32;
            let sum: f32 = pred_data.iter()
                .zip(target_data.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum();
            return Ok(sum / n);
        }
    }
    Ok(0.0)
}

/// Root Mean Squared Error (RMSE)
/// RMSE = sqrt(MSE)
pub fn root_mean_squared_error(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    Ok(mean_squared_error(predictions, targets)?.sqrt())
}

/// Mean Absolute Percentage Error (MAPE)
/// MAPE = (1/n) * Σ|y_pred - y_true| / |y_true|
///
/// Note: MAPE is undefined when y_true = 0
pub fn mean_absolute_percentage_error(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            let mut sum = 0.0_f32;
            let mut count = 0;

            for (p, t) in pred_data.iter().zip(target_data.iter()) {
                if t.abs() > 1e-10 {  // Avoid division by zero
                    sum += ((p - t) / t).abs();
                    count += 1;
                }
            }

            if count > 0 {
                return Ok(sum / count as f32);
            }
        }
    }
    Ok(0.0)
}

/// Symmetric Mean Absolute Percentage Error (sMAPE)
/// sMAPE = (2/n) * Σ|y_pred - y_true| / (|y_pred| + |y_true|)
///
/// More robust than MAPE when values are close to zero
pub fn symmetric_mape(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            let mut sum = 0.0_f32;
            let mut count = 0;

            for (p, t) in pred_data.iter().zip(target_data.iter()) {
                let denom = p.abs() + t.abs();
                if denom > 1e-10 {
                    sum += 2.0 * (p - t).abs() / denom;
                    count += 1;
                }
            }

            if count > 0 {
                return Ok(sum / count as f32);
            }
        }
    }
    Ok(0.0)
}

/// Mean Absolute Scaled Error (MASE)
/// MASE = MAE / MAE_naive
///
/// Compares forecast accuracy to naive seasonal forecast
pub fn mean_absolute_scaled_error(
    predictions: &Tensor,
    targets: &Tensor,
    naive: &Tensor,
) -> MlResult<f32> {
    let mae_model = mean_absolute_error(predictions, targets)?;
    let mae_naive = mean_absolute_error(naive, targets)?;

    if mae_naive > 1e-10 {
        Ok(mae_model / mae_naive)
    } else {
        Ok(0.0)
    }
}

/// R-squared (Coefficient of Determination)
/// R² = 1 - SS_res / SS_tot
/// where SS_res = Σ(y - y_pred)², SS_tot = Σ(y - y_mean)²
pub fn r_squared(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            let n = target_data.len() as f32;
            let mean: f32 = target_data.iter().sum::<f32>() / n;

            let ss_res: f32 = pred_data.iter()
                .zip(target_data.iter())
                .map(|(p, t)| (t - p).powi(2))
                .sum();

            let ss_tot: f32 = target_data.iter()
                .map(|t| (t - mean).powi(2))
                .sum();

            if ss_tot > 1e-10 {
                return Ok(1.0 - ss_res / ss_tot);
            }
        }
    }
    Ok(0.0)
}

/// Quantile Loss (Pinball Loss)
/// L_q(y, y_pred) = q * (y - y_pred)^+ + (1-q) * (y_pred - y)^+
pub fn quantile_loss(predictions: &Tensor, targets: &Tensor, quantile: f32) -> MlResult<f32> {
    assert!(quantile > 0.0 && quantile < 1.0, "Quantile must be in (0, 1)");

    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            let n = pred_data.len() as f32;
            let sum: f32 = pred_data.iter()
                .zip(target_data.iter())
                .map(|(p, t)| {
                    let error = t - p;
                    if error >= 0.0 {
                        quantile * error
                    } else {
                        (quantile - 1.0) * error
                    }
                })
                .sum();
            return Ok(sum / n);
        }
    }
    Ok(0.0)
}

/// Continuous Ranked Probability Score (CRPS)
/// Measures probabilistic forecast quality
/// For Gaussian: CRPS = σ * (z * (2*Φ(z) - 1) + 2*φ(z) - 1/√π)
/// where z = (y - μ) / σ
pub fn crps_gaussian(targets: &Tensor, mu: &Tensor, sigma: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(y), Some(m), Some(s)) = (targets.as_slice(), mu.as_slice(), sigma.as_slice()) {
            let n = y.len() as f32;
            let inv_sqrt_pi = 1.0 / std::f32::consts::PI.sqrt();

            let sum: f32 = y.iter()
                .zip(m.iter())
                .zip(s.iter())
                .map(|((yi, mi), si)| {
                    let z = (yi - mi) / si;
                    let phi_z = standard_normal_cdf(z);
                    let pdf_z = standard_normal_pdf(z);
                    si * (z * (2.0 * phi_z - 1.0) + 2.0 * pdf_z - inv_sqrt_pi)
                })
                .sum();

            return Ok(sum / n);
        }
    }
    Ok(0.0)
}

/// Direction Accuracy (for price movement prediction)
/// Percentage of correctly predicted directions
pub fn direction_accuracy(predictions: &Tensor, targets: &Tensor) -> MlResult<f32> {
    #[cfg(feature = "cpu")]
    {
        if let (Some(pred_data), Some(target_data)) = (predictions.as_slice(), targets.as_slice()) {
            if pred_data.len() < 2 {
                return Ok(0.0);
            }

            let mut correct = 0;
            let total = pred_data.len() - 1;

            for i in 1..pred_data.len() {
                let pred_dir = pred_data[i] - pred_data[i-1];
                let true_dir = target_data[i] - target_data[i-1];

                // Same direction (both positive, both negative, or both zero)
                if pred_dir * true_dir > 0.0 || (pred_dir.abs() < 1e-10 && true_dir.abs() < 1e-10) {
                    correct += 1;
                }
            }

            return Ok(correct as f32 / total as f32);
        }
    }
    Ok(0.0)
}

// Helper functions

fn standard_normal_cdf(x: f32) -> f32 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / std::f32::consts::SQRT_2))
}

fn standard_normal_pdf(x: f32) -> f32 {
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
    inv_sqrt_2pi * (-0.5 * x * x).exp()
}

fn erf(x: f32) -> f32 {
    // Horner form approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::Device;

    #[test]
    fn test_mae() {
        let device = Device::Cpu;
        let pred = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3], &device).unwrap();
        let target = Tensor::from_slice(&[1.5, 2.5, 2.5], vec![3], &device).unwrap();

        let mae = mean_absolute_error(&pred, &target).unwrap();
        // |1-1.5| + |2-2.5| + |3-2.5| = 0.5 + 0.5 + 0.5 = 1.5 / 3 = 0.5
        assert!((mae - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_mse() {
        let device = Device::Cpu;
        let pred = Tensor::from_slice(&[1.0, 2.0], vec![2], &device).unwrap();
        let target = Tensor::from_slice(&[2.0, 3.0], vec![2], &device).unwrap();

        let mse = mean_squared_error(&pred, &target).unwrap();
        // (1-2)² + (2-3)² = 1 + 1 = 2 / 2 = 1.0
        assert!((mse - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_r_squared() {
        let device = Device::Cpu;
        // Perfect prediction
        let target = Tensor::from_slice(&[1.0, 2.0, 3.0], vec![3], &device).unwrap();
        let pred = target.clone();

        let r2 = r_squared(&pred, &target).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantile_loss() {
        let device = Device::Cpu;
        let pred = Tensor::from_slice(&[2.0], vec![1], &device).unwrap();
        let target = Tensor::from_slice(&[3.0], vec![1], &device).unwrap();

        // At q=0.5, this should equal MAE/2
        let loss = quantile_loss(&pred, &target, 0.5).unwrap();
        assert!((loss - 0.5).abs() < 1e-6);
    }
}
