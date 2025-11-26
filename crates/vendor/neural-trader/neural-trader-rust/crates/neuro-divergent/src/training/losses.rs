//! Loss functions for neural network training
//!
//! Comprehensive suite of loss functions for time series forecasting:
//! - Regression losses (MSE, MAE, Huber)
//! - Probabilistic losses (Quantile, Pinball)
//! - Percentage errors (MAPE, SMAPE)
//! - Custom weighted losses

use ndarray::{Array1, Array2, Axis};
use crate::{Result, NeuroDivergentError};

/// Loss function trait
pub trait LossFunction: Send + Sync {
    /// Compute loss
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64>;

    /// Compute gradient of loss w.r.t. predictions
    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>>;

    /// Loss function name
    fn name(&self) -> &str;
}

/// Mean Squared Error (MSE) loss
#[derive(Debug, Clone)]
pub struct MSELoss;

impl LossFunction for MSELoss {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let diff = predictions - targets;
        Ok(diff.mapv(|x| x.powi(2)).mean().unwrap_or(0.0))
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let n = predictions.len() as f64;
        Ok((predictions - targets) * (2.0 / n))
    }

    fn name(&self) -> &str {
        "MSE"
    }
}

/// Mean Absolute Error (MAE) loss
#[derive(Debug, Clone)]
pub struct MAELoss;

impl LossFunction for MAELoss {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let diff = predictions - targets;
        Ok(diff.mapv(|x| x.abs()).mean().unwrap_or(0.0))
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let n = predictions.len() as f64;
        Ok((predictions - targets).mapv(|x| x.signum()) * (1.0 / n))
    }

    fn name(&self) -> &str {
        "MAE"
    }
}

/// Huber loss - combines MSE and MAE
#[derive(Debug, Clone)]
pub struct HuberLoss {
    delta: f64,
}

impl HuberLoss {
    pub fn new(delta: f64) -> Self {
        Self { delta }
    }
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl LossFunction for HuberLoss {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let diff = predictions - targets;
        let loss = diff.mapv(|x| {
            let abs_x = x.abs();
            if abs_x <= self.delta {
                0.5 * x.powi(2)
            } else {
                self.delta * (abs_x - 0.5 * self.delta)
            }
        });

        Ok(loss.mean().unwrap_or(0.0))
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let n = predictions.len() as f64;
        let diff = predictions - targets;

        Ok(diff.mapv(|x| {
            if x.abs() <= self.delta {
                x
            } else {
                self.delta * x.signum()
            }
        }) / n)
    }

    fn name(&self) -> &str {
        "Huber"
    }
}

/// Quantile loss for probabilistic forecasting
#[derive(Debug, Clone)]
pub struct QuantileLoss {
    quantile: f64,
}

impl QuantileLoss {
    pub fn new(quantile: f64) -> Result<Self> {
        if quantile <= 0.0 || quantile >= 1.0 {
            return Err(NeuroDivergentError::TrainingError(
                "Quantile must be between 0 and 1".to_string()
            ));
        }
        Ok(Self { quantile })
    }
}

impl LossFunction for QuantileLoss {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let errors = targets - predictions;
        let loss = errors.mapv(|e| {
            if e >= 0.0 {
                self.quantile * e
            } else {
                (self.quantile - 1.0) * e
            }
        });

        Ok(loss.mean().unwrap_or(0.0))
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let n = predictions.len() as f64;
        let errors = targets - predictions;

        Ok(errors.mapv(|e| {
            if e >= 0.0 {
                -self.quantile
            } else {
                -(self.quantile - 1.0)
            }
        }) / n)
    }

    fn name(&self) -> &str {
        "Quantile"
    }
}

/// Mean Absolute Percentage Error (MAPE)
#[derive(Debug, Clone)]
pub struct MAPELoss {
    epsilon: f64,
}

impl MAPELoss {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for MAPELoss {
    fn default() -> Self {
        Self::new(1e-8)
    }
}

impl LossFunction for MAPELoss {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let mut total = 0.0;
        let mut count = 0;

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if target.abs() > self.epsilon {
                total += ((target - pred) / target).abs();
                count += 1;
            }
        }

        if count > 0 {
            Ok(100.0 * total / count as f64)
        } else {
            Ok(0.0)
        }
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let n = predictions.len() as f64;
        let grad = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                if target.abs() > self.epsilon {
                    -100.0 / (n * target)
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>();

        Ok(Array1::from_vec(grad))
    }

    fn name(&self) -> &str {
        "MAPE"
    }
}

/// Symmetric Mean Absolute Percentage Error (SMAPE)
#[derive(Debug, Clone)]
pub struct SMAPELoss {
    epsilon: f64,
}

impl SMAPELoss {
    pub fn new(epsilon: f64) -> Self {
        Self { epsilon }
    }
}

impl Default for SMAPELoss {
    fn default() -> Self {
        Self::new(1e-8)
    }
}

impl LossFunction for SMAPELoss {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let smape = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let numerator = (pred - target).abs();
                let denominator = (pred.abs() + target.abs() + self.epsilon);
                200.0 * numerator / denominator
            })
            .sum::<f64>() / predictions.len() as f64;

        Ok(smape)
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions and targets must have same length".to_string()
            ));
        }

        let n = predictions.len() as f64;
        let grad = predictions.iter()
            .zip(targets.iter())
            .map(|(pred, target)| {
                let abs_sum = pred.abs() + target.abs() + self.epsilon;
                let sign = if pred > target { 1.0 } else { -1.0 };
                200.0 * sign / (n * abs_sum)
            })
            .collect::<Vec<_>>();

        Ok(Array1::from_vec(grad))
    }

    fn name(&self) -> &str {
        "SMAPE"
    }
}

/// Weighted loss - applies per-sample weights
#[derive(Debug, Clone)]
pub struct WeightedLoss<L: LossFunction> {
    base_loss: L,
    weights: Array1<f64>,
}

impl<L: LossFunction> WeightedLoss<L> {
    pub fn new(base_loss: L, weights: Array1<f64>) -> Self {
        Self { base_loss, weights }
    }
}

impl<L: LossFunction> LossFunction for WeightedLoss<L> {
    fn forward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<f64> {
        if predictions.len() != targets.len() || predictions.len() != self.weights.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions, targets, and weights must have same length".to_string()
            ));
        }

        // Compute per-sample losses
        let per_sample_loss = predictions.iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                let pred = Array1::from_vec(vec![*p]);
                let targ = Array1::from_vec(vec![*t]);
                self.base_loss.forward(&pred, &targ).unwrap_or(0.0)
            })
            .collect::<Vec<_>>();

        let weighted_loss = per_sample_loss.iter()
            .zip(self.weights.iter())
            .map(|(loss, weight)| loss * weight)
            .sum::<f64>();

        Ok(weighted_loss / self.weights.sum())
    }

    fn backward(&self, predictions: &Array1<f64>, targets: &Array1<f64>) -> Result<Array1<f64>> {
        if predictions.len() != targets.len() || predictions.len() != self.weights.len() {
            return Err(NeuroDivergentError::TrainingError(
                "Predictions, targets, and weights must have same length".to_string()
            ));
        }

        let base_grad = self.base_loss.backward(predictions, targets)?;
        let weighted_grad = &base_grad * &self.weights / self.weights.sum();

        Ok(weighted_grad)
    }

    fn name(&self) -> &str {
        "Weighted"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::arr1;

    #[test]
    fn test_mse_loss() {
        let loss = MSELoss;
        let pred = arr1(&[1.0, 2.0, 3.0]);
        let target = arr1(&[1.5, 2.5, 3.5]);

        let loss_val = loss.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss_val, 0.25, epsilon = 1e-10);

        let grad = loss.backward(&pred, &target).unwrap();
        let expected_grad = arr1(&[-1.0/3.0, -1.0/3.0, -1.0/3.0]);
        for (g, e) in grad.iter().zip(expected_grad.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_mae_loss() {
        let loss = MAELoss;
        let pred = arr1(&[1.0, 2.0, 3.0]);
        let target = arr1(&[1.5, 2.5, 3.5]);

        let loss_val = loss.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss_val, 0.5, epsilon = 1e-10);

        let grad = loss.backward(&pred, &target).unwrap();
        let expected_grad = arr1(&[-1.0/3.0, -1.0/3.0, -1.0/3.0]);
        for (g, e) in grad.iter().zip(expected_grad.iter()) {
            assert_relative_eq!(g, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::new(1.0);

        // Within delta - should behave like MSE
        let pred = arr1(&[1.0]);
        let target = arr1(&[1.5]);
        let loss_val = loss.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss_val, 0.125, epsilon = 1e-10);

        // Beyond delta - should behave like MAE
        let pred = arr1(&[1.0]);
        let target = arr1(&[3.0]);
        let loss_val = loss.forward(&pred, &target).unwrap();
        assert_relative_eq!(loss_val, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_quantile_loss() {
        let loss = QuantileLoss::new(0.5).unwrap();
        let pred = arr1(&[1.0, 2.0, 3.0]);
        let target = arr1(&[1.5, 1.5, 3.5]);

        let loss_val = loss.forward(&pred, &target).unwrap();
        assert!(loss_val > 0.0);

        // Test gradient
        let grad = loss.backward(&pred, &target).unwrap();
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_mape_loss() {
        let loss = MAPELoss::default();
        let pred = arr1(&[10.0, 20.0, 30.0]);
        let target = arr1(&[11.0, 22.0, 33.0]);

        let loss_val = loss.forward(&pred, &target).unwrap();
        assert!(loss_val > 0.0 && loss_val < 100.0);
    }

    #[test]
    fn test_weighted_loss() {
        let base_loss = MSELoss;
        let weights = arr1(&[1.0, 2.0, 1.0]);
        let loss = WeightedLoss::new(base_loss, weights);

        let pred = arr1(&[1.0, 2.0, 3.0]);
        let target = arr1(&[1.5, 2.5, 3.5]);

        let loss_val = loss.forward(&pred, &target).unwrap();
        assert!(loss_val > 0.0);

        let grad = loss.backward(&pred, &target).unwrap();
        assert_eq!(grad.len(), 3);
    }
}
