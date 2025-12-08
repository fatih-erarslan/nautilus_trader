//! Loss functions for neural forecasting

use ndarray::Array3;
use crate::Result;

/// Loss functions for financial forecasting
pub struct ForecastLoss;

impl ForecastLoss {
    /// Mean Squared Error
    pub fn mse(predictions: &Array3<f32>, targets: &Array3<f32>) -> f32 {
        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        squared_diff.mean().unwrap_or(0.0)
    }
    
    /// Mean Absolute Error
    pub fn mae(predictions: &Array3<f32>, targets: &Array3<f32>) -> f32 {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|x| x.abs());
        abs_diff.mean().unwrap_or(0.0)
    }
    
    /// Huber loss (robust to outliers)
    pub fn huber(predictions: &Array3<f32>, targets: &Array3<f32>, delta: f32) -> f32 {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|x| x.abs());
        
        let loss = diff.mapv(|x| {
            if x.abs() <= delta {
                0.5 * x * x
            } else {
                delta * x.abs() - 0.5 * delta * delta
            }
        });
        
        loss.mean().unwrap_or(0.0)
    }
    
    /// Quantile loss for probabilistic forecasting
    pub fn quantile_loss(predictions: &Array3<f32>, targets: &Array3<f32>, quantile: f32) -> f32 {
        let diff = targets - predictions;
        let loss = diff.mapv(|x| {
            if x >= 0.0 {
                quantile * x
            } else {
                (quantile - 1.0) * x
            }
        });
        
        loss.mean().unwrap_or(0.0)
    }
    
    /// Directional loss (penalizes wrong direction predictions)
    pub fn directional_loss(predictions: &Array3<f32>, targets: &Array3<f32>) -> f32 {
        let mut total_loss = 0.0;
        let mut count = 0;
        
        for i in 1..predictions.shape()[1] {
            for j in 0..predictions.shape()[0] {
                for k in 0..predictions.shape()[2] {
                    let pred_change = predictions[(j, i, k)] - predictions[(j, i-1, k)];
                    let actual_change = targets[(j, i, k)] - targets[(j, i-1, k)];
                    
                    // Penalty if directions don't match
                    if pred_change * actual_change < 0.0 {
                        total_loss += 1.0;
                    }
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            total_loss / count as f32
        } else {
            0.0
        }
    }
}