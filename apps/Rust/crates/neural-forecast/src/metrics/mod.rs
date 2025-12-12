//! Forecasting metrics and evaluation

use ndarray::{Array1, Array3};
use serde::{Serialize, Deserialize};
use crate::Result;

/// Forecasting metrics calculator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastMetrics {
    /// Mean Absolute Error
    pub mae: f32,
    /// Mean Squared Error
    pub mse: f32,
    /// Root Mean Squared Error
    pub rmse: f32,
    /// Mean Absolute Percentage Error
    pub mape: f32,
    /// R-squared (coefficient of determination)
    pub r2: f32,
    /// Directional accuracy percentage
    pub directional_accuracy: f32,
}

impl ForecastMetrics {
    /// Calculate metrics from predictions and targets
    pub fn calculate(predictions: &Array3<f32>, targets: &Array3<f32>) -> Result<Self> {
        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        let abs_diff = diff.mapv(|x| x.abs());
        
        let mae = abs_diff.mean().unwrap_or(0.0);
        let mse = squared_diff.mean().unwrap_or(0.0);
        let rmse = mse.sqrt();
        
        // MAPE calculation
        let mape = {
            let mut total_percentage_error = 0.0;
            let mut count = 0;
            
            for ((pred, target), _) in predictions.iter().zip(targets.iter()).zip(diff.iter()) {
                if target.abs() > 1e-8 {
                    total_percentage_error += ((pred - target) / target).abs();
                    count += 1;
                }
            }
            
            if count > 0 {
                (total_percentage_error / count as f32) * 100.0
            } else {
                0.0
            }
        };
        
        // R² calculation
        let target_mean = targets.mean().unwrap_or(0.0);
        let ss_tot = targets.mapv(|x| (x - target_mean).powi(2)).sum();
        let ss_res = squared_diff.sum();
        
        let r2 = if ss_tot > 1e-8 {
            1.0 - (ss_res / ss_tot)
        } else {
            0.0
        };
        
        // Directional accuracy (simplified)
        let directional_accuracy = {
            let mut correct = 0;
            let mut total = 0;
            
            // This is a simplified calculation - in practice you'd need price data
            for i in 1..predictions.shape()[1] {
                for j in 0..predictions.shape()[0] {
                    for k in 0..predictions.shape()[2] {
                        let pred_direction = predictions[(j, i, k)] > predictions[(j, i-1, k)];
                        let actual_direction = targets[(j, i, k)] > targets[(j, i-1, k)];
                        
                        if pred_direction == actual_direction {
                            correct += 1;
                        }
                        total += 1;
                    }
                }
            }
            
            if total > 0 {
                (correct as f32 / total as f32) * 100.0
            } else {
                0.0
            }
        };
        
        Ok(Self {
            mae,
            mse,
            rmse,
            mape,
            r2,
            directional_accuracy,
        })
    }
}