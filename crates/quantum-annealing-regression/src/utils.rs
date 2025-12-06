//! Utility Functions
//!
//! This module provides various utility functions for data processing,
//! validation, and common operations.

use crate::error::*;

/// Data processing utilities
pub struct DataUtils;

impl DataUtils {
    /// Normalize features to zero mean and unit variance
    pub fn normalize_features(features: &mut [Vec<f64>]) -> QarResult<(Vec<f64>, Vec<f64>)> {
        if features.is_empty() {
            return Err(QarError::InvalidInput("Empty features".to_string()));
        }
        
        let num_features = features[0].len();
        let num_samples = features.len();
        
        if num_features == 0 {
            return Err(QarError::InvalidInput("Zero feature dimension".to_string()));
        }
        
        let mut means = vec![0.0; num_features];
        let mut stds = vec![0.0; num_features];
        
        // Calculate means
        for feature in features.iter() {
            for (i, &value) in feature.iter().enumerate() {
                means[i] += value;
            }
        }
        for mean in means.iter_mut() {
            *mean /= num_samples as f64;
        }
        
        // Calculate standard deviations
        for feature in features.iter() {
            for (i, &value) in feature.iter().enumerate() {
                stds[i] += (value - means[i]).powi(2);
            }
        }
        for std in stds.iter_mut() {
            *std = (*std / num_samples as f64).sqrt();
            if *std == 0.0 {
                *std = 1.0; // Avoid division by zero
            }
        }
        
        // Normalize features
        for feature in features.iter_mut() {
            for (i, value) in feature.iter_mut().enumerate() {
                *value = (*value - means[i]) / stds[i];
            }
        }
        
        Ok((means, stds))
    }
    
    /// Split data into training and testing sets
    pub fn train_test_split(
        features: Vec<Vec<f64>>,
        targets: Vec<f64>,
        test_ratio: f64,
        seed: Option<u64>,
    ) -> QarResult<(Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>, Vec<f64>)> {
        if features.len() != targets.len() {
            return Err(QarError::InvalidInput("Features and targets length mismatch".to_string()));
        }
        
        if test_ratio < 0.0 || test_ratio > 1.0 {
            return Err(QarError::InvalidInput("Test ratio must be between 0 and 1".to_string()));
        }
        
        let total_samples = features.len();
        let test_samples = (total_samples as f64 * test_ratio).round() as usize;
        let train_samples = total_samples - test_samples;
        
        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..total_samples).collect();
        if let Some(seed) = seed {
            use rand::{SeedableRng, seq::SliceRandom};
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            indices.shuffle(&mut rng);
        }
        
        // Split indices
        let train_indices = &indices[0..train_samples];
        let test_indices = &indices[train_samples..];
        
        // Extract training data
        let train_features: Vec<Vec<f64>> = train_indices.iter()
            .map(|&i| features[i].clone())
            .collect();
        let train_targets: Vec<f64> = train_indices.iter()
            .map(|&i| targets[i])
            .collect();
        
        // Extract testing data
        let test_features: Vec<Vec<f64>> = test_indices.iter()
            .map(|&i| features[i].clone())
            .collect();
        let test_targets: Vec<f64> = test_indices.iter()
            .map(|&i| targets[i])
            .collect();
        
        Ok((train_features, train_targets, test_features, test_targets))
    }
    
    /// Calculate mean absolute error
    pub fn mean_absolute_error(actual: &[f64], predicted: &[f64]) -> QarResult<f64> {
        if actual.len() != predicted.len() {
            return Err(QarError::InvalidInput("Array length mismatch".to_string()));
        }
        
        if actual.is_empty() {
            return Err(QarError::InvalidInput("Empty arrays".to_string()));
        }
        
        let mae = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).abs())
            .sum::<f64>() / actual.len() as f64;
        
        Ok(mae)
    }
    
    /// Calculate mean squared error
    pub fn mean_squared_error(actual: &[f64], predicted: &[f64]) -> QarResult<f64> {
        if actual.len() != predicted.len() {
            return Err(QarError::InvalidInput("Array length mismatch".to_string()));
        }
        
        if actual.is_empty() {
            return Err(QarError::InvalidInput("Empty arrays".to_string()));
        }
        
        let mse = actual.iter()
            .zip(predicted.iter())
            .map(|(&a, &p)| (a - p).powi(2))
            .sum::<f64>() / actual.len() as f64;
        
        Ok(mse)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_normalize_features() {
        let mut features = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        
        let result = DataUtils::normalize_features(&mut features);
        assert!(result.is_ok());
        
        let (means, stds) = result.unwrap();
        assert_relative_eq!(means[0], 3.0, epsilon = 1e-10);
        assert_relative_eq!(means[1], 4.0, epsilon = 1e-10);
        
        // Check that normalized features have approximately zero mean
        let new_mean_0 = features.iter().map(|f| f[0]).sum::<f64>() / features.len() as f64;
        let new_mean_1 = features.iter().map(|f| f[1]).sum::<f64>() / features.len() as f64;
        
        assert_relative_eq!(new_mean_0, 0.0, epsilon = 1e-10);
        assert_relative_eq!(new_mean_1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_train_test_split() {
        let features = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0]
        ];
        let targets = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = DataUtils::train_test_split(features, targets, 0.2, Some(42));
        assert!(result.is_ok());
        
        let (train_features, train_targets, test_features, test_targets) = result.unwrap();
        
        assert_eq!(train_features.len(), 4);
        assert_eq!(test_features.len(), 1);
        assert_eq!(train_targets.len(), 4);
        assert_eq!(test_targets.len(), 1);
        
        // Total samples should be preserved
        assert_eq!(train_features.len() + test_features.len(), 5);
    }

    #[test]
    fn test_mean_absolute_error() {
        let actual = vec![1.0, 2.0, 3.0, 4.0];
        let predicted = vec![1.1, 1.9, 3.2, 3.8];
        
        let mae = DataUtils::mean_absolute_error(&actual, &predicted).unwrap();
        let expected_mae = (0.1 + 0.1 + 0.2 + 0.2) / 4.0;
        
        assert_relative_eq!(mae, expected_mae, epsilon = 1e-10);
    }

    #[test]
    fn test_mean_squared_error() {
        let actual = vec![1.0, 2.0, 3.0];
        let predicted = vec![1.1, 1.9, 3.1];
        
        let mse = DataUtils::mean_squared_error(&actual, &predicted).unwrap();
        let expected_mse = (0.01 + 0.01 + 0.01) / 3.0;
        
        assert_relative_eq!(mse, expected_mse, epsilon = 1e-10);
    }

    #[test]
    fn test_error_metrics_validation() {
        let actual = vec![1.0, 2.0];
        let predicted = vec![1.0]; // Different length
        
        let mae_result = DataUtils::mean_absolute_error(&actual, &predicted);
        assert!(mae_result.is_err());
        
        let mse_result = DataUtils::mean_squared_error(&actual, &predicted);
        assert!(mse_result.is_err());
    }
}