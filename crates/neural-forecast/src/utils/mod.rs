//! Utility functions for neural forecasting

use ndarray::{Array1, Array2, Array3, s};
use crate::Result;
use std::collections::HashMap;

/// Utility functions for data manipulation and model operations
pub struct Utils;

impl Utils {
    /// Create sliding windows from time series data
    pub fn create_sliding_windows(
        data: &Array2<f32>,
        window_size: usize,
        horizon: usize,
        stride: usize,
    ) -> Result<(Array3<f32>, Array3<f32>)> {
        let (seq_len, n_features) = data.dim();
        
        if seq_len < window_size + horizon {
            return Err(crate::NeuralForecastError::PreprocessingError(
                "Sequence too short for window size and horizon".to_string()
            ));
        }
        
        let num_windows = (seq_len - window_size - horizon) / stride + 1;
        let mut inputs = Array3::zeros((num_windows, window_size, n_features));
        let mut targets = Array3::zeros((num_windows, horizon, n_features));
        
        for i in 0..num_windows {
            let start_idx = i * stride;
            let input_end = start_idx + window_size;
            let target_end = input_end + horizon;
            
            // Copy input window
            for j in 0..window_size {
                for k in 0..n_features {
                    inputs[(i, j, k)] = data[(start_idx + j, k)];
                }
            }
            
            // Copy target window
            for j in 0..horizon {
                for k in 0..n_features {
                    targets[(i, j, k)] = data[(input_end + j, k)];
                }
            }
        }
        
        Ok((inputs, targets))
    }
    
    /// Split data into train/validation/test sets
    pub fn train_val_test_split(
        data: &Array3<f32>,
        train_ratio: f32,
        val_ratio: f32,
    ) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>)> {
        let total_samples = data.shape()[0];
        let train_size = (total_samples as f32 * train_ratio) as usize;
        let val_size = (total_samples as f32 * val_ratio) as usize;
        
        let train_data = data.slice(s![..train_size, .., ..]).to_owned();
        let val_data = data.slice(s![train_size..train_size + val_size, .., ..]).to_owned();
        let test_data = data.slice(s![train_size + val_size.., .., ..]).to_owned();
        
        Ok((train_data, val_data, test_data))
    }
    
    /// Calculate feature importance using permutation method
    pub fn calculate_feature_importance(
        baseline_score: f32,
        permuted_scores: &[f32],
    ) -> Vec<f32> {
        permuted_scores
            .iter()
            .map(|&score| baseline_score - score)
            .collect()
    }
    
    /// Apply rolling window statistics
    pub fn rolling_statistics(
        data: &Array2<f32>,
        window_size: usize,
    ) -> Result<HashMap<String, Array2<f32>>> {
        let (seq_len, n_features) = data.dim();
        let mut results = HashMap::new();
        
        if window_size > seq_len {
            return Err(crate::NeuralForecastError::PreprocessingError(
                "Window size larger than sequence length".to_string()
            ));
        }
        
        let output_len = seq_len - window_size + 1;
        let mut rolling_mean = Array2::zeros((output_len, n_features));
        let mut rolling_std = Array2::zeros((output_len, n_features));
        let mut rolling_min = Array2::zeros((output_len, n_features));
        let mut rolling_max = Array2::zeros((output_len, n_features));
        
        for i in 0..output_len {
            for j in 0..n_features {
                let window = data.slice(s![i..i + window_size, j]);
                
                rolling_mean[(i, j)] = window.mean().unwrap_or(0.0);
                rolling_std[(i, j)] = window.std(0.0);
                rolling_min[(i, j)] = window.iter().copied().fold(f32::INFINITY, f32::min);
                rolling_max[(i, j)] = window.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            }
        }
        
        results.insert("mean".to_string(), rolling_mean);
        results.insert("std".to_string(), rolling_std);
        results.insert("min".to_string(), rolling_min);
        results.insert("max".to_string(), rolling_max);
        
        Ok(results)
    }
    
    /// Pad sequences to same length
    pub fn pad_sequences(
        sequences: Vec<Array2<f32>>,
        max_length: Option<usize>,
        pad_value: f32,
    ) -> Result<Array3<f32>> {
        if sequences.is_empty() {
            return Err(crate::NeuralForecastError::PreprocessingError(
                "No sequences to pad".to_string()
            ));
        }
        
        let n_features = sequences[0].shape()[1];
        let max_len = max_length.unwrap_or_else(|| {
            sequences.iter().map(|seq| seq.shape()[0]).max().unwrap_or(0)
        });
        
        let mut padded = Array3::from_elem((sequences.len(), max_len, n_features), pad_value);
        
        for (i, seq) in sequences.iter().enumerate() {
            let seq_len = seq.shape()[0];
            let copy_len = std::cmp::min(seq_len, max_len);
            
            for j in 0..copy_len {
                for k in 0..n_features {
                    padded[(i, j, k)] = seq[(j, k)];
                }
            }
        }
        
        Ok(padded)
    }
}