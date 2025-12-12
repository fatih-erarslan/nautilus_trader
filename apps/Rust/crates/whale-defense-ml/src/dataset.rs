//! Dataset management for whale detection training
//! 
//! This module provides data structures and utilities for managing
//! training datasets for whale detection models.

use candle_core::{Device, Tensor};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{Result, WhaleMLError};
use crate::features::{extract_features_batch, MarketFeatures};

/// Whale event representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleEvent {
    /// Timestamp of the event
    pub timestamp: i64,
    /// Threat level (1-5)
    pub threat_level: u8,
    /// Estimated size of the whale trade
    pub estimated_size: f64,
    /// Event type
    pub event_type: WhaleEventType,
}

/// Types of whale events
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum WhaleEventType {
    /// Large market buy order
    LargeBuy,
    /// Large market sell order
    LargeSell,
    /// Suspicious accumulation pattern
    Accumulation,
    /// Suspicious distribution pattern
    Distribution,
    /// Potential manipulation
    Manipulation,
}

/// Dataset for whale detection training
pub struct WhaleDataset {
    /// Market data sequences
    pub sequences: Array2<f32>,
    /// Binary labels (0: no whale, 1: whale)
    pub labels: Array1<u8>,
    /// Sequence length
    pub sequence_length: usize,
    /// Prediction horizon
    pub prediction_horizon: usize,
    /// Feature dimension
    pub feature_dim: usize,
}

impl WhaleDataset {
    /// Create a new dataset from market data and whale events
    pub fn new(
        prices: &[f32],
        volumes: &[f32],
        whale_events: &[WhaleEvent],
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Result<Self> {
        // Extract features
        let sequences = extract_features_batch(prices, volumes, 20, sequence_length)?;
        
        // Create labels based on whale events
        let labels = Self::create_labels(
            prices.len(),
            whale_events,
            sequence_length,
            prediction_horizon,
        )?;
        
        // Ensure sequences and labels match
        if sequences.shape()[0] != labels.len() {
            return Err(WhaleMLError::InvalidDimensions {
                expected: format!("sequences.len() == labels.len()"),
                actual: format!("sequences: {}, labels: {}", sequences.shape()[0], labels.len()),
            });
        }
        
        let feature_dim = sequences.shape()[1] / sequence_length;
        
        Ok(Self {
            sequences,
            labels,
            sequence_length,
            prediction_horizon,
            feature_dim,
        })
    }
    
    /// Create labels from whale events
    fn create_labels(
        data_length: usize,
        whale_events: &[WhaleEvent],
        sequence_length: usize,
        prediction_horizon: usize,
    ) -> Result<Array1<u8>> {
        let num_sequences = data_length.saturating_sub(sequence_length);
        let mut labels = Array1::zeros(num_sequences);
        
        // For each sequence, check if there's a whale event in the prediction horizon
        for i in 0..num_sequences {
            let start_time = i + sequence_length;
            let end_time = start_time + prediction_horizon;
            
            // Check if any whale event occurs in this window
            for event in whale_events {
                let event_idx = event.timestamp as usize;
                if event_idx >= start_time && event_idx < end_time {
                    labels[i] = 1;
                    break;
                }
            }
        }
        
        Ok(labels)
    }
    
    /// Split dataset into train and validation sets
    pub fn train_test_split(&self, test_ratio: f32) -> Result<(Self, Self)> {
        let n_samples = self.sequences.shape()[0];
        let split_idx = ((n_samples as f32) * (1.0 - test_ratio)) as usize;
        
        // Split sequences
        let train_sequences = self.sequences.slice(ndarray::s![..split_idx, ..]).to_owned();
        let test_sequences = self.sequences.slice(ndarray::s![split_idx.., ..]).to_owned();
        
        // Split labels
        let train_labels = self.labels.slice(ndarray::s![..split_idx]).to_owned();
        let test_labels = self.labels.slice(ndarray::s![split_idx..]).to_owned();
        
        let train_dataset = Self {
            sequences: train_sequences,
            labels: train_labels,
            sequence_length: self.sequence_length,
            prediction_horizon: self.prediction_horizon,
            feature_dim: self.feature_dim,
        };
        
        let test_dataset = Self {
            sequences: test_sequences,
            labels: test_labels,
            sequence_length: self.sequence_length,
            prediction_horizon: self.prediction_horizon,
            feature_dim: self.feature_dim,
        };
        
        Ok((train_dataset, test_dataset))
    }
    
    /// Get a batch of data as tensors
    pub fn get_batch(
        &self,
        batch_indices: &[usize],
        device: &Device,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = batch_indices.len();
        let seq_features = self.sequence_length * self.feature_dim;
        
        // Collect batch data
        let mut batch_sequences = Vec::with_capacity(batch_size * seq_features);
        let mut batch_labels = Vec::with_capacity(batch_size);
        
        for &idx in batch_indices {
            if idx >= self.sequences.shape()[0] {
                return Err(WhaleMLError::InvalidDimensions {
                    expected: format!("index < {}", self.sequences.shape()[0]),
                    actual: format!("index: {}", idx),
                });
            }
            
            // Add sequence
            for &val in self.sequences.row(idx).iter() {
                batch_sequences.push(val);
            }
            
            // Add label
            batch_labels.push(self.labels[idx] as i64);
        }
        
        // Reshape sequences to (batch, sequence_length, feature_dim)
        let sequences_tensor = Tensor::from_vec(
            batch_sequences,
            (batch_size, self.sequence_length, self.feature_dim),
            device,
        )?;
        
        let labels_tensor = Tensor::from_vec(batch_labels, batch_size, device)?;
        
        Ok((sequences_tensor, labels_tensor))
    }
    
    /// Get dataset statistics
    pub fn get_stats(&self) -> DatasetStats {
        let total_samples = self.labels.len();
        let whale_samples = self.labels.iter().filter(|&&x| x == 1).count();
        let normal_samples = total_samples - whale_samples;
        
        DatasetStats {
            total_samples,
            whale_samples,
            normal_samples,
            whale_ratio: whale_samples as f32 / total_samples as f32,
            sequence_length: self.sequence_length,
            feature_dim: self.feature_dim,
        }
    }
}

/// Dataset statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetStats {
    /// Total number of samples
    pub total_samples: usize,
    /// Number of whale samples
    pub whale_samples: usize,
    /// Number of normal samples
    pub normal_samples: usize,
    /// Ratio of whale samples
    pub whale_ratio: f32,
    /// Sequence length
    pub sequence_length: usize,
    /// Feature dimension
    pub feature_dim: usize,
}

/// Data preprocessor for normalization and augmentation
pub struct DataPreprocessor {
    /// Mean values for normalization
    means: HashMap<String, f32>,
    /// Standard deviations for normalization
    stds: HashMap<String, f32>,
    /// Whether to apply augmentation
    augmentation: bool,
}

impl DataPreprocessor {
    /// Create a new preprocessor
    pub fn new(augmentation: bool) -> Self {
        Self {
            means: HashMap::new(),
            stds: HashMap::new(),
            augmentation,
        }
    }
    
    /// Fit the preprocessor on training data
    pub fn fit(&mut self, sequences: &Array2<f32>) -> Result<()> {
        let n_features = sequences.shape()[1];
        
        for i in 0..n_features {
            let feature_data = sequences.column(i);
            let mean = feature_data.mean().unwrap_or(0.0);
            let std = feature_data.std(0.0);
            
            self.means.insert(format!("feature_{}", i), mean);
            self.stds.insert(format!("feature_{}", i), std.max(1e-7));
        }
        
        Ok(())
    }
    
    /// Transform sequences using fitted parameters
    pub fn transform(&self, sequences: &Array2<f32>) -> Result<Array2<f32>> {
        let mut normalized = sequences.clone();
        
        for i in 0..sequences.shape()[1] {
            let key = format!("feature_{}", i);
            let mean = self.means.get(&key).copied().unwrap_or(0.0);
            let std = self.stds.get(&key).copied().unwrap_or(1.0);
            
            normalized.column_mut(i).mapv_inplace(|x| (x - mean) / std);
        }
        
        Ok(normalized)
    }
    
    /// Apply data augmentation
    pub fn augment(&self, sequence: &Array1<f32>) -> Array1<f32> {
        if !self.augmentation {
            return sequence.clone();
        }
        
        // Simple augmentation: add small random noise
        let noise_scale = 0.01;
        let noise = Array1::from_shape_fn(sequence.len(), |_| {
            rand::random::<f32>() * 2.0 * noise_scale - noise_scale
        });
        
        sequence + &noise
    }
}

/// Batch iterator for training
pub struct BatchIterator<'a> {
    dataset: &'a WhaleDataset,
    batch_size: usize,
    current_idx: usize,
    shuffle: bool,
    indices: Vec<usize>,
}

impl<'a> BatchIterator<'a> {
    /// Create a new batch iterator
    pub fn new(dataset: &'a WhaleDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.sequences.shape()[0]).collect();
        
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        Self {
            dataset,
            batch_size,
            current_idx: 0,
            shuffle,
            indices,
        }
    }
    
    /// Get the next batch
    pub fn next_batch(&mut self, device: &Device) -> Result<Option<(Tensor, Tensor)>> {
        if self.current_idx >= self.indices.len() {
            return Ok(None);
        }
        
        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        
        let batch = self.dataset.get_batch(batch_indices, device)?;
        self.current_idx = end_idx;
        
        Ok(Some(batch))
    }
    
    /// Reset the iterator
    pub fn reset(&mut self) {
        self.current_idx = 0;
        
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_dataset_creation() {
        let prices: Vec<f32> = (0..1000).map(|i| 45000.0 + (i as f32 * 10.0)).collect();
        let volumes: Vec<f32> = (0..1000).map(|i| 1000000.0 + (i as f32 * 1000.0)).collect();
        
        let whale_events = vec![
            WhaleEvent {
                timestamp: 100,
                threat_level: 3,
                estimated_size: 1e7,
                event_type: WhaleEventType::LargeBuy,
            },
            WhaleEvent {
                timestamp: 500,
                threat_level: 4,
                estimated_size: 2e7,
                event_type: WhaleEventType::Manipulation,
            },
        ];
        
        let dataset = WhaleDataset::new(&prices, &volumes, &whale_events, 60, 15);
        assert!(dataset.is_ok());
        
        let dataset = dataset.unwrap();
        let stats = dataset.get_stats();
        
        assert!(stats.total_samples > 0);
        assert!(stats.whale_samples > 0);
        assert!(stats.whale_ratio > 0.0 && stats.whale_ratio < 1.0);
    }
    
    #[test]
    fn test_train_test_split() {
        let prices: Vec<f32> = (0..1000).map(|i| 45000.0 + (i as f32)).collect();
        let volumes: Vec<f32> = vec![1000000.0; 1000];
        let whale_events = vec![];
        
        let dataset = WhaleDataset::new(&prices, &volumes, &whale_events, 60, 15).unwrap();
        let (train, test) = dataset.train_test_split(0.2).unwrap();
        
        assert!(train.sequences.shape()[0] > test.sequences.shape()[0]);
        assert_eq!(
            train.sequences.shape()[0] + test.sequences.shape()[0],
            dataset.sequences.shape()[0]
        );
    }
}