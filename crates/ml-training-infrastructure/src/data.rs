//! Data loading and preprocessing module

use crate::{Result, TrainingError};
use crate::config::{DataConfig, NormalizationMethod};
use ndarray::{Array1, Array2, Array3, Axis, s};
use polars::prelude::*;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use dashmap::DashMap;
use chrono::{DateTime, Utc};
use serde::{Serialize, Deserialize};

/// Training data container
#[derive(Debug, Clone)]
pub struct TrainingData {
    /// Input sequences [batch, sequence, features]
    pub x_train: Array3<f32>,
    /// Target sequences [batch, horizon, features]
    pub y_train: Array3<f32>,
    /// Validation inputs
    pub x_val: Array3<f32>,
    /// Validation targets
    pub y_val: Array3<f32>,
    /// Test inputs
    pub x_test: Array3<f32>,
    /// Test targets
    pub y_test: Array3<f32>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Timestamps for each sample
    pub timestamps: Vec<DateTime<Utc>>,
    /// Asset identifiers
    pub assets: Vec<String>,
    /// Normalization parameters
    pub normalization: NormalizationParams,
}

/// Normalization parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizationParams {
    /// Mean values per feature
    pub mean: Array1<f32>,
    /// Standard deviation per feature
    pub std: Array1<f32>,
    /// Minimum values per feature
    pub min: Array1<f32>,
    /// Maximum values per feature
    pub max: Array1<f32>,
    /// Normalization method used
    pub method: NormalizationMethod,
}

/// Data loader for financial time series
pub struct DataLoader {
    config: Arc<DataConfig>,
    cache: Arc<DashMap<String, TrainingData>>,
    normalizer: Arc<RwLock<Option<NormalizationParams>>>,
}

impl DataLoader {
    /// Create new data loader
    pub fn new(config: Arc<DataConfig>) -> Self {
        Self {
            config,
            cache: Arc::new(DashMap::new()),
            normalizer: Arc::new(RwLock::new(None)),
        }
    }
    
    /// Load data from file
    pub async fn load(&self, path: &Path) -> Result<TrainingData> {
        let cache_key = path.to_string_lossy().to_string();
        
        // Check cache
        if self.config.enable_cache {
            if let Some(cached) = self.cache.get(&cache_key) {
                tracing::info!("Loading data from cache");
                return Ok(cached.clone());
            }
        }
        
        // Load from file
        let df = self.load_dataframe(path).await?;
        
        // Preprocess data
        let data = self.preprocess(df).await?;
        
        // Cache if enabled
        if self.config.enable_cache {
            self.cache.insert(cache_key, data.clone());
        }
        
        Ok(data)
    }
    
    /// Load dataframe from file
    async fn load_dataframe(&self, path: &Path) -> Result<DataFrame> {
        let extension = path.extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| TrainingError::DataLoading("Invalid file extension".to_string()))?;
        
        let df = match extension {
            "parquet" => {
                ParquetReader::new(std::fs::File::open(path)?)
                    .finish()
                    .map_err(|e| TrainingError::DataLoading(e.to_string()))?
            }
            "csv" => {
                CsvReader::new(std::fs::File::open(path)?)
                    .has_header(true)
                    .finish()
                    .map_err(|e| TrainingError::DataLoading(e.to_string()))?
            }
            _ => {
                return Err(TrainingError::DataLoading(
                    format!("Unsupported file format: {}", extension)
                ));
            }
        };
        
        Ok(df)
    }
    
    /// Preprocess dataframe into training data
    async fn preprocess(&self, df: DataFrame) -> Result<TrainingData> {
        // Extract features
        let features = self.extract_features(&df)?;
        
        // Create sequences
        let sequences = self.create_sequences(&features)?;
        
        // Split data
        let (train, val, test) = self.split_data(sequences)?;
        
        // Normalize data
        let normalized = self.normalize_data(train, val, test).await?;
        
        Ok(normalized)
    }
    
    /// Extract feature matrix from dataframe
    fn extract_features(&self, df: &DataFrame) -> Result<Array2<f32>> {
        let mut feature_arrays = Vec::new();
        
        for feature_name in &self.config.features {
            let series = df.column(feature_name)
                .map_err(|e| TrainingError::DataLoading(
                    format!("Feature {} not found: {}", feature_name, e)
                ))?;
            
            let array = series.f64()
                .map_err(|e| TrainingError::DataLoading(e.to_string()))?
                .to_ndarray()
                .map_err(|e| TrainingError::DataLoading(e.to_string()))?
                .mapv(|x| x as f32);
            
            feature_arrays.push(array);
        }
        
        // Stack features into 2D array
        let n_samples = feature_arrays[0].len();
        let n_features = feature_arrays.len();
        let mut features = Array2::<f32>::zeros((n_samples, n_features));
        
        for (i, array) in feature_arrays.iter().enumerate() {
            features.slice_mut(s![.., i]).assign(array);
        }
        
        Ok(features)
    }
    
    /// Create sequences for time series prediction
    fn create_sequences(&self, features: &Array2<f32>) -> Result<Vec<(Array2<f32>, Array2<f32>)>> {
        let n_samples = features.shape()[0];
        let seq_len = self.config.sequence_length;
        let horizon = self.config.horizon;
        
        if n_samples < seq_len + horizon {
            return Err(TrainingError::DataLoading(
                "Insufficient data for sequence creation".to_string()
            ));
        }
        
        let mut sequences = Vec::new();
        
        for i in 0..(n_samples - seq_len - horizon + 1) {
            let x = features.slice(s![i..i + seq_len, ..]).to_owned();
            let y = features.slice(s![i + seq_len..i + seq_len + horizon, ..]).to_owned();
            sequences.push((x, y));
        }
        
        Ok(sequences)
    }
    
    /// Split sequences into train/val/test sets
    fn split_data(&self, sequences: Vec<(Array2<f32>, Array2<f32>)>) -> Result<(
        Vec<(Array2<f32>, Array2<f32>)>,
        Vec<(Array2<f32>, Array2<f32>)>,
        Vec<(Array2<f32>, Array2<f32>)>,
    )> {
        let n_samples = sequences.len();
        let (train_ratio, val_ratio, _test_ratio) = self.config.split_ratios;
        
        let train_size = (n_samples as f32 * train_ratio) as usize;
        let val_size = (n_samples as f32 * val_ratio) as usize;
        
        let train = sequences[..train_size].to_vec();
        let val = sequences[train_size..train_size + val_size].to_vec();
        let test = sequences[train_size + val_size..].to_vec();
        
        Ok((train, val, test))
    }
    
    /// Normalize data
    async fn normalize_data(
        &self,
        train: Vec<(Array2<f32>, Array2<f32>)>,
        val: Vec<(Array2<f32>, Array2<f32>)>,
        test: Vec<(Array2<f32>, Array2<f32>)>,
    ) -> Result<TrainingData> {
        // Stack sequences into 3D arrays
        let (x_train, y_train) = self.stack_sequences(train);
        let (x_val, y_val) = self.stack_sequences(val);
        let (x_test, y_test) = self.stack_sequences(test);
        
        // Calculate normalization parameters from training data
        let norm_params = self.calculate_normalization_params(&x_train)?;
        
        // Apply normalization
        let x_train_norm = self.apply_normalization(&x_train, &norm_params)?;
        let y_train_norm = self.apply_normalization(&y_train, &norm_params)?;
        let x_val_norm = self.apply_normalization(&x_val, &norm_params)?;
        let y_val_norm = self.apply_normalization(&y_val, &norm_params)?;
        let x_test_norm = self.apply_normalization(&x_test, &norm_params)?;
        let y_test_norm = self.apply_normalization(&y_test, &norm_params)?;
        
        // Store normalization parameters
        *self.normalizer.write().await = Some(norm_params.clone());
        
        Ok(TrainingData {
            x_train: x_train_norm,
            y_train: y_train_norm,
            x_val: x_val_norm,
            y_val: y_val_norm,
            x_test: x_test_norm,
            y_test: y_test_norm,
            feature_names: self.config.features.clone(),
            timestamps: Vec::new(), // TODO: Extract from dataframe
            assets: Vec::new(), // TODO: Extract from dataframe
            normalization: norm_params,
        })
    }
    
    /// Stack sequences into 3D array
    fn stack_sequences(&self, sequences: Vec<(Array2<f32>, Array2<f32>)>) -> (Array3<f32>, Array3<f32>) {
        let n_sequences = sequences.len();
        let seq_len = self.config.sequence_length;
        let horizon = self.config.horizon;
        let n_features = self.config.features.len();
        
        let mut x = Array3::<f32>::zeros((n_sequences, seq_len, n_features));
        let mut y = Array3::<f32>::zeros((n_sequences, horizon, n_features));
        
        for (i, (x_seq, y_seq)) in sequences.iter().enumerate() {
            x.slice_mut(s![i, .., ..]).assign(x_seq);
            y.slice_mut(s![i, .., ..]).assign(y_seq);
        }
        
        (x, y)
    }
    
    /// Calculate normalization parameters
    fn calculate_normalization_params(&self, data: &Array3<f32>) -> Result<NormalizationParams> {
        let n_features = data.shape()[2];
        let mut mean = Array1::<f32>::zeros(n_features);
        let mut std = Array1::<f32>::zeros(n_features);
        let mut min = Array1::<f32>::from_elem(n_features, f32::INFINITY);
        let mut max = Array1::<f32>::from_elem(n_features, f32::NEG_INFINITY);
        
        // Calculate statistics per feature
        for i in 0..n_features {
            let feature_data = data.slice(s![.., .., i]);
            let flat_data = feature_data.as_slice().unwrap();
            
            // Mean
            let feature_mean = flat_data.iter().sum::<f32>() / flat_data.len() as f32;
            mean[i] = feature_mean;
            
            // Std
            let variance = flat_data.iter()
                .map(|x| (x - feature_mean).powi(2))
                .sum::<f32>() / flat_data.len() as f32;
            std[i] = variance.sqrt();
            
            // Min/Max
            for &value in flat_data {
                if value < min[i] {
                    min[i] = value;
                }
                if value > max[i] {
                    max[i] = value;
                }
            }
        }
        
        Ok(NormalizationParams {
            mean,
            std,
            min,
            max,
            method: self.config.normalization,
        })
    }
    
    /// Apply normalization to data
    fn apply_normalization(&self, data: &Array3<f32>, params: &NormalizationParams) -> Result<Array3<f32>> {
        let mut normalized = data.clone();
        
        match params.method {
            NormalizationMethod::Standard => {
                for i in 0..data.shape()[2] {
                    let feature_slice = normalized.slice_mut(s![.., .., i]);
                    let mean = params.mean[i];
                    let std = params.std[i];
                    
                    if std > 1e-8 {
                        feature_slice.mapv_inplace(|x| (x - mean) / std);
                    }
                }
            }
            NormalizationMethod::MinMax => {
                for i in 0..data.shape()[2] {
                    let feature_slice = normalized.slice_mut(s![.., .., i]);
                    let min = params.min[i];
                    let max = params.max[i];
                    let range = max - min;
                    
                    if range > 1e-8 {
                        feature_slice.mapv_inplace(|x| (x - min) / range);
                    }
                }
            }
            NormalizationMethod::Robust => {
                // TODO: Implement robust scaling
                tracing::warn!("Robust scaling not yet implemented, using standard scaling");
                return self.apply_normalization(data, params);
            }
            NormalizationMethod::None => {
                // No normalization
            }
        }
        
        Ok(normalized)
    }
    
    /// Inverse transform normalized data
    pub fn inverse_transform(&self, data: &Array3<f32>, params: &NormalizationParams) -> Result<Array3<f32>> {
        let mut denormalized = data.clone();
        
        match params.method {
            NormalizationMethod::Standard => {
                for i in 0..data.shape()[2] {
                    let feature_slice = denormalized.slice_mut(s![.., .., i]);
                    let mean = params.mean[i];
                    let std = params.std[i];
                    
                    feature_slice.mapv_inplace(|x| x * std + mean);
                }
            }
            NormalizationMethod::MinMax => {
                for i in 0..data.shape()[2] {
                    let feature_slice = denormalized.slice_mut(s![.., .., i]);
                    let min = params.min[i];
                    let max = params.max[i];
                    let range = max - min;
                    
                    feature_slice.mapv_inplace(|x| x * range + min);
                }
            }
            _ => {}
        }
        
        Ok(denormalized)
    }
}

/// Batch data loader for efficient training
pub struct BatchLoader {
    data: Arc<TrainingData>,
    batch_size: usize,
    shuffle: bool,
    current_idx: usize,
    indices: Vec<usize>,
}

impl BatchLoader {
    /// Create new batch loader
    pub fn new(data: Arc<TrainingData>, batch_size: usize, shuffle: bool) -> Self {
        let n_samples = data.x_train.shape()[0];
        let indices = (0..n_samples).collect();
        
        let mut loader = Self {
            data,
            batch_size,
            shuffle,
            current_idx: 0,
            indices,
        };
        
        if shuffle {
            loader.shuffle_indices();
        }
        
        loader
    }
    
    /// Shuffle indices
    fn shuffle_indices(&mut self) {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        self.indices.shuffle(&mut rng);
    }
    
    /// Get next batch
    pub fn next_batch(&mut self) -> Option<(Array3<f32>, Array3<f32>)> {
        if self.current_idx >= self.indices.len() {
            return None;
        }
        
        let end_idx = (self.current_idx + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_idx..end_idx];
        
        let mut x_batch = Vec::new();
        let mut y_batch = Vec::new();
        
        for &idx in batch_indices {
            x_batch.push(self.data.x_train.slice(s![idx, .., ..]).to_owned());
            y_batch.push(self.data.y_train.slice(s![idx, .., ..]).to_owned());
        }
        
        self.current_idx = end_idx;
        
        // Stack into batch
        let x = ndarray::stack(Axis(0), &x_batch.iter().map(|a| a.view()).collect::<Vec<_>>()).ok()?;
        let y = ndarray::stack(Axis(0), &y_batch.iter().map(|a| a.view()).collect::<Vec<_>>()).ok()?;
        
        Some((x, y))
    }
    
    /// Reset loader
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            self.shuffle_indices();
        }
    }
    
    /// Get number of batches
    pub fn num_batches(&self) -> usize {
        (self.indices.len() + self.batch_size - 1) / self.batch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_data_loader() {
        let config = Arc::new(DataConfig::default());
        let loader = DataLoader::new(config);
        
        // Test normalization
        let data = Array3::<f32>::ones((10, 20, 5));
        let params = loader.calculate_normalization_params(&data).unwrap();
        assert_eq!(params.mean.len(), 5);
    }
}