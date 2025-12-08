//! Gradient boosting model implementations (XGBoost and LightGBM)

use crate::{Result, TrainingError};
use crate::data::TrainingData;
use crate::config::{XGBoostConfig, LightGBMConfig, TrainingParams};
use crate::models::{Model, ModelType, ModelParameters, ModelMetadata, TrainingMetrics, MetricSet, calculate_metrics};
use async_trait::async_trait;
use ndarray::{Array1, Array2, Array3, Axis};
use serde::{Serialize, Deserialize};
use std::path::Path;
use std::sync::Arc;
use parking_lot::RwLock;

/// XGBoost model implementation
pub struct XGBoostModel {
    config: XGBoostConfig,
    booster: Arc<RwLock<Option<xgboost::Booster>>>,
    metadata: ModelMetadata,
    feature_importance: Option<Vec<f32>>,
}

impl XGBoostModel {
    /// Create new XGBoost model
    pub fn new(config: XGBoostConfig) -> Result<Self> {
        Ok(Self {
            config,
            booster: Arc::new(RwLock::new(None)),
            metadata: ModelMetadata {
                model_type: ModelType::XGBoost,
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                config: serde_json::to_value(&config).unwrap(),
                metrics: None,
            },
            feature_importance: None,
        })
    }
    
    /// Prepare data for XGBoost
    fn prepare_data(&self, data: &TrainingData) -> Result<(xgboost::DMatrix, xgboost::DMatrix, xgboost::DMatrix)> {
        // Flatten 3D arrays to 2D for XGBoost
        let x_train_2d = self.flatten_sequences(&data.x_train)?;
        let y_train_2d = self.flatten_targets(&data.y_train)?;
        
        let x_val_2d = self.flatten_sequences(&data.x_val)?;
        let y_val_2d = self.flatten_targets(&data.y_val)?;
        
        let x_test_2d = self.flatten_sequences(&data.x_test)?;
        let y_test_2d = self.flatten_targets(&data.y_test)?;
        
        // Create DMatrix objects
        let dtrain = xgboost::DMatrix::from_dense(&x_train_2d, data.x_train.shape()[0])
            .map_err(|e| TrainingError::Training(format!("Failed to create training DMatrix: {}", e)))?
            .set_labels(&y_train_2d)
            .map_err(|e| TrainingError::Training(format!("Failed to set training labels: {}", e)))?;
            
        let dval = xgboost::DMatrix::from_dense(&x_val_2d, data.x_val.shape()[0])
            .map_err(|e| TrainingError::Training(format!("Failed to create validation DMatrix: {}", e)))?
            .set_labels(&y_val_2d)
            .map_err(|e| TrainingError::Training(format!("Failed to set validation labels: {}", e)))?;
            
        let dtest = xgboost::DMatrix::from_dense(&x_test_2d, data.x_test.shape()[0])
            .map_err(|e| TrainingError::Training(format!("Failed to create test DMatrix: {}", e)))?
            .set_labels(&y_test_2d)
            .map_err(|e| TrainingError::Training(format!("Failed to set test labels: {}", e)))?;
        
        Ok((dtrain, dval, dtest))
    }
    
    /// Flatten sequences for gradient boosting
    fn flatten_sequences(&self, sequences: &Array3<f32>) -> Result<Vec<f32>> {
        let (n_samples, seq_len, n_features) = sequences.dim();
        let flattened_size = n_samples * seq_len * n_features;
        let mut flattened = Vec::with_capacity(flattened_size);
        
        for i in 0..n_samples {
            for j in 0..seq_len {
                for k in 0..n_features {
                    flattened.push(sequences[[i, j, k]]);
                }
            }
        }
        
        Ok(flattened)
    }
    
    /// Flatten targets (use only last time step for now)
    fn flatten_targets(&self, targets: &Array3<f32>) -> Result<Vec<f32>> {
        let (n_samples, horizon, _n_features) = targets.dim();
        let mut flattened = Vec::with_capacity(n_samples);
        
        // For simplicity, predict only the last time step of the first feature
        for i in 0..n_samples {
            flattened.push(targets[[i, horizon - 1, 0]]);
        }
        
        Ok(flattened)
    }
    
    /// Convert predictions back to 3D array
    fn unflatten_predictions(&self, predictions: &[f32], original_shape: (usize, usize, usize)) -> Result<Array3<f32>> {
        let (n_samples, horizon, n_features) = original_shape;
        let mut result = Array3::<f32>::zeros((n_samples, horizon, n_features));
        
        // For now, replicate the single prediction across all time steps and features
        for i in 0..n_samples {
            for j in 0..horizon {
                for k in 0..n_features {
                    result[[i, j, k]] = predictions[i];
                }
            }
        }
        
        Ok(result)
    }
}

#[async_trait]
impl Model for XGBoostModel {
    async fn train(&mut self, data: &TrainingData, config: &TrainingParams) -> Result<TrainingMetrics> {
        let start_time = std::time::Instant::now();
        
        // Prepare data
        let (dtrain, dval, _dtest) = self.prepare_data(data)?;
        
        // Set up parameters
        let mut params = vec![
            ("max_depth", self.config.max_depth.to_string()),
            ("eta", self.config.learning_rate.to_string()),
            ("subsample", self.config.subsample.to_string()),
            ("colsample_bytree", self.config.colsample_bytree.to_string()),
            ("alpha", self.config.reg_alpha.to_string()),
            ("lambda", self.config.reg_lambda.to_string()),
            ("objective", self.config.objective.clone()),
            ("eval_metric", self.config.eval_metric.clone()),
            ("tree_method", self.config.tree_method.clone()),
        ];
        
        // Add GPU support if available
        #[cfg(feature = "gpu")]
        if self.config.tree_method == "gpu_hist" {
            params.push(("gpu_id", "0".to_string()));
        }
        
        // Set up evaluation sets
        let eval_sets = vec![(&dtrain, "train"), (&dval, "valid")];
        
        // Training loop with early stopping
        let mut train_loss = Vec::new();
        let mut val_loss = Vec::new();
        let mut train_metrics = Vec::new();
        let mut val_metrics = Vec::new();
        let mut best_epoch = 0;
        let mut best_val_loss = f32::INFINITY;
        let mut early_stop_counter = 0;
        
        // Create booster
        let mut booster = xgboost::Booster::new(&dtrain, &params)
            .map_err(|e| TrainingError::Training(format!("Failed to create booster: {}", e)))?;
        
        for epoch in 0..self.config.n_estimators {
            // Update booster
            booster.update(&dtrain, epoch as i32)
                .map_err(|e| TrainingError::Training(format!("Training failed at epoch {}: {}", epoch, e)))?;
            
            // Evaluate
            let train_eval = booster.evaluate(&dtrain)
                .map_err(|e| TrainingError::Training(format!("Train evaluation failed: {}", e)))?;
            let val_eval = booster.evaluate(&dval)
                .map_err(|e| TrainingError::Training(format!("Validation evaluation failed: {}", e)))?;
            
            train_loss.push(train_eval);
            val_loss.push(val_eval);
            
            // Calculate detailed metrics
            let train_pred = booster.predict(&dtrain)
                .map_err(|e| TrainingError::Training(format!("Train prediction failed: {}", e)))?;
            let val_pred = booster.predict(&dval)
                .map_err(|e| TrainingError::Training(format!("Validation prediction failed: {}", e)))?;
            
            // Convert predictions back to 3D for metric calculation
            let train_pred_3d = self.unflatten_predictions(&train_pred, data.y_train.dim())?;
            let val_pred_3d = self.unflatten_predictions(&val_pred, data.y_val.dim())?;
            
            let train_metric = calculate_metrics(&train_pred_3d, &data.y_train)?;
            let val_metric = calculate_metrics(&val_pred_3d, &data.y_val)?;
            
            train_metrics.push(train_metric);
            val_metrics.push(val_metric);
            
            // Early stopping check
            if val_eval < best_val_loss {
                best_val_loss = val_eval;
                best_epoch = epoch;
                early_stop_counter = 0;
            } else {
                early_stop_counter += 1;
            }
            
            if let Some(patience) = self.config.early_stopping_rounds {
                if early_stop_counter >= patience {
                    tracing::info!("Early stopping at epoch {}", epoch);
                    break;
                }
            }
            
            if epoch % 10 == 0 {
                tracing::info!(
                    "Epoch {}/{}: train_loss={:.4}, val_loss={:.4}",
                    epoch, self.config.n_estimators, train_eval, val_eval
                );
            }
        }
        
        // Store the trained booster
        *self.booster.write() = Some(booster);
        
        // Get feature importance
        if let Some(ref booster) = *self.booster.read() {
            // Note: xgboost-rs might not have direct feature importance API
            // This is a placeholder - implement actual feature importance extraction
            self.feature_importance = Some(vec![1.0; data.feature_names.len()]);
        }
        
        let training_time_secs = start_time.elapsed().as_secs_f64();
        
        Ok(TrainingMetrics {
            train_loss,
            val_loss,
            train_metrics,
            val_metrics,
            best_epoch,
            training_time_secs,
            early_stopped: early_stop_counter > 0,
        })
    }
    
    async fn predict(&self, inputs: &Array3<f32>) -> Result<Array3<f32>> {
        let booster = self.booster.read();
        let booster = booster.as_ref()
            .ok_or_else(|| TrainingError::Training("Model not trained".to_string()))?;
        
        // Flatten input
        let flattened = self.flatten_sequences(inputs)?;
        let n_samples = inputs.shape()[0];
        
        // Create DMatrix
        let dmatrix = xgboost::DMatrix::from_dense(&flattened, n_samples)
            .map_err(|e| TrainingError::Training(format!("Failed to create DMatrix: {}", e)))?;
        
        // Make predictions
        let predictions = booster.predict(&dmatrix)
            .map_err(|e| TrainingError::Training(format!("Prediction failed: {}", e)))?;
        
        // Unflatten predictions
        let output_shape = (inputs.shape()[0], self.config.max_depth as usize, inputs.shape()[2]);
        self.unflatten_predictions(&predictions, output_shape)
    }
    
    async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.predict(input).await?);
        }
        Ok(results)
    }
    
    async fn save(&self, path: &Path) -> Result<()> {
        let booster = self.booster.read();
        let booster = booster.as_ref()
            .ok_or_else(|| TrainingError::Training("Model not trained".to_string()))?;
        
        booster.save(path)
            .map_err(|e| TrainingError::Persistence(format!("Failed to save model: {}", e)))?;
        
        // Save metadata
        let metadata_path = path.with_extension("meta.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        
        Ok(())
    }
    
    async fn load(&mut self, path: &Path) -> Result<()> {
        let booster = xgboost::Booster::load(path)
            .map_err(|e| TrainingError::Persistence(format!("Failed to load model: {}", e)))?;
        
        *self.booster.write() = Some(booster);
        
        // Load metadata
        let metadata_path = path.with_extension("meta.json");
        if metadata_path.exists() {
            let metadata_json = std::fs::read_to_string(metadata_path)?;
            self.metadata = serde_json::from_str(&metadata_json)?;
        }
        
        Ok(())
    }
    
    fn parameters(&self) -> ModelParameters {
        ModelParameters {
            weights: Vec::new(),
            biases: Vec::new(),
            extra: serde_json::json!({
                "feature_importance": self.feature_importance,
                "config": self.config,
            }),
        }
    }
    
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<()> {
        // XGBoost parameters are set during training
        Ok(())
    }
    
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
    
    fn validate_input(&self, input: &Array3<f32>) -> Result<()> {
        if input.shape()[2] == 0 {
            return Err(TrainingError::Validation("Input features cannot be empty".to_string()));
        }
        Ok(())
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::XGBoost
    }
}

/// LightGBM model implementation
pub struct LightGBMModel {
    config: LightGBMConfig,
    booster: Arc<RwLock<Option<lightgbm::Booster>>>,
    metadata: ModelMetadata,
    feature_importance: Option<Vec<f32>>,
}

impl LightGBMModel {
    /// Create new LightGBM model
    pub fn new(config: LightGBMConfig) -> Result<Self> {
        Ok(Self {
            config,
            booster: Arc::new(RwLock::new(None)),
            metadata: ModelMetadata {
                model_type: ModelType::LightGBM,
                version: "1.0.0".to_string(),
                created_at: chrono::Utc::now(),
                modified_at: chrono::Utc::now(),
                config: serde_json::to_value(&config).unwrap(),
                metrics: None,
            },
            feature_importance: None,
        })
    }
    
    /// Prepare data for LightGBM
    fn prepare_data(&self, data: &TrainingData) -> Result<(lightgbm::Dataset, lightgbm::Dataset)> {
        // Flatten sequences
        let x_train = self.flatten_to_2d(&data.x_train)?;
        let y_train = self.flatten_targets(&data.y_train)?;
        
        let x_val = self.flatten_to_2d(&data.x_val)?;
        let y_val = self.flatten_targets(&data.y_val)?;
        
        // Create datasets
        let train_dataset = lightgbm::Dataset::from_vec(x_train, y_train, data.x_train.shape()[0])
            .map_err(|e| TrainingError::Training(format!("Failed to create training dataset: {}", e)))?;
            
        let val_dataset = lightgbm::Dataset::from_vec(x_val, y_val, data.x_val.shape()[0])
            .map_err(|e| TrainingError::Training(format!("Failed to create validation dataset: {}", e)))?;
        
        Ok((train_dataset, val_dataset))
    }
    
    /// Flatten 3D array to 2D
    fn flatten_to_2d(&self, array: &Array3<f32>) -> Result<Vec<Vec<f32>>> {
        let (n_samples, seq_len, n_features) = array.dim();
        let mut result = Vec::with_capacity(n_samples);
        
        for i in 0..n_samples {
            let mut sample = Vec::with_capacity(seq_len * n_features);
            for j in 0..seq_len {
                for k in 0..n_features {
                    sample.push(array[[i, j, k]]);
                }
            }
            result.push(sample);
        }
        
        Ok(result)
    }
    
    /// Flatten targets
    fn flatten_targets(&self, targets: &Array3<f32>) -> Result<Vec<f32>> {
        let (n_samples, horizon, _) = targets.dim();
        let mut result = Vec::with_capacity(n_samples);
        
        // Use last time step of first feature
        for i in 0..n_samples {
            result.push(targets[[i, horizon - 1, 0]]);
        }
        
        Ok(result)
    }
}

#[async_trait]
impl Model for LightGBMModel {
    async fn train(&mut self, data: &TrainingData, _config: &TrainingParams) -> Result<TrainingMetrics> {
        let start_time = std::time::Instant::now();
        
        // Prepare data
        let (train_dataset, val_dataset) = self.prepare_data(data)?;
        
        // Set up parameters
        let params = vec![
            ("boosting_type", self.config.boosting_type.clone()),
            ("num_iterations", self.config.num_iterations.to_string()),
            ("learning_rate", self.config.learning_rate.to_string()),
            ("num_leaves", self.config.num_leaves.to_string()),
            ("max_depth", self.config.max_depth.to_string()),
            ("feature_fraction", self.config.feature_fraction.to_string()),
            ("bagging_fraction", self.config.bagging_fraction.to_string()),
            ("bagging_freq", self.config.bagging_freq.to_string()),
            ("lambda_l1", self.config.lambda_l1.to_string()),
            ("lambda_l2", self.config.lambda_l2.to_string()),
            ("objective", self.config.objective.clone()),
            ("metric", self.config.metric.clone()),
            ("device_type", self.config.device_type.clone()),
        ];
        
        // Train model
        let booster = lightgbm::Booster::train(
            train_dataset,
            val_dataset,
            params.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        ).map_err(|e| TrainingError::Training(format!("LightGBM training failed: {}", e)))?;
        
        // Store booster
        *self.booster.write() = Some(booster);
        
        // Get feature importance
        if let Some(ref booster) = *self.booster.read() {
            self.feature_importance = Some(booster.feature_importance()
                .map_err(|e| TrainingError::Training(format!("Failed to get feature importance: {}", e)))?);
        }
        
        let training_time_secs = start_time.elapsed().as_secs_f64();
        
        // For now, return simplified metrics
        Ok(TrainingMetrics {
            train_loss: vec![0.0], // TODO: Implement proper metric tracking
            val_loss: vec![0.0],
            train_metrics: vec![],
            val_metrics: vec![],
            best_epoch: 0,
            training_time_secs,
            early_stopped: false,
        })
    }
    
    async fn predict(&self, inputs: &Array3<f32>) -> Result<Array3<f32>> {
        let booster = self.booster.read();
        let booster = booster.as_ref()
            .ok_or_else(|| TrainingError::Training("Model not trained".to_string()))?;
        
        // Flatten input
        let flattened = self.flatten_to_2d(inputs)?;
        
        // Make predictions
        let mut predictions = Vec::new();
        for sample in flattened {
            let pred = booster.predict(sample)
                .map_err(|e| TrainingError::Training(format!("Prediction failed: {}", e)))?;
            predictions.extend(pred);
        }
        
        // Reshape predictions
        let (n_samples, _, n_features) = inputs.dim();
        let horizon = self.config.max_depth.max(1) as usize;
        let mut result = Array3::<f32>::zeros((n_samples, horizon, n_features));
        
        for i in 0..n_samples {
            for j in 0..horizon {
                for k in 0..n_features {
                    result[[i, j, k]] = predictions[i];
                }
            }
        }
        
        Ok(result)
    }
    
    async fn predict_batch(&self, inputs: &[Array3<f32>]) -> Result<Vec<Array3<f32>>> {
        let mut results = Vec::new();
        for input in inputs {
            results.push(self.predict(input).await?);
        }
        Ok(results)
    }
    
    async fn save(&self, path: &Path) -> Result<()> {
        let booster = self.booster.read();
        let booster = booster.as_ref()
            .ok_or_else(|| TrainingError::Training("Model not trained".to_string()))?;
        
        booster.save_file(path.to_str().unwrap())
            .map_err(|e| TrainingError::Persistence(format!("Failed to save model: {}", e)))?;
        
        // Save metadata
        let metadata_path = path.with_extension("meta.json");
        let metadata_json = serde_json::to_string_pretty(&self.metadata)?;
        std::fs::write(metadata_path, metadata_json)?;
        
        Ok(())
    }
    
    async fn load(&mut self, path: &Path) -> Result<()> {
        let booster = lightgbm::Booster::from_file(path.to_str().unwrap())
            .map_err(|e| TrainingError::Persistence(format!("Failed to load model: {}", e)))?;
        
        *self.booster.write() = Some(booster);
        
        // Load metadata
        let metadata_path = path.with_extension("meta.json");
        if metadata_path.exists() {
            let metadata_json = std::fs::read_to_string(metadata_path)?;
            self.metadata = serde_json::from_str(&metadata_json)?;
        }
        
        Ok(())
    }
    
    fn parameters(&self) -> ModelParameters {
        ModelParameters {
            weights: Vec::new(),
            biases: Vec::new(),
            extra: serde_json::json!({
                "feature_importance": self.feature_importance,
                "config": self.config,
            }),
        }
    }
    
    fn set_parameters(&mut self, _params: ModelParameters) -> Result<()> {
        Ok(())
    }
    
    fn metadata(&self) -> ModelMetadata {
        self.metadata.clone()
    }
    
    fn validate_input(&self, input: &Array3<f32>) -> Result<()> {
        if input.shape()[2] == 0 {
            return Err(TrainingError::Validation("Input features cannot be empty".to_string()));
        }
        Ok(())
    }
    
    fn model_type(&self) -> ModelType {
        ModelType::LightGBM
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_xgboost_creation() {
        let config = XGBoostConfig {
            n_estimators: 100,
            max_depth: 6,
            learning_rate: 0.1,
            subsample: 0.8,
            colsample_bytree: 0.8,
            reg_alpha: 0.0,
            reg_lambda: 1.0,
            objective: "reg:squarederror".to_string(),
            eval_metric: "rmse".to_string(),
            early_stopping_rounds: Some(10),
            tree_method: "hist".to_string(),
        };
        
        let model = XGBoostModel::new(config);
        assert!(model.is_ok());
    }
    
    #[test]
    fn test_lightgbm_creation() {
        let config = LightGBMConfig {
            boosting_type: "gbdt".to_string(),
            num_iterations: 100,
            learning_rate: 0.1,
            num_leaves: 31,
            max_depth: -1,
            feature_fraction: 0.9,
            bagging_fraction: 0.8,
            bagging_freq: 5,
            lambda_l1: 0.0,
            lambda_l2: 0.0,
            objective: "regression".to_string(),
            metric: "rmse".to_string(),
            early_stopping_rounds: Some(10),
            device_type: "cpu".to_string(),
        };
        
        let model = LightGBMModel::new(config);
        assert!(model.is_ok());
    }
}