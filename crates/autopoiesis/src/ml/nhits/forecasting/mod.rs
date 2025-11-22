//! NHITS Forecasting Pipeline
//! 
//! This module provides a complete production-ready forecasting system built on top of NHITS.
//! It includes multi-horizon prediction, ensemble methods, online learning, uncertainty quantification,
//! anomaly detection, and adaptive retraining capabilities.

use std::sync::{Arc, Mutex};
use std::collections::{HashMap, VecDeque};
use anyhow::{Result, Context};
use serde::{Serialize, Deserialize};
use tokio::sync::{RwLock, broadcast};
use ndarray::{Array1, Array2, Array3, Axis, s};
use statrs::distribution::{Normal, ContinuousCDF};
use chrono::{DateTime, Utc, Duration};

use crate::ml::nhits::{NHITS, NHITSConfig, NHITSError, ModelState};
use crate::consciousness::ConsciousnessField;
use crate::core::autopoiesis::AutopoieticSystem;

/// Forecasting pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastingConfig {
    /// Prediction horizons (e.g., [1, 7, 30] for 1-day, 1-week, 1-month)
    pub horizons: Vec<usize>,
    
    /// Number of models in ensemble
    pub ensemble_size: usize,
    
    /// Confidence levels for prediction intervals
    pub confidence_levels: Vec<f64>,
    
    /// Window size for online learning
    pub online_window_size: usize,
    
    /// Update frequency for online learning
    pub update_frequency: usize,
    
    /// Anomaly detection threshold (in standard deviations)
    pub anomaly_threshold: f64,
    
    /// Retraining trigger conditions
    pub retraining_config: RetrainingConfig,
    
    /// Data preprocessing configuration
    pub preprocessing_config: PreprocessingConfig,
    
    /// Model persistence settings
    pub persistence_config: PersistenceConfig,
}

impl Default for ForecastingConfig {
    fn default() -> Self {
        Self {
            horizons: vec![1, 7, 30],
            ensemble_size: 5,
            confidence_levels: vec![0.90, 0.95, 0.99],
            online_window_size: 1000,
            update_frequency: 100,
            anomaly_threshold: 3.0,
            retraining_config: RetrainingConfig::default(),
            preprocessing_config: PreprocessingConfig::default(),
            persistence_config: PersistenceConfig::default(),
        }
    }
}

/// Retraining configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrainingConfig {
    /// Performance degradation threshold
    pub performance_threshold: f64,
    
    /// Maximum time between retrainings
    pub max_time_between_retraining: Duration,
    
    /// Minimum samples for retraining
    pub min_samples: usize,
    
    /// Concept drift detection enabled
    pub detect_concept_drift: bool,
}

impl Default for RetrainingConfig {
    fn default() -> Self {
        Self {
            performance_threshold: 0.2,
            max_time_between_retraining: Duration::days(7),
            min_samples: 1000,
            detect_concept_drift: true,
        }
    }
}

/// Data preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    /// Enable normalization
    pub normalize: bool,
    
    /// Detrending method
    pub detrending: DetrendingMethod,
    
    /// Seasonal decomposition
    pub seasonal_decomposition: bool,
    
    /// Feature engineering settings
    pub feature_engineering: FeatureEngineeringConfig,
    
    /// Outlier handling
    pub outlier_handling: OutlierHandling,
}

impl Default for PreprocessingConfig {
    fn default() -> Self {
        Self {
            normalize: true,
            detrending: DetrendingMethod::Linear,
            seasonal_decomposition: true,
            feature_engineering: FeatureEngineeringConfig::default(),
            outlier_handling: OutlierHandling::Clip,
        }
    }
}

/// Feature engineering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureEngineeringConfig {
    /// Lag features to generate
    pub lag_features: Vec<usize>,
    
    /// Rolling statistics window sizes
    pub rolling_windows: Vec<usize>,
    
    /// Fourier features for seasonality
    pub fourier_features: Option<usize>,
    
    /// Calendar features
    pub calendar_features: bool,
}

impl Default for FeatureEngineeringConfig {
    fn default() -> Self {
        Self {
            lag_features: vec![1, 7, 30],
            rolling_windows: vec![7, 30],
            fourier_features: Some(10),
            calendar_features: true,
        }
    }
}

/// Model persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Auto-save interval
    pub auto_save_interval: Option<Duration>,
    
    /// Maximum saved versions
    pub max_versions: usize,
    
    /// Compression enabled
    pub compress: bool,
    
    /// Save path
    pub save_path: String,
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            auto_save_interval: Some(Duration::hours(1)),
            max_versions: 10,
            compress: true,
            save_path: "./models/nhits".to_string(),
        }
    }
}

/// Detrending methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DetrendingMethod {
    None,
    Linear,
    Polynomial(usize),
    MovingAverage(usize),
}

/// Outlier handling strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OutlierHandling {
    None,
    Clip,
    Remove,
    Impute,
}

/// Forecast result with uncertainty quantification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForecastResult {
    /// Point forecasts for each horizon
    pub forecasts: HashMap<usize, Array1<f64>>,
    
    /// Prediction intervals for each horizon and confidence level
    pub intervals: HashMap<(usize, f64), (Array1<f64>, Array1<f64>)>,
    
    /// Uncertainty scores
    pub uncertainty: HashMap<usize, Array1<f64>>,
    
    /// Anomaly scores
    pub anomaly_scores: Option<Array1<f64>>,
    
    /// Model confidence
    pub confidence: f64,
    
    /// Forecast timestamp
    pub timestamp: DateTime<Utc>,
    
    /// Model version
    pub model_version: String,
}

/// Performance metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Mean Absolute Error
    pub mae: f64,
    
    /// Root Mean Square Error
    pub rmse: f64,
    
    /// Mean Absolute Percentage Error
    pub mape: f64,
    
    /// Coverage of prediction intervals
    pub interval_coverage: HashMap<f64, f64>,
    
    /// Forecast bias
    pub bias: f64,
    
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Main forecasting pipeline
pub struct ForecastingPipeline {
    /// Configuration
    config: ForecastingConfig,
    
    /// Ensemble of NHITS models
    models: Vec<Arc<RwLock<NHITS>>>,
    
    /// Consciousness field integration
    consciousness: Arc<ConsciousnessField>,
    
    /// Autopoietic system
    // Removed autopoietic due to associated type complexity
    // autopoietic: Arc<dyn AutopoieticSystem>,
    
    /// Online learning buffer
    online_buffer: Arc<RwLock<VecDeque<(Array1<f64>, Array1<f64>)>>>,
    
    /// Performance history
    performance_history: Arc<RwLock<VecDeque<PerformanceMetrics>>>,
    
    /// Last retraining time
    last_retraining: Arc<RwLock<DateTime<Utc>>>,
    
    /// Model versions
    model_versions: Arc<RwLock<HashMap<String, DateTime<Utc>>>>,
    
    /// Event broadcaster
    event_tx: broadcast::Sender<ForecastingEvent>,
    
    /// Preprocessing state
    preprocessing_state: Arc<RwLock<PreprocessingState>>,
}

/// Preprocessing state
#[derive(Debug, Clone)]
struct PreprocessingState {
    /// Normalization parameters
    mean: Option<f64>,
    std: Option<f64>,
    
    /// Trend parameters
    trend_params: Option<Vec<f64>>,
    
    /// Seasonal components
    seasonal_components: Option<Array1<f64>>,
    
    /// Feature statistics
    feature_stats: HashMap<String, (f64, f64)>,
}

/// Forecasting events
#[derive(Debug, Clone)]
pub enum ForecastingEvent {
    ForecastGenerated(ForecastResult),
    ModelRetrained(String),
    AnomalyDetected(f64),
    PerformanceDegraded(f64),
    ConceptDriftDetected,
    ModelSaved(String),
}

impl ForecastingPipeline {
    /// Create a new forecasting pipeline
    pub async fn new(
        config: ForecastingConfig,
        consciousness: Arc<ConsciousnessField>,
        autopoietic: Arc<AutopoieticSystem>,
    ) -> Result<Self> {
        let (event_tx, _) = broadcast::channel(1024);
        
        // Create ensemble of models
        let mut models = Vec::new();
        for i in 0..config.ensemble_size {
            let mut model_config = NHITSConfig::default();
            // Add variation to ensemble
            model_config.learning_rate *= 0.8 + 0.4 * (i as f64 / config.ensemble_size as f64);
            
            let model = NHITS::new(
                model_config,
                consciousness.clone(),
                autopoietic.clone(),
            );
            models.push(Arc::new(RwLock::new(model)));
        }
        
        Ok(Self {
            config,
            models,
            consciousness,
            autopoietic,
            online_buffer: Arc::new(RwLock::new(VecDeque::new())),
            performance_history: Arc::new(RwLock::new(VecDeque::new())),
            last_retraining: Arc::new(RwLock::new(Utc::now())),
            model_versions: Arc::new(RwLock::new(HashMap::new())),
            event_tx,
            preprocessing_state: Arc::new(RwLock::new(PreprocessingState {
                mean: None,
                std: None,
                trend_params: None,
                seasonal_components: None,
                feature_stats: HashMap::new(),
            })),
        })
    }
    
    /// Generate multi-horizon forecasts with uncertainty
    pub async fn forecast(
        &self,
        input: &Array1<f64>,
        external_features: Option<&Array2<f64>>,
    ) -> Result<ForecastResult> {
        // Preprocess input
        let (processed_input, features) = self.preprocess_data(input, external_features).await?;
        
        // Generate ensemble forecasts
        let mut horizon_forecasts = HashMap::new();
        let mut horizon_predictions = HashMap::new();
        
        for horizon in &self.config.horizons {
            let mut predictions = Vec::new();
            
            // Get predictions from each model
            for model in &self.models {
                let model = model.read().await;
                let pred = model.predict(&processed_input, *horizon)
                    .context("Failed to generate prediction")?;
                predictions.push(pred);
            }
            
            // Calculate ensemble forecast and uncertainty
            let ensemble_result = self.ensemble_predictions(&predictions)?;
            horizon_forecasts.insert(*horizon, ensemble_result.0);
            horizon_predictions.insert(*horizon, predictions);
        }
        
        // Calculate prediction intervals
        let intervals = self.calculate_prediction_intervals(&horizon_predictions)?;
        
        // Calculate uncertainty scores
        let uncertainty = self.calculate_uncertainty(&horizon_predictions)?;
        
        // Detect anomalies
        let anomaly_scores = self.detect_anomalies(&processed_input).await?;
        
        // Calculate model confidence
        let confidence = self.calculate_confidence(&uncertainty)?;
        
        let result = ForecastResult {
            forecasts: horizon_forecasts,
            intervals,
            uncertainty,
            anomaly_scores: Some(anomaly_scores),
            confidence,
            timestamp: Utc::now(),
            model_version: self.get_current_version().await?,
        };
        
        // Broadcast event
        let _ = self.event_tx.send(ForecastingEvent::ForecastGenerated(result.clone()));
        
        Ok(result)
    }
    
    /// Update models with new data (online learning)
    pub async fn update(&self, input: &Array1<f64>, target: &Array1<f64>) -> Result<()> {
        // Add to online buffer
        {
            let mut buffer = self.online_buffer.write().await;
            buffer.push_back((input.clone(), target.clone()));
            
            if buffer.len() > self.config.online_window_size {
                buffer.pop_front();
            }
        }
        
        // Check if update is needed
        let buffer = self.online_buffer.read().await;
        if buffer.len() % self.config.update_frequency == 0 && !buffer.is_empty() {
            // Perform online update
            for model in &self.models {
                let mut model = model.write().await;
                
                // Convert buffer to training batch
                let batch_inputs: Vec<Array1<f64>> = buffer.iter()
                    .map(|(inp, _)| inp.clone())
                    .collect();
                let batch_targets: Vec<Array1<f64>> = buffer.iter()
                    .map(|(_, tgt)| tgt.clone())
                    .collect();
                
                // Update model
                model.update_online(&batch_inputs, &batch_targets)
                    .context("Failed to update model online")?;
            }
        }
        
        // Check if retraining is needed
        if self.should_retrain().await? {
            self.retrain().await?;
        }
        
        Ok(())
    }
    
    /// Retrain models from scratch
    pub async fn retrain(&self) -> Result<()> {
        let buffer = self.online_buffer.read().await;
        if buffer.len() < self.config.retraining_config.min_samples {
            return Ok(());
        }
        
        // Prepare training data
        let inputs: Vec<Array1<f64>> = buffer.iter()
            .map(|(inp, _)| inp.clone())
            .collect();
        let targets: Vec<Array1<f64>> = buffer.iter()
            .map(|(_, tgt)| tgt.clone())
            .collect();
        
        // Retrain each model in ensemble
        for (i, model) in self.models.iter().enumerate() {
            let mut model = model.write().await;
            
            // Reset and retrain
            model.reset_weights()?;
            model.train(&inputs, &targets, 100)
                .context(format!("Failed to retrain model {}", i))?;
        }
        
        // Update last retraining time
        *self.last_retraining.write().await = Utc::now();
        
        // Save retrained models
        let version = self.save_models().await?;
        
        // Broadcast event
        let _ = self.event_tx.send(ForecastingEvent::ModelRetrained(version));
        
        Ok(())
    }
    
    /// Preprocess data with feature engineering
    async fn preprocess_data(
        &self,
        input: &Array1<f64>,
        external_features: Option<&Array2<f64>>,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let mut processed = input.clone();
        let mut features = Vec::new();
        
        let mut state = self.preprocessing_state.write().await;
        
        // Normalization
        if self.config.preprocessing_config.normalize {
            if state.mean.is_none() {
                state.mean = Some(input.mean().unwrap());
                state.std = Some(input.std(0.0));
            }
            
            let mean = state.mean.unwrap();
            let std = state.std.unwrap();
            processed = (processed - mean) / std;
        }
        
        // Detrending
        match self.config.preprocessing_config.detrending {
            DetrendingMethod::Linear => {
                let trend = self.calculate_linear_trend(&processed)?;
                processed = processed - &trend;
                features.push(trend);
            }
            DetrendingMethod::Polynomial(degree) => {
                let trend = self.calculate_polynomial_trend(&processed, degree)?;
                processed = processed - &trend;
                features.push(trend);
            }
            DetrendingMethod::MovingAverage(window) => {
                let trend = self.calculate_moving_average(&processed, window)?;
                processed = processed - &trend;
                features.push(trend);
            }
            DetrendingMethod::None => {}
        }
        
        // Feature engineering
        let eng_features = self.engineer_features(&processed).await?;
        features.extend(eng_features);
        
        // Combine with external features if provided
        if let Some(ext_features) = external_features {
            let n_samples = processed.len();
            let mut all_features = Array2::zeros((n_samples, features.len() + ext_features.ncols()));
            
            for (i, feat) in features.iter().enumerate() {
                all_features.slice_mut(s![.., i]).assign(feat);
            }
            
            all_features.slice_mut(s![.., features.len()..]).assign(ext_features);
            Ok((processed, all_features))
        } else {
            let n_samples = processed.len();
            let n_features = features.len();
            let mut feature_matrix = Array2::zeros((n_samples, n_features));
            
            for (i, feat) in features.iter().enumerate() {
                feature_matrix.slice_mut(s![.., i]).assign(feat);
            }
            
            Ok((processed, feature_matrix))
        }
    }
    
    /// Engineer features from time series
    async fn engineer_features(&self, data: &Array1<f64>) -> Result<Vec<Array1<f64>>> {
        let mut features = Vec::new();
        let config = &self.config.preprocessing_config.feature_engineering;
        
        // Lag features
        for &lag in &config.lag_features {
            let mut lag_feature = Array1::zeros(data.len());
            if lag < data.len() {
                lag_feature.slice_mut(s![lag..]).assign(&data.slice(s![..-lag]));
                features.push(lag_feature);
            }
        }
        
        // Rolling statistics
        for &window in &config.rolling_windows {
            let rolling_mean = self.calculate_rolling_mean(data, window)?;
            let rolling_std = self.calculate_rolling_std(data, window)?;
            features.push(rolling_mean);
            features.push(rolling_std);
        }
        
        // Fourier features
        if let Some(n_fourier) = config.fourier_features {
            let fourier_features = self.calculate_fourier_features(data, n_fourier)?;
            features.extend(fourier_features);
        }
        
        // Calendar features
        if config.calendar_features {
            let calendar_features = self.calculate_calendar_features(data.len())?;
            features.extend(calendar_features);
        }
        
        Ok(features)
    }
    
    /// Calculate ensemble predictions
    fn ensemble_predictions(&self, predictions: &[Array1<f64>]) -> Result<(Array1<f64>, f64)> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No predictions to ensemble"));
        }
        
        let n_points = predictions[0].len();
        let mut ensemble = Array1::zeros(n_points);
        
        // Calculate mean prediction
        for pred in predictions {
            ensemble += pred;
        }
        ensemble /= predictions.len() as f64;
        
        // Calculate variance for uncertainty
        let mut variance = 0.0;
        for pred in predictions {
            let diff = pred - &ensemble;
            variance += diff.dot(&diff);
        }
        variance /= (predictions.len() * n_points) as f64;
        
        Ok((ensemble, variance))
    }
    
    /// Calculate prediction intervals
    fn calculate_prediction_intervals(
        &self,
        predictions: &HashMap<usize, Vec<Array1<f64>>>,
    ) -> Result<HashMap<(usize, f64), (Array1<f64>, Array1<f64>)>> {
        let mut intervals = HashMap::new();
        
        for (horizon, preds) in predictions {
            for &confidence in &self.config.confidence_levels {
                let interval = self.calculate_interval_for_horizon(preds, confidence)?;
                intervals.insert((*horizon, confidence), interval);
            }
        }
        
        Ok(intervals)
    }
    
    /// Calculate interval for specific horizon
    fn calculate_interval_for_horizon(
        &self,
        predictions: &[Array1<f64>],
        confidence: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        if predictions.is_empty() {
            return Err(anyhow::anyhow!("No predictions for interval calculation"));
        }
        
        let n_points = predictions[0].len();
        let mut lower = Array1::zeros(n_points);
        let mut upper = Array1::zeros(n_points);
        
        // Calculate percentiles
        let alpha = (1.0 - confidence) / 2.0;
        let lower_idx = (alpha * predictions.len() as f64) as usize;
        let upper_idx = ((1.0 - alpha) * predictions.len() as f64) as usize;
        
        for i in 0..n_points {
            let mut point_preds: Vec<f64> = predictions.iter()
                .map(|p| p[i])
                .collect();
            point_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            lower[i] = point_preds[lower_idx];
            upper[i] = point_preds[upper_idx.min(point_preds.len() - 1)];
        }
        
        Ok((lower, upper))
    }
    
    /// Calculate uncertainty scores
    fn calculate_uncertainty(
        &self,
        predictions: &HashMap<usize, Vec<Array1<f64>>>,
    ) -> Result<HashMap<usize, Array1<f64>>> {
        let mut uncertainty = HashMap::new();
        
        for (horizon, preds) in predictions {
            let n_points = preds[0].len();
            let mut unc = Array1::zeros(n_points);
            
            // Calculate standard deviation across ensemble
            for i in 0..n_points {
                let point_preds: Vec<f64> = preds.iter()
                    .map(|p| p[i])
                    .collect();
                
                let mean = point_preds.iter().sum::<f64>() / point_preds.len() as f64;
                let variance = point_preds.iter()
                    .map(|&x| (x - mean).powi(2))
                    .sum::<f64>() / point_preds.len() as f64;
                
                unc[i] = variance.sqrt();
            }
            
            uncertainty.insert(*horizon, unc);
        }
        
        Ok(uncertainty)
    }
    
    /// Detect anomalies in input data
    async fn detect_anomalies(&self, data: &Array1<f64>) -> Result<Array1<f64>> {
        let n_points = data.len();
        let mut anomaly_scores = Array1::zeros(n_points);
        
        // Use statistical approach for anomaly detection
        let mean = data.mean().unwrap();
        let std = data.std(0.0);
        
        for i in 0..n_points {
            let z_score = (data[i] - mean).abs() / std;
            anomaly_scores[i] = z_score;
            
            if z_score > self.config.anomaly_threshold {
                let _ = self.event_tx.send(ForecastingEvent::AnomalyDetected(z_score));
            }
        }
        
        Ok(anomaly_scores)
    }
    
    /// Calculate model confidence
    fn calculate_confidence(&self, uncertainty: &HashMap<usize, Array1<f64>>) -> Result<f64> {
        if uncertainty.is_empty() {
            return Ok(0.0);
        }
        
        // Average normalized uncertainty across horizons
        let mut total_confidence = 0.0;
        
        for (_, unc) in uncertainty {
            let mean_unc = unc.mean().unwrap();
            // Convert uncertainty to confidence (inverse relationship)
            let horizon_confidence = 1.0 / (1.0 + mean_unc);
            total_confidence += horizon_confidence;
        }
        
        Ok(total_confidence / uncertainty.len() as f64)
    }
    
    /// Check if retraining is needed
    async fn should_retrain(&self) -> Result<bool> {
        // Check time since last retraining
        let last_retrain = *self.last_retraining.read().await;
        let time_since = Utc::now() - last_retrain;
        
        if time_since > self.config.retraining_config.max_time_between_retraining {
            return Ok(true);
        }
        
        // Check performance degradation
        let history = self.performance_history.read().await;
        if history.len() < 2 {
            return Ok(false);
        }
        
        let recent = &history[history.len() - 1];
        let baseline = &history[0];
        
        let degradation = (recent.mae - baseline.mae) / baseline.mae;
        if degradation > self.config.retraining_config.performance_threshold {
            let _ = self.event_tx.send(ForecastingEvent::PerformanceDegraded(degradation));
            return Ok(true);
        }
        
        // Check concept drift
        if self.config.retraining_config.detect_concept_drift {
            if self.detect_concept_drift().await? {
                let _ = self.event_tx.send(ForecastingEvent::ConceptDriftDetected);
                return Ok(true);
            }
        }
        
        Ok(false)
    }
    
    /// Detect concept drift in data
    async fn detect_concept_drift(&self) -> Result<bool> {
        let buffer = self.online_buffer.read().await;
        if buffer.len() < 200 {
            return Ok(false);
        }
        
        // Simple drift detection using distribution comparison
        let mid = buffer.len() / 2;
        let first_half: Vec<f64> = buffer.iter()
            .take(mid)
            .flat_map(|(inp, _)| inp.to_vec())
            .collect();
        let second_half: Vec<f64> = buffer.iter()
            .skip(mid)
            .flat_map(|(inp, _)| inp.to_vec())
            .collect();
        
        // Kolmogorov-Smirnov test approximation
        let ks_stat = self.calculate_ks_statistic(&first_half, &second_half)?;
        
        Ok(ks_stat > 0.05) // Threshold for drift detection
    }
    
    /// Calculate Kolmogorov-Smirnov statistic
    fn calculate_ks_statistic(&self, data1: &[f64], data2: &[f64]) -> Result<f64> {
        let mut all_values: Vec<f64> = data1.iter().chain(data2.iter()).cloned().collect();
        all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut max_diff = 0.0;
        
        for &value in &all_values {
            let cdf1 = data1.iter().filter(|&&x| x <= value).count() as f64 / data1.len() as f64;
            let cdf2 = data2.iter().filter(|&&x| x <= value).count() as f64 / data2.len() as f64;
            let diff = (cdf1 - cdf2).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        
        Ok(max_diff)
    }
    
    /// Save models to disk
    pub async fn save_models(&self) -> Result<String> {
        let version = format!("v{}", Utc::now().timestamp());
        let base_path = &self.config.persistence_config.save_path;
        
        // Create directory if needed
        tokio::fs::create_dir_all(base_path).await?;
        
        // Save each model
        for (i, model) in self.models.iter().enumerate() {
            let model = model.read().await;
            let path = format!("{}/model_{}_{}.bin", base_path, i, version);
            
            // Serialize model state
            let state = model.save_state()?;
            let serialized = bincode::serialize(&state)?;
            
            // Optionally compress
            let data = if self.config.persistence_config.compress {
                compress_data(&serialized)?
            } else {
                serialized
            };
            
            tokio::fs::write(&path, data).await?;
        }
        
        // Update version tracking
        self.model_versions.write().await.insert(version.clone(), Utc::now());
        
        // Clean old versions
        self.clean_old_versions().await?;
        
        let _ = self.event_tx.send(ForecastingEvent::ModelSaved(version.clone()));
        
        Ok(version)
    }
    
    /// Load models from disk
    pub async fn load_models(&self, version: &str) -> Result<()> {
        let base_path = &self.config.persistence_config.save_path;
        
        for (i, model) in self.models.iter().enumerate() {
            let path = format!("{}/model_{}_{}.bin", base_path, i, version);
            
            let data = tokio::fs::read(&path).await?;
            
            // Optionally decompress
            let serialized = if self.config.persistence_config.compress {
                decompress_data(&data)?
            } else {
                data
            };
            
            // Deserialize and load state
            let state: crate::ml::nhits::ModelState = bincode::deserialize(&serialized)?;
            let mut model = model.write().await;
            model.load_state(state)?;
        }
        
        Ok(())
    }
    
    /// Get current model version
    async fn get_current_version(&self) -> Result<String> {
        let versions = self.model_versions.read().await;
        versions.iter()
            .max_by_key(|(_, time)| *time)
            .map(|(version, _)| version.clone())
            .ok_or_else(|| anyhow::anyhow!("No model version available"))
    }
    
    /// Clean old model versions
    async fn clean_old_versions(&self) -> Result<()> {
        let mut versions = self.model_versions.write().await;
        
        if versions.len() <= self.config.persistence_config.max_versions {
            return Ok(());
        }
        
        // Sort by timestamp and keep only recent versions
        let mut sorted: Vec<_> = versions.iter().collect();
        sorted.sort_by_key(|(_, time)| *time);
        
        let to_remove = sorted.len() - self.config.persistence_config.max_versions;
        let base_path = &self.config.persistence_config.save_path;
        
        for (version, _) in sorted.iter().take(to_remove) {
            // Remove files
            for i in 0..self.config.ensemble_size {
                let path = format!("{}/model_{}_v{}.bin", base_path, i, version);
                let _ = tokio::fs::remove_file(&path).await;
            }
            
            versions.remove(*version);
        }
        
        Ok(())
    }
    
    /// Calculate linear trend
    fn calculate_linear_trend(&self, data: &Array1<f64>) -> Result<Array1<f64>> {
        let n = data.len() as f64;
        let x: Array1<f64> = Array1::range(0.0, n, 1.0);
        
        let x_mean = x.mean().unwrap();
        let y_mean = data.mean().unwrap();
        
        let numerator = x.iter().zip(data.iter())
            .map(|(xi, yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<f64>();
        
        let denominator = x.iter()
            .map(|xi| (xi - x_mean).powi(2))
            .sum::<f64>();
        
        let slope = numerator / denominator;
        let intercept = y_mean - slope * x_mean;
        
        Ok(x.mapv(|xi| slope * xi + intercept))
    }
    
    /// Calculate polynomial trend
    fn calculate_polynomial_trend(&self, data: &Array1<f64>, degree: usize) -> Result<Array1<f64>> {
        // Simplified polynomial fitting
        // In production, use proper polynomial regression
        let n = data.len();
        let mut trend = Array1::zeros(n);
        
        // For now, approximate with moving polynomial
        let window = n / (degree + 1);
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2).min(n);
            let slice = data.slice(s![start..end]);
            trend[i] = slice.mean().unwrap();
        }
        
        Ok(trend)
    }
    
    /// Calculate moving average
    fn calculate_moving_average(&self, data: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
        let n = data.len();
        let mut ma = Array1::zeros(n);
        
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);
            let slice = data.slice(s![start..end]);
            ma[i] = slice.mean().unwrap();
        }
        
        Ok(ma)
    }
    
    /// Calculate rolling mean
    fn calculate_rolling_mean(&self, data: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
        self.calculate_moving_average(data, window)
    }
    
    /// Calculate rolling standard deviation
    fn calculate_rolling_std(&self, data: &Array1<f64>, window: usize) -> Result<Array1<f64>> {
        let n = data.len();
        let mut std = Array1::zeros(n);
        
        for i in 0..n {
            let start = i.saturating_sub(window / 2);
            let end = (i + window / 2 + 1).min(n);
            let slice = data.slice(s![start..end]);
            std[i] = slice.std(0.0);
        }
        
        Ok(std)
    }
    
    /// Calculate Fourier features
    fn calculate_fourier_features(&self, data: &Array1<f64>, n_features: usize) -> Result<Vec<Array1<f64>>> {
        let n = data.len();
        let mut features = Vec::new();
        
        for k in 1..=n_features {
            let mut sin_feature = Array1::zeros(n);
            let mut cos_feature = Array1::zeros(n);
            
            for i in 0..n {
                let angle = 2.0 * std::f64::consts::PI * k as f64 * i as f64 / n as f64;
                sin_feature[i] = angle.sin();
                cos_feature[i] = angle.cos();
            }
            
            features.push(sin_feature);
            features.push(cos_feature);
        }
        
        Ok(features)
    }
    
    /// Calculate calendar features
    fn calculate_calendar_features(&self, n: usize) -> Result<Vec<Array1<f64>>> {
        let mut features = Vec::new();
        
        // Day of week
        let mut dow = Array1::zeros(n);
        // Hour of day
        let mut hod = Array1::zeros(n);
        // Day of month
        let mut dom = Array1::zeros(n);
        // Month of year
        let mut moy = Array1::zeros(n);
        
        let now = Utc::now();
        
        for i in 0..n {
            let timestamp = now - Duration::hours((n - i - 1) as i64);
            dow[i] = timestamp.weekday().num_days_from_monday() as f64 / 7.0;
            hod[i] = timestamp.hour() as f64 / 24.0;
            dom[i] = timestamp.day() as f64 / 31.0;
            moy[i] = timestamp.month() as f64 / 12.0;
        }
        
        features.push(dow);
        features.push(hod);
        features.push(dom);
        features.push(moy);
        
        Ok(features)
    }
    
    /// Calculate performance metrics
    pub async fn calculate_metrics(
        &self,
        predictions: &ForecastResult,
        actuals: &HashMap<usize, Array1<f64>>,
    ) -> Result<PerformanceMetrics> {
        let mut total_mae = 0.0;
        let mut total_mse = 0.0;
        let mut total_mape = 0.0;
        let mut total_bias = 0.0;
        let mut total_points = 0;
        
        let mut interval_coverage = HashMap::new();
        
        for (horizon, pred) in &predictions.forecasts {
            if let Some(actual) = actuals.get(horizon) {
                let n = pred.len().min(actual.len());
                
                for i in 0..n {
                    let error = pred[i] - actual[i];
                    total_mae += error.abs();
                    total_mse += error.powi(2);
                    if actual[i] != 0.0 {
                        total_mape += (error / actual[i]).abs();
                    }
                    total_bias += error;
                }
                
                total_points += n;
                
                // Check interval coverage
                for &confidence in &self.config.confidence_levels {
                    if let Some((lower, upper)) = predictions.intervals.get(&(*horizon, confidence)) {
                        let mut covered = 0;
                        for i in 0..n {
                            if actual[i] >= lower[i] && actual[i] <= upper[i] {
                                covered += 1;
                            }
                        }
                        let coverage = covered as f64 / n as f64;
                        interval_coverage.insert(confidence, coverage);
                    }
                }
            }
        }
        
        let metrics = PerformanceMetrics {
            mae: total_mae / total_points as f64,
            rmse: (total_mse / total_points as f64).sqrt(),
            mape: total_mape / total_points as f64,
            interval_coverage,
            bias: total_bias / total_points as f64,
            timestamp: Utc::now(),
        };
        
        // Update performance history
        self.performance_history.write().await.push_back(metrics.clone());
        
        Ok(metrics)
    }
    
    /// Get event receiver
    pub fn subscribe(&self) -> broadcast::Receiver<ForecastingEvent> {
        self.event_tx.subscribe()
    }
}

// Helper functions for compression
fn compress_data(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::write::GzEncoder;
    use flate2::Compression;
    use std::io::Write;
    
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    Ok(encoder.finish()?)
}

fn decompress_data(data: &[u8]) -> Result<Vec<u8>> {
    use flate2::read::GzDecoder;
    use std::io::Read;
    
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

#[cfg(examples)]
pub mod example;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_forecasting_pipeline() {
        let config = ForecastingConfig::default();
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        
        let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic)
            .await
            .unwrap();
        
        // Test data
        let input = Array1::range(0.0, 100.0, 1.0);
        
        // Generate forecast
        let result = pipeline.forecast(&input, None).await.unwrap();
        
        assert!(!result.forecasts.is_empty());
        assert!(result.confidence > 0.0);
    }
    
    #[tokio::test]
    async fn test_online_learning() {
        let config = ForecastingConfig {
            online_window_size: 10,
            update_frequency: 5,
            ..Default::default()
        };
        
        let consciousness = Arc::new(ConsciousnessField::new());
        let autopoietic = Arc::new(AutopoieticSystem::new());
        
        let pipeline = ForecastingPipeline::new(config, consciousness, autopoietic)
            .await
            .unwrap();
        
        // Add data points
        for i in 0..10 {
            let input = Array1::from_vec(vec![i as f64; 10]);
            let target = Array1::from_vec(vec![(i + 1) as f64; 10]);
            pipeline.update(&input, &target).await.unwrap();
        }
        
        // Check buffer size
        let buffer = pipeline.online_buffer.read().await;
        assert_eq!(buffer.len(), 10);
    }
}