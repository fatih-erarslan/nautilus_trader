//! Neural Signal Processor for CDFA using ruv_FANN
//! 
//! This module implements neural network-based signal processing for CDFA using the
//! existing ruv_FANN neural network library. It provides real-time signal analysis,
//! pattern recognition, and adaptive filtering for trading signals.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use ndarray::{Array1, Array2};

// Import ruv_FANN components
use ruv_fann::{
    Network, NetworkBuilder, NetworkError, ActivationFunction, TrainingAlgorithm,
    TrainingData, TrainingState, ParallelTrainingOptions
};

use crate::errors::SwarmError;
use super::ml_integration::{NeuralSignalProcessor, SignalFeatures, ProcessedSignal, MLExperience};

/// Neural Signal Processor Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProcessorConfig {
    /// Input layer size
    pub input_size: usize,
    
    /// Hidden layer sizes
    pub hidden_layers: Vec<usize>,
    
    /// Output layer size
    pub output_size: usize,
    
    /// Activation function
    pub activation_function: String,
    
    /// Learning rate
    pub learning_rate: f32,
    
    /// Training algorithm
    pub training_algorithm: String,
    
    /// Maximum training epochs
    pub max_epochs: usize,
    
    /// Target error threshold
    pub target_error: f32,
    
    /// Signal buffer size for temporal processing
    pub signal_buffer_size: usize,
    
    /// Feature extraction window size
    pub feature_window_size: usize,
    
    /// Update frequency for model retraining
    pub update_frequency: usize,
    
    /// Parallel training options
    pub parallel_training: bool,
    
    /// Model ensemble size
    pub ensemble_size: usize,
}

impl Default for NeuralProcessorConfig {
    fn default() -> Self {
        Self {
            input_size: 20,           // 20 input features
            hidden_layers: vec![32, 16], // Two hidden layers
            output_size: 5,           // 5 processed signal outputs
            activation_function: "sigmoid".to_string(),
            learning_rate: 0.01,
            training_algorithm: "rprop".to_string(),
            max_epochs: 1000,
            target_error: 0.001,
            signal_buffer_size: 1000,
            feature_window_size: 50,
            update_frequency: 100,
            parallel_training: true,
            ensemble_size: 3,
        }
    }
}

/// Signal processing metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralProcessorMetrics {
    pub total_signals_processed: u64,
    pub average_processing_time_ms: f64,
    pub model_accuracy: f64,
    pub training_error: f32,
    pub feature_importance: HashMap<String, f64>,
    pub ensemble_consensus: f64,
    pub last_training_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
}

impl Default for NeuralProcessorMetrics {
    fn default() -> Self {
        Self {
            total_signals_processed: 0,
            average_processing_time_ms: 0.0,
            model_accuracy: 0.0,
            training_error: 1.0,
            feature_importance: HashMap::new(),
            ensemble_consensus: 0.0,
            last_training_time: Utc::now(),
            last_update: Utc::now(),
        }
    }
}

/// Neural Signal Processor Implementation
pub struct NeuralSignalProcessorImpl {
    /// Configuration
    config: NeuralProcessorConfig,
    
    /// Neural network ensemble
    networks: Arc<RwLock<Vec<Network<f32>>>>,
    
    /// Signal buffer for temporal processing
    signal_buffer: Arc<RwLock<Vec<SignalFeatures>>>,
    
    /// Training data buffer
    training_buffer: Arc<RwLock<Vec<(Vec<f32>, Vec<f32>)>>>,
    
    /// Feature extractors
    feature_extractors: Arc<RwLock<HashMap<String, Box<dyn FeatureExtractor + Send + Sync>>>>,
    
    /// Processing metrics
    metrics: Arc<RwLock<NeuralProcessorMetrics>>,
    
    /// Update counter
    update_counter: Arc<RwLock<usize>>,
    
    /// Model validation data
    validation_data: Arc<RwLock<Vec<(Vec<f32>, Vec<f32>)>>>,
}

/// Feature extractor trait for signal preprocessing
#[async_trait]
pub trait FeatureExtractor: Send + Sync {
    async fn extract_features(&self, signals: &[SignalFeatures]) -> Result<Vec<f32>, SwarmError>;
    fn get_feature_names(&self) -> Vec<String>;
    fn get_feature_importance(&self) -> HashMap<String, f64>;
}

/// Technical indicator feature extractor
pub struct TechnicalIndicatorExtractor {
    window_size: usize,
    indicators: Vec<String>,
}

impl TechnicalIndicatorExtractor {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            indicators: vec![
                "sma".to_string(),
                "ema".to_string(),
                "rsi".to_string(),
                "macd".to_string(),
                "bollinger_bands".to_string(),
                "stochastic".to_string(),
                "atr".to_string(),
                "adx".to_string(),
            ],
        }
    }
    
    fn calculate_sma(&self, prices: &[f64]) -> f64 {
        prices.iter().sum::<f64>() / prices.len() as f64
    }
    
    fn calculate_ema(&self, prices: &[f64], alpha: f64) -> f64 {
        let mut ema = prices[0];
        for &price in &prices[1..] {
            ema = alpha * price + (1.0 - alpha) * ema;
        }
        ema
    }
    
    fn calculate_rsi(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 50.0;
        }
        
        let mut gains = 0.0;
        let mut losses = 0.0;
        
        for i in 1..prices.len() {
            let change = prices[i] - prices[i - 1];
            if change > 0.0 {
                gains += change;
            } else {
                losses -= change;
            }
        }
        
        let avg_gain = gains / (prices.len() - 1) as f64;
        let avg_loss = losses / (prices.len() - 1) as f64;
        
        if avg_loss == 0.0 {
            return 100.0;
        }
        
        let rs = avg_gain / avg_loss;
        100.0 - (100.0 / (1.0 + rs))
    }
    
    fn calculate_macd(&self, prices: &[f64]) -> (f64, f64) {
        let ema12 = self.calculate_ema(prices, 2.0 / 13.0);
        let ema26 = self.calculate_ema(prices, 2.0 / 27.0);
        let macd = ema12 - ema26;
        let signal = self.calculate_ema(&[macd], 2.0 / 10.0);
        (macd, signal)
    }
}

#[async_trait]
impl FeatureExtractor for TechnicalIndicatorExtractor {
    async fn extract_features(&self, signals: &[SignalFeatures]) -> Result<Vec<f32>, SwarmError> {
        if signals.is_empty() {
            return Err(SwarmError::ParameterError("No signals provided".to_string()));
        }
        
        let mut features = Vec::new();
        
        // Extract price data
        let prices: Vec<f64> = signals.iter()
            .filter_map(|s| s.values.get("price").copied())
            .collect();
        
        if prices.len() < self.window_size {
            // Pad with last available price if needed
            let last_price = prices.last().unwrap_or(&0.0);
            let padded_prices: Vec<f64> = (0..self.window_size)
                .map(|i| if i < prices.len() { prices[i] } else { *last_price })
                .collect();
            return self.extract_from_prices(&padded_prices).await;
        }
        
        // Use most recent window
        let window_prices = &prices[prices.len().saturating_sub(self.window_size)..];
        self.extract_from_prices(window_prices).await
    }
    
    fn get_feature_names(&self) -> Vec<String> {
        self.indicators.clone()
    }
    
    fn get_feature_importance(&self) -> HashMap<String, f64> {
        // Static importance weights (in real implementation, this would be learned)
        let mut importance = HashMap::new();
        importance.insert("sma".to_string(), 0.15);
        importance.insert("ema".to_string(), 0.18);
        importance.insert("rsi".to_string(), 0.12);
        importance.insert("macd".to_string(), 0.20);
        importance.insert("bollinger_bands".to_string(), 0.10);
        importance.insert("stochastic".to_string(), 0.08);
        importance.insert("atr".to_string(), 0.09);
        importance.insert("adx".to_string(), 0.08);
        importance
    }
}

impl TechnicalIndicatorExtractor {
    async fn extract_from_prices(&self, prices: &[f64]) -> Result<Vec<f32>, SwarmError> {
        let mut features = Vec::new();
        
        // SMA
        features.push(self.calculate_sma(prices) as f32);
        
        // EMA
        features.push(self.calculate_ema(prices, 0.2) as f32);
        
        // RSI
        features.push(self.calculate_rsi(prices) as f32);
        
        // MACD
        let (macd, signal) = self.calculate_macd(prices);
        features.push(macd as f32);
        features.push(signal as f32);
        
        // Bollinger Bands (simplified)
        let sma = self.calculate_sma(prices);
        let variance = prices.iter()
            .map(|p| (p - sma).powi(2))
            .sum::<f64>() / prices.len() as f64;
        let std_dev = variance.sqrt();
        features.push((sma + 2.0 * std_dev) as f32); // Upper band
        features.push((sma - 2.0 * std_dev) as f32); // Lower band
        
        // Stochastic (simplified)
        let high = prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let low = prices.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let current = prices.last().unwrap_or(&0.0);
        let stoch_k = if high != low {
            ((current - low) / (high - low)) * 100.0
        } else {
            50.0
        };
        features.push(stoch_k as f32);
        
        Ok(features)
    }
}

impl NeuralSignalProcessorImpl {
    /// Create new neural signal processor
    pub async fn new(config: NeuralProcessorConfig) -> Result<Self, SwarmError> {
        let mut networks = Vec::new();
        
        // Create ensemble of neural networks
        for _ in 0..config.ensemble_size {
            let mut network_builder = NetworkBuilder::new();
            
            // Add input layer
            network_builder = network_builder.input_layer(config.input_size);
            
            // Add hidden layers
            for &hidden_size in &config.hidden_layers {
                network_builder = network_builder.hidden_layer(hidden_size);
            }
            
            // Add output layer
            network_builder = network_builder.output_layer(config.output_size);
            
            let network = network_builder.build();
            networks.push(network);
        }
        
        // Initialize feature extractors
        let mut feature_extractors: HashMap<String, Box<dyn FeatureExtractor + Send + Sync>> = HashMap::new();
        feature_extractors.insert(
            "technical_indicators".to_string(),
            Box::new(TechnicalIndicatorExtractor::new(config.feature_window_size))
        );
        
        Ok(Self {
            config,
            networks: Arc::new(RwLock::new(networks)),
            signal_buffer: Arc::new(RwLock::new(Vec::new())),
            training_buffer: Arc::new(RwLock::new(Vec::new())),
            feature_extractors: Arc::new(RwLock::new(feature_extractors)),
            metrics: Arc::new(RwLock::new(NeuralProcessorMetrics::default())),
            update_counter: Arc::new(RwLock::new(0)),
            validation_data: Arc::new(RwLock::new(Vec::new())),
        })
    }
    
    /// Extract features from signal buffer
    async fn extract_features(&self, signals: &[SignalFeatures]) -> Result<Vec<f32>, SwarmError> {
        let extractors = self.feature_extractors.read().await;
        let mut all_features = Vec::new();
        
        for (name, extractor) in extractors.iter() {
            let features = extractor.extract_features(signals).await?;
            all_features.extend(features);
        }
        
        // Pad or truncate to match input size
        all_features.resize(self.config.input_size, 0.0);
        
        Ok(all_features)
    }
    
    /// Process signals through neural network ensemble
    async fn process_through_ensemble(&self, input_features: &[f32]) -> Result<Vec<f32>, SwarmError> {
        let networks = self.networks.read().await;
        let mut ensemble_outputs = Vec::new();
        
        for network in networks.iter() {
            let output = network.run(&input_features.to_vec())
                .map_err(|e| SwarmError::ParameterError(format!("Neural network run failed: {}", e)))?;
            ensemble_outputs.push(output);
        }
        
        // Calculate ensemble average
        let output_size = ensemble_outputs[0].len();
        let mut averaged_output = vec![0.0; output_size];
        
        for outputs in &ensemble_outputs {
            for (i, &value) in outputs.iter().enumerate() {
                averaged_output[i] += value;
            }
        }
        
        for value in &mut averaged_output {
            *value /= ensemble_outputs.len() as f32;
        }
        
        // Calculate ensemble consensus (standard deviation)
        let mut consensus_score = 0.0;
        for i in 0..output_size {
            let mean = averaged_output[i];
            let variance = ensemble_outputs.iter()
                .map(|outputs| (outputs[i] - mean).powi(2))
                .sum::<f32>() / ensemble_outputs.len() as f32;
            consensus_score += variance.sqrt();
        }
        consensus_score /= output_size as f32;
        
        // Update consensus metric
        {
            let mut metrics = self.metrics.write().await;
            metrics.ensemble_consensus = 1.0 - (consensus_score as f64).min(1.0);
        }
        
        Ok(averaged_output)
    }
    
    /// Train neural networks with accumulated data
    async fn train_networks(&self) -> Result<(), SwarmError> {
        let training_data = self.training_buffer.read().await;
        if training_data.len() < 10 {
            return Ok("); // Need minimum data for training
        }
        
        let mut networks = self.networks.write().await;
        let mut total_error = 0.0;
        
        for network in networks.iter_mut() {
            // Prepare training data
            let mut train_inputs = Vec::new();
            let mut train_outputs = Vec::new();
            
            for (input, output) in training_data.iter() {
                train_inputs.push(input.clone());
                train_outputs.push(output.clone());
            }
            
            // Create training data structure
            let training_set = TrainingData::new(train_inputs, train_outputs)
                .map_err(|e| SwarmError::ParameterError(format!("Training data creation failed: {}", e)))?;
            
            // Train network
            let final_error = if self.config.parallel_training {
                let options = ParallelTrainingOptions::default();
                network.train_on_data_parallel(
                    &training_set,
                    self.config.max_epochs,
                    self.config.target_error,
                    options
                )
            } else {
                network.train_on_data(
                    &training_set,
                    self.config.max_epochs,
                    self.config.target_error
                )
            }.map_err(|e| SwarmError::ParameterError(format!("Network training failed: {}", e)))?;
            
            total_error += final_error;
        }
        
        // Update training metrics
        {
            let mut metrics = self.metrics.write().await;
            metrics.training_error = total_error / networks.len() as f32;
            metrics.last_training_time = Utc::now();
        }
        
        Ok(())
    }
    
    /// Update processing metrics
    async fn update_metrics(&self, processing_time: f64, accuracy: f64) -> Result<(), SwarmError> {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_signals_processed += 1;
        
        // Update average processing time
        let alpha = 0.1;
        metrics.average_processing_time_ms = alpha * processing_time + (1.0 - alpha) * metrics.average_processing_time_ms;
        
        // Update model accuracy
        metrics.model_accuracy = alpha * accuracy + (1.0 - alpha) * metrics.model_accuracy;
        
        // Update feature importance
        let extractors = self.feature_extractors.read().await;
        for (name, extractor) in extractors.iter() {
            let importance = extractor.get_feature_importance();
            for (feature_name, importance_value) in importance {
                metrics.feature_importance.insert(format!("{}_{}", name, feature_name), importance_value);
            }
        }
        
        metrics.last_update = Utc::now();
        
        Ok(())
    }
}

#[async_trait]
impl NeuralSignalProcessor for NeuralSignalProcessorImpl {
    async fn process_signals(
        &self,
        signals: &[SignalFeatures],
        context: &HashMap<String, f64>,
    ) -> Result<Vec<ProcessedSignal>, SwarmError> {
        let start_time = std::time::Instant::now();
        
        // Add signals to buffer
        {
            let mut buffer = self.signal_buffer.write().await;
            buffer.extend_from_slice(signals);
            
            // Maintain buffer size
            if buffer.len() > self.config.signal_buffer_size {
                let excess = buffer.len() - self.config.signal_buffer_size;
                buffer.drain(0..excess);
            }
        }
        
        // Extract features from signal buffer
        let buffer = self.signal_buffer.read().await;
        let input_features = self.extract_features(&buffer).await?;
        
        // Process through neural network ensemble
        let output = self.process_through_ensemble(&input_features).await?;
        
        // Convert output to processed signals
        let mut processed_signals = Vec::new();
        
        for (i, &value) in output.iter().enumerate() {
            let signal = ProcessedSignal {
                signal_id: format!("neural_output_{}", i),
                processed_value: value as f64,
                confidence: 0.8, // Base confidence, could be computed from ensemble consensus
                feature_importance: HashMap::new(), // Could be populated with detailed analysis
                processing_metadata: HashMap::from([
                    ("model_type".to_string(), serde_json::Value::String("neural_ensemble".to_string())),
                    ("ensemble_size".to_string(), serde_json::Value::Number(self.config.ensemble_size.into())),
                    ("input_features".to_string(), serde_json::Value::Number(input_features.len().into())),
                ]),
                timestamp: Utc::now(),
            };
            processed_signals.push(signal);
        }
        
        // Update metrics
        let processing_time = start_time.elapsed().as_millis() as f64;
        self.update_metrics(processing_time, 0.9).await?; // Placeholder accuracy
        
        // Check if training is needed
        {
            let mut counter = self.update_counter.write().await;
            *counter += 1;
            
            if *counter >= self.config.update_frequency {
                *counter = 0;
                
                // Train networks asynchronously
                let self_clone = Arc::new(self);
                tokio::spawn(async move {
                    if let Err(e) = self_clone.train_networks().await {
                        eprintln!("Neural network training failed: {}", e);
                    }
                });
            }
        }
        
        Ok(processed_signals)
    }
    
    async fn learn_from_feedback(
        &self,
        signal_id: &str,
        feedback: &MLExperience,
    ) -> Result<(), SwarmError> {
        // Add to training buffer
        {
            let mut buffer = self.training_buffer.write().await;
            
            let input = feedback.state_vector.iter().map(|&x| x as f32).collect();
            let output = feedback.next_state_vector.iter().map(|&x| x as f32).collect();
            
            buffer.push((input, output));
            
            // Maintain buffer size
            if buffer.len() > self.config.signal_buffer_size {
                buffer.remove(0);
            }
        }
        
        Ok(())
    }
    
    async fn get_processing_metrics(&self) -> Result<serde_json::Value, SwarmError> {
        let metrics = self.metrics.read().await;
        
        serde_json::to_value(&*metrics)
            .map_err(|e| SwarmError::SerializationError(format!("Failed to serialize metrics: {}", e)))
    }
    
    async fn update_processing_parameters(&self, parameters: &HashMap<String, f64>) -> Result<(), SwarmError> {
        // Update learning rate if provided
        if let Some(&learning_rate) = parameters.get("learning_rate") {
            // This would require modification of the network's learning rate
            // For now, we'll store it for the next training cycle
        }
        
        // Update other parameters as needed
        // This is a simplified implementation - in practice, you'd want more sophisticated parameter updates
        
        Ok(())
    }
    
    async fn save_processing_model(&self, path: &str) -> Result<(), SwarmError> {
        let networks = self.networks.read().await;
        let metrics = self.metrics.read().await;
        
        let save_data = serde_json::json!({
            "config": self.config,
            "metrics": *metrics,
            "network_count": networks.len(),
            "timestamp": Utc::now()
        });
        
        std::fs::write(
            format!("{}/neural_processor_metadata.json", path),
            serde_json::to_string_pretty(&save_data)
                .map_err(|e| SwarmError::SerializationError(format!("Serialization failed: {}", e)))?
        ).map_err(|e| SwarmError::IOError(format!("Failed to save metadata: {}", e)))?;
        
        // Save individual networks (this would require implementing serialization for ruv_FANN networks)
        // For now, we'll just save the metadata
        
        Ok(())
    }
    
    async fn load_processing_model(&self, path: &str) -> Result<(), SwarmError> {
        let metadata_path = format!("{}/neural_processor_metadata.json", path);
        
        let data = std::fs::read_to_string(&metadata_path)
            .map_err(|e| SwarmError::IOError(format!("Failed to read metadata: {}", e)))?;
        
        let save_data: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| SwarmError::SerializationError(format!("Deserialization failed: {}", e)))?;
        
        // Restore metrics
        if let Some(metrics_data) = save_data.get("metrics") {
            let restored_metrics: NeuralProcessorMetrics = serde_json::from_value(metrics_data.clone())
                .map_err(|e| SwarmError::SerializationError(format!("Metrics deserialization failed: {}", e)))?;
            
            let mut metrics = self.metrics.write().await;
            *metrics = restored_metrics;
        }
        
        // Load individual networks (placeholder - would require actual network deserialization)
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neural_processor_config() {
        let config = NeuralProcessorConfig::default();
        assert!(config.input_size > 0);
        assert!(!config.hidden_layers.is_empty());
        assert!(config.output_size > 0);
        assert!(config.learning_rate > 0.0);
    }
    
    #[tokio::test]
    async fn test_technical_indicator_extractor() {
        let extractor = TechnicalIndicatorExtractor::new(10);
        let feature_names = extractor.get_feature_names();
        assert!(!feature_names.is_empty());
        
        let importance = extractor.get_feature_importance();
        assert!(!importance.is_empty());
        
        // Test with mock signal data
        let mut signals = Vec::new();
        for i in 0..20 {
            let mut signal = SignalFeatures {
                signal_id: format!("test_{}", i),
                values: HashMap::new(),
                timestamp: Utc::now(),
            };
            signal.values.insert("price".to_string(), 100.0 + i as f64);
            signals.push(signal);
        }
        
        let features = extractor.extract_features(&signals).await.unwrap();
        assert!(!features.is_empty());
    }
    
    #[tokio::test]
    async fn test_neural_processor_creation() {
        let config = NeuralProcessorConfig::default();
        let processor = NeuralSignalProcessorImpl::new(config).await.unwrap();
        
        let metrics = processor.metrics.read().await;
        assert_eq!(metrics.total_signals_processed, 0);
    }
    
    #[tokio::test]
    async fn test_processor_metrics_default() {
        let metrics = NeuralProcessorMetrics::default();
        assert_eq!(metrics.total_signals_processed, 0);
        assert_eq!(metrics.average_processing_time_ms, 0.0);
        assert_eq!(metrics.model_accuracy, 0.0);
    }
}