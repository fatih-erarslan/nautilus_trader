//! Quantum-Hive integration for CEFLANN-ELM
//! 
//! Seamless integration with the quantum-hive trading system, providing
//! real-time neuromorphic signal processing for ultra-fast market analysis.

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use tokio::sync::{mpsc, broadcast};
use serde::{Serialize, Deserialize};
use anyhow::{Result, anyhow};
use tracing::{info, debug, warn, error};
use nalgebra::{DMatrix, DVector};

// use quantum_hive::{
//     HiveNode, SignalProcessor, QuantumSignal, MarketData,
//     NeuralHive, TradingSignal, PerformanceMetrics as HiveMetrics
// }; // Disabled due to circular dependency

use crate::{CEFLANN, ELMConfig, PerformanceMetrics};

/// CEFLANN-ELM integration adapter for quantum-hive
pub struct CEFLANNHiveAdapter {
    /// ELM model instance
    elm_model: Arc<Mutex<CEFLANN>>,
    
    /// Configuration
    config: HiveIntegrationConfig,
    
    /// Signal input channel
    signal_rx: mpsc::Receiver<QuantumSignal>,
    
    /// Trading signal output channel
    trading_tx: broadcast::Sender<TradingSignal>,
    
    /// Performance monitoring
    performance_metrics: Arc<Mutex<HivePerformanceMetrics>>,
    
    /// Feature buffer for batch processing
    feature_buffer: CircularBuffer<DVector<f64>>,
    
    /// Prediction cache for real-time trading
    prediction_cache: PredictionCache,
}

/// Configuration for hive integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveIntegrationConfig {
    /// Maximum batch size for processing
    pub max_batch_size: usize,
    
    /// Buffer size for streaming data
    pub buffer_size: usize,
    
    /// Prediction update frequency (Hz)
    pub update_frequency: f64,
    
    /// Enable real-time feature extraction
    pub real_time_features: bool,
    
    /// Trading signal confidence threshold
    pub confidence_threshold: f64,
    
    /// Enable adaptive retraining
    pub adaptive_retraining: bool,
    
    /// Retraining window size
    pub retraining_window: usize,
    
    /// Performance monitoring interval (seconds)
    pub monitoring_interval: u64,
}

impl Default for HiveIntegrationConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 128,
            buffer_size: 1024,
            update_frequency: 1000.0, // 1kHz
            real_time_features: true,
            confidence_threshold: 0.7,
            adaptive_retraining: true,
            retraining_window: 10000,
            monitoring_interval: 10,
        }
    }
}

/// Extended performance metrics for hive integration
#[derive(Debug, Clone, Default)]
pub struct HivePerformanceMetrics {
    pub signals_processed: u64,
    pub predictions_generated: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub retraining_events: u64,
    pub average_latency_ns: u64,
    pub throughput_signals_per_sec: f64,
    pub prediction_accuracy: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
}

/// Circular buffer for efficient streaming data management
pub struct CircularBuffer<T> {
    buffer: Vec<Option<T>>,
    capacity: usize,
    head: usize,
    size: usize,
}

/// Prediction cache for ultra-fast lookups
pub struct PredictionCache {
    cache: HashMap<u64, CachedPrediction>,
    max_size: usize,
    ttl_seconds: u64,
}

/// Cached prediction with metadata
#[derive(Debug, Clone)]
struct CachedPrediction {
    prediction: DVector<f64>,
    confidence: f64,
    timestamp: u64,
    features_hash: u64,
}

impl CEFLANNHiveAdapter {
    /// Create new hive adapter
    pub fn new(
        elm_config: ELMConfig,
        hive_config: HiveIntegrationConfig,
    ) -> Result<Self> {
        info!("Initializing CEFLANN-Hive integration adapter");
        
        // Initialize ELM model
        let elm_model = Arc::new(Mutex::new(CEFLANN::new(elm_config)?));
        
        // Create communication channels
        let (signal_tx, signal_rx) = mpsc::channel(hive_config.buffer_size);
        let (trading_tx, _) = broadcast::channel(hive_config.buffer_size);
        
        // Initialize components
        let feature_buffer = CircularBuffer::new(hive_config.buffer_size);
        let prediction_cache = PredictionCache::new(1000, 60); // 1000 entries, 60s TTL
        
        Ok(Self {
            elm_model,
            config: hive_config,
            signal_rx,
            trading_tx,
            performance_metrics: Arc::new(Mutex::new(HivePerformanceMetrics::default())),
            feature_buffer,
            prediction_cache,
        })
    }
    
    /// Start the hive integration service
    pub async fn start(&mut self) -> Result<()> {
        info!("Starting CEFLANN-Hive integration service");
        
        // Spawn performance monitoring task
        let metrics_clone = Arc::clone(&self.performance_metrics);
        let monitoring_interval = self.config.monitoring_interval;
        tokio::spawn(async move {
            Self::monitor_performance(metrics_clone, monitoring_interval).await;
        });
        
        // Main processing loop
        loop {
            tokio::select! {
                // Process incoming quantum signals
                Some(signal) = self.signal_rx.recv() => {
                    if let Err(e) = self.process_quantum_signal(signal).await {
                        error!("Error processing quantum signal: {}", e);
                    }
                }
                
                // Periodic batch processing
                _ = tokio::time::sleep(tokio::time::Duration::from_millis(
                    (1000.0 / self.config.update_frequency) as u64
                )) => {
                    if let Err(e) = self.process_batch().await {
                        error!("Error in batch processing: {}", e);
                    }
                }
            }
        }
    }
    
    /// Process incoming quantum signal
    async fn process_quantum_signal(&mut self, signal: QuantumSignal) -> Result<()> {
        let start_time = std::time::Instant::now();
        
        // Extract features from quantum signal
        let features = self.extract_features(&signal)?;
        
        // Add to buffer
        self.feature_buffer.push(features.clone());
        
        // Real-time prediction if enabled
        if self.config.real_time_features {
            let prediction = self.predict_single(features).await?;
            
            // Generate trading signal if confidence is high enough
            if prediction.confidence >= self.config.confidence_threshold {
                let trading_signal = TradingSignal {
                    asset: signal.asset.clone(),
                    action: self.interpret_prediction(&prediction.prediction),
                    confidence: prediction.confidence,
                    timestamp: chrono::Utc::now(),
                    metadata: HashMap::from([
                        ("model".to_string(), "CEFLANN-ELM".to_string()),
                        ("latency_ns".to_string(), start_time.elapsed().as_nanos().to_string()),
                    ]),
                };
                
                // Broadcast trading signal
                if let Err(e) = self.trading_tx.send(trading_signal) {
                    warn!("Failed to broadcast trading signal: {}", e);
                }
            }
        }
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.signals_processed += 1;
            metrics.average_latency_ns = 
                (metrics.average_latency_ns + start_time.elapsed().as_nanos() as u64) / 2;
        }
        
        Ok(())
    }
    
    /// Process batch of buffered features
    async fn process_batch(&mut self) -> Result<()> {
        if self.feature_buffer.is_empty() {
            return Ok(());
        }
        
        let start_time = std::time::Instant::now();
        
        // Extract batch from buffer
        let batch_size = self.config.max_batch_size.min(self.feature_buffer.len());
        let features_batch: Vec<DVector<f64>> = (0..batch_size)
            .filter_map(|_| self.feature_buffer.pop())
            .collect();
        
        if features_batch.is_empty() {
            return Ok(());
        }
        
        // Convert to matrix for batch prediction
        let input_matrix = self.vectors_to_matrix(&features_batch)?;
        
        // Batch prediction
        let mut elm = self.elm_model.lock().unwrap();
        let predictions = elm.predict(&input_matrix)?;
        drop(elm);
        
        // Process predictions
        for (i, prediction_row) in predictions.row_iter().enumerate() {
            let prediction = prediction_row.transpose();
            let confidence = self.calculate_confidence(&prediction);
            
            // Cache prediction
            let features_hash = self.hash_features(&features_batch[i]);
            self.prediction_cache.insert(CachedPrediction {
                prediction: prediction.clone(),
                confidence,
                timestamp: chrono::Utc::now().timestamp() as u64,
                features_hash,
            });
            
            // Generate trading signal if needed
            if confidence >= self.config.confidence_threshold {
                // Implementation would generate and broadcast trading signal
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.predictions_generated += predictions.nrows() as u64;
            let batch_time = start_time.elapsed().as_millis() as f64 / 1000.0;
            metrics.throughput_signals_per_sec = 
                predictions.nrows() as f64 / batch_time.max(0.001);
        }
        
        debug!("Processed batch of {} predictions in {}ms", 
               predictions.nrows(), start_time.elapsed().as_millis());
        
        Ok(())
    }
    
    /// Single prediction with caching
    async fn predict_single(&mut self, features: DVector<f64>) -> Result<CachedPrediction> {
        let features_hash = self.hash_features(&features);
        
        // Check cache first
        if let Some(cached) = self.prediction_cache.get(&features_hash) {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.cache_hits += 1;
            return Ok(cached.clone());
        }
        
        // Make new prediction
        let mut elm = self.elm_model.lock().unwrap();
        let prediction = elm.predict_single(&features)?;
        drop(elm);
        
        let confidence = self.calculate_confidence(&prediction);
        let cached_prediction = CachedPrediction {
            prediction,
            confidence,
            timestamp: chrono::Utc::now().timestamp() as u64,
            features_hash,
        };
        
        // Cache result
        self.prediction_cache.insert(cached_prediction.clone());
        
        // Update metrics
        {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.cache_misses += 1;
            metrics.predictions_generated += 1;
        }
        
        Ok(cached_prediction)
    }
    
    /// Extract features from quantum signal
    fn extract_features(&self, signal: &QuantumSignal) -> Result<DVector<f64>> {
        // Convert quantum signal to feature vector
        let mut features = Vec::new();
        
        // Price features
        features.push(signal.price);
        features.push(signal.volume);
        
        // Quantum coherence features
        if let Some(coherence) = &signal.quantum_coherence {
            features.extend_from_slice(&coherence.amplitudes);
            features.extend_from_slice(&coherence.phases);
        }
        
        // Technical indicators
        if let Some(indicators) = &signal.technical_indicators {
            features.extend(indicators.values());
        }
        
        // Pad or truncate to expected dimension
        let expected_dim = 8; // From ELM config
        features.resize(expected_dim, 0.0);
        
        Ok(DVector::from_vec(features))
    }
    
    /// Convert vector list to matrix
    fn vectors_to_matrix(&self, vectors: &[DVector<f64>]) -> Result<DMatrix<f64>> {
        if vectors.is_empty() {
            return Err(anyhow!("Empty vector list"));
        }
        
        let rows = vectors.len();
        let cols = vectors[0].len();
        
        let mut data = Vec::with_capacity(rows * cols);
        for vector in vectors {
            data.extend_from_slice(vector.as_slice());
        }
        
        Ok(DMatrix::from_row_slice(rows, cols, &data))
    }
    
    /// Calculate prediction confidence
    fn calculate_confidence(&self, prediction: &DVector<f64>) -> f64 {
        // Simple confidence based on magnitude and stability
        let magnitude = prediction.norm();
        let stability = 1.0 / (1.0 + prediction.variance());
        
        (magnitude * stability).min(1.0)
    }
    
    /// Hash features for caching
    fn hash_features(&self, features: &DVector<f64>) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        for &value in features.iter() {
            (value * 1e6) as i64.hash(&mut hasher); // Hash with precision
        }
        hasher.finish()
    }
    
    /// Interpret prediction as trading action
    fn interpret_prediction(&self, prediction: &DVector<f64>) -> String {
        if prediction[0] > 0.1 {
            "BUY".to_string()
        } else if prediction[0] < -0.1 {
            "SELL".to_string()
        } else {
            "HOLD".to_string()
        }
    }
    
    /// Performance monitoring task
    async fn monitor_performance(
        metrics: Arc<Mutex<HivePerformanceMetrics>>,
        interval_seconds: u64,
    ) {
        let mut interval = tokio::time::interval(
            tokio::time::Duration::from_secs(interval_seconds)
        );
        
        loop {
            interval.tick().await;
            
            let metrics_snapshot = {
                let metrics = metrics.lock().unwrap();
                metrics.clone()
            };
            
            info!("CEFLANN-Hive Performance: {} signals/sec, {} predictions, {:.2}% cache hit rate",
                  metrics_snapshot.throughput_signals_per_sec,
                  metrics_snapshot.predictions_generated,
                  (metrics_snapshot.cache_hits as f64 / 
                   (metrics_snapshot.cache_hits + metrics_snapshot.cache_misses).max(1) as f64) * 100.0);
        }
    }
    
    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> HivePerformanceMetrics {
        self.performance_metrics.lock().unwrap().clone()
    }
    
    /// Adaptive retraining based on performance
    pub async fn adaptive_retrain(&mut self, new_data: &[(DVector<f64>, DVector<f64>)]) -> Result<()> {
        if !self.config.adaptive_retraining || new_data.len() < 100 {
            return Ok(());
        }
        
        info!("Starting adaptive retraining with {} new samples", new_data.len());
        
        // Convert training data to matrices
        let inputs = self.training_data_to_matrix(&new_data.iter().map(|(x, _)| x.clone()).collect::<Vec<_>>())?;
        let targets = self.training_data_to_matrix(&new_data.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>())?;
        
        // Retrain model
        {
            let mut elm = self.elm_model.lock().unwrap();
            elm.train(&inputs, &targets)?;
        }
        
        // Update metrics
        {
            let mut metrics = self.performance_metrics.lock().unwrap();
            metrics.retraining_events += 1;
        }
        
        // Clear prediction cache after retraining
        self.prediction_cache.clear();
        
        info!("Adaptive retraining completed successfully");
        
        Ok(())
    }
    
    /// Convert training data to matrix
    fn training_data_to_matrix(&self, data: &[DVector<f64>]) -> Result<DMatrix<f64>> {
        if data.is_empty() {
            return Err(anyhow!("Empty training data"));
        }
        
        let rows = data.len();
        let cols = data[0].len();
        
        let mut matrix_data = Vec::with_capacity(rows * cols);
        for vector in data {
            matrix_data.extend_from_slice(vector.as_slice());
        }
        
        Ok(DMatrix::from_row_slice(rows, cols, &matrix_data))
    }
}

impl<T> CircularBuffer<T> {
    fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![None; capacity],
            capacity,
            head: 0,
            size: 0,
        }
    }
    
    fn push(&mut self, item: T) {
        self.buffer[self.head] = Some(item);
        self.head = (self.head + 1) % self.capacity;
        if self.size < self.capacity {
            self.size += 1;
        }
    }
    
    fn pop(&mut self) -> Option<T> {
        if self.size == 0 {
            return None;
        }
        
        let index = (self.head + self.capacity - self.size) % self.capacity;
        let item = self.buffer[index].take();
        self.size -= 1;
        item
    }
    
    fn len(&self) -> usize {
        self.size
    }
    
    fn is_empty(&self) -> bool {
        self.size == 0
    }
}

impl PredictionCache {
    fn new(max_size: usize, ttl_seconds: u64) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            ttl_seconds,
        }
    }
    
    fn insert(&mut self, prediction: CachedPrediction) {
        let hash = prediction.features_hash;
        
        // Remove expired entries
        self.cleanup_expired();
        
        // Ensure capacity
        if self.cache.len() >= self.max_size {
            // Remove oldest entry
            if let Some(&oldest_key) = self.cache.keys().next() {
                self.cache.remove(&oldest_key);
            }
        }
        
        self.cache.insert(hash, prediction);
    }
    
    fn get(&mut self, hash: &u64) -> Option<&CachedPrediction> {
        self.cleanup_expired();
        self.cache.get(hash)
    }
    
    fn clear(&mut self) {
        self.cache.clear();
    }
    
    fn cleanup_expired(&mut self) {
        let current_time = chrono::Utc::now().timestamp() as u64;
        let ttl = self.ttl_seconds;
        
        self.cache.retain(|_, prediction| {
            current_time - prediction.timestamp < ttl
        });
    }
}

/// Hive node implementation for CEFLANN-ELM
pub struct CEFLANNHiveNode {
    adapter: CEFLANNHiveAdapter,
    node_id: String,
}

impl CEFLANNHiveNode {
    pub fn new(
        node_id: String,
        elm_config: ELMConfig,
        hive_config: HiveIntegrationConfig,
    ) -> Result<Self> {
        let adapter = CEFLANNHiveAdapter::new(elm_config, hive_config)?;
        
        Ok(Self {
            adapter,
            node_id,
        })
    }
}

impl HiveNode for CEFLANNHiveNode {
    fn node_id(&self) -> &str {
        &self.node_id
    }
    
    fn process_signal(&mut self, signal: QuantumSignal) -> Result<TradingSignal> {
        // Synchronous wrapper for async processing
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(async {
            self.adapter.process_quantum_signal(signal).await?;
            // Return a placeholder trading signal
            Ok(TradingSignal {
                asset: "PLACEHOLDER".to_string(),
                action: "HOLD".to_string(),
                confidence: 0.5,
                timestamp: chrono::Utc::now(),
                metadata: HashMap::new(),
            })
        })
    }
    
    fn get_metrics(&self) -> HiveMetrics {
        let ceflann_metrics = self.adapter.get_performance_metrics();
        
        HiveMetrics {
            node_id: self.node_id.clone(),
            signals_processed: ceflann_metrics.signals_processed,
            average_latency_ms: ceflann_metrics.average_latency_ns as f64 / 1_000_000.0,
            throughput: ceflann_metrics.throughput_signals_per_sec,
            accuracy: ceflann_metrics.prediction_accuracy,
            sharpe_ratio: ceflann_metrics.sharpe_ratio,
            max_drawdown: ceflann_metrics.max_drawdown,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_circular_buffer() {
        let mut buffer = CircularBuffer::new(3);
        
        assert!(buffer.is_empty());
        
        buffer.push(1);
        buffer.push(2);
        buffer.push(3);
        
        assert_eq!(buffer.len(), 3);
        
        buffer.push(4); // Should overwrite oldest
        
        assert_eq!(buffer.pop(), Some(2));
        assert_eq!(buffer.pop(), Some(3));
        assert_eq!(buffer.pop(), Some(4));
        assert_eq!(buffer.pop(), None);
    }
    
    #[test]
    fn test_prediction_cache() {
        let mut cache = PredictionCache::new(2, 3600);
        
        let pred1 = CachedPrediction {
            prediction: DVector::from_vec(vec![0.1, 0.2]),
            confidence: 0.8,
            timestamp: chrono::Utc::now().timestamp() as u64,
            features_hash: 12345,
        };
        
        cache.insert(pred1.clone());
        
        let retrieved = cache.get(&12345).unwrap();
        assert_relative_eq!(retrieved.confidence, 0.8);
    }
    
    #[test]
    fn test_feature_extraction() {
        let config = HiveIntegrationConfig::default();
        let elm_config = ELMConfig::default();
        let adapter = CEFLANNHiveAdapter::new(elm_config, config).unwrap();
        
        let signal = QuantumSignal {
            asset: "BTC-USD".to_string(),
            price: 50000.0,
            volume: 1000.0,
            quantum_coherence: None,
            technical_indicators: None,
            timestamp: chrono::Utc::now(),
        };
        
        let features = adapter.extract_features(&signal).unwrap();
        assert_eq!(features.len(), 8);
        assert_relative_eq!(features[0], 50000.0);
        assert_relative_eq!(features[1], 1000.0);
    }
}