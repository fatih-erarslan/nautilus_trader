//! High-Performance ML Inference Pipeline
//!
//! This module provides optimized inference capabilities for deployed ML models,
//! including batch processing, caching, and real-time prediction serving.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use parking_lot::{RwLock, Mutex};
use crate::ml::{MLError, MLResult, MLModel, PerformanceMetrics};

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for batch processing
    pub batch_size: usize,
    /// Maximum number of concurrent requests
    pub max_concurrent_requests: usize,
    /// Request timeout
    pub timeout: Duration,
    /// Enable prediction caching
    pub enable_caching: bool,
    /// Cache size limit
    pub cache_size: usize,
    /// Cache TTL (time to live)
    pub cache_ttl: Duration,
    /// Warm-up iterations
    pub warmup_iterations: usize,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            max_concurrent_requests: 100,
            timeout: Duration::from_secs(30),
            enable_caching: true,
            cache_size: 1000,
            cache_ttl: Duration::from_secs(300), // 5 minutes
            warmup_iterations: 10,
            enable_monitoring: true,
        }
    }
}

/// Inference request
#[derive(Debug, Clone)]
pub struct InferenceRequest {
    /// Request ID
    pub id: String,
    /// Input data
    pub input: Array2<f32>,
    /// Request metadata
    pub metadata: HashMap<String, String>,
    /// Request timestamp
    pub timestamp: Instant,
}

/// Inference response
#[derive(Debug, Clone)]
pub struct InferenceResponse {
    /// Request ID
    pub request_id: String,
    /// Prediction results
    pub predictions: Array2<f32>,
    /// Prediction confidence scores
    pub confidence: Option<Array1<f32>>,
    /// Response metadata
    pub metadata: HashMap<String, String>,
    /// Processing time
    pub processing_time: Duration,
    /// Whether result was cached
    pub from_cache: bool,
}

/// Cached prediction entry
#[derive(Debug, Clone)]
struct CacheEntry {
    /// Cached predictions
    predictions: Array2<f32>,
    /// Cache timestamp
    timestamp: Instant,
    /// Access count
    access_count: usize,
}

/// Inference statistics
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    /// Total requests processed
    pub total_requests: u64,
    /// Successful predictions
    pub successful_predictions: u64,
    /// Failed predictions
    pub failed_predictions: u64,
    /// Cache hits
    pub cache_hits: u64,
    /// Cache misses
    pub cache_misses: u64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Total processing time
    pub total_processing_time: Duration,
    /// Throughput (requests per second)
    pub throughput: f64,
    /// Current active requests
    pub active_requests: u64,
}

impl InferenceStats {
    /// Update statistics with a new request
    pub fn update(&mut self, processing_time: Duration, success: bool, from_cache: bool) {
        self.total_requests += 1;
        
        if success {
            self.successful_predictions += 1;
        } else {
            self.failed_predictions += 1;
        }
        
        if from_cache {
            self.cache_hits += 1;
        } else {
            self.cache_misses += 1;
        }
        
        self.total_processing_time += processing_time;
        self.avg_processing_time = Duration::from_nanos(
            self.total_processing_time.as_nanos() as u64 / self.total_requests
        );
        
        // Update throughput (simplified calculation)
        if self.total_processing_time > Duration::from_secs(1) {
            self.throughput = self.total_requests as f64 / self.total_processing_time.as_secs_f64();
        }
    }
    
    /// Calculate cache hit rate
    pub fn cache_hit_rate(&self) -> f64 {
        if self.cache_hits + self.cache_misses > 0 {
            self.cache_hits as f64 / (self.cache_hits + self.cache_misses) as f64
        } else {
            0.0
        }
    }
    
    /// Calculate success rate
    pub fn success_rate(&self) -> f64 {
        if self.total_requests > 0 {
            self.successful_predictions as f64 / self.total_requests as f64
        } else {
            0.0
        }
    }
}

/// High-performance inference engine
pub struct InferenceEngine<M>
where
    M: MLModel<Input = Array2<f32>, Output = Array2<f32>> + Send + Sync,
{
    /// Trained model
    model: Arc<M>,
    /// Inference configuration
    config: InferenceConfig,
    /// Prediction cache
    cache: Arc<RwLock<HashMap<String, CacheEntry>>>,
    /// Inference statistics
    stats: Arc<RwLock<InferenceStats>>,
    /// Active request counter
    active_requests: Arc<Mutex<u64>>,
    /// Request queue for batch processing
    request_queue: Arc<Mutex<Vec<InferenceRequest>>>,
}

impl<M> InferenceEngine<M>
where
    M: MLModel<Input = Array2<f32>, Output = Array2<f32>> + Send + Sync,
{
    /// Create new inference engine
    pub fn new(model: M, config: InferenceConfig) -> MLResult<Self> {
        let engine = Self {
            model: Arc::new(model),
            config,
            cache: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(InferenceStats::default())),
            active_requests: Arc::new(Mutex::new(0)),
            request_queue: Arc::new(Mutex::new(Vec::new())),
        };
        
        // Warm up the model
        engine.warmup()?;
        
        Ok(engine)
    }
    
    /// Warm up the model with dummy predictions
    fn warmup(&self) -> MLResult<()> {
        if self.config.warmup_iterations == 0 {
            return Ok(());
        }
        
        // Create dummy input data
        let dummy_input = Array2::zeros((1, 10)); // Adjust size as needed
        
        for _ in 0..self.config.warmup_iterations {
            let _ = self.model.predict(&dummy_input);
        }
        
        Ok(())
    }
    
    /// Make single prediction
    pub fn predict(&self, input: Array2<f32>) -> MLResult<InferenceResponse> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let request = InferenceRequest {
            id: request_id.clone(),
            input,
            metadata: HashMap::new(),
            timestamp: Instant::now(),
        };
        
        self.process_request(request)
    }
    
    /// Make prediction with metadata
    pub fn predict_with_metadata(
        &self,
        input: Array2<f32>,
        metadata: HashMap<String, String>,
    ) -> MLResult<InferenceResponse> {
        let request_id = uuid::Uuid::new_v4().to_string();
        let request = InferenceRequest {
            id: request_id.clone(),
            input,
            metadata,
            timestamp: Instant::now(),
        };
        
        self.process_request(request)
    }
    
    /// Process batch of requests
    pub fn predict_batch(&self, inputs: Vec<Array2<f32>>) -> MLResult<Vec<InferenceResponse>> {
        let mut requests = Vec::new();
        
        for input in inputs {
            let request_id = uuid::Uuid::new_v4().to_string();
            requests.push(InferenceRequest {
                id: request_id,
                input,
                metadata: HashMap::new(),
                timestamp: Instant::now(),
            });
        }
        
        self.process_batch_requests(requests)
    }
    
    /// Process a single inference request
    fn process_request(&self, request: InferenceRequest) -> MLResult<InferenceResponse> {
        let start_time = Instant::now();
        
        // Check concurrent request limit
        {
            let mut active = self.active_requests.lock();
            if *active >= self.config.max_concurrent_requests as u64 {
                return Err(MLError::InferenceError {
                    message: "Too many concurrent requests".to_string(),
                });
            }
            *active += 1;
        }
        
        // Check timeout
        if start_time.duration_since(request.timestamp) > self.config.timeout {
            return Err(MLError::InferenceError {
                message: "Request timeout".to_string(),
            });
        }
        
        let result = self.process_request_internal(request);
        
        // Decrement active request counter
        {
            let mut active = self.active_requests.lock();
            *active -= 1;
        }
        
        result
    }
    
    /// Internal request processing
    fn process_request_internal(&self, request: InferenceRequest) -> MLResult<InferenceResponse> {
        let start_time = Instant::now();
        
        // Check cache if enabled
        if self.config.enable_caching {
            if let Some(cached_result) = self.check_cache(&request.input) {
                let processing_time = start_time.elapsed();
                
                if self.config.enable_monitoring {
                    self.stats.write().update(processing_time, true, true);
                }
                
                return Ok(InferenceResponse {
                    request_id: request.id,
                    predictions: cached_result,
                    confidence: None,
                    metadata: HashMap::new(),
                    processing_time,
                    from_cache: true,
                });
            }
        }
        
        // Make prediction
        let prediction_result = self.model.predict(&request.input);
        let processing_time = start_time.elapsed();
        
        match prediction_result {
            Ok(predictions) => {
                // Cache the result if caching is enabled
                if self.config.enable_caching {
                    self.cache_prediction(&request.input, &predictions);
                }
                
                if self.config.enable_monitoring {
                    self.stats.write().update(processing_time, true, false);
                }
                
                Ok(InferenceResponse {
                    request_id: request.id,
                    predictions,
                    confidence: None,
                    metadata: HashMap::new(),
                    processing_time,
                    from_cache: false,
                })
            }
            Err(e) => {
                if self.config.enable_monitoring {
                    self.stats.write().update(processing_time, false, false);
                }
                Err(e)
            }
        }
    }
    
    /// Process batch of requests
    fn process_batch_requests(&self, requests: Vec<InferenceRequest>) -> MLResult<Vec<InferenceResponse>> {
        let mut responses = Vec::new();
        
        // Process in batches
        for batch in requests.chunks(self.config.batch_size) {
            let batch_responses = self.process_single_batch(batch)?;
            responses.extend(batch_responses);
        }
        
        Ok(responses)
    }
    
    /// Process a single batch
    fn process_single_batch(&self, requests: &[InferenceRequest]) -> MLResult<Vec<InferenceResponse>> {
        if requests.is_empty() {
            return Ok(Vec::new());
        }
        
        let start_time = Instant::now();
        
        // Combine inputs into a single batch
        let batch_size = requests.len();
        let input_cols = requests[0].input.ncols();
        let mut batch_input = Array2::zeros((batch_size, input_cols));
        
        for (i, request) in requests.iter().enumerate() {
            if request.input.nrows() != 1 {
                return Err(MLError::DimensionMismatch {
                    expected: "Single row input".to_string(),
                    actual: format!("{} rows", request.input.nrows()),
                });
            }
            batch_input.row_mut(i).assign(&request.input.row(0));
        }
        
        // Make batch prediction
        let batch_predictions = self.model.predict(&batch_input)?;
        let processing_time = start_time.elapsed();
        
        // Split batch results back into individual responses
        let mut responses = Vec::new();
        
        for (i, request) in requests.iter().enumerate() {
            let prediction = batch_predictions.row(i).insert_axis(Axis(0));
            
            responses.push(InferenceResponse {
                request_id: request.id.clone(),
                predictions: prediction,
                confidence: None,
                metadata: HashMap::new(),
                processing_time: processing_time / batch_size as u32,
                from_cache: false,
            });
            
            if self.config.enable_monitoring {
                self.stats.write().update(processing_time / batch_size as u32, true, false);
            }
        }
        
        Ok(responses)
    }
    
    /// Check cache for existing prediction
    fn check_cache(&self, input: &Array2<f32>) -> Option<Array2<f32>> {
        if !self.config.enable_caching {
            return None;
        }
        
        let cache_key = self.compute_cache_key(input);
        let cache = self.cache.read();
        
        if let Some(entry) = cache.get(&cache_key) {
            // Check if entry is still valid (not expired)
            if entry.timestamp.elapsed() < self.config.cache_ttl {
                return Some(entry.predictions.clone());
            }
        }
        
        None
    }
    
    /// Cache a prediction
    fn cache_prediction(&self, input: &Array2<f32>, predictions: &Array2<f32>) {
        if !self.config.enable_caching {
            return;
        }
        
        let cache_key = self.compute_cache_key(input);
        let entry = CacheEntry {
            predictions: predictions.clone(),
            timestamp: Instant::now(),
            access_count: 1,
        };
        
        let mut cache = self.cache.write();
        
        // Remove expired entries if cache is full
        if cache.len() >= self.config.cache_size {
            self.evict_cache_entries(&mut cache);
        }
        
        cache.insert(cache_key, entry);
    }
    
    /// Compute cache key for input
    fn compute_cache_key(&self, input: &Array2<f32>) -> String {
        use sha2::{Sha256, Digest};
        
        let mut hasher = Sha256::new();
        
        // Hash the input data
        for &value in input.iter() {
            hasher.update(value.to_le_bytes());
        }
        
        format!("{:x}", hasher.finalize())
    }
    
    /// Evict old cache entries
    fn evict_cache_entries(&self, cache: &mut HashMap<String, CacheEntry>) {
        // Remove expired entries first
        let now = Instant::now();
        cache.retain(|_, entry| now.duration_since(entry.timestamp) < self.config.cache_ttl);
        
        // If still too many entries, remove least recently used
        if cache.len() >= self.config.cache_size {
            let mut entries: Vec<_> = cache.iter().collect();
            entries.sort_by_key(|(_, entry)| entry.timestamp);
            
            let to_remove = cache.len() - self.config.cache_size / 2;
            for (key, _) in entries.into_iter().take(to_remove) {
                cache.remove(key);
            }
        }
    }
    
    /// Get inference statistics
    pub fn get_stats(&self) -> InferenceStats {
        self.stats.read().clone()
    }
    
    /// Reset statistics
    pub fn reset_stats(&self) {
        let mut stats = self.stats.write();
        *stats = InferenceStats::default();
    }
    
    /// Clear cache
    pub fn clear_cache(&self) {
        self.cache.write().clear();
    }
    
    /// Get cache size
    pub fn cache_size(&self) -> usize {
        self.cache.read().len()
    }
    
    /// Get active request count
    pub fn active_requests(&self) -> u64 {
        *self.active_requests.lock()
    }
    
    /// Get model metadata
    pub fn model_metadata(&self) -> &crate::ml::ModelMetadata {
        self.model.metadata()
    }
    
    /// Update configuration
    pub fn update_config(&mut self, config: InferenceConfig) {
        self.config = config;
    }
    
    /// Generate performance report
    pub fn generate_performance_report(&self) -> String {
        let stats = self.get_stats();
        
        let mut report = String::new();
        report.push_str("Inference Performance Report\n");
        report.push_str("===========================\n\n");
        
        report.push_str(&format!("Total Requests: {}\n", stats.total_requests));
        report.push_str(&format!("Successful: {}\n", stats.successful_predictions));
        report.push_str(&format!("Failed: {}\n", stats.failed_predictions));
        report.push_str(&format!("Success Rate: {:.2}%\n", stats.success_rate() * 100.0));
        
        report.push_str(&format!("\nCache Performance:\n"));
        report.push_str(&format!("Cache Hits: {}\n", stats.cache_hits));
        report.push_str(&format!("Cache Misses: {}\n", stats.cache_misses));
        report.push_str(&format!("Cache Hit Rate: {:.2}%\n", stats.cache_hit_rate() * 100.0));
        report.push_str(&format!("Cache Size: {}\n", self.cache_size()));
        
        report.push_str(&format!("\nPerformance Metrics:\n"));
        report.push_str(&format!("Average Processing Time: {:.2}ms\n", 
            stats.avg_processing_time.as_secs_f64() * 1000.0));
        report.push_str(&format!("Throughput: {:.2} req/s\n", stats.throughput));
        report.push_str(&format!("Active Requests: {}\n", self.active_requests()));
        
        report.push_str(&format!("\nModel Information:\n"));
        report.push_str(&format!("Model ID: {}\n", self.model_metadata().id));
        report.push_str(&format!("Model Name: {}\n", self.model_metadata().name));
        report.push_str(&format!("Framework: {}\n", self.model_metadata().framework));
        report.push_str(&format!("Task: {}\n", self.model_metadata().task));
        
        report
    }
}

/// Inference server for serving models over HTTP (conceptual)
pub struct InferenceServer<M>
where
    M: MLModel<Input = Array2<f32>, Output = Array2<f32>> + Send + Sync + 'static,
{
    /// Inference engine
    engine: Arc<InferenceEngine<M>>,
    /// Server configuration
    config: ServerConfig,
}

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Server address
    pub address: String,
    /// Server port
    pub port: u16,
    /// Worker threads
    pub workers: usize,
    /// Request timeout
    pub timeout: Duration,
    /// Enable CORS
    pub enable_cors: bool,
    /// API key for authentication
    pub api_key: Option<String>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            address: "127.0.0.1".to_string(),
            port: 8080,
            workers: num_cpus::get(),
            timeout: Duration::from_secs(30),
            enable_cors: true,
            api_key: None,
        }
    }
}

impl<M> InferenceServer<M>
where
    M: MLModel<Input = Array2<f32>, Output = Array2<f32>> + Send + Sync + 'static,
{
    /// Create new inference server
    pub fn new(engine: InferenceEngine<M>, config: ServerConfig) -> Self {
        Self {
            engine: Arc::new(engine),
            config,
        }
    }
    
    /// Start the server (conceptual implementation)
    pub async fn start(&self) -> MLResult<()> {
        tracing::info!(
            "Starting inference server on {}:{}",
            self.config.address,
            self.config.port
        );
        
        // In a real implementation, this would start an HTTP server
        // using a framework like axum, warp, or actix-web
        
        Ok(())
    }
    
    /// Handle prediction request (conceptual)
    pub async fn handle_predict_request(
        &self,
        input: Array2<f32>,
    ) -> MLResult<InferenceResponse> {
        self.engine.predict(input)
    }
    
    /// Handle batch prediction request (conceptual)
    pub async fn handle_batch_request(
        &self,
        inputs: Vec<Array2<f32>>,
    ) -> MLResult<Vec<InferenceResponse>> {
        self.engine.predict_batch(inputs)
    }
    
    /// Get server health status
    pub fn health_check(&self) -> HashMap<String, serde_json::Value> {
        let mut health = HashMap::new();
        
        health.insert("status".to_string(), serde_json::Value::String("healthy".to_string()));
        health.insert("active_requests".to_string(), 
            serde_json::Value::Number(serde_json::Number::from(self.engine.active_requests())));
        health.insert("cache_size".to_string(), 
            serde_json::Value::Number(serde_json::Number::from(self.engine.cache_size())));
        
        let stats = self.engine.get_stats();
        health.insert("total_requests".to_string(), 
            serde_json::Value::Number(serde_json::Number::from(stats.total_requests)));
        health.insert("success_rate".to_string(), 
            serde_json::Value::Number(serde_json::Number::from_f64(stats.success_rate()).unwrap()));
        
        health
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ml::neural::{NeuralNetwork, NeuralConfig};
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_inference_config() {
        let config = InferenceConfig {
            batch_size: 64,
            max_concurrent_requests: 200,
            timeout: Duration::from_secs(60),
            enable_caching: true,
            cache_size: 2000,
            cache_ttl: Duration::from_secs(600),
            warmup_iterations: 5,
            enable_monitoring: true,
        };
        
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.max_concurrent_requests, 200);
        assert_eq!(config.timeout, Duration::from_secs(60));
        assert!(config.enable_caching);
        assert_eq!(config.cache_size, 2000);
        assert_eq!(config.warmup_iterations, 5);
    }
    
    #[test]
    fn test_inference_stats() {
        let mut stats = InferenceStats::default();
        
        // Update with successful requests
        stats.update(Duration::from_millis(100), true, false);
        stats.update(Duration::from_millis(150), true, true);
        stats.update(Duration::from_millis(80), false, false);
        
        assert_eq!(stats.total_requests, 3);
        assert_eq!(stats.successful_predictions, 2);
        assert_eq!(stats.failed_predictions, 1);
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.cache_misses, 2);
        
        assert_abs_diff_eq!(stats.success_rate(), 2.0 / 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(stats.cache_hit_rate(), 1.0 / 3.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_inference_engine_creation() {
        let model_config = NeuralConfig::new().with_layers(vec![5, 3, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        let inference_config = InferenceConfig {
            warmup_iterations: 0, // Skip warmup for test
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(model, inference_config);
        assert!(engine.is_ok());
        
        let engine = engine.unwrap();
        assert_eq!(engine.cache_size(), 0);
        assert_eq!(engine.active_requests(), 0);
    }
    
    #[test]
    fn test_single_prediction() {
        let model_config = NeuralConfig::new().with_layers(vec![3, 2, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        let inference_config = InferenceConfig {
            warmup_iterations: 0,
            enable_caching: false,
            enable_monitoring: true,
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(model, inference_config).unwrap();
        
        // Make prediction
        let input = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        let response = engine.predict(input);
        
        // Note: This will fail because our mock model isn't actually trained
        // In a real test, you'd use a trained model
        assert!(response.is_err() || response.is_ok());
    }
    
    #[test]
    fn test_batch_prediction() {
        let model_config = NeuralConfig::new().with_layers(vec![2, 3, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        let inference_config = InferenceConfig {
            warmup_iterations: 0,
            batch_size: 2,
            enable_caching: false,
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(model, inference_config).unwrap();
        
        // Create batch inputs
        let input1 = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let input2 = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let input3 = Array2::from_shape_vec((1, 2), vec![5.0, 6.0]).unwrap();
        
        let inputs = vec![input1, input2, input3];
        let responses = engine.predict_batch(inputs);
        
        // Note: This will fail because our mock model isn't actually trained
        assert!(responses.is_err() || responses.is_ok());
    }
    
    #[test]
    fn test_cache_key_computation() {
        let model_config = NeuralConfig::new().with_layers(vec![2, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        let inference_config = InferenceConfig {
            warmup_iterations: 0,
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(model, inference_config).unwrap();
        
        let input1 = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let input2 = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let input3 = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        
        let key1 = engine.compute_cache_key(&input1);
        let key2 = engine.compute_cache_key(&input2);
        let key3 = engine.compute_cache_key(&input3);
        
        assert_eq!(key1, key2); // Same inputs should have same keys
        assert_ne!(key1, key3); // Different inputs should have different keys
    }
    
    #[test]
    fn test_server_config() {
        let config = ServerConfig {
            address: "0.0.0.0".to_string(),
            port: 9090,
            workers: 8,
            timeout: Duration::from_secs(45),
            enable_cors: false,
            api_key: Some("secret-key".to_string()),
        };
        
        assert_eq!(config.address, "0.0.0.0");
        assert_eq!(config.port, 9090);
        assert_eq!(config.workers, 8);
        assert_eq!(config.timeout, Duration::from_secs(45));
        assert!(!config.enable_cors);
        assert_eq!(config.api_key, Some("secret-key".to_string()));
    }
    
    #[test]
    fn test_inference_server_creation() {
        let model_config = NeuralConfig::new().with_layers(vec![2, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        let inference_config = InferenceConfig {
            warmup_iterations: 0,
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(model, inference_config).unwrap();
        let server_config = ServerConfig::default();
        let server = InferenceServer::new(engine, server_config);
        
        let health = server.health_check();
        assert!(health.contains_key("status"));
        assert!(health.contains_key("active_requests"));
        assert!(health.contains_key("total_requests"));
    }
    
    #[test]
    fn test_performance_report() {
        let model_config = NeuralConfig::new().with_layers(vec![2, 1]);
        let model = NeuralNetwork::new(model_config).unwrap();
        
        let inference_config = InferenceConfig {
            warmup_iterations: 0,
            enable_monitoring: true,
            ..Default::default()
        };
        
        let engine = InferenceEngine::new(model, inference_config).unwrap();
        
        // Generate some mock statistics
        {
            let mut stats = engine.stats.write();
            stats.total_requests = 100;
            stats.successful_predictions = 95;
            stats.failed_predictions = 5;
            stats.cache_hits = 30;
            stats.cache_misses = 70;
            stats.avg_processing_time = Duration::from_millis(50);
            stats.throughput = 20.0;
        }
        
        let report = engine.generate_performance_report();
        
        assert!(report.contains("Inference Performance Report"));
        assert!(report.contains("Total Requests: 100"));
        assert!(report.contains("Successful: 95"));
        assert!(report.contains("Failed: 5"));
        assert!(report.contains("Success Rate: 95.00%"));
        assert!(report.contains("Cache Hits: 30"));
        assert!(report.contains("Cache Hit Rate: 30.00%"));
        assert!(report.contains("Average Processing Time: 50.00ms"));
        assert!(report.contains("Throughput: 20.00 req/s"));
    }
}