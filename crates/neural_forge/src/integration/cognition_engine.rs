//! Cognition Engine integration for Neural Forge
//! 
//! Provides seamless integration with the cognition-engine NHITS forecasting system

use std::process::{Command, Stdio};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use serde_json;
use tracing::{info, warn, error, debug};

use crate::prelude::*;
use crate::integration::{CognitionEngineConfig, NHITSConfig, InferenceConfig};

/// Cognition Engine interface
pub struct CognitionEngine {
    config: CognitionEngineConfig,
    client: Option<CognitionClient>,
    performance_stats: Arc<RwLock<PerformanceStats>>,
    health_monitor: HealthMonitor,
}

/// Cognition Engine client for communication
pub struct CognitionClient {
    endpoint: String,
    timeout_ms: u64,
    max_retries: usize,
}

/// Performance statistics
#[derive(Debug, Clone, Default)]
pub struct PerformanceStats {
    pub total_predictions: u64,
    pub average_latency_us: f64,
    pub max_latency_us: u64,
    pub min_latency_us: u64,
    pub error_rate: f64,
    pub throughput_per_sec: f64,
    pub memory_usage_mb: f64,
}

/// Health monitoring
pub struct HealthMonitor {
    last_health_check: std::time::Instant,
    health_status: HealthStatus,
    consecutive_failures: usize,
}

/// Health status
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

/// NHITS prediction request
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NHITSRequest {
    /// Input time series data
    pub data: Vec<Vec<f64>>,
    
    /// Forecast horizon
    pub horizon: usize,
    
    /// Model configuration
    pub config: NHITSConfig,
    
    /// Request metadata
    pub metadata: RequestMetadata,
}

/// NHITS prediction response
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct NHITSResponse {
    /// Forecasted values
    pub predictions: Vec<f64>,
    
    /// Prediction intervals (if available)
    pub intervals: Option<Vec<(f64, f64)>>,
    
    /// Model confidence scores
    pub confidence: Option<Vec<f64>>,
    
    /// Response metadata
    pub metadata: ResponseMetadata,
}

/// Request metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RequestMetadata {
    /// Request ID
    pub request_id: String,
    
    /// Timestamp
    pub timestamp: u64,
    
    /// Priority level
    pub priority: u8,
    
    /// Timeout (milliseconds)
    pub timeout_ms: u64,
}

/// Response metadata
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResponseMetadata {
    /// Request ID
    pub request_id: String,
    
    /// Processing time (microseconds)
    pub processing_time_us: u64,
    
    /// Model version
    pub model_version: String,
    
    /// Status code
    pub status: ResponseStatus,
}

/// Response status
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ResponseStatus {
    Success,
    PartialSuccess,
    Error(String),
    Timeout,
}

impl CognitionEngine {
    /// Create new cognition engine instance
    pub fn new(config: CognitionEngineConfig) -> Result<Self> {
        info!("Initializing Cognition Engine integration");
        
        // Validate configuration
        config.validate()?;
        
        // Initialize client if needed
        let client = if config.enabled {
            Some(CognitionClient::new(&config)?)
        } else {
            None
        };
        
        let performance_stats = Arc::new(RwLock::new(PerformanceStats::default()));
        let health_monitor = HealthMonitor::new();
        
        Ok(Self {
            config,
            client,
            performance_stats,
            health_monitor,
        })
    }
    
    /// Make NHITS prediction
    pub async fn predict(&mut self, request: NHITSRequest) -> Result<NHITSResponse> {
        if !self.config.enabled {
            return Err(NeuralForgeError::backend("Cognition engine not enabled"));
        }
        
        let start_time = std::time::Instant::now();
        
        // Check health before prediction
        if !self.is_healthy().await {
            warn!("Cognition engine not healthy, attempting recovery");
            self.attempt_recovery().await?;
        }
        
        // Make prediction
        let response = match &self.client {
            Some(client) => client.predict(request).await?,
            None => return Err(NeuralForgeError::backend("No cognition engine client")),
        };
        
        // Update performance statistics
        let latency_us = start_time.elapsed().as_micros() as u64;
        self.update_performance_stats(latency_us, &response).await;
        
        // Check latency threshold
        if latency_us > self.config.inference.max_latency_us {
            warn!(
                "Prediction latency {}μs exceeds threshold {}μs",
                latency_us, self.config.inference.max_latency_us
            );
        }
        
        Ok(response)
    }
    
    /// Batch prediction for improved throughput
    pub async fn predict_batch(&mut self, requests: Vec<NHITSRequest>) -> Result<Vec<NHITSResponse>> {
        if !self.config.enabled {
            return Err(NeuralForgeError::backend("Cognition engine not enabled"));
        }
        
        let start_time = std::time::Instant::now();
        info!("Processing batch of {} predictions", requests.len());
        
        let mut responses = Vec::with_capacity(requests.len());
        
        // Process in parallel chunks for optimal performance
        let chunk_size = self.config.inference.batch_size;
        for chunk in requests.chunks(chunk_size) {
            let chunk_responses = self.process_chunk(chunk).await?;
            responses.extend(chunk_responses);
        }
        
        let total_time = start_time.elapsed();
        info!(
            "Batch prediction completed: {} requests in {:.2?}",
            requests.len(), total_time
        );
        
        Ok(responses)
    }
    
    /// Stream predictions for real-time processing
    pub async fn predict_stream(
        &mut self,
        mut data_stream: tokio::sync::mpsc::Receiver<NHITSRequest>,
    ) -> tokio::sync::mpsc::Receiver<Result<NHITSResponse>> {
        let (tx, rx) = tokio::sync::mpsc::channel(1000);
        
        tokio::spawn(async move {
            while let Some(request) = data_stream.recv().await {
                let result = self.predict(request).await;
                if tx.send(result).await.is_err() {
                    break;
                }
            }
        });
        
        rx
    }
    
    /// Check engine health status
    pub async fn is_healthy(&mut self) -> bool {
        if std::time::Instant::now().duration_since(self.health_monitor.last_health_check) 
            < std::time::Duration::from_millis(self.config.monitoring.health_check_frequency) {
            return self.health_monitor.health_status == HealthStatus::Healthy;
        }
        
        // Perform health check
        let health_status = self.perform_health_check().await;
        self.health_monitor.health_status = health_status.clone();
        self.health_monitor.last_health_check = std::time::Instant::now();
        
        match health_status {
            HealthStatus::Healthy => {
                self.health_monitor.consecutive_failures = 0;
                true
            }
            _ => {
                self.health_monitor.consecutive_failures += 1;
                false
            }
        }
    }
    
    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> PerformanceStats {
        self.performance_stats.read().await.clone()
    }
    
    /// Reset performance statistics
    pub async fn reset_performance_stats(&self) {
        *self.performance_stats.write().await = PerformanceStats::default();
    }
    
    /// Attempt to recover unhealthy engine
    async fn attempt_recovery(&mut self) -> Result<()> {
        info!("Attempting cognition engine recovery");
        
        // Strategy depends on failure type and configuration
        match self.config.inference.fallback_strategy {
            crate::integration::FallbackStrategy::None => {
                Err(NeuralForgeError::backend("No fallback strategy configured"))
            }
            crate::integration::FallbackStrategy::SimpleModel => {
                warn!("Using simple model fallback");
                // Implement simple model fallback
                Ok(())
            }
            crate::integration::FallbackStrategy::LastKnownGood => {
                warn!("Using last known good configuration");
                // Implement last known good fallback
                Ok(())
            }
            crate::integration::FallbackStrategy::Ensemble => {
                warn!("Using ensemble fallback");
                // Implement ensemble fallback
                Ok(())
            }
            crate::integration::FallbackStrategy::HumanIntervention => {
                error!("Human intervention required for recovery");
                Err(NeuralForgeError::backend("Human intervention required"))
            }
        }
    }
    
    /// Process prediction chunk in parallel
    async fn process_chunk(&mut self, chunk: &[NHITSRequest]) -> Result<Vec<NHITSResponse>> {
        let mut responses = Vec::with_capacity(chunk.len());
        
        // For now, process sequentially - can be parallelized later
        for request in chunk {
            let response = self.predict(request.clone()).await?;
            responses.push(response);
        }
        
        Ok(responses)
    }
    
    /// Perform health check
    async fn perform_health_check(&self) -> HealthStatus {
        match &self.client {
            Some(client) => client.health_check().await,
            None => HealthStatus::Unhealthy,
        }
    }
    
    /// Update performance statistics
    async fn update_performance_stats(&self, latency_us: u64, response: &NHITSResponse) {
        let mut stats = self.performance_stats.write().await;
        
        stats.total_predictions += 1;
        
        // Update latency statistics
        if stats.total_predictions == 1 {
            stats.min_latency_us = latency_us;
            stats.max_latency_us = latency_us;
            stats.average_latency_us = latency_us as f64;
        } else {
            stats.min_latency_us = stats.min_latency_us.min(latency_us);
            stats.max_latency_us = stats.max_latency_us.max(latency_us);
            
            // Exponential moving average
            let alpha = 0.1;
            stats.average_latency_us = alpha * (latency_us as f64) + (1.0 - alpha) * stats.average_latency_us;
        }
        
        // Update error rate
        let is_error = matches!(response.metadata.status, ResponseStatus::Error(_) | ResponseStatus::Timeout);
        if is_error {
            stats.error_rate = (stats.error_rate * (stats.total_predictions - 1) as f64 + 1.0) / stats.total_predictions as f64;
        } else {
            stats.error_rate = (stats.error_rate * (stats.total_predictions - 1) as f64) / stats.total_predictions as f64;
        }
        
        // Update throughput (moving average over last 60 seconds)
        // This is a simplified calculation - would need proper time windowing
        stats.throughput_per_sec = 1_000_000.0 / stats.average_latency_us;
    }
}

impl CognitionClient {
    /// Create new cognition client
    pub fn new(config: &CognitionEngineConfig) -> Result<Self> {
        let endpoint = format!("http://localhost:8080"); // Default endpoint
        
        Ok(Self {
            endpoint,
            timeout_ms: 5000, // 5 second timeout
            max_retries: 3,
        })
    }
    
    /// Make prediction request
    pub async fn predict(&self, request: NHITSRequest) -> Result<NHITSResponse> {
        debug!("Making NHITS prediction request: {}", request.metadata.request_id);
        
        // For now, simulate prediction - would integrate with actual cognition-engine
        let processing_start = std::time::Instant::now();
        
        // Simulate processing time based on data size
        let processing_time_us = (request.data.len() * 10) as u64; // 10μs per data point
        tokio::time::sleep(std::time::Duration::from_micros(processing_time_us)).await;
        
        // Generate mock response
        let predictions = (0..request.horizon)
            .map(|i| {
                // Simple trend extrapolation for demonstration
                if let Some(last_window) = request.data.last() {
                    if let Some(last_value) = last_window.last() {
                        last_value + (i as f64 * 0.01) // Small upward trend
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            })
            .collect();
        
        let response = NHITSResponse {
            predictions,
            intervals: None, // Would include prediction intervals in real implementation
            confidence: None, // Would include confidence scores in real implementation
            metadata: ResponseMetadata {
                request_id: request.metadata.request_id,
                processing_time_us: processing_start.elapsed().as_micros() as u64,
                model_version: "nhits-v1.0".to_string(),
                status: ResponseStatus::Success,
            },
        };
        
        debug!("NHITS prediction completed in {}μs", response.metadata.processing_time_us);
        Ok(response)
    }
    
    /// Health check
    pub async fn health_check(&self) -> HealthStatus {
        // Simulate health check - would ping actual service
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        HealthStatus::Healthy
    }
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            last_health_check: std::time::Instant::now(),
            health_status: HealthStatus::Unknown,
            consecutive_failures: 0,
        }
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Utility functions for NHITS integration
pub mod utils {
    use super::*;
    
    /// Convert time series data to NHITS format
    pub fn prepare_nhits_data(
        data: &[f64],
        window_size: usize,
        horizon: usize,
    ) -> Result<Vec<Vec<f64>>> {
        if data.len() < window_size {
            return Err(NeuralForgeError::data(
                "Insufficient data for NHITS window size"
            ));
        }
        
        let mut windows = Vec::new();
        
        for i in 0..=(data.len().saturating_sub(window_size)) {
            let window = data[i..i + window_size].to_vec();
            windows.push(window);
        }
        
        Ok(windows)
    }
    
    /// Generate request ID
    pub fn generate_request_id() -> String {
        use std::time::{SystemTime, UNIX_EPOCH};
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("nhits_{}", timestamp)
    }
    
    /// Calculate prediction accuracy
    pub fn calculate_accuracy(predictions: &[f64], actual: &[f64]) -> f64 {
        if predictions.len() != actual.len() || predictions.is_empty() {
            return 0.0;
        }
        
        let mape = predictions
            .iter()
            .zip(actual.iter())
            .map(|(pred, act)| {
                if act.abs() < f64::EPSILON {
                    0.0
                } else {
                    ((pred - act) / act).abs()
                }
            })
            .sum::<f64>() / predictions.len() as f64;
        
        (1.0 - mape).max(0.0)
    }
    
    /// Benchmark NHITS performance
    pub async fn benchmark_nhits(
        engine: &mut CognitionEngine,
        test_data: &[Vec<f64>],
        horizon: usize,
        iterations: usize,
    ) -> Result<BenchmarkResults> {
        let mut latencies = Vec::with_capacity(iterations);
        let mut errors = 0;
        
        info!("Starting NHITS benchmark with {} iterations", iterations);
        
        for i in 0..iterations {
            let request = NHITSRequest {
                data: test_data.to_vec(),
                horizon,
                config: engine.config.nhits.clone(),
                metadata: RequestMetadata {
                    request_id: format!("benchmark_{}", i),
                    timestamp: std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs(),
                    priority: 0,
                    timeout_ms: 5000,
                },
            };
            
            let start = std::time::Instant::now();
            match engine.predict(request).await {
                Ok(_) => {
                    latencies.push(start.elapsed().as_micros() as u64);
                }
                Err(e) => {
                    errors += 1;
                    warn!("Benchmark iteration {} failed: {}", i, e);
                }
            }
        }
        
        let results = BenchmarkResults::calculate(latencies, errors, iterations);
        info!("NHITS benchmark completed: {:?}", results);
        
        Ok(results)
    }
}

/// Benchmark results
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub mean_latency_us: f64,
    pub median_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
    pub min_latency_us: u64,
    pub max_latency_us: u64,
    pub error_rate: f64,
    pub throughput_per_sec: f64,
}

impl BenchmarkResults {
    pub fn calculate(mut latencies: Vec<u64>, errors: usize, total: usize) -> Self {
        if latencies.is_empty() {
            return Self {
                mean_latency_us: 0.0,
                median_latency_us: 0,
                p95_latency_us: 0,
                p99_latency_us: 0,
                min_latency_us: 0,
                max_latency_us: 0,
                error_rate: 1.0,
                throughput_per_sec: 0.0,
            };
        }
        
        latencies.sort_unstable();
        
        let mean_latency_us = latencies.iter().sum::<u64>() as f64 / latencies.len() as f64;
        let median_latency_us = latencies[latencies.len() / 2];
        let p95_latency_us = latencies[(latencies.len() as f64 * 0.95) as usize];
        let p99_latency_us = latencies[(latencies.len() as f64 * 0.99) as usize];
        let min_latency_us = latencies[0];
        let max_latency_us = latencies[latencies.len() - 1];
        let error_rate = errors as f64 / total as f64;
        let throughput_per_sec = 1_000_000.0 / mean_latency_us;
        
        Self {
            mean_latency_us,
            median_latency_us,
            p95_latency_us,
            p99_latency_us,
            min_latency_us,
            max_latency_us,
            error_rate,
            throughput_per_sec,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_cognition_engine_creation() {
        let config = CognitionEngineConfig::default();
        let engine = CognitionEngine::new(config);
        assert!(engine.is_ok());
    }
    
    #[tokio::test]
    async fn test_nhits_data_preparation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let windows = utils::prepare_nhits_data(&data, 5, 3).unwrap();
        assert_eq!(windows.len(), 6);
        assert_eq!(windows[0], vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(windows[5], vec![6.0, 7.0, 8.0, 9.0, 10.0]);
    }
    
    #[tokio::test]
    async fn test_accuracy_calculation() {
        let predictions = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let actual = vec![1.1, 1.9, 3.2, 3.8, 5.1];
        let accuracy = utils::calculate_accuracy(&predictions, &actual);
        assert!(accuracy > 0.9); // Should be high accuracy
    }
    
    #[test]
    fn test_request_id_generation() {
        let id1 = utils::generate_request_id();
        let id2 = utils::generate_request_id();
        assert_ne!(id1, id2);
        assert!(id1.starts_with("nhits_"));
    }
}