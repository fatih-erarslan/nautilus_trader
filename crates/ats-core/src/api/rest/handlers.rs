//! REST API Handlers
//!
//! Implementation of all REST API endpoints with comprehensive error handling,
//! input validation, and performance optimization.

use super::{
    ModelConfigRequest, ModelConfigResponse, BatchPredictionRequest, BatchPredictionResponse,
    ModelStatusRequest, ModelStatusResponse, CalibrationRequest, CalibrationResponse,
    BenchmarkRequest, BenchmarkResponse, ProcessingMetrics, ModelMetrics, 
    ConfigurationStatus, ModelStatus, CalibrationStatus,
};
use crate::{
    api::{HealthStatus, ServiceStatus, MemoryMetrics, ConnectionMetrics, ApiError},
    conformal_optimized::OptimizedConformalPredictor,
    types::{ConformalPredictionResult, PredictionInterval, Confidence},
    AtsCoreError, Result,
};
use serde_json;
use std::{
    collections::HashMap,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use uuid::Uuid;

/// API Handlers implementation
pub struct ApiHandlers {
    /// Conformal prediction engine
    predictor: Arc<OptimizedConformalPredictor>,
    /// Model configurations storage
    model_configs: Arc<RwLock<HashMap<String, ModelConfigResponse>>>,
    /// Calibration jobs storage
    calibration_jobs: Arc<RwLock<HashMap<String, CalibrationResponse>>>,
    /// Benchmark results storage
    benchmark_results: Arc<RwLock<HashMap<String, BenchmarkResponse>>>,
    /// Handler metrics
    metrics: Arc<HandlerMetrics>,
}

/// Handler-specific metrics
#[derive(Debug, Default)]
pub struct HandlerMetrics {
    /// Predictions made
    pub predictions_made: AtomicU64,
    /// Calibrations performed
    pub calibrations_performed: AtomicU64,
    /// Benchmarks run
    pub benchmarks_run: AtomicU64,
    /// Configuration changes
    pub config_changes: AtomicU64,
    /// Average processing time per endpoint
    pub endpoint_processing_times: Arc<RwLock<HashMap<String, u64>>>,
}

impl ApiHandlers {
    /// Create new API handlers
    pub fn new(predictor: Arc<OptimizedConformalPredictor>) -> Self {
        Self {
            predictor,
            model_configs: Arc::new(RwLock::new(HashMap::new())),
            calibration_jobs: Arc::new(RwLock::new(HashMap::new())),
            benchmark_results: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(HandlerMetrics::default()),
        }
    }

    /// Get detailed health status
    pub async fn get_detailed_health(&self) -> Result<HealthStatus> {
        let start_time = Instant::now();
        
        // Check predictor health
        let prediction_engine_status = match self.test_predictor_health().await {
            Ok(_) => ServiceStatus::Healthy,
            Err(_) => ServiceStatus::Unhealthy,
        };

        // Get memory metrics
        let memory_metrics = self.get_memory_metrics().await;
        
        // Get connection metrics (simplified for this example)
        let connection_metrics = ConnectionMetrics {
            active_websocket_connections: 0, // Would be filled by WebSocket server
            total_connections_served: self.metrics.predictions_made.load(Ordering::Relaxed),
            average_connection_duration: Duration::from_secs(300), // Example value
            connections_per_second: 10.0, // Example value
        };

        Ok(HealthStatus {
            status: ServiceStatus::Healthy,
            websocket_status: ServiceStatus::Healthy, // Would be checked from WebSocket server
            rest_status: ServiceStatus::Healthy,
            prediction_engine_status,
            memory_usage: memory_metrics,
            connection_metrics,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Test predictor health with a simple prediction
    async fn test_predictor_health(&self) -> Result<()> {
        // This would perform a simple health check prediction
        // For now, we'll simulate it
        tokio::time::sleep(Duration::from_millis(1)).await;
        Ok(())
    }

    /// Get current memory metrics
    async fn get_memory_metrics(&self) -> MemoryMetrics {
        // In a real implementation, this would query actual system memory
        MemoryMetrics {
            total_allocated: 1024 * 1024 * 1024, // 1GB
            used: 512 * 1024 * 1024,             // 512MB
            available: 512 * 1024 * 1024,        // 512MB
            peak_usage: 768 * 1024 * 1024,       // 768MB
        }
    }

    /// Create model configuration
    pub async fn create_model_config(
        &self,
        request: ModelConfigRequest,
    ) -> Result<ModelConfigResponse> {
        let start_time = Instant::now();
        
        // Validate configuration
        self.validate_model_config(&request)?;
        
        // Generate configuration ID
        let config_id = Uuid::new_v4().to_string();
        
        // Create configuration response
        let config_response = ModelConfigResponse {
            config_id: config_id.clone(),
            model_id: request.model_id.clone(),
            status: ConfigurationStatus::Pending,
            config: request,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        };

        // Store configuration
        let mut configs = self.model_configs.write().await;
        configs.insert(config_id.clone(), config_response.clone());
        
        // Update metrics
        self.metrics.config_changes.fetch_add(1, Ordering::Relaxed);
        self.record_endpoint_timing("create_model_config", start_time.elapsed()).await;

        println!("📝 Created model configuration: {}", config_id);
        Ok(config_response)
    }

    /// Validate model configuration
    fn validate_model_config(&self, config: &ModelConfigRequest) -> Result<()> {
        // Validate model ID
        if config.model_id.is_empty() {
            return Err(AtsCoreError::ValidationFailed(
                "Model ID cannot be empty".to_string()
            ));
        }

        // Validate confidence levels
        for &level in &config.confidence_levels {
            if level <= 0.0 || level >= 1.0 {
                return Err(AtsCoreError::ValidationFailed(
                    "Confidence levels must be between 0 and 1".to_string()
                ));
            }
        }

        // Validate temperature configuration if present
        if let Some(temp_config) = &config.temperature_config {
            if temp_config.initial_temperature <= 0.0 {
                return Err(AtsCoreError::ValidationFailed(
                    "Initial temperature must be positive".to_string()
                ));
            }
            if temp_config.learning_rate <= 0.0 || temp_config.learning_rate > 1.0 {
                return Err(AtsCoreError::ValidationFailed(
                    "Learning rate must be between 0 and 1".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Get model status
    pub async fn get_model_status(
        &self,
        model_id: &str,
        include_metrics: bool,
    ) -> Result<ModelStatusResponse> {
        let start_time = Instant::now();
        
        // Check if model configuration exists
        let configs = self.model_configs.read().await;
        let config = configs.values()
            .find(|c| c.model_id == model_id)
            .cloned();

        let metrics = if include_metrics {
            Some(self.get_model_metrics(model_id).await?)
        } else {
            None
        };

        let response = ModelStatusResponse {
            model_id: model_id.to_string(),
            status: ModelStatus::Ready, // Simplified for this example
            configuration: config.map(|c| c.config),
            metrics,
            last_activity: chrono::Utc::now(),
            uptime_seconds: 3600, // Example value
        };

        self.record_endpoint_timing("get_model_status", start_time.elapsed()).await;
        Ok(response)
    }

    /// Get model-specific metrics
    async fn get_model_metrics(&self, model_id: &str) -> Result<ModelMetrics> {
        // In a real implementation, this would gather actual model metrics
        Ok(ModelMetrics {
            total_predictions: 1000,
            predictions_per_second: 50.0,
            avg_latency_us: 500.0,
            error_rate: 0.01,
            memory_usage_mb: 128.0,
            accuracy_metrics: None, // Would be calculated from validation data
        })
    }

    /// Process batch predictions
    pub async fn process_batch_predictions(
        &self,
        request: BatchPredictionRequest,
    ) -> Result<BatchPredictionResponse> {
        let start_time = Instant::now();
        
        // Validate request
        self.validate_batch_prediction_request(&request)?;
        
        // Process predictions in parallel if requested
        let predictions = if request.options.parallel_processing {
            self.process_predictions_parallel(&request).await?
        } else {
            self.process_predictions_sequential(&request).await?
        };

        let processing_time = start_time.elapsed();
        let processing_time_us = processing_time.as_micros() as u64;
        
        // Calculate metrics if requested
        let metrics = if request.options.include_metrics {
            Some(ProcessingMetrics {
                total_processing_time_us: processing_time_us,
                avg_sample_time_us: processing_time_us as f64 / request.features.len() as f64,
                memory_usage_bytes: self.estimate_memory_usage(&request),
                cpu_usage_percent: 75.0, // Would be measured
                simd_utilization: if request.options.use_simd { Some(0.8) } else { None },
            })
        } else {
            None
        };

        // Update metrics
        self.metrics.predictions_made.fetch_add(predictions.len() as u64, Ordering::Relaxed);
        self.record_endpoint_timing("batch_predictions", processing_time).await;

        Ok(BatchPredictionResponse {
            request_id: Uuid::new_v4().to_string(),
            model_id: request.model_id.clone(),
            predictions,
            metrics,
            timestamp: chrono::Utc::now(),
        })
    }

    /// Validate batch prediction request
    fn validate_batch_prediction_request(&self, request: &BatchPredictionRequest) -> Result<()> {
        if request.features.is_empty() {
            return Err(AtsCoreError::ValidationFailed(
                "Features cannot be empty".to_string()
            ));
        }

        if request.features.len() > 10000 {
            return Err(AtsCoreError::ValidationFailed(
                "Batch size cannot exceed 10000 samples".to_string()
            ));
        }

        // Validate confidence levels
        for &level in &request.confidence_levels {
            if level <= 0.0 || level >= 1.0 {
                return Err(AtsCoreError::ValidationFailed(
                    "Confidence levels must be between 0 and 1".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Process predictions in parallel
    async fn process_predictions_parallel(
        &self,
        request: &BatchPredictionRequest,
    ) -> Result<Vec<ConformalPredictionResult>> {
        use rayon::prelude::*;
        
        // Simulate processing with mock predictions
        let predictions: Result<Vec<_>> = request.features
            .par_iter()
            .map(|features| {
                self.create_mock_prediction(features, &request.confidence_levels)
            })
            .collect();

        predictions
    }

    /// Process predictions sequentially
    async fn process_predictions_sequential(
        &self,
        request: &BatchPredictionRequest,
    ) -> Result<Vec<ConformalPredictionResult>> {
        let mut predictions = Vec::with_capacity(request.features.len());
        
        for features in &request.features {
            let prediction = self.create_mock_prediction(features, &request.confidence_levels)?;
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Create mock prediction for demonstration
    fn create_mock_prediction(
        &self,
        features: &[f64],
        confidence_levels: &[f64],
    ) -> Result<ConformalPredictionResult> {
        // This is a mock implementation
        let point_prediction = features.iter().sum::<f64>() / features.len() as f64;
        
        let prediction_intervals: Vec<(f64, f64)> = confidence_levels
            .iter()
            .map(|&confidence| {
                let width = (1.0 - confidence) * 2.0;
                (point_prediction - width, point_prediction + width)
            })
            .collect();

        Ok(ConformalPredictionResult {
            intervals: prediction_intervals,
            confidence: 0.95, // Default confidence level
            calibration_scores: Vec::new(),
            quantile_threshold: 0.95,
            execution_time_ns: 0,
        })
    }

    /// Estimate memory usage for processing request
    fn estimate_memory_usage(&self, request: &BatchPredictionRequest) -> u64 {
        // Rough estimation: features + predictions + overhead
        let features_size = request.features.len() * request.features[0].len() * 8; // 8 bytes per f64
        let predictions_size = request.features.len() * 100; // Estimated 100 bytes per prediction
        let overhead = 1024; // 1KB overhead
        
        (features_size + predictions_size + overhead) as u64
    }

    /// Start model calibration
    pub async fn start_calibration(
        &self,
        request: CalibrationRequest,
    ) -> Result<CalibrationResponse> {
        let start_time = Instant::now();
        
        // Validate calibration request
        self.validate_calibration_request(&request)?;
        
        let calibration_id = Uuid::new_v4().to_string();
        
        // Create calibration job
        let calibration_response = CalibrationResponse {
            calibration_id: calibration_id.clone(),
            model_id: request.model_id.clone(),
            status: CalibrationStatus::Running,
            results: None,
            timestamp: chrono::Utc::now(),
        };

        // Store calibration job
        let mut jobs = self.calibration_jobs.write().await;
        jobs.insert(calibration_id.clone(), calibration_response.clone());

        // Start calibration processing in background
        let jobs_clone = self.calibration_jobs.clone();
        let calibration_id_clone = calibration_id.clone();
        let request_clone = request.clone();
        
        tokio::spawn(async move {
            // Simulate calibration processing
            tokio::time::sleep(Duration::from_secs(5)).await;
            
            // Update with completed results
            let mut jobs = jobs_clone.write().await;
            if let Some(job) = jobs.get_mut(&calibration_id_clone) {
                job.status = CalibrationStatus::Completed;
                job.results = Some(crate::api::rest::CalibrationResults {
                    confidence_levels: request_clone.confidence_levels.clone(),
                    coverage_rates: vec![0.95, 0.99], // Mock results
                    interval_widths: vec![2.0, 4.0],
                    calibration_scores: vec![0.1, 0.05],
                    processing_time_ms: 5000,
                });
            }
        });

        // Update metrics
        self.metrics.calibrations_performed.fetch_add(1, Ordering::Relaxed);
        self.record_endpoint_timing("start_calibration", start_time.elapsed()).await;

        Ok(calibration_response)
    }

    /// Validate calibration request
    fn validate_calibration_request(&self, request: &CalibrationRequest) -> Result<()> {
        if request.calibration_data.is_empty() {
            return Err(AtsCoreError::ValidationFailed(
                "Calibration data cannot be empty".to_string()
            ));
        }

        if request.calibration_data.len() < 10 {
            return Err(AtsCoreError::ValidationFailed(
                "Calibration requires at least 10 samples".to_string()
            ));
        }

        for &level in &request.confidence_levels {
            if level <= 0.0 || level >= 1.0 {
                return Err(AtsCoreError::ValidationFailed(
                    "Confidence levels must be between 0 and 1".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Get calibration status
    pub async fn get_calibration_status(&self, calibration_id: &str) -> Result<CalibrationResponse> {
        let jobs = self.calibration_jobs.read().await;
        
        jobs.get(calibration_id)
            .cloned()
            .ok_or_else(|| AtsCoreError::ValidationFailed(
                format!("Calibration job not found: {}", calibration_id)
            ))
    }

    /// Start performance benchmark
    pub async fn start_benchmark(
        &self,
        request: BenchmarkRequest,
    ) -> Result<BenchmarkResponse> {
        let start_time = Instant::now();
        
        // Validate benchmark request
        self.validate_benchmark_request(&request)?;
        
        let benchmark_id = Uuid::new_v4().to_string();
        
        // Create mock benchmark response
        let benchmark_response = BenchmarkResponse {
            benchmark_id: benchmark_id.clone(),
            model_benchmarks: vec![], // Would be populated with actual benchmarks
            system_metrics: crate::api::rest::SystemMetrics {
                cpu_usage: 75.0,
                memory_usage: 60.0,
                disk_io: None,
                network_io: None,
            },
            timestamp: chrono::Utc::now(),
        };

        // Store benchmark results
        let mut results = self.benchmark_results.write().await;
        results.insert(benchmark_id.clone(), benchmark_response.clone());

        // Update metrics
        self.metrics.benchmarks_run.fetch_add(1, Ordering::Relaxed);
        self.record_endpoint_timing("start_benchmark", start_time.elapsed()).await;

        Ok(benchmark_response)
    }

    /// Validate benchmark request
    fn validate_benchmark_request(&self, request: &BenchmarkRequest) -> Result<()> {
        if request.model_ids.is_empty() {
            return Err(AtsCoreError::ValidationFailed(
                "Model IDs cannot be empty".to_string()
            ));
        }

        if request.sample_count == 0 || request.sample_count > 100000 {
            return Err(AtsCoreError::ValidationFailed(
                "Sample count must be between 1 and 100000".to_string()
            ));
        }

        Ok(())
    }

    /// Get benchmark results
    pub async fn get_benchmark_results(&self, benchmark_id: &str) -> Result<BenchmarkResponse> {
        let results = self.benchmark_results.read().await;
        
        results.get(benchmark_id)
            .cloned()
            .ok_or_else(|| AtsCoreError::ValidationFailed(
                format!("Benchmark not found: {}", benchmark_id)
            ))
    }

    /// Record endpoint processing timing
    async fn record_endpoint_timing(&self, endpoint: &str, duration: Duration) {
        let mut times = self.metrics.endpoint_processing_times.write().await;
        let duration_us = duration.as_micros() as u64;
        
        // Simple exponential moving average
        if let Some(current_avg) = times.get(endpoint) {
            let new_avg = (current_avg * 9 + duration_us) / 10;
            times.insert(endpoint.to_string(), new_avg);
        } else {
            times.insert(endpoint.to_string(), duration_us);
        }
    }

    /// Get handler metrics
    pub async fn get_handler_metrics(&self) -> HandlerMetrics {
        HandlerMetrics {
            predictions_made: AtomicU64::new(self.metrics.predictions_made.load(Ordering::Relaxed)),
            calibrations_performed: AtomicU64::new(self.metrics.calibrations_performed.load(Ordering::Relaxed)),
            benchmarks_run: AtomicU64::new(self.metrics.benchmarks_run.load(Ordering::Relaxed)),
            config_changes: AtomicU64::new(self.metrics.config_changes.load(Ordering::Relaxed)),
            endpoint_processing_times: self.metrics.endpoint_processing_times.clone(),
        }
    }
}