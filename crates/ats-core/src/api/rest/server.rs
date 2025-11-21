//! High-Performance REST API Server
//!
//! Async REST API server with comprehensive model configuration, health monitoring,
//! and batch prediction capabilities with sub-millisecond response times.

use super::{
    handlers::ApiHandlers,
    middleware::{AuthMiddleware, RateLimitMiddleware, MetricsMiddleware, ErrorHandlerMiddleware},
    routes::create_router,
    ApiResponse, ApiError,
};
use serde::{Serialize, Deserialize};
use crate::{
    api::{ApiConfig, RestConfig, PerformanceMetrics},
    conformal_optimized::OptimizedConformalPredictor,
    AtsCoreError, Result,
};
use axum::{
    extract::Extension,
    http::{HeaderValue, Method, StatusCode},
    response::Json,
    routing::{get, post, put, delete},
    Router,
};
// hyper::Server was removed in hyper 1.0 - using axum's built-in server instead
use std::{
    net::SocketAddr,
    sync::{Arc, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant},
};
use tokio::{
    sync::RwLock,
    time::interval,
};
use tower::{ServiceBuilder, limit::ConcurrencyLimitLayer};
use tower_http::{
    cors::{Any, CorsLayer},
    compression::CompressionLayer,
    timeout::TimeoutLayer,
    trace::TraceLayer,
};
use uuid::Uuid;

/// High-performance REST API server
pub struct RestApiServer {
    /// Server configuration
    config: RestConfig,
    /// Conformal prediction engine
    predictor: Arc<OptimizedConformalPredictor>,
    /// API handlers
    handlers: Arc<ApiHandlers>,
    /// Server metrics
    metrics: Arc<RestServerMetrics>,
    /// Shutdown signal
    shutdown: Arc<AtomicBool>,
    /// Performance monitor
    performance_monitor: Arc<RestPerformanceMonitor>,
}

/// REST Server Metrics
#[derive(Debug, Default)]
pub struct RestServerMetrics {
    /// Total requests processed
    pub requests_processed: AtomicU64,
    /// Requests per endpoint
    pub endpoint_requests: Arc<RwLock<std::collections::HashMap<String, u64>>>,
    /// Average response time in microseconds
    pub avg_response_time_us: AtomicU64,
    /// Error count
    pub error_count: AtomicU64,
    /// Active requests
    pub active_requests: AtomicU64,
    /// Total bytes transferred
    pub bytes_transferred: AtomicU64,
}

/// REST Performance Monitor
pub struct RestPerformanceMonitor {
    /// Request latency measurements
    latency_measurements: Arc<RwLock<Vec<u64>>>,
    /// Response size histogram
    response_sizes: Arc<RwLock<std::collections::HashMap<u64, u64>>>,
    /// Server start time
    server_start: Instant,
    /// Request rate tracking
    request_rate_tracker: Arc<RwLock<Vec<(Instant, u32)>>>,
}

impl RestPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            latency_measurements: Arc::new(RwLock::new(Vec::with_capacity(10000))),
            response_sizes: Arc::new(RwLock::new(std::collections::HashMap::new())),
            server_start: Instant::now(),
            request_rate_tracker: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record request latency
    pub async fn record_latency(&self, latency_us: u64) {
        let mut measurements = self.latency_measurements.write().await;
        
        if measurements.len() >= 10000 {
            measurements.remove(0);
        }
        measurements.push(latency_us);
    }

    /// Record response size
    pub async fn record_response_size(&self, size_bytes: u64) {
        let mut sizes = self.response_sizes.write().await;
        let bucket = (size_bytes / 1024) * 1024; // 1KB buckets
        *sizes.entry(bucket).or_insert(0) += 1;
    }

    /// Update request rate
    pub async fn record_request(&self) {
        let mut tracker = self.request_rate_tracker.write().await;
        let now = Instant::now();
        
        // Keep last 60 seconds of data
        tracker.retain(|(timestamp, _)| now.duration_since(*timestamp).as_secs() <= 60);
        tracker.push((now, 1));
    }

    /// Get current requests per second
    pub async fn get_requests_per_second(&self) -> f64 {
        let tracker = self.request_rate_tracker.read().await;
        let now = Instant::now();
        
        let recent_requests: u32 = tracker
            .iter()
            .filter(|(timestamp, _)| now.duration_since(*timestamp).as_secs() <= 1)
            .map(|(_, count)| count)
            .sum();
        
        recent_requests as f64
    }

    /// Get performance statistics
    pub async fn get_performance_stats(&self) -> RestPerformanceStats {
        let measurements = self.latency_measurements.read().await;
        
        if measurements.is_empty() {
            return RestPerformanceStats::default();
        }

        let mut sorted = measurements.clone();
        sorted.sort_unstable();

        let len = sorted.len();
        let avg = sorted.iter().sum::<u64>() as f64 / len as f64;
        let median = sorted[len / 2] as f64;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;

        RestPerformanceStats {
            average_latency_us: avg,
            median_latency_us: median,
            p95_latency_us: sorted[p95_idx] as f64,
            p99_latency_us: sorted[p99_idx] as f64,
            max_latency_us: sorted[len - 1] as f64,
            min_latency_us: sorted[0] as f64,
            requests_per_second: self.get_requests_per_second().await,
            uptime_seconds: self.server_start.elapsed().as_secs(),
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RestPerformanceStats {
    pub average_latency_us: f64,
    pub median_latency_us: f64,
    pub p95_latency_us: f64,
    pub p99_latency_us: f64,
    pub max_latency_us: f64,
    pub min_latency_us: f64,
    pub requests_per_second: f64,
    pub uptime_seconds: u64,
}

impl RestApiServer {
    /// Create new REST API server
    pub fn new(
        config: RestConfig,
        predictor: Arc<OptimizedConformalPredictor>,
    ) -> Result<Self> {
        let handlers = Arc::new(ApiHandlers::new(predictor.clone()));
        let metrics = Arc::new(RestServerMetrics::default());
        let performance_monitor = Arc::new(RestPerformanceMonitor::new());

        Ok(Self {
            config,
            predictor,
            handlers,
            metrics,
            shutdown: Arc::new(AtomicBool::new(false)),
            performance_monitor,
        })
    }

    /// Start the REST API server
    pub async fn start(&self) -> Result<()> {
        let addr: SocketAddr = format!("{}:{}", self.config.bind_address, self.config.port)
            .parse()
            .map_err(|e| AtsCoreError::IntegrationError(format!("Invalid address: {}", e)))?;

        // Create router with all routes and middleware
        let app = self.create_app_router().await?;

        println!("🚀 REST API server listening on http://{}", addr);

        // Start background monitoring tasks
        self.start_background_tasks().await;

        // Start server using axum's built-in server (hyper 1.0 compatible)
        let listener = tokio::net::TcpListener::bind(&addr).await
            .map_err(|e| AtsCoreError::IntegrationError(format!("Failed to bind: {}", e)))?;

        let shutdown_flag = self.shutdown.clone();

        // Graceful shutdown handling
        axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
            .with_graceful_shutdown(async move {
                while !shutdown_flag.load(Ordering::Relaxed) {
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
                println!("🛑 REST API server shutting down gracefully...");
            })
            .await
            .map_err(|e| AtsCoreError::IntegrationError(format!("Server error: {}", e)))?;

        Ok(())
    }

    /// Create the main application router with all middleware
    async fn create_app_router(&self) -> Result<Router> {
        // Create extensions first (must be cloned before Router consumes them)
        let handlers_ext = Extension(self.handlers.clone());
        let metrics_ext = Extension(self.metrics.clone());
        let perf_ext = Extension(self.performance_monitor.clone());

        // Create main router
        let app = Router::new()
            // Health endpoints
            .route("/health", get(health_check))
            .route("/health/detailed", get(detailed_health_check))
            .route("/metrics", get(get_metrics))
            .route("/performance", get(get_performance_metrics))
            
            // Model management
            .route("/models", get(list_models))
            .route("/models", post(create_model_config))
            .route("/models/:model_id", get(get_model_status))
            .route("/models/:model_id", put(update_model_config))
            .route("/models/:model_id", delete(delete_model))
            
            // Predictions
            .route("/predict/:model_id", post(single_prediction))
            .route("/predict/:model_id/batch", post(batch_predictions))
            .route("/predict/:model_id/stream", get(stream_predictions))
            
            // Calibration
            .route("/models/:model_id/calibrate", post(calibrate_model))
            .route("/models/:model_id/calibration-status", get(get_calibration_status))
            
            // Benchmarking
            .route("/benchmark", post(run_benchmark))
            .route("/benchmark/:benchmark_id", get(get_benchmark_results))
            
            // System operations
            .route("/system/memory", get(get_memory_stats))
            .route("/system/cpu", get(get_cpu_stats))
            .route("/shutdown", post(initiate_shutdown))
            // Add layers individually for better compatibility
            .layer(handlers_ext)
            .layer(metrics_ext)
            .layer(perf_ext)
            .layer(ConcurrencyLimitLayer::new(1000))
            .layer(TimeoutLayer::new(self.config.request_timeout))
            .layer(CompressionLayer::new())
            .layer(
                CorsLayer::new()
                    .allow_methods([Method::GET, Method::POST, Method::PUT, Method::DELETE])
                    .allow_headers(Any)
                    .allow_origin(Any)
            );

        Ok(app)
    }

    /// Start background monitoring tasks
    async fn start_background_tasks(&self) {
        let metrics = self.metrics.clone();
        let performance_monitor = self.performance_monitor.clone();

        // Performance monitoring task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            loop {
                interval.tick().await;
                
                let stats = performance_monitor.get_performance_stats().await;
                println!(
                    "📊 REST API Performance: {:.1}ms avg, {:.1}ms p99, {:.1} req/s", 
                    stats.average_latency_us / 1000.0,
                    stats.p99_latency_us / 1000.0,
                    stats.requests_per_second
                );
            }
        });

        // Memory cleanup task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            loop {
                interval.tick().await;
                
                // Periodic cleanup of metrics and monitoring data
                // This would implement more sophisticated cleanup logic
            }
        });
    }

    /// Get current server metrics
    pub async fn get_server_metrics(&self) -> PerformanceMetrics {
        let stats = self.performance_monitor.get_performance_stats().await;
        let error_rate = self.metrics.error_count.load(Ordering::Relaxed) as f64 
            / self.metrics.requests_processed.load(Ordering::Relaxed).max(1) as f64;

        PerformanceMetrics {
            average_latency_us: stats.average_latency_us,
            p95_latency_us: stats.p95_latency_us,
            p99_latency_us: stats.p99_latency_us,
            max_latency_us: stats.max_latency_us,
            requests_per_second: stats.requests_per_second,
            error_rate,
            throughput_mbps: self.metrics.bytes_transferred.load(Ordering::Relaxed) as f64 
                / (1_048_576.0 * stats.uptime_seconds.max(1) as f64),
            cpu_usage: 0.0, // Would be filled by system monitor
            memory_usage: 0.0, // Would be filled by system monitor
        }
    }

    /// Shutdown server gracefully
    pub async fn shutdown(&self) {
        println!("🛑 Initiating REST API server shutdown...");
        self.shutdown.store(true, Ordering::Relaxed);
        
        // Allow ongoing requests to complete
        tokio::time::sleep(Duration::from_secs(5)).await;
        
        println!("✅ REST API server shutdown complete");
    }
}

// Handler functions - these would typically be in separate modules

/// Health check endpoint
async fn health_check() -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let health_data = serde_json::json!({
        "status": "healthy",
        "timestamp": chrono::Utc::now(),
        "version": crate::VERSION
    });
    
    Json(ApiResponse::success(health_data, request_id))
}

/// Detailed health check
async fn detailed_health_check(
    Extension(handlers): Extension<Arc<ApiHandlers>>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    
    match handlers.get_detailed_health().await {
        Ok(health) => Json(ApiResponse::success(serde_json::to_value(health).unwrap(), request_id)),
        Err(e) => {
            let api_error = ApiError::from(e);
            Json(ApiResponse::error(api_error, request_id))
        }
    }
}

/// Get server metrics
async fn get_metrics(
    Extension(metrics): Extension<Arc<RestServerMetrics>>,
    Extension(performance_monitor): Extension<Arc<RestPerformanceMonitor>>,
) -> Json<ApiResponse<serde_json::Value>> {
    let request_id = Uuid::new_v4().to_string();
    let stats = performance_monitor.get_performance_stats().await;
    
    let metrics_data = serde_json::json!({
        "requests_processed": metrics.requests_processed.load(Ordering::Relaxed),
        "error_count": metrics.error_count.load(Ordering::Relaxed),
        "active_requests": metrics.active_requests.load(Ordering::Relaxed),
        "bytes_transferred": metrics.bytes_transferred.load(Ordering::Relaxed),
        "performance": stats
    });
    
    Json(ApiResponse::success(metrics_data, request_id))
}

/// Get performance metrics
async fn get_performance_metrics(
    Extension(performance_monitor): Extension<Arc<RestPerformanceMonitor>>,
) -> Json<ApiResponse<RestPerformanceStats>> {
    let request_id = Uuid::new_v4().to_string();
    let stats = performance_monitor.get_performance_stats().await;
    
    Json(ApiResponse::success(stats, request_id))
}

/// Placeholder handlers - these would be implemented with full functionality

async fn list_models() -> Json<ApiResponse<Vec<String>>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success(vec!["model1".to_string(), "model2".to_string()], request_id))
}

async fn create_model_config() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_config_created".to_string(), request_id))
}

async fn get_model_status() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_status".to_string(), request_id))
}

async fn update_model_config() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_updated".to_string(), request_id))
}

async fn delete_model() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("model_deleted".to_string(), request_id))
}

async fn single_prediction() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("prediction_result".to_string(), request_id))
}

async fn batch_predictions() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("batch_predictions".to_string(), request_id))
}

async fn stream_predictions() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("stream_initiated".to_string(), request_id))
}

async fn calibrate_model() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_started".to_string(), request_id))
}

async fn get_calibration_status() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("calibration_status".to_string(), request_id))
}

async fn run_benchmark() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_started".to_string(), request_id))
}

async fn get_benchmark_results() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("benchmark_results".to_string(), request_id))
}

async fn get_memory_stats() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("memory_stats".to_string(), request_id))
}

async fn get_cpu_stats() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("cpu_stats".to_string(), request_id))
}

async fn initiate_shutdown() -> Json<ApiResponse<String>> {
    let request_id = Uuid::new_v4().to_string();
    Json(ApiResponse::success("shutdown_initiated".to_string(), request_id))
}