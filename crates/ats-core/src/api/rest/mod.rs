//! REST API Module
//!
//! High-performance REST API for model configuration, health monitoring,
//! and non-real-time operations with comprehensive error handling.

pub mod server;
pub mod handlers;
pub mod middleware;
pub mod routes;

use crate::{
    api::{ApiError, PerformanceMetrics, HealthStatus},
    types::{ConformalPredictionResult, PredictionInterval, Confidence},
    conformal_optimized::OptimizedConformalPredictor,
    AtsCoreError, Result,
};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// REST API Request/Response Types

/// Model Configuration Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigRequest {
    /// Model identifier
    pub model_id: String,
    /// Model type (lstm, transformer, etc.)
    pub model_type: String,
    /// Confidence levels for prediction intervals
    pub confidence_levels: Vec<f64>,
    /// Temperature scaling configuration
    pub temperature_config: Option<TemperatureConfig>,
    /// Model-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Temperature Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemperatureConfig {
    /// Initial temperature value
    pub initial_temperature: f64,
    /// Learning rate for temperature updates
    pub learning_rate: f64,
    /// Maximum temperature value
    pub max_temperature: f64,
    /// Minimum temperature value
    pub min_temperature: f64,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Model Configuration Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfigResponse {
    /// Configuration ID
    pub config_id: String,
    /// Model ID
    pub model_id: String,
    /// Configuration status
    pub status: ConfigurationStatus,
    /// Configuration details
    pub config: ModelConfigRequest,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Last updated timestamp
    pub updated_at: chrono::DateTime<chrono::Utc>,
}

/// Configuration Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConfigurationStatus {
    Pending,
    Active,
    Inactive,
    Error { message: String },
}

/// Batch Prediction Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionRequest {
    /// Model ID
    pub model_id: String,
    /// Input features (batch)
    pub features: Vec<Vec<f64>>,
    /// Confidence levels
    pub confidence_levels: Vec<f64>,
    /// Processing options
    pub options: PredictionOptions,
}

/// Prediction Options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionOptions {
    /// Use SIMD optimizations
    pub use_simd: bool,
    /// Parallel processing
    pub parallel_processing: bool,
    /// Maximum processing time
    pub timeout_ms: Option<u64>,
    /// Return detailed metrics
    pub include_metrics: bool,
}

impl Default for PredictionOptions {
    fn default() -> Self {
        Self {
            use_simd: true,
            parallel_processing: true,
            timeout_ms: Some(5000),
            include_metrics: false,
        }
    }
}

/// Batch Prediction Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchPredictionResponse {
    /// Request ID for tracking
    pub request_id: String,
    /// Model ID
    pub model_id: String,
    /// Predictions
    pub predictions: Vec<ConformalPredictionResult>,
    /// Processing metrics
    pub metrics: Option<ProcessingMetrics>,
    /// Processing timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Processing Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetrics {
    /// Total processing time in microseconds
    pub total_processing_time_us: u64,
    /// Average per-sample processing time
    pub avg_sample_time_us: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// SIMD utilization
    pub simd_utilization: Option<f64>,
}

/// Model Status Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusRequest {
    /// Model ID
    pub model_id: String,
    /// Include detailed metrics
    pub include_metrics: bool,
}

/// Model Status Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatusResponse {
    /// Model ID
    pub model_id: String,
    /// Model status
    pub status: ModelStatus,
    /// Configuration
    pub configuration: Option<ModelConfigRequest>,
    /// Performance metrics
    pub metrics: Option<ModelMetrics>,
    /// Last activity timestamp
    pub last_activity: chrono::DateTime<chrono::Utc>,
    /// Model uptime
    pub uptime_seconds: u64,
}

/// Model Status Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelStatus {
    Initializing,
    Ready,
    Processing,
    Error { message: String },
    Offline,
}

/// Model-specific Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Predictions per second
    pub predictions_per_second: f64,
    /// Average prediction latency
    pub avg_latency_us: f64,
    /// Error rate
    pub error_rate: f64,
    /// Memory usage
    pub memory_usage_mb: f64,
    /// Accuracy metrics (if available)
    pub accuracy_metrics: Option<AccuracyMetrics>,
}

/// Accuracy Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    /// Mean absolute error
    pub mae: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Coverage percentage for prediction intervals
    pub coverage_rate: f64,
    /// Interval width (average)
    pub avg_interval_width: f64,
}

/// System Health Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckRequest {
    /// Include detailed diagnostics
    pub detailed: bool,
    /// Check specific components
    pub components: Option<Vec<String>>,
}

/// Calibration Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationRequest {
    /// Model ID
    pub model_id: String,
    /// Calibration data (features, true_values pairs)
    pub calibration_data: Vec<CalibrationSample>,
    /// Confidence levels to calibrate
    pub confidence_levels: Vec<f64>,
    /// Calibration method
    pub method: CalibrationMethod,
}

/// Calibration Sample
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationSample {
    /// Input features
    pub features: Vec<f64>,
    /// True target value
    pub true_value: f64,
}

/// Calibration Method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationMethod {
    /// Split conformal prediction
    SplitConformal,
    /// Cross conformal prediction
    CrossConformal { folds: u32 },
    /// Jackknife+ method
    JackknifeePlus,
    /// Adaptive conformal prediction
    Adaptive { alpha: f64 },
}

/// Calibration Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResponse {
    /// Calibration ID
    pub calibration_id: String,
    /// Model ID
    pub model_id: String,
    /// Calibration status
    pub status: CalibrationStatus,
    /// Calibration results
    pub results: Option<CalibrationResults>,
    /// Processing timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Calibration Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CalibrationStatus {
    Running,
    Completed,
    Failed { error: String },
}

/// Calibration Results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CalibrationResults {
    /// Confidence levels
    pub confidence_levels: Vec<f64>,
    /// Achieved coverage rates
    pub coverage_rates: Vec<f64>,
    /// Average interval widths
    pub interval_widths: Vec<f64>,
    /// Calibration scores
    pub calibration_scores: Vec<f64>,
    /// Processing time
    pub processing_time_ms: u64,
}

/// Performance Benchmark Request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkRequest {
    /// Models to benchmark
    pub model_ids: Vec<String>,
    /// Number of samples for benchmarking
    pub sample_count: u32,
    /// Confidence levels to test
    pub confidence_levels: Vec<f64>,
    /// Benchmark options
    pub options: BenchmarkOptions,
}

/// Benchmark Options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkOptions {
    /// Include memory profiling
    pub memory_profiling: bool,
    /// Include latency distribution
    pub latency_distribution: bool,
    /// Warm-up iterations
    pub warmup_iterations: u32,
    /// Test iterations
    pub test_iterations: u32,
}

/// Performance Benchmark Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    /// Benchmark ID
    pub benchmark_id: String,
    /// Model benchmarks
    pub model_benchmarks: Vec<ModelBenchmark>,
    /// System metrics during benchmark
    pub system_metrics: SystemMetrics,
    /// Benchmark timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Individual Model Benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelBenchmark {
    /// Model ID
    pub model_id: String,
    /// Latency statistics
    pub latency_stats: LatencyStats,
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    /// Memory usage
    pub memory_usage: MemoryUsage,
    /// Error statistics
    pub error_stats: ErrorStats,
}

/// Latency Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Average latency in microseconds
    pub avg_us: f64,
    /// Median latency
    pub median_us: f64,
    /// 95th percentile
    pub p95_us: f64,
    /// 99th percentile
    pub p99_us: f64,
    /// Maximum latency
    pub max_us: f64,
    /// Standard deviation
    pub std_dev_us: f64,
}

/// Throughput Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    /// Predictions per second
    pub predictions_per_second: f64,
    /// Data throughput in MB/s
    pub data_throughput_mbps: f64,
    /// Peak throughput
    pub peak_throughput: f64,
}

/// Memory Usage Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsage {
    /// Peak memory usage in MB
    pub peak_mb: f64,
    /// Average memory usage
    pub avg_mb: f64,
    /// Memory allocations count
    pub allocations: u64,
    /// Memory deallocations count
    pub deallocations: u64,
}

/// Error Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorStats {
    /// Total errors
    pub total_errors: u64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Error types breakdown
    pub error_breakdown: HashMap<String, u64>,
}

/// System Metrics during Benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage percentage
    pub memory_usage: f64,
    /// Disk I/O statistics
    pub disk_io: Option<DiskIOStats>,
    /// Network I/O statistics
    pub network_io: Option<NetworkIOStats>,
}

/// Disk I/O Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiskIOStats {
    /// Bytes read
    pub bytes_read: u64,
    /// Bytes written
    pub bytes_written: u64,
    /// Read operations
    pub read_ops: u64,
    /// Write operations
    pub write_ops: u64,
}

/// Network I/O Statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkIOStats {
    /// Bytes received
    pub bytes_received: u64,
    /// Bytes sent
    pub bytes_sent: u64,
    /// Packets received
    pub packets_received: u64,
    /// Packets sent
    pub packets_sent: u64,
}

/// API Response wrapper for consistent error handling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    /// Success flag
    pub success: bool,
    /// Response data
    pub data: Option<T>,
    /// Error information
    pub error: Option<ApiError>,
    /// Request ID for tracing
    pub request_id: String,
    /// Response timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl<T> ApiResponse<T> {
    pub fn success(data: T, request_id: String) -> Self {
        Self {
            success: true,
            data: Some(data),
            error: None,
            request_id,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn error(error: ApiError, request_id: String) -> Self {
        Self {
            success: false,
            data: None,
            error: Some(error),
            request_id,
            timestamp: chrono::Utc::now(),
        }
    }
}

/// Pagination parameters for list endpoints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginationParams {
    /// Page number (0-based)
    pub page: u32,
    /// Items per page
    pub limit: u32,
    /// Sort field
    pub sort_by: Option<String>,
    /// Sort order
    pub sort_order: Option<SortOrder>,
}

/// Sort Order
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SortOrder {
    Asc,
    Desc,
}

impl Default for PaginationParams {
    fn default() -> Self {
        Self {
            page: 0,
            limit: 50,
            sort_by: None,
            sort_order: Some(SortOrder::Desc),
        }
    }
}

/// Paginated Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaginatedResponse<T> {
    /// Items in current page
    pub items: Vec<T>,
    /// Current page number
    pub page: u32,
    /// Items per page
    pub limit: u32,
    /// Total item count
    pub total_count: u64,
    /// Total pages
    pub total_pages: u32,
    /// Has next page
    pub has_next: bool,
    /// Has previous page
    pub has_previous: bool,
}