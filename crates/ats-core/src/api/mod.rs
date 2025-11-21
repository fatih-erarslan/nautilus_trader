//! API Integration Layer
//! 
//! High-performance WebSocket and REST API layer for real-time conformal prediction streaming
//! with sub-25μs end-to-end latency guarantees.

pub mod websocket;
pub mod rest;
// Missing files - commented out until implemented
// pub mod client;
pub mod security;
// pub mod monitoring;

use crate::{AtsCoreError, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// API Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    /// WebSocket server configuration
    pub websocket: WebSocketConfig,
    /// REST API configuration
    pub rest: RestConfig,
    /// Security configuration
    pub security: SecurityConfig,
    /// Monitoring configuration
    pub monitoring: MonitoringConfig,
}

/// WebSocket Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Maximum connections
    pub max_connections: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Buffer size for high-frequency data
    pub buffer_size: usize,
    /// Enable SIMD optimizations
    pub simd_enabled: bool,
}

/// REST API Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestConfig {
    /// Server bind address
    pub bind_address: String,
    /// Server port
    pub port: u16,
    /// Request timeout
    pub request_timeout: Duration,
    /// Maximum request body size
    pub max_body_size: usize,
    /// Enable compression
    pub compression_enabled: bool,
}

/// Security Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable TLS/SSL
    pub tls_enabled: bool,
    /// TLS certificate path
    pub cert_path: Option<String>,
    /// TLS private key path
    pub key_path: Option<String>,
    /// JWT secret key
    pub jwt_secret: String,
    /// Token expiration time
    pub token_expiry: Duration,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
}

/// Rate Limiting Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Requests per second per client
    pub requests_per_second: u32,
    /// Burst allowance
    pub burst_size: u32,
    /// Window duration
    pub window_duration: Duration,
}

/// Monitoring Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Enable metrics collection
    pub metrics_enabled: bool,
    /// Metrics collection interval
    pub metrics_interval: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Performance monitoring enabled
    pub performance_monitoring: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            websocket: WebSocketConfig {
                bind_address: "0.0.0.0".to_string(),
                port: 8080,
                max_connections: 1000,
                connection_timeout: Duration::from_secs(30),
                heartbeat_interval: Duration::from_secs(10),
                buffer_size: 8192,
                simd_enabled: true,
            },
            rest: RestConfig {
                bind_address: "0.0.0.0".to_string(),
                port: 8081,
                request_timeout: Duration::from_secs(30),
                max_body_size: 10 * 1024 * 1024, // 10MB
                compression_enabled: true,
            },
            security: SecurityConfig {
                tls_enabled: false,
                cert_path: None,
                key_path: None,
                jwt_secret: "your-secret-key-here".to_string(),
                token_expiry: Duration::from_hours(24),
                rate_limit: RateLimitConfig {
                    requests_per_second: 100,
                    burst_size: 200,
                    window_duration: Duration::from_secs(60),
                },
            },
            monitoring: MonitoringConfig {
                metrics_enabled: true,
                metrics_interval: Duration::from_secs(1),
                health_check_interval: Duration::from_secs(5),
                performance_monitoring: true,
            },
        }
    }
}

/// API Health Status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health status
    pub status: ServiceStatus,
    /// WebSocket server status
    pub websocket_status: ServiceStatus,
    /// REST API status
    pub rest_status: ServiceStatus,
    /// Conformal prediction engine status
    pub prediction_engine_status: ServiceStatus,
    /// Memory usage
    pub memory_usage: MemoryMetrics,
    /// Connection metrics
    pub connection_metrics: ConnectionMetrics,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Service Status Enum
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ServiceStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Offline,
}

/// Memory Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total allocated memory in bytes
    pub total_allocated: u64,
    /// Currently used memory in bytes
    pub used: u64,
    /// Available memory in bytes
    pub available: u64,
    /// Peak memory usage in bytes
    pub peak_usage: u64,
}

/// Connection Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionMetrics {
    /// Active WebSocket connections
    pub active_websocket_connections: u32,
    /// Total connections served
    pub total_connections_served: u64,
    /// Average connection duration
    pub average_connection_duration: Duration,
    /// Connections per second
    pub connections_per_second: f64,
}

/// API Error Response
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiError {
    /// Error code
    pub code: String,
    /// Error message
    pub message: String,
    /// Error details
    pub details: Option<serde_json::Value>,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Request ID for tracing
    pub request_id: Option<String>,
}

impl From<AtsCoreError> for ApiError {
    fn from(error: AtsCoreError) -> Self {
        Self {
            code: match &error {
                AtsCoreError::ValidationFailed(_) => "VALIDATION_ERROR".to_string(),
                AtsCoreError::ComputationFailed(_) => "COMPUTATION_ERROR".to_string(),
                AtsCoreError::Configuration { .. } => "CONFIG_ERROR".to_string(),
                AtsCoreError::IntegrationError(_) => "INTEGRATION_ERROR".to_string(),
                AtsCoreError::Integration { .. } => "INTEGRATION_ERROR".to_string(),
                _ => "INTERNAL_ERROR".to_string(),
            },
            message: error.to_string(),
            details: None,
            timestamp: chrono::Utc::now(),
            request_id: None,
        }
    }
}

/// Performance Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average latency in microseconds
    pub average_latency_us: f64,
    /// 95th percentile latency in microseconds
    pub p95_latency_us: f64,
    /// 99th percentile latency in microseconds
    pub p99_latency_us: f64,
    /// Maximum latency in microseconds
    pub max_latency_us: f64,
    /// Requests per second
    pub requests_per_second: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Throughput in MB/s
    pub throughput_mbps: f64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage percentage
    pub memory_usage: f64,
}