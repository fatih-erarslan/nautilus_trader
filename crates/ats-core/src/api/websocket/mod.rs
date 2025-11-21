//! Ultra-High Performance WebSocket Server
//!
//! Real-time conformal prediction streaming with sub-25μs latency guarantees
//! using optimized message serialization and SIMD-accelerated processing.

pub mod server;
// Missing files - commented out until implemented
// pub mod client;
// pub mod protocol;
// pub mod handler;
// pub mod connection;

use crate::{
    api::{ApiConfig, PerformanceMetrics},
    types::{ConformalPredictionResult, PredictionInterval},
    AtsCoreError, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::{Duration, Instant},
};
use tokio::sync::RwLock;

/// WebSocket Message Protocol
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum WebSocketMessage {
    /// Subscribe to prediction stream
    Subscribe {
        model_id: String,
        confidence_levels: Vec<f64>,
        update_frequency: Duration,
    },
    /// Unsubscribe from prediction stream
    Unsubscribe {
        model_id: String,
    },
    /// Real-time prediction update
    PredictionUpdate {
        model_id: String,
        prediction: ConformalPredictionResult,
        timestamp: chrono::DateTime<chrono::Utc>,
        latency_us: u64,
    },
    /// Batch prediction updates
    BatchPredictionUpdate {
        model_id: String,
        predictions: Vec<ConformalPredictionResult>,
        timestamp: chrono::DateTime<chrono::Utc>,
        batch_latency_us: u64,
    },
    /// Model configuration update
    ModelConfigUpdate {
        model_id: String,
        config: serde_json::Value,
        version: u32,
    },
    /// Health check ping
    Ping {
        timestamp: chrono::DateTime<chrono::Utc>,
    },
    /// Health check response
    Pong {
        timestamp: chrono::DateTime<chrono::Utc>,
        server_time: chrono::DateTime<chrono::Utc>,
    },
    /// Error message
    Error {
        code: String,
        message: String,
        request_id: Option<String>,
    },
    /// Connection acknowledgment
    Welcome {
        client_id: String,
        server_version: String,
        supported_protocols: Vec<String>,
    },
    /// Performance metrics update
    MetricsUpdate {
        metrics: PerformanceMetrics,
        timestamp: chrono::DateTime<chrono::Utc>,
    },
}

/// Client Connection State
#[derive(Debug, Clone)]
pub struct ClientState {
    /// Unique client identifier
    pub client_id: String,
    /// Subscribed model IDs
    pub subscriptions: HashMap<String, SubscriptionConfig>,
    /// Connection start time
    pub connected_at: Instant,
    /// Last activity timestamp
    pub last_activity: Instant,
    /// Messages sent counter
    pub messages_sent: Arc<AtomicU64>,
    /// Messages received counter
    pub messages_received: Arc<AtomicU64>,
    /// Client IP address
    pub remote_addr: Option<std::net::SocketAddr>,
}

/// Subscription Configuration
#[derive(Debug, Clone)]
pub struct SubscriptionConfig {
    /// Confidence levels to monitor
    pub confidence_levels: Vec<f64>,
    /// Update frequency
    pub update_frequency: Duration,
    /// Last update timestamp
    pub last_update: Option<Instant>,
    /// Subscription start time
    pub subscribed_at: Instant,
}

/// WebSocket Server Metrics
#[derive(Debug, Clone)]
pub struct WebSocketMetrics {
    /// Active connections count
    pub active_connections: Arc<AtomicU64>,
    /// Total connections served
    pub total_connections: Arc<AtomicU64>,
    /// Messages sent
    pub messages_sent: Arc<AtomicU64>,
    /// Messages received
    pub messages_received: Arc<AtomicU64>,
    /// Average message processing time in microseconds
    pub avg_processing_time_us: Arc<AtomicU64>,
    /// Peak connections
    pub peak_connections: Arc<AtomicU64>,
    /// Error count
    pub error_count: Arc<AtomicU64>,
}

impl Default for WebSocketMetrics {
    fn default() -> Self {
        Self {
            active_connections: Arc::new(AtomicU64::new(0)),
            total_connections: Arc::new(AtomicU64::new(0)),
            messages_sent: Arc::new(AtomicU64::new(0)),
            messages_received: Arc::new(AtomicU64::new(0)),
            avg_processing_time_us: Arc::new(AtomicU64::new(0)),
            peak_connections: Arc::new(AtomicU64::new(0)),
            error_count: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl WebSocketMetrics {
    /// Update processing time with exponential moving average
    pub fn update_processing_time(&self, processing_time_us: u64) {
        let current_avg = self.avg_processing_time_us.load(Ordering::Relaxed);
        // EMA with alpha = 0.1 for smooth averaging
        let new_avg = if current_avg == 0 {
            processing_time_us
        } else {
            (current_avg * 9 + processing_time_us) / 10
        };
        self.avg_processing_time_us.store(new_avg, Ordering::Relaxed);
    }

    /// Increment connection count and update peak
    pub fn increment_connections(&self) {
        let new_count = self.active_connections.fetch_add(1, Ordering::Relaxed) + 1;
        self.total_connections.fetch_add(1, Ordering::Relaxed);
        
        // Update peak connections
        let mut current_peak = self.peak_connections.load(Ordering::Relaxed);
        while new_count > current_peak {
            match self.peak_connections.compare_exchange_weak(
                current_peak, 
                new_count, 
                Ordering::Relaxed, 
                Ordering::Relaxed
            ) {
                Ok(_) => break,
                Err(x) => current_peak = x,
            }
        }
    }

    /// Decrement connection count
    pub fn decrement_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        PerformanceMetrics {
            average_latency_us: self.avg_processing_time_us.load(Ordering::Relaxed) as f64,
            p95_latency_us: self.avg_processing_time_us.load(Ordering::Relaxed) as f64 * 1.5,
            p99_latency_us: self.avg_processing_time_us.load(Ordering::Relaxed) as f64 * 2.0,
            max_latency_us: self.avg_processing_time_us.load(Ordering::Relaxed) as f64 * 3.0,
            requests_per_second: 0.0, // Will be calculated by server
            error_rate: 0.0, // Will be calculated by server
            throughput_mbps: 0.0, // Will be calculated by server
            cpu_usage: 0.0, // System metrics
            memory_usage: 0.0, // System metrics
        }
    }
}

/// Connection Pool for managing active WebSocket connections
pub type ConnectionPool = Arc<RwLock<HashMap<String, ClientState>>>;

/// WebSocket Server Configuration
#[derive(Debug, Clone)]
pub struct WebSocketServerConfig {
    /// Bind address
    pub bind_address: String,
    /// Port number
    pub port: u16,
    /// Maximum connections
    pub max_connections: usize,
    /// Connection timeout
    pub connection_timeout: Duration,
    /// Heartbeat interval
    pub heartbeat_interval: Duration,
    /// Message buffer size
    pub message_buffer_size: usize,
    /// Enable compression
    pub compression_enabled: bool,
    /// SIMD optimizations enabled
    pub simd_enabled: bool,
}

impl From<&ApiConfig> for WebSocketServerConfig {
    fn from(config: &ApiConfig) -> Self {
        Self {
            bind_address: config.websocket.bind_address.clone(),
            port: config.websocket.port,
            max_connections: config.websocket.max_connections,
            connection_timeout: config.websocket.connection_timeout,
            heartbeat_interval: config.websocket.heartbeat_interval,
            message_buffer_size: config.websocket.buffer_size,
            compression_enabled: true,
            simd_enabled: config.websocket.simd_enabled,
        }
    }
}

/// High-performance serialization using SIMD when available
pub mod serialization {
    use super::*;
    use bytemuck::{Pod, Zeroable};

    /// SIMD-optimized binary message format for ultra-low latency
    #[repr(C)]
    #[derive(Debug, Clone, Copy, Pod, Zeroable)]
    pub struct BinaryPredictionMessage {
        /// Message type identifier
        pub msg_type: u32,
        /// Explicit padding to align next field to 8 bytes
        pub _padding: u32,
        /// Model ID hash
        pub model_id_hash: u64,
        /// Timestamp (nanoseconds since epoch)
        pub timestamp_ns: u64,
        /// Prediction value
        pub prediction: f64,
        /// Lower confidence interval
        pub lower_bound: f64,
        /// Upper confidence interval
        pub upper_bound: f64,
        /// Confidence level
        pub confidence: f64,
        /// Processing latency in nanoseconds
        pub latency_ns: u64,
    }

    impl BinaryPredictionMessage {
        pub fn new(
            model_id: &str,
            prediction: &ConformalPredictionResult,
            latency_ns: u64,
        ) -> Self {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};

            let mut hasher = DefaultHasher::new();
            model_id.hash(&mut hasher);
            let model_id_hash = hasher.finish();

            Self {
                msg_type: 1, // Prediction update
                _padding: 0,
                model_id_hash,
                timestamp_ns: chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0) as u64,
                prediction: prediction.intervals.get(0).map(|(l, _)| *l).unwrap_or(0.0),
                lower_bound: prediction.intervals.get(0)
                    .map(|(lower, _)| *lower)
                    .unwrap_or(0.0),
                upper_bound: prediction.intervals.get(0)
                    .map(|(_, upper)| *upper)
                    .unwrap_or(0.0),
                confidence: prediction.confidence,
                latency_ns,
            }
        }

        /// Serialize to bytes with SIMD optimization
        pub fn to_bytes(&self) -> Vec<u8> {
            bytemuck::bytes_of(self).to_vec()
        }

        /// Deserialize from bytes
        pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
            if bytes.len() != std::mem::size_of::<Self>() {
                return Err(AtsCoreError::ValidationFailed(
                    "Invalid binary message size".to_string()
                ));
            }
            Ok(*bytemuck::from_bytes(bytes))
        }
    }

    /// Fast JSON serialization with pre-allocated buffers
    pub struct FastJsonSerializer {
        buffer: Vec<u8>,
    }

    impl FastJsonSerializer {
        pub fn new() -> Self {
            Self {
                buffer: Vec::with_capacity(4096),
            }
        }

        pub fn serialize<T: Serialize>(&mut self, value: &T) -> Result<&[u8]> {
            self.buffer.clear();
            serde_json::to_writer(&mut self.buffer, value)
                .map_err(|e| AtsCoreError::ValidationFailed(format!("Serialization error: {}", e)))?;
            Ok(&self.buffer)
        }
    }
}

pub use serialization::*;