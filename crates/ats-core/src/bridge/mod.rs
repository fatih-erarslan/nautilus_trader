//! Bridge Layer Implementation
//!
//! High-performance bridge between TypeScript frontend and Rust backend
//! with optimized serialization, memory management, and type safety.

pub mod serialization;
// Missing files - commented out until implemented
// pub mod memory_manager;
// pub mod type_bridge;
// pub mod ffi_bridge;

use crate::{
    api::{
        websocket::{WebSocketMessage, BinaryPredictionMessage},
        rest::{ApiResponse, BatchPredictionRequest, BatchPredictionResponse},
        ApiConfig,
    },
    types::{ConformalPredictionResult, PredictionInterval},
    AtsCoreError, Result,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{Arc, atomic::{AtomicU64, Ordering}},
    time::Instant,
};

/// Main bridge interface for frontend-backend communication
pub struct AtsBridge {
    /// Serialization handler
    serializer: Arc<serialization::SerializationHandler>,
    // Commented out until modules are implemented
    // /// Memory manager
    // memory_manager: Arc<memory_manager::MemoryManager>,
    // /// Type bridge for safe conversions
    // type_bridge: Arc<type_bridge::TypeBridge>,
    /// Performance metrics
    metrics: Arc<BridgeMetrics>,
}

/// Bridge performance metrics
#[derive(Debug, Default)]
pub struct BridgeMetrics {
    /// Serialization operations
    pub serializations: AtomicU64,
    /// Deserialization operations  
    pub deserializations: AtomicU64,
    /// Average serialization time (nanoseconds)
    pub avg_serialization_time_ns: AtomicU64,
    /// Average deserialization time (nanoseconds)
    pub avg_deserialization_time_ns: AtomicU64,
    /// Memory allocations
    pub memory_allocations: AtomicU64,
    /// Memory deallocations
    pub memory_deallocations: AtomicU64,
    /// Peak memory usage
    pub peak_memory_usage: AtomicU64,
    /// Type conversions
    pub type_conversions: AtomicU64,
    /// Conversion errors
    pub conversion_errors: AtomicU64,
}

/// Serialization format options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SerializationFormat {
    /// JSON (human readable, slower)
    Json,
    /// MessagePack (binary, faster)
    MessagePack,
    /// Custom binary format (fastest)
    Binary,
    /// Protocol Buffers
    Protobuf,
}

/// Bridge configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgeConfig {
    /// Default serialization format
    pub default_format: String,
    /// Enable compression
    pub compression_enabled: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Enable SIMD optimizations
    pub simd_enabled: bool,
    /// Maximum message size
    pub max_message_size: usize,
    /// Enable zero-copy optimizations
    pub zero_copy_enabled: bool,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            default_format: "binary".to_string(),
            compression_enabled: true,
            memory_pool_size: 1024 * 1024, // 1MB pool
            simd_enabled: true,
            max_message_size: 10 * 1024 * 1024, // 10MB
            zero_copy_enabled: true,
        }
    }
}

impl AtsBridge {
    /// Create new bridge instance
    pub fn new(config: BridgeConfig) -> Result<Self> {
        Ok(Self {
            serializer: Arc::new(serialization::SerializationHandler::new(&config)?),
            // Commented out until modules are implemented
            // memory_manager: Arc::new(memory_manager::MemoryManager::new(config.memory_pool_size)?),
            // type_bridge: Arc::new(type_bridge::TypeBridge::new()),
            metrics: Arc::new(BridgeMetrics::default()),
        })
    }

    /// Serialize data for transmission to frontend
    pub fn serialize_for_frontend<T>(&self, data: &T, format: SerializationFormat) -> Result<Vec<u8>>
    where
        T: Serialize,
    {
        let start_time = Instant::now();
        
        let result = match format {
            SerializationFormat::Json => {
                serde_json::to_vec(data)
                    .map_err(|e| AtsCoreError::ValidationFailed(format!("JSON serialization failed: {}", e)))
            }
            SerializationFormat::Binary => {
                self.serializer.serialize_binary(data)
            }
            SerializationFormat::MessagePack => {
                rmp_serde::to_vec(data)
                    .map_err(|e| AtsCoreError::ValidationFailed(format!("MessagePack serialization failed: {}", e)))
            }
            SerializationFormat::Protobuf => {
                // Would implement protobuf serialization
                Err(AtsCoreError::ComputationFailed("Protobuf not implemented".to_string()))
            }
        };

        // Update metrics
        let elapsed = start_time.elapsed().as_nanos() as u64;
        self.metrics.serializations.fetch_add(1, Ordering::Relaxed);
        self.update_avg_serialization_time(elapsed);

        result
    }

    /// Deserialize data from frontend
    pub fn deserialize_from_frontend<T>(&self, data: &[u8], format: SerializationFormat) -> Result<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        let start_time = Instant::now();
        
        let result = match format {
            SerializationFormat::Json => {
                serde_json::from_slice(data)
                    .map_err(|e| AtsCoreError::ValidationFailed(format!("JSON deserialization failed: {}", e)))
            }
            SerializationFormat::Binary => {
                self.serializer.deserialize_binary(data)
            }
            SerializationFormat::MessagePack => {
                rmp_serde::from_slice(data)
                    .map_err(|e| AtsCoreError::ValidationFailed(format!("MessagePack deserialization failed: {}", e)))
            }
            SerializationFormat::Protobuf => {
                // Would implement protobuf deserialization
                Err(AtsCoreError::ComputationFailed("Protobuf not implemented".to_string()))
            }
        };

        // Update metrics
        let elapsed = start_time.elapsed().as_nanos() as u64;
        self.metrics.deserializations.fetch_add(1, Ordering::Relaxed);
        self.update_avg_deserialization_time(elapsed);

        if result.is_err() {
            self.metrics.conversion_errors.fetch_add(1, Ordering::Relaxed);
        }

        result
    }

    /// Convert WebSocket message to binary format for ultra-low latency
    pub fn websocket_to_binary(&self, message: &WebSocketMessage) -> Result<Vec<u8>> {
        match message {
            WebSocketMessage::PredictionUpdate { model_id, prediction, timestamp: _, latency_us } => {
                let binary_msg = BinaryPredictionMessage::new(
                    model_id,
                    prediction,
                    latency_us * 1000, // Convert μs to ns
                );
                Ok(binary_msg.to_bytes())
            }
            _ => {
                // For non-prediction messages, use regular serialization
                self.serialize_for_frontend(message, SerializationFormat::Binary)
            }
        }
    }

    /// Convert binary data back to WebSocket message
    pub fn binary_to_websocket(&self, data: &[u8]) -> Result<WebSocketMessage> {
        if data.len() == std::mem::size_of::<BinaryPredictionMessage>() {
            let binary_msg = BinaryPredictionMessage::from_bytes(data)?;
            Ok(self.convert_binary_to_websocket_message(binary_msg))
        } else {
            self.deserialize_from_frontend(data, SerializationFormat::Binary)
        }
    }

    /// Convert binary prediction message to WebSocket message
    fn convert_binary_to_websocket_message(&self, binary_msg: BinaryPredictionMessage) -> WebSocketMessage {
        WebSocketMessage::PredictionUpdate {
            model_id: format!("model_{}", binary_msg.model_id_hash), // Would need proper ID mapping
            prediction: ConformalPredictionResult {
                intervals: vec![(binary_msg.lower_bound, binary_msg.upper_bound)],
                confidence: binary_msg.confidence,
                calibration_scores: Vec::new(),
                quantile_threshold: binary_msg.confidence,
                execution_time_ns: binary_msg.latency_ns,
            },
            timestamp: chrono::Utc::now(),
            latency_us: (binary_msg.latency_ns / 1000) as u64,
        }
    }

    /// Process batch prediction with optimized serialization
    pub fn process_batch_prediction(
        &self,
        request: &BatchPredictionRequest,
    ) -> Result<BatchPredictionResponse> {
        // Type bridge not available - using request directly
        // let converted_request = self.type_bridge.convert_batch_request(request)?;

        // Process predictions (would call actual prediction engine)
        let predictions = self.mock_batch_predictions(request)?;
        
        // Convert back to response format
        let response = BatchPredictionResponse {
            request_id: uuid::Uuid::new_v4().to_string(),
            model_id: request.model_id.clone(),
            predictions,
            metrics: None, // Would include actual processing metrics
            timestamp: chrono::Utc::now(),
        };

        self.metrics.type_conversions.fetch_add(2, Ordering::Relaxed);
        Ok(response)
    }

    /// Mock batch predictions for demonstration
    fn mock_batch_predictions(&self, request: &BatchPredictionRequest) -> Result<Vec<ConformalPredictionResult>> {
        let mut results = Vec::with_capacity(request.features.len());

        for features in &request.features {
            let point_prediction = features.iter().sum::<f64>() / features.len() as f64;

            let intervals: Vec<(f64, f64)> = request.confidence_levels.iter().map(|&confidence| {
                let width = (1.0 - confidence) * 2.0;
                (point_prediction - width, point_prediction + width)
            }).collect();

            results.push(ConformalPredictionResult {
                intervals,
                confidence: request.confidence_levels.first().copied().unwrap_or(0.95),
                calibration_scores: Vec::new(),
                quantile_threshold: 0.95,
                execution_time_ns: 0,
            });
        }

        Ok(results)
    }

    // Memory management methods - commented out until memory_manager module is implemented
    // /// Allocate memory from managed pool
    // pub fn allocate_memory(&self, size: usize) -> Result<*mut u8> {
    //     let ptr = self.memory_manager.allocate(size)?;
    //     self.metrics.memory_allocations.fetch_add(1, Ordering::Relaxed);
    //     Ok(ptr)
    // }
    //
    // /// Deallocate memory back to pool
    // pub fn deallocate_memory(&self, ptr: *mut u8, size: usize) {
    //     self.memory_manager.deallocate(ptr, size);
    //     self.metrics.memory_deallocations.fetch_add(1, Ordering::Relaxed);
    // }

    /// Get bridge performance metrics
    pub fn get_metrics(&self) -> BridgePerformanceReport {
        BridgePerformanceReport {
            serializations: self.metrics.serializations.load(Ordering::Relaxed),
            deserializations: self.metrics.deserializations.load(Ordering::Relaxed),
            avg_serialization_time_ns: self.metrics.avg_serialization_time_ns.load(Ordering::Relaxed),
            avg_deserialization_time_ns: self.metrics.avg_deserialization_time_ns.load(Ordering::Relaxed),
            memory_allocations: self.metrics.memory_allocations.load(Ordering::Relaxed),
            memory_deallocations: self.metrics.memory_deallocations.load(Ordering::Relaxed),
            peak_memory_usage: self.metrics.peak_memory_usage.load(Ordering::Relaxed),
            type_conversions: self.metrics.type_conversions.load(Ordering::Relaxed),
            conversion_errors: self.metrics.conversion_errors.load(Ordering::Relaxed),
            memory_efficiency: self.calculate_memory_efficiency(),
            serialization_efficiency: self.calculate_serialization_efficiency(),
        }
    }

    /// Update average serialization time
    fn update_avg_serialization_time(&self, elapsed_ns: u64) {
        let current_avg = self.metrics.avg_serialization_time_ns.load(Ordering::Relaxed);
        let count = self.metrics.serializations.load(Ordering::Relaxed);
        
        let new_avg = if count == 1 {
            elapsed_ns
        } else {
            (current_avg * (count - 1) + elapsed_ns) / count
        };
        
        self.metrics.avg_serialization_time_ns.store(new_avg, Ordering::Relaxed);
    }

    /// Update average deserialization time
    fn update_avg_deserialization_time(&self, elapsed_ns: u64) {
        let current_avg = self.metrics.avg_deserialization_time_ns.load(Ordering::Relaxed);
        let count = self.metrics.deserializations.load(Ordering::Relaxed);
        
        let new_avg = if count == 1 {
            elapsed_ns
        } else {
            (current_avg * (count - 1) + elapsed_ns) / count
        };
        
        self.metrics.avg_deserialization_time_ns.store(new_avg, Ordering::Relaxed);
    }

    /// Calculate memory efficiency
    fn calculate_memory_efficiency(&self) -> f64 {
        let allocations = self.metrics.memory_allocations.load(Ordering::Relaxed);
        let deallocations = self.metrics.memory_deallocations.load(Ordering::Relaxed);
        
        if allocations == 0 {
            1.0
        } else {
            deallocations as f64 / allocations as f64
        }
    }

    /// Calculate serialization efficiency
    fn calculate_serialization_efficiency(&self) -> f64 {
        let total_ops = self.metrics.serializations.load(Ordering::Relaxed) 
                      + self.metrics.deserializations.load(Ordering::Relaxed);
        let errors = self.metrics.conversion_errors.load(Ordering::Relaxed);
        
        if total_ops == 0 {
            1.0
        } else {
            (total_ops - errors) as f64 / total_ops as f64
        }
    }

    /// Benchmark different serialization formats
    pub fn benchmark_serialization(&self, iterations: usize) -> SerializationBenchmark {
        let test_data = BatchPredictionRequest {
            model_id: "test_model".to_string(),
            features: vec![vec![1.0, 2.0, 3.0, 4.0, 5.0]; 100],
            confidence_levels: vec![0.95, 0.99],
            options: crate::api::rest::PredictionOptions {
                use_simd: true,
                parallel_processing: true,
                timeout_ms: Some(5000),
                include_metrics: false,
            },
        };

        let mut results = HashMap::new();

        // Benchmark each format
        for format in [SerializationFormat::Json, SerializationFormat::Binary, SerializationFormat::MessagePack] {
            let start = Instant::now();
            let mut total_size = 0;
            
            for _ in 0..iterations {
                if let Ok(serialized) = self.serialize_for_frontend(&test_data, format) {
                    total_size += serialized.len();
                    let _ = self.deserialize_from_frontend::<BatchPredictionRequest>(&serialized, format);
                }
            }
            
            let elapsed = start.elapsed();
            
            results.insert(format, FormatBenchmarkResult {
                total_time: elapsed,
                ops_per_second: (iterations * 2) as f64 / elapsed.as_secs_f64(), // serialize + deserialize
                average_size: total_size / iterations,
                throughput_mbps: (total_size as f64 / elapsed.as_secs_f64()) / 1_048_576.0,
            });
        }

        // Calculate winner BEFORE moving results into struct
        let winner = self.determine_benchmark_winner(&results);

        SerializationBenchmark {
            iterations,
            results,
            winner,
        }
    }

    /// Determine the best performing serialization format
    fn determine_benchmark_winner(&self, results: &HashMap<SerializationFormat, FormatBenchmarkResult>) -> SerializationFormat {
        results.iter()
            .max_by(|(_, a), (_, b)| a.ops_per_second.partial_cmp(&b.ops_per_second).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(format, _)| *format)
            .unwrap_or(SerializationFormat::Binary)
    }
}

/// Bridge performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BridgePerformanceReport {
    pub serializations: u64,
    pub deserializations: u64,
    pub avg_serialization_time_ns: u64,
    pub avg_deserialization_time_ns: u64,
    pub memory_allocations: u64,
    pub memory_deallocations: u64,
    pub peak_memory_usage: u64,
    pub type_conversions: u64,
    pub conversion_errors: u64,
    pub memory_efficiency: f64,
    pub serialization_efficiency: f64,
}

/// Serialization benchmark results
#[derive(Debug, Clone)]
pub struct SerializationBenchmark {
    pub iterations: usize,
    pub results: HashMap<SerializationFormat, FormatBenchmarkResult>,
    pub winner: SerializationFormat,
}

#[derive(Debug, Clone)]
pub struct FormatBenchmarkResult {
    pub total_time: std::time::Duration,
    pub ops_per_second: f64,
    pub average_size: usize,
    pub throughput_mbps: f64,
}

/// FFI exports for C/JavaScript interoperability
#[cfg(feature = "ffi")]
pub mod ffi_exports {
    use super::*;
    use std::ffi::{CStr, CString};
    use std::os::raw::c_char;

    #[no_mangle]
    pub extern "C" fn ats_bridge_create() -> *mut AtsBridge {
        match AtsBridge::new(BridgeConfig::default()) {
            Ok(bridge) => Box::into_raw(Box::new(bridge)),
            Err(_) => std::ptr::null_mut(),
        }
    }

    #[no_mangle]
    pub extern "C" fn ats_bridge_destroy(bridge: *mut AtsBridge) {
        if !bridge.is_null() {
            unsafe { Box::from_raw(bridge) };
        }
    }

    #[no_mangle]
    pub extern "C" fn ats_bridge_serialize_json(
        bridge: *mut AtsBridge,
        data: *const c_char,
        output: *mut *mut u8,
        output_len: *mut usize,
    ) -> i32 {
        if bridge.is_null() || data.is_null() || output.is_null() || output_len.is_null() {
            return -1;
        }

        let bridge = unsafe { &*bridge };
        let input_str = unsafe { CStr::from_ptr(data).to_str().unwrap_or("") };
        
        // Parse JSON and serialize to binary
        match serde_json::from_str::<serde_json::Value>(input_str) {
            Ok(value) => {
                match bridge.serialize_for_frontend(&value, SerializationFormat::Binary) {
                    Ok(serialized) => {
                        let boxed_slice = serialized.into_boxed_slice();
                        let len = boxed_slice.len();
                        let ptr = Box::into_raw(boxed_slice) as *mut u8;
                        
                        unsafe {
                            *output = ptr;
                            *output_len = len;
                        }
                        0
                    }
                    Err(_) => -2,
                }
            }
            Err(_) => -3,
        }
    }

    #[no_mangle]
    pub extern "C" fn ats_bridge_free_memory(ptr: *mut u8, len: usize) {
        if !ptr.is_null() {
            unsafe {
                let slice = std::slice::from_raw_parts_mut(ptr, len);
                let _ = Box::from_raw(slice as *mut [u8]);
            }
        }
    }
}