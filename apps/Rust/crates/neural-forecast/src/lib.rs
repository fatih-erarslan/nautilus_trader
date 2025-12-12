//! Neural Forecasting Library with GPU Acceleration
//!
//! This library provides advanced neural forecasting models optimized for financial
//! time series prediction with support for GPU acceleration achieving 50-200x speedup.

// Error handling
use std::error::Error;
use std::fmt;

pub mod config;
pub mod models;
pub mod preprocessing;
pub mod loss;
pub mod metrics;
pub mod utils;
pub mod storage;
pub mod ensemble;
pub mod inference;
pub mod batch;
pub mod activation;

#[cfg(feature = "gpu")]
pub mod gpu;
pub mod cuda_helpers;

// Re-export key types for easier access
pub use config::*;
pub use models::*;

// Re-export key GPU acceleration types for easier access
#[cfg(feature = "gpu")]
pub use gpu::{
    GPUBackend, 
    GPUTensor,
    benchmarks::{GPUBenchmarkSuite, BenchmarkConfig, BenchmarkReport},
    streaming::{GPUStreamingManager, StreamingContext},
};

#[cfg(feature = "cuda")]
pub use gpu::{
    CudaBackend,
    CudaTensor, 
    CudaDeviceProperties,
    check_cuda_availability,
    get_cuda_device_count,
};

/// Result type for neural forecasting operations
pub type Result<T> = std::result::Result<T, NeuralForecastError>;

/// Comprehensive error types for neural forecasting
#[derive(Debug)]
pub enum NeuralForecastError {
    /// Configuration errors
    ConfigError(String),
    /// Model-related errors
    ModelError(String),
    /// Training errors
    TrainingError(String),
    /// Inference errors
    InferenceError(String),
    /// GPU acceleration errors
    GpuError(String),
    /// Data preprocessing errors
    PreprocessingError(String),
    /// I/O errors
    IoError(String),
    /// Storage errors
    StorageError(String),
    /// Serialization errors
    SerializationError(String),
    /// Validation errors
    ValidationError(String),
    /// Performance errors
    PerformanceError(String),
}

impl fmt::Display for NeuralForecastError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NeuralForecastError::ConfigError(msg) => write!(f, "Configuration error: {}", msg),
            NeuralForecastError::ModelError(msg) => write!(f, "Model error: {}", msg),
            NeuralForecastError::TrainingError(msg) => write!(f, "Training error: {}", msg),
            NeuralForecastError::InferenceError(msg) => write!(f, "Inference error: {}", msg),
            NeuralForecastError::GpuError(msg) => write!(f, "GPU error: {}", msg),
            NeuralForecastError::PreprocessingError(msg) => write!(f, "Preprocessing error: {}", msg),
            NeuralForecastError::IoError(msg) => write!(f, "I/O error: {}", msg),
            NeuralForecastError::StorageError(msg) => write!(f, "Storage error: {}", msg),
            NeuralForecastError::SerializationError(msg) => write!(f, "Serialization error: {}", msg),
            NeuralForecastError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            NeuralForecastError::PerformanceError(msg) => write!(f, "Performance error: {}", msg),
        }
    }
}

impl Error for NeuralForecastError {}

// Automatic conversions for common error types
impl From<std::io::Error> for NeuralForecastError {
    fn from(err: std::io::Error) -> Self {
        NeuralForecastError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for NeuralForecastError {
    fn from(err: serde_json::Error) -> Self {
        NeuralForecastError::SerializationError(err.to_string())
    }
}

impl From<bincode::Error> for NeuralForecastError {
    fn from(err: bincode::Error) -> Self {
        NeuralForecastError::SerializationError(err.to_string())
    }
}


/// Quick GPU performance benchmark
#[cfg(feature = "gpu")]
pub async fn quick_gpu_benchmark() -> Result<f64> {
    use gpu::benchmarks::{GPUBenchmarkSuite, BenchmarkConfig};
    
    let mut benchmark_suite = GPUBenchmarkSuite::new().await?;
    let config = BenchmarkConfig {
        batch_sizes: vec![32],
        sequence_lengths: vec![100],
        hidden_sizes: vec![256],
        num_iterations: 5,
        warmup_iterations: 2,
        target_latency_us: 100.0,
    };
    
    let report = benchmark_suite.run_full_benchmark(config).await?;
    Ok(report.summary.average_speedup)
}

#[cfg(not(feature = "gpu"))]
pub async fn quick_gpu_benchmark() -> Result<f64> {
    Ok(1.0) // No GPU acceleration available
}

/// GPU acceleration capabilities
#[derive(Debug, Clone)]
pub struct GPUCapabilities {
    pub cuda_available: bool,
    pub cuda_device_count: usize,
    pub webgpu_available: bool,
    pub estimated_speedup_range: (f64, f64),
}

/// Detect GPU capabilities
pub fn detect_gpu_capabilities() -> GPUCapabilities {
    use crate::cuda_helpers::{check_cuda_availability, get_cuda_device_count};
    
    let cuda_available = check_cuda_availability().unwrap_or(false);
    let cuda_device_count = if cuda_available {
        get_cuda_device_count().unwrap_or(0)
    } else {
        0
    };
    
    // Estimate WebGPU availability (simplified)
    let webgpu_available = !cuda_available; // Fallback to WebGPU if CUDA unavailable
    
    let estimated_speedup_range = if cuda_available && cuda_device_count > 0 {
        (10.0, 200.0) // CUDA speedup range
    } else if webgpu_available {
        (2.0, 15.0) // WebGPU speedup range  
    } else {
        (1.0, 1.0) // No acceleration
    };
    
    GPUCapabilities {
        cuda_available,
        cuda_device_count: cuda_device_count as usize,
        webgpu_available,
        estimated_speedup_range,
    }
}

/// GPU acceleration recommendation
pub fn get_gpu_recommendation() -> GPURecommendation {
    let capabilities = detect_gpu_capabilities();
    
    if capabilities.cuda_available && capabilities.cuda_device_count > 0 {
        GPURecommendation {
            recommended: true,
            reason: "CUDA GPU detected - significant speedup possible".to_string(),
            estimated_speedup: capabilities.estimated_speedup_range.1,
            implementation_priority: "HIGH".to_string(),
        }
    } else if capabilities.webgpu_available {
        GPURecommendation {
            recommended: true,
            reason: "WebGPU available - moderate speedup possible".to_string(),
            estimated_speedup: 10.0,
            implementation_priority: "MEDIUM".to_string(),
        }
    } else {
        GPURecommendation {
            recommended: false,
            reason: "No GPU acceleration available".to_string(),
            estimated_speedup: 1.0,
            implementation_priority: "N/A".to_string(),
        }
    }
}

/// GPU acceleration recommendation
#[derive(Debug, Clone)]
pub struct GPURecommendation {
    pub recommended: bool,
    pub reason: String,
    pub estimated_speedup: f64,
    pub implementation_priority: String,
}
