//! GPU Correlation Acceleration Module for Parasitic Trading System
//!
//! This module provides high-performance correlation matrix computation for organism pairs
//! with GPU acceleration and SIMD fallback. Designed for sub-millisecond performance
//! requirements in real-time trading applications.
//!
//! Features:
//! - CUDA kernel simulation for GPU acceleration
//! - SIMD fallback using AVX2/AVX-512 when GPU unavailable  
//! - Adaptive engine that selects optimal compute backend
//! - Sub-millisecond latency for correlation matrix computation
//! - Memory-efficient streaming for large organism populations
//! - Thread-safe concurrent operations
//! - Comprehensive error handling and graceful degradation

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thiserror::Error;
use tokio::sync::{Mutex, RwLock};

pub mod correlation_matrix;
pub mod cuda_backend;
pub mod organism_vector;
pub mod simd_backend;

#[cfg(test)]
pub mod tests;

#[cfg(test)]
pub mod integration_test;

#[cfg(test)]
pub mod simple_test;

pub use correlation_matrix::*;
pub use cuda_backend::*;
pub use organism_vector::*;
pub use simd_backend::*;

/// Maximum supported organism count for correlation computation
pub const MAX_ORGANISMS: usize = 1024;

/// Target latency for correlation computation (microseconds)
pub const TARGET_LATENCY_MICROS: u64 = 1000; // 1ms

/// Memory alignment for SIMD operations
pub const SIMD_ALIGNMENT: usize = 64;

/// Core trait for correlation computation engines
#[async_trait::async_trait]
pub trait CorrelationEngine: Send + Sync {
    /// Compute correlation matrix for given organisms
    async fn compute_correlation_matrix(
        &self,
        organisms: &[OrganismVector],
    ) -> Result<CorrelationMatrix, CorrelationError>;

    /// Get engine type identifier
    fn engine_type(&self) -> &'static str;

    /// Check if engine is available on current system
    fn is_available(&self) -> bool;

    /// Get performance characteristics
    fn get_performance_info(&self) -> EnginePerformanceInfo;

    /// Cleanup resources
    async fn cleanup(&self) -> Result<(), CorrelationError> {
        Ok(())
    }
}

/// GPU-accelerated correlation engine using CUDA simulation
pub struct GpuCorrelationEngine {
    cuda_context: Arc<Mutex<CudaContext>>,
    device_info: GpuDeviceInfo,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
}

impl GpuCorrelationEngine {
    /// Create new GPU correlation engine
    pub async fn new() -> Result<Self, CorrelationError> {
        let device_info = detect_gpu_devices().await?;
        let cuda_device_info = cuda_backend::GpuDeviceInfo {
            device_name: device_info.name.clone(),
            memory_size: device_info.memory_size,
            compute_capability: (8, 6), // Default compute capability
            is_available: device_info.is_available,
            multiprocessor_count: 64,
            max_threads_per_block: 1024,
            warp_size: 32,
        };
        let cuda_context = Arc::new(Mutex::new(CudaContext::new(&cuda_device_info)?));
        let memory_pool = Arc::new(Mutex::new(GpuMemoryPool::new(device_info.memory_size)?));

        Ok(Self {
            cuda_context,
            device_info,
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
            memory_pool,
        })
    }

    /// Check if GPU is available
    pub fn is_gpu_available(&self) -> bool {
        self.device_info.is_available
    }

    /// Get GPU information string
    pub fn get_gpu_info(&self) -> String {
        format!(
            "{} ({}MB VRAM)",
            self.device_info.name,
            self.device_info.memory_size / (1024 * 1024)
        )
    }
}

#[async_trait::async_trait]
impl CorrelationEngine for GpuCorrelationEngine {
    async fn compute_correlation_matrix(
        &self,
        organisms: &[OrganismVector],
    ) -> Result<CorrelationMatrix, CorrelationError> {
        if organisms.is_empty() {
            return Err(CorrelationError::EmptyInput);
        }

        if organisms.len() > MAX_ORGANISMS {
            return Err(CorrelationError::InputTooLarge(organisms.len()));
        }

        let start_time = Instant::now();

        // Validate input data
        for (i, organism) in organisms.iter().enumerate() {
            organism
                .validate()
                .map_err(|e| CorrelationError::InvalidOrganism(i, e))?;
        }

        let cuda_context = self.cuda_context.lock().await;
        let mut memory_pool = self.memory_pool.lock().await;

        // Allocate GPU memory
        let input_buffer = memory_pool.allocate_input_buffer(organisms, &cuda_context)?;
        let output_buffer = memory_pool.allocate_output_buffer(organisms.len(), &cuda_context)?;

        // Upload data to GPU
        cuda_context
            .upload_organisms(&input_buffer, organisms)
            .await?;

        // Launch correlation kernel
        let kernel_params = CorrelationKernelParams {
            organism_count: organisms.len(),
            feature_size: organisms[0].features().len(),
            performance_size: organisms[0].performance_history().len(),
        };

        cuda_context
            .launch_correlation_kernel(&kernel_params, &input_buffer, &output_buffer)
            .await?;

        // Download result
        let correlation_data = cuda_context
            .download_correlation_matrix(&output_buffer, organisms.len())
            .await?;

        // Create correlation matrix
        let matrix = CorrelationMatrix::new(organisms.len(), correlation_data)?;

        // Update performance metrics
        let computation_time = start_time.elapsed();
        let mut metrics = self.performance_metrics.write().await;
        metrics.record_computation(organisms.len(), computation_time);

        if computation_time.as_micros() as u64 > TARGET_LATENCY_MICROS {
            tracing::warn!(
                "GPU correlation computation exceeded target latency: {}μs",
                computation_time.as_micros()
            );
        }

        Ok(matrix)
    }

    fn engine_type(&self) -> &'static str {
        "GPU-CUDA"
    }

    fn is_available(&self) -> bool {
        self.device_info.is_available
    }

    fn get_performance_info(&self) -> EnginePerformanceInfo {
        EnginePerformanceInfo {
            engine_type: self.engine_type().to_string(),
            max_organisms: MAX_ORGANISMS,
            estimated_latency_micros: 200, // GPU typically faster
            memory_requirement_mb: 512,
            parallel_capability: true,
        }
    }

    async fn cleanup(&self) -> Result<(), CorrelationError> {
        let mut cuda_context = self.cuda_context.lock().await;
        let mut memory_pool = self.memory_pool.lock().await;

        memory_pool.cleanup();
        cuda_context.cleanup()?;

        Ok(())
    }
}

/// SIMD-accelerated correlation engine (fallback when GPU unavailable)
pub struct SimdCorrelationEngine {
    simd_features: SimdFeatures,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
}

impl SimdCorrelationEngine {
    pub fn new() -> Self {
        Self {
            simd_features: detect_simd_features(),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::new())),
        }
    }
}

#[async_trait::async_trait]
impl CorrelationEngine for SimdCorrelationEngine {
    async fn compute_correlation_matrix(
        &self,
        organisms: &[OrganismVector],
    ) -> Result<CorrelationMatrix, CorrelationError> {
        if organisms.is_empty() {
            return Err(CorrelationError::EmptyInput);
        }

        if organisms.len() > MAX_ORGANISMS {
            return Err(CorrelationError::InputTooLarge(organisms.len()));
        }

        let start_time = Instant::now();

        // Validate input data
        for (i, organism) in organisms.iter().enumerate() {
            organism
                .validate()
                .map_err(|e| CorrelationError::InvalidOrganism(i, e))?;
        }

        // Compute correlation matrix using SIMD
        let correlation_data = if self.simd_features.has_avx512f {
            compute_correlation_matrix_avx512(organisms)?
        } else if self.simd_features.has_avx2 {
            compute_correlation_matrix_avx2(organisms)?
        } else {
            compute_correlation_matrix_scalar(organisms)?
        };

        let matrix = CorrelationMatrix::new(organisms.len(), correlation_data)?;

        // Update performance metrics
        let computation_time = start_time.elapsed();
        let mut metrics = self.performance_metrics.write().await;
        metrics.record_computation(organisms.len(), computation_time);

        Ok(matrix)
    }

    fn engine_type(&self) -> &'static str {
        if self.simd_features.has_avx512f {
            "SIMD-AVX512"
        } else if self.simd_features.has_avx2 {
            "SIMD-AVX2"
        } else {
            "SIMD-Scalar"
        }
    }

    fn is_available(&self) -> bool {
        true // SIMD fallback always available
    }

    fn get_performance_info(&self) -> EnginePerformanceInfo {
        let estimated_latency = if self.simd_features.has_avx512f {
            500
        } else if self.simd_features.has_avx2 {
            800
        } else {
            1200
        };

        EnginePerformanceInfo {
            engine_type: self.engine_type().to_string(),
            max_organisms: MAX_ORGANISMS,
            estimated_latency_micros: estimated_latency,
            memory_requirement_mb: 256,
            parallel_capability: false,
        }
    }
}

/// Adaptive correlation engine that selects optimal backend
pub struct AdaptiveCorrelationEngine {
    gpu_engine: Option<GpuCorrelationEngine>,
    simd_engine: SimdCorrelationEngine,
    active_backend: Arc<RwLock<String>>,
    performance_history: Arc<RwLock<HashMap<String, Vec<Duration>>>>,
}

impl AdaptiveCorrelationEngine {
    pub async fn new() -> Result<Self, CorrelationError> {
        let gpu_engine = match GpuCorrelationEngine::new().await {
            Ok(engine) => Some(engine),
            Err(_) => None,
        };

        let simd_engine = SimdCorrelationEngine::new();

        let active_backend = Arc::new(RwLock::new(
            if gpu_engine.is_some() {
                "GPU-CUDA"
            } else {
                "SIMD"
            }
            .to_string(),
        ));

        Ok(Self {
            gpu_engine,
            simd_engine,
            active_backend,
            performance_history: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub async fn get_active_backend(&self) -> String {
        self.active_backend.read().await.clone()
    }

    /// Select optimal backend based on workload characteristics
    async fn select_optimal_backend(&self, organism_count: usize) -> &dyn CorrelationEngine {
        // For small workloads, SIMD might be faster due to GPU setup overhead
        if organism_count < 32 {
            return &self.simd_engine;
        }

        // Check performance history to make informed decision
        let history = self.performance_history.read().await;

        if let Some(gpu_engine) = &self.gpu_engine {
            if let (Some(gpu_times), Some(simd_times)) =
                (history.get("GPU-CUDA"), history.get("SIMD"))
            {
                let gpu_avg = gpu_times.iter().map(|d| d.as_micros() as f64).sum::<f64>()
                    / gpu_times.len() as f64;
                let simd_avg = simd_times.iter().map(|d| d.as_micros() as f64).sum::<f64>()
                    / simd_times.len() as f64;

                if gpu_avg < simd_avg {
                    return gpu_engine;
                }
            } else {
                // No history available, prefer GPU for larger workloads
                return gpu_engine;
            }
        }

        &self.simd_engine
    }

    /// Record performance for backend selection
    async fn record_performance(&self, backend: &str, duration: Duration) {
        let mut history = self.performance_history.write().await;
        let times = history.entry(backend.to_string()).or_insert_with(Vec::new);

        times.push(duration);

        // Keep only recent measurements (sliding window)
        if times.len() > 10 {
            times.remove(0);
        }
    }
}

#[async_trait::async_trait]
impl CorrelationEngine for AdaptiveCorrelationEngine {
    async fn compute_correlation_matrix(
        &self,
        organisms: &[OrganismVector],
    ) -> Result<CorrelationMatrix, CorrelationError> {
        let start_time = Instant::now();

        let engine = self.select_optimal_backend(organisms.len()).await;
        let backend_name = engine.engine_type().to_string();

        // Update active backend
        *self.active_backend.write().await = backend_name.clone();

        let result = engine.compute_correlation_matrix(organisms).await;

        // Record performance
        let computation_time = start_time.elapsed();
        self.record_performance(&backend_name, computation_time)
            .await;

        result
    }

    fn engine_type(&self) -> &'static str {
        "Adaptive"
    }

    fn is_available(&self) -> bool {
        true // Always available due to SIMD fallback
    }

    fn get_performance_info(&self) -> EnginePerformanceInfo {
        if self.gpu_engine.is_some() {
            EnginePerformanceInfo {
                engine_type: "Adaptive-GPU-SIMD".to_string(),
                max_organisms: MAX_ORGANISMS,
                estimated_latency_micros: 200,
                memory_requirement_mb: 512,
                parallel_capability: true,
            }
        } else {
            self.simd_engine.get_performance_info()
        }
    }

    async fn cleanup(&self) -> Result<(), CorrelationError> {
        if let Some(gpu_engine) = &self.gpu_engine {
            gpu_engine.cleanup().await?;
        }
        self.simd_engine.cleanup().await?;
        Ok(())
    }
}

/// Performance metrics tracking
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    total_computations: u64,
    total_time: Duration,
    min_time: Option<Duration>,
    max_time: Option<Duration>,
    organism_counts: Vec<usize>,
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_computations: 0,
            total_time: Duration::ZERO,
            min_time: None,
            max_time: None,
            organism_counts: Vec::new(),
        }
    }

    pub fn record_computation(&mut self, organism_count: usize, duration: Duration) {
        self.total_computations += 1;
        self.total_time += duration;
        self.organism_counts.push(organism_count);

        match self.min_time {
            None => self.min_time = Some(duration),
            Some(min) if duration < min => self.min_time = Some(duration),
            _ => {}
        }

        match self.max_time {
            None => self.max_time = Some(duration),
            Some(max) if duration > max => self.max_time = Some(duration),
            _ => {}
        }
    }

    pub fn average_time(&self) -> Duration {
        if self.total_computations == 0 {
            Duration::ZERO
        } else {
            self.total_time / self.total_computations as u32
        }
    }
}

/// Engine performance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnginePerformanceInfo {
    pub engine_type: String,
    pub max_organisms: usize,
    pub estimated_latency_micros: u64,
    pub memory_requirement_mb: usize,
    pub parallel_capability: bool,
}

/// GPU device information
#[derive(Debug, Clone)]
pub struct GpuDeviceInfo {
    pub name: String,
    pub memory_size: usize,
    pub compute_capability: String,
    pub is_available: bool,
}

/// SIMD feature detection
#[derive(Debug, Clone)]
pub struct SimdFeatures {
    pub has_sse42: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512bw: bool,
    pub has_fma: bool,
}

/// Correlation computation errors
#[derive(Debug, Error, Clone, PartialEq)]
pub enum CorrelationError {
    #[error("Empty input provided")]
    EmptyInput,

    #[error("Input too large: {0} organisms (max {MAX_ORGANISMS})")]
    InputTooLarge(usize),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Invalid organism at index {0}: {1}")]
    InvalidOrganism(usize, String),

    #[error("GPU error: {0}")]
    GpuError(String),

    #[error("CUDA error: {0}")]
    CudaError(String),

    #[error("Memory allocation error: {0}")]
    MemoryError(String),

    #[error("Out of memory: requested {0} bytes")]
    OutOfMemory(usize),

    #[error("Matrix computation error: {0}")]
    ComputationError(String),

    #[error("Performance requirement not met: {0}μs (target: {TARGET_LATENCY_MICROS}μs)")]
    PerformanceError(u64),
}

/// Correlation kernel parameters for GPU computation
#[derive(Debug, Clone)]
pub struct CorrelationKernelParams {
    pub organism_count: usize,
    pub feature_size: usize,
    pub performance_size: usize,
}

// Feature detection functions

/// Detect available GPU devices
async fn detect_gpu_devices() -> Result<GpuDeviceInfo, CorrelationError> {
    // Simulate GPU detection - in real implementation would use CUDA/OpenCL
    tokio::task::spawn_blocking(|| {
        if std::env::var("CUDA_VISIBLE_DEVICES").is_ok() {
            Ok(GpuDeviceInfo {
                name: "NVIDIA GPU (Simulated)".to_string(),
                memory_size: 8 * 1024 * 1024 * 1024, // 8GB
                compute_capability: "8.6".to_string(),
                is_available: true,
            })
        } else {
            Err(CorrelationError::GpuError(
                "No CUDA devices found".to_string(),
            ))
        }
    })
    .await
    .map_err(|e| CorrelationError::GpuError(format!("Detection failed: {}", e)))?
}

/// Detect available SIMD features
fn detect_simd_features() -> SimdFeatures {
    SimdFeatures {
        has_sse42: is_x86_feature_detected!("sse4.2"),
        has_avx2: is_x86_feature_detected!("avx2"),
        has_avx512f: is_x86_feature_detected!("avx512f"),
        has_avx512bw: is_x86_feature_detected!("avx512bw"),
        has_fma: is_x86_feature_detected!("fma"),
    }
}

// SIMD correlation computation functions (implemented in simd_backend.rs)

fn compute_correlation_matrix_avx512(
    organisms: &[OrganismVector],
) -> Result<Vec<f32>, CorrelationError> {
    let n = organisms.len();
    let mut correlation_data = vec![0.0f32; n * n];

    unsafe {
        compute_correlations_avx512(organisms, &mut correlation_data)?;
    }

    Ok(correlation_data)
}

fn compute_correlation_matrix_avx2(
    organisms: &[OrganismVector],
) -> Result<Vec<f32>, CorrelationError> {
    let n = organisms.len();
    let mut correlation_data = vec![0.0f32; n * n];

    unsafe {
        compute_correlations_avx2(organisms, &mut correlation_data)?;
    }

    Ok(correlation_data)
}

fn compute_correlation_matrix_scalar(
    organisms: &[OrganismVector],
) -> Result<Vec<f32>, CorrelationError> {
    let n = organisms.len();
    let mut correlation_data = vec![0.0f32; n * n];

    compute_correlations_scalar(organisms, &mut correlation_data)?;

    Ok(correlation_data)
}
