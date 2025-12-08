//! Backend implementations for CDFA unified library
//! 
//! This module provides abstraction layers for different computational backends
//! used throughout the CDFA library for financial analysis.

use crate::error::{CdfaError, Result};
// Float type available through types module

/// SIMD computation backend for vectorized operations
#[derive(Debug, Clone, Copy)]
pub enum SimdBackend {
    /// Automatic detection of best SIMD backend
    Auto,
    /// Scalar fallback (no SIMD)
    Scalar,
    /// Wide crate backend
    Wide,
    /// Platform-specific SIMD
    Native,
}

impl Default for SimdBackend {
    fn default() -> Self {
        SimdBackend::Auto
    }
}

impl SimdBackend {
    /// Detect the best available SIMD backend
    pub fn detect() -> Self {
        #[cfg(feature = "simd")]
        {
            // Try to use Wide backend if available
            #[cfg(target_arch = "x86_64")]
            if std::arch::is_x86_feature_detected!("avx2") {
                return SimdBackend::Wide;
            }
            
            #[cfg(target_arch = "aarch64")]
            if std::arch::is_aarch64_feature_detected!("neon") {
                return SimdBackend::Wide;
            }
        }
        
        SimdBackend::Scalar
    }
    
    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        match self {
            SimdBackend::Auto => true,
            SimdBackend::Scalar => true,
            SimdBackend::Wide => cfg!(feature = "simd"),
            SimdBackend::Native => cfg!(feature = "simd"),
        }
    }
}

/// Parallel computation backend for multi-threaded operations
#[derive(Debug, Clone)]
pub enum ParallelBackend {
    /// Automatic detection of best parallel backend
    Auto,
    /// Single-threaded (no parallelism)
    Sequential,
    /// Rayon work-stealing
    Rayon,
    /// Manual thread pool
    ThreadPool { num_threads: usize },
}

impl Default for ParallelBackend {
    fn default() -> Self {
        ParallelBackend::Auto
    }
}

impl ParallelBackend {
    /// Detect the best available parallel backend
    pub fn detect() -> Self {
        #[cfg(feature = "parallel")]
        {
            ParallelBackend::Rayon
        }
        #[cfg(not(feature = "parallel"))]
        {
            ParallelBackend::Sequential
        }
    }
    
    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        match self {
            ParallelBackend::Auto => true,
            ParallelBackend::Sequential => true,
            ParallelBackend::Rayon => cfg!(feature = "parallel"),
            ParallelBackend::ThreadPool { .. } => cfg!(feature = "parallel"),
        }
    }
    
    /// Get the number of threads for this backend
    pub fn num_threads(&self) -> usize {
        match self {
            ParallelBackend::Auto | ParallelBackend::Rayon => {
                #[cfg(feature = "parallel")]
                {
                    rayon::current_num_threads()
                }
                #[cfg(not(feature = "parallel"))]
                {
                    1
                }
            },
            ParallelBackend::Sequential => 1,
            ParallelBackend::ThreadPool { num_threads } => *num_threads,
        }
    }
}

/// Machine learning backend for neural network operations
#[derive(Debug, Clone)]
pub enum MLBackend {
    /// CPU-based computation
    Cpu,
    /// GPU acceleration via Candle
    #[cfg(feature = "candle")]
    Candle,
    /// Manual implementations
    Manual,
}

impl Default for MLBackend {
    fn default() -> Self {
        #[cfg(feature = "candle")]
        {
            MLBackend::Candle
        }
        #[cfg(not(feature = "candle"))]
        {
            MLBackend::Cpu
        }
    }
}

impl MLBackend {
    /// Check if this backend is available
    pub fn is_available(&self) -> bool {
        match self {
            MLBackend::Cpu => true,
            MLBackend::Manual => true,
            #[cfg(feature = "candle")]
            MLBackend::Candle => true,
        }
    }
}

/// Result type for ML operations
pub type MLResult<T> = std::result::Result<T, MLError>;

/// Error types for ML operations
#[derive(Debug, thiserror::Error)]
pub enum MLError {
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    #[error("Backend not available: {0}")]
    BackendNotAvailable(String),
    
    #[error("Computation failed: {0}")]
    ComputationFailed(String),
    
    #[error("Model error: {0}")]
    ModelError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
}

impl From<MLError> for CdfaError {
    fn from(err: MLError) -> Self {
        CdfaError::computation_failed(format!("ML operation failed: {}", err))
    }
}

/// Configuration for computational backends
#[derive(Debug, Clone)]
pub struct BackendConfig {
    /// SIMD backend preference
    pub simd_backend: SimdBackend,
    
    /// Parallel backend preference
    pub parallel_backend: ParallelBackend,
    
    /// ML backend preference
    pub ml_backend: MLBackend,
    
    /// Enable fallback to slower backends if preferred is unavailable
    pub enable_fallback: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            simd_backend: SimdBackend::Auto,
            parallel_backend: ParallelBackend::Auto,
            ml_backend: MLBackend::default(),
            enable_fallback: true,
        }
    }
}

impl BackendConfig {
    /// Create a new backend configuration
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set SIMD backend preference
    pub fn with_simd_backend(mut self, backend: SimdBackend) -> Self {
        self.simd_backend = backend;
        self
    }
    
    /// Set parallel backend preference
    pub fn with_parallel_backend(mut self, backend: ParallelBackend) -> Self {
        self.parallel_backend = backend;
        self
    }
    
    /// Set ML backend preference
    pub fn with_ml_backend(mut self, backend: MLBackend) -> Self {
        self.ml_backend = backend;
        self
    }
    
    /// Enable or disable fallback mechanisms
    pub fn with_fallback(mut self, enable: bool) -> Self {
        self.enable_fallback = enable;
        self
    }
    
    /// Validate that the configured backends are available
    pub fn validate(&self) -> Result<()> {
        if !self.simd_backend.is_available() {
            if !self.enable_fallback {
                return Err(CdfaError::invalid_input("SIMD backend not available"));
            }
        }
        
        if !self.parallel_backend.is_available() {
            if !self.enable_fallback {
                return Err(CdfaError::invalid_input("Parallel backend not available"));
            }
        }
        
        if !self.ml_backend.is_available() {
            if !self.enable_fallback {
                return Err(CdfaError::invalid_input("ML backend not available"));
            }
        }
        
        Ok(())
    }
}

/// Backend manager for coordinating different computational backends
#[derive(Debug)]
pub struct BackendManager {
    config: BackendConfig,
    active_simd: SimdBackend,
    active_parallel: ParallelBackend,
    active_ml: MLBackend,
}

impl BackendManager {
    /// Create a new backend manager with the given configuration
    pub fn new(config: BackendConfig) -> Result<Self> {
        config.validate()?;
        
        let active_simd = if config.simd_backend.is_available() {
            config.simd_backend
        } else if config.enable_fallback {
            SimdBackend::Scalar
        } else {
            return Err(CdfaError::invalid_input("SIMD backend not available"));
        };
        
        let active_parallel = if config.parallel_backend.is_available() {
            config.parallel_backend.clone()
        } else if config.enable_fallback {
            ParallelBackend::Sequential
        } else {
            return Err(CdfaError::invalid_input("Parallel backend not available"));
        };
        
        let active_ml = if config.ml_backend.is_available() {
            config.ml_backend.clone()
        } else if config.enable_fallback {
            MLBackend::Cpu
        } else {
            return Err(CdfaError::invalid_input("ML backend not available"));
        };
        
        Ok(Self {
            config,
            active_simd,
            active_parallel,
            active_ml,
        })
    }
    
    /// Get the active SIMD backend
    pub fn simd_backend(&self) -> SimdBackend {
        self.active_simd
    }
    
    /// Get the active parallel backend
    pub fn parallel_backend(&self) -> &ParallelBackend {
        &self.active_parallel
    }
    
    /// Get the active ML backend
    pub fn ml_backend(&self) -> &MLBackend {
        &self.active_ml
    }
    
    /// Get backend configuration
    pub fn config(&self) -> &BackendConfig {
        &self.config
    }
}

impl Default for BackendManager {
    fn default() -> Self {
        Self::new(BackendConfig::default())
            .expect("Default backend configuration should always be valid")
    }
}