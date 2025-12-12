//! Error types for QBMIA GPU acceleration

use std::fmt;
use thiserror::Error;

/// Main error type for QBMIA GPU acceleration
#[derive(Error, Debug)]
pub enum QBMIAError {
    /// GPU initialization error
    #[error("GPU initialization failed: {0}")]
    GpuInitializationError(String),
    
    /// GPU computation error
    #[error("GPU computation failed: {0}")]
    GpuComputationError(String),
    
    /// Memory allocation error
    #[error("Memory allocation failed: {0}")]
    MemoryAllocationError(String),
    
    /// Buffer operation error
    #[error("Buffer operation failed: {0}")]
    BufferOperationError(String),
    
    /// Shader compilation error
    #[error("Shader compilation failed: {0}")]
    ShaderCompilationError(String),
    
    /// Kernel execution error
    #[error("Kernel execution failed: {0}")]
    KernelExecutionError(String),
    
    /// Invalid parameter error
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    
    /// Quantum state error
    #[error("Quantum state error: {0}")]
    QuantumStateError(String),
    
    /// Nash equilibrium solver error
    #[error("Nash equilibrium solver error: {0}")]
    NashSolverError(String),
    
    /// Pattern matching error
    #[error("Pattern matching error: {0}")]
    PatternMatchingError(String),
    
    /// SIMD operation error
    #[error("SIMD operation error: {0}")]
    SimdOperationError(String),
    
    /// Timeout error
    #[error("Operation timed out: {0}")]
    TimeoutError(String),
    
    /// Resource exhaustion error
    #[error("Resource exhausted: {0}")]
    ResourceExhaustedError(String),
    
    /// Synchronization error
    #[error("Synchronization error: {0}")]
    SynchronizationError(String),
    
    /// Hardware not supported error
    #[error("Hardware not supported: {0}")]
    HardwareNotSupportedError(String),
    
    /// Configuration error
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    /// IO error
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    /// JSON error
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),
    
    /// Bincode error
    #[error("Bincode error: {0}")]
    BincodeError(#[from] bincode::Error),
    
    /// Tokio task join error
    #[error("Tokio task join error: {0}")]
    TokioJoinError(#[from] tokio::task::JoinError),
    
    /// Send error
    #[error("Send error: {0}")]
    SendError(String),
    
    /// Receive error
    #[error("Receive error: {0}")]
    ReceiveError(String),
    
    /// Custom error
    #[error("Custom error: {0}")]
    Custom(String),
}

impl QBMIAError {
    /// Create a new GPU initialization error
    pub fn gpu_init<T: ToString>(msg: T) -> Self {
        Self::GpuInitializationError(msg.to_string())
    }
    
    /// Create a new GPU computation error
    pub fn gpu_compute<T: ToString>(msg: T) -> Self {
        Self::GpuComputationError(msg.to_string())
    }
    
    /// Create a new memory allocation error
    pub fn memory_alloc<T: ToString>(msg: T) -> Self {
        Self::MemoryAllocationError(msg.to_string())
    }
    
    /// Create a new buffer operation error
    pub fn buffer_op<T: ToString>(msg: T) -> Self {
        Self::BufferOperationError(msg.to_string())
    }
    
    /// Create a new shader compilation error
    pub fn shader_compile<T: ToString>(msg: T) -> Self {
        Self::ShaderCompilationError(msg.to_string())
    }
    
    /// Create a new kernel execution error
    pub fn kernel_exec<T: ToString>(msg: T) -> Self {
        Self::KernelExecutionError(msg.to_string())
    }
    
    /// Create a new invalid parameter error
    pub fn invalid_param<T: ToString>(msg: T) -> Self {
        Self::InvalidParameter(msg.to_string())
    }
    
    /// Create a new quantum state error
    pub fn quantum_state<T: ToString>(msg: T) -> Self {
        Self::QuantumStateError(msg.to_string())
    }
    
    /// Create a new Nash solver error
    pub fn nash_solver<T: ToString>(msg: T) -> Self {
        Self::NashSolverError(msg.to_string())
    }
    
    /// Create a new pattern matching error
    pub fn pattern_match<T: ToString>(msg: T) -> Self {
        Self::PatternMatchingError(msg.to_string())
    }
    
    /// Create a new SIMD operation error
    pub fn simd_op<T: ToString>(msg: T) -> Self {
        Self::SimdOperationError(msg.to_string())
    }
    
    /// Create a new timeout error
    pub fn timeout<T: ToString>(msg: T) -> Self {
        Self::TimeoutError(msg.to_string())
    }
    
    /// Create a new resource exhausted error
    pub fn resource_exhausted<T: ToString>(msg: T) -> Self {
        Self::ResourceExhaustedError(msg.to_string())
    }
    
    /// Create a new synchronization error
    pub fn sync<T: ToString>(msg: T) -> Self {
        Self::SynchronizationError(msg.to_string())
    }
    
    /// Create a new hardware not supported error
    pub fn hardware_not_supported<T: ToString>(msg: T) -> Self {
        Self::HardwareNotSupportedError(msg.to_string())
    }
    
    /// Create a new configuration error
    pub fn config<T: ToString>(msg: T) -> Self {
        Self::ConfigurationError(msg.to_string())
    }
    
    /// Create a new serialization error
    pub fn serialization<T: ToString>(msg: T) -> Self {
        Self::SerializationError(msg.to_string())
    }
    
    /// Create a new send error
    pub fn send<T: ToString>(msg: T) -> Self {
        Self::SendError(msg.to_string())
    }
    
    /// Create a new receive error
    pub fn receive<T: ToString>(msg: T) -> Self {
        Self::ReceiveError(msg.to_string())
    }
    
    /// Create a new custom error
    pub fn custom<T: ToString>(msg: T) -> Self {
        Self::Custom(msg.to_string())
    }
    
    /// Check if this error is related to GPU operations
    pub fn is_gpu_error(&self) -> bool {
        matches!(
            self,
            Self::GpuInitializationError(_) |
            Self::GpuComputationError(_) |
            Self::ShaderCompilationError(_) |
            Self::KernelExecutionError(_) |
            Self::HardwareNotSupportedError(_)
        )
    }
    
    /// Check if this error is related to memory operations
    pub fn is_memory_error(&self) -> bool {
        matches!(
            self,
            Self::MemoryAllocationError(_) |
            Self::BufferOperationError(_) |
            Self::ResourceExhaustedError(_)
        )
    }
    
    /// Check if this error is related to parameter validation
    pub fn is_parameter_error(&self) -> bool {
        matches!(
            self,
            Self::InvalidParameter(_) |
            Self::ConfigurationError(_)
        )
    }
    
    /// Check if this error is related to quantum computations
    pub fn is_quantum_error(&self) -> bool {
        matches!(
            self,
            Self::QuantumStateError(_)
        )
    }
    
    /// Check if this error is related to Nash equilibrium solving
    pub fn is_nash_error(&self) -> bool {
        matches!(
            self,
            Self::NashSolverError(_)
        )
    }
    
    /// Check if this error is related to pattern matching
    pub fn is_pattern_error(&self) -> bool {
        matches!(
            self,
            Self::PatternMatchingError(_)
        )
    }
    
    /// Check if this error is related to SIMD operations
    pub fn is_simd_error(&self) -> bool {
        matches!(
            self,
            Self::SimdOperationError(_)
        )
    }
    
    /// Check if this error is related to timeouts
    pub fn is_timeout_error(&self) -> bool {
        matches!(
            self,
            Self::TimeoutError(_)
        )
    }
    
    /// Check if this error is related to synchronization
    pub fn is_sync_error(&self) -> bool {
        matches!(
            self,
            Self::SynchronizationError(_) |
            Self::SendError(_) |
            Self::ReceiveError(_) |
            Self::TokioJoinError(_)
        )
    }
    
    /// Check if this error is related to serialization
    pub fn is_serialization_error(&self) -> bool {
        matches!(
            self,
            Self::SerializationError(_) |
            Self::JsonError(_) |
            Self::BincodeError(_)
        )
    }
    
    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Non-recoverable errors
            Self::GpuInitializationError(_) |
            Self::HardwareNotSupportedError(_) |
            Self::ConfigurationError(_) |
            Self::InvalidParameter(_) => false,
            
            // Potentially recoverable errors
            Self::GpuComputationError(_) |
            Self::MemoryAllocationError(_) |
            Self::BufferOperationError(_) |
            Self::KernelExecutionError(_) |
            Self::TimeoutError(_) |
            Self::ResourceExhaustedError(_) |
            Self::SynchronizationError(_) => true,
            
            // Domain-specific errors (usually recoverable)
            Self::QuantumStateError(_) |
            Self::NashSolverError(_) |
            Self::PatternMatchingError(_) |
            Self::SimdOperationError(_) => true,
            
            // Serialization errors (usually recoverable)
            Self::SerializationError(_) |
            Self::JsonError(_) |
            Self::BincodeError(_) => true,
            
            // IO errors (context-dependent)
            Self::IoError(_) => true,
            
            // Task errors (usually recoverable)
            Self::TokioJoinError(_) |
            Self::SendError(_) |
            Self::ReceiveError(_) => true,
            
            // Shader compilation errors (usually not recoverable)
            Self::ShaderCompilationError(_) => false,
            
            // Custom errors (unknown)
            Self::Custom(_) => true,
        }
    }
    
    /// Get the error category
    pub fn category(&self) -> ErrorCategory {
        match self {
            Self::GpuInitializationError(_) |
            Self::GpuComputationError(_) |
            Self::ShaderCompilationError(_) |
            Self::KernelExecutionError(_) |
            Self::HardwareNotSupportedError(_) => ErrorCategory::Gpu,
            
            Self::MemoryAllocationError(_) |
            Self::BufferOperationError(_) |
            Self::ResourceExhaustedError(_) => ErrorCategory::Memory,
            
            Self::InvalidParameter(_) |
            Self::ConfigurationError(_) => ErrorCategory::Parameter,
            
            Self::QuantumStateError(_) => ErrorCategory::Quantum,
            
            Self::NashSolverError(_) => ErrorCategory::Nash,
            
            Self::PatternMatchingError(_) => ErrorCategory::Pattern,
            
            Self::SimdOperationError(_) => ErrorCategory::Simd,
            
            Self::TimeoutError(_) => ErrorCategory::Timeout,
            
            Self::SynchronizationError(_) |
            Self::SendError(_) |
            Self::ReceiveError(_) |
            Self::TokioJoinError(_) => ErrorCategory::Synchronization,
            
            Self::SerializationError(_) |
            Self::JsonError(_) |
            Self::BincodeError(_) => ErrorCategory::Serialization,
            
            Self::IoError(_) => ErrorCategory::Io,
            
            Self::Custom(_) => ErrorCategory::Custom,
        }
    }
}

/// Error category for classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCategory {
    /// GPU-related errors
    Gpu,
    /// Memory-related errors
    Memory,
    /// Parameter validation errors
    Parameter,
    /// Quantum computation errors
    Quantum,
    /// Nash equilibrium solver errors
    Nash,
    /// Pattern matching errors
    Pattern,
    /// SIMD operation errors
    Simd,
    /// Timeout errors
    Timeout,
    /// Synchronization errors
    Synchronization,
    /// Serialization errors
    Serialization,
    /// IO errors
    Io,
    /// Custom errors
    Custom,
}

impl fmt::Display for ErrorCategory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gpu => write!(f, "GPU"),
            Self::Memory => write!(f, "Memory"),
            Self::Parameter => write!(f, "Parameter"),
            Self::Quantum => write!(f, "Quantum"),
            Self::Nash => write!(f, "Nash"),
            Self::Pattern => write!(f, "Pattern"),
            Self::Simd => write!(f, "SIMD"),
            Self::Timeout => write!(f, "Timeout"),
            Self::Synchronization => write!(f, "Synchronization"),
            Self::Serialization => write!(f, "Serialization"),
            Self::Io => write!(f, "IO"),
            Self::Custom => write!(f, "Custom"),
        }
    }
}

/// Result type alias for QBMIA operations
pub type QBMIAResult<T> = Result<T, QBMIAError>;

/// Error context for better error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    /// Operation that failed
    pub operation: String,
    
    /// Additional context information
    pub context: std::collections::HashMap<String, String>,
    
    /// Timestamp when error occurred
    pub timestamp: std::time::SystemTime,
}

impl ErrorContext {
    /// Create a new error context
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            context: std::collections::HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        }
    }
    
    /// Add context information
    pub fn with_context<K, V>(mut self, key: K, value: V) -> Self
    where
        K: ToString,
        V: ToString,
    {
        self.context.insert(key.to_string(), value.to_string());
        self
    }
    
    /// Get context value
    pub fn get_context(&self, key: &str) -> Option<&String> {
        self.context.get(key)
    }
}

/// Enhanced error with context
#[derive(Debug)]
pub struct QBMIAContextError {
    /// Original error
    pub error: QBMIAError,
    
    /// Error context
    pub context: ErrorContext,
}

impl fmt::Display for QBMIAContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} (operation: {})", self.error, self.context.operation)?;
        
        if !self.context.context.is_empty() {
            write!(f, " [")?;
            for (i, (key, value)) in self.context.context.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", key, value)?;
            }
            write!(f, "]")?;
        }
        
        Ok(())
    }
}

impl std::error::Error for QBMIAContextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.error)
    }
}

/// Trait for adding context to errors
pub trait ErrorContextExt<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> Result<T, QBMIAContextError>
    where
        F: FnOnce() -> ErrorContext;
}

impl<T> ErrorContextExt<T> for Result<T, QBMIAError> {
    fn with_context<F>(self, f: F) -> Result<T, QBMIAContextError>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|error| QBMIAContextError {
            error,
            context: f(),
        })
    }
}

/// Error recovery strategy
#[derive(Debug, Clone)]
pub enum ErrorRecoveryStrategy {
    /// Retry the operation
    Retry {
        /// Maximum number of retries
        max_retries: usize,
        /// Delay between retries
        delay: std::time::Duration,
    },
    /// Fallback to alternative method
    Fallback {
        /// Alternative method to use
        method: String,
    },
    /// Abort the operation
    Abort,
    /// Ignore the error
    Ignore,
}

impl QBMIAError {
    /// Get the recommended recovery strategy for this error
    pub fn recovery_strategy(&self) -> ErrorRecoveryStrategy {
        match self {
            // GPU initialization errors - abort
            Self::GpuInitializationError(_) |
            Self::HardwareNotSupportedError(_) |
            Self::ShaderCompilationError(_) => ErrorRecoveryStrategy::Abort,
            
            // Temporary GPU errors - retry
            Self::GpuComputationError(_) |
            Self::KernelExecutionError(_) |
            Self::TimeoutError(_) => ErrorRecoveryStrategy::Retry {
                max_retries: 3,
                delay: std::time::Duration::from_millis(100),
            },
            
            // Memory errors - retry with backoff
            Self::MemoryAllocationError(_) |
            Self::BufferOperationError(_) |
            Self::ResourceExhaustedError(_) => ErrorRecoveryStrategy::Retry {
                max_retries: 5,
                delay: std::time::Duration::from_millis(500),
            },
            
            // Parameter errors - abort
            Self::InvalidParameter(_) |
            Self::ConfigurationError(_) => ErrorRecoveryStrategy::Abort,
            
            // Domain-specific errors - fallback to CPU
            Self::QuantumStateError(_) |
            Self::NashSolverError(_) |
            Self::PatternMatchingError(_) => ErrorRecoveryStrategy::Fallback {
                method: "CPU".to_string(),
            },
            
            // SIMD errors - fallback to scalar
            Self::SimdOperationError(_) => ErrorRecoveryStrategy::Fallback {
                method: "scalar".to_string(),
            },
            
            // Synchronization errors - retry
            Self::SynchronizationError(_) |
            Self::SendError(_) |
            Self::ReceiveError(_) |
            Self::TokioJoinError(_) => ErrorRecoveryStrategy::Retry {
                max_retries: 3,
                delay: std::time::Duration::from_millis(50),
            },
            
            // Serialization errors - retry
            Self::SerializationError(_) |
            Self::JsonError(_) |
            Self::BincodeError(_) => ErrorRecoveryStrategy::Retry {
                max_retries: 2,
                delay: std::time::Duration::from_millis(10),
            },
            
            // IO errors - retry
            Self::IoError(_) => ErrorRecoveryStrategy::Retry {
                max_retries: 3,
                delay: std::time::Duration::from_millis(100),
            },
            
            // Custom errors - abort by default
            Self::Custom(_) => ErrorRecoveryStrategy::Abort,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_creation() {
        let error = QBMIAError::gpu_init("Failed to initialize GPU");
        assert!(error.is_gpu_error());
        assert!(!error.is_recoverable());
        assert_eq!(error.category(), ErrorCategory::Gpu);
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_operation".to_string())
            .with_context("device", "GeForce RTX 3080")
            .with_context("memory", "8GB");
        
        assert_eq!(context.operation, "test_operation");
        assert_eq!(context.get_context("device"), Some(&"GeForce RTX 3080".to_string()));
        assert_eq!(context.get_context("memory"), Some(&"8GB".to_string()));
    }
    
    #[test]
    fn test_recovery_strategy() {
        let gpu_error = QBMIAError::gpu_init("Test error");
        match gpu_error.recovery_strategy() {
            ErrorRecoveryStrategy::Abort => {},
            _ => panic!("Expected Abort strategy"),
        }
        
        let memory_error = QBMIAError::memory_alloc("Test error");
        match memory_error.recovery_strategy() {
            ErrorRecoveryStrategy::Retry { max_retries, .. } => {
                assert_eq!(max_retries, 5);
            },
            _ => panic!("Expected Retry strategy"),
        }
    }
}