//! Error types for Black Swan detection

use thiserror::Error;

/// Error types for Black Swan detection
#[derive(Error, Debug)]
pub enum BlackSwanError {
    /// Configuration validation error
    #[error("Configuration error: {0}")]
    Configuration(String),
    
    /// Insufficient data for analysis
    #[error("Insufficient data: need at least {required} points, got {actual}")]
    InsufficientData { required: usize, actual: usize },
    
    /// Mathematical computation error
    #[error("Mathematical error: {0}")]
    Mathematical(String),
    
    /// Statistical analysis error
    #[error("Statistical error: {0}")]
    Statistical(String),
    
    /// Memory allocation error
    #[error("Memory allocation error: {0}")]
    Memory(String),
    
    /// GPU computation error
    #[error("GPU error: {0}")]
    Gpu(String),
    
    /// SIMD operation error
    #[error("SIMD error: {0}")]
    Simd(String),
    
    /// Parallel processing error
    #[error("Parallel processing error: {0}")]
    Parallel(String),
    
    /// Cache operation error
    #[error("Cache error: {0}")]
    Cache(String),
    
    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    
    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    
    /// Invalid input data
    #[error("Invalid input: {0}")]
    InvalidInput(String),
    
    /// Numerical overflow or underflow
    #[error("Numerical overflow/underflow: {0}")]
    NumericalOverflow(String),
    
    /// Timeout error
    #[error("Operation timeout: {0}")]
    Timeout(String),
    
    /// Resource exhaustion
    #[error("Resource exhaustion: {0}")]
    ResourceExhaustion(String),
    
    /// Hardware error
    #[error("Hardware error: {0}")]
    Hardware(String),
    
    /// Threading error
    #[error("Threading error: {0}")]
    Threading(String),
    
    /// Performance degradation
    #[error("Performance degradation: {0}")]
    Performance(String),
    
    /// Python FFI error
    #[cfg(feature = "python")]
    #[error("Python FFI error: {0}")]
    Python(String),
    
    /// External library error
    #[error("External library error: {0}")]
    External(String),
}

/// Result type for Black Swan operations
pub type BlackSwanResult<T> = Result<T, BlackSwanError>;

/// Error context for detailed error reporting
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub file: String,
    pub line: u32,
    pub timestamp: u64,
    pub additional_info: Vec<(String, String)>,
}

impl ErrorContext {
    pub fn new(operation: &str, file: &str, line: u32) -> Self {
        Self {
            operation: operation.to_string(),
            file: file.to_string(),
            line,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            additional_info: Vec::new(),
        }
    }
    
    pub fn add_info<K: Into<String>, V: Into<String>>(&mut self, key: K, value: V) {
        self.additional_info.push((key.into(), value.into()));
    }
}

/// Macro for creating error with context
#[macro_export]
macro_rules! error_with_context {
    ($error_type:expr, $operation:expr) => {
        $error_type.with_context(ErrorContext::new($operation, file!(), line!()))
    };
    ($error_type:expr, $operation:expr, $($key:expr => $value:expr),*) => {
        {
            let mut context = ErrorContext::new($operation, file!(), line!());
            $(context.add_info($key, $value);)*
            $error_type.with_context(context)
        }
    };
}

/// Enhanced error with context
#[derive(Debug)]
pub struct ContextualError {
    pub error: BlackSwanError,
    pub context: ErrorContext,
}

impl std::fmt::Display for ContextualError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [{}:{}] at {}", 
               self.error, 
               self.context.file, 
               self.context.line,
               self.context.operation)?;
        
        if !self.context.additional_info.is_empty() {
            write!(f, " (")?;
            for (i, (key, value)) in self.context.additional_info.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}: {}", key, value)?;
            }
            write!(f, ")")?;
        }
        
        Ok(())
    }
}

impl std::error::Error for ContextualError {}

/// Extension trait for adding context to errors
pub trait ErrorContextExt {
    fn with_context(self, context: ErrorContext) -> ContextualError;
}

impl ErrorContextExt for BlackSwanError {
    fn with_context(self, context: ErrorContext) -> ContextualError {
        ContextualError {
            error: self,
            context,
        }
    }
}

/// Validation helpers
pub mod validation {
    use super::*;
    
    /// Validate that a value is within a range
    pub fn validate_range<T: PartialOrd + Copy + std::fmt::Debug>(
        value: T,
        min: T,
        max: T,
        name: &str,
    ) -> BlackSwanResult<()> {
        if value < min || value > max {
            return Err(BlackSwanError::InvalidInput(
                format!("{} must be between {:?} and {:?}, got {:?}", name, min, max, value)
            ));
        }
        Ok(())
    }
    
    /// Validate that a collection has minimum size
    pub fn validate_min_size<T>(
        collection: &[T],
        min_size: usize,
        name: &str,
    ) -> BlackSwanResult<()> {
        if collection.len() < min_size {
            return Err(BlackSwanError::InsufficientData {
                required: min_size,
                actual: collection.len(),
            });
        }
        Ok(())
    }
    
    /// Validate that a value is finite
    pub fn validate_finite(value: f64, name: &str) -> BlackSwanResult<()> {
        if !value.is_finite() {
            return Err(BlackSwanError::InvalidInput(
                format!("{} must be finite, got {}", name, value)
            ));
        }
        Ok(())
    }
    
    /// Validate that a value is positive
    pub fn validate_positive(value: f64, name: &str) -> BlackSwanResult<()> {
        if value <= 0.0 {
            return Err(BlackSwanError::InvalidInput(
                format!("{} must be positive, got {}", name, value)
            ));
        }
        Ok(())
    }
    
    /// Validate that a probability is in [0, 1]
    pub fn validate_probability(value: f64, name: &str) -> BlackSwanResult<()> {
        validate_range(value, 0.0, 1.0, name)
    }
    
    /// Validate that all values in a slice are finite
    pub fn validate_all_finite(values: &[f64], name: &str) -> BlackSwanResult<()> {
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(BlackSwanError::InvalidInput(
                    format!("{}[{}] must be finite, got {}", name, i, value)
                ));
            }
        }
        Ok(())
    }
}

/// Performance monitoring helpers
pub mod performance {
    use super::*;
    use std::time::Instant;
    
    /// Performance monitor for tracking operation times
    pub struct PerformanceMonitor {
        start_time: Instant,
        operation: String,
        target_ns: u64,
    }
    
    impl PerformanceMonitor {
        pub fn new(operation: &str, target_ns: u64) -> Self {
            Self {
                start_time: Instant::now(),
                operation: operation.to_string(),
                target_ns,
            }
        }
        
        pub fn check_performance(&self) -> BlackSwanResult<()> {
            let elapsed_ns = self.start_time.elapsed().as_nanos() as u64;
            if elapsed_ns > self.target_ns {
                return Err(BlackSwanError::Performance(
                    format!("Operation '{}' took {}ns, target was {}ns", 
                           self.operation, elapsed_ns, self.target_ns)
                ));
            }
            Ok(())
        }
        
        pub fn elapsed_ns(&self) -> u64 {
            self.start_time.elapsed().as_nanos() as u64
        }
    }
    
    impl Drop for PerformanceMonitor {
        fn drop(&mut self) {
            let elapsed_ns = self.elapsed_ns();
            if elapsed_ns > self.target_ns {
                log::warn!("Performance target missed for '{}': {}ns > {}ns", 
                          self.operation, elapsed_ns, self.target_ns);
            } else {
                log::debug!("Performance target met for '{}': {}ns <= {}ns", 
                           self.operation, elapsed_ns, self.target_ns);
            }
        }
    }
}

/// Memory monitoring helpers
pub mod memory {
    use super::*;
    
    /// Memory usage monitor
    pub struct MemoryMonitor {
        initial_usage: usize,
        max_allowed: usize,
        operation: String,
    }
    
    impl MemoryMonitor {
        pub fn new(operation: &str, max_allowed: usize) -> Self {
            Self {
                initial_usage: get_memory_usage(),
                max_allowed,
                operation: operation.to_string(),
            }
        }
        
        pub fn check_memory(&self) -> BlackSwanResult<()> {
            let current_usage = get_memory_usage();
            let delta = current_usage.saturating_sub(self.initial_usage);
            
            if delta > self.max_allowed {
                return Err(BlackSwanError::Memory(
                    format!("Operation '{}' used {}bytes, limit was {}bytes", 
                           self.operation, delta, self.max_allowed)
                ));
            }
            Ok(())
        }
    }
    
    /// Get current memory usage (simplified implementation)
    fn get_memory_usage() -> usize {
        // In a real implementation, this would use system APIs
        // For now, return a placeholder
        0
    }
}

/// Result extensions for chaining operations
pub trait ResultExt<T> {
    fn with_operation(self, operation: &str) -> BlackSwanResult<T>;
    fn with_info<K: Into<String>, V: Into<String>>(self, key: K, value: V) -> BlackSwanResult<T>;
}

impl<T> ResultExt<T> for BlackSwanResult<T> {
    fn with_operation(self, operation: &str) -> BlackSwanResult<T> {
        self.map_err(|e| {
            BlackSwanError::External(format!("Failed in {}: {}", operation, e))
        })
    }
    
    fn with_info<K: Into<String>, V: Into<String>>(self, key: K, value: V) -> BlackSwanResult<T> {
        self.map_err(|e| {
            BlackSwanError::External(format!("{} ({}={})", e, key.into(), value.into()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::validation::*;
    use super::performance::*;
    
    #[test]
    fn test_validation_range() {
        assert!(validate_range(0.5, 0.0, 1.0, "test").is_ok());
        assert!(validate_range(1.5, 0.0, 1.0, "test").is_err());
        assert!(validate_range(-0.5, 0.0, 1.0, "test").is_err());
    }
    
    #[test]
    fn test_validation_min_size() {
        let data = vec![1, 2, 3];
        assert!(validate_min_size(&data, 3, "test").is_ok());
        assert!(validate_min_size(&data, 5, "test").is_err());
    }
    
    #[test]
    fn test_validation_finite() {
        assert!(validate_finite(1.0, "test").is_ok());
        assert!(validate_finite(f64::NAN, "test").is_err());
        assert!(validate_finite(f64::INFINITY, "test").is_err());
    }
    
    #[test]
    fn test_validation_positive() {
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(0.0, "test").is_err());
        assert!(validate_positive(-1.0, "test").is_err());
    }
    
    #[test]
    fn test_validation_probability() {
        assert!(validate_probability(0.5, "test").is_ok());
        assert!(validate_probability(0.0, "test").is_ok());
        assert!(validate_probability(1.0, "test").is_ok());
        assert!(validate_probability(1.5, "test").is_err());
        assert!(validate_probability(-0.5, "test").is_err());
    }
    
    #[test]
    fn test_validation_all_finite() {
        let finite_data = vec![1.0, 2.0, 3.0];
        let infinite_data = vec![1.0, f64::INFINITY, 3.0];
        
        assert!(validate_all_finite(&finite_data, "test").is_ok());
        assert!(validate_all_finite(&infinite_data, "test").is_err());
    }
    
    #[test]
    fn test_performance_monitor() {
        let monitor = PerformanceMonitor::new("test", 1_000_000); // 1ms target
        std::thread::sleep(std::time::Duration::from_nanos(100));
        assert!(monitor.check_performance().is_ok());
        
        let elapsed = monitor.elapsed_ns();
        assert!(elapsed > 0);
        assert!(elapsed < 1_000_000); // Should be well under 1ms
    }
    
    #[test]
    fn test_error_context() {
        let mut context = ErrorContext::new("test_operation", "test.rs", 42);
        context.add_info("key", "value");
        
        let error = BlackSwanError::Configuration("test error".to_string());
        let contextual = error.with_context(context);
        
        let error_string = format!("{}", contextual);
        assert!(error_string.contains("test_operation"));
        assert!(error_string.contains("test.rs"));
        assert!(error_string.contains("42"));
        assert!(error_string.contains("key: value"));
    }
    
    #[test]
    fn test_result_extensions() {
        let result: BlackSwanResult<i32> = Ok(42);
        assert!(result.with_operation("test").is_ok());
        
        let error_result: BlackSwanResult<i32> = Err(BlackSwanError::Configuration("test".to_string()));
        assert!(error_result.with_operation("test").is_err());
        assert!(error_result.with_info("key", "value").is_err());
    }
}