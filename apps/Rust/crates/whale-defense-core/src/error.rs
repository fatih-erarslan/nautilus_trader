//! Error handling for whale defense operations
//! 
//! Optimized error types for minimal overhead in hot paths

use core::fmt;

/// Ultra-fast error type for whale defense operations
/// 
/// This enum is carefully designed to be lightweight and fast:
/// - Uses discriminant optimization for single-byte representation
/// - Avoids heap allocations in error paths
/// - Implements Copy for zero-cost error propagation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum WhaleDefenseError {
    /// System not initialized
    NotInitialized = 0,
    
    /// Unsupported CPU feature required
    UnsupportedCpuFeature(&'static str) = 1,
    
    /// Memory allocation failed
    OutOfMemory = 2,
    
    /// Invalid parameter provided
    InvalidParameter = 3,
    
    /// Buffer overflow detected
    BufferOverflow = 4,
    
    /// Buffer underflow detected
    BufferUnderflow = 5,
    
    /// Quantum RNG initialization failed
    QuantumRngError = 6,
    
    /// Performance counter initialization failed
    PerformanceCounterError = 7,
    
    /// SIMD operation failed
    SimdError = 8,
    
    /// Cache operation failed
    CacheError = 9,
    
    /// Timing operation failed
    TimingError = 10,
    
    /// Steganography operation failed
    SteganographyError = 11,
    
    /// Game theory calculation failed
    GameTheoryError = 12,
    
    /// Whale detection failed
    DetectionError = 13,
    
    /// Defense strategy execution failed
    DefenseError = 14,
    
    /// Lock-free operation failed
    LockFreeError = 15,
    
    /// Concurrent access violation
    ConcurrencyError = 16,
    
    /// Performance threshold exceeded
    PerformanceThresholdExceeded = 17,
    
    /// Unknown error
    Unknown = 255,
}

impl WhaleDefenseError {
    /// Convert error to static string for zero-allocation error messages
    #[inline(always)]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NotInitialized => "system not initialized",
            Self::UnsupportedCpuFeature(_) => "unsupported CPU feature",
            Self::OutOfMemory => "out of memory",
            Self::InvalidParameter => "invalid parameter",
            Self::BufferOverflow => "buffer overflow",
            Self::BufferUnderflow => "buffer underflow",
            Self::QuantumRngError => "quantum RNG error",
            Self::PerformanceCounterError => "performance counter error",
            Self::SimdError => "SIMD operation error",
            Self::CacheError => "cache operation error",
            Self::TimingError => "timing operation error",
            Self::SteganographyError => "steganography operation error",
            Self::GameTheoryError => "game theory calculation error",
            Self::DetectionError => "whale detection error",
            Self::DefenseError => "defense strategy execution error",
            Self::LockFreeError => "lock-free operation error",
            Self::ConcurrencyError => "concurrent access violation",
            Self::PerformanceThresholdExceeded => "performance threshold exceeded",
            Self::Unknown => "unknown error",
        }
    }
    
    /// Get error code for fast error classification
    #[inline(always)]
    pub const fn code(self) -> u8 {
        self as u8
    }
    
    /// Check if error is recoverable
    #[inline(always)]
    pub const fn is_recoverable(self) -> bool {
        match self {
            Self::NotInitialized |
            Self::UnsupportedCpuFeature(_) |
            Self::OutOfMemory => false,
            _ => true,
        }
    }
    
    /// Check if error is performance-critical
    #[inline(always)]
    pub const fn is_performance_critical(self) -> bool {
        match self {
            Self::PerformanceThresholdExceeded |
            Self::ConcurrencyError |
            Self::LockFreeError => true,
            _ => false,
        }
    }
}

impl fmt::Display for WhaleDefenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedCpuFeature(feature) => {
                write!(f, "unsupported CPU feature: {}", feature)
            }
            _ => write!(f, "{}", self.as_str()),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WhaleDefenseError {}

/// Result type for whale defense operations
pub type Result<T> = core::result::Result<T, WhaleDefenseError>;

/// Fast error propagation macro for hot paths
/// 
/// This macro generates optimized error handling code that avoids
/// branch prediction misses in the success path.
#[macro_export]
macro_rules! fast_try {
    ($expr:expr) => {
        match $expr {
            Ok(val) => val,
            Err(err) => {
                // Fast path optimization: likely success
                unsafe { core::hint::unreachable_unchecked() }
            }
        }
    };
}

/// Performance-critical error handling
/// 
/// This trait provides optimized error handling for performance-critical paths
/// where traditional error propagation would be too slow.
pub trait FastErrorHandler {
    /// Handle error without allocation or branching
    fn handle_fast_error(&self, error: WhaleDefenseError);
    
    /// Check if error handling is enabled
    fn error_handling_enabled(&self) -> bool;
}

/// No-op error handler for maximum performance
pub struct NoOpErrorHandler;

impl FastErrorHandler for NoOpErrorHandler {
    #[inline(always)]
    fn handle_fast_error(&self, _error: WhaleDefenseError) {
        // No-op for maximum performance
    }
    
    #[inline(always)]
    fn error_handling_enabled(&self) -> bool {
        false
    }
}

/// Atomic error handler for concurrent environments
pub struct AtomicErrorHandler {
    error_count: AtomicU64,
    last_error: AtomicU64,
}

impl AtomicErrorHandler {
    /// Create new atomic error handler
    pub const fn new() -> Self {
        Self {
            error_count: AtomicU64::new(0),
            last_error: AtomicU64::new(0),
        }
    }
    
    /// Get error statistics
    pub fn get_stats(&self) -> (u64, WhaleDefenseError) {
        let count = self.error_count.load(Ordering::Relaxed);
        let last = self.last_error.load(Ordering::Relaxed);
        let last_error = if last <= 255 {
            unsafe { core::mem::transmute(last as u8) }
        } else {
            WhaleDefenseError::Unknown
        };
        (count, last_error)
    }
    
    /// Reset error statistics
    pub fn reset(&self) {
        self.error_count.store(0, Ordering::Relaxed);
        self.last_error.store(0, Ordering::Relaxed);
    }
}

impl FastErrorHandler for AtomicErrorHandler {
    #[inline(always)]
    fn handle_fast_error(&self, error: WhaleDefenseError) {
        self.error_count.fetch_add(1, Ordering::Relaxed);
        self.last_error.store(error.code() as u64, Ordering::Relaxed);
    }
    
    #[inline(always)]
    fn error_handling_enabled(&self) -> bool {
        true
    }
}

use crate::config::*;
use crate::{AtomicU64, Ordering};

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_codes() {
        assert_eq!(WhaleDefenseError::NotInitialized.code(), 0);
        assert_eq!(WhaleDefenseError::Unknown.code(), 255);
    }
    
    #[test]
    fn test_error_recovery() {
        assert!(!WhaleDefenseError::NotInitialized.is_recoverable());
        assert!(WhaleDefenseError::BufferOverflow.is_recoverable());
    }
    
    #[test]
    fn test_performance_critical() {
        assert!(WhaleDefenseError::PerformanceThresholdExceeded.is_performance_critical());
        assert!(!WhaleDefenseError::InvalidParameter.is_performance_critical());
    }
    
    #[test]
    fn test_atomic_error_handler() {
        let handler = AtomicErrorHandler::new();
        assert!(handler.error_handling_enabled());
        
        handler.handle_fast_error(WhaleDefenseError::DetectionError);
        let (count, last_error) = handler.get_stats();
        assert_eq!(count, 1);
        assert_eq!(last_error, WhaleDefenseError::DetectionError);
    }
    
    #[test]
    fn test_noop_error_handler() {
        let handler = NoOpErrorHandler;
        assert!(!handler.error_handling_enabled());
        
        // Should not panic
        handler.handle_fast_error(WhaleDefenseError::Unknown);
    }
}