//! Error types with FFI-safe error codes
//!
//! This module provides comprehensive error handling for the QKS plugin with:
//! - FFI-safe error codes (i32)
//! - Thread-safe error message storage
//! - Conversion from internal error types
//! - Scientific error categorization

use std::ffi::{CString, NulError};
use std::fmt;

/// FFI-safe error codes
///
/// All error codes are negative integers to distinguish from success (0)
/// and allow positive values for valid handles/counts.
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QksErrorCode {
    /// Success (no error)
    Success = 0,

    /// Generic error
    GenericError = -1,

    /// Invalid handle
    InvalidHandle = -2,

    /// Null pointer passed to FFI
    NullPointer = -3,

    /// Invalid configuration
    InvalidConfig = -4,

    /// Layer not initialized
    LayerNotInitialized = -5,

    /// Layer initialization failed
    LayerInitFailed = -6,

    /// Memory allocation failed
    OutOfMemory = -7,

    /// Thread panic or internal error
    InternalError = -8,

    /// Invalid layer ID
    InvalidLayer = -9,

    /// Resource exhausted (energy, compute, etc.)
    ResourceExhausted = -10,

    /// Homeostasis violation
    HomeostasisViolation = -11,

    /// Consciousness threshold not met
    ConsciousnessError = -12,

    /// Metacognition error
    MetacognitionError = -13,

    /// Integration error
    IntegrationError = -14,

    /// FFI string conversion error
    StringConversionError = -15,

    /// Buffer too small
    BufferTooSmall = -16,

    /// Operation timeout
    Timeout = -17,

    /// Concurrent access violation
    ConcurrencyError = -18,

    /// Serialization/deserialization error
    SerializationError = -19,
}

impl QksErrorCode {
    /// Convert error code to human-readable string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Success => "Success",
            Self::GenericError => "Generic error",
            Self::InvalidHandle => "Invalid handle",
            Self::NullPointer => "Null pointer",
            Self::InvalidConfig => "Invalid configuration",
            Self::LayerNotInitialized => "Layer not initialized",
            Self::LayerInitFailed => "Layer initialization failed",
            Self::OutOfMemory => "Out of memory",
            Self::InternalError => "Internal error",
            Self::InvalidLayer => "Invalid layer ID",
            Self::ResourceExhausted => "Resource exhausted",
            Self::HomeostasisViolation => "Homeostasis violation",
            Self::ConsciousnessError => "Consciousness error",
            Self::MetacognitionError => "Metacognition error",
            Self::IntegrationError => "Integration error",
            Self::StringConversionError => "String conversion error",
            Self::BufferTooSmall => "Buffer too small",
            Self::Timeout => "Operation timeout",
            Self::ConcurrencyError => "Concurrency error",
            Self::SerializationError => "Serialization error",
        }
    }

    /// Check if error code represents success
    #[inline]
    pub fn is_success(&self) -> bool {
        matches!(self, Self::Success)
    }

    /// Check if error code represents an error
    #[inline]
    pub fn is_error(&self) -> bool {
        !self.is_success()
    }
}

impl From<i32> for QksErrorCode {
    fn from(code: i32) -> Self {
        match code {
            0 => Self::Success,
            -1 => Self::GenericError,
            -2 => Self::InvalidHandle,
            -3 => Self::NullPointer,
            -4 => Self::InvalidConfig,
            -5 => Self::LayerNotInitialized,
            -6 => Self::LayerInitFailed,
            -7 => Self::OutOfMemory,
            -8 => Self::InternalError,
            -9 => Self::InvalidLayer,
            -10 => Self::ResourceExhausted,
            -11 => Self::HomeostasisViolation,
            -12 => Self::ConsciousnessError,
            -13 => Self::MetacognitionError,
            -14 => Self::IntegrationError,
            -15 => Self::StringConversionError,
            -16 => Self::BufferTooSmall,
            -17 => Self::Timeout,
            -18 => Self::ConcurrencyError,
            -19 => Self::SerializationError,
            _ => Self::GenericError,
        }
    }
}

/// Rust error type for QKS operations
#[derive(Debug)]
pub enum QksError {
    /// Generic error with message
    Generic(String),

    /// Invalid handle
    InvalidHandle,

    /// Null pointer
    NullPointer,

    /// Invalid configuration
    InvalidConfig(String),

    /// Layer not initialized
    LayerNotInitialized(u8),

    /// Layer initialization failed
    LayerInitFailed(u8, String),

    /// Out of memory
    OutOfMemory,

    /// Internal error (panic, etc.)
    Internal(String),

    /// Invalid layer ID
    InvalidLayer(u8),

    /// Resource exhausted
    ResourceExhausted(String),

    /// Homeostasis violation
    HomeostasisViolation(String),

    /// Consciousness error
    Consciousness(String),

    /// Metacognition error
    Metacognition(String),

    /// Integration error
    Integration(String),

    /// String conversion error
    StringConversion(NulError),

    /// Buffer too small
    BufferTooSmall { required: usize, provided: usize },

    /// Timeout
    Timeout,

    /// Concurrency error
    Concurrency(String),

    /// Serialization error
    Serialization(String),

    /// Wrapped error from quantum-knowledge-core
    Core(quantum_knowledge_core::QksError),
}

impl fmt::Display for QksError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Generic(msg) => write!(f, "Generic error: {}", msg),
            Self::InvalidHandle => write!(f, "Invalid handle"),
            Self::NullPointer => write!(f, "Null pointer"),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::LayerNotInitialized(layer) => write!(f, "Layer {} not initialized", layer),
            Self::LayerInitFailed(layer, msg) => write!(f, "Layer {} initialization failed: {}", layer, msg),
            Self::OutOfMemory => write!(f, "Out of memory"),
            Self::Internal(msg) => write!(f, "Internal error: {}", msg),
            Self::InvalidLayer(layer) => write!(f, "Invalid layer ID: {}", layer),
            Self::ResourceExhausted(resource) => write!(f, "Resource exhausted: {}", resource),
            Self::HomeostasisViolation(msg) => write!(f, "Homeostasis violation: {}", msg),
            Self::Consciousness(msg) => write!(f, "Consciousness error: {}", msg),
            Self::Metacognition(msg) => write!(f, "Metacognition error: {}", msg),
            Self::Integration(msg) => write!(f, "Integration error: {}", msg),
            Self::StringConversion(e) => write!(f, "String conversion error: {}", e),
            Self::BufferTooSmall { required, provided } => {
                write!(f, "Buffer too small: required {}, provided {}", required, provided)
            }
            Self::Timeout => write!(f, "Operation timeout"),
            Self::Concurrency(msg) => write!(f, "Concurrency error: {}", msg),
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::Core(e) => write!(f, "Core error: {}", e),
        }
    }
}

impl std::error::Error for QksError {}

impl From<QksError> for QksErrorCode {
    fn from(error: QksError) -> Self {
        match error {
            QksError::Generic(_) => Self::GenericError,
            QksError::InvalidHandle => Self::InvalidHandle,
            QksError::NullPointer => Self::NullPointer,
            QksError::InvalidConfig(_) => Self::InvalidConfig,
            QksError::LayerNotInitialized(_) => Self::LayerNotInitialized,
            QksError::LayerInitFailed(_, _) => Self::LayerInitFailed,
            QksError::OutOfMemory => Self::OutOfMemory,
            QksError::Internal(_) => Self::InternalError,
            QksError::InvalidLayer(_) => Self::InvalidLayer,
            QksError::ResourceExhausted(_) => Self::ResourceExhausted,
            QksError::HomeostasisViolation(_) => Self::HomeostasisViolation,
            QksError::Consciousness(_) => Self::ConsciousnessError,
            QksError::Metacognition(_) => Self::MetacognitionError,
            QksError::Integration(_) => Self::IntegrationError,
            QksError::StringConversion(_) => Self::StringConversionError,
            QksError::BufferTooSmall { .. } => Self::BufferTooSmall,
            QksError::Timeout => Self::Timeout,
            QksError::Concurrency(_) => Self::ConcurrencyError,
            QksError::Serialization(_) => Self::SerializationError,
            QksError::Core(_) => Self::GenericError,
        }
    }
}

impl From<quantum_knowledge_core::QksError> for QksError {
    fn from(error: quantum_knowledge_core::QksError) -> Self {
        Self::Core(error)
    }
}

impl From<NulError> for QksError {
    fn from(error: NulError) -> Self {
        Self::StringConversion(error)
    }
}

/// Result type for QKS operations
pub type QksResult<T> = Result<T, QksError>;

/// FFI-safe result type (i32 error code, with out-parameter for value)
pub type QksFfiResult = i32;

/// Convert Rust Result to FFI result code
#[inline]
pub fn to_ffi_result<T>(result: QksResult<T>) -> QksFfiResult {
    match result {
        Ok(_) => QksErrorCode::Success as i32,
        Err(e) => QksErrorCode::from(e) as i32,
    }
}

/// Thread-local error message storage for FFI
thread_local! {
    static LAST_ERROR: std::cell::RefCell<Option<String>> = std::cell::RefCell::new(None);
}

/// Store error message for later retrieval via FFI
pub fn set_last_error(error: &QksError) {
    let msg = error.to_string();
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = Some(msg);
    });
}

/// Get last error message (for FFI)
pub fn get_last_error() -> Option<String> {
    LAST_ERROR.with(|cell| cell.borrow().clone())
}

/// Clear last error message
pub fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_conversion() {
        assert_eq!(QksErrorCode::Success as i32, 0);
        assert_eq!(QksErrorCode::InvalidHandle as i32, -2);

        let code: QksErrorCode = (-2).into();
        assert_eq!(code, QksErrorCode::InvalidHandle);
    }

    #[test]
    fn test_error_to_code() {
        let error = QksError::InvalidHandle;
        let code = QksErrorCode::from(error);
        assert_eq!(code, QksErrorCode::InvalidHandle);
    }

    #[test]
    fn test_error_message_storage() {
        clear_last_error();
        assert!(get_last_error().is_none());

        let error = QksError::Generic("test error".to_string());
        set_last_error(&error);

        let msg = get_last_error();
        assert!(msg.is_some());
        assert!(msg.unwrap().contains("test error"));

        clear_last_error();
        assert!(get_last_error().is_none());
    }
}
