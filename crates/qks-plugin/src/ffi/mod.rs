//! FFI module root - Foreign Function Interface for cross-language integration
//!
//! This module provides C-compatible FFI bindings for:
//! - Plugin lifecycle (create, destroy, start, stop)
//! - Cognitive loop execution
//! - State queries (Phi, energy, layers)
//! - Configuration management
//!
//! ## Memory Safety
//!
//! All FFI functions follow these safety guarantees:
//! - Opaque handles prevent use-after-free
//! - Null pointer checks on all inputs
//! - Thread-safe handle registry
//! - Automatic cleanup on handle destruction
//! - No undefined behavior in FFI boundary

pub mod types;
pub mod c_api;
pub mod callbacks;

// Re-export main types for convenience
pub use types::{
    QksHandle,
    QksConfigHandle,
    QksStateHandle,
    QksResult as FfiResult,
};

pub use c_api::{
    qks_create,
    qks_destroy,
    qks_initialize,
    qks_start,
    qks_stop,
    qks_process,
    qks_get_phi,
    qks_get_state,
    qks_get_error_message,
};

pub use callbacks::{
    QksCallback,
    QksCallbackContext,
    qks_set_callback,
};
