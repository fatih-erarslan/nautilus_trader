//! SOA (Service-Oriented Architecture) module for PADS
//!
//! This module provides SOA patterns and memory optimization
//! for distributed trading systems.

pub mod soa_integration;
pub mod soa_memory_optimization;
pub mod soa_neural_forecaster;

// Re-exports
pub use soa_integration::*;
pub use soa_memory_optimization::*;
pub use soa_neural_forecaster::*;