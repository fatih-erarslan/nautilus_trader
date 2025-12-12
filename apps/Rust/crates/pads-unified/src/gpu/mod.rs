//! GPU acceleration module for PADS
//!
//! This module provides GPU-accelerated computation capabilities
//! for high-performance trading algorithms.

pub mod enterprise_gpu_validation;
pub mod gpu_acceleration_comprehensive_test_suite;
pub mod gpu_acceleration_tdd_framework;
pub mod gpu_crate_integration_tests;
pub mod gpu_pipeline_acceleration;

// Re-exports
pub use enterprise_gpu_validation::*;
pub use gpu_pipeline_acceleration::*;