//! # Prospect Theory RS
//! 
//! High-precision financial prospect theory implementation with PyO3 bindings.
//! 
//! This crate implements Kahneman-Tversky prospect theory with financial-grade
//! precision for use in trading algorithms and risk assessment.

pub mod value_function;
pub mod probability_weighting;
pub mod errors;
pub mod utils;

// Core behavioral economics features only
pub mod types;
pub mod behavioral;
// Commenting out complex decision modules for now
// pub mod decision_context;
// pub mod decision_engine;
// Commenting out complex modules until basic functionality works
// pub mod mental_accounting;
// pub mod framing;
// pub mod ambiguity;
// pub mod reference_point;
// pub mod metrics;
// pub mod cache;

#[cfg(feature = "simd")]
pub mod simd;

#[cfg(feature = "high-performance")]
pub mod performance;

pub mod performance_simple;

#[cfg(feature = "python-bindings")]
pub mod python_bindings;

pub use value_function::{ValueFunction, ValueFunctionParams};
pub use probability_weighting::{ProbabilityWeighting, WeightingParams};
pub use errors::{ProspectTheoryError, Result};

/// Financial precision tolerance (1e-10)
pub const FINANCIAL_PRECISION: f64 = 1e-10;

/// Maximum allowed input value for safety
pub const MAX_INPUT_VALUE: f64 = 1e15;

/// Minimum allowed input value for safety  
pub const MIN_INPUT_VALUE: f64 = -1e15;

#[cfg(feature = "python-bindings")]
use pyo3::prelude::*;

#[cfg(feature = "python-bindings")]
#[pymodule]
fn prospect_theory_rs(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<python_bindings::PyValueFunction>()?;
    m.add_class::<python_bindings::PyProbabilityWeighting>()?;
    m.add_class::<python_bindings::PyValueFunctionParams>()?;
    m.add_class::<python_bindings::PyWeightingParams>()?;
    m.add_class::<python_bindings::PyProspectTheory>()?;
    Ok(())
}