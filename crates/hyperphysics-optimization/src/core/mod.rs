//! Core optimization primitives with formal verification support.
//!
//! This module provides the foundational types for all optimization algorithms:
//! - `Population`: Thread-safe population management with parallel evaluation
//! - `Individual`: Solution representation with fitness caching
//! - `Bounds`: Box constraints with violation detection
//! - `ObjectiveFunction`: Trait for optimization targets

mod population;
mod solution;
mod bounds;
mod objective;
mod config;

pub use population::*;
pub use solution::*;
pub use bounds::*;
pub use objective::*;
pub use config::*;
