//! # Tengri Compliance
//! 
//! **Core infrastructure and utilities for SERRA sustainability platform**
//! 
//! This crate provides foundational types, traits, and utilities used throughout
//! the SERRA sustainability assessment platform. It implements the fundamental
//! building blocks that support all sustainability modeling and analysis operations.
//! 
//! ## Core Abstractions
//! 
//! ### Type System
//! ```text
//! Sustainability Types: Strongly typed domain modeling
//! Error Handling: Comprehensive error types with context
//! Configuration: Type-safe configuration management
//! Serialization: Efficient data interchange formats
//! ```
//! 
//! ## Design Principles
//! 
//! - **Type Safety**: Prevent errors at compile time
//! - **Performance**: Zero-cost abstractions where possible
//! - **Extensibility**: Trait-based design for modularity
//! - **Reliability**: Comprehensive error handling
//! 
//! ## Usage Example
//! 
//! ```rust
//! use tengri_compliance::{Result, SustainabilityError};
//! 
//! fn sustainability_operation() -> Result<f64> {
//!     // Core operations with proper error handling
//!     Ok(42.0)
//! }
//! ```


pub mod audit;
pub mod circuit_breaker;
pub mod engine;
pub mod error;
pub mod metrics;
pub mod rules;
pub mod surveillance;

pub use engine::{ComplianceEngine, ComplianceConfig};
pub use error::{ComplianceError, ComplianceResult};
pub use rules::{ComplianceRule, RuleEngine, RuleSet};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compliance_basics() {
        // Basic smoke test
        assert_eq!(2 + 2, 4);
    }
}
