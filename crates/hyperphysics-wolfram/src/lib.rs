//! HyperPhysics Wolfram Bridge
//!
//! Enterprise-grade integration with Wolfram Language for:
//! - Scientific validation and verification
//! - Formal mathematical proofs
//! - Computer Algebra System (CAS) computations
//! - Algorithm correctness verification
//! - Research and literature integration
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    HyperPhysics Crates                      │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  hyperphysics-wolfram                       │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
//! │  │  Validation │  │  Research   │  │  HyperPhysics       │ │
//! │  │  Engine     │  │  Module     │  │  Integration        │ │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//!                              │
//!                              ▼
//! ┌─────────────────────────────────────────────────────────────┐
//! │                   WolframScript.app                         │
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
//! │  │  Symbolic   │  │  Numerical  │  │  Code Assistant     │ │
//! │  │  Compute    │  │  Analysis   │  │  (Opus 4.5)         │ │
//! │  └─────────────┘  └─────────────┘  └─────────────────────┘ │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use hyperphysics_wolfram::{WolframBridge, ValidationResult};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Initialize bridge
//!     let bridge = WolframBridge::new()?;
//!
//!     // Validate hyperbolic distance formula
//!     let validation = bridge.validate_hyperbolic_distance(
//!         [0.3, 0.2],
//!         [-0.1, 0.4]
//!     ).await?;
//!
//!     assert!(validation.is_valid);
//!     Ok(())
//! }
//! ```

#![deny(clippy::all)]
#![warn(missing_docs)]

mod discovery;
mod evaluator;
mod hyperphysics;
mod research;
mod types;
mod validation;

pub use discovery::*;
pub use evaluator::*;
pub use hyperphysics::*;
pub use research::*;
pub use types::*;
pub use validation::*;

#[cfg(feature = "napi-bindings")]
mod napi_bindings;

#[cfg(feature = "napi-bindings")]
pub use napi_bindings::*;

/// Re-export for convenience
pub mod prelude {
    pub use crate::discovery::{discover_installations, get_default_installation};
    pub use crate::evaluator::{evaluate_code, WolframEvaluator};
    pub use crate::hyperphysics::HyperPhysicsWolfram;
    pub use crate::research::ResearchEngine;
    pub use crate::types::*;
    pub use crate::validation::ValidationEngine;
}
