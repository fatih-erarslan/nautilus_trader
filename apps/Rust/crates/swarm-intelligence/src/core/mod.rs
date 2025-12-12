//! Core traits, types, and abstractions for swarm intelligence algorithms

pub mod traits;
pub mod types;
pub mod error;
pub mod metrics;
pub mod simd;

// Re-exports
pub use traits::*;
pub use types::*;
pub use error::*;

#[cfg(feature = "metrics")]
pub use metrics::*;

#[cfg(feature = "simd")]
pub use simd::*;