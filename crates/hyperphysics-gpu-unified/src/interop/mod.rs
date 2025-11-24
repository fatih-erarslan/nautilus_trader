//! Interoperability layer for external frameworks

#[cfg(feature = "burn")]
pub mod burn;

/// Re-export interop types
#[cfg(feature = "burn")]
pub use burn::{BurnBridge, BurnConfig};
