//! Lorentz/Hyperboloid Model for Hyperbolic Space
//!
//! This crate implements the Lorentz (hyperboloid) model of hyperbolic geometry
//! with f64 precision and SIMD-optimized Minkowski operations.
//!
//! # Mathematical Foundation
//!
//! The Lorentz model represents hyperbolic n-space as the upper sheet of a
//! two-sheeted hyperboloid in (n+1)-dimensional Minkowski space:
//!
//! ```text
//! H^n = { x ∈ R^{n+1} : ⟨x,x⟩_L = -1, x_0 > 0 }
//! ```
//!
//! where the Minkowski inner product is:
//!
//! ```text
//! ⟨x,y⟩_L = -x_0·y_0 + x_1·y_1 + ... + x_n·y_n
//! ```
//!
//! # References
//!
//! - Cannon et al. (1997) "Hyperbolic Geometry"
//! - Nickel & Kiela (2017) "Poincaré Embeddings for Learning Hierarchical Representations"
//! - Ungar (2001) "Hyperbolic Trigonometry and Its Application"

#![deny(missing_docs)]
#![deny(unsafe_code)]

mod error;
mod lorentz;
mod minkowski;
mod conversion;

pub use error::{LorentzError, Result};
pub use lorentz::{LorentzPoint, LorentzModel};
pub use minkowski::{MinkowskiOps, SimdMinkowski};
pub use conversion::{
    poincare_to_lorentz, lorentz_to_poincare,
    poincare_point_to_lorentz,
    batch_poincare_to_lorentz, batch_lorentz_to_poincare,
};

/// Epsilon for numerical stability
pub const EPSILON: f64 = 1e-12;

/// Default curvature (K = -1 for standard hyperbolic space)
pub const DEFAULT_CURVATURE: f64 = -1.0;
