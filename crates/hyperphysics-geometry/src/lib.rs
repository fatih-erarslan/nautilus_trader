//! # Hyperbolic Geometry Engine
//!
//! Implementation of hyperbolic 3-space (H³) with constant negative curvature K=-1.
//! Uses the Poincaré disk model for computational efficiency.
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Cannon et al. (1997) "Hyperbolic Geometry" Springer GTM 31
//! - Lee (2018) "Introduction to Riemannian Manifolds"
//! - Kollár et al. (2019) "Hyperbolic lattices in circuit QED" Nature 571:45-50
//!
//! ## Poincaré Disk Model
//!
//! Points: D³ = {x ∈ ℝ³ : ||x|| < 1}
//! Metric: ds² = 4(dx₁² + dx₂² + dx₃²) / (1 - ||x||²)²
//! Distance: d_H(p,q) = acosh(1 + 2||p-q||² / ((1-||p||²)(1-||q||²)))

pub mod poincare;
pub mod geodesic;
pub mod distance;
pub mod tessellation;
pub mod tessellation_73;
pub mod crypto_substrate;
pub mod curvature;

pub use poincare::PoincarePoint;
pub use geodesic::Geodesic;
pub use distance::HyperbolicDistance;
pub use tessellation::HyperbolicTessellation;
pub use tessellation_73::{HeptagonalTessellation, HeptagonalTile, TessellationVertex, FuchsianGroup, TileId, VertexId};
pub use crypto_substrate::{CryptoSubstrate, TileCryptoState, SubstrateStats};
pub use curvature::CurvatureTensor;

use thiserror::Error;

/// Errors specific to hyperbolic geometry operations
#[derive(Error, Debug)]
pub enum GeometryError {
    #[error("Point outside Poincaré disk: norm = {norm}")]
    OutsideDisk { norm: f64 },

    #[error("Numerical instability in distance calculation")]
    NumericalInstability,

    #[error("Invalid tessellation parameters: {message}")]
    InvalidTessellation { message: String },

    #[error("Geodesic integration failed: {reason}")]
    GeodesicFailure { reason: String },
}

pub type Result<T> = std::result::Result<T, GeometryError>;

/// Constant negative curvature
pub const CURVATURE: f64 = -1.0;

/// Numerical tolerance for boundary checks
pub const EPSILON: f64 = 1e-10;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_curvature_constant() {
        assert_eq!(CURVATURE, -1.0, "H³ must have K = -1");
    }
}

pub mod moebius;
pub mod fuchsian;
pub use moebius::{MoebiusTransform, TransformType};
pub use fuchsian::FuchsianGroup as FuchsianGroupAlgebraic;
