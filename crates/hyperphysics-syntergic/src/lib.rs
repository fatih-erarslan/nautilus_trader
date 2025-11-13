//! # Syntergic Field Theory Implementation
//!
//! Implementation of Grinberg-Zylberbaum's syntergic field theory on hyperbolic space.
//! This provides the first computational model of non-local consciousness correlations.
//!
//! ## Theoretical Foundation
//!
//! Based on peer-reviewed research:
//! - Grinberg-Zylberbaum et al. (1994) "Human communication and the electrophysiological activity of the brain" Subtle Energies 3(3):25-43
//! - Grinberg-Zylberbaum (1995) "Syntergic Theory" INPEC
//! - Pizzi et al. (2004) "Non-local correlations between separated neural networks" NeuroQuantology 2(1)
//!
//! ## Mathematical Framework
//!
//! The syntergic field Ψ(x,t) satisfies a wave-like equation on H³:
//!
//! ∂²Ψ/∂t² = c²∇²_H Ψ + κΨ
//!
//! Green's function for elliptic operator on H³ with K=-1:
//!
//! G(x,y) = (κ·exp(-κd(x,y))) / (4π·sinh(d(x,y)))
//!
//! where κ = √(-K) = 1 and d(x,y) is hyperbolic distance.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use hyperphysics_syntergic::{HyperbolicGreenFunction, SyntergicField};
//! use hyperphysics_geometry::PoincarePoint;
//! use nalgebra::Vector3;
//!
//! // Create Green's function
//! let green = HyperbolicGreenFunction::new(1.0);
//!
//! // Evaluate between two points
//! let p1 = PoincarePoint::new(Vector3::new(0.1, 0.0, 0.0)).unwrap();
//! let p2 = PoincarePoint::new(Vector3::new(0.0, 0.2, 0.0)).unwrap();
//! let g_value = green.evaluate(&p1, &p2);
//! ```

pub mod green_function;
pub mod neuronal_field;
pub mod syntergic_field;

pub use green_function::HyperbolicGreenFunction;
pub use neuronal_field::NeuronalField;
pub use syntergic_field::{SyntergicField, SyntergicMetrics};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SyntergicError {
    #[error("Geometry error: {0}")]
    Geometry(#[from] hyperphysics_geometry::GeometryError),

    #[error("pBit error: {0}")]
    PBit(#[from] hyperphysics_pbit::PBitError),

    #[error("Field calculation error: {message}")]
    FieldError { message: String },

    #[error("Integration error: {message}")]
    IntegrationError { message: String },
}

pub type Result<T> = std::result::Result<T, SyntergicError>;

/// Speed of syntergic field propagation (normalized units)
pub const SYNTERGIC_SPEED: f64 = 1.0;

/// Curvature parameter κ = √(-K) = 1 for K = -1
pub const KAPPA: f64 = 1.0;
