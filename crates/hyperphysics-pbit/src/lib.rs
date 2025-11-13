//! # Probabilistic Bit (pBit) Dynamics Simulator
//!
//! Implementation of stochastic binary variables with physically realistic dynamics.
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Camsari et al. (2017) "Stochastic p-bits for invertible logic" PRX 7:031014
//! - Kaiser & Datta (2021) "Probabilistic computing with p-bits" Nature Electronics
//! - Borders et al. (2019) "Integer factorization using stochastic MTJs" Nature 573:390
//! - Gillespie (1977) "Exact stochastic simulation" J. Phys. Chem 81:2340
//! - Metropolis et al. (1953) "Equation of state calculations" J. Chem. Phys 21:1087
//!
//! ## pBit Definition
//!
//! A pBit is a stochastic binary variable s ∈ {0,1} with probability:
//!
//! P(s=1) = σ(h_eff / T) = 1/(1 + exp(-h_eff/T))
//!
//! where h_eff = bias + Σ_j J_ij s_j

pub mod pbit;
pub mod lattice;
pub mod dynamics;
pub mod gillespie;
pub mod metropolis;
pub mod coupling;
pub mod sparse_matrix;
pub mod simd;

pub use pbit::PBit;
pub use lattice::PBitLattice;
pub use dynamics::{PBitDynamics, Algorithm, DynamicsStatistics};
pub use gillespie::GillespieSimulator;
pub use metropolis::MetropolisSimulator;
pub use coupling::CouplingNetwork;
pub use sparse_matrix::{SparseCouplingMatrix, CouplingStatistics};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum PBitError {
    #[error("Invalid probability: {prob} not in [0,1]")]
    InvalidProbability { prob: f64 },

    #[error("Invalid temperature: {temp} must be positive")]
    InvalidTemperature { temp: f64 },

    #[error("Lattice error: {message}")]
    LatticeError { message: String },

    #[error("Simulation error: {message}")]
    SimulationError { message: String },
}

pub type Result<T> = std::result::Result<T, PBitError>;

/// Boltzmann constant (J/K)
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boltzmann_constant() {
        assert_eq!(BOLTZMANN_CONSTANT, 1.380649e-23);
    }
}
