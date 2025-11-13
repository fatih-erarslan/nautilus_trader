//! # Thermodynamics Engine
//!
//! Implementation of thermodynamic laws for pBit systems.
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Landauer (1961) "Irreversibility and heat generation" IBM J. Res. Dev. 5(3):183
//! - Berut et al. (2012) "Experimental verification of Landauer's principle" Nature 483:187
//! - Jarzynski (1997) "Nonequilibrium equality for free energy" PRL 78:2690
//! - Mezard et al. (1987) "Spin Glass Theory and Beyond" World Scientific
//!
//! ## Core Concepts
//!
//! - Ising Hamiltonian: H = -Σ h_i s_i - Σ J_ij s_i s_j
//! - Gibbs Entropy: S = -k_B Σ P(s) ln P(s)
//! - Landauer Bound: E_min = k_B T ln(2)
//! - Second Law: ΔS_total ≥ 0

pub mod hamiltonian;
pub mod entropy;
pub mod landauer;
pub mod free_energy;
pub mod temperature;
pub mod observables;
pub mod negentropy;

pub use hamiltonian::HamiltonianCalculator;
pub use entropy::EntropyCalculator;
pub use landauer::LandauerEnforcer;
pub use free_energy::FreeEnergyCalculator;
pub use temperature::{Temperature, TemperatureSchedule, ScheduleType};
pub use observables::{Observable, Correlation, ObservableTimeSeries};
pub use negentropy::{NegentropyAnalyzer, NegentropyMeasurement, NegentropyFlow, NegentropyDynamics};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ThermoError {
    #[error("Energy conservation violated: ΔE = {delta_e}")]
    EnergyConservation { delta_e: f64 },

    #[error("Second law violation: ΔS = {delta_s} < 0")]
    SecondLawViolation { delta_s: f64 },

    #[error("Landauer bound violation: E = {energy} < E_min = {e_min}")]
    LandauerViolation { energy: f64, e_min: f64 },

    #[error("Invalid temperature: {temp}")]
    InvalidTemperature { temp: f64 },
}

pub type Result<T> = std::result::Result<T, ThermoError>;

/// Boltzmann constant (J/K)
pub const BOLTZMANN_CONSTANT: f64 = 1.380649e-23;

/// Natural logarithm of 2
pub const LN_2: f64 = 0.6931471805599453;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(BOLTZMANN_CONSTANT, 1.380649e-23);
        assert!((LN_2 - 2.0_f64.ln()).abs() < 1e-15);
    }
}
