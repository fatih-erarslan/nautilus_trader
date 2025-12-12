//! # Layer 1: Thermodynamic Computing API
//!
//! Energy-efficient computation based on Ising model criticality.
//!
//! ## Scientific Foundation
//!
//! - **Critical Temperature**: Tc = 2.269185 (Onsager's exact solution for 2D Ising)
//! - **Phase Transition**: Second-order continuous transition at Tc
//! - **Energy Management**: Boltzmann statistics for probabilistic computing
//! - **Entropy**: System entropy tracking for information-theoretic analysis
//!
//! ## Key Concepts
//!
//! ```text
//! Energy Landscape:
//!   E = -J Σ s_i s_j - h Σ s_i
//!
//! Boltzmann Distribution:
//!   P(s) = exp(-βE) / Z
//!   β = 1 / (k_B T)
//!
//! Partition Function:
//!   Z = Σ exp(-βE_i)
//! ```

use crate::{Result, QksError};

/// Onsager's exact critical temperature for 2D Ising model
/// Tc = 2 / ln(1 + √2) ≈ 2.269185
pub const ISING_CRITICAL_TEMP: f64 = 2.269_185_314_213_022;

/// Boltzmann constant (normalized units)
pub const BOLTZMANN_CONSTANT: f64 = 1.0;

/// Default coupling strength J
pub const DEFAULT_COUPLING: f64 = 1.0;

/// Energy state of the thermodynamic system
#[derive(Debug, Clone)]
pub struct EnergyState {
    /// Total system energy
    pub energy: f64,
    /// Current temperature
    pub temperature: f64,
    /// Magnetic field strength
    pub field: f64,
    /// System entropy
    pub entropy: f64,
    /// Free energy (F = E - TS)
    pub free_energy: f64,
}

/// Thermodynamic phase
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    /// Ordered phase (T < Tc)
    Ordered,
    /// Critical phase (T ≈ Tc)
    Critical,
    /// Disordered phase (T > Tc)
    Disordered,
}

/// Get current energy of the system
///
/// # Returns
/// Current energy in normalized units
///
/// # Example
/// ```rust,ignore
/// let energy = get_energy()?;
/// println!("System energy: {}", energy);
/// ```
pub fn get_energy() -> Result<f64> {
    // TODO: Interface with actual thermodynamic engine
    // For now, return placeholder value
    Ok(0.0)
}

/// Get full energy state including temperature and entropy
///
/// # Returns
/// Complete energy state descriptor
///
/// # Example
/// ```rust,ignore
/// let state = get_energy_state()?;
/// println!("T = {}, S = {}", state.temperature, state.entropy);
/// ```
pub fn get_energy_state() -> Result<EnergyState> {
    Ok(EnergyState {
        energy: 0.0,
        temperature: ISING_CRITICAL_TEMP,
        field: 0.0,
        entropy: 0.0,
        free_energy: 0.0,
    })
}

/// Set system temperature
///
/// # Arguments
/// * `temperature` - Target temperature (dimensionless units)
///
/// # Example
/// ```rust,ignore
/// // Set to critical temperature for optimal computation
/// set_temperature(ISING_CRITICAL_TEMP)?;
/// ```
pub fn set_temperature(temperature: f64) -> Result<()> {
    if temperature < 0.0 {
        return Err(QksError::InvalidConfig(
            "Temperature must be non-negative".to_string()
        ));
    }
    // TODO: Interface with thermodynamic engine
    Ok(())
}

/// Set magnetic field strength
///
/// # Arguments
/// * `field` - External field strength
///
/// # Example
/// ```rust,ignore
/// set_field(0.1)?; // Small field to break symmetry
/// ```
pub fn set_field(field: f64) -> Result<()> {
    // TODO: Interface with thermodynamic engine
    Ok(())
}

/// Check if system is at critical point
///
/// # Arguments
/// * `temperature` - Temperature to check
/// * `tolerance` - Tolerance for criticality (default: 0.01)
///
/// # Returns
/// `true` if T ≈ Tc within tolerance
pub fn is_critical(temperature: f64, tolerance: f64) -> bool {
    (temperature - ISING_CRITICAL_TEMP).abs() < tolerance
}

/// Determine thermodynamic phase
///
/// # Arguments
/// * `temperature` - System temperature
///
/// # Returns
/// Current thermodynamic phase
pub fn get_phase(temperature: f64) -> Phase {
    const CRITICAL_TOLERANCE: f64 = 0.05;

    if is_critical(temperature, CRITICAL_TOLERANCE) {
        Phase::Critical
    } else if temperature < ISING_CRITICAL_TEMP {
        Phase::Ordered
    } else {
        Phase::Disordered
    }
}

/// Compute Boltzmann weight for a given energy
///
/// # Arguments
/// * `energy` - Energy of the state
/// * `temperature` - System temperature
///
/// # Returns
/// Boltzmann weight exp(-βE)
///
/// # Formula
/// ```text
/// w(E) = exp(-E / (k_B T))
/// ```
pub fn boltzmann_weight(energy: f64, temperature: f64) -> f64 {
    let beta = 1.0 / (BOLTZMANN_CONSTANT * temperature);
    (-beta * energy).exp()
}

/// Sample from Boltzmann distribution
///
/// # Arguments
/// * `energies` - Array of energy values
/// * `temperature` - System temperature
///
/// # Returns
/// Sampled index according to Boltzmann probabilities
///
/// # Example
/// ```rust,ignore
/// let energies = vec![-1.0, 0.0, 1.0];
/// let idx = boltzmann_sample(&energies, 2.0)?;
/// ```
pub fn boltzmann_sample(energies: &[f64], temperature: f64) -> Result<usize> {
    if energies.is_empty() {
        return Err(QksError::InvalidConfig("Energy array is empty".to_string()));
    }

    // Compute partition function Z
    let weights: Vec<f64> = energies
        .iter()
        .map(|&e| boltzmann_weight(e, temperature))
        .collect();

    let partition_function: f64 = weights.iter().sum();

    if partition_function == 0.0 {
        return Err(QksError::Internal("Zero partition function".to_string()));
    }

    // Sample using inverse CDF
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let u: f64 = rng.gen();

    let mut cumulative = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumulative += w / partition_function;
        if u <= cumulative {
            return Ok(i);
        }
    }

    Ok(weights.len() - 1)
}

/// Compute partition function Z
///
/// # Arguments
/// * `energies` - Array of energy values
/// * `temperature` - System temperature
///
/// # Returns
/// Partition function Z = Σ exp(-βE_i)
pub fn partition_function(energies: &[f64], temperature: f64) -> f64 {
    energies
        .iter()
        .map(|&e| boltzmann_weight(e, temperature))
        .sum()
}

/// Compute free energy F = E - TS
///
/// # Arguments
/// * `energy` - Internal energy
/// * `temperature` - System temperature
/// * `entropy` - System entropy
///
/// # Returns
/// Helmholtz free energy
pub fn free_energy(energy: f64, temperature: f64, entropy: f64) -> f64 {
    energy - temperature * entropy
}

/// Initiate phase transition
///
/// # Arguments
/// * `target_temperature` - Temperature to transition to
/// * `rate` - Rate of temperature change
///
/// # Returns
/// Async handle to phase transition process
///
/// # Example
/// ```rust,ignore
/// // Anneal to critical point
/// critical_transition(ISING_CRITICAL_TEMP, 0.1)?;
/// ```
pub fn critical_transition(target_temperature: f64, rate: f64) -> Result<()> {
    if rate <= 0.0 {
        return Err(QksError::InvalidConfig(
            "Transition rate must be positive".to_string()
        ));
    }
    // TODO: Implement annealing schedule
    Ok(())
}

/// Compute heat capacity C = dE/dT
///
/// # Arguments
/// * `energies` - Energy samples at different temperatures
/// * `temperatures` - Corresponding temperatures
///
/// # Returns
/// Heat capacity estimate
pub fn heat_capacity(energies: &[f64], temperatures: &[f64]) -> Result<f64> {
    if energies.len() != temperatures.len() || energies.len() < 2 {
        return Err(QksError::InvalidConfig(
            "Invalid energy/temperature arrays".to_string()
        ));
    }

    // Simple finite difference approximation
    let de = energies[1] - energies[0];
    let dt = temperatures[1] - temperatures[0];

    if dt == 0.0 {
        return Err(QksError::Internal("Zero temperature difference".to_string()));
    }

    Ok(de / dt)
}

/// Check if system exhibits criticality (diverging susceptibility)
///
/// # Arguments
/// * `temperature` - Current temperature
///
/// # Returns
/// Criticality measure (∞ at Tc)
pub fn criticality_measure(temperature: f64) -> f64 {
    let dt = (temperature - ISING_CRITICAL_TEMP).abs();
    if dt < 1e-10 {
        f64::INFINITY
    } else {
        1.0 / dt // χ ~ |T - Tc|^(-γ) with γ=1 approximation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_critical_temperature() {
        assert_relative_eq!(ISING_CRITICAL_TEMP, 2.269185, epsilon = 1e-5);
    }

    #[test]
    fn test_boltzmann_weight() {
        let w = boltzmann_weight(0.0, 1.0);
        assert_relative_eq!(w, 1.0, epsilon = 1e-10);

        let w = boltzmann_weight(1.0, 1.0);
        assert_relative_eq!(w, (-1.0_f64).exp(), epsilon = 1e-10);
    }

    #[test]
    fn test_is_critical() {
        assert!(is_critical(ISING_CRITICAL_TEMP, 0.01));
        assert!(is_critical(2.27, 0.01));
        assert!(!is_critical(1.0, 0.01));
    }

    #[test]
    fn test_get_phase() {
        assert_eq!(get_phase(ISING_CRITICAL_TEMP), Phase::Critical);
        assert_eq!(get_phase(1.0), Phase::Ordered);
        assert_eq!(get_phase(3.0), Phase::Disordered);
    }

    #[test]
    fn test_partition_function() {
        let energies = vec![0.0, 1.0, 2.0];
        let z = partition_function(&energies, 1.0);
        let expected = 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp();
        assert_relative_eq!(z, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_free_energy() {
        let f = free_energy(10.0, 2.0, 3.0);
        assert_relative_eq!(f, 4.0, epsilon = 1e-10);
    }
}
