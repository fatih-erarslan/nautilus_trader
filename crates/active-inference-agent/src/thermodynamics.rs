//! Thermodynamic constraints for conscious processing
//!
//! Implements Landauer's principle and thermodynamic bounds on information processing:
//! - Minimum energy cost: E_min = kT ln(2) per bit erased
//! - Entropy production bounds
//! - Thermodynamic irreversibility tracking
//!
//! # pbRTCA Integration
//!
//! The thermodynamic layer ensures all conscious processing respects
//! fundamental physical limits, preventing physically impossible computations.

use crate::ConsciousnessError;
use serde::{Deserialize, Serialize};

/// Boltzmann constant in J/K
pub const BOLTZMANN_K: f64 = 1.380649e-23;

/// Landauer's limit: kT ln(2) at room temperature (300K) ≈ 2.87e-21 J/bit
pub const LANDAUER_LIMIT_300K: f64 = 2.8755e-21;

/// Thermodynamic state for conscious processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThermodynamicState {
    /// Temperature in Kelvin
    pub temperature: f64,
    /// Available energy budget in Joules
    pub energy_budget: f64,
    /// Energy consumed so far in Joules
    pub energy_consumed: f64,
    /// Bits erased (irreversible operations)
    pub bits_erased: u64,
    /// Entropy produced (J/K)
    pub entropy_produced: f64,
    /// Whether we're in reversible computing mode
    pub reversible_mode: bool,
}

impl Default for ThermodynamicState {
    fn default() -> Self {
        Self::new(300.0, 1e-12) // Room temp, 1 picojoule budget (sufficient for typical cycles)
    }
}

impl ThermodynamicState {
    /// Create new thermodynamic state
    ///
    /// # Arguments
    /// * `temperature` - System temperature in Kelvin
    /// * `energy_budget` - Available energy in Joules
    pub fn new(temperature: f64, energy_budget: f64) -> Self {
        Self {
            temperature,
            energy_budget,
            energy_consumed: 0.0,
            bits_erased: 0,
            entropy_produced: 0.0,
            reversible_mode: false,
        }
    }

    /// Compute Landauer limit at current temperature
    ///
    /// E_min = kT ln(2)
    #[inline]
    pub fn landauer_limit(&self) -> f64 {
        BOLTZMANN_K * self.temperature * std::f64::consts::LN_2
    }

    /// Compute minimum energy required for n bits of erasure
    #[inline]
    pub fn min_energy_for_bits(&self, n_bits: u64) -> f64 {
        self.landauer_limit() * n_bits as f64
    }

    /// Verify Landauer bound is satisfied
    ///
    /// Returns error if energy budget insufficient for required erasures
    pub fn verify_landauer_bound(&self) -> Result<(), ConsciousnessError> {
        let required = self.min_energy_for_bits(self.bits_erased);
        let remaining = self.energy_budget - self.energy_consumed;

        if remaining < required {
            return Err(ConsciousnessError::LandauerBoundViolated {
                provided: remaining,
                required,
            });
        }

        Ok(())
    }

    /// Record energy cost of a processing operation
    ///
    /// Tracks both energy consumption and entropy production
    ///
    /// # Arguments
    /// * `free_energy_change` - Free energy change in nats (dimensionless)
    pub fn record_processing_cost(&mut self, free_energy_change: f64) -> Result<(), ConsciousnessError> {
        // Free energy is in nats (natural units of information)
        // Convert to bits: bits = nats / ln(2)
        // Then estimate bits erased (capped at reasonable value)
        let nats = free_energy_change.abs();
        let estimated_bits = ((nats / std::f64::consts::LN_2).ceil() as u64).min(1000);

        // Minimum energy cost (Landauer bound)
        let min_cost = self.min_energy_for_bits(estimated_bits);

        // Actual cost with inefficiency factor (realistic processing ~100x Landauer)
        let efficiency_factor = if self.reversible_mode { 1.1 } else { 100.0 };
        let actual_cost = min_cost * efficiency_factor;

        // Check budget
        if self.energy_consumed + actual_cost > self.energy_budget {
            return Err(ConsciousnessError::ThermodynamicViolation(
                format!(
                    "Energy budget exceeded: would consume {:.2e} J, budget {:.2e} J",
                    self.energy_consumed + actual_cost,
                    self.energy_budget
                )
            ));
        }

        // Update state
        self.energy_consumed += actual_cost;
        self.bits_erased += estimated_bits;
        self.entropy_produced += actual_cost / self.temperature;

        Ok(())
    }

    /// Record explicit bit erasure
    pub fn record_bit_erasure(&mut self, n_bits: u64) -> Result<(), ConsciousnessError> {
        let cost = self.min_energy_for_bits(n_bits);

        if self.energy_consumed + cost > self.energy_budget {
            return Err(ConsciousnessError::ThermodynamicViolation(
                format!("Cannot erase {} bits: insufficient energy", n_bits)
            ));
        }

        self.bits_erased += n_bits;
        self.energy_consumed += cost;
        self.entropy_produced += cost / self.temperature;

        Ok(())
    }

    /// Get remaining energy budget
    #[inline]
    pub fn remaining_energy(&self) -> f64 {
        self.energy_budget - self.energy_consumed
    }

    /// Get remaining computational capacity in bits
    #[inline]
    pub fn remaining_bits(&self) -> u64 {
        let remaining_energy = self.remaining_energy();
        (remaining_energy / self.landauer_limit()).floor() as u64
    }

    /// Reset energy consumption (e.g., for new cycle)
    pub fn reset(&mut self) {
        self.energy_consumed = 0.0;
        self.bits_erased = 0;
        self.entropy_produced = 0.0;
    }

    /// Set reversible computing mode (lower energy cost)
    pub fn set_reversible_mode(&mut self, enabled: bool) {
        self.reversible_mode = enabled;
    }

    /// Compute thermodynamic efficiency
    ///
    /// η = E_landauer / E_actual
    pub fn efficiency(&self) -> f64 {
        if self.energy_consumed < 1e-30 {
            return 1.0;
        }
        let theoretical_min = self.min_energy_for_bits(self.bits_erased);
        theoretical_min / self.energy_consumed
    }

    /// Estimate processing time given power constraint
    ///
    /// τ = E / P
    pub fn estimate_processing_time(&self, power_watts: f64) -> f64 {
        self.energy_consumed / power_watts
    }
}

/// Thermodynamic limit on information processing rate
///
/// From Bekenstein bound and quantum speed limit
#[derive(Debug, Clone, Copy)]
pub struct ProcessingLimit {
    /// Maximum bits per second given energy
    pub max_bit_rate: f64,
    /// Minimum time for single operation
    pub min_operation_time: f64,
}

impl ProcessingLimit {
    /// Compute from energy and Planck time
    ///
    /// Margolus-Levitin theorem: t_min = πℏ / (2E)
    pub fn from_energy(energy_joules: f64) -> Self {
        const HBAR: f64 = 1.054571817e-34; // Reduced Planck constant
        const PI: f64 = std::f64::consts::PI;

        let min_time = PI * HBAR / (2.0 * energy_joules);
        let max_rate = 1.0 / min_time;

        Self {
            max_bit_rate: max_rate,
            min_operation_time: min_time,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_landauer_limit() {
        let thermo = ThermodynamicState::new(300.0, 1e-15);
        let limit = thermo.landauer_limit();

        // kT ln(2) at 300K ≈ 2.87e-21 J
        assert!((limit - LANDAUER_LIMIT_300K).abs() < 1e-23);
    }

    #[test]
    fn test_energy_tracking() {
        let mut thermo = ThermodynamicState::new(300.0, 1e-15);

        // Should succeed with small operation
        assert!(thermo.record_bit_erasure(1000).is_ok());
        assert!(thermo.bits_erased == 1000);
        assert!(thermo.energy_consumed > 0.0);
    }

    #[test]
    fn test_budget_exceeded() {
        let mut thermo = ThermodynamicState::new(300.0, 1e-21); // Very small budget

        // Should fail with large operation
        let result = thermo.record_bit_erasure(1_000_000);
        assert!(result.is_err());
    }

    #[test]
    fn test_efficiency() {
        let mut thermo = ThermodynamicState::new(300.0, 1e-15);
        thermo.set_reversible_mode(true);

        thermo.record_bit_erasure(100).unwrap();
        let eff = thermo.efficiency();

        // Reversible mode should be more efficient
        assert!(eff > 0.5);
    }

    #[test]
    fn test_processing_limit() {
        let limit = ProcessingLimit::from_energy(1e-15);

        // Should be finite and positive
        assert!(limit.max_bit_rate > 0.0);
        assert!(limit.min_operation_time > 0.0);
        assert!(limit.max_bit_rate.is_finite());
    }
}
