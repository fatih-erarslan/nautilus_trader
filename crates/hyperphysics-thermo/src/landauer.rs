//! Landauer's principle enforcement
//!
//! Research: Landauer (1961) "Irreversibility and heat generation in computing"
//! Research: Berut et al. (2012) "Experimental verification of Landauer's principle" Nature 483:187

use crate::{ThermoError, Result, BOLTZMANN_CONSTANT, LN_2};

/// Landauer principle enforcer
///
/// E_min = k_B T ln(2) per bit erased
pub struct LandauerEnforcer {
    boltzmann_constant: f64,
    temperature: f64,
}

impl LandauerEnforcer {
    /// Create new Landauer enforcer
    pub fn new(temperature: f64) -> Result<Self> {
        if temperature <= 0.0 {
            return Err(ThermoError::InvalidTemperature { temp: temperature });
        }

        Ok(Self {
            boltzmann_constant: BOLTZMANN_CONSTANT,
            temperature,
        })
    }

    /// Calculate minimum energy for erasing one bit
    ///
    /// E_min = k_B T ln(2)
    pub fn minimum_erasure_energy(&self) -> f64 {
        self.boltzmann_constant * self.temperature * LN_2
    }

    /// Calculate minimum energy for erasing N bits
    pub fn minimum_erasure_energy_n(&self, bits_erased: usize) -> f64 {
        self.minimum_erasure_energy() * (bits_erased as f64)
    }

    /// Verify that energy dissipated meets Landauer bound
    ///
    /// Returns Ok if E_dissipated ≥ E_min, Error otherwise
    pub fn verify_bound(&self, energy_dissipated: f64, bits_erased: usize) -> Result<()> {
        let e_min = self.minimum_erasure_energy_n(bits_erased);

        if energy_dissipated < e_min {
            return Err(ThermoError::LandauerViolation {
                energy: energy_dissipated,
                e_min,
            });
        }

        Ok(())
    }

    /// Calculate energy dissipated from entropy change
    ///
    /// Q = T ΔS (heat dissipated to environment)
    pub fn energy_from_entropy(&self, delta_entropy: f64) -> f64 {
        self.temperature * delta_entropy
    }

    /// Calculate entropy change from energy dissipated
    pub fn entropy_from_energy(&self, energy_dissipated: f64) -> f64 {
        energy_dissipated / self.temperature
    }

    /// Track energy dissipation over simulation step
    ///
    /// Verifies second law: ΔS ≥ 0
    pub fn track_dissipation(
        &self,
        initial_entropy: f64,
        final_entropy: f64,
    ) -> Result<f64> {
        let delta_s = final_entropy - initial_entropy;

        // Verify second law (with small numerical tolerance)
        // Tolerance: -1e-23 allows for floating-point rounding errors
        // but catches real violations like -1e-22
        if delta_s < -1e-23 {
            return Err(ThermoError::SecondLawViolation { delta_s });
        }

        let energy = self.energy_from_entropy(delta_s);
        Ok(energy)
    }

    /// Get temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) -> Result<()> {
        if temperature <= 0.0 {
            return Err(ThermoError::InvalidTemperature { temp: temperature });
        }
        self.temperature = temperature;
        Ok(())
    }

    /// Calculate Landauer bound at room temperature (300K)
    pub fn bound_at_room_temperature() -> f64 {
        // k_B = 1.380649e-23 J/K
        // T = 300 K
        // ln(2) = 0.693
        // E_min ≈ 2.87e-21 J
        BOLTZMANN_CONSTANT * 300.0 * LN_2
    }

    /// Calculate number of bits erasable with given energy at this temperature
    pub fn bits_erasable(&self, energy: f64) -> f64 {
        energy / self.minimum_erasure_energy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimum_erasure_energy() {
        let enforcer = LandauerEnforcer::new(300.0).unwrap();

        let e_min = enforcer.minimum_erasure_energy();

        // At 300K: E_min = k_B * 300 * ln(2) ≈ 2.87e-21 J
        assert!((e_min - 2.87e-21).abs() < 1e-23);
    }

    #[test]
    fn test_room_temperature_bound() {
        let e_min = LandauerEnforcer::bound_at_room_temperature();
        assert!((e_min - 2.87e-21).abs() < 1e-23);
    }

    #[test]
    fn test_verify_bound_satisfied() {
        let enforcer = LandauerEnforcer::new(300.0).unwrap();

        // Dissipate more than minimum: OK
        let result = enforcer.verify_bound(3.0e-21, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_verify_bound_violated() {
        let enforcer = LandauerEnforcer::new(300.0).unwrap();

        // Dissipate less than minimum: Violation
        let result = enforcer.verify_bound(1.0e-21, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_energy_entropy_conversion() {
        let enforcer = LandauerEnforcer::new(300.0).unwrap();

        let delta_s = 1.0e-22;
        let energy = enforcer.energy_from_entropy(delta_s);

        // E = T * ΔS = 300 * 1e-22 = 3e-20 J
        assert!((energy - 3.0e-20).abs() < 1e-32);

        // Reverse conversion
        let recovered_s = enforcer.entropy_from_energy(energy);
        assert!((recovered_s - delta_s).abs() < 1e-32);
    }

    #[test]
    fn test_second_law_check() {
        let enforcer = LandauerEnforcer::new(300.0).unwrap();

        // Entropy increase: OK
        let result = enforcer.track_dissipation(1.0e-22, 2.0e-22);
        assert!(result.is_ok());

        // Entropy decrease: Violation
        let result = enforcer.track_dissipation(2.0e-22, 1.0e-22);
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_scaling() {
        let enforcer_cold = LandauerEnforcer::new(100.0).unwrap();
        let enforcer_hot = LandauerEnforcer::new(300.0).unwrap();

        let e_cold = enforcer_cold.minimum_erasure_energy();
        let e_hot = enforcer_hot.minimum_erasure_energy();

        // Hot temperature requires more energy
        assert!(e_hot > e_cold);
        assert!((e_hot / e_cold - 3.0).abs() < 1e-10); // Ratio should be 300/100 = 3
    }

    #[test]
    fn test_bits_erasable() {
        let enforcer = LandauerEnforcer::new(300.0).unwrap();

        let e_min = enforcer.minimum_erasure_energy();
        let energy = 10.0 * e_min;

        let bits = enforcer.bits_erasable(energy);
        assert!((bits - 10.0).abs() < 1e-10);
    }
}
