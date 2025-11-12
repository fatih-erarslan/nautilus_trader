//! Free energy calculations for pBit systems

use crate::BOLTZMANN_CONSTANT;
// Future: PBitLattice integration for free energy calculations

/// Free energy calculator
///
/// F = E - TS (Helmholtz free energy)
/// G = H - TS (Gibbs free energy)
pub struct FreeEnergyCalculator {
    boltzmann_constant: f64,
}

impl FreeEnergyCalculator {
    /// Create new free energy calculator
    pub fn new() -> Self {
        Self {
            boltzmann_constant: BOLTZMANN_CONSTANT,
        }
    }

    /// Calculate Helmholtz free energy
    ///
    /// F = E - TS
    pub fn helmholtz_free_energy(
        &self,
        energy: f64,
        temperature: f64,
        entropy: f64,
    ) -> f64 {
        energy - temperature * entropy
    }

    /// Calculate free energy difference
    ///
    /// ΔF = ΔE - T ΔS
    pub fn free_energy_difference(
        &self,
        delta_e: f64,
        temperature: f64,
        delta_s: f64,
    ) -> f64 {
        delta_e - temperature * delta_s
    }

    /// Calculate partition function approximation
    ///
    /// Z ≈ Σ exp(-E_i / (k_B T))
    pub fn partition_function_approx(
        &self,
        energies: &[f64],
        temperature: f64,
    ) -> f64 {
        energies
            .iter()
            .map(|&e| (-e / (self.boltzmann_constant * temperature)).exp())
            .sum()
    }

    /// Calculate free energy from partition function
    ///
    /// F = -k_B T ln(Z)
    pub fn free_energy_from_partition(
        &self,
        partition_function: f64,
        temperature: f64,
    ) -> f64 {
        -self.boltzmann_constant * temperature * partition_function.ln()
    }

    /// Estimate equilibrium probability of a state
    ///
    /// P(s) = exp(-E(s) / (k_B T)) / Z
    pub fn equilibrium_probability(
        &self,
        energy: f64,
        partition_function: f64,
        temperature: f64,
    ) -> f64 {
        let boltzmann_factor = (-energy / (self.boltzmann_constant * temperature)).exp();
        boltzmann_factor / partition_function
    }

    /// Calculate mean energy at equilibrium
    ///
    /// <E> = -∂ln(Z)/∂β where β = 1/(k_B T)
    pub fn mean_energy(
        &self,
        energies: &[f64],
        temperature: f64,
    ) -> f64 {
        let z = self.partition_function_approx(energies, temperature);

        let weighted_sum: f64 = energies
            .iter()
            .map(|&e| {
                let prob = self.equilibrium_probability(e, z, temperature);
                prob * e
            })
            .sum();

        weighted_sum
    }

    /// Calculate heat capacity
    ///
    /// C = ∂<E>/∂T = (<E²> - <E>²) / (k_B T²)
    pub fn heat_capacity(
        &self,
        energies: &[f64],
        temperature: f64,
    ) -> f64 {
        let z = self.partition_function_approx(energies, temperature);

        let mean_e = energies
            .iter()
            .map(|&e| {
                let prob = self.equilibrium_probability(e, z, temperature);
                prob * e
            })
            .sum::<f64>();

        let mean_e_sq = energies
            .iter()
            .map(|&e| {
                let prob = self.equilibrium_probability(e, z, temperature);
                prob * e * e
            })
            .sum::<f64>();

        let variance = mean_e_sq - mean_e * mean_e;
        variance / (self.boltzmann_constant * temperature * temperature)
    }
}

impl Default for FreeEnergyCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_helmholtz_free_energy() {
        let calc = FreeEnergyCalculator::new();

        let e = 1.0e-20; // J
        let t = 300.0; // K
        let s = 1.0e-22; // J/K

        let f = calc.helmholtz_free_energy(e, t, s);

        // F = E - TS = 1e-20 - 300*1e-22 = 1e-20 - 3e-20 = -2e-20
        assert!((f + 2.0e-20).abs() < 1e-32);
    }

    #[test]
    fn test_partition_function() {
        let calc = FreeEnergyCalculator::new();

        // Two states with equal energy
        let energies = vec![1.0e-20, 1.0e-20];
        let z = calc.partition_function_approx(&energies, 300.0);

        // Z = 2 * exp(-E/(k_B T))
        let expected = 2.0 * (-1.0e-20 / (BOLTZMANN_CONSTANT * 300.0)).exp();
        assert!((z - expected).abs() < 1e-10);
    }

    #[test]
    fn test_equilibrium_probability() {
        let calc = FreeEnergyCalculator::new();

        let energies = vec![0.0, 1.0e-20];
        let z = calc.partition_function_approx(&energies, 300.0);

        // Ground state should have higher probability
        let p0 = calc.equilibrium_probability(energies[0], z, 300.0);
        let p1 = calc.equilibrium_probability(energies[1], z, 300.0);

        assert!(p0 > p1);
        assert!((p0 + p1 - 1.0).abs() < 1e-10); // Normalized
    }

    #[test]
    fn test_mean_energy() {
        let calc = FreeEnergyCalculator::new();

        let energies = vec![0.0, 1.0e-20, 2.0e-20];
        let mean_e = calc.mean_energy(&energies, 300.0);

        // Mean energy should be between min and max
        assert!(mean_e >= 0.0);
        assert!(mean_e <= 2.0e-20);
    }

    #[test]
    fn test_temperature_dependence() {
        let calc = FreeEnergyCalculator::new();

        let energies = vec![0.0, 1.0e-20];

        // At low T: mostly in ground state
        let mean_cold = calc.mean_energy(&energies, 10.0);

        // At high T: more excited states
        let mean_hot = calc.mean_energy(&energies, 1000.0);

        assert!(mean_hot > mean_cold);
    }
}
