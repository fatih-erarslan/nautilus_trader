//! Ising Hamiltonian energy calculations

use hyperphysics_pbit::PBitLattice;

/// Hamiltonian energy calculator for Ising model
///
/// H = -Σ_i h_i s_i - Σ_{i<j} J_ij s_i s_j
pub struct HamiltonianCalculator;

impl HamiltonianCalculator {
    /// Calculate total energy of lattice configuration
    ///
    /// Uses Ising model with spin representation: s ∈ {-1, +1}
    pub fn energy(lattice: &PBitLattice) -> f64 {
        let mut energy = 0.0;
        let states = lattice.states();

        for (i, pbit) in lattice.pbits().iter().enumerate() {
            let si = pbit.spin(); // Convert bool to spin

            // Bias term: -h_i * s_i
            energy -= pbit.bias() * si;

            // Coupling terms: -J_ij * s_i * s_j (count each pair once)
            for (j, strength) in pbit.couplings() {
                if *j > i {
                    let sj = if states[*j] { 1.0 } else { -1.0 };
                    energy -= strength * si * sj;
                }
            }
        }

        energy
    }

    /// Calculate energy difference for flipping pBit i
    ///
    /// ΔE = E(s') - E(s) = 2 * h_eff * s_i
    pub fn energy_difference(
        lattice: &PBitLattice,
        flip_index: usize,
    ) -> f64 {
        let pbit = &lattice.pbits()[flip_index];
        let states = lattice.states();
        let h_eff = pbit.effective_field(&states);
        let si = pbit.spin();

        2.0 * h_eff * si
    }

    /// Calculate energy per pBit
    pub fn energy_per_pbit(lattice: &PBitLattice) -> f64 {
        Self::energy(lattice) / (lattice.size() as f64)
    }

    /// Calculate local energy contribution of pBit i
    pub fn local_energy(lattice: &PBitLattice, index: usize) -> f64 {
        let pbit = &lattice.pbits()[index];
        let states = lattice.states();
        let si = pbit.spin();

        let mut local_e = -pbit.bias() * si;

        for (j, strength) in pbit.couplings() {
            let sj = if states[*j] { 1.0 } else { -1.0 };
            local_e -= 0.5 * strength * si * sj; // 0.5 to avoid double counting
        }

        local_e
    }

    /// Calculate average absolute coupling strength
    pub fn average_coupling_strength(lattice: &PBitLattice) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        for pbit in lattice.pbits() {
            for strength in pbit.couplings().values() {
                total += strength.abs();
                count += 1;
            }
        }

        if count > 0 {
            total / (count as f64)
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperphysics_pbit::{CouplingNetwork, PBitLattice};

    #[test]
    fn test_zero_energy_independent() {
        // All independent pBits with zero bias should have zero energy
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let energy = HamiltonianCalculator::energy(&lattice);

        assert_eq!(energy, 0.0);
    }

    #[test]
    fn test_energy_with_couplings() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let energy = HamiltonianCalculator::energy(&lattice);
        // Energy should be defined (not NaN or infinite)
        assert!(energy.is_finite());
    }

    #[test]
    fn test_energy_difference() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let e_before = HamiltonianCalculator::energy(&lattice);
        let delta_e = HamiltonianCalculator::energy_difference(&lattice, 0);

        // Flip pBit 0
        let pbits = lattice.pbits_mut();
        let state = pbits[0].state();
        pbits[0].set_state(!state);

        let e_after = HamiltonianCalculator::energy(&lattice);

        // Check energy difference matches
        assert!((e_after - e_before - delta_e).abs() < 1e-10);
    }

    #[test]
    fn test_local_energy_sum() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let total_energy = HamiltonianCalculator::energy(&lattice);

        let local_sum: f64 = (0..lattice.size())
            .map(|i| HamiltonianCalculator::local_energy(&lattice, i))
            .sum();

        // Local energies should sum to total (approximately, due to double counting factor)
        assert!((local_sum - total_energy).abs() < 1e-10);
    }
}
