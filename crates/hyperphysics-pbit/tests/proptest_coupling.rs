//! Property-based tests for coupling dynamics
//!
//! Tests verify mathematical properties of spin-spin coupling and interaction terms.

use proptest::prelude::*;
use hyperphysics_pbit::*;
use approx::assert_relative_eq;

#[cfg(test)]
mod coupling_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_coupling_symmetric(
            coupling_strength in -10.0f64..10.0,
            spin1 in -1i8..=1,
            spin2 in -1i8..=1
        ) {
            // Skip zero spins
            if spin1 == 0 || spin2 == 0 {
                return Ok(());
            }

            // Property: Coupling energy E(i,j) = E(j,i) (symmetric)
            let energy_ij = -coupling_strength * (spin1 as f64) * (spin2 as f64);
            let energy_ji = -coupling_strength * (spin2 as f64) * (spin1 as f64);

            prop_assert_eq!(energy_ij, energy_ji);
        }

        #[test]
        fn prop_coupling_scales_linearly(
            base_coupling in 0.1f64..10.0,
            scale_factor in 0.1f64..10.0
        ) {
            let lattice1 = PBitLattice::roi_48(base_coupling)?;
            let lattice2 = PBitLattice::roi_48(base_coupling * scale_factor)?;

            let sim1 = GillespieSimulator::new(lattice1, 1.0);
            let sim2 = GillespieSimulator::new(lattice2, 1.0);

            let energy1 = sim1.total_energy();
            let energy2 = sim2.total_energy();

            // Property: Energy should scale linearly with coupling strength
            // E2 / E1 ≈ scale_factor (within numerical tolerance)
            if energy1.abs() > 1e-10 {
                let ratio = energy2 / energy1;
                assert_relative_eq!(ratio, scale_factor, epsilon = 0.1);
            }
        }

        #[test]
        fn prop_antiferromagnetic_vs_ferromagnetic(
            coupling_magnitude in 0.1f64..10.0
        ) {
            // Property: Sign of coupling determines energy ordering
            let lattice_ferro = PBitLattice::roi_48(coupling_magnitude)?;
            let lattice_antiferro = PBitLattice::roi_48(-coupling_magnitude)?;

            let sim_ferro = GillespieSimulator::new(lattice_ferro, 1.0);
            let sim_antiferro = GillespieSimulator::new(lattice_antiferro, 1.0);

            let energy_ferro = sim_ferro.total_energy();
            let energy_antiferro = sim_antiferro.total_energy();

            // Energies should have opposite signs for same configuration
            prop_assert_eq!(energy_ferro, -energy_antiferro);
        }
    }
}

#[cfg(test)]
mod field_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_field_energy_linear_in_magnetization(
            field_strength in -10.0f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice, 1.0);

            // Apply external field
            sim.set_external_field(field_strength);

            let magnetization = sim.lattice().magnetization();
            let field_energy = -field_strength * magnetization;

            // Property: Field energy = -h·M (linear relationship)
            let measured_field_energy = sim.field_energy();
            assert_relative_eq!(measured_field_energy, field_energy, epsilon = 1e-10);
        }
    }
}
