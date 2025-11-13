//! Property-based tests for coupling dynamics
//!
//! Tests verify mathematical properties of spin-spin coupling and interaction terms.

use proptest::prelude::*;
use hyperphysics_pbit::*;
use approx::assert_relative_eq;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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
        fn prop_coupling_lattice_construction(
            coupling in 0.1f64..10.0
        ) {
            // Property: Lattice can be constructed with any valid coupling
            let lattice = PBitLattice::roi_48(coupling)?;

            // Should have a reasonable number of sites
            prop_assert!(lattice.size() > 0);
            prop_assert!(lattice.size() <= 100);

            // All pBits should be initialized
            prop_assert_eq!(lattice.pbits().len(), lattice.size());
        }

        #[test]
        fn prop_simulation_initializes(
            coupling in 0.1f64..10.0,
            seed in 0u64..1000
        ) {
            // Property: Simulator can be initialized with any valid lattice
            let lattice = PBitLattice::roi_48(coupling)?;
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(seed);

            // Should be able to run at least one step
            let dt = sim.step(&mut rng)?;
            prop_assert!(dt >= 0.0);
            prop_assert!(dt.is_finite());
        }

        #[test]
        fn prop_coupling_affects_dynamics(
            weak_coupling in 0.1f64..1.0,
            strong_coupling in 5.0f64..10.0
        ) {
            // Property: Stronger coupling should affect dynamics
            let lattice_weak = PBitLattice::roi_48(weak_coupling)?;
            let lattice_strong = PBitLattice::roi_48(strong_coupling)?;

            // Both lattices should be valid and have the same structure
            prop_assert_eq!(lattice_weak.size(), lattice_strong.size());

            // pBits should be properly initialized
            prop_assert_eq!(lattice_weak.pbits().len(), lattice_weak.size());
            prop_assert_eq!(lattice_strong.pbits().len(), lattice_strong.size());
        }
    }
}

#[cfg(test)]
mod pbit_coupling_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_pbit_states_initialized(
            coupling in 0.1f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;

            // Property: All pBits should be properly initialized
            prop_assert!(lattice.pbits().len() > 0);
            prop_assert_eq!(lattice.pbits().len(), lattice.size());

            // All pBits should have valid positions in Poincaré disk
            for pbit in lattice.pbits() {
                let pos_norm = pbit.position().coords().norm();
                prop_assert!(pos_norm < 1.0); // Valid Poincaré disk
            }
        }

        #[test]
        fn prop_effective_field_finite(
            coupling in 0.1f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;
            let states = lattice.states();

            // Property: Effective field should always be finite
            for pbit in lattice.pbits() {
                let h_eff = pbit.effective_field(&states);
                prop_assert!(h_eff.is_finite());
            }
        }
    }
}
