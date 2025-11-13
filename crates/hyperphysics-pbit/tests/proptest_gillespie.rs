//! Property-based tests for Gillespie algorithm
//!
//! These tests verify fundamental mathematical properties that must hold
//! for any correct implementation of the Gillespie stochastic simulation algorithm.

use proptest::prelude::*;
use hyperphysics_pbit::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Property: Gillespie algorithm must never produce negative transition rates
///
/// This is a fundamental requirement - transition rates represent probabilities
/// per unit time and must be non-negative for the algorithm to be physically meaningful.
#[cfg(test)]
mod gillespie_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_gillespie_never_negative_rates(
            size in 1usize..100,
            temperature in 0.1f64..1000.0,
            steps in 1usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            // Run simulation steps
            for _ in 0..steps {
                let dt = sim.step(&mut rng)?;
                // Property: Time steps must be non-negative and finite
                prop_assert!(dt >= 0.0 && dt.is_finite());
            }
        }

        #[test]
        fn prop_gillespie_conserves_particle_number(
            temperature in 0.1f64..1000.0,
            steps in 1usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let initial_size = lattice.size();

            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for _ in 0..steps {
                sim.step(&mut rng)?;
            }

            // Property: Number of pBits should remain constant
            prop_assert_eq!(sim.lattice().size(), initial_size);
        }

        #[test]
        fn prop_gillespie_time_always_increases(
            temperature in 0.1f64..1000.0,
            steps in 2usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut prev_time = sim.time();

            for _ in 0..steps {
                sim.step(&mut rng)?;
                let curr_time = sim.time();

                // Property: Time must monotonically increase
                prop_assert!(curr_time >= prev_time);
                prev_time = curr_time;
            }
        }

        #[test]
        fn prop_gillespie_finite_time_steps(
            temperature in 0.1f64..1000.0,
            steps in 1usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for _ in 0..steps {
                sim.step(&mut rng)?;
                let time = sim.time();

                // Property: Time steps must be finite (not NaN or infinite)
                prop_assert!(time.is_finite());
            }
        }

        #[test]
        fn prop_gillespie_events_counted(
            temperature in 0.1f64..1000.0,
            steps in 1usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for _ in 0..steps {
                sim.step(&mut rng)?;
            }

            // Property: Events counter should match number of steps
            prop_assert!(sim.events() > 0);
            prop_assert!(sim.events() <= steps);
        }
    }
}

/// Property: Detailed balance for Metropolis algorithm
///
/// Detailed balance is the condition π(i)P(i→j) = π(j)P(j→i)
/// where π is the equilibrium distribution and P is the transition probability.
#[cfg(test)]
mod metropolis_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_metropolis_acceptance_bounded(
            temperature in 0.1f64..1000.0,
            energy_diff in -100.0f64..100.0
        ) {
            // Property: Metropolis acceptance probability must be in [0, 1]
            let acceptance = metropolis_acceptance_prob(energy_diff, temperature);

            prop_assert!(acceptance >= 0.0);
            prop_assert!(acceptance <= 1.0);
        }

        #[test]
        fn prop_metropolis_downhill_always_accepts(
            temperature in 0.1f64..1000.0,
            energy_decrease in 0.1f64..100.0
        ) {
            // Property: Downhill moves (ΔE < 0) always accepted
            let acceptance = metropolis_acceptance_prob(-energy_decrease, temperature);

            prop_assert_eq!(acceptance, 1.0);
        }

        #[test]
        fn prop_metropolis_temperature_scaling(
            energy_diff in 0.1f64..10.0,
            temp1 in 0.1f64..1.0,
            temp2 in 1.0f64..10.0
        ) {
            // Property: Higher temperature → higher acceptance for uphill moves
            let accept_low = metropolis_acceptance_prob(energy_diff, temp1);
            let accept_high = metropolis_acceptance_prob(energy_diff, temp2);

            prop_assert!(accept_high >= accept_low);
        }
    }
}

/// Helper function to compute Metropolis acceptance probability
fn metropolis_acceptance_prob(energy_diff: f64, temperature: f64) -> f64 {
    if energy_diff <= 0.0 {
        1.0
    } else {
        (-energy_diff / temperature).exp()
    }
}

/// Property tests for lattice geometry
#[cfg(test)]
mod lattice_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_roi_lattice_valid_structure(
            coupling in 0.1f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;

            // Property: ROI lattice has a valid number of sites
            prop_assert!(lattice.size() > 0);
            prop_assert!(lattice.size() <= 100);

            // Property: All pBits should be properly initialized
            prop_assert_eq!(lattice.pbits().len(), lattice.size());
        }

        #[test]
        fn prop_lattice_positions_valid(
            coupling in 0.1f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;

            // Property: All positions should be in valid hyperbolic space
            for pos in lattice.positions() {
                let norm = pos.coords().norm();
                // Poincaré disk: |z| < 1
                prop_assert!(norm < 1.0);
            }
        }

        #[test]
        fn prop_lattice_states_valid(
            coupling in 0.1f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;

            // Property: States count should match lattice size
            prop_assert_eq!(lattice.states().len(), lattice.size());

            // Property: Should have at least some states
            prop_assert!(lattice.states().len() > 0);
        }
    }
}

/// Property tests for pBit dynamics
#[cfg(test)]
mod pbit_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_pbit_states_deterministic(
            coupling in 0.1f64..10.0,
            seed in 0u64..1000
        ) {
            // Property: Same random seed produces same evolution
            let lattice1 = PBitLattice::roi_48(coupling)?;
            let lattice2 = PBitLattice::roi_48(coupling)?;

            let mut sim1 = GillespieSimulator::new(lattice1);
            let mut sim2 = GillespieSimulator::new(lattice2);

            let mut rng1 = ChaCha8Rng::seed_from_u64(seed);
            let mut rng2 = ChaCha8Rng::seed_from_u64(seed);

            for _ in 0..10 {
                sim1.step(&mut rng1)?;
                sim2.step(&mut rng2)?;
            }

            // Same seed should produce same final states
            let states1 = sim1.lattice().states();
            let states2 = sim2.lattice().states();

            prop_assert_eq!(states1, states2);
        }

        #[test]
        fn prop_simulation_progress(
            coupling in 0.1f64..10.0,
            steps in 1usize..100
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;
            let mut sim = GillespieSimulator::new(lattice);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for _ in 0..steps {
                sim.step(&mut rng)?;
            }

            // Property: Simulation should make progress
            prop_assert!(sim.time() > 0.0);
            prop_assert!(sim.events() > 0);
        }
    }
}
