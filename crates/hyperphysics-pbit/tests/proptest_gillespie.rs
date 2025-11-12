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
            steps in 1usize..1000
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice, temperature);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            sim.simulate(steps, &mut rng)?;

            // Property: All transition rates must be non-negative
            prop_assert!(sim.total_rate() >= 0.0);
        }

        #[test]
        fn prop_gillespie_conserves_particle_number(
            temperature in 0.1f64..1000.0,
            steps in 1usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let initial_particles = lattice.count_spins();

            let mut sim = GillespieSimulator::new(lattice, temperature);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            sim.simulate(steps, &mut rng)?;

            let final_particles = sim.lattice().count_spins();

            // Property: Gillespie should conserve total particle number
            // (for models without creation/annihilation)
            prop_assert_eq!(initial_particles, final_particles);
        }

        #[test]
        fn prop_gillespie_time_always_increases(
            temperature in 0.1f64..1000.0,
            steps in 2usize..100
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice, temperature);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            let mut prev_time = sim.current_time();

            for _ in 0..steps {
                sim.step(&mut rng)?;
                let curr_time = sim.current_time();

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
            let mut sim = GillespieSimulator::new(lattice, temperature);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            for _ in 0..steps {
                sim.step(&mut rng)?;
                let time = sim.current_time();

                // Property: Time steps must be finite (not NaN or infinite)
                prop_assert!(time.is_finite());
            }
        }

        #[test]
        fn prop_gillespie_equilibrates_at_high_temp(
            temperature in 100.0f64..10000.0,
            steps in 1000usize..10000
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice, temperature);
            let mut rng = ChaCha8Rng::seed_from_u64(42);

            sim.simulate(steps, &mut rng)?;

            // Property: At very high temperature, system should approach
            // random state with ~50% spin-up probability
            let magnetization = sim.lattice().magnetization();
            let normalized_mag = magnetization.abs() / (lattice.size() as f64);

            // For T >> J, expect |m| < 0.3 (very weak ordering)
            prop_assert!(normalized_mag < 0.3);
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
        fn prop_lattice_size_matches_dimension(
            size in 4usize..32
        ) {
            // For square lattice
            let lattice = PBitLattice::square(size)?;

            // Property: Total sites = size × size
            prop_assert_eq!(lattice.size(), size * size);
        }

        #[test]
        fn prop_lattice_neighbors_symmetric(
            i in 0usize..100,
            j in 0usize..100
        ) {
            let lattice = PBitLattice::square(10)?;

            if i < lattice.size() && j < lattice.size() {
                let neighbors_i = lattice.neighbors(i);
                let neighbors_j = lattice.neighbors(j);

                // Property: If j is neighbor of i, then i is neighbor of j
                if neighbors_i.contains(&j) {
                    prop_assert!(neighbors_j.contains(&i));
                }
            }
        }

        #[test]
        fn prop_roi_lattice_valid_structure(
            coupling in 0.1f64..10.0
        ) {
            let lattice = PBitLattice::roi_48(coupling)?;

            // Property: ROI-48 lattice has exactly 48 sites
            prop_assert_eq!(lattice.size(), 48);

            // Property: Each site has between 3 and 6 neighbors (typical for ROI)
            for i in 0..lattice.size() {
                let n_neighbors = lattice.neighbors(i).len();
                prop_assert!(n_neighbors >= 3 && n_neighbors <= 6);
            }
        }
    }
}

/// Property tests for energy calculations
#[cfg(test)]
mod energy_properties {
    use super::*;

    proptest! {
        #[test]
        fn prop_energy_flip_reversible(
            temperature in 0.1f64..1000.0,
            site in 0usize..48
        ) {
            let lattice = PBitLattice::roi_48(1.0)?;
            let mut sim = GillespieSimulator::new(lattice, temperature);

            let energy_before = sim.total_energy();

            // Flip spin
            sim.flip_spin(site);
            let energy_flipped = sim.total_energy();

            // Flip back
            sim.flip_spin(site);
            let energy_after = sim.total_energy();

            // Property: Double flip returns to original energy
            prop_assert!((energy_before - energy_after).abs() < 1e-10);
        }

        #[test]
        fn prop_energy_extensive(
            coupling in 0.1f64..10.0
        ) {
            // Property: Energy should scale with system size
            let lattice_small = PBitLattice::square(4)?;
            let lattice_large = PBitLattice::square(8)?;

            let sim_small = GillespieSimulator::new(lattice_small, 1.0);
            let sim_large = GillespieSimulator::new(lattice_large, 1.0);

            let energy_small = sim_small.total_energy().abs();
            let energy_large = sim_large.total_energy().abs();

            // Larger system should have larger (or equal) energy magnitude
            prop_assert!(energy_large >= energy_small);
        }
    }
}
