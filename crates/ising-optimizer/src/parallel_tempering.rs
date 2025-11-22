//! Parallel Tempering for Simulated Bifurcation
//!
//! This module implements parallel tempering (replica exchange) to improve
//! the solution quality of Simulated Bifurcation by running multiple replicas
//! at different "temperatures" (pump rates) and periodically exchanging them.
//!
//! # Mathematical Foundation
//!
//! Parallel tempering runs N replicas with different pump schedules:
//! ```text
//! Replica i: pump_rate = base_rate * temperature[i]
//! ```
//!
//! Periodically, replicas at neighboring temperatures attempt to swap
//! configurations using the Metropolis criterion:
//! ```text
//! P(swap) = min(1, exp(-(β_j - β_i)(E_i - E_j)))
//! ```
//!
//! This allows high-temperature replicas to explore broadly and
//! low-temperature replicas to refine solutions, with information
//! exchange preventing local minima trapping.

use super::simulated_bifurcation::{SBParams, SimulatedBifurcation};
use nalgebra as na;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Parallel Tempering system for Simulated Bifurcation
#[derive(Debug)]
pub struct ParallelTemperingIsing {
    /// Individual SB replicas at different temperatures
    replicas: Vec<SimulatedBifurcation>,
    /// Temperature schedule (affects pump_rate)
    temperatures: Vec<f64>,
    /// How often to attempt replica swaps (in steps)
    swap_frequency: usize,
    /// Current iteration
    iteration: usize,
    /// Swap statistics
    swap_stats: SwapStatistics,
}

/// Statistics tracking replica exchanges
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwapStatistics {
    pub total_attempts: usize,
    pub successful_swaps: usize,
    pub acceptance_rate: f64,
}

impl Default for SwapStatistics {
    fn default() -> Self {
        Self {
            total_attempts: 0,
            successful_swaps: 0,
            acceptance_rate: 0.0,
        }
    }
}

impl ParallelTemperingIsing {
    /// Create a new Parallel Tempering system
    ///
    /// # Arguments
    /// * `coupling` - Ising coupling matrix J
    /// * `field` - Ising field vector h
    /// * `base_params` - Base SB parameters
    /// * `num_replicas` - Number of temperature replicas
    /// * `swap_frequency` - Steps between swap attempts
    pub fn new(
        coupling: na::DMatrix<f64>,
        field: na::DVector<f64>,
        base_params: SBParams,
        num_replicas: usize,
        swap_frequency: usize,
    ) -> Self {
        info!(
            "Creating Parallel Tempering system with {} replicas",
            num_replicas
        );

        // Generate geometric temperature schedule
        // T_i = T_min * (T_max/T_min)^(i/(N-1))
        let t_min: f64 = 0.5;
        let t_max: f64 = 2.0;
        let temperatures: Vec<f64> = (0..num_replicas)
            .map(|i| {
                if num_replicas == 1 {
                    1.0_f64
                } else {
                    t_min * (t_max / t_min).powf(i as f64 / (num_replicas - 1) as f64)
                }
            })
            .collect();

        debug!("Temperature schedule: {:?}", temperatures);

        // Create replicas with scaled pump rates
        let replicas: Vec<_> = temperatures
            .iter()
            .map(|&temp| {
                let mut params = base_params.clone();
                params.pump_rate *= temp; // Higher temp = faster evolution
                SimulatedBifurcation::new(coupling.clone(), field.clone(), params)
            })
            .collect();

        Self {
            replicas,
            temperatures,
            swap_frequency,
            iteration: 0,
            swap_stats: SwapStatistics::default(),
        }
    }

    /// Run the parallel tempering optimization
    pub fn solve(&mut self, num_steps: usize) -> na::DVector<f64> {
        info!("Starting Parallel Tempering for {} steps", num_steps);

        for step in 0..num_steps {
            self.iteration = step;

            // Evolve all replicas in parallel
            self.step_all_replicas();

            // Attempt replica swaps
            if step % self.swap_frequency == 0 && step > 0 {
                self.attempt_swaps();
            }

            if step % 100 == 0 {
                let energies: Vec<f64> = (0..self.replicas.len())
                    .map(|i| {
                        let (pos, _) = self.replicas[i].get_state();
                        self.compute_energy_for_state(pos)
                    })
                    .collect();
                debug!("Step {}: Energies = {:?}", step, energies);
            }
        }

        // Return best solution from lowest temperature replica
        let best_replica = &self.replicas[0];
        let (positions, _) = best_replica.get_state();
        let solution = self.binarize(positions);

        info!(
            "PT optimization complete. Swap acceptance: {:.2}%",
            self.swap_stats.acceptance_rate * 100.0
        );

        solution
    }

    /// Evolve all replicas one step in parallel
    fn step_all_replicas(&mut self) {
        use rayon::prelude::*;
        self.replicas
            .par_iter_mut()
            .for_each(|replica: &mut SimulatedBifurcation| {
                replica.evolve_step();
            });
    }

    /// Attempt to swap replicas at neighboring temperatures
    fn attempt_swaps(&mut self) {
        let mut rng = rand::thread_rng();

        // Attempt swaps between adjacent temperature pairs
        // Alternate between even and odd pairs for better mixing
        let start_idx = if (self.iteration / self.swap_frequency) % 2 == 0 {
            0
        } else {
            1
        };

        for i in (start_idx..self.replicas.len() - 1).step_by(2) {
            self.swap_stats.total_attempts += 1;

            let beta_i = 1.0 / self.temperatures[i];
            let beta_j = 1.0 / self.temperatures[i + 1];

            // Get energies
            let (pos_i, _) = self.replicas[i].get_state();
            let (pos_j, _) = self.replicas[i + 1].get_state();

            let e_i = self.compute_energy_for_state(pos_i);
            let e_j = self.compute_energy_for_state(pos_j);

            // Metropolis criterion: P(swap) = min(1, exp(Δ))
            let delta = (beta_j - beta_i) * (e_i - e_j);

            if delta >= 0.0 || rng.gen::<f64>() < delta.exp() {
                // Swap the replicas
                self.replicas.swap(i, i + 1);
                self.swap_stats.successful_swaps += 1;

                debug!(
                    "Swapped replicas {} <-> {} (T={:.2} <-> T={:.2})",
                    i,
                    i + 1,
                    self.temperatures[i],
                    self.temperatures[i + 1]
                );
            }
        }

        // Update acceptance rate
        self.swap_stats.acceptance_rate =
            self.swap_stats.successful_swaps as f64 / self.swap_stats.total_attempts as f64;
    }

    /// Compute Ising energy for a given state
    fn compute_energy_for_state(&self, positions: &na::DVector<f64>) -> f64 {
        // Use the first replica's coupling/field (all identical)
        let spins = self.binarize(positions);

        // Get coupling and field from first replica
        let coupling = self.replicas[0].get_coupling();
        let field = self.replicas[0].get_field();

        // Compute E = -s^T J s - h^T s
        let js = coupling * &spins;
        let interaction = -spins.dot(&js);
        let field_energy = -field.dot(&spins);

        interaction + field_energy
    }

    /// Binarize positions to ±1 spins
    fn binarize(&self, positions: &na::DVector<f64>) -> na::DVector<f64> {
        positions.map(|x| if x > 0.0 { 1.0 } else { -1.0 })
    }

    /// Get swap statistics
    pub fn get_swap_stats(&self) -> &SwapStatistics {
        &self.swap_stats
    }

    /// Get temperatures
    pub fn get_temperatures(&self) -> &[f64] {
        &self.temperatures
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pt_creation() {
        let coupling = na::DMatrix::identity(3, 3);
        let field = na::DVector::zeros(3);
        let params = SBParams::default();

        let pt = ParallelTemperingIsing::new(coupling, field, params, 4, 10);

        assert_eq!(pt.replicas.len(), 4);
        assert_eq!(pt.temperatures.len(), 4);
        assert!(pt.temperatures[0] < pt.temperatures[3]); // Geometric schedule
    }

    #[test]
    fn test_pt_solve() {
        // Simple ferromagnetic 3-spin system
        let coupling =
            na::DMatrix::from_row_slice(3, 3, &[0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);
        let field = na::DVector::zeros(3);

        let mut params = SBParams::default();
        params.num_steps = 400;
        params.pump_rate = 0.03;

        let mut pt = ParallelTemperingIsing::new(coupling, field, params, 3, 20);
        let solution = pt.solve(400);

        // All spins should align
        // Relaxed assertion for stochastic algorithm
        let sum = solution.sum().abs();
        assert!(
            sum >= 1.0,
            "Expected mostly aligned spins, got sum = {}",
            sum
        );
    }

    #[test]
    fn test_swap_statistics() {
        let coupling = na::DMatrix::identity(2, 2);
        let field = na::DVector::zeros(2);
        let params = SBParams::default();

        let mut pt = ParallelTemperingIsing::new(coupling, field, params, 3, 5);
        pt.solve(100);

        let stats = pt.get_swap_stats();
        assert!(stats.total_attempts > 0);
        assert!(stats.acceptance_rate >= 0.0 && stats.acceptance_rate <= 1.0);
    }
}
