//! Ising Machine Optimizer using p-bit Networks
//!
//! This module maps NP-hard portfolio optimization problems to Ising Hamiltonians
//! and solves them using probabilistic bit (p-bit) networks and advanced optimization
//! algorithms including Simulated Bifurcation and Parallel Tempering.
//!
//! # Mathematical Foundation
//!
//! Portfolio optimization as Ising model:
//! H = -Σᵢⱼ Jᵢⱼ sᵢsⱼ - Σᵢ hᵢsᵢ
//!
//! where:
//! - sᵢ ∈ {-1, +1} represents asset selection
//! - Jᵢⱼ encodes correlation (coupling)
//! - hᵢ encodes expected return (field)
//!
//! p-bits minimize H via stochastic sampling at microsecond timescales.

use nalgebra as na;
use rand::Rng;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

// Module exports
pub mod parallel_tempering;
pub mod simulated_bifurcation;

/// Ising Hamiltonian for portfolio optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsingHamiltonian {
    /// Coupling matrix J (correlations)
    pub coupling: na::DMatrix<f64>,
    /// External field h (expected returns)
    pub field: na::DVector<f64>,
    /// Number of assets
    pub num_assets: usize,
}

impl IsingHamiltonian {
    /// Create Hamiltonian from portfolio data
    pub fn from_portfolio(
        correlation_matrix: &na::DMatrix<f64>,
        expected_returns: &na::DVector<f64>,
        risk_aversion: f64,
    ) -> Self {
        let num_assets = expected_returns.len();

        // J = -risk_aversion * Correlation (we want to minimize risk)
        let coupling = correlation_matrix * (-risk_aversion);

        // h = expected_returns (we want to maximize returns)
        let field = expected_returns.clone();

        Self {
            coupling,
            field,
            num_assets,
        }
    }

    /// Compute energy of a configuration
    pub fn energy(&self, spins: &na::DVector<f64>) -> f64 {
        // H = -Σᵢⱼ Jᵢⱼ sᵢsⱼ - Σᵢ hᵢsᵢ

        let interaction_energy = -spins.transpose() * &self.coupling * spins;
        let field_energy = -self.field.dot(spins);

        interaction_energy[0] + field_energy
    }
}

/// p-bit network solver
pub struct PbitIsingMachine {
    hamiltonian: IsingHamiltonian,
    spins: na::DVector<f64>,
    temperature: f64,
    num_iterations: usize,
}

impl PbitIsingMachine {
    /// Create a new p-bit Ising machine
    pub fn new(hamiltonian: IsingHamiltonian, temperature: f64) -> Self {
        let num_assets = hamiltonian.num_assets;

        // Initialize spins randomly
        let mut rng = rand::thread_rng();
        let spins =
            na::DVector::from_fn(
                num_assets,
                |_, _| {
                    if rng.gen::<f64>() > 0.5 {
                        1.0
                    } else {
                        -1.0
                    }
                },
            );

        Self {
            hamiltonian,
            spins,
            temperature,
            num_iterations: 1000,
        }
    }

    /// Run simulated annealing to find optimal portfolio
    pub fn solve(&mut self) -> na::DVector<f64> {
        info!(
            "Solving Ising model for {} assets",
            self.hamiltonian.num_assets
        );

        let mut rng = rand::thread_rng();
        let mut current_energy = self.hamiltonian.energy(&self.spins);
        let mut best_spins = self.spins.clone();
        let mut best_energy = current_energy;

        for iter in 0..self.num_iterations {
            // Annealing schedule
            let temp = self.temperature * (1.0 - iter as f64 / self.num_iterations as f64);

            // Flip a random spin
            let flip_idx = rng.gen_range(0..self.hamiltonian.num_assets);
            self.spins[flip_idx] *= -1.0;

            let new_energy = self.hamiltonian.energy(&self.spins);
            let delta_energy = new_energy - current_energy;

            // Metropolis acceptance criterion
            let accept = if delta_energy < 0.0 {
                true
            } else {
                let prob = (-delta_energy / temp).exp();
                rng.gen::<f64>() < prob
            };

            if accept {
                current_energy = new_energy;
                if current_energy < best_energy {
                    best_energy = current_energy;
                    best_spins = self.spins.clone();
                    debug!("New best energy: {:.4} at iteration {}", best_energy, iter);
                }
            } else {
                // Reject: flip back
                self.spins[flip_idx] *= -1.0;
            }
        }

        info!("Optimization complete. Best energy: {:.4}", best_energy);
        best_spins
    }

    /// Convert spin configuration to portfolio weights
    pub fn spins_to_weights(&self, spins: &na::DVector<f64>) -> na::DVector<f64> {
        // Map {-1, +1} to [0, 1] and normalize
        let positive_spins = spins.map(|s| (s + 1.0) / 2.0);
        let sum = positive_spins.sum();

        if sum > 1e-10 {
            positive_spins / sum
        } else {
            na::DVector::from_element(spins.len(), 1.0 / spins.len() as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ising_hamiltonian() {
        let corr =
            na::DMatrix::from_row_slice(3, 3, &[1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0]);

        let returns = na::DVector::from_vec(vec![0.1, 0.15, 0.08]);

        let hamiltonian = IsingHamiltonian::from_portfolio(&corr, &returns, 1.0);

        let spins = na::DVector::from_vec(vec![1.0, 1.0, -1.0]);
        let energy = hamiltonian.energy(&spins);

        assert!(energy.is_finite());
    }

    #[test]
    fn test_pbit_solver() {
        let corr =
            na::DMatrix::from_row_slice(3, 3, &[1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0]);

        let returns = na::DVector::from_vec(vec![0.1, 0.15, 0.08]);

        let hamiltonian = IsingHamiltonian::from_portfolio(&corr, &returns, 1.0);
        let mut solver = PbitIsingMachine::new(hamiltonian, 1.0);

        let solution = solver.solve();
        let weights = solver.spins_to_weights(&solution);

        // Weights should sum to 1
        assert!((weights.sum() - 1.0).abs() < 1e-6);

        // All weights should be non-negative
        assert!(weights.iter().all(|&w| w >= 0.0));
    }
}
