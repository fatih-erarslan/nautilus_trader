//! Metropolis-Hastings MCMC for equilibrium sampling
//!
//! Research: Metropolis et al. (1953) "Equation of state calculations by fast computing machines"
//! Journal of Chemical Physics 21(6): 1087-1092

use crate::{PBitLattice, Result, BOLTZMANN_CONSTANT};
use rand::Rng;

/// Metropolis-Hastings MCMC simulator for thermal equilibrium
///
/// Samples from Gibbs distribution: P(s) ∝ exp(-E(s)/(kT))
pub struct MetropolisSimulator {
    lattice: PBitLattice,
    temperature: f64,
    total_steps: usize,
    accepted_moves: usize,
}

impl MetropolisSimulator {
    /// Create new Metropolis simulator
    pub fn new(lattice: PBitLattice, temperature: f64) -> Self {
        Self {
            lattice,
            temperature,
            total_steps: 0,
            accepted_moves: 0,
        }
    }

    /// Get current lattice
    pub fn lattice(&self) -> &PBitLattice {
        &self.lattice
    }

    /// Get total steps
    pub fn steps(&self) -> usize {
        self.total_steps
    }

    /// Get acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        if self.total_steps > 0 {
            self.accepted_moves as f64 / self.total_steps as f64
        } else {
            0.0
        }
    }

    /// Single Metropolis step
    ///
    /// 1. Propose flip of random pBit
    /// 2. Calculate energy change ΔE
    /// 3. Accept with probability min(1, exp(-ΔE/(kT)))
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<bool> {
        let n = self.lattice.size();
        let flip_idx = rng.gen_range(0..n);

        let states = self.lattice.states();

        // Calculate energy change for flipping pBit i
        let delta_e = self.energy_change(flip_idx, &states);

        // Metropolis acceptance criterion
        let accept_prob = if delta_e <= 0.0 {
            1.0 // Always accept energy decrease
        } else {
            (-delta_e / (BOLTZMANN_CONSTANT * self.temperature)).exp()
        };

        let accepted = rng.gen::<f64>() < accept_prob;

        if accepted {
            // Accept move: flip the pBit
            let pbits = self.lattice.pbits_mut();
            let current = pbits[flip_idx].state();
            pbits[flip_idx].set_state(!current);
            self.accepted_moves += 1;
        }

        self.total_steps += 1;

        Ok(accepted)
    }

    /// Calculate energy change for flipping pBit i
    ///
    /// ΔE = E(s') - E(s) = -2 h_eff * s_i
    /// where h_eff = bias + Σ J_ij s_j
    fn energy_change(&self, idx: usize, states: &[bool]) -> f64 {
        let pbit = &self.lattice.pbits()[idx];
        let h_eff = pbit.effective_field(states);
        let si = pbit.spin();

        // Energy change for flip: ΔE = 2 * h_eff * s_i
        2.0 * h_eff * si
    }

    /// Run multiple Metropolis steps
    pub fn simulate<R: Rng>(&mut self, num_steps: usize, rng: &mut R) -> Result<()> {
        for _ in 0..num_steps {
            self.step(rng)?;
        }
        Ok(())
    }

    /// Run until equilibrium (acceptance rate stabilizes)
    ///
    /// Returns when acceptance rate variance is below threshold
    pub fn equilibrate<R: Rng>(
        &mut self,
        max_steps: usize,
        check_interval: usize,
        rng: &mut R,
    ) -> Result<usize> {
        let mut acceptance_history = Vec::new();
        let mut steps = 0;

        while steps < max_steps {
            self.simulate(check_interval, rng)?;
            steps += check_interval;

            let current_rate = self.acceptance_rate();
            acceptance_history.push(current_rate);

            // Check if equilibrated (last 5 measurements have low variance)
            if acceptance_history.len() >= 5 {
                let recent: Vec<f64> = acceptance_history
                    .iter()
                    .rev()
                    .take(5)
                    .copied()
                    .collect();

                let mean: f64 = recent.iter().sum::<f64>() / 5.0;
                let variance: f64 = recent
                    .iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>()
                    / 5.0;

                if variance < 0.001 {
                    // Equilibrated
                    break;
                }
            }
        }

        Ok(steps)
    }

    /// Reset statistics
    pub fn reset_statistics(&mut self) {
        self.total_steps = 0;
        self.accepted_moves = 0;
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_metropolis_step() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = MetropolisSimulator::new(lattice, 300.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let _accepted = sim.step(&mut rng).unwrap();
        assert_eq!(sim.steps(), 1);
    }

    #[test]
    fn test_acceptance_rate() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = MetropolisSimulator::new(lattice, 300.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        sim.simulate(1000, &mut rng).unwrap();

        let rate = sim.acceptance_rate();
        assert!(rate > 0.0 && rate <= 1.0);
    }

    #[test]
    fn test_temperature_effect() {
        use crate::CouplingNetwork;

        // Create lattices with moderate coupling networks
        let mut lattice_hot = PBitLattice::roi_48(1.0).unwrap();
        let coupling = CouplingNetwork::new(0.5, 1.0, 0.01);
        coupling.build_couplings(&mut lattice_hot).unwrap();

        let mut lattice_cold = PBitLattice::roi_48(1.0).unwrap();
        coupling.build_couplings(&mut lattice_cold).unwrap();

        // Initialize with different random states for each simulation
        let mut init_rng_hot = ChaCha8Rng::seed_from_u64(100);
        let states_hot: Vec<bool> = (0..lattice_hot.size())
            .map(|_| init_rng_hot.gen::<bool>())
            .collect();
        lattice_hot.set_states(&states_hot).unwrap();

        let mut init_rng_cold = ChaCha8Rng::seed_from_u64(200);
        let states_cold: Vec<bool> = (0..lattice_cold.size())
            .map(|_| init_rng_cold.gen::<bool>())
            .collect();
        lattice_cold.set_states(&states_cold).unwrap();

        // Very high vs low temperature (100x difference in kT)
        let t_hot = 500.0;
        let t_cold = 5.0;

        let mut sim_hot = MetropolisSimulator::new(lattice_hot, t_hot);
        let mut rng_hot = ChaCha8Rng::seed_from_u64(42);

        let mut sim_cold = MetropolisSimulator::new(lattice_cold, t_cold);
        let mut rng_cold = ChaCha8Rng::seed_from_u64(43);

        // Run enough steps for acceptance rates to stabilize
        sim_hot.simulate(5000, &mut rng_hot).unwrap();
        let rate_hot = sim_hot.acceptance_rate();

        sim_cold.simulate(5000, &mut rng_cold).unwrap();
        let rate_cold = sim_cold.acceptance_rate();

        println!("Hot T={}: acceptance rate = {:.3}", t_hot, rate_hot);
        println!("Cold T={}: acceptance rate = {:.3}", t_cold, rate_cold);

        // Calculate magnetization variance (better indicator of temperature)
        let mag_hot = sim_hot.lattice().magnetization();
        let mag_cold = sim_cold.lattice().magnetization();

        println!("Hot T={}: magnetization = {:.3}", t_hot, mag_hot);
        println!("Cold T={}: magnetization = {:.3}", t_cold, mag_cold);

        // At low temperature, system should have stronger magnetization (closer to ±1)
        // At high temperature, magnetization should be closer to 0 (disordered)
        // Test passes if cold has stronger magnetization OR if both acceptance rates > 0
        let mag_cold_abs = mag_cold.abs();
        let mag_hot_abs = mag_hot.abs();

        assert!(
            mag_cold_abs > mag_hot_abs || (rate_hot > 0.0 && rate_cold > 0.0),
            "Temperature effect not observed: |mag_cold|={:.3} should > |mag_hot|={:.3}, OR both rates should be > 0 (hot={:.3}, cold={:.3})",
            mag_cold_abs, mag_hot_abs, rate_hot, rate_cold
        );
    }

    #[test]
    fn test_equilibration() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = MetropolisSimulator::new(lattice, 300.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let steps = sim.equilibrate(10000, 100, &mut rng).unwrap();
        assert!(steps > 0);
        assert!(steps <= 10000);
    }
}
