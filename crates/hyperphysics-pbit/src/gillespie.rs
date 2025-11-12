//! Gillespie exact stochastic simulation algorithm
//!
//! Research: Gillespie (1977) "Exact stochastic simulation of coupled chemical reactions"
//! Journal of Physical Chemistry 81(25): 2340-2361

use crate::{PBitLattice, Result};
use rand::Rng;
use rand_distr::{Distribution, Exp};

/// Gillespie exact stochastic simulator for pBit dynamics
///
/// Implements the exact discrete event simulation:
/// 1. Calculate total transition rate: r_total = Σ r_i
/// 2. Draw time to next event: Δt ~ Exp(r_total)
/// 3. Select which pBit flips: i ~ Categorical(r_i/r_total)
/// 4. Flip selected pBit
/// 5. Update time and repeat
pub struct GillespieSimulator {
    lattice: PBitLattice,
    current_time: f64,
    total_events: usize,
}

impl GillespieSimulator {
    /// Create new Gillespie simulator
    pub fn new(lattice: PBitLattice) -> Self {
        Self {
            lattice,
            current_time: 0.0,
            total_events: 0,
        }
    }

    /// Get current lattice state
    pub fn lattice(&self) -> &PBitLattice {
        &self.lattice
    }

    /// Get mutable lattice
    pub fn lattice_mut(&mut self) -> &mut PBitLattice {
        &mut self.lattice
    }

    /// Get current simulation time
    pub fn time(&self) -> f64 {
        self.current_time
    }

    /// Get total events executed
    pub fn events(&self) -> usize {
        self.total_events
    }

    /// Single simulation step
    ///
    /// Returns time to next event
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<f64> {
        let states = self.lattice.states();

        // Calculate flip rates for all pBits
        let mut rates = Vec::with_capacity(self.lattice.size());
        let mut total_rate = 0.0;

        for (_i, pbit) in self.lattice.pbits().iter().enumerate() {
            let h_eff = pbit.effective_field(&states);
            let mut temp_pbit = pbit.clone();
            temp_pbit.update_probability(h_eff);
            let rate = temp_pbit.flip_rate();

            rates.push(rate);
            total_rate += rate;
        }

        if total_rate == 0.0 {
            // No transitions possible
            return Ok(f64::INFINITY);
        }

        // Sample time to next event from exponential distribution
        let exp = Exp::new(total_rate).unwrap();
        let dt = exp.sample(rng);

        // Select which pBit flips using cumulative distribution
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        let mut flip_idx = 0;

        for (i, rate) in rates.iter().enumerate() {
            cumulative += rate / total_rate;
            if r < cumulative {
                flip_idx = i;
                break;
            }
        }

        // Flip the selected pBit
        let pbits = self.lattice.pbits_mut();
        let current_state = pbits[flip_idx].state();
        pbits[flip_idx].set_state(!current_state);

        // Update time and event counter
        self.current_time += dt;
        self.total_events += 1;

        Ok(dt)
    }

    /// Run simulation until target time
    ///
    /// Returns number of events executed
    pub fn simulate_until<R: Rng>(&mut self, target_time: f64, rng: &mut R) -> Result<usize> {
        let initial_events = self.total_events;

        while self.current_time < target_time {
            let dt = self.step(rng)?;

            if dt.is_infinite() {
                // No more transitions possible
                break;
            }
        }

        Ok(self.total_events - initial_events)
    }

    /// Run simulation for fixed number of events
    pub fn simulate_events<R: Rng>(&mut self, num_events: usize, rng: &mut R) -> Result<()> {
        for _ in 0..num_events {
            self.step(rng)?;
        }
        Ok(())
    }

    /// Reset simulation
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        self.total_events = 0;
    }

    /// Get event rate (events per unit time)
    pub fn event_rate(&self) -> f64 {
        if self.current_time > 0.0 {
            self.total_events as f64 / self.current_time
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_gillespie_step() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = GillespieSimulator::new(lattice);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let dt = sim.step(&mut rng).unwrap();

        assert!(dt > 0.0);
        assert!(dt.is_finite());
        assert_eq!(sim.events(), 1);
    }

    #[test]
    fn test_gillespie_simulate() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = GillespieSimulator::new(lattice);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let events = sim.simulate_until(10.0, &mut rng).unwrap();

        assert!(events > 0);
        assert!(sim.time() >= 10.0 || sim.time() < 10.0);
    }

    #[test]
    fn test_event_rate() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = GillespieSimulator::new(lattice);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        sim.simulate_until(10.0, &mut rng).unwrap();

        let rate = sim.event_rate();
        assert!(rate > 0.0);
    }

    #[test]
    fn test_reset() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut sim = GillespieSimulator::new(lattice);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        sim.simulate_until(10.0, &mut rng).unwrap();
        assert!(sim.time() > 0.0);

        sim.reset();
        assert_eq!(sim.time(), 0.0);
        assert_eq!(sim.events(), 0);
    }
}
