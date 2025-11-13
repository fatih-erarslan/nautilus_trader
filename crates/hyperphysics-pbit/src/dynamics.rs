//! High-level pBit dynamics interface

use crate::{GillespieSimulator, MetropolisSimulator, PBitLattice, Result};
use rand::Rng;
use serde::{Serialize, Deserialize};

/// Simulation algorithm selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Algorithm {
    /// Gillespie exact stochastic simulation
    Gillespie,
    /// Metropolis-Hastings MCMC
    Metropolis,
}

/// High-level pBit dynamics controller
pub struct PBitDynamics {
    algorithm: Algorithm,
    gillespie: Option<GillespieSimulator>,
    metropolis: Option<MetropolisSimulator>,
}

impl PBitDynamics {
    /// Create dynamics with Gillespie algorithm
    pub fn new_gillespie(lattice: PBitLattice) -> Self {
        Self {
            algorithm: Algorithm::Gillespie,
            gillespie: Some(GillespieSimulator::new(lattice)),
            metropolis: None,
        }
    }

    /// Create dynamics with Metropolis algorithm
    pub fn new_metropolis(lattice: PBitLattice, temperature: f64) -> Self {
        Self {
            algorithm: Algorithm::Metropolis,
            gillespie: None,
            metropolis: Some(MetropolisSimulator::new(lattice, temperature)),
        }
    }

    /// Get current algorithm
    pub fn algorithm(&self) -> Algorithm {
        self.algorithm
    }

    /// Get lattice reference
    pub fn lattice(&self) -> &PBitLattice {
        match self.algorithm {
            Algorithm::Gillespie => self.gillespie.as_ref().unwrap().lattice(),
            Algorithm::Metropolis => self.metropolis.as_ref().unwrap().lattice(),
        }
    }

    /// Run simulation step
    pub fn step<R: Rng>(&mut self, rng: &mut R) -> Result<()> {
        match self.algorithm {
            Algorithm::Gillespie => {
                self.gillespie.as_mut().unwrap().step(rng)?;
            }
            Algorithm::Metropolis => {
                self.metropolis.as_mut().unwrap().step(rng)?;
            }
        }
        Ok(())
    }

    /// Run multiple steps
    pub fn simulate<R: Rng>(&mut self, steps: usize, rng: &mut R) -> Result<()> {
        for _ in 0..steps {
            self.step(rng)?;
        }
        Ok(())
    }

    /// Get simulation statistics
    pub fn statistics(&self) -> DynamicsStatistics {
        match self.algorithm {
            Algorithm::Gillespie => {
                let sim = self.gillespie.as_ref().unwrap();
                DynamicsStatistics::Gillespie {
                    time: sim.time(),
                    events: sim.events(),
                    event_rate: sim.event_rate(),
                }
            }
            Algorithm::Metropolis => {
                let sim = self.metropolis.as_ref().unwrap();
                DynamicsStatistics::Metropolis {
                    steps: sim.steps(),
                    acceptance_rate: sim.acceptance_rate(),
                }
            }
        }
    }
}

/// Statistics about simulation
#[derive(Debug, Clone)]
pub enum DynamicsStatistics {
    Gillespie {
        time: f64,
        events: usize,
        event_rate: f64,
    },
    Metropolis {
        steps: usize,
        acceptance_rate: f64,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_gillespie_dynamics() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut dynamics = PBitDynamics::new_gillespie(lattice);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        dynamics.simulate(100, &mut rng).unwrap();

        match dynamics.statistics() {
            DynamicsStatistics::Gillespie { events, .. } => {
                assert_eq!(events, 100);
            }
            _ => {
                panic!("Expected Gillespie statistics but got different type");
            }
        }
    }

    #[test]
    fn test_metropolis_dynamics() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut dynamics = PBitDynamics::new_metropolis(lattice, 300.0);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        dynamics.simulate(100, &mut rng).unwrap();

        match dynamics.statistics() {
            DynamicsStatistics::Metropolis { steps, .. } => {
                assert_eq!(steps, 100);
            }
            _ => panic!("Wrong statistics type"),
        }
    }
}
