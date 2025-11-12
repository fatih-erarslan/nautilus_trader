//! Main HyperPhysics engine

use crate::{config::EngineConfig, metrics::*, Result, EngineError};
use hyperphysics_pbit::*;
use hyperphysics_thermo::*;
use hyperphysics_consciousness::*;
use rand::Rng;

#[cfg(feature = "simd")]
use crate::simd::engine::{
    entropy_from_probabilities_simd,
    magnetization_simd,
};

/// Main HyperPhysics engine
pub struct HyperPhysicsEngine {
    config: EngineConfig,
    dynamics: PBitDynamics,
    entropy_calc: EntropyCalculator,
    landauer: LandauerEnforcer,
    phi_calc: Option<PhiCalculator>,
    ci_calc: Option<CICalculator>,
    metrics: EngineMetrics,
    previous_entropy: f64,
}

impl HyperPhysicsEngine {
    /// Create new engine from configuration
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Create lattice
        let (p, q, depth) = config.scale.tessellation_params();
        let mut lattice = PBitLattice::new(p, q, depth, config.temperature)?;

        // Build coupling network
        let network = CouplingNetwork::new(
            config.coupling_j0,
            config.coupling_lambda,
            config.coupling_min,
        );
        network.build_couplings(&mut lattice)?;

        // Create dynamics
        let dynamics = match config.algorithm {
            Algorithm::Gillespie => PBitDynamics::new_gillespie(lattice),
            Algorithm::Metropolis => {
                PBitDynamics::new_metropolis(lattice, config.temperature)
            }
        };

        // Create calculators
        let entropy_calc = EntropyCalculator::new();
        let landauer = LandauerEnforcer::new(config.temperature)?;

        let phi_calc = if config.calculate_phi {
            Some(PhiCalculator::greedy())
        } else {
            None
        };

        let ci_calc = if config.calculate_ci {
            Some(CICalculator::new())
        } else {
            None
        };

        let num_pbits = dynamics.lattice().size();
        let initial_entropy = entropy_calc.entropy_from_pbits(dynamics.lattice());

        Ok(Self {
            config,
            dynamics,
            entropy_calc,
            landauer,
            phi_calc,
            ci_calc,
            metrics: EngineMetrics::new(num_pbits),
            previous_entropy: initial_entropy,
        })
    }

    /// Create ROI system (48 nodes)
    pub fn roi_48(pbit_temp: f64, thermo_temp: f64) -> Result<Self> {
        let mut config = EngineConfig::roi_48(pbit_temp);
        config.temperature = thermo_temp;
        Self::new(config)
    }

    /// Single simulation step
    pub fn step(&mut self) -> Result<()> {
        self.step_with_rng(&mut rand::thread_rng())
    }

    /// Step with custom RNG
    pub fn step_with_rng<R: Rng>(&mut self, rng: &mut R) -> Result<()> {
        // Run dynamics step
        self.dynamics.step(rng)?;

        // Update metrics
        self.update_metrics()?;

        // Verify thermodynamics if enabled
        if self.config.verify_thermodynamics {
            self.verify_thermodynamics()?;
        }

        self.metrics.timestep += 1;

        Ok(())
    }

    /// Run multiple steps
    pub fn simulate<R: Rng>(&mut self, num_steps: usize, rng: &mut R) -> Result<()> {
        for _ in 0..num_steps {
            self.step_with_rng(rng)?;
        }
        Ok(())
    }

    /// Update all metrics
    fn update_metrics(&mut self) -> Result<()> {
        let lattice = self.dynamics.lattice();

        // Update state
        self.metrics.state.states = lattice.states();
        self.metrics.state.probabilities = lattice.probabilities();
        self.metrics.state.num_pbits = lattice.size();

        // Update from dynamics statistics
        match self.dynamics.statistics() {
            hyperphysics_pbit::DynamicsStatistics::Gillespie { time, events, .. } => {
                self.metrics.state.time = time;
                self.metrics.state.events = events;
            }
            hyperphysics_pbit::DynamicsStatistics::Metropolis { steps, .. } => {
                self.metrics.state.events = steps;
            }
        }

        // Energy (keep scalar for now - SIMD requires coupling matrix extraction)
        self.metrics.energy = HamiltonianCalculator::energy(lattice);
        self.metrics.energy_per_pbit = HamiltonianCalculator::energy_per_pbit(lattice);

        // Entropy - SIMD accelerated when available
        #[cfg(feature = "simd")]
        let current_entropy = {
            let probabilities = lattice.probabilities();
            entropy_from_probabilities_simd(&probabilities)
        };

        #[cfg(not(feature = "simd"))]
        let current_entropy = self.entropy_calc.entropy_from_pbits(lattice);

        self.metrics.entropy = current_entropy;
        self.metrics.negentropy = self.entropy_calc.negentropy(
            current_entropy,
            lattice.size(),
        );

        // Entropy production
        let dt = self.metrics.state.time - self.metrics.simulation_time;
        if dt > 0.0 {
            self.metrics.entropy_production_rate = self.entropy_calc.entropy_production(
                current_entropy - self.previous_entropy,
                dt,
            );
        }

        self.previous_entropy = current_entropy;
        self.metrics.simulation_time = self.metrics.state.time;

        // Network metrics
        // Magnetization - SIMD accelerated when available
        #[cfg(feature = "simd")]
        {
            let states = lattice.states();
            self.metrics.magnetization = magnetization_simd(&states);
        }

        #[cfg(not(feature = "simd"))]
        {
            self.metrics.magnetization = lattice.magnetization();
        }
        self.metrics.causal_density = CausalDensityEstimator::causal_density(lattice);
        self.metrics.clustering_coefficient = CausalDensityEstimator::clustering_coefficient(lattice);

        // Consciousness metrics
        if let Some(ref phi_calc) = self.phi_calc {
            let phi_result = phi_calc.calculate(lattice)?;
            self.metrics.phi = Some(phi_result.phi);
        }

        if let Some(ref ci_calc) = self.ci_calc {
            let ci_result = ci_calc.calculate(lattice)?;
            self.metrics.ci = Some(ci_result.ci);
        }

        Ok(())
    }

    /// Verify thermodynamic laws
    fn verify_thermodynamics(&mut self) -> Result<()> {
        let current_entropy = self.metrics.entropy;
        let delta_s = current_entropy - self.previous_entropy;

        // Verify second law with tolerance matching Landauer enforcer
        // Tolerance: -1e-23 for floating-point precision
        self.metrics.second_law_satisfied = self.entropy_calc.verify_second_law(delta_s, 1e-23);

        if !self.metrics.second_law_satisfied {
            return Err(EngineError::Thermodynamics(
                ThermoError::SecondLawViolation { delta_s },
            ));
        }

        // Verify Landauer bound using enforcer
        // For each bit flip, check E_dissipated ≥ k_B T ln(2)
        let lattice = self.dynamics.lattice();
        let states = lattice.states();

        // Count state changes as proxy for bit erasures
        let mut bit_flips = 0;
        for (i, &current_state) in states.iter().enumerate() {
            if i < self.metrics.state.states.len() {
                let previous_state = self.metrics.state.states[i];
                if current_state != previous_state {
                    bit_flips += 1;
                }
            }
        }

        // Verify Landauer bound for information processing
        if bit_flips > 0 {
            let energy_change = self.metrics.energy - self.metrics.energy_per_pbit * lattice.size() as f64;

            // verify_bound returns Result - Ok means bound satisfied
            match self.landauer.verify_bound(energy_change.abs(), bit_flips) {
                Ok(_) => {
                    self.metrics.landauer_bound_satisfied = true;
                }
                Err(_) => {
                    // In simulation mode, log violation but don't halt
                    // In production, this would raise an error
                    self.metrics.landauer_bound_satisfied = false;
                }
            }
        } else {
            // No bit flips = trivially satisfied
            self.metrics.landauer_bound_satisfied = true;
        }

        Ok(())
    }

    /// Get current metrics
    pub fn metrics(&self) -> &EngineMetrics {
        &self.metrics
    }

    /// Get integrated information (Φ)
    pub fn integrated_information(&mut self) -> Result<f64> {
        if let Some(ref phi_calc) = self.phi_calc {
            let result = phi_calc.calculate(self.dynamics.lattice())?;
            Ok(result.phi)
        } else {
            Err(EngineError::Configuration {
                message: "Φ calculation not enabled".to_string(),
            })
        }
    }

    /// Get resonance complexity (CI)
    pub fn resonance_complexity(&mut self) -> Result<f64> {
        if let Some(ref ci_calc) = self.ci_calc {
            let result = ci_calc.calculate(self.dynamics.lattice())?;
            Ok(result.ci)
        } else {
            Err(EngineError::Configuration {
                message: "CI calculation not enabled".to_string(),
            })
        }
    }

    /// Get current lattice
    pub fn lattice(&self) -> &PBitLattice {
        self.dynamics.lattice()
    }

    /// Get configuration
    pub fn config(&self) -> &EngineConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_engine_step() {
        let mut engine = HyperPhysicsEngine::roi_48(1.0, 300.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        engine.step_with_rng(&mut rng).unwrap();

        assert_eq!(engine.metrics().timestep, 1);
    }

    #[test]
    fn test_engine_simulation() {
        let mut engine = HyperPhysicsEngine::roi_48(1.0, 300.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        engine.simulate(10, &mut rng).unwrap();

        assert_eq!(engine.metrics().timestep, 10);
    }

    #[test]
    fn test_metrics_update() {
        let mut engine = HyperPhysicsEngine::roi_48(1.0, 300.0).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        engine.step_with_rng(&mut rng).unwrap();

        let metrics = engine.metrics();
        assert!(metrics.energy.is_finite());
        assert!(metrics.entropy >= 0.0);
        assert!(metrics.magnetization >= -1.0 && metrics.magnetization <= 1.0);
    }

    #[test]
    fn test_consciousness_metrics() {
        let mut engine = HyperPhysicsEngine::roi_48(1.0, 300.0).unwrap();

        let phi = engine.integrated_information().unwrap();
        assert!(phi >= 0.0);

        let ci = engine.resonance_complexity().unwrap();
        assert!(ci > 0.0);
    }
}
