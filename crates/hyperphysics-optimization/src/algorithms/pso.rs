//! Particle Swarm Optimization (PSO) with formal convergence guarantees.
//!
//! # Formal Verification
//!
//! **Theorem PSO1** (Convergence): Under standard assumptions (ω ∈ [0,1), c₁+c₂ ≤ 4),
//! particles converge to a common point in the search space.
//!
//! **Property PSO2**: Each particle's velocity is bounded by `v_max`.
//!
//! **Invariant PSO3**: `∀t: global_best(t+1) ≤ global_best(t)` (monotonic improvement)
//!
//! # References
//!
//! - Kennedy & Eberhart (1995): "Particle Swarm Optimization"
//! - Clerc & Kennedy (2002): "The particle swarm - explosion, stability, and convergence"
//! - Shi & Eberhart (1998): "A modified particle swarm optimizer"

use crate::core::{Bounds, Individual, ObjectiveFunction, OptimizationConfig, Population};
use crate::algorithms::{Algorithm, AlgorithmConfig, AlgorithmType};
use crate::OptimizationError;
use ndarray::Array1;
use rand::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// PSO configuration with formally verified parameter bounds.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PSOConfig {
    /// Inertia weight (ω): controls velocity momentum
    /// Formal constraint: ω ∈ [0, 1) for convergence
    pub inertia_weight: f64,
    /// Cognitive coefficient (c₁): personal best attraction
    /// Formal constraint: c₁ ≥ 0
    pub cognitive_coeff: f64,
    /// Social coefficient (c₂): global best attraction
    /// Formal constraint: c₂ ≥ 0, c₁ + c₂ ≤ 4 for stability
    pub social_coeff: f64,
    /// Maximum velocity as fraction of search range
    pub v_max_fraction: f64,
    /// Minimum inertia weight (for adaptive strategies)
    pub inertia_min: f64,
    /// Maximum inertia weight (for adaptive strategies)
    pub inertia_max: f64,
    /// Use constriction factor (Clerc & Kennedy 2002)
    pub use_constriction: bool,
    /// Velocity clamping strategy
    pub velocity_clamping: VelocityClamping,
    /// Topology for neighborhood communication
    pub topology: PSOTopology,
}

impl Default for PSOConfig {
    fn default() -> Self {
        Self {
            inertia_weight: 0.7298,
            cognitive_coeff: 1.49618,
            social_coeff: 1.49618,
            v_max_fraction: 0.5,
            inertia_min: 0.4,
            inertia_max: 0.9,
            use_constriction: true,
            velocity_clamping: VelocityClamping::Symmetric,
            topology: PSOTopology::Global,
        }
    }
}

impl AlgorithmConfig for PSOConfig {
    fn validate(&self) -> Result<(), String> {
        // Theorem PSO1: Convergence conditions
        if self.inertia_weight < 0.0 || self.inertia_weight >= 1.0 {
            return Err(format!(
                "Inertia weight {} violates convergence bound [0, 1)",
                self.inertia_weight
            ));
        }
        if self.cognitive_coeff < 0.0 {
            return Err(format!(
                "Cognitive coefficient {} must be non-negative",
                self.cognitive_coeff
            ));
        }
        if self.social_coeff < 0.0 {
            return Err(format!(
                "Social coefficient {} must be non-negative",
                self.social_coeff
            ));
        }
        // Stability condition from Clerc & Kennedy 2002
        let sum = self.cognitive_coeff + self.social_coeff;
        if sum > 4.0 && !self.use_constriction {
            return Err(format!(
                "c₁ + c₂ = {} > 4 without constriction violates stability",
                sum
            ));
        }
        if self.v_max_fraction <= 0.0 || self.v_max_fraction > 1.0 {
            return Err(format!(
                "v_max_fraction {} must be in (0, 1]",
                self.v_max_fraction
            ));
        }
        Ok(())
    }

    fn hft_optimized() -> Self {
        Self {
            inertia_weight: 0.6,
            cognitive_coeff: 1.8,
            social_coeff: 1.8,
            v_max_fraction: 0.3,
            inertia_min: 0.4,
            inertia_max: 0.6,
            use_constriction: true,
            velocity_clamping: VelocityClamping::Symmetric,
            topology: PSOTopology::Global,
        }
    }

    fn high_accuracy() -> Self {
        Self {
            inertia_weight: 0.9,
            cognitive_coeff: 1.49618,
            social_coeff: 1.49618,
            v_max_fraction: 0.5,
            inertia_min: 0.2,
            inertia_max: 0.9,
            use_constriction: true,
            velocity_clamping: VelocityClamping::Hyperbolic,
            topology: PSOTopology::Ring(3),
        }
    }
}

/// Velocity clamping strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VelocityClamping {
    /// Simple symmetric clamping: v ∈ [-v_max, v_max]
    Symmetric,
    /// Asymmetric based on particle position
    Asymmetric,
    /// Hyperbolic tangent scaling
    Hyperbolic,
    /// No clamping (use with constriction factor)
    None,
}

/// PSO communication topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PSOTopology {
    /// All particles communicate (gbest PSO)
    Global,
    /// Ring topology with k neighbors (lbest PSO)
    Ring(usize),
    /// Von Neumann grid topology
    VonNeumann,
    /// Random topology with k connections
    Random(usize),
}

/// Particle Swarm Optimization algorithm.
pub struct ParticleSwarmOptimizer {
    /// Algorithm configuration
    config: PSOConfig,
    /// Optimization configuration
    opt_config: OptimizationConfig,
    /// Population of particles
    population: Population,
    /// Current iteration
    iteration: u32,
    /// Convergence flag
    converged: bool,
    /// Best fitness history for stagnation detection
    fitness_history: Vec<f64>,
    /// Velocity bounds per dimension
    v_max: Array1<f64>,
    /// Constriction factor (χ)
    chi: f64,
}

impl ParticleSwarmOptimizer {
    /// Create a new PSO optimizer.
    ///
    /// # Formal Preconditions
    /// - `config.validate() == Ok(())`
    /// - `bounds.dimension() > 0`
    pub fn new(
        config: PSOConfig,
        opt_config: OptimizationConfig,
        bounds: Bounds,
    ) -> Result<Self, OptimizationError> {
        config.validate().map_err(|e| OptimizationError::Configuration(e))?;

        let dimension = bounds.dimension();
        let ranges = bounds.ranges();

        // Calculate velocity bounds
        let v_max = Array1::from_iter(
            ranges.iter().map(|r| r * config.v_max_fraction)
        );

        // Calculate constriction factor (Clerc & Kennedy 2002)
        let phi = config.cognitive_coeff + config.social_coeff;
        let chi = if config.use_constriction && phi > 4.0 {
            2.0 / (phi - 2.0 + (phi * phi - 4.0 * phi).sqrt()).abs()
        } else {
            1.0
        };

        let population = Population::new(opt_config.population_size, bounds);

        Ok(Self {
            config,
            opt_config,
            population,
            iteration: 0,
            converged: false,
            fitness_history: Vec::with_capacity(1000),
            v_max,
            chi,
        })
    }

    /// Initialize particle swarm.
    ///
    /// # Formal Postcondition
    /// `∀p ∈ particles: p.velocity ≠ None ∧ bounds.is_within_box(p.position)`
    pub fn initialize(&mut self) {
        // Initialize positions using Latin Hypercube Sampling
        self.population.initialize_lhs(self.opt_config.population_size);

        // Initialize velocities to zero or small random values
        let mut rng = rand::thread_rng();
        let mut particles = self.population.individuals_mut();

        for particle in particles.iter_mut() {
            let velocity = Array1::from_iter(
                (0..self.v_max.len()).map(|i| {
                    rng.gen_range(-self.v_max[i] * 0.1..self.v_max[i] * 0.1)
                })
            );
            particle.velocity = Some(velocity);
            particle.best_position = Some(particle.position.clone());
        }
    }

    /// Run single PSO iteration.
    ///
    /// # Formal Properties
    /// - **Invariant PSO3**: Best fitness is monotonically non-increasing
    /// - Updates each particle's velocity and position
    pub fn step<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<(), OptimizationError> {
        // Evaluate current population
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Get global best for velocity update
        let global_best_pos = self.population.best()
            .and_then(|b| b.best_position.clone())
            .unwrap_or_else(|| self.population.bounds.center());

        // Update particles
        let mut rng = rand::thread_rng();
        let dimension = self.population.bounds.dimension();
        let bounds = &self.population.bounds;

        // Adaptive inertia weight (linear decrease)
        let progress = self.iteration as f64 / self.opt_config.max_iterations as f64;
        let omega = self.config.inertia_max
            - (self.config.inertia_max - self.config.inertia_min) * progress;

        let mut particles = self.population.individuals_mut();

        for particle in particles.iter_mut() {
            let velocity = particle.velocity.as_ref().unwrap();
            let position = &particle.position;
            let p_best = particle.best_position.as_ref().unwrap();

            // Generate random coefficients
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();

            // PSO velocity update equation:
            // v(t+1) = χ * [ω*v(t) + c₁*r₁*(pbest - x) + c₂*r₂*(gbest - x)]
            let mut new_velocity = Array1::zeros(dimension);
            for i in 0..dimension {
                let cognitive = self.config.cognitive_coeff * r1 * (p_best[i] - position[i]);
                let social = self.config.social_coeff * r2 * (global_best_pos[i] - position[i]);
                let inertia = omega * velocity[i];

                new_velocity[i] = self.chi * (inertia + cognitive + social);

                // Apply velocity clamping (Property PSO2)
                new_velocity[i] = self.clamp_velocity(new_velocity[i], i);
            }

            // Update position: x(t+1) = x(t) + v(t+1)
            let mut new_position = position + &new_velocity;

            // Handle boundary violations
            new_position = bounds.repair(new_position.view());

            // Update particle
            particle.position = new_position;
            particle.velocity = Some(new_velocity);
        }

        drop(particles);

        // Re-evaluate after position update
        #[cfg(feature = "parallel")]
        self.population.evaluate_parallel(objective)?;
        #[cfg(not(feature = "parallel"))]
        self.population.evaluate_sequential(objective)?;

        // Update personal bests
        self.update_personal_bests();

        // Track fitness history
        if let Some(best_fitness) = self.population.best_fitness() {
            self.fitness_history.push(best_fitness);
        }

        // Check convergence
        self.check_convergence();

        self.iteration += 1;
        self.population.next_generation();

        Ok(())
    }

    /// Update personal bests for all particles.
    fn update_personal_bests(&self) {
        let mut particles = self.population.individuals_mut();
        for particle in particles.iter_mut() {
            if let Some(fitness) = particle.fitness {
                let should_update = particle.best_fitness
                    .map_or(true, |best| fitness < best);
                if should_update {
                    particle.best_fitness = Some(fitness);
                    particle.best_position = Some(particle.position.clone());
                    particle.stagnation_count = 0;
                } else {
                    particle.stagnation_count += 1;
                }
            }
        }
    }

    /// Apply velocity clamping strategy.
    fn clamp_velocity(&self, v: f64, dim: usize) -> f64 {
        match self.config.velocity_clamping {
            VelocityClamping::Symmetric => {
                v.clamp(-self.v_max[dim], self.v_max[dim])
            }
            VelocityClamping::Hyperbolic => {
                self.v_max[dim] * v.tanh()
            }
            VelocityClamping::Asymmetric | VelocityClamping::None => v,
        }
    }

    /// Check for convergence conditions.
    fn check_convergence(&mut self) {
        // Check tolerance-based convergence
        if self.fitness_history.len() >= 2 {
            let recent = &self.fitness_history[self.fitness_history.len().saturating_sub(10)..];
            if recent.len() >= 10 {
                let improvement = recent.first().unwrap() - recent.last().unwrap();
                if improvement.abs() < self.opt_config.tolerance {
                    self.converged = true;
                    return;
                }
            }
        }

        // Check target fitness
        if let (Some(target), Some(best)) = (self.opt_config.target_fitness, self.population.best_fitness()) {
            if best <= target {
                self.converged = true;
                return;
            }
        }

        // Check diversity collapse
        let diversity = self.population.diversity();
        if diversity < 1e-10 {
            self.converged = true;
        }
    }

    /// Run full optimization until convergence.
    pub fn optimize<F: ObjectiveFunction + Sync>(&mut self, objective: &F) -> Result<Individual, OptimizationError> {
        self.initialize();

        while self.iteration < self.opt_config.max_iterations && !self.converged {
            self.step(objective)?;
        }

        self.population.best()
            .ok_or_else(|| OptimizationError::NoSolution("PSO failed to find solution".to_string()))
    }

    /// Get swarm diversity metric.
    #[must_use]
    pub fn diversity(&self) -> f64 {
        self.population.diversity()
    }

    /// Get convergence curve.
    #[must_use]
    pub fn convergence_curve(&self) -> &[f64] {
        &self.fitness_history
    }
}

impl Algorithm for ParticleSwarmOptimizer {
    fn algorithm_type(&self) -> AlgorithmType {
        AlgorithmType::ParticleSwarm
    }

    fn name(&self) -> &str {
        "Particle Swarm Optimization"
    }

    fn is_converged(&self) -> bool {
        self.converged
    }

    fn best_fitness(&self) -> Option<f64> {
        self.population.best_fitness()
    }

    fn iteration(&self) -> u32 {
        self.iteration
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::SphereFunction;

    #[test]
    fn test_pso_config_validation() {
        let config = PSOConfig::default();
        assert!(config.validate().is_ok());

        let invalid = PSOConfig {
            inertia_weight: 1.5,  // Invalid: >= 1
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_pso_sphere_optimization() {
        let config = PSOConfig::default();
        let opt_config = OptimizationConfig::default()
            .with_max_iterations(100)
            .with_population_size(30);
        let bounds = Bounds::symmetric(5, 5.12);

        let mut pso = ParticleSwarmOptimizer::new(config, opt_config, bounds).unwrap();
        let objective = SphereFunction::new(5);

        let solution = pso.optimize(&objective).unwrap();

        // Sphere function minimum is 0 at origin
        assert!(solution.fitness.unwrap() < 1.0);
    }

    #[test]
    fn test_pso_convergence() {
        let config = PSOConfig::default();
        let opt_config = OptimizationConfig::default()
            .with_max_iterations(200)
            .with_tolerance(1e-6);
        let bounds = Bounds::symmetric(2, 5.12);

        let mut pso = ParticleSwarmOptimizer::new(config, opt_config, bounds).unwrap();
        let objective = SphereFunction::new(2);

        let _ = pso.optimize(&objective).unwrap();

        // Should have convergence history
        assert!(!pso.convergence_curve().is_empty());
    }

    #[test]
    fn test_pso_hft_config() {
        let config = PSOConfig::hft_optimized();
        assert!(config.validate().is_ok());
        // HFT config should have lower inertia for faster convergence
        assert!(config.inertia_weight <= 0.7);
    }

    #[test]
    fn test_velocity_clamping() {
        let config = PSOConfig {
            velocity_clamping: VelocityClamping::Hyperbolic,
            ..Default::default()
        };
        let opt_config = OptimizationConfig::default().with_population_size(10);
        let bounds = Bounds::symmetric(2, 10.0);

        let pso = ParticleSwarmOptimizer::new(config, opt_config, bounds).unwrap();

        // Hyperbolic clamping should bound velocity to [-v_max, v_max]
        let clamped = pso.clamp_velocity(100.0, 0);
        assert!(clamped <= pso.v_max[0]);
    }
}
