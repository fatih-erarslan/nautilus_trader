//! Physics simulation execution
//!
//! Runs Rapier physics simulation to model market dynamics and evolution

use crate::{RapierHyperPhysicsAdapter, Result};
use nalgebra::Vector3;
use rapier3d::prelude::*;
use std::time::Instant;

/// Physics simulation result
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Total simulation time in microseconds
    pub elapsed_micros: u64,

    /// Number of steps executed
    pub steps: usize,

    /// Final energy of the system
    pub total_energy: f32,

    /// Convergence achieved
    pub converged: bool,

    /// Center of mass movement (market momentum)
    pub momentum_vector: Vector3<f32>,

    /// Average velocity (market activity)
    pub avg_velocity: f32,
}

/// Physics simulator configuration
#[derive(Debug, Clone)]
pub struct SimulatorConfig {
    /// Number of simulation steps
    pub steps: usize,

    /// Time step per iteration (seconds)
    pub dt: f32,

    /// Convergence threshold
    pub convergence_threshold: f32,

    /// Apply external forces (market shocks)
    pub external_forces: bool,
}

impl Default for SimulatorConfig {
    fn default() -> Self {
        Self {
            steps: 100,
            dt: 0.016, // ~60 FPS
            convergence_threshold: 0.001,
            external_forces: false,
        }
    }
}

/// Physics simulator
pub struct PhysicsSimulator {
    config: SimulatorConfig,
}

impl PhysicsSimulator {
    /// Create a new simulator with default config
    pub fn new() -> Self {
        Self {
            config: SimulatorConfig::default(),
        }
    }

    /// Create simulator with custom config
    pub fn with_config(config: SimulatorConfig) -> Self {
        Self { config }
    }

    /// Run physics simulation
    pub fn simulate(&self, adapter: &mut RapierHyperPhysicsAdapter) -> Result<SimulationResult> {
        let start = Instant::now();
        let mut previous_energy = self.calculate_total_energy(adapter);
        let mut converged = false;

        // Run simulation steps
        for step in 0..self.config.steps {
            adapter.step();

            // Check for convergence
            if step > 10 {
                let current_energy = self.calculate_total_energy(adapter);
                let energy_change = (current_energy - previous_energy).abs();

                if energy_change < self.config.convergence_threshold {
                    converged = true;
                    break;
                }

                previous_energy = current_energy;
            }
        }

        let elapsed = start.elapsed();
        let total_energy = self.calculate_total_energy(adapter);
        let momentum = self.calculate_momentum(adapter);
        let avg_velocity = self.calculate_avg_velocity(adapter);

        Ok(SimulationResult {
            elapsed_micros: elapsed.as_micros() as u64,
            steps: self.config.steps,
            total_energy,
            converged,
            momentum_vector: momentum,
            avg_velocity,
        })
    }

    /// Calculate total kinetic energy of the system
    fn calculate_total_energy(&self, adapter: &RapierHyperPhysicsAdapter) -> f32 {
        let mut total_energy = 0.0;

        for (_handle, rb) in adapter.rigid_bodies().iter() {
            if rb.is_dynamic() {
                let velocity = rb.linvel();
                let mass = rb.mass();
                total_energy += 0.5 * mass * velocity.norm_squared();
            }
        }

        total_energy
    }

    /// Calculate system momentum (market direction)
    fn calculate_momentum(&self, adapter: &RapierHyperPhysicsAdapter) -> Vector3<f32> {
        let mut total_momentum = Vector3::zeros();

        for (_handle, rb) in adapter.rigid_bodies().iter() {
            if rb.is_dynamic() {
                let velocity = rb.linvel();
                let mass = rb.mass();
                total_momentum += velocity * mass;
            }
        }

        total_momentum
    }

    /// Calculate average velocity magnitude
    fn calculate_avg_velocity(&self, adapter: &RapierHyperPhysicsAdapter) -> f32 {
        let mut total_velocity = 0.0;
        let mut count = 0;

        for (_handle, rb) in adapter.rigid_bodies().iter() {
            if rb.is_dynamic() {
                total_velocity += rb.linvel().norm();
                count += 1;
            }
        }

        if count > 0 {
            total_velocity / (count as f32)
        } else {
            0.0
        }
    }

    /// Apply external force to simulate market shock
    pub fn apply_market_shock(
        &self,
        adapter: &mut RapierHyperPhysicsAdapter,
        force: Vector3<f32>,
        target_fraction: f32,
    ) {
        let total_bodies = adapter.rigid_bodies().len();
        let target_count = ((total_bodies as f32) * target_fraction) as usize;

        let mut applied = 0;
        for (_handle, rb) in adapter.rigid_bodies_mut().iter_mut() {
            if rb.is_dynamic() && applied < target_count {
                rb.add_force(force, true);
                applied += 1;
            }
        }
    }
}

impl Default for PhysicsSimulator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulator_creation() {
        let simulator = PhysicsSimulator::new();
        assert_eq!(simulator.config.steps, 100);
    }

    #[test]
    fn test_simulation_with_falling_body() {
        let mut adapter = RapierHyperPhysicsAdapter::new();

        // Add a falling body
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector3::new(0.0, 10.0, 0.0))
            .build();
        let rb_handle = adapter.rigid_bodies_mut().insert(rb);

        let collider = ColliderBuilder::ball(0.5).build();
        adapter.add_collider_with_parent(collider, rb_handle);

        let simulator = PhysicsSimulator::with_config(SimulatorConfig {
            steps: 50,
            dt: 0.016,
            convergence_threshold: 0.001,
            external_forces: false,
        });

        let result = simulator.simulate(&mut adapter).unwrap();

        assert!(result.elapsed_micros > 0);
        assert!(result.total_energy > 0.0);
        assert!(result.avg_velocity > 0.0);
    }

    #[test]
    fn test_market_shock() {
        let mut adapter = RapierHyperPhysicsAdapter::new();

        // Add multiple bodies
        for i in 0..10 {
            let rb = RigidBodyBuilder::dynamic()
                .translation(Vector3::new(i as f32, 0.0, 0.0))
                .build();
            let rb_handle = adapter.rigid_bodies_mut().insert(rb);

            let collider = ColliderBuilder::ball(0.5).build();
            adapter.add_collider_with_parent(collider, rb_handle);
        }

        let simulator = PhysicsSimulator::new();
        let shock_force = Vector3::new(100.0, 0.0, 0.0);

        simulator.apply_market_shock(&mut adapter, shock_force, 0.5);

        // Verify forces were applied
        // (We can't directly check forces, but we can run simulation and check movement)
        let result = simulator.simulate(&mut adapter).unwrap();
        assert!(result.avg_velocity > 0.0);
    }
}
