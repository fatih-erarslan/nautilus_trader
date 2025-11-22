//! Rapier Physics Engine Integration for HyperPhysics HFT
//!
//! Maps market entities (orders, trades, participants) to rigid bodies in Rapier3D,
//! runs physics simulation to model market dynamics, and extracts trading signals.
//!
//! ## Performance Target
//! - Latency: <500μs per simulation cycle
//! - Throughput: 2000+ simulations/second
//! - Determinism: Optional (via feature flag)

use nalgebra::Vector3;
use rapier3d::prelude::*;
use thiserror::Error;

pub mod market_mapper;
pub mod physics_simulator;
pub mod signal_extractor;

pub use market_mapper::*;
pub use physics_simulator::*;
pub use signal_extractor::*;

/// Rapier-HyperPhysics integration result
pub type Result<T> = std::result::Result<T, RapierHyperPhysicsError>;

/// Errors for Rapier integration
#[derive(Error, Debug)]
pub enum RapierHyperPhysicsError {
    #[error("Physics simulation error: {0}")]
    SimulationError(String),

    #[error("Market mapping error: {0}")]
    MappingError(String),

    #[error("Signal extraction error: {0}")]
    SignalError(String),

    #[error("Configuration error: {0}")]
    ConfigError(String),
}

/// Main Rapier-HyperPhysics adapter
pub struct RapierHyperPhysicsAdapter {
    /// Rapier physics pipeline
    physics_pipeline: PhysicsPipeline,

    /// Gravity (market forces)
    gravity: Vector3<f32>,

    /// Integration parameters
    integration_params: IntegrationParameters,

    /// Island manager
    islands: IslandManager,

    /// Broad phase
    broad_phase: DefaultBroadPhase,

    /// Narrow phase
    narrow_phase: NarrowPhase,

    /// Rigid body set
    rigid_body_set: RigidBodySet,

    /// Collider set
    collider_set: ColliderSet,

    /// Impulse joint set
    impulse_joints: ImpulseJointSet,

    /// Multibody joint set
    multibody_joints: MultibodyJointSet,

    /// CCD solver
    ccd_solver: CCDSolver,

    /// Query pipeline
    _query_pipeline: QueryPipeline,
}

impl RapierHyperPhysicsAdapter {
    /// Create a new Rapier adapter with default configuration
    pub fn new() -> Self {
        let gravity = Vector3::new(0.0, -9.81, 0.0);
        let integration_params = IntegrationParameters::default();

        Self {
            physics_pipeline: PhysicsPipeline::new(),
            gravity,
            integration_params,
            islands: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            rigid_body_set: RigidBodySet::new(),
            collider_set: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            _query_pipeline: QueryPipeline::new(),
        }
    }

    /// Create adapter with custom gravity (market forces)
    pub fn with_gravity(mut self, gravity: Vector3<f32>) -> Self {
        self.gravity = gravity;
        self
    }

    /// Set integration time step
    pub fn with_timestep(mut self, dt: f32) -> Self {
        self.integration_params.dt = dt;
        self
    }

    /// Run a single physics simulation step
    pub fn step(&mut self) {
        self.physics_pipeline.step(
            &self.gravity,
            &self.integration_params,
            &mut self.islands,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.rigid_body_set,
            &mut self.collider_set,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            None, // No query pipeline modification
            &(),  // No hooks
            &(),  // No events
        );
    }

    /// Get reference to rigid body set
    pub fn rigid_bodies(&self) -> &RigidBodySet {
        &self.rigid_body_set
    }

    /// Get mutable reference to rigid body set
    pub fn rigid_bodies_mut(&mut self) -> &mut RigidBodySet {
        &mut self.rigid_body_set
    }

    /// Get reference to collider set
    pub fn colliders(&self) -> &ColliderSet {
        &self.collider_set
    }

    /// Get mutable reference to collider set
    pub fn colliders_mut(&mut self) -> &mut ColliderSet {
        &mut self.collider_set
    }

    /// Get mutable references to both rigid body and collider sets
    pub fn split_sets_mut(&mut self) -> (&mut RigidBodySet, &mut ColliderSet) {
        (&mut self.rigid_body_set, &mut self.collider_set)
    }

    /// Add a collider with a parent rigid body
    pub fn add_collider_with_parent(
        &mut self,
        collider: Collider,
        parent_handle: RigidBodyHandle,
    ) -> ColliderHandle {
        self.collider_set
            .insert_with_parent(collider, parent_handle, &mut self.rigid_body_set)
    }

    /// Reset the simulation
    pub fn reset(&mut self) {
        self.rigid_body_set = RigidBodySet::new();
        self.collider_set = ColliderSet::new();
        self.impulse_joints = ImpulseJointSet::new();
        self.multibody_joints = MultibodyJointSet::new();
        self.islands = IslandManager::new();
        self.broad_phase = DefaultBroadPhase::new();
        self.narrow_phase = NarrowPhase::new();
        self.ccd_solver = CCDSolver::new();
    }
}

impl Default for RapierHyperPhysicsAdapter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_creation() {
        let adapter = RapierHyperPhysicsAdapter::new();
        assert_eq!(adapter.rigid_bodies().len(), 0);
        assert_eq!(adapter.colliders().len(), 0);
    }

    #[test]
    fn test_custom_gravity() {
        let custom_gravity = Vector3::new(1.0, 2.0, 3.0);
        let adapter = RapierHyperPhysicsAdapter::new().with_gravity(custom_gravity);
        assert_eq!(adapter.gravity, custom_gravity);
    }

    #[test]
    fn test_reset() {
        let mut adapter = RapierHyperPhysicsAdapter::new();

        // Add a rigid body
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector3::new(0.0, 10.0, 0.0))
            .build();
        adapter.rigid_bodies_mut().insert(rb);

        assert_eq!(adapter.rigid_bodies().len(), 1);

        // Reset
        adapter.reset();
        assert_eq!(adapter.rigid_bodies().len(), 0);
    }

    #[test]
    fn test_simulation_step() {
        let mut adapter = RapierHyperPhysicsAdapter::new();

        // Add a falling rigid body
        let rb = RigidBodyBuilder::dynamic()
            .translation(Vector3::new(0.0, 10.0, 0.0))
            .build();
        let rb_handle = adapter.rigid_bodies_mut().insert(rb);

        // Add a collider
        let collider = ColliderBuilder::ball(0.5).build();
        adapter.add_collider_with_parent(collider, rb_handle);
        // Get initial position
        let initial_y = adapter.rigid_bodies()[rb_handle].translation().y;

        // Run simulation
        for _ in 0..10 {
            adapter.step();
        }

        // Check that body has fallen
        let final_y = adapter.rigid_bodies()[rb_handle].translation().y;
        assert!(
            final_y < initial_y,
            "Body should have fallen due to gravity"
        );
    }
}
