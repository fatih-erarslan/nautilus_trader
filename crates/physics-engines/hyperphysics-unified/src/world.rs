//! Physics world management

use crate::backend::{BackendError, PhysicsBackend, SimulationStats};
use crate::body::{BodyDesc, BodyHandle};
use crate::collider::{ColliderDesc, ColliderHandle};
use crate::constraint::{ConstraintDesc, ConstraintHandle};
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Vector3};
use serde::{Deserialize, Serialize};
use std::marker::PhantomData;

/// World configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    /// Gravity vector
    pub gravity: Vector3<f32>,
    /// Default timestep
    pub timestep: f32,
    /// Solver velocity iterations
    pub velocity_iterations: u32,
    /// Solver position iterations
    pub position_iterations: u32,
    /// Enable CCD globally
    pub ccd_enabled: bool,
    /// Sleep threshold (kinetic energy)
    pub sleep_threshold: f32,
    /// Maximum bodies
    pub max_bodies: u32,
}

impl Default for WorldConfig {
    fn default() -> Self {
        Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            timestep: 1.0 / 60.0,
            velocity_iterations: 8,
            position_iterations: 3,
            ccd_enabled: true,
            sleep_threshold: 0.001,
            max_bodies: 10000,
        }
    }
}

/// Physics world with pluggable backend
pub struct PhysicsWorld<B: PhysicsBackend> {
    backend: B,
    config: WorldConfig,
    accumulated_time: f32,
    step_count: u64,
}

impl<B: PhysicsBackend> PhysicsWorld<B> {
    /// Create world with default backend config
    pub fn new(config: WorldConfig) -> Result<Self, BackendError> {
        let mut backend = B::new(B::Config::default())?;
        backend.set_gravity(config.gravity);
        backend.set_solver_iterations(config.velocity_iterations, config.position_iterations);
        backend.set_ccd_enabled(config.ccd_enabled);

        Ok(Self {
            backend,
            config,
            accumulated_time: 0.0,
            step_count: 0,
        })
    }

    /// Create with custom backend config
    pub fn with_backend_config(
        config: WorldConfig,
        backend_config: B::Config,
    ) -> Result<Self, BackendError> {
        let mut backend = B::new(backend_config)?;
        backend.set_gravity(config.gravity);

        Ok(Self {
            backend,
            config,
            accumulated_time: 0.0,
            step_count: 0,
        })
    }

    /// Step simulation by fixed timestep
    pub fn step(&mut self) {
        self.backend.step(self.config.timestep);
        self.step_count += 1;
    }

    /// Step with custom dt
    pub fn step_dt(&mut self, dt: f32) {
        self.backend.step(dt);
        self.step_count += 1;
    }

    /// Fixed timestep with accumulator
    pub fn update(&mut self, dt: f32) -> u32 {
        self.accumulated_time += dt;
        let mut steps = 0;

        while self.accumulated_time >= self.config.timestep {
            self.step();
            self.accumulated_time -= self.config.timestep;
            steps += 1;
        }
        steps
    }

    /// Create rigid body
    pub fn create_body(&mut self, desc: &BodyDesc) -> Result<B::BodyHandle, BackendError> {
        self.backend.create_body(desc)
    }

    /// Remove rigid body
    pub fn remove_body(&mut self, handle: B::BodyHandle) -> Result<(), BackendError> {
        self.backend.remove_body(handle)
    }

    /// Create collider attached to body
    pub fn create_collider(
        &mut self,
        body: B::BodyHandle,
        desc: &ColliderDesc,
    ) -> Result<B::ColliderHandle, BackendError> {
        self.backend.create_collider(body, desc)
    }

    /// Get body transform
    pub fn body_transform(&self, handle: B::BodyHandle) -> Option<Transform> {
        self.backend.body_transform(handle)
    }

    /// Set body transform
    pub fn set_body_transform(&mut self, handle: B::BodyHandle, transform: Transform) {
        self.backend.set_body_transform(handle, transform);
    }

    /// Apply force to body
    pub fn apply_force(&mut self, handle: B::BodyHandle, force: Vector3<f32>) {
        self.backend.apply_force(handle, force);
    }

    /// Apply impulse to body
    pub fn apply_impulse(&mut self, handle: B::BodyHandle, impulse: Vector3<f32>) {
        self.backend.apply_impulse(handle, impulse);
    }

    /// Cast ray
    pub fn ray_cast(&self, ray: &RayCast) -> Option<RayHit<B::BodyHandle>> {
        self.backend.ray_cast(ray)
    }

    /// Query AABB
    pub fn query_aabb(&self, aabb: &AABB) -> Vec<B::BodyHandle> {
        self.backend.query_aabb(aabb)
    }

    /// Get contacts from last step
    pub fn contacts(&self) -> &[ContactManifold] {
        self.backend.contacts()
    }

    /// Get simulation statistics
    pub fn stats(&self) -> SimulationStats {
        self.backend.stats()
    }

    /// Get step count
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Reset world
    pub fn reset(&mut self) {
        self.backend.reset();
        self.accumulated_time = 0.0;
        self.step_count = 0;
    }

    /// Access backend directly
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// Access backend mutably
    pub fn backend_mut(&mut self) -> &mut B {
        &mut self.backend
    }
}
