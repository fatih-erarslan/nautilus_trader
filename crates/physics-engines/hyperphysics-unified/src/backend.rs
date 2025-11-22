//! Physics backend trait and implementations
//!
//! Zero-cost abstraction over multiple physics engines:
//! - Rapier3D (Rust native, high performance)
//! - Avian (Bevy ECS integration)
//! - JoltPhysics (AAA game engine, C++ FFI)
//! - MuJoCo (robotics, contact-rich simulation)
//! - Genesis (differentiable, GPU-accelerated)
//! - Taichi (GPU compute, differentiable)
//! - Project Chrono (multibody dynamics, vehicles)
//! - NVIDIA Warp (differentiable, autodiff)

use crate::{
    body::{BodyDesc, BodyHandle, RigidBody},
    collider::{Collider, ColliderDesc, ColliderHandle},
    constraint::{Constraint, ConstraintDesc, ConstraintHandle},
    query::{RayCast, RayHit, ShapeCast, ShapeHit},
    ContactManifold, PhysicsMaterial, Transform, AABB,
};
use nalgebra::{Point3, Vector3};
use std::any::Any;

/// Core physics backend trait - zero-cost abstraction
///
/// All physics engines implement this trait, allowing seamless switching
/// between backends at compile-time (zero overhead) or runtime (minimal overhead).
pub trait PhysicsBackend: Send + Sync + 'static {
    /// Backend-specific configuration type
    type Config: Default + Clone;

    /// Backend-specific body handle
    type BodyHandle: Copy + Eq + std::hash::Hash;

    /// Backend-specific collider handle
    type ColliderHandle: Copy + Eq + std::hash::Hash;

    /// Backend-specific constraint handle
    type ConstraintHandle: Copy + Eq + std::hash::Hash;

    /// Create a new backend instance with configuration
    fn new(config: Self::Config) -> Result<Self, BackendError>
    where
        Self: Sized;

    /// Get backend name and version
    fn info(&self) -> BackendInfo;

    /// Get backend capabilities
    fn capabilities(&self) -> BackendCapabilities;

    // ========== Simulation ==========

    /// Step simulation forward by dt seconds
    fn step(&mut self, dt: f32);

    /// Step with sub-stepping for stability
    fn step_with_substeps(&mut self, dt: f32, substeps: u32) {
        let sub_dt = dt / substeps as f32;
        for _ in 0..substeps {
            self.step(sub_dt);
        }
    }

    /// Set gravity vector
    fn set_gravity(&mut self, gravity: Vector3<f32>);

    /// Get current gravity vector
    fn gravity(&self) -> Vector3<f32>;

    // ========== Body Management ==========

    /// Create a rigid body from description
    fn create_body(&mut self, desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError>;

    /// Remove a rigid body
    fn remove_body(&mut self, handle: Self::BodyHandle) -> Result<(), BackendError>;

    /// Get body transform
    fn body_transform(&self, handle: Self::BodyHandle) -> Option<Transform>;

    /// Set body transform (teleport)
    fn set_body_transform(&mut self, handle: Self::BodyHandle, transform: Transform);

    /// Get body linear velocity
    fn body_linear_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>>;

    /// Set body linear velocity
    fn set_body_linear_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>);

    /// Get body angular velocity
    fn body_angular_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>>;

    /// Set body angular velocity
    fn set_body_angular_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>);

    /// Apply force at center of mass
    fn apply_force(&mut self, handle: Self::BodyHandle, force: Vector3<f32>);

    /// Apply force at world point
    fn apply_force_at_point(
        &mut self,
        handle: Self::BodyHandle,
        force: Vector3<f32>,
        point: Point3<f32>,
    );

    /// Apply impulse at center of mass
    fn apply_impulse(&mut self, handle: Self::BodyHandle, impulse: Vector3<f32>);

    /// Apply torque
    fn apply_torque(&mut self, handle: Self::BodyHandle, torque: Vector3<f32>);

    /// Get number of active bodies
    fn body_count(&self) -> usize;

    // ========== Collider Management ==========

    /// Create a collider attached to a body
    fn create_collider(
        &mut self,
        body: Self::BodyHandle,
        desc: &ColliderDesc,
    ) -> Result<Self::ColliderHandle, BackendError>;

    /// Remove a collider
    fn remove_collider(&mut self, handle: Self::ColliderHandle) -> Result<(), BackendError>;

    /// Set collider material
    fn set_collider_material(&mut self, handle: Self::ColliderHandle, material: PhysicsMaterial);

    /// Enable/disable collider
    fn set_collider_enabled(&mut self, handle: Self::ColliderHandle, enabled: bool);

    /// Get collider AABB
    fn collider_aabb(&self, handle: Self::ColliderHandle) -> Option<AABB>;

    // ========== Constraints ==========

    /// Create a constraint between two bodies
    fn create_constraint(
        &mut self,
        desc: &ConstraintDesc,
    ) -> Result<Self::ConstraintHandle, BackendError>;

    /// Remove a constraint
    fn remove_constraint(&mut self, handle: Self::ConstraintHandle) -> Result<(), BackendError>;

    // ========== Queries ==========

    /// Cast a ray and return first hit
    fn ray_cast(&self, ray: &RayCast) -> Option<RayHit<Self::BodyHandle>>;

    /// Cast a ray and return all hits
    fn ray_cast_all(&self, ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>>;

    /// Cast a shape and return first hit
    fn shape_cast(&self, cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>>;

    /// Query all bodies intersecting an AABB
    fn query_aabb(&self, aabb: &AABB) -> Vec<Self::BodyHandle>;

    /// Get all contact manifolds from last step
    fn contacts(&self) -> &[ContactManifold];

    // ========== State Management ==========

    /// Serialize world state for deterministic replay
    fn serialize_state(&self) -> Result<Vec<u8>, BackendError>;

    /// Deserialize and restore world state
    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), BackendError>;

    /// Reset world to initial state
    fn reset(&mut self);

    // ========== Performance ==========

    /// Get performance statistics
    fn stats(&self) -> SimulationStats;

    /// Warm start constraint solver (for determinism)
    fn warm_start(&mut self, _enabled: bool) {}

    /// Set solver iterations
    fn set_solver_iterations(&mut self, _velocity: u32, _position: u32) {}

    /// Enable/disable continuous collision detection
    fn set_ccd_enabled(&mut self, _enabled: bool) {}

    // ========== Extension ==========

    /// Downcast to concrete backend type for advanced features
    fn as_any(&self) -> &dyn Any;

    /// Downcast to concrete backend type (mutable)
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// Backend information
#[derive(Debug, Clone)]
pub struct BackendInfo {
    /// Backend name
    pub name: &'static str,
    /// Backend version
    pub version: &'static str,
    /// Backend description
    pub description: &'static str,
    /// Is GPU accelerated
    pub gpu_accelerated: bool,
    /// Supports differentiable simulation
    pub differentiable: bool,
}

/// Backend capabilities flags
#[derive(Debug, Clone, Copy, Default)]
pub struct BackendCapabilities {
    /// Supports 2D physics
    pub physics_2d: bool,
    /// Supports 3D physics
    pub physics_3d: bool,
    /// Supports soft bodies
    pub soft_bodies: bool,
    /// Supports cloth simulation
    pub cloth: bool,
    /// Supports fluid simulation
    pub fluids: bool,
    /// Supports articulated bodies (robots)
    pub articulated: bool,
    /// Supports continuous collision detection
    pub ccd: bool,
    /// Supports deterministic simulation
    pub deterministic: bool,
    /// Supports parallel simulation
    pub parallel: bool,
    /// Supports GPU acceleration
    pub gpu: bool,
    /// Supports autodiff/gradients
    pub differentiable: bool,
    /// Maximum body count (0 = unlimited)
    pub max_bodies: u32,
}

/// Backend error types
#[derive(Debug, thiserror::Error)]
pub enum BackendError {
    /// Backend initialization failed
    #[error("Backend initialization failed: {0}")]
    InitializationFailed(String),

    /// Invalid handle
    #[error("Invalid handle: {0}")]
    InvalidHandle(String),

    /// Unsupported operation
    #[error("Unsupported operation: {0}")]
    Unsupported(String),

    /// Serialization error
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Resource limit exceeded
    #[error("Resource limit exceeded: {0}")]
    ResourceLimit(String),

    /// GPU error
    #[error("GPU error: {0}")]
    GpuError(String),

    /// FFI error
    #[error("FFI error: {0}")]
    FfiError(String),
}

/// Simulation performance statistics
#[derive(Debug, Clone, Default)]
pub struct SimulationStats {
    /// Time spent in broadphase (μs)
    pub broadphase_us: u64,
    /// Time spent in narrowphase (μs)
    pub narrowphase_us: u64,
    /// Time spent in solver (μs)
    pub solver_us: u64,
    /// Time spent in integration (μs)
    pub integration_us: u64,
    /// Total step time (μs)
    pub total_us: u64,
    /// Number of active bodies
    pub active_bodies: u32,
    /// Number of sleeping bodies
    pub sleeping_bodies: u32,
    /// Number of contact pairs
    pub contact_pairs: u32,
    /// Number of constraints
    pub constraints: u32,
    /// Solver iterations used
    pub solver_iterations: u32,
    /// Memory usage (bytes)
    pub memory_bytes: u64,
}

// ============================================================================
// Backend implementations
// ============================================================================

#[cfg(feature = "rapier3d")]
pub mod rapier;

#[cfg(feature = "rapier3d")]
pub use rapier::RapierBackend;

// Placeholder modules for other backends
pub mod avian;
pub mod chrono;
pub mod genesis;
pub mod jolt;
pub mod mujoco;
pub mod taichi;
pub mod warp;
