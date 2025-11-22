//! Project Chrono multibody dynamics backend
//!
//! Fork: https://github.com/fatih-erarslan/chrono

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::BodyDesc;
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Vector3};
use std::any::Any;

#[derive(Debug, Clone, Default)]
pub struct ChronoConfig {
    pub solver_type: String,
    pub step_size: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChronoBodyHandle(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChronoColliderHandle(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChronoConstraintHandle(pub u32);

pub struct ChronoBackend {
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
}

impl PhysicsBackend for ChronoBackend {
    type Config = ChronoConfig;
    type BodyHandle = ChronoBodyHandle;
    type ColliderHandle = ChronoColliderHandle;
    type ConstraintHandle = ChronoConstraintHandle;

    fn new(_config: Self::Config) -> Result<Self, BackendError> {
        Ok(Self { gravity: Vector3::new(0.0, -9.81, 0.0), contacts: Vec::new(), stats: SimulationStats::default() })
    }

    fn info(&self) -> BackendInfo {
        BackendInfo { name: "Project Chrono", version: "9.0", description: "Multibody dynamics, vehicles, granular", gpu_accelerated: true, differentiable: false }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities { physics_3d: true, physics_2d: false, soft_bodies: true, cloth: false, fluids: true, articulated: true, ccd: true, deterministic: true, parallel: true, gpu: true, differentiable: false, max_bodies: 0 }
    }

    fn step(&mut self, _dt: f32) {}
    fn set_gravity(&mut self, gravity: Vector3<f32>) { self.gravity = gravity; }
    fn gravity(&self) -> Vector3<f32> { self.gravity }
    fn create_body(&mut self, _desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> { Err(BackendError::Unsupported("Chrono C++ API".into())) }
    fn remove_body(&mut self, _handle: Self::BodyHandle) -> Result<(), BackendError> { Ok(()) }
    fn body_transform(&self, _handle: Self::BodyHandle) -> Option<Transform> { None }
    fn set_body_transform(&mut self, _handle: Self::BodyHandle, _transform: Transform) {}
    fn body_linear_velocity(&self, _handle: Self::BodyHandle) -> Option<Vector3<f32>> { None }
    fn set_body_linear_velocity(&mut self, _handle: Self::BodyHandle, _velocity: Vector3<f32>) {}
    fn body_angular_velocity(&self, _handle: Self::BodyHandle) -> Option<Vector3<f32>> { None }
    fn set_body_angular_velocity(&mut self, _handle: Self::BodyHandle, _velocity: Vector3<f32>) {}
    fn apply_force(&mut self, _handle: Self::BodyHandle, _force: Vector3<f32>) {}
    fn apply_force_at_point(&mut self, _handle: Self::BodyHandle, _force: Vector3<f32>, _point: Point3<f32>) {}
    fn apply_impulse(&mut self, _handle: Self::BodyHandle, _impulse: Vector3<f32>) {}
    fn apply_torque(&mut self, _handle: Self::BodyHandle, _torque: Vector3<f32>) {}
    fn body_count(&self) -> usize { 0 }
    fn create_collider(&mut self, _body: Self::BodyHandle, _desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> { Err(BackendError::Unsupported("Not impl".into())) }
    fn remove_collider(&mut self, _handle: Self::ColliderHandle) -> Result<(), BackendError> { Ok(()) }
    fn set_collider_material(&mut self, _handle: Self::ColliderHandle, _material: PhysicsMaterial) {}
    fn set_collider_enabled(&mut self, _handle: Self::ColliderHandle, _enabled: bool) {}
    fn collider_aabb(&self, _handle: Self::ColliderHandle) -> Option<AABB> { None }
    fn create_constraint(&mut self, _desc: &ConstraintDesc) -> Result<Self::ConstraintHandle, BackendError> { Err(BackendError::Unsupported("Not impl".into())) }
    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> { Ok(()) }
    fn ray_cast(&self, _ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> { None }
    fn ray_cast_all(&self, _ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> { Vec::new() }
    fn shape_cast(&self, _cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> { None }
    fn query_aabb(&self, _aabb: &AABB) -> Vec<Self::BodyHandle> { Vec::new() }
    fn contacts(&self) -> &[ContactManifold] { &self.contacts }
    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> { Err(BackendError::Unsupported("Not impl".into())) }
    fn deserialize_state(&mut self, _data: &[u8]) -> Result<(), BackendError> { Err(BackendError::Unsupported("Not impl".into())) }
    fn reset(&mut self) {}
    fn stats(&self) -> SimulationStats { self.stats.clone() }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}
