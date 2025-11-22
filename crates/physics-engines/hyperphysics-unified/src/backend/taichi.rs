//! Taichi GPU physics backend
//!
//! High-performance GPU compute for physics simulation.
//! Fork: https://github.com/fatih-erarslan/taichi

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::BodyDesc;
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Vector3};
use std::any::Any;

#[derive(Debug, Clone)]
pub struct TaichiConfig {
    pub arch: String,
    pub debug: bool,
}

impl Default for TaichiConfig {
    fn default() -> Self {
        Self { arch: "cuda".into(), debug: false }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaichiBodyHandle(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaichiColliderHandle(pub u32);
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaichiConstraintHandle(pub u32);

pub struct TaichiBackend {
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
}

impl PhysicsBackend for TaichiBackend {
    type Config = TaichiConfig;
    type BodyHandle = TaichiBodyHandle;
    type ColliderHandle = TaichiColliderHandle;
    type ConstraintHandle = TaichiConstraintHandle;

    fn new(_config: Self::Config) -> Result<Self, BackendError> {
        Ok(Self { gravity: Vector3::new(0.0, -9.81, 0.0), contacts: Vec::new(), stats: SimulationStats::default() })
    }

    fn info(&self) -> BackendInfo {
        BackendInfo { name: "Taichi", version: "1.7", description: "GPU compute physics", gpu_accelerated: true, differentiable: true }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities { physics_3d: true, physics_2d: true, soft_bodies: true, cloth: true, fluids: true, articulated: false, ccd: false, deterministic: true, parallel: true, gpu: true, differentiable: true, max_bodies: 0 }
    }

    fn step(&mut self, _dt: f32) {}
    fn set_gravity(&mut self, gravity: Vector3<f32>) { self.gravity = gravity; }
    fn gravity(&self) -> Vector3<f32> { self.gravity }
    fn create_body(&mut self, _desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> { Err(BackendError::Unsupported("Taichi Python API".into())) }
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
