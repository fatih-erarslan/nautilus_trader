//! MuJoCo physics backend
//!
//! High-fidelity robotics simulation with contact-rich dynamics.
//! Fork: https://github.com/fatih-erarslan/mujoco
//!
//! ## Model Loading
//!
//! MuJoCo uses MJCF (MuJoCo XML Format) for model definitions.
//! Unlike other backends, bodies and constraints are defined in XML
//! rather than created programmatically.
//!
//! ```rust,ignore
//! let mut backend = MujocoBackend::new(MujocoConfig::default())?;
//! backend.load_mjcf("models/humanoid.xml")?;
//! backend.step(0.002);
//! ```

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::BodyDesc;
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Vector3};
use std::any::Any;
use std::path::Path;

// Import MuJoCo adapter when feature is enabled
#[cfg(feature = "mujoco")]
use mujoco_hyperphysics::MuJoCoAdapter;

/// MuJoCo configuration
#[derive(Debug, Clone)]
pub struct MujocoConfig {
    /// Simulation timestep
    pub timestep: f64,
    /// Number of constraint solver iterations
    pub iterations: i32,
    /// Tolerance for solver convergence
    pub tolerance: f64,
    /// Enable multi-threaded simulation
    pub nthread: i32,
    /// Path to MJCF model file (optional, can be loaded later)
    pub model_path: Option<String>,
}

impl Default for MujocoConfig {
    fn default() -> Self {
        Self {
            timestep: 0.002,
            iterations: 100,
            tolerance: 1e-8,
            nthread: 0, // Auto
            model_path: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MujocoBodyHandle(pub i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MujocoColliderHandle(pub i32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MujocoConstraintHandle(pub i32);

/// MuJoCo physics backend
pub struct MujocoBackend {
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
    config: MujocoConfig,

    // Real MuJoCo adapter when feature is enabled
    #[cfg(feature = "mujoco")]
    adapter: Option<MuJoCoAdapter>,

    // Cached body count from model
    body_count_cache: usize,
}

impl MujocoBackend {
    /// Load an MJCF model from file
    pub fn load_mjcf<P: AsRef<Path>>(&mut self, path: P) -> Result<(), BackendError> {
        #[cfg(feature = "mujoco")]
        {
            if self.adapter.is_none() {
                self.adapter = Some(MuJoCoAdapter::new(
                    mujoco_hyperphysics::MuJoCoConfig {
                        timestep: self.config.timestep,
                        gravity: nalgebra::Vector3::new(
                            self.gravity.x as f64,
                            self.gravity.y as f64,
                            self.gravity.z as f64,
                        ),
                        enable_contact: true,
                        solver_iterations: self.config.iterations,
                    }
                ));
            }

            if let Some(ref mut adapter) = self.adapter {
                adapter.load_xml(path)
                    .map_err(|e| BackendError::InitializationFailed(e.to_string()))?;
                self.body_count_cache = adapter.body_count() as usize;
            }
        }

        #[cfg(not(feature = "mujoco"))]
        {
            let _ = path;
            return Err(BackendError::Unsupported("MuJoCo feature not enabled".into()));
        }

        Ok(())
    }

    /// Check if a model is loaded
    pub fn is_model_loaded(&self) -> bool {
        #[cfg(feature = "mujoco")]
        {
            self.adapter.as_ref().map(|a| a.is_loaded()).unwrap_or(false)
        }
        #[cfg(not(feature = "mujoco"))]
        {
            false
        }
    }

    /// Get generalized coordinates (joint positions)
    pub fn get_qpos(&self) -> Vec<f64> {
        #[cfg(feature = "mujoco")]
        {
            self.adapter.as_ref().map(|a| a.get_qpos()).unwrap_or_default()
        }
        #[cfg(not(feature = "mujoco"))]
        {
            Vec::new()
        }
    }

    /// Set generalized coordinates
    pub fn set_qpos(&mut self, qpos: &[f64]) -> Result<(), BackendError> {
        #[cfg(feature = "mujoco")]
        {
            if let Some(ref mut adapter) = self.adapter {
                adapter.set_qpos(qpos)
                    .map_err(|e| BackendError::FfiError(e.to_string()))?;
            }
        }
        #[cfg(not(feature = "mujoco"))]
        {
            let _ = qpos;
        }
        Ok(())
    }

    /// Get generalized velocities
    pub fn get_qvel(&self) -> Vec<f64> {
        #[cfg(feature = "mujoco")]
        {
            self.adapter.as_ref().map(|a| a.get_qvel()).unwrap_or_default()
        }
        #[cfg(not(feature = "mujoco"))]
        {
            Vec::new()
        }
    }

    /// Set generalized velocities
    pub fn set_qvel(&mut self, qvel: &[f64]) -> Result<(), BackendError> {
        #[cfg(feature = "mujoco")]
        {
            if let Some(ref mut adapter) = self.adapter {
                adapter.set_qvel(qvel)
                    .map_err(|e| BackendError::FfiError(e.to_string()))?;
            }
        }
        #[cfg(not(feature = "mujoco"))]
        {
            let _ = qvel;
        }
        Ok(())
    }

    /// Get actuator control signals
    pub fn get_ctrl(&self) -> Vec<f64> {
        #[cfg(feature = "mujoco")]
        {
            self.adapter.as_ref().map(|a| a.get_ctrl()).unwrap_or_default()
        }
        #[cfg(not(feature = "mujoco"))]
        {
            Vec::new()
        }
    }

    /// Set actuator control signals
    pub fn set_ctrl(&mut self, ctrl: &[f64]) -> Result<(), BackendError> {
        #[cfg(feature = "mujoco")]
        {
            if let Some(ref mut adapter) = self.adapter {
                adapter.set_ctrl(ctrl)
                    .map_err(|e| BackendError::FfiError(e.to_string()))?;
            }
        }
        #[cfg(not(feature = "mujoco"))]
        {
            let _ = ctrl;
        }
        Ok(())
    }

    /// Get simulation time
    pub fn simulation_time(&self) -> f64 {
        #[cfg(feature = "mujoco")]
        {
            self.adapter.as_ref().map(|a| a.time()).unwrap_or(0.0)
        }
        #[cfg(not(feature = "mujoco"))]
        {
            0.0
        }
    }
}

impl PhysicsBackend for MujocoBackend {
    type Config = MujocoConfig;
    type BodyHandle = MujocoBodyHandle;
    type ColliderHandle = MujocoColliderHandle;
    type ConstraintHandle = MujocoConstraintHandle;

    fn new(config: Self::Config) -> Result<Self, BackendError> {
        let mut backend = Self {
            gravity: Vector3::new(0.0, 0.0, -9.81), // MuJoCo uses Z-up
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            config: config.clone(),
            #[cfg(feature = "mujoco")]
            adapter: None,
            body_count_cache: 0,
        };

        // Load model if path is provided
        if let Some(ref path) = config.model_path {
            backend.load_mjcf(path)?;
        }

        Ok(backend)
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "MuJoCo",
            version: "3.1",
            description: "Multi-Joint dynamics with Contact (robotics)",
            gpu_accelerated: false,
            differentiable: true,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            physics_3d: true,
            physics_2d: false,
            soft_bodies: true,
            cloth: false,
            fluids: false,
            articulated: true,
            ccd: false,
            deterministic: true,
            parallel: true,
            gpu: false,
            differentiable: true,
            max_bodies: 0,
        }
    }

    fn step(&mut self, _dt: f32) {
        let start = std::time::Instant::now();

        #[cfg(feature = "mujoco")]
        if let Some(ref mut adapter) = self.adapter {
            let _ = adapter.step();
        }

        self.stats.total_us = start.elapsed().as_micros() as u64;
        self.stats.active_bodies = self.body_count_cache as u32;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.gravity = gravity;
    }

    fn gravity(&self) -> Vector3<f32> {
        self.gravity
    }

    fn create_body(&mut self, _desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        Err(BackendError::Unsupported("MuJoCo uses MJCF/XML models".into()))
    }

    fn remove_body(&mut self, _handle: Self::BodyHandle) -> Result<(), BackendError> {
        Err(BackendError::Unsupported("MuJoCo models are static".into()))
    }

    fn body_transform(&self, _handle: Self::BodyHandle) -> Option<Transform> {
        None
    }

    fn set_body_transform(&mut self, _handle: Self::BodyHandle, _transform: Transform) {}

    fn body_linear_velocity(&self, _handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        None
    }

    fn set_body_linear_velocity(&mut self, _handle: Self::BodyHandle, _velocity: Vector3<f32>) {}

    fn body_angular_velocity(&self, _handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        None
    }

    fn set_body_angular_velocity(&mut self, _handle: Self::BodyHandle, _velocity: Vector3<f32>) {}

    fn apply_force(&mut self, _handle: Self::BodyHandle, _force: Vector3<f32>) {}

    fn apply_force_at_point(&mut self, _handle: Self::BodyHandle, _force: Vector3<f32>, _point: Point3<f32>) {}

    fn apply_impulse(&mut self, _handle: Self::BodyHandle, _impulse: Vector3<f32>) {}

    fn apply_torque(&mut self, _handle: Self::BodyHandle, _torque: Vector3<f32>) {}

    fn body_count(&self) -> usize {
        self.body_count_cache
    }

    fn create_collider(&mut self, _body: Self::BodyHandle, _desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> {
        Err(BackendError::Unsupported("MuJoCo uses MJCF/XML models".into()))
    }

    fn remove_collider(&mut self, _handle: Self::ColliderHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn set_collider_material(&mut self, _handle: Self::ColliderHandle, _material: PhysicsMaterial) {}

    fn set_collider_enabled(&mut self, _handle: Self::ColliderHandle, _enabled: bool) {}

    fn collider_aabb(&self, _handle: Self::ColliderHandle) -> Option<AABB> {
        None
    }

    fn create_constraint(&mut self, _desc: &ConstraintDesc) -> Result<Self::ConstraintHandle, BackendError> {
        Err(BackendError::Unsupported("Use MJCF constraints".into()))
    }

    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn ray_cast(&self, _ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> {
        None
    }

    fn ray_cast_all(&self, _ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> {
        Vec::new()
    }

    fn shape_cast(&self, _cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> {
        None
    }

    fn query_aabb(&self, _aabb: &AABB) -> Vec<Self::BodyHandle> {
        Vec::new()
    }

    fn contacts(&self) -> &[ContactManifold] {
        &self.contacts
    }

    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> {
        Err(BackendError::Unsupported("Not implemented".into()))
    }

    fn deserialize_state(&mut self, _data: &[u8]) -> Result<(), BackendError> {
        Err(BackendError::Unsupported("Not implemented".into()))
    }

    fn reset(&mut self) {
        #[cfg(feature = "mujoco")]
        if let Some(ref mut adapter) = self.adapter {
            let _ = adapter.reset();
        }
    }

    fn stats(&self) -> SimulationStats {
        self.stats.clone()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
