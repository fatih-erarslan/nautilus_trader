//! Genesis differentiable physics backend
//!
//! GPU-accelerated differentiable simulation for robotics/ML.
//! Fork: https://github.com/fatih-erarslan/Genesis
//!
//! ## Python Integration
//!
//! Genesis is a Python-based physics engine. This backend uses PyO3 to
//! call the Genesis Python API from Rust.
//!
//! ### Requirements
//! - Python 3.8+ with Genesis installed: `pip install genesis-world`
//! - CUDA-capable GPU for GPU acceleration
//!
//! ### Example
//! ```rust,ignore
//! let config = GenesisConfig {
//!     device: "cuda:0".into(),
//!     precision: "float32".into(),
//!     batch_size: 1,
//!     scene_path: Some("scenes/humanoid.xml".into()),
//! };
//! let mut backend = GenesisBackend::new(config)?;
//! backend.step(1.0 / 60.0);
//! ```

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::BodyDesc;
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Vector3};
use std::any::Any;

#[cfg(feature = "genesis")]
use pyo3::prelude::*;
#[cfg(feature = "genesis")]
use pyo3::types::PyDict;

/// Genesis configuration
#[derive(Debug, Clone)]
pub struct GenesisConfig {
    /// CUDA device (e.g., "cuda:0", "cuda:1", or "cpu")
    pub device: String,
    /// Precision ("float32" or "float64")
    pub precision: String,
    /// Batch size for parallel simulation
    pub batch_size: usize,
    /// Path to scene file (MJCF, URDF, or Genesis XML)
    pub scene_path: Option<String>,
    /// Enable gradient computation
    pub requires_grad: bool,
}

impl Default for GenesisConfig {
    fn default() -> Self {
        Self {
            device: "cuda:0".into(),
            precision: "float32".into(),
            batch_size: 1,
            scene_path: None,
            requires_grad: false,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenesisBodyHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenesisColliderHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenesisConstraintHandle(pub u32);

/// Genesis differentiable physics backend
pub struct GenesisBackend {
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
    config: GenesisConfig,

    // Python objects when feature is enabled
    #[cfg(feature = "genesis")]
    py_scene: Option<PyObject>,
    #[cfg(feature = "genesis")]
    py_simulator: Option<PyObject>,

    body_count_cache: usize,
}

impl GenesisBackend {
    /// Initialize the Genesis Python environment
    #[cfg(feature = "genesis")]
    fn init_python(&mut self) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            // Import Genesis
            let genesis = py.import("genesis")
                .map_err(|e| BackendError::InitializationFailed(
                    format!("Failed to import genesis: {}. Install with: pip install genesis-world", e)
                ))?;

            // Initialize Genesis
            let init_kwargs = PyDict::new(py);
            init_kwargs.set_item("backend", self.config.device.as_str())?;
            init_kwargs.set_item("precision", self.config.precision.as_str())?;

            genesis.call_method("init", (), Some(init_kwargs))
                .map_err(|e| BackendError::InitializationFailed(format!("genesis.init failed: {}", e)))?;

            // Create scene if path is provided
            if let Some(ref scene_path) = self.config.scene_path {
                let scene = genesis.call_method1("Scene", (scene_path,))
                    .map_err(|e| BackendError::InitializationFailed(format!("Failed to create scene: {}", e)))?;

                self.py_scene = Some(scene.into());

                // Build the scene
                let scene_obj = self.py_scene.as_ref().unwrap().as_ref(py);
                scene_obj.call_method0("build")
                    .map_err(|e| BackendError::InitializationFailed(format!("scene.build failed: {}", e)))?;

                // Get body count
                let n_bodies: usize = scene_obj.getattr("n_entities")
                    .and_then(|n| n.extract())
                    .unwrap_or(0);
                self.body_count_cache = n_bodies;
            }

            Ok(())
        })
    }

    /// Step the simulation using Python
    #[cfg(feature = "genesis")]
    fn step_python(&mut self, dt: f32) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            if let Some(ref scene) = self.py_scene {
                let scene_obj = scene.as_ref(py);
                scene_obj.call_method1("step", (dt,))
                    .map_err(|e| BackendError::FfiError(format!("scene.step failed: {}", e)))?;
            }
            Ok(())
        })
    }

    /// Load a scene from file
    pub fn load_scene(&mut self, path: &str) -> Result<(), BackendError> {
        self.config.scene_path = Some(path.to_string());
        #[cfg(feature = "genesis")]
        {
            self.init_python()?;
        }
        #[cfg(not(feature = "genesis"))]
        {
            return Err(BackendError::Unsupported("Genesis feature not enabled".into()));
        }
        Ok(())
    }

    /// Get the gradient of the simulation (for differentiable physics)
    #[cfg(feature = "genesis")]
    pub fn get_gradients(&self) -> Result<Vec<f64>, BackendError> {
        if !self.config.requires_grad {
            return Err(BackendError::Unsupported("Gradients not enabled in config".into()));
        }

        Python::with_gil(|py| {
            if let Some(ref scene) = self.py_scene {
                let scene_obj = scene.as_ref(py);
                let grads = scene_obj.call_method0("get_gradients")
                    .map_err(|e| BackendError::FfiError(format!("get_gradients failed: {}", e)))?;

                let grads_list: Vec<f64> = grads.extract()
                    .map_err(|e| BackendError::FfiError(format!("Failed to extract gradients: {}", e)))?;

                Ok(grads_list)
            } else {
                Err(BackendError::Unsupported("No scene loaded".into()))
            }
        })
    }
}

impl PhysicsBackend for GenesisBackend {
    type Config = GenesisConfig;
    type BodyHandle = GenesisBodyHandle;
    type ColliderHandle = GenesisColliderHandle;
    type ConstraintHandle = GenesisConstraintHandle;

    fn new(config: Self::Config) -> Result<Self, BackendError> {
        let mut backend = Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            config: config.clone(),
            #[cfg(feature = "genesis")]
            py_scene: None,
            #[cfg(feature = "genesis")]
            py_simulator: None,
            body_count_cache: 0,
        };

        // Initialize if scene path is provided
        if config.scene_path.is_some() {
            #[cfg(feature = "genesis")]
            {
                backend.init_python()?;
            }
        }

        Ok(backend)
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "Genesis",
            version: "0.3",
            description: "GPU-accelerated differentiable physics",
            gpu_accelerated: true,
            differentiable: true,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            physics_3d: true,
            physics_2d: false,
            soft_bodies: true,
            cloth: true,
            fluids: true,
            articulated: true,
            ccd: false,
            deterministic: true,
            parallel: true,
            gpu: true,
            differentiable: true,
            max_bodies: 0,
        }
    }

    fn step(&mut self, dt: f32) {
        let start = std::time::Instant::now();

        #[cfg(feature = "genesis")]
        {
            let _ = self.step_python(dt);
        }

        self.stats.total_us = start.elapsed().as_micros() as u64;
        self.stats.active_bodies = self.body_count_cache as u32;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) { self.gravity = gravity; }
    fn gravity(&self) -> Vector3<f32> { self.gravity }

    fn create_body(&mut self, _desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        Err(BackendError::Unsupported("Genesis uses scene files - load with load_scene()".into()))
    }
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
    fn body_count(&self) -> usize { self.body_count_cache }
    fn create_collider(&mut self, _body: Self::BodyHandle, _desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> {
        Err(BackendError::Unsupported("Genesis Python API required".into()))
    }
    fn remove_collider(&mut self, _handle: Self::ColliderHandle) -> Result<(), BackendError> { Ok(()) }
    fn set_collider_material(&mut self, _handle: Self::ColliderHandle, _material: PhysicsMaterial) {}
    fn set_collider_enabled(&mut self, _handle: Self::ColliderHandle, _enabled: bool) {}
    fn collider_aabb(&self, _handle: Self::ColliderHandle) -> Option<AABB> { None }
    fn create_constraint(&mut self, _desc: &ConstraintDesc) -> Result<Self::ConstraintHandle, BackendError> {
        Err(BackendError::Unsupported("Not implemented".into()))
    }
    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> { Ok(()) }
    fn ray_cast(&self, _ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> { None }
    fn ray_cast_all(&self, _ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> { Vec::new() }
    fn shape_cast(&self, _cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> { None }
    fn query_aabb(&self, _aabb: &AABB) -> Vec<Self::BodyHandle> { Vec::new() }
    fn contacts(&self) -> &[ContactManifold] { &self.contacts }
    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> { Err(BackendError::Unsupported("Not implemented".into())) }
    fn deserialize_state(&mut self, _data: &[u8]) -> Result<(), BackendError> { Err(BackendError::Unsupported("Not implemented".into())) }
    fn reset(&mut self) {}
    fn stats(&self) -> SimulationStats { self.stats.clone() }
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
}
