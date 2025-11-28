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
//! - PyTorch 2.8+ required
//! - CUDA-capable GPU for GPU acceleration
//!
//! ### Supported Solvers
//! - RigidSolver: Rigid body dynamics
//! - MPMSolver: Material Point Method
//! - SPHSolver: Smoothed Particle Hydrodynamics
//! - FEMSolver: Finite Element Method
//! - PBDSolver: Position-Based Dynamics
//! - SFSolver: Stable Fluids
//!
//! ### Example
//! ```rust,ignore
//! let config = GenesisConfig {
//!     backend: GenesisBackendType::Gpu,
//!     precision: GenesisPrecision::Float32,
//!     requires_grad: true,
//!     ..Default::default()
//! };
//! let mut backend = GenesisBackend::new(config)?;
//!
//! // Load a URDF robot
//! backend.load_urdf("robot.urdf")?;
//! backend.build()?;
//!
//! // Simulate
//! for _ in 0..1000 {
//!     backend.step(1.0 / 60.0);
//! }
//!
//! // Get gradients for optimization
//! let grads = backend.compute_gradients()?;
//! ```

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::BodyDesc;
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use std::any::Any;
use std::collections::HashMap;
use std::path::Path;

#[cfg(feature = "genesis")]
use pyo3::prelude::*;
#[cfg(feature = "genesis")]
use pyo3::types::{PyDict, PyList, PyTuple};

/// Genesis compute backend
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisBackendType {
    /// CPU backend
    Cpu,
    /// GPU backend (CUDA)
    Gpu,
    /// Metal backend (macOS)
    Metal,
    /// Vulkan backend
    Vulkan,
}

impl Default for GenesisBackendType {
    fn default() -> Self {
        Self::Gpu
    }
}

/// Genesis precision mode
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisPrecision {
    /// 32-bit floating point
    Float32,
    /// 64-bit floating point
    Float64,
}

impl Default for GenesisPrecision {
    fn default() -> Self {
        Self::Float32
    }
}

/// Genesis integrator type
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisIntegrator {
    /// Euler integration (fastest)
    Euler,
    /// Implicit Euler (more stable)
    ImplicitEuler,
}

impl Default for GenesisIntegrator {
    fn default() -> Self {
        Self::Euler
    }
}

/// Genesis constraint solver
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisConstraintSolver {
    /// Sequential Gauss-Seidel
    Pgs,
    /// Newton method
    Newton,
    /// CCD-based solver
    Ccd,
}

impl Default for GenesisConstraintSolver {
    fn default() -> Self {
        Self::Newton
    }
}

/// Genesis configuration
#[derive(Debug, Clone)]
pub struct GenesisConfig {
    /// Compute backend
    pub backend: GenesisBackendType,
    /// Precision ("float32" or "float64")
    pub precision: GenesisPrecision,
    /// Enable gradient computation
    pub requires_grad: bool,
    /// Time step for simulation
    pub dt: f32,
    /// Substeps per frame
    pub substeps: u32,
    /// Integrator type
    pub integrator: GenesisIntegrator,
    /// Constraint solver
    pub constraint_solver: GenesisConstraintSolver,
    /// Enable debug mode
    pub debug: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Enable viewer (for visualization)
    pub show_viewer: bool,
    /// Gravity vector
    pub gravity: Vector3<f32>,
}

impl Default for GenesisConfig {
    fn default() -> Self {
        Self {
            backend: GenesisBackendType::default(),
            precision: GenesisPrecision::default(),
            requires_grad: false,
            dt: 1.0 / 60.0,
            substeps: 1,
            integrator: GenesisIntegrator::default(),
            constraint_solver: GenesisConstraintSolver::default(),
            debug: false,
            seed: None,
            show_viewer: false,
            gravity: Vector3::new(0.0, -9.81, 0.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenesisBodyHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenesisColliderHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GenesisConstraintHandle(pub u32);

/// Entity type in Genesis
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenesisEntityType {
    /// Rigid body
    Rigid,
    /// URDF robot
    Urdf,
    /// MJCF model
    Mjcf,
    /// Terrain
    Terrain,
    /// Soft body (MPM/FEM)
    Soft,
    /// Fluid (SPH)
    Fluid,
    /// Cloth (PBD)
    Cloth,
}

/// Genesis entity information
#[derive(Debug, Clone)]
struct GenesisEntity {
    handle: GenesisBodyHandle,
    entity_type: GenesisEntityType,
    name: String,
    n_links: usize,
    n_dofs: usize,
    is_built: bool,
}

/// Genesis differentiable physics backend
pub struct GenesisBackend {
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
    config: GenesisConfig,

    // Python objects when feature is enabled
    #[cfg(feature = "genesis")]
    gs_module: Option<PyObject>,
    #[cfg(feature = "genesis")]
    scene: Option<PyObject>,
    #[cfg(feature = "genesis")]
    entities: HashMap<u32, PyObject>,

    // Entity management
    entity_info: HashMap<u32, GenesisEntity>,
    next_entity_id: u32,

    // State tracking
    initialized: bool,
    is_built: bool,
    sim_time: f64,
}

impl GenesisBackend {
    /// Initialize the Genesis Python environment
    #[cfg(feature = "genesis")]
    fn init_python(&mut self) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            // Import Genesis
            let gs = py.import("genesis")
                .map_err(|e| BackendError::InitializationFailed(
                    format!("Failed to import genesis: {}. Install with: pip install genesis-world", e)
                ))?;

            // Initialize Genesis
            let init_kwargs = PyDict::new(py);

            // Set backend
            let backend = match self.config.backend {
                GenesisBackendType::Cpu => gs.getattr("cpu")?,
                GenesisBackendType::Gpu => gs.getattr("gpu")?,
                GenesisBackendType::Metal => gs.getattr("metal")?,
                GenesisBackendType::Vulkan => gs.getattr("vulkan")?,
            };
            init_kwargs.set_item("backend", backend)?;

            // Set precision
            let precision = match self.config.precision {
                GenesisPrecision::Float32 => "32",
                GenesisPrecision::Float64 => "64",
            };
            init_kwargs.set_item("precision", precision)?;

            // Set seed if provided
            if let Some(seed) = self.config.seed {
                init_kwargs.set_item("seed", seed)?;
            }

            // Set debug mode
            init_kwargs.set_item("debug", self.config.debug)?;

            // Initialize
            gs.call_method("init", (), Some(init_kwargs))
                .map_err(|e| BackendError::InitializationFailed(format!("gs.init failed: {}", e)))?;

            self.gs_module = Some(gs.into());
            self.initialized = true;

            // Create scene
            self.create_scene(py)?;

            Ok(())
        })
    }

    #[cfg(feature = "genesis")]
    fn create_scene(&mut self, py: Python<'_>) -> Result<(), BackendError> {
        let gs = self.gs_module.as_ref().unwrap().bind(py);

        // Create SimOptions
        let sim_options_class = gs.getattr("options")?.getattr("SimOptions")?;
        let sim_kwargs = PyDict::new(py);
        sim_kwargs.set_item("dt", self.config.dt)?;
        sim_kwargs.set_item("substeps", self.config.substeps)?;
        sim_kwargs.set_item("requires_grad", self.config.requires_grad)?;

        // Set gravity
        let gravity_list = PyList::new(py, &[self.gravity.x, self.gravity.y, self.gravity.z])?;
        sim_kwargs.set_item("gravity", gravity_list)?;

        let sim_options = sim_options_class.call((), Some(sim_kwargs))?;

        // Create RigidOptions
        let rigid_options_class = gs.getattr("options")?.getattr("RigidOptions")?;
        let rigid_kwargs = PyDict::new(py);

        // Set constraint solver
        let solver = match self.config.constraint_solver {
            GenesisConstraintSolver::Pgs => gs.getattr("constraint_solver")?.getattr("PGS")?,
            GenesisConstraintSolver::Newton => gs.getattr("constraint_solver")?.getattr("Newton")?,
            GenesisConstraintSolver::Ccd => gs.getattr("constraint_solver")?.getattr("CCD")?,
        };
        rigid_kwargs.set_item("constraint_solver", solver)?;

        let rigid_options = rigid_options_class.call((), Some(rigid_kwargs))?;

        // Create VisOptions
        let vis_options_class = gs.getattr("options")?.getattr("VisOptions")?;
        let vis_kwargs = PyDict::new(py);
        let vis_options = vis_options_class.call((), Some(vis_kwargs))?;

        // Create Scene
        let scene_class = gs.getattr("Scene")?;
        let scene_kwargs = PyDict::new(py);
        scene_kwargs.set_item("sim_options", sim_options)?;
        scene_kwargs.set_item("rigid_options", rigid_options)?;
        scene_kwargs.set_item("vis_options", vis_options)?;
        scene_kwargs.set_item("show_viewer", self.config.show_viewer)?;

        let scene = scene_class.call((), Some(scene_kwargs))?;
        self.scene = Some(scene.into());

        Ok(())
    }

    /// Build the scene (compile kernels and prepare for simulation)
    pub fn build(&mut self) -> Result<(), BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                if let Some(ref scene) = self.scene {
                    let scene = scene.bind(py);
                    scene.call_method0("build")
                        .map_err(|e| BackendError::InitializationFailed(format!("scene.build failed: {}", e)))?;
                }
                Ok(())
            })?;
        }

        self.is_built = true;
        Ok(())
    }

    /// Load a URDF robot model
    pub fn load_urdf(&mut self, path: &str) -> Result<GenesisBodyHandle, BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let gs = self.gs_module.as_ref().ok_or_else(|| {
                    BackendError::InitializationFailed("Genesis not initialized".into())
                })?.bind(py);
                let scene = self.scene.as_ref().ok_or_else(|| {
                    BackendError::InitializationFailed("Scene not created".into())
                })?.bind(py);

                // Create URDF morph
                let morphs = gs.getattr("morphs")?;
                let urdf_class = morphs.getattr("URDF")?;
                let morph = urdf_class.call1((path,))?;

                // Create Rigid material
                let materials = gs.getattr("materials")?;
                let rigid_class = materials.getattr("Rigid")?;
                let material = rigid_class.call0()?;

                // Add entity to scene
                let entity = scene.call_method("add_entity", (morph,), Some({
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("material", material)?;
                    kwargs
                }))?;

                let handle = GenesisBodyHandle(self.next_entity_id);
                self.next_entity_id += 1;

                // Get entity info
                let n_links: usize = entity.getattr("n_links")
                    .and_then(|n| n.extract())
                    .unwrap_or(1);
                let n_dofs: usize = entity.getattr("n_dofs")
                    .and_then(|n| n.extract())
                    .unwrap_or(0);

                self.entities.insert(handle.0, entity.into());
                self.entity_info.insert(handle.0, GenesisEntity {
                    handle,
                    entity_type: GenesisEntityType::Urdf,
                    name: path.to_string(),
                    n_links,
                    n_dofs,
                    is_built: false,
                });

                Ok(handle)
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Load an MJCF model
    pub fn load_mjcf(&mut self, path: &str) -> Result<GenesisBodyHandle, BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let gs = self.gs_module.as_ref().ok_or_else(|| {
                    BackendError::InitializationFailed("Genesis not initialized".into())
                })?.bind(py);
                let scene = self.scene.as_ref().ok_or_else(|| {
                    BackendError::InitializationFailed("Scene not created".into())
                })?.bind(py);

                let morphs = gs.getattr("morphs")?;
                let mjcf_class = morphs.getattr("MJCF")?;
                let morph = mjcf_class.call1((path,))?;

                let materials = gs.getattr("materials")?;
                let rigid_class = materials.getattr("Rigid")?;
                let material = rigid_class.call0()?;

                let entity = scene.call_method("add_entity", (morph,), Some({
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("material", material)?;
                    kwargs
                }))?;

                let handle = GenesisBodyHandle(self.next_entity_id);
                self.next_entity_id += 1;

                let n_links: usize = entity.getattr("n_links")
                    .and_then(|n| n.extract())
                    .unwrap_or(1);
                let n_dofs: usize = entity.getattr("n_dofs")
                    .and_then(|n| n.extract())
                    .unwrap_or(0);

                self.entities.insert(handle.0, entity.into());
                self.entity_info.insert(handle.0, GenesisEntity {
                    handle,
                    entity_type: GenesisEntityType::Mjcf,
                    name: path.to_string(),
                    n_links,
                    n_dofs,
                    is_built: false,
                });

                Ok(handle)
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Add a primitive shape (box, sphere, capsule, etc.)
    pub fn add_primitive(&mut self, primitive_type: &str, size: [f32; 3], pos: [f32; 3]) -> Result<GenesisBodyHandle, BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let gs = self.gs_module.as_ref().ok_or_else(|| {
                    BackendError::InitializationFailed("Genesis not initialized".into())
                })?.bind(py);
                let scene = self.scene.as_ref().ok_or_else(|| {
                    BackendError::InitializationFailed("Scene not created".into())
                })?.bind(py);

                let morphs = gs.getattr("morphs")?;

                // Create morph based on type
                let morph = match primitive_type {
                    "box" | "Box" => {
                        let box_class = morphs.getattr("Box")?;
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("size", PyTuple::new(py, &size)?)?;
                        kwargs.set_item("pos", PyTuple::new(py, &pos)?)?;
                        box_class.call((), Some(kwargs))?
                    }
                    "sphere" | "Sphere" => {
                        let sphere_class = morphs.getattr("Sphere")?;
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("radius", size[0])?;
                        kwargs.set_item("pos", PyTuple::new(py, &pos)?)?;
                        sphere_class.call((), Some(kwargs))?
                    }
                    "capsule" | "Capsule" => {
                        let capsule_class = morphs.getattr("Capsule")?;
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("radius", size[0])?;
                        kwargs.set_item("length", size[1])?;
                        kwargs.set_item("pos", PyTuple::new(py, &pos)?)?;
                        capsule_class.call((), Some(kwargs))?
                    }
                    "cylinder" | "Cylinder" => {
                        let cylinder_class = morphs.getattr("Cylinder")?;
                        let kwargs = PyDict::new(py);
                        kwargs.set_item("radius", size[0])?;
                        kwargs.set_item("length", size[1])?;
                        kwargs.set_item("pos", PyTuple::new(py, &pos)?)?;
                        cylinder_class.call((), Some(kwargs))?
                    }
                    "plane" | "Plane" => {
                        let plane_class = morphs.getattr("Plane")?;
                        plane_class.call0()?
                    }
                    _ => return Err(BackendError::Unsupported(format!("Unknown primitive type: {}", primitive_type))),
                };

                let materials = gs.getattr("materials")?;
                let rigid_class = materials.getattr("Rigid")?;
                let material = rigid_class.call0()?;

                let entity = scene.call_method("add_entity", (morph,), Some({
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("material", material)?;
                    kwargs
                }))?;

                let handle = GenesisBodyHandle(self.next_entity_id);
                self.next_entity_id += 1;

                self.entities.insert(handle.0, entity.into());
                self.entity_info.insert(handle.0, GenesisEntity {
                    handle,
                    entity_type: GenesisEntityType::Rigid,
                    name: primitive_type.to_string(),
                    n_links: 1,
                    n_dofs: 0,
                    is_built: false,
                });

                Ok(handle)
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Get joint positions for articulated body
    pub fn get_qpos(&self, handle: GenesisBodyHandle) -> Result<Vec<f64>, BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let entity = self.entities.get(&handle.0).ok_or_else(|| {
                    BackendError::InvalidHandle("Entity not found".into())
                })?.bind(py);

                let qpos = entity.call_method0("get_qpos")?;
                let qpos_list: Vec<f64> = qpos.call_method0("tolist")?.extract()?;
                Ok(qpos_list)
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Set joint positions for articulated body
    pub fn set_qpos(&mut self, handle: GenesisBodyHandle, qpos: &[f64]) -> Result<(), BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let entity = self.entities.get(&handle.0).ok_or_else(|| {
                    BackendError::InvalidHandle("Entity not found".into())
                })?.bind(py);

                let qpos_list = PyList::new(py, qpos)?;
                entity.call_method1("set_qpos", (qpos_list,))?;
                Ok(())
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Get joint velocities for articulated body
    pub fn get_qvel(&self, handle: GenesisBodyHandle) -> Result<Vec<f64>, BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let entity = self.entities.get(&handle.0).ok_or_else(|| {
                    BackendError::InvalidHandle("Entity not found".into())
                })?.bind(py);

                let qvel = entity.call_method0("get_qvel")?;
                let qvel_list: Vec<f64> = qvel.call_method0("tolist")?.extract()?;
                Ok(qvel_list)
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Set joint velocities for articulated body
    pub fn set_qvel(&mut self, handle: GenesisBodyHandle, qvel: &[f64]) -> Result<(), BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let entity = self.entities.get(&handle.0).ok_or_else(|| {
                    BackendError::InvalidHandle("Entity not found".into())
                })?.bind(py);

                let qvel_list = PyList::new(py, qvel)?;
                entity.call_method1("set_qvel", (qvel_list,))?;
                Ok(())
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Set control (actuator inputs) for articulated body
    pub fn set_control(&mut self, handle: GenesisBodyHandle, control: &[f64]) -> Result<(), BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let entity = self.entities.get(&handle.0).ok_or_else(|| {
                    BackendError::InvalidHandle("Entity not found".into())
                })?.bind(py);

                let ctrl_list = PyList::new(py, control)?;
                entity.call_method1("set_dofs_force", (ctrl_list,))?;
                Ok(())
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Compute gradients for differentiable physics
    #[cfg(feature = "genesis")]
    pub fn compute_gradients(&self) -> Result<HashMap<String, Vec<f64>>, BackendError> {
        if !self.config.requires_grad {
            return Err(BackendError::Unsupported("Gradients not enabled in config".into()));
        }

        Python::with_gil(|py| {
            let mut gradients = HashMap::new();

            for (id, entity_obj) in &self.entities {
                let entity = entity_obj.bind(py);

                // Get position gradients
                if let Ok(grad_qpos) = entity.call_method0("get_grad_qpos") {
                    if let Ok(grad_list) = grad_qpos.call_method0("tolist") {
                        if let Ok(grads) = grad_list.extract::<Vec<f64>>() {
                            gradients.insert(format!("entity_{}_qpos_grad", id), grads);
                        }
                    }
                }

                // Get velocity gradients
                if let Ok(grad_qvel) = entity.call_method0("get_grad_qvel") {
                    if let Ok(grad_list) = grad_qvel.call_method0("tolist") {
                        if let Ok(grads) = grad_list.extract::<Vec<f64>>() {
                            gradients.insert(format!("entity_{}_qvel_grad", id), grads);
                        }
                    }
                }
            }

            Ok(gradients)
        })
    }

    /// Step the simulation using Python
    #[cfg(feature = "genesis")]
    fn step_python(&mut self, _dt: f32) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            if let Some(ref scene) = self.scene {
                let scene = scene.bind(py);
                scene.call_method0("step")
                    .map_err(|e| BackendError::FfiError(format!("scene.step failed: {}", e)))?;
            }
            Ok(())
        })
    }

    /// Get link position for a specific link of an articulated body
    pub fn get_link_pos(&self, handle: GenesisBodyHandle, link_idx: usize) -> Result<[f64; 3], BackendError> {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                let entity = self.entities.get(&handle.0).ok_or_else(|| {
                    BackendError::InvalidHandle("Entity not found".into())
                })?.bind(py);

                let links = entity.getattr("links")?;
                let link = links.get_item(link_idx)?;
                let pos = link.call_method0("get_pos")?;
                let pos_list: Vec<f64> = pos.call_method0("tolist")?.extract()?;

                Ok([
                    pos_list.get(0).copied().unwrap_or(0.0),
                    pos_list.get(1).copied().unwrap_or(0.0),
                    pos_list.get(2).copied().unwrap_or(0.0),
                ])
            })
        }

        #[cfg(not(feature = "genesis"))]
        Err(BackendError::Unsupported("Genesis feature not enabled".into()))
    }

    /// Get number of links for an entity
    pub fn get_n_links(&self, handle: GenesisBodyHandle) -> Option<usize> {
        self.entity_info.get(&handle.0).map(|e| e.n_links)
    }

    /// Get number of DOFs for an entity
    pub fn get_n_dofs(&self, handle: GenesisBodyHandle) -> Option<usize> {
        self.entity_info.get(&handle.0).map(|e| e.n_dofs)
    }
}

impl PhysicsBackend for GenesisBackend {
    type Config = GenesisConfig;
    type BodyHandle = GenesisBodyHandle;
    type ColliderHandle = GenesisColliderHandle;
    type ConstraintHandle = GenesisConstraintHandle;

    fn new(config: Self::Config) -> Result<Self, BackendError> {
        let gravity = config.gravity;
        let mut backend = Self {
            gravity,
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            config,
            #[cfg(feature = "genesis")]
            gs_module: None,
            #[cfg(feature = "genesis")]
            scene: None,
            #[cfg(feature = "genesis")]
            entities: HashMap::new(),
            entity_info: HashMap::new(),
            next_entity_id: 0,
            initialized: false,
            is_built: false,
            sim_time: 0.0,
        };

        #[cfg(feature = "genesis")]
        {
            backend.init_python()?;
        }

        Ok(backend)
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "Genesis",
            version: "0.3",
            description: "GPU-accelerated differentiable physics for robotics/ML",
            gpu_accelerated: true,
            differentiable: self.config.requires_grad,
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
            ccd: matches!(self.config.constraint_solver, GenesisConstraintSolver::Ccd),
            deterministic: true,
            parallel: true,
            gpu: !matches!(self.config.backend, GenesisBackendType::Cpu),
            differentiable: self.config.requires_grad,
            max_bodies: 0, // No hard limit
        }
    }

    fn step(&mut self, dt: f32) {
        let start = std::time::Instant::now();

        if !self.is_built {
            tracing::warn!("Scene not built - call build() before stepping");
            return;
        }

        #[cfg(feature = "genesis")]
        {
            let _ = self.step_python(dt);
        }

        self.sim_time += dt as f64;
        self.stats.total_us = start.elapsed().as_micros() as u64;
        self.stats.active_bodies = self.entity_info.len() as u32;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.gravity = gravity;
        // Updating gravity in an existing scene would require rebuilding
    }

    fn gravity(&self) -> Vector3<f32> {
        self.gravity
    }

    fn create_body(&mut self, desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        // Use add_primitive for basic shapes
        let pos = [desc.transform.position.x, desc.transform.position.y, desc.transform.position.z];
        let size = [0.1, 0.1, 0.1]; // Default size
        self.add_primitive("box", size, pos)
    }

    fn remove_body(&mut self, handle: Self::BodyHandle) -> Result<(), BackendError> {
        self.entity_info.remove(&handle.0);
        #[cfg(feature = "genesis")]
        {
            self.entities.remove(&handle.0);
        }
        Ok(())
    }

    fn body_transform(&self, handle: Self::BodyHandle) -> Option<Transform> {
        #[cfg(feature = "genesis")]
        {
            if let Ok(pos) = self.get_link_pos(handle, 0) {
                return Some(Transform {
                    position: Point3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32),
                    rotation: UnitQuaternion::identity(),
                });
            }
        }
        None
    }

    fn set_body_transform(&mut self, handle: Self::BodyHandle, transform: Transform) {
        #[cfg(feature = "genesis")]
        {
            // For articulated bodies, we'd set the root pose
            // This is simplified - full implementation would handle all link transforms
            Python::with_gil(|py| {
                if let Some(entity) = self.entities.get(&handle.0) {
                    let entity = entity.bind(py);
                    let pos = PyList::new(py, &[
                        transform.position.x as f64,
                        transform.position.y as f64,
                        transform.position.z as f64,
                    ]).ok()?;
                    entity.call_method1("set_pos", (pos,)).ok()?;
                }
                Some(())
            });
        }
    }

    fn body_linear_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        #[cfg(feature = "genesis")]
        {
            if let Ok(qvel) = self.get_qvel(handle) {
                if qvel.len() >= 3 {
                    return Some(Vector3::new(
                        qvel[0] as f32,
                        qvel[1] as f32,
                        qvel[2] as f32,
                    ));
                }
            }
        }
        None
    }

    fn set_body_linear_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        #[cfg(feature = "genesis")]
        {
            if let Ok(mut qvel) = self.get_qvel(handle) {
                if qvel.len() >= 3 {
                    qvel[0] = velocity.x as f64;
                    qvel[1] = velocity.y as f64;
                    qvel[2] = velocity.z as f64;
                    let _ = self.set_qvel(handle, &qvel);
                }
            }
        }
    }

    fn body_angular_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        #[cfg(feature = "genesis")]
        {
            if let Ok(qvel) = self.get_qvel(handle) {
                if qvel.len() >= 6 {
                    return Some(Vector3::new(
                        qvel[3] as f32,
                        qvel[4] as f32,
                        qvel[5] as f32,
                    ));
                }
            }
        }
        None
    }

    fn set_body_angular_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        #[cfg(feature = "genesis")]
        {
            if let Ok(mut qvel) = self.get_qvel(handle) {
                if qvel.len() >= 6 {
                    qvel[3] = velocity.x as f64;
                    qvel[4] = velocity.y as f64;
                    qvel[5] = velocity.z as f64;
                    let _ = self.set_qvel(handle, &qvel);
                }
            }
        }
    }

    fn apply_force(&mut self, handle: Self::BodyHandle, force: Vector3<f32>) {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                if let Some(entity) = self.entities.get(&handle.0) {
                    let entity = entity.bind(py);
                    let force_list = PyList::new(py, &[
                        force.x as f64,
                        force.y as f64,
                        force.z as f64,
                    ]).ok()?;
                    entity.call_method1("set_external_force", (force_list,)).ok()?;
                }
                Some(())
            });
        }
    }

    fn apply_force_at_point(&mut self, handle: Self::BodyHandle, force: Vector3<f32>, _point: Point3<f32>) {
        // Genesis applies force at CoM by default
        self.apply_force(handle, force);
    }

    fn apply_impulse(&mut self, handle: Self::BodyHandle, impulse: Vector3<f32>) {
        // Convert impulse to velocity change
        if let Some(vel) = self.body_linear_velocity(handle) {
            let new_vel = vel + impulse; // Simplified - should divide by mass
            self.set_body_linear_velocity(handle, new_vel);
        }
    }

    fn apply_torque(&mut self, handle: Self::BodyHandle, torque: Vector3<f32>) {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                if let Some(entity) = self.entities.get(&handle.0) {
                    let entity = entity.bind(py);
                    let torque_list = PyList::new(py, &[
                        torque.x as f64,
                        torque.y as f64,
                        torque.z as f64,
                    ]).ok()?;
                    entity.call_method1("set_external_torque", (torque_list,)).ok()?;
                }
                Some(())
            });
        }
    }

    fn body_count(&self) -> usize {
        self.entity_info.len()
    }

    fn create_collider(&mut self, _body: Self::BodyHandle, _desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> {
        // Genesis handles colliders through entity morphs
        Err(BackendError::Unsupported("Genesis uses morph-based collision geometry".into()))
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
        // Genesis uses joints defined in URDF/MJCF
        Err(BackendError::Unsupported("Genesis uses URDF/MJCF-defined joints".into()))
    }

    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn ray_cast(&self, _ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> {
        // Would need to implement via Genesis raycast API
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
        // Would serialize all entity qpos/qvel
        let mut state: Vec<(u32, Vec<f64>, Vec<f64>)> = Vec::new();

        #[cfg(feature = "genesis")]
        {
            for (id, _) in &self.entity_info {
                let handle = GenesisBodyHandle(*id);
                let qpos = self.get_qpos(handle).unwrap_or_default();
                let qvel = self.get_qvel(handle).unwrap_or_default();
                state.push((*id, qpos, qvel));
            }
        }

        bincode::serialize(&state)
            .map_err(|e| BackendError::SerializationError(e.to_string()))
    }

    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), BackendError> {
        let state: Vec<(u32, Vec<f64>, Vec<f64>)> = bincode::deserialize(data)
            .map_err(|e| BackendError::DeserializationError(e.to_string()))?;

        #[cfg(feature = "genesis")]
        {
            for (id, qpos, qvel) in state {
                let handle = GenesisBodyHandle(id);
                let _ = self.set_qpos(handle, &qpos);
                let _ = self.set_qvel(handle, &qvel);
            }
        }

        Ok(())
    }

    fn reset(&mut self) {
        #[cfg(feature = "genesis")]
        {
            Python::with_gil(|py| {
                if let Some(ref scene) = self.scene {
                    let scene = scene.bind(py);
                    let _ = scene.call_method0("reset");
                }
            });
        }

        self.sim_time = 0.0;
        self.contacts.clear();
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
