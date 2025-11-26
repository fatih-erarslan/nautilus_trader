//! Taichi GPU physics backend
//!
//! High-performance GPU compute for physics simulation.
//! Fork: https://github.com/fatih-erarslan/taichi
//!
//! ## Python Integration
//!
//! Taichi is a Python DSL for high-performance GPU computing. This backend
//! uses PyO3 to call Taichi's Python API from Rust.
//!
//! ### Requirements
//! - Python 3.8+ with Taichi installed: `pip install taichi`
//! - CUDA/Vulkan/Metal capable GPU
//!
//! ### Features
//! - GPU-accelerated particle simulation
//! - MPM (Material Point Method)
//! - SPH (Smoothed Particle Hydrodynamics)
//! - Differentiable simulation

use crate::backend::{BackendCapabilities, BackendError, BackendInfo, PhysicsBackend, SimulationStats};
use crate::body::BodyDesc;
use crate::collider::ColliderDesc;
use crate::constraint::ConstraintDesc;
use crate::query::{RayCast, RayHit, ShapeCast, ShapeHit};
use crate::{ContactManifold, PhysicsMaterial, Transform, AABB};
use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use std::any::Any;
use std::collections::HashMap;

#[cfg(feature = "taichi")]
use pyo3::prelude::*;
#[cfg(feature = "taichi")]
use pyo3::types::{PyDict, PyList, PyTuple};

/// Taichi architecture backend
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaichiArch {
    /// CUDA GPU backend
    Cuda,
    /// Vulkan GPU backend
    Vulkan,
    /// Metal GPU backend (macOS)
    Metal,
    /// OpenGL GPU backend
    OpenGL,
    /// CPU backend (fallback)
    Cpu,
}

impl Default for TaichiArch {
    fn default() -> Self {
        Self::Cuda
    }
}

impl TaichiArch {
    fn as_str(&self) -> &'static str {
        match self {
            TaichiArch::Cuda => "cuda",
            TaichiArch::Vulkan => "vulkan",
            TaichiArch::Metal => "metal",
            TaichiArch::OpenGL => "opengl",
            TaichiArch::Cpu => "cpu",
        }
    }
}

/// Simulation type for Taichi physics
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaichiSimulationType {
    /// Material Point Method for continuum mechanics
    MPM,
    /// Smoothed Particle Hydrodynamics for fluids
    SPH,
    /// Position-Based Dynamics for cloth/soft bodies
    PBD,
    /// Discrete Element Method for granular materials
    DEM,
    /// Custom kernel-based simulation
    Custom,
}

impl Default for TaichiSimulationType {
    fn default() -> Self {
        Self::MPM
    }
}

/// Configuration for Taichi backend
#[derive(Debug, Clone)]
pub struct TaichiConfig {
    /// GPU architecture to use
    pub arch: TaichiArch,
    /// Enable debug mode
    pub debug: bool,
    /// Simulation type
    pub simulation_type: TaichiSimulationType,
    /// Grid resolution for particle simulations
    pub grid_resolution: [u32; 3],
    /// Particle count for simulations
    pub max_particles: usize,
    /// Time step size
    pub dt: f32,
    /// Enable gradient computation for differentiable physics
    pub requires_grad: bool,
    /// Number of substeps per frame
    pub substeps: u32,
    /// Material properties (Young's modulus, Poisson ratio, etc.)
    pub material_params: TaichiMaterialParams,
}

/// Material parameters for Taichi simulations
#[derive(Debug, Clone)]
pub struct TaichiMaterialParams {
    /// Young's modulus (stiffness)
    pub youngs_modulus: f32,
    /// Poisson's ratio
    pub poisson_ratio: f32,
    /// Density
    pub density: f32,
    /// Friction coefficient
    pub friction: f32,
    /// Cohesion (for granular materials)
    pub cohesion: f32,
}

impl Default for TaichiMaterialParams {
    fn default() -> Self {
        Self {
            youngs_modulus: 1e5,
            poisson_ratio: 0.3,
            density: 1000.0,
            friction: 0.5,
            cohesion: 0.0,
        }
    }
}

impl Default for TaichiConfig {
    fn default() -> Self {
        Self {
            arch: TaichiArch::default(),
            debug: false,
            simulation_type: TaichiSimulationType::default(),
            grid_resolution: [64, 64, 64],
            max_particles: 100_000,
            dt: 1.0 / 60.0,
            requires_grad: false,
            substeps: 20,
            material_params: TaichiMaterialParams::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaichiBodyHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaichiColliderHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaichiConstraintHandle(pub u32);

/// Internal particle data
#[derive(Debug, Clone)]
struct ParticleData {
    positions: Vec<[f32; 3]>,
    velocities: Vec<[f32; 3]>,
    masses: Vec<f32>,
    volumes: Vec<f32>,
}

/// Body representation in Taichi
#[derive(Debug, Clone)]
struct TaichiBody {
    handle: TaichiBodyHandle,
    particle_start: usize,
    particle_count: usize,
    is_dynamic: bool,
    transform: Transform,
    linear_velocity: Vector3<f32>,
    angular_velocity: Vector3<f32>,
}

/// Taichi GPU physics backend
pub struct TaichiBackend {
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,
    config: TaichiConfig,

    // Python objects when feature is enabled
    #[cfg(feature = "taichi")]
    ti_module: Option<PyObject>,
    #[cfg(feature = "taichi")]
    position_field: Option<PyObject>,
    #[cfg(feature = "taichi")]
    velocity_field: Option<PyObject>,
    #[cfg(feature = "taichi")]
    grid_mass: Option<PyObject>,
    #[cfg(feature = "taichi")]
    grid_velocity: Option<PyObject>,
    #[cfg(feature = "taichi")]
    kernels: HashMap<String, PyObject>,

    // Body management
    bodies: HashMap<u32, TaichiBody>,
    next_body_id: u32,

    // Particle data (CPU mirror)
    particles: ParticleData,

    initialized: bool,
}

impl TaichiBackend {
    /// Initialize the Taichi Python environment
    #[cfg(feature = "taichi")]
    fn init_python(&mut self) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            // Import Taichi
            let ti = py.import("taichi")
                .map_err(|e| BackendError::InitializationFailed(
                    format!("Failed to import taichi: {}. Install with: pip install taichi", e)
                ))?;

            // Initialize Taichi with the specified architecture
            let init_kwargs = PyDict::new(py);
            let arch = ti.getattr(self.config.arch.as_str())
                .map_err(|e| BackendError::InitializationFailed(
                    format!("Unsupported architecture {}: {}", self.config.arch.as_str(), e)
                ))?;
            init_kwargs.set_item("arch", arch)?;
            init_kwargs.set_item("debug", self.config.debug)?;

            if self.config.requires_grad {
                init_kwargs.set_item("default_fp", ti.getattr("f32")?)?;
            }

            ti.call_method("init", (), Some(init_kwargs))
                .map_err(|e| BackendError::InitializationFailed(format!("ti.init failed: {}", e)))?;

            self.ti_module = Some(ti.into());

            // Create particle fields
            self.create_particle_fields(py)?;

            // Compile kernels based on simulation type
            self.compile_kernels(py)?;

            self.initialized = true;
            Ok(())
        })
    }

    #[cfg(feature = "taichi")]
    fn create_particle_fields(&mut self, py: Python<'_>) -> Result<(), BackendError> {
        let ti = self.ti_module.as_ref().unwrap().bind(py);
        let n_particles = self.config.max_particles;
        let grid_res = self.config.grid_resolution;

        // Create particle position field: ti.Vector.field(3, dtype=float, shape=n_particles)
        let vec3_type = ti.getattr("types")?.getattr("vector")?.call1((3, ti.getattr("f32")?))?;

        let field_kwargs = PyDict::new(py);
        field_kwargs.set_item("shape", n_particles)?;
        if self.config.requires_grad {
            field_kwargs.set_item("needs_grad", true)?;
        }

        let position_field = vec3_type.call_method("field", (), Some(field_kwargs))?;
        self.position_field = Some(position_field.into());

        // Create velocity field
        let field_kwargs = PyDict::new(py);
        field_kwargs.set_item("shape", n_particles)?;
        let velocity_field = vec3_type.call_method("field", (), Some(field_kwargs))?;
        self.velocity_field = Some(velocity_field.into());

        // Create grid fields for MPM/SPH
        let grid_shape = PyTuple::new(py, &[grid_res[0], grid_res[1], grid_res[2]])?;

        let field_kwargs = PyDict::new(py);
        field_kwargs.set_item("shape", &grid_shape)?;
        let grid_mass = ti.getattr("field")?.call((ti.getattr("f32")?,), Some(field_kwargs))?;
        self.grid_mass = Some(grid_mass.into());

        let field_kwargs = PyDict::new(py);
        field_kwargs.set_item("shape", &grid_shape)?;
        let grid_velocity = vec3_type.call_method("field", (), Some(field_kwargs))?;
        self.grid_velocity = Some(grid_velocity.into());

        Ok(())
    }

    #[cfg(feature = "taichi")]
    fn compile_kernels(&mut self, py: Python<'_>) -> Result<(), BackendError> {
        let ti = self.ti_module.as_ref().unwrap().bind(py);

        // Create and compile MPM kernels using Python exec
        let kernel_code = match self.config.simulation_type {
            TaichiSimulationType::MPM => self.get_mpm_kernel_code(),
            TaichiSimulationType::SPH => self.get_sph_kernel_code(),
            TaichiSimulationType::PBD => self.get_pbd_kernel_code(),
            TaichiSimulationType::DEM => self.get_dem_kernel_code(),
            TaichiSimulationType::Custom => String::new(),
        };

        if !kernel_code.is_empty() {
            // Execute kernel code to define kernels
            let globals = PyDict::new(py);
            globals.set_item("ti", ti)?;
            globals.set_item("x", self.position_field.as_ref().unwrap().bind(py))?;
            globals.set_item("v", self.velocity_field.as_ref().unwrap().bind(py))?;
            globals.set_item("grid_m", self.grid_mass.as_ref().unwrap().bind(py))?;
            globals.set_item("grid_v", self.grid_velocity.as_ref().unwrap().bind(py))?;
            globals.set_item("n_particles", self.config.max_particles)?;
            globals.set_item("n_grid", self.config.grid_resolution[0])?;
            globals.set_item("dx", 1.0 / self.config.grid_resolution[0] as f32)?;
            globals.set_item("dt", self.config.dt / self.config.substeps as f32)?;
            globals.set_item("gravity", self.gravity.y)?;
            globals.set_item("E", self.config.material_params.youngs_modulus)?;
            globals.set_item("nu", self.config.material_params.poisson_ratio)?;

            py.run(&kernel_code, Some(globals), None)
                .map_err(|e| BackendError::InitializationFailed(format!("Failed to compile kernels: {}", e)))?;

            // Store kernel references
            if let Ok(substep) = globals.get_item("substep") {
                if let Some(substep) = substep {
                    self.kernels.insert("substep".to_string(), substep.into());
                }
            }
            if let Ok(reset_grid) = globals.get_item("reset_grid") {
                if let Some(reset_grid) = reset_grid {
                    self.kernels.insert("reset_grid".to_string(), reset_grid.into());
                }
            }
        }

        Ok(())
    }

    fn get_mpm_kernel_code(&self) -> String {
        r#"
@ti.kernel
def reset_grid():
    for i, j, k in grid_m:
        grid_m[i, j, k] = 0.0
        grid_v[i, j, k] = [0.0, 0.0, 0.0]

@ti.kernel
def p2g():
    for p in range(n_particles):
        base = (x[p] * n_grid - 0.5).cast(int)
        fx = x[p] * n_grid - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        stress = -dt * 4 * E * (1.0 / n_grid ** 2)
        affine = ti.Matrix([[stress, 0, 0], [0, stress, 0], [0, 0, stress]])

        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            dpos = (offset.cast(float) - fx) * dx
            weight = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offset] += weight * (v[p] + affine @ dpos)
            grid_m[base + offset] += weight

@ti.kernel
def grid_op():
    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0:
            grid_v[i, j, k] /= grid_m[i, j, k]
            grid_v[i, j, k][1] += dt * gravity

            # Boundary conditions
            if i < 3 or i > n_grid - 3:
                grid_v[i, j, k][0] = 0
            if j < 3 or j > n_grid - 3:
                grid_v[i, j, k][1] = 0
            if k < 3 or k > n_grid - 3:
                grid_v[i, j, k][2] = 0

@ti.kernel
def g2p():
    for p in range(n_particles):
        base = (x[p] * n_grid - 0.5).cast(int)
        fx = x[p] * n_grid - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

        new_v = ti.Vector.zero(float, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i, j, k])
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * grid_v[base + offset]

        v[p] = new_v
        x[p] += dt * v[p]

def substep():
    reset_grid()
    p2g()
    grid_op()
    g2p()
"#.to_string()
    }

    fn get_sph_kernel_code(&self) -> String {
        r#"
h = dx * 2.0  # Smoothing length
rho_0 = 1000.0  # Reference density
k = 100.0  # Pressure stiffness
mu = 0.1  # Viscosity

@ti.func
def W(r, h):
    q = ti.sqrt(r.dot(r)) / h
    result = 0.0
    if q < 1.0:
        result = 315.0 / (64.0 * 3.14159 * h ** 9) * (h ** 2 - r.dot(r)) ** 3
    return result

@ti.func
def gradW(r, h):
    q = ti.sqrt(r.dot(r)) / h
    result = ti.Vector.zero(float, 3)
    if q < 1.0 and q > 1e-6:
        result = -45.0 / (3.14159 * h ** 6) * (h - ti.sqrt(r.dot(r))) ** 2 * r.normalized()
    return result

@ti.kernel
def compute_density():
    for p in range(n_particles):
        rho = 0.0
        for q in range(n_particles):
            r = x[p] - x[q]
            rho += W(r, h)
        grid_m[0, 0, p % n_grid] = rho  # Store density in grid_m

@ti.kernel
def compute_forces():
    for p in range(n_particles):
        rho_p = grid_m[0, 0, p % n_grid]
        pressure_p = k * (rho_p - rho_0)
        f = ti.Vector.zero(float, 3)

        for q in range(n_particles):
            if p != q:
                r = x[p] - x[q]
                rho_q = grid_m[0, 0, q % n_grid]
                pressure_q = k * (rho_q - rho_0)

                # Pressure force
                f -= (pressure_p + pressure_q) / (2 * rho_q) * gradW(r, h)

                # Viscosity
                f += mu * (v[q] - v[p]) / rho_q * W(r, h)

        v[p] += dt * (f + ti.Vector([0.0, gravity, 0.0]))

@ti.kernel
def integrate():
    for p in range(n_particles):
        x[p] += dt * v[p]
        # Boundary conditions
        for d in ti.static(range(3)):
            if x[p][d] < 0.05:
                x[p][d] = 0.05
                v[p][d] = 0
            if x[p][d] > 0.95:
                x[p][d] = 0.95
                v[p][d] = 0

def substep():
    compute_density()
    compute_forces()
    integrate()
"#.to_string()
    }

    fn get_pbd_kernel_code(&self) -> String {
        r#"
compliance = 1e-6  # Inverse stiffness

@ti.kernel
def predict_positions():
    for p in range(n_particles):
        v[p][1] += dt * gravity
        grid_v[0, 0, p % n_grid] = x[p] + dt * v[p]  # Store predicted position

@ti.kernel
def solve_constraints():
    # Distance constraints (simplified)
    for p in range(n_particles - 1):
        p1 = grid_v[0, 0, p % n_grid]
        p2 = grid_v[0, 0, (p + 1) % n_grid]
        diff = p1 - p2
        dist = ti.sqrt(diff.dot(diff))
        rest_length = dx
        if dist > 1e-6:
            correction = (dist - rest_length) / dist * diff * 0.5
            grid_v[0, 0, p % n_grid] -= correction
            grid_v[0, 0, (p + 1) % n_grid] += correction

@ti.kernel
def update_velocities():
    for p in range(n_particles):
        v[p] = (grid_v[0, 0, p % n_grid] - x[p]) / dt
        x[p] = grid_v[0, 0, p % n_grid]
        # Boundary
        for d in ti.static(range(3)):
            if x[p][d] < 0.05:
                x[p][d] = 0.05
                v[p][d] = 0
            if x[p][d] > 0.95:
                x[p][d] = 0.95
                v[p][d] = 0

def substep():
    predict_positions()
    for _ in range(10):  # Constraint iterations
        solve_constraints()
    update_velocities()
"#.to_string()
    }

    fn get_dem_kernel_code(&self) -> String {
        r#"
particle_radius = dx * 0.5
k_n = E  # Normal stiffness
k_t = E * 0.5  # Tangential stiffness
damping = 0.1

@ti.kernel
def compute_contacts():
    for p in range(n_particles):
        f = ti.Vector.zero(float, 3)
        for q in range(n_particles):
            if p != q:
                r = x[p] - x[q]
                dist = ti.sqrt(r.dot(r))
                overlap = 2 * particle_radius - dist
                if overlap > 0 and dist > 1e-6:
                    n = r / dist
                    # Normal force (Hertz contact)
                    f_n = k_n * overlap ** 1.5 * n
                    # Damping
                    v_rel = v[p] - v[q]
                    f_d = -damping * v_rel.dot(n) * n
                    f += f_n + f_d

        # Gravity
        f[1] += gravity
        v[p] += dt * f

@ti.kernel
def integrate():
    for p in range(n_particles):
        x[p] += dt * v[p]
        # Boundary conditions
        for d in ti.static(range(3)):
            if x[p][d] < particle_radius:
                x[p][d] = particle_radius
                v[p][d] *= -0.5  # Restitution
            if x[p][d] > 1.0 - particle_radius:
                x[p][d] = 1.0 - particle_radius
                v[p][d] *= -0.5

def substep():
    compute_contacts()
    integrate()
"#.to_string()
    }

    /// Step the simulation using Python
    #[cfg(feature = "taichi")]
    fn step_python(&mut self, dt: f32) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            // Run substeps
            let substeps = (dt / (self.config.dt / self.config.substeps as f32)).round() as u32;
            let substeps = substeps.max(1);

            if let Some(substep_kernel) = self.kernels.get("substep") {
                for _ in 0..substeps {
                    substep_kernel.call0(py)
                        .map_err(|e| BackendError::FfiError(format!("substep failed: {}", e)))?;
                }
            }

            // Sync particle data back to CPU
            self.sync_from_gpu(py)?;

            Ok(())
        })
    }

    #[cfg(feature = "taichi")]
    fn sync_from_gpu(&mut self, py: Python<'_>) -> Result<(), BackendError> {
        if let Some(ref pos_field) = self.position_field {
            let pos_field = pos_field.bind(py);
            let numpy = py.import("numpy")?;

            // Convert to numpy array and then to Rust Vec
            let pos_array = pos_field.call_method0("to_numpy")?;
            let pos_list: Vec<Vec<f32>> = pos_array.extract()?;

            self.particles.positions = pos_list.into_iter()
                .map(|v| [v.get(0).copied().unwrap_or(0.0),
                          v.get(1).copied().unwrap_or(0.0),
                          v.get(2).copied().unwrap_or(0.0)])
                .collect();
        }

        if let Some(ref vel_field) = self.velocity_field {
            let vel_field = vel_field.bind(py);
            let vel_array = vel_field.call_method0("to_numpy")?;
            let vel_list: Vec<Vec<f32>> = vel_array.extract()?;

            self.particles.velocities = vel_list.into_iter()
                .map(|v| [v.get(0).copied().unwrap_or(0.0),
                          v.get(1).copied().unwrap_or(0.0),
                          v.get(2).copied().unwrap_or(0.0)])
                .collect();
        }

        Ok(())
    }

    #[cfg(feature = "taichi")]
    fn sync_to_gpu(&self, py: Python<'_>) -> Result<(), BackendError> {
        if let Some(ref pos_field) = self.position_field {
            let pos_field = pos_field.bind(py);
            let numpy = py.import("numpy")?;

            // Convert positions to numpy array
            let pos_list: Vec<[f32; 3]> = self.particles.positions.clone();
            let pos_array = numpy.call_method1("array", (pos_list,))?;
            pos_field.call_method1("from_numpy", (pos_array,))?;
        }

        if let Some(ref vel_field) = self.velocity_field {
            let vel_field = vel_field.bind(py);
            let numpy = py.import("numpy")?;

            let vel_list: Vec<[f32; 3]> = self.particles.velocities.clone();
            let vel_array = numpy.call_method1("array", (vel_list,))?;
            vel_field.call_method1("from_numpy", (vel_array,))?;
        }

        Ok(())
    }

    /// Add particles for a body
    fn add_particles_for_body(&mut self, desc: &BodyDesc) -> (usize, usize) {
        let start = self.particles.positions.len();

        // Generate particles based on body shape (simplified - creates a cube of particles)
        let particle_spacing = 0.02; // 2cm spacing
        let half_extent = 0.1; // 10cm half-extent

        let mut count = 0;
        let mut x = -half_extent;
        while x <= half_extent {
            let mut y = -half_extent;
            while y <= half_extent {
                let mut z = -half_extent;
                while z <= half_extent {
                    let pos = [
                        desc.transform.position.x + x,
                        desc.transform.position.y + y,
                        desc.transform.position.z + z,
                    ];
                    self.particles.positions.push(pos);

                    let vel = [
                        desc.linear_velocity.x,
                        desc.linear_velocity.y,
                        desc.linear_velocity.z,
                    ];
                    self.particles.velocities.push(vel);

                    self.particles.masses.push(self.config.material_params.density * particle_spacing.powi(3));
                    self.particles.volumes.push(particle_spacing.powi(3));

                    count += 1;
                    z += particle_spacing;
                }
                y += particle_spacing;
            }
            x += particle_spacing;
        }

        (start, count)
    }

    /// Get particle positions for a body
    pub fn get_particle_positions(&self, handle: TaichiBodyHandle) -> Option<Vec<[f32; 3]>> {
        self.bodies.get(&handle.0).map(|body| {
            self.particles.positions[body.particle_start..body.particle_start + body.particle_count].to_vec()
        })
    }

    /// Get particle velocities for a body
    pub fn get_particle_velocities(&self, handle: TaichiBodyHandle) -> Option<Vec<[f32; 3]>> {
        self.bodies.get(&handle.0).map(|body| {
            self.particles.velocities[body.particle_start..body.particle_start + body.particle_count].to_vec()
        })
    }

    /// Get all particle positions (for visualization)
    pub fn all_particle_positions(&self) -> &[[f32; 3]] {
        &self.particles.positions
    }

    /// Get all particle velocities
    pub fn all_particle_velocities(&self) -> &[[f32; 3]] {
        &self.particles.velocities
    }

    /// Compute gradients for differentiable physics
    #[cfg(feature = "taichi")]
    pub fn compute_gradients(&self) -> Result<Vec<f32>, BackendError> {
        if !self.config.requires_grad {
            return Err(BackendError::Unsupported("Gradients not enabled in config".into()));
        }

        Python::with_gil(|py| {
            if let Some(ref pos_field) = self.position_field {
                let pos_field = pos_field.bind(py);
                let grad = pos_field.getattr("grad")?;
                let grad_array = grad.call_method0("to_numpy")?;
                let grads: Vec<Vec<f32>> = grad_array.extract()?;
                Ok(grads.into_iter().flatten().collect())
            } else {
                Err(BackendError::Unsupported("Position field not initialized".into()))
            }
        })
    }
}

impl PhysicsBackend for TaichiBackend {
    type Config = TaichiConfig;
    type BodyHandle = TaichiBodyHandle;
    type ColliderHandle = TaichiColliderHandle;
    type ConstraintHandle = TaichiConstraintHandle;

    fn new(config: Self::Config) -> Result<Self, BackendError> {
        let mut backend = Self {
            gravity: Vector3::new(0.0, -9.81, 0.0),
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            config,
            #[cfg(feature = "taichi")]
            ti_module: None,
            #[cfg(feature = "taichi")]
            position_field: None,
            #[cfg(feature = "taichi")]
            velocity_field: None,
            #[cfg(feature = "taichi")]
            grid_mass: None,
            #[cfg(feature = "taichi")]
            grid_velocity: None,
            #[cfg(feature = "taichi")]
            kernels: HashMap::new(),
            bodies: HashMap::new(),
            next_body_id: 0,
            particles: ParticleData {
                positions: Vec::new(),
                velocities: Vec::new(),
                masses: Vec::new(),
                volumes: Vec::new(),
            },
            initialized: false,
        };

        #[cfg(feature = "taichi")]
        {
            backend.init_python()?;
        }

        Ok(backend)
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "Taichi",
            version: "1.7",
            description: "GPU-accelerated particle physics (MPM/SPH/PBD/DEM)",
            gpu_accelerated: true,
            differentiable: self.config.requires_grad,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            physics_3d: true,
            physics_2d: true,
            soft_bodies: true,
            cloth: matches!(self.config.simulation_type, TaichiSimulationType::PBD),
            fluids: matches!(self.config.simulation_type, TaichiSimulationType::SPH | TaichiSimulationType::MPM),
            articulated: false,
            ccd: false,
            deterministic: true,
            parallel: true,
            gpu: true,
            differentiable: self.config.requires_grad,
            max_bodies: self.config.max_particles,
        }
    }

    fn step(&mut self, dt: f32) {
        let start = std::time::Instant::now();

        #[cfg(feature = "taichi")]
        {
            if self.initialized {
                let _ = self.step_python(dt);
            }
        }

        // Update body transforms from particle data
        for body in self.bodies.values_mut() {
            if body.is_dynamic && body.particle_count > 0 {
                // Compute center of mass from particles
                let mut com = Vector3::zeros();
                for i in body.particle_start..body.particle_start + body.particle_count {
                    if let Some(pos) = self.particles.positions.get(i) {
                        com += Vector3::new(pos[0], pos[1], pos[2]);
                    }
                }
                com /= body.particle_count as f32;
                body.transform.position = Point3::from(com);

                // Compute average velocity
                let mut avg_vel = Vector3::zeros();
                for i in body.particle_start..body.particle_start + body.particle_count {
                    if let Some(vel) = self.particles.velocities.get(i) {
                        avg_vel += Vector3::new(vel[0], vel[1], vel[2]);
                    }
                }
                body.linear_velocity = avg_vel / body.particle_count as f32;
            }
        }

        self.stats.total_us = start.elapsed().as_micros() as u64;
        self.stats.active_bodies = self.bodies.len() as u32;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.gravity = gravity;
        // Update gravity in Python kernels if needed
        #[cfg(feature = "taichi")]
        {
            // Kernels would need to be recompiled with new gravity
        }
    }

    fn gravity(&self) -> Vector3<f32> {
        self.gravity
    }

    fn create_body(&mut self, desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        let handle = TaichiBodyHandle(self.next_body_id);
        self.next_body_id += 1;

        let (particle_start, particle_count) = self.add_particles_for_body(desc);

        let body = TaichiBody {
            handle,
            particle_start,
            particle_count,
            is_dynamic: desc.body_type == crate::body::BodyType::Dynamic,
            transform: desc.transform,
            linear_velocity: desc.linear_velocity,
            angular_velocity: desc.angular_velocity,
        };

        self.bodies.insert(handle.0, body);

        // Sync to GPU
        #[cfg(feature = "taichi")]
        {
            Python::with_gil(|py| {
                let _ = self.sync_to_gpu(py);
            });
        }

        Ok(handle)
    }

    fn remove_body(&mut self, handle: Self::BodyHandle) -> Result<(), BackendError> {
        self.bodies.remove(&handle.0);
        Ok(())
    }

    fn body_transform(&self, handle: Self::BodyHandle) -> Option<Transform> {
        self.bodies.get(&handle.0).map(|b| b.transform)
    }

    fn set_body_transform(&mut self, handle: Self::BodyHandle, transform: Transform) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            let delta = transform.position - body.transform.position;
            body.transform = transform;

            // Move all particles
            for i in body.particle_start..body.particle_start + body.particle_count {
                if let Some(pos) = self.particles.positions.get_mut(i) {
                    pos[0] += delta.x;
                    pos[1] += delta.y;
                    pos[2] += delta.z;
                }
            }
        }
    }

    fn body_linear_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.bodies.get(&handle.0).map(|b| b.linear_velocity)
    }

    fn set_body_linear_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            body.linear_velocity = velocity;
            // Set all particle velocities
            for i in body.particle_start..body.particle_start + body.particle_count {
                if let Some(vel) = self.particles.velocities.get_mut(i) {
                    vel[0] = velocity.x;
                    vel[1] = velocity.y;
                    vel[2] = velocity.z;
                }
            }
        }
    }

    fn body_angular_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.bodies.get(&handle.0).map(|b| b.angular_velocity)
    }

    fn set_body_angular_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(body) = self.bodies.get_mut(&handle.0) {
            body.angular_velocity = velocity;
        }
    }

    fn apply_force(&mut self, handle: Self::BodyHandle, force: Vector3<f32>) {
        if let Some(body) = self.bodies.get(&handle.0) {
            if body.is_dynamic {
                // Distribute force across all particles
                let force_per_particle = force / body.particle_count as f32;
                let dt = self.config.dt;
                for i in body.particle_start..body.particle_start + body.particle_count {
                    if let (Some(vel), Some(mass)) = (
                        self.particles.velocities.get_mut(i),
                        self.particles.masses.get(i),
                    ) {
                        let accel = force_per_particle / *mass;
                        vel[0] += accel.x * dt;
                        vel[1] += accel.y * dt;
                        vel[2] += accel.z * dt;
                    }
                }
            }
        }
    }

    fn apply_force_at_point(&mut self, handle: Self::BodyHandle, force: Vector3<f32>, _point: Point3<f32>) {
        // For particle systems, apply force to nearest particle
        self.apply_force(handle, force);
    }

    fn apply_impulse(&mut self, handle: Self::BodyHandle, impulse: Vector3<f32>) {
        if let Some(body) = self.bodies.get(&handle.0) {
            if body.is_dynamic {
                let impulse_per_particle = impulse / body.particle_count as f32;
                for i in body.particle_start..body.particle_start + body.particle_count {
                    if let (Some(vel), Some(mass)) = (
                        self.particles.velocities.get_mut(i),
                        self.particles.masses.get(i),
                    ) {
                        vel[0] += impulse_per_particle.x / *mass;
                        vel[1] += impulse_per_particle.y / *mass;
                        vel[2] += impulse_per_particle.z / *mass;
                    }
                }
            }
        }
    }

    fn apply_torque(&mut self, _handle: Self::BodyHandle, _torque: Vector3<f32>) {
        // Torque is complex for particle systems - would need angular momentum tracking
    }

    fn body_count(&self) -> usize {
        self.bodies.len()
    }

    fn create_collider(&mut self, _body: Self::BodyHandle, _desc: &ColliderDesc) -> Result<Self::ColliderHandle, BackendError> {
        // Particle systems use implicit collision through contact detection
        Err(BackendError::Unsupported("Taichi uses implicit particle collision".into()))
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
        Err(BackendError::Unsupported("Constraints not supported in particle physics".into()))
    }

    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn ray_cast(&self, _ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> {
        // Would need spatial hashing for efficient ray-particle intersection
        None
    }

    fn ray_cast_all(&self, _ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> {
        Vec::new()
    }

    fn shape_cast(&self, _cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> {
        None
    }

    fn query_aabb(&self, aabb: &AABB) -> Vec<Self::BodyHandle> {
        // Return bodies whose particles intersect the AABB
        let mut result = Vec::new();
        for (id, body) in &self.bodies {
            for i in body.particle_start..body.particle_start + body.particle_count {
                if let Some(pos) = self.particles.positions.get(i) {
                    let p = Point3::new(pos[0], pos[1], pos[2]);
                    if p.x >= aabb.min.x && p.x <= aabb.max.x
                        && p.y >= aabb.min.y && p.y <= aabb.max.y
                        && p.z >= aabb.min.z && p.z <= aabb.max.z
                    {
                        result.push(TaichiBodyHandle(*id));
                        break;
                    }
                }
            }
        }
        result
    }

    fn contacts(&self) -> &[ContactManifold] {
        &self.contacts
    }

    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> {
        let state = (&self.particles.positions, &self.particles.velocities);
        bincode::serialize(&state)
            .map_err(|e| BackendError::SerializationError(e.to_string()))
    }

    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), BackendError> {
        let (positions, velocities): (Vec<[f32; 3]>, Vec<[f32; 3]>) = bincode::deserialize(data)
            .map_err(|e| BackendError::DeserializationError(e.to_string()))?;

        self.particles.positions = positions;
        self.particles.velocities = velocities;

        #[cfg(feature = "taichi")]
        {
            Python::with_gil(|py| {
                self.sync_to_gpu(py)
            })?;
        }

        Ok(())
    }

    fn reset(&mut self) {
        self.particles.positions.clear();
        self.particles.velocities.clear();
        self.particles.masses.clear();
        self.particles.volumes.clear();
        self.bodies.clear();
        self.next_body_id = 0;
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
