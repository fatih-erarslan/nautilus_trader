//! SPH Configuration and Physics Constants
//!
//! Physics constants ported from Sibernetic's owPhysicsConstant.h with
//! documentation of their physical meaning and derivation.

use serde::{Deserialize, Serialize};
use std::f32::consts::PI;

/// Main SPH simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SphConfig {
    /// Physics constants
    pub physics: PhysicsConstants,
    /// Simulation box bounds
    pub bounds: SimulationBounds,
    /// Solver parameters
    pub solver: SolverConfig,
    /// GPU/CPU backend selection
    pub backend: BackendConfig,
}

impl Default for SphConfig {
    fn default() -> Self {
        Self {
            physics: PhysicsConstants::celegans(),
            bounds: SimulationBounds::default(),
            solver: SolverConfig::default(),
            backend: BackendConfig::default(),
        }
    }
}

impl SphConfig {
    /// Configuration for C. elegans worm simulation
    pub fn celegans() -> Self {
        Self {
            physics: PhysicsConstants::celegans(),
            bounds: SimulationBounds::worm_box(),
            solver: SolverConfig::default(),
            backend: BackendConfig::default(),
        }
    }

    /// Configuration for general fluid simulation
    pub fn fluid() -> Self {
        Self {
            physics: PhysicsConstants::water(),
            bounds: SimulationBounds::default(),
            solver: SolverConfig::default(),
            backend: BackendConfig::default(),
        }
    }
}

/// Physics constants for SPH simulation
///
/// These values are derived from physical properties and the SPH formulation.
/// The default values are tuned for C. elegans simulation at micrometer scale.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConstants {
    // ===========================================
    // Fundamental Constants
    // ===========================================

    /// Rest density of liquid (kg/m³)
    /// Standard water density = 1000 kg/m³
    pub rho0: f32,

    /// Mass per particle (kg)
    /// For C. elegans: derived from worm mass / particle count
    /// Adult worm mass ≈ 3.25e-09 kg, with ~1e5 particles → mass ≈ 3.25e-14 kg
    pub mass: f32,

    /// Simulation time step (seconds)
    /// Must satisfy CFL condition: dt ≤ λ_v * (h / v_max)
    /// Typical value: 5.0e-06 s for stable simulation
    pub time_step: f32,

    /// Simulation scale factor
    /// Converts between simulation units and meters
    /// N * simulation_scale = distance in meters
    pub simulation_scale: f32,

    // ===========================================
    // SPH Kernel Parameters
    // ===========================================

    /// Smoothing radius (dimensionless)
    /// Spatial distance over which properties are smoothed
    /// Typical value: h = 3.34
    pub h: f32,

    /// Hash grid cell size
    /// Typically 2 * h for efficient neighbor search
    pub hash_grid_cell_size: f32,

    /// Equilibrium distance between particles
    /// Typically r0 = 0.5 * h
    pub r0: f32,

    // ===========================================
    // Material Properties
    // ===========================================

    /// Dynamic viscosity coefficient
    /// Water at 25°C: 0.89e-3 Pa·s
    pub viscosity: f32,

    /// Surface tension coefficient
    pub surface_tension: f32,

    /// Elasticity coefficient for elastic connections
    pub elasticity: f32,

    /// Maximum muscle force (N)
    pub max_muscle_force: f32,

    // ===========================================
    // External Forces
    // ===========================================

    /// Gravity vector (m/s²)
    pub gravity: [f32; 3],

    // ===========================================
    // PCISPH Parameters
    // ===========================================

    /// Maximum iterations for pressure correction
    pub max_iterations: u32,

    /// Density error threshold for convergence
    pub density_error_threshold: f32,

    // ===========================================
    // Precomputed Coefficients
    // ===========================================

    /// Wpoly6 kernel coefficient (precomputed)
    #[serde(skip)]
    pub wpoly6_coeff: f64,

    /// Gradient of Wspiky kernel coefficient (precomputed)
    #[serde(skip)]
    pub grad_wspiky_coeff: f64,

    /// Laplacian of Wviscosity kernel coefficient (precomputed)
    #[serde(skip)]
    pub lap_wviscosity_coeff: f64,

    /// Beta coefficient for PCISPH
    #[serde(skip)]
    pub beta: f64,
}

impl PhysicsConstants {
    /// Constants tuned for C. elegans simulation (ow-0.9.8)
    ///
    /// Based on Sibernetic owPhysicsConstant.h:
    /// - Half-resolution 0.8mm worm simulation
    /// - Adult worm length: 1 mm
    /// - Adult worm diameter: 60-80 μm
    /// - Worm density: ~1000 kg/m³ (similar to water)
    ///
    /// Reference: OpenWorm Sibernetic ow-0.9.8 branch
    pub fn celegans() -> Self {
        // Mass per particle for half-resolution 0.8mm worm simulation
        // From owPhysicsConstant.h: mass = 20.00e-13f
        let mass = 20.00e-13_f32;

        // Time step: 4.0f * 5.0e-06f = 2.0e-05 seconds
        // Must satisfy CFL condition for numerical stability
        let time_step = 4.0 * 5.0e-06_f32;

        // Rest density (water)
        let rho0 = 1000.0_f32;

        // Simulation scale: conversion factor between simulation and real-world meters
        // From owPhysicsConstant.h: simulationScale = 0.0037f * pow(mass, 1/3) / pow(0.00025f, 1/3)
        let simulation_scale = 0.0037 * mass.powf(1.0 / 3.0) / 0.00025_f32.powf(1.0 / 3.0);

        // Smoothing radius (dimensionless)
        let h = 3.34_f32;

        let mut constants = Self {
            rho0,
            mass,
            time_step,
            simulation_scale,
            h,
            // Hash grid cell size: 2 * h for efficient neighbor search
            hash_grid_cell_size: 2.0 * h,
            // Equilibrium distance between boundary particles: r0 = 0.5 * h
            r0: 0.5 * h,
            // Viscosity: 0.1 * 0.00004 (from owPhysicsConstant.h)
            viscosity: 0.1 * 0.00004,
            surface_tension: 0.0, // Computed during precompute
            // Elasticity coefficient: 4 * 1.5e-04 / mass
            elasticity: 4.0 * 1.5e-04 / mass,
            // Maximum muscle force magnitude
            max_muscle_force: 4000.0,
            // Gravity vector (y-axis up)
            gravity: [0.0, -9.8, 0.0],
            // PCISPH iterations (from owPhysicsConstant.h: maxIteration = 3)
            max_iterations: 3,
            density_error_threshold: 0.01,
            wpoly6_coeff: 0.0,
            grad_wspiky_coeff: 0.0,
            lap_wviscosity_coeff: 0.0,
            beta: 0.0,
        };

        constants.precompute_coefficients();
        constants
    }

    /// Constants for standard water simulation
    pub fn water() -> Self {
        let mass = 0.00025_f32;
        let time_step = 0.001_f32;
        let rho0 = 1000.0_f32;

        let simulation_scale = 0.0037 * mass.powf(1.0 / 3.0) / 0.00025_f32.powf(1.0 / 3.0);
        let h = 3.34_f32;

        let mut constants = Self {
            rho0,
            mass,
            time_step,
            simulation_scale,
            h,
            hash_grid_cell_size: 2.0 * h,
            r0: 0.5 * h,
            viscosity: 0.001,
            surface_tension: 0.0728, // Water at 20°C
            elasticity: 0.0,
            max_muscle_force: 0.0,
            gravity: [0.0, -9.8, 0.0],
            max_iterations: 3,
            density_error_threshold: 0.01,
            wpoly6_coeff: 0.0,
            grad_wspiky_coeff: 0.0,
            lap_wviscosity_coeff: 0.0,
            beta: 0.0,
        };

        constants.precompute_coefficients();
        constants
    }

    /// Precompute kernel coefficients for efficiency
    pub fn precompute_coefficients(&mut self) {
        let h_scaled = (self.h * self.simulation_scale) as f64;

        // Wpoly6 coefficient: 315 / (64 * π * h^9)
        // Reference: Müller et al. (2003), Eq. 20
        self.wpoly6_coeff = 315.0 / (64.0 * PI as f64 * h_scaled.powi(9));

        // Gradient of Wspiky coefficient: -45 / (π * h^6)
        // Reference: Müller et al. (2003), Eq. 21
        self.grad_wspiky_coeff = -45.0 / (PI as f64 * h_scaled.powi(6));

        // Laplacian of Wviscosity coefficient (same magnitude as grad_wspiky)
        // Reference: Müller et al. (2003), Eq. 22
        self.lap_wviscosity_coeff = -self.grad_wspiky_coeff;

        // Beta for PCISPH: dt² * m² * 2 / (ρ0²)
        // Reference: Solenthaler's dissertation, Eq. 3.6
        self.beta = (self.time_step as f64).powi(2)
            * (self.mass as f64).powi(2)
            * 2.0
            / (self.rho0 as f64).powi(2);
    }

    /// Get scaled smoothing radius in meters
    #[inline]
    pub fn h_scaled(&self) -> f32 {
        self.h * self.simulation_scale
    }

    /// Get squared scaled smoothing radius
    #[inline]
    pub fn h_scaled_sq(&self) -> f32 {
        let hs = self.h_scaled();
        hs * hs
    }

    /// Get inverted hash grid cell size for fast division
    #[inline]
    pub fn hash_grid_cell_size_inv(&self) -> f32 {
        1.0 / self.hash_grid_cell_size
    }

    /// Get inverted simulation scale for fast division
    #[inline]
    pub fn simulation_scale_inv(&self) -> f32 {
        1.0 / self.simulation_scale
    }

    /// Mass times Wpoly6 coefficient (precomputed for efficiency)
    #[inline]
    pub fn mass_wpoly6(&self) -> f32 {
        (self.mass as f64 * self.wpoly6_coeff) as f32
    }

    /// Mass times grad_wspiky coefficient (precomputed for efficiency)
    #[inline]
    pub fn mass_grad_wspiky(&self) -> f32 {
        (self.mass as f64 * self.grad_wspiky_coeff) as f32
    }
}

impl Default for PhysicsConstants {
    fn default() -> Self {
        Self::celegans()
    }
}

/// Simulation box bounds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationBounds {
    /// Minimum x coordinate
    pub x_min: f32,
    /// Maximum x coordinate
    pub x_max: f32,
    /// Minimum y coordinate
    pub y_min: f32,
    /// Maximum y coordinate
    pub y_max: f32,
    /// Minimum z coordinate
    pub z_min: f32,
    /// Maximum z coordinate
    pub z_max: f32,
}

impl Default for SimulationBounds {
    fn default() -> Self {
        Self {
            x_min: -100.0,
            x_max: 100.0,
            y_min: -100.0,
            y_max: 100.0,
            z_min: -100.0,
            z_max: 100.0,
        }
    }
}

impl SimulationBounds {
    /// Bounds for C. elegans worm simulation box
    pub fn worm_box() -> Self {
        Self {
            x_min: 0.0,
            x_max: 306.0,
            y_min: 0.0,
            y_max: 306.0,
            z_min: 0.0,
            z_max: 906.0,
        }
    }

    /// Get dimensions of the box
    pub fn dimensions(&self) -> [f32; 3] {
        [
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        ]
    }

    /// Check if a point is inside the bounds
    #[inline]
    pub fn contains(&self, pos: [f32; 3]) -> bool {
        pos[0] >= self.x_min && pos[0] <= self.x_max
            && pos[1] >= self.y_min && pos[1] <= self.y_max
            && pos[2] >= self.z_min && pos[2] <= self.z_max
    }

    /// Clamp a position to be within bounds
    #[inline]
    pub fn clamp(&self, pos: [f32; 3]) -> [f32; 3] {
        [
            pos[0].clamp(self.x_min, self.x_max),
            pos[1].clamp(self.y_min, self.y_max),
            pos[2].clamp(self.z_min, self.z_max),
        ]
    }
}

/// Integration method for time stepping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntegrationMethod {
    /// Semi-implicit Euler (default, from ow-0.9.8)
    /// v_{n+1} = v_n + a_n * dt
    /// x_{n+1} = x_n + v_{n+1} * dt
    #[default]
    SemiImplicitEuler,

    /// Leapfrog (Störmer-Verlet)
    /// Better energy conservation for long simulations
    /// From Sibernetic master branch option
    ///
    /// v_{n+1/2} = v_{n-1/2} + a_n * dt
    /// x_{n+1} = x_n + v_{n+1/2} * dt
    Leapfrog,

    /// Velocity Verlet
    /// Combines benefits of both, used in molecular dynamics
    /// x_{n+1} = x_n + v_n * dt + 0.5 * a_n * dt^2
    /// v_{n+1} = v_n + 0.5 * (a_n + a_{n+1}) * dt
    VelocityVerlet,
}

/// Solver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SolverConfig {
    /// Maximum number of particles
    pub max_particles: usize,
    /// Maximum neighbors per particle
    pub max_neighbors: usize,
    /// Integration method for time stepping
    pub integration_method: IntegrationMethod,
    /// Enable surface tension
    pub surface_tension_enabled: bool,
    /// Enable membrane interactions
    pub membranes_enabled: bool,
    /// Enable crawling mode optimizations
    pub crawling_mode: bool,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            max_particles: 1_000_000,
            max_neighbors: 32,
            integration_method: IntegrationMethod::SemiImplicitEuler,
            surface_tension_enabled: true,
            membranes_enabled: true,
            crawling_mode: false,
        }
    }
}

impl SolverConfig {
    /// Configuration for swimming simulation
    pub fn swimming() -> Self {
        Self {
            max_particles: 1_000_000,
            max_neighbors: 32,
            integration_method: IntegrationMethod::SemiImplicitEuler,
            surface_tension_enabled: true,
            membranes_enabled: true,
            crawling_mode: false,
        }
    }

    /// Configuration for crawling simulation (from ow-0.9.8)
    /// Uses different parameters for substrate-based locomotion
    pub fn crawling() -> Self {
        Self {
            max_particles: 1_000_000,
            max_neighbors: 32,
            integration_method: IntegrationMethod::SemiImplicitEuler,
            surface_tension_enabled: true,
            membranes_enabled: true,
            crawling_mode: true,
        }
    }
}

/// Backend configuration for CPU/GPU selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendConfig {
    /// Backend type
    pub backend_type: BackendType,
    /// Work group size for GPU kernels
    pub work_group_size: usize,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            backend_type: BackendType::CpuSimd,
            work_group_size: 256,
        }
    }
}

/// Available compute backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BackendType {
    /// CPU with SIMD optimizations
    CpuSimd,
    /// CPU without SIMD (reference implementation)
    CpuScalar,
    /// wgpu (Vulkan/Metal/DX12/WebGPU)
    Wgpu,
    /// OpenCL 3.0
    OpenCL,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = SphConfig::default();
        assert!(config.physics.mass > 0.0);
        assert!(config.physics.time_step > 0.0);
        assert!(config.physics.h > 0.0);
    }

    #[test]
    fn test_precomputed_coefficients() {
        let physics = PhysicsConstants::celegans();
        assert!(physics.wpoly6_coeff > 0.0);
        assert!(physics.grad_wspiky_coeff < 0.0); // Negative for gradient
        assert!(physics.beta > 0.0);
    }

    #[test]
    fn test_bounds_contains() {
        let bounds = SimulationBounds::default();
        assert!(bounds.contains([0.0, 0.0, 0.0]));
        assert!(!bounds.contains([200.0, 0.0, 0.0]));
    }
}
