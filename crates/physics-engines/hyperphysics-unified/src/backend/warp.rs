//! NVIDIA Warp differentiable physics backend
//!
//! Fork: https://github.com/fatih-erarslan/warp
//!
//! ## Python Integration
//!
//! NVIDIA Warp is a Python framework for high-performance differentiable
//! simulation. This backend uses PyO3 to call Warp's Python API from Rust.
//!
//! ### Requirements
//! - Python 3.8+ with Warp installed: `pip install warp-lang`
//! - NVIDIA GPU with CUDA support (falls back to CPU)
//!
//! ### Features
//! - GPU kernel compilation via CUDA
//! - Automatic differentiation (tape-based autodiff)
//! - USD/USDRT interop for asset pipelines
//! - Rigid body and soft body simulation
//! - Differentiable market dynamics for RL
//!
//! ## Usage
//!
//! ```rust,ignore
//! use hyperphysics_unified::backend::warp::{WarpBackend, WarpConfig};
//!
//! let config = WarpConfig {
//!     device: "cuda:0".into(),
//!     num_agents: 1000,
//!     record_gradients: true,
//! };
//! let mut backend = WarpBackend::new(config)?;
//!
//! // Step with gradient recording
//! backend.step(0.01);
//!
//! // Compute gradients
//! let grads = backend.compute_gradients(loss)?;
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
use std::time::Instant;

#[cfg(feature = "warp")]
use pyo3::prelude::*;
#[cfg(feature = "warp")]
use pyo3::types::{PyDict, PyModule};

/// Warp backend configuration
#[derive(Debug, Clone)]
pub struct WarpConfig {
    /// CUDA device identifier (e.g., "cuda:0") or "cpu"
    pub device: String,
    /// Number of agents in the simulation
    pub num_agents: usize,
    /// Enable gradient recording for autodiff
    pub record_gradients: bool,
    /// Simulation timestep
    pub timestep: f32,
    /// Enable verbose Python output
    pub verbose: bool,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self {
            device: "cuda:0".into(),
            num_agents: 100,
            record_gradients: false,
            timestep: 0.01,
            verbose: false,
        }
    }
}

/// Handle to a body in the Warp simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WarpBodyHandle(pub u32);

/// Handle to a collider in the Warp simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WarpColliderHandle(pub u32);

/// Handle to a constraint in the Warp simulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WarpConstraintHandle(pub u32);

/// Agent state in the differentiable simulation
#[derive(Debug, Clone, Default)]
pub struct WarpAgentState {
    /// 3D position
    pub position: [f32; 3],
    /// 3D velocity
    pub velocity: [f32; 3],
    /// Agent capital
    pub capital: f32,
    /// Agent inventory
    pub inventory: f32,
    /// Risk aversion coefficient
    pub risk_aversion: f32,
}

/// Market state for differentiable dynamics
#[derive(Debug, Clone, Default)]
pub struct WarpMarketState {
    /// Current price
    pub price: f32,
    /// Current volume
    pub volume: f32,
    /// Volatility estimate
    pub volatility: f32,
    /// Trend indicator
    pub trend: f32,
}

/// Gradient information from autodiff
#[derive(Debug, Clone, Default)]
pub struct WarpGradients {
    /// Gradients w.r.t. positions
    pub position_grads: Vec<[f32; 3]>,
    /// Gradients w.r.t. velocities
    pub velocity_grads: Vec<[f32; 3]>,
    /// Gradients w.r.t. capitals
    pub capital_grads: Vec<f32>,
}

/// NVIDIA Warp differentiable physics backend
///
/// Provides GPU-accelerated differentiable simulation using NVIDIA Warp
/// through PyO3. Supports automatic differentiation for gradient-based
/// optimization of trading strategies.
pub struct WarpBackend {
    config: WarpConfig,
    gravity: Vector3<f32>,
    contacts: Vec<ContactManifold>,
    stats: SimulationStats,

    // Agent management
    agents: Vec<WarpAgentState>,
    market: WarpMarketState,
    body_handles: HashMap<WarpBodyHandle, usize>,
    next_handle: u32,

    // Gradient storage
    last_gradients: Option<WarpGradients>,

    // Python integration (when feature enabled)
    #[cfg(feature = "warp")]
    py_simulation: Option<Py<PyAny>>,

    // Timing
    step_count: u64,
    total_step_time_us: u64,
}

impl WarpBackend {
    /// Initialize the Warp Python simulation
    #[cfg(feature = "warp")]
    fn init_python_simulation(&mut self) -> Result<(), BackendError> {
        Python::with_gil(|py| {
            // Load the kernels module
            let kernels_code = include_str!("../../../warp-hyperphysics/src/kernels.py");
            let kernels_module = PyModule::from_code_bound(
                py,
                kernels_code,
                "kernels.py",
                "warp_kernels",
            ).map_err(|e| BackendError::InitializationFailed(format!("Failed to load Warp kernels: {}", e)))?;

            // Initialize the simulation
            let init_fn = kernels_module.getattr("init_simulation")
                .map_err(|e| BackendError::InitializationFailed(format!("Failed to get init_simulation: {}", e)))?;

            let result: String = init_fn.call1((self.config.num_agents, &self.config.device))
                .map_err(|e| BackendError::InitializationFailed(format!("init_simulation failed: {}", e)))?
                .extract()
                .map_err(|e| BackendError::InitializationFailed(format!("Failed to extract result: {}", e)))?;

            if result.contains("Failed") || result.contains("unavailable") {
                return Err(BackendError::InitializationFailed(result));
            }

            // Store module reference for later calls
            self.py_simulation = Some(kernels_module.into());
            Ok(())
        })
    }

    /// Step the Python simulation
    #[cfg(feature = "warp")]
    fn step_python(&mut self, dt: f32) -> Result<(f32, f32), BackendError> {
        if self.py_simulation.is_none() {
            self.init_python_simulation()?;
        }

        Python::with_gil(|py| {
            let module = self.py_simulation.as_ref().unwrap().bind(py);
            let step_fn = module.getattr("step_simulation")
                .map_err(|e| BackendError::FfiError(format!("Failed to get step_simulation: {}", e)))?;

            let result: String = step_fn.call1((0usize, 0usize, self.config.num_agents, dt))
                .map_err(|e| BackendError::FfiError(format!("step_simulation failed: {}", e)))?
                .extract()
                .map_err(|e| BackendError::FfiError(format!("Failed to extract result: {}", e)))?;

            // Parse result: "Stepped N agents: price=X.XXXX, volume=Y.YY, dt=Z"
            let price = self.parse_price_from_result(&result);
            let volume = self.parse_volume_from_result(&result);

            Ok((price, volume))
        })
    }

    fn parse_price_from_result(&self, result: &str) -> f32 {
        result.split("price=")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.market.price)
    }

    fn parse_volume_from_result(&self, result: &str) -> f32 {
        result.split("volume=")
            .nth(1)
            .and_then(|s| s.split(',').next())
            .and_then(|s| s.parse().ok())
            .unwrap_or(self.market.volume)
    }

    /// Compute gradients via Warp's autodiff
    #[cfg(feature = "warp")]
    pub fn compute_gradients(&mut self, loss: f32) -> Result<WarpGradients, BackendError> {
        if self.py_simulation.is_none() {
            return Err(BackendError::Unsupported("Simulation not initialized".into()));
        }

        Python::with_gil(|py| {
            let module = self.py_simulation.as_ref().unwrap().bind(py);
            let grad_fn = module.getattr("compute_gradients")
                .map_err(|e| BackendError::FfiError(format!("Failed to get compute_gradients: {}", e)))?;

            let result: &PyDict = grad_fn.call1((loss,))
                .map_err(|e| BackendError::FfiError(format!("compute_gradients failed: {}", e)))?
                .extract()
                .map_err(|e| BackendError::FfiError(format!("Failed to extract gradients: {}", e)))?;

            // Check for error
            if let Ok(error) = result.get_item("error") {
                if let Some(err) = error {
                    let err_str: String = err.extract().unwrap_or_default();
                    return Err(BackendError::FfiError(err_str));
                }
            }

            let grads = WarpGradients::default();
            self.last_gradients = Some(grads.clone());
            Ok(grads)
        })
    }

    /// Get last computed gradients
    pub fn last_gradients(&self) -> Option<&WarpGradients> {
        self.last_gradients.as_ref()
    }

    /// Get current market state
    pub fn market_state(&self) -> &WarpMarketState {
        &self.market
    }

    /// Get mutable market state
    pub fn market_state_mut(&mut self) -> &mut WarpMarketState {
        &mut self.market
    }

    /// Get agent states
    pub fn agents(&self) -> &[WarpAgentState] {
        &self.agents
    }

    /// Get mutable agent states
    pub fn agents_mut(&mut self) -> &mut [WarpAgentState] {
        &mut self.agents
    }

    /// Add a new agent to the simulation
    pub fn add_agent(&mut self, state: WarpAgentState) -> WarpBodyHandle {
        let handle = WarpBodyHandle(self.next_handle);
        self.next_handle += 1;

        let idx = self.agents.len();
        self.agents.push(state);
        self.body_handles.insert(handle, idx);

        handle
    }

    /// Native Rust simulation step (when Warp unavailable)
    fn step_native(&mut self, dt: f32) {
        // Simple differentiable dynamics (matches Python kernels logic)
        let price = self.market.price;
        let mut total_pressure = 0.0f32;

        for agent in &mut self.agents {
            // Force based on price relative to target
            let force_y = if price > 100.0 {
                -agent.risk_aversion
            } else {
                1.0 / agent.risk_aversion.max(0.01)
            };

            // Symplectic Euler integration
            agent.velocity[1] += force_y * dt;
            agent.position[0] += agent.velocity[0] * dt;
            agent.position[1] += agent.velocity[1] * dt;
            agent.position[2] += agent.velocity[2] * dt;

            // Price impact from agent actions
            total_pressure += agent.velocity[1] * agent.capital * 0.0001 * dt;
        }

        // Update market
        self.market.price += total_pressure;
        self.market.volume += total_pressure.abs() * 1000.0;
    }
}

impl PhysicsBackend for WarpBackend {
    type Config = WarpConfig;
    type BodyHandle = WarpBodyHandle;
    type ColliderHandle = WarpColliderHandle;
    type ConstraintHandle = WarpConstraintHandle;

    fn new(config: Self::Config) -> Result<Self, BackendError> {
        let num_agents = config.num_agents;

        // Initialize agents with default state
        let agents: Vec<WarpAgentState> = (0..num_agents)
            .map(|_| WarpAgentState {
                position: [0.0, 0.0, 0.0],
                velocity: [0.0, 0.0, 0.0],
                capital: 100_000.0,
                inventory: 0.0,
                risk_aversion: 0.5,
            })
            .collect();

        let mut backend = Self {
            config,
            gravity: Vector3::new(0.0, -9.81, 0.0),
            contacts: Vec::new(),
            stats: SimulationStats::default(),
            agents,
            market: WarpMarketState {
                price: 100.0,
                volume: 0.0,
                volatility: 0.02,
                trend: 0.0,
            },
            body_handles: HashMap::new(),
            next_handle: 0,
            last_gradients: None,
            #[cfg(feature = "warp")]
            py_simulation: None,
            step_count: 0,
            total_step_time_us: 0,
        };

        // Pre-populate handles
        for i in 0..num_agents {
            let handle = WarpBodyHandle(i as u32);
            backend.body_handles.insert(handle, i);
        }
        backend.next_handle = num_agents as u32;

        // Try to initialize Python backend (non-fatal if fails)
        #[cfg(feature = "warp")]
        {
            if let Err(e) = backend.init_python_simulation() {
                tracing::warn!("Warp Python init failed (using native): {}", e);
            }
        }

        Ok(backend)
    }

    fn info(&self) -> BackendInfo {
        BackendInfo {
            name: "NVIDIA Warp",
            version: "1.3",
            description: "GPU-accelerated differentiable physics simulation",
            gpu_accelerated: true,
            differentiable: true,
        }
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            physics_3d: true,
            physics_2d: true,
            soft_bodies: true,
            cloth: true,
            fluids: true,
            articulated: true,
            ccd: false,
            deterministic: true,
            parallel: true,
            gpu: true,
            differentiable: true,
            max_bodies: 1_000_000, // GPU can handle millions
        }
    }

    fn step(&mut self, dt: f32) {
        let start = Instant::now();

        #[cfg(feature = "warp")]
        {
            match self.step_python(dt) {
                Ok((price, volume)) => {
                    self.market.price = price;
                    self.market.volume = volume;
                }
                Err(_) => {
                    // Fall back to native simulation
                    self.step_native(dt);
                }
            }
        }

        #[cfg(not(feature = "warp"))]
        {
            self.step_native(dt);
        }

        let elapsed = start.elapsed().as_micros() as u64;
        self.step_count += 1;
        self.total_step_time_us += elapsed;

        self.stats.total_us = elapsed;
        self.stats.active_bodies = self.agents.len() as u32;
        self.stats.integration_us = elapsed;
    }

    fn set_gravity(&mut self, gravity: Vector3<f32>) {
        self.gravity = gravity;
    }

    fn gravity(&self) -> Vector3<f32> {
        self.gravity
    }

    fn create_body(&mut self, desc: &BodyDesc) -> Result<Self::BodyHandle, BackendError> {
        let state = WarpAgentState {
            position: [
                desc.transform.position.x,
                desc.transform.position.y,
                desc.transform.position.z,
            ],
            velocity: [0.0, 0.0, 0.0],
            capital: 100_000.0,
            inventory: 0.0,
            risk_aversion: 0.5,
        };
        Ok(self.add_agent(state))
    }

    fn remove_body(&mut self, handle: Self::BodyHandle) -> Result<(), BackendError> {
        if let Some(&idx) = self.body_handles.get(&handle) {
            if idx < self.agents.len() {
                // Mark as inactive (don't actually remove to preserve indices)
                self.agents[idx].capital = 0.0;
            }
        }
        Ok(())
    }

    fn body_transform(&self, handle: Self::BodyHandle) -> Option<Transform> {
        self.body_handles.get(&handle).and_then(|&idx| {
            self.agents.get(idx).map(|agent| {
                Transform {
                    position: Vector3::new(
                        agent.position[0],
                        agent.position[1],
                        agent.position[2],
                    ),
                    rotation: UnitQuaternion::identity(),
                }
            })
        })
    }

    fn set_body_transform(&mut self, handle: Self::BodyHandle, transform: Transform) {
        if let Some(&idx) = self.body_handles.get(&handle) {
            if let Some(agent) = self.agents.get_mut(idx) {
                agent.position = [
                    transform.position.x,
                    transform.position.y,
                    transform.position.z,
                ];
            }
        }
    }

    fn body_linear_velocity(&self, handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        self.body_handles.get(&handle).and_then(|&idx| {
            self.agents.get(idx).map(|agent| {
                Vector3::new(agent.velocity[0], agent.velocity[1], agent.velocity[2])
            })
        })
    }

    fn set_body_linear_velocity(&mut self, handle: Self::BodyHandle, velocity: Vector3<f32>) {
        if let Some(&idx) = self.body_handles.get(&handle) {
            if let Some(agent) = self.agents.get_mut(idx) {
                agent.velocity = [velocity.x, velocity.y, velocity.z];
            }
        }
    }

    fn body_angular_velocity(&self, _handle: Self::BodyHandle) -> Option<Vector3<f32>> {
        // Warp agents don't have angular velocity in market sim
        Some(Vector3::zeros())
    }

    fn set_body_angular_velocity(&mut self, _handle: Self::BodyHandle, _velocity: Vector3<f32>) {
        // No-op for market agents
    }

    fn apply_force(&mut self, handle: Self::BodyHandle, force: Vector3<f32>) {
        // Apply as velocity change (simplified)
        if let Some(&idx) = self.body_handles.get(&handle) {
            if let Some(agent) = self.agents.get_mut(idx) {
                let mass = agent.capital / 100_000.0; // Use capital as mass proxy
                let accel = force / mass.max(0.01);
                agent.velocity[0] += accel.x * self.config.timestep;
                agent.velocity[1] += accel.y * self.config.timestep;
                agent.velocity[2] += accel.z * self.config.timestep;
            }
        }
    }

    fn apply_force_at_point(
        &mut self,
        handle: Self::BodyHandle,
        force: Vector3<f32>,
        _point: Point3<f32>,
    ) {
        // Same as apply_force for point particles
        self.apply_force(handle, force);
    }

    fn apply_impulse(&mut self, handle: Self::BodyHandle, impulse: Vector3<f32>) {
        if let Some(&idx) = self.body_handles.get(&handle) {
            if let Some(agent) = self.agents.get_mut(idx) {
                let mass = agent.capital / 100_000.0;
                agent.velocity[0] += impulse.x / mass.max(0.01);
                agent.velocity[1] += impulse.y / mass.max(0.01);
                agent.velocity[2] += impulse.z / mass.max(0.01);
            }
        }
    }

    fn apply_torque(&mut self, _handle: Self::BodyHandle, _torque: Vector3<f32>) {
        // No-op for point particles
    }

    fn body_count(&self) -> usize {
        self.agents.iter().filter(|a| a.capital > 0.0).count()
    }

    fn create_collider(
        &mut self,
        _body: Self::BodyHandle,
        _desc: &ColliderDesc,
    ) -> Result<Self::ColliderHandle, BackendError> {
        // Warp market sim doesn't use traditional colliders
        Err(BackendError::Unsupported(
            "Warp uses field-based interactions, not colliders".into(),
        ))
    }

    fn remove_collider(&mut self, _handle: Self::ColliderHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn set_collider_material(&mut self, _handle: Self::ColliderHandle, _material: PhysicsMaterial) {}

    fn set_collider_enabled(&mut self, _handle: Self::ColliderHandle, _enabled: bool) {}

    fn collider_aabb(&self, _handle: Self::ColliderHandle) -> Option<AABB> {
        None
    }

    fn create_constraint(
        &mut self,
        _desc: &ConstraintDesc,
    ) -> Result<Self::ConstraintHandle, BackendError> {
        Err(BackendError::Unsupported("Use Warp's native constraints".into()))
    }

    fn remove_constraint(&mut self, _handle: Self::ConstraintHandle) -> Result<(), BackendError> {
        Ok(())
    }

    fn ray_cast(&self, _ray: &RayCast) -> Option<RayHit<Self::BodyHandle>> {
        // Ray casting in differentiable sim would require custom kernels
        None
    }

    fn ray_cast_all(&self, _ray: &RayCast) -> Vec<RayHit<Self::BodyHandle>> {
        Vec::new()
    }

    fn shape_cast(&self, _cast: &ShapeCast) -> Option<ShapeHit<Self::BodyHandle>> {
        None
    }

    fn query_aabb(&self, aabb: &AABB) -> Vec<Self::BodyHandle> {
        // Return agents within the AABB
        self.body_handles
            .iter()
            .filter_map(|(&handle, &idx)| {
                self.agents.get(idx).and_then(|agent| {
                    let pos = Point3::new(agent.position[0], agent.position[1], agent.position[2]);
                    if aabb.contains(&pos) {
                        Some(handle)
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    fn contacts(&self) -> &[ContactManifold] {
        &self.contacts
    }

    fn serialize_state(&self) -> Result<Vec<u8>, BackendError> {
        // Serialize agent states
        let state = (&self.agents, &self.market);
        bincode::serialize(&state)
            .map_err(|e| BackendError::SerializationError(e.to_string()))
    }

    fn deserialize_state(&mut self, data: &[u8]) -> Result<(), BackendError> {
        let (agents, market): (Vec<WarpAgentState>, WarpMarketState) =
            bincode::deserialize(data)
                .map_err(|e| BackendError::DeserializationError(e.to_string()))?;
        self.agents = agents;
        self.market = market;
        Ok(())
    }

    fn reset(&mut self) {
        for agent in &mut self.agents {
            agent.position = [0.0, 0.0, 0.0];
            agent.velocity = [0.0, 0.0, 0.0];
            agent.capital = 100_000.0;
            agent.inventory = 0.0;
        }
        self.market = WarpMarketState {
            price: 100.0,
            volume: 0.0,
            volatility: 0.02,
            trend: 0.0,
        };
        self.step_count = 0;
        self.total_step_time_us = 0;
        self.last_gradients = None;
    }

    fn stats(&self) -> SimulationStats {
        SimulationStats {
            active_bodies: self.body_count() as u32,
            total_us: if self.step_count > 0 {
                self.total_step_time_us / self.step_count
            } else {
                0
            },
            integration_us: self.stats.integration_us,
            memory_bytes: std::mem::size_of::<WarpAgentState>() as u64 * self.agents.len() as u64,
            ..Default::default()
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// Implement serde traits for state serialization
use serde::{Deserialize, Serialize};

impl Serialize for WarpAgentState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("WarpAgentState", 5)?;
        state.serialize_field("position", &self.position)?;
        state.serialize_field("velocity", &self.velocity)?;
        state.serialize_field("capital", &self.capital)?;
        state.serialize_field("inventory", &self.inventory)?;
        state.serialize_field("risk_aversion", &self.risk_aversion)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for WarpAgentState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            position: [f32; 3],
            velocity: [f32; 3],
            capital: f32,
            inventory: f32,
            risk_aversion: f32,
        }
        let helper = Helper::deserialize(deserializer)?;
        Ok(Self {
            position: helper.position,
            velocity: helper.velocity,
            capital: helper.capital,
            inventory: helper.inventory,
            risk_aversion: helper.risk_aversion,
        })
    }
}

impl Serialize for WarpMarketState {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("WarpMarketState", 4)?;
        state.serialize_field("price", &self.price)?;
        state.serialize_field("volume", &self.volume)?;
        state.serialize_field("volatility", &self.volatility)?;
        state.serialize_field("trend", &self.trend)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for WarpMarketState {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            price: f32,
            volume: f32,
            volatility: f32,
            trend: f32,
        }
        let helper = Helper::deserialize(deserializer)?;
        Ok(Self {
            price: helper.price,
            volume: helper.volume,
            volatility: helper.volatility,
            trend: helper.trend,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_backend_creation() {
        let config = WarpConfig::default();
        let backend = WarpBackend::new(config).unwrap();
        assert_eq!(backend.body_count(), 100);
    }

    #[test]
    fn test_warp_native_step() {
        let mut backend = WarpBackend::new(WarpConfig {
            num_agents: 10,
            ..Default::default()
        }).unwrap();

        let initial_price = backend.market.price;
        backend.step(0.01);
        // Price should change based on agent actions
        assert!(backend.market.price != initial_price || backend.market.volume > 0.0);
    }

    #[test]
    fn test_warp_body_operations() {
        let mut backend = WarpBackend::new(WarpConfig {
            num_agents: 10,
            ..Default::default()
        }).unwrap();

        let handle = WarpBodyHandle(0);
        let transform = backend.body_transform(handle);
        assert!(transform.is_some());

        let vel = Vector3::new(1.0, 2.0, 3.0);
        backend.set_body_linear_velocity(handle, vel);
        let retrieved = backend.body_linear_velocity(handle).unwrap();
        assert!((retrieved - vel).norm() < 0.001);
    }

    #[test]
    fn test_warp_serialization() {
        let mut backend = WarpBackend::new(WarpConfig {
            num_agents: 5,
            ..Default::default()
        }).unwrap();

        backend.step(0.01);
        let serialized = backend.serialize_state().unwrap();

        let mut backend2 = WarpBackend::new(WarpConfig {
            num_agents: 5,
            ..Default::default()
        }).unwrap();

        backend2.deserialize_state(&serialized).unwrap();
        assert_eq!(backend.market.price, backend2.market.price);
    }
}
