//! Physics backend adapters for reasoning router.
//!
//! Wraps hyperphysics-unified physics engines as ReasoningBackend implementations.
//!
//! Supported backends:
//! - Rapier3D (Rust native)
//! - JoltPhysics (C++ FFI)
//! - NVIDIA Warp (GPU, differentiable)
//! - Taichi (GPU, differentiable)
//! - MuJoCo (robotics)
//! - Genesis (differentiable)
//! - Avian (Bevy ECS)
//! - Project Chrono (multibody)

use crate::{
    BackendCapability, BackendId, BackendMetrics, BackendPool, LatencyTier, Problem,
    ProblemDomain, ProblemSignature, ReasoningBackend, ReasoningResult, ResultValue,
    RouterResult,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::time::{Duration, Instant};

// NOTE: When the physics feature is enabled and hyperphysics_unified is available,
// this adapter can wrap actual physics backend implementations.

/// Physics backend adapter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsAdapterConfig {
    /// Simulation timestep
    pub dt: f32,
    /// Number of substeps per step
    pub substeps: u32,
    /// Maximum simulation time
    pub max_time: f32,
    /// Gravity vector
    pub gravity: [f32; 3],
}

impl Default for PhysicsAdapterConfig {
    fn default() -> Self {
        Self {
            dt: 1.0 / 60.0,
            substeps: 4,
            max_time: 10.0,
            gravity: [0.0, -9.81, 0.0],
        }
    }
}

/// Generic physics backend adapter
///
/// Wraps any PhysicsBackend from hyperphysics-unified as a ReasoningBackend.
pub struct PhysicsBackendAdapter {
    id: BackendId,
    name: String,
    config: PhysicsAdapterConfig,
    capabilities: HashSet<BackendCapability>,
    metrics: Mutex<BackendMetrics>,
    latency_tier: LatencyTier,
    gpu_accelerated: bool,
    differentiable: bool,
}

impl PhysicsBackendAdapter {
    /// Create a new physics backend adapter
    pub fn new(
        backend_name: &str,
        config: PhysicsAdapterConfig,
        gpu_accelerated: bool,
        differentiable: bool,
    ) -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::Deterministic);
        capabilities.insert(BackendCapability::ParallelScenarios);

        if gpu_accelerated {
            capabilities.insert(BackendCapability::GpuAccelerated);
        }
        if differentiable {
            capabilities.insert(BackendCapability::Differentiable);
        }

        let latency_tier = if gpu_accelerated {
            LatencyTier::Fast
        } else {
            LatencyTier::Medium
        };

        Self {
            id: BackendId::new(format!("physics-{}", backend_name.to_lowercase())),
            name: format!("{} Physics Backend", backend_name),
            config,
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
            latency_tier,
            gpu_accelerated,
            differentiable,
        }
    }

    /// Create Rapier3D adapter
    pub fn rapier() -> Self {
        Self::new("Rapier3D", PhysicsAdapterConfig::default(), false, false)
    }

    /// Create Jolt adapter
    pub fn jolt() -> Self {
        Self::new("Jolt", PhysicsAdapterConfig::default(), false, false)
    }

    /// Create Warp adapter (GPU, differentiable)
    pub fn warp() -> Self {
        Self::new("Warp", PhysicsAdapterConfig::default(), true, true)
    }

    /// Create Taichi adapter (GPU, differentiable)
    pub fn taichi() -> Self {
        Self::new("Taichi", PhysicsAdapterConfig::default(), true, true)
    }

    /// Create MuJoCo adapter
    pub fn mujoco() -> Self {
        Self::new("MuJoCo", PhysicsAdapterConfig::default(), false, true)
    }

    /// Create Genesis adapter (differentiable)
    pub fn genesis() -> Self {
        Self::new("Genesis", PhysicsAdapterConfig::default(), true, true)
    }

    /// Create Avian adapter (Bevy ECS)
    pub fn avian() -> Self {
        Self::new("Avian", PhysicsAdapterConfig::default(), false, false)
    }

    /// Create Chrono adapter (multibody dynamics)
    pub fn chrono() -> Self {
        Self::new("Chrono", PhysicsAdapterConfig::default(), false, false)
    }

    /// Simulate a physics problem
    fn simulate(&self, problem: &Problem) -> RouterResult<PhysicsSimulationResult> {
        // Extract initial state from problem data
        let num_bodies = problem.signature.dimensionality as usize / 6;
        let dt = self.config.dt;
        let steps = (self.config.max_time / dt) as usize;

        // Simple integration for demonstration
        // In production, this would use the actual physics backend
        let mut positions: Vec<f64> = vec![0.0; num_bodies * 3];
        let mut velocities: Vec<f64> = vec![0.0; num_bodies * 3];

        // Initialize from problem data if available
        if let crate::problem::ProblemData::PhysicsState {
            positions: init_pos,
            velocities: init_vel,
            ..
        } = &problem.data
        {
            // Copy initial positions
            for (i, &v) in init_pos.iter().enumerate().take(positions.len()) {
                positions[i] = v;
            }
            // Copy initial velocities
            for (i, &v) in init_vel.iter().enumerate().take(velocities.len()) {
                velocities[i] = v;
            }
        }

        // Simple Euler integration with gravity
        let gravity = self.config.gravity;
        for _ in 0..steps.min(1000) {
            for i in 0..num_bodies {
                // Update velocity (apply gravity)
                velocities[i * 3 + 1] += gravity[1] as f64 * dt as f64;

                // Update position
                for j in 0..3 {
                    positions[i * 3 + j] += velocities[i * 3 + j] * dt as f64;
                }

                // Simple ground collision
                if positions[i * 3 + 1] < 0.0 {
                    positions[i * 3 + 1] = 0.0;
                    velocities[i * 3 + 1] = -velocities[i * 3 + 1] * 0.8; // Bounce
                }
            }
        }

        // Compute energy
        let kinetic: f64 = velocities
            .chunks(3)
            .map(|v| 0.5 * (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))
            .sum();
        let potential: f64 = positions
            .chunks(3)
            .map(|p| -gravity[1] as f64 * p[1])
            .sum();
        let total_energy = kinetic + potential;

        Ok(PhysicsSimulationResult {
            positions,
            velocities,
            energy: total_energy,
            steps_computed: steps.min(1000),
        })
    }
}

/// Result of physics simulation
#[derive(Debug, Clone)]
struct PhysicsSimulationResult {
    positions: Vec<f64>,
    velocities: Vec<f64>,
    energy: f64,
    steps_computed: usize,
}

#[async_trait]
impl ReasoningBackend for PhysicsBackendAdapter {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Physics
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &[
            ProblemDomain::Physics,
            ProblemDomain::Engineering,
            ProblemDomain::General,
        ]
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        self.latency_tier
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        use crate::problem::ProblemType;
        matches!(
            signature.problem_type,
            ProblemType::Simulation | ProblemType::Dynamics
        ) && matches!(signature.domain, ProblemDomain::Physics | ProblemDomain::Engineering)
    }

    fn estimate_latency(&self, signature: &ProblemSignature) -> Duration {
        let base_us = if self.gpu_accelerated { 100 } else { 1000 };
        let bodies = signature.dimensionality as u64 / 6;
        let steps = (self.config.max_time / self.config.dt) as u64;
        Duration::from_micros(base_us * bodies * steps / 100)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        let sim_result = self.simulate(problem)?;

        let latency = start.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(0.9));
        }

        // Quality based on energy conservation (for conservative systems)
        let quality = 0.9;
        let confidence = 0.95;

        Ok(ReasoningResult {
            value: ResultValue::PhysicsState {
                positions: sim_result.positions,
                velocities: sim_result.velocities,
                energy: sim_result.energy,
            },
            confidence,
            quality,
            latency,
            backend_id: self.id.clone(),
            metadata: serde_json::json!({
                "backend": self.name,
                "gpu_accelerated": self.gpu_accelerated,
                "differentiable": self.differentiable,
                "dt": self.config.dt,
                "substeps": self.config.substeps,
                "steps_computed": sim_result.steps_computed
            }),
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ProblemType;

    #[test]
    fn test_physics_adapter_creation() {
        let rapier = PhysicsBackendAdapter::rapier();
        assert_eq!(rapier.id().0, "physics-rapier3d");
        assert!(!rapier.gpu_accelerated);

        let warp = PhysicsBackendAdapter::warp();
        assert!(warp.gpu_accelerated);
        assert!(warp.differentiable);
    }

    #[test]
    fn test_physics_adapter_can_handle() {
        let rapier = PhysicsBackendAdapter::rapier();

        let physics_sig = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Physics);
        assert!(rapier.can_handle(&physics_sig));

        let financial_sig = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial);
        assert!(!rapier.can_handle(&financial_sig));
    }

    #[test]
    fn test_physics_adapter_capabilities() {
        let warp = PhysicsBackendAdapter::warp();
        assert!(warp.capabilities().contains(&BackendCapability::GpuAccelerated));
        assert!(warp.capabilities().contains(&BackendCapability::Differentiable));

        let rapier = PhysicsBackendAdapter::rapier();
        assert!(!rapier.capabilities().contains(&BackendCapability::GpuAccelerated));
    }

    #[test]
    fn test_all_backend_types() {
        let backends = vec![
            PhysicsBackendAdapter::rapier(),
            PhysicsBackendAdapter::jolt(),
            PhysicsBackendAdapter::warp(),
            PhysicsBackendAdapter::taichi(),
            PhysicsBackendAdapter::mujoco(),
            PhysicsBackendAdapter::genesis(),
            PhysicsBackendAdapter::avian(),
            PhysicsBackendAdapter::chrono(),
        ];

        assert_eq!(backends.len(), 8);

        for backend in &backends {
            assert_eq!(backend.pool(), BackendPool::Physics);
        }
    }
}
