//! Unified backend trait and types for reasoning systems.

use crate::{BackendPool, LatencyTier, ProblemDomain, RouterResult};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Unique identifier for a backend
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BackendId(pub String);

impl BackendId {
    /// Create a new backend ID
    pub fn new(name: impl Into<String>) -> Self {
        Self(name.into())
    }
}

impl std::fmt::Display for BackendId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Capability flags for backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BackendCapability {
    /// Deterministic execution (bit-exact reproducibility)
    Deterministic,
    /// GPU acceleration available
    GpuAccelerated,
    /// SIMD optimization
    SimdOptimized,
    /// Differentiable (supports gradients)
    Differentiable,
    /// Supports parallel scenarios
    ParallelScenarios,
    /// Sparse data structures
    SparseSupport,
    /// Streaming/incremental updates
    Streaming,
    /// Constraint handling
    ConstraintHandling,
    /// Multi-objective optimization
    MultiObjective,
    /// Uncertainty quantification
    UncertaintyQuantification,
}

/// Performance metrics for a backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendMetrics {
    /// Total number of calls
    pub total_calls: u64,
    /// Successful calls
    pub successful_calls: u64,
    /// Failed calls
    pub failed_calls: u64,
    /// Average latency
    pub avg_latency: Duration,
    /// P50 latency
    pub p50_latency: Duration,
    /// P99 latency
    pub p99_latency: Duration,
    /// Min latency observed
    pub min_latency: Duration,
    /// Max latency observed
    pub max_latency: Duration,
    /// Success rate [0, 1]
    pub success_rate: f64,
    /// Average quality score [0, 1]
    pub avg_quality: f64,
    /// Timestamp of last update
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

impl Default for BackendMetrics {
    fn default() -> Self {
        Self {
            total_calls: 0,
            successful_calls: 0,
            failed_calls: 0,
            avg_latency: Duration::ZERO,
            p50_latency: Duration::ZERO,
            p99_latency: Duration::ZERO,
            min_latency: Duration::MAX,
            max_latency: Duration::ZERO,
            success_rate: 1.0,
            avg_quality: 0.5,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl BackendMetrics {
    /// Update metrics with a new observation
    pub fn record(&mut self, latency: Duration, success: bool, quality: Option<f64>) {
        self.total_calls += 1;
        if success {
            self.successful_calls += 1;
        } else {
            self.failed_calls += 1;
        }

        // Update latency stats (exponential moving average)
        let alpha = 0.1;
        let new_avg = self.avg_latency.as_secs_f64() * (1.0 - alpha) + latency.as_secs_f64() * alpha;
        self.avg_latency = Duration::from_secs_f64(new_avg);

        self.min_latency = self.min_latency.min(latency);
        self.max_latency = self.max_latency.max(latency);

        // Update success rate
        self.success_rate = self.successful_calls as f64 / self.total_calls as f64;

        // Update quality if provided
        if let Some(q) = quality {
            self.avg_quality = self.avg_quality * (1.0 - alpha) + q * alpha;
        }

        self.last_updated = chrono::Utc::now();
    }
}

/// Result from a reasoning backend
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// Computed result value (backend-specific interpretation)
    pub value: ResultValue,
    /// Confidence in the result [0, 1]
    pub confidence: f64,
    /// Quality score [0, 1]
    pub quality: f64,
    /// Execution latency
    pub latency: Duration,
    /// Backend that produced this result
    pub backend_id: BackendId,
    /// Optional metadata
    pub metadata: serde_json::Value,
}

/// Backend-specific result values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResultValue {
    /// Scalar value
    Scalar(f64),
    /// Vector of scalars
    Vector(Vec<f64>),
    /// Solution with parameters and fitness
    Solution {
        /// Optimal parameters
        parameters: Vec<f64>,
        /// Fitness/objective value
        fitness: f64,
    },
    /// Physics state
    PhysicsState {
        /// Positions (flattened 3D vectors)
        positions: Vec<f64>,
        /// Velocities (flattened 3D vectors)
        velocities: Vec<f64>,
        /// Total energy
        energy: f64,
    },
    /// Boolean verification result
    Verified(bool),
    /// Classification result
    Classification {
        /// Class label
        class: u32,
        /// Class probabilities
        probabilities: Vec<f64>,
    },
    /// Structured JSON output
    Structured(serde_json::Value),
}

/// Core trait for all reasoning backends
///
/// This trait provides a unified interface for:
/// - Physics engines (Rapier, Jolt, Warp, Taichi, HyperPhysics)
/// - Optimization algorithms (PSO, GA, ACO, etc.)
/// - Statistical methods (Monte Carlo, Bayesian, Kalman)
/// - Formal verification (Z3, Lean4)
#[async_trait]
pub trait ReasoningBackend: Send + Sync {
    /// Get unique identifier for this backend
    fn id(&self) -> &BackendId;

    /// Get human-readable name
    fn name(&self) -> &str;

    /// Get the backend pool this belongs to
    fn pool(&self) -> BackendPool;

    /// Get supported problem domains
    fn supported_domains(&self) -> &[ProblemDomain];

    /// Get backend capabilities
    fn capabilities(&self) -> &HashSet<BackendCapability>;

    /// Get typical latency tier
    fn latency_tier(&self) -> LatencyTier;

    /// Check if backend can handle a problem with given signature
    fn can_handle(&self, signature: &crate::problem::ProblemSignature) -> bool;

    /// Estimate execution time for a problem
    fn estimate_latency(&self, signature: &crate::problem::ProblemSignature) -> Duration;

    /// Execute reasoning on the given problem
    async fn execute(
        &self,
        problem: &crate::problem::Problem,
    ) -> RouterResult<ReasoningResult>;

    /// Get current performance metrics
    fn metrics(&self) -> BackendMetrics;

    /// Check if backend is healthy and available
    fn is_healthy(&self) -> bool {
        true
    }

    /// Warm up the backend (e.g., compile shaders, allocate pools)
    async fn warmup(&mut self) -> RouterResult<()> {
        Ok(())
    }
}

/// Wrapper for executing backends with timing and metrics
pub struct BackendExecutor {
    backend: Arc<dyn ReasoningBackend>,
    metrics: parking_lot::Mutex<BackendMetrics>,
}

impl BackendExecutor {
    /// Create a new executor for a backend
    pub fn new(backend: Arc<dyn ReasoningBackend>) -> Self {
        Self {
            backend,
            metrics: parking_lot::Mutex::new(BackendMetrics::default()),
        }
    }

    /// Execute with timing and metrics recording
    pub async fn execute_timed(
        &self,
        problem: &crate::problem::Problem,
    ) -> RouterResult<ReasoningResult> {
        let start = Instant::now();
        let result = self.backend.execute(problem).await;
        let latency = start.elapsed();

        let mut metrics = self.metrics.lock();
        match &result {
            Ok(r) => metrics.record(latency, true, Some(r.quality)),
            Err(_) => metrics.record(latency, false, None),
        }

        result
    }

    /// Get backend reference
    pub fn backend(&self) -> &dyn ReasoningBackend {
        &*self.backend
    }

    /// Get current metrics
    pub fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_id() {
        let id = BackendId::new("rapier-3d");
        assert_eq!(id.0, "rapier-3d");
        assert_eq!(format!("{}", id), "rapier-3d");
    }

    #[test]
    fn test_metrics_recording() {
        let mut metrics = BackendMetrics::default();

        metrics.record(Duration::from_millis(10), true, Some(0.9));
        assert_eq!(metrics.total_calls, 1);
        assert_eq!(metrics.successful_calls, 1);
        assert!(metrics.success_rate > 0.99);

        metrics.record(Duration::from_millis(20), false, None);
        assert_eq!(metrics.total_calls, 2);
        assert_eq!(metrics.failed_calls, 1);
        assert!((metrics.success_rate - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_result_value_variants() {
        let scalar = ResultValue::Scalar(42.0);
        let vector = ResultValue::Vector(vec![1.0, 2.0, 3.0]);
        let solution = ResultValue::Solution {
            parameters: vec![0.5, 0.3],
            fitness: 0.95,
        };

        // Ensure variants serialize correctly
        let _ = serde_json::to_string(&scalar).unwrap();
        let _ = serde_json::to_string(&vector).unwrap();
        let _ = serde_json::to_string(&solution).unwrap();
    }
}
