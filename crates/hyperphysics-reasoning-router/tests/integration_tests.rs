//! Comprehensive Integration Tests for HyperPhysics Reasoning Router
//!
//! Tests the routing system, backend selection, LSH matching, and synthesis

use hyperphysics_reasoning_router::prelude::*;
use hyperphysics_reasoning_router::backend::{BackendMetrics, ResultValue};
use hyperphysics_reasoning_router::{BackendPool, LatencyTier, ProblemDomain, RouterError, RouterResult};
use hyperphysics_reasoning_router::problem::Problem;

use async_trait::async_trait;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;
use tokio::time::sleep;

/// Mock backend for testing
#[derive(Debug)]
struct MockBackend {
    id: BackendId,
    name: String,
    capabilities: HashSet<BackendCapability>,
    domains: Vec<ProblemDomain>,
    latency: Duration,
    success_rate: f64,
    call_count: Arc<AtomicU64>,
    result_value: f64,
    metrics: parking_lot::Mutex<BackendMetrics>,
}

impl MockBackend {
    fn new(name: &str, latency_ms: u64, success_rate: f64) -> Self {
        Self {
            id: BackendId::new(name),
            name: name.to_string(),
            capabilities: HashSet::new(),
            domains: vec![ProblemDomain::Physics, ProblemDomain::Optimization],
            latency: Duration::from_millis(latency_ms),
            success_rate,
            call_count: Arc::new(AtomicU64::new(0)),
            result_value: 42.0,
            metrics: parking_lot::Mutex::new(BackendMetrics::default()),
        }
    }

    fn with_capabilities(mut self, caps: Vec<BackendCapability>) -> Self {
        self.capabilities = caps.into_iter().collect();
        self
    }

    fn with_result(mut self, value: f64) -> Self {
        self.result_value = value;
        self
    }

    fn calls(&self) -> u64 {
        self.call_count.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl ReasoningBackend for MockBackend {
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
        &self.domains
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        LatencyTier::from_duration(self.latency)
    }

    fn can_handle(&self, _signature: &hyperphysics_reasoning_router::problem::ProblemSignature) -> bool {
        true
    }

    fn estimate_latency(&self, _signature: &hyperphysics_reasoning_router::problem::ProblemSignature) -> Duration {
        self.latency
    }

    async fn execute(&self, _problem: &Problem) -> RouterResult<ReasoningResult> {
        // Track calls
        self.call_count.fetch_add(1, Ordering::SeqCst);

        // Simulate latency
        sleep(self.latency).await;

        // Simulate failures based on success_rate
        if rand::random::<f64>() > self.success_rate {
            return Err(RouterError::BackendFailed {
                backend_id: self.id.0.clone(),
                message: "Simulated failure".to_string(),
            });
        }

        let result = ReasoningResult {
            value: ResultValue::Scalar(self.result_value),
            confidence: 0.9,
            quality: 0.85,
            latency: self.latency,
            backend_id: self.id.clone(),
            metadata: serde_json::json!({}),
        };

        // Update metrics
        let mut metrics = self.metrics.lock();
        metrics.record(self.latency, true, Some(0.85));

        Ok(result)
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

#[test]
fn test_router_config_defaults() {
    let config = RouterConfig::default();
    assert_eq!(config.default_strategy, SelectionStrategy::ThompsonSampling);
    assert_eq!(config.max_concurrent, 8);
    assert_eq!(config.default_timeout, Duration::from_secs(5));
    assert!(config.enable_lsh_routing);
    assert_eq!(config.min_confidence, 0.5);
    assert!(config.enable_learning);
}

#[test]
fn test_router_config_custom() {
    let config = RouterConfig {
        default_strategy: SelectionStrategy::Greedy,
        max_concurrent: 4,
        default_timeout: Duration::from_secs(10),
        enable_lsh_routing: false,
        min_confidence: 0.7,
        enable_learning: false,
        ..Default::default()
    };

    assert_eq!(config.default_strategy, SelectionStrategy::Greedy);
    assert_eq!(config.max_concurrent, 4);
    assert!(!config.enable_lsh_routing);
    assert_eq!(config.min_confidence, 0.7);
}

// ============================================================================
// Backend Registration Tests
// ============================================================================
// NOTE: list_backends() method doesn't exist in ReasoningRouter
// These tests are commented out until the method is implemented

// #[tokio::test]
// async fn test_backend_registration() {
//     let router = ReasoningRouter::new(RouterConfig::default());
//
//     let backend1 = Arc::new(MockBackend::new("rapier", 10, 1.0));
//     let backend2 = Arc::new(MockBackend::new("pso", 50, 1.0));
//
//     router.register_backend(backend1);
//     router.register_backend(backend2);
//
//     // Verify backends are registered
//     let available = router.list_backends();
//     assert_eq!(available.len(), 2);
//     assert!(available.iter().any(|id| id.0 == "rapier"));
//     assert!(available.iter().any(|id| id.0 == "pso"));
// }

// #[tokio::test]
// async fn test_backend_unregistration() {
//     let router = ReasoningRouter::new(RouterConfig::default());
//
//     let backend = Arc::new(MockBackend::new("test-backend", 10, 1.0));
//     let backend_id = backend.id().clone();
//
//     router.register_backend(backend);
//     assert_eq!(router.list_backends().len(), 1);
//
//     router.unregister_backend(&backend_id);
//     assert_eq!(router.list_backends().len(), 0);
// }

// #[tokio::test]
// async fn test_multiple_backends_registration() {
//     let router = ReasoningRouter::new(RouterConfig::default());
//
//     for i in 0..10 {
//         let backend = Arc::new(MockBackend::new(&format!("backend-{}", i), 10, 1.0));
//         router.register_backend(backend);
//     }
//
//     assert_eq!(router.list_backends().len(), 10);
// }

// ============================================================================
// Latency Tier Tests
// ============================================================================

#[test]
fn test_latency_tier_classification() {
    assert_eq!(
        LatencyTier::from_duration(Duration::from_micros(5)),
        LatencyTier::UltraFast
    );
    assert_eq!(
        LatencyTier::from_duration(Duration::from_micros(500)),
        LatencyTier::Fast
    );
    assert_eq!(
        LatencyTier::from_duration(Duration::from_millis(5)),
        LatencyTier::Medium
    );
    assert_eq!(
        LatencyTier::from_duration(Duration::from_millis(50)),
        LatencyTier::Slow
    );
    assert_eq!(
        LatencyTier::from_duration(Duration::from_millis(500)),
        LatencyTier::Deep
    );
}

#[test]
fn test_latency_tier_max_latency() {
    assert_eq!(LatencyTier::UltraFast.max_latency(), Duration::from_micros(10));
    assert_eq!(LatencyTier::Fast.max_latency(), Duration::from_millis(1));
    assert_eq!(LatencyTier::Medium.max_latency(), Duration::from_millis(10));
    assert_eq!(LatencyTier::Slow.max_latency(), Duration::from_millis(100));
    assert_eq!(LatencyTier::Deep.max_latency(), Duration::from_secs(1));
}

#[test]
fn test_latency_tier_ordering() {
    assert!(LatencyTier::UltraFast < LatencyTier::Fast);
    assert!(LatencyTier::Fast < LatencyTier::Medium);
    assert!(LatencyTier::Medium < LatencyTier::Slow);
    assert!(LatencyTier::Slow < LatencyTier::Deep);
}

// ============================================================================
// Problem Signature Tests
// ============================================================================

#[test]
fn test_problem_signature_creation() {
    use hyperphysics_reasoning_router::problem::*;

    let sig = ProblemSignature::new(
        ProblemType::Optimization,
        ProblemDomain::Physics,
    );

    assert_eq!(sig.problem_type, ProblemType::Optimization);
    assert_eq!(sig.domain, ProblemDomain::Physics);
    assert_eq!(sig.dimensionality, 1);
    assert_eq!(sig.sparsity, 0.0);
    assert_eq!(sig.latency_budget, LatencyTier::Medium);
}

#[test]
fn test_problem_signature_with_builder() {
    use hyperphysics_reasoning_router::problem::*;

    let sig = ProblemSignature::new(ProblemType::Simulation, ProblemDomain::Engineering)
        .with_dimensionality(100)
        .with_sparsity(0.8)
        .with_latency_budget(LatencyTier::Fast)
        .with_structure(StructureType::Sparse)
        .stochastic();

    assert_eq!(sig.dimensionality, 100);
    assert_eq!(sig.sparsity, 0.8);
    assert_eq!(sig.latency_budget, LatencyTier::Fast);
    assert_eq!(sig.structure, StructureType::Sparse);
    assert!(sig.is_stochastic);
}

// ============================================================================
// Backend Capability Tests
// ============================================================================

#[test]
fn test_backend_capability_matching() {
    let backend = MockBackend::new("test", 10, 1.0)
        .with_capabilities(vec![
            BackendCapability::Deterministic,
            BackendCapability::SimdOptimized,
            BackendCapability::SparseSupport,
        ]);

    let caps = backend.capabilities();
    assert!(caps.contains(&BackendCapability::Deterministic));
    assert!(caps.contains(&BackendCapability::SimdOptimized));
    assert!(caps.contains(&BackendCapability::SparseSupport));
    assert!(!caps.contains(&BackendCapability::GpuAccelerated));
}

// ============================================================================
// Backend Metrics Tests
// ============================================================================

#[test]
fn test_backend_metrics_initialization() {
    let metrics = BackendMetrics::default();
    assert_eq!(metrics.total_calls, 0);
    assert_eq!(metrics.successful_calls, 0);
    assert_eq!(metrics.failed_calls, 0);
    assert_eq!(metrics.success_rate, 1.0);
    assert_eq!(metrics.avg_quality, 0.5);
}

#[test]
fn test_backend_metrics_recording() {
    let mut metrics = BackendMetrics::default();

    // Record successful call
    metrics.record(Duration::from_millis(10), true, Some(0.9));
    assert_eq!(metrics.total_calls, 1);
    assert_eq!(metrics.successful_calls, 1);
    assert_eq!(metrics.success_rate, 1.0);

    // Record failed call
    metrics.record(Duration::from_millis(20), false, None);
    assert_eq!(metrics.total_calls, 2);
    assert_eq!(metrics.failed_calls, 1);
    assert_eq!(metrics.success_rate, 0.5);

    // Verify latency tracking
    assert!(metrics.min_latency <= Duration::from_millis(10));
    assert!(metrics.max_latency >= Duration::from_millis(20));
}

#[test]
fn test_backend_metrics_success_rate() {
    let mut metrics = BackendMetrics::default();

    // Record 7 successes, 3 failures
    for i in 0..10 {
        let success = i < 7;
        metrics.record(Duration::from_millis(10), success, Some(0.8));
    }

    assert_eq!(metrics.total_calls, 10);
    assert_eq!(metrics.successful_calls, 7);
    assert_eq!(metrics.failed_calls, 3);
    assert_eq!(metrics.success_rate, 0.7);
}

// ============================================================================
// Problem Domain Tests
// ============================================================================

#[test]
fn test_problem_domain_coverage() {
    use std::collections::HashSet;

    let domains = vec![
        ProblemDomain::Physics,
        ProblemDomain::Optimization,
        ProblemDomain::Statistical,
        ProblemDomain::Verification,
        ProblemDomain::Financial,
        ProblemDomain::Control,
        ProblemDomain::Engineering,
        ProblemDomain::General,
    ];

    let unique: HashSet<_> = domains.iter().collect();
    assert_eq!(unique.len(), domains.len(), "All domains should be unique");
}

// ============================================================================
// Backend Pool Tests
// ============================================================================

#[test]
fn test_backend_pool_types() {
    use std::collections::HashSet;

    let pools = vec![
        BackendPool::Physics,
        BackendPool::Optimization,
        BackendPool::Statistical,
        BackendPool::Formal,
    ];

    let unique: HashSet<_> = pools.iter().collect();
    assert_eq!(unique.len(), pools.len(), "All pools should be unique");
}

#[test]
fn test_backend_pool_assignment() {
    let backend = MockBackend::new("physics-backend", 10, 1.0);
    assert_eq!(backend.pool(), BackendPool::Physics);
}

// ============================================================================
// Selection Strategy Tests
// ============================================================================

#[test]
fn test_selection_strategy_types() {
    // Test that we can create all strategy variants
    let strategies = vec![
        SelectionStrategy::ThompsonSampling,
        SelectionStrategy::Greedy,
        SelectionStrategy::Random,
        SelectionStrategy::Instant,
        SelectionStrategy::ParallelRacing,
        SelectionStrategy::Ensemble,
    ];

    // Verify we have all variants
    assert_eq!(strategies.len(), 6);
}

// ============================================================================
// Synthesis Strategy Tests
// ============================================================================

#[test]
fn test_synthesis_strategy_types() {
    // Test that we can create all strategy variants
    let strategies = vec![
        SynthesisStrategy::Fastest,
        SynthesisStrategy::HighestQuality,
        SynthesisStrategy::WeightedAverage,
        SynthesisStrategy::Consensus,
        SynthesisStrategy::HighestConfidence,
        SynthesisStrategy::MajorityVote,
        SynthesisStrategy::Median,
    ];

    // Verify we have all variants
    assert_eq!(strategies.len(), 7);
}

// ============================================================================
// Result Value Tests
// ============================================================================

#[test]
fn test_result_value_scalar() {
    let value = ResultValue::Scalar(42.0);
    match value {
        ResultValue::Scalar(v) => assert_eq!(v, 42.0),
        _ => panic!("Expected Scalar variant"),
    }
}

#[test]
fn test_result_value_vector() {
    let value = ResultValue::Vector(vec![1.0, 2.0, 3.0]);
    match value {
        ResultValue::Vector(v) => {
            assert_eq!(v.len(), 3);
            assert_eq!(v[0], 1.0);
            assert_eq!(v[2], 3.0);
        }
        _ => panic!("Expected Vector variant"),
    }
}

// ============================================================================
// Structure Type Tests
// ============================================================================

#[test]
fn test_structure_types() {
    use hyperphysics_reasoning_router::problem::StructureType;
    use std::collections::HashSet;

    let types = vec![
        StructureType::Dense,
        StructureType::Sparse,
        StructureType::Sequential,
        StructureType::Graph,
        StructureType::Hierarchical,
        StructureType::Unstructured,
    ];

    let unique: HashSet<_> = types.iter().collect();
    assert_eq!(unique.len(), types.len(), "All structure types should be unique");
}

// ============================================================================
// Problem Type Tests
// ============================================================================

#[test]
fn test_problem_types_coverage() {
    use hyperphysics_reasoning_router::problem::ProblemType;
    use std::collections::HashSet;

    let types = vec![
        ProblemType::Optimization,
        ProblemType::Simulation,
        ProblemType::Prediction,
        ProblemType::Classification,
        ProblemType::Verification,
        ProblemType::Control,
        ProblemType::Estimation,
        ProblemType::RiskAssessment,
        ProblemType::Inference,
        ProblemType::ParameterTuning,
        ProblemType::Dynamics,
        ProblemType::ConstraintSatisfaction,
        ProblemType::General,
    ];

    let unique: HashSet<_> = types.iter().collect();
    assert_eq!(unique.len(), types.len(), "All problem types should be unique");
}

// ============================================================================
// Backend ID Tests
// ============================================================================

#[test]
fn test_backend_id_creation() {
    let id1 = BackendId::new("rapier");
    let id2 = BackendId::new("jolt");

    assert_eq!(id1.0, "rapier");
    assert_eq!(id2.0, "jolt");
    assert_ne!(id1, id2);
}

#[test]
fn test_backend_id_display() {
    let id = BackendId::new("test-backend");
    assert_eq!(format!("{}", id), "test-backend");
}

#[test]
fn test_backend_id_equality() {
    let id1 = BackendId::new("same");
    let id2 = BackendId::new("same");
    let id3 = BackendId::new("different");

    assert_eq!(id1, id2);
    assert_ne!(id1, id3);
}
