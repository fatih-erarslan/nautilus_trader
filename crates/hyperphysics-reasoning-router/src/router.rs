//! Core reasoning router implementation.

use crate::backend::{BackendCapability, BackendExecutor, BackendId, BackendMetrics, ReasoningBackend, ReasoningResult};
use crate::lsh::{LSHConfig, LSHIndex, ProblemSolutionRecord};
use crate::problem::{Problem, ProblemSignature};
use crate::selector::{BackendSelector, SelectionStrategy};
use crate::synthesis::{ResultSynthesizer, SynthesisStrategy};
use crate::{BackendPool, LatencyTier, ProblemDomain, RouterError, RouterResult};

use dashmap::DashMap;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;
use tracing::{debug, info};

/// Router configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterConfig {
    /// Default selection strategy
    pub default_strategy: SelectionStrategy,
    /// Maximum concurrent backend executions
    pub max_concurrent: usize,
    /// Default timeout per backend
    pub default_timeout: Duration,
    /// Enable LSH-based routing
    pub enable_lsh_routing: bool,
    /// LSH configuration
    pub lsh_config: LSHConfig,
    /// Minimum confidence threshold
    pub min_confidence: f64,
    /// Enable performance learning
    pub enable_learning: bool,
    /// Synthesis strategy for ensemble results
    pub synthesis_strategy: SynthesisStrategy,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            default_strategy: SelectionStrategy::ThompsonSampling,
            max_concurrent: 8,
            default_timeout: Duration::from_secs(5),
            enable_lsh_routing: true,
            lsh_config: LSHConfig::default(),
            min_confidence: 0.5,
            enable_learning: true,
            synthesis_strategy: SynthesisStrategy::WeightedAverage,
        }
    }
}

/// The main reasoning router
pub struct ReasoningRouter {
    /// Configuration
    config: RouterConfig,
    /// Registered backends
    backends: DashMap<BackendId, Arc<BackendExecutor>>,
    /// Backend selector
    selector: RwLock<BackendSelector>,
    /// LSH index for similar problem lookup
    lsh_index: RwLock<LSHIndex<ProblemSolutionRecord>>,
    /// Result synthesizer
    synthesizer: ResultSynthesizer,
    /// Concurrency limiter
    semaphore: Arc<Semaphore>,
    /// Total problems solved
    problems_solved: std::sync::atomic::AtomicU64,
}

impl ReasoningRouter {
    /// Create a new reasoning router
    pub fn new(config: RouterConfig) -> Self {
        let semaphore = Arc::new(Semaphore::new(config.max_concurrent));
        let lsh_config = config.lsh_config.clone();
        let synthesis_strategy = config.synthesis_strategy;

        Self {
            selector: RwLock::new(BackendSelector::new(config.default_strategy)),
            lsh_index: RwLock::new(LSHIndex::new(lsh_config)),
            synthesizer: ResultSynthesizer::new(synthesis_strategy),
            semaphore,
            backends: DashMap::new(),
            config,
            problems_solved: std::sync::atomic::AtomicU64::new(0),
        }
    }

    /// Register a backend
    pub fn register_backend(&self, backend: Arc<dyn ReasoningBackend>) {
        let id = backend.id().clone();
        let executor = Arc::new(BackendExecutor::new(backend));
        self.backends.insert(id.clone(), executor);
        info!("Registered backend: {}", id);
    }

    /// Unregister a backend
    pub fn unregister_backend(&self, backend_id: &BackendId) {
        if self.backends.remove(backend_id).is_some() {
            info!("Unregistered backend: {}", backend_id);
        }
    }

    /// Solve a problem using the best available backend(s)
    pub async fn solve(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        self.solve_with_strategy(problem, None).await
    }

    /// Solve a problem with a specific strategy
    pub async fn solve_with_strategy(
        &self,
        problem: &Problem,
        strategy: Option<SelectionStrategy>,
    ) -> RouterResult<ReasoningResult> {
        let start = Instant::now();
        let strategy = strategy.unwrap_or(self.config.default_strategy);

        debug!(
            "Solving problem {} with strategy {:?}",
            problem.id, strategy
        );

        // Try LSH lookup for fast routing
        if self.config.enable_lsh_routing && strategy == SelectionStrategy::Instant {
            if let Some(result) = self.try_lsh_route(problem).await? {
                return Ok(result);
            }
        }

        // Collect capable backends
        let capable_executors: Vec<(BackendId, Arc<BackendExecutor>)> = self
            .backends
            .iter()
            .filter(|r| r.value().backend().can_handle(&problem.signature))
            .map(|r| (r.key().clone(), r.value().clone()))
            .collect();

        if capable_executors.is_empty() {
            return Err(RouterError::NoBackendAvailable(problem.signature.domain));
        }

        // Select backends using the selector
        let backend_ids = {
            let selector = self.selector.read();

            // Build wrapper backends for selection
            let wrappers: Vec<Arc<dyn ReasoningBackend>> = capable_executors
                .iter()
                .map(|(id, exec)| {
                    Arc::new(BackendWrapper {
                        id: id.clone(),
                        executor: exec.clone(),
                    }) as Arc<dyn ReasoningBackend>
                })
                .collect();

            selector.select(&problem.signature, &wrappers, Some(strategy))?
        };

        if backend_ids.is_empty() {
            return Err(RouterError::NoBackendAvailable(problem.signature.domain));
        }

        // Execute based on strategy
        let result = match strategy {
            SelectionStrategy::ParallelRacing => {
                self.execute_racing(problem, &backend_ids).await?
            }
            SelectionStrategy::Ensemble => {
                self.execute_ensemble(problem, &backend_ids).await?
            }
            _ => {
                // Single backend execution
                let backend_id = &backend_ids[0];
                self.execute_single(problem, backend_id).await?
            }
        };

        // Record result for learning
        if self.config.enable_learning {
            self.record_result(problem, &result);
        }

        self.problems_solved
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        debug!(
            "Problem {} solved in {:?} with quality {}",
            problem.id,
            start.elapsed(),
            result.quality
        );

        Ok(result)
    }

    /// Try LSH-based routing for fast lookup
    async fn try_lsh_route(&self, problem: &Problem) -> RouterResult<Option<ReasoningResult>> {
        let lsh_index = self.lsh_index.read();

        let results = lsh_index.query_signature(&problem.signature, 1);

        if let Some((similarity, record)) = results.first() {
            if *similarity > 0.95 {
                // Very similar problem found
                let backend_id = BackendId::new(&record.backend_id);
                if let Some(executor) = self.backends.get(&backend_id) {
                    debug!(
                        "LSH cache hit with similarity {:.3} -> backend {}",
                        similarity, backend_id
                    );
                    let result = executor.execute_timed(problem).await?;
                    return Ok(Some(result));
                }
            }
        }

        Ok(None)
    }

    /// Execute on a single backend
    async fn execute_single(
        &self,
        problem: &Problem,
        backend_id: &BackendId,
    ) -> RouterResult<ReasoningResult> {
        let executor = self
            .backends
            .get(backend_id)
            .ok_or_else(|| RouterError::BackendFailed {
                backend_id: backend_id.to_string(),
                message: "Backend not found".to_string(),
            })?;

        let _permit = self
            .semaphore
            .acquire()
            .await
            .map_err(|e| RouterError::BackendFailed {
                backend_id: backend_id.to_string(),
                message: format!("Semaphore error: {}", e),
            })?;

        let timeout = self.get_timeout_for_problem(problem);

        tokio::time::timeout(timeout, executor.execute_timed(problem))
            .await
            .map_err(|_| RouterError::Timeout(timeout))?
    }

    /// Execute parallel racing - first valid result wins
    async fn execute_racing(
        &self,
        problem: &Problem,
        backend_ids: &[BackendId],
    ) -> RouterResult<ReasoningResult> {
        use tokio::sync::mpsc;

        let (tx, mut rx) = mpsc::channel(backend_ids.len());
        let timeout = self.get_timeout_for_problem(problem);

        for backend_id in backend_ids {
            if let Some(executor) = self.backends.get(backend_id) {
                let executor = executor.clone();
                let problem = problem.clone();
                let tx = tx.clone();
                let semaphore = self.semaphore.clone();

                tokio::spawn(async move {
                    let _permit = semaphore.acquire().await;
                    if let Ok(result) = executor.execute_timed(&problem).await {
                        let _ = tx.send(result).await;
                    }
                });
            }
        }

        drop(tx); // Close sender

        // Wait for first result or timeout
        tokio::time::timeout(timeout, rx.recv())
            .await
            .map_err(|_| RouterError::Timeout(timeout))?
            .ok_or_else(|| RouterError::SynthesisFailed("No results from racing".to_string()))
    }

    /// Execute ensemble - run all and synthesize
    async fn execute_ensemble(
        &self,
        problem: &Problem,
        backend_ids: &[BackendId],
    ) -> RouterResult<ReasoningResult> {
        let timeout = self.get_timeout_for_problem(problem);
        let mut handles = Vec::new();

        for backend_id in backend_ids {
            if let Some(executor) = self.backends.get(backend_id) {
                let executor = executor.clone();
                let problem = problem.clone();
                let semaphore = self.semaphore.clone();

                handles.push(tokio::spawn(async move {
                    let _permit = semaphore.acquire().await;
                    executor.execute_timed(&problem).await
                }));
            }
        }

        // Collect results
        let mut results = Vec::new();
        for handle in handles {
            if let Ok(Ok(result)) = tokio::time::timeout(timeout, handle).await {
                if let Ok(r) = result {
                    results.push(r);
                }
            }
        }

        if results.is_empty() {
            return Err(RouterError::SynthesisFailed(
                "No results from ensemble".to_string(),
            ));
        }

        // Synthesize results
        self.synthesizer.synthesize(results)
    }

    /// Get timeout for a problem based on its signature
    fn get_timeout_for_problem(&self, problem: &Problem) -> Duration {
        let tier_timeout = problem.signature.latency_budget.max_latency();
        tier_timeout.max(self.config.default_timeout)
    }

    /// Record result for learning
    fn record_result(&self, problem: &Problem, result: &ReasoningResult) {
        // Update selector
        {
            let mut selector = self.selector.write();
            selector.record_result(
                &problem.signature,
                &result.backend_id,
                result.quality > self.config.min_confidence,
                result.quality,
                result.latency,
            );
        }

        // Update LSH index
        if result.quality > 0.8 {
            let record = ProblemSolutionRecord {
                features: problem.signature.to_feature_vector().to_vec(),
                backend_id: result.backend_id.to_string(),
                quality: result.quality,
                latency_ms: result.latency.as_secs_f64() * 1000.0,
            };

            let mut lsh_index = self.lsh_index.write();
            lsh_index.insert(record.features.clone(), record);
        }
    }

    /// Get router statistics
    pub fn stats(&self) -> RouterStats {
        let backends: Vec<_> = self
            .backends
            .iter()
            .map(|r| {
                let id = r.key().clone();
                let metrics = r.value().metrics();
                (id, metrics)
            })
            .collect();

        RouterStats {
            total_backends: backends.len(),
            problems_solved: self
                .problems_solved
                .load(std::sync::atomic::Ordering::Relaxed),
            lsh_entries: self.lsh_index.read().len(),
            backend_metrics: backends.into_iter().collect(),
        }
    }

    /// Warm up all backends
    pub async fn warmup(&self) -> RouterResult<()> {
        info!("Warming up {} backends...", self.backends.len());
        // Backends are immutable through the executor, so warmup would need
        // a different approach in production
        Ok(())
    }
}

/// Wrapper to make BackendExecutor implement ReasoningBackend
struct BackendWrapper {
    id: BackendId,
    executor: Arc<BackendExecutor>,
}

#[async_trait::async_trait]
impl ReasoningBackend for BackendWrapper {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        self.executor.backend().name()
    }

    fn pool(&self) -> BackendPool {
        self.executor.backend().pool()
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        self.executor.backend().supported_domains()
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        self.executor.backend().capabilities()
    }

    fn latency_tier(&self) -> LatencyTier {
        self.executor.backend().latency_tier()
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        self.executor.backend().can_handle(signature)
    }

    fn estimate_latency(&self, signature: &ProblemSignature) -> Duration {
        self.executor.backend().estimate_latency(signature)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        self.executor.execute_timed(problem).await
    }

    fn metrics(&self) -> BackendMetrics {
        self.executor.metrics()
    }
}

/// Router statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouterStats {
    /// Number of registered backends
    pub total_backends: usize,
    /// Total problems solved
    pub problems_solved: u64,
    /// Number of LSH index entries
    pub lsh_entries: usize,
    /// Per-backend metrics
    pub backend_metrics: HashMap<BackendId, crate::backend::BackendMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_router_config_default() {
        let config = RouterConfig::default();
        assert_eq!(config.max_concurrent, 8);
        assert_eq!(config.default_strategy, SelectionStrategy::ThompsonSampling);
        assert!(config.enable_learning);
    }

    #[test]
    fn test_router_creation() {
        let router = ReasoningRouter::new(RouterConfig::default());
        let stats = router.stats();
        assert_eq!(stats.total_backends, 0);
        assert_eq!(stats.problems_solved, 0);
    }
}
