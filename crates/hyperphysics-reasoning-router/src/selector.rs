//! Backend selection strategies: Thompson Sampling, Racing, Ensemble.

use crate::backend::{BackendId, ReasoningBackend};
use crate::problem::ProblemSignature;
use crate::{RouterError, RouterResult};
use rand::Rng;
use rand_distr::{Beta, Distribution};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Selection strategy for choosing backends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SelectionStrategy {
    /// Lookup table for exact signature matches (< 10μs)
    Instant,
    /// Thompson Sampling for exploration/exploitation (< 100μs)
    ThompsonSampling,
    /// Run multiple backends in parallel, first valid wins
    ParallelRacing,
    /// Run multiple backends and synthesize results
    Ensemble,
    /// Always use the best known backend
    Greedy,
    /// Random selection (for baseline comparison)
    Random,
}

/// Thompson Sampler for Bayesian bandit backend selection
pub struct ThompsonSampler {
    /// Beta distribution parameters (alpha, beta) per backend
    /// alpha = successes + 1, beta = failures + 1
    params: HashMap<BackendId, (f64, f64)>,
    /// Prior strength (how much to weight prior vs observations)
    prior_strength: f64,
}

impl ThompsonSampler {
    /// Create a new Thompson Sampler
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
            prior_strength: 1.0,
        }
    }

    /// Initialize or get parameters for a backend
    fn get_params(&self, backend_id: &BackendId) -> (f64, f64) {
        self.params
            .get(backend_id)
            .copied()
            .unwrap_or((self.prior_strength, self.prior_strength))
    }

    /// Sample from the posterior for a backend
    pub fn sample(&self, backend_id: &BackendId) -> f64 {
        let (alpha, beta_param) = self.get_params(backend_id);
        let beta = Beta::new(alpha, beta_param).unwrap_or(Beta::new(1.0, 1.0).unwrap());
        let mut rng = rand::thread_rng();
        beta.sample(&mut rng)
    }

    /// Update posterior with observation
    pub fn update(&mut self, backend_id: &BackendId, success: bool, quality: f64) {
        let (alpha, beta_param) = self.get_params(backend_id);

        // Scale update by quality
        let update_weight = if success { quality } else { 1.0 - quality };

        let new_params = if success {
            (alpha + update_weight, beta_param)
        } else {
            (alpha, beta_param + update_weight)
        };

        self.params.insert(backend_id.clone(), new_params);
    }

    /// Select best backend from candidates using Thompson Sampling
    pub fn select(&self, candidates: &[&BackendId]) -> Option<BackendId> {
        if candidates.is_empty() {
            return None;
        }

        candidates
            .iter()
            .map(|&id| {
                let sample = self.sample(id);
                (id.clone(), sample)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    /// Get the expected value (mean) for a backend
    pub fn expected_value(&self, backend_id: &BackendId) -> f64 {
        let (alpha, beta_param) = self.get_params(backend_id);
        alpha / (alpha + beta_param)
    }

    /// Get uncertainty (variance) for a backend
    pub fn uncertainty(&self, backend_id: &BackendId) -> f64 {
        let (alpha, beta_param) = self.get_params(backend_id);
        let sum = alpha + beta_param;
        (alpha * beta_param) / (sum * sum * (sum + 1.0))
    }
}

impl Default for ThompsonSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Backend selector combining multiple strategies
pub struct BackendSelector {
    /// Thompson sampler for Bayesian selection
    thompson: ThompsonSampler,
    /// Lookup table: signature hash -> best backend
    lookup_table: HashMap<u64, BackendId>,
    /// Performance history per (signature_hash, backend_id)
    performance_history: HashMap<(u64, BackendId), PerformanceStats>,
    /// Default strategy
    default_strategy: SelectionStrategy,
}

impl BackendSelector {
    /// Create a new backend selector
    pub fn new(strategy: SelectionStrategy) -> Self {
        Self {
            thompson: ThompsonSampler::new(),
            lookup_table: HashMap::new(),
            performance_history: HashMap::new(),
            default_strategy: strategy,
        }
    }

    /// Select backends for a problem
    pub fn select(
        &self,
        signature: &ProblemSignature,
        available: &[Arc<dyn ReasoningBackend>],
        strategy: Option<SelectionStrategy>,
    ) -> RouterResult<Vec<BackendId>> {
        let strategy = strategy.unwrap_or(self.default_strategy);

        // Filter to backends that can handle this problem
        let capable: Vec<_> = available
            .iter()
            .filter(|b| b.can_handle(signature))
            .collect();

        if capable.is_empty() {
            return Err(RouterError::NoBackendAvailable(signature.domain));
        }

        match strategy {
            SelectionStrategy::Instant => {
                // Try lookup table first
                let sig_hash = self.signature_hash(signature);
                if let Some(backend_id) = self.lookup_table.get(&sig_hash) {
                    if capable.iter().any(|b| b.id() == backend_id) {
                        return Ok(vec![backend_id.clone()]);
                    }
                }
                // Fall back to greedy
                self.select_greedy(&capable)
            }

            SelectionStrategy::ThompsonSampling => {
                let ids: Vec<&BackendId> = capable.iter().map(|b| b.id()).collect();
                if let Some(selected) = self.thompson.select(&ids) {
                    Ok(vec![selected])
                } else {
                    self.select_greedy(&capable)
                }
            }

            SelectionStrategy::ParallelRacing => {
                // Return multiple backends for parallel execution
                let max_racers = 3.min(capable.len());
                let mut ids: Vec<_> = capable.iter().map(|b| b.id().clone()).collect();

                // Sort by expected value descending
                ids.sort_by(|a, b| {
                    let ev_a = self.thompson.expected_value(a);
                    let ev_b = self.thompson.expected_value(b);
                    ev_b.partial_cmp(&ev_a).unwrap_or(std::cmp::Ordering::Equal)
                });

                ids.truncate(max_racers);
                Ok(ids)
            }

            SelectionStrategy::Ensemble => {
                // Return all capable backends for ensemble
                let max_ensemble = 5.min(capable.len());
                let mut ids: Vec<_> = capable.iter().map(|b| b.id().clone()).collect();
                ids.truncate(max_ensemble);
                Ok(ids)
            }

            SelectionStrategy::Greedy => self.select_greedy(&capable),

            SelectionStrategy::Random => {
                let mut rng = rand::thread_rng();
                let idx = rng.gen_range(0..capable.len());
                Ok(vec![capable[idx].id().clone()])
            }
        }
    }

    /// Greedy selection based on metrics
    fn select_greedy(
        &self,
        capable: &[&Arc<dyn ReasoningBackend>],
    ) -> RouterResult<Vec<BackendId>> {
        let best = capable
            .iter()
            .max_by(|a, b| {
                let metrics_a = a.metrics();
                let metrics_b = b.metrics();

                // Score = success_rate * quality / latency
                let score_a =
                    metrics_a.success_rate * metrics_a.avg_quality / (metrics_a.avg_latency.as_secs_f64() + 0.001);
                let score_b =
                    metrics_b.success_rate * metrics_b.avg_quality / (metrics_b.avg_latency.as_secs_f64() + 0.001);

                score_a.partial_cmp(&score_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|b| b.id().clone());

        match best {
            Some(id) => Ok(vec![id]),
            None => Err(RouterError::NoBackendAvailable(crate::ProblemDomain::General)),
        }
    }

    /// Record result and update selection model
    pub fn record_result(
        &mut self,
        signature: &ProblemSignature,
        backend_id: &BackendId,
        success: bool,
        quality: f64,
        latency: std::time::Duration,
    ) {
        // Update Thompson sampler
        self.thompson.update(backend_id, success, quality);

        // Update performance history
        let sig_hash = self.signature_hash(signature);
        let key = (sig_hash, backend_id.clone());
        let stats = self.performance_history.entry(key).or_insert_with(PerformanceStats::default);
        stats.record(success, quality, latency);

        // Update lookup table if this is the best result
        if success && quality > 0.8 {
            let current_best = self.lookup_table.get(&sig_hash);
            let should_update = match current_best {
                None => true,
                Some(current_id) => {
                    let current_ev = self.thompson.expected_value(current_id);
                    let new_ev = self.thompson.expected_value(backend_id);
                    new_ev > current_ev
                }
            };
            if should_update {
                self.lookup_table.insert(sig_hash, backend_id.clone());
            }
        }
    }

    /// Compute hash for a problem signature
    fn signature_hash(&self, signature: &ProblemSignature) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        signature.problem_type.hash(&mut hasher);
        signature.domain.hash(&mut hasher);
        signature.dimensionality.hash(&mut hasher);
        signature.latency_budget.hash(&mut hasher);
        signature.structure.hash(&mut hasher);

        hasher.finish()
    }

    /// Get statistics for a backend
    pub fn get_stats(&self, backend_id: &BackendId) -> Option<&PerformanceStats> {
        self.performance_history
            .iter()
            .find(|((_, id), _)| id == backend_id)
            .map(|(_, stats)| stats)
    }
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new(SelectionStrategy::ThompsonSampling)
    }
}

/// Performance statistics for a backend
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Total calls
    pub calls: u64,
    /// Successful calls
    pub successes: u64,
    /// Sum of quality scores
    pub quality_sum: f64,
    /// Sum of latencies (ms)
    pub latency_sum_ms: f64,
    /// Min latency (ms)
    pub min_latency_ms: f64,
    /// Max latency (ms)
    pub max_latency_ms: f64,
}

impl PerformanceStats {
    /// Record a result
    pub fn record(&mut self, success: bool, quality: f64, latency: std::time::Duration) {
        self.calls += 1;
        if success {
            self.successes += 1;
        }
        self.quality_sum += quality;

        let latency_ms = latency.as_secs_f64() * 1000.0;
        self.latency_sum_ms += latency_ms;

        if self.calls == 1 {
            self.min_latency_ms = latency_ms;
            self.max_latency_ms = latency_ms;
        } else {
            self.min_latency_ms = self.min_latency_ms.min(latency_ms);
            self.max_latency_ms = self.max_latency_ms.max(latency_ms);
        }
    }

    /// Get success rate
    pub fn success_rate(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.successes as f64 / self.calls as f64
        }
    }

    /// Get average quality
    pub fn avg_quality(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.quality_sum / self.calls as f64
        }
    }

    /// Get average latency in ms
    pub fn avg_latency_ms(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.latency_sum_ms / self.calls as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thompson_sampler_initialization() {
        let sampler = ThompsonSampler::new();
        let backend = BackendId::new("test-backend");

        // Initial expected value should be 0.5 (uniform prior)
        let ev = sampler.expected_value(&backend);
        assert!((ev - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_thompson_sampler_update() {
        let mut sampler = ThompsonSampler::new();
        let backend = BackendId::new("test-backend");

        // Record several successes
        for _ in 0..10 {
            sampler.update(&backend, true, 0.9);
        }

        // Expected value should be high
        let ev = sampler.expected_value(&backend);
        assert!(ev > 0.8);

        // Uncertainty should decrease
        let uncertainty = sampler.uncertainty(&backend);
        let initial_uncertainty = ThompsonSampler::new().uncertainty(&BackendId::new("new"));
        assert!(uncertainty < initial_uncertainty);
    }

    #[test]
    fn test_thompson_sampler_selection() {
        let mut sampler = ThompsonSampler::new();

        let backend_good = BackendId::new("good");
        let backend_bad = BackendId::new("bad");

        // Train: good backend has high success rate
        for _ in 0..20 {
            sampler.update(&backend_good, true, 0.95);
            sampler.update(&backend_bad, false, 0.1);
        }

        // Selection should favor good backend
        let mut good_count = 0;
        for _ in 0..100 {
            let selected = sampler.select(&[&backend_good, &backend_bad]).unwrap();
            if selected == backend_good {
                good_count += 1;
            }
        }

        // Good backend should be selected >90% of time
        assert!(good_count > 90);
    }

    #[test]
    fn test_performance_stats() {
        let mut stats = PerformanceStats::default();

        stats.record(true, 0.9, std::time::Duration::from_millis(10));
        stats.record(true, 0.8, std::time::Duration::from_millis(20));
        stats.record(false, 0.2, std::time::Duration::from_millis(100));

        assert_eq!(stats.calls, 3);
        assert_eq!(stats.successes, 2);
        assert!((stats.success_rate() - 2.0 / 3.0).abs() < 0.01);
        assert!((stats.avg_quality() - 0.633).abs() < 0.01);
        assert!((stats.avg_latency_ms() - 43.33).abs() < 1.0);
    }

    #[test]
    fn test_selection_strategy_variants() {
        // Just ensure all variants are distinct
        assert_ne!(SelectionStrategy::Instant, SelectionStrategy::ThompsonSampling);
        assert_ne!(SelectionStrategy::ParallelRacing, SelectionStrategy::Ensemble);
    }
}
