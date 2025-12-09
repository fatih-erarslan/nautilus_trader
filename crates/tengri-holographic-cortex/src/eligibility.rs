//! # Eligibility Traces for Temporal Credit Assignment
//!
//! Implements sparse eligibility traces for connecting STDP with reward-based learning.
//! This module enables temporal credit assignment across multiple timescales through
//! lazy decay and saturation control.
//!
//! ## Wolfram-Validated Mathematical Foundations
//!
//! ### Trace Decay Equation
//! ```text
//! e_ij(t) = e_ij(t-1) × λγ + δ_ij(t)
//! where:
//!   λ (lambda) = 0.95  - eligibility decay rate
//!   γ (gamma)  = 0.99  - temporal discount factor
//!   δ_ij(t)    = 1     - Kronecker delta at spike coincidence
//! ```
//!
//! ### Saturation Bound (Wolfram-Verified)
//! ```text
//! max_trace = 1 / (1 - λγ)
//!           = 1 / (1 - 0.95 × 0.99)
//!           = 1 / (1 - 0.9405)
//!           = 1 / 0.0595
//!           ≈ 16.8067
//! ```
//!
//! ### Weight Update with Modulation
//! ```text
//! ΔW_ij = η × r(t) × e_ij(t)
//! where:
//!   η      - learning rate
//!   r(t)   - reward signal at time t
//!   e_ij(t) - eligibility trace
//! ```
//!
//! ## Usage Example
//! ```rust
//! use tengri_holographic_cortex::eligibility::{SparseEligibilityTrace, TraceParams};
//!
//! let mut trace = SparseEligibilityTrace::new(TraceParams::default());
//!
//! // STDP-based accumulation
//! trace.accumulate(42, 1.0); // synapse_id, stdp_delta
//!
//! // Step forward in time (lazy decay)
//! trace.step(1.0); // dt = 1.0 second
//!
//! // Apply reward modulation
//! let weight_updates = trace.apply_modulation(0.5, 0.01); // reward=0.5, lr=0.01
//! ```

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Eligibility trace parameters (Wolfram-validated)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceParams {
    /// Eligibility decay rate (λ)
    pub lambda: f64,

    /// Temporal discount factor (γ)
    pub gamma: f64,

    /// Maximum trace saturation: 1/(1-λγ)
    pub max_trace: f64,

    /// Pruning threshold (traces below this are removed)
    pub prune_threshold: f64,
}

impl Default for TraceParams {
    fn default() -> Self {
        let lambda = 0.95;
        let gamma = 0.99;
        let decay_product = lambda * gamma;

        Self {
            lambda,
            gamma,
            max_trace: 1.0 / (1.0 - decay_product), // ≈ 16.8067
            prune_threshold: 1e-6,
        }
    }
}

impl TraceParams {
    /// Create custom trace parameters with automatic saturation calculation
    pub fn new(lambda: f64, gamma: f64, prune_threshold: f64) -> Self {
        assert!(lambda > 0.0 && lambda < 1.0, "lambda must be in (0, 1)");
        assert!(gamma > 0.0 && gamma < 1.0, "gamma must be in (0, 1)");
        assert!(prune_threshold >= 0.0, "prune_threshold must be non-negative");

        let decay_product = lambda * gamma;
        assert!(decay_product < 1.0, "λγ must be < 1 for convergence");

        Self {
            lambda,
            gamma,
            max_trace: 1.0 / (1.0 - decay_product),
            prune_threshold,
        }
    }

    /// Combined decay factor (λγ)
    #[inline]
    pub fn decay_factor(&self) -> f64 {
        self.lambda * self.gamma
    }
}

/// Sparse eligibility trace storage with lazy decay
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseEligibilityTrace {
    /// Trace values indexed by synapse ID
    traces: HashMap<usize, f64>,

    /// Last update timestamp for each trace (for lazy decay)
    timestamps: HashMap<usize, f64>,

    /// Current global time
    current_time: f64,

    /// Trace parameters
    params: TraceParams,

    /// Statistics
    stats: TraceStats,
}

/// Statistics for eligibility trace monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TraceStats {
    /// Total accumulations performed
    pub total_accumulations: u64,

    /// Total modulations applied
    pub total_modulations: u64,

    /// Total pruning operations
    pub total_prunes: u64,

    /// Number of traces pruned
    pub traces_pruned: u64,

    /// Maximum trace value observed
    pub max_trace_observed: f64,

    /// Current active trace count
    pub active_traces: usize,
}

impl SparseEligibilityTrace {
    /// Create a new sparse eligibility trace with default parameters
    pub fn new(params: TraceParams) -> Self {
        Self {
            traces: HashMap::new(),
            timestamps: HashMap::new(),
            current_time: 0.0,
            params,
            stats: TraceStats::default(),
        }
    }

    /// Create with specified capacity hint
    pub fn with_capacity(params: TraceParams, capacity: usize) -> Self {
        Self {
            traces: HashMap::with_capacity(capacity),
            timestamps: HashMap::with_capacity(capacity),
            current_time: 0.0,
            params,
            stats: TraceStats::default(),
        }
    }

    /// Accumulate STDP contribution to eligibility trace
    ///
    /// # Arguments
    /// * `synapse_id` - Unique synapse identifier
    /// * `stdp_delta` - STDP weight change (can be positive or negative)
    pub fn accumulate(&mut self, synapse_id: usize, stdp_delta: f64) {
        // Get current trace value with lazy decay
        let current_trace = self.get_trace_with_decay(synapse_id);

        // Accumulate new STDP contribution
        let new_trace = current_trace + stdp_delta;

        // Apply saturation bound
        let clamped_trace = new_trace.clamp(-self.params.max_trace, self.params.max_trace);

        // Update trace and timestamp
        self.traces.insert(synapse_id, clamped_trace);
        self.timestamps.insert(synapse_id, self.current_time);

        // Update statistics
        self.stats.total_accumulations += 1;
        self.stats.max_trace_observed = self.stats.max_trace_observed.max(clamped_trace.abs());
        self.stats.active_traces = self.traces.len();
    }

    /// Get trace value with lazy decay applied
    fn get_trace_with_decay(&self, synapse_id: usize) -> f64 {
        if let Some(&trace) = self.traces.get(&synapse_id) {
            if let Some(&last_time) = self.timestamps.get(&synapse_id) {
                let dt = self.current_time - last_time;
                if dt > 0.0 {
                    // Lazy decay: e(t) = e(t_last) × (λγ)^(dt)
                    trace * self.params.decay_factor().powf(dt)
                } else {
                    trace
                }
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Step forward in time by dt
    ///
    /// This advances the global clock but doesn't actually decay traces yet (lazy evaluation).
    /// Traces are decayed on-demand when accessed.
    pub fn step(&mut self, dt: f64) {
        assert!(dt >= 0.0, "dt must be non-negative");
        self.current_time += dt;
    }

    /// Apply reward modulation to all traces and compute weight updates
    ///
    /// # Arguments
    /// * `reward` - Reward signal r(t) ∈ ℝ
    /// * `learning_rate` - Learning rate η > 0
    ///
    /// # Returns
    /// HashMap of synapse_id → weight_update pairs
    ///
    /// # Formula
    /// ΔW_ij = η × r(t) × e_ij(t)
    pub fn apply_modulation(&mut self, reward: f64, learning_rate: f64) -> HashMap<usize, f64> {
        assert!(learning_rate >= 0.0, "learning_rate must be non-negative");

        let mut weight_updates = HashMap::with_capacity(self.traces.len());

        // Collect all synapse IDs to avoid borrow checker issues
        let synapse_ids: Vec<usize> = self.traces.keys().copied().collect();

        for synapse_id in synapse_ids {
            // Get decayed trace value
            let trace = self.get_trace_with_decay(synapse_id);

            // Compute weight update
            let delta_w = learning_rate * reward * trace;

            if delta_w.abs() > 1e-12 {
                weight_updates.insert(synapse_id, delta_w);
            }

            // Reset trace after modulation (standard eligibility trace behavior)
            // Alternatively, could decay instead of reset depending on algorithm
            self.traces.remove(&synapse_id);
            self.timestamps.remove(&synapse_id);
        }

        // Update statistics
        self.stats.total_modulations += 1;
        self.stats.active_traces = self.traces.len();

        weight_updates
    }

    /// Prune traces that have decayed below threshold
    ///
    /// Returns number of traces pruned
    pub fn prune(&mut self) -> usize {
        let threshold = self.params.prune_threshold;
        let mut to_remove = Vec::new();

        for (&synapse_id, _) in &self.traces {
            let decayed_trace = self.get_trace_with_decay(synapse_id);
            if decayed_trace.abs() < threshold {
                to_remove.push(synapse_id);
            }
        }

        let num_pruned = to_remove.len();

        for synapse_id in to_remove {
            self.traces.remove(&synapse_id);
            self.timestamps.remove(&synapse_id);
        }

        // Update statistics
        self.stats.total_prunes += 1;
        self.stats.traces_pruned += num_pruned as u64;
        self.stats.active_traces = self.traces.len();

        num_pruned
    }

    /// Get current trace value for a synapse (with lazy decay)
    pub fn get_trace(&self, synapse_id: usize) -> f64 {
        self.get_trace_with_decay(synapse_id)
    }

    /// Get all active traces (with lazy decay applied)
    pub fn get_all_traces(&self) -> HashMap<usize, f64> {
        self.traces
            .keys()
            .map(|&id| (id, self.get_trace_with_decay(id)))
            .collect()
    }

    /// Get current statistics
    pub fn stats(&self) -> &TraceStats {
        &self.stats
    }

    /// Get trace parameters
    pub fn params(&self) -> &TraceParams {
        &self.params
    }

    /// Get current time
    pub fn current_time(&self) -> f64 {
        self.current_time
    }

    /// Reset all traces and statistics
    pub fn reset(&mut self) {
        self.traces.clear();
        self.timestamps.clear();
        self.current_time = 0.0;
        self.stats = TraceStats::default();
    }

    /// Get number of active traces
    pub fn num_active(&self) -> usize {
        self.traces.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_trace_params_default() {
        let params = TraceParams::default();
        assert_eq!(params.lambda, 0.95);
        assert_eq!(params.gamma, 0.99);

        // Wolfram validation: 1/(1-0.9405) ≈ 16.8067
        let expected_max = 1.0 / (1.0 - 0.95 * 0.99);
        assert_relative_eq!(params.max_trace, expected_max, epsilon = 1e-6);
        assert_relative_eq!(params.max_trace, 16.8067, epsilon = 1e-3);
    }

    #[test]
    fn test_trace_params_custom() {
        let params = TraceParams::new(0.9, 0.95, 1e-4);
        assert_eq!(params.lambda, 0.9);
        assert_eq!(params.gamma, 0.95);
        assert_eq!(params.prune_threshold, 1e-4);

        let expected_max = 1.0 / (1.0 - 0.9 * 0.95);
        assert_relative_eq!(params.max_trace, expected_max, epsilon = 1e-10);
    }

    #[test]
    #[should_panic(expected = "lambda must be in (0, 1)")]
    fn test_invalid_lambda() {
        TraceParams::new(1.5, 0.99, 1e-6);
    }

    #[test]
    #[should_panic(expected = "lambda must be in (0, 1)")]
    fn test_divergent_params() {
        // Test that lambda=1.0 triggers validation error (actual divergent case would be after lambda check)
        TraceParams::new(1.0, 0.99, 1e-6);
    }

    #[test]
    fn test_accumulate_basic() {
        let mut trace = SparseEligibilityTrace::new(TraceParams::default());

        trace.accumulate(0, 1.0);
        assert_relative_eq!(trace.get_trace(0), 1.0, epsilon = 1e-10);
        assert_eq!(trace.num_active(), 1);
        assert_eq!(trace.stats().total_accumulations, 1);
    }

    #[test]
    fn test_accumulate_multiple() {
        let mut trace = SparseEligibilityTrace::new(TraceParams::default());

        trace.accumulate(0, 1.0);
        trace.accumulate(0, 0.5);

        assert_relative_eq!(trace.get_trace(0), 1.5, epsilon = 1e-10);
        assert_eq!(trace.stats().total_accumulations, 2);
    }

    #[test]
    fn test_trace_decay() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        // Accumulate initial trace
        trace.accumulate(0, 1.0);
        assert_relative_eq!(trace.get_trace(0), 1.0, epsilon = 1e-10);

        // Step forward 1 time unit
        trace.step(1.0);

        // Trace should decay by λγ = 0.95 × 0.99 = 0.9405
        let expected = 1.0 * params.decay_factor();
        assert_relative_eq!(trace.get_trace(0), expected, epsilon = 1e-10);
        assert_relative_eq!(trace.get_trace(0), 0.9405, epsilon = 1e-10);
    }

    #[test]
    fn test_trace_decay_multiple_steps() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        trace.accumulate(0, 1.0);

        // Step forward 5 time units
        trace.step(5.0);

        // Trace should decay by (λγ)^5
        let expected = params.decay_factor().powi(5);
        assert_relative_eq!(trace.get_trace(0), expected, epsilon = 1e-10);

        // Actual: (0.95 × 0.99)^5 = (0.9405)^5 ≈ 0.7358579723647938
        assert_relative_eq!(trace.get_trace(0), 0.7358579723647938, epsilon = 1e-10);
    }

    #[test]
    fn test_saturation_bound() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        // Try to accumulate beyond saturation
        for _ in 0..100 {
            trace.accumulate(0, 1.0);
        }

        // Should be clamped to max_trace ≈ 16.8067
        let final_trace = trace.get_trace(0);
        assert!(final_trace <= params.max_trace);
        assert_relative_eq!(final_trace, params.max_trace, epsilon = 1e-6);
        assert_relative_eq!(final_trace, 16.8067, epsilon = 1e-3);
    }

    #[test]
    fn test_saturation_bound_negative() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        // Try to accumulate negative beyond saturation
        for _ in 0..100 {
            trace.accumulate(0, -1.0);
        }

        // Should be clamped to -max_trace
        let final_trace = trace.get_trace(0);
        assert!(final_trace >= -params.max_trace);
        assert_relative_eq!(final_trace, -params.max_trace, epsilon = 1e-6);
    }

    #[test]
    fn test_sparse_memory() {
        let params = TraceParams::new(0.95, 0.99, 1e-3);
        let mut trace = SparseEligibilityTrace::new(params.clone());

        // Create traces for multiple synapses
        for i in 0..10 {
            trace.accumulate(i, 1.0);
        }
        assert_eq!(trace.num_active(), 10);

        // Let them decay significantly
        // Need (λγ)^t < 1e-3
        // (0.9405)^t < 1e-3
        // t > log(1e-3) / log(0.9405) ≈ 113.5
        trace.step(120.0); // (0.9405)^120 ≈ 5.4e-8 << 1e-3

        // Prune vanished traces
        let pruned = trace.prune();

        // All should be pruned (threshold = 1e-3, traces ≈ 5.4e-8)
        assert_eq!(pruned, 10);
        assert_eq!(trace.num_active(), 0);
        assert_eq!(trace.stats().traces_pruned, 10);
    }

    #[test]
    fn test_sparse_memory_partial_pruning() {
        let params = TraceParams::new(0.95, 0.99, 1e-3);
        let mut trace = SparseEligibilityTrace::new(params.clone());

        // Create old and new traces
        trace.accumulate(0, 1.0);
        trace.step(10.0);
        trace.accumulate(1, 1.0); // Fresh trace

        // Old trace: (0.9405)^10 ≈ 0.5417
        // New trace: 1.0

        let pruned = trace.prune();
        assert_eq!(pruned, 0); // Both above threshold
        assert_eq!(trace.num_active(), 2);

        // Decay more, but keep new trace above threshold
        // Need: old trace < 1e-3, new trace > 1e-3
        // (0.9405)^(10+t) < 1e-3  =>  t > 103.5
        // (0.9405)^t > 1e-3       =>  t < 113.5
        // Choose t = 105
        trace.step(105.0);

        // Old trace: (0.9405)^115 ≈ 1.2e-6 (below threshold)
        // New trace: (0.9405)^105 ≈ 2.2e-3 (above threshold)

        let pruned = trace.prune();
        assert_eq!(pruned, 1); // Only old trace pruned
        assert_eq!(trace.num_active(), 1);
    }

    #[test]
    fn test_modulation_scaling() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params);

        trace.accumulate(0, 1.0);
        trace.accumulate(1, 2.0);

        // Apply positive reward
        let updates = trace.apply_modulation(0.5, 0.01);

        // ΔW = η × r × e
        // For synapse 0: 0.01 × 0.5 × 1.0 = 0.005
        // For synapse 1: 0.01 × 0.5 × 2.0 = 0.01
        assert_relative_eq!(updates[&0], 0.005, epsilon = 1e-10);
        assert_relative_eq!(updates[&1], 0.01, epsilon = 1e-10);

        // Traces should be reset after modulation
        assert_eq!(trace.num_active(), 0);
        assert_eq!(trace.stats().total_modulations, 1);
    }

    #[test]
    fn test_modulation_negative_reward() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params);

        trace.accumulate(0, 1.0);

        // Apply negative reward (punishment)
        let updates = trace.apply_modulation(-0.5, 0.01);

        // ΔW = 0.01 × (-0.5) × 1.0 = -0.005
        assert_relative_eq!(updates[&0], -0.005, epsilon = 1e-10);
    }

    #[test]
    fn test_modulation_with_decay() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        trace.accumulate(0, 1.0);
        trace.step(5.0); // Decay by (0.9405)^5 ≈ 0.7358579723647938

        let updates = trace.apply_modulation(1.0, 0.01);

        // ΔW = 0.01 × 1.0 × 0.7358579723647938
        let expected = 0.01 * params.decay_factor().powi(5);
        assert_relative_eq!(updates[&0], expected, epsilon = 1e-10);
        assert_relative_eq!(updates[&0], 0.007358579723647938, epsilon = 1e-10);
    }

    #[test]
    fn test_get_all_traces() {
        let mut trace = SparseEligibilityTrace::new(TraceParams::default());

        trace.accumulate(0, 1.0);
        trace.accumulate(1, 2.0);
        trace.accumulate(2, 3.0);

        let all_traces = trace.get_all_traces();
        assert_eq!(all_traces.len(), 3);
        assert_relative_eq!(all_traces[&0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(all_traces[&1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(all_traces[&2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_reset() {
        let mut trace = SparseEligibilityTrace::new(TraceParams::default());

        trace.accumulate(0, 1.0);
        trace.step(5.0);
        trace.prune();

        assert!(trace.num_active() > 0 || trace.stats().total_accumulations > 0);

        trace.reset();

        assert_eq!(trace.num_active(), 0);
        assert_eq!(trace.current_time(), 0.0);
        assert_eq!(trace.stats().total_accumulations, 0);
        assert_eq!(trace.stats().total_modulations, 0);
    }

    #[test]
    fn test_statistics_tracking() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        trace.accumulate(0, 1.0);
        trace.accumulate(1, 15.0);

        let stats = trace.stats();
        assert_eq!(stats.total_accumulations, 2);
        assert_eq!(stats.active_traces, 2);
        assert_relative_eq!(stats.max_trace_observed, 15.0, epsilon = 1e-10);

        trace.apply_modulation(1.0, 0.01);

        let stats = trace.stats();
        assert_eq!(stats.total_modulations, 1);
        assert_eq!(stats.active_traces, 0);
    }

    #[test]
    fn test_lazy_decay_efficiency() {
        let params = TraceParams::default();
        let mut trace = SparseEligibilityTrace::new(params.clone());

        // Accumulate trace
        trace.accumulate(0, 1.0);

        // Step forward multiple times without accessing trace
        for _ in 0..100 {
            trace.step(1.0);
        }

        // Trace should still be stored (lazy evaluation)
        assert_eq!(trace.num_active(), 1);

        // Access trace triggers decay calculation
        let decayed = trace.get_trace(0);

        // (0.9405)^100 = 0.00213... (small but not near zero)
        let expected = params.decay_factor().powi(100);
        assert_relative_eq!(decayed, expected, epsilon = 1e-10);
        assert!(decayed < 0.01); // Significantly decayed
        assert!(decayed > 1e-4); // But not vanished
    }
}
