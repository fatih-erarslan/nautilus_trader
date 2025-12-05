//! # STDP Learning Module - Phase 2: Language Acquisition
//!
//! Implements Spike-Timing Dependent Plasticity with eligibility traces
//! for the "Creating Language" framework (Christiansen & Chater).
//!
//! ## Mathematical Foundation
//!
//! STDP weight update rule:
//! - Pre-before-post (LTP): Δw = A+ × exp(-Δt/τ+)
//! - Post-before-pre (LTD): Δw = -A- × exp(Δt/τ-)
//!
//! Three-factor learning with eligibility traces:
//! - e(t+dt) = e(t) × exp(-dt/τe) + STDP(Δt)
//! - Δw = η × e × M (modulation signal)
//!
//! ## References
//! - Bi & Poo (1998) "Synaptic modifications in cultured hippocampal neurons"
//! - Song et al. (2000) "Competitive Hebbian learning through STDP"
//! - Frémaux & Gerstner (2016) "Neuromodulated STDP and theory of three-factor learning"

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// STDP Configuration
// ============================================================================

/// STDP learning parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STDPConfig {
    /// LTP amplitude (A+)
    pub a_plus: f64,
    /// LTD amplitude (A-)
    pub a_minus: f64,
    /// LTP time constant (τ+) in ms
    pub tau_plus: f64,
    /// LTD time constant (τ-) in ms
    pub tau_minus: f64,
    /// Base learning rate (η)
    pub learning_rate: f64,
    /// Maximum weight
    pub w_max: f64,
    /// Minimum weight
    pub w_min: f64,
    /// Hyperbolic locality scale (λ)
    pub locality_scale: f64,
    /// Enable soft bounds
    pub soft_bounds: bool,
    /// Eligibility trace time constant (τe)
    pub tau_eligibility: f64,
    /// Enable three-factor learning
    pub three_factor: bool,
}

impl Default for STDPConfig {
    fn default() -> Self {
        Self {
            a_plus: 0.01,
            a_minus: 0.012, // Slightly asymmetric for stability
            tau_plus: 20.0,
            tau_minus: 20.0,
            learning_rate: 0.001,
            w_max: 1.0,
            w_min: 0.0,
            locality_scale: 2.0,
            soft_bounds: true,
            tau_eligibility: 100.0, // 100ms eligibility window
            three_factor: true,
        }
    }
}

// ============================================================================
// Eligibility Trace
// ============================================================================

/// Eligibility trace for three-factor learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EligibilityTrace {
    /// Trace value per synapse (pre_id, post_id) -> trace
    traces: HashMap<(u32, u32), f64>,
    /// Time constant for trace decay
    tau: f64,
    /// Last update time
    last_update: f64,
    /// Maximum trace magnitude
    max_trace: f64,
}

impl EligibilityTrace {
    /// Create new eligibility trace manager
    pub fn new(tau: f64) -> Self {
        Self {
            traces: HashMap::new(),
            tau,
            last_update: 0.0,
            max_trace: 1.0,
        }
    }

    /// Decay all traces by elapsed time
    pub fn decay(&mut self, current_time: f64) {
        let dt = current_time - self.last_update;
        if dt <= 0.0 {
            return;
        }

        let decay_factor = (-dt / self.tau).exp();

        // Decay all traces and remove near-zero ones
        self.traces.retain(|_, trace| {
            *trace *= decay_factor;
            trace.abs() > 1e-10
        });

        self.last_update = current_time;
    }

    /// Add STDP contribution to eligibility trace
    pub fn accumulate(&mut self, pre_id: u32, post_id: u32, stdp_value: f64, current_time: f64) {
        // First decay existing traces
        self.decay(current_time);

        // Accumulate new STDP contribution
        let key = (pre_id, post_id);
        let trace = self.traces.entry(key).or_insert(0.0);
        *trace = (*trace + stdp_value).clamp(-self.max_trace, self.max_trace);
    }

    /// Get current eligibility trace for a synapse
    pub fn get(&self, pre_id: u32, post_id: u32) -> f64 {
        self.traces.get(&(pre_id, post_id)).copied().unwrap_or(0.0)
    }

    /// Apply modulation signal and compute weight updates
    pub fn apply_modulation(&mut self, modulation: f64, learning_rate: f64) -> Vec<((u32, u32), f64)> {
        self.traces
            .iter()
            .map(|(&synapse, &trace)| {
                let delta_w = learning_rate * trace * modulation;
                (synapse, delta_w)
            })
            .collect()
    }

    /// Clear all traces
    pub fn clear(&mut self) {
        self.traces.clear();
    }

    /// Get number of active traces
    pub fn active_count(&self) -> usize {
        self.traces.len()
    }
}

// ============================================================================
// STDP Learning Rule
// ============================================================================

/// Spike-Timing Dependent Plasticity learning rule
#[derive(Debug, Clone)]
pub struct STDPLearner {
    /// Configuration
    config: STDPConfig,
    /// Eligibility traces for three-factor learning
    eligibility: EligibilityTrace,
    /// Spike history per neuron (neuron_id -> spike times)
    spike_history: HashMap<u32, VecDeque<f64>>,
    /// Maximum spike history length
    max_history: usize,
    /// Current modulation signal (reward/dopamine)
    modulation_signal: f64,
    /// Weight change accumulator
    weight_updates: HashMap<(u32, u32), f64>,
    /// Statistics
    stats: STDPStats,
}

/// STDP learning statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct STDPStats {
    /// Total LTP events
    pub ltp_events: u64,
    /// Total LTD events
    pub ltd_events: u64,
    /// Average weight change magnitude
    pub avg_weight_change: f64,
    /// Total modulation applications
    pub modulation_events: u64,
    /// Active eligibility traces
    pub active_traces: usize,
}

impl STDPLearner {
    /// Create new STDP learner
    pub fn new(config: STDPConfig) -> Self {
        let tau_e = config.tau_eligibility;
        Self {
            config,
            eligibility: EligibilityTrace::new(tau_e),
            spike_history: HashMap::new(),
            max_history: 100,
            modulation_signal: 1.0, // Neutral modulation by default
            weight_updates: HashMap::new(),
            stats: STDPStats::default(),
        }
    }

    /// Record a spike for a neuron
    pub fn record_spike(&mut self, neuron_id: u32, time: f64) {
        let history = self.spike_history
            .entry(neuron_id)
            .or_insert_with(|| VecDeque::with_capacity(self.max_history));

        history.push_back(time);
        while history.len() > self.max_history {
            history.pop_front();
        }
    }

    /// Compute raw STDP value for spike timing difference
    fn stdp_function(&self, delta_t: f64) -> f64 {
        if delta_t > 0.0 {
            // Pre-before-post: LTP
            self.config.a_plus * (-delta_t / self.config.tau_plus).exp()
        } else if delta_t < 0.0 {
            // Post-before-pre: LTD
            -self.config.a_minus * (delta_t / self.config.tau_minus).exp()
        } else {
            0.0
        }
    }

    /// Compute hyperbolic locality factor
    fn locality_factor(&self, hyperbolic_distance: f64) -> f64 {
        (-hyperbolic_distance / self.config.locality_scale).exp()
    }

    /// Compute soft bound factor for weight-dependent plasticity
    fn soft_bound_factor(&self, current_weight: f64, delta_w: f64) -> f64 {
        if !self.config.soft_bounds {
            return 1.0;
        }

        let w_norm = (current_weight - self.config.w_min) /
                     (self.config.w_max - self.config.w_min);

        if delta_w > 0.0 {
            // LTP: stronger when weight is low
            1.0 - w_norm
        } else {
            // LTD: stronger when weight is high
            w_norm
        }
    }

    /// Process spike pair for STDP learning
    pub fn process_spike_pair(
        &mut self,
        pre_id: u32,
        post_id: u32,
        pre_time: f64,
        post_time: f64,
        current_weight: f64,
        hyperbolic_distance: f64,
    ) -> f64 {
        let delta_t = post_time - pre_time;

        // Compute raw STDP
        let stdp_raw = self.stdp_function(delta_t);

        // Apply hyperbolic locality modulation
        let locality = self.locality_factor(hyperbolic_distance);

        // Apply soft bounds
        let soft_bound = self.soft_bound_factor(current_weight, stdp_raw);

        // Combined STDP value
        let stdp_value = stdp_raw * locality * soft_bound;

        // Update statistics
        if stdp_value > 0.0 {
            self.stats.ltp_events += 1;
        } else if stdp_value < 0.0 {
            self.stats.ltd_events += 1;
        }

        if self.config.three_factor {
            // Three-factor learning: accumulate eligibility trace
            self.eligibility.accumulate(pre_id, post_id, stdp_value, post_time);
            self.stats.active_traces = self.eligibility.active_count();

            // Weight update happens when modulation signal is applied
            0.0
        } else {
            // Standard STDP: immediate weight update
            let delta_w = self.config.learning_rate * stdp_value;
            self.accumulate_weight_update(pre_id, post_id, delta_w);
            delta_w
        }
    }

    /// Process all spike pairs from pre-synaptic neuron spike
    pub fn on_pre_spike(
        &mut self,
        pre_id: u32,
        pre_time: f64,
        post_neurons: &[(u32, f64, f64)], // (post_id, weight, distance)
    ) {
        self.record_spike(pre_id, pre_time);

        let tau_window = self.config.tau_plus * 5.0;

        for &(post_id, weight, distance) in post_neurons {
            // Collect spike times to avoid borrow conflict
            let post_times: Vec<f64> = self.spike_history
                .get(&post_id)
                .map(|h| h.iter().copied().collect())
                .unwrap_or_default();

            for post_time in post_times {
                let delta_t = post_time - pre_time;
                // Only consider recent spikes within STDP window
                if delta_t.abs() < tau_window {
                    self.process_spike_pair(
                        pre_id, post_id, pre_time, post_time, weight, distance
                    );
                }
            }
        }
    }

    /// Process all spike pairs from post-synaptic neuron spike
    pub fn on_post_spike(
        &mut self,
        post_id: u32,
        post_time: f64,
        pre_neurons: &[(u32, f64, f64)], // (pre_id, weight, distance)
    ) {
        self.record_spike(post_id, post_time);

        let tau_window = self.config.tau_plus * 5.0;

        for &(pre_id, weight, distance) in pre_neurons {
            // Collect spike times to avoid borrow conflict
            let pre_times: Vec<f64> = self.spike_history
                .get(&pre_id)
                .map(|h| h.iter().copied().collect())
                .unwrap_or_default();

            for pre_time in pre_times {
                let delta_t = post_time - pre_time;
                // Only consider recent spikes within STDP window
                if delta_t.abs() < tau_window {
                    self.process_spike_pair(
                        pre_id, post_id, pre_time, post_time, weight, distance
                    );
                }
            }
        }
    }

    /// Set modulation signal (reward/dopamine)
    pub fn set_modulation(&mut self, signal: f64) {
        self.modulation_signal = signal;
    }

    /// Apply modulation and compute final weight updates (three-factor)
    pub fn apply_modulation(&mut self, current_time: f64) -> Vec<((u32, u32), f64)> {
        if !self.config.three_factor {
            return Vec::new();
        }

        // Decay eligibility traces
        self.eligibility.decay(current_time);

        // Apply modulation signal to get weight updates
        let updates = self.eligibility.apply_modulation(
            self.modulation_signal,
            self.config.learning_rate
        );

        self.stats.modulation_events += 1;

        // Accumulate updates
        for &(synapse, delta_w) in &updates {
            self.accumulate_weight_update(synapse.0, synapse.1, delta_w);
        }

        updates
    }

    /// Accumulate weight update for a synapse
    fn accumulate_weight_update(&mut self, pre_id: u32, post_id: u32, delta_w: f64) {
        let key = (pre_id, post_id);
        *self.weight_updates.entry(key).or_insert(0.0) += delta_w;

        // Update running average
        self.stats.avg_weight_change =
            0.99 * self.stats.avg_weight_change + 0.01 * delta_w.abs();
    }

    /// Flush accumulated weight updates and return them
    pub fn flush_updates(&mut self) -> Vec<((u32, u32), f64)> {
        let updates: Vec<_> = self.weight_updates.drain().collect();
        updates
    }

    /// Apply weight update with bounds checking
    pub fn apply_weight_update(&self, current_weight: f64, delta_w: f64) -> f64 {
        (current_weight + delta_w).clamp(self.config.w_min, self.config.w_max)
    }

    /// Get statistics
    pub fn stats(&self) -> &STDPStats {
        &self.stats
    }

    /// Get configuration
    pub fn config(&self) -> &STDPConfig {
        &self.config
    }

    /// Reset learner state
    pub fn reset(&mut self) {
        self.eligibility.clear();
        self.spike_history.clear();
        self.weight_updates.clear();
        self.modulation_signal = 1.0;
        self.stats = STDPStats::default();
    }
}

// ============================================================================
// Learning Rate Scheduler
// ============================================================================

/// Learning rate scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Constant learning rate
    Constant,
    /// Exponential decay: η(t) = η₀ × exp(-t/τ)
    Exponential { tau: f64 },
    /// Step decay: η(t) = η₀ × γ^floor(t/step)
    Step { step: f64, gamma: f64 },
    /// Cosine annealing: η(t) = η_min + 0.5(η₀ - η_min)(1 + cos(πt/T))
    Cosine { t_max: f64, eta_min: f64 },
    /// Performance-based: adjust based on reward signal
    Adaptive { increase_factor: f64, decrease_factor: f64 },
}

/// Learning rate scheduler
#[derive(Debug, Clone)]
pub struct LearningRateScheduler {
    /// Initial learning rate
    initial_rate: f64,
    /// Current learning rate
    current_rate: f64,
    /// Schedule type
    schedule: LearningRateSchedule,
    /// Current time/step
    current_step: f64,
    /// Performance history for adaptive scheduling
    performance_history: VecDeque<f64>,
}

impl LearningRateScheduler {
    /// Create new scheduler
    pub fn new(initial_rate: f64, schedule: LearningRateSchedule) -> Self {
        Self {
            initial_rate,
            current_rate: initial_rate,
            schedule,
            current_step: 0.0,
            performance_history: VecDeque::with_capacity(100),
        }
    }

    /// Update scheduler and return current learning rate
    pub fn step(&mut self, time: f64, performance: Option<f64>) -> f64 {
        self.current_step = time;

        if let Some(perf) = performance {
            self.performance_history.push_back(perf);
            if self.performance_history.len() > 100 {
                self.performance_history.pop_front();
            }
        }

        self.current_rate = match &self.schedule {
            LearningRateSchedule::Constant => self.initial_rate,

            LearningRateSchedule::Exponential { tau } => {
                self.initial_rate * (-time / tau).exp()
            }

            LearningRateSchedule::Step { step, gamma } => {
                let n = (time / step).floor();
                self.initial_rate * gamma.powf(n)
            }

            LearningRateSchedule::Cosine { t_max, eta_min } => {
                let t_norm = (time / t_max).min(1.0);
                eta_min + 0.5 * (self.initial_rate - eta_min) *
                    (1.0 + (std::f64::consts::PI * t_norm).cos())
            }

            LearningRateSchedule::Adaptive { increase_factor, decrease_factor } => {
                if self.performance_history.len() >= 2 {
                    let recent: f64 = self.performance_history.iter()
                        .rev().take(10).sum::<f64>() / 10.0;
                    let older: f64 = self.performance_history.iter()
                        .rev().skip(10).take(10).sum::<f64>() / 10.0;

                    if recent > older * 1.01 {
                        // Performance improving: increase rate
                        (self.current_rate * increase_factor).min(self.initial_rate * 2.0)
                    } else if recent < older * 0.99 {
                        // Performance degrading: decrease rate
                        (self.current_rate * decrease_factor).max(self.initial_rate * 0.01)
                    } else {
                        self.current_rate
                    }
                } else {
                    self.initial_rate
                }
            }
        };

        self.current_rate
    }

    /// Get current learning rate
    pub fn get_rate(&self) -> f64 {
        self.current_rate
    }

    /// Reset scheduler
    pub fn reset(&mut self) {
        self.current_rate = self.initial_rate;
        self.current_step = 0.0;
        self.performance_history.clear();
    }
}

// ============================================================================
// Homeostatic Plasticity
// ============================================================================

/// Homeostatic plasticity for network stability
#[derive(Debug, Clone)]
pub struct HomeostaticPlasticity {
    /// Target firing rate (Hz)
    target_rate: f64,
    /// Time constant for rate estimation
    tau_rate: f64,
    /// Scaling factor for synaptic adjustments
    scaling_factor: f64,
    /// Per-neuron firing rate estimates
    rate_estimates: HashMap<u32, f64>,
    /// Per-neuron spike counts in current window
    spike_counts: HashMap<u32, u32>,
    /// Window duration for rate estimation
    window_duration: f64,
    /// Last window reset time
    last_reset: f64,
}

impl HomeostaticPlasticity {
    /// Create new homeostatic plasticity controller
    pub fn new(target_rate: f64, tau_rate: f64, scaling_factor: f64) -> Self {
        Self {
            target_rate,
            tau_rate,
            scaling_factor,
            rate_estimates: HashMap::new(),
            spike_counts: HashMap::new(),
            window_duration: 1000.0, // 1 second window
            last_reset: 0.0,
        }
    }

    /// Record a spike
    pub fn record_spike(&mut self, neuron_id: u32) {
        *self.spike_counts.entry(neuron_id).or_insert(0) += 1;
    }

    /// Update rate estimates and compute scaling factors
    pub fn update(&mut self, current_time: f64) -> HashMap<u32, f64> {
        let mut scaling_factors = HashMap::new();

        let dt = current_time - self.last_reset;
        if dt < self.window_duration {
            return scaling_factors;
        }

        // Update rate estimates
        for (&neuron_id, &count) in &self.spike_counts {
            let measured_rate = (count as f64 / dt) * 1000.0; // Convert to Hz

            let estimate = self.rate_estimates.entry(neuron_id).or_insert(self.target_rate);
            *estimate = *estimate * (-dt / self.tau_rate).exp() +
                       measured_rate * (1.0 - (-dt / self.tau_rate).exp());

            // Compute scaling factor to move toward target rate
            let ratio = self.target_rate / (*estimate).max(0.1);
            let scale = 1.0 + self.scaling_factor * (ratio - 1.0);
            scaling_factors.insert(neuron_id, scale.clamp(0.5, 2.0));
        }

        // Reset spike counts
        self.spike_counts.clear();
        self.last_reset = current_time;

        scaling_factors
    }

    /// Get current rate estimate for a neuron
    pub fn get_rate_estimate(&self, neuron_id: u32) -> f64 {
        self.rate_estimates.get(&neuron_id).copied().unwrap_or(self.target_rate)
    }
}

// ============================================================================
// Weight Normalization
// ============================================================================

/// Synaptic weight normalization strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NormalizationType {
    /// No normalization
    None,
    /// L1 normalization: Σ|w| = target
    L1 { target: f64 },
    /// L2 normalization: √(Σw²) = target
    L2 { target: f64 },
    /// Max normalization: max(|w|) = target
    Max { target: f64 },
    /// Multiplicative: scale all weights to maintain total
    Multiplicative { target_total: f64 },
}

/// Weight normalizer
#[derive(Debug, Clone)]
pub struct WeightNormalizer {
    /// Normalization type
    norm_type: NormalizationType,
    /// Apply normalization per neuron or globally
    #[allow(dead_code)]
    per_neuron: bool,
}

impl WeightNormalizer {
    /// Create new weight normalizer
    pub fn new(norm_type: NormalizationType, per_neuron: bool) -> Self {
        Self { norm_type, per_neuron }
    }

    /// Normalize a set of weights
    pub fn normalize(&self, weights: &mut [f64]) {
        if weights.is_empty() {
            return;
        }

        match &self.norm_type {
            NormalizationType::None => {}

            NormalizationType::L1 { target } => {
                let sum: f64 = weights.iter().map(|w| w.abs()).sum();
                if sum > 1e-10 {
                    let scale = target / sum;
                    for w in weights.iter_mut() {
                        *w *= scale;
                    }
                }
            }

            NormalizationType::L2 { target } => {
                let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
                if sum_sq > 1e-10 {
                    let scale = target / sum_sq.sqrt();
                    for w in weights.iter_mut() {
                        *w *= scale;
                    }
                }
            }

            NormalizationType::Max { target } => {
                let max_w = weights.iter().map(|w| w.abs()).fold(0.0, f64::max);
                if max_w > 1e-10 {
                    let scale = target / max_w;
                    for w in weights.iter_mut() {
                        *w *= scale;
                    }
                }
            }

            NormalizationType::Multiplicative { target_total } => {
                let sum: f64 = weights.iter().sum();
                if sum > 1e-10 {
                    let scale = target_total / sum;
                    for w in weights.iter_mut() {
                        *w *= scale;
                    }
                }
            }
        }
    }
}

// ============================================================================
// Integrated Chunk-Aware STDP
// ============================================================================

/// STDP integration for chunk processor
#[derive(Debug, Clone)]
pub struct ChunkAwareSTDP {
    /// Base STDP learner
    stdp: STDPLearner,
    /// Learning rate scheduler
    scheduler: LearningRateScheduler,
    /// Homeostatic controller
    homeostasis: HomeostaticPlasticity,
    /// Weight normalizer (for future batch normalization)
    #[allow(dead_code)]
    normalizer: WeightNormalizer,
    /// Current chunk level (affects learning rate)
    current_level: usize,
    /// Level-dependent learning rate multipliers
    #[allow(dead_code)]
    level_multipliers: Vec<f64>,
}

impl ChunkAwareSTDP {
    /// Create new chunk-aware STDP learner
    pub fn new(config: STDPConfig) -> Self {
        let base_rate = config.learning_rate;
        Self {
            stdp: STDPLearner::new(config),
            scheduler: LearningRateScheduler::new(
                base_rate,
                LearningRateSchedule::Cosine { t_max: 100000.0, eta_min: base_rate * 0.01 }
            ),
            homeostasis: HomeostaticPlasticity::new(10.0, 10000.0, 0.1),
            normalizer: WeightNormalizer::new(
                NormalizationType::L2 { target: 1.0 },
                true
            ),
            current_level: 0,
            level_multipliers: vec![1.0, 0.5, 0.25, 0.1], // Slower learning at higher levels
        }
    }

    /// Set current processing level
    pub fn set_level(&mut self, level: usize) {
        self.current_level = level;
    }

    /// Get level-adjusted learning rate
    #[allow(dead_code)]
    fn get_adjusted_rate(&self) -> f64 {
        let base = self.scheduler.get_rate();
        let multiplier = self.level_multipliers
            .get(self.current_level)
            .copied()
            .unwrap_or(0.1);
        base * multiplier
    }

    /// Process spike event
    pub fn on_spike(
        &mut self,
        neuron_id: u32,
        time: f64,
        is_pre: bool,
        connections: &[(u32, f64, f64)],
    ) {
        self.homeostasis.record_spike(neuron_id);

        if is_pre {
            self.stdp.on_pre_spike(neuron_id, time, connections);
        } else {
            self.stdp.on_post_spike(neuron_id, time, connections);
        }
    }

    /// Apply chunk completion reward/modulation
    pub fn on_chunk_complete(&mut self, chunk_quality: f64, time: f64) {
        // Set modulation based on chunk quality (compression efficiency, information preserved)
        self.stdp.set_modulation(chunk_quality);
        self.stdp.apply_modulation(time);

        // Update scheduler with performance
        self.scheduler.step(time, Some(chunk_quality));
    }

    /// Get and apply weight updates
    pub fn flush_updates(&mut self, current_time: f64) -> Vec<((u32, u32), f64)> {
        // Get homeostatic scaling factors
        let scaling = self.homeostasis.update(current_time);

        // Get STDP updates
        let mut updates = self.stdp.flush_updates();

        // Apply homeostatic scaling
        for ((pre_id, _post_id), delta_w) in updates.iter_mut() {
            if let Some(&scale) = scaling.get(pre_id) {
                *delta_w *= scale;
            }
        }

        updates
    }

    /// Get statistics
    pub fn stats(&self) -> &STDPStats {
        self.stdp.stats()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdp_function() {
        let config = STDPConfig::default();
        let learner = STDPLearner::new(config);

        // LTP: pre before post
        let ltp = learner.stdp_function(10.0);
        assert!(ltp > 0.0);

        // LTD: post before pre
        let ltd = learner.stdp_function(-10.0);
        assert!(ltd < 0.0);

        // Zero at delta_t = 0
        let zero = learner.stdp_function(0.0);
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn test_eligibility_trace() {
        let mut trace = EligibilityTrace::new(100.0);

        // Accumulate
        trace.accumulate(0, 1, 0.5, 0.0);
        assert!((trace.get(0, 1) - 0.5).abs() < 0.01);

        // Decay
        trace.decay(50.0);
        let decayed = trace.get(0, 1);
        assert!(decayed < 0.5);
        assert!(decayed > 0.0);
    }

    #[test]
    fn test_learning_rate_scheduler() {
        let mut scheduler = LearningRateScheduler::new(
            0.01,
            LearningRateSchedule::Exponential { tau: 1000.0 }
        );

        let rate_0 = scheduler.step(0.0, None);
        let rate_1000 = scheduler.step(1000.0, None);

        assert!((rate_0 - 0.01).abs() < 0.001);
        assert!(rate_1000 < rate_0);
    }

    #[test]
    fn test_homeostatic_plasticity() {
        let mut homeo = HomeostaticPlasticity::new(10.0, 10000.0, 0.1);

        // Record some spikes
        for _ in 0..20 {
            homeo.record_spike(0);
        }

        // Update should produce scaling factors
        let factors = homeo.update(1000.0);
        assert!(factors.contains_key(&0));
    }

    #[test]
    fn test_weight_normalization() {
        let normalizer = WeightNormalizer::new(
            NormalizationType::L2 { target: 1.0 },
            false
        );

        let mut weights = vec![0.5, 0.5, 0.5, 0.5];
        normalizer.normalize(&mut weights);

        let l2_norm: f64 = weights.iter().map(|w| w * w).sum::<f64>().sqrt();
        assert!((l2_norm - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_three_factor_learning() {
        let mut config = STDPConfig::default();
        config.three_factor = true;

        let mut learner = STDPLearner::new(config);

        // Process spike pair - should accumulate eligibility
        learner.process_spike_pair(0, 1, 0.0, 10.0, 0.5, 1.0);

        // Apply modulation
        learner.set_modulation(1.0);
        let updates = learner.apply_modulation(10.0);

        // Should have weight update
        assert!(!updates.is_empty() || learner.eligibility.active_count() > 0);
    }
}
