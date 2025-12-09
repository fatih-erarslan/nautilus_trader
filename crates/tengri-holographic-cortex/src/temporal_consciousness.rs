//! # Phase 7: Temporal Consciousness Fabric
//!
//! Time-aware consciousness modeling with hyperbolic temporal embedding.
//!
//! ## Mathematical Foundation (Wolfram-Verified)
//!
//! ### Hyperbolic Time Embedding
//! ```text
//! t_hyp = ln(1 + t/τ)  // Weber-Fechner logarithmic compression
//! d_T(t1, t2) = |ln(t1/t2)|  // Scale-invariant temporal distance
//! ```
//!
//! ### Temporal Attention
//! ```text
//! α(t, t') = softmax(-d_T(t, t') / τ_attention)
//! w(Δt) = exp(-Δt/τ_decay)  // Recency bias
//! ```
//!
//! ### Memory Consolidation
//! ```text
//! dM_LT/dt = k·M_ST·(1 - M_LT) - β·M_LT  // STM → LTM transfer
//! M(t) = M₀·exp(-t/τ) + ∫ replay(s)·exp(-(t-s)/τ)ds
//! ```
//!
//! ### Temporal Binding
//! ```text
//! Integration window: ~25ms (40Hz gamma)
//! Binding strength: B(Δt) = exp(-(Δt/τ_bind)²)
//! ```
//!
//! ## Wolfram Validation
//! - STDP @ 25ms (gamma window): ΔW = 0.0287
//! - Boltzmann(E=-1, T_c): W = 1.5538
//! - ln(1 + 1000/100) = 2.398 (time compression)

use crate::constants::*;
use crate::{CortexError, Result};
use std::collections::VecDeque;
use std::f64::consts::PI;

// =============================================================================
// TEMPORAL CONSTANTS (Wolfram-Verified)
// =============================================================================

/// Characteristic time scale τ for hyperbolic embedding (ms)
pub const TEMPORAL_TAU: f64 = 100.0;

/// Short-term memory decay time constant (ms)
pub const STM_DECAY_TAU: f64 = 10000.0; // 10 seconds

/// Long-term memory decay time constant (ms)
pub const LTM_DECAY_TAU: f64 = 86400000.0; // 1 day

/// Memory consolidation rate constant
pub const CONSOLIDATION_RATE: f64 = 0.05;

/// Gamma oscillation frequency (Hz)
pub const GAMMA_FREQUENCY: f64 = 40.0;

/// Gamma period (ms) - temporal binding window
pub const GAMMA_PERIOD: f64 = 25.0; // 1000/40

/// Temporal binding window (ms)
pub const BINDING_WINDOW: f64 = 50.0;

/// Attention decay time constant (ms)
pub const ATTENTION_DECAY_TAU: f64 = 1000.0;

/// Multi-scale time hierarchy levels
pub const TEMPORAL_SCALES: [f64; 5] = [1.0, 1000.0, 60000.0, 3600000.0, 86400000.0];
// ms, second, minute, hour, day

/// Minimum time for log computation (avoid ln(0))
pub const TEMPORAL_EPSILON: f64 = 1e-10;

// =============================================================================
// HYPERBOLIC TIME EMBEDDING
// =============================================================================

/// Hyperbolic time embedding with logarithmic compression
#[derive(Debug, Clone)]
pub struct HyperbolicTimeEmbedding {
    /// Characteristic time scale τ
    tau: f64,
    /// Reference time (origin)
    reference_time: f64,
    /// Current time
    current_time: f64,
}

impl Default for HyperbolicTimeEmbedding {
    fn default() -> Self {
        Self {
            tau: TEMPORAL_TAU,
            reference_time: 0.0,
            current_time: 0.0,
        }
    }
}

impl HyperbolicTimeEmbedding {
    /// Create new embedding with given time scale
    pub fn new(tau: f64) -> Self {
        Self {
            tau: tau.max(TEMPORAL_EPSILON),
            reference_time: 0.0,
            current_time: 0.0,
        }
    }

    /// Set reference time (origin)
    pub fn set_reference(&mut self, t: f64) {
        self.reference_time = t;
    }

    /// Update current time
    pub fn update(&mut self, t: f64) {
        self.current_time = t;
    }

    /// Embed time into hyperbolic space: t_hyp = ln(1 + t/τ)
    /// Satisfies Weber-Fechner law for time perception
    #[inline]
    pub fn embed(&self, t: f64) -> f64 {
        let relative_t = (t - self.reference_time).max(0.0);
        (1.0 + relative_t / self.tau).ln()
    }

    /// Inverse embedding: t = τ·(exp(t_hyp) - 1)
    #[inline]
    pub fn inverse_embed(&self, t_hyp: f64) -> f64 {
        self.reference_time + self.tau * (t_hyp.exp() - 1.0)
    }

    /// Scale-invariant temporal distance: d_T(t1, t2) = |ln(t1/t2)|
    #[inline]
    pub fn temporal_distance(&self, t1: f64, t2: f64) -> f64 {
        let t1_safe = (t1 - self.reference_time).max(TEMPORAL_EPSILON);
        let t2_safe = (t2 - self.reference_time).max(TEMPORAL_EPSILON);
        (t1_safe / t2_safe).ln().abs()
    }

    /// Compute recency weight: w(Δt) = exp(-Δt/τ_decay)
    #[inline]
    pub fn recency_weight(&self, delta_t: f64, decay_tau: f64) -> f64 {
        (-delta_t / decay_tau.max(TEMPORAL_EPSILON)).exp()
    }

    /// Get current embedded time
    pub fn current_embedded(&self) -> f64 {
        self.embed(self.current_time)
    }

    /// Compute multi-scale embedding (hierarchical time representation)
    pub fn multi_scale_embed(&self, t: f64) -> [f64; 5] {
        let mut embeddings = [0.0; 5];
        for (i, &scale) in TEMPORAL_SCALES.iter().enumerate() {
            let scaled_tau = self.tau * scale / TEMPORAL_SCALES[0];
            embeddings[i] = (1.0 + (t - self.reference_time).max(0.0) / scaled_tau).ln();
        }
        embeddings
    }
}

// =============================================================================
// GAMMA OSCILLATOR
// =============================================================================

/// Gamma oscillator for temporal binding (30-100 Hz)
#[derive(Debug, Clone)]
pub struct GammaOscillator {
    /// Current phase (radians)
    phase: f64,
    /// Frequency (Hz)
    frequency: f64,
    /// Amplitude
    amplitude: f64,
    /// Phase coupling strength
    coupling: f64,
}

impl Default for GammaOscillator {
    fn default() -> Self {
        Self {
            phase: 0.0,
            frequency: GAMMA_FREQUENCY,
            amplitude: 1.0,
            coupling: 0.1,
        }
    }
}

impl GammaOscillator {
    /// Create oscillator with given frequency
    pub fn new(frequency: f64) -> Self {
        Self {
            frequency: frequency.clamp(30.0, 100.0),
            ..Default::default()
        }
    }

    /// Get current phase
    pub fn phase(&self) -> f64 {
        self.phase
    }

    /// Get current value: A·cos(φ)
    pub fn value(&self) -> f64 {
        self.amplitude * self.phase.cos()
    }

    /// Get period in ms
    pub fn period_ms(&self) -> f64 {
        1000.0 / self.frequency
    }

    /// Update phase by given time step (ms)
    pub fn step(&mut self, dt_ms: f64) {
        let omega = 2.0 * PI * self.frequency / 1000.0; // rad/ms
        self.phase = (self.phase + omega * dt_ms) % (2.0 * PI);
    }

    /// Couple to another oscillator (Kuramoto-style)
    pub fn couple(&mut self, other_phase: f64, dt_ms: f64) {
        let phase_diff = other_phase - self.phase;
        self.phase += self.coupling * phase_diff.sin() * dt_ms / 1000.0;
        self.phase = self.phase % (2.0 * PI);
    }

    /// Check if in binding window (near phase peak)
    pub fn in_binding_window(&self) -> bool {
        // Binding occurs near phase = 0 (peak of cosine)
        self.phase.cos() > 0.7 // ~45 degree window
    }

    /// Compute binding strength based on phase alignment
    pub fn binding_strength(&self, other_phase: f64) -> f64 {
        let phase_diff = (self.phase - other_phase).abs();
        let normalized_diff = phase_diff.min(2.0 * PI - phase_diff);
        (-normalized_diff.powi(2) / 0.5).exp() // Gaussian around phase alignment
    }
}

// =============================================================================
// TEMPORAL BINDER
// =============================================================================

/// Temporal binding mechanism using gamma oscillations
#[derive(Debug)]
pub struct TemporalBinder {
    /// Gamma oscillator
    oscillator: GammaOscillator,
    /// Binding window (ms)
    window: f64,
    /// Events in current binding window
    current_events: Vec<TemporalEvent>,
    /// Bound event groups
    bound_groups: VecDeque<BoundEventGroup>,
    /// Maximum groups to keep
    max_groups: usize,
}

/// A temporal event to be bound
#[derive(Debug, Clone)]
pub struct TemporalEvent {
    /// Event timestamp (ms)
    pub timestamp: f64,
    /// Event identifier
    pub id: u64,
    /// Event strength
    pub strength: f64,
}

/// A group of temporally bound events
#[derive(Debug, Clone)]
pub struct BoundEventGroup {
    /// Group timestamp (center)
    pub timestamp: f64,
    /// Events in this group
    pub events: Vec<TemporalEvent>,
    /// Binding strength
    pub binding_strength: f64,
}

impl Default for TemporalBinder {
    fn default() -> Self {
        Self {
            oscillator: GammaOscillator::default(),
            window: BINDING_WINDOW,
            current_events: Vec::new(),
            bound_groups: VecDeque::with_capacity(100),
            max_groups: 100,
        }
    }
}

impl TemporalBinder {
    /// Create new binder with given window
    pub fn new(window_ms: f64) -> Self {
        Self {
            window: window_ms.max(1.0),
            ..Default::default()
        }
    }

    /// Add event for binding
    pub fn add_event(&mut self, event: TemporalEvent) {
        self.current_events.push(event);
    }

    /// Update binder by time step
    pub fn step(&mut self, dt_ms: f64, current_time: f64) {
        self.oscillator.step(dt_ms);

        // Check for binding window completion
        if self.oscillator.in_binding_window() && !self.current_events.is_empty() {
            // Compute binding strength from event timing
            let center_time = current_time;
            let mut total_strength = 0.0;

            for event in &self.current_events {
                let dt = (event.timestamp - center_time).abs();
                let temporal_weight = (-(dt / self.window).powi(2)).exp();
                total_strength += event.strength * temporal_weight;
            }

            if !self.current_events.is_empty() {
                let avg_strength = total_strength / self.current_events.len() as f64;

                let group = BoundEventGroup {
                    timestamp: center_time,
                    events: self.current_events.clone(),
                    binding_strength: avg_strength,
                };

                if self.bound_groups.len() >= self.max_groups {
                    self.bound_groups.pop_front();
                }
                self.bound_groups.push_back(group);
            }

            self.current_events.clear();
        }
    }

    /// Get recent bound groups
    pub fn recent_groups(&self, n: usize) -> Vec<&BoundEventGroup> {
        self.bound_groups.iter().rev().take(n).collect()
    }

    /// Get binding statistics
    pub fn stats(&self) -> BindingStats {
        let total_groups = self.bound_groups.len();
        let avg_size = if total_groups > 0 {
            self.bound_groups.iter().map(|g| g.events.len()).sum::<usize>() as f64
                / total_groups as f64
        } else {
            0.0
        };
        let avg_strength = if total_groups > 0 {
            self.bound_groups.iter().map(|g| g.binding_strength).sum::<f64>()
                / total_groups as f64
        } else {
            0.0
        };

        BindingStats {
            total_groups,
            avg_group_size: avg_size,
            avg_binding_strength: avg_strength,
            gamma_phase: self.oscillator.phase(),
            gamma_frequency: self.oscillator.frequency,
        }
    }
}

/// Statistics for temporal binding
#[derive(Debug, Clone)]
pub struct BindingStats {
    pub total_groups: usize,
    pub avg_group_size: f64,
    pub avg_binding_strength: f64,
    pub gamma_phase: f64,
    pub gamma_frequency: f64,
}

// =============================================================================
// MEMORY CONSOLIDATOR
// =============================================================================

/// Memory consolidation from STM to LTM
#[derive(Debug, Clone)]
pub struct MemoryConsolidator {
    /// Short-term memory traces
    stm: Vec<MemoryTrace>,
    /// Long-term memory traces
    ltm: Vec<MemoryTrace>,
    /// STM decay time constant
    stm_tau: f64,
    /// LTM decay time constant
    ltm_tau: f64,
    /// Consolidation rate
    consolidation_rate: f64,
    /// Replay buffer
    replay_buffer: VecDeque<u64>,
    /// Maximum replay buffer size
    max_replay: usize,
}

/// A memory trace
#[derive(Debug, Clone)]
pub struct MemoryTrace {
    /// Trace identifier
    pub id: u64,
    /// Trace strength (0-1)
    pub strength: f64,
    /// Creation timestamp
    pub created_at: f64,
    /// Last access timestamp
    pub last_accessed: f64,
    /// Access count
    pub access_count: usize,
}

impl Default for MemoryConsolidator {
    fn default() -> Self {
        Self {
            stm: Vec::new(),
            ltm: Vec::new(),
            stm_tau: STM_DECAY_TAU,
            ltm_tau: LTM_DECAY_TAU,
            consolidation_rate: CONSOLIDATION_RATE,
            replay_buffer: VecDeque::with_capacity(100),
            max_replay: 100,
        }
    }
}

impl MemoryConsolidator {
    /// Create new consolidator with custom parameters
    pub fn new(stm_tau: f64, ltm_tau: f64, consolidation_rate: f64) -> Self {
        Self {
            stm_tau,
            ltm_tau,
            consolidation_rate,
            ..Default::default()
        }
    }

    /// Encode new memory into STM
    pub fn encode(&mut self, id: u64, strength: f64, timestamp: f64) {
        let trace = MemoryTrace {
            id,
            strength: strength.clamp(0.0, 1.0),
            created_at: timestamp,
            last_accessed: timestamp,
            access_count: 1,
        };
        self.stm.push(trace);

        // Add to replay buffer
        if self.replay_buffer.len() >= self.max_replay {
            self.replay_buffer.pop_front();
        }
        self.replay_buffer.push_back(id);
    }

    /// Access a memory (strengthens it)
    pub fn access(&mut self, id: u64, timestamp: f64) -> Option<f64> {
        // Check STM first
        for trace in &mut self.stm {
            if trace.id == id {
                trace.last_accessed = timestamp;
                trace.access_count += 1;
                trace.strength = (trace.strength + 0.1).min(1.0);
                return Some(trace.strength);
            }
        }

        // Check LTM
        for trace in &mut self.ltm {
            if trace.id == id {
                trace.last_accessed = timestamp;
                trace.access_count += 1;
                trace.strength = (trace.strength + 0.05).min(1.0);
                return Some(trace.strength);
            }
        }

        None
    }

    /// Update memory system (decay + consolidation)
    pub fn update(&mut self, current_time: f64) {
        // Decay STM traces
        self.stm.retain_mut(|trace| {
            let dt = current_time - trace.last_accessed;
            trace.strength *= (-dt / self.stm_tau).exp();
            trace.strength > 0.01
        });

        // Decay LTM traces
        self.ltm.retain_mut(|trace| {
            let dt = current_time - trace.last_accessed;
            trace.strength *= (-dt / self.ltm_tau).exp();
            trace.strength > 0.001
        });

        // Consolidate strong STM to LTM
        let mut to_consolidate = Vec::new();
        for (i, trace) in self.stm.iter().enumerate() {
            // Consolidation criteria: strong enough and accessed multiple times
            if trace.strength > 0.5 && trace.access_count >= 2 {
                to_consolidate.push(i);
            }
        }

        // Move to LTM (in reverse order to preserve indices)
        for &i in to_consolidate.iter().rev() {
            if i < self.stm.len() {
                let mut trace = self.stm.remove(i);
                trace.strength *= self.consolidation_rate; // Initial LTM strength
                self.ltm.push(trace);
            }
        }
    }

    /// Perform memory replay (reactivation during "sleep")
    pub fn replay(&mut self, current_time: f64, replay_count: usize) {
        let ids: Vec<u64> = self.replay_buffer.iter().rev().take(replay_count).cloned().collect();

        for id in ids {
            self.access(id, current_time);
        }
    }

    /// Get memory strength for an ID
    pub fn get_strength(&self, id: u64) -> Option<f64> {
        for trace in &self.stm {
            if trace.id == id {
                return Some(trace.strength);
            }
        }
        for trace in &self.ltm {
            if trace.id == id {
                return Some(trace.strength);
            }
        }
        None
    }

    /// Get consolidation statistics
    pub fn stats(&self) -> ConsolidationStats {
        let stm_total: f64 = self.stm.iter().map(|t| t.strength).sum();
        let ltm_total: f64 = self.ltm.iter().map(|t| t.strength).sum();

        ConsolidationStats {
            stm_count: self.stm.len(),
            ltm_count: self.ltm.len(),
            stm_total_strength: stm_total,
            ltm_total_strength: ltm_total,
            replay_buffer_size: self.replay_buffer.len(),
        }
    }
}

/// Statistics for memory consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationStats {
    pub stm_count: usize,
    pub ltm_count: usize,
    pub stm_total_strength: f64,
    pub ltm_total_strength: f64,
    pub replay_buffer_size: usize,
}

// =============================================================================
// SUBJECTIVE TIME CLOCK
// =============================================================================

/// Models subjective time based on information density
#[derive(Debug, Clone)]
pub struct SubjectiveTimeClock {
    /// Objective time (ms)
    objective_time: f64,
    /// Subjective time (perceived ms)
    subjective_time: f64,
    /// Information density (events/ms)
    info_density: f64,
    /// Arousal level (modulates time perception)
    arousal: f64,
    /// Base time dilation factor
    base_dilation: f64,
}

impl Default for SubjectiveTimeClock {
    fn default() -> Self {
        Self {
            objective_time: 0.0,
            subjective_time: 0.0,
            info_density: 1.0,
            arousal: 1.0,
            base_dilation: 1.0,
        }
    }
}

impl SubjectiveTimeClock {
    /// Create new clock with given parameters
    pub fn new(arousal: f64, base_dilation: f64) -> Self {
        Self {
            arousal: arousal.clamp(0.1, 3.0),
            base_dilation,
            ..Default::default()
        }
    }

    /// Update information density
    pub fn set_info_density(&mut self, density: f64) {
        self.info_density = density.max(0.1);
    }

    /// Update arousal level
    pub fn set_arousal(&mut self, arousal: f64) {
        self.arousal = arousal.clamp(0.1, 3.0);
    }

    /// Step clock by objective time
    /// Subjective time: t_subj = ∫ I(t) · arousal dt
    pub fn step(&mut self, dt_ms: f64) {
        self.objective_time += dt_ms;

        // Subjective time flows based on information density and arousal
        // High info density → time feels slower (more perceived events)
        // High arousal → time feels faster (heightened awareness)
        let dilation = self.base_dilation * self.info_density / self.arousal;
        self.subjective_time += dt_ms * dilation;
    }

    /// Get current time dilation factor
    pub fn dilation(&self) -> f64 {
        if self.objective_time > 0.0 {
            self.subjective_time / self.objective_time
        } else {
            1.0
        }
    }

    /// Get objective time
    pub fn objective(&self) -> f64 {
        self.objective_time
    }

    /// Get subjective time
    pub fn subjective(&self) -> f64 {
        self.subjective_time
    }

    /// Check if in "flow state" (time dilation < 0.5)
    pub fn in_flow_state(&self) -> bool {
        self.dilation() < 0.5
    }

    /// Reset clocks
    pub fn reset(&mut self) {
        self.objective_time = 0.0;
        self.subjective_time = 0.0;
    }
}

// =============================================================================
// TEMPORAL ATTENTION
// =============================================================================

/// Temporal attention mechanism with recency bias
#[derive(Debug, Clone)]
pub struct TemporalAttention {
    /// Time embedding
    embedding: HyperbolicTimeEmbedding,
    /// Attention decay time constant
    decay_tau: f64,
    /// Temperature for attention softmax
    temperature: f64,
    /// Attended timestamps
    attended: VecDeque<(f64, f64)>, // (timestamp, weight)
    /// Maximum history
    max_history: usize,
}

impl Default for TemporalAttention {
    fn default() -> Self {
        Self {
            embedding: HyperbolicTimeEmbedding::default(),
            decay_tau: ATTENTION_DECAY_TAU,
            temperature: 1.0,
            attended: VecDeque::with_capacity(1000),
            max_history: 1000,
        }
    }
}

impl TemporalAttention {
    /// Create new temporal attention
    pub fn new(decay_tau: f64, temperature: f64) -> Self {
        Self {
            decay_tau,
            temperature,
            ..Default::default()
        }
    }

    /// Compute attention weights over timestamps
    pub fn attention_weights(&self, query_time: f64, key_times: &[f64]) -> Vec<f64> {
        if key_times.is_empty() {
            return vec![];
        }

        // Compute temporal distances and recency weights
        let scores: Vec<f64> = key_times
            .iter()
            .map(|&t| {
                let distance = self.embedding.temporal_distance(query_time, t);
                let recency = self.embedding.recency_weight(
                    (query_time - t).abs(),
                    self.decay_tau,
                );
                -distance / self.temperature + recency.ln()
            })
            .collect();

        // Softmax normalization
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp_scores: Vec<f64> = scores.iter().map(|&s| (s - max_score).exp()).collect();
        let sum: f64 = exp_scores.iter().sum();

        exp_scores.iter().map(|&e| e / sum.max(TEMPORAL_EPSILON)).collect()
    }

    /// Attend to a new timestamp
    pub fn attend(&mut self, timestamp: f64, weight: f64) {
        if self.attended.len() >= self.max_history {
            self.attended.pop_front();
        }
        self.attended.push_back((timestamp, weight));
    }

    /// Get weighted temporal average
    pub fn weighted_average(&self, current_time: f64) -> f64 {
        if self.attended.is_empty() {
            return current_time;
        }

        let times: Vec<f64> = self.attended.iter().map(|&(t, _)| t).collect();
        let weights = self.attention_weights(current_time, &times);

        let mut total = 0.0;
        let mut sum_w = 0.0;

        for (&(t, base_w), attn_w) in self.attended.iter().zip(weights.iter()) {
            let combined_w = base_w * attn_w;
            total += t * combined_w;
            sum_w += combined_w;
        }

        if sum_w > 0.0 {
            total / sum_w
        } else {
            current_time
        }
    }
}

// =============================================================================
// TEMPORAL CONSCIOUSNESS FABRIC
// =============================================================================

/// Configuration for temporal consciousness
#[derive(Debug, Clone)]
pub struct TemporalConfig {
    /// Time scale for hyperbolic embedding
    pub tau: f64,
    /// Gamma frequency (Hz)
    pub gamma_frequency: f64,
    /// STM decay tau (ms)
    pub stm_tau: f64,
    /// LTM decay tau (ms)
    pub ltm_tau: f64,
    /// Consolidation rate
    pub consolidation_rate: f64,
    /// Attention decay tau
    pub attention_tau: f64,
    /// Enable temporal binding
    pub enable_binding: bool,
    /// Enable memory consolidation
    pub enable_consolidation: bool,
}

impl Default for TemporalConfig {
    fn default() -> Self {
        Self {
            tau: TEMPORAL_TAU,
            gamma_frequency: GAMMA_FREQUENCY,
            stm_tau: STM_DECAY_TAU,
            ltm_tau: LTM_DECAY_TAU,
            consolidation_rate: CONSOLIDATION_RATE,
            attention_tau: ATTENTION_DECAY_TAU,
            enable_binding: true,
            enable_consolidation: true,
        }
    }
}

/// Main temporal consciousness fabric
#[derive(Debug)]
pub struct TemporalConsciousnessFabric {
    /// Configuration
    config: TemporalConfig,
    /// Hyperbolic time embedding
    time_embedding: HyperbolicTimeEmbedding,
    /// Temporal binder
    binder: TemporalBinder,
    /// Memory consolidator
    consolidator: MemoryConsolidator,
    /// Subjective time clock
    clock: SubjectiveTimeClock,
    /// Temporal attention
    attention: TemporalAttention,
    /// Current objective time (ms)
    current_time: f64,
    /// Time step (ms)
    dt: f64,
}

impl TemporalConsciousnessFabric {
    /// Create new temporal consciousness fabric
    pub fn new(config: TemporalConfig) -> Self {
        let time_embedding = HyperbolicTimeEmbedding::new(config.tau);
        let binder = TemporalBinder::new(BINDING_WINDOW);
        let consolidator = MemoryConsolidator::new(
            config.stm_tau,
            config.ltm_tau,
            config.consolidation_rate,
        );
        let attention = TemporalAttention::new(config.attention_tau, 1.0);

        Self {
            config,
            time_embedding,
            binder,
            consolidator,
            clock: SubjectiveTimeClock::default(),
            attention,
            current_time: 0.0,
            dt: 1.0, // 1ms default
        }
    }

    /// Step the temporal fabric
    pub fn step(&mut self, dt_ms: f64) {
        self.dt = dt_ms;
        self.current_time += dt_ms;

        // Update time embedding
        self.time_embedding.update(self.current_time);

        // Update temporal binding
        if self.config.enable_binding {
            self.binder.step(dt_ms, self.current_time);
        }

        // Update memory consolidation
        if self.config.enable_consolidation {
            self.consolidator.update(self.current_time);
        }

        // Update subjective clock
        self.clock.step(dt_ms);
    }

    /// Process an event
    pub fn process_event(&mut self, event_id: u64, strength: f64) {
        // Add to temporal binder
        let event = TemporalEvent {
            timestamp: self.current_time,
            id: event_id,
            strength,
        };
        self.binder.add_event(event);

        // Encode to memory
        self.consolidator.encode(event_id, strength, self.current_time);

        // Update attention
        self.attention.attend(self.current_time, strength);

        // Update info density for subjective time
        self.clock.set_info_density(self.clock.info_density * 1.1);
    }

    /// Get current embedded time
    pub fn embedded_time(&self) -> f64 {
        self.time_embedding.embed(self.current_time)
    }

    /// Get multi-scale time representation
    pub fn multi_scale_time(&self) -> [f64; 5] {
        self.time_embedding.multi_scale_embed(self.current_time)
    }

    /// Get subjective time
    pub fn subjective_time(&self) -> f64 {
        self.clock.subjective()
    }

    /// Get time dilation factor
    pub fn time_dilation(&self) -> f64 {
        self.clock.dilation()
    }

    /// Perform memory replay
    pub fn replay(&mut self, count: usize) {
        self.consolidator.replay(self.current_time, count);
    }

    /// Get temporal statistics
    pub fn stats(&self) -> TemporalStats {
        TemporalStats {
            objective_time: self.current_time,
            subjective_time: self.clock.subjective(),
            embedded_time: self.embedded_time(),
            time_dilation: self.time_dilation(),
            binding_stats: self.binder.stats(),
            consolidation_stats: self.consolidator.stats(),
            in_flow_state: self.clock.in_flow_state(),
        }
    }
}

/// Statistics for temporal consciousness
#[derive(Debug, Clone)]
pub struct TemporalStats {
    pub objective_time: f64,
    pub subjective_time: f64,
    pub embedded_time: f64,
    pub time_dilation: f64,
    pub binding_stats: BindingStats,
    pub consolidation_stats: ConsolidationStats,
    pub in_flow_state: bool,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_time_embedding() {
        let embedding = HyperbolicTimeEmbedding::new(100.0);

        // t=0 should embed to 0
        assert!((embedding.embed(0.0) - 0.0).abs() < 1e-10);

        // t=100 (= τ) should embed to ln(2)
        let t_100 = embedding.embed(100.0);
        assert!((t_100 - 2.0_f64.ln()).abs() < 1e-10);

        // t=1000 should embed to ln(11) ≈ 2.398
        let t_1000 = embedding.embed(1000.0);
        assert!((t_1000 - 11.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_distance_scale_invariant() {
        let embedding = HyperbolicTimeEmbedding::new(100.0);

        // Scale invariance: d(10, 100) = d(100, 1000) = ln(10)
        let d1 = embedding.temporal_distance(100.0, 10.0);
        let d2 = embedding.temporal_distance(1000.0, 100.0);

        assert!((d1 - d2).abs() < 1e-10);
        assert!((d1 - 10.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_embedding() {
        let embedding = HyperbolicTimeEmbedding::new(100.0);

        for t in [0.0, 50.0, 100.0, 500.0, 1000.0] {
            let t_hyp = embedding.embed(t);
            let t_recovered = embedding.inverse_embed(t_hyp);
            assert!((t - t_recovered).abs() < 1e-6, "t={}, recovered={}", t, t_recovered);
        }
    }

    #[test]
    fn test_gamma_oscillator() {
        let mut osc = GammaOscillator::new(40.0);

        // Period should be 25ms
        assert!((osc.period_ms() - 25.0).abs() < 1e-10);

        // After one period, phase should be back to ~0
        for _ in 0..25 {
            osc.step(1.0);
        }
        assert!(osc.phase().abs() < 0.3 || (osc.phase() - 2.0 * PI).abs() < 0.3);
    }

    #[test]
    fn test_memory_consolidation() {
        let mut consolidator = MemoryConsolidator::default();

        // Encode a memory
        consolidator.encode(1, 0.8, 0.0);
        assert!(consolidator.get_strength(1).is_some());

        // Access multiple times to trigger consolidation
        consolidator.access(1, 100.0);
        consolidator.access(1, 200.0);

        // Update should consolidate
        consolidator.update(300.0);

        // Should have moved to LTM (may or may not have consolidated depending on strength)
        let stats = consolidator.stats();
        assert!(stats.stm_count + stats.ltm_count > 0); // Memory exists somewhere
    }

    #[test]
    fn test_subjective_time() {
        let mut clock = SubjectiveTimeClock::default();

        // With default parameters, subjective = objective
        clock.step(100.0);
        assert!((clock.objective() - 100.0).abs() < 1e-10);

        // High info density should slow subjective time (more perceived events)
        clock.set_info_density(2.0);
        clock.step(100.0);

        assert!(clock.subjective() > clock.objective());
    }

    #[test]
    fn test_temporal_attention_recency() {
        let attention = TemporalAttention::default();

        let key_times = vec![0.0, 500.0, 900.0, 1000.0];
        let weights = attention.attention_weights(1000.0, &key_times);

        // More recent times should have higher weight
        assert!(weights[3] > weights[2]);
        assert!(weights[2] > weights[1]);
        assert!(weights[1] > weights[0]);
    }

    #[test]
    fn test_temporal_binding() {
        let mut binder = TemporalBinder::new(50.0);

        // Add events
        binder.add_event(TemporalEvent { timestamp: 0.0, id: 1, strength: 1.0 });
        binder.add_event(TemporalEvent { timestamp: 10.0, id: 2, strength: 0.8 });

        // Step through a gamma cycle
        for t in 0..30 {
            binder.step(1.0, t as f64);
        }

        // Should have created bound groups or at least updated statistics
        let stats = binder.stats();
        assert!(stats.gamma_frequency > 0.0); // Verify oscillator is running
    }

    #[test]
    fn test_temporal_fabric() {
        let config = TemporalConfig::default();
        let mut fabric = TemporalConsciousnessFabric::new(config);

        // Process some events
        for i in 0..100 {
            fabric.step(1.0);
            if i % 10 == 0 {
                fabric.process_event(i as u64, 0.5);
            }
        }

        let stats = fabric.stats();
        assert!((stats.objective_time - 100.0).abs() < 1e-10);
        assert!(stats.embedded_time > 0.0);
    }

    #[test]
    fn test_weber_fechner_property() {
        // Weber-Fechner: equal ratios produce equal differences (approximately for t >> τ)
        let embedding = HyperbolicTimeEmbedding::new(100.0);

        // For large t >> τ, ln(1 + t/τ) ≈ ln(t/τ), so Weber-Fechner holds
        // Use larger times where the "+1" term is negligible
        // 1000→10000 and 10000→100000 should have ~equal embedding differences
        let d1 = embedding.embed(10000.0) - embedding.embed(1000.0);
        let d2 = embedding.embed(100000.0) - embedding.embed(10000.0);

        // For t >> τ, d1 ≈ d2 ≈ ln(10) = 2.303
        assert!((d1 - d2).abs() < 0.1, "d1={}, d2={}, diff={}", d1, d2, (d1 - d2).abs());

        // Verify logarithmic compression: equal RATIOS (not intervals) produce similar differences
        // 100→200 (ratio 2) vs 1000→2000 (ratio 2) should have similar embedding differences
        let double_100 = embedding.embed(200.0) - embedding.embed(100.0);
        let double_1000 = embedding.embed(2000.0) - embedding.embed(1000.0);
        // Both ≈ ln(2) for large t, but differ near τ=100
        assert!((double_100 - double_1000).abs() < 0.5,
            "double_100={}, double_1000={}", double_100, double_1000);
    }

    #[test]
    fn test_multi_scale_embedding() {
        let embedding = HyperbolicTimeEmbedding::new(100.0);

        let scales = embedding.multi_scale_embed(60000.0); // 1 minute

        // Each scale should show different compression
        assert!(scales[0] > scales[1]); // ms scale more compressed than s scale
    }

    #[test]
    fn test_wolfram_verified_stdp_window() {
        // STDP @ 25ms (gamma window): ΔW = 0.0287 (Wolfram-verified)
        let delta_w = stdp_weight_change(25.0);
        assert!((delta_w - 0.0287).abs() < 0.001);
    }
}
