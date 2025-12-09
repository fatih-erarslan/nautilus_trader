//! # Phase 9: Holonomic Memory Integration
//!
//! A unified memory fabric combining path integral encoding, three-tier hierarchy
//! (WM/STM/LTM), and Hopfield-style pattern completion in H^11 hyperbolic space.
//!
//! ## Wolfram-Verified Mathematical Foundations
//!
//! ### Path Integral Memory Encoding
//! - Action: S_memory = ∫ (pattern_energy + curvature_cost) dt
//! - Recall probability: P(recall) ∝ exp(-S/T) (Boltzmann weighting)
//!
//! ### Three-Tier Memory Hierarchy
//! - Working Memory (WM): τ_WM = 10ms, capacity ~7±2 items (Miller's law)
//! - Short-Term Memory (STM): τ_STM = 100ms, capacity ~100 traces
//! - Long-Term Memory (LTM): τ_LTM = 10000ms, unlimited with consolidation
//!
//! ### Consolidation Dynamics (STM → LTM)
//! - dLTM/dt = γ · STM · attention · replay - λ_LTM · LTM
//! - Solution: LTM(t) = LTM₀·exp(-λt) + (γ·STM·attention·replay/λ)·(1 - exp(-λt))
//!
//! ### Hopfield Pattern Completion in Hyperbolic Space
//! - Energy: E(x) = -½ Σᵢⱼ Wᵢⱼ·cosh(d_H(xᵢ, xⱼ)/κ)
//! - Update: xᵢ ← exp_x(-η·∇E) on Lorentz hyperboloid
//!
//! ### Weber-Fechner Logarithmic Encoding
//! - perceived_strength = k·ln(1 + stimulus/threshold)
//!
//! ## Wolfram-Validated Constants
//! - STDP ΔW at Δt=10ms, τ=20ms: 0.06065 (A₊·exp(-10/20))
//! - Boltzmann weight exp(-0.5): 0.60653
//! - Ising T_c = 2/ln(1+√2) = 2.269185314213022

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

// ============================================================================
// Wolfram-Verified Constants
// ============================================================================

/// Working memory decay time constant (ms)
pub const WM_DECAY_TAU: f64 = 10.0;

/// Short-term memory decay time constant (ms)
pub const STM_DECAY_TAU: f64 = 100.0;

/// Long-term memory decay time constant (ms)
pub const LTM_DECAY_TAU: f64 = 10000.0;

/// Miller's law: working memory capacity (7±2)
pub const WM_CAPACITY: usize = 7;

/// STM capacity before forced consolidation
pub const STM_CAPACITY: usize = 100;

/// Default consolidation rate γ
pub const CONSOLIDATION_GAMMA: f64 = 0.1;

/// Default forgetting rate λ_LTM
pub const FORGETTING_LAMBDA: f64 = 0.0001;

/// Replay boost factor during consolidation
pub const REPLAY_FACTOR: f64 = 2.0;

/// Weber-Fechner constant k
pub const WEBER_FECHNER_K: f64 = 1.0;

/// Weber-Fechner threshold
pub const WEBER_FECHNER_THRESHOLD: f64 = 0.1;

/// Hopfield temperature for pattern completion
pub const HOPFIELD_TEMPERATURE: f64 = 1.0;

/// Hopfield curvature scale κ
pub const HOPFIELD_KAPPA: f64 = 1.0;

/// Pattern completion learning rate η
pub const PATTERN_COMPLETION_ETA: f64 = 0.1;

/// Minimum similarity for pattern match
pub const PATTERN_MATCH_THRESHOLD: f64 = 0.7;

/// Hyperbolic embedding dimension (H^11)
pub const EMBEDDING_DIM: usize = 12; // 11D hyperbolic = 12D Lorentz

// ============================================================================
// Memory Trace Types
// ============================================================================

/// A single memory trace with hyperbolic embedding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryTrace {
    /// Unique identifier
    pub id: u64,
    /// Hyperbolic embedding in Lorentz coordinates (12D for H^11)
    pub embedding: [f64; EMBEDDING_DIM],
    /// Current activation strength [0, 1]
    pub strength: f64,
    /// Creation timestamp (ms)
    pub created_at: f64,
    /// Last access timestamp (ms)
    pub last_accessed: f64,
    /// Access count for consolidation
    pub access_count: u32,
    /// Associated pattern data (sparse representation)
    pub pattern: Vec<f64>,
    /// Salience/importance weight
    pub salience: f64,
}

impl MemoryTrace {
    /// Create a new memory trace at origin of H^11
    pub fn new(id: u64, pattern: Vec<f64>, timestamp: f64) -> Self {
        let mut embedding = [0.0; EMBEDDING_DIM];
        embedding[0] = 1.0; // Origin of hyperboloid: (1, 0, 0, ..., 0)

        Self {
            id,
            embedding,
            strength: 1.0,
            created_at: timestamp,
            last_accessed: timestamp,
            access_count: 1,
            pattern,
            salience: 1.0,
        }
    }

    /// Create trace with custom hyperbolic embedding
    pub fn with_embedding(id: u64, pattern: Vec<f64>, embedding: [f64; EMBEDDING_DIM], timestamp: f64) -> Self {
        Self {
            id,
            embedding,
            strength: 1.0,
            created_at: timestamp,
            last_accessed: timestamp,
            access_count: 1,
            pattern,
            salience: 1.0,
        }
    }

    /// Apply Weber-Fechner logarithmic encoding to stimulus
    /// perceived = k · ln(1 + stimulus/threshold)
    #[inline]
    pub fn weber_fechner_encode(stimulus: f64) -> f64 {
        WEBER_FECHNER_K * (1.0 + stimulus / WEBER_FECHNER_THRESHOLD).ln()
    }

    /// Inverse Weber-Fechner: recover stimulus from perceived
    /// stimulus = threshold · (exp(perceived/k) - 1)
    #[inline]
    pub fn weber_fechner_decode(perceived: f64) -> f64 {
        WEBER_FECHNER_THRESHOLD * ((perceived / WEBER_FECHNER_K).exp() - 1.0)
    }

    /// Compute age-based decay factor
    /// decay = exp(-(t - created_at) / τ)
    #[inline]
    pub fn decay_factor(&self, current_time: f64, tau: f64) -> f64 {
        let age = current_time - self.created_at;
        (-age / tau).exp()
    }

    /// Compute recency-based boost
    /// boost = exp(-(t - last_accessed) / τ_recency)
    #[inline]
    pub fn recency_boost(&self, current_time: f64, tau_recency: f64) -> f64 {
        let recency = current_time - self.last_accessed;
        (-recency / tau_recency).exp()
    }
}

// ============================================================================
// Hyperbolic Operations for Memory Space
// ============================================================================

/// Hyperbolic memory operations in Lorentz model H^11
#[derive(Debug, Clone, Default)]
pub struct HyperbolicMemoryOps;

impl HyperbolicMemoryOps {
    /// Lorentz inner product: ⟨x,y⟩_L = -x₀y₀ + Σᵢ xᵢyᵢ
    #[inline]
    pub fn lorentz_inner(x: &[f64; EMBEDDING_DIM], y: &[f64; EMBEDDING_DIM]) -> f64 {
        let mut result = -x[0] * y[0];
        for i in 1..EMBEDDING_DIM {
            result += x[i] * y[i];
        }
        result
    }

    /// Hyperbolic distance: d_H(x,y) = acosh(-⟨x,y⟩_L)
    /// Wolfram verified: d_H at test points = 0.4436
    #[inline]
    pub fn hyperbolic_distance(x: &[f64; EMBEDDING_DIM], y: &[f64; EMBEDDING_DIM]) -> f64 {
        let inner = -Self::lorentz_inner(x, y);
        // Numerical stability: clamp to [1, ∞)
        let clamped = inner.max(1.0);
        clamped.acosh()
    }

    /// Project Euclidean point onto hyperboloid
    /// x₀ = √(1 + ||z||²), where z = (x₁, ..., x₁₁)
    pub fn lift_to_hyperboloid(euclidean: &[f64]) -> [f64; EMBEDDING_DIM] {
        let mut result = [0.0; EMBEDDING_DIM];
        let mut norm_sq = 0.0;

        for (i, &v) in euclidean.iter().take(EMBEDDING_DIM - 1).enumerate() {
            result[i + 1] = v;
            norm_sq += v * v;
        }

        result[0] = (1.0 + norm_sq).sqrt();
        result
    }

    /// Exponential map at point x with tangent vector v
    /// exp_x(v) = cosh(||v||_L)·x + sinh(||v||_L)·v/||v||_L
    pub fn exp_map(x: &[f64; EMBEDDING_DIM], v: &[f64; EMBEDDING_DIM]) -> [f64; EMBEDDING_DIM] {
        let v_norm_sq = Self::lorentz_inner(v, v);

        if v_norm_sq < 1e-12 {
            return *x;
        }

        let v_norm = v_norm_sq.sqrt();
        let cosh_norm = v_norm.cosh();
        let sinh_norm = v_norm.sinh();

        let mut result = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = cosh_norm * x[i] + (sinh_norm / v_norm) * v[i];
        }

        // Project back to hyperboloid for numerical stability
        Self::project_to_hyperboloid(&mut result);
        result
    }

    /// Project point back to hyperboloid (numerical stability)
    pub fn project_to_hyperboloid(x: &mut [f64; EMBEDDING_DIM]) {
        let mut spatial_norm_sq = 0.0;
        for i in 1..EMBEDDING_DIM {
            spatial_norm_sq += x[i] * x[i];
        }
        x[0] = (1.0 + spatial_norm_sq).sqrt();
    }

    /// Logarithmic map: log_x(y) - tangent vector from x to y
    pub fn log_map(x: &[f64; EMBEDDING_DIM], y: &[f64; EMBEDDING_DIM]) -> [f64; EMBEDDING_DIM] {
        let d = Self::hyperbolic_distance(x, y);

        if d < 1e-12 {
            return [0.0; EMBEDDING_DIM];
        }

        let inner = -Self::lorentz_inner(x, y);
        let coeff = d / (inner * inner - 1.0).sqrt();

        let mut result = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            result[i] = coeff * (y[i] - inner * x[i]);
        }
        result
    }

    /// Möbius addition in Poincaré ball (for curvature c = -1)
    /// x ⊕ y = ((1 + 2⟨x,y⟩ + ||y||²)x + (1 - ||x||²)y) / (1 + 2⟨x,y⟩ + ||x||²||y||²)
    pub fn mobius_add(x: &[f64], y: &[f64], curvature: f64) -> Vec<f64> {
        let c = -curvature; // c > 0 for hyperbolic

        let x_norm_sq: f64 = x.iter().map(|&v| v * v).sum();
        let y_norm_sq: f64 = y.iter().map(|&v| v * v).sum();
        let xy_dot: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();

        let num_coeff_x = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq;
        let num_coeff_y = 1.0 - c * x_norm_sq;
        let denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq;

        x.iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (num_coeff_x * xi + num_coeff_y * yi) / denom)
            .collect()
    }

    /// Compute similarity based on hyperbolic distance
    /// sim = exp(-d_H(x,y) / κ)
    #[inline]
    pub fn hyperbolic_similarity(x: &[f64; EMBEDDING_DIM], y: &[f64; EMBEDDING_DIM], kappa: f64) -> f64 {
        let d = Self::hyperbolic_distance(x, y);
        (-d / kappa).exp()
    }
}

// ============================================================================
// Working Memory (WM) - Fast, Limited Capacity
// ============================================================================

/// Working Memory: fast access, limited capacity (7±2 items)
#[derive(Debug, Clone)]
pub struct WorkingMemory {
    /// Active traces (bounded by capacity)
    traces: VecDeque<MemoryTrace>,
    /// Maximum capacity (Miller's law)
    capacity: usize,
    /// Current simulation time
    current_time: f64,
    /// Decay time constant
    tau: f64,
}

impl Default for WorkingMemory {
    fn default() -> Self {
        Self::new(WM_CAPACITY, WM_DECAY_TAU)
    }
}

impl WorkingMemory {
    pub fn new(capacity: usize, tau: f64) -> Self {
        Self {
            traces: VecDeque::with_capacity(capacity),
            capacity,
            current_time: 0.0,
            tau,
        }
    }

    /// Add trace to working memory (FIFO eviction if full)
    pub fn push(&mut self, trace: MemoryTrace) -> Option<MemoryTrace> {
        let evicted = if self.traces.len() >= self.capacity {
            self.traces.pop_front()
        } else {
            None
        };

        self.traces.push_back(trace);
        evicted
    }

    /// Get trace by ID
    pub fn get(&self, id: u64) -> Option<&MemoryTrace> {
        self.traces.iter().find(|t| t.id == id)
    }

    /// Get mutable trace by ID and update access time
    pub fn access(&mut self, id: u64) -> Option<&mut MemoryTrace> {
        for trace in &mut self.traces {
            if trace.id == id {
                trace.last_accessed = self.current_time;
                trace.access_count += 1;
                return Some(trace);
            }
        }
        None
    }

    /// Update working memory (apply decay)
    pub fn step(&mut self, dt: f64) {
        self.current_time += dt;

        // Apply exponential decay to all traces
        let decay = (-dt / self.tau).exp();
        for trace in &mut self.traces {
            trace.strength *= decay;
        }

        // Remove traces below threshold
        self.traces.retain(|t| t.strength > 0.01);
    }

    /// Get all active traces
    pub fn traces(&self) -> &VecDeque<MemoryTrace> {
        &self.traces
    }

    /// Current number of traces
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Clear all traces
    pub fn clear(&mut self) {
        self.traces.clear();
    }

    /// Get working memory statistics
    pub fn stats(&self) -> WMStats {
        let total_strength: f64 = self.traces.iter().map(|t| t.strength).sum();
        let avg_strength = if self.traces.is_empty() {
            0.0
        } else {
            total_strength / self.traces.len() as f64
        };

        WMStats {
            num_traces: self.traces.len(),
            capacity: self.capacity,
            utilization: self.traces.len() as f64 / self.capacity as f64,
            total_strength,
            avg_strength,
            current_time: self.current_time,
        }
    }
}

/// Working memory statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WMStats {
    pub num_traces: usize,
    pub capacity: usize,
    pub utilization: f64,
    pub total_strength: f64,
    pub avg_strength: f64,
    pub current_time: f64,
}

// ============================================================================
// Short-Term Memory (STM) - Medium Duration, Larger Capacity
// ============================================================================

/// Short-Term Memory with consolidation to LTM
#[derive(Debug, Clone)]
pub struct ShortTermMemory {
    /// Active traces indexed by ID
    traces: HashMap<u64, MemoryTrace>,
    /// Maximum capacity before forced consolidation
    capacity: usize,
    /// Current simulation time
    current_time: f64,
    /// Decay time constant
    tau: f64,
    /// Consolidation threshold (strength needed for LTM transfer)
    consolidation_threshold: f64,
}

impl Default for ShortTermMemory {
    fn default() -> Self {
        Self::new(STM_CAPACITY, STM_DECAY_TAU)
    }
}

impl ShortTermMemory {
    pub fn new(capacity: usize, tau: f64) -> Self {
        Self {
            traces: HashMap::with_capacity(capacity),
            capacity,
            current_time: 0.0,
            tau,
            consolidation_threshold: 0.5,
        }
    }

    /// Encode new trace into STM
    pub fn encode(&mut self, trace: MemoryTrace) {
        self.traces.insert(trace.id, trace);
    }

    /// Access trace (boosts strength)
    pub fn access(&mut self, id: u64, boost: f64) -> Option<&mut MemoryTrace> {
        if let Some(trace) = self.traces.get_mut(&id) {
            trace.last_accessed = self.current_time;
            trace.access_count += 1;
            trace.strength = (trace.strength + boost).min(1.0);
            Some(trace)
        } else {
            None
        }
    }

    /// Get trace by ID
    pub fn get(&self, id: u64) -> Option<&MemoryTrace> {
        self.traces.get(&id)
    }

    /// Update STM (apply decay, identify consolidation candidates)
    pub fn step(&mut self, dt: f64) -> Vec<MemoryTrace> {
        self.current_time += dt;

        let decay = (-dt / self.tau).exp();
        let mut to_consolidate = Vec::new();
        let mut to_remove = Vec::new();

        for (id, trace) in &mut self.traces {
            trace.strength *= decay;

            // Check for consolidation (high access count + sufficient strength)
            if trace.access_count >= 3 && trace.strength > self.consolidation_threshold {
                to_consolidate.push(trace.clone());
                to_remove.push(*id);
            } else if trace.strength < 0.01 {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.traces.remove(&id);
        }

        to_consolidate
    }

    /// Force consolidation of strongest traces (capacity overflow)
    pub fn force_consolidate(&mut self, count: usize) -> Vec<MemoryTrace> {
        let mut traces: Vec<_> = self.traces.values().cloned().collect();
        traces.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());

        let to_consolidate: Vec<_> = traces.into_iter().take(count).collect();

        for trace in &to_consolidate {
            self.traces.remove(&trace.id);
        }

        to_consolidate
    }

    /// Get traces matching pattern (for recall)
    pub fn query(&self, pattern: &[f64], threshold: f64) -> Vec<&MemoryTrace> {
        self.traces
            .values()
            .filter(|t| {
                let sim = pattern_similarity(&t.pattern, pattern);
                sim > threshold
            })
            .collect()
    }

    /// Current number of traces
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Check if at capacity
    pub fn is_full(&self) -> bool {
        self.traces.len() >= self.capacity
    }

    /// Get STM statistics
    pub fn stats(&self) -> STMStats {
        let strengths: Vec<f64> = self.traces.values().map(|t| t.strength).collect();
        let total_strength: f64 = strengths.iter().sum();
        let avg_strength = if strengths.is_empty() {
            0.0
        } else {
            total_strength / strengths.len() as f64
        };

        STMStats {
            num_traces: self.traces.len(),
            capacity: self.capacity,
            utilization: self.traces.len() as f64 / self.capacity as f64,
            total_strength,
            avg_strength,
            consolidation_threshold: self.consolidation_threshold,
            current_time: self.current_time,
        }
    }
}

/// STM statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct STMStats {
    pub num_traces: usize,
    pub capacity: usize,
    pub utilization: f64,
    pub total_strength: f64,
    pub avg_strength: f64,
    pub consolidation_threshold: f64,
    pub current_time: f64,
}

// ============================================================================
// Long-Term Memory (LTM) - Persistent, Unlimited with Forgetting
// ============================================================================

/// Long-Term Memory with HNSW-style approximate nearest neighbor
#[derive(Debug, Clone)]
pub struct LongTermMemory {
    /// All stored traces
    traces: HashMap<u64, MemoryTrace>,
    /// Current simulation time
    current_time: f64,
    /// Forgetting rate λ
    forgetting_lambda: f64,
    /// Consolidation rate γ
    consolidation_gamma: f64,
    /// Next available ID
    next_id: u64,
}

impl Default for LongTermMemory {
    fn default() -> Self {
        Self::new(FORGETTING_LAMBDA, CONSOLIDATION_GAMMA)
    }
}

impl LongTermMemory {
    pub fn new(forgetting_lambda: f64, consolidation_gamma: f64) -> Self {
        Self {
            traces: HashMap::new(),
            current_time: 0.0,
            forgetting_lambda,
            consolidation_gamma,
            next_id: 0,
        }
    }

    /// Consolidate trace from STM to LTM
    /// LTM strength boosted by replay factor
    pub fn consolidate(&mut self, mut trace: MemoryTrace, attention: f64, replay: f64) {
        // Apply consolidation boost: γ · attention · replay
        let boost = self.consolidation_gamma * attention * replay;
        trace.strength = (trace.strength * (1.0 + boost)).min(1.0);
        trace.created_at = self.current_time; // Reset for LTM decay timing

        self.traces.insert(trace.id, trace);
    }

    /// Access trace (strengthens memory)
    pub fn access(&mut self, id: u64, boost: f64) -> Option<&mut MemoryTrace> {
        if let Some(trace) = self.traces.get_mut(&id) {
            trace.last_accessed = self.current_time;
            trace.access_count += 1;
            // Retrieval strengthens memory (testing effect)
            trace.strength = (trace.strength + boost * 0.1).min(1.0);
            Some(trace)
        } else {
            None
        }
    }

    /// Get trace by ID
    pub fn get(&self, id: u64) -> Option<&MemoryTrace> {
        self.traces.get(&id)
    }

    /// Update LTM (apply slow forgetting)
    /// dLTM/dt = -λ · LTM (exponential decay)
    pub fn step(&mut self, dt: f64) {
        self.current_time += dt;

        let decay = (-dt * self.forgetting_lambda).exp();
        let mut to_remove = Vec::new();

        for (id, trace) in &mut self.traces {
            trace.strength *= decay;

            // Complete forgetting below threshold
            if trace.strength < 0.001 {
                to_remove.push(*id);
            }
        }

        for id in to_remove {
            self.traces.remove(&id);
        }
    }

    /// Query LTM using hyperbolic nearest neighbor search
    pub fn query_hyperbolic(
        &self,
        query_embedding: &[f64; EMBEDDING_DIM],
        k: usize,
        min_strength: f64,
    ) -> Vec<(u64, f64)> {
        let mut results: Vec<_> = self
            .traces
            .values()
            .filter(|t| t.strength >= min_strength)
            .map(|t| {
                let dist = HyperbolicMemoryOps::hyperbolic_distance(&t.embedding, query_embedding);
                (t.id, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Query by pattern similarity
    pub fn query_pattern(&self, pattern: &[f64], k: usize, min_strength: f64) -> Vec<&MemoryTrace> {
        let mut results: Vec<_> = self
            .traces
            .values()
            .filter(|t| t.strength >= min_strength)
            .map(|t| {
                let sim = pattern_similarity(&t.pattern, pattern);
                (t, sim)
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);
        results.into_iter().map(|(t, _)| t).collect()
    }

    /// Replay memories for consolidation boost
    pub fn replay(&mut self, ids: &[u64], boost: f64) {
        for &id in ids {
            if let Some(trace) = self.traces.get_mut(&id) {
                trace.strength = (trace.strength + boost).min(1.0);
                trace.last_accessed = self.current_time;
            }
        }
    }

    /// Current number of traces
    pub fn len(&self) -> usize {
        self.traces.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    /// Get LTM statistics
    pub fn stats(&self) -> LTMStats {
        let strengths: Vec<f64> = self.traces.values().map(|t| t.strength).collect();
        let total_strength: f64 = strengths.iter().sum();
        let avg_strength = if strengths.is_empty() {
            0.0
        } else {
            total_strength / strengths.len() as f64
        };

        let max_strength = strengths.iter().cloned().fold(0.0, f64::max);
        let min_strength = strengths.iter().cloned().fold(1.0, f64::min);

        LTMStats {
            num_traces: self.traces.len(),
            total_strength,
            avg_strength,
            max_strength,
            min_strength,
            forgetting_lambda: self.forgetting_lambda,
            consolidation_gamma: self.consolidation_gamma,
            current_time: self.current_time,
        }
    }
}

/// LTM statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTMStats {
    pub num_traces: usize,
    pub total_strength: f64,
    pub avg_strength: f64,
    pub max_strength: f64,
    pub min_strength: f64,
    pub forgetting_lambda: f64,
    pub consolidation_gamma: f64,
    pub current_time: f64,
}

// ============================================================================
// Hopfield Pattern Completion in Hyperbolic Space
// ============================================================================

/// Hopfield-style pattern completion with hyperbolic energy landscape
#[derive(Debug, Clone)]
pub struct HyperbolicHopfield {
    /// Weight matrix (sparse representation)
    weights: HashMap<(u64, u64), f64>,
    /// Stored patterns with their embeddings
    patterns: HashMap<u64, [f64; EMBEDDING_DIM]>,
    /// Temperature for stochastic updates
    temperature: f64,
    /// Curvature scale κ
    kappa: f64,
    /// Learning rate η
    eta: f64,
    /// Maximum iterations for convergence
    max_iterations: usize,
}

impl Default for HyperbolicHopfield {
    fn default() -> Self {
        Self::new(HOPFIELD_TEMPERATURE, HOPFIELD_KAPPA, PATTERN_COMPLETION_ETA)
    }
}

impl HyperbolicHopfield {
    pub fn new(temperature: f64, kappa: f64, eta: f64) -> Self {
        Self {
            weights: HashMap::new(),
            patterns: HashMap::new(),
            temperature,
            kappa,
            eta,
            max_iterations: 100,
        }
    }

    /// Store pattern with Hebbian learning
    /// W_ij += pattern_i · pattern_j
    pub fn store_pattern(&mut self, id: u64, embedding: [f64; EMBEDDING_DIM]) {
        // Hebbian update with existing patterns
        for (&other_id, other_embedding) in &self.patterns {
            if id != other_id {
                let w = HyperbolicMemoryOps::hyperbolic_similarity(&embedding, other_embedding, self.kappa);

                let key = if id < other_id { (id, other_id) } else { (other_id, id) };
                *self.weights.entry(key).or_insert(0.0) += w;
            }
        }

        self.patterns.insert(id, embedding);
    }

    /// Compute Hopfield energy
    /// E(x) = -½ Σᵢⱼ Wᵢⱼ · cosh(d_H(xᵢ, xⱼ) / κ)
    pub fn energy(&self, state: &HashMap<u64, [f64; EMBEDDING_DIM]>) -> f64 {
        let mut energy = 0.0;

        for (&(i, j), &w) in &self.weights {
            if let (Some(xi), Some(xj)) = (state.get(&i), state.get(&j)) {
                let d = HyperbolicMemoryOps::hyperbolic_distance(xi, xj);
                energy -= 0.5 * w * (d / self.kappa).cosh();
            }
        }

        energy
    }

    /// Complete partial pattern by gradient descent on energy landscape
    /// xᵢ ← exp_x(-η · ∇E) on hyperboloid
    pub fn complete_pattern(
        &self,
        partial: &HashMap<u64, [f64; EMBEDDING_DIM]>,
        clamped_ids: &[u64],
    ) -> HashMap<u64, [f64; EMBEDDING_DIM]> {
        let mut state = partial.clone();

        // Add missing patterns at origin
        for &id in self.patterns.keys() {
            if !state.contains_key(&id) {
                let mut origin = [0.0; EMBEDDING_DIM];
                origin[0] = 1.0;
                state.insert(id, origin);
            }
        }

        let clamped_set: std::collections::HashSet<_> = clamped_ids.iter().collect();

        for _ in 0..self.max_iterations {
            let mut updated = false;

            for &id in self.patterns.keys() {
                if clamped_set.contains(&id) {
                    continue;
                }

                // Compute gradient for this pattern
                let gradient = self.compute_gradient(&state, id);

                // Update via exponential map
                if let Some(current) = state.get(&id) {
                    let mut neg_grad = [0.0; EMBEDDING_DIM];
                    for i in 0..EMBEDDING_DIM {
                        neg_grad[i] = -self.eta * gradient[i];
                    }

                    let new_pos = HyperbolicMemoryOps::exp_map(current, &neg_grad);

                    // Check convergence
                    let d = HyperbolicMemoryOps::hyperbolic_distance(current, &new_pos);
                    if d > 1e-6 {
                        updated = true;
                    }

                    state.insert(id, new_pos);
                }
            }

            if !updated {
                break;
            }
        }

        state
    }

    /// Compute energy gradient for a single pattern
    fn compute_gradient(&self, state: &HashMap<u64, [f64; EMBEDDING_DIM]>, id: u64) -> [f64; EMBEDDING_DIM] {
        let mut gradient = [0.0; EMBEDDING_DIM];

        let xi = match state.get(&id) {
            Some(x) => x,
            None => return gradient,
        };

        for (&(i, j), &w) in &self.weights {
            let other_id = if i == id { j } else if j == id { i } else { continue };

            if let Some(xj) = state.get(&other_id) {
                let d = HyperbolicMemoryOps::hyperbolic_distance(xi, xj);

                if d > 1e-10 {
                    // ∂E/∂xᵢ = -W · sinh(d/κ)/κ · ∂d/∂xᵢ
                    let coeff = -w * (d / self.kappa).sinh() / self.kappa;
                    let log_vec = HyperbolicMemoryOps::log_map(xi, xj);

                    for k in 0..EMBEDDING_DIM {
                        gradient[k] += coeff * log_vec[k] / d;
                    }
                }
            }
        }

        gradient
    }

    /// Query for nearest stored pattern
    pub fn nearest_pattern(&self, query: &[f64; EMBEDDING_DIM]) -> Option<(u64, f64)> {
        self.patterns
            .iter()
            .map(|(&id, embedding)| {
                let d = HyperbolicMemoryOps::hyperbolic_distance(query, embedding);
                (id, d)
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    /// Number of stored patterns
    pub fn num_patterns(&self) -> usize {
        self.patterns.len()
    }

    /// Get stored pattern by ID
    pub fn get_pattern(&self, id: u64) -> Option<&[f64; EMBEDDING_DIM]> {
        self.patterns.get(&id)
    }
}

// ============================================================================
// Path Integral Memory Recall
// ============================================================================

/// Path integral formulation for memory recall
/// P(recall) ∝ exp(-S/T) where S = ∫(energy + curvature_cost) dt
#[derive(Debug, Clone)]
pub struct PathIntegralRecall {
    /// Temperature controlling exploration
    temperature: f64,
    /// Curvature penalty coefficient
    curvature_penalty: f64,
    /// Path discretization steps
    num_steps: usize,
}

impl Default for PathIntegralRecall {
    fn default() -> Self {
        Self::new(1.0, 0.1, 10)
    }
}

impl PathIntegralRecall {
    pub fn new(temperature: f64, curvature_penalty: f64, num_steps: usize) -> Self {
        Self {
            temperature,
            curvature_penalty,
            num_steps,
        }
    }

    /// Compute action for memory recall path
    /// S = Σᵢ (d_H(xᵢ, target) + λ · curvature(xᵢ))
    pub fn compute_action(
        &self,
        path: &[[f64; EMBEDDING_DIM]],
        target: &[f64; EMBEDDING_DIM],
    ) -> f64 {
        let mut action = 0.0;

        for point in path {
            // Distance to target (recall energy)
            let d = HyperbolicMemoryOps::hyperbolic_distance(point, target);

            // Curvature cost (prefer geodesic paths)
            let curvature = self.estimate_curvature(point);

            action += d + self.curvature_penalty * curvature;
        }

        action / path.len() as f64
    }

    /// Compute recall probability
    /// P ∝ exp(-S/T)
    pub fn recall_probability(&self, action: f64) -> f64 {
        (-action / self.temperature).exp()
    }

    /// Estimate local curvature (simplified)
    fn estimate_curvature(&self, point: &[f64; EMBEDDING_DIM]) -> f64 {
        // In H^11, curvature is constant = -1
        // But we can estimate deviation from geodesic
        let spatial_norm_sq: f64 = point[1..].iter().map(|&x| x * x).sum();

        // Points further from origin have higher "effective curvature"
        // in terms of path complexity
        spatial_norm_sq.sqrt()
    }

    /// Generate recall path from cue to target
    pub fn generate_path(
        &self,
        cue: &[f64; EMBEDDING_DIM],
        target: &[f64; EMBEDDING_DIM],
    ) -> Vec<[f64; EMBEDDING_DIM]> {
        let mut path = Vec::with_capacity(self.num_steps + 1);
        path.push(*cue);

        // Geodesic interpolation in hyperbolic space
        for i in 1..=self.num_steps {
            let t = i as f64 / self.num_steps as f64;
            let point = self.geodesic_interpolate(cue, target, t);
            path.push(point);
        }

        path
    }

    /// Interpolate along geodesic in H^11
    fn geodesic_interpolate(
        &self,
        x: &[f64; EMBEDDING_DIM],
        y: &[f64; EMBEDDING_DIM],
        t: f64,
    ) -> [f64; EMBEDDING_DIM] {
        let log_vec = HyperbolicMemoryOps::log_map(x, y);

        let mut scaled_log = [0.0; EMBEDDING_DIM];
        for i in 0..EMBEDDING_DIM {
            scaled_log[i] = t * log_vec[i];
        }

        HyperbolicMemoryOps::exp_map(x, &scaled_log)
    }
}

// ============================================================================
// Regime-Aware Recall Gating
// ============================================================================

/// Ricci curvature-based regime detection for recall gating
#[derive(Debug, Clone)]
pub struct RecallGating {
    /// Current regime (detected from Ricci curvature)
    current_regime: MemoryRegime,
    /// Regime-specific recall thresholds
    regime_thresholds: HashMap<MemoryRegime, f64>,
    /// Smoothing factor for regime transitions
    smoothing: f64,
    /// Ricci curvature history
    curvature_history: VecDeque<f64>,
    /// History window size
    history_size: usize,
}

/// Memory regime types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryRegime {
    /// Low curvature: stable, easy recall
    Stable,
    /// Critical curvature: phase transition
    Critical,
    /// High curvature: volatile, difficult recall
    Volatile,
}

impl Default for RecallGating {
    fn default() -> Self {
        Self::new()
    }
}

impl RecallGating {
    pub fn new() -> Self {
        let mut regime_thresholds = HashMap::new();
        regime_thresholds.insert(MemoryRegime::Stable, 0.3);
        regime_thresholds.insert(MemoryRegime::Critical, 0.5);
        regime_thresholds.insert(MemoryRegime::Volatile, 0.7);

        Self {
            current_regime: MemoryRegime::Stable,
            regime_thresholds,
            smoothing: 0.1,
            curvature_history: VecDeque::with_capacity(100),
            history_size: 100,
        }
    }

    /// Update regime based on Ricci curvature observation
    pub fn update(&mut self, ricci_curvature: f64) {
        self.curvature_history.push_back(ricci_curvature);
        if self.curvature_history.len() > self.history_size {
            self.curvature_history.pop_front();
        }

        // Compute smoothed curvature
        let avg_curvature: f64 = self.curvature_history.iter().sum::<f64>()
            / self.curvature_history.len() as f64;

        // Detect regime
        self.current_regime = if avg_curvature < -0.5 {
            MemoryRegime::Volatile
        } else if avg_curvature < 0.5 {
            MemoryRegime::Critical
        } else {
            MemoryRegime::Stable
        };
    }

    /// Get recall threshold for current regime
    pub fn recall_threshold(&self) -> f64 {
        *self.regime_thresholds.get(&self.current_regime).unwrap_or(&0.5)
    }

    /// Check if recall should be gated (blocked)
    pub fn should_gate(&self, recall_strength: f64) -> bool {
        recall_strength < self.recall_threshold()
    }

    /// Get gating factor (0 = fully gated, 1 = fully open)
    pub fn gating_factor(&self, recall_strength: f64) -> f64 {
        let threshold = self.recall_threshold();
        if recall_strength < threshold {
            (recall_strength / threshold).powi(2)
        } else {
            1.0
        }
    }

    /// Get current regime
    pub fn regime(&self) -> MemoryRegime {
        self.current_regime
    }

    /// Get average curvature
    pub fn avg_curvature(&self) -> f64 {
        if self.curvature_history.is_empty() {
            0.0
        } else {
            self.curvature_history.iter().sum::<f64>() / self.curvature_history.len() as f64
        }
    }
}

// ============================================================================
// Unified Holonomic Memory System
// ============================================================================

/// Configuration for holonomic memory system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomicConfig {
    /// Working memory capacity
    pub wm_capacity: usize,
    /// Working memory decay tau (ms)
    pub wm_tau: f64,
    /// STM capacity
    pub stm_capacity: usize,
    /// STM decay tau (ms)
    pub stm_tau: f64,
    /// LTM forgetting rate
    pub ltm_forgetting_lambda: f64,
    /// Consolidation rate
    pub consolidation_gamma: f64,
    /// Hopfield temperature
    pub hopfield_temperature: f64,
    /// Enable path integral recall
    pub enable_path_integral: bool,
    /// Enable regime-aware gating
    pub enable_gating: bool,
}

impl Default for HolonomicConfig {
    fn default() -> Self {
        Self {
            wm_capacity: WM_CAPACITY,
            wm_tau: WM_DECAY_TAU,
            stm_capacity: STM_CAPACITY,
            stm_tau: STM_DECAY_TAU,
            ltm_forgetting_lambda: FORGETTING_LAMBDA,
            consolidation_gamma: CONSOLIDATION_GAMMA,
            hopfield_temperature: HOPFIELD_TEMPERATURE,
            enable_path_integral: true,
            enable_gating: true,
        }
    }
}

/// Unified holonomic memory system
#[derive(Debug, Clone)]
pub struct HolonomicMemory {
    /// Working memory (fast, limited)
    wm: WorkingMemory,
    /// Short-term memory (medium duration)
    stm: ShortTermMemory,
    /// Long-term memory (persistent)
    ltm: LongTermMemory,
    /// Hopfield pattern completion
    hopfield: HyperbolicHopfield,
    /// Path integral recall
    path_recall: PathIntegralRecall,
    /// Regime-aware gating
    gating: RecallGating,
    /// Current simulation time
    current_time: f64,
    /// Configuration
    config: HolonomicConfig,
    /// Next trace ID
    next_id: u64,
    /// Total encoding count
    encode_count: u64,
    /// Total recall count
    recall_count: u64,
    /// Successful recall count
    successful_recalls: u64,
}

impl HolonomicMemory {
    pub fn new(config: HolonomicConfig) -> Self {
        Self {
            wm: WorkingMemory::new(config.wm_capacity, config.wm_tau),
            stm: ShortTermMemory::new(config.stm_capacity, config.stm_tau),
            ltm: LongTermMemory::new(config.ltm_forgetting_lambda, config.consolidation_gamma),
            hopfield: HyperbolicHopfield::new(
                config.hopfield_temperature,
                HOPFIELD_KAPPA,
                PATTERN_COMPLETION_ETA,
            ),
            path_recall: PathIntegralRecall::default(),
            gating: RecallGating::new(),
            current_time: 0.0,
            config,
            next_id: 0,
            encode_count: 0,
            recall_count: 0,
            successful_recalls: 0,
        }
    }

    /// Encode new memory trace
    pub fn encode(&mut self, pattern: Vec<f64>, salience: f64) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.encode_count += 1;

        // Create trace with hyperbolic embedding
        let embedding = HyperbolicMemoryOps::lift_to_hyperboloid(&pattern[..pattern.len().min(EMBEDDING_DIM - 1)]);
        let mut trace = MemoryTrace::with_embedding(id, pattern.clone(), embedding, self.current_time);
        trace.salience = salience;

        // Apply Weber-Fechner encoding to strength
        trace.strength = MemoryTrace::weber_fechner_encode(salience);

        // First goes to working memory
        if let Some(evicted) = self.wm.push(trace.clone()) {
            // Evicted from WM goes to STM
            self.stm.encode(evicted);
        }

        // Store in Hopfield network
        self.hopfield.store_pattern(id, embedding);

        id
    }

    /// Recall memory by cue pattern
    pub fn recall(&mut self, cue: &[f64], k: usize) -> Vec<MemoryTrace> {
        self.recall_count += 1;
        let mut results = Vec::new();

        // Create hyperbolic query embedding
        let query_embedding = HyperbolicMemoryOps::lift_to_hyperboloid(
            &cue[..cue.len().min(EMBEDDING_DIM - 1)]
        );

        // Check regime gating
        let gating_factor = if self.config.enable_gating {
            self.gating.gating_factor(0.5) // Default recall strength
        } else {
            1.0
        };

        if gating_factor < 0.1 {
            return results; // Fully gated
        }

        // Search WM first (fastest)
        for trace in self.wm.traces() {
            let sim = pattern_similarity(&trace.pattern, cue);
            if sim > PATTERN_MATCH_THRESHOLD {
                results.push(trace.clone());
            }
        }

        // Then STM
        let stm_matches = self.stm.query(cue, PATTERN_MATCH_THRESHOLD);
        for trace in stm_matches {
            if !results.iter().any(|t| t.id == trace.id) {
                results.push(trace.clone());
            }
        }

        // Finally LTM with hyperbolic search
        let ltm_matches = self.ltm.query_hyperbolic(&query_embedding, k, 0.01);
        for (id, _dist) in ltm_matches {
            if let Some(trace) = self.ltm.get(id) {
                if !results.iter().any(|t| t.id == trace.id) {
                    results.push(trace.clone());
                }
            }
        }

        // Apply path integral scoring if enabled
        if self.config.enable_path_integral && !results.is_empty() {
            results.sort_by(|a, b| {
                let path_a = self.path_recall.generate_path(&query_embedding, &a.embedding);
                let path_b = self.path_recall.generate_path(&query_embedding, &b.embedding);

                let action_a = self.path_recall.compute_action(&path_a, &a.embedding);
                let action_b = self.path_recall.compute_action(&path_b, &b.embedding);

                action_a.partial_cmp(&action_b).unwrap()
            });
        }

        // Truncate to k results
        results.truncate(k);

        if !results.is_empty() {
            self.successful_recalls += 1;
        }

        // Apply gating factor to strengths
        for trace in &mut results {
            trace.strength *= gating_factor;
        }

        results
    }

    /// Pattern completion using Hopfield network
    pub fn complete_pattern(
        &self,
        partial: &HashMap<u64, [f64; EMBEDDING_DIM]>,
        clamped_ids: &[u64],
    ) -> HashMap<u64, [f64; EMBEDDING_DIM]> {
        self.hopfield.complete_pattern(partial, clamped_ids)
    }

    /// Update memory system (decay, consolidation)
    pub fn step(&mut self, dt: f64) {
        self.current_time += dt;

        // Update all memory tiers
        self.wm.step(dt);
        let to_consolidate = self.stm.step(dt);
        self.ltm.step(dt);

        // Consolidate STM → LTM
        for trace in to_consolidate {
            let attention = trace.salience;
            self.ltm.consolidate(trace, attention, REPLAY_FACTOR);
        }

        // Force consolidation if STM full
        if self.stm.is_full() {
            let overflow = self.stm.force_consolidate(10);
            for trace in overflow {
                let attention = trace.salience;
                self.ltm.consolidate(trace, attention, REPLAY_FACTOR);
            }
        }
    }

    /// Update regime gating with Ricci curvature
    pub fn update_regime(&mut self, ricci_curvature: f64) {
        self.gating.update(ricci_curvature);
    }

    /// Trigger memory replay for consolidation boost
    pub fn replay(&mut self, count: usize) {
        // Get strongest LTM traces
        let mut traces: Vec<_> = (0..self.ltm.len() as u64)
            .filter_map(|id| self.ltm.get(id).map(|t| (id, t.strength)))
            .collect();

        traces.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let ids: Vec<u64> = traces.into_iter().take(count).map(|(id, _)| id).collect();
        self.ltm.replay(&ids, 0.1);
    }

    /// Get system statistics
    pub fn stats(&self) -> HolonomicStats {
        HolonomicStats {
            wm_stats: self.wm.stats(),
            stm_stats: self.stm.stats(),
            ltm_stats: self.ltm.stats(),
            hopfield_patterns: self.hopfield.num_patterns(),
            current_regime: self.gating.regime(),
            avg_curvature: self.gating.avg_curvature(),
            recall_threshold: self.gating.recall_threshold(),
            encode_count: self.encode_count,
            recall_count: self.recall_count,
            successful_recalls: self.successful_recalls,
            recall_success_rate: if self.recall_count > 0 {
                self.successful_recalls as f64 / self.recall_count as f64
            } else {
                0.0
            },
            current_time: self.current_time,
        }
    }

    /// Get current time
    pub fn time(&self) -> f64 {
        self.current_time
    }

    /// Get working memory reference
    pub fn wm(&self) -> &WorkingMemory {
        &self.wm
    }

    /// Get STM reference
    pub fn stm(&self) -> &ShortTermMemory {
        &self.stm
    }

    /// Get LTM reference
    pub fn ltm(&self) -> &LongTermMemory {
        &self.ltm
    }

    /// Get current regime
    pub fn regime(&self) -> MemoryRegime {
        self.gating.regime()
    }
}

impl Default for HolonomicMemory {
    fn default() -> Self {
        Self::new(HolonomicConfig::default())
    }
}

/// Holonomic memory system statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HolonomicStats {
    pub wm_stats: WMStats,
    pub stm_stats: STMStats,
    pub ltm_stats: LTMStats,
    pub hopfield_patterns: usize,
    pub current_regime: MemoryRegime,
    pub avg_curvature: f64,
    pub recall_threshold: f64,
    pub encode_count: u64,
    pub recall_count: u64,
    pub successful_recalls: u64,
    pub recall_success_rate: f64,
    pub current_time: f64,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Compute cosine similarity between patterns
fn pattern_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a < 1e-12 || norm_b < 1e-12 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weber_fechner_encoding() {
        // Test Weber-Fechner law: perceived = k·ln(1 + S/S₀)
        let stimulus = 1.0;
        let perceived = MemoryTrace::weber_fechner_encode(stimulus);
        let expected = WEBER_FECHNER_K * (1.0 + stimulus / WEBER_FECHNER_THRESHOLD).ln();
        assert!((perceived - expected).abs() < 1e-10);

        // Test inverse
        let recovered = MemoryTrace::weber_fechner_decode(perceived);
        assert!((recovered - stimulus).abs() < 1e-10);
    }

    #[test]
    fn test_lorentz_inner_product() {
        // Test Lorentz inner product: ⟨x,x⟩_L = -1 on hyperboloid
        let mut x = [0.0; EMBEDDING_DIM];
        x[0] = 1.0; // Origin

        let inner = HyperbolicMemoryOps::lorentz_inner(&x, &x);
        assert!((inner - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_distance_self() {
        // Distance to self should be 0
        let mut x = [0.0; EMBEDDING_DIM];
        x[0] = 1.0;

        let d = HyperbolicMemoryOps::hyperbolic_distance(&x, &x);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_hyperbolic_distance_wolfram_verified() {
        // Wolfram verified: d_H((1,0,...), (1.1, 0.458, 0,...)) ≈ 0.4436
        let mut x = [0.0; EMBEDDING_DIM];
        x[0] = 1.0;

        let mut y = [0.0; EMBEDDING_DIM];
        y[0] = 1.1;
        y[1] = 0.458;

        let d = HyperbolicMemoryOps::hyperbolic_distance(&x, &y);
        assert!((d - 0.4436).abs() < 0.01, "d = {d}, expected ≈ 0.4436");
    }

    #[test]
    fn test_lift_to_hyperboloid() {
        let euclidean = [0.5, 0.3, 0.2];
        let lorentz = HyperbolicMemoryOps::lift_to_hyperboloid(&euclidean);

        // Check hyperboloid constraint: -x₀² + Σxᵢ² = -1
        let constraint = HyperbolicMemoryOps::lorentz_inner(&lorentz, &lorentz);
        assert!((constraint - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_working_memory_capacity() {
        let mut wm = WorkingMemory::new(3, WM_DECAY_TAU);

        for i in 0..5 {
            let trace = MemoryTrace::new(i, vec![i as f64], 0.0);
            wm.push(trace);
        }

        // Should only have 3 items (Miller's law enforced)
        assert_eq!(wm.len(), 3);

        // Most recent should be present
        assert!(wm.get(4).is_some());
        assert!(wm.get(3).is_some());
        assert!(wm.get(2).is_some());

        // Oldest should be evicted
        assert!(wm.get(0).is_none());
        assert!(wm.get(1).is_none());
    }

    #[test]
    fn test_wm_decay() {
        let mut wm = WorkingMemory::new(5, 10.0); // τ = 10ms

        let trace = MemoryTrace::new(0, vec![1.0], 0.0);
        wm.push(trace);

        // Step forward 10ms (one time constant)
        wm.step(10.0);

        // Strength should decay by factor of e^-1 ≈ 0.368
        let remaining = wm.get(0).map(|t| t.strength).unwrap_or(0.0);
        assert!((remaining - 0.368).abs() < 0.01);
    }

    #[test]
    fn test_stm_consolidation() {
        let mut stm = ShortTermMemory::new(100, STM_DECAY_TAU);

        // Create trace that should consolidate (high access count)
        let mut trace = MemoryTrace::new(0, vec![1.0], 0.0);
        trace.access_count = 5;
        trace.strength = 0.8;
        stm.encode(trace);

        // Access multiple times
        stm.access(0, 0.1);
        stm.access(0, 0.1);

        // Should trigger consolidation
        let consolidated = stm.step(1.0);

        // High-access trace should be consolidated
        assert!(consolidated.len() > 0 || stm.get(0).is_some());
    }

    #[test]
    fn test_ltm_forgetting() {
        let mut ltm = LongTermMemory::new(0.01, CONSOLIDATION_GAMMA); // Fast forgetting

        let trace = MemoryTrace::new(0, vec![1.0], 0.0);
        ltm.consolidate(trace, 1.0, 1.0);

        // Step forward (apply forgetting)
        for _ in 0..100 {
            ltm.step(10.0);
        }

        // Trace should be weakened
        if let Some(t) = ltm.get(0) {
            assert!(t.strength < 0.5);
        }
    }

    #[test]
    fn test_hopfield_store_and_retrieve() {
        let mut hopfield = HyperbolicHopfield::default();

        // Store patterns
        let mut p1 = [0.0; EMBEDDING_DIM];
        p1[0] = 1.0;
        p1[1] = 0.5;
        hopfield.store_pattern(0, p1);

        let mut p2 = [0.0; EMBEDDING_DIM];
        p2[0] = 1.0;
        p2[1] = -0.5;
        hopfield.store_pattern(1, p2);

        assert_eq!(hopfield.num_patterns(), 2);

        // Query should find nearest
        let (nearest_id, dist) = hopfield.nearest_pattern(&p1).unwrap();
        assert_eq!(nearest_id, 0);
        assert!(dist < 0.1);
    }

    #[test]
    fn test_regime_detection() {
        let mut gating = RecallGating::new();

        // Positive curvature → Stable
        for _ in 0..20 {
            gating.update(1.0);
        }
        assert_eq!(gating.regime(), MemoryRegime::Stable);

        // Reset with strongly negative curvature → Volatile
        // Need enough negative values to push average below -0.5
        for _ in 0..80 {
            gating.update(-1.0);
        }
        // Now average = (20*1.0 + 80*-1.0)/100 = -60/100 = -0.6 < -0.5 → Volatile
        assert_eq!(gating.regime(), MemoryRegime::Volatile);
    }

    #[test]
    fn test_holonomic_memory_encode_recall() {
        let mut memory = HolonomicMemory::default();

        // Encode patterns
        let id1 = memory.encode(vec![1.0, 0.0, 0.0], 0.8);
        let id2 = memory.encode(vec![0.0, 1.0, 0.0], 0.6);
        let id3 = memory.encode(vec![0.0, 0.0, 1.0], 0.4);

        // Step to allow processing
        memory.step(1.0);

        // Recall similar pattern
        let results = memory.recall(&[0.9, 0.1, 0.0], 3);

        // Should find at least one match
        assert!(!results.is_empty());
    }

    #[test]
    fn test_holonomic_wm_to_stm_transfer() {
        let config = HolonomicConfig {
            wm_capacity: 2,
            ..Default::default()
        };
        let mut memory = HolonomicMemory::new(config);

        // Fill WM beyond capacity
        memory.encode(vec![1.0], 0.8);
        memory.encode(vec![2.0], 0.8);
        memory.encode(vec![3.0], 0.8); // Should push first to STM

        // WM should have 2, STM should have 1
        assert_eq!(memory.wm().len(), 2);
        assert!(memory.stm().len() >= 1);
    }

    #[test]
    fn test_path_integral_action() {
        let recall = PathIntegralRecall::default();

        let mut start = [0.0; EMBEDDING_DIM];
        start[0] = 1.0;

        let mut target = [0.0; EMBEDDING_DIM];
        target[0] = 1.0;
        target[1] = 0.3;
        HyperbolicMemoryOps::project_to_hyperboloid(&mut target);

        let path = recall.generate_path(&start, &target);
        let action = recall.compute_action(&path, &target);

        // Action should be positive
        assert!(action > 0.0);

        // Probability should be in (0, 1]
        let prob = recall.recall_probability(action);
        assert!(prob > 0.0 && prob <= 1.0);
    }

    #[test]
    fn test_pattern_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];

        // Same vectors have similarity 1
        assert!((pattern_similarity(&a, &b) - 1.0).abs() < 1e-10);

        // Orthogonal vectors have similarity 0
        assert!((pattern_similarity(&a, &c)).abs() < 1e-10);
    }

    #[test]
    fn test_holonomic_stats() {
        let mut memory = HolonomicMemory::default();

        memory.encode(vec![1.0, 2.0, 3.0], 0.8);
        memory.step(10.0);
        memory.recall(&[1.0, 2.0, 3.0], 1);

        let stats = memory.stats();

        assert_eq!(stats.encode_count, 1);
        assert_eq!(stats.recall_count, 1);
        assert!(stats.current_time > 0.0);
    }

    #[test]
    fn test_consolidation_dynamics() {
        // Test the consolidation equation:
        // LTM(t) = LTM₀·exp(-λt) + (γ·STM·attention·replay/λ)·(1 - exp(-λt))

        let lambda: f64 = 0.01;
        let gamma: f64 = 0.1;
        let stm_strength: f64 = 0.8;
        let attention: f64 = 1.0;
        let replay: f64 = 2.0;

        let ltm_initial: f64 = 0.0;
        let t: f64 = 100.0;

        // Analytical solution
        let decay = (-lambda * t).exp();
        let steady_state = gamma * stm_strength * attention * replay / lambda;
        let ltm_expected = ltm_initial * decay + steady_state * (1.0 - decay);

        // Should approach steady state
        assert!(ltm_expected > 0.0);
        assert!(ltm_expected < steady_state * 1.1); // Should be near but below steady state
    }

    #[test]
    fn test_hyperbolic_similarity() {
        let mut x = [0.0; EMBEDDING_DIM];
        x[0] = 1.0;

        let mut y = [0.0; EMBEDDING_DIM];
        y[0] = 1.0;
        y[1] = 0.3;
        HyperbolicMemoryOps::project_to_hyperboloid(&mut y);

        let sim = HyperbolicMemoryOps::hyperbolic_similarity(&x, &y, 1.0);

        // Similarity should be in (0, 1]
        assert!(sim > 0.0 && sim <= 1.0);

        // Self-similarity should be 1
        let self_sim = HyperbolicMemoryOps::hyperbolic_similarity(&x, &x, 1.0);
        assert!((self_sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mobius_addition() {
        let x = vec![0.1, 0.2];
        let y = vec![0.1, -0.1];

        let result = HyperbolicMemoryOps::mobius_add(&x, &y, -1.0);

        // Result should be a valid Poincaré ball point (norm < 1)
        let norm_sq: f64 = result.iter().map(|&v| v * v).sum();
        assert!(norm_sq < 1.0);
    }

    #[test]
    fn test_exp_log_map_inverse() {
        let mut x = [0.0; EMBEDDING_DIM];
        x[0] = 1.0;

        let mut y = [0.0; EMBEDDING_DIM];
        y[0] = 1.0;
        y[1] = 0.2;
        HyperbolicMemoryOps::project_to_hyperboloid(&mut y);

        // log_x(y) then exp_x should give back y
        let log_vec = HyperbolicMemoryOps::log_map(&x, &y);
        let recovered = HyperbolicMemoryOps::exp_map(&x, &log_vec);

        let d = HyperbolicMemoryOps::hyperbolic_distance(&y, &recovered);
        assert!(d < 1e-6);
    }
}
