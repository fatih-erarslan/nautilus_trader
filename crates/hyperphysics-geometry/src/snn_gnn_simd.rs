//! # SNN ∩ GNN: Spatiotemporal Graph Processing with Spike Dynamics
//!
//! Implements event-driven, energy-efficient graph computation where nodes
//! communicate via temporal spike codes. Optimized with SIMD AVX2 intrinsics
//! for high-performance spike processing.
//!
//! ## Core Intersection Points
//!
//! 1. **Spike-Based Message Passing**: Traditional GNN message passing transformed
//!    into spiking dynamics where propagation delays = geodesic distances
//!
//! 2. **Temporal Credit Assignment**: STDP on graph edges for spike-timing
//!    dependent learning
//!
//! 3. **Event-Driven Computation**: Asynchronous, sparse computation triggered
//!    only when spikes occur
//!
//! ## SIMD Optimization Strategy
//!
//! - Batch processing of spike events using AVX2 256-bit vectors
//! - Vectorized membrane potential updates (8 neurons per instruction)
//! - SIMD-accelerated distance calculations and weight updates
//! - Cache-aligned data structures for optimal memory bandwidth
//!
//! ## References
//!
//! - Massa et al. (2022) "Graph Neural Networks" Nature Reviews Methods Primers
//! - Neftci et al. (2019) "Surrogate gradient learning in SNNs" IEEE Signal Proc
//! - Loihi: Intel's neuromorphic chip architecture
//! - Kollár et al. (2019) "Hyperbolic lattices in circuit QED" Nature 571:45-50

use crate::hyperbolic_snn::LorentzVec;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, VecDeque};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================================
// SIMD Constants and Alignment
// ============================================================================

/// Cache line size for alignment (64 bytes on modern CPUs)
pub const CACHE_LINE_SIZE: usize = 64;

/// SIMD vector width (8 f32 values in AVX2)
pub const SIMD_WIDTH: usize = 8;

/// Batch size for SIMD processing
pub const SIMD_BATCH_SIZE: usize = 256;

// ============================================================================
// Core Spiking Graph Node
// ============================================================================

/// Spiking graph node with LIF dynamics and SIMD-optimized memory layout
#[repr(C, align(32))]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikingGraphNode {
    // Leaky Integrate-and-Fire dynamics
    /// Membrane potential (mV)
    pub membrane_potential: f32,
    /// Spike threshold (mV)
    pub threshold: f32,
    /// Leak constant (1/τ_m)
    pub leak_constant: f32,
    /// Reset potential (mV)
    pub reset_potential: f32,

    // Graph connectivity with temporal delays
    /// Neighbor node IDs
    pub neighbors: Vec<u32>,
    /// Synaptic weights for each neighbor
    pub synaptic_weights: Vec<f32>,
    /// Axonal delays (ms) for each neighbor - proportional to hyperbolic distance
    pub axonal_delays: Vec<f32>,

    // Spike history for STDP
    /// Ring buffer of recent spike times
    pub spike_times: VecDeque<f64>,
    /// Maximum spike history size
    pub max_spike_history: usize,
    /// Last spike time
    pub last_spike: f64,

    // Refractory period
    /// Time until neuron can fire again
    pub refractory_until: f64,
    /// Refractory period duration (ms)
    pub refractory_period: f32,

    // Position on hyperbolic lattice
    /// Position in Lorentz coordinates
    pub position: LorentzVec,
    /// Layer in hyperbolic hierarchy
    pub layer: u32,

    // Statistics
    /// Total spike count
    pub spike_count: u64,
    /// Node ID
    pub id: u32,
}

impl SpikingGraphNode {
    /// Create new spiking graph node
    pub fn new(id: u32, position: LorentzVec, layer: u32) -> Self {
        Self {
            membrane_potential: -70.0,
            threshold: -55.0,
            leak_constant: 0.05, // τ_m ≈ 20ms
            reset_potential: -75.0,
            neighbors: Vec::new(),
            synaptic_weights: Vec::new(),
            axonal_delays: Vec::new(),
            spike_times: VecDeque::with_capacity(100),
            max_spike_history: 100,
            last_spike: f64::NEG_INFINITY,
            refractory_until: 0.0,
            refractory_period: 2.0,
            position,
            layer,
            spike_count: 0,
            id,
        }
    }

    /// Add neighbor connection with hyperbolic distance-based delay
    pub fn add_neighbor(&mut self, neighbor_id: u32, weight: f32, distance: f32, propagation_speed: f32) {
        self.neighbors.push(neighbor_id);
        self.synaptic_weights.push(weight);
        // Delay proportional to hyperbolic distance
        let delay = distance / propagation_speed;
        self.axonal_delays.push(delay);
    }

    /// Receive spike from neighbor
    #[inline]
    pub fn receive_spike(&mut self, arrival_time: f64, weight: f32) -> bool {
        if arrival_time < self.refractory_until {
            return false;
        }

        // EPSP: Exponential Post-Synaptic Potential
        let tau = 5.0; // ms
        let dt = (arrival_time - self.last_spike).max(0.0) as f32;
        let epsp = weight * (-dt / tau).exp();
        self.membrane_potential += epsp;

        // Threshold crossing → emit spike
        if self.membrane_potential > self.threshold {
            self.emit_spike(arrival_time);
            return true;
        }

        false
    }

    /// Emit spike and reset
    #[inline]
    fn emit_spike(&mut self, time: f64) {
        self.last_spike = time;
        self.spike_count += 1;
        self.membrane_potential = self.reset_potential;
        self.refractory_until = time + self.refractory_period as f64;

        // Record spike time for STDP
        self.spike_times.push_back(time);
        if self.spike_times.len() > self.max_spike_history {
            self.spike_times.pop_front();
        }
    }

    /// Check if neuron just spiked
    #[inline]
    pub fn just_spiked(&self, current_time: f64) -> bool {
        (current_time - self.last_spike).abs() < 0.1 // Within 0.1ms
    }

    /// STDP: Update weight based on spike timing correlations
    #[inline]
    pub fn stdp_update(&mut self, pre_spike_time: f64, post_spike_time: f64, neighbor_idx: usize) {
        if neighbor_idx >= self.synaptic_weights.len() {
            return;
        }

        let dt = post_spike_time - pre_spike_time;
        let tau_stdp = 20.0; // ms

        // Hebbian: pre before post → strengthen (LTP)
        // Anti-Hebbian: post before pre → weaken (LTD)
        let dw = if dt > 0.0 {
            0.01 * (-(dt as f32) / tau_stdp).exp() // LTP
        } else {
            -0.01 * ((dt as f32) / tau_stdp).exp() // LTD
        };

        self.synaptic_weights[neighbor_idx] = (self.synaptic_weights[neighbor_idx] + dw)
            .clamp(0.0, 1.0);
    }

    /// Update membrane potential with leak
    #[inline]
    pub fn leak_update(&mut self, dt: f32) {
        // dV/dt = -leak_constant * (V - V_rest)
        let resting = -70.0f32;
        self.membrane_potential += -self.leak_constant * (self.membrane_potential - resting) * dt;
    }
}

// ============================================================================
// Spike Event for Event-Driven Processing
// ============================================================================

/// Spike event for priority queue processing
#[derive(Debug, Clone, Copy)]
pub struct SpikeEvent {
    /// Event time
    pub time: f64,
    /// Source node ID
    pub source: u32,
    /// Target node ID
    pub target: u32,
    /// Synaptic weight
    pub weight: f32,
}

impl PartialEq for SpikeEvent {
    fn eq(&self, other: &Self) -> bool {
        self.time == other.time
    }
}

impl Eq for SpikeEvent {}

impl PartialOrd for SpikeEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SpikeEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: smaller time = higher priority
        other.time.partial_cmp(&self.time).unwrap_or(Ordering::Equal)
    }
}

// ============================================================================
// SIMD-Optimized Batch Data Structures
// ============================================================================

/// SIMD-aligned batch of membrane potentials for vectorized updates
#[repr(C, align(32))]
#[derive(Debug, Clone)]
pub struct SimdMembraneBatch {
    /// Membrane potentials (8 values per batch)
    pub potentials: [f32; SIMD_WIDTH],
    /// Thresholds
    pub thresholds: [f32; SIMD_WIDTH],
    /// Leak constants
    pub leak_constants: [f32; SIMD_WIDTH],
    /// Input currents accumulated
    pub input_currents: [f32; SIMD_WIDTH],
    /// Node IDs in this batch
    pub node_ids: [u32; SIMD_WIDTH],
    /// Valid mask (which slots are active)
    pub valid_mask: u8,
}

impl Default for SimdMembraneBatch {
    fn default() -> Self {
        Self {
            potentials: [-70.0; SIMD_WIDTH],
            thresholds: [-55.0; SIMD_WIDTH],
            leak_constants: [0.05; SIMD_WIDTH],
            input_currents: [0.0; SIMD_WIDTH],
            node_ids: [0; SIMD_WIDTH],
            valid_mask: 0,
        }
    }
}

impl SimdMembraneBatch {
    /// SIMD-optimized membrane potential update
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn update_avx2(&mut self, dt: f32) -> u8 {
        // Load vectors
        let v_potential = _mm256_loadu_ps(self.potentials.as_ptr());
        let v_threshold = _mm256_loadu_ps(self.thresholds.as_ptr());
        let v_leak = _mm256_loadu_ps(self.leak_constants.as_ptr());
        let v_input = _mm256_loadu_ps(self.input_currents.as_ptr());
        let v_dt = _mm256_set1_ps(dt);
        let v_resting = _mm256_set1_ps(-70.0);

        // LIF update: V += (-leak * (V - V_rest) + I) * dt
        let v_diff = _mm256_sub_ps(v_potential, v_resting);
        let v_leak_term = _mm256_mul_ps(v_leak, v_diff);
        let v_total = _mm256_sub_ps(v_input, v_leak_term);
        let v_delta = _mm256_mul_ps(v_total, v_dt);
        let v_new = _mm256_add_ps(v_potential, v_delta);

        // Store updated potentials
        _mm256_storeu_ps(self.potentials.as_mut_ptr(), v_new);

        // Check threshold crossing (return bitmask of neurons that spiked)
        let v_crossed = _mm256_cmp_ps(v_new, v_threshold, _CMP_GT_OQ);
        let spike_mask = _mm256_movemask_ps(v_crossed) as u8;

        // Reset input currents
        let v_zero = _mm256_setzero_ps();
        _mm256_storeu_ps(self.input_currents.as_mut_ptr(), v_zero);

        spike_mask & self.valid_mask
    }

    /// Fallback scalar update
    pub fn update_scalar(&mut self, dt: f32) -> u8 {
        let mut spike_mask = 0u8;
        let resting = -70.0f32;

        for i in 0..SIMD_WIDTH {
            if (self.valid_mask >> i) & 1 == 1 {
                let leak_term = self.leak_constants[i] * (self.potentials[i] - resting);
                let delta = (self.input_currents[i] - leak_term) * dt;
                self.potentials[i] += delta;
                self.input_currents[i] = 0.0;

                if self.potentials[i] > self.thresholds[i] {
                    spike_mask |= 1 << i;
                }
            }
        }

        spike_mask
    }

    /// Add input current to specific slot
    #[inline]
    pub fn add_input(&mut self, slot: usize, current: f32) {
        if slot < SIMD_WIDTH {
            self.input_currents[slot] += current;
        }
    }
}

// ============================================================================
// SIMD Distance Calculations
// ============================================================================

/// SIMD-optimized hyperbolic distance calculations
#[repr(C, align(32))]
pub struct SimdDistanceCalculator {
    /// Batch of source positions (t, x, y, z)
    pub source_t: [f32; SIMD_WIDTH],
    pub source_x: [f32; SIMD_WIDTH],
    pub source_y: [f32; SIMD_WIDTH],
    pub source_z: [f32; SIMD_WIDTH],
}

impl SimdDistanceCalculator {
    /// Create new distance calculator
    pub fn new() -> Self {
        Self {
            source_t: [1.0; SIMD_WIDTH],
            source_x: [0.0; SIMD_WIDTH],
            source_y: [0.0; SIMD_WIDTH],
            source_z: [0.0; SIMD_WIDTH],
        }
    }

    /// Load source positions from nodes
    pub fn load_sources(&mut self, nodes: &[SpikingGraphNode], indices: &[usize]) {
        for (i, &idx) in indices.iter().enumerate().take(SIMD_WIDTH) {
            if idx < nodes.len() {
                self.source_t[i] = nodes[idx].position.t as f32;
                self.source_x[i] = nodes[idx].position.x as f32;
                self.source_y[i] = nodes[idx].position.y as f32;
                self.source_z[i] = nodes[idx].position.z as f32;
            }
        }
    }

    /// SIMD-optimized Minkowski inner product batch
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn minkowski_inner_avx2(
        &self,
        target_t: f32,
        target_x: f32,
        target_y: f32,
        target_z: f32,
    ) -> [f32; SIMD_WIDTH] {
        // Load source vectors
        let v_src_t = _mm256_loadu_ps(self.source_t.as_ptr());
        let v_src_x = _mm256_loadu_ps(self.source_x.as_ptr());
        let v_src_y = _mm256_loadu_ps(self.source_y.as_ptr());
        let v_src_z = _mm256_loadu_ps(self.source_z.as_ptr());

        // Broadcast target values
        let v_tgt_t = _mm256_set1_ps(target_t);
        let v_tgt_x = _mm256_set1_ps(target_x);
        let v_tgt_y = _mm256_set1_ps(target_y);
        let v_tgt_z = _mm256_set1_ps(target_z);

        // Minkowski inner: -t1*t2 + x1*x2 + y1*y2 + z1*z2
        let v_neg_one = _mm256_set1_ps(-1.0);

        // -t1*t2
        let v_t_prod = _mm256_mul_ps(v_src_t, v_tgt_t);
        let v_neg_t = _mm256_mul_ps(v_t_prod, v_neg_one);

        // + x1*x2 (using FMA)
        let v_result = _mm256_fmadd_ps(v_src_x, v_tgt_x, v_neg_t);

        // + y1*y2
        let v_result = _mm256_fmadd_ps(v_src_y, v_tgt_y, v_result);

        // + z1*z2
        let v_result = _mm256_fmadd_ps(v_src_z, v_tgt_z, v_result);

        let mut output = [0.0f32; SIMD_WIDTH];
        _mm256_storeu_ps(output.as_mut_ptr(), v_result);
        output
    }

    /// Fallback scalar Minkowski inner product
    pub fn minkowski_inner_scalar(
        &self,
        target_t: f32,
        target_x: f32,
        target_y: f32,
        target_z: f32,
    ) -> [f32; SIMD_WIDTH] {
        let mut output = [0.0f32; SIMD_WIDTH];

        for i in 0..SIMD_WIDTH {
            output[i] = -self.source_t[i] * target_t
                + self.source_x[i] * target_x
                + self.source_y[i] * target_y
                + self.source_z[i] * target_z;
        }

        output
    }

    /// Compute hyperbolic distances: d = acosh(-inner)
    pub fn hyperbolic_distances(
        &self,
        target_t: f32,
        target_x: f32,
        target_y: f32,
        target_z: f32,
    ) -> [f32; SIMD_WIDTH] {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                let inner = unsafe {
                    self.minkowski_inner_avx2(target_t, target_x, target_y, target_z)
                };
                let mut distances = [0.0f32; SIMD_WIDTH];
                for i in 0..SIMD_WIDTH {
                    let clamped = (-inner[i]).max(1.0);
                    distances[i] = clamped.acosh();
                }
                return distances;
            }
        }

        let inner = self.minkowski_inner_scalar(target_t, target_x, target_y, target_z);
        let mut distances = [0.0f32; SIMD_WIDTH];
        for i in 0..SIMD_WIDTH {
            let clamped = (-inner[i]).max(1.0);
            distances[i] = clamped.acosh();
        }
        distances
    }
}

impl Default for SimdDistanceCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Event-Driven Graph Processor
// ============================================================================

/// Event-driven spiking graph processor with SIMD optimization
pub struct EventDrivenGraphProcessor {
    /// Graph nodes
    pub nodes: Vec<SpikingGraphNode>,
    /// Event queue (priority queue sorted by time)
    pub event_queue: BinaryHeap<SpikeEvent>,
    /// Current simulation time
    pub current_time: f64,
    /// SIMD batch for membrane updates
    pub simd_batch: SimdMembraneBatch,
    /// SIMD distance calculator
    pub distance_calc: SimdDistanceCalculator,
    /// Spike history for avalanche detection
    pub recent_spikes: VecDeque<(f64, u32)>,
    /// Configuration
    pub config: ProcessorConfig,
    /// Statistics
    pub stats: ProcessorStats,
}

/// Processor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessorConfig {
    /// Maximum events to process per step
    pub max_events_per_step: usize,
    /// Time step for batch updates (ms)
    pub batch_dt: f32,
    /// STDP enabled
    pub enable_stdp: bool,
    /// Avalanche detection window (ms)
    pub avalanche_window: f64,
    /// Propagation speed (distance/ms)
    pub propagation_speed: f32,
}

impl Default for ProcessorConfig {
    fn default() -> Self {
        Self {
            max_events_per_step: 1000,
            batch_dt: 0.1,
            enable_stdp: true,
            avalanche_window: 10.0,
            propagation_speed: 1.0,
        }
    }
}

/// Processing statistics
#[derive(Debug, Clone, Default)]
pub struct ProcessorStats {
    /// Total events processed
    pub total_events: u64,
    /// Total spikes generated
    pub total_spikes: u64,
    /// Current avalanche size
    pub current_avalanche_size: usize,
    /// Avalanche sizes history
    pub avalanche_sizes: Vec<usize>,
    /// Average spike rate (Hz)
    pub avg_spike_rate: f64,
    /// SIMD utilization (0-1)
    pub simd_utilization: f64,
}

impl EventDrivenGraphProcessor {
    /// Create new processor with given nodes
    pub fn new(nodes: Vec<SpikingGraphNode>, config: ProcessorConfig) -> Self {
        Self {
            nodes,
            event_queue: BinaryHeap::new(),
            current_time: 0.0,
            simd_batch: SimdMembraneBatch::default(),
            distance_calc: SimdDistanceCalculator::new(),
            recent_spikes: VecDeque::with_capacity(1000),
            config,
            stats: ProcessorStats::default(),
        }
    }

    /// Create processor from hyperbolic lattice
    pub fn from_lattice(
        lattice: &crate::adversarial_lattice::AdversarialLattice,
        config: ProcessorConfig,
    ) -> Self {
        let mut nodes = Vec::with_capacity(lattice.sentries.len());

        // Create nodes from sentry positions
        for sentry in &lattice.sentries {
            let position = LorentzVec::from_hyperboloid(&sentry.position);
            nodes.push(SpikingGraphNode::new(
                sentry.id as u32,
                position,
                sentry.layer as u32,
            ));
        }

        // Create connections based on lattice topology
        for sentry in &lattice.sentries {
            for &neighbor_id in &sentry.neighbors {
                if neighbor_id < nodes.len() {
                    let source_pos = &nodes[sentry.id].position;
                    let target_pos = &nodes[neighbor_id].position;
                    let distance = source_pos.hyperbolic_distance(target_pos) as f32;

                    nodes[sentry.id].add_neighbor(
                        neighbor_id as u32,
                        0.5, // Initial weight
                        distance,
                        config.propagation_speed,
                    );
                }
            }
        }

        Self::new(nodes, config)
    }

    /// Process single step (event-driven)
    pub fn step(&mut self) -> Vec<u32> {
        let mut spiked_nodes = Vec::new();

        if let Some(event) = self.event_queue.pop() {
            self.current_time = event.time;
            self.stats.total_events += 1;

            // Deliver spike to target
            let target_idx = event.target as usize;
            if target_idx < self.nodes.len() {
                let spiked = self.nodes[target_idx].receive_spike(event.time, event.weight);

                if spiked {
                    self.stats.total_spikes += 1;
                    spiked_nodes.push(event.target);

                    // Record for avalanche detection
                    self.recent_spikes.push_back((event.time, event.target));

                    // Propagate spike
                    self.propagate_spike(event.target);

                    // STDP update
                    if self.config.enable_stdp {
                        self.apply_stdp(event.source, event.target, event.time);
                    }
                }
            }

            // Clean old spikes from history
            while let Some(&(t, _)) = self.recent_spikes.front() {
                if self.current_time - t > self.config.avalanche_window {
                    self.recent_spikes.pop_front();
                } else {
                    break;
                }
            }
        }

        spiked_nodes
    }

    /// Process multiple events in batch
    pub fn step_batch(&mut self, max_events: usize) -> Vec<u32> {
        let mut spiked_nodes = Vec::new();
        let events_to_process = max_events.min(self.event_queue.len());

        for _ in 0..events_to_process {
            let batch_spiked = self.step();
            spiked_nodes.extend(batch_spiked);
        }

        spiked_nodes
    }

    /// Propagate spike from source to all neighbors
    fn propagate_spike(&mut self, source_id: u32) {
        let source_idx = source_id as usize;
        if source_idx >= self.nodes.len() {
            return;
        }

        // Clone neighbor data to avoid borrow issues
        let neighbors: Vec<(u32, f32, f32)> = self.nodes[source_idx]
            .neighbors
            .iter()
            .zip(self.nodes[source_idx].synaptic_weights.iter())
            .zip(self.nodes[source_idx].axonal_delays.iter())
            .map(|((&n, &w), &d)| (n, w, d))
            .collect();

        for (target_id, weight, delay) in neighbors {
            self.event_queue.push(SpikeEvent {
                time: self.current_time + delay as f64,
                source: source_id,
                target: target_id,
                weight,
            });
        }
    }

    /// Apply STDP learning rule
    fn apply_stdp(&mut self, pre_id: u32, post_id: u32, post_spike_time: f64) {
        let pre_idx = pre_id as usize;
        let post_idx = post_id as usize;

        if pre_idx >= self.nodes.len() || post_idx >= self.nodes.len() {
            return;
        }

        // Get pre-synaptic spike time
        if let Some(&pre_spike_time) = self.nodes[pre_idx].spike_times.back() {
            // Find neighbor index
            if let Some(neighbor_idx) = self.nodes[pre_idx]
                .neighbors
                .iter()
                .position(|&n| n == post_id)
            {
                self.nodes[pre_idx].stdp_update(pre_spike_time, post_spike_time, neighbor_idx);
            }
        }
    }

    /// SIMD-optimized batch membrane update for all nodes
    #[cfg(target_arch = "x86_64")]
    pub fn batch_update_simd(&mut self, dt: f32) -> Vec<u32> {
        let mut spiked_nodes = Vec::new();

        // Process nodes in SIMD batches
        for batch_start in (0..self.nodes.len()).step_by(SIMD_WIDTH) {
            let batch_end = (batch_start + SIMD_WIDTH).min(self.nodes.len());
            let batch_size = batch_end - batch_start;

            // Load batch data
            self.simd_batch.valid_mask = (1u8 << batch_size) - 1;
            for (i, node) in self.nodes[batch_start..batch_end].iter().enumerate() {
                self.simd_batch.potentials[i] = node.membrane_potential;
                self.simd_batch.thresholds[i] = node.threshold;
                self.simd_batch.leak_constants[i] = node.leak_constant;
                self.simd_batch.input_currents[i] = 0.0;
                self.simd_batch.node_ids[i] = node.id;
            }

            // SIMD update
            let spike_mask = if is_x86_feature_detected!("avx2") {
                unsafe { self.simd_batch.update_avx2(dt) }
            } else {
                self.simd_batch.update_scalar(dt)
            };

            // Process spikes and store results - collect spiked IDs first
            let mut batch_spiked = Vec::new();
            for (i, node) in self.nodes[batch_start..batch_end].iter_mut().enumerate() {
                node.membrane_potential = self.simd_batch.potentials[i];

                if (spike_mask >> i) & 1 == 1 {
                    let node_id = self.simd_batch.node_ids[i];
                    node.emit_spike(self.current_time);
                    batch_spiked.push(node_id);
                }
            }

            // Propagate spikes after releasing the borrow
            for node_id in &batch_spiked {
                self.propagate_spike(*node_id);
            }
            spiked_nodes.extend(batch_spiked);
        }

        spiked_nodes
    }

    /// Fallback scalar batch update
    pub fn batch_update_scalar(&mut self, dt: f32) -> Vec<u32> {
        let mut spiked_nodes = Vec::new();

        for node in &mut self.nodes {
            node.leak_update(dt);

            if node.membrane_potential > node.threshold {
                let node_id = node.id;
                node.emit_spike(self.current_time);
                spiked_nodes.push(node_id);
            }
        }

        // Propagate spikes
        for &node_id in &spiked_nodes {
            self.propagate_spike(node_id);
        }

        spiked_nodes
    }

    /// Run simulation for given duration
    pub fn run(&mut self, duration: f64) -> SimulationResult {
        let start_time = self.current_time;
        let end_time = start_time + duration;

        // Include any spikes that occurred before run() (e.g., from inject_spike)
        let mut spike_history: Vec<(f64, u32)> = self.recent_spikes.iter().cloned().collect();

        while self.current_time < end_time {
            // Process events
            let max_events = self.config.max_events_per_step;
            for _ in 0..max_events {
                if self.event_queue.is_empty() {
                    break;
                }

                if let Some(event) = self.event_queue.peek() {
                    if event.time > end_time {
                        break;
                    }
                }

                let spiked = self.step();
                for node_id in spiked {
                    spike_history.push((self.current_time, node_id));
                }
            }

            // Batch update all nodes
            self.current_time += self.config.batch_dt as f64;

            #[cfg(target_arch = "x86_64")]
            let batch_spiked = self.batch_update_simd(self.config.batch_dt);

            #[cfg(not(target_arch = "x86_64"))]
            let batch_spiked = self.batch_update_scalar(self.config.batch_dt);

            for node_id in batch_spiked {
                spike_history.push((self.current_time, node_id));
            }
        }

        // Compute statistics
        let total_spikes = spike_history.len();
        let avg_rate = if duration > 0.0 && !self.nodes.is_empty() {
            (total_spikes as f64 / duration) / self.nodes.len() as f64 * 1000.0 // Hz
        } else {
            0.0
        };

        SimulationResult {
            duration,
            total_spikes,
            spike_history,
            avg_spike_rate: avg_rate,
            final_time: self.current_time,
        }
    }

    /// Inject external spike to specific node
    pub fn inject_spike(&mut self, node_id: u32, time: f64, current: f32) {
        let idx = node_id as usize;
        if idx < self.nodes.len() {
            self.nodes[idx].membrane_potential += current;

            if self.nodes[idx].membrane_potential > self.nodes[idx].threshold {
                self.nodes[idx].emit_spike(time);
                self.stats.total_spikes += 1; // Track injected spikes
                self.recent_spikes.push_back((time, node_id));
                self.propagate_spike(node_id);
            }
        }
    }

    /// Detect neuronal avalanches (for SOC analysis)
    pub fn detect_avalanche(&self) -> AvalancheInfo {
        if self.recent_spikes.is_empty() {
            return AvalancheInfo::default();
        }

        // Count spikes in avalanche window
        let size = self.recent_spikes.len();

        // Unique neurons involved
        let unique_neurons: std::collections::HashSet<u32> = self.recent_spikes
            .iter()
            .map(|(_, n)| *n)
            .collect();

        let duration = if size > 1 {
            self.recent_spikes.back().unwrap().0 - self.recent_spikes.front().unwrap().0
        } else {
            0.0
        };

        AvalancheInfo {
            size,
            unique_neurons: unique_neurons.len(),
            duration,
            start_time: self.recent_spikes.front().map(|(t, _)| *t).unwrap_or(0.0),
        }
    }

    /// Get current network state
    pub fn get_state(&self) -> NetworkState {
        let mut membrane_potentials = Vec::with_capacity(self.nodes.len());
        let mut spike_counts = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            membrane_potentials.push(node.membrane_potential);
            spike_counts.push(node.spike_count);
        }

        NetworkState {
            time: self.current_time,
            membrane_potentials,
            spike_counts,
            pending_events: self.event_queue.len(),
        }
    }
}

/// Simulation result
#[derive(Debug, Clone)]
pub struct SimulationResult {
    /// Total duration
    pub duration: f64,
    /// Total spikes
    pub total_spikes: usize,
    /// Spike history (time, node_id)
    pub spike_history: Vec<(f64, u32)>,
    /// Average spike rate
    pub avg_spike_rate: f64,
    /// Final simulation time
    pub final_time: f64,
}

/// Avalanche information
#[derive(Debug, Clone, Default)]
pub struct AvalancheInfo {
    /// Number of spikes in avalanche
    pub size: usize,
    /// Number of unique neurons involved
    pub unique_neurons: usize,
    /// Duration of avalanche
    pub duration: f64,
    /// Start time
    pub start_time: f64,
}

/// Network state snapshot
#[derive(Debug, Clone)]
pub struct NetworkState {
    /// Current time
    pub time: f64,
    /// All membrane potentials
    pub membrane_potentials: Vec<f32>,
    /// Spike counts per node
    pub spike_counts: Vec<u64>,
    /// Number of pending events
    pub pending_events: usize,
}

// ============================================================================
// Enhanced Self-Organized Criticality Analysis
// ============================================================================

/// Self-Organized Criticality (SOC) analyzer for spiking networks
///
/// Monitors branching ratio σ and avalanche size distribution P(s) ~ s^{-τ}
/// for criticality analysis. At the critical point σ = 1 and τ ≈ 3/2.
///
/// References:
/// - Beggs & Plenz (2003) "Neuronal Avalanches in Neocortical Circuits"
/// - Shew & Plenz (2013) "The Functional Benefits of Criticality"
#[derive(Debug, Clone)]
pub struct SOCAnalyzer {
    /// Target branching ratio (σ = 1 at criticality)
    pub sigma_target: f64,
    /// Measured branching ratio
    pub sigma_measured: f64,
    /// Avalanche size distribution (size -> count)
    avalanche_sizes: std::collections::HashMap<usize, usize>,
    /// Current avalanche spike count
    current_avalanche_size: usize,
    /// Is avalanche currently active
    avalanche_active: bool,
    /// Last spike time
    last_spike_time: f64,
    /// Inter-avalanche interval threshold (ms)
    iai_threshold: f64,
    /// Total spikes for branching ratio
    total_initiating_spikes: u64,
    /// Total triggered spikes
    total_triggered_spikes: u64,
    /// Estimated power-law exponent τ
    power_law_tau: f64,
    /// Kolmogorov-Smirnov statistic for power-law fit
    ks_statistic: f64,
    /// Adaptation rate for control
    adaptation_rate: f64,
}

impl SOCAnalyzer {
    /// Create new SOC analyzer
    pub fn new(iai_threshold: f64) -> Self {
        Self {
            sigma_target: 1.0,
            sigma_measured: 1.0,
            avalanche_sizes: std::collections::HashMap::new(),
            current_avalanche_size: 0,
            avalanche_active: false,
            last_spike_time: f64::NEG_INFINITY,
            iai_threshold,
            total_initiating_spikes: 0,
            total_triggered_spikes: 0,
            power_law_tau: 1.5,
            ks_statistic: 1.0,
            adaptation_rate: 0.01,
        }
    }

    /// Record a spike event
    pub fn record_spike(&mut self, time: f64, is_triggered: bool) {
        let dt = time - self.last_spike_time;

        if dt > self.iai_threshold && self.avalanche_active {
            // End current avalanche
            self.end_avalanche();
        }

        if !self.avalanche_active {
            // Start new avalanche
            self.avalanche_active = true;
            self.current_avalanche_size = 0;
            self.total_initiating_spikes += 1;
        }

        self.current_avalanche_size += 1;
        self.last_spike_time = time;

        if is_triggered {
            self.total_triggered_spikes += 1;
        }
    }

    /// End current avalanche and record size
    pub fn end_avalanche(&mut self) {
        if self.avalanche_active && self.current_avalanche_size > 0 {
            *self.avalanche_sizes.entry(self.current_avalanche_size).or_insert(0) += 1;
        }
        self.avalanche_active = false;
        self.current_avalanche_size = 0;

        // Update power-law estimate periodically
        if self.avalanche_sizes.len() >= 10 {
            self.estimate_power_law();
        }
    }

    /// Compute branching ratio σ = triggered / initiating
    pub fn update_sigma(&mut self) {
        if self.total_initiating_spikes > 0 {
            self.sigma_measured = self.total_triggered_spikes as f64 /
                                  self.total_initiating_spikes as f64;
        }
    }

    /// Estimate power-law exponent using Hill estimator
    fn estimate_power_law(&mut self) {
        // Collect avalanche sizes
        let mut sizes: Vec<f64> = Vec::new();
        for (&size, &count) in &self.avalanche_sizes {
            for _ in 0..count {
                sizes.push(size as f64);
            }
        }

        if sizes.len() < 10 {
            return;
        }

        // Sort in descending order
        sizes.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Use top 20% for Hill estimator (more robust)
        let n = sizes.len();
        let k = (n as f64 * 0.2).max(5.0) as usize;

        if k >= n {
            return;
        }

        let x_k = sizes[k];
        if x_k <= 1.0 {
            return;
        }

        // Hill estimator: α = 1 / (mean(log(x_i / x_k)))
        let sum_log: f64 = sizes[..k].iter()
            .map(|&x| (x / x_k).ln())
            .sum();

        if sum_log > 0.0 {
            self.power_law_tau = 1.0 + k as f64 / sum_log;
        }

        // Compute KS statistic for goodness of fit
        self.ks_statistic = self.compute_ks_statistic(&sizes);
    }

    /// Compute Kolmogorov-Smirnov statistic for power-law fit
    fn compute_ks_statistic(&self, sizes: &[f64]) -> f64 {
        if sizes.is_empty() {
            return 1.0;
        }

        let n = sizes.len();
        let x_min = sizes.iter().cloned().fold(f64::INFINITY, f64::min).max(1.0);

        // Empirical CDF vs theoretical power-law CDF
        let mut max_diff = 0.0f64;

        for (i, &x) in sizes.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n as f64;
            let theoretical_cdf = 1.0 - (x_min / x).powf(self.power_law_tau - 1.0);

            let diff = (empirical_cdf - theoretical_cdf).abs();
            max_diff = max_diff.max(diff);
        }

        max_diff
    }

    /// Get SOC factor for STDP modulation (near 1.0 at criticality)
    pub fn soc_factor(&self) -> f64 {
        // Factor that modulates learning to maintain criticality
        // > 1 if subcritical (need more activity)
        // < 1 if supercritical (need less activity)
        1.0 + self.adaptation_rate * (self.sigma_target - self.sigma_measured)
    }

    /// Check if system is near criticality
    pub fn is_critical(&self, sigma_tolerance: f64, tau_tolerance: f64) -> bool {
        let sigma_ok = (self.sigma_measured - self.sigma_target).abs() < sigma_tolerance;
        let tau_ok = (self.power_law_tau - 1.5).abs() < tau_tolerance;
        sigma_ok && tau_ok
    }

    /// Get avalanche size distribution
    pub fn size_distribution(&self) -> &std::collections::HashMap<usize, usize> {
        &self.avalanche_sizes
    }

    /// Get statistics summary
    pub fn stats(&self) -> SOCStats {
        SOCStats {
            sigma_measured: self.sigma_measured,
            sigma_target: self.sigma_target,
            power_law_tau: self.power_law_tau,
            ks_statistic: self.ks_statistic,
            total_avalanches: self.avalanche_sizes.values().sum(),
            largest_avalanche: self.avalanche_sizes.keys().copied().max().unwrap_or(0),
            is_critical: self.is_critical(0.1, 0.3),
        }
    }

    /// Reset analyzer
    pub fn reset(&mut self) {
        self.avalanche_sizes.clear();
        self.current_avalanche_size = 0;
        self.avalanche_active = false;
        self.total_initiating_spikes = 0;
        self.total_triggered_spikes = 0;
        self.sigma_measured = 1.0;
        self.power_law_tau = 1.5;
        self.ks_statistic = 1.0;
    }
}

/// SOC statistics summary
#[derive(Debug, Clone, Default)]
pub struct SOCStats {
    /// Measured branching ratio
    pub sigma_measured: f64,
    /// Target branching ratio
    pub sigma_target: f64,
    /// Power-law exponent τ
    pub power_law_tau: f64,
    /// KS statistic for power-law fit
    pub ks_statistic: f64,
    /// Total avalanches recorded
    pub total_avalanches: usize,
    /// Largest avalanche size
    pub largest_avalanche: usize,
    /// Whether system is near criticality
    pub is_critical: bool,
}

// ============================================================================
// Spike Code Types
// ============================================================================

/// Encoding scheme for spike trains
#[derive(Debug, Clone)]
pub enum SpikeCode {
    /// Rate coding: information in spike frequency
    Rate {
        window: f64,
        count: u32,
    },
    /// Temporal coding: information in precise timing
    Temporal {
        spike_times: Vec<f64>,
        reference: f64,
    },
    /// Population coding: distributed across neurons
    Population {
        firing_rates: Vec<f32>,
    },
}

impl SpikeCode {
    /// Decode spike train to continuous value
    pub fn decode(&self) -> f32 {
        match self {
            SpikeCode::Rate { window, count } => {
                (*count as f32) / (*window as f32 / 1000.0) // Hz
            }

            SpikeCode::Temporal { spike_times, reference } => {
                // First-spike latency coding
                if let Some(&first) = spike_times.first() {
                    let latency = first - reference;
                    1000.0 / latency.max(1.0) as f32
                } else {
                    0.0
                }
            }

            SpikeCode::Population { firing_rates } => {
                // Weighted sum (population vector)
                if firing_rates.is_empty() {
                    return 0.0;
                }
                firing_rates.iter().enumerate()
                    .map(|(i, &rate)| rate * (i as f32))
                    .sum::<f32>() / firing_rates.len() as f32
            }
        }
    }
}

// ============================================================================
// Energy Metrics
// ============================================================================

/// Energy consumption metrics for neuromorphic comparison
#[derive(Debug, Clone, Default)]
pub struct EnergyMetrics {
    /// Total spike count
    pub spike_count: u64,
    /// Total synaptic operations
    pub synaptic_ops: u64,
    /// Energy per spike (pJ) - typical for Loihi 2
    pub energy_per_spike: f32,
    /// Energy per synapse operation (pJ)
    pub energy_per_synapse: f32,
}

impl EnergyMetrics {
    /// Create with default neuromorphic energy values
    pub fn new() -> Self {
        Self {
            spike_count: 0,
            synaptic_ops: 0,
            energy_per_spike: 1.0,     // ~1 pJ on Loihi 2
            energy_per_synapse: 0.1,   // ~0.1 pJ
        }
    }

    /// Total energy consumption (pJ)
    pub fn total_energy(&self) -> f64 {
        (self.spike_count as f64) * (self.energy_per_spike as f64) +
        (self.synaptic_ops as f64) * (self.energy_per_synapse as f64)
    }

    /// Compare to dense GNN computation
    pub fn energy_vs_dense_gnn(&self, graph_size: usize, feature_dim: usize) -> f64 {
        let dense_ops = graph_size * feature_dim * graph_size; // O(N²d)
        let dense_energy = (dense_ops as f64) * 100.0; // ~100 pJ per MAC on GPU

        if dense_energy > 0.0 {
            self.total_energy() / dense_energy
        } else {
            0.0
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spiking_graph_node() {
        let mut node = SpikingGraphNode::new(0, LorentzVec::origin(), 0);
        node.add_neighbor(1, 0.5, 1.0, 1.0);

        assert_eq!(node.neighbors.len(), 1);
        assert_eq!(node.axonal_delays[0], 1.0);
    }

    #[test]
    fn test_spike_event_ordering() {
        let e1 = SpikeEvent { time: 1.0, source: 0, target: 1, weight: 0.5 };
        let e2 = SpikeEvent { time: 2.0, source: 0, target: 2, weight: 0.5 };

        // Min-heap: e1 should have higher priority (smaller time)
        assert!(e1 > e2);
    }

    #[test]
    fn test_simd_membrane_batch() {
        let mut batch = SimdMembraneBatch::default();
        batch.valid_mask = 0xFF;
        batch.input_currents = [10.0; SIMD_WIDTH];

        let spike_mask = batch.update_scalar(0.1);

        // With input current, some neurons should approach threshold
        // But one step isn't enough for -70 to reach -55
        assert_eq!(spike_mask, 0);
    }

    #[test]
    fn test_spike_receive() {
        let mut node = SpikingGraphNode::new(0, LorentzVec::origin(), 0);
        node.membrane_potential = -55.5; // Very close to threshold (-55.0)
        node.last_spike = 0.0; // Set valid last_spike to get proper EPSP decay

        // With membrane at -55.5, dt=1.0ms, tau=5.0ms:
        // EPSP = weight * exp(-dt/tau) = 1.0 * exp(-0.2) ≈ 0.818
        // New potential = -55.5 + 0.818 = -54.68 > -55.0 threshold
        let spiked = node.receive_spike(1.0, 1.0);
        assert!(spiked);
        assert_eq!(node.spike_count, 1);
    }

    #[test]
    fn test_stdp_update() {
        let mut node = SpikingGraphNode::new(0, LorentzVec::origin(), 0);
        node.add_neighbor(1, 0.5, 1.0, 1.0);

        let initial_weight = node.synaptic_weights[0];

        // Pre before post → LTP (strengthen)
        node.stdp_update(0.0, 10.0, 0);
        assert!(node.synaptic_weights[0] > initial_weight);

        // Post before pre → LTD (weaken)
        let weight_after_ltp = node.synaptic_weights[0];
        node.stdp_update(20.0, 10.0, 0);
        assert!(node.synaptic_weights[0] < weight_after_ltp);
    }

    #[test]
    fn test_event_driven_processor() {
        let mut node0 = SpikingGraphNode::new(0, LorentzVec::origin(), 0);
        let mut node1 = SpikingGraphNode::new(1, LorentzVec::from_spatial(0.5, 0.0, 0.0), 1);

        // Connect nodes for spike propagation
        node0.add_neighbor(1, 0.5, 1.0, 1.0);
        node1.add_neighbor(0, 0.5, 1.0, 1.0);

        // Set valid last_spike times for EPSP calculation
        node0.last_spike = 0.0;
        node1.last_spike = 0.0;

        let nodes = vec![node0, node1];

        let config = ProcessorConfig::default();
        let mut processor = EventDrivenGraphProcessor::new(nodes, config);

        // Inject strong spike to cross threshold
        // Membrane starts at -70, threshold at -55, so need +15 to spike
        processor.inject_spike(0, 0.0, 50.0);

        // Run simulation
        let result = processor.run(10.0);

        assert!(result.total_spikes > 0);
    }

    #[test]
    fn test_spike_code_decode() {
        let rate_code = SpikeCode::Rate { window: 100.0, count: 10 };
        let decoded = rate_code.decode();
        assert!((decoded - 100.0).abs() < 0.01); // 10 spikes / 0.1s = 100 Hz
    }

    #[test]
    fn test_energy_metrics() {
        let mut metrics = EnergyMetrics::new();
        metrics.spike_count = 1000;
        metrics.synaptic_ops = 10000;

        let energy = metrics.total_energy();
        assert!((energy - 2000.0).abs() < 0.01); // 1000 * 1.0 + 10000 * 0.1

        let ratio = metrics.energy_vs_dense_gnn(100, 64);
        assert!(ratio < 1.0); // SNN should be more efficient
    }

    #[test]
    fn test_distance_calculator() {
        let mut calc = SimdDistanceCalculator::new();

        // Set source at origin
        calc.source_t = [1.0; SIMD_WIDTH];
        calc.source_x = [0.0; SIMD_WIDTH];
        calc.source_y = [0.0; SIMD_WIDTH];
        calc.source_z = [0.0; SIMD_WIDTH];

        // Distance from origin to point
        let target = LorentzVec::from_spatial(0.5, 0.0, 0.0);
        let distances = calc.hyperbolic_distances(
            target.t as f32,
            target.x as f32,
            target.y as f32,
            target.z as f32,
        );

        // All distances should be the same
        for d in &distances {
            assert!(*d > 0.0);
            assert!((*d - distances[0]).abs() < 0.001);
        }
    }
}
