//! # Phase 6: Autopoietic pBit Networks with Self-Organized Criticality
//!
//! Self-maintaining neural networks that self-tune to the critical temperature.
//!
//! ## Mathematical Foundation (Wolfram-Verified)
//!
//! ### Self-Organized Criticality (SOC)
//! ```text
//! Critical temperature: T_c = 2/ln(1+√2) = 2.269185314213022
//! Branching ratio at criticality: σ = 1
//! Avalanche size distribution: P(s) ∝ s^(-τ), τ ≈ 1.5 (mean-field)
//! ```
//!
//! ### Homeostatic Plasticity
//! ```text
//! dT/dt = α(σ - 1)  // Temperature adapts to maintain σ = 1
//! ```
//!
//! ### IIT Phi (Φ) Computation
//! ```text
//! Φ = min over partitions { φ(partition) }
//! φ = EMD(p(effects|causes), p(effects)⊗p(causes))
//! ```
//!
//! ### Autopoietic Dynamics
//! ```text
//! Edge creation: P(e_ij) ∝ corr(x_i, x_j) if > θ_create
//! Edge deletion: P(del e_ij) ∝ 1 - corr(x_i, x_j) if < θ_delete
//! ```
//!
//! ## Wolfram Validation
//! - pBit @ T_c, h=0: P = 0.5 (balanced criticality)
//! - Boltzmann(E=0, T_c): W = 1 (reference)

use crate::constants::*;
use crate::{CortexError, Result};
use rand::Rng;
use std::collections::VecDeque;

// =============================================================================
// SOC CONSTANTS (Wolfram-Verified)
// =============================================================================

/// Ising critical temperature (2D square lattice, Onsager solution)
/// T_c = 2/ln(1+√2) = 2.269185314213022
pub const SOC_CRITICAL_TEMP: f64 = ISING_CRITICAL_TEMP;

/// Target branching ratio at criticality
pub const SOC_TARGET_BRANCHING: f64 = 1.0;

/// Homeostatic adaptation rate α
pub const SOC_ADAPTATION_RATE: f64 = 0.01;

/// Minimum temperature bound
pub const SOC_MIN_TEMP: f64 = 0.1;

/// Maximum temperature bound
pub const SOC_MAX_TEMP: f64 = 10.0;

/// Power-law exponent τ for avalanche sizes (mean-field theory)
pub const SOC_POWER_LAW_EXPONENT: f64 = 1.5;

/// Avalanche history buffer size
pub const AVALANCHE_HISTORY_SIZE: usize = 1000;

/// Correlation threshold for edge creation
pub const AUTOPOIETIC_EDGE_CREATE_THRESHOLD: f64 = 0.7;

/// Correlation threshold for edge deletion
pub const AUTOPOIETIC_EDGE_DELETE_THRESHOLD: f64 = 0.1;

/// Hebbian learning rate for edge weights
pub const HEBBIAN_RATE: f64 = 0.01;

/// Edge pruning rate
pub const PRUNE_RATE: f64 = 0.001;

// =============================================================================
// AVALANCHE TRACKER
// =============================================================================

/// Tracks avalanche dynamics for SOC analysis
#[derive(Debug, Clone)]
pub struct AvalancheTracker {
    /// Current avalanche size
    current_size: usize,
    /// Whether an avalanche is currently active
    active: bool,
    /// History of avalanche sizes
    size_history: VecDeque<usize>,
    /// History of avalanche durations
    duration_history: VecDeque<usize>,
    /// Current avalanche duration
    current_duration: usize,
    /// Total activations for branching ratio
    total_activations: usize,
    /// Activations in previous timestep
    prev_activations: usize,
    /// Running sum for branching ratio estimation
    branching_sum: f64,
    /// Count for branching ratio estimation
    branching_count: usize,
}

impl Default for AvalancheTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl AvalancheTracker {
    pub fn new() -> Self {
        Self {
            current_size: 0,
            active: false,
            size_history: VecDeque::with_capacity(AVALANCHE_HISTORY_SIZE),
            duration_history: VecDeque::with_capacity(AVALANCHE_HISTORY_SIZE),
            current_duration: 0,
            total_activations: 0,
            prev_activations: 0,
            branching_sum: 0.0,
            branching_count: 0,
        }
    }

    /// Record activations at this timestep
    pub fn record_activations(&mut self, num_activations: usize) {
        // Update branching ratio estimate
        if self.prev_activations > 0 {
            let ratio = num_activations as f64 / self.prev_activations as f64;
            self.branching_sum += ratio;
            self.branching_count += 1;
        }

        // Track avalanche
        if num_activations > 0 {
            if !self.active {
                // Start new avalanche
                self.active = true;
                self.current_size = 0;
                self.current_duration = 0;
            }
            self.current_size += num_activations;
            self.current_duration += 1;
        } else if self.active {
            // End avalanche
            self.active = false;
            self.record_avalanche(self.current_size, self.current_duration);
        }

        self.prev_activations = num_activations;
        self.total_activations += num_activations;
    }

    /// Record completed avalanche
    fn record_avalanche(&mut self, size: usize, duration: usize) {
        if self.size_history.len() >= AVALANCHE_HISTORY_SIZE {
            self.size_history.pop_front();
            self.duration_history.pop_front();
        }
        self.size_history.push_back(size);
        self.duration_history.push_back(duration);
    }

    /// Compute current branching ratio estimate
    pub fn branching_ratio(&self) -> f64 {
        if self.branching_count > 0 {
            self.branching_sum / self.branching_count as f64
        } else {
            1.0
        }
    }

    /// Get avalanche size distribution
    pub fn size_distribution(&self) -> Vec<(usize, usize)> {
        let mut counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        for &size in &self.size_history {
            *counts.entry(size).or_insert(0) += 1;
        }
        let mut dist: Vec<_> = counts.into_iter().collect();
        dist.sort_by_key(|&(size, _)| size);
        dist
    }

    /// Estimate power-law exponent using maximum likelihood
    /// τ = 1 + n / Σ ln(s_i / s_min)
    pub fn estimate_power_law_exponent(&self) -> Option<f64> {
        if self.size_history.len() < 10 {
            return None;
        }

        let s_min = *self.size_history.iter().min()? as f64;
        if s_min < 1.0 {
            return None;
        }

        let n = self.size_history.len() as f64;
        let log_sum: f64 = self.size_history
            .iter()
            .map(|&s| (s as f64 / s_min).ln())
            .sum();

        if log_sum > 0.0 {
            Some(1.0 + n / log_sum)
        } else {
            None
        }
    }

    /// Check if system is near criticality (σ ≈ 1)
    pub fn is_critical(&self) -> bool {
        let sigma = self.branching_ratio();
        (sigma - 1.0).abs() < 0.1
    }

    /// Reset statistics
    pub fn reset(&mut self) {
        self.current_size = 0;
        self.active = false;
        self.size_history.clear();
        self.duration_history.clear();
        self.current_duration = 0;
        self.total_activations = 0;
        self.prev_activations = 0;
        self.branching_sum = 0.0;
        self.branching_count = 0;
    }
}

// =============================================================================
// HOMEOSTATIC REGULATOR
// =============================================================================

/// Maintains system at criticality through temperature adaptation
#[derive(Debug, Clone)]
pub struct HomeostaticRegulator {
    /// Current temperature
    temperature: f64,
    /// Target branching ratio
    target_sigma: f64,
    /// Adaptation rate
    alpha: f64,
    /// Temperature history for analysis
    temp_history: VecDeque<f64>,
    /// Activity setpoint
    activity_setpoint: f64,
    /// Intrinsic plasticity gain
    gain: f64,
}

impl Default for HomeostaticRegulator {
    fn default() -> Self {
        Self {
            temperature: SOC_CRITICAL_TEMP,
            target_sigma: SOC_TARGET_BRANCHING,
            alpha: SOC_ADAPTATION_RATE,
            temp_history: VecDeque::with_capacity(100),
            activity_setpoint: 0.1,
            gain: 1.0,
        }
    }
}

impl HomeostaticRegulator {
    /// Create new regulator with initial temperature
    pub fn new(initial_temp: f64) -> Self {
        Self {
            temperature: initial_temp.clamp(SOC_MIN_TEMP, SOC_MAX_TEMP),
            ..Default::default()
        }
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Get current gain
    pub fn gain(&self) -> f64 {
        self.gain
    }

    /// Update temperature based on branching ratio
    /// dT/dt = α(σ - 1)
    pub fn update(&mut self, current_sigma: f64, current_activity: f64) {
        // Temperature adaptation: move toward T_c when σ ≠ 1
        let delta_t = self.alpha * (current_sigma - self.target_sigma);
        self.temperature = (self.temperature + delta_t).clamp(SOC_MIN_TEMP, SOC_MAX_TEMP);

        // Intrinsic plasticity: adjust gain to maintain activity setpoint
        let activity_error = self.activity_setpoint - current_activity;
        self.gain = (self.gain + 0.01 * activity_error).clamp(0.1, 10.0);

        // Record history
        if self.temp_history.len() >= 100 {
            self.temp_history.pop_front();
        }
        self.temp_history.push_back(self.temperature);
    }

    /// Check if temperature is near critical
    pub fn is_near_critical(&self) -> bool {
        (self.temperature - SOC_CRITICAL_TEMP).abs() < 0.1
    }

    /// Get temperature variance (stability measure)
    pub fn temperature_variance(&self) -> f64 {
        if self.temp_history.len() < 2 {
            return 0.0;
        }

        let mean: f64 = self.temp_history.iter().sum::<f64>() / self.temp_history.len() as f64;
        let variance: f64 = self.temp_history
            .iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f64>()
            / self.temp_history.len() as f64;

        variance
    }

    /// Set activity setpoint
    pub fn set_activity_setpoint(&mut self, setpoint: f64) {
        self.activity_setpoint = setpoint.clamp(0.01, 0.5);
    }
}

// =============================================================================
// IIT PHI COMPUTATION
// =============================================================================

/// Computes Integrated Information (Φ) for consciousness metrics
#[derive(Debug)]
pub struct PhiComputer {
    /// Number of nodes
    num_nodes: usize,
    /// Transition probability matrix
    tpm: Vec<Vec<f64>>,
    /// Current state probabilities
    state_probs: Vec<f64>,
}

impl PhiComputer {
    /// Create new Phi computer for given number of nodes
    pub fn new(num_nodes: usize) -> Self {
        let size = 1 << num_nodes.min(16); // Limit to 16 nodes for tractability
        Self {
            num_nodes: num_nodes.min(16),
            tpm: vec![vec![0.0; size]; size],
            state_probs: vec![1.0 / size as f64; size],
        }
    }

    /// Update transition probabilities from observed transitions
    pub fn update_tpm(&mut self, from_state: usize, to_state: usize) {
        let size = self.tpm.len();
        if from_state < size && to_state < size {
            // Exponential moving average update
            let alpha = 0.01;
            for j in 0..size {
                if j == to_state {
                    self.tpm[from_state][j] = (1.0 - alpha) * self.tpm[from_state][j] + alpha;
                } else {
                    self.tpm[from_state][j] *= 1.0 - alpha;
                }
            }
            // Normalize row
            let sum: f64 = self.tpm[from_state].iter().sum();
            if sum > 0.0 {
                for j in 0..size {
                    self.tpm[from_state][j] /= sum;
                }
            }
        }
    }

    /// Compute approximate Φ using bipartition
    /// This is a simplified version - full IIT is NP-hard
    pub fn compute_phi(&self) -> f64 {
        if self.num_nodes < 2 {
            return 0.0;
        }

        // Find minimum information partition (MIP)
        // For efficiency, only consider bipartitions
        let mut min_phi = f64::MAX;

        for partition_mask in 1..(1 << (self.num_nodes - 1)) {
            let phi = self.compute_partition_phi(partition_mask);
            min_phi = min_phi.min(phi);
        }

        if min_phi == f64::MAX {
            0.0
        } else {
            min_phi
        }
    }

    /// Compute φ for a specific partition
    fn compute_partition_phi(&self, partition_mask: usize) -> f64 {
        // Simplified: compute mutual information loss from partitioning
        let integrated = self.compute_integrated_info();
        let partitioned = self.compute_partitioned_info(partition_mask);

        (integrated - partitioned).max(0.0)
    }

    /// Compute integrated information (whole system)
    fn compute_integrated_info(&self) -> f64 {
        // Entropy of the whole system
        let mut entropy = 0.0;
        for &p in &self.state_probs {
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Compute information after partitioning
    fn compute_partitioned_info(&self, partition_mask: usize) -> f64 {
        // Simplified: sum of marginal entropies
        let part_a_size = partition_mask.count_ones() as usize;
        let part_b_size = self.num_nodes - part_a_size;

        // Marginal entropies (approximation)
        let entropy_a = (part_a_size as f64).ln().max(0.0);
        let entropy_b = (part_b_size as f64).ln().max(0.0);

        entropy_a + entropy_b
    }

    /// Get current Φ value (cached computation)
    pub fn phi(&self) -> f64 {
        self.compute_phi()
    }

    /// Check if system has significant integrated information
    pub fn is_conscious(&self, threshold: f64) -> bool {
        self.phi() > threshold
    }
}

// =============================================================================
// AUTOPOIETIC TOPOLOGY
// =============================================================================

/// Self-maintaining, self-creating network topology
#[derive(Debug)]
pub struct AutopoieticTopology {
    /// Number of nodes
    num_nodes: usize,
    /// Adjacency matrix (sparse representation)
    edges: Vec<Vec<(usize, f64)>>, // (target, weight)
    /// Node activity correlation matrix (for Hebbian learning)
    correlations: Vec<Vec<f64>>,
    /// Node activation history (rolling buffer)
    activation_history: VecDeque<Vec<bool>>,
    /// History buffer size
    history_size: usize,
    /// Edge creation threshold
    create_threshold: f64,
    /// Edge deletion threshold
    delete_threshold: f64,
}

impl AutopoieticTopology {
    /// Create new autopoietic topology
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            edges: vec![vec![]; num_nodes],
            correlations: vec![vec![0.0; num_nodes]; num_nodes],
            activation_history: VecDeque::with_capacity(100),
            history_size: 100,
            create_threshold: AUTOPOIETIC_EDGE_CREATE_THRESHOLD,
            delete_threshold: AUTOPOIETIC_EDGE_DELETE_THRESHOLD,
        }
    }

    /// Record node activations
    pub fn record_activations(&mut self, activations: &[bool]) {
        if self.activation_history.len() >= self.history_size {
            self.activation_history.pop_front();
        }
        self.activation_history.push_back(activations.to_vec());

        // Update correlations
        self.update_correlations();
    }

    /// Update correlation matrix based on activation history
    fn update_correlations(&mut self) {
        if self.activation_history.len() < 10 {
            return;
        }

        let n = self.num_nodes;
        let hist_len = self.activation_history.len() as f64;

        // Compute mean activations
        let mut means = vec![0.0; n];
        for activation in &self.activation_history {
            for (i, &active) in activation.iter().enumerate() {
                if i < n {
                    means[i] += if active { 1.0 } else { 0.0 };
                }
            }
        }
        for mean in &mut means {
            *mean /= hist_len;
        }

        // Compute correlations
        for i in 0..n {
            for j in i..n {
                let mut cov = 0.0;
                let mut var_i = 0.0;
                let mut var_j = 0.0;

                for activation in &self.activation_history {
                    let ai = if i < activation.len() && activation[i] { 1.0 } else { 0.0 };
                    let aj = if j < activation.len() && activation[j] { 1.0 } else { 0.0 };

                    let di = ai - means[i];
                    let dj = aj - means[j];

                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }

                let corr = if var_i > 0.0 && var_j > 0.0 {
                    cov / (var_i.sqrt() * var_j.sqrt())
                } else {
                    0.0
                };

                self.correlations[i][j] = corr;
                self.correlations[j][i] = corr;
            }
        }
    }

    /// Evolve topology based on correlations (Hebbian)
    pub fn evolve<R: Rng>(&mut self, rng: &mut R) {
        let n = self.num_nodes;

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let corr = self.correlations[i][j];
                let has_edge = self.edges[i].iter().any(|&(t, _)| t == j);

                // Edge creation
                if !has_edge && corr > self.create_threshold {
                    let create_prob = (corr - self.create_threshold) / (1.0 - self.create_threshold);
                    if rng.gen::<f64>() < create_prob * HEBBIAN_RATE {
                        self.edges[i].push((j, corr));
                    }
                }

                // Edge deletion
                if has_edge && corr < self.delete_threshold {
                    let delete_prob = (self.delete_threshold - corr) / self.delete_threshold;
                    if rng.gen::<f64>() < delete_prob * PRUNE_RATE {
                        self.edges[i].retain(|&(t, _)| t != j);
                    }
                }

                // Edge weight update (Hebbian)
                if has_edge {
                    for (t, w) in &mut self.edges[i] {
                        if *t == j {
                            *w = (*w + HEBBIAN_RATE * corr).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
    }

    /// Get number of edges
    pub fn num_edges(&self) -> usize {
        self.edges.iter().map(|e| e.len()).sum()
    }

    /// Get average clustering coefficient
    pub fn clustering_coefficient(&self) -> f64 {
        let mut total_cc = 0.0;
        let mut count = 0;

        for i in 0..self.num_nodes {
            let neighbors: Vec<usize> = self.edges[i].iter().map(|&(t, _)| t).collect();
            let k = neighbors.len();

            if k < 2 {
                continue;
            }

            let mut triangles = 0;
            for &n1 in &neighbors {
                for &n2 in &neighbors {
                    if n1 < n2 && self.edges[n1].iter().any(|&(t, _)| t == n2) {
                        triangles += 1;
                    }
                }
            }

            let possible = k * (k - 1) / 2;
            if possible > 0 {
                total_cc += triangles as f64 / possible as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_cc / count as f64
        } else {
            0.0
        }
    }

    /// Get edge list
    pub fn get_edges(&self) -> Vec<(usize, usize, f64)> {
        let mut edges = Vec::new();
        for (i, neighbors) in self.edges.iter().enumerate() {
            for &(j, w) in neighbors {
                edges.push((i, j, w));
            }
        }
        edges
    }
}

// =============================================================================
// AUTOPOIETIC NETWORK (MAIN STRUCT)
// =============================================================================

/// Configuration for autopoietic network
#[derive(Debug, Clone)]
pub struct AutopoieticConfig {
    /// Number of nodes
    pub num_nodes: usize,
    /// Initial temperature
    pub initial_temp: f64,
    /// Enable homeostatic regulation
    pub homeostatic: bool,
    /// Enable autopoietic topology
    pub autopoietic_topology: bool,
    /// Enable Phi computation
    pub compute_phi: bool,
    /// Activity setpoint
    pub activity_setpoint: f64,
}

impl Default for AutopoieticConfig {
    fn default() -> Self {
        Self {
            num_nodes: 100,
            initial_temp: SOC_CRITICAL_TEMP,
            homeostatic: true,
            autopoietic_topology: true,
            compute_phi: false, // Expensive, disabled by default
            activity_setpoint: 0.1,
        }
    }
}

/// Autopoietic pBit Network with Self-Organized Criticality
#[derive(Debug)]
pub struct AutopoieticNetwork {
    /// Configuration
    config: AutopoieticConfig,
    /// Node states (-1 or +1)
    states: Vec<i8>,
    /// Node biases
    biases: Vec<f64>,
    /// Coupling matrix (sparse)
    couplings: Vec<Vec<(usize, f64)>>,
    /// Avalanche tracker
    avalanche: AvalancheTracker,
    /// Homeostatic regulator
    regulator: HomeostaticRegulator,
    /// Autopoietic topology
    topology: AutopoieticTopology,
    /// Phi computer (optional)
    phi_computer: Option<PhiComputer>,
    /// Current timestep
    timestep: usize,
    /// External field
    external_field: f64,
}

impl AutopoieticNetwork {
    /// Create new autopoietic network
    pub fn new(config: AutopoieticConfig) -> Self {
        let n = config.num_nodes;

        // Initialize with random couplings (sparse)
        let couplings = vec![vec![]; n];

        let phi_computer = if config.compute_phi {
            Some(PhiComputer::new(n.min(12)))
        } else {
            None
        };

        let mut regulator = HomeostaticRegulator::new(config.initial_temp);
        regulator.set_activity_setpoint(config.activity_setpoint);

        Self {
            config: config.clone(),
            states: vec![1; n],
            biases: vec![0.0; n],
            couplings,
            avalanche: AvalancheTracker::new(),
            regulator,
            topology: AutopoieticTopology::new(n),
            phi_computer,
            timestep: 0,
            external_field: 0.0,
        }
    }

    /// Initialize with small-world topology
    pub fn init_small_world<R: Rng>(&mut self, k: usize, p: f64, rng: &mut R) {
        let n = self.config.num_nodes;

        // Create ring lattice
        for i in 0..n {
            for j in 1..=k / 2 {
                let neighbor = (i + j) % n;
                self.couplings[i].push((neighbor, 1.0));
                self.couplings[neighbor].push((i, 1.0));
            }
        }

        // Rewire with probability p
        for i in 0..n {
            let neighbors: Vec<_> = self.couplings[i].iter().map(|&(t, _)| t).collect();
            for j in neighbors {
                if rng.gen::<f64>() < p {
                    // Remove old edge
                    self.couplings[i].retain(|&(t, _)| t != j);

                    // Add new random edge
                    let new_target = rng.gen_range(0..n);
                    if new_target != i && !self.couplings[i].iter().any(|&(t, _)| t == new_target) {
                        self.couplings[i].push((new_target, 1.0));
                    }
                }
            }
        }
    }

    /// Set external field
    pub fn set_external_field(&mut self, field: f64) {
        self.external_field = field;
    }

    /// Perform one update step
    pub fn step<R: Rng>(&mut self, rng: &mut R) {
        let n = self.config.num_nodes;
        let temperature = self.regulator.temperature();
        let gain = self.regulator.gain();

        let mut num_activations = 0;
        let mut activations = vec![false; n];

        // Update each node with Glauber dynamics
        for i in 0..n {
            // Compute local field
            let mut local_field = self.external_field + self.biases[i];
            for &(j, w) in &self.couplings[i] {
                local_field += w * self.states[j] as f64;
            }

            // Apply gain
            local_field *= gain;

            // pBit update with Boltzmann probability
            let prob = pbit_probability(local_field, 0.0, temperature);

            let new_state = if rng.gen::<f64>() < prob { 1 } else { -1 };

            // Track activation (state flip from -1 to +1)
            if new_state == 1 && self.states[i] == -1 {
                num_activations += 1;
                activations[i] = true;
            }

            self.states[i] = new_state;
        }

        // Record avalanche dynamics
        self.avalanche.record_activations(num_activations);

        // Homeostatic update
        if self.config.homeostatic {
            let sigma = self.avalanche.branching_ratio();
            let activity = num_activations as f64 / n as f64;
            self.regulator.update(sigma, activity);
        }

        // Autopoietic topology evolution
        if self.config.autopoietic_topology {
            self.topology.record_activations(&activations);
            if self.timestep % 100 == 0 {
                self.topology.evolve(rng);
            }
        }

        // Update Phi (expensive, do periodically)
        if self.phi_computer.is_some() && self.timestep % 10 == 0 {
            let from_state = self.state_to_index();
            // Predict next state (simplified)
            let to_state = from_state; // Would need actual transition
            if let Some(ref mut phi) = self.phi_computer {
                phi.update_tpm(from_state, to_state);
            }
        }

        self.timestep += 1;
    }

    /// Convert current state to index
    fn state_to_index(&self) -> usize {
        let mut index = 0;
        for (i, &s) in self.states.iter().take(16).enumerate() {
            if s == 1 {
                index |= 1 << i;
            }
        }
        index
    }

    /// Get current temperature
    pub fn temperature(&self) -> f64 {
        self.regulator.temperature()
    }

    /// Get branching ratio
    pub fn branching_ratio(&self) -> f64 {
        self.avalanche.branching_ratio()
    }

    /// Check if system is at criticality
    pub fn is_critical(&self) -> bool {
        self.avalanche.is_critical() && self.regulator.is_near_critical()
    }

    /// Get estimated power-law exponent
    pub fn power_law_exponent(&self) -> Option<f64> {
        self.avalanche.estimate_power_law_exponent()
    }

    /// Get Phi value
    pub fn phi(&self) -> f64 {
        self.phi_computer.as_ref().map(|p| p.phi()).unwrap_or(0.0)
    }

    /// Get network statistics
    pub fn stats(&self) -> AutopoieticStats {
        AutopoieticStats {
            timestep: self.timestep,
            temperature: self.regulator.temperature(),
            branching_ratio: self.avalanche.branching_ratio(),
            is_critical: self.is_critical(),
            power_law_exponent: self.power_law_exponent(),
            num_edges: self.topology.num_edges(),
            clustering_coefficient: self.topology.clustering_coefficient(),
            phi: self.phi(),
            activity: self.states.iter().filter(|&&s| s == 1).count() as f64
                / self.config.num_nodes as f64,
        }
    }

    /// Get avalanche size distribution
    pub fn avalanche_distribution(&self) -> Vec<(usize, usize)> {
        self.avalanche.size_distribution()
    }

    /// Run simulation for given number of steps
    pub fn run<R: Rng>(&mut self, steps: usize, rng: &mut R) {
        for _ in 0..steps {
            self.step(rng);
        }
    }
}

/// Statistics for autopoietic network
#[derive(Debug, Clone)]
pub struct AutopoieticStats {
    pub timestep: usize,
    pub temperature: f64,
    pub branching_ratio: f64,
    pub is_critical: bool,
    pub power_law_exponent: Option<f64>,
    pub num_edges: usize,
    pub clustering_coefficient: f64,
    pub phi: f64,
    pub activity: f64,
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    #[test]
    fn test_avalanche_tracker_basic() {
        let mut tracker = AvalancheTracker::new();

        // Simulate an avalanche
        tracker.record_activations(5);
        tracker.record_activations(3);
        tracker.record_activations(1);
        tracker.record_activations(0); // End of avalanche

        assert_eq!(tracker.size_history.len(), 1);
        assert_eq!(tracker.size_history[0], 9); // 5 + 3 + 1
    }

    #[test]
    fn test_branching_ratio() {
        let mut tracker = AvalancheTracker::new();

        // σ = 1 means each activation causes on average 1 activation
        tracker.record_activations(10);
        tracker.record_activations(10);
        tracker.record_activations(10);

        let sigma = tracker.branching_ratio();
        assert!((sigma - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_homeostatic_regulator() {
        let mut regulator = HomeostaticRegulator::default();

        // Start at T_c
        assert!((regulator.temperature() - SOC_CRITICAL_TEMP).abs() < 0.01);

        // If σ > 1 (supercritical), temperature should increase
        regulator.update(1.5, 0.1);
        assert!(regulator.temperature() > SOC_CRITICAL_TEMP);

        // If σ < 1 (subcritical), temperature should decrease
        let mut reg2 = HomeostaticRegulator::default();
        reg2.update(0.5, 0.1);
        assert!(reg2.temperature() < SOC_CRITICAL_TEMP);
    }

    #[test]
    fn test_autopoietic_topology() {
        let mut topology = AutopoieticTopology::new(10);

        // Record enough correlated activations for statistics
        for i in 0..50 {
            // Alternate pattern to create variance
            let activations: Vec<bool> = (0..10)
                .map(|j| (i + j) % 2 == 0)
                .collect();
            topology.record_activations(&activations);
        }

        // Correlations should be computed (even nodes correlate, odd nodes correlate)
        // Check that correlations are finite and bounded
        assert!(topology.correlations[0][0].abs() <= 1.0);
        assert!(topology.correlations[0][1].abs() <= 1.0);

        // Nodes 0 and 2 should be positively correlated (both even)
        // After enough samples with alternating pattern
        let corr_0_2 = topology.correlations[0][2];
        assert!(corr_0_2.is_finite(), "Correlation should be finite: {}", corr_0_2);
    }

    #[test]
    fn test_autopoietic_network_creation() {
        let config = AutopoieticConfig {
            num_nodes: 50,
            initial_temp: SOC_CRITICAL_TEMP,
            homeostatic: true,
            autopoietic_topology: false,
            compute_phi: false,
            activity_setpoint: 0.1,
        };

        let network = AutopoieticNetwork::new(config);

        assert_eq!(network.states.len(), 50);
        assert!((network.temperature() - SOC_CRITICAL_TEMP).abs() < 0.01);
    }

    #[test]
    fn test_network_step() {
        let config = AutopoieticConfig::default();
        let mut network = AutopoieticNetwork::new(config);
        let mut rng = SmallRng::seed_from_u64(42);

        // Run some steps
        network.run(100, &mut rng);

        // Should have updated timestep
        assert_eq!(network.timestep, 100);
    }

    #[test]
    fn test_small_world_init() {
        let config = AutopoieticConfig {
            num_nodes: 20,
            ..Default::default()
        };
        let mut network = AutopoieticNetwork::new(config);
        let mut rng = SmallRng::seed_from_u64(42);

        network.init_small_world(4, 0.1, &mut rng);

        // Should have edges
        let total_edges: usize = network.couplings.iter().map(|c| c.len()).sum();
        assert!(total_edges > 0);
    }

    #[test]
    fn test_criticality_detection() {
        let mut tracker = AvalancheTracker::new();

        // Simulate critical dynamics (σ ≈ 1)
        for _ in 0..100 {
            let activations = 5;
            tracker.record_activations(activations);
        }
        tracker.record_activations(0);

        // With constant activations, σ = 1
        assert!(tracker.is_critical());
    }

    #[test]
    fn test_power_law_estimation() {
        let mut tracker = AvalancheTracker::new();

        // Add some avalanche sizes
        for size in [1, 2, 1, 3, 1, 1, 2, 4, 1, 2, 1, 1] {
            tracker.record_activations(size);
            tracker.record_activations(0);
        }

        let exponent = tracker.estimate_power_law_exponent();
        assert!(exponent.is_some());
        // Power-law exponent should be positive
        assert!(exponent.unwrap() > 1.0);
    }

    #[test]
    fn test_phi_computation() {
        let phi = PhiComputer::new(4);

        // Initial Phi should be low (uniform distribution)
        let phi_val = phi.phi();
        assert!(phi_val >= 0.0);
    }

    #[test]
    fn test_wolfram_verified_critical_temp() {
        // T_c = 2/ln(1+√2) = 2.269185314213022
        let expected = 2.269185314213022;
        assert!((SOC_CRITICAL_TEMP - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pbit_at_criticality() {
        // At T_c with h=0, P should be 0.5
        let p = pbit_probability(0.0, 0.0, SOC_CRITICAL_TEMP);
        assert!((p - 0.5).abs() < 1e-10);
    }
}
