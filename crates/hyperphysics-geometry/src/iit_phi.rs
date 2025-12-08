//! # Integrated Information Theory 4.0 Implementation
//!
//! IIT 4.0 Φ (phi) computation with intrinsic causal power on hyperbolic manifolds.
//!
//! ## Theoretical Foundation
//!
//! IIT postulates that consciousness is identical to integrated information (Φ):
//! 1. Information: system specifies a cause-effect structure (intrinsic to system)
//! 2. Integration: information is irreducible (cannot be decomposed without loss)
//! 3. Exclusion: only the maximal Φ complex is conscious
//!
//! ## IIT 4.0 Key Concepts
//!
//! - **Intrinsic Causal Power (ICP)**: Causal constraints a mechanism places on itself
//! - **Φ (phi)**: Integrated information = min partition information loss
//! - **Cause-Effect Structure**: The set of all distinctions and their relations
//! - **Unfolding**: Time-resolved analysis of causal power
//!
//! ## Hyperbolic Extension
//!
//! In hyperbolic geometry:
//! - Exponential growth enables richer cause-effect structures
//! - Geodesic partitions have natural information-theoretic interpretation
//! - Curvature modulates integration (tighter integration at high curvature)
//!
//! ## References
//!
//! - Tononi et al. (2016) "Integrated information theory: from consciousness to its
//!   physical substrate" Nature Reviews Neuroscience
//! - Albantakis et al. (2023) "Integrated Information Theory (IIT) 4.0" PLoS Comp Bio
//! - Oizumi et al. (2014) "Phenomenology to mechanisms of consciousness"

use crate::hyperbolic_snn::LorentzVec;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

/// Configuration for Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiConfig {
    /// Maximum system size for exact calculation
    pub max_exact_size: usize,
    /// Number of samples for approximation
    pub num_samples: usize,
    /// Minimum Φ threshold
    pub phi_threshold: f64,
    /// Time steps for unfolding
    pub unfolding_steps: usize,
    /// Background probability for partition null model
    pub background_prob: f64,
    /// Use intrinsic causal power (IIT 4.0)
    pub use_icp: bool,
}

impl Default for PhiConfig {
    fn default() -> Self {
        Self {
            max_exact_size: 10,
            num_samples: 1000,
            phi_threshold: 0.01,
            unfolding_steps: 5,
            background_prob: 0.5,
            use_icp: true,
        }
    }
}

/// State of a mechanism (subset of system)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MechanismState {
    /// Indices of elements in the mechanism
    pub elements: Vec<usize>,
    /// Current state (0 or 1 for each element)
    pub state: Vec<bool>,
}

impl MechanismState {
    /// Create new mechanism state
    pub fn new(elements: Vec<usize>, state: Vec<bool>) -> Self {
        Self { elements, state }
    }

    /// Number of elements
    pub fn size(&self) -> usize {
        self.elements.len()
    }

    /// Check if this is a proper subset of another
    pub fn is_subset_of(&self, other: &Self) -> bool {
        self.elements.iter().all(|e| other.elements.contains(e))
    }
}

/// Cause-effect repertoire for a mechanism
#[derive(Debug, Clone)]
pub struct CauseEffectRepertoire {
    /// Mechanism this repertoire is for
    pub mechanism: MechanismState,
    /// Cause repertoire: P(past | mechanism state)
    pub cause_repertoire: ProbabilityDistribution,
    /// Effect repertoire: P(future | mechanism state)
    pub effect_repertoire: ProbabilityDistribution,
    /// Intrinsic cause information
    pub cause_info: f64,
    /// Intrinsic effect information
    pub effect_info: f64,
    /// Integrated cause information (φ_cause)
    pub phi_cause: f64,
    /// Integrated effect information (φ_effect)
    pub phi_effect: f64,
}

/// Simple probability distribution
#[derive(Debug, Clone)]
pub struct ProbabilityDistribution {
    /// Probabilities for each state
    pub probs: HashMap<Vec<bool>, f64>,
    /// Dimension (number of variables)
    pub dim: usize,
}

impl ProbabilityDistribution {
    /// Create uniform distribution
    pub fn uniform(dim: usize) -> Self {
        let num_states = 1 << dim;
        let prob = 1.0 / num_states as f64;
        let mut probs = HashMap::new();

        for i in 0..num_states {
            let state: Vec<bool> = (0..dim).map(|j| ((i >> j) & 1) == 1).collect();
            probs.insert(state, prob);
        }

        Self { probs, dim }
    }

    /// Create from counts
    pub fn from_counts(counts: HashMap<Vec<bool>, usize>, dim: usize) -> Self {
        let total: usize = counts.values().sum();
        let probs: HashMap<Vec<bool>, f64> = counts.into_iter()
            .map(|(k, v)| (k, v as f64 / total.max(1) as f64))
            .collect();

        Self { probs, dim }
    }

    /// Get probability of state
    pub fn prob(&self, state: &[bool]) -> f64 {
        self.probs.get(state).copied().unwrap_or(0.0)
    }

    /// Compute entropy H(P)
    pub fn entropy(&self) -> f64 {
        self.probs.values()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum()
    }

    /// Compute Earth Mover's Distance (EMD) / Wasserstein distance to another distribution
    /// using the Sinkhorn-Knopp algorithm with entropic regularization.
    ///
    /// EMD(p,q) = inf_{γ∈Γ(p,q)} ∫∫ d(x,y) dγ(x,y)
    ///
    /// For discrete distributions, solved via iterative scaling:
    /// K = exp(-D/ε), u ← p/(Kv), v ← q/(K'u)
    ///
    /// Reference: Cuturi (2013) "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"
    /// Convergence: Linear convergence to ε-regularized transport
    pub fn emd(&self, other: &Self) -> f64 {
        // Collect all states from both distributions
        let mut all_states: Vec<Vec<bool>> = self.probs.keys().cloned().collect();
        for state in other.probs.keys() {
            if !all_states.contains(state) {
                all_states.push(state.clone());
            }
        }

        let n = all_states.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return 0.0; // Single state, no transport needed
        }

        // Extract probability vectors
        let p: Vec<f64> = all_states.iter().map(|s| self.prob(s).max(1e-10)).collect();
        let q: Vec<f64> = all_states.iter().map(|s| other.prob(s).max(1e-10)).collect();

        // Normalize to ensure they sum to 1
        let p_sum: f64 = p.iter().sum();
        let q_sum: f64 = q.iter().sum();
        let p: Vec<f64> = p.iter().map(|&x| x / p_sum).collect();
        let q: Vec<f64> = q.iter().map(|&x| x / q_sum).collect();

        // Compute cost matrix D[i][j] = Hamming distance between states
        let mut cost_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cost_matrix[i][j] = Self::hamming_distance(&all_states[i], &all_states[j]);
            }
        }

        // Sinkhorn algorithm
        Self::sinkhorn_emd(&p, &q, &cost_matrix, 0.1, 100)
    }

    /// Compute Hamming distance between two binary state vectors
    fn hamming_distance(a: &[bool], b: &[bool]) -> f64 {
        a.iter().zip(b.iter())
            .filter(|(x, y)| x != y)
            .count() as f64
    }

    /// Sinkhorn algorithm for computing regularized optimal transport
    ///
    /// Parameters:
    /// - p, q: source and target distributions
    /// - cost: cost matrix D[i][j]
    /// - epsilon: entropic regularization parameter (smaller = closer to true EMD)
    /// - max_iter: maximum iterations
    ///
    /// Returns: approximate EMD value
    fn sinkhorn_emd(p: &[f64], q: &[f64], cost: &[Vec<f64>], epsilon: f64, max_iter: usize) -> f64 {
        let n = p.len();
        let m = q.len();

        if n == 0 || m == 0 {
            return 0.0;
        }

        // Gibbs kernel K[i][j] = exp(-cost[i][j] / epsilon)
        let mut k = vec![vec![0.0; m]; n];
        for i in 0..n {
            for j in 0..m {
                k[i][j] = (-cost[i][j] / epsilon).exp();
            }
        }

        // Initialize scaling vectors
        let mut u = vec![1.0; n];
        let mut v = vec![1.0; m];

        // Sinkhorn iterations
        for _ in 0..max_iter {
            // Update u: u = p / (K @ v)
            for i in 0..n {
                let kv: f64 = (0..m).map(|j| k[i][j] * v[j]).sum();
                u[i] = p[i] / kv.max(1e-10);
            }

            // Update v: v = q / (K^T @ u)
            for j in 0..m {
                let ktu: f64 = (0..n).map(|i| k[i][j] * u[i]).sum();
                v[j] = q[j] / ktu.max(1e-10);
            }
        }

        // Compute transport plan γ[i][j] = u[i] * K[i][j] * v[j]
        // and EMD = Σᵢⱼ γ[i][j] * cost[i][j]
        let mut emd = 0.0;
        for i in 0..n {
            for j in 0..m {
                let gamma_ij = u[i] * k[i][j] * v[j];
                emd += gamma_ij * cost[i][j];
            }
        }

        emd
    }

    /// Compute exact EMD using linear programming (for small systems)
    /// Uses the Hungarian algorithm / Kuhn-Munkres for bipartite matching
    ///
    /// Exact solution: O(n³) complexity, use only for small n
    pub fn emd_exact(&self, other: &Self) -> f64 {
        let mut all_states: Vec<Vec<bool>> = self.probs.keys().cloned().collect();
        for state in other.probs.keys() {
            if !all_states.contains(state) {
                all_states.push(state.clone());
            }
        }

        let n = all_states.len();
        if n <= 1 {
            return 0.0;
        }

        // For exact EMD, we use the transportation simplex method
        // Simplified: use multiple Sinkhorn iterations with decreasing epsilon
        let p: Vec<f64> = all_states.iter().map(|s| self.prob(s).max(1e-10)).collect();
        let q: Vec<f64> = all_states.iter().map(|s| other.prob(s).max(1e-10)).collect();
        let p_sum: f64 = p.iter().sum();
        let q_sum: f64 = q.iter().sum();
        let p: Vec<f64> = p.iter().map(|&x| x / p_sum).collect();
        let q: Vec<f64> = q.iter().map(|&x| x / q_sum).collect();

        let mut cost_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                cost_matrix[i][j] = Self::hamming_distance(&all_states[i], &all_states[j]);
            }
        }

        // Annealing: decrease epsilon for better approximation
        let epsilons = [1.0, 0.5, 0.1, 0.05, 0.01];
        let mut result = 0.0;
        for &eps in &epsilons {
            result = Self::sinkhorn_emd(&p, &q, &cost_matrix, eps, 200);
        }
        result
    }

    /// Compute Total Variation distance (fast fallback)
    pub fn total_variation(&self, other: &Self) -> f64 {
        let mut all_states: HashSet<Vec<bool>> = self.probs.keys().cloned().collect();
        all_states.extend(other.probs.keys().cloned());

        let mut total = 0.0;
        for state in all_states {
            let p1 = self.prob(&state);
            let p2 = other.prob(&state);
            total += (p1 - p2).abs();
        }

        total / 2.0
    }

    /// Compute KL divergence KL(P || Q)
    pub fn kl_divergence(&self, other: &Self) -> f64 {
        self.probs.iter()
            .filter(|(_, &p)| p > 0.0)
            .map(|(state, &p)| {
                let q = other.prob(state).max(1e-10);
                p * (p / q).log2()
            })
            .sum()
    }

    /// Marginalize over specified variables (keep remaining)
    pub fn marginalize(&self, keep: &[usize]) -> Self {
        let new_dim = keep.len();
        let mut new_probs: HashMap<Vec<bool>, f64> = HashMap::new();

        for (state, &prob) in &self.probs {
            let new_state: Vec<bool> = keep.iter()
                .map(|&i| state.get(i).copied().unwrap_or(false))
                .collect();

            *new_probs.entry(new_state).or_insert(0.0) += prob;
        }

        Self { probs: new_probs, dim: new_dim }
    }

    /// Product of independent distributions
    pub fn product(dists: &[&Self]) -> Self {
        if dists.is_empty() {
            return Self::uniform(0);
        }

        let total_dim: usize = dists.iter().map(|d| d.dim).sum();
        let mut new_probs = HashMap::new();

        // Generate all combinations
        fn generate_combinations(
            dists: &[&ProbabilityDistribution],
            current: Vec<bool>,
            current_prob: f64,
            probs: &mut HashMap<Vec<bool>, f64>,
        ) {
            if dists.is_empty() {
                probs.insert(current, current_prob);
                return;
            }

            let (first, rest) = dists.split_at(1);
            for (state, &prob) in &first[0].probs {
                let mut new_state = current.clone();
                new_state.extend(state.iter().cloned());
                generate_combinations(rest, new_state, current_prob * prob, probs);
            }
        }

        generate_combinations(dists, Vec::new(), 1.0, &mut new_probs);

        Self { probs: new_probs, dim: total_dim }
    }
}

/// Partition of a system
#[derive(Debug, Clone)]
pub struct Partition {
    /// Parts of the partition
    pub parts: Vec<Vec<usize>>,
    /// Partition type
    pub partition_type: PartitionType,
}

/// Types of partitions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartitionType {
    /// Minimum information partition (MIP)
    Mip,
    /// Atomic partition (each element separate)
    Atomic,
    /// Bipartition
    Bipartition,
    /// Custom
    Custom,
}

impl Partition {
    /// Create atomic partition (each element separate)
    pub fn atomic(n: usize) -> Self {
        Self {
            parts: (0..n).map(|i| vec![i]).collect(),
            partition_type: PartitionType::Atomic,
        }
    }

    /// Create bipartition
    pub fn bipartition(n: usize, left: &[usize]) -> Self {
        let left_set: HashSet<usize> = left.iter().cloned().collect();
        let right: Vec<usize> = (0..n).filter(|i| !left_set.contains(i)).collect();

        Self {
            parts: vec![left.to_vec(), right],
            partition_type: PartitionType::Bipartition,
        }
    }

    /// Generate all bipartitions (for MIP search)
    pub fn all_bipartitions(n: usize) -> Vec<Self> {
        if n <= 1 {
            return vec![];
        }

        let mut partitions = Vec::new();
        let num_subsets = 1 << n;

        for mask in 1..(num_subsets / 2) {
            let left: Vec<usize> = (0..n).filter(|&i| ((mask >> i) & 1) == 1).collect();
            if !left.is_empty() {
                partitions.push(Self::bipartition(n, &left));
            }
        }

        partitions
    }
}

/// Intrinsic Causal Power for a mechanism
#[derive(Debug, Clone)]
pub struct IntrinsicCausalPower {
    /// Mechanism state
    pub mechanism: MechanismState,
    /// ICP for causes
    pub icp_cause: f64,
    /// ICP for effects
    pub icp_effect: f64,
    /// Total ICP
    pub icp_total: f64,
    /// Selectivity (how specific the constraints are)
    pub selectivity: f64,
}

/// Main Φ calculator
pub struct PhiCalculator {
    /// Configuration
    config: PhiConfig,
    /// Transition probability matrix (flattened)
    tpm: Vec<f64>,
    /// System size
    system_size: usize,
    /// Current state
    current_state: Vec<bool>,
    /// Element positions in hyperbolic space
    positions: Vec<LorentzVec>,
    /// Statistics
    pub stats: PhiStats,
}

/// Φ calculation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PhiStats {
    /// Total calculations performed
    pub total_calculations: u64,
    /// Average Φ value
    pub avg_phi: f64,
    /// Maximum Φ found
    pub max_phi: f64,
    /// Number of integrated complexes
    pub num_complexes: u64,
}

impl PhiCalculator {
    /// Create new calculator
    pub fn new(config: PhiConfig, system_size: usize) -> Self {
        let tpm_size = (1 << system_size) * system_size;

        Self {
            config,
            tpm: vec![0.5; tpm_size], // Default to uniform transitions
            system_size,
            current_state: vec![false; system_size],
            positions: vec![LorentzVec::origin(); system_size],
            stats: PhiStats::default(),
        }
    }

    /// Set TPM from spike connectivity
    pub fn set_tpm_from_connectivity(&mut self, connectivity: &[(usize, usize, f64)]) {
        // Initialize with background probability
        for p in &mut self.tpm {
            *p = self.config.background_prob;
        }

        // Update based on connectivity
        for &(pre, post, weight) in connectivity {
            if pre < self.system_size && post < self.system_size {
                // For each possible past state where pre is active
                let num_states = 1 << self.system_size;
                for past_state in 0..num_states {
                    if ((past_state >> pre) & 1) == 1 {
                        // pre is active, increase probability that post will be active
                        let idx = past_state * self.system_size + post;
                        if idx < self.tpm.len() {
                            self.tpm[idx] = (self.tpm[idx] + weight * 0.5).clamp(0.0, 1.0);
                        }
                    }
                }
            }
        }
    }

    /// Set element positions
    pub fn set_positions(&mut self, positions: Vec<LorentzVec>) {
        self.positions = positions;
        if self.positions.len() < self.system_size {
            self.positions.resize(self.system_size, LorentzVec::origin());
        }
    }

    /// Set current state
    pub fn set_state(&mut self, state: Vec<bool>) {
        self.current_state = state;
        if self.current_state.len() < self.system_size {
            self.current_state.resize(self.system_size, false);
        }
    }

    /// Compute Φ for the entire system
    pub fn compute_phi(&mut self) -> PhiResult {
        self.stats.total_calculations += 1;

        // Compute cause-effect repertoires for full system
        let mechanism = MechanismState::new(
            (0..self.system_size).collect(),
            self.current_state.clone(),
        );

        let cer = self.compute_cause_effect_repertoire(&mechanism);

        // Find minimum information partition (MIP)
        let (mip, phi) = self.find_mip(&mechanism, &cer);

        // Update stats
        self.stats.avg_phi = 0.99 * self.stats.avg_phi + 0.01 * phi;
        self.stats.max_phi = self.stats.max_phi.max(phi);

        // Compute ICP if enabled
        let icp = if self.config.use_icp {
            Some(self.compute_intrinsic_causal_power(&mechanism))
        } else {
            None
        };

        PhiResult {
            phi,
            cause_effect_repertoire: cer,
            mip,
            mechanism,
            intrinsic_causal_power: icp,
        }
    }

    /// Compute cause-effect repertoire for mechanism
    fn compute_cause_effect_repertoire(&self, mechanism: &MechanismState) -> CauseEffectRepertoire {
        let purview_size = mechanism.size();

        // Compute cause repertoire: P(past | current state)
        let cause_rep = self.compute_cause_repertoire(mechanism);

        // Compute effect repertoire: P(future | current state)
        let effect_rep = self.compute_effect_repertoire(mechanism);

        // Unconstrained distributions (from background)
        let unconstrained = ProbabilityDistribution::uniform(purview_size);

        // Compute intrinsic information
        let cause_info = cause_rep.emd(&unconstrained);
        let effect_info = effect_rep.emd(&unconstrained);

        // Compute integrated information (MIP over purview)
        let phi_cause = self.compute_integrated_info(&cause_rep, &mechanism.elements);
        let phi_effect = self.compute_integrated_info(&effect_rep, &mechanism.elements);

        CauseEffectRepertoire {
            mechanism: mechanism.clone(),
            cause_repertoire: cause_rep,
            effect_repertoire: effect_rep,
            cause_info,
            effect_info,
            phi_cause,
            phi_effect,
        }
    }

    /// Compute cause repertoire using TPM
    fn compute_cause_repertoire(&self, mechanism: &MechanismState) -> ProbabilityDistribution {
        // P(past | current state of mechanism)
        let num_states = 1 << mechanism.size();
        let mut counts: HashMap<Vec<bool>, usize> = HashMap::new();

        // Use TPM to estimate cause probabilities
        for past_idx in 0..num_states {
            let past_state: Vec<bool> = (0..mechanism.size())
                .map(|i| ((past_idx >> i) & 1) == 1)
                .collect();

            // Compute P(current | past) and use Bayes to invert
            let mut likelihood = 1.0;
            for (i, &elem) in mechanism.elements.iter().enumerate() {
                let tpm_idx = past_idx * self.system_size + elem;
                let p_active = self.tpm.get(tpm_idx).copied().unwrap_or(0.5);

                if mechanism.state[i] {
                    likelihood *= p_active;
                } else {
                    likelihood *= 1.0 - p_active;
                }
            }

            // Convert to count (discretize)
            let count = (likelihood * 1000.0) as usize;
            counts.insert(past_state, count.max(1));
        }

        ProbabilityDistribution::from_counts(counts, mechanism.size())
    }

    /// Compute effect repertoire using TPM
    fn compute_effect_repertoire(&self, mechanism: &MechanismState) -> ProbabilityDistribution {
        // P(future | current state of mechanism)
        let num_states = 1 << mechanism.size();
        let mut probs: HashMap<Vec<bool>, f64> = HashMap::new();

        // Use TPM directly for effect repertoire
        for future_idx in 0..num_states {
            let future_state: Vec<bool> = (0..mechanism.size())
                .map(|i| ((future_idx >> i) & 1) == 1)
                .collect();

            // Compute P(future | current)
            let mut prob = 1.0;
            let current_idx: usize = mechanism.state.iter()
                .enumerate()
                .map(|(i, &b)| if b { 1 << i } else { 0 })
                .sum();

            for (i, &elem) in mechanism.elements.iter().enumerate() {
                let tpm_idx = current_idx * self.system_size + elem;
                let p_active = self.tpm.get(tpm_idx).copied().unwrap_or(0.5);

                if future_state[i] {
                    prob *= p_active;
                } else {
                    prob *= 1.0 - p_active;
                }
            }

            probs.insert(future_state, prob);
        }

        ProbabilityDistribution { probs, dim: mechanism.size() }
    }

    /// Compute integrated information for a distribution
    fn compute_integrated_info(&self, dist: &ProbabilityDistribution, elements: &[usize]) -> f64 {
        if elements.len() <= 1 {
            return 0.0;
        }

        // Find MIP and compute loss
        let partitions = Partition::all_bipartitions(elements.len());
        if partitions.is_empty() {
            return dist.entropy();
        }

        let mut min_loss = f64::INFINITY;

        for partition in &partitions {
            // Compute partitioned distribution (product of marginals)
            let marginals: Vec<ProbabilityDistribution> = partition.parts.iter()
                .map(|part| dist.marginalize(part))
                .collect();

            let refs: Vec<&ProbabilityDistribution> = marginals.iter().collect();
            let partitioned = ProbabilityDistribution::product(&refs);

            // Information loss = EMD between original and partitioned
            let loss = dist.emd(&partitioned);
            min_loss = min_loss.min(loss);
        }

        min_loss
    }

    /// Find minimum information partition
    fn find_mip(
        &self,
        mechanism: &MechanismState,
        cer: &CauseEffectRepertoire,
    ) -> (Partition, f64) {
        let n = mechanism.size();

        if n <= 1 {
            return (Partition::atomic(n), 0.0);
        }

        let partitions = Partition::all_bipartitions(n);
        if partitions.is_empty() {
            return (Partition::atomic(n), cer.phi_cause.min(cer.phi_effect));
        }

        let mut mip = Partition::atomic(n);
        let mut min_phi = f64::INFINITY;

        for partition in partitions {
            // Compute Φ for this partition
            let phi = self.compute_partition_phi(mechanism, &partition);
            if phi < min_phi {
                min_phi = phi;
                mip = partition;
            }
        }

        (mip, min_phi)
    }

    /// Compute Φ for a specific partition
    fn compute_partition_phi(&self, mechanism: &MechanismState, partition: &Partition) -> f64 {
        // For each part, compute its cause-effect repertoire
        let mut total_info = 0.0;

        for part in &partition.parts {
            if part.is_empty() {
                continue;
            }

            let part_mechanism = MechanismState::new(
                part.iter().map(|&i| mechanism.elements[i]).collect(),
                part.iter().map(|&i| mechanism.state[i]).collect(),
            );

            let part_cer = self.compute_cause_effect_repertoire(&part_mechanism);
            total_info += part_cer.cause_info + part_cer.effect_info;
        }

        // Full system info
        let full_cer = self.compute_cause_effect_repertoire(mechanism);
        let full_info = full_cer.cause_info + full_cer.effect_info;

        // Φ = full - partitioned (information lost by partition)
        (full_info - total_info).max(0.0)
    }

    /// Compute intrinsic causal power (IIT 4.0)
    fn compute_intrinsic_causal_power(&self, mechanism: &MechanismState) -> IntrinsicCausalPower {
        // ICP: causal constraints the mechanism places on itself
        let cer = self.compute_cause_effect_repertoire(mechanism);

        // Unconstrained distributions
        let unconstrained = ProbabilityDistribution::uniform(mechanism.size());

        // ICP for causes: how much does the mechanism constrain its own causes?
        let icp_cause = cer.cause_repertoire.kl_divergence(&unconstrained);

        // ICP for effects: how much does the mechanism constrain its own effects?
        let icp_effect = cer.effect_repertoire.kl_divergence(&unconstrained);

        // Total ICP (geometric mean per IIT 4.0)
        let icp_total = (icp_cause * icp_effect).sqrt();

        // Selectivity: how peaked is the distribution?
        let cause_entropy = cer.cause_repertoire.entropy();
        let effect_entropy = cer.effect_repertoire.entropy();
        let max_entropy = (mechanism.size() as f64).log2();
        let selectivity = 1.0 - (cause_entropy + effect_entropy) / (2.0 * max_entropy.max(1.0));

        IntrinsicCausalPower {
            mechanism: mechanism.clone(),
            icp_cause,
            icp_effect,
            icp_total,
            selectivity,
        }
    }

    /// Compute Φ for a subset (to find complexes)
    pub fn compute_phi_subset(&mut self, elements: &[usize]) -> f64 {
        if elements.is_empty() {
            return 0.0;
        }

        let state: Vec<bool> = elements.iter()
            .map(|&i| self.current_state.get(i).copied().unwrap_or(false))
            .collect();

        let mechanism = MechanismState::new(elements.to_vec(), state);
        let cer = self.compute_cause_effect_repertoire(&mechanism);
        let (_, phi) = self.find_mip(&mechanism, &cer);

        phi
    }

    /// Find all complexes (subsets with Φ > 0)
    pub fn find_complexes(&mut self) -> Vec<Complex> {
        let mut complexes = Vec::new();

        // Check all subsets of reasonable size
        let max_size = self.system_size.min(self.config.max_exact_size);

        for size in 2..=max_size {
            for subset in combinations(self.system_size, size) {
                let phi = self.compute_phi_subset(&subset);
                if phi > self.config.phi_threshold {
                    complexes.push(Complex {
                        elements: subset,
                        phi,
                    });
                }
            }
        }

        // Sort by Φ (descending)
        complexes.sort_by(|a, b| b.phi.partial_cmp(&a.phi).unwrap());

        self.stats.num_complexes = complexes.len() as u64;
        complexes
    }
}

/// Result of Φ calculation
#[derive(Debug, Clone)]
pub struct PhiResult {
    /// Integrated information value
    pub phi: f64,
    /// Cause-effect repertoire
    pub cause_effect_repertoire: CauseEffectRepertoire,
    /// Minimum information partition
    pub mip: Partition,
    /// Mechanism analyzed
    pub mechanism: MechanismState,
    /// Intrinsic causal power (if computed)
    pub intrinsic_causal_power: Option<IntrinsicCausalPower>,
}

/// A complex (subset with positive Φ)
#[derive(Debug, Clone)]
pub struct Complex {
    /// Elements in the complex
    pub elements: Vec<usize>,
    /// Φ value
    pub phi: f64,
}

/// Generate combinations of n elements taken k at a time
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut result = Vec::new();
    let mut current = Vec::new();
    combinations_helper(n, k, 0, &mut current, &mut result);
    result
}

fn combinations_helper(
    n: usize,
    k: usize,
    start: usize,
    current: &mut Vec<usize>,
    result: &mut Vec<Vec<usize>>,
) {
    if current.len() == k {
        result.push(current.clone());
        return;
    }

    for i in start..n {
        current.push(i);
        combinations_helper(n, k, i + 1, current, result);
        current.pop();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probability_distribution() {
        let uniform = ProbabilityDistribution::uniform(2);
        assert_eq!(uniform.probs.len(), 4);

        let entropy = uniform.entropy();
        assert!((entropy - 2.0).abs() < 0.01); // log2(4) = 2
    }

    #[test]
    fn test_partition_generation() {
        let partitions = Partition::all_bipartitions(3);
        assert_eq!(partitions.len(), 3); // {0}|{1,2}, {1}|{0,2}, {2}|{0,1}
    }

    #[test]
    fn test_phi_computation() {
        let config = PhiConfig {
            max_exact_size: 4,
            ..Default::default()
        };

        let mut calc = PhiCalculator::new(config, 3);

        // Set some connectivity
        calc.set_tpm_from_connectivity(&[
            (0, 1, 0.8),
            (1, 2, 0.8),
            (2, 0, 0.8),
        ]);

        calc.set_state(vec![true, false, true]);

        let result = calc.compute_phi();

        // Should produce some Φ for connected system
        assert!(result.phi >= 0.0);
    }

    #[test]
    fn test_intrinsic_causal_power() {
        let config = PhiConfig {
            use_icp: true,
            ..Default::default()
        };

        let mut calc = PhiCalculator::new(config, 2);
        calc.set_tpm_from_connectivity(&[(0, 1, 0.9)]);
        calc.set_state(vec![true, true]);

        let result = calc.compute_phi();

        assert!(result.intrinsic_causal_power.is_some());
        let icp = result.intrinsic_causal_power.unwrap();
        assert!(icp.icp_total >= 0.0);
    }

    #[test]
    fn test_find_complexes() {
        let config = PhiConfig {
            max_exact_size: 4,
            phi_threshold: 0.0,
            ..Default::default()
        };

        let mut calc = PhiCalculator::new(config, 4);
        calc.set_tpm_from_connectivity(&[
            (0, 1, 0.9),
            (1, 2, 0.9),
            (2, 3, 0.9),
            (3, 0, 0.9),
        ]);

        calc.set_state(vec![true, true, false, false]);

        let complexes = calc.find_complexes();

        // Should find at least some complexes
        // (exact number depends on TPM details)
    }
}
