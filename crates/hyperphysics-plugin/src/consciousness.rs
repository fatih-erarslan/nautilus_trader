//! # Consciousness Emergence Module
//!
//! Implements Integrated Information Theory (IIT) Phi metrics and
//! Self-Organized Criticality (SOC) analysis for consciousness modeling.
//!
//! ## Theoretical Foundation
//!
//! ### Integrated Information Theory (IIT 3.0)
//! - Φ (Phi) measures integrated information
//! - Higher Φ indicates greater consciousness
//! - Computed via partition analysis and mutual information
//!
//! ### Self-Organized Criticality
//! - Systems at criticality exhibit power-law avalanches
//! - Branching ratio σ ≈ 1 indicates critical state
//! - Hurst exponent H ≈ 0.5 at criticality
//!
//! ## References
//!
//! - Tononi, G. (2008). "Consciousness as Integrated Information"
//! - Oizumi, M., Albantakis, L., & Tononi, G. (2014). "From the phenomenology
//!   to the mechanisms of consciousness: IIT 3.0"
//! - Beggs, J. M., & Plenz, D. (2003). "Neuronal avalanches in neocortical circuits"

use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

/// Minimum entropy threshold
pub const MIN_ENTROPY: f64 = 1e-10;

/// Critical branching ratio (σ = 1 at criticality)
pub const CRITICAL_BRANCHING_RATIO: f64 = 1.0;

/// Critical Hurst exponent
pub const CRITICAL_HURST: f64 = 0.5;

/// Avalanche exponent at criticality (τ ≈ 1.5)
pub const AVALANCHE_EXPONENT_CRITICAL: f64 = 1.5;

/// Tolerance for criticality detection
pub const CRITICALITY_TOLERANCE: f64 = 0.1;

/// Golden ratio (φ) for consciousness thresholds
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio
pub const PHI_INV: f64 = 0.618033988749895;

// ============================================================================
// Information Theory Functions
// ============================================================================

/// Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
pub fn shannon_entropy(probabilities: &[f64]) -> f64 {
    probabilities
        .iter()
        .filter(|&&p| p > MIN_ENTROPY)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Joint entropy: H(X,Y)
pub fn joint_entropy(joint_probs: &[Vec<f64>]) -> f64 {
    let mut entropy = 0.0;
    for row in joint_probs {
        for &p in row {
            if p > MIN_ENTROPY {
                entropy -= p * p.log2();
            }
        }
    }
    entropy
}

/// Conditional entropy: H(X|Y) = H(X,Y) - H(Y)
pub fn conditional_entropy(joint_probs: &[Vec<f64>], marginal_y: &[f64]) -> f64 {
    joint_entropy(joint_probs) - shannon_entropy(marginal_y)
}

/// Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
pub fn mutual_information(px: &[f64], py: &[f64], pxy: &[Vec<f64>]) -> f64 {
    shannon_entropy(px) + shannon_entropy(py) - joint_entropy(pxy)
}

/// Transfer entropy: T(X→Y) = H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t,X_t)
pub fn transfer_entropy(
    source_history: &[f64],
    target_history: &[f64],
    target_future: &[f64],
) -> f64 {
    if source_history.len() != target_history.len()
        || target_history.len() != target_future.len()
    {
        return 0.0;
    }

    // Simplified estimation using binned probabilities
    let bins = 10;
    let mut h_yt1_yt = estimate_conditional_entropy(target_future, target_history, bins);
    let mut h_yt1_yt_xt =
        estimate_conditional_entropy_joint(target_future, target_history, source_history, bins);

    // Ensure non-negative
    h_yt1_yt = h_yt1_yt.max(0.0);
    h_yt1_yt_xt = h_yt1_yt_xt.max(0.0);

    (h_yt1_yt - h_yt1_yt_xt).max(0.0)
}

/// Estimate conditional entropy via binning
fn estimate_conditional_entropy(x: &[f64], y: &[f64], bins: usize) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }

    // Build joint histogram
    let mut joint = vec![vec![0usize; bins]; bins];
    let mut marginal_y = vec![0usize; bins];

    let (x_min, x_max) = min_max(x);
    let (y_min, y_max) = min_max(y);

    for i in 0..n {
        let bx = bin_index(x[i], x_min, x_max, bins);
        let by = bin_index(y[i], y_min, y_max, bins);
        joint[bx][by] += 1;
        marginal_y[by] += 1;
    }

    // Convert to probabilities and compute H(X|Y)
    let n_f = n as f64;
    let mut h = 0.0;
    for bx in 0..bins {
        for by in 0..bins {
            if joint[bx][by] > 0 && marginal_y[by] > 0 {
                let p_xy = joint[bx][by] as f64 / n_f;
                let p_y = marginal_y[by] as f64 / n_f;
                let p_x_given_y = p_xy / p_y;
                h -= p_xy * p_x_given_y.log2();
            }
        }
    }

    h
}

/// Estimate conditional entropy with two conditions
fn estimate_conditional_entropy_joint(
    x: &[f64],
    y1: &[f64],
    y2: &[f64],
    bins: usize,
) -> f64 {
    let n = x.len().min(y1.len()).min(y2.len());
    if n == 0 {
        return 0.0;
    }

    // Simplified: combine y1 and y2 into single condition
    let combined: Vec<f64> = y1.iter().zip(y2).map(|(&a, &b)| a + b).collect();
    estimate_conditional_entropy(x, &combined, bins)
}

fn min_max(data: &[f64]) -> (f64, f64) {
    let min = data.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    (min, max)
}

fn bin_index(value: f64, min: f64, max: f64, bins: usize) -> usize {
    if max <= min {
        return 0;
    }
    let normalized = (value - min) / (max - min);
    ((normalized * bins as f64) as usize).min(bins - 1)
}

// ============================================================================
// IIT Phi Computation
// ============================================================================

/// Partition of a system for IIT analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Partition {
    /// Elements in partition A
    pub part_a: Vec<usize>,
    /// Elements in partition B
    pub part_b: Vec<usize>,
}

impl Partition {
    /// Create bipartition of n elements
    pub fn bipartitions(n: usize) -> Vec<Self> {
        if n == 0 {
            return vec![];
        }
        if n == 1 {
            return vec![Partition {
                part_a: vec![0],
                part_b: vec![],
            }];
        }

        let mut partitions = Vec::new();

        // Generate all non-trivial bipartitions
        for mask in 1..(1 << n) - 1 {
            let mut a = Vec::new();
            let mut b = Vec::new();

            for i in 0..n {
                if (mask >> i) & 1 == 1 {
                    a.push(i);
                } else {
                    b.push(i);
                }
            }

            // Only keep partitions where A comes "first" lexicographically
            if !a.is_empty() && !b.is_empty() && a[0] < b[0] {
                partitions.push(Partition { part_a: a, part_b: b });
            }
        }

        partitions
    }
}

/// IIT Phi Calculator
#[derive(Debug, Clone)]
pub struct PhiCalculator {
    /// Number of elements
    n_elements: usize,
    /// Transition probability matrix
    tpm: Vec<Vec<f64>>,
    /// Current state distribution
    state_dist: Vec<f64>,
}

impl PhiCalculator {
    /// Create new calculator
    pub fn new(n_elements: usize) -> Self {
        let n_states = 1 << n_elements;
        Self {
            n_elements,
            tpm: vec![vec![1.0 / n_states as f64; n_states]; n_states],
            state_dist: vec![1.0 / n_states as f64; n_states],
        }
    }

    /// Set transition probability matrix
    pub fn set_tpm(&mut self, tpm: Vec<Vec<f64>>) {
        self.tpm = tpm;
    }

    /// Set state distribution
    pub fn set_state_dist(&mut self, dist: Vec<f64>) {
        self.state_dist = dist;
    }

    /// Compute Φ (integrated information)
    ///
    /// Φ = min over partitions of [I(whole) - I(parts)]
    pub fn compute_phi(&self) -> f64 {
        let whole_info = self.whole_system_information();
        let partitions = Partition::bipartitions(self.n_elements);

        if partitions.is_empty() {
            return 0.0;
        }

        let mut min_phi = f64::INFINITY;

        for partition in &partitions {
            let parts_info = self.partitioned_information(partition);
            let phi = (whole_info - parts_info).max(0.0);
            min_phi = min_phi.min(phi);
        }

        min_phi
    }

    /// Information generated by whole system
    fn whole_system_information(&self) -> f64 {
        // H(future) - H(future|past)
        let future_entropy = shannon_entropy(&self.state_dist);
        let cond_entropy = self.conditional_future_entropy();
        (future_entropy - cond_entropy).max(0.0)
    }

    /// Conditional entropy H(future|past)
    fn conditional_future_entropy(&self) -> f64 {
        let mut h = 0.0;
        for (i, &p_past) in self.state_dist.iter().enumerate() {
            if p_past > MIN_ENTROPY {
                for &p_future_given_past in &self.tpm[i] {
                    if p_future_given_past > MIN_ENTROPY {
                        h -= p_past * p_future_given_past * p_future_given_past.log2();
                    }
                }
            }
        }
        h
    }

    /// Information generated by partitioned system
    fn partitioned_information(&self, partition: &Partition) -> f64 {
        // Simplified: sum of marginal informations
        let info_a = self.marginal_information(&partition.part_a);
        let info_b = self.marginal_information(&partition.part_b);
        info_a + info_b
    }

    /// Marginal information for subset
    fn marginal_information(&self, elements: &[usize]) -> f64 {
        if elements.is_empty() {
            return 0.0;
        }

        // Simplified computation using marginal entropy
        let marginal = self.marginalize(elements);
        shannon_entropy(&marginal)
    }

    /// Marginalize distribution over subset
    fn marginalize(&self, elements: &[usize]) -> Vec<f64> {
        let n_marginal = 1 << elements.len();
        let mut marginal = vec![0.0; n_marginal];

        for (state, &prob) in self.state_dist.iter().enumerate() {
            let mut marginal_state = 0;
            for (i, &elem) in elements.iter().enumerate() {
                if (state >> elem) & 1 == 1 {
                    marginal_state |= 1 << i;
                }
            }
            marginal[marginal_state] += prob;
        }

        marginal
    }
}

// ============================================================================
// Self-Organized Criticality
// ============================================================================

/// Avalanche data for SOC analysis
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Avalanche {
    /// Size (number of activations)
    pub size: usize,
    /// Duration (time steps)
    pub duration: usize,
    /// Peak activity
    pub peak: usize,
    /// Start time
    pub start_time: usize,
}

/// Criticality analysis results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CriticalityAnalysis {
    /// Branching ratio σ
    pub branching_ratio: f64,
    /// Avalanche size exponent τ
    pub size_exponent: f64,
    /// Avalanche duration exponent α
    pub duration_exponent: f64,
    /// Duration-size scaling β
    pub duration_size_scaling: f64,
    /// Hurst exponent H
    pub hurst_exponent: f64,
    /// Whether system is critical
    pub is_critical: bool,
    /// Distance from criticality
    pub criticality_distance: f64,
}

impl Default for CriticalityAnalysis {
    fn default() -> Self {
        Self {
            branching_ratio: 1.0,
            size_exponent: AVALANCHE_EXPONENT_CRITICAL,
            duration_exponent: 2.0,
            duration_size_scaling: 0.5,
            hurst_exponent: CRITICAL_HURST,
            is_critical: false,
            criticality_distance: 0.0,
        }
    }
}

impl CriticalityAnalysis {
    /// Create from avalanche data
    pub fn from_avalanches(avalanches: &[Avalanche]) -> Self {
        if avalanches.is_empty() {
            return Self::default();
        }

        // Compute branching ratio
        let branching_ratio = compute_branching_ratio(avalanches);

        // Estimate power law exponents
        let sizes: Vec<f64> = avalanches.iter().map(|a| a.size as f64).collect();
        let durations: Vec<f64> = avalanches.iter().map(|a| a.duration as f64).collect();

        let size_exponent = estimate_power_law_exponent(&sizes);
        let duration_exponent = estimate_power_law_exponent(&durations);

        // Duration-size scaling: T ~ S^β
        let duration_size_scaling = estimate_duration_scaling(avalanches);

        // Estimate Hurst exponent from peak activity
        let peaks: Vec<f64> = avalanches.iter().map(|a| a.peak as f64).collect();
        let hurst_exponent = estimate_hurst_exponent(&peaks);

        // Check criticality
        let sigma_dist = (branching_ratio - CRITICAL_BRANCHING_RATIO).abs();
        let tau_dist = (size_exponent - AVALANCHE_EXPONENT_CRITICAL).abs();
        let hurst_dist = (hurst_exponent - CRITICAL_HURST).abs();

        let criticality_distance = (sigma_dist + tau_dist + hurst_dist) / 3.0;
        let is_critical = criticality_distance < CRITICALITY_TOLERANCE;

        Self {
            branching_ratio,
            size_exponent,
            duration_exponent,
            duration_size_scaling,
            hurst_exponent,
            is_critical,
            criticality_distance,
        }
    }
}

/// Compute branching ratio from avalanches
fn compute_branching_ratio(avalanches: &[Avalanche]) -> f64 {
    if avalanches.is_empty() {
        return 1.0;
    }

    // σ = average(descendants / ancestors)
    let mut total_ratio = 0.0;
    let mut count = 0;

    for avalanche in avalanches {
        if avalanche.duration > 1 {
            // Approximate branching from size/duration
            let ratio = avalanche.size as f64 / avalanche.duration as f64;
            total_ratio += ratio;
            count += 1;
        }
    }

    if count > 0 {
        total_ratio / count as f64
    } else {
        1.0
    }
}

/// Estimate power law exponent using MLE
fn estimate_power_law_exponent(data: &[f64]) -> f64 {
    if data.is_empty() {
        return AVALANCHE_EXPONENT_CRITICAL;
    }

    // Filter positive values
    let valid: Vec<f64> = data.iter().filter(|&&x| x > 0.0).cloned().collect();
    if valid.is_empty() {
        return AVALANCHE_EXPONENT_CRITICAL;
    }

    let x_min = valid.iter().cloned().fold(f64::INFINITY, f64::min);
    if x_min <= 0.0 {
        return AVALANCHE_EXPONENT_CRITICAL;
    }

    // MLE: α = 1 + n / Σ ln(x_i / x_min)
    let n = valid.len() as f64;
    let sum_log: f64 = valid.iter().map(|&x| (x / x_min).ln()).sum();

    if sum_log > 0.0 {
        1.0 + n / sum_log
    } else {
        AVALANCHE_EXPONENT_CRITICAL
    }
}

/// Estimate duration-size scaling exponent
fn estimate_duration_scaling(avalanches: &[Avalanche]) -> f64 {
    if avalanches.len() < 2 {
        return 0.5;
    }

    // Linear regression on log-log: log(T) = β * log(S) + c
    let mut sum_log_s = 0.0;
    let mut sum_log_t = 0.0;
    let mut sum_log_s_log_t = 0.0;
    let mut sum_log_s_sq = 0.0;
    let mut count = 0;

    for a in avalanches {
        if a.size > 0 && a.duration > 0 {
            let log_s = (a.size as f64).ln();
            let log_t = (a.duration as f64).ln();
            sum_log_s += log_s;
            sum_log_t += log_t;
            sum_log_s_log_t += log_s * log_t;
            sum_log_s_sq += log_s * log_s;
            count += 1;
        }
    }

    if count < 2 {
        return 0.5;
    }

    let n = count as f64;
    let denom = n * sum_log_s_sq - sum_log_s * sum_log_s;

    if denom.abs() < MIN_ENTROPY {
        return 0.5;
    }

    (n * sum_log_s_log_t - sum_log_s * sum_log_t) / denom
}

/// Estimate Hurst exponent using R/S analysis
fn estimate_hurst_exponent(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 10 {
        return CRITICAL_HURST;
    }

    // Simplified R/S analysis
    let mean: f64 = data.iter().sum::<f64>() / n as f64;
    let deviations: Vec<f64> = data.iter().map(|&x| x - mean).collect();

    // Cumulative sum
    let mut cumsum = Vec::with_capacity(n);
    let mut running = 0.0;
    for &d in &deviations {
        running += d;
        cumsum.push(running);
    }

    // Range
    let max_cs = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_cs = cumsum.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = max_cs - min_cs;

    // Standard deviation
    let variance: f64 = deviations.iter().map(|&d| d * d).sum::<f64>() / n as f64;
    let std_dev = variance.sqrt();

    if std_dev < MIN_ENTROPY {
        return CRITICAL_HURST;
    }

    let rs = range / std_dev;

    // H ≈ log(R/S) / log(n)
    if rs > 0.0 && n > 1 {
        rs.ln() / (n as f64).ln()
    } else {
        CRITICAL_HURST
    }
}

// ============================================================================
// Pentagon Emergence (5-engine consciousness)
// ============================================================================

/// Pentagon emergence state for 5-engine pBit topology
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PentagonEmergence {
    /// IIT Phi value
    pub phi: f64,
    /// Criticality analysis
    pub criticality: CriticalityAnalysis,
    /// Phase coherence (Kuramoto)
    pub phase_coherence: f64,
    /// Golden alignment score
    pub golden_alignment: f64,
    /// Emergence level
    pub emergence_level: EmergenceLevel,
}

/// Emergence level classification
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EmergenceLevel {
    /// No emergence detected
    None,
    /// Weak emergence
    Weak,
    /// Moderate emergence
    Moderate,
    /// Strong emergence
    Strong,
    /// Critical emergence (at phase transition)
    Critical,
}

impl PentagonEmergence {
    /// Create from engine states and avalanche history
    pub fn from_states(
        states: &[f64; 5],
        phases: &[f64; 5],
        avalanches: &[Avalanche],
    ) -> Self {
        // Compute Phi (simplified for 5 elements)
        let mut phi_calc = PhiCalculator::new(5);
        let state_dist = states_to_distribution(states);
        phi_calc.set_state_dist(state_dist);
        let phi = phi_calc.compute_phi();

        // Criticality analysis
        let criticality = CriticalityAnalysis::from_avalanches(avalanches);

        // Phase coherence (Kuramoto order parameter)
        let phase_coherence = compute_phase_coherence(phases);

        // Golden alignment
        let golden_alignment = compute_golden_alignment(states);

        // Determine emergence level
        let emergence_level = classify_emergence(phi, &criticality, phase_coherence);

        Self {
            phi,
            criticality,
            phase_coherence,
            golden_alignment,
            emergence_level,
        }
    }
}

/// Convert states to probability distribution
fn states_to_distribution(states: &[f64; 5]) -> Vec<f64> {
    let n_states = 32; // 2^5
    let mut dist = vec![0.0; n_states];

    // Use softmax-like distribution based on state energies
    let temp = 1.0;
    let mut total = 0.0;

    for i in 0..n_states {
        let mut energy = 0.0;
        for (j, &state) in states.iter().enumerate() {
            let bit = if (i >> j) & 1 == 1 { 1.0 } else { -1.0 };
            energy -= bit * state;
        }
        dist[i] = (-energy / temp).exp();
        total += dist[i];
    }

    // Normalize
    if total > 0.0 {
        for d in &mut dist {
            *d /= total;
        }
    }

    dist
}

/// Compute Kuramoto order parameter
fn compute_phase_coherence(phases: &[f64; 5]) -> f64 {
    let mut sum_cos = 0.0;
    let mut sum_sin = 0.0;

    for &phase in phases {
        sum_cos += phase.cos();
        sum_sin += phase.sin();
    }

    (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / 5.0
}

/// Compute golden ratio alignment
fn compute_golden_alignment(states: &[f64; 5]) -> f64 {
    let mut alignment = 0.0;

    // Check if adjacent/skip-one ratios follow golden ratio
    for i in 0..5 {
        let adjacent = (i + 1) % 5;
        let skip = (i + 2) % 5;

        if states[adjacent].abs() > MIN_ENTROPY {
            let ratio = states[skip].abs() / states[adjacent].abs();
            alignment += 1.0 - (ratio - PHI_INV).abs().min(1.0);
        }
    }

    alignment / 5.0
}

/// Classify emergence level
fn classify_emergence(
    phi: f64,
    criticality: &CriticalityAnalysis,
    phase_coherence: f64,
) -> EmergenceLevel {
    // Critical emergence if at criticality with high coherence
    if criticality.is_critical && phase_coherence > PHI_INV {
        return EmergenceLevel::Critical;
    }

    // Strong emergence: high Phi and coherence
    if phi > PHI && phase_coherence > PHI_INV {
        return EmergenceLevel::Strong;
    }

    // Moderate emergence: either high Phi or high coherence
    if phi > 1.0 || phase_coherence > PHI_INV {
        return EmergenceLevel::Moderate;
    }

    // Weak emergence: some structure
    if phi > 0.5 || phase_coherence > 0.3 {
        return EmergenceLevel::Weak;
    }

    EmergenceLevel::None
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy() {
        // Uniform distribution has max entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&uniform);
        assert!((h - 2.0).abs() < 0.01); // log2(4) = 2

        // Deterministic distribution has 0 entropy
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&deterministic);
        assert!(h < 0.01);
    }

    #[test]
    fn test_mutual_information() {
        // Independent variables have 0 MI
        let px = vec![0.5, 0.5];
        let py = vec![0.5, 0.5];
        let pxy = vec![vec![0.25, 0.25], vec![0.25, 0.25]];
        let mi = mutual_information(&px, &py, &pxy);
        assert!(mi.abs() < 0.01);

        // Perfectly correlated have max MI
        let pxy_corr = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        let mi = mutual_information(&px, &py, &pxy_corr);
        assert!(mi > 0.9);
    }

    #[test]
    fn test_phi_calculator() {
        let calc = PhiCalculator::new(3);
        let phi = calc.compute_phi();
        assert!(phi >= 0.0);
    }

    #[test]
    fn test_partition_bipartitions() {
        let parts = Partition::bipartitions(3);
        assert!(!parts.is_empty());

        // Each partition should be non-empty on both sides
        for p in &parts {
            assert!(!p.part_a.is_empty());
            assert!(!p.part_b.is_empty());
        }
    }

    #[test]
    fn test_criticality_analysis() {
        let avalanches = vec![
            Avalanche { size: 5, duration: 3, peak: 2, start_time: 0 },
            Avalanche { size: 10, duration: 5, peak: 4, start_time: 10 },
            Avalanche { size: 3, duration: 2, peak: 2, start_time: 20 },
        ];

        let analysis = CriticalityAnalysis::from_avalanches(&avalanches);
        assert!(analysis.branching_ratio > 0.0);
        assert!(analysis.size_exponent > 0.0);
    }

    #[test]
    fn test_phase_coherence() {
        // All same phase → coherence = 1
        let same = [0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((compute_phase_coherence(&same) - 1.0).abs() < 0.01);

        // Uniform phases → low coherence
        use std::f64::consts::PI;
        let uniform: [f64; 5] = std::array::from_fn(|i| 2.0 * PI * i as f64 / 5.0);
        assert!(compute_phase_coherence(&uniform) < 0.2);
    }

    #[test]
    fn test_pentagon_emergence() {
        let states = [0.5, 0.3, 0.8, 0.2, 0.6];
        let phases = [0.0, 0.5, 1.0, 1.5, 2.0];
        let avalanches = vec![];

        let emergence = PentagonEmergence::from_states(&states, &phases, &avalanches);
        assert!(emergence.phi >= 0.0);
        assert!(emergence.phase_coherence >= 0.0 && emergence.phase_coherence <= 1.0);
    }
}
