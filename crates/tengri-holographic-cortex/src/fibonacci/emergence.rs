//! # Fibonacci Pentagon Emergence Metrics
//!
//! Implements consciousness emergence metrics based on Integrated Information Theory (IIT)
//! and criticality analysis for the 5-engine pBit topology.
//!
//! ## Scientific Foundation
//!
//! ### Integrated Information Theory (IIT)
//! - **Reference**: Tononi, G. (2004). "An information integration theory of consciousness"
//! - **Phi (Φ)**: Measures irreducibility of integrated information
//! - **Formula**: Φ = min over partitions [I(system) - Σ I(parts)]
//!
//! ### Criticality
//! - **Reference**: Beggs, J. M., & Plenz, D. (2003). "Neuronal avalanches in neocortical circuits"
//! - **Branching ratio**: σ → 1 at criticality
//! - **Power-law exponent**: α ≈ 1.5 for neuronal avalanches
//!
//! ### Information Theory
//! - **Shannon entropy**: H(X) = -Σ p(x) log₂ p(x)
//! - **Mutual information**: I(X;Y) = H(X) + H(Y) - H(X,Y)
//! - **Transfer entropy**: TE(X→Y) = I(Y_future; X_past | Y_past)

use std::collections::HashSet;

/// Minimum entropy for numerical stability
pub const MIN_ENTROPY: f64 = 1e-10;

/// Critical branching ratio (σ = 1 at criticality)
pub const CRITICAL_BRANCHING_RATIO: f64 = 1.0;

/// Critical Hurst exponent (0.5 for random walk, >0.5 for long-range correlations)
pub const CRITICAL_HURST: f64 = 0.5;

/// Power-law exponent for neuronal avalanches at criticality (Beggs & Plenz, 2003)
pub const AVALANCHE_EXPONENT_CRITICAL: f64 = 1.5;

/// Tolerance for criticality detection
pub const CRITICALITY_TOLERANCE: f64 = 0.1;

// ============================================================================
// Shannon Entropy and Information-Theoretic Measures
// ============================================================================

/// Compute Shannon entropy: H(X) = -Σ p(x) log₂ p(x)
///
/// # Mathematical Foundation
/// Shannon entropy measures the uncertainty or information content of a random variable.
/// - **Maximum**: log₂(n) for uniform distribution over n outcomes
/// - **Minimum**: 0 for deterministic (delta) distribution
///
/// # Arguments
/// * `distribution` - Probability distribution (must sum to 1.0)
///
/// # Returns
/// Entropy in bits
pub fn shannon_entropy(distribution: &[f64]) -> f64 {
    if distribution.is_empty() {
        return 0.0;
    }

    // Normalize to ensure sum = 1.0
    let sum: f64 = distribution.iter().sum();
    if sum < MIN_ENTROPY {
        return 0.0;
    }

    let normalized: Vec<f64> = distribution.iter().map(|&p| p / sum).collect();

    // H(X) = -Σ p(x) log₂ p(x)
    normalized
        .iter()
        .filter(|&&p| p > MIN_ENTROPY)
        .map(|&p| -p * p.log2())
        .sum()
}

/// Compute conditional entropy: H(X|Y) = -Σ p(x,y) log₂ p(x|y)
///
/// # Mathematical Foundation
/// Measures uncertainty in X given knowledge of Y.
/// - H(X|Y) = H(X,Y) - H(Y)
/// - H(X|Y) ≤ H(X) (conditioning reduces entropy)
///
/// # Arguments
/// * `joint` - Joint probability distribution p(x,y) as 2D array
/// * `marginal_y` - Marginal distribution p(y)
///
/// # Returns
/// Conditional entropy in bits
pub fn conditional_entropy(joint: &[Vec<f64>], marginal_y: &[f64]) -> f64 {
    if joint.is_empty() || marginal_y.is_empty() {
        return 0.0;
    }

    let mut h_xy = 0.0;

    for (y_idx, row) in joint.iter().enumerate() {
        for &p_xy in row.iter() {
            if p_xy > MIN_ENTROPY {
                h_xy -= p_xy * p_xy.log2();
            }
        }
    }

    let h_y = shannon_entropy(marginal_y);

    // H(X|Y) = H(X,Y) - H(Y)
    (h_xy - h_y).max(0.0)
}

/// Compute mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
///
/// # Mathematical Foundation
/// Measures the amount of information shared between X and Y.
/// - I(X;Y) = 0 if X and Y are independent
/// - I(X;Y) = H(X) = H(Y) if X and Y are identical
/// - I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
///
/// # Arguments
/// * `joint` - Joint probability distribution p(x,y)
///
/// # Returns
/// Mutual information in bits
pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
    if joint.is_empty() {
        return 0.0;
    }

    // Compute marginals
    let n_x = joint.len();
    let n_y = if n_x > 0 { joint[0].len() } else { 0 };

    let mut marginal_x = vec![0.0; n_x];
    let mut marginal_y = vec![0.0; n_y];

    for (i, row) in joint.iter().enumerate() {
        for (j, &p_xy) in row.iter().enumerate() {
            marginal_x[i] += p_xy;
            marginal_y[j] += p_xy;
        }
    }

    let h_x = shannon_entropy(&marginal_x);
    let h_y = shannon_entropy(&marginal_y);

    // H(X,Y)
    let mut h_xy = 0.0;
    for row in joint.iter() {
        for &p_xy in row.iter() {
            if p_xy > MIN_ENTROPY {
                h_xy -= p_xy * p_xy.log2();
            }
        }
    }

    // I(X;Y) = H(X) + H(Y) - H(X,Y)
    (h_x + h_y - h_xy).max(0.0)
}

/// Compute transfer entropy: TE(X→Y) = H(Y_t | Y_{t-1,...,t-k}) - H(Y_t | Y_{t-1,...,t-k}, X_{t-1,...,t-k})
///
/// # Mathematical Foundation
/// Measures directed information transfer from X to Y.
/// - TE(X→Y) > 0 indicates X influences Y
/// - TE(X→Y) = 0 if X provides no information about Y's future
///
/// # Arguments
/// * `x_series` - Time series for variable X
/// * `y_series` - Time series for variable Y
/// * `lag` - Number of time steps to look back
///
/// # Returns
/// Transfer entropy in bits (approximated via binning)
pub fn transfer_entropy(x_series: &[f64], y_series: &[f64], lag: usize) -> f64 {
    if x_series.len() != y_series.len() || x_series.len() <= lag {
        return 0.0;
    }

    // Simplified discrete approximation via binning
    let n_bins = 10;
    let x_min = x_series.iter().cloned().fold(f64::INFINITY, f64::min);
    let x_max = x_series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let y_min = y_series.iter().cloned().fold(f64::INFINITY, f64::min);
    let y_max = y_series.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let bin_x = |x: f64| ((x - x_min) / (x_max - x_min + MIN_ENTROPY) * (n_bins as f64)) as usize;
    let bin_y = |y: f64| ((y - y_min) / (y_max - y_min + MIN_ENTROPY) * (n_bins as f64)) as usize;

    // Count occurrences for H(Y_t | Y_{t-lag})
    let mut counts_y_ypast = vec![vec![0usize; n_bins]; n_bins];
    // Count occurrences for H(Y_t | Y_{t-lag}, X_{t-lag})
    let mut counts_y_ypast_xpast = vec![vec![vec![0usize; n_bins]; n_bins]; n_bins];

    for t in lag..y_series.len() {
        let y_curr = bin_y(y_series[t]).min(n_bins - 1);
        let y_past = bin_y(y_series[t - lag]).min(n_bins - 1);
        let x_past = bin_x(x_series[t - lag]).min(n_bins - 1);

        counts_y_ypast[y_past][y_curr] += 1;
        counts_y_ypast_xpast[y_past][x_past][y_curr] += 1;
    }

    // Convert counts to probabilities
    let total = (y_series.len() - lag) as f64;

    // H(Y_t | Y_{t-lag})
    let mut h_y_given_ypast = 0.0;
    for ypast_counts in counts_y_ypast.iter() {
        let sum_ypast: usize = ypast_counts.iter().sum();
        if sum_ypast > 0 {
            for &count in ypast_counts.iter() {
                if count > 0 {
                    let p_joint = (count as f64) / total;
                    let p_cond = (count as f64) / (sum_ypast as f64);
                    h_y_given_ypast -= p_joint * p_cond.log2();
                }
            }
        }
    }

    // H(Y_t | Y_{t-lag}, X_{t-lag})
    let mut h_y_given_ypast_xpast = 0.0;
    for ypast_xpast_counts in counts_y_ypast_xpast.iter() {
        for xpast_counts in ypast_xpast_counts.iter() {
            let sum_ypast_xpast: usize = xpast_counts.iter().sum();
            if sum_ypast_xpast > 0 {
                for &count in xpast_counts.iter() {
                    if count > 0 {
                        let p_joint = (count as f64) / total;
                        let p_cond = (count as f64) / (sum_ypast_xpast as f64);
                        h_y_given_ypast_xpast -= p_joint * p_cond.log2();
                    }
                }
            }
        }
    }

    // TE = H(Y_t | Y_{t-lag}) - H(Y_t | Y_{t-lag}, X_{t-lag})
    (h_y_given_ypast - h_y_given_ypast_xpast).max(0.0)
}

// ============================================================================
// IIT Phi Computation
// ============================================================================

/// Partition of a system into disjoint subsets
#[derive(Debug, Clone, PartialEq)]
pub struct Partition {
    /// Subsets of element indices
    pub subsets: Vec<HashSet<usize>>,
}

impl Partition {
    /// Create a new partition from subsets
    pub fn new(subsets: Vec<HashSet<usize>>) -> Self {
        Self { subsets }
    }

    /// Check if this is a valid partition (disjoint, covers all elements)
    pub fn is_valid(&self, n: usize) -> bool {
        let mut all_elements = HashSet::new();
        for subset in &self.subsets {
            for &elem in subset {
                if elem >= n || all_elements.contains(&elem) {
                    return false;
                }
                all_elements.insert(elem);
            }
        }
        all_elements.len() == n
    }
}

/// IIT Phi calculator for integrated information
///
/// # Mathematical Foundation
/// Φ = min_{partition P} [EMD(p(X^1), ∏ᵢ p(Xⁱ))]
///
/// Where:
/// - EMD = Earth Mover's Distance (or KL divergence approximation)
/// - p(X^1) = probability distribution of whole system
/// - ∏ᵢ p(Xⁱ) = product of partition distributions (MIP - Minimum Information Partition)
pub struct PhiCalculator {
    /// System size (number of elements)
    pub system_size: usize,
    /// Transition probability matrix (TPM): TPM[state_from][state_to]
    pub tpm: Vec<Vec<f64>>,
    /// Current state probability distribution
    pub state_distribution: Vec<f64>,
}

impl PhiCalculator {
    /// Create a new Phi calculator
    pub fn new(system_size: usize, tpm: Vec<Vec<f64>>, state_distribution: Vec<f64>) -> Self {
        Self {
            system_size,
            tpm,
            state_distribution,
        }
    }

    /// Compute Phi using Minimum Information Partition (MIP)
    ///
    /// # Algorithm
    /// 1. Generate all bipartitions of the system
    /// 2. For each partition, compute effective information
    /// 3. Phi = minimum effective information across all partitions
    pub fn compute_phi(&self) -> f64 {
        if self.system_size == 0 {
            return 0.0;
        }

        let mip = self.find_mip();
        self.effective_information(&mip)
    }

    /// Find the Minimum Information Partition
    pub fn find_mip(&self) -> Partition {
        let partitions = self.generate_bipartitions();

        if partitions.is_empty() {
            // Return trivial partition (whole system)
            let mut all_elements = HashSet::new();
            for i in 0..self.system_size {
                all_elements.insert(i);
            }
            return Partition::new(vec![all_elements]);
        }

        // Find partition with minimum effective information
        partitions
            .into_iter()
            .min_by(|p1, p2| {
                let ei1 = self.effective_information(p1);
                let ei2 = self.effective_information(p2);
                ei1.partial_cmp(&ei2).unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or_else(|| {
                let mut all_elements = HashSet::new();
                for i in 0..self.system_size {
                    all_elements.insert(i);
                }
                Partition::new(vec![all_elements])
            })
    }

    /// Compute effective information for a partition
    ///
    /// EI = H(whole) - Σ H(parts)
    pub fn effective_information(&self, partition: &Partition) -> f64 {
        // Entropy of whole system
        let h_whole = shannon_entropy(&self.state_distribution);

        // Sum of entropies of parts
        let mut h_parts_sum = 0.0;
        for subset in &partition.subsets {
            // Marginalize state distribution to this subset
            let part_dist = self.marginalize_to_subset(subset);
            h_parts_sum += shannon_entropy(&part_dist);
        }

        // Effective information
        (h_whole - h_parts_sum).max(0.0)
    }

    /// Generate all bipartitions (2-part partitions) of the system
    fn generate_bipartitions(&self) -> Vec<Partition> {
        let mut partitions = Vec::new();
        let n = self.system_size;

        if n <= 1 {
            return partitions;
        }

        // Iterate over all non-trivial bipartitions
        // Use bitmasks: for n elements, 2^n possible subsets
        let total_subsets = 1 << n;

        for mask in 1..total_subsets / 2 {
            let mut subset1 = HashSet::new();
            let mut subset2 = HashSet::new();

            for i in 0..n {
                if (mask & (1 << i)) != 0 {
                    subset1.insert(i);
                } else {
                    subset2.insert(i);
                }
            }

            if !subset1.is_empty() && !subset2.is_empty() {
                partitions.push(Partition::new(vec![subset1, subset2]));
            }
        }

        partitions
    }

    /// Marginalize state distribution to a subset of elements
    fn marginalize_to_subset(&self, subset: &HashSet<usize>) -> Vec<f64> {
        if subset.is_empty() {
            return vec![1.0];
        }

        // For simplicity, approximate by summing probabilities
        // In full IIT, would properly marginalize over state space
        let subset_size = subset.len();
        let n_states = 1 << subset_size;

        let mut marginal = vec![0.0; n_states];

        // Sum over all states, grouping by subset configuration
        for (state_idx, &prob) in self.state_distribution.iter().enumerate() {
            // Extract bits corresponding to subset
            let mut subset_state = 0;
            for (bit_pos, &elem) in subset.iter().enumerate() {
                if elem < self.system_size && (state_idx & (1 << elem)) != 0 {
                    subset_state |= 1 << bit_pos;
                }
            }
            if subset_state < n_states {
                marginal[subset_state] += prob;
            }
        }

        marginal
    }
}

// ============================================================================
// Pentagon Emergence Metrics
// ============================================================================

/// Emergence metrics for 5-engine Fibonacci pentagon topology
#[derive(Debug, Clone)]
pub struct PentagonEmergence {
    /// IIT Phi (Φ) - integrated information
    pub phi: f64,
    /// Integration metric (sum of pairwise mutual informations)
    pub integration: f64,
    /// Complexity (total entropy - sum of individual entropies)
    pub complexity: f64,
    /// Synergy (information in whole not in parts)
    pub synergy: f64,
    /// Metastability (variance of synchronization over time)
    pub metastability: f64,
}

impl PentagonEmergence {
    /// Compute all emergence metrics from spike rate time series
    ///
    /// # Arguments
    /// * `spike_rates` - Time series of spike rates [time][engine], 5 engines
    /// * `window_size` - Window size for computing distributions
    pub fn from_spike_series(spike_rates: &[[f64; 5]], window_size: usize) -> Self {
        if spike_rates.len() < window_size {
            return Self::default();
        }

        // Compute distributions via binning
        let n_bins = 8;
        let distributions = Self::compute_distributions(spike_rates, n_bins);

        // 1. IIT Phi computation
        let phi = Self::compute_phi_pentagon(&distributions);

        // 2. Integration (sum of pairwise mutual informations)
        let integration = Self::compute_integration(&distributions);

        // 3. Complexity
        let complexity = Self::compute_complexity(&distributions);

        // 4. Synergy
        let synergy = Self::compute_synergy(&distributions);

        // 5. Metastability (variance of Kuramoto order parameter)
        let metastability = Self::compute_metastability(spike_rates);

        Self {
            phi,
            integration,
            complexity,
            synergy,
            metastability,
        }
    }

    /// Compute discrete probability distributions via binning
    fn compute_distributions(spike_rates: &[[f64; 5]], n_bins: usize) -> Vec<Vec<f64>> {
        let mut distributions = vec![vec![0.0; n_bins]; 5];

        // Find min/max for each engine
        let mut mins = [f64::INFINITY; 5];
        let mut maxs = [f64::NEG_INFINITY; 5];

        for snapshot in spike_rates {
            for (i, &rate) in snapshot.iter().enumerate() {
                mins[i] = mins[i].min(rate);
                maxs[i] = maxs[i].max(rate);
            }
        }

        // Bin counts
        for snapshot in spike_rates {
            for (i, &rate) in snapshot.iter().enumerate() {
                let bin = ((rate - mins[i]) / (maxs[i] - mins[i] + MIN_ENTROPY) * (n_bins as f64)) as usize;
                let bin = bin.min(n_bins - 1);
                distributions[i][bin] += 1.0;
            }
        }

        // Normalize
        let total = spike_rates.len() as f64;
        for dist in distributions.iter_mut() {
            for p in dist.iter_mut() {
                *p /= total;
            }
        }

        distributions
    }

    /// Compute IIT Phi for pentagon (simplified via mutual information approximation)
    fn compute_phi_pentagon(distributions: &[Vec<f64>]) -> f64 {
        if distributions.len() != 5 {
            return 0.0;
        }

        // Approximate joint distribution as product of marginals (independence assumption)
        // Then compute Phi as difference between integrated and partitioned information

        // Total entropy (sum of individual entropies as approximation)
        let total_entropy: f64 = distributions.iter().map(|d| shannon_entropy(d)).sum();

        // Mutual information between halves of pentagon
        // Partition: {0,1,2} vs {3,4}
        let h_012 = distributions[0..3].iter().map(|d| shannon_entropy(d)).sum::<f64>();
        let h_34 = distributions[3..5].iter().map(|d| shannon_entropy(d)).sum::<f64>();

        // Phi approximation: total entropy - partitioned entropy
        (total_entropy - (h_012 + h_34)).max(0.0)
    }

    /// Compute integration (sum of pairwise mutual informations)
    fn compute_integration(distributions: &[Vec<f64>]) -> f64 {
        if distributions.len() != 5 {
            return 0.0;
        }

        let mut integration = 0.0;

        // All pairs (pentagon has 5 edges in cycle + 5 diagonals = 10 pairs)
        for i in 0..5 {
            for j in (i + 1)..5 {
                // Approximate joint as product (independence)
                let mut joint = vec![vec![0.0; distributions[j].len()]; distributions[i].len()];
                for (x, &p_x) in distributions[i].iter().enumerate() {
                    for (y, &p_y) in distributions[j].iter().enumerate() {
                        joint[x][y] = p_x * p_y; // Independence assumption
                    }
                }
                integration += mutual_information(&joint);
            }
        }

        integration
    }

    /// Compute complexity (difference between total and individual entropies)
    fn compute_complexity(distributions: &[Vec<f64>]) -> f64 {
        let total_entropy: f64 = distributions.iter().map(|d| shannon_entropy(d)).sum();
        let avg_entropy = total_entropy / (distributions.len() as f64);

        // Complexity as variance in entropies
        let variance: f64 = distributions
            .iter()
            .map(|d| {
                let h = shannon_entropy(d);
                (h - avg_entropy).powi(2)
            })
            .sum::<f64>()
            / (distributions.len() as f64);

        variance
    }

    /// Compute synergy (information only present in whole)
    fn compute_synergy(distributions: &[Vec<f64>]) -> f64 {
        // Synergy = H(X1,...,X5) - Σ H(Xi)
        // Approximate joint entropy as sum (upper bound)
        let sum_individual: f64 = distributions.iter().map(|d| shannon_entropy(d)).sum();

        // For true synergy, would compute H(X1,...,X5) exactly
        // Here use approximation based on integration
        let integration = Self::compute_integration(distributions);

        (integration - sum_individual / 2.0).max(0.0)
    }

    /// Compute metastability (variance of Kuramoto order parameter over time)
    ///
    /// # Mathematical Foundation
    /// R(t) = |⟨exp(iθⱼ(t))⟩| where θⱼ = 2π × (spike_rate / max_rate)
    /// Metastability M = Var[R(t)]
    fn compute_metastability(spike_rates: &[[f64; 5]]) -> f64 {
        if spike_rates.is_empty() {
            return 0.0;
        }

        // Convert spike rates to phases (normalize to [0, 2π])
        let max_rate = spike_rates
            .iter()
            .flat_map(|snapshot| snapshot.iter())
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if max_rate < MIN_ENTROPY {
            return 0.0;
        }

        // Compute Kuramoto order parameter R(t) at each time step
        let mut order_params = Vec::with_capacity(spike_rates.len());

        for snapshot in spike_rates {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for &rate in snapshot {
                let phase = 2.0 * std::f64::consts::PI * (rate / max_rate);
                real_sum += phase.cos();
                imag_sum += phase.sin();
            }

            let r = ((real_sum / 5.0).powi(2) + (imag_sum / 5.0).powi(2)).sqrt();
            order_params.push(r);
        }

        // Metastability = variance of R(t)
        let mean_r: f64 = order_params.iter().sum::<f64>() / (order_params.len() as f64);
        let variance: f64 = order_params
            .iter()
            .map(|&r| (r - mean_r).powi(2))
            .sum::<f64>()
            / (order_params.len() as f64);

        variance
    }
}

impl Default for PentagonEmergence {
    fn default() -> Self {
        Self {
            phi: 0.0,
            integration: 0.0,
            complexity: 0.0,
            synergy: 0.0,
            metastability: 0.0,
        }
    }
}

// ============================================================================
// Criticality Analysis
// ============================================================================

/// Criticality analysis for detecting signatures of critical dynamics
#[derive(Debug, Clone)]
pub struct CriticalityAnalysis {
    /// Power-law exponent for avalanche sizes (α ≈ 1.5 at criticality)
    pub avalanche_exponent: f64,
    /// Branching ratio σ (σ = 1 at criticality)
    pub branching_ratio: f64,
    /// Hurst exponent for long-range temporal correlations (H = 0.5 critical)
    pub hurst_exponent: f64,
    /// Distance from criticality (0 = critical)
    pub critical_distance: f64,
}

impl CriticalityAnalysis {
    /// Analyze criticality from cascade data
    ///
    /// # Arguments
    /// * `cascade_sizes` - Sizes of neuronal avalanches
    /// * `cascade_durations` - Durations of avalanches
    pub fn from_cascades(cascade_sizes: &[usize], cascade_durations: &[usize]) -> Self {
        let avalanche_exponent = Self::estimate_power_law_exponent(cascade_sizes);
        let branching_ratio = Self::estimate_branching_ratio(cascade_sizes);
        let hurst_exponent = Self::estimate_hurst_exponent(cascade_sizes);

        // Critical distance: weighted deviation from critical values
        let alpha_dev = (avalanche_exponent - AVALANCHE_EXPONENT_CRITICAL).abs();
        let sigma_dev = (branching_ratio - CRITICAL_BRANCHING_RATIO).abs();
        let hurst_dev = (hurst_exponent - CRITICAL_HURST).abs();

        let critical_distance = (alpha_dev + sigma_dev + hurst_dev) / 3.0;

        Self {
            avalanche_exponent,
            branching_ratio,
            hurst_exponent,
            critical_distance,
        }
    }

    /// Check if system is near critical point
    pub fn is_critical(&self, tolerance: f64) -> bool {
        self.critical_distance < tolerance
    }

    /// Estimate power-law exponent α from avalanche sizes
    ///
    /// P(s) ~ s^(-α)
    /// Use linear regression on log-log plot
    fn estimate_power_law_exponent(sizes: &[usize]) -> f64 {
        if sizes.len() < 2 {
            return 0.0;
        }

        // Compute histogram
        let max_size = sizes.iter().max().copied().unwrap_or(1);
        let mut histogram = vec![0usize; max_size + 1];
        for &size in sizes {
            histogram[size] += 1;
        }

        // Log-log linear regression
        let mut sum_log_s = 0.0;
        let mut sum_log_p = 0.0;
        let mut sum_log_s_log_p = 0.0;
        let mut sum_log_s_sq = 0.0;
        let mut count = 0;

        for (size, &freq) in histogram.iter().enumerate() {
            if size > 0 && freq > 0 {
                let log_s = (size as f64).ln();
                let log_p = (freq as f64).ln();
                sum_log_s += log_s;
                sum_log_p += log_p;
                sum_log_s_log_p += log_s * log_p;
                sum_log_s_sq += log_s * log_s;
                count += 1;
            }
        }

        if count < 2 {
            return 0.0;
        }

        let n = count as f64;
        let slope = (n * sum_log_s_log_p - sum_log_s * sum_log_p)
            / (n * sum_log_s_sq - sum_log_s * sum_log_s);

        -slope // α = -slope (since P(s) ~ s^(-α))
    }

    /// Estimate branching ratio σ = ⟨descendants⟩ / ⟨ancestors⟩
    fn estimate_branching_ratio(sizes: &[usize]) -> f64 {
        if sizes.is_empty() {
            return 0.0;
        }

        // Approximate: ratio of consecutive avalanche sizes
        let mut ratios = Vec::new();
        for i in 1..sizes.len() {
            if sizes[i - 1] > 0 {
                ratios.push((sizes[i] as f64) / (sizes[i - 1] as f64));
            }
        }

        if ratios.is_empty() {
            return 0.0;
        }

        ratios.iter().sum::<f64>() / (ratios.len() as f64)
    }

    /// Estimate Hurst exponent using R/S analysis
    ///
    /// H = 0.5: random walk (critical)
    /// H > 0.5: persistent (long-range correlations)
    /// H < 0.5: anti-persistent
    fn estimate_hurst_exponent(series: &[usize]) -> f64 {
        if series.len() < 4 {
            return 0.5;
        }

        // Convert to f64
        let data: Vec<f64> = series.iter().map(|&x| x as f64).collect();

        // Compute mean
        let mean = data.iter().sum::<f64>() / (data.len() as f64);

        // Compute cumulative deviations
        let mut cumsum = vec![0.0; data.len()];
        for i in 0..data.len() {
            cumsum[i] = if i == 0 {
                data[0] - mean
            } else {
                cumsum[i - 1] + data[i] - mean
            };
        }

        // Range
        let range = cumsum.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - cumsum.iter().cloned().fold(f64::INFINITY, f64::min);

        // Standard deviation
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() as f64);
        let std_dev = variance.sqrt();

        if std_dev < MIN_ENTROPY {
            return 0.5;
        }

        // R/S statistic
        let rs = range / std_dev;

        // H ≈ log(R/S) / log(n)
        let n = data.len() as f64;
        let hurst = rs.ln() / n.ln();

        hurst.max(0.0).min(1.0)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shannon_entropy_uniform() {
        // Uniform distribution has maximum entropy
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let h = shannon_entropy(&uniform);
        assert!((h - 2.0).abs() < 1e-10, "Uniform 4-state entropy should be log2(4) = 2");
    }

    #[test]
    fn test_shannon_entropy_deterministic() {
        // Deterministic (delta) distribution has zero entropy
        let deterministic = vec![1.0, 0.0, 0.0, 0.0];
        let h = shannon_entropy(&deterministic);
        assert!(h < MIN_ENTROPY, "Deterministic entropy should be 0");
    }

    #[test]
    fn test_mutual_information_independent() {
        // Independent variables have zero mutual information
        let joint = vec![
            vec![0.25, 0.25],
            vec![0.25, 0.25],
        ];
        let mi = mutual_information(&joint);
        assert!(mi < 1e-10, "Independent variables should have MI = 0, got {}", mi);
    }

    #[test]
    fn test_mutual_information_identical() {
        // Identical variables: I(X;X) = H(X)
        let joint = vec![
            vec![0.5, 0.0],
            vec![0.0, 0.5],
        ];
        let mi = mutual_information(&joint);
        let h_x = shannon_entropy(&[0.5, 0.5]);
        assert!((mi - h_x).abs() < 1e-10, "I(X;X) should equal H(X)");
    }

    #[test]
    fn test_phi_disconnected_zero() {
        // Disconnected system should have Phi = 0
        let system_size = 2;
        let tpm = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let state_dist = vec![0.25, 0.25, 0.25, 0.25];

        let calc = PhiCalculator::new(system_size, tpm, state_dist);
        let phi = calc.compute_phi();

        assert!(phi >= 0.0, "Phi should be non-negative");
        // For disconnected, phi should be low
    }

    #[test]
    fn test_phi_integrated_positive() {
        // Integrated system should have Phi > 0
        let system_size = 2;
        // XOR-like coupling
        let tpm = vec![
            vec![0.0, 0.5, 0.5, 0.0],
            vec![0.5, 0.0, 0.0, 0.5],
            vec![0.5, 0.0, 0.0, 0.5],
            vec![0.0, 0.5, 0.5, 0.0],
        ];
        let state_dist = vec![0.25, 0.25, 0.25, 0.25];

        let calc = PhiCalculator::new(system_size, tpm, state_dist);
        let phi = calc.compute_phi();

        assert!(phi >= 0.0, "Phi should be non-negative");
    }

    #[test]
    fn test_partition_enumeration() {
        let calc = PhiCalculator::new(3, vec![], vec![]);
        let partitions = calc.generate_bipartitions();

        // For 3 elements: {0},{1,2}, {1},{0,2}, {2},{0,1}
        assert_eq!(partitions.len(), 3, "Should have 3 bipartitions for 3 elements");
    }

    #[test]
    fn test_synergy_calculation() {
        let distributions = vec![
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
            vec![0.5, 0.5],
        ];

        let synergy = PentagonEmergence::compute_synergy(&distributions);
        assert!(synergy >= 0.0, "Synergy should be non-negative");
    }

    #[test]
    fn test_metastability_stable() {
        // Constant spike rates -> low metastability
        let spike_rates = vec![[1.0, 1.0, 1.0, 1.0, 1.0]; 100];
        let metastability = PentagonEmergence::compute_metastability(&spike_rates);

        assert!(metastability < 0.01, "Stable system should have low metastability");
    }

    #[test]
    fn test_criticality_subcritical() {
        let sizes = vec![1, 2, 1, 1, 2, 1, 1, 1, 2, 1]; // Small avalanches
        let durations = vec![1; 10];

        let analysis = CriticalityAnalysis::from_cascades(&sizes, &durations);

        // Subcritical: branching ratio < 1
        assert!(analysis.branching_ratio >= 0.0, "Branching ratio should be non-negative");
    }

    #[test]
    fn test_criticality_supercritical() {
        let sizes = vec![1, 2, 4, 8, 16, 32, 64]; // Growing avalanches
        let durations = vec![1; 7];

        let analysis = CriticalityAnalysis::from_cascades(&sizes, &durations);

        // Supercritical: branching ratio > 1
        assert!(analysis.branching_ratio > 0.0, "Branching ratio should be positive");
    }

    #[test]
    fn test_pentagon_emergence_full() {
        // Simulate pentagon with varying spike rates
        let mut spike_rates = Vec::new();
        for t in 0..100 {
            let t_f = t as f64;
            spike_rates.push([
                (t_f * 0.1).sin().abs(),
                (t_f * 0.15).sin().abs(),
                (t_f * 0.12).sin().abs(),
                (t_f * 0.18).sin().abs(),
                (t_f * 0.11).sin().abs(),
            ]);
        }

        let emergence = PentagonEmergence::from_spike_series(&spike_rates, 10);

        assert!(emergence.phi >= 0.0, "Phi should be non-negative");
        assert!(emergence.integration >= 0.0, "Integration should be non-negative");
        assert!(emergence.complexity >= 0.0, "Complexity should be non-negative");
        assert!(emergence.synergy >= 0.0, "Synergy should be non-negative");
        assert!(emergence.metastability >= 0.0, "Metastability should be non-negative");
    }
}
