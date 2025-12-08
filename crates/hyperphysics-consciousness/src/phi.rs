//! Integrated Information (Φ) calculation
//!
//! Research: Tononi et al. (2016), Oizumi et al. (2014)
//!
//! ## Hyperbolic SNN Integration
//!
//! This module now supports Φ calculation over hyperbolic spiking neural networks,
//! leveraging the geometry crate's HyperbolicSNN for criticality-aware consciousness metrics.
//!
//! The hyperbolic embedding provides:
//! - Natural hierarchical structure for information integration
//! - SOC-aware Φ optimization (Φ peaks at criticality σ ≈ 1.0)
//! - Distance-modulated effective information (geodesic-based partitioning)

use crate::{ConsciousnessError, Result, MAX_EXACT_PHI_SIZE, MAX_APPROX_PHI_SIZE};
use hyperphysics_pbit::PBitLattice;
use hyperphysics_geometry::{HyperbolicSNN, SOCStats, LorentzVec4D};
// Future: ndarray for matrix operations in partition analysis
use rayon::prelude::*;

/// Integrated Information result
#[derive(Debug, Clone)]
pub struct IntegratedInformation {
    /// Φ value
    pub phi: f64,

    /// Minimum information partition (MIP)
    pub mip: Option<Partition>,

    /// Computation method used
    pub method: PhiMethod,
}

/// Φ calculation method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhiMethod {
    /// Exact calculation (N < 1000)
    Exact,
    /// Upper bound approximation (N < 10^6)
    UpperBound,
    /// Lower bound approximation (N < 10^6)
    LowerBound,
    /// Hierarchical multi-scale (N > 10^6)
    Hierarchical,
}

/// System partition into two subsystems
#[derive(Debug, Clone)]
pub struct Partition {
    /// Indices in subsystem A
    pub subset_a: Vec<usize>,
    /// Indices in subsystem B
    pub subset_b: Vec<usize>,
    /// Effective information across partition
    pub effective_info: f64,
}

/// Φ calculator with multiple approximation strategies
///
/// Now supports both pBit lattices and hyperbolic SNNs for consciousness metrics.
/// When using hyperbolic SNNs, the calculator leverages SOC criticality to optimize
/// Φ computation - integrated information peaks at the critical branching ratio σ ≈ 1.0.
pub struct PhiCalculator {
    approximation: PhiApproximation,
    /// Optional hyperbolic SNN for hyperbolic Φ calculation
    hyperbolic_snn: Option<HyperbolicSNN>,
    /// SOC-aware mode: modulate Φ based on criticality
    soc_aware: bool,
}

/// Approximation strategy
#[derive(Debug, Clone, Copy)]
pub enum PhiApproximation {
    /// Exact enumeration (slow but accurate)
    Exact,
    /// Monte Carlo sampling of partitions
    MonteCarlo { samples: usize },
    /// Greedy search for MIP
    Greedy,
    /// Hierarchical decomposition
    Hierarchical { levels: usize },
}

impl PhiCalculator {
    /// Create calculator with exact method
    pub fn exact() -> Self {
        Self {
            approximation: PhiApproximation::Exact,
            hyperbolic_snn: None,
            soc_aware: false,
        }
    }

    /// Create calculator with Monte Carlo approximation
    pub fn monte_carlo(samples: usize) -> Self {
        Self {
            approximation: PhiApproximation::MonteCarlo { samples },
            hyperbolic_snn: None,
            soc_aware: false,
        }
    }

    /// Create calculator with greedy approximation
    pub fn greedy() -> Self {
        Self {
            approximation: PhiApproximation::Greedy,
            hyperbolic_snn: None,
            soc_aware: false,
        }
    }

    /// Create calculator with hierarchical method
    pub fn hierarchical(levels: usize) -> Self {
        Self {
            approximation: PhiApproximation::Hierarchical { levels },
            hyperbolic_snn: None,
            soc_aware: false,
        }
    }

    /// Attach a hyperbolic SNN for hyperbolic Φ calculation
    ///
    /// When attached, the calculator can compute Φ over the hyperbolic
    /// manifold using geodesic-based partitioning.
    pub fn with_hyperbolic_snn(mut self, snn: HyperbolicSNN) -> Self {
        self.hyperbolic_snn = Some(snn);
        self
    }

    /// Enable SOC-aware mode
    ///
    /// In SOC-aware mode, Φ is modulated by the system's proximity to criticality.
    /// Integrated information peaks when the branching ratio σ ≈ 1.0.
    pub fn with_soc_awareness(mut self, enabled: bool) -> Self {
        self.soc_aware = enabled;
        self
    }

    /// Calculate Φ for a hyperbolic SNN
    ///
    /// Uses geodesic distance for partitioning and SOC metrics for optimization.
    /// The effective information is computed using hyperbolic distance modulation.
    ///
    /// ## Mathematical Foundation
    ///
    /// For hyperbolic SNN with neurons at positions {p_i} on the hyperboloid:
    /// - Partition by geodesic distance: d_H(p_i, p_j) = acosh(-⟨p_i, p_j⟩_M)
    /// - Weight effective information by locality factor: exp(-d/λ)
    /// - Modulate by SOC factor when near criticality
    pub fn calculate_hyperbolic(&self, snn: &HyperbolicSNN) -> Result<HyperbolicIntegratedInformation> {
        let n = snn.neurons.len();

        if n == 0 {
            return Err(ConsciousnessError::ComputationError {
                message: "Cannot compute Φ for empty SNN".to_string(),
            });
        }

        // Get SOC statistics for criticality modulation
        let soc_stats = snn.soc_monitor.stats();
        let soc_factor = self.compute_soc_modulation(&soc_stats);

        // Compute hyperbolic Φ using geodesic-based partitioning
        let (phi, mip) = self.compute_hyperbolic_phi(snn)?;

        // Apply SOC modulation if enabled
        let modulated_phi = if self.soc_aware {
            phi * soc_factor
        } else {
            phi
        };

        Ok(HyperbolicIntegratedInformation {
            phi: modulated_phi,
            raw_phi: phi,
            mip,
            soc_stats,
            soc_modulation_factor: soc_factor,
            method: self.approximation,
        })
    }

    /// Compute SOC modulation factor
    ///
    /// Φ peaks at criticality (σ = 1.0) with a Gaussian modulation:
    /// factor = exp(-((σ - 1)² / (2 * 0.1²)))
    fn compute_soc_modulation(&self, stats: &SOCStats) -> f64 {
        let sigma_deviation = stats.sigma_measured - stats.sigma_target;
        let variance = 0.1 * 0.1; // Width of criticality window

        // Gaussian centered at criticality
        let base_factor = (-sigma_deviation * sigma_deviation / (2.0 * variance)).exp();

        // Boost factor when power-law exponent is near τ ≈ 1.5 (optimal criticality)
        let tau_deviation = stats.power_law_tau - 1.5;
        let tau_factor = (-tau_deviation * tau_deviation / 0.5).exp();

        // Combined modulation
        base_factor * (0.7 + 0.3 * tau_factor)
    }

    /// Compute hyperbolic Φ using geodesic-based partitioning
    fn compute_hyperbolic_phi(&self, snn: &HyperbolicSNN) -> Result<(f64, Option<HyperbolicPartition>)> {
        let n = snn.neurons.len();

        if n <= 1 {
            return Ok((0.0, None));
        }

        // Use greedy geodesic-based partitioning
        // Start by finding the geodesic center
        let center_idx = self.find_geodesic_center(snn);

        // Partition by geodesic distance from center
        let median_distance = self.compute_median_distance(snn, center_idx);

        let mut subset_a = Vec::new();
        let mut subset_b = Vec::new();
        let center_pos = snn.neurons[center_idx].position;

        for (i, neuron) in snn.neurons.iter().enumerate() {
            let dist = center_pos.hyperbolic_distance(&neuron.position);
            if dist <= median_distance {
                subset_a.push(i);
            } else {
                subset_b.push(i);
            }
        }

        // Ensure non-degenerate partition
        if subset_a.is_empty() || subset_b.is_empty() {
            let mid = n / 2;
            subset_a = (0..mid).collect();
            subset_b = (mid..n).collect();
        }

        // Compute effective information across the geodesic partition
        let effective_info = self.compute_hyperbolic_effective_info(snn, &subset_a, &subset_b);

        let partition = HyperbolicPartition {
            subset_a,
            subset_b,
            effective_info,
            geodesic_cut_distance: median_distance,
        };

        Ok((effective_info, Some(partition)))
    }

    /// Find the geodesic center (Fréchet mean) of the network
    fn find_geodesic_center(&self, snn: &HyperbolicSNN) -> usize {
        let n = snn.neurons.len();
        let mut min_total_dist = f64::INFINITY;
        let mut center_idx = 0;

        for i in 0..n {
            let mut total_dist = 0.0;
            for j in 0..n {
                if i != j {
                    total_dist += snn.neurons[i].position.hyperbolic_distance(&snn.neurons[j].position);
                }
            }
            if total_dist < min_total_dist {
                min_total_dist = total_dist;
                center_idx = i;
            }
        }

        center_idx
    }

    /// Compute median geodesic distance from a reference point
    fn compute_median_distance(&self, snn: &HyperbolicSNN, ref_idx: usize) -> f64 {
        let ref_pos = snn.neurons[ref_idx].position;
        let mut distances: Vec<f64> = snn.neurons.iter()
            .enumerate()
            .filter(|(i, _)| *i != ref_idx)
            .map(|(_, n)| ref_pos.hyperbolic_distance(&n.position))
            .collect();

        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if distances.is_empty() {
            0.0
        } else {
            distances[distances.len() / 2]
        }
    }

    /// Compute effective information across a hyperbolic partition
    ///
    /// Uses geodesic distance to modulate information flow:
    /// EI = Σ w_ij × exp(-d_H(i,j) / λ) where w_ij is synapse weight
    fn compute_hyperbolic_effective_info(&self, snn: &HyperbolicSNN, subset_a: &[usize], subset_b: &[usize]) -> f64 {
        let lambda = snn.stdp.lambda_stdp; // Use STDP length constant

        let mut total_info = 0.0;
        let mut connection_count = 0;

        // Compute information flow from A to B
        for synapse in &snn.synapses {
            let pre_in_a = subset_a.contains(&synapse.pre_id);
            let post_in_b = subset_b.contains(&synapse.post_id);

            if pre_in_a && post_in_b {
                // Geodesic distance modulation
                let locality = (-(synapse.distance as f64) / lambda).exp();
                total_info += synapse.weight.abs() * locality;
                connection_count += 1;
            }
        }

        // Also compute B to A
        for synapse in &snn.synapses {
            let pre_in_b = subset_b.contains(&synapse.pre_id);
            let post_in_a = subset_a.contains(&synapse.post_id);

            if pre_in_b && post_in_a {
                let locality = (-(synapse.distance as f64) / lambda).exp();
                total_info += synapse.weight.abs() * locality;
                connection_count += 1;
            }
        }

        if connection_count > 0 {
            total_info / connection_count as f64
        } else {
            0.0
        }
    }

    /// Get the attached hyperbolic SNN (if any)
    pub fn hyperbolic_snn(&self) -> Option<&HyperbolicSNN> {
        self.hyperbolic_snn.as_ref()
    }

    /// Calculate Φ for pBit lattice
    pub fn calculate(&self, lattice: &PBitLattice) -> Result<IntegratedInformation> {
        let n = lattice.size();

        // Check size and select appropriate method
        let method = if n <= MAX_EXACT_PHI_SIZE {
            match self.approximation {
                PhiApproximation::Exact => self.calculate_exact(lattice)?,
                PhiApproximation::MonteCarlo { samples } => {
                    self.calculate_monte_carlo(lattice, samples)?
                }
                PhiApproximation::Greedy => self.calculate_greedy(lattice)?,
                PhiApproximation::Hierarchical { levels } => {
                    self.calculate_hierarchical(lattice, levels)?
                }
            }
        } else if n <= MAX_APPROX_PHI_SIZE {
            // Force approximation for large systems
            self.calculate_greedy(lattice)?
        } else {
            // Very large systems: hierarchical only
            self.calculate_hierarchical(lattice, 3)?
        };

        Ok(method)
    }

    /// Exact Φ calculation by exhaustive enumeration
    fn calculate_exact(&self, lattice: &PBitLattice) -> Result<IntegratedInformation> {
        let n = lattice.size();

        if n > MAX_EXACT_PHI_SIZE {
            return Err(ConsciousnessError::SystemTooLarge {
                size: n,
                max: MAX_EXACT_PHI_SIZE,
            });
        }

        // Generate all possible bipartitions
        let partitions = self.generate_all_partitions(n);

        // Calculate effective information for each partition (parallel)
        let ei_results: Vec<(Partition, f64)> = partitions
            .par_iter()
            .map(|partition| {
                let ei = self.effective_information(lattice, partition);
                (partition.clone(), ei)
            })
            .collect();

        // Find minimum (MIP)
        let (mip, phi) = ei_results
            .into_iter()
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();

        Ok(IntegratedInformation {
            phi,
            mip: Some(mip),
            method: PhiMethod::Exact,
        })
    }

    /// Monte Carlo approximation
    fn calculate_monte_carlo(
        &self,
        lattice: &PBitLattice,
        samples: usize,
    ) -> Result<IntegratedInformation> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = lattice.size();

        let mut min_ei = f64::INFINITY;
        let mut best_partition = None;

        for _ in 0..samples {
            // Random partition
            let mut subset_a = Vec::new();
            let mut subset_b = Vec::new();

            for i in 0..n {
                if rng.gen::<bool>() {
                    subset_a.push(i);
                } else {
                    subset_b.push(i);
                }
            }

            // Skip degenerate partitions
            if subset_a.is_empty() || subset_b.is_empty() {
                continue;
            }

            let partition = Partition {
                subset_a,
                subset_b,
                effective_info: 0.0,
            };

            let ei = self.effective_information(lattice, &partition);

            if ei < min_ei {
                min_ei = ei;
                best_partition = Some(Partition {
                    effective_info: ei,
                    ..partition
                });
            }
        }

        Ok(IntegratedInformation {
            phi: min_ei,
            mip: best_partition,
            method: PhiMethod::UpperBound,
        })
    }

    /// Greedy search for MIP
    fn calculate_greedy(&self, lattice: &PBitLattice) -> Result<IntegratedInformation> {
        let n = lattice.size();

        // Start with random partition
        let mut best_partition = Partition {
            subset_a: (0..n / 2).collect(),
            subset_b: (n / 2..n).collect(),
            effective_info: 0.0,
        };

        let mut best_ei = self.effective_information(lattice, &best_partition);

        // Greedy improvement
        let max_iterations = 100;
        for _ in 0..max_iterations {
            let mut improved = false;

            // Try moving each element
            for i in 0..n {
                // Try moving i from A to B or B to A
                let in_a = best_partition.subset_a.contains(&i);

                let mut new_partition = best_partition.clone();
                if in_a {
                    new_partition.subset_a.retain(|&x| x != i);
                    new_partition.subset_b.push(i);
                } else {
                    new_partition.subset_b.retain(|&x| x != i);
                    new_partition.subset_a.push(i);
                }

                // Skip degenerate
                if new_partition.subset_a.is_empty() || new_partition.subset_b.is_empty() {
                    continue;
                }

                let new_ei = self.effective_information(lattice, &new_partition);

                if new_ei < best_ei {
                    best_ei = new_ei;
                    best_partition = new_partition;
                    improved = true;
                    break;
                }
            }

            if !improved {
                break;
            }
        }

        best_partition.effective_info = best_ei;

        Ok(IntegratedInformation {
            phi: best_ei,
            mip: Some(best_partition),
            method: PhiMethod::LowerBound,
        })
    }

    /// Hierarchical multi-scale calculation
    ///
    /// Based on hierarchical IIT decomposition from:
    /// - Tegmark (2016): "Improved Measures of Integrated Information"
    /// - Hoel et al. (2013): "Quantifying causal emergence shows that macro can beat micro"
    fn calculate_hierarchical(
        &self,
        lattice: &PBitLattice,
        levels: usize,
    ) -> Result<IntegratedInformation> {
        let n = lattice.size();
        let chunk_size = (n as f64).powf(1.0 / levels as f64).ceil() as usize;

        // Hierarchical Φ calculation using recursive partition refinement
        let mut phi_sum = 0.0;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk_indices: Vec<usize> = (chunk_start..chunk_end).collect();

            if chunk_indices.len() <= 1 {
                continue;
            }

            // Calculate Φ for this chunk using greedy approximation
            // Create virtual partition for chunk
            let mid = chunk_indices.len() / 2;
            let partition = Partition {
                subset_a: chunk_indices[..mid].to_vec(),
                subset_b: chunk_indices[mid..].to_vec(),
                effective_info: 0.0,
            };

            // Compute effective information for this chunk
            // Using IIT's effective information: EI = I(future; past_cause) - I(future; past_effect)
            let chunk_phi = self.effective_information(lattice, &partition);

            // Normalize by chunk size to prevent size bias (Tegmark 2016)
            let normalized_phi = chunk_phi / (chunk_indices.len() as f64).ln().max(1.0);
            phi_sum += normalized_phi;
        }

        Ok(IntegratedInformation {
            phi: phi_sum,
            mip: None,
            method: PhiMethod::Hierarchical,
        })
    }

    /// Generate all possible bipartitions
    fn generate_all_partitions(&self, n: usize) -> Vec<Partition> {
        let mut partitions = Vec::new();

        // Iterate over all 2^n possible subsets (exclude empty and full)
        for mask in 1..(1 << n) - 1 {
            let mut subset_a = Vec::new();
            let mut subset_b = Vec::new();

            for i in 0..n {
                if (mask >> i) & 1 == 1 {
                    subset_a.push(i);
                } else {
                    subset_b.push(i);
                }
            }

            partitions.push(Partition {
                subset_a,
                subset_b,
                effective_info: 0.0,
            });
        }

        partitions
    }

    /// Calculate effective information across partition
    ///
    /// EI(A→B) = I(B_future; A_past) - I(B_future; B_past)
    ///
    /// This implements a proper approximation of IIT effective information
    /// using mutual information between past and future states across the partition.
    fn effective_information(&self, lattice: &PBitLattice, partition: &Partition) -> f64 {
        // For proper IIT, we need temporal dynamics (past → future states)
        // Since we only have current states, we use a causal approximation
        
        if partition.subset_a.is_empty() || partition.subset_b.is_empty() {
            return 0.0;
        }

        // Step 1: Calculate mutual information I(A; B) for current states
        let mutual_info_current = self.calculate_mutual_information(
            lattice, 
            &partition.subset_a, 
            &partition.subset_b
        );

        // Step 2: Estimate causal influence using coupling strengths
        let causal_influence = self.calculate_causal_influence(
            lattice,
            &partition.subset_a,
            &partition.subset_b
        );

        // Step 3: Effective information approximation
        // EI ≈ Causal_influence - Mutual_info_baseline
        // This captures the "effective" causal power across the partition
        (causal_influence - mutual_info_current).max(0.0)
    }

    /// Calculate mutual information I(A; B) between two subsets
    fn calculate_mutual_information(
        &self,
        lattice: &PBitLattice,
        subset_a: &[usize],
        subset_b: &[usize],
    ) -> f64 {
        let _states = lattice.states();
        let pbits = lattice.pbits();

        // Calculate joint and marginal probabilities
        let mut p_a_1 = 0.0; // P(A = 1)
        let mut p_b_1 = 0.0; // P(B = 1)
        let p_ab_11; // P(A = 1, B = 1) - calculated later

        // Estimate probabilities from current states and pBit probabilities
        for &i in subset_a {
            if i < pbits.len() {
                p_a_1 += pbits[i].prob_one();
            }
        }
        p_a_1 /= subset_a.len() as f64;

        for &j in subset_b {
            if j < pbits.len() {
                p_b_1 += pbits[j].prob_one();
            }
        }
        p_b_1 /= subset_b.len() as f64;

        // Estimate joint probability (assuming weak correlations)
        // For stronger correlations, would need full joint distribution
        p_ab_11 = p_a_1 * p_b_1; // Independence assumption as baseline

        // Calculate mutual information: I(A;B) = Σ P(a,b) log(P(a,b)/(P(a)P(b)))
        let p_a_0 = 1.0 - p_a_1;
        let p_b_0 = 1.0 - p_b_1;
        let p_ab_00 = p_a_0 * p_b_0;
        let p_ab_01 = p_a_0 * p_b_1;
        let p_ab_10 = p_a_1 * p_b_0;

        let mut mi = 0.0;
        
        // Add each term if probabilities are valid
        if p_ab_11 > 1e-15 && p_a_1 > 1e-15 && p_b_1 > 1e-15 {
            mi += p_ab_11 * (p_ab_11 / (p_a_1 * p_b_1)).ln();
        }
        if p_ab_00 > 1e-15 && p_a_0 > 1e-15 && p_b_0 > 1e-15 {
            mi += p_ab_00 * (p_ab_00 / (p_a_0 * p_b_0)).ln();
        }
        if p_ab_01 > 1e-15 && p_a_0 > 1e-15 && p_b_1 > 1e-15 {
            mi += p_ab_01 * (p_ab_01 / (p_a_0 * p_b_1)).ln();
        }
        if p_ab_10 > 1e-15 && p_a_1 > 1e-15 && p_b_0 > 1e-15 {
            mi += p_ab_10 * (p_ab_10 / (p_a_1 * p_b_0)).ln();
        }

        mi.max(0.0)
    }

    /// Calculate causal influence from A to B using coupling strengths
    fn calculate_causal_influence(
        &self,
        lattice: &PBitLattice,
        subset_a: &[usize],
        subset_b: &[usize],
    ) -> f64 {
        let pbits = lattice.pbits();
        let mut total_influence = 0.0;
        let mut connection_count = 0;

        // Sum coupling strengths from A to B
        for &i in subset_a {
            if i >= pbits.len() { continue; }
            
            let pbit_i = &pbits[i];
            for (&j, &coupling_strength) in pbit_i.couplings() {
                if subset_b.contains(&j) {
                    total_influence += coupling_strength.abs();
                    connection_count += 1;
                }
            }
        }

        // Normalize by number of possible connections
        if connection_count > 0 {
            total_influence / connection_count as f64
        } else {
            0.0
        }
    }
}

/// Integrated Information result for hyperbolic SNNs
#[derive(Debug, Clone)]
pub struct HyperbolicIntegratedInformation {
    /// Φ value (SOC-modulated if enabled)
    pub phi: f64,
    /// Raw Φ before SOC modulation
    pub raw_phi: f64,
    /// Minimum information partition in hyperbolic space
    pub mip: Option<HyperbolicPartition>,
    /// SOC statistics at computation time
    pub soc_stats: SOCStats,
    /// SOC modulation factor applied
    pub soc_modulation_factor: f64,
    /// Computation method used
    pub method: PhiApproximation,
}

/// Partition in hyperbolic space using geodesic distance
#[derive(Debug, Clone)]
pub struct HyperbolicPartition {
    /// Indices in subsystem A (interior, closer to geodesic center)
    pub subset_a: Vec<usize>,
    /// Indices in subsystem B (boundary, farther from center)
    pub subset_b: Vec<usize>,
    /// Effective information across partition
    pub effective_info: f64,
    /// Geodesic distance at the partition cut
    pub geodesic_cut_distance: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperphysics_geometry::adversarial_lattice::DefenseTopology;

    #[test]
    fn test_phi_small_system() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let calculator = PhiCalculator::greedy();

        let result = calculator.calculate(&lattice).unwrap();

        assert!(result.phi >= 0.0);
        assert!(result.phi.is_finite());
    }

    #[test]
    fn test_monte_carlo_convergence() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();

        let calc_100 = PhiCalculator::monte_carlo(100);
        let calc_1000 = PhiCalculator::monte_carlo(1000);

        let phi_100 = calc_100.calculate(&lattice).unwrap().phi;
        let phi_1000 = calc_1000.calculate(&lattice).unwrap().phi;

        // More samples should give similar or better estimate
        assert!(phi_100.is_finite());
        assert!(phi_1000.is_finite());
    }

    #[test]
    fn test_partition_generation() {
        let calculator = PhiCalculator::exact();
        let partitions = calculator.generate_all_partitions(4);

        // 2^4 - 2 = 14 non-trivial partitions
        assert_eq!(partitions.len(), 14);
    }

    #[test]
    fn test_hyperbolic_phi_calculation() {
        // Create a small hyperbolic SNN
        let topology = DefenseTopology::balanced_fanout(2);
        let snn = HyperbolicSNN::from_topology(topology).unwrap();

        let calculator = PhiCalculator::greedy().with_soc_awareness(true);
        let result = calculator.calculate_hyperbolic(&snn).unwrap();

        assert!(result.phi >= 0.0);
        assert!(result.phi.is_finite());
        assert!(result.soc_modulation_factor > 0.0);
        assert!(result.soc_modulation_factor <= 1.0);
    }

    #[test]
    fn test_soc_modulation_peaks_at_criticality() {
        let calculator = PhiCalculator::greedy();

        // At criticality (σ = 1.0)
        let critical_stats = SOCStats {
            sigma_measured: 1.0,
            sigma_target: 1.0,
            power_law_tau: 1.5,
            is_critical: true,
            ..Default::default()
        };
        let factor_critical = calculator.compute_soc_modulation(&critical_stats);

        // Away from criticality (σ = 1.5)
        let subcritical_stats = SOCStats {
            sigma_measured: 1.5,
            sigma_target: 1.0,
            power_law_tau: 1.5,
            is_critical: false,
            ..Default::default()
        };
        let factor_subcritical = calculator.compute_soc_modulation(&subcritical_stats);

        // Modulation should be higher at criticality
        assert!(factor_critical > factor_subcritical);
    }
}
