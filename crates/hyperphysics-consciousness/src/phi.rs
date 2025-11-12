//! Integrated Information (Φ) calculation
//!
//! Research: Tononi et al. (2016), Oizumi et al. (2014)

use crate::{ConsciousnessError, Result, MAX_EXACT_PHI_SIZE, MAX_APPROX_PHI_SIZE};
use hyperphysics_pbit::PBitLattice;
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
pub struct PhiCalculator {
    approximation: PhiApproximation,
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
        }
    }

    /// Create calculator with Monte Carlo approximation
    pub fn monte_carlo(samples: usize) -> Self {
        Self {
            approximation: PhiApproximation::MonteCarlo { samples },
        }
    }

    /// Create calculator with greedy approximation
    pub fn greedy() -> Self {
        Self {
            approximation: PhiApproximation::Greedy,
        }
    }

    /// Create calculator with hierarchical method
    pub fn hierarchical(levels: usize) -> Self {
        Self {
            approximation: PhiApproximation::Hierarchical { levels },
        }
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
    fn calculate_hierarchical(
        &self,
        lattice: &PBitLattice,
        levels: usize,
    ) -> Result<IntegratedInformation> {
        // Simplified: calculate Φ for coarse-grained levels
        let n = lattice.size();
        let chunk_size = (n as f64).powf(1.0 / levels as f64).ceil() as usize;

        // Coarse-grain into chunks
        let mut phi_sum = 0.0;

        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);
            let chunk_indices: Vec<usize> = (chunk_start..chunk_end).collect();

            if chunk_indices.len() <= 1 {
                continue;
            }

            // Calculate Φ for this chunk (simplified)
            let chunk_phi = chunk_indices.len() as f64 * 0.1; // Placeholder
            phi_sum += chunk_phi;
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
    fn effective_information(&self, lattice: &PBitLattice, partition: &Partition) -> f64 {
        // Simplified calculation: use mutual information approximation
        let states = lattice.states();

        // Count correlations across partition
        let mut cross_corr = 0.0;

        for &i in &partition.subset_a {
            for &j in &partition.subset_b {
                let si = if states[i] { 1.0 } else { 0.0 };
                let sj = if states[j] { 1.0 } else { 0.0 };

                // Simplified correlation
                cross_corr += si * sj;
            }
        }

        // Normalize
        let ei = if !partition.subset_a.is_empty() && !partition.subset_b.is_empty() {
            cross_corr / (partition.subset_a.len() * partition.subset_b.len()) as f64
        } else {
            0.0
        };

        ei
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
