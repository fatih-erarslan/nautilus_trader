//! Coupling network with exponential hyperbolic decay
//!
//! Research: Krioukov et al. (2010) "Hyperbolic geometry of complex networks" PRE 82:036106

use crate::{PBitLattice, Result};
use hyperphysics_geometry::HyperbolicDistance;

/// Coupling network manager
///
/// Implements exponential decay: J_ij = J0 * exp(-d_H(i,j) / λ)
pub struct CouplingNetwork {
    /// Coupling strength at zero distance
    j0: f64,

    /// Coupling length scale
    lambda: f64,

    /// Minimum coupling threshold (for sparsity)
    j_min: f64,
}

impl CouplingNetwork {
    /// Create new coupling network
    ///
    /// # Arguments
    ///
    /// * `j0` - Coupling strength at zero distance
    /// * `lambda` - Coupling length scale (decay rate)
    /// * `j_min` - Minimum coupling threshold
    pub fn new(j0: f64, lambda: f64, j_min: f64) -> Self {
        Self { j0, lambda, j_min }
    }

    /// Calculate coupling strength between two points
    ///
    /// J_ij = J0 * exp(-d_H / λ)
    #[inline]
    pub fn coupling_strength(&self, distance: f64) -> f64 {
        self.j0 * (-distance / self.lambda).exp()
    }

    /// Calculate cutoff distance for sparse network
    ///
    /// d_cutoff = λ * ln(J0 / J_min)
    pub fn cutoff_distance(&self) -> f64 {
        self.lambda * (self.j0 / self.j_min).ln()
    }

    /// Build coupling network for entire lattice
    ///
    /// Only creates couplings above J_min threshold for sparsity
    pub fn build_couplings(&self, lattice: &mut PBitLattice) -> Result<()> {
        let cutoff = self.cutoff_distance();
        let positions = lattice.positions();
        let n = positions.len();

        // Pre-calculate all couplings to avoid borrowing issues
        let mut couplings: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

        // Calculate all pairwise distances and couplings
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }

                let distance = HyperbolicDistance::distance(&positions[i], &positions[j]);

                // Skip if beyond cutoff
                if distance > cutoff {
                    continue;
                }

                let strength = self.coupling_strength(distance);

                // Skip if below threshold
                if strength < self.j_min {
                    continue;
                }

                couplings[i].push((j, strength));
            }
        }

        // Apply couplings to lattice
        let pbits = lattice.pbits_mut();
        for (i, node_couplings) in couplings.iter().enumerate() {
            for &(j, strength) in node_couplings {
                pbits[i].add_coupling(j, strength);
            }
        }

        Ok(())
    }

    /// Count number of couplings above threshold
    pub fn count_couplings(&self, lattice: &PBitLattice) -> usize {
        lattice
            .pbits()
            .iter()
            .map(|p| p.couplings().len())
            .sum()
    }

    /// Calculate average coupling degree
    pub fn average_degree(&self, lattice: &PBitLattice) -> f64 {
        self.count_couplings(lattice) as f64 / lattice.size() as f64
    }

    /// Get coupling statistics
    pub fn statistics(&self, lattice: &PBitLattice) -> CouplingStatistics {
        let mut total = 0;
        let mut min_degree = usize::MAX;
        let mut max_degree = 0;
        let mut total_strength = 0.0;

        for pbit in lattice.pbits() {
            let degree = pbit.couplings().len();
            total += degree;
            min_degree = min_degree.min(degree);
            max_degree = max_degree.max(degree);

            total_strength += pbit.couplings().values().sum::<f64>();
        }

        CouplingStatistics {
            total_couplings: total,
            average_degree: total as f64 / lattice.size() as f64,
            min_degree,
            max_degree,
            average_strength: total_strength / total as f64,
        }
    }
}

/// Statistics about coupling network
#[derive(Debug, Clone)]
pub struct CouplingStatistics {
    pub total_couplings: usize,
    pub average_degree: f64,
    pub min_degree: usize,
    pub max_degree: usize,
    pub average_strength: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coupling_decay() {
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);

        // At zero distance: J = J0
        assert_eq!(network.coupling_strength(0.0), 1.0);

        // At distance λ: J = J0 * e^-1 ≈ 0.368
        assert!((network.coupling_strength(1.0) - 0.368).abs() < 0.001);

        // At large distance: J ≈ 0
        assert!(network.coupling_strength(10.0) < 1e-4);
    }

    #[test]
    fn test_cutoff_distance() {
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);

        // d_cutoff = ln(10^6) ≈ 13.8
        let cutoff = network.cutoff_distance();
        assert!((cutoff - 13.8).abs() < 0.1);
    }

    #[test]
    fn test_build_couplings() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);

        network.build_couplings(&mut lattice).unwrap();

        // Should have some couplings
        let stats = network.statistics(&lattice);
        assert!(stats.total_couplings > 0);
        assert!(stats.average_degree > 0.0);
    }

    #[test]
    fn test_sparse_network() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();

        // High threshold = sparse network
        let network = CouplingNetwork::new(1.0, 0.5, 0.1);
        network.build_couplings(&mut lattice).unwrap();

        let sparse_count = network.count_couplings(&lattice);

        // Low threshold = dense network
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let dense_count = network.count_couplings(&lattice);

        assert!(sparse_count < dense_count);
    }
}
