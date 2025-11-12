//! Causal density estimation

// Causal density estimation functions
use hyperphysics_pbit::PBitLattice;

/// Causal density estimator
///
/// Estimates the density of causal connections in the system
pub struct CausalDensityEstimator;

impl CausalDensityEstimator {
    /// Calculate causal density
    ///
    /// ρ_causal = (number of effective couplings) / (total possible couplings)
    pub fn causal_density(lattice: &PBitLattice) -> f64 {
        let n = lattice.size();
        let max_couplings = n * (n - 1); // Directed graph

        if max_couplings == 0 {
            return 0.0;
        }

        let actual_couplings: usize = lattice
            .pbits()
            .iter()
            .map(|p| p.couplings().len())
            .sum();

        actual_couplings as f64 / max_couplings as f64
    }

    /// Calculate average coupling strength
    pub fn average_coupling_strength(lattice: &PBitLattice) -> f64 {
        let mut total = 0.0;
        let mut count = 0;

        for pbit in lattice.pbits() {
            for &strength in pbit.couplings().values() {
                total += strength.abs();
                count += 1;
            }
        }

        if count > 0 {
            total / count as f64
        } else {
            0.0
        }
    }

    /// Calculate causal depth (longest causal chain)
    ///
    /// Simplified: use coupling network structure
    pub fn causal_depth(lattice: &PBitLattice) -> usize {
        let n = lattice.size();

        // Simplified BFS to find longest path
        let mut max_depth = 0;

        for start in 0..n {
            let depth = Self::bfs_depth(lattice, start);
            max_depth = max_depth.max(depth);
        }

        max_depth
    }

    /// BFS to find maximum depth from start node
    fn bfs_depth(lattice: &PBitLattice, start: usize) -> usize {
        use std::collections::{HashSet, VecDeque};

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back((start, 0));
        visited.insert(start);

        let mut max_depth = 0;

        while let Some((node, depth)) = queue.pop_front() {
            max_depth = max_depth.max(depth);

            if let Some(pbit) = lattice.pbit(node) {
                for &neighbor in pbit.couplings().keys() {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        max_depth
    }

    /// Calculate clustering coefficient (local density)
    pub fn clustering_coefficient(lattice: &PBitLattice) -> f64 {
        let n = lattice.size();
        let mut total_coeff = 0.0;

        for i in 0..n {
            if let Some(pbit) = lattice.pbit(i) {
                let neighbors: Vec<usize> = pbit.couplings().keys().copied().collect();
                let k = neighbors.len();

                if k < 2 {
                    continue;
                }

                // Count triangles
                let mut triangles = 0;

                for &j in &neighbors {
                    if let Some(neighbor_pbit) = lattice.pbit(j) {
                        for &k_node in &neighbors {
                            if j != k_node && neighbor_pbit.couplings().contains_key(&k_node) {
                                triangles += 1;
                            }
                        }
                    }
                }

                let possible = k * (k - 1);
                total_coeff += triangles as f64 / possible as f64;
            }
        }

        total_coeff / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyperphysics_pbit::CouplingNetwork;

    #[test]
    fn test_causal_density_no_couplings() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let density = CausalDensityEstimator::causal_density(&lattice);

        assert_eq!(density, 0.0);
    }

    #[test]
    fn test_causal_density_with_couplings() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let density = CausalDensityEstimator::causal_density(&lattice);

        assert!(density > 0.0);
        assert!(density <= 1.0);
    }

    #[test]
    fn test_average_coupling_strength() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let avg_strength = CausalDensityEstimator::average_coupling_strength(&lattice);

        assert!(avg_strength > 0.0);
        assert!(avg_strength <= 1.0);
    }

    #[test]
    fn test_causal_depth() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let depth = CausalDensityEstimator::causal_depth(&lattice);

        assert!(depth > 0);
        assert!(depth < lattice.size());
    }

    #[test]
    fn test_clustering_coefficient() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();
        let network = CouplingNetwork::new(1.0, 1.0, 1e-6);
        network.build_couplings(&mut lattice).unwrap();

        let cc = CausalDensityEstimator::clustering_coefficient(&lattice);

        assert!(cc >= 0.0);
        assert!(cc <= 1.0);
    }
}
