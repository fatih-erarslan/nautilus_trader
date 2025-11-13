//! Sparse coupling matrix using Compressed Sparse Row (CSR) format
//!
//! Efficient storage and computation for large-scale pBit networks with exponential decay coupling.
//! Memory: O(E) where E = expected number of edges (typically O(N log N) for hyperbolic graphs)
//!
//! Research:
//! - Saad (2003) "Iterative Methods for Sparse Linear Systems"
//! - Krioukov et al. (2010) "Hyperbolic geometry of complex networks" PRE 82:036106

use crate::{PBitLattice, Result, PBitError};
use hyperphysics_geometry::PoincarePoint;
use sprs::{CsMat, TriMat};
use rayon::prelude::*;

/// Sparse coupling matrix in CSR format
///
/// Implements exponential decay: J_ij = J₀ · exp(-d_H(i,j) / λ)
/// with cutoff at d_cutoff = λ · ln(J₀ / J_min)
#[derive(Debug, Clone)]
pub struct SparseCouplingMatrix {
    /// CSR sparse matrix storing coupling strengths
    csr: CsMat<f64>,

    /// Node positions in hyperbolic space
    positions: Vec<PoincarePoint>,

    /// Coupling strength at zero distance
    j0: f64,

    /// Coupling length scale (decay rate)
    lambda: f64,

    /// Cutoff distance (derived from J_min)
    cutoff: f64,
}

impl SparseCouplingMatrix {
    /// Build sparse coupling matrix from lattice
    ///
    /// # Arguments
    ///
    /// * `lattice` - pBit lattice with node positions
    /// * `j0` - Coupling strength at zero distance
    /// * `lambda` - Coupling length scale
    /// * `j_min` - Minimum coupling threshold (controls sparsity)
    ///
    /// # Returns
    ///
    /// Sparse coupling matrix in CSR format
    ///
    /// # Performance
    ///
    /// - Time: O(N² log N) for distance calculations (can be optimized with spatial indexing)
    /// - Space: O(E) where E = number of edges above threshold
    /// - Expected E = O(N log N) for hyperbolic networks
    pub fn from_lattice(
        lattice: &PBitLattice,
        j0: f64,
        lambda: f64,
        j_min: f64,
    ) -> Result<Self> {
        let n = lattice.size();
        let cutoff = lambda * (j0 / j_min).ln();
        let positions = lattice.positions();

        // Use triplet format for construction (row, col, value)
        let mut triplet = TriMat::new((n, n));

        // Calculate all pairwise couplings in parallel
        // Note: For very large N, use spatial indexing (k-d tree or ball tree)
        let edges: Vec<(usize, usize, f64)> = (0..n)
            .into_par_iter()
            .flat_map(|i| {
                let mut local_edges = Vec::new();
                for j in 0..n {
                    if i == j {
                        continue;
                    }

                    let d_hyp = positions[i].hyperbolic_distance(&positions[j]);

                    // Skip if beyond cutoff
                    if d_hyp > cutoff {
                        continue;
                    }

                    let coupling = j0 * (-d_hyp / lambda).exp();

                    // Skip if below threshold
                    if coupling < j_min {
                        continue;
                    }

                    local_edges.push((i, j, coupling));
                }
                local_edges
            })
            .collect();

        // Build triplet matrix
        for (i, j, coupling) in edges {
            triplet.add_triplet(i, j, coupling);
        }

        // Convert to CSR format (efficient for matrix-vector multiplication)
        let csr = triplet.to_csr();

        Ok(Self {
            csr,
            positions,
            j0,
            lambda,
            cutoff,
        })
    }

    /// Get CSR matrix
    pub fn csr(&self) -> &CsMat<f64> {
        &self.csr
    }

    /// Get number of nodes
    pub fn size(&self) -> usize {
        self.positions.len()
    }

    /// Get number of non-zero couplings
    pub fn nnz(&self) -> usize {
        self.csr.nnz()
    }

    /// Get sparsity (fraction of non-zero elements)
    pub fn sparsity(&self) -> f64 {
        let n = self.size();
        self.nnz() as f64 / (n * n) as f64
    }

    /// Get average degree (average number of neighbors per node)
    pub fn average_degree(&self) -> f64 {
        self.nnz() as f64 / self.size() as f64
    }

    /// Calculate effective field for all nodes given current states
    ///
    /// h_i = Σ_j J_ij s_j
    ///
    /// Uses efficient CSR matrix-vector multiplication
    ///
    /// # Performance
    ///
    /// Time: O(nnz) where nnz = number of non-zero couplings
    pub fn effective_fields(&self, states: &[bool]) -> Result<Vec<f64>> {
        if states.len() != self.size() {
            return Err(PBitError::LatticeError {
                message: format!(
                    "State array length {} != matrix size {}",
                    states.len(),
                    self.size()
                ),
            });
        }

        // Convert states to spins: false -> -1.0, true -> +1.0
        let spins: Vec<f64> = states.iter().map(|&s| if s { 1.0 } else { -1.0 }).collect();

        // Perform sparse matrix-vector multiplication: h = J * s
        // Manual multiplication for compatibility with sprs 0.11
        let mut fields = vec![0.0; self.size()];
        for (row_idx, row) in self.csr.outer_iterator().enumerate() {
            for (col_idx, &coupling) in row.iter() {
                fields[row_idx] += coupling * spins[col_idx];
            }
        }

        Ok(fields)
    }

    /// Calculate total energy of configuration
    ///
    /// E = -1/2 Σ_ij J_ij s_i s_j
    ///
    /// # Performance
    ///
    /// Time: O(nnz)
    pub fn energy(&self, states: &[bool]) -> Result<f64> {
        if states.len() != self.size() {
            return Err(PBitError::LatticeError {
                message: format!(
                    "State array length {} != matrix size {}",
                    states.len(),
                    self.size()
                ),
            });
        }

        let spins: Vec<f64> = states.iter().map(|&s| if s { 1.0 } else { -1.0 }).collect();

        let mut energy = 0.0;

        // Iterate over non-zero elements in CSR format
        for (i, row) in self.csr.outer_iterator().enumerate() {
            let s_i = spins[i];
            for (j, &j_ij) in row.iter() {
                let s_j = spins[j];
                energy += j_ij * s_i * s_j;
            }
        }

        // Factor of -1/2 (but we counted each edge once, so just -1/2 * 2 = -1)
        Ok(-0.5 * energy)
    }

    /// Get coupling strength between two nodes
    pub fn get_coupling(&self, i: usize, j: usize) -> f64 {
        self.csr.get(i, j).copied().unwrap_or(0.0)
    }

    /// Get neighbors of node i (indices and coupling strengths)
    pub fn neighbors(&self, i: usize) -> Vec<(usize, f64)> {
        if i >= self.size() {
            return Vec::new();
        }

        self.csr
            .outer_view(i)
            .unwrap()
            .iter()
            .map(|(j, &coupling)| (j, coupling))
            .collect()
    }

    /// Get degree of node i (number of neighbors)
    pub fn degree(&self, i: usize) -> usize {
        if i >= self.size() {
            return 0;
        }
        self.csr.outer_view(i).unwrap().nnz()
    }

    /// Calculate coupling statistics
    pub fn statistics(&self) -> CouplingStatistics {
        let n = self.size();
        let nnz = self.nnz();

        let mut degrees = vec![0; n];
        let mut total_strength = 0.0;
        let mut min_coupling = f64::MAX;
        let mut max_coupling: f64 = 0.0;

        for (i, row) in self.csr.outer_iterator().enumerate() {
            degrees[i] = row.nnz();
            for (_, &coupling) in row.iter() {
                total_strength += coupling;
                min_coupling = min_coupling.min(coupling);
                max_coupling = max_coupling.max(coupling);
            }
        }

        let min_degree = *degrees.iter().min().unwrap_or(&0);
        let max_degree = *degrees.iter().max().unwrap_or(&0);

        CouplingStatistics {
            total_couplings: nnz,
            average_degree: nnz as f64 / n as f64,
            min_degree,
            max_degree,
            average_strength: if nnz > 0 { total_strength / nnz as f64 } else { 0.0 },
            min_coupling: if nnz > 0 { min_coupling } else { 0.0 },
            max_coupling: if nnz > 0 { max_coupling } else { 0.0 },
            sparsity: nnz as f64 / (n * n) as f64,
        }
    }

    /// Export to dense matrix (for visualization or small systems only)
    ///
    /// Warning: O(N²) memory! Only use for small N < 1000
    pub fn to_dense(&self) -> Vec<Vec<f64>> {
        let n = self.size();
        let mut dense = vec![vec![0.0; n]; n];

        for (i, row) in self.csr.outer_iterator().enumerate() {
            for (j, &coupling) in row.iter() {
                dense[i][j] = coupling;
            }
        }

        dense
    }

    /// Get parameters
    pub fn j0(&self) -> f64 {
        self.j0
    }

    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }
}

/// Statistics about sparse coupling matrix
#[derive(Debug, Clone)]
pub struct CouplingStatistics {
    pub total_couplings: usize,
    pub average_degree: f64,
    pub min_degree: usize,
    pub max_degree: usize,
    pub average_strength: f64,
    pub min_coupling: f64,
    pub max_coupling: f64,
    pub sparsity: f64,
}

impl std::fmt::Display for CouplingStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Coupling Statistics:\n  \
             Total couplings: {}\n  \
             Average degree: {:.2}\n  \
             Degree range: [{}, {}]\n  \
             Coupling strength range: [{:.6}, {:.6}]\n  \
             Average strength: {:.6}\n  \
             Sparsity: {:.6}",
            self.total_couplings,
            self.average_degree,
            self.min_degree,
            self.max_degree,
            self.min_coupling,
            self.max_coupling,
            self.average_strength,
            self.sparsity
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_matrix_creation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let matrix = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 1.0, 1e-6).unwrap();

        assert_eq!(matrix.size(), lattice.size());
        assert!(matrix.nnz() > 0);
        assert!(matrix.sparsity() < 1.0);
    }

    #[test]
    fn test_effective_fields() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let matrix = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 1.0, 1e-6).unwrap();

        let states = vec![true; lattice.size()];
        let fields = matrix.effective_fields(&states).unwrap();

        assert_eq!(fields.len(), lattice.size());
        // All spins aligned -> all fields should be positive
        assert!(fields.iter().all(|&h| h >= 0.0));
    }

    #[test]
    fn test_energy_calculation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let matrix = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 1.0, 1e-6).unwrap();

        // All spins up: ferromagnetic ground state
        let all_up = vec![true; lattice.size()];
        let e_up = matrix.energy(&all_up).unwrap();

        // All spins down: also ferromagnetic ground state
        let all_down = vec![false; lattice.size()];
        let e_down = matrix.energy(&all_down).unwrap();

        // Energies should be equal due to symmetry
        assert!((e_up - e_down).abs() < 1e-10);

        // Random configuration: higher energy
        let mixed: Vec<bool> = vec![true, false, true, false].into_iter().cycle().take(lattice.size()).collect();
        let e_mixed = matrix.energy(&mixed).unwrap();

        // Mixed state should have higher (less negative) energy
        assert!(e_mixed > e_up);
    }

    #[test]
    fn test_sparsity_control() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();

        // High threshold -> high sparsity
        let sparse = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 0.5, 0.1).unwrap();

        // Low threshold -> low sparsity
        let dense = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 1.0, 1e-6).unwrap();

        assert!(sparse.nnz() < dense.nnz());
        assert!(sparse.sparsity() < dense.sparsity());
    }

    #[test]
    fn test_neighbors() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let matrix = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 1.0, 1e-6).unwrap();

        let neighbors = matrix.neighbors(0);
        assert!(neighbors.len() > 0);

        // All coupling strengths should be positive and finite
        for (_, coupling) in neighbors {
            assert!(coupling > 0.0 && coupling.is_finite());
        }
    }

    #[test]
    fn test_statistics() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let matrix = SparseCouplingMatrix::from_lattice(&lattice, 1.0, 1.0, 1e-6).unwrap();

        let stats = matrix.statistics();

        assert!(stats.total_couplings > 0);
        assert!(stats.average_degree > 0.0);
        assert!(stats.min_degree <= stats.max_degree);
        assert!(stats.min_coupling <= stats.max_coupling);
        assert!(stats.average_strength > 0.0);
        assert!(stats.sparsity > 0.0 && stats.sparsity < 1.0);
    }
}
