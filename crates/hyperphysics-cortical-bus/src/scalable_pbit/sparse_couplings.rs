//! # Sparse Coupling Matrix
//!
//! Compressed Sparse Row (CSR) format for efficient pBit coupling storage.
//! Memory: O(E) where E = number of edges, not O(N²).

/// Single coupling entry
#[derive(Debug, Clone, Copy)]
pub struct CouplingEntry {
    /// Target pBit index
    pub target: u32,
    /// Coupling strength J_ij
    pub strength: f32,
}

/// Sparse coupling matrix in CSR format
///
/// For a pBit at index i, its neighbors are stored in:
/// `values[row_offsets[i]..row_offsets[i+1]]`
pub struct SparseCouplings {
    /// Row offsets (size: num_pbits + 1)
    row_offsets: Vec<u32>,
    /// Column indices and values (size: nnz)
    entries: Vec<CouplingEntry>,
    /// Number of pBits
    num_pbits: usize,
    /// Whether the matrix is finalized
    finalized: bool,
    /// Temporary storage during construction
    temp_edges: Vec<Vec<CouplingEntry>>,
}

impl SparseCouplings {
    /// Create empty coupling matrix
    pub fn new(num_pbits: usize) -> Self {
        Self {
            row_offsets: vec![0; num_pbits + 1],
            entries: Vec::new(),
            num_pbits,
            finalized: false,
            temp_edges: vec![Vec::new(); num_pbits],
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(num_pbits: usize, edge_capacity: usize) -> Self {
        let mut csr = Self::new(num_pbits);
        csr.entries.reserve(edge_capacity);
        csr
    }

    /// Add a coupling (during construction phase)
    #[inline]
    pub fn add(&mut self, from: usize, to: usize, strength: f32) {
        debug_assert!(!self.finalized, "Cannot add after finalization");
        debug_assert!(from < self.num_pbits);
        debug_assert!(to < self.num_pbits);
        
        self.temp_edges[from].push(CouplingEntry {
            target: to as u32,
            strength,
        });
    }

    /// Finalize the matrix (convert to CSR format)
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }

        // Count edges per row
        let mut offset = 0u32;
        for i in 0..self.num_pbits {
            self.row_offsets[i] = offset;
            offset += self.temp_edges[i].len() as u32;
        }
        self.row_offsets[self.num_pbits] = offset;

        // Copy edges to CSR format
        self.entries.clear();
        self.entries.reserve(offset as usize);
        
        for edges in &self.temp_edges {
            self.entries.extend_from_slice(edges);
        }

        // Clear temporary storage
        self.temp_edges.clear();
        self.temp_edges.shrink_to_fit();
        
        self.finalized = true;
    }

    /// Get neighbors of a pBit (iterator over (index, strength))
    #[inline]
    pub fn neighbors(&self, idx: usize) -> NeighborIter<'_> {
        debug_assert!(self.finalized, "Must finalize before querying");
        let start = self.row_offsets[idx] as usize;
        let end = self.row_offsets[idx + 1] as usize;
        
        NeighborIter {
            entries: &self.entries[start..end],
            pos: 0,
        }
    }

    /// Get neighbor slice for a pBit
    #[inline]
    pub fn neighbor_slice(&self, idx: usize) -> &[CouplingEntry] {
        debug_assert!(self.finalized);
        let start = self.row_offsets[idx] as usize;
        let end = self.row_offsets[idx + 1] as usize;
        &self.entries[start..end]
    }

    /// Degree of a pBit (number of neighbors)
    #[inline]
    pub fn degree(&self, idx: usize) -> usize {
        debug_assert!(self.finalized);
        (self.row_offsets[idx + 1] - self.row_offsets[idx]) as usize
    }

    /// Total number of non-zero entries
    #[inline]
    pub fn nnz(&self) -> usize {
        if self.finalized {
            self.entries.len()
        } else {
            self.temp_edges.iter().map(|e| e.len()).sum()
        }
    }

    /// Number of pBits
    #[inline]
    pub fn num_pbits(&self) -> usize {
        self.num_pbits
    }

    /// Calculate effective field for a single pBit
    ///
    /// h_i = Σ_j J_ij s_j + bias_i
    #[inline]
    pub fn effective_field(&self, idx: usize, states: &super::PackedPBitArray, bias: f32) -> f32 {
        debug_assert!(self.finalized);
        
        let mut field = bias;
        for entry in self.neighbor_slice(idx) {
            let s_j = states.spin(entry.target as usize);
            field += entry.strength * s_j;
        }
        field
    }

    /// Calculate effective fields for all pBits (batch operation)
    pub fn effective_fields_batch(
        &self,
        states: &super::PackedPBitArray,
        biases: &[f32],
        output: &mut [f32],
    ) {
        debug_assert!(self.finalized);
        debug_assert_eq!(biases.len(), self.num_pbits);
        debug_assert_eq!(output.len(), self.num_pbits);

        for i in 0..self.num_pbits {
            output[i] = self.effective_field(i, states, biases[i]);
        }
    }

    /// Calculate energy change for flipping pBit i
    ///
    /// ΔE = 2 s_i (h_i + Σ_j J_ij s_j)
    #[inline]
    pub fn delta_energy(&self, idx: usize, states: &super::PackedPBitArray, bias: f32) -> f32 {
        let s_i = states.spin(idx);
        let h_i = self.effective_field(idx, states, bias);
        2.0 * s_i * h_i
    }
}

/// Iterator over neighbors
pub struct NeighborIter<'a> {
    entries: &'a [CouplingEntry],
    pos: usize,
}

impl<'a> Iterator for NeighborIter<'a> {
    type Item = (usize, f32);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.pos < self.entries.len() {
            let entry = &self.entries[self.pos];
            self.pos += 1;
            Some((entry.target as usize, entry.strength))
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.entries.len() - self.pos;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for NeighborIter<'a> {}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::PackedPBitArray;

    #[test]
    fn test_sparse_construction() {
        let mut csr = SparseCouplings::new(3);
        
        csr.add(0, 1, 1.0);
        csr.add(0, 2, 0.5);
        csr.add(1, 0, 1.0);
        csr.add(1, 2, -0.5);
        csr.add(2, 0, 0.5);
        csr.add(2, 1, -0.5);
        
        csr.finalize();
        
        assert_eq!(csr.nnz(), 6);
        assert_eq!(csr.degree(0), 2);
        assert_eq!(csr.degree(1), 2);
        assert_eq!(csr.degree(2), 2);
    }

    #[test]
    fn test_effective_field() {
        let mut csr = SparseCouplings::new(3);
        
        // All ferromagnetic couplings
        csr.add(0, 1, 1.0);
        csr.add(0, 2, 1.0);
        csr.add(1, 0, 1.0);
        csr.add(2, 0, 1.0);
        
        csr.finalize();
        
        let states = PackedPBitArray::new(3);
        states.set(1, true); // s_1 = +1
        states.set(2, true); // s_2 = +1
        
        // h_0 = J_01 * s_1 + J_02 * s_2 = 1.0 * 1.0 + 1.0 * 1.0 = 2.0
        let h_0 = csr.effective_field(0, &states, 0.0);
        assert!((h_0 - 2.0).abs() < 0.001);
    }

    #[test]
    fn test_delta_energy() {
        let mut csr = SparseCouplings::new(2);
        csr.add(0, 1, 1.0);
        csr.add(1, 0, 1.0);
        csr.finalize();
        
        let states = PackedPBitArray::new(2);
        // Both down: s_0 = s_1 = -1
        
        // h_0 = J_01 * s_1 = 1.0 * (-1) = -1
        // ΔE = 2 * s_0 * h_0 = 2 * (-1) * (-1) = 2
        // (flipping would increase energy since both spins are aligned down)
        let delta = csr.delta_energy(0, &states, 0.0);
        assert!((delta - 2.0).abs() < 0.001, "Expected 2.0, got {}", delta);
    }
}
