//! Sparse coupling matrix in CSR format
//!
//! No external dependencies. Optimized for sequential neighbor iteration.

/// Single coupling entry
#[derive(Debug, Clone, Copy)]
pub struct CouplingEntry {
    /// Target pBit index
    pub target: u32,
    /// Coupling strength J_ij
    pub strength: f32,
}

/// Sparse coupling matrix in Compressed Sparse Row (CSR) format
///
/// Memory: O(num_pbits + num_edges)
/// Neighbor lookup: O(degree)
pub struct ScalableCouplings {
    /// Row offsets: neighbors of pBit i are in entries[row_ptr[i]..row_ptr[i+1]]
    row_ptr: Vec<u32>,
    /// Column indices and coupling strengths
    entries: Vec<CouplingEntry>,
    /// Number of pBits
    num_pbits: usize,
    /// Construction complete?
    finalized: bool,
    /// Temporary storage during construction
    temp: Vec<Vec<CouplingEntry>>,
}

impl ScalableCouplings {
    /// Create empty coupling matrix
    pub fn new(num_pbits: usize) -> Self {
        Self {
            row_ptr: vec![0; num_pbits + 1],
            entries: Vec::new(),
            num_pbits,
            finalized: false,
            temp: vec![Vec::new(); num_pbits],
        }
    }

    /// Create with pre-allocated capacity
    pub fn with_capacity(num_pbits: usize, edge_capacity: usize) -> Self {
        let mut s = Self::new(num_pbits);
        s.entries.reserve(edge_capacity);
        s
    }

    /// Add a coupling (must call finalize() after all additions)
    #[inline]
    pub fn add(&mut self, from: usize, to: usize, strength: f32) {
        debug_assert!(!self.finalized, "Cannot add after finalize");
        debug_assert!(from < self.num_pbits);
        debug_assert!(to < self.num_pbits);

        self.temp[from].push(CouplingEntry {
            target: to as u32,
            strength,
        });
    }

    /// Add symmetric coupling (both directions)
    #[inline]
    pub fn add_symmetric(&mut self, i: usize, j: usize, strength: f32) {
        self.add(i, j, strength);
        self.add(j, i, strength);
    }

    /// Finalize the matrix (convert to CSR format)
    pub fn finalize(&mut self) {
        if self.finalized {
            return;
        }

        // Calculate row pointers
        let mut offset = 0u32;
        for i in 0..self.num_pbits {
            self.row_ptr[i] = offset;
            offset += self.temp[i].len() as u32;
        }
        self.row_ptr[self.num_pbits] = offset;

        // Copy entries in CSR order
        self.entries.clear();
        self.entries.reserve(offset as usize);

        for row in &self.temp {
            self.entries.extend_from_slice(row);
        }

        // Free temporary storage
        self.temp.clear();
        self.temp.shrink_to_fit();

        self.finalized = true;
    }

    /// Get neighbors of a pBit as a slice
    #[inline]
    pub fn neighbors(&self, idx: usize) -> &[CouplingEntry] {
        debug_assert!(self.finalized);
        let start = self.row_ptr[idx] as usize;
        let end = self.row_ptr[idx + 1] as usize;
        &self.entries[start..end]
    }

    /// Degree of a pBit (number of neighbors)
    #[inline]
    pub fn degree(&self, idx: usize) -> usize {
        debug_assert!(self.finalized);
        (self.row_ptr[idx + 1] - self.row_ptr[idx]) as usize
    }

    /// Total number of edges
    #[inline]
    pub fn num_edges(&self) -> usize {
        if self.finalized {
            self.entries.len()
        } else {
            self.temp.iter().map(|v| v.len()).sum()
        }
    }

    /// Number of pBits
    #[inline]
    pub fn num_pbits(&self) -> usize {
        self.num_pbits
    }

    /// Average degree
    pub fn avg_degree(&self) -> f64 {
        self.num_edges() as f64 / self.num_pbits as f64
    }

    /// Calculate effective field for pBit i
    ///
    /// h_i = bias_i + Σ_j J_ij s_j
    #[inline]
    pub fn effective_field(
        &self,
        idx: usize,
        states: &super::ScalablePBitArray,
        bias: f32,
    ) -> f32 {
        debug_assert!(self.finalized);

        let mut h = bias;
        for entry in self.neighbors(idx) {
            let s_j = states.spin(entry.target as usize);
            h += entry.strength * s_j;
        }
        h
    }

    /// Calculate energy change for flipping pBit i
    ///
    /// ΔE = 2 * s_i * h_i
    #[inline]
    pub fn delta_energy(
        &self,
        idx: usize,
        states: &super::ScalablePBitArray,
        bias: f32,
    ) -> f32 {
        let s_i = states.spin(idx);
        let h_i = self.effective_field(idx, states, bias);
        2.0 * s_i * h_i
    }

    /// Check if finalized
    #[inline]
    pub fn is_finalized(&self) -> bool {
        self.finalized
    }
}

impl Clone for ScalableCouplings {
    fn clone(&self) -> Self {
        Self {
            row_ptr: self.row_ptr.clone(),
            entries: self.entries.clone(),
            num_pbits: self.num_pbits,
            finalized: self.finalized,
            temp: self.temp.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalable::ScalablePBitArray;

    #[test]
    fn test_construction() {
        let mut c = ScalableCouplings::new(4);

        c.add(0, 1, 1.0);
        c.add(0, 2, 0.5);
        c.add(1, 0, 1.0);
        c.add(2, 0, 0.5);

        c.finalize();

        assert_eq!(c.degree(0), 2);
        assert_eq!(c.degree(1), 1);
        assert_eq!(c.degree(2), 1);
        assert_eq!(c.degree(3), 0);
    }

    #[test]
    fn test_effective_field() {
        let mut c = ScalableCouplings::new(3);

        // All ferromagnetic
        c.add_symmetric(0, 1, 1.0);
        c.add_symmetric(0, 2, 1.0);
        c.finalize();

        let states = ScalablePBitArray::new(3);
        states.set(1, true); // s_1 = +1
        states.set(2, true); // s_2 = +1
        // s_0 = -1 (default)

        // h_0 = 0 + J_01*s_1 + J_02*s_2 = 0 + 1.0*1.0 + 1.0*1.0 = 2.0
        let h = c.effective_field(0, &states, 0.0);
        assert!((h - 2.0).abs() < 0.001, "h = {}", h);
    }

    #[test]
    fn test_delta_energy() {
        let mut c = ScalableCouplings::new(2);
        c.add_symmetric(0, 1, 1.0);
        c.finalize();

        let states = ScalablePBitArray::new(2);
        // Both s_0 = s_1 = -1 (aligned down)

        // h_0 = J_01 * s_1 = 1.0 * (-1) = -1
        // ΔE = 2 * s_0 * h_0 = 2 * (-1) * (-1) = 2
        let de = c.delta_energy(0, &states, 0.0);
        assert!((de - 2.0).abs() < 0.001, "ΔE = {}", de);
    }

    #[test]
    fn test_symmetric() {
        let mut c = ScalableCouplings::new(3);
        c.add_symmetric(0, 1, 0.5);
        c.add_symmetric(1, 2, -0.3);
        c.finalize();

        assert_eq!(c.degree(0), 1);
        assert_eq!(c.degree(1), 2);
        assert_eq!(c.degree(2), 1);
        assert_eq!(c.num_edges(), 4);
    }
}
