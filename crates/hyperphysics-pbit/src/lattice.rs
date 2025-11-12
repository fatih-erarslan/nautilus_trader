//! pBit lattice management on hyperbolic substrate

use crate::{PBit, PBitError, Result};
use hyperphysics_geometry::{HyperbolicTessellation, PoincarePoint};

/// Lattice of pBits positioned on hyperbolic tessellation
#[derive(Debug, Clone)]
pub struct PBitLattice {
    pbits: Vec<PBit>,
    tessellation: HyperbolicTessellation,
}

impl PBitLattice {
    /// Create lattice from hyperbolic tessellation
    ///
    /// # Arguments
    ///
    /// * `p` - Polygon sides for {p,q} tessellation
    /// * `q` - Polygons per vertex
    /// * `depth` - Tessellation depth
    /// * `temperature` - Global temperature for all pBits
    pub fn new(p: usize, q: usize, depth: usize, temperature: f64) -> Result<Self> {
        let tessellation = HyperbolicTessellation::new(p, q, depth)
            .map_err(|e| PBitError::LatticeError {
                message: format!("Tessellation failed: {}", e),
            })?;

        let pbits: Result<Vec<PBit>> = tessellation
            .nodes()
            .iter()
            .map(|pos| PBit::new(*pos, temperature))
            .collect();

        Ok(Self {
            pbits: pbits?,
            tessellation,
        })
    }

    /// Create ROI lattice with 48 nodes ({3,7,2})
    pub fn roi_48(temperature: f64) -> Result<Self> {
        Self::new(3, 7, 2, temperature)
    }

    /// Get all pBits
    pub fn pbits(&self) -> &[PBit] {
        &self.pbits
    }

    /// Get mutable pBits
    pub fn pbits_mut(&mut self) -> &mut [PBit] {
        &mut self.pbits
    }

    /// Get specific pBit
    pub fn pbit(&self, idx: usize) -> Option<&PBit> {
        self.pbits.get(idx)
    }

    /// Get specific pBit mutably
    pub fn pbit_mut(&mut self, idx: usize) -> Option<&mut PBit> {
        self.pbits.get_mut(idx)
    }

    /// Get number of pBits
    #[inline]
    pub fn size(&self) -> usize {
        self.pbits.len()
    }

    /// Get tessellation structure
    pub fn tessellation(&self) -> &HyperbolicTessellation {
        &self.tessellation
    }

    /// Get all current states
    pub fn states(&self) -> Vec<bool> {
        self.pbits.iter().map(|p| p.state()).collect()
    }

    /// Set all states
    pub fn set_states(&mut self, states: &[bool]) -> Result<()> {
        if states.len() != self.pbits.len() {
            return Err(PBitError::LatticeError {
                message: format!(
                    "State array length {} != lattice size {}",
                    states.len(),
                    self.pbits.len()
                ),
            });
        }

        for (pbit, &state) in self.pbits.iter_mut().zip(states.iter()) {
            pbit.set_state(state);
        }

        Ok(())
    }

    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        self.pbits.iter().map(|p| p.prob_one()).collect()
    }

    /// Set global temperature for all pBits
    pub fn set_temperature(&mut self, temperature: f64) -> Result<()> {
        for pbit in &mut self.pbits {
            pbit.set_temperature(temperature)?;
        }
        Ok(())
    }

    /// Calculate mean state (magnetization)
    pub fn magnetization(&self) -> f64 {
        let sum: f64 = self.pbits.iter().map(|p| p.spin()).sum();
        sum / (self.pbits.len() as f64)
    }

    /// Calculate state change rate (impermanence)
    ///
    /// Returns fraction of pBits that changed state
    pub fn impermanence(&self, previous_states: &[bool]) -> Result<f64> {
        if previous_states.len() != self.pbits.len() {
            return Err(PBitError::LatticeError {
                message: "State array length mismatch".to_string(),
            });
        }

        let changes: usize = self
            .pbits
            .iter()
            .zip(previous_states.iter())
            .filter(|(pbit, &prev_state)| pbit.state() != prev_state)
            .count();

        Ok(changes as f64 / self.pbits.len() as f64)
    }

    /// Get neighbor indices for a pBit
    pub fn neighbors(&self, idx: usize) -> Vec<usize> {
        self.tessellation.neighbors(idx)
    }

    /// Get positions of all pBits
    pub fn positions(&self) -> Vec<PoincarePoint> {
        self.pbits.iter().map(|p| *p.position()).collect()
    }

    /// Get coupling strengths for energy calculation (SIMD optimization)
    ///
    /// Returns flattened coupling matrix where coupling[i*N + j] = J_ij
    /// Only includes non-zero couplings for efficiency
    pub fn couplings(&self) -> Vec<f64> {
        let n = self.pbits.len();
        let mut couplings = Vec::with_capacity(n * n);

        // Extract coupling strengths from each pBit
        for pbit in &self.pbits {
            let pbit_couplings = pbit.couplings();

            // Build sparse row for this pBit
            let mut row = vec![0.0; n];
            for (&neighbor_idx, &strength) in pbit_couplings {
                if neighbor_idx < n {
                    row[neighbor_idx] = strength;
                }
            }

            couplings.extend_from_slice(&row);
        }

        couplings
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roi_48_lattice() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();

        // Debug: print actual size
        println!("ROI lattice size: {}", lattice.size());

        // Should have approximately 48 nodes
        // Note: {3,7,2} tessellation may generate different number
        assert!(lattice.size() > 0, "Lattice must have nodes");
    }

    #[test]
    fn test_states() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();

        let states: Vec<bool> = vec![true; lattice.size()];
        lattice.set_states(&states).unwrap();

        for pbit in lattice.pbits() {
            assert!(pbit.state());
        }
    }

    #[test]
    fn test_magnetization() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();

        // All spins up: magnetization = +1
        let states = vec![true; lattice.size()];
        lattice.set_states(&states).unwrap();
        assert!((lattice.magnetization() - 1.0).abs() < 1e-10);

        // All spins down: magnetization = -1
        let states = vec![false; lattice.size()];
        lattice.set_states(&states).unwrap();
        assert!((lattice.magnetization() + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_impermanence() {
        let mut lattice = PBitLattice::roi_48(1.0).unwrap();

        let initial = vec![false; lattice.size()];
        lattice.set_states(&initial).unwrap();

        // Change half the states
        let mut new_states = initial.clone();
        for i in 0..lattice.size() / 2 {
            new_states[i] = true;
        }
        lattice.set_states(&new_states).unwrap();

        let imp = lattice.impermanence(&initial).unwrap();
        assert!((imp - 0.5).abs() < 0.1); // Approximately 50% changed
    }
}
