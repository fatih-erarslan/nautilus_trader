//! Neuronal field dynamics on hyperbolic substrate
//!
//! Implements neural activity patterns that source the syntergic field.
//!
//! Research:
//! - Freeman (2000) "Neurodynamics: An Exploration in Mesoscopic Brain Dynamics" Springer
//! - Grinberg-Zylberbaum (1995) "Syntergic Theory" INPEC

use crate::{Result, SyntergicError};
use hyperphysics_geometry::PoincarePoint;
use hyperphysics_pbit::{PBitLattice, SparseCouplingMatrix};

/// Neuronal field activity pattern
///
/// Represents the collective neural state that sources the syntergic field.
/// Activity is modeled as pBit states representing binary firing patterns.
#[derive(Debug, Clone)]
pub struct NeuronalField {
    /// Lattice of neuronal pBits
    lattice: PBitLattice,

    /// Coupling matrix for neuronal interactions
    coupling: Option<SparseCouplingMatrix>,

    /// Current activity pattern (firing rates)
    activity: Vec<f64>,
}

impl NeuronalField {
    /// Create new neuronal field
    ///
    /// # Arguments
    ///
    /// * `lattice` - pBit lattice representing neural network
    pub fn new(lattice: PBitLattice) -> Self {
        let n = lattice.size();
        Self {
            lattice,
            coupling: None,
            activity: vec![0.0; n],
        }
    }

    /// Initialize with coupling matrix
    ///
    /// # Arguments
    ///
    /// * `j0` - Coupling strength
    /// * `lambda` - Coupling length scale
    /// * `j_min` - Minimum coupling threshold
    pub fn with_coupling(mut self, j0: f64, lambda: f64, j_min: f64) -> Result<Self> {
        let coupling = SparseCouplingMatrix::from_lattice(&self.lattice, j0, lambda, j_min)?;
        self.coupling = Some(coupling);
        Ok(self)
    }

    /// Get lattice size
    pub fn size(&self) -> usize {
        self.lattice.size()
    }

    /// Get lattice reference
    pub fn lattice(&self) -> &PBitLattice {
        &self.lattice
    }

    /// Get current activity pattern
    pub fn activity(&self) -> &[f64] {
        &self.activity
    }

    /// Get positions of all neurons
    pub fn positions(&self) -> Vec<PoincarePoint> {
        self.lattice.positions()
    }

    /// Update activity based on current pBit states
    ///
    /// Activity is defined as the probability of firing (prob_one)
    pub fn update_activity(&mut self) {
        self.activity = self.lattice.probabilities();
    }

    /// Compute activity at specific position using spatial interpolation
    ///
    /// For positions between neurons, uses weighted average based on distance
    pub fn activity_at(&self, point: &PoincarePoint) -> f64 {
        let positions = self.lattice.positions();

        // Find nearest neurons and compute weighted average
        let mut total_weight = 0.0;
        let mut weighted_activity = 0.0;

        for (i, pos) in positions.iter().enumerate() {
            let distance = point.hyperbolic_distance(pos);

            // Gaussian weighting: w = exp(-d²/2σ²)
            let sigma = 0.5; // Spatial scale parameter
            let weight = (-distance * distance / (2.0 * sigma * sigma)).exp();

            weighted_activity += weight * self.activity[i];
            total_weight += weight;
        }

        if total_weight > 1e-10 {
            weighted_activity / total_weight
        } else {
            0.0
        }
    }

    /// Compute spatial gradient of activity at position
    ///
    /// Returns approximate gradient using finite differences
    pub fn activity_gradient(&self, point: &PoincarePoint) -> Result<[f64; 3]> {
        use nalgebra::Vector3;

        let epsilon = 1e-6;
        let f_0 = self.activity_at(point);

        let coords = point.coords();

        // Compute partial derivatives using central differences
        let mut gradient = [0.0; 3];

        for i in 0..3 {
            let mut plus_coords = coords;
            let mut minus_coords = coords;

            plus_coords[i] += epsilon;
            minus_coords[i] -= epsilon;

            // Ensure points remain in disk
            if plus_coords.norm() >= 1.0 || minus_coords.norm() >= 1.0 {
                continue;
            }

            let p_plus = PoincarePoint::new(plus_coords)
                .map_err(|e| SyntergicError::Geometry(e))?;
            let p_minus = PoincarePoint::new(minus_coords)
                .map_err(|e| SyntergicError::Geometry(e))?;

            let f_plus = self.activity_at(&p_plus);
            let f_minus = self.activity_at(&p_minus);

            gradient[i] = (f_plus - f_minus) / (2.0 * epsilon);
        }

        Ok(gradient)
    }

    /// Compute total activity (sum of all firing rates)
    pub fn total_activity(&self) -> f64 {
        self.activity.iter().sum()
    }

    /// Compute activity entropy (Shannon entropy of firing pattern)
    ///
    /// H = -Σ_i [p_i log(p_i) + (1-p_i) log(1-p_i)]
    pub fn activity_entropy(&self) -> f64 {
        self.activity
            .iter()
            .map(|&p| {
                let p_clipped = p.clamp(1e-10, 1.0 - 1e-10);
                let q = 1.0 - p_clipped;
                -p_clipped * p_clipped.ln() - q * q.ln()
            })
            .sum()
    }

    /// Compute activity variance (measure of heterogeneity)
    pub fn activity_variance(&self) -> f64 {
        let mean = self.total_activity() / self.size() as f64;
        let variance: f64 = self
            .activity
            .iter()
            .map(|&a| (a - mean).powi(2))
            .sum();

        variance / self.size() as f64
    }

    /// Get activity statistics
    pub fn statistics(&self) -> NeuronalStatistics {
        let total = self.total_activity();
        let mean = total / self.size() as f64;

        let min = self.activity.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = self.activity.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        let variance = self.activity_variance();
        let entropy = self.activity_entropy();

        NeuronalStatistics {
            total_activity: total,
            mean_activity: mean,
            min_activity: min,
            max_activity: max,
            variance,
            entropy,
            size: self.size(),
        }
    }
}

/// Statistics about neuronal field activity
#[derive(Debug, Clone)]
pub struct NeuronalStatistics {
    pub total_activity: f64,
    pub mean_activity: f64,
    pub min_activity: f64,
    pub max_activity: f64,
    pub variance: f64,
    pub entropy: f64,
    pub size: usize,
}

impl std::fmt::Display for NeuronalStatistics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Neuronal Field Statistics:\n  \
             Size: {} neurons\n  \
             Total activity: {:.4}\n  \
             Mean activity: {:.4}\n  \
             Activity range: [{:.4}, {:.4}]\n  \
             Variance: {:.6}\n  \
             Entropy: {:.4}",
            self.size,
            self.total_activity,
            self.mean_activity,
            self.min_activity,
            self.max_activity,
            self.variance,
            self.entropy
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuronal_field_creation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let field = NeuronalField::new(lattice);

        assert_eq!(field.size(), field.lattice().size());
        assert_eq!(field.activity().len(), field.size());
    }

    #[test]
    fn test_activity_update() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = NeuronalField::new(lattice);

        // Initial activity should be based on lattice probabilities
        field.update_activity();

        let stats = field.statistics();
        assert!(stats.mean_activity >= 0.0 && stats.mean_activity <= 1.0);
    }

    #[test]
    fn test_activity_entropy() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = NeuronalField::new(lattice);

        field.update_activity();
        let entropy = field.activity_entropy();

        assert!(entropy >= 0.0, "Entropy must be non-negative");
        assert!(entropy.is_finite(), "Entropy must be finite");
    }

    #[test]
    fn test_activity_interpolation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = NeuronalField::new(lattice);

        field.update_activity();

        let origin = PoincarePoint::origin();
        let activity = field.activity_at(&origin);

        assert!(activity >= 0.0 && activity <= 1.0);
    }
}
