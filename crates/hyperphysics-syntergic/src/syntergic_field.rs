//! Complete syntergic field implementation
//!
//! Combines Green's function, neuronal field, and non-local correlations.

use crate::{
    HyperbolicGreenFunction, NeuronalField, Result, SYNTERGIC_SPEED,
};
use hyperphysics_geometry::PoincarePoint;
use hyperphysics_pbit::PBitLattice;

/// Complete syntergic field system
///
/// Implements Grinberg-Zylberbaum's theory of non-local consciousness correlations
/// on hyperbolic substrate.
#[derive(Debug, Clone)]
pub struct SyntergicField {
    /// Neuronal field (source of syntergic field)
    neuronal_field: NeuronalField,

    /// Green's function for field propagation
    green_function: HyperbolicGreenFunction,

    /// Current syntergic field strength at each neuron
    field_values: Vec<f64>,

    /// Field propagation speed
    #[allow(dead_code)]
    speed: f64,

    /// Time elapsed
    time: f64,
}

impl SyntergicField {
    /// Create new syntergic field system
    ///
    /// # Arguments
    ///
    /// * `lattice` - Neural network lattice
    /// * `kappa` - Green's function parameter (typically 1.0)
    pub fn new(lattice: PBitLattice, kappa: f64) -> Self {
        let n = lattice.size();
        let neuronal_field = NeuronalField::new(lattice);
        let green_function = HyperbolicGreenFunction::new(kappa);

        Self {
            neuronal_field,
            green_function,
            field_values: vec![0.0; n],
            speed: SYNTERGIC_SPEED,
            time: 0.0,
        }
    }

    /// Initialize with coupling
    pub fn with_coupling(mut self, j0: f64, lambda: f64, j_min: f64) -> Result<Self> {
        self.neuronal_field = self.neuronal_field.with_coupling(j0, lambda, j_min)?;
        Ok(self)
    }

    /// Get size
    pub fn size(&self) -> usize {
        self.neuronal_field.size()
    }

    /// Get current time
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Get neuronal field reference
    pub fn neuronal_field(&self) -> &NeuronalField {
        &self.neuronal_field
    }

    /// Get field values at each neuron
    pub fn field_values(&self) -> &[f64] {
        &self.field_values
    }

    /// Update neuronal activity and recompute field
    ///
    /// Steps:
    /// 1. Update neuronal firing patterns
    /// 2. Compute syntergic field using Green's function
    /// 3. Calculate non-local correlations
    pub fn update(&mut self, dt: f64) -> Result<()> {
        // Update neuronal activity
        self.neuronal_field.update_activity();

        // Compute syntergic field at each neuron position
        let positions = self.neuronal_field.positions();
        let activity = self.neuronal_field.activity();

        // Ψ(x_i) = Σ_j G(x_i, x_j) ρ_j
        // where ρ_j is the neuronal activity (source strength)
        for i in 0..self.size() {
            let field = self.green_function.compute_field(
                &positions[i],
                &positions,
                activity,
            )?;

            self.field_values[i] = field;
        }

        self.time += dt;

        Ok(())
    }

    /// Compute syntergic field at arbitrary position
    pub fn field_at(&self, point: &PoincarePoint) -> Result<f64> {
        let positions = self.neuronal_field.positions();
        let activity = self.neuronal_field.activity();

        self.green_function.compute_field(point, &positions, activity)
    }

    /// Compute non-local correlation between two positions
    ///
    /// C(x,y) = ⟨Ψ(x) Ψ(y)⟩ / √(⟨Ψ²(x)⟩⟨Ψ²(y)⟩)
    ///
    /// This measures the degree of syntergic linkage between regions.
    pub fn correlation(&self, x: &PoincarePoint, y: &PoincarePoint) -> Result<f64> {
        let psi_x = self.field_at(x)?;
        let psi_y = self.field_at(y)?;

        // For normalized correlation, we would need ensemble averaging
        // For now, return simple product normalized by field strengths
        let norm_x = psi_x.abs();
        let norm_y = psi_y.abs();

        if norm_x < 1e-10 || norm_y < 1e-10 {
            return Ok(0.0);
        }

        Ok((psi_x * psi_y) / (norm_x * norm_y))
    }

    /// Compute total field energy
    ///
    /// E = 1/2 ∫ Ψ²(x) dV
    ///
    /// Approximated as discrete sum over neurons
    pub fn total_energy(&self) -> f64 {
        self.field_values.iter().map(|&psi| psi * psi).sum::<f64>() / 2.0
    }

    /// Compute field variance (measure of coherence)
    pub fn field_variance(&self) -> f64 {
        let mean: f64 = self.field_values.iter().sum::<f64>() / self.size() as f64;

        self.field_values
            .iter()
            .map(|&psi| (psi - mean).powi(2))
            .sum::<f64>()
            / self.size() as f64
    }

    /// Compute syntergic coherence
    ///
    /// Coherence = σ²(Ψ) / ⟨Ψ²⟩
    ///
    /// Measures degree of field organization
    pub fn coherence(&self) -> f64 {
        let variance = self.field_variance();
        let mean_square: f64 = self.field_values.iter().map(|&psi| psi * psi).sum::<f64>()
            / self.size() as f64;

        if mean_square < 1e-10 {
            return 0.0;
        }

        variance / mean_square
    }

    /// Get comprehensive metrics
    pub fn metrics(&self) -> SyntergicMetrics {
        let neuronal_stats = self.neuronal_field.statistics();

        SyntergicMetrics {
            time: self.time,
            total_energy: self.total_energy(),
            field_variance: self.field_variance(),
            coherence: self.coherence(),
            neuronal_activity: neuronal_stats.total_activity,
            neuronal_entropy: neuronal_stats.entropy,
            size: self.size(),
        }
    }
}

/// Syntergic field metrics
#[derive(Debug, Clone)]
pub struct SyntergicMetrics {
    pub time: f64,
    pub total_energy: f64,
    pub field_variance: f64,
    pub coherence: f64,
    pub neuronal_activity: f64,
    pub neuronal_entropy: f64,
    pub size: usize,
}

impl std::fmt::Display for SyntergicMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Syntergic Field Metrics @ t={:.2}:\n  \
             Size: {} nodes\n  \
             Total energy: {:.6}\n  \
             Field variance: {:.6}\n  \
             Coherence: {:.6}\n  \
             Neuronal activity: {:.4}\n  \
             Neuronal entropy: {:.4}",
            self.time,
            self.size,
            self.total_energy,
            self.field_variance,
            self.coherence,
            self.neuronal_activity,
            self.neuronal_entropy
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_syntergic_field_creation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let field = SyntergicField::new(lattice, 1.0);

        assert_eq!(field.size(), field.neuronal_field().size());
        assert_eq!(field.time(), 0.0);
    }

    #[test]
    fn test_field_update() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = SyntergicField::new(lattice, 1.0);

        field.update(0.01).unwrap();

        assert!(field.time() > 0.0);
        assert!(field.field_values().len() == field.size());
    }

    #[test]
    fn test_field_energy() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = SyntergicField::new(lattice, 1.0);

        field.update(0.01).unwrap();

        let energy = field.total_energy();
        assert!(energy >= 0.0, "Energy must be non-negative");
        assert!(energy.is_finite(), "Energy must be finite");
    }

    #[test]
    fn test_coherence() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = SyntergicField::new(lattice, 1.0);

        field.update(0.01).unwrap();

        let coherence = field.coherence();
        assert!(coherence >= 0.0);
        assert!(coherence.is_finite());
    }

    #[test]
    fn test_correlation() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = SyntergicField::new(lattice, 1.0);

        field.update(0.01).unwrap();

        let origin = PoincarePoint::origin();
        let correlation = field.correlation(&origin, &origin).unwrap();

        // Autocorrelation at same point should be 1 (or close to it)
        assert!((correlation - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_metrics() {
        let lattice = PBitLattice::roi_48(1.0).unwrap();
        let mut field = SyntergicField::new(lattice, 1.0);

        field.update(0.01).unwrap();

        let metrics = field.metrics();
        assert!(metrics.total_energy >= 0.0);
        assert!(metrics.coherence >= 0.0);
        assert!(metrics.time > 0.0);
    }
}
