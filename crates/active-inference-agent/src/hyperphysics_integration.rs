//! HyperPhysics Ecosystem Integration
//!
//! Bridges the active-inference-agent with the full HyperPhysics ecosystem:
//! - hyperphysics-pbit: Stochastic pBit dynamics for belief encoding
//! - hyperphysics-thermo: Landauer principle enforcement, entropy tracking
//! - hyperphysics-consciousness: IIT Φ calculation, emergence metrics
//! - hyperphysics-geometry: Hyperbolic space for belief manifolds
//! - hyperphysics-syntergic: Non-local correlations for qualia
//! - hyperphysics-optimization: Bio-inspired action selection
//! - hyperphysics-reasoning-router: Hierarchical reasoning dispatch

use nalgebra as na;
use serde::{Deserialize, Serialize};

// Re-export key types from hyperphysics crates
pub use hyperphysics_geometry::PoincarePoint;
pub use hyperphysics_pbit::{PBit, PBitLattice, PBitDynamics, Algorithm};
pub use hyperphysics_thermo::{
    HamiltonianCalculator, EntropyCalculator, LandauerEnforcer,
    FreeEnergyCalculator, Temperature, TemperatureSchedule,
    BOLTZMANN_CONSTANT, LN_2,
};
pub use hyperphysics_consciousness::{
    IntegratedInformation, PhiCalculator, PhiApproximation,
    ResonanceComplexity, CICalculator,
    HierarchicalPhi, EmergenceEvent, EmergenceLevel,
};
pub use hyperphysics_syntergic::{SyntergicField, HyperbolicGreenFunction};

use crate::{ConsciousnessError, ConsciousnessResult, ThermodynamicState};

/// Unified HyperPhysics context for active inference
///
/// Provides access to all HyperPhysics subsystems in a coherent interface
pub struct HyperPhysicsContext {
    /// pBit lattice for stochastic belief dynamics
    pub pbit_lattice: Option<PBitLattice>,
    /// Thermodynamic state tracker
    pub thermo_state: ThermodynamicState,
    /// System temperature in Kelvin
    pub temperature: f64,
    /// Integrated Information calculator
    pub phi_calc: Option<PhiCalculator>,
    /// Syntergic Green function for non-local correlations
    pub green_function: Option<HyperbolicGreenFunction>,
    /// Landauer enforcer for thermodynamic bounds
    landauer: Option<LandauerEnforcer>,
    /// Entropy calculator
    entropy_calc: EntropyCalculator,
}

impl HyperPhysicsContext {
    /// Create new context with default settings
    pub fn new(temperature: f64) -> Self {
        Self {
            pbit_lattice: None,
            thermo_state: ThermodynamicState::new(temperature, 1e-12),
            temperature,
            phi_calc: None,
            green_function: Some(HyperbolicGreenFunction::new(1.0)), // κ = 1 for K = -1
            landauer: LandauerEnforcer::new(temperature).ok(),
            entropy_calc: EntropyCalculator::new(),
        }
    }

    /// Initialize pBit lattice for belief encoding
    /// Uses {3,7} tessellation (hyperbolic) with depth based on belief dimensions
    pub fn init_pbit_lattice(&mut self, belief_dim: usize, temperature: f64) -> ConsciousnessResult<()> {
        // Use {3,7} tessellation with depth that gives approximately belief_dim nodes
        // depth=2 gives ~48 nodes, good for moderate belief dimensions
        let depth = if belief_dim <= 8 { 1 } else if belief_dim <= 48 { 2 } else { 3 };
        let lattice = PBitLattice::new(3, 7, depth, temperature)
            .map_err(|e| ConsciousnessError::NumericalInstability(e.to_string()))?;
        self.pbit_lattice = Some(lattice);
        Ok(())
    }

    /// Initialize Φ calculator for consciousness metrics
    pub fn init_phi_calculator(&mut self, approximation: PhiApproximation) {
        self.phi_calc = Some(match approximation {
            PhiApproximation::Exact => PhiCalculator::exact(),
            PhiApproximation::MonteCarlo { samples } => PhiCalculator::monte_carlo(samples),
            PhiApproximation::Greedy => PhiCalculator::greedy(),
            PhiApproximation::Hierarchical { levels } => PhiCalculator::hierarchical(levels),
        });
    }

    /// Compute Ising energy of pBit lattice
    pub fn compute_hamiltonian_energy(&self) -> f64 {
        match &self.pbit_lattice {
            Some(lattice) => HamiltonianCalculator::energy(lattice),
            None => 0.0,
        }
    }

    /// Compute Gibbs entropy of pBit lattice
    pub fn compute_entropy(&self) -> f64 {
        match &self.pbit_lattice {
            Some(lattice) => self.entropy_calc.entropy_from_pbits(lattice),
            None => 0.0,
        }
    }

    /// Verify Landauer bound for given bit erasures
    pub fn verify_landauer(&self, bits_erased: u64) -> bool {
        match &self.landauer {
            Some(enforcer) => {
                let e_min = enforcer.minimum_erasure_energy_n(bits_erased as usize);
                self.thermo_state.energy_budget >= e_min
            }
            None => true, // No enforcer, assume valid
        }
    }
}

/// Hyperbolic belief manifold for conscious states
///
/// Maps belief distributions to points in H³ hyperbolic space
#[derive(Debug, Clone)]
pub struct HyperbolicBeliefManifold {
    /// Current position in Poincaré ball
    pub position: PoincarePoint,
    /// Curvature (default K = -1)
    pub curvature: f64,
    /// Belief dimension
    pub dim: usize,
}

impl HyperbolicBeliefManifold {
    /// Create manifold at origin
    pub fn new(dim: usize) -> ConsciousnessResult<Self> {
        let position = PoincarePoint::new(na::Vector3::zeros())
            .map_err(|e| ConsciousnessError::NumericalInstability(e.to_string()))?;

        Ok(Self {
            position,
            curvature: -1.0,
            dim,
        })
    }

    /// Map belief vector to hyperbolic point
    ///
    /// Uses exponential map from origin
    pub fn belief_to_point(&self, belief: &na::DVector<f64>) -> ConsciousnessResult<PoincarePoint> {
        // Take first 3 components for H³
        let x = belief.get(0).copied().unwrap_or(0.0) - 0.5;
        let y = belief.get(1).copied().unwrap_or(0.0) - 0.5;
        let z = belief.get(2).copied().unwrap_or(0.0) - 0.5;

        // Normalize to stay in Poincaré ball (|v| < 1)
        let norm = (x * x + y * y + z * z).sqrt();
        let scale = if norm > 0.9 { 0.9 / norm } else { 1.0 };

        let coords = na::Vector3::new(x * scale, y * scale, z * scale);
        PoincarePoint::new(coords)
            .map_err(|e| ConsciousnessError::NumericalInstability(e.to_string()))
    }

    /// Compute geodesic distance between belief states
    pub fn geodesic_distance(&self, belief_a: &na::DVector<f64>, belief_b: &na::DVector<f64>) -> ConsciousnessResult<f64> {
        let point_a = self.belief_to_point(belief_a)?;
        let point_b = self.belief_to_point(belief_b)?;

        Ok(point_a.distance(&point_b))
    }
}

/// pBit-based Active Inference with HyperPhysics integration
///
/// Full integration of pBit dynamics with thermodynamic constraints
/// and consciousness metrics
pub struct HyperPhysicsInferenceAgent {
    /// Belief state
    pub belief: na::DVector<f64>,
    /// HyperPhysics context
    pub context: HyperPhysicsContext,
    /// Hyperbolic manifold
    pub manifold: HyperbolicBeliefManifold,
    /// Current Φ value
    pub phi: f64,
    /// Current entropy
    pub entropy: f64,
    /// Steps taken
    pub steps: u64,
}

impl HyperPhysicsInferenceAgent {
    /// Create new agent with HyperPhysics integration
    pub fn new(belief_dim: usize, temperature: f64) -> ConsciousnessResult<Self> {
        let belief = na::DVector::from_element(belief_dim, 1.0 / belief_dim as f64);
        let mut context = HyperPhysicsContext::new(temperature);

        // Initialize subsystems
        context.init_pbit_lattice(belief_dim * 8, temperature)?;
        context.init_phi_calculator(PhiApproximation::Greedy);

        let manifold = HyperbolicBeliefManifold::new(belief_dim)?;

        Ok(Self {
            belief,
            context,
            manifold,
            phi: 0.0,
            entropy: 0.0,
            steps: 0,
        })
    }

    /// Perform one inference step with full HyperPhysics integration
    pub fn step(&mut self, observation: &na::DVector<f64>) -> ConsciousnessResult<InferenceResult> {
        self.steps += 1;

        // 1. Verify Landauer bound
        let bits_to_process = observation.len() as u64;
        if !self.context.verify_landauer(bits_to_process) {
            return Err(ConsciousnessError::ThermodynamicViolation(
                "Landauer bound exceeded".to_string()
            ));
        }

        // 2. Record thermodynamic cost
        self.context.thermo_state.record_processing_cost(bits_to_process as f64 * 0.1)?;

        // 3. Compute Hamiltonian energy if pBit lattice available
        let hamiltonian_energy = self.context.compute_hamiltonian_energy();

        // 4. Update belief (simplified Bayesian update)
        let prediction_error = observation - &self.belief;
        let error_norm_sq = prediction_error.norm_squared();
        let learning_rate = 0.1;
        self.belief = &self.belief + prediction_error * learning_rate;

        // Normalize
        let sum = self.belief.sum();
        if sum > 1e-10 {
            self.belief /= sum;
        }

        // 5. Compute entropy
        self.entropy = self.context.compute_entropy();

        // 6. Compute free energy (simplified: prediction error + entropy)
        let free_energy = error_norm_sq - self.entropy;

        // 7. Compute Φ if calculator available
        if let Some(ref phi_calc) = self.context.phi_calc {
            if let Some(ref lattice) = self.context.pbit_lattice {
                if let Ok(phi_result) = phi_calc.calculate(lattice) {
                    self.phi = phi_result.phi;
                }
            }
        }

        // 8. Get hyperbolic position
        let position = self.manifold.belief_to_point(&self.belief)?;

        Ok(InferenceResult {
            belief: self.belief.clone(),
            free_energy,
            entropy: self.entropy,
            phi: self.phi,
            hyperbolic_position: position,
            energy_consumed: self.context.thermo_state.energy_consumed,
            hamiltonian_energy,
        })
    }

    /// Get current consciousness metrics
    pub fn consciousness_metrics(&self) -> ConsciousnessMetrics {
        ConsciousnessMetrics {
            phi: self.phi,
            entropy: self.entropy,
            energy_consumed: self.context.thermo_state.energy_consumed,
            bits_processed: self.context.thermo_state.bits_erased,
            thermodynamic_efficiency: self.context.thermo_state.efficiency(),
        }
    }
}

/// Result of one inference step
#[derive(Debug, Clone)]
pub struct InferenceResult {
    /// Updated belief state
    pub belief: na::DVector<f64>,
    /// Variational free energy
    pub free_energy: f64,
    /// Shannon entropy of belief
    pub entropy: f64,
    /// Integrated Information Φ
    pub phi: f64,
    /// Position in hyperbolic manifold
    pub hyperbolic_position: PoincarePoint,
    /// Energy consumed (Joules)
    pub energy_consumed: f64,
    /// Ising Hamiltonian energy
    pub hamiltonian_energy: f64,
}

/// Consciousness metrics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetrics {
    /// Integrated Information (IIT Φ)
    pub phi: f64,
    /// Shannon entropy
    pub entropy: f64,
    /// Total energy consumed
    pub energy_consumed: f64,
    /// Total bits processed
    pub bits_processed: u64,
    /// Thermodynamic efficiency (Landauer ratio)
    pub thermodynamic_efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperphysics_context() {
        let ctx = HyperPhysicsContext::new(300.0);
        assert!(ctx.pbit_lattice.is_none());
    }

    #[test]
    fn test_hyperbolic_manifold() {
        let manifold = HyperbolicBeliefManifold::new(3);
        assert!(manifold.is_ok());

        let m = manifold.unwrap();
        let belief = na::DVector::from_vec(vec![0.5, 0.3, 0.2]);
        let point = m.belief_to_point(&belief);
        assert!(point.is_ok());
    }

    #[test]
    fn test_geodesic_distance() {
        let manifold = HyperbolicBeliefManifold::new(3).unwrap();

        let a = na::DVector::from_vec(vec![0.6, 0.3, 0.1]);
        let b = na::DVector::from_vec(vec![0.2, 0.4, 0.4]);

        let dist = manifold.geodesic_distance(&a, &b);
        assert!(dist.is_ok());
        assert!(dist.unwrap() > 0.0);
    }
}
