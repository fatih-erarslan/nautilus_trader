//! # Consciousness Metrics Calculator
//!
//! Implementation of Integrated Information Theory (IIT) and Resonance Complexity Theory (RCT)
//!
//! ## Mathematical Foundation
//!
//! Based on peer-reviewed research:
//! - Tononi et al. (2016) "Integrated information theory" Nature Reviews Neuroscience 17:450
//! - Oizumi et al. (2014) "From phenomenology to mechanisms: IIT 3.0" PLOS Comp Bio
//! - Mayner et al. (2018) "PyPhi: A toolbox for IIT" PLOS Comp Bio 14(7):e1006343
//! - Sporns (2013) "Network attributes for segregation and integration" Curr Opin Neurobio
//!
//! ## Core Concepts
//!
//! **Integrated Information (Φ):**
//! - Quantifies consciousness as irreducibility
//! - Φ(S) = min over bipartitions P of [Effective Information across P]
//! - Computational complexity: O(2^N) - NP-hard
//!
//! **Resonance Complexity (CI):**
//! - CI = f(D, G, C, τ)
//! - D: Fractal dimension (spatial complexity)
//! - G: Gain (amplification factor)
//! - C: Coherence (temporal synchrony)
//! - τ: Dwell time (attractor stability)

pub mod phi;
pub mod ci;
pub mod causal_density;
pub mod hierarchical_phi;

pub use phi::{IntegratedInformation, PhiCalculator, PhiApproximation};
pub use ci::{ResonanceComplexity, CICalculator};
pub use causal_density::CausalDensityEstimator;
pub use hierarchical_phi::{HierarchicalPhi, HierarchicalPhiCalculator, SpatialCluster, ClusteringMethod};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum ConsciousnessError {
    #[error("System too large for exact Φ calculation: N = {size} > {max}")]
    SystemTooLarge { size: usize, max: usize },

    #[error("Invalid partition: {message}")]
    InvalidPartition { message: String },

    #[error("Computation error: {message}")]
    ComputationError { message: String },

    #[error("Numerical instability in {operation}")]
    NumericalInstability { operation: String },
}

pub type Result<T> = std::result::Result<T, ConsciousnessError>;

/// Maximum system size for exact Φ calculation
pub const MAX_EXACT_PHI_SIZE: usize = 1000;

/// Maximum system size for approximation algorithms
pub const MAX_APPROX_PHI_SIZE: usize = 1_000_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size_constants() {
        assert_eq!(MAX_EXACT_PHI_SIZE, 1000);
        assert_eq!(MAX_APPROX_PHI_SIZE, 1_000_000);
    }
}
