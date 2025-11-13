//! Engine configuration

use serde::{Deserialize, Serialize};

/// System scale
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Scale {
    /// 48 nodes (ROI study)
    Micro,
    /// 16,384 nodes (128×128)
    Small,
    /// 1,048,576 nodes (1024×1024)
    Medium,
    /// 1 billion nodes
    Large,
}

impl Scale {
    /// Get approximate number of nodes
    pub fn num_nodes(&self) -> usize {
        match self {
            Scale::Micro => 48,
            Scale::Small => 16_384,
            Scale::Medium => 1_048_576,
            Scale::Large => 1_000_000_000,
        }
    }

    /// Get tessellation parameters {p, q, depth}
    pub fn tessellation_params(&self) -> (usize, usize, usize) {
        match self {
            Scale::Micro => (3, 7, 2),    // {3,7,2} ≈ 48 nodes
            Scale::Small => (3, 7, 4),    // Larger depth
            Scale::Medium => (4, 5, 5),   // Different geometry
            Scale::Large => (5, 4, 6),    // Maximum depth
        }
    }
}

/// Engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// System scale
    pub scale: Scale,

    /// pBit temperature (Kelvin)
    pub temperature: f64,

    /// Coupling strength at zero distance
    pub coupling_j0: f64,

    /// Coupling length scale
    pub coupling_lambda: f64,

    /// Minimum coupling threshold
    pub coupling_min: f64,

    /// Simulation algorithm
    pub algorithm: hyperphysics_pbit::Algorithm,

    /// Calculate Φ (computationally expensive)
    pub calculate_phi: bool,

    /// Calculate CI
    pub calculate_ci: bool,

    /// Verify thermodynamic laws
    pub verify_thermodynamics: bool,
}

impl EngineConfig {
    /// Default configuration for ROI (48 nodes)
    pub fn roi_48(temperature: f64) -> Self {
        Self {
            scale: Scale::Micro,
            temperature,
            coupling_j0: 1.0,
            coupling_lambda: 1.0,
            coupling_min: 1e-6,
            algorithm: hyperphysics_pbit::Algorithm::Gillespie,
            calculate_phi: true,
            calculate_ci: true,
            verify_thermodynamics: true,
        }
    }

    /// Configuration for small scale (16K nodes)
    pub fn small_scale(temperature: f64) -> Self {
        Self {
            scale: Scale::Small,
            temperature,
            coupling_j0: 1.0,
            coupling_lambda: 1.0,
            coupling_min: 1e-6,
            algorithm: hyperphysics_pbit::Algorithm::Metropolis,
            calculate_phi: false, // Too expensive
            calculate_ci: true,
            verify_thermodynamics: true,
        }
    }

    /// Configuration for 128×128 lattice (16,384 nodes)
    pub fn roi_128x128(temperature: f64) -> Self {
        Self {
            scale: Scale::Small,
            temperature,
            coupling_j0: 0.8,
            coupling_lambda: 1.2,
            coupling_min: 1e-7,
            algorithm: hyperphysics_pbit::Algorithm::Metropolis,
            calculate_phi: false, // Use approximation
            calculate_ci: true,
            verify_thermodynamics: true,
        }
    }

    /// Configuration for 1024×1024 lattice (1M nodes)
    pub fn roi_1024x1024(temperature: f64) -> Self {
        Self {
            scale: Scale::Medium,
            temperature,
            coupling_j0: 0.6,
            coupling_lambda: 1.5,
            coupling_min: 1e-8,
            algorithm: hyperphysics_pbit::Algorithm::Metropolis,
            calculate_phi: false, // Use hierarchical approximation
            calculate_ci: true,
            verify_thermodynamics: false, // Too expensive
        }
    }

    /// Configuration for 32K×32K lattice (1B nodes)
    pub fn roi_32kx32k(temperature: f64) -> Self {
        Self {
            scale: Scale::Large,
            temperature,
            coupling_j0: 0.4,
            coupling_lambda: 2.0,
            coupling_min: 1e-9,
            algorithm: hyperphysics_pbit::Algorithm::Metropolis,
            calculate_phi: false, // Use hierarchical approximation only
            calculate_ci: false,  // Too expensive for 1B nodes
            verify_thermodynamics: false,
        }
    }

    /// Fast configuration (minimal metrics)
    pub fn fast(scale: Scale, temperature: f64) -> Self {
        Self {
            scale,
            temperature,
            coupling_j0: 1.0,
            coupling_lambda: 1.0,
            coupling_min: 1e-3, // Sparser network
            algorithm: hyperphysics_pbit::Algorithm::Metropolis,
            calculate_phi: false,
            calculate_ci: false,
            verify_thermodynamics: false,
        }
    }
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self::roi_48(300.0) // Room temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_nodes() {
        assert_eq!(Scale::Micro.num_nodes(), 48);
        assert_eq!(Scale::Small.num_nodes(), 16_384);
        assert_eq!(Scale::Medium.num_nodes(), 1_048_576);
        assert_eq!(Scale::Large.num_nodes(), 1_000_000_000);
    }

    #[test]
    fn test_default_config() {
        let config = EngineConfig::default();
        assert_eq!(config.scale, Scale::Micro);
        assert_eq!(config.temperature, 300.0);
    }
}
