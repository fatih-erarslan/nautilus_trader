//! Adapter implementations bridging autopoiesis components to HyperPhysics ecosystem
//!
//! This module provides adapter patterns that translate between autopoiesis abstractions
//! and concrete HyperPhysics implementations, maintaining mathematical rigor and
//! scientific grounding throughout.
//!
//! ## Adapter Architecture
//!
//! Each adapter implements a bidirectional translation:
//! - **Forward**: Autopoiesis concepts → HyperPhysics calculations
//! - **Reverse**: HyperPhysics results → Autopoiesis state updates
//!
//! ## References
//!
//! - Maturana & Varela (1980) "Autopoiesis and Cognition"
//! - Prigogine & Stengers (1984) "Order Out of Chaos"
//! - Tononi (2004) "An information integration theory of consciousness"

use nalgebra::DMatrix;

use crate::error::{AutopoiesisError, Result};

/// Adapter bridging dissipative structures to HyperPhysics thermodynamics
///
/// Maps Prigogine's entropy production and fluctuation theorems to
/// hyperphysics-thermo calculations using Landauer principle and Jarzynski equality.
///
/// ## Mathematical Foundation
///
/// The entropy production rate σ is computed as:
/// σ = dS/dt = Σᵢ Jᵢ Xᵢ
///
/// where Jᵢ are thermodynamic fluxes and Xᵢ are thermodynamic forces.
///
/// ## References
/// - Prigogine (1967) "Introduction to Thermodynamics of Irreversible Processes"
/// - Landauer (1961) "Irreversibility and Heat Generation in the Computing Process"
#[derive(Debug, Clone)]
pub struct ThermoAdapter {
    /// Current entropy production rate (σ)
    entropy_production: f64,
    /// Thermodynamic fluxes (Jᵢ)
    fluxes: Vec<f64>,
    /// Thermodynamic forces (Xᵢ)
    forces: Vec<f64>,
    /// Landauer limit for information erasure (kT ln 2)
    landauer_limit: f64,
    /// Temperature in Kelvin
    temperature: f64,
    /// Boltzmann constant (J/K)
    boltzmann_k: f64,
}

impl Default for ThermoAdapter {
    fn default() -> Self {
        Self::new(300.0) // Room temperature
    }
}

impl ThermoAdapter {
    /// Create a new thermodynamic adapter at specified temperature
    pub fn new(temperature: f64) -> Self {
        let boltzmann_k = crate::BOLTZMANN_CONSTANT;
        let landauer_limit = boltzmann_k * temperature * std::f64::consts::LN_2;

        Self {
            entropy_production: 0.0,
            fluxes: Vec::new(),
            forces: Vec::new(),
            landauer_limit,
            temperature,
            boltzmann_k,
        }
    }

    /// Compute entropy production rate from thermodynamic fluxes and forces
    ///
    /// σ = Σᵢ Jᵢ Xᵢ (Onsager reciprocal relations)
    pub fn compute_entropy_production(&mut self, fluxes: &[f64], forces: &[f64]) -> Result<f64> {
        if fluxes.len() != forces.len() {
            return Err(AutopoiesisError::NumericalError {
                operation: "entropy_production".to_string(),
                message: "Flux and force vectors must have equal length".to_string(),
            });
        }

        self.fluxes = fluxes.to_vec();
        self.forces = forces.to_vec();

        self.entropy_production = fluxes
            .iter()
            .zip(forces.iter())
            .map(|(j, x)| j * x)
            .sum();

        Ok(self.entropy_production)
    }

    /// Map dissipative entropy to HyperPhysics Hamiltonian energy
    ///
    /// Uses the fluctuation theorem: P(σ)/P(-σ) = exp(σ τ / kT)
    pub fn entropy_to_hamiltonian(&self, entropy: f64) -> f64 {
        // Convert entropy production to energy dissipation via temperature
        self.temperature * entropy
    }

    /// Verify Landauer principle compliance for information processing
    ///
    /// Returns true if the energy dissipation meets minimum Landauer bound
    pub fn verify_landauer_compliance(&self, bits_erased: usize, energy_dissipated: f64) -> bool {
        let min_energy = self.landauer_limit * bits_erased as f64;
        energy_dissipated >= min_energy
    }

    /// Get current entropy production rate
    pub fn entropy_production(&self) -> f64 {
        self.entropy_production
    }

    /// Get Landauer limit at current temperature
    pub fn landauer_limit(&self) -> f64 {
        self.landauer_limit
    }

    /// Update temperature and recalculate Landauer limit
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
        self.landauer_limit = self.boltzmann_k * temperature * std::f64::consts::LN_2;
    }
}

/// Adapter bridging syntergic coherence to HyperPhysics consciousness (IIT Φ)
///
/// Maps Grinberg-Zylberbaum's neuronal field coherence to Tononi's integrated
/// information (Φ), establishing equivalence between syntergic unity and
/// consciousness integration.
///
/// ## Mathematical Foundation
///
/// Syntergic coherence C maps to integrated information Φ via:
/// Φ ∝ C × I(X; X')
///
/// where I(X; X') is mutual information between system partitions.
///
/// ## References
/// - Grinberg-Zylberbaum (1995) "Syntergic Theory"
/// - Tononi (2004) "An information integration theory of consciousness"
#[derive(Debug, Clone)]
pub struct ConsciousnessAdapter {
    /// Current integrated information Φ
    phi: f64,
    /// Syntergic coherence value
    coherence: f64,
    /// Information matrix for partition analysis
    information_matrix: DMatrix<f64>,
    /// Unity threshold (typically 0.9)
    unity_threshold: f64,
}

impl Default for ConsciousnessAdapter {
    fn default() -> Self {
        Self::new(crate::SYNTERGIC_UNITY_THRESHOLD)
    }
}

impl ConsciousnessAdapter {
    /// Create new consciousness adapter with unity threshold
    pub fn new(unity_threshold: f64) -> Self {
        Self {
            phi: 0.0,
            coherence: 0.0,
            information_matrix: DMatrix::zeros(1, 1),
            unity_threshold,
        }
    }

    /// Map syntergic field coherence to integrated information Φ
    ///
    /// Uses mutual information between partitions scaled by coherence
    pub fn coherence_to_phi(&mut self, coherence: f64, partition_mi: &DMatrix<f64>) -> Result<f64> {
        if coherence < 0.0 || coherence > 1.0 {
            return Err(AutopoiesisError::NumericalError {
                operation: "coherence_to_phi".to_string(),
                message: format!("Coherence must be in [0,1], got {}", coherence),
            });
        }

        self.coherence = coherence;
        self.information_matrix = partition_mi.clone();

        // Compute minimum information partition (MIP)
        let min_partition_info = self.compute_mip(partition_mi)?;

        // Φ = coherence × minimum partition information
        self.phi = coherence * min_partition_info;

        Ok(self.phi)
    }

    /// Compute Minimum Information Partition (MIP)
    fn compute_mip(&self, mi_matrix: &DMatrix<f64>) -> Result<f64> {
        if mi_matrix.nrows() == 0 || mi_matrix.ncols() == 0 {
            return Ok(0.0);
        }

        // MIP is the partition that minimizes information across the cut
        // For symmetric systems, this is the minimum eigenvalue > 0
        let eigenvalues: Vec<f64> = mi_matrix
            .symmetric_eigenvalues()
            .iter()
            .copied()
            .filter(|&e| e > 1e-10)
            .collect();

        if eigenvalues.is_empty() {
            return Ok(0.0);
        }

        Ok(eigenvalues.iter().cloned().fold(f64::INFINITY, f64::min))
    }

    /// Check if syntergic unity threshold is achieved
    pub fn unity_achieved(&self) -> bool {
        self.coherence >= self.unity_threshold
    }

    /// Get current integrated information Φ
    pub fn phi(&self) -> f64 {
        self.phi
    }

    /// Get current syntergic coherence
    pub fn coherence(&self) -> f64 {
        self.coherence
    }
}

/// Adapter bridging Kuramoto synchronization to HyperPhysics syntergic fields
///
/// Maps phase synchronization dynamics from Strogatz's coupled oscillator models
/// to syntergic field coherence measures.
///
/// ## Mathematical Foundation
///
/// Kuramoto order parameter: r = |⟨e^(iθⱼ)⟩|
///
/// Maps to syntergic coherence via: C = r² (for normalized coherence)
///
/// ## References
/// - Strogatz (2000) "From Kuramoto to Crawford"
/// - Acebrón et al. (2005) "The Kuramoto model"
#[derive(Debug, Clone)]
pub struct SyncAdapter {
    /// Current Kuramoto order parameter r
    order_parameter: f64,
    /// Phase angles of oscillators
    phases: Vec<f64>,
    /// Natural frequencies
    frequencies: Vec<f64>,
    /// Coupling strength K
    coupling_strength: f64,
    /// Critical coupling for synchronization
    critical_coupling: f64,
}

impl Default for SyncAdapter {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl SyncAdapter {
    /// Create new synchronization adapter with coupling strength
    pub fn new(coupling_strength: f64) -> Self {
        Self {
            order_parameter: 0.0,
            phases: Vec::new(),
            frequencies: Vec::new(),
            coupling_strength,
            critical_coupling: 0.0,
        }
    }

    /// Compute Kuramoto order parameter from oscillator phases
    ///
    /// r = |N⁻¹ Σⱼ e^(iθⱼ)|
    pub fn compute_order_parameter(&mut self, phases: &[f64]) -> f64 {
        if phases.is_empty() {
            self.order_parameter = 0.0;
            return 0.0;
        }

        self.phases = phases.to_vec();
        let n = phases.len() as f64;

        let (sum_cos, sum_sin) = phases
            .iter()
            .fold((0.0, 0.0), |(c, s), &theta| {
                (c + theta.cos(), s + theta.sin())
            });

        self.order_parameter = ((sum_cos / n).powi(2) + (sum_sin / n).powi(2)).sqrt();
        self.order_parameter
    }

    /// Map order parameter to syntergic coherence
    ///
    /// Coherence C = r² for energy-like interpretation
    pub fn order_to_coherence(&self) -> f64 {
        self.order_parameter.powi(2)
    }

    /// Compute critical coupling for frequency distribution
    ///
    /// Kc = 2 / (π g(0)) for Lorentzian distribution
    pub fn compute_critical_coupling(&mut self, frequencies: &[f64]) -> f64 {
        self.frequencies = frequencies.to_vec();

        if frequencies.is_empty() {
            self.critical_coupling = 0.0;
            return 0.0;
        }

        // Estimate width of frequency distribution (std dev)
        let mean: f64 = frequencies.iter().sum::<f64>() / frequencies.len() as f64;
        let variance: f64 = frequencies
            .iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f64>() / frequencies.len() as f64;
        let std_dev = variance.sqrt();

        // Kc ≈ 2σ√(2/π) for Gaussian-like distribution
        self.critical_coupling = 2.0 * std_dev * (2.0 / std::f64::consts::PI).sqrt();
        self.critical_coupling
    }

    /// Check if system is synchronized (above critical coupling)
    pub fn is_synchronized(&self) -> bool {
        self.coupling_strength > self.critical_coupling && self.order_parameter > 0.5
    }

    /// Get current order parameter
    pub fn order_parameter(&self) -> f64 {
        self.order_parameter
    }

    /// Set coupling strength
    pub fn set_coupling_strength(&mut self, k: f64) {
        self.coupling_strength = k;
    }
}

/// Adapter bridging ecological network patterns to HyperPhysics risk networks
///
/// Maps Capra's Web of Life network patterns to financial risk network analysis,
/// using graph-theoretic measures of connectivity and flow.
///
/// ## Mathematical Foundation
///
/// Network health H derived from:
/// - Connectivity: average degree ⟨k⟩
/// - Clustering: C = (triangles) / (possible triangles)
/// - Flow efficiency: E = ⟨1/dᵢⱼ⟩
///
/// ## References
/// - Capra (1996) "The Web of Life"
/// - Newman (2003) "The Structure and Function of Complex Networks"
#[derive(Debug, Clone)]
pub struct NetworkAdapter {
    /// Number of nodes
    node_count: usize,
    /// Adjacency matrix
    adjacency: DMatrix<f64>,
    /// Network connectivity (average degree)
    connectivity: f64,
    /// Clustering coefficient
    clustering: f64,
    /// Flow efficiency
    efficiency: f64,
}

impl Default for NetworkAdapter {
    fn default() -> Self {
        Self::new(0)
    }
}

impl NetworkAdapter {
    /// Create new network adapter for given node count
    pub fn new(node_count: usize) -> Self {
        Self {
            node_count,
            adjacency: DMatrix::zeros(node_count, node_count),
            connectivity: 0.0,
            clustering: 0.0,
            efficiency: 0.0,
        }
    }

    /// Set network adjacency matrix and compute metrics
    pub fn set_adjacency(&mut self, adjacency: DMatrix<f64>) -> Result<()> {
        if adjacency.nrows() != adjacency.ncols() {
            return Err(AutopoiesisError::NetworkTopologyError {
                message: "Adjacency matrix must be square".to_string(),
            });
        }

        self.node_count = adjacency.nrows();
        self.adjacency = adjacency;

        self.compute_connectivity();
        self.compute_clustering();
        self.compute_efficiency();

        Ok(())
    }

    /// Compute average degree (connectivity)
    fn compute_connectivity(&mut self) {
        if self.node_count == 0 {
            self.connectivity = 0.0;
            return;
        }

        let total_degree: f64 = self.adjacency.iter().filter(|&&x| x > 0.0).count() as f64;
        self.connectivity = total_degree / self.node_count as f64;
    }

    /// Compute clustering coefficient
    fn compute_clustering(&mut self) {
        if self.node_count < 3 {
            self.clustering = 0.0;
            return;
        }

        let mut triangles = 0.0;
        let mut possible_triangles = 0.0;

        for i in 0..self.node_count {
            let neighbors: Vec<usize> = (0..self.node_count)
                .filter(|&j| self.adjacency[(i, j)] > 0.0)
                .collect();

            let k = neighbors.len();
            if k < 2 {
                continue;
            }

            possible_triangles += (k * (k - 1)) as f64 / 2.0;

            for (idx, &j) in neighbors.iter().enumerate() {
                for &l in neighbors.iter().skip(idx + 1) {
                    if self.adjacency[(j, l)] > 0.0 {
                        triangles += 1.0;
                    }
                }
            }
        }

        self.clustering = if possible_triangles > 0.0 {
            triangles / possible_triangles
        } else {
            0.0
        };
    }

    /// Compute global efficiency (inverse path length)
    fn compute_efficiency(&mut self) {
        if self.node_count < 2 {
            self.efficiency = 0.0;
            return;
        }

        // Use Floyd-Warshall for shortest paths
        let mut dist = self.adjacency.clone();

        // Initialize distances
        for i in 0..self.node_count {
            for j in 0..self.node_count {
                if i == j {
                    dist[(i, j)] = 0.0;
                } else if dist[(i, j)] == 0.0 {
                    dist[(i, j)] = f64::INFINITY;
                } else {
                    dist[(i, j)] = 1.0; // Unweighted
                }
            }
        }

        // Floyd-Warshall
        for k in 0..self.node_count {
            for i in 0..self.node_count {
                for j in 0..self.node_count {
                    if dist[(i, k)] + dist[(k, j)] < dist[(i, j)] {
                        dist[(i, j)] = dist[(i, k)] + dist[(k, j)];
                    }
                }
            }
        }

        // Compute efficiency
        let mut sum_inv_dist = 0.0;
        for i in 0..self.node_count {
            for j in 0..self.node_count {
                if i != j && dist[(i, j)].is_finite() && dist[(i, j)] > 0.0 {
                    sum_inv_dist += 1.0 / dist[(i, j)];
                }
            }
        }

        let n = self.node_count as f64;
        self.efficiency = sum_inv_dist / (n * (n - 1.0));
    }

    /// Compute composite network health score
    ///
    /// Combines connectivity, clustering, and efficiency into single metric
    pub fn network_health(&self) -> f64 {
        // Weighted combination: 40% connectivity, 30% clustering, 30% efficiency
        0.4 * (self.connectivity / self.node_count.max(1) as f64).min(1.0)
            + 0.3 * self.clustering
            + 0.3 * self.efficiency
    }

    /// Map network metrics to risk exposure
    ///
    /// Low connectivity and clustering indicate fragility → higher risk
    pub fn to_risk_exposure(&self) -> f64 {
        1.0 - self.network_health()
    }

    /// Get connectivity metric
    pub fn connectivity(&self) -> f64 {
        self.connectivity
    }

    /// Get clustering coefficient
    pub fn clustering(&self) -> f64 {
        self.clustering
    }

    /// Get network efficiency
    pub fn efficiency(&self) -> f64 {
        self.efficiency
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;

    #[test]
    fn test_thermo_adapter_entropy_production() {
        let mut adapter = ThermoAdapter::new(300.0);
        let fluxes = vec![1.0, 2.0, 3.0];
        let forces = vec![0.5, 0.5, 0.5];

        let sigma = adapter.compute_entropy_production(&fluxes, &forces).unwrap();
        assert_relative_eq!(sigma, 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_thermo_adapter_landauer() {
        let adapter = ThermoAdapter::new(300.0);
        let expected = crate::BOLTZMANN_CONSTANT * 300.0 * std::f64::consts::LN_2;
        assert_relative_eq!(adapter.landauer_limit(), expected, epsilon = 1e-30);
    }

    #[test]
    fn test_consciousness_adapter_phi() {
        let mut adapter = ConsciousnessAdapter::new(0.9);
        let mi_matrix = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 0.5, 0.3]));

        let phi = adapter.coherence_to_phi(0.8, &mi_matrix).unwrap();
        assert!(phi > 0.0);
        assert_relative_eq!(adapter.coherence(), 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_sync_adapter_order_parameter() {
        let mut adapter = SyncAdapter::new(1.0);

        // All phases aligned → r = 1
        let aligned_phases = vec![0.0, 0.0, 0.0, 0.0];
        let r = adapter.compute_order_parameter(&aligned_phases);
        assert_relative_eq!(r, 1.0, epsilon = 1e-10);

        // Phases uniformly distributed → r ≈ 0
        let uniform_phases = vec![0.0, std::f64::consts::FRAC_PI_2,
                                  std::f64::consts::PI, 3.0 * std::f64::consts::FRAC_PI_2];
        let r = adapter.compute_order_parameter(&uniform_phases);
        assert!(r < 0.1);
    }

    #[test]
    fn test_network_adapter_complete_graph() {
        let mut adapter = NetworkAdapter::new(4);
        let mut adj = DMatrix::from_element(4, 4, 1.0);
        for i in 0..4 {
            adj[(i, i)] = 0.0; // No self-loops
        }

        adapter.set_adjacency(adj).unwrap();

        // Complete graph has clustering = 1
        assert_relative_eq!(adapter.clustering(), 1.0, epsilon = 1e-10);
        // Complete graph has max efficiency
        assert!(adapter.efficiency() > 0.9);
    }
}
