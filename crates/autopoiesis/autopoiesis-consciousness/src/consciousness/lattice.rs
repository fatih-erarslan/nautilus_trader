//! Information Lattice Implementation
//! Quantum information structures for consciousness substrate
//! Implements lattice-based quantum coherence and information processing

use crate::prelude::*;
use nalgebra::{Matrix4, Vector4, Complex};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum coherence threshold for information integration
const QUANTUM_COHERENCE_THRESHOLD: f64 = 0.75;

/// Planck constant (reduced)
const HBAR: f64 = 1.054571817e-34;

/// Information lattice representing quantum substrate of consciousness
#[derive(Debug, Clone)]
pub struct InformationLattice {
    /// Lattice nodes with quantum states
    nodes: Vec<Vec<Vec<QuantumNode>>>,
    
    /// Lattice dimensions
    dimensions: (usize, usize, usize),
    
    /// Coupling matrix between nodes
    coupling_matrix: HashMap<(usize, usize, usize), f64>,
    
    /// Global entanglement measure
    entanglement_entropy: f64,
    
    /// Information density distribution
    information_density: Vec<Vec<Vec<f64>>>,
    
    /// Quantum coherence matrix
    coherence_matrix: Matrix4<Complex<f64>>,
    
    /// Time evolution operator
    evolution_operator: Matrix4<Complex<f64>>,
    
    /// Current system time
    time: f64,
}

/// Quantum node in the information lattice
#[derive(Debug, Clone)]
pub struct QuantumNode {
    /// Quantum state vector (4D for spin-1/2 + position)
    state: Vector4<Complex<f64>>,
    
    /// Local energy
    energy: f64,
    
    /// Entanglement connections
    entanglements: HashMap<(usize, usize, usize), f64>,
    
    /// Information content (bits)
    information_content: f64,
    
    /// Node position in lattice
    position: (usize, usize, usize),
    
    /// Local coherence measure
    coherence: f64,
}

/// Quantum information metrics
#[derive(Debug, Clone)]
pub struct QuantumMetrics {
    pub total_entanglement: f64,
    pub coherence_strength: f64,
    pub information_capacity: f64,
    pub quantum_efficiency: f64,
    pub decoherence_rate: f64,
}

impl InformationLattice {
    pub fn new(dimensions: (usize, usize, usize)) -> Self {
        let (x, y, z) = dimensions;
        let mut nodes = vec![vec![vec![QuantumNode::new((0, 0, 0)); z]; y]; x];
        
        // Initialize node positions
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    nodes[i][j][k].position = (i, j, k);
                }
            }
        }
        
        Self {
            nodes,
            dimensions,
            coupling_matrix: HashMap::new(),
            entanglement_entropy: 0.0,
            information_density: vec![vec![vec![0.0; z]; y]; x],
            coherence_matrix: Matrix4::identity(),
            evolution_operator: Matrix4::identity(),
            time: 0.0,
        }
    }
    
    /// Initialize lattice with quantum coherent state
    pub fn initialize_coherent_state(&mut self, base_frequency: f64) {
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    // Initialize node with coherent superposition
                    let phase = 2.0 * PI * base_frequency * self.time;
                    let alpha = Complex::new(phase.cos(), phase.sin());
                    
                    self.nodes[i][j][k].state = Vector4::new(
                        alpha * 0.5,
                        alpha * 0.5,
                        alpha * 0.5,
                        alpha * 0.5,
                    );
                    
                    // Set initial information content
                    let distance_from_center = self.compute_distance_from_center(i, j, k);
                    self.information_density[i][j][k] = 
                        1.0 / (1.0 + distance_from_center * distance_from_center);
                    
                    self.nodes[i][j][k].information_content = 
                        self.information_density[i][j][k] * 8.0; // Max 8 bits per node
                }
            }
        }
        
        // Initialize coupling matrix
        self.initialize_couplings();
        
        // Update coherence matrix
        self.update_coherence_matrix();
    }
    
    /// Compute distance from lattice center
    fn compute_distance_from_center(&self, i: usize, j: usize, k: usize) -> f64 {
        let (x, y, z) = self.dimensions;
        let center_x = x as f64 / 2.0;
        let center_y = y as f64 / 2.0;
        let center_z = z as f64 / 2.0;
        
        ((i as f64 - center_x).powi(2) + 
         (j as f64 - center_y).powi(2) + 
         (k as f64 - center_z).powi(2)).sqrt()
    }
    
    /// Initialize quantum couplings between nodes
    fn initialize_couplings(&mut self) {
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    // Couple to nearest neighbors
                    let neighbors = self.get_neighbors(i, j, k);
                    
                    for neighbor in neighbors {
                        let distance = self.compute_node_distance((i, j, k), neighbor);
                        let coupling_strength = 1.0 / (1.0 + distance);
                        
                        self.coupling_matrix.insert((i, j, k), coupling_strength);
                        
                        // Store entanglement in node
                        self.nodes[i][j][k].entanglements.insert(neighbor, coupling_strength);
                    }
                }
            }
        }
    }
    
    /// Get neighboring nodes
    fn get_neighbors(&self, i: usize, j: usize, k: usize) -> Vec<(usize, usize, usize)> {
        let mut neighbors = Vec::new();
        let (x, y, z) = self.dimensions;
        
        for di in -1..=1 {
            for dj in -1..=1 {
                for dk in -1..=1 {
                    if di == 0 && dj == 0 && dk == 0 { continue; }
                    
                    let ni = i as i32 + di;
                    let nj = j as i32 + dj;
                    let nk = k as i32 + dk;
                    
                    if ni >= 0 && ni < x as i32 && 
                       nj >= 0 && nj < y as i32 && 
                       nk >= 0 && nk < z as i32 {
                        neighbors.push((ni as usize, nj as usize, nk as usize));
                    }
                }
            }
        }
        
        neighbors
    }
    
    /// Compute distance between two nodes
    fn compute_node_distance(&self, pos1: (usize, usize, usize), 
                           pos2: (usize, usize, usize)) -> f64 {
        let (x1, y1, z1) = pos1;
        let (x2, y2, z2) = pos2;
        
        ((x1 as f64 - x2 as f64).powi(2) + 
         (y1 as f64 - y2 as f64).powi(2) + 
         (z1 as f64 - z2 as f64).powi(2)).sqrt()
    }
    
    /// Evolve quantum states using Schrödinger equation
    pub fn evolve_quantum_states(&mut self, dt: f64) {
        let (x, y, z) = self.dimensions;
        
        // Update evolution operator
        self.update_evolution_operator(dt);
        
        // Evolve each node's quantum state
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    // Apply Hamiltonian evolution
                    let hamiltonian = self.compute_local_hamiltonian(i, j, k);
                    let evolution_phase = Complex::new(0.0, -dt / HBAR);
                    
                    // Simplified evolution: H|ψ⟩ → e^(-iHt/ℏ)|ψ⟩
                    let evolved_state = self.evolution_operator * self.nodes[i][j][k].state;
                    self.nodes[i][j][k].state = evolved_state;
                    
                    // Update local coherence
                    self.nodes[i][j][k].coherence = self.compute_local_coherence(i, j, k);
                    
                    // Update information content based on quantum state
                    self.update_information_content(i, j, k);
                }
            }
        }
        
        // Update global measures
        self.update_entanglement_entropy();
        self.update_coherence_matrix();
        
        self.time += dt;
    }
    
    /// Compute local Hamiltonian for a node
    fn compute_local_hamiltonian(&self, i: usize, j: usize, k: usize) -> Matrix4<Complex<f64>> {
        let mut hamiltonian = Matrix4::zeros();
        
        // Kinetic energy term (simplified)
        let kinetic_energy = Complex::new(1.0, 0.0);
        hamiltonian[(0, 0)] = kinetic_energy;
        hamiltonian[(1, 1)] = kinetic_energy;
        hamiltonian[(2, 2)] = kinetic_energy;
        hamiltonian[(3, 3)] = kinetic_energy;
        
        // Interaction terms with neighbors
        let neighbors = self.get_neighbors(i, j, k);
        for neighbor in neighbors {
            if let Some(coupling) = self.coupling_matrix.get(&(i, j, k)) {
                let interaction = Complex::new(*coupling, 0.0);
                
                // Add off-diagonal interaction terms
                hamiltonian[(0, 1)] += interaction * 0.1;
                hamiltonian[(1, 0)] += interaction * 0.1;
                hamiltonian[(2, 3)] += interaction * 0.1;
                hamiltonian[(3, 2)] += interaction * 0.1;
            }
        }
        
        hamiltonian
    }
    
    /// Update evolution operator based on current Hamiltonian
    fn update_evolution_operator(&mut self, dt: f64) {
        // Simplified: assume average Hamiltonian across lattice
        let avg_energy = self.compute_average_energy();
        let phase = -dt * avg_energy / HBAR;
        let evolution_factor = Complex::new(phase.cos(), phase.sin());
        
        // Update diagonal terms
        for i in 0..4 {
            self.evolution_operator[(i, i)] = evolution_factor;
        }
    }
    
    /// Compute average energy across lattice
    fn compute_average_energy(&self) -> f64 {
        let mut total_energy = 0.0;
        let mut count = 0;
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    total_energy += self.nodes[i][j][k].energy;
                    count += 1;
                }
            }
        }
        
        if count > 0 { total_energy / count as f64 } else { 0.0 }
    }
    
    /// Compute local quantum coherence
    fn compute_local_coherence(&self, i: usize, j: usize, k: usize) -> f64 {
        let state = &self.nodes[i][j][k].state;
        
        // Compute purity of local state: Tr(ρ²)
        let mut purity = 0.0;
        for n in 0..4 {
            purity += (state[n].norm_sqr()).powi(2);
        }
        
        // Coherence measure from off-diagonal density matrix elements
        let mut coherence = 0.0;
        for n in 0..4 {
            for m in n+1..4 {
                coherence += (state[n] * state[m].conj()).norm();
            }
        }
        
        coherence / 6.0 // Normalize by number of off-diagonal pairs
    }
    
    /// Update information content based on quantum state
    fn update_information_content(&mut self, i: usize, j: usize, k: usize) {
        let coherence = self.nodes[i][j][k].coherence;
        let entanglement_sum: f64 = self.nodes[i][j][k].entanglements.values().sum();
        
        // Information content increases with coherence and entanglement
        self.nodes[i][j][k].information_content = 
            8.0 * coherence * (1.0 + entanglement_sum / 10.0);
        
        // Update information density
        self.information_density[i][j][k] = 
            self.nodes[i][j][k].information_content / 8.0;
    }
    
    /// Update global entanglement entropy
    fn update_entanglement_entropy(&mut self) {
        let mut entropy = 0.0;
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    // Von Neumann entropy contribution
                    let coherence = self.nodes[i][j][k].coherence;
                    if coherence > 0.0 {
                        entropy -= coherence * coherence.ln();
                    }
                }
            }
        }
        
        self.entanglement_entropy = entropy / (x * y * z) as f64;
    }
    
    /// Update global coherence matrix
    fn update_coherence_matrix(&mut self) {
        let (x, y, z) = self.dimensions;
        
        // Partition lattice into 4 regions and compute cross-coherences
        let regions = self.partition_into_regions(4);
        
        for i in 0..4 {
            for j in 0..4 {
                let coherence = self.compute_inter_region_coherence(&regions[i], &regions[j]);
                self.coherence_matrix[(i, j)] = Complex::new(coherence, 0.0);
            }
        }
    }
    
    /// Partition lattice into regions
    fn partition_into_regions(&self, num_regions: usize) -> Vec<Vec<(usize, usize, usize)>> {
        let mut regions = vec![Vec::new(); num_regions];
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let region_idx = (i * num_regions / x) % num_regions;
                    regions[region_idx].push((i, j, k));
                }
            }
        }
        
        regions
    }
    
    /// Compute coherence between two regions
    fn compute_inter_region_coherence(&self, region1: &[(usize, usize, usize)], 
                                     region2: &[(usize, usize, usize)]) -> f64 {
        let mut total_coherence = 0.0;
        let mut count = 0;
        
        for &(i1, j1, k1) in region1 {
            for &(i2, j2, k2) in region2 {
                // Compute quantum state overlap
                let state1 = &self.nodes[i1][j1][k1].state;
                let state2 = &self.nodes[i2][j2][k2].state;
                
                let overlap = (state1.dot(state2)).norm();
                total_coherence += overlap;
                count += 1;
            }
        }
        
        if count > 0 { total_coherence / count as f64 } else { 0.0 }
    }
    
    /// Apply quantum measurement to collapse states
    pub fn apply_measurement(&mut self, position: (usize, usize, usize), 
                           observable: Matrix4<Complex<f64>>) -> f64 {
        let (i, j, k) = position;
        
        if i < self.dimensions.0 && j < self.dimensions.1 && k < self.dimensions.2 {
            let state = &self.nodes[i][j][k].state;
            
            // Compute expectation value ⟨ψ|O|ψ⟩
            let expectation = (state.adjoint() * observable * state)[(0, 0)].re;
            
            // Collapse state (simplified)
            let collapsed_state = observable * state;
            let norm = collapsed_state.norm();
            
            if norm > 0.0 {
                self.nodes[i][j][k].state = collapsed_state / norm;
            }
            
            // Update local measures
            self.nodes[i][j][k].coherence = self.compute_local_coherence(i, j, k);
            self.update_information_content(i, j, k);
            
            expectation
        } else {
            0.0
        }
    }
    
    /// Create entanglement between two nodes
    pub fn create_entanglement(&mut self, pos1: (usize, usize, usize), 
                              pos2: (usize, usize, usize), strength: f64) {
        let (i1, j1, k1) = pos1;
        let (i2, j2, k2) = pos2;
        
        if i1 < self.dimensions.0 && j1 < self.dimensions.1 && k1 < self.dimensions.2 &&
           i2 < self.dimensions.0 && j2 < self.dimensions.1 && k2 < self.dimensions.2 {
            
            // Add entanglement connection
            self.nodes[i1][j1][k1].entanglements.insert(pos2, strength);
            self.nodes[i2][j2][k2].entanglements.insert(pos1, strength);
            
            // Update coupling matrix
            self.coupling_matrix.insert(pos1, strength);
            self.coupling_matrix.insert(pos2, strength);
            
            // Create entangled state (Bell-like)
            let alpha = Complex::new(strength.sqrt(), 0.0);
            let beta = Complex::new((1.0 - strength).sqrt(), 0.0);
            
            self.nodes[i1][j1][k1].state[0] = alpha;
            self.nodes[i1][j1][k1].state[1] = beta;
            self.nodes[i2][j2][k2].state[0] = alpha;
            self.nodes[i2][j2][k2].state[1] = -beta; // Antisymmetric
        }
    }
    
    /// Check if lattice supports consciousness
    pub fn supports_consciousness(&self) -> bool {
        let avg_coherence = self.coherence_matrix.trace().re / 4.0;
        avg_coherence >= QUANTUM_COHERENCE_THRESHOLD
    }
    
    /// Get quantum metrics
    pub fn get_quantum_metrics(&self) -> QuantumMetrics {
        let total_entanglement = self.compute_total_entanglement();
        let coherence_strength = self.coherence_matrix.trace().re / 4.0;
        let information_capacity = self.compute_information_capacity();
        let quantum_efficiency = self.compute_quantum_efficiency();
        let decoherence_rate = self.compute_decoherence_rate();
        
        QuantumMetrics {
            total_entanglement,
            coherence_strength,
            information_capacity,
            quantum_efficiency,
            decoherence_rate,
        }
    }
    
    /// Compute total entanglement in lattice
    fn compute_total_entanglement(&self) -> f64 {
        let mut total = 0.0;
        let (x, y, z) = self.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    total += self.nodes[i][j][k].entanglements.values().sum::<f64>();
                }
            }
        }
        
        total / 2.0 // Avoid double counting
    }
    
    /// Compute total information capacity
    fn compute_information_capacity(&self) -> f64 {
        let (x, y, z) = self.dimensions;
        let mut capacity = 0.0;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    capacity += self.nodes[i][j][k].information_content;
                }
            }
        }
        
        capacity
    }
    
    /// Compute quantum processing efficiency
    fn compute_quantum_efficiency(&self) -> f64 {
        let coherence = self.coherence_matrix.trace().re / 4.0;
        let entanglement = self.entanglement_entropy;
        
        coherence * (1.0 - entanglement.abs()) // High coherence, low entropy
    }
    
    /// Estimate decoherence rate
    fn compute_decoherence_rate(&self) -> f64 {
        // Simplified: based on average coupling strength
        let avg_coupling: f64 = self.coupling_matrix.values().sum::<f64>() / 
                               self.coupling_matrix.len() as f64;
        
        1.0 / (1.0 + avg_coupling) // Higher coupling = slower decoherence
    }
    
    /// Get lattice state for analysis
    pub fn get_lattice_state(&self) -> LatticeState {
        LatticeState {
            nodes: self.nodes.clone(),
            information_density: self.information_density.clone(),
            coherence_matrix: self.coherence_matrix.clone(),
            entanglement_entropy: self.entanglement_entropy,
            supports_consciousness: self.supports_consciousness(),
            quantum_metrics: self.get_quantum_metrics(),
        }
    }
}

impl QuantumNode {
    fn new(position: (usize, usize, usize)) -> Self {
        Self {
            state: Vector4::new(
                Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
            ),
            energy: 0.0,
            entanglements: HashMap::new(),
            information_content: 0.0,
            position,
            coherence: 1.0,
        }
    }
}

/// Complete lattice state snapshot
#[derive(Debug, Clone)]
pub struct LatticeState {
    pub nodes: Vec<Vec<Vec<QuantumNode>>>,
    pub information_density: Vec<Vec<Vec<f64>>>,
    pub coherence_matrix: Matrix4<Complex<f64>>,
    pub entanglement_entropy: f64,
    pub supports_consciousness: bool,
    pub quantum_metrics: QuantumMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_information_lattice_creation() {
        let lattice = InformationLattice::new((5, 5, 5));
        assert_eq!(lattice.dimensions, (5, 5, 5));
        assert!(!lattice.supports_consciousness());
    }
    
    #[test]
    fn test_coherent_state_initialization() {
        let mut lattice = InformationLattice::new((3, 3, 3));
        lattice.initialize_coherent_state(1.0);
        
        // Check that center node has higher information density
        assert!(lattice.information_density[1][1][1] > lattice.information_density[0][0][0]);
    }
    
    #[test]
    fn test_quantum_evolution() {
        let mut lattice = InformationLattice::new((3, 3, 3));
        lattice.initialize_coherent_state(1.0);
        
        let initial_coherence = lattice.coherence_matrix.trace().re;
        lattice.evolve_quantum_states(0.01);
        let evolved_coherence = lattice.coherence_matrix.trace().re;
        
        // Coherence should change after evolution
        assert!((evolved_coherence - initial_coherence).abs() > 0.0);
    }
    
    #[test]
    fn test_entanglement_creation() {
        let mut lattice = InformationLattice::new((3, 3, 3));
        lattice.create_entanglement((0, 0, 0), (2, 2, 2), 0.8);
        
        assert!(lattice.nodes[0][0][0].entanglements.contains_key(&(2, 2, 2)));
        assert!(lattice.nodes[2][2][2].entanglements.contains_key(&(0, 0, 0)));
    }
    
    #[test]
    fn test_quantum_measurement() {
        let mut lattice = InformationLattice::new((3, 3, 3));
        lattice.initialize_coherent_state(1.0);
        
        let observable = Matrix4::identity();
        let expectation = lattice.apply_measurement((1, 1, 1), observable);
        
        assert!(expectation >= 0.0);
    }
}