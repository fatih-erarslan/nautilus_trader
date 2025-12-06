//! pBit Lattice Bridge
//!
//! Bridges PBitState with the HyperPhysics PBitLattice from hyperphysics-swarm-intelligence.
//! This enables quantum algorithms to run on the spatiotemporal lattice infrastructure.
//!
//! ## Architecture
//!
//! ```text
//! QuantumState ←→ PBitState ←→ PBitLattice
//!   (complex)      (prob)      (Ising)
//! ```

use crate::error::{QuantumError, QuantumResult};
use crate::pbit_state::{PBitConfig, PBitCoupling, PBitState};
use crate::quantum_state::QuantumState;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for lattice bridge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeBridgeConfig {
    /// Lattice dimensions (x, y, z) - total nodes = num_qubits
    pub dimensions: (usize, usize, usize),
    /// Temperature for Boltzmann sampling
    pub temperature: f64,
    /// Coupling strength between adjacent lattice nodes
    pub coupling_strength: f64,
    /// External field
    pub external_field: f64,
    /// STDP learning rate
    pub stdp_learning_rate: f64,
    /// STDP temporal window (ms)
    pub stdp_window: f64,
    /// Use periodic boundary conditions
    pub periodic_boundary: bool,
    /// Number of equilibration sweeps
    pub equilibration_sweeps: usize,
    /// Annealing steps
    pub annealing_steps: usize,
    /// Target temperature after annealing
    pub target_temperature: f64,
}

impl Default for LatticeBridgeConfig {
    fn default() -> Self {
        Self {
            dimensions: (4, 4, 1), // 16 qubits in 2D lattice
            temperature: 1.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            stdp_learning_rate: 0.01,
            stdp_window: 20.0,
            periodic_boundary: true,
            equilibration_sweeps: 100,
            annealing_steps: 100,
            target_temperature: 0.1,
        }
    }
}

impl LatticeBridgeConfig {
    /// Create config for given number of qubits (auto-dimensions)
    pub fn for_qubits(num_qubits: usize) -> Self {
        let dims = Self::auto_dimensions(num_qubits);
        Self {
            dimensions: dims,
            ..Default::default()
        }
    }

    /// Compute optimal lattice dimensions for qubit count
    fn auto_dimensions(num_qubits: usize) -> (usize, usize, usize) {
        if num_qubits <= 1 {
            return (1, 1, 1);
        }

        // Try to find square-ish 2D layout first
        let sqrt = (num_qubits as f64).sqrt() as usize;
        if sqrt * sqrt == num_qubits {
            return (sqrt, sqrt, 1);
        }

        // Try to find factors
        for y in (1..=sqrt).rev() {
            if num_qubits % y == 0 {
                let x = num_qubits / y;
                return (x, y, 1);
            }
        }

        // Fallback to linear
        (num_qubits, 1, 1)
    }

    /// Total number of lattice nodes
    pub fn total_nodes(&self) -> usize {
        self.dimensions.0 * self.dimensions.1 * self.dimensions.2
    }
}

/// Represents a node in the lattice with its state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeNode {
    /// Position in lattice (x, y, z)
    pub position: (usize, usize, usize),
    /// Qubit index this node maps to
    pub qubit_index: usize,
    /// Current spin (-1 or +1)
    pub spin: f64,
    /// Probability of spin up
    pub probability_up: f64,
    /// Local energy
    pub energy: f64,
    /// Bias field
    pub bias: f64,
    /// Last spike time (for STDP)
    pub last_spike_time: f64,
    /// Activation level
    pub activation: f64,
}

impl LatticeNode {
    pub fn new(position: (usize, usize, usize), qubit_index: usize) -> Self {
        Self {
            position,
            qubit_index,
            spin: -1.0, // Down = |0⟩
            probability_up: 0.0,
            energy: 0.0,
            bias: 0.0,
            last_spike_time: f64::NEG_INFINITY,
            activation: 0.0,
        }
    }

    /// Update probability from local field
    pub fn update_probability(&mut self, local_field: f64, temperature: f64) {
        let exponent = -2.0 * local_field / temperature.max(1e-10);
        self.probability_up = 1.0 / (1.0 + exponent.exp());
    }

    /// Sample spin from probability
    pub fn sample(&mut self) {
        self.spin = if rand::random::<f64>() < self.probability_up {
            1.0
        } else {
            -1.0
        };
    }

    /// Compute entropy
    pub fn entropy(&self) -> f64 {
        let p = self.probability_up.clamp(1e-10, 1.0 - 1e-10);
        -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
    }
}

/// Coupling between lattice nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeCoupling {
    /// Source node position
    pub from: (usize, usize, usize),
    /// Target node position
    pub to: (usize, usize, usize),
    /// Coupling weight
    pub weight: f64,
    /// STDP trace
    pub trace: f64,
}

/// Lattice state that can bridge with PBitState
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeState {
    /// Configuration
    config: LatticeBridgeConfig,
    /// Lattice nodes (flattened for efficiency)
    nodes: Vec<LatticeNode>,
    /// Couplings between nodes
    couplings: HashMap<((usize, usize, usize), (usize, usize, usize)), LatticeCoupling>,
    /// Current simulation time
    time: f64,
    /// Cached basis state probabilities
    basis_probabilities: Vec<f64>,
}

impl LatticeState {
    /// Create a new lattice state
    pub fn new(config: LatticeBridgeConfig) -> QuantumResult<Self> {
        let (x, y, z) = config.dimensions;
        let total = x * y * z;

        if total == 0 {
            return Err(QuantumError::invalid_state("Lattice must have at least 1 node"));
        }
        if total > 32 {
            return Err(QuantumError::invalid_state("Maximum 32 qubits supported"));
        }

        // Create nodes
        let mut nodes = Vec::with_capacity(total);
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let qubit_idx = i * y * z + j * z + k;
                    nodes.push(LatticeNode::new((i, j, k), qubit_idx));
                }
            }
        }

        let mut state = Self {
            config,
            nodes,
            couplings: HashMap::new(),
            time: 0.0,
            basis_probabilities: vec![0.0; 1 << total],
        };

        // Initialize nearest-neighbor couplings
        state.initialize_couplings();

        // Set initial |0...0⟩ state
        state.basis_probabilities[0] = 1.0;

        Ok(state)
    }

    /// Create for given number of qubits
    pub fn for_qubits(num_qubits: usize) -> QuantumResult<Self> {
        Self::new(LatticeBridgeConfig::for_qubits(num_qubits))
    }

    /// Initialize nearest-neighbor couplings
    fn initialize_couplings(&mut self) {
        let j = self.config.coupling_strength;

        for node in &self.nodes {
            let neighbors = self.get_neighbors(node.position);
            for neighbor_pos in neighbors {
                let key = (node.position, neighbor_pos);
                if !self.couplings.contains_key(&key) {
                    self.couplings.insert(
                        key,
                        LatticeCoupling {
                            from: node.position,
                            to: neighbor_pos,
                            weight: j,
                            trace: 0.0,
                        },
                    );
                }
            }
        }
    }

    /// Get neighboring positions
    fn get_neighbors(&self, pos: (usize, usize, usize)) -> Vec<(usize, usize, usize)> {
        let (x, y, z) = self.config.dimensions;
        let (i, j, k) = pos;
        let mut neighbors = Vec::new();

        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0),
            (1, 0, 0),
            (0, -1, 0),
            (0, 1, 0),
            (0, 0, -1),
            (0, 0, 1),
        ];

        for (di, dj, dk) in deltas {
            let ni = if self.config.periodic_boundary {
                ((i as i32 + di).rem_euclid(x as i32)) as usize
            } else {
                let n = i as i32 + di;
                if n < 0 || n >= x as i32 {
                    continue;
                }
                n as usize
            };

            let nj = if self.config.periodic_boundary {
                ((j as i32 + dj).rem_euclid(y as i32)) as usize
            } else {
                let n = j as i32 + dj;
                if n < 0 || n >= y as i32 {
                    continue;
                }
                n as usize
            };

            let nk = if self.config.periodic_boundary {
                ((k as i32 + dk).rem_euclid(z as i32)) as usize
            } else {
                let n = k as i32 + dk;
                if n < 0 || n >= z as i32 {
                    continue;
                }
                n as usize
            };

            neighbors.push((ni, nj, nk));
        }

        neighbors
    }

    /// Compute local field at a node
    fn local_field(&self, node_idx: usize) -> f64 {
        let node = &self.nodes[node_idx];
        let mut field = self.config.external_field + node.bias;

        // Sum contributions from neighbors
        for neighbor_pos in self.get_neighbors(node.position) {
            let key = (node.position, neighbor_pos);
            if let Some(coupling) = self.couplings.get(&key) {
                if let Some(neighbor) = self.nodes.iter().find(|n| n.position == neighbor_pos) {
                    field += coupling.weight * neighbor.spin;
                }
            }
        }

        field
    }

    /// Perform one Monte Carlo sweep
    pub fn sweep(&mut self) {
        for i in 0..self.nodes.len() {
            let field = self.local_field(i);
            self.nodes[i].update_probability(field, self.config.temperature);
            self.nodes[i].sample();
            self.nodes[i].energy = -field * self.nodes[i].spin;
        }
        self.time += 1.0;
        self.update_basis_probabilities();
    }

    /// Equilibrate the lattice
    pub fn equilibrate(&mut self) {
        for _ in 0..self.config.equilibration_sweeps {
            self.sweep();
        }
    }

    /// Anneal to lower temperature
    pub fn anneal(&mut self) {
        let initial_temp = self.config.temperature;
        let target = self.config.target_temperature;
        let steps = self.config.annealing_steps;
        let temp_step = (initial_temp - target) / steps as f64;

        for _ in 0..steps {
            self.config.temperature -= temp_step;
            self.config.temperature = self.config.temperature.max(target);
            self.sweep();
        }
    }

    /// Apply STDP learning
    pub fn apply_stdp(&mut self, pre_idx: usize, post_idx: usize, post_spike_time: f64) {
        let pre_pos = self.nodes[pre_idx].position;
        let post_pos = self.nodes[post_idx].position;
        let key = (pre_pos, post_pos);

        if let Some(coupling) = self.couplings.get_mut(&key) {
            let pre_spike_time = self.nodes[pre_idx].last_spike_time;
            let dt = post_spike_time - pre_spike_time;
            let tau = self.config.stdp_window;

            let dw = if dt > 0.0 {
                self.config.stdp_learning_rate * (-dt / tau).exp()
            } else {
                -self.config.stdp_learning_rate * 0.5 * (dt / tau).exp()
            };

            coupling.weight += dw;
            coupling.weight = coupling.weight.clamp(-10.0, 10.0);
        }
    }

    /// Update basis state probabilities from node states
    fn update_basis_probabilities(&mut self) {
        let num_qubits = self.nodes.len();
        let num_states = 1 << num_qubits;

        // Use Boltzmann distribution
        let mut energies = vec![0.0; num_states];

        for state_idx in 0..num_states {
            let mut energy = 0.0;

            // Single-node contributions
            for node in &self.nodes {
                let spin = if (state_idx >> node.qubit_index) & 1 == 1 {
                    1.0
                } else {
                    -1.0
                };
                energy -= (self.config.external_field + node.bias) * spin;
            }

            // Coupling contributions
            for ((from, to), coupling) in &self.couplings {
                let from_idx = self
                    .nodes
                    .iter()
                    .find(|n| n.position == *from)
                    .map(|n| n.qubit_index)
                    .unwrap_or(0);
                let to_idx = self
                    .nodes
                    .iter()
                    .find(|n| n.position == *to)
                    .map(|n| n.qubit_index)
                    .unwrap_or(0);

                let spin_from = if (state_idx >> from_idx) & 1 == 1 { 1.0 } else { -1.0 };
                let spin_to = if (state_idx >> to_idx) & 1 == 1 { 1.0 } else { -1.0 };

                energy -= coupling.weight * spin_from * spin_to;
            }

            energies[state_idx] = energy;
        }

        // Boltzmann probabilities
        let min_energy = energies.iter().cloned().fold(f64::INFINITY, f64::min);
        let mut partition = 0.0;

        for (i, &e) in energies.iter().enumerate() {
            let boltzmann = (-(e - min_energy) / self.config.temperature).exp();
            self.basis_probabilities[i] = boltzmann;
            partition += boltzmann;
        }

        for p in &mut self.basis_probabilities {
            *p /= partition;
        }
    }

    /// Convert from PBitState
    pub fn from_pbit_state(pbit: &PBitState) -> QuantumResult<Self> {
        let config = LatticeBridgeConfig::for_qubits(pbit.num_qubits());
        let mut state = Self::new(config)?;

        // Copy probabilities
        state.basis_probabilities = pbit.probability_distribution();

        // Update node states from marginal probabilities
        for node in &mut state.nodes {
            if let Some(pbit_node) = pbit.get_pbit(node.qubit_index) {
                node.spin = pbit_node.spin;
                node.probability_up = pbit_node.probability_up;
                node.bias = pbit_node.bias;
            }
        }

        Ok(state)
    }

    /// Convert to PBitState
    pub fn to_pbit_state(&self) -> QuantumResult<PBitState> {
        let config = PBitConfig {
            temperature: self.config.temperature,
            coupling_strength: self.config.coupling_strength,
            external_field: self.config.external_field,
            seed: None,
        };

        let mut pbit = PBitState::with_config(self.nodes.len(), config)?;

        // Copy node states
        for node in &self.nodes {
            if let Some(pbit_node) = pbit.get_pbit_mut(node.qubit_index) {
                pbit_node.spin = node.spin;
                pbit_node.probability_up = node.probability_up;
                pbit_node.bias = node.bias;
            }
        }

        // Copy couplings
        for ((from, to), coupling) in &self.couplings {
            let from_idx = self
                .nodes
                .iter()
                .find(|n| n.position == *from)
                .map(|n| n.qubit_index);
            let to_idx = self
                .nodes
                .iter()
                .find(|n| n.position == *to)
                .map(|n| n.qubit_index);

            if let (Some(i), Some(j)) = (from_idx, to_idx) {
                pbit.add_coupling(if coupling.weight > 0.0 {
                    PBitCoupling::bell_coupling(i, j, coupling.weight.abs())
                } else {
                    PBitCoupling::anti_bell_coupling(i, j, coupling.weight.abs())
                });
            }
        }

        Ok(pbit)
    }

    /// Convert from QuantumState
    pub fn from_quantum_state(qs: &QuantumState) -> QuantumResult<Self> {
        let pbit = PBitState::from_quantum_state(qs)?;
        Self::from_pbit_state(&pbit)
    }

    /// Convert to QuantumState
    pub fn to_quantum_state(&self) -> QuantumResult<QuantumState> {
        let pbit = self.to_pbit_state()?;
        pbit.to_quantum_state()
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.nodes.len()
    }

    /// Get lattice dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.config.dimensions
    }

    /// Get probability distribution
    pub fn probability_distribution(&self) -> Vec<f64> {
        self.basis_probabilities.clone()
    }

    /// Get total energy
    pub fn total_energy(&self) -> f64 {
        self.nodes.iter().map(|n| n.energy).sum()
    }

    /// Get total entropy
    pub fn total_entropy(&self) -> f64 {
        self.nodes.iter().map(|n| n.entropy()).sum()
    }

    /// Get magnetization
    pub fn magnetization(&self) -> f64 {
        let sum: f64 = self.nodes.iter().map(|n| n.spin).sum();
        sum / self.nodes.len() as f64
    }

    /// Get correlation length (average nearest-neighbor correlation)
    pub fn correlation_length(&self) -> f64 {
        let mut correlation_sum = 0.0;
        let mut count = 0;

        for node in &self.nodes {
            for neighbor_pos in self.get_neighbors(node.position) {
                if let Some(neighbor) = self.nodes.iter().find(|n| n.position == neighbor_pos) {
                    correlation_sum += node.spin * neighbor.spin;
                    count += 1;
                }
            }
        }

        if count > 0 {
            correlation_sum / count as f64
        } else {
            0.0
        }
    }

    /// Measure all qubits
    pub fn measure(&mut self) -> usize {
        let random_value: f64 = rand::random();
        let mut cumulative = 0.0;

        for (state_idx, &prob) in self.basis_probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                // Collapse
                self.basis_probabilities.fill(0.0);
                self.basis_probabilities[state_idx] = 1.0;

                // Update nodes
                for node in &mut self.nodes {
                    let bit = (state_idx >> node.qubit_index) & 1;
                    node.spin = if bit == 1 { 1.0 } else { -1.0 };
                    node.probability_up = bit as f64;
                }

                return state_idx;
            }
        }

        self.basis_probabilities.len() - 1
    }
}

/// Trait for bridging quantum types with lattice
pub trait LatticeCompatible {
    /// Convert to lattice state
    fn to_lattice(&self) -> QuantumResult<LatticeState>;

    /// Create from lattice state
    fn from_lattice(lattice: &LatticeState) -> QuantumResult<Self>
    where
        Self: Sized;
}

impl LatticeCompatible for QuantumState {
    fn to_lattice(&self) -> QuantumResult<LatticeState> {
        LatticeState::from_quantum_state(self)
    }

    fn from_lattice(lattice: &LatticeState) -> QuantumResult<Self> {
        lattice.to_quantum_state()
    }
}

impl LatticeCompatible for PBitState {
    fn to_lattice(&self) -> QuantumResult<LatticeState> {
        LatticeState::from_pbit_state(self)
    }

    fn from_lattice(lattice: &LatticeState) -> QuantumResult<Self> {
        lattice.to_pbit_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_creation() {
        let state = LatticeState::for_qubits(4).unwrap();
        assert_eq!(state.num_qubits(), 4);
        assert_eq!(state.dimensions(), (2, 2, 1));
    }

    #[test]
    fn test_auto_dimensions() {
        assert_eq!(LatticeBridgeConfig::auto_dimensions(4), (2, 2, 1));
        assert_eq!(LatticeBridgeConfig::auto_dimensions(9), (3, 3, 1));
        assert_eq!(LatticeBridgeConfig::auto_dimensions(6), (3, 2, 1));
    }

    #[test]
    fn test_sweep() {
        let mut state = LatticeState::for_qubits(4).unwrap();
        state.sweep();
        assert!(state.time > 0.0);
    }

    #[test]
    fn test_annealing() {
        let mut state = LatticeState::for_qubits(4).unwrap();
        state.config.temperature = 10.0;
        state.anneal();
        assert!(state.config.temperature < 1.0);
    }

    #[test]
    fn test_pbit_conversion() {
        let pbit = PBitState::superposition(4).unwrap();
        let lattice = LatticeState::from_pbit_state(&pbit).unwrap();
        let pbit_back = lattice.to_pbit_state().unwrap();

        assert_eq!(pbit.num_qubits(), pbit_back.num_qubits());
    }

    #[test]
    fn test_quantum_conversion() {
        let qs = QuantumState::superposition(3).unwrap();
        let lattice = LatticeState::from_quantum_state(&qs).unwrap();
        let qs_back = lattice.to_quantum_state().unwrap();

        assert_eq!(qs.num_qubits(), qs_back.num_qubits());
    }
}
