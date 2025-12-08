//! pBit SpatioTemporal Lattice
//!
//! Implements a probabilistic bit (pBit) lattice that serves as the
//! computational fabric for swarm intelligence. Each node is a probabilistic
//! computing element that can represent superpositions of states.
//!
//! ## Theoretical Foundation
//!
//! Based on:
//! - Ising model for spin interactions
//! - Boltzmann machines for probabilistic inference
//! - Hopfield networks for associative memory
//! - STDP (Spike-Timing Dependent Plasticity) for learning

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use nalgebra::DVector;
use uuid::Uuid;

use crate::SwarmResult;

/// Configuration for the pBit lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeConfig {
    /// Lattice dimensions (x, y, z for 3D, or (x, y, 1) for 2D)
    pub dimensions: (usize, usize, usize),
    /// Temperature for Boltzmann sampling
    pub temperature: f64,
    /// Coupling strength between adjacent nodes
    pub coupling_strength: f64,
    /// External field strength
    pub external_field: f64,
    /// STDP learning rate
    pub stdp_learning_rate: f64,
    /// Temporal window for STDP (milliseconds)
    pub stdp_window: f64,
    /// Whether to use periodic boundary conditions
    pub periodic_boundary: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            dimensions: (16, 16, 4),
            temperature: 1.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            stdp_learning_rate: 0.01,
            stdp_window: 20.0,
            periodic_boundary: true,
            seed: None,
        }
    }
}

/// A single pBit node in the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeNode {
    /// Unique node identifier
    pub id: Uuid,
    /// Position in lattice (x, y, z)
    pub position: (usize, usize, usize),
    /// Current spin state (-1 or +1, but stored as f64 for continuous ops)
    pub spin: f64,
    /// Probability of spin-up state
    pub probability_up: f64,
    /// Local energy
    pub energy: f64,
    /// Activation level (for SNN integration)
    pub activation: f64,
    /// Last spike time (for STDP)
    pub last_spike_time: f64,
    /// Accumulated potential
    pub potential: f64,
    /// Refractory period remaining
    pub refractory: f64,
    /// Node-specific bias
    pub bias: f64,
    /// Information content (entropy)
    pub information: f64,
}

impl LatticeNode {
    /// Create a new lattice node
    pub fn new(position: (usize, usize, usize)) -> Self {
        Self {
            id: Uuid::new_v4(),
            position,
            spin: 1.0, // Start spin-up
            probability_up: 0.5,
            energy: 0.0,
            activation: 0.0,
            last_spike_time: f64::NEG_INFINITY,
            potential: 0.0,
            refractory: 0.0,
            bias: 0.0,
            information: 0.0,
        }
    }
    
    /// Update probability based on local field
    pub fn update_probability(&mut self, local_field: f64, temperature: f64) {
        // Boltzmann probability: P(up) = 1 / (1 + exp(-2 * h / T))
        let exponent = -2.0 * local_field / temperature;
        self.probability_up = 1.0 / (1.0 + exponent.exp());
    }
    
    /// Sample spin from probability
    pub fn sample_spin(&mut self, rng: &mut impl Rng) {
        self.spin = if rng.gen::<f64>() < self.probability_up { 1.0 } else { -1.0 };
    }
    
    /// Compute local entropy
    pub fn compute_entropy(&self) -> f64 {
        let p = self.probability_up.clamp(1e-10, 1.0 - 1e-10);
        -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
    }
}

/// Coupling between lattice nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coupling {
    /// Source node position
    pub from: (usize, usize, usize),
    /// Target node position
    pub to: (usize, usize, usize),
    /// Coupling weight (can be positive or negative)
    pub weight: f64,
    /// Learning eligibility trace
    pub trace: f64,
}

/// Complete state of the lattice at a point in time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatioTemporalState {
    /// Timestamp
    pub time: f64,
    /// All spin values flattened
    pub spins: Vec<f64>,
    /// All probabilities flattened
    pub probabilities: Vec<f64>,
    /// Total energy
    pub total_energy: f64,
    /// Total magnetization
    pub magnetization: f64,
    /// Total entropy
    pub entropy: f64,
    /// Correlation length
    pub correlation_length: f64,
}

/// The main pBit SpatioTemporal Lattice
#[derive(Debug)]
pub struct PBitLattice {
    /// Configuration
    config: LatticeConfig,
    /// Lattice nodes
    nodes: Vec<Vec<Vec<LatticeNode>>>,
    /// Coupling weights
    couplings: HashMap<((usize, usize, usize), (usize, usize, usize)), Coupling>,
    /// Random number generator
    rng: ChaCha8Rng,
    /// Current simulation time
    time: f64,
    /// State history for temporal processing
    history: Vec<SpatioTemporalState>,
    /// Maximum history length
    max_history: usize,
}

impl PBitLattice {
    /// Create a new pBit lattice
    pub fn new(config: LatticeConfig) -> SwarmResult<Self> {
        let (x, y, z) = config.dimensions;
        
        // Initialize nodes
        let mut nodes = Vec::with_capacity(x);
        for i in 0..x {
            let mut row = Vec::with_capacity(y);
            for j in 0..y {
                let mut col = Vec::with_capacity(z);
                for k in 0..z {
                    col.push(LatticeNode::new((i, j, k)));
                }
                row.push(col);
            }
            nodes.push(row);
        }
        
        // Initialize RNG
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };
        
        let mut lattice = Self {
            config,
            nodes,
            couplings: HashMap::new(),
            rng,
            time: 0.0,
            history: Vec::new(),
            max_history: 1000,
        };
        
        // Initialize couplings
        lattice.initialize_couplings();
        
        Ok(lattice)
    }
    
    /// Initialize nearest-neighbor couplings
    fn initialize_couplings(&mut self) {
        let (x, y, z) = self.config.dimensions;
        let j = self.config.coupling_strength;
        
        for i in 0..x {
            for jj in 0..y {
                for k in 0..z {
                    // Connect to neighbors
                    let neighbors = self.get_neighbors((i, jj, k));
                    for neighbor in neighbors {
                        let key = ((i, jj, k), neighbor);
                        if !self.couplings.contains_key(&key) {
                            self.couplings.insert(key, Coupling {
                                from: (i, jj, k),
                                to: neighbor,
                                weight: j,
                                trace: 0.0,
                            });
                        }
                    }
                }
            }
        }
    }
    
    /// Get neighboring positions
    fn get_neighbors(&self, pos: (usize, usize, usize)) -> Vec<(usize, usize, usize)> {
        let (x, y, z) = self.config.dimensions;
        let (i, j, k) = pos;
        let mut neighbors = Vec::new();
        
        // 6-connectivity for 3D lattice
        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];
        
        for (di, dj, dk) in deltas {
            let ni = if self.config.periodic_boundary {
                ((i as i32 + di).rem_euclid(x as i32)) as usize
            } else {
                let n = i as i32 + di;
                if n < 0 || n >= x as i32 { continue; }
                n as usize
            };
            
            let nj = if self.config.periodic_boundary {
                ((j as i32 + dj).rem_euclid(y as i32)) as usize
            } else {
                let n = j as i32 + dj;
                if n < 0 || n >= y as i32 { continue; }
                n as usize
            };
            
            let nk = if self.config.periodic_boundary {
                ((k as i32 + dk).rem_euclid(z as i32)) as usize
            } else {
                let n = k as i32 + dk;
                if n < 0 || n >= z as i32 { continue; }
                n as usize
            };
            
            neighbors.push((ni, nj, nk));
        }
        
        neighbors
    }
    
    /// Compute local field at a position
    fn local_field(&self, pos: (usize, usize, usize)) -> f64 {
        let mut field = self.config.external_field + self.nodes[pos.0][pos.1][pos.2].bias;
        
        // Sum contributions from neighbors
        for neighbor in self.get_neighbors(pos) {
            let key = (pos, neighbor);
            if let Some(coupling) = self.couplings.get(&key) {
                field += coupling.weight * self.nodes[neighbor.0][neighbor.1][neighbor.2].spin;
            }
        }
        
        field
    }
    
    /// Perform one Monte Carlo sweep
    pub fn sweep(&mut self) {
        let (x, y, z) = self.config.dimensions;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let local_field = self.local_field((i, j, k));
                    self.nodes[i][j][k].update_probability(local_field, self.config.temperature);
                    self.nodes[i][j][k].sample_spin(&mut self.rng);
                    
                    // Update energy
                    self.nodes[i][j][k].energy = -local_field * self.nodes[i][j][k].spin;
                    
                    // Update information content
                    self.nodes[i][j][k].information = self.nodes[i][j][k].compute_entropy();
                }
            }
        }
        
        self.time += 1.0;
    }
    
    /// Apply STDP learning between nodes
    pub fn apply_stdp(&mut self, pre_pos: (usize, usize, usize), post_pos: (usize, usize, usize), post_spike_time: f64) {
        let key = (pre_pos, post_pos);
        if let Some(coupling) = self.couplings.get_mut(&key) {
            let pre_spike_time = self.nodes[pre_pos.0][pre_pos.1][pre_pos.2].last_spike_time;
            let dt = post_spike_time - pre_spike_time;
            
            // STDP rule: potentiation if pre before post, depression if post before pre
            let tau = self.config.stdp_window;
            let dw = if dt > 0.0 {
                // Pre before post: potentiation
                self.config.stdp_learning_rate * (-dt / tau).exp()
            } else {
                // Post before pre: depression
                -self.config.stdp_learning_rate * 0.5 * (dt / tau).exp()
            };
            
            coupling.weight += dw;
            coupling.weight = coupling.weight.clamp(-10.0, 10.0);
        }
    }
    
    /// Inject a pattern into the lattice
    pub fn inject_pattern(&mut self, pattern: &DVector<f64>) {
        let (x, y, z) = self.config.dimensions;
        let total = x * y * z;
        
        for (idx, &value) in pattern.iter().enumerate().take(total) {
            let i = idx / (y * z);
            let j = (idx % (y * z)) / z;
            let k = idx % z;
            
            self.nodes[i][j][k].bias = value;
        }
    }
    
    /// Read current state as pattern
    pub fn read_pattern(&self) -> DVector<f64> {
        let (x, y, z) = self.config.dimensions;
        let total = x * y * z;
        let mut pattern = DVector::zeros(total);
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let idx = i * y * z + j * z + k;
                    pattern[idx] = self.nodes[i][j][k].spin;
                }
            }
        }
        
        pattern
    }
    
    /// Get current state snapshot
    pub fn get_state(&self) -> SpatioTemporalState {
        let (x, y, z) = self.config.dimensions;
        let mut spins = Vec::with_capacity(x * y * z);
        let mut probs = Vec::with_capacity(x * y * z);
        let mut total_energy = 0.0;
        let mut total_entropy = 0.0;
        let mut magnetization = 0.0;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let node = &self.nodes[i][j][k];
                    spins.push(node.spin);
                    probs.push(node.probability_up);
                    total_energy += node.energy;
                    total_entropy += node.information;
                    magnetization += node.spin;
                }
            }
        }
        
        let n = (x * y * z) as f64;
        magnetization /= n;
        
        SpatioTemporalState {
            time: self.time,
            spins,
            probabilities: probs,
            total_energy,
            magnetization,
            entropy: total_entropy,
            correlation_length: self.compute_correlation_length(),
        }
    }
    
    /// Compute correlation length
    fn compute_correlation_length(&self) -> f64 {
        // Simplified: compute average nearest-neighbor correlation
        let (x, y, z) = self.config.dimensions;
        let mut correlation_sum = 0.0;
        let mut count = 0;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let s1 = self.nodes[i][j][k].spin;
                    for neighbor in self.get_neighbors((i, j, k)) {
                        let s2 = self.nodes[neighbor.0][neighbor.1][neighbor.2].spin;
                        correlation_sum += s1 * s2;
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            correlation_sum / count as f64
        } else {
            0.0
        }
    }
    
    /// Record current state to history
    pub fn record_state(&mut self) {
        let state = self.get_state();
        self.history.push(state);
        
        // Trim history if too long
        while self.history.len() > self.max_history {
            self.history.remove(0);
        }
    }
    
    /// Get state history
    pub fn get_history(&self) -> &[SpatioTemporalState] {
        &self.history
    }
    
    /// Anneal the lattice (reduce temperature over time)
    pub fn anneal(&mut self, target_temperature: f64, steps: usize) {
        let initial_temp = self.config.temperature;
        let temp_step = (initial_temp - target_temperature) / steps as f64;
        
        for _ in 0..steps {
            self.config.temperature -= temp_step;
            self.config.temperature = self.config.temperature.max(target_temperature);
            self.sweep();
        }
    }
    
    /// Quench to ground state
    pub fn quench(&mut self, steps: usize) {
        self.anneal(0.01, steps);
    }
    
    /// Get node at position
    pub fn get_node(&self, pos: (usize, usize, usize)) -> Option<&LatticeNode> {
        let (x, y, z) = self.config.dimensions;
        if pos.0 < x && pos.1 < y && pos.2 < z {
            Some(&self.nodes[pos.0][pos.1][pos.2])
        } else {
            None
        }
    }
    
    /// Get mutable node at position
    pub fn get_node_mut(&mut self, pos: (usize, usize, usize)) -> Option<&mut LatticeNode> {
        let (x, y, z) = self.config.dimensions;
        if pos.0 < x && pos.1 < y && pos.2 < z {
            Some(&mut self.nodes[pos.0][pos.1][pos.2])
        } else {
            None
        }
    }
    
    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.config.dimensions
    }
    
    /// Get current time
    pub fn time(&self) -> f64 {
        self.time
    }
    
    /// Set temperature
    pub fn set_temperature(&mut self, temp: f64) {
        self.config.temperature = temp.max(0.001);
    }
    
    /// Get temperature
    pub fn temperature(&self) -> f64 {
        self.config.temperature
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_creation() {
        let config = LatticeConfig {
            dimensions: (8, 8, 2),
            ..Default::default()
        };
        let lattice = PBitLattice::new(config).unwrap();
        assert_eq!(lattice.dimensions(), (8, 8, 2));
    }
    
    #[test]
    fn test_sweep() {
        let config = LatticeConfig {
            dimensions: (4, 4, 1),
            temperature: 1.0,
            ..Default::default()
        };
        let mut lattice = PBitLattice::new(config).unwrap();
        
        for _ in 0..100 {
            lattice.sweep();
        }
        
        let state = lattice.get_state();
        assert!(state.time == 100.0);
    }
    
    #[test]
    fn test_annealing() {
        let config = LatticeConfig {
            dimensions: (8, 8, 1),
            temperature: 10.0,
            ..Default::default()
        };
        let mut lattice = PBitLattice::new(config).unwrap();
        
        lattice.anneal(0.1, 100);
        
        // After annealing, magnetization should be closer to ±1
        let state = lattice.get_state();
        assert!(state.magnetization.abs() > 0.5);
    }
}
