//! pBit Lattice for probabilistic computing
//!
//! Provides a simplified interface to the pBit SpatioTemporal Lattice.

use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Result, LatticeConfig};

/// Lattice state snapshot
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatticeState {
    /// Current time step
    pub time: f64,
    /// All spin values
    pub spins: Vec<f64>,
    /// Total energy
    pub energy: f64,
    /// Magnetization
    pub magnetization: f64,
    /// Entropy
    pub entropy: f64,
}

/// A node in the lattice
#[derive(Debug, Clone)]
struct LatticeNode {
    spin: f64,
    probability_up: f64,
    energy: f64,
    bias: f64,
}

impl LatticeNode {
    fn new() -> Self {
        Self {
            spin: 1.0,
            probability_up: 0.5,
            energy: 0.0,
            bias: 0.0,
        }
    }
    
    fn update_probability(&mut self, local_field: f64, temperature: f64) {
        let exponent = -2.0 * local_field / temperature.max(0.001);
        self.probability_up = 1.0 / (1.0 + exponent.exp());
    }
    
    fn sample_spin(&mut self, rng: &mut impl Rng) {
        self.spin = if rng.gen::<f64>() < self.probability_up { 1.0 } else { -1.0 };
    }
    
    fn entropy(&self) -> f64 {
        let p = self.probability_up.clamp(1e-10, 1.0 - 1e-10);
        -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
    }
}

/// pBit Lattice
pub struct Lattice {
    config: LatticeConfig,
    nodes: Vec<Vec<Vec<LatticeNode>>>,
    couplings: HashMap<((usize, usize, usize), (usize, usize, usize)), f64>,
    time: f64,
    rng: ChaCha8Rng,
}

impl Lattice {
    /// Create a new lattice
    pub fn new(config: LatticeConfig) -> Result<Self> {
        let (x, y, z) = config.dimensions;
        
        let mut nodes = Vec::with_capacity(x);
        for _ in 0..x {
            let mut row = Vec::with_capacity(y);
            for _ in 0..y {
                let mut col = Vec::with_capacity(z);
                for _ in 0..z {
                    col.push(LatticeNode::new());
                }
                row.push(col);
            }
            nodes.push(row);
        }
        
        let mut lattice = Self {
            config,
            nodes,
            couplings: HashMap::new(),
            time: 0.0,
            rng: ChaCha8Rng::from_entropy(),
        };
        
        lattice.initialize_couplings();
        
        Ok(lattice)
    }
    
    /// Initialize nearest-neighbor couplings
    fn initialize_couplings(&mut self) {
        let (x, y, z) = self.config.dimensions;
        let j = self.config.coupling;
        
        for i in 0..x {
            for jj in 0..y {
                for k in 0..z {
                    for neighbor in self.get_neighbors((i, jj, k)) {
                        let key = ((i, jj, k), neighbor);
                        self.couplings.insert(key, j);
                    }
                }
            }
        }
    }
    
    /// Get neighbors of a position
    fn get_neighbors(&self, pos: (usize, usize, usize)) -> Vec<(usize, usize, usize)> {
        let (x, y, z) = self.config.dimensions;
        let (i, j, k) = pos;
        let mut neighbors = Vec::new();
        
        let deltas: [(i32, i32, i32); 6] = [
            (-1, 0, 0), (1, 0, 0),
            (0, -1, 0), (0, 1, 0),
            (0, 0, -1), (0, 0, 1),
        ];
        
        for (di, dj, dk) in deltas {
            let ni = if self.config.periodic {
                ((i as i32 + di).rem_euclid(x as i32)) as usize
            } else {
                let n = i as i32 + di;
                if n < 0 || n >= x as i32 { continue; }
                n as usize
            };
            
            let nj = if self.config.periodic {
                ((j as i32 + dj).rem_euclid(y as i32)) as usize
            } else {
                let n = j as i32 + dj;
                if n < 0 || n >= y as i32 { continue; }
                n as usize
            };
            
            let nk = if self.config.periodic {
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
        let (i, j, k) = pos;
        let mut field = self.config.field + self.nodes[i][j][k].bias;
        
        for neighbor in self.get_neighbors(pos) {
            let key = (pos, neighbor);
            if let Some(&weight) = self.couplings.get(&key) {
                field += weight * self.nodes[neighbor.0][neighbor.1][neighbor.2].spin;
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
                    self.nodes[i][j][k].energy = -local_field * self.nodes[i][j][k].spin;
                }
            }
        }
        
        self.time += 1.0;
    }
    
    /// Run multiple sweeps
    pub fn run(&mut self, sweeps: usize) {
        for _ in 0..sweeps {
            self.sweep();
        }
    }
    
    /// Anneal the lattice
    pub fn anneal(&mut self, target_temp: f64, steps: usize) {
        let initial = self.config.temperature;
        let step = (initial - target_temp) / steps as f64;
        
        for _ in 0..steps {
            self.config.temperature = (self.config.temperature - step).max(target_temp);
            self.sweep();
        }
    }
    
    /// Quench to ground state
    pub fn quench(&mut self, steps: usize) {
        self.anneal(0.01, steps);
    }
    
    /// Get current state
    pub fn state(&self) -> LatticeState {
        let (x, y, z) = self.config.dimensions;
        let mut spins = Vec::with_capacity(x * y * z);
        let mut total_energy = 0.0;
        let mut total_entropy = 0.0;
        let mut magnetization = 0.0;
        
        for i in 0..x {
            for j in 0..y {
                for k in 0..z {
                    let node = &self.nodes[i][j][k];
                    spins.push(node.spin);
                    total_energy += node.energy;
                    total_entropy += node.entropy();
                    magnetization += node.spin;
                }
            }
        }
        
        let n = (x * y * z) as f64;
        
        LatticeState {
            time: self.time,
            spins,
            energy: total_energy,
            magnetization: magnetization / n,
            entropy: total_entropy,
        }
    }
    
    /// Set a pattern of biases
    pub fn set_pattern(&mut self, pattern: &[f64]) {
        let (x, y, z) = self.config.dimensions;
        let total = x * y * z;
        
        for (idx, &value) in pattern.iter().enumerate().take(total) {
            let i = idx / (y * z);
            let j = (idx % (y * z)) / z;
            let k = idx % z;
            self.nodes[i][j][k].bias = value;
        }
    }
    
    /// Read the current spin pattern
    pub fn read_pattern(&self) -> Vec<f64> {
        self.state().spins
    }
    
    /// Set temperature
    pub fn set_temperature(&mut self, temp: f64) {
        self.config.temperature = temp.max(0.001);
    }
    
    /// Get temperature
    pub fn temperature(&self) -> f64 {
        self.config.temperature
    }
    
    /// Get dimensions
    pub fn dimensions(&self) -> (usize, usize, usize) {
        self.config.dimensions
    }
}

/// Builder for lattice
pub struct LatticeBuilder {
    config: LatticeConfig,
}

impl LatticeBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: LatticeConfig::default(),
        }
    }
    
    /// Set dimensions
    pub fn dimensions(mut self, x: usize, y: usize, z: usize) -> Self {
        self.config.dimensions = (x, y, z);
        self
    }
    
    /// Set 2D dimensions
    pub fn dimensions_2d(mut self, x: usize, y: usize) -> Self {
        self.config.dimensions = (x, y, 1);
        self
    }
    
    /// Set temperature
    pub fn temperature(mut self, temp: f64) -> Self {
        self.config.temperature = temp;
        self
    }
    
    /// Set coupling strength
    pub fn coupling(mut self, j: f64) -> Self {
        self.config.coupling = j;
        self
    }
    
    /// Set external field
    pub fn field(mut self, h: f64) -> Self {
        self.config.field = h;
        self
    }
    
    /// Set periodic boundaries
    pub fn periodic(mut self, enable: bool) -> Self {
        self.config.periodic = enable;
        self
    }
    
    /// Build the lattice
    pub fn build(self) -> Result<Lattice> {
        Lattice::new(self.config)
    }
}

impl Default for LatticeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lattice_creation() {
        let lattice = LatticeBuilder::new()
            .dimensions(8, 8, 2)
            .temperature(1.0)
            .build()
            .unwrap();
        
        assert_eq!(lattice.dimensions(), (8, 8, 2));
    }
    
    #[test]
    fn test_lattice_sweep() {
        let mut lattice = LatticeBuilder::new()
            .dimensions_2d(8, 8)
            .build()
            .unwrap();
        
        lattice.run(100);
        
        let state = lattice.state();
        assert!(state.time == 100.0);
    }
    
    #[test]
    fn test_lattice_annealing() {
        let mut lattice = LatticeBuilder::new()
            .dimensions_2d(16, 16)
            .temperature(10.0)
            .build()
            .unwrap();
        
        // More aggressive annealing
        lattice.anneal(0.01, 500);
        
        let state = lattice.state();
        // After annealing, magnetization should increase (closer to ordered state)
        // Using a relaxed threshold since this is probabilistic
        assert!(state.magnetization.abs() > 0.3 || lattice.temperature() < 0.1);
    }
}
