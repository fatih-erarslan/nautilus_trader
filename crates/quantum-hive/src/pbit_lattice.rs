//! pBit Hyperbolic Lattice for Trading Hive
//!
//! Implements an Ising-model pBit lattice on hyperbolic geometry
//! for distributed trading coordination.
//!
//! ## Mathematical Foundation (Wolfram Validated)
//!
//! - **Hyperbolic Distance**: d(x,y) = acosh(1 + 2|x-y|²/((1-|x|²)(1-|y|²)))
//! - **pBit Coupling**: J_ij = J₀ × exp(-d_ij / ξ)
//! - **Boltzmann Weight**: W(E) = exp(-E/T)
//! - **Area Growth**: A(r) ~ 4π sinh²(r/2) (exponential)
//!
//! Validated constants:
//! - d([0,0], [0.5,0]) = 1.0986 (hyperbolic distance)
//! - J(d=1, J₀=1, ξ=1) = 0.3679 (coupling decay)
//! - W(E=-7, T=1) = 1096.6 (aligned spins)

use std::collections::HashMap;

/// Hyperbolic coordinates in Poincaré disk model
#[derive(Debug, Clone, Copy)]
pub struct HyperbolicPoint {
    pub x: f64,
    pub y: f64,
}

impl HyperbolicPoint {
    pub fn new(x: f64, y: f64) -> Self {
        // Clamp to unit disk
        let r = (x * x + y * y).sqrt();
        if r >= 1.0 {
            let scale = 0.99 / r;
            Self { x: x * scale, y: y * scale }
        } else {
            Self { x, y }
        }
    }

    /// Hyperbolic distance (Wolfram validated)
    /// d(x,y) = acosh(1 + 2|x-y|²/((1-|x|²)(1-|y|²)))
    pub fn distance(&self, other: &HyperbolicPoint) -> f64 {
        let norm_self_sq = self.x * self.x + self.y * self.y;
        let norm_other_sq = other.x * other.x + other.y * other.y;
        let diff_sq = (self.x - other.x).powi(2) + (self.y - other.y).powi(2);

        let denom = (1.0 - norm_self_sq) * (1.0 - norm_other_sq);
        if denom <= 0.0 {
            return f64::INFINITY;
        }

        let arg = 1.0 + 2.0 * diff_sq / denom;
        if arg <= 1.0 {
            return 0.0;
        }
        arg.acosh()
    }

    /// Polar coordinates
    pub fn to_polar(&self) -> (f64, f64) {
        let r = (self.x * self.x + self.y * self.y).sqrt();
        let theta = self.y.atan2(self.x);
        (r, theta)
    }

    /// From polar coordinates
    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self::new(r * theta.cos(), r * theta.sin())
    }
}

/// pBit node in the hyperbolic lattice
#[derive(Debug, Clone)]
pub struct PBitNode {
    pub id: usize,
    pub position: HyperbolicPoint,
    pub spin: i8,  // +1 or -1
    pub probability_up: f64,
    pub neighbors: Vec<usize>,
    pub couplings: HashMap<usize, f64>,
    pub local_field: f64,
    pub signal_strength: f64,
}

impl PBitNode {
    pub fn new(id: usize, position: HyperbolicPoint) -> Self {
        Self {
            id,
            position,
            spin: if rand::random() { 1 } else { -1 },
            probability_up: 0.5,
            neighbors: Vec::new(),
            couplings: HashMap::new(),
            local_field: 0.0,
            signal_strength: 0.0,
        }
    }
}

/// pBit Hyperbolic Lattice configuration
#[derive(Debug, Clone)]
pub struct PBitLatticeConfig {
    /// Number of nodes
    pub n_nodes: usize,
    /// Base coupling strength J₀
    pub j0: f64,
    /// Correlation length ξ
    pub xi: f64,
    /// Temperature
    pub temperature: f64,
    /// Number of neighbors per node (7 for heptagonal tiling)
    pub coordination: usize,
}

impl Default for PBitLatticeConfig {
    fn default() -> Self {
        Self {
            n_nodes: 100,
            j0: 1.0,
            xi: 1.0,
            temperature: 1.0,
            coordination: 7,  // Hyperbolic heptagonal tiling
        }
    }
}

/// pBit Hyperbolic Lattice
pub struct PBitLattice {
    pub config: PBitLatticeConfig,
    pub nodes: Vec<PBitNode>,
    pub total_energy: f64,
    pub magnetization: f64,
}

impl PBitLattice {
    /// Create new pBit lattice with hyperbolic geometry
    pub fn new(config: PBitLatticeConfig) -> Self {
        let mut nodes = Vec::with_capacity(config.n_nodes);

        // Generate nodes in hyperbolic disk using exponential distribution
        for i in 0..config.n_nodes {
            // Distribute radially with exponential density (hyperbolic area growth)
            let r = (i as f64 / config.n_nodes as f64).sqrt() * 0.9;
            let theta = i as f64 * 2.399963; // Golden angle for uniform distribution
            let position = HyperbolicPoint::from_polar(r, theta);
            nodes.push(PBitNode::new(i, position));
        }

        let mut lattice = Self {
            config,
            nodes,
            total_energy: 0.0,
            magnetization: 0.0,
        };

        lattice.build_connections();
        lattice.compute_energy();
        lattice
    }

    /// Build neighbor connections based on hyperbolic distance
    fn build_connections(&mut self) {
        let n = self.nodes.len();
        let coord = self.config.coordination;

        for i in 0..n {
            let pos_i = self.nodes[i].position;

            // Find k nearest neighbors
            let mut distances: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, pos_i.distance(&self.nodes[j].position)))
                .collect();

            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            let neighbors: Vec<usize> = distances.iter().take(coord).map(|(j, _)| *j).collect();
            
            // Compute couplings (Wolfram validated: J = J₀ exp(-d/ξ))
            let mut couplings = HashMap::new();
            for &j in &neighbors {
                let d = pos_i.distance(&self.nodes[j].position);
                let j_ij = self.config.j0 * (-d / self.config.xi).exp();
                couplings.insert(j, j_ij);
            }

            self.nodes[i].neighbors = neighbors;
            self.nodes[i].couplings = couplings;
        }
    }

    /// Compute total lattice energy
    /// H = -Σ J_ij s_i s_j
    fn compute_energy(&mut self) {
        let mut energy = 0.0;
        
        for node in &self.nodes {
            for (&j, &coupling) in &node.couplings {
                let s_i = node.spin as f64;
                let s_j = self.nodes[j].spin as f64;
                energy -= coupling * s_i * s_j;
            }
        }
        
        // Divide by 2 (each pair counted twice)
        self.total_energy = energy / 2.0;
        
        // Compute magnetization
        self.magnetization = self.nodes.iter().map(|n| n.spin as f64).sum::<f64>() / self.nodes.len() as f64;
    }

    /// Single Metropolis update step
    pub fn update_step(&mut self) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let i = rng.gen_range(0..self.nodes.len());

        // Compute local field h_i = Σ_j J_ij s_j
        let mut local_field = 0.0;
        for (&j, &coupling) in &self.nodes[i].couplings {
            local_field += coupling * self.nodes[j].spin as f64;
        }

        // Energy change for flipping: ΔE = 2 s_i h_i
        let s_i = self.nodes[i].spin as f64;
        let delta_e = 2.0 * s_i * local_field;

        // Metropolis acceptance
        let accept = if delta_e <= 0.0 {
            true
        } else {
            let p = (-delta_e / self.config.temperature).exp();
            rng.gen::<f64>() < p
        };

        if accept {
            self.nodes[i].spin *= -1;
            self.total_energy += delta_e;
        }

        // Update probability
        self.nodes[i].local_field = local_field;
        self.nodes[i].probability_up = 1.0 / (1.0 + (-local_field / self.config.temperature).exp());
    }

    /// Run thermalization
    pub fn thermalize(&mut self, steps: usize) {
        for _ in 0..steps {
            self.update_step();
        }
        self.compute_energy();
    }

    /// Get trading signal from local magnetization
    pub fn get_trading_signal(&self, node_id: usize) -> f64 {
        if node_id >= self.nodes.len() {
            return 0.0;
        }

        // Local magnetization from node + neighbors
        let node = &self.nodes[node_id];
        let mut local_mag = node.spin as f64;
        let mut count = 1.0;

        for &j in &node.neighbors {
            local_mag += self.nodes[j].spin as f64;
            count += 1.0;
        }

        local_mag / count  // Range: [-1, 1]
    }

    /// Get consensus signal (global magnetization)
    pub fn get_consensus_signal(&self) -> f64 {
        self.magnetization
    }

    /// Set external signal (market data) at a node
    pub fn set_market_signal(&mut self, node_id: usize, signal: f64) {
        if node_id < self.nodes.len() {
            self.nodes[node_id].signal_strength = signal;
            // Apply as external field bias
            self.nodes[node_id].local_field += signal * self.config.j0;
        }
    }

    /// Get lattice statistics
    pub fn get_stats(&self) -> LatticeStats {
        LatticeStats {
            n_nodes: self.nodes.len(),
            total_energy: self.total_energy,
            magnetization: self.magnetization,
            temperature: self.config.temperature,
            avg_coupling: self.nodes.iter()
                .flat_map(|n| n.couplings.values())
                .sum::<f64>() / self.nodes.iter().map(|n| n.couplings.len()).sum::<usize>() as f64,
        }
    }
}

/// Lattice statistics
#[derive(Debug, Clone)]
pub struct LatticeStats {
    pub n_nodes: usize,
    pub total_energy: f64,
    pub magnetization: f64,
    pub temperature: f64,
    pub avg_coupling: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hyperbolic_distance() {
        let p1 = HyperbolicPoint::new(0.0, 0.0);
        let p2 = HyperbolicPoint::new(0.5, 0.0);
        let d = p1.distance(&p2);
        // Wolfram validated: d([0,0], [0.5,0]) = 1.0986
        assert!((d - 1.0986).abs() < 0.001, "d = {}", d);
    }

    #[test]
    fn test_coupling_decay() {
        let config = PBitLatticeConfig::default();
        // J(d=1, J₀=1, ξ=1) = exp(-1) = 0.3679
        let j = config.j0 * (-1.0 / config.xi).exp();
        assert!((j - 0.3679).abs() < 0.001, "J = {}", j);
    }

    #[test]
    fn test_lattice_creation() {
        let config = PBitLatticeConfig { n_nodes: 50, ..Default::default() };
        let lattice = PBitLattice::new(config);
        
        assert_eq!(lattice.nodes.len(), 50);
        assert!(lattice.nodes[0].neighbors.len() == 7); // Heptagonal coordination
    }

    #[test]
    fn test_thermalization() {
        let mut lattice = PBitLattice::new(PBitLatticeConfig::default());
        let initial_energy = lattice.total_energy;
        
        lattice.thermalize(1000);
        
        // Energy should decrease (or stay similar) after thermalization
        assert!(lattice.total_energy <= initial_energy + 10.0);
    }

    #[test]
    fn test_trading_signal() {
        let mut lattice = PBitLattice::new(PBitLatticeConfig { n_nodes: 20, ..Default::default() });
        lattice.thermalize(100);
        
        let signal = lattice.get_trading_signal(0);
        assert!(signal >= -1.0 && signal <= 1.0);
    }
}
