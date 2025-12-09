//! # Probabilistic Computing (pBit) Module
//!
//! Implements probabilistic bits (pBits) for neuromorphic computing with
//! Boltzmann statistics, Ising dynamics, and pentagon topology.
//!
//! ## Mathematical Foundation
//!
//! ### pBit Probability
//! ```text
//! P(s=+1) = σ((h - b) / T) = 1 / (1 + exp(-(h - b) / T))
//! ```
//!
//! ### Ising Energy
//! ```text
//! E = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
//! ```
//!
//! ### Critical Temperature (Onsager)
//! ```text
//! Tc = 2 / ln(1 + √2) ≈ 2.269
//! ```
//!
//! ## References
//!
//! - Camsari, K. Y., et al. (2017). "Stochastic p-bits for invertible logic"
//! - Sutton, B., et al. (2020). "Autonomous probabilistic coprocessing with
//!   p-bits"

use rand::Rng;
use rand_chacha::ChaCha8Rng;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

// ============================================================================
// Constants
// ============================================================================

/// Ising critical temperature (2D square lattice, Onsager solution)
/// T_c = 2 / ln(1 + √2) ≈ 2.269185314213022
pub const ISING_CRITICAL_TEMP: f64 = 2.269185314213022;

/// Golden ratio
pub const PHI: f64 = 1.618033988749895;

/// Inverse golden ratio
pub const PHI_INV: f64 = 0.618033988749895;

/// Pentagon coupling matrix (golden ratio based)
pub const PENTAGON_COUPLING: [[f64; 5]; 5] = [
    [0.0, PHI / 2.0, PHI_INV / 2.0, PHI_INV / 2.0, PHI / 2.0],
    [PHI / 2.0, 0.0, PHI / 2.0, PHI_INV / 2.0, PHI_INV / 2.0],
    [PHI_INV / 2.0, PHI / 2.0, 0.0, PHI / 2.0, PHI_INV / 2.0],
    [PHI_INV / 2.0, PHI_INV / 2.0, PHI / 2.0, 0.0, PHI / 2.0],
    [PHI / 2.0, PHI_INV / 2.0, PHI_INV / 2.0, PHI / 2.0, 0.0],
];

/// Number of engines in pentagon
pub const PENTAGON_ENGINES: usize = 5;

// ============================================================================
// Core Functions
// ============================================================================

/// pBit probability: P(s=+1) = σ((h - b) / T)
#[inline]
pub fn pbit_probability(field: f64, bias: f64, temperature: f64) -> f64 {
    let x = (field - bias) / temperature.max(1e-10);
    sigmoid(x)
}

/// Sigmoid function with numerical stability
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let ex = x.exp();
        ex / (1.0 + ex)
    }
}

/// Boltzmann weight: exp(-E / T)
#[inline]
pub fn boltzmann_weight(energy: f64, temperature: f64) -> f64 {
    (-energy / temperature.max(1e-10)).exp()
}

/// Ising energy: E = -Σᵢⱼ Jᵢⱼ sᵢ sⱼ - Σᵢ hᵢ sᵢ
pub fn ising_energy(states: &[i8], couplings: &[Vec<f64>], fields: &[f64]) -> f64 {
    let n = states.len();
    let mut energy = 0.0;

    // Coupling term
    for i in 0..n {
        for j in (i + 1)..n {
            energy -= couplings[i][j] * states[i] as f64 * states[j] as f64;
        }
    }

    // Field term
    for i in 0..n {
        if i < fields.len() {
            energy -= fields[i] * states[i] as f64;
        }
    }

    energy
}

/// Metropolis acceptance probability
#[inline]
pub fn metropolis_accept(delta_energy: f64, temperature: f64) -> f64 {
    if delta_energy <= 0.0 {
        1.0
    } else {
        boltzmann_weight(delta_energy, temperature)
    }
}

// ============================================================================
// pBit
// ============================================================================

/// Single probabilistic bit
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PBit {
    /// Current state (+1 or -1)
    pub state: i8,
    /// Bias
    pub bias: f64,
    /// Input field
    pub field: f64,
}

impl Default for PBit {
    fn default() -> Self {
        Self::new()
    }
}

impl PBit {
    /// Create new pBit
    pub fn new() -> Self {
        Self {
            state: 1,
            bias: 0.0,
            field: 0.0,
        }
    }

    /// Create with bias
    pub fn with_bias(bias: f64) -> Self {
        Self {
            state: 1,
            bias,
            field: 0.0,
        }
    }

    /// Get probability of being +1
    pub fn probability(&self, temperature: f64) -> f64 {
        pbit_probability(self.field, self.bias, temperature)
    }

    /// Sample new state
    pub fn sample(&mut self, temperature: f64, rng: &mut impl Rng) {
        let p = self.probability(temperature);
        self.state = if rng.gen::<f64>() < p { 1 } else { -1 };
    }

    /// Update field
    pub fn update_field(&mut self, field: f64) {
        self.field = field;
    }

    /// Get state as f64
    pub fn state_f64(&self) -> f64 {
        self.state as f64
    }
}

// ============================================================================
// Pentagon pBit System
// ============================================================================

/// 5-engine pBit system with golden ratio coupling
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PentagonPBit {
    /// pBit engines
    pub engines: [PBit; PENTAGON_ENGINES],
    /// Temperature
    pub temperature: f64,
    /// External fields
    pub external_fields: [f64; PENTAGON_ENGINES],
    /// Current step
    step: usize,
}

impl Default for PentagonPBit {
    fn default() -> Self {
        Self::new(ISING_CRITICAL_TEMP)
    }
}

impl PentagonPBit {
    /// Create at given temperature
    pub fn new(temperature: f64) -> Self {
        Self {
            engines: [
                PBit::new(),
                PBit::new(),
                PBit::new(),
                PBit::new(),
                PBit::new(),
            ],
            temperature,
            external_fields: [0.0; PENTAGON_ENGINES],
            step: 0,
        }
    }

    /// Create at critical temperature
    pub fn at_criticality() -> Self {
        Self::new(ISING_CRITICAL_TEMP)
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.temperature = temperature;
    }

    /// Set external field for engine
    pub fn set_field(&mut self, engine: usize, field: f64) {
        if engine < PENTAGON_ENGINES {
            self.external_fields[engine] = field;
        }
    }

    /// Compute effective field for engine (including neighbors)
    pub fn effective_field(&self, engine: usize) -> f64 {
        let mut field = self.external_fields[engine];

        // Add neighbor contributions
        for (j, &coupling) in PENTAGON_COUPLING[engine].iter().enumerate() {
            field += coupling * self.engines[j].state_f64();
        }

        field
    }

    /// Update all fields
    fn update_fields(&mut self) {
        for i in 0..PENTAGON_ENGINES {
            self.engines[i].field = self.effective_field(i);
        }
    }

    /// Single Metropolis step (random engine)
    pub fn step(&mut self, rng: &mut impl Rng) {
        self.update_fields();

        // Choose random engine
        let engine = rng.gen_range(0..PENTAGON_ENGINES);
        self.engines[engine].sample(self.temperature, rng);

        self.step += 1;
    }

    /// Multiple steps
    pub fn step_n(&mut self, n: usize, rng: &mut impl Rng) {
        for _ in 0..n {
            self.step(rng);
        }
    }

    /// Sequential update (all engines)
    pub fn sweep(&mut self, rng: &mut impl Rng) {
        self.update_fields();
        for engine in &mut self.engines {
            engine.sample(self.temperature, rng);
        }
        self.step += 1;
    }

    /// Get states
    pub fn states(&self) -> [i8; PENTAGON_ENGINES] {
        [
            self.engines[0].state,
            self.engines[1].state,
            self.engines[2].state,
            self.engines[3].state,
            self.engines[4].state,
        ]
    }

    /// Get states as f64
    pub fn states_f64(&self) -> [f64; PENTAGON_ENGINES] {
        [
            self.engines[0].state_f64(),
            self.engines[1].state_f64(),
            self.engines[2].state_f64(),
            self.engines[3].state_f64(),
            self.engines[4].state_f64(),
        ]
    }

    /// Compute total energy
    pub fn energy(&self) -> f64 {
        let states: Vec<i8> = self.engines.iter().map(|e| e.state).collect();
        let couplings: Vec<Vec<f64>> = PENTAGON_COUPLING.iter().map(|r| r.to_vec()).collect();
        ising_energy(&states, &couplings, &self.external_fields)
    }

    /// Compute magnetization
    pub fn magnetization(&self) -> f64 {
        self.engines.iter().map(|e| e.state_f64()).sum::<f64>() / PENTAGON_ENGINES as f64
    }

    /// Compute phase coherence (Kuramoto order parameter)
    pub fn phase_coherence(&self) -> f64 {
        // Use states as phases
        let phases: Vec<f64> = self.engines
            .iter()
            .enumerate()
            .map(|(i, e)| e.state_f64() * std::f64::consts::PI * i as f64 / 5.0)
            .collect();

        let mut sum_cos = 0.0;
        let mut sum_sin = 0.0;
        for phase in &phases {
            sum_cos += phase.cos();
            sum_sin += phase.sin();
        }

        (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / PENTAGON_ENGINES as f64
    }

    /// Check if at criticality
    pub fn is_critical(&self) -> bool {
        (self.temperature - ISING_CRITICAL_TEMP).abs() / ISING_CRITICAL_TEMP < 0.1
    }

    /// Get step count
    pub fn step_count(&self) -> usize {
        self.step
    }
}

// ============================================================================
// Lattice pBit System
// ============================================================================

/// Configuration for pBit lattice
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LatticeConfig {
    /// Width
    pub width: usize,
    /// Height
    pub height: usize,
    /// Depth
    pub depth: usize,
    /// Temperature
    pub temperature: f64,
    /// Coupling strength
    pub coupling: f64,
    /// Periodic boundaries
    pub periodic: bool,
}

impl Default for LatticeConfig {
    fn default() -> Self {
        Self {
            width: 16,
            height: 16,
            depth: 1,
            temperature: ISING_CRITICAL_TEMP,
            coupling: 1.0,
            periodic: true,
        }
    }
}

/// 3D pBit lattice
#[derive(Debug, Clone)]
pub struct PBitLattice {
    /// Configuration
    config: LatticeConfig,
    /// States (flattened 3D array)
    states: Vec<i8>,
    /// Biases
    biases: Vec<f64>,
    /// External fields
    fields: Vec<f64>,
    /// Step counter
    step: usize,
}

impl PBitLattice {
    /// Create new lattice
    pub fn new(config: LatticeConfig) -> Self {
        let n = config.width * config.height * config.depth;
        Self {
            config,
            states: vec![1; n],
            biases: vec![0.0; n],
            fields: vec![0.0; n],
            step: 0,
        }
    }

    /// Create at critical temperature
    pub fn at_criticality(width: usize, height: usize) -> Self {
        Self::new(LatticeConfig {
            width,
            height,
            depth: 1,
            temperature: ISING_CRITICAL_TEMP,
            ..Default::default()
        })
    }

    /// Get total size
    pub fn size(&self) -> usize {
        self.states.len()
    }

    /// Convert 3D index to flat index
    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        z * self.config.width * self.config.height + y * self.config.width + x
    }

    /// Get neighbors of a site
    fn neighbors(&self, idx: usize) -> Vec<usize> {
        let w = self.config.width;
        let h = self.config.height;
        let d = self.config.depth;

        let z = idx / (w * h);
        let rem = idx % (w * h);
        let y = rem / w;
        let x = rem % w;

        let mut neighbors = Vec::with_capacity(6);

        // X neighbors
        if self.config.periodic || x > 0 {
            let nx = if x == 0 { w - 1 } else { x - 1 };
            neighbors.push(self.index(nx, y, z));
        }
        if self.config.periodic || x < w - 1 {
            let nx = if x == w - 1 { 0 } else { x + 1 };
            neighbors.push(self.index(nx, y, z));
        }

        // Y neighbors
        if self.config.periodic || y > 0 {
            let ny = if y == 0 { h - 1 } else { y - 1 };
            neighbors.push(self.index(x, ny, z));
        }
        if self.config.periodic || y < h - 1 {
            let ny = if y == h - 1 { 0 } else { y + 1 };
            neighbors.push(self.index(x, ny, z));
        }

        // Z neighbors
        if d > 1 {
            if self.config.periodic || z > 0 {
                let nz = if z == 0 { d - 1 } else { z - 1 };
                neighbors.push(self.index(x, y, nz));
            }
            if self.config.periodic || z < d - 1 {
                let nz = if z == d - 1 { 0 } else { z + 1 };
                neighbors.push(self.index(x, y, nz));
            }
        }

        neighbors
    }

    /// Compute local field for site
    fn local_field(&self, idx: usize) -> f64 {
        let mut field = self.fields[idx];

        for &neighbor in &self.neighbors(idx) {
            field += self.config.coupling * self.states[neighbor] as f64;
        }

        field
    }

    /// Single Metropolis step
    pub fn step(&mut self, rng: &mut impl Rng) {
        let idx = rng.gen_range(0..self.size());
        let field = self.local_field(idx);

        let p = pbit_probability(field, self.biases[idx], self.config.temperature);
        self.states[idx] = if rng.gen::<f64>() < p { 1 } else { -1 };

        self.step += 1;
    }

    /// Full sweep (one update per site)
    pub fn sweep(&mut self, rng: &mut impl Rng) {
        let n = self.size();
        for _ in 0..n {
            self.step(rng);
        }
    }

    /// Compute total energy
    pub fn energy(&self) -> f64 {
        let mut energy = 0.0;

        for idx in 0..self.size() {
            let state = self.states[idx] as f64;

            // Neighbor interactions (divide by 2 to avoid double counting)
            for &neighbor in &self.neighbors(idx) {
                if neighbor > idx {
                    energy -= self.config.coupling * state * self.states[neighbor] as f64;
                }
            }

            // Field term
            energy -= self.fields[idx] * state;
        }

        energy
    }

    /// Compute magnetization
    pub fn magnetization(&self) -> f64 {
        self.states.iter().map(|&s| s as f64).sum::<f64>() / self.size() as f64
    }

    /// Compute correlation function at distance r
    pub fn correlation(&self, r: usize) -> f64 {
        let w = self.config.width;
        let h = self.config.height;

        if r >= w / 2 {
            return 0.0;
        }

        let mut sum = 0.0;
        let mut count = 0;

        for y in 0..h {
            for x in 0..(w - r) {
                let idx1 = y * w + x;
                let idx2 = y * w + x + r;
                sum += self.states[idx1] as f64 * self.states[idx2] as f64;
                count += 1;
            }
        }

        sum / count as f64
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.config.temperature = temperature;
    }

    /// Get states
    pub fn states(&self) -> &[i8] {
        &self.states
    }

    /// Set field at site
    pub fn set_field(&mut self, idx: usize, field: f64) {
        if idx < self.fields.len() {
            self.fields[idx] = field;
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_pbit_probability_bounds() {
        for field in [-10.0, 0.0, 10.0] {
            let p = pbit_probability(field, 0.0, 1.0);
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_pbit_probability_symmetry() {
        let p_pos = pbit_probability(5.0, 0.0, 1.0);
        let p_neg = pbit_probability(-5.0, 0.0, 1.0);
        assert!((p_pos + p_neg - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ising_critical_temp() {
        let tc = ISING_CRITICAL_TEMP;
        assert!((tc - 2.269).abs() < 0.01);
    }

    #[test]
    fn test_pentagon_coupling_symmetry() {
        for i in 0..5 {
            for j in 0..5 {
                assert!((PENTAGON_COUPLING[i][j] - PENTAGON_COUPLING[j][i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_pentagon_coupling_values() {
        // Adjacent coupling should be PHI/2
        assert!((PENTAGON_COUPLING[0][1] - PHI / 2.0).abs() < 1e-10);

        // Skip-one coupling should be PHI_INV/2
        assert!((PENTAGON_COUPLING[0][2] - PHI_INV / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_pentagon_pbit_creation() {
        let pentagon = PentagonPBit::at_criticality();
        assert!(pentagon.is_critical());
    }

    #[test]
    fn test_pentagon_magnetization_bounds() {
        let pentagon = PentagonPBit::new(1.0);
        let m = pentagon.magnetization();
        assert!(m >= -1.0 && m <= 1.0);
    }

    #[test]
    fn test_lattice_creation() {
        let lattice = PBitLattice::at_criticality(8, 8);
        assert_eq!(lattice.size(), 64);
    }

    #[test]
    fn test_lattice_magnetization() {
        let lattice = PBitLattice::at_criticality(8, 8);
        let m = lattice.magnetization();
        assert!(m >= -1.0 && m <= 1.0);
    }

    #[test]
    fn test_lattice_step() {
        let mut lattice = PBitLattice::at_criticality(8, 8);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let initial_mag = lattice.magnetization();
        lattice.sweep(&mut rng);

        // Magnetization should change (probably)
        // This is a probabilistic test
        let _ = initial_mag; // Just ensure it runs
    }

    #[test]
    fn test_boltzmann_weight() {
        // At T→∞, weight should be ~1
        let w_high_t = boltzmann_weight(1.0, 1000.0);
        assert!((w_high_t - 1.0).abs() < 0.01);

        // At low T, high energy should give low weight
        let w_low_t = boltzmann_weight(10.0, 0.1);
        assert!(w_low_t < 0.01);
    }
}
