//! pBit State Abstraction Layer
//!
//! Provides a probabilistic bit (pBit) state representation that bridges
//! quantum computing primitives with Ising-model-based probabilistic computing.
//!
//! ## Mapping: Quantum → pBit
//!
//! - |α|² + |β|² = 1 → P(↑) + P(↓) = 1
//! - Complex amplitudes → Real probabilities
//! - Quantum gates → Boltzmann sampling with local field updates
//! - Entanglement → Ferromagnetic coupling (positive J)
//! - Measurement → Spin sampling from P(↑)

use crate::error::{QuantumError, QuantumResult};
use crate::quantum_state::QuantumState;
use num_complex::Complex64;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for pBit state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitConfig {
    /// Temperature for Boltzmann sampling (higher = more random)
    pub temperature: f64,
    /// Coupling strength for entangled qubits
    pub coupling_strength: f64,
    /// External field (bias)
    pub external_field: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PBitConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            coupling_strength: 1.0,
            external_field: 0.0,
            seed: None,
        }
    }
}

/// A single pBit representing one qubit
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBit {
    /// Index in the state vector
    pub index: usize,
    /// Current spin value (-1.0 or +1.0)
    pub spin: f64,
    /// Probability of spin-up (maps to |1⟩)
    pub probability_up: f64,
    /// Local energy
    pub energy: f64,
    /// Bias field
    pub bias: f64,
}

impl PBit {
    /// Create a new pBit in spin-down state (|0⟩)
    pub fn new(index: usize) -> Self {
        Self {
            index,
            spin: -1.0, // Down = |0⟩
            probability_up: 0.0,
            energy: 0.0,
            bias: 0.0,
        }
    }

    /// Create a pBit in superposition (equal probability)
    pub fn superposition(index: usize) -> Self {
        Self {
            index,
            spin: 1.0, // Will be sampled
            probability_up: 0.5,
            energy: 0.0,
            bias: 0.0,
        }
    }

    /// Update probability from local field using Boltzmann distribution
    pub fn update_probability(&mut self, local_field: f64, temperature: f64) {
        // P(↑) = 1 / (1 + exp(-2h/T)) = σ(2h/T)
        let exponent = -2.0 * local_field / temperature.max(1e-10);
        self.probability_up = 1.0 / (1.0 + exponent.exp());
    }

    /// Sample spin from probability distribution
    pub fn sample(&mut self, rng: &mut impl Rng) {
        self.spin = if rng.gen::<f64>() < self.probability_up {
            1.0 // Up = |1⟩
        } else {
            -1.0 // Down = |0⟩
        };
    }

    /// Get measurement result (0 or 1)
    pub fn measure(&self) -> usize {
        if self.spin > 0.0 { 1 } else { 0 }
    }

    /// Compute Shannon entropy of this pBit
    pub fn entropy(&self) -> f64 {
        let p = self.probability_up.clamp(1e-10, 1.0 - 1e-10);
        -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
    }
}

/// Coupling between pBits (represents entanglement)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitCoupling {
    /// First pBit index
    pub i: usize,
    /// Second pBit index
    pub j: usize,
    /// Coupling weight (positive = ferromagnetic = entangled)
    pub weight: f64,
}

impl PBitCoupling {
    /// Create Bell-state-like coupling (strong positive)
    pub fn bell_coupling(i: usize, j: usize, strength: f64) -> Self {
        Self {
            i,
            j,
            weight: strength.abs(), // Ferromagnetic for |00⟩ + |11⟩
        }
    }

    /// Create anti-correlated coupling for |01⟩ + |10⟩
    pub fn anti_bell_coupling(i: usize, j: usize, strength: f64) -> Self {
        Self {
            i,
            j,
            weight: -strength.abs(), // Antiferromagnetic for |01⟩ + |10⟩
        }
    }
}

/// pBit State: Probabilistic representation of quantum state
#[derive(Debug, Clone)]
pub struct PBitState {
    /// Number of qubits (pBits)
    num_qubits: usize,
    /// Individual pBit states
    pbits: Vec<PBit>,
    /// Couplings between pBits (encodes entanglement)
    couplings: HashMap<(usize, usize), PBitCoupling>,
    /// Configuration
    config: PBitConfig,
    /// Random number generator state (for reproducibility)
    rng: ChaCha8Rng,
    /// Cached basis state probabilities (for compatibility)
    basis_probabilities: Vec<f64>,
}

impl PBitState {
    /// Create a new pBit state in |0...0⟩
    pub fn new(num_qubits: usize) -> QuantumResult<Self> {
        Self::with_config(num_qubits, PBitConfig::default())
    }

    /// Create a new pBit state with custom configuration
    pub fn with_config(num_qubits: usize, config: PBitConfig) -> QuantumResult<Self> {
        if num_qubits == 0 {
            return Err(QuantumError::invalid_state("Number of qubits must be > 0"));
        }
        if num_qubits > 32 {
            return Err(QuantumError::invalid_state("Maximum 32 qubits supported"));
        }

        let pbits: Vec<PBit> = (0..num_qubits).map(PBit::new).collect();
        let rng = match config.seed {
            Some(seed) => ChaCha8Rng::seed_from_u64(seed),
            None => ChaCha8Rng::from_entropy(),
        };

        let num_states = 1 << num_qubits;
        let mut basis_probabilities = vec![0.0; num_states];
        basis_probabilities[0] = 1.0; // |0...0⟩ state

        Ok(Self {
            num_qubits,
            pbits,
            couplings: HashMap::new(),
            config,
            rng,
            basis_probabilities,
        })
    }

    /// Create a uniform superposition state
    pub fn superposition(num_qubits: usize) -> QuantumResult<Self> {
        let mut state = Self::new(num_qubits)?;
        for pbit in &mut state.pbits {
            *pbit = PBit::superposition(pbit.index);
        }
        state.update_basis_probabilities();
        Ok(state)
    }

    /// Create a Bell state (2 qubits, maximally entangled)
    pub fn bell_state() -> QuantumResult<Self> {
        let config = PBitConfig {
            coupling_strength: 10.0, // Strong coupling
            temperature: 0.1,        // Low temperature for coherence
            ..Default::default()
        };
        let mut state = Self::with_config(2, config.clone())?;

        // Both qubits in superposition
        state.pbits[0] = PBit::superposition(0);
        state.pbits[1] = PBit::superposition(1);

        // Add strong ferromagnetic coupling (|00⟩ + |11⟩)
        state.add_coupling(PBitCoupling::bell_coupling(0, 1, config.coupling_strength));

        state.update_basis_probabilities();
        Ok(state)
    }

    /// Create from a QuantumState (convert complex amplitudes to probabilities)
    pub fn from_quantum_state(qs: &QuantumState) -> QuantumResult<Self> {
        let num_qubits = qs.num_qubits();
        let mut state = Self::new(num_qubits)?;

        // Extract probabilities from complex amplitudes
        state.basis_probabilities = qs.probability_distribution();

        // Compute marginal probabilities for each qubit
        for q in 0..num_qubits {
            let mut prob_one = 0.0;
            for (basis_idx, &prob) in state.basis_probabilities.iter().enumerate() {
                if (basis_idx >> q) & 1 == 1 {
                    prob_one += prob;
                }
            }
            state.pbits[q].probability_up = prob_one;
            state.pbits[q].spin = if prob_one > 0.5 { 1.0 } else { -1.0 };
        }

        // Detect entanglement from correlations
        state.detect_entanglement_from_probabilities();

        Ok(state)
    }

    /// Convert to QuantumState (probabilities to amplitudes)
    pub fn to_quantum_state(&self) -> QuantumResult<QuantumState> {
        // Create amplitudes from probabilities (real, positive)
        let amplitudes: Vec<Complex64> = self
            .basis_probabilities
            .iter()
            .map(|&p| Complex64::new(p.sqrt(), 0.0))
            .collect();

        QuantumState::from_amplitudes(amplitudes)
    }

    /// Add coupling between two pBits
    pub fn add_coupling(&mut self, coupling: PBitCoupling) {
        let key = if coupling.i < coupling.j {
            (coupling.i, coupling.j)
        } else {
            (coupling.j, coupling.i)
        };
        self.couplings.insert(key, coupling);
    }

    /// Compute local field for a pBit
    fn local_field(&self, qubit: usize) -> f64 {
        let mut field = self.config.external_field + self.pbits[qubit].bias;

        // Add contributions from coupled qubits
        for ((i, j), coupling) in &self.couplings {
            if *i == qubit {
                field += coupling.weight * self.pbits[*j].spin;
            } else if *j == qubit {
                field += coupling.weight * self.pbits[*i].spin;
            }
        }

        field
    }

    /// Perform one Gibbs sampling sweep
    pub fn sweep(&mut self) {
        for q in 0..self.num_qubits {
            let field = self.local_field(q);
            self.pbits[q].update_probability(field, self.config.temperature);
            self.pbits[q].sample(&mut self.rng);
            self.pbits[q].energy = -field * self.pbits[q].spin;
        }
        self.update_basis_probabilities();
    }

    /// Run multiple sweeps (equilibration)
    pub fn equilibrate(&mut self, sweeps: usize) {
        for _ in 0..sweeps {
            self.sweep();
        }
    }

    /// Anneal from high to low temperature
    pub fn anneal(&mut self, target_temp: f64, steps: usize) {
        let initial_temp = self.config.temperature;
        let temp_step = (initial_temp - target_temp) / steps as f64;

        for _ in 0..steps {
            self.config.temperature -= temp_step;
            self.config.temperature = self.config.temperature.max(target_temp);
            self.sweep();
        }
    }

    /// Measure all qubits (collapse to classical state)
    pub fn measure(&mut self) -> usize {
        // Sample from basis probability distribution
        let random_value: f64 = self.rng.gen();
        let mut cumulative = 0.0;

        for (state_idx, &prob) in self.basis_probabilities.iter().enumerate() {
            cumulative += prob;
            if random_value <= cumulative {
                // Collapse state
                self.basis_probabilities.fill(0.0);
                self.basis_probabilities[state_idx] = 1.0;

                // Update individual pBits
                for q in 0..self.num_qubits {
                    let bit = (state_idx >> q) & 1;
                    self.pbits[q].spin = if bit == 1 { 1.0 } else { -1.0 };
                    self.pbits[q].probability_up = bit as f64;
                }

                return state_idx;
            }
        }

        self.basis_probabilities.len() - 1
    }

    /// Measure specific qubits
    pub fn measure_qubits(&mut self, qubits: &[usize]) -> Vec<usize> {
        let measured_state = self.measure();
        qubits
            .iter()
            .map(|&q| (measured_state >> q) & 1)
            .collect()
    }

    /// Get probability distribution over basis states
    pub fn probability_distribution(&self) -> Vec<f64> {
        self.basis_probabilities.clone()
    }

    /// Get probability of specific basis state
    pub fn probability(&self, state_idx: usize) -> f64 {
        self.basis_probabilities.get(state_idx).copied().unwrap_or(0.0)
    }

    /// Update basis probabilities from individual pBit probabilities and couplings
    fn update_basis_probabilities(&mut self) {
        let num_states = 1 << self.num_qubits;

        // For uncoupled qubits: P(state) = Π P(qubit_i = bit_i)
        // For coupled qubits: use Boltzmann distribution
        if self.couplings.is_empty() {
            // Product of marginal probabilities
            for state_idx in 0..num_states {
                let mut prob = 1.0;
                for q in 0..self.num_qubits {
                    let bit = (state_idx >> q) & 1;
                    let p_up = self.pbits[q].probability_up;
                    prob *= if bit == 1 { p_up } else { 1.0 - p_up };
                }
                self.basis_probabilities[state_idx] = prob;
            }
        } else {
            // Boltzmann distribution with couplings
            let mut energies = vec![0.0; num_states];
            for state_idx in 0..num_states {
                let mut energy = 0.0;

                // External field and bias contributions
                for q in 0..self.num_qubits {
                    let spin = if (state_idx >> q) & 1 == 1 { 1.0 } else { -1.0 };
                    energy -= (self.config.external_field + self.pbits[q].bias) * spin;
                }

                // Coupling contributions
                for ((i, j), coupling) in &self.couplings {
                    let spin_i = if (state_idx >> i) & 1 == 1 { 1.0 } else { -1.0 };
                    let spin_j = if (state_idx >> j) & 1 == 1 { 1.0 } else { -1.0 };
                    energy -= coupling.weight * spin_i * spin_j;
                }

                energies[state_idx] = energy;
            }

            // Compute Boltzmann probabilities: P(state) ∝ exp(-E/T)
            let min_energy = energies.iter().cloned().fold(f64::INFINITY, f64::min);
            let mut partition_sum = 0.0;
            for state_idx in 0..num_states {
                let boltzmann = (-(energies[state_idx] - min_energy) / self.config.temperature).exp();
                self.basis_probabilities[state_idx] = boltzmann;
                partition_sum += boltzmann;
            }

            // Normalize
            for prob in &mut self.basis_probabilities {
                *prob /= partition_sum;
            }
        }
    }

    /// Detect entanglement from basis state correlations
    fn detect_entanglement_from_probabilities(&mut self) {
        // Compute correlation matrix
        for i in 0..self.num_qubits {
            for j in (i + 1)..self.num_qubits {
                let mut p_00 = 0.0;
                let mut p_01 = 0.0;
                let mut p_10 = 0.0;
                let mut p_11 = 0.0;

                for (state_idx, &prob) in self.basis_probabilities.iter().enumerate() {
                    let bit_i = (state_idx >> i) & 1;
                    let bit_j = (state_idx >> j) & 1;
                    match (bit_i, bit_j) {
                        (0, 0) => p_00 += prob,
                        (0, 1) => p_01 += prob,
                        (1, 0) => p_10 += prob,
                        (1, 1) => p_11 += prob,
                        _ => {}
                    }
                }

                // Correlation = P(same) - P(different)
                let correlation = (p_00 + p_11) - (p_01 + p_10);

                // If significant correlation, add coupling
                if correlation.abs() > 0.1 {
                    let coupling = if correlation > 0.0 {
                        PBitCoupling::bell_coupling(i, j, correlation.abs() * self.config.coupling_strength)
                    } else {
                        PBitCoupling::anti_bell_coupling(i, j, correlation.abs() * self.config.coupling_strength)
                    };
                    self.add_coupling(coupling);
                }
            }
        }
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Get number of basis states
    pub fn num_states(&self) -> usize {
        1 << self.num_qubits
    }

    /// Get individual pBit
    pub fn get_pbit(&self, index: usize) -> Option<&PBit> {
        self.pbits.get(index)
    }

    /// Get mutable pBit
    pub fn get_pbit_mut(&mut self, index: usize) -> Option<&mut PBit> {
        self.pbits.get_mut(index)
    }

    /// Get configuration
    pub fn config(&self) -> &PBitConfig {
        &self.config
    }

    /// Set temperature
    pub fn set_temperature(&mut self, temperature: f64) {
        self.config.temperature = temperature.max(1e-10);
        self.update_basis_probabilities();
    }

    /// Compute total entropy of the state
    pub fn entropy(&self) -> f64 {
        self.basis_probabilities
            .iter()
            .filter(|&&p| p > 1e-10)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Compute magnetization (average spin)
    pub fn magnetization(&self) -> f64 {
        let mut mag = 0.0;
        for (state_idx, &prob) in self.basis_probabilities.iter().enumerate() {
            let mut state_mag = 0.0;
            for q in 0..self.num_qubits {
                let spin = if (state_idx >> q) & 1 == 1 { 1.0 } else { -1.0 };
                state_mag += spin;
            }
            mag += prob * state_mag;
        }
        mag / self.num_qubits as f64
    }

    /// Check if state is normalized
    pub fn is_normalized(&self) -> bool {
        let sum: f64 = self.basis_probabilities.iter().sum();
        (sum - 1.0).abs() < 1e-10
    }

    /// Compute fidelity with another pBit state
    pub fn fidelity(&self, other: &PBitState) -> QuantumResult<f64> {
        if self.num_qubits != other.num_qubits {
            return Err(QuantumError::computation_error(
                "fidelity",
                "States must have same number of qubits",
            ));
        }

        // Classical fidelity: F = (Σ √(p_i * q_i))²
        let overlap: f64 = self
            .basis_probabilities
            .iter()
            .zip(other.basis_probabilities.iter())
            .map(|(&p, &q)| (p * q).sqrt())
            .sum();

        Ok(overlap * overlap)
    }

    /// Apply decoherence (increase temperature, decay couplings)
    pub fn apply_decoherence(&mut self, rate: f64, time_step: f64) {
        let decay = (-rate * time_step).exp();

        // Decay coupling strengths
        for coupling in self.couplings.values_mut() {
            coupling.weight *= decay;
        }

        // Remove weak couplings
        self.couplings.retain(|_, c| c.weight.abs() > 0.01);

        // Increase temperature (thermal noise)
        self.config.temperature += rate * time_step * 0.1;

        self.update_basis_probabilities();
    }

    /// Convert to classical bitstring (most probable state)
    pub fn to_classical_bitstring(&self) -> String {
        let max_state = self
            .basis_probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        format!("{:0width$b}", max_state, width = self.num_qubits)
    }
}

/// Trait for types that can be represented as pBit states
pub trait ToPBitState {
    /// Convert to pBit state representation
    fn to_pbit_state(&self) -> QuantumResult<PBitState>;
}

/// Trait for types that can be constructed from pBit states
pub trait FromPBitState: Sized {
    /// Construct from pBit state representation
    fn from_pbit_state(state: &PBitState) -> QuantumResult<Self>;
}

impl ToPBitState for QuantumState {
    fn to_pbit_state(&self) -> QuantumResult<PBitState> {
        PBitState::from_quantum_state(self)
    }
}

impl FromPBitState for QuantumState {
    fn from_pbit_state(state: &PBitState) -> QuantumResult<Self> {
        state.to_quantum_state()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbit_creation() {
        let state = PBitState::new(3).unwrap();
        assert_eq!(state.num_qubits(), 3);
        assert_eq!(state.num_states(), 8);
        assert!((state.probability(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_superposition() {
        let state = PBitState::superposition(2).unwrap();
        assert!((state.probability(0) - 0.25).abs() < 1e-10);
        assert!((state.probability(1) - 0.25).abs() < 1e-10);
        assert!((state.probability(2) - 0.25).abs() < 1e-10);
        assert!((state.probability(3) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let state = PBitState::bell_state().unwrap();
        assert_eq!(state.num_qubits(), 2);
        
        // Bell state should have high probability for |00⟩ and |11⟩
        let p00 = state.probability(0);
        let p11 = state.probability(3);
        assert!(p00 + p11 > 0.9); // Should be close to 1.0
    }

    #[test]
    fn test_quantum_conversion() {
        let qs = QuantumState::superposition(2).unwrap();
        let pbit = PBitState::from_quantum_state(&qs).unwrap();
        let qs_back = pbit.to_quantum_state().unwrap();

        // Probabilities should match
        for i in 0..4 {
            let p1 = qs.probability_distribution()[i];
            let p2 = qs_back.probability_distribution()[i];
            assert!((p1 - p2).abs() < 0.1);
        }
    }

    #[test]
    fn test_measurement() {
        let mut state = PBitState::new(2).unwrap();
        let result = state.measure();
        assert_eq!(result, 0); // Should always measure |00⟩
    }

    #[test]
    fn test_annealing() {
        let mut state = PBitState::superposition(3).unwrap();
        state.config.temperature = 10.0;
        state.anneal(0.1, 100);

        // After annealing, state should be more ordered
        assert!(state.config.temperature < 0.2);
    }

    #[test]
    fn test_entropy() {
        let state = PBitState::new(2).unwrap();
        assert!(state.entropy().abs() < 1e-10); // Pure state has zero entropy

        let superposition = PBitState::superposition(2).unwrap();
        assert!(superposition.entropy() > 1.0); // Superposition has high entropy
    }
}
