//! Type definitions for quantum-inspired neural networks

use crate::tensor::{Tensor, TensorOps};
use std::fmt;

/// Complex number for quantum-inspired computations
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Complex {
    /// Real part
    pub re: f32,
    /// Imaginary part
    pub im: f32,
}

impl Complex {
    /// Create new complex number
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Create from polar form (r, θ) → r*e^(iθ) = r*(cos(θ) + i*sin(θ))
    pub fn from_polar(r: f32, theta: f32) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    /// Zero complex number
    pub const fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    /// Unit complex number (1 + 0i)
    pub const fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    /// Imaginary unit (0 + 1i)
    pub const fn i() -> Self {
        Self { re: 0.0, im: 1.0 }
    }

    /// Magnitude |z| = √(re² + im²)
    pub fn abs(&self) -> f32 {
        (self.re * self.re + self.im * self.im).sqrt()
    }

    /// Squared magnitude |z|² = re² + im²
    pub fn norm_sq(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle θ = atan2(im, re)
    pub fn arg(&self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate z* = re - i*im
    pub fn conj(&self) -> Self {
        Self { re: self.re, im: -self.im }
    }

    /// Exponential e^z = e^re * (cos(im) + i*sin(im))
    pub fn exp(&self) -> Self {
        let r = self.re.exp();
        Self {
            re: r * self.im.cos(),
            im: r * self.im.sin(),
        }
    }

    /// Natural logarithm ln(z) = ln|z| + i*arg(z)
    pub fn ln(&self) -> Self {
        Self {
            re: self.abs().ln(),
            im: self.arg(),
        }
    }

    /// Square root √z
    pub fn sqrt(&self) -> Self {
        let r = self.abs().sqrt();
        let theta = self.arg() / 2.0;
        Self::from_polar(r, theta)
    }
}

impl std::ops::Add for Complex {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

impl std::ops::Sub for Complex {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { re: self.re - rhs.re, im: self.im - rhs.im }
    }
}

impl std::ops::Mul for Complex {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl std::ops::Div for Complex {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        let denom = rhs.norm_sq();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denom,
            im: (self.im * rhs.re - self.re * rhs.im) / denom,
        }
    }
}

impl std::ops::Mul<f32> for Complex {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self { re: self.re * rhs, im: self.im * rhs }
    }
}

impl std::ops::Neg for Complex {
    type Output = Self;
    fn neg(self) -> Self {
        Self { re: -self.re, im: -self.im }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{:.4}+{:.4}i", self.re, self.im)
        } else {
            write!(f, "{:.4}{:.4}i", self.re, self.im)
        }
    }
}

/// Quantum-inspired state vector
/// Represents a superposition of basis states with complex amplitudes
#[derive(Debug, Clone)]
pub struct StateVector {
    /// Complex amplitudes
    pub amplitudes: Vec<Complex>,
    /// Number of "qubits" (log2 of dimension)
    pub num_qubits: usize,
}

impl StateVector {
    /// Create new state vector with given dimension (must be power of 2)
    pub fn new(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![Complex::zero(); dim];
        // Initialize to |0⟩ state
        amplitudes[0] = Complex::one();
        Self { amplitudes, num_qubits }
    }

    /// Create from amplitudes
    pub fn from_amplitudes(amplitudes: Vec<Complex>) -> Option<Self> {
        let dim = amplitudes.len();
        if dim == 0 || (dim & (dim - 1)) != 0 {
            return None; // Must be power of 2
        }
        let num_qubits = (dim as f32).log2() as usize;
        Some(Self { amplitudes, num_qubits })
    }

    /// Create uniform superposition |+⟩^n
    pub fn uniform(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let amp = Complex::new(1.0 / (dim as f32).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; dim],
            num_qubits,
        }
    }

    /// Dimension of state space
    pub fn dim(&self) -> usize {
        self.amplitudes.len()
    }

    /// Normalize state to unit length
    pub fn normalize(&mut self) {
        let norm: f32 = self.amplitudes.iter().map(|a| a.norm_sq()).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for a in &mut self.amplitudes {
                *a = *a * (1.0 / norm);
            }
        }
    }

    /// Check if normalized (|⟨ψ|ψ⟩| ≈ 1)
    pub fn is_normalized(&self, eps: f32) -> bool {
        let norm_sq: f32 = self.amplitudes.iter().map(|a| a.norm_sq()).sum();
        (norm_sq - 1.0).abs() < eps
    }

    /// Inner product ⟨ψ|φ⟩
    pub fn inner_product(&self, other: &Self) -> Complex {
        assert_eq!(self.dim(), other.dim());
        self.amplitudes.iter()
            .zip(other.amplitudes.iter())
            .map(|(a, b)| a.conj() * *b)
            .fold(Complex::zero(), |acc, x| acc + x)
    }

    /// Probability of measuring basis state |i⟩
    pub fn probability(&self, index: usize) -> f32 {
        self.amplitudes.get(index).map(|a| a.norm_sq()).unwrap_or(0.0)
    }

    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f32> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Apply phase rotation e^(iθ)|ψ⟩
    pub fn apply_global_phase(&mut self, theta: f32) {
        let phase = Complex::from_polar(1.0, theta);
        for a in &mut self.amplitudes {
            *a = *a * phase;
        }
    }

    /// Tensor product |ψ⟩ ⊗ |φ⟩
    pub fn tensor_product(&self, other: &Self) -> Self {
        let new_dim = self.dim() * other.dim();
        let mut amplitudes = Vec::with_capacity(new_dim);

        for a in &self.amplitudes {
            for b in &other.amplitudes {
                amplitudes.push(*a * *b);
            }
        }

        Self {
            amplitudes,
            num_qubits: self.num_qubits + other.num_qubits,
        }
    }
}

/// State encoding method for classical → quantum-inspired transformation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncodingType {
    /// Amplitude encoding: data values become amplitudes (normalized)
    Amplitude,
    /// Angle encoding: data values become rotation angles
    Angle,
    /// Phase encoding: data values become phases
    Phase,
    /// Hybrid: combination of amplitude and phase
    Hybrid,
    /// IQP (Instantaneous Quantum Polynomial) encoding
    IQP,
}

impl Default for EncodingType {
    fn default() -> Self {
        Self::Amplitude
    }
}

/// Biological quantum effect types for BioCognitive models
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BiologicalEffect {
    /// Quantum tunneling-inspired barrier crossing
    /// Allows signals to "tunnel" through activation barriers
    Tunneling,
    /// Quantum coherence-inspired synchronized oscillations
    /// Maintains phase relationships between neurons
    Coherence,
    /// Quantum criticality-inspired phase transitions
    /// Detects edge-of-chaos dynamics
    Criticality,
    /// Resonant energy transfer (like Förster resonance)
    /// Enables efficient information transfer
    ResonantTransfer,
    /// Spin coherence for memory persistence
    SpinCoherence,
}

/// Hidden state for quantum-inspired LSTM
#[derive(Debug, Clone)]
pub struct QuantumHiddenState {
    /// Complex-valued hidden state
    pub h: Vec<Complex>,
    /// Complex-valued cell state
    pub c: Vec<Complex>,
    /// Phase information for coherence
    pub phases: Vec<f32>,
}

impl QuantumHiddenState {
    /// Create new hidden state with given hidden size
    pub fn new(hidden_size: usize) -> Self {
        Self {
            h: vec![Complex::zero(); hidden_size],
            c: vec![Complex::zero(); hidden_size],
            phases: vec![0.0; hidden_size],
        }
    }

    /// Reset to zero state
    pub fn reset(&mut self) {
        for h in &mut self.h {
            *h = Complex::zero();
        }
        for c in &mut self.c {
            *c = Complex::zero();
        }
        for p in &mut self.phases {
            *p = 0.0;
        }
    }

    /// Get real-valued hidden state (magnitude)
    pub fn h_real(&self) -> Vec<f32> {
        self.h.iter().map(|c| c.abs()).collect()
    }

    /// Get real-valued cell state (magnitude)
    pub fn c_real(&self) -> Vec<f32> {
        self.c.iter().map(|c| c.abs()).collect()
    }
}

/// Output from quantum-inspired LSTM
#[derive(Debug, Clone)]
pub struct QuantumLSTMOutput {
    /// Output tensor (batch, seq_len, hidden_size)
    pub output: Tensor,
    /// Final hidden state
    pub hidden_state: QuantumHiddenState,
    /// Attention weights if attention was used
    pub attention_weights: Option<Tensor>,
    /// Coherence metrics
    pub coherence_metrics: Option<CoherenceMetrics>,
}

/// Metrics for tracking quantum-inspired coherence
#[derive(Debug, Clone)]
pub struct CoherenceMetrics {
    /// Average phase coherence across hidden units
    pub phase_coherence: f32,
    /// Fidelity of state preservation
    pub state_fidelity: f32,
    /// Entanglement-like correlations
    pub correlation_entropy: f32,
    /// Tunneling events count
    pub tunneling_events: usize,
}

impl Default for CoherenceMetrics {
    fn default() -> Self {
        Self {
            phase_coherence: 1.0,
            state_fidelity: 1.0,
            correlation_entropy: 0.0,
            tunneling_events: 0,
        }
    }
}

/// Quantum gate types for circuit-based operations
#[derive(Debug, Clone, Copy)]
pub enum GateType {
    /// Hadamard gate: creates superposition
    Hadamard,
    /// Pauli-X gate: bit flip
    PauliX,
    /// Pauli-Y gate: bit and phase flip
    PauliY,
    /// Pauli-Z gate: phase flip
    PauliZ,
    /// Rotation around X axis
    RX(f32),
    /// Rotation around Y axis
    RY(f32),
    /// Rotation around Z axis
    RZ(f32),
    /// Phase gate
    Phase(f32),
    /// CNOT (controlled-NOT)
    CNOT,
    /// Controlled-Z
    CZ,
    /// SWAP gate
    Swap,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_arithmetic() {
        let a = Complex::new(3.0, 4.0);
        let b = Complex::new(1.0, 2.0);

        // Addition
        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-6);
        assert!((sum.im - 6.0).abs() < 1e-6);

        // Multiplication
        let prod = a * b;
        // (3+4i)(1+2i) = 3 + 6i + 4i + 8i² = 3 + 10i - 8 = -5 + 10i
        assert!((prod.re - (-5.0)).abs() < 1e-6);
        assert!((prod.im - 10.0).abs() < 1e-6);

        // Magnitude
        assert!((a.abs() - 5.0).abs() < 1e-6); // |3+4i| = 5
    }

    #[test]
    fn test_state_vector() {
        let mut state = StateVector::new(2); // 4-dimensional
        assert_eq!(state.dim(), 4);

        // Should be |00⟩ state
        assert!((state.probability(0) - 1.0).abs() < 1e-6);
        assert!(state.probability(1) < 1e-6);

        // Create uniform superposition
        let uniform = StateVector::uniform(2);
        for i in 0..4 {
            assert!((uniform.probability(i) - 0.25).abs() < 1e-6);
        }
    }

    #[test]
    fn test_inner_product() {
        let state1 = StateVector::uniform(1);
        let state2 = StateVector::uniform(1);

        let ip = state1.inner_product(&state2);
        // ⟨+|+⟩ = 1
        assert!((ip.abs() - 1.0).abs() < 1e-6);
    }
}
