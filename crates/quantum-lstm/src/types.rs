//! Common types for Quantum LSTM

use ndarray::{Array1, Array2, Array3, Array4};
use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Type alias for complex numbers in quantum computations
pub type ComplexNum = Complex64;

/// Type alias for quantum state vectors
pub type StateVector = Array1<ComplexNum>;

/// Type alias for density matrices
pub type DensityMatrix = Array2<ComplexNum>;

/// Type alias for real-valued vectors
pub type RealVector = Array1<f64>;

/// Type alias for real-valued matrices
pub type RealMatrix = Array2<f64>;

/// Type alias for batch data (batch_size, sequence_length, features)
pub type BatchData = Array3<f64>;

/// Type alias for LSTM weights (4 gates x hidden_size x input_size)
pub type LSTMWeights = Array3<f64>;

/// Type alias for attention weights
pub type AttentionWeights = Array2<f64>;

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// State vector amplitudes
    pub amplitudes: StateVector,
    /// Number of qubits
    pub num_qubits: usize,
    /// Optional phase information
    pub global_phase: Option<f64>,
}

/// LSTM hidden state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiddenState {
    /// Hidden state values
    pub h: RealMatrix,
    /// Cell state values
    pub c: RealMatrix,
}

/// Quantum LSTM output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumLSTMOutput {
    /// Output sequence
    pub output: BatchData,
    /// Final hidden state
    pub hidden_state: HiddenState,
    /// Attention weights (if attention is used)
    pub attention_weights: Option<AttentionWeights>,
    /// Quantum fidelity metrics
    pub quantum_metrics: Option<QuantumMetrics>,
}

/// Quantum metrics for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    /// Average gate fidelity
    pub gate_fidelity: f64,
    /// State preparation fidelity
    pub state_fidelity: f64,
    /// Entanglement entropy
    pub entanglement_entropy: f64,
    /// Quantum volume
    pub quantum_volume: Option<f64>,
    /// Circuit depth
    pub circuit_depth: usize,
}

/// Encoding type for quantum state preparation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EncodingType {
    /// Amplitude encoding
    Amplitude,
    /// Angle encoding
    Angle,
    /// Basis encoding
    Basis,
    /// IQP (Instantaneous Quantum Polynomial) encoding
    IQP,
    /// Hybrid encoding
    Hybrid,
}

/// Gate type for quantum circuits
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GateType {
    /// Hadamard gate
    Hadamard,
    /// Pauli-X gate
    PauliX,
    /// Pauli-Y gate
    PauliY,
    /// Pauli-Z gate
    PauliZ,
    /// Rotation around X axis
    RX(f64),
    /// Rotation around Y axis
    RY(f64),
    /// Rotation around Z axis
    RZ(f64),
    /// Controlled-NOT gate
    CNOT,
    /// Controlled-Z gate
    CZ,
    /// Controlled rotation
    CRZ(f64),
    /// SWAP gate
    SWAP,
    /// Toffoli gate
    Toffoli,
}

/// Biological quantum effect type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BiologicalEffect {
    /// Quantum tunneling
    Tunneling,
    /// Quantum coherence
    Coherence,
    /// Quantum criticality
    Criticality,
    /// Quantum entanglement
    Entanglement,
    /// Quantum superposition
    Superposition,
}

/// Market tick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketTick {
    /// Timestamp
    pub timestamp: i64,
    /// Price
    pub price: f64,
    /// Volume
    pub volume: f64,
    /// Additional features
    pub features: Vec<f64>,
}

/// Time series window
#[derive(Debug, Clone)]
pub struct TimeSeriesWindow {
    /// Window data
    pub data: Array2<f64>,
    /// Start timestamp
    pub start_time: i64,
    /// End timestamp
    pub end_time: i64,
}

/// Cache key type
pub type CacheKey = u64;

/// Thread-safe reference to quantum state
pub type QuantumStateRef = Arc<parking_lot::RwLock<QuantumState>>;

/// Thread-safe reference to LSTM weights
pub type WeightsRef = Arc<parking_lot::RwLock<LSTMWeights>>;