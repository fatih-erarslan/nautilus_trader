//! Core types and traits for the QERC library
//!
//! This module defines the fundamental types, traits, and error handling
//! for quantum error correction in trading systems.

use std::collections::HashMap;
use std::fmt;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use num_complex::Complex64;

/// Main error type for QERC operations
#[derive(Error, Debug, Clone)]
pub enum QercError {
    /// Error in quantum state manipulation
    #[error("Quantum state error: {message}")]
    QuantumStateError { message: String },
    
    /// Error in error detection
    #[error("Error detection failed: {message}")]
    ErrorDetectionError { message: String },
    
    /// Error in syndrome measurement
    #[error("Syndrome measurement failed: {message}")]
    SyndromeMeasurementError { message: String },
    
    /// Error in syndrome decoding
    #[error("Syndrome decoding failed: {message}")]
    SyndromeDecodingError { message: String },
    
    /// Error in error correction
    #[error("Error correction failed: {message}")]
    ErrorCorrectionError { message: String },
    
    /// Error in fault-tolerant operations
    #[error("Fault-tolerant operation failed: {message}")]
    FaultToleranceError { message: String },
    
    /// Error in code construction
    #[error("Code construction failed: {message}")]
    CodeConstructionError { message: String },
    
    /// Error in integration with other systems
    #[error("Integration error: {message}")]
    IntegrationError { message: String },
    
    /// Hardware acceleration error
    #[error("Hardware acceleration error: {message}")]
    HardwareError { message: String },
    
    /// Performance monitoring error
    #[error("Performance monitoring error: {message}")]
    PerformanceError { message: String },
    
    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },
    
    /// Resource allocation error
    #[error("Resource allocation error: {message}")]
    ResourceError { message: String },
    
    /// Timeout error
    #[error("Operation timed out: {message}")]
    TimeoutError { message: String },
    
    /// Invalid operation error
    #[error("Invalid operation: {message}")]
    InvalidOperationError { message: String },
}

/// Result type for QERC operations
pub type QercResult<T> = Result<T, QercError>;

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// State vector amplitudes
    pub amplitudes: Vec<Complex64>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Density matrix representation (optional)
    pub density_matrix: Option<Vec<Vec<Complex64>>>,
    /// State normalization factor
    pub normalization: f64,
    /// Metadata for state tracking
    pub metadata: HashMap<String, String>,
}

impl QuantumState {
    /// Create a new quantum state with given amplitudes
    pub fn new(amplitudes: Vec<f64>) -> Self {
        let complex_amplitudes: Vec<Complex64> = amplitudes
            .into_iter()
            .map(|amp| Complex64::new(amp, 0.0))
            .collect();
        
        let num_qubits = (complex_amplitudes.len() as f64).log2().ceil() as usize;
        
        Self {
            amplitudes: complex_amplitudes,
            num_qubits,
            density_matrix: None,
            normalization: 1.0,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new multi-qubit quantum state
    pub fn new_multi_qubit(num_qubits: usize, amplitudes: Vec<f64>) -> Self {
        let complex_amplitudes: Vec<Complex64> = amplitudes
            .into_iter()
            .map(|amp| Complex64::new(amp, 0.0))
            .collect();
        
        Self {
            amplitudes: complex_amplitudes,
            num_qubits,
            density_matrix: None,
            normalization: 1.0,
            metadata: HashMap::new(),
        }
    }
    
    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    
    /// Get the state vector dimension
    pub fn dimension(&self) -> usize {
        self.amplitudes.len()
    }
    
    /// Check if the state is normalized
    pub fn is_normalized(&self) -> bool {
        let norm_squared: f64 = self.amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .sum();
        (norm_squared - 1.0).abs() < 1e-10
    }
    
    /// Normalize the quantum state
    pub fn normalize(&mut self) {
        let norm: f64 = self.amplitudes
            .iter()
            .map(|amp| amp.norm_sqr())
            .sum::<f64>()
            .sqrt();
        
        if norm > 0.0 {
            for amp in &mut self.amplitudes {
                *amp /= norm;
            }
            self.normalization = norm;
        }
    }
    
    /// Clone the quantum state
    pub fn clone(&self) -> Self {
        Self {
            amplitudes: self.amplitudes.clone(),
            num_qubits: self.num_qubits,
            density_matrix: self.density_matrix.clone(),
            normalization: self.normalization,
            metadata: self.metadata.clone(),
        }
    }
}

/// Types of quantum errors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorType {
    /// No error detected
    NoError,
    /// Bit flip error (X error)
    BitFlip,
    /// Phase flip error (Z error)
    PhaseFlip,
    /// Bit-phase flip error (Y error)
    BitPhaseFlip,
    /// Depolarizing error
    Depolarizing,
    /// Amplitude damping error
    AmplitudeDamping,
    /// Phase damping error
    PhaseDamping,
    /// Thermal noise error
    ThermalNoise,
    /// Measurement error
    MeasurementError,
    /// Gate error
    GateError,
    /// Correlated error
    CorrelatedError,
    /// Unknown error type
    Unknown,
}

impl fmt::Display for ErrorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ErrorType::NoError => write!(f, "NoError"),
            ErrorType::BitFlip => write!(f, "BitFlip"),
            ErrorType::PhaseFlip => write!(f, "PhaseFlip"),
            ErrorType::BitPhaseFlip => write!(f, "BitPhaseFlip"),
            ErrorType::Depolarizing => write!(f, "Depolarizing"),
            ErrorType::AmplitudeDamping => write!(f, "AmplitudeDamping"),
            ErrorType::PhaseDamping => write!(f, "PhaseDamping"),
            ErrorType::ThermalNoise => write!(f, "ThermalNoise"),
            ErrorType::MeasurementError => write!(f, "MeasurementError"),
            ErrorType::GateError => write!(f, "GateError"),
            ErrorType::CorrelatedError => write!(f, "CorrelatedError"),
            ErrorType::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Result of error detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorDetectionResult {
    /// Whether an error was detected
    pub has_error: bool,
    /// Type of error detected
    pub error_type: ErrorType,
    /// Location of single error (if applicable)
    pub error_location: Option<usize>,
    /// Locations of multiple errors
    pub error_locations: Vec<usize>,
    /// Confidence in error detection
    pub confidence: f64,
    /// Error syndrome
    pub syndrome: Option<Syndrome>,
    /// Detection timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl ErrorDetectionResult {
    /// Create a new error detection result with no error
    pub fn no_error() -> Self {
        Self {
            has_error: false,
            error_type: ErrorType::NoError,
            error_location: None,
            error_locations: Vec::new(),
            confidence: 1.0,
            syndrome: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new error detection result with single error
    pub fn single_error(error_type: ErrorType, location: usize, confidence: f64) -> Self {
        Self {
            has_error: true,
            error_type,
            error_location: Some(location),
            error_locations: vec![location],
            confidence,
            syndrome: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new error detection result with multiple errors
    pub fn multiple_errors(error_type: ErrorType, locations: Vec<usize>, confidence: f64) -> Self {
        Self {
            has_error: true,
            error_type,
            error_location: None,
            error_locations: locations,
            confidence,
            syndrome: None,
            timestamp: chrono::Utc::now(),
            metadata: HashMap::new(),
        }
    }
}

/// Syndrome representation for error correction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Syndrome {
    /// Binary syndrome vector
    pub syndrome_bits: Vec<bool>,
    /// Syndrome weight (number of triggered stabilizers)
    pub weight: usize,
    /// Syndrome confidence
    pub confidence: f64,
    /// Associated error pattern
    pub error_pattern: Option<Vec<usize>>,
    /// Measurement metadata
    pub metadata: HashMap<String, String>,
}

impl Syndrome {
    /// Create syndrome from binary string
    pub fn from_binary(binary_string: &str) -> Self {
        let syndrome_bits: Vec<bool> = binary_string
            .chars()
            .map(|c| c == '1')
            .collect();
        
        let weight = syndrome_bits.iter().filter(|&&bit| bit).count();
        
        Self {
            syndrome_bits,
            weight,
            confidence: 1.0,
            error_pattern: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Check if syndrome is trivial (all zeros)
    pub fn is_trivial(&self) -> bool {
        self.weight == 0
    }
    
    /// Get number of triggered stabilizers
    pub fn num_triggered_stabilizers(&self) -> usize {
        self.weight
    }
    
    /// Get syndrome length
    pub fn length(&self) -> usize {
        self.syndrome_bits.len()
    }
}

/// Measurement outcome
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MeasurementOutcome {
    /// Measurement result is 0
    Zero,
    /// Measurement result is 1
    One,
    /// Measurement failed
    Failed,
}

/// Quantum circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    /// Number of qubits in circuit
    pub num_qubits: usize,
    /// Number of classical bits
    pub num_classical: usize,
    /// Circuit gates
    pub gates: Vec<QuantumGate>,
    /// Circuit metadata
    pub metadata: HashMap<String, String>,
}

impl QuantumCircuit {
    /// Create new quantum circuit
    pub fn new(num_qubits: usize, num_classical: usize) -> Self {
        Self {
            num_qubits,
            num_classical,
            gates: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }
    
    /// Get circuit depth
    pub fn depth(&self) -> usize {
        self.gates.len()
    }
}

/// Quantum gate representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumGate {
    /// Gate type
    pub gate_type: GateType,
    /// Target qubits
    pub targets: Vec<usize>,
    /// Control qubits
    pub controls: Vec<usize>,
    /// Gate parameters
    pub parameters: Vec<f64>,
    /// Gate metadata
    pub metadata: HashMap<String, String>,
}

/// Types of quantum gates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GateType {
    /// Identity gate
    I,
    /// Pauli-X gate
    X,
    /// Pauli-Y gate
    Y,
    /// Pauli-Z gate
    Z,
    /// Hadamard gate
    H,
    /// Phase gate
    S,
    /// T gate
    T,
    /// CNOT gate
    CNOT,
    /// Toffoli gate
    Toffoli,
    /// Rotation gates
    RX,
    RY,
    RZ,
    /// Measurement
    Measure,
    /// Custom gate
    Custom(String),
}

/// Quantum computation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResult {
    /// Final quantum state
    pub final_state: QuantumState,
    /// Measurement results
    pub measurements: Vec<MeasurementOutcome>,
    /// Classical register values
    pub classical_register: Vec<bool>,
    /// Computation metadata
    pub metadata: HashMap<String, String>,
    /// Success flag
    pub success: bool,
    /// Error message (if any)
    pub error_message: Option<String>,
}

impl QuantumResult {
    /// Create successful result
    pub fn success(final_state: QuantumState) -> Self {
        Self {
            final_state,
            measurements: Vec::new(),
            classical_register: Vec::new(),
            metadata: HashMap::new(),
            success: true,
            error_message: None,
        }
    }
    
    /// Create failed result
    pub fn failure(error_message: String) -> Self {
        Self {
            final_state: QuantumState::new(vec![1.0, 0.0]), // Default state
            measurements: Vec::new(),
            classical_register: Vec::new(),
            metadata: HashMap::new(),
            success: false,
            error_message: Some(error_message),
        }
    }
}

/// Configuration for QERC operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QercConfig {
    /// Error detection threshold
    pub error_threshold: f64,
    /// Maximum correction rounds
    pub correction_rounds: usize,
    /// Syndrome measurement repetitions
    pub syndrome_repetitions: usize,
    /// Decoding algorithm preference
    pub decoding_algorithm: DecodingAlgorithm,
    /// Hardware acceleration settings
    pub hardware_acceleration: bool,
    /// Performance monitoring settings
    pub performance_monitoring: bool,
    /// Real-time constraints
    pub real_time_constraints: RealTimeConstraints,
    /// Memory management settings
    pub memory_management: MemoryManagement,
}

impl Default for QercConfig {
    fn default() -> Self {
        Self {
            error_threshold: 0.1,
            correction_rounds: 10,
            syndrome_repetitions: 3,
            decoding_algorithm: DecodingAlgorithm::MinimumWeight,
            hardware_acceleration: true,
            performance_monitoring: true,
            real_time_constraints: RealTimeConstraints::default(),
            memory_management: MemoryManagement::default(),
        }
    }
}

/// Decoding algorithm options
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecodingAlgorithm {
    /// Minimum weight decoding
    MinimumWeight,
    /// Maximum likelihood decoding
    MaximumLikelihood,
    /// Neural network decoding
    NeuralNetwork,
    /// Belief propagation decoding
    BeliefPropagation,
    /// Lookup table decoding
    LookupTable,
    /// Adaptive decoding
    Adaptive,
}

/// Real-time constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeConstraints {
    /// Maximum latency in microseconds
    pub max_latency_us: u64,
    /// Maximum memory usage in MB
    pub max_memory_mb: u64,
    /// Maximum CPU usage (0.0 - 1.0)
    pub max_cpu_usage: f64,
    /// Priority level
    pub priority: Priority,
}

impl Default for RealTimeConstraints {
    fn default() -> Self {
        Self {
            max_latency_us: 100, // 100μs for HFT
            max_memory_mb: 100,  // 100MB
            max_cpu_usage: 0.8,  // 80% CPU
            priority: Priority::High,
        }
    }
}

/// Priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Priority {
    /// Low priority
    Low,
    /// Medium priority
    Medium,
    /// High priority
    High,
    /// Critical priority
    Critical,
}

/// Memory management settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryManagement {
    /// Enable memory pooling
    pub enable_pooling: bool,
    /// Cache size in MB
    pub cache_size_mb: u64,
    /// Garbage collection frequency
    pub gc_frequency: u64,
    /// Memory compression
    pub enable_compression: bool,
}

impl Default for MemoryManagement {
    fn default() -> Self {
        Self {
            enable_pooling: true,
            cache_size_mb: 50,
            gc_frequency: 1000,
            enable_compression: true,
        }
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Error detection rate
    pub error_detection_rate: f64,
    /// Error correction rate
    pub error_correction_rate: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// Average latency in milliseconds
    pub latency_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// CPU utilization
    pub cpu_utilization: f64,
    /// Throughput (operations per second)
    pub throughput: f64,
    /// Success rate
    pub success_rate: f64,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            error_detection_rate: 0.0,
            error_correction_rate: 0.0,
            false_positive_rate: 0.0,
            latency_ms: 0.0,
            memory_usage_mb: 0.0,
            cpu_utilization: 0.0,
            throughput: 0.0,
            success_rate: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_state_creation() {
        let state = QuantumState::new(vec![1.0, 0.0]);
        assert_eq!(state.num_qubits(), 1);
        assert_eq!(state.dimension(), 2);
        assert!(state.is_normalized());
    }

    #[test]
    fn test_error_detection_result() {
        let result = ErrorDetectionResult::no_error();
        assert!(!result.has_error);
        assert_eq!(result.error_type, ErrorType::NoError);
        
        let result = ErrorDetectionResult::single_error(ErrorType::BitFlip, 0, 0.95);
        assert!(result.has_error);
        assert_eq!(result.error_type, ErrorType::BitFlip);
        assert_eq!(result.error_location, Some(0));
    }

    #[test]
    fn test_syndrome_creation() {
        let syndrome = Syndrome::from_binary("101010");
        assert_eq!(syndrome.length(), 6);
        assert_eq!(syndrome.weight, 3);
        assert!(!syndrome.is_trivial());
        
        let trivial = Syndrome::from_binary("000000");
        assert!(trivial.is_trivial());
    }

    #[test]
    fn test_quantum_circuit() {
        let mut circuit = QuantumCircuit::new(2, 2);
        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.depth(), 0);
        
        circuit.add_gate(QuantumGate {
            gate_type: GateType::H,
            targets: vec![0],
            controls: vec![],
            parameters: vec![],
            metadata: HashMap::new(),
        });
        
        assert_eq!(circuit.depth(), 1);
    }
}