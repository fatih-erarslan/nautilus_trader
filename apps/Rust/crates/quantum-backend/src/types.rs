//! Types for quantum backend

use serde::{Deserialize, Serialize};
use quantum_core::{QuantumGate as CoreGate, ComplexAmplitude};

/// Re-export gate types from quantum-core
pub use quantum_core::{
    QuantumState,
    QuantumCircuit,
    QuantumResult,
};

/// Extended gate types for backend operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GateType {
    Hadamard,
    PauliX,
    PauliY,
    PauliZ,
    CNOT,
    RX(f64),
    RY(f64),
    RZ(f64),
    Phase(f64),
    T,
    S,
    SWAP,
    CZ,
    Toffoli,
}

/// Quantum gate with extended functionality
#[derive(Debug, Clone)]
pub struct QuantumGate {
    gate_type: GateType,
    target: usize,
    control: Option<usize>,
    control2: Option<usize>, // For Toffoli
}

impl QuantumGate {
    pub fn gate_type(&self) -> &GateType {
        &self.gate_type
    }
    
    pub fn target(&self) -> usize {
        self.target
    }
    
    pub fn control(&self) -> Option<usize> {
        self.control
    }
    
    pub fn is_single_qubit(&self) -> bool {
        self.control.is_none()
    }
    
    pub fn is_rotation(&self) -> bool {
        matches!(self.gate_type, GateType::RX(_) | GateType::RY(_) | GateType::RZ(_))
    }
    
    // Gate constructors
    pub fn hadamard(target: usize) -> Self {
        Self {
            gate_type: GateType::Hadamard,
            target,
            control: None,
            control2: None,
        }
    }
    
    pub fn pauli_x(target: usize) -> Self {
        Self {
            gate_type: GateType::PauliX,
            target,
            control: None,
            control2: None,
        }
    }
    
    pub fn pauli_y(target: usize) -> Self {
        Self {
            gate_type: GateType::PauliY,
            target,
            control: None,
            control2: None,
        }
    }
    
    pub fn pauli_z(target: usize) -> Self {
        Self {
            gate_type: GateType::PauliZ,
            target,
            control: None,
            control2: None,
        }
    }
    
    pub fn cnot(control: usize, target: usize) -> Self {
        Self {
            gate_type: GateType::CNOT,
            target,
            control: Some(control),
            control2: None,
        }
    }
    
    pub fn rx(target: usize, angle: f64) -> Self {
        Self {
            gate_type: GateType::RX(angle),
            target,
            control: None,
            control2: None,
        }
    }
    
    pub fn ry(target: usize, angle: f64) -> Self {
        Self {
            gate_type: GateType::RY(angle),
            target,
            control: None,
            control2: None,
        }
    }
    
    pub fn rz(target: usize, angle: f64) -> Self {
        Self {
            gate_type: GateType::RZ(angle),
            target,
            control: None,
            control2: None,
        }
    }
}

/// Backend execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendStats {
    pub total_circuits_executed: u64,
    pub total_gates_applied: u64,
    pub average_circuit_depth: f64,
    pub average_execution_time_ms: f64,
    pub cache_hit_rate: f64,
    pub gpu_utilization: f64,
}

/// Quantum circuit execution options
#[derive(Debug, Clone)]
pub struct ExecutionOptions {
    pub shots: Option<usize>,
    pub optimization_level: u8,
    pub use_gpu: bool,
    pub error_mitigation: bool,
    pub parallel_experiments: usize,
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            shots: None,
            optimization_level: 2,
            use_gpu: true,
            error_mitigation: false,
            parallel_experiments: 1,
        }
    }
}

/// Quantum algorithm types
#[derive(Debug, Clone, PartialEq)]
pub enum AlgorithmType {
    VQE,
    QAOA,
    QPE,
    Grover,
    Shor,
    HHL,
    Custom(String),
}

/// Market-specific quantum operations
#[derive(Debug, Clone)]
pub struct MarketQuantumOp {
    pub op_type: MarketOpType,
    pub parameters: Vec<f64>,
    pub qubits: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MarketOpType {
    PortfolioOptimization,
    RiskAnalysis,
    PriceForecasting,
    ArbitrageDetection,
    NashEquilibrium,
    OptionPricing,
}