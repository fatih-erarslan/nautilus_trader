//! Core quantum device and circuit abstractions

use crate::{error::Result, types::*};
use quantum_core as qc;

/// Quantum device abstraction
pub struct QuantumDevice {
    inner: qc::QuantumDevice,
}

impl QuantumDevice {
    /// Create a new quantum device
    pub fn new(num_qubits: usize, backend: &str) -> Result<Self> {
        let device_type = match backend {
            "default.qubit" => qc::DeviceType::Simulator,
            "lightning.qubit" => qc::DeviceType::Simulator, // Fallback
            "lightning.gpu" => qc::DeviceType::HybridClassical,   // Fallback
            _ => qc::DeviceType::Simulator,
        };
        
        let inner = qc::QuantumDevice::new_simple(device_type, num_qubits)
            .map_err(|e| crate::error::QuantumLSTMError::Device(e.to_string()))?;
            
        Ok(Self { inner })
    }
    
    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.inner.capabilities.max_qubits
    }
}

/// Quantum circuit abstraction
pub struct QuantumCircuit {
    num_qubits: usize,
    gates: Vec<(GateType, Vec<usize>)>,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
        }
    }
    
    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: GateType, qubits: &[usize]) -> Result<()> {
        // Store gate for later execution
        self.gates.push((gate, qubits.to_vec()));
        Ok(())
    }
    
    /// Execute the circuit
    pub fn execute(&self, _device: &QuantumDevice) -> Result<QuantumState> {
        // Placeholder implementation
        let n_amplitudes = 1 << self.num_qubits;
        let amplitudes = ndarray::Array1::from_elem(
            n_amplitudes, 
            num_complex::Complex64::new(1.0 / (n_amplitudes as f64).sqrt(), 0.0)
        );
        
        Ok(QuantumState {
            amplitudes,
            num_qubits: self.num_qubits,
            global_phase: None,
        })
    }
}