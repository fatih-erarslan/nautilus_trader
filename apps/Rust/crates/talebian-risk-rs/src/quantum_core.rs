//! Quantum Core Stub Module
//!
//! Provides quantum computing functionality stubs for the Talebian Risk system
//! This is a temporary stub until the actual quantum-core dependency is available

use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use num_complex::Complex64;

/// Extension trait to add norm_sqr method to f64
pub trait F64Ext {
    fn norm_sqr(&self) -> f64;
}

impl F64Ext for f64 {
    fn norm_sqr(&self) -> f64 {
        self * self
    }
}

/// Quantum computation errors
#[derive(Error, Debug)]
pub enum QuantumError {
    #[error("Invalid quantum state: {0}")]
    InvalidState(String),
    #[error("Circuit error: {0}")]
    CircuitError(String),
    #[error("Measurement error: {0}")]
    MeasurementError(String),
    #[error("Device error: {0}")]
    DeviceError(String),
    #[error("Calculation error: {0}")]
    CalculationError(String),
}

impl QuantumError {
    pub fn device_error<S: Into<String>>(msg: S) -> Self {
        Self::DeviceError(msg.into())
    }
}

/// Result type for quantum operations
pub type QuantumResult<T> = Result<T, QuantumError>;

/// Complex amplitude for quantum states
pub type ComplexAmplitude = num_complex::Complex64;
pub use num_complex::Complex64 as Complex;

/// Gate operation descriptor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateOperation {
    pub gate: QuantumGate,
    pub target_qubits: Vec<usize>,
}

/// Quantum device types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceType {
    Simulator,
    Hardware,
    CloudSimulator,
    CloudHardware,
}

/// Quantum device abstraction
#[derive(Debug, Clone)]
pub struct QuantumDevice {
    pub device_type: DeviceType,
    pub max_qubits: usize,
    pub error_rate: f64,
}

impl QuantumDevice {
    pub fn new(device_type: DeviceType) -> Self {
        Self::new_with_params(device_type, None)
    }

    pub fn new_simple(device_type: DeviceType) -> Self {
        Self::new(device_type)
    }

    pub fn new_with_params(device_type: DeviceType, max_qubits: Option<usize>) -> Self {
        Self {
            device_type,
            max_qubits: max_qubits.unwrap_or(match device_type {
                DeviceType::Simulator => 30,
                DeviceType::Hardware => 20,
                DeviceType::CloudSimulator => 40,
                DeviceType::CloudHardware => 127,
            }),
            error_rate: match device_type {
                DeviceType::Simulator => 0.0,
                DeviceType::Hardware => 0.001,
                DeviceType::CloudSimulator => 0.0,
                DeviceType::CloudHardware => 0.01,
            },
        }
    }

    pub fn execute_circuit(&self, circuit: &QuantumCircuit) -> QuantumResult<QuantumState> {
        if circuit.num_qubits > self.max_qubits {
            return Err(QuantumError::DeviceError(format!(
                "Circuit requires {} qubits, device supports {}",
                circuit.num_qubits, self.max_qubits
            )));
        }

        let state_size = 2_usize.pow(circuit.num_qubits as u32);
        let initial_amplitude = 1.0 / (state_size as f64).sqrt();
        Ok(QuantumState {
            amplitudes: vec![initial_amplitude; state_size],
            phases: vec![0.0; state_size],
            entanglement: 0.5,
            complex_amplitudes: vec![Complex64::new(initial_amplitude, 0.0); state_size],
        })
    }
}

/// Quantum state representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub entanglement: f64,
    pub complex_amplitudes: Vec<Complex64>,
}

impl QuantumState {
    /// Create new quantum state with given number of qubits
    pub fn new(num_qubits: usize) -> QuantumResult<Self> {
        let state_size = 2_usize.pow(num_qubits as u32);
        let initial_amplitude = 1.0 / (state_size as f64).sqrt();
        Ok(Self {
            amplitudes: vec![initial_amplitude; state_size],
            phases: vec![0.0; state_size],
            entanglement: 0.0,
            complex_amplitudes: vec![Complex64::new(initial_amplitude, 0.0); state_size],
        })
    }

    /// Set amplitude at given index - supports both f64 and Complex64
    pub fn set_amplitude(&mut self, index: usize, amplitude: f64) -> QuantumResult<()> {
        if index >= self.amplitudes.len() {
            return Err(QuantumError::CalculationError(format!(
                "Index {} out of bounds for state size {}",
                index,
                self.amplitudes.len()
            )));
        }
        self.amplitudes[index] = amplitude;
        self.complex_amplitudes[index] = Complex64::new(amplitude, 0.0);
        Ok(())
    }
    
    /// Set complex amplitude at given index
    pub fn set_complex_amplitude(&mut self, index: usize, amplitude: Complex64) -> QuantumResult<()> {
        if index >= self.amplitudes.len() {
            return Err(QuantumError::CalculationError(format!(
                "Index {} out of bounds for state size {}",
                index,
                self.amplitudes.len()
            )));
        }
        self.amplitudes[index] = amplitude.norm();
        self.complex_amplitudes[index] = amplitude;
        Ok(())
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) -> QuantumResult<()> {
        let norm_squared: f64 = self.amplitudes.iter().map(|&a| a * a).sum();
        let norm = norm_squared.sqrt();

        if norm > 0.0 {
            for amplitude in &mut self.amplitudes {
                *amplitude /= norm;
            }
        }
        Ok(())
    }

    /// Get probability of measuring a specific state
    pub fn get_probability(&self, index: usize) -> f64 {
        if index < self.amplitudes.len() {
            self.amplitudes[index] * self.amplitudes[index]
        } else {
            0.0
        }
    }

    /// Measure the quantum state (collapse to classical state)
    pub fn measure(&mut self) -> QuantumResult<usize> {
        let probabilities: Vec<f64> = self.amplitudes.iter().map(|&a| a * a).collect();
        let total: f64 = probabilities.iter().sum();

        // Simple random measurement simulation
        let mut rng = rand::thread_rng();
        let random: f64 = rng.gen_range(0.0..total);

        let mut cumulative = 0.0;
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if random <= cumulative {
                // Collapse to this state
                self.amplitudes.fill(0.0);
                self.amplitudes[i] = 1.0;
                return Ok(i);
            }
        }

        Ok(0) // Fallback
    }

    /// Get amplitudes reference
    pub fn get_amplitudes(&self) -> &Vec<f64> {
        &self.amplitudes
    }
}

/// Quantum circuit builder
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub num_qubits: usize,
    pub depth: usize,
    pub gates: Vec<QuantumGate>,
    pub gate_sequence: Vec<GateOperation>,
}

impl QuantumCircuit {
    /// Create new quantum circuit with optional name
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            depth: 0,
            gates: Vec::new(),
            gate_sequence: Vec::new(),
        }
    }
    
    /// Create new quantum circuit with name (compatibility)
    pub fn new_with_name(_name: String, num_qubits: usize) -> Self {
        Self::new(num_qubits)
    }
    
    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: QuantumGate, qubits: Vec<usize>) -> QuantumResult<()> {
        self.gates.push(gate.clone());
        self.gate_sequence.push(GateOperation {
            gate,
            target_qubits: qubits,
        });
        self.depth += 1;
        Ok(())
    }

    /// Execute this circuit on a device
    pub fn execute(&self, device: &QuantumDevice) -> QuantumResult<QuantumState> {
        device.execute_circuit(self)
    }
}

/// Quantum gate types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGate {
    Hadamard(usize),
    PauliX(usize),
    PauliY(usize),
    PauliZ(usize),
    CNOT(usize, usize),
    Phase(usize, f64),
    Rotation(usize, f64, f64, f64),
    RotationX(usize, f64),
    RotationY(usize, f64),
    RotationZ(usize, f64),
    ControlledNot(usize, usize),
    ControlledPhase(usize, usize, f64),
    RZ { qubit: usize, angle: f64 },
    CPhase { control: usize, target: usize, angle: f64 },
}

impl QuantumGate {
    /// Create Hadamard gate
    pub fn hadamard(qubit: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::Hadamard(qubit))
    }
    
    /// Create X rotation gate
    pub fn rotation_x(qubit: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::RotationX(qubit, angle))
    }
    
    /// Create Y rotation gate  
    pub fn rotation_y(qubit: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::RotationY(qubit, angle))
    }
    
    /// Create Z rotation gate
    pub fn rotation_z(qubit: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::RotationZ(qubit, angle))
    }
    
    /// Create controlled NOT gate
    pub fn controlled_not(control: usize, target: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::ControlledNot(control, target))
    }
    
    /// Create controlled phase gate
    pub fn controlled_phase(control: usize, target: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::ControlledPhase(control, target, angle))
    }
}

/// Quantum processor for financial calculations
#[derive(Debug, Clone)]
pub struct QuantumProcessor {
    pub max_qubits: usize,
    pub error_rate: f64,
}

impl QuantumProcessor {
    pub fn new(max_qubits: usize) -> Self {
        Self {
            max_qubits,
            error_rate: 0.001,
        }
    }

    pub fn create_circuit(&self, num_qubits: usize) -> QuantumCircuit {
        QuantumCircuit {
            num_qubits,
            depth: 10,
            gates: Vec::new(),
            gate_sequence: Vec::new(),
        }
    }

    pub fn execute(&self, circuit: &QuantumCircuit) -> QuantumState {
        let state_size = 2_usize.pow(circuit.num_qubits as u32);
        let initial_amplitude = 1.0 / (state_size as f64).sqrt();
        QuantumState {
            amplitudes: vec![initial_amplitude; state_size],
            phases: vec![0.0; state_size],
            entanglement: 0.5,
            complex_amplitudes: vec![Complex64::new(initial_amplitude, 0.0); state_size],
        }
    }

    pub fn measure(&self, state: &QuantumState) -> Vec<f64> {
        state.amplitudes.iter().map(|&a| a * a).collect()
    }
}

/// Quantum annealer for optimization
#[derive(Debug, Clone)]
pub struct QuantumAnnealer {
    pub num_spins: usize,
    pub temperature: f64,
}

impl QuantumAnnealer {
    pub fn new(num_spins: usize) -> Self {
        Self {
            num_spins,
            temperature: 1.0,
        }
    }

    pub fn optimize(&self, _cost_function: &[f64]) -> Vec<f64> {
        // Stub optimization
        vec![0.5; self.num_spins]
    }
}

/// Quantum random number generator
pub struct QuantumRandom {
    seed: u64,
}

impl QuantumRandom {
    pub fn new() -> Self {
        Self { seed: 42 }
    }

    pub fn generate(&mut self) -> f64 {
        // Simple pseudo-random for stub
        self.seed = (self.seed * 1103515245 + 12345) & 0x7fffffff;
        (self.seed as f64) / (0x7fffffff as f64)
    }

    pub fn generate_batch(&mut self, size: usize) -> Vec<f64> {
        (0..size).map(|_| self.generate()).collect()
    }
}

impl Default for QuantumRandom {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantum entanglement measure
pub fn calculate_entanglement(state: &QuantumState) -> f64 {
    // Simplified entanglement calculation
    let n = state.amplitudes.len();
    if n < 2 {
        return 0.0;
    }

    let mut entropy = 0.0;
    for &amp in &state.amplitudes {
        let prob = amp * amp;
        if prob > 1e-10 {
            entropy -= prob * prob.ln();
        }
    }

    entropy / (n as f64).ln()
}

/// Quantum interference for enhanced calculations
pub fn apply_quantum_interference(values: &[f64], phases: &[f64]) -> Vec<f64> {
    values
        .iter()
        .zip(phases.iter())
        .map(|(&v, &p)| v * (1.0 + 0.1 * p.cos()))
        .collect()
}

/// Quantum amplitude amplification
pub fn amplitude_amplification(probabilities: &[f64], iterations: usize) -> Vec<f64> {
    let mut amplified = probabilities.to_vec();

    for _ in 0..iterations {
        let mean: f64 = amplified.iter().sum::<f64>() / amplified.len() as f64;
        for amp in &mut amplified {
            *amp = 2.0 * mean - *amp;
            *amp = (*amp).max(0.0).min(1.0);
        }

        // Renormalize
        let sum: f64 = amplified.iter().sum();
        if sum > 0.0 {
            for amp in &mut amplified {
                *amp /= sum;
            }
        }
    }

    amplified
}
