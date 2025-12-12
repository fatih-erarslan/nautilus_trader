//! Python bindings for quantum-core functionality
//!
//! This module provides comprehensive Python bindings for all quantum computing
//! features, enabling seamless integration with existing Python trading systems.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyDict, PyTuple};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use num_complex::Complex64;
use std::collections::HashMap;
use crate::{
    QuantumState, QuantumGate, QuantumCircuit, QuantumDevice, DeviceType,
    HardwareAccelerator, HardwareConfig, AccelerationType,
    QuantumResult, QuantumError,
};

/// Python wrapper for QuantumState
#[pyclass(name = "QuantumState")]
#[derive(Clone)]
pub struct PyQuantumState {
    inner: QuantumState,
}

#[pymethods]
impl PyQuantumState {
    /// Create a new quantum state
    #[new]
    pub fn new(num_qubits: usize) -> PyResult<Self> {
        let inner = QuantumState::new(num_qubits)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create quantum state: {}", e)))?;
        Ok(Self { inner })
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Get the amplitudes as a numpy array
    pub fn get_amplitudes<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray1<Complex64>> {
        let amplitudes = self.inner.get_amplitudes();
        Ok(PyArray1::from_slice(py, amplitudes))
    }

    /// Set amplitudes from a numpy array
    pub fn set_amplitudes(&mut self, amplitudes: PyReadonlyArray1<Complex64>) -> PyResult<()> {
        let amplitudes = amplitudes.as_slice()?;
        self.inner.set_amplitudes(amplitudes.to_vec())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set amplitudes: {}", e)))?;
        Ok(())
    }

    /// Get amplitude at specific index
    pub fn get_amplitude(&self, index: usize) -> PyResult<Complex64> {
        self.inner.get_amplitude(index)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get amplitude: {}", e)))
    }

    /// Set amplitude at specific index
    pub fn set_amplitude(&mut self, index: usize, amplitude: Complex64) -> PyResult<()> {
        self.inner.set_amplitude(index, amplitude)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to set amplitude: {}", e)))?;
        Ok(())
    }

    /// Normalize the quantum state
    pub fn normalize(&mut self) -> PyResult<()> {
        self.inner.normalize()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to normalize: {}", e)))?;
        Ok(())
    }

    /// Calculate the norm of the state
    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    /// Measure the quantum state
    pub fn measure(&mut self) -> PyResult<Vec<usize>> {
        self.inner.measure()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to measure: {}", e)))
    }

    /// Get probability of measuring specific outcome
    pub fn get_probability(&self, outcome: usize) -> PyResult<f64> {
        self.inner.get_probability(outcome)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get probability: {}", e)))
    }

    /// Clone the quantum state
    pub fn clone(&self) -> Self {
        Self { inner: self.inner.clone() }
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!("QuantumState(qubits={}, norm={:.6})", self.num_qubits(), self.norm())
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Python wrapper for QuantumGate
#[pyclass(name = "QuantumGate")]
#[derive(Clone)]
pub struct PyQuantumGate {
    inner: QuantumGate,
}

#[pymethods]
impl PyQuantumGate {
    /// Create Hadamard gate
    #[staticmethod]
    pub fn hadamard(qubit: usize) -> PyResult<Self> {
        let inner = QuantumGate::hadamard(qubit)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Hadamard gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create Pauli-X gate
    #[staticmethod]
    pub fn pauli_x(qubit: usize) -> PyResult<Self> {
        let inner = QuantumGate::pauli_x(qubit)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Pauli-X gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create Pauli-Y gate
    #[staticmethod]
    pub fn pauli_y(qubit: usize) -> PyResult<Self> {
        let inner = QuantumGate::pauli_y(qubit)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Pauli-Y gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create Pauli-Z gate
    #[staticmethod]
    pub fn pauli_z(qubit: usize) -> PyResult<Self> {
        let inner = QuantumGate::pauli_z(qubit)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create Pauli-Z gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create rotation-X gate
    #[staticmethod]
    pub fn rotation_x(qubit: usize, angle: f64) -> PyResult<Self> {
        let inner = QuantumGate::rotation_x(qubit, angle)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create rotation-X gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create rotation-Y gate
    #[staticmethod]
    pub fn rotation_y(qubit: usize, angle: f64) -> PyResult<Self> {
        let inner = QuantumGate::rotation_y(qubit, angle)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create rotation-Y gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create rotation-Z gate
    #[staticmethod]
    pub fn rotation_z(qubit: usize, angle: f64) -> PyResult<Self> {
        let inner = QuantumGate::rotation_z(qubit, angle)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create rotation-Z gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create controlled-NOT gate
    #[staticmethod]
    pub fn controlled_not(control: usize, target: usize) -> PyResult<Self> {
        let inner = QuantumGate::controlled_not(control, target)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create CNOT gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Create controlled-phase gate
    #[staticmethod]
    pub fn controlled_phase(control: usize, target: usize, phase: f64) -> PyResult<Self> {
        let inner = QuantumGate::controlled_phase(control, target, phase)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create controlled-phase gate: {}", e)))?;
        Ok(Self { inner })
    }

    /// Apply gate to quantum state
    pub fn apply(&self, state: &mut PyQuantumState) -> PyResult<()> {
        self.inner.apply(&mut state.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to apply gate: {}", e)))?;
        Ok(())
    }

    /// Get gate matrix
    pub fn matrix<'py>(&self, py: Python<'py>) -> PyResult<&'py PyArray2<Complex64>> {
        let matrix = self.inner.matrix()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gate matrix: {}", e)))?;
        
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        let data: Vec<Complex64> = matrix.iter().cloned().collect();
        
        Ok(PyArray2::from_vec2(py, &vec![data; rows])?)
    }

    /// Get target qubits
    pub fn target_qubits(&self) -> Vec<usize> {
        self.inner.target_qubits()
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!("QuantumGate(targets={:?})", self.target_qubits())
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Python wrapper for QuantumCircuit
#[pyclass(name = "QuantumCircuit")]
#[derive(Clone)]
pub struct PyQuantumCircuit {
    inner: QuantumCircuit,
}

#[pymethods]
impl PyQuantumCircuit {
    /// Create new quantum circuit
    #[new]
    pub fn new(num_qubits: usize) -> Self {
        let inner = QuantumCircuit::new(num_qubits);
        Self { inner }
    }

    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: &PyQuantumGate) -> PyResult<()> {
        self.inner.add_gate(gate.inner.clone())
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to add gate: {}", e)))?;
        Ok(())
    }

    /// Execute circuit on quantum state
    pub fn execute(&self, state: &mut PyQuantumState) -> PyResult<()> {
        self.inner.execute(&mut state.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to execute circuit: {}", e)))?;
        Ok(())
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// Get number of gates
    pub fn num_gates(&self) -> usize {
        self.inner.num_gates()
    }

    /// Get circuit depth
    pub fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Clear all gates
    pub fn clear(&mut self) {
        self.inner.clear();
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!("QuantumCircuit(qubits={}, gates={}, depth={})", 
                self.num_qubits(), self.num_gates(), self.depth())
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Python wrapper for QuantumDevice
#[pyclass(name = "QuantumDevice")]
#[derive(Clone)]
pub struct PyQuantumDevice {
    inner: QuantumDevice,
}

#[pymethods]
impl PyQuantumDevice {
    /// Create new quantum device
    #[new]
    pub fn new(device_type: &str, num_qubits: usize) -> PyResult<Self> {
        let device_type = match device_type {
            "simulator" => DeviceType::Simulator,
            "quantum_hardware" => DeviceType::QuantumHardware,
            "hybrid" => DeviceType::Hybrid,
            _ => return Err(PyValueError::new_err(format!("Unknown device type: {}", device_type))),
        };

        let inner = QuantumDevice::new(device_type, num_qubits)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create quantum device: {}", e)))?;
        Ok(Self { inner })
    }

    /// Execute circuit on device
    pub fn execute_circuit(&self, circuit: &PyQuantumCircuit, state: &mut PyQuantumState) -> PyResult<()> {
        self.inner.execute_circuit(&circuit.inner, &mut state.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to execute circuit: {}", e)))?;
        Ok(())
    }

    /// Get device capabilities
    pub fn get_capabilities(&self) -> PyResult<HashMap<String, bool>> {
        self.inner.get_capabilities()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get capabilities: {}", e)))
    }

    /// Get device type
    pub fn device_type(&self) -> String {
        match self.inner.device_type() {
            DeviceType::Simulator => "simulator".to_string(),
            DeviceType::QuantumHardware => "quantum_hardware".to_string(),
            DeviceType::Hybrid => "hybrid".to_string(),
        }
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.inner.num_qubits()
    }

    /// String representation
    pub fn __str__(&self) -> String {
        format!("QuantumDevice(type={}, qubits={})", self.device_type(), self.num_qubits())
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

/// Python wrapper for HardwareAccelerator
#[pyclass(name = "HardwareAccelerator")]
pub struct PyHardwareAccelerator {
    inner: HardwareAccelerator,
}

#[pymethods]
impl PyHardwareAccelerator {
    /// Create new hardware accelerator
    #[new]
    pub fn new(config: Option<PyDict>) -> PyResult<Self> {
        let config = if let Some(config_dict) = config {
            Self::parse_hardware_config(config_dict)?
        } else {
            HardwareConfig::default()
        };

        let inner = HardwareAccelerator::new(config)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create hardware accelerator: {}", e)))?;
        Ok(Self { inner })
    }

    /// Accelerated state multiplication
    pub fn accelerated_state_multiply(&self, state: &mut PyQuantumState, gate: &PyQuantumGate) -> PyResult<()> {
        let gate_matrix = gate.inner.matrix()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get gate matrix: {}", e)))?;
        
        self.inner.accelerated_state_multiply(&mut state.inner, &gate_matrix)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to accelerate state multiply: {}", e)))?;
        Ok(())
    }

    /// Accelerated circuit execution
    pub fn accelerated_circuit_execution(&self, circuit: &PyQuantumCircuit, state: &mut PyQuantumState) -> PyResult<()> {
        self.inner.accelerated_circuit_execution(&circuit.inner, &mut state.inner)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to accelerate circuit execution: {}", e)))?;
        Ok(())
    }

    /// Get performance metrics
    pub fn get_metrics(&self) -> PyResult<HashMap<String, u64>> {
        let metrics = self.inner.get_metrics()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get metrics: {}", e)))?;
        
        let mut result = HashMap::new();
        result.insert("gpu_operations".to_string(), metrics.gpu_operations);
        result.insert("cpu_operations".to_string(), metrics.cpu_operations);
        result.insert("gpu_time_us".to_string(), metrics.gpu_time_us);
        result.insert("cpu_time_us".to_string(), metrics.cpu_time_us);
        
        Ok(result)
    }

    /// Check CUDA availability
    #[staticmethod]
    pub fn is_cuda_available() -> bool {
        HardwareAccelerator::is_cuda_available()
    }

    /// Check OpenCL availability
    #[staticmethod]
    pub fn is_opencl_available() -> bool {
        HardwareAccelerator::is_opencl_available()
    }

    /// Check ROCm availability
    #[staticmethod]
    pub fn is_rocm_available() -> bool {
        HardwareAccelerator::is_rocm_available()
    }

    /// String representation
    pub fn __str__(&self) -> String {
        "HardwareAccelerator".to_string()
    }

    /// Representation
    pub fn __repr__(&self) -> String {
        self.__str__()
    }
}

impl PyHardwareAccelerator {
    fn parse_hardware_config(config_dict: &PyDict) -> PyResult<HardwareConfig> {
        let mut config = HardwareConfig::default();

        if let Some(acceleration_type) = config_dict.get_item("acceleration_type") {
            let accel_type_str = acceleration_type.extract::<String>()?;
            config.acceleration_type = match accel_type_str.as_str() {
                "cpu" => AccelerationType::CPU,
                "cuda" => AccelerationType::CUDA,
                "opencl" => AccelerationType::OpenCL,
                "rocm" => AccelerationType::ROCm,
                "oneapi" => AccelerationType::OneAPI,
                "metal" => AccelerationType::Metal,
                _ => return Err(PyValueError::new_err(format!("Unknown acceleration type: {}", accel_type_str))),
            };
        }

        if let Some(device_id) = config_dict.get_item("device_id") {
            config.device_id = device_id.extract::<usize>()?;
        }

        if let Some(num_threads) = config_dict.get_item("num_threads") {
            config.num_threads = num_threads.extract::<usize>()?;
        }

        if let Some(memory_limit_mb) = config_dict.get_item("memory_limit_mb") {
            config.memory_limit_mb = memory_limit_mb.extract::<usize>()?;
        }

        if let Some(enable_fallback) = config_dict.get_item("enable_fallback") {
            config.enable_fallback = enable_fallback.extract::<bool>()?;
        }

        if let Some(batch_size) = config_dict.get_item("batch_size") {
            config.batch_size = batch_size.extract::<usize>()?;
        }

        if let Some(enable_tensor_cores) = config_dict.get_item("enable_tensor_cores") {
            config.enable_tensor_cores = enable_tensor_cores.extract::<bool>()?;
        }

        if let Some(enable_fast_math) = config_dict.get_item("enable_fast_math") {
            config.enable_fast_math = enable_fast_math.extract::<bool>()?;
        }

        Ok(config)
    }
}

/// Quantum computing utility functions
#[pyfunction]
pub fn create_bell_state(py: Python) -> PyResult<PyQuantumState> {
    let mut state = PyQuantumState::new(2)?;
    
    // Apply Hadamard to first qubit
    let h_gate = PyQuantumGate::hadamard(0)?;
    h_gate.apply(&mut state)?;
    
    // Apply CNOT
    let cnot_gate = PyQuantumGate::controlled_not(0, 1)?;
    cnot_gate.apply(&mut state)?;
    
    Ok(state)
}

/// Create GHZ state
#[pyfunction]
pub fn create_ghz_state(num_qubits: usize) -> PyResult<PyQuantumState> {
    let mut state = PyQuantumState::new(num_qubits)?;
    
    // Apply Hadamard to first qubit
    let h_gate = PyQuantumGate::hadamard(0)?;
    h_gate.apply(&mut state)?;
    
    // Apply CNOT from first qubit to all others
    for i in 1..num_qubits {
        let cnot_gate = PyQuantumGate::controlled_not(0, i)?;
        cnot_gate.apply(&mut state)?;
    }
    
    Ok(state)
}

/// Create W state
#[pyfunction]
pub fn create_w_state(num_qubits: usize) -> PyResult<PyQuantumState> {
    let mut state = PyQuantumState::new(num_qubits)?;
    
    // Initialize W state manually
    let amplitude = Complex64::new(1.0 / (num_qubits as f64).sqrt(), 0.0);
    
    // Set amplitudes for W state
    for i in 0..num_qubits {
        let index = 1 << i; // 2^i
        state.set_amplitude(index, amplitude)?;
    }
    
    Ok(state)
}

/// Quantum random number generation
#[pyfunction]
pub fn quantum_random(num_bits: usize) -> PyResult<Vec<bool>> {
    let mut state = PyQuantumState::new(num_bits)?;
    
    // Apply Hadamard to all qubits
    for i in 0..num_bits {
        let h_gate = PyQuantumGate::hadamard(i)?;
        h_gate.apply(&mut state)?;
    }
    
    // Measure all qubits
    let measurement = state.measure()?;
    
    // Convert to boolean array
    let mut result = vec![false; num_bits];
    for outcome in measurement {
        for i in 0..num_bits {
            if (outcome >> i) & 1 == 1 {
                result[i] = true;
            }
        }
    }
    
    Ok(result)
}

/// Quantum Fourier Transform
#[pyfunction]
pub fn quantum_fourier_transform(state: &mut PyQuantumState) -> PyResult<()> {
    let num_qubits = state.num_qubits();
    
    for i in 0..num_qubits {
        // Apply Hadamard
        let h_gate = PyQuantumGate::hadamard(i)?;
        h_gate.apply(state)?;
        
        // Apply controlled phase rotations
        for j in (i + 1)..num_qubits {
            let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
            let cp_gate = PyQuantumGate::controlled_phase(j, i, angle)?;
            cp_gate.apply(state)?;
        }
    }
    
    // Reverse qubit order (swap gates)
    for i in 0..(num_qubits / 2) {
        let j = num_qubits - 1 - i;
        
        // Implement SWAP using three CNOTs
        let cnot1 = PyQuantumGate::controlled_not(i, j)?;
        let cnot2 = PyQuantumGate::controlled_not(j, i)?;
        let cnot3 = PyQuantumGate::controlled_not(i, j)?;
        
        cnot1.apply(state)?;
        cnot2.apply(state)?;
        cnot3.apply(state)?;
    }
    
    Ok(())
}

/// Quantum phase estimation
#[pyfunction]
pub fn quantum_phase_estimation(
    eigenstate: &mut PyQuantumState,
    unitary: &PyQuantumGate,
    precision_qubits: usize,
) -> PyResult<f64> {
    let total_qubits = eigenstate.num_qubits() + precision_qubits;
    let mut full_state = PyQuantumState::new(total_qubits)?;
    
    // Initialize with eigenstate in the last qubits
    // (This is a simplified version - full implementation would be more complex)
    
    // Apply Hadamard to precision qubits
    for i in 0..precision_qubits {
        let h_gate = PyQuantumGate::hadamard(i)?;
        h_gate.apply(&mut full_state)?;
    }
    
    // Apply controlled unitary operations
    for i in 0..precision_qubits {
        for _ in 0..(1 << i) {
            // Apply controlled unitary (simplified)
            unitary.apply(&mut full_state)?;
        }
    }
    
    // Apply inverse QFT to precision qubits
    quantum_fourier_transform(&mut full_state)?;
    
    // Measure precision qubits
    let measurement = full_state.measure()?;
    
    // Extract phase estimate
    let phase = measurement[0] as f64 / (1 << precision_qubits) as f64;
    
    Ok(phase * 2.0 * std::f64::consts::PI)
}

/// Get quantum module version
#[pyfunction]
pub fn get_version() -> String {
    crate::VERSION.to_string()
}

/// Initialize quantum module
#[pyfunction]
pub fn initialize_quantum() -> PyResult<()> {
    crate::initialize()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to initialize quantum module: {}", e)))
}

/// Python module definition
#[pymodule]
fn quantum_core(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyQuantumState>()?;
    m.add_class::<PyQuantumGate>()?;
    m.add_class::<PyQuantumCircuit>()?;
    m.add_class::<PyQuantumDevice>()?;
    m.add_class::<PyHardwareAccelerator>()?;
    
    m.add_function(wrap_pyfunction!(create_bell_state, m)?)?;
    m.add_function(wrap_pyfunction!(create_ghz_state, m)?)?;
    m.add_function(wrap_pyfunction!(create_w_state, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_random, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_fourier_transform, m)?)?;
    m.add_function(wrap_pyfunction!(quantum_phase_estimation, m)?)?;
    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(initialize_quantum, m)?)?;
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::Python;

    #[test]
    fn test_python_quantum_state() {
        Python::with_gil(|py| {
            let state = PyQuantumState::new(2).unwrap();
            assert_eq!(state.num_qubits(), 2);
            assert!((state.norm() - 1.0).abs() < 1e-10);
        });
    }

    #[test]
    fn test_python_quantum_gate() {
        Python::with_gil(|py| {
            let gate = PyQuantumGate::hadamard(0).unwrap();
            let mut state = PyQuantumState::new(1).unwrap();
            
            gate.apply(&mut state).unwrap();
            assert!((state.norm() - 1.0).abs() < 1e-10);
        });
    }

    #[test]
    fn test_python_quantum_circuit() {
        Python::with_gil(|py| {
            let mut circuit = PyQuantumCircuit::new(2);
            let h_gate = PyQuantumGate::hadamard(0).unwrap();
            let cnot_gate = PyQuantumGate::controlled_not(0, 1).unwrap();
            
            circuit.add_gate(&h_gate).unwrap();
            circuit.add_gate(&cnot_gate).unwrap();
            
            assert_eq!(circuit.num_qubits(), 2);
            assert_eq!(circuit.num_gates(), 2);
        });
    }

    #[test]
    fn test_bell_state_creation() {
        Python::with_gil(|py| {
            let state = create_bell_state(py).unwrap();
            assert_eq!(state.num_qubits(), 2);
            assert!((state.norm() - 1.0).abs() < 1e-10);
        });
    }

    #[test]
    fn test_hardware_accelerator() {
        Python::with_gil(|py| {
            let accelerator = PyHardwareAccelerator::new(None).unwrap();
            let metrics = accelerator.get_metrics().unwrap();
            
            // Should have initial metrics
            assert!(metrics.contains_key("gpu_operations"));
            assert!(metrics.contains_key("cpu_operations"));
        });
    }
}