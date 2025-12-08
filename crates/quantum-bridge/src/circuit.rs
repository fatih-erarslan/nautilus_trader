//! # Quantum Circuit Design and Compilation
//!
//! Quantum circuit representation, compilation, and optimization for PennyLane execution.
//! Enforces zero-mock policy with real quantum circuit operations.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use nalgebra::{DVector, DMatrix, Complex};
use serde::{Serialize, Deserialize};
use tracing::{debug, info, warn, instrument};

use crate::device::{QuantumDevice, DeviceCapabilities};
use crate::bridge::PythonRuntime;
use crate::error::{QuantumError, CircuitError};
use crate::types::{CircuitId, QubitId, ParameterId, GateId};

/// Quantum circuit representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    /// Unique circuit identifier
    id: CircuitId,
    /// Number of qubits
    num_qubits: u32,
    /// Quantum gates in execution order
    gates: Vec<QuantumGate>,
    /// Circuit parameters (for parameterized circuits)
    parameters: HashMap<ParameterId, f64>,
    /// Classical registers for measurements
    classical_registers: Vec<ClassicalRegister>,
    /// Circuit metadata
    metadata: CircuitMetadata,
}

/// Quantum gate operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGate {
    /// Single-qubit gates
    Identity { qubit: QubitId },
    PauliX { qubit: QubitId },
    PauliY { qubit: QubitId },
    PauliZ { qubit: QubitId },
    Hadamard { qubit: QubitId },
    S { qubit: QubitId },
    T { qubit: QubitId },
    
    /// Parameterized single-qubit rotations
    RX { qubit: QubitId, angle: f64 },
    RY { qubit: QubitId, angle: f64 },
    RZ { qubit: QubitId, angle: f64 },
    Phase { qubit: QubitId, angle: f64 },
    
    /// Two-qubit gates
    CNOT { control: QubitId, target: QubitId },
    CZ { control: QubitId, target: QubitId },
    CY { control: QubitId, target: QubitId },
    SWAP { qubit1: QubitId, qubit2: QubitId },
    
    /// Parameterized two-qubit gates
    CRX { control: QubitId, target: QubitId, angle: f64 },
    CRY { control: QubitId, target: QubitId, angle: f64 },
    CRZ { control: QubitId, target: QubitId, angle: f64 },
    
    /// Multi-qubit gates
    Toffoli { control1: QubitId, control2: QubitId, target: QubitId },
    Fredkin { control: QubitId, target1: QubitId, target2: QubitId },
    
    /// Measurement operations
    Measure { qubit: QubitId, classical_bit: u32 },
    MeasureAll,
    
    /// Custom parameterized gates
    Parameterized {
        gate_type: String,
        qubits: Vec<QubitId>,
        parameters: Vec<ParameterId>,
    },
    
    /// Barrier (for optimization boundaries)
    Barrier { qubits: Vec<QubitId> },
}

/// Classical register for measurement results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassicalRegister {
    /// Register name
    pub name: String,
    /// Number of classical bits
    pub size: u32,
}

/// Circuit metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitMetadata {
    /// Circuit name/description
    pub name: String,
    /// Creation timestamp
    pub created_at: chrono::DateTime<chrono::Utc>,
    /// Circuit depth (longest path)
    pub depth: u32,
    /// Total gate count
    pub gate_count: u32,
    /// Two-qubit gate count (for complexity estimation)
    pub two_qubit_gate_count: u32,
    /// Estimated execution time
    pub estimated_execution_time_ms: f64,
}

/// Circuit builder for fluent construction
pub struct CircuitBuilder {
    circuit: QuantumCircuit,
    next_gate_id: GateId,
    next_parameter_id: ParameterId,
}

/// Circuit compiler for PennyLane integration
pub struct CircuitCompiler {
    python_runtime: Arc<PythonRuntime>,
    optimization_level: OptimizationLevel,
    gate_cache: HashMap<String, PyObject>,
}

/// Circuit optimization levels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    /// No optimization
    None,
    /// Basic gate simplification
    Basic,
    /// Advanced optimization with gate fusion
    Advanced,
    /// Maximum optimization including device-specific optimizations
    Maximum,
}

/// Compiled circuit for execution
#[derive(Debug)]
pub struct CompiledCircuit {
    /// Original circuit ID
    pub circuit_id: CircuitId,
    /// Compiled PennyLane circuit function
    pub pennylane_circuit: PyObject,
    /// Target device
    pub target_device: QuantumDevice,
    /// Compilation parameters
    pub compilation_params: CompilationParams,
    /// Estimated execution time
    pub estimated_execution_time: std::time::Duration,
}

impl Clone for CompiledCircuit {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            Self {
                circuit_id: self.circuit_id,
                pennylane_circuit: self.pennylane_circuit.clone_ref(py),
                target_device: self.target_device.clone(),
                compilation_params: self.compilation_params.clone(),
                estimated_execution_time: self.estimated_execution_time,
            }
        })
    }
}

/// Circuit compilation parameters
#[derive(Debug, Clone)]
pub struct CompilationParams {
    /// Optimization level used
    pub optimization_level: OptimizationLevel,
    /// Number of shots for execution
    pub shots: u32,
    /// Device-specific optimizations applied
    pub device_optimizations: Vec<String>,
    /// Gate decomposition strategy
    pub decomposition_strategy: DecompositionStrategy,
}

/// Gate decomposition strategies
#[derive(Debug, Clone, Copy)]
pub enum DecompositionStrategy {
    /// Minimal decomposition
    Minimal,
    /// Native gate set optimization
    NativeGateSet,
    /// Error mitigation focused
    ErrorMitigation,
    /// Depth optimization
    DepthOptimal,
}

impl QuantumCircuit {
    /// Create new empty circuit
    pub fn new(num_qubits: u32) -> Self {
        Self {
            id: CircuitId::new(),
            num_qubits,
            gates: Vec::new(),
            parameters: HashMap::new(),
            classical_registers: vec![ClassicalRegister {
                name: "c".to_string(),
                size: num_qubits,
            }],
            metadata: CircuitMetadata {
                name: format!("Circuit_{}", num_qubits),
                created_at: chrono::Utc::now(),
                depth: 0,
                gate_count: 0,
                two_qubit_gate_count: 0,
                estimated_execution_time_ms: 0.0,
            },
        }
    }
    
    /// Get circuit ID
    pub fn id(&self) -> CircuitId {
        self.id
    }
    
    /// Get number of qubits
    pub fn qubit_count(&self) -> u32 {
        self.num_qubits
    }
    
    /// Get number of gates
    pub fn gate_count(&self) -> u32 {
        self.gates.len() as u32
    }
    
    /// Get circuit depth
    pub fn depth(&self) -> u32 {
        self.calculate_depth()
    }
    
    /// Get gates
    pub fn gates(&self) -> &[QuantumGate] {
        &self.gates
    }
    
    /// Add gate to circuit
    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
        self.update_metadata();
    }
    
    /// Set parameter value
    pub fn set_parameter(&mut self, param_id: ParameterId, value: f64) {
        self.parameters.insert(param_id, value);
    }
    
    /// Calculate circuit depth (critical path length)
    fn calculate_depth(&self) -> u32 {
        let mut qubit_depths = vec![0u32; self.num_qubits as usize];
        
        for gate in &self.gates {
            let involved_qubits = gate.involved_qubits();
            let max_depth = involved_qubits.iter()
                .map(|&q| qubit_depths[q as usize])
                .max()
                .unwrap_or(0);
            
            for &qubit in &involved_qubits {
                qubit_depths[qubit as usize] = max_depth + 1;
            }
        }
        
        qubit_depths.into_iter().max().unwrap_or(0)
    }
    
    /// Update circuit metadata
    fn update_metadata(&mut self) {
        self.metadata.gate_count = self.gates.len() as u32;
        self.metadata.depth = self.calculate_depth();
        self.metadata.two_qubit_gate_count = self.gates.iter()
            .filter(|gate| gate.is_two_qubit())
            .count() as u32;
        
        // Estimate execution time (rough approximation)
        self.metadata.estimated_execution_time_ms = 
            (self.metadata.gate_count as f64 * 0.1) + 
            (self.metadata.two_qubit_gate_count as f64 * 0.5);
    }
    
    /// Validate circuit consistency
    pub fn validate(&self) -> Result<(), CircuitError> {
        // Check qubit indices
        for gate in &self.gates {
            for &qubit in &gate.involved_qubits() {
                if qubit >= self.num_qubits {
                    return Err(CircuitError::InvalidQubitIndex {
                        qubit,
                        max_qubit: self.num_qubits - 1,
                    });
                }
            }
        }
        
        // Check parameter references
        for gate in &self.gates {
            if let QuantumGate::Parameterized { parameters, .. } = gate {
                for &param_id in parameters {
                    if !self.parameters.contains_key(&param_id) {
                        return Err(CircuitError::UndefinedParameter(param_id));
                    }
                }
            }
        }
        
        Ok(())
    }
}

impl QuantumGate {
    /// Get qubits involved in this gate
    pub fn involved_qubits(&self) -> Vec<QubitId> {
        match self {
            Self::Identity { qubit } |
            Self::PauliX { qubit } |
            Self::PauliY { qubit } |
            Self::PauliZ { qubit } |
            Self::Hadamard { qubit } |
            Self::S { qubit } |
            Self::T { qubit } |
            Self::RX { qubit, .. } |
            Self::RY { qubit, .. } |
            Self::RZ { qubit, .. } |
            Self::Phase { qubit, .. } => vec![*qubit],
            
            Self::CNOT { control, target } |
            Self::CZ { control, target } |
            Self::CY { control, target } |
            Self::SWAP { qubit1: control, qubit2: target } |
            Self::CRX { control, target, .. } |
            Self::CRY { control, target, .. } |
            Self::CRZ { control, target, .. } => vec![*control, *target],
            
            Self::Toffoli { control1, control2, target } => vec![*control1, *control2, *target],
            Self::Fredkin { control, target1, target2 } => vec![*control, *target1, *target2],
            
            Self::Measure { qubit, .. } => vec![*qubit],
            Self::MeasureAll => (0..32).collect(), // Placeholder, should use actual qubit count
            
            Self::Parameterized { qubits, .. } => qubits.clone(),
            Self::Barrier { qubits } => qubits.clone(),
        }
    }
    
    /// Check if this is a two-qubit gate
    pub fn is_two_qubit(&self) -> bool {
        matches!(self,
            Self::CNOT { .. } |
            Self::CZ { .. } |
            Self::CY { .. } |
            Self::SWAP { .. } |
            Self::CRX { .. } |
            Self::CRY { .. } |
            Self::CRZ { .. }
        )
    }
    
    /// Check if this gate has parameters
    pub fn is_parameterized(&self) -> bool {
        matches!(self,
            Self::RX { .. } |
            Self::RY { .. } |
            Self::RZ { .. } |
            Self::Phase { .. } |
            Self::CRX { .. } |
            Self::CRY { .. } |
            Self::CRZ { .. } |
            Self::Parameterized { .. }
        )
    }
}

impl CircuitBuilder {
    /// Create new circuit builder
    pub fn new(num_qubits: u32) -> Self {
        Self {
            circuit: QuantumCircuit::new(num_qubits),
            next_gate_id: 0,
            next_parameter_id: 0,
        }
    }
    
    /// Add Hadamard gate
    pub fn h(&mut self, qubit: QubitId) -> &mut Self {
        self.circuit.add_gate(QuantumGate::Hadamard { qubit });
        self
    }
    
    /// Add Pauli-X gate
    pub fn x(&mut self, qubit: QubitId) -> &mut Self {
        self.circuit.add_gate(QuantumGate::PauliX { qubit });
        self
    }
    
    /// Add Pauli-Y gate
    pub fn y(&mut self, qubit: QubitId) -> &mut Self {
        self.circuit.add_gate(QuantumGate::PauliY { qubit });
        self
    }
    
    /// Add Pauli-Z gate
    pub fn z(&mut self, qubit: QubitId) -> &mut Self {
        self.circuit.add_gate(QuantumGate::PauliZ { qubit });
        self
    }
    
    /// Add CNOT gate
    pub fn cnot(&mut self, control: QubitId, target: QubitId) -> &mut Self {
        self.circuit.add_gate(QuantumGate::CNOT { control, target });
        self
    }
    
    /// Add rotation-X gate
    pub fn rx(&mut self, qubit: QubitId, angle: f64) -> &mut Self {
        self.circuit.add_gate(QuantumGate::RX { qubit, angle });
        self
    }
    
    /// Add rotation-Y gate
    pub fn ry(&mut self, qubit: QubitId, angle: f64) -> &mut Self {
        self.circuit.add_gate(QuantumGate::RY { qubit, angle });
        self
    }
    
    /// Add rotation-Z gate
    pub fn rz(&mut self, qubit: QubitId, angle: f64) -> &mut Self {
        self.circuit.add_gate(QuantumGate::RZ { qubit, angle });
        self
    }
    
    /// Add measurement of all qubits
    pub fn measure_all(&mut self) -> &mut Self {
        self.circuit.add_gate(QuantumGate::MeasureAll);
        self
    }
    
    /// Add arbitrary gate
    pub fn add_gate(&mut self, gate: QuantumGate) -> &mut Self {
        self.circuit.add_gate(gate);
        self
    }
    
    /// Build the final circuit
    pub fn build(self) -> QuantumCircuit {
        self.circuit
    }
}

impl CircuitCompiler {
    /// Create new circuit compiler
    pub fn new(python_runtime: &PythonRuntime) -> Self {
        Self {
            python_runtime: Arc::new(python_runtime.clone()),
            optimization_level: OptimizationLevel::Advanced,
            gate_cache: HashMap::new(),
        }
    }
    
    /// Compile circuit for specific device
    #[instrument(skip(self, circuit, device))]
    pub async fn compile(
        &self,
        circuit: &QuantumCircuit,
        device: &QuantumDevice,
    ) -> Result<CompiledCircuit, QuantumError> {
        debug!("Compiling circuit {} for device {:?}", circuit.id(), device);
        
        // Validate circuit
        circuit.validate()
            .map_err(|e| QuantumError::CircuitCompilation(e.to_string()))?;
        
        // Compile to PennyLane
        let pennylane_circuit = self.compile_to_pennylane(circuit, device).await?;
        
        let compilation_params = CompilationParams {
            optimization_level: self.optimization_level,
            shots: 1024, // Default shots
            device_optimizations: self.get_device_optimizations(device),
            decomposition_strategy: DecompositionStrategy::NativeGateSet,
        };
        
        // Estimate execution time
        let estimated_execution_time = self.estimate_execution_time(circuit, device);
        
        debug!("Circuit compilation completed successfully");
        
        Ok(CompiledCircuit {
            circuit_id: circuit.id(),
            pennylane_circuit,
            target_device: *device,
            compilation_params,
            estimated_execution_time,
        })
    }
    
    /// Compile circuit to PennyLane representation
    async fn compile_to_pennylane(
        &self,
        circuit: &QuantumCircuit,
        device: &QuantumDevice,
    ) -> Result<PyObject, QuantumError> {
        Python::with_gil(|py| -> Result<PyObject, QuantumError> {
            // Create PennyLane device
            let device_obj = self.create_pennylane_device(py, device)?;
            
            // Create quantum function
            let qfunc = self.create_quantum_function(py, circuit)?;
            
            // Create QNode (compiled quantum circuit)
            let qnode = self.python_runtime.pennylane
                .call_method1("qnode", (qfunc, device_obj))?;
            
            Ok(qnode.to_object(py))
        }).map_err(|e| QuantumError::PythonExecution(e.to_string()))
    }
    
    /// Create PennyLane device object
    fn create_pennylane_device(&self, py: Python, device: &QuantumDevice) -> Result<PyObject, QuantumError> {
        let (device_name, device_kwargs) = match device {
            QuantumDevice::LightningGpu { device_id, .. } => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("device_id", device_id)?;
                ("lightning.gpu", kwargs)
            }
            QuantumDevice::LightningKokkos { backend, threads, .. } => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("kokkos_args", format!("{:?}", backend))?;
                kwargs.set_item("num_threads", threads)?;
                ("lightning.kokkos", kwargs)
            }
            QuantumDevice::LightningQubit { threads, .. } => {
                let kwargs = PyDict::new(py);
                kwargs.set_item("num_threads", threads)?;
                ("lightning.qubit", kwargs)
            }
        };
        
        let device = self.python_runtime.pennylane
            .call_method("device", (device_name, py.None()), Some(device_kwargs))?;
        
        Ok(device.to_object(py))
    }
    
    /// Create quantum function from circuit
    fn create_quantum_function(&self, py: Python, circuit: &QuantumCircuit) -> Result<PyObject, QuantumError> {
        // Create Python function that implements the quantum circuit
        let func_code = self.generate_pennylane_code(circuit)?;
        
        // Execute the function definition
        let locals = PyDict::new(py);
        locals.set_item("pennylane", self.python_runtime.pennylane)?;
        locals.set_item("numpy", self.python_runtime.numpy)?;
        
        py.run(&func_code, None, Some(&locals))?;
        
        // Get the quantum function
        let qfunc = locals.get_item("quantum_circuit")?
            .ok_or_else(|| QuantumError::CircuitCompilation("Failed to create quantum function".to_string()))?;
        
        Ok(qfunc.to_object(py))
    }
    
    /// Generate PennyLane code for circuit
    fn generate_pennylane_code(&self, circuit: &QuantumCircuit) -> Result<String, QuantumError> {
        let mut code = String::new();
        
        // Function header
        code.push_str("def quantum_circuit():\n");
        
        // Add gates
        for gate in circuit.gates() {
            let gate_code = self.gate_to_pennylane_code(gate)?;
            code.push_str(&format!("    {}\n", gate_code));
        }
        
        // Return measurements or expectation values
        if circuit.gates().iter().any(|g| matches!(g, QuantumGate::MeasureAll | QuantumGate::Measure { .. })) {
            code.push_str("    return pennylane.sample()\n");
        } else {
            code.push_str("    return [pennylane.expval(pennylane.PauliZ(i)) for i in range({})]\n");
        }
        
        Ok(code)
    }
    
    /// Convert gate to PennyLane code
    fn gate_to_pennylane_code(&self, gate: &QuantumGate) -> Result<String, QuantumError> {
        let code = match gate {
            QuantumGate::Hadamard { qubit } => format!("pennylane.Hadamard({})", qubit),
            QuantumGate::PauliX { qubit } => format!("pennylane.PauliX({})", qubit),
            QuantumGate::PauliY { qubit } => format!("pennylane.PauliY({})", qubit),
            QuantumGate::PauliZ { qubit } => format!("pennylane.PauliZ({})", qubit),
            QuantumGate::RX { qubit, angle } => format!("pennylane.RX({}, {})", angle, qubit),
            QuantumGate::RY { qubit, angle } => format!("pennylane.RY({}, {})", angle, qubit),
            QuantumGate::RZ { qubit, angle } => format!("pennylane.RZ({}, {})", angle, qubit),
            QuantumGate::CNOT { control, target } => format!("pennylane.CNOT([{}, {}])", control, target),
            QuantumGate::CZ { control, target } => format!("pennylane.CZ([{}, {}])", control, target),
            QuantumGate::SWAP { qubit1, qubit2 } => format!("pennylane.SWAP([{}, {}])", qubit1, qubit2),
            QuantumGate::MeasureAll => "# Measurement handled in return statement".to_string(),
            QuantumGate::Measure { .. } => "# Individual measurements handled in return statement".to_string(),
            _ => return Err(QuantumError::UnsupportedGate(format!("{:?}", gate))),
        };
        
        Ok(code)
    }
    
    /// Get device-specific optimizations
    fn get_device_optimizations(&self, device: &QuantumDevice) -> Vec<String> {
        match device {
            QuantumDevice::LightningGpu { .. } => vec![
                "gpu_acceleration".to_string(),
                "cuda_optimization".to_string(),
                "tensor_network".to_string(),
            ],
            QuantumDevice::LightningKokkos { .. } => vec![
                "kokkos_parallelization".to_string(),
                "vectorization".to_string(),
            ],
            QuantumDevice::LightningQubit { .. } => vec![
                "cpu_optimization".to_string(),
                "simd_vectorization".to_string(),
            ],
        }
    }
    
    /// Estimate execution time for circuit on device
    fn estimate_execution_time(&self, circuit: &QuantumCircuit, device: &QuantumDevice) -> std::time::Duration {
        let base_time_ns = match device {
            QuantumDevice::LightningGpu { .. } => 1_000_000, // 1ms base
            QuantumDevice::LightningKokkos { .. } => 5_000_000, // 5ms base
            QuantumDevice::LightningQubit { .. } => 10_000_000, // 10ms base
        };
        
        let gate_overhead_ns = circuit.gate_count() as u64 * 1000; // 1μs per gate
        let qubit_overhead_ns = (circuit.qubit_count() as u64).pow(2) * 100; // Quadratic scaling
        
        std::time::Duration::from_nanos(base_time_ns + gate_overhead_ns + qubit_overhead_ns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_circuit_builder() {
        let mut builder = CircuitBuilder::new(3);
        builder
            .h(0)
            .cnot(0, 1)
            .cnot(1, 2)
            .measure_all();
        
        let circuit = builder.build();
        
        assert_eq!(circuit.qubit_count(), 3);
        assert_eq!(circuit.gate_count(), 4);
        assert!(circuit.depth() > 0);
    }
    
    #[test]
    fn test_circuit_validation() {
        let mut circuit = QuantumCircuit::new(2);
        circuit.add_gate(QuantumGate::Hadamard { qubit: 0 });
        circuit.add_gate(QuantumGate::CNOT { control: 0, target: 1 });
        
        assert!(circuit.validate().is_ok());
        
        // Test invalid qubit index
        circuit.add_gate(QuantumGate::PauliX { qubit: 5 });
        assert!(circuit.validate().is_err());
    }
    
    #[test]
    fn test_gate_properties() {
        let hadamard = QuantumGate::Hadamard { qubit: 0 };
        let cnot = QuantumGate::CNOT { control: 0, target: 1 };
        let rx = QuantumGate::RX { qubit: 0, angle: std::f64::consts::PI };
        
        assert_eq!(hadamard.involved_qubits(), vec![0]);
        assert_eq!(cnot.involved_qubits(), vec![0, 1]);
        
        assert!(!hadamard.is_two_qubit());
        assert!(cnot.is_two_qubit());
        
        assert!(!hadamard.is_parameterized());
        assert!(rx.is_parameterized());
    }
}