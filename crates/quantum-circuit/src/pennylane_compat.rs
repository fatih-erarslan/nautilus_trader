//! PennyLane-compatible API for quantum circuits
//!
//! This module provides a PennyLane-inspired API for quantum circuit construction
//! and execution, making it familiar for users coming from PennyLane.

use crate::{
    Circuit, Result, QuantumError, StateVector, Operator,
    // Removed unused gates import
    simulation::Simulator,
    optimization::{VariationalOptimizer, OptimizationResult},
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Quantum device trait (PennyLane-compatible)
pub trait Device: Send + Sync {
    /// Execute a quantum circuit on the device
    fn execute(&mut self, circuit: &QNode, shots: Option<usize>) -> Result<DeviceResult>;
    
    /// Get device capabilities
    fn capabilities(&self) -> DeviceCapabilities;
    
    /// Get device name
    fn name(&self) -> &str;
    
    /// Get number of qubits supported
    fn num_qubits(&self) -> usize;
}

/// Device execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceResult {
    /// Final quantum state (if available)
    pub state: Option<StateVector>,
    /// Measurement samples
    pub samples: Option<Vec<Vec<u8>>>,
    /// Expectation values
    pub expectations: HashMap<String, f64>,
    /// Execution metadata
    pub metadata: ExecutionMetadata,
}

/// Device capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceCapabilities {
    /// Maximum number of qubits
    pub max_qubits: usize,
    /// Supported gates
    pub supported_gates: Vec<String>,
    /// Supports state vector simulation
    pub supports_state: bool,
    /// Supports sampling
    pub supports_sampling: bool,
    /// Supports expectation values
    pub supports_expectations: bool,
}

/// Execution metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of shots performed
    pub shots: Option<usize>,
    /// Circuit depth
    pub circuit_depth: usize,
    /// Gate count
    pub gate_count: usize,
}

/// Default quantum simulator device
pub struct DefaultQubitDevice {
    name: String,
    num_qubits: usize,
    simulator: Arc<Mutex<Simulator>>,
}

impl DefaultQubitDevice {
    /// Create a new default qubit device
    pub fn new(num_qubits: usize) -> Self {
        Self {
            name: "default.qubit".to_string(),
            num_qubits,
            simulator: Arc::new(Mutex::new(Simulator::new(num_qubits))),
        }
    }
}

unsafe impl Send for DefaultQubitDevice {}
unsafe impl Sync for DefaultQubitDevice {}

impl Device for DefaultQubitDevice {
    fn execute(&mut self, qnode: &QNode, shots: Option<usize>) -> Result<DeviceResult> {
        let start_time = std::time::Instant::now();
        let mut simulator = self.simulator.lock().unwrap();
        
        // Build circuit from QNode
        let circuit = qnode.build_circuit()?;
        
        // Execute circuit
        simulator.reset();
        simulator.execute_circuit(&circuit)?;
        
        let execution_time = start_time.elapsed().as_secs_f64() * 1000.0;
        
        let mut result = DeviceResult {
            state: Some(simulator.state().clone()),
            samples: None,
            expectations: HashMap::new(),
            metadata: ExecutionMetadata {
                execution_time_ms: execution_time,
                shots,
                circuit_depth: circuit.depth(),
                gate_count: circuit.gate_count(),
            },
        };
        
        // Generate samples if requested
        if let Some(n_shots) = shots {
            let samples = simulator.sample_measurements(n_shots)?;
            result.samples = Some(samples);
        }
        
        // Compute expectation values for observables in QNode
        for (name, observable) in &qnode.observables {
            let expectation = simulator.expectation_value(observable)?;
            result.expectations.insert(name.clone(), expectation);
        }
        
        Ok(result)
    }
    
    fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            max_qubits: self.num_qubits,
            supported_gates: vec![
                "PauliX".to_string(),
                "PauliY".to_string(),
                "PauliZ".to_string(),
                "Hadamard".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "CNOT".to_string(),
                "CZ".to_string(),
                "CRX".to_string(),
            ],
            supports_state: true,
            supports_sampling: true,
            supports_expectations: true,
        }
    }
    
    fn name(&self) -> &str {
        &self.name
    }
    
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

/// Quantum node (QNode) - PennyLane-compatible quantum function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QNode {
    /// Number of qubits
    pub num_qubits: usize,
    /// Circuit operations
    pub operations: Vec<Operation>,
    /// Observables to measure
    pub observables: HashMap<String, Operator>,
    /// Device name
    pub device_name: String,
    /// Interface type (auto, numpy, etc.)
    pub interface: String,
    /// Differentiation method
    pub diff_method: String,
}

/// Quantum operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Operation {
    /// Gate name
    pub name: String,
    /// Target wires/qubits
    pub wires: Vec<usize>,
    /// Parameters
    pub parameters: Vec<f64>,
    /// Hyperparameters
    pub hyperparameters: HashMap<String, String>,
}

impl QNode {
    /// Create a new QNode
    pub fn new(num_qubits: usize, device_name: String) -> Self {
        Self {
            num_qubits,
            operations: Vec::new(),
            observables: HashMap::new(),
            device_name,
            interface: "auto".to_string(),
            diff_method: "parameter-shift".to_string(),
        }
    }
    
    /// Add an operation to the QNode
    pub fn add_operation(&mut self, operation: Operation) -> Result<()> {
        // Validate wires
        for &wire in &operation.wires {
            if wire >= self.num_qubits {
                return Err(QuantumError::InvalidQubit(wire));
            }
        }
        
        self.operations.push(operation);
        Ok(())
    }
    
    /// Add an observable
    pub fn add_observable(&mut self, name: String, observable: Operator) -> Result<()> {
        let expected_dim = 1 << self.num_qubits;
        if observable.nrows() != expected_dim || observable.ncols() != expected_dim {
            return Err(QuantumError::DimensionMismatch {
                expected: expected_dim,
                actual: observable.nrows(),
            });
        }
        
        self.observables.insert(name, observable);
        Ok(())
    }
    
    /// Build a Circuit from the QNode operations
    pub fn build_circuit(&self) -> Result<Circuit> {
        let mut circuit = Circuit::new(self.num_qubits);
        
        for operation in &self.operations {
            let gate = crate::gates::create_gate(&operation.name, operation.wires.clone(), operation.parameters.clone())?;
            circuit.add_gate(gate)?;
        }
        
        Ok(circuit)
    }
    
    /// Execute the QNode with parameters
    pub fn execute(&self, device: &mut dyn Device, parameters: Option<&[f64]>) -> Result<DeviceResult> {
        // Update parameters if provided
        let mut qnode = self.clone();
        if let Some(params) = parameters {
            qnode.update_parameters(params)?;
        }
        
        device.execute(&qnode, None)
    }
    
    /// Update QNode parameters
    fn update_parameters(&mut self, parameters: &[f64]) -> Result<()> {
        let mut param_idx = 0;
        
        for operation in &mut self.operations {
            let expected_params = operation.parameters.len();
            if param_idx + expected_params <= parameters.len() {
                operation.parameters = parameters[param_idx..param_idx + expected_params].to_vec();
                param_idx += expected_params;
            }
        }
        
        Ok(())
    }
}

/// PennyLane-style decorator for quantum functions
pub fn qnode<F>(func: F, device: Arc<Mutex<dyn Device>>) -> QNodeDecorator<F>
where
    F: Fn(&mut QNodeBuilder) -> Result<()>,
{
    QNodeDecorator {
        func,
        device,
    }
}

/// QNode decorator structure
pub struct QNodeDecorator<F>
where
    F: Fn(&mut QNodeBuilder) -> Result<()>,
{
    func: F,
    device: Arc<Mutex<dyn Device>>,
}

impl<F> QNodeDecorator<F>
where
    F: Fn(&mut QNodeBuilder) -> Result<()>,
{
    /// Execute the decorated quantum function
    pub fn execute(&self, _parameters: &[f64]) -> Result<DeviceResult> {
        let mut device = self.device.lock().unwrap();
        let num_qubits = device.num_qubits();
        
        let mut builder = QNodeBuilder::new(num_qubits);
        (self.func)(&mut builder)?;
        
        let qnode = builder.build();
        device.execute(&qnode, None)
    }
}

/// Builder for constructing QNodes
pub struct QNodeBuilder {
    qnode: QNode,
}

impl QNodeBuilder {
    /// Create a new QNode builder
    pub fn new(num_qubits: usize) -> Self {
        Self {
            qnode: QNode::new(num_qubits, "default.qubit".to_string()),
        }
    }
    
    /// Add Hadamard gate
    pub fn hadamard(&mut self, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "Hadamard".to_string(),
            wires: vec![wire],
            parameters: vec![],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add Pauli-X gate
    pub fn pauli_x(&mut self, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "PauliX".to_string(),
            wires: vec![wire],
            parameters: vec![],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add Pauli-Y gate
    pub fn pauli_y(&mut self, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "PauliY".to_string(),
            wires: vec![wire],
            parameters: vec![],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add Pauli-Z gate
    pub fn pauli_z(&mut self, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "PauliZ".to_string(),
            wires: vec![wire],
            parameters: vec![],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add RX rotation gate
    pub fn rx(&mut self, angle: f64, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "RX".to_string(),
            wires: vec![wire],
            parameters: vec![angle],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add RY rotation gate
    pub fn ry(&mut self, angle: f64, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "RY".to_string(),
            wires: vec![wire],
            parameters: vec![angle],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add RZ rotation gate
    pub fn rz(&mut self, angle: f64, wire: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "RZ".to_string(),
            wires: vec![wire],
            parameters: vec![angle],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "CNOT".to_string(),
            wires: vec![control, target],
            parameters: vec![],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add CZ gate
    pub fn cz(&mut self, control: usize, target: usize) -> Result<&mut Self> {
        self.qnode.add_operation(Operation {
            name: "CZ".to_string(),
            wires: vec![control, target],
            parameters: vec![],
            hyperparameters: HashMap::new(),
        })?;
        Ok(self)
    }
    
    /// Add expectation value measurement
    pub fn expectation(&mut self, observable: Operator, name: String) -> Result<&mut Self> {
        self.qnode.add_observable(name, observable)?;
        Ok(self)
    }
    
    /// Build the final QNode
    pub fn build(self) -> QNode {
        self.qnode
    }
}

/// Create a device (PennyLane-style)
pub fn device(name: &str, num_qubits: usize) -> Result<Box<dyn Device>> {
    match name {
        "default.qubit" => Ok(Box::new(DefaultQubitDevice::new(num_qubits))),
        _ => Err(QuantumError::InvalidParameter(
            format!("Unknown device: {}", name)
        )),
    }
}

// Removed unused macro definition

/// Gradient computation using parameter-shift rule
pub struct ParameterShiftGradient {
    shift_value: f64,
}

impl ParameterShiftGradient {
    /// Create a new parameter-shift gradient computer
    pub fn new() -> Self {
        Self {
            shift_value: std::f64::consts::PI / 2.0,
        }
    }
    
    /// Compute gradients for a QNode
    pub fn compute_gradients(
        &self,
        qnode_fn: fn(&[f64]) -> Result<f64>,
        parameters: &[f64],
        _observable: &Operator,
    ) -> Result<Vec<f64>> {
        let mut gradients = Vec::with_capacity(parameters.len());
        
        for i in 0..parameters.len() {
            let mut params_plus = parameters.to_vec();
            let mut params_minus = parameters.to_vec();
            
            params_plus[i] += self.shift_value;
            params_minus[i] -= self.shift_value;
            
            let f_plus = qnode_fn(&params_plus)?;
            let f_minus = qnode_fn(&params_minus)?;
            
            let gradient = (f_plus - f_minus) / 2.0;
            gradients.push(gradient);
        }
        
        Ok(gradients)
    }
}

/// PennyLane-style optimization integration
pub fn optimize_qnode(
    qnode_fn: fn(&[f64]) -> f64,
    initial_params: &[f64],
    optimizer: &mut dyn VariationalOptimizer,
) -> Result<OptimizationResult> {
    // Use the basic Optimizer trait method instead
    optimizer.optimize(qnode_fn, initial_params)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_default_qubit_device() {
        let device = DefaultQubitDevice::new(2);
        assert_eq!(device.num_qubits(), 2);
        assert_eq!(device.name(), "default.qubit");
        
        let caps = device.capabilities();
        assert!(caps.supports_state);
        assert!(caps.supports_sampling);
        assert!(caps.max_qubits >= 2);
    }
    
    #[test]
    fn test_qnode_builder() {
        let mut builder = QNodeBuilder::new(2);
        
        builder.hadamard(0).unwrap()
               .cnot(0, 1).unwrap()
               .ry(std::f64::consts::PI / 4.0, 1).unwrap();
        
        let qnode = builder.build();
        assert_eq!(qnode.operations.len(), 3);
        assert_eq!(qnode.num_qubits, 2);
    }
    
    #[test]
    fn test_qnode_execution() {
        let mut device = DefaultQubitDevice::new(1);
        let mut builder = QNodeBuilder::new(1);
        
        // Build a simple circuit: H|0⟩
        builder.hadamard(0).unwrap();
        let qnode = builder.build();
        
        let result = qnode.execute(&mut device, None).unwrap();
        
        assert!(result.state.is_some());
        let state = result.state.unwrap();
        
        // Should be in |+⟩ state
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert_abs_diff_eq!(state[0].re, sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, sqrt2_inv, epsilon = 1e-10);
    }
    
    #[test]
    fn test_qnode_with_observable() {
        let mut device = DefaultQubitDevice::new(1);
        let mut builder = QNodeBuilder::new(1);
        
        // Build circuit and add observable
        builder.pauli_x(0).unwrap()
               .expectation(constants::pauli_z(), "Z".to_string()).unwrap();
        
        let qnode = builder.build();
        let result = qnode.execute(&mut device, None).unwrap();
        
        assert!(result.expectations.contains_key("Z"));
        let expectation = result.expectations["Z"];
        
        // X|0⟩ = |1⟩, so ⟨1|Z|1⟩ = -1
        assert_abs_diff_eq!(expectation, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_parameter_shift_gradient() {
        let gradient_computer = ParameterShiftGradient::new();
        
        // Simple quadratic function for testing
        let test_fn = |params: &[f64]| -> Result<f64> {
            Ok(params[0] * params[0])
        };
        
        let params = vec![2.0];
        let observable = constants::pauli_z();
        
        // Create a wrapper function to satisfy the fn pointer requirement
        fn test_wrapper(params: &[f64]) -> Result<f64> {
            Ok(params[0].sin())
        }
        
        let gradients = gradient_computer.compute_gradients(
            test_wrapper,
            &params,
            &observable,
        ).unwrap();
        
        // Gradient of x^2 at x=2 should be approximately 4
        assert_abs_diff_eq!(gradients[0], 4.0, epsilon = 0.1);
    }
    
    #[test]
    fn test_device_creation() {
        let device = device("default.qubit", 3).unwrap();
        assert_eq!(device.num_qubits(), 3);
        assert_eq!(device.name(), "default.qubit");
        
        // Test invalid device
        assert!(crate::pennylane_compat::device("invalid.device", 3).is_err());
    }
}