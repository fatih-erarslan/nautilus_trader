//! Quantum circuit builder and execution engine
//!
//! This module provides the main `Circuit` struct for building and executing
//! quantum circuits with automatic differentiation support.

use crate::{
    Complex, StateVector, Result, QuantumError,
    gates::{QuantumGate, ParametricGate},
};
use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// A quantum circuit composed of quantum gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Circuit {
    /// Number of qubits in the circuit
    pub n_qubits: usize,
    /// Gates in the circuit (stored as serializable gate data)
    gates: Vec<GateData>,
    /// Parameter values for parameterized gates
    parameters: Vec<f64>,
    /// Parameter names for tracking
    parameter_names: Vec<String>,
    /// Initial state preparation
    initial_state: Option<StateVector>,
}

/// Serializable gate data
#[derive(Debug, Clone, Serialize, Deserialize)]
struct GateData {
    name: String,
    qubits: Vec<usize>,
    parameters: Vec<f64>,
    parameter_indices: Vec<usize>,
}

impl Circuit {
    /// Create a new quantum circuit with specified number of qubits
    pub fn new(n_qubits: usize) -> Self {
        if n_qubits == 0 {
            panic!("Circuit must have at least 1 qubit");
        }
        if n_qubits > 20 {
            // Limit to 20 qubits for classical simulation
            panic!("Circuit limited to 20 qubits for classical simulation");
        }
        
        Self {
            n_qubits,
            gates: Vec::new(),
            parameters: Vec::new(),
            parameter_names: Vec::new(),
            initial_state: None,
        }
    }
    
    /// Set the initial state of the circuit
    pub fn set_initial_state(&mut self, state: StateVector) -> Result<()> {
        let expected_dim = 1 << self.n_qubits;
        if state.len() != expected_dim {
            return Err(QuantumError::DimensionMismatch {
                expected: expected_dim,
                actual: state.len(),
            });
        }
        self.initial_state = Some(state);
        Ok(())
    }
    
    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Box<dyn QuantumGate>) -> Result<()> {
        // Validate qubit indices
        for &qubit in &gate.qubits() {
            if qubit >= self.n_qubits {
                return Err(QuantumError::InvalidQubit(qubit));
            }
        }
        
        // Handle parameters if it's a parametric gate
        let mut parameter_indices = Vec::new();
        let mut gate_params = Vec::new();
        
        // Check if this is a parametric gate by attempting to downcast to known types
        if let Some(rx_gate) = gate.as_any().downcast_ref::<crate::gates::RX>() {
            let params = rx_gate.parameters();
            for (i, param) in params.iter().enumerate() {
                parameter_indices.push(self.parameters.len());
                self.parameters.push(*param);
                self.parameter_names.push(format!("{}_{}", gate.name(), i));
                gate_params.push(*param);
            }
        } else if let Some(ry_gate) = gate.as_any().downcast_ref::<crate::gates::RY>() {
            let params = ry_gate.parameters();
            for (i, param) in params.iter().enumerate() {
                parameter_indices.push(self.parameters.len());
                self.parameters.push(*param);
                self.parameter_names.push(format!("{}_{}", gate.name(), i));
                gate_params.push(*param);
            }
        } else if let Some(rz_gate) = gate.as_any().downcast_ref::<crate::gates::RZ>() {
            let params = rz_gate.parameters();
            for (i, param) in params.iter().enumerate() {
                parameter_indices.push(self.parameters.len());
                self.parameters.push(*param);
                self.parameter_names.push(format!("{}_{}", gate.name(), i));
                gate_params.push(*param);
            }
        } else if let Some(crx_gate) = gate.as_any().downcast_ref::<crate::gates::CRX>() {
            let params = crx_gate.parameters();
            for (i, param) in params.iter().enumerate() {
                parameter_indices.push(self.parameters.len());
                self.parameters.push(*param);
                self.parameter_names.push(format!("{}_{}", gate.name(), i));
                gate_params.push(*param);
            }
        }
        
        self.gates.push(GateData {
            name: gate.name().to_string(),
            qubits: gate.qubits(),
            parameters: gate_params,
            parameter_indices,
        });
        
        Ok(())
    }
    
    /// Add a parameterized gate with a custom parameter name
    pub fn add_parameterized_gate(
        &mut self,
        gate: Box<dyn QuantumGate>,
        param_names: Vec<String>,
    ) -> Result<()> {
        // Validate qubit indices
        for &qubit in &gate.qubits() {
            if qubit >= self.n_qubits {
                return Err(QuantumError::InvalidQubit(qubit));
            }
        }
        
        let mut parameter_indices = Vec::new();
        let mut gate_params = Vec::new();
        
        // Check for parametric gates using downcast to specific types
        let mut params = Vec::new();
        if let Some(rx_gate) = gate.as_any().downcast_ref::<crate::gates::RX>() {
            params = rx_gate.parameters();
        } else if let Some(ry_gate) = gate.as_any().downcast_ref::<crate::gates::RY>() {
            params = ry_gate.parameters();
        } else if let Some(rz_gate) = gate.as_any().downcast_ref::<crate::gates::RZ>() {
            params = rz_gate.parameters();
        } else if let Some(crx_gate) = gate.as_any().downcast_ref::<crate::gates::CRX>() {
            params = crx_gate.parameters();
        }
        
        if !params.is_empty() {
            if param_names.len() != params.len() {
                return Err(QuantumError::InvalidParameter(
                    format!("Expected {} parameter names, got {}", params.len(), param_names.len())
                ));
            }
            
            for (i, param) in params.iter().enumerate() {
                parameter_indices.push(self.parameters.len());
                self.parameters.push(*param);
                self.parameter_names.push(param_names[i].clone());
                gate_params.push(*param);
            }
        } else if !param_names.is_empty() {
            return Err(QuantumError::InvalidParameter(
                "Gate is not parameterized but parameter names were provided".to_string()
            ));
        }
        
        self.gates.push(GateData {
            name: gate.name().to_string(),
            qubits: gate.qubits(),
            parameters: gate_params,
            parameter_indices,
        });
        
        Ok(())
    }
    
    /// Execute the circuit and return the final state
    pub fn execute(&self) -> Result<StateVector> {
        let mut state = if let Some(ref initial) = self.initial_state {
            initial.clone()
        } else {
            // Start in |0...0⟩ state
            let mut state = Array1::zeros(1 << self.n_qubits);
            state[0] = Complex::new(1.0, 0.0);
            state
        };
        
        // Apply each gate
        for gate_data in &self.gates {
            let gate = self.create_gate_from_data(gate_data)?;
            gate.apply(&mut state)?;
        }
        
        Ok(state)
    }
    
    /// Execute the circuit with custom parameters
    pub fn execute_with_parameters(&self, params: &[f64]) -> Result<StateVector> {
        if params.len() != self.parameters.len() {
            return Err(QuantumError::InvalidParameter(
                format!("Expected {} parameters, got {}", self.parameters.len(), params.len())
            ));
        }
        
        let mut state = if let Some(ref initial) = self.initial_state {
            initial.clone()
        } else {
            let mut state = Array1::zeros(1 << self.n_qubits);
            state[0] = Complex::new(1.0, 0.0);
            state
        };
        
        for gate_data in &self.gates {
            let gate = self.create_gate_from_data_with_params(gate_data, params)?;
            gate.apply(&mut state)?;
        }
        
        Ok(state)
    }
    
    /// Get the current parameters
    pub fn parameters(&self) -> &[f64] {
        &self.parameters
    }
    
    /// Set circuit parameters
    pub fn set_parameters(&mut self, params: Vec<f64>) -> Result<()> {
        if params.len() != self.parameters.len() {
            return Err(QuantumError::InvalidParameter(
                format!("Expected {} parameters, got {}", self.parameters.len(), params.len())
            ));
        }
        self.parameters = params;
        Ok(())
    }
    
    /// Get parameter names
    pub fn parameter_names(&self) -> &[String] {
        &self.parameter_names
    }
    
    /// Compute gradients using parameter-shift rule
    pub fn parameter_gradients(&self, observable: &crate::Operator) -> Result<Vec<f64>> {
        let mut gradients = Vec::with_capacity(self.parameters.len());
        
        for i in 0..self.parameters.len() {
            // Parameter-shift rule: gradient = [f(θ + π/2) - f(θ - π/2)] / 2
            let shift = std::f64::consts::PI / 2.0;
            
            let mut params_plus = self.parameters.clone();
            params_plus[i] += shift;
            let state_plus = self.execute_with_parameters(&params_plus)?;
            let expectation_plus = crate::utils::expectation_value(&state_plus, observable)?;
            
            let mut params_minus = self.parameters.clone();
            params_minus[i] -= shift;
            let state_minus = self.execute_with_parameters(&params_minus)?;
            let expectation_minus = crate::utils::expectation_value(&state_minus, observable)?;
            
            let gradient = (expectation_plus.re - expectation_minus.re) / 2.0;
            gradients.push(gradient);
        }
        
        Ok(gradients)
    }
    
    /// Compute the expectation value of an observable
    pub fn expectation_value(&self, observable: &crate::Operator) -> Result<f64> {
        let state = self.execute()?;
        let expectation = crate::utils::expectation_value(&state, observable)?;
        Ok(expectation.re)
    }
    
    /// Compute the expectation value with custom parameters
    pub fn expectation_value_with_parameters(
        &self,
        observable: &crate::Operator,
        params: &[f64],
    ) -> Result<f64> {
        let state = self.execute_with_parameters(params)?;
        let expectation = crate::utils::expectation_value(&state, observable)?;
        Ok(expectation.re)
    }
    
    /// Get circuit depth (maximum number of gates applied to any qubit)
    pub fn depth(&self) -> usize {
        let mut qubit_depths = vec![0; self.n_qubits];
        
        for gate_data in &self.gates {
            let max_depth = gate_data.qubits.iter().map(|&q| qubit_depths[q]).max().unwrap_or(0);
            for &qubit in &gate_data.qubits {
                qubit_depths[qubit] = max_depth + 1;
            }
        }
        
        qubit_depths.into_iter().max().unwrap_or(0)
    }
    
    /// Get the number of gates in the circuit
    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }
    
    /// Get the number of parameters in the circuit
    pub fn parameter_count(&self) -> usize {
        self.parameters.len()
    }
    
    /// Clone the circuit with new parameters
    pub fn with_parameters(&self, params: Vec<f64>) -> Result<Self> {
        let mut circuit = self.clone();
        circuit.set_parameters(params)?;
        Ok(circuit)
    }
    
    /// Create a gate instance from gate data
    fn create_gate_from_data(&self, gate_data: &GateData) -> Result<Box<dyn QuantumGate>> {
        let params: Vec<f64> = gate_data.parameter_indices.iter()
            .map(|&i| self.parameters[i])
            .collect();
            
        crate::gates::create_gate(&gate_data.name, gate_data.qubits.clone(), params)
    }
    
    /// Create a gate instance from gate data with custom parameters
    fn create_gate_from_data_with_params(
        &self,
        gate_data: &GateData,
        circuit_params: &[f64],
    ) -> Result<Box<dyn QuantumGate>> {
        let params: Vec<f64> = gate_data.parameter_indices.iter()
            .map(|&i| circuit_params[i])
            .collect();
            
        crate::gates::create_gate(&gate_data.name, gate_data.qubits.clone(), params)
    }
}

/// Circuit builder for convenient circuit construction
pub struct CircuitBuilder {
    circuit: Circuit,
}

impl CircuitBuilder {
    /// Create a new circuit builder
    pub fn new(n_qubits: usize) -> Self {
        Self {
            circuit: Circuit::new(n_qubits),
        }
    }
    
    /// Add a Hadamard gate
    pub fn h(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::Hadamard::new(qubit))).unwrap();
        self
    }
    
    /// Add a Pauli-X gate
    pub fn x(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::PauliX::new(qubit))).unwrap();
        self
    }
    
    /// Add a Pauli-Y gate
    pub fn y(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::PauliY::new(qubit))).unwrap();
        self
    }
    
    /// Add a Pauli-Z gate
    pub fn z(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::PauliZ::new(qubit))).unwrap();
        self
    }
    
    /// Add an RX rotation gate
    pub fn rx(mut self, qubit: usize, angle: f64) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::RX::new(qubit, angle))).unwrap();
        self
    }
    
    /// Add an RY rotation gate
    pub fn ry(mut self, qubit: usize, angle: f64) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::RY::new(qubit, angle))).unwrap();
        self
    }
    
    /// Add an RZ rotation gate
    pub fn rz(mut self, qubit: usize, angle: f64) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::RZ::new(qubit, angle))).unwrap();
        self
    }
    
    /// Add a CNOT gate
    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::CNOT::new(control, target))).unwrap();
        self
    }
    
    /// Add a CZ gate
    pub fn cz(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::CZ::new(control, target))).unwrap();
        self
    }
    
    /// Add a controlled-RX gate
    pub fn crx(mut self, control: usize, target: usize, angle: f64) -> Self {
        self.circuit.add_gate(Box::new(crate::gates::CRX::new(control, target, angle))).unwrap();
        self
    }
    
    /// Set the initial state
    pub fn initial_state(mut self, state: StateVector) -> Result<Self> {
        self.circuit.set_initial_state(state)?;
        Ok(self)
    }
    
    /// Build the final circuit
    pub fn build(self) -> Circuit {
        self.circuit
    }
}

/// Variational Quantum Circuit (VQC) implementation
pub struct VariationalCircuit {
    circuit: Circuit,
    layers: usize,
    entanglement: EntanglementPattern,
}

/// Entanglement patterns for VQC
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntanglementPattern {
    Linear,
    Circular,
    Full,
    Custom(Vec<(usize, usize)>),
}

impl VariationalCircuit {
    /// Create a new variational quantum circuit
    pub fn new(n_qubits: usize, layers: usize, entanglement: EntanglementPattern) -> Self {
        Self {
            circuit: Circuit::new(n_qubits),
            layers,
            entanglement,
        }
    }
    
    /// Build the variational circuit with random initial parameters
    pub fn build_random(&mut self) -> Result<()> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for layer in 0..self.layers {
            // Single-qubit rotations
            for qubit in 0..self.circuit.n_qubits {
                let rx_angle = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
                let ry_angle = rng.gen_range(-std::f64::consts::PI..std::f64::consts::PI);
                
                self.circuit.add_parameterized_gate(
                    Box::new(crate::gates::RX::new(qubit, rx_angle)),
                    vec![format!("layer_{}_qubit_{}_rx", layer, qubit)],
                )?;
                self.circuit.add_parameterized_gate(
                    Box::new(crate::gates::RY::new(qubit, ry_angle)),
                    vec![format!("layer_{}_qubit_{}_ry", layer, qubit)],
                )?;
            }
            
            // Entangling gates
            let entangling_pairs = self.get_entangling_pairs();
            for (control, target) in entangling_pairs {
                self.circuit.add_gate(Box::new(crate::gates::CNOT::new(control, target)))?;
            }
        }
        
        Ok(())
    }
    
    /// Build the variational circuit with specified parameters
    pub fn build_with_parameters(&mut self, params: &[f64]) -> Result<()> {
        let expected_params = self.layers * self.circuit.n_qubits * 2; // 2 rotations per qubit per layer
        if params.len() != expected_params {
            return Err(QuantumError::InvalidParameter(
                format!("Expected {} parameters, got {}", expected_params, params.len())
            ));
        }
        
        let mut param_idx = 0;
        
        for layer in 0..self.layers {
            // Single-qubit rotations
            for qubit in 0..self.circuit.n_qubits {
                let rx_angle = params[param_idx];
                let ry_angle = params[param_idx + 1];
                param_idx += 2;
                
                self.circuit.add_parameterized_gate(
                    Box::new(crate::gates::RX::new(qubit, rx_angle)),
                    vec![format!("layer_{}_qubit_{}_rx", layer, qubit)],
                )?;
                self.circuit.add_parameterized_gate(
                    Box::new(crate::gates::RY::new(qubit, ry_angle)),
                    vec![format!("layer_{}_qubit_{}_ry", layer, qubit)],
                )?;
            }
            
            // Entangling gates
            let entangling_pairs = self.get_entangling_pairs();
            for (control, target) in entangling_pairs {
                self.circuit.add_gate(Box::new(crate::gates::CNOT::new(control, target)))?;
            }
        }
        
        Ok(())
    }
    
    /// Get the underlying circuit
    pub fn circuit(&self) -> &Circuit {
        &self.circuit
    }
    
    /// Get mutable reference to the underlying circuit
    pub fn circuit_mut(&mut self) -> &mut Circuit {
        &mut self.circuit
    }
    
    /// Get entangling pairs based on the pattern
    fn get_entangling_pairs(&self) -> Vec<(usize, usize)> {
        match &self.entanglement {
            EntanglementPattern::Linear => {
                (0..self.circuit.n_qubits - 1).map(|i| (i, i + 1)).collect()
            },
            EntanglementPattern::Circular => {
                let mut pairs: Vec<_> = (0..self.circuit.n_qubits - 1).map(|i| (i, i + 1)).collect();
                if self.circuit.n_qubits > 2 {
                    pairs.push((self.circuit.n_qubits - 1, 0));
                }
                pairs
            },
            EntanglementPattern::Full => {
                let mut pairs = Vec::new();
                for i in 0..self.circuit.n_qubits {
                    for j in (i + 1)..self.circuit.n_qubits {
                        pairs.push((i, j));
                    }
                }
                pairs
            },
            EntanglementPattern::Custom(pairs) => pairs.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_circuit_creation() {
        let circuit = Circuit::new(2);
        assert_eq!(circuit.n_qubits, 2);
        assert_eq!(circuit.gate_count(), 0);
        assert_eq!(circuit.parameter_count(), 0);
    }
    
    #[test]
    fn test_circuit_builder() {
        let circuit = CircuitBuilder::new(2)
            .h(0)
            .cnot(0, 1)
            .rx(1, std::f64::consts::PI / 4.0)
            .build();
            
        assert_eq!(circuit.gate_count(), 3);
        assert_eq!(circuit.parameter_count(), 1);
    }
    
    #[test]
    fn test_circuit_execution() {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(crate::gates::Hadamard::new(0))).unwrap();
        
        let state = circuit.execute().unwrap();
        let expected = constants::plus_state();
        
        assert_abs_diff_eq!(state[0].re, expected[0].re, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, expected[1].re, epsilon = 1e-10);
    }
    
    #[test]
    fn test_parameterized_circuit() {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(crate::gates::RX::new(0, std::f64::consts::PI))).unwrap();
        
        let state = circuit.execute().unwrap();
        
        // RX(π) should flip |0⟩ to |1⟩
        assert_abs_diff_eq!(state[0].norm(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].norm(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_variational_circuit() {
        let mut vqc = VariationalCircuit::new(2, 1, EntanglementPattern::Linear);
        vqc.build_random().unwrap();
        
        let circuit = vqc.circuit();
        assert_eq!(circuit.n_qubits, 2);
        assert!(circuit.parameter_count() > 0);
    }
    
    #[test]
    fn test_expectation_value() {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(crate::gates::PauliX::new(0))).unwrap();
        
        let pauli_z = constants::pauli_z();
        let expectation = circuit.expectation_value(&pauli_z).unwrap();
        
        // X|0⟩ = |1⟩, so ⟨1|Z|1⟩ = -1
        assert_abs_diff_eq!(expectation, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_parameter_gradients() {
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(crate::gates::RY::new(0, 0.0))).unwrap();
        
        let pauli_z = constants::pauli_z();
        let gradients = circuit.parameter_gradients(&pauli_z).unwrap();
        
        assert_eq!(gradients.len(), 1);
        // Gradient of ⟨RY(θ)|Z|RY(θ)⟩ at θ=0 should be 0
        assert_abs_diff_eq!(gradients[0], 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_circuit_depth() {
        let circuit = CircuitBuilder::new(3)
            .h(0)
            .cnot(0, 1)
            .cnot(1, 2)
            .x(0)
            .build();
            
        // H and X on qubit 0: depth 2
        // CNOT on qubits 0,1 and 1,2: sequential, so depth varies
        assert!(circuit.depth() >= 2);
    }
}