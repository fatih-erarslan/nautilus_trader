//! Quantum gates implementation with automatic differentiation support
//!
//! This module provides quantum gate implementations as unitary matrices,
//! supporting both fixed and parameterized gates for variational quantum circuits.

use crate::{Complex, Operator, Parameter, Result, QuantumError};
use ndarray::Array2;
// Removed unused import
use serde::{Deserialize, Serialize};

/// Trait for quantum gates
pub trait Gate {
    /// Apply the gate to a quantum state
    fn apply(&self, state: &mut crate::StateVector) -> Result<()>;
    
    /// Get the matrix representation of the gate
    fn matrix(&self) -> Operator;
    
    /// Get the qubits this gate acts on
    fn qubits(&self) -> Vec<usize>;
    
    /// Get the name of the gate
    fn name(&self) -> &str;
    
    /// Clone the gate
    fn clone_gate(&self) -> Box<dyn Gate>;
}

/// Trait for quantum gates (object-safe version)
pub trait QuantumGate: Gate + Send + Sync {
    /// Get gate as Any for downcasting
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Trait for parameterized gates
pub trait ParametricGate: QuantumGate + 'static {
    /// Get parameters
    fn parameters(&self) -> Vec<Parameter>;
    
    /// Set parameters
    fn set_parameters(&mut self, params: Vec<Parameter>) -> Result<()>;
    
    /// Get parameter gradients (for automatic differentiation)
    fn parameter_gradients(&self, state: &crate::StateVector) -> Result<Vec<Complex>>;
}

/// Identity gate (I)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Identity {
    pub qubit: usize,
}

impl Identity {
    pub fn new(qubit: usize) -> Self {
        Self { qubit }
    }
}

impl Gate for Identity {
    fn apply(&self, _state: &mut crate::StateVector) -> Result<()> {
        // Identity does nothing
        Ok(())
    }
    
    fn matrix(&self) -> Operator {
        crate::constants::identity()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "I"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for Identity {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Pauli-X gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliX {
    pub qubit: usize,
}

impl PauliX {
    pub fn new(qubit: usize) -> Self {
        Self { qubit }
    }
}

impl Gate for PauliX {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        crate::constants::pauli_x()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "PauliX"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for PauliX {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Pauli-Y gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliY {
    pub qubit: usize,
}

impl PauliY {
    pub fn new(qubit: usize) -> Self {
        Self { qubit }
    }
}

impl Gate for PauliY {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        crate::constants::pauli_y()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "PauliY"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for PauliY {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Pauli-Z gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliZ {
    pub qubit: usize,
}

impl PauliZ {
    pub fn new(qubit: usize) -> Self {
        Self { qubit }
    }
}

impl Gate for PauliZ {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        crate::constants::pauli_z()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "PauliZ"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for PauliZ {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Hadamard gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hadamard {
    pub qubit: usize,
}

impl Hadamard {
    pub fn new(qubit: usize) -> Self {
        Self { qubit }
    }
}

impl Gate for Hadamard {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        crate::constants::hadamard()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "Hadamard"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for Hadamard {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Parametric rotation around X axis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RX {
    pub qubit: usize,
    pub angle: Parameter,
}

impl RX {
    pub fn new(qubit: usize, angle: Parameter) -> Self {
        Self { qubit, angle }
    }
}

impl Gate for RX {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        let cos_half = (self.angle / 2.0).cos();
        let sin_half = (self.angle / 2.0).sin();
        
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(cos_half, 0.0),
                Complex::new(0.0, -sin_half),
                Complex::new(0.0, -sin_half),
                Complex::new(cos_half, 0.0),
            ]
        ).unwrap()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "RX"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for RX {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ParametricGate for RX {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.angle]
    }
    
    fn set_parameters(&mut self, params: Vec<Parameter>) -> Result<()> {
        if params.len() != 1 {
            return Err(QuantumError::InvalidParameter(
                format!("RX gate requires exactly 1 parameter, got {}", params.len())
            ));
        }
        self.angle = params[0];
        Ok(())
    }
    
    fn parameter_gradients(&self, _state: &crate::StateVector) -> Result<Vec<Complex>> {
        // Gradient of RX gate with respect to angle parameter
        let sin_half = (self.angle / 2.0).sin();
        let cos_half = (self.angle / 2.0).cos();
        
        // d/dθ RX(θ) = -i/2 * X * RX(θ)
        let grad_re = -0.5 * sin_half;
        let grad_im = -0.5 * cos_half;
        
        Ok(vec![Complex::new(grad_re, grad_im)])
    }
}

/// Parametric rotation around Y axis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RY {
    pub qubit: usize,
    pub angle: Parameter,
}

impl RY {
    pub fn new(qubit: usize, angle: Parameter) -> Self {
        Self { qubit, angle }
    }
}

impl Gate for RY {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        let cos_half = (self.angle / 2.0).cos();
        let sin_half = (self.angle / 2.0).sin();
        
        Array2::from_shape_vec(
            (2, 2),
            vec![
                Complex::new(cos_half, 0.0),
                Complex::new(-sin_half, 0.0),
                Complex::new(sin_half, 0.0),
                Complex::new(cos_half, 0.0),
            ]
        ).unwrap()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "RY"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for RY {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ParametricGate for RY {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.angle]
    }
    
    fn set_parameters(&mut self, params: Vec<Parameter>) -> Result<()> {
        if params.len() != 1 {
            return Err(QuantumError::InvalidParameter(
                format!("RY gate requires exactly 1 parameter, got {}", params.len())
            ));
        }
        self.angle = params[0];
        Ok(())
    }
    
    fn parameter_gradients(&self, _state: &crate::StateVector) -> Result<Vec<Complex>> {
        let sin_half = (self.angle / 2.0).sin();
        let cos_half = (self.angle / 2.0).cos();
        
        let grad_re = -0.5 * sin_half;
        let grad_im = 0.5 * cos_half;
        
        Ok(vec![Complex::new(grad_re, grad_im)])
    }
}

/// Parametric rotation around Z axis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RZ {
    pub qubit: usize,
    pub angle: Parameter,
}

impl RZ {
    pub fn new(qubit: usize, angle: Parameter) -> Self {
        Self { qubit, angle }
    }
}

impl Gate for RZ {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_single_qubit_gate(state, self.qubit, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        let exp_neg = Complex::new(0.0, -self.angle / 2.0).exp();
        let exp_pos = Complex::new(0.0, self.angle / 2.0).exp();
        
        Array2::from_shape_vec(
            (2, 2),
            vec![
                exp_neg,
                Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0),
                exp_pos,
            ]
        ).unwrap()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.qubit]
    }
    
    fn name(&self) -> &str {
        "RZ"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for RZ {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ParametricGate for RZ {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.angle]
    }
    
    fn set_parameters(&mut self, params: Vec<Parameter>) -> Result<()> {
        if params.len() != 1 {
            return Err(QuantumError::InvalidParameter(
                format!("RZ gate requires exactly 1 parameter, got {}", params.len())
            ));
        }
        self.angle = params[0];
        Ok(())
    }
    
    fn parameter_gradients(&self, _state: &crate::StateVector) -> Result<Vec<Complex>> {
        let exp_neg = Complex::new(0.0, -self.angle / 2.0).exp();
        let exp_pos = Complex::new(0.0, self.angle / 2.0).exp();
        
        let grad_00 = -0.5 * Complex::i() * exp_neg;
        let grad_11 = 0.5 * Complex::i() * exp_pos;
        
        Ok(vec![grad_00 + grad_11])
    }
}

/// CNOT gate (Controlled-X)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNOT {
    pub control: usize,
    pub target: usize,
}

impl CNOT {
    pub fn new(control: usize, target: usize) -> Self {
        Self { control, target }
    }
}

impl Gate for CNOT {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_two_qubit_gate(state, self.control, self.target, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
            ]
        ).unwrap()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.control, self.target]
    }
    
    fn name(&self) -> &str {
        "CNOT"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for CNOT {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Controlled-Z gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CZ {
    pub control: usize,
    pub target: usize,
}

impl CZ {
    pub fn new(control: usize, target: usize) -> Self {
        Self { control, target }
    }
}

impl Gate for CZ {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_two_qubit_gate(state, self.control, self.target, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(-1.0, 0.0),
            ]
        ).unwrap()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.control, self.target]
    }
    
    fn name(&self) -> &str {
        "CZ"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for CZ {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Parametric controlled rotation gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CRX {
    pub control: usize,
    pub target: usize,
    pub angle: Parameter,
}

impl CRX {
    pub fn new(control: usize, target: usize, angle: Parameter) -> Self {
        Self { control, target, angle }
    }
}

impl Gate for CRX {
    fn apply(&self, state: &mut crate::StateVector) -> Result<()> {
        apply_two_qubit_gate(state, self.control, self.target, &self.matrix())
    }
    
    fn matrix(&self) -> Operator {
        let cos_half = (self.angle / 2.0).cos();
        let sin_half = (self.angle / 2.0).sin();
        
        Array2::from_shape_vec(
            (4, 4),
            vec![
                Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(1.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, 0.0),
                Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(cos_half, 0.0), Complex::new(0.0, -sin_half),
                Complex::new(0.0, 0.0), Complex::new(0.0, 0.0), Complex::new(0.0, -sin_half), Complex::new(cos_half, 0.0),
            ]
        ).unwrap()
    }
    
    fn qubits(&self) -> Vec<usize> {
        vec![self.control, self.target]
    }
    
    fn name(&self) -> &str {
        "CRX"
    }
    
    fn clone_gate(&self) -> Box<dyn Gate> {
        Box::new(self.clone())
    }
}

impl QuantumGate for CRX {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl ParametricGate for CRX {
    fn parameters(&self) -> Vec<Parameter> {
        vec![self.angle]
    }
    
    fn set_parameters(&mut self, params: Vec<Parameter>) -> Result<()> {
        if params.len() != 1 {
            return Err(QuantumError::InvalidParameter(
                format!("CRX gate requires exactly 1 parameter, got {}", params.len())
            ));
        }
        self.angle = params[0];
        Ok(())
    }
    
    fn parameter_gradients(&self, _state: &crate::StateVector) -> Result<Vec<Complex>> {
        let sin_half = (self.angle / 2.0).sin();
        let cos_half = (self.angle / 2.0).cos();
        
        let grad_re = -0.5 * sin_half;
        let grad_im = -0.5 * cos_half;
        
        Ok(vec![Complex::new(grad_re, grad_im)])
    }
}

/// Helper function to apply a single-qubit gate
fn apply_single_qubit_gate(
    state: &mut crate::StateVector,
    qubit: usize,
    gate_matrix: &Operator,
) -> Result<()> {
    let n_qubits = (state.len() as f64).log2() as usize;
    
    if qubit >= n_qubits {
        return Err(QuantumError::InvalidQubit(qubit));
    }
    
    let mut new_state = state.clone();
    let qubit_mask = 1 << qubit;
    
    for i in 0..state.len() {
        let bit = (i & qubit_mask) >> qubit;
        let i_flipped = i ^ qubit_mask;
        
        if bit == 0 {
            // |0⟩ component
            new_state[i] = gate_matrix[[0, 0]] * state[i] + gate_matrix[[0, 1]] * state[i_flipped];
        } else {
            // |1⟩ component
            new_state[i] = gate_matrix[[1, 0]] * state[i_flipped] + gate_matrix[[1, 1]] * state[i];
        }
    }
    
    *state = new_state;
    Ok(())
}

/// Helper function to apply a two-qubit gate
fn apply_two_qubit_gate(
    state: &mut crate::StateVector,
    control: usize,
    target: usize,
    gate_matrix: &Operator,
) -> Result<()> {
    let n_qubits = (state.len() as f64).log2() as usize;
    
    if control >= n_qubits || target >= n_qubits {
        return Err(QuantumError::InvalidQubit(control.max(target)));
    }
    
    if control == target {
        return Err(QuantumError::InvalidParameter(
            "Control and target qubits must be different".to_string()
        ));
    }
    
    let mut new_state = state.clone();
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    
    for i in 0..state.len() {
        let control_bit = (i & control_mask) >> control;
        let target_bit = (i & target_mask) >> target;
        
        let basis_state = (control_bit << 1) | target_bit;
        
        // Find the four computational basis states
        let i00 = i & !(control_mask | target_mask);
        let i01 = i00 | target_mask;
        let i10 = i00 | control_mask;
        let i11 = i00 | control_mask | target_mask;
        
        match basis_state {
            0 => new_state[i] = gate_matrix[[0, 0]] * state[i00] + gate_matrix[[0, 1]] * state[i01] + 
                                gate_matrix[[0, 2]] * state[i10] + gate_matrix[[0, 3]] * state[i11],
            1 => new_state[i] = gate_matrix[[1, 0]] * state[i00] + gate_matrix[[1, 1]] * state[i01] + 
                                gate_matrix[[1, 2]] * state[i10] + gate_matrix[[1, 3]] * state[i11],
            2 => new_state[i] = gate_matrix[[2, 0]] * state[i00] + gate_matrix[[2, 1]] * state[i01] + 
                                gate_matrix[[2, 2]] * state[i10] + gate_matrix[[2, 3]] * state[i11],
            3 => new_state[i] = gate_matrix[[3, 0]] * state[i00] + gate_matrix[[3, 1]] * state[i01] + 
                                gate_matrix[[3, 2]] * state[i10] + gate_matrix[[3, 3]] * state[i11],
            _ => unreachable!(),
        }
    }
    
    *state = new_state;
    Ok(())
}

/// Factory function for creating gates
pub fn create_gate(name: &str, qubits: Vec<usize>, params: Vec<Parameter>) -> Result<Box<dyn QuantumGate>> {
    match name {
        "I" => {
            if qubits.len() != 1 {
                return Err(QuantumError::InvalidParameter("Identity gate requires 1 qubit".to_string()));
            }
            Ok(Box::new(Identity::new(qubits[0])))
        },
        "X" | "PauliX" => {
            if qubits.len() != 1 {
                return Err(QuantumError::InvalidParameter("Pauli-X gate requires 1 qubit".to_string()));
            }
            Ok(Box::new(PauliX::new(qubits[0])))
        },
        "Y" | "PauliY" => {
            if qubits.len() != 1 {
                return Err(QuantumError::InvalidParameter("Pauli-Y gate requires 1 qubit".to_string()));
            }
            Ok(Box::new(PauliY::new(qubits[0])))
        },
        "Z" | "PauliZ" => {
            if qubits.len() != 1 {
                return Err(QuantumError::InvalidParameter("Pauli-Z gate requires 1 qubit".to_string()));
            }
            Ok(Box::new(PauliZ::new(qubits[0])))
        },
        "H" | "Hadamard" => {
            if qubits.len() != 1 {
                return Err(QuantumError::InvalidParameter("Hadamard gate requires 1 qubit".to_string()));
            }
            Ok(Box::new(Hadamard::new(qubits[0])))
        },
        "RX" => {
            if qubits.len() != 1 || params.len() != 1 {
                return Err(QuantumError::InvalidParameter("RX gate requires 1 qubit and 1 parameter".to_string()));
            }
            Ok(Box::new(RX::new(qubits[0], params[0])))
        },
        "RY" => {
            if qubits.len() != 1 || params.len() != 1 {
                return Err(QuantumError::InvalidParameter("RY gate requires 1 qubit and 1 parameter".to_string()));
            }
            Ok(Box::new(RY::new(qubits[0], params[0])))
        },
        "RZ" => {
            if qubits.len() != 1 || params.len() != 1 {
                return Err(QuantumError::InvalidParameter("RZ gate requires 1 qubit and 1 parameter".to_string()));
            }
            Ok(Box::new(RZ::new(qubits[0], params[0])))
        },
        "CNOT" | "CX" => {
            if qubits.len() != 2 {
                return Err(QuantumError::InvalidParameter("CNOT gate requires 2 qubits".to_string()));
            }
            Ok(Box::new(CNOT::new(qubits[0], qubits[1])))
        },
        "CZ" => {
            if qubits.len() != 2 {
                return Err(QuantumError::InvalidParameter("CZ gate requires 2 qubits".to_string()));
            }
            Ok(Box::new(CZ::new(qubits[0], qubits[1])))
        },
        "CRX" => {
            if qubits.len() != 2 || params.len() != 1 {
                return Err(QuantumError::InvalidParameter("CRX gate requires 2 qubits and 1 parameter".to_string()));
            }
            Ok(Box::new(CRX::new(qubits[0], qubits[1], params[0])))
        },
        _ => Err(QuantumError::InvalidParameter(format!("Unknown gate: {}", name))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants;
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;
    
    #[test]
    fn test_pauli_gates() {
        let x = PauliX::new(0);
        let y = PauliY::new(0);
        let z = PauliZ::new(0);
        
        assert_eq!(x.matrix(), constants::pauli_x());
        assert_eq!(y.matrix(), constants::pauli_y());
        assert_eq!(z.matrix(), constants::pauli_z());
    }
    
    #[test]
    fn test_hadamard_gate() {
        let h = Hadamard::new(0);
        let matrix = h.matrix();
        let expected = constants::hadamard();
        
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(matrix[[i, j]].re, expected[[i, j]].re);
                assert_abs_diff_eq!(matrix[[i, j]].im, expected[[i, j]].im);
            }
        }
    }
    
    #[test]
    fn test_rotation_gates() {
        let rx = RX::new(0, PI / 2.0);
        let ry = RY::new(0, PI / 2.0);
        let rz = RZ::new(0, PI / 2.0);
        
        // RX(π/2) should be approximately [[1/√2, -i/√2], [-i/√2, 1/√2]]
        let rx_matrix = rx.matrix();
        assert_abs_diff_eq!(rx_matrix[[0, 0]].re, 1.0 / 2.0_f64.sqrt());
        assert_abs_diff_eq!(rx_matrix[[0, 1]].im, -1.0 / 2.0_f64.sqrt());
        
        // Test parameter interface
        assert_eq!(rx.parameters(), vec![PI / 2.0]);
        
        let mut rx_mut = rx.clone();
        rx_mut.set_parameters(vec![PI]).unwrap();
        assert_eq!(rx_mut.parameters(), vec![PI]);
    }
    
    #[test]
    fn test_cnot_gate() {
        let cnot = CNOT::new(0, 1);
        let matrix = cnot.matrix();
        
        // CNOT should be identity for |00⟩ and |01⟩, swap for |10⟩ and |11⟩
        assert_abs_diff_eq!(matrix[[0, 0]].re, 1.0);
        assert_abs_diff_eq!(matrix[[1, 1]].re, 1.0);
        assert_abs_diff_eq!(matrix[[2, 3]].re, 1.0);
        assert_abs_diff_eq!(matrix[[3, 2]].re, 1.0);
    }
    
    #[test]
    fn test_gate_factory() {
        let x_gate = create_gate("X", vec![0], vec![]).unwrap();
        assert_eq!(x_gate.name(), "PauliX");
        
        let rx_gate = create_gate("RX", vec![0], vec![PI / 4.0]).unwrap();
        assert_eq!(rx_gate.name(), "RX");
        
        let cnot_gate = create_gate("CNOT", vec![0, 1], vec![]).unwrap();
        assert_eq!(cnot_gate.name(), "CNOT");
    }
    
    #[test]
    fn test_gate_application() {
        let mut state = constants::zero_state();
        let h = Hadamard::new(0);
        
        h.apply(&mut state).unwrap();
        
        // Should be in |+⟩ state
        let expected = constants::plus_state();
        assert_abs_diff_eq!(state[0].re, expected[0].re);
        assert_abs_diff_eq!(state[1].re, expected[1].re);
    }
}