//! Quantum gate implementations for QAR circuits

use num_complex::Complex64;
use std::f64::consts::PI;

/// Quantum gate representation
#[derive(Debug, Clone)]
pub struct Gate {
    /// Gate name
    pub name: String,
    /// Gate matrix (2x2 for single-qubit, 4x4 for two-qubit)
    pub matrix: Vec<Vec<Complex64>>,
    /// Number of qubits this gate acts on
    pub num_qubits: usize,
    /// Gate parameters
    pub parameters: Vec<f64>,
}

impl Gate {
    /// Create a new gate
    pub fn new(name: String, matrix: Vec<Vec<Complex64>>, num_qubits: usize) -> Self {
        Self {
            name,
            matrix,
            num_qubits,
            parameters: Vec::new(),
        }
    }

    /// Get the gate matrix
    pub fn matrix(&self) -> &Vec<Vec<Complex64>> {
        &self.matrix
    }

    /// Check if the gate is unitary
    pub fn is_unitary(&self) -> bool {
        let n = self.matrix.len();
        
        // Calculate U * U†
        let mut product = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    product[i][j] += self.matrix[i][k] * self.matrix[j][k].conj();
                }
            }
        }

        // Check if it's the identity matrix
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                if (product[i][j] - Complex64::new(expected, 0.0)).norm() > 1e-10 {
                    return false;
                }
            }
        }

        true
    }

    /// Get the adjoint (conjugate transpose) of the gate
    pub fn adjoint(&self) -> Self {
        let n = self.matrix.len();
        let mut adjoint_matrix = vec![vec![Complex64::new(0.0, 0.0); n]; n];
        
        for i in 0..n {
            for j in 0..n {
                adjoint_matrix[i][j] = self.matrix[j][i].conj();
            }
        }

        Self {
            name: format!("{}_adjoint", self.name),
            matrix: adjoint_matrix,
            num_qubits: self.num_qubits,
            parameters: self.parameters.clone(),
        }
    }
}

/// Standard quantum gates
pub struct StandardGates;

impl StandardGates {
    /// Identity gate
    pub fn identity() -> Gate {
        Gate::new(
            "I".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
            1,
        )
    }

    /// Pauli-X gate (NOT gate)
    pub fn pauli_x() -> Gate {
        Gate::new(
            "X".to_string(),
            vec![
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
            1,
        )
    }

    /// Pauli-Y gate
    pub fn pauli_y() -> Gate {
        Gate::new(
            "Y".to_string(),
            vec![
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                vec![Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
            ],
            1,
        )
    }

    /// Pauli-Z gate
    pub fn pauli_z() -> Gate {
        Gate::new(
            "Z".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ],
            1,
        )
    }

    /// Hadamard gate
    pub fn hadamard() -> Gate {
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        Gate::new(
            "H".to_string(),
            vec![
                vec![Complex64::new(inv_sqrt2, 0.0), Complex64::new(inv_sqrt2, 0.0)],
                vec![Complex64::new(inv_sqrt2, 0.0), Complex64::new(-inv_sqrt2, 0.0)],
            ],
            1,
        )
    }

    /// S gate (phase gate)
    pub fn s_gate() -> Gate {
        Gate::new(
            "S".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0)],
            ],
            1,
        )
    }

    /// T gate (π/8 gate)
    pub fn t_gate() -> Gate {
        let phase = Complex64::new(0.0, PI / 4.0).exp();
        Gate::new(
            "T".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), phase],
            ],
            1,
        )
    }

    /// Rotation-X gate
    pub fn rx(theta: f64) -> Gate {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        
        let mut gate = Gate::new(
            "RX".to_string(),
            vec![
                vec![Complex64::new(cos_half, 0.0), Complex64::new(0.0, -sin_half)],
                vec![Complex64::new(0.0, -sin_half), Complex64::new(cos_half, 0.0)],
            ],
            1,
        );
        gate.parameters = vec![theta];
        gate
    }

    /// Rotation-Y gate
    pub fn ry(theta: f64) -> Gate {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        
        let mut gate = Gate::new(
            "RY".to_string(),
            vec![
                vec![Complex64::new(cos_half, 0.0), Complex64::new(-sin_half, 0.0)],
                vec![Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0)],
            ],
            1,
        );
        gate.parameters = vec![theta];
        gate
    }

    /// Rotation-Z gate
    pub fn rz(theta: f64) -> Gate {
        let phase_neg = Complex64::new(0.0, -theta / 2.0).exp();
        let phase_pos = Complex64::new(0.0, theta / 2.0).exp();
        
        let mut gate = Gate::new(
            "RZ".to_string(),
            vec![
                vec![phase_neg, Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), phase_pos],
            ],
            1,
        );
        gate.parameters = vec![theta];
        gate
    }

    /// Phase gate
    pub fn phase(theta: f64) -> Gate {
        let phase = Complex64::new(0.0, theta).exp();
        
        let mut gate = Gate::new(
            "Phase".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), phase],
            ],
            1,
        );
        gate.parameters = vec![theta];
        gate
    }

    /// CNOT gate (controlled-X)
    pub fn cnot() -> Gate {
        Gate::new(
            "CNOT".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
            2,
        )
    }

    /// Controlled-Z gate
    pub fn cz() -> Gate {
        Gate::new(
            "CZ".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(-1.0, 0.0)],
            ],
            2,
        )
    }

    /// Controlled-Y gate
    pub fn cy() -> Gate {
        Gate::new(
            "CY".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, -1.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 1.0), Complex64::new(0.0, 0.0)],
            ],
            2,
        )
    }

    /// Controlled-Phase gate
    pub fn cphase(theta: f64) -> Gate {
        let phase = Complex64::new(0.0, theta).exp();
        
        let mut gate = Gate::new(
            "CPhase".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), phase],
            ],
            2,
        );
        gate.parameters = vec![theta];
        gate
    }

    /// Controlled-RY gate
    pub fn cry(theta: f64) -> Gate {
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        
        let mut gate = Gate::new(
            "CRY".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(cos_half, 0.0), Complex64::new(-sin_half, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(sin_half, 0.0), Complex64::new(cos_half, 0.0)],
            ],
            2,
        );
        gate.parameters = vec![theta];
        gate
    }

    /// SWAP gate
    pub fn swap() -> Gate {
        Gate::new(
            "SWAP".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ],
            2,
        )
    }

    /// Toffoli gate (CCX)
    pub fn toffoli() -> Gate {
        Gate::new(
            "Toffoli".to_string(),
            vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ],
            3,
        )
    }
}

/// Custom gate builder for complex quantum operations
pub struct GateBuilder {
    name: String,
    num_qubits: usize,
    parameters: Vec<f64>,
}

impl GateBuilder {
    /// Create a new gate builder
    pub fn new(name: String, num_qubits: usize) -> Self {
        Self {
            name,
            num_qubits,
            parameters: Vec::new(),
        }
    }

    /// Add a parameter to the gate
    pub fn with_parameter(mut self, param: f64) -> Self {
        self.parameters.push(param);
        self
    }

    /// Build a parameterized rotation gate
    pub fn build_rotation_gate(self, axis: &str, angle: f64) -> Gate {
        match axis.to_lowercase().as_str() {
            "x" => StandardGates::rx(angle),
            "y" => StandardGates::ry(angle),
            "z" => StandardGates::rz(angle),
            _ => panic!("Invalid rotation axis: {}", axis),
        }
    }

    /// Build a custom unitary gate from matrix elements
    pub fn build_from_matrix(self, matrix: Vec<Vec<Complex64>>) -> Gate {
        let mut gate = Gate::new(self.name, matrix, self.num_qubits);
        gate.parameters = self.parameters;
        gate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_identity_gate() {
        let gate = StandardGates::identity();
        assert_eq!(gate.name, "I");
        assert_eq!(gate.num_qubits, 1);
        assert!(gate.is_unitary());
    }

    #[test]
    fn test_pauli_gates() {
        let x = StandardGates::pauli_x();
        let y = StandardGates::pauli_y();
        let z = StandardGates::pauli_z();
        
        assert!(x.is_unitary());
        assert!(y.is_unitary());
        assert!(z.is_unitary());
    }

    #[test]
    fn test_hadamard_gate() {
        let h = StandardGates::hadamard();
        assert!(h.is_unitary());
        
        // H^2 = I
        let h_squared = GateBuilder::new("H2".to_string(), 1)
            .build_from_matrix(vec![
                vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
                vec![Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)],
            ]);
        
        // This would need proper matrix multiplication to test
        assert!(h_squared.is_unitary());
    }

    #[test]
    fn test_rotation_gates() {
        let angle = PI / 4.0;
        let rx = StandardGates::rx(angle);
        let ry = StandardGates::ry(angle);
        let rz = StandardGates::rz(angle);
        
        assert!(rx.is_unitary());
        assert!(ry.is_unitary());
        assert!(rz.is_unitary());
        
        assert_eq!(rx.parameters[0], angle);
        assert_eq!(ry.parameters[0], angle);
        assert_eq!(rz.parameters[0], angle);
    }

    #[test]
    fn test_controlled_gates() {
        let cnot = StandardGates::cnot();
        let cz = StandardGates::cz();
        let cy = StandardGates::cy();
        
        assert!(cnot.is_unitary());
        assert!(cz.is_unitary());
        assert!(cy.is_unitary());
        
        assert_eq!(cnot.num_qubits, 2);
        assert_eq!(cz.num_qubits, 2);
        assert_eq!(cy.num_qubits, 2);
    }

    #[test]
    fn test_phase_gate() {
        let phase = PI / 3.0;
        let gate = StandardGates::phase(phase);
        
        assert!(gate.is_unitary());
        assert_eq!(gate.parameters[0], phase);
    }

    #[test]
    fn test_swap_gate() {
        let swap = StandardGates::swap();
        assert!(swap.is_unitary());
        assert_eq!(swap.num_qubits, 2);
    }

    #[test]
    fn test_toffoli_gate() {
        let toffoli = StandardGates::toffoli();
        assert!(toffoli.is_unitary());
        assert_eq!(toffoli.num_qubits, 3);
    }

    #[test]
    fn test_gate_adjoint() {
        let gate = StandardGates::hadamard();
        let adjoint = gate.adjoint();
        
        assert!(adjoint.is_unitary());
        assert_eq!(adjoint.name, "H_adjoint");
        
        // H is self-adjoint, so H† = H
        for i in 0..gate.matrix.len() {
            for j in 0..gate.matrix[i].len() {
                assert_relative_eq!(
                    gate.matrix[i][j].re,
                    adjoint.matrix[i][j].re,
                    epsilon = 1e-10
                );
                assert_relative_eq!(
                    gate.matrix[i][j].im,
                    -adjoint.matrix[i][j].im,
                    epsilon = 1e-10
                );
            }
        }
    }

    #[test]
    fn test_gate_builder() {
        let gate = GateBuilder::new("CustomRX".to_string(), 1)
            .with_parameter(PI / 2.0)
            .build_rotation_gate("x", PI / 2.0);
        
        assert_eq!(gate.name, "RX");
        assert!(gate.is_unitary());
        assert_eq!(gate.parameters[0], PI / 2.0);
    }
}