//! Quantum gate operations and implementations
//!
//! This module provides comprehensive quantum gate operations for building
//! quantum circuits in trading algorithms.

use crate::error::{QuantumError, QuantumResult};
use crate::quantum_state::{QuantumState, ComplexAmplitude};
// Complex64 imported via num_complex
use num_traits::{Zero, One};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Quantum gate types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum QuantumGate {
    /// Identity gate
    Identity { 
        /// Target qubit index
        qubit: usize 
    },
    /// Pauli-X gate (NOT gate)
    PauliX { 
        /// Target qubit index
        qubit: usize 
    },
    /// Pauli-Y gate
    PauliY { 
        /// Target qubit index
        qubit: usize 
    },
    /// Pauli-Z gate
    PauliZ { 
        /// Target qubit index
        qubit: usize 
    },
    /// Hadamard gate
    Hadamard { 
        /// Target qubit index
        qubit: usize 
    },
    /// S gate (phase gate)
    S { 
        /// Target qubit index
        qubit: usize 
    },
    /// T gate (π/8 gate)
    T { 
        /// Target qubit index
        qubit: usize 
    },
    /// Rotation around X-axis
    RX { 
        /// Target qubit index
        qubit: usize, 
        /// Rotation angle in radians
        angle: f64 
    },
    /// Rotation around Y-axis
    RY { 
        /// Target qubit index
        qubit: usize, 
        /// Rotation angle in radians
        angle: f64 
    },
    /// Rotation around Z-axis
    RZ { 
        /// Target qubit index
        qubit: usize, 
        /// Rotation angle in radians
        angle: f64 
    },
    /// Phase gate
    Phase { 
        /// Target qubit index
        qubit: usize, 
        /// Phase angle in radians
        phase: f64 
    },
    /// Controlled-NOT gate
    CNOT { 
        /// Control qubit index
        control: usize, 
        /// Target qubit index
        target: usize 
    },
    /// Controlled-Z gate
    CZ { 
        /// Control qubit index
        control: usize, 
        /// Target qubit index
        target: usize 
    },
    /// Controlled-Phase gate
    CPhase { 
        /// Control qubit index
        control: usize, 
        /// Target qubit index
        target: usize, 
        /// Phase angle in radians
        phase: f64 
    },
    /// Toffoli gate (CCX)
    Toffoli { 
        /// First control qubit index
        control1: usize, 
        /// Second control qubit index
        control2: usize, 
        /// Target qubit index
        target: usize 
    },
    /// Fredkin gate (CSWAP)
    Fredkin { 
        /// Control qubit index
        control: usize, 
        /// First target qubit index
        target1: usize, 
        /// Second target qubit index
        target2: usize 
    },
    /// SWAP gate
    SWAP { 
        /// First qubit index
        qubit1: usize, 
        /// Second qubit index
        qubit2: usize 
    },
    /// Quantum Fourier Transform
    QFT { 
        /// List of qubit indices to transform
        qubits: Vec<usize> 
    },
    /// Inverse QFT
    IQFT { 
        /// List of qubit indices to inverse transform
        qubits: Vec<usize> 
    },
    /// Custom unitary gate
    Custom {
        /// List of qubit indices the gate acts on
        qubits: Vec<usize>,
        /// Unitary matrix representation
        matrix: Vec<Vec<ComplexAmplitude>>,
        /// Human-readable name for the gate
        name: String,
    },
}

impl QuantumGate {
    /// Get the qubits affected by this gate
    pub fn affected_qubits(&self) -> Vec<usize> {
        match self {
            QuantumGate::Identity { qubit } => vec![*qubit],
            QuantumGate::PauliX { qubit } => vec![*qubit],
            QuantumGate::PauliY { qubit } => vec![*qubit],
            QuantumGate::PauliZ { qubit } => vec![*qubit],
            QuantumGate::Hadamard { qubit } => vec![*qubit],
            QuantumGate::S { qubit } => vec![*qubit],
            QuantumGate::T { qubit } => vec![*qubit],
            QuantumGate::RX { qubit, .. } => vec![*qubit],
            QuantumGate::RY { qubit, .. } => vec![*qubit],
            QuantumGate::RZ { qubit, .. } => vec![*qubit],
            QuantumGate::Phase { qubit, .. } => vec![*qubit],
            QuantumGate::CNOT { control, target } => vec![*control, *target],
            QuantumGate::CZ { control, target } => vec![*control, *target],
            QuantumGate::CPhase { control, target, .. } => vec![*control, *target],
            QuantumGate::Toffoli { control1, control2, target } => vec![*control1, *control2, *target],
            QuantumGate::Fredkin { control, target1, target2 } => vec![*control, *target1, *target2],
            QuantumGate::SWAP { qubit1, qubit2 } => vec![*qubit1, *qubit2],
            QuantumGate::QFT { qubits } => qubits.clone(),
            QuantumGate::IQFT { qubits } => qubits.clone(),
            QuantumGate::Custom { qubits, .. } => qubits.clone(),
        }
    }

    /// Get the target qubits for this gate
    pub fn target_qubits(&self) -> Vec<usize> {
        self.affected_qubits()
    }

    /// Get the name of the gate
    pub fn name(&self) -> &str {
        match self {
            QuantumGate::Identity { .. } => "I",
            QuantumGate::PauliX { .. } => "X",
            QuantumGate::PauliY { .. } => "Y",
            QuantumGate::PauliZ { .. } => "Z",
            QuantumGate::Hadamard { .. } => "H",
            QuantumGate::S { .. } => "S",
            QuantumGate::T { .. } => "T",
            QuantumGate::RX { .. } => "RX",
            QuantumGate::RY { .. } => "RY",
            QuantumGate::RZ { .. } => "RZ",
            QuantumGate::Phase { .. } => "P",
            QuantumGate::CNOT { .. } => "CNOT",
            QuantumGate::CZ { .. } => "CZ",
            QuantumGate::CPhase { .. } => "CP",
            QuantumGate::Toffoli { .. } => "CCX",
            QuantumGate::Fredkin { .. } => "CSWAP",
            QuantumGate::SWAP { .. } => "SWAP",
            QuantumGate::QFT { .. } => "QFT",
            QuantumGate::IQFT { .. } => "IQFT",
            QuantumGate::Custom { name, .. } => name,
        }
    }

    /// Check if gate is parameterized
    pub fn is_parameterized(&self) -> bool {
        matches!(self, 
            QuantumGate::RX { .. } | 
            QuantumGate::RY { .. } | 
            QuantumGate::RZ { .. } | 
            QuantumGate::Phase { .. } | 
            QuantumGate::CPhase { .. }
        )
    }

    /// Get gate parameters
    pub fn parameters(&self) -> Vec<f64> {
        match self {
            QuantumGate::RX { angle, .. } => vec![*angle],
            QuantumGate::RY { angle, .. } => vec![*angle],
            QuantumGate::RZ { angle, .. } => vec![*angle],
            QuantumGate::Phase { phase, .. } => vec![*phase],
            QuantumGate::CPhase { phase, .. } => vec![*phase],
            _ => vec![],
        }
    }
    
    /// Create a Hadamard gate
    pub fn hadamard(qubit: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::Hadamard { qubit })
    }
    
    /// Create a rotation Y gate
    pub fn rotation_y(qubit: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::RY { qubit, angle })
    }
    
    /// Create a rotation Z gate
    pub fn rotation_z(qubit: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::RZ { qubit, angle })
    }
    
    /// Create a rotation X gate
    pub fn rotation_x(qubit: usize, angle: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::RX { qubit, angle })
    }
    
    /// Create a CNOT gate
    pub fn controlled_not(control: usize, target: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::CNOT { control, target })
    }
    
    /// Create a controlled phase gate
    pub fn controlled_phase(control: usize, target: usize, phase: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::CPhase { control, target, phase })
    }
    
    /// Create a Pauli-X gate
    pub fn pauli_x(qubit: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::PauliX { qubit })
    }
    
    /// Create a Pauli-Y gate
    pub fn pauli_y(qubit: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::PauliY { qubit })
    }
    
    /// Create a Pauli-Z gate
    pub fn pauli_z(qubit: usize) -> QuantumResult<Self> {
        Ok(QuantumGate::PauliZ { qubit })
    }
    
    /// Create a phase gate
    pub fn phase(qubit: usize, phase: f64) -> QuantumResult<Self> {
        Ok(QuantumGate::Phase { qubit, phase })
    }
}

/// Gate operation interface
pub trait GateOperation {
    /// Apply gate to quantum state
    fn apply(&self, state: &mut QuantumState) -> QuantumResult<()>;
    /// Get gate adjoint (inverse)
    fn adjoint(&self) -> Self;
    /// Check if gate is self-adjoint
    fn is_self_adjoint(&self) -> bool;
    /// Get gate matrix (if applicable)
    fn matrix(&self) -> Option<Vec<Vec<ComplexAmplitude>>>;
}

/// Default gate operation implementation
pub struct DefaultGateOperation;

impl DefaultGateOperation {
    pub fn new() -> Self {
        Self
    }

    pub fn apply_hadamard(&self, state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
        apply_hadamard(state, qubit)
    }

    pub fn apply_pauli_x(&self, state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
        apply_pauli_x(state, qubit)
    }

    pub fn apply_pauli_y(&self, state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
        apply_pauli_y(state, qubit)
    }

    pub fn apply_pauli_z(&self, state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
        apply_pauli_z(state, qubit)
    }

    pub fn apply_cnot(&self, state: &mut QuantumState, control: usize, target: usize) -> QuantumResult<()> {
        apply_cnot(state, control, target)
    }

    pub fn apply_cz(&self, state: &mut QuantumState, control: usize, target: usize) -> QuantumResult<()> {
        apply_cz(state, control, target)
    }

    pub fn apply_phase(&self, state: &mut QuantumState, qubit: usize, phase: f64) -> QuantumResult<()> {
        apply_phase_gate(state, qubit, phase)
    }

    pub fn apply_rotation_x(&self, state: &mut QuantumState, qubit: usize, angle: f64) -> QuantumResult<()> {
        apply_rx_gate(state, qubit, angle)
    }

    pub fn apply_rotation_y(&self, state: &mut QuantumState, qubit: usize, angle: f64) -> QuantumResult<()> {
        apply_ry_gate(state, qubit, angle)
    }

    pub fn apply_rotation_z(&self, state: &mut QuantumState, qubit: usize, angle: f64) -> QuantumResult<()> {
        apply_rz_gate(state, qubit, angle)
    }

    pub fn apply_toffoli(&self, state: &mut QuantumState, control1: usize, control2: usize, target: usize) -> QuantumResult<()> {
        apply_toffoli(state, control1, control2, target)
    }

    pub fn apply_s(&self, state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
        apply_s_gate(state, qubit)
    }

    pub fn apply_t(&self, state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
        apply_t_gate(state, qubit)
    }

    pub fn apply_cphase(&self, state: &mut QuantumState, control: usize, target: usize, phase: f64) -> QuantumResult<()> {
        apply_cphase(state, control, target, phase)
    }

    pub fn apply_swap(&self, state: &mut QuantumState, qubit1: usize, qubit2: usize) -> QuantumResult<()> {
        apply_swap(state, qubit1, qubit2)
    }

    pub fn apply_fredkin(&self, state: &mut QuantumState, control: usize, target1: usize, target2: usize) -> QuantumResult<()> {
        apply_fredkin(state, control, target1, target2)
    }

    pub fn apply_qft(&self, state: &mut QuantumState, qubits: &[usize]) -> QuantumResult<()> {
        apply_qft(state, qubits)
    }

    pub fn apply_iqft(&self, state: &mut QuantumState, qubits: &[usize]) -> QuantumResult<()> {
        apply_iqft(state, qubits)
    }

    pub fn apply_custom(&self, state: &mut QuantumState, qubits: &[usize], matrix: &[Vec<ComplexAmplitude>]) -> QuantumResult<()> {
        apply_custom_gate(state, qubits, matrix)
    }
}

impl GateOperation for QuantumGate {
    fn apply(&self, state: &mut QuantumState) -> QuantumResult<()> {
        match self {
            QuantumGate::Identity { .. } => Ok(()),
            QuantumGate::PauliX { qubit } => apply_pauli_x(state, *qubit),
            QuantumGate::PauliY { qubit } => apply_pauli_y(state, *qubit),
            QuantumGate::PauliZ { qubit } => apply_pauli_z(state, *qubit),
            QuantumGate::Hadamard { qubit } => apply_hadamard(state, *qubit),
            QuantumGate::S { qubit } => apply_s_gate(state, *qubit),
            QuantumGate::T { qubit } => apply_t_gate(state, *qubit),
            QuantumGate::RX { qubit, angle } => apply_rx_gate(state, *qubit, *angle),
            QuantumGate::RY { qubit, angle } => apply_ry_gate(state, *qubit, *angle),
            QuantumGate::RZ { qubit, angle } => apply_rz_gate(state, *qubit, *angle),
            QuantumGate::Phase { qubit, phase } => apply_phase_gate(state, *qubit, *phase),
            QuantumGate::CNOT { control, target } => apply_cnot(state, *control, *target),
            QuantumGate::CZ { control, target } => apply_cz(state, *control, *target),
            QuantumGate::CPhase { control, target, phase } => apply_cphase(state, *control, *target, *phase),
            QuantumGate::Toffoli { control1, control2, target } => apply_toffoli(state, *control1, *control2, *target),
            QuantumGate::Fredkin { control, target1, target2 } => apply_fredkin(state, *control, *target1, *target2),
            QuantumGate::SWAP { qubit1, qubit2 } => apply_swap(state, *qubit1, *qubit2),
            QuantumGate::QFT { qubits } => apply_qft(state, qubits),
            QuantumGate::IQFT { qubits } => apply_iqft(state, qubits),
            QuantumGate::Custom { qubits, matrix, .. } => apply_custom_gate(state, qubits, matrix),
        }
    }

    fn adjoint(&self) -> Self {
        match self {
            QuantumGate::Identity { qubit } => QuantumGate::Identity { qubit: *qubit },
            QuantumGate::PauliX { qubit } => QuantumGate::PauliX { qubit: *qubit },
            QuantumGate::PauliY { qubit } => QuantumGate::PauliY { qubit: *qubit },
            QuantumGate::PauliZ { qubit } => QuantumGate::PauliZ { qubit: *qubit },
            QuantumGate::Hadamard { qubit } => QuantumGate::Hadamard { qubit: *qubit },
            QuantumGate::S { qubit } => QuantumGate::Phase { qubit: *qubit, phase: -PI / 2.0 },
            QuantumGate::T { qubit } => QuantumGate::Phase { qubit: *qubit, phase: -PI / 4.0 },
            QuantumGate::RX { qubit, angle } => QuantumGate::RX { qubit: *qubit, angle: -*angle },
            QuantumGate::RY { qubit, angle } => QuantumGate::RY { qubit: *qubit, angle: -*angle },
            QuantumGate::RZ { qubit, angle } => QuantumGate::RZ { qubit: *qubit, angle: -*angle },
            QuantumGate::Phase { qubit, phase } => QuantumGate::Phase { qubit: *qubit, phase: -*phase },
            QuantumGate::CNOT { control, target } => QuantumGate::CNOT { control: *control, target: *target },
            QuantumGate::CZ { control, target } => QuantumGate::CZ { control: *control, target: *target },
            QuantumGate::CPhase { control, target, phase } => QuantumGate::CPhase { control: *control, target: *target, phase: -*phase },
            QuantumGate::Toffoli { control1, control2, target } => QuantumGate::Toffoli { control1: *control1, control2: *control2, target: *target },
            QuantumGate::Fredkin { control, target1, target2 } => QuantumGate::Fredkin { control: *control, target1: *target1, target2: *target2 },
            QuantumGate::SWAP { qubit1, qubit2 } => QuantumGate::SWAP { qubit1: *qubit1, qubit2: *qubit2 },
            QuantumGate::QFT { qubits } => QuantumGate::IQFT { qubits: qubits.clone() },
            QuantumGate::IQFT { qubits } => QuantumGate::QFT { qubits: qubits.clone() },
            QuantumGate::Custom { qubits, matrix, name } => {
                let adjoint_matrix = matrix.iter().enumerate().map(|(i, row)| {
                    row.iter().enumerate().map(|(j, _)| matrix[j][i].conj()).collect()
                }).collect();
                QuantumGate::Custom {
                    qubits: qubits.clone(),
                    matrix: adjoint_matrix,
                    name: format!("{}_adj", name),
                }
            }
        }
    }

    fn is_self_adjoint(&self) -> bool {
        matches!(self, 
            QuantumGate::Identity { .. } | 
            QuantumGate::PauliX { .. } | 
            QuantumGate::PauliY { .. } | 
            QuantumGate::PauliZ { .. } | 
            QuantumGate::Hadamard { .. } | 
            QuantumGate::CNOT { .. } | 
            QuantumGate::CZ { .. } | 
            QuantumGate::Toffoli { .. } | 
            QuantumGate::Fredkin { .. } | 
            QuantumGate::SWAP { .. }
        )
    }

    fn matrix(&self) -> Option<Vec<Vec<ComplexAmplitude>>> {
        match self {
            QuantumGate::Identity { .. } => Some(vec![
                vec![ComplexAmplitude::one(), ComplexAmplitude::zero()],
                vec![ComplexAmplitude::zero(), ComplexAmplitude::one()],
            ]),
            QuantumGate::PauliX { .. } => Some(vec![
                vec![ComplexAmplitude::zero(), ComplexAmplitude::one()],
                vec![ComplexAmplitude::one(), ComplexAmplitude::zero()],
            ]),
            QuantumGate::PauliY { .. } => Some(vec![
                vec![ComplexAmplitude::zero(), ComplexAmplitude::new(0.0, -1.0)],
                vec![ComplexAmplitude::new(0.0, 1.0), ComplexAmplitude::zero()],
            ]),
            QuantumGate::PauliZ { .. } => Some(vec![
                vec![ComplexAmplitude::one(), ComplexAmplitude::zero()],
                vec![ComplexAmplitude::zero(), ComplexAmplitude::new(-1.0, 0.0)],
            ]),
            QuantumGate::Hadamard { .. } => {
                let sqrt_half = (0.5_f64).sqrt();
                Some(vec![
                    vec![ComplexAmplitude::new(sqrt_half, 0.0), ComplexAmplitude::new(sqrt_half, 0.0)],
                    vec![ComplexAmplitude::new(sqrt_half, 0.0), ComplexAmplitude::new(-sqrt_half, 0.0)],
                ])
            },
            QuantumGate::S { .. } => Some(vec![
                vec![ComplexAmplitude::one(), ComplexAmplitude::zero()],
                vec![ComplexAmplitude::zero(), ComplexAmplitude::new(0.0, 1.0)],
            ]),
            QuantumGate::T { .. } => Some(vec![
                vec![ComplexAmplitude::one(), ComplexAmplitude::zero()],
                vec![ComplexAmplitude::zero(), ComplexAmplitude::from_polar(1.0, PI / 4.0)],
            ]),
            QuantumGate::Phase { phase, .. } => Some(vec![
                vec![ComplexAmplitude::one(), ComplexAmplitude::zero()],
                vec![ComplexAmplitude::zero(), ComplexAmplitude::from_polar(1.0, *phase)],
            ]),
            QuantumGate::Custom { matrix, .. } => Some(matrix.clone()),
            _ => None, // Multi-qubit gates would need larger matrices
        }
    }
}

// Gate implementations

fn apply_pauli_x(state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("PauliX", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    
    for i in 0..amplitudes.len() {
        let j = i ^ qubit_mask;
        if i < j {
            amplitudes.swap(i, j);
        }
    }
    
    Ok(())
}

fn apply_pauli_y(state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("PauliY", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    
    for i in 0..amplitudes.len() {
        let j = i ^ qubit_mask;
        if i < j {
            let temp = amplitudes[i];
            amplitudes[i] = if (i & qubit_mask) == 0 { 
                ComplexAmplitude::new(0.0, -1.0) * amplitudes[j] 
            } else { 
                ComplexAmplitude::new(0.0, 1.0) * amplitudes[j] 
            };
            amplitudes[j] = if (j & qubit_mask) == 0 { 
                ComplexAmplitude::new(0.0, -1.0) * temp 
            } else { 
                ComplexAmplitude::new(0.0, 1.0) * temp 
            };
        }
    }
    
    Ok(())
}

fn apply_pauli_z(state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("PauliZ", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    
    for i in 0..amplitudes.len() {
        if (i & qubit_mask) != 0 {
            amplitudes[i] = -amplitudes[i];
        }
    }
    
    Ok(())
}

fn apply_hadamard(state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("Hadamard", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    let sqrt_half = (0.5_f64).sqrt();
    
    for i in 0..amplitudes.len() {
        let j = i ^ qubit_mask;
        if i < j {
            let temp = amplitudes[i];
            amplitudes[i] = ComplexAmplitude::new(sqrt_half, 0.0) * (temp + amplitudes[j]);
            amplitudes[j] = ComplexAmplitude::new(sqrt_half, 0.0) * (temp - amplitudes[j]);
        }
    }
    
    Ok(())
}

fn apply_s_gate(state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
    apply_phase_gate(state, qubit, PI / 2.0)
}

fn apply_t_gate(state: &mut QuantumState, qubit: usize) -> QuantumResult<()> {
    apply_phase_gate(state, qubit, PI / 4.0)
}

fn apply_rx_gate(state: &mut QuantumState, qubit: usize, angle: f64) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("RX", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    
    for i in 0..amplitudes.len() {
        let j = i ^ qubit_mask;
        if i < j {
            let temp = amplitudes[i];
            amplitudes[i] = ComplexAmplitude::new(cos_half, 0.0) * temp + 
                           ComplexAmplitude::new(0.0, -sin_half) * amplitudes[j];
            amplitudes[j] = ComplexAmplitude::new(0.0, -sin_half) * temp + 
                           ComplexAmplitude::new(cos_half, 0.0) * amplitudes[j];
        }
    }
    
    Ok(())
}

fn apply_ry_gate(state: &mut QuantumState, qubit: usize, angle: f64) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("RY", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    let cos_half = (angle / 2.0).cos();
    let sin_half = (angle / 2.0).sin();
    
    for i in 0..amplitudes.len() {
        let j = i ^ qubit_mask;
        if i < j {
            let temp = amplitudes[i];
            amplitudes[i] = ComplexAmplitude::new(cos_half, 0.0) * temp + 
                           ComplexAmplitude::new(-sin_half, 0.0) * amplitudes[j];
            amplitudes[j] = ComplexAmplitude::new(sin_half, 0.0) * temp + 
                           ComplexAmplitude::new(cos_half, 0.0) * amplitudes[j];
        }
    }
    
    Ok(())
}

fn apply_rz_gate(state: &mut QuantumState, qubit: usize, angle: f64) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("RZ", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    let phase_neg = ComplexAmplitude::from_polar(1.0, -angle / 2.0);
    let phase_pos = ComplexAmplitude::from_polar(1.0, angle / 2.0);
    
    for i in 0..amplitudes.len() {
        if (i & qubit_mask) == 0 {
            amplitudes[i] *= phase_neg;
        } else {
            amplitudes[i] *= phase_pos;
        }
    }
    
    Ok(())
}

fn apply_phase_gate(state: &mut QuantumState, qubit: usize, phase: f64) -> QuantumResult<()> {
    if qubit >= state.num_qubits() {
        return Err(QuantumError::gate_error("Phase", "Qubit index out of bounds"));
    }

    let amplitudes = state.amplitudes_mut();
    let qubit_mask = 1 << qubit;
    let phase_factor = ComplexAmplitude::from_polar(1.0, phase);
    
    for i in 0..amplitudes.len() {
        if (i & qubit_mask) != 0 {
            amplitudes[i] *= phase_factor;
        }
    }
    
    Ok(())
}

fn apply_cnot(state: &mut QuantumState, control: usize, target: usize) -> QuantumResult<()> {
    if control >= state.num_qubits() || target >= state.num_qubits() {
        return Err(QuantumError::gate_error("CNOT", "Qubit index out of bounds"));
    }
    if control == target {
        return Err(QuantumError::gate_error("CNOT", "Control and target must be different"));
    }

    let amplitudes = state.amplitudes_mut();
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    
    for i in 0..amplitudes.len() {
        if (i & control_mask) != 0 {
            let j = i ^ target_mask;
            if i < j {
                amplitudes.swap(i, j);
            }
        }
    }
    
    Ok(())
}

fn apply_cz(state: &mut QuantumState, control: usize, target: usize) -> QuantumResult<()> {
    if control >= state.num_qubits() || target >= state.num_qubits() {
        return Err(QuantumError::gate_error("CZ", "Qubit index out of bounds"));
    }
    if control == target {
        return Err(QuantumError::gate_error("CZ", "Control and target must be different"));
    }

    let amplitudes = state.amplitudes_mut();
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    
    for i in 0..amplitudes.len() {
        if (i & control_mask) != 0 && (i & target_mask) != 0 {
            amplitudes[i] = -amplitudes[i];
        }
    }
    
    Ok(())
}

fn apply_cphase(state: &mut QuantumState, control: usize, target: usize, phase: f64) -> QuantumResult<()> {
    if control >= state.num_qubits() || target >= state.num_qubits() {
        return Err(QuantumError::gate_error("CPhase", "Qubit index out of bounds"));
    }
    if control == target {
        return Err(QuantumError::gate_error("CPhase", "Control and target must be different"));
    }

    let amplitudes = state.amplitudes_mut();
    let control_mask = 1 << control;
    let target_mask = 1 << target;
    let phase_factor = ComplexAmplitude::from_polar(1.0, phase);
    
    for i in 0..amplitudes.len() {
        if (i & control_mask) != 0 && (i & target_mask) != 0 {
            amplitudes[i] *= phase_factor;
        }
    }
    
    Ok(())
}

fn apply_toffoli(state: &mut QuantumState, control1: usize, control2: usize, target: usize) -> QuantumResult<()> {
    if control1 >= state.num_qubits() || control2 >= state.num_qubits() || target >= state.num_qubits() {
        return Err(QuantumError::gate_error("Toffoli", "Qubit index out of bounds"));
    }
    if control1 == control2 || control1 == target || control2 == target {
        return Err(QuantumError::gate_error("Toffoli", "All qubits must be different"));
    }

    let amplitudes = state.amplitudes_mut();
    let control1_mask = 1 << control1;
    let control2_mask = 1 << control2;
    let target_mask = 1 << target;
    
    for i in 0..amplitudes.len() {
        if (i & control1_mask) != 0 && (i & control2_mask) != 0 {
            let j = i ^ target_mask;
            if i < j {
                amplitudes.swap(i, j);
            }
        }
    }
    
    Ok(())
}

fn apply_fredkin(state: &mut QuantumState, control: usize, target1: usize, target2: usize) -> QuantumResult<()> {
    if control >= state.num_qubits() || target1 >= state.num_qubits() || target2 >= state.num_qubits() {
        return Err(QuantumError::gate_error("Fredkin", "Qubit index out of bounds"));
    }
    if control == target1 || control == target2 || target1 == target2 {
        return Err(QuantumError::gate_error("Fredkin", "All qubits must be different"));
    }

    let amplitudes = state.amplitudes_mut();
    let control_mask = 1 << control;
    let target1_mask = 1 << target1;
    let target2_mask = 1 << target2;
    
    for i in 0..amplitudes.len() {
        if (i & control_mask) != 0 {
            let j = i ^ target1_mask ^ target2_mask;
            if i < j {
                amplitudes.swap(i, j);
            }
        }
    }
    
    Ok(())
}

fn apply_swap(state: &mut QuantumState, qubit1: usize, qubit2: usize) -> QuantumResult<()> {
    if qubit1 >= state.num_qubits() || qubit2 >= state.num_qubits() {
        return Err(QuantumError::gate_error("SWAP", "Qubit index out of bounds"));
    }
    if qubit1 == qubit2 {
        return Ok(()); // No-op for same qubit
    }

    let amplitudes = state.amplitudes_mut();
    let qubit1_mask = 1 << qubit1;
    let qubit2_mask = 1 << qubit2;
    
    for i in 0..amplitudes.len() {
        let bit1 = (i & qubit1_mask) != 0;
        let bit2 = (i & qubit2_mask) != 0;
        
        if bit1 != bit2 {
            let j = i ^ qubit1_mask ^ qubit2_mask;
            if i < j {
                amplitudes.swap(i, j);
            }
        }
    }
    
    Ok(())
}

fn apply_qft(state: &mut QuantumState, qubits: &[usize]) -> QuantumResult<()> {
    if qubits.iter().any(|&q| q >= state.num_qubits()) {
        return Err(QuantumError::gate_error("QFT", "Qubit index out of bounds"));
    }

    let n = qubits.len();
    for i in 0..n {
        apply_hadamard(state, qubits[i])?;
        for j in (i + 1)..n {
            let phase = PI / (1 << (j - i)) as f64;
            apply_cphase(state, qubits[j], qubits[i], phase)?;
        }
    }
    
    // Reverse qubit order
    for i in 0..(n / 2) {
        apply_swap(state, qubits[i], qubits[n - 1 - i])?;
    }
    
    Ok(())
}

fn apply_iqft(state: &mut QuantumState, qubits: &[usize]) -> QuantumResult<()> {
    if qubits.iter().any(|&q| q >= state.num_qubits()) {
        return Err(QuantumError::gate_error("IQFT", "Qubit index out of bounds"));
    }

    let n = qubits.len();
    
    // Reverse qubit order
    for i in 0..(n / 2) {
        apply_swap(state, qubits[i], qubits[n - 1 - i])?;
    }
    
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            let phase = -PI / (1 << (j - i)) as f64;
            apply_cphase(state, qubits[j], qubits[i], phase)?;
        }
        apply_hadamard(state, qubits[i])?;
    }
    
    Ok(())
}

fn apply_custom_gate(state: &mut QuantumState, qubits: &[usize], matrix: &[Vec<ComplexAmplitude>]) -> QuantumResult<()> {
    if qubits.iter().any(|&q| q >= state.num_qubits()) {
        return Err(QuantumError::gate_error("Custom", "Qubit index out of bounds"));
    }

    let n = qubits.len();
    let matrix_size = 1 << n;
    
    if matrix.len() != matrix_size || matrix.iter().any(|row| row.len() != matrix_size) {
        return Err(QuantumError::gate_error("Custom", "Matrix size mismatch"));
    }

    // Apply custom unitary matrix (simplified implementation)
    let amplitudes = state.amplitudes_mut();
    let mut new_amplitudes = amplitudes.to_vec();
    
    for i in 0..amplitudes.len() {
        let mut new_amp = ComplexAmplitude::zero();
        for j in 0..matrix_size {
            // Calculate basis state mapping (simplified)
            let basis_state = extract_qubit_state(i, qubits);
            if basis_state == j {
                for k in 0..matrix_size {
                    let mapped_i = map_qubit_state(i, qubits, k);
                    new_amp += matrix[j][k] * amplitudes[mapped_i];
                }
            }
        }
        new_amplitudes[i] = new_amp;
    }
    
    amplitudes.copy_from_slice(&new_amplitudes);
    Ok(())
}

// Helper functions for custom gate implementation
fn extract_qubit_state(global_state: usize, qubits: &[usize]) -> usize {
    let mut state = 0;
    for (i, &qubit) in qubits.iter().enumerate() {
        if (global_state & (1 << qubit)) != 0 {
            state |= 1 << i;
        }
    }
    state
}

fn map_qubit_state(global_state: usize, qubits: &[usize], local_state: usize) -> usize {
    let mut result = global_state;
    for (i, &qubit) in qubits.iter().enumerate() {
        if (local_state & (1 << i)) != 0 {
            result |= 1 << qubit;
        } else {
            result &= !(1 << qubit);
        }
    }
    result
}

/// Gate optimization utilities
pub struct GateOptimizer {
    optimizations: HashMap<String, Box<dyn Fn(&[QuantumGate]) -> Vec<QuantumGate>>>,
}

impl GateOptimizer {
    /// Create a new gate optimizer
    pub fn new() -> Self {
        let mut optimizer = Self {
            optimizations: HashMap::new(),
        };
        optimizer.add_default_optimizations();
        optimizer
    }

    /// Add default optimization rules
    fn add_default_optimizations(&mut self) {
        // Cancel adjacent inverse gates
        self.optimizations.insert("cancel_inverses".to_string(), Box::new(|gates| {
            let mut result = Vec::new();
            let mut i = 0;
            while i < gates.len() {
                if i + 1 < gates.len() && gates[i].adjoint() == gates[i + 1] {
                    i += 2; // Skip both gates
                } else {
                    result.push(gates[i].clone());
                    i += 1;
                }
            }
            result
        }));

        // Merge adjacent phase gates
        self.optimizations.insert("merge_phases".to_string(), Box::new(|gates| {
            let mut result = Vec::new();
            let mut i = 0;
            while i < gates.len() {
                if let QuantumGate::Phase { qubit, phase } = &gates[i] {
                    let mut total_phase = *phase;
                    let mut j = i + 1;
                    while j < gates.len() {
                        if let QuantumGate::Phase { qubit: q2, phase: p2 } = &gates[j] {
                            if qubit == q2 {
                                total_phase += p2;
                                j += 1;
                            } else {
                                break;
                            }
                        } else {
                            break;
                        }
                    }
                    if (total_phase % (2.0 * PI)).abs() > crate::DEFAULT_PRECISION {
                        result.push(QuantumGate::Phase { qubit: *qubit, phase: total_phase % (2.0 * PI) });
                    }
                    i = j;
                } else {
                    result.push(gates[i].clone());
                    i += 1;
                }
            }
            result
        }));
    }

    /// Optimize a sequence of gates
    pub fn optimize(&self, gates: &[QuantumGate]) -> Vec<QuantumGate> {
        let mut optimized = gates.to_vec();
        
        for optimization in self.optimizations.values() {
            optimized = optimization(&optimized);
        }
        
        optimized
    }
}

impl Default for GateOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantum_state::QuantumState;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_pauli_x_gate() {
        let mut state = QuantumState::new(1).unwrap();
        let gate = QuantumGate::PauliX { qubit: 0 };
        gate.apply(&mut state).unwrap();
        
        assert_abs_diff_eq!(state.get_amplitude(0).unwrap().norm_sqr(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_amplitude(1).unwrap().norm_sqr(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard_gate() {
        let mut state = QuantumState::new(1).unwrap();
        let gate = QuantumGate::Hadamard { qubit: 0 };
        gate.apply(&mut state).unwrap();
        
        assert_abs_diff_eq!(state.get_amplitude(0).unwrap().norm_sqr(), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_amplitude(1).unwrap().norm_sqr(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut state = QuantumState::new(2).unwrap();
        state.set_amplitude(0, ComplexAmplitude::new(0.0, 0.0)).unwrap();
        state.set_amplitude(1, ComplexAmplitude::new(1.0, 0.0)).unwrap();
        state.normalize().unwrap();
        
        let gate = QuantumGate::CNOT { control: 0, target: 1 };
        gate.apply(&mut state).unwrap();
        
        assert_abs_diff_eq!(state.get_amplitude(1).unwrap().norm_sqr(), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(state.get_amplitude(3).unwrap().norm_sqr(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_gate_matrix() {
        let gate = QuantumGate::PauliX { qubit: 0 };
        let matrix = gate.matrix().unwrap();
        
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 2);
        assert_eq!(matrix[0][0], ComplexAmplitude::zero());
        assert_eq!(matrix[0][1], ComplexAmplitude::one());
        assert_eq!(matrix[1][0], ComplexAmplitude::one());
        assert_eq!(matrix[1][1], ComplexAmplitude::zero());
    }

    #[test]
    fn test_gate_adjoint() {
        let gate = QuantumGate::RX { qubit: 0, angle: PI / 4.0 };
        let adjoint = gate.adjoint();
        
        if let QuantumGate::RX { angle, .. } = adjoint {
            assert_abs_diff_eq!(angle, -PI / 4.0, epsilon = 1e-10);
        } else {
            panic!("Adjoint should be RX gate");
        }
    }

    #[test]
    fn test_gate_optimizer() {
        let optimizer = GateOptimizer::new();
        let gates = vec![
            QuantumGate::PauliX { qubit: 0 },
            QuantumGate::PauliX { qubit: 0 },
            QuantumGate::Phase { qubit: 1, phase: PI / 4.0 },
            QuantumGate::Phase { qubit: 1, phase: PI / 8.0 },
        ];
        
        let optimized = optimizer.optimize(&gates);
        
        // Should cancel the two X gates and merge the phase gates
        assert_eq!(optimized.len(), 1);
        if let QuantumGate::Phase { phase, .. } = &optimized[0] {
            assert_abs_diff_eq!(*phase, 3.0 * PI / 8.0, epsilon = 1e-10);
        }
    }
}