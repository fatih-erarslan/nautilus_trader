//! pBit Gate Operations
//!
//! Implements quantum gate equivalents using probabilistic bit dynamics.
//! Gates operate on PBitState by modifying probabilities and couplings.
//!
//! ## Gate Mapping (Quantum → pBit)
//!
//! | Quantum Gate | pBit Operation |
//! |--------------|----------------|
//! | Hadamard (H) | Set P(↑) = 0.5 |
//! | Pauli-X      | Flip spin: s → -s |
//! | Pauli-Z      | Add π phase (in Ising: flip bias sign) |
//! | RX(θ)        | Rotate: P(↑) = sin²(θ/2) |
//! | RY(θ)        | Rotate: P(↑) = sin²(θ/2) |
//! | RZ(θ)        | Phase (handled via bias) |
//! | CNOT         | Conditional flip based on control spin |
//! | CZ           | Add antiferromagnetic coupling |
//! | SWAP         | Exchange pBit states |

use crate::error::{QuantumError, QuantumResult};
use crate::pbit_state::{PBitCoupling, PBitState};
use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

/// pBit gate trait
pub trait PBitGate: Send + Sync {
    /// Apply gate to pBit state
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()>;

    /// Gate name
    fn name(&self) -> &str;

    /// Target qubits
    fn targets(&self) -> Vec<usize>;

    /// Is this a parameterized gate?
    fn is_parameterized(&self) -> bool {
        false
    }
}

// ============================================================================
// Single-qubit gates
// ============================================================================

/// Hadamard gate: Creates equal superposition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HadamardGate {
    pub target: usize,
}

impl HadamardGate {
    pub fn new(target: usize) -> Self {
        Self { target }
    }
}

impl PBitGate for HadamardGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        if let Some(pbit) = state.get_pbit_mut(self.target) {
            pbit.probability_up = 0.5;
            pbit.spin = if rand::random::<f64>() < 0.5 { 1.0 } else { -1.0 };
        }

        // Recalculate basis probabilities
        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "H"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }
}

/// Pauli-X gate: Bit flip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliXGate {
    pub target: usize,
}

impl PauliXGate {
    pub fn new(target: usize) -> Self {
        Self { target }
    }
}

impl PBitGate for PauliXGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        if let Some(pbit) = state.get_pbit_mut(self.target) {
            pbit.spin *= -1.0;
            pbit.probability_up = 1.0 - pbit.probability_up;
        }

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "X"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }
}

/// Pauli-Y gate: Bit and phase flip
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliYGate {
    pub target: usize,
}

impl PauliYGate {
    pub fn new(target: usize) -> Self {
        Self { target }
    }
}

impl PBitGate for PauliYGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        if let Some(pbit) = state.get_pbit_mut(self.target) {
            pbit.spin *= -1.0;
            pbit.probability_up = 1.0 - pbit.probability_up;
            pbit.bias *= -1.0; // Phase component
        }

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "Y"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }
}

/// Pauli-Z gate: Phase flip (no effect in probability space for diagonal states)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PauliZGate {
    pub target: usize,
}

impl PauliZGate {
    pub fn new(target: usize) -> Self {
        Self { target }
    }
}

impl PBitGate for PauliZGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        // Z gate flips the sign of |1⟩ amplitude
        // In pBit representation, this affects the bias/phase
        if let Some(pbit) = state.get_pbit_mut(self.target) {
            pbit.bias *= -1.0;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "Z"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }
}

/// Rotation around X-axis: RX(θ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RXGate {
    pub target: usize,
    pub theta: f64,
}

impl RXGate {
    pub fn new(target: usize, theta: f64) -> Self {
        Self { target, theta }
    }
}

impl PBitGate for RXGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        if let Some(pbit) = state.get_pbit_mut(self.target) {
            // RX(θ)|0⟩ = cos(θ/2)|0⟩ - i·sin(θ/2)|1⟩
            // P(1) = sin²(θ/2) for |0⟩ initial state
            let current_p = pbit.probability_up;
            let cos_half = (self.theta / 2.0).cos();
            let sin_half = (self.theta / 2.0).sin();

            // Blend current probability with rotation
            let new_p = current_p * cos_half * cos_half + (1.0 - current_p) * sin_half * sin_half
                + 2.0 * (current_p * (1.0 - current_p)).sqrt() * cos_half * sin_half;

            pbit.probability_up = new_p.clamp(0.0, 1.0);
            pbit.spin = if pbit.probability_up > 0.5 { 1.0 } else { -1.0 };
        }

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "RX"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }
}

/// Rotation around Y-axis: RY(θ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RYGate {
    pub target: usize,
    pub theta: f64,
}

impl RYGate {
    pub fn new(target: usize, theta: f64) -> Self {
        Self { target, theta }
    }
}

impl PBitGate for RYGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        if let Some(pbit) = state.get_pbit_mut(self.target) {
            // RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            // P(1) = sin²(θ/2) for |0⟩ initial state
            let current_p = pbit.probability_up;
            let cos_half = (self.theta / 2.0).cos();
            let sin_half = (self.theta / 2.0).sin();

            // For RY, the rotation is real-valued
            let new_p = current_p * cos_half * cos_half + (1.0 - current_p) * sin_half * sin_half
                + 2.0 * (current_p * (1.0 - current_p)).sqrt() * cos_half * sin_half;

            pbit.probability_up = new_p.clamp(0.0, 1.0);
            pbit.spin = if pbit.probability_up > 0.5 { 1.0 } else { -1.0 };
        }

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "RY"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }
}

/// Rotation around Z-axis: RZ(θ)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RZGate {
    pub target: usize,
    pub theta: f64,
}

impl RZGate {
    pub fn new(target: usize, theta: f64) -> Self {
        Self { target, theta }
    }
}

impl PBitGate for RZGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Target qubit out of range"));
        }

        // RZ only affects phase, which in pBit is represented via bias
        let coupling = state.config().coupling_strength;
        if let Some(pbit) = state.get_pbit_mut(self.target) {
            // Phase affects energy landscape
            pbit.bias += self.theta * coupling;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "RZ"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }

    fn is_parameterized(&self) -> bool {
        true
    }
}

/// S gate (π/4 phase)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SGate {
    pub target: usize,
}

impl SGate {
    pub fn new(target: usize) -> Self {
        Self { target }
    }
}

impl PBitGate for SGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        RZGate::new(self.target, PI / 2.0).apply(state)
    }

    fn name(&self) -> &str {
        "S"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }
}

/// T gate (π/8 phase)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TGate {
    pub target: usize,
}

impl TGate {
    pub fn new(target: usize) -> Self {
        Self { target }
    }
}

impl PBitGate for TGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        RZGate::new(self.target, PI / 4.0).apply(state)
    }

    fn name(&self) -> &str {
        "T"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.target]
    }
}

// ============================================================================
// Two-qubit gates
// ============================================================================

/// CNOT (Controlled-X) gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CNOTGate {
    pub control: usize,
    pub target: usize,
}

impl CNOTGate {
    pub fn new(control: usize, target: usize) -> Self {
        Self { control, target }
    }
}

impl PBitGate for CNOTGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.control >= state.num_qubits() || self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Qubit index out of range"));
        }
        if self.control == self.target {
            return Err(QuantumError::invalid_state(
                "Control and target must be different",
            ));
        }

        // Get control spin
        let control_spin = state
            .get_pbit(self.control)
            .map(|p| p.spin)
            .unwrap_or(-1.0);

        // If control is |1⟩ (spin up), flip target
        if control_spin > 0.0 {
            if let Some(target_pbit) = state.get_pbit_mut(self.target) {
                target_pbit.spin *= -1.0;
                target_pbit.probability_up = 1.0 - target_pbit.probability_up;
            }
        }

        // Add coupling to maintain entanglement
        state.add_coupling(PBitCoupling::anti_bell_coupling(
            self.control,
            self.target,
            state.config().coupling_strength,
        ));

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "CNOT"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.control, self.target]
    }
}

/// CZ (Controlled-Z) gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CZGate {
    pub control: usize,
    pub target: usize,
}

impl CZGate {
    pub fn new(control: usize, target: usize) -> Self {
        Self { control, target }
    }
}

impl PBitGate for CZGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.control >= state.num_qubits() || self.target >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Qubit index out of range"));
        }

        // CZ adds antiferromagnetic coupling (penalizes |11⟩)
        state.add_coupling(PBitCoupling {
            i: self.control.min(self.target),
            j: self.control.max(self.target),
            weight: -state.config().coupling_strength, // Antiferromagnetic
        });

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "CZ"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.control, self.target]
    }
}

/// SWAP gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SWAPGate {
    pub qubit1: usize,
    pub qubit2: usize,
}

impl SWAPGate {
    pub fn new(qubit1: usize, qubit2: usize) -> Self {
        Self { qubit1, qubit2 }
    }
}

impl PBitGate for SWAPGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.qubit1 >= state.num_qubits() || self.qubit2 >= state.num_qubits() {
            return Err(QuantumError::invalid_state("Qubit index out of range"));
        }

        // Get both pBit states
        let pbit1 = state.get_pbit(self.qubit1).cloned();
        let pbit2 = state.get_pbit(self.qubit2).cloned();

        if let (Some(p1), Some(p2)) = (pbit1, pbit2) {
            if let Some(target1) = state.get_pbit_mut(self.qubit1) {
                target1.spin = p2.spin;
                target1.probability_up = p2.probability_up;
                target1.bias = p2.bias;
            }
            if let Some(target2) = state.get_pbit_mut(self.qubit2) {
                target2.spin = p1.spin;
                target2.probability_up = p1.probability_up;
                target2.bias = p1.bias;
            }
        }

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "SWAP"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.qubit1, self.qubit2]
    }
}

/// iSWAP gate (SWAP with phase)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ISWAPGate {
    pub qubit1: usize,
    pub qubit2: usize,
}

impl ISWAPGate {
    pub fn new(qubit1: usize, qubit2: usize) -> Self {
        Self { qubit1, qubit2 }
    }
}

impl PBitGate for ISWAPGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        // SWAP followed by phase adjustments
        SWAPGate::new(self.qubit1, self.qubit2).apply(state)?;

        // Add coupling for phase entanglement
        state.add_coupling(PBitCoupling::bell_coupling(
            self.qubit1,
            self.qubit2,
            state.config().coupling_strength * 0.5,
        ));

        Ok(())
    }

    fn name(&self) -> &str {
        "iSWAP"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.qubit1, self.qubit2]
    }
}

// ============================================================================
// Three-qubit gates
// ============================================================================

/// Toffoli (CCNOT) gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToffoliGate {
    pub control1: usize,
    pub control2: usize,
    pub target: usize,
}

impl ToffoliGate {
    pub fn new(control1: usize, control2: usize, target: usize) -> Self {
        Self {
            control1,
            control2,
            target,
        }
    }
}

impl PBitGate for ToffoliGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.control1 >= state.num_qubits()
            || self.control2 >= state.num_qubits()
            || self.target >= state.num_qubits()
        {
            return Err(QuantumError::invalid_state("Qubit index out of range"));
        }

        // Get control spins
        let c1_spin = state.get_pbit(self.control1).map(|p| p.spin).unwrap_or(-1.0);
        let c2_spin = state.get_pbit(self.control2).map(|p| p.spin).unwrap_or(-1.0);

        // If both controls are |1⟩, flip target
        if c1_spin > 0.0 && c2_spin > 0.0 {
            if let Some(target_pbit) = state.get_pbit_mut(self.target) {
                target_pbit.spin *= -1.0;
                target_pbit.probability_up = 1.0 - target_pbit.probability_up;
            }
        }

        // Add three-body interaction via pairwise couplings
        let j = state.config().coupling_strength * 0.5;
        state.add_coupling(PBitCoupling::bell_coupling(self.control1, self.control2, j));
        state.add_coupling(PBitCoupling::anti_bell_coupling(self.control1, self.target, j));
        state.add_coupling(PBitCoupling::anti_bell_coupling(self.control2, self.target, j));

        state.sweep();
        Ok(())
    }

    fn name(&self) -> &str {
        "CCX"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.control1, self.control2, self.target]
    }
}

/// Fredkin (CSWAP) gate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FredkinGate {
    pub control: usize,
    pub target1: usize,
    pub target2: usize,
}

impl FredkinGate {
    pub fn new(control: usize, target1: usize, target2: usize) -> Self {
        Self {
            control,
            target1,
            target2,
        }
    }
}

impl PBitGate for FredkinGate {
    fn apply(&self, state: &mut PBitState) -> QuantumResult<()> {
        if self.control >= state.num_qubits()
            || self.target1 >= state.num_qubits()
            || self.target2 >= state.num_qubits()
        {
            return Err(QuantumError::invalid_state("Qubit index out of range"));
        }

        // Get control spin
        let c_spin = state.get_pbit(self.control).map(|p| p.spin).unwrap_or(-1.0);

        // If control is |1⟩, swap targets
        if c_spin > 0.0 {
            SWAPGate::new(self.target1, self.target2).apply(state)?;
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "CSWAP"
    }

    fn targets(&self) -> Vec<usize> {
        vec![self.control, self.target1, self.target2]
    }
}

// ============================================================================
// Gate factory and circuit builder
// ============================================================================

/// pBit Circuit: sequence of gates
pub struct PBitCircuit {
    gates: Vec<Box<dyn PBitGate + Send + Sync>>,
    num_qubits: usize,
}

impl std::fmt::Debug for PBitCircuit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PBitCircuit")
            .field("num_gates", &self.gates.len())
            .field("num_qubits", &self.num_qubits)
            .finish()
    }
}

impl Default for PBitCircuit {
    fn default() -> Self {
        Self {
            gates: Vec::new(),
            num_qubits: 0,
        }
    }
}

impl PBitCircuit {
    /// Create a new empty circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
        }
    }

    /// Add Hadamard gate
    pub fn h(&mut self, target: usize) -> &mut Self {
        self.gates.push(Box::new(HadamardGate::new(target)));
        self
    }

    /// Add Pauli-X gate
    pub fn x(&mut self, target: usize) -> &mut Self {
        self.gates.push(Box::new(PauliXGate::new(target)));
        self
    }

    /// Add Pauli-Y gate
    pub fn y(&mut self, target: usize) -> &mut Self {
        self.gates.push(Box::new(PauliYGate::new(target)));
        self
    }

    /// Add Pauli-Z gate
    pub fn z(&mut self, target: usize) -> &mut Self {
        self.gates.push(Box::new(PauliZGate::new(target)));
        self
    }

    /// Add RX gate
    pub fn rx(&mut self, target: usize, theta: f64) -> &mut Self {
        self.gates.push(Box::new(RXGate::new(target, theta)));
        self
    }

    /// Add RY gate
    pub fn ry(&mut self, target: usize, theta: f64) -> &mut Self {
        self.gates.push(Box::new(RYGate::new(target, theta)));
        self
    }

    /// Add RZ gate
    pub fn rz(&mut self, target: usize, theta: f64) -> &mut Self {
        self.gates.push(Box::new(RZGate::new(target, theta)));
        self
    }

    /// Add S gate
    pub fn s(&mut self, target: usize) -> &mut Self {
        self.gates.push(Box::new(SGate::new(target)));
        self
    }

    /// Add T gate
    pub fn t(&mut self, target: usize) -> &mut Self {
        self.gates.push(Box::new(TGate::new(target)));
        self
    }

    /// Add CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) -> &mut Self {
        self.gates.push(Box::new(CNOTGate::new(control, target)));
        self
    }

    /// Add CZ gate
    pub fn cz(&mut self, control: usize, target: usize) -> &mut Self {
        self.gates.push(Box::new(CZGate::new(control, target)));
        self
    }

    /// Add SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) -> &mut Self {
        self.gates.push(Box::new(SWAPGate::new(qubit1, qubit2)));
        self
    }

    /// Add Toffoli gate
    pub fn ccx(&mut self, control1: usize, control2: usize, target: usize) -> &mut Self {
        self.gates
            .push(Box::new(ToffoliGate::new(control1, control2, target)));
        self
    }

    /// Add Fredkin gate
    pub fn cswap(&mut self, control: usize, target1: usize, target2: usize) -> &mut Self {
        self.gates
            .push(Box::new(FredkinGate::new(control, target1, target2)));
        self
    }

    /// Execute circuit on a state
    pub fn execute(&self, state: &mut PBitState) -> QuantumResult<()> {
        for gate in &self.gates {
            gate.apply(state)?;
        }
        Ok(())
    }

    /// Get number of gates
    pub fn num_gates(&self) -> usize {
        self.gates.len()
    }

    /// Get circuit depth (simplified: sequential gates)
    pub fn depth(&self) -> usize {
        self.gates.len()
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard() {
        let mut state = PBitState::new(1).unwrap();
        HadamardGate::new(0).apply(&mut state).unwrap();
        assert!((state.get_pbit(0).unwrap().probability_up - 0.5).abs() < 0.1);
    }

    #[test]
    fn test_pauli_x() {
        let mut state = PBitState::new(1).unwrap();
        // Start in |0⟩, apply X to get |1⟩
        PauliXGate::new(0).apply(&mut state).unwrap();
        assert!(state.get_pbit(0).unwrap().spin > 0.0);
    }

    #[test]
    fn test_cnot() {
        let mut state = PBitState::new(2).unwrap();
        // Apply H to control, then CNOT to create Bell state
        HadamardGate::new(0).apply(&mut state).unwrap();
        CNOTGate::new(0, 1).apply(&mut state).unwrap();

        // Bell state should have correlations
        let p00 = state.probability(0);
        let p11 = state.probability(3);
        assert!(p00 + p11 > 0.5); // Correlated states
    }

    #[test]
    fn test_circuit_builder() {
        let mut circuit = PBitCircuit::new(2);
        circuit.h(0).cnot(0, 1);

        let mut state = PBitState::new(2).unwrap();
        circuit.execute(&mut state).unwrap();

        assert_eq!(circuit.num_gates(), 2);
    }

    #[test]
    fn test_rotation_gates() {
        let mut state = PBitState::new(1).unwrap();

        // RY(π) should flip |0⟩ to |1⟩
        RYGate::new(0, PI).apply(&mut state).unwrap();
        assert!(state.get_pbit(0).unwrap().probability_up > 0.9);
    }
}
