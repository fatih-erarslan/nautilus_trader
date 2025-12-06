//! Quantum Circuit Management System
//!
//! This module provides comprehensive quantum circuit management including
//! circuit construction, optimization, execution, and monitoring.

use crate::quantum_state::QuantumState;
use crate::quantum_gates::QuantumGate;
use crate::error::{QuantumError, QuantumResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};
use chrono::{DateTime, Utc};

use uuid::Uuid;

/// Quantum circuit instruction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitInstruction {
    /// The quantum gate to be applied
    pub gate: QuantumGate,
    /// List of qubit indices this instruction acts on
    pub qubits: Vec<usize>,
    /// Optional parameters for parametrized gates
    pub parameters: Option<Vec<f64>>,
    /// Timestamp when this instruction was created
    pub timestamp: DateTime<Utc>,
}

/// Circuit execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitStats {
    /// Total number of gates in the circuit
    pub total_gates: usize,
    /// Count of each gate type
    pub gate_counts: HashMap<String, usize>,
    /// Circuit depth (number of sequential layers)
    pub depth: usize,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Quantum fidelity of the execution
    pub fidelity: f64,
}

/// Circuit optimization level
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Maximum,
}

/// Circuit execution backend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionBackend {
    Simulator,
    ClassicalOptimized,
    QuantumHardware,
    HybridGpu,
}

/// Circuit execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Execution backend to use
    pub backend: ExecutionBackend,
    /// Level of circuit optimization to apply
    pub optimization_level: OptimizationLevel,
    /// Number of measurement shots
    pub shots: usize,
    /// Timeout in milliseconds
    pub timeout_ms: u64,
    /// Maximum memory usage in MB
    pub max_memory_mb: usize,
    /// Whether to enable parallel execution
    pub parallel_execution: bool,
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            backend: ExecutionBackend::Simulator,
            optimization_level: OptimizationLevel::Basic,
            shots: 1024,
            timeout_ms: 30000,
            max_memory_mb: 1024,
            parallel_execution: true,
        }
    }
}

/// Main quantum circuit structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumCircuit {
    /// Unique identifier for the circuit
    pub id: String,
    /// Human-readable name for the circuit
    pub name: String,
    /// Number of qubits in the circuit
    pub num_qubits: usize,
    /// List of circuit instructions
    pub instructions: Vec<CircuitInstruction>,
    /// Timestamp when circuit was created
    pub created_at: DateTime<Utc>,
    /// Timestamp when circuit was last modified
    pub modified_at: DateTime<Utc>,
    /// Optional execution statistics
    pub stats: Option<CircuitStats>,
    /// Additional metadata for the circuit
    pub metadata: HashMap<String, String>,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(name: String, num_qubits: usize) -> Self {
        let id = format!("circuit_{}", Uuid::new_v4());
        let now = Utc::now();
        
        Self {
            id,
            name,
            num_qubits,
            instructions: Vec::new(),
            created_at: now,
            modified_at: now,
            stats: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create a new quantum circuit with just the number of qubits
    pub fn new_simple(num_qubits: usize) -> Self {
        Self::new(format!("circuit_{}", Uuid::new_v4()), num_qubits)
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: QuantumGate, qubits: Vec<usize>) -> QuantumResult<()> {
        if qubits.iter().any(|&q| q >= self.num_qubits) {
            return Err(QuantumError::invalid_qubit_index(
                qubits.iter().max().unwrap_or(&0).clone(),
                self.num_qubits
            ));
        }

        let gate_debug = format!("{:?}", gate);
        let instruction = CircuitInstruction {
            gate,
            qubits,
            parameters: None,
            timestamp: Utc::now(),
        };

        self.instructions.push(instruction);
        self.modified_at = Utc::now();

        debug!("Added gate to circuit {}: {}", self.id, gate_debug);
        Ok(())
    }
    
    /// Add a gate that already contains its qubit information
    pub fn add_gate_auto(&mut self, gate: QuantumGate) -> QuantumResult<()> {
        let qubits = gate.affected_qubits();
        self.add_gate(gate, qubits)
    }
    
    /// Execute the circuit on a quantum state
    pub fn execute(&self, state: &mut QuantumState) -> QuantumResult<()> {
        use crate::quantum_gates::DefaultGateOperation;
        
        let gate_operation = DefaultGateOperation::new();
        
        for instruction in &self.instructions {
            match &instruction.gate {
                QuantumGate::Hadamard { qubit } => {
                    gate_operation.apply_hadamard(state, *qubit)?;
                }
                QuantumGate::PauliX { qubit } => {
                    gate_operation.apply_pauli_x(state, *qubit)?;
                }
                QuantumGate::PauliY { qubit } => {
                    gate_operation.apply_pauli_y(state, *qubit)?;
                }
                QuantumGate::PauliZ { qubit } => {
                    gate_operation.apply_pauli_z(state, *qubit)?;
                }
                QuantumGate::CNOT { control, target } => {
                    gate_operation.apply_cnot(state, *control, *target)?;
                }
                QuantumGate::CZ { control, target } => {
                    gate_operation.apply_cz(state, *control, *target)?;
                }
                QuantumGate::Phase { qubit, phase } => {
                    gate_operation.apply_phase(state, *qubit, *phase)?;
                }
                QuantumGate::RX { qubit, angle } => {
                    gate_operation.apply_rotation_x(state, *qubit, *angle)?;
                }
                QuantumGate::RY { qubit, angle } => {
                    gate_operation.apply_rotation_y(state, *qubit, *angle)?;
                }
                QuantumGate::RZ { qubit, angle } => {
                    gate_operation.apply_rotation_z(state, *qubit, *angle)?;
                }
                QuantumGate::Identity { qubit: _ } => {
                    // No operation needed for identity
                }
                _ => {
                    return Err(QuantumError::circuit_error(
                        &self.id,
                        format!("Unsupported gate type: {:?}", instruction.gate)
                    ));
                }
            }
        }
        
        Ok(())
    }

    /// Add a Hadamard gate
    pub fn add_hadamard(&mut self, qubit: usize) -> QuantumResult<()> {
        let gate = QuantumGate::Hadamard { qubit };
        self.add_gate(gate, vec![qubit])
    }

    /// Add a Pauli-X gate
    pub fn add_pauli_x(&mut self, qubit: usize) -> QuantumResult<()> {
        let gate = QuantumGate::PauliX { qubit };
        self.add_gate(gate, vec![qubit])
    }

    /// Add a Pauli-Y gate
    pub fn add_pauli_y(&mut self, qubit: usize) -> QuantumResult<()> {
        let gate = QuantumGate::PauliY { qubit };
        self.add_gate(gate, vec![qubit])
    }

    /// Add a Pauli-Z gate
    pub fn add_pauli_z(&mut self, qubit: usize) -> QuantumResult<()> {
        let gate = QuantumGate::PauliZ { qubit };
        self.add_gate(gate, vec![qubit])
    }

    /// Add an identity gate
    pub fn add_identity(&mut self, qubit: usize) -> QuantumResult<()> {
        let gate = QuantumGate::Identity { qubit };
        self.add_gate(gate, vec![qubit])
    }

    /// Add a CNOT gate
    pub fn add_cnot(&mut self, control: usize, target: usize) -> QuantumResult<()> {
        let gate = QuantumGate::CNOT { control, target };
        self.add_gate(gate, vec![control, target])
    }

    /// Add a parametrized gate to the circuit
    pub fn add_parametrized_gate(
        &mut self,
        gate: QuantumGate,
        qubits: Vec<usize>,
        parameters: Vec<f64>,
    ) -> QuantumResult<()> {
        if qubits.iter().any(|&q| q >= self.num_qubits) {
            return Err(QuantumError::invalid_qubit_index(
                qubits.iter().max().unwrap_or(&0).clone(),
                self.num_qubits
            ));
        }

        let gate_debug = format!("{:?}", gate);
        let instruction = CircuitInstruction {
            gate,
            qubits,
            parameters: Some(parameters),
            timestamp: Utc::now(),
        };

        self.instructions.push(instruction);
        self.modified_at = Utc::now();

        debug!("Added parametrized gate to circuit {}: {}", self.id, gate_debug);
        Ok(())
    }

    /// Get circuit depth (number of sequential operations)
    pub fn depth(&self) -> usize {
        // Simple depth calculation - can be optimized
        self.instructions.len()
    }

    /// Get gate count statistics
    pub fn gate_counts(&self) -> HashMap<String, usize> {
        let mut counts = HashMap::new();
        for instruction in &self.instructions {
            let gate_name = format!("{:?}", instruction.gate);
            *counts.entry(gate_name).or_insert(0) += 1;
        }
        counts
    }

    /// Optimize the circuit
    pub fn optimize(&mut self, level: OptimizationLevel) -> QuantumResult<()> {
        let original_instruction_count = self.instructions.len();
        
        match level {
            OptimizationLevel::None => {
                debug!("No optimization requested for circuit {}", self.id);
                return Ok(());
            }
            OptimizationLevel::Basic => {
                self.remove_identity_gates()?;
                self.merge_adjacent_rotations()?;
            }
            OptimizationLevel::Aggressive => {
                self.remove_identity_gates()?;
                self.merge_adjacent_rotations()?;
                self.cancel_inverse_gates()?;
                self.reorder_commuting_gates()?;
            }
            OptimizationLevel::Maximum => {
                self.remove_identity_gates()?;
                self.merge_adjacent_rotations()?;
                self.cancel_inverse_gates()?;
                self.reorder_commuting_gates()?;
                self.decompose_multi_qubit_gates()?;
                self.apply_circuit_synthesis()?;
            }
        }

        let optimized_instruction_count = self.instructions.len();
        let reduction = original_instruction_count - optimized_instruction_count;
        
        info!(
            "Circuit {} optimized: {} -> {} instructions ({} reduction)",
            self.id, original_instruction_count, optimized_instruction_count, reduction
        );

        self.modified_at = Utc::now();
        Ok(())
    }

    /// Remove identity gates that don't affect the state
    fn remove_identity_gates(&mut self) -> QuantumResult<()> {
        self.instructions.retain(|instruction| {
            !matches!(instruction.gate, QuantumGate::Identity { .. })
        });
        Ok(())
    }

    /// Merge adjacent rotation gates on the same qubit
    fn merge_adjacent_rotations(&mut self) -> QuantumResult<()> {
        // Implementation for merging adjacent rotations
        // This is a simplified version - real implementation would be more complex
        Ok(())
    }

    /// Cancel inverse gate pairs
    fn cancel_inverse_gates(&mut self) -> QuantumResult<()> {
        // Implementation for canceling inverse gates
        // This is a simplified version - real implementation would be more complex
        Ok(())
    }

    /// Reorder commuting gates for better optimization
    fn reorder_commuting_gates(&mut self) -> QuantumResult<()> {
        // Implementation for reordering commuting gates
        // This is a simplified version - real implementation would be more complex
        Ok(())
    }

    /// Decompose multi-qubit gates into elementary gates
    fn decompose_multi_qubit_gates(&mut self) -> QuantumResult<()> {
        // Implementation for decomposing multi-qubit gates
        // This is a simplified version - real implementation would be more complex
        Ok(())
    }

    /// Apply advanced circuit synthesis
    fn apply_circuit_synthesis(&mut self) -> QuantumResult<()> {
        // Implementation for circuit synthesis
        // This is a simplified version - real implementation would be more complex
        Ok(())
    }

    /// Validate circuit structure
    pub fn validate(&self) -> QuantumResult<()> {
        if self.num_qubits == 0 {
            return Err(QuantumError::invalid_circuit("Circuit must have at least one qubit".to_string()));
        }

        if self.instructions.is_empty() {
            warn!("Circuit {} has no instructions", self.id);
        }

        for (i, instruction) in self.instructions.iter().enumerate() {
            if instruction.qubits.iter().any(|&q| q >= self.num_qubits) {
                return Err(QuantumError::invalid_circuit(
                    format!("Instruction {} references invalid qubit index", i)
                ));
            }
        }

        Ok(())
    }

    /// Calculate circuit statistics
    pub fn calculate_stats(&mut self) -> QuantumResult<CircuitStats> {
        let stats = CircuitStats {
            total_gates: self.instructions.len(),
            gate_counts: self.gate_counts(),
            depth: self.depth(),
            execution_time_ns: 0, // Will be updated during execution
            memory_usage_bytes: self.estimate_memory_usage(),
            fidelity: 1.0, // Will be updated during execution
        };

        self.stats = Some(stats.clone());
        Ok(stats)
    }

    /// Estimate memory usage for the circuit
    fn estimate_memory_usage(&self) -> usize {
        // Basic estimation: 2^n complex numbers * 16 bytes each
        let state_size = (1 << self.num_qubits) * 16;
        let instruction_size = self.instructions.len() * 128; // Rough estimate
        state_size + instruction_size
    }

    /// Public estimate memory usage for the circuit
    pub fn estimate_memory_usage_public(&self) -> usize {
        self.estimate_memory_usage()
    }
}

/// Circuit execution engine
#[derive(Debug)]
pub struct CircuitExecution {
    /// Execution configuration
    config: ExecutionConfig,
    /// Performance metrics
    metrics: Arc<Mutex<HashMap<String, f64>>>,
    /// History of circuit executions
    execution_history: Arc<RwLock<Vec<ExecutionResult>>>,
}

/// Execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// ID of the executed circuit
    pub circuit_id: String,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Quantum fidelity of the execution
    pub fidelity: f64,
    /// Whether execution was successful
    pub success: bool,
    /// Error message if execution failed
    pub error_message: Option<String>,
    /// Measurement results from the execution
    pub measurement_results: Option<Vec<i32>>,
    /// Timestamp of the execution
    pub timestamp: DateTime<Utc>,
}

impl CircuitExecution {
    /// Create a new circuit execution engine
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(Mutex::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Execute a quantum circuit
    pub async fn execute(&self, circuit: &mut QuantumCircuit) -> QuantumResult<ExecutionResult> {
        let start_time = std::time::Instant::now();
        
        // Validate circuit
        circuit.validate()?;

        // Calculate statistics
        circuit.calculate_stats()?;

        // Record execution metrics
        // metrics::counter!("quantum_circuit_executions_total", 1);
        // metrics::gauge!("quantum_circuit_depth", circuit.depth() as f64);
        // metrics::gauge!("quantum_circuit_num_qubits", circuit.num_qubits as f64);

        let execution_result = match self.config.backend {
            ExecutionBackend::Simulator => self.execute_simulator(circuit).await?,
            ExecutionBackend::ClassicalOptimized => self.execute_classical_optimized(circuit).await?,
            ExecutionBackend::QuantumHardware => self.execute_quantum_hardware(circuit).await?,
            ExecutionBackend::HybridGpu => self.execute_hybrid_gpu(circuit).await?,
        };

        let execution_time = start_time.elapsed();
        // metrics::histogram!("quantum_circuit_execution_duration_ms", execution_time.as_millis() as f64);

        // Store execution result
        let mut history = self.execution_history.write().await;
        history.push(execution_result.clone());

        // Update circuit statistics
        if let Some(ref mut stats) = circuit.stats {
            stats.execution_time_ns = execution_time.as_nanos() as u64;
            stats.fidelity = execution_result.fidelity;
        }

        info!(
            "Circuit {} executed successfully in {:?}",
            circuit.id, execution_time
        );

        Ok(execution_result)
    }

    /// Execute circuit on simulator
    async fn execute_simulator(&self, circuit: &QuantumCircuit) -> QuantumResult<ExecutionResult> {
        let mut state = QuantumState::new(circuit.num_qubits)?;
        let gate_operation = crate::quantum_gates::DefaultGateOperation::new();

        for instruction in &circuit.instructions {
            match &instruction.gate {
                QuantumGate::Hadamard { qubit } => {
                    gate_operation.apply_hadamard(&mut state, *qubit)?;
                }
                QuantumGate::PauliX { qubit } => {
                    gate_operation.apply_pauli_x(&mut state, *qubit)?;
                }
                QuantumGate::PauliY { qubit } => {
                    gate_operation.apply_pauli_y(&mut state, *qubit)?;
                }
                QuantumGate::PauliZ { qubit } => {
                    gate_operation.apply_pauli_z(&mut state, *qubit)?;
                }
                QuantumGate::CNOT { control, target } => {
                    gate_operation.apply_cnot(&mut state, *control, *target)?;
                }
                QuantumGate::CZ { control, target } => {
                    gate_operation.apply_cz(&mut state, *control, *target)?;
                }
                QuantumGate::Phase { qubit, phase } => {
                    gate_operation.apply_phase(&mut state, *qubit, *phase)?;
                }
                QuantumGate::RX { qubit, angle } => {
                    gate_operation.apply_rotation_x(&mut state, *qubit, *angle)?;
                }
                QuantumGate::RY { qubit, angle } => {
                    gate_operation.apply_rotation_y(&mut state, *qubit, *angle)?;
                }
                QuantumGate::RZ { qubit, angle } => {
                    gate_operation.apply_rotation_z(&mut state, *qubit, *angle)?;
                }
                QuantumGate::Toffoli { control1, control2, target } => {
                    gate_operation.apply_toffoli(&mut state, *control1, *control2, *target)?;
                }
                QuantumGate::Identity { qubit: _ } => {
                    // No operation needed for identity
                }
                QuantumGate::S { qubit } => {
                    gate_operation.apply_s(&mut state, *qubit)?;
                }
                QuantumGate::T { qubit } => {
                    gate_operation.apply_t(&mut state, *qubit)?;
                }
                QuantumGate::CPhase { control, target, phase } => {
                    gate_operation.apply_cphase(&mut state, *control, *target, *phase)?;
                }
                QuantumGate::SWAP { qubit1, qubit2 } => {
                    gate_operation.apply_swap(&mut state, *qubit1, *qubit2)?;
                }
                QuantumGate::Fredkin { control, target1, target2 } => {
                    gate_operation.apply_fredkin(&mut state, *control, *target1, *target2)?;
                }
                QuantumGate::QFT { qubits } => {
                    gate_operation.apply_qft(&mut state, qubits)?;
                }
                QuantumGate::IQFT { qubits } => {
                    gate_operation.apply_iqft(&mut state, qubits)?;
                }
                QuantumGate::Custom { qubits, matrix, name: _ } => {
                    gate_operation.apply_custom(&mut state, qubits, matrix)?;
                }
            }
        }

        // Perform measurements
        let measurements = self.perform_measurements(&state)?;

        Ok(ExecutionResult {
            circuit_id: circuit.id.clone(),
            execution_time_ns: 0, // Will be set by caller
            memory_usage_bytes: circuit.estimate_memory_usage_public(),
            fidelity: 0.99, // Simulated high fidelity
            success: true,
            error_message: None,
            measurement_results: Some(measurements),
            timestamp: Utc::now(),
        })
    }

    /// Execute circuit with classical optimization
    async fn execute_classical_optimized(&self, circuit: &QuantumCircuit) -> QuantumResult<ExecutionResult> {
        // For now, delegate to simulator but with optimization hints
        self.execute_simulator(circuit).await
    }

    /// Execute circuit on quantum hardware
    async fn execute_quantum_hardware(&self, circuit: &QuantumCircuit) -> QuantumResult<ExecutionResult> {
        // This would interface with real quantum hardware
        // For now, return a simulated result with lower fidelity
        let mut result = self.execute_simulator(circuit).await?;
        result.fidelity = 0.95; // Lower fidelity for hardware
        Ok(result)
    }

    /// Execute circuit on hybrid GPU system
    async fn execute_hybrid_gpu(&self, circuit: &QuantumCircuit) -> QuantumResult<ExecutionResult> {
        // This would use GPU acceleration for circuit simulation
        // For now, return a simulated result with better performance
        let mut result = self.execute_simulator(circuit).await?;
        result.fidelity = 0.98; // High fidelity with GPU optimization
        Ok(result)
    }

    /// Perform measurements on quantum state
    fn perform_measurements(&self, state: &QuantumState) -> QuantumResult<Vec<i32>> {
        let mut measurements = Vec::new();
        
        for _ in 0..self.config.shots {
            let measurement = state.measure_all()?;
            measurements.push(measurement as i32);
        }

        Ok(measurements)
    }

    /// Get execution history
    pub async fn get_execution_history(&self) -> Vec<ExecutionResult> {
        let history = self.execution_history.read().await;
        history.clone()
    }

    /// Get execution metrics
    pub fn get_metrics(&self) -> HashMap<String, f64> {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }

    /// Reset execution history
    pub async fn reset_history(&self) {
        let mut history = self.execution_history.write().await;
        history.clear();
    }
}

/// Circuit builder for easier circuit construction
#[derive(Debug)]
pub struct CircuitBuilder {
    /// The circuit being constructed
    circuit: QuantumCircuit,
}

impl CircuitBuilder {
    /// Create a new circuit builder
    pub fn new(name: String, num_qubits: usize) -> Self {
        Self {
            circuit: QuantumCircuit::new(name, num_qubits),
        }
    }

    /// Add Hadamard gate
    pub fn hadamard(mut self, qubit: usize) -> QuantumResult<Self> {
        self.circuit.add_hadamard(qubit)?;
        Ok(self)
    }

    /// Add Pauli-X gate
    pub fn pauli_x(mut self, qubit: usize) -> QuantumResult<Self> {
        self.circuit.add_pauli_x(qubit)?;
        Ok(self)
    }

    /// Add Pauli-Y gate
    pub fn pauli_y(mut self, qubit: usize) -> QuantumResult<Self> {
        self.circuit.add_pauli_y(qubit)?;
        Ok(self)
    }

    /// Add Pauli-Z gate
    pub fn pauli_z(mut self, qubit: usize) -> QuantumResult<Self> {
        self.circuit.add_pauli_z(qubit)?;
        Ok(self)
    }

    /// Add CNOT gate
    pub fn cnot(mut self, control: usize, target: usize) -> QuantumResult<Self> {
        self.circuit.add_cnot(control, target)?;
        Ok(self)
    }

    /// Add CZ gate
    pub fn cz(mut self, control: usize, target: usize) -> QuantumResult<Self> {
        let gate = QuantumGate::CZ { control, target };
        self.circuit.add_gate(gate, vec![control, target])?;
        Ok(self)
    }

    /// Add rotation-X gate
    pub fn rotation_x(mut self, qubit: usize, angle: f64) -> QuantumResult<Self> {
        let gate = QuantumGate::RX { qubit, angle };
        self.circuit.add_parametrized_gate(gate, vec![qubit], vec![angle])?;
        Ok(self)
    }

    /// Add rotation-Y gate
    pub fn rotation_y(mut self, qubit: usize, angle: f64) -> QuantumResult<Self> {
        let gate = QuantumGate::RY { qubit, angle };
        self.circuit.add_parametrized_gate(gate, vec![qubit], vec![angle])?;
        Ok(self)
    }

    /// Add rotation-Z gate
    pub fn rotation_z(mut self, qubit: usize, angle: f64) -> QuantumResult<Self> {
        let gate = QuantumGate::RZ { qubit, angle };
        self.circuit.add_parametrized_gate(gate, vec![qubit], vec![angle])?;
        Ok(self)
    }

    /// Add phase gate
    pub fn phase(mut self, qubit: usize, phase: f64) -> QuantumResult<Self> {
        let gate = QuantumGate::Phase { qubit, phase };
        self.circuit.add_parametrized_gate(gate, vec![qubit], vec![phase])?;
        Ok(self)
    }

    /// Add Toffoli gate
    pub fn toffoli(mut self, control1: usize, control2: usize, target: usize) -> QuantumResult<Self> {
        let gate = QuantumGate::Toffoli { control1, control2, target };
        self.circuit.add_gate(gate, vec![control1, control2, target])?;
        Ok(self)
    }

    /// Build the circuit
    pub fn build(self) -> QuantumCircuit {
        self.circuit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_circuit_creation() {
        let circuit = QuantumCircuit::new("test_circuit".to_string(), 3);
        assert_eq!(circuit.name, "test_circuit");
        assert_eq!(circuit.num_qubits, 3);
        assert!(circuit.instructions.is_empty());
    }

    #[test]
    fn test_add_gate() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        assert!(circuit.add_hadamard(0).is_ok());
        assert_eq!(circuit.instructions.len(), 1);
        assert!(matches!(circuit.instructions[0].gate, QuantumGate::Hadamard { .. }));
    }

    #[test]
    fn test_invalid_qubit_index() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        assert!(circuit.add_hadamard(3).is_err());
    }

    #[test]
    fn test_circuit_validation() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        assert!(circuit.validate().is_ok());
    }

    #[test]
    fn test_circuit_depth() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        circuit.add_cnot(0, 1).unwrap();
        assert_eq!(circuit.depth(), 2);
    }

    #[test]
    fn test_gate_counts() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        circuit.add_hadamard(1).unwrap();
        circuit.add_gate(QuantumGate::CNOT, vec![0, 1]).unwrap();
        
        let counts = circuit.gate_counts();
        assert_eq!(counts.get("Hadamard"), Some(&2));
        assert_eq!(counts.get("CNOT"), Some(&1));
    }

    #[test]
    fn test_circuit_optimization() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        circuit.add_identity(0).unwrap();
        circuit.add_gate(QuantumGate::CNOT, vec![0, 1]).unwrap();
        
        assert_eq!(circuit.instructions.len(), 3);
        assert!(circuit.optimize(OptimizationLevel::Basic).is_ok());
        assert_eq!(circuit.instructions.len(), 2); // Identity should be removed
    }

    #[tokio::test]
    async fn test_circuit_execution() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        circuit.add_cnot(0, 1).unwrap();
        
        let config = ExecutionConfig::default();
        let executor = CircuitExecution::new(config);
        let result = executor.execute(&mut circuit).await;
        
        assert!(result.is_ok());
        let execution_result = result.unwrap();
        assert!(execution_result.success);
        assert!(execution_result.measurement_results.is_some());
    }

    #[test]
    fn test_circuit_builder() {
        let circuit = CircuitBuilder::new("test".to_string(), 3)
            .hadamard(0).unwrap()
            .cnot(0, 1).unwrap()
            .pauli_x(2).unwrap()
            .rotation_x(1, std::f64::consts::PI / 4.0).unwrap()
            .build();
        
        assert_eq!(circuit.num_qubits, 3);
        assert_eq!(circuit.instructions.len(), 4);
        
        // Check gate types
        assert!(matches!(circuit.instructions[0].gate, QuantumGate::Hadamard { .. }));
        assert!(matches!(circuit.instructions[1].gate, QuantumGate::CNOT { .. }));
        assert!(matches!(circuit.instructions[2].gate, QuantumGate::PauliX { .. }));
        assert!(matches!(circuit.instructions[3].gate, QuantumGate::RX { .. }));
    }

    #[test]
    fn test_execution_config() {
        let config = ExecutionConfig {
            backend: ExecutionBackend::QuantumHardware,
            optimization_level: OptimizationLevel::Maximum,
            shots: 2048,
            timeout_ms: 60000,
            max_memory_mb: 2048,
            parallel_execution: true,
        };
        
        assert!(matches!(config.backend, ExecutionBackend::QuantumHardware));
        assert!(matches!(config.optimization_level, OptimizationLevel::Maximum));
        assert_eq!(config.shots, 2048);
    }

    #[test]
    fn test_parametrized_gates() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        let angle = std::f64::consts::PI / 4.0;
        
        let rx_gate = QuantumGate::RX { qubit: 0, angle };
        assert!(circuit.add_parametrized_gate(
            rx_gate, 
            vec![0], 
            vec![angle]
        ).is_ok());
        
        assert_eq!(circuit.instructions.len(), 1);
        assert!(circuit.instructions[0].parameters.is_some());
        assert_eq!(circuit.instructions[0].parameters.as_ref().unwrap()[0], angle);
    }

    #[tokio::test]
    async fn test_execution_history() {
        let config = ExecutionConfig::default();
        let executor = CircuitExecution::new(config);
        
        let mut circuit1 = QuantumCircuit::new("test1".to_string(), 2);
        circuit1.add_hadamard(0).unwrap();
        
        let mut circuit2 = QuantumCircuit::new("test2".to_string(), 2);
        circuit2.add_pauli_x(1).unwrap();
        
        executor.execute(&mut circuit1).await.unwrap();
        executor.execute(&mut circuit2).await.unwrap();
        
        let history = executor.get_execution_history().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].circuit_id, circuit1.id);
        assert_eq!(history[1].circuit_id, circuit2.id);
    }

    #[tokio::test]
    async fn test_different_execution_backends() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        
        let backends = vec![
            ExecutionBackend::Simulator,
            ExecutionBackend::ClassicalOptimized,
            ExecutionBackend::QuantumHardware,
            ExecutionBackend::HybridGpu,
        ];
        
        for backend in backends {
            let config = ExecutionConfig {
                backend,
                ..Default::default()
            };
            
            let executor = CircuitExecution::new(config);
            let result = executor.execute(&mut circuit.clone()).await;
            assert!(result.is_ok());
        }
    }

    #[test]
    fn test_circuit_stats() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 3);
        circuit.add_hadamard(0).unwrap();
        circuit.add_cnot(0, 1).unwrap();
        circuit.add_gate(QuantumGate::PauliX, vec![2]).unwrap();
        
        let stats = circuit.calculate_stats().unwrap();
        assert_eq!(stats.total_gates, 3);
        assert_eq!(stats.depth, 3);
        assert!(stats.memory_usage_bytes > 0);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let circuit = QuantumCircuit::new("test".to_string(), 4);
        let estimated_memory = circuit.estimate_memory_usage();
        
        // For 4 qubits: 2^4 = 16 complex numbers * 16 bytes = 256 bytes minimum
        assert!(estimated_memory >= 256);
    }

    #[test]
    fn test_circuit_metadata() {
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.metadata.insert("author".to_string(), "test_user".to_string());
        circuit.metadata.insert("purpose".to_string(), "testing".to_string());
        
        assert_eq!(circuit.metadata.get("author"), Some(&"test_user".to_string()));
        assert_eq!(circuit.metadata.get("purpose"), Some(&"testing".to_string()));
    }

    #[tokio::test]
    async fn test_circuit_reset_history() {
        let config = ExecutionConfig::default();
        let executor = CircuitExecution::new(config);
        
        let mut circuit = QuantumCircuit::new("test".to_string(), 2);
        circuit.add_hadamard(0).unwrap();
        
        executor.execute(&mut circuit).await.unwrap();
        assert_eq!(executor.get_execution_history().await.len(), 1);
        
        executor.reset_history().await;
        assert_eq!(executor.get_execution_history().await.len(), 0);
    }

    #[test]
    fn test_comprehensive_circuit_operations() {
        let mut circuit = QuantumCircuit::new("comprehensive_test".to_string(), 4);
        
        // Add various gates
        circuit.add_hadamard(0).unwrap();
        circuit.add_pauli_x(1).unwrap();
        circuit.add_pauli_y(2).unwrap();
        circuit.add_pauli_z(3).unwrap();
        circuit.add_cnot(0, 1).unwrap();
        let cz_gate = QuantumGate::CZ { control: 2, target: 3 };
        circuit.add_gate(cz_gate, vec![2, 3]).unwrap();
        let toffoli_gate = QuantumGate::Toffoli { control1: 0, control2: 1, target: 2 };
        circuit.add_gate(toffoli_gate, vec![0, 1, 2]).unwrap();
        
        // Add parametrized gates
        let rx_gate = QuantumGate::RX { qubit: 0, angle: std::f64::consts::PI / 2.0 };
        circuit.add_parametrized_gate(rx_gate, vec![0], vec![std::f64::consts::PI / 2.0]).unwrap();
        let ry_gate = QuantumGate::RY { qubit: 1, angle: std::f64::consts::PI / 4.0 };
        circuit.add_parametrized_gate(ry_gate, vec![1], vec![std::f64::consts::PI / 4.0]).unwrap();
        let rz_gate = QuantumGate::RZ { qubit: 2, angle: std::f64::consts::PI / 8.0 };
        circuit.add_parametrized_gate(rz_gate, vec![2], vec![std::f64::consts::PI / 8.0]).unwrap();
        let phase_gate = QuantumGate::Phase { qubit: 3, phase: std::f64::consts::PI / 6.0 };
        circuit.add_parametrized_gate(phase_gate, vec![3], vec![std::f64::consts::PI / 6.0]).unwrap();
        
        // Validate
        assert!(circuit.validate().is_ok());
        
        // Calculate stats
        let stats = circuit.calculate_stats().unwrap();
        assert_eq!(stats.total_gates, 11);
        assert!(stats.gate_counts.len() > 0);
        
        // Test optimization
        assert!(circuit.optimize(OptimizationLevel::Basic).is_ok());
        
        // Test depth calculation
        assert!(circuit.depth() > 0);
    }
}