//! pBit Execution Backend
//!
//! Provides a unified backend for executing quantum circuits using pBit dynamics.
//! This backend can serve as a drop-in replacement for quantum simulators.

use crate::error::{QuantumError, QuantumResult};
use crate::pbit_gates::PBitCircuit;
use crate::pbit_state::{PBitConfig, PBitState};
use crate::traits::{CircuitParams, DeviceCapabilities, DeviceStatus, ExecutionContext};
use crate::QuantumResult as QResult;
use crate::{ComputationMetadata, QuantumState};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

/// pBit Backend Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitBackendConfig {
    /// Base pBit configuration
    pub pbit_config: PBitConfig,
    /// Number of equilibration sweeps per gate
    pub sweeps_per_gate: usize,
    /// Number of measurement samples
    pub num_samples: usize,
    /// Annealing steps for optimization
    pub annealing_steps: usize,
    /// Target temperature after annealing
    pub target_temperature: f64,
    /// Enable automatic annealing
    pub auto_anneal: bool,
    /// Decoherence rate (0 = no decoherence)
    pub decoherence_rate: f64,
    /// Backend name identifier
    pub name: String,
}

impl Default for PBitBackendConfig {
    fn default() -> Self {
        Self {
            pbit_config: PBitConfig::default(),
            sweeps_per_gate: 10,
            num_samples: 1024,
            annealing_steps: 100,
            target_temperature: 0.1,
            auto_anneal: false,
            decoherence_rate: 0.0,
            name: "pbit-simulator".to_string(),
        }
    }
}

/// pBit Backend for quantum circuit execution
#[derive(Debug)]
pub struct PBitBackend {
    config: PBitBackendConfig,
    /// Current state (if initialized)
    state: Option<PBitState>,
    /// Execution statistics
    stats: BackendStats,
    /// Is backend ready
    ready: bool,
}

/// Backend execution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BackendStats {
    pub total_executions: u64,
    pub total_gates_applied: u64,
    pub total_measurements: u64,
    pub total_sweeps: u64,
    pub total_time_ns: u64,
    pub average_fidelity: f64,
}

impl PBitBackend {
    /// Create a new pBit backend
    pub fn new(config: PBitBackendConfig) -> Self {
        Self {
            config,
            state: None,
            stats: BackendStats::default(),
            ready: true,
        }
    }

    /// Create with default configuration
    pub fn default_backend() -> Self {
        Self::new(PBitBackendConfig::default())
    }

    /// Initialize state for given number of qubits
    pub fn initialize(&mut self, num_qubits: usize) -> QuantumResult<()> {
        self.state = Some(PBitState::with_config(
            num_qubits,
            self.config.pbit_config.clone(),
        )?);
        Ok(())
    }

    /// Initialize from existing quantum state
    pub fn initialize_from_quantum(&mut self, qs: &QuantumState) -> QuantumResult<()> {
        self.state = Some(PBitState::from_quantum_state(qs)?);
        Ok(())
    }

    /// Execute a pBit circuit
    pub fn execute_circuit(&mut self, circuit: &PBitCircuit) -> QuantumResult<PBitExecutionResult> {
        let start = Instant::now();

        // Ensure state is initialized
        if self.state.is_none() {
            self.initialize(circuit.num_qubits())?;
        }

        // Execute circuit and apply dynamics
        {
            let state = self.state.as_mut().ok_or_else(|| {
                QuantumError::computation_error("execute", "State not initialized")
            })?;

            // Execute circuit
            circuit.execute(state)?;

            // Apply equilibration sweeps
            for _ in 0..self.config.sweeps_per_gate {
                state.sweep();
            }

            // Apply annealing if enabled
            if self.config.auto_anneal {
                state.anneal(self.config.target_temperature, self.config.annealing_steps);
            }

            // Apply decoherence if configured
            if self.config.decoherence_rate > 0.0 {
                let time_step = circuit.num_gates() as f64 * 0.001; // 1ms per gate
                state.apply_decoherence(self.config.decoherence_rate, time_step);
            }
        }

        // Collect samples (separate borrow scope)
        let samples = self.collect_samples(self.config.num_samples)?;

        // Get final state metrics
        let state = self.state.as_ref().ok_or_else(|| {
            QuantumError::computation_error("metrics", "State not initialized")
        })?;

        // Compute probabilities from samples
        let num_states = 1 << state.num_qubits();
        let mut counts = vec![0usize; num_states];
        for sample in &samples {
            if *sample < num_states {
                counts[*sample] += 1;
            }
        }
        let probabilities: Vec<f64> = counts
            .iter()
            .map(|&c| c as f64 / samples.len() as f64)
            .collect();

        let elapsed = start.elapsed();

        // Update statistics
        self.stats.total_executions += 1;
        self.stats.total_gates_applied += circuit.num_gates() as u64;
        self.stats.total_measurements += self.config.num_samples as u64;
        self.stats.total_sweeps += (self.config.sweeps_per_gate * circuit.num_gates()) as u64;
        self.stats.total_time_ns += elapsed.as_nanos() as u64;

        Ok(PBitExecutionResult {
            probabilities,
            samples,
            counts,
            execution_time_ns: elapsed.as_nanos() as u64,
            num_qubits: state.num_qubits(),
            num_gates: circuit.num_gates(),
            final_entropy: state.entropy(),
            final_magnetization: state.magnetization(),
        })
    }

    /// Collect measurement samples
    fn collect_samples(&mut self, num_samples: usize) -> QuantumResult<Vec<usize>> {
        let state = self.state.as_mut().ok_or_else(|| {
            QuantumError::computation_error("sample", "State not initialized")
        })?;

        let probabilities = state.probability_distribution();
        let mut samples = Vec::with_capacity(num_samples);

        for _ in 0..num_samples {
            let random_value: f64 = rand::random();
            let mut cumulative = 0.0;

            for (idx, &prob) in probabilities.iter().enumerate() {
                cumulative += prob;
                if random_value <= cumulative {
                    samples.push(idx);
                    break;
                }
            }
        }

        Ok(samples)
    }

    /// Execute and convert to standard QuantumResult
    pub fn execute_to_quantum_result(
        &mut self,
        circuit: &PBitCircuit,
    ) -> QuantumResult<QResult> {
        let result = self.execute_circuit(circuit)?;

        // Convert back to QuantumState
        let state = self.state.as_ref().ok_or_else(|| {
            QuantumError::computation_error("convert", "State not initialized")
        })?;

        let quantum_state = state.to_quantum_state()?;

        let metadata = ComputationMetadata {
            num_qubits: result.num_qubits,
            gate_count: result.num_gates,
            circuit_depth: circuit.depth(),
            backend: self.config.name.clone(),
            error_correction: false,
        };

        Ok(QResult {
            state: quantum_state,
            probabilities: result.probabilities,
            metadata,
            fidelity: 1.0 - result.final_entropy / (result.num_qubits as f64).ln(),
            execution_time_ns: result.execution_time_ns,
        })
    }

    /// Get current state
    pub fn state(&self) -> Option<&PBitState> {
        self.state.as_ref()
    }

    /// Get mutable state
    pub fn state_mut(&mut self) -> Option<&mut PBitState> {
        self.state.as_mut()
    }

    /// Get execution statistics
    pub fn stats(&self) -> &BackendStats {
        &self.stats
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        self.stats = BackendStats::default();
    }

    /// Get device capabilities
    pub fn capabilities(&self) -> DeviceCapabilities {
        DeviceCapabilities {
            max_qubits: 32,
            supported_gates: vec![
                "H".to_string(),
                "X".to_string(),
                "Y".to_string(),
                "Z".to_string(),
                "RX".to_string(),
                "RY".to_string(),
                "RZ".to_string(),
                "S".to_string(),
                "T".to_string(),
                "CNOT".to_string(),
                "CZ".to_string(),
                "SWAP".to_string(),
                "CCX".to_string(),
                "CSWAP".to_string(),
            ],
            gate_fidelities: {
                let mut fidelities = HashMap::new();
                fidelities.insert("H".to_string(), 0.999);
                fidelities.insert("X".to_string(), 0.9999);
                fidelities.insert("CNOT".to_string(), 0.995);
                fidelities
            },
            connectivity: Vec::new(), // All-to-all
            coherence_times: vec![1000.0; 32], // 1ms per qubit
            error_rates: {
                let mut rates = HashMap::new();
                rates.insert("single_qubit".to_string(), 0.001);
                rates.insert("two_qubit".to_string(), 0.01);
                rates
            },
        }
    }

    /// Get device status
    pub fn status(&self) -> DeviceStatus {
        DeviceStatus {
            online: self.ready,
            queue_length: 0,
            utilization: 0.0,
            error_rate: 0.001,
            last_calibration: chrono::Utc::now(),
            message: "pBit simulator ready".to_string(),
        }
    }
}

/// Result of pBit circuit execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitExecutionResult {
    /// Probability distribution over basis states
    pub probabilities: Vec<f64>,
    /// Raw measurement samples
    pub samples: Vec<usize>,
    /// Counts per basis state
    pub counts: Vec<usize>,
    /// Execution time in nanoseconds
    pub execution_time_ns: u64,
    /// Number of qubits
    pub num_qubits: usize,
    /// Number of gates executed
    pub num_gates: usize,
    /// Final state entropy
    pub final_entropy: f64,
    /// Final magnetization
    pub final_magnetization: f64,
}

impl PBitExecutionResult {
    /// Get most likely measurement outcome
    pub fn most_likely_outcome(&self) -> usize {
        self.probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }

    /// Get expectation value of Z operator on qubit
    pub fn expectation_z(&self, qubit: usize) -> f64 {
        let mut expectation = 0.0;
        for (state_idx, &prob) in self.probabilities.iter().enumerate() {
            let bit = (state_idx >> qubit) & 1;
            let z_value = if bit == 0 { 1.0 } else { -1.0 };
            expectation += prob * z_value;
        }
        expectation
    }

    /// Convert to bitstring histogram
    pub fn to_histogram(&self) -> HashMap<String, usize> {
        let mut histogram = HashMap::new();
        for (idx, &count) in self.counts.iter().enumerate() {
            if count > 0 {
                let bitstring = format!("{:0width$b}", idx, width = self.num_qubits);
                histogram.insert(bitstring, count);
            }
        }
        histogram
    }
}

/// Thread-safe pBit backend pool
#[derive(Debug)]
pub struct PBitBackendPool {
    backends: Vec<Arc<RwLock<PBitBackend>>>,
    config: PBitBackendConfig,
}

impl PBitBackendPool {
    /// Create a new backend pool
    pub fn new(pool_size: usize, config: PBitBackendConfig) -> Self {
        let backends = (0..pool_size)
            .map(|_| Arc::new(RwLock::new(PBitBackend::new(config.clone()))))
            .collect();

        Self { backends, config }
    }

    /// Get an available backend
    pub async fn acquire(&self) -> Arc<RwLock<PBitBackend>> {
        // Simple round-robin; could be enhanced with load balancing
        let idx = rand::random::<usize>() % self.backends.len();
        self.backends[idx].clone()
    }

    /// Execute circuit on any available backend
    pub async fn execute(&self, circuit: &PBitCircuit) -> QuantumResult<PBitExecutionResult> {
        let backend = self.acquire().await;
        let mut guard = backend.write().await;
        guard.execute_circuit(circuit)
    }

    /// Get pool size
    pub fn size(&self) -> usize {
        self.backends.len()
    }
}

/// Trait for types that can be executed on pBit backend
#[async_trait]
pub trait PBitExecutable {
    /// Execute on pBit backend
    async fn execute_pbit(&self, backend: &mut PBitBackend) -> QuantumResult<PBitExecutionResult>;
}

#[async_trait]
impl PBitExecutable for PBitCircuit {
    async fn execute_pbit(&self, backend: &mut PBitBackend) -> QuantumResult<PBitExecutionResult> {
        backend.execute_circuit(self)
    }
}

/// Factory for creating configured backends
pub struct PBitBackendFactory;

impl PBitBackendFactory {
    /// Create a high-accuracy backend (low temperature, many sweeps)
    pub fn high_accuracy() -> PBitBackend {
        PBitBackend::new(PBitBackendConfig {
            pbit_config: PBitConfig {
                temperature: 0.01,
                coupling_strength: 10.0,
                ..Default::default()
            },
            sweeps_per_gate: 100,
            num_samples: 8192,
            auto_anneal: true,
            annealing_steps: 500,
            target_temperature: 0.001,
            ..Default::default()
        })
    }

    /// Create a fast backend (higher temperature, fewer sweeps)
    pub fn fast() -> PBitBackend {
        PBitBackend::new(PBitBackendConfig {
            pbit_config: PBitConfig {
                temperature: 1.0,
                coupling_strength: 1.0,
                ..Default::default()
            },
            sweeps_per_gate: 1,
            num_samples: 256,
            auto_anneal: false,
            ..Default::default()
        })
    }

    /// Create a backend with decoherence modeling
    pub fn with_decoherence(rate: f64) -> PBitBackend {
        PBitBackend::new(PBitBackendConfig {
            decoherence_rate: rate,
            ..Default::default()
        })
    }

    /// Create a deterministic backend (for testing)
    pub fn deterministic(seed: u64) -> PBitBackend {
        PBitBackend::new(PBitBackendConfig {
            pbit_config: PBitConfig {
                seed: Some(seed),
                temperature: 0.01,
                ..Default::default()
            },
            auto_anneal: true,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pbit_gates::*;

    #[test]
    fn test_backend_creation() {
        let backend = PBitBackend::default_backend();
        assert!(backend.ready);
    }

    #[test]
    fn test_circuit_execution() {
        let mut backend = PBitBackend::default_backend();

        let mut circuit = PBitCircuit::new(2);
        circuit.h(0).cnot(0, 1);

        let result = backend.execute_circuit(&circuit).unwrap();

        assert_eq!(result.num_qubits, 2);
        assert_eq!(result.num_gates, 2);
        assert!(!result.samples.is_empty());
    }

    #[test]
    fn test_measurement_samples() {
        let mut backend = PBitBackendFactory::deterministic(42);
        backend.initialize(1).unwrap();

        let mut circuit = PBitCircuit::new(1);
        circuit.h(0); // Superposition

        let result = backend.execute_circuit(&circuit).unwrap();

        // Should have roughly equal counts of 0 and 1
        let count_0 = result.counts[0];
        let count_1 = result.counts[1];
        let total = count_0 + count_1;

        assert!(count_0 > total / 4);
        assert!(count_1 > total / 4);
    }

    #[test]
    fn test_expectation_value() {
        let mut backend = PBitBackend::default_backend();

        // |0⟩ state should have <Z> = 1
        let mut circuit = PBitCircuit::new(1);
        // No gates, stays in |0⟩

        let result = backend.execute_circuit(&circuit).unwrap();
        let exp_z = result.expectation_z(0);

        assert!(exp_z > 0.9); // Should be close to 1
    }

    #[test]
    fn test_factory_backends() {
        let fast = PBitBackendFactory::fast();
        assert_eq!(fast.config.sweeps_per_gate, 1);

        let accurate = PBitBackendFactory::high_accuracy();
        assert_eq!(accurate.config.sweeps_per_gate, 100);
    }

    #[tokio::test]
    async fn test_backend_pool() {
        let pool = PBitBackendPool::new(4, PBitBackendConfig::default());
        assert_eq!(pool.size(), 4);

        let mut circuit = PBitCircuit::new(2);
        circuit.h(0);

        let result = pool.execute(&circuit).await.unwrap();
        assert_eq!(result.num_qubits, 2);
    }
}
