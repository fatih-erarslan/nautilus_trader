//! # Quantum Module
//!
//! Quantum computing functionality for parasitic organisms.
//! Provides classical-enhanced quantum simulation for trading applications.

pub mod grover;
pub mod grover_demo;
pub mod memory;
pub mod quantum_simulators;

// Re-export types
pub use grover::{
    ExploitationStrategy, GroverAlgorithmType, GroverOracle, GroverSearchConfig,
    GroverSearchEngine, GroverSearchResult, MarketOpportunityOracle, OrganismConfigOracle,
    PatternMatch, ProfitablePatternOracle, TradeOutcome, TradingPattern,
};
pub use grover_demo::GroverDemo;
pub use memory::{BiologicalMemorySystem, QuantumTradingMemory};
pub use quantum_simulators::{
    QuantumCircuit, QuantumGate, QuantumResult, QuantumSimulators, StatevectorSimulator,
};

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU8, Ordering};

/// Global quantum runtime mode state
static QUANTUM_MODE: AtomicU8 = AtomicU8::new(0); // 0 = Classical

/// Quantum computing mode for the trading system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum QuantumMode {
    /// Classical algorithms with quantum-inspired optimizations
    Classical = 0,
    /// Quantum-enhanced classical algorithms (hybrid approach)
    Enhanced = 1,
    /// Full quantum computing simulation
    Full = 2,
}

impl Default for QuantumMode {
    fn default() -> Self {
        QuantumMode::Classical
    }
}

impl From<u8> for QuantumMode {
    fn from(value: u8) -> Self {
        match value {
            0 => QuantumMode::Classical,
            1 => QuantumMode::Enhanced,
            2 => QuantumMode::Full,
            _ => QuantumMode::Classical,
        }
    }
}

impl QuantumMode {
    /// Get the current global quantum mode
    #[inline]
    pub fn current() -> Self {
        QUANTUM_MODE.load(Ordering::Relaxed).into()
    }

    /// Set the global quantum mode
    #[inline]
    pub fn set_global(mode: Self) {
        QUANTUM_MODE.store(mode as u8, Ordering::Relaxed);
    }

    /// Check if quantum features are enabled
    #[inline]
    pub fn is_quantum_enabled(&self) -> bool {
        matches!(self, QuantumMode::Enhanced | QuantumMode::Full)
    }

    /// Check if currently in full quantum mode
    #[inline]
    pub fn is_full_quantum(&self) -> bool {
        matches!(self, QuantumMode::Full)
    }
}

/// Quantum state representation for organism quantum operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumState {
    /// Number of qubits in the system
    num_qubits: u32,
    /// State vector representing quantum amplitudes
    state_vector: Vec<Complex64>,
    /// Current quantum mode
    mode: QuantumMode,
}

impl QuantumState {
    /// Create a new quantum state with specified number of qubits
    pub fn new(num_qubits: u32) -> Self {
        let num_states = 1usize << num_qubits;
        let mut state_vector = vec![Complex64::new(0.0, 0.0); num_states];
        state_vector[0] = Complex64::new(1.0, 0.0); // Initialize to |00...0⟩

        Self {
            num_qubits,
            state_vector,
            mode: QuantumMode::current(),
        }
    }

    /// Initialize superposition across all basis states
    pub fn initialize_superposition(&mut self) {
        let amplitude = Complex64::new(1.0 / (self.state_vector.len() as f64).sqrt(), 0.0);
        for state in &mut self.state_vector {
            *state = amplitude;
        }
    }

    /// Apply Hadamard gate to create superposition
    pub fn apply_hadamard_gate(&mut self, qubit: u32) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::InvalidQubit(qubit));
        }

        let sqrt2_inv = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        let mask = 1usize << qubit;

        for i in 0..self.state_vector.len() {
            if i & mask == 0 {
                let j = i | mask;
                if j < self.state_vector.len() {
                    let amp0 = self.state_vector[i];
                    let amp1 = self.state_vector[j];

                    self.state_vector[i] = sqrt2_inv * (amp0 + amp1);
                    self.state_vector[j] = sqrt2_inv * (amp0 - amp1);
                }
            }
        }

        Ok(())
    }

    /// Apply CNOT (controlled-NOT) gate
    pub fn apply_controlled_not(&mut self, control: u32, target: u32) -> Result<(), QuantumError> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantumError::InvalidQubit(control.max(target)));
        }

        let control_mask = 1usize << control;
        let target_mask = 1usize << target;

        for i in 0..self.state_vector.len() {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;
                if j < self.state_vector.len() {
                    self.state_vector.swap(i, j);
                }
            }
        }

        Ok(())
    }

    /// Apply phase gate
    pub fn apply_phase_gate(&mut self, qubit: u32, phase: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::InvalidQubit(qubit));
        }

        let phase_factor = Complex64::new(0.0, phase).exp();
        let mask = 1usize << qubit;

        for i in 0..self.state_vector.len() {
            if (i & mask) != 0 {
                self.state_vector[i] *= phase_factor;
            }
        }

        Ok(())
    }

    /// Apply X rotation gate
    pub fn apply_rotation_x(&mut self, qubit: u32, angle: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::InvalidQubit(qubit));
        }

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let mask = 1usize << qubit;

        for i in 0..self.state_vector.len() {
            if i & mask == 0 {
                let j = i | mask;
                if j < self.state_vector.len() {
                    let amp0 = self.state_vector[i];
                    let amp1 = self.state_vector[j];

                    self.state_vector[i] = cos_half * amp0 - Complex64::new(0.0, sin_half) * amp1;
                    self.state_vector[j] = cos_half * amp1 - Complex64::new(0.0, sin_half) * amp0;
                }
            }
        }

        Ok(())
    }

    /// Apply Y rotation gate
    pub fn apply_rotation_y(&mut self, qubit: u32, angle: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::InvalidQubit(qubit));
        }

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let mask = 1usize << qubit;

        for i in 0..self.state_vector.len() {
            if i & mask == 0 {
                let j = i | mask;
                if j < self.state_vector.len() {
                    let amp0 = self.state_vector[i];
                    let amp1 = self.state_vector[j];

                    self.state_vector[i] = cos_half * amp0 - sin_half * amp1;
                    self.state_vector[j] = cos_half * amp1 + sin_half * amp0;
                }
            }
        }

        Ok(())
    }

    /// Measure a specific qubit
    pub fn measure_qubit(&mut self, qubit: u32) -> Result<bool, QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::InvalidQubit(qubit));
        }

        // Calculate probability of measuring |1⟩
        let mask = 1usize << qubit;
        let prob_one: f64 = self
            .state_vector
            .iter()
            .enumerate()
            .filter(|(i, _)| (i & mask) != 0)
            .map(|(_, amp)| amp.norm_sqr())
            .sum();

        // Perform measurement
        let measurement_result = fastrand::f64() < prob_one;

        // Collapse the wavefunction
        let normalization = if measurement_result {
            prob_one.sqrt()
        } else {
            (1.0 - prob_one).sqrt()
        };

        if normalization > 1e-10 {
            for (i, amp) in self.state_vector.iter_mut().enumerate() {
                if ((i & mask) != 0) == measurement_result {
                    *amp /= normalization;
                } else {
                    *amp = Complex64::new(0.0, 0.0);
                }
            }
        }

        Ok(measurement_result)
    }

    /// Get probabilities for all basis states
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.state_vector.iter().map(|amp| amp.norm_sqr()).collect()
    }

    /// Get the number of qubits
    pub fn num_qubits(&self) -> u32 {
        self.num_qubits
    }
}

/// Quantum error types
#[derive(Debug, thiserror::Error)]
pub enum QuantumError {
    #[error("Invalid qubit index: {0}")]
    InvalidQubit(u32),
    #[error("Quantum operation failed: {0}")]
    OperationFailed(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("Simulation error: {0}")]
    Simulation(String),
    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),
    #[error("Mode switch error: {0}")]
    ModeSwitch(String),
}

/// Quantum configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    /// Default quantum mode
    pub default_mode: QuantumMode,
    /// Enable automatic mode switching
    pub auto_switch: bool,
    /// Maximum qubits for simulation
    pub max_qubits: u32,
    /// Maximum circuit depth for quantum operations
    pub max_circuit_depth: u32,
    /// Enable quantum error correction
    pub error_correction: bool,
    /// Noise model configuration
    pub noise_model: QuantumNoiseConfig,
}

/// Quantum noise model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumNoiseConfig {
    /// Enable decoherence modeling
    pub enable_decoherence: bool,
    /// T1 relaxation time (microseconds)
    pub t1_relaxation: f64,
    /// T2 dephasing time (microseconds)
    pub t2_dephasing: f64,
    /// Gate error probability
    pub gate_error_rate: f64,
    /// Measurement error probability
    pub measurement_error_rate: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            default_mode: QuantumMode::Classical,
            auto_switch: false,
            max_qubits: 20,
            max_circuit_depth: 100,
            error_correction: false,
            noise_model: QuantumNoiseConfig::default(),
        }
    }
}

impl Default for QuantumNoiseConfig {
    fn default() -> Self {
        Self {
            enable_decoherence: true,
            t1_relaxation: 100.0,
            t2_dephasing: 50.0,
            gate_error_rate: 0.001,
            measurement_error_rate: 0.01,
        }
    }
}

/// Initialize quantum runtime from environment
pub fn init_quantum_runtime() -> QuantumConfig {
    QuantumConfig::default()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantum_mode_switching() {
        assert_eq!(QuantumMode::current(), QuantumMode::Classical);

        QuantumMode::set_global(QuantumMode::Enhanced);
        assert_eq!(QuantumMode::current(), QuantumMode::Enhanced);

        // Reset
        QuantumMode::set_global(QuantumMode::Classical);
    }

    #[test]
    fn test_quantum_state_creation() {
        let qs = QuantumState::new(2);
        assert_eq!(qs.num_qubits(), 2);
        assert_eq!(qs.state_vector.len(), 4);
    }

    #[test]
    fn test_quantum_gates() {
        let mut qs = QuantumState::new(2);

        // Test Hadamard gate
        assert!(qs.apply_hadamard_gate(0).is_ok());

        // Test CNOT gate
        assert!(qs.apply_controlled_not(0, 1).is_ok());

        // Test phase gate
        assert!(qs.apply_phase_gate(0, std::f64::consts::PI / 2.0).is_ok());

        // Test measurement
        let result = qs.measure_qubit(0);
        assert!(result.is_ok());
    }
}
