//! Error types for the quantum core framework

use thiserror::Error;

/// Result type for quantum operations
pub type QuantumResult<T> = Result<T, QuantumError>;

/// Comprehensive error types for quantum operations
#[derive(Error, Debug, Clone)]
pub enum QuantumError {
    /// Invalid quantum state operation
    #[error("Invalid quantum state: {message}")]
    InvalidState {
        /// Error message
        message: String,
    },

    /// Quantum gate operation error
    #[error("Quantum gate error: {gate_type} - {message}")]
    GateError {
        /// Type of gate that failed
        gate_type: String,
        /// Error message
        message: String,
    },

    /// Quantum circuit execution error
    #[error("Circuit execution failed: {circuit_id} - {message}")]
    CircuitError {
        /// Circuit identifier
        circuit_id: String,
        /// Error message
        message: String,
    },

    /// Quantum device error
    #[error("Quantum device error: {device_type} - {message}")]
    DeviceError {
        /// Device type
        device_type: String,
        /// Error message
        message: String,
    },

    /// Hardware detection or initialization error
    #[error("Hardware error: {component} - {message}")]
    HardwareError {
        /// Hardware component
        component: String,
        /// Error message
        message: String,
    },

    /// Memory management error
    #[error("Memory error: {operation} - {message}")]
    MemoryError {
        /// Memory operation
        operation: String,
        /// Error message
        message: String,
    },

    /// Mathematical computation error
    #[error("Computation error: {operation} - {message}")]
    ComputationError {
        /// Mathematical operation
        operation: String,
        /// Error message
        message: String,
    },

    /// Decoherence or quantum state degradation
    #[error("Decoherence error: {message}")]
    DecoherenceError {
        /// Error message
        message: String,
    },

    /// Measurement error
    #[error("Measurement error: {measurement_type} - {message}")]
    MeasurementError {
        /// Type of measurement
        measurement_type: String,
        /// Error message
        message: String,
    },

    /// Entanglement operation error
    #[error("Entanglement error: {qubits:?} - {message}")]
    EntanglementError {
        /// Qubits involved in entanglement
        qubits: Vec<usize>,
        /// Error message
        message: String,
    },

    /// Quantum algorithm specific error
    #[error("Algorithm error: {algorithm} - {message}")]
    AlgorithmError {
        /// Algorithm name
        algorithm: String,
        /// Error message
        message: String,
    },

    /// Serialization/deserialization error
    #[error("Serialization error: {message}")]
    SerializationError {
        /// Error message
        message: String,
    },

    /// Thread safety or concurrency error
    #[error("Concurrency error: {message}")]
    ConcurrencyError {
        /// Error message
        message: String,
    },

    /// Resource exhaustion error
    #[error("Resource exhausted: {resource} - {message}")]
    ResourceExhausted {
        /// Resource type
        resource: String,
        /// Error message
        message: String,
    },

    /// Timeout error for long-running operations
    #[error("Operation timeout: {operation} - {duration_ms}ms")]
    TimeoutError {
        /// Operation that timed out
        operation: String,
        /// Duration in milliseconds
        duration_ms: u64,
    },

    /// Invalid configuration error
    #[error("Invalid configuration: {parameter} - {message}")]
    ConfigurationError {
        /// Configuration parameter
        parameter: String,
        /// Error message
        message: String,
    },

    /// External library or system error
    #[error("External error: {message}")]
    ExternalError {
        /// Error message
        message: String,
    },

    /// Invalid operation error (backwards compatibility)
    #[error("Invalid operation: {operation} - {message}")]
    InvalidOperation {
        /// Operation that failed
        operation: String,
        /// Error message
        message: String,
    },
}

impl QuantumError {
    /// Create a new invalid state error
    pub fn invalid_state(message: impl Into<String>) -> Self {
        Self::InvalidState {
            message: message.into(),
        }
    }

    /// Create a new gate error
    pub fn gate_error(gate_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::GateError {
            gate_type: gate_type.into(),
            message: message.into(),
        }
    }

    /// Create a new circuit error
    pub fn circuit_error(circuit_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self::CircuitError {
            circuit_id: circuit_id.into(),
            message: message.into(),
        }
    }

    /// Create a new device error
    pub fn device_error(device_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::DeviceError {
            device_type: device_type.into(),
            message: message.into(),
        }
    }

    /// Create a new hardware error
    pub fn hardware_error(component: impl Into<String>, message: impl Into<String>) -> Self {
        Self::HardwareError {
            component: component.into(),
            message: message.into(),
        }
    }

    /// Create a new memory error
    pub fn memory_error(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::MemoryError {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Create a new computation error
    pub fn computation_error(operation: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ComputationError {
            operation: operation.into(),
            message: message.into(),
        }
    }

    /// Create a new decoherence error
    pub fn decoherence_error(message: impl Into<String>) -> Self {
        Self::DecoherenceError {
            message: message.into(),
        }
    }

    /// Create a new measurement error
    pub fn measurement_error(measurement_type: impl Into<String>, message: impl Into<String>) -> Self {
        Self::MeasurementError {
            measurement_type: measurement_type.into(),
            message: message.into(),
        }
    }

    /// Create a new entanglement error
    pub fn entanglement_error(qubits: Vec<usize>, message: impl Into<String>) -> Self {
        Self::EntanglementError {
            qubits,
            message: message.into(),
        }
    }

    /// Create a new algorithm error
    pub fn algorithm_error(algorithm: impl Into<String>, message: impl Into<String>) -> Self {
        Self::AlgorithmError {
            algorithm: algorithm.into(),
            message: message.into(),
        }
    }

    /// Create a new timeout error
    pub fn timeout_error(operation: impl Into<String>, duration_ms: u64) -> Self {
        Self::TimeoutError {
            operation: operation.into(),
            duration_ms,
        }
    }

    /// Create a new configuration error
    pub fn configuration_error(parameter: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            parameter: parameter.into(),
            message: message.into(),
        }
    }

    /// Create a new serialization error
    pub fn serialization_error(message: impl Into<String>) -> Self {
        Self::SerializationError {
            message: message.into(),
        }
    }

    /// Create a new concurrency error
    pub fn concurrency_error(message: impl Into<String>) -> Self {
        Self::ConcurrencyError {
            message: message.into(),
        }
    }

    /// Create a new resource exhausted error
    pub fn resource_exhausted(resource: impl Into<String>, message: impl Into<String>) -> Self {
        Self::ResourceExhausted {
            resource: resource.into(),
            message: message.into(),
        }
    }

    /// Create a generic invalid operation error (backwards compatibility)
    pub fn invalid_operation(message: impl Into<String>) -> Self {
        Self::InvalidOperation {
            operation: "generic".to_string(),
            message: message.into(),
        }
    }

    /// Create a generic invalid circuit error (backwards compatibility)
    pub fn invalid_circuit(message: impl Into<String>) -> Self {
        Self::CircuitError {
            circuit_id: "unknown".to_string(),
            message: message.into(),
        }
    }

    /// Create an invalid qubit index error (backwards compatibility)
    pub fn invalid_qubit_index(index: usize, max_index: usize) -> Self {
        Self::InvalidState {
            message: format!("Invalid qubit index: {} (max: {})", index, max_index),
        }
    }

    /// Create a device not found error (backwards compatibility)
    pub fn device_not_found(device_id: impl Into<String>) -> Self {
        Self::DeviceError {
            device_type: "unknown".to_string(),
            message: format!("Device not found: {}", device_id.into()),
        }
    }

    /// Create a no available device error (backwards compatibility)
    pub fn no_available_device() -> Self {
        Self::DeviceError {
            device_type: "any".to_string(),
            message: "No available device found".to_string(),
        }
    }

    /// Create a metric error (backwards compatibility)
    pub fn metric_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            parameter: "metric".to_string(),
            message: message.into(),
        }
    }

    /// Create a processing error (backwards compatibility)
    pub fn processing_error(message: impl Into<String>) -> Self {
        Self::ComputationError {
            operation: "processing".to_string(),
            message: message.into(),
        }
    }

    /// Create a compilation error (backwards compatibility)
    pub fn compilation_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            parameter: "compilation".to_string(),
            message: message.into(),
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            QuantumError::DecoherenceError { .. } => true,
            QuantumError::TimeoutError { .. } => true,
            QuantumError::ResourceExhausted { .. } => true,
            QuantumError::HardwareError { .. } => false,
            QuantumError::DeviceError { .. } => false,
            _ => false,
        }
    }

    /// Check if error indicates quantum advantage failure
    pub fn indicates_quantum_disadvantage(&self) -> bool {
        match self {
            QuantumError::AlgorithmError { algorithm, .. } => {
                algorithm.contains("quantum") && !algorithm.contains("classical")
            }
            QuantumError::DecoherenceError { .. } => true,
            _ => false,
        }
    }
}

/// Error recovery strategies
#[derive(Debug, Clone, PartialEq)]
pub enum RecoveryStrategy {
    /// Retry the operation
    Retry {
        /// Maximum number of retries
        max_retries: usize,
        /// Delay between retries in milliseconds
        delay_ms: u64,
    },
    /// Fallback to classical computation
    FallbackToClassical,
    /// Switch to different quantum device
    SwitchDevice,
    /// Reduce problem size
    ReduceProblemSize,
    /// No recovery possible
    NoRecovery,
}

impl QuantumError {
    /// Get recommended recovery strategy
    pub fn recovery_strategy(&self) -> RecoveryStrategy {
        match self {
            QuantumError::DecoherenceError { .. } => RecoveryStrategy::Retry {
                max_retries: 3,
                delay_ms: 100,
            },
            QuantumError::TimeoutError { .. } => RecoveryStrategy::ReduceProblemSize,
            QuantumError::ResourceExhausted { .. } => RecoveryStrategy::ReduceProblemSize,
            QuantumError::DeviceError { .. } => RecoveryStrategy::SwitchDevice,
            QuantumError::AlgorithmError { .. } => RecoveryStrategy::FallbackToClassical,
            QuantumError::InvalidOperation { .. } => RecoveryStrategy::NoRecovery,
            _ => RecoveryStrategy::NoRecovery,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let error = QuantumError::invalid_state("Test state error");
        assert!(matches!(error, QuantumError::InvalidState { .. }));
    }

    #[test]
    fn test_error_recoverability() {
        let recoverable = QuantumError::decoherence_error("Test decoherence");
        assert!(recoverable.is_recoverable());

        let non_recoverable = QuantumError::device_error("test", "Test device error");
        assert!(!non_recoverable.is_recoverable());
    }

    #[test]
    fn test_recovery_strategy() {
        let error = QuantumError::timeout_error("test_operation", 5000);
        let strategy = error.recovery_strategy();
        assert!(matches!(strategy, RecoveryStrategy::ReduceProblemSize));
    }

    #[test]
    fn test_quantum_disadvantage_detection() {
        let error = QuantumError::algorithm_error("quantum_algorithm", "Failed to converge");
        assert!(error.indicates_quantum_disadvantage());

        let error = QuantumError::gate_error("hadamard", "Invalid qubit index");
        assert!(!error.indicates_quantum_disadvantage());
    }
}