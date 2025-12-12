//! # Quantum Uncertainty Error Types
//!
//! This module defines error types for the quantum uncertainty quantification system.

use thiserror::Error;

/// Quantum uncertainty error types
#[derive(Error, Debug)]
pub enum QuantumUncertaintyError {
    /// Quantum circuit simulation error
    #[error("Quantum circuit simulation error: {message}")]
    QuantumCircuitError { message: String },

    /// Quantum state error
    #[error("Quantum state error: {message}")]
    QuantumStateError { message: String },

    /// Feature extraction error
    #[error("Feature extraction error: {message}")]
    FeatureExtractionError { message: String },

    /// Correlation analysis error
    #[error("Correlation analysis error: {message}")]
    CorrelationAnalysisError { message: String },

    /// Conformal prediction error
    #[error("Conformal prediction error: {message}")]
    ConformalPredictionError { message: String },

    /// Measurement optimization error
    #[error("Measurement optimization error: {message}")]
    MeasurementOptimizationError { message: String },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigurationError { message: String },

    /// Numerical computation error
    #[error("Numerical computation error: {message}")]
    NumericalError { message: String },

    /// IO error
    #[error("IO error: {source}")]
    IoError {
        #[from]
        source: std::io::Error,
    },

    /// Serialization error
    #[error("Serialization error: {source}")]
    SerializationError {
        #[from]
        source: serde_json::Error,
    },

    /// External library error
    #[error("External library error: {message}")]
    ExternalLibraryError { message: String },
}

/// Result type for quantum uncertainty operations
pub type Result<T> = std::result::Result<T, QuantumUncertaintyError>;

impl QuantumUncertaintyError {
    /// Create a quantum circuit error
    pub fn quantum_circuit_error(message: impl Into<String>) -> Self {
        Self::QuantumCircuitError {
            message: message.into(),
        }
    }

    /// Create a quantum state error
    pub fn quantum_state_error(message: impl Into<String>) -> Self {
        Self::QuantumStateError {
            message: message.into(),
        }
    }

    /// Create a feature extraction error
    pub fn feature_extraction_error(message: impl Into<String>) -> Self {
        Self::FeatureExtractionError {
            message: message.into(),
        }
    }

    /// Create a correlation analysis error
    pub fn correlation_analysis_error(message: impl Into<String>) -> Self {
        Self::CorrelationAnalysisError {
            message: message.into(),
        }
    }

    /// Create a conformal prediction error
    pub fn conformal_prediction_error(message: impl Into<String>) -> Self {
        Self::ConformalPredictionError {
            message: message.into(),
        }
    }

    /// Create a measurement optimization error
    pub fn measurement_optimization_error(message: impl Into<String>) -> Self {
        Self::MeasurementOptimizationError {
            message: message.into(),
        }
    }

    /// Create a configuration error
    pub fn configuration_error(message: impl Into<String>) -> Self {
        Self::ConfigurationError {
            message: message.into(),
        }
    }

    /// Create a numerical error
    pub fn numerical_error(message: impl Into<String>) -> Self {
        Self::NumericalError {
            message: message.into(),
        }
    }

    /// Create an external library error
    pub fn external_library_error(message: impl Into<String>) -> Self {
        Self::ExternalLibraryError {
            message: message.into(),
        }
    }
}