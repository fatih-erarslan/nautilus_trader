//! Error types for the hyperphysics-tengri-bridge crate

use thiserror::Error;

/// Errors that can occur in the HyperPhysics-Tengri bridge
#[derive(Error, Debug)]
pub enum BridgeError {
    /// Autopoiesis integration error
    #[error("Autopoiesis error: {message}")]
    AutopoiesisError {
        /// Error message
        message: String,
    },

    /// Consciousness integration error
    #[error("Consciousness error: {message}")]
    ConsciousnessError {
        /// Error message
        message: String,
    },

    /// Thermodynamic integration error
    #[error("Thermo error: {message}")]
    ThermoError {
        /// Error message
        message: String,
    },

    /// Risk integration error
    #[error("Risk error: {message}")]
    RiskError {
        /// Error message
        message: String,
    },

    /// P-bit integration error
    #[error("Pbit error: {message}")]
    PbitError {
        /// Error message
        message: String,
    },

    /// Quantum integration error
    #[error("Quantum error: {message}")]
    QuantumError {
        /// Error message
        message: String,
    },

    /// Syntergic integration error
    #[error("Syntergic error: {message}")]
    SyntergicError {
        /// Error message
        message: String,
    },

    /// Market data error
    #[error("Market data error: {message}")]
    MarketDataError {
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message
        message: String,
    },

    /// Signal generation error
    #[error("Signal generation error: {message}")]
    SignalError {
        /// Error message
        message: String,
    },

    /// Insufficient data for analysis
    #[error("Insufficient data: need {required} samples, have {available}")]
    InsufficientData {
        /// Required samples
        required: usize,
        /// Available samples
        available: usize,
    },

    /// Numerical computation error
    #[error("Numerical error in {operation}: {message}")]
    NumericalError {
        /// Operation that failed
        operation: String,
        /// Error details
        message: String,
    },

    /// External dependency error
    #[error("External error: {0}")]
    External(#[from] anyhow::Error),
}

/// Result type for bridge operations
pub type Result<T> = std::result::Result<T, BridgeError>;

impl From<hyperphysics_autopoiesis::AutopoiesisError> for BridgeError {
    fn from(err: hyperphysics_autopoiesis::AutopoiesisError) -> Self {
        BridgeError::AutopoiesisError {
            message: err.to_string(),
        }
    }
}

impl From<hyperphysics_consciousness::ConsciousnessError> for BridgeError {
    fn from(err: hyperphysics_consciousness::ConsciousnessError) -> Self {
        BridgeError::ConsciousnessError {
            message: err.to_string(),
        }
    }
}

impl From<hyperphysics_thermo::ThermoError> for BridgeError {
    fn from(err: hyperphysics_thermo::ThermoError) -> Self {
        BridgeError::ThermoError {
            message: err.to_string(),
        }
    }
}

impl From<hyperphysics_syntergic::SyntergicError> for BridgeError {
    fn from(err: hyperphysics_syntergic::SyntergicError) -> Self {
        BridgeError::SyntergicError {
            message: err.to_string(),
        }
    }
}
