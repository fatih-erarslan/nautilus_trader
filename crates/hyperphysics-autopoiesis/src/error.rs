//! Error types for the hyperphysics-autopoiesis bridge

use thiserror::Error;

/// Errors that can occur in the autopoiesis bridge
#[derive(Error, Debug)]
pub enum AutopoiesisError {
    /// Autopoietic system health below threshold
    #[error("Autopoietic health degraded: {health:.4} < threshold {threshold:.4}")]
    HealthDegraded {
        /// Current health value
        health: f64,
        /// Required threshold
        threshold: f64,
    },

    /// Operational closure violated
    #[error("Operational closure violated: not all consumed components are produced internally")]
    OperationalClosureViolation,

    /// Dissipative structure instability
    #[error("Dissipative structure unstable: entropy production {entropy:.4} exceeds safe limit")]
    DissipativeInstability {
        /// Current entropy production rate
        entropy: f64,
    },

    /// Bifurcation transition failure
    #[error("Bifurcation transition failed at control parameter {parameter:.4}: {reason}")]
    BifurcationFailure {
        /// Control parameter value
        parameter: f64,
        /// Failure reason
        reason: String,
    },

    /// Syntergic coherence loss
    #[error("Syntergic coherence lost: {coherence:.4} below unity threshold")]
    SyntergicCoherenceLoss {
        /// Current coherence value
        coherence: f64,
    },

    /// Synchronization failure in coupled oscillators
    #[error("Synchronization failure: Kuramoto order parameter {order:.4} indicates desynchronization")]
    SynchronizationFailure {
        /// Kuramoto order parameter
        order: f64,
    },

    /// Network topology error
    #[error("Network topology error: {message}")]
    NetworkTopologyError {
        /// Error message
        message: String,
    },

    /// Thermodynamic constraint violation
    #[error("Thermodynamic constraint violated: {constraint}")]
    ThermodynamicViolation {
        /// Constraint description
        constraint: String,
    },

    /// Consciousness integration error
    #[error("Consciousness integration error: {message}")]
    ConsciousnessError {
        /// Error message
        message: String,
    },

    /// Trading system error
    #[error("Trading system error: {message}")]
    TradingError {
        /// Error message
        message: String,
    },

    /// Configuration error
    #[error("Configuration error: {message}")]
    ConfigError {
        /// Error message
        message: String,
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

/// Result type for autopoiesis operations
pub type Result<T> = std::result::Result<T, AutopoiesisError>;

impl From<hyperphysics_thermo::ThermoError> for AutopoiesisError {
    fn from(err: hyperphysics_thermo::ThermoError) -> Self {
        AutopoiesisError::ThermodynamicViolation {
            constraint: err.to_string(),
        }
    }
}

impl From<hyperphysics_consciousness::ConsciousnessError> for AutopoiesisError {
    fn from(err: hyperphysics_consciousness::ConsciousnessError) -> Self {
        AutopoiesisError::ConsciousnessError {
            message: err.to_string(),
        }
    }
}

impl From<hyperphysics_syntergic::SyntergicError> for AutopoiesisError {
    fn from(_err: hyperphysics_syntergic::SyntergicError) -> Self {
        AutopoiesisError::SyntergicCoherenceLoss {
            coherence: 0.0, // Default when converting from field errors
        }
    }
}
