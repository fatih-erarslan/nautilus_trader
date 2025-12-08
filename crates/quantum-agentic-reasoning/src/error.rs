//! Error handling for QAR - MINIMAL IMPLEMENTATION

use serde::{Deserialize, Serialize};

/// Minimal error types for QAR operations
#[derive(Debug, Clone, Serialize, Deserialize, thiserror::Error)]
pub enum QARError {
    #[error("Prospect Theory error: {0}")]
    ProspectTheory(#[from] prospect_theory::ProspectTheoryError),
    #[error("Whale Defense error: {0}")]
    WhaleDefense(String),
    #[error("Decision Engine error: {0}")]
    DecisionEngine(String),
    #[error("Generic error: {message}")]
    Generic { message: String },
    #[error("General error: {0}")]
    General(String),
    #[error("Performance constraint violation: {message}")]
    Performance { message: String },
}

pub type QARResult<T> = std::result::Result<T, QARError>;