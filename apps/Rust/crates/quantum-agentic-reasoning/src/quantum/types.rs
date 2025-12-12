//! Quantum types and data structures
//!
//! This module defines quantum-specific types and data structures used throughout
//! the quantum computing modules.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Quantum gate parameter type
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GateParam {
    /// Parameter name
    pub name: String,
    /// Parameter value
    pub value: f64,
}

/// Quantum measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementResult {
    /// Qubit index
    pub qubit: usize,
    /// Measurement outcome (0 or 1)
    pub outcome: u8,
    /// Measurement probability
    pub probability: f64,
}

/// Quantum circuit compilation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompilationResult {
    /// Compiled gate sequence
    pub gates: Vec<String>,
    /// Compilation metadata
    pub metadata: HashMap<String, String>,
    /// Success flag
    pub success: bool,
}

/// Quantum execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Number of shots
    pub shots: u64,
    /// Success rate
    pub success_rate: f64,
    /// Fidelity estimate
    pub fidelity: Option<f64>,
}

/// Quantum pattern recognition result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternResult {
    /// Pattern name
    pub pattern: String,
    /// Confidence score
    pub confidence: f64,
    /// Pattern parameters
    pub parameters: HashMap<String, f64>,
}