//! Symbolic Decision Logger - Verbose Agency Decision Tracking
//!
//! Provides detailed logging of cognitive decision-making with:
//! - Algorithmic step-by-step execution
//! - Symbolic path language (inference trees)
//! - Wolfram symbolic validation
//! - Performance metrics

use crate::error::Result;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::{debug, info};

/// Decision phase in the cognitive cycle
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DecisionPhase {
    Perception,
    Cognition,
    Deliberation,
    Intention,
    Integration,
    Action,
}

impl DecisionPhase {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Perception => "Perception",
            Self::Cognition => "Cognition",
            Self::Deliberation => "Deliberation",
            Self::Intention => "Intention",
            Self::Integration => "Integration",
            Self::Action => "Action",
        }
    }
}

/// Computation step with symbolic representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationStep {
    /// Human-readable description
    pub description: String,
    
    /// Mathematical formula (LaTeX notation)
    pub formula: String,
    
    /// Input values
    pub inputs: Vec<(String, f64)>,
    
    /// Computed result
    pub result: f64,
    
    /// Execution time in microseconds
    pub duration_us: u64,
}

/// Symbolic inference path (tree structure)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolicPath {
    /// Root node (initial perception)
    pub root: String,
    
    /// Inference steps (parent → child relationships)
    pub edges: Vec<(String, String, String)>, // (from, to, reason)
    
    /// Terminal node (final action)
    pub terminal: String,
    
    /// Total path cost (free energy)
    pub cost: f64,
}

/// Wolfram validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WolframValidation {
    /// Formula to validate
    pub formula: String,
    
    /// Wolfram computation result
    pub wolfram_result: String,
    
    /// Rust computation result
    pub rust_result: f64,
    
    /// Validation passed
    pub valid: bool,
    
    /// Error tolerance
    pub tolerance: f64,
}

/// Symbolic Decision Logger
pub struct SymbolicDecisionLogger {
    /// Current decision ID
    decision_id: u64,
    
    /// Computation steps for current decision
    current_steps: Vec<ComputationStep>,
    
    /// Start time
    start_time: Instant,
}

impl SymbolicDecisionLogger {
    pub fn new() -> Self {
        Self {
            decision_id: 0,
            current_steps: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    /// Start a new decision cycle
    pub fn begin_decision(&mut self) {
        self.decision_id += 1;
        self.current_steps.clear();
        self.start_time = Instant::now();
        
        info!(
            decision_id = self.decision_id,
            "🧠 BEGIN DECISION CYCLE"
        );
    }
    
    /// Log a computational step
    pub fn log_step(
        &mut self,
        phase: DecisionPhase,
        description: &str,
        formula: &str,
        inputs: Vec<(&str, f64)>,
        result: f64,
    ) {
        let step_start = Instant::now();
        
        let step = ComputationStep {
            description: description.to_string(),
            formula: formula.to_string(),
            inputs: inputs.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
            result,
            duration_us: step_start.elapsed().as_micros() as u64,
        };
        
        self.current_steps.push(step.clone());
        
        // Verbose logging with symbolic representation
        info!(
            decision_id = self.decision_id,
            phase = phase.name(),
            step = description,
            formula = formula,
            result = result,
            "📐 COMPUTATION STEP"
        );
        
        debug!(
            decision_id = self.decision_id,
            inputs = ?inputs,
            duration_us = step.duration_us,
            "  └─ Inputs & Timing"
        );
    }
    
    /// Log symbolic inference path
    pub fn log_symbolic_path(&self, path: &SymbolicPath) {
        info!(
            decision_id = self.decision_id,
            root = path.root,
            terminal = path.terminal,
            cost = path.cost,
            "🌳 SYMBOLIC PATH"
        );
        
        for (from, to, reason) in &path.edges {
            debug!(
                decision_id = self.decision_id,
                "  {} → {} ({})",
                from, to, reason
            );
        }
    }
    
    /// Log Wolfram symbolic validation
    pub fn log_wolfram_validation(&self, validation: &WolframValidation) {
        let status = if validation.valid { "✅" } else { "❌" };
        
        info!(
            decision_id = self.decision_id,
            formula = validation.formula,
            wolfram = validation.wolfram_result,
            rust = validation.rust_result,
            valid = validation.valid,
            "{} WOLFRAM VALIDATION",
            status
        );
        
        if !validation.valid {
            let error = (validation.rust_result - 
                         validation.wolfram_result.parse::<f64>().unwrap_or(0.0)).abs();
            debug!(
                decision_id = self.decision_id,
                error = error,
                tolerance = validation.tolerance,
                "  └─ Validation failed: error exceeds tolerance"
            );
        }
    }
    
    /// Complete decision cycle and log summary
    pub fn end_decision(&self, final_action: &str, free_energy: f64) {
        let total_time = self.start_time.elapsed();
        
        info!(
            decision_id = self.decision_id,
            action = final_action,
            free_energy = free_energy,
            total_steps = self.current_steps.len(),
            total_time_ms = total_time.as_millis(),
            "✨ DECISION COMPLETE"
        );
        
        // Log performance summary
        let total_compute_us: u64 = self.current_steps.iter()
            .map(|s| s.duration_us)
            .sum();
            
        debug!(
            decision_id = self.decision_id,
            compute_us = total_compute_us,
            overhead_us = total_time.as_micros() as u64 - total_compute_us,
            "  └─ Performance: {:.2}μs compute, {:.2}μs overhead",
            total_compute_us,
            total_time.as_micros() as u64 - total_compute_us
        );
    }
    
    /// Get current decision summary
    pub fn summary(&self) -> String {
        format!(
            "Decision #{}: {} steps, {:.2}ms elapsed",
            self.decision_id,
            self.current_steps.len(),
            self.start_time.elapsed().as_secs_f64() * 1000.0
        )
    }
}

impl Default for SymbolicDecisionLogger {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_symbolic_logger() {
        let mut logger = SymbolicDecisionLogger::new();
        
        logger.begin_decision();
        assert_eq!(logger.decision_id, 1);
        
        logger.log_step(
            DecisionPhase::Perception,
            "Compute prediction error",
            "PE = bottom_up - top_down",
            vec![("bottom_up", 1.0), ("top_down", 0.8)],
            0.2,
        );
        
        assert_eq!(logger.current_steps.len(), 1);
        assert!(!logger.summary().is_empty());
    }
    
    #[test]
    fn test_symbolic_path() {
        let path = SymbolicPath {
            root: "Sensory Input".to_string(),
            edges: vec![
                ("Sensory Input".to_string(), "Prediction Error".to_string(), "PE > threshold".to_string()),
                ("Prediction Error".to_string(), "Update Beliefs".to_string(), "Bayesian inference".to_string()),
                ("Update Beliefs".to_string(), "Select Action".to_string(), "Policy optimization".to_string()),
            ],
            terminal: "Motor Command".to_string(),
            cost: 0.5,
        };
        
        assert_eq!(path.edges.len(), 3);
        assert_eq!(path.cost, 0.5);
    }
}
