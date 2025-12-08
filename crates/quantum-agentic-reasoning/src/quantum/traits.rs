//! Quantum traits and interfaces
//!
//! This module defines common traits and interfaces for quantum operations.

use crate::core::QarResult;
use async_trait::async_trait;
use std::collections::HashMap;
use super::types::*;

/// Trait for quantum circuit execution
#[async_trait]
pub trait QuantumExecutor {
    /// Execute a quantum circuit
    async fn execute(&self, circuit: &str, shots: u64) -> QarResult<ExecutionStats>;
    
    /// Get executor capabilities
    fn get_capabilities(&self) -> Vec<String>;
    
    /// Check if the executor is available
    async fn is_available(&self) -> bool;
}

/// Trait for quantum pattern recognition
#[async_trait]
pub trait PatternRecognizer {
    /// Recognize patterns in quantum data
    async fn recognize_pattern(&self, data: &[f64]) -> QarResult<PatternResult>;
    
    /// Get supported pattern types
    fn supported_patterns(&self) -> Vec<String>;
}

/// Trait for quantum optimization
#[async_trait]
pub trait QuantumOptimizer {
    /// Optimize a quantum circuit
    async fn optimize(&self, circuit: &str) -> QarResult<CompilationResult>;
    
    /// Get optimization strategies
    fn get_strategies(&self) -> Vec<String>;
}

/// Trait for quantum backend management
#[async_trait]
pub trait QuantumBackend {
    /// Connect to the backend
    async fn connect(&mut self) -> QarResult<()>;
    
    /// Disconnect from the backend
    async fn disconnect(&mut self) -> QarResult<()>;
    
    /// Check backend status
    async fn status(&self) -> QarResult<String>;
    
    /// Get backend information
    fn info(&self) -> HashMap<String, String>;
}