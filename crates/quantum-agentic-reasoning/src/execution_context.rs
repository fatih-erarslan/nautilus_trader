//! Execution context for quantum agentic reasoning
//!
//! This module provides the execution environment and context management
//! for QAR operations, including state tracking and resource coordination.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use crate::config::QarConfig;
use crate::error::QarError;
use crate::hardware::{HardwareManager, ResourceHandle};
use crate::memory::{MemoryManager, DecisionMemory};

/// Main execution context for QAR operations
#[derive(Debug)]
pub struct ExecutionContext {
    config: QarConfig,
    hardware_manager: Arc<Mutex<HardwareManager>>,
    memory_manager: Arc<Mutex<MemoryManager>>,
    execution_state: Arc<Mutex<ExecutionState>>,
    active_resources: Arc<Mutex<Vec<ResourceHandle>>>,
    performance_tracker: Arc<Mutex<PerformanceTracker>>,
}

/// Current execution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionState {
    pub session_id: String,
    pub start_time: u64,
    pub operations_count: u64,
    pub quantum_operations: u64,
    pub classical_operations: u64,
    pub average_latency_ns: u64,
    pub error_count: u64,
    pub current_phase: ExecutionPhase,
    pub resource_utilization: HashMap<String, f64>,
}

/// Execution phases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExecutionPhase {
    Initialization,
    MarketAnalysis,
    PatternRecognition,
    QuantumDecision,
    ClassicalFallback,
    ResultSynthesis,
    MemoryUpdate,
    Cleanup,
}

/// Performance tracking for the execution context
#[derive(Debug)]
struct PerformanceTracker {
    operation_times: HashMap<String, Vec<Duration>>,
    error_log: Vec<ExecutionError>,
    quantum_fidelity_scores: Vec<f64>,
    memory_usage_history: Vec<(Instant, f64)>,
    latency_violations: u64,
}

/// Execution error tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionError {
    pub timestamp: u64,
    pub operation: String,
    pub error_type: String,
    pub error_message: String,
    pub recovery_action: Option<String>,
}

impl ExecutionContext {
    /// Create new execution context
    pub fn new(config: &QarConfig) -> Result<Self, QarError> {
        let hardware_manager = Arc::new(Mutex::new(
            HardwareManager::new(config.hardware.clone())?
        ));
        
        let memory_manager = Arc::new(Mutex::new(
            MemoryManager::new(config.memory.clone())?
        ));
        
        let execution_state = Arc::new(Mutex::new(ExecutionState {
            session_id: Self::generate_session_id(),
            start_time: Self::current_timestamp(),
            operations_count: 0,
            quantum_operations: 0,
            classical_operations: 0,
            average_latency_ns: 0,
            error_count: 0,
            current_phase: ExecutionPhase::Initialization,
            resource_utilization: HashMap::new(),
        }));
        
        let performance_tracker = Arc::new(Mutex::new(PerformanceTracker {
            operation_times: HashMap::new(),
            error_log: Vec::new(),
            quantum_fidelity_scores: Vec::new(),
            memory_usage_history: Vec::new(),
            latency_violations: 0,
        }));
        
        Ok(Self {
            config: config.clone(),
            hardware_manager,
            memory_manager,
            execution_state,
            active_resources: Arc::new(Mutex::new(Vec::new())),
            performance_tracker,
        })
    }
    
    /// Begin a new execution phase
    pub fn begin_phase(&self, phase: ExecutionPhase) -> Result<PhaseGuard, QarError> {
        let start_time = Instant::now();
        
        // Update execution state
        {
            let mut state = self.execution_state.lock()
                .map_err(|_| QarError::ContextError("Failed to lock execution state".to_string()))?;
            state.current_phase = phase.clone();
        }
        
        // Allocate resources if needed
        let resource_handle = self.allocate_phase_resources(&phase)?;
        
        Ok(PhaseGuard {
            context: self,
            phase,
            start_time,
            resource_handle,
        })
    }
    
    /// Record operation performance
    pub fn record_operation(&self, operation: &str, duration: Duration, success: bool) -> Result<(), QarError> {
        // Update execution state
        {
            let mut state = self.execution_state.lock()
                .map_err(|_| QarError::ContextError("Failed to lock execution state".to_string()))?;
            
            state.operations_count += 1;
            
            // Update average latency
            let duration_ns = duration.as_nanos() as u64;
            state.average_latency_ns = 
                (state.average_latency_ns * (state.operations_count - 1) + duration_ns) / state.operations_count;
            
            if !success {
                state.error_count += 1;
            }
        }
        
        // Update performance tracker
        {
            let mut tracker = self.performance_tracker.lock()
                .map_err(|_| QarError::ContextError("Failed to lock performance tracker".to_string()))?;
            
            tracker.operation_times
                .entry(operation.to_string())
                .or_insert_with(Vec::new)
                .push(duration);
            
            // Check for latency violations
            if duration.as_nanos() as u64 > 1_000_000 { // 1ms threshold
                tracker.latency_violations += 1;
            }
        }
        
        // Update memory manager
        if let Ok(memory_manager) = self.memory_manager.lock() {
            let _ = memory_manager.update_performance(operation, duration.as_nanos() as u64, success);
        }
        
        Ok(())
    }
    
    /// Record quantum operation
    pub fn record_quantum_operation(&self, fidelity: Option<f64>) -> Result<(), QarError> {
        let mut state = self.execution_state.lock()
            .map_err(|_| QarError::ContextError("Failed to lock execution state".to_string()))?;
        
        state.quantum_operations += 1;
        
        if let Some(fidelity_score) = fidelity {
            if let Ok(mut tracker) = self.performance_tracker.lock() {
                tracker.quantum_fidelity_scores.push(fidelity_score);
            }
        }
        
        Ok(())
    }
    
    /// Record classical operation
    pub fn record_classical_operation(&self) -> Result<(), QarError> {
        let mut state = self.execution_state.lock()
            .map_err(|_| QarError::ContextError("Failed to lock execution state".to_string()))?;
        
        state.classical_operations += 1;
        Ok(())
    }
    
    /// Record execution error
    pub fn record_error(&self, operation: &str, error: &QarError, recovery_action: Option<String>) -> Result<(), QarError> {
        let execution_error = ExecutionError {
            timestamp: Self::current_timestamp(),
            operation: operation.to_string(),
            error_type: std::any::type_name::<QarError>().to_string(),
            error_message: format!("{}", error),
            recovery_action,
        };
        
        if let Ok(mut tracker) = self.performance_tracker.lock() {
            tracker.error_log.push(execution_error);
        }
        
        Ok(())
    }
    
    /// Store decision in memory
    pub fn store_decision(&self, decision: DecisionMemory) -> Result<(), QarError> {
        let memory_manager = self.memory_manager.lock()
            .map_err(|_| QarError::ContextError("Failed to lock memory manager".to_string()))?;
        
        memory_manager.store_decision(decision)
    }
    
    /// Get current execution statistics
    pub fn get_execution_stats(&self) -> Result<ExecutionStats, QarError> {
        let state = self.execution_state.lock()
            .map_err(|_| QarError::ContextError("Failed to lock execution state".to_string()))?;
        
        let tracker = self.performance_tracker.lock()
            .map_err(|_| QarError::ContextError("Failed to lock performance tracker".to_string()))?;
        
        let average_quantum_fidelity = if tracker.quantum_fidelity_scores.is_empty() {
            None
        } else {
            Some(tracker.quantum_fidelity_scores.iter().sum::<f64>() / tracker.quantum_fidelity_scores.len() as f64)
        };
        
        Ok(ExecutionStats {
            session_id: state.session_id.clone(),
            uptime_ms: Self::current_timestamp() - state.start_time,
            total_operations: state.operations_count,
            quantum_operations: state.quantum_operations,
            classical_operations: state.classical_operations,
            average_latency_ns: state.average_latency_ns,
            error_count: state.error_count,
            error_rate: state.error_count as f64 / state.operations_count.max(1) as f64,
            quantum_classical_ratio: state.quantum_operations as f64 / state.classical_operations.max(1) as f64,
            average_quantum_fidelity,
            latency_violations: tracker.latency_violations,
            current_phase: state.current_phase.clone(),
        })
    }
    
    /// Get hardware manager reference
    pub fn get_hardware_manager(&self) -> Arc<Mutex<HardwareManager>> {
        Arc::clone(&self.hardware_manager)
    }
    
    /// Get memory manager reference
    pub fn get_memory_manager(&self) -> Arc<Mutex<MemoryManager>> {
        Arc::clone(&self.memory_manager)
    }
    
    /// Check if quantum computing is available
    pub fn is_quantum_available(&self) -> bool {
        if let Ok(hardware) = self.hardware_manager.lock() {
            hardware.is_quantum_available()
        } else {
            false
        }
    }
    
    /// Cleanup resources and finalize execution
    pub fn cleanup(&self) -> Result<(), QarError> {
        // Perform memory cleanup
        if let Ok(memory_manager) = self.memory_manager.lock() {
            let _ = memory_manager.perform_cleanup();
        }
        
        // Clear active resources
        if let Ok(mut resources) = self.active_resources.lock() {
            resources.clear();
        }
        
        Ok(())
    }
    
    // Helper methods
    fn generate_session_id() -> String {
        format!("qar_session_{}", Self::current_timestamp())
    }
    
    fn current_timestamp() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(Duration::from_secs(0))
            .as_millis() as u64
    }
    
    fn allocate_phase_resources(&self, phase: &ExecutionPhase) -> Result<Option<ResourceHandle>, QarError> {
        let (cpu_fraction, memory_mb) = match phase {
            ExecutionPhase::Initialization => (0.1, 100.0),
            ExecutionPhase::MarketAnalysis => (0.3, 200.0),
            ExecutionPhase::PatternRecognition => (0.4, 300.0),
            ExecutionPhase::QuantumDecision => (0.6, 500.0),
            ExecutionPhase::ClassicalFallback => (0.5, 400.0),
            ExecutionPhase::ResultSynthesis => (0.2, 150.0),
            ExecutionPhase::MemoryUpdate => (0.1, 100.0),
            ExecutionPhase::Cleanup => (0.05, 50.0),
        };
        
        if let Ok(hardware) = self.hardware_manager.lock() {
            match hardware.allocate_resources(cpu_fraction, memory_mb) {
                Ok(handle) => {
                    if let Ok(mut resources) = self.active_resources.lock() {
                        resources.push(handle);
                    }
                    Ok(None) // Handle is stored in active_resources
                },
                Err(e) => {
                    // Non-fatal error, continue without dedicated resources
                    eprintln!("Warning: Failed to allocate resources for phase {:?}: {}", phase, e);
                    Ok(None)
                }
            }
        } else {
            Ok(None)
        }
    }
}

/// RAII guard for execution phases
pub struct PhaseGuard<'a> {
    context: &'a ExecutionContext,
    phase: ExecutionPhase,
    start_time: Instant,
    resource_handle: Option<ResourceHandle>,
}

impl<'a> Drop for PhaseGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start_time.elapsed();
        let operation_name = format!("phase_{:?}", self.phase);
        
        // Record phase completion
        let _ = self.context.record_operation(&operation_name, duration, true);
    }
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    pub session_id: String,
    pub uptime_ms: u64,
    pub total_operations: u64,
    pub quantum_operations: u64,
    pub classical_operations: u64,
    pub average_latency_ns: u64,
    pub error_count: u64,
    pub error_rate: f64,
    pub quantum_classical_ratio: f64,
    pub average_quantum_fidelity: Option<f64>,
    pub latency_violations: u64,
    pub current_phase: ExecutionPhase,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QarConfig;
    
    #[test]
    fn test_execution_context_creation() {
        let config = QarConfig::default();
        let context = ExecutionContext::new(&config);
        assert!(context.is_ok());
    }
    
    #[test]
    fn test_phase_execution() {
        let config = QarConfig::default();
        let context = ExecutionContext::new(&config).unwrap();
        
        let phase_guard = context.begin_phase(ExecutionPhase::MarketAnalysis);
        assert!(phase_guard.is_ok());
        
        // Phase guard automatically ends when dropped
        drop(phase_guard);
        
        let stats = context.get_execution_stats().unwrap();
        assert!(stats.total_operations > 0);
    }
    
    #[test]
    fn test_operation_recording() {
        let config = QarConfig::default();
        let context = ExecutionContext::new(&config).unwrap();
        
        let duration = Duration::from_micros(500);
        assert!(context.record_operation("test_op", duration, true).is_ok());
        
        let stats = context.get_execution_stats().unwrap();
        assert_eq!(stats.total_operations, 1);
        assert!(stats.average_latency_ns > 0);
    }
    
    #[test]
    fn test_quantum_classical_tracking() {
        let config = QarConfig::default();
        let context = ExecutionContext::new(&config).unwrap();
        
        assert!(context.record_quantum_operation(Some(0.95)).is_ok());
        assert!(context.record_classical_operation().is_ok());
        
        let stats = context.get_execution_stats().unwrap();
        assert_eq!(stats.quantum_operations, 1);
        assert_eq!(stats.classical_operations, 1);
        assert_eq!(stats.quantum_classical_ratio, 1.0);
    }
}