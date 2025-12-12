//! Cognitive runtime execution engine
//!
//! Manages the cognitive loop execution across all 8 layers with:
//! - Perception → Inference → Action loop
//! - Resource scheduling and energy management
//! - Real-time homeostatic regulation
//! - Thread-safe state management

use crate::config::QksConfig;
use crate::error::{QksError, QksResult};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cognitive runtime state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    /// Runtime not initialized
    Uninitialized,

    /// Runtime initialized but not running
    Initialized,

    /// Runtime actively processing
    Running,

    /// Runtime paused
    Paused,

    /// Runtime stopped
    Stopped,

    /// Runtime in error state
    Error,
}

/// Cognitive runtime for executing the 8-layer architecture
pub struct CognitiveRuntime {
    /// Runtime configuration
    config: Arc<QksConfig>,

    /// Current runtime state
    state: Arc<RwLock<RuntimeState>>,

    /// Core integration API (from quantum-knowledge-core)
    integration_api: Option<Arc<quantum_knowledge_core::integration::CognitiveSystemAPI>>,

    /// Runtime metrics
    metrics: Arc<RwLock<RuntimeMetrics>>,

    /// Start time
    start_time: Option<Instant>,
}

/// Runtime performance metrics
#[derive(Debug, Clone)]
pub struct RuntimeMetrics {
    /// Total iterations executed
    pub total_iterations: u64,

    /// Total time elapsed (microseconds)
    pub total_time_us: u64,

    /// Average iteration time (microseconds)
    pub avg_iteration_us: f64,

    /// Current Phi value
    pub current_phi: f64,

    /// Current energy level
    pub current_energy: f64,

    /// Homeostasis stability (0.0 to 1.0)
    pub homeostasis_stability: f64,

    /// Active layers count
    pub active_layers: usize,
}

impl Default for RuntimeMetrics {
    fn default() -> Self {
        Self {
            total_iterations: 0,
            total_time_us: 0,
            avg_iteration_us: 0.0,
            current_phi: 0.0,
            current_energy: 0.0,
            homeostasis_stability: 0.0,
            active_layers: 0,
        }
    }
}

impl CognitiveRuntime {
    /// Create new cognitive runtime with configuration
    pub fn new(config: QksConfig) -> Self {
        Self {
            config: Arc::new(config),
            state: Arc::new(RwLock::new(RuntimeState::Uninitialized)),
            integration_api: None,
            metrics: Arc::new(RwLock::new(RuntimeMetrics::default())),
            start_time: None,
        }
    }

    /// Initialize the runtime (creates cognitive system)
    pub fn initialize(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        if *state != RuntimeState::Uninitialized {
            return Err(QksError::Generic("Runtime already initialized".to_string()));
        }

        // Create cognitive system with configuration
        let system_config = quantum_knowledge_core::integration::SystemConfig::default();
        let api = quantum_knowledge_core::integration::CognitiveSystemAPI::new(system_config);

        self.integration_api = Some(Arc::new(api));
        *state = RuntimeState::Initialized;

        Ok(())
    }

    /// Start the runtime
    pub fn start(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        match *state {
            RuntimeState::Initialized | RuntimeState::Paused => {
                *state = RuntimeState::Running;
                self.start_time = Some(Instant::now());
                Ok(())
            }
            RuntimeState::Running => Err(QksError::Generic("Runtime already running".to_string())),
            RuntimeState::Uninitialized => Err(QksError::Generic("Runtime not initialized".to_string())),
            RuntimeState::Stopped => Err(QksError::Generic("Runtime stopped, must reinitialize".to_string())),
            RuntimeState::Error => Err(QksError::Generic("Runtime in error state".to_string())),
        }
    }

    /// Pause the runtime
    pub fn pause(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        if *state == RuntimeState::Running {
            *state = RuntimeState::Paused;
            Ok(())
        } else {
            Err(QksError::Generic("Runtime not running".to_string()))
        }
    }

    /// Stop the runtime
    pub fn stop(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        *state = RuntimeState::Stopped;
        Ok(())
    }

    /// Execute one iteration of the cognitive loop
    pub fn iterate(&mut self) -> QksResult<IterationResult> {
        let state = *self.state.read();
        if state != RuntimeState::Running {
            return Err(QksError::Generic("Runtime not running".to_string()));
        }

        let api = self.integration_api.as_ref()
            .ok_or_else(|| QksError::Generic("Integration API not initialized".to_string()))?;

        let start = Instant::now();

        // Execute one cognitive cycle
        let result = api.iterate()
            .map_err(|e| QksError::Integration(e.to_string()))?;

        let elapsed = start.elapsed();

        // Update metrics
        let mut metrics = self.metrics.write();
        metrics.total_iterations += 1;
        metrics.total_time_us += elapsed.as_micros() as u64;
        metrics.avg_iteration_us = metrics.total_time_us as f64 / metrics.total_iterations as f64;

        // Get current system state
        if let Ok(cognitive_state) = api.get_cognitive_state() {
            // Phi is stored in metadata, not as a direct field
            metrics.current_phi = cognitive_state.metadata.get("phi").copied().unwrap_or(0.0);
            metrics.current_energy = cognitive_state.energy_level;
        }

        if let Ok(homeo_status) = api.get_homeostatic_status() {
            metrics.homeostasis_stability = homeo_status.stability;
        }

        if let Ok(status) = api.get_system_status() {
            metrics.active_layers = status.orchestrator_metrics.active_layers;
        }

        Ok(IterationResult {
            iteration_number: metrics.total_iterations,
            duration: elapsed,
            phi: metrics.current_phi,
            energy: metrics.current_energy,
            stability: metrics.homeostasis_stability,
        })
    }

    /// Run for a specified number of iterations
    pub fn run(&mut self, iterations: usize) -> QksResult<Vec<IterationResult>> {
        let mut results = Vec::with_capacity(iterations);

        for _ in 0..iterations {
            let result = self.iterate()?;
            results.push(result);
        }

        Ok(results)
    }

    /// Get current runtime state
    pub fn get_state(&self) -> RuntimeState {
        *self.state.read()
    }

    /// Get runtime metrics
    pub fn get_metrics(&self) -> RuntimeMetrics {
        self.metrics.read().clone()
    }

    /// Get uptime duration
    pub fn uptime(&self) -> Option<Duration> {
        self.start_time.map(|start| start.elapsed())
    }

    /// Reset runtime (clear state, keep config)
    pub fn reset(&mut self) -> QksResult<()> {
        self.stop()?;

        // Clear integration API
        self.integration_api = None;

        // Reset metrics
        *self.metrics.write() = RuntimeMetrics::default();

        // Reset state
        *self.state.write() = RuntimeState::Uninitialized;

        self.start_time = None;

        Ok(())
    }
}

/// Result from a single iteration
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Iteration number
    pub iteration_number: u64,

    /// Time taken for this iteration
    pub duration: Duration,

    /// Current Phi value
    pub phi: f64,

    /// Current energy level
    pub energy: f64,

    /// Homeostasis stability
    pub stability: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QksConfig;

    #[test]
    fn test_runtime_creation() {
        let config = QksConfig::default();
        let runtime = CognitiveRuntime::new(config);

        assert_eq!(runtime.get_state(), RuntimeState::Uninitialized);
    }

    #[test]
    fn test_runtime_lifecycle() {
        let config = QksConfig::default();
        let mut runtime = CognitiveRuntime::new(config);

        // Initialize
        assert!(runtime.initialize().is_ok());
        assert_eq!(runtime.get_state(), RuntimeState::Initialized);

        // Start
        assert!(runtime.start().is_ok());
        assert_eq!(runtime.get_state(), RuntimeState::Running);

        // Pause
        assert!(runtime.pause().is_ok());
        assert_eq!(runtime.get_state(), RuntimeState::Paused);

        // Resume
        assert!(runtime.start().is_ok());
        assert_eq!(runtime.get_state(), RuntimeState::Running);

        // Stop
        assert!(runtime.stop().is_ok());
        assert_eq!(runtime.get_state(), RuntimeState::Stopped);
    }

    #[test]
    fn test_runtime_iteration() {
        let config = QksConfig::default();
        let mut runtime = CognitiveRuntime::new(config);

        runtime.initialize().unwrap();
        runtime.start().unwrap();

        // Run a few iterations
        let result = runtime.iterate();
        assert!(result.is_ok());

        let metrics = runtime.get_metrics();
        assert_eq!(metrics.total_iterations, 1);
    }
}
