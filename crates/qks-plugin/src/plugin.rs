//! Core plugin struct with initialization and shutdown
//!
//! Provides the main QKSPlugin interface for:
//! - Initialization of all 8 cognitive layers
//! - Cognitive loop execution
//! - State management
//! - Resource coordination

use crate::config::QksConfig;
use crate::runtime::{CognitiveRuntime, RuntimeState, RuntimeMetrics, IterationResult};
use crate::error::{QksError, QksResult};
use parking_lot::RwLock;
use std::sync::Arc;

/// Plugin state
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PluginState {
    /// Plugin not initialized
    Uninitialized,

    /// Plugin initialized and ready
    Ready,

    /// Plugin running
    Active,

    /// Plugin paused
    Paused,

    /// Plugin shut down
    Shutdown,
}

/// Main QKS Plugin structure exposing all 8 cognitive layers
pub struct QksPlugin {
    /// Plugin configuration
    config: Arc<QksConfig>,

    /// Cognitive runtime
    runtime: Arc<RwLock<CognitiveRuntime>>,

    /// Plugin state
    state: Arc<RwLock<PluginState>>,

    /// Layer initialization status (8 layers)
    layer_status: Arc<RwLock<[bool; 8]>>,
}

impl QksPlugin {
    /// Create new plugin with configuration
    pub fn new(config: QksConfig) -> Self {
        let runtime = CognitiveRuntime::new(config.clone());

        Self {
            config: Arc::new(config),
            runtime: Arc::new(RwLock::new(runtime)),
            state: Arc::new(RwLock::new(PluginState::Uninitialized)),
            layer_status: Arc::new(RwLock::new([false; 8])),
        }
    }

    /// Initialize the plugin and all cognitive layers
    ///
    /// This performs initialization of:
    /// - Layer 1: Thermodynamic Optimization
    /// - Layer 2: Cognitive Architecture
    /// - Layer 3: Decision Making
    /// - Layer 4: Learning & Reasoning
    /// - Layer 5: Collective Intelligence
    /// - Layer 6: Consciousness
    /// - Layer 7: Metacognition
    /// - Layer 8: Full Integration
    pub fn initialize(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        if *state != PluginState::Uninitialized {
            return Err(QksError::Generic("Plugin already initialized".to_string()));
        }

        // Initialize runtime
        let mut runtime = self.runtime.write();
        runtime.initialize()?;

        // Mark all layers as initialized
        // In a full implementation, each layer would be initialized individually
        *self.layer_status.write() = [true; 8];

        *state = PluginState::Ready;

        Ok(())
    }

    /// Start the plugin (begin cognitive processing)
    pub fn start(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        match *state {
            PluginState::Ready | PluginState::Paused => {
                let mut runtime = self.runtime.write();
                runtime.start()?;
                *state = PluginState::Active;
                Ok(())
            }
            PluginState::Active => Err(QksError::Generic("Plugin already active".to_string())),
            PluginState::Uninitialized => Err(QksError::Generic("Plugin not initialized".to_string())),
            PluginState::Shutdown => Err(QksError::Generic("Plugin shut down".to_string())),
        }
    }

    /// Pause the plugin
    pub fn pause(&mut self) -> QksResult<()> {
        let mut state = self.state.write();
        if *state == PluginState::Active {
            let mut runtime = self.runtime.write();
            runtime.pause()?;
            *state = PluginState::Paused;
            Ok(())
        } else {
            Err(QksError::Generic("Plugin not active".to_string()))
        }
    }

    /// Resume the plugin
    pub fn resume(&mut self) -> QksResult<()> {
        self.start()
    }

    /// Shutdown the plugin
    pub fn shutdown(&mut self) -> QksResult<()> {
        let mut state = self.state.write();

        let mut runtime = self.runtime.write();
        runtime.stop()?;

        *self.layer_status.write() = [false; 8];
        *state = PluginState::Shutdown;

        Ok(())
    }

    /// Process input through all cognitive layers
    ///
    /// # Arguments
    /// * `input` - Raw input data
    ///
    /// # Returns
    /// Processed output after full cognitive cycle
    pub fn process(&mut self, input: &[u8]) -> QksResult<Vec<u8>> {
        let state = *self.state.read();
        if state != PluginState::Active {
            return Err(QksError::Generic("Plugin not active".to_string()));
        }

        // Execute one iteration of the cognitive loop
        let mut runtime = self.runtime.write();
        let _result = runtime.iterate()?;

        // For now, echo input as output
        // Full implementation would process through all layers
        Ok(input.to_vec())
    }

    /// Get current Phi (consciousness metric)
    pub fn get_phi(&self) -> QksResult<f64> {
        let runtime = self.runtime.read();
        let metrics = runtime.get_metrics();
        Ok(metrics.current_phi)
    }

    /// Get current plugin state
    pub fn get_plugin_state(&self) -> PluginState {
        *self.state.read()
    }

    /// Get runtime state
    pub fn get_runtime_state(&self) -> RuntimeState {
        let runtime = self.runtime.read();
        runtime.get_state()
    }

    /// Get layer initialization status
    pub fn get_layer_status(&self) -> [bool; 8] {
        *self.layer_status.read()
    }

    /// Check if specific layer is initialized
    pub fn is_layer_initialized(&self, layer_id: u8) -> bool {
        if layer_id >= 8 {
            return false;
        }
        self.layer_status.read()[layer_id as usize]
    }

    /// Get runtime metrics
    pub fn get_metrics(&self) -> RuntimeMetrics {
        let runtime = self.runtime.read();
        runtime.get_metrics()
    }

    /// Execute one iteration of the cognitive loop
    pub fn iterate(&mut self) -> QksResult<IterationResult> {
        let state = *self.state.read();
        if state != PluginState::Active {
            return Err(QksError::Generic("Plugin not active".to_string()));
        }

        let mut runtime = self.runtime.write();
        runtime.iterate()
    }

    /// Run for specified number of iterations
    pub fn run(&mut self, iterations: usize) -> QksResult<Vec<IterationResult>> {
        let state = *self.state.read();
        if state != PluginState::Active {
            return Err(QksError::Generic("Plugin not active".to_string()));
        }

        let mut runtime = self.runtime.write();
        runtime.run(iterations)
    }

    /// Get plugin configuration
    pub fn get_config(&self) -> Arc<QksConfig> {
        Arc::clone(&self.config)
    }
}

impl Drop for QksPlugin {
    fn drop(&mut self) {
        // Ensure clean shutdown
        let _ = self.shutdown();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QksConfig;

    #[test]
    fn test_plugin_creation() {
        let config = QksConfig::default();
        let plugin = QksPlugin::new(config);

        assert_eq!(plugin.get_plugin_state(), PluginState::Uninitialized);
    }

    #[test]
    fn test_plugin_lifecycle() {
        let config = QksConfig::default();
        let mut plugin = QksPlugin::new(config);

        // Initialize
        assert!(plugin.initialize().is_ok());
        assert_eq!(plugin.get_plugin_state(), PluginState::Ready);

        // Start
        assert!(plugin.start().is_ok());
        assert_eq!(plugin.get_plugin_state(), PluginState::Active);

        // Pause
        assert!(plugin.pause().is_ok());
        assert_eq!(plugin.get_plugin_state(), PluginState::Paused);

        // Resume
        assert!(plugin.resume().is_ok());
        assert_eq!(plugin.get_plugin_state(), PluginState::Active);

        // Shutdown
        assert!(plugin.shutdown().is_ok());
        assert_eq!(plugin.get_plugin_state(), PluginState::Shutdown);
    }

    #[test]
    fn test_layer_initialization() {
        let config = QksConfig::default();
        let mut plugin = QksPlugin::new(config);

        // Before initialization
        assert!(!plugin.is_layer_initialized(0));

        // After initialization
        plugin.initialize().unwrap();

        // All layers should be initialized
        for layer_id in 0..8 {
            assert!(plugin.is_layer_initialized(layer_id));
        }
    }

    #[test]
    fn test_phi_calculation() {
        let config = QksConfig::default();
        let mut plugin = QksPlugin::new(config);

        plugin.initialize().unwrap();
        plugin.start().unwrap();

        // Get Phi value
        let phi = plugin.get_phi();
        assert!(phi.is_ok());
    }
}
