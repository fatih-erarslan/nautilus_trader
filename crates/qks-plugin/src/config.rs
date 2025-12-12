//! Configuration builder pattern for plugin customization
//!
//! Scientific configuration based on:
//! - IIT consciousness threshold (Φ ≥ 1.0)
//! - Homeostatic setpoints (Cannon, 1932)
//! - Layer weights from cognitive neuroscience
//! - Meta-learning hyperparameters (Finn et al., 2017)

use serde::{Serialize, Deserialize};

/// Main plugin configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QksConfig {
    /// Consciousness threshold (Φ value)
    /// Default: 1.0 (based on IIT 3.0)
    pub phi_threshold: f64,

    /// Homeostatic setpoints for each variable
    pub homeostatic_setpoints: HomeostasisConfig,

    /// Weight for each cognitive layer (0.0 to 1.0)
    pub layer_weights: [f64; 8],

    /// Enable meta-learning (Layer 7)
    pub enable_meta_learning: bool,

    /// Enable collective intelligence (Layer 5)
    pub enable_collective: bool,

    /// Enable GPU acceleration
    pub enable_gpu: bool,

    /// Maximum iterations per cognitive cycle
    pub max_iterations_per_cycle: usize,

    /// Energy budget per cycle (thermodynamic units)
    pub energy_budget_per_cycle: f64,

    /// Thread pool size (0 = auto-detect)
    pub thread_pool_size: usize,

    /// Enable tracing/logging
    pub enable_tracing: bool,
}

/// Homeostasis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HomeostasisConfig {
    /// Energy level setpoint (0.0 to 1.0)
    pub energy_setpoint: f64,

    /// Entropy setpoint (bits)
    pub entropy_setpoint: f64,

    /// Temperature setpoint (normalized)
    pub temperature_setpoint: f64,

    /// Criticality setpoint (edge of chaos)
    pub criticality_setpoint: f64,

    /// Phi (consciousness) setpoint
    pub phi_setpoint: f64,

    /// PID controller gains
    pub pid_gains: PIDGains,
}

/// PID controller gains for homeostatic regulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIDGains {
    /// Proportional gain
    pub kp: f64,

    /// Integral gain
    pub ki: f64,

    /// Derivative gain
    pub kd: f64,
}

impl Default for QksConfig {
    fn default() -> Self {
        Self {
            // Consciousness threshold from IIT 3.0
            phi_threshold: 1.0,

            // Homeostatic defaults
            homeostatic_setpoints: HomeostasisConfig::default(),

            // Equal layer weights by default
            layer_weights: [1.0; 8],

            // Enable advanced features
            enable_meta_learning: true,
            enable_collective: true,
            enable_gpu: cfg!(target_os = "macos"),

            // Cognitive cycle parameters
            max_iterations_per_cycle: 100,
            energy_budget_per_cycle: 100.0,

            // Concurrency
            thread_pool_size: 0, // Auto-detect

            // Logging
            enable_tracing: true,
        }
    }
}

impl Default for HomeostasisConfig {
    fn default() -> Self {
        Self {
            energy_setpoint: 0.7,
            entropy_setpoint: 2.0,
            temperature_setpoint: 0.5,
            criticality_setpoint: 0.8,
            phi_setpoint: 1.0,
            pid_gains: PIDGains::default(),
        }
    }
}

impl Default for PIDGains {
    fn default() -> Self {
        // Tuned PID gains from control theory
        Self {
            kp: 1.0,
            ki: 0.1,
            kd: 0.05,
        }
    }
}

/// Configuration builder pattern
pub struct QksConfigBuilder {
    config: QksConfig,
}

impl QksConfigBuilder {
    /// Create new configuration builder
    pub fn new() -> Self {
        Self {
            config: QksConfig::default(),
        }
    }

    /// Set consciousness threshold
    pub fn phi_threshold(mut self, phi: f64) -> Self {
        self.config.phi_threshold = phi;
        self
    }

    /// Set energy setpoint
    pub fn energy_setpoint(mut self, energy: f64) -> Self {
        self.config.homeostatic_setpoints.energy_setpoint = energy;
        self
    }

    /// Set layer weights
    pub fn layer_weights(mut self, weights: [f64; 8]) -> Self {
        self.config.layer_weights = weights;
        self
    }

    /// Enable/disable meta-learning
    pub fn meta_learning(mut self, enable: bool) -> Self {
        self.config.enable_meta_learning = enable;
        self
    }

    /// Enable/disable collective intelligence
    pub fn collective(mut self, enable: bool) -> Self {
        self.config.enable_collective = enable;
        self
    }

    /// Enable/disable GPU acceleration
    pub fn gpu(mut self, enable: bool) -> Self {
        self.config.enable_gpu = enable;
        self
    }

    /// Set max iterations per cycle
    pub fn max_iterations(mut self, max: usize) -> Self {
        self.config.max_iterations_per_cycle = max;
        self
    }

    /// Set energy budget
    pub fn energy_budget(mut self, budget: f64) -> Self {
        self.config.energy_budget_per_cycle = budget;
        self
    }

    /// Set thread pool size
    pub fn threads(mut self, count: usize) -> Self {
        self.config.thread_pool_size = count;
        self
    }

    /// Set PID gains
    pub fn pid_gains(mut self, kp: f64, ki: f64, kd: f64) -> Self {
        self.config.homeostatic_setpoints.pid_gains = PIDGains { kp, ki, kd };
        self
    }

    /// Build final configuration
    pub fn build(self) -> QksConfig {
        self.config
    }
}

impl Default for QksConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = QksConfig::default();
        assert_eq!(config.phi_threshold, 1.0);
        assert_eq!(config.layer_weights.len(), 8);
        assert!(config.enable_meta_learning);
    }

    #[test]
    fn test_config_builder() {
        let config = QksConfigBuilder::new()
            .phi_threshold(1.5)
            .energy_setpoint(0.8)
            .meta_learning(false)
            .gpu(true)
            .build();

        assert_eq!(config.phi_threshold, 1.5);
        assert_eq!(config.homeostatic_setpoints.energy_setpoint, 0.8);
        assert!(!config.enable_meta_learning);
        assert!(config.enable_gpu);
    }

    #[test]
    fn test_serialization() {
        let config = QksConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: QksConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.phi_threshold, deserialized.phi_threshold);
    }
}
