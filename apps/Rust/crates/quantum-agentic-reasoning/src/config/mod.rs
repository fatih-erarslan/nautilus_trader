//! Configuration module for Quantum Agentic Reasoning
//!
//! This module provides configuration structures and default values
//! for the QAR system.

use serde::{Deserialize, Serialize};
use std::time::Duration;
use crate::core::QarError;

/// Hardware configuration for quantum and classical computing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Prefer quantum computing when available
    pub prefer_quantum: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Maximum number of qubits to use
    pub max_qubits: usize,
    /// Quantum device backend preference
    pub quantum_backend: String,
    /// Number of shots for quantum measurements
    pub shots: Option<usize>,
    /// Quantum computation timeout
    pub quantum_timeout: Duration,
    /// Fallback to classical after this many quantum failures
    pub quantum_fallback_threshold: usize,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            prefer_quantum: true,
            enable_gpu: true,
            max_qubits: 16,
            quantum_backend: "default.qubit".to_string(),
            shots: None,
            quantum_timeout: Duration::from_secs(10),
            quantum_fallback_threshold: 3,
        }
    }
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum length of decision history
    pub memory_length: usize,
    /// Maximum number of patterns to store
    pub max_patterns: usize,
    /// Cache size for quantum circuits
    pub circuit_cache_size: usize,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Memory cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            memory_length: 50,
            max_patterns: 1000,
            circuit_cache_size: 100,
            enable_compression: true,
            cleanup_interval: Duration::from_secs(300), // 5 minutes
        }
    }
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Number of parallel threads
    pub num_threads: Option<usize>,
    /// Enable performance monitoring
    pub enable_monitoring: bool,
    /// Performance metrics collection interval
    pub metrics_interval: Duration,
    /// Enable caching
    pub enable_caching: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            num_threads: None, // Use system default
            enable_monitoring: true,
            metrics_interval: Duration::from_secs(60),
            enable_caching: true,
        }
    }
}

/// Decision engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionConfig {
    /// Confidence threshold for actionable decisions
    pub decision_threshold: f64,
    /// Use quantum circuits for decision making
    pub use_quantum_decision: bool,
    /// Use quantum circuits for pattern recognition
    pub use_quantum_patterns: bool,
    /// Use quantum circuits for market analysis
    pub use_quantum_analysis: bool,
    /// Learning rate for adaptive algorithms
    pub learning_rate: f64,
    /// Adaptation rate for parameter updates
    pub adaptation_rate: f64,
}

impl Default for DecisionConfig {
    fn default() -> Self {
        Self {
            decision_threshold: 0.3,
            use_quantum_decision: true,
            use_quantum_patterns: true,
            use_quantum_analysis: true,
            learning_rate: 0.01,
            adaptation_rate: 0.05,
        }
    }
}

/// Logging and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level (trace, debug, info, warn, error)
    pub level: String,
    /// Enable structured logging
    pub structured: bool,
    /// Log file path
    pub file_path: Option<String>,
    /// Maximum log file size in MB
    pub max_file_size: u64,
    /// Number of log files to keep
    pub max_files: usize,
    /// Enable performance logging
    pub enable_performance: bool,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            structured: true,
            file_path: Some("qar.log".to_string()),
            max_file_size: 100, // 100 MB
            max_files: 5,
            enable_performance: true,
        }
    }
}

/// Main configuration structure for QAR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QarConfig {
    /// Hardware configuration
    pub hardware: HardwareConfig,
    /// Memory management configuration
    pub memory: MemoryConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Decision engine configuration
    pub decision: DecisionConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
    /// Enable experimental features
    pub experimental: bool,
    /// Debug mode
    pub debug: bool,
}

impl Default for QarConfig {
    fn default() -> Self {
        Self {
            hardware: HardwareConfig::default(),
            memory: MemoryConfig::default(),
            performance: PerformanceConfig::default(),
            decision: DecisionConfig::default(),
            logging: LoggingConfig::default(),
            experimental: false,
            debug: false,
        }
    }
}

impl QarConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Load configuration from a file
    pub fn load_from_file(path: &str) -> Result<Self, QarError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| QarError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        let config: Self = toml::from_str(&content)
            .map_err(|e| QarError::ConfigError(format!("Failed to parse config file: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to a file
    pub fn save_to_file(&self, path: &str) -> Result<(), QarError> {
        self.validate()?;
        
        let content = toml::to_string_pretty(self)
            .map_err(|e| QarError::ConfigError(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| QarError::ConfigError(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<(), QarError> {
        // Validate hardware config
        if self.hardware.max_qubits == 0 {
            return Err(QarError::ConfigError("max_qubits must be greater than 0".to_string()));
        }
        
        if self.hardware.max_qubits > 32 {
            return Err(QarError::ConfigError("max_qubits cannot exceed 32".to_string()));
        }

        // Validate memory config
        if self.memory.memory_length == 0 {
            return Err(QarError::ConfigError("memory_length must be greater than 0".to_string()));
        }

        if self.memory.max_patterns == 0 {
            return Err(QarError::ConfigError("max_patterns must be greater than 0".to_string()));
        }

        // Validate decision config
        if !(0.0..=1.0).contains(&self.decision.decision_threshold) {
            return Err(QarError::ConfigError("decision_threshold must be between 0.0 and 1.0".to_string()));
        }

        if self.decision.learning_rate <= 0.0 || self.decision.learning_rate > 1.0 {
            return Err(QarError::ConfigError("learning_rate must be between 0.0 and 1.0".to_string()));
        }

        if self.decision.adaptation_rate <= 0.0 || self.decision.adaptation_rate > 1.0 {
            return Err(QarError::ConfigError("adaptation_rate must be between 0.0 and 1.0".to_string()));
        }

        // Validate logging config
        let valid_levels = ["trace", "debug", "info", "warn", "error"];
        if !valid_levels.contains(&self.logging.level.as_str()) {
            return Err(QarError::ConfigError(format!(
                "Invalid log level '{}'. Must be one of: {}",
                self.logging.level,
                valid_levels.join(", ")
            )));
        }

        Ok(())
    }

    /// Create a development configuration
    pub fn development() -> Self {
        Self {
            hardware: HardwareConfig {
                prefer_quantum: false, // Use classical for development
                enable_gpu: true,
                max_qubits: 8,
                quantum_backend: "default.qubit".to_string(),
                shots: Some(1000),
                quantum_timeout: Duration::from_secs(5),
                quantum_fallback_threshold: 1,
            },
            memory: MemoryConfig {
                memory_length: 20,
                max_patterns: 100,
                circuit_cache_size: 50,
                enable_compression: false,
                cleanup_interval: Duration::from_secs(60),
            },
            performance: PerformanceConfig {
                enable_simd: false,
                num_threads: Some(2),
                enable_monitoring: true,
                metrics_interval: Duration::from_secs(10),
                enable_caching: true,
            },
            decision: DecisionConfig {
                decision_threshold: 0.2,
                use_quantum_decision: false,
                use_quantum_patterns: false,
                use_quantum_analysis: false,
                learning_rate: 0.1,
                adaptation_rate: 0.1,
            },
            logging: LoggingConfig {
                level: "debug".to_string(),
                structured: true,
                file_path: Some("qar_dev.log".to_string()),
                max_file_size: 10,
                max_files: 3,
                enable_performance: true,
            },
            experimental: true,
            debug: true,
        }
    }

    /// Create a production configuration
    pub fn production() -> Self {
        Self {
            hardware: HardwareConfig {
                prefer_quantum: true,
                enable_gpu: true,
                max_qubits: 16,
                quantum_backend: "lightning.qubit".to_string(),
                shots: None,
                quantum_timeout: Duration::from_secs(30),
                quantum_fallback_threshold: 5,
            },
            memory: MemoryConfig {
                memory_length: 100,
                max_patterns: 5000,
                circuit_cache_size: 500,
                enable_compression: true,
                cleanup_interval: Duration::from_secs(600),
            },
            performance: PerformanceConfig {
                enable_simd: true,
                num_threads: None,
                enable_monitoring: true,
                metrics_interval: Duration::from_secs(300),
                enable_caching: true,
            },
            decision: DecisionConfig {
                decision_threshold: 0.4,
                use_quantum_decision: true,
                use_quantum_patterns: true,
                use_quantum_analysis: true,
                learning_rate: 0.005,
                adaptation_rate: 0.02,
            },
            logging: LoggingConfig {
                level: "info".to_string(),
                structured: true,
                file_path: Some("/var/log/qar.log".to_string()),
                max_file_size: 1000,
                max_files: 10,
                enable_performance: true,
            },
            experimental: false,
            debug: false,
        }
    }

    /// Create a high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            hardware: HardwareConfig {
                prefer_quantum: true,
                enable_gpu: true,
                max_qubits: 24,
                quantum_backend: "lightning.gpu".to_string(),
                shots: None,
                quantum_timeout: Duration::from_secs(60),
                quantum_fallback_threshold: 10,
            },
            memory: MemoryConfig {
                memory_length: 200,
                max_patterns: 10000,
                circuit_cache_size: 1000,
                enable_compression: true,
                cleanup_interval: Duration::from_secs(900),
            },
            performance: PerformanceConfig {
                enable_simd: true,
                num_threads: None,
                enable_monitoring: true,
                metrics_interval: Duration::from_secs(60),
                enable_caching: true,
            },
            decision: DecisionConfig {
                decision_threshold: 0.35,
                use_quantum_decision: true,
                use_quantum_patterns: true,
                use_quantum_analysis: true,
                learning_rate: 0.001,
                adaptation_rate: 0.01,
            },
            logging: LoggingConfig {
                level: "warn".to_string(),
                structured: true,
                file_path: Some("/var/log/qar_performance.log".to_string()),
                max_file_size: 500,
                max_files: 20,
                enable_performance: true,
            },
            experimental: true,
            debug: false,
        }
    }

    /// Get the number of threads to use
    pub fn get_num_threads(&self) -> usize {
        self.performance.num_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4)
        })
    }

    /// Check if quantum computing is enabled
    pub fn is_quantum_enabled(&self) -> bool {
        self.hardware.prefer_quantum && 
        (self.decision.use_quantum_decision || 
         self.decision.use_quantum_patterns || 
         self.decision.use_quantum_analysis)
    }

    /// Get the effective quantum backend
    pub fn get_quantum_backend(&self) -> &str {
        if self.hardware.enable_gpu {
            match self.hardware.quantum_backend.as_str() {
                "default.qubit" => "lightning.gpu",
                "lightning.qubit" => "lightning.gpu",
                backend => backend,
            }
        } else {
            &self.hardware.quantum_backend
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = QarConfig::default();
        assert!(config.validate().is_ok());
        assert!(config.hardware.prefer_quantum);
        assert_eq!(config.decision.decision_threshold, 0.3);
    }

    #[test]
    fn test_development_config() {
        let config = QarConfig::development();
        assert!(config.validate().is_ok());
        assert!(!config.hardware.prefer_quantum);
        assert!(config.debug);
        assert_eq!(config.logging.level, "debug");
    }

    #[test]
    fn test_production_config() {
        let config = QarConfig::production();
        assert!(config.validate().is_ok());
        assert!(config.hardware.prefer_quantum);
        assert!(!config.debug);
        assert_eq!(config.logging.level, "info");
    }

    #[test]
    fn test_config_validation() {
        let mut config = QarConfig::default();
        
        // Test invalid decision threshold
        config.decision.decision_threshold = 1.5;
        assert!(config.validate().is_err());
        
        config.decision.decision_threshold = 0.5;
        assert!(config.validate().is_ok());
        
        // Test invalid max_qubits
        config.hardware.max_qubits = 0;
        assert!(config.validate().is_err());
        
        config.hardware.max_qubits = 50;
        assert!(config.validate().is_err());
        
        config.hardware.max_qubits = 16;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_save_load() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = QarConfig::development();
        
        // Save config
        config.save_to_file(temp_file.path().to_str().unwrap()).unwrap();
        
        // Load config
        let loaded_config = QarConfig::load_from_file(temp_file.path().to_str().unwrap()).unwrap();
        
        assert_eq!(config.debug, loaded_config.debug);
        assert_eq!(config.hardware.prefer_quantum, loaded_config.hardware.prefer_quantum);
        assert_eq!(config.decision.decision_threshold, loaded_config.decision.decision_threshold);
    }

    #[test]
    fn test_quantum_enabled() {
        let mut config = QarConfig::default();
        assert!(config.is_quantum_enabled());
        
        config.hardware.prefer_quantum = false;
        assert!(!config.is_quantum_enabled());
        
        config.hardware.prefer_quantum = true;
        config.decision.use_quantum_decision = false;
        config.decision.use_quantum_patterns = false;
        config.decision.use_quantum_analysis = false;
        assert!(!config.is_quantum_enabled());
    }

    #[test]
    fn test_quantum_backend() {
        let mut config = QarConfig::default();
        config.hardware.quantum_backend = "default.qubit".to_string();
        config.hardware.enable_gpu = true;
        
        assert_eq!(config.get_quantum_backend(), "lightning.gpu");
        
        config.hardware.enable_gpu = false;
        assert_eq!(config.get_quantum_backend(), "default.qubit");
    }

    #[test]
    fn test_num_threads() {
        let mut config = QarConfig::default();
        config.performance.num_threads = Some(8);
        assert_eq!(config.get_num_threads(), 8);
        
        config.performance.num_threads = None;
        assert!(config.get_num_threads() >= 1);
    }
}