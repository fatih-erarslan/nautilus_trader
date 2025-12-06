//! Configuration management for QBMIA Core

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use crate::error::{QBMIAError, Result};

/// Main configuration for QBMIA Agent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub agent_id: String,
    pub checkpoint_dir: String,
    pub checkpoint_interval: u64,
    pub quantum: QuantumConfig,
    pub memory: MemoryConfig,
    pub strategy: StrategyConfig,
    pub hardware: HardwareConfig,
    pub logging: LoggingConfig,
    pub performance: PerformanceConfig,
}

/// Quantum simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    pub num_qubits: usize,
    pub num_layers: usize,
    pub learning_rate: f64,
    pub convergence_threshold: f64,
    pub max_iterations: usize,
    pub device_type: DeviceType,
}

/// Memory system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub capacity: usize,
    pub short_term_size: usize,
    pub episodic_size: usize,
    pub consolidation_rate: f64,
    pub recall_threshold: f64,
    pub attention_enabled: bool,
}

/// Strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub machiavellian_sensitivity: f64,
    pub wealth_threshold: f64,
    pub memory_decay: f64,
    pub volatility_threshold: f64,
    pub manipulation_patterns: HashMap<String, f64>,
}

/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub force_cpu: bool,
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub max_workers: usize,
    pub memory_limit_mb: usize,
    pub enable_profiling: bool,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub enable_file_logging: bool,
    pub log_file: String,
    pub enable_metrics: bool,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub enable_benchmarking: bool,
    pub target_latency_ms: f64,
    pub memory_efficiency_target: f64,
    pub enable_optimization: bool,
}

/// Device type for quantum simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    Cpu,
    Gpu,
    Auto,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            agent_id: "QBMIA_RUST_001".to_string(),
            checkpoint_dir: "./checkpoints".to_string(),
            checkpoint_interval: 300, // 5 minutes
            quantum: QuantumConfig::default(),
            memory: MemoryConfig::default(),
            strategy: StrategyConfig::default(),
            hardware: HardwareConfig::default(),
            logging: LoggingConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            num_qubits: 16,
            num_layers: 3,
            learning_rate: 0.1,
            convergence_threshold: 1e-4,
            max_iterations: 200,
            device_type: DeviceType::Auto,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            capacity: 10000,
            short_term_size: 100,
            episodic_size: 1000,
            consolidation_rate: 0.1,
            recall_threshold: 0.7,
            attention_enabled: true,
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        let mut manipulation_patterns = HashMap::new();
        manipulation_patterns.insert("spoofing".to_string(), 0.3);
        manipulation_patterns.insert("layering".to_string(), 0.25);
        manipulation_patterns.insert("wash_trading".to_string(), 0.2);
        manipulation_patterns.insert("pump_dump".to_string(), 0.15);
        manipulation_patterns.insert("front_running".to_string(), 0.1);
        
        Self {
            machiavellian_sensitivity: 0.7,
            wealth_threshold: 0.8,
            memory_decay: 0.95,
            volatility_threshold: 0.3,
            manipulation_patterns,
        }
    }
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            force_cpu: false,
            enable_simd: true,
            enable_parallel: true,
            max_workers: num_cpus::get(),
            memory_limit_mb: 4096,
            enable_profiling: true,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "INFO".to_string(),
            enable_file_logging: true,
            log_file: "qbmia.log".to_string(),
            enable_metrics: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_benchmarking: true,
            target_latency_ms: 1.0, // Sub-millisecond target
            memory_efficiency_target: 0.8,
            enable_optimization: true,
        }
    }
}

impl Config {
    /// Load configuration from JSON file
    pub fn from_file(path: &str) -> Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to JSON file
    pub fn to_file(&self, path: &str) -> Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<()> {
        if self.quantum.num_qubits == 0 {
            return Err(QBMIAError::config("num_qubits must be greater than 0"));
        }
        
        if self.quantum.num_qubits > 64 {
            return Err(QBMIAError::config("num_qubits cannot exceed 64"));
        }
        
        if self.quantum.learning_rate <= 0.0 || self.quantum.learning_rate >= 1.0 {
            return Err(QBMIAError::config("learning_rate must be between 0 and 1"));
        }
        
        if self.memory.capacity == 0 {
            return Err(QBMIAError::config("memory capacity must be greater than 0"));
        }
        
        if self.hardware.max_workers == 0 {
            return Err(QBMIAError::config("max_workers must be greater than 0"));
        }
        
        if self.performance.target_latency_ms <= 0.0 {
            return Err(QBMIAError::config("target_latency_ms must be positive"));
        }
        
        Ok(())
    }
    
    /// Get optimal configuration for the current hardware
    pub fn optimize_for_hardware(&mut self) -> Result<()> {
        // Detect available hardware
        let num_cores = num_cpus::get();
        let available_memory = self.get_available_memory()?;
        
        // Adjust worker count
        self.hardware.max_workers = std::cmp::min(num_cores, 16);
        
        // Adjust memory usage
        if available_memory < 2048 {
            self.memory.capacity = std::cmp::min(self.memory.capacity, 5000);
            self.quantum.num_qubits = std::cmp::min(self.quantum.num_qubits, 12);
        }
        
        // Enable SIMD if available
        #[cfg(target_arch = "x86_64")]
        {
            self.hardware.enable_simd = is_x86_feature_detected!("avx2");
        }
        
        log::info!("Optimized configuration for hardware: {} cores, {} MB memory", 
                  num_cores, available_memory);
        
        Ok(())
    }
    
    fn get_available_memory(&self) -> Result<usize> {
        // Simplified memory detection - in practice would use system APIs
        Ok(8192) // Default 8GB
    }
}

// Helper function to check if running on a specific CPU
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.agent_id, "QBMIA_RUST_001");
        assert_eq!(config.quantum.num_qubits, 16);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        config.quantum.num_qubits = 0;
        assert!(config.validate().is_err());
        
        config.quantum.num_qubits = 16;
        config.quantum.learning_rate = 2.0;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(config.agent_id, deserialized.agent_id);
    }
    
    #[test]
    fn test_config_file_operations() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path().to_str().unwrap();
        
        // Save config
        config.to_file(path).unwrap();
        
        // Load config
        let loaded_config = Config::from_file(path).unwrap();
        assert_eq!(config.agent_id, loaded_config.agent_id);
    }
    
    #[test]
    fn test_hardware_optimization() {
        let mut config = Config::default();
        config.optimize_for_hardware().unwrap();
        assert!(config.hardware.max_workers > 0);
    }
}