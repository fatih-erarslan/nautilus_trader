//! # Configuration Module
//!
//! Loads and manages configuration from TOML files.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub system: SystemConfig,
    pub performance: PerformanceConfig,
    pub quantum: QuantumConfig,
    pub organisms: OrganismsConfig,
    pub voting: VotingConfig,
    pub risk: RiskConfig,
    pub analytics: AnalyticsConfig,
    pub evolution: EvolutionConfig,
    pub mcp: McpConfig,
    pub cqgs: CqgsConfig,
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    pub enabled: bool,
    pub max_pairs: usize,
    pub selection_interval_ms: u64,
    pub parallel_processing: bool,
    pub thread_pool_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub simd_enabled: bool,
    pub gpu_correlation: bool,
    pub cache_size: usize,
    pub parallel_organisms: bool,
    pub max_latency_ms: u64,
    pub batch_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    pub mode: String,
    pub max_qubits: u32,
    pub auto_switch: bool,
    pub grover_iterations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismsConfig {
    pub cuckoo: OrganismSettings,
    pub wasp: OrganismSettings,
    pub cordyceps: OrganismSettings,
    pub mycelial: OrganismSettings,
    pub octopus: OrganismSettings,
    pub anglerfish: OrganismSettings,
    pub komodo: OrganismSettings,
    pub tardigrade: OrganismSettings,
    pub electric_eel: OrganismSettings,
    pub platypus: OrganismSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismSettings {
    pub enabled: bool,
    #[serde(flatten)]
    pub params: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VotingConfig {
    pub consensus_threshold: f64,
    pub emergence_weight: f64,
    pub quantum_enhancement: bool,
    pub minimum_organisms: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_host_resistance: f64,
    pub parasite_detection_threshold: f64,
    pub emergency_cryptobiosis: f64,
    pub diversity_requirement: f64,
    pub max_exposure: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    pub enabled: bool,
    pub metrics_interval_ms: u64,
    pub performance_tracking: bool,
    pub emergence_detection: bool,
    pub dashboard_port: u16,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    pub enabled: bool,
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub selection_pressure: f64,
    pub generations_per_epoch: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpConfig {
    pub enabled: bool,
    pub port: u16,
    pub websocket_enabled: bool,
    pub resource_handlers: bool,
    pub tool_handlers: bool,
    pub max_connections: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CqgsConfig {
    pub enabled: bool,
    pub sentinels: u32,
    pub hyperbolic_topology: bool,
    pub neural_enhancement: bool,
    pub zero_mock_enforcement: bool,
    pub governance_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub file: String,
    pub console: bool,
    pub format: String,
    pub rotation: String,
}

impl Config {
    /// Load configuration from TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let contents = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;

        let config: Config = toml::from_str(&contents)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))?;

        config.validate()?;

        Ok(config)
    }

    /// Load configuration from environment with fallback to file
    pub fn from_env_or_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Check for environment variable override
        if let Ok(config_path) = std::env::var("PARASITIC_CONFIG") {
            return Self::from_file(config_path);
        }

        // Fall back to provided path
        Self::from_file(path)
    }

    /// Validate configuration values
    pub fn validate(&self) -> Result<()> {
        // Validate performance settings
        if self.performance.max_latency_ms == 0 {
            anyhow::bail!("max_latency_ms must be greater than 0");
        }

        // Validate voting thresholds
        if self.voting.consensus_threshold < 0.0 || self.voting.consensus_threshold > 1.0 {
            anyhow::bail!("consensus_threshold must be between 0.0 and 1.0");
        }

        // Validate risk parameters
        if self.risk.max_exposure < 0.0 || self.risk.max_exposure > 1.0 {
            anyhow::bail!("max_exposure must be between 0.0 and 1.0");
        }

        // Validate evolution parameters
        if self.evolution.mutation_rate < 0.0 || self.evolution.mutation_rate > 1.0 {
            anyhow::bail!("mutation_rate must be between 0.0 and 1.0");
        }

        if self.evolution.crossover_rate < 0.0 || self.evolution.crossover_rate > 1.0 {
            anyhow::bail!("crossover_rate must be between 0.0 and 1.0");
        }

        // Validate CQGS settings
        if self.cqgs.sentinels == 0 {
            anyhow::bail!("CQGS sentinels must be greater than 0");
        }

        if self.cqgs.governance_threshold < 0.0 || self.cqgs.governance_threshold > 1.0 {
            anyhow::bail!("governance_threshold must be between 0.0 and 1.0");
        }

        Ok(())
    }

    /// Get enabled organisms
    pub fn enabled_organisms(&self) -> Vec<String> {
        let mut enabled = Vec::new();

        if self.organisms.cuckoo.enabled {
            enabled.push("cuckoo".to_string());
        }
        if self.organisms.wasp.enabled {
            enabled.push("wasp".to_string());
        }
        if self.organisms.cordyceps.enabled {
            enabled.push("cordyceps".to_string());
        }
        if self.organisms.mycelial.enabled {
            enabled.push("mycelial".to_string());
        }
        if self.organisms.octopus.enabled {
            enabled.push("octopus".to_string());
        }
        if self.organisms.anglerfish.enabled {
            enabled.push("anglerfish".to_string());
        }
        if self.organisms.komodo.enabled {
            enabled.push("komodo".to_string());
        }
        if self.organisms.tardigrade.enabled {
            enabled.push("tardigrade".to_string());
        }
        if self.organisms.electric_eel.enabled {
            enabled.push("electric_eel".to_string());
        }
        if self.organisms.platypus.enabled {
            enabled.push("platypus".to_string());
        }

        enabled
    }
}

impl Default for Config {
    fn default() -> Self {
        // Load default configuration from embedded TOML
        let default_toml = include_str!("../config/parasitic.toml");
        toml::from_str(default_toml).expect("Default config should be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_load_default_config() {
        let config = Config::default();
        assert!(config.system.enabled);
        assert_eq!(config.system.max_pairs, 500);
        assert_eq!(config.cqgs.sentinels, 49);
    }

    #[test]
    fn test_load_from_file() {
        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.toml");

        let config_content = r#"
[system]
enabled = true
max_pairs = 100
selection_interval_ms = 50
parallel_processing = false
thread_pool_size = 4

[performance]
simd_enabled = false
gpu_correlation = false
cache_size = 500
parallel_organisms = false
max_latency_ms = 5
batch_size = 50

[quantum]
mode = "classical"
max_qubits = 4
auto_switch = false
grover_iterations = 50

[organisms]
[organisms.cuckoo]
enabled = false

[organisms.wasp]
enabled = false

[organisms.cordyceps]
enabled = false

[organisms.mycelial]
enabled = false

[organisms.octopus]
enabled = false

[organisms.anglerfish]
enabled = false

[organisms.komodo]
enabled = false

[organisms.tardigrade]
enabled = false

[organisms.electric_eel]
enabled = false

[organisms.platypus]
enabled = false

[voting]
consensus_threshold = 0.5
emergence_weight = 1.0
quantum_enhancement = false
minimum_organisms = 2

[risk]
max_host_resistance = 0.7
parasite_detection_threshold = 0.8
emergency_cryptobiosis = 0.1
diversity_requirement = 0.3
max_exposure = 0.2

[analytics]
enabled = false
metrics_interval_ms = 2000
performance_tracking = false
emergence_detection = false
dashboard_port = 8082

[evolution]
enabled = false
population_size = 50
mutation_rate = 0.05
crossover_rate = 0.6
selection_pressure = 0.4
generations_per_epoch = 5

[mcp]
enabled = false
port = 8080
websocket_enabled = false
resource_handlers = false
tool_handlers = false
max_connections = 50

[cqgs]
enabled = false
sentinels = 10
hyperbolic_topology = false
neural_enhancement = false
zero_mock_enforcement = false
governance_threshold = 0.8

[logging]
level = "debug"
file = "test.log"
console = false
format = "text"
rotation = "hourly"
"#;

        fs::write(&file_path, config_content).unwrap();

        let config = Config::from_file(&file_path).unwrap();
        assert_eq!(config.system.max_pairs, 100);
        assert_eq!(config.performance.max_latency_ms, 5);
        assert_eq!(config.cqgs.sentinels, 10);
    }

    #[test]
    fn test_validate_config() {
        let mut config = Config::default();

        // Valid config should pass
        assert!(config.validate().is_ok());

        // Invalid consensus threshold
        config.voting.consensus_threshold = 1.5;
        assert!(config.validate().is_err());
        config.voting.consensus_threshold = 0.6;

        // Invalid mutation rate
        config.evolution.mutation_rate = -0.1;
        assert!(config.validate().is_err());
        config.evolution.mutation_rate = 0.1;

        // Invalid max exposure
        config.risk.max_exposure = 2.0;
        assert!(config.validate().is_err());
        config.risk.max_exposure = 0.3;

        // Zero sentinels
        config.cqgs.sentinels = 0;
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_enabled_organisms() {
        let config = Config::default();
        let enabled = config.enabled_organisms();

        // All organisms should be enabled by default
        assert_eq!(enabled.len(), 10);
        assert!(enabled.contains(&"cuckoo".to_string()));
        assert!(enabled.contains(&"wasp".to_string()));
        assert!(enabled.contains(&"cordyceps".to_string()));
        assert!(enabled.contains(&"mycelial".to_string()));
        assert!(enabled.contains(&"octopus".to_string()));
        assert!(enabled.contains(&"anglerfish".to_string()));
        assert!(enabled.contains(&"komodo".to_string()));
        assert!(enabled.contains(&"tardigrade".to_string()));
        assert!(enabled.contains(&"electric_eel".to_string()));
        assert!(enabled.contains(&"platypus".to_string()));
    }
}
