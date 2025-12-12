//! Configuration management for PADS system

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use crate::error::{PadsError, PadsResult};

/// Main PADS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadsConfig {
    /// System name
    pub name: String,
    /// Enable quantum agents
    pub enable_quantum_agents: bool,
    /// Enable risk management
    pub enable_risk_management: bool,
    /// Enable pattern analysis
    pub enable_pattern_analysis: bool,
    /// Enable panarchy system
    pub enable_panarchy_system: bool,
    /// Enable board system
    pub enable_board_system: bool,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Agent configuration
    pub agents: AgentConfig,
    /// Risk configuration
    pub risk: RiskConfig,
    /// Analysis configuration
    pub analysis: AnalysisConfig,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Target decision latency in nanoseconds
    pub decision_latency_ns: u64,
    /// Target analysis latency in nanoseconds
    pub analysis_latency_ns: u64,
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
    /// Cache size
    pub cache_size: usize,
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Number of quantum agents
    pub count: usize,
    /// Agent coordination mode
    pub coordination_mode: String,
    /// Update frequency in milliseconds
    pub update_frequency_ms: u64,
    /// Agent-specific parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

/// Risk configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum risk tolerance
    pub max_risk_tolerance: f64,
    /// Risk assessment window in seconds
    pub assessment_window_seconds: u64,
    /// Enable black swan detection
    pub enable_black_swan_detection: bool,
    /// Risk thresholds
    pub thresholds: HashMap<String, f64>,
}

/// Analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Enable antifragility analysis
    pub enable_antifragility: bool,
    /// Enable panarchy analysis
    pub enable_panarchy: bool,
    /// Enable narrative forecasting
    pub enable_narrative_forecasting: bool,
    /// Analysis window size
    pub window_size: usize,
    /// Analysis parameters
    pub parameters: HashMap<String, serde_json::Value>,
}

impl Default for PadsConfig {
    fn default() -> Self {
        Self {
            name: "PADS-Unified".to_string(),
            enable_quantum_agents: true,
            enable_risk_management: true,
            enable_pattern_analysis: true,
            enable_panarchy_system: true,
            enable_board_system: true,
            performance: PerformanceConfig::default(),
            agents: AgentConfig::default(),
            risk: RiskConfig::default(),
            analysis: AnalysisConfig::default(),
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            decision_latency_ns: 10_000,
            analysis_latency_ns: 5_000,
            enable_simd: true,
            enable_parallel: true,
            cache_size: 10_000,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("quantum_depth".to_string(), serde_json::Value::Number(serde_json::Number::from(4)));
        parameters.insert("coherence_threshold".to_string(), serde_json::Value::Number(serde_json::Number::from_f64(0.8).unwrap()));
        
        Self {
            count: 12,
            coordination_mode: "collaborative".to_string(),
            update_frequency_ms: 100,
            parameters,
        }
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("volatility".to_string(), 0.05);
        thresholds.insert("drawdown".to_string(), 0.1);
        thresholds.insert("correlation".to_string(), 0.8);
        
        Self {
            max_risk_tolerance: 0.05,
            assessment_window_seconds: 300,
            enable_black_swan_detection: true,
            thresholds,
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        let mut parameters = HashMap::new();
        parameters.insert("pcr_period".to_string(), serde_json::Value::Number(serde_json::Number::from(14)));
        parameters.insert("antifragility_window".to_string(), serde_json::Value::Number(serde_json::Number::from(50)));
        
        Self {
            enable_antifragility: true,
            enable_panarchy: true,
            enable_narrative_forecasting: true,
            window_size: 100,
            parameters,
        }
    }
}

impl PadsConfig {
    /// Create configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> PadsResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PadsError::ConfigError(format!("Failed to read config file: {}", e)))?;
        
        if content.trim().ends_with(".toml") || content.contains("[") {
            toml::from_str(&content)
                .map_err(|e| PadsError::ConfigError(format!("Failed to parse TOML config: {}", e)))
        } else {
            serde_json::from_str(&content)
                .map_err(|e| PadsError::ConfigError(format!("Failed to parse JSON config: {}", e)))
        }
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> PadsResult<()> {
        let content = if path.as_ref().extension().and_then(|s| s.to_str()) == Some("toml") {
            toml::to_string_pretty(self)
                .map_err(|e| PadsError::ConfigError(format!("Failed to serialize TOML config: {}", e)))?
        } else {
            serde_json::to_string_pretty(self)
                .map_err(|e| PadsError::ConfigError(format!("Failed to serialize JSON config: {}", e)))?
        };
        
        std::fs::write(path, content)
            .map_err(|e| PadsError::ConfigError(format!("Failed to write config file: {}", e)))
    }
    
    /// Validate configuration
    pub fn validate(&self) -> PadsResult<()> {
        if self.agents.count == 0 {
            return Err(PadsError::ConfigError("Agent count must be greater than 0".to_string()));
        }
        
        if self.risk.max_risk_tolerance < 0.0 || self.risk.max_risk_tolerance > 1.0 {
            return Err(PadsError::ConfigError("Risk tolerance must be between 0.0 and 1.0".to_string()));
        }
        
        if self.performance.decision_latency_ns == 0 {
            return Err(PadsError::ConfigError("Decision latency target must be greater than 0".to_string()));
        }
        
        Ok(())
    }
    
    /// Update configuration with overrides
    pub fn update_from_overrides(&mut self, overrides: &HashMap<String, serde_json::Value>) -> PadsResult<()> {
        for (key, value) in overrides {
            match key.as_str() {
                "agents.count" => {
                    if let Some(count) = value.as_u64() {
                        self.agents.count = count as usize;
                    }
                }
                "risk.max_risk_tolerance" => {
                    if let Some(tolerance) = value.as_f64() {
                        self.risk.max_risk_tolerance = tolerance;
                    }
                }
                "performance.enable_simd" => {
                    if let Some(enable) = value.as_bool() {
                        self.performance.enable_simd = enable;
                    }
                }
                _ => {
                    // Store unknown overrides in parameters
                    if key.starts_with("agents.") {
                        self.agents.parameters.insert(key.strip_prefix("agents.").unwrap().to_string(), value.clone());
                    } else if key.starts_with("risk.") {
                        // Handle risk parameter overrides
                    } else if key.starts_with("analysis.") {
                        self.analysis.parameters.insert(key.strip_prefix("analysis.").unwrap().to_string(), value.clone());
                    }
                }
            }
        }
        
        self.validate()
    }
}