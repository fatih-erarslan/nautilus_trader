//! # PADS Configuration
//!
//! Configuration management for the Panarchy Adaptive Decision System.

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use crate::core::{DecisionLayer, AdaptiveCyclePhase};
use super::types::AdaptiveConfig;

/// Main PADS system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PadsConfig {
    /// System identification
    pub system_id: String,
    
    /// Decision layer configuration
    pub decision_layers: DecisionLayerConfig,
    
    /// Panarchy framework settings
    pub panarchy: PanarchyConfig,
    
    /// Decision engine configuration
    pub decision_engine: DecisionEngineConfig,
    
    /// Integration settings
    pub integration: IntegrationConfig,
    
    /// Governance configuration
    pub governance: GovernanceConfig,
    
    /// Monitoring and metrics
    pub monitoring: MonitoringConfig,
    
    /// Performance settings
    pub performance: PerformanceConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
    
    /// Adaptive behavior settings
    pub adaptive: AdaptiveConfig,
}

impl Default for PadsConfig {
    fn default() -> Self {
        Self {
            system_id: "pads-default".to_string(),
            decision_layers: DecisionLayerConfig::default(),
            panarchy: PanarchyConfig::default(),
            decision_engine: DecisionEngineConfig::default(),
            integration: IntegrationConfig::default(),
            governance: GovernanceConfig::default(),
            monitoring: MonitoringConfig::default(),
            performance: PerformanceConfig::default(),
            security: SecurityConfig::default(),
            adaptive: AdaptiveConfig::default(),
        }
    }
}

impl PadsConfig {
    /// Create a new configuration builder
    pub fn builder() -> PadsConfigBuilder {
        PadsConfigBuilder::new()
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate decision layers
        self.decision_layers.validate()?;
        
        // Validate panarchy settings
        self.panarchy.validate()?;
        
        // Validate decision engine
        self.decision_engine.validate()?;
        
        // Validate integration settings
        self.integration.validate()?;
        
        // Validate governance
        self.governance.validate()?;
        
        // Validate monitoring
        self.monitoring.validate()?;
        
        // Validate performance settings
        self.performance.validate()?;
        
        // Validate security
        self.security.validate()?;
        
        Ok(())
    }
}

/// Decision layer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionLayerConfig {
    /// Enabled decision layers
    pub enabled_layers: Vec<DecisionLayer>,
    
    /// Layer-specific time horizons
    pub time_horizons: HashMap<DecisionLayer, Duration>,
    
    /// Layer priorities
    pub priorities: HashMap<DecisionLayer, u8>,
    
    /// Maximum concurrent decisions per layer
    pub max_concurrent_decisions: HashMap<DecisionLayer, usize>,
    
    /// Cross-layer coordination enabled
    pub cross_layer_coordination: bool,
    
    /// Escalation thresholds
    pub escalation_thresholds: HashMap<DecisionLayer, f64>,
}

impl Default for DecisionLayerConfig {
    fn default() -> Self {
        let mut time_horizons = HashMap::new();
        time_horizons.insert(DecisionLayer::Tactical, Duration::from_millis(100));
        time_horizons.insert(DecisionLayer::Operational, Duration::from_secs(30));
        time_horizons.insert(DecisionLayer::Strategic, Duration::from_secs(1800));
        time_horizons.insert(DecisionLayer::MetaStrategic, Duration::from_secs(21600));
        
        let mut priorities = HashMap::new();
        priorities.insert(DecisionLayer::Tactical, 4);
        priorities.insert(DecisionLayer::Operational, 3);
        priorities.insert(DecisionLayer::Strategic, 2);
        priorities.insert(DecisionLayer::MetaStrategic, 1);
        
        let mut max_concurrent = HashMap::new();
        max_concurrent.insert(DecisionLayer::Tactical, 10);
        max_concurrent.insert(DecisionLayer::Operational, 5);
        max_concurrent.insert(DecisionLayer::Strategic, 3);
        max_concurrent.insert(DecisionLayer::MetaStrategic, 1);
        
        let mut escalation_thresholds = HashMap::new();
        escalation_thresholds.insert(DecisionLayer::Tactical, 0.8);
        escalation_thresholds.insert(DecisionLayer::Operational, 0.7);
        escalation_thresholds.insert(DecisionLayer::Strategic, 0.6);
        escalation_thresholds.insert(DecisionLayer::MetaStrategic, 0.5);
        
        Self {
            enabled_layers: DecisionLayer::all_layers(),
            time_horizons,
            priorities,
            max_concurrent_decisions: max_concurrent,
            cross_layer_coordination: true,
            escalation_thresholds,
        }
    }
}

impl DecisionLayerConfig {
    fn validate(&self) -> Result<(), String> {
        if self.enabled_layers.is_empty() {
            return Err("At least one decision layer must be enabled".to_string());
        }
        
        for layer in &self.enabled_layers {
            if !self.time_horizons.contains_key(layer) {
                return Err(format!("Missing time horizon for layer {:?}", layer));
            }
            
            if !self.priorities.contains_key(layer) {
                return Err(format!("Missing priority for layer {:?}", layer));
            }
        }
        
        Ok(())
    }
}

/// Panarchy framework configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyConfig {
    /// Enabled adaptive cycle phases
    pub enabled_phases: Vec<AdaptiveCyclePhase>,
    
    /// Phase transition thresholds
    pub transition_thresholds: HashMap<AdaptiveCyclePhase, f64>,
    
    /// Cross-scale interaction enabled
    pub cross_scale_interaction: bool,
    
    /// Resilience monitoring enabled
    pub resilience_monitoring: bool,
    
    /// Emergence detection enabled
    pub emergence_detection: bool,
    
    /// Transformation tracking enabled
    pub transformation_tracking: bool,
    
    /// Panarchy cycle duration
    pub cycle_duration: Duration,
    
    /// Number of scales to track
    pub scale_count: usize,
}

impl Default for PanarchyConfig {
    fn default() -> Self {
        let mut transition_thresholds = HashMap::new();
        transition_thresholds.insert(AdaptiveCyclePhase::Growth, 0.8);
        transition_thresholds.insert(AdaptiveCyclePhase::Conservation, 0.9);
        transition_thresholds.insert(AdaptiveCyclePhase::Release, 0.7);
        transition_thresholds.insert(AdaptiveCyclePhase::Reorganization, 0.6);
        
        Self {
            enabled_phases: AdaptiveCyclePhase::all_phases(),
            transition_thresholds,
            cross_scale_interaction: true,
            resilience_monitoring: true,
            emergence_detection: true,
            transformation_tracking: true,
            cycle_duration: Duration::from_secs(3600), // 1 hour
            scale_count: 4,
        }
    }
}

impl PanarchyConfig {
    fn validate(&self) -> Result<(), String> {
        if self.enabled_phases.is_empty() {
            return Err("At least one adaptive cycle phase must be enabled".to_string());
        }
        
        if self.scale_count == 0 {
            return Err("Scale count must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

/// Decision engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionEngineConfig {
    /// Multi-criteria analysis enabled
    pub multi_criteria_analysis: bool,
    
    /// Decision tree depth limit
    pub max_tree_depth: usize,
    
    /// Maximum alternatives to consider
    pub max_alternatives: usize,
    
    /// Uncertainty quantification enabled
    pub uncertainty_quantification: bool,
    
    /// Risk assessment enabled
    pub risk_assessment: bool,
    
    /// Learning from outcomes enabled
    pub outcome_learning: bool,
    
    /// Optimization algorithms to use
    pub optimization_algorithms: Vec<String>,
    
    /// Default decision timeout
    pub decision_timeout: Duration,
}

impl Default for DecisionEngineConfig {
    fn default() -> Self {
        Self {
            multi_criteria_analysis: true,
            max_tree_depth: 10,
            max_alternatives: 20,
            uncertainty_quantification: true,
            risk_assessment: true,
            outcome_learning: true,
            optimization_algorithms: vec![
                "genetic".to_string(),
                "particle_swarm".to_string(),
                "simulated_annealing".to_string(),
            ],
            decision_timeout: Duration::from_secs(30),
        }
    }
}

impl DecisionEngineConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_tree_depth == 0 {
            return Err("Tree depth must be greater than zero".to_string());
        }
        
        if self.max_alternatives == 0 {
            return Err("Max alternatives must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

/// Integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntegrationConfig {
    /// Swarm algorithm integration enabled
    pub swarm_integration: bool,
    
    /// Quantum agent bridge enabled
    pub quantum_agent_bridge: bool,
    
    /// CDFA coordination enabled
    pub cdfa_coordination: bool,
    
    /// Performance feedback loops enabled
    pub performance_feedback: bool,
    
    /// External system connectors
    pub external_connectors: Vec<String>,
    
    /// Integration timeout
    pub integration_timeout: Duration,
    
    /// Maximum concurrent integrations
    pub max_concurrent_integrations: usize,
}

impl Default for IntegrationConfig {
    fn default() -> Self {
        Self {
            swarm_integration: true,
            quantum_agent_bridge: true,
            cdfa_coordination: true,
            performance_feedback: true,
            external_connectors: vec![
                "nautilus_trader".to_string(),
                "risk_engine".to_string(),
                "portfolio_manager".to_string(),
            ],
            integration_timeout: Duration::from_secs(60),
            max_concurrent_integrations: 5,
        }
    }
}

impl IntegrationConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_concurrent_integrations == 0 {
            return Err("Max concurrent integrations must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

/// Governance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GovernanceConfig {
    /// Autonomous governance enabled
    pub autonomous_governance: bool,
    
    /// Policy engine enabled
    pub policy_engine: bool,
    
    /// Compliance monitoring enabled
    pub compliance_monitoring: bool,
    
    /// Security management enabled
    pub security_management: bool,
    
    /// Audit logging enabled
    pub audit_logging: bool,
    
    /// Policy validation interval
    pub policy_validation_interval: Duration,
    
    /// Compliance check interval
    pub compliance_check_interval: Duration,
}

impl Default for GovernanceConfig {
    fn default() -> Self {
        Self {
            autonomous_governance: true,
            policy_engine: true,
            compliance_monitoring: true,
            security_management: true,
            audit_logging: true,
            policy_validation_interval: Duration::from_secs(300), // 5 minutes
            compliance_check_interval: Duration::from_secs(600),  // 10 minutes
        }
    }
}

impl GovernanceConfig {
    fn validate(&self) -> Result<(), String> {
        // All governance features are optional, so no strict validation needed
        Ok(())
    }
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Real-time monitoring enabled
    pub real_time_monitoring: bool,
    
    /// Performance analysis enabled
    pub performance_analysis: bool,
    
    /// Adaptation tracking enabled
    pub adaptation_tracking: bool,
    
    /// System health monitoring enabled
    pub system_health_monitoring: bool,
    
    /// Metrics collection enabled
    pub metrics_collection: bool,
    
    /// Monitoring interval
    pub monitoring_interval: Duration,
    
    /// Metrics retention period
    pub metrics_retention: Duration,
    
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("cpu_usage".to_string(), 0.8);
        alert_thresholds.insert("memory_usage".to_string(), 0.85);
        alert_thresholds.insert("decision_latency".to_string(), 1000.0); // ms
        alert_thresholds.insert("error_rate".to_string(), 0.05);
        
        Self {
            real_time_monitoring: true,
            performance_analysis: true,
            adaptation_tracking: true,
            system_health_monitoring: true,
            metrics_collection: true,
            monitoring_interval: Duration::from_secs(1),
            metrics_retention: Duration::from_secs(86400 * 7), // 7 days
            alert_thresholds,
        }
    }
}

impl MonitoringConfig {
    fn validate(&self) -> Result<(), String> {
        if self.monitoring_interval.is_zero() {
            return Err("Monitoring interval must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Maximum CPU usage target (0.0 to 1.0)
    pub max_cpu_usage: f64,
    
    /// Maximum memory usage target (0.0 to 1.0)
    pub max_memory_usage: f64,
    
    /// Target decision latency (milliseconds)
    pub target_latency: f64,
    
    /// Target throughput (decisions per second)
    pub target_throughput: f64,
    
    /// Thread pool size
    pub thread_pool_size: usize,
    
    /// Async runtime configuration
    pub async_runtime: AsyncRuntimeConfig,
    
    /// Caching configuration
    pub caching: CachingConfig,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            max_cpu_usage: 0.8,
            max_memory_usage: 0.85,
            target_latency: 100.0, // 100ms
            target_throughput: 1000.0, // 1000 decisions/sec
            thread_pool_size: num_cpus::get(),
            async_runtime: AsyncRuntimeConfig::default(),
            caching: CachingConfig::default(),
        }
    }
}

impl PerformanceConfig {
    fn validate(&self) -> Result<(), String> {
        if !(0.0..=1.0).contains(&self.max_cpu_usage) {
            return Err("Max CPU usage must be between 0.0 and 1.0".to_string());
        }
        
        if !(0.0..=1.0).contains(&self.max_memory_usage) {
            return Err("Max memory usage must be between 0.0 and 1.0".to_string());
        }
        
        if self.thread_pool_size == 0 {
            return Err("Thread pool size must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

/// Async runtime configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AsyncRuntimeConfig {
    /// Worker thread count
    pub worker_threads: usize,
    
    /// Blocking thread count
    pub blocking_threads: usize,
    
    /// Stack size for threads
    pub stack_size: usize,
    
    /// Thread naming prefix
    pub thread_name_prefix: String,
}

impl Default for AsyncRuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: num_cpus::get(),
            blocking_threads: 32,
            stack_size: 2 * 1024 * 1024, // 2MB
            thread_name_prefix: "pads-".to_string(),
        }
    }
}

/// Caching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachingConfig {
    /// Cache enabled
    pub enabled: bool,
    
    /// Maximum cache size (entries)
    pub max_size: usize,
    
    /// Cache TTL (time to live)
    pub ttl: Duration,
    
    /// Cache cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for CachingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size: 10000,
            ttl: Duration::from_secs(300), // 5 minutes
            cleanup_interval: Duration::from_secs(60), // 1 minute
        }
    }
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Authentication enabled
    pub authentication: bool,
    
    /// Authorization enabled
    pub authorization: bool,
    
    /// Encryption enabled
    pub encryption: bool,
    
    /// Audit logging enabled
    pub audit_logging: bool,
    
    /// Rate limiting enabled
    pub rate_limiting: bool,
    
    /// Maximum requests per second
    pub max_requests_per_second: u32,
    
    /// Security scan interval
    pub security_scan_interval: Duration,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            authentication: true,
            authorization: true,
            encryption: true,
            audit_logging: true,
            rate_limiting: true,
            max_requests_per_second: 1000,
            security_scan_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl SecurityConfig {
    fn validate(&self) -> Result<(), String> {
        if self.max_requests_per_second == 0 {
            return Err("Max requests per second must be greater than zero".to_string());
        }
        
        Ok(())
    }
}

/// Configuration builder for PADS
pub struct PadsConfigBuilder {
    config: PadsConfig,
}

impl PadsConfigBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: PadsConfig::default(),
        }
    }
    
    /// Set system ID
    pub fn with_system_id(mut self, system_id: String) -> Self {
        self.config.system_id = system_id;
        self
    }
    
    /// Set number of decision layers
    pub fn with_decision_layers(mut self, count: usize) -> Self {
        if count > 0 && count <= 4 {
            self.config.decision_layers.enabled_layers = DecisionLayer::all_layers()
                .into_iter()
                .take(count)
                .collect();
        }
        self
    }
    
    /// Enable adaptive cycles
    pub fn with_adaptive_cycles(mut self, enabled: bool) -> Self {
        if !enabled {
            self.config.panarchy.enabled_phases.clear();
        }
        self
    }
    
    /// Enable real-time monitoring
    pub fn with_real_time_monitoring(mut self, enabled: bool) -> Self {
        self.config.monitoring.real_time_monitoring = enabled;
        self
    }
    
    /// Set thread pool size
    pub fn with_thread_pool_size(mut self, size: usize) -> Self {
        self.config.performance.thread_pool_size = size;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> PadsConfig {
        self.config
    }
}

impl Default for PadsConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = PadsConfig::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_builder() {
        let config = PadsConfig::builder()
            .with_system_id("test-pads".to_string())
            .with_decision_layers(3)
            .with_adaptive_cycles(true)
            .with_real_time_monitoring(true)
            .with_thread_pool_size(8)
            .build();
        
        assert_eq!(config.system_id, "test-pads");
        assert_eq!(config.decision_layers.enabled_layers.len(), 3);
        assert!(config.monitoring.real_time_monitoring);
        assert_eq!(config.performance.thread_pool_size, 8);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = PadsConfig::default();
        config.decision_layers.enabled_layers.clear();
        
        assert!(config.validate().is_err());
    }
}