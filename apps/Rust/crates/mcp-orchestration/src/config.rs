//! Configuration management for MCP orchestration system.

use crate::types::{LoadBalancingStrategy, RecoveryStrategy};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::time::Duration;

/// Main orchestration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationConfig {
    /// Agent configuration
    pub agents: AgentConfig,
    /// Communication configuration
    pub communication: CommunicationConfig,
    /// Task queue configuration
    pub task_queue: TaskQueueConfig,
    /// Load balancer configuration
    pub load_balancer: LoadBalancerConfig,
    /// Memory configuration
    pub memory: MemoryConfig,
    /// Health monitoring configuration
    pub health: HealthConfig,
    /// Recovery configuration
    pub recovery: RecoveryConfig,
    /// Metrics configuration
    pub metrics: MetricsConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
}

impl Default for OrchestrationConfig {
    fn default() -> Self {
        Self {
            agents: AgentConfig::default(),
            communication: CommunicationConfig::default(),
            task_queue: TaskQueueConfig::default(),
            load_balancer: LoadBalancerConfig::default(),
            memory: MemoryConfig::default(),
            health: HealthConfig::default(),
            recovery: RecoveryConfig::default(),
            metrics: MetricsConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl OrchestrationConfig {
    /// Load configuration from file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name(path.as_ref().to_str().unwrap()))
            .build()?;
        
        settings.try_deserialize()
    }
    
    /// Load configuration from multiple sources
    pub fn from_sources() -> Result<Self, config::ConfigError> {
        let settings = config::Config::builder()
            .add_source(config::File::with_name("config/default").required(false))
            .add_source(config::File::with_name("config/local").required(false))
            .add_source(config::Environment::with_prefix("MCP"))
            .build()?;
        
        settings.try_deserialize()
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        // Validate agent configuration
        if self.agents.max_agents == 0 {
            return Err("max_agents must be greater than 0".to_string());
        }
        
        if self.agents.max_agents > 100 {
            return Err("max_agents cannot exceed 100".to_string());
        }
        
        // Validate communication configuration
        if self.communication.max_message_size == 0 {
            return Err("max_message_size must be greater than 0".to_string());
        }
        
        if self.communication.message_timeout_ms == 0 {
            return Err("message_timeout_ms must be greater than 0".to_string());
        }
        
        // Validate task queue configuration
        if self.task_queue.max_queue_size == 0 {
            return Err("max_queue_size must be greater than 0".to_string());
        }
        
        // Validate memory configuration
        if self.memory.max_memory_size == 0 {
            return Err("max_memory_size must be greater than 0".to_string());
        }
        
        if self.memory.cache_size == 0 {
            return Err("cache_size must be greater than 0".to_string());
        }
        
        Ok(())
    }
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of agents in the swarm
    pub max_agents: usize,
    /// Agent heartbeat interval in milliseconds
    pub heartbeat_interval_ms: u64,
    /// Agent heartbeat timeout in milliseconds
    pub heartbeat_timeout_ms: u64,
    /// Agent startup timeout in milliseconds
    pub startup_timeout_ms: u64,
    /// Agent shutdown timeout in milliseconds
    pub shutdown_timeout_ms: u64,
    /// Enable agent auto-restart
    pub auto_restart: bool,
    /// Maximum restart attempts
    pub max_restart_attempts: u32,
    /// Agent resource limits
    pub resource_limits: HashMap<String, String>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_agents: 20,
            heartbeat_interval_ms: 30000,  // 30 seconds
            heartbeat_timeout_ms: 90000,   // 90 seconds
            startup_timeout_ms: 60000,     // 60 seconds
            shutdown_timeout_ms: 30000,    // 30 seconds
            auto_restart: true,
            max_restart_attempts: 3,
            resource_limits: HashMap::new(),
        }
    }
}

/// Communication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// Maximum message size in bytes
    pub max_message_size: usize,
    /// Message timeout in milliseconds
    pub message_timeout_ms: u64,
    /// Maximum retry attempts for failed messages
    pub max_retry_attempts: u32,
    /// Retry backoff multiplier
    pub retry_backoff_multiplier: f64,
    /// Enable message compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: String,
    /// Enable message encryption
    pub enable_encryption: bool,
    /// Message queue buffer size
    pub queue_buffer_size: usize,
    /// Enable broadcast optimization
    pub enable_broadcast_optimization: bool,
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            max_message_size: 16 * 1024 * 1024, // 16MB
            message_timeout_ms: 5000,           // 5 seconds
            max_retry_attempts: 3,
            retry_backoff_multiplier: 2.0,
            enable_compression: true,
            compression_algorithm: "lz4".to_string(),
            enable_encryption: false,
            queue_buffer_size: 1000,
            enable_broadcast_optimization: true,
        }
    }
}

/// Task queue configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQueueConfig {
    /// Maximum queue size
    pub max_queue_size: usize,
    /// Task timeout in milliseconds
    pub task_timeout_ms: u64,
    /// Maximum task retry attempts
    pub max_retry_attempts: u32,
    /// Task priority levels
    pub priority_levels: u8,
    /// Enable task persistence
    pub enable_persistence: bool,
    /// Persistence storage path
    pub persistence_path: String,
    /// Task cleanup interval in milliseconds
    pub cleanup_interval_ms: u64,
    /// Enable task metrics collection
    pub enable_metrics: bool,
}

impl Default for TaskQueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            task_timeout_ms: 30000,         // 30 seconds
            max_retry_attempts: 3,
            priority_levels: 4,
            enable_persistence: false,
            persistence_path: "./tasks".to_string(),
            cleanup_interval_ms: 60000,     // 60 seconds
            enable_metrics: true,
        }
    }
}

/// Load balancer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancerConfig {
    /// Load balancing strategy
    pub strategy: LoadBalancingStrategy,
    /// Enable adaptive strategy selection
    pub enable_adaptive: bool,
    /// Strategy adaptation interval in milliseconds
    pub adaptation_interval_ms: u64,
    /// Load threshold for overloaded agents
    pub overload_threshold: f64,
    /// Enable consistent hashing
    pub enable_consistent_hashing: bool,
    /// Number of virtual nodes for consistent hashing
    pub virtual_nodes: usize,
    /// Enable performance-based weighting
    pub enable_performance_weighting: bool,
    /// Performance history size
    pub performance_history_size: usize,
}

impl Default for LoadBalancerConfig {
    fn default() -> Self {
        Self {
            strategy: LoadBalancingStrategy::PerformanceBased,
            enable_adaptive: true,
            adaptation_interval_ms: 60000,  // 60 seconds
            overload_threshold: 0.8,
            enable_consistent_hashing: true,
            virtual_nodes: 100,
            enable_performance_weighting: true,
            performance_history_size: 100,
        }
    }
}

/// Memory configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory size in bytes
    pub max_memory_size: u64,
    /// Cache size for frequently accessed regions
    pub cache_size: usize,
    /// Memory region expiration check interval in milliseconds
    pub expiration_check_interval_ms: u64,
    /// Enable memory compression
    pub enable_compression: bool,
    /// Memory compression algorithm
    pub compression_algorithm: String,
    /// Enable memory persistence
    pub enable_persistence: bool,
    /// Persistence storage path
    pub persistence_path: String,
    /// Enable memory encryption
    pub enable_encryption: bool,
    /// Memory statistics update interval in milliseconds
    pub stats_update_interval_ms: u64,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_size: 1024 * 1024 * 1024, // 1GB
            cache_size: 1000,
            expiration_check_interval_ms: 30000,  // 30 seconds
            enable_compression: true,
            compression_algorithm: "lz4".to_string(),
            enable_persistence: false,
            persistence_path: "./memory".to_string(),
            enable_encryption: false,
            stats_update_interval_ms: 10000,      // 10 seconds
        }
    }
}

/// Health monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthConfig {
    /// Health check interval in milliseconds
    pub check_interval_ms: u64,
    /// Health check timeout in milliseconds
    pub check_timeout_ms: u64,
    /// Enable system metrics collection
    pub enable_system_metrics: bool,
    /// System metrics collection interval in milliseconds
    pub system_metrics_interval_ms: u64,
    /// Health history size
    pub health_history_size: usize,
    /// Enable health alerting
    pub enable_alerting: bool,
    /// Alert thresholds
    pub alert_thresholds: HashMap<String, f64>,
    /// Status aggregation interval in milliseconds
    pub status_aggregation_interval_ms: u64,
}

impl Default for HealthConfig {
    fn default() -> Self {
        let mut alert_thresholds = HashMap::new();
        alert_thresholds.insert("cpu_usage".to_string(), 80.0);
        alert_thresholds.insert("memory_usage".to_string(), 85.0);
        alert_thresholds.insert("disk_usage".to_string(), 90.0);
        
        Self {
            check_interval_ms: 30000,        // 30 seconds
            check_timeout_ms: 5000,          // 5 seconds
            enable_system_metrics: true,
            system_metrics_interval_ms: 5000, // 5 seconds
            health_history_size: 100,
            enable_alerting: true,
            alert_thresholds,
            status_aggregation_interval_ms: 15000, // 15 seconds
        }
    }
}

/// Recovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryConfig {
    /// Default recovery strategy
    pub default_strategy: RecoveryStrategy,
    /// Enable automatic recovery
    pub enable_auto_recovery: bool,
    /// Recovery attempt interval in milliseconds
    pub recovery_interval_ms: u64,
    /// Maximum recovery attempts
    pub max_recovery_attempts: u32,
    /// Circuit breaker failure threshold
    pub circuit_breaker_threshold: u32,
    /// Circuit breaker timeout in milliseconds
    pub circuit_breaker_timeout_ms: u64,
    /// Enable graceful degradation
    pub enable_graceful_degradation: bool,
    /// Recovery strategies by component
    pub component_strategies: HashMap<String, RecoveryStrategy>,
}

impl Default for RecoveryConfig {
    fn default() -> Self {
        Self {
            default_strategy: RecoveryStrategy::Restart,
            enable_auto_recovery: true,
            recovery_interval_ms: 10000,     // 10 seconds
            max_recovery_attempts: 3,
            circuit_breaker_threshold: 5,
            circuit_breaker_timeout_ms: 60000, // 60 seconds
            enable_graceful_degradation: true,
            component_strategies: HashMap::new(),
        }
    }
}

/// Metrics configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enable_metrics: bool,
    /// Metrics collection interval in milliseconds
    pub collection_interval_ms: u64,
    /// Metrics retention period in seconds
    pub retention_period_seconds: u64,
    /// Enable metrics export
    pub enable_export: bool,
    /// Metrics export format
    pub export_format: String,
    /// Metrics export endpoint
    pub export_endpoint: String,
    /// Enable Prometheus metrics
    pub enable_prometheus: bool,
    /// Prometheus metrics port
    pub prometheus_port: u16,
    /// Custom metrics configuration
    pub custom_metrics: HashMap<String, String>,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enable_metrics: true,
            collection_interval_ms: 10000,   // 10 seconds
            retention_period_seconds: 86400, // 24 hours
            enable_export: false,
            export_format: "json".to_string(),
            export_endpoint: "http://localhost:9090".to_string(),
            enable_prometheus: true,
            prometheus_port: 9090,
            custom_metrics: HashMap::new(),
        }
    }
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    /// Enable NUMA optimization
    pub enable_numa: bool,
    /// Thread pool size
    pub thread_pool_size: usize,
    /// Enable lockfree data structures
    pub enable_lockfree: bool,
    /// Memory pool size
    pub memory_pool_size: usize,
    /// Enable batch processing
    pub enable_batch_processing: bool,
    /// Batch size for operations
    pub batch_size: usize,
    /// Enable performance profiling
    pub enable_profiling: bool,
    /// Profiling output path
    pub profiling_output_path: String,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_simd: true,
            enable_numa: true,
            thread_pool_size: num_cpus::get(),
            enable_lockfree: true,
            memory_pool_size: 1000,
            enable_batch_processing: true,
            batch_size: 100,
            enable_profiling: false,
            profiling_output_path: "./profiling".to_string(),
        }
    }
}

/// Configuration utilities
impl OrchestrationConfig {
    /// Get agent heartbeat interval as Duration
    pub fn agent_heartbeat_interval(&self) -> Duration {
        Duration::from_millis(self.agents.heartbeat_interval_ms)
    }
    
    /// Get agent heartbeat timeout as Duration
    pub fn agent_heartbeat_timeout(&self) -> Duration {
        Duration::from_millis(self.agents.heartbeat_timeout_ms)
    }
    
    /// Get message timeout as Duration
    pub fn message_timeout(&self) -> Duration {
        Duration::from_millis(self.communication.message_timeout_ms)
    }
    
    /// Get task timeout as Duration
    pub fn task_timeout(&self) -> Duration {
        Duration::from_millis(self.task_queue.task_timeout_ms)
    }
    
    /// Get health check interval as Duration
    pub fn health_check_interval(&self) -> Duration {
        Duration::from_millis(self.health.check_interval_ms)
    }
    
    /// Get health check timeout as Duration
    pub fn health_check_timeout(&self) -> Duration {
        Duration::from_millis(self.health.check_timeout_ms)
    }
    
    /// Get recovery interval as Duration
    pub fn recovery_interval(&self) -> Duration {
        Duration::from_millis(self.recovery.recovery_interval_ms)
    }
    
    /// Get metrics collection interval as Duration
    pub fn metrics_collection_interval(&self) -> Duration {
        Duration::from_millis(self.metrics.collection_interval_ms)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::NamedTempFile;
    
    #[test]
    fn test_default_config() {
        let config = OrchestrationConfig::default();
        assert!(config.validate().is_ok());
        
        assert_eq!(config.agents.max_agents, 20);
        assert_eq!(config.communication.max_message_size, 16 * 1024 * 1024);
        assert_eq!(config.task_queue.max_queue_size, 10000);
        assert_eq!(config.memory.cache_size, 1000);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = OrchestrationConfig::default();
        
        // Test invalid max_agents
        config.agents.max_agents = 0;
        assert!(config.validate().is_err());
        
        config.agents.max_agents = 101;
        assert!(config.validate().is_err());
        
        // Reset to valid value
        config.agents.max_agents = 20;
        assert!(config.validate().is_ok());
        
        // Test invalid message size
        config.communication.max_message_size = 0;
        assert!(config.validate().is_err());
        
        // Reset to valid value
        config.communication.max_message_size = 1024;
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_durations() {
        let config = OrchestrationConfig::default();
        
        assert_eq!(config.agent_heartbeat_interval(), Duration::from_millis(30000));
        assert_eq!(config.agent_heartbeat_timeout(), Duration::from_millis(90000));
        assert_eq!(config.message_timeout(), Duration::from_millis(5000));
        assert_eq!(config.task_timeout(), Duration::from_millis(30000));
        assert_eq!(config.health_check_interval(), Duration::from_millis(30000));
        assert_eq!(config.health_check_timeout(), Duration::from_millis(5000));
        assert_eq!(config.recovery_interval(), Duration::from_millis(10000));
        assert_eq!(config.metrics_collection_interval(), Duration::from_millis(10000));
    }
    
    #[test]
    fn test_config_serialization() {
        let config = OrchestrationConfig::default();
        
        // Test JSON serialization
        let json = serde_json::to_string_pretty(&config).unwrap();
        let deserialized: OrchestrationConfig = serde_json::from_str(&json).unwrap();
        
        assert_eq!(config.agents.max_agents, deserialized.agents.max_agents);
        assert_eq!(config.communication.max_message_size, deserialized.communication.max_message_size);
        
        // Test TOML serialization
        let toml = toml::to_string_pretty(&config).unwrap();
        let deserialized: OrchestrationConfig = toml::from_str(&toml).unwrap();
        
        assert_eq!(config.agents.max_agents, deserialized.agents.max_agents);
        assert_eq!(config.communication.max_message_size, deserialized.communication.max_message_size);
    }
    
    #[test]
    fn test_config_from_file() {
        let config = OrchestrationConfig::default();
        
        // Create temporary file
        let mut temp_file = NamedTempFile::new().unwrap();
        let toml_content = toml::to_string_pretty(&config).unwrap();
        fs::write(temp_file.path(), toml_content).unwrap();
        
        // Load from file
        let loaded_config = OrchestrationConfig::from_file(temp_file.path()).unwrap();
        
        assert_eq!(config.agents.max_agents, loaded_config.agents.max_agents);
        assert_eq!(config.communication.max_message_size, loaded_config.communication.max_message_size);
    }
}