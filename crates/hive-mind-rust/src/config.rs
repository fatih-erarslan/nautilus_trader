//! Configuration management for the hive mind system

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::time::Duration;
use uuid::Uuid;
use crate::error::{ConfigError, Result};

/// Main configuration structure for the hive mind system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HiveMindConfig {
    /// Unique identifier for this hive mind instance
    pub instance_id: Uuid,
    
    /// Network configuration
    pub network: NetworkConfig,
    
    /// Consensus configuration
    pub consensus: ConsensusConfig,
    
    /// Memory management configuration
    pub memory: MemoryConfig,
    
    /// Neural processing configuration
    pub neural: NeuralConfig,
    
    /// Agent management configuration
    pub agents: AgentConfig,
    
    /// Metrics and monitoring configuration
    pub metrics: MetricsConfig,
    
    /// Security configuration
    pub security: SecurityConfig,
}

/// Network configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    /// Local listening address
    pub listen_addr: String,
    
    /// Port for P2P communication
    pub p2p_port: u16,
    
    /// Port for API server
    pub api_port: u16,
    
    /// Maximum number of peer connections
    pub max_peers: usize,
    
    /// Connection timeout duration
    pub connection_timeout: Duration,
    
    /// Message retry attempts
    pub retry_attempts: u32,
    
    /// Enable discovery protocols
    pub enable_discovery: bool,
    
    /// Bootstrap peers for initial connection
    pub bootstrap_peers: Vec<String>,
}

/// Consensus algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsensusConfig {
    /// Consensus algorithm type
    pub algorithm: ConsensusAlgorithm,
    
    /// Minimum number of nodes for consensus
    pub min_nodes: usize,
    
    /// Consensus timeout duration
    pub timeout: Duration,
    
    /// Byzantine fault tolerance threshold (fraction of malicious nodes)
    pub byzantine_threshold: f64,
    
    /// Leader election timeout
    pub leader_election_timeout: Duration,
    
    /// Heartbeat interval for leader
    pub heartbeat_interval: Duration,
}

/// Available consensus algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConsensusAlgorithm {
    /// Raft consensus algorithm
    Raft,
    /// Practical Byzantine Fault Tolerance
    Pbft,
    /// Gossip-based consensus
    Gossip,
    /// Hybrid consensus combining multiple algorithms
    Hybrid,
}

/// Memory management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory pool size in bytes
    pub max_pool_size: usize,
    
    /// Memory cleanup interval
    pub cleanup_interval: Duration,
    
    /// Knowledge graph configuration
    pub knowledge_graph: KnowledgeGraphConfig,
    
    /// Memory replication factor
    pub replication_factor: usize,
    
    /// Enable memory compression
    pub enable_compression: bool,
    
    /// Persistence configuration
    pub persistence: PersistenceConfig,
}

/// Knowledge graph configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KnowledgeGraphConfig {
    /// Maximum number of nodes in the graph
    pub max_nodes: usize,
    
    /// Maximum number of edges per node
    pub max_edges_per_node: usize,
    
    /// Graph traversal timeout
    pub traversal_timeout: Duration,
    
    /// Enable semantic similarity calculations
    pub enable_similarity: bool,
}

/// Persistence configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistenceConfig {
    /// Database type
    pub database_type: DatabaseType,
    
    /// Database connection string
    pub connection_string: String,
    
    /// Maximum number of database connections
    pub max_connections: u32,
    
    /// Connection timeout
    pub connection_timeout: Duration,
    
    /// Enable automatic backups
    pub enable_backups: bool,
    
    /// Backup interval
    pub backup_interval: Duration,
}

/// Supported database types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatabaseType {
    Sqlite,
    PostgreSql,
    Sled,
    InMemory,
}

/// Neural processing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    /// Enable neural pattern recognition
    pub enable_pattern_recognition: bool,
    
    /// Neural network model configuration
    pub model: NeuralModelConfig,
    
    /// Training configuration
    pub training: TrainingConfig,
    
    /// Inference configuration
    pub inference: InferenceConfig,
}

/// Neural model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralModelConfig {
    /// Model architecture type
    pub architecture: NeuralArchitecture,
    
    /// Input dimension
    pub input_dim: usize,
    
    /// Hidden layer dimensions
    pub hidden_dims: Vec<usize>,
    
    /// Output dimension
    pub output_dim: usize,
    
    /// Activation function
    pub activation: ActivationFunction,
}

/// Neural network architectures
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NeuralArchitecture {
    FeedForward,
    Transformer,
    Lstm,
    Gru,
    Cnn,
}

/// Activation functions
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationFunction {
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Swish,
}

/// Training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// Batch size
    pub batch_size: usize,
    
    /// Maximum number of epochs
    pub max_epochs: usize,
    
    /// Early stopping patience
    pub patience: usize,
    
    /// Enable distributed training
    pub enable_distributed: bool,
}

/// Inference configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Batch size for inference
    pub batch_size: usize,
    
    /// Inference timeout
    pub timeout: Duration,
    
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    
    /// Model cache size
    pub cache_size: usize,
}

/// Agent management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    
    /// Agent spawning strategy
    pub spawning_strategy: SpawningStrategy,
    
    /// Agent heartbeat interval
    pub heartbeat_interval: Duration,
    
    /// Agent timeout duration
    pub agent_timeout: Duration,
    
    /// Load balancing configuration
    pub load_balancing: LoadBalancingConfig,
}

/// Agent spawning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpawningStrategy {
    /// Spawn agents on demand
    OnDemand,
    /// Pre-spawn a pool of agents
    PreSpawn,
    /// Adaptive spawning based on load
    Adaptive,
}

/// Load balancing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadBalancingConfig {
    /// Load balancing algorithm
    pub algorithm: LoadBalancingAlgorithm,
    
    /// Health check interval
    pub health_check_interval: Duration,
    
    /// Maximum load per agent
    pub max_load_per_agent: f64,
}

/// Load balancing algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LoadBalancingAlgorithm {
    RoundRobin,
    LeastConnections,
    WeightedRandom,
    ConsistentHash,
}

/// Metrics and monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    /// Enable metrics collection
    pub enabled: bool,
    
    /// Metrics collection interval
    pub collection_interval: Duration,
    
    /// Prometheus metrics port
    pub prometheus_port: u16,
    
    /// Enable performance profiling
    pub enable_profiling: bool,
    
    /// Metrics retention period
    pub retention_period: Duration,
}

/// Security configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable encryption for all communications
    pub enable_encryption: bool,
    
    /// Encryption algorithm
    pub encryption_algorithm: EncryptionAlgorithm,
    
    /// Key rotation interval
    pub key_rotation_interval: Duration,
    
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
}

/// Encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EncryptionAlgorithm {
    ChaCha20Poly1305,
    Aes256Gcm,
    XChaCha20Poly1305,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Authentication method
    pub method: AuthenticationMethod,
    
    /// Token expiration time
    pub token_expiration: Duration,
    
    /// Enable mutual authentication
    pub enable_mutual_auth: bool,
}

/// Authentication methods
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthenticationMethod {
    Ed25519,
    X25519,
    Secp256k1,
    Certificate,
}

impl Default for HiveMindConfig {
    fn default() -> Self {
        Self {
            instance_id: Uuid::new_v4(),
            network: NetworkConfig::default(),
            consensus: ConsensusConfig::default(),
            memory: MemoryConfig::default(),
            neural: NeuralConfig::default(),
            agents: AgentConfig::default(),
            metrics: MetricsConfig::default(),
            security: SecurityConfig::default(),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            p2p_port: 8080,
            api_port: 8090,
            max_peers: 50,
            connection_timeout: Duration::from_secs(30),
            retry_attempts: 3,
            enable_discovery: true,
            bootstrap_peers: Vec::new(),
        }
    }
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            algorithm: ConsensusAlgorithm::Raft,
            min_nodes: 3,
            timeout: Duration::from_secs(10),
            byzantine_threshold: 0.33,
            leader_election_timeout: Duration::from_secs(5),
            heartbeat_interval: Duration::from_secs(1),
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1024 * 1024 * 1024, // 1GB
            cleanup_interval: Duration::from_secs(300), // 5 minutes
            knowledge_graph: KnowledgeGraphConfig::default(),
            replication_factor: 3,
            enable_compression: true,
            persistence: PersistenceConfig::default(),
        }
    }
}

impl Default for KnowledgeGraphConfig {
    fn default() -> Self {
        Self {
            max_nodes: 100_000,
            max_edges_per_node: 1000,
            traversal_timeout: Duration::from_secs(5),
            enable_similarity: true,
        }
    }
}

impl Default for PersistenceConfig {
    fn default() -> Self {
        Self {
            database_type: DatabaseType::Sqlite,
            connection_string: "sqlite://hive_mind.db".to_string(),
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            enable_backups: true,
            backup_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            enable_pattern_recognition: true,
            model: NeuralModelConfig::default(),
            training: TrainingConfig::default(),
            inference: InferenceConfig::default(),
        }
    }
}

impl Default for NeuralModelConfig {
    fn default() -> Self {
        Self {
            architecture: NeuralArchitecture::Transformer,
            input_dim: 512,
            hidden_dims: vec![1024, 512, 256],
            output_dim: 128,
            activation: ActivationFunction::Gelu,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            batch_size: 32,
            max_epochs: 100,
            patience: 10,
            enable_distributed: true,
        }
    }
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 16,
            timeout: Duration::from_secs(5),
            enable_gpu: false, // Default to CPU
            cache_size: 1000,
        }
    }
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_agents: 100,
            spawning_strategy: SpawningStrategy::Adaptive,
            heartbeat_interval: Duration::from_secs(30),
            agent_timeout: Duration::from_secs(300),
            load_balancing: LoadBalancingConfig::default(),
        }
    }
}

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            algorithm: LoadBalancingAlgorithm::LeastConnections,
            health_check_interval: Duration::from_secs(10),
            max_load_per_agent: 0.8,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_secs(60),
            prometheus_port: 9090,
            enable_profiling: false,
            retention_period: Duration::from_secs(86400 * 7), // 7 days
        }
    }
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            enable_encryption: true,
            encryption_algorithm: EncryptionAlgorithm::ChaCha20Poly1305,
            key_rotation_interval: Duration::from_secs(86400), // 24 hours
            authentication: AuthenticationConfig::default(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            method: AuthenticationMethod::Ed25519,
            token_expiration: Duration::from_secs(3600), // 1 hour
            enable_mutual_auth: true,
        }
    }
}

impl HiveMindConfig {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(&path).map_err(|_| {
            ConfigError::FileNotFound {
                path: path.as_ref().display().to_string(),
            }
        })?;
        
        toml::from_str(&content).map_err(|e| {
            HiveMindError::Config(ConfigError::ParseFailed {
                reason: e.to_string(),
            })
        })
    }
    
    /// Save configuration to file
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self).map_err(|e| {
            HiveMindError::Config(ConfigError::ParseFailed {
                reason: e.to_string(),
            })
        })?;
        
        std::fs::write(path, content)?;
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate network configuration
        if self.network.max_peers == 0 {
            return Err(ConfigError::InvalidParameter {
                parameter: "network.max_peers must be > 0".to_string(),
            }.into());
        }
        
        // Validate consensus configuration
        if self.consensus.min_nodes < 1 {
            return Err(ConfigError::InvalidParameter {
                parameter: "consensus.min_nodes must be >= 1".to_string(),
            }.into());
        }
        
        if !(0.0..=0.5).contains(&self.consensus.byzantine_threshold) {
            return Err(ConfigError::InvalidParameter {
                parameter: "consensus.byzantine_threshold must be between 0.0 and 0.5".to_string(),
            }.into());
        }
        
        // Validate agent configuration
        if self.agents.max_agents == 0 {
            return Err(ConfigError::InvalidParameter {
                parameter: "agents.max_agents must be > 0".to_string(),
            }.into());
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = HiveMindConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = HiveMindConfig::default();
        let serialized = toml::to_string(&config).expect("Config serialization should never fail for valid config");
        let deserialized: HiveMindConfig = toml::from_str(&serialized).expect("Config deserialization should never fail for valid serialized config");
        assert_eq!(config.network.p2p_port, deserialized.network.p2p_port);
    }

    #[test]
    fn test_config_file_operations() {
        let config = HiveMindConfig::default();
        let temp_file = NamedTempFile::new().expect("Should be able to create temporary file for tests");
        
        // Save config
        config.save_to_file(temp_file.path()).expect("Should be able to save config to temporary file");
        
        // Load config
        let loaded_config = HiveMindConfig::load_from_file(temp_file.path()).expect("Should be able to load config from temporary file");
        assert_eq!(config.network.p2p_port, loaded_config.network.p2p_port);
    }

    #[test]
    fn test_config_validation() {
        let mut config = HiveMindConfig::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid max_peers should fail
        config.network.max_peers = 0;
        assert!(config.validate().is_err());
    }
}