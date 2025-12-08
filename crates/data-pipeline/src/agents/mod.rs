//! # Data Pipeline Agents
//!
//! Specialized ruv-swarm agents for high-frequency data processing and feature extraction.
//! Each agent is designed for ultra-low latency operation with sub-100μs response times.

pub mod base;
pub mod data_ingestion;
pub mod feature_engineering;
pub mod data_validation;
pub mod stream_processing;
pub mod data_transformation;
pub mod cache_management;
pub mod coordination;
pub mod registry;

// Re-export key types and traits
pub use base::{DataAgent, DataAgentBuilder, DataAgentInfo, DataAgentState, DataAgentType};
pub use data_ingestion::{DataIngestionAgent, DataIngestionConfig, MarketDataSource};
pub use feature_engineering::{FeatureEngineeringAgent, FeatureEngineeringConfig, QuantumFeatureExtractor};
pub use data_validation::{DataValidationAgent, DataValidationConfig, TengriValidator};
pub use stream_processing::{StreamProcessingAgent, StreamProcessingConfig, AdaptiveBuffer};
pub use data_transformation::{DataTransformationAgent, DataTransformationConfig, DataNormalizer};
pub use cache_management::{CacheManagementAgent, CacheManagementConfig, IntelligentCache};
pub use coordination::{DataSwarmCoordinator, DataSwarmState};
pub use registry::{DataAgentRegistry, DataAgentManager};

use std::sync::Arc;
use tokio::sync::RwLock;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error};

/// Data processing swarm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSwarmConfig {
    /// Maximum number of agents in the swarm
    pub max_agents: usize,
    /// Target latency for data processing (microseconds)
    pub target_latency_us: u64,
    /// Number of processing threads per agent
    pub threads_per_agent: usize,
    /// Enable quantum enhancement features
    pub quantum_enabled: bool,
    /// TENGRI integration settings
    pub tengri_config: TengriConfig,
    /// Performance optimization settings
    pub performance_config: PerformanceConfig,
}

impl Default for DataSwarmConfig {
    fn default() -> Self {
        Self {
            max_agents: 20,
            target_latency_us: 100,
            threads_per_agent: 4,
            quantum_enabled: true,
            tengri_config: TengriConfig::default(),
            performance_config: PerformanceConfig::default(),
        }
    }
}

/// TENGRI integration configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriConfig {
    /// Enable TENGRI oversight
    pub enabled: bool,
    /// TENGRI endpoint URL
    pub endpoint: String,
    /// Authentication token
    pub auth_token: String,
    /// Data integrity validation level
    pub validation_level: ValidationLevel,
}

impl Default for TengriConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            endpoint: "http://localhost:8080".to_string(),
            auth_token: "default_token".to_string(),
            validation_level: ValidationLevel::Strict,
        }
    }
}

/// Data validation levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ValidationLevel {
    /// Minimal validation for maximum speed
    Fast,
    /// Standard validation with good performance
    Standard,
    /// Strict validation with comprehensive checks
    Strict,
    /// Paranoid validation with maximum security
    Paranoid,
}

/// Performance optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable SIMD optimizations
    pub simd_enabled: bool,
    /// Memory pool size (MB)
    pub memory_pool_size_mb: usize,
    /// CPU affinity settings
    pub cpu_affinity: Vec<usize>,
    /// Enable lock-free data structures
    pub lock_free: bool,
    /// Prefetch settings
    pub prefetch_enabled: bool,
    /// Cache line optimization
    pub cache_line_optimized: bool,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            simd_enabled: true,
            memory_pool_size_mb: 1024,
            cpu_affinity: vec![0, 1, 2, 3],
            lock_free: true,
            prefetch_enabled: true,
            cache_line_optimized: true,
        }
    }
}

/// Main data processing swarm
pub struct DataProcessingSwarm {
    config: Arc<DataSwarmConfig>,
    coordinator: Arc<DataSwarmCoordinator>,
    registry: Arc<DataAgentRegistry>,
    state: Arc<RwLock<DataSwarmState>>,
}

impl DataProcessingSwarm {
    /// Create a new data processing swarm
    pub async fn new(config: DataSwarmConfig) -> Result<Self> {
        info!("Initializing data processing swarm with config: {:?}", config);
        
        let config = Arc::new(config);
        let coordinator = Arc::new(DataSwarmCoordinator::new(config.clone()).await?);
        let registry = Arc::new(DataAgentRegistry::new(config.clone())?);
        let state = Arc::new(RwLock::new(DataSwarmState::default()));
        
        Ok(Self {
            config,
            coordinator,
            registry,
            state,
        })
    }
    
    /// Deploy all data processing agents
    pub async fn deploy(&self) -> Result<()> {
        info!("Deploying data processing swarm agents");
        
        // Deploy Data Ingestion Agent
        let ingestion_agent = DataIngestionAgent::new(
            DataIngestionConfig::default()
        ).await?;
        self.registry.register_agent(Box::new(ingestion_agent)).await?;
        
        // Deploy Feature Engineering Agent
        let feature_agent = FeatureEngineeringAgent::new(
            FeatureEngineeringConfig::default()
        ).await?;
        self.registry.register_agent(Box::new(feature_agent)).await?;
        
        // Deploy Data Validation Agent
        let validation_agent = DataValidationAgent::new(
            DataValidationConfig::default()
        ).await?;
        self.registry.register_agent(Box::new(validation_agent)).await?;
        
        // Deploy Stream Processing Agent
        let stream_agent = StreamProcessingAgent::new(
            StreamProcessingConfig::default()
        ).await?;
        self.registry.register_agent(Box::new(stream_agent)).await?;
        
        // Deploy Data Transformation Agent
        let transform_agent = DataTransformationAgent::new(
            DataTransformationConfig::default()
        ).await?;
        self.registry.register_agent(Box::new(transform_agent)).await?;
        
        // Deploy Cache Management Agent
        let cache_agent = CacheManagementAgent::new(
            CacheManagementConfig::default()
        ).await?;
        self.registry.register_agent(Box::new(cache_agent)).await?;
        
        // Start coordination
        self.coordinator.start().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.agents_deployed = 6;
            state.swarm_active = true;
        }
        
        info!("Data processing swarm deployed successfully");
        Ok(())
    }
    
    /// Start the swarm
    pub async fn start(&self) -> Result<()> {
        info!("Starting data processing swarm");
        
        // Start all agents
        self.registry.start_all_agents().await?;
        
        // Start coordinator
        self.coordinator.start().await?;
        
        info!("Data processing swarm started successfully");
        Ok(())
    }
    
    /// Stop the swarm
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping data processing swarm");
        
        // Stop coordinator
        self.coordinator.stop().await?;
        
        // Stop all agents
        self.registry.stop_all_agents().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.swarm_active = false;
        }
        
        info!("Data processing swarm stopped successfully");
        Ok(())
    }
    
    /// Get swarm state
    pub async fn get_state(&self) -> DataSwarmState {
        self.state.read().await.clone()
    }
    
    /// Get swarm metrics
    pub async fn get_metrics(&self) -> Result<DataSwarmMetrics> {
        self.coordinator.get_metrics().await
    }
}

/// Data swarm metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSwarmMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub messages_processed: u64,
    pub average_latency_us: f64,
    pub throughput_ops_per_sec: f64,
    pub error_rate: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_swarm_creation() {
        let config = DataSwarmConfig::default();
        let swarm = DataProcessingSwarm::new(config).await;
        assert!(swarm.is_ok());
    }
    
    #[test]
    async fn test_swarm_deployment() {
        let config = DataSwarmConfig::default();
        let swarm = DataProcessingSwarm::new(config).await.unwrap();
        
        let result = swarm.deploy().await;
        assert!(result.is_ok());
    }
}