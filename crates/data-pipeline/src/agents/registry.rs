//! # Data Agent Registry
//!
//! Agent registry and management system for the ruv-swarm data processing agents.
//! Provides centralized agent lifecycle management and discovery services.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, Mutex};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataAgentInfo,
    DataMessage, CoordinationMessage, HealthStatus, AgentMetrics
};
use crate::agents::coordination::DataSwarmCoordinator;

/// Data agent registry
pub struct DataAgentRegistry {
    config: Arc<RegistryConfig>,
    agents: Arc<RwLock<HashMap<DataAgentId, RegisteredAgent>>>,
    agent_factories: Arc<RwLock<HashMap<DataAgentType, AgentFactory>>>,
    registry_metrics: Arc<RwLock<RegistryMetrics>>,
    state: Arc<RwLock<RegistryState>>,
}

/// Registry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    /// Agent timeout duration
    pub agent_timeout: Duration,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Automatic cleanup settings
    pub cleanup_config: CleanupConfig,
    /// Service discovery settings
    pub discovery_config: DiscoveryConfig,
}

impl Default for RegistryConfig {
    fn default() -> Self {
        Self {
            max_agents: 50,
            agent_timeout: Duration::from_secs(300),
            health_check_interval: Duration::from_secs(30),
            cleanup_config: CleanupConfig::default(),
            discovery_config: DiscoveryConfig::default(),
        }
    }
}

/// Cleanup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupConfig {
    /// Enable automatic cleanup
    pub enabled: bool,
    /// Cleanup interval
    pub cleanup_interval: Duration,
    /// Timeout for unresponsive agents
    pub unresponsive_timeout: Duration,
    /// Maximum failed health checks before removal
    pub max_failed_health_checks: u32,
}

impl Default for CleanupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cleanup_interval: Duration::from_secs(60),
            unresponsive_timeout: Duration::from_secs(120),
            max_failed_health_checks: 3,
        }
    }
}

/// Service discovery configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryConfig {
    /// Enable service discovery
    pub enabled: bool,
    /// Discovery protocol
    pub protocol: DiscoveryProtocol,
    /// Multicast address for discovery
    pub multicast_address: String,
    /// Discovery port
    pub discovery_port: u16,
    /// Announcement interval
    pub announcement_interval: Duration,
}

impl Default for DiscoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            protocol: DiscoveryProtocol::Multicast,
            multicast_address: "239.1.1.1".to_string(),
            discovery_port: 8080,
            announcement_interval: Duration::from_secs(30),
        }
    }
}

/// Discovery protocols
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DiscoveryProtocol {
    Multicast,
    Broadcast,
    DHT,
    DNS,
}

/// Registered agent information
#[derive(Debug, Clone)]
pub struct RegisteredAgent {
    pub agent: Arc<dyn DataAgent>,
    pub info: DataAgentInfo,
    pub registration_time: chrono::DateTime<chrono::Utc>,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub health_status: HealthStatus,
    pub failed_health_checks: u32,
    pub message_tx: mpsc::UnboundedSender<DataMessage>,
    pub coordination_tx: mpsc::UnboundedSender<CoordinationMessage>,
}

/// Agent factory for creating agents
pub type AgentFactory = Box<dyn Fn() -> Result<Box<dyn DataAgent>> + Send + Sync>;

/// Registry metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryMetrics {
    pub total_agents: usize,
    pub active_agents: usize,
    pub failed_agents: usize,
    pub registrations: u64,
    pub deregistrations: u64,
    pub health_checks_performed: u64,
    pub health_checks_failed: u64,
    pub average_agent_uptime: Duration,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for RegistryMetrics {
    fn default() -> Self {
        Self {
            total_agents: 0,
            active_agents: 0,
            failed_agents: 0,
            registrations: 0,
            deregistrations: 0,
            health_checks_performed: 0,
            health_checks_failed: 0,
            average_agent_uptime: Duration::from_secs(0),
            last_update: chrono::Utc::now(),
        }
    }
}

/// Registry state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryState {
    pub registry_active: bool,
    pub agents_by_type: HashMap<DataAgentType, usize>,
    pub total_capacity: usize,
    pub used_capacity: usize,
    pub last_cleanup: chrono::DateTime<chrono::Utc>,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for RegistryState {
    fn default() -> Self {
        Self {
            registry_active: false,
            agents_by_type: HashMap::new(),
            total_capacity: 0,
            used_capacity: 0,
            last_cleanup: chrono::Utc::now(),
            last_health_check: chrono::Utc::now(),
        }
    }
}

/// Agent manager for high-level operations
pub struct DataAgentManager {
    registry: Arc<DataAgentRegistry>,
    coordinator: Arc<DataSwarmCoordinator>,
    manager_metrics: Arc<RwLock<ManagerMetrics>>,
    state: Arc<RwLock<ManagerState>>,
}

/// Manager metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerMetrics {
    pub agents_started: u64,
    pub agents_stopped: u64,
    pub agents_restarted: u64,
    pub coordination_events: u64,
    pub management_operations: u64,
    pub average_operation_time_ms: f64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for ManagerMetrics {
    fn default() -> Self {
        Self {
            agents_started: 0,
            agents_stopped: 0,
            agents_restarted: 0,
            coordination_events: 0,
            management_operations: 0,
            average_operation_time_ms: 0.0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Manager state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManagerState {
    pub manager_active: bool,
    pub auto_scaling_enabled: bool,
    pub target_agent_count: usize,
    pub scaling_operations: u32,
    pub last_scaling_event: Option<chrono::DateTime<chrono::Utc>>,
}

impl Default for ManagerState {
    fn default() -> Self {
        Self {
            manager_active: false,
            auto_scaling_enabled: true,
            target_agent_count: 10,
            scaling_operations: 0,
            last_scaling_event: None,
        }
    }
}

impl DataAgentRegistry {
    /// Create a new data agent registry
    pub fn new(config: Arc<crate::agents::DataSwarmConfig>) -> Result<Self> {
        let registry_config = RegistryConfig::default();
        let config = Arc::new(registry_config);
        let agents = Arc::new(RwLock::new(HashMap::new()));
        let agent_factories = Arc::new(RwLock::new(HashMap::new()));
        let registry_metrics = Arc::new(RwLock::new(RegistryMetrics::default()));
        let state = Arc::new(RwLock::new(RegistryState::default()));
        
        Ok(Self {
            config,
            agents,
            agent_factories,
            registry_metrics,
            state,
        })
    }
    
    /// Register an agent
    pub async fn register_agent(&self, agent: Box<dyn DataAgent>) -> Result<DataAgentId> {
        let agent_id = agent.get_id();
        let agent_info = agent.get_info().await;
        
        // Check capacity
        let current_count = self.agents.read().await.len();
        if current_count >= self.config.max_agents {
            return Err(anyhow::anyhow!("Registry at maximum capacity"));
        }
        
        // Create communication channels
        let (message_tx, message_rx) = mpsc::unbounded_channel();
        let (coordination_tx, coordination_rx) = mpsc::unbounded_channel();
        
        let registered_agent = RegisteredAgent {
            agent: agent.into(),
            info: agent_info.clone(),
            registration_time: chrono::Utc::now(),
            last_heartbeat: chrono::Utc::now(),
            health_status: HealthStatus {
                status: crate::agents::base::HealthLevel::Unknown,
                last_check: chrono::Utc::now(),
                uptime: Duration::from_secs(0),
                issues: Vec::new(),
                metrics: crate::agents::base::HealthMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_mb: 0.0,
                    network_usage_mbps: 0.0,
                    disk_usage_mb: 0.0,
                    error_rate: 0.0,
                    response_time_ms: 0.0,
                },
            },
            failed_health_checks: 0,
            message_tx,
            coordination_tx,
        };
        
        // Register the agent
        self.agents.write().await.insert(agent_id, registered_agent);
        
        // Update metrics
        {
            let mut metrics = self.registry_metrics.write().await;
            metrics.total_agents = self.agents.read().await.len();
            metrics.registrations += 1;
            metrics.last_update = chrono::Utc::now();
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            let agent_count = state.agents_by_type.entry(agent_info.agent_type).or_insert(0);
            *agent_count += 1;
            state.used_capacity = self.agents.read().await.len();
        }
        
        info!("Agent {} registered with type {:?}", agent_id, agent_info.agent_type);
        Ok(agent_id)
    }
    
    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: DataAgentId) -> Result<()> {
        let removed_agent = self.agents.write().await.remove(&agent_id);
        
        if let Some(agent) = removed_agent {
            // Update metrics
            {
                let mut metrics = self.registry_metrics.write().await;
                metrics.total_agents = self.agents.read().await.len();
                metrics.deregistrations += 1;
                metrics.last_update = chrono::Utc::now();
            }
            
            // Update state
            {
                let mut state = self.state.write().await;
                if let Some(count) = state.agents_by_type.get_mut(&agent.info.agent_type) {
                    *count = count.saturating_sub(1);
                }
                state.used_capacity = self.agents.read().await.len();
            }
            
            info!("Agent {} unregistered", agent_id);
            Ok(())
        } else {
            Err(anyhow::anyhow!("Agent {} not found in registry", agent_id))
        }
    }
    
    /// Get agent by ID
    pub async fn get_agent(&self, agent_id: DataAgentId) -> Option<Arc<dyn DataAgent>> {
        self.agents.read().await.get(&agent_id).map(|registered| registered.agent.clone())
    }
    
    /// Get agents by type
    pub async fn get_agents_by_type(&self, agent_type: DataAgentType) -> Vec<Arc<dyn DataAgent>> {
        self.agents.read().await.values()
            .filter(|registered| registered.info.agent_type == agent_type)
            .map(|registered| registered.agent.clone())
            .collect()
    }
    
    /// Get all agents
    pub async fn get_all_agents(&self) -> Vec<Arc<dyn DataAgent>> {
        self.agents.read().await.values()
            .map(|registered| registered.agent.clone())
            .collect()
    }
    
    /// Perform health check on all agents
    pub async fn health_check_all(&self) -> Result<HashMap<DataAgentId, HealthStatus>> {
        let mut results = HashMap::new();
        let agents = self.agents.read().await;
        
        for (agent_id, registered) in agents.iter() {
            match registered.agent.health_check().await {
                Ok(health_status) => {
                    results.insert(*agent_id, health_status);
                    
                    // Update health status in registry
                    // Note: This would require making agents mutable or using interior mutability
                }
                Err(e) => {
                    warn!("Health check failed for agent {}: {}", agent_id, e);
                    
                    // Update failed health check count
                    // Note: This would require making agents mutable or using interior mutability
                }
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.registry_metrics.write().await;
            metrics.health_checks_performed += agents.len() as u64;
            metrics.health_checks_failed += (agents.len() - results.len()) as u64;
            metrics.last_update = chrono::Utc::now();
        }
        
        Ok(results)
    }
    
    /// Start all agents
    pub async fn start_all_agents(&self) -> Result<()> {
        let agents = self.get_all_agents().await;
        
        for agent in agents {
            if let Err(e) = agent.start().await {
                error!("Failed to start agent {}: {}", agent.get_id(), e);
            }
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.registry_active = true;
        }
        
        info!("All agents started");
        Ok(())
    }
    
    /// Stop all agents
    pub async fn stop_all_agents(&self) -> Result<()> {
        let agents = self.get_all_agents().await;
        
        for agent in agents {
            if let Err(e) = agent.stop().await {
                error!("Failed to stop agent {}: {}", agent.get_id(), e);
            }
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.registry_active = false;
        }
        
        info!("All agents stopped");
        Ok(())
    }
    
    /// Register agent factory
    pub async fn register_factory(&self, agent_type: DataAgentType, factory: AgentFactory) {
        self.agent_factories.write().await.insert(agent_type, factory);
        info!("Factory registered for agent type {:?}", agent_type);
    }
    
    /// Create agent using factory
    pub async fn create_agent(&self, agent_type: DataAgentType) -> Result<DataAgentId> {
        let factory = {
            let factories = self.agent_factories.read().await;
            factories.get(&agent_type).cloned()
        };
        
        if let Some(factory) = factory {
            let agent = factory()?;
            self.register_agent(agent).await
        } else {
            Err(anyhow::anyhow!("No factory registered for agent type {:?}", agent_type))
        }
    }
    
    /// Get registry metrics
    pub async fn get_metrics(&self) -> RegistryMetrics {
        self.registry_metrics.read().await.clone()
    }
    
    /// Get registry state
    pub async fn get_state(&self) -> RegistryState {
        self.state.read().await.clone()
    }
    
    /// Cleanup unresponsive agents
    pub async fn cleanup_unresponsive_agents(&self) -> Result<usize> {
        let mut cleanup_count = 0;
        let unresponsive_timeout = self.config.cleanup_config.unresponsive_timeout;
        let max_failed_checks = self.config.cleanup_config.max_failed_health_checks;
        let now = chrono::Utc::now();
        
        let agents_to_remove: Vec<DataAgentId> = {
            let agents = self.agents.read().await;
            agents.iter()
                .filter(|(_, registered)| {
                    let time_since_heartbeat = now.signed_duration_since(registered.last_heartbeat);
                    let is_unresponsive = time_since_heartbeat.num_seconds() > unresponsive_timeout.as_secs() as i64;
                    let has_failed_checks = registered.failed_health_checks >= max_failed_checks;
                    
                    is_unresponsive || has_failed_checks
                })
                .map(|(id, _)| *id)
                .collect()
        };
        
        for agent_id in agents_to_remove {
            if let Err(e) = self.unregister_agent(agent_id).await {
                error!("Failed to cleanup agent {}: {}", agent_id, e);
            } else {
                cleanup_count += 1;
            }
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.last_cleanup = now;
        }
        
        if cleanup_count > 0 {
            info!("Cleaned up {} unresponsive agents", cleanup_count);
        }
        
        Ok(cleanup_count)
    }
    
    /// Start background tasks
    pub async fn start_background_tasks(&self) {
        if self.config.cleanup_config.enabled {
            self.start_cleanup_task().await;
        }
        
        self.start_health_check_task().await;
    }
    
    /// Start cleanup background task
    async fn start_cleanup_task(&self) {
        let registry = Arc::new(self.clone());
        let interval = self.config.cleanup_config.cleanup_interval;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = registry.cleanup_unresponsive_agents().await {
                    error!("Cleanup task failed: {}", e);
                }
            }
        });
    }
    
    /// Start health check background task
    async fn start_health_check_task(&self) {
        let registry = Arc::new(self.clone());
        let interval = self.config.health_check_interval;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                if let Err(e) = registry.health_check_all().await {
                    error!("Health check task failed: {}", e);
                }
                
                // Update state
                {
                    let mut state = registry.state.write().await;
                    state.last_health_check = chrono::Utc::now();
                }
            }
        });
    }
}

impl Clone for DataAgentRegistry {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            agents: self.agents.clone(),
            agent_factories: self.agent_factories.clone(),
            registry_metrics: self.registry_metrics.clone(),
            state: self.state.clone(),
        }
    }
}

impl DataAgentManager {
    /// Create a new data agent manager
    pub fn new(registry: Arc<DataAgentRegistry>, coordinator: Arc<DataSwarmCoordinator>) -> Self {
        let manager_metrics = Arc::new(RwLock::new(ManagerMetrics::default()));
        let state = Arc::new(RwLock::new(ManagerState::default()));
        
        Self {
            registry,
            coordinator,
            manager_metrics,
            state,
        }
    }
    
    /// Start the manager
    pub async fn start(&self) -> Result<()> {
        info!("Starting data agent manager");
        
        // Start registry background tasks
        self.registry.start_background_tasks().await;
        
        // Start coordinator
        self.coordinator.start().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.manager_active = true;
        }
        
        info!("Data agent manager started successfully");
        Ok(())
    }
    
    /// Stop the manager
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping data agent manager");
        
        // Stop all agents
        self.registry.stop_all_agents().await?;
        
        // Stop coordinator
        self.coordinator.stop().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.manager_active = false;
        }
        
        info!("Data agent manager stopped successfully");
        Ok(())
    }
    
    /// Scale agents up
    pub async fn scale_up(&self, agent_type: DataAgentType, count: usize) -> Result<Vec<DataAgentId>> {
        let start_time = Instant::now();
        let mut created_agents = Vec::new();
        
        for _ in 0..count {
            match self.registry.create_agent(agent_type).await {
                Ok(agent_id) => {
                    created_agents.push(agent_id);
                    
                    // Register with coordinator
                    if let Err(e) = self.coordinator.register_agent(agent_id, agent_type, Vec::new()).await {
                        error!("Failed to register agent {} with coordinator: {}", agent_id, e);
                    }
                }
                Err(e) => {
                    error!("Failed to create agent of type {:?}: {}", agent_type, e);
                }
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.manager_metrics.write().await;
            metrics.agents_started += created_agents.len() as u64;
            metrics.management_operations += 1;
            let operation_time = start_time.elapsed().as_millis() as f64;
            metrics.average_operation_time_ms = 
                (metrics.average_operation_time_ms + operation_time) / 2.0;
            metrics.last_update = chrono::Utc::now();
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.scaling_operations += 1;
            state.last_scaling_event = Some(chrono::Utc::now());
        }
        
        info!("Scaled up {} agents of type {:?}", created_agents.len(), agent_type);
        Ok(created_agents)
    }
    
    /// Scale agents down
    pub async fn scale_down(&self, agent_type: DataAgentType, count: usize) -> Result<usize> {
        let start_time = Instant::now();
        let agents_to_remove = self.registry.get_agents_by_type(agent_type).await;
        let remove_count = count.min(agents_to_remove.len());
        let mut removed_count = 0;
        
        for agent in agents_to_remove.iter().take(remove_count) {
            let agent_id = agent.get_id();
            
            // Stop agent
            if let Err(e) = agent.stop().await {
                error!("Failed to stop agent {}: {}", agent_id, e);
                continue;
            }
            
            // Unregister from coordinator
            if let Err(e) = self.coordinator.unregister_agent(agent_id).await {
                error!("Failed to unregister agent {} from coordinator: {}", agent_id, e);
            }
            
            // Unregister from registry
            if let Err(e) = self.registry.unregister_agent(agent_id).await {
                error!("Failed to unregister agent {} from registry: {}", agent_id, e);
            } else {
                removed_count += 1;
            }
        }
        
        // Update metrics
        {
            let mut metrics = self.manager_metrics.write().await;
            metrics.agents_stopped += removed_count as u64;
            metrics.management_operations += 1;
            let operation_time = start_time.elapsed().as_millis() as f64;
            metrics.average_operation_time_ms = 
                (metrics.average_operation_time_ms + operation_time) / 2.0;
            metrics.last_update = chrono::Utc::now();
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.scaling_operations += 1;
            state.last_scaling_event = Some(chrono::Utc::now());
        }
        
        info!("Scaled down {} agents of type {:?}", removed_count, agent_type);
        Ok(removed_count)
    }
    
    /// Get manager metrics
    pub async fn get_metrics(&self) -> ManagerMetrics {
        self.manager_metrics.read().await.clone()
    }
    
    /// Get manager state
    pub async fn get_state(&self) -> ManagerState {
        self.state.read().await.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_registry_creation() {
        let config = Arc::new(crate::agents::DataSwarmConfig::default());
        let registry = DataAgentRegistry::new(config);
        assert!(registry.is_ok());
    }
    
    #[test]
    async fn test_agent_registration() {
        let config = Arc::new(crate::agents::DataSwarmConfig::default());
        let registry = DataAgentRegistry::new(config).unwrap();
        
        // Create a mock agent
        let mock_agent = create_mock_agent();
        let agent_id = mock_agent.get_id();
        
        let result = registry.register_agent(mock_agent).await;
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), agent_id);
        
        let metrics = registry.get_metrics().await;
        assert_eq!(metrics.total_agents, 1);
        assert_eq!(metrics.registrations, 1);
    }
    
    #[test]
    async fn test_agent_unregistration() {
        let config = Arc::new(crate::agents::DataSwarmConfig::default());
        let registry = DataAgentRegistry::new(config).unwrap();
        
        let mock_agent = create_mock_agent();
        let agent_id = mock_agent.get_id();
        
        registry.register_agent(mock_agent).await.unwrap();
        
        let result = registry.unregister_agent(agent_id).await;
        assert!(result.is_ok());
        
        let metrics = registry.get_metrics().await;
        assert_eq!(metrics.total_agents, 0);
        assert_eq!(metrics.deregistrations, 1);
    }
    
    fn create_mock_agent() -> Box<dyn DataAgent> {
        // This would create a mock agent implementation for testing
        // For now, we'll just use a placeholder
        todo!("Create mock agent implementation")
    }
}