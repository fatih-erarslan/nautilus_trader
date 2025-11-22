//! Core hive mind implementation

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::RwLock;
use tokio::time::{interval, timeout};
use uuid::Uuid;
use tracing::{info, error, debug, warn};

use crate::{
    config::HiveMindConfig,
    consensus::ConsensusEngine,
    memory::CollectiveMemory,
    neural::NeuralCoordinator,
    network::P2PNetwork,
    agents::AgentManager,
    metrics::MetricsCollector,
    error::{HiveMindError, Result},
    utils::concurrency::{CircuitBreaker, HealthMonitor, RecoveryManager},
};

/// The main hive mind coordinator
#[derive(Debug)]
pub struct HiveMind {
    /// Unique identifier for this hive mind instance
    id: Uuid,
    
    /// Configuration
    config: HiveMindConfig,
    
    /// Consensus engine for distributed decision making
    consensus: Arc<ConsensusEngine>,
    
    /// Collective memory system
    memory: Arc<RwLock<CollectiveMemory>>,
    
    /// Neural coordination system
    neural: Arc<NeuralCoordinator>,
    
    /// P2P networking layer
    network: Arc<P2PNetwork>,
    
    /// Agent management system
    agents: Arc<AgentManager>,
    
    /// Metrics collection system
    metrics: Arc<MetricsCollector>,
    
    /// Current state of the hive mind
    state: Arc<RwLock<HiveMindState>>,
    
    /// Circuit breakers for fault tolerance
    circuit_breakers: Arc<RwLock<HashMap<String, CircuitBreaker>>>,
    
    /// Recovery manager for self-healing
    recovery_manager: Arc<RecoveryManager>,
    
    /// Health monitoring system
    health_monitor: Arc<HealthMonitor>,
    
    /// System start time for uptime calculation
    start_time: Instant,
}

/// Current state of the hive mind system
#[derive(Debug, Clone)]
pub struct HiveMindState {
    /// Whether the system is running
    pub is_running: bool,
    
    /// Current operational mode
    pub mode: OperationalMode,
    
    /// Current consensus leader (if any)
    pub consensus_leader: Option<Uuid>,
    
    /// Number of active agents
    pub active_agents: usize,
    
    /// Number of connected peers
    pub connected_peers: usize,
    
    /// Memory usage statistics
    pub memory_usage: MemoryUsageStats,
    
    /// Neural processing statistics
    pub neural_stats: NeuralStats,
    
    /// System performance metrics
    pub performance: PerformanceStats,
    
    /// System health status
    pub health: SystemHealth,
    
    /// Last health check timestamp
    pub last_health_check: Instant,
}

/// Operational modes for the hive mind
#[derive(Debug, Clone, PartialEq)]
pub enum OperationalMode {
    Normal,
    Degraded,
    Recovery,
    Maintenance,
    Emergency,
}

/// System health status
#[derive(Debug, Clone)]
pub struct SystemHealth {
    pub overall_status: HealthStatus,
    pub consensus_health: HealthStatus,
    pub memory_health: HealthStatus,
    pub neural_health: HealthStatus,
    pub network_health: HealthStatus,
    pub agent_health: HealthStatus,
    pub recovery_attempts: u32,
    pub last_failure: Option<Instant>,
}

/// Health status levels
#[derive(Debug, Clone, PartialEq)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Failed,
    Recovering,
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryUsageStats {
    pub total_capacity: usize,
    pub used_capacity: usize,
    pub knowledge_nodes: usize,
    pub active_sessions: usize,
}

/// Neural processing statistics
#[derive(Debug, Clone)]
pub struct NeuralStats {
    pub patterns_recognized: u64,
    pub inference_count: u64,
    pub training_iterations: u64,
    pub model_accuracy: f64,
}

/// Performance statistics
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    pub messages_processed: u64,
    pub avg_response_time_ms: f64,
    pub consensus_success_rate: f64,
    pub uptime_seconds: u64,
}

impl HiveMind {
    /// Initialize the hive mind system
    pub async fn new(config: HiveMindConfig) -> Result<Self> {
        info!("Initializing Hive Mind system with ID: {}", config.instance_id);
        
        // Validate configuration
        config.validate()?;
        
        // Initialize metrics collector first
        let metrics = Arc::new(MetricsCollector::new(&config.metrics)?);
        
        // Initialize networking
        let network = Arc::new(P2PNetwork::new(&config.network, metrics.clone()).await?);
        
        // Initialize consensus engine
        let consensus = Arc::new(ConsensusEngine::new(
            &config.consensus,
            network.clone(),
            metrics.clone(),
        ).await?);
        
        // Initialize collective memory
        let memory = Arc::new(RwLock::new(
            CollectiveMemory::new(&config.memory, metrics.clone()).await?
        ));
        
        // Initialize neural coordinator
        let neural = Arc::new(NeuralCoordinator::new(
            &config.neural,
            memory.clone(),
            metrics.clone(),
        ).await?);
        
        // Initialize agent manager
        let agents = Arc::new(AgentManager::new(
            &config.agents,
            network.clone(),
            consensus.clone(),
            memory.clone(),
            neural.clone(),
            metrics.clone(),
        ).await?);
        
        // Initialize circuit breakers
        let circuit_breakers = Arc::new(RwLock::new(HashMap::new()));
        
        // Initialize recovery manager
        let recovery_manager = Arc::new(RecoveryManager::new(config.clone()));
        
        // Initialize health monitor
        let health_monitor = Arc::new(HealthMonitor::new(config.clone()));
        
        // Initialize state with fault tolerance features
        let state = Arc::new(RwLock::new(HiveMindState {
            is_running: false,
            mode: OperationalMode::Normal,
            consensus_leader: None,
            active_agents: 0,
            connected_peers: 0,
            memory_usage: MemoryUsageStats {
                total_capacity: config.memory.max_pool_size,
                used_capacity: 0,
                knowledge_nodes: 0,
                active_sessions: 0,
            },
            neural_stats: NeuralStats {
                patterns_recognized: 0,
                inference_count: 0,
                training_iterations: 0,
                model_accuracy: 0.0,
            },
            performance: PerformanceStats {
                messages_processed: 0,
                avg_response_time_ms: 0.0,
                consensus_success_rate: 0.0,
                uptime_seconds: 0,
            },
            health: SystemHealth {
                overall_status: HealthStatus::Healthy,
                consensus_health: HealthStatus::Healthy,
                memory_health: HealthStatus::Healthy,
                neural_health: HealthStatus::Healthy,
                network_health: HealthStatus::Healthy,
                agent_health: HealthStatus::Healthy,
                recovery_attempts: 0,
                last_failure: None,
            },
            last_health_check: Instant::now(),
        }));
        
        Ok(Self {
            id: config.instance_id,
            config,
            consensus,
            memory,
            neural,
            network,
            agents,
            metrics,
            state,
            circuit_breakers,
            recovery_manager,
            health_monitor,
            start_time: Instant::now(),
        })
    }
    
    /// Start the hive mind system with fault tolerance
    pub async fn start(&self) -> Result<()> {
        info!("Starting Hive Mind system with fault tolerance");
        
        // Initialize circuit breakers
        self.initialize_circuit_breakers().await?;
        
        // Start subsystems with recovery mechanisms
        self.start_subsystems_with_recovery().await?;
        
        // Start health monitoring
        self.start_health_monitoring().await?;
        
        // Start recovery manager
        self.start_recovery_manager().await?;
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.is_running = true;
            state.mode = OperationalMode::Normal;
        }
        
        // Start background tasks
        self.start_background_tasks().await?;
        
        info!("Hive Mind system started successfully with fault tolerance");
        Ok(())
    }
    
    /// Stop the hive mind system
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Hive Mind system");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.is_running = false;
        }
        
        // Stop components in reverse order
        self.agents.stop().await?;
        self.neural.stop().await?;
        self.consensus.stop().await?;
        self.network.stop().await?;
        self.metrics.stop().await?;
        
        info!("Hive Mind system stopped");
        Ok(())
    }
    
    /// Get current system state
    pub async fn get_state(&self) -> HiveMindState {
        self.state.read().await.clone()
    }
    
    /// Submit a decision proposal to the hive mind
    pub async fn submit_proposal(&self, proposal: serde_json::Value) -> Result<Uuid> {
        debug!("Submitting proposal to hive mind");
        self.consensus.submit_proposal(proposal).await
    }
    
    /// Query the collective memory
    pub async fn query_memory(&self, query: &str) -> Result<Vec<serde_json::Value>> {
        debug!("Querying collective memory: {}", query);
        let memory = self.memory.read().await;
        memory.search(query).await
    }
    
    /// Store knowledge in collective memory
    pub async fn store_knowledge(&self, key: &str, value: serde_json::Value) -> Result<()> {
        debug!("Storing knowledge: {}", key);
        let mut memory = self.memory.write().await;
        memory.store(key, value).await
    }
    
    /// Get neural insights on provided data
    pub async fn get_neural_insights(&self, data: &[f64]) -> Result<serde_json::Value> {
        debug!("Getting neural insights");
        self.neural.analyze_pattern(data).await
    }
    
    /// Spawn a new agent with specific capabilities
    pub async fn spawn_agent(&self, capabilities: Vec<String>) -> Result<Uuid> {
        debug!("Spawning new agent with capabilities: {:?}", capabilities);
        self.agents.spawn_agent(capabilities).await
    }
    
    /// Get list of active agents
    pub async fn get_active_agents(&self) -> Result<Vec<Uuid>> {
        self.agents.get_active_agents().await
    }
    
    /// Initialize circuit breakers for all critical components
    async fn initialize_circuit_breakers(&self) -> Result<()> {
        let mut breakers = self.circuit_breakers.write().await;
        
        // Create circuit breakers for each subsystem
        breakers.insert("consensus".to_string(), CircuitBreaker::new(5, Duration::from_secs(30)));
        breakers.insert("memory".to_string(), CircuitBreaker::new(5, Duration::from_secs(30)));
        breakers.insert("neural".to_string(), CircuitBreaker::new(3, Duration::from_secs(60)));
        breakers.insert("network".to_string(), CircuitBreaker::new(10, Duration::from_secs(10)));
        breakers.insert("agents".to_string(), CircuitBreaker::new(5, Duration::from_secs(30)));
        
        info!("Circuit breakers initialized for fault tolerance");
        Ok(())
    }
    
    /// Start subsystems with recovery mechanisms
    async fn start_subsystems_with_recovery(&self) -> Result<()> {
        let max_retries = 3;
        let retry_delay = Duration::from_secs(2);
        
        // Start networking layer with retry
        for attempt in 1..=max_retries {
            match timeout(Duration::from_secs(30), self.network.start()).await {
                Ok(Ok(())) => {
                    info!("Network layer started on attempt {}", attempt);
                    break;
                },
                Ok(Err(e)) => {
                    warn!("Network start failed on attempt {}: {}", attempt, e);
                    if attempt == max_retries {
                        return Err(e);
                    }
                    tokio::time::sleep(retry_delay).await;
                },
                Err(_) => {
                    warn!("Network start timed out on attempt {}", attempt);
                    if attempt == max_retries {
                        return Err(HiveMindError::Timeout("Network start timeout".to_string()));
                    }
                    tokio::time::sleep(retry_delay).await;
                }
            }
        }
        
        // Start consensus engine with fallback
        self.start_consensus_with_fallback().await?;
        
        // Start neural coordinator with graceful degradation
        self.start_neural_with_degradation().await?;
        
        // Start agent manager
        self.agents.start().await?;
        info!("Agent manager started");
        
        // Start metrics collection
        self.metrics.start().await?;
        info!("Metrics collector started");
        
        Ok(())
    }
    
    /// Start consensus with fallback protocols
    async fn start_consensus_with_fallback(&self) -> Result<()> {
        match self.consensus.start().await {
            Ok(()) => {
                info!("Consensus engine started with primary protocol");
                Ok(())
            },
            Err(e) => {
                warn!("Primary consensus failed, attempting fallback: {}", e);
                // Here we could switch to a simpler consensus algorithm
                // For now, we'll retry with exponential backoff
                for i in 1..=3 {
                    tokio::time::sleep(Duration::from_millis(100 * 2_u64.pow(i))).await;
                    if let Ok(()) = self.consensus.start().await {
                        info!("Consensus engine started with fallback on attempt {}", i + 1);
                        return Ok(());
                    }
                }
                Err(e)
            }
        }
    }
    
    /// Start neural with graceful degradation
    async fn start_neural_with_degradation(&self) -> Result<()> {
        match self.neural.start().await {
            Ok(()) => {
                info!("Neural coordinator started normally");
                Ok(())
            },
            Err(e) => {
                warn!("Neural coordinator failed to start, enabling degraded mode: {}", e);
                // Update state to reflect degraded neural capabilities
                {
                    let mut state = self.state.write().await;
                    state.mode = OperationalMode::Degraded;
                    state.health.neural_health = HealthStatus::Failed;
                }
                // Continue without neural capabilities for now
                Ok(())
            }
        }
    }
    
    /// Start health monitoring system
    async fn start_health_monitoring(&self) -> Result<()> {
        let health_monitor = self.health_monitor.clone();
        let state = self.state.clone();
        let consensus = self.consensus.clone();
        let memory = self.memory.clone();
        let neural = self.neural.clone();
        let network = self.network.clone();
        let agents = self.agents.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                let health = health_monitor.check_system_health(
                    &consensus,
                    &memory,
                    &neural,
                    &network,
                    &agents,
                ).await;
                
                // Update system state with health information
                {
                    let mut state_guard = state.write().await;
                    state_guard.health = health.clone();
                    state_guard.last_health_check = Instant::now();
                    
                    // Update operational mode based on health
                    match health.overall_status {
                        HealthStatus::Healthy => {
                            if state_guard.mode == OperationalMode::Degraded {
                                state_guard.mode = OperationalMode::Normal;
                                info!("System recovered to normal operation");
                            }
                        },
                        HealthStatus::Warning => {
                            if state_guard.mode == OperationalMode::Normal {
                                state_guard.mode = OperationalMode::Degraded;
                                warn!("System entering degraded mode");
                            }
                        },
                        HealthStatus::Critical | HealthStatus::Failed => {
                            if state_guard.mode != OperationalMode::Recovery {
                                state_guard.mode = OperationalMode::Recovery;
                                error!("System entering recovery mode");
                            }
                        },
                        HealthStatus::Recovering => {
                            state_guard.mode = OperationalMode::Recovery;
                        }
                    }
                }
            }
        });
        
        info!("Health monitoring started");
        Ok(())
    }
    
    /// Start recovery manager for self-healing
    async fn start_recovery_manager(&self) -> Result<()> {
        let recovery_manager = self.recovery_manager.clone();
        let state = self.state.clone();
        let consensus = self.consensus.clone();
        let memory = self.memory.clone();
        let neural = self.neural.clone();
        let network = self.network.clone();
        let agents = self.agents.clone();
        let circuit_breakers = self.circuit_breakers.clone();
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Check if recovery is needed
                let needs_recovery = {
                    let state_guard = state.read().await;
                    state_guard.mode == OperationalMode::Recovery ||
                    state_guard.health.overall_status == HealthStatus::Failed
                };
                
                if needs_recovery {
                    info!("Starting automated recovery procedures");
                    
                    if let Err(e) = recovery_manager.attempt_recovery(
                        &consensus,
                        &memory,
                        &neural,
                        &network,
                        &agents,
                        &circuit_breakers,
                    ).await {
                        error!("Recovery attempt failed: {}", e);
                        
                        // Increment recovery attempts
                        let mut state_guard = state.write().await;
                        state_guard.health.recovery_attempts += 1;
                        
                        // If too many recovery attempts, enter maintenance mode
                        if state_guard.health.recovery_attempts > 5 {
                            state_guard.mode = OperationalMode::Maintenance;
                            error!("Too many recovery attempts, entering maintenance mode");
                        }
                    } else {
                        info!("Recovery attempt successful");
                        let mut state_guard = state.write().await;
                        state_guard.health.recovery_attempts = 0;
                    }
                }
            }
        });
        
        info!("Recovery manager started");
        Ok(())
    }
    
    /// Start background monitoring and maintenance tasks
    async fn start_background_tasks(&self) -> Result<()> {
        let state = self.state.clone();
        let memory = self.memory.clone();
        let agents = self.agents.clone();
        let network = self.network.clone();
        let consensus = self.consensus.clone();
        let neural = self.neural.clone();
        let start_time = self.start_time;
        
        // State monitoring task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = Self::update_system_state(
                    &state,
                    &memory,
                    &agents,
                    &network,
                    &consensus,
                    &neural,
                    start_time,
                ).await {
                    error!("Failed to update system state: {}", e);
                }
            }
        });
        
        // Performance optimization task
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(300)); // Every 5 minutes
            
            loop {
                interval.tick().await;
                
                // Perform maintenance tasks like memory cleanup, agent optimization, etc.
                debug!("Running performance optimization tasks");
                
                // Memory cleanup
                if let Ok(mut memory_guard) = memory.try_write() {
                    if let Err(e) = memory_guard.cleanup_expired().await {
                        warn!("Memory cleanup failed: {}", e);
                    }
                }
                
                // Agent optimization
                if let Err(e) = agents.optimize_agent_distribution().await {
                    warn!("Agent optimization failed: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Update system state with current metrics
    async fn update_system_state(
        state: &Arc<RwLock<HiveMindState>>,
        memory: &Arc<RwLock<CollectiveMemory>>,
        agents: &Arc<AgentManager>,
        network: &Arc<P2PNetwork>,
        consensus: &Arc<ConsensusEngine>,
        neural: &Arc<NeuralCoordinator>,
        start_time: Instant,
    ) -> Result<()> {
        let mut state_guard = state.write().await;
        
        // Update agent count
        state_guard.active_agents = agents.get_agent_count().await?;
        
        // Update peer count
        state_guard.connected_peers = network.get_peer_count().await?;
        
        // Update consensus leader
        state_guard.consensus_leader = consensus.get_current_leader().await?;
        
        // Update memory usage
        {
            let memory_guard = memory.read().await;
            let memory_stats = memory_guard.get_usage_stats().await?;
            state_guard.memory_usage.used_capacity = memory_stats.used_capacity;
            state_guard.memory_usage.knowledge_nodes = memory_stats.knowledge_nodes;
            state_guard.memory_usage.active_sessions = memory_stats.active_sessions;
        }
        
        // Update neural stats
        let neural_stats = neural.get_statistics().await?;
        state_guard.neural_stats = neural_stats;
        
        // Update performance stats
        state_guard.performance.uptime_seconds = start_time.elapsed().as_secs();
        
        debug!("System state updated successfully");
        Ok(())
    }
    
    /// Emergency shutdown with data preservation
    pub async fn emergency_shutdown(&self) -> Result<()> {
        error!("Initiating emergency shutdown");
        
        // Update state to emergency mode
        {
            let mut state = self.state.write().await;
            state.mode = OperationalMode::Emergency;
            state.is_running = false;
        }
        
        // Preserve critical data first
        if let Err(e) = self.preserve_critical_data().await {
            error!("Failed to preserve critical data during emergency shutdown: {}", e);
        }
        
        // Force stop all components quickly
        let shutdown_timeout = Duration::from_secs(10);
        
        if let Err(e) = timeout(shutdown_timeout, self.stop()).await {
            error!("Emergency shutdown timed out: {:?}", e);
            // Force kill processes if necessary
            std::process::exit(1);
        }
        
        error!("Emergency shutdown completed");
        Ok(())
    }
    
    /// Preserve critical system data before shutdown
    async fn preserve_critical_data(&self) -> Result<()> {
        // Save consensus state
        if let Err(e) = self.consensus.save_state().await {
            warn!("Failed to save consensus state: {}", e);
        }
        
        // Save memory snapshots
        if let Ok(memory) = self.memory.try_read() {
            if let Err(e) = memory.create_snapshot().await {
                warn!("Failed to create memory snapshot: {}", e);
            }
        }
        
        // Save agent configurations
        if let Err(e) = self.agents.save_configurations().await {
            warn!("Failed to save agent configurations: {}", e);
        }
        
        info!("Critical data preservation completed");
        Ok(())
    }
    
    /// System recovery from previous state
    pub async fn recover_from_previous_state(&self) -> Result<()> {
        info!("Attempting system recovery from previous state");
        
        // Set recovery mode
        {
            let mut state = self.state.write().await;
            state.mode = OperationalMode::Recovery;
        }
        
        // Restore consensus state
        if let Err(e) = self.consensus.restore_state().await {
            warn!("Failed to restore consensus state: {}", e);
        }
        
        // Restore memory from snapshots
        if let Ok(mut memory) = self.memory.try_write() {
            if let Err(e) = memory.restore_from_snapshot().await {
                warn!("Failed to restore memory from snapshot: {}", e);
            }
        }
        
        // Restore agent configurations
        if let Err(e) = self.agents.restore_configurations().await {
            warn!("Failed to restore agent configurations: {}", e);
        }
        
        // Validate system integrity
        self.validate_system_integrity().await?;
        
        info!("System recovery completed successfully");
        Ok(())
    }
    
    /// Validate system integrity after recovery
    async fn validate_system_integrity(&self) -> Result<()> {
        // Check each subsystem
        let health = self.health_monitor.check_system_health(
            &self.consensus,
            &self.memory,
            &self.neural,
            &self.network,
            &self.agents,
        ).await;
        
        match health.overall_status {
            HealthStatus::Healthy | HealthStatus::Warning => {
                info!("System integrity validation passed");
                Ok(())
            },
            HealthStatus::Critical | HealthStatus::Failed => {
                error!("System integrity validation failed");
                Err(HiveMindError::SystemIntegrityFailure("Critical system components failed validation".to_string()))
            },
            HealthStatus::Recovering => {
                warn!("System still recovering, integrity check deferred");
                Ok(())
            }
        }
    }
}

/// Builder for constructing a HiveMind instance
pub struct HiveMindBuilder {
    config: HiveMindConfig,
}

impl HiveMindBuilder {
    /// Create a new builder with the given configuration
    pub fn new(config: HiveMindConfig) -> Self {
        Self { config }
    }
    
    /// Build the HiveMind instance
    pub async fn build(self) -> Result<HiveMind> {
        HiveMind::new(self.config).await
    }
    
    /// Configure networking settings
    pub fn with_network_config(mut self, network_config: crate::config::NetworkConfig) -> Self {
        self.config.network = network_config;
        self
    }
    
    /// Configure consensus settings
    pub fn with_consensus_config(mut self, consensus_config: crate::config::ConsensusConfig) -> Self {
        self.config.consensus = consensus_config;
        self
    }
    
    /// Configure memory settings
    pub fn with_memory_config(mut self, memory_config: crate::config::MemoryConfig) -> Self {
        self.config.memory = memory_config;
        self
    }
    
    /// Configure neural settings
    pub fn with_neural_config(mut self, neural_config: crate::config::NeuralConfig) -> Self {
        self.config.neural = neural_config;
        self
    }
    
    /// Configure agent settings
    pub fn with_agent_config(mut self, agent_config: crate::config::AgentConfig) -> Self {
        self.config.agents = agent_config;
        self
    }
}

impl Default for HiveMindState {
    fn default() -> Self {
        Self {
            is_running: false,
            mode: OperationalMode::Normal,
            consensus_leader: None,
            active_agents: 0,
            connected_peers: 0,
            memory_usage: MemoryUsageStats {
                total_capacity: 0,
                used_capacity: 0,
                knowledge_nodes: 0,
                active_sessions: 0,
            },
            neural_stats: NeuralStats {
                patterns_recognized: 0,
                inference_count: 0,
                training_iterations: 0,
                model_accuracy: 0.0,
            },
            performance: PerformanceStats {
                messages_processed: 0,
                avg_response_time_ms: 0.0,
                consensus_success_rate: 0.0,
                uptime_seconds: 0,
            },
            health: SystemHealth {
                overall_status: HealthStatus::Healthy,
                consensus_health: HealthStatus::Healthy,
                memory_health: HealthStatus::Healthy,
                neural_health: HealthStatus::Healthy,
                network_health: HealthStatus::Healthy,
                agent_health: HealthStatus::Healthy,
                recovery_attempts: 0,
                last_failure: None,
            },
            last_health_check: Instant::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::HiveMindConfig;

    #[tokio::test]
    async fn test_hive_mind_creation() {
        let config = HiveMindConfig::default();
        let hive_mind = HiveMind::new(config).await;
        assert!(hive_mind.is_ok());
    }

    #[tokio::test]
    async fn test_hive_mind_builder() {
        let config = HiveMindConfig::default();
        let builder = HiveMindBuilder::new(config);
        let hive_mind = builder.build().await;
        assert!(hive_mind.is_ok());
    }

    #[tokio::test]
    async fn test_state_default() {
        let state = HiveMindState::default();
        assert!(!state.is_running);
        assert_eq!(state.active_agents, 0);
        assert_eq!(state.connected_peers, 0);
    }
}