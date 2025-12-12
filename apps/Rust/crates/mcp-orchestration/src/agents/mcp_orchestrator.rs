//! MCP Orchestrator Agent
//!
//! Central MCP coordination agent with Claude-Flow integration for managing
//! the entire swarm coordination system.

use std::sync::Arc;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, mpsc, broadcast, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use dashmap::DashMap;
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use parking_lot::RwLock as ParkingRwLock;
use atomic::{Atomic, Ordering};
use tracing::{debug, info, warn, error, instrument};
use anyhow::Result;
use futures::{Future, StreamExt};
use tokio::time::{sleep, timeout};

use crate::types::*;
use crate::error::OrchestrationError;
use crate::config::OrchestrationConfig;
use crate::agent::{Agent, AgentId, AgentInfo, AgentState, AgentType};
use crate::communication::{Message, MessageRouter, MessageType, CommunicationLayer};
use crate::coordination::{CoordinationEngine, CoordinationState};
use crate::health::{HealthChecker, HealthStatus, HealthMonitor};
use crate::load_balancer::{LoadBalancer, LoadBalancingStrategy};
use crate::memory::{SharedMemory, MemoryRegion, MemoryCoordinator};
use crate::metrics::{OrchestrationMetrics, MetricsCollector};
use crate::recovery::{RecoveryManager, RecoveryStrategy};
use crate::task_queue::{Task, TaskQueue, TaskPriority, TaskDistributor};

/// MCP Orchestrator Agent Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpOrchestratorConfig {
    /// Maximum number of agents in the swarm
    pub max_agents: usize,
    /// Coordination timeout in milliseconds
    pub coordination_timeout: u64,
    /// Heartbeat interval in milliseconds
    pub heartbeat_interval: u64,
    /// Message routing latency target in nanoseconds
    pub target_latency_ns: u64,
    /// Claude-Flow MCP server configuration
    pub claude_flow: ClaudeFlowConfig,
    /// Swarm topology configuration
    pub swarm_topology: SwarmTopologyConfig,
    /// TENGRI oversight configuration
    pub tengri_oversight: TengriOversightConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaudeFlowConfig {
    /// MCP server host
    pub host: String,
    /// MCP server port
    pub port: u16,
    /// API key for Claude-Flow integration
    pub api_key: String,
    /// Protocol version
    pub protocol_version: String,
    /// Enable real-time coordination
    pub enable_real_time: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmTopologyConfig {
    /// Hierarchical levels in the swarm
    pub hierarchy_levels: usize,
    /// Agents per level
    pub agents_per_level: Vec<usize>,
    /// Inter-level communication protocol
    pub communication_protocol: String,
    /// Topology optimization interval
    pub optimization_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriOversightConfig {
    /// Enable TENGRI oversight
    pub enabled: bool,
    /// Oversight validation interval
    pub validation_interval: u64,
    /// Quantum ML integration
    pub quantum_ml_enabled: bool,
    /// Watchdog configuration
    pub watchdog_config: WatchdogConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchdogConfig {
    /// Number of watchdog agents
    pub count: usize,
    /// Watchdog monitoring interval
    pub monitoring_interval: u64,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Maximum message latency in nanoseconds
    pub max_latency_ns: u64,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
    /// Maximum memory usage percentage
    pub max_memory_usage: f64,
    /// Maximum error rate percentage
    pub max_error_rate: f64,
}

/// Swarm coordination state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmCoordinationState {
    /// Active swarms
    pub active_swarms: HashMap<String, SwarmInfo>,
    /// Total agent count
    pub total_agents: usize,
    /// Current coordination phase
    pub coordination_phase: CoordinationPhase,
    /// Performance metrics
    pub performance_metrics: SwarmPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmInfo {
    /// Swarm identifier
    pub id: String,
    /// Swarm type
    pub swarm_type: SwarmType,
    /// Agent count
    pub agent_count: usize,
    /// Health status
    pub health_status: HealthStatus,
    /// Performance metrics
    pub metrics: SwarmMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwarmType {
    RiskManagement,
    TradingStrategy,
    DataPipeline,
    TengriWatchdog,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationPhase {
    Initialization,
    ActiveCoordination,
    OptimizationPhase,
    RecoveryMode,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmPerformanceMetrics {
    /// Average message latency in nanoseconds
    pub avg_latency_ns: u64,
    /// Messages per second
    pub messages_per_second: u64,
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage percentage
    pub memory_usage: f64,
    /// Error rate percentage
    pub error_rate: f64,
    /// Coordination efficiency
    pub coordination_efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmMetrics {
    /// Messages processed
    pub messages_processed: u64,
    /// Tasks completed
    pub tasks_completed: u64,
    /// Average response time
    pub avg_response_time: Duration,
    /// Success rate
    pub success_rate: f64,
}

/// MCP Orchestrator Agent
pub struct McpOrchestratorAgent {
    /// Agent ID
    id: AgentId,
    /// Configuration
    config: McpOrchestratorConfig,
    /// Agent state
    state: Arc<RwLock<AgentState>>,
    /// Coordination state
    coordination_state: Arc<RwLock<SwarmCoordinationState>>,
    /// Message router
    message_router: Arc<MessageRouter>,
    /// Communication layer
    communication: Arc<CommunicationLayer>,
    /// Coordination engine
    coordination_engine: Arc<CoordinationEngine>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
    /// Load balancer
    load_balancer: Arc<LoadBalancer>,
    /// Shared memory coordinator
    memory_coordinator: Arc<MemoryCoordinator>,
    /// Metrics collector
    metrics_collector: Arc<MetricsCollector>,
    /// Recovery manager
    recovery_manager: Arc<RecoveryManager>,
    /// Task distributor
    task_distributor: Arc<TaskDistributor>,
    /// Command channel
    command_tx: mpsc::UnboundedSender<McpCommand>,
    command_rx: Arc<Mutex<mpsc::UnboundedReceiver<McpCommand>>>,
    /// Event broadcast
    event_tx: broadcast::Sender<McpEvent>,
    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<mpsc::UnboundedSender<()>>>>,
    /// Performance tracking
    performance_tracker: Arc<PerformanceTracker>,
    /// Running state
    running: Arc<Atomic<bool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpCommand {
    StartCoordination,
    StopCoordination,
    AddSwarm(SwarmInfo),
    RemoveSwarm(String),
    OptimizeTopology,
    RecoverFromFailure(String),
    UpdateConfiguration(McpOrchestratorConfig),
    GetStatus,
    GetMetrics,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum McpEvent {
    CoordinationStarted,
    CoordinationStopped,
    SwarmAdded(SwarmInfo),
    SwarmRemoved(String),
    TopologyOptimized,
    FailureRecovered(String),
    ConfigurationUpdated,
    PerformanceAlert(String),
    ShutdownInitiated,
}

/// Performance tracking system
pub struct PerformanceTracker {
    /// Latency measurements
    latency_tracker: Arc<RwLock<Vec<u64>>>,
    /// Throughput measurements
    throughput_tracker: Arc<RwLock<Vec<u64>>>,
    /// Error tracking
    error_tracker: Arc<RwLock<Vec<String>>>,
    /// Start time
    start_time: Instant,
    /// Metrics update interval
    update_interval: Duration,
}

impl Default for McpOrchestratorConfig {
    fn default() -> Self {
        Self {
            max_agents: 25,
            coordination_timeout: 5000,
            heartbeat_interval: 1000,
            target_latency_ns: 1000, // 1 microsecond
            claude_flow: ClaudeFlowConfig {
                host: "localhost".to_string(),
                port: 3000,
                api_key: "default".to_string(),
                protocol_version: "1.0".to_string(),
                enable_real_time: true,
            },
            swarm_topology: SwarmTopologyConfig {
                hierarchy_levels: 3,
                agents_per_level: vec![1, 6, 18],
                communication_protocol: "ruv-swarm".to_string(),
                optimization_interval: 30000,
            },
            tengri_oversight: TengriOversightConfig {
                enabled: true,
                validation_interval: 5000,
                quantum_ml_enabled: true,
                watchdog_config: WatchdogConfig {
                    count: 8,
                    monitoring_interval: 1000,
                    alert_thresholds: AlertThresholds {
                        max_latency_ns: 10000, // 10 microseconds
                        max_cpu_usage: 80.0,
                        max_memory_usage: 85.0,
                        max_error_rate: 1.0,
                    },
                },
            },
        }
    }
}

impl McpOrchestratorAgent {
    /// Create a new MCP orchestrator agent
    pub async fn new(
        config: McpOrchestratorConfig,
        message_router: Arc<MessageRouter>,
        communication: Arc<CommunicationLayer>,
        coordination_engine: Arc<CoordinationEngine>,
        health_monitor: Arc<HealthMonitor>,
        load_balancer: Arc<LoadBalancer>,
        memory_coordinator: Arc<MemoryCoordinator>,
        metrics_collector: Arc<MetricsCollector>,
        recovery_manager: Arc<RecoveryManager>,
        task_distributor: Arc<TaskDistributor>,
    ) -> Result<Self> {
        let id = AgentId::new();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, _) = broadcast::channel(1024);
        let (shutdown_tx, _) = mpsc::unbounded_channel();

        let initial_state = SwarmCoordinationState {
            active_swarms: HashMap::new(),
            total_agents: 0,
            coordination_phase: CoordinationPhase::Initialization,
            performance_metrics: SwarmPerformanceMetrics {
                avg_latency_ns: 0,
                messages_per_second: 0,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                error_rate: 0.0,
                coordination_efficiency: 100.0,
            },
        };

        let performance_tracker = Arc::new(PerformanceTracker {
            latency_tracker: Arc::new(RwLock::new(Vec::new())),
            throughput_tracker: Arc::new(RwLock::new(Vec::new())),
            error_tracker: Arc::new(RwLock::new(Vec::new())),
            start_time: Instant::now(),
            update_interval: Duration::from_millis(config.heartbeat_interval),
        });

        Ok(Self {
            id,
            config,
            state: Arc::new(RwLock::new(AgentState::Initializing)),
            coordination_state: Arc::new(RwLock::new(initial_state)),
            message_router,
            communication,
            coordination_engine,
            health_monitor,
            load_balancer,
            memory_coordinator,
            metrics_collector,
            recovery_manager,
            task_distributor,
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            event_tx,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            performance_tracker,
            running: Arc::new(Atomic::new(false)),
        })
    }

    /// Start the MCP orchestrator agent
    #[instrument(skip(self), fields(agent_id = %self.id))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting MCP Orchestrator Agent {}", self.id);
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Running;
        }
        
        self.running.store(true, Ordering::SeqCst);
        
        // Start coordination phase
        {
            let mut coord_state = self.coordination_state.write().await;
            coord_state.coordination_phase = CoordinationPhase::ActiveCoordination;
        }
        
        // Spawn background tasks
        self.spawn_background_tasks().await?;
        
        // Start main event loop
        self.run_event_loop().await?;
        
        Ok(())
    }

    /// Stop the MCP orchestrator agent
    #[instrument(skip(self), fields(agent_id = %self.id))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping MCP Orchestrator Agent {}", self.id);
        
        self.running.store(false, Ordering::SeqCst);
        
        // Signal shutdown
        if let Some(shutdown_tx) = self.shutdown_tx.lock().await.take() {
            let _ = shutdown_tx.send(());
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Stopped;
        }
        
        // Update coordination phase
        {
            let mut coord_state = self.coordination_state.write().await;
            coord_state.coordination_phase = CoordinationPhase::Shutdown;
        }
        
        // Send shutdown event
        let _ = self.event_tx.send(McpEvent::ShutdownInitiated);
        
        Ok(())
    }

    /// Spawn background tasks
    async fn spawn_background_tasks(&self) -> Result<()> {
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.performance_monitoring_task().await;
        });

        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.health_monitoring_task().await;
        });

        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.topology_optimization_task().await;
        });

        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.coordination_task().await;
        });

        Ok(())
    }

    /// Main event loop
    async fn run_event_loop(&self) -> Result<()> {
        let mut command_rx = self.command_rx.lock().await;
        
        while self.running.load(Ordering::SeqCst) {
            tokio::select! {
                Some(command) = command_rx.recv() => {
                    if let Err(e) = self.handle_command(command).await {
                        error!("Error handling command: {}", e);
                    }
                }
                _ = sleep(Duration::from_millis(self.config.heartbeat_interval)) => {
                    self.heartbeat_task().await;
                }
            }
        }
        
        Ok(())
    }

    /// Handle incoming commands
    async fn handle_command(&self, command: McpCommand) -> Result<()> {
        match command {
            McpCommand::StartCoordination => self.start_coordination().await,
            McpCommand::StopCoordination => self.stop_coordination().await,
            McpCommand::AddSwarm(swarm_info) => self.add_swarm(swarm_info).await,
            McpCommand::RemoveSwarm(swarm_id) => self.remove_swarm(swarm_id).await,
            McpCommand::OptimizeTopology => self.optimize_topology().await,
            McpCommand::RecoverFromFailure(error_msg) => self.recover_from_failure(error_msg).await,
            McpCommand::UpdateConfiguration(config) => self.update_configuration(config).await,
            McpCommand::GetStatus => self.get_status().await,
            McpCommand::GetMetrics => self.get_metrics().await,
            McpCommand::Shutdown => self.stop().await,
        }
    }

    /// Start coordination
    async fn start_coordination(&self) -> Result<()> {
        info!("Starting swarm coordination");
        
        // Initialize swarms
        self.initialize_swarms().await?;
        
        // Start coordination engine
        self.coordination_engine.start().await?;
        
        // Update state
        {
            let mut coord_state = self.coordination_state.write().await;
            coord_state.coordination_phase = CoordinationPhase::ActiveCoordination;
        }
        
        let _ = self.event_tx.send(McpEvent::CoordinationStarted);
        
        Ok(())
    }

    /// Stop coordination
    async fn stop_coordination(&self) -> Result<()> {
        info!("Stopping swarm coordination");
        
        // Stop coordination engine
        self.coordination_engine.stop().await?;
        
        // Update state
        {
            let mut coord_state = self.coordination_state.write().await;
            coord_state.coordination_phase = CoordinationPhase::Shutdown;
        }
        
        let _ = self.event_tx.send(McpEvent::CoordinationStopped);
        
        Ok(())
    }

    /// Initialize swarms
    async fn initialize_swarms(&self) -> Result<()> {
        info!("Initializing swarms");
        
        // Initialize Risk Management Swarm (5 agents)
        let risk_swarm = SwarmInfo {
            id: "risk-management".to_string(),
            swarm_type: SwarmType::RiskManagement,
            agent_count: 5,
            health_status: HealthStatus::Healthy,
            metrics: SwarmMetrics {
                messages_processed: 0,
                tasks_completed: 0,
                avg_response_time: Duration::from_millis(0),
                success_rate: 100.0,
            },
        };
        
        // Initialize Trading Strategy Swarm (6 agents)
        let trading_swarm = SwarmInfo {
            id: "trading-strategy".to_string(),
            swarm_type: SwarmType::TradingStrategy,
            agent_count: 6,
            health_status: HealthStatus::Healthy,
            metrics: SwarmMetrics {
                messages_processed: 0,
                tasks_completed: 0,
                avg_response_time: Duration::from_millis(0),
                success_rate: 100.0,
            },
        };
        
        // Initialize Data Pipeline Swarm (6 agents)
        let data_swarm = SwarmInfo {
            id: "data-pipeline".to_string(),
            swarm_type: SwarmType::DataPipeline,
            agent_count: 6,
            health_status: HealthStatus::Healthy,
            metrics: SwarmMetrics {
                messages_processed: 0,
                tasks_completed: 0,
                avg_response_time: Duration::from_millis(0),
                success_rate: 100.0,
            },
        };
        
        // Initialize TENGRI Watchdog Swarm (8 agents)
        let tengri_swarm = SwarmInfo {
            id: "tengri-watchdog".to_string(),
            swarm_type: SwarmType::TengriWatchdog,
            agent_count: 8,
            health_status: HealthStatus::Healthy,
            metrics: SwarmMetrics {
                messages_processed: 0,
                tasks_completed: 0,
                avg_response_time: Duration::from_millis(0),
                success_rate: 100.0,
            },
        };
        
        // Add swarms to coordination state
        {
            let mut coord_state = self.coordination_state.write().await;
            coord_state.active_swarms.insert(risk_swarm.id.clone(), risk_swarm);
            coord_state.active_swarms.insert(trading_swarm.id.clone(), trading_swarm);
            coord_state.active_swarms.insert(data_swarm.id.clone(), data_swarm);
            coord_state.active_swarms.insert(tengri_swarm.id.clone(), tengri_swarm);
            coord_state.total_agents = 25;
        }
        
        Ok(())
    }

    /// Add a swarm to the coordination system
    async fn add_swarm(&self, swarm_info: SwarmInfo) -> Result<()> {
        info!("Adding swarm: {}", swarm_info.id);
        
        {
            let mut coord_state = self.coordination_state.write().await;
            coord_state.total_agents += swarm_info.agent_count;
            coord_state.active_swarms.insert(swarm_info.id.clone(), swarm_info.clone());
        }
        
        let _ = self.event_tx.send(McpEvent::SwarmAdded(swarm_info));
        
        Ok(())
    }

    /// Remove a swarm from the coordination system
    async fn remove_swarm(&self, swarm_id: String) -> Result<()> {
        info!("Removing swarm: {}", swarm_id);
        
        {
            let mut coord_state = self.coordination_state.write().await;
            if let Some(swarm) = coord_state.active_swarms.remove(&swarm_id) {
                coord_state.total_agents -= swarm.agent_count;
            }
        }
        
        let _ = self.event_tx.send(McpEvent::SwarmRemoved(swarm_id));
        
        Ok(())
    }

    /// Optimize topology
    async fn optimize_topology(&self) -> Result<()> {
        info!("Optimizing swarm topology");
        
        // Perform topology optimization
        self.load_balancer.optimize().await?;
        
        let _ = self.event_tx.send(McpEvent::TopologyOptimized);
        
        Ok(())
    }

    /// Recover from failure
    async fn recover_from_failure(&self, error_msg: String) -> Result<()> {
        warn!("Recovering from failure: {}", error_msg);
        
        // Trigger recovery
        self.recovery_manager.recover(&error_msg).await?;
        
        let _ = self.event_tx.send(McpEvent::FailureRecovered(error_msg));
        
        Ok(())
    }

    /// Update configuration
    async fn update_configuration(&self, config: McpOrchestratorConfig) -> Result<()> {
        info!("Updating configuration");
        
        // Update internal config (note: this is a simplified implementation)
        // In a real implementation, we would need to make config mutable
        
        let _ = self.event_tx.send(McpEvent::ConfigurationUpdated);
        
        Ok(())
    }

    /// Get status
    async fn get_status(&self) -> Result<()> {
        let coord_state = self.coordination_state.read().await;
        info!("Status: {} swarms, {} agents, phase: {:?}", 
              coord_state.active_swarms.len(), 
              coord_state.total_agents, 
              coord_state.coordination_phase);
        Ok(())
    }

    /// Get metrics
    async fn get_metrics(&self) -> Result<()> {
        let coord_state = self.coordination_state.read().await;
        info!("Metrics: avg_latency={}ns, msgs/sec={}, cpu={}%, mem={}%",
              coord_state.performance_metrics.avg_latency_ns,
              coord_state.performance_metrics.messages_per_second,
              coord_state.performance_metrics.cpu_usage,
              coord_state.performance_metrics.memory_usage);
        Ok(())
    }

    /// Performance monitoring task
    async fn performance_monitoring_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            // Collect performance metrics
            let start = Instant::now();
            
            // Simulate metric collection
            let latency = start.elapsed().as_nanos() as u64;
            self.performance_tracker.latency_tracker.write().await.push(latency);
            
            // Update coordination state metrics
            {
                let mut coord_state = self.coordination_state.write().await;
                coord_state.performance_metrics.avg_latency_ns = latency;
                coord_state.performance_metrics.messages_per_second = 10000; // Example
                coord_state.performance_metrics.cpu_usage = 45.0; // Example
                coord_state.performance_metrics.memory_usage = 60.0; // Example
            }
            
            sleep(self.performance_tracker.update_interval).await;
        }
    }

    /// Health monitoring task
    async fn health_monitoring_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            // Check health of all swarms
            let coord_state = self.coordination_state.read().await;
            
            for (swarm_id, swarm_info) in &coord_state.active_swarms {
                if swarm_info.health_status != HealthStatus::Healthy {
                    warn!("Swarm {} health status: {:?}", swarm_id, swarm_info.health_status);
                }
            }
            
            sleep(Duration::from_millis(self.config.heartbeat_interval)).await;
        }
    }

    /// Topology optimization task
    async fn topology_optimization_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            // Perform periodic topology optimization
            if let Err(e) = self.optimize_topology().await {
                error!("Topology optimization failed: {}", e);
            }
            
            sleep(Duration::from_millis(self.config.swarm_topology.optimization_interval)).await;
        }
    }

    /// Coordination task
    async fn coordination_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            // Perform coordination activities
            self.coordinate_swarms().await;
            
            sleep(Duration::from_millis(self.config.coordination_timeout)).await;
        }
    }

    /// Coordinate swarms
    async fn coordinate_swarms(&self) {
        debug!("Coordinating swarms");
        
        // Coordinate between all active swarms
        let coord_state = self.coordination_state.read().await;
        
        for (swarm_id, _swarm_info) in &coord_state.active_swarms {
            debug!("Coordinating swarm: {}", swarm_id);
            // Perform swarm-specific coordination
        }
    }

    /// Heartbeat task
    async fn heartbeat_task(&self) {
        debug!("Heartbeat");
        
        // Update last heartbeat timestamp
        // Check system health
        // Report to TENGRI oversight if enabled
    }

    /// Send command to the orchestrator
    pub async fn send_command(&self, command: McpCommand) -> Result<()> {
        self.command_tx.send(command)
            .map_err(|e| OrchestrationError::CommunicationError(e.to_string()))?;
        Ok(())
    }

    /// Subscribe to events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<McpEvent> {
        self.event_tx.subscribe()
    }

    /// Get current coordination state
    pub async fn get_coordination_state(&self) -> SwarmCoordinationState {
        self.coordination_state.read().await.clone()
    }

    /// Check if agent is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Clone for McpOrchestratorAgent {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone implementation
        // In a real system, you'd need to handle shared resources more carefully
        Self {
            id: self.id.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            coordination_state: Arc::clone(&self.coordination_state),
            message_router: Arc::clone(&self.message_router),
            communication: Arc::clone(&self.communication),
            coordination_engine: Arc::clone(&self.coordination_engine),
            health_monitor: Arc::clone(&self.health_monitor),
            load_balancer: Arc::clone(&self.load_balancer),
            memory_coordinator: Arc::clone(&self.memory_coordinator),
            metrics_collector: Arc::clone(&self.metrics_collector),
            recovery_manager: Arc::clone(&self.recovery_manager),
            task_distributor: Arc::clone(&self.task_distributor),
            command_tx: self.command_tx.clone(),
            command_rx: Arc::clone(&self.command_rx),
            event_tx: self.event_tx.clone(),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
            performance_tracker: Arc::clone(&self.performance_tracker),
            running: Arc::clone(&self.running),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_mcp_orchestrator_agent_creation() {
        // This test would require setting up all the dependencies
        // In a real implementation, you'd use dependency injection or mocking
        // For now, we'll just test the configuration
        let config = McpOrchestratorConfig::default();
        assert_eq!(config.max_agents, 25);
        assert_eq!(config.target_latency_ns, 1000);
    }

    #[tokio::test]
    async fn test_swarm_coordination_state() {
        let state = SwarmCoordinationState {
            active_swarms: HashMap::new(),
            total_agents: 0,
            coordination_phase: CoordinationPhase::Initialization,
            performance_metrics: SwarmPerformanceMetrics {
                avg_latency_ns: 0,
                messages_per_second: 0,
                cpu_usage: 0.0,
                memory_usage: 0.0,
                error_rate: 0.0,
                coordination_efficiency: 100.0,
            },
        };
        
        assert_eq!(state.total_agents, 0);
        assert!(matches!(state.coordination_phase, CoordinationPhase::Initialization));
    }
}