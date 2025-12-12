//! Agent Lifecycle Manager Agent
//!
//! Manages agent spawn/shutdown coordination across all swarms with real-time
//! lifecycle tracking, resource management, and graceful orchestration.

use std::sync::Arc;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};
use std::process::{Command, Stdio};
use tokio::sync::{RwLock, mpsc, broadcast, Mutex, Semaphore};
use tokio::process::{Child, Command as TokioCommand};
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
use rayon::prelude::*;

use crate::types::*;
use crate::error::OrchestrationError;
use crate::agent::{Agent, AgentId, AgentInfo, AgentState, AgentType};
use crate::communication::{Message, MessageRouter, MessageType};
use crate::health::{HealthStatus, HealthChecker};
use crate::metrics::{OrchestrationMetrics};
use crate::memory::{SharedMemory, MemoryCoordinator};

/// Agent lifecycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleConfig {
    /// Maximum concurrent agent spawns
    pub max_concurrent_spawns: usize,
    /// Agent startup timeout in milliseconds
    pub startup_timeout: u64,
    /// Agent shutdown timeout in milliseconds
    pub shutdown_timeout: u64,
    /// Health check interval in milliseconds
    pub health_check_interval: u64,
    /// Resource monitoring interval in milliseconds
    pub resource_monitor_interval: u64,
    /// Maximum memory per agent in bytes
    pub max_memory_per_agent: u64,
    /// Maximum CPU percentage per agent
    pub max_cpu_per_agent: f64,
    /// Auto-restart failed agents
    pub auto_restart_failed: bool,
    /// Graceful shutdown timeout
    pub graceful_shutdown_timeout: u64,
}

/// Agent lifecycle state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentLifecycleState {
    Pending,
    Spawning,
    Running,
    Stopping,
    Stopped,
    Failed,
    Restarting,
    Terminated,
}

/// Agent instance information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInstance {
    /// Agent ID
    pub id: AgentId,
    /// Agent type
    pub agent_type: AgentType,
    /// Swarm ID
    pub swarm_id: String,
    /// Current lifecycle state
    pub state: AgentLifecycleState,
    /// Process ID
    pub process_id: Option<u32>,
    /// Spawn timestamp
    pub spawn_time: Instant,
    /// Last health check
    pub last_health_check: Instant,
    /// Resource usage
    pub resource_usage: ResourceUsage,
    /// Configuration
    pub config: AgentConfig,
    /// Restart count
    pub restart_count: u32,
    /// Health status
    pub health_status: HealthStatus,
    /// Performance metrics
    pub performance_metrics: AgentPerformanceMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Network bytes sent
    pub network_bytes_sent: u64,
    /// Network bytes received
    pub network_bytes_received: u64,
    /// Disk bytes read
    pub disk_bytes_read: u64,
    /// Disk bytes written
    pub disk_bytes_written: u64,
    /// Open file descriptors
    pub open_fds: u32,
    /// Thread count
    pub thread_count: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    /// Agent executable path
    pub executable: String,
    /// Command line arguments
    pub args: Vec<String>,
    /// Environment variables
    pub env: HashMap<String, String>,
    /// Working directory
    pub working_dir: Option<String>,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Startup parameters
    pub startup_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory in bytes
    pub max_memory: u64,
    /// Maximum CPU percentage
    pub max_cpu: f64,
    /// Maximum network bandwidth
    pub max_network_bps: u64,
    /// Maximum disk I/O
    pub max_disk_iops: u64,
    /// Maximum file descriptors
    pub max_fds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceMetrics {
    /// Messages processed per second
    pub messages_per_second: f64,
    /// Average response time
    pub avg_response_time_ms: f64,
    /// Error rate percentage
    pub error_rate: f64,
    /// Uptime in seconds
    pub uptime_seconds: u64,
    /// Last performance update
    pub last_update: Instant,
}

/// Lifecycle command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleCommand {
    SpawnAgent {
        agent_type: AgentType,
        swarm_id: String,
        config: AgentConfig,
    },
    ShutdownAgent {
        agent_id: AgentId,
        graceful: bool,
    },
    RestartAgent {
        agent_id: AgentId,
    },
    KillAgent {
        agent_id: AgentId,
    },
    GetAgentStatus {
        agent_id: AgentId,
    },
    GetSwarmStatus {
        swarm_id: String,
    },
    GetAllAgents,
    ScaleSwarm {
        swarm_id: String,
        target_count: usize,
    },
    SetResourceLimits {
        agent_id: AgentId,
        limits: ResourceLimits,
    },
    UpdateAgentConfig {
        agent_id: AgentId,
        config: AgentConfig,
    },
    HealthCheck {
        agent_id: Option<AgentId>,
    },
    CollectMetrics,
    Shutdown,
}

/// Lifecycle event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LifecycleEvent {
    AgentSpawned {
        agent_id: AgentId,
        swarm_id: String,
        process_id: u32,
    },
    AgentStopped {
        agent_id: AgentId,
        swarm_id: String,
        reason: String,
    },
    AgentFailed {
        agent_id: AgentId,
        swarm_id: String,
        error: String,
    },
    AgentRestarted {
        agent_id: AgentId,
        swarm_id: String,
        restart_count: u32,
    },
    ResourceLimitExceeded {
        agent_id: AgentId,
        resource_type: String,
        current_value: f64,
        limit: f64,
    },
    HealthCheckFailed {
        agent_id: AgentId,
        reason: String,
    },
    SwarmScaled {
        swarm_id: String,
        old_count: usize,
        new_count: usize,
    },
}

/// Agent Lifecycle Manager
pub struct AgentLifecycleManager {
    /// Manager ID
    id: AgentId,
    /// Configuration
    config: LifecycleConfig,
    /// Manager state
    state: Arc<RwLock<AgentState>>,
    /// Active agents
    agents: Arc<RwLock<HashMap<AgentId, AgentInstance>>>,
    /// Swarm agent mappings
    swarm_agents: Arc<RwLock<HashMap<String, HashSet<AgentId>>>>,
    /// Process handles
    processes: Arc<RwLock<HashMap<AgentId, Child>>>,
    /// Message router
    message_router: Arc<MessageRouter>,
    /// Health checker
    health_checker: Arc<HealthChecker>,
    /// Memory coordinator
    memory_coordinator: Arc<MemoryCoordinator>,
    /// Spawn semaphore
    spawn_semaphore: Arc<Semaphore>,
    /// Command channel
    command_tx: mpsc::UnboundedSender<LifecycleCommand>,
    command_rx: Arc<Mutex<mpsc::UnboundedReceiver<LifecycleCommand>>>,
    /// Event broadcast
    event_tx: broadcast::Sender<LifecycleEvent>,
    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<mpsc::UnboundedSender<()>>>>,
    /// Running state
    running: Arc<Atomic<bool>>,
    /// Resource monitor
    resource_monitor: Arc<ResourceMonitor>,
    /// Lifecycle metrics
    lifecycle_metrics: Arc<RwLock<LifecycleMetrics>>,
}

/// Resource monitoring system
pub struct ResourceMonitor {
    /// CPU usage tracker
    cpu_tracker: Arc<RwLock<HashMap<AgentId, VecDeque<f64>>>>,
    /// Memory usage tracker
    memory_tracker: Arc<RwLock<HashMap<AgentId, VecDeque<u64>>>>,
    /// Network usage tracker
    network_tracker: Arc<RwLock<HashMap<AgentId, VecDeque<(u64, u64)>>>>,
    /// Monitoring interval
    interval: Duration,
    /// History size
    history_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LifecycleMetrics {
    /// Total agents spawned
    pub total_spawned: u64,
    /// Total agents shutdown
    pub total_shutdown: u64,
    /// Total agents failed
    pub total_failed: u64,
    /// Total restarts
    pub total_restarts: u64,
    /// Average startup time
    pub avg_startup_time_ms: f64,
    /// Average shutdown time
    pub avg_shutdown_time_ms: f64,
    /// Current active agents
    pub active_agents: usize,
    /// Resource utilization
    pub resource_utilization: ResourceUtilization,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// Total CPU usage
    pub total_cpu_usage: f64,
    /// Total memory usage
    pub total_memory_usage: u64,
    /// Total network usage
    pub total_network_usage: u64,
    /// Average CPU per agent
    pub avg_cpu_per_agent: f64,
    /// Average memory per agent
    pub avg_memory_per_agent: u64,
}

impl Default for LifecycleConfig {
    fn default() -> Self {
        Self {
            max_concurrent_spawns: 5,
            startup_timeout: 30000,
            shutdown_timeout: 10000,
            health_check_interval: 5000,
            resource_monitor_interval: 1000,
            max_memory_per_agent: 2_000_000_000, // 2GB
            max_cpu_per_agent: 50.0,
            auto_restart_failed: true,
            graceful_shutdown_timeout: 15000,
        }
    }
}

impl AgentLifecycleManager {
    /// Create a new agent lifecycle manager
    pub async fn new(
        config: LifecycleConfig,
        message_router: Arc<MessageRouter>,
        health_checker: Arc<HealthChecker>,
        memory_coordinator: Arc<MemoryCoordinator>,
    ) -> Result<Self> {
        let id = AgentId::new();
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let (event_tx, _) = broadcast::channel(1024);
        let (shutdown_tx, _) = mpsc::unbounded_channel();

        let spawn_semaphore = Arc::new(Semaphore::new(config.max_concurrent_spawns));

        let resource_monitor = Arc::new(ResourceMonitor {
            cpu_tracker: Arc::new(RwLock::new(HashMap::new())),
            memory_tracker: Arc::new(RwLock::new(HashMap::new())),
            network_tracker: Arc::new(RwLock::new(HashMap::new())),
            interval: Duration::from_millis(config.resource_monitor_interval),
            history_size: 100,
        });

        let initial_metrics = LifecycleMetrics {
            total_spawned: 0,
            total_shutdown: 0,
            total_failed: 0,
            total_restarts: 0,
            avg_startup_time_ms: 0.0,
            avg_shutdown_time_ms: 0.0,
            active_agents: 0,
            resource_utilization: ResourceUtilization {
                total_cpu_usage: 0.0,
                total_memory_usage: 0,
                total_network_usage: 0,
                avg_cpu_per_agent: 0.0,
                avg_memory_per_agent: 0,
            },
        };

        Ok(Self {
            id,
            config,
            state: Arc::new(RwLock::new(AgentState::Initializing)),
            agents: Arc::new(RwLock::new(HashMap::new())),
            swarm_agents: Arc::new(RwLock::new(HashMap::new())),
            processes: Arc::new(RwLock::new(HashMap::new())),
            message_router,
            health_checker,
            memory_coordinator,
            spawn_semaphore,
            command_tx,
            command_rx: Arc::new(Mutex::new(command_rx)),
            event_tx,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            running: Arc::new(Atomic::new(false)),
            resource_monitor,
            lifecycle_metrics: Arc::new(RwLock::new(initial_metrics)),
        })
    }

    /// Start the lifecycle manager
    #[instrument(skip(self), fields(manager_id = %self.id))]
    pub async fn start(&self) -> Result<()> {
        info!("Starting Agent Lifecycle Manager {}", self.id);
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Running;
        }
        
        self.running.store(true, Ordering::SeqCst);
        
        // Initialize swarm mappings
        self.initialize_swarm_mappings().await?;
        
        // Spawn background tasks
        self.spawn_background_tasks().await?;
        
        // Start main event loop
        self.run_event_loop().await?;
        
        Ok(())
    }

    /// Stop the lifecycle manager
    #[instrument(skip(self), fields(manager_id = %self.id))]
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping Agent Lifecycle Manager {}", self.id);
        
        self.running.store(false, Ordering::SeqCst);
        
        // Shutdown all agents gracefully
        self.shutdown_all_agents().await?;
        
        // Signal shutdown
        if let Some(shutdown_tx) = self.shutdown_tx.lock().await.take() {
            let _ = shutdown_tx.send(());
        }
        
        // Update state
        {
            let mut state = self.state.write().await;
            *state = AgentState::Stopped;
        }
        
        Ok(())
    }

    /// Initialize swarm mappings
    async fn initialize_swarm_mappings(&self) -> Result<()> {
        let mut swarm_agents = self.swarm_agents.write().await;
        
        // Initialize empty mappings for known swarms
        swarm_agents.insert("risk-management".to_string(), HashSet::new());
        swarm_agents.insert("trading-strategy".to_string(), HashSet::new());
        swarm_agents.insert("data-pipeline".to_string(), HashSet::new());
        swarm_agents.insert("tengri-watchdog".to_string(), HashSet::new());
        
        Ok(())
    }

    /// Spawn background tasks
    async fn spawn_background_tasks(&self) -> Result<()> {
        // Health monitoring task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.health_monitoring_task().await;
        });

        // Resource monitoring task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.resource_monitoring_task().await;
        });

        // Metrics collection task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.metrics_collection_task().await;
        });

        // Cleanup task
        let self_clone = self.clone();
        tokio::spawn(async move {
            self_clone.cleanup_task().await;
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
                        error!("Error handling lifecycle command: {}", e);
                    }
                }
                _ = sleep(Duration::from_millis(100)) => {
                    // Regular maintenance
                    self.maintenance_cycle().await;
                }
            }
        }
        
        Ok(())
    }

    /// Handle incoming commands
    async fn handle_command(&self, command: LifecycleCommand) -> Result<()> {
        match command {
            LifecycleCommand::SpawnAgent { agent_type, swarm_id, config } => {
                self.spawn_agent(agent_type, swarm_id, config).await
            }
            LifecycleCommand::ShutdownAgent { agent_id, graceful } => {
                self.shutdown_agent(agent_id, graceful).await
            }
            LifecycleCommand::RestartAgent { agent_id } => {
                self.restart_agent(agent_id).await
            }
            LifecycleCommand::KillAgent { agent_id } => {
                self.kill_agent(agent_id).await
            }
            LifecycleCommand::GetAgentStatus { agent_id } => {
                self.get_agent_status(agent_id).await
            }
            LifecycleCommand::GetSwarmStatus { swarm_id } => {
                self.get_swarm_status(swarm_id).await
            }
            LifecycleCommand::GetAllAgents => {
                self.get_all_agents().await
            }
            LifecycleCommand::ScaleSwarm { swarm_id, target_count } => {
                self.scale_swarm(swarm_id, target_count).await
            }
            LifecycleCommand::SetResourceLimits { agent_id, limits } => {
                self.set_resource_limits(agent_id, limits).await
            }
            LifecycleCommand::UpdateAgentConfig { agent_id, config } => {
                self.update_agent_config(agent_id, config).await
            }
            LifecycleCommand::HealthCheck { agent_id } => {
                self.perform_health_check(agent_id).await
            }
            LifecycleCommand::CollectMetrics => {
                self.collect_metrics().await
            }
            LifecycleCommand::Shutdown => {
                self.stop().await
            }
        }
    }

    /// Spawn a new agent
    async fn spawn_agent(
        &self,
        agent_type: AgentType,
        swarm_id: String,
        config: AgentConfig,
    ) -> Result<()> {
        info!("Spawning agent of type {:?} for swarm {}", agent_type, swarm_id);
        
        // Acquire spawn semaphore
        let _permit = self.spawn_semaphore.acquire().await?;
        
        let agent_id = AgentId::new();
        let spawn_time = Instant::now();
        
        // Create agent instance
        let mut agent_instance = AgentInstance {
            id: agent_id.clone(),
            agent_type,
            swarm_id: swarm_id.clone(),
            state: AgentLifecycleState::Spawning,
            process_id: None,
            spawn_time,
            last_health_check: spawn_time,
            resource_usage: ResourceUsage {
                cpu_usage: 0.0,
                memory_usage: 0,
                network_bytes_sent: 0,
                network_bytes_received: 0,
                disk_bytes_read: 0,
                disk_bytes_written: 0,
                open_fds: 0,
                thread_count: 0,
            },
            config: config.clone(),
            restart_count: 0,
            health_status: HealthStatus::Unknown,
            performance_metrics: AgentPerformanceMetrics {
                messages_per_second: 0.0,
                avg_response_time_ms: 0.0,
                error_rate: 0.0,
                uptime_seconds: 0,
                last_update: spawn_time,
            },
        };
        
        // Add to agents map
        {
            let mut agents = self.agents.write().await;
            agents.insert(agent_id.clone(), agent_instance.clone());
        }
        
        // Add to swarm mapping
        {
            let mut swarm_agents = self.swarm_agents.write().await;
            swarm_agents.entry(swarm_id.clone()).or_insert_with(HashSet::new).insert(agent_id.clone());
        }
        
        // Start the agent process
        match self.start_agent_process(&config, &agent_id).await {
            Ok(process) => {
                let process_id = process.id().unwrap_or(0);
                
                // Store process handle
                {
                    let mut processes = self.processes.write().await;
                    processes.insert(agent_id.clone(), process);
                }
                
                // Update agent instance
                {
                    let mut agents = self.agents.write().await;
                    if let Some(agent) = agents.get_mut(&agent_id) {
                        agent.process_id = Some(process_id);
                        agent.state = AgentLifecycleState::Running;
                    }
                }
                
                // Update metrics
                {
                    let mut metrics = self.lifecycle_metrics.write().await;
                    metrics.total_spawned += 1;
                    metrics.active_agents += 1;
                    
                    let startup_time = spawn_time.elapsed().as_millis() as f64;
                    metrics.avg_startup_time_ms = 
                        (metrics.avg_startup_time_ms + startup_time) / 2.0;
                }
                
                let _ = self.event_tx.send(LifecycleEvent::AgentSpawned {
                    agent_id,
                    swarm_id,
                    process_id,
                });
                
                info!("Agent spawned successfully with PID {}", process_id);
            }
            Err(e) => {
                error!("Failed to spawn agent: {}", e);
                
                // Update agent state to failed
                {
                    let mut agents = self.agents.write().await;
                    if let Some(agent) = agents.get_mut(&agent_id) {
                        agent.state = AgentLifecycleState::Failed;
                    }
                }
                
                // Update metrics
                {
                    let mut metrics = self.lifecycle_metrics.write().await;
                    metrics.total_failed += 1;
                }
                
                let _ = self.event_tx.send(LifecycleEvent::AgentFailed {
                    agent_id,
                    swarm_id,
                    error: e.to_string(),
                });
                
                return Err(e);
            }
        }
        
        Ok(())
    }

    /// Start agent process
    async fn start_agent_process(&self, config: &AgentConfig, agent_id: &AgentId) -> Result<Child> {
        let mut cmd = TokioCommand::new(&config.executable);
        cmd.args(&config.args);
        cmd.envs(&config.env);
        cmd.env("AGENT_ID", agent_id.to_string());
        cmd.stdin(Stdio::null());
        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());
        
        if let Some(working_dir) = &config.working_dir {
            cmd.current_dir(working_dir);
        }
        
        let process = timeout(
            Duration::from_millis(self.config.startup_timeout),
            cmd.spawn()
        ).await??;
        
        Ok(process)
    }

    /// Shutdown an agent
    async fn shutdown_agent(&self, agent_id: AgentId, graceful: bool) -> Result<()> {
        info!("Shutting down agent {} (graceful: {})", agent_id, graceful);
        
        let swarm_id = {
            let agents = self.agents.read().await;
            agents.get(&agent_id).map(|a| a.swarm_id.clone())
        };
        
        if let Some(swarm_id) = swarm_id {
            // Update agent state
            {
                let mut agents = self.agents.write().await;
                if let Some(agent) = agents.get_mut(&agent_id) {
                    agent.state = AgentLifecycleState::Stopping;
                }
            }
            
            // Stop the process
            let shutdown_start = Instant::now();
            if let Err(e) = self.stop_agent_process(&agent_id, graceful).await {
                error!("Error stopping agent process: {}", e);
            }
            
            // Remove from tracking
            {
                let mut agents = self.agents.write().await;
                if let Some(agent) = agents.get_mut(&agent_id) {
                    agent.state = AgentLifecycleState::Stopped;
                }
            }
            
            // Remove from swarm mapping
            {
                let mut swarm_agents = self.swarm_agents.write().await;
                if let Some(agents) = swarm_agents.get_mut(&swarm_id) {
                    agents.remove(&agent_id);
                }
            }
            
            // Remove process handle
            {
                let mut processes = self.processes.write().await;
                processes.remove(&agent_id);
            }
            
            // Update metrics
            {
                let mut metrics = self.lifecycle_metrics.write().await;
                metrics.total_shutdown += 1;
                metrics.active_agents = metrics.active_agents.saturating_sub(1);
                
                let shutdown_time = shutdown_start.elapsed().as_millis() as f64;
                metrics.avg_shutdown_time_ms = 
                    (metrics.avg_shutdown_time_ms + shutdown_time) / 2.0;
            }
            
            let _ = self.event_tx.send(LifecycleEvent::AgentStopped {
                agent_id,
                swarm_id,
                reason: if graceful { "graceful_shutdown" } else { "forced_shutdown" }.to_string(),
            });
            
            info!("Agent shutdown completed");
        }
        
        Ok(())
    }

    /// Stop agent process
    async fn stop_agent_process(&self, agent_id: &AgentId, graceful: bool) -> Result<()> {
        let mut process = {
            let mut processes = self.processes.write().await;
            processes.remove(agent_id)
        };
        
        if let Some(mut process) = process {
            if graceful {
                // Try graceful shutdown first
                if let Some(id) = process.id() {
                    #[cfg(unix)]
                    {
                        use std::process::Command;
                        let _ = Command::new("kill")
                            .arg("-TERM")
                            .arg(id.to_string())
                            .output();
                    }
                    
                    // Wait for graceful shutdown
                    match timeout(
                        Duration::from_millis(self.config.graceful_shutdown_timeout),
                        process.wait()
                    ).await {
                        Ok(Ok(_)) => {
                            info!("Agent {} shutdown gracefully", agent_id);
                            return Ok(());
                        }
                        Ok(Err(e)) => {
                            warn!("Error during graceful shutdown: {}", e);
                        }
                        Err(_) => {
                            warn!("Graceful shutdown timeout, forcing kill");
                        }
                    }
                }
            }
            
            // Force kill if graceful failed or not requested
            if let Err(e) = process.kill().await {
                error!("Error force killing agent process: {}", e);
            }
        }
        
        Ok(())
    }

    /// Restart an agent
    async fn restart_agent(&self, agent_id: AgentId) -> Result<()> {
        info!("Restarting agent {}", agent_id);
        
        let (agent_type, swarm_id, config) = {
            let agents = self.agents.read().await;
            if let Some(agent) = agents.get(&agent_id) {
                (agent.agent_type.clone(), agent.swarm_id.clone(), agent.config.clone())
            } else {
                return Err(OrchestrationError::AgentNotFound(agent_id.to_string()).into());
            }
        };
        
        // Update restart count
        {
            let mut agents = self.agents.write().await;
            if let Some(agent) = agents.get_mut(&agent_id) {
                agent.restart_count += 1;
                agent.state = AgentLifecycleState::Restarting;
            }
        }
        
        // Shutdown existing instance
        self.shutdown_agent(agent_id.clone(), false).await?;
        
        // Wait a bit before restarting
        sleep(Duration::from_millis(1000)).await;
        
        // Spawn new instance
        self.spawn_agent(agent_type, swarm_id.clone(), config).await?;
        
        // Update metrics
        {
            let mut metrics = self.lifecycle_metrics.write().await;
            metrics.total_restarts += 1;
        }
        
        let restart_count = {
            let agents = self.agents.read().await;
            agents.get(&agent_id).map(|a| a.restart_count).unwrap_or(0)
        };
        
        let _ = self.event_tx.send(LifecycleEvent::AgentRestarted {
            agent_id,
            swarm_id,
            restart_count,
        });
        
        info!("Agent restarted successfully");
        Ok(())
    }

    /// Kill an agent forcefully
    async fn kill_agent(&self, agent_id: AgentId) -> Result<()> {
        info!("Killing agent {} forcefully", agent_id);
        
        self.shutdown_agent(agent_id, false).await?;
        
        Ok(())
    }

    /// Get agent status
    async fn get_agent_status(&self, agent_id: AgentId) -> Result<()> {
        let agents = self.agents.read().await;
        if let Some(agent) = agents.get(&agent_id) {
            info!("Agent {} status: {:?}, PID: {:?}, Uptime: {}s",
                  agent_id, agent.state, agent.process_id, 
                  agent.spawn_time.elapsed().as_secs());
        } else {
            warn!("Agent {} not found", agent_id);
        }
        
        Ok(())
    }

    /// Get swarm status
    async fn get_swarm_status(&self, swarm_id: String) -> Result<()> {
        let swarm_agents = self.swarm_agents.read().await;
        if let Some(agents) = swarm_agents.get(&swarm_id) {
            info!("Swarm {} status: {} agents", swarm_id, agents.len());
            
            let agents_map = self.agents.read().await;
            for agent_id in agents {
                if let Some(agent) = agents_map.get(agent_id) {
                    info!("  Agent {}: {:?}", agent_id, agent.state);
                }
            }
        } else {
            warn!("Swarm {} not found", swarm_id);
        }
        
        Ok(())
    }

    /// Get all agents
    async fn get_all_agents(&self) -> Result<()> {
        let agents = self.agents.read().await;
        info!("Total agents: {}", agents.len());
        
        for (agent_id, agent) in agents.iter() {
            info!("Agent {}: {:?} ({})", agent_id, agent.state, agent.swarm_id);
        }
        
        Ok(())
    }

    /// Scale a swarm
    async fn scale_swarm(&self, swarm_id: String, target_count: usize) -> Result<()> {
        info!("Scaling swarm {} to {} agents", swarm_id, target_count);
        
        let current_count = {
            let swarm_agents = self.swarm_agents.read().await;
            swarm_agents.get(&swarm_id).map(|a| a.len()).unwrap_or(0)
        };
        
        if target_count > current_count {
            // Scale up
            let agents_to_add = target_count - current_count;
            for _ in 0..agents_to_add {
                let config = self.get_default_agent_config(&swarm_id).await?;
                let agent_type = self.get_swarm_agent_type(&swarm_id);
                self.spawn_agent(agent_type, swarm_id.clone(), config).await?;
            }
        } else if target_count < current_count {
            // Scale down
            let agents_to_remove = current_count - target_count;
            let agents_to_shutdown = {
                let swarm_agents = self.swarm_agents.read().await;
                swarm_agents.get(&swarm_id)
                    .map(|agents| agents.iter().take(agents_to_remove).cloned().collect::<Vec<_>>())
                    .unwrap_or_default()
            };
            
            for agent_id in agents_to_shutdown {
                self.shutdown_agent(agent_id, true).await?;
            }
        }
        
        let _ = self.event_tx.send(LifecycleEvent::SwarmScaled {
            swarm_id,
            old_count: current_count,
            new_count: target_count,
        });
        
        Ok(())
    }

    /// Get default agent configuration for a swarm
    async fn get_default_agent_config(&self, swarm_id: &str) -> Result<AgentConfig> {
        let executable = match swarm_id {
            "risk-management" => "risk_agent",
            "trading-strategy" => "trading_agent",
            "data-pipeline" => "data_agent",
            "tengri-watchdog" => "watchdog_agent",
            _ => "generic_agent",
        };
        
        Ok(AgentConfig {
            executable: executable.to_string(),
            args: vec!["--swarm".to_string(), swarm_id.to_string()],
            env: HashMap::new(),
            working_dir: None,
            resource_limits: ResourceLimits {
                max_memory: self.config.max_memory_per_agent,
                max_cpu: self.config.max_cpu_per_agent,
                max_network_bps: 100_000_000, // 100 MB/s
                max_disk_iops: 1000,
                max_fds: 1024,
            },
            startup_params: HashMap::new(),
        })
    }

    /// Get agent type for a swarm
    fn get_swarm_agent_type(&self, swarm_id: &str) -> AgentType {
        match swarm_id {
            "risk-management" => AgentType::RiskManager,
            "trading-strategy" => AgentType::TradingStrategy,
            "data-pipeline" => AgentType::DataProcessor,
            "tengri-watchdog" => AgentType::Watchdog,
            _ => AgentType::Generic,
        }
    }

    /// Set resource limits for an agent
    async fn set_resource_limits(&self, agent_id: AgentId, limits: ResourceLimits) -> Result<()> {
        {
            let mut agents = self.agents.write().await;
            if let Some(agent) = agents.get_mut(&agent_id) {
                agent.config.resource_limits = limits;
            }
        }
        
        Ok(())
    }

    /// Update agent configuration
    async fn update_agent_config(&self, agent_id: AgentId, config: AgentConfig) -> Result<()> {
        {
            let mut agents = self.agents.write().await;
            if let Some(agent) = agents.get_mut(&agent_id) {
                agent.config = config;
            }
        }
        
        Ok(())
    }

    /// Perform health check
    async fn perform_health_check(&self, agent_id: Option<AgentId>) -> Result<()> {
        match agent_id {
            Some(id) => {
                if let Some(agent) = self.agents.read().await.get(&id) {
                    let health = self.health_checker.check_agent_health(&agent.id).await;
                    info!("Agent {} health: {:?}", id, health);
                }
            }
            None => {
                // Check all agents
                let agents = self.agents.read().await;
                for (agent_id, _) in agents.iter() {
                    let health = self.health_checker.check_agent_health(agent_id).await;
                    info!("Agent {} health: {:?}", agent_id, health);
                }
            }
        }
        
        Ok(())
    }

    /// Collect metrics
    async fn collect_metrics(&self) -> Result<()> {
        let metrics = self.lifecycle_metrics.read().await;
        info!("Lifecycle Metrics: {} active, {} spawned, {} failed, {} restarts",
              metrics.active_agents, metrics.total_spawned, 
              metrics.total_failed, metrics.total_restarts);
        
        Ok(())
    }

    /// Shutdown all agents
    async fn shutdown_all_agents(&self) -> Result<()> {
        info!("Shutting down all agents");
        
        let agent_ids = {
            let agents = self.agents.read().await;
            agents.keys().cloned().collect::<Vec<_>>()
        };
        
        for agent_id in agent_ids {
            if let Err(e) = self.shutdown_agent(agent_id, true).await {
                error!("Error shutting down agent: {}", e);
            }
        }
        
        Ok(())
    }

    /// Health monitoring task
    async fn health_monitoring_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            let agents = self.agents.read().await;
            
            for (agent_id, agent) in agents.iter() {
                if agent.state == AgentLifecycleState::Running {
                    match self.health_checker.check_agent_health(agent_id).await {
                        HealthStatus::Healthy => {
                            // Update last health check
                            // Note: This would need mutable access in a real implementation
                        }
                        HealthStatus::Unhealthy => {
                            warn!("Agent {} is unhealthy", agent_id);
                            
                            if self.config.auto_restart_failed {
                                if let Err(e) = self.restart_agent(agent_id.clone()).await {
                                    error!("Failed to restart unhealthy agent: {}", e);
                                }
                            }
                        }
                        HealthStatus::Unknown => {
                            // Handle unknown health status
                        }
                    }
                }
            }
            
            sleep(Duration::from_millis(self.config.health_check_interval)).await;
        }
    }

    /// Resource monitoring task
    async fn resource_monitoring_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            let agents = self.agents.read().await;
            
            for (agent_id, agent) in agents.iter() {
                if agent.state == AgentLifecycleState::Running {
                    if let Some(process_id) = agent.process_id {
                        let resource_usage = self.get_process_resource_usage(process_id).await;
                        
                        // Check resource limits
                        if resource_usage.cpu_usage > agent.config.resource_limits.max_cpu {
                            let _ = self.event_tx.send(LifecycleEvent::ResourceLimitExceeded {
                                agent_id: agent_id.clone(),
                                resource_type: "cpu".to_string(),
                                current_value: resource_usage.cpu_usage,
                                limit: agent.config.resource_limits.max_cpu,
                            });
                        }
                        
                        if resource_usage.memory_usage > agent.config.resource_limits.max_memory {
                            let _ = self.event_tx.send(LifecycleEvent::ResourceLimitExceeded {
                                agent_id: agent_id.clone(),
                                resource_type: "memory".to_string(),
                                current_value: resource_usage.memory_usage as f64,
                                limit: agent.config.resource_limits.max_memory as f64,
                            });
                        }
                        
                        // Update resource tracking
                        self.update_resource_tracking(agent_id.clone(), &resource_usage).await;
                    }
                }
            }
            
            sleep(self.resource_monitor.interval).await;
        }
    }

    /// Get process resource usage
    async fn get_process_resource_usage(&self, process_id: u32) -> ResourceUsage {
        // This is a simplified implementation
        // In a real system, you would use system APIs to get actual resource usage
        ResourceUsage {
            cpu_usage: 25.0, // Example
            memory_usage: 500_000_000, // Example
            network_bytes_sent: 1000,
            network_bytes_received: 1500,
            disk_bytes_read: 100,
            disk_bytes_written: 200,
            open_fds: 10,
            thread_count: 5,
        }
    }

    /// Update resource tracking
    async fn update_resource_tracking(&self, agent_id: AgentId, resource_usage: &ResourceUsage) {
        // Update CPU tracking
        {
            let mut cpu_tracker = self.resource_monitor.cpu_tracker.write().await;
            let cpu_history = cpu_tracker.entry(agent_id.clone()).or_insert_with(VecDeque::new);
            cpu_history.push_back(resource_usage.cpu_usage);
            if cpu_history.len() > self.resource_monitor.history_size {
                cpu_history.pop_front();
            }
        }
        
        // Update memory tracking
        {
            let mut memory_tracker = self.resource_monitor.memory_tracker.write().await;
            let memory_history = memory_tracker.entry(agent_id.clone()).or_insert_with(VecDeque::new);
            memory_history.push_back(resource_usage.memory_usage);
            if memory_history.len() > self.resource_monitor.history_size {
                memory_history.pop_front();
            }
        }
        
        // Update network tracking
        {
            let mut network_tracker = self.resource_monitor.network_tracker.write().await;
            let network_history = network_tracker.entry(agent_id).or_insert_with(VecDeque::new);
            network_history.push_back((resource_usage.network_bytes_sent, resource_usage.network_bytes_received));
            if network_history.len() > self.resource_monitor.history_size {
                network_history.pop_front();
            }
        }
    }

    /// Metrics collection task
    async fn metrics_collection_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            self.update_lifecycle_metrics().await;
            sleep(Duration::from_secs(10)).await;
        }
    }

    /// Update lifecycle metrics
    async fn update_lifecycle_metrics(&self) {
        let agents = self.agents.read().await;
        
        let mut total_cpu = 0.0;
        let mut total_memory = 0u64;
        let mut active_count = 0;
        
        for agent in agents.values() {
            if agent.state == AgentLifecycleState::Running {
                total_cpu += agent.resource_usage.cpu_usage;
                total_memory += agent.resource_usage.memory_usage;
                active_count += 1;
            }
        }
        
        let mut metrics = self.lifecycle_metrics.write().await;
        metrics.active_agents = active_count;
        metrics.resource_utilization.total_cpu_usage = total_cpu;
        metrics.resource_utilization.total_memory_usage = total_memory;
        
        if active_count > 0 {
            metrics.resource_utilization.avg_cpu_per_agent = total_cpu / active_count as f64;
            metrics.resource_utilization.avg_memory_per_agent = total_memory / active_count as u64;
        }
    }

    /// Cleanup task
    async fn cleanup_task(&self) {
        while self.running.load(Ordering::SeqCst) {
            // Clean up terminated agents
            let terminated_agents = {
                let agents = self.agents.read().await;
                agents.iter()
                    .filter(|(_, agent)| agent.state == AgentLifecycleState::Terminated)
                    .map(|(id, _)| id.clone())
                    .collect::<Vec<_>>()
            };
            
            for agent_id in terminated_agents {
                let mut agents = self.agents.write().await;
                agents.remove(&agent_id);
            }
            
            sleep(Duration::from_secs(60)).await;
        }
    }

    /// Maintenance cycle
    async fn maintenance_cycle(&self) {
        debug!("Running lifecycle maintenance cycle");
        
        // Update metrics
        self.update_lifecycle_metrics().await;
        
        // Check for zombie processes
        // Implementation would go here
    }

    /// Send command to lifecycle manager
    pub async fn send_command(&self, command: LifecycleCommand) -> Result<()> {
        self.command_tx.send(command)
            .map_err(|e| OrchestrationError::CommunicationError(e.to_string()))?;
        Ok(())
    }

    /// Subscribe to lifecycle events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<LifecycleEvent> {
        self.event_tx.subscribe()
    }

    /// Get current lifecycle metrics
    pub async fn get_lifecycle_metrics(&self) -> LifecycleMetrics {
        self.lifecycle_metrics.read().await.clone()
    }

    /// Check if manager is running
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl Clone for AgentLifecycleManager {
    fn clone(&self) -> Self {
        Self {
            id: self.id.clone(),
            config: self.config.clone(),
            state: Arc::clone(&self.state),
            agents: Arc::clone(&self.agents),
            swarm_agents: Arc::clone(&self.swarm_agents),
            processes: Arc::clone(&self.processes),
            message_router: Arc::clone(&self.message_router),
            health_checker: Arc::clone(&self.health_checker),
            memory_coordinator: Arc::clone(&self.memory_coordinator),
            spawn_semaphore: Arc::clone(&self.spawn_semaphore),
            command_tx: self.command_tx.clone(),
            command_rx: Arc::clone(&self.command_rx),
            event_tx: self.event_tx.clone(),
            shutdown_tx: Arc::clone(&self.shutdown_tx),
            running: Arc::clone(&self.running),
            resource_monitor: Arc::clone(&self.resource_monitor),
            lifecycle_metrics: Arc::clone(&self.lifecycle_metrics),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_lifecycle_manager_creation() {
        let config = LifecycleConfig::default();
        assert_eq!(config.max_concurrent_spawns, 5);
        assert_eq!(config.startup_timeout, 30000);
    }

    #[tokio::test]
    async fn test_agent_instance_creation() {
        let agent_id = AgentId::new();
        let spawn_time = Instant::now();
        
        let agent = AgentInstance {
            id: agent_id,
            agent_type: AgentType::Generic,
            swarm_id: "test-swarm".to_string(),
            state: AgentLifecycleState::Pending,
            process_id: None,
            spawn_time,
            last_health_check: spawn_time,
            resource_usage: ResourceUsage {
                cpu_usage: 0.0,
                memory_usage: 0,
                network_bytes_sent: 0,
                network_bytes_received: 0,
                disk_bytes_read: 0,
                disk_bytes_written: 0,
                open_fds: 0,
                thread_count: 0,
            },
            config: AgentConfig {
                executable: "test_agent".to_string(),
                args: vec![],
                env: HashMap::new(),
                working_dir: None,
                resource_limits: ResourceLimits {
                    max_memory: 1_000_000_000,
                    max_cpu: 50.0,
                    max_network_bps: 100_000_000,
                    max_disk_iops: 1000,
                    max_fds: 1024,
                },
                startup_params: HashMap::new(),
            },
            restart_count: 0,
            health_status: HealthStatus::Unknown,
            performance_metrics: AgentPerformanceMetrics {
                messages_per_second: 0.0,
                avg_response_time_ms: 0.0,
                error_rate: 0.0,
                uptime_seconds: 0,
                last_update: spawn_time,
            },
        };

        assert_eq!(agent.state, AgentLifecycleState::Pending);
        assert_eq!(agent.swarm_id, "test-swarm");
        assert_eq!(agent.restart_count, 0);
    }
}