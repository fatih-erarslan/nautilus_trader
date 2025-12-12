//! Main MCP orchestrator for coordinating all system components.

use crate::agent::{AgentInfo, AgentRegistry};
use crate::communication::{CommunicationLayer, MessageRouter};
use crate::config::OrchestrationConfig;
use crate::coordination::{AdaptiveCoordinationEngine, CoordinationEngine};
use crate::error::{OrchestrationError, Result};
use crate::health::{BasicHealthChecker, HealthMonitor};
use crate::load_balancer::{AdaptiveLoadBalancer, LoadBalancer};
use crate::memory::{MemoryCoordinator, SharedMemory};
use crate::metrics::{InMemoryMetricsCollector, MetricsAggregator, MetricsCollector, OrchestrationMetrics};
use crate::recovery::{AdaptiveRecoveryManager, RecoveryManager};
use crate::task_queue::{PriorityTaskQueue, Task, TaskDistributor, TaskQueue};
use crate::types::{AgentId, AgentType, Timestamp};
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::signal;
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{error, info, warn};

/// MCP orchestrator state
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrchestratorState {
    /// Orchestrator is starting up
    Starting,
    /// Orchestrator is running
    Running,
    /// Orchestrator is stopping
    Stopping,
    /// Orchestrator has stopped
    Stopped,
    /// Orchestrator has failed
    Failed,
}

/// Main MCP orchestrator
#[derive(Debug)]
pub struct McpOrchestrator {
    /// Configuration
    config: OrchestrationConfig,
    /// Orchestrator state
    state: Arc<RwLock<OrchestratorState>>,
    /// Start time
    start_time: Instant,
    
    // Core components
    /// Communication layer
    communication: Arc<dyn CommunicationLayer>,
    /// Agent registry
    agent_registry: Arc<AgentRegistry>,
    /// Task queue
    task_queue: Arc<dyn TaskQueue>,
    /// Task distributor
    task_distributor: Arc<TaskDistributor>,
    /// Load balancer
    load_balancer: Arc<dyn LoadBalancer>,
    /// Shared memory
    shared_memory: Arc<dyn SharedMemory>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
    /// Recovery manager
    recovery_manager: Arc<dyn RecoveryManager>,
    /// Coordination engine
    coordination_engine: Arc<dyn CoordinationEngine>,
    /// Metrics collector
    metrics_collector: Arc<dyn MetricsCollector>,
    /// Metrics aggregator
    metrics_aggregator: Arc<MetricsAggregator>,
    
    /// Event broadcaster for orchestrator events
    event_broadcaster: broadcast::Sender<OrchestratorEvent>,
}

/// Orchestrator events
#[derive(Debug, Clone)]
pub enum OrchestratorEvent {
    /// Orchestrator started
    Started,
    /// Orchestrator ready for operations
    Ready,
    /// Orchestrator stopping
    Stopping,
    /// Orchestrator stopped
    Stopped,
    /// Orchestrator failed
    Failed(String),
    /// Component started
    ComponentStarted(String),
    /// Component failed
    ComponentFailed(String, String),
}

impl McpOrchestrator {
    /// Create a new MCP orchestrator with the given configuration
    fn new(config: OrchestrationConfig) -> Result<Self> {
        let start_time = Instant::now();
        let (event_broadcaster, _) = broadcast::channel(1000);
        
        // Create communication layer
        let communication = Arc::new(MessageRouter::new());
        
        // Create agent registry
        let agent_registry = Arc::new(AgentRegistry::new(Arc::clone(&communication)));
        
        // Create task queue and distributor
        let task_queue = Arc::new(PriorityTaskQueue::new());
        let task_distributor = Arc::new(TaskDistributor::new(
            Arc::clone(&task_queue),
            Arc::clone(&agent_registry),
        ));
        
        // Create load balancer
        let load_balancer = Arc::new(AdaptiveLoadBalancer::new(Arc::clone(&agent_registry)));
        
        // Create shared memory
        let shared_memory = Arc::new(MemoryCoordinator::new(config.memory.cache_size));
        
        // Create health monitor
        let health_monitor = Arc::new(HealthMonitor::new(Arc::clone(&agent_registry)));
        
        // Create recovery manager
        let recovery_manager = Arc::new(AdaptiveRecoveryManager::new(
            config.recovery.circuit_breaker_threshold,
            config.recovery.circuit_breaker_timeout_ms,
            1000, // max history size
        ));
        
        // Create coordination engine
        let coordination_engine = Arc::new(AdaptiveCoordinationEngine::new(
            Arc::clone(&agent_registry),
            Arc::clone(&communication),
            Arc::clone(&task_queue),
            Arc::clone(&task_distributor),
            Arc::clone(&load_balancer),
            Arc::clone(&shared_memory),
            Arc::clone(&health_monitor),
            Arc::clone(&recovery_manager),
        ));
        
        // Create metrics collector and aggregator
        let metrics_collector = Arc::new(InMemoryMetricsCollector::new());
        let metrics_aggregator = Arc::new(MetricsAggregator::new(
            Arc::clone(&metrics_collector),
            config.metrics.enable_prometheus,
            config.metrics_collection_interval(),
        ));
        
        Ok(Self {
            config,
            state: Arc::new(RwLock::new(OrchestratorState::Starting)),
            start_time,
            communication,
            agent_registry,
            task_queue,
            task_distributor,
            load_balancer,
            shared_memory,
            health_monitor,
            recovery_manager,
            coordination_engine,
            metrics_collector,
            metrics_aggregator,
            event_broadcaster,
        })
    }
    
    /// Check if the orchestrator is running
    pub async fn is_running(&self) -> bool {
        matches!(*self.state.read(), OrchestratorState::Running)
    }
    
    /// Get the orchestrator state
    pub async fn get_state(&self) -> OrchestratorState {
        self.state.read().clone()
    }
    
    /// Get system uptime
    pub fn uptime(&self) -> Duration {
        self.start_time.elapsed()
    }
    
    /// Start the orchestrator
    pub async fn start(&self) -> Result<()> {
        info!("Starting MCP Orchestrator...");
        
        *self.state.write() = OrchestratorState::Starting;
        self.broadcast_event(OrchestratorEvent::Started);
        
        // Start core components in order
        self.start_components().await?;
        
        // Start monitoring and management tasks
        self.start_management_tasks().await?;
        
        // Register built-in health checkers
        self.register_health_checkers().await?;
        
        *self.state.write() = OrchestratorState::Running;
        self.broadcast_event(OrchestratorEvent::Ready);
        
        info!("MCP Orchestrator started successfully");
        Ok(())
    }
    
    /// Start all core components
    async fn start_components(&self) -> Result<()> {
        // Start communication layer
        if let Err(e) = self.communication.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "communication".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("communication".to_string()));
        
        // Start agent registry
        if let Err(e) = self.agent_registry.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "agent_registry".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("agent_registry".to_string()));
        
        // Start task queue
        if let Err(e) = self.task_queue.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "task_queue".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("task_queue".to_string()));
        
        // Start task distributor
        if let Err(e) = self.task_distributor.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "task_distributor".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("task_distributor".to_string()));
        
        // Start load balancer
        if let Err(e) = self.load_balancer.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "load_balancer".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("load_balancer".to_string()));
        
        // Start shared memory
        if let Err(e) = self.shared_memory.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "shared_memory".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("shared_memory".to_string()));
        
        // Start health monitor
        if let Err(e) = self.health_monitor.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "health_monitor".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("health_monitor".to_string()));
        
        // Start recovery manager
        if let Err(e) = self.recovery_manager.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "recovery_manager".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("recovery_manager".to_string()));
        
        // Start coordination engine
        if let Err(e) = self.coordination_engine.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "coordination_engine".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("coordination_engine".to_string()));
        
        // Start metrics aggregator
        if let Err(e) = self.metrics_aggregator.start().await {
            self.broadcast_event(OrchestratorEvent::ComponentFailed(
                "metrics_aggregator".to_string(),
                e.to_string(),
            ));
            return Err(e);
        }
        self.broadcast_event(OrchestratorEvent::ComponentStarted("metrics_aggregator".to_string()));
        
        Ok(())
    }
    
    /// Start management and monitoring tasks
    async fn start_management_tasks(&self) -> Result<()> {
        // Start signal handler
        self.start_signal_handler().await?;
        
        // Start periodic maintenance
        self.start_periodic_maintenance().await?;
        
        Ok(())
    }
    
    /// Start signal handler for graceful shutdown
    async fn start_signal_handler(&self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let event_broadcaster = self.event_broadcaster.clone();
        
        tokio::spawn(async move {
            match signal::ctrl_c().await {
                Ok(()) => {
                    warn!("Received SIGINT, initiating graceful shutdown...");
                    *state.write() = OrchestratorState::Stopping;
                    let _ = event_broadcaster.send(OrchestratorEvent::Stopping);
                }
                Err(err) => {
                    error!("Failed to listen for shutdown signal: {}", err);
                }
            }
        });
        
        Ok(())
    }
    
    /// Start periodic maintenance tasks
    async fn start_periodic_maintenance(&self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let metrics_collector = Arc::clone(&self.metrics_collector);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                if !matches!(*state.read(), OrchestratorState::Running) {
                    break;
                }
                
                // Perform periodic maintenance
                Self::perform_maintenance(&metrics_collector).await;
            }
        });
        
        Ok(())
    }
    
    /// Perform periodic maintenance
    async fn perform_maintenance(metrics_collector: &Arc<dyn MetricsCollector>) {
        // Update maintenance metrics
        metrics_collector.record_metric(
            crate::metrics::Metric::counter(
                "orchestrator_maintenance_runs_total",
                "Total number of maintenance runs",
                1,
            )
        );
        
        // Log maintenance completion
        info!("Periodic maintenance completed");
    }
    
    /// Register built-in health checkers
    async fn register_health_checkers(&self) -> Result<()> {
        // Register health checkers for core components
        let components = [
            "communication",
            "agent_registry",
            "task_queue",
            "load_balancer",
            "shared_memory",
            "recovery_manager",
            "coordination_engine",
            "metrics_aggregator",
        ];
        
        for component in &components {
            let checker = Arc::new(BasicHealthChecker::new(component.to_string()));
            self.health_monitor.register_checker(checker).await?;
        }
        
        Ok(())
    }
    
    /// Shutdown the orchestrator gracefully
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down MCP Orchestrator...");
        
        *self.state.write() = OrchestratorState::Stopping;
        self.broadcast_event(OrchestratorEvent::Stopping);
        
        // Stop components in reverse order
        self.stop_components().await?;
        
        *self.state.write() = OrchestratorState::Stopped;
        self.broadcast_event(OrchestratorEvent::Stopped);
        
        info!("MCP Orchestrator shut down successfully");
        Ok(())
    }
    
    /// Stop all components
    async fn stop_components(&self) -> Result<()> {
        // Stop coordination engine first
        if let Err(e) = self.coordination_engine.stop().await {
            warn!("Error stopping coordination engine: {}", e);
        }
        
        // Stop other components
        // Note: Some components may not have explicit stop methods
        // In a real implementation, each would have proper shutdown logic
        
        Ok(())
    }
    
    /// Register a new agent
    pub async fn register_agent(&self, agent_info: AgentInfo) -> Result<()> {
        self.agent_registry.register_agent(agent_info).await
    }
    
    /// Unregister an agent
    pub async fn unregister_agent(&self, agent_id: AgentId) -> Result<()> {
        self.agent_registry.unregister_agent(agent_id).await
    }
    
    /// Submit a task for processing
    pub async fn submit_task(&self, task: Task) -> Result<crate::types::TaskId> {
        self.task_queue.submit_task(task).await
    }
    
    /// Get orchestration metrics
    pub async fn get_metrics(&self) -> Result<OrchestrationMetrics> {
        Ok(self.metrics_aggregator.get_orchestration_metrics())
    }
    
    /// Get system status
    pub async fn get_system_status(&self) -> Result<SystemStatus> {
        let health_status = self.health_monitor.get_system_status().await?;
        let coordination_state = self.coordination_engine.get_state().await?;
        let load_stats = self.load_balancer.get_load_stats().await?;
        let recovery_stats = self.recovery_manager.get_recovery_stats().await?;
        let memory_stats = self.shared_memory.get_memory_stats().await?;
        let task_stats = self.task_queue.get_queue_stats().await?;
        
        Ok(SystemStatus {
            orchestrator_state: self.get_state().await,
            uptime: self.uptime(),
            health_status,
            coordination_state,
            load_stats,
            recovery_stats,
            memory_stats,
            task_stats,
            timestamp: Timestamp::now(),
        })
    }
    
    /// Subscribe to orchestrator events
    pub async fn subscribe_events(&self) -> Result<broadcast::Receiver<OrchestratorEvent>> {
        Ok(self.event_broadcaster.subscribe())
    }
    
    /// Broadcast an orchestrator event
    fn broadcast_event(&self, event: OrchestratorEvent) {
        let _ = self.event_broadcaster.send(event);
    }
}

/// System status information
#[derive(Debug, Clone)]
pub struct SystemStatus {
    /// Orchestrator state
    pub orchestrator_state: OrchestratorState,
    /// System uptime
    pub uptime: Duration,
    /// Health status
    pub health_status: crate::health::SystemHealthStatus,
    /// Coordination state
    pub coordination_state: crate::coordination::CoordinationState,
    /// Load balancing statistics
    pub load_stats: crate::load_balancer::LoadBalancingStats,
    /// Recovery statistics
    pub recovery_stats: crate::recovery::RecoveryStatistics,
    /// Memory statistics
    pub memory_stats: crate::memory::MemoryStatistics,
    /// Task queue statistics
    pub task_stats: crate::task_queue::QueueStatistics,
    /// Status timestamp
    pub timestamp: Timestamp,
}

/// Builder for creating MCP orchestrator instances
#[derive(Debug)]
pub struct OrchestratorBuilder {
    config: Option<OrchestrationConfig>,
}

impl OrchestratorBuilder {
    /// Create a new orchestrator builder
    pub fn new() -> Self {
        Self { config: None }
    }
    
    /// Set the orchestration configuration
    pub fn with_config(mut self, config: OrchestrationConfig) -> Self {
        self.config = Some(config);
        self
    }
    
    /// Build the orchestrator
    pub async fn build(self) -> Result<McpOrchestrator> {
        let config = self.config.unwrap_or_default();
        
        // Validate configuration
        config.validate().map_err(|e| OrchestrationError::config(e))?;
        
        McpOrchestrator::new(config)
    }
}

impl Default for OrchestratorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// Note: Trait implementations are included in their respective modules

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};
    
    #[tokio::test]
    async fn test_orchestrator_builder() {
        let config = OrchestrationConfig::default();
        let orchestrator = OrchestratorBuilder::new()
            .with_config(config)
            .build()
            .await
            .unwrap();
        
        assert_eq!(orchestrator.get_state().await, OrchestratorState::Starting);
    }
    
    #[tokio::test]
    async fn test_orchestrator_lifecycle() {
        let orchestrator = OrchestratorBuilder::new()
            .build()
            .await
            .unwrap();
        
        // Test start
        orchestrator.start().await.unwrap();
        assert_eq!(orchestrator.get_state().await, OrchestratorState::Running);
        assert!(orchestrator.is_running().await);
        
        // Test uptime
        let uptime = orchestrator.uptime();
        assert!(uptime.as_millis() > 0);
        
        // Test shutdown
        orchestrator.shutdown().await.unwrap();
        assert_eq!(orchestrator.get_state().await, OrchestratorState::Stopped);
        assert!(!orchestrator.is_running().await);
    }
    
    #[tokio::test]
    async fn test_orchestrator_events() {
        let orchestrator = OrchestratorBuilder::new()
            .build()
            .await
            .unwrap();
        
        let mut event_receiver = orchestrator.subscribe_events().await.unwrap();
        
        // Start orchestrator
        orchestrator.start().await.unwrap();
        
        // Check for started event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        match event.unwrap().unwrap() {
            OrchestratorEvent::Started => {},
            _ => panic!("Expected Started event"),
        }
        
        // Check for ready event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        // Should receive multiple component started events and then ready
        let mut received_ready = false;
        for _ in 0..20 { // Give enough tries to receive all events
            if let Ok(Ok(event)) = tokio::time::timeout(Duration::from_millis(100), event_receiver.recv()).await {
                if matches!(event, OrchestratorEvent::Ready) {
                    received_ready = true;
                    break;
                }
            } else {
                break;
            }
        }
        
        assert!(received_ready, "Should have received Ready event");
        
        orchestrator.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_orchestrator_agent_management() {
        let orchestrator = OrchestratorBuilder::new()
            .build()
            .await
            .unwrap();
        
        orchestrator.start().await.unwrap();
        
        // Register an agent
        let agent_info = AgentInfo::new(
            AgentId::new(),
            AgentType::Risk,
            "Test Agent".to_string(),
            "1.0.0".to_string(),
        );
        let agent_id = agent_info.id;
        
        orchestrator.register_agent(agent_info).await.unwrap();
        
        // Verify agent was registered
        let agents = orchestrator.agent_registry.get_all_agents().await.unwrap();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].id, agent_id);
        
        // Unregister agent
        orchestrator.unregister_agent(agent_id).await.unwrap();
        
        let agents = orchestrator.agent_registry.get_all_agents().await.unwrap();
        assert_eq!(agents.len(), 0);
        
        orchestrator.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_orchestrator_task_submission() {
        let orchestrator = OrchestratorBuilder::new()
            .build()
            .await
            .unwrap();
        
        orchestrator.start().await.unwrap();
        
        // Submit a task
        let task = Task::new("test_task", crate::types::TaskPriority::High, b"payload".to_vec());
        let task_id = orchestrator.submit_task(task).await.unwrap();
        
        // Verify task was submitted
        let task_status = orchestrator.task_queue.get_task_status(task_id).await.unwrap();
        assert_eq!(task_status, crate::types::TaskStatus::Queued);
        
        orchestrator.shutdown().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_orchestrator_system_status() {
        let orchestrator = OrchestratorBuilder::new()
            .build()
            .await
            .unwrap();
        
        orchestrator.start().await.unwrap();
        
        // Wait a bit for system to stabilize
        sleep(Duration::from_millis(100)).await;
        
        // Get system status
        let status = orchestrator.get_system_status().await.unwrap();
        
        assert_eq!(status.orchestrator_state, OrchestratorState::Running);
        assert!(status.uptime.as_millis() > 0);
        
        orchestrator.shutdown().await.unwrap();
    }
}