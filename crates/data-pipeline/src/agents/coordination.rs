//! # Data Swarm Coordination
//!
//! Coordination engine for data processing agents in the ruv-swarm system.
//! Provides intelligent agent coordination, load balancing, and task distribution.

use std::sync::Arc;
use std::time::{Duration, Instant};
use std::collections::HashMap;
use tokio::sync::{RwLock, mpsc, Mutex, broadcast};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, error, debug};
use uuid::Uuid;

use crate::agents::base::{
    DataAgent, DataAgentId, DataAgentType, DataAgentState, DataMessage, 
    CoordinationMessage, CoordinationType, MessagePriority
};

/// Data swarm coordinator
pub struct DataSwarmCoordinator {
    config: Arc<CoordinationConfig>,
    agents: Arc<RwLock<HashMap<DataAgentId, AgentInfo>>>,
    task_queue: Arc<RwLock<TaskQueue>>,
    load_balancer: Arc<LoadBalancer>,
    message_router: Arc<MessageRouter>,
    coordination_metrics: Arc<RwLock<CoordinationMetrics>>,
    state: Arc<RwLock<CoordinationState>>,
    event_bus: Arc<EventBus>,
}

/// Coordination configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Maximum number of agents
    pub max_agents: usize,
    /// Load balancing strategy
    pub load_balancing_strategy: LoadBalancingStrategy,
    /// Task distribution strategy
    pub task_distribution: TaskDistributionStrategy,
    /// Health check interval
    pub health_check_interval: Duration,
    /// Coordination timeout
    pub coordination_timeout: Duration,
    /// Failover settings
    pub failover_config: FailoverConfig,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            max_agents: 20,
            load_balancing_strategy: LoadBalancingStrategy::WeightedRoundRobin,
            task_distribution: TaskDistributionStrategy::LeastLoaded,
            health_check_interval: Duration::from_secs(30),
            coordination_timeout: Duration::from_secs(10),
            failover_config: FailoverConfig::default(),
            performance_thresholds: PerformanceThresholds::default(),
        }
    }
}

/// Load balancing strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum LoadBalancingStrategy {
    RoundRobin,
    WeightedRoundRobin,
    LeastConnections,
    LeastLoaded,
    Random,
    Hash,
    Adaptive,
}

/// Task distribution strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TaskDistributionStrategy {
    RoundRobin,
    LeastLoaded,
    CapabilityBased,
    PriorityBased,
    Adaptive,
    Predictive,
}

/// Failover configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailoverConfig {
    /// Enable automatic failover
    pub enabled: bool,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Retry delay
    pub retry_delay: Duration,
    /// Circuit breaker threshold
    pub circuit_breaker_threshold: u32,
    /// Recovery strategy
    pub recovery_strategy: RecoveryStrategy,
}

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_retries: 3,
            retry_delay: Duration::from_millis(100),
            circuit_breaker_threshold: 5,
            recovery_strategy: RecoveryStrategy::Gradual,
        }
    }
}

/// Recovery strategies
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Immediate,
    Gradual,
    Manual,
}

/// Performance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum response time in milliseconds
    pub max_response_time_ms: f64,
    /// Minimum success rate
    pub min_success_rate: f64,
    /// Maximum error rate
    pub max_error_rate: f64,
    /// Maximum memory usage percentage
    pub max_memory_usage: f64,
    /// Maximum CPU usage percentage
    pub max_cpu_usage: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            max_response_time_ms: 1000.0,
            min_success_rate: 0.95,
            max_error_rate: 0.05,
            max_memory_usage: 0.80,
            max_cpu_usage: 0.80,
        }
    }
}

/// Agent information for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: DataAgentId,
    pub agent_type: DataAgentType,
    pub state: DataAgentState,
    pub capabilities: Vec<String>,
    pub load_score: f64,
    pub performance_metrics: AgentPerformanceMetrics,
    pub last_heartbeat: chrono::DateTime<chrono::Utc>,
    pub circuit_breaker_state: CircuitBreakerState,
}

/// Agent performance metrics for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentPerformanceMetrics {
    pub response_time_ms: f64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub throughput: f64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub queue_length: usize,
}

impl Default for AgentPerformanceMetrics {
    fn default() -> Self {
        Self {
            response_time_ms: 0.0,
            success_rate: 1.0,
            error_rate: 0.0,
            throughput: 0.0,
            memory_usage: 0.0,
            cpu_usage: 0.0,
            queue_length: 0,
        }
    }
}

/// Circuit breaker state
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CircuitBreakerState {
    Closed,
    Open,
    HalfOpen,
}

/// Task queue for coordination
pub struct TaskQueue {
    tasks: std::collections::VecDeque<Task>,
    priority_queue: std::collections::BinaryHeap<PriorityTask>,
    task_metrics: TaskMetrics,
}

/// Task for agent processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: Uuid,
    pub task_type: TaskType,
    pub priority: MessagePriority,
    pub data: DataMessage,
    pub assigned_agent: Option<DataAgentId>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub retry_count: u32,
}

/// Task types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum TaskType {
    DataIngestion,
    FeatureExtraction,
    DataValidation,
    StreamProcessing,
    DataTransformation,
    CacheOperation,
    HealthCheck,
    Coordination,
}

/// Priority task wrapper for binary heap
#[derive(Debug, Clone)]
pub struct PriorityTask {
    task: Task,
    priority_score: u64,
}

impl PartialEq for PriorityTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for PriorityTask {}

impl PartialOrd for PriorityTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority_score.cmp(&other.priority_score)
    }
}

/// Task metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_completion_time_ms: f64,
    pub queue_length: usize,
    pub throughput: f64,
}

impl Default for TaskMetrics {
    fn default() -> Self {
        Self {
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            average_completion_time_ms: 0.0,
            queue_length: 0,
            throughput: 0.0,
        }
    }
}

/// Load balancer
pub struct LoadBalancer {
    strategy: LoadBalancingStrategy,
    agent_weights: Arc<RwLock<HashMap<DataAgentId, f64>>>,
    round_robin_index: Arc<Mutex<usize>>,
}

impl LoadBalancer {
    pub fn new(strategy: LoadBalancingStrategy) -> Self {
        Self {
            strategy,
            agent_weights: Arc::new(RwLock::new(HashMap::new())),
            round_robin_index: Arc::new(Mutex::new(0)),
        }
    }
    
    pub async fn select_agent(&self, agents: &HashMap<DataAgentId, AgentInfo>, task: &Task) -> Option<DataAgentId> {
        let available_agents: Vec<_> = agents.iter()
            .filter(|(_, info)| matches!(info.state, DataAgentState::Running))
            .filter(|(_, info)| matches!(info.circuit_breaker_state, CircuitBreakerState::Closed | CircuitBreakerState::HalfOpen))
            .collect();
        
        if available_agents.is_empty() {
            return None;
        }
        
        match self.strategy {
            LoadBalancingStrategy::RoundRobin => {
                let mut index = self.round_robin_index.lock().await;
                let selected = available_agents.get(*index % available_agents.len())?;
                *index += 1;
                Some(*selected.0)
            }
            LoadBalancingStrategy::LeastLoaded => {
                available_agents.iter()
                    .min_by(|a, b| a.1.load_score.partial_cmp(&b.1.load_score).unwrap())
                    .map(|(id, _)| **id)
            }
            LoadBalancingStrategy::WeightedRoundRobin => {
                let weights = self.agent_weights.read().await;
                let total_weight: f64 = available_agents.iter()
                    .map(|(id, _)| weights.get(id).unwrap_or(&1.0))
                    .sum();
                
                if total_weight == 0.0 {
                    return available_agents.first().map(|(id, _)| **id);
                }
                
                let mut random_value = rand::random::<f64>() * total_weight;
                
                for (id, _) in &available_agents {
                    let weight = weights.get(id).unwrap_or(&1.0);
                    random_value -= weight;
                    if random_value <= 0.0 {
                        return Some(**id);
                    }
                }
                
                available_agents.first().map(|(id, _)| **id)
            }
            LoadBalancingStrategy::Random => {
                let index = rand::random::<usize>() % available_agents.len();
                available_agents.get(index).map(|(id, _)| **id)
            }
            _ => {
                // Default to round robin for other strategies
                let mut index = self.round_robin_index.lock().await;
                let selected = available_agents.get(*index % available_agents.len())?;
                *index += 1;
                Some(*selected.0)
            }
        }
    }
    
    pub async fn update_agent_weight(&self, agent_id: DataAgentId, weight: f64) {
        self.agent_weights.write().await.insert(agent_id, weight);
    }
}

/// Message router for agent communication
pub struct MessageRouter {
    routes: Arc<RwLock<HashMap<DataAgentId, mpsc::UnboundedSender<DataMessage>>>>,
    broadcast_tx: broadcast::Sender<DataMessage>,
}

impl MessageRouter {
    pub fn new() -> Self {
        let (broadcast_tx, _) = broadcast::channel(1000);
        
        Self {
            routes: Arc::new(RwLock::new(HashMap::new())),
            broadcast_tx,
        }
    }
    
    pub async fn register_agent(&self, agent_id: DataAgentId, sender: mpsc::UnboundedSender<DataMessage>) {
        self.routes.write().await.insert(agent_id, sender);
    }
    
    pub async fn unregister_agent(&self, agent_id: DataAgentId) {
        self.routes.write().await.remove(&agent_id);
    }
    
    pub async fn route_message(&self, message: DataMessage) -> Result<()> {
        if let Some(destination) = message.destination {
            let routes = self.routes.read().await;
            if let Some(sender) = routes.get(&destination) {
                sender.send(message)?;
            }
        } else {
            // Broadcast message
            self.broadcast_tx.send(message)?;
        }
        
        Ok(())
    }
    
    pub fn subscribe(&self) -> broadcast::Receiver<DataMessage> {
        self.broadcast_tx.subscribe()
    }
}

/// Event bus for coordination events
pub struct EventBus {
    event_tx: broadcast::Sender<CoordinationEvent>,
}

/// Coordination events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEvent {
    AgentRegistered(DataAgentId),
    AgentUnregistered(DataAgentId),
    AgentStateChanged(DataAgentId, DataAgentState),
    TaskAssigned(Uuid, DataAgentId),
    TaskCompleted(Uuid, DataAgentId),
    TaskFailed(Uuid, DataAgentId, String),
    LoadBalancingUpdate,
    HealthCheckFailed(DataAgentId),
    CircuitBreakerTripped(DataAgentId),
}

impl EventBus {
    pub fn new() -> Self {
        let (event_tx, _) = broadcast::channel(1000);
        
        Self { event_tx }
    }
    
    pub fn publish(&self, event: CoordinationEvent) -> Result<()> {
        self.event_tx.send(event)?;
        Ok(())
    }
    
    pub fn subscribe(&self) -> broadcast::Receiver<CoordinationEvent> {
        self.event_tx.subscribe()
    }
}

/// Coordination metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationMetrics {
    pub active_agents: usize,
    pub total_tasks: u64,
    pub completed_tasks: u64,
    pub failed_tasks: u64,
    pub average_task_completion_time_ms: f64,
    pub throughput_tasks_per_sec: f64,
    pub load_balancing_efficiency: f64,
    pub failover_events: u64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for CoordinationMetrics {
    fn default() -> Self {
        Self {
            active_agents: 0,
            total_tasks: 0,
            completed_tasks: 0,
            failed_tasks: 0,
            average_task_completion_time_ms: 0.0,
            throughput_tasks_per_sec: 0.0,
            load_balancing_efficiency: 1.0,
            failover_events: 0,
            last_update: chrono::Utc::now(),
        }
    }
}

/// Coordination state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationState {
    pub coordinator_active: bool,
    pub agents_healthy: usize,
    pub agents_degraded: usize,
    pub agents_failed: usize,
    pub task_queue_length: usize,
    pub coordination_latency_ms: f64,
    pub last_health_check: chrono::DateTime<chrono::Utc>,
}

impl Default for CoordinationState {
    fn default() -> Self {
        Self {
            coordinator_active: false,
            agents_healthy: 0,
            agents_degraded: 0,
            agents_failed: 0,
            task_queue_length: 0,
            coordination_latency_ms: 0.0,
            last_health_check: chrono::Utc::now(),
        }
    }
}

/// Data swarm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSwarmState {
    pub swarm_active: bool,
    pub agents_deployed: usize,
    pub coordination_healthy: bool,
    pub total_throughput: f64,
    pub average_latency_us: f64,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl Default for DataSwarmState {
    fn default() -> Self {
        Self {
            swarm_active: false,
            agents_deployed: 0,
            coordination_healthy: true,
            total_throughput: 0.0,
            average_latency_us: 0.0,
            last_update: chrono::Utc::now(),
        }
    }
}

impl TaskQueue {
    pub fn new() -> Self {
        Self {
            tasks: std::collections::VecDeque::new(),
            priority_queue: std::collections::BinaryHeap::new(),
            task_metrics: TaskMetrics::default(),
        }
    }
    
    pub fn enqueue_task(&mut self, task: Task) {
        let priority_score = match task.priority {
            MessagePriority::Critical => 1000,
            MessagePriority::High => 100,
            MessagePriority::Normal => 10,
            MessagePriority::Low => 1,
        };
        
        self.priority_queue.push(PriorityTask {
            task: task.clone(),
            priority_score,
        });
        
        self.task_metrics.total_tasks += 1;
        self.task_metrics.queue_length = self.priority_queue.len();
    }
    
    pub fn dequeue_task(&mut self) -> Option<Task> {
        if let Some(priority_task) = self.priority_queue.pop() {
            self.task_metrics.queue_length = self.priority_queue.len();
            Some(priority_task.task)
        } else {
            None
        }
    }
    
    pub fn get_metrics(&self) -> &TaskMetrics {
        &self.task_metrics
    }
    
    pub fn complete_task(&mut self, _task_id: Uuid, completion_time_ms: f64) {
        self.task_metrics.completed_tasks += 1;
        self.task_metrics.average_completion_time_ms = 
            (self.task_metrics.average_completion_time_ms + completion_time_ms) / 2.0;
    }
    
    pub fn fail_task(&mut self, _task_id: Uuid) {
        self.task_metrics.failed_tasks += 1;
    }
}

impl DataSwarmCoordinator {
    /// Create a new data swarm coordinator
    pub async fn new(config: Arc<crate::agents::DataSwarmConfig>) -> Result<Self> {
        let coordination_config = CoordinationConfig::default();
        let config = Arc::new(coordination_config);
        let agents = Arc::new(RwLock::new(HashMap::new()));
        let task_queue = Arc::new(RwLock::new(TaskQueue::new()));
        let load_balancer = Arc::new(LoadBalancer::new(config.load_balancing_strategy));
        let message_router = Arc::new(MessageRouter::new());
        let coordination_metrics = Arc::new(RwLock::new(CoordinationMetrics::default()));
        let state = Arc::new(RwLock::new(CoordinationState::default()));
        let event_bus = Arc::new(EventBus::new());
        
        Ok(Self {
            config,
            agents,
            task_queue,
            load_balancer,
            message_router,
            coordination_metrics,
            state,
            event_bus,
        })
    }
    
    /// Start coordination
    pub async fn start(&self) -> Result<()> {
        info!("Starting data swarm coordinator");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.coordinator_active = true;
            state.last_health_check = chrono::Utc::now();
        }
        
        // Start background tasks
        self.start_health_check_task().await;
        self.start_load_balancing_task().await;
        
        info!("Data swarm coordinator started successfully");
        Ok(())
    }
    
    /// Stop coordination
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping data swarm coordinator");
        
        // Update state
        {
            let mut state = self.state.write().await;
            state.coordinator_active = false;
        }
        
        info!("Data swarm coordinator stopped successfully");
        Ok(())
    }
    
    /// Register agent
    pub async fn register_agent(&self, agent_id: DataAgentId, agent_type: DataAgentType, capabilities: Vec<String>) -> Result<()> {
        let agent_info = AgentInfo {
            id: agent_id,
            agent_type,
            state: DataAgentState::Ready,
            capabilities,
            load_score: 0.0,
            performance_metrics: AgentPerformanceMetrics::default(),
            last_heartbeat: chrono::Utc::now(),
            circuit_breaker_state: CircuitBreakerState::Closed,
        };
        
        self.agents.write().await.insert(agent_id, agent_info);
        
        // Publish event
        self.event_bus.publish(CoordinationEvent::AgentRegistered(agent_id))?;
        
        info!("Agent {} registered with type {:?}", agent_id, agent_type);
        Ok(())
    }
    
    /// Unregister agent
    pub async fn unregister_agent(&self, agent_id: DataAgentId) -> Result<()> {
        self.agents.write().await.remove(&agent_id);
        self.message_router.unregister_agent(agent_id).await;
        
        // Publish event
        self.event_bus.publish(CoordinationEvent::AgentUnregistered(agent_id))?;
        
        info!("Agent {} unregistered", agent_id);
        Ok(())
    }
    
    /// Submit task for processing
    pub async fn submit_task(&self, task_type: TaskType, priority: MessagePriority, data: DataMessage) -> Result<Uuid> {
        let task = Task {
            id: Uuid::new_v4(),
            task_type,
            priority,
            data,
            assigned_agent: None,
            created_at: chrono::Utc::now(),
            deadline: None,
            retry_count: 0,
        };
        
        let task_id = task.id;
        
        // Enqueue task
        self.task_queue.write().await.enqueue_task(task);
        
        // Try to assign task immediately
        self.process_task_queue().await?;
        
        Ok(task_id)
    }
    
    /// Process task queue
    async fn process_task_queue(&self) -> Result<()> {
        let task = {
            let mut queue = self.task_queue.write().await;
            queue.dequeue_task()
        };
        
        if let Some(mut task) = task {
            let agents = self.agents.read().await;
            
            if let Some(agent_id) = self.load_balancer.select_agent(&agents, &task).await {
                task.assigned_agent = Some(agent_id);
                
                // Route task to agent
                self.message_router.route_message(task.data.clone()).await?;
                
                // Publish event
                self.event_bus.publish(CoordinationEvent::TaskAssigned(task.id, agent_id))?;
                
                debug!("Task {} assigned to agent {}", task.id, agent_id);
            } else {
                // No available agents, re-queue task
                self.task_queue.write().await.enqueue_task(task);
                warn!("No available agents for task processing");
            }
        }
        
        Ok(())
    }
    
    /// Start health check background task
    async fn start_health_check_task(&self) {
        let agents = self.agents.clone();
        let event_bus = self.event_bus.clone();
        let interval = self.config.health_check_interval;
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                let agent_ids: Vec<_> = {
                    let agents_guard = agents.read().await;
                    agents_guard.keys().copied().collect()
                };
                
                for agent_id in agent_ids {
                    // Check agent health (simplified)
                    let is_healthy = true; // Would implement actual health check
                    
                    if !is_healthy {
                        if let Err(e) = event_bus.publish(CoordinationEvent::HealthCheckFailed(agent_id)) {
                            error!("Failed to publish health check failed event: {}", e);
                        }
                    }
                }
            }
        });
    }
    
    /// Start load balancing background task
    async fn start_load_balancing_task(&self) {
        let agents = self.agents.clone();
        let load_balancer = self.load_balancer.clone();
        let event_bus = self.event_bus.clone();
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval_timer.tick().await;
                
                // Update agent weights based on performance
                {
                    let agents_guard = agents.read().await;
                    for (agent_id, info) in agents_guard.iter() {
                        let weight = Self::calculate_agent_weight(&info.performance_metrics);
                        load_balancer.update_agent_weight(*agent_id, weight).await;
                    }
                }
                
                if let Err(e) = event_bus.publish(CoordinationEvent::LoadBalancingUpdate) {
                    error!("Failed to publish load balancing update event: {}", e);
                }
            }
        });
    }
    
    /// Calculate agent weight based on performance metrics
    fn calculate_agent_weight(metrics: &AgentPerformanceMetrics) -> f64 {
        let response_factor = 1.0 / (1.0 + metrics.response_time_ms / 1000.0);
        let success_factor = metrics.success_rate;
        let load_factor = 1.0 / (1.0 + metrics.queue_length as f64 / 100.0);
        
        response_factor * success_factor * load_factor
    }
    
    /// Get coordination metrics
    pub async fn get_metrics(&self) -> Result<crate::agents::DataSwarmMetrics> {
        let coordination_metrics = self.coordination_metrics.read().await;
        let agents = self.agents.read().await;
        
        Ok(crate::agents::DataSwarmMetrics {
            total_agents: agents.len(),
            active_agents: agents.values()
                .filter(|info| matches!(info.state, DataAgentState::Running))
                .count(),
            messages_processed: coordination_metrics.completed_tasks,
            average_latency_us: coordination_metrics.average_task_completion_time_ms * 1000.0,
            throughput_ops_per_sec: coordination_metrics.throughput_tasks_per_sec,
            error_rate: coordination_metrics.failed_tasks as f64 / coordination_metrics.total_tasks.max(1) as f64,
            memory_usage_mb: 0.0, // Would be calculated
            cpu_usage_percent: 0.0, // Would be calculated
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    
    #[test]
    async fn test_coordinator_creation() {
        let config = Arc::new(crate::agents::DataSwarmConfig::default());
        let coordinator = DataSwarmCoordinator::new(config).await;
        assert!(coordinator.is_ok());
    }
    
    #[test]
    async fn test_agent_registration() {
        let config = Arc::new(crate::agents::DataSwarmConfig::default());
        let coordinator = DataSwarmCoordinator::new(config).await.unwrap();
        
        let agent_id = DataAgentId::new_v4();
        let result = coordinator.register_agent(
            agent_id,
            DataAgentType::DataIngestion,
            vec!["market_data".to_string()]
        ).await;
        
        assert!(result.is_ok());
        
        let agents = coordinator.agents.read().await;
        assert!(agents.contains_key(&agent_id));
    }
    
    #[test]
    async fn test_load_balancer() {
        let load_balancer = LoadBalancer::new(LoadBalancingStrategy::LeastLoaded);
        
        let mut agents = HashMap::new();
        let agent1 = DataAgentId::new_v4();
        let agent2 = DataAgentId::new_v4();
        
        agents.insert(agent1, AgentInfo {
            id: agent1,
            agent_type: DataAgentType::DataIngestion,
            state: DataAgentState::Running,
            capabilities: Vec::new(),
            load_score: 0.5,
            performance_metrics: AgentPerformanceMetrics::default(),
            last_heartbeat: chrono::Utc::now(),
            circuit_breaker_state: CircuitBreakerState::Closed,
        });
        
        agents.insert(agent2, AgentInfo {
            id: agent2,
            agent_type: DataAgentType::FeatureEngineering,
            state: DataAgentState::Running,
            capabilities: Vec::new(),
            load_score: 0.3,
            performance_metrics: AgentPerformanceMetrics::default(),
            last_heartbeat: chrono::Utc::now(),
            circuit_breaker_state: CircuitBreakerState::Closed,
        });
        
        let task = Task {
            id: Uuid::new_v4(),
            task_type: TaskType::DataIngestion,
            priority: MessagePriority::Normal,
            data: DataMessage {
                id: Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                source: agent1,
                destination: None,
                message_type: crate::agents::base::DataMessageType::MarketData,
                payload: serde_json::json!({}),
                metadata: crate::agents::base::MessageMetadata {
                    priority: MessagePriority::Normal,
                    expires_at: None,
                    retry_count: 0,
                    trace_id: "test".to_string(),
                    span_id: "test".to_string(),
                },
            },
            assigned_agent: None,
            created_at: chrono::Utc::now(),
            deadline: None,
            retry_count: 0,
        };
        
        let selected = load_balancer.select_agent(&agents, &task).await;
        assert!(selected.is_some());
        assert_eq!(selected.unwrap(), agent2); // Should select agent with lower load
    }
    
    #[test]
    async fn test_task_queue() {
        let mut queue = TaskQueue::new();
        
        let task = Task {
            id: Uuid::new_v4(),
            task_type: TaskType::DataIngestion,
            priority: MessagePriority::High,
            data: DataMessage {
                id: Uuid::new_v4(),
                timestamp: chrono::Utc::now(),
                source: DataAgentId::new_v4(),
                destination: None,
                message_type: crate::agents::base::DataMessageType::MarketData,
                payload: serde_json::json!({}),
                metadata: crate::agents::base::MessageMetadata {
                    priority: MessagePriority::High,
                    expires_at: None,
                    retry_count: 0,
                    trace_id: "test".to_string(),
                    span_id: "test".to_string(),
                },
            },
            assigned_agent: None,
            created_at: chrono::Utc::now(),
            deadline: None,
            retry_count: 0,
        };
        
        queue.enqueue_task(task.clone());
        assert_eq!(queue.get_metrics().queue_length, 1);
        
        let dequeued = queue.dequeue_task();
        assert!(dequeued.is_some());
        assert_eq!(dequeued.unwrap().id, task.id);
        assert_eq!(queue.get_metrics().queue_length, 0);
    }
}