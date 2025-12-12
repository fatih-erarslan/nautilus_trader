//! Coordination engine for MCP orchestration system.

use crate::agent::{Agent, AgentInfo, AgentRegistry};
use crate::message_router::{MessageRouter, MCPMessage as Message, MCPMessageType as MessageType};
use crate::health_monitoring::{HealthMonitor};
use crate::load_balancing::LoadBalancer;
use crate::deployment::DeploymentManager;
use crate::error::{OrchestrationError, Result};
use crate::types::{HealthStatus};
use crate::types::{AgentId, AgentState, AgentType, TaskId, Timestamp};
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::broadcast;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

/// Coordination state for the orchestration system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationState {
    /// Current coordination mode
    pub mode: CoordinationMode,
    /// Active coordination sessions
    pub active_sessions: HashMap<String, CoordinationSession>,
    /// System readiness status
    pub system_ready: bool,
    /// Last coordination update
    pub last_update: Timestamp,
    /// Coordination statistics
    pub stats: CoordinationStats,
}

/// Coordination modes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationMode {
    /// Centralized coordination (single coordinator)
    Centralized,
    /// Distributed coordination (peer-to-peer)
    Distributed,
    /// Hierarchical coordination (tree structure)
    Hierarchical,
    /// Mesh coordination (full connectivity)
    Mesh,
    /// Hybrid coordination (adaptive)
    Hybrid,
}

/// Coordination session information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationSession {
    /// Session ID
    pub session_id: String,
    /// Session type
    pub session_type: CoordinationSessionType,
    /// Participating agents
    pub participants: Vec<AgentId>,
    /// Session coordinator
    pub coordinator: Option<AgentId>,
    /// Session start time
    pub started_at: Timestamp,
    /// Session status
    pub status: CoordinationSessionStatus,
    /// Session metadata
    pub metadata: HashMap<String, String>,
}

/// Coordination session types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationSessionType {
    /// Task assignment coordination
    TaskAssignment,
    /// Resource allocation coordination
    ResourceAllocation,
    /// State synchronization
    StateSynchronization,
    /// Emergency coordination
    Emergency,
    /// System shutdown coordination
    Shutdown,
}

/// Coordination session status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationSessionStatus {
    /// Session is active
    Active,
    /// Session is paused
    Paused,
    /// Session completed successfully
    Completed,
    /// Session failed
    Failed,
    /// Session was cancelled
    Cancelled,
}

/// Coordination statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStats {
    /// Total coordination sessions
    pub total_sessions: u64,
    /// Active sessions
    pub active_sessions: u64,
    /// Successful sessions
    pub successful_sessions: u64,
    /// Failed sessions
    pub failed_sessions: u64,
    /// Average session duration
    pub avg_session_duration: f64,
    /// Coordination efficiency score
    pub efficiency_score: f64,
}

/// Coordination event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinationEvent {
    /// System started
    SystemStarted {
        timestamp: Timestamp,
    },
    /// System ready for operations
    SystemReady {
        timestamp: Timestamp,
    },
    /// Coordination mode changed
    ModeChanged {
        old_mode: CoordinationMode,
        new_mode: CoordinationMode,
        timestamp: Timestamp,
    },
    /// Coordination session started
    SessionStarted {
        session_id: String,
        session_type: CoordinationSessionType,
        participants: Vec<AgentId>,
        timestamp: Timestamp,
    },
    /// Coordination session completed
    SessionCompleted {
        session_id: String,
        status: CoordinationSessionStatus,
        duration_ms: u64,
        timestamp: Timestamp,
    },
    /// Agent joined coordination
    AgentJoined {
        agent_id: AgentId,
        session_id: String,
        timestamp: Timestamp,
    },
    /// Agent left coordination
    AgentLeft {
        agent_id: AgentId,
        session_id: String,
        timestamp: Timestamp,
    },
}

/// Coordination engine trait
#[async_trait]
pub trait CoordinationEngine: Send + Sync {
    /// Start coordination engine
    async fn start(&self) -> Result<()>;
    
    /// Stop coordination engine
    async fn stop(&self) -> Result<()>;
    
    /// Get coordination state
    async fn get_state(&self) -> Result<CoordinationState>;
    
    /// Set coordination mode
    async fn set_mode(&self, mode: CoordinationMode) -> Result<()>;
    
    /// Start a coordination session
    async fn start_session(
        &self,
        session_type: CoordinationSessionType,
        participants: Vec<AgentId>,
    ) -> Result<String>;
    
    /// End a coordination session
    async fn end_session(&self, session_id: String) -> Result<()>;
    
    /// Join a coordination session
    async fn join_session(&self, session_id: String, agent_id: AgentId) -> Result<()>;
    
    /// Leave a coordination session
    async fn leave_session(&self, session_id: String, agent_id: AgentId) -> Result<()>;
    
    /// Coordinate task assignment
    async fn coordinate_task_assignment(&self, task: Task) -> Result<Option<AgentId>>;
    
    /// Coordinate resource allocation
    async fn coordinate_resource_allocation(&self, resource_request: ResourceRequest) -> Result<ResourceAllocation>;
    
    /// Coordinate system state synchronization
    async fn coordinate_state_sync(&self) -> Result<()>;
    
    /// Subscribe to coordination events
    async fn subscribe_events(&self) -> Result<broadcast::Receiver<CoordinationEvent>>;
}

/// Resource request for coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// Requester agent ID
    pub requester: AgentId,
    /// Resource type
    pub resource_type: String,
    /// Resource amount requested
    pub amount: u64,
    /// Request priority
    pub priority: u8,
    /// Request deadline
    pub deadline: Option<Timestamp>,
}

/// Resource allocation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAllocation {
    /// Request ID
    pub request_id: String,
    /// Allocated amount
    pub allocated_amount: u64,
    /// Resource provider
    pub provider: Option<AgentId>,
    /// Allocation success
    pub success: bool,
    /// Allocation message
    pub message: Option<String>,
}

/// Adaptive coordination engine implementation
#[derive(Debug)]
pub struct AdaptiveCoordinationEngine {
    /// Agent registry
    agent_registry: Arc<AgentRegistry>,
    /// Communication layer
    communication: Arc<dyn CommunicationLayer>,
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
    /// Coordination state
    state: Arc<RwLock<CoordinationState>>,
    /// Event broadcaster
    event_broadcaster: broadcast::Sender<CoordinationEvent>,
    /// Running flag
    running: Arc<RwLock<bool>>,
}

impl AdaptiveCoordinationEngine {
    /// Create a new adaptive coordination engine
    pub fn new(
        agent_registry: Arc<AgentRegistry>,
        communication: Arc<dyn CommunicationLayer>,
        task_queue: Arc<dyn TaskQueue>,
        task_distributor: Arc<TaskDistributor>,
        load_balancer: Arc<dyn LoadBalancer>,
        shared_memory: Arc<dyn SharedMemory>,
        health_monitor: Arc<HealthMonitor>,
        recovery_manager: Arc<dyn RecoveryManager>,
    ) -> Self {
        let (event_broadcaster, _) = broadcast::channel(1000);
        
        Self {
            agent_registry,
            communication,
            task_queue,
            task_distributor,
            load_balancer,
            shared_memory,
            health_monitor,
            recovery_manager,
            state: Arc::new(RwLock::new(CoordinationState {
                mode: CoordinationMode::Hybrid,
                active_sessions: HashMap::new(),
                system_ready: false,
                last_update: Timestamp::now(),
                stats: CoordinationStats {
                    total_sessions: 0,
                    active_sessions: 0,
                    successful_sessions: 0,
                    failed_sessions: 0,
                    avg_session_duration: 0.0,
                    efficiency_score: 0.0,
                },
            })),
            event_broadcaster,
            running: Arc::new(RwLock::new(false)),
        }
    }
    
    /// Start coordination monitoring
    async fn start_monitoring(&self) -> Result<()> {
        // Start system readiness monitoring
        self.start_readiness_monitoring().await?;
        
        // Start coordination session management
        self.start_session_management().await?;
        
        // Start adaptive mode selection
        self.start_adaptive_mode_selection().await?;
        
        Ok(())
    }
    
    /// Start system readiness monitoring
    async fn start_readiness_monitoring(&self) -> Result<()> {
        let agent_registry = Arc::clone(&self.agent_registry);
        let health_monitor = Arc::clone(&self.health_monitor);
        let state = Arc::clone(&self.state);
        let event_broadcaster = self.event_broadcaster.clone();
        let running = Arc::clone(&self.running);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                if !*running.read() {
                    break;
                }
                
                // Check system readiness
                let is_ready = Self::check_system_readiness(&agent_registry, &health_monitor).await;
                
                let mut state = state.write();
                let was_ready = state.system_ready;
                state.system_ready = is_ready;
                state.last_update = Timestamp::now();
                
                // Broadcast readiness event
                if !was_ready && is_ready {
                    let _ = event_broadcaster.send(CoordinationEvent::SystemReady {
                        timestamp: Timestamp::now(),
                    });
                    info!("System is ready for coordination");
                }
            }
        });
        
        Ok(())
    }
    
    /// Check if the system is ready for coordination
    async fn check_system_readiness(
        agent_registry: &Arc<AgentRegistry>,
        health_monitor: &Arc<HealthMonitor>,
    ) -> bool {
        // Check if we have enough healthy agents
        if let Ok(healthy_agents) = agent_registry.get_healthy_agents().await {
            if healthy_agents.len() < 2 {
                return false;
            }
        } else {
            return false;
        }
        
        // Check system health
        if let Ok(system_status) = health_monitor.get_system_status().await {
            if system_status.overall_status != HealthStatus::Healthy {
                return false;
            }
        } else {
            return false;
        }
        
        true
    }
    
    /// Start coordination session management
    async fn start_session_management(&self) -> Result<()> {
        let state = Arc::clone(&self.state);
        let event_broadcaster = self.event_broadcaster.clone();
        let running = Arc::clone(&self.running);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                if !*running.read() {
                    break;
                }
                
                // Clean up completed sessions
                let mut state = state.write();
                let mut sessions_to_remove = Vec::new();
                
                for (session_id, session) in &state.active_sessions {
                    if matches!(session.status, CoordinationSessionStatus::Completed | 
                                              CoordinationSessionStatus::Failed | 
                                              CoordinationSessionStatus::Cancelled) {
                        sessions_to_remove.push(session_id.clone());
                    }
                }
                
                for session_id in sessions_to_remove {
                    if let Some(session) = state.active_sessions.remove(&session_id) {
                        let duration = session.started_at.elapsed().as_millis() as u64;
                        
                        // Update statistics
                        if session.status == CoordinationSessionStatus::Completed {
                            state.stats.successful_sessions += 1;
                        } else {
                            state.stats.failed_sessions += 1;
                        }
                        
                        // Update average duration
                        let total_completed = state.stats.successful_sessions + state.stats.failed_sessions;
                        let current_avg = state.stats.avg_session_duration;
                        state.stats.avg_session_duration = 
                            (current_avg * (total_completed - 1) as f64 + duration as f64) / total_completed as f64;
                        
                        // Broadcast completion event
                        let _ = event_broadcaster.send(CoordinationEvent::SessionCompleted {
                            session_id,
                            status: session.status,
                            duration_ms: duration,
                            timestamp: Timestamp::now(),
                        });
                    }
                }
                
                state.stats.active_sessions = state.active_sessions.len() as u64;
            }
        });
        
        Ok(())
    }
    
    /// Start adaptive mode selection
    async fn start_adaptive_mode_selection(&self) -> Result<()> {
        let agent_registry = Arc::clone(&self.agent_registry);
        let health_monitor = Arc::clone(&self.health_monitor);
        let state = Arc::clone(&self.state);
        let event_broadcaster = self.event_broadcaster.clone();
        let running = Arc::clone(&self.running);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                if !*running.read() {
                    break;
                }
                
                // Determine optimal coordination mode
                let optimal_mode = Self::determine_optimal_mode(&agent_registry, &health_monitor).await;
                
                let mut state = state.write();
                let current_mode = state.mode;
                
                if optimal_mode != current_mode {
                    state.mode = optimal_mode;
                    state.last_update = Timestamp::now();
                    
                    // Broadcast mode change event
                    let _ = event_broadcaster.send(CoordinationEvent::ModeChanged {
                        old_mode: current_mode,
                        new_mode: optimal_mode,
                        timestamp: Timestamp::now(),
                    });
                    
                    info!("Coordination mode changed from {:?} to {:?}", current_mode, optimal_mode);
                }
            }
        });
        
        Ok(())
    }
    
    /// Determine optimal coordination mode based on system state
    async fn determine_optimal_mode(
        agent_registry: &Arc<AgentRegistry>,
        health_monitor: &Arc<HealthMonitor>,
    ) -> CoordinationMode {
        // Get agent count
        let agent_count = agent_registry.get_all_agents().await
            .map(|agents| agents.len())
            .unwrap_or(0);
        
        // Get system health
        let system_health = health_monitor.get_system_status().await
            .map(|status| status.overall_status)
            .unwrap_or(HealthStatus::Unknown);
        
        // Simple heuristics for mode selection
        match (agent_count, system_health) {
            (0..=2, _) => CoordinationMode::Centralized,
            (3..=5, HealthStatus::Healthy) => CoordinationMode::Distributed,
            (6..=10, HealthStatus::Healthy) => CoordinationMode::Hierarchical,
            (11..=20, HealthStatus::Healthy) => CoordinationMode::Mesh,
            (_, HealthStatus::Degraded) => CoordinationMode::Centralized, // Fallback to simple mode
            (_, HealthStatus::Unhealthy) => CoordinationMode::Centralized, // Emergency fallback
            _ => CoordinationMode::Hybrid, // Default adaptive mode
        }
    }
}

#[async_trait]
impl CoordinationEngine for AdaptiveCoordinationEngine {
    async fn start(&self) -> Result<()> {
        *self.running.write() = true;
        
        // Start monitoring tasks
        self.start_monitoring().await?;
        
        // Broadcast system started event
        let _ = self.event_broadcaster.send(CoordinationEvent::SystemStarted {
            timestamp: Timestamp::now(),
        });
        
        info!("Adaptive coordination engine started successfully");
        Ok(())
    }
    
    async fn stop(&self) -> Result<()> {
        *self.running.write() = false;
        
        // Cancel all active sessions
        let mut state = self.state.write();
        for (_, mut session) in state.active_sessions.iter_mut() {
            session.status = CoordinationSessionStatus::Cancelled;
        }
        
        info!("Adaptive coordination engine stopped");
        Ok(())
    }
    
    async fn get_state(&self) -> Result<CoordinationState> {
        Ok(self.state.read().clone())
    }
    
    async fn set_mode(&self, mode: CoordinationMode) -> Result<()> {
        let mut state = self.state.write();
        let old_mode = state.mode;
        state.mode = mode;
        state.last_update = Timestamp::now();
        
        // Broadcast mode change event
        let _ = self.event_broadcaster.send(CoordinationEvent::ModeChanged {
            old_mode,
            new_mode: mode,
            timestamp: Timestamp::now(),
        });
        
        info!("Coordination mode manually set to {:?}", mode);
        Ok(())
    }
    
    async fn start_session(
        &self,
        session_type: CoordinationSessionType,
        participants: Vec<AgentId>,
    ) -> Result<String> {
        let session_id = uuid::Uuid::new_v4().to_string();
        
        let session = CoordinationSession {
            session_id: session_id.clone(),
            session_type: session_type.clone(),
            participants: participants.clone(),
            coordinator: participants.first().copied(),
            started_at: Timestamp::now(),
            status: CoordinationSessionStatus::Active,
            metadata: HashMap::new(),
        };
        
        // Add session to state
        let mut state = self.state.write();
        state.active_sessions.insert(session_id.clone(), session);
        state.stats.total_sessions += 1;
        state.stats.active_sessions = state.active_sessions.len() as u64;
        state.last_update = Timestamp::now();
        
        // Broadcast session started event
        let _ = self.event_broadcaster.send(CoordinationEvent::SessionStarted {
            session_id: session_id.clone(),
            session_type,
            participants,
            timestamp: Timestamp::now(),
        });
        
        debug!("Coordination session {} started", session_id);
        Ok(session_id)
    }
    
    async fn end_session(&self, session_id: String) -> Result<()> {
        let mut state = self.state.write();
        
        if let Some(mut session) = state.active_sessions.get_mut(&session_id) {
            session.status = CoordinationSessionStatus::Completed;
            state.last_update = Timestamp::now();
            
            debug!("Coordination session {} ended", session_id);
            Ok(())
        } else {
            Err(OrchestrationError::not_found(format!("Session {}", session_id)))
        }
    }
    
    async fn join_session(&self, session_id: String, agent_id: AgentId) -> Result<()> {
        let mut state = self.state.write();
        
        if let Some(session) = state.active_sessions.get_mut(&session_id) {
            if !session.participants.contains(&agent_id) {
                session.participants.push(agent_id);
                state.last_update = Timestamp::now();
                
                // Broadcast agent joined event
                let _ = self.event_broadcaster.send(CoordinationEvent::AgentJoined {
                    agent_id,
                    session_id: session_id.clone(),
                    timestamp: Timestamp::now(),
                });
                
                debug!("Agent {} joined coordination session {}", agent_id, session_id);
            }
            Ok(())
        } else {
            Err(OrchestrationError::not_found(format!("Session {}", session_id)))
        }
    }
    
    async fn leave_session(&self, session_id: String, agent_id: AgentId) -> Result<()> {
        let mut state = self.state.write();
        
        if let Some(session) = state.active_sessions.get_mut(&session_id) {
            session.participants.retain(|&id| id != agent_id);
            state.last_update = Timestamp::now();
            
            // Broadcast agent left event
            let _ = self.event_broadcaster.send(CoordinationEvent::AgentLeft {
                agent_id,
                session_id: session_id.clone(),
                timestamp: Timestamp::now(),
            });
            
            debug!("Agent {} left coordination session {}", agent_id, session_id);
            Ok(())
        } else {
            Err(OrchestrationError::not_found(format!("Session {}", session_id)))
        }
    }
    
    async fn coordinate_task_assignment(&self, task: Task) -> Result<Option<AgentId>> {
        // Start coordination session for task assignment
        let participants = if let Some(agent_type) = task.agent_type {
            // Get agents of the required type
            self.agent_registry.get_agents_by_type(agent_type).await?
                .into_iter()
                .filter(|agent| agent.is_available())
                .map(|agent| agent.id)
                .collect()
        } else {
            // Get all available agents
            self.agent_registry.get_available_agents().await?
                .into_iter()
                .map(|agent| agent.id)
                .collect()
        };
        
        if participants.is_empty() {
            return Ok(None);
        }
        
        // Use load balancer to select best agent
        let selected_agent = self.load_balancer.select_agent(task.agent_type).await?;
        
        Ok(selected_agent)
    }
    
    async fn coordinate_resource_allocation(&self, resource_request: ResourceRequest) -> Result<ResourceAllocation> {
        // Simple resource allocation (in practice, this would be more complex)
        let allocation = ResourceAllocation {
            request_id: uuid::Uuid::new_v4().to_string(),
            allocated_amount: resource_request.amount,
            provider: None, // Would select appropriate provider
            success: true,
            message: Some("Resource allocated successfully".to_string()),
        };
        
        Ok(allocation)
    }
    
    async fn coordinate_state_sync(&self) -> Result<()> {
        // Synchronize state across all agents
        let agents = self.agent_registry.get_all_agents().await?;
        
        for agent in agents {
            if agent.state == AgentState::Running {
                // Send state sync message to agent
                let sync_message = Message::control(
                    AgentId::new(), // System agent
                    agent.id,
                    b"state_sync".to_vec(),
                );
                
                if let Err(e) = self.communication.send_message(sync_message).await {
                    warn!("Failed to send state sync to agent {}: {}", agent.id, e);
                }
            }
        }
        
        Ok(())
    }
    
    async fn subscribe_events(&self) -> Result<broadcast::Receiver<CoordinationEvent>> {
        Ok(self.event_broadcaster.subscribe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication::MessageRouter;
    use crate::health::BasicHealthChecker;
    use crate::load_balancer::AdaptiveLoadBalancer;
    use crate::memory::MemoryCoordinator;
    use crate::recovery::AdaptiveRecoveryManager;
    use crate::task_queue::PriorityTaskQueue;
    use tokio::time::{sleep, Duration};
    
    async fn create_test_coordination_engine() -> AdaptiveCoordinationEngine {
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication.clone()));
        let task_queue = Arc::new(PriorityTaskQueue::new());
        let task_distributor = Arc::new(TaskDistributor::new(task_queue.clone(), agent_registry.clone()));
        let load_balancer = Arc::new(AdaptiveLoadBalancer::new(agent_registry.clone()));
        let shared_memory = Arc::new(MemoryCoordinator::new(1000));
        let health_monitor = Arc::new(HealthMonitor::new(agent_registry.clone()));
        let recovery_manager = Arc::new(AdaptiveRecoveryManager::new(3, 5000, 100));
        
        AdaptiveCoordinationEngine::new(
            agent_registry,
            communication,
            task_queue,
            task_distributor,
            load_balancer,
            shared_memory,
            health_monitor,
            recovery_manager,
        )
    }
    
    #[tokio::test]
    async fn test_coordination_engine_lifecycle() {
        let engine = create_test_coordination_engine().await;
        
        // Test start
        engine.start().await.unwrap();
        
        let state = engine.get_state().await.unwrap();
        assert_eq!(state.mode, CoordinationMode::Hybrid);
        assert_eq!(state.active_sessions.len(), 0);
        
        // Test stop
        engine.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_coordination_mode_change() {
        let engine = create_test_coordination_engine().await;
        engine.start().await.unwrap();
        
        // Change mode
        engine.set_mode(CoordinationMode::Centralized).await.unwrap();
        
        let state = engine.get_state().await.unwrap();
        assert_eq!(state.mode, CoordinationMode::Centralized);
        
        engine.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_coordination_sessions() {
        let engine = create_test_coordination_engine().await;
        engine.start().await.unwrap();
        
        let participants = vec![AgentId::new(), AgentId::new()];
        
        // Start session
        let session_id = engine.start_session(
            CoordinationSessionType::TaskAssignment,
            participants.clone(),
        ).await.unwrap();
        
        let state = engine.get_state().await.unwrap();
        assert_eq!(state.active_sessions.len(), 1);
        assert!(state.active_sessions.contains_key(&session_id));
        
        // Join session
        let new_agent = AgentId::new();
        engine.join_session(session_id.clone(), new_agent).await.unwrap();
        
        let state = engine.get_state().await.unwrap();
        let session = &state.active_sessions[&session_id];
        assert!(session.participants.contains(&new_agent));
        
        // Leave session
        engine.leave_session(session_id.clone(), new_agent).await.unwrap();
        
        let state = engine.get_state().await.unwrap();
        let session = &state.active_sessions[&session_id];
        assert!(!session.participants.contains(&new_agent));
        
        // End session
        engine.end_session(session_id.clone()).await.unwrap();
        
        let state = engine.get_state().await.unwrap();
        let session = &state.active_sessions[&session_id];
        assert_eq!(session.status, CoordinationSessionStatus::Completed);
        
        engine.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_coordination_events() {
        let engine = create_test_coordination_engine().await;
        let mut event_receiver = engine.subscribe_events().await.unwrap();
        
        engine.start().await.unwrap();
        
        // Check for system started event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        match event.unwrap().unwrap() {
            CoordinationEvent::SystemStarted { .. } => {},
            _ => panic!("Expected SystemStarted event"),
        }
        
        // Start a coordination session
        let participants = vec![AgentId::new()];
        let _session_id = engine.start_session(
            CoordinationSessionType::TaskAssignment,
            participants.clone(),
        ).await.unwrap();
        
        // Check for session started event
        let event = tokio::time::timeout(Duration::from_secs(1), event_receiver.recv()).await;
        assert!(event.is_ok());
        
        match event.unwrap().unwrap() {
            CoordinationEvent::SessionStarted { session_type, participants: event_participants, .. } => {
                assert!(matches!(session_type, CoordinationSessionType::TaskAssignment));
                assert_eq!(event_participants, participants);
            },
            _ => panic!("Expected SessionStarted event"),
        }
        
        engine.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_task_assignment_coordination() {
        let engine = create_test_coordination_engine().await;
        engine.start().await.unwrap();
        
        // Register an agent
        let agent_info = crate::agent::AgentInfo::new(
            AgentId::new(),
            AgentType::Risk,
            "Test Agent".to_string(),
            "1.0.0".to_string(),
        );
        engine.agent_registry.register_agent(agent_info).await.unwrap();
        
        // Create a task
        let task = Task::new("test_task", crate::types::TaskPriority::High, b"payload".to_vec())
            .with_agent_type(AgentType::Risk);
        
        // Coordinate task assignment
        let assigned_agent = engine.coordinate_task_assignment(task).await.unwrap();
        assert!(assigned_agent.is_some());
        
        engine.stop().await.unwrap();
    }
    
    #[tokio::test]
    async fn test_resource_allocation_coordination() {
        let engine = create_test_coordination_engine().await;
        engine.start().await.unwrap();
        
        let resource_request = ResourceRequest {
            requester: AgentId::new(),
            resource_type: "memory".to_string(),
            amount: 1024,
            priority: 1,
            deadline: None,
        };
        
        let allocation = engine.coordinate_resource_allocation(resource_request).await.unwrap();
        assert!(allocation.success);
        assert_eq!(allocation.allocated_amount, 1024);
        
        engine.stop().await.unwrap();
    }
}