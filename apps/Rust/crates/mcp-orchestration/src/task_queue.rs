//! Task queue management and distribution system for MCP orchestration.

use crate::agent::{AgentInfo, AgentRegistry};
use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, AgentType, TaskId, TaskPriority, TaskStatus, Timestamp};
use async_trait::async_trait;
use dashmap::DashMap;
use parking_lot::RwLock;
use priority_queue::PriorityQueue;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::{mpsc, oneshot, RwLock as TokioRwLock};
use tokio::time::{interval, sleep, timeout};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

/// Task definition for the orchestration system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    /// Unique task identifier
    pub id: TaskId,
    /// Task name/description
    pub name: String,
    /// Task priority
    pub priority: TaskPriority,
    /// Required agent type
    pub agent_type: Option<AgentType>,
    /// Preferred agent ID
    pub preferred_agent: Option<AgentId>,
    /// Task payload
    pub payload: Vec<u8>,
    /// Task parameters
    pub parameters: HashMap<String, String>,
    /// Task dependencies
    pub dependencies: Vec<TaskId>,
    /// Task timeout in milliseconds
    pub timeout: u64,
    /// Maximum retry attempts
    pub max_retries: u32,
    /// Current retry count
    pub retry_count: u32,
    /// Task creation timestamp
    pub created_at: Timestamp,
    /// Task deadline
    pub deadline: Option<Timestamp>,
    /// Task status
    pub status: TaskStatus,
    /// Assigned agent ID
    pub assigned_agent: Option<AgentId>,
    /// Task start time
    pub started_at: Option<Timestamp>,
    /// Task completion time
    pub completed_at: Option<Timestamp>,
    /// Task result
    pub result: Option<Vec<u8>>,
    /// Task error message
    pub error: Option<String>,
    /// Task metadata
    pub metadata: HashMap<String, String>,
}

impl Task {
    /// Create a new task
    pub fn new<S: Into<String>>(name: S, priority: TaskPriority, payload: Vec<u8>) -> Self {
        Self {
            id: TaskId::new(),
            name: name.into(),
            priority,
            agent_type: None,
            preferred_agent: None,
            payload,
            parameters: HashMap::new(),
            dependencies: Vec::new(),
            timeout: 30000, // 30 seconds default
            max_retries: 3,
            retry_count: 0,
            created_at: Timestamp::now(),
            deadline: None,
            status: TaskStatus::Queued,
            assigned_agent: None,
            started_at: None,
            completed_at: None,
            result: None,
            error: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Set the required agent type
    pub fn with_agent_type(mut self, agent_type: AgentType) -> Self {
        self.agent_type = Some(agent_type);
        self
    }
    
    /// Set the preferred agent
    pub fn with_preferred_agent(mut self, agent_id: AgentId) -> Self {
        self.preferred_agent = Some(agent_id);
        self
    }
    
    /// Set task timeout
    pub fn with_timeout(mut self, timeout: u64) -> Self {
        self.timeout = timeout;
        self
    }
    
    /// Set task deadline
    pub fn with_deadline(mut self, deadline: Timestamp) -> Self {
        self.deadline = Some(deadline);
        self
    }
    
    /// Add task dependency
    pub fn with_dependency(mut self, task_id: TaskId) -> Self {
        self.dependencies.push(task_id);
        self
    }
    
    /// Add task parameter
    pub fn with_parameter<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.parameters.insert(key.into(), value.into());
        self
    }
    
    /// Add task metadata
    pub fn with_metadata<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.metadata.insert(key.into(), value.into());
        self
    }
    
    /// Check if task has expired
    pub fn is_expired(&self) -> bool {
        if let Some(deadline) = self.deadline {
            Timestamp::now() >= deadline
        } else {
            false
        }
    }
    
    /// Check if task can be retried
    pub fn can_retry(&self) -> bool {
        self.retry_count < self.max_retries
    }
    
    /// Get task age
    pub fn age(&self) -> Duration {
        Duration::from_millis(self.created_at.elapsed().as_millis() as u64)
    }
    
    /// Get task processing time
    pub fn processing_time(&self) -> Option<Duration> {
        if let (Some(started), Some(completed)) = (self.started_at, self.completed_at) {
            Some(Duration::from_millis(completed.as_millis() - started.as_millis()))
        } else {
            None
        }
    }
    
    /// Mark task as started
    pub fn start(&mut self, agent_id: AgentId) {
        self.status = TaskStatus::Processing;
        self.assigned_agent = Some(agent_id);
        self.started_at = Some(Timestamp::now());
    }
    
    /// Mark task as completed
    pub fn complete(&mut self, result: Vec<u8>) {
        self.status = TaskStatus::Completed;
        self.completed_at = Some(Timestamp::now());
        self.result = Some(result);
    }
    
    /// Mark task as failed
    pub fn fail(&mut self, error: String) {
        self.status = TaskStatus::Failed;
        self.completed_at = Some(Timestamp::now());
        self.error = Some(error);
    }
    
    /// Mark task as cancelled
    pub fn cancel(&mut self) {
        self.status = TaskStatus::Cancelled;
        self.completed_at = Some(Timestamp::now());
    }
    
    /// Mark task as timed out
    pub fn timeout(&mut self) {
        self.status = TaskStatus::TimedOut;
        self.completed_at = Some(Timestamp::now());
        self.error = Some("Task timed out".to_string());
    }
    
    /// Increment retry count
    pub fn increment_retry(&mut self) {
        self.retry_count += 1;
        self.status = TaskStatus::Queued;
        self.assigned_agent = None;
        self.started_at = None;
    }
}

/// Task queue trait for managing tasks
#[async_trait]
pub trait TaskQueue: Send + Sync {
    /// Submit a task to the queue
    async fn submit_task(&self, task: Task) -> Result<TaskId>;
    
    /// Get the next task for processing
    async fn get_next_task(&self, agent_type: Option<AgentType>) -> Result<Option<Task>>;
    
    /// Complete a task
    async fn complete_task(&self, task_id: TaskId, result: Vec<u8>) -> Result<()>;
    
    /// Fail a task
    async fn fail_task(&self, task_id: TaskId, error: String) -> Result<()>;
    
    /// Cancel a task
    async fn cancel_task(&self, task_id: TaskId) -> Result<()>;
    
    /// Get task status
    async fn get_task_status(&self, task_id: TaskId) -> Result<TaskStatus>;
    
    /// Get task details
    async fn get_task(&self, task_id: TaskId) -> Result<Task>;
    
    /// Get queue statistics
    async fn get_queue_stats(&self) -> Result<QueueStatistics>;
    
    /// Get pending tasks
    async fn get_pending_tasks(&self) -> Result<Vec<Task>>;
    
    /// Get completed tasks
    async fn get_completed_tasks(&self, limit: Option<usize>) -> Result<Vec<Task>>;
}

/// Queue statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueueStatistics {
    /// Total tasks submitted
    pub total_tasks: u64,
    /// Tasks currently queued
    pub queued_tasks: u64,
    /// Tasks currently processing
    pub processing_tasks: u64,
    /// Tasks completed
    pub completed_tasks: u64,
    /// Tasks failed
    pub failed_tasks: u64,
    /// Tasks cancelled
    pub cancelled_tasks: u64,
    /// Tasks timed out
    pub timed_out_tasks: u64,
    /// Average task processing time
    pub avg_processing_time: f64,
    /// Queue depth by priority
    pub queue_depth_by_priority: HashMap<TaskPriority, u64>,
    /// Queue depth by agent type
    pub queue_depth_by_agent_type: HashMap<AgentType, u64>,
}

/// Priority-based task queue implementation
#[derive(Debug)]
pub struct PriorityTaskQueue {
    /// Priority queue for tasks
    queue: Arc<RwLock<PriorityQueue<TaskId, TaskPriority>>>,
    /// Task storage
    tasks: Arc<DashMap<TaskId, Task>>,
    /// Task completion channels
    completion_channels: Arc<DashMap<TaskId, oneshot::Sender<Result<Vec<u8>>>>>,
    /// Queue statistics
    stats: Arc<RwLock<QueueStatistics>>,
    /// Task counter
    task_counter: Arc<AtomicU64>,
}

impl PriorityTaskQueue {
    /// Create a new priority task queue
    pub fn new() -> Self {
        Self {
            queue: Arc::new(RwLock::new(PriorityQueue::new())),
            tasks: Arc::new(DashMap::new()),
            completion_channels: Arc::new(DashMap::new()),
            stats: Arc::new(RwLock::new(QueueStatistics {
                total_tasks: 0,
                queued_tasks: 0,
                processing_tasks: 0,
                completed_tasks: 0,
                failed_tasks: 0,
                cancelled_tasks: 0,
                timed_out_tasks: 0,
                avg_processing_time: 0.0,
                queue_depth_by_priority: HashMap::new(),
                queue_depth_by_agent_type: HashMap::new(),
            })),
            task_counter: Arc::new(AtomicU64::new(0)),
        }
    }
    
    /// Start background task processing
    pub async fn start(&self) -> Result<()> {
        // Start task timeout checker
        let tasks = Arc::clone(&self.tasks);
        let queue = Arc::clone(&self.queue);
        let stats = Arc::clone(&self.stats);
        
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                let mut expired_tasks = Vec::new();
                
                // Check for expired tasks
                for entry in tasks.iter() {
                    let task = entry.value();
                    if task.status == TaskStatus::Processing && task.is_expired() {
                        expired_tasks.push(task.id);
                    }
                }
                
                // Handle expired tasks
                for task_id in expired_tasks {
                    if let Some(mut task) = tasks.get_mut(&task_id) {
                        task.timeout();
                        
                        let mut stats = stats.write();
                        stats.processing_tasks -= 1;
                        stats.timed_out_tasks += 1;
                    }
                }
            }
        });
        
        info!("Priority task queue started successfully");
        Ok(())
    }
    
    /// Update queue statistics
    fn update_stats(&self, task: &Task, operation: &str) {
        let mut stats = self.stats.write();
        
        match operation {
            "submit" => {
                stats.total_tasks += 1;
                stats.queued_tasks += 1;
                *stats.queue_depth_by_priority.entry(task.priority).or_insert(0) += 1;
                if let Some(agent_type) = task.agent_type {
                    *stats.queue_depth_by_agent_type.entry(agent_type).or_insert(0) += 1;
                }
            }
            "start" => {
                stats.queued_tasks -= 1;
                stats.processing_tasks += 1;
                *stats.queue_depth_by_priority.entry(task.priority).or_insert(0) -= 1;
                if let Some(agent_type) = task.agent_type {
                    *stats.queue_depth_by_agent_type.entry(agent_type).or_insert(0) -= 1;
                }
            }
            "complete" => {
                stats.processing_tasks -= 1;
                stats.completed_tasks += 1;
                if let Some(processing_time) = task.processing_time() {
                    let current_avg = stats.avg_processing_time;
                    let completed_count = stats.completed_tasks;
                    stats.avg_processing_time = 
                        (current_avg * (completed_count - 1) as f64 + processing_time.as_millis() as f64) 
                        / completed_count as f64;
                }
            }
            "fail" => {
                stats.processing_tasks -= 1;
                stats.failed_tasks += 1;
            }
            "cancel" => {
                if task.status == TaskStatus::Processing {
                    stats.processing_tasks -= 1;
                } else {
                    stats.queued_tasks -= 1;
                }
                stats.cancelled_tasks += 1;
            }
            _ => {}
        }
    }
}

#[async_trait]
impl TaskQueue for PriorityTaskQueue {
    async fn submit_task(&self, task: Task) -> Result<TaskId> {
        let task_id = task.id;
        
        // Check task dependencies
        for dep_id in &task.dependencies {
            if let Some(dep_task) = self.tasks.get(dep_id) {
                if dep_task.status != TaskStatus::Completed {
                    return Err(OrchestrationError::task(
                        format!("Task {} depends on incomplete task {}", task_id, dep_id)
                    ));
                }
            } else {
                return Err(OrchestrationError::task(
                    format!("Task {} depends on non-existent task {}", task_id, dep_id)
                ));
            }
        }
        
        // Add task to queue
        self.queue.write().push(task_id, task.priority);
        
        // Store task
        self.tasks.insert(task_id, task.clone());
        
        // Update statistics
        self.update_stats(&task, "submit");
        
        self.task_counter.fetch_add(1, Ordering::Relaxed);
        
        debug!("Task {} submitted to queue", task_id);
        Ok(task_id)
    }
    
    async fn get_next_task(&self, agent_type: Option<AgentType>) -> Result<Option<Task>> {
        let mut queue = self.queue.write();
        
        // Find the highest priority task that matches the agent type
        let mut task_id = None;
        let mut tasks_to_requeue = Vec::new();
        
        while let Some((id, priority)) = queue.pop() {
            if let Some(task) = self.tasks.get(&id) {
                if let Some(required_type) = agent_type {
                    if let Some(task_agent_type) = task.agent_type {
                        if task_agent_type == required_type {
                            task_id = Some(id);
                            break;
                        } else {
                            tasks_to_requeue.push((id, priority));
                        }
                    } else {
                        // Task doesn't require specific agent type
                        task_id = Some(id);
                        break;
                    }
                } else {
                    // Agent can handle any task
                    task_id = Some(id);
                    break;
                }
            } else {
                // Task no longer exists, skip
                continue;
            }
        }
        
        // Requeue tasks that didn't match
        for (id, priority) in tasks_to_requeue {
            queue.push(id, priority);
        }
        
        drop(queue);
        
        if let Some(id) = task_id {
            if let Some(mut task) = self.tasks.get_mut(&id) {
                // Don't actually start the task here, just return it
                // The task will be marked as started when assigned to an agent
                return Ok(Some(task.clone()));
            }
        }
        
        Ok(None)
    }
    
    async fn complete_task(&self, task_id: TaskId, result: Vec<u8>) -> Result<()> {
        let mut task = self.tasks.get_mut(&task_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Task {}", task_id)))?;
        
        task.complete(result.clone());
        self.update_stats(&task, "complete");
        
        // Notify completion channel if exists
        if let Some((_, sender)) = self.completion_channels.remove(&task_id) {
            let _ = sender.send(Ok(result));
        }
        
        debug!("Task {} completed successfully", task_id);
        Ok(())
    }
    
    async fn fail_task(&self, task_id: TaskId, error: String) -> Result<()> {
        let mut task = self.tasks.get_mut(&task_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Task {}", task_id)))?;
        
        // Check if task can be retried
        if task.can_retry() {
            task.increment_retry();
            self.queue.write().push(task_id, task.priority);
            debug!("Task {} failed, retrying (attempt {})", task_id, task.retry_count);
        } else {
            task.fail(error.clone());
            self.update_stats(&task, "fail");
            
            // Notify completion channel if exists
            if let Some((_, sender)) = self.completion_channels.remove(&task_id) {
                let _ = sender.send(Err(OrchestrationError::task(error)));
            }
            
            debug!("Task {} failed permanently", task_id);
        }
        
        Ok(())
    }
    
    async fn cancel_task(&self, task_id: TaskId) -> Result<()> {
        let mut task = self.tasks.get_mut(&task_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Task {}", task_id)))?;
        
        task.cancel();
        self.update_stats(&task, "cancel");
        
        // Remove from queue if still queued
        let mut queue = self.queue.write();
        // Note: PriorityQueue doesn't have efficient removal by key
        // In practice, we'd need a more sophisticated data structure
        
        // Notify completion channel if exists
        if let Some((_, sender)) = self.completion_channels.remove(&task_id) {
            let _ = sender.send(Err(OrchestrationError::task("Task cancelled".to_string())));
        }
        
        debug!("Task {} cancelled", task_id);
        Ok(())
    }
    
    async fn get_task_status(&self, task_id: TaskId) -> Result<TaskStatus> {
        let task = self.tasks.get(&task_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Task {}", task_id)))?;
        
        Ok(task.status.clone())
    }
    
    async fn get_task(&self, task_id: TaskId) -> Result<Task> {
        let task = self.tasks.get(&task_id)
            .ok_or_else(|| OrchestrationError::not_found(format!("Task {}", task_id)))?;
        
        Ok(task.clone())
    }
    
    async fn get_queue_stats(&self) -> Result<QueueStatistics> {
        Ok(self.stats.read().clone())
    }
    
    async fn get_pending_tasks(&self) -> Result<Vec<Task>> {
        let mut pending_tasks = Vec::new();
        
        for entry in self.tasks.iter() {
            let task = entry.value();
            if matches!(task.status, TaskStatus::Queued | TaskStatus::Processing) {
                pending_tasks.push(task.clone());
            }
        }
        
        // Sort by priority
        pending_tasks.sort_by(|a, b| a.priority.cmp(&b.priority));
        
        Ok(pending_tasks)
    }
    
    async fn get_completed_tasks(&self, limit: Option<usize>) -> Result<Vec<Task>> {
        let mut completed_tasks = Vec::new();
        
        for entry in self.tasks.iter() {
            let task = entry.value();
            if matches!(task.status, TaskStatus::Completed | TaskStatus::Failed | TaskStatus::Cancelled | TaskStatus::TimedOut) {
                completed_tasks.push(task.clone());
            }
        }
        
        // Sort by completion time (most recent first)
        completed_tasks.sort_by(|a, b| {
            match (a.completed_at, b.completed_at) {
                (Some(a_time), Some(b_time)) => b_time.cmp(&a_time),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => std::cmp::Ordering::Equal,
            }
        });
        
        if let Some(limit) = limit {
            completed_tasks.truncate(limit);
        }
        
        Ok(completed_tasks)
    }
}

/// Task distributor for load balancing across agents
#[derive(Debug)]
pub struct TaskDistributor {
    /// Task queue
    task_queue: Arc<dyn TaskQueue>,
    /// Agent registry
    agent_registry: Arc<AgentRegistry>,
    /// Active task assignments
    active_assignments: Arc<DashMap<TaskId, AgentId>>,
    /// Agent workload tracking
    agent_workloads: Arc<DashMap<AgentId, u64>>,
}

impl TaskDistributor {
    /// Create a new task distributor
    pub fn new(
        task_queue: Arc<dyn TaskQueue>,
        agent_registry: Arc<AgentRegistry>,
    ) -> Self {
        Self {
            task_queue,
            agent_registry,
            active_assignments: Arc::new(DashMap::new()),
            agent_workloads: Arc::new(DashMap::new()),
        }
    }
    
    /// Start the task distributor
    pub async fn start(&self) -> Result<()> {
        // Start task distribution loop
        let distributor = self.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = distributor.distribute_tasks().await {
                    warn!("Task distribution error: {}", e);
                }
            }
        });
        
        info!("Task distributor started successfully");
        Ok(())
    }
    
    /// Distribute tasks to available agents
    async fn distribute_tasks(&self) -> Result<()> {
        // Get available agents
        let available_agents = self.agent_registry.get_available_agents().await?;
        
        if available_agents.is_empty() {
            return Ok(());
        }
        
        // Try to assign tasks to available agents
        for agent_info in available_agents {
            if let Some(task) = self.task_queue.get_next_task(Some(agent_info.agent_type)).await? {
                // Assign task to agent
                self.assign_task_to_agent(task, agent_info).await?;
            }
        }
        
        Ok(())
    }
    
    /// Assign a task to a specific agent
    async fn assign_task_to_agent(&self, mut task: Task, agent_info: AgentInfo) -> Result<()> {
        let task_id = task.id;
        let agent_id = agent_info.id;
        
        // Mark task as started
        task.start(agent_id);
        
        // Update active assignments
        self.active_assignments.insert(task_id, agent_id);
        
        // Update agent workload
        *self.agent_workloads.entry(agent_id).or_insert(0) += 1;
        
        // Update agent state to busy
        self.agent_registry.update_agent_state(agent_id, crate::types::AgentState::Busy).await?;
        
        debug!("Task {} assigned to agent {} ({})", task_id, agent_id, agent_info.agent_type);
        
        // In a real implementation, we would send the task to the agent
        // For now, we'll simulate task completion after a delay
        let task_queue = Arc::clone(&self.task_queue);
        let agent_registry = Arc::clone(&self.agent_registry);
        let active_assignments = Arc::clone(&self.active_assignments);
        let agent_workloads = Arc::clone(&self.agent_workloads);
        
        tokio::spawn(async move {
            // Simulate task processing
            sleep(Duration::from_millis(1000)).await;
            
            // Complete the task
            let result = format!("Task {} completed by agent {}", task_id, agent_id);
            if let Err(e) = task_queue.complete_task(task_id, result.into_bytes()).await {
                error!("Failed to complete task {}: {}", task_id, e);
            }
            
            // Clean up assignments
            active_assignments.remove(&task_id);
            if let Some(mut workload) = agent_workloads.get_mut(&agent_id) {
                *workload -= 1;
                if *workload == 0 {
                    // Agent is no longer busy
                    if let Err(e) = agent_registry.update_agent_state(agent_id, crate::types::AgentState::Running).await {
                        error!("Failed to update agent state: {}", e);
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Get distribution statistics
    pub async fn get_distribution_stats(&self) -> Result<DistributionStatistics> {
        let mut stats = DistributionStatistics::default();
        
        for entry in self.active_assignments.iter() {
            stats.active_assignments += 1;
        }
        
        for entry in self.agent_workloads.iter() {
            let workload = *entry.value();
            stats.total_workload += workload;
            if workload > 0 {
                stats.busy_agents += 1;
            }
        }
        
        Ok(stats)
    }
}

impl Clone for TaskDistributor {
    fn clone(&self) -> Self {
        Self {
            task_queue: Arc::clone(&self.task_queue),
            agent_registry: Arc::clone(&self.agent_registry),
            active_assignments: Arc::clone(&self.active_assignments),
            agent_workloads: Arc::clone(&self.agent_workloads),
        }
    }
}

/// Distribution statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DistributionStatistics {
    /// Number of active task assignments
    pub active_assignments: u64,
    /// Total workload across all agents
    pub total_workload: u64,
    /// Number of busy agents
    pub busy_agents: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication::MessageRouter;
    use std::sync::Arc;
    
    #[tokio::test]
    async fn test_task_creation() {
        let task = Task::new("test_task", TaskPriority::High, b"test_payload".to_vec())
            .with_agent_type(AgentType::Risk)
            .with_timeout(5000)
            .with_parameter("key", "value");
        
        assert_eq!(task.name, "test_task");
        assert_eq!(task.priority, TaskPriority::High);
        assert_eq!(task.payload, b"test_payload");
        assert_eq!(task.agent_type, Some(AgentType::Risk));
        assert_eq!(task.timeout, 5000);
        assert_eq!(task.parameters.get("key").unwrap(), "value");
    }
    
    #[tokio::test]
    async fn test_priority_task_queue() {
        let queue = PriorityTaskQueue::new();
        queue.start().await.unwrap();
        
        // Submit tasks with different priorities
        let task1 = Task::new("low_priority", TaskPriority::Low, b"payload1".to_vec());
        let task2 = Task::new("high_priority", TaskPriority::High, b"payload2".to_vec());
        let task3 = Task::new("critical", TaskPriority::Critical, b"payload3".to_vec());
        
        let id1 = queue.submit_task(task1).await.unwrap();
        let id2 = queue.submit_task(task2).await.unwrap();
        let id3 = queue.submit_task(task3).await.unwrap();
        
        // Get tasks in priority order
        let next_task = queue.get_next_task(None).await.unwrap().unwrap();
        assert_eq!(next_task.priority, TaskPriority::Critical);
        
        let next_task = queue.get_next_task(None).await.unwrap().unwrap();
        assert_eq!(next_task.priority, TaskPriority::High);
        
        let next_task = queue.get_next_task(None).await.unwrap().unwrap();
        assert_eq!(next_task.priority, TaskPriority::Low);
        
        // Complete a task
        queue.complete_task(id3, b"result".to_vec()).await.unwrap();
        let status = queue.get_task_status(id3).await.unwrap();
        assert_eq!(status, TaskStatus::Completed);
        
        // Fail a task
        queue.fail_task(id2, "test error".to_string()).await.unwrap();
        let status = queue.get_task_status(id2).await.unwrap();
        assert_eq!(status, TaskStatus::Failed);
    }
    
    #[tokio::test]
    async fn test_task_distributor() {
        let task_queue = Arc::new(PriorityTaskQueue::new());
        let communication = Arc::new(MessageRouter::new());
        let agent_registry = Arc::new(AgentRegistry::new(communication));
        
        let distributor = TaskDistributor::new(task_queue.clone(), agent_registry.clone());
        
        // Register an agent
        let agent_info = crate::agent::AgentInfo::new(
            AgentId::new(),
            AgentType::Risk,
            "Risk Agent".to_string(),
            "1.0.0".to_string(),
        );
        agent_registry.register_agent(agent_info).await.unwrap();
        
        // Submit a task
        let task = Task::new("test_task", TaskPriority::High, b"payload".to_vec())
            .with_agent_type(AgentType::Risk);
        let task_id = task_queue.submit_task(task).await.unwrap();
        
        // Start distributor
        distributor.start().await.unwrap();
        
        // Wait for distribution
        sleep(Duration::from_millis(200)).await;
        
        let stats = distributor.get_distribution_stats().await.unwrap();
        assert_eq!(stats.active_assignments, 1);
    }
    
    #[tokio::test]
    async fn test_task_dependencies() {
        let queue = PriorityTaskQueue::new();
        
        // Create and submit first task
        let task1 = Task::new("task1", TaskPriority::High, b"payload1".to_vec());
        let task1_id = queue.submit_task(task1).await.unwrap();
        
        // Complete first task
        queue.complete_task(task1_id, b"result1".to_vec()).await.unwrap();
        
        // Create second task that depends on first
        let task2 = Task::new("task2", TaskPriority::High, b"payload2".to_vec())
            .with_dependency(task1_id);
        
        // Should succeed because task1 is completed
        let _task2_id = queue.submit_task(task2).await.unwrap();
        
        // Create third task that depends on non-existent task
        let non_existent_id = TaskId::new();
        let task3 = Task::new("task3", TaskPriority::High, b"payload3".to_vec())
            .with_dependency(non_existent_id);
        
        // Should fail because dependency doesn't exist
        let result = queue.submit_task(task3).await;
        assert!(result.is_err());
    }
}