// Agent coordination logic

use super::{AgentId, AgentMetadata, AgentStatus, Task, TaskResult, FederationTopology};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Coordination strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CoordinationStrategy {
    /// Round-robin task assignment
    RoundRobin,

    /// Least loaded agent gets task
    LeastLoaded,

    /// Agent with best capability match
    CapabilityMatch,

    /// Random assignment
    Random,

    /// Priority-based assignment
    Priority,
}

/// Agent coordinator
pub struct AgentCoordinator {
    /// Federation topology
    topology: Arc<RwLock<FederationTopology>>,

    /// Task queue
    task_queue: Arc<Mutex<VecDeque<Task>>>,

    /// Pending tasks (assigned but not completed)
    pending_tasks: Arc<RwLock<HashMap<Uuid, Task>>>,

    /// Completed tasks
    completed_tasks: Arc<RwLock<HashMap<Uuid, TaskResult>>>,

    /// Coordination strategy
    strategy: CoordinationStrategy,

    /// Agent workloads (agent_id -> current task count)
    workloads: Arc<RwLock<HashMap<AgentId, usize>>>,
}

impl AgentCoordinator {
    /// Create new coordinator
    pub fn new(topology: Arc<RwLock<FederationTopology>>, strategy: CoordinationStrategy) -> Self {
        Self {
            topology,
            task_queue: Arc::new(Mutex::new(VecDeque::new())),
            pending_tasks: Arc::new(RwLock::new(HashMap::new())),
            completed_tasks: Arc::new(RwLock::new(HashMap::new())),
            strategy,
            workloads: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Submit a task
    pub async fn submit_task(&self, task: Task) -> Result<Uuid> {
        let task_id = task.id;

        // Add to queue
        self.task_queue.lock().await.push_back(task);

        // Try to assign immediately
        self.schedule_tasks().await?;

        Ok(task_id)
    }

    /// Submit multiple tasks
    pub async fn submit_tasks(&self, tasks: Vec<Task>) -> Result<Vec<Uuid>> {
        let mut task_ids = Vec::with_capacity(tasks.len());

        {
            let mut queue = self.task_queue.lock().await;
            for task in tasks {
                task_ids.push(task.id);
                queue.push_back(task);
            }
        }

        // Schedule all tasks
        self.schedule_tasks().await?;

        Ok(task_ids)
    }

    /// Schedule tasks to agents
    async fn schedule_tasks(&self) -> Result<()> {
        let mut queue = self.task_queue.lock().await;

        while let Some(task) = queue.pop_front() {
            // Find suitable agent
            let agent_id = self.select_agent(&task).await?;

            // Assign task
            {
                let mut pending = self.pending_tasks.write().await;
                let mut task = task;
                task.assigned_to = Some(agent_id.clone());
                pending.insert(task.id, task);
            }

            // Update workload
            {
                let mut workloads = self.workloads.write().await;
                *workloads.entry(agent_id).or_insert(0) += 1;
            }
        }

        Ok(())
    }

    /// Select agent based on coordination strategy
    async fn select_agent(&self, task: &Task) -> Result<AgentId> {
        let topology = self.topology.read().await;
        let agents = topology.get_agents();

        if agents.is_empty() {
            return Err(DistributedError::FederationError(
                "No agents available".to_string(),
            ));
        }

        match self.strategy {
            CoordinationStrategy::RoundRobin => self.select_round_robin(&agents).await,
            CoordinationStrategy::LeastLoaded => self.select_least_loaded(&agents).await,
            CoordinationStrategy::CapabilityMatch => self.select_capability_match(&agents, task).await,
            CoordinationStrategy::Random => self.select_random(&agents),
            CoordinationStrategy::Priority => self.select_priority(&agents, task).await,
        }
    }

    /// Round-robin selection
    async fn select_round_robin(&self, agents: &[&AgentMetadata]) -> Result<AgentId> {
        // Simple: select first idle agent
        for agent in agents {
            if agent.status == AgentStatus::Idle {
                return Ok(agent.id.clone());
            }
        }

        // If none idle, select first agent
        Ok(agents[0].id.clone())
    }

    /// Least loaded selection
    async fn select_least_loaded(&self, agents: &[&AgentMetadata]) -> Result<AgentId> {
        let workloads = self.workloads.read().await;

        let mut best_agent = None;
        let mut min_load = usize::MAX;

        for agent in agents {
            if agent.status == AgentStatus::Unavailable {
                continue;
            }

            let load = workloads.get(&agent.id).copied().unwrap_or(0);
            if load < min_load {
                min_load = load;
                best_agent = Some(agent.id.clone());
            }
        }

        best_agent.ok_or_else(|| {
            DistributedError::FederationError("No available agents".to_string())
        })
    }

    /// Capability-based selection
    async fn select_capability_match(
        &self,
        agents: &[&AgentMetadata],
        task: &Task,
    ) -> Result<AgentId> {
        let mut best_agent = None;
        let mut best_score = 0;

        for agent in agents {
            if agent.status == AgentStatus::Unavailable {
                continue;
            }

            // Count matching capabilities
            let score = task
                .required_capabilities
                .iter()
                .filter(|cap| agent.capabilities.contains(cap))
                .count();

            if score > best_score {
                best_score = score;
                best_agent = Some(agent.id.clone());
            }
        }

        best_agent.ok_or_else(|| {
            DistributedError::FederationError("No matching agents found".to_string())
        })
    }

    /// Random selection
    fn select_random(&self, agents: &[&AgentMetadata]) -> Result<AgentId> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();

        let available: Vec<_> = agents
            .iter()
            .filter(|a| a.status != AgentStatus::Unavailable)
            .collect();

        available
            .choose(&mut rng)
            .map(|a| a.id.clone())
            .ok_or_else(|| DistributedError::FederationError("No agents available".to_string()))
    }

    /// Priority-based selection
    async fn select_priority(&self, agents: &[&AgentMetadata], task: &Task) -> Result<AgentId> {
        // For now, use capability match with priority consideration
        self.select_capability_match(agents, task).await
    }

    /// Complete a task
    pub async fn complete_task(&self, result: TaskResult) -> Result<()> {
        let task_id = result.task_id;

        // Remove from pending
        {
            let mut pending = self.pending_tasks.write().await;
            pending.remove(&task_id);
        }

        // Add to completed
        {
            let mut completed = self.completed_tasks.write().await;
            completed.insert(task_id, result.clone());
        }

        // Update workload
        {
            let mut workloads = self.workloads.write().await;
            if let Some(count) = workloads.get_mut(&result.agent_id) {
                *count = count.saturating_sub(1);
            }
        }

        Ok(())
    }

    /// Get task status
    pub async fn get_task_status(&self, task_id: &Uuid) -> Result<TaskStatus> {
        // Check completed
        {
            let completed = self.completed_tasks.read().await;
            if let Some(result) = completed.get(task_id) {
                return Ok(TaskStatus::Completed(result.clone()));
            }
        }

        // Check pending
        {
            let pending = self.pending_tasks.read().await;
            if let Some(task) = pending.get(task_id) {
                return Ok(TaskStatus::Pending(task.clone()));
            }
        }

        // Check queue
        {
            let queue = self.task_queue.lock().await;
            if let Some(task) = queue.iter().find(|t| &t.id == task_id) {
                return Ok(TaskStatus::Queued(task.clone()));
            }
        }

        Err(DistributedError::FederationError(format!(
            "Task not found: {}",
            task_id
        )))
    }

    /// Get coordination statistics
    pub async fn stats(&self) -> CoordinationStats {
        let queue_size = self.task_queue.lock().await.len();
        let pending_count = self.pending_tasks.read().await.len();
        let completed_count = self.completed_tasks.read().await.len();
        let active_agents = self.workloads.read().await.len();

        CoordinationStats {
            queued_tasks: queue_size,
            pending_tasks: pending_count,
            completed_tasks: completed_count,
            active_agents,
            strategy: self.strategy,
        }
    }
}

/// Task status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Task is queued
    Queued(Task),

    /// Task is assigned and pending
    Pending(Task),

    /// Task is completed
    Completed(TaskResult),
}

/// Coordination statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationStats {
    /// Number of queued tasks
    pub queued_tasks: usize,

    /// Number of pending tasks
    pub pending_tasks: usize,

    /// Number of completed tasks
    pub completed_tasks: usize,

    /// Number of active agents
    pub active_agents: usize,

    /// Coordination strategy
    pub strategy: CoordinationStrategy,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::federation::{TopologyConfig, ResourceLimits};

    #[tokio::test]
    async fn test_task_submission() {
        let topology = Arc::new(RwLock::new(FederationTopology::new(
            TopologyConfig::default(),
        )));

        let coordinator = AgentCoordinator::new(topology, CoordinationStrategy::RoundRobin);

        let task = Task {
            id: Uuid::new_v4(),
            task_type: "test".to_string(),
            payload: serde_json::json!({}),
            priority: 5,
            required_capabilities: vec![],
            deadline: None,
            assigned_to: None,
        };

        // This will fail because no agents, but tests the flow
        let result = coordinator.submit_task(task).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_coordination_stats() {
        let topology = Arc::new(RwLock::new(FederationTopology::new(
            TopologyConfig::default(),
        )));

        let coordinator = AgentCoordinator::new(topology, CoordinationStrategy::LeastLoaded);

        let stats = coordinator.stats().await;
        assert_eq!(stats.queued_tasks, 0);
        assert_eq!(stats.pending_tasks, 0);
        assert_eq!(stats.completed_tasks, 0);
    }
}
