//! Queen coordinator - central orchestration logic

use crate::{error::*, types::*};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the Queen coordinator
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QueenConfig {
    /// Queen's name/identifier
    pub name: String,

    /// Maximum tasks to manage concurrently
    pub max_concurrent_tasks: usize,

    /// Enable intelligent task delegation
    pub intelligent_delegation: bool,
}

impl Default for QueenConfig {
    fn default() -> Self {
        Self {
            name: "Queen-Coordinator".to_string(),
            max_concurrent_tasks: 100,
            intelligent_delegation: true,
        }
    }
}

/// Queen coordinator that manages worker agents
pub struct Queen {
    config: QueenConfig,
    registered_workers: Arc<DashMap<AgentId, AgentType>>,
    active_tasks: Arc<RwLock<Vec<String>>>,
}

impl Queen {
    /// Create a new Queen instance
    pub fn new(config: QueenConfig) -> Result<Self> {
        Ok(Self {
            config,
            registered_workers: Arc::new(DashMap::new()),
            active_tasks: Arc::new(RwLock::new(Vec::new())),
        })
    }

    /// Register a worker with the queen
    pub async fn register_worker(&self, agent_id: AgentId, agent_type: AgentType) -> Result<()> {
        self.registered_workers.insert(agent_id.clone(), agent_type.clone());
        tracing::info!("Queen registered worker: {} ({})", agent_id, agent_type);
        Ok(())
    }

    /// Delegate a task to appropriate workers
    pub async fn delegate_task(
        &self,
        task: &Task,
        _workers: &DashMap<AgentId, crate::worker::Worker>,
    ) -> Result<Vec<(AgentId, Task)>> {
        let mut assignments = Vec::new();

        // Analyze task to determine required agent types
        let required_types = self.analyze_task_requirements(&task.description);

        // Find workers matching required types
        for required_type in required_types {
            if let Some(agent_id) = self.find_worker_by_type(&required_type) {
                // Create a subtask for this worker
                let subtask = Task {
                    id: format!("{}-{}", task.id, agent_id),
                    description: self.create_subtask_description(&task.description, &required_type),
                    priority: task.priority,
                    created_at: task.created_at,
                };

                assignments.push((agent_id, subtask));
            }
        }

        if assignments.is_empty() {
            return Err(HiveMindError::CoordinationError(
                "No suitable workers found for task".to_string(),
            ));
        }

        // Track active task
        let mut active = self.active_tasks.write().await;
        active.push(task.id.clone());

        Ok(assignments)
    }

    /// Analyze task description to determine required agent types
    fn analyze_task_requirements(&self, description: &str) -> Vec<AgentType> {
        let lower = description.to_lowercase();
        let mut types = Vec::new();

        if lower.contains("research") || lower.contains("analyze") || lower.contains("investigate") {
            types.push(AgentType::Researcher);
        }

        if lower.contains("code") || lower.contains("implement") || lower.contains("build") {
            types.push(AgentType::Coder);
        }

        if lower.contains("test") || lower.contains("validate") {
            types.push(AgentType::Tester);
        }

        if lower.contains("review") || lower.contains("audit") {
            types.push(AgentType::Reviewer);
        }

        if lower.contains("optimize") || lower.contains("performance") {
            types.push(AgentType::Optimizer);
        }

        if lower.contains("document") || lower.contains("doc") {
            types.push(AgentType::Documenter);
        }

        if lower.contains("design") || lower.contains("architect") {
            types.push(AgentType::Architect);
        }

        // Default to coordinator if no specific type detected
        if types.is_empty() {
            types.push(AgentType::Coordinator);
        }

        types
    }

    /// Find a worker by type
    fn find_worker_by_type(&self, agent_type: &AgentType) -> Option<AgentId> {
        for entry in self.registered_workers.iter() {
            if entry.value() == agent_type {
                return Some(entry.key().clone());
            }
        }
        None
    }

    /// Create a subtask description for a specific agent type
    fn create_subtask_description(&self, original: &str, agent_type: &AgentType) -> String {
        format!("[{}] {}", agent_type, original)
    }

    /// Get queen status
    pub async fn status(&self) -> String {
        let active_tasks = self.active_tasks.read().await;
        format!(
            "{}: {} workers, {} active tasks",
            self.config.name,
            self.registered_workers.len(),
            active_tasks.len()
        )
    }

    /// Shutdown the queen
    pub async fn shutdown(&self) -> Result<()> {
        self.registered_workers.clear();
        let mut active = self.active_tasks.write().await;
        active.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_analyze_task_requirements() {
        let config = QueenConfig::default();
        let queen = Queen::new(config).unwrap();

        let types = queen.analyze_task_requirements("Research and implement a new feature");
        assert!(types.contains(&AgentType::Researcher));
        assert!(types.contains(&AgentType::Coder));
    }

    #[tokio::test]
    async fn test_register_worker() {
        let config = QueenConfig::default();
        let queen = Queen::new(config).unwrap();

        let agent_id = AgentId::new();
        let result = queen.register_worker(agent_id, AgentType::Coder).await;
        assert!(result.is_ok());
    }
}
