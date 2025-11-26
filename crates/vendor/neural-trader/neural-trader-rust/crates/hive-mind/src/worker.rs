//! Worker agent implementation

use crate::{error::*, memory::DistributedMemory, types::*};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for a worker agent
#[derive(Clone)]
pub struct WorkerConfig {
    pub agent_type: AgentType,
    pub name: String,
    pub memory: Arc<DistributedMemory>,
}

/// A worker agent in the hive
pub struct Worker {
    id: AgentId,
    config: WorkerConfig,
    #[allow(dead_code)]
    capabilities: AgentCapabilities,
    active: Arc<RwLock<bool>>,
}

impl Worker {
    /// Create a new worker
    pub fn new(config: WorkerConfig) -> Result<Self> {
        let capabilities = AgentCapabilities::from_agent_type(&config.agent_type);

        Ok(Self {
            id: AgentId::new(),
            config,
            capabilities,
            active: Arc::new(RwLock::new(true)),
        })
    }

    /// Get the worker's ID
    pub fn id(&self) -> AgentId {
        self.id.clone()
    }

    /// Get the worker's type
    pub fn agent_type(&self) -> &AgentType {
        &self.config.agent_type
    }

    /// Check if worker is active
    pub fn is_active(&self) -> bool {
        // Use try_read to avoid blocking
        self.active.try_read().map(|a| *a).unwrap_or(false)
    }

    /// Execute a task
    pub async fn execute_task(&self, task: Task) -> Result<TaskResult> {
        // Check if task matches worker capabilities
        if !self.can_execute_task(&task) {
            return Err(HiveMindError::TaskExecutionFailed(format!(
                "Worker {} cannot execute task type",
                self.id
            )));
        }

        tracing::info!("Worker {} executing task: {}", self.id, task.description);

        // Simulate task execution
        // In a real implementation, this would call actual agent logic
        let output = self.process_task(&task).await?;

        // Store result in memory
        let result = TaskResult::success(task.id.clone(), self.id.clone(), output);

        self.config
            .memory
            .store_task_result(&task.id, &result)
            .await?;

        Ok(result)
    }

    /// Check if worker can execute a task
    fn can_execute_task(&self, task: &Task) -> bool {
        let desc = task.description.to_lowercase();

        match &self.config.agent_type {
            AgentType::Researcher => {
                desc.contains("research") || desc.contains("analyze")
            }
            AgentType::Coder => {
                desc.contains("code") || desc.contains("implement")
            }
            AgentType::Tester => {
                desc.contains("test") || desc.contains("validate")
            }
            AgentType::Reviewer => {
                desc.contains("review") || desc.contains("audit")
            }
            AgentType::Optimizer => {
                desc.contains("optimize") || desc.contains("performance")
            }
            AgentType::Documenter => {
                desc.contains("document") || desc.contains("doc")
            }
            AgentType::Architect => {
                desc.contains("design") || desc.contains("architect")
            }
            AgentType::Coordinator => true, // Can handle any task
            AgentType::Custom(_) => true,   // Custom agents are flexible
        }
    }

    /// Process the task (simulate work)
    async fn process_task(&self, task: &Task) -> Result<String> {
        // Simulate some work
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        Ok(format!(
            "Task '{}' completed by {} ({})",
            task.description, self.config.name, self.config.agent_type
        ))
    }

    /// Shutdown the worker
    pub async fn shutdown(&mut self) -> Result<()> {
        let mut active = self.active.write().await;
        *active = false;
        tracing::info!("Worker {} shutdown", self.id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_memory() -> Arc<DistributedMemory> {
        Arc::new(DistributedMemory::new(Default::default()).unwrap())
    }

    #[test]
    fn test_worker_creation() {
        let config = WorkerConfig {
            agent_type: AgentType::Coder,
            name: "test-coder".to_string(),
            memory: create_test_memory(),
        };

        let worker = Worker::new(config);
        assert!(worker.is_ok());
    }

    #[test]
    fn test_can_execute_task() {
        let config = WorkerConfig {
            agent_type: AgentType::Coder,
            name: "test-coder".to_string(),
            memory: create_test_memory(),
        };

        let worker = Worker::new(config).unwrap();
        let task = Task::new("Implement a new feature".to_string());

        assert!(worker.can_execute_task(&task));
    }

    #[tokio::test]
    async fn test_execute_task() {
        let config = WorkerConfig {
            agent_type: AgentType::Coder,
            name: "test-coder".to_string(),
            memory: create_test_memory(),
        };

        let worker = Worker::new(config).unwrap();
        let task = Task::new("Implement a new feature".to_string());

        let result = worker.execute_task(task).await;
        assert!(result.is_ok());
    }
}
