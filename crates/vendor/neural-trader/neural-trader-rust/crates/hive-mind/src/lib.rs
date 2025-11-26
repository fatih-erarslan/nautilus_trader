//! # Hive Mind - Distributed Multi-Agent Coordination
//!
//! This crate implements a distributed hive mind system for coordinating
//! multiple AI agents in a fault-tolerant, consensus-driven architecture.
//!
//! ## Architecture
//!
//! - **Queen**: Central coordinator that manages worker agents
//! - **Workers**: Specialized agents that execute tasks
//! - **Memory**: Distributed shared memory with conflict resolution
//! - **Consensus**: Democratic decision-making mechanisms
//!
//! ## Example
//!
//! ```rust,no_run
//! use nt_hive_mind::{HiveMind, HiveMindConfig, AgentType};
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let config = HiveMindConfig::default();
//!     let mut hive = HiveMind::new(config)?;
//!
//!     // Spawn worker agents
//!     hive.spawn_worker(AgentType::Researcher, "research-1".to_string()).await?;
//!     hive.spawn_worker(AgentType::Coder, "coder-1".to_string()).await?;
//!
//!     // Orchestrate a task
//!     let result = hive.orchestrate_task("Build a trading strategy").await?;
//!
//!     Ok(())
//! }
//! ```

pub mod consensus;
pub mod error;
pub mod memory;
pub mod queen;
pub mod types;
pub mod worker;

pub use consensus::{ConsensusBuilder, ConsensusConfig, ConsensusResult};
pub use error::{HiveMindError, Result};
pub use memory::{DistributedMemory, MemoryConfig};
pub use queen::{Queen, QueenConfig};
pub use types::{AgentId, AgentType, Task, TaskResult};
pub use worker::{Worker, WorkerConfig};

use dashmap::DashMap;
use std::sync::Arc;

/// Main HiveMind coordinator struct
#[derive(Clone)]
pub struct HiveMind {
    queen: Arc<Queen>,
    workers: Arc<DashMap<AgentId, Worker>>,
    memory: Arc<DistributedMemory>,
    consensus: Arc<ConsensusBuilder>,
    config: Arc<HiveMindConfig>,
}

/// Configuration for the HiveMind system
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HiveMindConfig {
    /// Maximum number of worker agents
    pub max_workers: usize,

    /// Queen configuration
    pub queen_config: QueenConfig,

    /// Memory configuration
    pub memory_config: MemoryConfig,

    /// Consensus configuration
    pub consensus_config: ConsensusConfig,

    /// Enable fault tolerance
    pub fault_tolerance: bool,

    /// Enable collective intelligence
    pub collective_intelligence: bool,
}

impl Default for HiveMindConfig {
    fn default() -> Self {
        Self {
            max_workers: 10,
            queen_config: QueenConfig::default(),
            memory_config: MemoryConfig::default(),
            consensus_config: ConsensusConfig::default(),
            fault_tolerance: true,
            collective_intelligence: true,
        }
    }
}

impl HiveMind {
    /// Create a new HiveMind instance
    pub fn new(config: HiveMindConfig) -> Result<Self> {
        let queen = Arc::new(Queen::new(config.queen_config.clone())?);
        let workers = Arc::new(DashMap::new());
        let memory = Arc::new(DistributedMemory::new(config.memory_config.clone())?);
        let consensus = Arc::new(ConsensusBuilder::new(config.consensus_config.clone())?);

        Ok(Self {
            queen,
            workers,
            memory,
            consensus,
            config: Arc::new(config),
        })
    }

    /// Spawn a new worker agent
    pub async fn spawn_worker(&mut self, agent_type: AgentType, name: String) -> Result<AgentId> {
        if self.workers.len() >= self.config.max_workers {
            return Err(HiveMindError::MaxWorkersReached);
        }

        let worker_config = WorkerConfig {
            agent_type: agent_type.clone(),
            name: name.clone(),
            memory: self.memory.clone(),
        };

        let worker = Worker::new(worker_config)?;
        let agent_id = worker.id();

        self.workers.insert(agent_id.clone(), worker);
        self.queen.register_worker(agent_id.clone(), agent_type).await?;

        tracing::info!("Spawned worker: {} ({})", name, agent_id);
        Ok(agent_id)
    }

    /// Orchestrate a task across the hive
    pub async fn orchestrate_task(&self, task_description: &str) -> Result<TaskResult> {
        let task = Task::new(task_description.to_string());

        // Queen delegates task to appropriate workers
        let assignments = self.queen.delegate_task(&task, &self.workers).await?;

        // Workers execute their assigned parts
        let mut results = Vec::new();
        for (agent_id, subtask) in assignments {
            if let Some(worker) = self.workers.get(&agent_id) {
                let result = worker.execute_task(subtask).await?;
                results.push(result);
            }
        }

        // Build consensus on results
        let final_result = self.consensus.build_consensus(&results).await?;

        // Store in collective memory
        self.memory.store_result(&task, &final_result).await?;

        Ok(final_result)
    }

    /// Get the current status of the hive
    pub async fn status(&self) -> HiveMindStatus {
        HiveMindStatus {
            total_workers: self.workers.len(),
            active_workers: self.workers.iter().filter(|w| w.is_active()).count(),
            memory_usage: self.memory.usage().await,
            queen_status: self.queen.status().await,
        }
    }

    /// Shutdown the hive gracefully
    pub async fn shutdown(&mut self) -> Result<()> {
        tracing::info!("Shutting down HiveMind...");

        // Stop all workers
        for mut entry in self.workers.iter_mut() {
            entry.shutdown().await?;
        }

        // Clear workers
        self.workers.clear();

        // Shutdown queen
        self.queen.shutdown().await?;

        tracing::info!("HiveMind shutdown complete");
        Ok(())
    }
}

/// Status information for the HiveMind
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HiveMindStatus {
    pub total_workers: usize,
    pub active_workers: usize,
    pub memory_usage: usize,
    pub queen_status: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_hive_mind_creation() {
        let config = HiveMindConfig::default();
        let hive = HiveMind::new(config);
        assert!(hive.is_ok());
    }

    #[tokio::test]
    async fn test_spawn_worker() {
        let config = HiveMindConfig::default();
        let mut hive = HiveMind::new(config).unwrap();

        let result = hive.spawn_worker(AgentType::Researcher, "test-worker".to_string()).await;
        assert!(result.is_ok());

        let status = hive.status().await;
        assert_eq!(status.total_workers, 1);
    }
}
