//! Distributed memory management for the hive

use crate::{error::*, types::*};
use dashmap::DashMap;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Configuration for distributed memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    /// Maximum memory entries
    pub max_entries: usize,

    /// Enable persistence
    pub persistence: bool,

    /// Cache size in MB
    pub cache_size_mb: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_entries: 10_000,
            persistence: true,
            cache_size_mb: 256,
        }
    }
}

/// Distributed memory store
pub struct DistributedMemory {
    config: MemoryConfig,
    store: Arc<DashMap<String, MemoryEntry>>,
    task_results: Arc<DashMap<String, TaskResult>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct MemoryEntry {
    key: String,
    value: String,
    created_at: chrono::DateTime<chrono::Utc>,
    updated_at: chrono::DateTime<chrono::Utc>,
    version: u64,
}

impl DistributedMemory {
    /// Create a new distributed memory instance
    pub fn new(config: MemoryConfig) -> Result<Self> {
        Ok(Self {
            config,
            store: Arc::new(DashMap::new()),
            task_results: Arc::new(DashMap::new()),
        })
    }

    /// Store a key-value pair
    pub async fn store(&self, key: String, value: String) -> Result<()> {
        let now = chrono::Utc::now();

        let entry = if let Some(mut existing) = self.store.get_mut(&key) {
            existing.value = value.clone();
            existing.updated_at = now;
            existing.version += 1;
            existing.clone()
        } else {
            MemoryEntry {
                key: key.clone(),
                value,
                created_at: now,
                updated_at: now,
                version: 1,
            }
        };

        self.store.insert(key, entry);
        Ok(())
    }

    /// Retrieve a value by key
    pub async fn retrieve(&self, key: &str) -> Result<Option<String>> {
        Ok(self.store.get(key).map(|entry| entry.value.clone()))
    }

    /// Store a task result
    pub async fn store_task_result(&self, task_id: &str, result: &TaskResult) -> Result<()> {
        self.task_results.insert(task_id.to_string(), result.clone());
        Ok(())
    }

    /// Retrieve a task result
    pub async fn get_task_result(&self, task_id: &str) -> Result<Option<TaskResult>> {
        Ok(self.task_results.get(task_id).map(|r| r.clone()))
    }

    /// Store the final result of a task
    pub async fn store_result(&self, task: &Task, result: &TaskResult) -> Result<()> {
        // Store in both locations for redundancy
        self.store_task_result(&task.id, result).await?;

        // Also store as a general memory entry
        let key = format!("result:{}", task.id);
        let value = serde_json::to_string(result)?;
        self.store(key, value).await?;

        Ok(())
    }

    /// Get current memory usage
    pub async fn usage(&self) -> usize {
        self.store.len() + self.task_results.len()
    }

    /// Clear all memory
    pub async fn clear(&self) -> Result<()> {
        self.store.clear();
        self.task_results.clear();
        Ok(())
    }

    /// List all keys
    pub async fn list_keys(&self) -> Vec<String> {
        self.store.iter().map(|entry| entry.key().clone()).collect()
    }

    /// Get memory statistics
    pub async fn stats(&self) -> MemoryStats {
        MemoryStats {
            total_entries: self.store.len(),
            task_results: self.task_results.len(),
            max_entries: self.config.max_entries,
            usage_percent: (self.store.len() as f64 / self.config.max_entries as f64 * 100.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStats {
    pub total_entries: usize,
    pub task_results: usize,
    pub max_entries: usize,
    pub usage_percent: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_store_and_retrieve() {
        let config = MemoryConfig::default();
        let memory = DistributedMemory::new(config).unwrap();

        memory.store("test_key".to_string(), "test_value".to_string()).await.unwrap();

        let value = memory.retrieve("test_key").await.unwrap();
        assert_eq!(value, Some("test_value".to_string()));
    }

    #[tokio::test]
    async fn test_task_result_storage() {
        let config = MemoryConfig::default();
        let memory = DistributedMemory::new(config).unwrap();

        let result = TaskResult::success(
            "task-1".to_string(),
            AgentId::new(),
            "Success".to_string(),
        );

        memory.store_task_result("task-1", &result).await.unwrap();

        let retrieved = memory.get_task_result("task-1").await.unwrap();
        assert!(retrieved.is_some());
    }

    #[tokio::test]
    async fn test_memory_stats() {
        let config = MemoryConfig::default();
        let memory = DistributedMemory::new(config).unwrap();

        memory.store("key1".to_string(), "value1".to_string()).await.unwrap();
        memory.store("key2".to_string(), "value2".to_string()).await.unwrap();

        let stats = memory.stats().await;
        assert_eq!(stats.total_entries, 2);
    }
}
