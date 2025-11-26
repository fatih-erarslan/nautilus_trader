// Sandbox manager for pool management and lifecycle

use super::{E2bClient, Sandbox, SandboxConfig, SandboxStatus, SandboxExecutor, ExecutionRequest, ExecutionResult};
use crate::{DistributedError, Result};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use tokio::time::{Duration, interval};

/// Sandbox pool for managing multiple sandboxes
pub struct SandboxPool {
    /// Available sandboxes
    sandboxes: Arc<RwLock<HashMap<String, Sandbox>>>,

    /// E2B client
    client: Arc<E2bClient>,

    /// Pool configuration
    config: PoolConfig,

    /// Executor for running code
    executor: Arc<SandboxExecutor>,
}

/// Pool configuration
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Minimum number of sandboxes to maintain
    pub min_size: usize,

    /// Maximum number of sandboxes
    pub max_size: usize,

    /// Idle timeout before cleanup
    pub idle_timeout: Duration,

    /// Maximum age before recycling
    pub max_age: Duration,

    /// Auto-scaling enabled
    pub auto_scale: bool,

    /// Default sandbox configuration
    pub default_sandbox_config: SandboxConfig,
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_size: 2,
            max_size: 10,
            idle_timeout: Duration::from_secs(300),
            max_age: Duration::from_secs(3600),
            auto_scale: true,
            default_sandbox_config: SandboxConfig::default(),
        }
    }
}

impl SandboxPool {
    /// Create new sandbox pool
    pub fn new(client: Arc<E2bClient>, config: PoolConfig) -> Self {
        let executor = Arc::new(SandboxExecutor::new(Arc::clone(&client)));

        Self {
            sandboxes: Arc::new(RwLock::new(HashMap::new())),
            client,
            config,
            executor,
        }
    }

    /// Initialize pool with minimum sandboxes
    pub async fn initialize(&self) -> Result<()> {
        for _ in 0..self.config.min_size {
            self.create_sandbox(self.config.default_sandbox_config.clone())
                .await?;
        }

        Ok(())
    }

    /// Create a new sandbox
    pub async fn create_sandbox(&self, config: SandboxConfig) -> Result<String> {
        // Check pool size limit
        let current_size = self.sandboxes.read().await.len();
        if current_size >= self.config.max_size {
            return Err(DistributedError::ResourceLimitExceeded(
                format!("Pool at maximum size: {}", self.config.max_size),
            ));
        }

        // Create sandbox via E2B API
        let sandbox_id = self.client.create_sandbox(&config.template).await?;
        let mut sandbox = Sandbox::new(sandbox_id.clone(), config);
        sandbox.set_status(SandboxStatus::Ready);

        // Add to pool
        self.sandboxes.write().await.insert(sandbox_id.clone(), sandbox);

        tracing::info!("Created sandbox: {}", sandbox_id);
        Ok(sandbox_id)
    }

    /// Get an available sandbox
    pub async fn get_sandbox(&self) -> Result<String> {
        let sandboxes = self.sandboxes.read().await;

        // Find ready sandbox
        for (id, sandbox) in sandboxes.iter() {
            if sandbox.status == SandboxStatus::Ready {
                return Ok(id.clone());
            }
        }

        // No available sandboxes
        drop(sandboxes);

        // Try to create new sandbox if auto-scaling enabled
        if self.config.auto_scale {
            return self.create_sandbox(self.config.default_sandbox_config.clone()).await;
        }

        Err(DistributedError::E2bError(
            "No available sandboxes".to_string(),
        ))
    }

    /// Execute code in the pool
    pub async fn execute(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        let sandbox_id = self.get_sandbox().await?;

        let mut sandboxes = self.sandboxes.write().await;
        let sandbox = sandboxes
            .get_mut(&sandbox_id)
            .ok_or_else(|| DistributedError::SandboxNotFound(sandbox_id.clone()))?;

        self.executor.execute(sandbox, request).await
    }

    /// Delete a sandbox
    pub async fn delete_sandbox(&self, sandbox_id: &str) -> Result<()> {
        // Remove from pool
        let mut sandboxes = self.sandboxes.write().await;
        let sandbox = sandboxes
            .remove(sandbox_id)
            .ok_or_else(|| DistributedError::SandboxNotFound(sandbox_id.to_string()))?;

        drop(sandboxes);

        // Delete via E2B API
        self.client.delete_sandbox(&sandbox.id).await?;

        tracing::info!("Deleted sandbox: {}", sandbox_id);
        Ok(())
    }

    /// Clean up idle sandboxes
    pub async fn cleanup_idle(&self) -> Result<usize> {
        let mut to_delete = Vec::new();

        {
            let sandboxes = self.sandboxes.read().await;

            for (id, sandbox) in sandboxes.iter() {
                if sandbox.is_idle(self.config.idle_timeout)
                    && sandboxes.len() > self.config.min_size
                {
                    to_delete.push(id.clone());
                }
            }
        }

        let count = to_delete.len();
        for id in to_delete {
            let _ = self.delete_sandbox(&id).await;
        }

        Ok(count)
    }

    /// Start background maintenance task
    pub async fn start_maintenance(&self) {
        let pool = Arc::new(self.clone_for_maintenance());
        let mut interval = interval(Duration::from_secs(60));

        tokio::spawn(async move {
            loop {
                interval.tick().await;

                // Cleanup idle sandboxes
                if let Ok(count) = pool.cleanup_idle().await {
                    if count > 0 {
                        tracing::info!("Cleaned up {} idle sandboxes", count);
                    }
                }

                // Ensure minimum pool size
                let current_size = pool.sandboxes.read().await.len();
                if current_size < pool.config.min_size {
                    for _ in current_size..pool.config.min_size {
                        let _ = pool
                            .create_sandbox(pool.config.default_sandbox_config.clone())
                            .await;
                    }
                }
            }
        });
    }

    /// Get pool statistics
    pub async fn stats(&self) -> PoolStats {
        let sandboxes = self.sandboxes.read().await;

        let mut stats = PoolStats::default();
        stats.total_sandboxes = sandboxes.len();

        for sandbox in sandboxes.values() {
            match sandbox.status {
                SandboxStatus::Ready => stats.ready_sandboxes += 1,
                SandboxStatus::Running => stats.running_sandboxes += 1,
                SandboxStatus::Error => stats.error_sandboxes += 1,
                _ => {}
            }

            stats.total_executions += sandbox.execution_count;
            stats.total_cpu_time_ms += sandbox.total_cpu_time_ms;
        }

        stats
    }

    /// Helper for cloning pool reference for background tasks
    fn clone_for_maintenance(&self) -> Self {
        Self {
            sandboxes: Arc::clone(&self.sandboxes),
            client: Arc::clone(&self.client),
            config: self.config.clone(),
            executor: Arc::clone(&self.executor),
        }
    }
}

/// Pool statistics
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoolStats {
    /// Total number of sandboxes
    pub total_sandboxes: usize,

    /// Number of ready sandboxes
    pub ready_sandboxes: usize,

    /// Number of running sandboxes
    pub running_sandboxes: usize,

    /// Number of error sandboxes
    pub error_sandboxes: usize,

    /// Total executions across all sandboxes
    pub total_executions: u64,

    /// Total CPU time used
    pub total_cpu_time_ms: u64,
}

/// Sandbox manager for high-level operations
pub struct SandboxManager {
    /// Sandbox pool
    pool: Arc<SandboxPool>,
}

impl SandboxManager {
    /// Create new manager
    pub fn new(client: Arc<E2bClient>, config: PoolConfig) -> Self {
        Self {
            pool: Arc::new(SandboxPool::new(client, config)),
        }
    }

    /// Initialize manager
    pub async fn initialize(&self) -> Result<()> {
        self.pool.initialize().await?;
        self.pool.start_maintenance().await;
        Ok(())
    }

    /// Execute code
    pub async fn execute(&self, request: ExecutionRequest) -> Result<ExecutionResult> {
        self.pool.execute(request).await
    }

    /// Get statistics
    pub async fn stats(&self) -> PoolStats {
        self.pool.stats().await
    }

    /// Execute batch of requests
    pub async fn execute_batch(&self, requests: Vec<ExecutionRequest>) -> Result<Vec<ExecutionResult>> {
        let mut results = Vec::with_capacity(requests.len());

        for request in requests {
            match self.execute(request).await {
                Ok(result) => results.push(result),
                Err(e) => {
                    tracing::error!("Batch execution error: {}", e);
                    // Continue with other requests
                }
            }
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_config_default() {
        let config = PoolConfig::default();
        assert_eq!(config.min_size, 2);
        assert_eq!(config.max_size, 10);
        assert!(config.auto_scale);
    }

    #[test]
    fn test_pool_stats_default() {
        let stats = PoolStats::default();
        assert_eq!(stats.total_sandboxes, 0);
        assert_eq!(stats.ready_sandboxes, 0);
    }

    #[tokio::test]
    async fn test_pool_creation() {
        let client = Arc::new(E2bClient::new("test-key".to_string()));
        let config = PoolConfig::default();
        let pool = SandboxPool::new(client, config);

        let stats = pool.stats().await;
        assert_eq!(stats.total_sandboxes, 0);
    }
}
