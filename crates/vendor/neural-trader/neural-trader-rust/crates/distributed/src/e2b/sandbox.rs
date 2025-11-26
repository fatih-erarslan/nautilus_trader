// Sandbox abstraction for E2B execution environments

use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use uuid::Uuid;

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Sandbox template (e.g., "python", "node", "rust")
    pub template: String,

    /// CPU limit (cores)
    pub cpu_limit: f32,

    /// Memory limit (MB)
    pub memory_limit_mb: u32,

    /// Disk limit (MB)
    pub disk_limit_mb: u32,

    /// Network access enabled
    pub network_enabled: bool,

    /// Execution timeout (seconds)
    pub timeout_seconds: u32,

    /// Environment variables
    pub env_vars: Vec<(String, String)>,

    /// Working directory
    pub working_dir: String,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            template: "node".to_string(),
            cpu_limit: 1.0,
            memory_limit_mb: 512,
            disk_limit_mb: 1024,
            network_enabled: true,
            timeout_seconds: 300,
            env_vars: Vec::new(),
            working_dir: "/workspace".to_string(),
        }
    }
}

/// Sandbox status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SandboxStatus {
    /// Sandbox is being created
    Creating,

    /// Sandbox is ready for execution
    Ready,

    /// Sandbox is currently executing code
    Running,

    /// Sandbox is stopped
    Stopped,

    /// Sandbox encountered an error
    Error,

    /// Sandbox is being destroyed
    Destroying,
}

/// Sandbox instance
#[derive(Debug, Clone)]
pub struct Sandbox {
    /// Unique sandbox identifier
    pub id: String,

    /// Sandbox configuration
    pub config: SandboxConfig,

    /// Current status
    pub status: SandboxStatus,

    /// Creation timestamp
    pub created_at: Instant,

    /// Last activity timestamp
    pub last_activity: Instant,

    /// Number of executions
    pub execution_count: u64,

    /// Total CPU time used (ms)
    pub total_cpu_time_ms: u64,

    /// Total memory used (bytes)
    pub total_memory_bytes: u64,
}

impl Sandbox {
    /// Create a new sandbox instance
    pub fn new(id: String, config: SandboxConfig) -> Self {
        let now = Instant::now();
        Self {
            id,
            config,
            status: SandboxStatus::Creating,
            created_at: now,
            last_activity: now,
            execution_count: 0,
            total_cpu_time_ms: 0,
            total_memory_bytes: 0,
        }
    }

    /// Update sandbox status
    pub fn set_status(&mut self, status: SandboxStatus) {
        self.status = status;
        self.last_activity = Instant::now();
    }

    /// Record execution statistics
    pub fn record_execution(&mut self, cpu_time_ms: u64, memory_bytes: u64) {
        self.execution_count += 1;
        self.total_cpu_time_ms += cpu_time_ms;
        self.total_memory_bytes += memory_bytes;
        self.last_activity = Instant::now();
    }

    /// Check if sandbox is idle
    pub fn is_idle(&self, timeout: Duration) -> bool {
        self.status == SandboxStatus::Ready
            && self.last_activity.elapsed() > timeout
    }

    /// Get sandbox age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get idle time
    pub fn idle_time(&self) -> Duration {
        self.last_activity.elapsed()
    }

    /// Validate configuration
    pub fn validate_config(&self) -> Result<()> {
        if self.config.cpu_limit <= 0.0 {
            return Err(DistributedError::ConfigError(
                "CPU limit must be positive".to_string(),
            ));
        }

        if self.config.memory_limit_mb == 0 {
            return Err(DistributedError::ConfigError(
                "Memory limit must be positive".to_string(),
            ));
        }

        if self.config.timeout_seconds == 0 {
            return Err(DistributedError::ConfigError(
                "Timeout must be positive".to_string(),
            ));
        }

        Ok(())
    }

    /// Check if resource limits are exceeded
    pub fn check_limits(&self) -> Result<()> {
        // Check execution count limit (prevent abuse)
        if self.execution_count > 10000 {
            return Err(DistributedError::ResourceLimitExceeded(
                format!("Execution count limit exceeded: {}", self.execution_count),
            ));
        }

        // Check age limit (sandboxes should be recycled)
        if self.age() > Duration::from_secs(3600) {
            return Err(DistributedError::ResourceLimitExceeded(
                format!("Sandbox too old: {:?}", self.age()),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_sandbox_creation() {
        let config = SandboxConfig::default();
        let sandbox = Sandbox::new("test-123".to_string(), config);

        assert_eq!(sandbox.id, "test-123");
        assert_eq!(sandbox.status, SandboxStatus::Creating);
        assert_eq!(sandbox.execution_count, 0);
    }

    #[test]
    fn test_sandbox_status_update() {
        let config = SandboxConfig::default();
        let mut sandbox = Sandbox::new("test-123".to_string(), config);

        sandbox.set_status(SandboxStatus::Ready);
        assert_eq!(sandbox.status, SandboxStatus::Ready);

        sandbox.set_status(SandboxStatus::Running);
        assert_eq!(sandbox.status, SandboxStatus::Running);
    }

    #[test]
    fn test_execution_recording() {
        let config = SandboxConfig::default();
        let mut sandbox = Sandbox::new("test-123".to_string(), config);

        sandbox.record_execution(100, 1024);
        assert_eq!(sandbox.execution_count, 1);
        assert_eq!(sandbox.total_cpu_time_ms, 100);
        assert_eq!(sandbox.total_memory_bytes, 1024);

        sandbox.record_execution(200, 2048);
        assert_eq!(sandbox.execution_count, 2);
        assert_eq!(sandbox.total_cpu_time_ms, 300);
        assert_eq!(sandbox.total_memory_bytes, 3072);
    }

    #[test]
    fn test_idle_detection() {
        let config = SandboxConfig::default();
        let mut sandbox = Sandbox::new("test-123".to_string(), config);
        sandbox.set_status(SandboxStatus::Ready);

        // Should not be idle immediately
        assert!(!sandbox.is_idle(Duration::from_millis(100)));

        // Wait and check again
        thread::sleep(Duration::from_millis(150));
        assert!(sandbox.is_idle(Duration::from_millis(100)));
    }

    #[test]
    fn test_config_validation() {
        let mut config = SandboxConfig::default();
        let sandbox = Sandbox::new("test-123".to_string(), config.clone());
        assert!(sandbox.validate_config().is_ok());

        config.cpu_limit = 0.0;
        let sandbox = Sandbox::new("test-123".to_string(), config);
        assert!(sandbox.validate_config().is_err());
    }

    #[test]
    fn test_default_config() {
        let config = SandboxConfig::default();
        assert_eq!(config.template, "node");
        assert_eq!(config.cpu_limit, 1.0);
        assert_eq!(config.memory_limit_mb, 512);
        assert!(config.network_enabled);
    }
}
