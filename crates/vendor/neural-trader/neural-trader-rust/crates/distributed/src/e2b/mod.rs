// E2B sandbox integration module
//
// Provides isolated execution environments for:
// - Safe strategy backtesting
// - User-submitted code execution
// - Resource-intensive computations
// - Parallel test execution

mod sandbox;
mod executor;
mod manager;

pub use sandbox::{Sandbox, SandboxConfig, SandboxStatus};
pub use executor::{SandboxExecutor, ExecutionRequest, ExecutionResult};
pub use manager::{SandboxManager, SandboxPool};

use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Result of sandbox execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxResult {
    /// Unique execution ID
    pub execution_id: Uuid,

    /// Sandbox ID where execution occurred
    pub sandbox_id: String,

    /// Exit code (0 = success)
    pub exit_code: i32,

    /// Standard output
    pub stdout: String,

    /// Standard error
    pub stderr: String,

    /// Execution duration in milliseconds
    pub duration_ms: u64,

    /// Resource usage statistics
    pub resource_usage: ResourceUsage,

    /// Execution metadata
    pub metadata: HashMap<String, String>,
}

/// Resource usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time in milliseconds
    pub cpu_time_ms: u64,

    /// Peak memory usage in bytes
    pub peak_memory_bytes: u64,

    /// Network bytes sent
    pub network_sent_bytes: u64,

    /// Network bytes received
    pub network_received_bytes: u64,

    /// Disk I/O bytes
    pub disk_io_bytes: u64,
}

impl Default for ResourceUsage {
    fn default() -> Self {
        Self {
            cpu_time_ms: 0,
            peak_memory_bytes: 0,
            network_sent_bytes: 0,
            network_received_bytes: 0,
            disk_io_bytes: 0,
        }
    }
}

/// E2B API client for Node.js bridge
#[derive(Debug, Clone)]
pub struct E2bClient {
    api_key: String,
    base_url: String,
    http_client: reqwest::Client,
}

impl E2bClient {
    /// Create new E2B client
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            base_url: "https://api.e2b.dev".to_string(),
            http_client: reqwest::Client::new(),
        }
    }

    /// Create a new sandbox
    pub async fn create_sandbox(&self, template: &str) -> Result<String> {
        let response = self
            .http_client
            .post(format!("{}/sandboxes", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "template": template,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(DistributedError::E2bError(format!(
                "Failed to create sandbox: {}",
                response.status()
            )));
        }

        let data: serde_json::Value = response.json().await?;
        let sandbox_id = data["sandboxId"]
            .as_str()
            .ok_or_else(|| DistributedError::E2bError("No sandbox ID in response".to_string()))?;

        Ok(sandbox_id.to_string())
    }

    /// Execute code in sandbox
    pub async fn execute(&self, sandbox_id: &str, code: &str) -> Result<SandboxResult> {
        let response = self
            .http_client
            .post(format!("{}/sandboxes/{}/execute", self.base_url, sandbox_id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&serde_json::json!({
                "code": code,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(DistributedError::E2bError(format!(
                "Execution failed: {}",
                response.status()
            )));
        }

        let data: serde_json::Value = response.json().await?;

        Ok(SandboxResult {
            execution_id: Uuid::new_v4(),
            sandbox_id: sandbox_id.to_string(),
            exit_code: data["exitCode"].as_i64().unwrap_or(0) as i32,
            stdout: data["stdout"].as_str().unwrap_or("").to_string(),
            stderr: data["stderr"].as_str().unwrap_or("").to_string(),
            duration_ms: data["durationMs"].as_u64().unwrap_or(0),
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        })
    }

    /// Delete sandbox
    pub async fn delete_sandbox(&self, sandbox_id: &str) -> Result<()> {
        let response = self
            .http_client
            .delete(format!("{}/sandboxes/{}", self.base_url, sandbox_id))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(DistributedError::E2bError(format!(
                "Failed to delete sandbox: {}",
                response.status()
            )));
        }

        Ok(())
    }

    /// List all sandboxes
    pub async fn list_sandboxes(&self) -> Result<Vec<String>> {
        let response = self
            .http_client
            .get(format!("{}/sandboxes", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(DistributedError::E2bError(format!(
                "Failed to list sandboxes: {}",
                response.status()
            )));
        }

        let data: serde_json::Value = response.json().await?;
        let sandboxes = data["sandboxes"]
            .as_array()
            .ok_or_else(|| DistributedError::E2bError("Invalid response format".to_string()))?;

        Ok(sandboxes
            .iter()
            .filter_map(|s| s["sandboxId"].as_str().map(String::from))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_usage_default() {
        let usage = ResourceUsage::default();
        assert_eq!(usage.cpu_time_ms, 0);
        assert_eq!(usage.peak_memory_bytes, 0);
    }

    #[test]
    fn test_sandbox_result_serialization() {
        let result = SandboxResult {
            execution_id: Uuid::new_v4(),
            sandbox_id: "test-sandbox".to_string(),
            exit_code: 0,
            stdout: "Hello, world!".to_string(),
            stderr: String::new(),
            duration_ms: 100,
            resource_usage: ResourceUsage::default(),
            metadata: HashMap::new(),
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: SandboxResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.sandbox_id, deserialized.sandbox_id);
    }
}
