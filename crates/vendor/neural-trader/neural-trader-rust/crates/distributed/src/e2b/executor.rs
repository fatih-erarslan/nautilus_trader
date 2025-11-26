// Sandbox executor for running code in isolated environments

use super::{E2bClient, Sandbox, SandboxResult, SandboxStatus};
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::time::{timeout, Duration};
use uuid::Uuid;

/// Execution request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRequest {
    /// Code to execute
    pub code: String,

    /// Language/runtime
    pub language: String,

    /// Execution timeout (seconds)
    pub timeout_seconds: u32,

    /// Input data
    pub input: Option<String>,

    /// Environment variables
    pub env_vars: Vec<(String, String)>,
}

/// Execution result with additional context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Execution ID
    pub id: Uuid,

    /// Request that was executed
    pub request: ExecutionRequest,

    /// Sandbox result
    pub result: SandboxResult,

    /// Success flag
    pub success: bool,

    /// Error message (if failed)
    pub error: Option<String>,
}

/// Sandbox executor
pub struct SandboxExecutor {
    /// E2B client
    client: Arc<E2bClient>,

    /// Default timeout
    default_timeout: Duration,
}

impl SandboxExecutor {
    /// Create new executor
    pub fn new(client: Arc<E2bClient>) -> Self {
        Self {
            client,
            default_timeout: Duration::from_secs(300),
        }
    }

    /// Execute code in a sandbox
    pub async fn execute(
        &self,
        sandbox: &mut Sandbox,
        request: ExecutionRequest,
    ) -> Result<ExecutionResult> {
        // Validate sandbox is ready
        if sandbox.status != SandboxStatus::Ready {
            return Err(DistributedError::E2bError(format!(
                "Sandbox not ready: {:?}",
                sandbox.status
            )));
        }

        // Check resource limits
        sandbox.check_limits()?;

        // Update status to running
        sandbox.set_status(SandboxStatus::Running);

        // Execute with timeout
        let timeout_duration = Duration::from_secs(request.timeout_seconds as u64);
        let execution_id = Uuid::new_v4();

        let result = match timeout(
            timeout_duration,
            self.client.execute(&sandbox.id, &request.code),
        )
        .await
        {
            Ok(Ok(result)) => {
                // Record execution statistics
                sandbox.record_execution(
                    result.duration_ms,
                    result.resource_usage.peak_memory_bytes,
                );

                // Update status back to ready
                sandbox.set_status(SandboxStatus::Ready);

                ExecutionResult {
                    id: execution_id,
                    request: request.clone(),
                    result: result.clone(),
                    success: result.exit_code == 0,
                    error: if result.exit_code != 0 {
                        Some(result.stderr.clone())
                    } else {
                        None
                    },
                }
            }
            Ok(Err(e)) => {
                sandbox.set_status(SandboxStatus::Error);
                return Err(e);
            }
            Err(_) => {
                sandbox.set_status(SandboxStatus::Error);
                return Err(DistributedError::Timeout);
            }
        };

        Ok(result)
    }

    /// Execute multiple requests in parallel
    pub async fn execute_batch(
        &self,
        sandboxes: &mut [Sandbox],
        requests: Vec<ExecutionRequest>,
    ) -> Result<Vec<ExecutionResult>> {
        if sandboxes.is_empty() {
            return Err(DistributedError::E2bError(
                "No sandboxes available".to_string(),
            ));
        }

        let mut results = Vec::with_capacity(requests.len());
        let mut tasks = Vec::new();

        for (i, request) in requests.into_iter().enumerate() {
            let sandbox_idx = i % sandboxes.len();
            let sandbox = &mut sandboxes[sandbox_idx];

            // Clone necessary data for the task
            let executor = self.clone_executor();
            let mut sandbox_clone = sandbox.clone();

            let task = tokio::spawn(async move {
                executor.execute(&mut sandbox_clone, request).await
            });

            tasks.push(task);
        }

        // Wait for all tasks to complete
        for task in tasks {
            match task.await {
                Ok(Ok(result)) => results.push(result),
                Ok(Err(e)) => return Err(e),
                Err(e) => return Err(DistributedError::E2bError(format!("Task join error: {}", e))),
            }
        }

        Ok(results)
    }

    /// Helper to clone executor for parallel execution
    fn clone_executor(&self) -> Self {
        Self {
            client: Arc::clone(&self.client),
            default_timeout: self.default_timeout,
        }
    }

    /// Execute with retry logic
    pub async fn execute_with_retry(
        &self,
        sandbox: &mut Sandbox,
        request: ExecutionRequest,
        max_retries: u32,
    ) -> Result<ExecutionResult> {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts < max_retries {
            match self.execute(sandbox, request.clone()).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e);

                    if attempts < max_retries {
                        // Wait before retrying (exponential backoff)
                        let wait_ms = 2_u64.pow(attempts) * 100;
                        tokio::time::sleep(Duration::from_millis(wait_ms)).await;
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            DistributedError::E2bError("Max retries exceeded".to_string())
        }))
    }

    /// Validate execution request
    pub fn validate_request(&self, request: &ExecutionRequest) -> Result<()> {
        if request.code.is_empty() {
            return Err(DistributedError::E2bError(
                "Code cannot be empty".to_string(),
            ));
        }

        if request.timeout_seconds == 0 {
            return Err(DistributedError::E2bError(
                "Timeout must be positive".to_string(),
            ));
        }

        if request.timeout_seconds > 3600 {
            return Err(DistributedError::E2bError(
                "Timeout too long (max 3600s)".to_string(),
            ));
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::e2b::SandboxConfig;

    #[test]
    fn test_execution_request_validation() {
        let client = Arc::new(E2bClient::new("test-key".to_string()));
        let executor = SandboxExecutor::new(client);

        let valid_request = ExecutionRequest {
            code: "console.log('hello')".to_string(),
            language: "node".to_string(),
            timeout_seconds: 60,
            input: None,
            env_vars: vec![],
        };

        assert!(executor.validate_request(&valid_request).is_ok());

        let invalid_request = ExecutionRequest {
            code: String::new(),
            language: "node".to_string(),
            timeout_seconds: 60,
            input: None,
            env_vars: vec![],
        };

        assert!(executor.validate_request(&invalid_request).is_err());
    }

    #[test]
    fn test_execution_result_serialization() {
        let request = ExecutionRequest {
            code: "test".to_string(),
            language: "node".to_string(),
            timeout_seconds: 60,
            input: None,
            env_vars: vec![],
        };

        let result = ExecutionResult {
            id: Uuid::new_v4(),
            request: request.clone(),
            result: SandboxResult {
                execution_id: Uuid::new_v4(),
                sandbox_id: "test".to_string(),
                exit_code: 0,
                stdout: "output".to_string(),
                stderr: String::new(),
                duration_ms: 100,
                resource_usage: super::super::ResourceUsage::default(),
                metadata: std::collections::HashMap::new(),
            },
            success: true,
            error: None,
        };

        let json = serde_json::to_string(&result).unwrap();
        let deserialized: ExecutionResult = serde_json::from_str(&json).unwrap();
        assert_eq!(result.success, deserialized.success);
    }
}
