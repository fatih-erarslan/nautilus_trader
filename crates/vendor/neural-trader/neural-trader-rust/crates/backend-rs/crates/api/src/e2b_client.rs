use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use reqwest::{Client, StatusCode};
use std::collections::HashMap;
use tokio::time::{Duration, sleep};

#[derive(Debug, Clone)]
pub struct E2BClient {
    pub api_key: String,
    client: Client,
    base_url: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct SandboxConfig {
    #[serde(rename = "templateID")]
    pub template: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>, // seconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_vars: Option<HashMap<String, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Sandbox {
    #[serde(rename = "sandboxID")]
    pub sandbox_id: String,
    #[serde(rename = "templateID")]
    pub template: String,
    #[serde(default)]
    pub status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ExecutionRequest {
    pub code: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub env_vars: Option<HashMap<String, String>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ExecutionResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub execution_time_ms: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct FileUpload {
    pub path: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LogEntry {
    pub timestamp: String,
    pub level: String,
    pub message: String,
}

impl E2BClient {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("Failed to build HTTP client"),
            base_url: "https://api.e2b.dev".to_string(),
        }
    }

    /// Create a new sandbox
    pub async fn create_sandbox(&self, config: SandboxConfig) -> Result<Sandbox> {
        let url = format!("{}/sandboxes", self.base_url);

        let response = self.client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&config)
            .send()
            .await
            .context("Failed to send sandbox creation request")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Sandbox creation failed ({}): {}", status, error_text);
        }

        let mut sandbox: Sandbox = response.json().await
            .context("Failed to parse sandbox response")?;
        
        // E2B sandboxes are immediately ready, set status
        if sandbox.status.is_empty() {
            sandbox.status = "running".to_string();
        }

        Ok(sandbox)
    }

    /// Wait for sandbox to be ready
    async fn wait_for_sandbox(&self, sandbox_id: &str, timeout_secs: u64) -> Result<()> {
        let start = std::time::Instant::now();

        loop {
            let status = self.get_sandbox_status(sandbox_id).await?;

            if status == "running" {
                return Ok(());
            } else if status == "failed" {
                anyhow::bail!("Sandbox failed to start");
            }

            if start.elapsed().as_secs() > timeout_secs {
                anyhow::bail!("Sandbox startup timeout");
            }

            sleep(Duration::from_secs(2)).await;
        }
    }

    /// Get sandbox status
    pub async fn get_sandbox_status(&self, sandbox_id: &str) -> Result<String> {
        let url = format!("{}/sandboxes/{}", self.base_url, sandbox_id);

        let response = self.client
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to get sandbox status")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get sandbox status: {}", response.status());
        }

        let sandbox: Sandbox = response.json().await?;
        Ok(sandbox.status)
    }

    /// Execute code in sandbox
    pub async fn execute_code(&self, sandbox_id: &str, request: ExecutionRequest) -> Result<ExecutionResult> {
        let url = format!("{}/sandboxes/{}/execute", self.base_url, sandbox_id);

        let start = std::time::Instant::now();

        let response = self.client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .context("Failed to execute code")?;

        let execution_time_ms = start.elapsed().as_millis() as u64;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();

            return Ok(ExecutionResult {
                stdout: String::new(),
                stderr: error_text.clone(),
                exit_code: -1,
                error: Some(format!("Execution failed ({}): {}", status, error_text)),
                execution_time_ms,
            });
        }

        let mut result: ExecutionResult = response.json().await
            .context("Failed to parse execution result")?;

        result.execution_time_ms = execution_time_ms;
        Ok(result)
    }

    /// Upload file to sandbox
    pub async fn upload_file(&self, sandbox_id: &str, file: FileUpload) -> Result<()> {
        let url = format!("{}/sandboxes/{}/files", self.base_url, sandbox_id);

        let response = self.client
            .post(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&file)
            .send()
            .await
            .context("Failed to upload file")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("File upload failed ({}): {}", status, error_text);
        }

        Ok(())
    }

    /// Get sandbox logs
    pub async fn get_logs(&self, sandbox_id: &str, lines: usize) -> Result<Vec<LogEntry>> {
        let url = format!("{}/sandboxes/{}/logs?lines={}", self.base_url, sandbox_id, lines);

        let response = self.client
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to get logs")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to get logs: {}", response.status());
        }

        let logs: Vec<LogEntry> = response.json().await
            .context("Failed to parse logs")?;

        Ok(logs)
    }

    /// Terminate sandbox
    pub async fn terminate_sandbox(&self, sandbox_id: &str) -> Result<()> {
        let url = format!("{}/sandboxes/{}", self.base_url, sandbox_id);

        let response = self.client
            .delete(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to terminate sandbox")?;

        if !response.status().is_success() && response.status() != StatusCode::NOT_FOUND {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Sandbox termination failed ({}): {}", status, error_text);
        }

        Ok(())
    }

    /// List all sandboxes
    pub async fn list_sandboxes(&self) -> Result<Vec<Sandbox>> {
        let url = format!("{}/sandboxes", self.base_url);

        let response = self.client
            .get(&url)
            .header("X-API-Key", &self.api_key)
            .send()
            .await
            .context("Failed to list sandboxes")?;

        if !response.status().is_success() {
            anyhow::bail!("Failed to list sandboxes: {}", response.status());
        }

        let sandboxes: Vec<Sandbox> = response.json().await
            .context("Failed to parse sandboxes list")?;

        Ok(sandboxes)
    }

    /// Configure sandbox environment
    pub async fn configure_sandbox(
        &self,
        sandbox_id: &str,
        env_vars: HashMap<String, String>
    ) -> Result<()> {
        let url = format!("{}/sandboxes/{}/config", self.base_url, sandbox_id);

        let response = self.client
            .patch(&url)
            .header("X-API-Key", &self.api_key)
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({ "env_vars": env_vars }))
            .send()
            .await
            .context("Failed to configure sandbox")?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            anyhow::bail!("Sandbox configuration failed ({}): {}", status, error_text);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_client_creation() {
        let client = E2BClient::new("test_key".to_string());
        assert_eq!(client.api_key, "test_key");
    }

    #[test]
    fn test_sandbox_config_serialization() {
        let config = SandboxConfig {
            template: "base".to_string(),
            timeout: Some(3600),
            env_vars: Some(HashMap::from([
                ("API_KEY".to_string(), "test".to_string()),
            ])),
            metadata: None,
        };

        let json = serde_json::to_string(&config).unwrap();
        assert!(json.contains("templateID"));
        assert!(json.contains("base"));
    }
}
