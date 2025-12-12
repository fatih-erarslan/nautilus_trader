//! Sandbox management

use crate::{
    client::E2BClient,
    error::{Error, Result},
    types::{CodeRequest, ExecutionResult, FileInfo},
    DEFAULT_TEMPLATE,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// Sandbox configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    /// Template ID (e.g., "base", "python", "nodejs")
    pub template: String,
    /// Timeout in milliseconds
    pub timeout_ms: Option<u64>,
    /// Custom metadata
    pub metadata: Option<HashMap<String, String>>,
    /// Keep sandbox alive after operations
    pub keep_alive: bool,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            template: DEFAULT_TEMPLATE.to_string(),
            timeout_ms: Some(300_000), // 5 minutes
            keep_alive: false,
            metadata: None,
        }
    }
}

/// Sandbox status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SandboxStatus {
    Creating,
    Running,
    Paused,
    Closing,
    Closed,
}

/// A sandbox instance
pub struct Sandbox {
    id: String,
    client_id: String,
    config: SandboxConfig,
    client: E2BClient,
    closed: Arc<AtomicBool>,
}

impl Sandbox {
    /// Create a new sandbox handle
    pub(crate) fn new(
        id: String,
        client_id: String,
        config: SandboxConfig,
        client: E2BClient,
    ) -> Self {
        Self {
            id,
            client_id,
            config,
            client,
            closed: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Get sandbox ID
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get client ID
    pub fn client_id(&self) -> &str {
        &self.client_id
    }

    /// Check if sandbox is closed
    pub fn is_closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }

    /// Execute code in the sandbox
    pub async fn execute(&self, code: &str) -> Result<ExecutionResult> {
        self.execute_with_options(CodeRequest {
            code: code.to_string(),
            ..Default::default()
        })
        .await
    }

    /// Execute code with full options
    pub async fn execute_with_options(&self, request: CodeRequest) -> Result<ExecutionResult> {
        if self.is_closed() {
            return Err(Error::SandboxClosed);
        }

        let url = format!(
            "{}/sandboxes/{}/code/execution",
            self.client.base_url(),
            self.id
        );

        let response = self
            .client
            .http()
            .post(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let result: ExecutionResult = response.json().await?;
        Ok(result)
    }

    /// Run a shell command
    pub async fn run_command(&self, command: &str) -> Result<ExecutionResult> {
        self.run_command_with_options(command, None, None).await
    }

    /// Run a shell command with options
    pub async fn run_command_with_options(
        &self,
        command: &str,
        cwd: Option<&str>,
        env: Option<HashMap<String, String>>,
    ) -> Result<ExecutionResult> {
        if self.is_closed() {
            return Err(Error::SandboxClosed);
        }

        let url = format!(
            "{}/sandboxes/{}/commands",
            self.client.base_url(),
            self.id
        );

        #[derive(Serialize)]
        struct CommandRequest {
            command: String,
            #[serde(skip_serializing_if = "Option::is_none")]
            cwd: Option<String>,
            #[serde(skip_serializing_if = "Option::is_none")]
            env: Option<HashMap<String, String>>,
        }

        let request = CommandRequest {
            command: command.to_string(),
            cwd: cwd.map(|s| s.to_string()),
            env,
        };

        let response = self
            .client
            .http()
            .post(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let result: ExecutionResult = response.json().await?;
        Ok(result)
    }

    /// Write a file to the sandbox
    pub async fn write_file(&self, path: &str, content: &[u8]) -> Result<()> {
        if self.is_closed() {
            return Err(Error::SandboxClosed);
        }

        let url = format!(
            "{}/sandboxes/{}/files",
            self.client.base_url(),
            self.id
        );

        #[derive(Serialize)]
        struct WriteFileRequest {
            path: String,
            content: String, // Base64 encoded
        }

        let request = WriteFileRequest {
            path: path.to_string(),
            content: base64::Engine::encode(&base64::engine::general_purpose::STANDARD, content),
        };

        let response = self
            .client
            .http()
            .post(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(())
    }

    /// Read a file from the sandbox
    pub async fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        if self.is_closed() {
            return Err(Error::SandboxClosed);
        }

        let url = format!(
            "{}/sandboxes/{}/files?path={}",
            self.client.base_url(),
            self.id,
            urlencoding::encode(path)
        );

        let response = self
            .client
            .http()
            .get(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        #[derive(Deserialize)]
        struct ReadFileResponse {
            content: String, // Base64 encoded
        }

        let file_response: ReadFileResponse = response.json().await?;
        let content = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &file_response.content,
        )
        .map_err(|e| Error::Filesystem(format!("Failed to decode file content: {}", e)))?;

        Ok(content)
    }

    /// List files in a directory
    pub async fn list_files(&self, path: &str) -> Result<Vec<FileInfo>> {
        if self.is_closed() {
            return Err(Error::SandboxClosed);
        }

        let url = format!(
            "{}/sandboxes/{}/files/list?path={}",
            self.client.base_url(),
            self.id,
            urlencoding::encode(path)
        );

        let response = self
            .client
            .http()
            .get(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        let files: Vec<FileInfo> = response.json().await?;
        Ok(files)
    }

    /// Close the sandbox
    pub async fn close(&self) -> Result<()> {
        if self.closed.swap(true, Ordering::SeqCst) {
            return Ok(()); // Already closed
        }

        let url = format!(
            "{}/sandboxes/{}",
            self.client.base_url(),
            self.id
        );

        let response = self
            .client
            .http()
            .delete(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() && status.as_u16() != 404 {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(())
    }

    /// Keep the sandbox alive (extend timeout)
    pub async fn keep_alive(&self) -> Result<()> {
        if self.is_closed() {
            return Err(Error::SandboxClosed);
        }

        let url = format!(
            "{}/sandboxes/{}/keep-alive",
            self.client.base_url(),
            self.id
        );

        let response = self
            .client
            .http()
            .post(&url)
            .header("X-E2B-API-Key", self.client.api_key())
            .send()
            .await?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Api {
                status: status.as_u16(),
                message: body,
            });
        }

        Ok(())
    }
}

impl Drop for Sandbox {
    fn drop(&mut self) {
        // Mark as closed, but don't block on actual cleanup
        self.closed.store(true, Ordering::SeqCst);
    }
}
