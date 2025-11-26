//! Test Fixtures and Mock Services for Agent Deployment Tests
//!
//! Provides comprehensive mock implementations for:
//! - E2B API client
//! - OpenRouter API client
//! - Agent deployment system
//! - Database operations
//! - Test data generators

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokio::sync::Mutex;

// ============================================================================
// Common Types
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum AgentType {
    Coordinator,
    Researcher,
    Analyst,
    Coder,
    Tester,
    Optimizer,
}

impl Default for AgentType {
    fn default() -> Self {
        Self::Researcher
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentStatus {
    Pending,
    Running,
    Stopped,
    Failed,
    Terminated,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SwarmTopology {
    Mesh,
    Hierarchical,
    Ring,
    Star,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SwarmStatus {
    Active,
    Degraded,
    Inactive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CoordinationStrategy {
    Sequential,
    Parallel,
    Adaptive,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum NetworkMode {
    Isolated,
    Restricted,
    Full,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum LogLevel {
    Debug,
    Info,
    Warning,
    Error,
}

// ============================================================================
// E2B Mock Client
// ============================================================================

#[derive(Clone)]
pub struct MockE2BClient {
    api_key: String,
    sandboxes: Arc<RwLock<HashMap<String, SandboxState>>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxConfig {
    pub template: String,
    pub timeout: u64,
    pub env_vars: Vec<(String, String)>,
    pub metadata: JsonValue,
    #[serde(default)]
    pub max_memory_mb: Option<u64>,
    #[serde(default)]
    pub max_cpu_percent: Option<f64>,
    #[serde(default)]
    pub max_disk_mb: Option<u64>,
    #[serde(default)]
    pub network_mode: NetworkMode,
}

impl Default for SandboxConfig {
    fn default() -> Self {
        Self {
            template: "nodejs".to_string(),
            timeout: 3600,
            env_vars: Vec::new(),
            metadata: serde_json::json!({}),
            max_memory_mb: None,
            max_cpu_percent: None,
            max_disk_mb: None,
            network_mode: NetworkMode::Restricted,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxInfo {
    pub sandbox_id: String,
    pub status: String,
    pub template: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone)]
struct SandboxState {
    info: SandboxInfo,
    config: SandboxConfig,
    files: HashMap<String, Vec<u8>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandOutput {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: i32,
}

impl MockE2BClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            sandboxes: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn is_initialized(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub fn rotate_api_key(&mut self, new_key: impl Into<String>) {
        self.api_key = new_key.into();
    }

    pub async fn create_sandbox(&self, config: SandboxConfig) -> Result<SandboxInfo> {
        if self.api_key.is_empty() || self.api_key == "invalid" {
            return Err(anyhow!("Invalid API key - unauthorized"));
        }

        let sandbox_id = format!("sb_{}", uuid::Uuid::new_v4());

        let info = SandboxInfo {
            sandbox_id: sandbox_id.clone(),
            status: "running".to_string(),
            template: config.template.clone(),
            created_at: chrono::Utc::now(),
        };

        let state = SandboxState {
            info: info.clone(),
            config,
            files: HashMap::new(),
        };

        self.sandboxes.write().unwrap().insert(sandbox_id, state);

        Ok(info)
    }

    pub async fn get_sandbox_status(&self, sandbox_id: &str) -> Result<String> {
        let sandboxes = self.sandboxes.read().unwrap();
        sandboxes.get(sandbox_id)
            .map(|s| s.info.status.clone())
            .ok_or_else(|| anyhow!("Sandbox not found"))
    }

    pub async fn execute_command(
        &self,
        sandbox_id: &str,
        command: &str,
        timeout_ms: Option<u64>
    ) -> Result<CommandOutput> {
        let sandboxes = self.sandboxes.read().unwrap();
        let _sandbox = sandboxes.get(sandbox_id)
            .ok_or_else(|| anyhow!("Sandbox not found"))?;

        // Simulate timeout
        if let Some(timeout) = timeout_ms {
            if timeout < 100 && command.contains("sleep") {
                return Err(anyhow!("Command execution timeout"));
            }
        }

        // Simulate command execution
        let output = if command.contains("echo") {
            let text = command.strip_prefix("echo").unwrap_or("").trim().trim_matches('\'').trim_matches('"');
            CommandOutput {
                stdout: format!("{}\n", text),
                stderr: String::new(),
                exit_code: 0,
            }
        } else if command.contains("printenv") {
            let var = command.strip_prefix("printenv").unwrap_or("").trim();
            let value = _sandbox.config.env_vars.iter()
                .find(|(k, _)| k == var)
                .map(|(_, v)| v.clone())
                .unwrap_or_default();

            CommandOutput {
                stdout: format!("{}\n", value),
                stderr: String::new(),
                exit_code: 0,
            }
        } else if command.starts_with("node") || command == "console.log('Hello from agent!');" {
            CommandOutput {
                stdout: "Hello from agent!\n".to_string(),
                stderr: String::new(),
                exit_code: 0,
            }
        } else if command.contains("nonexistent-command") {
            CommandOutput {
                stdout: String::new(),
                stderr: "command not found\n".to_string(),
                exit_code: 127,
            }
        } else {
            CommandOutput {
                stdout: "command executed\n".to_string(),
                stderr: String::new(),
                exit_code: 0,
            }
        };

        Ok(output)
    }

    pub async fn upload_file(&self, sandbox_id: &str, path: &str, content: &[u8]) -> Result<()> {
        let mut sandboxes = self.sandboxes.write().unwrap();
        let sandbox = sandboxes.get_mut(sandbox_id)
            .ok_or_else(|| anyhow!("Sandbox not found"))?;

        sandbox.files.insert(path.to_string(), content.to_vec());
        Ok(())
    }

    pub async fn read_file(&self, sandbox_id: &str, path: &str) -> Result<String> {
        // Block path traversal attempts
        if path.contains("..") || path.starts_with("/etc/") || path.starts_with("/proc/") {
            return Err(anyhow!("Access denied: path traversal not allowed"));
        }

        let sandboxes = self.sandboxes.read().unwrap();
        let sandbox = sandboxes.get(sandbox_id)
            .ok_or_else(|| anyhow!("Sandbox not found"))?;

        sandbox.files.get(path)
            .map(|content| String::from_utf8_lossy(content).to_string())
            .ok_or_else(|| anyhow!("File not found"))
    }

    pub async fn list_files(&self, sandbox_id: &str, dir_path: &str) -> Result<Vec<String>> {
        let sandboxes = self.sandboxes.read().unwrap();
        let sandbox = sandboxes.get(sandbox_id)
            .ok_or_else(|| anyhow!("Sandbox not found"))?;

        let files: Vec<String> = sandbox.files.keys()
            .filter(|path| path.starts_with(dir_path))
            .cloned()
            .collect();

        Ok(files)
    }

    pub async fn stop_sandbox(&self, sandbox_id: &str) -> Result<()> {
        let mut sandboxes = self.sandboxes.write().unwrap();
        if let Some(sandbox) = sandboxes.get_mut(sandbox_id) {
            sandbox.info.status = "stopped".to_string();
            Ok(())
        } else {
            Err(anyhow!("Sandbox not found"))
        }
    }

    pub async fn get_sandbox_health(&self, sandbox_id: &str) -> Result<SandboxHealth> {
        let sandboxes = self.sandboxes.read().unwrap();
        sandboxes.get(sandbox_id)
            .ok_or_else(|| anyhow!("Sandbox not found"))?;

        Ok(SandboxHealth {
            cpu_usage: 25.5,
            memory_mb: 256,
            disk_mb: 50,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxHealth {
    pub cpu_usage: f64,
    pub memory_mb: u64,
    pub disk_mb: u64,
}

// ============================================================================
// OpenRouter Mock Client
// ============================================================================

#[derive(Clone)]
pub struct MockOpenRouterClient {
    api_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionConfig {
    pub model: String,
    pub max_tokens: u32,
    pub temperature: f64,
    #[serde(default)]
    pub stream: bool,
}

impl Default for CompletionConfig {
    fn default() -> Self {
        Self {
            model: "anthropic/claude-3-sonnet".to_string(),
            max_tokens: 4000,
            temperature: 0.7,
            stream: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub tokens_used: u32,
    pub cost_usd: f64,
    pub model: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub content: String,
    pub tokens_used: Option<u32>,
    pub finish_reason: Option<String>,
}

pub struct StreamingResponse {
    chunks: Vec<StreamChunk>,
    index: usize,
}

impl StreamingResponse {
    pub async fn next(&mut self) -> Option<StreamChunk> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            Some(chunk)
        } else {
            None
        }
    }
}

impl MockOpenRouterClient {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
        }
    }

    pub fn is_initialized(&self) -> bool {
        !self.api_key.is_empty()
    }

    pub async fn create_completion(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<CompletionConfig>,
    ) -> Result<CompletionResponse> {
        let config = config.unwrap_or_default();

        // Simulate token usage and cost
        let prompt_tokens: u32 = messages.iter()
            .map(|m| m.content.split_whitespace().count() as u32 * 2)
            .sum();

        let completion_tokens = config.max_tokens.min(500);
        let total_tokens = prompt_tokens + completion_tokens;

        let cost_per_token = match config.model.as_str() {
            "anthropic/claude-3-opus" => 0.000015,
            "anthropic/claude-3-sonnet" => 0.000003,
            "anthropic/claude-3-haiku" => 0.00000025,
            _ => 0.000001,
        };

        let content = format!(
            "This is a mock response to: {}. Generated with {} model.",
            messages.last().map(|m| &m.content).unwrap_or(&"".to_string()),
            config.model
        );

        Ok(CompletionResponse {
            content,
            tokens_used: total_tokens,
            cost_usd: total_tokens as f64 * cost_per_token,
            model: config.model,
        })
    }

    pub async fn create_streaming_completion(
        &self,
        messages: Vec<ChatMessage>,
        config: Option<CompletionConfig>,
    ) -> Result<StreamingResponse> {
        let config = config.unwrap_or_default();

        let chunks = vec![
            StreamChunk {
                content: "This is ".to_string(),
                tokens_used: None,
                finish_reason: None,
            },
            StreamChunk {
                content: "a streaming ".to_string(),
                tokens_used: None,
                finish_reason: None,
            },
            StreamChunk {
                content: "response.".to_string(),
                tokens_used: Some(150),
                finish_reason: Some("stop".to_string()),
            },
        ];

        Ok(StreamingResponse {
            chunks,
            index: 0,
        })
    }
}

// ============================================================================
// Agent Deployment Mocks
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub agent_type: AgentType,
    pub name: String,
    #[serde(default)]
    pub sandbox_template: String,
    #[serde(default)]
    pub capabilities: Vec<String>,
    #[serde(default)]
    pub llm_model: String,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub env_vars: Vec<(String, String)>,
    #[serde(default)]
    pub max_memory_mb: Option<u64>,
    #[serde(default)]
    pub max_cpu_percent: Option<f64>,
    #[serde(default)]
    pub auto_restart: bool,
    #[serde(default)]
    pub max_restart_attempts: u32,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            agent_type: AgentType::Researcher,
            name: "default-agent".to_string(),
            sandbox_template: "nodejs".to_string(),
            capabilities: vec!["general".to_string()],
            llm_model: "anthropic/claude-3-sonnet".to_string(),
            max_tokens: 4000,
            env_vars: Vec::new(),
            max_memory_mb: None,
            max_cpu_percent: None,
            auto_restart: false,
            max_restart_attempts: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDeployment {
    pub agent_id: String,
    pub sandbox_id: String,
    pub agent_type: AgentType,
    pub name: String,
    pub status: DeploymentStatus,
    pub sandbox_template: String,
    pub llm_model: String,
    pub capabilities: Vec<String>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub updated_at: chrono::DateTime<chrono::Utc>,
    pub terminated_at: Option<chrono::DateTime<chrono::Utc>>,
    pub metadata: JsonValue,
    pub env_vars: Vec<(String, String)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub task_id: String,
    pub task_type: String,
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: u32,
    #[serde(default)]
    pub execute_code: bool,
    #[serde(default)]
    pub stream_response: bool,
    #[serde(default)]
    pub language: String,
    #[serde(default)]
    pub timeout_ms: Option<u64>,
}

impl Default for AgentTask {
    fn default() -> Self {
        Self {
            task_id: "default-task".to_string(),
            task_type: "general".to_string(),
            prompt: String::new(),
            max_tokens: 4000,
            execute_code: false,
            stream_response: false,
            language: "javascript".to_string(),
            timeout_ms: None,
        }
    }
}

impl Clone for AgentTask {
    fn clone(&self) -> Self {
        Self {
            task_id: self.task_id.clone(),
            task_type: self.task_type.clone(),
            prompt: self.prompt.clone(),
            max_tokens: self.max_tokens,
            execute_code: self.execute_code,
            stream_response: self.stream_response,
            language: self.language.clone(),
            timeout_ms: self.timeout_ms,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskExecution {
    pub task_id: String,
    pub success: bool,
    pub output: String,
    pub llm_response: Option<String>,
    pub code_output: Option<CommandOutput>,
    pub tokens_used: u32,
    pub cost_usd: f64,
    pub completed_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentStatus {
    pub agent_id: String,
    pub status: DeploymentStatus,
    pub uptime_seconds: u64,
    pub tasks_completed: u64,
    pub total_tokens_used: u64,
    pub restart_count: u32,
    pub metadata: JsonValue,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentHealth {
    pub status: HealthStatus,
    pub sandbox_responsive: bool,
    pub llm_accessible: bool,
    pub cpu_usage: f64,
    pub memory_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub agent_id: String,
    pub cpu_usage: f64,
    pub memory_mb: u64,
    pub requests_processed: u64,
    pub tokens_used: u64,
    pub cost_usd: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    pub agent_id: String,
    pub level: LogLevel,
    pub message: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

// ============================================================================
// Mock Agent Deployer
// ============================================================================

#[derive(Clone)]
pub struct MockAgentDeployer {
    deployments: Arc<RwLock<HashMap<String, AgentDeployment>>>,
}

impl MockAgentDeployer {
    pub const MAX_CONCURRENT_AGENTS: usize = 10;

    pub fn new() -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn deploy_agent(&self, config: AgentConfig) -> Result<AgentDeployment> {
        // Validation
        if config.name.is_empty() {
            return Err(anyhow!("Agent name cannot be empty"));
        }

        // Check concurrent limit
        let current_count = self.deployments.read().unwrap().len();
        if current_count >= Self::MAX_CONCURRENT_AGENTS {
            return Err(anyhow!("Maximum concurrent agent limit reached"));
        }

        let agent_id = format!("agent_{}", uuid::Uuid::new_v4());
        let sandbox_id = format!("sb_{}", uuid::Uuid::new_v4());

        let deployment = AgentDeployment {
            agent_id: agent_id.clone(),
            sandbox_id,
            agent_type: config.agent_type,
            name: config.name,
            status: DeploymentStatus::Running,
            sandbox_template: config.sandbox_template,
            llm_model: config.llm_model,
            capabilities: config.capabilities,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
            terminated_at: None,
            metadata: serde_json::json!({
                "env_configured": !config.env_vars.is_empty()
            }),
            env_vars: config.env_vars,
        };

        self.deployments.write().unwrap().insert(agent_id, deployment.clone());

        Ok(deployment)
    }

    pub async fn get_agent_status(&self, agent_id: &str) -> Result<AgentStatus> {
        let deployments = self.deployments.read().unwrap();
        let deployment = deployments.get(agent_id)
            .ok_or_else(|| anyhow!("Agent not found"))?;

        let uptime = (chrono::Utc::now() - deployment.created_at).num_seconds().max(0) as u64;

        Ok(AgentStatus {
            agent_id: agent_id.to_string(),
            status: deployment.status.clone(),
            uptime_seconds: uptime,
            tasks_completed: 0,
            total_tokens_used: 0,
            restart_count: 0,
            metadata: deployment.metadata.clone(),
        })
    }

    pub async fn terminate_agent(&self, agent_id: &str) -> Result<()> {
        let mut deployments = self.deployments.write().unwrap();
        if let Some(deployment) = deployments.get_mut(agent_id) {
            deployment.status = DeploymentStatus::Terminated;
            deployment.terminated_at = Some(chrono::Utc::now());
            Ok(())
        } else {
            Err(anyhow!("Agent not found"))
        }
    }
}

// ============================================================================
// Mock Agent Database
// ============================================================================

#[derive(Clone)]
pub struct MockAgentDatabase {
    deployments: Arc<RwLock<HashMap<String, AgentDeployment>>>,
    metrics: Arc<RwLock<HashMap<String, Vec<AgentMetrics>>>>,
}

impl MockAgentDatabase {
    pub fn new() -> Self {
        Self {
            deployments: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn store_deployment(&self, deployment: &AgentDeployment) -> Result<()> {
        self.deployments.write().unwrap().insert(
            deployment.agent_id.clone(),
            deployment.clone()
        );
        Ok(())
    }

    pub async fn get_deployment(&self, agent_id: &str) -> Result<AgentDeployment> {
        self.deployments.read().unwrap().get(agent_id)
            .cloned()
            .ok_or_else(|| anyhow!("Deployment not found"))
    }

    pub async fn list_active_deployments(&self) -> Result<Vec<AgentDeployment>> {
        Ok(self.deployments.read().unwrap().values()
            .filter(|d| matches!(d.status, DeploymentStatus::Running))
            .cloned()
            .collect())
    }

    pub async fn update_deployment_status(
        &self,
        agent_id: &str,
        status: DeploymentStatus,
        _metadata: Option<JsonValue>
    ) -> Result<()> {
        let mut deployments = self.deployments.write().unwrap();
        if let Some(deployment) = deployments.get_mut(agent_id) {
            deployment.status = status;
            deployment.updated_at = chrono::Utc::now();
            Ok(())
        } else {
            Err(anyhow!("Deployment not found"))
        }
    }

    pub async fn store_metrics(&self, metrics: &AgentMetrics) -> Result<()> {
        self.metrics.write().unwrap()
            .entry(metrics.agent_id.clone())
            .or_insert_with(Vec::new)
            .push(metrics.clone());
        Ok(())
    }

    pub async fn get_agent_metrics(&self, agent_id: &str, limit: usize) -> Result<Vec<AgentMetrics>> {
        Ok(self.metrics.read().unwrap()
            .get(agent_id)
            .map(|m| m.iter().rev().take(limit).cloned().collect())
            .unwrap_or_default())
    }
}

// Re-export for convenience
pub use crate::fixtures::{
    MockE2BClient, MockOpenRouterClient, MockAgentDeployer, MockAgentDatabase,
};
