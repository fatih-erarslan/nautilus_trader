//! Type definitions for E2B API

use serde::{Deserialize, Serialize};

/// Execution result from sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Standard output
    pub stdout: String,
    /// Standard error
    pub stderr: String,
    /// Exit code
    pub exit_code: i32,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
    /// Any artifacts produced (files, images, etc.)
    #[serde(default)]
    pub artifacts: Vec<Artifact>,
}

/// Artifact produced by code execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    /// Artifact type (file, image, chart, etc.)
    pub artifact_type: ArtifactType,
    /// Name or path
    pub name: String,
    /// Base64 encoded content (for binary artifacts)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// MIME type
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mime_type: Option<String>,
}

/// Types of artifacts
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactType {
    File,
    Image,
    Chart,
    DataFrame,
    Html,
    Markdown,
    Json,
    Other(String),
}

/// File info in sandbox filesystem
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// File path
    pub path: String,
    /// File name
    pub name: String,
    /// Is directory
    pub is_dir: bool,
    /// File size in bytes
    pub size: u64,
}

/// Process info in sandbox
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessInfo {
    /// Process ID
    pub pid: u32,
    /// Command
    pub command: String,
    /// Process state
    pub state: ProcessState,
}

/// Process state
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProcessState {
    Running,
    Sleeping,
    Stopped,
    Zombie,
    Dead,
}

/// Sandbox metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SandboxMetadata {
    /// Sandbox ID
    pub id: String,
    /// Template used
    pub template: String,
    /// Creation timestamp
    pub created_at: String,
    /// Custom metadata
    #[serde(default)]
    pub metadata: std::collections::HashMap<String, String>,
}

/// Code interpreter request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeRequest {
    /// Code to execute
    pub code: String,
    /// Language (python, javascript, etc.)
    #[serde(default = "default_language")]
    pub language: String,
    /// Timeout in milliseconds
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    /// Working directory
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cwd: Option<String>,
    /// Environment variables
    #[serde(default)]
    pub env: std::collections::HashMap<String, String>,
}

fn default_language() -> String {
    "python".to_string()
}

impl Default for CodeRequest {
    fn default() -> Self {
        Self {
            code: String::new(),
            language: default_language(),
            timeout_ms: Some(30000),
            cwd: None,
            env: std::collections::HashMap::new(),
        }
    }
}

/// Streaming execution event
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExecutionEvent {
    /// Standard output chunk
    Stdout { data: String },
    /// Standard error chunk  
    Stderr { data: String },
    /// Artifact produced
    Artifact { artifact: Artifact },
    /// Execution completed
    Done { result: ExecutionResult },
    /// Error occurred
    Error { message: String },
}
