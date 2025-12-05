//! Audit logging for MCP authentication events

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Audit log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditLevel {
    /// Informational events
    Info,
    /// Security-relevant events
    Security,
    /// Warning conditions
    Warning,
    /// Authentication failures
    AuthFailure,
    /// Potential attack detected
    Attack,
    /// Critical security event
    Critical,
}

impl AuditLevel {
    /// Get severity score (higher = more severe)
    pub fn severity(&self) -> u8 {
        match self {
            AuditLevel::Info => 0,
            AuditLevel::Security => 1,
            AuditLevel::Warning => 2,
            AuditLevel::AuthFailure => 3,
            AuditLevel::Attack => 4,
            AuditLevel::Critical => 5,
        }
    }
}

/// Single audit log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Entry timestamp
    pub timestamp: DateTime<Utc>,

    /// Audit level
    pub level: AuditLevel,

    /// Event category
    pub category: String,

    /// Event description
    pub message: String,

    /// Client ID (if applicable)
    pub client_id: Option<String>,

    /// Request ID (if applicable)
    pub request_id: Option<String>,

    /// Tool method (if applicable)
    pub tool: Option<String>,

    /// Additional context data
    pub context: Option<serde_json::Value>,

    /// Source IP/identifier (if available)
    pub source: Option<String>,
}

impl AuditEntry {
    /// Create new audit entry
    pub fn new(level: AuditLevel, category: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            timestamp: Utc::now(),
            level,
            category: category.into(),
            message: message.into(),
            client_id: None,
            request_id: None,
            tool: None,
            context: None,
            source: None,
        }
    }

    /// Add client ID
    pub fn with_client(mut self, client_id: impl Into<String>) -> Self {
        self.client_id = Some(client_id.into());
        self
    }

    /// Add request ID
    pub fn with_request(mut self, request_id: impl Into<String>) -> Self {
        self.request_id = Some(request_id.into());
        self
    }

    /// Add tool name
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = Some(tool.into());
        self
    }

    /// Add context data
    pub fn with_context(mut self, context: serde_json::Value) -> Self {
        self.context = Some(context);
        self
    }

    /// Add source identifier
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self
    }
}

/// Audit log storage and management
pub struct AuditLog {
    /// Log entries (bounded circular buffer)
    entries: Arc<RwLock<Vec<AuditEntry>>>,

    /// Maximum entries to keep
    max_entries: usize,

    /// Minimum level to log
    min_level: AuditLevel,

    /// Whether logging is enabled
    enabled: bool,
}

impl AuditLog {
    /// Create new audit log
    pub fn new(max_entries: usize, min_level: AuditLevel) -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::with_capacity(max_entries))),
            max_entries,
            min_level,
            enabled: true,
        }
    }

    /// Create disabled audit log (no-op)
    pub fn disabled() -> Self {
        Self {
            entries: Arc::new(RwLock::new(Vec::new())),
            max_entries: 0,
            min_level: AuditLevel::Critical,
            enabled: false,
        }
    }

    /// Log an entry
    pub async fn log(&self, entry: AuditEntry) {
        if !self.enabled || entry.level.severity() < self.min_level.severity() {
            return;
        }

        // Also emit to tracing
        match entry.level {
            AuditLevel::Info => tracing::info!(
                category = %entry.category,
                client = ?entry.client_id,
                request = ?entry.request_id,
                "{}",
                entry.message
            ),
            AuditLevel::Security | AuditLevel::Warning => tracing::warn!(
                category = %entry.category,
                client = ?entry.client_id,
                request = ?entry.request_id,
                "{}",
                entry.message
            ),
            AuditLevel::AuthFailure | AuditLevel::Attack | AuditLevel::Critical => tracing::error!(
                category = %entry.category,
                client = ?entry.client_id,
                request = ?entry.request_id,
                tool = ?entry.tool,
                "{}",
                entry.message
            ),
        }

        let mut entries = self.entries.write().await;

        // Rotate if at capacity
        if entries.len() >= self.max_entries {
            entries.remove(0);
        }

        entries.push(entry);
    }

    /// Log synchronously (non-blocking, best effort)
    pub fn log_sync(&self, entry: AuditEntry) {
        if !self.enabled || entry.level.severity() < self.min_level.severity() {
            return;
        }

        // Emit to tracing
        match entry.level {
            AuditLevel::Info => tracing::info!(
                category = %entry.category,
                "{}",
                entry.message
            ),
            AuditLevel::Security | AuditLevel::Warning => tracing::warn!(
                category = %entry.category,
                "{}",
                entry.message
            ),
            AuditLevel::AuthFailure | AuditLevel::Attack | AuditLevel::Critical => tracing::error!(
                category = %entry.category,
                "{}",
                entry.message
            ),
        }

        // For sync context, just emit to tracing - don't try to write to async storage
        // The tracing output is the primary audit mechanism anyway
    }

    /// Get recent entries
    pub async fn recent(&self, count: usize) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries.iter().rev().take(count).cloned().collect()
    }

    /// Get entries by level
    pub async fn by_level(&self, level: AuditLevel) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries.iter()
            .filter(|e| e.level == level)
            .cloned()
            .collect()
    }

    /// Get entries for a specific client
    pub async fn by_client(&self, client_id: &str) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries.iter()
            .filter(|e| e.client_id.as_deref() == Some(client_id))
            .cloned()
            .collect()
    }

    /// Get attack/security entries
    pub async fn security_events(&self) -> Vec<AuditEntry> {
        let entries = self.entries.read().await;
        entries.iter()
            .filter(|e| e.level.severity() >= AuditLevel::AuthFailure.severity())
            .cloned()
            .collect()
    }

    /// Get entry count
    pub async fn len(&self) -> usize {
        self.entries.read().await.len()
    }

    /// Check if log is empty
    pub async fn is_empty(&self) -> bool {
        self.entries.read().await.is_empty()
    }

    /// Clear all entries
    pub async fn clear(&self) {
        self.entries.write().await.clear();
    }

    /// Export entries as JSON
    pub async fn export_json(&self) -> serde_json::Value {
        let entries = self.entries.read().await;
        serde_json::to_value(&*entries).unwrap_or(serde_json::Value::Array(vec![]))
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new(10_000, AuditLevel::Info)
    }
}

// Convenience functions for common audit events
impl AuditLog {
    /// Log successful authentication
    pub async fn auth_success(&self, client_id: &str) {
        self.log(
            AuditEntry::new(AuditLevel::Info, "auth", "Authentication successful")
                .with_client(client_id)
        ).await;
    }

    /// Log failed authentication
    pub async fn auth_failure(&self, client_id: &str, reason: &str) {
        self.log(
            AuditEntry::new(AuditLevel::AuthFailure, "auth", format!("Authentication failed: {}", reason))
                .with_client(client_id)
        ).await;
    }

    /// Log injection attack detected
    pub async fn injection_detected(&self, client_id: Option<&str>, pattern: &str, request_id: &str) {
        let mut entry = AuditEntry::new(
            AuditLevel::Attack,
            "injection",
            format!("Injection pattern detected: {}", pattern)
        ).with_request(request_id);

        if let Some(cid) = client_id {
            entry = entry.with_client(cid);
        }

        self.log(entry).await;
    }

    /// Log rate limit exceeded
    pub async fn rate_limit(&self, client_id: &str, requests: u32, limit: u32) {
        self.log(
            AuditEntry::new(
                AuditLevel::Warning,
                "rate_limit",
                format!("Rate limit exceeded: {}/{} requests/min", requests, limit)
            ).with_client(client_id)
        ).await;
    }

    /// Log tool access
    pub async fn tool_access(&self, client_id: &str, tool: &str, request_id: &str) {
        self.log(
            AuditEntry::new(AuditLevel::Info, "access", format!("Tool accessed: {}", tool))
                .with_client(client_id)
                .with_request(request_id)
                .with_tool(tool)
        ).await;
    }

    /// Log unauthorized tool access attempt
    pub async fn unauthorized_tool(&self, client_id: &str, tool: &str, request_id: &str) {
        self.log(
            AuditEntry::new(
                AuditLevel::AuthFailure,
                "access",
                format!("Unauthorized tool access attempted: {}", tool)
            )
                .with_client(client_id)
                .with_request(request_id)
                .with_tool(tool)
        ).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_logging() {
        let log = AuditLog::new(100, AuditLevel::Info);

        log.log(AuditEntry::new(AuditLevel::Info, "test", "Test message")).await;

        assert_eq!(log.len().await, 1);
    }

    #[tokio::test]
    async fn test_level_filtering() {
        let log = AuditLog::new(100, AuditLevel::Warning);

        // Info should be filtered out
        log.log(AuditEntry::new(AuditLevel::Info, "test", "Info message")).await;
        assert_eq!(log.len().await, 0);

        // Warning should be logged
        log.log(AuditEntry::new(AuditLevel::Warning, "test", "Warning message")).await;
        assert_eq!(log.len().await, 1);
    }

    #[tokio::test]
    async fn test_recent_entries() {
        let log = AuditLog::new(100, AuditLevel::Info);

        for i in 0..5 {
            log.log(AuditEntry::new(AuditLevel::Info, "test", format!("Message {}", i))).await;
        }

        let recent = log.recent(3).await;
        assert_eq!(recent.len(), 3);
        assert!(recent[0].message.contains("4")); // Most recent first
    }
}
