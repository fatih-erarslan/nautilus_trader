//! Audit Logging Module
//!
//! Provides comprehensive audit trail capabilities:
//! - Structured logging for all operations
//! - Sensitive data masking
//! - User action tracking
//! - Security event logging
//! - Compliance reporting

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, RwLock};
use chrono::{DateTime, Utc};
use std::collections::VecDeque;
use uuid::Uuid;

/// Audit event severity levels
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[napi]
pub enum AuditLevel {
    /// Informational event
    Info,
    /// Warning event
    Warning,
    /// Security-related event
    Security,
    /// Error event
    Error,
    /// Critical security event
    Critical,
}

/// Audit event categories
#[derive(Debug, Serialize, Deserialize)]
#[napi]
pub enum AuditCategory {
    /// Authentication events
    Authentication,
    /// Authorization/permission checks
    Authorization,
    /// Trading operations
    Trading,
    /// Portfolio changes
    Portfolio,
    /// Configuration changes
    Configuration,
    /// Data access
    DataAccess,
    /// System events
    System,
    /// Security events
    Security,
}

/// Audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[napi(object)]
pub struct AuditEvent {
    /// Unique event ID
    pub event_id: String,
    /// Timestamp of event
    pub timestamp: String,
    /// Event level
    pub level: AuditLevel,
    /// Event category
    pub category: AuditCategory,
    /// User ID (if applicable)
    pub user_id: Option<String>,
    /// Username (if applicable)
    pub username: Option<String>,
    /// IP address
    pub ip_address: Option<String>,
    /// Action performed
    pub action: String,
    /// Resource affected
    pub resource: Option<String>,
    /// Event outcome (success/failure)
    pub outcome: String,
    /// Additional details (JSON string)
    pub details: Option<String>,
    /// Error message (if outcome is failure)
    pub error_message: Option<String>,
}

impl AuditEvent {
    /// Create a new audit event
    pub fn new(
        level: AuditLevel,
        category: AuditCategory,
        action: String,
        outcome: String,
    ) -> Self {
        Self {
            event_id: Uuid::new_v4().to_string(),
            timestamp: Utc::now().to_rfc3339(),
            level,
            category,
            user_id: None,
            username: None,
            ip_address: None,
            action,
            resource: None,
            outcome,
            details: None,
            error_message: None,
        }
    }

    /// Add user information
    pub fn with_user(mut self, user_id: String, username: String) -> Self {
        self.user_id = Some(user_id);
        self.username = Some(username);
        self
    }

    /// Add IP address
    pub fn with_ip(mut self, ip_address: String) -> Self {
        self.ip_address = Some(ip_address);
        self
    }

    /// Add resource information
    pub fn with_resource(mut self, resource: String) -> Self {
        self.resource = Some(resource);
        self
    }

    /// Add details (automatically masks sensitive data)
    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(mask_sensitive_data(details).to_string());
        self
    }

    /// Add error message
    pub fn with_error(mut self, error: String) -> Self {
        self.error_message = Some(error);
        self
    }

    /// Format event for logging
    pub fn format_log(&self) -> String {
        format!(
            "[{}] {} | {} | {} | User: {} | IP: {} | Action: {} | Resource: {} | Outcome: {}{}{}",
            self.timestamp,
            match self.level {
                AuditLevel::Info => "INFO",
                AuditLevel::Warning => "WARN",
                AuditLevel::Security => "SECURITY",
                AuditLevel::Error => "ERROR",
                AuditLevel::Critical => "CRITICAL",
            },
            match self.category {
                AuditCategory::Authentication => "AUTH",
                AuditCategory::Authorization => "AUTHZ",
                AuditCategory::Trading => "TRADE",
                AuditCategory::Portfolio => "PORTFOLIO",
                AuditCategory::Configuration => "CONFIG",
                AuditCategory::DataAccess => "DATA",
                AuditCategory::System => "SYSTEM",
                AuditCategory::Security => "SECURITY",
            },
            self.event_id,
            self.username.as_deref().unwrap_or("anonymous"),
            self.ip_address.as_deref().unwrap_or("unknown"),
            self.action,
            self.resource.as_deref().unwrap_or("N/A"),
            self.outcome,
            self.details.as_ref().map(|d| format!(" | Details: {}", d)).unwrap_or_default(),
            self.error_message.as_ref().map(|e| format!(" | Error: {}", e)).unwrap_or_default(),
        )
    }
}

/// Audit logger configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Maximum number of events to keep in memory
    pub max_events_in_memory: usize,
    /// Whether to log to console
    pub log_to_console: bool,
    /// Whether to persist logs to file
    pub log_to_file: bool,
    /// Log file path
    pub log_file_path: Option<String>,
    /// Minimum level to log
    pub min_level: AuditLevel,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            max_events_in_memory: 10000,
            log_to_console: true,
            log_to_file: true,
            log_file_path: Some("./logs/audit.log".to_string()),
            min_level: AuditLevel::Info,
        }
    }
}

/// Audit logger
pub struct AuditLogger {
    events: Arc<RwLock<VecDeque<AuditEvent>>>,
    config: AuditConfig,
}

impl AuditLogger {
    pub fn new(config: AuditConfig) -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_events_in_memory))),
            config,
        }
    }

    /// Log an audit event
    pub fn log(&self, event: AuditEvent) {
        // Check if event meets minimum level
        let event_level_value = match event.level {
            AuditLevel::Info => 0,
            AuditLevel::Warning => 1,
            AuditLevel::Security => 2,
            AuditLevel::Error => 3,
            AuditLevel::Critical => 4,
        };

        let min_level_value = match self.config.min_level {
            AuditLevel::Info => 0,
            AuditLevel::Warning => 1,
            AuditLevel::Security => 2,
            AuditLevel::Error => 3,
            AuditLevel::Critical => 4,
        };

        if event_level_value < min_level_value {
            return;
        }

        // Log to console
        if self.config.log_to_console {
            match event.level {
                AuditLevel::Info => tracing::info!("{}", event.format_log()),
                AuditLevel::Warning => tracing::warn!("{}", event.format_log()),
                AuditLevel::Security => tracing::warn!("🔒 {}", event.format_log()),
                AuditLevel::Error => tracing::error!("{}", event.format_log()),
                AuditLevel::Critical => tracing::error!("🚨 {}", event.format_log()),
            }
        }

        // Store in memory
        if let Ok(mut events) = self.events.write() {
            if events.len() >= self.config.max_events_in_memory {
                events.pop_front();
            }
            events.push_back(event.clone());
        }

        // TODO: Persist to file if configured
        if self.config.log_to_file {
            // In production, implement file logging with rotation
        }
    }

    /// Get recent audit events
    pub fn get_recent_events(&self, limit: usize) -> Vec<AuditEvent> {
        if let Ok(events) = self.events.read() {
            events.iter()
                .rev()
                .take(limit)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Query audit events by criteria
    pub fn query_events(
        &self,
        user_id: Option<String>,
        category: Option<AuditCategory>,
        level: Option<AuditLevel>,
        limit: usize,
    ) -> Vec<AuditEvent> {
        if let Ok(events) = self.events.read() {
            events.iter()
                .rev()
                .filter(|e| {
                    let user_match = user_id.as_ref()
                        .map(|uid| e.user_id.as_ref() == Some(uid))
                        .unwrap_or(true);

                    let category_match = category.as_ref()
                        .map(|cat| {
                            // Compare category discriminants
                            std::mem::discriminant(&e.category) == std::mem::discriminant(cat)
                        })
                        .unwrap_or(true);

                    let level_match = level
                        .map(|lvl| e.level == lvl)
                        .unwrap_or(true);

                    user_match && category_match && level_match
                })
                .take(limit)
                .cloned()
                .collect()
        } else {
            Vec::new()
        }
    }

    /// Get audit statistics
    pub fn get_statistics(&self) -> AuditStatistics {
        if let Ok(events) = self.events.read() {
            let total = events.len();
            let mut by_level = std::collections::HashMap::new();
            let mut by_category = std::collections::HashMap::new();
            let mut failed_operations = 0;

            for event in events.iter() {
                *by_level.entry(format!("{:?}", event.level)).or_insert(0) += 1;
                *by_category.entry(format!("{:?}", event.category)).or_insert(0) += 1;

                if event.outcome == "failure" || event.outcome == "error" {
                    failed_operations += 1;
                }
            }

            AuditStatistics {
                total_events: total,
                failed_operations,
                success_rate: if total > 0 {
                    ((total - failed_operations) as f64 / total as f64) * 100.0
                } else {
                    100.0
                },
                events_by_level: by_level,
                events_by_category: by_category,
            }
        } else {
            AuditStatistics::default()
        }
    }

    /// Clear audit log (admin operation)
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.write() {
            events.clear();
        }
    }
}

/// Audit statistics
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct AuditStatistics {
    pub total_events: usize,
    pub failed_operations: usize,
    pub success_rate: f64,
    pub events_by_level: std::collections::HashMap<String, usize>,
    pub events_by_category: std::collections::HashMap<String, usize>,
}

/// Mask sensitive data in JSON values
fn mask_sensitive_data(mut value: serde_json::Value) -> serde_json::Value {
    const SENSITIVE_FIELDS: &[&str] = &[
        "password", "api_key", "secret", "token", "private_key",
        "credit_card", "ssn", "bank_account", "pin", "cvv",
    ];

    if let serde_json::Value::Object(ref mut map) = value {
        for (key, val) in map.iter_mut() {
            let key_lower = key.to_lowercase();
            if SENSITIVE_FIELDS.iter().any(|&field| key_lower.contains(field)) {
                *val = serde_json::Value::String("***MASKED***".to_string());
            } else if val.is_object() || val.is_array() {
                *val = mask_sensitive_data(val.clone());
            }
        }
    } else if let serde_json::Value::Array(ref mut arr) = value {
        for item in arr.iter_mut() {
            *item = mask_sensitive_data(item.clone());
        }
    }

    value
}

/// Global audit logger instance
static AUDIT_LOGGER: once_cell::sync::Lazy<Arc<RwLock<Option<AuditLogger>>>> =
    once_cell::sync::Lazy::new(|| Arc::new(RwLock::new(None)));

/// Initialize audit logger
#[napi]
pub fn init_audit_logger(
    max_events: Option<u32>,
    log_to_console: Option<bool>,
    log_to_file: Option<bool>,
) -> Result<String> {
    let config = AuditConfig {
        max_events_in_memory: max_events.unwrap_or(10000) as usize,
        log_to_console: log_to_console.unwrap_or(true),
        log_to_file: log_to_file.unwrap_or(true),
        log_file_path: Some("./logs/audit.log".to_string()),
        min_level: AuditLevel::Info,
    };

    let logger = AuditLogger::new(config);

    let mut audit = AUDIT_LOGGER.write()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    *audit = Some(logger);

    Ok("Audit logger initialized".to_string())
}

/// Log an audit event
#[napi]
pub fn log_audit_event(
    level: String,
    category: String,
    action: String,
    outcome: String,
    user_id: Option<String>,
    username: Option<String>,
    ip_address: Option<String>,
    resource: Option<String>,
    details: Option<String>,
) -> Result<String> {
    let audit = AUDIT_LOGGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let logger = audit.as_ref()
        .ok_or_else(|| Error::from_reason("Audit logger not initialized"))?;

    let audit_level = match level.as_str() {
        "info" => AuditLevel::Info,
        "warning" => AuditLevel::Warning,
        "security" => AuditLevel::Security,
        "error" => AuditLevel::Error,
        "critical" => AuditLevel::Critical,
        _ => AuditLevel::Info,
    };

    let audit_category = match category.as_str() {
        "authentication" => AuditCategory::Authentication,
        "authorization" => AuditCategory::Authorization,
        "trading" => AuditCategory::Trading,
        "portfolio" => AuditCategory::Portfolio,
        "configuration" => AuditCategory::Configuration,
        "data_access" => AuditCategory::DataAccess,
        "system" => AuditCategory::System,
        "security" => AuditCategory::Security,
        _ => AuditCategory::System,
    };

    let mut event = AuditEvent::new(audit_level, audit_category, action, outcome);

    if let (Some(uid), Some(uname)) = (user_id, username) {
        event = event.with_user(uid, uname);
    }

    if let Some(ip) = ip_address {
        event = event.with_ip(ip);
    }

    if let Some(res) = resource {
        event = event.with_resource(res);
    }

    if let Some(det) = details {
        if let Ok(json) = serde_json::from_str(&det) {
            event = event.with_details(json);
        }
    }

    logger.log(event.clone());

    Ok(event.event_id)
}

/// Get recent audit events
#[napi]
pub fn get_audit_events(limit: Option<u32>) -> Result<Vec<AuditEvent>> {
    let audit = AUDIT_LOGGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let logger = audit.as_ref()
        .ok_or_else(|| Error::from_reason("Audit logger not initialized"))?;

    Ok(logger.get_recent_events(limit.unwrap_or(100) as usize))
}

/// Get audit statistics
#[napi]
pub fn get_audit_statistics() -> Result<String> {
    let audit = AUDIT_LOGGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let logger = audit.as_ref()
        .ok_or_else(|| Error::from_reason("Audit logger not initialized"))?;

    let stats = logger.get_statistics();

    serde_json::to_string_pretty(&stats)
        .map_err(|e| Error::from_reason(format!("JSON error: {}", e)))
}

/// Clear audit log (admin operation)
#[napi]
pub fn clear_audit_log() -> Result<String> {
    let audit = AUDIT_LOGGER.read()
        .map_err(|e| Error::from_reason(format!("Lock error: {}", e)))?;

    let logger = audit.as_ref()
        .ok_or_else(|| Error::from_reason("Audit logger not initialized"))?;

    logger.clear();

    Ok("Audit log cleared successfully".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sensitive_data_masking() {
        let data = serde_json::json!({
            "username": "testuser",
            "password": "secret123",
            "api_key": "key123",
            "balance": 1000.0,
        });

        let masked = mask_sensitive_data(data);
        assert_eq!(masked["password"], "***MASKED***");
        assert_eq!(masked["api_key"], "***MASKED***");
        assert_eq!(masked["username"], "testuser");
        assert_eq!(masked["balance"], 1000.0);
    }

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(
            AuditLevel::Info,
            AuditCategory::Trading,
            "execute_trade".to_string(),
            "success".to_string(),
        )
        .with_user("user123".to_string(), "testuser".to_string())
        .with_ip("192.168.1.1".to_string());

        assert_eq!(event.action, "execute_trade");
        assert_eq!(event.username, Some("testuser".to_string()));
        assert_eq!(event.ip_address, Some("192.168.1.1".to_string()));
    }
}
