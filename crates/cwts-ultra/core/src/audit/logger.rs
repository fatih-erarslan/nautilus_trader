use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use thiserror::Error;
use tokio::fs::OpenOptions;
use tokio::io::AsyncWriteExt;
use tokio::sync::mpsc;

/// Audit logger for compliance and security monitoring
///
/// References:
/// - SOX Section 404: Internal Controls over Financial Reporting
/// - PCI DSS Requirement 10: Log and Monitor Access
/// - ISO 27001: Information Security Management
#[derive(Debug)]
pub struct AuditLogger {
    sender: mpsc::UnboundedSender<AuditEvent>,
    _handle: tokio::task::JoinHandle<()>,
    service_name: String,
}

/// Audit event types for comprehensive logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    ConnectionEstablished,
    ConnectionClosed,
    DataReceived,
    DataValidationPassed,
    DataValidationFailed,
    CircuitBreakerTriggered,
    CacheHit,
    CacheMiss,
    SecurityViolation,
    PerformanceAlert,
    SystemError,
}

/// Comprehensive audit event structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub service_name: String,
    pub event_type: AuditEventType,
    pub user_id: Option<String>,
    pub session_id: Option<String>,
    pub source_ip: Option<String>,
    pub endpoint: Option<String>,
    pub data_hash: Option<String>,
    pub success: bool,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

impl AuditLogger {
    /// Initialize audit logger with service name
    pub fn new(service_name: &str) -> Result<Self, AuditError> {
        let (sender, receiver) = mpsc::unbounded_channel();
        let service_name = service_name.to_string();
        let service_name_clone = service_name.clone();

        // Spawn background task for async logging
        let handle = tokio::spawn(async move {
            Self::log_processor(receiver, service_name_clone).await;
        });

        Ok(Self {
            sender,
            _handle: handle,
            service_name,
        })
    }

    /// Log connection establishment
    pub async fn log_connection_established(&self) -> Result<(), AuditError> {
        let event = AuditEvent {
            event_id: Self::generate_event_id(),
            timestamp: Utc::now(),
            service_name: self.service_name.clone(),
            event_type: AuditEventType::ConnectionEstablished,
            user_id: None,
            session_id: None,
            source_ip: None,
            endpoint: Some("wss://stream.binance.com:9443/ws".to_string()),
            data_hash: None,
            success: true,
            error_message: None,
            metadata: HashMap::new(),
        };

        self.send_event(event).await
    }

    /// Log connection closure
    pub async fn log_connection_closed(&self, reason: Option<String>) -> Result<(), AuditError> {
        let mut metadata = HashMap::new();
        if let Some(reason) = reason {
            metadata.insert("close_reason".to_string(), reason);
        }

        let event = AuditEvent {
            event_id: Self::generate_event_id(),
            timestamp: Utc::now(),
            service_name: self.service_name.clone(),
            event_type: AuditEventType::ConnectionClosed,
            user_id: None,
            session_id: None,
            source_ip: None,
            endpoint: Some("wss://stream.binance.com:9443/ws".to_string()),
            data_hash: None,
            success: true,
            error_message: None,
            metadata,
        };

        self.send_event(event).await
    }

    /// Log data reception with integrity hash
    pub async fn log_data_received<T: Serialize>(&self, data: &T) -> Result<(), AuditError> {
        let data_json = serde_json::to_string(data)
            .map_err(|e| AuditError::SerializationError(e.to_string()))?;

        let data_hash = Self::calculate_data_hash(&data_json);

        let mut metadata = HashMap::new();
        metadata.insert("data_size".to_string(), data_json.len().to_string());
        metadata.insert(
            "data_type".to_string(),
            std::any::type_name::<T>().to_string(),
        );

        let event = AuditEvent {
            event_id: Self::generate_event_id(),
            timestamp: Utc::now(),
            service_name: self.service_name.clone(),
            event_type: AuditEventType::DataReceived,
            user_id: None,
            session_id: None,
            source_ip: None,
            endpoint: None,
            data_hash: Some(data_hash),
            success: true,
            error_message: None,
            metadata,
        };

        self.send_event(event).await
    }

    /// Log validation events
    pub async fn log_validation_result(
        &self,
        success: bool,
        error: Option<String>,
    ) -> Result<(), AuditError> {
        let event_type = if success {
            AuditEventType::DataValidationPassed
        } else {
            AuditEventType::DataValidationFailed
        };

        let event = AuditEvent {
            event_id: Self::generate_event_id(),
            timestamp: Utc::now(),
            service_name: self.service_name.clone(),
            event_type,
            user_id: None,
            session_id: None,
            source_ip: None,
            endpoint: None,
            data_hash: None,
            success,
            error_message: error,
            metadata: HashMap::new(),
        };

        self.send_event(event).await
    }

    /// Log circuit breaker events
    pub async fn log_circuit_breaker_triggered(&self, state: &str) -> Result<(), AuditError> {
        let mut metadata = HashMap::new();
        metadata.insert("circuit_state".to_string(), state.to_string());

        let event = AuditEvent {
            event_id: Self::generate_event_id(),
            timestamp: Utc::now(),
            service_name: self.service_name.clone(),
            event_type: AuditEventType::CircuitBreakerTriggered,
            user_id: None,
            session_id: None,
            source_ip: None,
            endpoint: None,
            data_hash: None,
            success: false,
            error_message: Some(format!("Circuit breaker state changed to: {}", state)),
            metadata,
        };

        self.send_event(event).await
    }

    /// Log security violations
    pub async fn log_security_violation(
        &self,
        violation_type: &str,
        details: &str,
    ) -> Result<(), AuditError> {
        let mut metadata = HashMap::new();
        metadata.insert("violation_type".to_string(), violation_type.to_string());
        metadata.insert("details".to_string(), details.to_string());
        metadata.insert("severity".to_string(), "HIGH".to_string());

        let event = AuditEvent {
            event_id: Self::generate_event_id(),
            timestamp: Utc::now(),
            service_name: self.service_name.clone(),
            event_type: AuditEventType::SecurityViolation,
            user_id: None,
            session_id: None,
            source_ip: None,
            endpoint: None,
            data_hash: None,
            success: false,
            error_message: Some(details.to_string()),
            metadata,
        };

        self.send_event(event).await
    }

    /// Send event to background processor
    async fn send_event(&self, event: AuditEvent) -> Result<(), AuditError> {
        self.sender
            .send(event)
            .map_err(|_| AuditError::ChannelClosed)?;
        Ok(())
    }

    /// Background log processor
    async fn log_processor(
        mut receiver: mpsc::UnboundedReceiver<AuditEvent>,
        service_name: String,
    ) {
        let log_file_path = PathBuf::from(format!("logs/audit_{}.log", service_name));

        // Ensure log directory exists
        if let Some(parent) = log_file_path.parent() {
            let _ = tokio::fs::create_dir_all(parent).await;
        }

        let mut log_file = match OpenOptions::new()
            .create(true)
            .append(true)
            .open(&log_file_path)
            .await
        {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Failed to open audit log file: {}", e);
                return;
            }
        };

        while let Some(event) = receiver.recv().await {
            let log_entry = match serde_json::to_string(&event) {
                Ok(json) => format!("{}\n", json),
                Err(e) => {
                    eprintln!("Failed to serialize audit event: {}", e);
                    continue;
                }
            };

            if let Err(e) = log_file.write_all(log_entry.as_bytes()).await {
                eprintln!("Failed to write audit log: {}", e);
            }

            if let Err(e) = log_file.flush().await {
                eprintln!("Failed to flush audit log: {}", e);
            }
        }
    }

    /// Generate unique event ID
    fn generate_event_id() -> String {
        use rand::RngCore;
        let mut bytes = [0u8; 16];
        rand::thread_rng().fill_bytes(&mut bytes);
        hex::encode(bytes)
    }

    /// Calculate SHA-256 hash of data
    fn calculate_data_hash(data: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(data.as_bytes());
        hex::encode(hasher.finalize())
    }

    /// Get audit statistics for monitoring
    pub fn get_audit_stats(&self) -> AuditStats {
        AuditStats {
            service_name: self.service_name.clone(),
            channel_capacity: None, // UnboundedSender has no capacity limit
            is_closed: self.sender.is_closed(),
        }
    }
}

/// Audit statistics for monitoring
#[derive(Debug, Clone)]
pub struct AuditStats {
    pub service_name: String,
    pub channel_capacity: Option<usize>,
    pub is_closed: bool,
}

/// Audit logger errors
#[derive(Debug, Error)]
pub enum AuditError {
    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Channel closed")]
    ChannelClosed,

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("System time error")]
    SystemTimeError,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{sleep, Duration};

    #[tokio::test]
    async fn test_audit_logger_creation() {
        let logger = AuditLogger::new("test_service");
        assert!(logger.is_ok());
    }

    #[tokio::test]
    async fn test_log_connection_events() {
        let logger = AuditLogger::new("test_service").unwrap();

        assert!(logger.log_connection_established().await.is_ok());
        assert!(logger
            .log_connection_closed(Some("test_reason".to_string()))
            .await
            .is_ok());

        // Give background task time to process
        sleep(Duration::from_millis(100)).await;
    }

    #[tokio::test]
    async fn test_log_data_events() {
        let logger = AuditLogger::new("test_service").unwrap();

        let test_data = serde_json::json!({"test": "data"});
        assert!(logger.log_data_received(&test_data).await.is_ok());

        // Give background task time to process
        sleep(Duration::from_millis(100)).await;
    }
}
