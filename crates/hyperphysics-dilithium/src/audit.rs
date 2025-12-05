//! Cryptographic Audit Logging
//!
//! Provides comprehensive audit trail for all cryptographic operations,
//! enabling compliance with security standards and forensic analysis.
//!
//! # Features
//!
//! - Immutable audit log entries
//! - Cryptographic integrity verification
//! - Structured event types for all operations
//! - Integration with external logging systems
//! - Performance metrics tracking
//!
//! # Compliance
//!
//! Supports logging requirements for:
//! - SOC 2 Type II
//! - PCI DSS
//! - HIPAA
//! - GDPR (right to audit)
//!
//! # Example
//!
//! ```
//! use hyperphysics_dilithium::audit::{AuditLogger, AuditEvent, AuditEventType};
//!
//! let logger = AuditLogger::new();
//!
//! logger.log(AuditEvent::new(
//!     AuditEventType::KeyGeneration,
//!     "Generated Dilithium-3 key pair",
//! ).with_key_id("key-123"));
//! ```

use crate::SecurityLevel;
use serde::{Deserialize, Serialize};
use sha3::{Digest, Sha3_256};
use std::collections::VecDeque;
use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};

/// Audit event type
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuditEventType {
    // Key Management
    /// Key pair generated
    KeyGeneration,
    /// Key imported
    KeyImport,
    /// Key exported (public key)
    KeyExport,
    /// Key deleted
    KeyDeletion,
    /// Key rotated
    KeyRotation,

    // Signing Operations
    /// Message signed
    SignatureCreated,
    /// Signature verified (success)
    SignatureVerified,
    /// Signature verification failed
    SignatureVerificationFailed,
    /// Batch verification completed
    BatchVerification,

    // Channel Operations
    /// Secure channel established
    ChannelEstablished,
    /// Secure channel closed
    ChannelClosed,
    /// Message sent through channel
    ChannelMessageSent,
    /// Message received through channel
    ChannelMessageReceived,

    // HSM Operations
    /// HSM operation performed
    HsmOperation,
    /// HSM health check
    HsmHealthCheck,

    // Security Events
    /// Authentication attempt
    AuthenticationAttempt,
    /// Access denied
    AccessDenied,
    /// Tampering detected
    TamperingDetected,
    /// Replay attack detected
    ReplayAttackDetected,

    // System Events
    /// System initialized
    SystemInit,
    /// Configuration changed
    ConfigurationChange,
    /// Error occurred
    Error,
}

impl std::fmt::Display for AuditEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AuditEventType::KeyGeneration => write!(f, "KEY_GENERATION"),
            AuditEventType::KeyImport => write!(f, "KEY_IMPORT"),
            AuditEventType::KeyExport => write!(f, "KEY_EXPORT"),
            AuditEventType::KeyDeletion => write!(f, "KEY_DELETION"),
            AuditEventType::KeyRotation => write!(f, "KEY_ROTATION"),
            AuditEventType::SignatureCreated => write!(f, "SIGNATURE_CREATED"),
            AuditEventType::SignatureVerified => write!(f, "SIGNATURE_VERIFIED"),
            AuditEventType::SignatureVerificationFailed => write!(f, "SIGNATURE_VERIFICATION_FAILED"),
            AuditEventType::BatchVerification => write!(f, "BATCH_VERIFICATION"),
            AuditEventType::ChannelEstablished => write!(f, "CHANNEL_ESTABLISHED"),
            AuditEventType::ChannelClosed => write!(f, "CHANNEL_CLOSED"),
            AuditEventType::ChannelMessageSent => write!(f, "CHANNEL_MESSAGE_SENT"),
            AuditEventType::ChannelMessageReceived => write!(f, "CHANNEL_MESSAGE_RECEIVED"),
            AuditEventType::HsmOperation => write!(f, "HSM_OPERATION"),
            AuditEventType::HsmHealthCheck => write!(f, "HSM_HEALTH_CHECK"),
            AuditEventType::AuthenticationAttempt => write!(f, "AUTHENTICATION_ATTEMPT"),
            AuditEventType::AccessDenied => write!(f, "ACCESS_DENIED"),
            AuditEventType::TamperingDetected => write!(f, "TAMPERING_DETECTED"),
            AuditEventType::ReplayAttackDetected => write!(f, "REPLAY_ATTACK_DETECTED"),
            AuditEventType::SystemInit => write!(f, "SYSTEM_INIT"),
            AuditEventType::ConfigurationChange => write!(f, "CONFIGURATION_CHANGE"),
            AuditEventType::Error => write!(f, "ERROR"),
        }
    }
}

/// Severity level for audit events
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AuditSeverity {
    /// Debug-level event
    Debug,
    /// Informational event
    Info,
    /// Warning event
    Warning,
    /// Error event
    Error,
    /// Critical security event
    Critical,
}

impl Default for AuditSeverity {
    fn default() -> Self {
        AuditSeverity::Info
    }
}

/// Audit log entry
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event ID
    pub id: String,
    /// Event timestamp (Unix epoch nanoseconds)
    pub timestamp: u128,
    /// Event type
    pub event_type: AuditEventType,
    /// Severity level
    pub severity: AuditSeverity,
    /// Human-readable description
    pub description: String,
    /// Optional key ID involved
    pub key_id: Option<String>,
    /// Optional channel ID involved
    pub channel_id: Option<String>,
    /// Optional security level
    pub security_level: Option<SecurityLevel>,
    /// Optional operation duration in microseconds
    pub duration_us: Option<u64>,
    /// Optional success indicator
    pub success: Option<bool>,
    /// Optional source identifier (node, component)
    pub source: Option<String>,
    /// Optional target identifier
    pub target: Option<String>,
    /// Optional additional metadata
    pub metadata: Option<serde_json::Value>,
    /// Hash of previous event (for chain integrity)
    pub prev_hash: Option<String>,
    /// Hash of this event
    pub hash: String,
}

impl AuditEvent {
    /// Create new audit event
    pub fn new(event_type: AuditEventType, description: impl Into<String>) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);

        let id = format!(
            "{}-{:016x}",
            event_type,
            timestamp
        );

        let mut event = Self {
            id,
            timestamp,
            event_type,
            severity: AuditSeverity::Info,
            description: description.into(),
            key_id: None,
            channel_id: None,
            security_level: None,
            duration_us: None,
            success: None,
            source: None,
            target: None,
            metadata: None,
            prev_hash: None,
            hash: String::new(),
        };

        event.hash = event.compute_hash();
        event
    }

    /// Set severity level
    pub fn with_severity(mut self, severity: AuditSeverity) -> Self {
        self.severity = severity;
        self.hash = self.compute_hash();
        self
    }

    /// Set key ID
    pub fn with_key_id(mut self, key_id: impl Into<String>) -> Self {
        self.key_id = Some(key_id.into());
        self.hash = self.compute_hash();
        self
    }

    /// Set channel ID
    pub fn with_channel_id(mut self, channel_id: impl Into<String>) -> Self {
        self.channel_id = Some(channel_id.into());
        self.hash = self.compute_hash();
        self
    }

    /// Set security level
    pub fn with_security_level(mut self, level: SecurityLevel) -> Self {
        self.security_level = Some(level);
        self.hash = self.compute_hash();
        self
    }

    /// Set operation duration
    pub fn with_duration(mut self, duration_us: u64) -> Self {
        self.duration_us = Some(duration_us);
        self.hash = self.compute_hash();
        self
    }

    /// Set success status
    pub fn with_success(mut self, success: bool) -> Self {
        self.success = Some(success);
        self.hash = self.compute_hash();
        self
    }

    /// Set source identifier
    pub fn with_source(mut self, source: impl Into<String>) -> Self {
        self.source = Some(source.into());
        self.hash = self.compute_hash();
        self
    }

    /// Set target identifier
    pub fn with_target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self.hash = self.compute_hash();
        self
    }

    /// Set metadata
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self.hash = self.compute_hash();
        self
    }

    /// Set previous hash (for chain)
    pub fn with_prev_hash(mut self, prev_hash: impl Into<String>) -> Self {
        self.prev_hash = Some(prev_hash.into());
        self.hash = self.compute_hash();
        self
    }

    /// Compute SHA3-256 hash of the event
    fn compute_hash(&self) -> String {
        let mut hasher = Sha3_256::new();

        hasher.update(self.id.as_bytes());
        hasher.update(self.timestamp.to_le_bytes());
        hasher.update(self.event_type.to_string().as_bytes());
        hasher.update(self.description.as_bytes());

        if let Some(ref key_id) = self.key_id {
            hasher.update(key_id.as_bytes());
        }
        if let Some(ref channel_id) = self.channel_id {
            hasher.update(channel_id.as_bytes());
        }
        if let Some(ref prev_hash) = self.prev_hash {
            hasher.update(prev_hash.as_bytes());
        }

        hex::encode(hasher.finalize())
    }

    /// Verify event integrity
    pub fn verify_integrity(&self) -> bool {
        let expected_hash = self.compute_hash();
        // Use constant-time comparison to prevent timing attacks
        constant_time_eq(self.hash.as_bytes(), expected_hash.as_bytes())
    }
}

/// Constant-time equality comparison
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (ai, bi) in a.iter().zip(b.iter()) {
        diff |= ai ^ bi;
    }
    diff == 0
}

/// Audit log subscriber trait
///
/// Implement this trait to receive audit events in real-time.
pub trait AuditSubscriber: Send + Sync {
    /// Called when an audit event is logged
    fn on_event(&self, event: &AuditEvent);
}

/// Audit logger
///
/// Thread-safe audit logging with chain integrity verification.
pub struct AuditLogger {
    /// In-memory event buffer
    events: Arc<RwLock<VecDeque<AuditEvent>>>,
    /// Maximum events to keep in memory
    max_events: usize,
    /// Subscribers for real-time notifications
    subscribers: Arc<RwLock<Vec<Arc<dyn AuditSubscriber>>>>,
    /// Last event hash for chain integrity
    last_hash: Arc<RwLock<Option<String>>>,
    /// Minimum severity level to log
    min_severity: AuditSeverity,
}

impl AuditLogger {
    /// Create new audit logger with default settings
    pub fn new() -> Self {
        Self {
            events: Arc::new(RwLock::new(VecDeque::with_capacity(10000))),
            max_events: 10000,
            subscribers: Arc::new(RwLock::new(Vec::new())),
            last_hash: Arc::new(RwLock::new(None)),
            min_severity: AuditSeverity::Info,
        }
    }

    /// Set maximum events to keep in memory
    pub fn with_max_events(mut self, max: usize) -> Self {
        self.max_events = max;
        self
    }

    /// Set minimum severity level
    pub fn with_min_severity(mut self, severity: AuditSeverity) -> Self {
        self.min_severity = severity;
        self
    }

    /// Subscribe to audit events
    pub fn subscribe(&self, subscriber: Arc<dyn AuditSubscriber>) {
        if let Ok(mut subs) = self.subscribers.write() {
            subs.push(subscriber);
        }
    }

    /// Log an audit event
    pub fn log(&self, mut event: AuditEvent) {
        // Filter by severity
        if event.severity < self.min_severity {
            return;
        }

        // Add chain link
        if let Ok(last_hash) = self.last_hash.read() {
            if let Some(ref hash) = *last_hash {
                event = event.with_prev_hash(hash.clone());
            }
        }

        // Update last hash
        if let Ok(mut last_hash) = self.last_hash.write() {
            *last_hash = Some(event.hash.clone());
        }

        // Notify subscribers
        if let Ok(subs) = self.subscribers.read() {
            for sub in subs.iter() {
                sub.on_event(&event);
            }
        }

        // Store event
        if let Ok(mut events) = self.events.write() {
            if events.len() >= self.max_events {
                events.pop_front();
            }
            events.push_back(event);
        }
    }

    /// Get all events (most recent first)
    pub fn get_events(&self) -> Vec<AuditEvent> {
        self.events
            .read()
            .map(|e| e.iter().rev().cloned().collect())
            .unwrap_or_default()
    }

    /// Get events by type
    pub fn get_events_by_type(&self, event_type: AuditEventType) -> Vec<AuditEvent> {
        self.events
            .read()
            .map(|e| {
                e.iter()
                    .filter(|ev| ev.event_type == event_type)
                    .rev()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get events by key ID
    pub fn get_events_by_key(&self, key_id: &str) -> Vec<AuditEvent> {
        self.events
            .read()
            .map(|e| {
                e.iter()
                    .filter(|ev| ev.key_id.as_deref() == Some(key_id))
                    .rev()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get events in time range
    pub fn get_events_in_range(&self, start: u128, end: u128) -> Vec<AuditEvent> {
        self.events
            .read()
            .map(|e| {
                e.iter()
                    .filter(|ev| ev.timestamp >= start && ev.timestamp <= end)
                    .rev()
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Verify chain integrity
    pub fn verify_chain_integrity(&self) -> ChainVerificationResult {
        let events: Vec<_> = self
            .events
            .read()
            .map(|e| e.iter().cloned().collect())
            .unwrap_or_default();

        if events.is_empty() {
            return ChainVerificationResult {
                valid: true,
                total_events: 0,
                valid_events: 0,
                first_invalid_index: None,
                errors: vec![],
            };
        }

        let mut errors = Vec::new();
        let mut valid_count = 0;
        let mut first_invalid_index = None;

        for (i, event) in events.iter().enumerate() {
            // Verify event hash
            if !event.verify_integrity() {
                if first_invalid_index.is_none() {
                    first_invalid_index = Some(i);
                }
                errors.push(format!("Event {} has invalid hash", event.id));
                continue;
            }

            // Verify chain link
            if i > 0 {
                let expected_prev = &events[i - 1].hash;
                if event.prev_hash.as_ref() != Some(expected_prev) {
                    if first_invalid_index.is_none() {
                        first_invalid_index = Some(i);
                    }
                    errors.push(format!("Event {} has broken chain link", event.id));
                    continue;
                }
            }

            valid_count += 1;
        }

        ChainVerificationResult {
            valid: errors.is_empty(),
            total_events: events.len(),
            valid_events: valid_count,
            first_invalid_index,
            errors,
        }
    }

    /// Clear all events (for testing)
    pub fn clear(&self) {
        if let Ok(mut events) = self.events.write() {
            events.clear();
        }
        if let Ok(mut last_hash) = self.last_hash.write() {
            *last_hash = None;
        }
    }

    /// Export events as JSON
    pub fn export_json(&self) -> Result<String, serde_json::Error> {
        let events = self.get_events();
        serde_json::to_string_pretty(&events)
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of chain integrity verification
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChainVerificationResult {
    /// Overall validity
    pub valid: bool,
    /// Total number of events
    pub total_events: usize,
    /// Number of valid events
    pub valid_events: usize,
    /// Index of first invalid event (if any)
    pub first_invalid_index: Option<usize>,
    /// Error messages
    pub errors: Vec<String>,
}

/// Global audit logger instance
static GLOBAL_LOGGER: std::sync::OnceLock<AuditLogger> = std::sync::OnceLock::new();

/// Get or initialize global audit logger
pub fn global_logger() -> &'static AuditLogger {
    GLOBAL_LOGGER.get_or_init(AuditLogger::new)
}

/// Convenience macro for logging audit events
#[macro_export]
macro_rules! audit_log {
    ($event_type:expr, $desc:expr) => {
        $crate::audit::global_logger().log(
            $crate::audit::AuditEvent::new($event_type, $desc)
        )
    };
    ($event_type:expr, $desc:expr, $($method:ident($arg:expr)),*) => {
        $crate::audit::global_logger().log(
            $crate::audit::AuditEvent::new($event_type, $desc)
            $(.$method($arg))*
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audit_event_creation() {
        let event = AuditEvent::new(AuditEventType::KeyGeneration, "Generated test key")
            .with_key_id("key-123")
            .with_security_level(SecurityLevel::Standard)
            .with_success(true);

        assert_eq!(event.event_type, AuditEventType::KeyGeneration);
        assert_eq!(event.key_id, Some("key-123".to_string()));
        assert!(event.verify_integrity());
    }

    #[test]
    fn test_audit_event_integrity() {
        let event = AuditEvent::new(AuditEventType::SignatureCreated, "Signed message");

        assert!(event.verify_integrity());

        // Tamper with event
        let mut tampered = event.clone();
        tampered.description = "Modified description".to_string();

        // Hash should no longer match
        assert!(!tampered.verify_integrity());
    }

    #[test]
    fn test_audit_logger_basic() {
        let logger = AuditLogger::new();

        logger.log(AuditEvent::new(
            AuditEventType::KeyGeneration,
            "Test key generated",
        ));
        logger.log(AuditEvent::new(
            AuditEventType::SignatureCreated,
            "Test signature",
        ));

        let events = logger.get_events();
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_audit_chain_integrity() {
        let logger = AuditLogger::new();

        for i in 0..5 {
            logger.log(AuditEvent::new(
                AuditEventType::SignatureCreated,
                format!("Signature {}", i),
            ));
        }

        let result = logger.verify_chain_integrity();
        assert!(result.valid);
        assert_eq!(result.total_events, 5);
        assert_eq!(result.valid_events, 5);
    }

    #[test]
    fn test_audit_event_filtering() {
        let logger = AuditLogger::new();

        logger.log(
            AuditEvent::new(AuditEventType::KeyGeneration, "Key 1").with_key_id("key-1"),
        );
        logger.log(
            AuditEvent::new(AuditEventType::SignatureCreated, "Sig 1").with_key_id("key-1"),
        );
        logger.log(
            AuditEvent::new(AuditEventType::SignatureCreated, "Sig 2").with_key_id("key-2"),
        );

        let key_events = logger.get_events_by_key("key-1");
        assert_eq!(key_events.len(), 2);

        let sig_events = logger.get_events_by_type(AuditEventType::SignatureCreated);
        assert_eq!(sig_events.len(), 2);
    }

    #[test]
    fn test_severity_filtering() {
        let logger = AuditLogger::new().with_min_severity(AuditSeverity::Warning);

        logger.log(
            AuditEvent::new(AuditEventType::SignatureVerified, "Normal event")
                .with_severity(AuditSeverity::Info),
        );
        logger.log(
            AuditEvent::new(AuditEventType::AccessDenied, "Warning event")
                .with_severity(AuditSeverity::Warning),
        );

        let events = logger.get_events();
        assert_eq!(events.len(), 1); // Only warning-level event logged
    }

    struct TestSubscriber {
        count: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl AuditSubscriber for TestSubscriber {
        fn on_event(&self, _event: &AuditEvent) {
            self.count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn test_audit_subscriber() {
        let logger = AuditLogger::new();
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        let subscriber = Arc::new(TestSubscriber {
            count: count.clone(),
        });
        logger.subscribe(subscriber);

        logger.log(AuditEvent::new(AuditEventType::KeyGeneration, "Test"));
        logger.log(AuditEvent::new(AuditEventType::SignatureCreated, "Test"));

        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn test_export_json() {
        let logger = AuditLogger::new();

        logger.log(AuditEvent::new(AuditEventType::SystemInit, "Test export"));

        let json = logger.export_json().expect("Export failed");
        assert!(json.contains("SYSTEM_INIT"));
        assert!(json.contains("Test export"));
    }
}
