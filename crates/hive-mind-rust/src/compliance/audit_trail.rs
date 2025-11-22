//! # SOX Section 404 Compliant Audit Trail System
//!
//! This module implements an immutable, tamper-proof audit trail system that meets
//! Sarbanes-Oxley Section 404 requirements for internal controls over financial reporting.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tokio::sync::RwLock;
use uuid::Uuid;
use zeroize::ZeroizeOnDrop;

use crate::error::{Result, HiveMindError};

/// Immutable audit trail system for SOX compliance
#[derive(Debug)]
pub struct AuditTrail {
    /// Immutable log storage
    immutable_log: Arc<RwLock<ImmutableLog>>,
    
    /// Audit configuration
    config: AuditConfig,
    
    /// Digital signature manager for log integrity
    signature_manager: Arc<SignatureManager>,
    
    /// Event indexing for fast queries
    event_index: Arc<RwLock<HashMap<AuditEventType, Vec<Uuid>>>>,
}

/// Immutable log implementation using blockchain-like structure
#[derive(Debug)]
pub struct ImmutableLog {
    /// Chain of audit events
    events: Vec<AuditEvent>,
    
    /// Current block hash for integrity verification
    current_hash: String,
    
    /// Genesis hash for the audit trail
    genesis_hash: String,
}

/// Individual audit event with full traceability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier
    pub id: Uuid,
    
    /// Event type for categorization
    pub event_type: AuditEventType,
    
    /// UTC timestamp when event occurred
    pub timestamp: DateTime<Utc>,
    
    /// User ID who triggered the event
    pub user_id: Option<String>,
    
    /// Session ID for correlation
    pub session_id: Option<String>,
    
    /// IP address of the requester
    pub source_ip: Option<String>,
    
    /// Detailed event description
    pub description: String,
    
    /// Structured event data
    pub event_data: serde_json::Value,
    
    /// Previous event hash for chaining
    pub previous_hash: String,
    
    /// Digital signature for integrity
    pub signature: String,
    
    /// Risk level of the event
    pub risk_level: RiskLevel,
    
    /// Compliance flags
    pub compliance_flags: Vec<ComplianceFlag>,
    
    /// Event hash for integrity verification
    pub event_hash: String,
}

/// Types of auditable events
#[derive(Debug, Clone, Serialize, Deserialize, Hash, Eq, PartialEq)]
pub enum AuditEventType {
    // Authentication Events
    UserLogin,
    UserLogout,
    FailedLogin,
    PasswordChange,
    MFAChallenge,
    
    // Authorization Events
    AccessGranted,
    AccessDenied,
    RoleAssignment,
    PrivilegeEscalation,
    
    // Financial Transaction Events
    TradeExecution,
    OrderPlacement,
    OrderCancellation,
    SettlementProcessed,
    PaymentProcessed,
    
    // Data Access Events
    DataRead,
    DataWrite,
    DataDelete,
    DataExport,
    DataImport,
    
    // System Events
    SystemStartup,
    SystemShutdown,
    ConfigurationChange,
    EmergencyShutdown,
    RecoveryInitiated,
    
    // Compliance Events
    RegulatoryReport,
    AuditQuery,
    ComplianceViolation,
    RiskAlert,
    SuspiciousActivity,
    
    // Administrative Events
    UserCreation,
    UserDeletion,
    SystemMaintenance,
    BackupCreated,
    BackupRestored,
}

/// Risk levels for audit events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Compliance flags for regulatory requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceFlag {
    SOX404,
    PCIDSS,
    GDPR,
    BaselIII,
    MiFIDII,
    AMLKYC,
}

/// Audit trail configuration
#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Maximum events before archival
    pub max_events_in_memory: usize,
    
    /// Retention period in days
    pub retention_days: u32,
    
    /// Enable real-time alerting
    pub enable_real_time_alerts: bool,
    
    /// Archive storage path
    pub archive_path: String,
}

/// Digital signature manager for audit log integrity
#[derive(Debug)]
pub struct SignatureManager {
    /// Private key for signing (zeroized on drop)
    private_key: ZeroizeOnDrop<[u8; 32]>,
    
    /// Public key for verification
    public_key: [u8; 32],
}

impl AuditTrail {
    /// Create a new audit trail system
    pub async fn new() -> Result<Self> {
        let config = AuditConfig::default();
        let signature_manager = Arc::new(SignatureManager::new()?);
        
        // Initialize with genesis event
        let genesis_event = AuditEvent::genesis();
        let genesis_hash = genesis_event.event_hash.clone();
        
        let immutable_log = Arc::new(RwLock::new(ImmutableLog {
            events: vec![genesis_event],
            current_hash: genesis_hash.clone(),
            genesis_hash,
        }));
        
        let event_index = Arc::new(RwLock::new(HashMap::new()));
        
        Ok(Self {
            immutable_log,
            config,
            signature_manager,
            event_index,
        })
    }
    
    /// Start the audit trail system
    pub async fn start(&self) -> Result<()> {
        self.log_event(
            AuditEventType::SystemStartup,
            "Audit trail system started".to_string(),
            serde_json::json!({"component": "audit_trail"}),
            None,
            None,
            None,
        ).await?;
        
        tracing::info!("Audit trail system started with SOX 404 compliance");
        Ok(())
    }
    
    /// Log an audit event with full traceability
    pub async fn log_event(
        &self,
        event_type: AuditEventType,
        description: String,
        event_data: serde_json::Value,
        user_id: Option<String>,
        session_id: Option<String>,
        source_ip: Option<String>,
    ) -> Result<Uuid> {
        let event_id = Uuid::new_v4();
        
        // Get previous hash for chaining
        let previous_hash = {
            let log = self.immutable_log.read().await;
            log.current_hash.clone()
        };
        
        // Determine risk level based on event type
        let risk_level = Self::determine_risk_level(&event_type);
        
        // Assign compliance flags
        let compliance_flags = Self::assign_compliance_flags(&event_type);
        
        // Create the audit event
        let mut event = AuditEvent {
            id: event_id,
            event_type: event_type.clone(),
            timestamp: Utc::now(),
            user_id,
            session_id,
            source_ip,
            description,
            event_data,
            previous_hash,
            signature: String::new(), // Will be filled by signing
            risk_level,
            compliance_flags,
            event_hash: String::new(), // Will be filled after hashing
        };
        
        // Calculate event hash
        event.event_hash = self.calculate_event_hash(&event)?;
        
        // Sign the event for integrity
        event.signature = self.signature_manager.sign(&event.event_hash)?;
        
        // Add to immutable log
        {
            let mut log = self.immutable_log.write().await;
            log.events.push(event.clone());
            log.current_hash = event.event_hash.clone();
        }
        
        // Update index
        {
            let mut index = self.event_index.write().await;
            index.entry(event_type.clone()).or_insert_with(Vec::new).push(event_id);
        }
        
        // Check for real-time alerts
        if self.config.enable_real_time_alerts && matches!(risk_level, RiskLevel::High | RiskLevel::Critical) {
            self.send_alert(&event).await?;
        }
        
        tracing::debug!("Audit event logged: {:?} - {}", event_type, event_id);
        Ok(event_id)
    }
    
    /// Query audit events by type
    pub async fn query_by_type(&self, event_type: &AuditEventType) -> Result<Vec<AuditEvent>> {
        let index = self.event_index.read().await;
        let event_ids = index.get(event_type).cloned().unwrap_or_default();
        
        let log = self.immutable_log.read().await;
        let events: Vec<AuditEvent> = log.events.iter()
            .filter(|e| event_ids.contains(&e.id))
            .cloned()
            .collect();
        
        Ok(events)
    }
    
    /// Query audit events by user ID
    pub async fn query_by_user(&self, user_id: &str) -> Result<Vec<AuditEvent>> {
        let log = self.immutable_log.read().await;
        let events: Vec<AuditEvent> = log.events.iter()
            .filter(|e| e.user_id.as_ref().map(|u| u == user_id).unwrap_or(false))
            .cloned()
            .collect();
        
        Ok(events)
    }
    
    /// Query audit events by time range
    pub async fn query_by_time_range(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<Vec<AuditEvent>> {
        let log = self.immutable_log.read().await;
        let events: Vec<AuditEvent> = log.events.iter()
            .filter(|e| e.timestamp >= start && e.timestamp <= end)
            .cloned()
            .collect();
        
        Ok(events)
    }
    
    /// Verify the integrity of the audit trail
    pub async fn verify_integrity(&self) -> Result<bool> {
        let log = self.immutable_log.read().await;
        
        if log.events.is_empty() {
            return Ok(true);
        }
        
        // Verify the chain of hashes
        for i in 1..log.events.len() {
            let current_event = &log.events[i];
            let previous_event = &log.events[i - 1];
            
            // Verify hash chaining
            if current_event.previous_hash != previous_event.event_hash {
                tracing::error!("Hash chain broken at event: {}", current_event.id);
                return Ok(false);
            }
            
            // Verify event signature
            if !self.signature_manager.verify(&current_event.event_hash, &current_event.signature)? {
                tracing::error!("Invalid signature for event: {}", current_event.id);
                return Ok(false);
            }
            
            // Verify event hash
            let calculated_hash = self.calculate_event_hash(current_event)?;
            if current_event.event_hash != calculated_hash {
                tracing::error!("Hash mismatch for event: {}", current_event.id);
                return Ok(false);
            }
        }
        
        tracing::info!("Audit trail integrity verification successful");
        Ok(true)
    }
    
    /// Generate compliance report for auditors
    pub async fn generate_compliance_report(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
        flags: Vec<ComplianceFlag>,
    ) -> Result<ComplianceReport> {
        let events = self.query_by_time_range(start, end).await?;
        
        let filtered_events: Vec<AuditEvent> = events.into_iter()
            .filter(|e| e.compliance_flags.iter().any(|f| flags.contains(f)))
            .collect();
        
        let report = ComplianceReport {
            report_id: Uuid::new_v4(),
            generated_at: Utc::now(),
            period_start: start,
            period_end: end,
            compliance_flags: flags,
            total_events: filtered_events.len(),
            high_risk_events: filtered_events.iter().filter(|e| matches!(e.risk_level, RiskLevel::High | RiskLevel::Critical)).count(),
            events: filtered_events,
            integrity_verified: self.verify_integrity().await?,
        };
        
        Ok(report)
    }
    
    /// Export audit trail for regulatory authorities
    pub async fn export_for_regulator(&self, format: ExportFormat) -> Result<String> {
        let log = self.immutable_log.read().await;
        
        match format {
            ExportFormat::Json => Ok(serde_json::to_string_pretty(&log.events)?),
            ExportFormat::Csv => {
                let mut csv = String::from("id,event_type,timestamp,user_id,description,risk_level\n");
                for event in &log.events {
                    csv.push_str(&format!(
                        "{},{:?},{},{},{},{:?}\n",
                        event.id,
                        event.event_type,
                        event.timestamp.to_rfc3339(),
                        event.user_id.as_deref().unwrap_or("N/A"),
                        event.description.replace(',', ";"),
                        event.risk_level
                    ));
                }
                Ok(csv)
            }
            ExportFormat::Xml => {
                // Basic XML export - in production, use proper XML library
                let mut xml = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<audit_trail>\n");
                for event in &log.events {
                    xml.push_str(&format!(
                        "  <event id=\"{}\" type=\"{:?}\" timestamp=\"{}\" risk=\"{:?}\">\n    <description>{}</description>\n  </event>\n",
                        event.id,
                        event.event_type,
                        event.timestamp.to_rfc3339(),
                        event.risk_level,
                        event.description
                    ));
                }
                xml.push_str("</audit_trail>");
                Ok(xml)
            }
        }
    }
    
    /// Calculate hash for an audit event
    fn calculate_event_hash(&self, event: &AuditEvent) -> Result<String> {
        let mut hasher = Sha256::new();
        
        // Hash all fields except the hash itself and signature
        hasher.update(event.id.as_bytes());
        hasher.update(format!("{:?}", event.event_type).as_bytes());
        hasher.update(event.timestamp.to_rfc3339().as_bytes());
        hasher.update(event.user_id.as_deref().unwrap_or("").as_bytes());
        hasher.update(event.session_id.as_deref().unwrap_or("").as_bytes());
        hasher.update(event.source_ip.as_deref().unwrap_or("").as_bytes());
        hasher.update(event.description.as_bytes());
        hasher.update(event.event_data.to_string().as_bytes());
        hasher.update(event.previous_hash.as_bytes());
        hasher.update(format!("{:?}", event.risk_level).as_bytes());
        hasher.update(format!("{:?}", event.compliance_flags).as_bytes());
        
        Ok(format!("{:x}", hasher.finalize()))
    }
    
    /// Determine risk level based on event type
    fn determine_risk_level(event_type: &AuditEventType) -> RiskLevel {
        match event_type {
            AuditEventType::FailedLogin | AuditEventType::AccessDenied => RiskLevel::Medium,
            AuditEventType::PrivilegeEscalation | AuditEventType::EmergencyShutdown 
                | AuditEventType::ComplianceViolation => RiskLevel::Critical,
            AuditEventType::TradeExecution | AuditEventType::PaymentProcessed 
                | AuditEventType::SuspiciousActivity => RiskLevel::High,
            AuditEventType::DataDelete | AuditEventType::ConfigurationChange => RiskLevel::High,
            _ => RiskLevel::Low,
        }
    }
    
    /// Assign compliance flags based on event type
    fn assign_compliance_flags(event_type: &AuditEventType) -> Vec<ComplianceFlag> {
        match event_type {
            AuditEventType::TradeExecution | AuditEventType::OrderPlacement 
                | AuditEventType::OrderCancellation => vec![ComplianceFlag::MiFIDII, ComplianceFlag::SOX404],
            AuditEventType::PaymentProcessed | AuditEventType::SettlementProcessed => 
                vec![ComplianceFlag::PCIDSS, ComplianceFlag::SOX404],
            AuditEventType::SuspiciousActivity => vec![ComplianceFlag::AMLKYC],
            AuditEventType::DataRead | AuditEventType::DataWrite | AuditEventType::DataDelete => 
                vec![ComplianceFlag::GDPR],
            AuditEventType::RiskAlert => vec![ComplianceFlag::BaselIII],
            _ => vec![ComplianceFlag::SOX404],
        }
    }
    
    /// Send real-time alert for high-risk events
    async fn send_alert(&self, event: &AuditEvent) -> Result<()> {
        tracing::warn!(
            "HIGH RISK AUDIT EVENT: {:?} - {} (User: {}, Risk: {:?})",
            event.event_type,
            event.description,
            event.user_id.as_deref().unwrap_or("Unknown"),
            event.risk_level
        );
        
        // In production, integrate with alerting systems (PagerDuty, Slack, etc.)
        Ok(())
    }
}

impl AuditEvent {
    /// Create the genesis event for the audit trail
    fn genesis() -> Self {
        let id = Uuid::new_v4();
        let timestamp = Utc::now();
        let description = "Audit trail system genesis".to_string();
        let event_data = serde_json::json!({
            "system": "hive-mind-rust",
            "version": env!("CARGO_PKG_VERSION"),
            "genesis": true
        });
        
        let mut hasher = Sha256::new();
        hasher.update(id.as_bytes());
        hasher.update(timestamp.to_rfc3339().as_bytes());
        hasher.update(description.as_bytes());
        let genesis_hash = format!("{:x}", hasher.finalize());
        
        Self {
            id,
            event_type: AuditEventType::SystemStartup,
            timestamp,
            user_id: None,
            session_id: None,
            source_ip: None,
            description,
            event_data,
            previous_hash: "0".repeat(64), // Genesis has no previous hash
            signature: "genesis".to_string(),
            risk_level: RiskLevel::Low,
            compliance_flags: vec![ComplianceFlag::SOX404],
            event_hash: genesis_hash,
        }
    }
}

impl SignatureManager {
    /// Create a new signature manager
    fn new() -> Result<Self> {
        use ed25519_dalek::{SigningKey, VerifyingKey};
        use rand::rngs::OsRng;
        
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        
        let private_key = ZeroizeOnDrop::new(signing_key.to_bytes());
        let public_key = verifying_key.to_bytes();
        
        Ok(Self {
            private_key,
            public_key,
        })
    }
    
    /// Sign a hash for integrity verification
    fn sign(&self, hash: &str) -> Result<String> {
        use ed25519_dalek::SigningKey;
        
        let signing_key = SigningKey::from_bytes(&self.private_key);
        let signature = signing_key.sign(hash.as_bytes());
        
        Ok(hex::encode(signature.to_bytes()))
    }
    
    /// Verify a signature
    fn verify(&self, hash: &str, signature: &str) -> Result<bool> {
        use ed25519_dalek::{VerifyingKey, Signature};
        
        if signature == "genesis" {
            return Ok(true); // Genesis event doesn't have a real signature
        }
        
        let verifying_key = VerifyingKey::from_bytes(&self.public_key)
            .map_err(|e| HiveMindError::Cryptographic(format!("Invalid public key: {}", e)))?;
        
        let signature_bytes = hex::decode(signature)
            .map_err(|e| HiveMindError::Cryptographic(format!("Invalid signature format: {}", e)))?;
        
        let signature = Signature::from_bytes(&signature_bytes.try_into().map_err(|_| {
            HiveMindError::Cryptographic("Invalid signature length".to_string())
        })?)
        .map_err(|e| HiveMindError::Cryptographic(format!("Invalid signature: {}", e)))?;
        
        Ok(verifying_key.verify(hash.as_bytes(), &signature).is_ok())
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            max_events_in_memory: 100_000,
            retention_days: 2555, // 7 years for financial records
            enable_real_time_alerts: true,
            archive_path: "./audit_archives".to_string(),
        }
    }
}

/// Compliance report structure for auditors
#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub report_id: Uuid,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub compliance_flags: Vec<ComplianceFlag>,
    pub total_events: usize,
    pub high_risk_events: usize,
    pub events: Vec<AuditEvent>,
    pub integrity_verified: bool,
}

/// Export formats for regulatory reporting
#[derive(Debug, Clone)]
pub enum ExportFormat {
    Json,
    Csv,
    Xml,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_audit_trail_creation() {
        let audit_trail = AuditTrail::new().await.unwrap();
        assert!(audit_trail.verify_integrity().await.unwrap());
    }

    #[tokio::test]
    async fn test_event_logging() {
        let audit_trail = AuditTrail::new().await.unwrap();
        
        let event_id = audit_trail.log_event(
            AuditEventType::UserLogin,
            "User logged in successfully".to_string(),
            serde_json::json!({"ip": "192.168.1.1"}),
            Some("user123".to_string()),
            Some("session456".to_string()),
            Some("192.168.1.1".to_string()),
        ).await.unwrap();
        
        assert!(!event_id.is_nil());
        assert!(audit_trail.verify_integrity().await.unwrap());
    }

    #[tokio::test]
    async fn test_compliance_report() {
        let audit_trail = AuditTrail::new().await.unwrap();
        
        // Log a few events
        audit_trail.log_event(
            AuditEventType::TradeExecution,
            "Trade executed".to_string(),
            serde_json::json!({"amount": 1000}),
            Some("trader1".to_string()),
            None,
            None,
        ).await.unwrap();
        
        let start = Utc::now() - chrono::Duration::hours(1);
        let end = Utc::now() + chrono::Duration::hours(1);
        
        let report = audit_trail.generate_compliance_report(
            start,
            end,
            vec![ComplianceFlag::MiFIDII],
        ).await.unwrap();
        
        assert_eq!(report.total_events, 1);
        assert!(report.integrity_verified);
    }
}