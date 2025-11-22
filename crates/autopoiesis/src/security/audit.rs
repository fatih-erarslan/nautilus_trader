//! Enterprise-grade audit logging system for SOX and GDPR compliance
//! 
//! This module implements comprehensive audit logging with:
//! - Real-time security event tracking
//! - SOX-compliant financial audit trails  
//! - GDPR-compliant data access logging
//! - Encrypted audit log storage
//! - Automated compliance reporting

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use ring::aead::{Aad, BoundKey, Nonce, OpeningKey, SealingKey, UnboundKey, AES_256_GCM, NONCE_LEN};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use tokio::fs::{File, OpenOptions};
use tokio::io::{AsyncWriteExt, BufWriter};
use tokio::sync::RwLock;
use tracing::{error, info, warn};

/// Comprehensive audit event structure for compliance logging
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unique event identifier
    pub event_id: String,
    
    /// Event timestamp (ISO 8601 format)
    pub timestamp: DateTime<Utc>,
    
    /// Type of audit event
    pub event_type: AuditEventType,
    
    /// User identifier (if applicable)
    pub user_id: Option<String>,
    
    /// Session identifier
    pub session_id: Option<String>,
    
    /// Source IP address
    pub ip_address: Option<IpAddr>,
    
    /// User agent / client information
    pub user_agent: Option<String>,
    
    /// Resource being accessed
    pub resource: String,
    
    /// Action performed
    pub action: AuditAction,
    
    /// Result of the action
    pub result: AuditResult,
    
    /// Risk assessment score (0-10)
    pub risk_score: u8,
    
    /// Data classification level
    pub data_classification: DataClassification,
    
    /// Compliance tags
    pub compliance_tags: Vec<ComplianceTag>,
    
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    
    /// Request details (for API calls)
    pub request_details: Option<RequestDetails>,
    
    /// Response details
    pub response_details: Option<ResponseDetails>,
    
    /// Error information (if applicable)
    pub error_info: Option<ErrorInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditEventType {
    /// User authentication events
    Authentication,
    
    /// Authorization decisions
    Authorization,
    
    /// Data access operations
    DataAccess,
    
    /// Data modification operations  
    DataModification,
    
    /// Data deletion operations
    DataDeletion,
    
    /// System configuration changes
    SystemConfiguration,
    
    /// Security policy violations
    SecurityViolation,
    
    /// Suspicious activity detection
    SuspiciousActivity,
    
    /// Financial transactions
    FinancialTransaction,
    
    /// Compliance operations
    ComplianceOperation,
    
    /// Privacy operations (GDPR)
    PrivacyOperation,
    
    /// Audit log access
    AuditAccess,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditAction {
    // Authentication actions
    Login,
    Logout,
    PasswordChange,
    MfaChallenge,
    MfaVerification,
    
    // Data actions
    Create,
    Read,
    Update,
    Delete,
    Export,
    Import,
    Backup,
    Restore,
    
    // System actions
    Configure,
    Deploy,
    Restart,
    Shutdown,
    
    // Financial actions
    Trade,
    Transfer,
    Withdraw,
    Deposit,
    
    // Compliance actions
    ConsentGiven,
    ConsentWithdrawn,
    DataErasure,
    DataPortability,
    
    // Security actions
    Block,
    Quarantine,
    Alert,
    
    // Custom action
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuditResult {
    Success,
    Failed,
    Blocked,
    Denied,
    Warning,
    Error,
    Timeout,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
    Financial,
    PersonalData,
    HealthData,
    BiometricData,
    CreditCardData,
    TradingData,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplianceTag {
    SOX,
    GDPR,
    PCI_DSS,
    HIPAA,
    SOC2,
    ISO27001,
    CCPA,
    PIPEDA,
    Financial,
    Privacy,
    Security,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestDetails {
    pub method: String,
    pub url: String,
    pub headers: HashMap<String, String>,
    pub body_size: usize,
    pub query_params: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResponseDetails {
    pub status_code: u16,
    pub headers: HashMap<String, String>,
    pub body_size: usize,
    pub processing_time_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorInfo {
    pub error_code: String,
    pub error_message: String,
    pub error_category: ErrorCategory,
    pub stack_trace: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorCategory {
    Authentication,
    Authorization,
    Validation,
    Business,
    System,
    Network,
    Database,
    External,
}

/// Enterprise audit logger with encryption and compliance features
pub struct ComplianceAuditLogger {
    /// Encryption key for audit logs
    sealing_key: SealingKey<OneNonceSequence>,
    
    /// Opening key for reading encrypted logs
    opening_key: OpeningKey<OneNonceSequence>,
    
    /// In-memory event buffer
    event_buffer: Arc<RwLock<Vec<AuditEvent>>>,
    
    /// Configuration
    config: AuditConfig,
    
    /// File writer for persistent logging
    log_writer: Arc<RwLock<BufWriter<File>>>,
    
    /// Random number generator
    rng: SystemRandom,
    
    /// Compliance metrics
    metrics: Arc<RwLock<ComplianceMetrics>>,
}

#[derive(Debug, Clone)]
pub struct AuditConfig {
    /// Log file path
    pub log_file_path: String,
    
    /// Maximum events in buffer before flush
    pub buffer_size: usize,
    
    /// Automatic flush interval in seconds
    pub flush_interval_seconds: u64,
    
    /// Enable encryption of audit logs
    pub encrypt_logs: bool,
    
    /// Retention period in days
    pub retention_days: u32,
    
    /// Enable real-time alerts
    pub real_time_alerts: bool,
    
    /// Minimum risk score for immediate alerts
    pub alert_threshold: u8,
    
    /// SOX compliance mode
    pub sox_compliance: bool,
    
    /// GDPR compliance mode
    pub gdpr_compliance: bool,
    
    /// Enable audit log integrity verification
    pub integrity_verification: bool,
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            log_file_path: "audit.log".to_string(),
            buffer_size: 1000,
            flush_interval_seconds: 60,
            encrypt_logs: true,
            retention_days: 2555, // 7 years for financial compliance
            real_time_alerts: true,
            alert_threshold: 7,
            sox_compliance: true,
            gdpr_compliance: true,
            integrity_verification: true,
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct ComplianceMetrics {
    pub total_events: u64,
    pub high_risk_events: u64,
    pub failed_authentications: u64,
    pub data_access_events: u64,
    pub financial_transactions: u64,
    pub privacy_operations: u64,
    pub security_violations: u64,
    pub sox_events: u64,
    pub gdpr_events: u64,
    pub last_24h_events: u64,
    pub encryption_failures: u64,
    pub integrity_violations: u64,
}

/// OneNonce sequence for AEAD encryption
struct OneNonceSequence {
    current: u32,
}

impl OneNonceSequence {
    fn new() -> Self {
        Self { current: 0 }
    }
}

impl ring::aead::NonceSequence for OneNonceSequence {
    fn advance(&mut self) -> Result<Nonce, ring::error::Unspecified> {
        let mut nonce_bytes = [0u8; NONCE_LEN];
        let bytes = self.current.to_le_bytes();
        nonce_bytes[..bytes.len()].copy_from_slice(&bytes);
        self.current += 1;
        Nonce::try_assume_unique_for_key(&nonce_bytes)
    }
}

impl ComplianceAuditLogger {
    /// Create a new compliance audit logger
    pub async fn new(config: AuditConfig) -> Result<Self> {
        // Generate encryption keys
        let rng = SystemRandom::new();
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)
            .map_err(|_| anyhow!("Failed to generate encryption key"))?;
        
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .map_err(|_| anyhow!("Failed to create encryption key"))?;
        
        let sealing_key = SealingKey::new(unbound_key, OneNonceSequence::new());
        
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .map_err(|_| anyhow!("Failed to create decryption key"))?;
        
        let opening_key = OpeningKey::new(unbound_key, OneNonceSequence::new());
        
        // Open log file
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&config.log_file_path)
            .await
            .map_err(|e| anyhow!("Failed to open audit log file: {}", e))?;
        
        let log_writer = Arc::new(RwLock::new(BufWriter::new(file)));
        
        Ok(Self {
            sealing_key,
            opening_key,
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            config,
            log_writer,
            rng,
            metrics: Arc::new(RwLock::new(ComplianceMetrics::default())),
        })
    }
    
    /// Log an audit event with full compliance tracking
    pub async fn log_event(&mut self, mut event: AuditEvent) -> Result<()> {
        // Generate unique event ID if not provided
        if event.event_id.is_empty() {
            event.event_id = uuid::Uuid::new_v4().to_string();
        }
        
        // Add compliance tags based on configuration
        if self.config.sox_compliance {
            event.compliance_tags.push(ComplianceTag::SOX);
        }
        
        if self.config.gdpr_compliance {
            event.compliance_tags.push(ComplianceTag::GDPR);
        }
        
        // Update metrics
        self.update_metrics(&event).await;
        
        // Check for immediate alerts
        if self.config.real_time_alerts && event.risk_score >= self.config.alert_threshold {
            self.send_alert(&event).await?;
        }
        
        // Log to structured logging
        self.log_to_tracing(&event).await;
        
        // Add to buffer
        {
            let mut buffer = self.event_buffer.write().await;
            buffer.push(event.clone());
            
            // Flush if buffer is full
            if buffer.len() >= self.config.buffer_size {
                self.flush_buffer().await?;
            }
        }
        
        Ok(())
    }
    
    /// Update compliance metrics
    async fn update_metrics(&self, event: &AuditEvent) {
        let mut metrics = self.metrics.write().await;
        
        metrics.total_events += 1;
        
        if event.risk_score >= 7 {
            metrics.high_risk_events += 1;
        }
        
        match &event.event_type {
            AuditEventType::Authentication => {
                if matches!(event.result, AuditResult::Failed | AuditResult::Blocked) {
                    metrics.failed_authentications += 1;
                }
            },
            AuditEventType::DataAccess => {
                metrics.data_access_events += 1;
            },
            AuditEventType::FinancialTransaction => {
                metrics.financial_transactions += 1;
            },
            AuditEventType::PrivacyOperation => {
                metrics.privacy_operations += 1;
            },
            AuditEventType::SecurityViolation => {
                metrics.security_violations += 1;
            },
            _ => {}
        }
        
        if event.compliance_tags.contains(&ComplianceTag::SOX) {
            metrics.sox_events += 1;
        }
        
        if event.compliance_tags.contains(&ComplianceTag::GDPR) {
            metrics.gdpr_events += 1;
        }
        
        // Update 24h counter (simplified - in production, use time-based sliding window)
        metrics.last_24h_events += 1;
    }
    
    /// Send real-time alert for high-risk events
    async fn send_alert(&self, event: &AuditEvent) -> Result<()> {
        error!(
            target: "compliance_alert",
            event_id = %event.event_id,
            event_type = ?event.event_type,
            user = event.user_id.as_deref().unwrap_or("unknown"),
            risk_score = event.risk_score,
            resource = %event.resource,
            ip = event.ip_address.map(|ip| ip.to_string()).as_deref().unwrap_or("unknown"),
            "HIGH RISK AUDIT EVENT - IMMEDIATE ATTENTION REQUIRED"
        );
        
        // In production, integrate with alerting systems (email, Slack, PagerDuty, etc.)
        // This is a placeholder for demonstration
        
        Ok(())
    }
    
    /// Log to structured tracing system
    async fn log_to_tracing(&self, event: &AuditEvent) {
        match event.risk_score {
            8..=10 => {
                error!(
                    target: "audit",
                    event_id = %event.event_id,
                    event_type = ?event.event_type,
                    user = event.user_id.as_deref().unwrap_or("anonymous"),
                    action = ?event.action,
                    result = ?event.result,
                    risk_score = event.risk_score,
                    resource = %event.resource,
                    data_classification = ?event.data_classification,
                    compliance_tags = ?event.compliance_tags,
                    sox_compliant = event.compliance_tags.contains(&ComplianceTag::SOX),
                    gdpr_relevant = event.compliance_tags.contains(&ComplianceTag::GDPR),
                    "Critical audit event"
                );
            },
            5..=7 => {
                warn!(
                    target: "audit",
                    event_id = %event.event_id,
                    event_type = ?event.event_type,
                    user = event.user_id.as_deref().unwrap_or("anonymous"),
                    action = ?event.action,
                    result = ?event.result,
                    risk_score = event.risk_score,
                    resource = %event.resource,
                    sox_compliant = event.compliance_tags.contains(&ComplianceTag::SOX),
                    gdpr_relevant = event.compliance_tags.contains(&ComplianceTag::GDPR),
                    "Medium risk audit event"
                );
            },
            _ => {
                info!(
                    target: "audit",
                    event_id = %event.event_id,
                    event_type = ?event.event_type,
                    user = event.user_id.as_deref().unwrap_or("anonymous"),
                    action = ?event.action,
                    result = ?event.result,
                    resource = %event.resource,
                    sox_compliant = event.compliance_tags.contains(&ComplianceTag::SOX),
                    gdpr_relevant = event.compliance_tags.contains(&ComplianceTag::GDPR),
                    "Audit event logged"
                );
            }
        }
    }
    
    /// Flush event buffer to encrypted log file
    pub async fn flush_buffer(&mut self) -> Result<()> {
        let events = {
            let mut buffer = self.event_buffer.write().await;
            std::mem::take(&mut *buffer)
        };
        
        if events.is_empty() {
            return Ok(());
        }
        
        let mut writer = self.log_writer.write().await;
        
        for event in events {
            let json_data = serde_json::to_string(&event)
                .map_err(|e| anyhow!("Failed to serialize audit event: {}", e))?;
            
            if self.config.encrypt_logs {
                // Encrypt the log entry
                let mut data = json_data.into_bytes();
                let tag = self.sealing_key.seal_in_place_separate_tag(
                    Aad::empty(),
                    &mut data,
                ).map_err(|_| anyhow!("Failed to encrypt audit log entry"))?;
                
                // Write encrypted data with tag
                let encrypted_entry = EncryptedLogEntry {
                    data,
                    tag: tag.as_ref().to_vec(),
                    timestamp: event.timestamp,
                };
                
                let encrypted_json = serde_json::to_string(&encrypted_entry)
                    .map_err(|e| anyhow!("Failed to serialize encrypted entry: {}", e))?;
                
                writer.write_all(encrypted_json.as_bytes()).await
                    .map_err(|e| anyhow!("Failed to write encrypted audit log: {}", e))?;
            } else {
                // Write plaintext (not recommended for production)
                writer.write_all(json_data.as_bytes()).await
                    .map_err(|e| anyhow!("Failed to write audit log: {}", e))?;
            }
            
            writer.write_all(b"\n").await
                .map_err(|e| anyhow!("Failed to write newline: {}", e))?;
        }
        
        writer.flush().await
            .map_err(|e| anyhow!("Failed to flush audit log: {}", e))?;
        
        Ok(())
    }
    
    /// Generate compliance report
    pub async fn generate_compliance_report(&self, 
        start_date: DateTime<Utc>, 
        end_date: DateTime<Utc>
    ) -> Result<ComplianceReport> {
        let metrics = self.metrics.read().await.clone();
        
        Ok(ComplianceReport {
            reporting_period: ReportingPeriod {
                start: start_date,
                end: end_date,
            },
            summary: ReportSummary {
                total_events: metrics.total_events,
                high_risk_events: metrics.high_risk_events,
                security_violations: metrics.security_violations,
                failed_authentications: metrics.failed_authentications,
                data_access_events: metrics.data_access_events,
                financial_transactions: metrics.financial_transactions,
                privacy_operations: metrics.privacy_operations,
            },
            compliance_scores: ComplianceScores {
                sox_score: self.calculate_sox_compliance_score(&metrics),
                gdpr_score: self.calculate_gdpr_compliance_score(&metrics),
                overall_score: self.calculate_overall_compliance_score(&metrics),
            },
            recommendations: self.generate_recommendations(&metrics),
            risk_assessment: self.assess_risk_trends(&metrics),
            alerts_summary: AlertsSummary {
                critical_alerts: metrics.high_risk_events,
                resolved_alerts: 0, // Placeholder
                pending_alerts: 0,  // Placeholder
            },
        })
    }
    
    fn calculate_sox_compliance_score(&self, metrics: &ComplianceMetrics) -> f64 {
        // Simplified SOX compliance scoring
        let mut score = 100.0;
        
        // Deduct points for security violations
        score -= (metrics.security_violations as f64) * 2.0;
        
        // Deduct points for failed authentications
        if metrics.failed_authentications > 100 {
            score -= 10.0;
        }
        
        // Ensure minimum score
        score.max(0.0)
    }
    
    fn calculate_gdpr_compliance_score(&self, metrics: &ComplianceMetrics) -> f64 {
        // Simplified GDPR compliance scoring
        let mut score = 100.0;
        
        // GDPR focuses on privacy operations and data protection
        if metrics.privacy_operations == 0 && metrics.data_access_events > 1000 {
            score -= 20.0; // No privacy operations but lots of data access
        }
        
        score.max(0.0)
    }
    
    fn calculate_overall_compliance_score(&self, metrics: &ComplianceMetrics) -> f64 {
        let sox_score = self.calculate_sox_compliance_score(metrics);
        let gdpr_score = self.calculate_gdpr_compliance_score(metrics);
        (sox_score + gdpr_score) / 2.0
    }
    
    fn generate_recommendations(&self, _metrics: &ComplianceMetrics) -> Vec<String> {
        vec![
            "Maintain current security monitoring levels".to_string(),
            "Consider implementing additional MFA for high-risk operations".to_string(),
            "Regular review of access permissions recommended".to_string(),
            "Ensure timely response to security alerts".to_string(),
        ]
    }
    
    fn assess_risk_trends(&self, metrics: &ComplianceMetrics) -> RiskAssessment {
        let risk_level = if metrics.high_risk_events > 100 {
            RiskLevel::High
        } else if metrics.high_risk_events > 50 {
            RiskLevel::Medium
        } else {
            RiskLevel::Low
        };
        
        RiskAssessment {
            current_risk_level: risk_level,
            trend: RiskTrend::Stable, // Placeholder
            key_indicators: vec![
                format!("High-risk events: {}", metrics.high_risk_events),
                format!("Failed authentications: {}", metrics.failed_authentications),
                format!("Security violations: {}", metrics.security_violations),
            ],
        }
    }
    
    /// Get current compliance metrics
    pub async fn get_metrics(&self) -> ComplianceMetrics {
        self.metrics.read().await.clone()
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct EncryptedLogEntry {
    data: Vec<u8>,
    tag: Vec<u8>,
    timestamp: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub reporting_period: ReportingPeriod,
    pub summary: ReportSummary,
    pub compliance_scores: ComplianceScores,
    pub recommendations: Vec<String>,
    pub risk_assessment: RiskAssessment,
    pub alerts_summary: AlertsSummary,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportingPeriod {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ReportSummary {
    pub total_events: u64,
    pub high_risk_events: u64,
    pub security_violations: u64,
    pub failed_authentications: u64,
    pub data_access_events: u64,
    pub financial_transactions: u64,
    pub privacy_operations: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ComplianceScores {
    pub sox_score: f64,
    pub gdpr_score: f64,
    pub overall_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RiskAssessment {
    pub current_risk_level: RiskLevel,
    pub trend: RiskTrend,
    pub key_indicators: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum RiskTrend {
    Decreasing,
    Stable,
    Increasing,
    Volatile,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertsSummary {
    pub critical_alerts: u64,
    pub resolved_alerts: u64,
    pub pending_alerts: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;
    
    #[tokio::test]
    async fn test_audit_logger_creation() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AuditConfig {
            log_file_path: temp_file.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        
        let logger = ComplianceAuditLogger::new(config).await;
        assert!(logger.is_ok());
    }
    
    #[tokio::test]
    async fn test_audit_event_logging() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AuditConfig {
            log_file_path: temp_file.path().to_string_lossy().to_string(),
            buffer_size: 1,
            ..Default::default()
        };
        
        let mut logger = ComplianceAuditLogger::new(config).await.unwrap();
        
        let event = AuditEvent {
            event_id: "test-001".to_string(),
            timestamp: Utc::now(),
            event_type: AuditEventType::Authentication,
            user_id: Some("test_user".to_string()),
            session_id: Some("session_123".to_string()),
            ip_address: Some("127.0.0.1".parse().unwrap()),
            user_agent: Some("test-client".to_string()),
            resource: "/api/login".to_string(),
            action: AuditAction::Login,
            result: AuditResult::Success,
            risk_score: 2,
            data_classification: DataClassification::Internal,
            compliance_tags: vec![ComplianceTag::SOX],
            metadata: HashMap::new(),
            request_details: None,
            response_details: None,
            error_info: None,
        };
        
        let result = logger.log_event(event).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_compliance_report_generation() {
        let temp_file = NamedTempFile::new().unwrap();
        let config = AuditConfig {
            log_file_path: temp_file.path().to_string_lossy().to_string(),
            ..Default::default()
        };
        
        let logger = ComplianceAuditLogger::new(config).await.unwrap();
        
        let start_date = Utc::now() - Duration::days(30);
        let end_date = Utc::now();
        
        let report = logger.generate_compliance_report(start_date, end_date).await;
        assert!(report.is_ok());
        
        let report = report.unwrap();
        assert!(report.compliance_scores.overall_score >= 0.0);
        assert!(report.compliance_scores.overall_score <= 100.0);
    }
}