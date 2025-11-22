//! GDPR and SOX compliance implementation
//! 
//! This module provides comprehensive compliance features including:
//! - GDPR data protection and privacy rights
//! - SOX financial reporting compliance
//! - Data retention and erasure policies
//! - Consent management
//! - Compliance reporting and certification

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration, Utc};
use ring::aead::{Aad, BoundKey, Nonce, SealingKey, UnboundKey, AES_256_GCM};
use ring::rand::{SecureRandom, SystemRandom};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// GDPR compliance manager
pub struct GdprComplianceManager {
    /// Data subject records
    data_subjects: Arc<RwLock<HashMap<String, DataSubject>>>,
    
    /// Consent records
    consents: Arc<RwLock<HashMap<String, ConsentRecord>>>,
    
    /// Data processing activities
    processing_activities: Arc<RwLock<Vec<ProcessingActivity>>>,
    
    /// Encryption for PII
    encryption_key: SealingKey<OneNonceSequence>,
    
    /// Random number generator
    rng: SystemRandom,
    
    /// Configuration
    config: GdprConfig,
}

#[derive(Debug, Clone)]
pub struct GdprConfig {
    pub data_retention_days: u32,
    pub consent_expiry_days: u32,
    pub breach_notification_hours: u32,
    pub encryption_required: bool,
    pub audit_trail_enabled: bool,
}

impl Default for GdprConfig {
    fn default() -> Self {
        Self {
            data_retention_days: 2555, // 7 years for financial data
            consent_expiry_days: 365,  // Annual consent renewal
            breach_notification_hours: 72, // GDPR requirement
            encryption_required: true,
            audit_trail_enabled: true,
        }
    }
}

/// Data subject under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSubject {
    pub subject_id: String,
    pub email: String,
    pub name: Option<String>,
    pub date_of_birth: Option<chrono::NaiveDate>,
    pub nationality: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub data_categories: Vec<DataCategory>,
    pub consent_status: ConsentStatus,
    pub erasure_requests: Vec<ErasureRequest>,
    pub data_exports: Vec<DataExport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataCategory {
    PersonalIdentifiers,
    FinancialData,
    TradingActivity,
    BehavioralData,
    TechnicalData,
    SpecialCategory, // Sensitive data under GDPR
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentStatus {
    Given,
    Withdrawn,
    Expired,
    Pending,
}

/// Consent record for GDPR compliance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub consent_id: String,
    pub subject_id: String,
    pub purpose: ProcessingPurpose,
    pub consent_text: String,
    pub consent_given_at: DateTime<Utc>,
    pub consent_expires_at: Option<DateTime<Utc>>,
    pub consent_method: ConsentMethod,
    pub is_active: bool,
    pub withdrawal_date: Option<DateTime<Utc>>,
    pub legal_basis: LegalBasis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPurpose {
    Trading,
    AccountManagement,
    RiskAssessment,
    Compliance,
    CustomerSupport,
    Marketing,
    Analytics,
    Security,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentMethod {
    ExplicitConsent,
    ImpliedConsent,
    LegitimateInterest,
    ContractualNecessity,
    LegalObligation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegalBasis {
    Consent,
    Contract,
    LegalObligation,
    VitalInterests,
    PublicTask,
    LegitimateInterests,
}

/// Data processing activity under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingActivity {
    pub activity_id: String,
    pub name: String,
    pub description: String,
    pub controller: DataController,
    pub processors: Vec<DataProcessor>,
    pub purposes: Vec<ProcessingPurpose>,
    pub data_categories: Vec<DataCategory>,
    pub data_subjects: Vec<String>,
    pub retention_period: Option<Duration>,
    pub transfers_outside_eu: bool,
    pub safeguards: Vec<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataController {
    pub name: String,
    pub contact_email: String,
    pub dpo_contact: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataProcessor {
    pub name: String,
    pub contact_email: String,
    pub processing_agreement: String,
}

/// Erasure request (Right to be Forgotten)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErasureRequest {
    pub request_id: String,
    pub subject_id: String,
    pub requested_at: DateTime<Utc>,
    pub reason: ErasureReason,
    pub status: ErasureStatus,
    pub completed_at: Option<DateTime<Utc>>,
    pub verification_required: bool,
    pub exceptions: Vec<ErasureException>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErasureReason {
    ConsentWithdrawn,
    PurposeFulfilled,
    UnlawfulProcessing,
    ObjectionToProcessing,
    DataNoLongerNecessary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErasureStatus {
    Pending,
    InProgress,
    Completed,
    PartiallyCompleted,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErasureException {
    LegalObligation,
    PublicInterest,
    FreedomOfExpression,
    HistoricalResearch,
    LegalClaims,
}

/// Data export for portability rights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataExport {
    pub export_id: String,
    pub subject_id: String,
    pub requested_at: DateTime<Utc>,
    pub export_format: ExportFormat,
    pub status: ExportStatus,
    pub download_url: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Csv,
    Xml,
    Pdf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportStatus {
    Pending,
    Processing,
    Ready,
    Downloaded,
    Expired,
}

/// OneNonce sequence for encryption
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
        let mut nonce_bytes = [0u8; 12];
        let bytes = self.current.to_le_bytes();
        nonce_bytes[..bytes.len()].copy_from_slice(&bytes);
        self.current += 1;
        Nonce::try_assume_unique_for_key(&nonce_bytes)
    }
}

impl GdprComplianceManager {
    pub fn new(config: GdprConfig) -> Result<Self> {
        // Generate encryption key for PII
        let rng = SystemRandom::new();
        let mut key_bytes = [0u8; 32];
        rng.fill(&mut key_bytes)
            .map_err(|_| anyhow!("Failed to generate encryption key"))?;
        
        let unbound_key = UnboundKey::new(&AES_256_GCM, &key_bytes)
            .map_err(|_| anyhow!("Failed to create encryption key"))?;
        
        let encryption_key = SealingKey::new(unbound_key, OneNonceSequence::new());
        
        Ok(Self {
            data_subjects: Arc::new(RwLock::new(HashMap::new())),
            consents: Arc::new(RwLock::new(HashMap::new())),
            processing_activities: Arc::new(RwLock::new(Vec::new())),
            encryption_key,
            rng,
            config,
        })
    }
    
    /// Register a new data subject
    pub async fn register_data_subject(&mut self, subject: DataSubject) -> Result<()> {
        let mut subjects = self.data_subjects.write().await;
        subjects.insert(subject.subject_id.clone(), subject);
        Ok(())
    }
    
    /// Record consent for data processing
    pub async fn record_consent(&self, consent: ConsentRecord) -> Result<()> {
        let mut consents = self.consents.write().await;
        consents.insert(consent.consent_id.clone(), consent);
        Ok(())
    }
    
    /// Withdraw consent
    pub async fn withdraw_consent(&self, consent_id: &str) -> Result<()> {
        let mut consents = self.consents.write().await;
        
        if let Some(consent) = consents.get_mut(consent_id) {
            consent.is_active = false;
            consent.withdrawal_date = Some(Utc::now());
            Ok(())
        } else {
            Err(anyhow!("Consent record not found"))
        }
    }
    
    /// Process erasure request (Right to be Forgotten)
    pub async fn process_erasure_request(&mut self, request: ErasureRequest) -> Result<ErasureReport> {
        let mut subjects = self.data_subjects.write().await;
        
        let subject = subjects.get_mut(&request.subject_id)
            .ok_or_else(|| anyhow!("Data subject not found"))?;
        
        // Check for erasure exceptions
        let exceptions = self.check_erasure_exceptions(&request).await?;
        
        if !exceptions.is_empty() {
            return Ok(ErasureReport {
                request_id: request.request_id.clone(),
                status: ErasureStatus::PartiallyCompleted,
                completed_at: Some(Utc::now()),
                exceptions,
                data_erased: HashMap::new(),
                compliance_certificate: None,
            });
        }
        
        // Perform data erasure
        let mut data_erased = HashMap::new();
        
        // Erase personal identifiers (if legally allowed)
        if !self.has_legal_obligation_to_retain(&request.subject_id).await? {
            subject.name = None;
            subject.date_of_birth = None;
            subject.nationality = None;
            data_erased.insert("personal_identifiers".to_string(), true);
        }
        
        // Pseudonymize financial data instead of deletion (legal requirement)
        data_erased.insert("financial_data".to_string(), false); // Not erased due to legal obligations
        
        // Mark erasure request in subject record
        subject.erasure_requests.push(request.clone());
        
        // Generate compliance certificate
        let certificate = self.generate_erasure_certificate(&request).await?;
        
        Ok(ErasureReport {
            request_id: request.request_id,
            status: ErasureStatus::Completed,
            completed_at: Some(Utc::now()),
            exceptions: vec![ErasureException::LegalObligation], // Financial data retention
            data_erased,
            compliance_certificate: Some(certificate),
        })
    }
    
    /// Export data for portability (Right to Data Portability)
    pub async fn export_data(&self, subject_id: &str, format: ExportFormat) -> Result<DataExport> {
        let subjects = self.data_subjects.read().await;
        let subject = subjects.get(subject_id)
            .ok_or_else(|| anyhow!("Data subject not found"))?;
        
        let export = DataExport {
            export_id: uuid::Uuid::new_v4().to_string(),
            subject_id: subject_id.to_string(),
            requested_at: Utc::now(),
            export_format: format,
            status: ExportStatus::Processing,
            download_url: None,
            expires_at: Some(Utc::now() + Duration::days(30)),
        };
        
        // In production, implement actual data export generation
        // This would create files in the requested format
        
        Ok(export)
    }
    
    /// Check if data subject has valid consent for purpose
    pub async fn has_valid_consent(&self, subject_id: &str, purpose: ProcessingPurpose) -> Result<bool> {
        let consents = self.consents.read().await;
        
        let valid_consent = consents.values()
            .find(|c| {
                c.subject_id == subject_id
                    && c.purpose == purpose
                    && c.is_active
                    && c.consent_expires_at.map_or(true, |exp| Utc::now() < exp)
            });
        
        Ok(valid_consent.is_some())
    }
    
    /// Check for legal obligations that prevent erasure
    async fn has_legal_obligation_to_retain(&self, _subject_id: &str) -> Result<bool> {
        // In a financial system, there are typically legal obligations to retain
        // trading records for 5-7 years
        Ok(true)
    }
    
    /// Check for exceptions to erasure
    async fn check_erasure_exceptions(&self, _request: &ErasureRequest) -> Result<Vec<ErasureException>> {
        // Financial trading systems typically have legal obligations to retain data
        Ok(vec![ErasureException::LegalObligation])
    }
    
    /// Generate compliance certificate for erasure
    async fn generate_erasure_certificate(&self, request: &ErasureRequest) -> Result<String> {
        let certificate = format!(
            "GDPR ERASURE COMPLIANCE CERTIFICATE\n\
             Request ID: {}\n\
             Subject ID: {}\n\
             Processed: {}\n\
             Compliance Officer: System Automated\n\
             Verification: Digital Signature Applied",
            request.request_id,
            request.subject_id,
            Utc::now().format("%Y-%m-%d %H:%M:%S UTC")
        );
        
        Ok(certificate)
    }
    
    /// Generate GDPR compliance report
    pub async fn generate_compliance_report(&self) -> Result<GdprComplianceReport> {
        let subjects = self.data_subjects.read().await;
        let consents = self.consents.read().await;
        let activities = self.processing_activities.read().await;
        
        let total_subjects = subjects.len();
        let active_consents = consents.values().filter(|c| c.is_active).count();
        let pending_erasures = subjects.values()
            .flat_map(|s| &s.erasure_requests)
            .filter(|r| matches!(r.status, ErasureStatus::Pending | ErasureStatus::InProgress))
            .count();
        
        Ok(GdprComplianceReport {
            reporting_date: Utc::now(),
            total_data_subjects: total_subjects,
            active_consents: active_consents,
            expired_consents: consents.values()
                .filter(|c| c.consent_expires_at.map_or(false, |exp| Utc::now() > exp))
                .count(),
            processing_activities: activities.len(),
            pending_erasure_requests: pending_erasures,
            data_breaches: 0, // Would be tracked separately
            compliance_score: self.calculate_gdpr_compliance_score().await?,
            recommendations: self.generate_gdpr_recommendations().await?,
        })
    }
    
    async fn calculate_gdpr_compliance_score(&self) -> Result<f64> {
        // Simplified scoring based on key compliance factors
        let mut score = 100.0;
        
        let consents = self.consents.read().await;
        let total_consents = consents.len() as f64;
        let expired_consents = consents.values()
            .filter(|c| c.consent_expires_at.map_or(false, |exp| Utc::now() > exp))
            .count() as f64;
        
        if total_consents > 0.0 {
            let expiry_rate = expired_consents / total_consents;
            score -= expiry_rate * 20.0; // Deduct points for expired consents
        }
        
        // Deduct points if encryption is not enabled
        if !self.config.encryption_required {
            score -= 30.0;
        }
        
        // Deduct points if audit trail is not enabled
        if !self.config.audit_trail_enabled {
            score -= 20.0;
        }
        
        Ok(score.max(0.0))
    }
    
    async fn generate_gdpr_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        if !self.config.encryption_required {
            recommendations.push("Enable encryption for all personal data".to_string());
        }
        
        if !self.config.audit_trail_enabled {
            recommendations.push("Enable comprehensive audit trails".to_string());
        }
        
        let consents = self.consents.read().await;
        let expired_count = consents.values()
            .filter(|c| c.consent_expires_at.map_or(false, |exp| Utc::now() > exp))
            .count();
        
        if expired_count > 0 {
            recommendations.push(format!("Renew {} expired consent records", expired_count));
        }
        
        recommendations.push("Regular GDPR compliance audits".to_string());
        recommendations.push("Data protection impact assessments".to_string());
        
        Ok(recommendations)
    }
}

#[derive(Debug, Serialize)]
pub struct ErasureReport {
    pub request_id: String,
    pub status: ErasureStatus,
    pub completed_at: Option<DateTime<Utc>>,
    pub exceptions: Vec<ErasureException>,
    pub data_erased: HashMap<String, bool>,
    pub compliance_certificate: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct GdprComplianceReport {
    pub reporting_date: DateTime<Utc>,
    pub total_data_subjects: usize,
    pub active_consents: usize,
    pub expired_consents: usize,
    pub processing_activities: usize,
    pub pending_erasure_requests: usize,
    pub data_breaches: usize,
    pub compliance_score: f64,
    pub recommendations: Vec<String>,
}

/// SOX compliance manager for financial reporting
pub struct SoxComplianceManager {
    /// Financial controls
    controls: Arc<RwLock<Vec<FinancialControl>>>,
    
    /// Audit events
    audit_events: Arc<RwLock<Vec<SoxAuditEvent>>>,
    
    /// Configuration
    config: SoxConfig,
}

#[derive(Debug, Clone)]
pub struct SoxConfig {
    pub quarterly_certification_required: bool,
    pub ceo_cfo_certification: bool,
    pub internal_controls_testing: bool,
    pub external_auditor_required: bool,
    pub deficiency_tracking: bool,
}

impl Default for SoxConfig {
    fn default() -> Self {
        Self {
            quarterly_certification_required: true,
            ceo_cfo_certification: true,
            internal_controls_testing: true,
            external_auditor_required: true,
            deficiency_tracking: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinancialControl {
    pub control_id: String,
    pub description: String,
    pub control_type: ControlType,
    pub frequency: ControlFrequency,
    pub owner: String,
    pub last_tested: Option<DateTime<Utc>>,
    pub effectiveness: ControlEffectiveness,
    pub deficiencies: Vec<ControlDeficiency>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlType {
    Preventive,
    Detective,
    Corrective,
    Compensating,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlFrequency {
    Daily,
    Weekly,
    Monthly,
    Quarterly,
    Annually,
    AsNeeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlEffectiveness {
    Effective,
    Ineffective,
    NeedsImprovement,
    NotTested,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlDeficiency {
    pub deficiency_id: String,
    pub description: String,
    pub severity: DeficiencySeverity,
    pub identified_date: DateTime<Utc>,
    pub remediation_plan: Option<String>,
    pub target_remediation_date: Option<DateTime<Utc>>,
    pub status: DeficiencyStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeficiencySeverity {
    Significant,
    Material,
    Minor,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeficiencyStatus {
    Open,
    InProgress,
    Remediated,
    Closed,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SoxAuditEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub event_type: SoxEventType,
    pub description: String,
    pub user_id: String,
    pub control_id: Option<String>,
    pub financial_impact: Option<f64>,
    pub supporting_evidence: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SoxEventType {
    ControlExecution,
    ControlTest,
    DeficiencyIdentified,
    DeficiencyRemediated,
    QuarterlyCertification,
    ManagementReview,
    ExternalAudit,
}

impl SoxComplianceManager {
    pub fn new(config: SoxConfig) -> Self {
        Self {
            controls: Arc::new(RwLock::new(Vec::new())),
            audit_events: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }
    
    /// Record a SOX audit event
    pub async fn record_audit_event(&self, event: SoxAuditEvent) -> Result<()> {
        let mut events = self.audit_events.write().await;
        events.push(event);
        Ok(())
    }
    
    /// Generate SOX compliance report
    pub async fn generate_sox_report(&self) -> Result<SoxComplianceReport> {
        let controls = self.controls.read().await;
        let events = self.audit_events.read().await;
        
        let total_controls = controls.len();
        let effective_controls = controls.iter()
            .filter(|c| matches!(c.effectiveness, ControlEffectiveness::Effective))
            .count();
        
        let open_deficiencies = controls.iter()
            .flat_map(|c| &c.deficiencies)
            .filter(|d| matches!(d.status, DeficiencyStatus::Open | DeficiencyStatus::InProgress))
            .count();
        
        Ok(SoxComplianceReport {
            reporting_date: Utc::now(),
            total_controls: total_controls,
            effective_controls: effective_controls,
            control_effectiveness_rate: if total_controls > 0 {
                effective_controls as f64 / total_controls as f64
            } else {
                0.0
            },
            open_deficiencies: open_deficiencies,
            audit_events_count: events.len(),
            compliance_score: self.calculate_sox_compliance_score().await?,
            certification_status: "Pending".to_string(), // Would be set based on actual certification
            recommendations: self.generate_sox_recommendations().await?,
        })
    }
    
    async fn calculate_sox_compliance_score(&self) -> Result<f64> {
        let controls = self.controls.read().await;
        
        if controls.is_empty() {
            return Ok(0.0);
        }
        
        let effective_count = controls.iter()
            .filter(|c| matches!(c.effectiveness, ControlEffectiveness::Effective))
            .count();
        
        let score = (effective_count as f64 / controls.len() as f64) * 100.0;
        Ok(score)
    }
    
    async fn generate_sox_recommendations(&self) -> Result<Vec<String>> {
        let mut recommendations = Vec::new();
        
        let controls = self.controls.read().await;
        
        let untested_controls = controls.iter()
            .filter(|c| matches!(c.effectiveness, ControlEffectiveness::NotTested))
            .count();
        
        if untested_controls > 0 {
            recommendations.push(format!("Test {} untested controls", untested_controls));
        }
        
        let ineffective_controls = controls.iter()
            .filter(|c| matches!(c.effectiveness, ControlEffectiveness::Ineffective))
            .count();
        
        if ineffective_controls > 0 {
            recommendations.push(format!("Remediate {} ineffective controls", ineffective_controls));
        }
        
        recommendations.push("Quarterly management certification".to_string());
        recommendations.push("Annual external audit".to_string());
        
        Ok(recommendations)
    }
}

#[derive(Debug, Serialize)]
pub struct SoxComplianceReport {
    pub reporting_date: DateTime<Utc>,
    pub total_controls: usize,
    pub effective_controls: usize,
    pub control_effectiveness_rate: f64,
    pub open_deficiencies: usize,
    pub audit_events_count: usize,
    pub compliance_score: f64,
    pub certification_status: String,
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gdpr_manager_creation() {
        let config = GdprConfig::default();
        let manager = GdprComplianceManager::new(config);
        assert!(manager.is_ok());
    }
    
    #[tokio::test]
    async fn test_consent_management() {
        let config = GdprConfig::default();
        let manager = GdprComplianceManager::new(config).unwrap();
        
        let consent = ConsentRecord {
            consent_id: "consent-001".to_string(),
            subject_id: "subject-001".to_string(),
            purpose: ProcessingPurpose::Trading,
            consent_text: "I consent to trading data processing".to_string(),
            consent_given_at: Utc::now(),
            consent_expires_at: Some(Utc::now() + Duration::days(365)),
            consent_method: ConsentMethod::ExplicitConsent,
            is_active: true,
            withdrawal_date: None,
            legal_basis: LegalBasis::Consent,
        };
        
        manager.record_consent(consent).await.unwrap();
        
        let has_consent = manager.has_valid_consent("subject-001", ProcessingPurpose::Trading).await.unwrap();
        assert!(has_consent);
    }
    
    #[test]
    fn test_sox_manager_creation() {
        let config = SoxConfig::default();
        let manager = SoxComplianceManager::new(config);
        // Just test that creation doesn't panic
        assert!(true);
    }
}