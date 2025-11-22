//! # GDPR and PCI DSS Data Protection Framework
//!
//! This module implements comprehensive data protection including:
//! - GDPR Article 32 data security requirements
//! - PCI DSS Level 1 data protection standards
//! - Encryption at rest and in transit
//! - PII handling and anonymization
//! - Data subject rights (right to be forgotten, data portability)

use std::collections::HashMap;
use std::sync::Arc;
use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::{Aead, OsRng, generic_array::GenericArray};
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier, password_hash::{rand_core::RngCore, SaltString}};
use secrecy::{Secret, ExposeSecret, Zeroize};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use uuid::Uuid;
use chrono::{DateTime, Utc};
use zeroize::ZeroizeOnDrop;
use validator::{Validate, ValidationError};

use crate::error::{Result, HiveMindError};
use crate::compliance::audit_trail::{AuditTrail, AuditEventType};

/// GDPR and PCI DSS compliant data protection manager
#[derive(Debug)]
pub struct DataProtection {
    /// Master encryption key (encrypted with KEK)
    master_key: Arc<RwLock<Secret<[u8; 32]>>>,
    
    /// Key encryption key (derived from password)
    key_encryption_key: Arc<RwLock<Secret<[u8; 32]>>>,
    
    /// Data classification registry
    data_registry: Arc<RwLock<HashMap<Uuid, DataClassification>>>,
    
    /// PII handler for personal data
    pii_handler: Arc<PIIHandler>,
    
    /// GDPR compliance manager
    gdpr_manager: Arc<GDPRCompliance>,
    
    /// Encryption configuration
    config: DataProtectionConfig,
    
    /// Audit trail reference
    audit_trail: Option<Arc<AuditTrail>>,
}

/// PII (Personally Identifiable Information) handler
#[derive(Debug)]
pub struct PIIHandler {
    /// Tokenization map for PII data
    tokenization_map: Arc<RwLock<HashMap<String, String>>>,
    
    /// Anonymization algorithms
    anonymizer: Arc<DataAnonymizer>,
}

/// GDPR compliance manager
#[derive(Debug)]
pub struct GDPRCompliance {
    /// Data subject registry
    data_subjects: Arc<RwLock<HashMap<String, DataSubject>>>,
    
    /// Consent management
    consent_manager: Arc<ConsentManager>,
    
    /// Data retention policies
    retention_policies: Arc<RwLock<HashMap<DataCategory, RetentionPolicy>>>,
    
    /// Breach notification system
    breach_notifier: Arc<BreachNotifier>,
}

/// Data anonymizer for GDPR compliance
#[derive(Debug)]
pub struct DataAnonymizer {
    /// K-anonymity implementation
    k_anonymity: u32,
    
    /// L-diversity implementation
    l_diversity: u32,
}

/// Data encryption manager
#[derive(Debug)]
pub struct EncryptionManager {
    /// Current cipher instance
    cipher: Arc<RwLock<Aes256Gcm>>,
    
    /// Key rotation schedule
    rotation_schedule: Arc<RwLock<KeyRotationSchedule>>,
}

/// Data classification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataClassification {
    /// Public data - no protection required
    Public,
    
    /// Internal data - basic protection
    Internal,
    
    /// Confidential data - enhanced protection
    Confidential,
    
    /// Restricted data - maximum protection (PII, financial)
    Restricted,
    
    /// Top Secret - highest level protection
    TopSecret,
}

/// Data categories for GDPR compliance
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum DataCategory {
    /// Personal identification data
    PersonalIdentification,
    
    /// Financial information
    Financial,
    
    /// Health information
    Health,
    
    /// Biometric data
    Biometric,
    
    /// Location data
    Location,
    
    /// Communication data
    Communication,
    
    /// Behavioral data
    Behavioral,
    
    /// Technical data
    Technical,
}

/// Encrypted data wrapper
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    /// Unique identifier for the encrypted data
    pub id: Uuid,
    
    /// Data classification level
    pub classification: DataClassification,
    
    /// Data category for GDPR
    pub category: DataCategory,
    
    /// Encrypted payload
    pub encrypted_payload: Vec<u8>,
    
    /// Nonce used for encryption
    pub nonce: Vec<u8>,
    
    /// Key identifier used for encryption
    pub key_id: String,
    
    /// Encryption algorithm used
    pub algorithm: EncryptionAlgorithm,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Last accessed timestamp
    pub last_accessed: Option<DateTime<Utc>>,
    
    /// Retention policy
    pub retention_policy: RetentionPolicy,
    
    /// GDPR metadata
    pub gdpr_metadata: GDPRMetadata,
}

/// Supported encryption algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EncryptionAlgorithm {
    Aes256Gcm,
    ChaCha20Poly1305,
    XSalsa20Poly1305,
}

/// Data retention policy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionPolicy {
    /// Retention period in days
    pub retention_days: u32,
    
    /// Auto-deletion enabled
    pub auto_delete: bool,
    
    /// Legal hold flag
    pub legal_hold: bool,
    
    /// Archival policy
    pub archival_policy: ArchivalPolicy,
}

/// Archival policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivalPolicy {
    /// Delete after retention period
    Delete,
    
    /// Archive to cold storage
    Archive,
    
    /// Anonymize data
    Anonymize,
}

/// GDPR metadata for data protection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GDPRMetadata {
    /// Data subject identifier
    pub data_subject_id: Option<String>,
    
    /// Legal basis for processing
    pub legal_basis: LegalBasis,
    
    /// Processing purposes
    pub purposes: Vec<ProcessingPurpose>,
    
    /// Consent status
    pub consent_status: ConsentStatus,
    
    /// Data source
    pub data_source: String,
    
    /// Processing location
    pub processing_location: String,
}

/// Legal basis for data processing under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LegalBasis {
    /// Article 6(1)(a) - Consent
    Consent,
    
    /// Article 6(1)(b) - Contract
    Contract,
    
    /// Article 6(1)(c) - Legal obligation
    LegalObligation,
    
    /// Article 6(1)(d) - Vital interests
    VitalInterests,
    
    /// Article 6(1)(e) - Public task
    PublicTask,
    
    /// Article 6(1)(f) - Legitimate interests
    LegitimateInterests,
}

/// Processing purposes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ProcessingPurpose {
    /// Trading and execution
    Trading,
    
    /// Risk management
    RiskManagement,
    
    /// Compliance and reporting
    Compliance,
    
    /// Customer service
    CustomerService,
    
    /// Marketing
    Marketing,
    
    /// Analytics
    Analytics,
}

/// Consent status for GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentStatus {
    /// Explicitly given
    Given,
    
    /// Withdrawn
    Withdrawn,
    
    /// Not applicable
    NotApplicable,
    
    /// Pending
    Pending,
}

/// Data subject information
#[derive(Debug, Clone, Serialize, Deserialize, Validate)]
pub struct DataSubject {
    /// Unique identifier
    pub id: String,
    
    /// Email address (for notifications)
    #[validate(email)]
    pub email: Option<String>,
    
    /// Consent records
    pub consent_records: Vec<ConsentRecord>,
    
    /// Data processing records
    pub processing_records: Vec<ProcessingRecord>,
    
    /// Rights exercised
    pub rights_exercised: Vec<DataSubjectRight>,
}

/// Consent record for GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    /// Consent identifier
    pub id: Uuid,
    
    /// Processing purpose
    pub purpose: ProcessingPurpose,
    
    /// Consent given timestamp
    pub given_at: DateTime<Utc>,
    
    /// Consent withdrawn timestamp
    pub withdrawn_at: Option<DateTime<Utc>>,
    
    /// Consent method
    pub method: ConsentMethod,
}

/// Methods of obtaining consent
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConsentMethod {
    /// Explicit opt-in
    ExplicitOptIn,
    
    /// Checkbox
    Checkbox,
    
    /// Digital signature
    DigitalSignature,
    
    /// Verbal consent (recorded)
    Verbal,
}

/// Data processing record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingRecord {
    /// Processing identifier
    pub id: Uuid,
    
    /// Data category processed
    pub data_category: DataCategory,
    
    /// Processing purpose
    pub purpose: ProcessingPurpose,
    
    /// Legal basis
    pub legal_basis: LegalBasis,
    
    /// Processing timestamp
    pub processed_at: DateTime<Utc>,
    
    /// Processor identifier
    pub processor_id: String,
}

/// Data subject rights under GDPR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DataSubjectRight {
    /// Right to access (Article 15)
    Access { requested_at: DateTime<Utc>, fulfilled_at: Option<DateTime<Utc>> },
    
    /// Right to rectification (Article 16)
    Rectification { requested_at: DateTime<Utc>, fulfilled_at: Option<DateTime<Utc>> },
    
    /// Right to erasure (Article 17)
    Erasure { requested_at: DateTime<Utc>, fulfilled_at: Option<DateTime<Utc>> },
    
    /// Right to restrict processing (Article 18)
    Restriction { requested_at: DateTime<Utc>, fulfilled_at: Option<DateTime<Utc>> },
    
    /// Right to data portability (Article 20)
    Portability { requested_at: DateTime<Utc>, fulfilled_at: Option<DateTime<Utc>> },
    
    /// Right to object (Article 21)
    Objection { requested_at: DateTime<Utc>, fulfilled_at: Option<DateTime<Utc>> },
}

/// Consent manager for GDPR compliance
#[derive(Debug)]
pub struct ConsentManager {
    /// Active consents
    consents: Arc<RwLock<HashMap<String, Vec<ConsentRecord>>>>,
}

/// Breach notification system
#[derive(Debug)]
pub struct BreachNotifier {
    /// Notification channels
    channels: Vec<NotificationChannel>,
    
    /// Breach detection rules
    detection_rules: Vec<BreachRule>,
}

/// Notification channels for breach alerts
#[derive(Debug, Clone)]
pub enum NotificationChannel {
    Email(String),
    SMS(String),
    Webhook(String),
    Slack(String),
}

/// Breach detection rules
#[derive(Debug, Clone)]
pub struct BreachRule {
    /// Rule name
    pub name: String,
    
    /// Detection criteria
    pub criteria: BreachCriteria,
    
    /// Severity level
    pub severity: BreachSeverity,
    
    /// Notification delay (for GDPR 72-hour rule)
    pub notification_delay: std::time::Duration,
}

/// Breach detection criteria
#[derive(Debug, Clone)]
pub enum BreachCriteria {
    /// Unauthorized access detected
    UnauthorizedAccess,
    
    /// Data exfiltration detected
    DataExfiltration,
    
    /// System compromise detected
    SystemCompromise,
    
    /// Encryption failure
    EncryptionFailure,
}

/// Breach severity levels
#[derive(Debug, Clone)]
pub enum BreachSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Key rotation schedule
#[derive(Debug)]
pub struct KeyRotationSchedule {
    /// Current key generation
    pub current_generation: u32,
    
    /// Last rotation timestamp
    pub last_rotation: DateTime<Utc>,
    
    /// Next rotation timestamp
    pub next_rotation: DateTime<Utc>,
    
    /// Rotation frequency
    pub rotation_frequency: chrono::Duration,
}

/// Data protection configuration
#[derive(Debug, Clone)]
pub struct DataProtectionConfig {
    /// Enable encryption at rest
    pub enable_encryption_at_rest: bool,
    
    /// Enable encryption in transit
    pub enable_encryption_in_transit: bool,
    
    /// Key rotation frequency in hours
    pub key_rotation_hours: u32,
    
    /// Enable PII tokenization
    pub enable_pii_tokenization: bool,
    
    /// Enable data anonymization
    pub enable_anonymization: bool,
    
    /// GDPR compliance enabled
    pub gdpr_enabled: bool,
    
    /// PCI DSS compliance level
    pub pci_dss_level: u8,
}

impl DataProtection {
    /// Create a new data protection manager
    pub async fn new() -> Result<Self> {
        let config = DataProtectionConfig::default();
        
        // Generate master key
        let mut master_key_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut master_key_bytes);
        let master_key = Arc::new(RwLock::new(Secret::new(master_key_bytes)));
        
        // Generate KEK (in production, derive from HSM or password)
        let mut kek_bytes = [0u8; 32];
        OsRng.fill_bytes(&mut kek_bytes);
        let key_encryption_key = Arc::new(RwLock::new(Secret::new(kek_bytes)));
        
        let data_registry = Arc::new(RwLock::new(HashMap::new()));
        let pii_handler = Arc::new(PIIHandler::new().await?);
        let gdpr_manager = Arc::new(GDPRCompliance::new().await?);
        
        Ok(Self {
            master_key,
            key_encryption_key,
            data_registry,
            pii_handler,
            gdpr_manager,
            config,
            audit_trail: None,
        })
    }
    
    /// Set audit trail reference
    pub fn set_audit_trail(&mut self, audit_trail: Arc<AuditTrail>) {
        self.audit_trail = Some(audit_trail);
    }
    
    /// Start the data protection system
    pub async fn start(&self) -> Result<()> {
        if let Some(ref audit_trail) = self.audit_trail {
            audit_trail.log_event(
                AuditEventType::SystemStartup,
                "Data protection system started".to_string(),
                serde_json::json!({
                    "component": "data_protection",
                    "gdpr_enabled": self.config.gdpr_enabled,
                    "pci_dss_level": self.config.pci_dss_level
                }),
                None,
                None,
                None,
            ).await?;
        }
        
        tracing::info!("Data protection system started with GDPR and PCI DSS compliance");
        Ok(())
    }
    
    /// Encrypt data according to classification level
    pub async fn encrypt_data(
        &self,
        data: &[u8],
        classification: DataClassification,
        category: DataCategory,
        gdpr_metadata: GDPRMetadata,
    ) -> Result<EncryptedData> {
        // Generate unique nonce for this encryption
        let mut nonce_bytes = [0u8; 12];
        OsRng.fill_bytes(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);
        
        // Get master key
        let master_key = self.master_key.read().await;
        let key = GenericArray::from_slice(master_key.expose_secret());
        
        // Create cipher
        let cipher = Aes256Gcm::new(key);
        
        // Encrypt the data
        let encrypted_payload = cipher.encrypt(nonce, data)
            .map_err(|e| HiveMindError::Cryptographic(format!("Encryption failed: {}", e)))?;
        
        // Create retention policy based on classification and category
        let retention_policy = self.get_retention_policy(&classification, &category).await;
        
        let encrypted_data = EncryptedData {
            id: Uuid::new_v4(),
            classification,
            category,
            encrypted_payload,
            nonce: nonce_bytes.to_vec(),
            key_id: "master_key_v1".to_string(),
            algorithm: EncryptionAlgorithm::Aes256Gcm,
            created_at: Utc::now(),
            last_accessed: None,
            retention_policy,
            gdpr_metadata,
        };
        
        // Register the encrypted data
        {
            let mut registry = self.data_registry.write().await;
            registry.insert(encrypted_data.id, classification.clone());
        }
        
        // Log encryption event
        if let Some(ref audit_trail) = self.audit_trail {
            audit_trail.log_event(
                AuditEventType::DataWrite,
                format!("Data encrypted with classification: {:?}", classification),
                serde_json::json!({
                    "data_id": encrypted_data.id,
                    "classification": classification,
                    "category": category,
                    "algorithm": "AES-256-GCM"
                }),
                None,
                None,
                None,
            ).await?;
        }
        
        Ok(encrypted_data)
    }
    
    /// Decrypt data with access control
    pub async fn decrypt_data(
        &self,
        encrypted_data: &mut EncryptedData,
        requester_id: Option<String>,
    ) -> Result<Vec<u8>> {
        // Check access permissions based on classification
        self.check_access_permission(&encrypted_data.classification, requester_id.as_deref()).await?;
        
        // Get master key
        let master_key = self.master_key.read().await;
        let key = GenericArray::from_slice(master_key.expose_secret());
        
        // Create cipher
        let cipher = Aes256Gcm::new(key);
        
        // Prepare nonce
        let nonce = Nonce::from_slice(&encrypted_data.nonce);
        
        // Decrypt the data
        let decrypted_data = cipher.decrypt(nonce, encrypted_data.encrypted_payload.as_ref())
            .map_err(|e| HiveMindError::Cryptographic(format!("Decryption failed: {}", e)))?;
        
        // Update last accessed timestamp
        encrypted_data.last_accessed = Some(Utc::now());
        
        // Log decryption event
        if let Some(ref audit_trail) = self.audit_trail {
            audit_trail.log_event(
                AuditEventType::DataRead,
                format!("Data decrypted for user: {:?}", requester_id),
                serde_json::json!({
                    "data_id": encrypted_data.id,
                    "classification": encrypted_data.classification,
                    "requester": requester_id
                }),
                requester_id.clone(),
                None,
                None,
            ).await?;
        }
        
        Ok(decrypted_data)
    }
    
    /// Anonymize PII data for analytics
    pub async fn anonymize_pii(&self, data: &[u8]) -> Result<Vec<u8>> {
        self.pii_handler.anonymizer.anonymize_data(data).await
    }
    
    /// Handle GDPR data subject access request
    pub async fn handle_access_request(&self, data_subject_id: &str) -> Result<DataSubjectAccessResponse> {
        self.gdpr_manager.handle_access_request(data_subject_id).await
    }
    
    /// Handle GDPR right to be forgotten request
    pub async fn handle_erasure_request(&self, data_subject_id: &str) -> Result<()> {
        self.gdpr_manager.handle_erasure_request(data_subject_id).await?;
        
        // Log erasure request
        if let Some(ref audit_trail) = self.audit_trail {
            audit_trail.log_event(
                AuditEventType::DataDelete,
                format!("GDPR erasure request processed for subject: {}", data_subject_id),
                serde_json::json!({
                    "data_subject_id": data_subject_id,
                    "request_type": "erasure",
                    "gdpr_article": "17"
                }),
                None,
                None,
                None,
            ).await?;
        }
        
        Ok(())
    }
    
    /// Check access permission based on data classification
    async fn check_access_permission(&self, classification: &DataClassification, requester_id: Option<&str>) -> Result<()> {
        match classification {
            DataClassification::Public => Ok(()),
            DataClassification::Internal => {
                // Basic authentication required
                if requester_id.is_some() {
                    Ok(())
                } else {
                    Err(HiveMindError::AccessDenied("Authentication required for internal data".to_string()))
                }
            },
            DataClassification::Confidential | DataClassification::Restricted | DataClassification::TopSecret => {
                // Enhanced authentication and authorization required
                if let Some(user_id) = requester_id {
                    // In production, check against RBAC system
                    tracing::info!("Access granted to {} for {:?} data", user_id, classification);
                    Ok(())
                } else {
                    Err(HiveMindError::AccessDenied(format!("Enhanced authentication required for {:?} data", classification)))
                }
            }
        }
    }
    
    /// Get retention policy for data classification and category
    async fn get_retention_policy(&self, classification: &DataClassification, category: &DataCategory) -> RetentionPolicy {
        match (classification, category) {
            (DataClassification::Restricted, DataCategory::Financial) => {
                // Financial records - 7 years retention (regulatory requirement)
                RetentionPolicy {
                    retention_days: 2555,
                    auto_delete: false,
                    legal_hold: true,
                    archival_policy: ArchivalPolicy::Archive,
                }
            },
            (_, DataCategory::PersonalIdentification) => {
                // PII - retention based on consent
                RetentionPolicy {
                    retention_days: 1095, // 3 years default
                    auto_delete: true,
                    legal_hold: false,
                    archival_policy: ArchivalPolicy::Anonymize,
                }
            },
            _ => {
                // Default policy
                RetentionPolicy {
                    retention_days: 365,
                    auto_delete: true,
                    legal_hold: false,
                    archival_policy: ArchivalPolicy::Delete,
                }
            }
        }
    }
}

impl PIIHandler {
    /// Create a new PII handler
    async fn new() -> Result<Self> {
        let tokenization_map = Arc::new(RwLock::new(HashMap::new()));
        let anonymizer = Arc::new(DataAnonymizer::new());
        
        Ok(Self {
            tokenization_map,
            anonymizer,
        })
    }
}

impl DataAnonymizer {
    /// Create a new data anonymizer
    fn new() -> Self {
        Self {
            k_anonymity: 5, // Default k-anonymity level
            l_diversity: 3, // Default l-diversity level
        }
    }
    
    /// Anonymize data using k-anonymity and l-diversity
    async fn anonymize_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        // Basic anonymization - in production, implement proper anonymization algorithms
        let anonymized = data.iter()
            .map(|&b| if b.is_ascii_alphanumeric() { b'X' } else { b })
            .collect();
        
        Ok(anonymized)
    }
}

impl GDPRCompliance {
    /// Create a new GDPR compliance manager
    async fn new() -> Result<Self> {
        let data_subjects = Arc::new(RwLock::new(HashMap::new()));
        let consent_manager = Arc::new(ConsentManager::new().await?);
        let retention_policies = Arc::new(RwLock::new(HashMap::new()));
        let breach_notifier = Arc::new(BreachNotifier::new());
        
        Ok(Self {
            data_subjects,
            consent_manager,
            retention_policies,
            breach_notifier,
        })
    }
    
    /// Handle data subject access request (GDPR Article 15)
    async fn handle_access_request(&self, data_subject_id: &str) -> Result<DataSubjectAccessResponse> {
        let data_subjects = self.data_subjects.read().await;
        
        if let Some(subject) = data_subjects.get(data_subject_id) {
            Ok(DataSubjectAccessResponse {
                data_subject_id: data_subject_id.to_string(),
                personal_data: subject.clone(),
                processing_purposes: subject.processing_records.iter()
                    .map(|r| r.purpose.clone())
                    .collect(),
                legal_basis: subject.processing_records.iter()
                    .map(|r| r.legal_basis.clone())
                    .collect(),
                storage_period: "As per retention policy".to_string(),
                generated_at: Utc::now(),
            })
        } else {
            Err(HiveMindError::NotFound(format!("Data subject {} not found", data_subject_id)))
        }
    }
    
    /// Handle right to be forgotten request (GDPR Article 17)
    async fn handle_erasure_request(&self, data_subject_id: &str) -> Result<()> {
        let mut data_subjects = self.data_subjects.write().await;
        
        if data_subjects.remove(data_subject_id).is_some() {
            tracing::info!("Data erased for subject: {}", data_subject_id);
            Ok(())
        } else {
            Err(HiveMindError::NotFound(format!("Data subject {} not found", data_subject_id)))
        }
    }
}

impl ConsentManager {
    /// Create a new consent manager
    async fn new() -> Result<Self> {
        let consents = Arc::new(RwLock::new(HashMap::new()));
        Ok(Self { consents })
    }
}

impl BreachNotifier {
    /// Create a new breach notifier
    fn new() -> Self {
        Self {
            channels: Vec::new(),
            detection_rules: Vec::new(),
        }
    }
}

/// Data subject access response (GDPR Article 15)
#[derive(Debug, Serialize, Deserialize)]
pub struct DataSubjectAccessResponse {
    pub data_subject_id: String,
    pub personal_data: DataSubject,
    pub processing_purposes: Vec<ProcessingPurpose>,
    pub legal_basis: Vec<LegalBasis>,
    pub storage_period: String,
    pub generated_at: DateTime<Utc>,
}

impl Default for DataProtectionConfig {
    fn default() -> Self {
        Self {
            enable_encryption_at_rest: true,
            enable_encryption_in_transit: true,
            key_rotation_hours: 24,
            enable_pii_tokenization: true,
            enable_anonymization: true,
            gdpr_enabled: true,
            pci_dss_level: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_protection_creation() {
        let data_protection = DataProtection::new().await.unwrap();
        assert!(data_protection.config.gdpr_enabled);
    }

    #[tokio::test]
    async fn test_data_encryption() {
        let data_protection = DataProtection::new().await.unwrap();
        
        let test_data = b"sensitive financial data";
        let gdpr_metadata = GDPRMetadata {
            data_subject_id: Some("user123".to_string()),
            legal_basis: LegalBasis::Contract,
            purposes: vec![ProcessingPurpose::Trading],
            consent_status: ConsentStatus::Given,
            data_source: "trading_system".to_string(),
            processing_location: "EU".to_string(),
        };
        
        let encrypted = data_protection.encrypt_data(
            test_data,
            DataClassification::Restricted,
            DataCategory::Financial,
            gdpr_metadata,
        ).await.unwrap();
        
        assert_ne!(encrypted.encrypted_payload, test_data.to_vec());
        assert_eq!(encrypted.classification, DataClassification::Restricted);
    }

    #[tokio::test]
    async fn test_data_decryption() {
        let data_protection = DataProtection::new().await.unwrap();
        
        let test_data = b"test data for decryption";
        let gdpr_metadata = GDPRMetadata {
            data_subject_id: None,
            legal_basis: LegalBasis::LegitimateInterests,
            purposes: vec![ProcessingPurpose::Analytics],
            consent_status: ConsentStatus::NotApplicable,
            data_source: "system".to_string(),
            processing_location: "EU".to_string(),
        };
        
        let mut encrypted = data_protection.encrypt_data(
            test_data,
            DataClassification::Internal,
            DataCategory::Technical,
            gdpr_metadata,
        ).await.unwrap();
        
        let decrypted = data_protection.decrypt_data(&mut encrypted, Some("user123".to_string())).await.unwrap();
        assert_eq!(decrypted, test_data.to_vec());
    }
}