//! Enterprise Security Module for Cerebellar Norse Trading System
//! 
//! Provides comprehensive security controls, encryption, access management,
//! audit logging, and regulatory compliance for high-frequency trading systems.

use std::collections::HashMap;
use std::sync::{Arc, RwLock, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};
use anyhow::{Result, anyhow};
use serde::{Serialize, Deserialize};
use tracing::{debug, info, warn, error};
use sha2::{Sha256, Digest};
use rand::{Rng, thread_rng};
use aes_gcm::{Aes256Gcm, Key, Nonce, aead::{Aead, NewAead}};

/// Security configuration for trading systems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    /// Enable encryption for all data
    pub encryption_enabled: bool,
    /// Encryption key rotation interval (hours)
    pub key_rotation_interval: u64,
    /// Audit logging level
    pub audit_level: AuditLevel,
    /// Access control mode
    pub access_control_mode: AccessControlMode,
    /// Maximum failed authentication attempts
    pub max_auth_failures: u32,
    /// Session timeout (minutes)
    pub session_timeout: u64,
    /// Enable regulatory compliance monitoring
    pub compliance_monitoring: bool,
    /// Data retention period (days)
    pub data_retention_days: u32,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            encryption_enabled: true,
            key_rotation_interval: 24, // 24 hours
            audit_level: AuditLevel::Full,
            access_control_mode: AccessControlMode::RoleBased,
            max_auth_failures: 3,
            session_timeout: 30, // 30 minutes
            compliance_monitoring: true,
            data_retention_days: 2555, // 7 years for financial regulations
        }
    }
}

/// Audit logging levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AuditLevel {
    /// Minimal logging
    Minimal,
    /// Standard compliance logging
    Standard,
    /// Full detailed logging
    Full,
    /// Debug level (development only)
    Debug,
}

/// Access control modes
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AccessControlMode {
    /// Role-based access control
    RoleBased,
    /// Attribute-based access control
    AttributeBased,
    /// Zero-trust security model
    ZeroTrust,
}

/// Enterprise security manager
pub struct SecurityManager {
    config: SecurityConfig,
    encryption_manager: Arc<EncryptionManager>,
    access_control: Arc<AccessControlManager>,
    audit_logger: Arc<AuditLogger>,
    compliance_monitor: Arc<ComplianceMonitor>,
    threat_detector: Arc<ThreatDetector>,
}

impl SecurityManager {
    /// Create new security manager
    pub fn new(config: SecurityConfig) -> Result<Self> {
        let encryption_manager = Arc::new(EncryptionManager::new(&config)?);
        let access_control = Arc::new(AccessControlManager::new(&config)?);
        let audit_logger = Arc::new(AuditLogger::new(&config)?);
        let compliance_monitor = Arc::new(ComplianceMonitor::new(&config)?);
        let threat_detector = Arc::new(ThreatDetector::new(&config)?);

        info!("Security manager initialized with {} configuration", 
              if config.encryption_enabled { "encrypted" } else { "unencrypted" });

        Ok(Self {
            config,
            encryption_manager,
            access_control,
            audit_logger,
            compliance_monitor,
            threat_detector,
        })
    }

    /// Validate system security before trading operations
    pub fn validate_system_security(&self) -> Result<SecurityValidation> {
        let mut validation = SecurityValidation::new();

        // Check encryption status
        if !self.encryption_manager.is_encryption_active() {
            validation.add_warning("Encryption is not active".to_string());
        }

        // Validate access controls
        let access_status = self.access_control.validate_access_controls()?;
        validation.merge_access_validation(access_status);

        // Check compliance status
        let compliance_status = self.compliance_monitor.check_compliance()?;
        validation.merge_compliance_validation(compliance_status);

        // Perform threat assessment
        let threat_level = self.threat_detector.assess_current_threat_level()?;
        validation.threat_level = threat_level;

        // Log security validation
        self.audit_logger.log_security_event(SecurityEvent::SystemValidation {
            timestamp: SystemTime::now(),
            validation_result: validation.clone(),
        })?;

        Ok(validation)
    }

    /// Secure neural network model data
    pub fn secure_model_data(&self, model_data: &[u8]) -> Result<SecureModelData> {
        // Encrypt model parameters
        let encrypted_data = self.encryption_manager.encrypt_data(model_data)?;
        
        // Generate integrity hash
        let mut hasher = Sha256::new();
        hasher.update(model_data);
        let integrity_hash = hasher.finalize().to_vec();

        // Log data security operation
        self.audit_logger.log_security_event(SecurityEvent::DataSecured {
            timestamp: SystemTime::now(),
            data_type: "neural_model".to_string(),
            data_size: model_data.len(),
            encryption_method: "AES-256-GCM".to_string(),
        })?;

        Ok(SecureModelData {
            encrypted_data,
            integrity_hash,
            encryption_metadata: EncryptionMetadata {
                algorithm: "AES-256-GCM".to_string(),
                key_id: self.encryption_manager.get_current_key_id(),
                timestamp: SystemTime::now(),
            },
        })
    }

    /// Validate trading decision for compliance
    pub fn validate_trading_decision(&self, decision: &TradingDecision) -> Result<ComplianceValidation> {
        let validation = self.compliance_monitor.validate_trading_decision(decision)?;

        // Log trading decision validation
        self.audit_logger.log_security_event(SecurityEvent::TradingDecisionValidated {
            timestamp: SystemTime::now(),
            decision_id: decision.id.clone(),
            validation_result: validation.clone(),
        })?;

        Ok(validation)
    }

    /// Authenticate user access
    pub fn authenticate_user(&self, credentials: &UserCredentials) -> Result<AuthenticationResult> {
        let auth_result = self.access_control.authenticate_user(credentials)?;

        // Log authentication attempt
        self.audit_logger.log_security_event(SecurityEvent::AuthenticationAttempt {
            timestamp: SystemTime::now(),
            user_id: credentials.user_id.clone(),
            success: auth_result.is_authenticated,
            method: auth_result.authentication_method.clone(),
        })?;

        Ok(auth_result)
    }

    /// Get security metrics for monitoring
    pub fn get_security_metrics(&self) -> Result<SecurityMetrics> {
        Ok(SecurityMetrics {
            encryption_status: self.encryption_manager.get_status(),
            access_control_stats: self.access_control.get_statistics(),
            audit_log_count: self.audit_logger.get_log_count(),
            compliance_status: self.compliance_monitor.get_status(),
            threat_level: self.threat_detector.get_current_threat_level(),
            system_uptime: SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
        })
    }
}

/// Encryption manager for sensitive data
pub struct EncryptionManager {
    current_key: Arc<RwLock<Key<Aes256Gcm>>>,
    key_rotation_timer: Arc<Mutex<SystemTime>>,
    key_history: Arc<RwLock<Vec<EncryptionKey>>>,
    config: SecurityConfig,
}

impl EncryptionManager {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        let key_data = Self::generate_encryption_key();
        let current_key = Arc::new(RwLock::new(*Key::<Aes256Gcm>::from_slice(&key_data)));
        
        let mut key_history = Vec::new();
        key_history.push(EncryptionKey {
            id: "key_001".to_string(),
            created_at: SystemTime::now(),
            expires_at: SystemTime::now() + std::time::Duration::from_secs(config.key_rotation_interval * 3600),
            active: true,
        });

        Ok(Self {
            current_key,
            key_rotation_timer: Arc::new(Mutex::new(SystemTime::now())),
            key_history: Arc::new(RwLock::new(key_history)),
            config: config.clone(),
        })
    }

    pub fn encrypt_data(&self, data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.encryption_enabled {
            return Ok(data.to_vec());
        }

        let key = self.current_key.read().unwrap();
        let cipher = Aes256Gcm::new(&key);
        
        // Generate random nonce
        let mut nonce_bytes = [0u8; 12];
        thread_rng().fill(&mut nonce_bytes);
        let nonce = Nonce::from_slice(&nonce_bytes);

        let mut encrypted = cipher.encrypt(nonce, data)
            .map_err(|e| anyhow!("Encryption failed: {}", e))?;
        
        // Prepend nonce to encrypted data
        let mut result = nonce_bytes.to_vec();
        result.append(&mut encrypted);
        
        Ok(result)
    }

    pub fn decrypt_data(&self, encrypted_data: &[u8]) -> Result<Vec<u8>> {
        if !self.config.encryption_enabled {
            return Ok(encrypted_data.to_vec());
        }

        if encrypted_data.len() < 12 {
            return Err(anyhow!("Invalid encrypted data format"));
        }

        // Extract nonce and encrypted content
        let (nonce_bytes, ciphertext) = encrypted_data.split_at(12);
        let nonce = Nonce::from_slice(nonce_bytes);

        let key = self.current_key.read().unwrap();
        let cipher = Aes256Gcm::new(&key);
        
        let decrypted = cipher.decrypt(nonce, ciphertext)
            .map_err(|e| anyhow!("Decryption failed: {}", e))?;
        
        Ok(decrypted)
    }

    pub fn is_encryption_active(&self) -> bool {
        self.config.encryption_enabled
    }

    pub fn get_current_key_id(&self) -> String {
        let history = self.key_history.read().unwrap();
        history.iter()
            .find(|k| k.active)
            .map(|k| k.id.clone())
            .unwrap_or_else(|| "unknown".to_string())
    }

    pub fn get_status(&self) -> EncryptionStatus {
        EncryptionStatus {
            active: self.config.encryption_enabled,
            current_key_id: self.get_current_key_id(),
            key_rotation_due: self.is_key_rotation_due(),
            keys_in_history: self.key_history.read().unwrap().len(),
        }
    }

    fn generate_encryption_key() -> [u8; 32] {
        let mut key = [0u8; 32];
        thread_rng().fill(&mut key);
        key
    }

    fn is_key_rotation_due(&self) -> bool {
        let timer = self.key_rotation_timer.lock().unwrap();
        let rotation_interval = std::time::Duration::from_secs(self.config.key_rotation_interval * 3600);
        SystemTime::now().duration_since(*timer).unwrap_or_default() > rotation_interval
    }
}

/// Access control manager
pub struct AccessControlManager {
    user_database: Arc<RwLock<HashMap<String, UserAccount>>>,
    active_sessions: Arc<RwLock<HashMap<String, UserSession>>>,
    role_definitions: Arc<RwLock<HashMap<String, Role>>>,
    config: SecurityConfig,
}

impl AccessControlManager {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        let mut role_definitions = HashMap::new();
        
        // Define standard trading system roles
        role_definitions.insert("admin".to_string(), Role {
            name: "Administrator".to_string(),
            permissions: vec![
                Permission::SystemAdmin,
                Permission::UserManagement,
                Permission::TradingControl,
                Permission::DataAccess,
                Permission::AuditAccess,
            ],
            restrictions: vec![],
        });

        role_definitions.insert("trader".to_string(), Role {
            name: "Trader".to_string(),
            permissions: vec![
                Permission::TradingControl,
                Permission::MarketDataAccess,
                Permission::PortfolioView,
            ],
            restrictions: vec![
                Restriction::TradingHours,
                Restriction::PositionLimits,
            ],
        });

        role_definitions.insert("risk_manager".to_string(), Role {
            name: "Risk Manager".to_string(),
            permissions: vec![
                Permission::RiskMonitoring,
                Permission::PositionOverride,
                Permission::AuditAccess,
                Permission::DataAccess,
            ],
            restrictions: vec![],
        });

        role_definitions.insert("auditor".to_string(), Role {
            name: "Auditor".to_string(),
            permissions: vec![
                Permission::AuditAccess,
                Permission::DataAccess,
            ],
            restrictions: vec![
                Restriction::ReadOnly,
            ],
        });

        Ok(Self {
            user_database: Arc::new(RwLock::new(HashMap::new())),
            active_sessions: Arc::new(RwLock::new(HashMap::new())),
            role_definitions: Arc::new(RwLock::new(role_definitions)),
            config: config.clone(),
        })
    }

    pub fn authenticate_user(&self, credentials: &UserCredentials) -> Result<AuthenticationResult> {
        let users = self.user_database.read().unwrap();
        
        if let Some(user) = users.get(&credentials.user_id) {
            // Verify password (in production, use proper password hashing)
            if self.verify_password(&credentials.password, &user.password_hash) {
                // Create new session
                let session = UserSession {
                    session_id: self.generate_session_id(),
                    user_id: credentials.user_id.clone(),
                    created_at: SystemTime::now(),
                    expires_at: SystemTime::now() + 
                        std::time::Duration::from_secs(self.config.session_timeout * 60),
                    permissions: user.role_permissions.clone(),
                };

                let session_id = session.session_id.clone();
                self.active_sessions.write().unwrap().insert(session_id.clone(), session);

                Ok(AuthenticationResult {
                    is_authenticated: true,
                    session_id: Some(session_id),
                    user_role: user.role.clone(),
                    permissions: user.role_permissions.clone(),
                    authentication_method: "password".to_string(),
                })
            } else {
                Ok(AuthenticationResult {
                    is_authenticated: false,
                    session_id: None,
                    user_role: "".to_string(),
                    permissions: vec![],
                    authentication_method: "password".to_string(),
                })
            }
        } else {
            Ok(AuthenticationResult {
                is_authenticated: false,
                session_id: None,
                user_role: "".to_string(),
                permissions: vec![],
                authentication_method: "password".to_string(),
            })
        }
    }

    pub fn validate_access_controls(&self) -> Result<AccessValidation> {
        let users_count = self.user_database.read().unwrap().len();
        let active_sessions_count = self.active_sessions.read().unwrap().len();
        let roles_count = self.role_definitions.read().unwrap().len();

        Ok(AccessValidation {
            total_users: users_count,
            active_sessions: active_sessions_count,
            defined_roles: roles_count,
            access_violations: 0, // Would be calculated from audit logs
        })
    }

    pub fn get_statistics(&self) -> AccessControlStats {
        let sessions = self.active_sessions.read().unwrap();
        let expired_sessions = sessions.iter()
            .filter(|(_, session)| SystemTime::now() > session.expires_at)
            .count();

        AccessControlStats {
            total_users: self.user_database.read().unwrap().len(),
            active_sessions: sessions.len() - expired_sessions,
            expired_sessions,
            failed_login_attempts: 0, // Would be tracked separately
        }
    }

    fn verify_password(&self, provided: &str, stored_hash: &str) -> bool {
        // In production, use proper password hashing (bcrypt, scrypt, etc.)
        let mut hasher = Sha256::new();
        hasher.update(provided.as_bytes());
        let provided_hash = format!("{:x}", hasher.finalize());
        provided_hash == stored_hash
    }

    fn generate_session_id(&self) -> String {
        let mut rng = thread_rng();
        let session_bytes: [u8; 32] = rng.gen();
        hex::encode(session_bytes)
    }
}

/// Audit logging system
pub struct AuditLogger {
    log_buffer: Arc<Mutex<Vec<AuditLogEntry>>>,
    config: SecurityConfig,
}

impl AuditLogger {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        Ok(Self {
            log_buffer: Arc::new(Mutex::new(Vec::new())),
            config: config.clone(),
        })
    }

    pub fn log_security_event(&self, event: SecurityEvent) -> Result<()> {
        let entry = AuditLogEntry {
            timestamp: SystemTime::now(),
            event_type: event.get_event_type(),
            severity: event.get_severity(),
            description: event.get_description(),
            user_id: event.get_user_id(),
            source_ip: event.get_source_ip(),
            additional_data: event.to_json()?,
        };

        let mut buffer = self.log_buffer.lock().unwrap();
        buffer.push(entry);

        // In production, would write to persistent storage
        self.write_to_persistent_storage(&buffer)?;

        Ok(())
    }

    pub fn get_log_count(&self) -> usize {
        self.log_buffer.lock().unwrap().len()
    }

    fn write_to_persistent_storage(&self, _entries: &[AuditLogEntry]) -> Result<()> {
        // Placeholder for persistent storage implementation
        // Would integrate with enterprise logging systems (ELK, Splunk, etc.)
        Ok(())
    }
}

/// Compliance monitoring system
pub struct ComplianceMonitor {
    regulation_rules: Arc<RwLock<Vec<ComplianceRule>>>,
    violation_history: Arc<RwLock<Vec<ComplianceViolation>>>,
    config: SecurityConfig,
}

impl ComplianceMonitor {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        let mut rules = Vec::new();

        // MiFID II compliance rules
        rules.push(ComplianceRule {
            id: "MIFID_001".to_string(),
            name: "Best Execution".to_string(),
            regulation: Regulation::MiFIDII,
            description: "Ensure best execution for client orders".to_string(),
            check_function: ComplianceCheck::BestExecution,
        });

        // GDPR compliance rules
        rules.push(ComplianceRule {
            id: "GDPR_001".to_string(),
            name: "Data Retention".to_string(),
            regulation: Regulation::GDPR,
            description: "Enforce data retention policies".to_string(),
            check_function: ComplianceCheck::DataRetention,
        });

        // SEC compliance rules
        rules.push(ComplianceRule {
            id: "SEC_001".to_string(),
            name: "Market Manipulation".to_string(),
            regulation: Regulation::SEC,
            description: "Detect potential market manipulation".to_string(),
            check_function: ComplianceCheck::MarketManipulation,
        });

        Ok(Self {
            regulation_rules: Arc::new(RwLock::new(rules)),
            violation_history: Arc::new(RwLock::new(Vec::new())),
            config: config.clone(),
        })
    }

    pub fn check_compliance(&self) -> Result<ComplianceStatus> {
        let rules = self.regulation_rules.read().unwrap();
        let violations = self.violation_history.read().unwrap();

        let total_rules = rules.len();
        let recent_violations = violations.iter()
            .filter(|v| {
                SystemTime::now().duration_since(v.timestamp)
                    .unwrap_or_default().as_secs() < 86400 // Last 24 hours
            })
            .count();

        Ok(ComplianceStatus {
            overall_status: if recent_violations == 0 { 
                ComplianceLevel::Compliant 
            } else { 
                ComplianceLevel::Warning 
            },
            total_rules,
            active_violations: recent_violations,
            last_audit: SystemTime::now(),
        })
    }

    pub fn validate_trading_decision(&self, decision: &TradingDecision) -> Result<ComplianceValidation> {
        let mut validation = ComplianceValidation {
            is_compliant: true,
            violations: Vec::new(),
            warnings: Vec::new(),
        };

        // Check each compliance rule
        let rules = self.regulation_rules.read().unwrap();
        for rule in rules.iter() {
            match self.check_rule_against_decision(rule, decision) {
                RuleCheckResult::Pass => continue,
                RuleCheckResult::Warning(msg) => validation.warnings.push(msg),
                RuleCheckResult::Violation(msg) => {
                    validation.is_compliant = false;
                    validation.violations.push(msg);
                }
            }
        }

        Ok(validation)
    }

    pub fn get_status(&self) -> ComplianceOverview {
        let rules = self.regulation_rules.read().unwrap();
        let violations = self.violation_history.read().unwrap();

        ComplianceOverview {
            total_rules: rules.len(),
            total_violations: violations.len(),
            regulations_covered: vec![
                Regulation::MiFIDII,
                Regulation::GDPR,
                Regulation::SEC,
            ],
            last_compliance_check: SystemTime::now(),
        }
    }

    fn check_rule_against_decision(&self, rule: &ComplianceRule, decision: &TradingDecision) -> RuleCheckResult {
        match rule.check_function {
            ComplianceCheck::BestExecution => {
                // Check if execution price is within acceptable range
                if decision.execution_price > decision.market_price * 1.001 {
                    RuleCheckResult::Violation("Execution price exceeds best execution threshold".to_string())
                } else {
                    RuleCheckResult::Pass
                }
            }
            ComplianceCheck::DataRetention => {
                // Check if decision includes proper data retention metadata
                RuleCheckResult::Pass // Placeholder
            }
            ComplianceCheck::MarketManipulation => {
                // Check for suspicious trading patterns
                if decision.volume > 1000000.0 && decision.execution_speed_ms < 1 {
                    RuleCheckResult::Warning("High volume ultra-fast execution detected".to_string())
                } else {
                    RuleCheckResult::Pass
                }
            }
        }
    }
}

/// Threat detection system
pub struct ThreatDetector {
    threat_patterns: Arc<RwLock<Vec<ThreatPattern>>>,
    detected_threats: Arc<RwLock<Vec<ThreatEvent>>>,
    config: SecurityConfig,
}

impl ThreatDetector {
    pub fn new(config: &SecurityConfig) -> Result<Self> {
        let threat_patterns = vec![
            ThreatPattern {
                id: "SQL_INJECTION".to_string(),
                name: "SQL Injection Attack".to_string(),
                severity: ThreatSeverity::High,
                pattern_regex: r"('.*(union|select|insert|update|delete|drop).*)|(--)".to_string(),
            },
            ThreatPattern {
                id: "EXCESSIVE_REQUESTS".to_string(),
                name: "DoS/DDoS Attack".to_string(),
                severity: ThreatSeverity::Medium,
                pattern_regex: r".*".to_string(), // Would be rate-based detection
            },
            ThreatPattern {
                id: "PRIVILEGE_ESCALATION".to_string(),
                name: "Privilege Escalation".to_string(),
                severity: ThreatSeverity::Critical,
                pattern_regex: r".*(sudo|admin|root|elevated).*".to_string(),
            },
        ];

        Ok(Self {
            threat_patterns: Arc::new(RwLock::new(threat_patterns)),
            detected_threats: Arc::new(RwLock::new(Vec::new())),
            config: config.clone(),
        })
    }

    pub fn assess_current_threat_level(&self) -> Result<ThreatLevel> {
        let threats = self.detected_threats.read().unwrap();
        let recent_threats = threats.iter()
            .filter(|t| {
                SystemTime::now().duration_since(t.timestamp)
                    .unwrap_or_default().as_secs() < 3600 // Last hour
            })
            .collect::<Vec<_>>();

        let critical_threats = recent_threats.iter().filter(|t| t.severity == ThreatSeverity::Critical).count();
        let high_threats = recent_threats.iter().filter(|t| t.severity == ThreatSeverity::High).count();

        if critical_threats > 0 {
            Ok(ThreatLevel::Critical)
        } else if high_threats > 2 {
            Ok(ThreatLevel::High)
        } else if !recent_threats.is_empty() {
            Ok(ThreatLevel::Medium)
        } else {
            Ok(ThreatLevel::Low)
        }
    }

    pub fn get_current_threat_level(&self) -> ThreatLevel {
        self.assess_current_threat_level().unwrap_or(ThreatLevel::Unknown)
    }
}

// Data structures and enums

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityValidation {
    pub encryption_status: bool,
    pub access_control_status: bool,
    pub compliance_status: bool,
    pub threat_level: ThreatLevel,
    pub warnings: Vec<String>,
    pub errors: Vec<String>,
}

impl SecurityValidation {
    pub fn new() -> Self {
        Self {
            encryption_status: false,
            access_control_status: false,
            compliance_status: false,
            threat_level: ThreatLevel::Unknown,
            warnings: Vec::new(),
            errors: Vec::new(),
        }
    }

    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    pub fn merge_access_validation(&mut self, access: AccessValidation) {
        self.access_control_status = access.access_violations == 0;
    }

    pub fn merge_compliance_validation(&mut self, compliance: ComplianceStatus) {
        self.compliance_status = matches!(compliance.overall_status, ComplianceLevel::Compliant);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureModelData {
    pub encrypted_data: Vec<u8>,
    pub integrity_hash: Vec<u8>,
    pub encryption_metadata: EncryptionMetadata,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    pub algorithm: String,
    pub key_id: String,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    pub id: String,
    pub symbol: String,
    pub order_type: String,
    pub volume: f64,
    pub market_price: f64,
    pub execution_price: f64,
    pub execution_speed_ms: u64,
    pub timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserCredentials {
    pub user_id: String,
    pub password: String,
    pub additional_factors: Option<HashMap<String, String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationResult {
    pub is_authenticated: bool,
    pub session_id: Option<String>,
    pub user_role: String,
    pub permissions: Vec<Permission>,
    pub authentication_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityMetrics {
    pub encryption_status: EncryptionStatus,
    pub access_control_stats: AccessControlStats,
    pub audit_log_count: usize,
    pub compliance_status: ComplianceOverview,
    pub threat_level: ThreatLevel,
    pub system_uptime: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionStatus {
    pub active: bool,
    pub current_key_id: String,
    pub key_rotation_due: bool,
    pub keys_in_history: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionKey {
    pub id: String,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserAccount {
    pub user_id: String,
    pub username: String,
    pub password_hash: String,
    pub role: String,
    pub role_permissions: Vec<Permission>,
    pub created_at: SystemTime,
    pub last_login: Option<SystemTime>,
    pub active: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserSession {
    pub session_id: String,
    pub user_id: String,
    pub created_at: SystemTime,
    pub expires_at: SystemTime,
    pub permissions: Vec<Permission>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Role {
    pub name: String,
    pub permissions: Vec<Permission>,
    pub restrictions: Vec<Restriction>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Permission {
    SystemAdmin,
    UserManagement,
    TradingControl,
    DataAccess,
    AuditAccess,
    MarketDataAccess,
    PortfolioView,
    RiskMonitoring,
    PositionOverride,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Restriction {
    TradingHours,
    PositionLimits,
    ReadOnly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessValidation {
    pub total_users: usize,
    pub active_sessions: usize,
    pub defined_roles: usize,
    pub access_violations: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlStats {
    pub total_users: usize,
    pub active_sessions: usize,
    pub expired_sessions: usize,
    pub failed_login_attempts: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub timestamp: SystemTime,
    pub event_type: String,
    pub severity: SecuritySeverity,
    pub description: String,
    pub user_id: Option<String>,
    pub source_ip: Option<String>,
    pub additional_data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityEvent {
    SystemValidation {
        timestamp: SystemTime,
        validation_result: SecurityValidation,
    },
    DataSecured {
        timestamp: SystemTime,
        data_type: String,
        data_size: usize,
        encryption_method: String,
    },
    TradingDecisionValidated {
        timestamp: SystemTime,
        decision_id: String,
        validation_result: ComplianceValidation,
    },
    AuthenticationAttempt {
        timestamp: SystemTime,
        user_id: String,
        success: bool,
        method: String,
    },
}

impl SecurityEvent {
    pub fn get_event_type(&self) -> String {
        match self {
            SecurityEvent::SystemValidation { .. } => "system_validation".to_string(),
            SecurityEvent::DataSecured { .. } => "data_secured".to_string(),
            SecurityEvent::TradingDecisionValidated { .. } => "trading_decision_validated".to_string(),
            SecurityEvent::AuthenticationAttempt { .. } => "authentication_attempt".to_string(),
        }
    }

    pub fn get_severity(&self) -> SecuritySeverity {
        match self {
            SecurityEvent::SystemValidation { .. } => SecuritySeverity::Info,
            SecurityEvent::DataSecured { .. } => SecuritySeverity::Info,
            SecurityEvent::TradingDecisionValidated { .. } => SecuritySeverity::Warning,
            SecurityEvent::AuthenticationAttempt { success: false, .. } => SecuritySeverity::Warning,
            SecurityEvent::AuthenticationAttempt { success: true, .. } => SecuritySeverity::Info,
        }
    }

    pub fn get_description(&self) -> String {
        match self {
            SecurityEvent::SystemValidation { .. } => "System security validation performed".to_string(),
            SecurityEvent::DataSecured { data_type, .. } => format!("Data secured: {}", data_type),
            SecurityEvent::TradingDecisionValidated { .. } => "Trading decision compliance validated".to_string(),
            SecurityEvent::AuthenticationAttempt { success, user_id, .. } => {
                format!("Authentication attempt for user {}: {}", user_id, if *success { "Success" } else { "Failed" })
            }
        }
    }

    pub fn get_user_id(&self) -> Option<String> {
        match self {
            SecurityEvent::AuthenticationAttempt { user_id, .. } => Some(user_id.clone()),
            _ => None,
        }
    }

    pub fn get_source_ip(&self) -> Option<String> {
        // Would be extracted from request context in real implementation
        None
    }

    pub fn to_json(&self) -> Result<String> {
        serde_json::to_string(self).map_err(|e| anyhow!("JSON serialization failed: {}", e))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecuritySeverity {
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceRule {
    pub id: String,
    pub name: String,
    pub regulation: Regulation,
    pub description: String,
    pub check_function: ComplianceCheck,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Regulation {
    MiFIDII,
    GDPR,
    SEC,
    CFTC,
    FCA,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceCheck {
    BestExecution,
    DataRetention,
    MarketManipulation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub timestamp: SystemTime,
    pub rule_id: String,
    pub severity: SecuritySeverity,
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceStatus {
    pub overall_status: ComplianceLevel,
    pub total_rules: usize,
    pub active_violations: usize,
    pub last_audit: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplianceLevel {
    Compliant,
    Warning,
    NonCompliant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceValidation {
    pub is_compliant: bool,
    pub violations: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceOverview {
    pub total_rules: usize,
    pub total_violations: usize,
    pub regulations_covered: Vec<Regulation>,
    pub last_compliance_check: SystemTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuleCheckResult {
    Pass,
    Warning(String),
    Violation(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatPattern {
    pub id: String,
    pub name: String,
    pub severity: ThreatSeverity,
    pub pattern_regex: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThreatEvent {
    pub timestamp: SystemTime,
    pub pattern_id: String,
    pub severity: ThreatSeverity,
    pub source: String,
    pub description: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatSeverity {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreatLevel {
    Low,
    Medium,
    High,
    Critical,
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_manager_creation() {
        let config = SecurityConfig::default();
        let security_manager = SecurityManager::new(config).unwrap();
        
        let metrics = security_manager.get_security_metrics().unwrap();
        assert!(metrics.encryption_status.active);
    }

    #[test]
    fn test_encryption_manager() {
        let config = SecurityConfig::default();
        let encryption_manager = EncryptionManager::new(&config).unwrap();
        
        let test_data = b"sensitive trading data";
        let encrypted = encryption_manager.encrypt_data(test_data).unwrap();
        let decrypted = encryption_manager.decrypt_data(&encrypted).unwrap();
        
        assert_eq!(test_data.to_vec(), decrypted);
    }

    #[test]
    fn test_access_control_authentication() {
        let config = SecurityConfig::default();
        let access_manager = AccessControlManager::new(&config).unwrap();
        
        let credentials = UserCredentials {
            user_id: "test_user".to_string(),
            password: "test_password".to_string(),
            additional_factors: None,
        };
        
        let result = access_manager.authenticate_user(&credentials).unwrap();
        assert!(!result.is_authenticated); // No user exists yet
    }

    #[test]
    fn test_compliance_monitor() {
        let config = SecurityConfig::default();
        let compliance_monitor = ComplianceMonitor::new(&config).unwrap();
        
        let status = compliance_monitor.check_compliance().unwrap();
        assert!(status.total_rules > 0);
    }

    #[test]
    fn test_threat_detector() {
        let config = SecurityConfig::default();
        let threat_detector = ThreatDetector::new(&config).unwrap();
        
        let threat_level = threat_detector.assess_current_threat_level().unwrap();
        assert_eq!(threat_level, ThreatLevel::Low); // No threats detected
    }
}