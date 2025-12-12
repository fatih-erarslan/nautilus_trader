//! Hardware Security Module (HSM) Integration
//!
//! This module provides integration with Hardware Security Modules for
//! secure key generation, storage, and cryptographic operations.

pub mod manager;
pub mod pkcs11;
pub mod cloud_hsm;
pub mod operations;

pub use manager::*;
pub use pkcs11::*;
pub use cloud_hsm::*;
pub use operations::*;

use crate::error::QuantumSecurityError;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// HSM Provider Types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HSMProvider {
    /// PKCS#11 compatible HSM
    PKCS11,
    /// AWS CloudHSM
    AWSCloudHSM,
    /// Azure Dedicated HSM
    AzureDedicatedHSM,
    /// Azure Key Vault (Managed HSM)
    AzureKeyVault,
    /// Google Cloud HSM
    GoogleCloudHSM,
    /// Thales nShield
    ThalesNShield,
    /// Utimaco CryptoServer
    UtimacoCryptoServer,
    /// Safenet Luna
    SafenetLuna,
    /// YubiKey (for development/testing)
    YubiKey,
    /// Software emulation (for testing)
    SoftwareEmulation,
}

/// HSM Operation Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HSMOperation {
    /// Generate cryptographic key
    GenerateKey {
        key_type: HSMKeyType,
        key_size: u32,
        attributes: HashMap<String, String>,
    },
    /// Import existing key
    ImportKey {
        key_data: SecureBytes,
        key_type: HSMKeyType,
        attributes: HashMap<String, String>,
    },
    /// Export key (if allowed)
    ExportKey {
        key_handle: HSMKeyHandle,
        format: KeyExportFormat,
    },
    /// Delete key
    DeleteKey {
        key_handle: HSMKeyHandle,
    },
    /// Encrypt data
    Encrypt {
        key_handle: HSMKeyHandle,
        plaintext: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    },
    /// Decrypt data
    Decrypt {
        key_handle: HSMKeyHandle,
        ciphertext: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    },
    /// Sign data
    Sign {
        key_handle: HSMKeyHandle,
        data: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    },
    /// Verify signature
    Verify {
        key_handle: HSMKeyHandle,
        data: Vec<u8>,
        signature: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    },
    /// Generate random bytes
    GenerateRandom {
        length: usize,
    },
    /// Derive key
    DeriveKey {
        base_key_handle: HSMKeyHandle,
        derivation_data: Vec<u8>,
        algorithm: String,
        parameters: Option<Vec<u8>>,
    },
}

/// HSM Key Types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum HSMKeyType {
    /// AES symmetric key
    AES,
    /// ChaCha20 symmetric key
    ChaCha20,
    /// RSA key pair
    RSA,
    /// ECDSA key pair
    ECDSA,
    /// Ed25519 key pair
    Ed25519,
    /// CRYSTALS-Kyber key pair
    Kyber,
    /// CRYSTALS-Dilithium key pair
    Dilithium,
    /// FALCON key pair
    Falcon,
    /// SPHINCS+ key pair
    SphincsPlus,
    /// HMAC key
    HMAC,
    /// Key derivation key
    KDF,
    /// Generic secret
    Generic,
}

/// HSM Key Handle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMKeyHandle {
    pub handle_id: String,
    pub key_type: HSMKeyType,
    pub provider: HSMProvider,
    pub slot_id: Option<u32>,
    pub key_label: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub attributes: HashMap<String, String>,
}

/// Key Export Formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyExportFormat {
    Raw,
    PKCS8,
    PKCS1,
    X509,
    JWK,
    PEM,
    DER,
}

/// HSM Session Information
#[derive(Debug, Clone)]
pub struct HSMSession {
    pub session_id: Uuid,
    pub provider: HSMProvider,
    pub slot_id: Option<u32>,
    pub authenticated: bool,
    pub read_write: bool,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub operation_count: u64,
}

/// HSM Status Information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMStatus {
    pub provider: HSMProvider,
    pub available: bool,
    pub authenticated: bool,
    pub firmware_version: String,
    pub hardware_version: String,
    pub serial_number: String,
    pub total_memory: u64,
    pub free_memory: u64,
    pub key_count: u32,
    pub max_sessions: u32,
    pub active_sessions: u32,
    pub temperature: Option<f64>,
    pub last_check: chrono::DateTime<chrono::Utc>,
}

/// HSM Performance Metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HSMPerformanceMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub average_latency_us: f64,
    pub max_latency_us: u64,
    pub min_latency_us: u64,
    pub operations_per_second: f64,
    pub operation_distribution: HashMap<String, u64>,
    pub error_distribution: HashMap<String, u64>,
    pub session_count: u64,
    pub session_duration_avg_ms: f64,
    pub authentication_count: u64,
    pub authentication_failures: u64,
    pub memory_usage_percent: f64,
    pub key_operations: u64,
    pub random_generation_bytes: u64,
}

/// HSM Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMConfiguration {
    pub provider: HSMProvider,
    pub connection_config: HSMConnectionConfig,
    pub security_config: HSMSecurityConfig,
    pub performance_config: HSMPerformanceConfig,
    pub operation_config: HSMOperationConfig,
}

/// HSM Connection Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMConnectionConfig {
    /// PKCS#11 library path
    pub library_path: Option<String>,
    /// Slot ID to use
    pub slot_id: Option<u32>,
    /// Token label
    pub token_label: Option<String>,
    /// Connection pool configuration
    pub pool_config: PoolConfig,
    /// Timeout settings
    pub timeouts: TimeoutConfig,
    /// Retry configuration
    pub retry_config: RetryConfig,
}

/// Connection Pool Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoolConfig {
    pub min_connections: u32,
    pub max_connections: u32,
    pub connection_timeout_ms: u32,
    pub idle_timeout_ms: u32,
    pub max_lifetime_ms: u32,
}

/// Timeout Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeoutConfig {
    pub connect_timeout_ms: u32,
    pub operation_timeout_ms: u32,
    pub read_timeout_ms: u32,
    pub write_timeout_ms: u32,
}

/// Retry Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u32,
    pub max_delay_ms: u32,
    pub backoff_factor: f64,
    pub jitter: bool,
}

/// HSM Security Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMSecurityConfig {
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Access control configuration
    pub access_control: AccessControlConfig,
    /// Audit configuration
    pub audit: AuditConfig,
    /// Tamper detection enabled
    pub tamper_detection: bool,
    /// Secure boot verification
    pub secure_boot: bool,
}

/// HSM Authentication Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// User PIN/password
    pub user_pin: Option<SecureBytes>,
    /// SO (Security Officer) PIN
    pub so_pin: Option<SecureBytes>,
    /// Certificate-based authentication
    pub certificate_auth: bool,
    /// Multi-factor authentication required
    pub mfa_required: bool,
    /// Session timeout
    pub session_timeout_minutes: u32,
    /// Auto-logout on inactivity
    pub auto_logout: bool,
}

/// Access Control Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccessControlConfig {
    /// Role-based access control
    pub rbac_enabled: bool,
    /// Dual control required for sensitive operations
    pub dual_control: bool,
    /// Key access policies
    pub key_access_policies: HashMap<String, Vec<String>>,
    /// Operation restrictions
    pub operation_restrictions: HashMap<String, Vec<String>>,
}

/// Audit Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    /// Audit logging enabled
    pub enabled: bool,
    /// Log all operations
    pub log_all_operations: bool,
    /// Log failed operations only
    pub log_failures_only: bool,
    /// Include sensitive data in logs
    pub include_sensitive_data: bool,
    /// Log rotation settings
    pub rotation_settings: LogRotationSettings,
}

/// Log Rotation Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationSettings {
    pub max_file_size_mb: u32,
    pub max_files: u32,
    pub compression: bool,
    pub encryption: bool,
}

/// HSM Performance Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMPerformanceConfig {
    /// Enable operation caching
    pub caching_enabled: bool,
    /// Cache size
    pub cache_size: usize,
    /// Cache TTL
    pub cache_ttl_seconds: u32,
    /// Batch operations when possible
    pub batch_operations: bool,
    /// Maximum batch size
    pub max_batch_size: u32,
    /// Parallel operations
    pub parallel_operations: u32,
    /// Load balancing across multiple HSMs
    pub load_balancing: bool,
}

/// HSM Operation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMOperationConfig {
    /// Allowed key types
    pub allowed_key_types: Vec<HSMKeyType>,
    /// Key generation policies
    pub key_generation_policies: KeyGenerationPolicies,
    /// Key lifecycle policies
    pub key_lifecycle_policies: KeyLifecyclePolicies,
    /// Backup and recovery policies
    pub backup_policies: BackupPolicies,
}

/// Key Generation Policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyGenerationPolicies {
    /// Require hardware key generation
    pub hardware_generation_required: bool,
    /// Minimum key sizes
    pub min_key_sizes: HashMap<HSMKeyType, u32>,
    /// Default key attributes
    pub default_attributes: HashMap<HSMKeyType, HashMap<String, String>>,
    /// Key escrow required
    pub key_escrow_required: bool,
}

/// Key Lifecycle Policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyLifecyclePolicies {
    /// Automatic key rotation
    pub auto_rotation: bool,
    /// Key rotation intervals
    pub rotation_intervals: HashMap<HSMKeyType, chrono::Duration>,
    /// Key archival policies
    pub archival_policies: HashMap<HSMKeyType, chrono::Duration>,
    /// Key destruction policies
    pub destruction_policies: HashMap<HSMKeyType, chrono::Duration>,
}

/// Backup Policies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupPolicies {
    /// Automatic backup enabled
    pub auto_backup: bool,
    /// Backup frequency
    pub backup_frequency: chrono::Duration,
    /// Backup encryption required
    pub backup_encryption: bool,
    /// Backup verification required
    pub backup_verification: bool,
    /// Offsite backup required
    pub offsite_backup: bool,
}

/// HSM Operation Result
#[derive(Debug, Clone)]
pub struct HSMOperationResult {
    pub operation_id: Uuid,
    pub operation_type: String,
    pub success: bool,
    pub result_data: Option<Vec<u8>>,
    pub key_handle: Option<HSMKeyHandle>,
    pub execution_time_us: u64,
    pub error_message: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// HSM Event Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HSMEvent {
    /// HSM connected
    Connected { provider: HSMProvider },
    /// HSM disconnected
    Disconnected { provider: HSMProvider, reason: String },
    /// Authentication successful
    AuthenticationSuccess { user_type: String },
    /// Authentication failed
    AuthenticationFailed { user_type: String, reason: String },
    /// Key generated
    KeyGenerated { key_handle: String, key_type: HSMKeyType },
    /// Key imported
    KeyImported { key_handle: String, key_type: HSMKeyType },
    /// Key deleted
    KeyDeleted { key_handle: String },
    /// Operation performed
    OperationPerformed { operation: String, duration_us: u64 },
    /// Operation failed
    OperationFailed { operation: String, error: String },
    /// Tamper detected
    TamperDetected { details: String },
    /// Hardware error
    HardwareError { error: String },
    /// Firmware update
    FirmwareUpdate { version: String },
    /// Backup completed
    BackupCompleted { backup_id: String },
    /// Restore completed
    RestoreCompleted { backup_id: String },
}

impl Default for HSMConfiguration {
    fn default() -> Self {
        Self {
            provider: HSMProvider::SoftwareEmulation,
            connection_config: HSMConnectionConfig::default(),
            security_config: HSMSecurityConfig::default(),
            performance_config: HSMPerformanceConfig::default(),
            operation_config: HSMOperationConfig::default(),
        }
    }
}

impl Default for HSMConnectionConfig {
    fn default() -> Self {
        Self {
            library_path: None,
            slot_id: None,
            token_label: None,
            pool_config: PoolConfig::default(),
            timeouts: TimeoutConfig::default(),
            retry_config: RetryConfig::default(),
        }
    }
}

impl Default for PoolConfig {
    fn default() -> Self {
        Self {
            min_connections: 1,
            max_connections: 10,
            connection_timeout_ms: 30000,
            idle_timeout_ms: 300000,
            max_lifetime_ms: 3600000,
        }
    }
}

impl Default for TimeoutConfig {
    fn default() -> Self {
        Self {
            connect_timeout_ms: 30000,
            operation_timeout_ms: 60000,
            read_timeout_ms: 30000,
            write_timeout_ms: 30000,
        }
    }
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_delay_ms: 1000,
            max_delay_ms: 10000,
            backoff_factor: 2.0,
            jitter: true,
        }
    }
}

impl Default for HSMSecurityConfig {
    fn default() -> Self {
        Self {
            authentication: AuthenticationConfig::default(),
            access_control: AccessControlConfig::default(),
            audit: AuditConfig::default(),
            tamper_detection: true,
            secure_boot: true,
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            user_pin: None,
            so_pin: None,
            certificate_auth: false,
            mfa_required: false,
            session_timeout_minutes: 60,
            auto_logout: true,
        }
    }
}

impl Default for AccessControlConfig {
    fn default() -> Self {
        Self {
            rbac_enabled: true,
            dual_control: false,
            key_access_policies: HashMap::new(),
            operation_restrictions: HashMap::new(),
        }
    }
}

impl Default for AuditConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_all_operations: true,
            log_failures_only: false,
            include_sensitive_data: false,
            rotation_settings: LogRotationSettings::default(),
        }
    }
}

impl Default for LogRotationSettings {
    fn default() -> Self {
        Self {
            max_file_size_mb: 100,
            max_files: 30,
            compression: true,
            encryption: true,
        }
    }
}

impl Default for HSMPerformanceConfig {
    fn default() -> Self {
        Self {
            caching_enabled: true,
            cache_size: 1000,
            cache_ttl_seconds: 3600,
            batch_operations: true,
            max_batch_size: 100,
            parallel_operations: 4,
            load_balancing: false,
        }
    }
}

impl Default for HSMOperationConfig {
    fn default() -> Self {
        let mut min_key_sizes = HashMap::new();
        min_key_sizes.insert(HSMKeyType::AES, 256);
        min_key_sizes.insert(HSMKeyType::RSA, 2048);
        min_key_sizes.insert(HSMKeyType::ECDSA, 256);
        
        Self {
            allowed_key_types: vec![
                HSMKeyType::AES,
                HSMKeyType::ChaCha20,
                HSMKeyType::RSA,
                HSMKeyType::ECDSA,
                HSMKeyType::Ed25519,
                HSMKeyType::Kyber,
                HSMKeyType::Dilithium,
                HSMKeyType::Falcon,
                HSMKeyType::SphincsPlus,
            ],
            key_generation_policies: KeyGenerationPolicies {
                hardware_generation_required: true,
                min_key_sizes,
                default_attributes: HashMap::new(),
                key_escrow_required: false,
            },
            key_lifecycle_policies: KeyLifecyclePolicies {
                auto_rotation: true,
                rotation_intervals: HashMap::new(),
                archival_policies: HashMap::new(),
                destruction_policies: HashMap::new(),
            },
            backup_policies: BackupPolicies {
                auto_backup: true,
                backup_frequency: chrono::Duration::hours(24),
                backup_encryption: true,
                backup_verification: true,
                offsite_backup: false,
            },
        }
    }
}

impl HSMKeyHandle {
    /// Create new HSM key handle
    pub fn new(
        key_type: HSMKeyType,
        provider: HSMProvider,
        key_label: String,
    ) -> Self {
        Self {
            handle_id: Uuid::new_v4().to_string(),
            key_type,
            provider,
            slot_id: None,
            key_label,
            created_at: chrono::Utc::now(),
            attributes: HashMap::new(),
        }
    }
    
    /// Set slot ID
    pub fn with_slot_id(mut self, slot_id: u32) -> Self {
        self.slot_id = Some(slot_id);
        self
    }
    
    /// Add attribute
    pub fn with_attribute(mut self, key: String, value: String) -> Self {
        self.attributes.insert(key, value);
        self
    }
    
    /// Check if key is quantum-safe
    pub fn is_quantum_safe(&self) -> bool {
        matches!(self.key_type, 
            HSMKeyType::Kyber | 
            HSMKeyType::Dilithium | 
            HSMKeyType::Falcon | 
            HSMKeyType::SphincsPlus
        )
    }
    
    /// Get key age
    pub fn age(&self) -> chrono::Duration {
        chrono::Utc::now().signed_duration_since(self.created_at)
    }
}

impl HSMOperationResult {
    /// Create successful result
    pub fn success(
        operation_type: String,
        execution_time_us: u64,
    ) -> Self {
        Self {
            operation_id: Uuid::new_v4(),
            operation_type,
            success: true,
            result_data: None,
            key_handle: None,
            execution_time_us,
            error_message: None,
            metadata: HashMap::new(),
        }
    }
    
    /// Create failed result
    pub fn failure(
        operation_type: String,
        execution_time_us: u64,
        error_message: String,
    ) -> Self {
        Self {
            operation_id: Uuid::new_v4(),
            operation_type,
            success: false,
            result_data: None,
            key_handle: None,
            execution_time_us,
            error_message: Some(error_message),
            metadata: HashMap::new(),
        }
    }
    
    /// Set result data
    pub fn with_data(mut self, data: Vec<u8>) -> Self {
        self.result_data = Some(data);
        self
    }
    
    /// Set key handle
    pub fn with_key_handle(mut self, handle: HSMKeyHandle) -> Self {
        self.key_handle = Some(handle);
        self
    }
    
    /// Add metadata
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hsm_key_handle_creation() {
        let handle = HSMKeyHandle::new(
            HSMKeyType::AES,
            HSMProvider::PKCS11,
            "test_key".to_string(),
        )
        .with_slot_id(1)
        .with_attribute("purpose".to_string(), "encryption".to_string());
        
        assert_eq!(handle.key_type, HSMKeyType::AES);
        assert_eq!(handle.provider, HSMProvider::PKCS11);
        assert_eq!(handle.key_label, "test_key");
        assert_eq!(handle.slot_id, Some(1));
        assert_eq!(handle.attributes.get("purpose"), Some(&"encryption".to_string()));
        assert!(!handle.is_quantum_safe());
    }
    
    #[test]
    fn test_quantum_safe_key_types() {
        let kyber_handle = HSMKeyHandle::new(
            HSMKeyType::Kyber,
            HSMProvider::PKCS11,
            "kyber_key".to_string(),
        );
        
        assert!(kyber_handle.is_quantum_safe());
        
        let rsa_handle = HSMKeyHandle::new(
            HSMKeyType::RSA,
            HSMProvider::PKCS11,
            "rsa_key".to_string(),
        );
        
        assert!(!rsa_handle.is_quantum_safe());
    }
    
    #[test]
    fn test_hsm_operation_result() {
        let success_result = HSMOperationResult::success("generate_key".to_string(), 1000)
            .with_data(vec![1, 2, 3, 4])
            .with_metadata("algorithm".to_string(), "AES-256".to_string());
        
        assert!(success_result.success);
        assert_eq!(success_result.operation_type, "generate_key");
        assert_eq!(success_result.execution_time_us, 1000);
        assert_eq!(success_result.result_data, Some(vec![1, 2, 3, 4]));
        
        let failure_result = HSMOperationResult::failure(
            "sign_data".to_string(),
            500,
            "Key not found".to_string(),
        );
        
        assert!(!failure_result.success);
        assert_eq!(failure_result.error_message, Some("Key not found".to_string()));
    }
    
    #[test]
    fn test_hsm_configuration_defaults() {
        let config = HSMConfiguration::default();
        
        assert_eq!(config.provider, HSMProvider::SoftwareEmulation);
        assert!(config.security_config.tamper_detection);
        assert!(config.performance_config.caching_enabled);
        assert!(config.operation_config.key_generation_policies.hardware_generation_required);
    }
}