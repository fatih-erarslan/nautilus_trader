//! Quantum Security Configuration
//!
//! Central configuration management for the quantum security framework

use crate::algorithms::AlgorithmConfig;
use crate::auth::AuthenticationPolicy;
use crate::key_distribution::QKDConfig;
use crate::threat_detection::ThreatDetectionConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Main Quantum Security Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSecurityConfig {
    /// Algorithm configuration
    pub algorithms: AlgorithmConfig,
    /// Quantum Key Distribution configuration
    pub qkd: QKDConfig,
    /// Authentication configuration
    pub authentication: AuthenticationConfig,
    /// Threat detection configuration
    pub threat_detection: ThreatDetectionConfig,
    /// Hardware Security Module configuration
    pub hsm: HSMConfig,
    /// Communication configuration
    pub communication: CommunicationConfig,
    /// Performance configuration
    pub performance: PerformanceConfig,
    /// Compliance configuration
    pub compliance: ComplianceConfig,
    /// Session configuration
    pub session_timeout_hours: u32,
    /// Maximum latency threshold in microseconds
    pub max_latency_us: u64,
    /// Security level
    pub security_level: crate::types::SecurityLevel,
    /// Enable side channel protection
    pub enable_side_channel_protection: bool,
    /// Default signature algorithm
    pub default_signature_algorithm: String,
}

/// Authentication Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthenticationConfig {
    /// Default authentication policy
    pub default_policy: AuthenticationPolicy,
    /// Additional authentication policies
    pub policies: HashMap<String, AuthenticationPolicy>,
    /// MFA settings
    pub mfa_settings: MFASettings,
    /// Biometric settings
    pub biometric_settings: BiometricSettings,
    /// Certificate settings
    pub certificate_settings: CertificateSettings,
    /// Session settings
    pub session_settings: SessionSettings,
}

/// Multi-Factor Authentication Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MFASettings {
    /// Require MFA for all operations
    pub require_mfa: bool,
    /// TOTP settings
    pub totp_settings: TOTPSettings,
    /// Hardware key settings
    pub hardware_key_settings: HardwareKeySettings,
    /// Backup codes enabled
    pub backup_codes_enabled: bool,
    /// Maximum authentication attempts
    pub max_attempts: u32,
    /// Lockout duration after max attempts
    pub lockout_duration_minutes: u32,
}

/// Time-based One-Time Password Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TOTPSettings {
    /// Secret key length in bytes
    pub secret_length: usize,
    /// Time step in seconds
    pub time_step: u32,
    /// Code length (digits)
    pub code_length: u8,
    /// Time window tolerance
    pub window_size: u32,
    /// Algorithm (SHA1, SHA256, SHA512)
    pub algorithm: String,
}

/// Hardware Key Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareKeySettings {
    /// Supported FIDO2 algorithms
    pub supported_algorithms: Vec<String>,
    /// Require user presence verification
    pub require_user_presence: bool,
    /// Require user verification (PIN/biometric)
    pub require_user_verification: bool,
    /// Attestation requirement
    pub attestation_required: bool,
    /// Allowed authenticator types
    pub allowed_authenticators: Vec<String>,
}

/// Biometric Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BiometricSettings {
    /// Enabled biometric types
    pub enabled_types: Vec<String>,
    /// Quality threshold (0.0 - 1.0)
    pub quality_threshold: f64,
    /// False acceptance rate threshold
    pub far_threshold: f64,
    /// False rejection rate threshold
    pub frr_threshold: f64,
    /// Template encryption enabled
    pub template_encryption: bool,
    /// Liveness detection enabled
    pub liveness_detection: bool,
}

/// Certificate Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateSettings {
    /// Certificate authority settings
    pub ca_settings: CASettings,
    /// Certificate validation settings
    pub validation_settings: CertificateValidationSettings,
    /// Certificate lifecycle settings
    pub lifecycle_settings: CertificateLifecycleSettings,
    /// Quantum-safe certificates enabled
    pub quantum_safe_enabled: bool,
}

/// Certificate Authority Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CASettings {
    /// Root CA certificate path
    pub root_ca_path: String,
    /// Intermediate CA certificate path
    pub intermediate_ca_path: String,
    /// CRL (Certificate Revocation List) URL
    pub crl_url: Option<String>,
    /// OCSP (Online Certificate Status Protocol) URL
    pub ocsp_url: Option<String>,
    /// Certificate validity period
    pub validity_period_days: u32,
}

/// Certificate Validation Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateValidationSettings {
    /// Check certificate revocation
    pub check_revocation: bool,
    /// Validate certificate chain
    pub validate_chain: bool,
    /// Check certificate expiration
    pub check_expiration: bool,
    /// Allow self-signed certificates
    pub allow_self_signed: bool,
    /// Certificate cache timeout
    pub cache_timeout_hours: u32,
}

/// Certificate Lifecycle Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertificateLifecycleSettings {
    /// Auto-renewal enabled
    pub auto_renewal: bool,
    /// Renewal threshold (days before expiry)
    pub renewal_threshold_days: u32,
    /// Key rotation enabled
    pub key_rotation: bool,
    /// Key rotation interval
    pub key_rotation_interval_days: u32,
    /// Backup key generation
    pub backup_key_generation: bool,
}

/// Session Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    /// Default session timeout
    pub default_timeout_hours: u32,
    /// Maximum session timeout
    pub max_timeout_hours: u32,
    /// Session inactivity timeout
    pub inactivity_timeout_minutes: u32,
    /// Concurrent sessions allowed
    pub max_concurrent_sessions: u32,
    /// Session renewal enabled
    pub session_renewal_enabled: bool,
    /// Session encryption enabled
    pub session_encryption_enabled: bool,
}

/// Hardware Security Module Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMConfig {
    /// HSM enabled
    pub enabled: bool,
    /// HSM type (PKCS#11, Azure Key Vault, AWS CloudHSM, etc.)
    pub hsm_type: String,
    /// Connection settings
    pub connection: HSMConnectionConfig,
    /// Key management settings
    pub key_management: HSMKeyManagementConfig,
    /// Performance settings
    pub performance: HSMPerformanceConfig,
    /// Security settings
    pub security: HSMSecurityConfig,
}

/// HSM Connection Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMConnectionConfig {
    /// PKCS#11 library path
    pub pkcs11_library: Option<String>,
    /// Slot ID
    pub slot_id: Option<u32>,
    /// Token label
    pub token_label: Option<String>,
    /// User PIN
    pub user_pin: Option<String>,
    /// Connection pool size
    pub pool_size: u32,
    /// Connection timeout
    pub timeout_seconds: u32,
    /// Retry attempts
    pub retry_attempts: u32,
}

/// HSM Key Management Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMKeyManagementConfig {
    /// Key generation in HSM
    pub generate_in_hsm: bool,
    /// Key extraction allowed
    pub allow_key_extraction: bool,
    /// Key backup enabled
    pub key_backup: bool,
    /// Key escrow enabled
    pub key_escrow: bool,
    /// Key rotation enabled
    pub automatic_rotation: bool,
    /// Rotation interval
    pub rotation_interval_days: u32,
}

/// HSM Performance Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMPerformanceConfig {
    /// Operation caching enabled
    pub caching_enabled: bool,
    /// Cache size
    pub cache_size: usize,
    /// Cache TTL
    pub cache_ttl_seconds: u32,
    /// Parallel operations
    pub parallel_operations: u32,
    /// Batch operations enabled
    pub batch_operations: bool,
}

/// HSM Security Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HSMSecurityConfig {
    /// Authentication required
    pub authentication_required: bool,
    /// Dual control enabled
    pub dual_control: bool,
    /// Audit logging enabled
    pub audit_logging: bool,
    /// Tamper detection enabled
    pub tamper_detection: bool,
    /// Secure boot verification
    pub secure_boot: bool,
}

/// Communication Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationConfig {
    /// TLS settings
    pub tls: TLSConfig,
    /// Noise protocol settings
    pub noise: NoiseConfig,
    /// Message encryption settings
    pub message_encryption: MessageEncryptionConfig,
    /// Channel settings
    pub channels: ChannelConfig,
}

/// TLS Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TLSConfig {
    /// Minimum TLS version
    pub min_version: String,
    /// Cipher suites
    pub cipher_suites: Vec<String>,
    /// Certificate verification
    pub verify_certificates: bool,
    /// Mutual TLS enabled
    pub mutual_tls: bool,
    /// Post-quantum cipher suites
    pub post_quantum_ciphers: Vec<String>,
}

/// Noise Protocol Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    /// Noise protocol pattern
    pub pattern: String,
    /// DH function
    pub dh_function: String,
    /// Cipher function
    pub cipher_function: String,
    /// Hash function
    pub hash_function: String,
    /// PSK enabled
    pub psk_enabled: bool,
}

/// Message Encryption Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageEncryptionConfig {
    /// Default encryption algorithm
    pub default_algorithm: String,
    /// Key derivation function
    pub key_derivation: String,
    /// Message authentication
    pub message_authentication: bool,
    /// Forward secrecy enabled
    pub forward_secrecy: bool,
    /// Quantum-safe encryption
    pub quantum_safe: bool,
}

/// Channel Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Default channel type
    pub default_type: String,
    /// Buffer sizes
    pub buffer_sizes: HashMap<String, usize>,
    /// Timeouts
    pub timeouts: HashMap<String, u32>,
    /// Retry settings
    pub retry_settings: RetrySettings,
}

/// Retry Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrySettings {
    /// Maximum retry attempts
    pub max_attempts: u32,
    /// Initial delay (milliseconds)
    pub initial_delay_ms: u32,
    /// Maximum delay (milliseconds)
    pub max_delay_ms: u32,
    /// Backoff multiplier
    pub backoff_multiplier: f64,
    /// Jitter enabled
    pub jitter_enabled: bool,
}

/// Performance Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Target latency (microseconds)
    pub target_latency_us: u64,
    /// Maximum latency (microseconds)
    pub max_latency_us: u64,
    /// Throughput targets
    pub throughput_targets: ThroughputTargets,
    /// Resource limits
    pub resource_limits: ResourceLimits,
    /// Optimization settings
    pub optimization: OptimizationSettings,
}

/// Throughput Targets
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputTargets {
    /// Encryption operations per second
    pub encryptions_per_sec: u64,
    /// Signature operations per second
    pub signatures_per_sec: u64,
    /// Key generation operations per second
    pub key_generations_per_sec: u64,
    /// Authentication operations per second
    pub authentications_per_sec: u64,
}

/// Resource Limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    /// Maximum memory usage (MB)
    pub max_memory_mb: usize,
    /// Maximum CPU usage (percentage)
    pub max_cpu_percent: f64,
    /// Maximum concurrent operations
    pub max_concurrent_ops: u32,
    /// Maximum key cache size
    pub max_key_cache_size: usize,
}

/// Optimization Settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSettings {
    /// SIMD acceleration enabled
    pub simd_acceleration: bool,
    /// Hardware acceleration enabled
    pub hardware_acceleration: bool,
    /// Assembly optimizations enabled
    pub assembly_optimizations: bool,
    /// Parallel processing enabled
    pub parallel_processing: bool,
    /// Memory pool enabled
    pub memory_pool: bool,
}

/// Compliance Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceConfig {
    /// Enabled compliance frameworks
    pub enabled_frameworks: Vec<String>,
    /// Audit logging settings
    pub audit_logging: AuditLoggingConfig,
    /// Data retention settings
    pub data_retention: DataRetentionConfig,
    /// Privacy settings
    pub privacy: PrivacyConfig,
    /// Regulatory settings
    pub regulatory: RegulatoryConfig,
}

/// Audit Logging Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLoggingConfig {
    /// Audit logging enabled
    pub enabled: bool,
    /// Log level
    pub log_level: String,
    /// Log format
    pub log_format: String,
    /// Log rotation settings
    pub rotation: LogRotationConfig,
    /// Log encryption enabled
    pub encryption_enabled: bool,
    /// Log integrity protection
    pub integrity_protection: bool,
}

/// Log Rotation Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Maximum file size (MB)
    pub max_file_size_mb: usize,
    /// Maximum number of files
    pub max_files: u32,
    /// Rotation frequency
    pub rotation_frequency: String,
    /// Compression enabled
    pub compression_enabled: bool,
}

/// Data Retention Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionConfig {
    /// Default retention period (days)
    pub default_retention_days: u32,
    /// Retention policies
    pub policies: HashMap<String, u32>,
    /// Automatic deletion enabled
    pub auto_deletion: bool,
    /// Secure deletion enabled
    pub secure_deletion: bool,
}

/// Privacy Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Data anonymization enabled
    pub anonymization_enabled: bool,
    /// PII detection enabled
    pub pii_detection: bool,
    /// Data masking enabled
    pub data_masking: bool,
    /// Consent management enabled
    pub consent_management: bool,
    /// Right to be forgotten enabled
    pub right_to_be_forgotten: bool,
}

/// Regulatory Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegulatoryConfig {
    /// Jurisdiction
    pub jurisdiction: String,
    /// Regulatory frameworks
    pub frameworks: Vec<String>,
    /// Compliance reporting enabled
    pub reporting_enabled: bool,
    /// Regulatory notifications enabled
    pub notifications_enabled: bool,
    /// Cross-border data transfer restrictions
    pub cross_border_restrictions: bool,
}

impl Default for QuantumSecurityConfig {
    fn default() -> Self {
        Self {
            algorithms: AlgorithmConfig::default(),
            qkd: QKDConfig::default(),
            authentication: AuthenticationConfig::default(),
            threat_detection: ThreatDetectionConfig::default(),
            hsm: HSMConfig::default(),
            communication: CommunicationConfig::default(),
            performance: PerformanceConfig::default(),
            compliance: ComplianceConfig::default(),
            session_timeout_hours: 8,
            max_latency_us: 100,
            security_level: crate::types::SecurityLevel::Standard,
            enable_side_channel_protection: true,
            default_signature_algorithm: "CRYSTALS-Dilithium".to_string(),
        }
    }
}

impl Default for AuthenticationConfig {
    fn default() -> Self {
        Self {
            default_policy: AuthenticationPolicy::default_policy(),
            policies: HashMap::new(),
            mfa_settings: MFASettings::default(),
            biometric_settings: BiometricSettings::default(),
            certificate_settings: CertificateSettings::default(),
            session_settings: SessionSettings::default(),
        }
    }
}

impl Default for MFASettings {
    fn default() -> Self {
        Self {
            require_mfa: true,
            totp_settings: TOTPSettings::default(),
            hardware_key_settings: HardwareKeySettings::default(),
            backup_codes_enabled: true,
            max_attempts: 3,
            lockout_duration_minutes: 15,
        }
    }
}

impl Default for TOTPSettings {
    fn default() -> Self {
        Self {
            secret_length: 32,
            time_step: 30,
            code_length: 6,
            window_size: 1,
            algorithm: "SHA256".to_string(),
        }
    }
}

impl Default for HardwareKeySettings {
    fn default() -> Self {
        Self {
            supported_algorithms: vec!["ES256".to_string(), "EdDSA".to_string()],
            require_user_presence: true,
            require_user_verification: false,
            attestation_required: true,
            allowed_authenticators: vec!["platform".to_string(), "cross-platform".to_string()],
        }
    }
}

impl Default for BiometricSettings {
    fn default() -> Self {
        Self {
            enabled_types: vec!["fingerprint".to_string(), "face".to_string()],
            quality_threshold: 0.8,
            far_threshold: 0.001,
            frr_threshold: 0.01,
            template_encryption: true,
            liveness_detection: true,
        }
    }
}

impl Default for CertificateSettings {
    fn default() -> Self {
        Self {
            ca_settings: CASettings::default(),
            validation_settings: CertificateValidationSettings::default(),
            lifecycle_settings: CertificateLifecycleSettings::default(),
            quantum_safe_enabled: true,
        }
    }
}

impl Default for CASettings {
    fn default() -> Self {
        Self {
            root_ca_path: "/etc/ssl/certs/root-ca.pem".to_string(),
            intermediate_ca_path: "/etc/ssl/certs/intermediate-ca.pem".to_string(),
            crl_url: None,
            ocsp_url: None,
            validity_period_days: 365,
        }
    }
}

impl Default for CertificateValidationSettings {
    fn default() -> Self {
        Self {
            check_revocation: true,
            validate_chain: true,
            check_expiration: true,
            allow_self_signed: false,
            cache_timeout_hours: 24,
        }
    }
}

impl Default for CertificateLifecycleSettings {
    fn default() -> Self {
        Self {
            auto_renewal: true,
            renewal_threshold_days: 30,
            key_rotation: true,
            key_rotation_interval_days: 90,
            backup_key_generation: true,
        }
    }
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            default_timeout_hours: 8,
            max_timeout_hours: 24,
            inactivity_timeout_minutes: 30,
            max_concurrent_sessions: 5,
            session_renewal_enabled: true,
            session_encryption_enabled: true,
        }
    }
}

impl Default for HSMConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            hsm_type: "PKCS11".to_string(),
            connection: HSMConnectionConfig::default(),
            key_management: HSMKeyManagementConfig::default(),
            performance: HSMPerformanceConfig::default(),
            security: HSMSecurityConfig::default(),
        }
    }
}

impl Default for HSMConnectionConfig {
    fn default() -> Self {
        Self {
            pkcs11_library: None,
            slot_id: None,
            token_label: None,
            user_pin: None,
            pool_size: 10,
            timeout_seconds: 30,
            retry_attempts: 3,
        }
    }
}

impl Default for HSMKeyManagementConfig {
    fn default() -> Self {
        Self {
            generate_in_hsm: true,
            allow_key_extraction: false,
            key_backup: true,
            key_escrow: false,
            automatic_rotation: true,
            rotation_interval_days: 90,
        }
    }
}

impl Default for HSMPerformanceConfig {
    fn default() -> Self {
        Self {
            caching_enabled: true,
            cache_size: 1000,
            cache_ttl_seconds: 3600,
            parallel_operations: 4,
            batch_operations: true,
        }
    }
}

impl Default for HSMSecurityConfig {
    fn default() -> Self {
        Self {
            authentication_required: true,
            dual_control: false,
            audit_logging: true,
            tamper_detection: true,
            secure_boot: true,
        }
    }
}

impl Default for CommunicationConfig {
    fn default() -> Self {
        Self {
            tls: TLSConfig::default(),
            noise: NoiseConfig::default(),
            message_encryption: MessageEncryptionConfig::default(),
            channels: ChannelConfig::default(),
        }
    }
}

impl Default for TLSConfig {
    fn default() -> Self {
        Self {
            min_version: "1.3".to_string(),
            cipher_suites: vec![
                "TLS_AES_256_GCM_SHA384".to_string(),
                "TLS_CHACHA20_POLY1305_SHA256".to_string(),
            ],
            verify_certificates: true,
            mutual_tls: true,
            post_quantum_ciphers: vec![
                "KYBER1024_AES256".to_string(),
                "DILITHIUM5_RSA".to_string(),
            ],
        }
    }
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            pattern: "Noise_XX_25519_ChaChaPoly_BLAKE2s".to_string(),
            dh_function: "25519".to_string(),
            cipher_function: "ChaChaPoly".to_string(),
            hash_function: "BLAKE2s".to_string(),
            psk_enabled: false,
        }
    }
}

impl Default for MessageEncryptionConfig {
    fn default() -> Self {
        Self {
            default_algorithm: "ChaCha20Poly1305".to_string(),
            key_derivation: "HKDF-SHA256".to_string(),
            message_authentication: true,
            forward_secrecy: true,
            quantum_safe: true,
        }
    }
}

impl Default for ChannelConfig {
    fn default() -> Self {
        let mut buffer_sizes = HashMap::new();
        buffer_sizes.insert("default".to_string(), 8192);
        buffer_sizes.insert("bulk".to_string(), 65536);
        
        let mut timeouts = HashMap::new();
        timeouts.insert("connect".to_string(), 30);
        timeouts.insert("read".to_string(), 60);
        timeouts.insert("write".to_string(), 60);
        
        Self {
            default_type: "tcp".to_string(),
            buffer_sizes,
            timeouts,
            retry_settings: RetrySettings::default(),
        }
    }
}

impl Default for RetrySettings {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay_ms: 100,
            max_delay_ms: 5000,
            backoff_multiplier: 2.0,
            jitter_enabled: true,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            target_latency_us: 50,
            max_latency_us: 100,
            throughput_targets: ThroughputTargets::default(),
            resource_limits: ResourceLimits::default(),
            optimization: OptimizationSettings::default(),
        }
    }
}

impl Default for ThroughputTargets {
    fn default() -> Self {
        Self {
            encryptions_per_sec: 10000,
            signatures_per_sec: 5000,
            key_generations_per_sec: 100,
            authentications_per_sec: 1000,
        }
    }
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_mb: 512,
            max_cpu_percent: 80.0,
            max_concurrent_ops: 1000,
            max_key_cache_size: 10000,
        }
    }
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        Self {
            simd_acceleration: true,
            hardware_acceleration: true,
            assembly_optimizations: true,
            parallel_processing: true,
            memory_pool: true,
        }
    }
}

impl Default for ComplianceConfig {
    fn default() -> Self {
        Self {
            enabled_frameworks: vec![
                "SOX".to_string(),
                "GDPR".to_string(),
                "CCPA".to_string(),
                "NIST".to_string(),
            ],
            audit_logging: AuditLoggingConfig::default(),
            data_retention: DataRetentionConfig::default(),
            privacy: PrivacyConfig::default(),
            regulatory: RegulatoryConfig::default(),
        }
    }
}

impl Default for AuditLoggingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            log_level: "INFO".to_string(),
            log_format: "JSON".to_string(),
            rotation: LogRotationConfig::default(),
            encryption_enabled: true,
            integrity_protection: true,
        }
    }
}

impl Default for LogRotationConfig {
    fn default() -> Self {
        Self {
            max_file_size_mb: 100,
            max_files: 30,
            rotation_frequency: "daily".to_string(),
            compression_enabled: true,
        }
    }
}

impl Default for DataRetentionConfig {
    fn default() -> Self {
        let mut policies = HashMap::new();
        policies.insert("audit_logs".to_string(), 2555); // 7 years
        policies.insert("security_events".to_string(), 1095); // 3 years
        policies.insert("session_data".to_string(), 90); // 3 months
        
        Self {
            default_retention_days: 365,
            policies,
            auto_deletion: true,
            secure_deletion: true,
        }
    }
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self {
            anonymization_enabled: true,
            pii_detection: true,
            data_masking: true,
            consent_management: true,
            right_to_be_forgotten: true,
        }
    }
}

impl Default for RegulatoryConfig {
    fn default() -> Self {
        Self {
            jurisdiction: "US".to_string(),
            frameworks: vec![
                "SOX".to_string(),
                "FINRA".to_string(),
                "SEC".to_string(),
            ],
            reporting_enabled: true,
            notifications_enabled: true,
            cross_border_restrictions: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config_creation() {
        let config = QuantumSecurityConfig::default();
        
        assert_eq!(config.session_timeout_hours, 8);
        assert_eq!(config.max_latency_us, 100);
        assert!(config.authentication.mfa_settings.require_mfa);
        assert!(config.threat_detection.quantum_analysis_enabled);
        assert!(config.compliance.audit_logging.enabled);
    }
    
    #[test]
    fn test_authentication_config() {
        let auth_config = AuthenticationConfig::default();
        
        assert!(auth_config.mfa_settings.require_mfa);
        assert_eq!(auth_config.mfa_settings.max_attempts, 3);
        assert_eq!(auth_config.session_settings.default_timeout_hours, 8);
        assert!(auth_config.certificate_settings.quantum_safe_enabled);
    }
    
    #[test]
    fn test_performance_config() {
        let perf_config = PerformanceConfig::default();
        
        assert_eq!(perf_config.target_latency_us, 50);
        assert_eq!(perf_config.max_latency_us, 100);
        assert!(perf_config.optimization.simd_acceleration);
        assert!(perf_config.optimization.parallel_processing);
    }
    
    #[test]
    fn test_hsm_config() {
        let hsm_config = HSMConfig::default();
        
        assert!(!hsm_config.enabled); // Disabled by default
        assert_eq!(hsm_config.hsm_type, "PKCS11");
        assert!(hsm_config.key_management.generate_in_hsm);
        assert!(!hsm_config.key_management.allow_key_extraction);
        assert!(hsm_config.security.authentication_required);
    }
}