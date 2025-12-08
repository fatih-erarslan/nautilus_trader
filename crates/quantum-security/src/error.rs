//! Quantum Security Error Types
//!
//! Comprehensive error handling for the quantum security framework

use crate::key_distribution::QKDError;
use thiserror::Error;
use uuid::Uuid;

/// Main Quantum Security Error Type
#[derive(Error, Debug)]
pub enum QuantumSecurityError {
    // Cryptographic Errors
    #[error("Cryptographic operation failed: {0}")]
    CryptographicError(String),
    
    #[error("Key generation failed: {0}")]
    KeyGenerationError(String),
    
    #[error("Key derivation failed: {0}")]
    KeyDerivationError(String),
    
    #[error("Encryption failed: {0}")]
    EncryptionError(String),
    
    #[error("Decryption failed: {0}")]
    DecryptionError(String),
    
    #[error("Signature generation failed: {0}")]
    SignatureError(String),
    
    #[error("Signature verification failed: {0}")]
    VerificationError(String),
    
    #[error("Hash computation failed: {0}")]
    HashError(String),
    
    // Algorithm Errors
    #[error("Invalid algorithm: {0}")]
    InvalidAlgorithm(String),
    
    #[error("Unsupported algorithm: {0}")]
    UnsupportedAlgorithm(String),
    
    #[error("Algorithm not available: {0}")]
    AlgorithmNotAvailable(String),
    
    #[error("Invalid algorithm parameters: {0}")]
    InvalidParameters(String),
    
    #[error("Unsupported protocol: {0}")]
    UnsupportedProtocol(String),
    
    // Key Management Errors
    #[error("Invalid key type: {0}")]
    InvalidKeyType(String),
    
    #[error("Invalid key size: expected {expected}, got {actual}")]
    InvalidKeySize { expected: usize, actual: usize },
    
    #[error("Key not found: {0}")]
    KeyNotFound(String),
    
    #[error("Key expired: {0}")]
    KeyExpired(String),
    
    #[error("Key already exists: {0}")]
    KeyAlreadyExists(String),
    
    #[error("Key corruption detected: {0}")]
    KeyCorruption(String),
    
    #[error("Key rotation failed: {0}")]
    KeyRotationError(String),
    
    // Authentication Errors
    #[error("Authentication failed: {0}")]
    AuthenticationFailed(String),
    
    #[error("Authorization failed: {0}")]
    AuthorizationFailed(String),
    
    #[error("Invalid credentials: {0}")]
    InvalidCredentials(String),
    
    #[error("MFA challenge failed: {0}")]
    MFAChallengeFailed(String),
    
    #[error("Biometric verification failed: {0}")]
    BiometricVerificationFailed(String),
    
    #[error("Certificate validation failed: {0}")]
    CertificateValidationFailed(String),
    
    #[error("Session expired: {0}")]
    SessionExpired(Uuid),
    
    #[error("Session not found: {0}")]
    SessionNotFound(Uuid),
    
    #[error("Maximum authentication attempts exceeded")]
    MaxAttemptsExceeded,
    
    // QKD Errors
    #[error("QKD error: {0}")]
    QKDError(#[from] QKDError),
    
    #[error("QKD session failed: {0}")]
    QKDSessionFailed(String),
    
    #[error("Quantum channel error: {0}")]
    QuantumChannelError(String),
    
    #[error("Key distribution failed: {0}")]
    KeyDistributionFailed(String),
    
    #[error("Quantum noise detected: {0}")]
    QuantumNoiseDetected(String),
    
    #[error("Eavesdropping detected: {0}")]
    EavesdroppingDetected(String),
    
    // Network and Node Errors
    #[error("Node not found: {0}")]
    NodeNotFound(String),
    
    #[error("Node already exists: {0}")]
    NodeAlreadyExists(String),
    
    #[error("Node offline: {0}")]
    NodeOffline(String),
    
    #[error("Network error: {0}")]
    NetworkError(String),
    
    #[error("Connection failed: {0}")]
    ConnectionFailed(String),
    
    #[error("Communication timeout: {0}")]
    CommunicationTimeout(String),
    
    // Threat Detection Errors
    #[error("Threat detection failed: {0}")]
    ThreatDetectionFailed(String),
    
    #[error("ML model error: {0}")]
    MLModelError(String),
    
    #[error("Anomaly detection failed: {0}")]
    AnomalyDetectionFailed(String),
    
    #[error("Threat analysis failed: {0}")]
    ThreatAnalysisFailed(String),
    
    #[error("Security policy violation: {0}")]
    SecurityPolicyViolation(String),
    
    // HSM Errors
    #[error("HSM operation failed: {0}")]
    HSMOperationFailed(String),
    
    #[error("HSM not available: {0}")]
    HSMNotAvailable(String),
    
    #[error("HSM authentication failed: {0}")]
    HSMAuthenticationFailed(String),
    
    #[error("HSM slot error: {0}")]
    HSMSlotError(String),
    
    #[error("HSM token error: {0}")]
    HSMTokenError(String),
    
    #[error("PKCS#11 error: {0}")]
    PKCS11Error(String),
    
    // Data Validation Errors
    #[error("Invalid data format: {0}")]
    InvalidDataFormat(String),
    
    #[error("Invalid data: {0}")]
    InvalidData(String),
    
    #[error("Data corruption detected: {0}")]
    DataCorruption(String),
    
    #[error("Data validation failed: {0}")]
    DataValidationFailed(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Deserialization error: {0}")]
    DeserializationError(String),
    
    // Configuration Errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    
    #[error("Invalid configuration: {0}")]
    InvalidConfiguration(String),
    
    #[error("Missing configuration: {0}")]
    MissingConfiguration(String),
    
    #[error("Configuration validation failed: {0}")]
    ConfigurationValidationFailed(String),
    
    // Performance Errors
    #[error("Performance threshold exceeded: {0}")]
    PerformanceThresholdExceeded(String),
    
    #[error("Latency threshold exceeded: expected < {expected}μs, got {actual}μs")]
    LatencyThresholdExceeded { expected: u64, actual: u64 },
    
    #[error("Throughput threshold not met: expected {expected}/s, got {actual}/s")]
    ThroughputThresholdNotMet { expected: u64, actual: u64 },
    
    #[error("Resource limit exceeded: {0}")]
    ResourceLimitExceeded(String),
    
    #[error("Memory limit exceeded: {0}")]
    MemoryLimitExceeded(String),
    
    // System Errors
    #[error("System error: {0}")]
    SystemError(String),
    
    #[error("Hardware error: {0}")]
    HardwareError(String),
    
    #[error("Platform not supported: {0}")]
    PlatformNotSupported(String),
    
    #[error("Feature not available: {0}")]
    FeatureNotAvailable(String),
    
    #[error("Service unavailable: {0}")]
    ServiceUnavailable(String),
    
    // I/O Errors
    #[error("I/O error: {0}")]
    IOError(#[from] std::io::Error),
    
    #[error("File not found: {0}")]
    FileNotFound(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Storage error: {0}")]
    StorageError(String),
    
    // Compliance and Audit Errors
    #[error("Compliance violation: {0}")]
    ComplianceViolation(String),
    
    #[error("Audit trail corruption: {0}")]
    AuditTrailCorruption(String),
    
    #[error("Regulatory requirement not met: {0}")]
    RegulatoryRequirementNotMet(String),
    
    #[error("Data retention policy violation: {0}")]
    DataRetentionPolicyViolation(String),
    
    #[error("Privacy policy violation: {0}")]
    PrivacyPolicyViolation(String),
    
    // External Service Errors
    #[error("External service error: {0}")]
    ExternalServiceError(String),
    
    #[error("API error: {0}")]
    APIError(String),
    
    #[error("Certificate authority error: {0}")]
    CertificateAuthorityError(String),
    
    #[error("Time service error: {0}")]
    TimeServiceError(String),
    
    #[error("Random number generator error: {0}")]
    RandomNumberGeneratorError(String),
    
    // Concurrency Errors
    #[error("Lock acquisition failed: {0}")]
    LockAcquisitionFailed(String),
    
    #[error("Concurrency error: {0}")]
    ConcurrencyError(String),
    
    #[error("Deadlock detected: {0}")]
    DeadlockDetected(String),
    
    #[error("Race condition detected: {0}")]
    RaceConditionDetected(String),
    
    // Protocol Errors
    #[error("Protocol error: {0}")]
    ProtocolError(String),
    
    #[error("Protocol version mismatch: expected {expected}, got {actual}")]
    ProtocolVersionMismatch { expected: String, actual: String },
    
    #[error("Message format error: {0}")]
    MessageFormatError(String),
    
    #[error("Handshake failed: {0}")]
    HandshakeFailed(String),
    
    // Quantum-Specific Errors
    #[error("Quantum decoherence detected: {0}")]
    QuantumDecoherence(String),
    
    #[error("Quantum measurement error: {0}")]
    QuantumMeasurementError(String),
    
    #[error("Quantum state preparation failed: {0}")]
    QuantumStatePreparationFailed(String),
    
    #[error("Quantum gate operation failed: {0}")]
    QuantumGateOperationFailed(String),
    
    #[error("Quantum error correction failed: {0}")]
    QuantumErrorCorrectionFailed(String),
    
    #[error("Bell inequality violation: {0}")]
    BellInequalityViolation(String),
    
    #[error("Quantum supremacy threshold not met: {0}")]
    QuantumSupremacyThresholdNotMet(String),
    
    // Generic Errors
    #[error("Unknown error: {0}")]
    Unknown(String),
    
    #[error("Internal error: {0}")]
    Internal(String),
    
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    
    #[error("Operation cancelled: {0}")]
    OperationCancelled(String),
    
    #[error("Operation timeout: {0}")]
    OperationTimeout(String),
    
    // Additional Authentication Errors
    #[error("Token not found: {0}")]
    TokenNotFound(String),
    
    #[error("Token expired: {0}")]
    TokenExpired(String),
    
    #[error("Token revoked: {0}")]
    TokenRevoked(String),
    
    #[error("Token exhausted: {0}")]
    TokenExhausted(String),
    
    #[error("Challenge not found: {0}")]
    ChallengeNotFound(String),
    
    #[error("Challenge expired: {0}")]
    ChallengeExpired(String),
    
    #[error("Context not found: {0}")]
    ContextNotFound(String),
    
    #[error("Context expired: {0}")]
    ContextExpired(String),
    
    #[error("Incomplete authentication: {0}")]
    IncompleteAuthentication(String),
    
    // Certificate Authority Errors
    #[error("Certificate not found: {0}")]
    CertificateNotFound(String),
    
    #[error("CA not found: {0}")]
    CANotFound(String),
    
    #[error("CA not trusted: {0}")]
    CANotTrusted(String),
    
    // Algorithm Errors
    #[error("Algorithm not found: {0}")]
    AlgorithmNotFound(String),
    
    #[error("Algorithm not enabled: {0}")]
    AlgorithmNotEnabled(String),
    
    #[error("Algorithm mismatch: {0}")]
    AlgorithmMismatch(String),
    
    // HSM Errors
    #[error("HSM session limit reached: {0}")]
    HSMSessionLimitReached(String),
    
    // Template and Policy Errors
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    
    #[error("Policy not found: {0}")]
    PolicyNotFound(String),
    
    #[error("Invalid enrollment: {0}")]
    InvalidEnrollment(String),
    
    // Signature Errors
    #[error("Invalid signature size: {0}")]
    InvalidSignatureSize(String),
    
    #[error("Signing failed: {0}")]
    SigningFailed(String),
    
    // Operation Errors
    #[error("Invalid operation: {0}")]
    InvalidOperation(String),
    
    #[error("Unsupported method: {0}")]
    UnsupportedMethod(String),
}

/// Result type for quantum security operations
pub type QuantumSecurityResult<T> = Result<T, QuantumSecurityError>;

/// Error severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ErrorSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Error context information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub component: String,
    pub session_id: Option<Uuid>,
    pub agent_id: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub additional_info: std::collections::HashMap<String, String>,
}

/// Enhanced error with context
#[derive(Debug)]
pub struct QuantumSecurityErrorWithContext {
    pub error: QuantumSecurityError,
    pub severity: ErrorSeverity,
    pub context: ErrorContext,
    pub retry_recommended: bool,
    pub recovery_suggestions: Vec<String>,
}

impl QuantumSecurityError {
    /// Get error severity
    pub fn severity(&self) -> ErrorSeverity {
        match self {
            // Critical security errors
            QuantumSecurityError::EavesdroppingDetected(_) |
            QuantumSecurityError::SecurityPolicyViolation(_) |
            QuantumSecurityError::KeyCorruption(_) |
            QuantumSecurityError::DataCorruption(_) |
            QuantumSecurityError::AuditTrailCorruption(_) => ErrorSeverity::Critical,
            
            // High severity errors
            QuantumSecurityError::AuthenticationFailed(_) |
            QuantumSecurityError::AuthorizationFailed(_) |
            QuantumSecurityError::CertificateValidationFailed(_) |
            QuantumSecurityError::HSMOperationFailed(_) |
            QuantumSecurityError::QuantumChannelError(_) => ErrorSeverity::Error,
            
            // Medium severity errors
            QuantumSecurityError::SessionExpired(_) |
            QuantumSecurityError::NodeOffline(_) |
            QuantumSecurityError::PerformanceThresholdExceeded(_) |
            QuantumSecurityError::LatencyThresholdExceeded { .. } => ErrorSeverity::Warning,
            
            // Low severity errors
            QuantumSecurityError::ConfigurationError(_) |
            QuantumSecurityError::NotImplemented(_) => ErrorSeverity::Info,
            
            // Fatal errors
            QuantumSecurityError::SystemError(_) |
            QuantumSecurityError::HardwareError(_) => ErrorSeverity::Fatal,
            
            // Default to error level
            _ => ErrorSeverity::Error,
        }
    }
    
    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            // Non-recoverable errors
            QuantumSecurityError::HardwareError(_) |
            QuantumSecurityError::SystemError(_) |
            QuantumSecurityError::PlatformNotSupported(_) |
            QuantumSecurityError::DataCorruption(_) |
            QuantumSecurityError::KeyCorruption(_) => false,
            
            // Potentially recoverable errors
            QuantumSecurityError::NetworkError(_) |
            QuantumSecurityError::ConnectionFailed(_) |
            QuantumSecurityError::CommunicationTimeout(_) |
            QuantumSecurityError::NodeOffline(_) |
            QuantumSecurityError::ServiceUnavailable(_) => true,
            
            // Default to recoverable
            _ => true,
        }
    }
    
    /// Check if retry is recommended
    pub fn should_retry(&self) -> bool {
        match self {
            // Retry recommended for transient errors
            QuantumSecurityError::NetworkError(_) |
            QuantumSecurityError::ConnectionFailed(_) |
            QuantumSecurityError::CommunicationTimeout(_) |
            QuantumSecurityError::ServiceUnavailable(_) |
            QuantumSecurityError::OperationTimeout(_) => true,
            
            // No retry for permanent errors
            QuantumSecurityError::InvalidCredentials(_) |
            QuantumSecurityError::AuthenticationFailed(_) |
            QuantumSecurityError::InvalidAlgorithm(_) |
            QuantumSecurityError::UnsupportedAlgorithm(_) |
            QuantumSecurityError::NotImplemented(_) => false,
            
            // Default to no retry
            _ => false,
        }
    }
    
    /// Get error category
    pub fn category(&self) -> &'static str {
        match self {
            QuantumSecurityError::CryptographicError(_) |
            QuantumSecurityError::KeyGenerationError(_) |
            QuantumSecurityError::EncryptionError(_) |
            QuantumSecurityError::DecryptionError(_) |
            QuantumSecurityError::SignatureError(_) |
            QuantumSecurityError::VerificationError(_) => "cryptographic",
            
            QuantumSecurityError::AuthenticationFailed(_) |
            QuantumSecurityError::AuthorizationFailed(_) |
            QuantumSecurityError::InvalidCredentials(_) |
            QuantumSecurityError::SessionExpired(_) |
            QuantumSecurityError::SessionNotFound(_) => "authentication",
            
            QuantumSecurityError::QKDError(_) |
            QuantumSecurityError::QuantumChannelError(_) |
            QuantumSecurityError::QuantumNoiseDetected(_) |
            QuantumSecurityError::EavesdroppingDetected(_) => "quantum",
            
            QuantumSecurityError::NetworkError(_) |
            QuantumSecurityError::ConnectionFailed(_) |
            QuantumSecurityError::CommunicationTimeout(_) |
            QuantumSecurityError::NodeOffline(_) => "network",
            
            QuantumSecurityError::HSMOperationFailed(_) |
            QuantumSecurityError::HSMNotAvailable(_) |
            QuantumSecurityError::PKCS11Error(_) => "hsm",
            
            QuantumSecurityError::ThreatDetectionFailed(_) |
            QuantumSecurityError::SecurityPolicyViolation(_) |
            QuantumSecurityError::AnomalyDetectionFailed(_) => "security",
            
            QuantumSecurityError::PerformanceThresholdExceeded(_) |
            QuantumSecurityError::LatencyThresholdExceeded { .. } |
            QuantumSecurityError::ResourceLimitExceeded(_) => "performance",
            
            QuantumSecurityError::ConfigurationError(_) |
            QuantumSecurityError::InvalidConfiguration(_) => "configuration",
            
            _ => "general",
        }
    }
    
    /// Get recovery suggestions
    pub fn recovery_suggestions(&self) -> Vec<String> {
        match self {
            QuantumSecurityError::NetworkError(_) => vec![
                "Check network connectivity".to_string(),
                "Verify firewall settings".to_string(),
                "Retry the operation".to_string(),
            ],
            
            QuantumSecurityError::AuthenticationFailed(_) => vec![
                "Verify credentials".to_string(),
                "Check authentication policy".to_string(),
                "Reset authentication session".to_string(),
            ],
            
            QuantumSecurityError::LatencyThresholdExceeded { .. } => vec![
                "Check system load".to_string(),
                "Review performance configuration".to_string(),
                "Consider hardware optimization".to_string(),
            ],
            
            QuantumSecurityError::HSMNotAvailable(_) => vec![
                "Check HSM connectivity".to_string(),
                "Verify HSM configuration".to_string(),
                "Fallback to software cryptography".to_string(),
            ],
            
            QuantumSecurityError::KeyExpired(_) => vec![
                "Generate new key".to_string(),
                "Check key rotation policy".to_string(),
                "Update key management settings".to_string(),
            ],
            
            _ => vec!["Check logs for more details".to_string()],
        }
    }
}

impl ErrorContext {
    /// Create new error context
    pub fn new(operation: String, component: String) -> Self {
        Self {
            operation,
            component,
            session_id: None,
            agent_id: None,
            timestamp: chrono::Utc::now(),
            additional_info: std::collections::HashMap::new(),
        }
    }
    
    /// Set session ID
    pub fn with_session_id(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }
    
    /// Set agent ID
    pub fn with_agent_id(mut self, agent_id: String) -> Self {
        self.agent_id = Some(agent_id);
        self
    }
    
    /// Add additional information
    pub fn with_info(mut self, key: String, value: String) -> Self {
        self.additional_info.insert(key, value);
        self
    }
}

impl QuantumSecurityErrorWithContext {
    /// Create new error with context
    pub fn new(error: QuantumSecurityError, context: ErrorContext) -> Self {
        let severity = error.severity();
        let retry_recommended = error.should_retry();
        let recovery_suggestions = error.recovery_suggestions();
        
        Self {
            error,
            severity,
            context,
            retry_recommended,
            recovery_suggestions,
        }
    }
    
    /// Log the error with appropriate level
    pub fn log(&self) {
        match self.severity {
            ErrorSeverity::Fatal => tracing::error!(
                operation = %self.context.operation,
                component = %self.context.component,
                error = %self.error,
                "Fatal quantum security error"
            ),
            ErrorSeverity::Critical => tracing::error!(
                operation = %self.context.operation,
                component = %self.context.component,
                error = %self.error,
                "Critical quantum security error"
            ),
            ErrorSeverity::Error => tracing::error!(
                operation = %self.context.operation,
                component = %self.context.component,
                error = %self.error,
                "Quantum security error"
            ),
            ErrorSeverity::Warning => tracing::warn!(
                operation = %self.context.operation,
                component = %self.context.component,
                error = %self.error,
                "Quantum security warning"
            ),
            ErrorSeverity::Info => tracing::info!(
                operation = %self.context.operation,
                component = %self.context.component,
                error = %self.error,
                "Quantum security info"
            ),
        }
    }
}

// Conversion from common error types
impl From<serde_json::Error> for QuantumSecurityError {
    fn from(err: serde_json::Error) -> Self {
        QuantumSecurityError::SerializationError(err.to_string())
    }
}

impl From<bincode::Error> for QuantumSecurityError {
    fn from(err: bincode::Error) -> Self {
        QuantumSecurityError::SerializationError(err.to_string())
    }
}

impl From<chrono::ParseError> for QuantumSecurityError {
    fn from(err: chrono::ParseError) -> Self {
        QuantumSecurityError::DataValidationFailed(err.to_string())
    }
}

impl From<uuid::Error> for QuantumSecurityError {
    fn from(err: uuid::Error) -> Self {
        QuantumSecurityError::DataValidationFailed(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_error_severity() {
        assert_eq!(
            QuantumSecurityError::EavesdroppingDetected("test".to_string()).severity(),
            ErrorSeverity::Critical
        );
        
        assert_eq!(
            QuantumSecurityError::SessionExpired(Uuid::new_v4()).severity(),
            ErrorSeverity::Warning
        );
        
        assert_eq!(
            QuantumSecurityError::SystemError("test".to_string()).severity(),
            ErrorSeverity::Fatal
        );
    }
    
    #[test]
    fn test_error_recoverability() {
        assert!(!QuantumSecurityError::HardwareError("test".to_string()).is_recoverable());
        assert!(QuantumSecurityError::NetworkError("test".to_string()).is_recoverable());
    }
    
    #[test]
    fn test_error_retry_recommendation() {
        assert!(QuantumSecurityError::NetworkError("test".to_string()).should_retry());
        assert!(!QuantumSecurityError::InvalidCredentials("test".to_string()).should_retry());
    }
    
    #[test]
    fn test_error_category() {
        assert_eq!(
            QuantumSecurityError::CryptographicError("test".to_string()).category(),
            "cryptographic"
        );
        
        assert_eq!(
            QuantumSecurityError::AuthenticationFailed("test".to_string()).category(),
            "authentication"
        );
        
        assert_eq!(
            QuantumSecurityError::QKDError(QKDError::HighErrorRate { measured: 0.1, threshold: 0.05 }).category(),
            "quantum"
        );
    }
    
    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("test_op".to_string(), "test_component".to_string())
            .with_session_id(Uuid::new_v4())
            .with_agent_id("test_agent".to_string())
            .with_info("key".to_string(), "value".to_string());
        
        assert_eq!(context.operation, "test_op");
        assert_eq!(context.component, "test_component");
        assert!(context.session_id.is_some());
        assert!(context.agent_id.is_some());
        assert_eq!(context.additional_info.get("key"), Some(&"value".to_string()));
    }
    
    #[test]
    fn test_error_with_context() {
        let error = QuantumSecurityError::NetworkError("test".to_string());
        let context = ErrorContext::new("test_op".to_string(), "test_component".to_string());
        let error_with_context = QuantumSecurityErrorWithContext::new(error, context);
        
        assert_eq!(error_with_context.severity, ErrorSeverity::Error);
        assert!(error_with_context.retry_recommended);
        assert!(!error_with_context.recovery_suggestions.is_empty());
    }
}