//! Common Types for Quantum Security Framework
//!
//! Shared data structures and types used across the quantum security system

use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter, Result as FmtResult};
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Encryption metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    pub algorithm: String,
    pub key_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub content_type: String,
    pub compression: Option<String>,
    pub additional_data: Option<Vec<u8>>,
    pub sender_id: Option<String>,
    pub recipient_ids: Vec<String>,
    pub expiry: Option<chrono::DateTime<chrono::Utc>>,
    pub classification: Option<String>,
}

/// Encrypted data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptedData {
    pub algorithm: String,
    pub ciphertext: Vec<u8>,
    pub nonce: Option<Vec<u8>>,
    pub tag: Option<Vec<u8>>,
    pub key_id: Option<String>,
    pub metadata: Option<EncryptionMetadata>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}


/// Secure byte storage with automatic zeroing
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
pub struct SecureBytes {
    data: Vec<u8>,
}

impl SecureBytes {
    /// Create new secure bytes
    pub fn new(data: Vec<u8>) -> Self {
        Self { data }
    }
    
    /// Create empty secure bytes
    pub fn empty() -> Self {
        Self { data: Vec::new() }
    }
    
    /// Create from slice
    pub fn from_slice(slice: &[u8]) -> Self {
        Self { data: slice.to_vec() }
    }
    
    /// Get length
    pub fn len(&self) -> usize {
        self.data.len()
    }
    
    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    
    /// Expose data (use carefully)
    pub fn expose(&self) -> &[u8] {
        &self.data
    }
    
    /// Get as bytes (alias for expose)
    pub fn as_bytes(&self) -> &[u8] {
        &self.data
    }
    
    /// Convert to vector (consumes self)
    pub fn into_vec(self) -> Vec<u8> {
        self.data.clone()
    }
    
    /// Append data
    pub fn append(&mut self, other: &[u8]) {
        self.data.extend_from_slice(other);
    }
    
    /// Split at position
    pub fn split_at(&self, mid: usize) -> (SecureBytes, SecureBytes) {
        let (left, right) = self.data.split_at(mid);
        (SecureBytes::from_slice(left), SecureBytes::from_slice(right))
    }
    
    /// Take first n bytes
    pub fn take(&self, n: usize) -> SecureBytes {
        let len = n.min(self.data.len());
        SecureBytes::from_slice(&self.data[..len])
    }
    
    /// XOR with another SecureBytes
    pub fn xor(&self, other: &SecureBytes) -> SecureBytes {
        let mut result = Vec::with_capacity(self.data.len().max(other.data.len()));
        for i in 0..result.capacity() {
            let a = self.data.get(i).unwrap_or(&0);
            let b = other.data.get(i).unwrap_or(&0);
            result.push(a ^ b);
        }
        SecureBytes::new(result)
    }
}

impl From<Vec<u8>> for SecureBytes {
    fn from(data: Vec<u8>) -> Self {
        Self::new(data)
    }
}

impl From<&[u8]> for SecureBytes {
    fn from(data: &[u8]) -> Self {
        Self::from_slice(data)
    }
}

impl PartialEq for SecureBytes {
    fn eq(&self, other: &Self) -> bool {
        use subtle::ConstantTimeEq;
        self.data.len() == other.data.len() &&
            self.data.ct_eq(&other.data).into()
    }
}

impl Eq for SecureBytes {}

/// Signature types for different algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignatureType {
    /// CRYSTALS-Dilithium signature
    Dilithium,
    /// FALCON signature
    Falcon,
    /// SPHINCS+ signature
    SphincsPlus,
    /// Classical ECDSA (for compatibility)
    ECDSA,
    /// Classical RSA PSS (for compatibility)
    RSAPSS,
    /// Ed25519 (for compatibility)
    Ed25519,
}

/// Quantum signature with metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumSignature {
    pub signature_type: SignatureType,
    pub signature_data: Vec<u8>,
    pub public_key_id: Option<String>,
    pub algorithm_parameters: Option<Vec<u8>>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub signer_info: Option<SignerInfo>,
}

/// Signer information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignerInfo {
    pub signer_id: String,
    pub certificate_chain: Option<Vec<Vec<u8>>>,
    pub attributes: std::collections::HashMap<String, String>,
}


/// Secure key material for sessions
#[derive(Debug, Clone)]
pub struct SecureKeyMaterial {
    pub encryption_key: SecureBytes,
    pub mac_key: SecureBytes,
    pub signature_keypair: Option<QuantumKeyPair>,
    pub key_derivation_salt: SecureBytes,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

/// Quantum key pair for signatures
#[derive(Debug, Clone)]
pub struct QuantumKeyPair {
    pub algorithm: SignatureType,
    pub public_key: Vec<u8>,
    pub private_key: SecureBytes,
    pub key_id: String,
}

/// Communication channel handle
#[derive(Debug, Clone)]
pub struct QuantumChannelHandle {
    pub channel_id: Uuid,
    pub local_agent_id: String,
    pub remote_agent_id: String,
    pub encryption_keys: SecureKeyMaterial,
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub channel_type: ChannelType,
}

/// Channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChannelType {
    /// Direct TCP connection
    TCP,
    /// WebSocket connection
    WebSocket,
    /// UDP connection
    UDP,
    /// Quantum channel
    Quantum,
    /// Hybrid classical-quantum channel
    Hybrid,
}

/// Security level definitions
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// Level 1 security (basic)
    Level1,
    /// Level 2 security (standard)
    Level2,
    /// Level 3 security (enhanced)
    Level3,
    /// Level 4 security (high)
    Level4,
    /// Level 5 security (maximum)
    Level5,
    /// Basic security (128-bit equivalent)
    Basic,
    /// Standard security (192-bit equivalent)
    Standard,
    /// High security (256-bit equivalent)
    High,
    /// Quantum-safe security
    QuantumSafe,
    /// Maximum security (all available protections)
    Maximum,
}

/// Quantum security metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QuantumSecurityMetrics {
    pub total_operations: u64,
    pub successful_operations: u64,
    pub failed_operations: u64,
    pub error_count: u64,
    pub average_latency_us: f64,
    pub max_latency_us: u64,
    pub min_latency_us: u64,
    pub throughput_ops_per_sec: f64,
    pub operations_by_type: std::collections::HashMap<String, u64>,
    pub error_distribution: std::collections::HashMap<String, u64>,
    pub performance_alerts: u64,
    pub security_violations: u64,
    pub quantum_operations: u64,
    pub classical_operations: u64,
    pub hybrid_operations: u64,
    pub key_rotations: u64,
    pub successful_authentications: u64,
    pub failed_authentications: u64,
    pub threat_detections: u64,
    pub false_positives: u64,
    pub compliance_checks: u64,
    pub compliance_violations: u64,
}

/// Agent information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub agent_id: String,
    pub agent_type: String,
    pub capabilities: Vec<String>,
    pub security_level: SecurityLevel,
    pub status: AgentStatus,
    pub last_seen: chrono::DateTime<chrono::Utc>,
    pub authentication_level: Option<crate::auth::AuthenticationLevel>,
    pub risk_score: f64,
}

/// Agent status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum AgentStatus {
    Online,
    Offline,
    Suspended,
    Compromised,
    Maintenance,
}

/// Cryptographic key information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyInfo {
    pub key_id: String,
    pub algorithm: String,
    pub key_type: KeyType,
    pub key_size: u32,
    pub usage: Vec<KeyUsage>,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub status: KeyStatus,
    pub quantum_safe: bool,
}

/// Key types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyType {
    Symmetric,
    AsymmetricPrivate,
    AsymmetricPublic,
    MacKey,
    DerivationKey,
    QuantumKey,
}

/// Key usage flags
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum KeyUsage {
    Encryption,
    Decryption,
    Signing,
    Verification,
    KeyAgreement,
    KeyEncapsulation,
    KeyDerivation,
    MessageAuthentication,
    DataAuthentication,
}

/// Key status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum KeyStatus {
    Active,
    Suspended,
    Revoked,
    Expired,
    Compromised,
}

/// Performance metrics for operations
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation_count: u64,
    pub total_duration_us: u64,
    pub min_duration_us: u64,
    pub max_duration_us: u64,
    pub average_duration_us: f64,
    pub success_rate: f64,
    pub error_rate: f64,
    pub throughput_ops_per_sec: f64,
    pub p50_latency_us: u64,
    pub p95_latency_us: u64,
    pub p99_latency_us: u64,
}

/// Security event for monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityEvent {
    pub event_id: Uuid,
    pub event_type: String,
    pub severity: EventSeverity,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub source: String,
    pub target: Option<String>,
    pub description: String,
    pub data: std::collections::HashMap<String, String>,
    pub resolution_status: ResolutionStatus,
}

/// Event severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
    Fatal,
}

/// Resolution status for security events
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ResolutionStatus {
    Open,
    InProgress,
    Resolved,
    Closed,
    Escalated,
}

/// Network connection information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    pub connection_id: Uuid,
    pub local_address: String,
    pub remote_address: String,
    pub protocol: String,
    pub encryption: EncryptionInfo,
    pub established_at: chrono::DateTime<chrono::Utc>,
    pub last_activity: chrono::DateTime<chrono::Utc>,
    pub bytes_sent: u64,
    pub bytes_received: u64,
    pub status: ConnectionStatus,
}

/// Encryption information for connections
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionInfo {
    pub cipher_suite: String,
    pub key_exchange: String,
    pub signature_algorithm: String,
    pub hash_algorithm: String,
    pub quantum_safe: bool,
    pub security_level: SecurityLevel,
}

/// Connection status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ConnectionStatus {
    Connecting,
    Established,
    Closing,
    Closed,
    Error,
}

/// Time-based value with automatic expiration
#[derive(Debug, Clone)]
pub struct ExpiringValue<T> {
    pub value: T,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: chrono::DateTime<chrono::Utc>,
}

impl<T> ExpiringValue<T> {
    /// Create new expiring value
    pub fn new(value: T, ttl: chrono::Duration) -> Self {
        let now = chrono::Utc::now();
        Self {
            value,
            created_at: now,
            expires_at: now + ttl,
        }
    }
    
    /// Check if expired
    pub fn is_expired(&self) -> bool {
        chrono::Utc::now() > self.expires_at
    }
    
    /// Get remaining time to live
    pub fn remaining_ttl(&self) -> chrono::Duration {
        self.expires_at.signed_duration_since(chrono::Utc::now())
    }
    
    /// Get age
    pub fn age(&self) -> chrono::Duration {
        chrono::Utc::now().signed_duration_since(self.created_at)
    }
}

/// Rate limiter configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub max_requests: u64,
    pub time_window: chrono::Duration,
    pub burst_size: u64,
    pub strategy: RateLimitStrategy,
}

/// Rate limiting strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitStrategy {
    FixedWindow,
    SlidingWindow,
    TokenBucket,
    LeakyBucket,
    Adaptive,
}

/// Configuration validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub recommendations: Vec<String>,
}

impl ValidationResult {
    /// Create new validation result
    pub fn new() -> Self {
        Self {
            valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            recommendations: Vec::new(),
        }
    }
    
    /// Add error
    pub fn add_error(&mut self, error: String) {
        self.valid = false;
        self.errors.push(error);
    }
    
    /// Add warning
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }
    
    /// Add recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }
    
    /// Check if has errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    /// Check if has warnings
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

impl Default for ValidationResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Display implementations
impl Display for SecurityLevel {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            SecurityLevel::Level1 => write!(f, "Level 1"),
            SecurityLevel::Level2 => write!(f, "Level 2"),
            SecurityLevel::Level3 => write!(f, "Level 3"),
            SecurityLevel::Level4 => write!(f, "Level 4"),
            SecurityLevel::Level5 => write!(f, "Level 5"),
            SecurityLevel::Basic => write!(f, "Basic"),
            SecurityLevel::Standard => write!(f, "Standard"),
            SecurityLevel::High => write!(f, "High"),
            SecurityLevel::QuantumSafe => write!(f, "Quantum-Safe"),
            SecurityLevel::Maximum => write!(f, "Maximum"),
        }
    }
}

impl Display for SignatureType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            SignatureType::Dilithium => write!(f, "CRYSTALS-Dilithium"),
            SignatureType::Falcon => write!(f, "FALCON"),
            SignatureType::SphincsPlus => write!(f, "SPHINCS+"),
            SignatureType::ECDSA => write!(f, "ECDSA"),
            SignatureType::RSAPSS => write!(f, "RSA-PSS"),
            SignatureType::Ed25519 => write!(f, "Ed25519"),
        }
    }
}

impl Display for KeyType {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            KeyType::Symmetric => write!(f, "Symmetric"),
            KeyType::AsymmetricPrivate => write!(f, "Asymmetric Private"),
            KeyType::AsymmetricPublic => write!(f, "Asymmetric Public"),
            KeyType::MacKey => write!(f, "MAC Key"),
            KeyType::DerivationKey => write!(f, "Derivation Key"),
            KeyType::QuantumKey => write!(f, "Quantum Key"),
        }
    }
}

impl Display for AgentStatus {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            AgentStatus::Online => write!(f, "Online"),
            AgentStatus::Offline => write!(f, "Offline"),
            AgentStatus::Suspended => write!(f, "Suspended"),
            AgentStatus::Compromised => write!(f, "Compromised"),
            AgentStatus::Maintenance => write!(f, "Maintenance"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_secure_bytes() {
        let data = vec![1, 2, 3, 4, 5];
        let secure = SecureBytes::new(data.clone());
        
        assert_eq!(secure.len(), 5);
        assert!(!secure.is_empty());
        assert_eq!(secure.expose(), &data);
        
        let (left, right) = secure.split_at(2);
        assert_eq!(left.expose(), &[1, 2]);
        assert_eq!(right.expose(), &[3, 4, 5]);
    }
    
    #[test]
    fn test_secure_bytes_xor() {
        let a = SecureBytes::new(vec![0xFF, 0x00, 0xAA]);
        let b = SecureBytes::new(vec![0x0F, 0xFF, 0x55]);
        let result = a.xor(&b);
        
        assert_eq!(result.expose(), &[0xF0, 0xFF, 0xFF]);
    }
    
    #[test]
    fn test_expiring_value() {
        let value = ExpiringValue::new("test".to_string(), chrono::Duration::seconds(1));
        
        assert!(!value.is_expired());
        assert!(value.remaining_ttl().num_seconds() > 0);
        assert!(value.age().num_milliseconds() >= 0);
    }
    
    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new();
        assert!(result.valid);
        assert!(!result.has_errors());
        
        result.add_error("Test error".to_string());
        assert!(!result.valid);
        assert!(result.has_errors());
        
        result.add_warning("Test warning".to_string());
        assert!(result.has_warnings());
    }
    
    #[test]
    fn test_quantum_signature() {
        let signature = QuantumSignature {
            signature_type: SignatureType::Dilithium,
            signature_data: vec![1, 2, 3, 4],
            public_key_id: Some("key123".to_string()),
            algorithm_parameters: None,
            timestamp: chrono::Utc::now(),
            signer_info: None,
        };
        
        assert_eq!(signature.signature_type, SignatureType::Dilithium);
        assert_eq!(signature.signature_data, vec![1, 2, 3, 4]);
        assert_eq!(signature.public_key_id, Some("key123".to_string()));
    }
    
    #[test]
    fn test_security_level_ordering() {
        assert!(SecurityLevel::Basic < SecurityLevel::Standard);
        assert!(SecurityLevel::Standard < SecurityLevel::High);
        assert!(SecurityLevel::High < SecurityLevel::QuantumSafe);
        assert!(SecurityLevel::QuantumSafe < SecurityLevel::Maximum);
    }
    
    #[test]
    fn test_display_implementations() {
        assert_eq!(SecurityLevel::QuantumSafe.to_string(), "Quantum-Safe");
        assert_eq!(SignatureType::Dilithium.to_string(), "CRYSTALS-Dilithium");
        assert_eq!(KeyType::Symmetric.to_string(), "Symmetric");
        assert_eq!(AgentStatus::Online.to_string(), "Online");
    }
}