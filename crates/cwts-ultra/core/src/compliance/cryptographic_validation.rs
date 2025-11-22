//! Cryptographic Validation for SEC Rule 15c3-5 Compliance
//! 
//! Implements mathematically rigorous cryptographic validation using
//! industry-standard algorithms for audit integrity and authorization

use std::time::SystemTime;
use serde::{Serialize, Deserialize};
use uuid::Uuid;

/// Ed25519 digital signature implementation
pub struct DigitalSignatureValidator {
    keypair: ed25519_dalek::Keypair,
}

impl DigitalSignatureValidator {
    pub fn new() -> Self {
        use rand::rngs::OsRng;
        let mut csprng = OsRng{};
        let keypair = ed25519_dalek::Keypair::generate(&mut csprng);
        
        Self { keypair }
    }
    
    /// Generate a cryptographically secure signature
    pub fn sign(&self, message: &[u8]) -> String {
        use ed25519_dalek::Signer;
        let signature = self.keypair.sign(message);
        hex::encode(signature.to_bytes())
    }
    
    /// Verify a digital signature
    pub fn verify(&self, message: &[u8], signature_hex: &str) -> bool {
        use ed25519_dalek::{PublicKey, Signature, Verifier};
        
        if let Ok(signature_bytes) = hex::decode(signature_hex) {
            if let Ok(signature) = Signature::try_from(signature_bytes.as_slice()) {
                return self.keypair.public.verify(message, &signature).is_ok();
            }
        }
        false
    }
}

/// BLAKE3 cryptographic hash implementation
pub struct CryptographicHasher;

impl CryptographicHasher {
    /// Calculate BLAKE3 hash with nanosecond timestamp
    pub fn hash_with_timestamp(data: &[u8]) -> String {
        let mut hasher = sha2::Sha256::new();
        hasher.update(data);
        
        // Add nanosecond timestamp for uniqueness
        let timestamp = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        hasher.update(&timestamp.to_le_bytes());
        
        hex::encode(hasher.finalize().as_bytes())
    }
    
    /// Calculate deterministic hash without timestamp
    pub fn hash_deterministic(data: &[u8]) -> String {
        let mut hasher = sha2::Sha256::new();
        hasher.update(data);
        hex::encode(hasher.finalize().as_bytes())
    }
    
    /// Calculate composite hash from multiple data sources
    pub fn hash_composite(data_sources: &[&[u8]]) -> String {
        let mut hasher = sha2::Sha256::new();
        
        for data in data_sources {
            hasher.update(data);
        }
        
        hex::encode(hasher.finalize().as_bytes())
    }
}

/// Authorization validation with cryptographic proofs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthorizationProof {
    pub user_id: String,
    pub timestamp: SystemTime,
    pub action: String,
    pub signature: String,
    pub hash_chain: Vec<String>,
}

impl AuthorizationProof {
    /// Create new authorization proof
    pub fn new(
        user_id: String,
        action: String,
        validator: &DigitalSignatureValidator,
    ) -> Self {
        let timestamp = SystemTime::now();
        let message = format!("{}:{}:{}", user_id, action, 
            timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos());
        
        let signature = validator.sign(message.as_bytes());
        
        // Create hash chain for audit trail
        let hash_chain = vec![
            CryptographicHasher::hash_deterministic(user_id.as_bytes()),
            CryptographicHasher::hash_deterministic(action.as_bytes()),
            CryptographicHasher::hash_with_timestamp(message.as_bytes()),
        ];
        
        Self {
            user_id,
            timestamp,
            action,
            signature,
            hash_chain,
        }
    }
    
    /// Verify authorization proof integrity
    pub fn verify(&self, validator: &DigitalSignatureValidator) -> bool {
        let message = format!("{}:{}:{}", 
            self.user_id, 
            self.action,
            self.timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        );
        
        validator.verify(message.as_bytes(), &self.signature)
    }
}

/// Comprehensive audit hash calculation
pub struct AuditHashCalculator;

impl AuditHashCalculator {
    /// Calculate hash for order validation
    pub fn calculate_order_hash(
        order_id: &Uuid,
        client_id: &str,
        risk_checks: &[String],
        timestamp: SystemTime,
    ) -> String {
        let data_sources = vec![
            order_id.as_bytes(),
            client_id.as_bytes(),
            &timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos().to_le_bytes(),
        ];
        
        let mut all_data = Vec::new();
        for source in data_sources {
            all_data.extend_from_slice(source);
        }
        
        // Add risk check data
        for check in risk_checks {
            all_data.extend_from_slice(check.as_bytes());
        }
        
        CryptographicHasher::hash_composite(&[&all_data])
    }
    
    /// Calculate hash for kill switch events
    pub fn calculate_kill_switch_hash(
        event_id: &Uuid,
        trigger_type: &str,
        triggered_by: &str,
        affected_orders: &[Uuid],
        timestamp: SystemTime,
    ) -> String {
        let mut data = Vec::new();
        
        data.extend_from_slice(event_id.as_bytes());
        data.extend_from_slice(trigger_type.as_bytes());
        data.extend_from_slice(triggered_by.as_bytes());
        data.extend_from_slice(&timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos().to_le_bytes());
        
        for order_id in affected_orders {
            data.extend_from_slice(order_id.as_bytes());
        }
        
        CryptographicHasher::hash_deterministic(&data)
    }
    
    /// Calculate hash for risk limit updates
    pub fn calculate_limits_hash(
        client_id: &str,
        limits_data: &str,
        timestamp: SystemTime,
    ) -> String {
        let data_sources = vec![
            client_id.as_bytes(),
            limits_data.as_bytes(),
            &timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos().to_le_bytes(),
        ];
        
        CryptographicHasher::hash_composite(&data_sources)
    }
    
    /// Calculate hash for position updates
    pub fn calculate_position_hash(
        instrument_id: &str,
        position_data: &str,
        timestamp: SystemTime,
    ) -> String {
        let data_sources = vec![
            instrument_id.as_bytes(),
            position_data.as_bytes(),
            &timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos().to_le_bytes(),
        ];
        
        CryptographicHasher::hash_composite(&data_sources)
    }
    
    /// Calculate hash for deactivation events
    pub fn calculate_deactivation_hash(
        authorized_by: &str,
        timestamp: &SystemTime,
    ) -> String {
        let message = format!("DEACTIVATION:{}:{}", 
            authorized_by,
            timestamp.duration_since(SystemTime::UNIX_EPOCH).unwrap().as_nanos()
        );
        
        CryptographicHasher::hash_with_timestamp(message.as_bytes())
    }
}

/// Secure random number generation for cryptographic operations
pub struct SecureRandomGenerator;

impl SecureRandomGenerator {
    /// Generate cryptographically secure random bytes
    pub fn generate_bytes(length: usize) -> Vec<u8> {
        use rand::{RngCore, rngs::OsRng};
        let mut bytes = vec![0u8; length];
        OsRng.fill_bytes(&mut bytes);
        bytes
    }
    
    /// Generate secure session token
    pub fn generate_session_token() -> String {
        let bytes = Self::generate_bytes(32);
        hex::encode(bytes)
    }
    
    /// Generate secure nonce
    pub fn generate_nonce() -> u64 {
        use rand::{RngCore, rngs::OsRng};
        OsRng.next_u64()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_digital_signature() {
        let validator = DigitalSignatureValidator::new();
        let message = b"test authorization message";
        
        let signature = validator.sign(message);
        assert_eq!(signature.len(), 128); // Ed25519 signature is 64 bytes = 128 hex chars
        
        let is_valid = validator.verify(message, &signature);
        assert!(is_valid);
        
        // Test invalid signature
        let invalid_signature = "0".repeat(128);
        let is_invalid = validator.verify(message, &invalid_signature);
        assert!(!is_invalid);
    }
    
    #[test]
    fn test_sha256_hash() {
        let data = b"test data for hashing";
        let hash = CryptographicHasher::hash_deterministic(data);
        
        assert_eq!(hash.len(), 64); // BLAKE3 produces 32-byte hash = 64 hex chars
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        
        // Same input should produce same hash
        let hash2 = CryptographicHasher::hash_deterministic(data);
        assert_eq!(hash, hash2);
    }
    
    #[test]
    fn test_authorization_proof() {
        let validator = DigitalSignatureValidator::new();
        let proof = AuthorizationProof::new(
            "test_user".to_string(),
            "activate_kill_switch".to_string(),
            &validator,
        );
        
        assert!(proof.verify(&validator));
        assert_eq!(proof.hash_chain.len(), 3);
    }
    
    #[test]
    fn test_audit_hash_calculation() {
        let order_id = Uuid::new_v4();
        let hash = AuditHashCalculator::calculate_order_hash(
            &order_id,
            "test_client",
            &vec!["check1".to_string(), "check2".to_string()],
            SystemTime::now(),
        );
        
        assert_eq!(hash.len(), 64);
        assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
    }
    
    #[test]
    fn test_secure_random_generation() {
        let bytes1 = SecureRandomGenerator::generate_bytes(32);
        let bytes2 = SecureRandomGenerator::generate_bytes(32);
        
        assert_eq!(bytes1.len(), 32);
        assert_eq!(bytes2.len(), 32);
        assert_ne!(bytes1, bytes2); // Should be different
        
        let token = SecureRandomGenerator::generate_session_token();
        assert_eq!(token.len(), 64); // 32 bytes = 64 hex chars
    }
}