//! Post-Quantum Cryptographic Algorithms
//!
//! This module implements NIST-approved post-quantum cryptographic algorithms
//! including CRYSTALS-Kyber, CRYSTALS-Dilithium, FALCON, and SPHINCS+.

pub mod crystals_kyber;
pub mod crystals_dilithium;
pub mod falcon;
pub mod sphincs_plus;
pub mod manager;

pub use crystals_kyber::*;
pub use crystals_dilithium::*;
pub use falcon::*;
pub use sphincs_plus::*;
pub use manager::*;

use crate::error::QuantumSecurityError;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::fmt::{Display, Formatter, Result as FmtResult};
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Post-Quantum Cryptographic Algorithm Types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum PQCAlgorithm {
    /// CRYSTALS-Kyber - Key Encapsulation Mechanism
    Kyber512,
    Kyber768,
    Kyber1024,
    
    /// CRYSTALS-Dilithium - Digital Signature Algorithm
    Dilithium2,
    Dilithium3,
    Dilithium5,
    
    /// FALCON - Digital Signature Algorithm
    Falcon512,
    Falcon1024,
    
    /// SPHINCS+ - Stateless Hash-Based Signatures
    SphincsPlus128s,
    SphincsPlus128f,
    SphincsPlus192s,
    SphincsPlus192f,
    SphincsPlus256s,
    SphincsPlus256f,
}

impl Display for PQCAlgorithm {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            PQCAlgorithm::Kyber512 => write!(f, "CRYSTALS-Kyber-512"),
            PQCAlgorithm::Kyber768 => write!(f, "CRYSTALS-Kyber-768"),
            PQCAlgorithm::Kyber1024 => write!(f, "CRYSTALS-Kyber-1024"),
            PQCAlgorithm::Dilithium2 => write!(f, "CRYSTALS-Dilithium-2"),
            PQCAlgorithm::Dilithium3 => write!(f, "CRYSTALS-Dilithium-3"),
            PQCAlgorithm::Dilithium5 => write!(f, "CRYSTALS-Dilithium-5"),
            PQCAlgorithm::Falcon512 => write!(f, "FALCON-512"),
            PQCAlgorithm::Falcon1024 => write!(f, "FALCON-1024"),
            PQCAlgorithm::SphincsPlus128s => write!(f, "SPHINCS+-128s"),
            PQCAlgorithm::SphincsPlus128f => write!(f, "SPHINCS+-128f"),
            PQCAlgorithm::SphincsPlus192s => write!(f, "SPHINCS+-192s"),
            PQCAlgorithm::SphincsPlus192f => write!(f, "SPHINCS+-192f"),
            PQCAlgorithm::SphincsPlus256s => write!(f, "SPHINCS+-256s"),
            PQCAlgorithm::SphincsPlus256f => write!(f, "SPHINCS+-256f"),
        }
    }
}

impl Zeroize for PQCAlgorithm {
    fn zeroize(&mut self) {
        // For enum types, we can't really zero the discriminant,
        // so we just set it to a default value
        *self = PQCAlgorithm::Kyber512;
    }
}

/// Post-Quantum Cryptographic Key Types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PQCKey {
    /// Key Encapsulation Mechanism Keys
    KyberPublicKey {
        algorithm: PQCAlgorithm,
        key_data: Vec<u8>,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    KyberPrivateKey {
        algorithm: PQCAlgorithm,
        key_data: SecureBytes,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    
    /// Digital Signature Keys
    DilithiumPublicKey {
        algorithm: PQCAlgorithm,
        key_data: Vec<u8>,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    DilithiumPrivateKey {
        algorithm: PQCAlgorithm,
        key_data: SecureBytes,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    
    /// FALCON Keys
    FalconPublicKey {
        algorithm: PQCAlgorithm,
        key_data: Vec<u8>,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    FalconPrivateKey {
        algorithm: PQCAlgorithm,
        key_data: SecureBytes,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    
    /// SPHINCS+ Keys
    SphincsPlusPublicKey {
        algorithm: PQCAlgorithm,
        key_data: Vec<u8>,
        created_at: chrono::DateTime<chrono::Utc>,
    },
    SphincsPlusPrivateKey {
        algorithm: PQCAlgorithm,
        key_data: SecureBytes,
        created_at: chrono::DateTime<chrono::Utc>,
    },
}

/// Post-Quantum Cryptographic Key Pair
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PQCKeyPair {
    pub public_key: PQCKey,
    pub private_key: PQCKey,
    pub algorithm: PQCAlgorithm,
    pub key_id: Uuid,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub expires_at: Option<chrono::DateTime<chrono::Utc>>,
    pub usage: KeyUsage,
}

// KeyUsage is defined in types.rs to avoid duplication

/// Encapsulated Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncapsulatedKey {
    pub ciphertext: Vec<u8>,
    pub shared_secret: SecureBytes,
    pub algorithm: PQCAlgorithm,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Digital Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DigitalSignature {
    pub signature: Vec<u8>,
    pub algorithm: PQCAlgorithm,
    pub signer_id: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub message_hash: Vec<u8>,
}

/// Algorithm Performance Metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlgorithmMetrics {
    pub key_generation_count: u64,
    pub key_generation_time_us: u64,
    pub encapsulation_count: u64,
    pub encapsulation_time_us: u64,
    pub decapsulation_count: u64,
    pub decapsulation_time_us: u64,
    pub signature_count: u64,
    pub signature_time_us: u64,
    pub verification_count: u64,
    pub verification_time_us: u64,
    pub error_count: u64,
    pub last_operation: Option<chrono::DateTime<chrono::Utc>>,
}

/// Algorithm Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmConfig {
    pub enabled_algorithms: Vec<PQCAlgorithm>,
    pub default_kem_algorithm: PQCAlgorithm,
    pub default_signature_algorithm: PQCAlgorithm,
    pub key_rotation_interval: chrono::Duration,
    pub max_key_age: chrono::Duration,
    pub performance_monitoring: bool,
    pub security_level: SecurityLevel,
}

impl Default for AlgorithmConfig {
    fn default() -> Self {
        Self {
            enabled_algorithms: vec![
                PQCAlgorithm::Kyber1024,
                PQCAlgorithm::Dilithium5,
                PQCAlgorithm::Falcon1024,
                PQCAlgorithm::SphincsPlus256s,
            ],
            default_kem_algorithm: PQCAlgorithm::Kyber1024,
            default_signature_algorithm: PQCAlgorithm::Dilithium5,
            key_rotation_interval: chrono::Duration::hours(24),
            max_key_age: chrono::Duration::days(30),
            performance_monitoring: true,
            security_level: SecurityLevel::High,
        }
    }
}

/// Security Level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum SecurityLevel {
    /// NIST Level 1 (equivalent to AES-128)
    Level1,
    /// NIST Level 3 (equivalent to AES-192)
    Level3,
    /// NIST Level 5 (equivalent to AES-256)
    Level5,
    /// High security with performance considerations
    High,
    /// Maximum security regardless of performance
    Maximum,
}

impl PQCAlgorithm {
    /// Get the security level of the algorithm
    pub fn security_level(&self) -> SecurityLevel {
        match self {
            PQCAlgorithm::Kyber512 | PQCAlgorithm::Dilithium2 | PQCAlgorithm::Falcon512 => SecurityLevel::Level1,
            PQCAlgorithm::Kyber768 | PQCAlgorithm::Dilithium3 => SecurityLevel::Level3,
            PQCAlgorithm::Kyber1024 | PQCAlgorithm::Dilithium5 | PQCAlgorithm::Falcon1024 => SecurityLevel::Level5,
            PQCAlgorithm::SphincsPlus128s | PQCAlgorithm::SphincsPlus128f => SecurityLevel::Level1,
            PQCAlgorithm::SphincsPlus192s | PQCAlgorithm::SphincsPlus192f => SecurityLevel::Level3,
            PQCAlgorithm::SphincsPlus256s | PQCAlgorithm::SphincsPlus256f => SecurityLevel::Level5,
        }
    }
    
    /// Check if algorithm is for Key Encapsulation Mechanism
    pub fn is_kem(&self) -> bool {
        matches!(self, PQCAlgorithm::Kyber512 | PQCAlgorithm::Kyber768 | PQCAlgorithm::Kyber1024)
    }
    
    /// Check if algorithm is for Digital Signatures
    pub fn is_signature(&self) -> bool {
        !self.is_kem()
    }
    
    /// Get expected key sizes
    pub fn key_sizes(&self) -> (usize, usize) {
        match self {
            PQCAlgorithm::Kyber512 => (800, 1632),
            PQCAlgorithm::Kyber768 => (1184, 2400),
            PQCAlgorithm::Kyber1024 => (1568, 3168),
            PQCAlgorithm::Dilithium2 => (1312, 2528),
            PQCAlgorithm::Dilithium3 => (1952, 4000),
            PQCAlgorithm::Dilithium5 => (2592, 4864),
            PQCAlgorithm::Falcon512 => (897, 1281),
            PQCAlgorithm::Falcon1024 => (1793, 2305),
            PQCAlgorithm::SphincsPlus128s => (32, 64),
            PQCAlgorithm::SphincsPlus128f => (32, 64),
            PQCAlgorithm::SphincsPlus192s => (48, 96),
            PQCAlgorithm::SphincsPlus192f => (48, 96),
            PQCAlgorithm::SphincsPlus256s => (64, 128),
            PQCAlgorithm::SphincsPlus256f => (64, 128),
        }
    }
    
    /// Get expected signature size
    pub fn signature_size(&self) -> usize {
        match self {
            PQCAlgorithm::Dilithium2 => 2420,
            PQCAlgorithm::Dilithium3 => 3293,
            PQCAlgorithm::Dilithium5 => 4595,
            PQCAlgorithm::Falcon512 => 690,
            PQCAlgorithm::Falcon1024 => 1330,
            PQCAlgorithm::SphincsPlus128s => 7856,
            PQCAlgorithm::SphincsPlus128f => 17088,
            PQCAlgorithm::SphincsPlus192s => 16224,
            PQCAlgorithm::SphincsPlus192f => 35664,
            PQCAlgorithm::SphincsPlus256s => 29792,
            PQCAlgorithm::SphincsPlus256f => 49856,
            _ => 0, // KEM algorithms don't have signatures
        }
    }
}

impl PQCKey {
    /// Get the algorithm used for this key
    pub fn algorithm(&self) -> PQCAlgorithm {
        match self {
            PQCKey::KyberPublicKey { algorithm, .. } => algorithm.clone(),
            PQCKey::KyberPrivateKey { algorithm, .. } => algorithm.clone(),
            PQCKey::DilithiumPublicKey { algorithm, .. } => algorithm.clone(),
            PQCKey::DilithiumPrivateKey { algorithm, .. } => algorithm.clone(),
            PQCKey::FalconPublicKey { algorithm, .. } => algorithm.clone(),
            PQCKey::FalconPrivateKey { algorithm, .. } => algorithm.clone(),
            PQCKey::SphincsPlusPublicKey { algorithm, .. } => algorithm.clone(),
            PQCKey::SphincsPlusPrivateKey { algorithm, .. } => algorithm.clone(),
        }
    }
    
    /// Get the creation time of this key
    pub fn created_at(&self) -> chrono::DateTime<chrono::Utc> {
        match self {
            PQCKey::KyberPublicKey { created_at, .. } => *created_at,
            PQCKey::KyberPrivateKey { created_at, .. } => *created_at,
            PQCKey::DilithiumPublicKey { created_at, .. } => *created_at,
            PQCKey::DilithiumPrivateKey { created_at, .. } => *created_at,
            PQCKey::FalconPublicKey { created_at, .. } => *created_at,
            PQCKey::FalconPrivateKey { created_at, .. } => *created_at,
            PQCKey::SphincsPlusPublicKey { created_at, .. } => *created_at,
            PQCKey::SphincsPlusPrivateKey { created_at, .. } => *created_at,
        }
    }
    
    /// Check if this is a public key
    pub fn is_public(&self) -> bool {
        matches!(self, 
            PQCKey::KyberPublicKey { .. } | 
            PQCKey::DilithiumPublicKey { .. } | 
            PQCKey::FalconPublicKey { .. } | 
            PQCKey::SphincsPlusPublicKey { .. }
        )
    }
    
    /// Check if this is a private key
    pub fn is_private(&self) -> bool {
        !self.is_public()
    }
    
    /// Get key data (for public keys only)
    pub fn key_data(&self) -> Option<&[u8]> {
        match self {
            PQCKey::KyberPublicKey { key_data, .. } => Some(key_data),
            PQCKey::DilithiumPublicKey { key_data, .. } => Some(key_data),
            PQCKey::FalconPublicKey { key_data, .. } => Some(key_data),
            PQCKey::SphincsPlusPublicKey { key_data, .. } => Some(key_data),
            _ => None, // Private keys should not expose key data directly
        }
    }
}

impl PQCKeyPair {
    /// Create a new key pair
    pub fn new(
        public_key: PQCKey,
        private_key: PQCKey,
        algorithm: PQCAlgorithm,
        usage: KeyUsage,
    ) -> Self {
        Self {
            public_key,
            private_key,
            algorithm,
            key_id: Uuid::new_v4(),
            created_at: chrono::Utc::now(),
            expires_at: None,
            usage,
        }
    }
    
    /// Check if the key pair is expired
    pub fn is_expired(&self) -> bool {
        if let Some(expires_at) = self.expires_at {
            chrono::Utc::now() > expires_at
        } else {
            false
        }
    }
    
    /// Set expiration time
    pub fn set_expiration(&mut self, expires_at: chrono::DateTime<chrono::Utc>) {
        self.expires_at = Some(expires_at);
    }
    
    /// Get age in seconds
    pub fn age_seconds(&self) -> i64 {
        chrono::Utc::now().signed_duration_since(self.created_at).num_seconds()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algorithm_security_levels() {
        assert_eq!(PQCAlgorithm::Kyber512.security_level(), SecurityLevel::Level1);
        assert_eq!(PQCAlgorithm::Kyber768.security_level(), SecurityLevel::Level3);
        assert_eq!(PQCAlgorithm::Kyber1024.security_level(), SecurityLevel::Level5);
    }
    
    #[test]
    fn test_algorithm_types() {
        assert!(PQCAlgorithm::Kyber512.is_kem());
        assert!(PQCAlgorithm::Dilithium2.is_signature());
        assert!(!PQCAlgorithm::Kyber512.is_signature());
        assert!(!PQCAlgorithm::Dilithium2.is_kem());
    }
    
    #[test]
    fn test_key_sizes() {
        let (pub_size, priv_size) = PQCAlgorithm::Kyber1024.key_sizes();
        assert_eq!(pub_size, 1568);
        assert_eq!(priv_size, 3168);
    }
    
    #[test]
    fn test_signature_sizes() {
        assert_eq!(PQCAlgorithm::Dilithium5.signature_size(), 4595);
        assert_eq!(PQCAlgorithm::Falcon1024.signature_size(), 1330);
        assert_eq!(PQCAlgorithm::Kyber1024.signature_size(), 0); // KEM has no signatures
    }
}