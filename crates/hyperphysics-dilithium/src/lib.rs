//! Post-Quantum Cryptography for HyperPhysics
//!
//! This crate provides CRYSTALS-Dilithium (ML-DSA) lattice-based digital signatures
//! for quantum-resistant authentication of consciousness networks.
//!
//! # Research Foundation
//!
//! Based on peer-reviewed research:
//! - Ducas et al. (2018): CRYSTALS-Dilithium specification (IACR TCHES)
//! - NIST FIPS 204 (2024): ML-DSA standard
//! - Nejatollahi et al. (2019): Lattice cryptography implementations (ACM)
//!
//! # Security
//!
//! - Quantum-resistant: Based on Module-LWE and Module-SIS hardness
//! - Provable security: Reduction to worst-case lattice problems
//! - Side-channel resistant: Constant-time implementations
//!
//! # Example
//!
//! ```rust
//! use hyperphysics_dilithium::*;
//!
//! # async fn example() -> Result<()> {
//! // Generate quantum-resistant keypair
//! let keypair = DilithiumKeypair::generate(SecurityLevel::High)?;
//!
//! // Sign consciousness state
//! let message = b"consciousness emergence detected";
//! let signature = keypair.sign(message)?;
//!
//! // Verify signature
//! assert!(keypair.verify(message, &signature)?);
//! # Ok(())
//! # }
//! ```


pub use hyperphysics_core::{Result, EngineError};
pub use hyperphysics_consciousness::{EmergenceEvent, HierarchicalResult};

pub mod keypair;
pub mod signature;
pub mod parameters;
pub mod lattice;
pub mod verification;
pub mod crypto_pbit;
pub mod crypto_lattice;
pub mod secure_channel;
pub mod zk_proofs;

#[cfg(feature = "gpu-acceleration")]
pub mod gpu;

#[cfg(feature = "hybrid-mode")]
pub mod hybrid;

// Re-exports
pub use keypair::DilithiumKeypair;
pub use signature::DilithiumSignature;
pub use parameters::SecurityLevel;
pub use verification::ConsciousnessAuthenticator;
pub use crypto_pbit::{CryptographicPBit, HyperbolicPoint, SignedPBitState};
pub use crypto_lattice::{CryptoLattice, SignedLatticeState};
pub use secure_channel::{SecureGPUChannel, SecureMessage, KyberKeypair, KyberCiphertext};
pub use zk_proofs::{PhiProof, ConsciousnessQualityProof, EmergenceProof};

/// Dilithium-specific error types
#[derive(Debug, thiserror::Error)]
pub enum DilithiumError {
    #[error("Key generation failed: {0}")]
    KeyGenerationFailed(String),
    
    #[error("Signature generation failed: {0}")]
    SignatureFailed(String),
    
    #[error("Signature verification failed: {0}")]
    VerificationFailed(String),
    
    #[error("Invalid signature")]
    InvalidSignature,
    
    #[error("Invalid security level")]
    InvalidSecurityLevel,
    
    #[error("Lattice operation failed: {0}")]
    LatticeOperationFailed(String),
    
    #[error("GPU acceleration error: {0}")]
    GPUError(String),
    
    #[error("Serialization error: {0}")]
    SerializationError(String),
    
    #[error("Invalid probability: {value} (must be in [0, 1])")]
    InvalidProbability { value: f64 },
    
    #[error("Missing signature")]
    MissingSignature,
    
    #[error("Timestamp error")]
    TimestampError,
    
    #[error("Invalid position: {position:?}")]
    InvalidPosition { position: (i64, i64) },
    
    #[error("Neighborhood inconsistent at position: {position:?}")]
    NeighborhoodInconsistent { position: (i64, i64) },
    
    #[error("Lattice integrity compromised: {invalid_count} invalid signatures at positions: {positions:?}")]
    LatticeIntegrityCompromised {
        invalid_count: usize,
        positions: Vec<(i64, i64)>,
    },
    
    #[error("Channel not established")]
    ChannelNotEstablished,
    
    #[error("Encryption failed")]
    EncryptionFailed,
    
    #[error("Decryption failed")]
    DecryptionFailed,
    
    #[error("Invalid channel")]
    InvalidChannel,
    
    #[error("ZK property not satisfied: {property}")]
    ZKPropertyNotSatisfied { property: String },
    
    #[error("ZK proof generation failed")]
    ZKProofGenerationFailed,
    
    #[error("ZK verification failed")]
    ZKVerificationFailed,
}

impl From<DilithiumError> for EngineError {
    fn from(err: DilithiumError) -> Self {
        // Map to appropriate EngineError variant
        EngineError::Internal {
            message: format!("Dilithium cryptography error: {}", err),
        }
    }
}

/// Post-quantum signature result type
pub type DilithiumResult<T> = std::result::Result<T, DilithiumError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_sign_verify() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Key generation failed");
        
        let message = b"test message";
        let signature = keypair.sign(message)
            .expect("Signing failed");
        
        assert!(keypair.verify(message, &signature)
            .expect("Verification failed"));
    }
    
    #[test]
    fn test_invalid_signature_fails() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Key generation failed");
        
        let message = b"test message";
        let mut signature = keypair.sign(message)
            .expect("Signing failed");
        
        // Corrupt signature
        signature.signature_bytes[0] ^= 1;
        
        assert!(!keypair.verify(message, &signature)
            .unwrap_or(false));
    }
}
