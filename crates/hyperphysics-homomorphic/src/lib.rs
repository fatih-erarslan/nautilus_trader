//! # Homomorphic Encryption for Privacy-Preserving Consciousness Metrics
//!
//! This crate implements privacy-preserving computation of consciousness metrics (Φ)
//! using the BFV (Brakerski-Fan-Vercauteren) fully homomorphic encryption scheme.
//!
//! ## Mathematical Foundation
//!
//! The BFV scheme enables computation on encrypted integers:
//! - **Encryption**: c = Enc(pk, m) where m is a plaintext message
//! - **Homomorphic Addition**: Enc(m₁) + Enc(m₂) = Enc(m₁ + m₂)
//! - **Homomorphic Multiplication**: Enc(m₁) × Enc(m₂) = Enc(m₁ × m₂)
//! - **Decryption**: m = Dec(sk, c)
//!
//! ## Security
//!
//! Security based on Ring-LWE (Ring Learning With Errors) problem:
//! - Quantum-resistant (post-quantum cryptography)
//! - Recommended parameters for 128-bit security
//! - Noise management through modulus switching
//!
//! ## Applications
//!
//! - **Private Φ Computation**: Calculate consciousness metrics without revealing pBit states
//! - **Secure Multi-Party Verification**: Multiple parties verify consciousness without sharing data
//! - **Encrypted Tessellation States**: Homomorphic operations on hyperbolic tile states
//!
//! ## References
//!
//! - Fan & Vercauteren (2012) "Somewhat Practical Fully Homomorphic Encryption"
//! - Microsoft SEAL: https://github.com/microsoft/SEAL
//! - Brakerski (2012) "Fully Homomorphic Encryption without Modulus Switching"
//!
//! ## Example
//!
//! ```rust,no_run
//! use hyperphysics_homomorphic::{BfvParameters, EncryptedPhi};
//!
//! // Generate encryption parameters
//! let params = BfvParameters::default_128bit_security()?;
//!
//! // Generate keys
//! let (public_key, secret_key) = params.generate_keys()?;
//!
//! // Encrypt a consciousness metric
//! let phi_value = 42;
//! let encrypted_phi = params.encrypt(phi_value, &public_key)?;
//!
//! // Perform homomorphic operations
//! let doubled = encrypted_phi.mul_plain(2)?;
//!
//! // Decrypt
//! let result = params.decrypt(&doubled, &secret_key)?;
//! assert_eq!(result, 84);
//! # Ok::<(), hyperphysics_homomorphic::HomomorphicError>(())
//! ```

pub mod bfv;
pub mod encrypted_phi;
pub mod keys;
pub mod parameters;

pub use bfv::BfvContext;
pub use encrypted_phi::{EncryptedPhi, EncryptedState};
pub use keys::{PublicKey, SecretKey, RelinearizationKeys};
pub use parameters::BfvParameters;

use thiserror::Error;

/// Errors that can occur during homomorphic encryption operations
#[derive(Error, Debug)]
pub enum HomomorphicError {
    #[error("Encryption failed: {message}")]
    EncryptionFailure { message: String },

    #[error("Decryption failed: {message}")]
    DecryptionFailure { message: String },

    #[error("Parameter generation failed: {message}")]
    ParameterError { message: String },

    #[error("Key generation failed: {message}")]
    KeyGenerationError { message: String },

    #[error("Homomorphic operation failed: {operation}")]
    OperationFailure { operation: String },

    #[error("Noise budget exhausted (remaining: {remaining} bits)")]
    NoiseBudgetExhausted { remaining: i32 },

    #[error("Invalid plaintext value: {message}")]
    InvalidPlaintext { message: String },

    #[error("SEAL library error: {message}")]
    SealError { message: String },

    #[error("Serialization error: {message}")]
    SerializationError { message: String },
}

pub type Result<T> = std::result::Result<T, HomomorphicError>;

/// Default security level (128-bit post-quantum security)
pub const DEFAULT_SECURITY_LEVEL: usize = 128;

/// Default polynomial modulus degree (affects performance and security)
pub const DEFAULT_POLY_MODULUS_DEGREE: usize = 8192;

/// Default plaintext modulus (for BFV arithmetic)
pub const DEFAULT_PLAIN_MODULUS: u64 = 1024;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_constants() {
        assert_eq!(DEFAULT_SECURITY_LEVEL, 128);
        assert!(DEFAULT_POLY_MODULUS_DEGREE >= 4096, "Poly modulus degree too small for security");
        assert!(DEFAULT_PLAIN_MODULUS > 0, "Plaintext modulus must be positive");
    }
}
