//! ML-DSA parameter sets and security levels

use serde::{Serialize, Deserialize};

/// ML-DSA security level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(zeroize::Zeroize)]
pub enum SecurityLevel {
    /// ML-DSA-44: 128-bit quantum security
    Standard,
    /// ML-DSA-65: 192-bit quantum security (recommended)
    High,
    /// ML-DSA-87: 256-bit quantum security
    Maximum,
}

impl SecurityLevel {
    /// Get public key size in bytes
    pub fn public_key_size(&self) -> usize {
        match self {
            SecurityLevel::Standard => 1312,
            SecurityLevel::High => 1952,
            SecurityLevel::Maximum => 2592,
        }
    }
    
    /// Get secret key size in bytes
    pub fn secret_key_size(&self) -> usize {
        match self {
            SecurityLevel::Standard => 2560,
            SecurityLevel::High => 4032,
            SecurityLevel::Maximum => 4896,
        }
    }
    
    /// Get signature size in bytes
    pub fn signature_size(&self) -> usize {
        match self {
            SecurityLevel::Standard => 2420,
            SecurityLevel::High => 3293,
            SecurityLevel::Maximum => 4595,
        }
    }
    
    /// Get quantum security level in bits
    pub fn quantum_security_bits(&self) -> usize {
        match self {
            SecurityLevel::Standard => 128,
            SecurityLevel::High => 192,
            SecurityLevel::Maximum => 256,
        }
    }
}
