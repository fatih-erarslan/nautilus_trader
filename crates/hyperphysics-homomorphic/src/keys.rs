//! Cryptographic Key Management
//!
//! This module provides wrappers for BFV encryption keys:
//! - Public keys (for encryption)
//! - Secret keys (for decryption)
//! - Relinearization keys (for reducing ciphertext size after multiplication)
//! - Galois keys (for rotations and permutations)

use crate::{HomomorphicError, Result};
use seal_fhe::{
    PublicKey as SealPublicKey,
    SecretKey as SealSecretKey,
    RelinKeys as SealRelinKeys,
    GaloisKeys as SealGaloisKeys,
};
use serde::{Deserialize, Serialize};

/// Public key for BFV encryption
///
/// Used to encrypt plaintexts. Can be safely shared.
#[derive(Debug, Clone)]
pub struct PublicKey {
    inner: SealPublicKey,
}

impl PublicKey {
    pub(crate) fn new(inner: SealPublicKey) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &SealPublicKey {
        &self.inner
    }
}

/// Secret key for BFV decryption
///
/// Used to decrypt ciphertexts. Must be kept private.
#[derive(Debug, Clone)]
pub struct SecretKey {
    inner: SealSecretKey,
}

impl SecretKey {
    pub(crate) fn new(inner: SealSecretKey) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &SealSecretKey {
        &self.inner
    }
}

/// Relinearization keys for reducing ciphertext size
///
/// After homomorphic multiplication, ciphertexts grow in size.
/// Relinearization keys allow reducing them back to standard size.
#[derive(Debug, Clone)]
pub struct RelinearizationKeys {
    inner: SealRelinKeys,
}

impl RelinearizationKeys {
    pub(crate) fn new(inner: SealRelinKeys) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &SealRelinKeys {
        &self.inner
    }
}

/// Galois keys for rotations and permutations
///
/// Enable rotation operations on batched ciphertexts.
#[derive(Debug, Clone)]
pub struct GaloisKeys {
    inner: SealGaloisKeys,
}

impl GaloisKeys {
    pub(crate) fn new(inner: SealGaloisKeys) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &SealGaloisKeys {
        &self.inner
    }
}

/// Complete key set for BFV operations
#[derive(Debug, Clone)]
pub struct KeySet {
    pub public_key: PublicKey,
    pub secret_key: SecretKey,
    pub relin_keys: Option<RelinearizationKeys>,
    pub galois_keys: Option<GaloisKeys>,
}

impl KeySet {
    /// Create a new key set
    pub fn new(
        public_key: PublicKey,
        secret_key: SecretKey,
        relin_keys: Option<RelinearizationKeys>,
        galois_keys: Option<GaloisKeys>,
    ) -> Self {
        Self {
            public_key,
            secret_key,
            relin_keys,
            galois_keys,
        }
    }

    /// Check if relinearization keys are available
    pub fn has_relin_keys(&self) -> bool {
        self.relin_keys.is_some()
    }

    /// Check if Galois keys are available
    pub fn has_galois_keys(&self) -> bool {
        self.galois_keys.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_key_set_creation() {
        // This test will be expanded once we can actually generate keys
        // For now, just verify the API compiles
    }
}
