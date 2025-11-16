//! Cryptographic Key Management
//!
//! This module provides wrappers for BFV encryption keys:
//! - Public keys (for encryption)
//! - Secret keys (for decryption)
//! - Relinearization keys (for reducing ciphertext size after multiplication)

use fhe::bfv::{PublicKey as FhePublicKey, SecretKey as FheSecretKey};

/// Public key for BFV encryption
///
/// Used to encrypt plaintexts. Can be safely shared.
#[derive(Debug, Clone)]
pub struct PublicKey {
    pub(crate) inner: FhePublicKey,
}

impl PublicKey {
    pub(crate) fn new(inner: FhePublicKey) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &FhePublicKey {
        &self.inner
    }
}

/// Secret key for BFV decryption
///
/// Used to decrypt ciphertexts. Must be kept private.
#[derive(Debug, Clone)]
pub struct SecretKey {
    pub(crate) inner: FheSecretKey,
}

impl SecretKey {
    pub(crate) fn new(inner: FheSecretKey) -> Self {
        Self { inner }
    }

    pub(crate) fn inner(&self) -> &FheSecretKey {
        &self.inner
    }
}

/// Complete key set for BFV operations
#[derive(Debug, Clone)]
pub struct KeySet {
    pub public_key: PublicKey,
    pub secret_key: SecretKey,
}

impl KeySet {
    /// Create a new key set
    pub fn new(public_key: PublicKey, secret_key: SecretKey) -> Self {
        Self {
            public_key,
            secret_key,
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_key_set_creation() {
        // This test will be expanded when we can actually generate keys
        // For now, just verify the API compiles
    }
}
