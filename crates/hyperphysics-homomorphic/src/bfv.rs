//! BFV Context and Operations
//!
//! This module provides a high-level wrapper around the SEAL BFV implementation,
//! managing the encryption context and providing convenient operations.

use crate::{
    BfvParameters, HomomorphicError, Result,
    PublicKey, SecretKey, RelinearizationKeys, KeySet,
};
use seal_fhe::{
    Context, KeyGenerator, Encryptor, Decryptor, BFVEvaluator,
    Plaintext, Ciphertext, BFVEncoder,
};

/// BFV encryption context
///
/// Manages all the state needed for BFV operations including
/// parameters, keys, and crypto primitives.
pub struct BfvContext {
    params: BfvParameters,
    context: Context,
}

impl BfvContext {
    /// Create a new BFV context from parameters
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use hyperphysics_homomorphic::{BfvContext, BfvParameters};
    ///
    /// let params = BfvParameters::default_128bit_security()?;
    /// let ctx = BfvContext::new(params)?;
    /// # Ok::<(), hyperphysics_homomorphic::HomomorphicError>(())
    /// ```
    pub fn new(params: BfvParameters) -> Result<Self> {
        let context = params.build_context()?;
        Ok(Self { params, context })
    }

    /// Get the encryption parameters
    pub fn parameters(&self) -> &BfvParameters {
        &self.params
    }

    /// Get the SEAL context (low-level access)
    pub(crate) fn seal_context(&self) -> &Context {
        &self.context
    }

    /// Generate a fresh key set
    ///
    /// This generates:
    /// - Public key (for encryption)
    /// - Secret key (for decryption)
    /// - Relinearization keys (for ciphertext size reduction)
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use hyperphysics_homomorphic::{BfvContext, BfvParameters};
    ///
    /// let params = BfvParameters::default_128bit_security()?;
    /// let ctx = BfvContext::new(params)?;
    /// let keys = ctx.generate_keys()?;
    /// # Ok::<(), hyperphysics_homomorphic::HomomorphicError>(())
    /// ```
    pub fn generate_keys(&self) -> Result<KeySet> {
        let keygen = KeyGenerator::new(&self.context)
            .map_err(|e| HomomorphicError::KeyGenerationError {
                message: format!("Failed to create key generator: {:?}", e),
            })?;

        let secret_key = SecretKey::new(keygen.secret_key());
        let public_key = PublicKey::new(keygen.create_public_key()
            .map_err(|e| HomomorphicError::KeyGenerationError {
                message: format!("Failed to create public key: {:?}", e),
            })?);

        let relin_keys = Some(RelinearizationKeys::new(keygen.create_relin_keys()
            .map_err(|e| HomomorphicError::KeyGenerationError {
                message: format!("Failed to create relinearization keys: {:?}", e),
            })?));

        Ok(KeySet::new(public_key, secret_key, relin_keys, None))
    }

    /// Encrypt a plaintext value
    ///
    /// # Arguments
    ///
    /// * `value` - The integer value to encrypt
    /// * `public_key` - The public key to use for encryption
    ///
    /// # Returns
    ///
    /// An encrypted ciphertext
    pub fn encrypt(&self, value: i64, public_key: &PublicKey) -> Result<Ciphertext> {
        // Create encoder
        let encoder = BFVEncoder::new(&self.context)
            .map_err(|e| HomomorphicError::EncryptionFailure {
                message: format!("Failed to create encoder: {:?}", e),
            })?;

        // Encode the value
        let plaintext = encoder.encode_signed(value)
            .map_err(|e| HomomorphicError::InvalidPlaintext {
                message: format!("Failed to encode value {}: {:?}", value, e),
            })?;

        // Create encryptor
        let encryptor = Encryptor::with_public_key(&self.context, public_key.inner())
            .map_err(|e| HomomorphicError::EncryptionFailure {
                message: format!("Failed to create encryptor: {:?}", e),
            })?;

        // Encrypt
        let ciphertext = encryptor.encrypt(&plaintext)
            .map_err(|e| HomomorphicError::EncryptionFailure {
                message: format!("Failed to encrypt: {:?}", e),
            })?;

        Ok(ciphertext)
    }

    /// Decrypt a ciphertext
    ///
    /// # Arguments
    ///
    /// * `ciphertext` - The encrypted value
    /// * `secret_key` - The secret key to use for decryption
    ///
    /// # Returns
    ///
    /// The decrypted integer value
    pub fn decrypt(&self, ciphertext: &Ciphertext, secret_key: &SecretKey) -> Result<i64> {
        // Create decryptor
        let decryptor = Decryptor::new(&self.context, secret_key.inner())
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to create decryptor: {:?}", e),
            })?;

        // Decrypt
        let plaintext = decryptor.decrypt(ciphertext)
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to decrypt: {:?}", e),
            })?;

        // Decode
        let encoder = BFVEncoder::new(&self.context)
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to create encoder: {:?}", e),
            })?;

        let value = encoder.decode_signed(&plaintext)
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to decode plaintext: {:?}", e),
            })?;

        Ok(value)
    }

    /// Check remaining noise budget in a ciphertext
    ///
    /// Returns the number of bits of noise budget remaining.
    /// When this reaches 0, the ciphertext can no longer be decrypted correctly.
    pub fn noise_budget(&self, ciphertext: &Ciphertext, secret_key: &SecretKey) -> Result<i32> {
        let decryptor = Decryptor::new(&self.context, secret_key.inner())
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to create decryptor: {:?}", e),
            })?;

        let budget = decryptor.invariant_noise_budget(ciphertext)
            .map_err(|e| HomomorphicError::OperationFailure {
                operation: format!("noise budget check: {:?}", e),
            })?;

        Ok(budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let _ctx = BfvContext::new(params)?;
        Ok(())
    }

    #[test]
    fn test_key_generation() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        assert!(keys.has_relin_keys());
        Ok(())
    }

    #[test]
    fn test_encrypt_decrypt() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let value = 42i64;
        let ciphertext = ctx.encrypt(value, &keys.public_key)?;
        let decrypted = ctx.decrypt(&ciphertext, &keys.secret_key)?;

        assert_eq!(value, decrypted);
        Ok(())
    }

    #[test]
    fn test_noise_budget() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let ciphertext = ctx.encrypt(10, &keys.public_key)?;
        let budget = ctx.noise_budget(&ciphertext, &keys.secret_key)?;

        assert!(budget > 100, "Fresh ciphertext should have substantial noise budget");
        Ok(())
    }
}
