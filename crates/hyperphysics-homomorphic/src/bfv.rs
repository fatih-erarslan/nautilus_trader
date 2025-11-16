//! BFV Context and Operations
//!
//! This module provides a high-level wrapper around the fhe.rs BFV implementation,
//! managing the encryption context and providing convenient operations.

use crate::{
    BfvParameters, HomomorphicError, Result,
    PublicKey, SecretKey, KeySet,
};
use fhe::bfv::{
    BfvParameters as FheBfvParameters, Ciphertext, Encoding, Plaintext,
    PublicKey as FhePublicKey, SecretKey as FheSecretKey,
};
use fhe_traits::{FheEncoder, FheEncrypter, FheDecrypter, FheDecoder};
use std::sync::Arc;
use rand::thread_rng;

/// BFV encryption context
///
/// Manages all the state needed for BFV operations including
/// parameters and crypto primitives.
pub struct BfvContext {
    params: BfvParameters,
    fhe_params: Arc<FheBfvParameters>,
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
        let fhe_params = params.build()?;
        Ok(Self { params, fhe_params })
    }

    /// Get the encryption parameters
    pub fn parameters(&self) -> &BfvParameters {
        &self.params
    }

    /// Get the fhe.rs parameters (low-level access)
    #[allow(dead_code)]
    pub(crate) fn fhe_parameters(&self) -> &Arc<FheBfvParameters> {
        &self.fhe_params
    }

    /// Generate a fresh key set
    ///
    /// This generates:
    /// - Public key (for encryption)
    /// - Secret key (for decryption)
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
        let mut rng = thread_rng();

        let secret_key = FheSecretKey::random(&self.fhe_params, &mut rng);
        let public_key = FhePublicKey::new(&secret_key, &mut rng);

        Ok(KeySet::new(
            PublicKey::new(public_key),
            SecretKey::new(secret_key),
        ))
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
        let mut rng = thread_rng();

        // Encode the value as a plaintext
        let plaintext = Plaintext::try_encode(&[value], Encoding::poly(), &self.fhe_params)
            .map_err(|e| HomomorphicError::EncryptionFailure {
                message: format!("Failed to encode value {}: {:?}", value, e),
            })?;

        // Encrypt using the public key
        let ciphertext = public_key.inner().try_encrypt(&plaintext, &mut rng)
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
        // Decrypt
        let plaintext = secret_key.inner().try_decrypt(ciphertext)
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to decrypt: {:?}", e),
            })?;

        // Decode
        let values = Vec::<i64>::try_decode(&plaintext, Encoding::poly())
            .map_err(|e| HomomorphicError::DecryptionFailure {
                message: format!("Failed to decode plaintext: {:?}", e),
            })?;

        values.get(0).copied().ok_or_else(|| HomomorphicError::DecryptionFailure {
            message: "Decrypted plaintext is empty".to_string(),
        })
    }

    /// Add two ciphertexts homomorphically
    ///
    /// Computes Enc(a + b) from Enc(a) and Enc(b)
    pub fn add(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Ok(a + b)
    }

    /// Subtract two ciphertexts homomorphically
    ///
    /// Computes Enc(a - b) from Enc(a) and Enc(b)
    pub fn sub(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Ok(a - b)
    }

    /// Multiply two ciphertexts homomorphically
    ///
    /// Computes Enc(a * b) from Enc(a) and Enc(b)
    pub fn mul(&self, a: &Ciphertext, b: &Ciphertext) -> Result<Ciphertext> {
        Ok(a * b)
    }

    /// Multiply a ciphertext by a plaintext scalar
    ///
    /// Computes Enc(a * scalar) from Enc(a) and scalar
    pub fn mul_plain(&self, ciphertext: &Ciphertext, scalar: i64) -> Result<Ciphertext> {
        // Encode the scalar
        let plaintext = Plaintext::try_encode(&[scalar], Encoding::poly(), &self.fhe_params)
            .map_err(|e| HomomorphicError::OperationFailure {
                operation: format!("encoding scalar {}: {:?}", scalar, e),
            })?;

        // Multiply
        Ok(ciphertext * &plaintext)
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
        let _keys = ctx.generate_keys()?;
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
    fn test_homomorphic_addition() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let a = 10i64;
        let b = 32i64;

        let enc_a = ctx.encrypt(a, &keys.public_key)?;
        let enc_b = ctx.encrypt(b, &keys.public_key)?;

        let enc_sum = ctx.add(&enc_a, &enc_b)?;
        let decrypted_sum = ctx.decrypt(&enc_sum, &keys.secret_key)?;

        assert_eq!(a + b, decrypted_sum);
        Ok(())
    }

    #[test]
    fn test_homomorphic_subtraction() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let a = 50i64;
        let b = 8i64;

        let enc_a = ctx.encrypt(a, &keys.public_key)?;
        let enc_b = ctx.encrypt(b, &keys.public_key)?;

        let enc_diff = ctx.sub(&enc_a, &enc_b)?;
        let decrypted_diff = ctx.decrypt(&enc_diff, &keys.secret_key)?;

        assert_eq!(a - b, decrypted_diff);
        Ok(())
    }

    #[test]
    fn test_homomorphic_multiplication() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let a = 20i64;
        let b = -7i64;

        let enc_a = ctx.encrypt(a, &keys.public_key)?;
        let enc_b = ctx.encrypt(b, &keys.public_key)?;

        let enc_product = ctx.mul(&enc_a, &enc_b)?;
        let decrypted_product = ctx.decrypt(&enc_product, &keys.secret_key)?;

        assert_eq!(a * b, decrypted_product);
        Ok(())
    }

    #[test]
    fn test_mul_plain() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let value = 15i64;
        let scalar = 3i64;

        let enc_value = ctx.encrypt(value, &keys.public_key)?;
        let enc_result = ctx.mul_plain(&enc_value, scalar)?;
        let decrypted_result = ctx.decrypt(&enc_result, &keys.secret_key)?;

        assert_eq!(value * scalar, decrypted_result);
        Ok(())
    }

    #[test]
    fn test_multiple_operations() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        // Compute (10 + 5) * 3 = 45
        let enc_10 = ctx.encrypt(10, &keys.public_key)?;
        let enc_5 = ctx.encrypt(5, &keys.public_key)?;
        let enc_3 = ctx.encrypt(3, &keys.public_key)?;

        let enc_sum = ctx.add(&enc_10, &enc_5)?;
        let enc_result = ctx.mul(&enc_sum, &enc_3)?;

        let result = ctx.decrypt(&enc_result, &keys.secret_key)?;
        assert_eq!(45, result);

        Ok(())
    }
}
