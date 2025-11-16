//! Encrypted Consciousness Metrics
//!
//! This module provides high-level abstractions for computing consciousness
//! metrics (Φ) on encrypted data using homomorphic encryption.

use fhe::bfv::Ciphertext;

/// An encrypted consciousness metric value (Φ)
///
/// Represents an encrypted integrated information value that can be
/// computed on without decryption.
#[derive(Debug, Clone)]
pub struct EncryptedPhi {
    pub(crate) ciphertext: Ciphertext,
}

impl EncryptedPhi {
    /// Create from a raw ciphertext
    pub fn from_ciphertext(ciphertext: Ciphertext) -> Self {
        Self { ciphertext }
    }

    /// Get the underlying ciphertext
    pub fn ciphertext(&self) -> &Ciphertext {
        &self.ciphertext
    }

    /// Get a mutable reference to the ciphertext
    pub fn ciphertext_mut(&mut self) -> &mut Ciphertext {
        &mut self.ciphertext
    }

    /// Consume self and return the inner ciphertext
    pub fn into_ciphertext(self) -> Ciphertext {
        self.ciphertext
    }
}

/// An encrypted pBit state for homomorphic tessellation operations
#[derive(Debug, Clone)]
pub struct EncryptedState {
    /// Encrypted x-coordinate
    pub x: Ciphertext,
    /// Encrypted y-coordinate
    pub y: Ciphertext,
    /// Encrypted z-coordinate
    pub z: Ciphertext,
}

impl EncryptedState {
    /// Create a new encrypted state from coordinate ciphertexts
    pub fn new(x: Ciphertext, y: Ciphertext, z: Ciphertext) -> Self {
        Self { x, y, z }
    }

    /// Get the encrypted coordinates
    pub fn coordinates(&self) -> (&Ciphertext, &Ciphertext, &Ciphertext) {
        (&self.x, &self.y, &self.z)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BfvContext, BfvParameters, Result};

    #[test]
    fn test_encrypted_phi_creation() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let phi_value = 100i64;
        let ciphertext = ctx.encrypt(phi_value, &keys.public_key)?;

        let encrypted_phi = EncryptedPhi::from_ciphertext(ciphertext);
        let decrypted = ctx.decrypt(encrypted_phi.ciphertext(), &keys.secret_key)?;

        assert_eq!(phi_value, decrypted);
        Ok(())
    }

    #[test]
    fn test_encrypted_state() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let x = 10i64;
        let y = 20i64;
        let z = 30i64;

        let enc_x = ctx.encrypt(x, &keys.public_key)?;
        let enc_y = ctx.encrypt(y, &keys.public_key)?;
        let enc_z = ctx.encrypt(z, &keys.public_key)?;

        let state = EncryptedState::new(enc_x, enc_y, enc_z);
        let (enc_x_ref, enc_y_ref, enc_z_ref) = state.coordinates();

        assert_eq!(x, ctx.decrypt(enc_x_ref, &keys.secret_key)?);
        assert_eq!(y, ctx.decrypt(enc_y_ref, &keys.secret_key)?);
        assert_eq!(z, ctx.decrypt(enc_z_ref, &keys.secret_key)?);

        Ok(())
    }
}
