//! Encrypted pBit State Operations
//!
//! This module provides homomorphic encryption for pBit states and operations,
//! enabling privacy-preserving computation of consciousness metrics.
//!
//! ## Architecture
//!
//! ```text
//! PBit (plaintext) → EncryptedPBitState (ciphertext)
//!     ↓                       ↓
//! effective_field()    encrypted_effective_field()
//!     ↓                       ↓
//! Φ calculation        Encrypted Φ calculation
//! ```
//!
//! ## Security Model
//!
//! - **Public**: Lattice structure, positions, coupling topology
//! - **Private**: pBit states, biases, probabilities
//! - **Encrypted Operations**: Effective field, state sums, linear combinations
//!
//! ## References
//!
//! - Bordage et al. (2021) "Privacy-Preserving Neural Network Inference"
//! - Juvekar et al. (2018) "GAZELLE: Low Latency Framework for FHE"

use crate::{BfvContext, HomomorphicError, Result, PublicKey, SecretKey};
use fhe::bfv::Ciphertext;
use std::collections::HashMap;

/// An encrypted pBit state for privacy-preserving computations
///
/// Represents a pBit's internal state in encrypted form, allowing
/// homomorphic operations without revealing sensitive information.
#[derive(Debug, Clone)]
pub struct EncryptedPBitState {
    /// Encrypted state value (0 or 1)
    pub enc_state: Ciphertext,

    /// Encrypted probability of being in state 1
    pub enc_prob_one: Option<Ciphertext>,

    /// Encrypted bias (external field)
    pub enc_bias: Ciphertext,

    /// Temperature (kept plaintext - public parameter)
    pub temperature: f64,

    /// Coupling strengths: (neighbor_index, strength)
    ///
    /// Note: Coupling strengths are kept plaintext for efficiency.
    /// They're typically public in pBit models (derived from geometry).
    /// This also avoids the complexity of relinearization after multiplication.
    pub couplings: HashMap<usize, f64>,
}

impl EncryptedPBitState {
    /// Create a new encrypted pBit state from plaintext values
    ///
    /// # Arguments
    ///
    /// * `state` - Current boolean state (0 or 1)
    /// * `bias` - External field bias
    /// * `temperature` - Temperature parameter (kept plaintext)
    /// * `couplings` - Coupling strengths to neighbors
    /// * `ctx` - BFV encryption context
    /// * `pk` - Public key for encryption
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use hyperphysics_homomorphic::{BfvContext, BfvParameters, EncryptedPBitState};
    /// use std::collections::HashMap;
    ///
    /// let params = BfvParameters::default_128bit_security()?;
    /// let ctx = BfvContext::new(params)?;
    /// let keys = ctx.generate_keys()?;
    ///
    /// let state = true;
    /// let bias = 0.5;
    /// let temperature = 1.0;
    /// let couplings = HashMap::from([(0, 0.2), (1, -0.3)]);
    ///
    /// let enc_pbit = EncryptedPBitState::encrypt(
    ///     state, bias, temperature, &couplings, &ctx, &keys.public_key
    /// )?;
    /// # Ok::<(), hyperphysics_homomorphic::HomomorphicError>(())
    /// ```
    pub fn encrypt(
        state: bool,
        bias: f64,
        temperature: f64,
        couplings: &HashMap<usize, f64>,
        ctx: &BfvContext,
        pk: &PublicKey,
    ) -> Result<Self> {
        // Convert state to integer (0 or 1)
        let state_int = if state { 1i64 } else { 0i64 };
        let enc_state = ctx.encrypt(state_int, pk)?;

        // Convert bias to fixed-point integer (scale by 1000 for precision)
        let bias_scaled = (bias * 1000.0).round() as i64;
        let enc_bias = ctx.encrypt(bias_scaled, pk)?;

        // Keep couplings as plaintext (more efficient, typically public)
        let couplings_copy = couplings.clone();

        Ok(Self {
            enc_state,
            enc_prob_one: None, // Will be computed separately if needed
            enc_bias,
            temperature,
            couplings: couplings_copy,
        })
    }

    /// Decrypt the pBit state
    ///
    /// # Arguments
    ///
    /// * `ctx` - BFV context
    /// * `sk` - Secret key for decryption
    ///
    /// # Returns
    ///
    /// Tuple of (state, bias, couplings)
    pub fn decrypt(
        &self,
        ctx: &BfvContext,
        sk: &SecretKey,
    ) -> Result<(bool, f64, HashMap<usize, f64>)> {
        // Decrypt state
        let state_int = ctx.decrypt(&self.enc_state, sk)?;
        let state = state_int != 0;

        // Decrypt bias (unscale)
        let bias_scaled = ctx.decrypt(&self.enc_bias, sk)?;
        let bias = (bias_scaled as f64) / 1000.0;

        // Couplings are plaintext
        let couplings = self.couplings.clone();

        Ok((state, bias, couplings))
    }

    /// Get the encrypted state as a ciphertext
    pub fn encrypted_state(&self) -> &Ciphertext {
        &self.enc_state
    }

    /// Get the encrypted bias
    pub fn encrypted_bias(&self) -> &Ciphertext {
        &self.enc_bias
    }

    /// Get the plaintext temperature
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
}

/// Encrypted pBit lattice operations
pub struct EncryptedPBitOps;

impl EncryptedPBitOps {
    /// Calculate encrypted effective field for a pBit
    ///
    /// Computes: h_eff = bias + Σ J_ij * s_j (all encrypted)
    ///
    /// This is a core operation in pBit dynamics and is perfectly suited
    /// for homomorphic encryption (linear combination of encrypted values).
    ///
    /// # Arguments
    ///
    /// * `enc_pbit` - The encrypted pBit whose field we're calculating
    /// * `neighbor_states` - Encrypted states of neighboring pBits
    /// * `ctx` - BFV context for homomorphic operations
    ///
    /// # Returns
    ///
    /// Encrypted effective field value
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use hyperphysics_homomorphic::{BfvContext, BfvParameters, EncryptedPBitOps};
    ///
    /// let params = BfvParameters::default_128bit_security()?;
    /// let ctx = BfvContext::new(params)?;
    /// let keys = ctx.generate_keys()?;
    ///
    /// // enc_pbit and neighbor_states would be encrypted
    /// // let enc_h_eff = EncryptedPBitOps::encrypted_effective_field(
    /// //     &enc_pbit, &neighbor_states, &ctx
    /// // )?;
    /// # Ok::<(), hyperphysics_homomorphic::HomomorphicError>(())
    /// ```
    pub fn encrypted_effective_field(
        enc_pbit: &EncryptedPBitState,
        neighbor_states: &HashMap<usize, Ciphertext>,
        ctx: &BfvContext,
    ) -> Result<Ciphertext> {
        // Start with the bias
        let mut h_eff = enc_pbit.enc_bias.clone();

        // Add coupling contributions: Σ J_ij * s_j
        // Use plaintext multiplication (couplings are public, states are encrypted)
        for (&neighbor_idx, &coupling_strength) in enc_pbit.couplings.iter() {
            if let Some(enc_neighbor_state) = neighbor_states.get(&neighbor_idx) {
                // Scale coupling to match our fixed-point representation
                let coupling_scaled = (coupling_strength * 1000.0).round() as i64;

                // Multiply (plaintext coupling) * (encrypted state)
                let coupling_contribution = ctx.mul_plain(enc_neighbor_state, coupling_scaled)?;

                // Add to effective field
                h_eff = ctx.add(&h_eff, &coupling_contribution)?;
            }
        }

        Ok(h_eff)
    }

    /// Sum encrypted states across a subset of pBits
    ///
    /// Used in Φ calculation for partition effective information.
    /// Computes: Σ s_i for i in subset
    ///
    /// # Arguments
    ///
    /// * `enc_states` - Vector of encrypted pBit states
    /// * `subset_indices` - Indices of pBits to sum
    /// * `ctx` - BFV context
    ///
    /// # Returns
    ///
    /// Encrypted sum of states
    pub fn sum_encrypted_states(
        enc_states: &[Ciphertext],
        subset_indices: &[usize],
        ctx: &BfvContext,
    ) -> Result<Ciphertext> {
        if subset_indices.is_empty() {
            return Err(HomomorphicError::OperationFailure {
                operation: "Cannot sum empty subset".to_string(),
            });
        }

        // Start with first element
        let mut sum = enc_states[subset_indices[0]].clone();

        // Add remaining elements
        for &idx in &subset_indices[1..] {
            if idx >= enc_states.len() {
                return Err(HomomorphicError::OperationFailure {
                    operation: format!("Index {} out of bounds (len {})", idx, enc_states.len()),
                });
            }
            sum = ctx.add(&sum, &enc_states[idx])?;
        }

        Ok(sum)
    }

    /// Compute weighted encrypted sum (scaled for later division)
    ///
    /// Computes: Σ s_i * scale
    ///
    /// Note: BFV doesn't support division on encrypted data.
    /// This function returns a scaled sum that can be divided
    /// plaintext after decryption for computing means.
    pub fn scaled_sum_encrypted_states(
        enc_states: &[Ciphertext],
        subset_indices: &[usize],
        scale: i64,
        ctx: &BfvContext,
    ) -> Result<Ciphertext> {
        let sum = Self::sum_encrypted_states(enc_states, subset_indices, ctx)?;
        ctx.mul_plain(&sum, scale)
    }

    /// Encrypt a vector of pBit states
    ///
    /// Convenience method for encrypting multiple states at once.
    ///
    /// # Arguments
    ///
    /// * `states` - Vector of boolean states
    /// * `ctx` - BFV context
    /// * `pk` - Public key
    ///
    /// # Returns
    ///
    /// Vector of encrypted states
    pub fn encrypt_states(
        states: &[bool],
        ctx: &BfvContext,
        pk: &PublicKey,
    ) -> Result<Vec<Ciphertext>> {
        states
            .iter()
            .map(|&state| {
                let state_int = if state { 1i64 } else { 0i64 };
                ctx.encrypt(state_int, pk)
            })
            .collect()
    }

    /// Decrypt a vector of encrypted states
    ///
    /// # Arguments
    ///
    /// * `enc_states` - Vector of encrypted states
    /// * `ctx` - BFV context
    /// * `sk` - Secret key
    ///
    /// # Returns
    ///
    /// Vector of decrypted boolean states
    pub fn decrypt_states(
        enc_states: &[Ciphertext],
        ctx: &BfvContext,
        sk: &SecretKey,
    ) -> Result<Vec<bool>> {
        enc_states
            .iter()
            .map(|enc_state| {
                let state_int = ctx.decrypt(enc_state, sk)?;
                Ok(state_int != 0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BfvParameters;

    #[test]
    fn test_encrypt_decrypt_pbit_state() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let state = true;
        let bias = 0.5;
        let temperature = 1.0;
        let couplings = HashMap::from([(0, 0.2), (1, -0.3), (2, 0.15)]);

        let enc_pbit = EncryptedPBitState::encrypt(
            state,
            bias,
            temperature,
            &couplings,
            &ctx,
            &keys.public_key,
        )?;

        let (dec_state, dec_bias, dec_couplings) = enc_pbit.decrypt(&ctx, &keys.secret_key)?;

        assert_eq!(state, dec_state);
        assert!((bias - dec_bias).abs() < 0.01, "Bias mismatch: {} vs {}", bias, dec_bias);
        assert_eq!(temperature, enc_pbit.temperature());

        for (&idx, &orig_strength) in &couplings {
            let dec_strength = dec_couplings.get(&idx).unwrap();
            assert!(
                (orig_strength - dec_strength).abs() < 0.01,
                "Coupling {} mismatch: {} vs {}",
                idx,
                orig_strength,
                dec_strength
            );
        }

        Ok(())
    }

    #[test]
    fn test_encrypted_effective_field() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        // Create a pBit with known bias and couplings
        let bias = 0.1;
        let couplings = HashMap::from([(0, 0.2), (1, -0.3)]);

        let enc_pbit = EncryptedPBitState::encrypt(
            false,
            bias,
            1.0,
            &couplings,
            &ctx,
            &keys.public_key,
        )?;

        // Create neighbor states
        let neighbor_states_bool = vec![true, false]; // Neighbors 0 and 1
        let enc_neighbor_states = EncryptedPBitOps::encrypt_states(
            &neighbor_states_bool,
            &ctx,
            &keys.public_key,
        )?;

        let mut neighbor_map = HashMap::new();
        neighbor_map.insert(0, enc_neighbor_states[0].clone());
        neighbor_map.insert(1, enc_neighbor_states[1].clone());

        // Calculate encrypted effective field
        let enc_h_eff = EncryptedPBitOps::encrypted_effective_field(
            &enc_pbit,
            &neighbor_map,
            &ctx,
        )?;

        // Decrypt and verify
        // Expected: h_eff = bias + J_0*s_0 + J_1*s_1
        //                 = 0.1 + 0.2*1 + (-0.3)*0
        //                 = 0.1 + 0.2 = 0.3
        let h_eff_scaled = ctx.decrypt(&enc_h_eff, &keys.secret_key)?;
        let h_eff = (h_eff_scaled as f64) / 1000.0;

        let expected = 0.3;
        assert!(
            (h_eff - expected).abs() < 0.01,
            "Effective field mismatch: {} vs {}",
            h_eff,
            expected
        );

        Ok(())
    }

    #[test]
    fn test_sum_encrypted_states() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let states = vec![true, false, true, true, false];
        let enc_states = EncryptedPBitOps::encrypt_states(&states, &ctx, &keys.public_key)?;

        // Sum subset [0, 2, 3] (should be 1 + 1 + 1 = 3)
        let subset = vec![0, 2, 3];
        let enc_sum = EncryptedPBitOps::sum_encrypted_states(&enc_states, &subset, &ctx)?;

        let sum = ctx.decrypt(&enc_sum, &keys.secret_key)?;
        assert_eq!(sum, 3);

        Ok(())
    }

    #[test]
    fn test_scaled_sum_encrypted_states() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let states = vec![true, true, false, true]; // Sum = 3
        let enc_states = EncryptedPBitOps::encrypt_states(&states, &ctx, &keys.public_key)?;

        let subset = vec![0, 1, 2, 3];
        let scale = 10i64; // Use smaller scale to avoid modulus issues
        let enc_scaled = EncryptedPBitOps::scaled_sum_encrypted_states(
            &enc_states,
            &subset,
            scale,
            &ctx,
        )?;

        let scaled_sum = ctx.decrypt(&enc_scaled, &keys.secret_key)?;

        // Expected: 3 * 10 = 30
        assert_eq!(scaled_sum, 30);

        Ok(())
    }

    #[test]
    fn test_encrypt_decrypt_state_vector() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        let states = vec![true, false, true, false, false, true];
        let enc_states = EncryptedPBitOps::encrypt_states(&states, &ctx, &keys.public_key)?;
        let dec_states = EncryptedPBitOps::decrypt_states(&enc_states, &ctx, &keys.secret_key)?;

        assert_eq!(states, dec_states);

        Ok(())
    }

    #[test]
    fn test_effective_field_with_negative_couplings() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let ctx = BfvContext::new(params)?;
        let keys = ctx.generate_keys()?;

        // Bias + negative coupling + positive coupling
        let bias = 0.5;
        let couplings = HashMap::from([(0, -0.4), (1, 0.3)]);

        let enc_pbit = EncryptedPBitState::encrypt(
            false,
            bias,
            1.0,
            &couplings,
            &ctx,
            &keys.public_key,
        )?;

        // Both neighbors in state 1
        let enc_neighbor_states = EncryptedPBitOps::encrypt_states(
            &[true, true],
            &ctx,
            &keys.public_key,
        )?;

        let mut neighbor_map = HashMap::new();
        neighbor_map.insert(0, enc_neighbor_states[0].clone());
        neighbor_map.insert(1, enc_neighbor_states[1].clone());

        let enc_h_eff = EncryptedPBitOps::encrypted_effective_field(
            &enc_pbit,
            &neighbor_map,
            &ctx,
        )?;

        // Expected: 0.5 + (-0.4)*1 + 0.3*1 = 0.5 - 0.4 + 0.3 = 0.4
        let h_eff_scaled = ctx.decrypt(&enc_h_eff, &keys.secret_key)?;
        let h_eff = (h_eff_scaled as f64) / 1000.0;

        assert!(
            (h_eff - 0.4).abs() < 0.01,
            "Effective field: {} (expected 0.4)",
            h_eff
        );

        Ok(())
    }
}
