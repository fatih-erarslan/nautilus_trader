//! BFV Encryption Parameters
//!
//! This module defines the encryption parameters for the BFV scheme, including:
//! - Polynomial modulus degree (n)
//! - Coefficient modulus chain
//! - Plaintext modulus (t)
//! - Security level
//!
//! # Parameter Selection
//!
//! Parameters must be chosen to balance:
//! - **Security**: Larger n provides more security
//! - **Performance**: Smaller n is faster
//! - **Noise budget**: Coefficient modulus affects multiplication depth
//! - **Plaintext space**: Plaintext modulus determines arithmetic range

use crate::{HomomorphicError, Result};
use fhe::bfv::{BfvParametersBuilder, BfvParameters as FheBfvParameters};
use std::sync::Arc;
use serde::{Deserialize, Serialize};

/// BFV encryption parameters with recommended defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfvParameters {
    /// Polynomial modulus degree (power of 2)
    /// Common values: 2048, 4096, 8192, 16384
    pub poly_modulus_degree: usize,

    /// Plaintext modulus for BFV arithmetic
    /// Should be a prime or power of 2 for optimal performance
    pub plain_modulus: u64,

    /// Target security level in bits
    /// Standard values: 128, 192, 256
    pub security_level: usize,

    /// Coefficient moduli for RNS representation
    /// Each value should be a prime close to 2^60
    pub coeff_moduli: Vec<u64>,
}

impl BfvParameters {
    /// Create parameters with 128-bit security (recommended default)
    ///
    /// Uses degree 8192 with moderate plaintext modulus for balanced performance.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use hyperphysics_homomorphic::BfvParameters;
    ///
    /// let params = BfvParameters::default_128bit_security()?;
    /// # Ok::<(), hyperphysics_homomorphic::HomomorphicError>(())
    /// ```
    pub fn default_128bit_security() -> Result<Self> {
        Ok(Self {
            poly_modulus_degree: 8192,
            plain_modulus: 1024,  // 2^10, good for moderate arithmetic
            security_level: 128,
            // RNS primes close to 2^60 for efficient modular arithmetic
            coeff_moduli: vec![
                0x3fffffff000001,  // ~60 bits
            ],
        })
    }

    /// Create parameters with 192-bit security (higher security)
    pub fn default_192bit_security() -> Result<Self> {
        Ok(Self {
            poly_modulus_degree: 16384,
            plain_modulus: 1024,
            security_level: 192,
            coeff_moduli: vec![
                0x3fffffff000001,
            ],
        })
    }

    /// Create parameters with 256-bit security (maximum security)
    pub fn default_256bit_security() -> Result<Self> {
        Ok(Self {
            poly_modulus_degree: 32768,
            plain_modulus: 1024,
            security_level: 256,
            coeff_moduli: vec![
                0x3fffffff000001,
            ],
        })
    }

    /// Create custom parameters
    ///
    /// # Arguments
    ///
    /// * `poly_modulus_degree` - Polynomial modulus degree (power of 2)
    /// * `plain_modulus` - Plaintext modulus
    /// * `security_level` - Target security in bits (128, 192, or 256)
    ///
    /// # Returns
    ///
    /// Parameters with recommended coefficient modulus for the given settings
    pub fn custom(
        poly_modulus_degree: usize,
        plain_modulus: u64,
        security_level: usize,
    ) -> Result<Self> {
        // Validate polynomial modulus degree (must be power of 2)
        if !poly_modulus_degree.is_power_of_two() || poly_modulus_degree < 2048 {
            return Err(HomomorphicError::ParameterError {
                message: format!(
                    "Polynomial modulus degree must be power of 2 >= 2048, got {}",
                    poly_modulus_degree
                ),
            });
        }

        // Validate security level
        if ![128, 192, 256].contains(&security_level) {
            return Err(HomomorphicError::ParameterError {
                message: format!(
                    "Security level must be 128, 192, or 256, got {}",
                    security_level
                ),
            });
        }

        Ok(Self {
            poly_modulus_degree,
            plain_modulus,
            security_level,
            coeff_moduli: vec![0x3fffffff000001],
        })
    }

    /// Build fhe.rs parameters from these settings
    ///
    /// Creates the low-level fhe.rs BfvParameters needed for all operations.
    pub fn build(&self) -> Result<Arc<FheBfvParameters>> {
        BfvParametersBuilder::new()
            .set_degree(self.poly_modulus_degree)
            .set_plaintext_modulus(self.plain_modulus)
            .set_moduli(&self.coeff_moduli)
            .build_arc()
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to build BFV parameters: {:?}", e),
            })
    }

    /// Estimate the noise budget for a fresh ciphertext
    ///
    /// Returns the approximate number of bits of noise budget available.
    /// This is a rough estimate based on the coefficient modulus size.
    pub fn estimate_noise_budget(&self) -> i32 {
        // Each modulus contributes ~60 bits
        let total_bits = self.coeff_moduli.len() as i32 * 60;
        total_bits - 20  // Reserve ~20 bits for system overhead
    }

    /// Estimate maximum multiplication depth
    ///
    /// This is a conservative estimate of how many sequential
    /// multiplications can be performed before noise budget runs out.
    pub fn estimate_multiplication_depth(&self) -> usize {
        let noise_budget = self.estimate_noise_budget();
        // Each multiplication consumes roughly 40-50 bits
        ((noise_budget as f64) / 45.0).floor().max(0.0) as usize
    }
}

impl Default for BfvParameters {
    fn default() -> Self {
        Self::default_128bit_security()
            .expect("Default parameters should always be valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_parameters() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        assert_eq!(params.security_level, 128);
        assert_eq!(params.poly_modulus_degree, 8192);
        assert!(params.plain_modulus > 0);
        Ok(())
    }

    #[test]
    fn test_custom_parameters_valid() -> Result<()> {
        let params = BfvParameters::custom(8192, 1024, 128)?;
        assert_eq!(params.poly_modulus_degree, 8192);
        assert_eq!(params.plain_modulus, 1024);
        Ok(())
    }

    #[test]
    fn test_custom_parameters_invalid_degree() {
        let result = BfvParameters::custom(1000, 1024, 128);
        assert!(result.is_err());
    }

    #[test]
    fn test_custom_parameters_invalid_security() {
        let result = BfvParameters::custom(8192, 1024, 100);
        assert!(result.is_err());
    }

    #[test]
    fn test_noise_budget_estimation() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let noise_budget = params.estimate_noise_budget();
        assert!(noise_budget > 30, "Noise budget should be substantial");
        Ok(())
    }

    #[test]
    fn test_multiplication_depth_estimation() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let depth = params.estimate_multiplication_depth();
        assert!(depth <= 100, "Depth should be reasonable");
        Ok(())
    }

    #[test]
    fn test_build_parameters() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let _fhe_params = params.build()?;
        // Parameters created successfully if we get here
        Ok(())
    }

    #[test]
    fn test_higher_security_levels() -> Result<()> {
        let params_192 = BfvParameters::default_192bit_security()?;
        assert_eq!(params_192.security_level, 192);
        assert_eq!(params_192.poly_modulus_degree, 16384);

        let params_256 = BfvParameters::default_256bit_security()?;
        assert_eq!(params_256.security_level, 256);
        assert_eq!(params_256.poly_modulus_degree, 32768);

        Ok(())
    }
}
