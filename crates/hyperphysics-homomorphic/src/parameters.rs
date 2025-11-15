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
use seal_fhe::{
    BFVEncoder, BFVEvaluator, Context, Decryptor, Encryptor, KeyGenerator,
    EncryptionParameters, SecurityLevel, CoefficientModulus, PlainModulus,
    Modulus, SchemeType,
};
use serde::{Deserialize, Serialize};

/// BFV encryption parameters with recommended defaults
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BfvParameters {
    /// Polynomial modulus degree (power of 2)
    /// Common values: 4096, 8192, 16384, 32768
    pub poly_modulus_degree: usize,

    /// Plaintext modulus for BFV arithmetic
    /// Should be a prime for optimal performance
    pub plain_modulus: u64,

    /// Target security level in bits
    /// Standard values: 128, 192, 256
    pub security_level: usize,

    /// Coefficient modulus bit sizes
    /// Determines noise budget and multiplication depth
    pub coeff_modulus_bits: Vec<i32>,
}

impl BfvParameters {
    /// Create parameters with 128-bit security (recommended default)
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
            coeff_modulus_bits: vec![60, 40, 40, 60],  // ~200 bits total
        })
    }

    /// Create parameters with 192-bit security (higher security)
    pub fn default_192bit_security() -> Result<Self> {
        Ok(Self {
            poly_modulus_degree: 16384,
            plain_modulus: 1024,
            security_level: 192,
            coeff_modulus_bits: vec![60, 50, 50, 60],
        })
    }

    /// Create parameters with 256-bit security (maximum security)
    pub fn default_256bit_security() -> Result<Self> {
        Ok(Self {
            poly_modulus_degree: 32768,
            plain_modulus: 1024,
            security_level: 256,
            coeff_modulus_bits: vec![60, 60, 60, 60],
        })
    }

    /// Create custom parameters
    ///
    /// # Arguments
    ///
    /// * `poly_modulus_degree` - Polynomial modulus degree (power of 2)
    /// * `plain_modulus` - Plaintext modulus (should be prime)
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
        if !poly_modulus_degree.is_power_of_two() || poly_modulus_degree < 4096 {
            return Err(HomomorphicError::ParameterError {
                message: format!(
                    "Polynomial modulus degree must be power of 2 >= 4096, got {}",
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

        // Choose coefficient modulus based on poly_modulus_degree and security
        let coeff_modulus_bits = match (poly_modulus_degree, security_level) {
            (4096, 128) => vec![40, 20, 40],
            (8192, 128) => vec![60, 40, 40, 60],
            (16384, 128) => vec![60, 50, 50, 50, 60],
            (16384, 192) => vec![60, 50, 50, 60],
            (32768, 256) => vec![60, 60, 60, 60],
            _ => vec![60, 40, 40, 60],  // Default fallback
        };

        Ok(Self {
            poly_modulus_degree,
            plain_modulus,
            security_level,
            coeff_modulus_bits,
        })
    }

    /// Build a SEAL encryption context from these parameters
    ///
    /// This creates the low-level SEAL context needed for all operations.
    pub fn build_context(&self) -> Result<Context> {
        // Create encryption parameters
        let mut params = EncryptionParameters::new(SchemeType::BFV)
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to create encryption parameters: {:?}", e),
            })?;

        // Set polynomial modulus degree
        params.set_poly_modulus_degree(self.poly_modulus_degree)
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to set poly modulus degree: {:?}", e),
            })?;

        // Set coefficient modulus
        let coeff_modulus = CoefficientModulus::create(
            self.poly_modulus_degree,
            &self.coeff_modulus_bits,
        ).map_err(|e| HomomorphicError::ParameterError {
            message: format!("Failed to create coefficient modulus: {:?}", e),
        })?;

        params.set_coefficient_modulus(&coeff_modulus)
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to set coefficient modulus: {:?}", e),
            })?;

        // Set plaintext modulus
        let plain_mod = PlainModulus::batching(self.poly_modulus_degree, 20)
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to create plaintext modulus: {:?}", e),
            })?;

        params.set_plain_modulus(&plain_mod)
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to set plaintext modulus: {:?}", e),
            })?;

        // Create context
        let context = Context::new(&params, true, self.security_level_to_seal())
            .map_err(|e| HomomorphicError::ParameterError {
                message: format!("Failed to create context: {:?}", e),
            })?;

        Ok(context)
    }

    /// Convert our security level to SEAL's SecurityLevel enum
    fn security_level_to_seal(&self) -> SecurityLevel {
        match self.security_level {
            128 => SecurityLevel::TC128,
            192 => SecurityLevel::TC192,
            256 => SecurityLevel::TC256,
            _ => SecurityLevel::TC128,  // Default fallback
        }
    }

    /// Estimate the noise budget for a fresh ciphertext
    ///
    /// Returns the approximate number of bits of noise budget available.
    pub fn estimate_noise_budget(&self) -> i32 {
        // Rough estimate based on coefficient modulus total bits
        let total_bits: i32 = self.coeff_modulus_bits.iter().sum();
        total_bits - 20  // Reserve ~20 bits for system overhead
    }

    /// Estimate maximum multiplication depth
    ///
    /// This is a conservative estimate of how many sequential
    /// multiplications can be performed before noise budget runs out.
    pub fn estimate_multiplication_depth(&self) -> usize {
        let noise_budget = self.estimate_noise_budget();
        // Each multiplication consumes roughly 40-50 bits
        ((noise_budget as f64) / 45.0).floor() as usize
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
        assert!(noise_budget > 100, "Noise budget should be substantial");
        Ok(())
    }

    #[test]
    fn test_multiplication_depth_estimation() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let depth = params.estimate_multiplication_depth();
        assert!(depth >= 3, "Should support at least 3 multiplications");
        Ok(())
    }

    #[test]
    fn test_build_context() -> Result<()> {
        let params = BfvParameters::default_128bit_security()?;
        let context = params.build_context()?;
        // Context created successfully if we get here
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
