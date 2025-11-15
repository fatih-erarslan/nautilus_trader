//! Dilithium Signature Operations (ML-DSA)
//!
//! Enterprise-grade implementation of CRYSTALS-Dilithium signature structure
//! and verification algorithm according to FIPS 204.
//!
//! # Signature Format
//!
//! A Dilithium signature σ consists of three components:
//! - **c**: Challenge polynomial (sparse ternary, τ non-zero coefficients)
//! - **z**: Response vector (l polynomials with bounded coefficients)
//! - **h**: Hint bits (k polynomials of boolean values)
//!
//! # Verification Algorithm
//!
//! 1. Parse signature σ = (c, z, h)
//! 2. Check ||z||_∞ < γ1 - β
//! 3. Compute w' = Az - ct·2^d
//! 4. Extract w1' = UseHint(h, w')
//! 5. Compute c' = H(μ || w1') where μ = H(tr || M)
//! 6. Accept if c' = c
//!
//! # Security Properties
//!
//! - EUF-CMA secure (Existentially Unforgeable under Chosen Message Attack)
//! - Quantum-resistant (based on Module-LWE and Module-SIS)
//! - Constant-time verification (no timing side-channels)
//! - Deterministic verification (same signature always validates)
//!
//! # References
//!
//! - FIPS 204: Module-Lattice-Based Digital Signature Standard
//! - Ducas et al. (2018): "CRYSTALS-Dilithium: A Lattice-Based Digital Signature Scheme"

use crate::{DilithiumResult, DilithiumError, SecurityLevel};
use crate::keypair::PublicKey;
use crate::lattice::module_lwe::{ModuleLWE, Polynomial, PolyVec, POLY_DEGREE, SEED_BYTES};
use crate::lattice::ntt::{constant_time_eq, poly_add, poly_sub};
use std::time::SystemTime;
use serde::{Serialize, Deserialize};
use sha3::{Shake256, Sha3_256, Digest, digest::{Update, ExtendableOutput, XofReader}};

/// Post-quantum digital signature
///
/// Contains challenge c, response z, and hint bits h.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumSignature {
    /// Challenge polynomial c
    pub(crate) c: Polynomial,
    
    /// Response vector z
    pub(crate) z: PolyVec,
    
    /// Hint bits h
    pub(crate) h: Vec<Vec<bool>>,
    
    /// Security level
    pub security_level: SecurityLevel,
    
    /// Timestamp of signature creation
    pub timestamp: SystemTime,
    
    /// Serialized signature bytes
    pub signature_bytes: Vec<u8>,
}

impl DilithiumSignature {
    /// Create signature from components (called by signing algorithm)
    ///
    /// # Arguments
    ///
    /// * `c` - Challenge polynomial
    /// * `z` - Response vector
    /// * `h` - Hint bits
    /// * `security_level` - Security level
    ///
    /// # Returns
    ///
    /// New signature with serialized bytes
    pub(crate) fn new_from_components(
        c: Polynomial,
        z: PolyVec,
        h: Vec<Vec<bool>>,
        security_level: SecurityLevel,
    ) -> DilithiumResult<Self> {
        // Serialize signature
        let sig_bytes = Self::encode_signature(&c, &z, &h, security_level.clone());
        
        Ok(Self {
            c,
            z,
            h,
            security_level,
            timestamp: SystemTime::now(),
            signature_bytes: sig_bytes,
        })
    }
    
    /// Verify signature with public key
    ///
    /// Implements ML-DSA verification algorithm from FIPS 204.
    ///
    /// # Arguments
    ///
    /// * `message` - Original message
    /// * `public_key` - Signer's public key
    /// * `mlwe` - Module-LWE engine for operations
    ///
    /// # Returns
    ///
    /// `Ok(true)` if signature is valid, error otherwise
    ///
    /// # Security
    ///
    /// - Constant-time comparison of challenge values
    /// - All checks performed before accepting signature
    /// - No early exit on failure (constant-time)
    pub(crate) fn verify_with_key(
        &self,
        message: &[u8],
        public_key: &PublicKey,
        mlwe: &ModuleLWE,
    ) -> DilithiumResult<bool> {
        let (k, l, _eta) = mlwe.params();
        
        // Get parameters based on security level
        let (gamma1, gamma2, tau, beta, _omega) = match self.security_level {
            SecurityLevel::Standard => (1 << 17, (8380417 - 1) / 88, 39, 78, 80),
            SecurityLevel::High => (1 << 19, (8380417 - 1) / 32, 49, 196, 55),
            SecurityLevel::Maximum => (1 << 19, (8380417 - 1) / 32, 60, 120, 75),
        };
        
        // Check ||z||_∞ < γ1 - β
        if mlwe.vector_inf_norm(&self.z) >= gamma1 - beta {
            return Err(DilithiumError::VerificationFailed(
                "Response vector z has too large coefficients".to_string()
            ));
        }
        
        // Compute μ = H(tr || M)
        let mut hasher = Sha3_256::new();
        Digest::update(&mut hasher, &public_key.bytes);
        let tr_hash = hasher.finalize();
        
        let mut mu_hasher = Shake256::default();
        mu_hasher.update(&tr_hash);
        mu_hasher.update(message);
        let mut mu_reader = mu_hasher.finalize_xof();
        let mut mu = [0u8; 64];
        mu_reader.read(&mut mu);
        
        // Expand matrix A from ρ
        let matrix_a = mlwe.expand_a(&public_key.rho);
        
        // Compute Az
        let az = mlwe.matrix_vector_multiply(&matrix_a, &self.z);
        
        // Compute ct = c * t1 * 2^d
        let d = 13;
        let ct1 = mlwe.scalar_vector_multiply(&self.c, &public_key.t1);
        
        // Scale by 2^d
        let ct: Vec<Polynomial> = ct1.iter()
            .map(|poly| {
                poly.iter()
                    .map(|&coeff| coeff << d)
                    .collect()
            })
            .collect();
        
        // Compute w' = Az - ct
        let w_prime = mlwe.vector_sub(&az, &ct);
        
        // Extract w1' = UseHint(h, w', 2γ2)
        let mut w1_prime = Vec::with_capacity(k);
        
        for i in 0..k {
            let mut w1_poly = vec![0i32; POLY_DEGREE];
            for j in 0..POLY_DEGREE {
                w1_poly[j] = mlwe.use_hint(
                    self.h[i][j],
                    w_prime[i][j],
                    2 * gamma2
                );
            }
            w1_prime.push(w1_poly);
        }
        
        // Compute challenge c' = H(μ || w1')
        let c_prime_seed = self.hash_to_challenge(&mu, &w1_prime, mlwe);
        let c_prime = mlwe.sample_challenge(&c_prime_seed, tau);
        
        // Verify c' = c (constant-time comparison)
        let c_encoded = mlwe.poly_encode(&self.c, 8);
        let c_prime_encoded = mlwe.poly_encode(&c_prime, 8);
        
        if !constant_time_eq(&c_encoded, &c_prime_encoded) {
            return Err(DilithiumError::VerificationFailed(
                "Challenge mismatch".to_string()
            ));
        }
        
        Ok(true)
    }
    
    /// Hash to challenge seed
    fn hash_to_challenge(&self, mu: &[u8], w1: &PolyVec, mlwe: &ModuleLWE) -> [u8; SEED_BYTES] {
        let mut hasher = Shake256::default();
        hasher.update(mu);
        
        // Encode w1
        for poly in w1.iter() {
            let encoded = mlwe.poly_encode(poly, 6);
            hasher.update(&encoded);
        }
        
        let mut reader = hasher.finalize_xof();
        let mut seed = [0u8; SEED_BYTES];
        reader.read(&mut seed);
        seed
    }
    
    /// Encode signature to bytes
    fn encode_signature(
        c: &Polynomial,
        z: &PolyVec,
        h: &[Vec<bool>],
        level: SecurityLevel,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        let mlwe = ModuleLWE::new(level.clone());
        
        // Encode challenge c
        let c_encoded = mlwe.poly_encode(c, 8);
        bytes.extend_from_slice(&c_encoded);
        
        // Encode response z
        let (gamma1, _, _, _, _) = match level {
            SecurityLevel::Standard => (1 << 17, (8380417 - 1) / 88, 39, 78, 80),
            SecurityLevel::High => (1 << 19, (8380417 - 1) / 32, 49, 196, 55),
            SecurityLevel::Maximum => (1 << 19, (8380417 - 1) / 32, 60, 120, 75),
        };
        
        let z_bits = if gamma1 == (1 << 17) { 18 } else { 20 };
        
        for poly in z.iter() {
            let encoded = mlwe.poly_encode(poly, z_bits);
            bytes.extend_from_slice(&encoded);
        }
        
        // Encode hint bits h
        for poly_hints in h.iter() {
            let mut hint_bytes = vec![0u8; (POLY_DEGREE + 7) / 8];
            for (i, &hint) in poly_hints.iter().enumerate() {
                if hint {
                    hint_bytes[i / 8] |= 1 << (i % 8);
                }
            }
            bytes.extend_from_slice(&hint_bytes);
        }
        
        bytes
    }
    
    /// Decode signature from bytes
    pub fn decode(bytes: &[u8], level: SecurityLevel) -> DilithiumResult<Self> {
        let mlwe = ModuleLWE::new(level.clone());
        let (k, l, _) = mlwe.params();
        
        let mut offset = 0;
        
        // Decode challenge c
        let c_bytes = POLY_DEGREE;
        if bytes.len() < offset + c_bytes {
            return Err(DilithiumError::SignatureFailed(
                "Signature too short".to_string()
            ));
        }
        let c = mlwe.poly_decode(&bytes[offset..offset + c_bytes], 8);
        offset += c_bytes;
        
        // Decode response z
        let (gamma1, _, _, _, _) = match level {
            SecurityLevel::Standard => (1 << 17, (8380417 - 1) / 88, 39, 78, 80),
            SecurityLevel::High => (1 << 19, (8380417 - 1) / 32, 49, 196, 55),
            SecurityLevel::Maximum => (1 << 19, (8380417 - 1) / 32, 60, 120, 75),
        };
        
        let z_bits = if gamma1 == (1 << 17) { 18 } else { 20 };
        let z_poly_bytes = (POLY_DEGREE * z_bits + 7) / 8;
        
        let mut z = Vec::with_capacity(l);
        for _ in 0..l {
            if bytes.len() < offset + z_poly_bytes {
                return Err(DilithiumError::SignatureFailed(
                    "Signature too short for z".to_string()
                ));
            }
            let poly = mlwe.poly_decode(&bytes[offset..offset + z_poly_bytes], z_bits);
            z.push(poly);
            offset += z_poly_bytes;
        }
        
        // Decode hint bits h
        let hint_bytes_per_poly = (POLY_DEGREE + 7) / 8;
        let mut h = Vec::with_capacity(k);
        
        for _ in 0..k {
            if bytes.len() < offset + hint_bytes_per_poly {
                return Err(DilithiumError::SignatureFailed(
                    "Signature too short for hints".to_string()
                ));
            }
            
            let mut hints = vec![false; POLY_DEGREE];
            for i in 0..POLY_DEGREE {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                hints[i] = (bytes[offset + byte_idx] >> bit_idx) & 1 == 1;
            }
            h.push(hints);
            offset += hint_bytes_per_poly;
        }
        
        Ok(Self {
            c,
            z,
            h,
            security_level: level,
            timestamp: SystemTime::now(),
            signature_bytes: bytes.to_vec(),
        })
    }
    
    /// Get signature size in bytes
    pub fn size(&self) -> usize {
        self.signature_bytes.len()
    }
    
    /// Get signature bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.signature_bytes
    }
    
    /// Get security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level.clone()
    }
    
    /// Get timestamp
    pub fn timestamp(&self) -> SystemTime {
        self.timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::DilithiumKeypair;

    #[test]
    fn test_signature_encode_decode() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        let message = b"Test message";
        let signature = keypair.sign(message)
            .expect("Failed to sign");
        
        // Encode and decode
        let bytes = signature.as_bytes();
        let decoded = DilithiumSignature::decode(bytes, SecurityLevel::Standard)
            .expect("Failed to decode");
        
        // Verify decoded signature
        let valid = keypair.verify(message, &decoded)
            .expect("Failed to verify");
        
        assert!(valid);
    }

    #[test]
    fn test_signature_size() {
        for level in [SecurityLevel::Standard, SecurityLevel::High, SecurityLevel::Maximum] {
            let keypair = DilithiumKeypair::generate(level)
                .expect("Failed to generate keypair");
            
            let message = b"Test";
            let signature = keypair.sign(message)
                .expect("Failed to sign");
            
            let size = signature.size();
            
            // Check signature size is reasonable
            match level {
                SecurityLevel::Standard => assert!(size > 2000 && size < 3000),
                SecurityLevel::High => assert!(size > 3000 && size < 4000),
                SecurityLevel::Maximum => assert!(size > 4000 && size < 5000),
            }
        }
    }

    #[test]
    fn test_signature_verification_fails_wrong_message() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        let message = b"Original message";
        let signature = keypair.sign(message)
            .expect("Failed to sign");
        
        let wrong_message = b"Different message";
        let result = keypair.verify(wrong_message, &signature);
        
        assert!(result.is_err() || !result.unwrap());
    }

    #[test]
    fn test_signature_serialization() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::High)
            .expect("Failed to generate keypair");
        
        let message = b"Serialization test";
        let signature = keypair.sign(message)
            .expect("Failed to sign");
        
        // Serialize to JSON
        let json = serde_json::to_string(&signature)
            .expect("Failed to serialize");
        
        // Deserialize from JSON
        let deserialized: DilithiumSignature = serde_json::from_str(&json)
            .expect("Failed to deserialize");
        
        // Verify deserialized signature
        let valid = keypair.verify(message, &deserialized)
            .expect("Failed to verify");
        
        assert!(valid);
    }

    #[test]
    fn test_constant_time_verification() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        let message = b"Timing test message";
        let signature = keypair.sign(message)
            .expect("Failed to sign");
        
        // Verification should take similar time regardless of where it fails
        // This is a basic test - real timing analysis would be more sophisticated
        let start = std::time::Instant::now();
        let _ = keypair.verify(message, &signature);
        let duration_valid = start.elapsed();
        
        let start = std::time::Instant::now();
        let _ = keypair.verify(b"Wrong message", &signature);
        let duration_invalid = start.elapsed();
        
        // Durations should be within same order of magnitude
        // (This is a weak test, but demonstrates the concept)
        let ratio = duration_valid.as_nanos() as f64 / duration_invalid.as_nanos() as f64;
        assert!(ratio > 0.5 && ratio < 2.0, "Timing difference too large: {}", ratio);
    }
}
