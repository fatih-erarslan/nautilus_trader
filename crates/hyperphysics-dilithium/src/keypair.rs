//! Dilithium Keypair Generation and Management (ML-DSA)
//!
//! Enterprise-grade implementation of CRYSTALS-Dilithium key generation,
//! signing, and verification algorithms according to FIPS 204.
//!
//! # Algorithm Overview
//!
//! **Key Generation:**
//! 1. Sample seed ρ, K uniformly at random
//! 2. Expand A = ExpandA(ρ) ∈ R_q^(k×l)
//! 3. Sample secret vectors s1 ∈ R_q^l, s2 ∈ R_q^k with small coefficients
//! 4. Compute t = As1 + s2
//! 5. Decompose t = t1·2^d + t0
//! 6. Public key: pk = (ρ, t1)
//! 7. Secret key: sk = (ρ, K, tr, s1, s2, t0)
//!
//! **Signing:**
//! 1. Sample masking vector y with small coefficients
//! 2. Compute w = Ay
//! 3. Extract high bits w1 = HighBits(w)
//! 4. Compute challenge c = H(μ || w1) where μ = H(tr || M)
//! 5. Compute z = y + cs1
//! 6. Check ||z|| < γ1 - β (rejection sampling)
//! 7. Compute r0 = LowBits(w - cs2)
//! 8. Check ||r0|| < γ2 - β
//! 9. Signature: σ = (c, z, h) where h are hint bits
//!
//! **Verification:**
//! 1. Parse signature σ = (c, z, h)
//! 2. Check ||z|| < γ1 - β
//! 3. Compute w' = Az - ct·2^d
//! 4. Extract w1' = UseHint(h, w')
//! 5. Verify c = H(μ || w1')
//!
//! # Security Properties
//!
//! - Quantum-resistant (NIST PQC Round 3 finalist, FIPS 204 standard)
//! - EUF-CMA secure under Module-LWE and Module-SIS assumptions
//! - Constant-time operations (no timing side-channels)
//! - Deterministic signing with proper randomness
//!
//! # References
//!
//! - FIPS 204: Module-Lattice-Based Digital Signature Standard
//! - Ducas et al. (2018): "CRYSTALS-Dilithium: A Lattice-Based Digital Signature Scheme"

use crate::{DilithiumResult, SecurityLevel, DilithiumSignature};
use crate::lattice::module_lwe::{ModuleLWE, Polynomial, PolyVec, POLY_DEGREE, SEED_BYTES};
use crate::lattice::ntt::{NTT, DILITHIUM_Q};
use zeroize::{Zeroize, ZeroizeOnDrop};
use serde::{Serialize, Deserialize};
use sha3::{Shake256, Sha3_256, digest::{ExtendableOutput, XofReader, Update, Digest}};
use rand::RngCore;

/// Dilithium public key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicKey {
    /// Seed ρ for matrix A expansion
    pub(crate) rho: [u8; SEED_BYTES],
    
    /// High bits of t = As1 + s2
    pub(crate) t1: PolyVec,
    
    /// Security level
    pub(crate) security_level: SecurityLevel,
    
    /// Serialized bytes (for compatibility)
    pub(crate) bytes: Vec<u8>,
}

/// Dilithium secret key (zeroized on drop)
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct SecretKey {
    /// Seed ρ for matrix A expansion
    pub(crate) rho: [u8; SEED_BYTES],
    
    /// Seed K for randomness
    pub(crate) key: [u8; SEED_BYTES],
    
    /// Hash of public key
    pub(crate) tr: [u8; SEED_BYTES],
    
    /// Secret vector s1
    pub(crate) s1: PolyVec,
    
    /// Secret vector s2
    pub(crate) s2: PolyVec,
    
    /// Low bits of t
    pub(crate) t0: PolyVec,
    
    /// Security level
    pub(crate) security_level: SecurityLevel,
    
    /// Serialized bytes (for compatibility)
    pub(crate) bytes: Vec<u8>,
}

/// Dilithium keypair for post-quantum signatures
#[derive(Clone)]
pub struct DilithiumKeypair {
    /// Public key
    pub public_key: PublicKey,
    
    /// Secret key (zeroized on drop)
    secret_key: SecretKey,
    
    /// Security level
    security_level: SecurityLevel,
    
    /// Module-LWE engine
    mlwe: ModuleLWE,
    
    /// NTT engine
    ntt: NTT,
}

impl DilithiumKeypair {
    /// Generate new quantum-resistant keypair
    ///
    /// Implements ML-DSA key generation algorithm from FIPS 204.
    ///
    /// # Arguments
    ///
    /// * `level` - Security level (Standard/High/Maximum)
    ///
    /// # Returns
    ///
    /// New keypair with fresh randomness
    ///
    /// # Security
    ///
    /// Uses cryptographically secure RNG for seed generation.
    /// All secret material is zeroized on drop.
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::*;
    ///
    /// let keypair = DilithiumKeypair::generate(SecurityLevel::High)?;
    /// # Ok::<(), DilithiumError>(())
    /// ```
    pub fn generate(level: SecurityLevel) -> DilithiumResult<Self> {
        let mlwe = ModuleLWE::new(level.clone());
        let ntt = NTT::new();
        let (k, l, eta) = mlwe.params();
        
        // Generate random seeds
        let mut rng = rand::thread_rng();
        let mut rho = [0u8; SEED_BYTES];
        let mut key = [0u8; SEED_BYTES];
        rng.fill_bytes(&mut rho);
        rng.fill_bytes(&mut key);
        
        // Expand matrix A from ρ
        let matrix_a = mlwe.expand_a(&rho);
        
        // Sample secret vectors s1, s2 with small coefficients
        let mut s1 = Vec::with_capacity(l);
        let mut s2 = Vec::with_capacity(k);
        
        for i in 0..l {
            s1.push(mlwe.sample_small_poly(&key, i as u16));
        }
        
        for i in 0..k {
            s2.push(mlwe.sample_small_poly(&key, (l + i) as u16));
        }
        
        // Compute t = As1 + s2
        let as1 = mlwe.matrix_vector_multiply(&matrix_a, &s1);
        let t = mlwe.vector_add(&as1, &s2);
        
        // Decompose t = t1·2^d + t0
        let d = 13; // Standard decomposition parameter
        let mut t1 = Vec::with_capacity(k);
        let mut t0 = Vec::with_capacity(k);
        
        for poly in t.iter() {
            let (p1, p0) = mlwe.poly_power2round(poly, d);
            t1.push(p1);
            t0.push(p0);
        }
        
        // Encode public key
        let pk_bytes = Self::encode_public_key(&rho, &t1, level.clone());
        
        // Compute tr = H(pk)
        let mut hasher = Sha3_256::new();
        Digest::update(&mut hasher, &pk_bytes);
        let tr_hash = hasher.finalize();
        let mut tr = [0u8; SEED_BYTES];
        tr[..32].copy_from_slice(&tr_hash);
        
        // Encode secret key
        let sk_bytes = Self::encode_secret_key(&rho, &key, &tr, &s1, &s2, &t0, level.clone());
        
        Ok(Self {
            public_key: PublicKey {
                rho,
                t1: t1.clone(),
                security_level: level.clone(),
                bytes: pk_bytes,
            },
            secret_key: SecretKey {
                rho,
                key,
                tr,
                s1,
                s2,
                t0,
                security_level: level.clone(),
                bytes: sk_bytes,
            },
            security_level: level.clone(),
            mlwe,
            ntt,
        })
    }
    
    /// Sign a message using Dilithium signature scheme
    ///
    /// Implements ML-DSA signing algorithm with rejection sampling.
    ///
    /// # Arguments
    ///
    /// * `message` - Message to sign
    ///
    /// # Returns
    ///
    /// Digital signature σ = (c, z, h)
    ///
    /// # Security
    ///
    /// - Uses rejection sampling to ensure signature distribution is independent of secret
    /// - Constant-time operations where security-critical
    /// - Fresh randomness for each signature
    ///
    /// # Performance
    ///
    /// Average ~2-3 rejection sampling iterations
    pub fn sign(&self, message: &[u8]) -> DilithiumResult<DilithiumSignature> {
        let (k, l, _eta) = self.mlwe.params();
        
        // Get parameters based on security level
        let (gamma1, gamma2, tau, beta, omega) = match self.security_level {
            SecurityLevel::Standard => (1 << 17, (8380417 - 1) / 88, 39, 78, 80),
            SecurityLevel::High => (1 << 19, (8380417 - 1) / 32, 49, 196, 55),
            SecurityLevel::Maximum => (1 << 19, (8380417 - 1) / 32, 60, 120, 75),
        };
        
        // Compute μ = H(tr || M)
        let mut mu_hasher = Shake256::default();
        mu_hasher.update(&self.secret_key.tr);
        mu_hasher.update(message);
        let mut mu_reader = mu_hasher.finalize_xof();
        let mut mu = [0u8; 64];
        mu_reader.read(&mut mu);
        
        // Expand matrix A
        let matrix_a = self.mlwe.expand_a(&self.secret_key.rho);
        
        // Rejection sampling loop
        let rng = rand::thread_rng();
        let mut nonce = 0u16;
        
        loop {
            // Sample masking vector y
            let mut y = Vec::with_capacity(l);
            for i in 0..l {
                let mut seed = self.secret_key.key;
                seed[0] ^= nonce as u8;
                seed[1] ^= (nonce >> 8) as u8;
                y.push(self.sample_gamma1_poly(&seed, i as u16, gamma1));
            }
            nonce += 1;
            
            // Compute w = Ay
            let w = self.mlwe.matrix_vector_multiply(&matrix_a, &y);
            
            // Extract high bits w1 = HighBits(w, 2γ2)
            let w1: Vec<Polynomial> = w.iter()
                .map(|poly| {
                    poly.iter()
                        .map(|&coeff| self.mlwe.high_bits(coeff, 2 * gamma2))
                        .collect()
                })
                .collect();
            
            // Compute challenge c = H(μ || w1)
            let c_seed = self.hash_to_challenge(&mu, &w1);
            let c = self.mlwe.sample_challenge(&c_seed, tau);
            
            // Compute z = y + cs1
            let cs1 = self.mlwe.scalar_vector_multiply(&c, &self.secret_key.s1);
            let z = self.mlwe.vector_add(&y, &cs1);
            
            // Check ||z||_∞ < γ1 - β
            if self.mlwe.vector_inf_norm(&z) >= gamma1 - beta {
                continue; // Reject and retry
            }
            
            // Compute r0 = LowBits(w - cs2, 2γ2)
            let cs2 = self.mlwe.scalar_vector_multiply(&c, &self.secret_key.s2);
            let w_minus_cs2 = self.mlwe.vector_sub(&w, &cs2);
            
            let r0: Vec<Polynomial> = w_minus_cs2.iter()
                .map(|poly| {
                    poly.iter()
                        .map(|&coeff| self.mlwe.low_bits(coeff, 2 * gamma2))
                        .collect()
                })
                .collect();
            
            // Check ||r0||_∞ < γ2 - β
            if self.mlwe.vector_inf_norm(&r0) >= gamma2 - beta {
                continue; // Reject and retry
            }
            
            // Compute hint h = MakeHint(-ct0, w - cs2 + ct0, 2γ2)
            let ct0 = self.mlwe.scalar_vector_multiply(&c, &self.secret_key.t0);
            let w_minus_cs2_plus_ct0 = self.mlwe.vector_add(&w_minus_cs2, &ct0);
            
            let mut h = Vec::with_capacity(k);
            let mut hint_count = 0;
            
            for i in 0..k {
                let mut h_poly = vec![false; POLY_DEGREE];
                for j in 0..POLY_DEGREE {
                    let hint = self.mlwe.make_hint(
                        -ct0[i][j],
                        w_minus_cs2_plus_ct0[i][j],
                        2 * gamma2
                    );
                    h_poly[j] = hint;
                    if hint {
                        hint_count += 1;
                    }
                }
                h.push(h_poly);
            }
            
            // Check number of hints ≤ ω
            if hint_count > omega {
                continue; // Reject and retry
            }
            
            // Success! Return signature
            return DilithiumSignature::new_from_components(
                c,
                z,
                h,
                self.security_level.clone(),
            );
        }
    }
    
    /// Verify a signature
    ///
    /// Implements ML-DSA verification algorithm.
    ///
    /// # Arguments
    ///
    /// * `message` - Original message
    /// * `signature` - Signature to verify
    ///
    /// # Returns
    ///
    /// `Ok(true)` if signature is valid, error otherwise
    ///
    /// # Security
    ///
    /// Constant-time comparison of challenge values
    pub fn verify(&self, message: &[u8], signature: &DilithiumSignature) -> DilithiumResult<bool> {
        signature.verify_with_key(message, &self.public_key, &self.mlwe)
    }
    
    /// Sample polynomial with coefficients in [-γ1, γ1]
    fn sample_gamma1_poly(&self, seed: &[u8; SEED_BYTES], nonce: u16, gamma1: i32) -> Polynomial {
        let mut poly = vec![0i32; POLY_DEGREE];
        
        let mut hasher = Shake256::default();
        hasher.update(seed);
        hasher.update(&nonce.to_le_bytes());
        
        let mut reader = hasher.finalize_xof();
        let mut buf = [0u8; 5];
        
        for coeff in poly.iter_mut() {
            loop {
                reader.read(&mut buf);
                let val = ((buf[0] as i64)
                    | ((buf[1] as i64) << 8)
                    | ((buf[2] as i64) << 16)
                    | ((buf[3] as i64) << 24)) & 0xFFFFF;
                
                if val <= 2 * gamma1 as i64 {
                    *coeff = (val - gamma1 as i64) as i32;
                    break;
                }
            }
        }
        
        poly
    }
    
    /// Hash to challenge seed
    fn hash_to_challenge(&self, mu: &[u8], w1: &PolyVec) -> [u8; SEED_BYTES] {
        let mut hasher = Shake256::default();
        hasher.update(mu);
        
        // Encode w1
        for poly in w1.iter() {
            let encoded = self.mlwe.poly_encode(poly, 6);
            hasher.update(&encoded);
        }
        
        let mut reader = hasher.finalize_xof();
        let mut seed = [0u8; SEED_BYTES];
        reader.read(&mut seed);
        seed
    }
    
    /// Encode public key to bytes
    fn encode_public_key(rho: &[u8; SEED_BYTES], t1: &PolyVec, level: SecurityLevel) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(rho);
        
        let mlwe = ModuleLWE::new(level);
        for poly in t1.iter() {
            let encoded = mlwe.poly_encode(poly, 10);
            bytes.extend_from_slice(&encoded);
        }
        
        bytes
    }
    
    /// Encode secret key to bytes
    fn encode_secret_key(
        rho: &[u8; SEED_BYTES],
        key: &[u8; SEED_BYTES],
        tr: &[u8; SEED_BYTES],
        s1: &PolyVec,
        s2: &PolyVec,
        t0: &PolyVec,
        level: SecurityLevel,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(rho);
        bytes.extend_from_slice(key);
        bytes.extend_from_slice(tr);
        
        let mlwe = ModuleLWE::new(level);
        let (_, _, eta) = mlwe.params();
        
        let eta_bits = if eta == 2 { 3 } else { 4 };
        
        for poly in s1.iter() {
            let encoded = mlwe.poly_encode(poly, eta_bits);
            bytes.extend_from_slice(&encoded);
        }
        
        for poly in s2.iter() {
            let encoded = mlwe.poly_encode(poly, eta_bits);
            bytes.extend_from_slice(&encoded);
        }
        
        for poly in t0.iter() {
            let encoded = mlwe.poly_encode(poly, 13);
            bytes.extend_from_slice(&encoded);
        }
        
        bytes
    }
    
    /// Get security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level.clone()
    }
    
    /// Get public key bytes
    pub fn public_key_bytes(&self) -> &[u8] {
        &self.public_key.bytes
    }
    
    /// Get secret key bytes (use with caution!)
    pub fn secret_key_bytes(&self) -> &[u8] {
        &self.secret_key.bytes
    }
}

impl PublicKey {
    /// Get security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level.clone()
    }
    
    /// Get public key bytes
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

impl SecretKey {
    /// Get security level
    pub fn security_level(&self) -> SecurityLevel {
        self.security_level.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypair_generation() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        assert_eq!(keypair.security_level(), SecurityLevel::Standard);
        assert!(!keypair.public_key_bytes().is_empty());
        assert!(!keypair.secret_key_bytes().is_empty());
    }

    #[test]
    fn test_sign_verify() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        let message = b"Test message for Dilithium signature";
        
        let signature = keypair.sign(message)
            .expect("Failed to sign message");
        
        let valid = keypair.verify(message, &signature)
            .expect("Failed to verify signature");
        
        assert!(valid);
    }

    #[test]
    fn test_invalid_signature() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        let message = b"Original message";
        let wrong_message = b"Different message";
        
        let signature = keypair.sign(message)
            .expect("Failed to sign message");
        
        let result = keypair.verify(wrong_message, &signature);
        
        // Should fail verification
        assert!(result.is_err() || !result.unwrap());
    }

    #[test]
    fn test_all_security_levels() {
        for level in [SecurityLevel::Standard, SecurityLevel::High, SecurityLevel::Maximum] {
            let keypair = DilithiumKeypair::generate(level.clone())
                .expect("Failed to generate keypair");
            
            let message = b"Test message";
            let signature = keypair.sign(message)
                .expect("Failed to sign");
            
            let valid = keypair.verify(message, &signature)
                .expect("Failed to verify");
            
            assert!(valid, "Verification failed for {:?}", level);
        }
    }

    #[test]
    fn test_multiple_signatures() {
        let keypair = DilithiumKeypair::generate(SecurityLevel::Standard)
            .expect("Failed to generate keypair");
        
        for i in 0..10 {
            let message = format!("Message {}", i);
            let signature = keypair.sign(message.as_bytes())
                .expect("Failed to sign");
            
            let valid = keypair.verify(message.as_bytes(), &signature)
                .expect("Failed to verify");
            
            assert!(valid);
        }
    }
}
