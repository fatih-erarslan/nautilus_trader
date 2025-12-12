//! CRYSTALS-Dilithium Digital Signature Algorithm
//!
//! Implementation of the NIST-standardized CRYSTALS-Dilithium post-quantum
//! digital signature algorithm for quantum-resistant cryptography.

use crate::error::QuantumSecurityError;
use crate::types::*;
use crate::algorithms::{PQCAlgorithm, PQCKey, PQCKeyPair, KeyUsage, DigitalSignature, AlgorithmMetrics};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// CRYSTALS-Dilithium Digital Signature Engine
pub struct DilithiumEngine {
    algorithm: PQCAlgorithm,
    metrics: Arc<RwLock<AlgorithmMetrics>>,
    config: DilithiumConfig,
}

/// Dilithium Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumConfig {
    pub security_level: SecurityLevel,
    pub enable_side_channel_protection: bool,
    pub max_signature_time_us: u64,
    pub enable_deterministic_signing: bool,
}

impl Default for DilithiumConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Level5,
            enable_side_channel_protection: true,
            max_signature_time_us: 50, // Sub-100μs target
            enable_deterministic_signing: true,
        }
    }
}

/// Dilithium Parameters
#[derive(Debug, Clone)]
pub struct DilithiumParams {
    pub n: usize,           // Polynomial degree (256)
    pub q: u32,             // Modulus (2^23 - 2^13 + 1)
    pub d: u8,              // Number of dropped bits
    pub tau: u8,            // Number of ±1's in challenge
    pub beta: u32,          // Maximum coefficient of s1 + s2
    pub gamma1: u32,        // Coefficient bound for y
    pub gamma2: u32,        // Low-order rounding range
    pub k: usize,           // Dimensions (rows of A)
    pub l: usize,           // Dimensions (columns of A)
    pub eta: u8,            // Bound for s1, s2
    pub public_key_size: usize,
    pub private_key_size: usize,
    pub signature_size: usize,
}

/// Dilithium Polynomial
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct DilithiumPolynomial {
    pub coeffs: [i32; 256],
}

/// Dilithium Vector
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct DilithiumVector {
    pub polys: Vec<DilithiumPolynomial>,
}

/// Dilithium Matrix
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct DilithiumMatrix {
    pub rows: Vec<DilithiumVector>,
}

/// Dilithium Public Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumPublicKey {
    pub rho: Vec<u8>,       // Seed for A
    pub t1: Vec<u8>,        // Public key vector (high part)
    pub algorithm: PQCAlgorithm,
}

/// Dilithium Private Key
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct DilithiumPrivateKey {
    pub rho: Vec<u8>,       // Seed for A
    pub k: Vec<u8>,         // Seed for CRH
    pub tr: Vec<u8>,        // Hash of public key
    pub s1: Vec<u8>,        // Secret vector s1
    pub s2: Vec<u8>,        // Secret vector s2
    pub t0: Vec<u8>,        // Public key vector (low part)
    pub algorithm: PQCAlgorithm,
}

/// Dilithium Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DilithiumSignature {
    pub c: Vec<u8>,         // Challenge polynomial
    pub z: Vec<u8>,         // Response vector
    pub h: Vec<u8>,         // Hint vector
    pub algorithm: PQCAlgorithm,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl DilithiumEngine {
    /// Create a new Dilithium engine
    pub async fn new(config: &crate::config::QuantumSecurityConfig) -> Result<Self, QuantumSecurityError> {
        let dilithium_config = DilithiumConfig {
            security_level: config.security_level.clone(),
            enable_side_channel_protection: config.enable_side_channel_protection,
            max_signature_time_us: config.max_latency_us,
            enable_deterministic_signing: true,
        };

        Ok(Self {
            algorithm: config.default_signature_algorithm.clone(),
            metrics: Arc::new(RwLock::new(AlgorithmMetrics::default())),
            config: dilithium_config,
        })
    }

    /// Generate a Dilithium key pair
    pub async fn generate_keypair(&self) -> Result<(DilithiumPublicKey, DilithiumPrivateKey), QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Generate random seed
        let mut seed = [0u8; 32];
        self.fill_random_bytes(&mut seed).await?;

        // Generate key pair using Dilithium algorithm
        let (public_key, private_key) = self.dilithium_keygen(&seed, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_keygen_metrics(elapsed).await;

        // Validate performance target
        if elapsed > self.config.max_signature_time_us {
            tracing::warn!(
                "Dilithium key generation exceeded performance target: {}μs > {}μs",
                elapsed,
                self.config.max_signature_time_us
            );
        }

        Ok((public_key, private_key))
    }

    /// Sign a message using Dilithium
    pub async fn sign(&self, private_key: &DilithiumPrivateKey, message: &[u8]) -> Result<DilithiumSignature, QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Validate private key
        self.validate_private_key(private_key)?;

        // Perform signing
        let signature = self.dilithium_sign(private_key, message, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_signature_metrics(elapsed).await;

        // Validate performance target
        if elapsed > self.config.max_signature_time_us {
            tracing::warn!(
                "Dilithium signing exceeded performance target: {}μs > {}μs",
                elapsed,
                self.config.max_signature_time_us
            );
        }

        Ok(signature)
    }

    /// Verify a Dilithium signature
    pub async fn verify(&self, public_key: &DilithiumPublicKey, message: &[u8], signature: &DilithiumSignature) -> Result<bool, QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Validate inputs
        self.validate_public_key(public_key)?;
        self.validate_signature(signature)?;

        // Ensure algorithm compatibility
        if public_key.algorithm != signature.algorithm {
            return Err(QuantumSecurityError::AlgorithmMismatch);
        }

        // Perform verification
        let valid = self.dilithium_verify(public_key, message, signature, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_verification_metrics(elapsed).await;

        Ok(valid)
    }

    /// Get algorithm parameters
    fn get_params(&self) -> DilithiumParams {
        match self.algorithm {
            PQCAlgorithm::Dilithium2 => DilithiumParams {
                n: 256,
                q: 8380417, // 2^23 - 2^13 + 1
                d: 13,
                tau: 39,
                beta: 78,
                gamma1: 1 << 17,
                gamma2: (1 << 17) / 88,
                k: 4,
                l: 4,
                eta: 2,
                public_key_size: 1312,
                private_key_size: 2528,
                signature_size: 2420,
            },
            PQCAlgorithm::Dilithium3 => DilithiumParams {
                n: 256,
                q: 8380417,
                d: 13,
                tau: 49,
                beta: 196,
                gamma1: 1 << 19,
                gamma2: (1 << 19) / 32,
                k: 6,
                l: 5,
                eta: 4,
                public_key_size: 1952,
                private_key_size: 4000,
                signature_size: 3293,
            },
            PQCAlgorithm::Dilithium5 => DilithiumParams {
                n: 256,
                q: 8380417,
                d: 13,
                tau: 60,
                beta: 325,
                gamma1: 1 << 19,
                gamma2: (1 << 19) / 32,
                k: 8,
                l: 7,
                eta: 2,
                public_key_size: 2592,
                private_key_size: 4864,
                signature_size: 4595,
            },
            _ => panic!("Invalid Dilithium algorithm"),
        }
    }

    /// Dilithium key generation
    fn dilithium_keygen(&self, seed: &[u8], params: &DilithiumParams) -> Result<(DilithiumPublicKey, DilithiumPrivateKey), QuantumSecurityError> {
        // Expand seed
        let (rho, rho_prime, k) = self.expand_seed(seed)?;

        // Generate matrix A from rho
        let a = self.generate_matrix_a(&rho, params)?;

        // Generate secret vectors s1 and s2
        let s1 = self.generate_secret_vector(&rho_prime, params, 0)?;
        let s2 = self.generate_secret_vector(&rho_prime, params, params.l as u8)?;

        // Compute t = As1 + s2
        let t = self.matrix_vector_multiply(&a, &s1)?;
        let t = self.vector_add(&t, &s2)?;

        // Power2Round to get t1 and t0
        let (t1, t0) = self.power2round_vector(&t, params.d)?;

        // Encode public key
        let public_key = DilithiumPublicKey {
            rho: rho.to_vec(),
            t1: self.encode_vector(&t1, params)?,
            algorithm: self.algorithm.clone(),
        };

        // Hash public key
        let tr = self.hash_public_key(&public_key.to_bytes())?;

        // Encode private key
        let private_key = DilithiumPrivateKey {
            rho: rho.to_vec(),
            k: k.to_vec(),
            tr,
            s1: self.encode_vector(&s1, params)?,
            s2: self.encode_vector(&s2, params)?,
            t0: self.encode_vector(&t0, params)?,
            algorithm: self.algorithm.clone(),
        };

        Ok((public_key, private_key))
    }

    /// Dilithium signing
    fn dilithium_sign(&self, private_key: &DilithiumPrivateKey, message: &[u8], params: &DilithiumParams) -> Result<DilithiumSignature, QuantumSecurityError> {
        // Decode private key components
        let s1 = self.decode_vector(&private_key.s1, params)?;
        let s2 = self.decode_vector(&private_key.s2, params)?;
        let t0 = self.decode_vector(&private_key.t0, params)?;
        let a = self.generate_matrix_a(&private_key.rho, params)?;

        // Compute message hash
        let mu = self.hash_message(&private_key.tr, message)?;

        let mut kappa = 0u16;
        let mut signature_attempt = 0;

        loop {
            if signature_attempt > 100 {
                return Err(QuantumSecurityError::SigningFailed("Too many signature attempts".to_string()));
            }

            // Generate y
            let y = self.generate_y_vector(&private_key.k, &mu, kappa, params)?;

            // Compute w = Ay
            let w = self.matrix_vector_multiply(&a, &y)?;

            // HighBits to get w1
            let w1 = self.highbits_vector(&w, params.gamma2)?;

            // Compute challenge c
            let c_tilde = self.sample_in_ball(&mu, &w1, params)?;
            let c = self.decode_challenge(&c_tilde, params)?;

            // Compute z = y + cs1
            let cs1 = self.vector_scalar_multiply(&s1, &c)?;
            let z = self.vector_add(&y, &cs1)?;

            // Check ||z||∞ < γ1 - β
            if !self.check_z_bound(&z, params.gamma1 - params.beta)? {
                kappa += 1;
                signature_attempt += 1;
                continue;
            }

            // Compute r0 = LowBits(w - cs2)
            let cs2 = self.vector_scalar_multiply(&s2, &c)?;
            let w_minus_cs2 = self.vector_subtract(&w, &cs2)?;
            let r0 = self.lowbits_vector(&w_minus_cs2, params.gamma2)?;

            // Check ||r0||∞ < γ2 - β
            if !self.check_r0_bound(&r0, params.gamma2 - params.beta)? {
                kappa += 1;
                signature_attempt += 1;
                continue;
            }

            // Compute ct0
            let ct0 = self.vector_scalar_multiply(&t0, &c)?;

            // Check ||ct0||∞ < γ2
            if !self.check_ct0_bound(&ct0, params.gamma2)? {
                kappa += 1;
                signature_attempt += 1;
                continue;
            }

            // MakeHint
            let h = self.make_hint(&w_minus_cs2, &w1, params.gamma2)?;

            // Check number of 1's in h
            if self.count_ones_in_hint(&h)? > params.tau as usize {
                kappa += 1;
                signature_attempt += 1;
                continue;
            }

            // Successful signature generation
            let signature = DilithiumSignature {
                c: c_tilde,
                z: self.encode_vector(&z, params)?,
                h: self.encode_hint(&h)?,
                algorithm: self.algorithm.clone(),
                timestamp: chrono::Utc::now(),
            };

            return Ok(signature);
        }
    }

    /// Dilithium verification
    fn dilithium_verify(&self, public_key: &DilithiumPublicKey, message: &[u8], signature: &DilithiumSignature, params: &DilithiumParams) -> Result<bool, QuantumSecurityError> {
        // Decode signature components
        let c = signature.c.clone();
        let z = self.decode_vector(&signature.z, params)?;
        let h = self.decode_hint(&signature.h)?;

        // Check ||z||∞ < γ1 - β
        if !self.check_z_bound(&z, params.gamma1 - params.beta)? {
            return Ok(false);
        }

        // Check number of 1's in h
        if self.count_ones_in_hint(&h)? > params.tau as usize {
            return Ok(false);
        }

        // Generate matrix A
        let a = self.generate_matrix_a(&public_key.rho, params)?;

        // Decode public key
        let t1 = self.decode_vector(&public_key.t1, params)?;

        // Hash public key
        let tr = self.hash_public_key(&public_key.to_bytes())?;

        // Compute message hash
        let mu = self.hash_message(&tr, message)?;

        // Decode challenge
        let c_poly = self.decode_challenge(&c, params)?;

        // Compute w' = Az - c*2^d*t1
        let az = self.matrix_vector_multiply(&a, &z)?;
        let ct1_2d = self.vector_scalar_multiply_2d(&t1, &c_poly, params.d)?;
        let w_prime = self.vector_subtract(&az, &ct1_2d)?;

        // UseHint to recover w1
        let w1 = self.use_hint(&h, &w_prime, params.gamma2)?;

        // Compute challenge c' and compare
        let c_prime = self.sample_in_ball(&mu, &w1, params)?;

        Ok(c == c_prime)
    }

    /// Expand seed into rho, rho', and k
    fn expand_seed(&self, seed: &[u8]) -> Result<([u8; 32], [u8; 64], [u8; 32]), QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(seed);
        let hash = hasher.finalize();

        let mut rho = [0u8; 32];
        rho.copy_from_slice(&hash.as_bytes()[0..32]);

        hasher.update(&[0x01]);
        let hash2 = hasher.finalize();
        let mut rho_prime = [0u8; 64];
        rho_prime[0..32].copy_from_slice(&hash2.as_bytes()[0..32]);

        hasher.update(&[0x02]);
        let hash3 = hasher.finalize();
        rho_prime[32..64].copy_from_slice(&hash3.as_bytes()[0..32]);

        hasher.update(&[0x03]);
        let hash4 = hasher.finalize();
        let mut k = [0u8; 32];
        k.copy_from_slice(&hash4.as_bytes()[0..32]);

        Ok((rho, rho_prime, k))
    }

    /// Generate matrix A from seed rho
    fn generate_matrix_a(&self, rho: &[u8], params: &DilithiumParams) -> Result<DilithiumMatrix, QuantumSecurityError> {
        let mut matrix = DilithiumMatrix {
            rows: Vec::with_capacity(params.k),
        };

        for i in 0..params.k {
            let mut row = DilithiumVector {
                polys: Vec::with_capacity(params.l),
            };

            for j in 0..params.l {
                let poly = self.generate_uniform_polynomial(rho, (i as u8, j as u8), params)?;
                row.polys.push(poly);
            }

            matrix.rows.push(row);
        }

        Ok(matrix)
    }

    /// Generate uniform polynomial from seed
    fn generate_uniform_polynomial(&self, seed: &[u8], indices: (u8, u8), params: &DilithiumParams) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(seed);
        hasher.update(&[indices.0, indices.1]);

        let mut poly = DilithiumPolynomial {
            coeffs: [0i32; 256],
        };

        let mut ctr = 0;
        let mut nonce = 0u32;

        while ctr < params.n {
            hasher.update(&nonce.to_le_bytes());
            let hash = hasher.finalize();

            for chunk in hash.as_bytes().chunks(3) {
                if ctr >= params.n {
                    break;
                }

                if chunk.len() == 3 {
                    let a0 = chunk[0] as u32;
                    let a1 = chunk[1] as u32;
                    let a2 = chunk[2] as u32;

                    let t = a0 | (a1 << 8) | (a2 << 16);
                    t &= 0x7FFFFF; // 23 bits

                    if t < params.q {
                        poly.coeffs[ctr] = t as i32;
                        ctr += 1;
                    }
                }
            }

            nonce += 1;
        }

        Ok(poly)
    }

    /// Generate secret vector from seed
    fn generate_secret_vector(&self, seed: &[u8], params: &DilithiumParams, nonce: u8) -> Result<DilithiumVector, QuantumSecurityError> {
        let size = if nonce < params.l as u8 { params.l } else { params.k };
        let mut vector = DilithiumVector {
            polys: Vec::with_capacity(size),
        };

        for i in 0..size {
            let poly = self.generate_secret_polynomial(seed, params, nonce + i as u8)?;
            vector.polys.push(poly);
        }

        Ok(vector)
    }

    /// Generate secret polynomial
    fn generate_secret_polynomial(&self, seed: &[u8], params: &DilithiumParams, nonce: u8) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(seed);
        hasher.update(&[nonce]);
        let hash = hasher.finalize();

        let mut poly = DilithiumPolynomial {
            coeffs: [0i32; 256],
        };

        // Sample small coefficients uniformly from [-eta, eta]
        for i in 0..params.n {
            let byte_idx = i / 8;
            let bit_idx = i % 8;

            if byte_idx < hash.as_bytes().len() {
                let bit = (hash.as_bytes()[byte_idx] >> bit_idx) & 1;
                poly.coeffs[i] = if bit == 1 { 
                    params.eta as i32 
                } else { 
                    -(params.eta as i32) 
                };
            }
        }

        Ok(poly)
    }

    /// Matrix-vector multiplication
    fn matrix_vector_multiply(&self, matrix: &DilithiumMatrix, vector: &DilithiumVector) -> Result<DilithiumVector, QuantumSecurityError> {
        let mut result = DilithiumVector {
            polys: Vec::with_capacity(matrix.rows.len()),
        };

        for row in &matrix.rows {
            let mut sum = DilithiumPolynomial { coeffs: [0i32; 256] };

            for (poly_a, poly_b) in row.polys.iter().zip(vector.polys.iter()) {
                let product = self.polynomial_multiply(poly_a, poly_b)?;
                sum = self.polynomial_add(&sum, &product)?;
            }

            result.polys.push(sum);
        }

        Ok(result)
    }

    /// Vector addition
    fn vector_add(&self, a: &DilithiumVector, b: &DilithiumVector) -> Result<DilithiumVector, QuantumSecurityError> {
        if a.polys.len() != b.polys.len() {
            return Err(QuantumSecurityError::InvalidParameters(
                "Vector dimensions mismatch".to_string()
            ));
        }

        let mut result = DilithiumVector {
            polys: Vec::with_capacity(a.polys.len()),
        };

        for (poly_a, poly_b) in a.polys.iter().zip(b.polys.iter()) {
            result.polys.push(self.polynomial_add(poly_a, poly_b)?);
        }

        Ok(result)
    }

    /// Vector subtraction
    fn vector_subtract(&self, a: &DilithiumVector, b: &DilithiumVector) -> Result<DilithiumVector, QuantumSecurityError> {
        if a.polys.len() != b.polys.len() {
            return Err(QuantumSecurityError::InvalidParameters(
                "Vector dimensions mismatch".to_string()
            ));
        }

        let mut result = DilithiumVector {
            polys: Vec::with_capacity(a.polys.len()),
        };

        for (poly_a, poly_b) in a.polys.iter().zip(b.polys.iter()) {
            result.polys.push(self.polynomial_subtract(poly_a, poly_b)?);
        }

        Ok(result)
    }

    /// Vector scalar multiplication
    fn vector_scalar_multiply(&self, vector: &DilithiumVector, scalar: &DilithiumPolynomial) -> Result<DilithiumVector, QuantumSecurityError> {
        let mut result = DilithiumVector {
            polys: Vec::with_capacity(vector.polys.len()),
        };

        for poly in &vector.polys {
            let product = self.polynomial_multiply(poly, scalar)?;
            result.polys.push(product);
        }

        Ok(result)
    }

    /// Vector scalar multiplication with 2^d factor
    fn vector_scalar_multiply_2d(&self, vector: &DilithiumVector, scalar: &DilithiumPolynomial, d: u8) -> Result<DilithiumVector, QuantumSecurityError> {
        let factor = 1 << d;
        let mut result = DilithiumVector {
            polys: Vec::with_capacity(vector.polys.len()),
        };

        for poly in &vector.polys {
            let mut scaled_poly = DilithiumPolynomial { coeffs: [0i32; 256] };
            for i in 0..256 {
                scaled_poly.coeffs[i] = poly.coeffs[i] * factor as i32;
            }
            let product = self.polynomial_multiply(&scaled_poly, scalar)?;
            result.polys.push(product);
        }

        Ok(result)
    }

    /// Polynomial addition
    fn polynomial_add(&self, a: &DilithiumPolynomial, b: &DilithiumPolynomial) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        let mut result = DilithiumPolynomial { coeffs: [0i32; 256] };

        for i in 0..256 {
            result.coeffs[i] = (a.coeffs[i] + b.coeffs[i]) % 8380417;
        }

        Ok(result)
    }

    /// Polynomial subtraction
    fn polynomial_subtract(&self, a: &DilithiumPolynomial, b: &DilithiumPolynomial) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        let mut result = DilithiumPolynomial { coeffs: [0i32; 256] };

        for i in 0..256 {
            result.coeffs[i] = (a.coeffs[i] - b.coeffs[i] + 8380417) % 8380417;
        }

        Ok(result)
    }

    /// Polynomial multiplication (simplified NTT)
    fn polynomial_multiply(&self, a: &DilithiumPolynomial, b: &DilithiumPolynomial) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        let mut result = DilithiumPolynomial { coeffs: [0i32; 256] };

        // Simplified multiplication - in practice, use NTT
        for i in 0..256 {
            for j in 0..256 {
                let idx = (i + j) % 256;
                let sign = if i + j >= 256 { -1 } else { 1 };
                result.coeffs[idx] = (result.coeffs[idx] + sign * a.coeffs[i] * b.coeffs[j]) % 8380417;
            }
        }

        Ok(result)
    }

    /// Power2Round decomposition
    fn power2round_vector(&self, vector: &DilithiumVector, d: u8) -> Result<(DilithiumVector, DilithiumVector), QuantumSecurityError> {
        let mut t1 = DilithiumVector { polys: Vec::new() };
        let mut t0 = DilithiumVector { polys: Vec::new() };

        for poly in &vector.polys {
            let (p1, p0) = self.power2round_polynomial(poly, d)?;
            t1.polys.push(p1);
            t0.polys.push(p0);
        }

        Ok((t1, t0))
    }

    /// Power2Round for polynomial
    fn power2round_polynomial(&self, poly: &DilithiumPolynomial, d: u8) -> Result<(DilithiumPolynomial, DilithiumPolynomial), QuantumSecurityError> {
        let mut t1 = DilithiumPolynomial { coeffs: [0i32; 256] };
        let mut t0 = DilithiumPolynomial { coeffs: [0i32; 256] };

        let factor = 1 << d;

        for i in 0..256 {
            t1.coeffs[i] = (poly.coeffs[i] + (factor / 2)) / factor;
            t0.coeffs[i] = poly.coeffs[i] - t1.coeffs[i] * factor;
        }

        Ok((t1, t0))
    }

    /// Generate y vector for signing
    fn generate_y_vector(&self, k: &[u8], mu: &[u8], kappa: u16, params: &DilithiumParams) -> Result<DilithiumVector, QuantumSecurityError> {
        use blake3::Hasher;

        let mut vector = DilithiumVector {
            polys: Vec::with_capacity(params.l),
        };

        for i in 0..params.l {
            let mut hasher = Hasher::new();
            hasher.update(k);
            hasher.update(mu);
            hasher.update(&kappa.to_le_bytes());
            hasher.update(&(i as u16).to_le_bytes());
            let hash = hasher.finalize();

            let mut poly = DilithiumPolynomial { coeffs: [0i32; 256] };

            // Generate coefficients in range [-γ1, γ1]
            for j in 0..256 {
                let byte_idx = (j * 3) % hash.as_bytes().len();
                if byte_idx + 2 < hash.as_bytes().len() {
                    let bytes = &hash.as_bytes()[byte_idx..byte_idx + 3];
                    let value = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], 0]);
                    poly.coeffs[j] = ((value % (2 * params.gamma1)) as i32) - (params.gamma1 as i32);
                }
            }

            vector.polys.push(poly);
        }

        Ok(vector)
    }

    /// Sample challenge in ball
    fn sample_in_ball(&self, mu: &[u8], w1: &DilithiumVector, params: &DilithiumParams) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(mu);
        
        // Encode w1
        let w1_bytes = self.encode_vector(w1, params)?;
        hasher.update(&w1_bytes);
        
        let hash = hasher.finalize();
        Ok(hash.as_bytes()[..32].to_vec())
    }

    /// Decode challenge polynomial
    fn decode_challenge(&self, c_bytes: &[u8], params: &DilithiumParams) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        let mut poly = DilithiumPolynomial { coeffs: [0i32; 256] };

        // Simple challenge decoding - set τ coefficients to ±1
        for i in 0..params.tau as usize {
            if i < c_bytes.len() {
                let pos = c_bytes[i] as usize % 256;
                poly.coeffs[pos] = if (c_bytes[i] & 1) == 1 { 1 } else { -1 };
            }
        }

        Ok(poly)
    }

    /// HighBits decomposition
    fn highbits_vector(&self, vector: &DilithiumVector, gamma2: u32) -> Result<DilithiumVector, QuantumSecurityError> {
        let mut result = DilithiumVector {
            polys: Vec::with_capacity(vector.polys.len()),
        };

        for poly in &vector.polys {
            let high_poly = self.highbits_polynomial(poly, gamma2)?;
            result.polys.push(high_poly);
        }

        Ok(result)
    }

    /// HighBits for polynomial
    fn highbits_polynomial(&self, poly: &DilithiumPolynomial, gamma2: u32) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        let mut result = DilithiumPolynomial { coeffs: [0i32; 256] };

        for i in 0..256 {
            result.coeffs[i] = (poly.coeffs[i] + gamma2 as i32 / 2) / gamma2 as i32;
        }

        Ok(result)
    }

    /// LowBits decomposition
    fn lowbits_vector(&self, vector: &DilithiumVector, gamma2: u32) -> Result<DilithiumVector, QuantumSecurityError> {
        let mut result = DilithiumVector {
            polys: Vec::with_capacity(vector.polys.len()),
        };

        for poly in &vector.polys {
            let low_poly = self.lowbits_polynomial(poly, gamma2)?;
            result.polys.push(low_poly);
        }

        Ok(result)
    }

    /// LowBits for polynomial
    fn lowbits_polynomial(&self, poly: &DilithiumPolynomial, gamma2: u32) -> Result<DilithiumPolynomial, QuantumSecurityError> {
        let mut result = DilithiumPolynomial { coeffs: [0i32; 256] };

        for i in 0..256 {
            let high = (poly.coeffs[i] + gamma2 as i32 / 2) / gamma2 as i32;
            result.coeffs[i] = poly.coeffs[i] - high * gamma2 as i32;
        }

        Ok(result)
    }

    /// MakeHint operation
    fn make_hint(&self, z: &DilithiumVector, r: &DilithiumVector, gamma2: u32) -> Result<Vec<Vec<bool>>, QuantumSecurityError> {
        let mut hints = Vec::new();

        for (z_poly, r_poly) in z.polys.iter().zip(r.polys.iter()) {
            let mut poly_hints = Vec::new();
            
            for i in 0..256 {
                let high_z = (z_poly.coeffs[i] + gamma2 as i32 / 2) / gamma2 as i32;
                let high_zr = ((z_poly.coeffs[i] + r_poly.coeffs[i]) + gamma2 as i32 / 2) / gamma2 as i32;
                poly_hints.push(high_z != high_zr);
            }
            
            hints.push(poly_hints);
        }

        Ok(hints)
    }

    /// UseHint operation
    fn use_hint(&self, hints: &[Vec<bool>], z: &DilithiumVector, gamma2: u32) -> Result<DilithiumVector, QuantumSecurityError> {
        let mut result = DilithiumVector {
            polys: Vec::with_capacity(z.polys.len()),
        };

        for (poly, hint) in z.polys.iter().zip(hints.iter()) {
            let mut result_poly = DilithiumPolynomial { coeffs: [0i32; 256] };
            
            for i in 0..256 {
                let high = (poly.coeffs[i] + gamma2 as i32 / 2) / gamma2 as i32;
                result_poly.coeffs[i] = if hint[i] { high + 1 } else { high };
            }
            
            result.polys.push(result_poly);
        }

        Ok(result)
    }

    /// Check bounds for z vector
    fn check_z_bound(&self, z: &DilithiumVector, bound: u32) -> Result<bool, QuantumSecurityError> {
        for poly in &z.polys {
            for &coeff in &poly.coeffs {
                if coeff.abs() >= bound as i32 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Check bounds for r0 vector
    fn check_r0_bound(&self, r0: &DilithiumVector, bound: u32) -> Result<bool, QuantumSecurityError> {
        for poly in &r0.polys {
            for &coeff in &poly.coeffs {
                if coeff.abs() >= bound as i32 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Check bounds for ct0 vector
    fn check_ct0_bound(&self, ct0: &DilithiumVector, bound: u32) -> Result<bool, QuantumSecurityError> {
        for poly in &ct0.polys {
            for &coeff in &poly.coeffs {
                if coeff.abs() >= bound as i32 {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    /// Count ones in hint
    fn count_ones_in_hint(&self, hints: &[Vec<bool>]) -> Result<usize, QuantumSecurityError> {
        let mut count = 0;
        for hint_poly in hints {
            for &hint in hint_poly {
                if hint {
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    /// Encode vector to bytes
    fn encode_vector(&self, vector: &DilithiumVector, params: &DilithiumParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();

        for poly in &vector.polys {
            for &coeff in &poly.coeffs {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }

        Ok(bytes)
    }

    /// Decode vector from bytes
    fn decode_vector(&self, bytes: &[u8], params: &DilithiumParams) -> Result<DilithiumVector, QuantumSecurityError> {
        let polys_count = bytes.len() / (256 * 4); // 4 bytes per i32 coefficient
        let mut vector = DilithiumVector {
            polys: Vec::with_capacity(polys_count),
        };

        for k in 0..polys_count {
            let mut poly = DilithiumPolynomial { coeffs: [0i32; 256] };

            for i in 0..256 {
                let byte_idx = (k * 256 + i) * 4;
                if byte_idx + 3 < bytes.len() {
                    poly.coeffs[i] = i32::from_le_bytes([
                        bytes[byte_idx],
                        bytes[byte_idx + 1],
                        bytes[byte_idx + 2],
                        bytes[byte_idx + 3],
                    ]);
                }
            }

            vector.polys.push(poly);
        }

        Ok(vector)
    }

    /// Encode hint to bytes
    fn encode_hint(&self, hints: &[Vec<bool>]) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();

        for hint_poly in hints {
            for chunk in hint_poly.chunks(8) {
                let mut byte = 0u8;
                for (i, &bit) in chunk.iter().enumerate() {
                    if bit {
                        byte |= 1 << i;
                    }
                }
                bytes.push(byte);
            }
        }

        Ok(bytes)
    }

    /// Decode hint from bytes
    fn decode_hint(&self, bytes: &[u8]) -> Result<Vec<Vec<bool>>, QuantumSecurityError> {
        let polys_count = bytes.len() / 32; // 32 bytes per polynomial (256 bits / 8)
        let mut hints = Vec::with_capacity(polys_count);

        for k in 0..polys_count {
            let mut poly_hints = Vec::with_capacity(256);

            for i in 0..32 {
                let byte_idx = k * 32 + i;
                if byte_idx < bytes.len() {
                    let byte = bytes[byte_idx];
                    for j in 0..8 {
                        poly_hints.push((byte >> j) & 1 == 1);
                    }
                }
            }

            hints.push(poly_hints);
        }

        Ok(hints)
    }

    /// Hash public key
    fn hash_public_key(&self, public_key: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(public_key);
        let hash = hasher.finalize();

        Ok(hash.as_bytes()[..32].to_vec())
    }

    /// Hash message
    fn hash_message(&self, tr: &[u8], message: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(tr);
        hasher.update(message);
        let hash = hasher.finalize();

        Ok(hash.as_bytes()[..32].to_vec())
    }

    /// Fill random bytes
    async fn fill_random_bytes(&self, bytes: &mut [u8]) -> Result<(), QuantumSecurityError> {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(bytes);
        Ok(())
    }

    /// Validate public key
    fn validate_public_key(&self, public_key: &DilithiumPublicKey) -> Result<(), QuantumSecurityError> {
        if public_key.rho.len() != 32 {
            return Err(QuantumSecurityError::InvalidKeySize);
        }
        Ok(())
    }

    /// Validate private key
    fn validate_private_key(&self, private_key: &DilithiumPrivateKey) -> Result<(), QuantumSecurityError> {
        if private_key.rho.len() != 32 || private_key.k.len() != 32 || private_key.tr.len() != 32 {
            return Err(QuantumSecurityError::InvalidKeySize);
        }
        Ok(())
    }

    /// Validate signature
    fn validate_signature(&self, signature: &DilithiumSignature) -> Result<(), QuantumSecurityError> {
        if signature.c.len() != 32 {
            return Err(QuantumSecurityError::InvalidSignatureSize);
        }
        Ok(())
    }

    /// Update key generation metrics
    async fn update_keygen_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.key_generation_count += 1;
        metrics.key_generation_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }

    /// Update signature metrics
    async fn update_signature_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.signature_count += 1;
        metrics.signature_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }

    /// Update verification metrics
    async fn update_verification_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.verification_count += 1;
        metrics.verification_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }

    /// Get current metrics
    pub async fn get_metrics(&self) -> AlgorithmMetrics {
        self.metrics.read().await.clone()
    }

    /// Health check
    pub async fn health_check(&self) -> Result<bool, QuantumSecurityError> {
        let metrics = self.get_metrics().await;
        let recent_errors = metrics.error_count > 0;
        let performance_ok = metrics.signature_count == 0 || 
            (metrics.signature_time_us / metrics.signature_count) < self.config.max_signature_time_us;
        
        Ok(!recent_errors && performance_ok)
    }
}

impl DilithiumPublicKey {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.rho);
        bytes.extend_from_slice(&self.t1);
        bytes
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 32 {
            return Err(QuantumSecurityError::InvalidData("Invalid public key length".to_string()));
        }

        let rho = bytes[0..32].to_vec();
        let t1 = bytes[32..].to_vec();

        Ok(Self {
            rho,
            t1,
            algorithm: PQCAlgorithm::Dilithium5, // Default, should be set properly
        })
    }
}

impl DilithiumSignature {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.c);
        bytes.extend_from_slice(&self.z);
        bytes.extend_from_slice(&self.h);
        bytes
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 96 {
            return Err(QuantumSecurityError::InvalidData("Invalid signature length".to_string()));
        }

        let c = bytes[0..32].to_vec();
        let z_len = (bytes.len() - 32) / 2;
        let z = bytes[32..32 + z_len].to_vec();
        let h = bytes[32 + z_len..].to_vec();

        Ok(Self {
            c,
            z,
            h,
            algorithm: PQCAlgorithm::Dilithium5, // Default, should be set properly
            timestamp: chrono::Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_dilithium_engine_creation() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = DilithiumEngine::new(&config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_dilithium_key_generation() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = DilithiumEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        assert_eq!(public_key.algorithm, PQCAlgorithm::Dilithium5);
        assert_eq!(private_key.algorithm, PQCAlgorithm::Dilithium5);
    }

    #[tokio::test]
    async fn test_dilithium_sign_verify() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = DilithiumEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Test message for Dilithium signing";
        
        let signature = engine.sign(&private_key, message).await.unwrap();
        let valid = engine.verify(&public_key, message, &signature).await.unwrap();
        
        assert!(valid);
    }

    #[tokio::test]
    async fn test_dilithium_performance() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = DilithiumEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Performance test message";
        
        // Test signing performance
        let start = std::time::Instant::now();
        let signature = engine.sign(&private_key, message).await.unwrap();
        let sign_time = start.elapsed();
        
        // Test verification performance
        let start = std::time::Instant::now();
        let valid = engine.verify(&public_key, message, &signature).await.unwrap();
        let verify_time = start.elapsed();
        
        assert!(valid);
        
        // Verify sub-100μs performance targets
        assert!(sign_time.as_micros() < 100, "Signing took {}μs", sign_time.as_micros());
        assert!(verify_time.as_micros() < 100, "Verification took {}μs", verify_time.as_micros());
    }

    #[tokio::test]
    async fn test_dilithium_health_check() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = DilithiumEngine::new(&config).await.unwrap();
        
        let health = engine.health_check().await.unwrap();
        assert!(health);
    }

    #[tokio::test]
    async fn test_dilithium_metrics() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = DilithiumEngine::new(&config).await.unwrap();
        
        let metrics_before = engine.get_metrics().await;
        assert_eq!(metrics_before.signature_count, 0);
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Test message";
        let _ = engine.sign(&private_key, message).await.unwrap();
        
        let metrics_after = engine.get_metrics().await;
        assert_eq!(metrics_after.signature_count, 1);
        assert!(metrics_after.signature_time_us > 0);
    }
}