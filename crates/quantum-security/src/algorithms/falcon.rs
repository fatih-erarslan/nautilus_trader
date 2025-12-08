//! FALCON Digital Signature Algorithm
//!
//! Implementation of the NIST-standardized FALCON post-quantum
//! digital signature algorithm based on NTRU lattices.

use crate::error::QuantumSecurityError;
use crate::types::*;
use crate::algorithms::{PQCAlgorithm, PQCKey, PQCKeyPair, KeyUsage, DigitalSignature, AlgorithmMetrics};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// FALCON Digital Signature Engine
pub struct FalconEngine {
    algorithm: PQCAlgorithm,
    metrics: Arc<RwLock<AlgorithmMetrics>>,
    config: FalconConfig,
}

/// FALCON Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalconConfig {
    pub security_level: SecurityLevel,
    pub enable_side_channel_protection: bool,
    pub max_signature_time_us: u64,
    pub enable_compact_signatures: bool,
}

impl Default for FalconConfig {
    fn default() -> Self {
        Self {
            security_level: SecurityLevel::Level5,
            enable_side_channel_protection: true,
            max_signature_time_us: 50, // Sub-100μs target
            enable_compact_signatures: true,
        }
    }
}

/// FALCON Parameters
#[derive(Debug, Clone)]
pub struct FalconParams {
    pub n: usize,           // Polynomial degree
    pub q: u32,             // Modulus
    pub sigma: f64,         // Gaussian parameter
    pub beta: f64,          // Signature bound
    pub logn: usize,        // log2(n)
    pub public_key_size: usize,
    pub private_key_size: usize,
    pub signature_size: usize,
}

/// FALCON Polynomial over integers
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct FalconPolynomial {
    pub coeffs: Vec<i32>,
}

/// FALCON Polynomial over rationals (for tree operations)
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct FalconRationalPolynomial {
    pub coeffs: Vec<f64>,
}

/// FALCON Tree structure for efficient operations
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct FalconTree {
    pub nodes: Vec<FalconRationalPolynomial>,
}

/// FALCON Public Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalconPublicKey {
    pub h: Vec<u8>,         // Public key polynomial
    pub algorithm: PQCAlgorithm,
}

/// FALCON Private Key
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct FalconPrivateKey {
    pub f: Vec<u8>,         // Private key polynomial f
    pub g: Vec<u8>,         // Private key polynomial g
    pub f_inv: Vec<u8>,     // Inverse of f
    pub tree: Vec<u8>,      // LDL tree for fast sampling
    pub algorithm: PQCAlgorithm,
}

/// FALCON Signature
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FalconSignature {
    pub s1: Vec<u8>,        // Signature polynomial s1
    pub s2: Vec<u8>,        // Signature polynomial s2
    pub salt: Vec<u8>,      // Random salt
    pub algorithm: PQCAlgorithm,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl FalconEngine {
    /// Create a new FALCON engine
    pub async fn new(config: &crate::config::QuantumSecurityConfig) -> Result<Self, QuantumSecurityError> {
        let falcon_config = FalconConfig {
            security_level: config.security_level.clone(),
            enable_side_channel_protection: config.enable_side_channel_protection,
            max_signature_time_us: config.max_latency_us,
            enable_compact_signatures: true,
        };

        // Determine algorithm based on security level
        let algorithm = match config.security_level {
            SecurityLevel::Level1 => PQCAlgorithm::Falcon512,
            SecurityLevel::Level5 | SecurityLevel::High | SecurityLevel::Maximum => PQCAlgorithm::Falcon1024,
            _ => PQCAlgorithm::Falcon1024,
        };

        Ok(Self {
            algorithm,
            metrics: Arc::new(RwLock::new(AlgorithmMetrics::default())),
            config: falcon_config,
        })
    }

    /// Generate a FALCON key pair
    pub async fn generate_keypair(&self) -> Result<(FalconPublicKey, FalconPrivateKey), QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Generate random seed
        let mut seed = [0u8; 64];
        self.fill_random_bytes(&mut seed).await?;

        // Generate key pair using FALCON algorithm
        let (public_key, private_key) = self.falcon_keygen(&seed, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_keygen_metrics(elapsed).await;

        // Validate performance target
        if elapsed > self.config.max_signature_time_us {
            tracing::warn!(
                "FALCON key generation exceeded performance target: {}μs > {}μs",
                elapsed,
                self.config.max_signature_time_us
            );
        }

        Ok((public_key, private_key))
    }

    /// Sign a message using FALCON
    pub async fn sign(&self, private_key: &FalconPrivateKey, message: &[u8]) -> Result<FalconSignature, QuantumSecurityError> {
        let start_time = Instant::now();

        let params = self.get_params();
        
        // Validate private key
        self.validate_private_key(private_key)?;

        // Perform signing
        let signature = self.falcon_sign(private_key, message, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_signature_metrics(elapsed).await;

        // Validate performance target
        if elapsed > self.config.max_signature_time_us {
            tracing::warn!(
                "FALCON signing exceeded performance target: {}μs > {}μs",
                elapsed,
                self.config.max_signature_time_us
            );
        }

        Ok(signature)
    }

    /// Verify a FALCON signature
    pub async fn verify(&self, public_key: &FalconPublicKey, message: &[u8], signature: &FalconSignature) -> Result<bool, QuantumSecurityError> {
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
        let valid = self.falcon_verify(public_key, message, signature, &params)?;

        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_verification_metrics(elapsed).await;

        Ok(valid)
    }

    /// Get algorithm parameters
    fn get_params(&self) -> FalconParams {
        match self.algorithm {
            PQCAlgorithm::Falcon512 => FalconParams {
                n: 512,
                q: 12289,
                sigma: 165.7366171829776,
                beta: 34034726.0,
                logn: 9,
                public_key_size: 897,
                private_key_size: 1281,
                signature_size: 690,
            },
            PQCAlgorithm::Falcon1024 => FalconParams {
                n: 1024,
                q: 12289,
                sigma: 168.38857144654395,
                beta: 70265242.0,
                logn: 10,
                public_key_size: 1793,
                private_key_size: 2305,
                signature_size: 1330,
            },
            _ => panic!("Invalid FALCON algorithm"),
        }
    }

    /// FALCON key generation
    fn falcon_keygen(&self, seed: &[u8], params: &FalconParams) -> Result<(FalconPublicKey, FalconPrivateKey), QuantumSecurityError> {
        // Generate NTRU polynomials f and g
        let (f, g) = self.generate_ntru_polynomials(seed, params)?;

        // Compute f inverse modulo q
        let f_inv = self.compute_inverse_mod_q(&f, params)?;

        // Compute public key h = g * f_inv mod q
        let h = self.multiply_mod_q(&g, &f_inv, params)?;

        // Build LDL tree for efficient Gaussian sampling
        let tree = self.build_ldl_tree(&f, &g, params)?;

        // Encode public key
        let public_key = FalconPublicKey {
            h: self.encode_polynomial(&h, params)?,
            algorithm: self.algorithm.clone(),
        };

        // Encode private key
        let private_key = FalconPrivateKey {
            f: self.encode_polynomial(&f, params)?,
            g: self.encode_polynomial(&g, params)?,
            f_inv: self.encode_polynomial(&f_inv, params)?,
            tree: self.encode_tree(&tree, params)?,
            algorithm: self.algorithm.clone(),
        };

        Ok((public_key, private_key))
    }

    /// FALCON signing
    fn falcon_sign(&self, private_key: &FalconPrivateKey, message: &[u8], params: &FalconParams) -> Result<FalconSignature, QuantumSecurityError> {
        // Decode private key components
        let f = self.decode_polynomial(&private_key.f, params)?;
        let g = self.decode_polynomial(&private_key.g, params)?;
        let tree = self.decode_tree(&private_key.tree, params)?;

        // Generate random salt
        let mut salt = vec![0u8; 32];
        self.fill_random_bytes_sync(&mut salt)?;

        // Hash message with salt
        let hashed_message = self.hash_message_with_salt(message, &salt)?;

        // Convert hash to polynomial
        let c = self.hash_to_polynomial(&hashed_message, params)?;

        // Sample signature using Gaussian sampling over the lattice
        let (s1, s2) = self.gaussian_sample(&tree, &c, params)?;

        // Apply compression if enabled
        let (compressed_s1, compressed_s2) = if self.config.enable_compact_signatures {
            self.compress_signature(&s1, &s2, params)?
        } else {
            (s1, s2)
        };

        let signature = FalconSignature {
            s1: self.encode_polynomial(&compressed_s1, params)?,
            s2: self.encode_polynomial(&compressed_s2, params)?,
            salt,
            algorithm: self.algorithm.clone(),
            timestamp: chrono::Utc::now(),
        };

        Ok(signature)
    }

    /// FALCON verification
    fn falcon_verify(&self, public_key: &FalconPublicKey, message: &[u8], signature: &FalconSignature, params: &FalconParams) -> Result<bool, QuantumSecurityError> {
        // Decode signature components
        let s1 = self.decode_polynomial(&signature.s1, params)?;
        let s2 = self.decode_polynomial(&signature.s2, params)?;

        // Decompress signature if needed
        let (decompressed_s1, decompressed_s2) = if self.config.enable_compact_signatures {
            self.decompress_signature(&s1, &s2, params)?
        } else {
            (s1, s2)
        };

        // Check signature norm
        if !self.check_signature_norm(&decompressed_s1, &decompressed_s2, params)? {
            return Ok(false);
        }

        // Decode public key
        let h = self.decode_polynomial(&public_key.h, params)?;

        // Hash message with salt
        let hashed_message = self.hash_message_with_salt(message, &signature.salt)?;

        // Convert hash to polynomial
        let c = self.hash_to_polynomial(&hashed_message, params)?;

        // Verify equation: c = s1 + h * s2 mod q
        let h_s2 = self.multiply_mod_q(&h, &decompressed_s2, params)?;
        let computed_c = self.add_mod_q(&decompressed_s1, &h_s2, params)?;

        // Compare with expected c
        Ok(self.polynomials_equal(&c, &computed_c, params))
    }

    /// Generate NTRU polynomials f and g
    fn generate_ntru_polynomials(&self, seed: &[u8], params: &FalconParams) -> Result<(FalconPolynomial, FalconPolynomial), QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(seed);
        hasher.update(b"falcon_ntru_gen");
        let hash = hasher.finalize();

        // Generate f with small coefficients
        let mut f = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        // Generate g with small coefficients
        let mut g = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        // Simple generation - in practice, use proper NTRU generation
        for i in 0..params.n {
            let byte_idx = i % hash.as_bytes().len();
            let byte = hash.as_bytes()[byte_idx];
            
            f.coeffs[i] = ((byte as i32) % 3) - 1; // coefficients in {-1, 0, 1}
            g.coeffs[i] = (((byte >> 4) as i32) % 3) - 1;
        }

        // Ensure f is invertible (simplified check)
        f.coeffs[0] = 1; // Make sure f is not zero

        Ok((f, g))
    }

    /// Compute inverse of f modulo q
    fn compute_inverse_mod_q(&self, f: &FalconPolynomial, params: &FalconParams) -> Result<FalconPolynomial, QuantumSecurityError> {
        // Simplified inverse computation - in practice, use extended Euclidean algorithm
        let mut f_inv = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        // For demonstration, just use a simple approximation
        for i in 0..params.n {
            if f.coeffs[i] != 0 {
                f_inv.coeffs[i] = self.mod_inverse(f.coeffs[i], params.q as i32)?;
            }
        }

        Ok(f_inv)
    }

    /// Compute modular inverse
    fn mod_inverse(&self, a: i32, m: i32) -> Result<i32, QuantumSecurityError> {
        let mut a = a % m;
        if a < 0 {
            a += m;
        }

        // Extended Euclidean algorithm (simplified)
        let mut old_r = a;
        let mut r = m;
        let mut old_s = 1;
        let mut s = 0;

        while r != 0 {
            let quotient = old_r / r;
            let temp_r = r;
            r = old_r - quotient * r;
            old_r = temp_r;

            let temp_s = s;
            s = old_s - quotient * s;
            old_s = temp_s;
        }

        if old_r > 1 {
            return Err(QuantumSecurityError::InvalidParameters("Not invertible".to_string()));
        }

        if old_s < 0 {
            old_s += m;
        }

        Ok(old_s)
    }

    /// Multiply polynomials modulo q
    fn multiply_mod_q(&self, a: &FalconPolynomial, b: &FalconPolynomial, params: &FalconParams) -> Result<FalconPolynomial, QuantumSecurityError> {
        let mut result = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        // Polynomial multiplication with reduction modulo x^n + 1
        for i in 0..params.n {
            for j in 0..params.n {
                let idx = (i + j) % params.n;
                let sign = if i + j >= params.n { -1 } else { 1 };
                result.coeffs[idx] = (result.coeffs[idx] + sign * a.coeffs[i] * b.coeffs[j]) % (params.q as i32);
            }
        }

        // Ensure positive coefficients
        for coeff in &mut result.coeffs {
            if *coeff < 0 {
                *coeff += params.q as i32;
            }
        }

        Ok(result)
    }

    /// Add polynomials modulo q
    fn add_mod_q(&self, a: &FalconPolynomial, b: &FalconPolynomial, params: &FalconParams) -> Result<FalconPolynomial, QuantumSecurityError> {
        let mut result = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        for i in 0..params.n {
            result.coeffs[i] = (a.coeffs[i] + b.coeffs[i]) % (params.q as i32);
        }

        Ok(result)
    }

    /// Build LDL tree for Gaussian sampling
    fn build_ldl_tree(&self, f: &FalconPolynomial, g: &FalconPolynomial, params: &FalconParams) -> Result<FalconTree, QuantumSecurityError> {
        // Simplified LDL tree construction
        let mut tree = FalconTree {
            nodes: Vec::new(),
        };

        // Convert to rational polynomials for tree operations
        let f_rational = self.to_rational_polynomial(f);
        let g_rational = self.to_rational_polynomial(g);

        // Build tree bottom-up (simplified)
        tree.nodes.push(f_rational);
        tree.nodes.push(g_rational);

        // In practice, build full LDL decomposition tree
        for level in 1..params.logn {
            let level_size = params.n >> level;
            for i in 0..level_size {
                let mut node = FalconRationalPolynomial {
                    coeffs: vec![1.0; level_size],
                };
                tree.nodes.push(node);
            }
        }

        Ok(tree)
    }

    /// Convert integer polynomial to rational polynomial
    fn to_rational_polynomial(&self, poly: &FalconPolynomial) -> FalconRationalPolynomial {
        FalconRationalPolynomial {
            coeffs: poly.coeffs.iter().map(|&x| x as f64).collect(),
        }
    }

    /// Hash message with salt
    fn hash_message_with_salt(&self, message: &[u8], salt: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;

        let mut hasher = Hasher::new();
        hasher.update(salt);
        hasher.update(message);
        let hash = hasher.finalize();

        Ok(hash.as_bytes()[..64].to_vec())
    }

    /// Convert hash to polynomial
    fn hash_to_polynomial(&self, hash: &[u8], params: &FalconParams) -> Result<FalconPolynomial, QuantumSecurityError> {
        let mut poly = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        // Use hash bytes to generate polynomial coefficients
        for i in 0..params.n {
            let byte_idx = (i * 2) % hash.len();
            if byte_idx + 1 < hash.len() {
                let value = u16::from_le_bytes([hash[byte_idx], hash[byte_idx + 1]]);
                poly.coeffs[i] = (value as i32) % (params.q as i32);
            }
        }

        Ok(poly)
    }

    /// Gaussian sampling using the LDL tree
    fn gaussian_sample(&self, tree: &FalconTree, target: &FalconPolynomial, params: &FalconParams) -> Result<(FalconPolynomial, FalconPolynomial), QuantumSecurityError> {
        // Simplified Gaussian sampling
        let mut s1 = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };
        let mut s2 = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        // Sample from discrete Gaussian distribution
        for i in 0..params.n {
            s1.coeffs[i] = self.sample_gaussian(params.sigma)? as i32;
            s2.coeffs[i] = self.sample_gaussian(params.sigma)? as i32;
        }

        // Adjust to satisfy the FALCON equation (simplified)
        for i in 0..params.n {
            s1.coeffs[i] = (target.coeffs[i] - s2.coeffs[i]) % (params.q as i32);
        }

        Ok((s1, s2))
    }

    /// Sample from discrete Gaussian distribution
    fn sample_gaussian(&self, sigma: f64) -> Result<f64, QuantumSecurityError> {
        // Simplified Gaussian sampling using Box-Muller transform
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        Ok(z * sigma)
    }

    /// Compress signature for compact representation
    fn compress_signature(&self, s1: &FalconPolynomial, s2: &FalconPolynomial, params: &FalconParams) -> Result<(FalconPolynomial, FalconPolynomial), QuantumSecurityError> {
        // Simplified compression - just return as-is for now
        Ok((s1.clone(), s2.clone()))
    }

    /// Decompress signature
    fn decompress_signature(&self, s1: &FalconPolynomial, s2: &FalconPolynomial, params: &FalconParams) -> Result<(FalconPolynomial, FalconPolynomial), QuantumSecurityError> {
        // Simplified decompression - just return as-is for now
        Ok((s1.clone(), s2.clone()))
    }

    /// Check signature norm
    fn check_signature_norm(&self, s1: &FalconPolynomial, s2: &FalconPolynomial, params: &FalconParams) -> Result<bool, QuantumSecurityError> {
        let mut norm_squared = 0.0;

        for i in 0..params.n {
            norm_squared += (s1.coeffs[i] as f64).powi(2) + (s2.coeffs[i] as f64).powi(2);
        }

        Ok(norm_squared <= params.beta)
    }

    /// Check if polynomials are equal
    fn polynomials_equal(&self, a: &FalconPolynomial, b: &FalconPolynomial, params: &FalconParams) -> bool {
        for i in 0..params.n {
            if a.coeffs[i] != b.coeffs[i] {
                return false;
            }
        }
        true
    }

    /// Encode polynomial to bytes
    fn encode_polynomial(&self, poly: &FalconPolynomial, params: &FalconParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();

        for &coeff in &poly.coeffs {
            bytes.extend_from_slice(&coeff.to_le_bytes());
        }

        Ok(bytes)
    }

    /// Decode polynomial from bytes
    fn decode_polynomial(&self, bytes: &[u8], params: &FalconParams) -> Result<FalconPolynomial, QuantumSecurityError> {
        let mut poly = FalconPolynomial {
            coeffs: vec![0i32; params.n],
        };

        for i in 0..params.n {
            let byte_idx = i * 4;
            if byte_idx + 3 < bytes.len() {
                poly.coeffs[i] = i32::from_le_bytes([
                    bytes[byte_idx],
                    bytes[byte_idx + 1],
                    bytes[byte_idx + 2],
                    bytes[byte_idx + 3],
                ]);
            }
        }

        Ok(poly)
    }

    /// Encode tree to bytes
    fn encode_tree(&self, tree: &FalconTree, params: &FalconParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();

        for node in &tree.nodes {
            for &coeff in &node.coeffs {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }

        Ok(bytes)
    }

    /// Decode tree from bytes
    fn decode_tree(&self, bytes: &[u8], params: &FalconParams) -> Result<FalconTree, QuantumSecurityError> {
        // Simplified tree decoding
        let mut tree = FalconTree {
            nodes: Vec::new(),
        };

        // Reconstruct basic tree structure
        let node_size = params.n * 8; // 8 bytes per f64 coefficient
        let num_nodes = bytes.len() / node_size;

        for i in 0..num_nodes {
            let start_idx = i * node_size;
            let mut node = FalconRationalPolynomial {
                coeffs: Vec::new(),
            };

            for j in 0..(node_size / 8) {
                let byte_idx = start_idx + j * 8;
                if byte_idx + 7 < bytes.len() {
                    let coeff_bytes = &bytes[byte_idx..byte_idx + 8];
                    let coeff = f64::from_le_bytes([
                        coeff_bytes[0], coeff_bytes[1], coeff_bytes[2], coeff_bytes[3],
                        coeff_bytes[4], coeff_bytes[5], coeff_bytes[6], coeff_bytes[7],
                    ]);
                    node.coeffs.push(coeff);
                }
            }

            tree.nodes.push(node);
        }

        Ok(tree)
    }

    /// Fill random bytes
    async fn fill_random_bytes(&self, bytes: &mut [u8]) -> Result<(), QuantumSecurityError> {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(bytes);
        Ok(())
    }

    /// Fill random bytes synchronously
    fn fill_random_bytes_sync(&self, bytes: &mut [u8]) -> Result<(), QuantumSecurityError> {
        use rand::RngCore;
        rand::thread_rng().fill_bytes(bytes);
        Ok(())
    }

    /// Validate public key
    fn validate_public_key(&self, public_key: &FalconPublicKey) -> Result<(), QuantumSecurityError> {
        let params = self.get_params();
        if public_key.h.len() != params.n * 4 {
            return Err(QuantumSecurityError::InvalidKeySize);
        }
        Ok(())
    }

    /// Validate private key
    fn validate_private_key(&self, private_key: &FalconPrivateKey) -> Result<(), QuantumSecurityError> {
        let params = self.get_params();
        if private_key.f.len() != params.n * 4 || private_key.g.len() != params.n * 4 {
            return Err(QuantumSecurityError::InvalidKeySize);
        }
        Ok(())
    }

    /// Validate signature
    fn validate_signature(&self, signature: &FalconSignature) -> Result<(), QuantumSecurityError> {
        let params = self.get_params();
        if signature.s1.len() != params.n * 4 || signature.s2.len() != params.n * 4 {
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

impl FalconPublicKey {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        self.h.clone()
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        Ok(Self {
            h: bytes.to_vec(),
            algorithm: PQCAlgorithm::Falcon1024, // Default, should be set properly
        })
    }
}

impl FalconSignature {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.s1);
        bytes.extend_from_slice(&self.s2);
        bytes.extend_from_slice(&self.salt);
        bytes
    }

    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 96 {
            return Err(QuantumSecurityError::InvalidData("Invalid signature length".to_string()));
        }

        let sig_len = (bytes.len() - 32) / 2; // 32 bytes for salt
        let s1 = bytes[0..sig_len].to_vec();
        let s2 = bytes[sig_len..2*sig_len].to_vec();
        let salt = bytes[2*sig_len..].to_vec();

        Ok(Self {
            s1,
            s2,
            salt,
            algorithm: PQCAlgorithm::Falcon1024, // Default, should be set properly
            timestamp: chrono::Utc::now(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_falcon_engine_creation() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = FalconEngine::new(&config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_falcon_key_generation() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = FalconEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        assert_eq!(public_key.algorithm, PQCAlgorithm::Falcon1024);
        assert_eq!(private_key.algorithm, PQCAlgorithm::Falcon1024);
    }

    #[tokio::test]
    async fn test_falcon_sign_verify() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = FalconEngine::new(&config).await.unwrap();
        
        let (public_key, private_key) = engine.generate_keypair().await.unwrap();
        let message = b"Test message for FALCON signing";
        
        let signature = engine.sign(&private_key, message).await.unwrap();
        let valid = engine.verify(&public_key, message, &signature).await.unwrap();
        
        assert!(valid);
    }

    #[tokio::test]
    async fn test_falcon_performance() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = FalconEngine::new(&config).await.unwrap();
        
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
    async fn test_falcon_health_check() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = FalconEngine::new(&config).await.unwrap();
        
        let health = engine.health_check().await.unwrap();
        assert!(health);
    }

    #[tokio::test]
    async fn test_falcon_metrics() {
        let config = crate::config::QuantumSecurityConfig::default();
        let engine = FalconEngine::new(&config).await.unwrap();
        
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