//! CRYSTALS-Kyber Implementation
//!
//! CRYSTALS-Kyber is a key encapsulation mechanism (KEM) selected by NIST
//! for standardization. It provides IND-CCA2 security based on the hardness
//! of the Module-LWE problem.

use crate::error::QuantumSecurityError;
use crate::types::*;
use crate::algorithms::{PQCAlgorithm, PQCKey, PQCKeyPair, KeyUsage, EncapsulatedKey, AlgorithmMetrics};
use rand::{CryptoRng, RngCore};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// CRYSTALS-Kyber Key Encapsulation Mechanism
pub struct CrystalsKyber {
    algorithm: PQCAlgorithm,
    metrics: Arc<RwLock<AlgorithmMetrics>>,
}

/// Kyber Parameters
#[derive(Debug, Clone)]
pub struct KyberParams {
    pub n: usize,           // Polynomial degree
    pub k: usize,           // Module rank
    pub q: u16,             // Modulus
    pub eta1: u8,           // Noise parameter 1
    pub eta2: u8,           // Noise parameter 2
    pub du: u8,             // Compression parameter u
    pub dv: u8,             // Compression parameter v
    pub public_key_size: usize,
    pub private_key_size: usize,
    pub ciphertext_size: usize,
    pub shared_secret_size: usize,
}

/// Kyber Polynomial
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KyberPolynomial {
    pub coeffs: [u16; 256],
}

/// Kyber Vector
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KyberVector {
    pub polys: Vec<KyberPolynomial>,
}

/// Kyber Matrix
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KyberMatrix {
    pub rows: Vec<KyberVector>,
}

/// Kyber Public Key
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyberPublicKey {
    pub t: Vec<u8>,         // Public key polynomial vector
    pub rho: Vec<u8>,       // Public seed
    pub algorithm: PQCAlgorithm,
}

/// Kyber Private Key
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct KyberPrivateKey {
    pub s: Vec<u8>,         // Private key polynomial vector
    pub pk: Vec<u8>,        // Corresponding public key
    pub h: Vec<u8>,         // Hash of public key
    pub z: Vec<u8>,         // Random value for CCA security
    pub algorithm: PQCAlgorithm,
}

/// Kyber Ciphertext
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KyberCiphertext {
    pub c1: Vec<u8>,        // Ciphertext polynomial vector
    pub c2: Vec<u8>,        // Ciphertext polynomial
    pub algorithm: PQCAlgorithm,
}

impl CrystalsKyber {
    /// Create new CRYSTALS-Kyber instance
    pub fn new(algorithm: PQCAlgorithm) -> Result<Self, QuantumSecurityError> {
        if !algorithm.is_kem() {
            return Err(QuantumSecurityError::InvalidAlgorithm(
                format!("Algorithm {:?} is not a KEM", algorithm)
            ));
        }
        
        Ok(Self {
            algorithm,
            metrics: Arc::new(RwLock::new(AlgorithmMetrics::default())),
        })
    }
    
    /// Generate a new key pair
    pub async fn generate_keypair<R: CryptoRng + RngCore>(
        &self,
        rng: &mut R,
    ) -> Result<PQCKeyPair, QuantumSecurityError> {
        let start_time = Instant::now();
        
        let params = self.get_params();
        
        // Generate random seed
        let mut seed = vec![0u8; 32];
        rng.fill_bytes(&mut seed);
        
        // Generate key pair using Kyber algorithm
        let (public_key, private_key) = self.kyber_keygen(&seed, &params)?;
        
        // Create PQC key structures
        let pqc_public_key = PQCKey::KyberPublicKey {
            algorithm: self.algorithm.clone(),
            key_data: public_key.to_bytes(),
            created_at: chrono::Utc::now(),
        };
        
        let pqc_private_key = PQCKey::KyberPrivateKey {
            algorithm: self.algorithm.clone(),
            key_data: SecureBytes::new(private_key.to_bytes()),
            created_at: chrono::Utc::now(),
        };
        
        let keypair = PQCKeyPair::new(
            pqc_public_key,
            pqc_private_key,
            self.algorithm.clone(),
            KeyUsage::KeyEncapsulation,
        );
        
        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_keygen_metrics(elapsed).await;
        
        Ok(keypair)
    }
    
    /// Encapsulate a shared secret
    pub async fn encapsulate<R: CryptoRng + RngCore>(
        &self,
        public_key: &PQCKey,
        rng: &mut R,
    ) -> Result<EncapsulatedKey, QuantumSecurityError> {
        let start_time = Instant::now();
        
        let kyber_pk = self.extract_kyber_public_key(public_key)?;
        let params = self.get_params();
        
        // Generate random message
        let mut message = vec![0u8; 32];
        rng.fill_bytes(&mut message);
        
        // Perform encapsulation
        let (ciphertext, shared_secret) = self.kyber_encaps(&kyber_pk, &message, &params)?;
        
        let encapsulated_key = EncapsulatedKey {
            ciphertext: ciphertext.to_bytes(),
            shared_secret: SecureBytes::new(shared_secret),
            algorithm: self.algorithm.clone(),
            timestamp: chrono::Utc::now(),
        };
        
        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_encaps_metrics(elapsed).await;
        
        Ok(encapsulated_key)
    }
    
    /// Decapsulate a shared secret
    pub async fn decapsulate(
        &self,
        private_key: &PQCKey,
        encapsulated_key: &EncapsulatedKey,
    ) -> Result<SecureBytes, QuantumSecurityError> {
        let start_time = Instant::now();
        
        let kyber_sk = self.extract_kyber_private_key(private_key)?;
        let ciphertext = KyberCiphertext::from_bytes(&encapsulated_key.ciphertext)?;
        let params = self.get_params();
        
        // Perform decapsulation
        let shared_secret = self.kyber_decaps(&kyber_sk, &ciphertext, &params)?;
        
        // Update metrics
        let elapsed = start_time.elapsed().as_micros() as u64;
        self.update_decaps_metrics(elapsed).await;
        
        Ok(SecureBytes::new(shared_secret))
    }
    
    /// Get algorithm parameters
    fn get_params(&self) -> KyberParams {
        match self.algorithm {
            PQCAlgorithm::Kyber512 => KyberParams {
                n: 256,
                k: 2,
                q: 3329,
                eta1: 3,
                eta2: 2,
                du: 10,
                dv: 4,
                public_key_size: 800,
                private_key_size: 1632,
                ciphertext_size: 768,
                shared_secret_size: 32,
            },
            PQCAlgorithm::Kyber768 => KyberParams {
                n: 256,
                k: 3,
                q: 3329,
                eta1: 2,
                eta2: 2,
                du: 10,
                dv: 4,
                public_key_size: 1184,
                private_key_size: 2400,
                ciphertext_size: 1088,
                shared_secret_size: 32,
            },
            PQCAlgorithm::Kyber1024 => KyberParams {
                n: 256,
                k: 4,
                q: 3329,
                eta1: 2,
                eta2: 2,
                du: 11,
                dv: 5,
                public_key_size: 1568,
                private_key_size: 3168,
                ciphertext_size: 1568,
                shared_secret_size: 32,
            },
            _ => panic!("Invalid Kyber algorithm"),
        }
    }
    
    /// Kyber key generation
    fn kyber_keygen(
        &self,
        seed: &[u8],
        params: &KyberParams,
    ) -> Result<(KyberPublicKey, KyberPrivateKey), QuantumSecurityError> {
        // Expand seed to generate matrix A and noise vectors
        let (rho, sigma) = self.expand_seed(seed)?;
        
        // Generate matrix A from rho
        let a = self.generate_matrix_a(&rho, params)?;
        
        // Generate secret vector s and error vector e
        let s = self.generate_noise_vector(&sigma, params, 0)?;
        let e = self.generate_noise_vector(&sigma, params, params.k as u8)?;
        
        // Compute t = As + e
        let t = self.matrix_vector_multiply(&a, &s)?;
        let t = self.vector_add(&t, &e)?;
        
        // Encode public key
        let public_key = KyberPublicKey {
            t: self.encode_vector(&t, params)?,
            rho: rho.to_vec(),
            algorithm: self.algorithm.clone(),
        };
        
        // Encode private key
        let private_key = KyberPrivateKey {
            s: self.encode_vector(&s, params)?,
            pk: public_key.to_bytes(),
            h: self.hash_public_key(&public_key.to_bytes())?,
            z: self.generate_random_z()?,
            algorithm: self.algorithm.clone(),
        };
        
        Ok((public_key, private_key))
    }
    
    /// Kyber encapsulation
    fn kyber_encaps(
        &self,
        public_key: &KyberPublicKey,
        message: &[u8],
        params: &KyberParams,
    ) -> Result<(KyberCiphertext, Vec<u8>), QuantumSecurityError> {
        // Decode public key
        let t = self.decode_vector(&public_key.t, params)?;
        let a = self.generate_matrix_a(&public_key.rho, params)?;
        
        // Generate random coins
        let coins = self.generate_coins(message, &public_key.to_bytes())?;
        
        // Generate noise vectors
        let r = self.generate_noise_vector(&coins, params, 0)?;
        let e1 = self.generate_noise_vector(&coins, params, params.k as u8)?;
        let e2 = self.generate_noise_polynomial(&coins, params, (2 * params.k) as u8)?;
        
        // Compute ciphertext
        let u = self.matrix_transpose_vector_multiply(&a, &r)?;
        let u = self.vector_add(&u, &e1)?;
        
        let v = self.vector_dot_product(&t, &r)?;
        let v = self.polynomial_add(&v, &e2)?;
        let v = self.polynomial_add(&v, &self.decode_message(message, params)?)?;
        
        let ciphertext = KyberCiphertext {
            c1: self.compress_vector(&u, params.du)?,
            c2: self.compress_polynomial(&v, params.dv)?,
            algorithm: self.algorithm.clone(),
        };
        
        // Derive shared secret
        let shared_secret = self.kdf(&ciphertext.to_bytes(), &public_key.to_bytes())?;
        
        Ok((ciphertext, shared_secret))
    }
    
    /// Kyber decapsulation
    fn kyber_decaps(
        &self,
        private_key: &KyberPrivateKey,
        ciphertext: &KyberCiphertext,
        params: &KyberParams,
    ) -> Result<Vec<u8>, QuantumSecurityError> {
        // Decode private key
        let s = self.decode_vector(&private_key.s, params)?;
        
        // Decompress ciphertext
        let u = self.decompress_vector(&ciphertext.c1, params.du)?;
        let v = self.decompress_polynomial(&ciphertext.c2, params.dv)?;
        
        // Compute message
        let vs = self.vector_dot_product(&s, &u)?;
        let message_poly = self.polynomial_subtract(&v, &vs)?;
        let message = self.encode_message(&message_poly, params)?;
        
        // Re-encrypt to verify
        let public_key = KyberPublicKey::from_bytes(&private_key.pk)?;
        let (ciphertext_prime, _) = self.kyber_encaps(&public_key, &message, params)?;
        
        // Check if ciphertexts match (CCA security)
        let shared_secret = if ciphertext.to_bytes() == ciphertext_prime.to_bytes() {
            self.kdf(&ciphertext.to_bytes(), &private_key.pk)?
        } else {
            self.kdf(&ciphertext.to_bytes(), &private_key.z)?
        };
        
        Ok(shared_secret)
    }
    
    /// Expand seed into rho and sigma
    fn expand_seed(&self, seed: &[u8]) -> Result<([u8; 32], [u8; 32]), QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(seed);
        let hash = hasher.finalize();
        
        let mut rho = [0u8; 32];
        let mut sigma = [0u8; 32];
        
        rho.copy_from_slice(&hash.as_bytes()[0..32]);
        hasher.update(&[0x01]);
        let hash2 = hasher.finalize();
        sigma.copy_from_slice(&hash2.as_bytes()[0..32]);
        
        Ok((rho, sigma))
    }
    
    /// Generate matrix A from seed rho
    fn generate_matrix_a(
        &self,
        rho: &[u8],
        params: &KyberParams,
    ) -> Result<KyberMatrix, QuantumSecurityError> {
        let mut matrix = KyberMatrix {
            rows: Vec::with_capacity(params.k),
        };
        
        for i in 0..params.k {
            let mut row = KyberVector {
                polys: Vec::with_capacity(params.k),
            };
            
            for j in 0..params.k {
                let poly = self.generate_uniform_polynomial(rho, (i as u8, j as u8), params)?;
                row.polys.push(poly);
            }
            
            matrix.rows.push(row);
        }
        
        Ok(matrix)
    }
    
    /// Generate uniform polynomial from seed
    fn generate_uniform_polynomial(
        &self,
        seed: &[u8],
        indices: (u8, u8),
        params: &KyberParams,
    ) -> Result<KyberPolynomial, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(seed);
        hasher.update(&[indices.0, indices.1]);
        
        let mut poly = KyberPolynomial {
            coeffs: [0u16; 256],
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
                    let d1 = (chunk[0] as u16) | ((chunk[1] as u16 & 0x0f) << 8);
                    let d2 = ((chunk[1] as u16) >> 4) | ((chunk[2] as u16) << 4);
                    
                    if d1 < params.q {
                        poly.coeffs[ctr] = d1;
                        ctr += 1;
                    }
                    if ctr < params.n && d2 < params.q {
                        poly.coeffs[ctr] = d2;
                        ctr += 1;
                    }
                }
            }
            
            nonce += 1;
        }
        
        Ok(poly)
    }
    
    /// Generate noise vector
    fn generate_noise_vector(
        &self,
        seed: &[u8],
        params: &KyberParams,
        nonce: u8,
    ) -> Result<KyberVector, QuantumSecurityError> {
        let mut vector = KyberVector {
            polys: Vec::with_capacity(params.k),
        };
        
        for i in 0..params.k {
            let poly = self.generate_noise_polynomial(seed, params, nonce + i as u8)?;
            vector.polys.push(poly);
        }
        
        Ok(vector)
    }
    
    /// Generate noise polynomial
    fn generate_noise_polynomial(
        &self,
        seed: &[u8],
        params: &KyberParams,
        nonce: u8,
    ) -> Result<KyberPolynomial, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(seed);
        hasher.update(&[nonce]);
        let hash = hasher.finalize();
        
        let mut poly = KyberPolynomial {
            coeffs: [0u16; 256],
        };
        
        // Generate small coefficients using rejection sampling
        for i in 0..params.n {
            let byte_idx = i / 4;
            let bit_idx = (i % 4) * 2;
            
            if byte_idx < hash.as_bytes().len() {
                let bits = (hash.as_bytes()[byte_idx] >> bit_idx) & 0x03;
                let coeff = match bits {
                    0 => 0,
                    1 => 1,
                    2 => params.q - 1,
                    3 => params.q - 2,
                    _ => 0,
                };
                poly.coeffs[i] = coeff;
            }
        }
        
        Ok(poly)
    }
    
    /// Matrix-vector multiplication
    fn matrix_vector_multiply(
        &self,
        matrix: &KyberMatrix,
        vector: &KyberVector,
    ) -> Result<KyberVector, QuantumSecurityError> {
        let mut result = KyberVector {
            polys: Vec::with_capacity(matrix.rows.len()),
        };
        
        for row in &matrix.rows {
            let mut sum = KyberPolynomial { coeffs: [0u16; 256] };
            
            for (poly_a, poly_b) in row.polys.iter().zip(vector.polys.iter()) {
                let product = self.polynomial_multiply(poly_a, poly_b)?;
                sum = self.polynomial_add(&sum, &product)?;
            }
            
            result.polys.push(sum);
        }
        
        Ok(result)
    }
    
    /// Matrix transpose-vector multiplication
    fn matrix_transpose_vector_multiply(
        &self,
        matrix: &KyberMatrix,
        vector: &KyberVector,
    ) -> Result<KyberVector, QuantumSecurityError> {
        let mut result = KyberVector {
            polys: Vec::with_capacity(matrix.rows[0].polys.len()),
        };
        
        for j in 0..matrix.rows[0].polys.len() {
            let mut sum = KyberPolynomial { coeffs: [0u16; 256] };
            
            for (i, row) in matrix.rows.iter().enumerate() {
                let product = self.polynomial_multiply(&row.polys[j], &vector.polys[i])?;
                sum = self.polynomial_add(&sum, &product)?;
            }
            
            result.polys.push(sum);
        }
        
        Ok(result)
    }
    
    /// Vector addition
    fn vector_add(&self, a: &KyberVector, b: &KyberVector) -> Result<KyberVector, QuantumSecurityError> {
        if a.polys.len() != b.polys.len() {
            return Err(QuantumSecurityError::InvalidParameters(
                "Vector dimensions mismatch".to_string()
            ));
        }
        
        let mut result = KyberVector {
            polys: Vec::with_capacity(a.polys.len()),
        };
        
        for (poly_a, poly_b) in a.polys.iter().zip(b.polys.iter()) {
            result.polys.push(self.polynomial_add(poly_a, poly_b)?);
        }
        
        Ok(result)
    }
    
    /// Vector dot product
    fn vector_dot_product(&self, a: &KyberVector, b: &KyberVector) -> Result<KyberPolynomial, QuantumSecurityError> {
        if a.polys.len() != b.polys.len() {
            return Err(QuantumSecurityError::InvalidParameters(
                "Vector dimensions mismatch".to_string()
            ));
        }
        
        let mut result = KyberPolynomial { coeffs: [0u16; 256] };
        
        for (poly_a, poly_b) in a.polys.iter().zip(b.polys.iter()) {
            let product = self.polynomial_multiply(poly_a, poly_b)?;
            result = self.polynomial_add(&result, &product)?;
        }
        
        Ok(result)
    }
    
    /// Polynomial addition
    fn polynomial_add(&self, a: &KyberPolynomial, b: &KyberPolynomial) -> Result<KyberPolynomial, QuantumSecurityError> {
        let mut result = KyberPolynomial { coeffs: [0u16; 256] };
        
        for i in 0..256 {
            result.coeffs[i] = (a.coeffs[i] + b.coeffs[i]) % 3329;
        }
        
        Ok(result)
    }
    
    /// Polynomial subtraction
    fn polynomial_subtract(&self, a: &KyberPolynomial, b: &KyberPolynomial) -> Result<KyberPolynomial, QuantumSecurityError> {
        let mut result = KyberPolynomial { coeffs: [0u16; 256] };
        
        for i in 0..256 {
            result.coeffs[i] = (a.coeffs[i] + 3329 - b.coeffs[i]) % 3329;
        }
        
        Ok(result)
    }
    
    /// Polynomial multiplication (simplified NTT)
    fn polynomial_multiply(&self, a: &KyberPolynomial, b: &KyberPolynomial) -> Result<KyberPolynomial, QuantumSecurityError> {
        let mut result = KyberPolynomial { coeffs: [0u16; 256] };
        
        // Simplified multiplication - in practice, use NTT
        for i in 0..256 {
            for j in 0..256 {
                let idx = (i + j) % 256;
                let sign = if i + j >= 256 { 3329 - 1 } else { 1 };
                result.coeffs[idx] = (result.coeffs[idx] + (a.coeffs[i] as u32 * b.coeffs[j] as u32 * sign as u32) as u16) % 3329;
            }
        }
        
        Ok(result)
    }
    
    /// Encode vector to bytes
    fn encode_vector(&self, vector: &KyberVector, params: &KyberParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();
        
        for poly in &vector.polys {
            for coeff in poly.coeffs.iter() {
                bytes.extend_from_slice(&coeff.to_le_bytes());
            }
        }
        
        Ok(bytes)
    }
    
    /// Decode vector from bytes
    fn decode_vector(&self, bytes: &[u8], params: &KyberParams) -> Result<KyberVector, QuantumSecurityError> {
        let mut vector = KyberVector {
            polys: Vec::with_capacity(params.k),
        };
        
        for k in 0..params.k {
            let mut poly = KyberPolynomial { coeffs: [0u16; 256] };
            
            for i in 0..256 {
                let byte_idx = (k * 256 + i) * 2;
                if byte_idx + 1 < bytes.len() {
                    poly.coeffs[i] = u16::from_le_bytes([bytes[byte_idx], bytes[byte_idx + 1]]);
                }
            }
            
            vector.polys.push(poly);
        }
        
        Ok(vector)
    }
    
    /// Compress vector
    fn compress_vector(&self, vector: &KyberVector, d: u8) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();
        
        for poly in &vector.polys {
            for coeff in poly.coeffs.iter() {
                let compressed = (((*coeff as u32) << d) + 1664) / 3329;
                bytes.push(compressed as u8);
            }
        }
        
        Ok(bytes)
    }
    
    /// Decompress vector
    fn decompress_vector(&self, bytes: &[u8], d: u8) -> Result<KyberVector, QuantumSecurityError> {
        let polys_count = bytes.len() / 256;
        let mut vector = KyberVector {
            polys: Vec::with_capacity(polys_count),
        };
        
        for k in 0..polys_count {
            let mut poly = KyberPolynomial { coeffs: [0u16; 256] };
            
            for i in 0..256 {
                let byte_idx = k * 256 + i;
                if byte_idx < bytes.len() {
                    let compressed = bytes[byte_idx] as u32;
                    poly.coeffs[i] = ((compressed * 3329 + (1 << (d - 1))) >> d) as u16;
                }
            }
            
            vector.polys.push(poly);
        }
        
        Ok(vector)
    }
    
    /// Compress polynomial
    fn compress_polynomial(&self, poly: &KyberPolynomial, d: u8) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut bytes = Vec::new();
        
        for coeff in poly.coeffs.iter() {
            let compressed = (((*coeff as u32) << d) + 1664) / 3329;
            bytes.push(compressed as u8);
        }
        
        Ok(bytes)
    }
    
    /// Decompress polynomial
    fn decompress_polynomial(&self, bytes: &[u8], d: u8) -> Result<KyberPolynomial, QuantumSecurityError> {
        let mut poly = KyberPolynomial { coeffs: [0u16; 256] };
        
        for i in 0..256.min(bytes.len()) {
            let compressed = bytes[i] as u32;
            poly.coeffs[i] = ((compressed * 3329 + (1 << (d - 1))) >> d) as u16;
        }
        
        Ok(poly)
    }
    
    /// Decode message
    fn decode_message(&self, message: &[u8], params: &KyberParams) -> Result<KyberPolynomial, QuantumSecurityError> {
        let mut poly = KyberPolynomial { coeffs: [0u16; 256] };
        
        for i in 0..32.min(message.len()) {
            let byte = message[i];
            for j in 0..8 {
                let bit = (byte >> j) & 1;
                if i * 8 + j < 256 {
                    poly.coeffs[i * 8 + j] = if bit == 1 { 1664 } else { 0 };
                }
            }
        }
        
        Ok(poly)
    }
    
    /// Encode message
    fn encode_message(&self, poly: &KyberPolynomial, params: &KyberParams) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut message = vec![0u8; 32];
        
        for i in 0..32 {
            let mut byte = 0u8;
            for j in 0..8 {
                let coeff_idx = i * 8 + j;
                if coeff_idx < 256 {
                    let bit = if poly.coeffs[coeff_idx] > 1664 { 1 } else { 0 };
                    byte |= bit << j;
                }
            }
            message[i] = byte;
        }
        
        Ok(message)
    }
    
    /// Generate coins for CCA security
    fn generate_coins(&self, message: &[u8], public_key: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(message);
        hasher.update(public_key);
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes().to_vec())
    }
    
    /// Hash public key
    fn hash_public_key(&self, public_key: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(public_key);
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes().to_vec())
    }
    
    /// Generate random z for CCA security
    fn generate_random_z(&self) -> Result<Vec<u8>, QuantumSecurityError> {
        let mut z = vec![0u8; 32];
        rand::RngCore::fill_bytes(&mut rand::thread_rng(), &mut z);
        Ok(z)
    }
    
    /// Key derivation function
    fn kdf(&self, ciphertext: &[u8], key: &[u8]) -> Result<Vec<u8>, QuantumSecurityError> {
        use blake3::Hasher;
        
        let mut hasher = Hasher::new();
        hasher.update(ciphertext);
        hasher.update(key);
        let hash = hasher.finalize();
        
        Ok(hash.as_bytes()[..32].to_vec())
    }
    
    /// Extract Kyber public key from PQC key
    fn extract_kyber_public_key(&self, key: &PQCKey) -> Result<KyberPublicKey, QuantumSecurityError> {
        match key {
            PQCKey::KyberPublicKey { key_data, algorithm, .. } => {
                if *algorithm != self.algorithm {
                    return Err(QuantumSecurityError::InvalidAlgorithm(
                        format!("Expected {:?}, got {:?}", self.algorithm, algorithm)
                    ));
                }
                KyberPublicKey::from_bytes(key_data)
            }
            _ => Err(QuantumSecurityError::InvalidKeyType("Expected Kyber public key".to_string()))
        }
    }
    
    /// Extract Kyber private key from PQC key
    fn extract_kyber_private_key(&self, key: &PQCKey) -> Result<KyberPrivateKey, QuantumSecurityError> {
        match key {
            PQCKey::KyberPrivateKey { key_data, algorithm, .. } => {
                if *algorithm != self.algorithm {
                    return Err(QuantumSecurityError::InvalidAlgorithm(
                        format!("Expected {:?}, got {:?}", self.algorithm, algorithm)
                    ));
                }
                KyberPrivateKey::from_bytes(key_data.expose())
            }
            _ => Err(QuantumSecurityError::InvalidKeyType("Expected Kyber private key".to_string()))
        }
    }
    
    /// Update key generation metrics
    async fn update_keygen_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.key_generation_count += 1;
        metrics.key_generation_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }
    
    /// Update encapsulation metrics
    async fn update_encaps_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.encapsulation_count += 1;
        metrics.encapsulation_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }
    
    /// Update decapsulation metrics
    async fn update_decaps_metrics(&self, elapsed_us: u64) {
        let mut metrics = self.metrics.write().await;
        metrics.decapsulation_count += 1;
        metrics.decapsulation_time_us += elapsed_us;
        metrics.last_operation = Some(chrono::Utc::now());
    }
    
    /// Get metrics
    pub async fn get_metrics(&self) -> AlgorithmMetrics {
        self.metrics.read().await.clone()
    }
}

impl KyberPublicKey {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.t);
        bytes.extend_from_slice(&self.rho);
        bytes
    }
    
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 64 {
            return Err(QuantumSecurityError::InvalidData("Invalid public key length".to_string()));
        }
        
        let t_len = bytes.len() - 32;
        let t = bytes[..t_len].to_vec();
        let rho = bytes[t_len..].to_vec();
        
        Ok(Self {
            t,
            rho,
            algorithm: PQCAlgorithm::Kyber1024, // Default, should be set properly
        })
    }
}

impl KyberPrivateKey {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.s);
        bytes.extend_from_slice(&self.pk);
        bytes.extend_from_slice(&self.h);
        bytes.extend_from_slice(&self.z);
        bytes
    }
    
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 128 {
            return Err(QuantumSecurityError::InvalidData("Invalid private key length".to_string()));
        }
        
        // This is a simplified parsing - in practice, need proper length calculations
        let s_len = bytes.len() / 4;
        let s = bytes[..s_len].to_vec();
        let pk = bytes[s_len..2*s_len].to_vec();
        let h = bytes[2*s_len..3*s_len].to_vec();
        let z = bytes[3*s_len..].to_vec();
        
        Ok(Self {
            s,
            pk,
            h,
            z,
            algorithm: PQCAlgorithm::Kyber1024, // Default, should be set properly
        })
    }
}

impl KyberCiphertext {
    /// Convert to bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&self.c1);
        bytes.extend_from_slice(&self.c2);
        bytes
    }
    
    /// Create from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, QuantumSecurityError> {
        if bytes.len() < 64 {
            return Err(QuantumSecurityError::InvalidData("Invalid ciphertext length".to_string()));
        }
        
        let c1_len = bytes.len() / 2;
        let c1 = bytes[..c1_len].to_vec();
        let c2 = bytes[c1_len..].to_vec();
        
        Ok(Self {
            c1,
            c2,
            algorithm: PQCAlgorithm::Kyber1024, // Default, should be set properly
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::thread_rng;
    
    #[tokio::test]
    async fn test_kyber_keygen() {
        let kyber = CrystalsKyber::new(PQCAlgorithm::Kyber1024).unwrap();
        let mut rng = thread_rng();
        
        let keypair = kyber.generate_keypair(&mut rng).await.unwrap();
        assert_eq!(keypair.algorithm, PQCAlgorithm::Kyber1024);
        assert_eq!(keypair.usage, KeyUsage::KeyEncapsulation);
    }
    
    #[tokio::test]
    async fn test_kyber_encaps_decaps() {
        let kyber = CrystalsKyber::new(PQCAlgorithm::Kyber1024).unwrap();
        let mut rng = thread_rng();
        
        let keypair = kyber.generate_keypair(&mut rng).await.unwrap();
        let encapsulated = kyber.encapsulate(&keypair.public_key, &mut rng).await.unwrap();
        let decapsulated = kyber.decapsulate(&keypair.private_key, &encapsulated).await.unwrap();
        
        assert_eq!(encapsulated.shared_secret.expose(), decapsulated.expose());
    }
    
    #[tokio::test]
    async fn test_kyber_metrics() {
        let kyber = CrystalsKyber::new(PQCAlgorithm::Kyber1024).unwrap();
        let mut rng = thread_rng();
        
        let _ = kyber.generate_keypair(&mut rng).await.unwrap();
        
        let metrics = kyber.get_metrics().await;
        assert_eq!(metrics.key_generation_count, 1);
        assert!(metrics.key_generation_time_us > 0);
    }
}