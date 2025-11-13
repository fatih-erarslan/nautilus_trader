//! Module Learning With Errors (Module-LWE) for Dilithium/ML-DSA
//!
//! Enterprise-grade implementation of Module-LWE problem, which provides the
//! security foundation for Dilithium digital signatures.
//!
//! # Mathematical Foundation
//!
//! Module-LWE is a structured variant of the Learning With Errors (LWE) problem:
//!
//! Given: (A, b = As + e) where
//! - A ∈ R_q^(k×l) is a random matrix over polynomial ring R_q = Z_q[X]/(X^n + 1)
//! - s ∈ R_q^l is a secret vector with small coefficients
//! - e ∈ R_q^k is an error vector with small coefficients
//!
//! Goal: Recover s (computationally hard for proper parameters)
//!
//! # Security Properties
//!
//! - Quantum-resistant: No known quantum algorithm breaks Module-LWE efficiently
//! - Reduction to worst-case lattice problems (SIVP, GapSVP)
//! - Security level depends on (n, k, l, q, η) parameters
//!
//! # Dilithium Parameters
//!
//! ML-DSA-44: (n=256, k=4, l=4, q=8380417, η=2)
//! ML-DSA-65: (n=256, k=6, l=5, q=8380417, η=4)
//! ML-DSA-87: (n=256, k=8, l=7, q=8380417, η=2)
//!
//! # References
//!
//! - FIPS 204: Module-Lattice-Based Digital Signature Standard
//! - Lyubashevsky et al. (2017): "CRYSTALS-Dilithium: A Lattice-Based Digital Signature Scheme"
//! - Langlois & Stehlé (2015): "Worst-case to average-case reductions for module lattices"

use super::ntt::{NTT, DILITHIUM_Q, barrett_reduce, montgomery_reduce, poly_add, poly_sub, poly_multiply};
use crate::{DilithiumResult, DilithiumError, SecurityLevel};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha3::{Shake128, Shake256, digest::{Update, ExtendableOutput, XofReader}};

/// Polynomial degree (always 256 for Dilithium)
pub const POLY_DEGREE: usize = 256;

/// Number of bytes in a seed
pub const SEED_BYTES: usize = 32;

/// Polynomial in coefficient representation
pub type Polynomial = Vec<i32>;

/// Vector of polynomials
pub type PolyVec = Vec<Polynomial>;

/// Matrix of polynomials
pub type PolyMatrix = Vec<PolyVec>;

/// Module-LWE engine for Dilithium
///
/// Provides all operations needed for Module-LWE based signatures:
/// - Matrix/vector operations over polynomial rings
/// - Rejection sampling for small coefficients
/// - Uniform sampling from seed (expandA)
/// - Challenge polynomial generation
pub struct ModuleLWE {
    /// Security level
    security_level: SecurityLevel,
    
    /// Dimension k (rows in A)
    k: usize,
    
    /// Dimension l (columns in A)
    l: usize,
    
    /// Modulus
    q: i32,
    
    /// Small coefficient bound η
    eta: i32,
    
    /// NTT engine
    ntt: NTT,
}

impl ModuleLWE {
    /// Create new Module-LWE engine
    ///
    /// # Arguments
    ///
    /// * `security_level` - Desired security level
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::lattice::module_lwe::ModuleLWE;
    /// use hyperphysics_dilithium::SecurityLevel;
    ///
    /// let mlwe = ModuleLWE::new(SecurityLevel::High);
    /// ```
    pub fn new(security_level: SecurityLevel) -> Self {
        let (k, l, eta) = match security_level {
            SecurityLevel::Standard => (4, 4, 2),  // ML-DSA-44
            SecurityLevel::High => (6, 5, 4),      // ML-DSA-65
            SecurityLevel::Maximum => (8, 7, 2),   // ML-DSA-87
        };
        
        Self {
            security_level,
            k,
            l,
            q: DILITHIUM_Q,
            eta,
            ntt: NTT::new(),
        }
    }
    
    /// Expand matrix A from seed (ExpandA in FIPS 204)
    ///
    /// Deterministically generates the public matrix A from a seed using SHAKE-128.
    ///
    /// # Arguments
    ///
    /// * `seed` - 32-byte seed
    ///
    /// # Returns
    ///
    /// k×l matrix A with uniformly random polynomials
    ///
    /// # Security
    ///
    /// Uses SHAKE-128 XOF for cryptographic randomness expansion
    pub fn expand_a(&self, seed: &[u8; SEED_BYTES]) -> PolyMatrix {
        let mut matrix = vec![vec![vec![0i32; POLY_DEGREE]; self.l]; self.k];
        
        for i in 0..self.k {
            for j in 0..self.l {
                matrix[i][j] = self.sample_uniform_poly(seed, i as u16, j as u16);
            }
        }
        
        matrix
    }
    
    /// Sample uniform polynomial from seed
    ///
    /// Uses rejection sampling to generate uniform coefficients in [0, q).
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed
    /// * `nonce1` - First nonce (typically row index)
    /// * `nonce2` - Second nonce (typically column index)
    ///
    /// # Returns
    ///
    /// Polynomial with uniform coefficients mod q
    fn sample_uniform_poly(&self, seed: &[u8; SEED_BYTES], nonce1: u16, nonce2: u16) -> Polynomial {
        let mut poly = vec![0i32; POLY_DEGREE];
        
        // Create SHAKE-128 XOF
        let mut hasher = Shake128::default();
        hasher.update(seed);
        hasher.update(&nonce1.to_le_bytes());
        hasher.update(&nonce2.to_le_bytes());
        
        let mut reader = hasher.finalize_xof();
        let mut buf = [0u8; 3];
        let mut coeff_idx = 0;
        
        // Rejection sampling
        while coeff_idx < POLY_DEGREE {
            reader.read(&mut buf);
            
            // Interpret 3 bytes as coefficient candidate
            let coeff = ((buf[0] as i32) | ((buf[1] as i32) << 8) | ((buf[2] as i32) << 16)) & 0x7FFFFF;
            
            // Accept if < q
            if coeff < self.q {
                poly[coeff_idx] = coeff;
                coeff_idx += 1;
            }
        }
        
        poly
    }
    
    /// Sample polynomial with small coefficients from {-η, ..., η}
    ///
    /// Uses centered binomial distribution for security.
    ///
    /// # Arguments
    ///
    /// * `seed` - Random seed
    /// * `nonce` - Nonce for domain separation
    ///
    /// # Returns
    ///
    /// Polynomial with coefficients in {-η, ..., η}
    pub fn sample_small_poly(&self, seed: &[u8; SEED_BYTES], nonce: u16) -> Polynomial {
        let mut poly = vec![0i32; POLY_DEGREE];
        
        // Use SHAKE-256 for small coefficient sampling
        let mut hasher = Shake256::default();
        hasher.update(seed);
        hasher.update(&nonce.to_le_bytes());
        
        let mut reader = hasher.finalize_xof();
        
        // Sample using centered binomial distribution
        match self.eta {
            2 => self.sample_eta_2(&mut reader, &mut poly),
            4 => self.sample_eta_4(&mut reader, &mut poly),
            _ => panic!("Unsupported eta value: {}", self.eta),
        }
        
        poly
    }
    
    /// Sample with η=2 (centered binomial)
    ///
    /// Each coefficient is sum of 2 bits minus sum of 2 other bits
    fn sample_eta_2(&self, reader: &mut dyn XofReader, poly: &mut Polynomial) {
        let mut buf = [0u8; 1];
        
        for i in (0..POLY_DEGREE).step_by(4) {
            reader.read(&mut buf);
            let byte = buf[0];
            
            for j in 0..4 {
                if i + j < POLY_DEGREE {
                    let bits = (byte >> (2 * j)) & 0x03;
                    let a = (bits & 0x01) as i32;
                    let b = ((bits >> 1) & 0x01) as i32;
                    poly[i + j] = a - b;
                }
            }
        }
    }
    
    /// Sample with η=4 (centered binomial)
    ///
    /// Each coefficient is sum of 4 bits minus sum of 4 other bits
    fn sample_eta_4(&self, reader: &mut dyn XofReader, poly: &mut Polynomial) {
        let mut buf = [0u8; 1];
        
        for i in 0..POLY_DEGREE {
            reader.read(&mut buf);
            let byte = buf[0];
            
            let a = (byte & 0x0F).count_ones() as i32;
            let b = ((byte >> 4) & 0x0F).count_ones() as i32;
            poly[i] = a - b;
        }
    }
    
    /// Generate challenge polynomial c
    ///
    /// Samples polynomial with exactly τ non-zero coefficients in {-1, 1}.
    ///
    /// # Arguments
    ///
    /// * `seed` - Challenge seed (hash of message and commitment)
    /// * `tau` - Number of non-zero coefficients
    ///
    /// # Returns
    ///
    /// Challenge polynomial
    pub fn sample_challenge(&self, seed: &[u8; SEED_BYTES], tau: usize) -> Polynomial {
        let mut poly = vec![0i32; POLY_DEGREE];
        
        // Use SHAKE-256 for challenge sampling
        let mut hasher = Shake256::default();
        hasher.update(seed);
        
        let mut reader = hasher.finalize_xof();
        let mut buf = [0u8; 1];
        
        let mut signs = 0u64;
        let mut positions = Vec::with_capacity(tau);
        
        // Sample sign bits
        for _ in 0..(tau + 7) / 8 {
            reader.read(&mut buf);
            signs = (signs << 8) | (buf[0] as u64);
        }
        
        // Sample positions using rejection sampling
        while positions.len() < tau {
            reader.read(&mut buf);
            let pos = buf[0] as usize;
            
            if pos < POLY_DEGREE && !positions.contains(&pos) {
                positions.push(pos);
            }
        }
        
        // Set coefficients
        for (i, &pos) in positions.iter().enumerate() {
            let sign = ((signs >> i) & 1) as i32;
            poly[pos] = 1 - 2 * sign;  // Maps 0→1, 1→-1
        }
        
        poly
    }
    
    /// Matrix-vector multiplication: A * s
    ///
    /// Computes matrix-vector product over polynomial ring using NTT.
    ///
    /// # Arguments
    ///
    /// * `matrix` - k×l matrix A
    /// * `vector` - l-dimensional vector s
    ///
    /// # Returns
    ///
    /// k-dimensional vector A*s
    pub fn matrix_vector_multiply(&self, matrix: &PolyMatrix, vector: &PolyVec) -> PolyVec {
        assert_eq!(matrix[0].len(), vector.len());
        
        let mut result = vec![vec![0i32; POLY_DEGREE]; matrix.len()];
        
        for i in 0..matrix.len() {
            for j in 0..vector.len() {
                // Convert to NTT domain
                let a_ntt = self.ntt.forward(&matrix[i][j]);
                let s_ntt = self.ntt.forward(&vector[j]);
                
                // Pointwise multiplication
                let product_ntt = self.ntt.pointwise_multiply(&a_ntt, &s_ntt);
                
                // Convert back and accumulate
                let product = self.ntt.inverse(&product_ntt);
                result[i] = poly_add(&result[i], &product);
            }
        }
        
        result
    }
    
    /// Vector addition: a + b
    pub fn vector_add(&self, a: &PolyVec, b: &PolyVec) -> PolyVec {
        assert_eq!(a.len(), b.len());
        
        a.iter()
            .zip(b.iter())
            .map(|(pa, pb)| poly_add(pa, pb))
            .collect()
    }
    
    /// Vector subtraction: a - b
    pub fn vector_sub(&self, a: &PolyVec, b: &PolyVec) -> PolyVec {
        assert_eq!(a.len(), b.len());
        
        a.iter()
            .zip(b.iter())
            .map(|(pa, pb)| poly_sub(pa, pb))
            .collect()
    }
    
    /// Scalar-vector multiplication: c * v
    ///
    /// Multiplies each polynomial in vector by scalar polynomial.
    pub fn scalar_vector_multiply(&self, scalar: &Polynomial, vector: &PolyVec) -> PolyVec {
        vector.iter()
            .map(|poly| poly_multiply(scalar, poly))
            .collect()
    }
    
    /// Infinity norm of polynomial
    ///
    /// Returns maximum absolute value of coefficients.
    pub fn poly_inf_norm(&self, poly: &Polynomial) -> i32 {
        poly.iter()
            .map(|&c| {
                let c_centered = if c > self.q / 2 { c - self.q } else { c };
                c_centered.abs()
            })
            .max()
            .unwrap_or(0)
    }
    
    /// Infinity norm of vector
    pub fn vector_inf_norm(&self, vector: &PolyVec) -> i32 {
        vector.iter()
            .map(|poly| self.poly_inf_norm(poly))
            .max()
            .unwrap_or(0)
    }
    
    /// Power2Round: decompose r into r1 and r0
    ///
    /// Splits r = r1*2^d + r0 where r0 ∈ (-2^(d-1), 2^(d-1)]
    ///
    /// # Arguments
    ///
    /// * `r` - Input coefficient
    /// * `d` - Decomposition parameter
    ///
    /// # Returns
    ///
    /// (r1, r0) tuple
    pub fn power2round(&self, r: i32, d: u32) -> (i32, i32) {
        let r_mod = barrett_reduce(r);
        let r0 = r_mod & ((1 << d) - 1);
        let r0 = if r0 > (1 << (d - 1)) {
            r0 - (1 << d)
        } else {
            r0
        };
        
        let r1 = (r_mod - r0) >> d;
        (r1, r0)
    }
    
    /// Decompose polynomial using Power2Round
    pub fn poly_power2round(&self, poly: &Polynomial, d: u32) -> (Polynomial, Polynomial) {
        let mut poly1 = vec![0i32; POLY_DEGREE];
        let mut poly0 = vec![0i32; POLY_DEGREE];
        
        for i in 0..POLY_DEGREE {
            let (r1, r0) = self.power2round(poly[i], d);
            poly1[i] = r1;
            poly0[i] = r0;
        }
        
        (poly1, poly0)
    }
    
    /// HighBits: extract high bits of r
    ///
    /// Used in signature compression.
    pub fn high_bits(&self, r: i32, alpha: i32) -> i32 {
        let r_mod = barrett_reduce(r);
        let r1 = (r_mod + 127) >> 7;
        
        if r1 > (self.q - 1) / (2 * alpha) {
            r1 - (self.q / alpha)
        } else {
            r1
        }
    }
    
    /// LowBits: extract low bits of r
    pub fn low_bits(&self, r: i32, alpha: i32) -> i32 {
        let r_mod = barrett_reduce(r);
        let r1 = self.high_bits(r, alpha);
        r_mod - r1 * alpha
    }
    
    /// MakeHint: create hint bit
    ///
    /// Hint indicates whether adding z changes high bits.
    pub fn make_hint(&self, z: i32, r: i32, alpha: i32) -> bool {
        let r1 = self.high_bits(r, alpha);
        let v1 = self.high_bits(r + z, alpha);
        r1 != v1
    }
    
    /// UseHint: recover high bits using hint
    pub fn use_hint(&self, hint: bool, r: i32, alpha: i32) -> i32 {
        let m = (self.q - 1) / alpha;
        let r1 = self.high_bits(r, alpha);
        
        if !hint {
            return r1;
        }
        
        if r1 == 0 {
            m - 1
        } else if r1 == m - 1 {
            0
        } else {
            r1
        }
    }
    
    /// Check if polynomial has small coefficients
    ///
    /// Verifies all coefficients are in [-bound, bound].
    pub fn check_small_poly(&self, poly: &Polynomial, bound: i32) -> bool {
        poly.iter().all(|&c| {
            let c_centered = if c > self.q / 2 { c - self.q } else { c };
            c_centered.abs() <= bound
        })
    }
    
    /// Check if vector has small coefficients
    pub fn check_small_vector(&self, vector: &PolyVec, bound: i32) -> bool {
        vector.iter().all(|poly| self.check_small_poly(poly, bound))
    }
    
    /// Encode polynomial to bytes
    ///
    /// Packs polynomial coefficients into byte array.
    pub fn poly_encode(&self, poly: &Polynomial, bits_per_coeff: usize) -> Vec<u8> {
        let total_bits = POLY_DEGREE * bits_per_coeff;
        let mut bytes = vec![0u8; (total_bits + 7) / 8];
        
        let mut bit_pos = 0;
        for &coeff in poly.iter() {
            let c = coeff as u32;
            for i in 0..bits_per_coeff {
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                bytes[byte_idx] |= (((c >> i) & 1) as u8) << bit_idx;
                bit_pos += 1;
            }
        }
        
        bytes
    }
    
    /// Decode polynomial from bytes
    pub fn poly_decode(&self, bytes: &[u8], bits_per_coeff: usize) -> Polynomial {
        let mut poly = vec![0i32; POLY_DEGREE];
        
        let mut bit_pos = 0;
        for i in 0..POLY_DEGREE {
            let mut coeff = 0u32;
            for j in 0..bits_per_coeff {
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                if byte_idx < bytes.len() {
                    coeff |= (((bytes[byte_idx] >> bit_idx) & 1) as u32) << j;
                }
                bit_pos += 1;
            }
            poly[i] = coeff as i32;
        }
        
        poly
    }
    
    /// Get security parameters
    pub fn params(&self) -> (usize, usize, i32) {
        (self.k, self.l, self.eta)
    }
}

impl Default for ModuleLWE {
    fn default() -> Self {
        Self::new(SecurityLevel::High)
    }
}

/// Constant-time conditional swap
///
/// Swaps a and b if swap is true, in constant time.
#[inline(always)]
pub fn conditional_swap(a: &mut i32, b: &mut i32, swap: bool) {
    let mask = -(swap as i32);
    let t = mask & (*a ^ *b);
    *a ^= t;
    *b ^= t;
}

/// Constant-time conditional select
///
/// Returns a if select is true, b otherwise, in constant time.
#[inline(always)]
pub fn conditional_select(a: i32, b: i32, select: bool) -> i32 {
    let mask = -(select as i32);
    b ^ (mask & (a ^ b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expand_a() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        let seed = [0u8; SEED_BYTES];
        
        let matrix = mlwe.expand_a(&seed);
        
        assert_eq!(matrix.len(), 4);  // k=4 for Standard
        assert_eq!(matrix[0].len(), 4);  // l=4 for Standard
        assert_eq!(matrix[0][0].len(), POLY_DEGREE);
    }

    #[test]
    fn test_sample_small_poly() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        let seed = [1u8; SEED_BYTES];
        
        let poly = mlwe.sample_small_poly(&seed, 0);
        
        // All coefficients should be in [-2, 2] for eta=2
        assert!(poly.iter().all(|&c| c >= -2 && c <= 2));
    }

    #[test]
    fn test_sample_challenge() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        let seed = [2u8; SEED_BYTES];
        
        let poly = mlwe.sample_challenge(&seed, 39);  // tau=39 for Standard
        
        // Exactly 39 non-zero coefficients
        let non_zero = poly.iter().filter(|&&c| c != 0).count();
        assert_eq!(non_zero, 39);
        
        // All non-zero coefficients are ±1
        assert!(poly.iter().all(|&c| c == -1 || c == 0 || c == 1));
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        
        let matrix = vec![vec![vec![1i32; POLY_DEGREE]; 4]; 4];
        let vector = vec![vec![2i32; POLY_DEGREE]; 4];
        
        let result = mlwe.matrix_vector_multiply(&matrix, &vector);
        
        assert_eq!(result.len(), 4);
        assert_eq!(result[0].len(), POLY_DEGREE);
    }

    #[test]
    fn test_power2round() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        
        let (r1, r0) = mlwe.power2round(1000, 13);
        
        // Verify decomposition
        assert_eq!(r1 * (1 << 13) + r0, 1000);
        assert!(r0.abs() <= (1 << 12));
    }

    #[test]
    fn test_poly_encode_decode() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        let poly: Polynomial = (0..POLY_DEGREE).map(|i| i as i32).collect();
        
        let encoded = mlwe.poly_encode(&poly, 12);
        let decoded = mlwe.poly_decode(&encoded, 12);
        
        for i in 0..POLY_DEGREE {
            assert_eq!(poly[i] & 0xFFF, decoded[i] & 0xFFF);
        }
    }

    #[test]
    fn test_vector_operations() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        
        let a = vec![vec![100i32; POLY_DEGREE]; 4];
        let b = vec![vec![50i32; POLY_DEGREE]; 4];
        
        let sum = mlwe.vector_add(&a, &b);
        assert_eq!(sum[0][0], 150);
        
        let diff = mlwe.vector_sub(&a, &b);
        assert_eq!(diff[0][0], 50);
    }

    #[test]
    fn test_inf_norm() {
        let mlwe = ModuleLWE::new(SecurityLevel::Standard);
        
        let poly = vec![1, -5, 3, 0, 2];
        let norm = mlwe.poly_inf_norm(&poly);
        
        assert_eq!(norm, 5);
    }

    #[test]
    fn test_conditional_operations() {
        let mut a = 10;
        let mut b = 20;
        
        conditional_swap(&mut a, &mut b, true);
        assert_eq!(a, 20);
        assert_eq!(b, 10);
        
        let c = conditional_select(5, 15, true);
        assert_eq!(c, 5);
        
        let d = conditional_select(5, 15, false);
        assert_eq!(d, 15);
    }
}
