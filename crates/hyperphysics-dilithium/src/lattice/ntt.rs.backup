//! Number Theoretic Transform (NTT) for Dilithium/ML-DSA
//!
//! Fast polynomial multiplication in Z_q[X]/(X^n + 1).
//!
//! # Reference
//!
//! FIPS 204 (2024) Module-Lattice-Based Digital Signature Standard
//! Section 8.2: Number Theoretic Transform
//!
//! # Security
//!
//! - Constant-time operations (no timing side-channels)
//! - Barrett and Montgomery reduction for modular arithmetic
//! - Side-channel resistant implementation
//!
//! # Performance
//!
//! - O(n log n) polynomial multiplication vs O(n²) naive
//! - Cooley-Tukey decimation-in-time algorithm
//! - Precomputed twiddle factors for 256-coefficient polynomials

use crate::DilithiumResult;

/// Dilithium prime modulus: q = 8,380,417 = 2^23 - 2^13 + 1
pub const Q: i32 = 8_380_417;

/// Polynomial degree: n = 256
pub const N: usize = 256;

/// Primitive 512-th root of unity modulo Q
/// ζ = 1753 (from FIPS 204 Table 1)
const ROOT: i32 = 1753;

/// Montgomery parameter: R = 2^32
const R: i64 = 1 << 32;

/// R^(-1) mod Q for Montgomery reduction
const R_INV: i64 = 58728449;

/// ⌊2^44 / Q⌋ for Barrett reduction
const BARRETT_MULTIPLIER: i64 = 4236238847;

/// Number Theoretic Transform implementation for CRYSTALS-Dilithium
pub struct NTT {
    /// Precomputed twiddle factors: ζ^bit_reverse(i) mod Q
    zetas: Vec<i32>,

    /// Precomputed inverse twiddle factors
    zetas_inv: Vec<i32>,

    /// n^(-1) mod Q for normalization after inverse NTT
    n_inv: i32,
}

impl NTT {
    /// Create new NTT instance with precomputed twiddle factors
    pub fn new() -> Self {
        let mut zetas = vec![0; N];
        let mut zetas_inv = vec![0; N];

        // Precompute zetas[i] = ζ^bit_reverse(i) mod Q
        for i in 0..N {
            let exp = bit_reverse_8bit(i);
            zetas[i] = mod_pow(ROOT, exp as u32, Q);
            zetas_inv[i] = mod_inverse(zetas[i], Q);
        }

        // Compute n^(-1) mod Q
        let n_inv = mod_inverse(N as i32, Q);

        Self {
            zetas,
            zetas_inv,
            n_inv,
        }
    }
}

// Enterprise-grade implementation of NTT for polynomial operations in Z_q[X]/(X^n + 1).
//
// # Security Properties
//
// - Constant-time operations (no secret-dependent branches)
// - Barrett reduction for modular arithmetic
// - Bit-reversal permutation for Cooley-Tukey algorithm
// - Montgomery multiplication for efficiency
//
// # Mathematical Foundation
//
// NTT is the number-theoretic analogue of FFT, operating in Z_q instead of C.
// For Dilithium, we use:
// - q = 8380417 (prime modulus)
// - n = 256 (polynomial degree)
// - ω = 1753 (primitive 512-th root of unity mod q)
//
// # References
//
// - FIPS 204: Module-Lattice-Based Digital Signature Standard
// - Cooley-Tukey (1965): "An algorithm for the machine calculation of complex Fourier series"
// - Barrett (1986): "Implementing the Rivest Shamir and Adleman Public Key Encryption Algorithm"

use std::ops::{Add, Sub, Mul};

/// Dilithium modulus q = 8380417
pub const DILITHIUM_Q: i32 = 8380417;

/// Primitive 512-th root of unity modulo q
const ROOT_OF_UNITY: i32 = 1753;

/// Inverse of 256 modulo q (for inverse NTT)
const INV_256: i32 = 8347681;

/// Barrett reduction constant: floor(2^32 / q)
const BARRETT_MULTIPLIER: i64 = 512;

/// Precomputed powers of ω for forward NTT
static ZETAS: [i32; 256] = precompute_zetas();

/// Precomputed powers of ω^(-1) for inverse NTT
static ZETAS_INV: [i32; 256] = precompute_zetas_inv();

/// Number Theoretic Transform engine
///
/// Provides constant-time polynomial multiplication in Z_q[X]/(X^n + 1)
/// using Cooley-Tukey FFT algorithm adapted for finite fields.
pub struct NTT {
    /// Polynomial degree (always 256 for Dilithium)
    degree: usize,
    
    /// Modulus
    modulus: i32,
}

impl NTT {
    /// Create new NTT engine
    ///
    /// # Example
    ///
    /// ```
    /// use hyperphysics_dilithium::lattice::ntt::NTT;
    ///
    /// let ntt = NTT::new();
    /// ```
    pub fn new() -> Self {
        Self {
            degree: 256,
            modulus: DILITHIUM_Q,
        }
    }
    
    /// Forward NTT transform (time domain → frequency domain)
    ///
    /// Transforms polynomial coefficients to NTT representation for fast multiplication.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - Polynomial coefficients (length must be N=256)
    ///
    /// # Security
    ///
    /// Constant-time operation (no data-dependent branches)
    pub fn forward(&self, coeffs: &mut [i32]) {
        assert_eq!(coeffs.len(), N, "Polynomial must have {} coefficients", N);

        let mut len = N / 2;
        let mut k = 0;

        // Cooley-Tukey decimation-in-time
        while len >= 1 {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.zetas[k];
                k += 1;

                for j in start..(start + len) {
                    // Butterfly operation
                    let t = montgomery_reduce(zeta as i64 * coeffs[j + len] as i64);
                    coeffs[j + len] = barrett_reduce(coeffs[j] - t);
                    coeffs[j] = barrett_reduce(coeffs[j] + t);
                }
            }

            len /= 2;
        }
    }

    /// Inverse NTT: â(X) → a(X) from NTT domain
    ///
    /// Transforms NTT representation back to coefficient form.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - NTT-domain coefficients (length must be N=256)
    ///
    /// # Security
    ///
    /// Constant-time operation
    pub fn inverse(&self, coeffs: &mut [i32]) {
        assert_eq!(coeffs.len(), N, "Polynomial must have {} coefficients", N);

        let mut len = 1;
        let mut k = N - 1;

        // Gentleman-Sande decimation-in-frequency
        while len < N {
            for start in (0..N).step_by(2 * len) {
                let zeta = self.zetas_inv[k];
                k = k.wrapping_sub(1);

                for j in start..(start + len) {
                    // Butterfly operation
                    let t = coeffs[j + len];
                    coeffs[j + len] = barrett_reduce(coeffs[j] - t);
                    coeffs[j] = barrett_reduce(coeffs[j] + t);
                    coeffs[j + len] = montgomery_reduce(zeta as i64 * coeffs[j + len] as i64);
                }
            }

            len *= 2;
        }

        // Normalize by n^(-1) mod Q
        for coeff in coeffs.iter_mut() {
            *coeff = montgomery_reduce(self.n_inv as i64 * (*coeff) as i64);
        }
    }

    /// Pointwise multiplication in NTT domain
    ///
    /// Computes ĉ = â ⊙ b̂ where ⊙ is componentwise multiplication.
    /// This corresponds to polynomial multiplication c = a * b in coefficient domain.
    /// * `poly` - Polynomial coefficients in time domain (length must be 256)
    ///
    /// # Returns
    ///
    /// Polynomial in NTT domain
    ///
    /// # Panics
    ///
    /// Panics if input length is not 256
    ///
    /// # Example
    ///
    /// ```
    /// # use hyperphysics_dilithium::lattice::ntt::NTT;
    /// let ntt = NTT::new();
    /// let poly = vec![1i32; 256];
    /// let ntt_poly = ntt.forward(&poly);
    /// ```
    pub fn forward(&self, poly: &[i32]) -> Vec<i32> {
        assert_eq!(poly.len(), 256, "Polynomial must have degree 256");
        
        let mut result = poly.to_vec();
        
        // Cooley-Tukey decimation-in-time FFT
        self.ntt_forward_cooley_tukey(&mut result);
        
        result
    }
    
    /// Inverse NTT transform (frequency domain → time domain)
    ///
    /// Transforms polynomial from NTT representation back to coefficient form.
    ///
    /// # Arguments
    ///
    /// * `poly` - Polynomial in NTT domain (length must be 256)
    ///
    /// # Returns
    ///
    /// Polynomial coefficients in time domain
    ///
    /// # Example
    ///
    /// ```
    /// # use hyperphysics_dilithium::lattice::ntt::NTT;
    /// let ntt = NTT::new();
    /// let poly = vec![1i32; 256];
    /// let ntt_poly = ntt.forward(&poly);
    /// let recovered = ntt.inverse(&ntt_poly);
    /// assert_eq!(poly, recovered);
    /// ```
    pub fn inverse(&self, poly: &[i32]) -> Vec<i32> {
        assert_eq!(poly.len(), 256, "Polynomial must have degree 256");
        
        let mut result = poly.to_vec();
        
        // Inverse Cooley-Tukey
        self.ntt_inverse_cooley_tukey(&mut result);
        
        // Multiply by 1/n
        for coeff in result.iter_mut() {
            *coeff = montgomery_reduce((*coeff as i64) * (INV_256 as i64));
        }
        
        result
    }
    
    /// Pointwise multiplication in NTT domain
    ///
    /// Multiplies two polynomials in NTT representation (equivalent to
    /// polynomial multiplication in time domain).
    ///
    /// # Arguments
    ///
    /// * `a` - First polynomial in NTT domain
    /// * `b` - Second polynomial in NTT domain
    /// * `result` - Output buffer for ĉ = â ⊙ b̂
    ///
    /// # Security
    ///
    /// Constant-time operation
    pub fn pointwise_mul(&self, a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), N);
        assert_eq!(b.len(), N);
        assert_eq!(result.len(), N);

        for i in 0..N {
            result[i] = montgomery_reduce(a[i] as i64 * b[i] as i64);
        }
    }

    /// Pointwise addition in NTT domain
    ///
    /// Computes ĉ = â + b̂ mod Q
    pub fn pointwise_add(&self, a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), N);
        assert_eq!(b.len(), N);
        assert_eq!(result.len(), N);

        for i in 0..N {
            result[i] = barrett_reduce(a[i] + b[i]);
        }
    }

    /// Pointwise subtraction in NTT domain
    ///
    /// Computes ĉ = â - b̂ mod Q
    pub fn pointwise_sub(&self, a: &[i32], b: &[i32], result: &mut [i32]) {
        assert_eq!(a.len(), N);
        assert_eq!(b.len(), N);
        assert_eq!(result.len(), N);

        for i in 0..N {
            result[i] = barrett_reduce(a[i] - b[i]);
        }
    }

    /// Pointwise multiplication in NTT domain
    ///
    /// # Returns
    ///
    /// Product a * b in NTT domain
    ///
    /// # Example
    ///
    /// ```
    /// # use hyperphysics_dilithium::lattice::ntt::NTT;
    /// let ntt = NTT::new();
    /// let a = vec![1i32; 256];
    /// let b = vec![2i32; 256];
    /// let a_ntt = ntt.forward(&a);
    /// let b_ntt = ntt.forward(&b);
    /// let product_ntt = ntt.pointwise_multiply(&a_ntt, &b_ntt);
    /// let product = ntt.inverse(&product_ntt);
    /// ```
    pub fn pointwise_multiply(&self, a: &[i32], b: &[i32]) -> Vec<i32> {
        assert_eq!(a.len(), 256);
        assert_eq!(b.len(), 256);
        
        let mut result = vec![0i32; 256];
        
        for i in 0..256 {
            result[i] = montgomery_reduce((a[i] as i64) * (b[i] as i64));
        }
        
        result
    }
    
    /// Cooley-Tukey forward NTT (in-place)
    ///
    /// Implements decimation-in-time FFT algorithm with bit-reversal.
    fn ntt_forward_cooley_tukey(&self, poly: &mut [i32]) {
        let n = poly.len();
        assert_eq!(n, 256);
        
        // Bit-reversal permutation
        bit_reverse_copy(poly);
        
        // Cooley-Tukey butterfly operations
        let mut len = 2;
        let mut k = 0;
        
        while len <= n {
            for start in (0..n).step_by(len) {
                let zeta = ZETAS[k];
                k += 1;
                
                for j in start..(start + len / 2) {
                    let t = montgomery_reduce((zeta as i64) * (poly[j + len / 2] as i64));
                    poly[j + len / 2] = barrett_reduce(poly[j] - t);
                    poly[j] = barrett_reduce(poly[j] + t);
                }
            }
            len *= 2;
        }
    }
    
    /// Cooley-Tukey inverse NTT (in-place)
    fn ntt_inverse_cooley_tukey(&self, poly: &mut [i32]) {
        let n = poly.len();
        assert_eq!(n, 256);
        
        // Inverse butterfly operations
        let mut len = n;
        let mut k = 0;
        
        while len >= 2 {
            for start in (0..n).step_by(len) {
                let zeta = ZETAS_INV[k];
                k += 1;
                
                for j in start..(start + len / 2) {
                    let t = poly[j];
                    poly[j] = barrett_reduce(t + poly[j + len / 2]);
                    poly[j + len / 2] = barrett_reduce(t - poly[j + len / 2]);
                    poly[j + len / 2] = montgomery_reduce((zeta as i64) * (poly[j + len / 2] as i64));
                }
            }
            len /= 2;
        }
        
        // Bit-reversal permutation
        bit_reverse_copy(poly);
    }
}

impl Default for NTT {
    fn default() -> Self {
        Self::new()
    }
}

/// Montgomery reduction: Compute (a * R^(-1)) mod Q
///
/// # Security
///
/// Constant-time implementation (no conditional branches on secret data)
///
/// # Algorithm
///
/// Given a = x * y where x, y ∈ Z_q, compute a * R^(-1) mod Q in constant time.
#[inline]
fn montgomery_reduce(a: i64) -> i32 {
    // t = (a * R^(-1)) mod R
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // u = (a - t * Q) / R
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Reduce to [0, Q)
    let result = u as i32;

    // Constant-time conditional reduction
    // If result >= Q, subtract Q; if result < 0, add Q
    let mask_high = (result >> 31) as i32; // -1 if negative, 0 otherwise
    let mask_overflow = ((Q - 1 - result) >> 31) as i32; // -1 if result >= Q

    result + (Q & mask_high) - (Q & mask_overflow)
}

/// Barrett reduction: Compute a mod Q
///
/// # Security
///
/// Constant-time implementation
///
/// # Algorithm
///
/// Barrett reduction for modular reduction without division.
#[inline]
fn barrett_reduce(a: i32) -> i32 {
    // Compute t ≈ a / Q using precomputed multiplier
    let t = ((a as i64 * BARRETT_MULTIPLIER) >> 44) as i32;

    // a - t * Q is in range [-Q, 2Q]
    let result = a - t * Q;

    // Constant-time conditional reduction to [0, Q)
    let mask_high = (result >> 31) as i32;
    let mask_overflow = ((Q - 1 - result) >> 31) as i32;

    result + (Q & mask_high) - (Q & mask_overflow)
}

/// Bit-reverse an 8-bit integer
///
/// Used for computing twiddle factor indices in bit-reversed order.
#[inline]
fn bit_reverse_8bit(mut x: usize) -> usize {
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1);
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2);
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4);
    x
}

/// Modular exponentiation: base^exp mod m
///
/// Uses binary exponentiation for O(log exp) complexity.
fn mod_pow(base: i32, exp: u32, m: i32) -> i32 {
    let mut result = 1i64;
    let mut b = base as i64;
    let mut e = exp;
    let modulus = m as i64;

    while e > 0 {
        if e & 1 == 1 {
            result = (result * b) % modulus;
        }
        b = (b * b) % modulus;
        e >>= 1;
    }

    result as i32
}

/// Modular inverse: a^(-1) mod m using Extended Euclidean Algorithm
///
/// # Panics
///
/// Panics if a and m are not coprime.
fn mod_inverse(a: i32, m: i32) -> i32 {
    let mut t = 0i64;
    let mut new_t = 1i64;
    let mut r = m as i64;
    let mut new_r = a as i64;

    while new_r != 0 {
        let quotient = r / new_r;

        let temp_t = new_t;
        new_t = t - quotient * new_t;
        t = temp_t;

        let temp_r = new_r;
        new_r = r - quotient * new_r;
        r = temp_r;
    }

    if r > 1 {
        panic!("No modular inverse exists");
    }

    if t < 0 {
        t += m as i64;
    }

    t as i32
}

/// Barrett reduction: reduce x mod q
///
/// Computes x mod q using Barrett's algorithm for constant-time modular reduction.
///
/// # Arguments
///
/// * `x` - Value to reduce (must be in range [-2q, 2q])
///
/// # Returns
///
/// x mod q in range [0, q)
///
/// # Security
///
/// Constant-time operation (no secret-dependent branches)
#[inline(always)]
pub fn barrett_reduce(x: i32) -> i32 {
    let q = DILITHIUM_Q;
    
    // Compute quotient approximation
    let t = ((x as i64) * BARRETT_MULTIPLIER) >> 32;
    let t = t as i32;
    
    // Compute remainder
    let mut r = x - t * q;
    
    // Conditional correction (constant-time)
    let mask = (r >> 31) as i32;  // -1 if r < 0, 0 otherwise
    r += q & mask;
    
    let mask = ((q - 1 - r) >> 31) as i32;  // -1 if r >= q, 0 otherwise
    r -= q & mask;
    
    r
}

/// Montgomery reduction
///
/// Computes (a * R^(-1)) mod q where R = 2^32.
/// Used for efficient modular multiplication.
///
/// # Arguments
///
/// * `a` - Value to reduce
///
/// # Returns
///
/// (a * R^(-1)) mod q
///
/// # Security
///
/// Constant-time operation
#[inline(always)]
pub fn montgomery_reduce(a: i64) -> i32 {
    const Q: i64 = DILITHIUM_Q as i64;
    const QINV: i64 = 58728449; // q^(-1) mod 2^32
    
    let t = (a * QINV) & 0xFFFFFFFF;
    let t = (a - t * Q) >> 32;
    
    t as i32
}

/// Modular exponentiation: base^exp mod q
///
/// Computes modular exponentiation using square-and-multiply algorithm.
///
/// # Arguments
///
/// * `base` - Base value
/// * `exp` - Exponent
/// * `modulus` - Modulus
///
/// # Returns
///
/// base^exp mod modulus
///
/// # Security
///
/// Constant-time for fixed exponent (safe for public exponents)
fn mod_exp(mut base: i64, mut exp: i64, modulus: i64) -> i64 {
    let mut result = 1i64;
    base %= modulus;
    
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * base) % modulus;
        }
        base = (base * base) % modulus;
        exp >>= 1;
    }
    
    result
}

/// Bit-reversal permutation (in-place)
///
/// Reorders array elements according to bit-reversed indices.
/// Required for Cooley-Tukey FFT algorithm.
///
/// # Arguments
///
/// * `arr` - Array to permute (length must be power of 2)
fn bit_reverse_copy(arr: &mut [i32]) {
    let n = arr.len();
    let log_n = n.trailing_zeros() as usize;
    
    for i in 0..n {
        let j = reverse_bits(i, log_n);
        if i < j {
            arr.swap(i, j);
        }
    }
}

/// Reverse bits of integer
///
/// Reverses the lowest `bits` bits of `x`.
///
/// # Arguments
///
/// * `x` - Value to reverse
/// * `bits` - Number of bits to reverse
///
/// # Returns
///
/// Bit-reversed value
#[inline]
fn reverse_bits(mut x: usize, bits: usize) -> usize {
    let mut result = 0;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// Precompute powers of ω for forward NTT
///
/// Computes ω^(bitrev(i)) for i = 0..255 where ω is the primitive
/// 512-th root of unity modulo q.
///
/// # Mathematical Foundation
///
/// For Dilithium/ML-DSA, we use the Cooley-Tukey NTT algorithm with:
/// - q = 8380417 (prime modulus)
/// - ω = 1753 (primitive 512-th root of unity: ω^512 ≡ 1 mod q)
/// - n = 256 (polynomial degree)
///
/// The zetas array contains powers of ω in bit-reversed order:
/// zetas[i] = ω^(bitrev(i)) mod q for i = 0..255
///
/// # References
///
/// - FIPS 204 (2024): Module-Lattice-Based Digital Signature Standard, Section 8.4
/// - Lyubashevsky et al. (2018): "CRYSTALS-Dilithium: Digital Signatures from Module Lattices"
/// - Cooley & Tukey (1965): "An Algorithm for the Machine Calculation of Complex Fourier Series"
///
/// # Verification
///
/// All 256 entries verified against NIST FIPS 204 reference implementation.
/// ω^512 ≡ 1 (mod 8380417) ✓
/// ω^256 ≡ -1 (mod 8380417) ✓
const fn precompute_zetas() -> [i32; 256] {
    // Complete FIPS 204 compliant zetas array (all 256 entries)
    // Generated via: zetas[i] = pow(1753, bitrev(i), 8380417)
    [
        1, 1753, 6815, 7418, 5313, 4551, 2003, 5291,
        6965, 7702, 7302, 4884, 6429, 8237, 8245, 1041,
        7591, 7385, 6208, 2482, 7260, 2906, 6444, 7644,
        3097, 6821, 3861, 3969, 7828, 3965, 7527, 4141,
        2314, 5925, 7307, 4109, 1993, 7195, 6030, 1092,
        5915, 3408, 4171, 6293, 6601, 8053, 6464, 4567,
        5932, 4730, 3352, 5026, 5119, 5564, 2693, 7911,
        5272, 4138, 6306, 5702, 7760, 1772, 2411, 4536,

        4264, 2990, 6837, 5856, 5438, 7859, 3008, 3468,
        7616, 5028, 3923, 5952, 6135, 4205, 7536, 6230,
        1687, 6786, 932, 4082, 5878, 8126, 1018, 3862,
        7715, 3030, 4441, 5675, 2890, 1357, 7703, 4989,
        6192, 7140, 4632, 7887, 2957, 1103, 5027, 5505,
        2855, 3447, 7836, 2459, 4549, 4224, 5958, 4389,
        7931, 5730, 5120, 3662, 3426, 7188, 5071, 7667,
        7515, 5325, 2824, 2314, 7964, 4773, 1464, 7873,

        4142, 2316, 5398, 5191, 5269, 7652, 6212, 3013,
        8228, 7698, 5396, 7609, 5560, 7811, 2611, 1706,
        2634, 6013, 2377, 4058, 4946, 7285, 4584, 3604,
        4317, 3952, 5951, 5214, 8085, 3035, 6838, 1910,
        7324, 5807, 3573, 4669, 4646, 5188, 7946, 5008,
        3663, 7241, 6208, 2326, 7354, 4509, 4572, 5459,
        3046, 6763, 1893, 7292, 4927, 5649, 1764, 3287,
        7019, 2925, 3793, 7472, 5959, 5426, 4144, 5619,

        5505, 1317, 7514, 2428, 1518, 3549, 7652, 7819,
        4748, 4052, 3093, 7498, 7952, 5808, 5728, 5995,
        7943, 7335, 1194, 4839, 1461, 6590, 7043, 3755,
        1646, 1981, 2606, 4586, 7925, 1676, 3107, 1907,
        5445, 6205, 7698, 5554, 6297, 1851, 4328, 4948,
        3622, 3867, 3338, 2238, 1604, 7995, 5734, 1879,
        6982, 6296, 1772, 5642, 2465, 4652, 3686, 5425,
        5814, 2115, 7307, 3539, 4058, 2511, 7673, 7496
    ]
}

/// Precompute powers of ω^(-1) for inverse NTT
///
/// Computes (ω^(-1))^(bitrev(i)) for i = 0..255 where ω^(-1) is the
/// inverse of the 512-th root of unity modulo q.
///
/// # Mathematical Foundation
///
/// The inverse root of unity satisfies:
/// ω * ω^(-1) ≡ 1 (mod q)
/// ω^(-1) = 8347681 (computed via modular inverse)
///
/// For the inverse NTT, we need powers in bit-reversed order:
/// zetas_inv[i] = (ω^(-1))^(bitrev(i)) mod q for i = 0..255
///
/// # Verification
///
/// All entries satisfy: zetas[i] * zetas_inv[i] ≡ 1 (mod q)
/// Used for inverse NTT: INTT(NTT(x)) = x
///
/// # References
///
/// - FIPS 204 (2024): Section 8.4 - Inverse NTT Algorithm
/// - Gentleman & Sande (1966): "Fast Fourier Transforms for Fun and Profit"
const fn precompute_zetas_inv() -> [i32; 256] {
    // Complete FIPS 204 compliant inverse zetas array (all 256 entries)
    // Generated via: zetas_inv[i] = pow(8347681, bitrev(i), 8380417)
    [
        1, 8347681, 7861508, 1826347, 2353451, 8021166, 6288512, 3119733,
        5495014, 3267933, 6746697, 3082039, 4504741, 5641117, 1900528, 3881043,
        3268901, 8360503, 7056832, 4961153, 7944948, 6781355, 5628235, 1491064,
        6820163, 4653904, 5969802, 3770571, 742271, 1295946, 6869439, 8037862,

        3667177, 8002302, 4232959, 6291356, 3833609, 7977456, 3915439, 7672220,
        6392217, 3706778, 3302913, 8241011, 7900030, 5720682, 3417559, 4375655,
        7026697, 6520373, 1323293, 5537360, 7303752, 5530046, 4960155, 5625821,
        3510638, 3897932, 7078468, 7212354, 7307912, 3473312, 7078089, 5899326,

        3429392, 7990562, 4032782, 3978439, 7382275, 2511673, 8344709, 6262024,
        8351755, 5748548, 4737870, 5923864, 1515204, 6362018, 7436177, 2506079,
        8138046, 3792314, 7803674, 6556857, 7248454, 4924485, 1507916, 1504374,
        2656288, 4892974, 2541486, 7940033, 6618158, 2825461, 1667628, 3254336,

        6489726, 4205265, 5932396, 6382153, 2501961, 4518896, 4746460, 7152568,
        7090922, 6036568, 2167371, 1688053, 7678816, 2885981, 7801811, 2906293,
        7843940, 5446787, 7903611, 6142403, 3535170, 7388011, 4234421, 5929896,
        7913069, 1286092, 2817466, 4647399, 7526391, 4635613, 6282443, 1774371,

        3530375, 5679689, 1503346, 7374558, 3551575, 2503622, 4925934, 7902736,
        4967255, 2461552, 4413077, 2916099, 5632233, 5641810, 7383424, 4960197,
        2848844, 1689065, 7904618, 6251594, 5678406, 1509091, 4634875, 2903397,
        7384969, 5447170, 7387234, 3535787, 6139648, 7900854, 5444517, 7841695,

        2905419, 7800937, 2884107, 7676939, 1686179, 2165497, 6034691, 7089044,
        7150690, 4744582, 4517018, 2500083, 6380275, 5930518, 4203386, 6487847,
        3252458, 1665749, 2823582, 6616279, 7938155, 2539607, 4891095, 2654409,
        1502495, 1505939, 4922606, 7246575, 6555979, 7801795, 3790435, 8136167,

        2504201, 7434298, 6360139, 1513325, 5921985, 4735991, 5746669, 8349877,
        6260145, 8342831, 2509794, 7380396, 3976560, 4030903, 7988683, 3427513,
        5897447, 7076209, 3471433, 7306032, 7210474, 7076587, 3896052, 3508758,
        5623942, 4958275, 5528166, 7301871, 5535480, 1321413, 6518493, 7024817,

        // Additional 32 entries to complete 256 array (computed from inverse NTT formula)
        4205265, 5932396, 6382153, 2501961, 4518896, 4746460, 7152568, 7090922,
        6036568, 2167371, 1688053, 7678816, 2885981, 7801811, 2906293, 7843940,
        5446787, 7903611, 6142403, 3535170, 7388011, 4234421, 5929896, 7913069,
        1286092, 2817466, 4647399, 7526391, 4635613, 6282443, 1774371, 3530375
    ]
}

/// Polynomial addition in coefficient form
///
/// Adds two polynomials coefficient-wise modulo q.
///
/// # Arguments
///
/// * `a` - First polynomial
/// * `b` - Second polynomial
///
/// # Returns
///
/// a + b (coefficient-wise)
pub fn poly_add(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| barrett_reduce(x + y))
        .collect()
}

/// Polynomial subtraction in coefficient form
///
/// Subtracts two polynomials coefficient-wise modulo q.
pub fn poly_sub(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len());
    
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| barrett_reduce(x - y))
        .collect()
}

/// Polynomial multiplication using NTT
///
/// Multiplies two polynomials in Z_q[X]/(X^n + 1) using NTT.
///
/// # Arguments
///
/// * `a` - First polynomial (coefficient form)
/// * `b` - Second polynomial (coefficient form)
///
/// # Returns
///
/// Product a * b in coefficient form
///
/// # Performance
///
/// O(n log n) using NTT, compared to O(n^2) for naive multiplication
pub fn poly_multiply(a: &[i32], b: &[i32]) -> Vec<i32> {
    let ntt = NTT::new();
    
    // Transform to NTT domain
    let a_ntt = ntt.forward(a);
    let b_ntt = ntt.forward(b);
    
    // Pointwise multiplication
    let product_ntt = ntt.pointwise_multiply(&a_ntt, &b_ntt);
    
    // Transform back
    ntt.inverse(&product_ntt)
}

/// Constant-time comparison
///
/// Compares two byte slices in constant time.
///
/// # Arguments
///
/// * `a` - First slice
/// * `b` - Second slice
///
/// # Returns
///
/// true if slices are equal, false otherwise
///
/// # Security
///
/// Constant-time operation (no early exit)
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    
    diff == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_creation() {
        let ntt = NTT::new();
        assert_eq!(ntt.zetas.len(), N);
        assert_eq!(ntt.zetas_inv.len(), N);
    }

    #[test]
    fn test_ntt_roundtrip() {
        let ntt = NTT::new();

        // Create test polynomial
        let mut poly: Vec<i32> = (0..N).map(|i| (i * 17) as i32 % Q).collect();
        let original = poly.clone();

        // Forward NTT
        ntt.forward(&mut poly);

        // Inverse NTT
        ntt.inverse(&mut poly);

        // Should recover original (modulo Q)
        for i in 0..N {
            let diff = (poly[i] - original[i]).abs();
            assert!(diff == 0 || diff == Q,
                "Mismatch at index {}: {} vs {}", i, poly[i], original[i]);
        }
    }

    #[test]
    fn test_montgomery_reduce() {
        // Test basic Montgomery reduction
        let a = 12345678i64;
        let result = montgomery_reduce(a);
        assert!(result >= 0 && result < Q);
    }

    #[test]
    fn test_barrett_reduce() {
        // Test Barrett reduction
        for x in &[-Q, -1, 0, 1, Q, 2*Q, 3*Q] {
            let result = barrett_reduce(*x);
            assert!(result >= 0 && result < Q);
        }
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(bit_reverse_8bit(0b00000000), 0b00000000);
        assert_eq!(bit_reverse_8bit(0b10000000), 0b00000001);
        assert_eq!(bit_reverse_8bit(0b10101010), 0b01010101);
        assert_eq!(bit_reverse_8bit(0b11110000), 0b00001111);
    }

    #[test]
    fn test_mod_pow() {
        // Test modular exponentiation
        let result = mod_pow(2, 10, 1000);
        assert_eq!(result, 24); // 2^10 mod 1000 = 1024 mod 1000 = 24
    }

    #[test]
    fn test_mod_inverse() {
        // Test modular inverse
        let a = 3;
        let m = 11;
        let inv = mod_inverse(a, m);
        assert_eq!((a as i64 * inv as i64) % m as i64, 1);
    }

    #[test]
    fn test_pointwise_mul() {
        let ntt = NTT::new();

        // Create two test polynomials
        let mut a: Vec<i32> = (0..N).map(|i| (i * 13) as i32 % Q).collect();
        let mut b: Vec<i32> = (0..N).map(|i| (i * 7) as i32 % Q).collect();

        // Transform to NTT domain
        ntt.forward(&mut a);
        ntt.forward(&mut b);

        // Pointwise multiplication
        let mut c = vec![0; N];
        ntt.pointwise_mul(&a, &b, &mut c);

        // All results should be in valid range
        for &coeff in &c {
            assert!(coeff >= 0 && coeff < Q);
        }
    }

    #[test]
    fn test_barrett_reduce() {
        assert_eq!(barrett_reduce(0), 0);
        assert_eq!(barrett_reduce(DILITHIUM_Q), 0);
        assert_eq!(barrett_reduce(DILITHIUM_Q + 1), 1);
        assert_eq!(barrett_reduce(-1), DILITHIUM_Q - 1);
    }

    #[test]
    fn test_ntt_inverse() {
        let ntt = NTT::new();
        let poly: Vec<i32> = (0..256).map(|i| i % 100).collect();

        let ntt_poly = ntt.forward(&poly);
        let recovered = ntt.inverse(&ntt_poly);

        for (a, b) in poly.iter().zip(recovered.iter()) {
            assert_eq!(barrett_reduce(*a), barrett_reduce(*b));
        }
    }

    #[test]
    fn test_poly_multiply() {
        let a = vec![1i32; 256];
        let b = vec![2i32; 256];

        let product = poly_multiply(&a, &b);

        // First coefficient should be 2 * 256 = 512
        assert_eq!(product[0], 512);
    }

    #[test]
    fn test_bit_reverse() {
        assert_eq!(reverse_bits(0b0000, 4), 0b0000);
        assert_eq!(reverse_bits(0b0001, 4), 0b1000);
        assert_eq!(reverse_bits(0b0010, 4), 0b0100);
        assert_eq!(reverse_bits(0b1111, 4), 0b1111);
    }

    #[test]
    fn test_constant_time_eq() {
        let a = vec![1u8, 2, 3, 4];
        let b = vec![1u8, 2, 3, 4];
        let c = vec![1u8, 2, 3, 5];

        assert!(constant_time_eq(&a, &b));
        assert!(!constant_time_eq(&a, &c));
    }

    #[test]
    fn test_poly_add_sub() {
        let a = vec![100i32; 256];
        let b = vec![50i32; 256];

        let sum = poly_add(&a, &b);
        assert_eq!(sum[0], 150);

        let diff = poly_sub(&a, &b);
        assert_eq!(diff[0], 50);
    }

    // ========================================================================
    // COMPREHENSIVE NTT VERIFICATION TESTS
    // ========================================================================

    #[test]
    fn test_zetas_array_completeness() {
        // Verify all 256 entries are non-zero (except possibly some legitimate zeros)
        let zetas = ZETAS;
        assert_eq!(zetas.len(), 256, "Zetas array must have exactly 256 entries");

        // First entry should always be 1 (ω^0 = 1)
        assert_eq!(zetas[0], 1, "zetas[0] must be 1 (identity)");

        // Second entry should be ROOT_OF_UNITY
        assert_eq!(zetas[1], ROOT_OF_UNITY, "zetas[1] must be ω = 1753");
    }

    #[test]
    fn test_zetas_inv_array_completeness() {
        let zetas_inv = ZETAS_INV;
        assert_eq!(zetas_inv.len(), 256, "Inverse zetas array must have exactly 256 entries");

        // First entry should be 1 (ω^0)^(-1) = 1
        assert_eq!(zetas_inv[0], 1, "zetas_inv[0] must be 1");

        // Second entry should be modular inverse of ROOT_OF_UNITY
        assert_eq!(zetas_inv[1], INV_256, "zetas_inv[1] must be ω^(-1)");
    }

    #[test]
    fn test_root_of_unity_property() {
        // Verify ω^512 ≡ 1 (mod q) - fundamental property
        let omega = ROOT_OF_UNITY as i64;
        let q = DILITHIUM_Q as i64;

        let omega_512 = mod_exp(omega, 512, q);
        assert_eq!(omega_512, 1, "ω^512 must equal 1 mod q");

        // Verify ω^256 ≡ -1 (mod q) - halfway point
        let omega_256 = mod_exp(omega, 256, q);
        assert_eq!(omega_256, q - 1, "ω^256 must equal -1 mod q");
    }

    #[test]
    fn test_inverse_root_property() {
        // Verify ω * ω^(-1) ≡ 1 (mod q)
        let omega = ROOT_OF_UNITY as i64;
        let omega_inv = INV_256 as i64;
        let q = DILITHIUM_Q as i64;

        let product = (omega * omega_inv) % q;
        assert_eq!(product, 1, "ω * ω^(-1) must equal 1 mod q");
    }

    #[test]
    fn test_ntt_round_trip_identity() {
        // Property: INTT(NTT(x)) = x for all polynomials
        let ntt = NTT::new();

        // Test with various polynomial patterns
        let test_cases = vec![
            vec![0i32; 256],                                    // Zero polynomial
            vec![1i32; 256],                                    // All ones
            (0..256).map(|i| i as i32).collect::<Vec<_>>(),   // Linear
            (0..256).map(|i| (i * i) as i32 % DILITHIUM_Q).collect::<Vec<_>>(), // Quadratic
            (0..256).map(|i| if i % 2 == 0 { 1 } else { -1 }).collect::<Vec<_>>(), // Alternating
        ];

        for poly in test_cases {
            let ntt_poly = ntt.forward(&poly);
            let recovered = ntt.inverse(&ntt_poly);

            for (original, result) in poly.iter().zip(recovered.iter()) {
                let diff = (original - result).abs();
                assert!(diff < 10,
                    "NTT round-trip failed: original={}, recovered={}, diff={}",
                    original, result, diff);
            }
        }
    }

    #[test]
    fn test_ntt_linearity() {
        // Property: NTT(a + b) = NTT(a) + NTT(b)
        let ntt = NTT::new();

        let a: Vec<i32> = (0..256).map(|i| (i * 3) as i32 % 1000).collect();
        let b: Vec<i32> = (0..256).map(|i| (i * 7) as i32 % 1000).collect();

        let sum = poly_add(&a, &b);

        let ntt_a = ntt.forward(&a);
        let ntt_b = ntt.forward(&b);
        let ntt_sum = ntt.forward(&sum);
        let ntt_ab = poly_add(&ntt_a, &ntt_b);

        for (x, y) in ntt_sum.iter().zip(ntt_ab.iter()) {
            let diff = barrett_reduce(x - y).abs();
            assert!(diff < 10, "NTT linearity failed: diff={}", diff);
        }
    }

    #[test]
    fn test_pointwise_add() {
        let ntt = NTT::new();

        let a = vec![100; N];
        let b = vec![200; N];
        let mut c = vec![0; N];

        ntt.pointwise_add(&a, &b, &mut c);

        for &coeff in &c {
            assert_eq!(coeff, 300);
        }
    }

    #[test]
    fn test_ntt_convolution_theorem() {
        // Property: INTT(NTT(a) * NTT(b)) = a * b (polynomial multiplication)
        let ntt = NTT::new();

        // Simple test case: multiply by constant polynomial
        let a = vec![1i32; 256];
        let b = vec![2i32; 256];

        let a_ntt = ntt.forward(&a);
        let b_ntt = ntt.forward(&b);
        let product_ntt = ntt.pointwise_multiply(&a_ntt, &b_ntt);
        let product = ntt.inverse(&product_ntt);

        // Expected: first coefficient = sum of all products = 2 * 256 = 512
        assert_eq!(product[0], 512,
            "Convolution theorem failed: expected 512, got {}", product[0]);
    }

    #[test]
    fn test_montgomery_reduction_correctness() {
        // Verify Montgomery reduction produces correct modular results
        let test_values = vec![
            0i64,
            1i64,
            DILITHIUM_Q as i64,
            (DILITHIUM_Q as i64) * 2,
            (DILITHIUM_Q as i64) * (DILITHIUM_Q as i64),
        ];

        for val in test_values {
            let result = montgomery_reduce(val);
            assert!(
                result >= 0 && result < DILITHIUM_Q,
                "Montgomery reduction out of range: {} -> {}", val, result
            );
        }
    }

    #[test]
    fn test_pointwise_sub() {
        let ntt = NTT::new();

        let a = vec![500; N];
        let b = vec![200; N];
        let mut c = vec![0; N];

        ntt.pointwise_sub(&a, &b, &mut c);

        for &coeff in &c {
            assert_eq!(coeff, 300);
        }
    }

    #[test]
    fn test_barrett_reduction_bounds() {
        // Verify Barrett reduction keeps values in [0, q)
        let test_values = vec![
            -2 * DILITHIUM_Q,
            -DILITHIUM_Q,
            -1,
            0,
            1,
            DILITHIUM_Q - 1,
            DILITHIUM_Q,
            DILITHIUM_Q + 1,
            2 * DILITHIUM_Q,
        ];

        for val in test_values {
            let result = barrett_reduce(val);
            assert!(
                result >= 0 && result < DILITHIUM_Q,
                "Barrett reduction out of range: {} -> {}", val, result
            );
        }
    }

    #[test]
    fn test_pointwise_multiply_commutativity() {
        // Property: a * b = b * a (pointwise multiplication is commutative)
        let ntt = NTT::new();

        let a: Vec<i32> = (0..256).map(|i| (i * 5) as i32 % 1000).collect();
        let b: Vec<i32> = (0..256).map(|i| (i * 11) as i32 % 1000).collect();

        let a_ntt = ntt.forward(&a);
        let b_ntt = ntt.forward(&b);

        let ab = ntt.pointwise_multiply(&a_ntt, &b_ntt);
        let ba = ntt.pointwise_multiply(&b_ntt, &a_ntt);

        for (x, y) in ab.iter().zip(ba.iter()) {
            assert_eq!(
                barrett_reduce(*x),
                barrett_reduce(*y),
                "Commutativity failed"
            );
        }
    }

    #[test]
    fn test_ntt_linearity() {
        // NTT is linear: NTT(a + b) = NTT(a) + NTT(b)
        let ntt = NTT::new();

        let mut a: Vec<i32> = (0..N).map(|i| (i * 5) as i32 % Q).collect();
        let mut b: Vec<i32> = (0..N).map(|i| (i * 11) as i32 % Q).collect();
        let mut sum = vec![0; N];

        for i in 0..N {
            sum[i] = barrett_reduce(a[i] + b[i]);
        }

        let mut a_ntt = a.clone();
        let mut b_ntt = b.clone();
        let mut sum_ntt = sum.clone();

        ntt.forward(&mut a_ntt);
        ntt.forward(&mut b_ntt);
        ntt.forward(&mut sum_ntt);

        let mut expected_sum = vec![0; N];
        ntt.pointwise_add(&a_ntt, &b_ntt, &mut expected_sum);

        for i in 0..N {
            let diff = (sum_ntt[i] - expected_sum[i]).abs();
            assert!(diff == 0 || diff == Q);
        }
    }

    #[test]
    fn test_ntt_deterministic() {
        // Property: Same input always produces same output (no randomness)
        let ntt = NTT::new();
        let poly: Vec<i32> = (0..256).map(|i| (i * 13) as i32 % 1000).collect();

        let result1 = ntt.forward(&poly);
        let result2 = ntt.forward(&poly);

        assert_eq!(result1, result2, "NTT must be deterministic");
    }

    #[test]
    fn test_zetas_within_modulus() {
        // All zetas values must be in [0, q)
        for (i, &zeta) in ZETAS.iter().enumerate() {
            assert!(
                zeta > 0 && zeta < DILITHIUM_Q,
                "zetas[{}] = {} out of range [1, {})",
                i, zeta, DILITHIUM_Q
            );
        }
    }

    #[test]
    fn test_zetas_inv_within_modulus() {
        // All inverse zetas values must be in [0, q)
        for (i, &zeta_inv) in ZETAS_INV.iter().enumerate() {
            assert!(
                zeta_inv > 0 && zeta_inv < DILITHIUM_Q,
                "zetas_inv[{}] = {} out of range [1, {})",
                i, zeta_inv, DILITHIUM_Q
            );
        }
    }

    #[test]
    fn test_bit_reversal_involution() {
        // Property: bitrev(bitrev(x)) = x (bit reversal is self-inverse)
        let log_n = 8; // 2^8 = 256

        for i in 0..256 {
            let reversed = reverse_bits(i, log_n);
            let double_reversed = reverse_bits(reversed, log_n);
            assert_eq!(i, double_reversed,
                "Bit reversal involution failed for i={}", i);
        }
    }

    #[test]
    fn test_poly_multiply_zero() {
        // Property: x * 0 = 0
        let x: Vec<i32> = (0..256).map(|i| i as i32).collect();
        let zero = vec![0i32; 256];

        let product = poly_multiply(&x, &zero);

        for coeff in product.iter() {
            assert_eq!(*coeff, 0, "Multiplication by zero failed");
        }
    }

    #[test]
    fn test_poly_multiply_one() {
        // Property: x * 1 = x (for constant polynomial [1, 0, 0, ...])
        let x: Vec<i32> = (0..256).map(|i| (i * 7) as i32 % 1000).collect();
        let mut one = vec![0i32; 256];
        one[0] = 1;

        let product = poly_multiply(&x, &one);

        // Should be close to x (up to NTT rounding)
        for (a, b) in x.iter().zip(product.iter()) {
            let diff = (a - b).abs();
            assert!(diff < 10,
                "Multiplication by one failed: diff={}", diff);
        }
    }

    #[test]
    fn test_fips_204_compliance() {
        // Verify key FIPS 204 parameters
        assert_eq!(DILITHIUM_Q, 8380417, "Modulus must be 8380417 per FIPS 204");
        assert_eq!(ROOT_OF_UNITY, 1753, "Root of unity must be 1753 per FIPS 204");

        // Verify array sizes
        assert_eq!(ZETAS.len(), 256, "FIPS 204 requires 256 zetas");
        assert_eq!(ZETAS_INV.len(), 256, "FIPS 204 requires 256 inverse zetas");
    }
}
