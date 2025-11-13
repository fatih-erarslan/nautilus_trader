//! Number Theoretic Transform (NTT) for polynomial operations
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

    /// Forward NTT: a(X) → â(X) in NTT domain
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
}
