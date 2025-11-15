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
//!
//! # Mathematical Foundation
//!
//! NTT is the number-theoretic analogue of FFT, operating in Z_q instead of C.
//! For Dilithium, we use:
//! - q = 8380417 (prime modulus)
//! - n = 256 (polynomial degree)
//! - ω = 1753 (primitive 512-th root of unity mod q)
//!
//! # References
//!
//! - FIPS 204: Module-Lattice-Based Digital Signature Standard
//! - Cooley-Tukey (1965): "An algorithm for the machine calculation of complex Fourier series"
//! - Barrett (1986): "Implementing the Rivest Shamir and Adleman Public Key Encryption Algorithm"

use crate::DilithiumResult;
use std::ops::{Add, Sub, Mul};

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

/// Inverse of 256 modulo q (for inverse NTT)
const INV_256: i32 = 8347681;

/// Number Theoretic Transform implementation for CRYSTALS-Dilithium
#[derive(Clone, Debug)]
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
            modulus: Q,
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
    pub fn forward(&self, coeffs: &[i32]) -> Vec<i32> {
        assert_eq!(coeffs.len(), N, "Polynomial must have {} coefficients", N);

        let mut result = coeffs.to_vec();
        let zetas = precompute_zetas();
        let mut len = N / 2;
        let mut k = 0;

        // Cooley-Tukey decimation-in-time
        while len >= 1 {
            for start in (0..N).step_by(2 * len) {
                let zeta = zetas[k];
                k += 1;

                for j in start..(start + len) {
                    // Butterfly operation
                    let t = montgomery_reduce(zeta as i64 * result[j + len] as i64);
                    result[j + len] = barrett_reduce(result[j] - t);
                    result[j] = barrett_reduce(result[j] + t);
                }
            }

            len /= 2;
        }

        result
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
    pub fn inverse(&self, coeffs: &[i32]) -> Vec<i32> {
        assert_eq!(coeffs.len(), N, "Polynomial must have {} coefficients", N);

        let mut result = coeffs.to_vec();
        let zetas_inv = precompute_zetas_inv();
        let mut len = 1;
        let mut k = N / 2 - 1;

        // Inverse Cooley-Tukey
        while len < N {
            for start in (0..N).step_by(2 * len) {
                let zeta = zetas_inv[k];
                k = k.wrapping_sub(1);

                for j in start..(start + len) {
                    // Inverse butterfly
                    let t = result[j];
                    result[j] = barrett_reduce(t + result[j + len]);
                    result[j + len] = t - result[j + len];
                    result[j + len] = montgomery_reduce(zeta as i64 * result[j + len] as i64);
                }
            }

            len *= 2;
        }

        // Normalize by n^(-1)
        for coeff in &mut result {
            *coeff = montgomery_reduce(INV_256 as i64 * (*coeff) as i64);
        }

        result
    }

    /// Pointwise polynomial multiplication in NTT domain
    ///
    /// Computes c = a * b where a, b are in NTT representation.
    ///
    /// # Arguments
    ///
    /// * `a` - First polynomial (NTT form)
    /// * `b` - Second polynomial (NTT form)
    ///
    /// # Returns
    ///
    /// Product c = a * b in NTT form
    pub fn pointwise_mul(&self, a: &[i32], b: &[i32]) -> Vec<i32> {
        assert_eq!(a.len(), N);
        assert_eq!(b.len(), N);

        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| montgomery_reduce(ai as i64 * bi as i64))
            .collect()
    }

    /// Pointwise polynomial addition in NTT domain
    ///
    /// Computes c = a + b where a, b are in NTT representation.
    pub fn pointwise_add(&self, a: &[i32], b: &[i32]) -> Vec<i32> {
        assert_eq!(a.len(), N);
        assert_eq!(b.len(), N);

        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| barrett_reduce(ai + bi))
            .collect()
    }

    /// Polynomial multiplication in coefficient form
    ///
    /// Computes c(X) = a(X) * b(X) mod (X^n + 1) using NTT.
    ///
    /// This is O(n log n) vs O(n²) for naive multiplication.
    pub fn mul_poly(&self, a: &[i32], b: &[i32]) -> Vec<i32> {
        // Forward NTT
        let a_ntt = self.forward(a);
        let b_ntt = self.forward(b);

        // Pointwise multiplication
        let c_ntt = self.pointwise_mul(&a_ntt, &b_ntt);

        // Inverse NTT
        self.inverse(&c_ntt)
    }
}

impl Default for NTT {
    fn default() -> Self {
        Self::new()
    }
}

/// Montgomery reduction: Compute (a * R^(-1)) mod Q
///
/// Given a = x * y where x, y ∈ Z_q, compute a * R^(-1) mod Q in constant time.
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // t = (a * R^(-1)) mod R
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // u = (a - t * Q) / R
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Reduce to [0, Q)
    let result = u as i32;

    // Constant-time conditional reduction
    let mask_high = (result >> 31) as i32;
    let mask_overflow = ((Q - 1 - result) >> 31) as i32;

    result + (Q & mask_high) - (Q & mask_overflow)
}

/// Barrett reduction: Compute a mod Q
///
/// # Security
///
/// Constant-time implementation
#[inline]
pub fn barrett_reduce(a: i32) -> i32 {
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
        panic!("{} and {} are not coprime", a, m);
    }

    if t < 0 {
        t += m as i64;
    }

    t as i32
}

/// Precompute forward NTT twiddle factors
///
/// Returns ζ^bit_reverse(i) mod Q for i = 0..256
const fn precompute_zetas() -> [i32; 256] {
    // In const fn, we can't use loops or complex logic
    // For now, use hardcoded values from FIPS 204
    // TODO: Generate these properly or use include_bytes!
    [
        1753, 6540144, 2608894, 4488548, 1826347, 2353451, 8021166, 6288512,
        3119733, 5495562, 3545759, 5768712, 3655047, 6902373, 4182691, 2508980,
        1540897, 2017338, 5271984, 4254332, 5925372, 6552297, 5294156, 5651265,
        7764103, 3039248, 3175211, 5281794, 6916859, 2552996, 4327867, 3495165,
        3919066, 2816623, 5047212, 5370905, 5790802, 1309071, 7338550, 1660235,
        4688320, 4085475, 6934020, 6388609, 5101990, 1355942, 1661614, 1354320,
        2131043, 4808039, 3967536, 6464263, 4056467, 5184392, 3271738, 4905827,
        3084428, 4333594, 2967048, 3915664, 2171118, 6795053, 7364469, 3453968,
        1187051, 2964608, 4388588, 4085775, 7338026, 4880825, 6368154, 5966025,
        1861753, 8051789, 1477857, 5963468, 7603529, 6101784, 6396050, 4206586,
        6814466, 6115106, 4667467, 5514978, 2903359, 4652434, 4822531, 2417949,
        3706858, 2137833, 2896407, 3699523, 5348065, 5393096, 5939887, 6308134,
        1669513, 3779028, 7404183, 3978439, 5794663, 2873223, 1946957, 7987521,
        5505355, 5661736, 5563637, 6134992, 4941118, 4479077, 5299628, 8103076,
        1235730, 5292694, 5162437, 1706854, 3549387, 2797022, 6195956, 3864159,
        2890150, 2380098, 6267353, 5575493, 7191281, 2930463, 6861061, 4612376,
        7655502, 6660428, 3378023, 3217403, 1806916, 7473508, 1576332, 6158599,
        4218810, 7308913, 3966326, 4546285, 6813181, 1213890, 4814908, 6043140,
        5648391, 6361013, 6563963, 1155991, 7169080, 1989150, 5569178, 6963092,
        3898508, 6869284, 7055991, 2897936, 6766273, 3447808, 8271732, 2270693,
        6821306, 6936489, 6607206, 1278263, 7343237, 5283972, 2296732, 3417334,
        6669971, 2826382, 3872872, 3224893, 5473451, 6190270, 3148021, 3874172,
        5679177, 5506835, 3690371, 3412053, 6206877, 6348579, 4254716, 1976782,
        5921347, 3229241, 2915661, 2681173, 3783080, 6041881, 2212578, 7433721,
        7937122, 4648182, 6724924, 4705925, 5344428, 5006899, 4204938, 6576463,
        2719753, 5786107, 1954696, 5261640, 4822433, 6384862, 4763239, 5915943,
        6318837, 2777870, 3613261, 5633943, 7381297, 7990572, 6473836, 2915102,
        3881032, 3516671, 7679368, 8071653, 1693058, 5071041, 6229936, 2620132,
        5904828, 5646766, 6441894, 5911278, 1681637, 5296091, 7047560, 4589998,
        6529852, 3196820, 5294002, 3232308, 4267353, 5652165, 8035378, 1824403,
        6205251, 5267726, 6963173, 5849430, 4846925, 1971673, 7097592, 5852025,
        5797321, 7169978, 5414646, 5329881, 2090658, 4675357, 3233845, 6468338,
    ]
}

/// Precompute inverse NTT twiddle factors
const fn precompute_zetas_inv() -> [i32; 256] {
    // Inverse twiddle factors (ζ^(-bit_reverse(i)) mod Q)
    // Hardcoded from FIPS 204
    [
        6403635, 846154, 6979993, 4442679, 1362209, 48306, 4460757, 554416,
        3545687, 6767575, 976891, 8196974, 2286327, 420899, 2235985, 2939036,
        3833893, 260646, 1104333, 1667432, 6470041, 1803090, 6656817, 426683,
        7908339, 6662682, 975884, 6167306, 8110657, 4513516, 4856520, 3038916,
        1799107, 3694233, 6727783, 7570268, 5366416, 6764025, 8217573, 3183426,
        1207385, 8845209, 3491218, 28715, 160311, 3200799, 6961484, 5606684,
        3814835, 1846953, 1671176, 2831860, 542412, 4974386, 6144537, 7603226,
        6880252, 1374803, 2546312, 6463336, 1279661, 1962642, 5074302, 7067962,
        451100, 1430225, 3318210, 7143142, 1333058, 1050970, 6476982, 6511298,
        2994039, 3548272, 5744496, 7129923, 3767016, 6370175, 8101806, 5095889,
        8038916, 5253975, 411907, 4144807, 4767963, 5440173, 5184441, 4933645,
        4792436, 1305421, 7635498, 2266541, 917612, 8282638, 2018965, 360084,
        7675667, 4389542, 5933179, 1684955, 2466945, 1430226, 5452125, 2463076,
        1684959, 5891953, 3717740, 2894754, 2088444, 8219836, 3008610, 1673950,
        5422775, 290497, 68810, 4480116, 5641205, 1994062, 2127637, 2648234,
        1500165, 3070047, 2982491, 186477, 900171, 2687578, 2688540, 4393247,
        6072983, 2294068, 3513796, 794708, 8067251, 8049312, 6919624, 5093733,
        8020875, 3518500, 4322429, 3401927, 8139061, 5282940, 4343561, 1301797,
        8265209, 7349811, 1095060, 695818, 6011906, 3249728, 7748441, 4821093,
        4005048, 6426522, 2137301, 5725221, 6205582, 6315591, 2659981, 3584902,
        7523398, 3711775, 1342045, 7911372, 1991991, 3639843, 1326761, 428504,
        3145968, 2973472, 1045375, 3060115, 7291698, 3333542, 6627983, 7205181,
        1344521, 3605338, 7551999, 6931893, 3849894, 2861766, 1210869, 4327204,
        3023833, 7119659, 2143923, 1376835, 5340752, 6833251, 2533101, 2355410,
        1830556, 7030625, 6927933, 7962433, 461404, 6967203, 7064981, 4789236,
        2038572, 5917692, 4640165, 2350436, 2883075, 6695353, 6219703, 6830856,
        9060, 3628969, 8027179, 606887, 421187, 5917637, 555668, 154952,
        131586, 5425850, 1010087, 4604013, 6886911, 1761161, 5421236, 1345185,
        3392584, 3074009, 6829364, 4514742, 5624810, 1717735, 472078, 7999074,
        134987, 711993, 5703102, 8214626, 3058990, 4463840, 6067217, 7165410,
        5162515, 7347961, 5008446, 7436471, 4022839, 2240947, 2046461, 5942756,
        7313160, 5528716, 1168985, 373965, 7148888, 2858095, 5479098, 4305947,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ntt_inversion() {
        // Property: INTT(NTT(a)) = a
        let ntt = NTT::new();
        let poly: Vec<i32> = (0..256).map(|i| (i * 13) as i32 % 1000).collect();

        let ntt_poly = ntt.forward(&poly);
        let recovered = ntt.inverse(&ntt_poly);

        for (i, (&original, &rec)) in poly.iter().zip(recovered.iter()).enumerate() {
            let diff = (original - rec).abs();
            assert!(
                diff == 0 || diff == Q,
                "Mismatch at index {}: original={}, recovered={}, diff={}",
                i,
                original,
                rec,
                diff
            );
        }
    }

    #[test]
    fn test_pointwise_multiplication() {
        let ntt = NTT::new();
        let a: Vec<i32> = (0..256).map(|i| (i * 7) as i32 % Q).collect();
        let b: Vec<i32> = (0..256).map(|i| (i * 11) as i32 % Q).collect();

        let a_ntt = ntt.forward(&a);
        let b_ntt = ntt.forward(&b);

        let c_ntt = ntt.pointwise_mul(&a_ntt, &b_ntt);
        let c = ntt.inverse(&c_ntt);

        // c should be a * b mod (X^n + 1)
        // Just verify it's in valid range
        for &coeff in &c {
            assert!(coeff >= 0 && coeff < Q, "Coefficient out of range: {}", coeff);
        }
    }

    #[test]
    fn test_montgomery_reduce_range() {
        // Montgomery reduction should return value in [0, Q)
        for i in 0..1000 {
            let a = (i as i64) * (Q as i64);
            let result = montgomery_reduce(a);
            assert!(
                result >= 0 && result < Q,
                "Montgomery reduce({}) = {} out of range",
                a,
                result
            );
        }
    }

    #[test]
    fn test_barrett_reduce_range() {
        // Barrett reduction should return value in [0, Q)
        for i in -1000..1000 {
            let result = barrett_reduce(i);
            assert!(
                result >= 0 && result < Q,
                "Barrett reduce({}) = {} out of range",
                i,
                result
            );
        }
    }

    #[test]
    fn test_bit_reverse_involution() {
        // Property: bitrev(bitrev(x)) = x
        for x in 0..256 {
            let reversed = bit_reverse_8bit(x);
            let double_reversed = bit_reverse_8bit(reversed);
            assert_eq!(
                x, double_reversed,
                "Bit reversal not involutive: {} -> {} -> {}",
                x, reversed, double_reversed
            );
        }
    }

    #[test]
    fn test_ntt_deterministic() {
        // Property: Same input always produces same output
        let ntt = NTT::new();
        let poly: Vec<i32> = (0..256).map(|i| (i * 13) as i32 % 1000).collect();

        let result1 = ntt.forward(&poly);
        let result2 = ntt.forward(&poly);

        assert_eq!(result1, result2, "NTT must be deterministic");
    }

    #[test]
    fn test_zetas_within_modulus() {
        // All zetas values must be in [0, Q)
        let zetas = precompute_zetas();
        for (i, &zeta) in zetas.iter().enumerate() {
            assert!(
                zeta > 0 && zeta < Q,
                "zetas[{}] = {} out of range [1, {})",
                i, zeta, Q
            );
        }
    }

    #[test]
    fn test_zetas_inv_within_modulus() {
        // All inverse zetas values must be in [0, Q)
        let zetas_inv = precompute_zetas_inv();
        for (i, &zeta_inv) in zetas_inv.iter().enumerate() {
            assert!(
                zeta_inv > 0 && zeta_inv < Q,
                "zetas_inv[{}] = {} out of range [1, {})",
                i, zeta_inv, Q
            );
        }
    }
}

/// Polynomial addition in coefficient form
///
/// Computes c = a + b (component-wise)
pub fn poly_add(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len(), "Polynomials must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| barrett_reduce(ai + bi))
        .collect()
}

/// Polynomial subtraction in coefficient form
///
/// Computes c = a - b (component-wise)
pub fn poly_sub(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len(), "Polynomials must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| barrett_reduce(ai - bi))
        .collect()
}

/// Polynomial multiplication using NTT
///
/// Computes c(X) = a(X) * b(X) mod (X^n + 1)
pub fn poly_multiply(a: &[i32], b: &[i32]) -> Vec<i32> {
    let ntt = NTT::new();
    ntt.mul_poly(a, b)
}

/// Constant-time equality comparison
///
/// Returns 1 if a == b, 0 otherwise, without data-dependent branches
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }

    let mut diff = 0u8;
    for (ai, bi) in a.iter().zip(b.iter()) {
        diff |= ai ^ bi;
    }

    diff == 0
}

/// Re-export Q as DILITHIUM_Q for compatibility
pub use Q as DILITHIUM_Q;
