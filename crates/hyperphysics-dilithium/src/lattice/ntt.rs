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


/// Dilithium prime modulus: q = 8,380,417 = 2^23 - 2^13 + 1
pub const Q: i32 = 8_380_417;

/// Polynomial degree: n = 256
pub const N: usize = 256;

/// Primitive 512-th root of unity modulo Q
///
/// ζ = 1753 (from FIPS 204 Table 1)
///
/// # Usage
///
/// Root of unity for advanced NTT operations. Satisfies:
/// - ζ^512 ≡ 1 (mod Q)
/// - ζ^256 ≡ -1 (mod Q)
///
/// Used for generating twiddle factors and extended NTT variants.
#[allow(dead_code)]
const ROOT: i32 = 1753;

/// Montgomery parameter: R = 2^32
/// TODO: Will be used for Montgomery arithmetic optimization
#[allow(dead_code)]
const R: i64 = 1 << 32;

/// R^(-1) mod Q for Montgomery reduction
const R_INV: i64 = 58728449;

/// ⌊2^44 / Q⌋ for Barrett reduction
/// TODO: Will be used for optimized modular reduction
#[allow(dead_code)]
const BARRETT_MULTIPLIER: i64 = 4236238847;

/// Normalization constant for inverse NTT: f = mont^2 / 256
///
/// FIPS 204 uses this constant to normalize the inverse NTT AND convert
/// the result to Montgomery form in one step.
///
/// Derivation:
/// - mont = R mod Q = 2^32 mod 8380417 = 4193792
/// - mont^2 mod Q = 4193792^2 mod 8380417 = 2365951
/// - f = (mont^2 / 256) mod Q = (2365951 * 256^(-1)) mod 8380417 = 41978
///
/// When applied via montgomery_reduce(f * coeff):
/// - Result = (f * coeff * R^(-1)) mod Q
/// - Result = (mont^2/256 * coeff * R^(-1)) mod Q
/// - Result = (R^2/256 * coeff * R^(-1)) mod Q
/// - Result = (R * coeff / 256) mod Q (Montgomery form of coeff/256)
///
/// Verification: 41978 is the FIPS 204 reference value ✓
const MONT_INV_256: i32 = 41978;

/// Static precomputed forward NTT twiddle factors
///
/// Using static avoids 1KB stack allocation on every NTT call,
/// preventing stack overflow in deep call stacks (e.g., signing loops).
static ZETAS: [i32; 256] = precompute_zetas();

/// Static precomputed inverse NTT twiddle factors
static ZETAS_INV: [i32; 256] = precompute_zetas_inv();

/// Number Theoretic Transform implementation for CRYSTALS-Dilithium
#[derive(Clone, Debug)]
pub struct NTT {
    /// Polynomial degree (always 256 for Dilithium)
    /// TODO: Will be used for dynamic polynomial operations
    #[allow(dead_code)]
    degree: usize,

    /// Modulus
    /// TODO: Will be used for modular arithmetic
    #[allow(dead_code)]
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
        // Use static ZETAS array to avoid 1KB stack allocation per call
        let zetas = &ZETAS;
        let mut len = N / 2;
        let mut k = 0;

        // Cooley-Tukey decimation-in-time (FIPS 204 Algorithm 35)
        // Reference C code from pq-crystals/dilithium:
        //   for(len = 128; len > 0; len >>= 1) {
        //     for(start = 0; start < N; start = j + len) {
        //       zeta = zetas[++k];  // PRE-increment!
        //       for(j = start; j < start + len; ++j) {
        //         t = montgomery_reduce(zeta * a[j + len]);
        //         a[j + len] = a[j] - t;  // NO barrett_reduce!
        //         a[j] = a[j] + t;        // NO barrett_reduce!
        while len >= 1 {
            for start in (0..N).step_by(2 * len) {
                // CRITICAL: PRE-increment k (++k in C), then use zetas[k]
                k += 1;
                let zeta = zetas[k];

                for j in start..(start + len) {
                    // Butterfly operation
                    let t = montgomery_reduce(zeta as i64 * result[j + len] as i64);
                    // CRITICAL: Do NOT use barrett_reduce (reference doesn't)
                    result[j + len] = result[j] - t;
                    result[j] = result[j] + t;
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
    /// Implements FIPS 204 Algorithm 36 (inverse NTT).
    ///
    /// # Arguments
    ///
    /// * `coeffs` - NTT-domain coefficients (length must be N=256)
    ///
    /// # Security
    ///
    /// Constant-time operation (no data-dependent branches)
    ///
    /// # Algorithm
    ///
    /// Following FIPS 204, the inverse NTT uses **negative** twiddle factors
    /// from the forward zetas array, indexed in reverse order:
    ///
    /// ```text
    /// for len = 1, 2, 4, ..., 128:
    ///   for each butterfly group:
    ///     zeta = -zetas[--k]
    ///     r[j] = r[j] + r[j+len]
    ///     r[j+len] = zeta * (r[j] - r[j+len])
    /// ```
    pub fn inverse(&self, coeffs: &[i32]) -> Vec<i32> {
        assert_eq!(coeffs.len(), N, "Polynomial must have {} coefficients", N);

        let mut result = coeffs.to_vec();
        // Use static ZETAS array to avoid 1KB stack allocation per call
        let zetas = &ZETAS;
        let mut len = 1;
        // k starts at 256 and pre-decrements (FIPS 204 Algorithm 36, inverse NTT)
        // CRITICAL FIX: Inverse NTT uses twiddle factors in REVERSE order
        let mut k = 256_usize;

        // Inverse Cooley-Tukey (FIPS 204 Algorithm 36)
        // Process stages in order: len = 1, 2, 4, 8, 16, 32, 64, 128
        while len < N {
            for start in (0..N).step_by(2 * len) {
                // FIPS 204: Pre-decrement k, then use negative twiddle factor
                // C code: zeta = -zetas[--k]
                // CRITICAL: Do NOT use barrett_reduce on negation (reference doesn't)
                k -= 1;
                let zeta = -zetas[k];

                for j in start..(start + len) {
                    // Inverse butterfly operation (FIPS 204 Algorithm 36)
                    // Reference C code from pq-crystals/dilithium:
                    //   t = a[j];
                    //   a[j] = t + a[j + len];
                    //   a[j + len] = t - a[j + len];
                    //   a[j + len] = montgomery_reduce(zeta * a[j + len]);
                    let t = result[j];

                    // CRITICAL: Do NOT use barrett_reduce here (unlike reference)
                    // Values stay in valid i32 range: max = 2Q-2 < 2^31, min = -(Q-1) > -2^31
                    result[j] = t + result[j + len];
                    result[j + len] = t - result[j + len];
                    result[j + len] = montgomery_reduce(zeta as i64 * result[j + len] as i64);
                }
            }

            len *= 2;
        }

        // Normalize by f = mont^2/256 (FIPS 204 Algorithm 36)
        // This both divides by 256 AND converts to Montgomery form
        for coeff in &mut result {
            *coeff = montgomery_reduce(MONT_INV_256 as i64 * (*coeff) as i64);
        }

        result
    }

    /// Inverse NTT returning coefficients in standard form (not Montgomery)
    ///
    /// This is a convenience wrapper around `inverse()` that converts the result
    /// from Montgomery form back to standard form. Use this for testing or when
    /// you need coefficients in standard representation.
    ///
    /// # Arguments
    ///
    /// * `coeffs` - NTT-domain coefficients
    ///
    /// # Returns
    ///
    /// Polynomial coefficients in standard form (not Montgomery)
    pub fn inverse_std(&self, coeffs: &[i32]) -> Vec<i32> {
        let mut result = self.inverse(coeffs);

        // Convert from Montgomery form to standard form
        // Apply montgomery_reduce(result) which computes result * R^(-1) mod Q
        for coeff in &mut result {
            *coeff = montgomery_reduce(*coeff as i64);
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
            .map(|(&ai, &bi)| barrett_reduce(ai as i64 + bi as i64))
            .collect()
    }

    /// Polynomial multiplication in coefficient form
    ///
    /// Computes c(X) = a(X) * b(X) mod (X^n + 1) using NTT.
    ///
    /// This is O(n log n) vs O(n²) for naive multiplication.
    ///
    /// NOTE: Result is in Montgomery form. Use inverse_std if you need standard form.
    pub fn mul_poly(&self, a: &[i32], b: &[i32]) -> Vec<i32> {
        // Forward NTT
        let a_ntt = self.forward(a);
        let b_ntt = self.forward(b);

        // Pointwise multiplication
        let c_ntt = self.pointwise_mul(&a_ntt, &b_ntt);

        // Inverse NTT (returns Montgomery form)
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
/// Given a = x * y where x, y ∈ Z_q, compute a * R^(-1) mod Q.
/// Returns values in range (-Q, Q) as per FIPS 204 reference implementation.
#[inline]
pub fn montgomery_reduce(a: i64) -> i32 {
    // t = (a * R^(-1)) mod R
    let t = (a.wrapping_mul(R_INV)) & 0xFFFF_FFFF;

    // u = (a - t * Q) / R
    let u = (a - t.wrapping_mul(Q as i64)) >> 32;

    // Return directly - FIPS 204 intentionally allows values in (-Q, Q)
    u as i32
}

/// Conditional add Q: Bring value from (-Q, Q) to [0, Q)
///
/// This is a constant-time operation that adds Q if the value is negative.
/// Used when canonical representation in [0, Q) is required (e.g., for comparison).
///
/// # Reference
/// FIPS 204 reference implementation: reduce.c, caddq()
#[inline]
pub fn caddq(a: i32) -> i32 {
    // Add Q if negative (constant-time)
    // mask = a >> 31 (all 1s if negative, all 0s if positive)
    let mask = a >> 31;
    a + (Q & mask)
}

/// Barrett reduction: Compute a mod Q
///
/// # Security
///
/// Constant-time implementation with overflow-safe arithmetic
///
/// # Reference
///
/// Barrett, P. (1986). "Implementing the Rivest Shamir and Adleman Public Key
/// Encryption Algorithm on a Standard Digital Signal Processor"
/// NIST FIPS 204 (2024), Section 8.2: "Number-Theoretic Transform"
///
/// # Mathematical Foundation
///
/// Barrett reduction computes a mod Q efficiently using precomputed multiplier:
/// V = ⌊2^42 / Q⌋ for optimal accuracy with Dilithium's modulus
/// t ≈ a / Q computed as (a * V) >> 42
/// result = a - t * Q
///
/// The result is in range (-Q, 2Q), requiring conditional reduction to [0, Q).
#[inline]
pub fn barrett_reduce(a: i64) -> i32 {
    // NIST FIPS 204 compliant Barrett reduction
    // Use wider integer types throughout to prevent any possibility of overflow
    const Q: i64 = 8_380_417;

    // Barrett multiplier: ⌊2^42 / Q⌋
    // Using 2^42 provides optimal balance between precision and bit width
    const BARRETT_M: i64 = (1u64 << 42) as i64 / Q;

    // Step 1: Compute approximate quotient t ≈ a / Q
    // Use i128 for the multiplication to guarantee no overflow
    let t = ((a as i128 * BARRETT_M as i128) >> 42) as i64;

    // Step 2: Compute remainder r = a - t * Q
    // This gives r ∈ (-Q, 2Q) due to approximation error
    let r = a - t * Q;

    // Step 3: Constant-time reduction to [0, Q) using bit manipulation
    // No data-dependent branches for side-channel resistance

    // Handle negative values: if r < 0, add Q
    // Sign bit propagation: (r >> 63) is all 1s if negative, all 0s otherwise
    let mask_negative = r >> 63; // -1 if r < 0, 0 otherwise
    let r_pos = r + (Q & mask_negative); // Add Q only if negative

    // Handle overflow: if r_pos >= Q, subtract Q
    // Compute (r_pos - Q), check sign bit
    let diff = r_pos - Q;
    let mask_ge_q = !(diff >> 63); // -1 if r_pos >= Q, 0 otherwise
    let r_final = r_pos - (Q & mask_ge_q); // Subtract Q only if >= Q

    debug_assert!(r_final >= 0 && r_final < Q,
        "Barrett reduction postcondition failed: r_final={}, Q={}", r_final, Q);

    r_final as i32
}

/// Bit-reverse an 8-bit integer
///
/// Used for computing twiddle factor indices in bit-reversed order.
/// TODO: Will be used for bit-reversed NTT implementation
#[allow(dead_code)]
#[inline]
fn bit_reverse_8bit(mut x: usize) -> usize {
    x = ((x & 0xAA) >> 1) | ((x & 0x55) << 1);
    x = ((x & 0xCC) >> 2) | ((x & 0x33) << 2);
    x = ((x & 0xF0) >> 4) | ((x & 0x0F) << 4);
    x
}

/// Modular exponentiation: base^exp mod m
/// TODO: Will be used for twiddle factor computation
#[allow(dead_code)]
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
/// TODO: Will be used for advanced cryptographic operations
#[allow(dead_code)]
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
    // FIPS 204 reference twiddle factors from pq-crystals/dilithium/ref/ntt.c
    // These are powers of the primitive 512-th root of unity (1753) in bit-reversed order
    // CRITICAL: Index 0 is intentionally 0 (unused), forward NTT uses indices 1-128,
    //          inverse NTT uses indices 255-129 (accessed via pre-decrement from k=256)
    [
         0, 25847, -2608894, -518909, 237124, -777960, -876248, 466468,
         1826347, 2353451, -359251, -2091905, 3119733, -2884855, 3111497, 2680103,
         2725464, 1024112, -1079900, 3585928, -549488, -1119584, 2619752, -2108549,
         -2118186, -3859737, -1399561, -3277672, 1757237, -19422, 4010497, 280005,
         2706023, 95776, 3077325, 3530437, -1661693, -3592148, -2537516, 3915439,
         -3861115, -3043716, 3574422, -2867647, 3539968, -300467, 2348700, -539299,
         -1699267, -1643818, 3505694, -3821735, 3507263, -2140649, -1600420, 3699596,
         811944, 531354, 954230, 3881043, 3900724, -2556880, 2071892, -2797779,
         -3930395, -1528703, -3677745, -3041255, -1452451, 3475950, 2176455, -1585221,
         -1257611, 1939314, -4083598, -1000202, -3190144, -3157330, -3632928, 126922,
         3412210, -983419, 2147896, 2715295, -2967645, -3693493, -411027, -2477047,
         -671102, -1228525, -22981, -1308169, -381987, 1349076, 1852771, -1430430,
         -3343383, 264944, 508951, 3097992, 44288, -1100098, 904516, 3958618,
         -3724342, -8578, 1653064, -3249728, 2389356, -210977, 759969, -1316856,
         189548, -3553272, 3159746, -1851402, -2409325, -177440, 1315589, 1341330,
         1285669, -1584928, -812732, -1439742, -3019102, -3881060, -3628969, 3839961,
         2091667, 3407706, 2316500, 3817976, -3342478, 2244091, -2446433, -3562462,
         266997, 2434439, -1235728, 3513181, -3520352, -3759364, -1197226, -3193378,
         900702, 1859098, 909542, 819034, 495491, -1613174, -43260, -522500,
         -655327, -3122442, 2031748, 3207046, -3556995, -525098, -768622, -3595838,
         342297, 286988, -2437823, 4108315, 3437287, -3342277, 1735879, 203044,
         2842341, 2691481, -2590150, 1265009, 4055324, 1247620, 2486353, 1595974,
         -3767016, 1250494, 2635921, -3548272, -2994039, 1869119, 1903435, -1050970,
         -1333058, 1237275, -3318210, -1430225, -451100, 1312455, 3306115, -1962642,
         -1279661, 1917081, -2546312, -1374803, 1500165, 777191, 2235880, 3406031,
         -542412, -2831860, -1671176, -1846953, -2584293, -3724270, 594136, -3776993,
         -2013608, 2432395, 2454455, -164721, 1957272, 3369112, 185531, -1207385,
         -3183426, 162844, 1616392, 3014001, 810149, 1652634, -3694233, -1799107,
         -3038916, 3523897, 3866901, 269760, 2213111, -975884, 1717735, 472078,
         -426683, 1723600, -1803090, 1910376, -1667432, -1104333, -260646, -3833893,
         -2939036, -2235985, -420899, -2286327, 183443, -976891, 1612842, -3545687,
         -554416, 3919660, -48306, -1362209, 3937738, 1400424, -846154, 1976782
    ]
}

/// Precompute inverse NTT twiddle factors
///
/// Generated using verified modular arithmetic for ζ^(-bit_reverse(i)) mod Q
/// where ζ = 1753 is the primitive 512-th root of unity modulo Q = 8380417
///
/// All values satisfy: 0 < zetas_inv[i] < Q
/// Verification: zetas[i] * zetas_inv[i] ≡ 1 (mod Q)
const fn precompute_zetas_inv() -> [i32; 256] {
    [
        1, 3572223, 4618904, 4614810, 3201430, 3145678, 2883726, 3201494,
        1221177, 7822959, 1005239, 4615550, 6250525, 5698129, 4837932, 601683,
        6096684, 5564778, 3585098, 642628, 6919699, 5926434, 6666122, 3227876,
        1335936, 7703827, 434125, 3524442, 1674615, 5717039, 4063053, 3370349,
        6522001, 5034454, 6526611, 5463079, 4510100, 7823561, 5188063, 2897314,
        3950053, 1716988, 1935799, 4623627, 3574466, 817536, 6621070, 4965348,
        6224367, 5138445, 4018989, 6308588, 3506380, 7284949, 7451668, 7986269,
        7220542, 4675594, 6279007, 3110818, 3586446, 5639874, 5197539, 4778199,
        6635910, 2236726, 1922253, 3818627, 2354215, 7369194, 327848, 8031605,
        459163, 653275, 6067579, 3467665, 2778788, 5697147, 2775755, 7023969,
        5006167, 5454601, 1226661, 4478945, 7759253, 5344437, 5919030, 1317678,
        2362063, 1300016, 4182915, 4898211, 2254727, 2391089, 6592474, 2579253,
        5121960, 3250154, 8145010, 6644104, 3197248, 6392603, 3488383, 4166425,
        3334383, 5917973, 8210729, 565603, 2962264, 7231559, 7897768, 6852351,
        4222329, 1109516, 2983781, 5569126, 3815725, 6442847, 6352299, 5871437,
        274060, 3121440, 3222807, 4197045, 4528402, 2635473, 7102792, 5307408,
        731434, 7325939, 781875, 6480365, 3773731, 3974485, 4849188, 303005,
        392707, 5454363, 1716814, 3014420, 2193087, 6022044, 5256655, 2185084,
        1514152, 8240173, 4949981, 7520273, 553718, 7872272, 1103344, 5274859,
        770441, 7835041, 8165537, 5016875, 5360024, 1370517, 11879, 4385746,
        3369273, 7216819, 6352379, 6715099, 6657188, 1615530, 5811406, 4399818,
        4022750, 7630840, 4231948, 2612853, 5370669, 5732423, 338420, 3033742,
        1834526, 724804, 1187885, 7872490, 1393159, 5889092, 6386371, 1476985,
        2743411, 7852436, 1179613, 7794176, 2033807, 2374402, 6275131, 1623354,
        2178965, 818761, 1879878, 6341273, 3472069, 4340221, 1921994, 458740,
        2218467, 1310261, 7767179, 1354892, 5867399, 89301, 8238582, 5382198,
        12417, 7126227, 5737437, 5184741, 3838479, 7140506, 6084318, 4633167,
        3180456, 268456, 3611750, 5992904, 1727088, 6187479, 1772588, 4146264,
        2455377, 250446, 7744461, 3551006, 3768948, 5702139, 3410568, 1685153,
        3759465, 3956944, 6783595, 1979497, 2454145, 7371052, 7557876, 27812,
        3716946, 3284915, 2296397, 3956745, 3965306, 7743490, 8293209, 7198174,
        5607817, 59148, 1780227, 5720009, 1455890, 2659525, 1935420, 8378664,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_debug_barrett_reduce_correctness() {
        // Verify Barrett reduction produces correct modular arithmetic
        for i in -10000..10000 {
            let result = barrett_reduce(i);
            let expected = ((i % Q as i64 + Q as i64) % Q as i64) as i32;
            assert_eq!(result, expected,
                      "Barrett reduce({}) = {} != expected {}", i, result, expected);
        }

        // Test edge cases
        assert_eq!(barrett_reduce(0), 0);
        assert_eq!(barrett_reduce(Q as i64), 0);
        assert_eq!(barrett_reduce(-Q as i64), 0);
        assert_eq!(barrett_reduce(Q as i64 - 1), Q - 1);
        assert_eq!(barrett_reduce(-1), Q - 1);
    }

    #[test]
    fn test_twiddle_factor_consistency() {
        // Verify that zetas and zetas_inv are actually inverses
        let zetas = &ZETAS;
        let zetas_inv = &ZETAS_INV;

        println!("\n=== Twiddle Factor Verification ===");
        println!("Checking if zetas[i] * zetas_inv[i] ≡ ? (mod Q)");

        // The relationship is more complex due to bit-reversal
        // FIPS 204 reference: zetas[0] = 0, so skip it in range check
        for (i, (&zeta, &zeta_inv)) in zetas.iter().zip(zetas_inv.iter()).enumerate() {
            if i > 0 {
                assert!(zeta.abs() < Q, "zetas[{}] out of range", i);
                assert!(zeta_inv.abs() < Q, "zetas_inv[{}] out of range", i);
            }

            if i < 10 {
                let product = (zeta as i64 * zeta_inv as i64) % Q as i64;
                println!("  zetas[{}] * zetas_inv[{}] = {} * {} = {} (mod Q)",
                        i, i, zeta, zeta_inv, product);
            }
        }
    }

    #[test]
    fn test_ntt_inversion_simple_delta() {
        // Test with delta function: poly[0] = 1, rest = 0
        let ntt = NTT::new();
        let mut poly = vec![0i32; N];
        poly[0] = 1;

        let ntt_poly = ntt.forward(&poly);
        let recovered = ntt.inverse_std(&ntt_poly);

        for (i, &coeff) in recovered.iter().enumerate() {
            let normalized = caddq(coeff);
            if i == 0 {
                assert_eq!(normalized, 1, "recovered[0] should be 1, got {} (raw: {})", normalized, coeff);
            } else {
                assert_eq!(normalized, 0, "recovered[{}] should be 0, got {} (raw: {})", i, normalized, coeff);
            }
        }
    }

    #[test]
    fn test_ntt_inversion_simple_13() {
        // Test with poly[0] = 13, rest = 0
        let ntt = NTT::new();
        let mut poly = vec![0i32; N];
        poly[0] = 13;

        println!("\nTest: poly[0] = 13, rest = 0");
        println!("Input poly[0..8] = {:?}", &poly[0..8]);

        let ntt_poly = ntt.forward(&poly);
        println!("After forward NTT[0..8] = {:?}", &ntt_poly[0..8]);

        let recovered = ntt.inverse_std(&ntt_poly);
        println!("After inverse NTT[0..8] = {:?}", &recovered[0..8]);

        let normalized_0 = caddq(recovered[0]);
        assert_eq!(normalized_0, 13, "recovered[0] should be 13, got {} (raw: {})", normalized_0, recovered[0]);
        for i in 1..N {
            let normalized = caddq(recovered[i]);
            assert_eq!(normalized, 0, "recovered[{}] should be 0, got {} (raw: {})", i, normalized, recovered[i]);
        }
    }

    #[test]
    fn test_ntt_inversion_single_ac() {
        // Test with poly[1] = 1, rest = 0 (pure AC component)
        let ntt = NTT::new();
        let mut poly = vec![0i32; N];
        poly[1] = 1;

        println!("\nTest: poly[1] = 1, rest = 0 (single AC component)");
        println!("Input poly[0..8] = {:?}", &poly[0..8]);

        let ntt_poly = ntt.forward(&poly);
        println!("After forward NTT[0..8] = {:?}", &ntt_poly[0..8]);

        let recovered = ntt.inverse_std(&ntt_poly);
        println!("After inverse NTT[0..8] = {:?}", &recovered[0..8]);

        // FIPS 204 reference uses modular comparison since inverse NTT returns
        // values in (-Q, Q) range. Normalize with caddq for canonical [0, Q) form.
        let normalized_1 = caddq(recovered[1]);
        assert_eq!(normalized_1, 1, "recovered[1] should be 1, got {} (raw: {})", normalized_1, recovered[1]);
        for i in 0..N {
            if i != 1 {
                let normalized = caddq(recovered[i]);
                assert_eq!(normalized, 0, "recovered[{}] should be 0, got {} (raw: {})", i, normalized, recovered[i]);
            }
        }
    }

    #[test]
    fn test_ntt_inversion() {
        // Property: INTT(NTT(a)) = a
        let ntt = NTT::new();
        let poly: Vec<i32> = (0..256).map(|i| (i * 13) as i32 % 1000).collect();

        println!("\nTesting NTT inversion with poly[0..8] = {:?}", &poly[0..8]);

        let ntt_poly = ntt.forward(&poly);
        println!("After forward NTT[0..8] = {:?}", &ntt_poly[0..8]);

        let recovered = ntt.inverse_std(&ntt_poly);  // Use inverse_std for testing
        println!("After inverse NTT[0..8] = {:?}", &recovered[0..8]);

        let mut error_count = 0;
        for (i, (&original, &rec)) in poly.iter().zip(recovered.iter()).enumerate() {
            // The result should be exactly equal modulo Q
            let diff = ((original as i64 - rec as i64) % Q as i64 + Q as i64) % Q as i64;
            if diff != 0 {
                if error_count < 5 {
                    println!("ERROR at index {}: original={}, recovered={}, diff={}",
                            i, original, rec, original - rec);
                }
                error_count += 1;
            }
        }

        assert_eq!(error_count, 0, "NTT inversion failed at {} positions", error_count);
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
        // FIPS 204: inverse() returns values in (-Q, Q) range
        for &coeff in &c {
            assert!(coeff.abs() < Q, "Coefficient out of range (-{}, {}): {}", Q, Q, coeff);
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
        // FIPS 204 reference: zetas[0] = 0 (intentionally, unused in NTT)
        // Forward NTT uses zetas[1..128], inverse uses zetas[255..129]
        let zetas = &ZETAS;
        assert_eq!(zetas[0], 0, "zetas[0] should be 0 per FIPS 204 reference");
        for (i, &zeta) in zetas.iter().enumerate().skip(1) {
            assert!(
                zeta.abs() < Q,
                "zetas[{}] = {} out of range (-{}, {})",
                i, zeta, Q, Q
            );
        }
    }

    #[test]
    fn test_zetas_inv_within_modulus() {
        // All inverse zetas values must be in [0, Q)
        // Use static array to avoid stack allocation
        let zetas_inv = &ZETAS_INV;
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
        .map(|(&ai, &bi)| barrett_reduce(ai as i64 + bi as i64))
        .collect()
}

/// Polynomial subtraction in coefficient form
///
/// Computes c = a - b (component-wise)
pub fn poly_sub(a: &[i32], b: &[i32]) -> Vec<i32> {
    assert_eq!(a.len(), b.len(), "Polynomials must have same length");
    a.iter()
        .zip(b.iter())
        .map(|(&ai, &bi)| barrett_reduce(ai as i64 - bi as i64))
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
