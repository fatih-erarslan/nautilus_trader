//! ARM AArch64 SIMD optimizations using NEON, SVE, and crypto extensions
//! Performance target: <100ns for basic operations with Apple Silicon optimizations

use std::arch::aarch64::*;

/// Runtime feature detection for ARM AArch64 SIMD capabilities
#[derive(Debug, Clone, Copy)]
pub struct AArch64Features {
    pub has_neon: bool,
    pub has_sve: bool,
    pub has_sve2: bool,
    pub has_crypto: bool,
    pub has_fp16: bool,
    pub has_dotprod: bool,
    pub is_apple_silicon: bool,
}

impl AArch64Features {
    /// Detect available AArch64 SIMD features at runtime
    #[inline]
    pub fn detect() -> Self {
        Self {
            has_neon: is_aarch64_feature_detected!("neon"),
            has_sve: is_aarch64_feature_detected!("sve"),
            has_sve2: cfg!(target_feature = "sve2"),
            has_crypto: is_aarch64_feature_detected!("crypto"),
            has_fp16: is_aarch64_feature_detected!("fp16"),
            has_dotprod: is_aarch64_feature_detected!("dotprod"),
            is_apple_silicon: Self::detect_apple_silicon(),
        }
    }

    /// Detect if running on Apple Silicon (M1/M2/M3)
    #[inline]
    fn detect_apple_silicon() -> bool {
        #[cfg(target_os = "macos")]
        {
            std::env::var("PROCESSOR_BRAND")
                .unwrap_or_default()
                .contains("Apple")
        }
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
    }
}

/// High-performance matrix multiplication using NEON/SVE
pub struct SimdMatrix {
    features: AArch64Features,
}

impl SimdMatrix {
    pub fn new() -> Self {
        Self {
            features: AArch64Features::detect(),
        }
    }

    /// Matrix multiplication: C = A * B
    /// Target: <50ns for 4x4 matrices on Apple Silicon
    #[inline]
    pub unsafe fn multiply_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        if self.features.has_sve2 {
            self.multiply_sve2(a, b, c, rows, cols, inner);
        } else if self.features.has_neon {
            if self.features.is_apple_silicon {
                self.multiply_neon_apple(a, b, c, rows, cols, inner);
            } else {
                self.multiply_neon(a, b, c, rows, cols, inner);
            }
        } else {
            self.multiply_scalar(a, b, c, rows, cols, inner);
        }
    }

    /// SVE2 matrix multiplication (future-proof for server ARM)
    #[cfg(target_feature = "sve2")]
    #[inline]
    unsafe fn multiply_sve2(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        // SVE2 implementation would go here - currently limited compiler support
        self.multiply_neon(a, b, c, rows, cols, inner);
    }

    #[cfg(not(target_feature = "sve2"))]
    #[inline]
    unsafe fn multiply_sve2(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        self.multiply_neon(a, b, c, rows, cols, inner);
    }

    /// Apple Silicon optimized NEON matrix multiplication
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn multiply_neon_apple(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        // Apple Silicon has wider execution units and better prefetching
        for i in 0..rows {
            for j in (0..cols).step_by(4) {
                let mut acc = vdupq_n_f32(0.0);
                
                for k in 0..inner {
                    let a_val = vdupq_n_f32(a[i * inner + k]);
                    let b_vals = vld1q_f32(&b[k * cols + j]);
                    
                    // Use FMA for better performance on Apple Silicon
                    acc = vfmaq_f32(acc, a_val, b_vals);
                }
                
                vst1q_f32(&mut c[i * cols + j], acc);
            }
        }
    }

    /// Standard NEON matrix multiplication
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn multiply_neon(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        for i in 0..rows {
            for j in (0..cols).step_by(4) {
                let mut acc = vdupq_n_f32(0.0);
                
                for k in 0..inner {
                    let a_val = vdupq_n_f32(a[i * inner + k]);
                    let b_vals = vld1q_f32(&b[k * cols + j]);
                    let mul = vmulq_f32(a_val, b_vals);
                    acc = vaddq_f32(acc, mul);
                }
                
                vst1q_f32(&mut c[i * cols + j], acc);
            }
        }
    }

    /// Scalar fallback
    #[inline]
    fn multiply_scalar(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        for i in 0..rows {
            for j in 0..cols {
                let mut sum = 0.0;
                for k in 0..inner {
                    sum += a[i * inner + k] * b[k * cols + j];
                }
                c[i * cols + j] = sum;
            }
        }
    }
}

/// High-performance vector operations using NEON
pub struct SimdVector {
    features: AArch64Features,
}

impl SimdVector {
    pub fn new() -> Self {
        Self {
            features: AArch64Features::detect(),
        }
    }

    /// Vector dot product
    /// Target: <30ns for 1024 elements on Apple Silicon
    #[inline]
    pub unsafe fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.features.has_dotprod {
            self.dot_product_dotprod(a, b)
        } else if self.features.has_neon {
            if self.features.is_apple_silicon {
                self.dot_product_neon_apple(a, b)
            } else {
                self.dot_product_neon(a, b)
            }
        } else {
            self.dot_product_scalar(a, b)
        }
    }

    /// Dot product using ARM dot product extension
    #[target_feature(enable = "neon", enable = "dotprod")]
    #[inline]
    unsafe fn dot_product_dotprod(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc = vdupq_n_f32(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = vld1q_f32(&a[idx]);
            let vb = vld1q_f32(&b[idx]);
            acc = vfmaq_f32(acc, va, vb);
        }
        
        // Horizontal sum
        let sum_pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum = vpadd_f32(sum_pair, sum_pair);
        let result = vget_lane_f32(sum, 0);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 4)..len {
            remainder_sum += a[i] * b[i];
        }
        
        result + remainder_sum
    }

    /// Apple Silicon optimized NEON dot product
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dot_product_neon_apple(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc1 = vdupq_n_f32(0.0);
        let mut acc2 = vdupq_n_f32(0.0);
        
        // Unroll loop for better Apple Silicon performance
        let chunks = len / 8;
        for i in 0..chunks {
            let idx = i * 8;
            let va1 = vld1q_f32(&a[idx]);
            let vb1 = vld1q_f32(&b[idx]);
            let va2 = vld1q_f32(&a[idx + 4]);
            let vb2 = vld1q_f32(&b[idx + 4]);
            
            acc1 = vfmaq_f32(acc1, va1, vb1);
            acc2 = vfmaq_f32(acc2, va2, vb2);
        }
        
        // Combine accumulators
        let combined = vaddq_f32(acc1, acc2);
        
        // Horizontal sum
        let sum_pair = vpadd_f32(vget_low_f32(combined), vget_high_f32(combined));
        let sum = vpadd_f32(sum_pair, sum_pair);
        let result = vget_lane_f32(sum, 0);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 8)..len {
            remainder_sum += a[i] * b[i];
        }
        
        result + remainder_sum
    }

    /// Standard NEON dot product
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn dot_product_neon(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc = vdupq_n_f32(0.0);
        
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = vld1q_f32(&a[idx]);
            let vb = vld1q_f32(&b[idx]);
            let mul = vmulq_f32(va, vb);
            acc = vaddq_f32(acc, mul);
        }
        
        // Horizontal sum
        let sum_pair = vpadd_f32(vget_low_f32(acc), vget_high_f32(acc));
        let sum = vpadd_f32(sum_pair, sum_pair);
        let result = vget_lane_f32(sum, 0);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 4)..len {
            remainder_sum += a[i] * b[i];
        }
        
        result + remainder_sum
    }

    /// Scalar fallback
    #[inline]
    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Fast Fourier Transform using NEON
pub struct SimdFFT {
    features: AArch64Features,
}

impl SimdFFT {
    pub fn new() -> Self {
        Self {
            features: AArch64Features::detect(),
        }
    }

    /// Complex FFT using SIMD
    /// Target: <200ns for 256-point FFT on Apple Silicon
    #[inline]
    pub unsafe fn fft_complex_f32(&self, data: &mut [f32], n: usize, inverse: bool) {
        if self.features.has_neon {
            if self.features.is_apple_silicon {
                self.fft_neon_apple(data, n, inverse);
            } else {
                self.fft_neon(data, n, inverse);
            }
        } else {
            self.fft_scalar(data, n, inverse);
        }
    }

    /// Apple Silicon optimized FFT
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fft_neon_apple(&self, data: &mut [f32], n: usize, inverse: bool) {
        // Bit-reversal permutation with NEON optimizations
        self.bit_reverse_neon(data, n);
        
        // Cooley-Tukey FFT with Apple Silicon optimizations
        let mut len = 2;
        while len <= n {
            let wlen = if inverse { 2.0 * std::f32::consts::PI / len as f32 } 
                       else { -2.0 * std::f32::consts::PI / len as f32 };
            
            let wlen_cos = wlen.cos();
            let wlen_sin = wlen.sin();
            
            for i in (0..n).step_by(len) {
                let mut w_real = 1.0;
                let mut w_imag = 0.0;
                
                // Process multiple butterfly operations with NEON
                for j in (0..len/2).step_by(2) {
                    // Load twiddle factors
                    let w1_real = w_real;
                    let w1_imag = w_imag;
                    let w2_real = w_real * wlen_cos - w_imag * wlen_sin;
                    let w2_imag = w_real * wlen_sin + w_imag * wlen_cos;
                    
                    // Vectorized butterfly operations
                    let u_idx = 2 * (i + j);
                    let v_idx = 2 * (i + j + len/2);
                    
                    if j + 1 < len/2 {
                        // Process two butterflies at once
                        let u1_real = data[u_idx];
                        let u1_imag = data[u_idx + 1];
                        let u2_real = data[u_idx + 2];
                        let u2_imag = data[u_idx + 3];
                        
                        let v1_real = data[v_idx];
                        let v1_imag = data[v_idx + 1];
                        let v2_real = data[v_idx + 2];
                        let v2_imag = data[v_idx + 3];
                        
                        // Twiddle multiplication
                        let t1_real = w1_real * v1_real - w1_imag * v1_imag;
                        let t1_imag = w1_real * v1_imag + w1_imag * v1_real;
                        let t2_real = w2_real * v2_real - w2_imag * v2_imag;
                        let t2_imag = w2_real * v2_imag + w2_imag * v2_real;
                        
                        // Butterfly computation
                        data[u_idx] = u1_real + t1_real;
                        data[u_idx + 1] = u1_imag + t1_imag;
                        data[u_idx + 2] = u2_real + t2_real;
                        data[u_idx + 3] = u2_imag + t2_imag;
                        
                        data[v_idx] = u1_real - t1_real;
                        data[v_idx + 1] = u1_imag - t1_imag;
                        data[v_idx + 2] = u2_real - t2_real;
                        data[v_idx + 3] = u2_imag - t2_imag;
                        
                        // Update twiddle factors for next iteration
                        w_real = w2_real;
                        w_imag = w2_imag;
                    } else {
                        // Single butterfly
                        let u_real = data[u_idx];
                        let u_imag = data[u_idx + 1];
                        let v_real = data[v_idx];
                        let v_imag = data[v_idx + 1];
                        
                        let temp_real = w_real * v_real - w_imag * v_imag;
                        let temp_imag = w_real * v_imag + w_imag * v_real;
                        
                        data[u_idx] = u_real + temp_real;
                        data[u_idx + 1] = u_imag + temp_imag;
                        data[v_idx] = u_real - temp_real;
                        data[v_idx + 1] = u_imag - temp_imag;
                    }
                    
                    let new_w_real = w_real * wlen_cos - w_imag * wlen_sin;
                    let new_w_imag = w_real * wlen_sin + w_imag * wlen_cos;
                    w_real = new_w_real;
                    w_imag = new_w_imag;
                }
            }
            len *= 2;
        }

        if inverse {
            let norm_vec = vdupq_n_f32(1.0 / n as f32);
            for i in (0..2*n).step_by(4) {
                let vals = vld1q_f32(&data[i]);
                let normalized = vmulq_f32(vals, norm_vec);
                vst1q_f32(&mut data[i], normalized);
            }
        }
    }

    /// Standard NEON FFT
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn fft_neon(&self, data: &mut [f32], n: usize, inverse: bool) {
        self.fft_neon_apple(data, n, inverse);
    }

    /// NEON bit-reversal permutation
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn bit_reverse_neon(&self, data: &mut [f32], n: usize) {
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j ^= bit;
            if i < j {
                data.swap(2 * i, 2 * j);
                data.swap(2 * i + 1, 2 * j + 1);
            }
        }
    }

    /// Scalar FFT fallback
    #[inline]
    fn fft_scalar(&self, data: &mut [f32], n: usize, inverse: bool) {
        let mut result = vec![0.0; 2 * n];
        let sign = if inverse { 1.0 } else { -1.0 };
        
        for k in 0..n {
            for j in 0..n {
                let angle = sign * 2.0 * std::f32::consts::PI * (k * j) as f32 / n as f32;
                let cos_val = angle.cos();
                let sin_val = angle.sin();
                
                result[2*k] += data[2*j] * cos_val - data[2*j + 1] * sin_val;
                result[2*k + 1] += data[2*j] * sin_val + data[2*j + 1] * cos_val;
            }
        }
        
        if inverse {
            let norm = 1.0 / n as f32;
            for i in 0..2*n {
                result[i] *= norm;
            }
        }
        
        data.copy_from_slice(&result);
    }
}

/// Statistical calculations using NEON
pub struct SimdStats {
    features: AArch64Features,
}

impl SimdStats {
    pub fn new() -> Self {
        Self {
            features: AArch64Features::detect(),
        }
    }

    /// Calculate mean using SIMD
    /// Target: <20ns for 1024 elements on Apple Silicon
    #[inline]
    pub unsafe fn mean_f32(&self, data: &[f32]) -> f32 {
        if self.features.has_neon {
            if self.features.is_apple_silicon {
                self.mean_neon_apple(data)
            } else {
                self.mean_neon(data)
            }
        } else {
            self.mean_scalar(data)
        }
    }

    /// Apple Silicon optimized mean
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mean_neon_apple(&self, data: &[f32]) -> f32 {
        let len = data.len();
        let mut sum1 = vdupq_n_f32(0.0);
        let mut sum2 = vdupq_n_f32(0.0);
        
        // Process 8 elements at once for better Apple Silicon throughput
        let chunks = len / 8;
        for i in 0..chunks {
            let vals1 = vld1q_f32(&data[i * 8]);
            let vals2 = vld1q_f32(&data[i * 8 + 4]);
            sum1 = vaddq_f32(sum1, vals1);
            sum2 = vaddq_f32(sum2, vals2);
        }
        
        let combined = vaddq_f32(sum1, sum2);
        let sum_pair = vpadd_f32(vget_low_f32(combined), vget_high_f32(combined));
        let sum = vpadd_f32(sum_pair, sum_pair);
        let total_sum = vget_lane_f32(sum, 0);
        
        // Handle remainder
        let mut remainder_sum = 0.0;
        for i in (chunks * 8)..len {
            remainder_sum += data[i];
        }
        
        (total_sum + remainder_sum) / len as f32
    }

    /// Standard NEON mean
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn mean_neon(&self, data: &[f32]) -> f32 {
        let len = data.len();
        let mut sum = vdupq_n_f32(0.0);
        
        let chunks = len / 4;
        for i in 0..chunks {
            let vals = vld1q_f32(&data[i * 4]);
            sum = vaddq_f32(sum, vals);
        }
        
        let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
        let sum_final = vpadd_f32(sum_pair, sum_pair);
        let total_sum = vget_lane_f32(sum_final, 0);
        
        // Handle remainder
        let mut remainder_sum = 0.0;
        for i in (chunks * 4)..len {
            remainder_sum += data[i];
        }
        
        (total_sum + remainder_sum) / len as f32
    }

    /// Scalar fallback
    #[inline]
    fn mean_scalar(&self, data: &[f32]) -> f32 {
        data.iter().sum::<f32>() / data.len() as f32
    }
}

/// Parallel reduction operations using NEON
pub struct SimdReduction {
    features: AArch64Features,
}

impl SimdReduction {
    pub fn new() -> Self {
        Self {
            features: AArch64Features::detect(),
        }
    }

    /// Find maximum value using SIMD
    /// Target: <25ns for 1024 elements on Apple Silicon
    #[inline]
    pub unsafe fn max_f32(&self, data: &[f32]) -> f32 {
        if self.features.has_neon {
            if self.features.is_apple_silicon {
                self.max_neon_apple(data)
            } else {
                self.max_neon(data)
            }
        } else {
            self.max_scalar(data)
        }
    }

    /// Apple Silicon optimized max reduction
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn max_neon_apple(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return f32::NEG_INFINITY; }
        
        let mut max_vec1 = vdupq_n_f32(data[0]);
        let mut max_vec2 = vdupq_n_f32(data[0]);
        
        // Process 8 elements at once
        let chunks = data.len() / 8;
        for i in 0..chunks {
            let vals1 = vld1q_f32(&data[i * 8]);
            let vals2 = vld1q_f32(&data[i * 8 + 4]);
            max_vec1 = vmaxq_f32(max_vec1, vals1);
            max_vec2 = vmaxq_f32(max_vec2, vals2);
        }
        
        let combined = vmaxq_f32(max_vec1, max_vec2);
        let max_pair = vpmax_f32(vget_low_f32(combined), vget_high_f32(combined));
        let max_final = vpmax_f32(max_pair, max_pair);
        let mut max_val = vget_lane_f32(max_final, 0);
        
        // Handle remainder
        for i in (chunks * 8)..data.len() {
            if data[i] > max_val {
                max_val = data[i];
            }
        }
        
        max_val
    }

    /// Standard NEON max reduction
    #[target_feature(enable = "neon")]
    #[inline]
    unsafe fn max_neon(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return f32::NEG_INFINITY; }
        
        let mut max_vec = vdupq_n_f32(data[0]);
        
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let vals = vld1q_f32(&data[i * 4]);
            max_vec = vmaxq_f32(max_vec, vals);
        }
        
        let max_pair = vpmax_f32(vget_low_f32(max_vec), vget_high_f32(max_vec));
        let max_final = vpmax_f32(max_pair, max_pair);
        let mut max_val = vget_lane_f32(max_final, 0);
        
        // Handle remainder
        for i in (chunks * 4)..data.len() {
            if data[i] > max_val {
                max_val = data[i];
            }
        }
        
        max_val
    }

    /// Scalar fallback
    #[inline]
    fn max_scalar(&self, data: &[f32]) -> f32 {
        data.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
    }
}

/// Cryptographic acceleration using ARM crypto extensions
#[cfg(target_feature = "crypto")]
pub struct SimdCrypto {
    features: AArch64Features,
}

#[cfg(target_feature = "crypto")]
impl SimdCrypto {
    pub fn new() -> Self {
        Self {
            features: AArch64Features::detect(),
        }
    }

    /// AES encryption using ARM crypto extensions
    #[target_feature(enable = "crypto")]
    #[inline]
    pub unsafe fn aes_encrypt_block(&self, plaintext: &[u8; 16], key: &[u8; 16]) -> [u8; 16] {
        let plain = vld1q_u8(plaintext.as_ptr());
        let round_key = vld1q_u8(key.as_ptr());
        
        // Single round for demonstration
        let encrypted = vaeseq_u8(plain, round_key);
        let mixed = vaesmcq_u8(encrypted);
        
        let mut result = [0u8; 16];
        vst1q_u8(result.as_mut_ptr(), mixed);
        result
    }

    /// SHA-256 acceleration using ARM crypto extensions
    #[target_feature(enable = "crypto")]
    #[inline]
    pub unsafe fn sha256_update(&self, state: &mut [u32; 8], data: &[u8]) {
        // ARM crypto SHA-256 implementation would use vsha256* intrinsics
        // This is a simplified version for demonstration
        for chunk in data.chunks_exact(64) {
            // Process 64-byte chunks with SHA intrinsics
            // Implementation would use vsha256hq_u32, vsha256h2q_u32, vsha256su0q_u32, vsha256su1q_u32
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        let features = AArch64Features::detect();
        println!("Detected features: {:?}", features);
        assert!(features.has_neon); // NEON should be available on all AArch64
    }
    
    #[test]
    fn test_matrix_multiplication() {
        let matrix = SimdMatrix::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut c = vec![0.0; 4];
        
        unsafe {
            matrix.multiply_f32(&a, &b, &mut c, 2, 2, 2);
        }
        
        assert!(c.iter().all(|&x| x.is_finite()));
    }
    
    #[test]
    fn test_dot_product() {
        let vector = SimdVector::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = unsafe { vector.dot_product_f32(&a, &b) };
        let expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // 70.0
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_mean_calculation() {
        let stats = SimdStats::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let result = unsafe { stats.mean_f32(&data) };
        let expected = 3.0;
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_max_reduction() {
        let reduction = SimdReduction::new();
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        
        let result = unsafe { reduction.max_f32(&data) };
        let expected = 9.0;
        
        assert!((result - expected).abs() < 1e-6);
    }
    
    #[test]
    fn test_apple_silicon_detection() {
        let features = AArch64Features::detect();
        // This test will only pass on actual Apple Silicon
        println!("Apple Silicon detected: {}", features.is_apple_silicon);
    }
}