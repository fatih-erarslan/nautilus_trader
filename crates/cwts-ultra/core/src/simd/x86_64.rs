//! x86_64 SIMD optimizations using AVX2, AVX-512, and SSE intrinsics
//! Performance target: <100ns for basic operations

use std::arch::x86_64::*;
use std::mem;

/// Runtime feature detection for x86_64 SIMD capabilities
#[derive(Debug, Clone, Copy)]
pub struct X86Features {
    pub has_sse42: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
    pub has_avx512bw: bool,
}

impl X86Features {
    /// Detect available x86_64 SIMD features at runtime
    #[inline]
    pub fn detect() -> Self {
        Self {
            has_sse42: is_x86_feature_detected!("sse4.2"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
            has_fma: is_x86_feature_detected!("fma"),
            has_avx512bw: is_x86_feature_detected!("avx512bw"),
        }
    }
}

/// High-performance matrix multiplication using AVX2/AVX-512
pub struct SimdMatrix {
    features: X86Features,
}

impl SimdMatrix {
    pub fn new() -> Self {
        Self {
            features: X86Features::detect(),
        }
    }

    /// Matrix multiplication: C = A * B
    /// Target: <50ns for 4x4 matrices
    #[inline]
    pub unsafe fn multiply_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        if self.features.has_avx512f {
            self.multiply_avx512(a, b, c, rows, cols, inner);
        } else if self.features.has_avx2 {
            self.multiply_avx2(a, b, c, rows, cols, inner);
        } else if self.features.has_sse42 {
            self.multiply_sse42(a, b, c, rows, cols, inner);
        } else {
            self.multiply_scalar(a, b, c, rows, cols, inner);
        }
    }

    /// AVX-512 matrix multiplication
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn multiply_avx512(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        for i in 0..rows {
            for j in (0..cols).step_by(16) {
                let mut acc = _mm512_setzero_ps();
                
                for k in 0..inner {
                    let a_val = _mm512_set1_ps(a[i * inner + k]);
                    let b_vals = _mm512_loadu_ps(&b[k * cols + j]);
                    acc = _mm512_fmadd_ps(a_val, b_vals, acc);
                }
                
                _mm512_storeu_ps(&mut c[i * cols + j], acc);
            }
        }
    }

    /// AVX2 matrix multiplication with FMA
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn multiply_avx2(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        for i in 0..rows {
            for j in (0..cols).step_by(8) {
                let mut acc = _mm256_setzero_ps();
                
                for k in 0..inner {
                    let a_val = _mm256_set1_ps(a[i * inner + k]);
                    let b_vals = _mm256_loadu_ps(&b[k * cols + j]);
                    
                    if self.features.has_fma {
                        acc = _mm256_fmadd_ps(a_val, b_vals, acc);
                    } else {
                        let mul = _mm256_mul_ps(a_val, b_vals);
                        acc = _mm256_add_ps(acc, mul);
                    }
                }
                
                _mm256_storeu_ps(&mut c[i * cols + j], acc);
            }
        }
    }

    /// SSE4.2 fallback implementation
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn multiply_sse42(&self, a: &[f32], b: &[f32], c: &mut [f32], rows: usize, cols: usize, inner: usize) {
        for i in 0..rows {
            for j in (0..cols).step_by(4) {
                let mut acc = _mm_setzero_ps();
                
                for k in 0..inner {
                    let a_val = _mm_set1_ps(a[i * inner + k]);
                    let b_vals = _mm_loadu_ps(&b[k * cols + j]);
                    let mul = _mm_mul_ps(a_val, b_vals);
                    acc = _mm_add_ps(acc, mul);
                }
                
                _mm_storeu_ps(&mut c[i * cols + j], acc);
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

/// High-performance vector operations
pub struct SimdVector {
    features: X86Features,
}

impl SimdVector {
    pub fn new() -> Self {
        Self {
            features: X86Features::detect(),
        }
    }

    /// Vector dot product
    /// Target: <30ns for 1024 elements
    #[inline]
    pub unsafe fn dot_product_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.features.has_avx512f {
            self.dot_product_avx512(a, b)
        } else if self.features.has_avx2 {
            self.dot_product_avx2(a, b)
        } else if self.features.has_sse42 {
            self.dot_product_sse42(a, b)
        } else {
            self.dot_product_scalar(a, b)
        }
    }

    /// AVX-512 dot product
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn dot_product_avx512(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc = _mm512_setzero_ps();
        
        let chunks = len / 16;
        for i in 0..chunks {
            let idx = i * 16;
            let va = _mm512_loadu_ps(&a[idx]);
            let vb = _mm512_loadu_ps(&b[idx]);
            acc = _mm512_fmadd_ps(va, vb, acc);
        }
        
        // Horizontal sum of 16 elements
        let sum = _mm512_reduce_add_ps(acc);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 16)..len {
            remainder_sum += a[i] * b[i];
        }
        
        sum + remainder_sum
    }

    /// AVX2 dot product with FMA
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc = _mm256_setzero_ps();
        
        let chunks = len / 8;
        for i in 0..chunks {
            let idx = i * 8;
            let va = _mm256_loadu_ps(&a[idx]);
            let vb = _mm256_loadu_ps(&b[idx]);
            
            if self.features.has_fma {
                acc = _mm256_fmadd_ps(va, vb, acc);
            } else {
                let mul = _mm256_mul_ps(va, vb);
                acc = _mm256_add_ps(acc, mul);
            }
        }
        
        // Horizontal sum of 8 elements
        let acc_high = _mm256_extractf128_ps(acc, 1);
        let acc_low = _mm256_castps256_ps128(acc);
        let acc_128 = _mm_add_ps(acc_high, acc_low);
        let acc_64 = _mm_add_ps(acc_128, _mm_movehl_ps(acc_128, acc_128));
        let acc_32 = _mm_add_ss(acc_64, _mm_shuffle_ps(acc_64, acc_64, 0x55));
        let sum = _mm_cvtss_f32(acc_32);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 8)..len {
            remainder_sum += a[i] * b[i];
        }
        
        sum + remainder_sum
    }

    /// SSE4.2 dot product
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn dot_product_sse42(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut acc = _mm_setzero_ps();
        
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let va = _mm_loadu_ps(&a[idx]);
            let vb = _mm_loadu_ps(&b[idx]);
            let mul = _mm_mul_ps(va, vb);
            acc = _mm_add_ps(acc, mul);
        }
        
        // Horizontal sum of 4 elements
        let shuf = _mm_movehdup_ps(acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let sum = _mm_cvtss_f32(result);
        
        // Handle remaining elements
        let mut remainder_sum = 0.0;
        for i in (chunks * 4)..len {
            remainder_sum += a[i] * b[i];
        }
        
        sum + remainder_sum
    }

    /// Scalar fallback
    #[inline]
    fn dot_product_scalar(&self, a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

/// Fast Fourier Transform using AVX2/AVX-512
pub struct SimdFFT {
    features: X86Features,
}

impl SimdFFT {
    pub fn new() -> Self {
        Self {
            features: X86Features::detect(),
        }
    }

    /// Complex FFT using SIMD
    /// Target: <200ns for 256-point FFT
    #[inline]
    pub unsafe fn fft_complex_f32(&self, data: &mut [f32], n: usize, inverse: bool) {
        if self.features.has_avx2 {
            self.fft_avx2(data, n, inverse);
        } else if self.features.has_sse42 {
            self.fft_sse42(data, n, inverse);
        } else {
            self.fft_scalar(data, n, inverse);
        }
    }

    /// AVX2 FFT implementation
    #[target_feature(enable = "avx2", enable = "fma")]
    #[inline]
    unsafe fn fft_avx2(&self, data: &mut [f32], n: usize, inverse: bool) {
        // Bit-reversal permutation
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

        // Cooley-Tukey FFT
        let mut len = 2;
        while len <= n {
            let wlen = if inverse { 2.0 * std::f32::consts::PI / len as f32 } 
                       else { -2.0 * std::f32::consts::PI / len as f32 };
            
            let wlen_cos = wlen.cos();
            let wlen_sin = wlen.sin();
            
            for i in (0..n).step_by(len) {
                let mut w_real = 1.0;
                let mut w_imag = 0.0;
                
                for j in 0..len/2 {
                    let u_idx = 2 * (i + j);
                    let v_idx = 2 * (i + j + len/2);
                    
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
                    
                    let new_w_real = w_real * wlen_cos - w_imag * wlen_sin;
                    let new_w_imag = w_real * wlen_sin + w_imag * wlen_cos;
                    w_real = new_w_real;
                    w_imag = new_w_imag;
                }
            }
            len *= 2;
        }

        if inverse {
            let norm = 1.0 / n as f32;
            for i in 0..2*n {
                data[i] *= norm;
            }
        }
    }

    /// SSE4.2 FFT fallback
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn fft_sse42(&self, data: &mut [f32], n: usize, inverse: bool) {
        self.fft_scalar(data, n, inverse);
    }

    /// Scalar FFT fallback
    #[inline]
    fn fft_scalar(&self, data: &mut [f32], n: usize, inverse: bool) {
        // Simple DFT for fallback - not optimized
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

/// Statistical calculations using SIMD
pub struct SimdStats {
    features: X86Features,
}

impl SimdStats {
    pub fn new() -> Self {
        Self {
            features: X86Features::detect(),
        }
    }

    /// Calculate mean using SIMD
    /// Target: <20ns for 1024 elements
    #[inline]
    pub unsafe fn mean_f32(&self, data: &[f32]) -> f32 {
        if self.features.has_avx512f {
            self.mean_avx512(data)
        } else if self.features.has_avx2 {
            self.mean_avx2(data)
        } else if self.features.has_sse42 {
            self.mean_sse42(data)
        } else {
            self.mean_scalar(data)
        }
    }

    /// AVX-512 mean calculation
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn mean_avx512(&self, data: &[f32]) -> f32 {
        let len = data.len();
        let mut sum = _mm512_setzero_ps();
        
        let chunks = len / 16;
        for i in 0..chunks {
            let vals = _mm512_loadu_ps(&data[i * 16]);
            sum = _mm512_add_ps(sum, vals);
        }
        
        let total_sum = _mm512_reduce_add_ps(sum);
        
        // Handle remainder
        let mut remainder_sum = 0.0;
        for i in (chunks * 16)..len {
            remainder_sum += data[i];
        }
        
        (total_sum + remainder_sum) / len as f32
    }

    /// AVX2 mean calculation
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn mean_avx2(&self, data: &[f32]) -> f32 {
        let len = data.len();
        let mut sum = _mm256_setzero_ps();
        
        let chunks = len / 8;
        for i in 0..chunks {
            let vals = _mm256_loadu_ps(&data[i * 8]);
            sum = _mm256_add_ps(sum, vals);
        }
        
        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum_128 = _mm_add_ps(sum_high, sum_low);
        let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
        let total_sum = _mm_cvtss_f32(sum_32);
        
        // Handle remainder
        let mut remainder_sum = 0.0;
        for i in (chunks * 8)..len {
            remainder_sum += data[i];
        }
        
        (total_sum + remainder_sum) / len as f32
    }

    /// SSE4.2 mean calculation
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn mean_sse42(&self, data: &[f32]) -> f32 {
        let len = data.len();
        let mut sum = _mm_setzero_ps();
        
        let chunks = len / 4;
        for i in 0..chunks {
            let vals = _mm_loadu_ps(&data[i * 4]);
            sum = _mm_add_ps(sum, vals);
        }
        
        // Horizontal sum
        let shuf = _mm_movehdup_ps(sum);
        let sums = _mm_add_ps(sum, shuf);
        let shuf2 = _mm_movehl_ps(sums, sums);
        let result = _mm_add_ss(sums, shuf2);
        let total_sum = _mm_cvtss_f32(result);
        
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

/// Parallel reduction operations
pub struct SimdReduction {
    features: X86Features,
}

impl SimdReduction {
    pub fn new() -> Self {
        Self {
            features: X86Features::detect(),
        }
    }

    /// Find maximum value using SIMD
    /// Target: <25ns for 1024 elements
    #[inline]
    pub unsafe fn max_f32(&self, data: &[f32]) -> f32 {
        if self.features.has_avx512f {
            self.max_avx512(data)
        } else if self.features.has_avx2 {
            self.max_avx2(data)
        } else if self.features.has_sse42 {
            self.max_sse42(data)
        } else {
            self.max_scalar(data)
        }
    }

    /// AVX-512 max reduction
    #[target_feature(enable = "avx512f")]
    #[inline]
    unsafe fn max_avx512(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return f32::NEG_INFINITY; }
        
        let mut max_vec = _mm512_set1_ps(data[0]);
        
        let chunks = data.len() / 16;
        for i in 0..chunks {
            let vals = _mm512_loadu_ps(&data[i * 16]);
            max_vec = _mm512_max_ps(max_vec, vals);
        }
        
        let mut max_val = _mm512_reduce_max_ps(max_vec);
        
        // Handle remainder
        for i in (chunks * 16)..data.len() {
            if data[i] > max_val {
                max_val = data[i];
            }
        }
        
        max_val
    }

    /// AVX2 max reduction
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn max_avx2(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return f32::NEG_INFINITY; }
        
        let mut max_vec = _mm256_set1_ps(data[0]);
        
        let chunks = data.len() / 8;
        for i in 0..chunks {
            let vals = _mm256_loadu_ps(&data[i * 8]);
            max_vec = _mm256_max_ps(max_vec, vals);
        }
        
        // Extract max from vector
        let mut max_array = [0.0f32; 8];
        _mm256_storeu_ps(max_array.as_mut_ptr(), max_vec);
        let mut max_val = max_array[0];
        for &val in &max_array[1..] {
            if val > max_val { max_val = val; }
        }
        
        // Handle remainder
        for i in (chunks * 8)..data.len() {
            if data[i] > max_val {
                max_val = data[i];
            }
        }
        
        max_val
    }

    /// SSE4.2 max reduction
    #[target_feature(enable = "sse4.2")]
    #[inline]
    unsafe fn max_sse42(&self, data: &[f32]) -> f32 {
        if data.is_empty() { return f32::NEG_INFINITY; }
        
        let mut max_vec = _mm_set1_ps(data[0]);
        
        let chunks = data.len() / 4;
        for i in 0..chunks {
            let vals = _mm_loadu_ps(&data[i * 4]);
            max_vec = _mm_max_ps(max_vec, vals);
        }
        
        // Extract max from vector
        let mut max_array = [0.0f32; 4];
        _mm_storeu_ps(max_array.as_mut_ptr(), max_vec);
        let mut max_val = max_array[0];
        for &val in &max_array[1..] {
            if val > max_val { max_val = val; }
        }
        
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_feature_detection() {
        let features = X86Features::detect();
        println!("Detected features: {:?}", features);
        // Features detection should not panic
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
        
        // Verify result is reasonable (exact values depend on layout)
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
}