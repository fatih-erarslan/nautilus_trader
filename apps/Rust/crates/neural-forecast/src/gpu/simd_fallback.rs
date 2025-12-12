//! SIMD optimized CPU fallback implementations
//!
//! This module provides vectorized CPU implementations for when GPU is not available,
//! using platform-specific SIMD instructions for maximum performance.

use std::arch::x86_64::*;
use bytemuck::{Pod, Zeroable};
use crate::{Result, NeuralForecastError};

/// SIMD configuration for optimal performance
#[derive(Debug, Clone)]
pub struct SIMDConfig {
    pub use_avx512: bool,
    pub use_avx2: bool,
    pub use_sse: bool,
    pub use_neon: bool,
    pub vector_width: usize,
}

impl Default for SIMDConfig {
    fn default() -> Self {
        Self {
            use_avx512: is_x86_feature_detected!("avx512f"),
            use_avx2: is_x86_feature_detected!("avx2"),
            use_sse: is_x86_feature_detected!("sse4.1"),
            use_neon: cfg!(target_arch = "aarch64"),
            vector_width: Self::detect_vector_width(),
        }
    }
}

impl SIMDConfig {
    /// Detect optimal vector width for the platform
    fn detect_vector_width() -> usize {
        if is_x86_feature_detected!("avx512f") {
            64 // 512 bits / 8 bits per byte
        } else if is_x86_feature_detected!("avx2") {
            32 // 256 bits / 8 bits per byte
        } else if is_x86_feature_detected!("sse4.1") {
            16 // 128 bits / 8 bits per byte
        } else {
            8  // Fallback to 64-bit operations
        }
    }
}

/// SIMD accelerated matrix operations
pub struct SIMDMatrixOps {
    config: SIMDConfig,
}

impl SIMDMatrixOps {
    /// Create new SIMD matrix operations
    pub fn new() -> Self {
        Self {
            config: SIMDConfig::default(),
        }
    }

    /// Matrix multiplication with SIMD optimization
    pub fn matmul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(NeuralForecastError::ComputeError("Matrix dimensions mismatch".to_string()));
        }

        if self.config.use_avx512 {
            unsafe { self.matmul_avx512(a, b, c, m, n, k) }
        } else if self.config.use_avx2 {
            unsafe { self.matmul_avx2(a, b, c, m, n, k) }
        } else if self.config.use_sse {
            unsafe { self.matmul_sse(a, b, c, m, n, k) }
        } else {
            self.matmul_scalar(a, b, c, m, n, k)
        }
    }

    /// AVX-512 optimized matrix multiplication
    #[target_feature(enable = "avx512f")]
    unsafe fn matmul_avx512(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
        const BLOCK_SIZE: usize = 64;
        
        // Tiled matrix multiplication with AVX-512
        for ii in (0..m).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                for kk in (0..k).step_by(BLOCK_SIZE) {
                    let i_max = std::cmp::min(ii + BLOCK_SIZE, m);
                    let j_max = std::cmp::min(jj + BLOCK_SIZE, n);
                    let k_max = std::cmp::min(kk + BLOCK_SIZE, k);
                    
                    for i in ii..i_max {
                        for j in (jj..j_max).step_by(16) {
                            let mut sum = _mm512_setzero_ps();
                            
                            for l in kk..k_max {
                                let a_val = _mm512_set1_ps(a[i * k + l]);
                                let b_vals = _mm512_loadu_ps(&b[l * n + j]);
                                sum = _mm512_fmadd_ps(a_val, b_vals, sum);
                            }
                            
                            let c_vals = _mm512_loadu_ps(&c[i * n + j]);
                            let result = _mm512_add_ps(c_vals, sum);
                            _mm512_storeu_ps(&mut c[i * n + j], result);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// AVX2 optimized matrix multiplication
    #[target_feature(enable = "avx2")]
    unsafe fn matmul_avx2(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
        const BLOCK_SIZE: usize = 32;
        
        for ii in (0..m).step_by(BLOCK_SIZE) {
            for jj in (0..n).step_by(BLOCK_SIZE) {
                for kk in (0..k).step_by(BLOCK_SIZE) {
                    let i_max = std::cmp::min(ii + BLOCK_SIZE, m);
                    let j_max = std::cmp::min(jj + BLOCK_SIZE, n);
                    let k_max = std::cmp::min(kk + BLOCK_SIZE, k);
                    
                    for i in ii..i_max {
                        for j in (jj..j_max).step_by(8) {
                            let mut sum = _mm256_setzero_ps();
                            
                            for l in kk..k_max {
                                let a_val = _mm256_set1_ps(a[i * k + l]);
                                let b_vals = _mm256_loadu_ps(&b[l * n + j]);
                                sum = _mm256_fmadd_ps(a_val, b_vals, sum);
                            }
                            
                            let c_vals = _mm256_loadu_ps(&c[i * n + j]);
                            let result = _mm256_add_ps(c_vals, sum);
                            _mm256_storeu_ps(&mut c[i * n + j], result);
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// SSE optimized matrix multiplication
    #[target_feature(enable = "sse4.1")]
    unsafe fn matmul_sse(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut sum = _mm_setzero_ps();
                
                for l in 0..k {
                    let a_val = _mm_set1_ps(a[i * k + l]);
                    let b_vals = _mm_loadu_ps(&b[l * n + j]);
                    sum = _mm_add_ps(sum, _mm_mul_ps(a_val, b_vals));
                }
                
                let c_vals = _mm_loadu_ps(&c[i * n + j]);
                let result = _mm_add_ps(c_vals, sum);
                _mm_storeu_ps(&mut c[i * n + j], result);
            }
        }
        
        Ok(())
    }

    /// Scalar fallback matrix multiplication
    fn matmul_scalar(&self, a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) -> Result<()> {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] += sum;
            }
        }
        
        Ok(())
    }

    /// Vectorized activation functions
    pub fn relu_f32(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(NeuralForecastError::ComputeError("Input and output length mismatch".to_string()));
        }

        if self.config.use_avx512 {
            unsafe { self.relu_avx512(input, output) }
        } else if self.config.use_avx2 {
            unsafe { self.relu_avx2(input, output) }
        } else if self.config.use_sse {
            unsafe { self.relu_sse(input, output) }
        } else {
            self.relu_scalar(input, output)
        }
    }

    /// AVX-512 ReLU implementation
    #[target_feature(enable = "avx512f")]
    unsafe fn relu_avx512(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let zero = _mm512_setzero_ps();
        let chunks = input.len() / 16;
        let remainder = input.len() % 16;
        
        for i in 0..chunks {
            let offset = i * 16;
            let vals = _mm512_loadu_ps(&input[offset]);
            let result = _mm512_max_ps(vals, zero);
            _mm512_storeu_ps(&mut output[offset], result);
        }
        
        // Handle remainder
        for i in (chunks * 16)..(chunks * 16 + remainder) {
            output[i] = input[i].max(0.0);
        }
        
        Ok(())
    }

    /// AVX2 ReLU implementation
    #[target_feature(enable = "avx2")]
    unsafe fn relu_avx2(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let zero = _mm256_setzero_ps();
        let chunks = input.len() / 8;
        let remainder = input.len() % 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let vals = _mm256_loadu_ps(&input[offset]);
            let result = _mm256_max_ps(vals, zero);
            _mm256_storeu_ps(&mut output[offset], result);
        }
        
        // Handle remainder
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            output[i] = input[i].max(0.0);
        }
        
        Ok(())
    }

    /// SSE ReLU implementation
    #[target_feature(enable = "sse4.1")]
    unsafe fn relu_sse(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        let zero = _mm_setzero_ps();
        let chunks = input.len() / 4;
        let remainder = input.len() % 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            let vals = _mm_loadu_ps(&input[offset]);
            let result = _mm_max_ps(vals, zero);
            _mm_storeu_ps(&mut output[offset], result);
        }
        
        // Handle remainder
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            output[i] = input[i].max(0.0);
        }
        
        Ok(())
    }

    /// Scalar ReLU implementation
    fn relu_scalar(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        for i in 0..input.len() {
            output[i] = input[i].max(0.0);
        }
        Ok(())
    }

    /// Vectorized softmax with numerical stability
    pub fn softmax_f32(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(NeuralForecastError::ComputeError("Input and output length mismatch".to_string()));
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Compute exp(x - max) and sum
        let mut sum = 0.0;
        for i in 0..input.len() {
            let exp_val = (input[i] - max_val).exp();
            output[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        let inv_sum = 1.0 / sum;
        if self.config.use_avx512 {
            unsafe { self.scale_avx512(output, inv_sum) }
        } else if self.config.use_avx2 {
            unsafe { self.scale_avx2(output, inv_sum) }
        } else if self.config.use_sse {
            unsafe { self.scale_sse(output, inv_sum) }
        } else {
            self.scale_scalar(output, inv_sum)
        }
    }

    /// AVX-512 scaling
    #[target_feature(enable = "avx512f")]
    unsafe fn scale_avx512(&self, values: &mut [f32], scale: f32) -> Result<()> {
        let scale_vec = _mm512_set1_ps(scale);
        let chunks = values.len() / 16;
        let remainder = values.len() % 16;
        
        for i in 0..chunks {
            let offset = i * 16;
            let vals = _mm512_loadu_ps(&values[offset]);
            let result = _mm512_mul_ps(vals, scale_vec);
            _mm512_storeu_ps(&mut values[offset], result);
        }
        
        // Handle remainder
        for i in (chunks * 16)..(chunks * 16 + remainder) {
            values[i] *= scale;
        }
        
        Ok(())
    }

    /// AVX2 scaling
    #[target_feature(enable = "avx2")]
    unsafe fn scale_avx2(&self, values: &mut [f32], scale: f32) -> Result<()> {
        let scale_vec = _mm256_set1_ps(scale);
        let chunks = values.len() / 8;
        let remainder = values.len() % 8;
        
        for i in 0..chunks {
            let offset = i * 8;
            let vals = _mm256_loadu_ps(&values[offset]);
            let result = _mm256_mul_ps(vals, scale_vec);
            _mm256_storeu_ps(&mut values[offset], result);
        }
        
        // Handle remainder
        for i in (chunks * 8)..(chunks * 8 + remainder) {
            values[i] *= scale;
        }
        
        Ok(())
    }

    /// SSE scaling
    #[target_feature(enable = "sse4.1")]
    unsafe fn scale_sse(&self, values: &mut [f32], scale: f32) -> Result<()> {
        let scale_vec = _mm_set1_ps(scale);
        let chunks = values.len() / 4;
        let remainder = values.len() % 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            let vals = _mm_loadu_ps(&values[offset]);
            let result = _mm_mul_ps(vals, scale_vec);
            _mm_storeu_ps(&mut values[offset], result);
        }
        
        // Handle remainder
        for i in (chunks * 4)..(chunks * 4 + remainder) {
            values[i] *= scale;
        }
        
        Ok(())
    }

    /// Scalar scaling
    fn scale_scalar(&self, values: &mut [f32], scale: f32) -> Result<()> {
        for value in values {
            *value *= scale;
        }
        Ok(())
    }

    /// Vectorized layer normalization
    pub fn layer_norm_f32(&self, input: &[f32], output: &mut [f32], gamma: &[f32], beta: &[f32], eps: f32) -> Result<()> {
        if input.len() != output.len() || input.len() != gamma.len() || input.len() != beta.len() {
            return Err(NeuralForecastError::ComputeError("Array length mismatch".to_string()));
        }

        // Compute mean
        let mean = input.iter().sum::<f32>() / input.len() as f32;
        
        // Compute variance
        let variance = input.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f32>() / input.len() as f32;
        
        let inv_std = 1.0 / (variance + eps).sqrt();
        
        // Normalize and apply scale/shift
        for i in 0..input.len() {
            let normalized = (input[i] - mean) * inv_std;
            output[i] = gamma[i] * normalized + beta[i];
        }
        
        Ok(())
    }

    /// Vectorized batch normalization
    pub fn batch_norm_f32(&self, 
        input: &[f32], 
        output: &mut [f32], 
        mean: &[f32], 
        variance: &[f32],
        gamma: &[f32], 
        beta: &[f32], 
        eps: f32,
        batch_size: usize,
        channels: usize
    ) -> Result<()> {
        if input.len() != output.len() || input.len() != batch_size * channels {
            return Err(NeuralForecastError::ComputeError("Array length mismatch".to_string()));
        }

        for b in 0..batch_size {
            for c in 0..channels {
                let idx = b * channels + c;
                let normalized = (input[idx] - mean[c]) / (variance[c] + eps).sqrt();
                output[idx] = gamma[c] * normalized + beta[c];
            }
        }
        
        Ok(())
    }

    /// Get performance characteristics
    pub fn get_performance_info(&self) -> SIMDPerformanceInfo {
        SIMDPerformanceInfo {
            config: self.config.clone(),
            theoretical_throughput: self.calculate_theoretical_throughput(),
            cache_line_size: self.get_cache_line_size(),
            preferred_alignment: self.get_preferred_alignment(),
        }
    }

    /// Calculate theoretical throughput
    fn calculate_theoretical_throughput(&self) -> f64 {
        let base_frequency = 3.0e9; // 3 GHz base frequency
        let vector_width = self.config.vector_width as f64;
        let ops_per_cycle = if self.config.use_avx512 {
            2.0 // Two FMA units
        } else if self.config.use_avx2 {
            2.0 // Two FMA units
        } else {
            1.0 // One FMA unit
        };
        
        // Operations per second = frequency * vector_width * ops_per_cycle
        base_frequency * vector_width * ops_per_cycle / 4.0 // Divide by 4 for FP32
    }

    /// Get cache line size
    fn get_cache_line_size(&self) -> usize {
        64 // Standard cache line size for modern x86 processors
    }

    /// Get preferred memory alignment
    fn get_preferred_alignment(&self) -> usize {
        if self.config.use_avx512 {
            64 // 512-bit alignment
        } else if self.config.use_avx2 {
            32 // 256-bit alignment
        } else {
            16 // 128-bit alignment
        }
    }
}

/// SIMD performance information
#[derive(Debug, Clone)]
pub struct SIMDPerformanceInfo {
    pub config: SIMDConfig,
    pub theoretical_throughput: f64,
    pub cache_line_size: usize,
    pub preferred_alignment: usize,
}

/// SIMD benchmarking utilities
pub mod benchmarks {
    use super::*;
    use std::time::Instant;

    /// Benchmark matrix multiplication performance
    pub fn benchmark_matmul(size: usize, iterations: usize) -> Result<SIMDBenchmarkResult> {
        let simd_ops = SIMDMatrixOps::new();
        
        let a = vec![1.0f32; size * size];
        let b = vec![2.0f32; size * size];
        let mut c = vec![0.0f32; size * size];
        
        let start = Instant::now();
        
        for _ in 0..iterations {
            simd_ops.matmul_f32(&a, &b, &mut c, size, size, size)?;
        }
        
        let duration = start.elapsed();
        let ops = (size * size * size * 2) as f64 * iterations as f64; // 2 ops per multiply-add
        let throughput = ops / duration.as_secs_f64();
        
        Ok(SIMDBenchmarkResult {
            operation: "matmul".to_string(),
            size,
            iterations,
            duration,
            throughput,
            efficiency: throughput / simd_ops.calculate_theoretical_throughput(),
        })
    }

    /// Benchmark activation function performance
    pub fn benchmark_activation(size: usize, iterations: usize) -> Result<SIMDBenchmarkResult> {
        let simd_ops = SIMDMatrixOps::new();
        
        let input = vec![1.0f32; size];
        let mut output = vec![0.0f32; size];
        
        let start = Instant::now();
        
        for _ in 0..iterations {
            simd_ops.relu_f32(&input, &mut output)?;
        }
        
        let duration = start.elapsed();
        let ops = size as f64 * iterations as f64;
        let throughput = ops / duration.as_secs_f64();
        
        Ok(SIMDBenchmarkResult {
            operation: "relu".to_string(),
            size,
            iterations,
            duration,
            throughput,
            efficiency: throughput / simd_ops.calculate_theoretical_throughput(),
        })
    }
}

/// SIMD benchmark result
#[derive(Debug, Clone)]
pub struct SIMDBenchmarkResult {
    pub operation: String,
    pub size: usize,
    pub iterations: usize,
    pub duration: std::time::Duration,
    pub throughput: f64,
    pub efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_matmul() {
        let simd_ops = SIMDMatrixOps::new();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];  // 2x2 matrix
        let b = vec![5.0, 6.0, 7.0, 8.0];  // 2x2 matrix
        let mut c = vec![0.0; 4];           // 2x2 result
        
        simd_ops.matmul_f32(&a, &b, &mut c, 2, 2, 2).unwrap();
        
        // Expected: [[19, 22], [43, 50]]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_simd_relu() {
        let simd_ops = SIMDMatrixOps::new();
        
        let input = vec![-1.0, 0.0, 1.0, -2.0, 3.0];
        let mut output = vec![0.0; 5];
        
        simd_ops.relu_f32(&input, &mut output).unwrap();
        
        assert_eq!(output, vec![0.0, 0.0, 1.0, 0.0, 3.0]);
    }

    #[test]
    fn test_simd_softmax() {
        let simd_ops = SIMDMatrixOps::new();
        
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];
        
        simd_ops.softmax_f32(&input, &mut output).unwrap();
        
        let sum: f32 = output.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(output[2] > output[1] && output[1] > output[0]);
    }

    #[test]
    fn test_simd_config() {
        let config = SIMDConfig::default();
        
        assert!(config.vector_width > 0);
        
        #[cfg(target_arch = "x86_64")]
        {
            assert!(config.use_sse || config.use_avx2 || config.use_avx512);
        }
    }

    #[test]
    fn test_performance_info() {
        let simd_ops = SIMDMatrixOps::new();
        let info = simd_ops.get_performance_info();
        
        assert!(info.theoretical_throughput > 0.0);
        assert!(info.cache_line_size > 0);
        assert!(info.preferred_alignment > 0);
    }
}