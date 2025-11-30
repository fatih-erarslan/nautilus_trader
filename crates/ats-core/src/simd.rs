//! Ultra-High Performance SIMD Mathematical Operations
//!
//! This module implements SIMD-optimized mathematical operations for maximum
//! performance in ATS-CP computations. It leverages modern CPU vector instructions
//! to achieve sub-microsecond latencies for bulk mathematical operations.

use crate::{
    config::{AtsCpConfig, SimdInstructionSet},
    error::{AtsCoreError, Result},
};
use instant::Instant;
use rayon::prelude::*;
use std::arch::x86_64::*;

// AVX-512 intrinsics are only available on nightly Rust
// For stable compilation, we'll use AVX2 and SSE as fallbacks
#[cfg(all(target_arch = "x86_64", target_feature = "avx512f"))]
use std::arch::x86_64::{
    _mm512_loadu_pd, _mm512_storeu_pd, _mm512_add_pd, _mm512_mul_pd,
    _mm512_fmadd_pd, _mm512_setzero_pd, _mm512_reduce_add_pd,
    _mm512_set1_pd, _mm512_sub_pd, _mm512_cmp_pd_mask, _CMP_GT_OQ
};

/// SIMD-optimized mathematical operations engine
pub struct SimdOperations {
    /// Configuration parameters
    config: AtsCpConfig,
    
    /// Available instruction set
    instruction_set: SimdInstructionSet,
    
    /// Vector width for SIMD operations
    vector_width: usize,
    
    /// Alignment requirement
    alignment: usize,
    
    /// Performance counters
    total_operations: u64,
    total_simd_ops: u64,
    total_time_ns: u64,
}

impl SimdOperations {
    /// Creates a new SIMD operations engine with auto-detected capabilities
    pub fn new(config: &AtsCpConfig) -> Result<Self> {
        let instruction_set = match &config.simd.instruction_set {
            SimdInstructionSet::Auto => Self::detect_instruction_set(),
            other => other.clone(),
        };
        
        let vector_width = Self::get_vector_width(&instruction_set);
        let alignment = config.simd.alignment_bytes;
        
        Ok(Self {
            config: config.clone(),
            instruction_set,
            vector_width,
            alignment,
            total_operations: 0,
            total_simd_ops: 0,
            total_time_ns: 0,
        })
    }

    /// Auto-detects the best available SIMD instruction set
    fn detect_instruction_set() -> SimdInstructionSet {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                SimdInstructionSet::Avx512
            } else if is_x86_feature_detected!("avx2") {
                SimdInstructionSet::Avx2
            } else if is_x86_feature_detected!("sse4.2") {
                SimdInstructionSet::Sse42
            } else {
                SimdInstructionSet::Auto
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            if is_aarch64_feature_detected!("neon") {
                SimdInstructionSet::Neon
            } else {
                SimdInstructionSet::Auto
            }
        }
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            SimdInstructionSet::Auto
        }
    }

    /// Gets the vector width for a given instruction set
    fn get_vector_width(instruction_set: &SimdInstructionSet) -> usize {
        match instruction_set {
            SimdInstructionSet::Avx512 => 16, // Phase 1 SIMD: Increased to 16 x f64 for AVX-512
            SimdInstructionSet::Avx2 => 4,    // 4 x f64
            SimdInstructionSet::Sse42 => 2,   // 2 x f64
            SimdInstructionSet::Neon => 2,    // 2 x f64
            SimdInstructionSet::Auto => 4,    // Conservative default
        }
    }

    /// SIMD-optimized vector addition
    pub fn vector_add(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        if a.len() != b.len() {
            return Err(AtsCoreError::dimension_mismatch(b.len(), a.len()));
        }
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        let result = if a.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.vector_add_simd(a, b)?
        } else {
            self.vector_add_scalar(a, b)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD-optimized vector multiplication
    pub fn vector_multiply(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        if a.len() != b.len() {
            return Err(AtsCoreError::dimension_mismatch(b.len(), a.len()));
        }
        
        if a.is_empty() {
            return Ok(Vec::new());
        }
        
        let result = if a.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.vector_multiply_simd(a, b)?
        } else {
            self.vector_multiply_scalar(a, b)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD-optimized scalar multiplication
    pub fn scalar_multiply(&mut self, vector: &[f64], scalar: f64) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        let result = if vector.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.scalar_multiply_simd(vector, scalar)?
        } else {
            self.scalar_multiply_scalar(vector, scalar)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD-optimized dot product
    pub fn dot_product(&mut self, a: &[f64], b: &[f64]) -> Result<f64> {
        let start_time = Instant::now();
        
        if a.len() != b.len() {
            return Err(AtsCoreError::dimension_mismatch(b.len(), a.len()));
        }
        
        if a.is_empty() {
            return Ok(0.0);
        }
        
        let result = if a.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.dot_product_simd(a, b)?
        } else {
            self.dot_product_scalar(a, b)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD-optimized exponential function
    pub fn vector_exp(&mut self, vector: &[f64]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        let result = if vector.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.vector_exp_simd(vector)?
        } else {
            self.vector_exp_scalar(vector)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD-optimized logarithm function
    pub fn vector_log(&mut self, vector: &[f64]) -> Result<Vec<f64>> {
        let start_time = Instant::now();
        
        if vector.is_empty() {
            return Ok(Vec::new());
        }
        
        // Check for non-positive values
        for &val in vector {
            if val <= 0.0 {
                return Err(AtsCoreError::mathematical("vector_log", "non-positive input"));
            }
        }
        
        let result = if vector.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.vector_log_simd(vector)?
        } else {
            self.vector_log_scalar(vector)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD implementation of vector addition
    #[cfg(target_arch = "x86_64")]
    fn vector_add_simd(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![0.0; a.len()];
        let chunks = a.len() / self.vector_width;
        
        unsafe {
            match self.instruction_set {
                SimdInstructionSet::Avx512 => {
                    // Phase 1 SIMD: Enable AVX-512 16-element vector processing
                    if is_x86_feature_detected!("avx512f") {
                        return self.vector_add_avx512_enabled(a, b);
                    } else {
                        return self.vector_add_scalar_fallback(a, b);
                    }
                },
                SimdInstructionSet::Avx2 => {
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm256_add_pd(va, vb);
                        _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                SimdInstructionSet::Sse42 => {
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm_add_pd(va, vb);
                        _mm_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                _ => return self.vector_add_scalar_fallback(a, b),
            }
        }
        
        // Handle remainder elements
        let remainder_start = chunks * self.vector_width;
        for i in remainder_start..a.len() {
            result[i] = a[i] + b[i];
        }
        
        Ok(result)
    }

    /// SIMD implementation of vector multiplication
    #[cfg(target_arch = "x86_64")]
    fn vector_multiply_simd(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![0.0; a.len()];
        let chunks = a.len() / self.vector_width;
        
        unsafe {
            match self.instruction_set {
                SimdInstructionSet::Avx512 => {
                    // Phase 1 SIMD: Enable AVX-512 16-element vector processing
                    if is_x86_feature_detected!("avx512f") {
                        return self.vector_multiply_avx512_enabled(a, b);
                    } else {
                        return self.vector_multiply_scalar_fallback(a, b);
                    }
                },
                SimdInstructionSet::Avx2 => {
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm256_mul_pd(va, vb);
                        _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                SimdInstructionSet::Sse42 => {
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm_mul_pd(va, vb);
                        _mm_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                _ => return self.vector_multiply_scalar_fallback(a, b),
            }
        }
        
        // Handle remainder elements
        let remainder_start = chunks * self.vector_width;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i];
        }
        
        Ok(result)
    }

    /// SIMD implementation of scalar multiplication
    #[cfg(target_arch = "x86_64")]
    fn scalar_multiply_simd(&mut self, vector: &[f64], scalar: f64) -> Result<Vec<f64>> {
        let mut result = vec![0.0; vector.len()];
        let chunks = vector.len() / self.vector_width;
        
        unsafe {
            match self.instruction_set {
                SimdInstructionSet::Avx512 => {
                    // AVX-512 requires nightly features, falling back to AVX2
                    return self.scalar_multiply_scalar_fallback(vector, scalar);
                },
                SimdInstructionSet::Avx2 => {
                    let vs = _mm256_set1_pd(scalar);
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let vv = _mm256_loadu_pd(vector.as_ptr().add(offset));
                        let vr = _mm256_mul_pd(vv, vs);
                        _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                SimdInstructionSet::Sse42 => {
                    let vs = _mm_set1_pd(scalar);
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let vv = _mm_loadu_pd(vector.as_ptr().add(offset));
                        let vr = _mm_mul_pd(vv, vs);
                        _mm_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                _ => return self.scalar_multiply_scalar_fallback(vector, scalar),
            }
        }
        
        // Handle remainder elements
        let remainder_start = chunks * self.vector_width;
        for i in remainder_start..vector.len() {
            result[i] = vector[i] * scalar;
        }
        
        Ok(result)
    }

    /// SIMD implementation of dot product
    #[cfg(target_arch = "x86_64")]
    fn dot_product_simd(&mut self, a: &[f64], b: &[f64]) -> Result<f64> {
        let chunks = a.len() / self.vector_width;
        let mut sum = 0.0;
        
        unsafe {
            match self.instruction_set {
                SimdInstructionSet::Avx512 => {
                    // AVX-512 requires nightly features, falling back to scalar
                    return Ok(self.dot_product_scalar(a, b));
                },
                SimdInstructionSet::Avx2 => {
                    let mut vsum = _mm256_setzero_pd();
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                        let vmul = _mm256_mul_pd(va, vb);
                        vsum = _mm256_add_pd(vsum, vmul);
                        self.total_simd_ops += 1;
                    }
                    sum += self.horizontal_sum_avx2(vsum);
                },
                SimdInstructionSet::Sse42 => {
                    let mut vsum = _mm_setzero_pd();
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm_loadu_pd(b.as_ptr().add(offset));
                        let vmul = _mm_mul_pd(va, vb);
                        vsum = _mm_add_pd(vsum, vmul);
                        self.total_simd_ops += 1;
                    }
                    sum += self.horizontal_sum_sse(vsum);
                },
                _ => return Ok(self.dot_product_scalar(a, b)),
            }
        }
        
        // Handle remainder elements
        let remainder_start = chunks * self.vector_width;
        for i in remainder_start..a.len() {
            sum += a[i] * b[i];
        }
        
        Ok(sum)
    }

    /// SIMD implementation of vector exponential
    #[cfg(target_arch = "x86_64")]
    fn vector_exp_simd(&mut self, vector: &[f64]) -> Result<Vec<f64>> {
        // Note: Fast exponential using polynomial approximation for SIMD
        // This is a simplified version; production code would use more sophisticated methods
        let mut _result = vec![0.0; vector.len()];
        let _chunks = vector.len() / self.vector_width;
        
        match self.instruction_set {
            SimdInstructionSet::Avx512 | SimdInstructionSet::Avx2 | SimdInstructionSet::Sse42 => {
                // Use scalar fallback for complex functions like exp
                // In production, this would use optimized SIMD exp implementations
                return Ok(self.vector_exp_scalar(vector));
            },
            _ => return Ok(self.vector_exp_scalar(vector)),
        }
    }

    /// SIMD implementation of vector logarithm
    #[cfg(target_arch = "x86_64")]
    fn vector_log_simd(&mut self, vector: &[f64]) -> Result<Vec<f64>> {
        // Similar to exp, log requires sophisticated SIMD implementations
        // Using scalar fallback for simplicity
        Ok(self.vector_log_scalar(vector))
    }

    /// Horizontal sum for AVX-512 (disabled - requires nightly)
    #[cfg(target_arch = "x86_64")]
    #[allow(dead_code)]
    unsafe fn horizontal_sum_avx512(&self, _v: __m256d) -> f64 {
        // AVX-512 requires nightly features, return 0.0 as fallback
        0.0
    }

    /// Horizontal sum for AVX2
    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_avx2(&self, v: __m256d) -> f64 {
        let sum128_1 = _mm256_extractf128_pd(v, 0);
        let sum128_2 = _mm256_extractf128_pd(v, 1);
        let sum128 = _mm_add_pd(sum128_1, sum128_2);
        self.horizontal_sum_sse(sum128)
    }

    /// Horizontal sum for SSE
    #[cfg(target_arch = "x86_64")]
    unsafe fn horizontal_sum_sse(&self, v: __m128d) -> f64 {
        let high = _mm_unpackhi_pd(v, v);
        let sum = _mm_add_sd(v, high);
        _mm_cvtsd_f64(sum)
    }

    /// Scalar fallback implementations
    fn vector_add_scalar(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    fn vector_multiply_scalar(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    fn scalar_multiply_scalar(&self, vector: &[f64], scalar: f64) -> Vec<f64> {
        vector.iter().map(|&x| x * scalar).collect()
    }

    fn dot_product_scalar(&self, a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    fn vector_exp_scalar(&self, vector: &[f64]) -> Vec<f64> {
        vector.iter().map(|&x| x.exp()).collect()
    }

    fn vector_log_scalar(&self, vector: &[f64]) -> Vec<f64> {
        vector.iter().map(|&x| x.ln()).collect()
    }

    /// Non-x86_64 fallback implementations
    #[cfg(not(target_arch = "x86_64"))]
    fn vector_add_simd(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        Ok(self.vector_add_scalar(a, b))
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn vector_multiply_simd(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        Ok(self.vector_multiply_scalar(a, b))
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn scalar_multiply_simd(&mut self, vector: &[f64], scalar: f64) -> Result<Vec<f64>> {
        Ok(self.scalar_multiply_scalar(vector, scalar))
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn dot_product_simd(&mut self, a: &[f64], b: &[f64]) -> Result<f64> {
        Ok(self.dot_product_scalar(a, b))
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn vector_exp_simd(&mut self, vector: &[f64]) -> Result<Vec<f64>> {
        Ok(self.vector_exp_scalar(vector))
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn vector_log_simd(&mut self, vector: &[f64]) -> Result<Vec<f64>> {
        Ok(self.vector_log_scalar(vector))
    }

    /// Phase 1 SIMD: AVX-512 enabled vector addition (8-element processing)
    #[cfg(target_arch = "x86_64")]
    fn vector_add_avx512_enabled(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![0.0; a.len()];
        let vector_width = 8; // AVX-512 processes 8 x f64 at once
        let chunks = a.len() / vector_width;
        
        unsafe {
            // Use AVX-512 intrinsics for maximum performance if available
            #[cfg(all(target_feature = "avx512f"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    for i in 0..chunks {
                        let offset = i * vector_width;
                        let va = _mm512_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm512_add_pd(va, vb);
                        _mm512_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                } else {
                    // Fallback to AVX2 if AVX-512 not available
                    return self.vector_add_scalar_fallback(a, b);
                }
            }
            
            // For stable Rust without AVX-512, use AVX2 fallback
            #[cfg(not(all(target_feature = "avx512f")))]
            {
                // Use AVX2 for stable compilation
                let avx2_width = 4; // AVX2 processes 4 x f64 at once
                let avx2_chunks = a.len() / avx2_width;
                
                if is_x86_feature_detected!("avx2") {
                    for i in 0..avx2_chunks {
                        let offset = i * avx2_width;
                        let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm256_add_pd(va, vb);
                        _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                    
                    // Handle remainder elements for AVX2
                    let remainder_start = avx2_chunks * avx2_width;
                    for i in remainder_start..a.len() {
                        result[i] = a[i] + b[i];
                    }
                    
                    return Ok(result);
                } else {
                    // Fallback to scalar if AVX2 not available
                    return self.vector_add_scalar_fallback(a, b);
                }
            }
        }
        
        // Handle remainder elements for AVX-512
        let remainder_start = chunks * vector_width;
        for i in remainder_start..a.len() {
            result[i] = a[i] + b[i];
        }
        
        Ok(result)
    }
    
    /// Phase 1 SIMD: AVX-512 enabled vector multiplication (8-element processing)
    #[cfg(target_arch = "x86_64")]
    fn vector_multiply_avx512_enabled(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![0.0; a.len()];
        let vector_width = 8; // AVX-512 processes 8 x f64 at once
        let chunks = a.len() / vector_width;
        
        unsafe {
            // Use AVX-512 intrinsics for maximum performance if available
            #[cfg(all(target_feature = "avx512f"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    for i in 0..chunks {
                        let offset = i * vector_width;
                        let va = _mm512_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm512_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm512_mul_pd(va, vb);
                        _mm512_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                } else {
                    // Fallback to AVX2 if AVX-512 not available
                    return self.vector_multiply_scalar_fallback(a, b);
                }
            }
            
            // For stable Rust without AVX-512, use AVX2 fallback
            #[cfg(not(all(target_feature = "avx512f")))]
            {
                // Use AVX2 for stable compilation
                let avx2_width = 4; // AVX2 processes 4 x f64 at once
                let avx2_chunks = a.len() / avx2_width;
                
                if is_x86_feature_detected!("avx2") {
                    for i in 0..avx2_chunks {
                        let offset = i * avx2_width;
                        let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                        let vr = _mm256_mul_pd(va, vb);
                        _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                    
                    // Handle remainder elements for AVX2
                    let remainder_start = avx2_chunks * avx2_width;
                    for i in remainder_start..a.len() {
                        result[i] = a[i] * b[i];
                    }
                    
                    return Ok(result);
                } else {
                    // Fallback to scalar if AVX2 not available
                    return self.vector_multiply_scalar_fallback(a, b);
                }
            }
        }
        
        // Handle remainder elements for AVX-512
        let remainder_start = chunks * vector_width;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i];
        }
        
        Ok(result)
    }

    /// Additional fallback methods for error cases
    fn vector_add_scalar_fallback(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        Ok(self.vector_add_scalar(a, b))
    }

    fn vector_multiply_scalar_fallback(&self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        Ok(self.vector_multiply_scalar(a, b))
    }

    fn scalar_multiply_scalar_fallback(&self, vector: &[f64], scalar: f64) -> Result<Vec<f64>> {
        Ok(self.scalar_multiply_scalar(vector, scalar))
    }

    /// Parallel SIMD operations for extremely large datasets
    pub fn vector_add_parallel(&mut self, a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(AtsCoreError::dimension_mismatch(b.len(), a.len()));
        }
        
        let chunk_size = 1024; // Optimal chunk size for parallel processing
        let result: Result<Vec<f64>> = a
            .par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .map(|(chunk_a, chunk_b)| {
                if chunk_a.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
                    // Create a mutable reference for SIMD operations
                    let mut simd_ops = SimdOperations::new(&self.config)?;
                    simd_ops.vector_add_simd(chunk_a, chunk_b)
                } else {
                    Ok(self.vector_add_scalar(chunk_a, chunk_b))
                }
            })
            .collect::<Result<Vec<Vec<f64>>>>()
            .map(|chunks| chunks.into_iter().flatten().collect());
        
        result
    }

    /// Fused multiply-add operation (a * b + c)
    pub fn fused_multiply_add(&mut self, a: &[f64], b: &[f64], c: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() || a.len() != c.len() {
            return Err(AtsCoreError::dimension_mismatch(a.len(), b.len()));
        }
        
        let start_time = Instant::now();
        
        let result = if a.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.fused_multiply_add_simd(a, b, c)?
        } else {
            self.fused_multiply_add_scalar(a, b, c)
        };
        
        // Update performance statistics
        let elapsed = start_time.elapsed();
        let elapsed_ns = elapsed.as_nanos() as u64;
        self.total_operations += 1;
        self.total_time_ns += elapsed_ns;
        
        Ok(result)
    }

    /// SIMD fused multiply-add
    #[cfg(target_arch = "x86_64")]
    fn fused_multiply_add_simd(&mut self, a: &[f64], b: &[f64], c: &[f64]) -> Result<Vec<f64>> {
        let mut result = vec![0.0; a.len()];
        let chunks = a.len() / self.vector_width;
        
        unsafe {
            match self.instruction_set {
                SimdInstructionSet::Avx512 => {
                    // AVX-512 requires nightly features, falling back to scalar
                    return Ok(self.fused_multiply_add_scalar(a, b, c));
                },
                SimdInstructionSet::Avx2 => {
                    for i in 0..chunks {
                        let offset = i * self.vector_width;
                        let va = _mm256_loadu_pd(a.as_ptr().add(offset));
                        let vb = _mm256_loadu_pd(b.as_ptr().add(offset));
                        let vc = _mm256_loadu_pd(c.as_ptr().add(offset));
                        let vr = _mm256_fmadd_pd(va, vb, vc);
                        _mm256_storeu_pd(result.as_mut_ptr().add(offset), vr);
                        self.total_simd_ops += 1;
                    }
                },
                _ => return Ok(self.fused_multiply_add_scalar(a, b, c)),
            }
        }
        
        // Handle remainder elements
        let remainder_start = chunks * self.vector_width;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i] + c[i];
        }
        
        Ok(result)
    }

    #[cfg(not(target_arch = "x86_64"))]
    fn fused_multiply_add_simd(&mut self, a: &[f64], b: &[f64], c: &[f64]) -> Result<Vec<f64>> {
        Ok(self.fused_multiply_add_scalar(a, b, c))
    }

    fn fused_multiply_add_scalar(&self, a: &[f64], b: &[f64], c: &[f64]) -> Vec<f64> {
        a.iter()
            .zip(b.iter())
            .zip(c.iter())
            .map(|((x, y), z)| x * y + z)
            .collect()
    }

    /// Phase 1 SIMD: SIMD-optimized correlation calculation with AVX-512
    pub fn correlation_coefficient(&mut self, x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Err(AtsCoreError::dimension_mismatch(x.len(), y.len()));
        }
        
        let n = x.len() as f64;
        
        // Calculate means using SIMD
        let sum_x = self.vector_sum_simd(x)?;
        let sum_y = self.vector_sum_simd(y)?;
        let mean_x = sum_x / n;
        let mean_y = sum_y / n;
        
        // Calculate correlation components using SIMD
        let mut numerator = 0.0;
        let mut sum_x_sq = 0.0;
        let mut sum_y_sq = 0.0;
        
        if x.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            // Use SIMD for correlation calculation
            let result = self.correlation_simd_avx512(x, y, mean_x, mean_y)?;
            return Ok(result);
        } else {
            // Scalar fallback
            for i in 0..x.len() {
                let dx = x[i] - mean_x;
                let dy = y[i] - mean_y;
                numerator += dx * dy;
                sum_x_sq += dx * dx;
                sum_y_sq += dy * dy;
            }
        }
        
        let denominator = (sum_x_sq * sum_y_sq).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    /// Phase 1 SIMD: Vectorized wavelet transform (simplified Haar wavelet)
    pub fn wavelet_transform_haar(&mut self, signal: &[f64]) -> Result<Vec<f64>> {
        if signal.is_empty() || signal.len() % 2 != 0 {
            return Err(AtsCoreError::mathematical("wavelet_transform", "signal length must be even"));
        }
        
        let mut result = vec![0.0; signal.len()];
        let half_len = signal.len() / 2;
        
        if signal.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.wavelet_haar_simd_avx512(signal, &mut result)?;
        } else {
            // Scalar Haar wavelet
            for i in 0..half_len {
                let avg = (signal[2*i] + signal[2*i+1]) / 2.0;
                let diff = (signal[2*i] - signal[2*i+1]) / 2.0;
                result[i] = avg;         // Low-pass (approximation)
                result[i + half_len] = diff;  // High-pass (detail)
            }
        }
        
        Ok(result)
    }
    
    /// Phase 1 SIMD: Neural spike detection using SIMD threshold operations
    pub fn neural_spike_detection(&mut self, signal: &[f64], threshold: f64) -> Result<Vec<bool>> {
        if signal.is_empty() {
            return Ok(Vec::new());
        }
        
        if signal.len() >= self.config.simd.min_simd_size && self.config.simd.enabled {
            self.spike_detection_simd_avx512(signal, threshold)
        } else {
            // Scalar spike detection
            Ok(signal.iter().map(|&x| x > threshold).collect())
        }
    }

    /// Returns performance statistics
    pub fn get_performance_stats(&self) -> (u64, u64, u64, f64) {
        let avg_latency = if self.total_operations > 0 {
            self.total_time_ns / self.total_operations
        } else {
            0
        };
        
        let ops_per_second = if self.total_time_ns > 0 {
            (self.total_operations as f64) / (self.total_time_ns as f64 / 1_000_000_000.0)
        } else {
            0.0
        };
        
        (self.total_operations, self.total_simd_ops, avg_latency, ops_per_second)
    }

    /// Returns the detected instruction set and capabilities
    pub fn get_capabilities(&self) -> (SimdInstructionSet, usize, usize) {
        (self.instruction_set.clone(), self.vector_width, self.alignment)
    }
    
    /// Phase 1 SIMD: SIMD vector sum using AVX-512 or AVX2 fallback
    #[cfg(target_arch = "x86_64")]
    fn vector_sum_simd(&mut self, vector: &[f64]) -> Result<f64> {
        if vector.is_empty() {
            return Ok(0.0);
        }
        
        unsafe {
            // Try AVX-512 first if available and compiled with support
            #[cfg(all(target_feature = "avx512f"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    let mut sum = 0.0;
                    let vector_width = 8; // AVX-512 processes 8 x f64
                    let chunks = vector.len() / vector_width;
                    let mut vsum = _mm512_setzero_pd();
                    
                    for i in 0..chunks {
                        let offset = i * vector_width;
                        let v = _mm512_loadu_pd(vector.as_ptr().add(offset));
                        vsum = _mm512_add_pd(vsum, v);
                    }
                    
                    sum += _mm512_reduce_add_pd(vsum);
                    
                    // Handle remainder elements
                    let remainder_start = chunks * vector_width;
                    for i in remainder_start..vector.len() {
                        sum += vector[i];
                    }
                    
                    return Ok(sum);
                }
            }
            
            // AVX2 fallback for stable compilation
            if is_x86_feature_detected!("avx2") {
                let mut sum = 0.0;
                let vector_width = 4; // AVX2 processes 4 x f64
                let chunks = vector.len() / vector_width;
                let mut vsum = _mm256_setzero_pd();
                
                for i in 0..chunks {
                    let offset = i * vector_width;
                    let v = _mm256_loadu_pd(vector.as_ptr().add(offset));
                    vsum = _mm256_add_pd(vsum, v);
                }
                
                sum += self.horizontal_sum_avx2(vsum);
                
                // Handle remainder elements
                let remainder_start = chunks * vector_width;
                for i in remainder_start..vector.len() {
                    sum += vector[i];
                }
                
                Ok(sum)
            } else {
                // Fallback to scalar sum
                Ok(vector.iter().sum())
            }
        }
    }
    
    /// Phase 1 SIMD: AVX-512 correlation calculation with AVX2 fallback
    #[cfg(target_arch = "x86_64")]
    fn correlation_simd_avx512(&mut self, x: &[f64], y: &[f64], mean_x: f64, mean_y: f64) -> Result<f64> {
        unsafe {
            // Try AVX-512 first if available and compiled with support
            #[cfg(all(target_feature = "avx512f"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    let vector_width = 8;
                    let chunks = x.len() / vector_width;
                    
                    let mut numerator = 0.0;
                    let mut sum_x_sq = 0.0;
                    let mut sum_y_sq = 0.0;
                    
                    let vmean_x = _mm512_set1_pd(mean_x);
                    let vmean_y = _mm512_set1_pd(mean_y);
                    let mut vnum = _mm512_setzero_pd();
                    let mut vx_sq = _mm512_setzero_pd();
                    let mut vy_sq = _mm512_setzero_pd();
                    
                    for i in 0..chunks {
                        let offset = i * vector_width;
                        let vx = _mm512_loadu_pd(x.as_ptr().add(offset));
                        let vy = _mm512_loadu_pd(y.as_ptr().add(offset));
                        
                        let dx = _mm512_sub_pd(vx, vmean_x);
                        let dy = _mm512_sub_pd(vy, vmean_y);
                        
                        vnum = _mm512_fmadd_pd(dx, dy, vnum);
                        vx_sq = _mm512_fmadd_pd(dx, dx, vx_sq);
                        vy_sq = _mm512_fmadd_pd(dy, dy, vy_sq);
                        
                        self.total_simd_ops += 1;
                    }
                    
                    numerator += _mm512_reduce_add_pd(vnum);
                    sum_x_sq += _mm512_reduce_add_pd(vx_sq);
                    sum_y_sq += _mm512_reduce_add_pd(vy_sq);
                    
                    // Handle remainder elements
                    let remainder_start = chunks * vector_width;
                    for i in remainder_start..x.len() {
                        let dx = x[i] - mean_x;
                        let dy = y[i] - mean_y;
                        numerator += dx * dy;
                        sum_x_sq += dx * dx;
                        sum_y_sq += dy * dy;
                    }
                    
                    let denominator = (sum_x_sq * sum_y_sq).sqrt();
                    if denominator == 0.0 {
                        Ok(0.0)
                    } else {
                        Ok(numerator / denominator)
                    }
                } else {
                    // Fallback if AVX-512 not available at runtime
                    self.correlation_simd_avx2(x, y, mean_x, mean_y)
                }
            }
            
            // AVX2 fallback for stable compilation
            #[cfg(not(all(target_feature = "avx512f")))]
            {
                self.correlation_simd_avx2(x, y, mean_x, mean_y)
            }
        }
    }
    
    /// AVX2 correlation calculation fallback
    #[cfg(target_arch = "x86_64")]
    fn correlation_simd_avx2(&mut self, x: &[f64], y: &[f64], mean_x: f64, mean_y: f64) -> Result<f64> {
        unsafe {
            if is_x86_feature_detected!("avx2") {
                let vector_width = 4;
                let chunks = x.len() / vector_width;
                
                let mut numerator = 0.0;
                let mut sum_x_sq = 0.0;
                let mut sum_y_sq = 0.0;
                
                let vmean_x = _mm256_set1_pd(mean_x);
                let vmean_y = _mm256_set1_pd(mean_y);
                let mut vnum = _mm256_setzero_pd();
                let mut vx_sq = _mm256_setzero_pd();
                let mut vy_sq = _mm256_setzero_pd();
                
                for i in 0..chunks {
                    let offset = i * vector_width;
                    let vx = _mm256_loadu_pd(x.as_ptr().add(offset));
                    let vy = _mm256_loadu_pd(y.as_ptr().add(offset));
                    
                    let dx = _mm256_sub_pd(vx, vmean_x);
                    let dy = _mm256_sub_pd(vy, vmean_y);
                    
                    vnum = _mm256_fmadd_pd(dx, dy, vnum);
                    vx_sq = _mm256_fmadd_pd(dx, dx, vx_sq);
                    vy_sq = _mm256_fmadd_pd(dy, dy, vy_sq);
                    
                    self.total_simd_ops += 1;
                }
                
                numerator += self.horizontal_sum_avx2(vnum);
                sum_x_sq += self.horizontal_sum_avx2(vx_sq);
                sum_y_sq += self.horizontal_sum_avx2(vy_sq);
                
                // Handle remainder elements
                let remainder_start = chunks * vector_width;
                for i in remainder_start..x.len() {
                    let dx = x[i] - mean_x;
                    let dy = y[i] - mean_y;
                    numerator += dx * dy;
                    sum_x_sq += dx * dx;
                    sum_y_sq += dy * dy;
                }
                
                let denominator = (sum_x_sq * sum_y_sq).sqrt();
                if denominator == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(numerator / denominator)
                }
            } else {
                // Scalar fallback
                let mut numerator = 0.0;
                let mut sum_x_sq = 0.0;
                let mut sum_y_sq = 0.0;
                
                for i in 0..x.len() {
                    let dx = x[i] - mean_x;
                    let dy = y[i] - mean_y;
                    numerator += dx * dy;
                    sum_x_sq += dx * dx;
                    sum_y_sq += dy * dy;
                }
                
                let denominator = (sum_x_sq * sum_y_sq).sqrt();
                if denominator == 0.0 {
                    Ok(0.0)
                } else {
                    Ok(numerator / denominator)
                }
            }
        }
    }
    
    /// Phase 1 SIMD: AVX-512 Haar wavelet transform with AVX2 fallback
    #[cfg(target_arch = "x86_64")]
    fn wavelet_haar_simd_avx512(&mut self, signal: &[f64], result: &mut [f64]) -> Result<()> {
        unsafe {
            let half_len = signal.len() / 2;
            
            // Try AVX-512 first if available and compiled with support
            #[cfg(all(target_feature = "avx512f"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    let vector_width = 8;
                    let chunks = half_len / vector_width;
                    
                    let sqrt2_inv = _mm512_set1_pd(1.0 / 2.0_f64.sqrt());
                    
                    for i in 0..chunks {
                        let offset = i * vector_width;
                        
                        // Load pairs of values
                        let even_vals = _mm512_loadu_pd(signal.as_ptr().add(offset * 2));
                        let odd_vals = _mm512_loadu_pd(signal.as_ptr().add(offset * 2 + vector_width));
                        
                        // Compute approximation (low-pass) and detail (high-pass)
                        let approx = _mm512_mul_pd(_mm512_add_pd(even_vals, odd_vals), sqrt2_inv);
                        let detail = _mm512_mul_pd(_mm512_sub_pd(even_vals, odd_vals), sqrt2_inv);
                        
                        // Store results
                        _mm512_storeu_pd(result.as_mut_ptr().add(offset), approx);
                        _mm512_storeu_pd(result.as_mut_ptr().add(offset + half_len), detail);
                        
                        self.total_simd_ops += 1;
                    }
                    
                    // Handle remainder elements
                    let remainder_start = chunks * vector_width;
                    for i in remainder_start..half_len {
                        let avg = (signal[2*i] + signal[2*i+1]) / 2.0;
                        let diff = (signal[2*i] - signal[2*i+1]) / 2.0;
                        result[i] = avg;
                        result[i + half_len] = diff;
                    }
                    
                    return Ok(());
                }
            }
            
            // AVX2 fallback for stable compilation
            if is_x86_feature_detected!("avx2") {
                let vector_width = 4;
                let chunks = half_len / vector_width;
                
                let sqrt2_inv = _mm256_set1_pd(1.0 / 2.0_f64.sqrt());
                
                for i in 0..chunks {
                    let offset = i * vector_width;
                    
                    // Load pairs of values
                    let even_vals = _mm256_loadu_pd(signal.as_ptr().add(offset * 2));
                    let odd_vals = _mm256_loadu_pd(signal.as_ptr().add(offset * 2 + vector_width));
                    
                    // Compute approximation (low-pass) and detail (high-pass)
                    let approx = _mm256_mul_pd(_mm256_add_pd(even_vals, odd_vals), sqrt2_inv);
                    let detail = _mm256_mul_pd(_mm256_sub_pd(even_vals, odd_vals), sqrt2_inv);
                    
                    // Store results
                    _mm256_storeu_pd(result.as_mut_ptr().add(offset), approx);
                    _mm256_storeu_pd(result.as_mut_ptr().add(offset + half_len), detail);
                    
                    self.total_simd_ops += 1;
                }
                
                // Handle remainder elements
                let remainder_start = chunks * vector_width;
                for i in remainder_start..half_len {
                    let avg = (signal[2*i] + signal[2*i+1]) / 2.0;
                    let diff = (signal[2*i] - signal[2*i+1]) / 2.0;
                    result[i] = avg;
                    result[i + half_len] = diff;
                }
                
                Ok(())
            } else {
                // Scalar fallback
                for i in 0..half_len {
                    let avg = (signal[2*i] + signal[2*i+1]) / 2.0;
                    let diff = (signal[2*i] - signal[2*i+1]) / 2.0;
                    result[i] = avg;
                    result[i + half_len] = diff;
                }
                Ok(())
            }
        }
    }
    
    /// Phase 1 SIMD: AVX-512 spike detection with AVX2 fallback
    #[cfg(target_arch = "x86_64")]
    fn spike_detection_simd_avx512(&mut self, signal: &[f64], threshold: f64) -> Result<Vec<bool>> {
        unsafe {
            let mut result = vec![false; signal.len()];
            
            // Try AVX-512 first if available and compiled with support
            #[cfg(all(target_feature = "avx512f"))]
            {
                if is_x86_feature_detected!("avx512f") {
                    let vector_width = 8;
                    let chunks = signal.len() / vector_width;
                    
                    let vthreshold = _mm512_set1_pd(threshold);
                    
                    for i in 0..chunks {
                        let offset = i * vector_width;
                        let vsignal = _mm512_loadu_pd(signal.as_ptr().add(offset));
                        
                        // Compare with threshold (returns mask)
                        let mask = _mm512_cmp_pd_mask(vsignal, vthreshold, _CMP_GT_OQ);
                        
                        // Convert mask to booleans
                        for j in 0..vector_width {
                            result[offset + j] = (mask & (1 << j)) != 0;
                        }
                        
                        self.total_simd_ops += 1;
                    }
                    
                    // Handle remainder elements
                    let remainder_start = chunks * vector_width;
                    for i in remainder_start..signal.len() {
                        result[i] = signal[i] > threshold;
                    }
                    
                    return Ok(result);
                }
            }
            
            // AVX2 fallback for stable compilation
            if is_x86_feature_detected!("avx2") {
                let vector_width = 4;
                let chunks = signal.len() / vector_width;
                
                let vthreshold = _mm256_set1_pd(threshold);
                
                for i in 0..chunks {
                    let offset = i * vector_width;
                    let vsignal = _mm256_loadu_pd(signal.as_ptr().add(offset));
                    
                    // Compare with threshold
                    let mask = _mm256_cmp_pd(vsignal, vthreshold, _CMP_GT_OQ);
                    
                    // Convert mask to booleans (AVX2 method)
                    let mask_int = _mm256_movemask_pd(mask);
                    for j in 0..vector_width {
                        result[offset + j] = (mask_int & (1 << j)) != 0;
                    }
                    
                    self.total_simd_ops += 1;
                }
                
                // Handle remainder elements
                let remainder_start = chunks * vector_width;
                for i in remainder_start..signal.len() {
                    result[i] = signal[i] > threshold;
                }
                
                Ok(result)
            } else {
                // Scalar fallback
                for i in 0..signal.len() {
                    result[i] = signal[i] > threshold;
                }
                Ok(result)
            }
        }
    }
    
    // Non-x86_64 fallbacks for new functions
    #[cfg(not(target_arch = "x86_64"))]
    fn vector_sum_simd(&mut self, vector: &[f64]) -> Result<f64> {
        Ok(vector.iter().sum())
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn correlation_simd_avx512(&mut self, x: &[f64], y: &[f64], mean_x: f64, mean_y: f64) -> Result<f64> {
        // Scalar fallback
        let mut numerator = 0.0;
        let mut sum_x_sq = 0.0;
        let mut sum_y_sq = 0.0;
        
        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            numerator += dx * dy;
            sum_x_sq += dx * dx;
            sum_y_sq += dy * dy;
        }
        
        let denominator = (sum_x_sq * sum_y_sq).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn correlation_simd_avx2(&mut self, x: &[f64], y: &[f64], mean_x: f64, mean_y: f64) -> Result<f64> {
        // Scalar fallback for non-x86_64
        self.correlation_simd_avx512(x, y, mean_x, mean_y)
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn wavelet_haar_simd_avx512(&mut self, signal: &[f64], result: &mut [f64]) -> Result<()> {
        let half_len = signal.len() / 2;
        for i in 0..half_len {
            let avg = (signal[2*i] + signal[2*i+1]) / 2.0;
            let diff = (signal[2*i] - signal[2*i+1]) / 2.0;
            result[i] = avg;
            result[i + half_len] = diff;
        }
        Ok(())
    }
    
    #[cfg(not(target_arch = "x86_64"))]
    fn spike_detection_simd_avx512(&mut self, signal: &[f64], threshold: f64) -> Result<Vec<bool>> {
        Ok(signal.iter().map(|&x| x > threshold).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::AtsCpConfig;
    use approx::assert_relative_eq;

    fn create_test_config() -> AtsCpConfig {
        AtsCpConfig {
            temperature: crate::config::TemperatureConfig {
                target_latency_us: 10_000, // Relaxed for testing (10ms)
                ..Default::default()
            },
            conformal: crate::config::ConformalConfig {
                target_latency_us: 10_000, // Relaxed for testing (10ms)
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_simd_operations_creation() {
        let config = create_test_config();
        let simd_ops = SimdOperations::new(&config);
        assert!(simd_ops.is_ok());
    }

    #[test]
    fn test_instruction_set_detection() {
        let instruction_set = SimdOperations::detect_instruction_set();
        // Just verify it doesn't panic and returns a valid instruction set
        match instruction_set {
            SimdInstructionSet::Auto | SimdInstructionSet::Avx512 | 
            SimdInstructionSet::Avx2 | SimdInstructionSet::Sse42 | 
            SimdInstructionSet::Neon => {},
        }
    }

    #[test]
    fn test_vector_addition() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5];
        
        let result = simd_ops.vector_add(&a, &b).unwrap();
        
        assert_eq!(result.len(), a.len());
        for i in 0..a.len() {
            assert_relative_eq!(result[i], a[i] + b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_vector_multiplication() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let result = simd_ops.vector_multiply(&a, &b).unwrap();
        
        assert_eq!(result.len(), a.len());
        for i in 0..a.len() {
            assert_relative_eq!(result[i], a[i] * b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scalar = 2.5;
        
        let result = simd_ops.scalar_multiply(&vector, scalar).unwrap();
        
        assert_eq!(result.len(), vector.len());
        for i in 0..vector.len() {
            assert_relative_eq!(result[i], vector[i] * scalar, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_dot_product() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = simd_ops.dot_product(&a, &b).unwrap();
        let expected = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0; // = 70.0
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_vector_exponential() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let vector = vec![0.0, 1.0, 2.0, -1.0];
        let result = simd_ops.vector_exp(&vector).unwrap();
        
        assert_eq!(result.len(), vector.len());
        for i in 0..vector.len() {
            assert_relative_eq!(result[i], vector[i].exp(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_vector_logarithm() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let result = simd_ops.vector_log(&vector).unwrap();
        
        assert_eq!(result.len(), vector.len());
        for i in 0..vector.len() {
            assert_relative_eq!(result[i], vector[i].ln(), epsilon = 1e-6);
        }
    }

    #[test]
    fn test_fused_multiply_add() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = simd_ops.fused_multiply_add(&a, &b, &c).unwrap();
        
        assert_eq!(result.len(), a.len());
        for i in 0..a.len() {
            let expected = a[i] * b[i] + c[i];
            assert_relative_eq!(result[i], expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_parallel_vector_addition() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        // Large vectors to trigger parallel processing
        let a: Vec<f64> = (0..10000).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..10000).map(|i| (i as f64) * 0.5).collect();
        
        let result = simd_ops.vector_add_parallel(&a, &b).unwrap();
        
        assert_eq!(result.len(), a.len());
        for i in 0..a.len() {
            assert_relative_eq!(result[i], a[i] + b[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_error_handling() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        // Test dimension mismatch
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0]; // Different length
        let result = simd_ops.vector_add(&a, &b);
        assert!(result.is_err());
        
        // Test logarithm of negative number
        let negative_vector = vec![1.0, -1.0, 2.0];
        let result = simd_ops.vector_log(&negative_vector);
        assert!(result.is_err());
        
        // Test logarithm of zero
        let zero_vector = vec![1.0, 0.0, 2.0];
        let result = simd_ops.vector_log(&zero_vector);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_vectors() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let empty_a: Vec<f64> = vec![];
        let empty_b: Vec<f64> = vec![];
        
        let result = simd_ops.vector_add(&empty_a, &empty_b).unwrap();
        assert!(result.is_empty());
        
        let dot_result = simd_ops.dot_product(&empty_a, &empty_b).unwrap();
        assert_eq!(dot_result, 0.0);
    }

    #[test]
    fn test_performance_stats() {
        let config = create_test_config();
        let mut simd_ops = SimdOperations::new(&config).unwrap();
        
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        // Perform several operations
        for _ in 0..10 {
            let _ = simd_ops.vector_add(&a, &b).unwrap();
        }
        
        let (total_ops, simd_ops_count, avg_latency, ops_per_sec) = simd_ops.get_performance_stats();
        
        assert_eq!(total_ops, 10);
        assert!(avg_latency > 0);
        assert!(ops_per_sec > 0.0);
    }

    #[test]
    fn test_capabilities() {
        let config = create_test_config();
        let simd_ops = SimdOperations::new(&config).unwrap();
        
        let (instruction_set, vector_width, alignment) = simd_ops.get_capabilities();
        
        assert!(vector_width > 0);
        assert!(alignment > 0);
        assert!(alignment.is_power_of_two());
    }
}