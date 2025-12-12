//! SIMD optimizations for swarm intelligence algorithms
//!
//! Provides high-performance vectorized operations using modern SIMD instructions
//! for accelerated swarm computations.

use crate::core::{SwarmError, Position, Velocity};
use nalgebra::DVector;
use std::arch::x86_64::*;

/// SIMD-optimized operations for swarm algorithms
pub struct SimdProcessor {
    /// Target dimension alignment for SIMD operations
    simd_alignment: usize,
    
    /// Available SIMD instruction sets
    instruction_sets: SimdCapabilities,
    
    /// Whether SIMD is enabled
    enabled: bool,
}

/// Available SIMD instruction sets
#[derive(Debug, Clone)]
pub struct SimdCapabilities {
    pub has_sse: bool,
    pub has_sse2: bool,
    pub has_sse3: bool,
    pub has_ssse3: bool,
    pub has_sse4_1: bool,
    pub has_sse4_2: bool,
    pub has_avx: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_fma: bool,
}

impl SimdProcessor {
    /// Create a new SIMD processor with auto-detection
    pub fn new() -> Self {
        Self {
            simd_alignment: 8, // Default to AVX2 alignment
            instruction_sets: Self::detect_capabilities(),
            enabled: true,
        }
    }
    
    /// Detect available SIMD capabilities
    fn detect_capabilities() -> SimdCapabilities {
        // Use is_x86_feature_detected! macro for runtime detection
        SimdCapabilities {
            has_sse: is_x86_feature_detected!("sse"),
            has_sse2: is_x86_feature_detected!("sse2"),
            has_sse3: is_x86_feature_detected!("sse3"),
            has_ssse3: is_x86_feature_detected!("ssse3"),
            has_sse4_1: is_x86_feature_detected!("sse4.1"),
            has_sse4_2: is_x86_feature_detected!("sse4.2"),
            has_avx: is_x86_feature_detected!("avx"),
            has_avx2: is_x86_feature_detected!("avx2"),
            has_avx512f: is_x86_feature_detected!("avx512f"),
            has_fma: is_x86_feature_detected!("fma"),
        }
    }
    
    /// Vector addition with SIMD optimization
    pub fn vector_add(&self, a: &Position, b: &Position) -> Result<Position, SwarmError> {
        if !self.enabled || a.len() != b.len() {
            return Ok(a + b); // Fallback to standard operation
        }
        
        if self.instruction_sets.has_avx2 && a.len() >= 4 {
            self.vector_add_avx2(a, b)
        } else if self.instruction_sets.has_sse2 && a.len() >= 2 {
            self.vector_add_sse2(a, b)
        } else {
            Ok(a + b)
        }
    }
    
    /// AVX2-optimized vector addition
    #[target_feature(enable = "avx2")]
    unsafe fn vector_add_avx2(&self, a: &Position, b: &Position) -> Result<Position, SwarmError> {
        let len = a.len();
        let mut result = Position::zeros(len);
        
        let chunks = len / 4;
        let remainder = len % 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            
            // Load 4 doubles from each vector
            let a_chunk = _mm256_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm256_loadu_pd(b.as_slice().as_ptr().add(offset));
            
            // Add vectors
            let sum = _mm256_add_pd(a_chunk, b_chunk);
            
            // Store result
            _mm256_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), sum);
        }
        
        // Handle remainder elements
        for i in (chunks * 4)..len {
            result[i] = a[i] + b[i];
        }
        
        Ok(result)
    }
    
    /// SSE2-optimized vector addition
    #[target_feature(enable = "sse2")]
    unsafe fn vector_add_sse2(&self, a: &Position, b: &Position) -> Result<Position, SwarmError> {
        let len = a.len();
        let mut result = Position::zeros(len);
        
        let chunks = len / 2;
        let remainder = len % 2;
        
        for i in 0..chunks {
            let offset = i * 2;
            
            // Load 2 doubles from each vector
            let a_chunk = _mm_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm_loadu_pd(b.as_slice().as_ptr().add(offset));
            
            // Add vectors
            let sum = _mm_add_pd(a_chunk, b_chunk);
            
            // Store result
            _mm_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), sum);
        }
        
        // Handle remainder elements
        for i in (chunks * 2)..len {
            result[i] = a[i] + b[i];
        }
        
        Ok(result)
    }
    
    /// Vector subtraction with SIMD optimization
    pub fn vector_sub(&self, a: &Position, b: &Position) -> Result<Position, SwarmError> {
        if !self.enabled || a.len() != b.len() {
            return Ok(a - b);
        }
        
        if self.instruction_sets.has_avx2 && a.len() >= 4 {
            self.vector_sub_avx2(a, b)
        } else if self.instruction_sets.has_sse2 && a.len() >= 2 {
            self.vector_sub_sse2(a, b)
        } else {
            Ok(a - b)
        }
    }
    
    /// AVX2-optimized vector subtraction
    #[target_feature(enable = "avx2")]
    unsafe fn vector_sub_avx2(&self, a: &Position, b: &Position) -> Result<Position, SwarmError> {
        let len = a.len();
        let mut result = Position::zeros(len);
        
        let chunks = len / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            
            let a_chunk = _mm256_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm256_loadu_pd(b.as_slice().as_ptr().add(offset));
            
            let diff = _mm256_sub_pd(a_chunk, b_chunk);
            
            _mm256_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), diff);
        }
        
        // Handle remainder
        for i in (chunks * 4)..len {
            result[i] = a[i] - b[i];
        }
        
        Ok(result)
    }
    
    /// SSE2-optimized vector subtraction
    #[target_feature(enable = "sse2")]
    unsafe fn vector_sub_sse2(&self, a: &Position, b: &Position) -> Result<Position, SwarmError> {
        let len = a.len();
        let mut result = Position::zeros(len);
        
        let chunks = len / 2;
        
        for i in 0..chunks {
            let offset = i * 2;
            
            let a_chunk = _mm_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm_loadu_pd(b.as_slice().as_ptr().add(offset));
            
            let diff = _mm_sub_pd(a_chunk, b_chunk);
            
            _mm_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), diff);
        }
        
        // Handle remainder
        for i in (chunks * 2)..len {
            result[i] = a[i] - b[i];
        }
        
        Ok(result)
    }
    
    /// Scalar multiplication with SIMD optimization
    pub fn scalar_multiply(&self, vector: &Position, scalar: f64) -> Result<Position, SwarmError> {
        if !self.enabled {
            return Ok(vector * scalar);
        }
        
        if self.instruction_sets.has_avx2 && vector.len() >= 4 {
            self.scalar_multiply_avx2(vector, scalar)
        } else if self.instruction_sets.has_sse2 && vector.len() >= 2 {
            self.scalar_multiply_sse2(vector, scalar)
        } else {
            Ok(vector * scalar)
        }
    }
    
    /// AVX2-optimized scalar multiplication
    #[target_feature(enable = "avx2")]
    unsafe fn scalar_multiply_avx2(&self, vector: &Position, scalar: f64) -> Result<Position, SwarmError> {
        let len = vector.len();
        let mut result = Position::zeros(len);
        
        let scalar_vec = _mm256_set1_pd(scalar);
        let chunks = len / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            
            let vec_chunk = _mm256_loadu_pd(vector.as_slice().as_ptr().add(offset));
            let product = _mm256_mul_pd(vec_chunk, scalar_vec);
            
            _mm256_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), product);
        }
        
        // Handle remainder
        for i in (chunks * 4)..len {
            result[i] = vector[i] * scalar;
        }
        
        Ok(result)
    }
    
    /// SSE2-optimized scalar multiplication
    #[target_feature(enable = "sse2")]
    unsafe fn scalar_multiply_sse2(&self, vector: &Position, scalar: f64) -> Result<Position, SwarmError> {
        let len = vector.len();
        let mut result = Position::zeros(len);
        
        let scalar_vec = _mm_set1_pd(scalar);
        let chunks = len / 2;
        
        for i in 0..chunks {
            let offset = i * 2;
            
            let vec_chunk = _mm_loadu_pd(vector.as_slice().as_ptr().add(offset));
            let product = _mm_mul_pd(vec_chunk, scalar_vec);
            
            _mm_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), product);
        }
        
        // Handle remainder
        for i in (chunks * 2)..len {
            result[i] = vector[i] * scalar;
        }
        
        Ok(result)
    }
    
    /// Dot product with SIMD optimization
    pub fn dot_product(&self, a: &Position, b: &Position) -> Result<f64, SwarmError> {
        if !self.enabled || a.len() != b.len() {
            return Ok(a.dot(b));
        }
        
        if self.instruction_sets.has_avx2 && a.len() >= 4 {
            self.dot_product_avx2(a, b)
        } else if self.instruction_sets.has_sse2 && a.len() >= 2 {
            self.dot_product_sse2(a, b)
        } else {
            Ok(a.dot(b))
        }
    }
    
    /// AVX2-optimized dot product
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(&self, a: &Position, b: &Position) -> Result<f64, SwarmError> {
        let len = a.len();
        let chunks = len / 4;
        
        let mut sum_vec = _mm256_setzero_pd();
        
        for i in 0..chunks {
            let offset = i * 4;
            
            let a_chunk = _mm256_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm256_loadu_pd(b.as_slice().as_ptr().add(offset));
            
            let product = _mm256_mul_pd(a_chunk, b_chunk);
            sum_vec = _mm256_add_pd(sum_vec, product);
        }
        
        // Horizontal sum of the 4 doubles in sum_vec
        let sum_high = _mm256_extractf128_pd(sum_vec, 1);
        let sum_low = _mm256_extractf128_pd(sum_vec, 0);
        let sum_combined = _mm_add_pd(sum_high, sum_low);
        let sum_shuffled = _mm_shuffle_pd(sum_combined, sum_combined, 1);
        let final_sum = _mm_add_pd(sum_combined, sum_shuffled);
        
        let mut result = _mm_cvtsd_f64(final_sum);
        
        // Handle remainder elements
        for i in (chunks * 4)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    /// SSE2-optimized dot product
    #[target_feature(enable = "sse2")]
    unsafe fn dot_product_sse2(&self, a: &Position, b: &Position) -> Result<f64, SwarmError> {
        let len = a.len();
        let chunks = len / 2;
        
        let mut sum_vec = _mm_setzero_pd();
        
        for i in 0..chunks {
            let offset = i * 2;
            
            let a_chunk = _mm_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm_loadu_pd(b.as_slice().as_ptr().add(offset));
            
            let product = _mm_mul_pd(a_chunk, b_chunk);
            sum_vec = _mm_add_pd(sum_vec, product);
        }
        
        // Horizontal sum
        let sum_shuffled = _mm_shuffle_pd(sum_vec, sum_vec, 1);
        let final_sum = _mm_add_pd(sum_vec, sum_shuffled);
        
        let mut result = _mm_cvtsd_f64(final_sum);
        
        // Handle remainder elements
        for i in (chunks * 2)..len {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }
    
    /// Vector norm (L2) with SIMD optimization
    pub fn vector_norm(&self, vector: &Position) -> Result<f64, SwarmError> {
        if !self.enabled {
            return Ok(vector.norm());
        }
        
        let dot_product = self.dot_product(vector, vector)?;
        Ok(dot_product.sqrt())
    }
    
    /// Fused multiply-add operations for PSO velocity updates
    pub fn fused_multiply_add(
        &self,
        a: &Position,
        b: &Position,
        c: &Position,
        scalar: f64,
    ) -> Result<Position, SwarmError> {
        if !self.enabled {
            return Ok(a + scalar * (b - c));
        }
        
        if self.instruction_sets.has_fma && self.instruction_sets.has_avx2 && a.len() >= 4 {
            self.fused_multiply_add_fma(a, b, c, scalar)
        } else {
            Ok(a + scalar * (b - c))
        }
    }
    
    /// FMA-optimized fused multiply-add
    #[target_feature(enable = "fma,avx2")]
    unsafe fn fused_multiply_add_fma(
        &self,
        a: &Position,
        b: &Position,
        c: &Position,
        scalar: f64,
    ) -> Result<Position, SwarmError> {
        let len = a.len();
        let mut result = Position::zeros(len);
        
        let scalar_vec = _mm256_set1_pd(scalar);
        let chunks = len / 4;
        
        for i in 0..chunks {
            let offset = i * 4;
            
            let a_chunk = _mm256_loadu_pd(a.as_slice().as_ptr().add(offset));
            let b_chunk = _mm256_loadu_pd(b.as_slice().as_ptr().add(offset));
            let c_chunk = _mm256_loadu_pd(c.as_slice().as_ptr().add(offset));
            
            // Compute (b - c) * scalar + a using FMA
            let diff = _mm256_sub_pd(b_chunk, c_chunk);
            let fma_result = _mm256_fmadd_pd(diff, scalar_vec, a_chunk);
            
            _mm256_storeu_pd(result.as_mut_slice().as_mut_ptr().add(offset), fma_result);
        }
        
        // Handle remainder
        for i in (chunks * 4)..len {
            result[i] = a[i] + scalar * (b[i] - c[i]);
        }
        
        Ok(result)
    }
    
    /// Get SIMD capabilities
    pub fn capabilities(&self) -> &SimdCapabilities {
        &self.instruction_sets
    }
    
    /// Enable or disable SIMD operations
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
    
    /// Check if SIMD is enabled and available
    pub fn is_enabled(&self) -> bool {
        self.enabled && (self.instruction_sets.has_sse2 || self.instruction_sets.has_avx2)
    }
}

impl Default for SimdProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Initialize SIMD support for the swarm intelligence framework
pub fn initialize_simd_support() -> Result<(), SwarmError> {
    let processor = SimdProcessor::new();
    
    if !processor.is_enabled() {
        tracing::warn!("SIMD support not available or disabled");
        return Err(SwarmError::initialization("SIMD not available"));
    }
    
    let caps = processor.capabilities();
    tracing::info!(
        "SIMD initialized - AVX2: {}, AVX: {}, SSE4.2: {}, FMA: {}",
        caps.has_avx2,
        caps.has_avx,
        caps.has_sse4_2,
        caps.has_fma
    );
    
    Ok(())
}

/// Global SIMD processor instance
static mut GLOBAL_SIMD_PROCESSOR: Option<SimdProcessor> = None;
static SIMD_INIT: std::sync::Once = std::sync::Once::new();

/// Get the global SIMD processor
pub fn get_simd_processor() -> &'static SimdProcessor {
    unsafe {
        SIMD_INIT.call_once(|| {
            GLOBAL_SIMD_PROCESSOR = Some(SimdProcessor::new());
        });
        
        GLOBAL_SIMD_PROCESSOR.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simd_capabilities() {
        let processor = SimdProcessor::new();
        let caps = processor.capabilities();
        
        // At least SSE2 should be available on x86_64
        if cfg!(target_arch = "x86_64") {
            assert!(caps.has_sse2);
        }
    }
    
    #[test]
    fn test_vector_operations() {
        let processor = SimdProcessor::new();
        
        let a = Position::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Position::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
        
        // Test addition
        let sum = processor.vector_add(&a, &b).unwrap();
        let expected = Position::from_vec(vec![6.0, 8.0, 10.0, 12.0]);
        assert_relative_eq!(sum.as_slice(), expected.as_slice(), epsilon = 1e-10);
        
        // Test subtraction
        let diff = processor.vector_sub(&b, &a).unwrap();
        let expected_diff = Position::from_vec(vec![4.0, 4.0, 4.0, 4.0]);
        assert_relative_eq!(diff.as_slice(), expected_diff.as_slice(), epsilon = 1e-10);
        
        // Test scalar multiplication
        let scaled = processor.scalar_multiply(&a, 2.0).unwrap();
        let expected_scaled = Position::from_vec(vec![2.0, 4.0, 6.0, 8.0]);
        assert_relative_eq!(scaled.as_slice(), expected_scaled.as_slice(), epsilon = 1e-10);
        
        // Test dot product
        let dot = processor.dot_product(&a, &b).unwrap();
        let expected_dot = 1.0*5.0 + 2.0*6.0 + 3.0*7.0 + 4.0*8.0;
        assert_relative_eq!(dot, expected_dot, epsilon = 1e-10);
        
        // Test norm
        let norm = processor.vector_norm(&a).unwrap();
        let expected_norm = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert_relative_eq!(norm, expected_norm, epsilon = 1e-10);
    }
    
    #[test]
    fn test_fused_multiply_add() {
        let processor = SimdProcessor::new();
        
        let a = Position::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Position::from_vec(vec![5.0, 6.0, 7.0, 8.0]);
        let c = Position::from_vec(vec![2.0, 3.0, 4.0, 5.0]);
        
        let result = processor.fused_multiply_add(&a, &b, &c, 2.0).unwrap();
        
        // Expected: a + 2.0 * (b - c)
        let expected = Position::from_vec(vec![
            1.0 + 2.0 * (5.0 - 2.0), // 1 + 2*3 = 7
            2.0 + 2.0 * (6.0 - 3.0), // 2 + 2*3 = 8
            3.0 + 2.0 * (7.0 - 4.0), // 3 + 2*3 = 9
            4.0 + 2.0 * (8.0 - 5.0), // 4 + 2*3 = 10
        ]);
        
        assert_relative_eq!(result.as_slice(), expected.as_slice(), epsilon = 1e-10);
    }
    
    #[test]
    fn test_simd_vs_standard() {
        let mut processor = SimdProcessor::new();
        
        let a = Position::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let b = Position::from_vec(vec![6.0, 7.0, 8.0, 9.0, 10.0]);
        
        // SIMD result
        processor.set_enabled(true);
        let simd_sum = processor.vector_add(&a, &b).unwrap();
        let simd_dot = processor.dot_product(&a, &b).unwrap();
        
        // Standard result
        processor.set_enabled(false);
        let std_sum = processor.vector_add(&a, &b).unwrap();
        let std_dot = processor.dot_product(&a, &b).unwrap();
        
        // Should be identical
        assert_relative_eq!(simd_sum.as_slice(), std_sum.as_slice(), epsilon = 1e-12);
        assert_relative_eq!(simd_dot, std_dot, epsilon = 1e-12);
    }
}