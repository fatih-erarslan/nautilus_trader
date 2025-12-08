//! SIMD Operations for Advanced Detectors
//!
//! High-performance SIMD-optimized mathematical operations using f32x8 vectors
//! for sub-microsecond performance targets. This module provides optimized
//! implementations of common operations used across all detector modules.

use crate::*;

#[cfg(feature = "simd")]
use wide::{f32x8, CmpLe};

/// SIMD-optimized mathematical operations
pub struct SimdOps;

impl SimdOps {
    /// SIMD-optimized dot product
    #[cfg(feature = "simd")]
    pub fn dot_product_simd(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let mut sum = f32x8::splat(0.0);
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process 8 elements at a time
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let a_vec = f32x8::from([
                chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3],
                chunk_a[4], chunk_a[5], chunk_a[6], chunk_a[7]
            ]);
            let b_vec = f32x8::from([
                chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3],
                chunk_b[4], chunk_b[5], chunk_b[6], chunk_b[7]
            ]);
            
            sum = sum + (a_vec * b_vec);
        }
        
        // Sum the SIMD lanes
        let sum_array: [f32; 8] = sum.into();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Process remainder
        for (&ai, &bi) in remainder_a.iter().zip(remainder_b.iter()) {
            total += ai * bi;
        }
        
        Ok(total)
    }
    
    /// Fallback dot product for non-SIMD builds
    #[cfg(not(feature = "simd"))]
    pub fn dot_product_simd(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum())
    }
    
    /// SIMD-optimized element-wise addition
    #[cfg(feature = "simd")]
    pub fn add_arrays_simd(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let mut result = vec![0.0; a.len()];
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let mut result_chunks = result.chunks_exact_mut(8);
        
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process 8 elements at a time
        for ((chunk_a, chunk_b), result_chunk) in chunks_a.zip(chunks_b).zip(result_chunks.by_ref()) {
            let a_vec = f32x8::from([
                chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3],
                chunk_a[4], chunk_a[5], chunk_a[6], chunk_a[7]
            ]);
            let b_vec = f32x8::from([
                chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3],
                chunk_b[4], chunk_b[5], chunk_b[6], chunk_b[7]
            ]);
            
            let sum_vec = a_vec + b_vec;
            let sum_array: [f32; 8] = sum_vec.into();
            result_chunk.copy_from_slice(&sum_array);
        }
        
        // Process remainder
        let remainder_start = a.len() - remainder_a.len();
        for (i, (&ai, &bi)) in remainder_a.iter().zip(remainder_b.iter()).enumerate() {
            result[remainder_start + i] = ai + bi;
        }
        
        Ok(result)
    }
    
    /// Fallback element-wise addition
    #[cfg(not(feature = "simd"))]
    pub fn add_arrays_simd(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect())
    }
    
    /// SIMD-optimized element-wise multiplication
    #[cfg(feature = "simd")]
    pub fn multiply_arrays_simd(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let mut result = vec![0.0; a.len()];
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let mut result_chunks = result.chunks_exact_mut(8);
        
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process 8 elements at a time
        for ((chunk_a, chunk_b), result_chunk) in chunks_a.zip(chunks_b).zip(result_chunks.by_ref()) {
            let a_vec = f32x8::from([
                chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3],
                chunk_a[4], chunk_a[5], chunk_a[6], chunk_a[7]
            ]);
            let b_vec = f32x8::from([
                chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3],
                chunk_b[4], chunk_b[5], chunk_b[6], chunk_b[7]
            ]);
            
            let prod_vec = a_vec * b_vec;
            let prod_array: [f32; 8] = prod_vec.into();
            result_chunk.copy_from_slice(&prod_array);
        }
        
        // Process remainder
        let remainder_start = a.len() - remainder_a.len();
        for (i, (&ai, &bi)) in remainder_a.iter().zip(remainder_b.iter()).enumerate() {
            result[remainder_start + i] = ai * bi;
        }
        
        Ok(result)
    }
    
    /// Fallback element-wise multiplication
    #[cfg(not(feature = "simd"))]
    pub fn multiply_arrays_simd(a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        Ok(a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).collect())
    }
    
    /// SIMD-optimized scalar multiplication
    #[cfg(feature = "simd")]
    pub fn scalar_multiply_simd(array: &[f32], scalar: f32) -> Vec<f32> {
        let mut result = vec![0.0; array.len()];
        let scalar_vec = f32x8::splat(scalar);
        
        let chunks = array.chunks_exact(8);
        let mut result_chunks = result.chunks_exact_mut(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time
        for (chunk, result_chunk) in chunks.zip(result_chunks.by_ref()) {
            let array_vec = f32x8::from([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7]
            ]);
            
            let prod_vec = array_vec * scalar_vec;
            let prod_array: [f32; 8] = prod_vec.into();
            result_chunk.copy_from_slice(&prod_array);
        }
        
        // Process remainder
        let remainder_start = array.len() - remainder.len();
        for (i, &ai) in remainder.iter().enumerate() {
            result[remainder_start + i] = ai * scalar;
        }
        
        result
    }
    
    /// Fallback scalar multiplication
    #[cfg(not(feature = "simd"))]
    pub fn scalar_multiply_simd(array: &[f32], scalar: f32) -> Vec<f32> {
        array.iter().map(|&x| x * scalar).collect()
    }
    
    /// SIMD-optimized sum of squares
    #[cfg(feature = "simd")]
    pub fn sum_of_squares_simd(array: &[f32]) -> f32 {
        let mut sum = f32x8::splat(0.0);
        let chunks = array.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time
        for chunk in chunks {
            let vec = f32x8::from([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7]
            ]);
            
            sum = sum + (vec * vec);
        }
        
        // Sum the SIMD lanes
        let sum_array: [f32; 8] = sum.into();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Process remainder
        for &x in remainder {
            total += x * x;
        }
        
        total
    }
    
    /// Fallback sum of squares
    #[cfg(not(feature = "simd"))]
    pub fn sum_of_squares_simd(array: &[f32]) -> f32 {
        array.iter().map(|&x| x * x).sum()
    }
    
    /// SIMD-optimized distance calculation (Euclidean)
    #[cfg(feature = "simd")]
    pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let mut sum_sq_diff = f32x8::splat(0.0);
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process 8 elements at a time
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let a_vec = f32x8::from([
                chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3],
                chunk_a[4], chunk_a[5], chunk_a[6], chunk_a[7]
            ]);
            let b_vec = f32x8::from([
                chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3],
                chunk_b[4], chunk_b[5], chunk_b[6], chunk_b[7]
            ]);
            
            let diff = a_vec - b_vec;
            sum_sq_diff = sum_sq_diff + (diff * diff);
        }
        
        // Sum the SIMD lanes
        let sum_array: [f32; 8] = sum_sq_diff.into();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Process remainder
        for (&ai, &bi) in remainder_a.iter().zip(remainder_b.iter()) {
            let diff = ai - bi;
            total += diff * diff;
        }
        
        Ok(total.sqrt())
    }
    
    /// Fallback Euclidean distance
    #[cfg(not(feature = "simd"))]
    pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let sum_sq_diff: f32 = a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum();
        
        Ok(sum_sq_diff.sqrt())
    }
    
    /// SIMD-optimized Manhattan distance
    #[cfg(feature = "simd")]
    pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let mut sum_abs_diff = f32x8::splat(0.0);
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process 8 elements at a time
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let a_vec = f32x8::from([
                chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3],
                chunk_a[4], chunk_a[5], chunk_a[6], chunk_a[7]
            ]);
            let b_vec = f32x8::from([
                chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3],
                chunk_b[4], chunk_b[5], chunk_b[6], chunk_b[7]
            ]);
            
            let diff = a_vec - b_vec;
            let abs_diff = diff.abs();
            sum_abs_diff = sum_abs_diff + abs_diff;
        }
        
        // Sum the SIMD lanes
        let sum_array: [f32; 8] = sum_abs_diff.into();
        let mut total = sum_array.iter().sum::<f32>();
        
        // Process remainder
        for (&ai, &bi) in remainder_a.iter().zip(remainder_b.iter()) {
            total += (ai - bi).abs();
        }
        
        Ok(total)
    }
    
    /// Fallback Manhattan distance
    #[cfg(not(feature = "simd"))]
    pub fn manhattan_distance_simd(a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        Ok(a.iter()
           .zip(b.iter())
           .map(|(&ai, &bi)| (ai - bi).abs())
           .sum())
    }
    
    /// SIMD-optimized min/max finding
    #[cfg(feature = "simd")]
    pub fn find_min_max_simd(array: &[f32]) -> (f32, f32) {
        if array.is_empty() {
            return (0.0, 0.0);
        }
        
        let mut min_vec = f32x8::splat(f32::INFINITY);
        let mut max_vec = f32x8::splat(f32::NEG_INFINITY);
        
        let chunks = array.chunks_exact(8);
        let remainder = chunks.remainder();
        
        // Process 8 elements at a time
        for chunk in chunks {
            let vec = f32x8::from([
                chunk[0], chunk[1], chunk[2], chunk[3],
                chunk[4], chunk[5], chunk[6], chunk[7]
            ]);
            
            min_vec = min_vec.min(vec);
            max_vec = max_vec.max(vec);
        }
        
        // Find min/max in SIMD lanes
        let min_array: [f32; 8] = min_vec.into();
        let max_array: [f32; 8] = max_vec.into();
        
        let mut min_val = min_array.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let mut max_val = max_array.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        
        // Process remainder
        for &x in remainder {
            min_val = min_val.min(x);
            max_val = max_val.max(x);
        }
        
        (min_val, max_val)
    }
    
    /// Fallback min/max finding
    #[cfg(not(feature = "simd"))]
    pub fn find_min_max_simd(array: &[f32]) -> (f32, f32) {
        if array.is_empty() {
            return (0.0, 0.0);
        }
        
        let min_val = array.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let max_val = array.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        
        (min_val, max_val)
    }
    
    /// SIMD-optimized array comparison (element-wise)
    #[cfg(feature = "simd")]
    pub fn compare_arrays_simd(a: &[f32], b: &[f32], tolerance: f32) -> Result<Vec<bool>> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        let mut result = vec![false; a.len()];
        let tolerance_vec = f32x8::splat(tolerance);
        
        let chunks_a = a.chunks_exact(8);
        let chunks_b = b.chunks_exact(8);
        let mut result_chunks = result.chunks_exact_mut(8);
        
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();
        
        // Process 8 elements at a time
        for ((chunk_a, chunk_b), result_chunk) in chunks_a.zip(chunks_b).zip(result_chunks.by_ref()) {
            let a_vec = f32x8::from([
                chunk_a[0], chunk_a[1], chunk_a[2], chunk_a[3],
                chunk_a[4], chunk_a[5], chunk_a[6], chunk_a[7]
            ]);
            let b_vec = f32x8::from([
                chunk_b[0], chunk_b[1], chunk_b[2], chunk_b[3],
                chunk_b[4], chunk_b[5], chunk_b[6], chunk_b[7]
            ]);
            
            let diff = (a_vec - b_vec).abs();
            let comparison = diff.cmp_le(tolerance_vec);
            
            // Convert SIMD mask to boolean array
            let comparison_array: [f32; 8] = comparison.into();
            for i in 0..8 {
                result_chunk[i] = comparison_array[i] != 0.0;
            }
        }
        
        // Process remainder
        let remainder_start = a.len() - remainder_a.len();
        for (i, (&ai, &bi)) in remainder_a.iter().zip(remainder_b.iter()).enumerate() {
            result[remainder_start + i] = (ai - bi).abs() <= tolerance;
        }
        
        Ok(result)
    }
    
    /// Fallback array comparison
    #[cfg(not(feature = "simd"))]
    pub fn compare_arrays_simd(a: &[f32], b: &[f32], tolerance: f32) -> Result<Vec<bool>> {
        if a.len() != b.len() {
            return Err(DetectorError::SimdError {
                message: "Arrays must have the same length".to_string()
            });
        }
        
        Ok(a.iter()
           .zip(b.iter())
           .map(|(&ai, &bi)| (ai - bi).abs() <= tolerance)
           .collect())
    }
}

/// Cache-optimized operations for better memory access patterns
pub struct CacheOptimizedOps;

impl CacheOptimizedOps {
    /// Transpose matrix for better cache locality
    pub fn transpose_matrix(matrix: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>> {
        if matrix.len() != rows * cols {
            return Err(DetectorError::InvalidInput {
                message: "Matrix dimensions don't match data length".to_string()
            });
        }
        
        let mut transposed = vec![0.0; rows * cols];
        
        for i in 0..rows {
            for j in 0..cols {
                transposed[j * rows + i] = matrix[i * cols + j];
            }
        }
        
        Ok(transposed)
    }
    
    /// Blocked matrix multiplication for better cache performance
    pub fn blocked_matrix_multiply(
        a: &[f32], b: &[f32], 
        rows_a: usize, cols_a: usize, cols_b: usize,
        block_size: usize
    ) -> Result<Vec<f32>> {
        if a.len() != rows_a * cols_a || b.len() != cols_a * cols_b {
            return Err(DetectorError::InvalidInput {
                message: "Matrix dimensions don't match".to_string()
            });
        }
        
        let mut result = vec![0.0; rows_a * cols_b];
        
        for i_block in (0..rows_a).step_by(block_size) {
            for j_block in (0..cols_b).step_by(block_size) {
                for k_block in (0..cols_a).step_by(block_size) {
                    
                    let i_end = (i_block + block_size).min(rows_a);
                    let j_end = (j_block + block_size).min(cols_b);
                    let k_end = (k_block + block_size).min(cols_a);
                    
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;
                            for k in k_block..k_end {
                                sum += a[i * cols_a + k] * b[k * cols_b + j];
                            }
                            result[i * cols_b + j] += sum;
                        }
                    }
                }
            }
        }
        
        Ok(result)
    }
    
    /// Prefetch data for better cache performance
    #[cfg(target_arch = "x86_64")]
    pub fn prefetch_data(data: &[f32], stride: usize) {
        for i in (0..data.len()).step_by(stride) {
            unsafe {
                std::arch::x86_64::_mm_prefetch(
                    data.as_ptr().add(i) as *const i8,
                    std::arch::x86_64::_MM_HINT_T0
                );
            }
        }
    }
    
    /// Fallback for non-x86_64 architectures
    #[cfg(not(target_arch = "x86_64"))]
    pub fn prefetch_data(_data: &[f32], _stride: usize) {
        // No-op on other architectures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let result = SimdOps::dot_product_simd(&a, &b).unwrap();
        let expected: f32 = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum();
        
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_add_arrays() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        
        let result = SimdOps::add_arrays_simd(&a, &b).unwrap();
        let expected: Vec<f32> = a.iter().zip(b.iter()).map(|(&ai, &bi)| ai + bi).collect();
        
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_simd_scalar_multiply() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scalar = 2.5;
        
        let result = SimdOps::scalar_multiply_simd(&a, scalar);
        let expected: Vec<f32> = a.iter().map(|&x| x * scalar).collect();
        
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_simd_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        
        let result = SimdOps::euclidean_distance_simd(&a, &b).unwrap();
        let expected = ((5.0-1.0).powi(2) + (6.0-2.0).powi(2) + (7.0-3.0).powi(2) + (8.0-4.0).powi(2)).sqrt();
        
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_find_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        
        let (min_val, max_val) = SimdOps::find_min_max_simd(&data);
        
        assert_relative_eq!(min_val, 1.0, epsilon = 1e-6);
        assert_relative_eq!(max_val, 9.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_cache_optimized_transpose() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let transposed = CacheOptimizedOps::transpose_matrix(&matrix, 2, 3).unwrap();
        
        let expected = vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 3x2 matrix
        
        for (r, e) in transposed.iter().zip(expected.iter()) {
            assert_relative_eq!(r, e, epsilon = 1e-6);
        }
    }
}