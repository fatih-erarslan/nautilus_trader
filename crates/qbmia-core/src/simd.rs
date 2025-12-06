//! SIMD optimization utilities for QBMIA calculations

#[cfg(feature = "simd")]
use wide::f64x4;
use crate::error::Result;

/// SIMD-optimized vector operations
pub struct SimdOps;

impl SimdOps {
    /// SIMD dot product calculation
    #[cfg(feature = "simd")]
    pub fn dot_product_simd(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for SIMD dot product"));
        }

        let mut result = 0.0;
        let chunks = a.len() / 4;
        
        // Process 4 elements at a time with SIMD
        for i in 0..chunks {
            let start_idx = i * 4;
            let a_chunk = f64x4::new([
                a[start_idx],
                a[start_idx + 1],
                a[start_idx + 2],
                a[start_idx + 3],
            ]);
            let b_chunk = f64x4::new([
                b[start_idx],
                b[start_idx + 1],
                b[start_idx + 2],
                b[start_idx + 3],
            ]);
            
            let product = a_chunk * b_chunk;
            result += product.to_array().iter().sum::<f64>();
        }
        
        // Handle remaining elements
        let remainder_start = chunks * 4;
        for i in remainder_start..a.len() {
            result += a[i] * b[i];
        }
        
        Ok(result)
    }

    /// Fallback dot product for when SIMD is not available
    #[cfg(not(feature = "simd"))]
    pub fn dot_product_simd(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for dot product"));
        }

        let result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(result)
    }

    /// SIMD vector addition
    #[cfg(feature = "simd")]
    pub fn vector_add_simd(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for SIMD addition"));
        }

        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;
        
        // Process 4 elements at a time with SIMD
        for i in 0..chunks {
            let start_idx = i * 4;
            let a_chunk = f64x4::new([
                a[start_idx],
                a[start_idx + 1],
                a[start_idx + 2],
                a[start_idx + 3],
            ]);
            let b_chunk = f64x4::new([
                b[start_idx],
                b[start_idx + 1],
                b[start_idx + 2],
                b[start_idx + 3],
            ]);
            
            let sum = a_chunk + b_chunk;
            result.extend_from_slice(&sum.to_array());
        }
        
        // Handle remaining elements
        let remainder_start = chunks * 4;
        for i in remainder_start..a.len() {
            result.push(a[i] + b[i]);
        }
        
        Ok(result)
    }

    /// Fallback vector addition
    #[cfg(not(feature = "simd"))]
    pub fn vector_add_simd(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for addition"));
        }

        let result = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        Ok(result)
    }

    /// SIMD matrix-vector multiplication
    #[cfg(feature = "simd")]
    pub fn matrix_vector_mul_simd(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>> {
        let rows = matrix.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        
        let cols = matrix[0].len();
        if cols != vector.len() {
            return Err(crate::error::QBMIAError::numerical("Matrix-vector dimension mismatch"));
        }

        let mut result = Vec::with_capacity(rows);
        
        for row in matrix {
            let dot_product = Self::dot_product_simd(row, vector)?;
            result.push(dot_product);
        }
        
        Ok(result)
    }

    /// Fallback matrix-vector multiplication
    #[cfg(not(feature = "simd"))]
    pub fn matrix_vector_mul_simd(matrix: &[Vec<f64>], vector: &[f64]) -> Result<Vec<f64>> {
        let rows = matrix.len();
        if rows == 0 {
            return Ok(Vec::new());
        }
        
        let cols = matrix[0].len();
        if cols != vector.len() {
            return Err(crate::error::QBMIAError::numerical("Matrix-vector dimension mismatch"));
        }

        let mut result = Vec::with_capacity(rows);
        
        for row in matrix {
            let dot_product = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
            result.push(dot_product);
        }
        
        Ok(result)
    }

    /// SIMD element-wise multiplication
    #[cfg(feature = "simd")]
    pub fn element_wise_mul_simd(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for element-wise multiplication"));
        }

        let mut result = Vec::with_capacity(a.len());
        let chunks = a.len() / 4;
        
        // Process 4 elements at a time with SIMD
        for i in 0..chunks {
            let start_idx = i * 4;
            let a_chunk = f64x4::new([
                a[start_idx],
                a[start_idx + 1],
                a[start_idx + 2],
                a[start_idx + 3],
            ]);
            let b_chunk = f64x4::new([
                b[start_idx],
                b[start_idx + 1],
                b[start_idx + 2],
                b[start_idx + 3],
            ]);
            
            let product = a_chunk * b_chunk;
            result.extend_from_slice(&product.to_array());
        }
        
        // Handle remaining elements
        let remainder_start = chunks * 4;
        for i in remainder_start..a.len() {
            result.push(a[i] * b[i]);
        }
        
        Ok(result)
    }

    /// Fallback element-wise multiplication
    #[cfg(not(feature = "simd"))]
    pub fn element_wise_mul_simd(a: &[f64], b: &[f64]) -> Result<Vec<f64>> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for element-wise multiplication"));
        }

        let result = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
        Ok(result)
    }

    /// Check if SIMD is available and working
    pub fn is_simd_available() -> bool {
        #[cfg(feature = "simd")]
        {
            // Test SIMD functionality
            let test_a = [1.0, 2.0, 3.0, 4.0];
            let test_b = [1.0, 1.0, 1.0, 1.0];
            Self::dot_product_simd(&test_a, &test_b).is_ok()
        }
        #[cfg(not(feature = "simd"))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = SimdOps::dot_product_simd(&a, &b).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_vector_addition() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 1.0, 1.0, 1.0];
        
        let result = SimdOps::vector_add_simd(&a, &b).unwrap();
        assert_eq!(result, vec![2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_element_wise_multiplication() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 2.0, 2.0, 2.0];
        
        let result = SimdOps::element_wise_mul_simd(&a, &b).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_vector_multiplication() {
        let matrix = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let vector = vec![1.0, 1.0];
        
        let result = SimdOps::matrix_vector_mul_simd(&matrix, &vector).unwrap();
        assert_eq!(result, vec![3.0, 7.0]);
    }

    #[test]
    fn test_simd_availability() {
        // This should not panic regardless of SIMD availability
        let _available = SimdOps::is_simd_available();
    }

    #[test]
    fn test_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 1.0, 1.0];
        
        assert!(SimdOps::dot_product_simd(&a, &b).is_err());
        assert!(SimdOps::vector_add_simd(&a, &b).is_err());
        assert!(SimdOps::element_wise_mul_simd(&a, &b).is_err());
    }
}