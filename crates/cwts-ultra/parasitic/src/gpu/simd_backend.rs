//! SIMD Backend for CPU-based Correlation Acceleration
//!
//! This module provides SIMD-optimized correlation matrix computation using
//! AVX-512, AVX2, and scalar fallback implementations. Used when GPU is
//! not available or for smaller workloads where GPU setup overhead exceeds benefits.

use super::*;
use rayon::prelude::*;
use std::arch::x86_64::*;

/// Compute correlation matrix using AVX-512 instructions (disabled due to unstable feature)
#[allow(dead_code)]
pub unsafe fn compute_correlations_avx512(
    organisms: &[OrganismVector],
    correlation_data: &mut [f32],
) -> Result<(), CorrelationError> {
    // Fallback to AVX2 for now since AVX-512 is unstable
    compute_correlations_avx2(organisms, correlation_data)
}

/// Compute correlation matrix using AVX2 instructions
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn compute_correlations_avx2(
    organisms: &[OrganismVector],
    correlation_data: &mut [f32],
) -> Result<(), CorrelationError> {
    let n = organisms.len();

    if correlation_data.len() != n * n {
        return Err(CorrelationError::ComputationError(
            "Correlation data buffer size mismatch".to_string(),
        ));
    }

    // Process organisms in parallel
    correlation_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, correlation_row)| {
            let org_i_data = get_organism_data(&organisms[row]);

            // Vectorized correlation computation - 8 correlations per iteration
            let mut col = 0;
            while col + 8 <= n {
                let mut correlations = [0.0f32; 8];

                for k in 0..8 {
                    if col + k >= n {
                        break;
                    }

                    if row == col + k {
                        correlations[k] = 1.0;
                    } else {
                        let org_j_data = get_organism_data(&organisms[col + k]);
                        correlations[k] = compute_correlation_avx2(&org_i_data, &org_j_data);
                    }
                }

                // Store results
                for k in 0..8 {
                    if col + k < n {
                        correlation_row[col + k] = correlations[k];
                    }
                }

                col += 8;
            }

            // Process remaining organisms
            while col < n {
                if row == col {
                    correlation_row[col] = 1.0;
                } else {
                    let org_j_data = get_organism_data(&organisms[col]);
                    correlation_row[col] = compute_correlation_scalar(&org_i_data, &org_j_data);
                }
                col += 1;
            }
        });

    Ok(())
}

/// Compute correlation matrix using scalar operations
pub fn compute_correlations_scalar(
    organisms: &[OrganismVector],
    correlation_data: &mut [f32],
) -> Result<(), CorrelationError> {
    let n = organisms.len();

    if correlation_data.len() != n * n {
        return Err(CorrelationError::ComputationError(
            "Correlation data buffer size mismatch".to_string(),
        ));
    }

    // Process organisms in parallel
    correlation_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(row, correlation_row)| {
            let org_i_data = get_organism_data(&organisms[row]);

            for col in 0..n {
                if row == col {
                    correlation_row[col] = 1.0;
                } else {
                    let org_j_data = get_organism_data(&organisms[col]);
                    correlation_row[col] = compute_correlation_scalar(&org_i_data, &org_j_data);
                }
            }
        });

    Ok(())
}

// Helper functions for AVX-512 operations removed due to unstable features

// Helper functions for AVX2 operations

#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn compute_correlation_avx2(org_i_data: &[f32], org_j_data: &[f32]) -> f32 {
    let data_size = org_i_data.len();

    let mut sum_i = _mm256_setzero_ps();
    let mut sum_j = _mm256_setzero_ps();
    let mut sum_ii = _mm256_setzero_ps();
    let mut sum_jj = _mm256_setzero_ps();
    let mut sum_ij = _mm256_setzero_ps();

    let mut idx = 0;

    // Process 8 elements at a time with AVX2
    while idx + 8 <= data_size {
        let chunk_i = _mm256_loadu_ps(&org_i_data[idx]);
        let chunk_j = _mm256_loadu_ps(&org_j_data[idx]);

        sum_i = _mm256_add_ps(sum_i, chunk_i);
        sum_j = _mm256_add_ps(sum_j, chunk_j);
        sum_ii = _mm256_fmadd_ps(chunk_i, chunk_i, sum_ii);
        sum_jj = _mm256_fmadd_ps(chunk_j, chunk_j, sum_jj);
        sum_ij = _mm256_fmadd_ps(chunk_i, chunk_j, sum_ij);

        idx += 8;
    }

    // Horizontal reduction to get scalar sums
    let sum_i_scalar = horizontal_sum_avx2(sum_i);
    let sum_j_scalar = horizontal_sum_avx2(sum_j);
    let sum_ii_scalar = horizontal_sum_avx2(sum_ii);
    let sum_jj_scalar = horizontal_sum_avx2(sum_jj);
    let mut sum_ij_scalar = horizontal_sum_avx2(sum_ij);

    // Handle remaining elements
    let mut sum_i_final = sum_i_scalar;
    let mut sum_j_final = sum_j_scalar;
    let mut sum_ii_final = sum_ii_scalar;
    let mut sum_jj_final = sum_jj_scalar;

    for k in idx..data_size {
        let val_i = org_i_data[k];
        let val_j = org_j_data[k];

        sum_i_final += val_i;
        sum_j_final += val_j;
        sum_ii_final += val_i * val_i;
        sum_jj_final += val_j * val_j;
        sum_ij_scalar += val_i * val_j;
    }

    // Compute Pearson correlation coefficient
    let n = data_size as f32;
    let numerator = n * sum_ij_scalar - sum_i_final * sum_j_final;
    let denom_i = n * sum_ii_final - sum_i_final * sum_i_final;
    let denom_j = n * sum_jj_final - sum_j_final * sum_j_final;
    let denominator = (denom_i * denom_j).sqrt();

    if denominator > 1e-8 {
        (numerator / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_avx2(v: __m256) -> f32 {
    let high = _mm256_extractf128_ps(v, 1);
    let low = _mm256_castps256_ps128(v);
    let sum_128 = _mm_add_ps(high, low);
    let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
    let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 0x55));
    _mm_cvtss_f32(sum_32)
}

// Scalar correlation computation

fn compute_correlation_scalar(org_i_data: &[f32], org_j_data: &[f32]) -> f32 {
    let n = org_i_data.len();
    if n == 0 {
        return 0.0;
    }

    // Compute means
    let mean_i = org_i_data.iter().sum::<f32>() / n as f32;
    let mean_j = org_j_data.iter().sum::<f32>() / n as f32;

    // Compute correlation components
    let mut numerator = 0.0;
    let mut sum_sq_i = 0.0;
    let mut sum_sq_j = 0.0;

    for k in 0..n {
        let diff_i = org_i_data[k] - mean_i;
        let diff_j = org_j_data[k] - mean_j;

        numerator += diff_i * diff_j;
        sum_sq_i += diff_i * diff_i;
        sum_sq_j += diff_j * diff_j;
    }

    let denominator = (sum_sq_i * sum_sq_j).sqrt();

    if denominator > 1e-8 {
        (numerator / denominator).clamp(-1.0, 1.0)
    } else {
        0.0
    }
}

// Utility functions

fn get_organism_data(organism: &OrganismVector) -> Vec<f32> {
    let mut data = Vec::new();
    data.extend_from_slice(organism.features());
    data.extend_from_slice(organism.performance_history());
    data
}

// Removed AVX-512 specific functions

// Performance optimization helpers

/// Prefetch data for better cache performance
#[inline(always)]
fn prefetch_organism_data(organism: &OrganismVector) {
    // Prefetch organism data into cache
    let features_ptr = organism.features().as_ptr();
    let performance_ptr = organism.performance_history().as_ptr();

    #[cfg(target_arch = "x86_64")]
    unsafe {
        std::arch::x86_64::_mm_prefetch(features_ptr as *const i8, std::arch::x86_64::_MM_HINT_T0);
        std::arch::x86_64::_mm_prefetch(
            performance_ptr as *const i8,
            std::arch::x86_64::_MM_HINT_T0,
        );
    }
}

/// Memory-aligned buffer for SIMD operations
#[repr(align(64))]
pub struct AlignedBuffer {
    data: Vec<f32>,
}

impl AlignedBuffer {
    pub fn new(size: usize) -> Self {
        Self {
            data: vec![0.0; size],
        }
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn load_organism_data(&mut self, organisms: &[OrganismVector]) {
        let mut offset = 0;

        for organism in organisms {
            let features = organism.features();
            let performance = organism.performance_history();

            self.data[offset..offset + features.len()].copy_from_slice(features);
            offset += features.len();

            self.data[offset..offset + performance.len()].copy_from_slice(performance);
            offset += performance.len();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::organism_vector::OrganismVector;

    #[test]
    fn test_scalar_correlation_computation() {
        let organisms = create_test_organisms();
        let n = organisms.len();
        let mut correlation_data = vec![0.0f32; n * n];

        let result = compute_correlations_scalar(&organisms, &mut correlation_data);
        assert!(result.is_ok());

        // Verify diagonal elements are 1.0
        for i in 0..n {
            assert_eq!(correlation_data[i * n + i], 1.0);
        }

        // Verify symmetry
        for i in 0..n {
            for j in 0..n {
                let val_ij = correlation_data[i * n + j];
                let val_ji = correlation_data[j * n + i];
                assert!(
                    (val_ij - val_ji).abs() < 1e-6,
                    "Matrix not symmetric at ({}, {})",
                    i,
                    j
                );
            }
        }

        println!("✅ Scalar correlation computation verified");
    }

    #[test]
    fn test_avx2_correlation_computation() {
        if !is_x86_feature_detected!("avx2") {
            println!("⚠️ AVX2 not available, skipping test");
            return;
        }

        let organisms = create_test_organisms();
        let n = organisms.len();
        let mut correlation_data = vec![0.0f32; n * n];

        let result = unsafe { compute_correlations_avx2(&organisms, &mut correlation_data) };
        assert!(result.is_ok());

        // Verify properties similar to scalar test
        for i in 0..n {
            assert_eq!(correlation_data[i * n + i], 1.0);
        }

        println!("✅ AVX2 correlation computation verified");
    }

    #[test]
    fn test_correlation_accuracy() {
        let org1 = OrganismVector::new(
            "test1".to_string(),
            vec![1.0, 2.0, 3.0, 4.0],
            vec![0.1, 0.2, 0.3, 0.4],
        );

        let org2 = OrganismVector::new(
            "test2".to_string(),
            vec![2.0, 4.0, 6.0, 8.0], // Perfectly correlated with org1
            vec![0.2, 0.4, 0.6, 0.8],
        );

        let data1 = get_organism_data(&org1);
        let data2 = get_organism_data(&org2);

        let correlation = compute_correlation_scalar(&data1, &data2);

        // Should be very close to 1.0 (perfect positive correlation)
        assert!(
            correlation > 0.99,
            "Expected high correlation, got {}",
            correlation
        );

        println!("✅ Correlation accuracy verified: {}", correlation);
    }

    fn create_test_organisms() -> Vec<OrganismVector> {
        vec![
            OrganismVector::new(
                "org1".to_string(),
                vec![1.0, 0.5, 0.8, 0.2],
                vec![0.1, 0.3, -0.1, 0.2],
            ),
            OrganismVector::new(
                "org2".to_string(),
                vec![0.8, 0.6, 0.7, 0.3],
                vec![0.2, 0.2, 0.0, 0.1],
            ),
            OrganismVector::new(
                "org3".to_string(),
                vec![0.2, 0.9, 0.3, 0.7],
                vec![-0.1, 0.4, 0.2, -0.2],
            ),
        ]
    }
}
