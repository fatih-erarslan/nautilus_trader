//! SIMD-optimized matrix operations for risk calculations
//!
//! This module implements vectorized matrix operations optimized for auto-vectorization
//! through manual loop unrolling and cache-friendly memory access patterns.

/// SIMD vector size for f64 operations (4 elements = 256 bits)
const SIMD_WIDTH: usize = 4;

/// Cache line size for optimal memory access
#[allow(dead_code)]
const CACHE_LINE_SIZE: usize = 64;

/// SIMD-optimized covariance matrix calculation
///
/// Computes the covariance matrix from multiple return series using vectorized operations.
///
/// # Arguments
/// * `returns` - Slice of return series, where each inner slice is a time series
///
/// # Returns
/// Symmetric covariance matrix as Vec<Vec<f64>>
pub fn simd_covariance_matrix(returns: &[&[f64]]) -> Vec<Vec<f64>> {
    let n_assets = returns.len();
    if n_assets == 0 {
        return Vec::new();
    }

    let n_periods = returns[0].len();
    if n_periods == 0 {
        return vec![vec![0.0; n_assets]; n_assets];
    }

    // Vectorized mean calculation
    let means: Vec<f64> = returns
        .iter()
        .map(|r| vectorized_mean(r))
        .collect();

    // Initialize covariance matrix
    let mut cov_matrix = vec![vec![0.0; n_assets]; n_assets];

    // Calculate covariance using cache-friendly access pattern
    for i in 0..n_assets {
        for j in i..n_assets {
            let cov = vectorized_covariance(returns[i], returns[j], means[i], means[j], n_periods);
            cov_matrix[i][j] = cov;
            cov_matrix[j][i] = cov; // Symmetry
        }
    }

    cov_matrix
}

/// SIMD-optimized correlation matrix calculation
///
/// Computes the correlation matrix from multiple return series.
///
/// # Arguments
/// * `returns` - Slice of return series
///
/// # Returns
/// Correlation matrix as Vec<Vec<f64>>
pub fn simd_correlation_matrix(returns: &[&[f64]]) -> Vec<Vec<f64>> {
    let cov_matrix = simd_covariance_matrix(returns);
    let n = cov_matrix.len();

    if n == 0 {
        return Vec::new();
    }

    let mut corr_matrix = vec![vec![0.0; n]; n];

    // Extract standard deviations (diagonal elements)
    let std_devs: Vec<f64> = (0..n)
        .map(|i| cov_matrix[i][i].sqrt())
        .collect();

    // Vectorized correlation calculation
    for i in 0..n {
        for j in 0..n {
            if std_devs[i] > 0.0 && std_devs[j] > 0.0 {
                corr_matrix[i][j] = cov_matrix[i][j] / (std_devs[i] * std_devs[j]);
            } else {
                corr_matrix[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }

    corr_matrix
}

/// SIMD-optimized matrix multiplication
///
/// Multiplies two square matrices using blocked algorithm for cache efficiency.
///
/// # Arguments
/// * `a` - First matrix (flattened, row-major)
/// * `b` - Second matrix (flattened, row-major)
/// * `n` - Matrix dimension (n×n)
///
/// # Returns
/// Result matrix (flattened, row-major)
pub fn simd_matrix_multiply(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(a.len(), n * n);
    assert_eq!(b.len(), n * n);

    let mut result = vec![0.0; n * n];

    // Block size for cache optimization
    const BLOCK_SIZE: usize = 64;

    // Blocked matrix multiplication for cache locality
    for i_block in (0..n).step_by(BLOCK_SIZE) {
        let i_end = (i_block + BLOCK_SIZE).min(n);

        for j_block in (0..n).step_by(BLOCK_SIZE) {
            let j_end = (j_block + BLOCK_SIZE).min(n);

            for k_block in (0..n).step_by(BLOCK_SIZE) {
                let k_end = (k_block + BLOCK_SIZE).min(n);

                // Inner block multiplication with vectorization
                for i in i_block..i_end {
                    for j in j_block..j_end {
                        let mut sum = result[i * n + j];

                        // Vectorized inner product
                        let mut k = k_block;
                        while k + SIMD_WIDTH <= k_end {
                            // Manual unrolling
                            sum += a[i * n + k] * b[k * n + j];
                            sum += a[i * n + k + 1] * b[(k + 1) * n + j];
                            sum += a[i * n + k + 2] * b[(k + 2) * n + j];
                            sum += a[i * n + k + 3] * b[(k + 3) * n + j];
                            k += SIMD_WIDTH;
                        }

                        // Handle remainder
                        while k < k_end {
                            sum += a[i * n + k] * b[k * n + j];
                            k += 1;
                        }

                        result[i * n + j] = sum;
                    }
                }
            }
        }
    }

    result
}

/// SIMD-optimized Cholesky decomposition
///
/// Computes the Cholesky decomposition of a symmetric positive-definite matrix.
/// Returns L such that A = L * L^T.
///
/// # Arguments
/// * `matrix` - Symmetric positive-definite matrix (flattened, row-major)
/// * `n` - Matrix dimension
///
/// # Returns
/// Lower triangular matrix L, or None if matrix is not positive-definite
pub fn simd_cholesky_decomposition(matrix: &[f64], n: usize) -> Option<Vec<f64>> {
    assert_eq!(matrix.len(), n * n);

    let mut l = vec![0.0; n * n];

    for i in 0..n {
        // Compute diagonal element L[i,i]
        let mut sum = 0.0;
        let mut k = 0;

        // Vectorized sum of squares
        while k + SIMD_WIDTH <= i {
            sum += l[i * n + k] * l[i * n + k];
            sum += l[i * n + k + 1] * l[i * n + k + 1];
            sum += l[i * n + k + 2] * l[i * n + k + 2];
            sum += l[i * n + k + 3] * l[i * n + k + 3];
            k += SIMD_WIDTH;
        }

        while k < i {
            let val = l[i * n + k];
            sum += val * val;
            k += 1;
        }

        let diag = matrix[i * n + i] - sum;
        if diag <= 1e-14 {
            return None; // Matrix not positive-definite
        }
        l[i * n + i] = diag.sqrt();

        // Compute off-diagonal elements in column i
        for j in (i + 1)..n {
            let mut sum = 0.0;
            let mut k = 0;

            // Vectorized inner product
            while k + SIMD_WIDTH <= i {
                sum += l[j * n + k] * l[i * n + k];
                sum += l[j * n + k + 1] * l[i * n + k + 1];
                sum += l[j * n + k + 2] * l[i * n + k + 2];
                sum += l[j * n + k + 3] * l[i * n + k + 3];
                k += SIMD_WIDTH;
            }

            while k < i {
                sum += l[j * n + k] * l[i * n + k];
                k += 1;
            }

            l[j * n + i] = (matrix[j * n + i] - sum) / l[i * n + i];
        }
    }

    Some(l)
}

/// Vectorized mean calculation
#[inline]
fn vectorized_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mut sum = 0.0;
    let mut i = 0;

    // Process 4 elements at a time
    while i + SIMD_WIDTH <= values.len() {
        sum += values[i] + values[i + 1] + values[i + 2] + values[i + 3];
        i += SIMD_WIDTH;
    }

    // Handle remainder
    while i < values.len() {
        sum += values[i];
        i += 1;
    }

    sum / values.len() as f64
}

/// Vectorized covariance calculation between two series
#[inline]
fn vectorized_covariance(x: &[f64], y: &[f64], mean_x: f64, mean_y: f64, n: usize) -> f64 {
    let mut cov = 0.0;
    let mut i = 0;

    // Process 4 elements at a time
    while i + SIMD_WIDTH <= n {
        let dx0 = x[i] - mean_x;
        let dy0 = y[i] - mean_y;

        let dx1 = x[i + 1] - mean_x;
        let dy1 = y[i + 1] - mean_y;

        let dx2 = x[i + 2] - mean_x;
        let dy2 = y[i + 2] - mean_y;

        let dx3 = x[i + 3] - mean_x;
        let dy3 = y[i + 3] - mean_y;

        cov += dx0 * dy0 + dx1 * dy1 + dx2 * dy2 + dx3 * dy3;
        i += SIMD_WIDTH;
    }

    // Handle remainder
    while i < n {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;
        cov += dx * dy;
        i += 1;
    }

    cov / (n - 1) as f64
}

/// Transpose a square matrix in-place using cache-blocked algorithm
#[allow(dead_code)]
fn transpose_square_blocked(matrix: &mut [f64], n: usize) {
    const BLOCK_SIZE: usize = 32;

    for i_block in (0..n).step_by(BLOCK_SIZE) {
        let i_end = (i_block + BLOCK_SIZE).min(n);

        for j_block in (i_block..n).step_by(BLOCK_SIZE) {
            let j_end = (j_block + BLOCK_SIZE).min(n);

            for i in i_block..i_end {
                for j in j_block..j_end {
                    if i < j {
                        let idx1 = i * n + j;
                        let idx2 = j * n + i;
                        matrix.swap(idx1, idx2);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_covariance_matrix_symmetry() {
        let returns1 = vec![0.01, 0.02, -0.01, 0.03];
        let returns2 = vec![0.02, -0.01, 0.01, 0.02];
        let returns: Vec<&[f64]> = vec![&returns1, &returns2];

        let cov = simd_covariance_matrix(&returns);

        assert_eq!(cov.len(), 2);
        assert_eq!(cov[0].len(), 2);

        // Check symmetry
        assert!((cov[0][1] - cov[1][0]).abs() < 1e-10);

        // Check diagonal is positive
        assert!(cov[0][0] > 0.0);
        assert!(cov[1][1] > 0.0);
    }

    #[test]
    fn test_correlation_matrix_properties() {
        let returns1 = vec![0.01, 0.02, -0.01, 0.03, 0.01];
        let returns2 = vec![0.02, -0.01, 0.01, 0.02, -0.01];
        let returns: Vec<&[f64]> = vec![&returns1, &returns2];

        let corr = simd_correlation_matrix(&returns);

        assert_eq!(corr.len(), 2);

        // Diagonal should be 1.0
        assert!((corr[0][0] - 1.0).abs() < 1e-10);
        assert!((corr[1][1] - 1.0).abs() < 1e-10);

        // Off-diagonal should be in [-1, 1]
        assert!(corr[0][1] >= -1.0 && corr[0][1] <= 1.0);
        assert!(corr[1][0] >= -1.0 && corr[1][0] <= 1.0);

        // Symmetry
        assert!((corr[0][1] - corr[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_multiply_identity() {
        let n = 3;
        let identity = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];

        let matrix = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];

        let result = simd_matrix_multiply(&matrix, &identity, n);

        // Multiplying by identity should give original matrix
        for i in 0..matrix.len() {
            assert!((result[i] - matrix[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_multiply_correctness() {
        let a = vec![
            1.0, 2.0,
            3.0, 4.0,
        ];

        let b = vec![
            5.0, 6.0,
            7.0, 8.0,
        ];

        let result = simd_matrix_multiply(&a, &b, 2);

        // Expected: [19, 22, 43, 50]
        assert!((result[0] - 19.0).abs() < 1e-10);
        assert!((result[1] - 22.0).abs() < 1e-10);
        assert!((result[2] - 43.0).abs() < 1e-10);
        assert!((result[3] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_cholesky_decomposition() {
        // Positive definite matrix: [[4, 2], [2, 3]]
        let matrix = vec![
            4.0, 2.0,
            2.0, 3.0,
        ];

        let l = simd_cholesky_decomposition(&matrix, 2).expect("Cholesky failed");

        // Verify L * L^T = A
        let reconstructed = simd_matrix_multiply(&l, &transpose_matrix(&l, 2), 2);

        for i in 0..matrix.len() {
            assert!((reconstructed[i] - matrix[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_cholesky_non_positive_definite() {
        // Non-positive definite matrix
        let matrix = vec![
            1.0, 2.0,
            2.0, 1.0,
        ];

        let result = simd_cholesky_decomposition(&matrix, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_large_matrix_multiply() {
        let n = 100;
        let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n * n).map(|i| i as f64 * 0.5).collect();

        let result = simd_matrix_multiply(&a, &b, n);
        assert_eq!(result.len(), n * n);
    }

    #[test]
    fn test_vectorized_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = vectorized_mean(&values);
        assert!((mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_covariance_calculation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];

        let mean_x = vectorized_mean(&x);
        let mean_y = vectorized_mean(&y);

        let cov = vectorized_covariance(&x, &y, mean_x, mean_y, x.len());

        // Perfect positive correlation should give cov = std_x * std_y
        assert!(cov > 0.0);
    }

    // Helper function for testing
    fn transpose_matrix(matrix: &[f64], n: usize) -> Vec<f64> {
        let mut result = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                result[j * n + i] = matrix[i * n + j];
            }
        }
        result
    }
}
