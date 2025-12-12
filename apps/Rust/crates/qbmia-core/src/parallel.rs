//! Parallel processing utilities for QBMIA operations

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use crate::error::Result;

/// Parallel computation utilities
pub struct ParallelOps;

impl ParallelOps {
    /// Parallel vector operation processing
    #[cfg(feature = "parallel")]
    pub fn parallel_vector_op<F>(data: &[f64], chunk_size: usize, op: F) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64> + Sync + Send,
    {
        let results: Vec<_> = data
            .par_chunks(chunk_size)
            .map(|chunk| op(chunk))
            .collect();

        let mut final_result = Vec::new();
        for mut result in results {
            final_result.append(&mut result);
        }

        Ok(final_result)
    }

    /// Fallback sequential processing
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_vector_op<F>(data: &[f64], chunk_size: usize, op: F) -> Result<Vec<f64>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let mut final_result = Vec::new();
        
        for chunk in data.chunks(chunk_size) {
            let mut result = op(chunk);
            final_result.append(&mut result);
        }

        Ok(final_result)
    }

    /// Parallel matrix operations
    #[cfg(feature = "parallel")]
    pub fn parallel_matrix_op<F>(matrix: &[Vec<f64>], op: F) -> Result<Vec<Vec<f64>>>
    where
        F: Fn(&[f64]) -> Vec<f64> + Sync + Send,
    {
        let results: Vec<_> = matrix
            .par_iter()
            .map(|row| op(row))
            .collect();

        Ok(results)
    }

    /// Fallback sequential matrix operations
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_matrix_op<F>(matrix: &[Vec<f64>], op: F) -> Result<Vec<Vec<f64>>>
    where
        F: Fn(&[f64]) -> Vec<f64>,
    {
        let results: Vec<_> = matrix
            .iter()
            .map(|row| op(row))
            .collect();

        Ok(results)
    }

    /// Parallel reduction operation
    #[cfg(feature = "parallel")]
    pub fn parallel_reduce<T, F, R>(data: &[T], identity: R, reduce_op: F) -> Result<R>
    where
        T: Sync,
        F: Fn(R, &T) -> R + Sync + Send,
        R: Send + Clone + Sync,
    {
        let result = data
            .par_iter()
            .fold(|| identity.clone(), |acc, item| reduce_op(acc, item))
            .reduce(|| identity.clone(), |a, b| {
                // Since we need to combine two R values, we use the first one as base
                a
            });

        Ok(result)
    }

    /// Fallback sequential reduction
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_reduce<T, F, R>(data: &[T], identity: R, reduce_op: F) -> Result<R>
    where
        F: Fn(R, &T) -> R,
        R: Clone,
    {
        let result = data.iter().fold(identity, reduce_op);
        Ok(result)
    }

    /// Parallel map operation
    #[cfg(feature = "parallel")]
    pub fn parallel_map<T, U, F>(data: &[T], map_op: F) -> Result<Vec<U>>
    where
        T: Sync,
        U: Send,
        F: Fn(&T) -> U + Sync + Send,
    {
        let results: Vec<_> = data
            .par_iter()
            .map(map_op)
            .collect();

        Ok(results)
    }

    /// Fallback sequential map
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_map<T, U, F>(data: &[T], map_op: F) -> Result<Vec<U>>
    where
        F: Fn(&T) -> U,
    {
        let results: Vec<_> = data
            .iter()
            .map(map_op)
            .collect();

        Ok(results)
    }

    /// Parallel filter operation
    #[cfg(feature = "parallel")]
    pub fn parallel_filter<T, F>(data: &[T], filter_op: F) -> Result<Vec<T>>
    where
        T: Sync + Clone + Send,
        F: Fn(&T) -> bool + Sync + Send,
    {
        let results: Vec<_> = data
            .par_iter()
            .filter(|item| filter_op(item))
            .cloned()
            .collect();

        Ok(results)
    }

    /// Fallback sequential filter
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_filter<T, F>(data: &[T], filter_op: F) -> Result<Vec<T>>
    where
        T: Clone,
        F: Fn(&T) -> bool,
    {
        let results: Vec<_> = data
            .iter()
            .filter(|item| filter_op(item))
            .cloned()
            .collect();

        Ok(results)
    }

    /// Parallel dot product calculation
    #[cfg(feature = "parallel")]
    pub fn parallel_dot_product(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for parallel dot product"));
        }

        let chunk_size = (a.len() / rayon::current_num_threads()).max(1000);
        
        let result: f64 = a
            .par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .map(|(a_chunk, b_chunk)| {
                a_chunk.iter().zip(b_chunk.iter()).map(|(x, y)| x * y).sum::<f64>()
            })
            .sum();

        Ok(result)
    }

    /// Fallback sequential dot product
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_dot_product(a: &[f64], b: &[f64]) -> Result<f64> {
        if a.len() != b.len() {
            return Err(crate::error::QBMIAError::numerical("Vector length mismatch for dot product"));
        }

        let result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        Ok(result)
    }

    /// Parallel matrix multiplication
    #[cfg(feature = "parallel")]
    pub fn parallel_matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let rows_a = a.len();
        let cols_a = if rows_a > 0 { a[0].len() } else { 0 };
        let rows_b = b.len();
        let cols_b = if rows_b > 0 { b[0].len() } else { 0 };

        if cols_a != rows_b {
            return Err(crate::error::QBMIAError::numerical("Matrix multiplication dimension mismatch"));
        }

        // Transpose matrix B for better cache locality
        let b_transposed: Vec<Vec<f64>> = (0..cols_b)
            .map(|j| (0..rows_b).map(|i| b[i][j]).collect())
            .collect();

        let result: Vec<_> = a
            .par_iter()
            .map(|row_a| {
                b_transposed
                    .iter()
                    .map(|col_b| {
                        row_a.iter().zip(col_b.iter()).map(|(x, y)| x * y).sum()
                    })
                    .collect()
            })
            .collect();

        Ok(result)
    }

    /// Fallback sequential matrix multiplication
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let rows_a = a.len();
        let cols_a = if rows_a > 0 { a[0].len() } else { 0 };
        let rows_b = b.len();
        let cols_b = if rows_b > 0 { b[0].len() } else { 0 };

        if cols_a != rows_b {
            return Err(crate::error::QBMIAError::numerical("Matrix multiplication dimension mismatch"));
        }

        let mut result = vec![vec![0.0; cols_b]; rows_a];

        for i in 0..rows_a {
            for j in 0..cols_b {
                for k in 0..cols_a {
                    result[i][j] += a[i][k] * b[k][j];
                }
            }
        }

        Ok(result)
    }

    /// Parallel sum calculation
    #[cfg(feature = "parallel")]
    pub fn parallel_sum(data: &[f64]) -> Result<f64> {
        let chunk_size = (data.len() / rayon::current_num_threads()).max(1000);
        
        let result: f64 = data
            .par_chunks(chunk_size)
            .map(|chunk| chunk.iter().sum::<f64>())
            .sum();

        Ok(result)
    }

    /// Fallback sequential sum
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_sum(data: &[f64]) -> Result<f64> {
        Ok(data.iter().sum())
    }

    /// Check if parallel processing is available
    pub fn is_parallel_available() -> bool {
        #[cfg(feature = "parallel")]
        {
            true
        }
        #[cfg(not(feature = "parallel"))]
        {
            false
        }
    }

    /// Get optimal chunk size for parallel processing
    #[cfg(feature = "parallel")]
    pub fn optimal_chunk_size(data_len: usize) -> usize {
        let num_threads = rayon::current_num_threads();
        (data_len / num_threads).max(1000).min(10000)
    }

    /// Fallback chunk size calculation
    #[cfg(not(feature = "parallel"))]
    pub fn optimal_chunk_size(data_len: usize) -> usize {
        data_len.min(10000)
    }

    /// Configure parallel thread pool
    #[cfg(feature = "parallel")]
    pub fn configure_thread_pool(num_threads: Option<usize>) -> Result<()> {
        if let Some(threads) = num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .map_err(|e| crate::error::QBMIAError::config(format!("Failed to configure thread pool: {}", e)))?;
        }
        Ok(())
    }

    /// No-op for sequential version
    #[cfg(not(feature = "parallel"))]
    pub fn configure_thread_pool(_num_threads: Option<usize>) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ParallelOps::parallel_sum(&data).unwrap();
        assert_eq!(result, 15.0);
    }

    #[test]
    fn test_parallel_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = ParallelOps::parallel_dot_product(&a, &b).unwrap();
        assert_eq!(result, 32.0);
    }

    #[test]
    fn test_parallel_map() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ParallelOps::parallel_map(&data, |x| x * 2.0).unwrap();
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_parallel_filter() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = ParallelOps::parallel_filter(&data, |x| *x > 3.0).unwrap();
        assert_eq!(result, vec![4.0, 5.0]);
    }

    #[test]
    fn test_parallel_reduce() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = ParallelOps::parallel_reduce(&data, 0.0, |acc, x| acc + x).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_parallel_matrix_multiply() {
        let a = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let b = vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        
        let result = ParallelOps::parallel_matrix_multiply(&a, &b).unwrap();
        let expected = vec![
            vec![19.0, 22.0],  // [1*5+2*7, 1*6+2*8]
            vec![43.0, 50.0],  // [3*5+4*7, 3*6+4*8]
        ];
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_vector_operation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = ParallelOps::parallel_vector_op(&data, 2, |chunk| {
            chunk.iter().map(|x| x * 2.0).collect()
        }).unwrap();
        
        assert_eq!(result, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn test_matrix_operation() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        
        let result = ParallelOps::parallel_matrix_op(&matrix, |row| {
            row.iter().map(|x| x * 2.0).collect()
        }).unwrap();
        
        let expected = vec![
            vec![2.0, 4.0, 6.0],
            vec![8.0, 10.0, 12.0],
        ];
        
        assert_eq!(result, expected);
    }

    #[test]
    fn test_parallel_availability() {
        // Should not panic regardless of parallel feature
        let _available = ParallelOps::is_parallel_available();
    }

    #[test]
    fn test_optimal_chunk_size() {
        let chunk_size = ParallelOps::optimal_chunk_size(10000);
        assert!(chunk_size > 0);
        assert!(chunk_size <= 10000);
    }

    #[test]
    fn test_thread_pool_configuration() {
        // Should not fail even if parallel feature is disabled
        assert!(ParallelOps::configure_thread_pool(Some(4)).is_ok());
        assert!(ParallelOps::configure_thread_pool(None).is_ok());
    }
}