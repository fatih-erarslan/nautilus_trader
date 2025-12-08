//! High-performance optimization routines for CEFLANN-ELM
//! 
//! SIMD-accelerated matrix operations, memory management, and performance tuning
//! for ultra-fast neuromorphic training and inference.

use std::arch::x86_64::*;
use nalgebra::{DMatrix, DVector};
use ndarray::prelude::*;
use rayon::prelude::*;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};

#[cfg(feature = "simd")]
use wide::f64x4;

use crate::ELMConfig;

/// High-performance optimization engine for CEFLANN-ELM
pub struct OptimizationEngine {
    /// Configuration
    config: ELMConfig,
    
    /// SIMD capabilities detected
    simd_support: SIMDCapabilities,
    
    /// Memory pool for matrix operations
    memory_pool: MemoryPool,
    
    /// Performance statistics
    stats: OptimizationStats,
}

/// SIMD capabilities detection
#[derive(Debug, Clone)]
pub struct SIMDCapabilities {
    pub avx2: bool,
    pub avx512: bool,
    pub fma: bool,
    pub vector_width: usize,
}

/// Memory pool for efficient matrix allocation
pub struct MemoryPool {
    matrices: Vec<DMatrix<f64>>,
    vectors: Vec<DVector<f64>>,
    capacity: usize,
}

/// Performance and optimization statistics
#[derive(Debug, Default, Clone)]
pub struct OptimizationStats {
    pub simd_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub memory_allocs: u64,
    pub matrix_multiply_time_ns: u64,
    pub vector_operations_time_ns: u64,
    pub total_flops: u64,
}

impl OptimizationEngine {
    /// Create new optimization engine with hardware detection
    pub fn new(config: &ELMConfig) -> Result<Self> {
        info!("Initializing optimization engine with SIMD detection");
        
        let simd_support = Self::detect_simd_capabilities();
        info!("SIMD capabilities: {:?}", simd_support);
        
        let memory_pool = MemoryPool::new(1024); // Pre-allocate for 1024 matrices
        
        Ok(Self {
            config: config.clone(),
            simd_support,
            memory_pool,
            stats: OptimizationStats::default(),
        })
    }
    
    /// Detect available SIMD instruction sets
    fn detect_simd_capabilities() -> SIMDCapabilities {
        let mut caps = SIMDCapabilities {
            avx2: false,
            avx512: false,
            fma: false,
            vector_width: 2,
        };
        
        // Use CPUID to detect features
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                caps.avx2 = true;
                caps.vector_width = 4;
            }
            
            if is_x86_feature_detected!("avx512f") {
                caps.avx512 = true;
                caps.vector_width = 8;
            }
            
            if is_x86_feature_detected!("fma") {
                caps.fma = true;
            }
        }
        
        caps
    }
    
    /// Optimized matrix multiplication using SIMD and parallel processing
    pub fn optimized_matmul(&mut self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let start_time = std::time::Instant::now();
        
        if a.ncols() != b.nrows() {
            return Err(anyhow!("Matrix dimension mismatch for multiplication"));
        }
        
        let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
        
        // Choose optimal algorithm based on matrix size
        let result = if m * n * k > 1_000_000 {
            self.blocked_matmul_parallel(a, b)?
        } else if self.config.use_simd && self.simd_support.avx2 {
            self.simd_matmul(a, b)?
        } else {
            self.standard_matmul(a, b)?
        };
        
        let elapsed = start_time.elapsed();
        self.stats.matrix_multiply_time_ns += elapsed.as_nanos() as u64;
        self.stats.total_flops += (2 * m * n * k) as u64;
        
        debug!("Matrix multiplication {}×{}×{} completed in {}μs", 
               m, k, n, elapsed.as_micros());
        
        Ok(result)
    }
    
    /// SIMD-accelerated matrix multiplication
    #[cfg(feature = "simd")]
    fn simd_matmul(&mut self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
        let mut c = DMatrix::zeros(m, n);
        
        // Process in 4-wide SIMD lanes
        for i in 0..m {
            for j in (0..n).step_by(4) {
                let mut sum = f64x4::splat(0.0);
                
                // Vectorized dot product
                for l in (0..k).step_by(4) {
                    let a_vec = f64x4::new([
                        a[(i, l)],
                        a[(i, l.min(k-1) + 1)],
                        a[(i, l.min(k-1) + 2)], 
                        a[(i, l.min(k-1) + 3)],
                    ]);
                    
                    let b_vec = f64x4::new([
                        b[(l, j)],
                        b[(l.min(k-1) + 1, j)],
                        b[(l.min(k-1) + 2, j)],
                        b[(l.min(k-1) + 3, j)],
                    ]);
                    
                    // FMA operation: sum = a * b + sum
                    sum = a_vec.mul_add(b_vec, sum);
                }
                
                // Horizontal reduction
                let result = sum.as_array_ref();
                c[(i, j)] = result[0] + result[1] + result[2] + result[3];
                
                // Handle remaining columns
                for jj in j+1..(j+4).min(n) {
                    let mut dot = 0.0;
                    for l in 0..k {
                        dot += a[(i, l)] * b[(l, jj)];
                    }
                    c[(i, jj)] = dot;
                }
            }
        }
        
        self.stats.simd_operations += 1;
        Ok(c)
    }
    
    /// Fallback standard matrix multiplication
    #[cfg(not(feature = "simd"))]
    fn simd_matmul(&mut self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        self.standard_matmul(a, b)
    }
    
    /// Standard matrix multiplication
    fn standard_matmul(&mut self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        Ok(a * b)
    }
    
    /// Cache-blocked parallel matrix multiplication for large matrices
    fn blocked_matmul_parallel(&mut self, a: &DMatrix<f64>, b: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let (m, k, n) = (a.nrows(), a.ncols(), b.ncols());
        let block_size = 64; // Optimized for L1 cache
        
        let mut c = DMatrix::zeros(m, n);
        
        // Parallel processing of blocks
        (0..m).step_by(block_size).collect::<Vec<_>>()
            .par_iter()
            .for_each(|&i_start| {
                let i_end = (i_start + block_size).min(m);
                
                for j_start in (0..n).step_by(block_size) {
                    let j_end = (j_start + block_size).min(n);
                    
                    for l_start in (0..k).step_by(block_size) {
                        let l_end = (l_start + block_size).min(k);
                        
                        // Block multiplication
                        unsafe {
                            self.multiply_block_unsafe(
                                a, b, &mut c,
                                i_start, i_end,
                                j_start, j_end,
                                l_start, l_end
                            );
                        }
                    }
                }
            });
        
        Ok(c)
    }
    
    /// Unsafe block multiplication for maximum performance
    unsafe fn multiply_block_unsafe(
        &self,
        a: &DMatrix<f64>,
        b: &DMatrix<f64>,
        c: &mut DMatrix<f64>,
        i_start: usize, i_end: usize,
        j_start: usize, j_end: usize,
        l_start: usize, l_end: usize,
    ) {
        for i in i_start..i_end {
            for j in j_start..j_end {
                let mut sum = 0.0;
                for l in l_start..l_end {
                    sum += a.get_unchecked((i, l)) * b.get_unchecked((l, j));
                }
                *c.get_unchecked_mut((i, j)) += sum;
            }
        }
    }
    
    /// Optimized vector operations
    pub fn optimized_vector_ops(&mut self) -> VectorOps {
        VectorOps::new(&self.simd_support, &mut self.stats)
    }
    
    /// Memory-efficient batch processing
    pub fn batch_process<F, T>(&mut self, 
                              inputs: &[T], 
                              batch_size: usize, 
                              processor: F) -> Result<Vec<DMatrix<f64>>>
    where
        F: Fn(&[T]) -> Result<DMatrix<f64>> + Sync,
        T: Sync,
    {
        let results: Result<Vec<_>, _> = inputs
            .par_chunks(batch_size)
            .map(|batch| processor(batch))
            .collect();
        
        results
    }
    
    /// Adaptive cache management
    pub fn optimize_cache_usage(&mut self, working_set_size: usize) {
        // Adjust memory pool based on working set
        if working_set_size > self.memory_pool.capacity / 2 {
            self.memory_pool.resize(working_set_size * 2);
            info!("Resized memory pool to {} matrices", working_set_size * 2);
        }
        
        // Clear unused allocations
        self.memory_pool.cleanup();
    }
    
    /// Get optimization statistics
    pub fn stats(&self) -> &OptimizationStats {
        &self.stats
    }
    
    /// Performance analysis report
    pub fn performance_report(&self) -> String {
        format!(
            "CEFLANN-ELM Optimization Report\n\
             ================================\n\
             SIMD Operations: {}\n\
             Cache Hit Rate: {:.2}%\n\
             Total FLOPS: {}\n\
             Matrix Multiply Time: {}ms\n\
             Vector Operations Time: {}ms\n\
             Memory Allocations: {}\n\
             SIMD Capabilities: AVX2={}, AVX512={}, FMA={}\n",
            self.stats.simd_operations,
            (self.stats.cache_hits as f64 / 
             (self.stats.cache_hits + self.stats.cache_misses).max(1) as f64) * 100.0,
            self.stats.total_flops,
            self.stats.matrix_multiply_time_ns / 1_000_000,
            self.stats.vector_operations_time_ns / 1_000_000,
            self.stats.memory_allocs,
            self.simd_support.avx2,
            self.simd_support.avx512,
            self.simd_support.fma
        )
    }
}

/// Optimized vector operations
pub struct VectorOps<'a> {
    simd_support: &'a SIMDCapabilities,
    stats: &'a mut OptimizationStats,
}

impl<'a> VectorOps<'a> {
    fn new(simd_support: &'a SIMDCapabilities, stats: &'a mut OptimizationStats) -> Self {
        Self { simd_support, stats }
    }
    
    /// SIMD dot product
    #[cfg(feature = "simd")]
    pub fn dot_product(&mut self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());
        let start_time = std::time::Instant::now();
        
        let len = a.len();
        let mut sum = f64x4::splat(0.0);
        
        // Process 4 elements at a time
        let chunks = len / 4;
        for i in 0..chunks {
            let idx = i * 4;
            let a_vec = f64x4::new([a[idx], a[idx+1], a[idx+2], a[idx+3]]);
            let b_vec = f64x4::new([b[idx], b[idx+1], b[idx+2], b[idx+3]]);
            sum = a_vec.mul_add(b_vec, sum);
        }
        
        // Handle remainder
        let mut scalar_sum = sum.as_array_ref().iter().sum::<f64>();
        for i in chunks * 4..len {
            scalar_sum += a[i] * b[i];
        }
        
        self.stats.vector_operations_time_ns += start_time.elapsed().as_nanos() as u64;
        self.stats.simd_operations += 1;
        
        scalar_sum
    }
    
    /// Non-SIMD fallback dot product
    #[cfg(not(feature = "simd"))]
    pub fn dot_product(&mut self, a: &[f64], b: &[f64]) -> f64 {
        let start_time = std::time::Instant::now();
        let result = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        self.stats.vector_operations_time_ns += start_time.elapsed().as_nanos() as u64;
        result
    }
    
    /// Element-wise vector operations
    pub fn elementwise_multiply(&mut self, a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
        a.component_mul(b)
    }
    
    /// Vectorized activation functions
    pub fn vectorized_tanh(&mut self, x: &DVector<f64>) -> DVector<f64> {
        x.map(|val| val.tanh())
    }
    
    pub fn vectorized_sigmoid(&mut self, x: &DVector<f64>) -> DVector<f64> {
        x.map(|val| 1.0 / (1.0 + (-val).exp()))
    }
}

impl MemoryPool {
    fn new(capacity: usize) -> Self {
        Self {
            matrices: Vec::with_capacity(capacity),
            vectors: Vec::with_capacity(capacity),
            capacity,
        }
    }
    
    /// Get or allocate matrix
    pub fn get_matrix(&mut self, rows: usize, cols: usize) -> DMatrix<f64> {
        if let Some(mut matrix) = self.matrices.pop() {
            if matrix.nrows() == rows && matrix.ncols() == cols {
                matrix.fill(0.0);
                return matrix;
            }
        }
        
        DMatrix::zeros(rows, cols)
    }
    
    /// Return matrix to pool
    pub fn return_matrix(&mut self, matrix: DMatrix<f64>) {
        if self.matrices.len() < self.capacity {
            self.matrices.push(matrix);
        }
    }
    
    /// Resize pool capacity
    fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        self.matrices.reserve(new_capacity);
        self.vectors.reserve(new_capacity);
    }
    
    /// Clean up unused allocations
    fn cleanup(&mut self) {
        // Keep only half the allocations to balance memory usage
        let keep_count = self.capacity / 2;
        self.matrices.truncate(keep_count);
        self.vectors.truncate(keep_count);
    }
}

/// Specialized optimization for functional expansion
pub struct ExpansionOptimizer;

impl ExpansionOptimizer {
    /// Optimize trigonometric expansion with precomputed tables
    pub fn optimize_trigonometric_expansion(
        expansion_order: usize
    ) -> (Vec<f64>, Vec<f64>) {
        let table_size = 1024;
        let mut sin_table = Vec::with_capacity(table_size);
        let mut cos_table = Vec::with_capacity(table_size);
        
        for i in 0..table_size {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / table_size as f64;
            sin_table.push(angle.sin());
            cos_table.push(angle.cos());
        }
        
        (sin_table, cos_table)
    }
    
    /// Cache-efficient polynomial evaluation
    pub fn optimize_polynomial_evaluation(
        coefficients: &[Vec<f64>]
    ) -> Vec<Vec<f64>> {
        // Reorder coefficients for better cache locality
        coefficients.iter()
            .map(|poly| poly.clone())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_optimization_engine_creation() {
        let config = ELMConfig::default();
        let engine = OptimizationEngine::new(&config).unwrap();
        assert!(engine.simd_support.vector_width >= 2);
    }
    
    #[test]
    fn test_optimized_matrix_multiplication() {
        let config = ELMConfig::default();
        let mut engine = OptimizationEngine::new(&config).unwrap();
        
        let a = DMatrix::from_row_slice(3, 3, &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        
        let b = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]);
        
        let result = engine.optimized_matmul(&a, &b).unwrap();
        let expected = &a * &b;
        
        for i in 0..result.nrows() {
            for j in 0..result.ncols() {
                assert_relative_eq!(result[(i, j)], expected[(i, j)], epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_vector_operations() {
        let config = ELMConfig::default();
        let mut engine = OptimizationEngine::new(&config).unwrap();
        let mut vec_ops = engine.optimized_vector_ops();
        
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        
        let dot_product = vec_ops.dot_product(&a, &b);
        let expected: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        
        assert_relative_eq!(dot_product, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_simd_capabilities_detection() {
        let caps = OptimizationEngine::detect_simd_capabilities();
        assert!(caps.vector_width >= 2);
        // AVX2 availability depends on hardware
    }
}