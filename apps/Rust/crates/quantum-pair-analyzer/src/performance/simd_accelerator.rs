// SIMD-Accelerated Quantum Pair Analysis for Sub-100ns Latency
// Copyright (c) 2025 TENGRI Trading Swarm - Performance-Optimizer Agent

use std::arch::x86_64::*;
use std::simd::{f32x16, f64x8, u32x16, StdFloat, Simd};
use nalgebra::{DMatrix, DVector};
use anyhow::Result;
use tracing::{info, debug};
use crate::AnalyzerError;

/// SIMD configuration for optimal performance
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable AVX-512 instructions
    pub avx512_enabled: bool,
    /// Enable AVX2 instructions
    pub avx2_enabled: bool,
    /// Enable SSE4.2 instructions
    pub sse42_enabled: bool,
    /// Vector width for operations
    pub vector_width: usize,
    /// Batch size for SIMD operations
    pub batch_size: usize,
    /// Enable FMA (Fused Multiply-Add)
    pub fma_enabled: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            avx512_enabled: is_x86_feature_detected!("avx512f"),
            avx2_enabled: is_x86_feature_detected!("avx2"),
            sse42_enabled: is_x86_feature_detected!("sse4.2"),
            vector_width: 64, // 512-bit vectors
            batch_size: 1024,
            fma_enabled: is_x86_feature_detected!("fma"),
        }
    }
}

/// High-performance SIMD accelerator for quantum operations
pub struct SimdAccelerator {
    config: SimdConfig,
    cpu_features: CpuFeatures,
}

/// CPU feature detection
#[derive(Debug, Clone)]
pub struct CpuFeatures {
    pub avx512f: bool,
    pub avx512bw: bool,
    pub avx512dq: bool,
    pub avx512vl: bool,
    pub avx2: bool,
    pub fma: bool,
    pub sse42: bool,
    pub bmi1: bool,
    pub bmi2: bool,
}

impl CpuFeatures {
    pub fn detect() -> Self {
        Self {
            avx512f: is_x86_feature_detected!("avx512f"),
            avx512bw: is_x86_feature_detected!("avx512bw"),
            avx512dq: is_x86_feature_detected!("avx512dq"),
            avx512vl: is_x86_feature_detected!("avx512vl"),
            avx2: is_x86_feature_detected!("avx2"),
            fma: is_x86_feature_detected!("fma"),
            sse42: is_x86_feature_detected!("sse4.2"),
            bmi1: is_x86_feature_detected!("bmi1"),
            bmi2: is_x86_feature_detected!("bmi2"),
        }
    }
}

impl SimdAccelerator {
    /// Create new SIMD accelerator
    pub fn new(config: SimdConfig) -> Result<Self, AnalyzerError> {
        let cpu_features = CpuFeatures::detect();
        
        info!("SIMD accelerator initialized with features: {:?}", cpu_features);
        
        Ok(Self {
            config,
            cpu_features,
        })
    }
    
    /// Calculate correlation matrix using SIMD
    pub fn calculate_correlation_matrix_simd(
        &self,
        data: &[f64],
        n_assets: usize,
    ) -> Result<DMatrix<f64>, AnalyzerError> {
        debug!("Calculating correlation matrix with SIMD for {} assets", n_assets);
        
        if self.cpu_features.avx512f {
            self.calculate_correlation_avx512(data, n_assets)
        } else if self.cpu_features.avx2 {
            self.calculate_correlation_avx2(data, n_assets)
        } else {
            self.calculate_correlation_sse42(data, n_assets)
        }
    }
    
    /// AVX-512 correlation calculation
    #[target_feature(enable = "avx512f")]
    unsafe fn calculate_correlation_avx512(
        &self,
        data: &[f64],
        n_assets: usize,
    ) -> Result<DMatrix<f64>, AnalyzerError> {
        let mut correlation_matrix = DMatrix::zeros(n_assets, n_assets);
        let data_len = data.len() / n_assets;
        
        // Process 8 correlations at once with AVX-512
        for i in 0..n_assets {
            let asset_i = &data[i * data_len..(i + 1) * data_len];
            let mean_i = self.calculate_mean_avx512(asset_i);
            let var_i = self.calculate_variance_avx512(asset_i, mean_i);
            
            for j in (i..n_assets).step_by(8) {
                let end_idx = (j + 8).min(n_assets);
                let batch_size = end_idx - j;
                
                let mut correlations = [0.0; 8];
                
                for k in 0..batch_size {
                    let asset_j = &data[(j + k) * data_len..((j + k) + 1) * data_len];
                    let mean_j = self.calculate_mean_avx512(asset_j);
                    let var_j = self.calculate_variance_avx512(asset_j, mean_j);
                    
                    let covariance = self.calculate_covariance_avx512(asset_i, asset_j, mean_i, mean_j);
                    correlations[k] = covariance / (var_i.sqrt() * var_j.sqrt());
                }
                
                // Store results
                for k in 0..batch_size {
                    correlation_matrix[(i, j + k)] = correlations[k];
                    correlation_matrix[(j + k, i)] = correlations[k];
                }
            }
        }
        
        Ok(correlation_matrix)
    }
    
    /// AVX-512 mean calculation
    #[target_feature(enable = "avx512f")]
    unsafe fn calculate_mean_avx512(&self, data: &[f64]) -> f64 {
        let mut sum = _mm512_setzero_pd();
        let len = data.len();
        
        // Process 8 elements at once
        for chunk in data.chunks_exact(8) {
            let values = _mm512_loadu_pd(chunk.as_ptr());
            sum = _mm512_add_pd(sum, values);
        }
        
        // Handle remaining elements
        let remainder = len % 8;
        if remainder > 0 {
            let mut remaining_sum = 0.0;
            for &value in &data[len - remainder..] {
                remaining_sum += value;
            }
            sum = _mm512_add_pd(sum, _mm512_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m512d, [f64; 8]>(sum);
        sum_array.iter().sum::<f64>() / len as f64
    }
    
    /// AVX-512 variance calculation
    #[target_feature(enable = "avx512f")]
    unsafe fn calculate_variance_avx512(&self, data: &[f64], mean: f64) -> f64 {
        let mut sum_sq_diff = _mm512_setzero_pd();
        let mean_vec = _mm512_set1_pd(mean);
        let len = data.len();
        
        // Process 8 elements at once
        for chunk in data.chunks_exact(8) {
            let values = _mm512_loadu_pd(chunk.as_ptr());
            let diff = _mm512_sub_pd(values, mean_vec);
            let sq_diff = _mm512_mul_pd(diff, diff);
            sum_sq_diff = _mm512_add_pd(sum_sq_diff, sq_diff);
        }
        
        // Handle remaining elements
        let remainder = len % 8;
        if remainder > 0 {
            let mut remaining_sum = 0.0;
            for &value in &data[len - remainder..] {
                let diff = value - mean;
                remaining_sum += diff * diff;
            }
            sum_sq_diff = _mm512_add_pd(sum_sq_diff, _mm512_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m512d, [f64; 8]>(sum_sq_diff);
        sum_array.iter().sum::<f64>() / (len - 1) as f64
    }
    
    /// AVX-512 covariance calculation
    #[target_feature(enable = "avx512f")]
    unsafe fn calculate_covariance_avx512(
        &self,
        data_x: &[f64],
        data_y: &[f64],
        mean_x: f64,
        mean_y: f64,
    ) -> f64 {
        let mut sum_prod_diff = _mm512_setzero_pd();
        let mean_x_vec = _mm512_set1_pd(mean_x);
        let mean_y_vec = _mm512_set1_pd(mean_y);
        let len = data_x.len();
        
        // Process 8 elements at once
        for (chunk_x, chunk_y) in data_x.chunks_exact(8).zip(data_y.chunks_exact(8)) {
            let values_x = _mm512_loadu_pd(chunk_x.as_ptr());
            let values_y = _mm512_loadu_pd(chunk_y.as_ptr());
            
            let diff_x = _mm512_sub_pd(values_x, mean_x_vec);
            let diff_y = _mm512_sub_pd(values_y, mean_y_vec);
            let prod_diff = _mm512_mul_pd(diff_x, diff_y);
            
            sum_prod_diff = _mm512_add_pd(sum_prod_diff, prod_diff);
        }
        
        // Handle remaining elements
        let remainder = len % 8;
        if remainder > 0 {
            let mut remaining_sum = 0.0;
            for i in len - remainder..len {
                remaining_sum += (data_x[i] - mean_x) * (data_y[i] - mean_y);
            }
            sum_prod_diff = _mm512_add_pd(sum_prod_diff, _mm512_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m512d, [f64; 8]>(sum_prod_diff);
        sum_array.iter().sum::<f64>() / (len - 1) as f64
    }
    
    /// AVX2 correlation calculation (fallback)
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_correlation_avx2(
        &self,
        data: &[f64],
        n_assets: usize,
    ) -> Result<DMatrix<f64>, AnalyzerError> {
        let mut correlation_matrix = DMatrix::zeros(n_assets, n_assets);
        let data_len = data.len() / n_assets;
        
        // Process 4 correlations at once with AVX2
        for i in 0..n_assets {
            let asset_i = &data[i * data_len..(i + 1) * data_len];
            let mean_i = self.calculate_mean_avx2(asset_i);
            let var_i = self.calculate_variance_avx2(asset_i, mean_i);
            
            for j in (i..n_assets).step_by(4) {
                let end_idx = (j + 4).min(n_assets);
                let batch_size = end_idx - j;
                
                let mut correlations = [0.0; 4];
                
                for k in 0..batch_size {
                    let asset_j = &data[(j + k) * data_len..((j + k) + 1) * data_len];
                    let mean_j = self.calculate_mean_avx2(asset_j);
                    let var_j = self.calculate_variance_avx2(asset_j, mean_j);
                    
                    let covariance = self.calculate_covariance_avx2(asset_i, asset_j, mean_i, mean_j);
                    correlations[k] = covariance / (var_i.sqrt() * var_j.sqrt());
                }
                
                // Store results
                for k in 0..batch_size {
                    correlation_matrix[(i, j + k)] = correlations[k];
                    correlation_matrix[(j + k, i)] = correlations[k];
                }
            }
        }
        
        Ok(correlation_matrix)
    }
    
    /// AVX2 mean calculation
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_mean_avx2(&self, data: &[f64]) -> f64 {
        let mut sum = _mm256_setzero_pd();
        let len = data.len();
        
        // Process 4 elements at once
        for chunk in data.chunks_exact(4) {
            let values = _mm256_loadu_pd(chunk.as_ptr());
            sum = _mm256_add_pd(sum, values);
        }
        
        // Handle remaining elements
        let remainder = len % 4;
        if remainder > 0 {
            let mut remaining_sum = 0.0;
            for &value in &data[len - remainder..] {
                remaining_sum += value;
            }
            sum = _mm256_add_pd(sum, _mm256_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum);
        sum_array.iter().sum::<f64>() / len as f64
    }
    
    /// AVX2 variance calculation
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_variance_avx2(&self, data: &[f64], mean: f64) -> f64 {
        let mut sum_sq_diff = _mm256_setzero_pd();
        let mean_vec = _mm256_set1_pd(mean);
        let len = data.len();
        
        // Process 4 elements at once
        for chunk in data.chunks_exact(4) {
            let values = _mm256_loadu_pd(chunk.as_ptr());
            let diff = _mm256_sub_pd(values, mean_vec);
            let sq_diff = _mm256_mul_pd(diff, diff);
            sum_sq_diff = _mm256_add_pd(sum_sq_diff, sq_diff);
        }
        
        // Handle remaining elements
        let remainder = len % 4;
        if remainder > 0 {
            let mut remaining_sum = 0.0;
            for &value in &data[len - remainder..] {
                let diff = value - mean;
                remaining_sum += diff * diff;
            }
            sum_sq_diff = _mm256_add_pd(sum_sq_diff, _mm256_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum_sq_diff);
        sum_array.iter().sum::<f64>() / (len - 1) as f64
    }
    
    /// AVX2 covariance calculation
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_covariance_avx2(
        &self,
        data_x: &[f64],
        data_y: &[f64],
        mean_x: f64,
        mean_y: f64,
    ) -> f64 {
        let mut sum_prod_diff = _mm256_setzero_pd();
        let mean_x_vec = _mm256_set1_pd(mean_x);
        let mean_y_vec = _mm256_set1_pd(mean_y);
        let len = data_x.len();
        
        // Process 4 elements at once
        for (chunk_x, chunk_y) in data_x.chunks_exact(4).zip(data_y.chunks_exact(4)) {
            let values_x = _mm256_loadu_pd(chunk_x.as_ptr());
            let values_y = _mm256_loadu_pd(chunk_y.as_ptr());
            
            let diff_x = _mm256_sub_pd(values_x, mean_x_vec);
            let diff_y = _mm256_sub_pd(values_y, mean_y_vec);
            let prod_diff = _mm256_mul_pd(diff_x, diff_y);
            
            sum_prod_diff = _mm256_add_pd(sum_prod_diff, prod_diff);
        }
        
        // Handle remaining elements
        let remainder = len % 4;
        if remainder > 0 {
            let mut remaining_sum = 0.0;
            for i in len - remainder..len {
                remaining_sum += (data_x[i] - mean_x) * (data_y[i] - mean_y);
            }
            sum_prod_diff = _mm256_add_pd(sum_prod_diff, _mm256_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum_prod_diff);
        sum_array.iter().sum::<f64>() / (len - 1) as f64
    }
    
    /// SSE4.2 correlation calculation (fallback)
    #[target_feature(enable = "sse4.2")]
    unsafe fn calculate_correlation_sse42(
        &self,
        data: &[f64],
        n_assets: usize,
    ) -> Result<DMatrix<f64>, AnalyzerError> {
        let mut correlation_matrix = DMatrix::zeros(n_assets, n_assets);
        let data_len = data.len() / n_assets;
        
        // Process 2 correlations at once with SSE4.2
        for i in 0..n_assets {
            let asset_i = &data[i * data_len..(i + 1) * data_len];
            let mean_i = self.calculate_mean_sse42(asset_i);
            let var_i = self.calculate_variance_sse42(asset_i, mean_i);
            
            for j in (i..n_assets).step_by(2) {
                let end_idx = (j + 2).min(n_assets);
                let batch_size = end_idx - j;
                
                let mut correlations = [0.0; 2];
                
                for k in 0..batch_size {
                    let asset_j = &data[(j + k) * data_len..((j + k) + 1) * data_len];
                    let mean_j = self.calculate_mean_sse42(asset_j);
                    let var_j = self.calculate_variance_sse42(asset_j, mean_j);
                    
                    let covariance = self.calculate_covariance_sse42(asset_i, asset_j, mean_i, mean_j);
                    correlations[k] = covariance / (var_i.sqrt() * var_j.sqrt());
                }
                
                // Store results
                for k in 0..batch_size {
                    correlation_matrix[(i, j + k)] = correlations[k];
                    correlation_matrix[(j + k, i)] = correlations[k];
                }
            }
        }
        
        Ok(correlation_matrix)
    }
    
    /// SSE4.2 mean calculation
    #[target_feature(enable = "sse4.2")]
    unsafe fn calculate_mean_sse42(&self, data: &[f64]) -> f64 {
        let mut sum = _mm_setzero_pd();
        let len = data.len();
        
        // Process 2 elements at once
        for chunk in data.chunks_exact(2) {
            let values = _mm_loadu_pd(chunk.as_ptr());
            sum = _mm_add_pd(sum, values);
        }
        
        // Handle remaining elements
        let remainder = len % 2;
        if remainder > 0 {
            let remaining_sum = data[len - 1];
            sum = _mm_add_pd(sum, _mm_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m128d, [f64; 2]>(sum);
        sum_array.iter().sum::<f64>() / len as f64
    }
    
    /// SSE4.2 variance calculation
    #[target_feature(enable = "sse4.2")]
    unsafe fn calculate_variance_sse42(&self, data: &[f64], mean: f64) -> f64 {
        let mut sum_sq_diff = _mm_setzero_pd();
        let mean_vec = _mm_set1_pd(mean);
        let len = data.len();
        
        // Process 2 elements at once
        for chunk in data.chunks_exact(2) {
            let values = _mm_loadu_pd(chunk.as_ptr());
            let diff = _mm_sub_pd(values, mean_vec);
            let sq_diff = _mm_mul_pd(diff, diff);
            sum_sq_diff = _mm_add_pd(sum_sq_diff, sq_diff);
        }
        
        // Handle remaining elements
        let remainder = len % 2;
        if remainder > 0 {
            let diff = data[len - 1] - mean;
            let remaining_sum = diff * diff;
            sum_sq_diff = _mm_add_pd(sum_sq_diff, _mm_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m128d, [f64; 2]>(sum_sq_diff);
        sum_array.iter().sum::<f64>() / (len - 1) as f64
    }
    
    /// SSE4.2 covariance calculation
    #[target_feature(enable = "sse4.2")]
    unsafe fn calculate_covariance_sse42(
        &self,
        data_x: &[f64],
        data_y: &[f64],
        mean_x: f64,
        mean_y: f64,
    ) -> f64 {
        let mut sum_prod_diff = _mm_setzero_pd();
        let mean_x_vec = _mm_set1_pd(mean_x);
        let mean_y_vec = _mm_set1_pd(mean_y);
        let len = data_x.len();
        
        // Process 2 elements at once
        for (chunk_x, chunk_y) in data_x.chunks_exact(2).zip(data_y.chunks_exact(2)) {
            let values_x = _mm_loadu_pd(chunk_x.as_ptr());
            let values_y = _mm_loadu_pd(chunk_y.as_ptr());
            
            let diff_x = _mm_sub_pd(values_x, mean_x_vec);
            let diff_y = _mm_sub_pd(values_y, mean_y_vec);
            let prod_diff = _mm_mul_pd(diff_x, diff_y);
            
            sum_prod_diff = _mm_add_pd(sum_prod_diff, prod_diff);
        }
        
        // Handle remaining elements
        let remainder = len % 2;
        if remainder > 0 {
            let remaining_sum = (data_x[len - 1] - mean_x) * (data_y[len - 1] - mean_y);
            sum_prod_diff = _mm_add_pd(sum_prod_diff, _mm_set1_pd(remaining_sum));
        }
        
        // Horizontal sum
        let sum_array = std::mem::transmute::<__m128d, [f64; 2]>(sum_prod_diff);
        sum_array.iter().sum::<f64>() / (len - 1) as f64
    }
    
    /// Fast distance calculation using SIMD
    pub fn calculate_distance_matrix_simd(
        &self,
        points: &[f64],
        n_points: usize,
        dimensions: usize,
    ) -> Result<DMatrix<f64>, AnalyzerError> {
        let mut distance_matrix = DMatrix::zeros(n_points, n_points);
        
        if self.cpu_features.avx512f {
            unsafe { self.calculate_distance_avx512(points, n_points, dimensions, &mut distance_matrix) }?;
        } else if self.cpu_features.avx2 {
            unsafe { self.calculate_distance_avx2(points, n_points, dimensions, &mut distance_matrix) }?;
        } else {
            unsafe { self.calculate_distance_sse42(points, n_points, dimensions, &mut distance_matrix) }?;
        }
        
        Ok(distance_matrix)
    }
    
    /// AVX-512 distance calculation
    #[target_feature(enable = "avx512f")]
    unsafe fn calculate_distance_avx512(
        &self,
        points: &[f64],
        n_points: usize,
        dimensions: usize,
        distance_matrix: &mut DMatrix<f64>,
    ) -> Result<(), AnalyzerError> {
        for i in 0..n_points {
            let point_i = &points[i * dimensions..(i + 1) * dimensions];
            
            for j in i + 1..n_points {
                let point_j = &points[j * dimensions..(j + 1) * dimensions];
                
                let mut sum_sq_diff = _mm512_setzero_pd();
                
                // Process 8 dimensions at once
                for (chunk_i, chunk_j) in point_i.chunks_exact(8).zip(point_j.chunks_exact(8)) {
                    let values_i = _mm512_loadu_pd(chunk_i.as_ptr());
                    let values_j = _mm512_loadu_pd(chunk_j.as_ptr());
                    
                    let diff = _mm512_sub_pd(values_i, values_j);
                    let sq_diff = _mm512_mul_pd(diff, diff);
                    sum_sq_diff = _mm512_add_pd(sum_sq_diff, sq_diff);
                }
                
                // Handle remaining dimensions
                let remainder = dimensions % 8;
                if remainder > 0 {
                    let mut remaining_sum = 0.0;
                    for k in dimensions - remainder..dimensions {
                        let diff = point_i[k] - point_j[k];
                        remaining_sum += diff * diff;
                    }
                    sum_sq_diff = _mm512_add_pd(sum_sq_diff, _mm512_set1_pd(remaining_sum));
                }
                
                // Calculate distance
                let sum_array = std::mem::transmute::<__m512d, [f64; 8]>(sum_sq_diff);
                let distance = sum_array.iter().sum::<f64>().sqrt();
                
                distance_matrix[(i, j)] = distance;
                distance_matrix[(j, i)] = distance;
            }
        }
        
        Ok(())
    }
    
    /// AVX2 distance calculation
    #[target_feature(enable = "avx2")]
    unsafe fn calculate_distance_avx2(
        &self,
        points: &[f64],
        n_points: usize,
        dimensions: usize,
        distance_matrix: &mut DMatrix<f64>,
    ) -> Result<(), AnalyzerError> {
        for i in 0..n_points {
            let point_i = &points[i * dimensions..(i + 1) * dimensions];
            
            for j in i + 1..n_points {
                let point_j = &points[j * dimensions..(j + 1) * dimensions];
                
                let mut sum_sq_diff = _mm256_setzero_pd();
                
                // Process 4 dimensions at once
                for (chunk_i, chunk_j) in point_i.chunks_exact(4).zip(point_j.chunks_exact(4)) {
                    let values_i = _mm256_loadu_pd(chunk_i.as_ptr());
                    let values_j = _mm256_loadu_pd(chunk_j.as_ptr());
                    
                    let diff = _mm256_sub_pd(values_i, values_j);
                    let sq_diff = _mm256_mul_pd(diff, diff);
                    sum_sq_diff = _mm256_add_pd(sum_sq_diff, sq_diff);
                }
                
                // Handle remaining dimensions
                let remainder = dimensions % 4;
                if remainder > 0 {
                    let mut remaining_sum = 0.0;
                    for k in dimensions - remainder..dimensions {
                        let diff = point_i[k] - point_j[k];
                        remaining_sum += diff * diff;
                    }
                    sum_sq_diff = _mm256_add_pd(sum_sq_diff, _mm256_set1_pd(remaining_sum));
                }
                
                // Calculate distance
                let sum_array = std::mem::transmute::<__m256d, [f64; 4]>(sum_sq_diff);
                let distance = sum_array.iter().sum::<f64>().sqrt();
                
                distance_matrix[(i, j)] = distance;
                distance_matrix[(j, i)] = distance;
            }
        }
        
        Ok(())
    }
    
    /// SSE4.2 distance calculation
    #[target_feature(enable = "sse4.2")]
    unsafe fn calculate_distance_sse42(
        &self,
        points: &[f64],
        n_points: usize,
        dimensions: usize,
        distance_matrix: &mut DMatrix<f64>,
    ) -> Result<(), AnalyzerError> {
        for i in 0..n_points {
            let point_i = &points[i * dimensions..(i + 1) * dimensions];
            
            for j in i + 1..n_points {
                let point_j = &points[j * dimensions..(j + 1) * dimensions];
                
                let mut sum_sq_diff = _mm_setzero_pd();
                
                // Process 2 dimensions at once
                for (chunk_i, chunk_j) in point_i.chunks_exact(2).zip(point_j.chunks_exact(2)) {
                    let values_i = _mm_loadu_pd(chunk_i.as_ptr());
                    let values_j = _mm_loadu_pd(chunk_j.as_ptr());
                    
                    let diff = _mm_sub_pd(values_i, values_j);
                    let sq_diff = _mm_mul_pd(diff, diff);
                    sum_sq_diff = _mm_add_pd(sum_sq_diff, sq_diff);
                }
                
                // Handle remaining dimensions
                let remainder = dimensions % 2;
                if remainder > 0 {
                    let diff = point_i[dimensions - 1] - point_j[dimensions - 1];
                    let remaining_sum = diff * diff;
                    sum_sq_diff = _mm_add_pd(sum_sq_diff, _mm_set1_pd(remaining_sum));
                }
                
                // Calculate distance
                let sum_array = std::mem::transmute::<__m128d, [f64; 2]>(sum_sq_diff);
                let distance = sum_array.iter().sum::<f64>().sqrt();
                
                distance_matrix[(i, j)] = distance;
                distance_matrix[(j, i)] = distance;
            }
        }
        
        Ok(())
    }
    
    /// Get CPU features
    pub fn get_cpu_features(&self) -> &CpuFeatures {
        &self.cpu_features
    }
    
    /// Get performance characteristics
    pub fn get_performance_info(&self) -> SimdPerformanceInfo {
        SimdPerformanceInfo {
            vector_width: if self.cpu_features.avx512f { 512 } else if self.cpu_features.avx2 { 256 } else { 128 },
            max_throughput_gflops: self.estimate_throughput(),
            memory_bandwidth_gb_s: self.estimate_memory_bandwidth(),
            instruction_set: self.get_instruction_set(),
        }
    }
    
    /// Estimate throughput in GFLOPS
    fn estimate_throughput(&self) -> f64 {
        if self.cpu_features.avx512f {
            // AVX-512 can do 8 double precision ops per cycle
            8.0 * 3.0 // Assume 3 GHz CPU
        } else if self.cpu_features.avx2 {
            // AVX2 can do 4 double precision ops per cycle
            4.0 * 3.0
        } else {
            // SSE can do 2 double precision ops per cycle
            2.0 * 3.0
        }
    }
    
    /// Estimate memory bandwidth
    fn estimate_memory_bandwidth(&self) -> f64 {
        // Rough estimate based on typical memory bandwidth
        if self.cpu_features.avx512f {
            100.0 // GB/s
        } else if self.cpu_features.avx2 {
            80.0
        } else {
            60.0
        }
    }
    
    /// Get instruction set name
    fn get_instruction_set(&self) -> &'static str {
        if self.cpu_features.avx512f {
            "AVX-512"
        } else if self.cpu_features.avx2 {
            "AVX2"
        } else {
            "SSE4.2"
        }
    }
}

/// SIMD performance information
#[derive(Debug, Clone)]
pub struct SimdPerformanceInfo {
    pub vector_width: usize,
    pub max_throughput_gflops: f64,
    pub memory_bandwidth_gb_s: f64,
    pub instruction_set: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();
        println!("CPU Features: {:?}", features);
        assert!(features.sse42); // SSE4.2 should be available on most modern CPUs
    }
    
    #[test]
    fn test_simd_accelerator_creation() {
        let config = SimdConfig::default();
        let accelerator = SimdAccelerator::new(config);
        assert!(accelerator.is_ok());
    }
    
    #[test]
    fn test_correlation_matrix_calculation() {
        let config = SimdConfig::default();
        let accelerator = SimdAccelerator::new(config).unwrap();
        
        // Create test data
        let n_assets = 10;
        let data_len = 1000;
        let data: Vec<f64> = (0..n_assets * data_len)
            .map(|i| (i as f64).sin())
            .collect();
        
        let correlation_matrix = accelerator.calculate_correlation_matrix_simd(&data, n_assets);
        assert!(correlation_matrix.is_ok());
        
        let matrix = correlation_matrix.unwrap();
        assert_eq!(matrix.nrows(), n_assets);
        assert_eq!(matrix.ncols(), n_assets);
        
        // Check diagonal elements are 1.0 (approximately)
        for i in 0..n_assets {
            assert!((matrix[(i, i)] - 1.0).abs() < 0.1);
        }
    }
    
    #[test]
    fn test_distance_matrix_calculation() {
        let config = SimdConfig::default();
        let accelerator = SimdAccelerator::new(config).unwrap();
        
        // Create test points
        let n_points = 100;
        let dimensions = 16;
        let points: Vec<f64> = (0..n_points * dimensions)
            .map(|i| (i as f64).sin())
            .collect();
        
        let distance_matrix = accelerator.calculate_distance_matrix_simd(&points, n_points, dimensions);
        assert!(distance_matrix.is_ok());
        
        let matrix = distance_matrix.unwrap();
        assert_eq!(matrix.nrows(), n_points);
        assert_eq!(matrix.ncols(), n_points);
        
        // Check diagonal elements are 0.0
        for i in 0..n_points {
            assert!((matrix[(i, i)]).abs() < 1e-10);
        }
        
        // Check symmetry
        for i in 0..n_points {
            for j in i + 1..n_points {
                assert!((matrix[(i, j)] - matrix[(j, i)]).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_performance_info() {
        let config = SimdConfig::default();
        let accelerator = SimdAccelerator::new(config).unwrap();
        
        let perf_info = accelerator.get_performance_info();
        assert!(perf_info.vector_width >= 128);
        assert!(perf_info.max_throughput_gflops > 0.0);
        assert!(perf_info.memory_bandwidth_gb_s > 0.0);
    }
}