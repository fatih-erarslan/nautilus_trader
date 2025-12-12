//! SIMD optimizations for whale defense operations
//! 
//! Ultra-fast vectorized computations using AVX-512 and other SIMD instruction sets.

use crate::{
    error::{WhaleDefenseError, Result},
    config::*,
};
use core::{
    arch::x86_64::*,
    mem::MaybeUninit,
    slice,
};

/// Check if AVX-512 is supported on this CPU
#[inline(always)]
pub fn check_avx512_support() -> bool {
    unsafe {
        let cpuid = __cpuid(7);
        (cpuid.ebx & (1 << 16)) != 0 // AVX-512F
    }
}

/// Check if AVX2 is supported on this CPU
#[inline(always)]
pub fn check_avx2_support() -> bool {
    unsafe {
        let cpuid = __cpuid(7);
        (cpuid.ebx & (1 << 5)) != 0 // AVX2
    }
}

/// Check if SSE4.1 is supported on this CPU
#[inline(always)]
pub fn check_sse41_support() -> bool {
    unsafe {
        let cpuid = __cpuid(1);
        (cpuid.ecx & (1 << 19)) != 0 // SSE4.1
    }
}

/// SIMD-optimized whale pattern matching
/// 
/// Uses AVX-512 to process multiple market data points simultaneously.
/// Target performance: <100 nanoseconds for 8 data points.
#[target_feature(enable = "avx512f")]
#[inline(always)]
pub unsafe fn simd_whale_pattern_match(
    prices: &[f64],
    volumes: &[f64],
    thresholds: &[f64; 4],
) -> Result<[bool; 8]> {
    if prices.len() < 8 || volumes.len() < 8 {
        return Err(WhaleDefenseError::InvalidParameter);
    }
    
    // Load 8 prices and volumes into AVX-512 registers
    let prices_vec = _mm512_loadu_pd(prices.as_ptr());
    let volumes_vec = _mm512_loadu_pd(volumes.as_ptr());
    
    // Load thresholds
    let volume_threshold = _mm512_set1_pd(thresholds[0]);
    let price_threshold = _mm512_set1_pd(thresholds[1]);
    let impact_threshold = _mm512_set1_pd(thresholds[2]);
    let momentum_threshold = _mm512_set1_pd(thresholds[3]);
    
    // Calculate price impact (simplified: price * volume)
    let price_impact = _mm512_mul_pd(prices_vec, volumes_vec);
    
    // Compare against thresholds
    let volume_mask = _mm512_cmp_pd_mask(volumes_vec, volume_threshold, _CMP_GT_OQ);
    let price_mask = _mm512_cmp_pd_mask(prices_vec, price_threshold, _CMP_GT_OQ);
    let impact_mask = _mm512_cmp_pd_mask(price_impact, impact_threshold, _CMP_GT_OQ);
    
    // Combine masks (whale detected if volume AND (price OR impact))
    let whale_mask = volume_mask & (price_mask | impact_mask);
    
    // Convert mask to boolean array
    let mut results = [false; 8];
    for i in 0..8 {
        results[i] = (whale_mask & (1 << i)) != 0;
    }
    
    Ok(results)
}

/// SIMD-optimized volume analysis
/// 
/// Processes volume data to detect anomalies using vectorized operations.
#[target_feature(enable = "avx512f")]
#[inline(always)]
pub unsafe fn simd_volume_analysis(volumes: &[f64], window_size: usize) -> Result<Vec<f64>> {
    if volumes.len() < window_size || window_size < 8 {
        return Err(WhaleDefenseError::InvalidParameter);
    }
    
    let mut anomaly_scores = Vec::with_capacity(volumes.len() - window_size + 1);
    
    // Process windows of volume data
    for i in 0..=(volumes.len() - window_size) {
        let window = &volumes[i..i + window_size];
        
        // Calculate mean and standard deviation using SIMD
        let mean = simd_calculate_mean(window)?;
        let std_dev = simd_calculate_std_dev(window, mean)?;
        
        // Calculate anomaly score for current value
        let current_value = volumes[i + window_size - 1];
        let z_score = if std_dev > 0.0 {
            (current_value - mean) / std_dev
        } else {
            0.0
        };
        
        anomaly_scores.push(z_score.abs());
    }
    
    Ok(anomaly_scores)
}

/// SIMD-optimized mean calculation
#[target_feature(enable = "avx512f")]
#[inline(always)]
unsafe fn simd_calculate_mean(data: &[f64]) -> Result<f64> {
    if data.is_empty() {
        return Ok(0.0);
    }
    
    let mut sum = 0.0;
    let chunks = data.len() / 8;
    let remainder = data.len() % 8;
    
    // Process 8 elements at a time
    if chunks > 0 {
        let mut sum_vec = _mm512_setzero_pd();
        
        for i in 0..chunks {
            let chunk_vec = _mm512_loadu_pd(data.as_ptr().add(i * 8));
            sum_vec = _mm512_add_pd(sum_vec, chunk_vec);
        }
        
        // Horizontal sum of vector elements
        sum = simd_horizontal_sum_f64(sum_vec);
    }
    
    // Process remaining elements
    for i in (chunks * 8)..data.len() {
        sum += data[i];
    }
    
    Ok(sum / data.len() as f64)
}

/// SIMD-optimized standard deviation calculation
#[target_feature(enable = "avx512f")]
#[inline(always)]
unsafe fn simd_calculate_std_dev(data: &[f64], mean: f64) -> Result<f64> {
    if data.len() <= 1 {
        return Ok(0.0);
    }
    
    let mean_vec = _mm512_set1_pd(mean);
    let mut sum_squared_diff = 0.0;
    let chunks = data.len() / 8;
    
    // Process 8 elements at a time
    if chunks > 0 {
        let mut sum_vec = _mm512_setzero_pd();
        
        for i in 0..chunks {
            let chunk_vec = _mm512_loadu_pd(data.as_ptr().add(i * 8));
            let diff_vec = _mm512_sub_pd(chunk_vec, mean_vec);
            let squared_diff_vec = _mm512_mul_pd(diff_vec, diff_vec);
            sum_vec = _mm512_add_pd(sum_vec, squared_diff_vec);
        }
        
        sum_squared_diff = simd_horizontal_sum_f64(sum_vec);
    }
    
    // Process remaining elements
    for i in (chunks * 8)..data.len() {
        let diff = data[i] - mean;
        sum_squared_diff += diff * diff;
    }
    
    let variance = sum_squared_diff / (data.len() - 1) as f64;
    Ok(variance.sqrt())
}

/// Horizontal sum of AVX-512 vector
#[target_feature(enable = "avx512f")]
#[inline(always)]
unsafe fn simd_horizontal_sum_f64(vec: __m512d) -> f64 {
    // Extract high and low 256-bit lanes
    let high = _mm512_extractf64x4_pd(vec, 1);
    let low = _mm512_extractf64x4_pd(vec, 0);
    
    // Add the lanes
    let sum_256 = _mm256_add_pd(high, low);
    
    // Horizontal add within 256-bit vector
    let sum_128_1 = _mm256_extractf128_pd(sum_256, 1);
    let sum_128_2 = _mm256_extractf128_pd(sum_256, 0);
    let sum_128 = _mm_add_pd(sum_128_1, sum_128_2);
    
    // Final horizontal add
    let sum_low = _mm_unpacklo_pd(sum_128, sum_128);
    let sum_high = _mm_unpackhi_pd(sum_128, sum_128);
    let final_sum = _mm_add_pd(sum_low, sum_high);
    
    _mm_cvtsd_f64(final_sum)
}

/// SIMD-optimized price impact calculation
/// 
/// Calculates price impact for multiple orders simultaneously.
#[target_feature(enable = "avx2")]
#[inline(always)]
pub unsafe fn simd_price_impact_calculation(
    prices: &[f64],
    volumes: &[f64],
    base_prices: &[f64],
) -> Result<Vec<f64>> {
    if prices.len() != volumes.len() || prices.len() != base_prices.len() {
        return Err(WhaleDefenseError::InvalidParameter);
    }
    
    let len = prices.len();
    let mut impacts = Vec::with_capacity(len);
    impacts.set_len(len);
    
    let chunks = len / 4; // AVX2 processes 4 f64 values
    
    // Process 4 elements at a time with AVX2
    for i in 0..chunks {
        let offset = i * 4;
        
        let prices_vec = _mm256_loadu_pd(prices.as_ptr().add(offset));
        let volumes_vec = _mm256_loadu_pd(volumes.as_ptr().add(offset));
        let base_prices_vec = _mm256_loadu_pd(base_prices.as_ptr().add(offset));
        
        // Calculate price change ratio: (price - base_price) / base_price
        let price_diff = _mm256_sub_pd(prices_vec, base_prices_vec);
        let price_ratio = _mm256_div_pd(price_diff, base_prices_vec);
        
        // Calculate impact: price_ratio * volume
        let impact_vec = _mm256_mul_pd(price_ratio, volumes_vec);
        
        // Store results
        _mm256_storeu_pd(impacts.as_mut_ptr().add(offset), impact_vec);
    }
    
    // Process remaining elements
    for i in (chunks * 4)..len {
        let price_ratio = (prices[i] - base_prices[i]) / base_prices[i];
        impacts[i] = price_ratio * volumes[i];
    }
    
    Ok(impacts)
}

/// SIMD-optimized moving average calculation
/// 
/// Computes moving averages for whale detection smoothing.
#[target_feature(enable = "avx2")]
#[inline(always)]
pub unsafe fn simd_moving_average(data: &[f64], window_size: usize) -> Result<Vec<f64>> {
    if data.len() < window_size || window_size == 0 {
        return Err(WhaleDefenseError::InvalidParameter);
    }
    
    let result_len = data.len() - window_size + 1;
    let mut averages = Vec::with_capacity(result_len);
    averages.set_len(result_len);
    
    let window_size_f64 = window_size as f64;
    let window_size_vec = _mm256_set1_pd(window_size_f64);
    
    // Calculate moving averages
    for i in 0..result_len {
        let window = &data[i..i + window_size];
        
        // Calculate sum using SIMD
        let chunks = window_size / 4;
        let mut sum = 0.0;
        
        if chunks > 0 {
            let mut sum_vec = _mm256_setzero_pd();
            
            for j in 0..chunks {
                let chunk_vec = _mm256_loadu_pd(window.as_ptr().add(j * 4));
                sum_vec = _mm256_add_pd(sum_vec, chunk_vec);
            }
            
            // Horizontal sum
            sum = simd_horizontal_sum_f64_avx2(sum_vec);
        }
        
        // Process remaining elements
        for j in (chunks * 4)..window_size {
            sum += window[j];
        }
        
        averages[i] = sum / window_size_f64;
    }
    
    Ok(averages)
}

/// Horizontal sum for AVX2 (256-bit) vectors
#[target_feature(enable = "avx2")]
#[inline(always)]
unsafe fn simd_horizontal_sum_f64_avx2(vec: __m256d) -> f64 {
    let high = _mm256_extractf128_pd(vec, 1);
    let low = _mm256_extractf128_pd(vec, 0);
    let sum_128 = _mm_add_pd(high, low);
    
    let sum_low = _mm_unpacklo_pd(sum_128, sum_128);
    let sum_high = _mm_unpackhi_pd(sum_128, sum_128);
    let final_sum = _mm_add_pd(sum_low, sum_high);
    
    _mm_cvtsd_f64(final_sum)
}

/// SIMD-optimized correlation calculation
/// 
/// Calculates correlation between price and volume for whale detection.
#[target_feature(enable = "avx2")]
#[inline(always)]
pub unsafe fn simd_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() || x.is_empty() {
        return Err(WhaleDefenseError::InvalidParameter);
    }
    
    let n = x.len() as f64;
    let n_vec = _mm256_set1_pd(n);
    
    // Calculate means
    let mean_x = simd_calculate_mean(x)?;
    let mean_y = simd_calculate_mean(y)?;
    
    let mean_x_vec = _mm256_set1_pd(mean_x);
    let mean_y_vec = _mm256_set1_pd(mean_y);
    
    let chunks = x.len() / 4;
    let mut sum_xy = 0.0;
    let mut sum_x_squared = 0.0;
    let mut sum_y_squared = 0.0;
    
    // Process 4 elements at a time
    if chunks > 0 {
        let mut sum_xy_vec = _mm256_setzero_pd();
        let mut sum_x2_vec = _mm256_setzero_pd();
        let mut sum_y2_vec = _mm256_setzero_pd();
        
        for i in 0..chunks {
            let offset = i * 4;
            
            let x_vec = _mm256_loadu_pd(x.as_ptr().add(offset));
            let y_vec = _mm256_loadu_pd(y.as_ptr().add(offset));
            
            let x_diff = _mm256_sub_pd(x_vec, mean_x_vec);
            let y_diff = _mm256_sub_pd(y_vec, mean_y_vec);
            
            let xy_product = _mm256_mul_pd(x_diff, y_diff);
            let x_squared = _mm256_mul_pd(x_diff, x_diff);
            let y_squared = _mm256_mul_pd(y_diff, y_diff);
            
            sum_xy_vec = _mm256_add_pd(sum_xy_vec, xy_product);
            sum_x2_vec = _mm256_add_pd(sum_x2_vec, x_squared);
            sum_y2_vec = _mm256_add_pd(sum_y2_vec, y_squared);
        }
        
        sum_xy = simd_horizontal_sum_f64_avx2(sum_xy_vec);
        sum_x_squared = simd_horizontal_sum_f64_avx2(sum_x2_vec);
        sum_y_squared = simd_horizontal_sum_f64_avx2(sum_y2_vec);
    }
    
    // Process remaining elements
    for i in (chunks * 4)..x.len() {
        let x_diff = x[i] - mean_x;
        let y_diff = y[i] - mean_y;
        
        sum_xy += x_diff * y_diff;
        sum_x_squared += x_diff * x_diff;
        sum_y_squared += y_diff * y_diff;
    }
    
    // Calculate correlation coefficient
    let denominator = (sum_x_squared * sum_y_squared).sqrt();
    if denominator > 0.0 {
        Ok(sum_xy / denominator)
    } else {
        Ok(0.0)
    }
}

/// SIMD feature detection at runtime
pub struct SimdFeatures {
    pub has_sse41: bool,
    pub has_avx2: bool,
    pub has_avx512f: bool,
    pub has_avx512dq: bool,
    pub has_avx512bw: bool,
}

impl SimdFeatures {
    /// Detect available SIMD features
    pub fn detect() -> Self {
        Self {
            has_sse41: check_sse41_support(),
            has_avx2: check_avx2_support(),
            has_avx512f: check_avx512_support(),
            has_avx512dq: unsafe {
                let cpuid = __cpuid(7);
                (cpuid.ebx & (1 << 17)) != 0
            },
            has_avx512bw: unsafe {
                let cpuid = __cpuid(7);
                (cpuid.ebx & (1 << 30)) != 0
            },
        }
    }
    
    /// Get optimal vector width for current CPU
    pub fn optimal_vector_width(&self) -> usize {
        if self.has_avx512f {
            8 // 512 bits / 64 bits per f64
        } else if self.has_avx2 {
            4 // 256 bits / 64 bits per f64
        } else if self.has_sse41 {
            2 // 128 bits / 64 bits per f64
        } else {
            1 // Scalar fallback
        }
    }
}

/// Dynamic dispatch for SIMD operations based on CPU features
pub struct SimdDispatcher {
    features: SimdFeatures,
}

impl SimdDispatcher {
    /// Create new SIMD dispatcher
    pub fn new() -> Self {
        Self {
            features: SimdFeatures::detect(),
        }
    }
    
    /// Get CPU features
    pub fn features(&self) -> &SimdFeatures {
        &self.features
    }
    
    /// Dispatch whale pattern matching to optimal SIMD implementation
    pub fn dispatch_whale_pattern_match(
        &self,
        prices: &[f64],
        volumes: &[f64],
        thresholds: &[f64; 4],
    ) -> Result<Vec<bool>> {
        unsafe {
            if self.features.has_avx512f && prices.len() >= 8 {
                let mut results = Vec::new();
                for chunk in prices.chunks(8) {
                    if chunk.len() == 8 && volumes.len() >= prices.len() {
                        let volume_chunk = &volumes[results.len()..results.len() + 8];
                        let chunk_results = simd_whale_pattern_match(chunk, volume_chunk, thresholds)?;
                        results.extend_from_slice(&chunk_results);
                    } else {
                        // Fallback for partial chunks
                        for i in 0..chunk.len() {
                            let volume_idx = results.len() + i;
                            if volume_idx < volumes.len() {
                                let whale_detected = volumes[volume_idx] > thresholds[0] &&
                                                   (chunk[i] > thresholds[1] || 
                                                    chunk[i] * volumes[volume_idx] > thresholds[2]);
                                results.push(whale_detected);
                            }
                        }
                    }
                }
                Ok(results)
            } else {
                // Scalar fallback
                let mut results = Vec::with_capacity(prices.len());
                for (i, &price) in prices.iter().enumerate() {
                    if i < volumes.len() {
                        let volume = volumes[i];
                        let whale_detected = volume > thresholds[0] &&
                                           (price > thresholds[1] || 
                                            price * volume > thresholds[2]);
                        results.push(whale_detected);
                    }
                }
                Ok(results)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simd_feature_detection() {
        let features = SimdFeatures::detect();
        println!("SSE4.1: {}", features.has_sse41);
        println!("AVX2: {}", features.has_avx2);
        println!("AVX-512F: {}", features.has_avx512f);
        
        let optimal_width = features.optimal_vector_width();
        assert!(optimal_width >= 1);
    }
    
    #[test]
    fn test_simd_dispatcher() {
        let dispatcher = SimdDispatcher::new();
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0];
        let volumes = vec![1000.0, 2000.0, 1500.0, 3000.0, 1200.0, 2500.0, 1800.0, 2200.0];
        let thresholds = [1500.0, 100.0, 150000.0, 0.05];
        
        let results = dispatcher.dispatch_whale_pattern_match(&prices, &volumes, &thresholds);
        assert!(results.is_ok());
        
        let whale_detected = results.unwrap();
        assert_eq!(whale_detected.len(), prices.len());
    }
    
    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        unsafe {
            if check_avx2_support() {
                let result = simd_moving_average(&data, 3);
                assert!(result.is_ok());
                
                let averages = result.unwrap();
                assert_eq!(averages.len(), data.len() - 3 + 1);
                
                // First average should be (1+2+3)/3 = 2.0
                assert!((averages[0] - 2.0).abs() < 1e-10);
            }
        }
    }
    
    #[test]
    fn test_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        unsafe {
            if check_avx2_support() {
                let result = simd_correlation(&x, &y);
                assert!(result.is_ok());
                
                let correlation = result.unwrap();
                // Perfect positive correlation should be close to 1.0
                assert!((correlation - 1.0).abs() < 1e-10);
            }
        }
    }
}