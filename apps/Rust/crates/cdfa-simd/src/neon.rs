//! ARM NEON SIMD optimizations for AArch64 processors
//! 
//! High-performance implementations using 128-bit vectors
//! Target: Optimal performance on ARM processors (Apple Silicon, AWS Graviton, etc.)

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;
use crate::AlignedVec;

/// NEON optimized correlation calculation
/// 
/// Computes Pearson correlation using NEON intrinsics
/// Performance target: <150ns for 256 element vectors
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn correlation_neon(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    if n == 0 {
        return 0.0;
    }
    
    // Process 2 doubles at a time with NEON
    let chunks = n / 2;
    
    // Accumulators
    let mut sum_x = vdupq_n_f64(0.0);
    let mut sum_y = vdupq_n_f64(0.0);
    let mut sum_xx = vdupq_n_f64(0.0);
    let mut sum_yy = vdupq_n_f64(0.0);
    let mut sum_xy = vdupq_n_f64(0.0);
    
    // Main vectorized loop
    for i in 0..chunks {
        let x_vec = vld1q_f64(&x[i * 2]);
        let y_vec = vld1q_f64(&y[i * 2]);
        
        sum_x = vaddq_f64(sum_x, x_vec);
        sum_y = vaddq_f64(sum_y, y_vec);
        sum_xx = vfmaq_f64(sum_xx, x_vec, x_vec);
        sum_yy = vfmaq_f64(sum_yy, y_vec, y_vec);
        sum_xy = vfmaq_f64(sum_xy, x_vec, y_vec);
    }
    
    // Horizontal sum
    let sum_x_scalar = vaddvq_f64(sum_x);
    let sum_y_scalar = vaddvq_f64(sum_y);
    let sum_xx_scalar = vaddvq_f64(sum_xx);
    let sum_yy_scalar = vaddvq_f64(sum_yy);
    let sum_xy_scalar = vaddvq_f64(sum_xy);
    
    // Handle remainder
    let mut sums = [sum_x_scalar, sum_y_scalar, sum_xx_scalar, sum_yy_scalar, sum_xy_scalar];
    for i in (chunks * 2)..n {
        sums[0] += x[i];
        sums[1] += y[i];
        sums[2] += x[i] * x[i];
        sums[3] += y[i] * y[i];
        sums[4] += x[i] * y[i];
    }
    
    let n_f = n as f64;
    let numerator = n_f * sums[4] - sums[0] * sums[1];
    let denominator = ((n_f * sums[2] - sums[0] * sums[0]) * 
                       (n_f * sums[3] - sums[1] * sums[1])).sqrt();
    
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// NEON optimized wavelet transform (Haar wavelet)
/// 
/// Fast discrete wavelet transform for ARM
/// Performance target: <150ns for small transforms
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dwt_haar_neon(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
    let n = signal.len();
    debug_assert_eq!(n % 2, 0);
    debug_assert_eq!(approx.len(), n / 2);
    debug_assert_eq!(detail.len(), n / 2);
    
    let sqrt2_inv = vdupq_n_f64(1.0 / 2.0_f64.sqrt());
    let half_n = n / 2;
    
    // Process 2 pairs at a time
    let chunks = half_n / 2;
    
    for i in 0..chunks {
        // Load 4 consecutive values [e0, o0, e1, o1]
        let v1 = vld1q_f64(&signal[i * 4]);
        let v2 = vld1q_f64(&signal[i * 4 + 2]);
        
        // Extract even and odd elements
        // Using table lookup for element reordering
        let tbl_even = vcombine_u8(
            vcreate_u8(0x0706050403020100),  // First double
            vcreate_u8(0x1716151413121110)   // Third double
        );
        let tbl_odd = vcombine_u8(
            vcreate_u8(0x0f0e0d0c0b0a0908),  // Second double
            vcreate_u8(0x1f1e1d1c1b1a1918)   // Fourth double
        );
        
        // Reinterpret as byte vectors for table lookup
        let v1_bytes = vreinterpretq_u8_f64(v1);
        let v2_bytes = vreinterpretq_u8_f64(v2);
        
        // Combine vectors
        let combined = vcombine_u8(
            vget_low_u8(v1_bytes),
            vget_low_u8(v2_bytes)
        );
        
        // Extract evens and odds
        let evens_bytes = vqtbl1q_u8(combined, tbl_even);
        let odds_bytes = vqtbl1q_u8(combined, tbl_odd);
        
        let evens = vreinterpretq_f64_u8(evens_bytes);
        let odds = vreinterpretq_f64_u8(odds_bytes);
        
        // Calculate approximation and detail coefficients
        let sum = vaddq_f64(evens, odds);
        let diff = vsubq_f64(evens, odds);
        
        let approx_vals = vmulq_f64(sum, sqrt2_inv);
        let detail_vals = vmulq_f64(diff, sqrt2_inv);
        
        vst1q_f64(&mut approx[i * 2], approx_vals);
        vst1q_f64(&mut detail[i * 2], detail_vals);
    }
    
    // Handle remainder
    for i in (chunks * 2)..half_n {
        let even = signal[2 * i];
        let odd = signal[2 * i + 1];
        approx[i] = (even + odd) / 2.0_f64.sqrt();
        detail[i] = (even - odd) / 2.0_f64.sqrt();
    }
}

/// NEON optimized Euclidean distance
/// 
/// Vectorized L2 distance computation
/// Performance target: <100ns for 256 element vectors
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn euclidean_distance_neon(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    let mut sum = vdupq_n_f64(0.0);
    let chunks = n / 2;
    
    // Unrolled loop for better performance
    let unroll_chunks = chunks / 4;
    for i in 0..unroll_chunks {
        let x0 = vld1q_f64(&x[i * 8]);
        let x1 = vld1q_f64(&x[i * 8 + 2]);
        let x2 = vld1q_f64(&x[i * 8 + 4]);
        let x3 = vld1q_f64(&x[i * 8 + 6]);
        
        let y0 = vld1q_f64(&y[i * 8]);
        let y1 = vld1q_f64(&y[i * 8 + 2]);
        let y2 = vld1q_f64(&y[i * 8 + 4]);
        let y3 = vld1q_f64(&y[i * 8 + 6]);
        
        let diff0 = vsubq_f64(x0, y0);
        let diff1 = vsubq_f64(x1, y1);
        let diff2 = vsubq_f64(x2, y2);
        let diff3 = vsubq_f64(x3, y3);
        
        sum = vfmaq_f64(sum, diff0, diff0);
        sum = vfmaq_f64(sum, diff1, diff1);
        sum = vfmaq_f64(sum, diff2, diff2);
        sum = vfmaq_f64(sum, diff3, diff3);
    }
    
    // Handle remaining chunks
    for i in (unroll_chunks * 4)..chunks {
        let x_vec = vld1q_f64(&x[i * 2]);
        let y_vec = vld1q_f64(&y[i * 2]);
        let diff = vsubq_f64(x_vec, y_vec);
        sum = vfmaq_f64(sum, diff, diff);
    }
    
    // Horizontal sum
    let mut total = vaddvq_f64(sum);
    
    // Handle remainder
    for i in (chunks * 2)..n {
        let diff = x[i] - y[i];
        total += diff * diff;
    }
    
    total.sqrt()
}

/// NEON optimized signal fusion
/// 
/// Weighted combination of multiple signals
/// Performance target: <300ns for fusion operation
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn signal_fusion_neon(
    signals: &[&[f64]], 
    weights: &[f64], 
    output: &mut [f64]
) {
    debug_assert_eq!(signals.len(), weights.len());
    if signals.is_empty() {
        return;
    }
    
    let n = output.len();
    let chunks = n / 2;
    
    // Broadcast weights
    let weight_vecs: Vec<float64x2_t> = weights.iter()
        .map(|&w| vdupq_n_f64(w))
        .collect();
    
    for i in 0..chunks {
        let mut sum = vdupq_n_f64(0.0);
        
        for (signal, &weight_vec) in signals.iter().zip(&weight_vecs) {
            let signal_vec = vld1q_f64(&signal[i * 2]);
            sum = vfmaq_f64(sum, signal_vec, weight_vec);
        }
        
        vst1q_f64(&mut output[i * 2], sum);
    }
    
    // Handle remainder
    for i in (chunks * 2)..n {
        let mut sum = 0.0;
        for (signal, &weight) in signals.iter().zip(weights) {
            sum += signal[i] * weight;
        }
        output[i] = sum;
    }
}

/// NEON optimized Shannon entropy
/// 
/// Fast entropy computation
/// Performance target: <200ns for probability vectors
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn shannon_entropy_neon(probabilities: &[f64]) -> f64 {
    let n = probabilities.len();
    let chunks = n / 2;
    
    let mut entropy_sum = 0.0;
    
    // NEON doesn't have fast log, so we process in smaller chunks
    // and use scalar log for now
    for i in 0..n {
        if probabilities[i] > 0.0 {
            entropy_sum -= probabilities[i] * probabilities[i].ln();
        }
    }
    
    entropy_sum
}

/// NEON optimized moving average
/// 
/// Fast windowed averaging using NEON
/// Performance target: <150ns for typical windows
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn moving_average_neon(
    signal: &[f64], 
    window: usize, 
    output: &mut [f64]
) {
    debug_assert!(window > 0);
    debug_assert!(signal.len() >= window);
    debug_assert_eq!(output.len(), signal.len() - window + 1);
    
    let inv_window = 1.0 / window as f64;
    
    // Calculate first window sum
    let mut scalar_sum = 0.0;
    
    // Initial window
    for i in 0..window {
        scalar_sum += signal[i];
    }
    output[0] = scalar_sum * inv_window;
    
    // Sliding window
    for i in 1..output.len() {
        scalar_sum = scalar_sum - signal[i - 1] + signal[i + window - 1];
        output[i] = scalar_sum * inv_window;
    }
}

/// NEON optimized variance calculation
/// 
/// Fast statistical variance computation
/// Performance target: <150ns for typical vectors
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn variance_neon(data: &[f64]) -> f64 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    
    // First pass: calculate mean
    let mut sum = vdupq_n_f64(0.0);
    let chunks = n / 2;
    
    for i in 0..chunks {
        let v = vld1q_f64(&data[i * 2]);
        sum = vaddq_f64(sum, v);
    }
    
    let mut total_sum = vaddvq_f64(sum);
    
    for i in (chunks * 2)..n {
        total_sum += data[i];
    }
    
    let mean = total_sum / n as f64;
    let mean_vec = vdupq_n_f64(mean);
    
    // Second pass: calculate variance
    let mut var_sum = vdupq_n_f64(0.0);
    
    for i in 0..chunks {
        let v = vld1q_f64(&data[i * 2]);
        let diff = vsubq_f64(v, mean_vec);
        var_sum = vfmaq_f64(var_sum, diff, diff);
    }
    
    let mut total_var = vaddvq_f64(var_sum);
    
    for i in (chunks * 2)..n {
        let diff = data[i] - mean;
        total_var += diff * diff;
    }
    
    total_var / (n - 1) as f64
}

/// NEON optimized dot product
/// 
/// Fast vector dot product computation
/// Performance target: <50ns for typical vectors
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn dot_product_neon(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    let mut sum = vdupq_n_f64(0.0);
    let chunks = n / 2;
    
    // Unrolled for performance
    let unroll_chunks = chunks / 4;
    for i in 0..unroll_chunks {
        let x0 = vld1q_f64(&x[i * 8]);
        let x1 = vld1q_f64(&x[i * 8 + 2]);
        let x2 = vld1q_f64(&x[i * 8 + 4]);
        let x3 = vld1q_f64(&x[i * 8 + 6]);
        
        let y0 = vld1q_f64(&y[i * 8]);
        let y1 = vld1q_f64(&y[i * 8 + 2]);
        let y2 = vld1q_f64(&y[i * 8 + 4]);
        let y3 = vld1q_f64(&y[i * 8 + 6]);
        
        sum = vfmaq_f64(sum, x0, y0);
        sum = vfmaq_f64(sum, x1, y1);
        sum = vfmaq_f64(sum, x2, y2);
        sum = vfmaq_f64(sum, x3, y3);
    }
    
    // Handle remaining chunks
    for i in (unroll_chunks * 4)..chunks {
        let x_vec = vld1q_f64(&x[i * 2]);
        let y_vec = vld1q_f64(&y[i * 2]);
        sum = vfmaq_f64(sum, x_vec, y_vec);
    }
    
    // Horizontal sum
    let mut total = vaddvq_f64(sum);
    
    // Handle remainder
    for i in (chunks * 2)..n {
        total += x[i] * y[i];
    }
    
    total
}

/// NEON optimized matrix-vector multiplication
/// 
/// Fast matrix-vector product for small matrices
/// Performance target: <500ns for 16x16 matrix
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
pub unsafe fn matrix_vector_multiply_neon(
    matrix: &[f64],  // Row-major, m x n
    vector: &[f64],  // Length n
    output: &mut [f64],  // Length m
    m: usize,
    n: usize
) {
    debug_assert_eq!(matrix.len(), m * n);
    debug_assert_eq!(vector.len(), n);
    debug_assert_eq!(output.len(), m);
    
    for i in 0..m {
        let row = &matrix[i * n..(i + 1) * n];
        output[i] = dot_product_neon(row, vector);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_correlation_neon() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        unsafe {
            let corr = correlation_neon(&x, &y);
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_euclidean_distance_neon() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        
        unsafe {
            let dist = euclidean_distance_neon(&x, &y);
            assert!((dist - 8.0).abs() < 1e-10);  // sqrt(16*4) = 8
        }
    }
    
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_dot_product_neon() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        
        unsafe {
            let dot = dot_product_neon(&x, &y);
            assert_eq!(dot, 40.0);  // 1*2 + 2*3 + 3*4 + 4*5 = 40
        }
    }
}