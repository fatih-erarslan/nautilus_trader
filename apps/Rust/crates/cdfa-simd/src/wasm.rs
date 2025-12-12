//! WebAssembly SIMD optimizations
//! 
//! High-performance implementations using 128-bit WASM SIMD
//! Target: Near-native performance in web browsers and WASM runtimes

#[cfg(target_arch = "wasm32")]
use std::arch::wasm32::*;
use crate::AlignedVec;

/// WASM SIMD optimized correlation calculation
/// 
/// Computes Pearson correlation using WASM SIMD intrinsics
/// Performance target: <200ns for 256 element vectors
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn correlation_wasm(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    if n == 0 {
        return 0.0;
    }
    
    // Process 2 doubles at a time with WASM SIMD
    let chunks = n / 2;
    
    // Accumulators
    let mut sum_x = f64x2_splat(0.0);
    let mut sum_y = f64x2_splat(0.0);
    let mut sum_xx = f64x2_splat(0.0);
    let mut sum_yy = f64x2_splat(0.0);
    let mut sum_xy = f64x2_splat(0.0);
    
    // Main vectorized loop
    for i in 0..chunks {
        let x_vec = v128_load(&x[i * 2] as *const f64 as *const v128);
        let y_vec = v128_load(&y[i * 2] as *const f64 as *const v128);
        
        sum_x = f64x2_add(sum_x, x_vec);
        sum_y = f64x2_add(sum_y, y_vec);
        sum_xx = f64x2_add(sum_xx, f64x2_mul(x_vec, x_vec));
        sum_yy = f64x2_add(sum_yy, f64x2_mul(y_vec, y_vec));
        sum_xy = f64x2_add(sum_xy, f64x2_mul(x_vec, y_vec));
    }
    
    // Horizontal sum
    let sum_x_scalar = f64x2_extract_lane::<0>(sum_x) + f64x2_extract_lane::<1>(sum_x);
    let sum_y_scalar = f64x2_extract_lane::<0>(sum_y) + f64x2_extract_lane::<1>(sum_y);
    let sum_xx_scalar = f64x2_extract_lane::<0>(sum_xx) + f64x2_extract_lane::<1>(sum_xx);
    let sum_yy_scalar = f64x2_extract_lane::<0>(sum_yy) + f64x2_extract_lane::<1>(sum_yy);
    let sum_xy_scalar = f64x2_extract_lane::<0>(sum_xy) + f64x2_extract_lane::<1>(sum_xy);
    
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

/// WASM SIMD optimized wavelet transform (Haar wavelet)
/// 
/// Fast discrete wavelet transform for WebAssembly
/// Performance target: <200ns for small transforms
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn dwt_haar_wasm(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
    let n = signal.len();
    debug_assert_eq!(n % 2, 0);
    debug_assert_eq!(approx.len(), n / 2);
    debug_assert_eq!(detail.len(), n / 2);
    
    let sqrt2_inv = f64x2_splat(1.0 / 2.0_f64.sqrt());
    let half_n = n / 2;
    
    // Process 2 pairs at a time
    let chunks = half_n / 2;
    
    for i in 0..chunks {
        // Load 4 consecutive values
        let v1 = v128_load(&signal[i * 4] as *const f64 as *const v128);
        let v2 = v128_load(&signal[i * 4 + 2] as *const f64 as *const v128);
        
        // Extract even and odd elements
        // For WASM SIMD, we need to use shuffles
        let evens = f64x2_make(
            f64x2_extract_lane::<0>(v1),  // signal[i*4]
            f64x2_extract_lane::<0>(v2)   // signal[i*4 + 2]
        );
        
        let odds = f64x2_make(
            f64x2_extract_lane::<1>(v1),  // signal[i*4 + 1]
            f64x2_extract_lane::<1>(v2)   // signal[i*4 + 3]
        );
        
        // Calculate approximation and detail coefficients
        let sum = f64x2_add(evens, odds);
        let diff = f64x2_sub(evens, odds);
        
        let approx_vals = f64x2_mul(sum, sqrt2_inv);
        let detail_vals = f64x2_mul(diff, sqrt2_inv);
        
        v128_store(&mut approx[i * 2] as *mut f64 as *mut v128, approx_vals);
        v128_store(&mut detail[i * 2] as *mut f64 as *mut v128, detail_vals);
    }
    
    // Handle remainder
    for i in (chunks * 2)..half_n {
        let even = signal[2 * i];
        let odd = signal[2 * i + 1];
        approx[i] = (even + odd) / 2.0_f64.sqrt();
        detail[i] = (even - odd) / 2.0_f64.sqrt();
    }
}

/// WASM SIMD optimized Euclidean distance
/// 
/// Vectorized L2 distance computation
/// Performance target: <150ns for 256 element vectors
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn euclidean_distance_wasm(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    let mut sum = f64x2_splat(0.0);
    let chunks = n / 2;
    
    // Unrolled loop for better performance
    let unroll_chunks = chunks / 4;
    for i in 0..unroll_chunks {
        let x0 = v128_load(&x[i * 8] as *const f64 as *const v128);
        let x1 = v128_load(&x[i * 8 + 2] as *const f64 as *const v128);
        let x2 = v128_load(&x[i * 8 + 4] as *const f64 as *const v128);
        let x3 = v128_load(&x[i * 8 + 6] as *const f64 as *const v128);
        
        let y0 = v128_load(&y[i * 8] as *const f64 as *const v128);
        let y1 = v128_load(&y[i * 8 + 2] as *const f64 as *const v128);
        let y2 = v128_load(&y[i * 8 + 4] as *const f64 as *const v128);
        let y3 = v128_load(&y[i * 8 + 6] as *const f64 as *const v128);
        
        let diff0 = f64x2_sub(x0, y0);
        let diff1 = f64x2_sub(x1, y1);
        let diff2 = f64x2_sub(x2, y2);
        let diff3 = f64x2_sub(x3, y3);
        
        sum = f64x2_add(sum, f64x2_mul(diff0, diff0));
        sum = f64x2_add(sum, f64x2_mul(diff1, diff1));
        sum = f64x2_add(sum, f64x2_mul(diff2, diff2));
        sum = f64x2_add(sum, f64x2_mul(diff3, diff3));
    }
    
    // Handle remaining chunks
    for i in (unroll_chunks * 4)..chunks {
        let x_vec = v128_load(&x[i * 2] as *const f64 as *const v128);
        let y_vec = v128_load(&y[i * 2] as *const f64 as *const v128);
        let diff = f64x2_sub(x_vec, y_vec);
        sum = f64x2_add(sum, f64x2_mul(diff, diff));
    }
    
    // Horizontal sum
    let mut total = f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum);
    
    // Handle remainder
    for i in (chunks * 2)..n {
        let diff = x[i] - y[i];
        total += diff * diff;
    }
    
    total.sqrt()
}

/// WASM SIMD optimized signal fusion
/// 
/// Weighted combination of multiple signals
/// Performance target: <400ns for fusion operation
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn signal_fusion_wasm(
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
    let weight_vecs: Vec<v128> = weights.iter()
        .map(|&w| f64x2_splat(w))
        .collect();
    
    for i in 0..chunks {
        let mut sum = f64x2_splat(0.0);
        
        for (signal, &weight_vec) in signals.iter().zip(&weight_vecs) {
            let signal_vec = v128_load(&signal[i * 2] as *const f64 as *const v128);
            sum = f64x2_add(sum, f64x2_mul(signal_vec, weight_vec));
        }
        
        v128_store(&mut output[i * 2] as *mut f64 as *mut v128, sum);
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

/// WASM SIMD optimized moving average
/// 
/// Fast windowed averaging using WASM SIMD
/// Performance target: <200ns for typical windows
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn moving_average_wasm(
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

/// WASM SIMD optimized variance calculation
/// 
/// Fast statistical variance computation
/// Performance target: <200ns for typical vectors
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn variance_wasm(data: &[f64]) -> f64 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    
    // First pass: calculate mean
    let mut sum = f64x2_splat(0.0);
    let chunks = n / 2;
    
    for i in 0..chunks {
        let v = v128_load(&data[i * 2] as *const f64 as *const v128);
        sum = f64x2_add(sum, v);
    }
    
    let mut total_sum = f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum);
    
    for i in (chunks * 2)..n {
        total_sum += data[i];
    }
    
    let mean = total_sum / n as f64;
    let mean_vec = f64x2_splat(mean);
    
    // Second pass: calculate variance
    let mut var_sum = f64x2_splat(0.0);
    
    for i in 0..chunks {
        let v = v128_load(&data[i * 2] as *const f64 as *const v128);
        let diff = f64x2_sub(v, mean_vec);
        var_sum = f64x2_add(var_sum, f64x2_mul(diff, diff));
    }
    
    let mut total_var = f64x2_extract_lane::<0>(var_sum) + f64x2_extract_lane::<1>(var_sum);
    
    for i in (chunks * 2)..n {
        let diff = data[i] - mean;
        total_var += diff * diff;
    }
    
    total_var / (n - 1) as f64
}

/// WASM SIMD optimized dot product
/// 
/// Fast vector dot product computation
/// Performance target: <100ns for typical vectors
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn dot_product_wasm(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    let mut sum = f64x2_splat(0.0);
    let chunks = n / 2;
    
    // Unrolled for performance
    let unroll_chunks = chunks / 4;
    for i in 0..unroll_chunks {
        let x0 = v128_load(&x[i * 8] as *const f64 as *const v128);
        let x1 = v128_load(&x[i * 8 + 2] as *const f64 as *const v128);
        let x2 = v128_load(&x[i * 8 + 4] as *const f64 as *const v128);
        let x3 = v128_load(&x[i * 8 + 6] as *const f64 as *const v128);
        
        let y0 = v128_load(&y[i * 8] as *const f64 as *const v128);
        let y1 = v128_load(&y[i * 8 + 2] as *const f64 as *const v128);
        let y2 = v128_load(&y[i * 8 + 4] as *const f64 as *const v128);
        let y3 = v128_load(&y[i * 8 + 6] as *const f64 as *const v128);
        
        sum = f64x2_add(sum, f64x2_mul(x0, y0));
        sum = f64x2_add(sum, f64x2_mul(x1, y1));
        sum = f64x2_add(sum, f64x2_mul(x2, y2));
        sum = f64x2_add(sum, f64x2_mul(x3, y3));
    }
    
    // Handle remaining chunks
    for i in (unroll_chunks * 4)..chunks {
        let x_vec = v128_load(&x[i * 2] as *const f64 as *const v128);
        let y_vec = v128_load(&y[i * 2] as *const f64 as *const v128);
        sum = f64x2_add(sum, f64x2_mul(x_vec, y_vec));
    }
    
    // Horizontal sum
    let mut total = f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum);
    
    // Handle remainder
    for i in (chunks * 2)..n {
        total += x[i] * y[i];
    }
    
    total
}

/// WASM SIMD optimized min/max finding
/// 
/// Fast minimum and maximum element search
/// Performance target: <100ns for typical vectors
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn minmax_wasm(data: &[f64]) -> (f64, f64) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    let n = data.len();
    let chunks = n / 2;
    
    let mut min_vec = f64x2_splat(data[0]);
    let mut max_vec = f64x2_splat(data[0]);
    
    for i in 0..chunks {
        let v = v128_load(&data[i * 2] as *const f64 as *const v128);
        min_vec = f64x2_pmin(min_vec, v);
        max_vec = f64x2_pmax(max_vec, v);
    }
    
    // Horizontal min/max
    let min_0 = f64x2_extract_lane::<0>(min_vec);
    let min_1 = f64x2_extract_lane::<1>(min_vec);
    let max_0 = f64x2_extract_lane::<0>(max_vec);
    let max_1 = f64x2_extract_lane::<1>(max_vec);
    
    let mut min = min_0.min(min_1);
    let mut max = max_0.max(max_1);
    
    // Handle remainder
    for i in (chunks * 2)..n {
        min = min.min(data[i]);
        max = max.max(data[i]);
    }
    
    (min, max)
}

/// WASM SIMD optimized sum of absolute differences
/// 
/// Fast SAD computation for signal comparison
/// Performance target: <150ns for typical vectors
#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
pub unsafe fn sum_absolute_differences_wasm(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    let mut sum = f64x2_splat(0.0);
    let chunks = n / 2;
    
    for i in 0..chunks {
        let x_vec = v128_load(&x[i * 2] as *const f64 as *const v128);
        let y_vec = v128_load(&y[i * 2] as *const f64 as *const v128);
        let diff = f64x2_sub(x_vec, y_vec);
        sum = f64x2_add(sum, f64x2_abs(diff));
    }
    
    // Horizontal sum
    let mut total = f64x2_extract_lane::<0>(sum) + f64x2_extract_lane::<1>(sum);
    
    // Handle remainder
    for i in (chunks * 2)..n {
        total += (x[i] - y[i]).abs();
    }
    
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_correlation_wasm() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        unsafe {
            let corr = correlation_wasm(&x, &y);
            assert!((corr - 1.0).abs() < 1e-10);
        }
    }
    
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_euclidean_distance_wasm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![5.0, 6.0, 7.0, 8.0];
        
        unsafe {
            let dist = euclidean_distance_wasm(&x, &y);
            assert!((dist - 8.0).abs() < 1e-10);  // sqrt(16*4) = 8
        }
    }
    
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_dot_product_wasm() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![2.0, 3.0, 4.0, 5.0];
        
        unsafe {
            let dot = dot_product_wasm(&x, &y);
            assert_eq!(dot, 40.0);  // 1*2 + 2*3 + 3*4 + 4*5 = 40
        }
    }
    
    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_minmax_wasm() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        
        unsafe {
            let (min, max) = minmax_wasm(&data);
            assert_eq!(min, 1.0);
            assert_eq!(max, 9.0);
        }
    }
}