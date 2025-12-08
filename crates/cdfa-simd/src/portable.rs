//! Portable SIMD implementations using the `wide` crate
//! 
//! Works on stable Rust across all platforms

use wide::f64x4;

/// Portable SIMD correlation calculation
pub fn correlation_portable(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    if n == 0 {
        return 0.0;
    }
    
    // Process 4 doubles at a time
    let chunks = n / 4;
    let remainder = n % 4;
    
    // Accumulators
    let mut sum_x = f64x4::ZERO;
    let mut sum_y = f64x4::ZERO;
    let mut sum_xx = f64x4::ZERO;
    let mut sum_yy = f64x4::ZERO;
    let mut sum_xy = f64x4::ZERO;
    
    // Process chunks
    for i in 0..chunks {
        let base = i * 4;
        let x_vec = f64x4::new([x[base], x[base + 1], x[base + 2], x[base + 3]]);
        let y_vec = f64x4::new([y[base], y[base + 1], y[base + 2], y[base + 3]]);
        
        sum_x += x_vec;
        sum_y += y_vec;
        sum_xx += x_vec * x_vec;
        sum_yy += y_vec * y_vec;
        sum_xy += x_vec * y_vec;
    }
    
    // Reduce SIMD accumulators
    let mut sum_x_scalar = sum_x.reduce_add();
    let mut sum_y_scalar = sum_y.reduce_add();
    let mut sum_xx_scalar = sum_xx.reduce_add();
    let mut sum_yy_scalar = sum_yy.reduce_add();
    let mut sum_xy_scalar = sum_xy.reduce_add();
    
    // Handle remainder
    for i in (chunks * 4)..n {
        sum_x_scalar += x[i];
        sum_y_scalar += y[i];
        sum_xx_scalar += x[i] * x[i];
        sum_yy_scalar += y[i] * y[i];
        sum_xy_scalar += x[i] * y[i];
    }
    
    // Calculate correlation
    let n_f64 = n as f64;
    let numerator = n_f64 * sum_xy_scalar - sum_x_scalar * sum_y_scalar;
    let denominator = ((n_f64 * sum_xx_scalar - sum_x_scalar * sum_x_scalar) *
                      (n_f64 * sum_yy_scalar - sum_y_scalar * sum_y_scalar)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Portable SIMD Euclidean distance
pub fn euclidean_distance_portable(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    if n == 0 {
        return 0.0;
    }
    
    let chunks = n / 4;
    let remainder = n % 4;
    
    let mut sum = f64x4::ZERO;
    
    // Process chunks
    for i in 0..chunks {
        let base = i * 4;
        let x_vec = f64x4::new([x[base], x[base + 1], x[base + 2], x[base + 3]]);
        let y_vec = f64x4::new([y[base], y[base + 1], y[base + 2], y[base + 3]]);
        let diff = x_vec - y_vec;
        sum += diff * diff;
    }
    
    let mut sum_scalar = sum.reduce_add();
    
    // Handle remainder
    for i in (chunks * 4)..n {
        let diff = x[i] - y[i];
        sum_scalar += diff * diff;
    }
    
    sum_scalar.sqrt()
}

/// Portable SIMD DWT Haar transform
pub fn dwt_haar_portable(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
    let n = signal.len();
    let half_n = n / 2;
    
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    let sqrt2_inv_vec = f64x4::splat(sqrt2_inv);
    
    let chunks = half_n / 4;
    let remainder = half_n % 4;
    
    // Process chunks
    for i in 0..chunks {
        let base_in = i * 8;
        let base_out = i * 4;
        
        // Load pairs
        let even1 = f64x4::new([signal[base_in], signal[base_in + 2], 
                                signal[base_in + 4], signal[base_in + 6]]);
        let odd1 = f64x4::new([signal[base_in + 1], signal[base_in + 3], 
                               signal[base_in + 5], signal[base_in + 7]]);
        
        let approx_vec = (even1 + odd1) * sqrt2_inv_vec;
        let detail_vec = (even1 - odd1) * sqrt2_inv_vec;
        
        // Store results
        approx_vec.to_array().iter().enumerate().for_each(|(j, &val)| {
            approx[base_out + j] = val;
        });
        detail_vec.to_array().iter().enumerate().for_each(|(j, &val)| {
            detail[base_out + j] = val;
        });
    }
    
    // Handle remainder
    for i in (chunks * 4)..half_n {
        let even_idx = i * 2;
        let odd_idx = even_idx + 1;
        approx[i] = (signal[even_idx] + signal[odd_idx]) * sqrt2_inv;
        detail[i] = (signal[even_idx] - signal[odd_idx]) * sqrt2_inv;
    }
}

/// Portable SIMD matrix multiplication
pub fn matrix_multiply_portable(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    // Clear output matrix
    c.iter_mut().for_each(|x| *x = 0.0);
    
    // Use tiled approach for cache efficiency
    const TILE_SIZE: usize = 64;
    
    for i_tile in (0..m).step_by(TILE_SIZE) {
        let i_end = (i_tile + TILE_SIZE).min(m);
        
        for k_tile in (0..k).step_by(TILE_SIZE) {
            let k_end = (k_tile + TILE_SIZE).min(k);
            
            for j_tile in (0..n).step_by(TILE_SIZE) {
                let j_end = (j_tile + TILE_SIZE).min(n);
                
                // Process tile
                for i in i_tile..i_end {
                    for kk in k_tile..k_end {
                        let a_val = a[i * k + kk];
                        let a_vec = f64x4::splat(a_val);
                        
                        let j_chunks = (j_end - j_tile) / 4;
                        
                        for j_idx in 0..j_chunks {
                            let j = j_tile + j_idx * 4;
                            let b_vec = f64x4::new([
                                b[kk * n + j],
                                b[kk * n + j + 1],
                                b[kk * n + j + 2],
                                b[kk * n + j + 3],
                            ]);
                            
                            let c_vec = f64x4::new([
                                c[i * n + j],
                                c[i * n + j + 1],
                                c[i * n + j + 2],
                                c[i * n + j + 3],
                            ]);
                            
                            let result = c_vec + a_vec * b_vec;
                            let result_arr = result.to_array();
                            
                            c[i * n + j] = result_arr[0];
                            c[i * n + j + 1] = result_arr[1];
                            c[i * n + j + 2] = result_arr[2];
                            c[i * n + j + 3] = result_arr[3];
                        }
                        
                        // Handle remainder
                        for j in (j_tile + j_chunks * 4)..j_end {
                            c[i * n + j] += a_val * b[kk * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Portable SIMD softmax
pub fn softmax_portable(input: &[f64], output: &mut [f64]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    
    // Find max for numerical stability
    let mut max_val = input[0];
    for &val in input.iter() {
        if val > max_val {
            max_val = val;
        }
    }
    
    let max_vec = f64x4::splat(max_val);
    
    // Compute exp(x - max) and sum
    let chunks = n / 4;
    let remainder = n % 4;
    
    let mut sum = 0.0;
    
    // Process chunks
    for i in 0..chunks {
        let base = i * 4;
        let x_vec = f64x4::new([input[base], input[base + 1], input[base + 2], input[base + 3]]);
        let exp_vec = (x_vec - max_vec).exp();
        let exp_arr = exp_vec.to_array();
        
        for j in 0..4 {
            output[base + j] = exp_arr[j];
            sum += exp_arr[j];
        }
    }
    
    // Handle remainder
    for i in (chunks * 4)..n {
        output[i] = (input[i] - max_val).exp();
        sum += output[i];
    }
    
    // Normalize
    let sum_inv = 1.0 / sum;
    let sum_inv_vec = f64x4::splat(sum_inv);
    
    // Process chunks
    for i in 0..chunks {
        let base = i * 4;
        let out_vec = f64x4::new([output[base], output[base + 1], output[base + 2], output[base + 3]]);
        let normalized = out_vec * sum_inv_vec;
        let norm_arr = normalized.to_array();
        
        for j in 0..4 {
            output[base + j] = norm_arr[j];
        }
    }
    
    // Handle remainder
    for i in (chunks * 4)..n {
        output[i] *= sum_inv;
    }
}