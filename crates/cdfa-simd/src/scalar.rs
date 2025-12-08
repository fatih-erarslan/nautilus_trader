//! Scalar fallback implementations
//! 
//! Pure Rust implementations without SIMD for maximum compatibility

/// Scalar correlation calculation
pub fn correlation_scalar(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    
    if n == 0 {
        return 0.0;
    }
    
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    
    for i in 0..n {
        sum_x += x[i];
        sum_y += y[i];
        sum_xx += x[i] * x[i];
        sum_yy += y[i] * y[i];
        sum_xy += x[i] * y[i];
    }
    
    let n_f64 = n as f64;
    let numerator = n_f64 * sum_xy - sum_x * sum_y;
    let denominator = ((n_f64 * sum_xx - sum_x * sum_x) * 
                      (n_f64 * sum_yy - sum_y * sum_y)).sqrt();
    
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Scalar Euclidean distance
pub fn euclidean_distance_scalar(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    
    let mut sum = 0.0;
    for i in 0..x.len() {
        let diff = x[i] - y[i];
        sum += diff * diff;
    }
    
    sum.sqrt()
}

/// Scalar DWT Haar transform
pub fn dwt_haar_scalar(signal: &[f64], approx: &mut [f64], detail: &mut [f64]) {
    let n = signal.len();
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    
    for i in 0..n/2 {
        let even_idx = i * 2;
        let odd_idx = even_idx + 1;
        approx[i] = (signal[even_idx] + signal[odd_idx]) * sqrt2_inv;
        detail[i] = (signal[even_idx] - signal[odd_idx]) * sqrt2_inv;
    }
}

/// Scalar matrix multiplication
pub fn matrix_multiply_scalar(a: &[f64], b: &[f64], c: &mut [f64], m: usize, n: usize, k: usize) {
    // Initialize output to zero
    for i in 0..m*n {
        c[i] = 0.0;
    }
    
    // Standard triple loop with cache-friendly order (i-k-j)
    for i in 0..m {
        for kk in 0..k {
            let a_val = a[i * k + kk];
            for j in 0..n {
                c[i * n + j] += a_val * b[kk * n + j];
            }
        }
    }
}

/// Scalar softmax
pub fn softmax_scalar(input: &[f64], output: &mut [f64]) {
    let n = input.len();
    if n == 0 {
        return;
    }
    
    // Find max for numerical stability
    let max_val = input.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    
    // Compute exp(x - max) and sum
    let mut sum = 0.0;
    for i in 0..n {
        output[i] = (input[i] - max_val).exp();
        sum += output[i];
    }
    
    // Normalize
    let sum_inv = 1.0 / sum;
    for i in 0..n {
        output[i] *= sum_inv;
    }
}