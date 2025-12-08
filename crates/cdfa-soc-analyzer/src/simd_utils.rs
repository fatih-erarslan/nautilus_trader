//! SIMD utility functions for optimized calculations

use wide::{f32x8, CmpGt};

/// SIMD-optimized mean calculation
#[inline]
pub fn simd_mean(data: &[f32]) -> f32 {
    let n = data.len();
    if n == 0 {
        return 0.0;
    }
    
    if n >= 8 {
        let mut sum = f32x8::splat(0.0);
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            sum += f32x8::from(chunk);
        }
        
        (sum.reduce_add() + remainder.iter().sum::<f32>()) / n as f32
    } else {
        data.iter().sum::<f32>() / n as f32
    }
}

/// SIMD-optimized standard deviation calculation
#[inline]
pub fn simd_std_dev(data: &[f32], mean: f32) -> f32 {
    let n = data.len();
    if n <= 1 {
        return 0.0;
    }
    
    if n >= 8 {
        let mean_simd = f32x8::splat(mean);
        let mut sum_sq = f32x8::splat(0.0);
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vals = f32x8::from(chunk);
            let diff = vals - mean_simd;
            sum_sq += diff * diff;
        }
        
        let sum_sq_scalar = sum_sq.reduce_add() + 
            remainder.iter().map(|&x| (x - mean).powi(2)).sum::<f32>();
        
        (sum_sq_scalar / (n - 1) as f32).sqrt()
    } else {
        let sum_sq: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum();
        (sum_sq / (n - 1) as f32).sqrt()
    }
}

/// SIMD-optimized min/max calculation
#[inline]
pub fn simd_min_max(data: &[f32]) -> (f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0);
    }
    
    if data.len() >= 8 {
        let mut min_simd = f32x8::splat(f32::INFINITY);
        let mut max_simd = f32x8::splat(f32::NEG_INFINITY);
        
        let chunks = data.chunks_exact(8);
        let remainder = chunks.remainder();
        
        for chunk in chunks {
            let vals = f32x8::from(chunk);
            min_simd = min_simd.min(vals);
            max_simd = max_simd.max(vals);
        }
        
        let mut min_val = min_simd.to_array().iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let mut max_val = max_simd.to_array().iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        for &x in remainder {
            min_val = min_val.min(x);
            max_val = max_val.max(x);
        }
        
        (min_val, max_val)
    } else {
        data.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &x| {
            (min.min(x), max.max(x))
        })
    }
}

/// SIMD-optimized dot product
#[inline]
pub fn simd_dot_product(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    
    if n >= 8 {
        let mut sum = f32x8::splat(0.0);
        let chunks_a = a[..n].chunks_exact(8);
        let chunks_b = b[..n].chunks_exact(8);
        let remainder_len = chunks_a.remainder().len();
        
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let a_simd = f32x8::from(chunk_a);
            let b_simd = f32x8::from(chunk_b);
            sum += a_simd * b_simd;
        }
        
        let mut result = sum.reduce_add();
        
        // Handle remainder
        if remainder_len > 0 {
            let start = n - remainder_len;
            for i in 0..remainder_len {
                result += a[start + i] * b[start + i];
            }
        }
        
        result
    } else {
        a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
    }
}

/// SIMD-optimized element-wise operations
pub struct SimdOps;

impl SimdOps {
    /// Add scalar to all elements
    #[inline]
    pub fn add_scalar(data: &mut [f32], scalar: f32) {
        if data.len() >= 8 {
            let scalar_simd = f32x8::splat(scalar);
            
            for chunk in data.chunks_exact_mut(8) {
                let vals = f32x8::from(&chunk[..]);
                let result = vals + scalar_simd;
                chunk.copy_from_slice(&result.to_array());
            }
            
            // Handle remainder
            let remainder_start = data.len() / 8 * 8;
            for i in remainder_start..data.len() {
                data[i] += scalar;
            }
        } else {
            for x in data {
                *x += scalar;
            }
        }
    }
    
    /// Multiply all elements by scalar
    #[inline]
    pub fn mul_scalar(data: &mut [f32], scalar: f32) {
        if data.len() >= 8 {
            let scalar_simd = f32x8::splat(scalar);
            
            for chunk in data.chunks_exact_mut(8) {
                let vals = f32x8::from(&chunk[..]);
                let result = vals * scalar_simd;
                chunk.copy_from_slice(&result.to_array());
            }
            
            // Handle remainder
            let remainder_start = data.len() / 8 * 8;
            for i in remainder_start..data.len() {
                data[i] *= scalar;
            }
        } else {
            for x in data {
                *x *= scalar;
            }
        }
    }
    
    /// Apply threshold (set values below threshold to zero)
    #[inline]
    pub fn threshold(data: &mut [f32], threshold: f32) {
        if data.len() >= 8 {
            let threshold_simd = f32x8::splat(threshold);
            let zero_simd = f32x8::splat(0.0);
            
            for chunk in data.chunks_exact_mut(8) {
                let vals = f32x8::from(&chunk[..]);
                let mask = vals.cmp_gt(threshold_simd);
                let result = mask.blend(vals, zero_simd);
                chunk.copy_from_slice(&result.to_array());
            }
            
            // Handle remainder
            let remainder_start = data.len() / 8 * 8;
            for i in remainder_start..data.len() {
                if data[i] <= threshold {
                    data[i] = 0.0;
                }
            }
        } else {
            for x in data {
                if *x <= threshold {
                    *x = 0.0;
                }
            }
        }
    }
    
    /// Clip values to range [min, max]
    #[inline]
    pub fn clip(data: &mut [f32], min: f32, max: f32) {
        if data.len() >= 8 {
            let min_simd = f32x8::splat(min);
            let max_simd = f32x8::splat(max);
            
            for chunk in data.chunks_exact_mut(8) {
                let vals = f32x8::from(&chunk[..]);
                let result = vals.max(min_simd).min(max_simd);
                chunk.copy_from_slice(&result.to_array());
            }
            
            // Handle remainder
            let remainder_start = data.len() / 8 * 8;
            for i in remainder_start..data.len() {
                data[i] = data[i].max(min).min(max);
            }
        } else {
            for x in data {
                *x = x.max(min).min(max);
            }
        }
    }
}

/// SIMD-optimized moving average
#[inline]
pub fn simd_moving_average(data: &[f32], window: usize) -> Vec<f32> {
    let n = data.len();
    if n < window || window == 0 {
        return vec![];
    }
    
    let mut result = vec![0.0f32; n - window + 1];
    let window_f32 = window as f32;
    
    // Calculate first window sum
    let mut window_sum = data[..window].iter().sum::<f32>();
    result[0] = window_sum / window_f32;
    
    // Sliding window
    for i in 1..result.len() {
        window_sum = window_sum - data[i - 1] + data[i + window - 1];
        result[i] = window_sum / window_f32;
    }
    
    result
}

/// SIMD-optimized percentile calculation
pub fn simd_percentile(data: &mut [f32], percentile: f32) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    
    // Sort data (using standard sort for now, could be optimized with SIMD sorting networks)
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    
    let n = data.len() as f32;
    let index = (percentile / 100.0 * (n - 1.0)) as usize;
    let fraction = percentile / 100.0 * (n - 1.0) - index as f32;
    
    if index + 1 < data.len() {
        data[index] * (1.0 - fraction) + data[index + 1] * fraction
    } else {
        data[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simd_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mean = simd_mean(&data);
        assert_relative_eq!(mean, 5.5, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = simd_mean(&data);
        let std_dev = simd_std_dev(&data, mean);
        assert_relative_eq!(std_dev, 1.5811388, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_min_max() {
        let data = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 4.0, 6.0];
        let (min, max) = simd_min_max(&data);
        assert_eq!(min, 1.0);
        assert_eq!(max, 9.0);
    }
    
    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let dot = simd_dot_product(&a, &b);
        assert_eq!(dot, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
    }
    
    #[test]
    fn test_simd_ops() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        
        // Test add scalar
        SimdOps::add_scalar(&mut data, 2.0);
        assert_eq!(data, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
        
        // Test mul scalar
        SimdOps::mul_scalar(&mut data, 0.5);
        assert_eq!(data, vec![1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]);
        
        // Test threshold
        SimdOps::threshold(&mut data, 3.0);
        assert_eq!(data, vec![0.0, 0.0, 0.0, 0.0, 3.5, 4.0, 4.5, 5.0]);
        
        // Test clip
        data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        SimdOps::clip(&mut data, 2.5, 6.5);
        assert_eq!(data, vec![2.5, 2.5, 3.0, 4.0, 5.0, 6.0, 6.5, 6.5]);
    }
    
    #[test]
    fn test_moving_average() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let ma = simd_moving_average(&data, 3);
        assert_eq!(ma.len(), 8);
        assert_relative_eq!(ma[0], 2.0, epsilon = 1e-6);
        assert_relative_eq!(ma[1], 3.0, epsilon = 1e-6);
        assert_relative_eq!(ma[7], 9.0, epsilon = 1e-6);
    }
}