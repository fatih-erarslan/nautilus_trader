//! Entropy calculations with SIMD optimization

use crate::{Result, SOCError};
use ndarray::ArrayView1;
use wide::f32x8;

/// Calculate sample entropy using SIMD optimization
pub fn sample_entropy(data: ArrayView1<f64>, m: usize, r: f64) -> Result<f64> {
    let n = data.len();
    if n < m + 2 {
        return Ok(0.5); // Default for too short series
    }
    
    // Convert to f32 for SIMD (maintaining precision for our use case)
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    
    // Calculate mean and standard deviation
    let mean = data_f32.iter().sum::<f32>() / n as f32;
    let variance = data_f32.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    let std_dev = variance.sqrt();
    
    if std_dev < 1e-9 {
        return Ok(0.5); // Default for constant series
    }
    
    let tolerance = (r as f32) * std_dev;
    
    // Count template matches for patterns of length m and m+1
    let count_m = count_template_matches_simd(&data_f32, m, tolerance)?;
    let count_m1 = count_template_matches_simd(&data_f32, m + 1, tolerance)?;
    
    if count_m == 0 || count_m1 == 0 {
        return Ok(0.5);
    }
    
    let sample_entropy = -(count_m1 as f64 / count_m as f64).ln();
    Ok(sample_entropy)
}

/// Calculate entropy rate using conditional entropy
pub fn entropy_rate(data: ArrayView1<f64>, lag: usize, n_bins: usize) -> Result<f64> {
    let n = data.len();
    if n <= lag {
        return Ok(0.0);
    }
    
    // Create histogram bins
    let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max_val - min_val).abs() < 1e-12 {
        return Ok(0.0); // Constant series has zero entropy rate
    }
    
    let bin_width = (max_val - min_val) / n_bins as f64;
    
    // Count joint occurrences P(X_t, X_{t-lag})
    let mut joint_counts = vec![vec![0u32; n_bins]; n_bins];
    let mut marginal_counts = vec![0u32; n_bins];
    
    for i in lag..n {
        let bin_curr = ((data[i] - min_val) / bin_width).floor() as usize;
        let bin_prev = ((data[i - lag] - min_val) / bin_width).floor() as usize;
        
        let bin_curr = bin_curr.min(n_bins - 1);
        let bin_prev = bin_prev.min(n_bins - 1);
        
        joint_counts[bin_prev][bin_curr] += 1;
        marginal_counts[bin_prev] += 1;
    }
    
    let total_count = (n - lag) as f64;
    
    // Calculate conditional entropy H(X_t | X_{t-lag})
    let mut conditional_entropy = 0.0;
    
    for i in 0..n_bins {
        if marginal_counts[i] > 0 {
            let p_prev = marginal_counts[i] as f64 / total_count;
            
            for j in 0..n_bins {
                if joint_counts[i][j] > 0 {
                    let p_joint = joint_counts[i][j] as f64 / total_count;
                    let p_conditional = joint_counts[i][j] as f64 / marginal_counts[i] as f64;
                    
                    conditional_entropy -= p_joint * p_conditional.ln();
                }
            }
        }
    }
    
    Ok(conditional_entropy)
}

/// Count template matches using SIMD acceleration
fn count_template_matches_simd(data: &[f32], m: usize, tolerance: f32) -> Result<u32> {
    let n = data.len();
    if n < m + 1 {
        return Ok(0);
    }
    
    let mut count = 0u32;
    let tolerance_vec = f32x8::splat(tolerance);
    
    // Main SIMD loop for template matching
    for i in 0..=(n - m) {
        for j in (i + 1)..=(n - m) {
            if j + 7 < n - m + 1 {
                // SIMD comparison for 8 templates at once
                let mut all_match = true;
                
                for k in 0..m {
                    let template_val = f32x8::splat(data[i + k]);
                    
                    // Load 8 comparison values
                    let compare_vals = if j + k + 7 < n {
                        f32x8::new([
                            data[j + k],
                            data[j + k + 1],
                            data[j + k + 2], 
                            data[j + k + 3],
                            data[j + k + 4],
                            data[j + k + 5],
                            data[j + k + 6],
                            data[j + k + 7],
                        ])
                    } else {
                        // Handle boundary case
                        let mut vals = [0.0f32; 8];
                        for l in 0..8 {
                            if j + k + l < n {
                                vals[l] = data[j + k + l];
                            }
                        }
                        f32x8::new(vals)
                    };
                    
                    let diff = (template_val - compare_vals).abs();
                    let within_tolerance = diff.cmp_le(tolerance_vec);
                    
                    if !within_tolerance.all() {
                        all_match = false;
                        break;
                    }
                }
                
                if all_match {
                    count += 1;
                }
            } else {
                // Fallback to scalar comparison for remaining elements
                let mut matches = true;
                for k in 0..m {
                    if (data[i + k] - data[j + k]).abs() > tolerance {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    count += 1;
                }
            }
        }
    }
    
    Ok(count)
}

/// Calculate approximate entropy (alternative to sample entropy)
pub fn approximate_entropy(data: ArrayView1<f64>, m: usize, r: f64) -> Result<f64> {
    let n = data.len();
    if n < m + 1 {
        return Ok(0.0);
    }
    
    let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
    
    // Calculate mean and std
    let mean = data_f32.iter().sum::<f32>() / n as f32;
    let variance = data_f32.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
    let std_dev = variance.sqrt();
    
    if std_dev < 1e-9 {
        return Ok(0.0);
    }
    
    let tolerance = (r as f32) * std_dev;
    
    // Calculate phi(m) and phi(m+1)
    let phi_m = calculate_phi(&data_f32, m, tolerance)?;
    let phi_m1 = calculate_phi(&data_f32, m + 1, tolerance)?;
    
    Ok(phi_m - phi_m1)
}

/// Calculate phi function for approximate entropy
fn calculate_phi(data: &[f32], m: usize, tolerance: f32) -> Result<f64> {
    let n = data.len();
    let mut phi = 0.0;
    
    for i in 0..=(n - m) {
        let mut matches = 0;
        
        for j in 0..=(n - m) {
            let mut is_match = true;
            for k in 0..m {
                if (data[i + k] - data[j + k]).abs() > tolerance {
                    is_match = false;
                    break;
                }
            }
            if is_match {
                matches += 1;
            }
        }
        
        let pattern_prob = matches as f64 / (n - m + 1) as f64;
        if pattern_prob > 0.0 {
            phi += pattern_prob.ln();
        }
    }
    
    phi /= (n - m + 1) as f64;
    Ok(phi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;
    
    #[test]
    fn test_sample_entropy() {
        let data = Array1::from_vec((0..100).map(|x| (x as f64 * 0.1).sin()).collect());
        let result = sample_entropy(data.view(), 2, 0.2);
        assert!(result.is_ok());
        
        let entropy = result.unwrap();
        assert!(entropy >= 0.0);
        assert!(entropy < 10.0); // Reasonable upper bound
    }
    
    #[test]
    fn test_entropy_rate() {
        let data = Array1::from_vec((0..100).map(|x| x as f64).collect());
        let result = entropy_rate(data.view(), 1, 10);
        assert!(result.is_ok());
        
        let rate = result.unwrap();
        assert!(rate >= 0.0);
    }
    
    #[test]
    fn test_constant_series() {
        let data = Array1::from_vec(vec![1.0; 50]);
        
        let entropy = sample_entropy(data.view(), 2, 0.2).unwrap();
        assert_eq!(entropy, 0.5); // Default for constant series
        
        let rate = entropy_rate(data.view(), 1, 10).unwrap();
        assert_eq!(rate, 0.0); // Zero entropy rate for constant series
    }
    
    #[test]
    fn test_short_series() {
        let data = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        
        let entropy = sample_entropy(data.view(), 2, 0.2).unwrap();
        assert_eq!(entropy, 0.5); // Default for too short series
    }
}