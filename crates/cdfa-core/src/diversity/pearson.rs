use ndarray::{Array2, ArrayView1, ArrayView2};

/// Pearson correlation coefficient
/// 
/// Measures linear correlation between two variables
/// Returns a value between -1 (perfect negative correlation) and 1 (perfect positive correlation)
pub fn pearson_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if x.len() != y.len() {
        return Err("Arrays must have the same length");
    }
    
    let n = x.len();
    if n < 2 {
        return Err("Arrays must have at least 2 elements");
    }
    
    let x_mean = x.mean().unwrap();
    let y_mean = y.mean().unwrap();
    
    let mut cov = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;
    
    for i in 0..n {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        
        cov += x_diff * y_diff;
        x_var += x_diff * x_diff;
        y_var += y_diff * y_diff;
    }
    
    // Handle edge cases
    if x_var < f64::EPSILON || y_var < f64::EPSILON {
        return Ok(0.0); // No correlation if one variable has no variance
    }
    
    let correlation = cov / (x_var.sqrt() * y_var.sqrt());
    
    // Ensure result is in [-1, 1] (numerical precision issues)
    Ok(correlation.max(-1.0).min(1.0))
}

/// Compute Pearson correlation matrix for multiple variables
pub fn pearson_correlation_matrix(data: &ArrayView2<f64>) -> Result<Array2<f64>, &'static str> {
    let (n_samples, n_features) = data.dim();
    
    if n_samples < 2 {
        return Err("Need at least 2 samples");
    }
    
    let mut corr_matrix = Array2::zeros((n_features, n_features));
    
    // Compute means for each feature
    let means: Vec<f64> = (0..n_features)
        .map(|j| data.column(j).mean().unwrap())
        .collect();
    
    // Compute standard deviations for each feature
    let mut stds = vec![0.0; n_features];
    for j in 0..n_features {
        let mean = means[j];
        let variance: f64 = data.column(j)
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / (n_samples - 1) as f64;
        stds[j] = variance.sqrt();
    }
    
    // Compute correlations
    for i in 0..n_features {
        corr_matrix[[i, i]] = 1.0; // Diagonal is always 1
        
        for j in (i + 1)..n_features {
            if stds[i] < f64::EPSILON || stds[j] < f64::EPSILON {
                // Zero variance, no correlation
                corr_matrix[[i, j]] = 0.0;
                corr_matrix[[j, i]] = 0.0;
            } else {
                // Compute covariance
                let mut cov = 0.0;
                for k in 0..n_samples {
                    cov += (data[[k, i]] - means[i]) * (data[[k, j]] - means[j]);
                }
                cov /= (n_samples - 1) as f64;
                
                let corr = cov / (stds[i] * stds[j]);
                let corr_clamped = corr.max(-1.0).min(1.0);
                
                corr_matrix[[i, j]] = corr_clamped;
                corr_matrix[[j, i]] = corr_clamped; // Symmetric
            }
        }
    }
    
    Ok(corr_matrix)
}

/// Fast Pearson correlation using SIMD-friendly operations
pub fn pearson_correlation_fast(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if x.len() != y.len() {
        return Err("Arrays must have the same length");
    }
    
    let n = x.len();
    if n < 2 {
        return Err("Arrays must have at least 2 elements");
    }
    
    let n_f64 = n as f64;
    
    // Calculate sums in a single pass
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    
    // This loop is SIMD-friendly
    for i in 0..n {
        let xi = x[i];
        let yi = y[i];
        
        sum_x += xi;
        sum_y += yi;
        sum_xx += xi * xi;
        sum_yy += yi * yi;
        sum_xy += xi * yi;
    }
    
    // Calculate correlation using the computational formula
    let numerator = n_f64 * sum_xy - sum_x * sum_y;
    let denominator_x = n_f64 * sum_xx - sum_x * sum_x;
    let denominator_y = n_f64 * sum_yy - sum_y * sum_y;
    
    if denominator_x <= 0.0 || denominator_y <= 0.0 {
        return Ok(0.0); // No variance
    }
    
    let correlation = numerator / (denominator_x.sqrt() * denominator_y.sqrt());
    
    // Clamp to [-1, 1]
    Ok(correlation.max(-1.0).min(1.0))
}

/// Compute partial correlation between x and y, controlling for z
pub fn partial_correlation(
    x: &ArrayView1<f64>, 
    y: &ArrayView1<f64>, 
    z: &ArrayView1<f64>
) -> Result<f64, &'static str> {
    if x.len() != y.len() || x.len() != z.len() {
        return Err("All arrays must have the same length");
    }
    
    if x.len() < 3 {
        return Err("Need at least 3 samples for partial correlation");
    }
    
    // Calculate pairwise correlations
    let r_xy = pearson_correlation(x, y)?;
    let r_xz = pearson_correlation(x, z)?;
    let r_yz = pearson_correlation(y, z)?;
    
    // Calculate partial correlation
    let denominator = ((1.0 - r_xz * r_xz) * (1.0 - r_yz * r_yz)).sqrt();
    
    if denominator < f64::EPSILON {
        // Perfect correlation with control variable
        return Ok(0.0);
    }
    
    let partial_corr = (r_xy - r_xz * r_yz) / denominator;
    
    // Clamp to [-1, 1]
    Ok(partial_corr.max(-1.0).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    
    #[test]
    fn test_pearson_perfect_positive() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let corr = pearson_correlation(&x.view(), &y.view()).unwrap();
        assert!((corr - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_pearson_perfect_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];
        let corr = pearson_correlation(&x.view(), &y.view()).unwrap();
        assert!((corr + 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_pearson_no_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 5.0, 3.0, 4.0, 1.0];
        let corr = pearson_correlation(&x.view(), &y.view()).unwrap();
        assert!(corr.abs() < 0.3); // Weak correlation
    }
    
    #[test]
    fn test_pearson_fast_matches_standard() {
        let x = array![1.2, 2.3, 3.4, 4.5, 5.6, 6.7, 7.8, 8.9];
        let y = array![2.1, 4.2, 5.3, 8.4, 10.5, 11.6, 14.7, 16.8];
        
        let corr_standard = pearson_correlation(&x.view(), &y.view()).unwrap();
        let corr_fast = pearson_correlation_fast(&x.view(), &y.view()).unwrap();
        
        assert!((corr_standard - corr_fast).abs() < 1e-10);
    }
    
    #[test]
    fn test_correlation_matrix() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0]
        ];
        
        let corr_matrix = pearson_correlation_matrix(&data.view()).unwrap();
        
        // Check diagonal is all 1s
        for i in 0..3 {
            assert!((corr_matrix[[i, i]] - 1.0).abs() < 1e-10);
        }
        
        // Check symmetry
        for i in 0..3 {
            for j in 0..3 {
                assert!((corr_matrix[[i, j]] - corr_matrix[[j, i]]).abs() < 1e-10);
            }
        }
        
        // All columns are perfectly correlated in this example
        assert!((corr_matrix[[0, 1]] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[[0, 2]] - 1.0).abs() < 1e-10);
        assert!((corr_matrix[[1, 2]] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_partial_correlation() {
        // Example where z partially explains the correlation between x and y
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let z = array![1.0, 1.5, 2.0, 2.5, 3.0];
        
        let partial_corr = partial_correlation(&x.view(), &y.view(), &z.view()).unwrap();
        
        // Partial correlation should be high but not perfect
        assert!(partial_corr > 0.9 && partial_corr < 1.0);
    }
}