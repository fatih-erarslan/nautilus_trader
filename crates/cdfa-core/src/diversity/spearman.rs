use ndarray::{Array1, ArrayView1};

/// Spearman rank correlation coefficient
/// 
/// Measures monotonic relationships between two variables
/// Returns a value between -1 (perfect negative correlation) and 1 (perfect positive correlation)
pub fn spearman_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if x.len() != y.len() {
        return Err("Arrays must have the same length");
    }
    
    let n = x.len();
    if n < 2 {
        return Err("Arrays must have at least 2 elements");
    }
    
    // Convert to ranks
    let x_ranks = compute_ranks(x);
    let y_ranks = compute_ranks(y);
    
    // Calculate Pearson correlation on ranks
    pearson_correlation(&x_ranks.view(), &y_ranks.view())
}

/// Convert values to ranks, handling ties with average rank
fn compute_ranks(data: &ArrayView1<f64>) -> Array1<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    // Sort by value
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut ranks = Array1::zeros(n);
    let mut i = 0;
    
    while i < n {
        let mut j = i;
        // Find all elements with the same value (ties)
        while j < n && (indexed[j].1 - indexed[i].1).abs() < f64::EPSILON {
            j += 1;
        }
        
        // Assign average rank to all tied values
        let avg_rank = (i + j) as f64 / 2.0 + 0.5; // +0.5 for 1-based ranking
        
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        
        i = j;
    }
    
    ranks
}

/// Pearson correlation coefficient (used internally for Spearman)
fn pearson_correlation(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if x.len() != y.len() || x.is_empty() {
        return Err("Invalid input arrays");
    }
    
    let n = x.len() as f64;
    
    // Calculate means
    let x_mean = x.sum() / n;
    let y_mean = y.sum() / n;
    
    // Calculate covariance and standard deviations
    let mut cov = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;
    
    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        
        cov += x_diff * y_diff;
        x_var += x_diff * x_diff;
        y_var += y_diff * y_diff;
    }
    
    let x_std = x_var.sqrt();
    let y_std = y_var.sqrt();
    
    if x_std < f64::EPSILON || y_std < f64::EPSILON {
        // One or both arrays have zero variance
        return Ok(0.0);
    }
    
    Ok(cov / (x_std * y_std))
}

/// Fast Spearman correlation using the formula: rho = 1 - 6*sum(d^2)/(n*(n^2-1))
/// This works only when there are no ties
pub fn spearman_correlation_fast(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if x.len() != y.len() {
        return Err("Arrays must have the same length");
    }
    
    let n = x.len();
    if n < 2 {
        return Err("Arrays must have at least 2 elements");
    }
    
    // Check for ties
    if has_ties(x) || has_ties(y) {
        // Fall back to standard method
        return spearman_correlation(x, y);
    }
    
    // Convert to ranks
    let x_ranks = compute_ranks_no_ties(x);
    let y_ranks = compute_ranks_no_ties(y);
    
    // Calculate sum of squared rank differences
    let sum_d_squared: f64 = x_ranks
        .iter()
        .zip(y_ranks.iter())
        .map(|(xi, yi)| (xi - yi).powi(2))
        .sum();
    
    let n_f64 = n as f64;
    let rho = 1.0 - (6.0 * sum_d_squared) / (n_f64 * (n_f64 * n_f64 - 1.0));
    
    Ok(rho)
}

/// Check if array has tied values
fn has_ties(data: &ArrayView1<f64>) -> bool {
    let mut sorted: Vec<f64> = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    for i in 1..sorted.len() {
        if (sorted[i] - sorted[i-1]).abs() < f64::EPSILON {
            return true;
        }
    }
    false
}

/// Compute ranks when there are no ties (faster)
fn compute_ranks_no_ties(data: &ArrayView1<f64>) -> Array1<f64> {
    let n = data.len();
    let mut indexed: Vec<(usize, f64)> = data
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut ranks = Array1::zeros(n);
    for (rank, (idx, _)) in indexed.iter().enumerate() {
        ranks[*idx] = (rank + 1) as f64; // 1-based ranking
    }
    
    ranks
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_spearman_perfect_positive() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let rho = spearman_correlation(&x.view(), &y.view()).unwrap();
        assert!((rho - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_spearman_perfect_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 8.0, 6.0, 4.0, 2.0];
        let rho = spearman_correlation(&x.view(), &y.view()).unwrap();
        assert!((rho + 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_spearman_with_ties() {
        let x = array![1.0, 2.0, 2.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 3.0, 5.0];
        let rho = spearman_correlation(&x.view(), &y.view()).unwrap();
        assert!(rho > 0.8 && rho <= 1.0);
    }
    
    #[test]
    fn test_spearman_fast_no_ties() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        
        let rho_standard = spearman_correlation(&x.view(), &y.view()).unwrap();
        let rho_fast = spearman_correlation_fast(&x.view(), &y.view()).unwrap();
        
        assert!((rho_standard - rho_fast).abs() < 1e-10);
    }
    
    #[test]
    fn test_rank_computation() {
        let data = array![3.0, 1.0, 4.0, 1.0, 5.0];
        let ranks = compute_ranks(&data.view());
        let expected = array![3.0, 1.5, 4.0, 1.5, 5.0]; // Tied values get average rank
        
        for i in 0..ranks.len() {
            assert!((ranks[i] - expected[i]).abs() < 1e-10);
        }
    }
}