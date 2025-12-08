use ndarray::{Array1, ArrayView1};

/// Jensen-Shannon divergence
/// 
/// A symmetric and smoothed version of the Kullback-Leibler divergence
/// Returns a value between 0 (identical distributions) and 1 (maximally different)
pub fn jensen_shannon_divergence(p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if p.len() != q.len() {
        return Err("Distributions must have the same length");
    }
    
    if p.is_empty() {
        return Err("Distributions cannot be empty");
    }
    
    // Validate that inputs are valid probability distributions
    let p_sum = p.sum();
    let q_sum = q.sum();
    
    if (p_sum - 1.0).abs() > 1e-6 || (q_sum - 1.0).abs() > 1e-6 {
        return Err("Input arrays must be valid probability distributions (sum to 1)");
    }
    
    // Check for non-negative values
    if p.iter().any(|&x| x < 0.0) || q.iter().any(|&x| x < 0.0) {
        return Err("Probability values must be non-negative");
    }
    
    // Calculate average distribution
    let m = (p + q) / 2.0;
    
    // Calculate KL divergences
    let kl_pm = kullback_leibler_divergence(p, &m.view())?;
    let kl_qm = kullback_leibler_divergence(q, &m.view())?;
    
    // Jensen-Shannon divergence
    let js_div = (kl_pm + kl_qm) / 2.0;
    
    Ok(js_div)
}

/// Jensen-Shannon distance (square root of divergence)
/// 
/// This is a proper metric that satisfies the triangle inequality
pub fn jensen_shannon_distance(p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> Result<f64, &'static str> {
    let js_div = jensen_shannon_divergence(p, q)?;
    Ok(js_div.sqrt())
}

/// Kullback-Leibler divergence (used internally)
fn kullback_leibler_divergence(p: &ArrayView1<f64>, q: &ArrayView1<f64>) -> Result<f64, &'static str> {
    if p.len() != q.len() {
        return Err("Distributions must have the same length");
    }
    
    let mut kl_div = 0.0;
    
    for i in 0..p.len() {
        if p[i] > 0.0 {
            if q[i] == 0.0 {
                // KL divergence is infinite when p[i] > 0 but q[i] = 0
                return Ok(f64::INFINITY);
            }
            kl_div += p[i] * (p[i] / q[i]).ln();
        }
    }
    
    Ok(kl_div)
}

/// Generalized Jensen-Shannon divergence for multiple distributions
pub fn generalized_jensen_shannon_divergence(
    distributions: &[ArrayView1<f64>], 
    weights: Option<&Array1<f64>>
) -> Result<f64, &'static str> {
    if distributions.is_empty() {
        return Err("Need at least one distribution");
    }
    
    let n_dists = distributions.len();
    let dist_len = distributions[0].len();
    
    // Check all distributions have the same length
    for dist in distributions.iter() {
        if dist.len() != dist_len {
            return Err("All distributions must have the same length");
        }
    }
    
    // Handle weights - create uniform weights if none provided
    let uniform_weights = Array1::from_elem(n_dists, 1.0 / n_dists as f64);
    let w = match weights {
        Some(w) => w.view(),
        None => uniform_weights.view(),
    };
    
    if w.len() != n_dists {
        return Err("Number of weights must match number of distributions");
    }
    
    // Validate weights
    let weight_sum = w.sum();
    if (weight_sum - 1.0).abs() > 1e-6 {
        return Err("Weights must sum to 1");
    }
    
    if w.iter().any(|&x| x < 0.0) {
        return Err("Weights must be non-negative");
    }
    
    // Calculate weighted average distribution
    let mut m = Array1::zeros(dist_len);
    for (i, dist) in distributions.iter().enumerate() {
        m = m + w[i] * dist;
    }
    
    // Calculate weighted sum of KL divergences
    let mut js_div = 0.0;
    for (i, dist) in distributions.iter().enumerate() {
        let kl = kullback_leibler_divergence(dist, &m.view())?;
        js_div += w[i] * kl;
    }
    
    Ok(js_div)
}

/// Convert raw data to probability distribution using histogram
pub fn to_probability_distribution(data: &ArrayView1<f64>, n_bins: usize) -> Result<Array1<f64>, &'static str> {
    if data.is_empty() {
        return Err("Data cannot be empty");
    }
    
    if n_bins == 0 {
        return Err("Number of bins must be positive");
    }
    
    let min_val = data.fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = data.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    if (max_val - min_val).abs() < f64::EPSILON {
        // All values are the same
        let mut hist = Array1::zeros(n_bins);
        hist[0] = 1.0;
        return Ok(hist);
    }
    
    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut hist = Array1::zeros(n_bins);
    
    // Count values in each bin
    for &value in data.iter() {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1); // Handle edge case where value == max_val
        hist[bin_idx] += 1.0;
    }
    
    // Normalize to probability distribution
    let total = hist.sum();
    if total > 0.0 {
        hist /= total;
    }
    
    Ok(hist)
}

/// Calculate JS divergence between empirical distributions
pub fn jensen_shannon_divergence_empirical(
    x: &ArrayView1<f64>, 
    y: &ArrayView1<f64>, 
    n_bins: usize
) -> Result<f64, &'static str> {
    // Convert to probability distributions
    let p = to_probability_distribution(x, n_bins)?;
    let q = to_probability_distribution(y, n_bins)?;
    
    // Calculate JS divergence
    jensen_shannon_divergence(&p.view(), &q.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_js_divergence_identical() {
        let p = array![0.25, 0.25, 0.25, 0.25];
        let q = array![0.25, 0.25, 0.25, 0.25];
        let js_div = jensen_shannon_divergence(&p.view(), &q.view()).unwrap();
        assert!(js_div.abs() < 1e-10);
    }
    
    #[test]
    fn test_js_divergence_different() {
        let p = array![0.5, 0.5, 0.0, 0.0];
        let q = array![0.0, 0.0, 0.5, 0.5];
        let js_div = jensen_shannon_divergence(&p.view(), &q.view()).unwrap();
        assert!(js_div > 0.5); // Should be significant
    }
    
    #[test]
    fn test_js_divergence_symmetric() {
        let p = array![0.3, 0.3, 0.2, 0.2];
        let q = array![0.2, 0.4, 0.3, 0.1];
        
        let js_pq = jensen_shannon_divergence(&p.view(), &q.view()).unwrap();
        let js_qp = jensen_shannon_divergence(&q.view(), &p.view()).unwrap();
        
        assert!((js_pq - js_qp).abs() < 1e-10);
    }
    
    #[test]
    fn test_js_distance_metric_properties() {
        let p = array![0.25, 0.25, 0.25, 0.25];
        let q = array![0.3, 0.3, 0.2, 0.2];
        let r = array![0.1, 0.2, 0.3, 0.4];
        
        // Identity
        let d_pp = jensen_shannon_distance(&p.view(), &p.view()).unwrap();
        assert!(d_pp.abs() < 1e-10);
        
        // Symmetry
        let d_pq = jensen_shannon_distance(&p.view(), &q.view()).unwrap();
        let d_qp = jensen_shannon_distance(&q.view(), &p.view()).unwrap();
        assert!((d_pq - d_qp).abs() < 1e-10);
        
        // Triangle inequality (approximately, JS distance satisfies this)
        let d_pr = jensen_shannon_distance(&p.view(), &r.view()).unwrap();
        let d_qr = jensen_shannon_distance(&q.view(), &r.view()).unwrap();
        assert!(d_pr <= d_pq + d_qr + 1e-10);
    }
    
    #[test]
    fn test_generalized_js_divergence() {
        let p1 = array![0.5, 0.5, 0.0];
        let p2 = array![0.0, 0.5, 0.5];
        let p3 = array![0.33, 0.34, 0.33];
        
        let distributions = vec![p1.view(), p2.view(), p3.view()];
        let weights = array![0.3, 0.3, 0.4];
        
        let js_div = generalized_jensen_shannon_divergence(&distributions, Some(&weights)).unwrap();
        assert!(js_div > 0.0);
        
        // Test with equal weights
        let js_div_equal = generalized_jensen_shannon_divergence(&distributions, None).unwrap();
        assert!(js_div_equal > 0.0);
    }
    
    #[test]
    fn test_empirical_js_divergence() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0];
        let y = array![3.0, 4.0, 5.0, 6.0, 7.0, 4.0, 5.0, 6.0];
        
        let js_div = jensen_shannon_divergence_empirical(&x.view(), &y.view(), 5).unwrap();
        assert!(js_div > 0.1); // Should show some divergence
    }
    
    #[test]
    fn test_to_probability_distribution() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let prob_dist = to_probability_distribution(&data.view(), 5).unwrap();
        
        // Should sum to 1
        assert!((prob_dist.sum() - 1.0).abs() < 1e-10);
        
        // Should have one value per bin
        assert_eq!(prob_dist, array![0.2, 0.2, 0.2, 0.2, 0.2]);
    }
}