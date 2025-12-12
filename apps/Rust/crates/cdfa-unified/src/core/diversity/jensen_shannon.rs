//! Jensen-Shannon divergence implementation

use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::ArrayView1;
use num_traits::{Float as FloatTrait, Zero};
// LN_2 available via std if needed

/// Calculate Jensen-Shannon divergence between two probability distributions
pub fn jensen_shannon_divergence(p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Result<Float> {
    if p.len() != q.len() {
        return Err(CdfaError::invalid_input("Distributions must have the same length"));
    }
    
    if p.is_empty() {
        return Err(CdfaError::invalid_input("Distributions cannot be empty"));
    }
    
    // Validate that inputs are probability distributions
    let p_sum: Float = p.sum();
    let q_sum: Float = q.sum();
    
    let epsilon = 1e-10;
    if (p_sum - 1.0).abs() > epsilon || (q_sum - 1.0).abs() > epsilon {
        return Err(CdfaError::invalid_input("Inputs must be probability distributions (sum to 1)"));
    }
    
    // Check for negative values
    if p.iter().any(|&x| x < 0.0) || q.iter().any(|&x| x < 0.0) {
        return Err(CdfaError::invalid_input("Probability distributions cannot have negative values"));
    }
    
    // Calculate M = (P + Q) / 2
    let m: Vec<Float> = p.iter().zip(q.iter()).map(|(&pi, &qi)| (pi + qi) / 2.0).collect();
    
    // Calculate KL divergences
    let kl_pm = kl_divergence(p, &m)?;
    let kl_qm = kl_divergence(q, &m)?;
    
    // Jensen-Shannon divergence = (KL(P||M) + KL(Q||M)) / 2
    let js_div = (kl_pm + kl_qm) / 2.0;
    
    Ok(js_div)
}

/// Calculate Jensen-Shannon distance (square root of divergence)
pub fn jensen_shannon_distance(p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Result<Float> {
    let divergence = jensen_shannon_divergence(p, q)?;
    Ok(divergence.sqrt())
}

/// Calculate empirical Jensen-Shannon divergence from non-normalized data
pub fn jensen_shannon_divergence_empirical(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    // Normalize to probability distributions
    let x_sum = x.sum();
    let y_sum = y.sum();
    
    if x_sum <= Float::zero() || y_sum <= Float::zero() {
        return Err(CdfaError::invalid_input("Data sums must be positive for normalization"));
    }
    
    let x_norm: Vec<Float> = x.iter().map(|&v| v / x_sum).collect();
    let y_norm: Vec<Float> = y.iter().map(|&v| v / y_sum).collect();
    
    let x_view = ndarray::ArrayView1::from(&x_norm);
    let y_view = ndarray::ArrayView1::from(&y_norm);
    
    jensen_shannon_divergence(&x_view, &y_view)
}

/// Calculate Kullback-Leibler divergence
fn kl_divergence(p: &ArrayView1<Float>, q: &[Float]) -> Result<Float> {
    if p.len() != q.len() {
        return Err(CdfaError::invalid_input("Distributions must have the same length"));
    }
    
    let mut kl = 0.0;
    let epsilon = 1e-10;
    
    for (i, &pi) in p.iter().enumerate() {
        if pi > 0.0 {
            let qi = q[i];
            if qi <= 0.0 {
                // If qi is 0 but pi > 0, divergence is infinite
                return Ok(Float::INFINITY);
            }
            kl += pi * (pi / qi).ln();
        }
    }
    
    Ok(kl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_jensen_shannon_divergence_identical() {
        let p = array![0.5, 0.3, 0.2];
        let q = array![0.5, 0.3, 0.2];
        
        let divergence = jensen_shannon_divergence(&p.view(), &q.view()).unwrap();
        assert_abs_diff_eq!(divergence, 0.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jensen_shannon_divergence_uniform() {
        let p = array![1.0, 0.0];
        let q = array![0.0, 1.0];
        
        let divergence = jensen_shannon_divergence(&p.view(), &q.view()).unwrap();
        
        // For completely opposite distributions, JS divergence should be ln(2)
        let expected = std::f64::consts::LN_2;
        assert_abs_diff_eq!(divergence, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jensen_shannon_distance() {
        let p = array![1.0, 0.0];
        let q = array![0.0, 1.0];
        
        let distance = jensen_shannon_distance(&p.view(), &q.view()).unwrap();
        
        // Distance should be sqrt(ln(2))
        let expected = std::f64::consts::LN_2.sqrt();
        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_jensen_shannon_divergence_empirical() {
        let x = array![2.0, 1.0]; // Will be normalized to [2/3, 1/3]
        let y = array![1.0, 2.0]; // Will be normalized to [1/3, 2/3]
        
        let divergence = jensen_shannon_divergence_empirical(&x.view(), &y.view()).unwrap();
        assert!(divergence > 0.0);
    }
    
    #[test]
    fn test_invalid_input() {
        let p = array![0.6, 0.3, 0.2]; // Doesn't sum to 1
        let q = array![0.5, 0.3, 0.2];
        
        assert!(jensen_shannon_divergence(&p.view(), &q.view()).is_err());
    }
}