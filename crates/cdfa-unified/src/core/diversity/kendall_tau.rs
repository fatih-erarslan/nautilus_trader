//! Kendall's tau rank correlation implementation

use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::ArrayView1;
use num_traits::{Float as FloatTrait, Zero, One};

/// Calculate Kendall's tau correlation coefficient
pub fn kendall_tau(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(CdfaError::invalid_input("Arrays must have the same length"));
    }
    
    if x.is_empty() {
        return Err(CdfaError::invalid_input("Arrays cannot be empty"));
    }
    
    let n = x.len();
    let mut concordant = 0;
    let mut discordant = 0;
    
    // Count concordant and discordant pairs
    for i in 0..n {
        for j in i + 1..n {
            let x_diff = x[i] - x[j];
            let y_diff = y[i] - y[j];
            
            if x_diff * y_diff > Float::zero() {
                concordant += 1;
            } else if x_diff * y_diff < Float::zero() {
                discordant += 1;
            }
            // If x_diff * y_diff == 0, it's a tie and we don't count it
        }
    }
    
    let total_pairs = n * (n - 1) / 2;
    let tau = (concordant as f64 - discordant as f64) / total_pairs as f64;
    
    Ok(tau)
}

/// Fast Kendall's tau implementation (same as regular for now)
pub fn kendall_tau_fast(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    kendall_tau(x, y)
}

/// Calculate Kendall's distance (1 - tau)
pub fn kendall_distance(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    let tau = kendall_tau(x, y)?;
    Ok(Float::one() - tau)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_kendall_tau_perfect_positive() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let tau = kendall_tau(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(tau, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_kendall_tau_perfect_negative() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 4.0, 3.0, 2.0, 1.0];
        
        let tau = kendall_tau(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(tau, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_kendall_distance() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        
        let distance = kendall_distance(&x.view(), &y.view()).unwrap();
        assert_abs_diff_eq!(distance, 0.0, epsilon = 1e-10);
    }
}