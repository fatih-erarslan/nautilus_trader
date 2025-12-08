//! Benefit ratio analysis module
//!
//! This module calculates the benefit ratio component of antifragility,
//! which measures whether performance improvements outweigh volatility costs.

use crate::{AntifragilityError, AntifragilityResult};
use ndarray::prelude::*;
use num_traits::Float;

/// Calculate benefit ratio between performance and volatility changes
pub fn calculate_benefit_ratio(
    perf_roc_smoothed: &Array1<f64>,
    vol_roc_smoothed: &Array1<f64>,
) -> AntifragilityResult<f64> {
    let n = perf_roc_smoothed.len();
    
    if n != vol_roc_smoothed.len() {
        return Err(AntifragilityError::InvalidParameters {
            message: format!("Array lengths must match: {} vs {}", n, vol_roc_smoothed.len()),
        });
    }
    
    if n == 0 {
        return Ok(0.5);
    }
    
    let mut benefit_ratios = Vec::new();
    
    for i in 0..n {
        let perf_change = perf_roc_smoothed[i];
        let vol_change = vol_roc_smoothed[i];
        
        // Calculate benefit ratio with safe division
        let ratio = if vol_change.abs() > 1e-6 {
            perf_change / vol_change
        } else {
            0.0
        };
        
        if ratio.is_finite() {
            benefit_ratios.push(ratio);
        }
    }
    
    if benefit_ratios.is_empty() {
        return Ok(0.5);
    }
    
    let mean_ratio = benefit_ratios.iter().sum::<f64>() / benefit_ratios.len() as f64;
    
    // Transform to 0-1 scale using tanh
    Ok((mean_ratio.tanh() + 1.0) / 2.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_benefit_ratio() {
        let perf_roc = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
        let vol_roc = Array1::from_vec(vec![0.1, 0.1, 0.1, 0.1, 0.1]);
        
        let result = calculate_benefit_ratio(&perf_roc, &vol_roc);
        assert!(result.is_ok());
        
        let ratio = result.unwrap();
        assert!(ratio >= 0.0 && ratio <= 1.0);
        assert!(ratio > 0.5); // Should be positive since perf > vol
    }
}