//! Mathematical utilities with high-precision implementations

use ndarray::{Array1, Array2};
use crate::error::CdfaResult;
use crate::precision::kahan_simple::KahanAccumulator;

/// Calculate matrix norm using high-precision summation
pub fn matrix_norm(matrix: &Array2<f64>) -> f64 {
    let mut acc = KahanAccumulator::new();
    for &value in matrix.iter() {
        acc.add(value * value);
    }
    acc.sum().sqrt()
}

/// Calculate vector magnitude using high-precision summation
pub fn vector_magnitude(vector: &Array1<f64>) -> f64 {
    let mut acc = KahanAccumulator::new();
    for &value in vector.iter() {
        acc.add(value * value);
    }
    acc.sum().sqrt()
}

/// Normalize vector to unit length
pub fn normalize_vector(vector: &Array1<f64>) -> CdfaResult<Array1<f64>> {
    let magnitude = vector_magnitude(vector);
    if magnitude == 0.0 {
        return Ok(vector.clone());
    }
    Ok(vector / magnitude)
}

/// Calculate cosine similarity between two vectors using high-precision arithmetic
pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> CdfaResult<f64> {
    if a.len() != b.len() {
        return Err(crate::error::CdfaError::DimensionMismatch {
            expected: a.len(),
            actual: b.len(),
        });
    }
    
    // High-precision dot product calculation
    let mut dot_acc = KahanAccumulator::new();
    for (&ai, &bi) in a.iter().zip(b.iter()) {
        dot_acc.add(ai * bi);
    }
    let dot_product = dot_acc.sum();
    
    let norm_a = vector_magnitude(a);
    let norm_b = vector_magnitude(b);
    
    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    
    Ok(dot_product / (norm_a * norm_b))
}

/// High-precision financial calculation utilities
pub mod financial {
    use super::*;
    use crate::precision::stable::*;
    
    /// Calculate portfolio returns with high precision
    pub fn portfolio_returns(weights: &[f64], returns: &[f64]) -> CdfaResult<f64> {
        if weights.len() != returns.len() {
            return Err(crate::error::CdfaError::DimensionMismatch {
                expected: weights.len(),
                actual: returns.len(),
            });
        }
        
        crate::precision::kahan_simple::financial::portfolio_return(weights, returns)
    }
    
    /// Calculate Value at Risk (VaR) with numerical stability
    pub fn value_at_risk(returns: &[f64], confidence_level: f64) -> CdfaResult<f64> {
        if returns.is_empty() {
            return Err(crate::error::CdfaError::InvalidInput { message: "Empty returns array".to_string() });
        }
        
        if confidence_level <= 0.0 || confidence_level >= 1.0 {
            return Err(crate::error::CdfaError::InvalidInput { message: "Confidence level must be between 0 and 1".to_string() });
        }
        
        let mut sorted_returns = returns.to_vec();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = ((1.0 - confidence_level) * sorted_returns.len() as f64).floor() as usize;
        Ok(-sorted_returns[index.min(sorted_returns.len() - 1)])
    }
    
    /// Calculate Sharpe ratio with high precision
    pub fn sharpe_ratio(returns: &[f64], risk_free_rate: f64) -> CdfaResult<f64> {
        if returns.len() < 2 {
            return Err(crate::error::CdfaError::InvalidInput { message: "Need at least 2 returns for Sharpe ratio".to_string() });
        }
        
        // Calculate excess returns
        let excess_returns: Vec<f64> = returns.iter().map(|&r| r - risk_free_rate).collect();
        
        // Use Welford's algorithm for numerical stability
        let (mean, variance) = welford_variance(&excess_returns)?;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            Ok(0.0)
        } else {
            Ok(mean / std_dev)
        }
    }
    
    /// Calculate compound annual growth rate (CAGR) with high precision
    pub fn cagr(initial_value: f64, final_value: f64, years: f64) -> CdfaResult<f64> {
        if initial_value <= 0.0 {
            return Err(crate::error::CdfaError::InvalidInput { message: "Initial value must be positive".to_string() });
        }
        
        if final_value <= 0.0 {
            return Err(crate::error::CdfaError::InvalidInput { message: "Final value must be positive".to_string() });
        }
        
        if years <= 0.0 {
            return Err(crate::error::CdfaError::InvalidInput { message: "Years must be positive".to_string() });
        }
        
        Ok((final_value / initial_value).powf(1.0 / years) - 1.0)
    }
    
    /// Calculate maximum drawdown with numerical precision
    pub fn max_drawdown(values: &[f64]) -> CdfaResult<f64> {
        if values.is_empty() {
            return Err(crate::error::CdfaError::InvalidInput { message: "Empty values array".to_string() });
        }
        
        let mut peak = values[0];
        let mut max_dd = 0.0;
        
        for &value in values.iter().skip(1) {
            if value > peak {
                peak = value;
            } else {
                let drawdown = (peak - value) / peak;
                if drawdown > max_dd {
                    max_dd = drawdown;
                }
            }
        }
        
        Ok(max_dd)
    }
}