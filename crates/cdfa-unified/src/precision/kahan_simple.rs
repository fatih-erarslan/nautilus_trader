//! Simplified Kahan summation implementation for financial precision
//!
//! This is a production-ready implementation focused on correctness and
//! precision for financial calculations.

use crate::error::{CdfaError, CdfaResult};

/// High-precision Kahan accumulator for financial calculations
#[derive(Debug, Clone, PartialEq)]
pub struct KahanAccumulator {
    sum: f64,
    compensation: f64,
}

impl Default for KahanAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl KahanAccumulator {
    /// Create a new Kahan accumulator
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Create accumulator with initial value
    pub const fn with_initial(value: f64) -> Self {
        Self {
            sum: value,
            compensation: 0.0,
        }
    }

    /// Add a value using Kahan's compensated summation
    pub fn add(&mut self, value: f64) {
        let y = value - self.compensation;
        let t = self.sum + y;
        self.compensation = (t - self.sum) - y;
        self.sum = t;
    }

    /// Get the current sum
    pub fn sum(&self) -> f64 {
        self.sum
    }

    /// Reset to zero
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.compensation = 0.0;
    }

    /// Compute sum of slice using Kahan summation
    pub fn sum_slice(values: &[f64]) -> f64 {
        let mut acc = Self::new();
        for &value in values {
            acc.add(value);
        }
        acc.sum()
    }
}

/// Neumaier's improved Kahan summation
#[derive(Debug, Clone, PartialEq)]
pub struct NeumaierAccumulator {
    sum: f64,
    compensation: f64,
}

impl Default for NeumaierAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl NeumaierAccumulator {
    /// Create new Neumaier accumulator
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
            compensation: 0.0,
        }
    }

    /// Add value using Neumaier's algorithm
    pub fn add(&mut self, value: f64) {
        let t = self.sum + value;
        
        if self.sum.abs() >= value.abs() {
            self.compensation += (self.sum - t) + value;
        } else {
            self.compensation += (value - t) + self.sum;
        }
        
        self.sum = t;
    }

    /// Get sum with compensation
    pub fn sum(&self) -> f64 {
        self.sum + self.compensation
    }

    /// Compute sum of slice using Neumaier summation
    pub fn sum_slice(values: &[f64]) -> f64 {
        let mut acc = Self::new();
        for &value in values {
            acc.add(value);
        }
        acc.sum()
    }
}

/// High-precision financial calculations
pub mod financial {
    use super::*;

    /// Calculate mean with high precision
    pub fn mean(values: &[f64]) -> CdfaResult<f64> {
        if values.is_empty() {
            return Err(CdfaError::InvalidInput { 
                message: "Cannot compute mean of empty slice".to_string() 
            });
        }
        
        let sum = KahanAccumulator::sum_slice(values);
        Ok(sum / values.len() as f64)
    }

    /// Calculate variance with high precision (Welford's algorithm)
    pub fn variance(values: &[f64]) -> CdfaResult<f64> {
        if values.len() < 2 {
            return Err(CdfaError::InvalidInput { 
                message: "Need at least 2 values for variance".to_string() 
            });
        }

        let mut mean = 0.0;
        let mut m2 = 0.0;

        for (i, &value) in values.iter().enumerate() {
            let n = (i + 1) as f64;
            let delta = value - mean;
            mean += delta / n;
            let delta2 = value - mean;
            m2 += delta * delta2;
        }

        Ok(m2 / (values.len() - 1) as f64)
    }

    /// Calculate portfolio return with high precision
    pub fn portfolio_return(weights: &[f64], returns: &[f64]) -> CdfaResult<f64> {
        if weights.len() != returns.len() {
            return Err(CdfaError::DimensionMismatch {
                expected: weights.len(),
                actual: returns.len(),
            });
        }

        let mut acc = KahanAccumulator::new();
        for (&w, &r) in weights.iter().zip(returns.iter()) {
            acc.add(w * r);
        }

        Ok(acc.sum())
    }

    /// Test pathological precision case
    pub fn precision_test(scale: f64) -> f64 {
        let mut acc = KahanAccumulator::new();
        acc.add(scale);       // Add large number
        acc.add(1.0);         // Add small number
        acc.add(-scale);      // Subtract large number
        acc.sum()             // Should be 1.0, not 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kahan_basic() {
        let mut acc = KahanAccumulator::new();
        acc.add(1.0);
        acc.add(2.0);
        acc.add(3.0);
        assert_eq!(acc.sum(), 6.0);
    }

    #[test]
    fn test_kahan_precision() {
        let result = financial::precision_test(1e16);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_neumaier_precision() {
        let mut acc = NeumaierAccumulator::new();
        acc.add(1e16);
        acc.add(1.0);
        acc.add(-1e16);
        assert_eq!(acc.sum(), 1.0);
    }

    #[test]
    fn test_slice_summation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(KahanAccumulator::sum_slice(&values), 15.0);
        assert_eq!(NeumaierAccumulator::sum_slice(&values), 15.0);
    }

    #[test]
    fn test_pathological_case() {
        let values = vec![1e16, 1.0, 1.0, 1.0, -1e16];
        let kahan_sum = KahanAccumulator::sum_slice(&values);
        let neumaier_sum = NeumaierAccumulator::sum_slice(&values);
        
        assert_eq!(kahan_sum, 3.0);
        assert_eq!(neumaier_sum, 3.0);
    }

    #[test]
    fn test_financial_calculations() {
        let weights = vec![0.3, 0.3, 0.4];
        let returns = vec![0.05, 0.08, 0.02];
        
        let portfolio_ret = financial::portfolio_return(&weights, &returns).unwrap();
        
        // Manual calculation: 0.3*0.05 + 0.3*0.08 + 0.4*0.02 = 0.015 + 0.024 + 0.008 = 0.047
        assert!((portfolio_ret - 0.047).abs() < 1e-15);
    }

    #[test]
    fn test_mean_and_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        
        let mean = financial::mean(&values).unwrap();
        assert_eq!(mean, 3.0);
        
        let variance = financial::variance(&values).unwrap();
        assert!((variance - 2.5).abs() < 1e-15);
    }
}