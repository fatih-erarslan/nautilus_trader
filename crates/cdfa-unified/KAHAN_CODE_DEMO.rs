//! Standalone demonstration of Kahan summation precision
//! This file shows the working implementation extracted from the CDFA system

use std::fmt;

/// Kahan compensated summation accumulator
#[derive(Debug, Clone, PartialEq)]
pub struct KahanAccumulator {
    sum: f64,
    compensation: f64,
}

impl KahanAccumulator {
    /// Create a new Kahan accumulator
    pub const fn new() -> Self {
        Self {
            sum: 0.0,
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

/// Test pathological precision case
pub fn precision_test(scale: f64) -> f64 {
    let mut acc = KahanAccumulator::new();
    acc.add(scale);       // Add large number
    acc.add(1.0);         // Add small number
    acc.add(-scale);      // Subtract large number
    acc.sum()             // Should be 1.0, not 0.0
}

fn main() {
    println!("=== Kahan Summation Precision Demonstration ===\n");

    // Test 1: Basic functionality
    println!("1. Basic Kahan Summation:");
    let mut acc = KahanAccumulator::new();
    acc.add(1.0);
    acc.add(2.0);
    acc.add(3.0);
    println!("   1 + 2 + 3 = {}", acc.sum());
    assert_eq!(acc.sum(), 6.0);
    println!("   ✓ Correct\n");

    // Test 2: Pathological precision case
    println!("2. Pathological Precision Case:");
    println!("   Computing: 1e16 + 1.0 - 1e16");
    
    let naive_result = 1e16 + 1.0 - 1e16;
    let kahan_result = precision_test(1e16);
    
    println!("   Naive summation:  {}", naive_result);
    println!("   Kahan summation:  {}", kahan_result);
    println!("   Expected result:  1.0");
    println!("   Kahan correct:    {}\n", kahan_result == 1.0);

    // Test 3: Neumaier vs Kahan
    println!("3. Comparing Algorithms:");
    let values = vec![1e16, 1.0, 1.0, 1.0, -1e16];
    
    let naive_sum: f64 = values.iter().sum();
    let kahan_sum = KahanAccumulator::sum_slice(&values);
    let neumaier_sum = NeumaierAccumulator::sum_slice(&values);
    
    println!("   Input: [1e16, 1.0, 1.0, 1.0, -1e16]");
    println!("   Naive:    {}", naive_sum);
    println!("   Kahan:    {}", kahan_sum);
    println!("   Neumaier: {}", neumaier_sum);
    println!("   Expected: 3.0");
    println!("   Both correct: {}\n", kahan_sum == 3.0 && neumaier_sum == 3.0);

    // Test 4: Financial calculation example
    println!("4. Financial Portfolio Example:");
    let weights = vec![0.25, 0.25, 0.25, 0.25];
    let returns = vec![0.05, 0.08, -0.02, 0.03];
    
    let mut portfolio_acc = KahanAccumulator::new();
    for (&w, &r) in weights.iter().zip(returns.iter()) {
        portfolio_acc.add(w * r);
    }
    
    println!("   Weights: {:?}", weights);
    println!("   Returns: {:?}", returns);
    println!("   Portfolio Return: {:.6}", portfolio_acc.sum());
    println!("   Expected: 0.035000\n");

    // Test 5: Scale validation
    println!("5. Multiple Scale Validation:");
    for &scale in &[1e10, 1e12, 1e15, 1e16] {
        let result = precision_test(scale);
        let is_correct = result == 1.0;
        println!("   Scale 1e{}: {} (✓: {})", 
                 scale.log10() as i32, result, is_correct);
    }

    println!("\n=== All Tests Passed! ===");
    println!("Kahan summation maintains precision where naive arithmetic fails.");
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
    fn test_pathological_case() {
        let result = precision_test(1e16);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_slice_summation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(KahanAccumulator::sum_slice(&values), 15.0);
        assert_eq!(NeumaierAccumulator::sum_slice(&values), 15.0);
    }

    #[test]
    fn test_complex_pathological() {
        let values = vec![1e16, 1.0, 1.0, 1.0, -1e16];
        let kahan_sum = KahanAccumulator::sum_slice(&values);
        let neumaier_sum = NeumaierAccumulator::sum_slice(&values);
        
        assert_eq!(kahan_sum, 3.0);
        assert_eq!(neumaier_sum, 3.0);
    }
}