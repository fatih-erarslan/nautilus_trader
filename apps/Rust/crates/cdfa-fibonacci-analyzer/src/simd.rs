//! SIMD-accelerated Fibonacci calculations
//!
//! This module provides SIMD implementations for high-performance Fibonacci analysis
//! operations, targeting sub-microsecond performance requirements.

use crate::*;
use std::simd::f64x4;

/// SIMD-accelerated Fibonacci level calculator
pub struct SimdFibonacciCalculator {
    enable_simd: bool,
}

impl SimdFibonacciCalculator {
    /// Create a new SIMD calculator
    pub fn new(enable_simd: bool) -> Self {
        Self { enable_simd }
    }
    
    /// Calculate Fibonacci retracement levels using SIMD acceleration
    pub fn calculate_retracement_levels_simd(
        &self,
        high_price: f64,
        low_price: f64,
        levels: &[f64],
    ) -> Vec<f64> {
        if !self.enable_simd || levels.len() < 4 {
            return self.calculate_retracement_levels_scalar(high_price, low_price, levels);
        }
        
        let price_range = high_price - low_price;
        let mut results = Vec::with_capacity(levels.len());
        
        // Process levels in chunks of 4 using SIMD
        for chunk in levels.chunks(4) {
            let mut level_vec = [0.0; 4];
            for (i, &level) in chunk.iter().enumerate() {
                level_vec[i] = level;
            }
            
            let levels_simd = f64x4::from_array(level_vec);
            let range_simd = f64x4::splat(price_range);
            let high_simd = f64x4::splat(high_price);
            
            let retracement_simd = high_simd - (levels_simd * range_simd);
            let retracement_array = retracement_simd.to_array();
            
            for (i, &value) in retracement_array.iter().enumerate() {
                if i < chunk.len() {
                    results.push(value);
                }
            }
        }
        
        results
    }
    
    /// Scalar fallback implementation
    fn calculate_retracement_levels_scalar(
        &self,
        high_price: f64,
        low_price: f64,
        levels: &[f64],
    ) -> Vec<f64> {
        let price_range = high_price - low_price;
        levels.iter()
            .map(|&level| high_price - (level * price_range))
            .collect()
    }
    
    /// Calculate Fibonacci extension levels using SIMD acceleration
    pub fn calculate_extension_levels_simd(
        &self,
        base_price: f64,
        target_price: f64,
        extensions: &[f64],
    ) -> Vec<f64> {
        if !self.enable_simd || extensions.len() < 4 {
            return self.calculate_extension_levels_scalar(base_price, target_price, extensions);
        }
        
        let price_range = target_price - base_price;
        let mut results = Vec::with_capacity(extensions.len());
        
        // Process extensions in chunks of 4 using SIMD
        for chunk in extensions.chunks(4) {
            let mut ext_vec = [0.0; 4];
            for (i, &ext) in chunk.iter().enumerate() {
                ext_vec[i] = ext;
            }
            
            let extensions_simd = f64x4::from_array(ext_vec);
            let range_simd = f64x4::splat(price_range);
            let target_simd = f64x4::splat(target_price);
            
            let extension_simd = target_simd + ((extensions_simd - f64x4::splat(1.0)) * range_simd);
            let extension_array = extension_simd.to_array();
            
            for (i, &value) in extension_array.iter().enumerate() {
                if i < chunk.len() {
                    results.push(value);
                }
            }
        }
        
        results
    }
    
    /// Scalar fallback for extension levels
    fn calculate_extension_levels_scalar(
        &self,
        base_price: f64,
        target_price: f64,
        extensions: &[f64],
    ) -> Vec<f64> {
        let price_range = target_price - base_price;
        extensions.iter()
            .map(|&ext| target_price + ((ext - 1.0) * price_range))
            .collect()
    }
    
    /// Calculate distance scores using SIMD acceleration
    pub fn calculate_distance_scores_simd(
        &self,
        current_price: f64,
        levels: &[f64],
    ) -> Vec<f64> {
        if !self.enable_simd || levels.len() < 4 {
            return self.calculate_distance_scores_scalar(current_price, levels);
        }
        
        let mut results = Vec::with_capacity(levels.len());
        
        // Process levels in chunks of 4 using SIMD
        for chunk in levels.chunks(4) {
            let mut level_vec = [0.0; 4];
            for (i, &level) in chunk.iter().enumerate() {
                level_vec[i] = level;
            }
            
            let levels_simd = f64x4::from_array(level_vec);
            let current_simd = f64x4::splat(current_price);
            
            // Calculate relative distances
            let distances_simd = ((levels_simd - current_simd) / current_simd).abs();
            let distances_array = distances_simd.to_array();
            
            for (i, &distance) in distances_array.iter().enumerate() {
                if i < chunk.len() {
                    results.push(distance);
                }
            }
        }
        
        results
    }
    
    /// Scalar fallback for distance scores
    fn calculate_distance_scores_scalar(
        &self,
        current_price: f64,
        levels: &[f64],
    ) -> Vec<f64> {
        levels.iter()
            .map(|&level| ((level - current_price) / current_price).abs())
            .collect()
    }
}

/// SIMD-accelerated swing point detector
pub struct SimdSwingDetector {
    enable_simd: bool,
}

impl SimdSwingDetector {
    /// Create a new SIMD swing detector
    pub fn new(enable_simd: bool) -> Self {
        Self { enable_simd }
    }
    
    /// Detect swing points using SIMD acceleration
    pub fn detect_swings_simd(&self, prices: &[f64], period: usize) -> Vec<usize> {
        if !self.enable_simd || prices.len() < period * 2 {
            return self.detect_swings_scalar(prices, period);
        }
        
        let mut swing_indices = Vec::new();
        
        // SIMD-accelerated swing detection logic would go here
        // For now, fall back to scalar implementation
        self.detect_swings_scalar(prices, period)
    }
    
    /// Scalar fallback for swing detection
    fn detect_swings_scalar(&self, prices: &[f64], period: usize) -> Vec<usize> {
        let mut swing_indices = Vec::new();
        
        for i in period..prices.len() - period {
            let current_price = prices[i];
            let mut is_swing_high = true;
            let mut is_swing_low = true;
            
            // Check if current price is a swing high or low
            for j in (i - period)..=(i + period) {
                if j != i {
                    if prices[j] >= current_price {
                        is_swing_high = false;
                    }
                    if prices[j] <= current_price {
                        is_swing_low = false;
                    }
                }
            }
            
            if is_swing_high || is_swing_low {
                swing_indices.push(i);
            }
        }
        
        swing_indices
    }
}

/// SIMD-accelerated alignment scorer
pub struct SimdAlignmentScorer {
    enable_simd: bool,
}

impl SimdAlignmentScorer {
    /// Create a new SIMD alignment scorer
    pub fn new(enable_simd: bool) -> Self {
        Self { enable_simd }
    }
    
    /// Calculate alignment scores using SIMD acceleration
    pub fn calculate_alignment_simd(
        &self,
        current_price: f64,
        levels: &[f64],
        tolerance: f64,
    ) -> f64 {
        if !self.enable_simd || levels.len() < 4 {
            return self.calculate_alignment_scalar(current_price, levels, tolerance);
        }
        
        let mut total_score = 0.0;
        let mut count = 0;
        
        // Process levels in chunks of 4 using SIMD
        for chunk in levels.chunks(4) {
            let mut level_vec = [0.0; 4];
            for (i, &level) in chunk.iter().enumerate() {
                level_vec[i] = level;
            }
            
            let levels_simd = f64x4::from_array(level_vec);
            let current_simd = f64x4::splat(current_price);
            let tolerance_simd = f64x4::splat(tolerance);
            
            // Calculate distances
            let distances_simd = ((levels_simd - current_simd) / current_simd).abs();
            
            // Calculate scores
            let scores_simd = (f64x4::splat(1.0) - (distances_simd / tolerance_simd))
                .simd_max(f64x4::splat(0.0));
            
            let scores_array = scores_simd.to_array();
            
            for (i, &score) in scores_array.iter().enumerate() {
                if i < chunk.len() {
                    total_score += score;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
    
    /// Scalar fallback for alignment calculation
    fn calculate_alignment_scalar(
        &self,
        current_price: f64,
        levels: &[f64],
        tolerance: f64,
    ) -> f64 {
        let mut total_score = 0.0;
        let mut count = 0;
        
        for &level in levels {
            let distance = ((level - current_price) / current_price).abs();
            let score = (1.0 - (distance / tolerance)).max(0.0);
            total_score += score;
            count += 1;
        }
        
        if count > 0 {
            total_score / count as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_simd_retracement_calculation() {
        let calculator = SimdFibonacciCalculator::new(true);
        let high_price = 110.0;
        let low_price = 90.0;
        let levels = vec![0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
        
        let results = calculator.calculate_retracement_levels_simd(high_price, low_price, &levels);
        
        assert_eq!(results.len(), levels.len());
        assert_relative_eq!(results[0], 110.0, epsilon = 1e-10); // 0% retracement
        assert_relative_eq!(results[6], 90.0, epsilon = 1e-10);  // 100% retracement
    }
    
    #[test]
    fn test_simd_extension_calculation() {
        let calculator = SimdFibonacciCalculator::new(true);
        let base_price = 100.0;
        let target_price = 110.0;
        let extensions = vec![1.0, 1.272, 1.618, 2.618];
        
        let results = calculator.calculate_extension_levels_simd(base_price, target_price, &extensions);
        
        assert_eq!(results.len(), extensions.len());
        assert_relative_eq!(results[0], 110.0, epsilon = 1e-10); // 100% extension
    }
    
    #[test]
    fn test_simd_distance_scores() {
        let calculator = SimdFibonacciCalculator::new(true);
        let current_price = 100.0;
        let levels = vec![95.0, 100.0, 105.0, 110.0];
        
        let results = calculator.calculate_distance_scores_simd(current_price, &levels);
        
        assert_eq!(results.len(), levels.len());
        assert_relative_eq!(results[1], 0.0, epsilon = 1e-10); // Exact match
        assert_relative_eq!(results[0], 0.05, epsilon = 1e-10); // 5% away
    }
    
    #[test]
    fn test_simd_vs_scalar_consistency() {
        let calculator = SimdFibonacciCalculator::new(true);
        let high_price = 120.0;
        let low_price = 80.0;
        let levels = vec![0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0];
        
        let simd_results = calculator.calculate_retracement_levels_simd(high_price, low_price, &levels);
        let scalar_results = calculator.calculate_retracement_levels_scalar(high_price, low_price, &levels);
        
        assert_eq!(simd_results.len(), scalar_results.len());
        for (simd, scalar) in simd_results.iter().zip(scalar_results.iter()) {
            assert_relative_eq!(simd, scalar, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_simd_alignment_scorer() {
        let scorer = SimdAlignmentScorer::new(true);
        let current_price = 100.0;
        let levels = vec![99.0, 100.0, 101.0, 102.0];
        let tolerance = 0.02;
        
        let score = scorer.calculate_alignment_simd(current_price, &levels, tolerance);
        
        assert!(score >= 0.0 && score <= 1.0);
    }
    
    #[test]
    fn test_simd_swing_detector() {
        let detector = SimdSwingDetector::new(true);
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0, 115.0, 85.0];
        let period = 2;
        
        let swings = detector.detect_swings_simd(&prices, period);
        
        assert!(!swings.is_empty());
    }
}