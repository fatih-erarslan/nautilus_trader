//! Swing point detection algorithms
//!
//! This module implements various algorithms for detecting swing highs and lows
//! in price data, essential for Fibonacci analysis.

use crate::*;
use std::collections::VecDeque;

/// Swing point detector implementation
#[derive(Debug)]
pub struct SwingPointDetector {
    period: usize,
    min_strength: f64,
    use_confirmation: bool,
    history_buffer: VecDeque<f64>,
}

impl SwingPointDetector {
    /// Create a new swing point detector
    pub fn new(period: usize) -> Self {
        Self {
            period,
            min_strength: 0.0,
            use_confirmation: true,
            history_buffer: VecDeque::new(),
        }
    }
    
    /// Create a detector with custom parameters
    pub fn with_params(period: usize, min_strength: f64, use_confirmation: bool) -> Self {
        Self {
            period,
            min_strength,
            use_confirmation,
            history_buffer: VecDeque::new(),
        }
    }
    
    /// Detect swing points in price data
    pub fn detect_swings(&self, prices: &[f64]) -> FibonacciResult<SwingPoints> {
        if prices.len() < self.period * 2 + 1 {
            return Ok(SwingPoints::new(self.period));
        }
        
        let mut swing_points = SwingPoints::new(self.period);
        
        // Detect swing highs and lows
        for i in self.period..prices.len() - self.period {
            if self.is_swing_high(prices, i) {
                let strength = self.calculate_swing_strength(prices, i, true);
                if strength >= self.min_strength {
                    let swing_point = SwingPoint::new(i, prices[i]);
                    swing_points.add_high(swing_point);
                }
            }
            
            if self.is_swing_low(prices, i) {
                let strength = self.calculate_swing_strength(prices, i, false);
                if strength >= self.min_strength {
                    let swing_point = SwingPoint::new(i, prices[i]);
                    swing_points.add_low(swing_point);
                }
            }
        }
        
        Ok(swing_points)
    }
    
    /// Check if index represents a swing high
    fn is_swing_high(&self, prices: &[f64], index: usize) -> bool {
        let current_price = prices[index];
        
        // Check left side
        for i in (index - self.period)..index {
            if prices[i] >= current_price {
                return false;
            }
        }
        
        // Check right side
        for i in (index + 1)..=(index + self.period) {
            if prices[i] >= current_price {
                return false;
            }
        }
        
        true
    }
    
    /// Check if index represents a swing low
    fn is_swing_low(&self, prices: &[f64], index: usize) -> bool {
        let current_price = prices[index];
        
        // Check left side
        for i in (index - self.period)..index {
            if prices[i] <= current_price {
                return false;
            }
        }
        
        // Check right side
        for i in (index + 1)..=(index + self.period) {
            if prices[i] <= current_price {
                return false;
            }
        }
        
        true
    }
    
    /// Calculate swing strength
    fn calculate_swing_strength(&self, prices: &[f64], index: usize, is_high: bool) -> f64 {
        let current_price = prices[index];
        let start = index.saturating_sub(self.period);
        let end = (index + self.period + 1).min(prices.len());
        
        let mut total_diff = 0.0;
        let mut count = 0;
        
        for i in start..end {
            if i != index {
                let diff = if is_high {
                    current_price - prices[i]
                } else {
                    prices[i] - current_price
                };
                
                if diff > 0.0 {
                    total_diff += diff;
                    count += 1;
                }
            }
        }
        
        if count > 0 {
            total_diff / count as f64
        } else {
            0.0
        }
    }
    
    /// Update period for swing detection
    pub fn update_period(&mut self, period: usize) {
        self.period = period;
    }
    
    /// Set minimum strength threshold
    pub fn set_min_strength(&mut self, min_strength: f64) {
        self.min_strength = min_strength;
    }
    
    /// Enable or disable confirmation requirement
    pub fn set_use_confirmation(&mut self, use_confirmation: bool) {
        self.use_confirmation = use_confirmation;
    }
}

/// Advanced swing point detector with multiple algorithms
pub struct AdvancedSwingDetector {
    base_detector: SwingPointDetector,
    algorithm: SwingAlgorithm,
    adaptive_period: bool,
    volatility_adjustment: bool,
}

/// Swing detection algorithms
#[derive(Debug, Clone, Copy)]
pub enum SwingAlgorithm {
    /// Classic period-based detection
    Classic,
    /// Fractal-based detection
    Fractal,
    /// Pivot point detection
    Pivot,
    /// Adaptive period detection
    Adaptive,
    /// Volume-weighted detection
    VolumeWeighted,
}

impl AdvancedSwingDetector {
    /// Create a new advanced swing detector
    pub fn new(period: usize, algorithm: SwingAlgorithm) -> Self {
        Self {
            base_detector: SwingPointDetector::new(period),
            algorithm,
            adaptive_period: false,
            volatility_adjustment: false,
        }
    }
    
    /// Enable adaptive period adjustment
    pub fn with_adaptive_period(mut self, adaptive: bool) -> Self {
        self.adaptive_period = adaptive;
        self
    }
    
    /// Enable volatility-based adjustment
    pub fn with_volatility_adjustment(mut self, volatility: bool) -> Self {
        self.volatility_adjustment = volatility;
        self
    }
    
    /// Detect swings using the configured algorithm
    pub fn detect_swings_advanced(
        &self,
        prices: &[f64],
        volumes: Option<&[f64]>,
    ) -> FibonacciResult<SwingPoints> {
        match self.algorithm {
            SwingAlgorithm::Classic => self.base_detector.detect_swings(prices),
            SwingAlgorithm::Fractal => self.detect_fractal_swings(prices),
            SwingAlgorithm::Pivot => self.detect_pivot_swings(prices),
            SwingAlgorithm::Adaptive => self.detect_adaptive_swings(prices),
            SwingAlgorithm::VolumeWeighted => {
                if let Some(vols) = volumes {
                    self.detect_volume_weighted_swings(prices, vols)
                } else {
                    self.base_detector.detect_swings(prices)
                }
            }
        }
    }
    
    /// Fractal-based swing detection
    fn detect_fractal_swings(&self, prices: &[f64]) -> FibonacciResult<SwingPoints> {
        let mut swing_points = SwingPoints::new(self.base_detector.period);
        
        if prices.len() < 5 {
            return Ok(swing_points);
        }
        
        // Fractal detection requires at least 5 points
        for i in 2..prices.len() - 2 {
            // Fractal up: middle is higher than 2 bars on each side
            if prices[i] > prices[i-1] && prices[i] > prices[i-2] &&
               prices[i] > prices[i+1] && prices[i] > prices[i+2] {
                let swing_point = SwingPoint::new(i, prices[i]);
                swing_points.add_high(swing_point);
            }
            
            // Fractal down: middle is lower than 2 bars on each side
            if prices[i] < prices[i-1] && prices[i] < prices[i-2] &&
               prices[i] < prices[i+1] && prices[i] < prices[i+2] {
                let swing_point = SwingPoint::new(i, prices[i]);
                swing_points.add_low(swing_point);
            }
        }
        
        Ok(swing_points)
    }
    
    /// Pivot point-based swing detection
    fn detect_pivot_swings(&self, prices: &[f64]) -> FibonacciResult<SwingPoints> {
        let mut swing_points = SwingPoints::new(self.base_detector.period);
        let lookback = self.base_detector.period.max(3);
        
        if prices.len() < lookback * 2 + 1 {
            return Ok(swing_points);
        }
        
        for i in lookback..prices.len() - lookback {
            let current_price = prices[i];
            
            // Check for pivot high
            let mut is_pivot_high = true;
            for j in (i - lookback)..i {
                if prices[j] >= current_price {
                    is_pivot_high = false;
                    break;
                }
            }
            
            if is_pivot_high {
                for j in (i + 1)..=(i + lookback) {
                    if prices[j] >= current_price {
                        is_pivot_high = false;
                        break;
                    }
                }
            }
            
            if is_pivot_high {
                let swing_point = SwingPoint::new(i, current_price);
                swing_points.add_high(swing_point);
            }
            
            // Check for pivot low
            let mut is_pivot_low = true;
            for j in (i - lookback)..i {
                if prices[j] <= current_price {
                    is_pivot_low = false;
                    break;
                }
            }
            
            if is_pivot_low {
                for j in (i + 1)..=(i + lookback) {
                    if prices[j] <= current_price {
                        is_pivot_low = false;
                        break;
                    }
                }
            }
            
            if is_pivot_low {
                let swing_point = SwingPoint::new(i, current_price);
                swing_points.add_low(swing_point);
            }
        }
        
        Ok(swing_points)
    }
    
    /// Adaptive period swing detection
    fn detect_adaptive_swings(&self, prices: &[f64]) -> FibonacciResult<SwingPoints> {
        // Calculate adaptive period based on volatility
        let volatility = self.calculate_volatility(prices)?;
        let adaptive_period = self.calculate_adaptive_period(volatility);
        
        let mut adaptive_detector = SwingPointDetector::new(adaptive_period);
        adaptive_detector.detect_swings(prices)
    }
    
    /// Volume-weighted swing detection
    fn detect_volume_weighted_swings(&self, prices: &[f64], volumes: &[f64]) -> FibonacciResult<SwingPoints> {
        if prices.len() != volumes.len() {
            return Err(FibonacciError::InvalidInput(
                "Price and volume arrays must have same length".to_string()
            ));
        }
        
        let mut swing_points = SwingPoints::new(self.base_detector.period);
        let period = self.base_detector.period;
        
        for i in period..prices.len() - period {
            let volume_weighted_strength = self.calculate_volume_weighted_strength(prices, volumes, i);
            
            if self.base_detector.is_swing_high(prices, i) && volume_weighted_strength > 0.5 {
                let swing_point = SwingPoint::new(i, prices[i]);
                swing_points.add_high(swing_point);
            }
            
            if self.base_detector.is_swing_low(prices, i) && volume_weighted_strength > 0.5 {
                let swing_point = SwingPoint::new(i, prices[i]);
                swing_points.add_low(swing_point);
            }
        }
        
        Ok(swing_points)
    }
    
    /// Calculate volatility for adaptive period
    fn calculate_volatility(&self, prices: &[f64]) -> FibonacciResult<f64> {
        if prices.len() < 2 {
            return Ok(0.0);
        }
        
        let mut sum_squared_returns = 0.0;
        let mut count = 0;
        
        for i in 1..prices.len() {
            let return_val = (prices[i] / prices[i-1]).ln();
            sum_squared_returns += return_val * return_val;
            count += 1;
        }
        
        if count > 0 {
            Ok((sum_squared_returns / count as f64).sqrt())
        } else {
            Ok(0.0)
        }
    }
    
    /// Calculate adaptive period based on volatility
    fn calculate_adaptive_period(&self, volatility: f64) -> usize {
        let base_period = self.base_detector.period as f64;
        let volatility_factor = (volatility * 100.0).max(1.0).min(5.0);
        
        (base_period * volatility_factor).round() as usize
    }
    
    /// Calculate volume-weighted strength
    fn calculate_volume_weighted_strength(&self, prices: &[f64], volumes: &[f64], index: usize) -> f64 {
        let period = self.base_detector.period;
        let start = index.saturating_sub(period);
        let end = (index + period + 1).min(prices.len());
        
        let mut total_volume = 0.0;
        let mut weighted_volume = 0.0;
        let current_price = prices[index];
        
        for i in start..end {
            if i != index {
                let price_diff = (prices[i] - current_price).abs();
                let volume_weight = volumes[i] * (1.0 + price_diff / current_price);
                
                total_volume += volumes[i];
                weighted_volume += volume_weight;
            }
        }
        
        if total_volume > 0.0 {
            weighted_volume / total_volume
        } else {
            0.0
        }
    }
}

/// Swing point filter for removing noise
pub struct SwingPointFilter {
    min_retracement: f64,
    min_time_distance: usize,
    use_atr_filter: bool,
    atr_multiplier: f64,
}

impl SwingPointFilter {
    /// Create a new swing point filter
    pub fn new(min_retracement: f64, min_time_distance: usize) -> Self {
        Self {
            min_retracement,
            min_time_distance,
            use_atr_filter: false,
            atr_multiplier: 2.0,
        }
    }
    
    /// Enable ATR-based filtering
    pub fn with_atr_filter(mut self, use_atr: bool, multiplier: f64) -> Self {
        self.use_atr_filter = use_atr;
        self.atr_multiplier = multiplier;
        self
    }
    
    /// Filter swing points to remove noise
    pub fn filter_swings(&self, swing_points: &mut SwingPoints, prices: &[f64]) -> FibonacciResult<()> {
        self.filter_by_retracement(swing_points, prices)?;
        self.filter_by_time_distance(swing_points)?;
        
        if self.use_atr_filter {
            self.filter_by_atr(swing_points, prices)?;
        }
        
        Ok(())
    }
    
    /// Filter by minimum retracement
    fn filter_by_retracement(&self, swing_points: &mut SwingPoints, prices: &[f64]) -> FibonacciResult<()> {
        // Filter swing highs
        swing_points.swing_highs.retain(|swing| {
            let mut max_retracement = 0.0;
            
            // Check retracement from this swing
            for i in swing.index..prices.len() {
                let retracement = (swing.price - prices[i]) / swing.price;
                if retracement > max_retracement {
                    max_retracement = retracement;
                }
            }
            
            max_retracement >= self.min_retracement
        });
        
        // Filter swing lows
        swing_points.swing_lows.retain(|swing| {
            let mut max_retracement = 0.0;
            
            // Check retracement from this swing
            for i in swing.index..prices.len() {
                let retracement = (prices[i] - swing.price) / swing.price;
                if retracement > max_retracement {
                    max_retracement = retracement;
                }
            }
            
            max_retracement >= self.min_retracement
        });
        
        Ok(())
    }
    
    /// Filter by minimum time distance
    fn filter_by_time_distance(&self, swing_points: &mut SwingPoints) -> FibonacciResult<()> {
        // Filter swing highs
        let mut filtered_highs: Vec<SwingPoint> = Vec::new();
        for swing in &swing_points.swing_highs {
            if filtered_highs.is_empty() || 
               swing.index >= filtered_highs.last().unwrap().index + self.min_time_distance {
                filtered_highs.push(swing.clone());
            }
        }
        swing_points.swing_highs = filtered_highs;
        
        // Filter swing lows
        let mut filtered_lows: Vec<SwingPoint> = Vec::new();
        for swing in &swing_points.swing_lows {
            if filtered_lows.is_empty() || 
               swing.index >= filtered_lows.last().unwrap().index + self.min_time_distance {
                filtered_lows.push(swing.clone());
            }
        }
        swing_points.swing_lows = filtered_lows;
        
        Ok(())
    }
    
    /// Filter by ATR-based significance
    fn filter_by_atr(&self, swing_points: &mut SwingPoints, prices: &[f64]) -> FibonacciResult<()> {
        let atr = self.calculate_atr(prices)?;
        let threshold = atr * self.atr_multiplier;
        
        // Filter swing highs
        swing_points.swing_highs.retain(|swing| {
            let mut max_move = 0.0;
            
            // Check move from this swing
            for i in swing.index..prices.len() {
                let move_size = (swing.price - prices[i]).abs();
                if move_size > max_move {
                    max_move = move_size;
                }
            }
            
            max_move >= threshold
        });
        
        // Filter swing lows
        swing_points.swing_lows.retain(|swing| {
            let mut max_move = 0.0;
            
            // Check move from this swing
            for i in swing.index..prices.len() {
                let move_size = (prices[i] - swing.price).abs();
                if move_size > max_move {
                    max_move = move_size;
                }
            }
            
            max_move >= threshold
        });
        
        Ok(())
    }
    
    /// Calculate ATR for filtering
    fn calculate_atr(&self, prices: &[f64]) -> FibonacciResult<f64> {
        if prices.len() < 14 {
            return Ok(0.0);
        }
        
        let mut true_ranges = Vec::new();
        
        for i in 1..prices.len() {
            let tr = (prices[i] - prices[i-1]).abs();
            true_ranges.push(tr);
        }
        
        let period = 14.min(true_ranges.len());
        let atr = true_ranges[true_ranges.len() - period..]
            .iter()
            .sum::<f64>() / period as f64;
        
        Ok(atr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_swing_point_detector() {
        let detector = SwingPointDetector::new(2);
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0, 115.0, 85.0];
        
        let result = detector.detect_swings(&prices);
        assert!(result.is_ok());
        
        let swing_points = result.unwrap();
        assert!(!swing_points.swing_highs.is_empty());
        assert!(!swing_points.swing_lows.is_empty());
    }
    
    #[test]
    fn test_advanced_swing_detector() {
        let detector = AdvancedSwingDetector::new(3, SwingAlgorithm::Fractal);
        let prices = vec![100.0, 102.0, 104.0, 102.0, 100.0, 98.0, 100.0, 102.0];
        
        let result = detector.detect_swings_advanced(&prices, None);
        assert!(result.is_ok());
        
        let swing_points = result.unwrap();
        assert!(!swing_points.is_empty());
    }
    
    #[test]
    fn test_volume_weighted_detection() {
        let detector = AdvancedSwingDetector::new(2, SwingAlgorithm::VolumeWeighted);
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0, 115.0];
        let volumes = vec![1000.0, 1500.0, 800.0, 2000.0, 900.0, 1800.0];
        
        let result = detector.detect_swings_advanced(&prices, Some(&volumes));
        assert!(result.is_ok());
        
        let swing_points = result.unwrap();
        assert!(!swing_points.is_empty());
    }
    
    #[test]
    fn test_swing_point_filter() {
        let mut detector = SwingPointDetector::new(2);
        let prices = vec![100.0, 105.0, 95.0, 110.0, 90.0, 115.0, 85.0, 120.0];
        
        let mut swing_points = detector.detect_swings(&prices).unwrap();
        let initial_count = swing_points.total_points();
        
        let filter = SwingPointFilter::new(0.05, 2);
        let result = filter.filter_swings(&mut swing_points, &prices);
        assert!(result.is_ok());
        
        // Should have filtered out some points
        assert!(swing_points.total_points() <= initial_count);
    }
    
    #[test]
    fn test_adaptive_swing_detection() {
        let detector = AdvancedSwingDetector::new(3, SwingAlgorithm::Adaptive);
        let prices = vec![100.0, 102.0, 104.0, 102.0, 100.0, 98.0, 100.0, 102.0, 104.0];
        
        let result = detector.detect_swings_advanced(&prices, None);
        assert!(result.is_ok());
        
        let swing_points = result.unwrap();
        assert!(!swing_points.is_empty());
    }
}