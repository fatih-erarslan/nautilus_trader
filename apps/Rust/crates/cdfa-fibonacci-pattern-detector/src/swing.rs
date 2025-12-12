/*!
# Swing Point Detection Module

High-performance swing high and low detection for pattern analysis.
Identifies significant turning points in price data using multiple algorithms.

## Features

- **Window-based Detection**: Configurable window size for swing identification
- **Strength Filtering**: Minimum strength requirements for swing points
- **Multiple Algorithms**: Various swing detection methods
- **Real-time Processing**: Efficient online swing detection
- **Noise Filtering**: Removes insignificant price movements

## Algorithms

- **Peak/Valley Detection**: Simple local maxima/minima identification
- **Strength-based**: Swing points with minimum strength requirements
- **Adaptive**: Dynamic window sizing based on volatility
- **Fractal**: Williams fractal-based swing detection
*/

use crate::{PatternResult, PatternError};
use std::collections::VecDeque;

/// Swing point detection algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwingAlgorithm {
    /// Simple peak/valley detection
    PeakValley,
    /// Strength-based detection
    Strength,
    /// Adaptive window sizing
    Adaptive,
    /// Williams fractal detection
    Fractal,
    /// Zigzag pattern detection
    ZigZag,
}

/// Swing point with metadata
#[derive(Debug, Clone)]
pub struct SwingPoint {
    /// Index in the price series
    pub index: usize,
    /// Price value at swing point
    pub price: f64,
    /// Swing type (high or low)
    pub swing_type: SwingType,
    /// Strength of the swing point
    pub strength: f64,
    /// Confirmation status
    pub confirmed: bool,
}

/// Swing point type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SwingType {
    /// Swing high (local maximum)
    High,
    /// Swing low (local minimum)
    Low,
}

/// Swing detector configuration
#[derive(Debug, Clone)]
pub struct SwingDetectorConfig {
    /// Window size for swing detection
    pub window_size: usize,
    /// Minimum swing strength
    pub min_strength: f64,
    /// Algorithm to use
    pub algorithm: SwingAlgorithm,
    /// Minimum price movement percentage
    pub min_movement: f64,
    /// Require confirmation
    pub require_confirmation: bool,
    /// Maximum lookback for confirmation
    pub confirmation_lookback: usize,
}

impl Default for SwingDetectorConfig {
    fn default() -> Self {
        Self {
            window_size: 5,
            min_strength: 0.01,
            algorithm: SwingAlgorithm::PeakValley,
            min_movement: 0.001,
            require_confirmation: false,
            confirmation_lookback: 3,
        }
    }
}

/// High-performance swing detector
pub struct SwingDetector {
    config: SwingDetectorConfig,
    price_buffer: VecDeque<f64>,
    swing_buffer: VecDeque<SwingPoint>,
    last_swing_high: Option<SwingPoint>,
    last_swing_low: Option<SwingPoint>,
}

impl SwingDetector {
    /// Create new swing detector
    pub fn new(window_size: usize, min_strength: f64) -> Self {
        let config = SwingDetectorConfig {
            window_size,
            min_strength,
            ..Default::default()
        };
        
        Self {
            config,
            price_buffer: VecDeque::new(),
            swing_buffer: VecDeque::new(),
            last_swing_high: None,
            last_swing_low: None,
        }
    }
    
    /// Create with custom configuration
    pub fn with_config(config: SwingDetectorConfig) -> Self {
        Self {
            config,
            price_buffer: VecDeque::new(),
            swing_buffer: VecDeque::new(),
            last_swing_high: None,
            last_swing_low: None,
        }
    }
    
    /// Detect swing points in price arrays
    pub fn detect_swings(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> PatternResult<(Vec<usize>, Vec<usize>)> {
        if highs.len() != lows.len() || highs.len() != closes.len() {
            return Err(PatternError::InvalidInput(
                "Price arrays must have same length".to_string()
            ));
        }
        
        if highs.len() < self.config.window_size * 2 {
            return Ok((Vec::new(), Vec::new()));
        }
        
        match self.config.algorithm {
            SwingAlgorithm::PeakValley => self.detect_peak_valley(highs, lows, closes),
            SwingAlgorithm::Strength => self.detect_strength_based(highs, lows, closes),
            SwingAlgorithm::Adaptive => self.detect_adaptive(highs, lows, closes),
            SwingAlgorithm::Fractal => self.detect_fractal(highs, lows, closes),
            SwingAlgorithm::ZigZag => self.detect_zigzag(highs, lows, closes),
        }
    }
    
    /// Simple peak/valley detection
    fn detect_peak_valley(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> PatternResult<(Vec<usize>, Vec<usize>)> {
        let mut swing_highs = Vec::new();
        let mut swing_lows = Vec::new();
        let window = self.config.window_size;
        
        for i in window..highs.len() - window {
            let mut is_swing_high = true;
            let mut is_swing_low = true;
            
            // Check if current point is highest/lowest in window
            for j in 1..=window {
                if highs[i] <= highs[i - j] || highs[i] <= highs[i + j] {
                    is_swing_high = false;
                }
                if lows[i] >= lows[i - j] || lows[i] >= lows[i + j] {
                    is_swing_low = false;
                }
            }
            
            // Apply minimum movement filter
            if is_swing_high {
                let movement = self.calculate_movement(highs[i], closes, i, window);
                if movement >= self.config.min_movement {
                    swing_highs.push(i);
                }
            }
            
            if is_swing_low {
                let movement = self.calculate_movement(lows[i], closes, i, window);
                if movement >= self.config.min_movement {
                    swing_lows.push(i);
                }
            }
        }
        
        Ok((swing_highs, swing_lows))
    }
    
    /// Strength-based swing detection
    fn detect_strength_based(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> PatternResult<(Vec<usize>, Vec<usize>)> {
        let mut swing_highs = Vec::new();
        let mut swing_lows = Vec::new();
        let window = self.config.window_size;
        
        for i in window..highs.len() - window {
            let high_strength = self.calculate_swing_strength(highs, i, window, SwingType::High);
            let low_strength = self.calculate_swing_strength(lows, i, window, SwingType::Low);
            
            if high_strength >= self.config.min_strength {
                swing_highs.push(i);
            }
            
            if low_strength >= self.config.min_strength {
                swing_lows.push(i);
            }
        }
        
        Ok((swing_highs, swing_lows))
    }
    
    /// Adaptive window swing detection
    fn detect_adaptive(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> PatternResult<(Vec<usize>, Vec<usize>)> {
        let mut swing_highs = Vec::new();
        let mut swing_lows = Vec::new();
        let base_window = self.config.window_size;
        
        // Calculate adaptive window based on volatility
        let volatility = self.calculate_volatility(closes);
        let adaptive_window = self.adapt_window_size(base_window, volatility);
        
        for i in adaptive_window..highs.len() - adaptive_window {
            let mut is_swing_high = true;
            let mut is_swing_low = true;
            
            // Use adaptive window for detection
            for j in 1..=adaptive_window {
                if highs[i] <= highs[i - j] || highs[i] <= highs[i + j] {
                    is_swing_high = false;
                }
                if lows[i] >= lows[i - j] || lows[i] >= lows[i + j] {
                    is_swing_low = false;
                }
            }
            
            if is_swing_high {
                let strength = self.calculate_swing_strength(highs, i, adaptive_window, SwingType::High);
                if strength >= self.config.min_strength {
                    swing_highs.push(i);
                }
            }
            
            if is_swing_low {
                let strength = self.calculate_swing_strength(lows, i, adaptive_window, SwingType::Low);
                if strength >= self.config.min_strength {
                    swing_lows.push(i);
                }
            }
        }
        
        Ok((swing_highs, swing_lows))
    }
    
    /// Williams fractal detection
    fn detect_fractal(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> PatternResult<(Vec<usize>, Vec<usize>)> {
        let mut swing_highs = Vec::new();
        let mut swing_lows = Vec::new();
        let window = self.config.window_size.max(2); // Minimum window of 2 for fractals
        
        for i in window..highs.len() - window {
            // Fractal high: current high is highest in window
            let mut is_fractal_high = true;
            for j in 1..=window {
                if highs[i] <= highs[i - j] || highs[i] <= highs[i + j] {
                    is_fractal_high = false;
                    break;
                }
            }
            
            // Fractal low: current low is lowest in window
            let mut is_fractal_low = true;
            for j in 1..=window {
                if lows[i] >= lows[i - j] || lows[i] >= lows[i + j] {
                    is_fractal_low = false;
                    break;
                }
            }
            
            if is_fractal_high {
                swing_highs.push(i);
            }
            
            if is_fractal_low {
                swing_lows.push(i);
            }
        }
        
        Ok((swing_highs, swing_lows))
    }
    
    /// ZigZag pattern detection
    fn detect_zigzag(
        &self,
        highs: &[f64],
        lows: &[f64],
        closes: &[f64],
    ) -> PatternResult<(Vec<usize>, Vec<usize>)> {
        let mut swing_highs = Vec::new();
        let mut swing_lows = Vec::new();
        
        if closes.len() < 3 {
            return Ok((swing_highs, swing_lows));
        }
        
        let min_change = self.config.min_movement;
        let mut current_trend = TrendDirection::Neutral;
        let mut last_extreme_idx = 0;
        let mut last_extreme_price = closes[0];
        
        for i in 1..closes.len() {
            let price = closes[i];
            let high = highs[i];
            let low = lows[i];
            
            match current_trend {
                TrendDirection::Neutral => {
                    // Determine initial trend
                    if price > last_extreme_price * (1.0 + min_change) {
                        current_trend = TrendDirection::Up;
                        last_extreme_idx = i;
                        last_extreme_price = high;
                    } else if price < last_extreme_price * (1.0 - min_change) {
                        current_trend = TrendDirection::Down;
                        last_extreme_idx = i;
                        last_extreme_price = low;
                    }
                }
                TrendDirection::Up => {
                    if high > last_extreme_price {
                        // New high in uptrend
                        last_extreme_idx = i;
                        last_extreme_price = high;
                    } else if low < last_extreme_price * (1.0 - min_change) {
                        // Trend reversal to down
                        swing_highs.push(last_extreme_idx);
                        current_trend = TrendDirection::Down;
                        last_extreme_idx = i;
                        last_extreme_price = low;
                    }
                }
                TrendDirection::Down => {
                    if low < last_extreme_price {
                        // New low in downtrend
                        last_extreme_idx = i;
                        last_extreme_price = low;
                    } else if high > last_extreme_price * (1.0 + min_change) {
                        // Trend reversal to up
                        swing_lows.push(last_extreme_idx);
                        current_trend = TrendDirection::Up;
                        last_extreme_idx = i;
                        last_extreme_price = high;
                    }
                }
            }
        }
        
        // Add final extreme if trend is established
        match current_trend {
            TrendDirection::Up => swing_highs.push(last_extreme_idx),
            TrendDirection::Down => swing_lows.push(last_extreme_idx),
            TrendDirection::Neutral => {}
        }
        
        Ok((swing_highs, swing_lows))
    }
    
    /// Calculate swing strength
    fn calculate_swing_strength(
        &self,
        prices: &[f64],
        index: usize,
        window: usize,
        swing_type: SwingType,
    ) -> f64 {
        let center_price = prices[index];
        let mut strength = 0.0;
        let mut count = 0;
        
        for i in 1..=window {
            if index >= i && index + i < prices.len() {
                let left_price = prices[index - i];
                let right_price = prices[index + i];
                
                match swing_type {
                    SwingType::High => {
                        if center_price > left_price && center_price > right_price {
                            let left_diff = (center_price - left_price) / left_price;
                            let right_diff = (center_price - right_price) / right_price;
                            strength += (left_diff + right_diff) / 2.0;
                            count += 1;
                        }
                    }
                    SwingType::Low => {
                        if center_price < left_price && center_price < right_price {
                            let left_diff = (left_price - center_price) / left_price;
                            let right_diff = (right_price - center_price) / right_price;
                            strength += (left_diff + right_diff) / 2.0;
                            count += 1;
                        }
                    }
                }
            }
        }
        
        if count > 0 {
            strength / count as f64
        } else {
            0.0
        }
    }
    
    /// Calculate price movement
    fn calculate_movement(
        &self,
        price: f64,
        closes: &[f64],
        index: usize,
        window: usize,
    ) -> f64 {
        let start_idx = index.saturating_sub(window);
        let end_idx = (index + window).min(closes.len() - 1);
        
        let mut min_price = f64::INFINITY;
        let mut max_price = f64::NEG_INFINITY;
        
        for i in start_idx..=end_idx {
            min_price = min_price.min(closes[i]);
            max_price = max_price.max(closes[i]);
        }
        
        if min_price > 0.0 {
            (max_price - min_price) / min_price
        } else {
            0.0
        }
    }
    
    /// Calculate volatility
    fn calculate_volatility(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        
        let mut returns = Vec::new();
        for i in 1..prices.len() {
            let return_val = (prices[i] - prices[i - 1]) / prices[i - 1];
            returns.push(return_val);
        }
        
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let variance = returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / returns.len() as f64;
        
        variance.sqrt()
    }
    
    /// Adapt window size based on volatility
    fn adapt_window_size(&self, base_window: usize, volatility: f64) -> usize {
        let volatility_factor = (volatility * 100.0).max(0.1).min(10.0);
        let adapted_window = (base_window as f64 / volatility_factor).round() as usize;
        adapted_window.max(2).min(base_window * 2)
    }
    
    /// Add new price data for real-time processing
    pub fn add_price(&mut self, price: f64) -> Vec<SwingPoint> {
        self.price_buffer.push_back(price);
        
        // Maintain buffer size
        if self.price_buffer.len() > self.config.window_size * 4 {
            self.price_buffer.pop_front();
        }
        
        // Detect new swing points
        self.detect_realtime_swings()
    }
    
    /// Detect swing points in real-time
    fn detect_realtime_swings(&mut self) -> Vec<SwingPoint> {
        let mut new_swings = Vec::new();
        
        if self.price_buffer.len() < self.config.window_size * 2 + 1 {
            return new_swings;
        }
        
        let prices: Vec<f64> = self.price_buffer.iter().copied().collect();
        let center_idx = prices.len() / 2;
        
        // Check for swing high
        if self.is_swing_high(&prices, center_idx) {
            let swing_point = SwingPoint {
                index: center_idx,
                price: prices[center_idx],
                swing_type: SwingType::High,
                strength: self.calculate_swing_strength(&prices, center_idx, self.config.window_size, SwingType::High),
                confirmed: !self.config.require_confirmation,
            };
            
            if swing_point.strength >= self.config.min_strength {
                new_swings.push(swing_point.clone());
                self.last_swing_high = Some(swing_point);
            }
        }
        
        // Check for swing low
        if self.is_swing_low(&prices, center_idx) {
            let swing_point = SwingPoint {
                index: center_idx,
                price: prices[center_idx],
                swing_type: SwingType::Low,
                strength: self.calculate_swing_strength(&prices, center_idx, self.config.window_size, SwingType::Low),
                confirmed: !self.config.require_confirmation,
            };
            
            if swing_point.strength >= self.config.min_strength {
                new_swings.push(swing_point.clone());
                self.last_swing_low = Some(swing_point);
            }
        }
        
        new_swings
    }
    
    /// Check if index is a swing high
    fn is_swing_high(&self, prices: &[f64], index: usize) -> bool {
        let window = self.config.window_size;
        
        if index < window || index + window >= prices.len() {
            return false;
        }
        
        let center_price = prices[index];
        
        for i in 1..=window {
            if center_price <= prices[index - i] || center_price <= prices[index + i] {
                return false;
            }
        }
        
        true
    }
    
    /// Check if index is a swing low
    fn is_swing_low(&self, prices: &[f64], index: usize) -> bool {
        let window = self.config.window_size;
        
        if index < window || index + window >= prices.len() {
            return false;
        }
        
        let center_price = prices[index];
        
        for i in 1..=window {
            if center_price >= prices[index - i] || center_price >= prices[index + i] {
                return false;
            }
        }
        
        true
    }
    
    /// Get current swing buffer
    pub fn get_swing_buffer(&self) -> &VecDeque<SwingPoint> {
        &self.swing_buffer
    }
    
    /// Get last swing high
    pub fn get_last_swing_high(&self) -> Option<&SwingPoint> {
        self.last_swing_high.as_ref()
    }
    
    /// Get last swing low
    pub fn get_last_swing_low(&self) -> Option<&SwingPoint> {
        self.last_swing_low.as_ref()
    }
    
    /// Reset detector state
    pub fn reset(&mut self) {
        self.price_buffer.clear();
        self.swing_buffer.clear();
        self.last_swing_high = None;
        self.last_swing_low = None;
    }
}

/// Trend direction for ZigZag detection
#[derive(Debug, Clone, Copy, PartialEq)]
enum TrendDirection {
    Up,
    Down,
    Neutral,
}

/// Swing point statistics
#[derive(Debug, Clone)]
pub struct SwingStatistics {
    /// Total swing highs detected
    pub total_highs: usize,
    /// Total swing lows detected
    pub total_lows: usize,
    /// Average swing strength
    pub avg_strength: f64,
    /// Average time between swings
    pub avg_time_between: f64,
    /// Swing success rate (if validated)
    pub success_rate: f64,
}

impl SwingStatistics {
    /// Calculate statistics from swing points
    pub fn calculate(swing_points: &[SwingPoint]) -> Self {
        let total_highs = swing_points.iter().filter(|s| s.swing_type == SwingType::High).count();
        let total_lows = swing_points.iter().filter(|s| s.swing_type == SwingType::Low).count();
        
        let avg_strength = if !swing_points.is_empty() {
            swing_points.iter().map(|s| s.strength).sum::<f64>() / swing_points.len() as f64
        } else {
            0.0
        };
        
        let avg_time_between = if swing_points.len() > 1 {
            let mut total_time = 0.0;
            for i in 1..swing_points.len() {
                total_time += (swing_points[i].index - swing_points[i - 1].index) as f64;
            }
            total_time / (swing_points.len() - 1) as f64
        } else {
            0.0
        };
        
        let success_rate = swing_points.iter().filter(|s| s.confirmed).count() as f64 / swing_points.len() as f64;
        
        Self {
            total_highs,
            total_lows,
            avg_strength,
            avg_time_between,
            success_rate,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_swing_detector_creation() {
        let detector = SwingDetector::new(5, 0.01);
        assert_eq!(detector.config.window_size, 5);
        assert_eq!(detector.config.min_strength, 0.01);
    }
    
    #[test]
    fn test_peak_valley_detection() {
        let detector = SwingDetector::new(2, 0.01);
        
        let highs = vec![100.0, 105.0, 110.0, 108.0, 115.0, 112.0, 118.0, 115.0, 120.0];
        let lows = vec![95.0, 98.0, 102.0, 105.0, 107.0, 108.0, 110.0, 112.0, 115.0];
        let closes = vec![98.0, 103.0, 108.0, 106.0, 112.0, 110.0, 115.0, 113.0, 118.0];
        
        let result = detector.detect_swings(&highs, &lows, &closes);
        assert!(result.is_ok());
        
        let (swing_highs, swing_lows) = result.unwrap();
        assert!(!swing_highs.is_empty() || !swing_lows.is_empty());
    }
    
    #[test]
    fn test_strength_calculation() {
        let detector = SwingDetector::new(2, 0.01);
        let prices = vec![100.0, 105.0, 110.0, 108.0, 115.0];
        
        let strength = detector.calculate_swing_strength(&prices, 2, 2, SwingType::High);
        assert!(strength >= 0.0);
    }
    
    #[test]
    fn test_volatility_calculation() {
        let detector = SwingDetector::new(5, 0.01);
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0];
        
        let volatility = detector.calculate_volatility(&prices);
        assert!(volatility > 0.0);
    }
    
    #[test]
    fn test_adaptive_window_sizing() {
        let detector = SwingDetector::new(5, 0.01);
        
        let low_vol_window = detector.adapt_window_size(5, 0.01);
        let high_vol_window = detector.adapt_window_size(5, 0.1);
        
        assert!(low_vol_window >= high_vol_window);
    }
    
    #[test]
    fn test_fractal_detection() {
        let config = SwingDetectorConfig {
            algorithm: SwingAlgorithm::Fractal,
            window_size: 2,
            ..Default::default()
        };
        let detector = SwingDetector::with_config(config);
        
        let highs = vec![100.0, 105.0, 110.0, 108.0, 115.0, 112.0, 118.0];
        let lows = vec![95.0, 98.0, 102.0, 105.0, 107.0, 108.0, 110.0];
        let closes = vec![98.0, 103.0, 108.0, 106.0, 112.0, 110.0, 115.0];
        
        let result = detector.detect_swings(&highs, &lows, &closes);
        assert!(result.is_ok());
    }
    
    #[test]
    fn test_zigzag_detection() {
        let config = SwingDetectorConfig {
            algorithm: SwingAlgorithm::ZigZag,
            min_movement: 0.05,
            ..Default::default()
        };
        let detector = SwingDetector::with_config(config);
        
        let highs = vec![100.0, 105.0, 110.0, 108.0, 115.0, 112.0, 118.0, 115.0, 120.0];
        let lows = vec![95.0, 98.0, 102.0, 105.0, 107.0, 108.0, 110.0, 112.0, 115.0];
        let closes = vec![98.0, 103.0, 108.0, 106.0, 112.0, 110.0, 115.0, 113.0, 118.0];
        
        let result = detector.detect_swings(&highs, &lows, &closes);
        assert!(result.is_ok());
        
        let (swing_highs, swing_lows) = result.unwrap();
        println!("ZigZag swing highs: {:?}", swing_highs);
        println!("ZigZag swing lows: {:?}", swing_lows);
    }
    
    #[test]
    fn test_realtime_swing_detection() {
        let mut detector = SwingDetector::new(3, 0.01);
        let prices = vec![100.0, 101.0, 102.0, 105.0, 103.0, 104.0, 107.0, 106.0, 108.0];
        
        let mut all_swings = Vec::new();
        for price in prices {
            let new_swings = detector.add_price(price);
            all_swings.extend(new_swings);
        }
        
        println!("Real-time detected swings: {}", all_swings.len());
    }
    
    #[test]
    fn test_swing_statistics() {
        let swing_points = vec![
            SwingPoint {
                index: 0,
                price: 100.0,
                swing_type: SwingType::High,
                strength: 0.05,
                confirmed: true,
            },
            SwingPoint {
                index: 5,
                price: 95.0,
                swing_type: SwingType::Low,
                strength: 0.03,
                confirmed: true,
            },
            SwingPoint {
                index: 10,
                price: 110.0,
                swing_type: SwingType::High,
                strength: 0.07,
                confirmed: false,
            },
        ];
        
        let stats = SwingStatistics::calculate(&swing_points);
        
        assert_eq!(stats.total_highs, 2);
        assert_eq!(stats.total_lows, 1);
        assert_relative_eq!(stats.avg_strength, 0.05, epsilon = 1e-6);
        assert_relative_eq!(stats.avg_time_between, 5.0, epsilon = 1e-6);
        assert_relative_eq!(stats.success_rate, 2.0/3.0, epsilon = 1e-6);
    }
}