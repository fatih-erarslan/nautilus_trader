//! Core data structures and types for Fibonacci analysis
//!
//! This module defines the fundamental types used throughout the Fibonacci analyzer,
//! including swing points, retracement levels, extension levels, and trend analysis.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use crate::FibonacciError;

/// Trend direction enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Up,
    Down,
    Sideways,
    Unknown,
}

impl TrendDirection {
    /// Convert trend to numeric value for calculations
    pub fn as_numeric(&self) -> f64 {
        match self {
            TrendDirection::Up => 1.0,
            TrendDirection::Down => -1.0,
            TrendDirection::Sideways => 0.0,
            TrendDirection::Unknown => 0.0,
        }
    }
    
    /// Get trend from numeric value
    pub fn from_numeric(value: f64) -> Self {
        if value > 0.1 {
            TrendDirection::Up
        } else if value < -0.1 {
            TrendDirection::Down
        } else {
            TrendDirection::Sideways
        }
    }
}

/// Swing point structure representing a high or low point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwingPoint {
    pub index: usize,
    pub price: f64,
    pub timestamp: Option<u64>,
}

impl SwingPoint {
    pub fn new(index: usize, price: f64) -> Self {
        Self {
            index,
            price,
            timestamp: None,
        }
    }
    
    pub fn with_timestamp(index: usize, price: f64, timestamp: u64) -> Self {
        Self {
            index,
            price,
            timestamp: Some(timestamp),
        }
    }
}

/// Collection of swing points
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwingPoints {
    pub swing_highs: Vec<SwingPoint>,
    pub swing_lows: Vec<SwingPoint>,
    pub detection_period: usize,
}

impl SwingPoints {
    pub fn new(detection_period: usize) -> Self {
        Self {
            swing_highs: Vec::new(),
            swing_lows: Vec::new(),
            detection_period,
        }
    }
    
    pub fn add_high(&mut self, point: SwingPoint) {
        self.swing_highs.push(point);
    }
    
    pub fn add_low(&mut self, point: SwingPoint) {
        self.swing_lows.push(point);
    }
    
    pub fn is_empty(&self) -> bool {
        self.swing_highs.is_empty() && self.swing_lows.is_empty()
    }
    
    pub fn total_points(&self) -> usize {
        self.swing_highs.len() + self.swing_lows.len()
    }
    
    /// Get the most recent swing high
    pub fn last_high(&self) -> Option<&SwingPoint> {
        self.swing_highs.last()
    }
    
    /// Get the most recent swing low
    pub fn last_low(&self) -> Option<&SwingPoint> {
        self.swing_lows.last()
    }
    
    /// Get the most recent swing point (either high or low)
    pub fn last_swing(&self) -> Option<&SwingPoint> {
        let last_high = self.last_high();
        let last_low = self.last_low();
        
        match (last_high, last_low) {
            (Some(high), Some(low)) => {
                if high.index > low.index {
                    Some(high)
                } else {
                    Some(low)
                }
            }
            (Some(high), None) => Some(high),
            (None, Some(low)) => Some(low),
            (None, None) => None,
        }
    }
    
    /// Clear all swing points
    pub fn clear(&mut self) {
        self.swing_highs.clear();
        self.swing_lows.clear();
    }
}

/// Fibonacci retracement levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetracementLevels {
    pub levels: HashMap<String, f64>,
    pub trend: TrendDirection,
    pub high_price: f64,
    pub low_price: f64,
    pub price_range: f64,
}

impl RetracementLevels {
    pub fn new(
        levels: HashMap<String, f64>,
        trend: TrendDirection,
        high_price: f64,
        low_price: f64,
    ) -> Self {
        let price_range = high_price - low_price;
        Self {
            levels,
            trend,
            high_price,
            low_price,
            price_range,
        }
    }
    
    pub fn empty() -> Self {
        Self {
            levels: HashMap::new(),
            trend: TrendDirection::Unknown,
            high_price: 0.0,
            low_price: 0.0,
            price_range: 0.0,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
    
    /// Get level by name
    pub fn get_level(&self, name: &str) -> Option<f64> {
        self.levels.get(name).copied()
    }
    
    /// Get all level values as a vector
    pub fn level_values(&self) -> Vec<f64> {
        self.levels.values().copied().collect()
    }
    
    /// Get level closest to the given price
    pub fn closest_level(&self, price: f64) -> Option<(String, f64)> {
        let mut closest_name = String::new();
        let mut closest_price = 0.0;
        let mut min_distance = f64::INFINITY;
        
        for (name, &level_price) in &self.levels {
            let distance = (price - level_price).abs();
            if distance < min_distance {
                min_distance = distance;
                closest_name = name.clone();
                closest_price = level_price;
            }
        }
        
        if min_distance < f64::INFINITY {
            Some((closest_name, closest_price))
        } else {
            None
        }
    }
    
    /// Calculate distance to nearest level as percentage of price
    pub fn distance_to_nearest_level(&self, price: f64) -> f64 {
        if let Some((_, level_price)) = self.closest_level(price) {
            (price - level_price).abs() / price
        } else {
            1.0 // No levels available
        }
    }
}

/// Fibonacci extension levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtensionLevels {
    pub levels: HashMap<String, f64>,
    pub trend: TrendDirection,
    pub base_price: f64,
    pub target_price: f64,
    pub projection_range: f64,
}

impl ExtensionLevels {
    pub fn new(
        levels: HashMap<String, f64>,
        trend: TrendDirection,
        base_price: f64,
        target_price: f64,
    ) -> Self {
        let projection_range = (target_price - base_price).abs();
        Self {
            levels,
            trend,
            base_price,
            target_price,
            projection_range,
        }
    }
    
    pub fn empty() -> Self {
        Self {
            levels: HashMap::new(),
            trend: TrendDirection::Unknown,
            base_price: 0.0,
            target_price: 0.0,
            projection_range: 0.0,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }
    
    /// Get level by name
    pub fn get_level(&self, name: &str) -> Option<f64> {
        self.levels.get(name).copied()
    }
    
    /// Get all level values as a vector
    pub fn level_values(&self) -> Vec<f64> {
        self.levels.values().copied().collect()
    }
    
    /// Get level closest to the given price
    pub fn closest_level(&self, price: f64) -> Option<(String, f64)> {
        let mut closest_name = String::new();
        let mut closest_price = 0.0;
        let mut min_distance = f64::INFINITY;
        
        for (name, &level_price) in &self.levels {
            let distance = (price - level_price).abs();
            if distance < min_distance {
                min_distance = distance;
                closest_name = name.clone();
                closest_price = level_price;
            }
        }
        
        if min_distance < f64::INFINITY {
            Some((closest_name, closest_price))
        } else {
            None
        }
    }
    
    /// Calculate distance to nearest level as percentage of price
    pub fn distance_to_nearest_level(&self, price: f64) -> f64 {
        if let Some((_, level_price)) = self.closest_level(price) {
            (price - level_price).abs() / price
        } else {
            1.0 // No levels available
        }
    }
}

/// Volatility bands based on ATR and Fibonacci ratios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityBands {
    pub bands: HashMap<String, (f64, f64)>, // (upper, lower) for each level
    pub base_price: f64,
    pub atr_value: f64,
    pub normalized_volatility: f64,
}

impl VolatilityBands {
    pub fn new(
        bands: HashMap<String, (f64, f64)>,
        base_price: f64,
        atr_value: f64,
        normalized_volatility: f64,
    ) -> Self {
        Self {
            bands,
            base_price,
            atr_value,
            normalized_volatility,
        }
    }
    
    pub fn empty() -> Self {
        Self {
            bands: HashMap::new(),
            base_price: 0.0,
            atr_value: 0.0,
            normalized_volatility: 0.0,
        }
    }
    
    pub fn is_empty(&self) -> bool {
        self.bands.is_empty()
    }
    
    /// Get band by name
    pub fn get_band(&self, name: &str) -> Option<(f64, f64)> {
        self.bands.get(name).copied()
    }
    
    /// Check if price is within a specific band
    pub fn is_price_in_band(&self, price: f64, band_name: &str) -> bool {
        if let Some((upper, lower)) = self.get_band(band_name) {
            price >= lower && price <= upper
        } else {
            false
        }
    }
    
    /// Find which band contains the given price
    pub fn find_containing_band(&self, price: f64) -> Option<String> {
        for (name, &(upper, lower)) in &self.bands {
            if price >= lower && price <= upper {
                return Some(name.clone());
            }
        }
        None
    }
}

/// Fibonacci confluence zone - area where multiple levels converge
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfluenceZone {
    pub center_price: f64,
    pub width: f64,
    pub strength: f64, // Number of levels in the zone
    pub participating_levels: Vec<String>,
}

impl ConfluenceZone {
    pub fn new(center_price: f64, width: f64, strength: f64) -> Self {
        Self {
            center_price,
            width,
            strength,
            participating_levels: Vec::new(),
        }
    }
    
    pub fn add_level(&mut self, level_name: String) {
        self.participating_levels.push(level_name);
    }
    
    pub fn contains_price(&self, price: f64) -> bool {
        let half_width = self.width / 2.0;
        price >= (self.center_price - half_width) && price <= (self.center_price + half_width)
    }
    
    pub fn distance_to_price(&self, price: f64) -> f64 {
        (price - self.center_price).abs()
    }
    
    pub fn relative_distance_to_price(&self, price: f64) -> f64 {
        self.distance_to_price(price) / self.center_price
    }
}

/// Complete Fibonacci analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciAnalysis {
    pub swing_points: SwingPoints,
    pub retracements: RetracementLevels,
    pub extensions: ExtensionLevels,
    pub volatility_bands: VolatilityBands,
    pub confluence_zones: Vec<ConfluenceZone>,
    pub current_price: f64,
    pub alignment_score: f64,
    pub trend_strength: f64,
    pub analysis_timestamp: u64,
}

impl FibonacciAnalysis {
    pub fn new(
        swing_points: SwingPoints,
        retracements: RetracementLevels,
        extensions: ExtensionLevels,
        volatility_bands: VolatilityBands,
        current_price: f64,
        alignment_score: f64,
    ) -> Self {
        Self {
            swing_points,
            retracements,
            extensions,
            volatility_bands,
            confluence_zones: Vec::new(),
            current_price,
            alignment_score,
            trend_strength: 0.0,
            analysis_timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        }
    }
    
    pub fn add_confluence_zone(&mut self, zone: ConfluenceZone) {
        self.confluence_zones.push(zone);
    }
    
    pub fn set_trend_strength(&mut self, strength: f64) {
        self.trend_strength = strength;
    }
    
    /// Get the strongest confluence zone
    pub fn strongest_confluence_zone(&self) -> Option<&ConfluenceZone> {
        self.confluence_zones
            .iter()
            .max_by(|a, b| a.strength.partial_cmp(&b.strength).unwrap_or(std::cmp::Ordering::Equal))
    }
    
    /// Check if current price is near a significant level
    pub fn is_near_significant_level(&self, tolerance: f64) -> bool {
        let retr_distance = self.retracements.distance_to_nearest_level(self.current_price);
        let ext_distance = self.extensions.distance_to_nearest_level(self.current_price);
        
        retr_distance <= tolerance || ext_distance <= tolerance
    }
    
    /// Get summary statistics
    pub fn summary(&self) -> HashMap<String, f64> {
        let mut summary = HashMap::new();
        
        summary.insert("current_price".to_string(), self.current_price);
        summary.insert("alignment_score".to_string(), self.alignment_score);
        summary.insert("trend_strength".to_string(), self.trend_strength);
        summary.insert("swing_highs_count".to_string(), self.swing_points.swing_highs.len() as f64);
        summary.insert("swing_lows_count".to_string(), self.swing_points.swing_lows.len() as f64);
        summary.insert("retracement_levels_count".to_string(), self.retracements.levels.len() as f64);
        summary.insert("extension_levels_count".to_string(), self.extensions.levels.len() as f64);
        summary.insert("confluence_zones_count".to_string(), self.confluence_zones.len() as f64);
        summary.insert("volatility_bands_count".to_string(), self.volatility_bands.bands.len() as f64);
        
        if let Some(strongest_zone) = self.strongest_confluence_zone() {
            summary.insert("strongest_confluence_strength".to_string(), strongest_zone.strength);
            summary.insert("strongest_confluence_distance".to_string(), 
                          strongest_zone.relative_distance_to_price(self.current_price));
        }
        
        summary
    }
}

/// Utility functions for Fibonacci calculations
pub mod utils {
    use super::*;
    
    /// Calculate Fibonacci number at position n
    pub fn fibonacci_number(n: usize) -> u64 {
        if n <= 1 {
            return n as u64;
        }
        
        let mut a = 0u64;
        let mut b = 1u64;
        
        for _ in 2..=n {
            let temp = a + b;
            a = b;
            b = temp;
        }
        
        b
    }
    
    /// Calculate Fibonacci ratio between consecutive numbers
    pub fn fibonacci_ratio(n: usize) -> f64 {
        if n <= 1 {
            return 1.0;
        }
        
        let fib_n = fibonacci_number(n) as f64;
        let fib_n_minus_1 = fibonacci_number(n - 1) as f64;
        
        if fib_n_minus_1 == 0.0 {
            return 1.0;
        }
        
        fib_n / fib_n_minus_1
    }
    
    /// Generate Fibonacci ratios up to a certain position
    pub fn fibonacci_ratios(max_n: usize) -> Vec<f64> {
        (1..=max_n).map(fibonacci_ratio).collect()
    }
    
    /// Check if a value is close to a Fibonacci ratio
    pub fn is_fibonacci_ratio(value: f64, tolerance: f64) -> bool {
        let common_ratios = [0.236, 0.382, 0.5, 0.618, 0.786, 1.0, 1.272, 1.618, 2.618, 3.618];
        
        for &ratio in &common_ratios {
            if (value - ratio).abs() <= tolerance {
                return true;
            }
        }
        
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_trend_direction() {
        assert_eq!(TrendDirection::Up.as_numeric(), 1.0);
        assert_eq!(TrendDirection::Down.as_numeric(), -1.0);
        assert_eq!(TrendDirection::Sideways.as_numeric(), 0.0);
        
        assert_eq!(TrendDirection::from_numeric(0.5), TrendDirection::Up);
        assert_eq!(TrendDirection::from_numeric(-0.5), TrendDirection::Down);
        assert_eq!(TrendDirection::from_numeric(0.05), TrendDirection::Sideways);
    }
    
    #[test]
    fn test_swing_points() {
        let mut swing_points = SwingPoints::new(14);
        assert!(swing_points.is_empty());
        
        swing_points.add_high(SwingPoint::new(5, 105.0));
        swing_points.add_low(SwingPoint::new(10, 95.0));
        
        assert_eq!(swing_points.total_points(), 2);
        assert!(!swing_points.is_empty());
        
        let last_swing = swing_points.last_swing().unwrap();
        assert_eq!(last_swing.index, 10);
        assert_eq!(last_swing.price, 95.0);
    }
    
    #[test]
    fn test_retracement_levels() {
        let mut levels = HashMap::new();
        levels.insert("50.0".to_string(), 100.0);
        levels.insert("61.8".to_string(), 95.0);
        
        let retracements = RetracementLevels::new(levels, TrendDirection::Up, 110.0, 90.0);
        
        assert_eq!(retracements.price_range, 20.0);
        assert_eq!(retracements.get_level("50.0"), Some(100.0));
        
        let (name, price) = retracements.closest_level(98.0).unwrap();
        assert_eq!(name, "50.0");
        assert_eq!(price, 100.0);
        
        let distance = retracements.distance_to_nearest_level(98.0);
        assert_relative_eq!(distance, 2.0 / 98.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_confluence_zone() {
        let mut zone = ConfluenceZone::new(100.0, 2.0, 3.0);
        zone.add_level("50.0".to_string());
        zone.add_level("61.8".to_string());
        
        assert!(zone.contains_price(99.5));
        assert!(zone.contains_price(100.5));
        assert!(!zone.contains_price(98.5));
        
        assert_eq!(zone.distance_to_price(102.0), 2.0);
        assert_eq!(zone.participating_levels.len(), 2);
    }
    
    #[test]
    fn test_fibonacci_utils() {
        use utils::*;
        
        assert_eq!(fibonacci_number(0), 0);
        assert_eq!(fibonacci_number(1), 1);
        assert_eq!(fibonacci_number(5), 5);
        assert_eq!(fibonacci_number(10), 55);
        
        let ratio = fibonacci_ratio(10);
        assert_relative_eq!(ratio, 1.618, epsilon = 0.01);
        
        assert!(is_fibonacci_ratio(0.618, 0.01));
        assert!(is_fibonacci_ratio(1.618, 0.01));
        assert!(!is_fibonacci_ratio(0.123, 0.01));
    }
}