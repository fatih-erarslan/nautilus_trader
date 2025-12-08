//! Configuration for Fibonacci analysis
//!
//! This module provides configuration structures and defaults for Fibonacci analysis
//! parameters including retracement levels, extension levels, and analysis settings.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

/// Configuration for Fibonacci analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FibonacciConfig {
    /// Fibonacci retracement levels as percentage ratios
    pub retracement_levels: HashMap<String, f64>,
    
    /// Fibonacci extension levels as ratio multipliers
    pub extension_levels: HashMap<String, f64>,
    
    /// Period for swing point detection
    pub swing_period: usize,
    
    /// Tolerance for alignment score calculation (as fraction of price)
    pub alignment_tolerance: f64,
    
    /// Base tolerance for regime score adaptation
    pub base_tolerance: f64,
    
    /// Maximum tolerance factor for regime adaptation
    pub max_tolerance_factor: f64,
    
    /// ATR period for volatility bands
    pub atr_period: usize,
    
    /// Hysteresis threshold for trend stability
    pub trend_hysteresis_threshold: f64,
    
    /// Enable SIMD acceleration
    pub enable_simd: bool,
    
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Cache size for analysis results
    pub cache_size: usize,
    
    /// Minimum data points required for analysis
    pub min_data_points: usize,
    
    /// Maximum lookback period for calculations
    pub max_lookback: usize,
}

impl Default for FibonacciConfig {
    fn default() -> Self {
        let mut retracement_levels = HashMap::new();
        retracement_levels.insert("0.0".to_string(), 0.0);
        retracement_levels.insert("23.6".to_string(), 0.236);
        retracement_levels.insert("38.2".to_string(), 0.382);
        retracement_levels.insert("50.0".to_string(), 0.5);
        retracement_levels.insert("61.8".to_string(), 0.618);
        retracement_levels.insert("78.6".to_string(), 0.786);
        retracement_levels.insert("100.0".to_string(), 1.0);
        
        let mut extension_levels = HashMap::new();
        extension_levels.insert("100.0".to_string(), 1.0);
        extension_levels.insert("127.2".to_string(), 1.272);
        extension_levels.insert("161.8".to_string(), 1.618);
        extension_levels.insert("261.8".to_string(), 2.618);
        extension_levels.insert("361.8".to_string(), 3.618);
        
        Self {
            retracement_levels,
            extension_levels,
            swing_period: 14,
            alignment_tolerance: 0.006,
            base_tolerance: 0.006,
            max_tolerance_factor: 2.0,
            atr_period: 14,
            trend_hysteresis_threshold: 0.01,
            enable_simd: true,
            enable_parallel: true,
            cache_size: 1000,
            min_data_points: 20,
            max_lookback: 1000,
        }
    }
}

impl FibonacciConfig {
    /// Create a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Create a configuration optimized for high-frequency trading
    pub fn high_frequency() -> Self {
        let mut config = Self::default();
        config.swing_period = 5;
        config.alignment_tolerance = 0.003;
        config.atr_period = 5;
        config.min_data_points = 10;
        config.max_lookback = 200;
        config
    }
    
    /// Create a configuration optimized for daily timeframes
    pub fn daily() -> Self {
        let mut config = Self::default();
        config.swing_period = 20;
        config.alignment_tolerance = 0.01;
        config.atr_period = 20;
        config.min_data_points = 50;
        config.max_lookback = 2000;
        config
    }
    
    /// Create a configuration with custom retracement levels
    pub fn with_custom_retracements(mut self, levels: HashMap<String, f64>) -> Self {
        self.retracement_levels = levels;
        self
    }
    
    /// Create a configuration with custom extension levels
    pub fn with_custom_extensions(mut self, levels: HashMap<String, f64>) -> Self {
        self.extension_levels = levels;
        self
    }
    
    /// Set swing period
    pub fn with_swing_period(mut self, period: usize) -> Self {
        self.swing_period = period;
        self
    }
    
    /// Set alignment tolerance
    pub fn with_alignment_tolerance(mut self, tolerance: f64) -> Self {
        self.alignment_tolerance = tolerance;
        self
    }
    
    /// Set ATR period
    pub fn with_atr_period(mut self, period: usize) -> Self {
        self.atr_period = period;
        self
    }
    
    /// Enable or disable SIMD acceleration
    pub fn with_simd(mut self, enable: bool) -> Self {
        self.enable_simd = enable;
        self
    }
    
    /// Enable or disable parallel processing
    pub fn with_parallel(mut self, enable: bool) -> Self {
        self.enable_parallel = enable;
        self
    }
    
    /// Set cache size
    pub fn with_cache_size(mut self, size: usize) -> Self {
        self.cache_size = size;
        self
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.swing_period == 0 {
            return Err("Swing period must be greater than 0".to_string());
        }
        
        if self.alignment_tolerance <= 0.0 || self.alignment_tolerance > 1.0 {
            return Err("Alignment tolerance must be between 0 and 1".to_string());
        }
        
        if self.base_tolerance <= 0.0 || self.base_tolerance > 1.0 {
            return Err("Base tolerance must be between 0 and 1".to_string());
        }
        
        if self.max_tolerance_factor <= 1.0 {
            return Err("Max tolerance factor must be greater than 1".to_string());
        }
        
        if self.atr_period == 0 {
            return Err("ATR period must be greater than 0".to_string());
        }
        
        if self.trend_hysteresis_threshold < 0.0 || self.trend_hysteresis_threshold > 1.0 {
            return Err("Trend hysteresis threshold must be between 0 and 1".to_string());
        }
        
        if self.cache_size == 0 {
            return Err("Cache size must be greater than 0".to_string());
        }
        
        if self.min_data_points == 0 {
            return Err("Minimum data points must be greater than 0".to_string());
        }
        
        if self.max_lookback == 0 {
            return Err("Maximum lookback must be greater than 0".to_string());
        }
        
        // Validate retracement levels
        for (name, &value) in &self.retracement_levels {
            if value < 0.0 || value > 1.0 {
                return Err(format!("Retracement level '{}' must be between 0 and 1, got {}", name, value));
            }
        }
        
        // Validate extension levels
        for (name, &value) in &self.extension_levels {
            if value < 1.0 || value > 10.0 {
                return Err(format!("Extension level '{}' must be between 1 and 10, got {}", name, value));
            }
        }
        
        Ok(())
    }
    
    /// Get the golden ratio (φ) - fundamental to Fibonacci analysis
    pub fn golden_ratio() -> f64 {
        (1.0 + 5.0_f64.sqrt()) / 2.0
    }
    
    /// Get the inverse golden ratio (1/φ)
    pub fn inverse_golden_ratio() -> f64 {
        1.0 / Self::golden_ratio()
    }
    
    /// Get common Fibonacci ratios derived from the golden ratio
    pub fn fibonacci_ratios() -> Vec<f64> {
        vec![
            0.236,  // φ^(-2)
            0.382,  // φ^(-1) - 1
            0.5,    // 1/2
            0.618,  // φ^(-1)
            0.786,  // sqrt(φ^(-1))
            1.0,    // 1
            1.272,  // sqrt(φ)
            1.618,  // φ
            2.618,  // φ^2
            3.618,  // φ^2 + 1
        ]
    }
    
    /// Create a minimal configuration for testing
    pub fn minimal() -> Self {
        let mut config = Self::default();
        config.swing_period = 3;
        config.atr_period = 3;
        config.min_data_points = 5;
        config.max_lookback = 50;
        config.cache_size = 10;
        config
    }
}

/// Preset configurations for different trading scenarios
pub struct FibonacciPresets;

impl FibonacciPresets {
    /// Scalping configuration (very short-term)
    pub fn scalping() -> FibonacciConfig {
        FibonacciConfig::default()
            .with_swing_period(3)
            .with_alignment_tolerance(0.002)
            .with_atr_period(3)
            .with_cache_size(500)
    }
    
    /// Day trading configuration
    pub fn day_trading() -> FibonacciConfig {
        FibonacciConfig::default()
            .with_swing_period(7)
            .with_alignment_tolerance(0.004)
            .with_atr_period(7)
            .with_cache_size(1000)
    }
    
    /// Swing trading configuration
    pub fn swing_trading() -> FibonacciConfig {
        FibonacciConfig::default()
            .with_swing_period(14)
            .with_alignment_tolerance(0.008)
            .with_atr_period(14)
            .with_cache_size(2000)
    }
    
    /// Position trading configuration (long-term)
    pub fn position_trading() -> FibonacciConfig {
        FibonacciConfig::default()
            .with_swing_period(30)
            .with_alignment_tolerance(0.015)
            .with_atr_period(30)
            .with_cache_size(5000)
    }
    
    /// High-precision configuration with more Fibonacci levels
    pub fn high_precision() -> FibonacciConfig {
        let mut retracement_levels = HashMap::new();
        retracement_levels.insert("0.0".to_string(), 0.0);
        retracement_levels.insert("14.6".to_string(), 0.146);
        retracement_levels.insert("23.6".to_string(), 0.236);
        retracement_levels.insert("38.2".to_string(), 0.382);
        retracement_levels.insert("50.0".to_string(), 0.5);
        retracement_levels.insert("61.8".to_string(), 0.618);
        retracement_levels.insert("70.7".to_string(), 0.707);
        retracement_levels.insert("78.6".to_string(), 0.786);
        retracement_levels.insert("85.4".to_string(), 0.854);
        retracement_levels.insert("100.0".to_string(), 1.0);
        
        let mut extension_levels = HashMap::new();
        extension_levels.insert("100.0".to_string(), 1.0);
        extension_levels.insert("127.2".to_string(), 1.272);
        extension_levels.insert("138.2".to_string(), 1.382);
        extension_levels.insert("161.8".to_string(), 1.618);
        extension_levels.insert("200.0".to_string(), 2.0);
        extension_levels.insert("261.8".to_string(), 2.618);
        extension_levels.insert("361.8".to_string(), 3.618);
        extension_levels.insert("423.6".to_string(), 4.236);
        
        FibonacciConfig::default()
            .with_custom_retracements(retracement_levels)
            .with_custom_extensions(extension_levels)
            .with_alignment_tolerance(0.003)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_config() {
        let config = FibonacciConfig::default();
        assert_eq!(config.swing_period, 14);
        assert_eq!(config.retracement_levels.len(), 7);
        assert_eq!(config.extension_levels.len(), 5);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_high_frequency_config() {
        let config = FibonacciConfig::high_frequency();
        assert_eq!(config.swing_period, 5);
        assert_eq!(config.alignment_tolerance, 0.003);
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = FibonacciConfig::default();
        
        // Test invalid swing period
        config.swing_period = 0;
        assert!(config.validate().is_err());
        
        // Test invalid alignment tolerance
        config = FibonacciConfig::default();
        config.alignment_tolerance = -0.1;
        assert!(config.validate().is_err());
        
        config.alignment_tolerance = 1.5;
        assert!(config.validate().is_err());
    }
    
    #[test]
    fn test_golden_ratio() {
        let phi = FibonacciConfig::golden_ratio();
        assert!((phi - 1.618033988749).abs() < 1e-10);
        
        let inv_phi = FibonacciConfig::inverse_golden_ratio();
        assert!((inv_phi - 0.618033988749).abs() < 1e-10);
    }
    
    #[test]
    fn test_fibonacci_ratios() {
        let ratios = FibonacciConfig::fibonacci_ratios();
        assert_eq!(ratios.len(), 10);
        assert!((ratios[3] - 0.618).abs() < 1e-3);
        assert!((ratios[7] - 1.618).abs() < 1e-3);
    }
    
    #[test]
    fn test_presets() {
        let scalping = FibonacciPresets::scalping();
        assert_eq!(scalping.swing_period, 3);
        assert!(scalping.validate().is_ok());
        
        let day_trading = FibonacciPresets::day_trading();
        assert_eq!(day_trading.swing_period, 7);
        assert!(day_trading.validate().is_ok());
        
        let high_precision = FibonacciPresets::high_precision();
        assert_eq!(high_precision.retracement_levels.len(), 10);
        assert_eq!(high_precision.extension_levels.len(), 8);
        assert!(high_precision.validate().is_ok());
    }
    
    #[test]
    fn test_config_builder() {
        let config = FibonacciConfig::default()
            .with_swing_period(10)
            .with_alignment_tolerance(0.01)
            .with_atr_period(20)
            .with_simd(false)
            .with_parallel(false)
            .with_cache_size(500);
        
        assert_eq!(config.swing_period, 10);
        assert_eq!(config.alignment_tolerance, 0.01);
        assert_eq!(config.atr_period, 20);
        assert!(!config.enable_simd);
        assert!(!config.enable_parallel);
        assert_eq!(config.cache_size, 500);
        assert!(config.validate().is_ok());
    }
}