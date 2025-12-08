//! Volatility analysis for Fibonacci levels

use wide::f64x4;
use std::collections::VecDeque;

/// Analyzer for volatility around Fibonacci levels
#[derive(Debug)]
pub struct VolatilityAnalyzer {
    window_size: usize,
    volatility_threshold: f64,
}

impl VolatilityAnalyzer {
    pub fn new(window_size: usize, volatility_threshold: f64) -> Self {
        Self {
            window_size,
            volatility_threshold,
        }
    }
    
    /// Calculate volatility around a price level
    pub fn calculate_level_volatility(&self, prices: &[f64], level: f64, range_pct: f64) -> f64 {
        let range = level * range_pct;
        let lower = level - range;
        let upper = level + range;
        
        let mut near_level_returns = Vec::new();
        
        for window in prices.windows(2) {
            let price = window[0];
            let next_price = window[1];
            
            if price >= lower && price <= upper {
                let return_val = (next_price - price) / price;
                near_level_returns.push(return_val);
            }
        }
        
        if near_level_returns.len() < 2 {
            return 0.0;
        }
        
        // Calculate standard deviation
        let mean = near_level_returns.iter().sum::<f64>() / near_level_returns.len() as f64;
        let variance = near_level_returns.iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>() / near_level_returns.len() as f64;
        
        variance.sqrt()
    }
    
    /// Analyze volatility expansion/contraction
    pub fn analyze_volatility_regime(&self, prices: &[f64]) -> VolatilityRegime {
        if prices.len() < self.window_size * 2 {
            return VolatilityRegime::Unknown;
        }
        
        let mut rolling_vol = VecDeque::new();
        
        // Calculate rolling volatility
        for i in self.window_size..prices.len() {
            let window = &prices[i - self.window_size..i];
            let returns: Vec<f64> = window.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
            
            if returns.len() > 1 {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                let vol = (returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64)
                    .sqrt();
                
                rolling_vol.push_back(vol);
                if rolling_vol.len() > self.window_size {
                    rolling_vol.pop_front();
                }
            }
        }
        
        if rolling_vol.len() < 2 {
            return VolatilityRegime::Unknown;
        }
        
        // Analyze trend in volatility
        let recent_vol: f64 = rolling_vol.iter().rev().take(5).sum::<f64>() / 5.0_f64.min(rolling_vol.len() as f64);
        let historical_vol: f64 = rolling_vol.iter().sum::<f64>() / rolling_vol.len() as f64;
        
        let vol_ratio = recent_vol / historical_vol;
        
        if vol_ratio > 1.5 {
            VolatilityRegime::Expanding
        } else if vol_ratio < 0.7 {
            VolatilityRegime::Contracting
        } else {
            VolatilityRegime::Stable
        }
    }
    
    /// Check if volatility is clustered around Fibonacci levels
    pub fn check_level_clustering(&self, prices: &[f64], fib_levels: &[f64]) -> Vec<(f64, f64)> {
        let mut clustered_levels = Vec::new();
        
        for &level in fib_levels {
            let vol = self.calculate_level_volatility(prices, level, 0.01); // 1% range
            if vol > self.volatility_threshold {
                clustered_levels.push((level, vol));
            }
        }
        
        // Sort by volatility (descending)
        clustered_levels.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        clustered_levels
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VolatilityRegime {
    Expanding,
    Contracting,
    Stable,
    Unknown,
}

impl VolatilityAnalyzer {
    /// Calculate volatility bands for price levels
    pub fn calculate_bands(&self, prices: &[f64], _volumes: &[f64]) -> Result<crate::core::VolatilityBands, crate::FibonacciError> {
        if prices.len() < self.window_size {
            return Err(crate::FibonacciError::InvalidInput("Insufficient data for volatility bands".to_string()));
        }
        
        // Calculate ATR-based bands
        let atr = self.calculate_atr(prices);
        let current_price = prices.last().unwrap();
        
        let mut bands = std::collections::HashMap::new();
        
        // Create bands at different ATR multiples
        for multiplier in &[0.5, 1.0, 1.5, 2.0, 3.0] {
            let band_width = atr * multiplier;
            bands.insert(
                format!("atr_{}", multiplier),
                (current_price + band_width, current_price - band_width)
            );
        }
        
        // Calculate normalized volatility
        let volatility = if prices.len() >= 2 {
            let returns: Vec<f64> = prices.windows(2)
                .map(|w| (w[1] - w[0]) / w[0])
                .collect();
            
            if !returns.is_empty() {
                let mean = returns.iter().sum::<f64>() / returns.len() as f64;
                (returns.iter()
                    .map(|r| (r - mean).powi(2))
                    .sum::<f64>() / returns.len() as f64)
                    .sqrt()
            } else {
                0.0
            }
        } else {
            0.0
        };
        
        Ok(crate::core::VolatilityBands::new(
            bands,
            *current_price,
            atr,
            volatility / atr.max(0.0001), // Normalize by ATR
        ))
    }
    
    /// Calculate Average True Range
    fn calculate_atr(&self, prices: &[f64]) -> f64 {
        if prices.len() < 2 {
            return 0.0;
        }
        
        let mut true_ranges = Vec::new();
        
        for i in 1..prices.len() {
            let high = prices[i].max(prices[i-1]);
            let low = prices[i].min(prices[i-1]);
            let true_range = high - low;
            true_ranges.push(true_range);
        }
        
        // Calculate average
        if true_ranges.len() >= self.window_size {
            let recent_ranges = &true_ranges[true_ranges.len() - self.window_size..];
            recent_ranges.iter().sum::<f64>() / self.window_size as f64
        } else {
            true_ranges.iter().sum::<f64>() / true_ranges.len() as f64
        }
    }
}

/// Individual volatility band
pub struct VolatilityBand {
    pub upper: f64,
    pub lower: f64,
    pub atr_multiple: f64,
}

impl Default for VolatilityAnalyzer {
    fn default() -> Self {
        Self::new(20, 0.02) // 20 periods, 2% volatility threshold
    }
}