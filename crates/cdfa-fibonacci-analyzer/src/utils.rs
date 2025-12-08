//! Utility functions for Fibonacci analysis

use crate::types::*;

/// Convert Fibonacci level to percentage
pub fn level_to_percentage(level: f64) -> f64 {
    level * 100.0
}

/// Calculate price from Fibonacci level
pub fn calculate_price_from_level(high: f64, low: f64, level: f64) -> f64 {
    high - (high - low) * level
}

/// Calculate Fibonacci level from price
pub fn calculate_level_from_price(high: f64, low: f64, price: f64) -> f64 {
    if (high - low).abs() < 1e-10 {
        return 0.5; // Default to 50% if range is too small
    }
    (high - price) / (high - low)
}

/// Validate price data
pub fn validate_prices(prices: &[f64]) -> Result<(), String> {
    if prices.is_empty() {
        return Err("Price data is empty".to_string());
    }
    
    if prices.iter().any(|&p| p <= 0.0 || !p.is_finite()) {
        return Err("Invalid price data: prices must be positive and finite".to_string());
    }
    
    Ok(())
}

/// Find swing highs and lows
pub fn find_swings(prices: &[f64], lookback: usize) -> (Vec<(usize, f64)>, Vec<(usize, f64)>) {
    let mut highs = Vec::new();
    let mut lows = Vec::new();
    
    if prices.len() < lookback * 2 + 1 {
        return (highs, lows);
    }
    
    for i in lookback..(prices.len() - lookback) {
        let mut is_high = true;
        let mut is_low = true;
        
        for j in (i - lookback)..=(i + lookback) {
            if j != i {
                if prices[j] >= prices[i] {
                    is_high = false;
                }
                if prices[j] <= prices[i] {
                    is_low = false;
                }
            }
        }
        
        if is_high {
            highs.push((i, prices[i]));
        }
        if is_low {
            lows.push((i, prices[i]));
        }
    }
    
    (highs, lows)
}

/// Calculate confidence score based on multiple factors
pub fn calculate_confidence(
    price_distance: f64,
    volume_confirmation: f64,
    time_consistency: f64,
    pattern_strength: f64,
) -> f64 {
    // Weighted average of factors
    let weights = [0.3, 0.2, 0.2, 0.3];
    let factors = [
        1.0 - price_distance.min(1.0),
        volume_confirmation,
        time_consistency,
        pattern_strength,
    ];
    
    let confidence: f64 = factors.iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum();
    
    confidence.clamp(0.0, 1.0)
}

/// Format level for display
pub fn format_level(level: f64, price: f64) -> String {
    format!("{:.1}% (${:.2})", level * 100.0, price)
}

/// Calculate golden ratio powers
pub fn golden_powers(n: usize) -> Vec<f64> {
    let phi = 1.618033988749895;
    let mut powers = Vec::with_capacity(n);
    let mut current = 1.0;
    
    for _ in 0..n {
        powers.push(current);
        current *= phi;
    }
    
    powers
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_level_calculations() {
        let high = 100.0;
        let low = 50.0;
        
        // Test 50% level
        let price_50 = calculate_price_from_level(high, low, 0.5);
        assert!((price_50 - 75.0).abs() < 1e-10);
        
        // Test inverse calculation
        let level = calculate_level_from_price(high, low, price_50);
        assert!((level - 0.5).abs() < 1e-10);
    }
    
    #[test]
    fn test_swing_detection() {
        let prices = vec![1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0];
        let (highs, lows) = find_swings(&prices, 2);
        
        assert_eq!(highs.len(), 2);
        assert_eq!(lows.len(), 2);
        assert_eq!(highs[0].1, 3.0);
        assert_eq!(lows[0].1, 1.0);
    }
}