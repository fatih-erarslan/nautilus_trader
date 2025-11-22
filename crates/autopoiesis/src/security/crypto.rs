//! Cryptographically secure implementations to replace weak randomness
//! 
//! This module provides secure alternatives to the standard `rand` crate
//! for all cryptographic and financial operations that require unpredictable randomness.

use anyhow::{anyhow, Result};
use ring::rand::{SecureRandom, SystemRandom};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::f64;

/// Cryptographically secure random number generator
/// 
/// This replaces all usage of the `rand` crate in financial and security-critical contexts
/// with cryptographically secure alternatives using the `ring` crate's SystemRandom.
pub struct SecureRng {
    rng: SystemRandom,
}

impl SecureRng {
    /// Create a new cryptographically secure RNG
    pub fn new() -> Self {
        Self {
            rng: SystemRandom::new(),
        }
    }
    
    /// Generate secure random bytes
    pub fn fill_bytes(&self, dest: &mut [u8]) -> Result<()> {
        self.rng.fill(dest)
            .map_err(|_| anyhow!("Failed to generate secure random bytes"))
    }
    
    /// Generate a secure random u32
    pub fn next_u32(&self) -> Result<u32> {
        let mut bytes = [0u8; 4];
        self.fill_bytes(&mut bytes)?;
        Ok(u32::from_le_bytes(bytes))
    }
    
    /// Generate a secure random u64
    pub fn next_u64(&self) -> Result<u64> {
        let mut bytes = [0u8; 8];
        self.fill_bytes(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
    
    /// Generate a secure random f64 in range [0.0, 1.0)
    pub fn next_f64(&self) -> Result<f64> {
        let value = self.next_u64()?;
        // Convert to f64 in range [0.0, 1.0) with proper precision
        Ok((value >> 11) as f64 * (1.0 / (1u64 << 53) as f64))
    }
    
    /// Generate a secure random f64 in range [min, max)
    pub fn next_f64_range(&self, min: f64, max: f64) -> Result<f64> {
        if min >= max {
            return Err(anyhow!("Invalid range: min must be less than max"));
        }
        let random = self.next_f64()?;
        Ok(min + random * (max - min))
    }
    
    /// Generate a secure random integer in range [min, max]
    pub fn next_u32_range(&self, min: u32, max: u32) -> Result<u32> {
        if min > max {
            return Err(anyhow!("Invalid range: min must be <= max"));
        }
        if min == max {
            return Ok(min);
        }
        
        let range = max - min + 1;
        let limit = u32::MAX - (u32::MAX % range);
        
        loop {
            let value = self.next_u32()?;
            if value < limit {
                return Ok(min + (value % range));
            }
        }
    }
    
    /// Generate secure random Decimal for financial calculations
    pub fn next_decimal(&self) -> Result<Decimal> {
        let value = self.next_f64()?;
        Decimal::try_from(value)
            .map_err(|e| anyhow!("Failed to convert random value to Decimal: {}", e))
    }
    
    /// Generate secure random Decimal in range [min, max)
    pub fn next_decimal_range(&self, min: Decimal, max: Decimal) -> Result<Decimal> {
        if min >= max {
            return Err(anyhow!("Invalid range: min must be less than max"));
        }
        
        let random = self.next_f64()?;
        let min_f64 = min.to_f64().ok_or_else(|| anyhow!("Failed to convert min to f64"))?;
        let max_f64 = max.to_f64().ok_or_else(|| anyhow!("Failed to convert max to f64"))?;
        
        let result = min_f64 + random * (max_f64 - min_f64);
        Decimal::try_from(result)
            .map_err(|e| anyhow!("Failed to convert result to Decimal: {}", e))
    }
    
    /// Generate secure market volatility with proper statistical distribution
    pub fn next_market_volatility(&self, base_volatility: f64, max_deviation: f64) -> Result<f64> {
        // Use Box-Muller transform for normal distribution
        let u1 = self.next_f64()?;
        let u2 = self.next_f64()?;
        
        // Ensure u1 is not zero to avoid log(0)
        let u1 = if u1 == 0.0 { f64::EPSILON } else { u1 };
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * f64::consts::PI * u2).cos();
        let volatility = base_volatility + (z * max_deviation);
        
        // Ensure volatility is positive
        Ok(volatility.max(0.001))
    }
    
    /// Generate secure price change with realistic market dynamics
    pub fn next_price_change(&self, base_change: f64, volatility: f64) -> Result<f64> {
        // Generate normally distributed price change
        let u1 = self.next_f64()?;
        let u2 = self.next_f64()?;
        
        let u1 = if u1 == 0.0 { f64::EPSILON } else { u1 };
        
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * f64::consts::PI * u2).sin();
        Ok(base_change + (z * volatility))
    }
    
    /// Generate secure trading signal with controlled randomness
    pub fn next_trading_signal(&self, signal_strength: f64) -> Result<TradingSignal> {
        let random = self.next_f64()?;
        
        // Apply signal strength bias
        let biased_random = if signal_strength > 0.0 {
            random + signal_strength * 0.3 // Slight bias towards positive
        } else if signal_strength < 0.0 {
            random + signal_strength * 0.3 // Slight bias towards negative
        } else {
            random // Neutral
        };
        
        match biased_random {
            x if x < 0.33 => Ok(TradingSignal::Sell),
            x if x > 0.67 => Ok(TradingSignal::Buy),
            _ => Ok(TradingSignal::Hold),
        }
    }
}

impl Default for SecureRng {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradingSignal {
    Buy,
    Sell,
    Hold,
}

/// Secure alternatives for common random operations in trading
pub struct SecureTrading {
    rng: SecureRng,
}

impl SecureTrading {
    pub fn new() -> Self {
        Self {
            rng: SecureRng::new(),
        }
    }
    
    /// Generate secure market data simulation (for testing only)
    pub fn generate_secure_market_data(&self, 
        base_price: Decimal, 
        volatility: f64, 
        trend: f64
    ) -> Result<MarketDataPoint> {
        let price_change = self.rng.next_price_change(trend, volatility)?;
        let volume_multiplier = self.rng.next_f64_range(0.5, 2.0)?;
        
        let new_price = base_price + Decimal::try_from(price_change)?;
        let base_volume = 1000000; // Base volume
        let volume = (base_volume as f64 * volume_multiplier) as u64;
        
        Ok(MarketDataPoint {
            price: new_price,
            volume,
            timestamp: chrono::Utc::now(),
            volatility,
        })
    }
    
    /// Generate secure portfolio weights with proper constraints
    pub fn generate_secure_portfolio_weights(&self, num_assets: usize) -> Result<Vec<f64>> {
        if num_assets == 0 {
            return Err(anyhow!("Number of assets must be greater than 0"));
        }
        
        let mut weights = Vec::with_capacity(num_assets);
        let mut sum = 0.0;
        
        // Generate random weights
        for _ in 0..num_assets {
            let weight = self.rng.next_f64()?;
            weights.push(weight);
            sum += weight;
        }
        
        // Normalize to sum to 1.0
        for weight in &mut weights {
            *weight /= sum;
        }
        
        Ok(weights)
    }
    
    /// Generate secure order timing with anti-pattern measures
    pub fn generate_secure_order_timing(&self, base_interval_ms: u64) -> Result<u64> {
        // Add jitter to prevent predictable timing patterns
        let jitter_factor = self.rng.next_f64_range(0.8, 1.2)?;
        let timing = (base_interval_ms as f64 * jitter_factor) as u64;
        
        // Ensure minimum timing for security
        Ok(timing.max(100)) // Minimum 100ms
    }
    
    /// Generate secure API key with proper entropy
    pub fn generate_secure_api_key(&self, length: usize) -> Result<String> {
        const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        
        let mut key = Vec::with_capacity(length);
        for _ in 0..length {
            let index = self.rng.next_u32_range(0, CHARSET.len() as u32 - 1)? as usize;
            key.push(CHARSET[index]);
        }
        
        String::from_utf8(key)
            .map_err(|e| anyhow!("Failed to create valid UTF-8 API key: {}", e))
    }
    
    /// Generate secure session ID
    pub fn generate_secure_session_id(&self) -> Result<String> {
        let mut bytes = [0u8; 32];
        self.rng.fill_bytes(&mut bytes)?;
        Ok(hex::encode(bytes))
    }
    
    /// Generate secure nonce for cryptographic operations
    pub fn generate_secure_nonce(&self, length: usize) -> Result<Vec<u8>> {
        let mut nonce = vec![0u8; length];
        self.rng.fill_bytes(&mut nonce)?;
        Ok(nonce)
    }
}

impl Default for SecureTrading {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataPoint {
    pub price: Decimal,
    pub volume: u64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub volatility: f64,
}

/// Global secure RNG instance for the application
use std::sync::OnceLock;

static GLOBAL_SECURE_RNG: OnceLock<SecureRng> = OnceLock::new();

/// Get the global secure RNG instance
pub fn global_secure_rng() -> &'static SecureRng {
    GLOBAL_SECURE_RNG.get_or_init(|| SecureRng::new())
}

/// Convenience functions for common secure random operations
pub mod secure_random {
    use super::*;
    
    /// Generate secure random f64 in range [0.0, 1.0)
    pub fn f64() -> Result<f64> {
        global_secure_rng().next_f64()
    }
    
    /// Generate secure random f64 in range [min, max)
    pub fn f64_range(min: f64, max: f64) -> Result<f64> {
        global_secure_rng().next_f64_range(min, max)
    }
    
    /// Generate secure random u32
    pub fn u32() -> Result<u32> {
        global_secure_rng().next_u32()
    }
    
    /// Generate secure random u32 in range [min, max]
    pub fn u32_range(min: u32, max: u32) -> Result<u32> {
        global_secure_rng().next_u32_range(min, max)
    }
    
    /// Generate secure random Decimal
    pub fn decimal() -> Result<Decimal> {
        global_secure_rng().next_decimal()
    }
    
    /// Generate secure random Decimal in range [min, max)
    pub fn decimal_range(min: Decimal, max: Decimal) -> Result<Decimal> {
        global_secure_rng().next_decimal_range(min, max)
    }
    
    /// Generate secure market volatility
    pub fn market_volatility(base: f64, deviation: f64) -> Result<f64> {
        global_secure_rng().next_market_volatility(base, deviation)
    }
    
    /// Generate secure price change
    pub fn price_change(base: f64, volatility: f64) -> Result<f64> {
        global_secure_rng().next_price_change(base, volatility)
    }
    
    /// Generate secure trading signal
    pub fn trading_signal(strength: f64) -> Result<TradingSignal> {
        global_secure_rng().next_trading_signal(strength)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_secure_rng_basic() {
        let rng = SecureRng::new();
        
        // Test basic random generation
        assert!(rng.next_u32().is_ok());
        assert!(rng.next_u64().is_ok());
        assert!(rng.next_f64().is_ok());
        
        let f64_val = rng.next_f64().unwrap();
        assert!(f64_val >= 0.0 && f64_val < 1.0);
    }
    
    #[test]
    fn test_secure_rng_ranges() {
        let rng = SecureRng::new();
        
        // Test range generation
        let val = rng.next_f64_range(10.0, 20.0).unwrap();
        assert!(val >= 10.0 && val < 20.0);
        
        let int_val = rng.next_u32_range(100, 200).unwrap();
        assert!(int_val >= 100 && int_val <= 200);
    }
    
    #[test]
    fn test_secure_trading() {
        let trading = SecureTrading::new();
        
        // Test API key generation
        let api_key = trading.generate_secure_api_key(32).unwrap();
        assert_eq!(api_key.len(), 32);
        
        // Test session ID generation
        let session_id = trading.generate_secure_session_id().unwrap();
        assert_eq!(session_id.len(), 64); // 32 bytes * 2 for hex encoding
        
        // Test portfolio weights
        let weights = trading.generate_secure_portfolio_weights(5).unwrap();
        assert_eq!(weights.len(), 5);
        
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 0.0001); // Should sum to 1.0
    }
    
    #[test]
    fn test_secure_random_convenience() {
        // Test convenience functions
        assert!(secure_random::f64().is_ok());
        assert!(secure_random::u32().is_ok());
        assert!(secure_random::decimal().is_ok());
        
        let val = secure_random::f64_range(5.0, 10.0).unwrap();
        assert!(val >= 5.0 && val < 10.0);
        
        let signal = secure_random::trading_signal(0.5).unwrap();
        assert!(matches!(signal, TradingSignal::Buy | TradingSignal::Sell | TradingSignal::Hold));
    }
    
    #[test]
    fn test_market_data_generation() {
        let trading = SecureTrading::new();
        let base_price = Decimal::new(10000, 2); // $100.00
        
        let data = trading.generate_secure_market_data(base_price, 0.02, 0.001).unwrap();
        assert!(data.price > Decimal::ZERO);
        assert!(data.volume > 0);
        assert!(data.volatility > 0.0);
    }
}