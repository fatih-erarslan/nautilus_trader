//! Secure Random Number Generation - Zero Mock Implementation
//!
//! Cryptographically secure random number generation using ring crate
//! for all financial calculations. Eliminates insecure thread_rng usage.

use ring::rand::{SecureRandom, SystemRandom};
use ring::error::Unspecified;
use std::sync::Arc;
use serde::{Serialize, Deserialize};

/// Cryptographically secure RNG for financial applications
pub struct SecureFinancialRng {
    rng: Arc<SystemRandom>,
}

impl SecureFinancialRng {
    /// Create new cryptographically secure RNG
    pub fn new() -> Self {
        Self {
            rng: Arc::new(SystemRandom::new()),
        }
    }
    
    /// Generate secure random f64 in range [0, 1)
    pub fn gen_f64(&self) -> Result<f64, Unspecified> {
        let mut bytes = [0u8; 8];
        self.rng.fill(&mut bytes)?;
        let uint_val = u64::from_le_bytes(bytes);
        Ok((uint_val as f64) / (u64::MAX as f64))
    }
    
    /// Generate secure random f64 in specified range
    pub fn gen_range_f64(&self, min: f64, max: f64) -> Result<f64, Unspecified> {
        let unit = self.gen_f64()?;
        Ok(min + unit * (max - min))
    }
    
    /// Generate secure random bytes for cryptographic operations
    pub fn fill_bytes(&self, dest: &mut [u8]) -> Result<(), Unspecified> {
        self.rng.fill(dest)
    }
    
    /// Generate secure random trading nonce
    pub fn gen_trading_nonce(&self) -> Result<u64, Unspecified> {
        let mut bytes = [0u8; 8];
        self.rng.fill(&mut bytes)?;
        Ok(u64::from_le_bytes(bytes))
    }
}

impl Default for SecureFinancialRng {
    fn default() -> Self {
        Self::new()
    }
}

/// Replace all insecure random usage with this secure implementation
pub fn secure_random() -> &'static SecureFinancialRng {
    use std::sync::OnceLock;
    static RNG: OnceLock<SecureFinancialRng> = OnceLock::new();
    RNG.get_or_init(|| SecureFinancialRng::new())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_secure_rng_generation() {
        let rng = SecureFinancialRng::new();
        
        // Test f64 generation
        let val = rng.gen_f64().unwrap();
        assert!(val >= 0.0 && val < 1.0);
        
        // Test range generation
        let range_val = rng.gen_range_f64(10.0, 100.0).unwrap();
        assert!(range_val >= 10.0 && range_val <= 100.0);
        
        // Test nonce generation
        let nonce1 = rng.gen_trading_nonce().unwrap();
        let nonce2 = rng.gen_trading_nonce().unwrap();
        assert_ne!(nonce1, nonce2); // Extremely high probability
    }
}