//! Rate limiting for API requests

use governor::{Quota, RateLimiter as GovernorRateLimiter, state::{InMemoryState, NotKeyed}, clock::DefaultClock};
use std::num::NonZeroU32;
use std::time::Duration;
use tracing::debug;

/// Rate limiter for API requests
pub struct RateLimiter {
    limiter: GovernorRateLimiter<NotKeyed, InMemoryState, DefaultClock>,
    name: String,
    requests_per_period: u32,
    period: Duration,
}

impl RateLimiter {
    /// Create a new rate limiter
    pub fn new(requests_per_period: u32, period: Duration) -> Self {
        let quota = Quota::with_period(period)
            .unwrap()
            .allow_burst(NonZeroU32::new(requests_per_period).unwrap());
            
        let limiter = GovernorRateLimiter::direct(quota);
        
        Self {
            limiter,
            name: "default".to_string(),
            requests_per_period,
            period,
        }
    }
    
    /// Create a named rate limiter
    pub fn with_name(name: String, requests_per_period: u32, period: Duration) -> Self {
        let mut limiter = Self::new(requests_per_period, period);
        limiter.name = name;
        limiter
    }
    
    /// Wait until the next request is allowed
    pub async fn wait(&self) {
        match self.limiter.until_ready().await {
            Ok(_) => {
                debug!("Rate limiter '{}' allowed request", self.name);
            },
            Err(_) => {
                // This shouldn't happen with the direct limiter
                debug!("Rate limiter '{}' encountered error", self.name);
            }
        }
    }
    
    /// Check if a request is currently allowed (non-blocking)
    pub fn check(&self) -> bool {
        self.limiter.check().is_ok()
    }
    
    /// Get rate limiter information
    pub fn info(&self) -> RateLimiterInfo {
        RateLimiterInfo {
            name: self.name.clone(),
            requests_per_period: self.requests_per_period,
            period_seconds: self.period.as_secs(),
            current_burst: self.limiter.get_current_burst(),
        }
    }
}

/// Rate limiter information
#[derive(Debug, Clone)]
pub struct RateLimiterInfo {
    pub name: String,
    pub requests_per_period: u32,
    pub period_seconds: u64,
    pub current_burst: u32,
}

/// Initialize rate limiter system
pub async fn init() -> crate::Result<()> {
    debug!("Rate limiter system initialized");
    Ok(())
}

/// Create rate limiters for common exchanges
pub fn create_exchange_limiters() -> std::collections::HashMap<String, RateLimiter> {
    let mut limiters = std::collections::HashMap::new();
    
    // Binance - 1200 requests per minute
    limiters.insert(
        "binance".to_string(),
        RateLimiter::with_name(
            "binance".to_string(),
            1200,
            Duration::from_secs(60)
        )
    );
    
    // Coinbase Pro - 300 requests per minute
    limiters.insert(
        "coinbase".to_string(),
        RateLimiter::with_name(
            "coinbase".to_string(),
            300,
            Duration::from_secs(60)
        )
    );
    
    // Kraken - 180 requests per minute
    limiters.insert(
        "kraken".to_string(),
        RateLimiter::with_name(
            "kraken".to_string(),
            180,
            Duration::from_secs(60)
        )
    );
    
    // OKX - 600 requests per minute
    limiters.insert(
        "okx".to_string(),
        RateLimiter::with_name(
            "okx".to_string(),
            600,
            Duration::from_secs(60)
        )
    );
    
    // Bybit - 600 requests per minute
    limiters.insert(
        "bybit".to_string(),
        RateLimiter::with_name(
            "bybit".to_string(),
            600,
            Duration::from_secs(60)
        )
    );
    
    limiters
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let limiter = RateLimiter::new(2, Duration::from_secs(1));
        
        let start = Instant::now();
        
        // First two requests should be immediate
        limiter.wait().await;
        limiter.wait().await;
        
        let first_two_elapsed = start.elapsed();
        assert!(first_two_elapsed < Duration::from_millis(100));
        
        // Third request should be delayed
        limiter.wait().await;
        
        let third_elapsed = start.elapsed();
        assert!(third_elapsed >= Duration::from_millis(500));
    }
    
    #[test]
    fn test_rate_limiter_check() {
        let limiter = RateLimiter::new(1, Duration::from_secs(1));
        
        // First check should pass
        assert!(limiter.check());
        
        // Immediate second check should fail (burst exhausted)
        assert!(!limiter.check());
    }
    
    #[test]
    fn test_exchange_limiters() {
        let limiters = create_exchange_limiters();
        
        assert!(limiters.contains_key("binance"));
        assert!(limiters.contains_key("coinbase"));
        assert!(limiters.contains_key("kraken"));
        assert!(limiters.contains_key("okx"));
        assert!(limiters.contains_key("bybit"));
        
        let binance_info = limiters["binance"].info();
        assert_eq!(binance_info.requests_per_period, 1200);
        assert_eq!(binance_info.period_seconds, 60);
    }
}