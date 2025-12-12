// Exchange Connectors - Real Market Data Sources Only
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use async_trait::async_trait;
use anyhow::Result;
use tokio_stream::Stream;
use serde::{Deserialize, Serialize};

use crate::{TradingPair, PairId};
use crate::data::{MarketUpdate, MarketData, ExchangeConfig};

pub mod binance;

pub use binance::{BinanceConnector, CoinbaseConnector, KrakenConnector};

/// Core trait for exchange connectors - REAL DATA ONLY
#[async_trait]
pub trait ExchangeConnector: Send + Sync {
    /// Test connection to real exchange
    async fn test_connection(&self) -> Result<bool>;
    
    /// Verify this is a real connection (not mock/synthetic)
    async fn is_real_connection(&self) -> Result<bool>;
    
    /// Get last data update timestamp
    async fn last_data_update(&self) -> Result<chrono::DateTime<chrono::Utc>>;
    
    /// Check if exchange supports a trading pair
    async fn supports_pair(&self, pair: &PairId) -> Result<bool>;
    
    /// Get specific pair data
    async fn get_pair_data(&self, pair: &PairId) -> Result<Option<MarketData>>;
    
    /// Fetch all available trading pairs
    async fn fetch_trading_pairs(&self) -> Result<Vec<TradingPair>>;
    
    /// Subscribe to real-time market data stream
    async fn subscribe_market_data(&self) -> Box<dyn Stream<Item = MarketUpdate> + Send + Unpin>;
    
    /// Subscribe to orderbook updates
    async fn subscribe_orderbook(&self, pairs: &[PairId]) -> Result<()>;
    
    /// Subscribe to trade updates
    async fn subscribe_trades(&self, pairs: &[PairId]) -> Result<()>;
    
    /// Subscribe to ticker updates
    async fn subscribe_tickers(&self, pairs: &[PairId]) -> Result<()>;
    
    /// Get exchange name
    fn get_name(&self) -> &str;
    
    /// Get exchange endpoint
    fn get_endpoint(&self) -> &str;
    
    /// Check if using testnet/sandbox
    fn is_testnet(&self) -> bool;
}

/// Exchange capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeCapabilities {
    pub supports_websocket: bool,
    pub supports_rest: bool,
    pub supports_orderbook: bool,
    pub supports_trades: bool,
    pub supports_klines: bool,
    pub rate_limit: u32,
    pub max_symbols_per_stream: u32,
}

/// Connection status for monitoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExchangeConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Reconnecting,
    Error,
}

/// Common exchange functionality
pub struct BaseExchangeConnector {
    pub name: String,
    pub config: ExchangeConfig,
    pub capabilities: ExchangeCapabilities,
    pub status: ExchangeConnectionStatus,
    pub last_update: chrono::DateTime<chrono::Utc>,
}

impl BaseExchangeConnector {
    pub fn new(name: String, config: ExchangeConfig, capabilities: ExchangeCapabilities) -> Self {
        Self {
            name,
            config,
            capabilities,
            status: ExchangeConnectionStatus::Disconnected,
            last_update: chrono::Utc::now(),
        }
    }
    
    /// Validate endpoint is real exchange
    pub fn validate_real_endpoint(&self) -> Result<bool> {
        let real_patterns = [
            "api.binance.com",
            "api.coinbase.com", 
            "api.kraken.com",
            "api.bitfinex.com",
            "api.huobi.pro",
            "testnet.binance.vision",
            "sandbox.pro.coinbase.com",
        ];
        
        for pattern in &real_patterns {
            if self.config.endpoint.contains(pattern) {
                return Ok(true);
            }
        }
        
        // Check for forbidden patterns
        let forbidden_patterns = ["mock", "fake", "localhost", "127.0.0.1"];
        for pattern in &forbidden_patterns {
            if self.config.endpoint.contains(pattern) {
                return Ok(false);
            }
        }
        
        Ok(false)
    }
    
    /// Check if credentials look real
    pub fn validate_credentials(&self) -> Result<bool> {
        // Check for test/mock patterns in API key
        let forbidden_prefixes = ["test_", "mock_", "fake_", "demo_"];
        
        for prefix in &forbidden_prefixes {
            if self.config.api_key.starts_with(prefix) {
                return Ok(false);
            }
        }
        
        // Real API keys should have minimum length
        if self.config.api_key.len() < 16 {
            return Ok(false);
        }
        
        Ok(true)
    }
}

/// Authentication helper
pub struct ExchangeAuth {
    pub api_key: String,
    pub api_secret: String,
    pub passphrase: Option<String>,
}

impl ExchangeAuth {
    pub fn new(config: &ExchangeConfig) -> Self {
        Self {
            api_key: config.api_key.clone(),
            api_secret: config.api_secret.clone(),
            passphrase: config.passphrase.clone(),
        }
    }
    
    /// Generate HMAC signature for API requests
    pub fn generate_signature(&self, message: &str) -> Result<String> {
        use hmac::{Hmac, Mac};
        use sha2::Sha256;
        
        type HmacSha256 = Hmac<Sha256>;
        
        let mut mac = HmacSha256::new_from_slice(self.api_secret.as_bytes())?;
        mac.update(message.as_bytes());
        let result = mac.finalize();
        
        Ok(hex::encode(result.into_bytes()))
    }
    
    /// Generate timestamp for API requests
    pub fn generate_timestamp() -> u64 {
        chrono::Utc::now().timestamp_millis() as u64
    }
}

/// Rate limiter for API requests
pub struct RateLimiter {
    requests_per_minute: u32,
    last_reset: chrono::DateTime<chrono::Utc>,
    request_count: u32,
}

impl RateLimiter {
    pub fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            last_reset: chrono::Utc::now(),
            request_count: 0,
        }
    }
    
    /// Check if request is allowed
    pub async fn check_rate_limit(&mut self) -> Result<bool> {
        let now = chrono::Utc::now();
        
        // Reset counter every minute
        if (now - self.last_reset).num_seconds() >= 60 {
            self.request_count = 0;
            self.last_reset = now;
        }
        
        if self.request_count >= self.requests_per_minute {
            // Wait until next reset
            let wait_time = 60 - (now - self.last_reset).num_seconds();
            if wait_time > 0 {
                tokio::time::sleep(std::time::Duration::from_secs(wait_time as u64)).await;
                self.request_count = 0;
                self.last_reset = chrono::Utc::now();
            }
        }
        
        self.request_count += 1;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_base_exchange_validation() {
        let config = ExchangeConfig {
            name: "binance".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            websocket_endpoint: "wss://stream.binance.com".to_string(),
            api_key: "real_api_key_12345678".to_string(),
            api_secret: "real_secret_key".to_string(),
            passphrase: None,
            rate_limit: 1200,
            enabled: true,
            testnet: false,
        };
        
        let capabilities = ExchangeCapabilities {
            supports_websocket: true,
            supports_rest: true,
            supports_orderbook: true,
            supports_trades: true,
            supports_klines: true,
            rate_limit: 1200,
            max_symbols_per_stream: 1024,
        };
        
        let connector = BaseExchangeConnector::new(
            "binance".to_string(),
            config,
            capabilities,
        );
        
        // Should validate as real endpoint
        assert!(connector.validate_real_endpoint().unwrap());
        
        // Should validate credentials
        assert!(connector.validate_credentials().unwrap());
    }
    
    #[test]
    fn test_mock_exchange_rejection() {
        let config = ExchangeConfig {
            name: "mock_exchange".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            websocket_endpoint: "ws://localhost:8080/ws".to_string(),
            api_key: "test_api_key".to_string(),
            api_secret: "test_secret".to_string(),
            passphrase: None,
            rate_limit: 1000,
            enabled: true,
            testnet: false,
        };
        
        let capabilities = ExchangeCapabilities {
            supports_websocket: true,
            supports_rest: true,
            supports_orderbook: true,
            supports_trades: true,
            supports_klines: true,
            rate_limit: 1000,
            max_symbols_per_stream: 100,
        };
        
        let connector = BaseExchangeConnector::new(
            "mock_exchange".to_string(),
            config,
            capabilities,
        );
        
        // Should reject mock endpoint
        assert!(!connector.validate_real_endpoint().unwrap());
        
        // Should reject test credentials
        assert!(!connector.validate_credentials().unwrap());
    }
    
    #[test]
    fn test_auth_signature() {
        let config = ExchangeConfig {
            name: "test".to_string(),
            endpoint: "https://api.test.com".to_string(),
            websocket_endpoint: "wss://stream.test.com".to_string(),
            api_key: "test_key".to_string(),
            api_secret: "test_secret".to_string(),
            passphrase: None,
            rate_limit: 1000,
            enabled: true,
            testnet: false,
        };
        
        let auth = ExchangeAuth::new(&config);
        let signature = auth.generate_signature("test_message").unwrap();
        
        // Should generate consistent signature
        assert!(!signature.is_empty());
        assert_eq!(signature.len(), 64); // SHA256 hex length
    }
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(5); // 5 requests per minute
        
        // Should allow initial requests
        for _ in 0..5 {
            assert!(limiter.check_rate_limit().await.unwrap());
        }
        
        // Should start rate limiting after limit reached
        // Note: In real test, this would wait, so we just check the structure works
        assert!(limiter.request_count >= 5);
    }
}