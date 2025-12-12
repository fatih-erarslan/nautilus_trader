// Data Source Validation - Anti-Mock Implementation
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::time::Duration;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use regex::Regex;

use super::{ValidationError, ViolationType, ViolationSeverity};

/// Data source validator for real-time enforcement
#[derive(Debug)]
pub struct DataSourceValidator {
    connection_validators: HashMap<String, ConnectionValidator>,
    response_validators: Vec<ResponseValidator>,
    endpoint_whitelist: HashMap<String, Vec<String>>,
    forbidden_patterns: Vec<Regex>,
}

impl DataSourceValidator {
    pub fn new() -> Self {
        let mut connection_validators = HashMap::new();
        
        // WebSocket connection validators
        connection_validators.insert("websocket".to_string(), ConnectionValidator {
            required_headers: vec![
                "Upgrade: websocket".to_string(),
                "Connection: Upgrade".to_string(),
            ],
            forbidden_headers: vec![
                "X-Mock-Response".to_string(),
                "X-Test-Mode".to_string(),
                "X-Synthetic-Data".to_string(),
            ],
            timeout: Duration::from_secs(30),
        });
        
        // REST API validators
        connection_validators.insert("rest".to_string(), ConnectionValidator {
            required_headers: vec![
                "Content-Type: application/json".to_string(),
            ],
            forbidden_headers: vec![
                "X-Mock-Data".to_string(),
                "X-Synthetic".to_string(),
                "X-Test-Environment".to_string(),
                "X-Fake-Response".to_string(),
            ],
            timeout: Duration::from_secs(10),
        });
        
        let response_validators = vec![
            ResponseValidator {
                pattern: Regex::new(r#""mock":\s*true"#).unwrap(),
                description: "Mock flag in JSON response".to_string(),
                severity: ViolationSeverity::Critical,
            },
            ResponseValidator {
                pattern: Regex::new(r#""test_mode":\s*true"#).unwrap(),
                description: "Test mode flag".to_string(),
                severity: ViolationSeverity::Critical,
            },
            ResponseValidator {
                pattern: Regex::new(r#""synthetic":\s*true"#).unwrap(),
                description: "Synthetic data flag".to_string(),
                severity: ViolationSeverity::Critical,
            },
            ResponseValidator {
                pattern: Regex::new(r#""fake":\s*true"#).unwrap(),
                description: "Fake data flag".to_string(),
                severity: ViolationSeverity::Critical,
            },
            ResponseValidator {
                pattern: Regex::new(r"test_data_\d+").unwrap(),
                description: "Test data pattern".to_string(),
                severity: ViolationSeverity::High,
            },
            ResponseValidator {
                pattern: Regex::new(r"mock_\w+").unwrap(),
                description: "Mock identifier pattern".to_string(),
                severity: ViolationSeverity::Critical,
            },
        ];
        
        // Endpoint whitelist for real exchanges
        let mut endpoint_whitelist = HashMap::new();
        endpoint_whitelist.insert("binance".to_string(), vec![
            "api.binance.com".to_string(),
            "stream.binance.com".to_string(),
            "testnet.binance.vision".to_string(), // Testnet allowed for testing
        ]);
        endpoint_whitelist.insert("coinbase".to_string(), vec![
            "api.coinbase.com".to_string(),
            "api-public.sandbox.pro.coinbase.com".to_string(), // Sandbox allowed
            "ws-feed.pro.coinbase.com".to_string(),
        ]);
        endpoint_whitelist.insert("kraken".to_string(), vec![
            "api.kraken.com".to_string(),
            "ws.kraken.com".to_string(),
        ]);
        
        // Forbidden patterns (regex)
        let forbidden_patterns = vec![
            Regex::new(r"localhost|127\.0\.0\.1|0\.0\.0\.0").unwrap(),
            Regex::new(r"mock|fake|dummy|test|synthetic").unwrap(),
            Regex::new(r"sandbox.*(?!coinbase|binance)").unwrap(), // Allow official sandboxes only
        ];
        
        Self {
            connection_validators,
            response_validators,
            endpoint_whitelist,
            forbidden_patterns,
        }
    }
    
    /// Validate endpoint is real and whitelisted
    pub fn validate_endpoint(&self, exchange: &str, endpoint: &str) -> Result<(), ValidationError> {
        // Check against whitelist
        if let Some(whitelist) = self.endpoint_whitelist.get(exchange) {
            let is_whitelisted = whitelist.iter()
                .any(|allowed| endpoint.contains(allowed));
            
            if !is_whitelisted {
                return Err(ValidationError::InvalidEndpoint(
                    format!("Endpoint {} not in whitelist for {}", endpoint, exchange)
                ));
            }
        } else {
            return Err(ValidationError::InvalidEndpoint(
                format!("Unknown exchange: {}", exchange)
            ));
        }
        
        // Check forbidden patterns
        for pattern in &self.forbidden_patterns {
            if pattern.is_match(endpoint) {
                return Err(ValidationError::InvalidEndpoint(
                    format!("Forbidden pattern detected in endpoint: {}", endpoint)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate API credentials are production-ready
    pub fn validate_credentials(&self, api_key: &str, secret: &str) -> Result<(), ValidationError> {
        // Check key length (real keys are typically longer)
        if api_key.len() < 16 {
            return Err(ValidationError::TestCredentialsDetected(
                "API key too short for production".to_string()
            ));
        }
        
        // Check for test/mock prefixes
        let forbidden_prefixes = ["test_", "mock_", "fake_", "demo_", "sample_"];
        for prefix in &forbidden_prefixes {
            if api_key.starts_with(prefix) || secret.starts_with(prefix) {
                return Err(ValidationError::TestCredentialsDetected(
                    format!("Credentials contain forbidden prefix: {}", prefix)
                ));
            }
        }
        
        // Check for obvious test patterns
        let test_patterns = [
            "test", "mock", "fake", "demo", "example", "sample",
            "12345", "abcde", "aaaaa", "00000"
        ];
        
        for pattern in &test_patterns {
            if api_key.to_lowercase().contains(pattern) ||
               secret.to_lowercase().contains(pattern) {
                return Err(ValidationError::TestCredentialsDetected(
                    format!("Credentials contain test pattern: {}", pattern)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate connection headers
    pub fn validate_connection(&self, connection_type: &str, headers: &HashMap<String, String>) -> Result<(), ValidationError> {
        if let Some(validator) = self.connection_validators.get(connection_type) {
            // Check for forbidden headers
            for forbidden in &validator.forbidden_headers {
                let (key, value) = forbidden.split_once(": ").unwrap_or((forbidden, ""));
                if let Some(header_value) = headers.get(key) {
                    if header_value.contains(value) {
                        return Err(ValidationError::MockPatternDetected(
                            format!("Forbidden header detected: {}", forbidden)
                        ));
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate response data doesn't contain mock indicators
    pub fn validate_response(&self, response: &str) -> Result<(), ValidationError> {
        for validator in &self.response_validators {
            if validator.pattern.is_match(response) {
                match validator.severity {
                    ViolationSeverity::Critical => {
                        return Err(ValidationError::MockPatternDetected(
                            format!("CRITICAL: {}", validator.description)
                        ));
                    },
                    ViolationSeverity::High => {
                        tracing::warn!("⚠️ HIGH SEVERITY: {} detected in response", validator.description);
                    },
                    ViolationSeverity::Medium => {
                        tracing::debug!("Medium severity: {} detected in response", validator.description);
                    },
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate data freshness
    pub fn validate_data_freshness(&self, timestamp: chrono::DateTime<chrono::Utc>, max_age: Duration) -> Result<(), ValidationError> {
        let age = chrono::Utc::now() - timestamp;
        let age_duration = age.to_std().unwrap_or(Duration::from_secs(u64::MAX));
        
        if age_duration > max_age {
            return Err(ValidationError::StaleData(
                format!("Data is {} seconds old, max allowed: {} seconds", 
                        age_duration.as_secs(), max_age.as_secs())
            ));
        }
        
        Ok(())
    }
    
    /// Comprehensive validation for market data
    pub fn validate_market_data(&self, data: &crate::data::MarketData) -> Result<(), ValidationError> {
        // Check for obvious synthetic patterns in symbol
        if data.symbol.contains("test") || data.symbol.contains("mock") || data.symbol.contains("fake") {
            return Err(ValidationError::SyntheticDataDetected(
                format!("Synthetic pattern in symbol: {}", data.symbol)
            ));
        }
        
        // Validate price data makes sense
        if data.price <= 0.0 || data.bid <= 0.0 || data.ask <= 0.0 {
            return Err(ValidationError::MockPatternDetected(
                "Invalid price data (non-positive values)".to_string()
            ));
        }
        
        // Validate bid-ask spread is reasonable
        if data.bid >= data.ask {
            return Err(ValidationError::MockPatternDetected(
                "Invalid bid-ask spread (bid >= ask)".to_string()
            ));
        }
        
        // Check spread isn't impossibly tight (could indicate synthetic data)
        let spread_pct = (data.ask - data.bid) / data.price;
        if spread_pct < 0.0001 && data.price > 1.0 {
            return Err(ValidationError::SyntheticDataDetected(
                "Suspiciously tight spread detected".to_string()
            ));
        }
        
        // Validate volume is positive
        if data.volume < 0.0 || data.volume_24h < 0.0 {
            return Err(ValidationError::MockPatternDetected(
                "Invalid volume data (negative values)".to_string()
            ));
        }
        
        // Check data freshness
        self.validate_data_freshness(data.timestamp, Duration::from_secs(300))?;
        
        Ok(())
    }
}

/// Connection validator configuration
#[derive(Debug, Clone)]
pub struct ConnectionValidator {
    pub required_headers: Vec<String>,
    pub forbidden_headers: Vec<String>,
    pub timeout: Duration,
}

/// Response pattern validator
#[derive(Debug)]
pub struct ResponseValidator {
    pub pattern: Regex,
    pub description: String,
    pub severity: ViolationSeverity,
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    
    #[test]
    fn test_endpoint_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid endpoints
        assert!(validator.validate_endpoint("binance", "https://api.binance.com/api/v3").is_ok());
        assert!(validator.validate_endpoint("coinbase", "https://api.coinbase.com/v2").is_ok());
        
        // Test invalid endpoints
        assert!(validator.validate_endpoint("binance", "https://mock-binance.com/api").is_err());
        assert!(validator.validate_endpoint("binance", "http://localhost:8080/api").is_err());
        assert!(validator.validate_endpoint("unknown", "https://api.unknown.com").is_err());
    }
    
    #[test]
    fn test_credential_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid credentials
        assert!(validator.validate_credentials("real_api_key_1234567890", "real_secret_key").is_ok());
        
        // Test invalid credentials
        assert!(validator.validate_credentials("test_api_key", "test_secret").is_err());
        assert!(validator.validate_credentials("mock_key", "mock_secret").is_err());
        assert!(validator.validate_credentials("short", "secret").is_err()); // Too short
        assert!(validator.validate_credentials("demo_key_12345678", "demo_secret").is_err());
    }
    
    #[test]
    fn test_response_validation() {
        let validator = DataSourceValidator::new();
        
        // Test clean response
        let clean_response = r#"{"symbol":"BTCUSDT","price":"50000.00","volume":"1.5"}"#;
        assert!(validator.validate_response(clean_response).is_ok());
        
        // Test mock response
        let mock_response = r#"{"symbol":"BTCUSDT","price":"50000.00","mock":true}"#;
        assert!(validator.validate_response(mock_response).is_err());
        
        // Test test mode response
        let test_response = r#"{"symbol":"BTCUSDT","test_mode":true,"price":"50000.00"}"#;
        assert!(validator.validate_response(test_response).is_err());
        
        // Test synthetic response
        let synthetic_response = r#"{"symbol":"BTCUSDT","synthetic":true,"price":"50000.00"}"#;
        assert!(validator.validate_response(synthetic_response).is_err());
    }
    
    #[test]
    fn test_market_data_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid market data
        let valid_data = crate::data::MarketData {
            symbol: "BTCUSDT".to_string(),
            timestamp: Utc::now(),
            price: 50000.0,
            volume: 1.5,
            bid: 49995.0,
            ask: 50005.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            change_24h: 0.02,
            volume_24h: 1500.0,
            quote_volume: 75000000.0,
            volatility: Some(0.25),
            liquidity_score: Some(0.85),
        };
        assert!(validator.validate_market_data(&valid_data).is_ok());
        
        // Test invalid market data (negative price)
        let mut invalid_data = valid_data.clone();
        invalid_data.price = -100.0;
        assert!(validator.validate_market_data(&invalid_data).is_err());
        
        // Test invalid market data (inverted spread)
        let mut invalid_spread = valid_data.clone();
        invalid_spread.bid = 50010.0;
        invalid_spread.ask = 49990.0; // bid > ask
        assert!(validator.validate_market_data(&invalid_spread).is_err());
        
        // Test synthetic symbol
        let mut synthetic_data = valid_data.clone();
        synthetic_data.symbol = "TEST_SYMBOL".to_string();
        assert!(validator.validate_market_data(&synthetic_data).is_err());
        
        // Test stale data
        let mut stale_data = valid_data.clone();
        stale_data.timestamp = Utc::now() - chrono::Duration::minutes(10);
        assert!(validator.validate_market_data(&stale_data).is_err());
    }
    
    #[test]
    fn test_data_freshness() {
        let validator = DataSourceValidator::new();
        
        // Test fresh data
        let fresh_timestamp = Utc::now();
        assert!(validator.validate_data_freshness(fresh_timestamp, Duration::from_secs(300)).is_ok());
        
        // Test stale data
        let stale_timestamp = Utc::now() - chrono::Duration::seconds(600);
        assert!(validator.validate_data_freshness(stale_timestamp, Duration::from_secs(300)).is_err());
    }
}