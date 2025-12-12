// Anti-Mock Enforcement Implementation - Production Data Only
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use anyhow::Result;
use tracing::{error, warn, info, debug};
use serde::{Deserialize, Serialize};

use super::{ValidationError, RuntimeViolation, ViolationType, ViolationSeverity, DataSource};

/// Violation detector for mock patterns
#[derive(Debug)]
pub struct ViolationDetector {
    forbidden_patterns: Vec<String>,
    endpoint_validators: HashMap<String, EndpointValidator>,
    credential_validators: Vec<CredentialValidator>,
}

impl ViolationDetector {
    pub fn new() -> Self {
        let forbidden_patterns = vec![
            // Mock patterns
            "mock.".to_string(),
            "fake_".to_string(),
            "dummy_".to_string(),
            "test_data".to_string(),
            "sample_".to_string(),
            "example_".to_string(),
            
            // Random data generators
            "rand::random".to_string(),
            "fastrand::".to_string(),
            "thread_rng()".to_string(),
            
            // Test frameworks
            "mockito".to_string(),
            "wiremock".to_string(),
            "testcontainers".to_string(),
            
            // Synthetic endpoints
            "localhost".to_string(),
            "127.0.0.1".to_string(),
            "0.0.0.0".to_string(),
            "mock-api".to_string(),
            "sandbox".to_string(),
            "testnet".to_string(),
        ];
        
        let mut endpoint_validators = HashMap::new();
        
        // Exchange endpoint validators
        endpoint_validators.insert("binance".to_string(), EndpointValidator {
            valid_hosts: vec![
                "api.binance.com".to_string(),
                "stream.binance.com".to_string(),
                "testnet.binance.vision".to_string(), // Allowed for testing
            ],
            forbidden_hosts: vec![
                "localhost".to_string(),
                "mock-binance".to_string(),
                "fake-api".to_string(),
            ],
        });
        
        endpoint_validators.insert("coinbase".to_string(), EndpointValidator {
            valid_hosts: vec![
                "api.coinbase.com".to_string(),
                "api-public.sandbox.pro.coinbase.com".to_string(), // Sandbox allowed
                "ws-feed.pro.coinbase.com".to_string(),
            ],
            forbidden_hosts: vec![
                "localhost".to_string(),
                "mock-coinbase".to_string(),
            ],
        });
        
        let credential_validators = vec![
            CredentialValidator {
                pattern: r"^test_.*".to_string(),
                description: "Test API keys".to_string(),
                severity: ViolationSeverity::High,
            },
            CredentialValidator {
                pattern: r"^mock_.*".to_string(),
                description: "Mock API keys".to_string(),
                severity: ViolationSeverity::Critical,
            },
            CredentialValidator {
                pattern: r"^fake_.*".to_string(),
                description: "Fake API keys".to_string(),
                severity: ViolationSeverity::Critical,
            },
            CredentialValidator {
                pattern: r"^demo_.*".to_string(),
                description: "Demo API keys".to_string(),
                severity: ViolationSeverity::Medium,
            },
        ];
        
        Self {
            forbidden_patterns,
            endpoint_validators,
            credential_validators,
        }
    }
    
    /// Detect forbidden patterns in source code
    pub fn detect_code_violations(&self, source_code: &str) -> Vec<String> {
        let mut violations = Vec::new();
        
        for pattern in &self.forbidden_patterns {
            if source_code.contains(pattern) {
                violations.push(format!("Forbidden pattern detected: {}", pattern));
            }
        }
        
        violations
    }
    
    /// Validate endpoint is not mock/synthetic
    pub fn validate_endpoint(&self, exchange: &str, endpoint: &str) -> Result<(), ValidationError> {
        if let Some(validator) = self.endpoint_validators.get(exchange) {
            // Check if endpoint is in forbidden list
            for forbidden in &validator.forbidden_hosts {
                if endpoint.contains(forbidden) {
                    return Err(ValidationError::InvalidEndpoint(
                        format!("Forbidden host {} in endpoint: {}", forbidden, endpoint)
                    ));
                }
            }
            
            // Check if endpoint is in valid list
            let is_valid = validator.valid_hosts.iter()
                .any(|valid| endpoint.contains(valid));
            
            if !is_valid {
                return Err(ValidationError::InvalidEndpoint(
                    format!("Unknown/invalid endpoint for {}: {}", exchange, endpoint)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Validate API credentials are not test/mock
    pub fn validate_credentials(&self, api_key: &str, secret: &str) -> Result<(), ValidationError> {
        for validator in &self.credential_validators {
            let regex = regex::Regex::new(&validator.pattern).unwrap();
            
            if regex.is_match(api_key) {
                match validator.severity {
                    ViolationSeverity::Critical => {
                        return Err(ValidationError::TestCredentialsDetected(
                            format!("Critical: {}", validator.description)
                        ));
                    },
                    ViolationSeverity::High => {
                        warn!("⚠️ HIGH SEVERITY: {} detected in API key", validator.description);
                    },
                    ViolationSeverity::Medium => {
                        debug!("Medium severity: {} detected in API key", validator.description);
                    },
                }
            }
            
            if regex.is_match(secret) {
                match validator.severity {
                    ViolationSeverity::Critical => {
                        return Err(ValidationError::TestCredentialsDetected(
                            format!("Critical: {} in secret", validator.description)
                        ));
                    },
                    ViolationSeverity::High => {
                        warn!("⚠️ HIGH SEVERITY: {} detected in API secret", validator.description);
                    },
                    ViolationSeverity::Medium => {
                        debug!("Medium severity: {} detected in API secret", validator.description);
                    },
                }
            }
        }
        
        Ok(())
    }
}

/// Data source validator
#[derive(Debug)]
pub struct DataSourceValidator {
    connection_validators: HashMap<String, ConnectionValidator>,
    response_validators: Vec<ResponseValidator>,
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
            ],
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
            ],
        });
        
        let response_validators = vec![
            ResponseValidator {
                pattern: r#""mock":\s*true"#.to_string(),
                description: "Mock flag in response".to_string(),
                severity: ViolationSeverity::Critical,
            },
            ResponseValidator {
                pattern: r#""test_mode":\s*true"#.to_string(),
                description: "Test mode flag".to_string(),
                severity: ViolationSeverity::High,
            },
            ResponseValidator {
                pattern: r#""synthetic":\s*true"#.to_string(),
                description: "Synthetic data flag".to_string(),
                severity: ViolationSeverity::Critical,
            },
        ];
        
        Self {
            connection_validators,
            response_validators,
        }
    }
    
    /// Validate connection is to real source
    pub fn validate_connection(&self, connection_type: &str, headers: &HashMap<String, String>) -> Result<(), ValidationError> {
        if let Some(validator) = self.connection_validators.get(connection_type) {
            // Check for forbidden headers
            for forbidden in &validator.forbidden_headers {
                let (key, value) = forbidden.split_once(": ").unwrap_or((forbidden, ""));
                if headers.get(key).map_or(false, |v| v.contains(value)) {
                    return Err(ValidationError::MockPatternDetected(
                        format!("Forbidden header: {}", forbidden)
                    ));
                }
            }
        }
        
        Ok(())
    }
    
    /// Validate response doesn't contain mock indicators
    pub fn validate_response(&self, response: &str) -> Result<(), ValidationError> {
        for validator in &self.response_validators {
            let regex = regex::Regex::new(&validator.pattern).unwrap();
            
            if regex.is_match(response) {
                match validator.severity {
                    ViolationSeverity::Critical => {
                        return Err(ValidationError::MockPatternDetected(
                            format!("Critical: {}", validator.description)
                        ));
                    },
                    ViolationSeverity::High => {
                        warn!("⚠️ HIGH SEVERITY: {} detected in response", validator.description);
                    },
                    ViolationSeverity::Medium => {
                        debug!("Medium severity: {} detected in response", validator.description);
                    },
                }
            }
        }
        
        Ok(())
    }
}

/// Supporting structures
#[derive(Debug, Clone)]
pub struct EndpointValidator {
    pub valid_hosts: Vec<String>,
    pub forbidden_hosts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CredentialValidator {
    pub pattern: String,
    pub description: String,
    pub severity: ViolationSeverity,
}

#[derive(Debug, Clone)]
pub struct ConnectionValidator {
    pub required_headers: Vec<String>,
    pub forbidden_headers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ResponseValidator {
    pub pattern: String,
    pub description: String,
    pub severity: ViolationSeverity,
}

/// Example real data source implementation
#[derive(Debug)]
pub struct RealExchangeDataSource {
    pub name: String,
    pub endpoint: String,
    pub last_update: chrono::DateTime<chrono::Utc>,
    pub api_key: String,
    pub is_connected: bool,
}

#[async_trait::async_trait]
impl DataSource for RealExchangeDataSource {
    async fn contains_synthetic_patterns(&self) -> Result<bool> {
        // Check endpoint for synthetic patterns
        let synthetic_indicators = [
            "mock", "fake", "test", "sandbox", "localhost", "127.0.0.1"
        ];
        
        for indicator in &synthetic_indicators {
            if self.endpoint.to_lowercase().contains(indicator) {
                return Ok(true);
            }
        }
        
        // Check API key for test patterns
        if self.api_key.starts_with("test_") || 
           self.api_key.starts_with("mock_") ||
           self.api_key.starts_with("fake_") {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    async fn verify_real_endpoints(&self) -> Result<bool> {
        // Verify endpoint is a real exchange
        let real_endpoints = [
            "api.binance.com",
            "api.coinbase.com", 
            "api.kraken.com",
            "api.bitfinex.com",
            "api.huobi.pro",
        ];
        
        for endpoint in &real_endpoints {
            if self.endpoint.contains(endpoint) {
                return Ok(true);
            }
        }
        
        // Check for testnet (allowed in some cases)
        if self.endpoint.contains("testnet") {
            info!("Using testnet endpoint: {}", self.endpoint);
            return Ok(true);
        }
        
        Ok(false)
    }
    
    fn last_update_age(&self) -> Duration {
        let now = chrono::Utc::now();
        (now - self.last_update).to_std().unwrap_or(Duration::from_secs(u64::MAX))
    }
    
    fn get_source_name(&self) -> String {
        self.name.clone()
    }
    
    fn get_endpoint(&self) -> String {
        self.endpoint.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_violation_detector() {
        let detector = ViolationDetector::new();
        
        // Test forbidden pattern detection
        let code_with_mock = "let data = mock.get_data();";
        let violations = detector.detect_code_violations(code_with_mock);
        assert!(!violations.is_empty());
        
        // Test clean code
        let clean_code = "let data = api.get_real_data();";
        let violations = detector.detect_code_violations(clean_code);
        assert!(violations.is_empty());
    }
    
    #[test]
    fn test_endpoint_validation() {
        let detector = ViolationDetector::new();
        
        // Test valid endpoint
        assert!(detector.validate_endpoint("binance", "https://api.binance.com/api/v3").is_ok());
        
        // Test invalid endpoint
        assert!(detector.validate_endpoint("binance", "https://mock-binance.com/api").is_err());
        
        // Test localhost (forbidden)
        assert!(detector.validate_endpoint("binance", "http://localhost:8080/api").is_err());
    }
    
    #[test]
    fn test_credential_validation() {
        let detector = ViolationDetector::new();
        
        // Test valid credentials
        assert!(detector.validate_credentials("real_api_key_123", "real_secret_456").is_ok());
        
        // Test test credentials (should fail)
        assert!(detector.validate_credentials("test_api_key", "test_secret").is_err());
        
        // Test mock credentials (should fail)
        assert!(detector.validate_credentials("mock_key", "mock_secret").is_err());
    }
    
    #[tokio::test]
    async fn test_real_data_source() {
        let source = RealExchangeDataSource {
            name: "Binance".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            last_update: chrono::Utc::now(),
            api_key: "real_api_key".to_string(),
            is_connected: true,
        };
        
        // Should not contain synthetic patterns
        assert!(!source.contains_synthetic_patterns().await.unwrap());
        
        // Should verify as real endpoint
        assert!(source.verify_real_endpoints().await.unwrap());
        
        // Should be production ready
        assert!(source.is_production_ready().await.unwrap());
    }
    
    #[tokio::test]
    async fn test_mock_data_source() {
        let mock_source = RealExchangeDataSource {
            name: "Mock Exchange".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            last_update: chrono::Utc::now(),
            api_key: "test_api_key".to_string(),
            is_connected: true,
        };
        
        // Should contain synthetic patterns
        assert!(mock_source.contains_synthetic_patterns().await.unwrap());
        
        // Should not verify as real endpoint
        assert!(!mock_source.verify_real_endpoints().await.unwrap());
        
        // Should not be production ready
        assert!(!mock_source.is_production_ready().await.unwrap());
    }
}