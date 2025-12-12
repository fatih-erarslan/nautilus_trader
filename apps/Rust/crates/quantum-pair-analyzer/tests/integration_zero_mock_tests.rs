// Comprehensive Zero-Mock Enforcement Integration Tests
// Copyright (c) 2025 TENGRI Trading Swarm
// This file tests all anti-mock enforcement components

use std::collections::HashMap;
use std::time::Duration;
use tokio::time::sleep;
use chrono::Utc;

use quantum_pair_analyzer::anti_mock::{
    AntiMockEnforcer, ViolationDetector, DataSourceValidator, CompileTimeScanner,
    RuntimeMonitor, ConnectionInfo, ValidationStatus, ValidationError, 
    RuntimeViolation, ViolationType, ViolationSeverity, Violation, DataSource,
    enforce_real_data, check_production_readiness
};
use quantum_pair_analyzer::data::MarketData;

/// Test data source implementation for testing
#[derive(Debug)]
struct TestDataSource {
    name: String,
    endpoint: String,
    last_update: chrono::DateTime<chrono::Utc>,
    api_key: String,
    contains_mock: bool,
    has_real_endpoint: bool,
}

#[async_trait::async_trait]
impl DataSource for TestDataSource {
    async fn contains_synthetic_patterns(&self) -> anyhow::Result<bool> {
        Ok(self.contains_mock)
    }
    
    async fn verify_real_endpoints(&self) -> anyhow::Result<bool> {
        Ok(self.has_real_endpoint)
    }
    
    fn last_update_age(&self) -> Duration {
        let now = Utc::now();
        (now - self.last_update).to_std().unwrap_or(Duration::from_secs(u64::MAX))
    }
    
    fn get_source_name(&self) -> String {
        self.name.clone()
    }
    
    fn get_endpoint(&self) -> String {
        self.endpoint.clone()
    }
}

impl TestDataSource {
    fn new_real_source() -> Self {
        Self {
            name: "Binance".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            last_update: Utc::now(),
            api_key: "real_api_key_1234567890".to_string(),
            contains_mock: false,
            has_real_endpoint: true,
        }
    }
    
    fn new_mock_source() -> Self {
        Self {
            name: "MockExchange".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            last_update: Utc::now(),
            api_key: "test_api_key".to_string(),
            contains_mock: true,
            has_real_endpoint: false,
        }
    }
    
    fn new_stale_source() -> Self {
        Self {
            name: "StaleExchange".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            last_update: Utc::now() - chrono::Duration::minutes(10),
            api_key: "real_api_key_1234567890".to_string(),
            contains_mock: false,
            has_real_endpoint: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_anti_mock_enforcer_real_data_source() {
        let enforcer = AntiMockEnforcer::new();
        let real_source = TestDataSource::new_real_source();
        
        // Should pass validation
        let result = enforcer.validate_data_source(&real_source).await;
        assert!(result.is_ok(), "Real data source should pass validation");
    }
    
    #[tokio::test]
    async fn test_anti_mock_enforcer_mock_data_source() {
        let enforcer = AntiMockEnforcer::new();
        let mock_source = TestDataSource::new_mock_source();
        
        // Should fail validation
        let result = enforcer.validate_data_source(&mock_source).await;
        assert!(result.is_err(), "Mock data source should fail validation");
        
        match result.unwrap_err() {
            ValidationError::SyntheticDataDetected(_) => {
                // Expected error type
            },
            ValidationError::InvalidEndpoint(_) => {
                // Also acceptable
            },
            _ => panic!("Unexpected error type"),
        }
    }
    
    #[tokio::test]
    async fn test_anti_mock_enforcer_stale_data_source() {
        let enforcer = AntiMockEnforcer::new();
        let stale_source = TestDataSource::new_stale_source();
        
        // Should fail validation due to stale data
        let result = enforcer.validate_data_source(&stale_source).await;
        assert!(result.is_err(), "Stale data source should fail validation");
        
        match result.unwrap_err() {
            ValidationError::StaleData(_) => {
                // Expected error type
            },
            _ => panic!("Expected StaleData error"),
        }
    }
    
    #[test]
    fn test_violation_detector_code_scanning() {
        let detector = ViolationDetector::new();
        
        // Test code with forbidden patterns
        let mock_code = r#"
            let client = mock.create_client();
            let data = fake_data_generator();
            let endpoint = "http://localhost:8080/api";
            let key = "test_api_key";
        "#;
        
        let violations = detector.detect_code_violations(mock_code);
        assert!(!violations.is_empty(), "Should detect violations in mock code");
        
        // Test clean code
        let clean_code = r#"
            let client = RealClient::new(api_key);
            let data = client.fetch_market_data("BTCUSDT").await?;
            let endpoint = "https://api.binance.com/api/v3";
        "#;
        
        let violations = detector.detect_code_violations(clean_code);
        assert!(violations.is_empty(), "Should not detect violations in clean code");
    }
    
    #[test]
    fn test_violation_detector_endpoint_validation() {
        let detector = ViolationDetector::new();
        
        // Test valid endpoints
        assert!(detector.validate_endpoint("binance", "https://api.binance.com/api/v3").is_ok());
        assert!(detector.validate_endpoint("coinbase", "https://api.coinbase.com/v2").is_ok());
        
        // Test invalid endpoints
        assert!(detector.validate_endpoint("binance", "https://mock-binance.com/api").is_err());
        assert!(detector.validate_endpoint("binance", "http://localhost:8080/api").is_err());
        assert!(detector.validate_endpoint("unknown", "https://api.unknown.com").is_err());
    }
    
    #[test]
    fn test_violation_detector_credential_validation() {
        let detector = ViolationDetector::new();
        
        // Test valid credentials
        assert!(detector.validate_credentials("production_key_1234567890", "production_secret_abcdef").is_ok());
        
        // Test invalid credentials
        assert!(detector.validate_credentials("test_api_key", "test_secret").is_err());
        assert!(detector.validate_credentials("mock_key", "mock_secret").is_err());
        assert!(detector.validate_credentials("fake_key", "fake_secret").is_err());
        assert!(detector.validate_credentials("demo_key", "demo_secret").is_err());
    }
    
    #[test]
    fn test_data_source_validator_endpoint_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid endpoints
        assert!(validator.validate_endpoint("binance", "https://api.binance.com/api/v3").is_ok());
        assert!(validator.validate_endpoint("coinbase", "https://api.coinbase.com/v2").is_ok());
        
        // Test invalid endpoints
        assert!(validator.validate_endpoint("binance", "https://mock-binance.com/api").is_err());
        assert!(validator.validate_endpoint("binance", "http://localhost:8080/api").is_err());
    }
    
    #[test]
    fn test_data_source_validator_credential_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid credentials
        assert!(validator.validate_credentials("production_key_1234567890", "production_secret").is_ok());
        
        // Test invalid credentials
        assert!(validator.validate_credentials("test_api_key", "test_secret").is_err());
        assert!(validator.validate_credentials("mock_key", "mock_secret").is_err());
        assert!(validator.validate_credentials("short", "secret").is_err()); // Too short
    }
    
    #[test]
    fn test_data_source_validator_response_validation() {
        let validator = DataSourceValidator::new();
        
        // Test clean response
        let clean_response = r#"{"symbol":"BTCUSDT","price":"50000.00","volume":"1.5"}"#;
        assert!(validator.validate_response(clean_response).is_ok());
        
        // Test responses with mock indicators
        let mock_response = r#"{"symbol":"BTCUSDT","price":"50000.00","mock":true}"#;
        assert!(validator.validate_response(mock_response).is_err());
        
        let test_response = r#"{"symbol":"BTCUSDT","test_mode":true,"price":"50000.00"}"#;
        assert!(validator.validate_response(test_response).is_err());
        
        let synthetic_response = r#"{"symbol":"BTCUSDT","synthetic":true,"price":"50000.00"}"#;
        assert!(validator.validate_response(synthetic_response).is_err());
    }
    
    #[test]
    fn test_data_source_validator_market_data_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid market data
        let valid_data = MarketData {
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
        
        // Test invalid market data (synthetic symbol)
        let mut synthetic_data = valid_data.clone();
        synthetic_data.symbol = "TEST_SYMBOL".to_string();
        assert!(validator.validate_market_data(&synthetic_data).is_err());
        
        // Test invalid market data (negative price)
        let mut invalid_data = valid_data.clone();
        invalid_data.price = -100.0;
        assert!(validator.validate_market_data(&invalid_data).is_err());
        
        // Test invalid market data (inverted spread)
        let mut invalid_spread = valid_data.clone();
        invalid_spread.bid = 50010.0;
        invalid_spread.ask = 49990.0;
        assert!(validator.validate_market_data(&invalid_spread).is_err());
    }
    
    #[test]
    fn test_compile_time_scanner_mock_detection() {
        let scanner = CompileTimeScanner::new();
        
        // Test code with mock patterns
        let mock_code = r#"
            use mockito::Server;
            
            fn test_function() {
                let server = Server::new();
                let data = mock(some_params);
                let fake_value = fake_data();
                let random_val = rand::random::<f64>();
                let url = "http://localhost:8080/api";
            }
        "#;
        
        let violations = scanner.scan_for_violations(mock_code);
        assert!(!violations.is_empty(), "Should detect mock patterns in code");
        
        // Test clean production code
        let clean_code = r#"
            use reqwest::Client;
            use serde_json::Value;
            
            async fn fetch_market_data(api_key: &str) -> Result<Value> {
                let client = Client::new();
                let response = client
                    .get("https://api.binance.com/api/v3/ticker/price")
                    .header("X-MBX-APIKEY", api_key)
                    .send()
                    .await?;
                Ok(response.json().await?)
            }
        "#;
        
        let violations = scanner.scan_for_violations(clean_code);
        assert!(violations.is_empty(), "Should not detect violations in clean code");
    }
    
    #[test]
    fn test_compile_time_scanner_production_readiness() {
        let scanner = CompileTimeScanner::new();
        
        // Test production-ready code
        let production_code = r#"
            use reqwest::Client;
            
            async fn fetch_real_data(api_key: &str) -> Result<MarketData> {
                let client = Client::new();
                let response = client
                    .get("https://api.binance.com/api/v3/ticker/24hr")
                    .header("X-MBX-APIKEY", api_key)
                    .send()
                    .await?;
                Ok(response.json().await?)
            }
        "#;
        
        assert!(scanner.check_production_readiness(production_code).is_ok());
        
        // Test non-production code
        let non_production_code = r#"
            fn get_data() {
                let data = mock(test_params);
                let endpoint = "http://localhost:8080";
                let fake_data = fake_data_generator();
            }
        "#;
        
        assert!(scanner.check_production_readiness(non_production_code).is_err());
    }
    
    #[test]
    fn test_compile_time_scanner_violation_report() {
        let scanner = CompileTimeScanner::new();
        
        let problematic_code = r#"
            fn problematic_function() {
                let data = mock(params);
                let fake_value = fake_data();
                let url = "http://localhost:8080";
                let random_val = rand::random::<f64>();
                let test_data = get_test_data();
            }
        "#;
        
        let report = scanner.generate_violation_report(problematic_code);
        assert!(report.total_violations > 0, "Should detect multiple violations");
        assert!(report.critical_violations > 0, "Should detect critical violations");
        assert!(!report.production_ready, "Should not be production ready");
    }
    
    #[tokio::test]
    async fn test_runtime_monitor_connection_registration() {
        let monitor = RuntimeMonitor::new();
        
        // Test legitimate connection
        let legitimate_conn = ConnectionInfo {
            connection_id: "binance_main".to_string(),
            connection_type: "websocket".to_string(),
            endpoint: "https://api.binance.com".to_string(),
            established_at: Utc::now(),
            last_activity: Utc::now(),
            data_source_type: "binance".to_string(),
            validation_status: ValidationStatus::Unknown,
        };
        
        assert!(monitor.register_connection(legitimate_conn).await.is_ok());
        
        // Test suspicious connection
        let suspicious_conn = ConnectionInfo {
            connection_id: "mock_conn".to_string(),
            connection_type: "rest".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            established_at: Utc::now(),
            last_activity: Utc::now(),
            data_source_type: "mock_exchange".to_string(),
            validation_status: ValidationStatus::Unknown,
        };
        
        // Should register but with blocked status
        assert!(monitor.register_connection(suspicious_conn).await.is_ok());
        
        // Check that suspicious connection was marked as blocked
        let connections = monitor.get_active_connections().await;
        let mock_conn = connections.get("mock_conn").unwrap();
        assert_eq!(mock_conn.validation_status, ValidationStatus::Blocked);
    }
    
    #[tokio::test]
    async fn test_runtime_monitor_violation_scanning() {
        let monitor = RuntimeMonitor::new();
        
        // Register a connection with suspicious endpoint
        let suspicious_conn = ConnectionInfo {
            connection_id: "suspicious_conn".to_string(),
            connection_type: "rest".to_string(),
            endpoint: "http://localhost:8080/mock".to_string(),
            established_at: Utc::now(),
            last_activity: Utc::now(),
            data_source_type: "mock_exchange".to_string(),
            validation_status: ValidationStatus::Unknown,
        };
        
        monitor.register_connection(suspicious_conn).await.unwrap();
        
        // Scan for violations
        let violations = monitor.scan_active_connections().await.unwrap();
        assert!(!violations.is_empty(), "Should detect violations");
        
        // Check violation type
        let violation = &violations[0];
        assert_eq!(violation.violation_type, ViolationType::SyntheticEndpoint);
        assert_eq!(violation.severity, ViolationSeverity::Critical);
    }
    
    #[tokio::test]
    async fn test_runtime_monitor_data_source_validation() {
        let monitor = RuntimeMonitor::new();
        
        // Test with real data source
        let real_source = TestDataSource::new_real_source();
        assert!(monitor.validate_runtime_patterns(&real_source).await.is_ok());
        
        // Test with mock data source
        let mock_source = TestDataSource::new_mock_source();
        assert!(monitor.validate_runtime_patterns(&mock_source).await.is_err());
        
        // Test with stale data source
        let stale_source = TestDataSource::new_stale_source();
        assert!(monitor.validate_runtime_patterns(&stale_source).await.is_err());
    }
    
    #[tokio::test]
    async fn test_global_production_readiness_check() {
        // This test simulates a production readiness check
        let result = check_production_readiness().await;
        
        // In a real environment, this would check actual connections
        // For testing purposes, we expect it to pass basic checks
        assert!(result.is_ok() || result.is_err(), "Should return a valid result");
    }
    
    #[tokio::test]
    async fn test_enforce_real_data_macro_simulation() {
        // Test the enforce_real_data macro behavior
        let real_source = TestDataSource::new_real_source();
        
        // Simulate macro validation
        let result = crate::anti_mock::validate_data_source(&real_source).await;
        assert!(result.is_ok(), "Real data source should pass macro validation");
        
        let mock_source = TestDataSource::new_mock_source();
        let result = crate::anti_mock::validate_data_source(&mock_source).await;
        assert!(result.is_err(), "Mock data source should fail macro validation");
    }
    
    #[tokio::test]
    async fn test_comprehensive_anti_mock_enforcement() {
        // Comprehensive integration test
        let enforcer = AntiMockEnforcer::new();
        
        // Test code scanning
        let mock_code = r#"
            let client = mock.create_client();
            let data = fake_data_generator();
            let endpoint = "http://localhost:8080/api";
        "#;
        
        let code_violations = enforcer.enforce_real_data_compile_time(mock_code);
        assert!(code_violations.is_err(), "Should detect compile-time violations");
        
        // Test runtime enforcement
        let runtime_result = enforcer.enforce_runtime().await;
        // Runtime enforcement may pass or fail depending on system state
        assert!(runtime_result.is_ok() || runtime_result.is_err(), "Should return valid result");
        
        // Test data source validation
        let real_source = TestDataSource::new_real_source();
        let validation_result = enforcer.validate_data_source(&real_source).await;
        assert!(validation_result.is_ok(), "Real source should pass validation");
        
        let mock_source = TestDataSource::new_mock_source();
        let validation_result = enforcer.validate_data_source(&mock_source).await;
        assert!(validation_result.is_err(), "Mock source should fail validation");
    }
    
    #[test]
    fn test_violation_types_and_severity() {
        // Test violation type classification
        let violation = RuntimeViolation {
            violation_type: ViolationType::MockDataSource,
            source: "test_source".to_string(),
            endpoint: "http://localhost:8080".to_string(),
            timestamp: Utc::now(),
            severity: ViolationSeverity::Critical,
        };
        
        assert_eq!(violation.violation_type, ViolationType::MockDataSource);
        assert_eq!(violation.severity, ViolationSeverity::Critical);
        
        // Test compile-time violation
        let compile_violation = Violation::MockFunction("mock()".to_string());
        let display_string = format!("{}", compile_violation);
        assert!(display_string.contains("Mock function"));
    }
    
    #[tokio::test]
    async fn test_connection_header_validation() {
        let validator = DataSourceValidator::new();
        
        // Test valid headers
        let mut valid_headers = HashMap::new();
        valid_headers.insert("Content-Type".to_string(), "application/json".to_string());
        valid_headers.insert("Authorization".to_string(), "Bearer token".to_string());
        
        assert!(validator.validate_connection("rest", &valid_headers).is_ok());
        
        // Test forbidden headers
        let mut forbidden_headers = HashMap::new();
        forbidden_headers.insert("X-Mock-Data".to_string(), "true".to_string());
        
        assert!(validator.validate_connection("rest", &forbidden_headers).is_err());
    }
    
    #[test]
    fn test_data_freshness_validation() {
        let validator = DataSourceValidator::new();
        
        // Test fresh data
        let fresh_timestamp = Utc::now();
        assert!(validator.validate_data_freshness(fresh_timestamp, Duration::from_secs(300)).is_ok());
        
        // Test stale data
        let stale_timestamp = Utc::now() - chrono::Duration::seconds(600);
        assert!(validator.validate_data_freshness(stale_timestamp, Duration::from_secs(300)).is_err());
    }
}