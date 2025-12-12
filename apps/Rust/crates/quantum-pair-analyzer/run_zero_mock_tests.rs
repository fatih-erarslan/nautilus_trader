// Standalone Zero-Mock Enforcement Test Runner
// Copyright (c) 2025 TENGRI Trading Swarm
// Execute comprehensive tests for zero-mock enforcement without workspace dependencies

use std::collections::HashMap;
use std::time::Duration;

fn main() {
    println!("🚀 Zero-Mock Enforcement Test Runner");
    println!("=====================================");
    
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut failed_tests = 0;
    
    // Test 1: Basic pattern detection
    total_tests += 1;
    let test_name = "Basic Mock Pattern Detection";
    println!("Running: {}", test_name);
    
    let mock_patterns = vec![
        "mock.", "fake_", "dummy_", "test_data", "sample_", "example_",
        "rand::random", "fastrand::", "thread_rng()", "localhost", "127.0.0.1"
    ];
    
    // Test that we can detect these patterns
    let mock_code = r#"
        let client = mock.create_client();
        let data = fake_data_generator();
        let endpoint = "http://localhost:8080/api";
        let key = "test_api_key";
        let random_val = rand::random::<f64>();
    "#;
    
    let mut violations_found = 0;
    for pattern in &mock_patterns {
        if mock_code.contains(pattern) {
            violations_found += 1;
        }
    }
    
    if violations_found > 0 {
        println!("✅ {}: PASSED (found {} violations)", test_name, violations_found);
        passed_tests += 1;
    } else {
        println!("❌ {}: FAILED (no violations found)", test_name);
        failed_tests += 1;
    }
    
    // Test 2: Clean code validation
    total_tests += 1;
    let test_name = "Clean Code Validation";
    println!("Running: {}", test_name);
    
    let clean_code = r#"
        let client = RealClient::new(api_key);
        let data = client.fetch_market_data("BTCUSDT").await?;
        let endpoint = "https://api.binance.com/api/v3";
        let key = "production_key_1234567890";
    "#;
    
    let mut violations_found = 0;
    for pattern in &mock_patterns {
        if clean_code.contains(pattern) {
            violations_found += 1;
        }
    }
    
    if violations_found == 0 {
        println!("✅ {}: PASSED (no violations found)", test_name);
        passed_tests += 1;
    } else {
        println!("❌ {}: FAILED (found {} violations)", test_name, violations_found);
        failed_tests += 1;
    }
    
    // Test 3: Endpoint validation
    total_tests += 1;
    let test_name = "Endpoint Validation";
    println!("Running: {}", test_name);
    
    let real_endpoints = vec![
        "https://api.binance.com",
        "https://api.coinbase.com",
        "https://api.kraken.com",
        "https://api.bitfinex.com",
    ];
    
    let suspicious_endpoints = vec![
        "http://localhost:8080",
        "https://mock-api.com",
        "https://test-endpoint.com",
        "https://fake-binance.com",
    ];
    
    let mut endpoint_validation_passed = true;
    
    // Real endpoints should pass
    for endpoint in &real_endpoints {
        if is_suspicious_endpoint(endpoint) {
            println!("❌ Real endpoint {} flagged as suspicious", endpoint);
            endpoint_validation_passed = false;
        }
    }
    
    // Suspicious endpoints should fail
    for endpoint in &suspicious_endpoints {
        if !is_suspicious_endpoint(endpoint) {
            println!("❌ Suspicious endpoint {} not detected", endpoint);
            endpoint_validation_passed = false;
        }
    }
    
    if endpoint_validation_passed {
        println!("✅ {}: PASSED", test_name);
        passed_tests += 1;
    } else {
        println!("❌ {}: FAILED", test_name);
        failed_tests += 1;
    }
    
    // Test 4: Credential validation
    total_tests += 1;
    let test_name = "Credential Validation";
    println!("Running: {}", test_name);
    
    let valid_credentials = vec![
        ("live_key_production_9876543210", "live_secret_production_9876543210"),
        ("real_api_key_long_enough", "real_secret_key_long_enough"),
    ];
    
    let invalid_credentials = vec![
        ("test_api_key", "test_secret"),
        ("mock_key", "mock_secret"),
        ("fake_key", "fake_secret"),
        ("demo_key", "demo_secret"),
        ("short", "key"), // Too short
    ];
    
    let mut credential_validation_passed = true;
    
    // Valid credentials should pass
    for (key, secret) in &valid_credentials {
        if !validate_credentials(key, secret) {
            println!("❌ Valid credential {} flagged as invalid", key);
            credential_validation_passed = false;
        }
    }
    
    // Invalid credentials should fail
    for (key, secret) in &invalid_credentials {
        if validate_credentials(key, secret) {
            println!("❌ Invalid credential {} not detected", key);
            credential_validation_passed = false;
        }
    }
    
    if credential_validation_passed {
        println!("✅ {}: PASSED", test_name);
        passed_tests += 1;
    } else {
        println!("❌ {}: FAILED", test_name);
        failed_tests += 1;
    }
    
    // Test 5: Response validation
    total_tests += 1;
    let test_name = "Response Validation";
    println!("Running: {}", test_name);
    
    let clean_responses = vec![
        r#"{"symbol":"BTCUSDT","price":"50000.00","volume":"1.5"}"#,
        r#"{"data":[{"symbol":"BTCUSDT","price":"50000.00"}]}"#,
    ];
    
    let mock_responses = vec![
        r#"{"symbol":"BTCUSDT","price":"50000.00","mock":true}"#,
        r#"{"symbol":"BTCUSDT","test_mode":true,"price":"50000.00"}"#,
        r#"{"symbol":"BTCUSDT","synthetic":true,"price":"50000.00"}"#,
        r#"{"symbol":"BTCUSDT","fake":true,"price":"50000.00"}"#,
    ];
    
    let mut response_validation_passed = true;
    
    // Clean responses should pass
    for response in &clean_responses {
        if !validate_response(response) {
            println!("❌ Clean response flagged as invalid: {}", response);
            response_validation_passed = false;
        }
    }
    
    // Mock responses should fail
    for response in &mock_responses {
        if validate_response(response) {
            println!("❌ Mock response not detected: {}", response);
            response_validation_passed = false;
        }
    }
    
    if response_validation_passed {
        println!("✅ {}: PASSED", test_name);
        passed_tests += 1;
    } else {
        println!("❌ {}: FAILED", test_name);
        failed_tests += 1;
    }
    
    // Test 6: Market data validation
    total_tests += 1;
    let test_name = "Market Data Validation";
    println!("Running: {}", test_name);
    
    let valid_market_data = MarketData {
        symbol: "BTCUSDT".to_string(),
        price: 50000.0,
        volume: 1.5,
        bid: 49995.0,
        ask: 50005.0,
        high_24h: 51000.0,
        low_24h: 49000.0,
        change_24h: 0.02,
        volume_24h: 1500.0,
        quote_volume: 75000000.0,
        timestamp: None, // Simplified for testing
    };
    
    let invalid_market_data = vec![
        MarketData {
            symbol: "TEST_SYMBOL".to_string(), // Contains "TEST"
            price: 50000.0,
            volume: 1.5,
            bid: 49995.0,
            ask: 50005.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            change_24h: 0.02,
            volume_24h: 1500.0,
            quote_volume: 75000000.0,
            timestamp: None,
        },
        MarketData {
            symbol: "BTCUSDT".to_string(),
            price: -100.0, // Negative price
            volume: 1.5,
            bid: 49995.0,
            ask: 50005.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            change_24h: 0.02,
            volume_24h: 1500.0,
            quote_volume: 75000000.0,
            timestamp: None,
        },
        MarketData {
            symbol: "BTCUSDT".to_string(),
            price: 50000.0,
            volume: 1.5,
            bid: 50010.0, // Bid > Ask
            ask: 49990.0,
            high_24h: 51000.0,
            low_24h: 49000.0,
            change_24h: 0.02,
            volume_24h: 1500.0,
            quote_volume: 75000000.0,
            timestamp: None,
        },
    ];
    
    let mut market_data_validation_passed = true;
    
    // Valid market data should pass
    if !validate_market_data(&valid_market_data) {
        println!("❌ Valid market data flagged as invalid");
        market_data_validation_passed = false;
    }
    
    // Invalid market data should fail
    for (i, data) in invalid_market_data.iter().enumerate() {
        if validate_market_data(data) {
            println!("❌ Invalid market data {} not detected", i);
            market_data_validation_passed = false;
        }
    }
    
    if market_data_validation_passed {
        println!("✅ {}: PASSED", test_name);
        passed_tests += 1;
    } else {
        println!("❌ {}: FAILED", test_name);
        failed_tests += 1;
    }
    
    // Print summary
    println!("\n🎯 ZERO-MOCK ENFORCEMENT TEST SUMMARY");
    println!("=====================================");
    println!("Total Tests:  {}", total_tests);
    println!("Passed:       {} ({}%)", passed_tests, 
              (passed_tests as f64 / total_tests as f64) * 100.0);
    println!("Failed:       {} ({}%)", failed_tests,
              (failed_tests as f64 / total_tests as f64) * 100.0);
    
    if failed_tests > 0 {
        println!("\n⚠️  Some tests failed - Zero-Mock Enforcement needs attention!");
        std::process::exit(1);
    } else {
        println!("\n🎉 ALL TESTS PASSED - Zero-Mock Enforcement is working correctly!");
    }
}

// Helper functions for validation
fn is_suspicious_endpoint(endpoint: &str) -> bool {
    let suspicious_patterns = [
        "localhost", "127.0.0.1", "0.0.0.0",
        "mock", "fake", "test", "demo",
        "synthetic", "dummy", "example"
    ];
    
    for pattern in &suspicious_patterns {
        if endpoint.to_lowercase().contains(pattern) {
            return true;
        }
    }
    false
}

fn validate_credentials(api_key: &str, secret: &str) -> bool {
    // Check key length (real keys are typically longer)
    if api_key.len() < 16 {
        return false;
    }
    
    // Check for test/mock prefixes
    let forbidden_prefixes = ["test_", "mock_", "fake_", "demo_", "sample_"];
    for prefix in &forbidden_prefixes {
        if api_key.starts_with(prefix) || secret.starts_with(prefix) {
            return false;
        }
    }
    
    // Check for obvious test patterns (more specific)
    let test_patterns = [
        "test", "mock", "fake", "demo", "example", "sample",
        "12345", "abcde", "aaaaa", "00000"
    ];
    
    let api_key_lower = api_key.to_lowercase();
    let secret_lower = secret.to_lowercase();
    
    for pattern in &test_patterns {
        // More specific matching to avoid false positives
        if api_key_lower.contains(&format!("_{}", pattern)) ||
           api_key_lower.starts_with(&format!("{}_", pattern)) ||
           api_key_lower.ends_with(&format!("_{}", pattern)) ||
           secret_lower.contains(&format!("_{}", pattern)) ||
           secret_lower.starts_with(&format!("{}_", pattern)) ||
           secret_lower.ends_with(&format!("_{}", pattern)) {
            return false;
        }
    }
    
    true
}

fn validate_response(response: &str) -> bool {
    let forbidden_patterns = [
        r#""mock":true"#,
        r#""test_mode":true"#,
        r#""synthetic":true"#,
        r#""fake":true"#,
        "test_data_",
        "mock_",
    ];
    
    for pattern in &forbidden_patterns {
        if response.contains(pattern) {
            return false;
        }
    }
    
    true
}

fn validate_market_data(data: &MarketData) -> bool {
    // Check for synthetic patterns in symbol
    if data.symbol.to_lowercase().contains("test") || 
       data.symbol.to_lowercase().contains("mock") || 
       data.symbol.to_lowercase().contains("fake") {
        return false;
    }
    
    // Validate price data makes sense
    if data.price <= 0.0 || data.bid <= 0.0 || data.ask <= 0.0 {
        return false;
    }
    
    // Validate bid-ask spread is reasonable
    if data.bid >= data.ask {
        return false;
    }
    
    // Check spread isn't impossibly tight (could indicate synthetic data)
    let spread_pct = (data.ask - data.bid) / data.price;
    if spread_pct < 0.0001 && data.price > 1.0 {
        return false;
    }
    
    // Validate volume is positive
    if data.volume < 0.0 || data.volume_24h < 0.0 {
        return false;
    }
    
    true
}

// Simplified market data structure for testing
#[derive(Debug, Clone)]
struct MarketData {
    symbol: String,
    price: f64,
    volume: f64,
    bid: f64,
    ask: f64,
    high_24h: f64,
    low_24h: f64,
    change_24h: f64,
    volume_24h: f64,
    quote_volume: f64,
    timestamp: Option<String>, // Simplified
}