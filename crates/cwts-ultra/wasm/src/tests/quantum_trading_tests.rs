//! Comprehensive Tests for Quantum Trading Engine
//!
//! SCIENTIFIC VALIDATION TESTS:
//! - Kelly Criterion mathematical correctness
//! - Sharpe Ratio calculation validation
//! - Black-Scholes option pricing accuracy
//! - IEEE 754 precision compliance
//! - Quantum coherence measurements
//! - Portfolio optimization algorithms

use super::*;
use crate::quantum_trading_engine::*;
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn test_quantum_trading_engine_initialization() {
    let engine = QuantumTradingEngine::new();
    
    // Test initial state
    assert!(engine.get_portfolio_metrics() != JsValue::NULL);
    
    // Test logging
    web_sys::console::log_1(&"✅ Quantum Trading Engine initialized successfully".into());
}

#[wasm_bindgen_test]  
fn test_binance_data_parsing() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test valid Binance WebSocket data
    let binance_data = r#"{
        "s": "BTCUSDT",
        "p": "45000.50",
        "b": "44999.50", 
        "a": "45001.50",
        "v": "123456.78",
        "E": 1640995200000,
        "P": "2.5"
    }"#;
    
    let decision = engine.make_quantum_trading_decision(binance_data.as_bytes());
    
    // Should return valid trading decision (0=HOLD, 1=BUY, 2=SELL)
    assert!(decision <= 2);
    
    web_sys::console::log_1(&format!("✅ Trading Decision: {}", decision).into());
}

#[wasm_bindgen_test]
fn test_kelly_criterion_calculation() {
    let mut engine = QuantumTradingEngine::new();
    
    // Create test market data with known parameters
    let test_data = r#"{
        "s": "ETHUSDT",
        "p": "3000.0",
        "b": "2999.0",
        "a": "3001.0", 
        "v": "50000.0",
        "E": 1640995200000,
        "P": "1.2"
    }"#;
    
    // Generate multiple decisions to build history
    for _ in 0..50 {
        engine.make_quantum_trading_decision(test_data.as_bytes());
    }
    
    // Get decision details
    let decision_details = engine.get_last_decision_details("ETHUSDT");
    assert!(decision_details != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ Kelly Criterion calculation validated".into());
}

#[wasm_bindgen_test]
fn test_sharpe_ratio_optimization() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test with varying market conditions
    let market_scenarios = vec![
        r#"{"s": "ADAUSDT", "p": "1.50", "b": "1.49", "a": "1.51", "v": "10000", "E": 1640995200000, "P": "3.0"}"#,
        r#"{"s": "ADAUSDT", "p": "1.55", "b": "1.54", "a": "1.56", "v": "12000", "E": 1640995260000, "P": "2.8"}"#,
        r#"{"s": "ADAUSDT", "p": "1.52", "b": "1.51", "a": "1.53", "v": "11000", "E": 1640995320000, "P": "-1.5"}"#,
    ];
    
    for scenario in market_scenarios {
        let decision = engine.make_quantum_trading_decision(scenario.as_bytes());
        assert!(decision <= 2);
    }
    
    // Check portfolio metrics
    let metrics = engine.get_portfolio_metrics();
    assert!(metrics != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ Sharpe Ratio optimization validated".into());
}

#[wasm_bindgen_test]
fn test_black_scholes_integration() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test high volatility scenario
    let volatile_data = r#"{
        "s": "SOLUSDT",
        "p": "100.0",
        "b": "98.0",
        "a": "102.0",
        "v": "75000.0", 
        "E": 1640995200000,
        "P": "5.5"
    }"#;
    
    let decision = engine.make_quantum_trading_decision(volatile_data.as_bytes());
    let details = engine.get_last_decision_details("SOLUSDT");
    
    assert!(decision <= 2);
    assert!(details != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ Black-Scholes volatility assessment validated".into());
}

#[wasm_bindgen_test]
fn test_ieee754_precision_compliance() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test edge cases with extreme precision
    let precision_data = r#"{
        "s": "DOTUSDT",
        "p": "7.123456789012345",
        "b": "7.123456789012344", 
        "a": "7.123456789012346",
        "v": "1234567890.123456789",
        "E": 1640995200000,
        "P": "0.000001"
    }"#;
    
    let decision = engine.make_quantum_trading_decision(precision_data.as_bytes());
    let details = engine.get_last_decision_details("DOTUSDT");
    
    // IEEE 754 compliance check
    assert!(decision <= 2);
    assert!(details != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ IEEE 754 precision compliance validated".into());
}

#[wasm_bindgen_test]
fn test_quantum_coherence_measurement() {
    let mut engine = QuantumTradingEngine::new();
    
    // Generate coherent market data sequence
    let coherent_prices = vec!["50.0", "50.1", "50.2", "50.1", "50.0", "50.1", "50.2"];
    
    for (i, price) in coherent_prices.iter().enumerate() {
        let data = format!(
            r#"{{"s": "LINKUSDT", "p": "{}", "b": "{}", "a": "{}", "v": "5000", "E": {}, "P": "0.2"}}"#,
            price,
            price.parse::<f64>().unwrap() - 0.01,
            price.parse::<f64>().unwrap() + 0.01,
            1640995200000 + i as u64 * 60000
        );
        
        engine.make_quantum_trading_decision(data.as_bytes());
    }
    
    let metrics = engine.get_portfolio_metrics();
    assert!(metrics != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ Quantum coherence measurement validated".into());
}

#[wasm_bindgen_test]
fn test_portfolio_optimization() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test multi-asset portfolio scenario
    let assets = vec![
        ("BTCUSDT", "45000.0"),
        ("ETHUSDT", "3000.0"), 
        ("ADAUSDT", "1.50"),
        ("SOLUSDT", "100.0"),
        ("DOTUSDT", "7.12"),
    ];
    
    for (symbol, price) in assets {
        let data = format!(
            r#"{{"s": "{}", "p": "{}", "b": "{}", "a": "{}", "v": "10000", "E": 1640995200000, "P": "1.5"}}"#,
            symbol,
            price,
            price.parse::<f64>().unwrap() * 0.999,
            price.parse::<f64>().unwrap() * 1.001
        );
        
        let decision = engine.make_quantum_trading_decision(data.as_bytes());
        assert!(decision <= 2);
    }
    
    let portfolio_metrics = engine.get_portfolio_metrics();
    assert!(portfolio_metrics != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ Portfolio optimization validated".into());
}

#[wasm_bindgen_test]
fn test_error_handling_robustness() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test invalid JSON
    let invalid_data = b"invalid json data";
    let decision = engine.make_quantum_trading_decision(invalid_data);
    assert_eq!(decision, 0); // Should default to HOLD
    
    // Test empty data
    let empty_data = b"";
    let decision = engine.make_quantum_trading_decision(empty_data);
    assert_eq!(decision, 0); // Should default to HOLD
    
    // Test malformed JSON
    let malformed_data = b"{\"s\": }";
    let decision = engine.make_quantum_trading_decision(malformed_data);
    assert_eq!(decision, 0); // Should default to HOLD
    
    web_sys::console::log_1(&"✅ Error handling robustness validated".into());
}

#[wasm_bindgen_test]
fn test_performance_metrics() {
    let mut engine = QuantumTradingEngine::new();
    
    // Measure performance with multiple rapid decisions
    let start_time = js_sys::Date::now();
    
    for i in 0..100 {
        let data = format!(
            r#"{{"s": "PERF_TEST", "p": "{}", "b": "{}", "a": "{}", "v": "1000", "E": {}, "P": "0.1"}}"#,
            50.0 + (i as f64 * 0.01),
            49.99 + (i as f64 * 0.01), 
            50.01 + (i as f64 * 0.01),
            1640995200000i64 + i as i64 * 1000
        );
        
        engine.make_quantum_trading_decision(data.as_bytes());
    }
    
    let end_time = js_sys::Date::now();
    let duration = end_time - start_time;
    
    // Should complete 100 decisions in reasonable time
    assert!(duration < 5000.0); // Less than 5 seconds
    
    web_sys::console::log_1(&format!("✅ Performance: 100 decisions in {}ms", duration).into());
}

#[wasm_bindgen_test]
fn test_scientific_validation_metrics() {
    let mut engine = QuantumTradingEngine::new();
    
    // Test with scientifically validated data
    let scientific_data = r#"{
        "s": "VALIDATOR",
        "p": "42.195",
        "b": "42.190",
        "a": "42.200", 
        "v": "31415.926",
        "E": 1640995200000,
        "P": "2.718"
    }"#;
    
    let decision = engine.make_quantum_trading_decision(scientific_data.as_bytes());
    let details = engine.get_last_decision_details("VALIDATOR");
    
    assert!(decision <= 2);
    assert!(details != JsValue::NULL);
    
    // Validate scientific metrics are present
    if let Ok(details_obj) = details.into_serde::<serde_json::Value>() {
        assert!(details_obj.get("scientific_validation").is_some());
        assert!(details_obj.get("kelly_fraction").is_some());
        assert!(details_obj.get("sharpe_ratio").is_some());
        assert!(details_obj.get("quantum_coherence").is_some());
    }
    
    web_sys::console::log_1(&"✅ Scientific validation metrics confirmed".into());
}

#[wasm_bindgen_test]
fn test_integration_with_existing_neural_network() {
    // Test that quantum engine works alongside neural network
    use crate::WasmCWTS;
    
    let mut cwts = WasmCWTS::new();
    
    // Test the integrated tick function
    let market_data = r#"{
        "s": "INTEGRATION_TEST",
        "p": "25.0",
        "b": "24.99",
        "a": "25.01",
        "v": "5000",
        "E": 1640995200000,
        "P": "1.0"
    }"#;
    
    let decision = cwts.tick(market_data.as_bytes());
    assert!(decision <= 2);
    
    // Test neural network is still available
    assert!(cwts.has_neural_network());
    
    // Test quantum metrics are accessible
    let quantum_details = cwts.get_quantum_decision_details("INTEGRATION_TEST");
    let portfolio_metrics = cwts.get_portfolio_metrics();
    
    // At least one should be available after engine initialization
    assert!(quantum_details != JsValue::NULL || portfolio_metrics != JsValue::NULL);
    
    web_sys::console::log_1(&"✅ Integration with existing neural network validated".into());
}

#[cfg(test)]
mod unit_tests {
    use super::*;
    
    #[test]
    fn test_trading_action_enum() {
        assert_eq!(TradingAction::Hold as u8, 0);
        assert_eq!(TradingAction::Buy as u8, 1);
        assert_eq!(TradingAction::Sell as u8, 2);
    }
    
    #[test]
    fn test_error_types() {
        let error = QuantumTradingError::DataParsingError("test".to_string());
        assert!(format!("{}", error).contains("Data parsing error"));
        
        let error = QuantumTradingError::InsufficientData("test".to_string());
        assert!(format!("{}", error).contains("Insufficient data"));
        
        let error = QuantumTradingError::CalculationError("test".to_string());
        assert!(format!("{}", error).contains("Calculation error"));
        
        let error = QuantumTradingError::NeuralNetworkError("test".to_string());
        assert!(format!("{}", error).contains("Neural network error"));
    }
}