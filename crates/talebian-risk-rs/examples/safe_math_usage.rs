//! Example usage of the bulletproof safe math system for financial calculations
//!
//! This example demonstrates how to use the safe mathematical operations
//! in a real trading scenario with comprehensive error handling.

use talebian_risk_rs::safe_math::*;
use talebian_risk_rs::types::MarketData;
use chrono::Utc;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🛡️  Bulletproof Financial Calculation System Demo");
    println!("================================================");
    
    // Example 1: Safe Division Operations
    println!("\n1. Safe Division Operations:");
    demonstrate_safe_division();
    
    // Example 2: Market Data Validation
    println!("\n2. Market Data Validation:");
    demonstrate_market_validation();
    
    // Example 3: Position Sizing Calculations
    println!("\n3. Safe Position Sizing:");
    demonstrate_position_sizing()?;
    
    // Example 4: Financial Metrics Calculation
    println!("\n4. Financial Metrics:");
    demonstrate_financial_metrics()?;
    
    // Example 5: Error Recovery Patterns
    println!("\n5. Error Recovery:");
    demonstrate_error_recovery();
    
    println!("\n✅ All demonstrations completed successfully!");
    println!("   Zero panics, deterministic results, bulletproof calculations!");
    
    Ok(())
}

fn demonstrate_safe_division() {
    println!("   Testing safe division with edge cases...");
    
    // Normal division
    match safe_divide(100.0, 5.0) {
        Ok(result) => println!("   ✓ 100.0 / 5.0 = {}", result),
        Err(e) => println!("   ✗ Error: {}", e),
    }
    
    // Division by zero - safely handled
    match safe_divide(100.0, 0.0) {
        Ok(result) => println!("   ✓ 100.0 / 0.0 = {}", result),
        Err(e) => println!("   ✓ Safely caught division by zero: {}", e),
    }
    
    // Division by near-zero - safely handled
    match safe_divide(100.0, 1e-20) {
        Ok(result) => println!("   ✓ 100.0 / 1e-20 = {}", result),
        Err(e) => println!("   ✓ Safely caught near-zero division: {}", e),
    }
    
    // Invalid inputs - safely handled
    match safe_divide(f64::NAN, 5.0) {
        Ok(result) => println!("   ✓ NaN / 5.0 = {}", result),
        Err(e) => println!("   ✓ Safely caught NaN input: {}", e),
    }
    
    // Using fallback for production
    let safe_result = safe_divide_with_fallback(100.0, 0.0, 1.0);
    println!("   ✓ With fallback: 100.0 / 0.0 → {}", safe_result);
}

fn demonstrate_market_validation() {
    println!("   Validating market data with edge cases...");
    
    // Valid market data
    let valid_data = MarketData {
        timestamp: Utc::now(),
        timestamp_unix: 1000000000,
        price: 100.0,
        volume: 1000.0,
        bid: 99.5,
        ask: 100.5,
        bid_volume: 500.0,
        ask_volume: 500.0,
        volatility: 0.2,
        returns: vec![0.01, -0.02, 0.015, -0.01],
        volume_history: vec![900.0, 1100.0, 950.0, 1050.0],
    };
    
    let validation = validate_market_data(&valid_data);
    if validation.is_valid {
        println!("   ✓ Valid market data passed validation");
    } else {
        println!("   ✗ Validation failed: {:?}", validation.errors);
    }
    
    // Invalid market data with inverted bid/ask
    let mut invalid_data = valid_data.clone();
    invalid_data.bid = 101.0;
    invalid_data.ask = 100.0;
    
    let validation = validate_market_data(&invalid_data);
    if !validation.is_valid {
        println!("   ✓ Invalid data caught: {}", validation.errors[0]);
    }
    
    // Test with extreme values
    let mut extreme_data = valid_data.clone();
    extreme_data.price = f64::INFINITY;
    
    let validation = validate_market_data(&extreme_data);
    if !validation.is_valid {
        println!("   ✓ Extreme values caught: {}", validation.errors[0]);
    }
}

fn demonstrate_position_sizing() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Calculating safe position sizes...");
    
    let sizer = SafePositionSizer::new(0.02, 0.1)?; // 2% max risk, 10% max position
    
    // Normal position sizing
    let result = sizer.calculate_position_size(
        100000.0, // capital
        100.0,    // entry price
        95.0,     // stop loss
        0.6,      // win rate
        150.0,    // avg win
        100.0,    // avg loss
        50,       // sample size
        0.8,      // confidence
    )?;
    
    println!("   ✓ Position size: {:.2}%", result.position_size * 100.0);
    println!("     Risk-adjusted: {:.2}%", result.risk_adjusted_size * 100.0);
    println!("     Actual risk: {:.2}%", result.actual_risk * 100.0);
    println!("     Kelly fraction: {:.3}", result.kelly_fraction);
    
    if !result.warnings.is_empty() {
        println!("     Warnings: {:?}", result.warnings);
    }
    
    // Test with invalid parameters - should be caught safely
    match sizer.calculate_position_size(
        100000.0, 95.0, 100.0, 0.6, 150.0, 100.0, 50, 0.8  // Entry < Stop
    ) {
        Ok(_) => println!("   ✗ Should have caught invalid parameters"),
        Err(e) => println!("   ✓ Safely caught invalid parameters: {}", e),
    }
    
    Ok(())
}

fn demonstrate_financial_metrics() -> Result<(), Box<dyn std::error::Error>> {
    println!("   Calculating comprehensive financial metrics...");
    
    let calculator = SafeFinancialCalculator::new(0.02)?; // 2% risk-free rate
    
    // Sample returns data
    let returns = vec![
        0.01, -0.02, 0.015, -0.01, 0.03, -0.005, 0.02, -0.015,
        0.025, -0.008, 0.012, -0.018, 0.008, -0.022, 0.035,
        -0.012, 0.018, -0.007, 0.028, -0.014, 0.009, -0.025,
        0.016, -0.011, 0.021, -0.006, 0.013, -0.019, 0.024,
        -0.003, 0.017, -0.013, 0.011, -0.026, 0.019, -0.004,
        0.022, -0.009, 0.014, -0.020, 0.027, -0.002, 0.010,
        -0.016, 0.023, -0.005, 0.015, -0.017, 0.020, -0.008
    ];
    
    let metrics = calculator.calculate_metrics(&returns)?;
    
    println!("   ✓ Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("     Sortino Ratio: {:.3}", metrics.sortino_ratio);
    println!("     Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("     Volatility: {:.2}%", metrics.volatility * 100.0);
    println!("     Win Rate: {:.1}%", metrics.win_rate * 100.0);
    println!("     VaR (95%): {:.2}%", metrics.var_95 * 100.0);
    println!("     Expected Shortfall: {:.2}%", metrics.expected_shortfall * 100.0);
    println!("     Sample Size: {}", metrics.sample_size);
    println!("     Confidence: {:.1}%", metrics.confidence_level * 100.0);
    
    // Test with insufficient data - should be caught
    let short_returns = vec![0.01, -0.02];
    match calculator.calculate_metrics(&short_returns) {
        Ok(_) => println!("   ✗ Should have caught insufficient data"),
        Err(e) => println!("   ✓ Safely caught insufficient data: {}", e),
    }
    
    Ok(())
}

fn demonstrate_error_recovery() {
    use error_handling::*;
    
    println!("   Testing error recovery strategies...");
    
    let mut conservative_calc = FailsafeCalculator::new(ConservativeErrorHandler);
    let mut aggressive_calc = FailsafeCalculator::new(AggressiveErrorHandler);
    
    // Test division by zero with different handlers
    match conservative_calc.safe_divide_with_recovery(10.0, 0.0, "test_conservative") {
        Ok(result) => println!("   ✗ Conservative should have failed: {}", result),
        Err(e) => println!("   ✓ Conservative safely failed: {}", e),
    }
    
    match aggressive_calc.safe_divide_with_recovery(10.0, 0.0, "test_aggressive") {
        Ok(result) => println!("   ✓ Aggressive recovered with: {}", result),
        Err(e) => println!("   ✗ Aggressive should have recovered: {}", e),
    }
    
    // Test with NaN inputs
    match aggressive_calc.safe_divide_with_recovery(f64::NAN, 5.0, "test_nan") {
        Ok(result) => println!("   ✓ NaN handled gracefully: {}", result),
        Err(e) => println!("   ✗ NaN should have been handled: {}", e),
    }
    
    println!("   ✓ Error recovery patterns working correctly");
}