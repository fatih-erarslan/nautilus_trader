//! Standalone validation tests that can run independently

use crate::validation::financial::{FinancialValidator, ValidationSeverity, AssetValidationRules};
use crate::types::Float;

/// Run comprehensive standalone validation tests
pub fn run_validation_tests() -> Result<(), String> {
    println!("🧪 Running Financial Validation Tests");
    println!("=====================================");

    test_basic_price_validation()?;
    test_volume_validation()?;
    test_flash_crash_detection()?;
    test_market_crash_scenarios()?;
    test_circuit_breaker()?;
    test_manipulation_detection()?;
    test_asset_specific_validation()?;
    test_ohlcv_validation()?;

    println!("✅ All validation tests passed!");
    Ok(())
}

fn test_basic_price_validation() -> Result<(), String> {
    println!("\n📊 Testing Basic Price Validation");
    
    let validator = FinancialValidator::new();
    
    // Valid prices
    let valid_prices = vec![100.0, 0.01, 1e6];
    for price in valid_prices {
        validator.validate_price(price, "stock")
            .map_err(|e| format!("Valid price {} rejected: {}", price, e))?;
    }
    
    // Invalid prices
    let invalid_prices = vec![
        (-10.0, "negative"),
        (0.0, "zero"),
        (Float::NAN, "NaN"),
        (Float::INFINITY, "infinity"),
        (1e20, "too large"),
    ];
    
    for (price, description) in invalid_prices {
        if validator.validate_price(price, "stock").is_ok() {
            return Err(format!("Invalid price {} ({}) was accepted", price, description));
        }
    }
    
    println!("  ✓ Basic price validation works correctly");
    Ok(())
}

fn test_volume_validation() -> Result<(), String> {
    println!("\n📈 Testing Volume Validation");
    
    let validator = FinancialValidator::new();
    
    // Valid volumes
    let valid_volumes = vec![0.0, 1000.0, 1e10];
    for volume in valid_volumes {
        validator.validate_volume(volume, "stock")
            .map_err(|e| format!("Valid volume {} rejected: {}", volume, e))?;
    }
    
    // Invalid volumes
    let invalid_volumes = vec![
        (-100.0, "negative"),
        (Float::NAN, "NaN"),
        (Float::INFINITY, "infinity"),
        (1e20, "too large"),
    ];
    
    for (volume, description) in invalid_volumes {
        if validator.validate_volume(volume, "stock").is_ok() {
            return Err(format!("Invalid volume {} ({}) was accepted", volume, description));
        }
    }
    
    println!("  ✓ Volume validation works correctly");
    Ok(())
}

fn test_flash_crash_detection() -> Result<(), String> {
    println!("\n💥 Testing Flash Crash Detection");
    
    let mut validator = FinancialValidator::new();
    
    // Normal price series (should pass)
    let normal_prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
    let report = validator.validate_price_series(&normal_prices, "stock");
    if report.flash_crashes_detected > 0 {
        return Err("Normal price series incorrectly flagged as flash crash".to_string());
    }
    
    // Flash crash series (should detect crash)
    let crash_prices = vec![100.0, 101.0, 5.0, 102.0]; // 95% drop
    let report = validator.validate_price_series(&crash_prices, "stock");
    if report.flash_crashes_detected == 0 {
        return Err("Flash crash not detected".to_string());
    }
    
    println!("  ✓ Flash crash detection works correctly");
    Ok(())
}

fn test_market_crash_scenarios() -> Result<(), String> {
    println!("\n📉 Testing Market Crash Scenarios");
    
    let mut validator = FinancialValidator::new();
    
    // Test various historical market crashes
    let scenarios = vec![
        (vec![2000.0, 1560.0], "Black Monday 1987 (22% drop)"),
        (vec![100.0, 95.0, 85.0, 70.0, 50.0], "Gradual bear market"),
        (vec![1100.0, 1000.0, 950.0, 1020.0], "Flash crash with recovery"),
    ];
    
    for (prices, description) in scenarios {
        let report = validator.validate_price_series(&prices, "stock");
        
        // Historical events should be valid (even if with warnings)
        let has_blocking_errors = report.critical_errors > 0 || 
                                (report.errors > 0 && !description.contains("Flash crash"));
        
        if has_blocking_errors {
            return Err(format!("Historical scenario '{}' incorrectly rejected", description));
        }
        
        println!("  ✓ {}: validated successfully", description);
    }
    
    Ok(())
}

fn test_circuit_breaker() -> Result<(), String> {
    println!("\n🔴 Testing Circuit Breaker");
    
    let mut validator = FinancialValidator::new();
    
    // Create data with multiple extreme outliers
    let mut extreme_prices = vec![100.0; 20];
    // Add extreme outliers
    extreme_prices[5] = 10000.0;   // 100x spike
    extreme_prices[10] = 50000.0;  // 500x spike
    extreme_prices[15] = 100000.0; // 1000x spike
    
    let report = validator.validate_price_series(&extreme_prices, "stock");
    
    // Circuit breaker should trigger for such extreme anomalies
    let has_circuit_breaker_issue = report.issues.iter()
        .any(|issue| issue.code == "CIRCUIT_BREAKER");
    
    if !has_circuit_breaker_issue {
        return Err("Circuit breaker did not trigger for extreme anomalies".to_string());
    }
    
    println!("  ✓ Circuit breaker triggered correctly");
    Ok(())
}

fn test_manipulation_detection() -> Result<(), String> {
    println!("\n🎭 Testing Manipulation Detection");
    
    let mut validator = FinancialValidator::new();
    
    // Test artificially stable prices
    let stable_prices: Vec<Float> = vec![100.0; 25];
    let report = validator.validate_price_series(&stable_prices, "stock");
    
    if report.manipulation_patterns_detected == 0 {
        return Err("Stable price manipulation not detected".to_string());
    }
    
    // Test artificial alternating pattern
    let alternating: Vec<Float> = (0..20)
        .map(|i| if i % 2 == 0 { 100.0 } else { 101.0 })
        .collect();
    let report = validator.validate_price_series(&alternating, "stock");
    
    if report.manipulation_patterns_detected == 0 {
        return Err("Alternating pattern manipulation not detected".to_string());
    }
    
    println!("  ✓ Manipulation detection works correctly");
    Ok(())
}

fn test_asset_specific_validation() -> Result<(), String> {
    println!("\n🏦 Testing Asset-Specific Validation");
    
    let validator = FinancialValidator::new();
    
    // Test that crypto allows smaller prices than stocks
    let micro_price = 1e-10;
    
    // Should pass for crypto
    if validator.validate_price(micro_price, "crypto").is_err() {
        return Err("Micro price incorrectly rejected for crypto".to_string());
    }
    
    // Should fail for stock
    if validator.validate_price(micro_price, "stock").is_ok() {
        return Err("Micro price incorrectly accepted for stock".to_string());
    }
    
    // Test forex-specific ranges
    let forex_price = 1.1234;
    if validator.validate_price(forex_price, "forex").is_err() {
        return Err("Valid forex price incorrectly rejected".to_string());
    }
    
    println!("  ✓ Asset-specific validation works correctly");
    Ok(())
}

fn test_ohlcv_validation() -> Result<(), String> {
    println!("\n📊 Testing OHLCV Validation");
    
    let mut validator = FinancialValidator::new();
    
    // Valid OHLCV data
    let timestamps = vec![1640995200000, 1640995260000, 1640995320000];
    let open = vec![100.0, 102.0, 104.0];
    let high = vec![105.0, 107.0, 109.0];
    let low = vec![98.0, 100.0, 102.0];
    let close = vec![102.0, 104.0, 106.0];
    let volume = vec![1000.0, 1100.0, 1200.0];
    
    let report = validator.validate_market_data(
        &timestamps, &open, &high, &low, &close, &volume, "stock"
    );
    
    if !report.passed() {
        return Err(format!("Valid OHLCV data rejected: {} errors", report.errors));
    }
    
    // Invalid OHLCV data (low > high)
    let invalid_high = vec![105.0, 107.0, 95.0]; // Last high < low
    let invalid_low = vec![98.0, 100.0, 102.0];
    
    let report = validator.validate_market_data(
        &timestamps, &open, &invalid_high, &invalid_low, &close, &volume, "stock"
    );
    
    if report.passed() {
        return Err("Invalid OHLCV data (low > high) was accepted".to_string());
    }
    
    println!("  ✓ OHLCV validation works correctly");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standalone_validation() {
        run_validation_tests().expect("Standalone validation tests failed");
    }
}

/// Main function for running standalone tests
pub fn main() {
    match run_validation_tests() {
        Ok(_) => {
            println!("\n🎉 All financial validation tests completed successfully!");
            std::process::exit(0);
        }
        Err(e) => {
            eprintln!("\n❌ Validation test failed: {}", e);
            std::process::exit(1);
        }
    }
}