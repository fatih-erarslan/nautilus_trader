//! Fibonacci Analyzer Example
//!
//! This example demonstrates the core functionality of the CDFA Fibonacci Analyzer,
//! including configuration, basic analysis, swing point detection, and performance
//! monitoring. The analyzer provides sub-microsecond performance with SIMD acceleration
//! for high-frequency trading applications.
//!
//! ## Features Demonstrated
//! - Basic Fibonacci analysis with default configuration
//! - Custom configuration with different trading strategies
//! - Swing point detection and analysis
//! - Retracement and extension level calculations
//! - Multi-timeframe analysis
//! - Performance optimization and monitoring
//! - Error handling and validation
//!
//! ## Usage
//! ```bash
//! cargo run --example fibonacci_analysis --features simd
//! ```

use std::collections::HashMap;
use std::time::Instant;
use cdfa_fibonacci_analyzer::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    println!("🧮 CDFA Fibonacci Analyzer Example");
    println!("=====================================");
    
    // Generate sample market data
    let sample_data = generate_sample_market_data();
    
    // Run various analysis examples
    run_basic_analysis(&sample_data)?;
    run_custom_configuration_analysis(&sample_data)?;
    run_enhanced_analysis(&sample_data)?;
    run_performance_benchmark(&sample_data)?;
    run_multi_timeframe_analysis(&sample_data)?;
    run_error_handling_examples()?;
    
    println!("\n✅ All examples completed successfully!");
    Ok(())
}

/// Generate realistic sample market data for testing
fn generate_sample_market_data() -> (Vec<f64>, Vec<f64>) {
    println!("\n📊 Generating sample market data...");
    
    let mut prices = Vec::new();
    let mut volumes = Vec::new();
    
    // Generate trending price data with volatility
    let base_price = 100.0;
    let trend_factor = 0.001;
    let volatility = 0.02;
    
    for i in 0..200 {
        let trend = base_price + (i as f64 * trend_factor);
        let noise = (i as f64 * 0.1).sin() * volatility * base_price;
        let price = trend + noise;
        
        // Add some swing points
        let swing_adjustment = if i % 25 == 0 {
            if i % 50 == 0 { 2.0 } else { -1.5 }
        } else {
            0.0
        };
        
        prices.push(price + swing_adjustment);
        volumes.push(1000.0 + (i as f64 * 10.0).sin().abs() * 500.0);
    }
    
    println!("   Generated {} price points with {} volume points", prices.len(), volumes.len());
    println!("   Price range: {:.2} - {:.2}", 
             prices.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
             prices.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)));
    
    (prices, volumes)
}

/// Basic Fibonacci analysis example
fn run_basic_analysis(data: &(Vec<f64>, Vec<f64>)) -> Result<(), FibonacciError> {
    println!("\n🔍 Basic Fibonacci Analysis");
    println!("---------------------------");
    
    let (prices, volumes) = data;
    
    // Create analyzer with default configuration
    let analyzer = FibonacciAnalyzer::default();
    
    // Run analysis
    let start = Instant::now();
    let result = analyzer.analyze(prices, volumes)?;
    let duration = start.elapsed();
    
    // Display results
    println!("   Analysis completed in: {:?}", duration);
    println!("   Signal: {:.4}", result.signal.value());
    println!("   Confidence: {:.4}", result.confidence.value());
    println!("   Analysis type: {}", result.metadata.get("analysis_type").unwrap_or(&"unknown".to_string()));
    println!("   Data points: {}", result.metadata.get("data_points").unwrap_or(&"0".to_string()));
    println!("   Current price: ${}", result.metadata.get("current_price").unwrap_or(&"0".to_string()));
    println!("   Fibonacci alignment: {}", result.metadata.get("fibonacci_alignment").unwrap_or(&"0".to_string()));
    println!("   Swing highs: {}", result.metadata.get("swing_highs").unwrap_or(&"0".to_string()));
    println!("   Swing lows: {}", result.metadata.get("swing_lows").unwrap_or(&"0".to_string()));
    
    // Performance metrics
    let analysis_time_us = result.metadata.get("analysis_time_us")
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0);
    println!("   Analysis time: {}μs", analysis_time_us);
    
    Ok(())
}

/// Custom configuration analysis example
fn run_custom_configuration_analysis(data: &(Vec<f64>, Vec<f64>)) -> Result<(), FibonacciError> {
    println!("\n⚙️ Custom Configuration Analysis");
    println!("-------------------------------");
    
    let (prices, volumes) = data;
    
    // Test different trading strategy configurations
    let strategies = vec![
        ("Scalping", FibonacciPresets::scalping()),
        ("Day Trading", FibonacciPresets::day_trading()),
        ("Swing Trading", FibonacciPresets::swing_trading()),
        ("Position Trading", FibonacciPresets::position_trading()),
        ("High Precision", FibonacciPresets::high_precision()),
    ];
    
    for (name, config) in strategies {
        println!("\n   📈 {} Strategy:", name);
        
        // Validate configuration
        if let Err(e) = config.validate() {
            println!("      ❌ Configuration validation failed: {}", e);
            continue;
        }
        
        let analyzer = FibonacciAnalyzer::new(config);
        let start = Instant::now();
        let result = analyzer.analyze(prices, volumes)?;
        let duration = start.elapsed();
        
        println!("      Signal: {:.4}", result.signal.value());
        println!("      Confidence: {:.4}", result.confidence.value());
        println!("      Analysis time: {:?}", duration);
        
        // Show configuration details
        let config = analyzer.config();
        println!("      Swing period: {}", config.swing_period);
        println!("      Alignment tolerance: {:.4}", config.alignment_tolerance);
        println!("      ATR period: {}", config.atr_period);
        println!("      Retracement levels: {}", config.retracement_levels.len());
        println!("      Extension levels: {}", config.extension_levels.len());
    }
    
    Ok(())
}

/// Enhanced Fibonacci analysis example with detailed results
fn run_enhanced_analysis(data: &(Vec<f64>, Vec<f64>)) -> Result<(), FibonacciError> {
    println!("\n🚀 Enhanced Fibonacci Analysis");
    println!("------------------------------");
    
    let (prices, volumes) = data;
    
    // Create enhanced analyzer with custom parameters
    let mut custom_retracements = HashMap::new();
    custom_retracements.insert("0.0".to_string(), 0.0);
    custom_retracements.insert("23.6".to_string(), 0.236);
    custom_retracements.insert("38.2".to_string(), 0.382);
    custom_retracements.insert("50.0".to_string(), 0.5);
    custom_retracements.insert("61.8".to_string(), 0.618);
    custom_retracements.insert("78.6".to_string(), 0.786);
    custom_retracements.insert("100.0".to_string(), 1.0);
    
    let config = FibonacciConfig::default()
        .with_custom_retracements(custom_retracements)
        .with_swing_period(10)
        .with_alignment_tolerance(0.005)
        .with_atr_period(10)
        .with_simd(true)
        .with_parallel(true);
    
    let analyzer = FibonacciAnalyzer::new(config);
    
    // Run analysis
    let result = analyzer.analyze(prices, volumes)?;
    
    println!("   Enhanced Analysis Results:");
    println!("   Signal: {:.4}", result.signal.value());
    println!("   Confidence: {:.4}", result.confidence.value());
    
    // Show cache statistics
    if let Some(cache_stats) = analyzer.cache_stats() {
        println!("   Cache Statistics:");
        println!("     Size: {}", cache_stats.size);
        println!("     Hits: {}", cache_stats.hits);
        println!("     Misses: {}", cache_stats.misses);
        println!("     Hit rate: {:.2}%", cache_stats.hit_rate * 100.0);
    }
    
    // Test cache clearing
    analyzer.clear_cache();
    println!("   Cache cleared successfully");
    
    Ok(())
}

/// Performance benchmark example
fn run_performance_benchmark(data: &(Vec<f64>, Vec<f64>)) -> Result<(), FibonacciError> {
    println!("\n⚡ Performance Benchmark");
    println!("------------------------");
    
    let (prices, volumes) = data;
    
    // Test different configurations for performance
    let configs = vec![
        ("Default", FibonacciConfig::default()),
        ("High Frequency", FibonacciConfig::high_frequency()),
        ("SIMD Disabled", FibonacciConfig::default().with_simd(false)),
        ("Parallel Disabled", FibonacciConfig::default().with_parallel(false)),
        ("Minimal", FibonacciConfig::minimal()),
    ];
    
    for (name, config) in configs {
        let analyzer = FibonacciAnalyzer::new(config);
        
        // Warm up
        let _ = analyzer.analyze(prices, volumes)?;
        
        // Benchmark multiple runs
        let iterations = 100;
        let start = Instant::now();
        
        for _ in 0..iterations {
            let _ = analyzer.analyze(prices, volumes)?;
        }
        
        let total_duration = start.elapsed();
        let avg_duration = total_duration / iterations;
        
        println!("   {} Configuration:", name);
        println!("     Average time: {:?}", avg_duration);
        println!("     Throughput: {:.0} analyses/sec", 
                 1_000_000.0 / avg_duration.as_micros() as f64);
    }
    
    Ok(())
}

/// Multi-timeframe analysis example
fn run_multi_timeframe_analysis(data: &(Vec<f64>, Vec<f64>)) -> Result<(), FibonacciError> {
    println!("\n📊 Multi-Timeframe Analysis");
    println!("---------------------------");
    
    let (prices, volumes) = data;
    
    // Simulate different timeframes by sampling data
    let timeframes = vec![
        ("1m", 1),
        ("5m", 5),
        ("15m", 15),
        ("1h", 60),
    ];
    
    for (timeframe, sample_rate) in timeframes {
        println!("\n   📈 {} Timeframe Analysis:", timeframe);
        
        // Sample data for this timeframe
        let sampled_prices: Vec<f64> = prices.iter()
            .enumerate()
            .filter(|(i, _)| i % sample_rate == 0)
            .map(|(_, &p)| p)
            .collect();
        
        let sampled_volumes: Vec<f64> = volumes.iter()
            .enumerate()
            .filter(|(i, _)| i % sample_rate == 0)
            .map(|(_, &v)| v)
            .collect();
        
        if sampled_prices.len() < 20 {
            println!("      ⚠️ Insufficient data for {} timeframe", timeframe);
            continue;
        }
        
        // Configure analyzer for this timeframe
        let config = match timeframe {
            "1m" => FibonacciPresets::scalping(),
            "5m" => FibonacciPresets::day_trading(),
            "15m" => FibonacciPresets::swing_trading(),
            "1h" => FibonacciPresets::position_trading(),
            _ => FibonacciConfig::default(),
        };
        
        let analyzer = FibonacciAnalyzer::new(config);
        let result = analyzer.analyze(&sampled_prices, &sampled_volumes)?;
        
        println!("      Signal: {:.4}", result.signal.value());
        println!("      Confidence: {:.4}", result.confidence.value());
        println!("      Data points: {}", sampled_prices.len());
        
        // Show key levels
        let current_price = sampled_prices.last().unwrap_or(&0.0);
        println!("      Current price: ${:.2}", current_price);
    }
    
    Ok(())
}

/// Error handling examples
fn run_error_handling_examples() -> Result<(), FibonacciError> {
    println!("\n🛠️ Error Handling Examples");
    println!("---------------------------");
    
    let analyzer = FibonacciAnalyzer::default();
    
    // Test empty data
    println!("   Testing empty data:");
    match analyzer.analyze(&[], &[]) {
        Ok(_) => println!("      ❌ Should have failed"),
        Err(e) => println!("      ✅ Correctly handled: {}", e),
    }
    
    // Test mismatched lengths
    println!("   Testing mismatched data lengths:");
    match analyzer.analyze(&[100.0, 101.0], &[1000.0]) {
        Ok(_) => println!("      ❌ Should have failed"),
        Err(e) => println!("      ✅ Correctly handled: {}", e),
    }
    
    // Test invalid prices
    println!("   Testing invalid prices:");
    match analyzer.analyze(&[100.0, 0.0, 102.0], &[1000.0, 1100.0, 1200.0]) {
        Ok(_) => println!("      ❌ Should have failed"),
        Err(e) => println!("      ✅ Correctly handled: {}", e),
    }
    
    match analyzer.analyze(&[100.0, f64::NAN, 102.0], &[1000.0, 1100.0, 1200.0]) {
        Ok(_) => println!("      ❌ Should have failed"),
        Err(e) => println!("      ✅ Correctly handled: {}", e),
    }
    
    // Test configuration validation
    println!("   Testing configuration validation:");
    let mut invalid_config = FibonacciConfig::default();
    invalid_config.swing_period = 0;
    
    match invalid_config.validate() {
        Ok(_) => println!("      ❌ Should have failed"),
        Err(e) => println!("      ✅ Correctly handled: {}", e),
    }
    
    // Test boundary conditions
    println!("   Testing boundary conditions:");
    let minimal_data = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let minimal_volumes = vec![1000.0; 5];
    
    match analyzer.analyze(&minimal_data, &minimal_volumes) {
        Ok(result) => {
            println!("      ✅ Handled minimal data successfully");
            println!("         Signal: {:.4}", result.signal.value());
            println!("         Confidence: {:.4}", result.confidence.value());
        }
        Err(e) => println!("      ⚠️ Minimal data error: {}", e),
    }
    
    Ok(())
}

/// Helper function to demonstrate configuration building
fn demonstrate_configuration_building() {
    println!("\n🔧 Configuration Building Examples");
    println!("----------------------------------");
    
    // Basic configuration
    let basic_config = FibonacciConfig::new();
    println!("   Basic config - swing period: {}", basic_config.swing_period);
    
    // Custom configuration using builder pattern
    let custom_config = FibonacciConfig::default()
        .with_swing_period(20)
        .with_alignment_tolerance(0.01)
        .with_atr_period(15)
        .with_simd(true)
        .with_parallel(true)
        .with_cache_size(2000);
    
    println!("   Custom config - swing period: {}", custom_config.swing_period);
    println!("   Custom config - alignment tolerance: {:.4}", custom_config.alignment_tolerance);
    println!("   Custom config - ATR period: {}", custom_config.atr_period);
    println!("   Custom config - SIMD enabled: {}", custom_config.enable_simd);
    println!("   Custom config - parallel enabled: {}", custom_config.enable_parallel);
    println!("   Custom config - cache size: {}", custom_config.cache_size);
    
    // Validate configuration
    match custom_config.validate() {
        Ok(_) => println!("   ✅ Configuration is valid"),
        Err(e) => println!("   ❌ Configuration error: {}", e),
    }
    
    // Mathematical constants
    println!("   Golden ratio (φ): {:.10}", FibonacciConfig::golden_ratio());
    println!("   Inverse golden ratio (1/φ): {:.10}", FibonacciConfig::inverse_golden_ratio());
    
    let ratios = FibonacciConfig::fibonacci_ratios();
    println!("   Fibonacci ratios: {:?}", ratios);
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_example_data_generation() {
        let (prices, volumes) = generate_sample_market_data();
        
        assert_eq!(prices.len(), 200);
        assert_eq!(volumes.len(), 200);
        assert!(prices.iter().all(|&p| p > 0.0));
        assert!(volumes.iter().all(|&v| v > 0.0));
    }
    
    #[test]
    fn test_basic_analysis_example() {
        let data = generate_sample_market_data();
        assert!(run_basic_analysis(&data).is_ok());
    }
    
    #[test]
    fn test_custom_configuration_example() {
        let data = generate_sample_market_data();
        assert!(run_custom_configuration_analysis(&data).is_ok());
    }
    
    #[test]
    fn test_enhanced_analysis_example() {
        let data = generate_sample_market_data();
        assert!(run_enhanced_analysis(&data).is_ok());
    }
    
    #[test]
    fn test_performance_benchmark_example() {
        let data = generate_sample_market_data();
        assert!(run_performance_benchmark(&data).is_ok());
    }
    
    #[test]
    fn test_error_handling_examples() {
        assert!(run_error_handling_examples().is_ok());
    }
}