//! Basic usage example for regime detection

use regime_detector::{RegimeDetector, types::MarketRegime};

fn main() {
    println!("Regime Detection Performance Test");
    println!("=================================");
    
    // Create detector
    let detector = RegimeDetector::new();
    
    // Test 1: Bull trend
    println!("\n1. Testing Bull Trend Detection:");
    let bull_prices: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 0.1).collect();
    let volumes: Vec<f32> = vec![1000.0; 100];
    
    let start = std::time::Instant::now();
    let result = detector.detect_regime(&bull_prices, &volumes);
    let elapsed = start.elapsed();
    
    println!("   Detected: {}", result.regime);
    println!("   Confidence: {:.2}%", result.confidence * 100.0);
    println!("   Latency: {}ns", elapsed.as_nanos());
    println!("   Reported Latency: {}ns", result.latency_ns);
    
    // Test 2: Bear trend
    println!("\n2. Testing Bear Trend Detection:");
    let bear_prices: Vec<f32> = (0..100).map(|i| 200.0 - i as f32 * 0.1).collect();
    
    let start = std::time::Instant::now();
    let result = detector.detect_regime(&bear_prices, &volumes);
    let elapsed = start.elapsed();
    
    println!("   Detected: {}", result.regime);
    println!("   Confidence: {:.2}%", result.confidence * 100.0);
    println!("   Latency: {}ns", elapsed.as_nanos());
    
    // Test 3: Ranging market
    println!("\n3. Testing Ranging Market Detection:");
    let ranging_prices: Vec<f32> = (0..100)
        .map(|i| 100.0 + (i as f32 * 0.1).sin() * 0.5)
        .collect();
    
    let start = std::time::Instant::now();
    let result = detector.detect_regime(&ranging_prices, &volumes);
    let elapsed = start.elapsed();
    
    println!("   Detected: {}", result.regime);
    println!("   Confidence: {:.2}%", result.confidence * 100.0);
    println!("   Latency: {}ns", elapsed.as_nanos());
    
    // Test 4: High volatility
    println!("\n4. Testing High Volatility Detection:");
    let mut volatile_prices = vec![100.0];
    for i in 1..100 {
        let change = if i % 2 == 0 { 2.0 } else { -2.0 };
        volatile_prices.push(volatile_prices[i-1] + change);
    }
    
    let start = std::time::Instant::now();
    let result = detector.detect_regime(&volatile_prices, &volumes);
    let elapsed = start.elapsed();
    
    println!("   Detected: {}", result.regime);
    println!("   Confidence: {:.2}%", result.confidence * 100.0);
    println!("   Latency: {}ns", elapsed.as_nanos());
    
    // Performance benchmark
    println!("\n5. Performance Benchmark (1000 iterations):");
    let (min, median, max) = detector.benchmark_latency(1000);
    
    println!("   Min latency: {}ns", min);
    println!("   Median latency: {}ns", median);
    println!("   Max latency: {}ns", max);
    
    if median < 100 {
        println!("   ✅ SUB-100NS REQUIREMENT MET!");
    } else {
        println!("   ❌ Sub-100ns requirement not met");
    }
    
    // Test features
    println!("\n6. Feature Analysis:");
    println!("   Trend Strength: {:.4}", result.features.trend_strength);
    println!("   Volatility: {:.4}", result.features.volatility);
    println!("   Autocorrelation: {:.4}", result.features.autocorrelation);
    println!("   VWAP Ratio: {:.4}", result.features.vwap_ratio);
    println!("   Hurst Exponent: {:.4}", result.features.hurst_exponent);
    println!("   RSI: {:.1}", result.features.rsi);
    println!("   Microstructure Noise: {:.4}", result.features.microstructure_noise);
    println!("   Order Flow Imbalance: {:.4}", result.features.order_flow_imbalance);
    
    println!("\n7. Transition Probabilities:");
    for (regime, prob) in &result.transition_probs {
        println!("   {} -> {}: {:.1}%", result.regime, regime, prob * 100.0);
    }
}