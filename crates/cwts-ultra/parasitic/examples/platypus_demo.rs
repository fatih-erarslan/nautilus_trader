//! Platypus Electroreceptor demonstration
//! Shows basic usage and capabilities

use chrono::Utc;
use parasitic::organisms::platypus::PlatypusElectroreceptor;
use parasitic::traits::{MarketData, Organism};

fn create_test_market_data(price: f64, volatility: f64, liquidity: f64) -> MarketData {
    MarketData {
        symbol: "BTC_USD".to_string(),
        timestamp: Utc::now(),
        price,
        volume: 1000.0,
        volatility,
        bid: price * 0.999,
        ask: price * 1.001,
        spread_percent: 0.002,
        market_cap: Some(1000000000.0),
        liquidity_score: liquidity,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦫 Platypus Electroreceptor Demo");
    println!("================================");

    // Create a new Platypus Electroreceptor
    let platypus = PlatypusElectroreceptor::new()?;
    println!(
        "✅ Created {} ({})",
        platypus.name(),
        platypus.organism_type()
    );
    println!("   Sensitivity: {:.3}", platypus.get_sensitivity_level());

    // Create test market data
    let market_data = create_test_market_data(50000.0, 0.03, 0.9);
    println!("\n📊 Market Data:");
    println!("   Price: ${:.0}", market_data.price);
    println!("   Volatility: {:.3}", market_data.volatility);
    println!("   Liquidity: {:.3}", market_data.liquidity_score);

    // Detect subtle signals
    println!("\n🔍 Detecting subtle electrical signals...");
    let start = std::time::Instant::now();
    let detection_result = platypus.detect_subtle_signals(&market_data, 0.95)?;
    let duration = start.elapsed();

    println!(
        "   Detected {} signals in {:.1}μs",
        detection_result.detected_signals.len(),
        duration.as_micros()
    );
    println!(
        "   Overall signal strength: {:.6}",
        detection_result.overall_signal_strength
    );
    println!(
        "   Detection confidence: {:.3}",
        detection_result.detection_confidence
    );

    // Show some detected signals
    for (i, signal) in detection_result.detected_signals.iter().take(3).enumerate() {
        println!(
            "   Signal {}: freq={:.3}Hz, amp={:.4}, strength={:.6}",
            i + 1,
            signal.frequency,
            signal.amplitude,
            signal.signal_strength
        );
    }

    // Test weak signal amplification
    if !detection_result.detected_signals.is_empty() {
        println!("\n🔊 Amplifying weak signals...");
        let amp_result = platypus.amplify_weak_signals(&detection_result.detected_signals, 5.0)?;
        println!(
            "   Amplification factor: {:.2}x",
            amp_result.amplification_factor
        );
        println!(
            "   Signal-to-noise ratio: {:.2}",
            amp_result.signal_to_noise_ratio
        );

        // Test pattern recognition
        println!("\n🔍 Recognizing electrical patterns...");
        let pattern_result =
            platypus.recognize_electrical_patterns(&amp_result.amplified_signals)?;
        println!("   Found {} patterns", pattern_result.total_patterns_found);
        if pattern_result.total_patterns_found > 0 {
            println!(
                "   Average confidence: {:.3}",
                pattern_result.average_pattern_confidence
            );

            for (i, pattern) in pattern_result
                .identified_patterns
                .iter()
                .take(2)
                .enumerate()
            {
                println!(
                    "   Pattern {}: {} (confidence={:.3}, strength={:.4})",
                    i + 1,
                    pattern.pattern_type,
                    pattern.confidence_score,
                    pattern.pattern_strength
                );
            }
        }
    }

    // Show performance metrics
    let metrics = platypus.get_metrics()?;
    println!("\n📈 Performance Metrics:");
    println!("   Total operations: {}", metrics.total_operations);
    println!("   Success rate: {:.1}%", metrics.accuracy_rate * 100.0);
    println!(
        "   Avg processing time: {:.0}ns",
        metrics.average_processing_time_ns
    );
    println!(
        "   Memory usage: {:.1}KB",
        metrics.memory_usage_bytes as f64 / 1024.0
    );

    // Show custom metrics
    println!("\n🧠 Electroreception Metrics:");
    for (key, value) in metrics.custom_metrics.iter() {
        println!("   {}: {:.3}", key, value);
    }

    println!("\n✅ Demo completed successfully!");
    Ok(())
}
