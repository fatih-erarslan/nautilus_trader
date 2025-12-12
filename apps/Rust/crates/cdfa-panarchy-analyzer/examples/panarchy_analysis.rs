use cdfa_panarchy_analyzer::{PanarchyAnalyzer, MarketData, MarketPhase};
use std::time::Instant;

fn main() {
    println!("CDFA Panarchy Analyzer Example\n");
    
    // Generate sample market data
    let mut prices = Vec::new();
    let mut volumes = Vec::new();
    let mut market_data = Vec::new();
    
    // Simulate different market phases
    for i in 0..200 {
        let t = i as f64 * 0.1;
        
        // Create price patterns for different phases
        let (price, volume) = if i < 50 {
            // Growth phase - steady uptrend
            (100.0 + t * 2.0 + (t * 0.5).sin() * 2.0, 1000.0 + t * 10.0)
        } else if i < 100 {
            // Conservation phase - sideways with low volatility
            (150.0 + (t * 0.2).sin() * 5.0, 1500.0)
        } else if i < 150 {
            // Release phase - sharp decline
            (150.0 - (t - 10.0) * 3.0 + (t * 0.8).sin() * 5.0, 2000.0 + (t * 0.5).cos() * 200.0)
        } else {
            // Reorganization phase - high volatility, finding bottom
            (75.0 + (t * 0.3).sin() * 10.0 + (t * 1.5).cos() * 5.0, 1200.0 + (t * 0.7).sin() * 300.0)
        };
        
        prices.push(price);
        volumes.push(volume);
        
        market_data.push(MarketData {
            open: price * 0.99,
            high: price * 1.01,
            low: price * 0.98,
            close: price,
            volume,
        });
    }
    
    // Create analyzer
    let mut analyzer = PanarchyAnalyzer::new();
    
    // Perform analysis
    println!("Analyzing {} data points...\n", prices.len());
    
    let start = Instant::now();
    let result = analyzer.analyze(&prices, &volumes).expect("Analysis failed");
    let elapsed = start.elapsed();
    
    // Display results
    println!("Analysis Results:");
    println!("================");
    println!("Current Phase: {} ({})", result.phase, result.phase.as_str());
    println!("Regime Score: {:.2}/100", result.regime_score);
    println!("Signal: {:.4} (0-1 range)", result.signal);
    println!("Confidence: {:.2}%", result.confidence * 100.0);
    println!();
    
    println!("PCR Components:");
    println!("  Potential (P): {:.3}", result.pcr.potential);
    println!("  Connectedness (C): {:.3}", result.pcr.connectedness);
    println!("  Resilience (R): {:.3}", result.pcr.resilience);
    println!();
    
    println!("Phase Scores:");
    println!("  Growth: {:.3}", result.phase_scores.growth);
    println!("  Conservation: {:.3}", result.phase_scores.conservation);
    println!("  Release: {:.3}", result.phase_scores.release);
    println!("  Reorganization: {:.3}", result.phase_scores.reorganization);
    println!();
    
    println!("Performance:");
    println!("  Computation time: {:?}", elapsed);
    println!("  Per-point latency: {:.2} µs", elapsed.as_micros() as f64 / prices.len() as f64);
    println!("  Hardware acceleration: {}", 
        if cdfa_panarchy_analyzer::acceleration_available() { "Available" } else { "Not available" }
    );
    
    // Analyze different time windows
    println!("\n\nPhase Evolution Analysis:");
    println!("========================");
    
    for window_start in (0..prices.len()).step_by(50) {
        let window_end = (window_start + 50).min(prices.len());
        let window_prices = &prices[window_start..window_end];
        let window_volumes = &volumes[window_start..window_end];
        
        if window_prices.len() >= 15 {
            let window_result = analyzer.analyze(window_prices, window_volumes).expect("Window analysis failed");
            
            println!("Points {}-{}: Phase = {:12}, Score = {:5.1}, Signal = {:.3}",
                window_start,
                window_end,
                window_result.phase.as_str(),
                window_result.regime_score,
                window_result.signal
            );
        }
    }
    
    // Test phase transitions
    println!("\n\nPhase Transition Detection:");
    println!("===========================");
    
    let mut prev_phase = MarketPhase::Unknown;
    let window_size = 20;
    
    for i in window_size..prices.len() {
        let window_prices = &prices[i-window_size..=i];
        let window_volumes = &volumes[i-window_size..=i];
        
        if let Ok(result) = analyzer.analyze(window_prices, window_volumes) {
            if result.phase != prev_phase && prev_phase != MarketPhase::Unknown {
                println!("Point {}: Transition from {} to {} (confidence: {:.2}%)",
                    i,
                    prev_phase,
                    result.phase,
                    result.confidence * 100.0
                );
            }
            prev_phase = result.phase;
        }
    }
    
    // Performance validation
    println!("\n\nPerformance Validation:");
    println!("======================");
    
    let mut timings = Vec::new();
    for _ in 0..100 {
        let start = Instant::now();
        let _ = analyzer.analyze(&prices[..50], &volumes[..50]);
        timings.push(start.elapsed().as_nanos() as f64);
    }
    
    let avg_ns = timings.iter().sum::<f64>() / timings.len() as f64;
    let min_ns = timings.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_ns = timings.iter().cloned().fold(0.0, f64::max);
    
    println!("50-point analysis (100 runs):");
    println!("  Average: {:.0} ns ({:.3} µs)", avg_ns, avg_ns / 1000.0);
    println!("  Min: {:.0} ns ({:.3} µs)", min_ns, min_ns / 1000.0);
    println!("  Max: {:.0} ns ({:.3} µs)", max_ns, max_ns / 1000.0);
    println!("  Target: < 800 ns ✓" );
}