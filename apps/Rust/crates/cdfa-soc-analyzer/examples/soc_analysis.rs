use cdfa_soc_analyzer::{SOCAnalyzer, SOCParameters, SOCRegime};
use ndarray::Array1;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("CDFA SOC Analyzer Example");
    println!("========================");
    
    // Initialize the analyzer
    println!("Initializing SOC analyzer...");
    cdfa_soc_analyzer::init()?;
    
    println!("Hardware acceleration available: {}", cdfa_soc_analyzer::acceleration_available());
    println!("Version: {}", cdfa_soc_analyzer::version());
    println!();
    
    // Create sample data with different patterns
    let test_cases = vec![
        ("Random Walk", generate_random_walk(500)),
        ("Sine Wave", generate_sine_wave(500)),
        ("Trending Data", generate_trending_data(500)),
        ("Noisy Data", generate_noisy_data(500)),
        ("Critical SOC Pattern", generate_critical_pattern(500)),
    ];
    
    // Create analyzer with default parameters
    let params = SOCParameters::default();
    let analyzer = SOCAnalyzer::new(params);
    
    println!("Analyzing different data patterns:");
    println!("---------------------------------");
    
    for (name, data) in test_cases {
        println!("\n📊 Analyzing: {}", name);
        
        let start = Instant::now();
        let result = analyzer.analyze(&data)?;
        let elapsed = start.elapsed();
        
        println!("  ⏱️  Computation time: {:?} ({} ns)", elapsed, result.computation_time_ns);
        println!("  📈 Sample entropy: {:.4}", result.sample_entropy);
        println!("  📊 Entropy rate: {:.4}", result.entropy_rate);
        println!("  🧠 Complexity: {:.4}", result.complexity_measure);
        println!("  ⚖️  Equilibrium: {:.4}", result.equilibrium_score);
        println!("  💥 Fragility: {:.4}", result.fragility_score);
        println!("  🎯 Regime: {} (confidence: {:.2}%)", 
                 result.regime.as_str(), result.regime_confidence * 100.0);
        println!("  🌋 Avalanches: {} events (total magnitude: {:.2})", 
                 result.avalanche_count(), result.total_avalanche_magnitude);
        
        // Check if we meet performance targets
        if result.computation_time_ns <= cdfa_soc_analyzer::perf::FULL_ANALYSIS_TARGET_NS {
            println!("  ✅ Performance target met!");
        } else {
            println!("  ⚠️  Performance target missed");
        }
        
        // Interpret regime
        match result.regime {
            SOCRegime::Critical => println!("  📍 System is at critical state - expect significant changes"),
            SOCRegime::Stable => println!("  📍 System is stable - predictable behavior expected"),
            SOCRegime::Unstable => println!("  📍 System is unstable - chaotic behavior possible"),
            SOCRegime::Unknown => println!("  📍 System state unclear - more data may be needed"),
        }
    }
    
    // Demonstrate parameter customization
    println!("\n\n🔧 Testing Custom Parameters:");
    println!("-----------------------------");
    
    let mut custom_params = SOCParameters::default();
    custom_params.sample_entropy_r = 0.15; // More sensitive
    custom_params.critical_threshold_complexity = 0.6; // Lower threshold
    
    let custom_analyzer = SOCAnalyzer::new(custom_params);
    let test_data = generate_critical_pattern(300);
    
    let result1 = analyzer.analyze(&test_data)?;
    let result2 = custom_analyzer.analyze(&test_data)?;
    
    println!("Default params regime: {} (confidence: {:.2}%)", 
             result1.regime.as_str(), result1.regime_confidence * 100.0);
    println!("Custom params regime: {} (confidence: {:.2}%)", 
             result2.regime.as_str(), result2.regime_confidence * 100.0);
    
    // Performance comparison
    println!("\n\n⚡ Performance Comparison:");
    println!("-------------------------");
    
    let data_sizes = [100, 500, 1000, 2000];
    
    for &size in &data_sizes {
        let data = generate_random_walk(size);
        
        let start = Instant::now();
        let result = analyzer.analyze(&data)?;
        let elapsed = start.elapsed();
        
        let ops_per_sec = 1_000_000_000.0 / elapsed.as_nanos() as f64;
        
        println!("  Size {}: {:.2} μs ({:.0} ops/sec)", 
                 size, elapsed.as_micros(), ops_per_sec);
        
        if result.computation_time_ns <= cdfa_soc_analyzer::perf::FULL_ANALYSIS_TARGET_NS {
            println!("    ✅ Sub-microsecond target achieved!");
        }
    }
    
    println!("\n🎉 Analysis complete!");
    Ok(())
}

fn generate_random_walk(n: usize) -> Array1<f64> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);
    let mut value = 0.0;
    
    for _ in 0..n {
        value += rng.gen_range(-0.1..0.1);
        data.push(value);
    }
    
    Array1::from_vec(data)
}

fn generate_sine_wave(n: usize) -> Array1<f64> {
    let data: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.03).sin() * 0.3)
        .collect();
    Array1::from_vec(data)
}

fn generate_trending_data(n: usize) -> Array1<f64> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n)
        .map(|i| i as f64 * 0.01 + rng.gen_range(-0.05..0.05))
        .collect();
    Array1::from_vec(data)
}

fn generate_noisy_data(n: usize) -> Array1<f64> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    let data: Vec<f64> = (0..n)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    Array1::from_vec(data)
}

fn generate_critical_pattern(n: usize) -> Array1<f64> {
    use rand::prelude::*;
    let mut rng = rand::thread_rng();
    let mut data = Vec::with_capacity(n);
    let mut volatility = 0.1;
    
    for i in 0..n {
        // Increase volatility over time (approaching criticality)
        volatility *= 1.001;
        
        // Add occasional "avalanche" events
        let avalanche = if rng.gen::<f64>() < 0.05 {
            rng.gen_range(-0.5..0.5) * volatility * 3.0
        } else {
            0.0
        };
        
        let base_value = (i as f64 * 0.02).sin();
        let noise = rng.gen_range(-volatility..volatility);
        
        data.push(base_value + noise + avalanche);
    }
    
    Array1::from_vec(data)
}