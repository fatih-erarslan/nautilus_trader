use cdfa_soc_analyzer::{SocAnalyzer, SocParameters, SocRegime};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate sample market data
    let mut prices = vec![100.0];
    let mut volumes = vec![1000.0];
    
    // Simulate price movements with some volatility clustering (SOC-like behavior)
    for i in 1..500 {
        let volatility = if i % 50 < 10 { 0.03 } else { 0.01 }; // Volatility clustering
        let change = (rand::random::<f32>() - 0.5) * 2.0 * volatility;
        let new_price = prices.last().unwrap() * (1.0 + change);
        prices.push(new_price);
        
        let volume = 1000.0 * (1.0 + (rand::random::<f32>() - 0.5) * 0.4);
        volumes.push(volume);
    }
    
    // Create analyzer with custom parameters
    let params = SocParameters {
        sample_entropy_m: 2,
        sample_entropy_r: 0.2,
        critical_threshold_complexity: 0.7,
        critical_threshold_equilibrium: 0.8,
        critical_threshold_fragility: 0.7,
        ..Default::default()
    };
    
    let analyzer = SocAnalyzer::with_params(params);
    
    // Measure performance
    let start = Instant::now();
    let metrics = analyzer.analyze(&prices, &volumes)?;
    let duration = start.elapsed();
    
    println!("SOC Analysis completed in {:?}", duration);
    println!("Performance: {:.2} points/µs", prices.len() as f64 / duration.as_micros() as f64);
    
    // Display results
    if let Some(last_idx) = metrics.soc_index.last() {
        println!("\nLatest SOC Metrics:");
        println!("  SOC Index: {:.4}", last_idx);
        println!("  Equilibrium: {:.4}", metrics.equilibrium.last().unwrap());
        println!("  Fragility: {:.4}", metrics.fragility.last().unwrap());
        println!("  Complexity: {:.4}", metrics.complexity.last().unwrap());
        println!("  Entropy: {:.4}", metrics.entropy.last().unwrap());
        println!("  Momentum: {:.4}", metrics.momentum.last().unwrap());
        println!("  Divergence: {:.4}", metrics.divergence.last().unwrap());
        
        match metrics.regime.last().unwrap() {
            SocRegime::Critical => println!("  Regime: CRITICAL ⚠️"),
            SocRegime::NearCritical => println!("  Regime: Near-Critical 🟡"),
            SocRegime::Unstable => println!("  Regime: Unstable 🔴"),
            SocRegime::Stable => println!("  Regime: Stable 🟢"),
            SocRegime::Normal => println!("  Regime: Normal ✓"),
        }
    }
    
    // Analyze regime distribution
    println!("\nRegime Distribution:");
    let mut regime_counts = std::collections::HashMap::new();
    for regime in &metrics.regime {
        *regime_counts.entry(regime).or_insert(0) += 1;
    }
    
    for (regime, count) in regime_counts {
        let percentage = (count as f32 / metrics.regime.len() as f32) * 100.0;
        println!("  {:?}: {} ({:.1}%)", regime, count, percentage);
    }
    
    // Find critical transitions
    let mut transitions = Vec::new();
    for i in 1..metrics.regime.len() {
        if metrics.regime[i] != metrics.regime[i-1] {
            transitions.push((i, metrics.regime[i-1], metrics.regime[i]));
        }
    }
    
    if !transitions.is_empty() {
        println!("\nRegime Transitions Detected:");
        for (idx, from, to) in transitions.iter().take(5) {
            println!("  At index {}: {:?} → {:?}", idx, from, to);
        }
        if transitions.len() > 5 {
            println!("  ... and {} more transitions", transitions.len() - 5);
        }
    }
    
    // Performance characteristics
    println!("\nPerformance Characteristics:");
    println!("  Data points analyzed: {}", prices.len());
    println!("  Time per point: {:.2}ns", duration.as_nanos() as f64 / prices.len() as f64);
    
    // Test individual components for sub-microsecond performance
    println!("\nComponent Performance Tests:");
    
    // Test SOC index calculation
    let returns: Vec<f32> = prices.windows(2)
        .map(|w| (w[1] - w[0]) / w[0])
        .collect();
    
    let start = Instant::now();
    let _ = analyzer.calculate_soc_index(&returns, 20, 40)?;
    let soc_duration = start.elapsed();
    println!("  SOC Index calculation: {:?}", soc_duration);
    
    // Test momentum calculation
    let start = Instant::now();
    let _ = analyzer.calculate_soc_momentum(&metrics.soc_index, 10)?;
    let momentum_duration = start.elapsed();
    println!("  Momentum calculation: {:?}", momentum_duration);
    
    // Test divergence calculation
    let start = Instant::now();
    let _ = analyzer.calculate_soc_divergence(&metrics.equilibrium, &metrics.fragility, 5)?;
    let divergence_duration = start.elapsed();
    println!("  Divergence calculation: {:?}", divergence_duration);
    
    Ok(())
}