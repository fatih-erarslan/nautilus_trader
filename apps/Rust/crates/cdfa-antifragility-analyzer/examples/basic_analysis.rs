use cdfa_antifragility_analyzer::{AntifragilityAnalyzer, AntifragilityParameters};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();
    
    // Create analyzer with default parameters
    let analyzer = AntifragilityAnalyzer::new();
    
    // Generate sample data
    let (prices, volumes) = generate_sample_data();
    
    // Perform analysis
    println!("Analyzing {} data points...", prices.len());
    let result = analyzer.analyze_prices(&prices, &volumes)?;
    
    // Display results
    println!("\n{}", result.summary());
    
    // Display component breakdown
    println!("\nComponent Analysis:");
    println!("  Convexity Score: {:.4}", result.convexity_score);
    println!("  Asymmetry Score: {:.4}", result.asymmetry_score);
    println!("  Recovery Score: {:.4}", result.recovery_score);
    println!("  Benefit Ratio Score: {:.4}", result.benefit_ratio_score);
    
    // Classification
    println!("\nClassification: {}", result.dominant_characteristic());
    
    if result.is_antifragile() {
        println!("✅ System shows antifragile characteristics");
    } else if result.is_fragile() {
        println!("❌ System shows fragile characteristics");
    } else {
        println!("⚖️  System shows robust characteristics");
    }
    
    // Performance metrics
    let performance = analyzer.get_performance_metrics();
    println!("\nPerformance Metrics:");
    println!("  Total Analyses: {}", performance.total_analyses);
    println!("  Average Time: {:?}", performance.average_analysis_time);
    println!("  Success Rate: {:.2}%", performance.success_rate() * 100.0);
    
    Ok(())
}

fn generate_sample_data() -> (Vec<f64>, Vec<f64>) {
    let n = 500;
    let mut prices = Vec::with_capacity(n);
    let mut volumes = Vec::with_capacity(n);
    
    let mut price = 100.0;
    for i in 0..n {
        let t = i as f64 * 0.01;
        
        // Create price movement with some volatility patterns
        let trend = 0.0001 * t; // Slight upward trend
        let noise = 0.02 * (t * 10.0).sin(); // Market noise
        let volatility_shock = if i % 100 == 0 {
            0.05 * (t * 50.0).sin() // Periodic volatility shocks
        } else {
            0.0
        };
        
        let return_rate = trend + noise + volatility_shock;
        price *= 1.0 + return_rate;
        prices.push(price);
        
        // Generate volume with some correlation to price changes
        let volume_base = 1000.0;
        let volume_noise = 200.0 * (t * 5.0).cos();
        let volume_spike = if return_rate.abs() > 0.03 {
            500.0 // Higher volume during large price moves
        } else {
            0.0
        };
        
        volumes.push(volume_base + volume_noise + volume_spike);
    }
    
    (prices, volumes)
}