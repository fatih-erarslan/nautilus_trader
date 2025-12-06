//! Command-line interface for hedge algorithms

use clap::{Parser, Subcommand};
use hedge_algorithms::{
    HedgeAlgorithms, HedgeConfig, MarketData, OptionsHedger, OptionType,
    PairsTrader, VolatilityHedger, WhaleDetector,
};
use std::path::Path;

#[derive(Parser)]
#[command(name = "hedge_cli")]
#[command(about = "Advanced hedge algorithms CLI")]
#[command(version = "0.1.0")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run hedge algorithm
    Run {
        /// Configuration file path
        #[arg(short, long)]
        config: Option<String>,
        /// Market data file path
        #[arg(short, long)]
        data: String,
        /// Output file path
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Calculate Black-Scholes option price
    BlackScholes {
        /// Spot price
        #[arg(short, long)]
        spot: f64,
        /// Strike price
        #[arg(short, long)]
        strike: f64,
        /// Time to expiry in years
        #[arg(short, long)]
        time: f64,
        /// Volatility
        #[arg(short, long)]
        volatility: f64,
        /// Risk-free rate
        #[arg(short, long)]
        rate: Option<f64>,
        /// Option type (call or put)
        #[arg(short, long, default_value = "call")]
        option_type: String,
    },
    /// Test pairs trading
    PairsTest {
        /// Price data file for asset A
        #[arg(short, long)]
        asset_a: String,
        /// Price data file for asset B
        #[arg(short, long)]
        asset_b: String,
        /// Output signals file
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Detect whale activity
    WhaleDetect {
        /// Market data file
        #[arg(short, long)]
        data: String,
        /// Output file
        #[arg(short, long)]
        output: Option<String>,
    },
    /// Benchmark performance
    Benchmark {
        /// Number of iterations
        #[arg(short, long, default_value = "1000")]
        iterations: usize,
        /// Number of experts
        #[arg(short, long, default_value = "10")]
        experts: usize,
    },
    /// Validate configuration
    ValidateConfig {
        /// Configuration file path
        #[arg(short, long)]
        config: String,
    },
    /// Generate sample configuration
    GenerateConfig {
        /// Output file path
        #[arg(short, long)]
        output: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::init();
    
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Run { config, data, output } => {
            run_hedge_algorithm(config, data, output)?;
        }
        Commands::BlackScholes { spot, strike, time, volatility, rate, option_type } => {
            calculate_black_scholes(spot, strike, time, volatility, rate, option_type)?;
        }
        Commands::PairsTest { asset_a, asset_b, output } => {
            test_pairs_trading(asset_a, asset_b, output)?;
        }
        Commands::WhaleDetect { data, output } => {
            detect_whale_activity(data, output)?;
        }
        Commands::Benchmark { iterations, experts } => {
            run_benchmark(iterations, experts)?;
        }
        Commands::ValidateConfig { config } => {
            validate_config(config)?;
        }
        Commands::GenerateConfig { output } => {
            generate_config(output)?;
        }
    }
    
    Ok(())
}

fn run_hedge_algorithm(
    config_path: Option<String>,
    data_path: String,
    output_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running hedge algorithm...");
    
    // Load configuration
    let config = if let Some(config_path) = config_path {
        HedgeConfig::from_file(&config_path)?
    } else {
        HedgeConfig::default()
    };
    
    // Validate configuration
    config.validate()?;
    
    // Create hedge algorithm
    let hedge = HedgeAlgorithms::new(config)?;
    
    // Load market data
    let market_data = load_market_data(&data_path)?;
    
    // Process data
    let mut recommendations = Vec::new();
    for data in market_data {
        hedge.update_market_data(&data)?;
        let recommendation = hedge.get_hedge_recommendation()?;
        recommendations.push(recommendation);
    }
    
    // Save results
    if let Some(output_path) = output_path {
        save_recommendations(&recommendations, &output_path)?;
        println!("Results saved to: {}", output_path);
    } else {
        // Print summary
        println!("Processed {} market data points", recommendations.len());
        
        if let Some(last_rec) = recommendations.last() {
            println!("Last recommendation:");
            println!("  Position size: {:.6}", last_rec.position_size);
            println!("  Hedge ratio: {:.6}", last_rec.hedge_ratio);
            println!("  Confidence: {:.2}%", last_rec.confidence * 100.0);
            println!("  Expected return: {:.6}", last_rec.expected_return);
            println!("  Volatility: {:.6}", last_rec.volatility);
            println!("  Sharpe ratio: {:.6}", last_rec.sharpe_ratio);
        }
    }
    
    println!("Hedge algorithm completed successfully!");
    Ok(())
}

fn calculate_black_scholes(
    spot: f64,
    strike: f64,
    time: f64,
    volatility: f64,
    rate: Option<f64>,
    option_type: String,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Calculating Black-Scholes option price...");
    
    let mut config = HedgeConfig::default();
    if let Some(rate) = rate {
        config.options_config.bs_params.risk_free_rate = rate;
    }
    
    let hedger = OptionsHedger::new(config);
    
    let option_type = match option_type.to_lowercase().as_str() {
        "call" => OptionType::Call,
        "put" => OptionType::Put,
        _ => return Err("Invalid option type. Use 'call' or 'put'".into()),
    };
    
    let price = hedger.black_scholes_price(spot, strike, time, volatility, option_type)?;
    let greeks = hedger.calculate_greeks(spot, strike, time, volatility, option_type)?;
    
    println!("Black-Scholes Results:");
    println!("  Option price: {:.6}", price);
    println!("  Delta: {:.6}", greeks.delta);
    println!("  Gamma: {:.6}", greeks.gamma);
    println!("  Theta: {:.6}", greeks.theta);
    println!("  Vega: {:.6}", greeks.vega);
    println!("  Rho: {:.6}", greeks.rho);
    
    Ok(())
}

fn test_pairs_trading(
    asset_a_path: String,
    asset_b_path: String,
    output_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing pairs trading...");
    
    let config = HedgeConfig::default();
    let mut pairs_trader = PairsTrader::new(config);
    
    // Load price data
    let prices_a = load_price_data(&asset_a_path)?;
    let prices_b = load_price_data(&asset_b_path)?;
    
    if prices_a.len() != prices_b.len() {
        return Err("Price data files must have the same length".into());
    }
    
    let mut signals = Vec::new();
    
    for (price_a, price_b) in prices_a.iter().zip(prices_b.iter()) {
        pairs_trader.update(*price_a, *price_b)?;
        
        if let Some(signal) = pairs_trader.generate_signal()? {
            signals.push(signal);
        }
    }
    
    println!("Pairs trading results:");
    println!("  Signals generated: {}", signals.len());
    println!("  Hedge ratio: {:.6}", pairs_trader.get_hedge_ratio());
    
    if let Some(output_path) = output_path {
        save_pairs_signals(&signals, &output_path)?;
        println!("Signals saved to: {}", output_path);
    }
    
    Ok(())
}

fn detect_whale_activity(
    data_path: String,
    output_path: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Detecting whale activity...");
    
    let config = HedgeConfig::default();
    let mut whale_detector = WhaleDetector::new(config);
    
    // Load market data
    let market_data = load_market_data(&data_path)?;
    
    for data in market_data {
        whale_detector.update(&data)?;
    }
    
    let statistics = whale_detector.get_whale_statistics();
    let recent_activities = whale_detector.get_recent_activities(10);
    
    println!("Whale detection results:");
    println!("  Total activities: {}", statistics.total_activities);
    println!("  Accumulation: {}", statistics.accumulation_count);
    println!("  Distribution: {}", statistics.distribution_count);
    println!("  Rapid entry: {}", statistics.rapid_entry_count);
    println!("  Average confidence: {:.2}%", statistics.avg_confidence * 100.0);
    
    if let Some(output_path) = output_path {
        save_whale_activities(&recent_activities, &output_path)?;
        println!("Activities saved to: {}", output_path);
    }
    
    Ok(())
}

fn run_benchmark(iterations: usize, experts: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("Running benchmark with {} iterations and {} experts...", iterations, experts);
    
    let config = HedgeConfig::default();
    let hedge = HedgeAlgorithms::new(config)?;
    
    let start_time = std::time::Instant::now();
    
    for i in 0..iterations {
        let market_data = generate_sample_market_data(i);
        hedge.update_market_data(&market_data)?;
        let _recommendation = hedge.get_hedge_recommendation()?;
    }
    
    let elapsed = start_time.elapsed();
    
    println!("Benchmark results:");
    println!("  Total time: {:.2} seconds", elapsed.as_secs_f64());
    println!("  Time per iteration: {:.2} microseconds", elapsed.as_micros() as f64 / iterations as f64);
    println!("  Throughput: {:.0} iterations/second", iterations as f64 / elapsed.as_secs_f64());
    
    Ok(())
}

fn validate_config(config_path: String) -> Result<(), Box<dyn std::error::Error>> {
    println!("Validating configuration file: {}", config_path);
    
    let config = HedgeConfig::from_file(&config_path)?;
    
    match config.validate() {
        Ok(()) => {
            println!("Configuration is valid!");
            println!("  Learning rate: {}", config.learning_rate);
            println!("  Weight decay: {}", config.weight_decay);
            println!("  Max history: {}", config.max_history);
            println!("  Factor count: {}", config.factor_config.num_factors);
        }
        Err(e) => {
            println!("Configuration validation failed: {}", e);
            return Err(e.into());
        }
    }
    
    Ok(())
}

fn generate_config(output_path: String) -> Result<(), Box<dyn std::error::Error>> {
    println!("Generating sample configuration file: {}", output_path);
    
    let config = HedgeConfig::default();
    config.to_file(&output_path)?;
    
    println!("Sample configuration generated successfully!");
    Ok(())
}

// Helper functions

fn load_market_data(path: &str) -> Result<Vec<MarketData>, Box<dyn std::error::Error>> {
    // Simple CSV loader - in practice, this would be more sophisticated
    let content = std::fs::read_to_string(path)?;
    let mut data = Vec::new();
    
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; } // Skip header
        
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 6 {
            let timestamp = chrono::Utc::now(); // In practice, parse from data
            let symbol = "SAMPLE".to_string();
            let open: f64 = parts[1].parse()?;
            let high: f64 = parts[2].parse()?;
            let low: f64 = parts[3].parse()?;
            let close: f64 = parts[4].parse()?;
            let volume: f64 = parts[5].parse()?;
            
            data.push(MarketData::new(symbol, timestamp, [open, high, low, close, volume]));
        }
    }
    
    Ok(data)
}

fn load_price_data(path: &str) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let content = std::fs::read_to_string(path)?;
    let mut prices = Vec::new();
    
    for line in content.lines() {
        if let Ok(price) = line.trim().parse::<f64>() {
            prices.push(price);
        }
    }
    
    Ok(prices)
}

fn save_recommendations(
    recommendations: &[hedge_algorithms::HedgeRecommendation],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(recommendations)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn save_pairs_signals(
    signals: &[hedge_algorithms::PairSignal],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(signals)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn save_whale_activities(
    activities: &[hedge_algorithms::WhaleActivity],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let json = serde_json::to_string_pretty(activities)?;
    std::fs::write(path, json)?;
    Ok(())
}

fn generate_sample_market_data(index: usize) -> MarketData {
    let base_price = 100.0;
    let volatility = 0.02;
    let random_factor = ((index as f64 * 0.1).sin() + 1.0) * 0.5;
    
    let price = base_price * (1.0 + volatility * random_factor);
    let volume = 1000.0 * (1.0 + random_factor);
    
    MarketData::new(
        "SAMPLE".to_string(),
        chrono::Utc::now(),
        [price * 0.995, price * 1.005, price * 0.99, price, volume]
    )
}