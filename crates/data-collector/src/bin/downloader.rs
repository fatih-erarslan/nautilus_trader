//! Command-line interface for downloading cryptocurrency data

use clap::{Parser, Subcommand};
use data_collector::{DataCollector, CollectorConfig};
use std::path::PathBuf;
use tracing::{info, error};
use indicatif::{ProgressBar, ProgressStyle};

#[derive(Parser)]
#[command(name = "data-downloader")]
#[command(about = "A comprehensive cryptocurrency data downloader for backtesting")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Output directory
    #[arg(short, long, default_value = "./data")]
    output: PathBuf,
}

#[derive(Subcommand)]
enum Commands {
    /// Download historical data for specific parameters
    Download {
        /// Exchange name (binance, coinbase, kraken)
        #[arg(short, long)]
        exchange: String,
        
        /// Trading symbol (e.g., BTCUSDT)
        #[arg(short, long)]
        symbol: String,
        
        /// Time interval (1m, 5m, 15m, 1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        interval: String,
        
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start: String,
        
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: String,
    },
    
    /// Download comprehensive dataset for backtesting
    Backtest {
        /// Number of years of data to download
        #[arg(short, long, default_value = "5")]
        years: u32,
        
        /// Symbols to download (comma-separated)
        #[arg(short, long)]
        symbols: Option<String>,
        
        /// Exchanges to use (comma-separated)
        #[arg(short, long)]
        exchanges: Option<String>,
        
        /// Time intervals (comma-separated)
        #[arg(short, long)]
        intervals: Option<String>,
    },
    
    /// List available symbols from exchanges
    List {
        /// Exchange name
        #[arg(short, long)]
        exchange: String,
    },
    
    /// Validate exchange connections
    Validate,
    
    /// Generate default configuration file
    Config {
        /// Output path for config file
        #[arg(short, long, default_value = "config.toml")]
        output: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("data_collector={},downloader={}", log_level, log_level))
        .init();
    
    info!("🚀 Starting Cryptocurrency Data Downloader");
    
    match cli.command {
        Commands::Download { exchange, symbol, interval, start, end } => {
            download_data(&cli, &exchange, &symbol, &interval, &start, &end).await?;
        },
        Commands::Backtest { years, symbols, exchanges, intervals } => {
            download_backtest_data(&cli, years, symbols, exchanges, intervals).await?;
        },
        Commands::List { exchange } => {
            list_symbols(&cli, &exchange).await?;
        },
        Commands::Validate => {
            validate_connections(&cli).await?;
        },
        Commands::Config { output } => {
            generate_config(&output)?;
        },
    }
    
    info!("✅ Data downloader completed successfully");
    Ok(())
}

async fn download_data(
    cli: &Cli,
    exchange: &str,
    symbol: &str,
    interval: &str,
    start: &str,
    end: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("📊 Downloading {} {} data from {} ({} to {})", symbol, interval, exchange, start, end);
    
    let config = load_config(&cli.config)?;
    let collector = DataCollector::with_config(config).await?;
    
    let progress = ProgressBar::new_spinner();
    progress.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.green} {msg}")
            .unwrap()
    );
    progress.set_message(format!("Downloading {} data...", symbol));
    
    let stats = collector.download_historical_data(exchange, symbol, interval, start, end).await?;
    
    progress.finish_with_message("Download completed");
    
    info!("📈 Download Statistics:");
    info!("  Total Records: {}", stats.total_records);
    info!("  Successful Requests: {}", stats.successful_requests);
    info!("  Failed Requests: {}", stats.failed_requests);
    info!("  Data Quality Score: {:.2}%", stats.data_quality_score * 100.0);
    info!("  Collection Duration: {}ms", stats.collection_duration_ms);
    info!("  Average Latency: {:.2}ms", stats.average_latency_ms);
    
    Ok(())
}

async fn download_backtest_data(
    cli: &Cli,
    years: u32,
    symbols: Option<String>,
    exchanges: Option<String>,
    intervals: Option<String>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🔄 Downloading {} years of historical data for backtesting", years);
    
    let config = load_config(&cli.config)?;
    let collector = DataCollector::with_config(config).await?;
    
    // Calculate date range
    let end_date = chrono::Utc::now().date_naive();
    let start_date = end_date - chrono::Duration::days((years * 365) as i64);
    
    let start_str = start_date.format("%Y-%m-%d").to_string();
    let end_str = end_date.format("%Y-%m-%d").to_string();
    
    // Parse symbols, exchanges, and intervals
    let symbol_list = parse_comma_separated(symbols, &config.collection.default_symbols);
    let exchange_list = parse_comma_separated(exchanges, &config.get_enabled_exchanges());
    let interval_list = parse_comma_separated(intervals, &config.collection.default_intervals);
    
    info!("📋 Download Plan:");
    info!("  Symbols: {}", symbol_list.join(", "));
    info!("  Exchanges: {}", exchange_list.join(", "));
    info!("  Intervals: {}", interval_list.join(", "));
    info!("  Date Range: {} to {}", start_str, end_str);
    
    let total_combinations = symbol_list.len() * exchange_list.len() * interval_list.len();
    
    let progress = ProgressBar::new(total_combinations as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} {msg}")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏  ")
    );
    
    let mut total_stats = data_collector::collectors::CollectionStats {
        total_records: 0,
        successful_requests: 0,
        failed_requests: 0,
        data_quality_score: 0.0,
        collection_duration_ms: 0,
        average_latency_ms: 0.0,
        rate_limit_hits: 0,
    };
    
    for exchange in &exchange_list {
        for symbol in &symbol_list {
            for interval in &interval_list {
                progress.set_message(format!("{}:{} ({})", exchange, symbol, interval));
                
                match collector.download_historical_data(
                    exchange, symbol, interval, &start_str, &end_str
                ).await {
                    Ok(stats) => {
                        total_stats.total_records += stats.total_records;
                        total_stats.successful_requests += stats.successful_requests;
                        total_stats.failed_requests += stats.failed_requests;
                        total_stats.collection_duration_ms += stats.collection_duration_ms;
                        
                        info!("✅ {}/{}:{} - {} records", exchange, symbol, interval, stats.total_records);
                    },
                    Err(e) => {
                        error!("❌ {}/{}:{} - Error: {}", exchange, symbol, interval, e);
                        total_stats.failed_requests += 1;
                    }
                }
                
                progress.inc(1);
                
                // Small delay to be respectful to APIs
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
    }
    
    progress.finish_with_message("Backtest data download completed");
    
    // Calculate final statistics
    total_stats.data_quality_score = if total_stats.successful_requests + total_stats.failed_requests > 0 {
        total_stats.successful_requests as f64 / (total_stats.successful_requests + total_stats.failed_requests) as f64
    } else { 0.0 };
    
    total_stats.average_latency_ms = if total_stats.successful_requests > 0 {
        total_stats.collection_duration_ms as f64 / total_stats.successful_requests as f64
    } else { 0.0 };
    
    info!("🎯 Final Backtest Data Statistics:");
    info!("  Total Records: {}", total_stats.total_records);
    info!("  Successful Downloads: {}", total_stats.successful_requests);
    info!("  Failed Downloads: {}", total_stats.failed_requests);
    info!("  Overall Quality Score: {:.2}%", total_stats.data_quality_score * 100.0);
    info!("  Total Duration: {:.2} minutes", total_stats.collection_duration_ms as f64 / 60000.0);
    
    Ok(())
}

async fn list_symbols(cli: &Cli, exchange: &str) -> Result<(), Box<dyn std::error::Error>> {
    info!("📋 Listing available symbols from {}", exchange);
    
    let config = load_config(&cli.config)?;
    let collector = DataCollector::with_config(config).await?;
    
    let available_exchanges = collector.get_available_exchanges();
    
    if !available_exchanges.contains(&exchange.to_string()) {
        error!("❌ Exchange '{}' not available. Available exchanges: {}", 
               exchange, available_exchanges.join(", "));
        return Ok(());
    }
    
    // TODO: Implement symbol listing per exchange
    info!("⏳ Fetching symbols from {}...", exchange);
    
    // For now, show default symbols
    info!("📊 Available symbols (showing defaults):");
    for symbol in &config.collection.default_symbols {
        println!("  {}", symbol);
    }
    
    Ok(())
}

async fn validate_connections(cli: &Cli) -> Result<(), Box<dyn std::error::Error>> {
    info!("🔍 Validating exchange connections...");
    
    let config = load_config(&cli.config)?;
    let collector = DataCollector::with_config(config).await?;
    
    let results = collector.validate_all_exchanges().await?;
    
    info!("🌐 Exchange Validation Results:");
    for (exchange, is_valid) in results {
        let status = if is_valid { "✅ Connected" } else { "❌ Failed" };
        info!("  {}: {}", exchange, status);
    }
    
    Ok(())
}

fn generate_config(output_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    info!("⚙️  Generating default configuration file at {:?}", output_path);
    
    let config = CollectorConfig::default();
    config.save_to_file(output_path.to_str().unwrap())?;
    
    info!("✅ Configuration file generated successfully");
    info!("💡 Edit the configuration file to customize settings before running downloads");
    
    Ok(())
}

fn load_config(config_path: &PathBuf) -> Result<CollectorConfig, Box<dyn std::error::Error>> {
    if config_path.exists() {
        info!("📄 Loading configuration from {:?}", config_path);
        let config = CollectorConfig::load_from_file(config_path.to_str().unwrap())?;
        
        // Validate configuration
        let warnings = config.validate()?;
        for warning in warnings {
            info!("⚠️  Configuration warning: {}", warning);
        }
        
        Ok(config)
    } else {
        info!("📄 Using default configuration (no config file found)");
        Ok(CollectorConfig::default())
    }
}

fn parse_comma_separated(input: Option<String>, defaults: &[String]) -> Vec<String> {
    match input {
        Some(s) => s.split(',').map(|s| s.trim().to_string()).collect(),
        None => defaults.to_vec(),
    }
}