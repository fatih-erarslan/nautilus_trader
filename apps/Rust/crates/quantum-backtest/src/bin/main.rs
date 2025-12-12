//! Quantum-Enhanced Cryptocurrency Backtesting
//! 
//! Command-line interface for running comprehensive backtests with quantum pattern recognition

use quantum_backtest::{BacktestEngine, BacktestConfig, BacktestResult, init};
use chrono::{DateTime, Utc};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use tracing::{info, error};
use serde_json;

#[derive(Parser)]
#[command(name = "quantum-backtest")]
#[command(about = "Quantum-Enhanced Cryptocurrency Backtesting Engine")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
    
    /// Enable verbose logging
    #[arg(short, long)]
    verbose: bool,
    
    /// Configuration file path
    #[arg(short, long)]
    config: Option<PathBuf>,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a full 5-year backtest
    FullBacktest {
        /// Assets to trade (comma-separated)
        #[arg(short, long)]
        assets: Option<String>,
        
        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        initial_capital: f64,
        
        /// Enable quantum patterns
        #[arg(short, long, default_value = "true")]
        quantum: bool,
        
        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// Run a custom backtest with specific parameters
    Custom {
        /// Start date (YYYY-MM-DD)
        #[arg(long)]
        start: String,
        
        /// End date (YYYY-MM-DD)
        #[arg(long)]
        end: String,
        
        /// Assets to trade (comma-separated)
        #[arg(short, long)]
        assets: String,
        
        /// Initial capital
        #[arg(short, long, default_value = "100000")]
        initial_capital: f64,
        
        /// Time frame (1h, 4h, 1d)
        #[arg(short, long, default_value = "1h")]
        timeframe: String,
        
        /// Output file for results
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    
    /// Generate default configuration file
    Config {
        /// Output path for config file
        #[arg(short, long, default_value = "backtest-config.toml")]
        output: PathBuf,
    },
    
    /// Quick performance test
    QuickTest {
        /// Number of days to test
        #[arg(short, long, default_value = "30")]
        days: u32,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    
    // Initialize logging
    let log_level = if cli.verbose { "debug" } else { "info" };
    tracing_subscriber::fmt()
        .with_env_filter(format!("quantum_backtest={}", log_level))
        .init();
    
    // Initialize backtesting system
    init().await?;
    
    info!("🚀 Quantum-Enhanced Cryptocurrency Backtesting Engine");
    
    match cli.command {
        Commands::FullBacktest { assets, initial_capital, quantum, output } => {
            run_full_backtest(assets, initial_capital, quantum, output).await?;
        },
        Commands::Custom { start, end, assets, initial_capital, timeframe, output } => {
            run_custom_backtest(start, end, assets, initial_capital, timeframe, output).await?;
        },
        Commands::Config { output } => {
            generate_config(output)?;
        },
        Commands::QuickTest { days } => {
            run_quick_test(days).await?;
        },
    }
    
    Ok(())
}

async fn run_full_backtest(
    assets: Option<String>,
    initial_capital: f64,
    quantum: bool,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🎯 Starting 5-year comprehensive backtest");
    
    let mut config = BacktestConfig::default();
    config.initial_capital = initial_capital;
    config.enable_quantum = quantum;
    
    if let Some(assets_str) = assets {
        config.assets = assets_str.split(',').map(|s| s.trim().to_string()).collect();
    }
    
    info!("📊 Configuration:");
    info!("  • Period: {} to {}", config.start_date, config.end_date);
    info!("  • Assets: {:?}", config.assets);
    info!("  • Initial Capital: ${:.2}", config.initial_capital);
    info!("  • Quantum Enabled: {}", config.enable_quantum);
    info!("  • Timeframe: {}", config.timeframe);
    
    // Create and run backtest engine
    let mut engine = BacktestEngine::new(config).await?;
    let result = engine.run().await?;
    
    // Display results
    display_backtest_results(&result);
    
    // Save results if output specified
    if let Some(output_path) = output {
        save_results(&result, &output_path)?;
        info!("📄 Results saved to: {:?}", output_path);
    }
    
    Ok(())
}

async fn run_custom_backtest(
    start: String,
    end: String,
    assets: String,
    initial_capital: f64,
    timeframe: String,
    output: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("🎯 Starting custom backtest");
    
    let start_date = DateTime::parse_from_str(&format!("{} 00:00:00 +0000", start), "%Y-%m-%d %H:%M:%S %z")?
        .with_timezone(&Utc);
    let end_date = DateTime::parse_from_str(&format!("{} 23:59:59 +0000", end), "%Y-%m-%d %H:%M:%S %z")?
        .with_timezone(&Utc);
    
    let assets_vec: Vec<String> = assets.split(',').map(|s| s.trim().to_string()).collect();
    
    let mut config = BacktestConfig::default();
    config.start_date = start_date;
    config.end_date = end_date;
    config.assets = assets_vec;
    config.initial_capital = initial_capital;
    config.timeframe = timeframe;
    
    info!("📊 Configuration:");
    info!("  • Period: {} to {}", config.start_date, config.end_date);
    info!("  • Assets: {:?}", config.assets);
    info!("  • Initial Capital: ${:.2}", config.initial_capital);
    info!("  • Timeframe: {}", config.timeframe);
    
    let mut engine = BacktestEngine::new(config).await?;
    let result = engine.run().await?;
    
    display_backtest_results(&result);
    
    if let Some(output_path) = output {
        save_results(&result, &output_path)?;
        info!("📄 Results saved to: {:?}", output_path);
    }
    
    Ok(())
}

async fn run_quick_test(days: u32) -> Result<(), Box<dyn std::error::Error>> {
    info!("⚡ Running quick performance test ({} days)", days);
    
    let end_date = Utc::now();
    let start_date = end_date - chrono::Duration::days(days as i64);
    
    let mut config = BacktestConfig::default();
    config.start_date = start_date;
    config.end_date = end_date;
    config.assets = vec!["BTCUSDT".to_string()];
    config.initial_capital = 10000.0;
    
    let mut engine = BacktestEngine::new(config).await?;
    let result = engine.run().await?;
    
    info!("⚡ Quick Test Results:");
    info!("  • Execution Time: {}ms", result.execution_time_ms);
    info!("  • Total Return: {:.2}%", result.performance.total_return);
    info!("  • Sharpe Ratio: {:.2}", result.performance.sharpe_ratio);
    info!("  • Max Drawdown: {:.2}%", result.drawdown_metrics.max_drawdown);
    
    Ok(())
}

fn generate_config(output_path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    info!("⚙️ Generating default configuration");
    
    let config = BacktestConfig::default();
    let toml_content = toml::to_string_pretty(&config)?;
    
    std::fs::write(&output_path, toml_content)?;
    
    info!("✅ Configuration generated: {:?}", output_path);
    Ok(())
}

fn display_backtest_results(result: &BacktestResult) {
    println!("\n🎯 QUANTUM-ENHANCED BACKTEST RESULTS");
    println!("═══════════════════════════════════════════════════════════");
    
    // Performance Overview
    println!("\n📈 PERFORMANCE OVERVIEW");
    println!("  • Total Return:        {:.2}%", result.performance.total_return);
    println!("  • Annualized Return:   {:.2}%", result.performance.annualized_return);
    println!("  • Volatility:          {:.2}%", result.performance.volatility);
    println!("  • Sharpe Ratio:        {:.3}", result.performance.sharpe_ratio);
    println!("  • Sortino Ratio:       {:.3}", result.performance.sortino_ratio);
    println!("  • Calmar Ratio:        {:.3}", result.performance.calmar_ratio);
    
    // Trading Statistics
    println!("\n📊 TRADING STATISTICS");
    println!("  • Total Trades:        {}", result.performance.total_trades);
    println!("  • Winning Trades:      {} ({:.1}%)", 
             result.performance.winning_trades, result.performance.win_rate);
    println!("  • Losing Trades:       {}", result.performance.losing_trades);
    println!("  • Profit Factor:       {:.2}", result.performance.profit_factor);
    println!("  • Avg Winning Trade:   ${:.2}", result.performance.avg_winning_trade);
    println!("  • Avg Losing Trade:    ${:.2}", result.performance.avg_losing_trade);
    println!("  • Best Trade:          ${:.2}", result.performance.best_trade);
    println!("  • Worst Trade:         ${:.2}", result.performance.worst_trade);
    
    // Risk Metrics
    println!("\n⚠️  RISK METRICS");
    println!("  • Max Drawdown:        {:.2}%", result.drawdown_metrics.max_drawdown);
    println!("  • Max DD Duration:     {} days", result.drawdown_metrics.max_drawdown_duration);
    println!("  • Recovery Factor:     {:.2}", result.drawdown_metrics.recovery_factor);
    println!("  • VaR (95%):          {:.2}%", result.risk_metrics.var_95);
    println!("  • CVaR (95%):         {:.2}%", result.risk_metrics.cvar_95);
    println!("  • Downside Deviation:  {:.2}%", result.risk_metrics.downside_deviation);
    
    // Quantum Analysis
    println!("\n🔬 QUANTUM PATTERN ANALYSIS");
    let quantum_signals = result.quantum_signals.len();
    let strong_signals = result.quantum_signals.iter()
        .filter(|s| matches!(s.signal_type, quantum_backtest::SignalType::StrongBuy | quantum_backtest::SignalType::StrongSell))
        .count();
    
    println!("  • Total Quantum Signals: {}", quantum_signals);
    println!("  • Strong Signals:         {} ({:.1}%)", 
             strong_signals, 
             if quantum_signals > 0 { strong_signals as f64 / quantum_signals as f64 * 100.0 } else { 0.0 });
    
    if let Some(signal) = result.quantum_signals.first() {
        println!("  • Avg Confidence:        {:.3}", 
                 result.quantum_signals.iter().map(|s| s.confidence).sum::<f64>() / quantum_signals as f64);
    }
    
    // Final Portfolio
    if let Some((final_date, final_value)) = result.portfolio_value_history.last() {
        println!("\n💰 FINAL PORTFOLIO");
        println!("  • Final Date:          {}", final_date.format("%Y-%m-%d"));
        println!("  • Final Value:         ${:.2}", final_value);
        println!("  • Initial Capital:     ${:.2}", result.config.initial_capital);
        println!("  • Absolute Gain:       ${:.2}", final_value - result.config.initial_capital);
    }
    
    // Execution Stats
    println!("\n⚡ EXECUTION STATISTICS");
    println!("  • Execution Time:      {}ms", result.execution_time_ms);
    println!("  • Assets Traded:       {:?}", result.config.assets);
    println!("  • Timeframe:           {}", result.config.timeframe);
    println!("  • Quantum Enabled:     {}", result.config.enable_quantum);
    
    println!("\n═══════════════════════════════════════════════════════════");
    
    // Performance Rating
    let performance_rating = calculate_performance_rating(&result.performance, &result.drawdown_metrics);
    println!("🏆 OVERALL PERFORMANCE RATING: {}", performance_rating);
    
    println!("═══════════════════════════════════════════════════════════\n");
}

fn calculate_performance_rating(perf: &quantum_backtest::PerformanceMetrics, drawdown: &quantum_backtest::DrawdownMetrics) -> &'static str {
    let score = 
        (if perf.sharpe_ratio > 2.0 { 3 } else if perf.sharpe_ratio > 1.0 { 2 } else { 1 }) +
        (if perf.calmar_ratio > 1.0 { 3 } else if perf.calmar_ratio > 0.5 { 2 } else { 1 }) +
        (if drawdown.max_drawdown < 10.0 { 3 } else if drawdown.max_drawdown < 20.0 { 2 } else { 1 }) +
        (if perf.win_rate > 60.0 { 3 } else if perf.win_rate > 50.0 { 2 } else { 1 });
    
    match score {
        10..=12 => "🥇 EXCELLENT",
        7..=9 => "🥈 GOOD", 
        4..=6 => "🥉 AVERAGE",
        _ => "❌ POOR",
    }
}

fn save_results(result: &BacktestResult, output_path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let json_content = serde_json::to_string_pretty(result)?;
    std::fs::write(output_path, json_content)?;
    Ok(())
}