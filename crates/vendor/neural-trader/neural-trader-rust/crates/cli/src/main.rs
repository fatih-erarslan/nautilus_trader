//! Neural Trader CLI
//!
//! Command-line interface for the Neural Trading platform

use clap::{Parser, Subcommand};
use colored::*;

mod commands;

#[derive(Parser)]
#[command(name = "neural-trader")]
#[command(author, version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, global = true)]
    config: Option<String>,

    /// Profile name
    #[arg(short, long, global = true)]
    profile: Option<String>,

    /// Verbose output
    #[arg(short, long, global = true)]
    verbose: bool,

    /// Quiet mode (suppress output)
    #[arg(short, long, global = true)]
    quiet: bool,

    /// Output in JSON format
    #[arg(long, global = true)]
    json: bool,

    /// Pretty-print JSON output
    #[arg(long, global = true)]
    pretty: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize a new trading project
    Init(commands::init::InitArgs),

    /// List available trading strategies
    ListStrategies(commands::list_strategies::ListStrategiesArgs),

    /// List available broker integrations
    ListBrokers(commands::list_brokers::ListBrokersArgs),

    /// Run historical backtest
    Backtest(commands::backtest::BacktestArgs),

    /// Run paper trading (simulated)
    Paper(commands::paper::PaperArgs),

    /// Run live trading (real money)
    Live(commands::live::LiveArgs),

    /// Execute trading (live or paper)
    Trade(commands::trade::TradeArgs),

    /// Train neural forecasting models
    #[command(name = "train-neural")]
    TrainNeural(commands::train_neural::TrainNeuralArgs),

    /// Show status of running agents
    Status(commands::status::StatusArgs),

    /// Manage API keys and secrets
    Secrets(commands::secrets::SecretsArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Setup logging
    setup_logging(cli.verbose, cli.quiet)?;

    // Print banner
    if !cli.quiet && !cli.json {
        print_banner();
    }

    // Execute command
    let result = match cli.command {
        Commands::Init(args) => commands::init::execute(args).await,
        Commands::ListStrategies(args) => commands::list_strategies::execute(args).await,
        Commands::ListBrokers(args) => commands::list_brokers::execute(args).await,
        Commands::Backtest(args) => commands::backtest::execute(args).await,
        Commands::Paper(args) => commands::paper::execute(args).await,
        Commands::Live(args) => commands::live::execute(args).await,
        Commands::Trade(args) => commands::trade::execute(args).await,
        Commands::TrainNeural(args) => commands::train_neural::execute(args).await,
        Commands::Status(args) => commands::status::execute(args).await,
        Commands::Secrets(args) => commands::secrets::execute(args).await,
    };

    if let Err(e) = result {
        if cli.json {
            let error = serde_json::json!({
                "error": e.to_string(),
                "success": false
            });
            if cli.pretty {
                println!("{}", serde_json::to_string_pretty(&error)?);
            } else {
                println!("{}", serde_json::to_string(&error)?);
            }
        } else {
            eprintln!("{} {}", "Error:".red().bold(), e);
        }
        std::process::exit(1);
    }

    Ok(())
}

fn setup_logging(verbose: bool, quiet: bool) -> anyhow::Result<()> {
    if quiet {
        return Ok(());
    }

    let level = if verbose {
        tracing::Level::DEBUG
    } else {
        tracing::Level::INFO
    };

    tracing_subscriber::fmt()
        .with_max_level(level)
        .with_target(false)
        .with_thread_ids(false)
        .without_time()
        .init();

    Ok(())
}

fn print_banner() {
    println!(
        "{}",
        "╔══════════════════════════════════════╗".bright_cyan()
    );
    println!(
        "{}",
        "║   Neural Trader - Rust Edition      ║".bright_cyan()
    );
    println!(
        "{}",
        "║   AI-Powered Algorithmic Trading    ║".bright_cyan()
    );
    println!(
        "{}",
        "╚══════════════════════════════════════╝".bright_cyan()
    );
    println!();
}
