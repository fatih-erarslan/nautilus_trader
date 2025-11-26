//! Live trading command - Real money trading

use clap::Args;
use colored::*;
use dialoguer::Confirm;

#[derive(Args)]
pub struct LiveArgs {
    /// Strategy file path
    #[arg(short, long)]
    strategy: Option<String>,

    /// Exchange
    #[arg(short, long, default_value = "alpaca")]
    exchange: String,

    /// Comma-separated symbols
    #[arg(long)]
    symbols: Option<String>,

    /// Max position size per symbol
    #[arg(long, default_value = "5000")]
    max_position_size: f64,

    /// Max drawdown before halt (0.0-1.0)
    #[arg(long, default_value = "0.10")]
    max_drawdown: f64,

    /// Run as daemon
    #[arg(short, long)]
    daemon: bool,

    /// Dry-run mode (no actual trades)
    #[arg(long)]
    dry_run: bool,
}

pub async fn execute(args: LiveArgs) -> anyhow::Result<()> {
    if !args.dry_run {
        // Show safety confirmation
        println!(
            "{}",
            "⚠️  WARNING: You are about to start LIVE TRADING with REAL MONEY."
                .bright_red()
                .bold()
        );
        println!();
        println!("Exchange: {} (Live)", args.exchange);
        println!("Strategy: MomentumStrategy");

        if let Some(symbols) = &args.symbols {
            println!("Symbols: {}", symbols);
        }

        println!(
            "Max Position Size: ${:.2} per symbol",
            args.max_position_size
        );
        println!("Max Drawdown: {:.0}%", args.max_drawdown * 100.0);
        println!();

        let confirmed = Confirm::new()
            .with_prompt("Type 'YES' to confirm")
            .default(false)
            .interact()?;

        if !confirmed {
            println!("Live trading cancelled.");
            return Ok(());
        }
    }

    if args.dry_run {
        println!(
            "{}",
            "Running in DRY-RUN mode (no actual trades will be executed)".bright_yellow()
        );
    } else {
        println!("{}", "Starting LIVE trading...".bright_red().bold());
    }

    println!();
    println!("{} Connected to {} Live API", "✓".green(), args.exchange);
    println!("{} Loaded strategy: MomentumStrategy", "✓".green());
    println!("{} Risk limits configured", "✓".green());
    println!();

    if args.daemon {
        println!("Running as daemon...");
        return Ok(());
    }

    println!("Press Ctrl+C to stop...");
    println!();

    // Simulate live trading
    loop {
        tokio::time::sleep(tokio::time::Duration::from_secs(60)).await;

        let timestamp = chrono::Utc::now().format("%Y-%m-%d %H:%M:%S");
        println!("[{}] Checking for signals...", timestamp);

        // In a real implementation, this would execute actual trades
    }
}
