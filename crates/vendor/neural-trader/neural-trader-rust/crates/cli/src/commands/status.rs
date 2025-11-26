//! Status command - Show running agents

use clap::Args;
use colored::*;

#[derive(Args)]
pub struct StatusArgs {
    /// Show detailed status
    #[arg(short, long)]
    verbose: bool,

    /// Watch status (auto-refresh)
    #[arg(short, long)]
    watch: bool,

    /// Output in JSON format
    #[arg(long)]
    json: bool,

    /// Pretty-print JSON
    #[arg(long)]
    pretty: bool,
}

pub async fn execute(args: StatusArgs) -> anyhow::Result<()> {
    if args.json {
        let status = serde_json::json!({
            "active_agents": 2,
            "agents": [
                {
                    "id": "agent-01",
                    "strategy": "Momentum",
                    "exchange": "Alpaca",
                    "status": "Running",
                    "pnl_24h": 1234.56
                },
                {
                    "id": "agent-02",
                    "strategy": "Arbitrage",
                    "exchange": "Binance",
                    "status": "Idle",
                    "pnl_24h": 567.89
                }
            ],
            "total_pnl_24h": 1802.45,
            "total_trades_24h": 47,
            "win_rate": 0.638
        });

        if args.pretty {
            println!("{}", serde_json::to_string_pretty(&status)?);
        } else {
            println!("{}", serde_json::to_string(&status)?);
        }

        return Ok(());
    }

    println!("{}", "Neural Trader Status".bright_yellow().bold());
    println!("{}", "====================".bright_yellow());
    println!();

    println!("Active Agents: {}", "2".green().bold());
    println!();

    // Print table header
    println!(
        "{:<12} {:<15} {:<12} {:<12} {:<15}",
        "Agent ID".cyan(),
        "Strategy".cyan(),
        "Exchange".cyan(),
        "Status".cyan(),
        "PnL (24h)".cyan()
    );
    println!("{}", "─".repeat(66).bright_black());

    // Print agents
    println!(
        "{:<12} {:<15} {:<12} {:<12} {}",
        "agent-01",
        "Momentum",
        "Alpaca",
        "Running".green(),
        "+$1,234.56".green()
    );

    println!(
        "{:<12} {:<15} {:<12} {:<12} {}",
        "agent-02",
        "Arbitrage",
        "Binance",
        "Idle".yellow(),
        "+$567.89".green()
    );

    println!();
    println!("Total PnL (24h): {}", "+$1,802.45".green().bold());
    println!("Total Trades (24h): {}", "47".cyan());
    println!("Win Rate: {}", "63.8%".green());
    println!();

    if args.verbose {
        println!("{}", "Recent Activity:".bright_yellow());
        println!("[10:45:23] agent-01: Bought 0.5 BTC @ $43,250");
        println!("[10:43:12] agent-02: Detected arbitrage opportunity (1.2%)");
        println!("[10:40:05] agent-01: Sold 10 ETH @ $2,315");
    }

    if args.watch {
        println!();
        println!("Watching for changes (press Ctrl+C to exit)...");
        loop {
            tokio::time::sleep(tokio::time::Duration::from_secs(5)).await;
            // In a real implementation, refresh the status
        }
    }

    Ok(())
}
