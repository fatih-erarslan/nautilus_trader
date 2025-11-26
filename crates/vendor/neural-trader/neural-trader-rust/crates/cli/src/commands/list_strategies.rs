//! List available trading strategies

use clap::Args;
use colored::*;

#[derive(Args)]
pub struct ListStrategiesArgs {
    /// Show detailed information
    #[arg(short, long)]
    detailed: bool,

    /// Filter by category
    #[arg(short, long)]
    category: Option<String>,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

pub async fn execute(args: ListStrategiesArgs) -> anyhow::Result<()> {
    if !args.json {
        println!("{}", "Available Trading Strategies".bright_cyan().bold());
        println!();
    }

    let strategies = vec![
        StrategyInfo {
            name: "pairs-trading".to_string(),
            category: "Statistical Arbitrage".to_string(),
            description: "Pairs trading using cointegration analysis".to_string(),
            requires_data: vec!["historical_prices".to_string()],
            risk_level: "Medium".to_string(),
        },
        StrategyInfo {
            name: "mean-reversion".to_string(),
            category: "Mean Reversion".to_string(),
            description: "Mean reversion with Bollinger Bands".to_string(),
            requires_data: vec!["historical_prices".to_string(), "volume".to_string()],
            risk_level: "Low".to_string(),
        },
        StrategyInfo {
            name: "momentum".to_string(),
            category: "Momentum".to_string(),
            description: "Momentum trading with RSI and MACD".to_string(),
            requires_data: vec!["historical_prices".to_string(), "volume".to_string()],
            risk_level: "High".to_string(),
        },
        StrategyInfo {
            name: "market-making".to_string(),
            category: "Market Making".to_string(),
            description: "Automated market making with inventory management".to_string(),
            requires_data: vec!["orderbook".to_string(), "trades".to_string()],
            risk_level: "Medium".to_string(),
        },
        StrategyInfo {
            name: "breakout".to_string(),
            category: "Breakout".to_string(),
            description: "Breakout detection with volume confirmation".to_string(),
            requires_data: vec!["historical_prices".to_string(), "volume".to_string()],
            risk_level: "High".to_string(),
        },
    ];

    // Filter by category if specified
    let filtered: Vec<_> = if let Some(cat) = &args.category {
        strategies
            .into_iter()
            .filter(|s| s.category.to_lowercase().contains(&cat.to_lowercase()))
            .collect()
    } else {
        strategies
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&filtered)?);
    } else {
        for (i, strategy) in filtered.iter().enumerate() {
            println!(
                "{} {}",
                format!("{}.", i + 1).bright_white().bold(),
                strategy.name.bright_green().bold()
            );
            println!("   Category: {}", strategy.category.cyan());
            println!("   Description: {}", strategy.description);

            if args.detailed {
                println!(
                    "   Required Data: {}",
                    strategy.requires_data.join(", ").yellow()
                );
                let risk_color = match strategy.risk_level.as_str() {
                    "Low" => strategy.risk_level.green(),
                    "Medium" => strategy.risk_level.yellow(),
                    "High" => strategy.risk_level.red(),
                    _ => strategy.risk_level.normal(),
                };
                println!("   Risk Level: {}", risk_color);
            }
            println!();
        }

        println!("{}", "Usage:".bright_yellow());
        println!(
            "  {} backtest --strategy <name> --symbols <symbols>",
            "neural-trader".cyan()
        );
        println!(
            "  {} trade --strategy <name> --broker <broker> --paper",
            "neural-trader".cyan()
        );
    }

    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct StrategyInfo {
    name: String,
    category: String,
    description: String,
    requires_data: Vec<String>,
    risk_level: String,
}
