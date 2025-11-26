//! List available broker integrations

use clap::Args;
use colored::*;

#[derive(Args)]
pub struct ListBrokersArgs {
    /// Show detailed information
    #[arg(short, long)]
    detailed: bool,

    /// Filter by type (stocks, crypto, forex, options)
    #[arg(short = 't', long)]
    broker_type: Option<String>,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

pub async fn execute(args: ListBrokersArgs) -> anyhow::Result<()> {
    if !args.json {
        println!("{}", "Available Broker Integrations".bright_cyan().bold());
        println!();
    }

    let brokers = vec![
        BrokerInfo {
            name: "alpaca".to_string(),
            display_name: "Alpaca".to_string(),
            broker_type: vec!["stocks".to_string(), "crypto".to_string()],
            features: vec![
                "paper-trading".to_string(),
                "real-time-data".to_string(),
                "commission-free".to_string(),
            ],
            regions: vec!["US".to_string()],
            status: "Active".to_string(),
        },
        BrokerInfo {
            name: "ibkr".to_string(),
            display_name: "Interactive Brokers".to_string(),
            broker_type: vec![
                "stocks".to_string(),
                "options".to_string(),
                "futures".to_string(),
                "forex".to_string(),
            ],
            features: vec![
                "paper-trading".to_string(),
                "options-trading".to_string(),
                "advanced-orders".to_string(),
            ],
            regions: vec!["Global".to_string()],
            status: "Active".to_string(),
        },
        BrokerInfo {
            name: "polygon".to_string(),
            display_name: "Polygon.io (Data Only)".to_string(),
            broker_type: vec!["stocks".to_string(), "crypto".to_string()],
            features: vec![
                "real-time-data".to_string(),
                "historical-data".to_string(),
                "websocket-stream".to_string(),
            ],
            regions: vec!["US".to_string()],
            status: "Active".to_string(),
        },
        BrokerInfo {
            name: "ccxt".to_string(),
            display_name: "CCXT (Crypto Exchange)".to_string(),
            broker_type: vec!["crypto".to_string()],
            features: vec![
                "multi-exchange".to_string(),
                "spot-trading".to_string(),
                "futures-trading".to_string(),
            ],
            regions: vec!["Global".to_string()],
            status: "Active".to_string(),
        },
        BrokerInfo {
            name: "oanda".to_string(),
            display_name: "OANDA".to_string(),
            broker_type: vec!["forex".to_string()],
            features: vec![
                "paper-trading".to_string(),
                "24-5-trading".to_string(),
                "low-spreads".to_string(),
            ],
            regions: vec!["Global".to_string()],
            status: "Active".to_string(),
        },
        BrokerInfo {
            name: "questrade".to_string(),
            display_name: "Questrade".to_string(),
            broker_type: vec!["stocks".to_string(), "options".to_string()],
            features: vec![
                "canadian-markets".to_string(),
                "tfsa-rrsp".to_string(),
                "low-fees".to_string(),
            ],
            regions: vec!["Canada".to_string()],
            status: "Active".to_string(),
        },
    ];

    // Filter by type if specified
    let filtered: Vec<_> = if let Some(btype) = &args.broker_type {
        brokers
            .into_iter()
            .filter(|b| {
                b.broker_type
                    .iter()
                    .any(|t| t.to_lowercase().contains(&btype.to_lowercase()))
            })
            .collect()
    } else {
        brokers
    };

    if args.json {
        println!("{}", serde_json::to_string_pretty(&filtered)?);
    } else {
        for (i, broker) in filtered.iter().enumerate() {
            println!(
                "{} {}",
                format!("{}.", i + 1).bright_white().bold(),
                broker.display_name.bright_green().bold()
            );
            println!("   ID: {}", broker.name.cyan());
            println!("   Types: {}", broker.broker_type.join(", ").yellow());

            if args.detailed {
                println!("   Features:");
                for feature in &broker.features {
                    println!("     • {}", feature.green());
                }
                println!("   Regions: {}", broker.regions.join(", "));
                println!("   Status: {}", broker.status.bright_green());
            }
            println!();
        }

        println!("{}", "Usage:".bright_yellow());
        println!(
            "  {} trade --strategy <strategy> --broker <broker-id> --paper",
            "neural-trader".cyan()
        );
        println!(
            "  {} secrets set <broker-id>_API_KEY <your-key>",
            "neural-trader".cyan()
        );
    }

    Ok(())
}

#[derive(Debug, serde::Serialize)]
struct BrokerInfo {
    name: String,
    display_name: String,
    #[serde(rename = "type")]
    broker_type: Vec<String>,
    features: Vec<String>,
    regions: Vec<String>,
    status: String,
}
