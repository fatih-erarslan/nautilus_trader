//! Initialize command - Create new trading project

use clap::Args;
use colored::*;
use std::fs;
use std::path::PathBuf;

#[derive(Args)]
pub struct InitArgs {
    /// Project name
    project_name: String,

    /// Template to use (momentum, mean-reversion, arbitrage, ml)
    #[arg(short, long)]
    template: Option<String>,

    /// Run interactive wizard
    #[arg(short, long)]
    interactive: bool,

    /// Exchange to configure (alpaca, coinbase, binance)
    #[arg(short, long)]
    exchange: Option<String>,

    /// Enable paper trading mode
    #[arg(short, long, default_value = "true")]
    paper: bool,
}

pub async fn execute(args: InitArgs) -> anyhow::Result<()> {
    println!("{}", "Initializing new trading project...".bright_green());

    // Create project directory
    let project_dir = PathBuf::from(&args.project_name);
    if project_dir.exists() {
        anyhow::bail!("Directory '{}' already exists", args.project_name);
    }

    fs::create_dir_all(&project_dir)?;
    fs::create_dir_all(project_dir.join("src"))?;
    fs::create_dir_all(project_dir.join("tests"))?;
    fs::create_dir_all(project_dir.join("config"))?;

    println!(
        "{} Created project directory: {}",
        "✓".green(),
        args.project_name
    );

    // Generate config file
    let config_content = generate_config(&args)?;
    fs::write(project_dir.join("config.toml"), config_content)?;
    println!("{} Generated config file: config.toml", "✓".green());

    // Generate strategy template
    let template = args.template.as_deref().unwrap_or("momentum");
    let strategy_content = generate_strategy_template(template)?;
    fs::write(project_dir.join("src/strategy.rs"), strategy_content)?;
    println!("{} Created strategy template: src/strategy.rs", "✓".green());

    // Generate test file
    let test_content = generate_test_template()?;
    fs::write(project_dir.join("tests/strategy_test.rs"), test_content)?;
    println!("{} Created tests: tests/strategy_test.rs", "✓".green());

    // Initialize git repository
    if let Ok(_) = std::process::Command::new("git")
        .args(&["init"])
        .current_dir(&project_dir)
        .output()
    {
        println!("{} Initialized Git repository", "✓".green());
    }

    // Print next steps
    println!();
    println!("{}", "Next steps:".bright_yellow().bold());
    println!("  1. cd {}", args.project_name);
    println!("  2. Edit config.toml with your API keys");
    println!("  3. Run: neural-trader backtest --start 2024-01-01 --end 2024-12-31");
    println!();

    Ok(())
}

fn generate_config(args: &InitArgs) -> anyhow::Result<String> {
    let exchange = args.exchange.as_deref().unwrap_or("alpaca");
    let paper = args.paper;

    Ok(format!(
        r#"# Neural Trader Configuration
# Generated on {}

[general]
profile = "default"
log_level = "info"

[strategy]
name = "MyStrategy"
symbols = ["BTC-USD", "ETH-USD"]
initial_capital = 100_000.0

[exchange]
name = "{}"
paper_trading = {}
api_url = "https://paper-api.alpaca.markets"

[risk]
max_position_size = 5000.0
max_portfolio_exposure = 0.20
max_drawdown = 0.15

[execution]
order_type = "market"
time_in_force = "gtc"
"#,
        chrono::Utc::now().format("%Y-%m-%d %H:%M:%S"),
        exchange,
        paper
    ))
}

fn generate_strategy_template(template: &str) -> anyhow::Result<String> {
    let content = match template {
        "momentum" => {
            r#"// Momentum Strategy Template

use neural_trader::Strategy;

pub struct MomentumStrategy {
    period: usize,
    threshold: f64,
}

impl Strategy for MomentumStrategy {
    fn on_data(&mut self, data: MarketData) -> Signal {
        // Implement momentum logic here
        Signal::Hold
    }
}
"#
        }
        "mean-reversion" => {
            r#"// Mean Reversion Strategy Template

use neural_trader::Strategy;

pub struct MeanReversionStrategy {
    lookback: usize,
    z_score_threshold: f64,
}

impl Strategy for MeanReversionStrategy {
    fn on_data(&mut self, data: MarketData) -> Signal {
        // Implement mean reversion logic here
        Signal::Hold
    }
}
"#
        }
        _ => {
            r#"// Custom Strategy Template

use neural_trader::Strategy;

pub struct CustomStrategy {}

impl Strategy for CustomStrategy {
    fn on_data(&mut self, data: MarketData) -> Signal {
        // Implement your strategy here
        Signal::Hold
    }
}
"#
        }
    };

    Ok(content.to_string())
}

fn generate_test_template() -> anyhow::Result<String> {
    Ok(r#"// Strategy tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy() {
        // Add your tests here
        assert!(true);
    }
}
"#
    .to_string())
}
