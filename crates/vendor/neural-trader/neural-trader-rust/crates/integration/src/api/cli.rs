//! Command-line interface.

use clap::{Parser, Subcommand};
use crate::{Config, NeuralTrader, Result};
use tracing::info;

/// Neural Trader CLI
#[derive(Parser)]
#[command(name = "neural-trader")]
#[command(about = "Advanced AI-powered trading system", long_about = None)]
pub struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "config.toml")]
    config: String,

    /// Log level (trace, debug, info, warn, error)
    #[arg(short, long, default_value = "info")]
    log_level: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the trading system
    Start {
        /// Run in daemon mode
        #[arg(short, long)]
        daemon: bool,
    },

    /// Stop the trading system
    Stop,

    /// Execute a specific strategy
    Execute {
        /// Strategy name
        strategy: String,
    },

    /// Get portfolio status
    Portfolio,

    /// Perform risk analysis
    Risk,

    /// Train a neural model
    Train {
        /// Model type
        model: String,

        /// Training data path
        #[arg(short, long)]
        data: String,
    },

    /// Generate performance report
    Report {
        /// Time period (day, week, month, year)
        #[arg(short, long, default_value = "month")]
        period: String,
    },

    /// System health check
    Health,

    /// Start API server
    Serve {
        /// Host address
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Port number
        #[arg(short, long, default_value = "8080")]
        port: u16,
    },
}

impl Cli {
    /// Runs the CLI application.
    pub async fn run() -> Result<()> {
        let cli = Cli::parse();

        // Initialize logging
        tracing_subscriber::fmt()
            .with_env_filter(&cli.log_level)
            .init();

        info!("Neural Trader CLI starting");

        // Load configuration
        let config = Config::from_file(&cli.config)?;
        config.validate()?;

        match cli.command {
            Commands::Start { daemon } => {
                info!("Starting trading system (daemon: {})", daemon);
                let trader = NeuralTrader::new(config).await?;
                trader.start_trading().await?;

                if daemon {
                    // Keep running
                    tokio::signal::ctrl_c().await?;
                    trader.shutdown().await?;
                }
            }

            Commands::Stop => {
                info!("Stopping trading system");
                // TODO: Implement stop via IPC/signal
            }

            Commands::Execute { strategy } => {
                info!("Executing strategy: {}", strategy);
                let trader = NeuralTrader::new(config).await?;
                let result = trader.execute_strategy(&strategy).await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }

            Commands::Portfolio => {
                let trader = NeuralTrader::new(config).await?;
                let portfolio = trader.get_portfolio().await?;
                println!("{}", serde_json::to_string_pretty(&portfolio)?);
            }

            Commands::Risk => {
                let trader = NeuralTrader::new(config).await?;
                let report = trader.analyze_risk().await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }

            Commands::Train { model, data } => {
                info!("Training model: {} with data: {}", model, data);
                let trader = NeuralTrader::new(config).await?;
                let model_config = crate::types::ModelTrainingConfig {
                    model_type: model,
                    training_data: data,
                    parameters: serde_json::json!({}),
                    validation_split: 0.2,
                    epochs: 100,
                };
                let model_id = trader.train_model(model_config).await?;
                println!("Model trained successfully: {}", model_id);
            }

            Commands::Report { period } => {
                let trader = NeuralTrader::new(config).await?;
                let time_period = match period.as_str() {
                    "day" => crate::types::TimePeriod::Day,
                    "week" => crate::types::TimePeriod::Week,
                    "month" => crate::types::TimePeriod::Month,
                    "year" => crate::types::TimePeriod::Year,
                    _ => crate::types::TimePeriod::Month,
                };
                let report = trader.generate_report(time_period).await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
            }

            Commands::Health => {
                let trader = NeuralTrader::new(config).await?;
                let health = trader.health_check().await?;
                println!("{}", serde_json::to_string_pretty(&health)?);
            }

            Commands::Serve { host, port } => {
                info!("Starting API server on {}:{}", host, port);
                let trader = std::sync::Arc::new(NeuralTrader::new(config).await?);
                let api = crate::api::RestApi::new(trader);
                api.serve(&host, port).await?;
            }
        }

        Ok(())
    }
}
