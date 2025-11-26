use crate::types::{ExecutionError, Result};
use serde::{Deserialize, Serialize};
use std::env;

/// Execution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    pub broker: String,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub paper_trading: bool,
    pub base_url: String,
}

impl ExecutionConfig {
    /// Create a new execution configuration
    pub fn new(
        broker: String,
        api_key: Option<String>,
        secret_key: Option<String>,
        paper_trading: bool,
    ) -> Self {
        let base_url = if paper_trading {
            Self::get_paper_trading_url(&broker)
        } else {
            Self::get_live_trading_url(&broker)
        };

        Self {
            broker,
            api_key,
            secret_key,
            paper_trading,
            base_url,
        }
    }

    /// Create configuration from environment variables
    pub fn from_env() -> Result<Self> {
        // Load .env file if present
        let _ = dotenv::dotenv();

        let broker = env::var("BROKER").unwrap_or_else(|_| "alpaca".to_string());
        let api_key = env::var("BROKER_API_KEY").ok();
        let secret_key = env::var("BROKER_SECRET_KEY").ok();

        // Paper trading is the default for safety
        let paper_trading = env::var("PAPER_TRADING")
            .unwrap_or_else(|_| "true".to_string())
            .to_lowercase() == "true";

        // Check if live trading is explicitly enabled
        let live_trading_enabled = env::var("ENABLE_LIVE_TRADING")
            .unwrap_or_else(|_| "false".to_string())
            .to_lowercase() == "true";

        // If paper_trading is false but live trading is not explicitly enabled, error
        if !paper_trading && !live_trading_enabled {
            return Err(ExecutionError::LiveTradingDisabled);
        }

        let config = Self::new(broker, api_key, secret_key, paper_trading);

        // Validate configuration
        config.validate()?;

        Ok(config)
    }

    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // For live trading, API keys are required
        if !self.paper_trading {
            if self.api_key.is_none() {
                return Err(ExecutionError::ConfigError(
                    "API key is required for live trading".to_string(),
                ));
            }
            if self.secret_key.is_none() {
                return Err(ExecutionError::ConfigError(
                    "Secret key is required for live trading".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// Get paper trading URL for broker
    fn get_paper_trading_url(broker: &str) -> String {
        match broker.to_lowercase().as_str() {
            "alpaca" => "https://paper-api.alpaca.markets".to_string(),
            "interactive_brokers" | "ib" => "https://localhost:5000".to_string(), // TWS paper account
            _ => "http://localhost:8080".to_string(), // Default mock endpoint
        }
    }

    /// Get live trading URL for broker
    fn get_live_trading_url(broker: &str) -> String {
        match broker.to_lowercase().as_str() {
            "alpaca" => "https://api.alpaca.markets".to_string(),
            "interactive_brokers" | "ib" => "https://localhost:5000".to_string(), // TWS live account
            _ => "http://localhost:8080".to_string(), // Default mock endpoint
        }
    }

    /// Check if live trading is enabled
    pub fn is_live_trading(&self) -> bool {
        !self.paper_trading
    }

    /// Get broker name
    pub fn broker_name(&self) -> &str {
        &self.broker
    }

    /// Get API endpoint URL
    pub fn get_endpoint(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self::new("alpaca".to_string(), None, None, true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_creation() {
        let config = ExecutionConfig::new(
            "alpaca".to_string(),
            Some("test_key".to_string()),
            Some("test_secret".to_string()),
            true,
        );

        assert_eq!(config.broker, "alpaca");
        assert_eq!(config.paper_trading, true);
        assert_eq!(config.base_url, "https://paper-api.alpaca.markets");
    }

    #[test]
    fn test_live_trading_validation() {
        let config = ExecutionConfig::new(
            "alpaca".to_string(),
            None,
            None,
            false,
        );

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_paper_trading_no_keys_required() {
        let config = ExecutionConfig::new(
            "alpaca".to_string(),
            None,
            None,
            true,
        );

        assert!(config.validate().is_ok());
    }
}
