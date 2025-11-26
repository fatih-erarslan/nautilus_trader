//! Unified configuration management for the neural-trader system.

use serde::{Serialize, Deserialize};
use std::path::Path;
use crate::error::{Error, Result};

/// Main configuration structure for the Neural Trader system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Broker configurations
    pub brokers: BrokerConfig,

    /// Strategy configurations
    pub strategies: StrategyConfig,

    /// Risk management configurations
    pub risk: RiskConfig,

    /// Neural network configurations
    pub neural: NeuralConfig,

    /// Memory system configurations
    pub memory: MemoryConfig,

    /// Distributed system configurations
    pub distributed: DistributedConfig,

    /// API server configurations
    pub api: ApiConfig,

    /// Runtime configurations
    pub runtime: RuntimeConfig,

    /// Logging and monitoring configurations
    pub observability: ObservabilityConfig,
}

impl Config {
    /// Loads configuration from a TOML file.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::config(format!("Failed to read config file: {}", e)))?;

        toml::from_str(&content)
            .map_err(|e| Error::config(format!("Failed to parse config: {}", e)))
    }

    /// Loads configuration from environment variables.
    pub fn from_env() -> Result<Self> {
        let mut config_builder = ::config::Config::builder();

        // Load from environment with NT_ prefix
        config_builder = config_builder.add_source(
            ::config::Environment::with_prefix("NT")
                .separator("__")
                .try_parsing(true)
        );

        config_builder.build()
            .and_then(|c| c.try_deserialize())
            .map_err(|e| Error::config(format!("Failed to load from environment: {}", e)))
    }

    /// Loads configuration from multiple sources with precedence:
    /// 1. CLI arguments (highest priority)
    /// 2. Environment variables
    /// 3. Config file
    /// 4. Defaults (lowest priority)
    pub fn load() -> Result<Self> {
        let mut config_builder = ::config::Config::builder();

        // 1. Load defaults
        config_builder = config_builder.add_source(::config::File::from_str(
            include_str!("../config.default.toml"),
            ::config::FileFormat::Toml,
        ));

        // 2. Load from config file if it exists
        if Path::new("config.toml").exists() {
            config_builder = config_builder.add_source(::config::File::with_name("config"));
        }

        // 3. Load from .env file if it exists
        if Path::new(".env").exists() {
            dotenvy::dotenv().ok();
        }

        // 4. Override with environment variables
        config_builder = config_builder.add_source(
            ::config::Environment::with_prefix("NT")
                .separator("__")
                .try_parsing(true)
        );

        config_builder.build()
            .and_then(|c| c.try_deserialize())
            .map_err(|e| Error::config(format!("Failed to load configuration: {}", e)))
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<()> {
        // Validate broker config
        if self.brokers.enabled.is_empty() {
            return Err(Error::config("At least one broker must be enabled"));
        }

        // Validate strategies
        if self.strategies.enabled.is_empty() {
            return Err(Error::config("At least one strategy must be enabled"));
        }

        // Validate risk limits
        if self.risk.max_position_size <= rust_decimal::Decimal::ZERO {
            return Err(Error::config("Max position size must be positive"));
        }

        Ok(())
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            brokers: BrokerConfig::default(),
            strategies: StrategyConfig::default(),
            risk: RiskConfig::default(),
            neural: NeuralConfig::default(),
            memory: MemoryConfig::default(),
            distributed: DistributedConfig::default(),
            api: ApiConfig::default(),
            runtime: RuntimeConfig::default(),
            observability: ObservabilityConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BrokerConfig {
    pub enabled: Vec<String>,
    pub alpaca: Option<AlpacaConfig>,
    pub binance: Option<BinanceConfig>,
    pub coinbase: Option<CoinbaseConfig>,
    pub interactive_brokers: Option<InteractiveBrokersConfig>,
    // Add other brokers...
}

impl Default for BrokerConfig {
    fn default() -> Self {
        Self {
            enabled: vec!["alpaca".to_string()],
            alpaca: Some(AlpacaConfig::default()),
            binance: None,
            coinbase: None,
            interactive_brokers: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlpacaConfig {
    pub api_key: String,
    pub api_secret: String,
    pub paper_trading: bool,
    pub base_url: Option<String>,
}

impl Default for AlpacaConfig {
    fn default() -> Self {
        Self {
            api_key: String::new(),
            api_secret: String::new(),
            paper_trading: true,
            base_url: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinanceConfig {
    pub api_key: String,
    pub api_secret: String,
    pub testnet: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoinbaseConfig {
    pub api_key: String,
    pub api_secret: String,
    pub passphrase: String,
    pub sandbox: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractiveBrokersConfig {
    pub account_id: String,
    pub host: String,
    pub port: u16,
    pub client_id: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    pub enabled: Vec<String>,
    pub momentum: Option<MomentumConfig>,
    pub mean_reversion: Option<MeanReversionConfig>,
    pub pairs_trading: Option<PairsTradingConfig>,
    // Add other strategies...
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            enabled: vec!["momentum".to_string()],
            momentum: Some(MomentumConfig::default()),
            mean_reversion: None,
            pairs_trading: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumConfig {
    pub lookback_period: usize,
    pub top_n: usize,
    pub rebalance_frequency: String,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            lookback_period: 20,
            top_n: 10,
            rebalance_frequency: "daily".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionConfig {
    pub z_score_threshold: f64,
    pub lookback_period: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairsTradingConfig {
    pub correlation_threshold: f64,
    pub cointegration_pvalue: f64,
    pub lookback_period: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    pub max_position_size: rust_decimal::Decimal,
    pub max_portfolio_heat: f64,
    pub max_drawdown: f64,
    pub var_confidence: f64,
    pub kelly_fraction: f64,
    pub enable_gpu: bool,
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_position_size: rust_decimal::Decimal::from(100000),
            max_portfolio_heat: 0.02,
            max_drawdown: 0.20,
            var_confidence: 0.95,
            kelly_fraction: 0.25,
            enable_gpu: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralConfig {
    pub models: Vec<String>,
    pub training: TrainingConfig,
    pub inference: InferenceConfig,
}

impl Default for NeuralConfig {
    fn default() -> Self {
        Self {
            models: vec!["lstm".to_string(), "transformer".to_string()],
            training: TrainingConfig::default(),
            inference: InferenceConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub epochs: usize,
    pub learning_rate: f64,
    pub validation_split: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            epochs: 100,
            learning_rate: 0.001,
            validation_split: 0.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    pub batch_size: usize,
    pub use_gpu: bool,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            batch_size: 64,
            use_gpu: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub agentdb_path: String,
    pub reasoning_bank_path: String,
    pub cache_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            agentdb_path: "./data/agentdb".to_string(),
            reasoning_bank_path: "./data/reasoning_bank".to_string(),
            cache_size: 10000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub enable_e2b: bool,
    pub enable_federation: bool,
    pub node_id: Option<String>,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enable_e2b: false,
            enable_federation: false,
            node_id: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiConfig {
    pub rest: RestConfig,
    pub websocket: WebSocketConfig,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            rest: RestConfig::default(),
            websocket: WebSocketConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RestConfig {
    pub host: String,
    pub port: u16,
    pub enable_cors: bool,
}

impl Default for RestConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            enable_cors: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebSocketConfig {
    pub port: u16,
    pub max_connections: usize,
}

impl Default for WebSocketConfig {
    fn default() -> Self {
        Self {
            port: 8081,
            max_connections: 1000,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    pub worker_threads: Option<usize>,
    pub max_blocking_threads: Option<usize>,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            worker_threads: None, // Use Tokio defaults
            max_blocking_threads: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservabilityConfig {
    pub log_level: String,
    pub log_format: String,
    pub enable_tracing: bool,
    pub enable_metrics: bool,
    pub metrics_port: u16,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            log_format: "json".to_string(),
            enable_tracing: true,
            enable_metrics: true,
            metrics_port: 9090,
        }
    }
}
