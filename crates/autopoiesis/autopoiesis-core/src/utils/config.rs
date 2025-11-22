//! Configuration management utilities

use crate::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main configuration structure for the autopoiesis system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// System-wide settings
    pub system: SystemConfig,
    
    /// Market data configuration
    pub market_data: MarketDataConfig,
    
    /// Trading configuration
    pub trading: TradingConfig,
    
    /// Risk management configuration
    pub risk: RiskConfig,
    
    /// Database configuration
    pub database: DatabaseConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Feature flags
    pub features: FeatureFlags,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemConfig {
    /// System name
    pub name: String,
    
    /// System version
    pub version: String,
    
    /// Environment (development, staging, production)
    pub environment: Environment,
    
    /// Number of worker threads
    pub worker_threads: usize,
    
    /// Enable metrics collection
    pub enable_metrics: bool,
    
    /// Metrics collection interval in seconds
    pub metrics_interval_seconds: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Environment {
    Development,
    Staging,
    Production,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketDataConfig {
    /// Enabled data sources
    pub sources: Vec<DataSourceConfig>,
    
    /// Data refresh interval in milliseconds
    pub refresh_interval_ms: u64,
    
    /// Maximum data age in seconds
    pub max_data_age_seconds: u64,
    
    /// Enable data validation
    pub enable_validation: bool,
    
    /// Data storage settings
    pub storage: DataStorageSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourceConfig {
    /// Data source name
    pub name: String,
    
    /// Data source type
    pub source_type: String,
    
    /// API endpoint URL
    pub endpoint: String,
    
    /// API key (if required)
    pub api_key: Option<String>,
    
    /// Rate limit (requests per second)
    pub rate_limit: u32,
    
    /// Weight in data aggregation
    pub weight: f64,
    
    /// Enable/disable this source
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStorageSettings {
    /// Storage backend type
    pub backend: StorageBackend,
    
    /// Connection string
    pub connection_string: String,
    
    /// Maximum connections in pool
    pub max_connections: u32,
    
    /// Data retention period in days
    pub retention_days: u32,
    
    /// Enable compression
    pub enable_compression: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    PostgreSQL,
    MySQL,
    SQLite,
    InfluxDB,
    TimescaleDB,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingConfig {
    /// Enabled trading strategies
    pub strategies: Vec<StrategyConfig>,
    
    /// Default position size
    pub default_position_size: f64,
    
    /// Maximum concurrent orders
    pub max_concurrent_orders: u32,
    
    /// Order timeout in seconds
    pub order_timeout_seconds: u32,
    
    /// Enable paper trading mode
    pub paper_trading: bool,
    
    /// Exchange configurations
    pub exchanges: Vec<ExchangeConfigData>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,
    
    /// Strategy type
    pub strategy_type: String,
    
    /// Strategy parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Enable/disable this strategy
    pub enabled: bool,
    
    /// Allocated capital
    pub allocated_capital: f64,
    
    /// Risk limits
    pub risk_limits: StrategyRiskLimits,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyRiskLimits {
    /// Maximum position size
    pub max_position_size: f64,
    
    /// Maximum daily loss
    pub max_daily_loss: f64,
    
    /// Maximum drawdown
    pub max_drawdown: f64,
    
    /// Stop loss percentage
    pub stop_loss_pct: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfigData {
    /// Exchange name
    pub name: String,
    
    /// API endpoint
    pub api_endpoint: String,
    
    /// API credentials
    pub credentials: ExchangeCredentials,
    
    /// Supported trading pairs
    pub supported_pairs: Vec<String>,
    
    /// Fee structure
    pub fees: ExchangeFees,
    
    /// Rate limits
    pub rate_limits: ExchangeRateLimits,
    
    /// Enable/disable this exchange
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeCredentials {
    /// API key
    pub api_key: String,
    
    /// Secret key
    pub secret_key: String,
    
    /// Passphrase (if required)
    pub passphrase: Option<String>,
    
    /// Sandbox mode
    pub sandbox: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeFees {
    /// Maker fee percentage
    pub maker_fee_pct: f64,
    
    /// Taker fee percentage
    pub taker_fee_pct: f64,
    
    /// Withdrawal fees by currency
    pub withdrawal_fees: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeRateLimits {
    /// Orders per second
    pub orders_per_second: u32,
    
    /// Requests per minute
    pub requests_per_minute: u32,
    
    /// Weight per request
    pub weight_per_request: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Global risk limits
    pub global_limits: GlobalRiskLimits,
    
    /// Position sizing method
    pub position_sizing: PositionSizingConfig,
    
    /// Risk monitoring settings
    pub monitoring: RiskMonitoringConfig,
    
    /// Emergency stop conditions
    pub emergency_stops: EmergencyStopConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalRiskLimits {
    /// Maximum total exposure
    pub max_total_exposure: f64,
    
    /// Maximum daily loss
    pub max_daily_loss: f64,
    
    /// Maximum drawdown
    pub max_drawdown: f64,
    
    /// Maximum correlation between positions
    pub max_correlation: f64,
    
    /// Blacklisted symbols
    pub blacklisted_symbols: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizingConfig {
    /// Sizing method
    pub method: PositionSizingMethod,
    
    /// Base position size
    pub base_size: f64,
    
    /// Volatility adjustment factor
    pub volatility_adjustment: f64,
    
    /// Minimum position size
    pub min_size: f64,
    
    /// Maximum position size
    pub max_size: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PositionSizingMethod {
    Fixed,
    PercentOfEquity,
    VolatilityAdjusted,
    KellyOptimal,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskMonitoringConfig {
    /// Monitoring frequency in seconds
    pub frequency_seconds: u32,
    
    /// VaR calculation method
    pub var_method: VarMethod,
    
    /// VaR confidence level
    pub var_confidence: f64,
    
    /// VaR lookback period in days
    pub var_lookback_days: u32,
    
    /// Enable real-time alerts
    pub enable_alerts: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VarMethod {
    Historical,
    Parametric,
    MonteCarlo,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergencyStopConfig {
    /// Enable emergency stops
    pub enabled: bool,
    
    /// Maximum drawdown trigger
    pub max_drawdown_trigger: f64,
    
    /// Daily loss trigger
    pub daily_loss_trigger: f64,
    
    /// Market volatility trigger
    pub volatility_trigger: f64,
    
    /// Auto-recovery settings
    pub auto_recovery: AutoRecoveryConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoRecoveryConfig {
    /// Enable auto-recovery
    pub enabled: bool,
    
    /// Recovery delay in minutes
    pub recovery_delay_minutes: u32,
    
    /// Gradual position restoration
    pub gradual_restoration: bool,
    
    /// Maximum recovery attempts
    pub max_attempts: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    /// Database URL
    pub url: String,
    
    /// Maximum connections
    pub max_connections: u32,
    
    /// Connection timeout in seconds
    pub connection_timeout_seconds: u32,
    
    /// Enable SSL
    pub enable_ssl: bool,
    
    /// Migration settings
    pub migrations: MigrationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationConfig {
    /// Auto-run migrations on startup
    pub auto_migrate: bool,
    
    /// Migrations directory
    pub migrations_dir: String,
    
    /// Enable migration validation
    pub validate_migrations: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    /// Log level
    pub level: LogLevel,
    
    /// Log format
    pub format: LogFormat,
    
    /// Output destinations
    pub outputs: Vec<LogOutput>,
    
    /// Log rotation settings
    pub rotation: LogRotationConfig,
    
    /// Structured logging
    pub structured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogFormat {
    Plain,
    Json,
    Logfmt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogOutput {
    Console,
    File(String),
    Syslog,
    Network(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    /// Enable log rotation
    pub enabled: bool,
    
    /// Maximum file size in MB
    pub max_size_mb: u64,
    
    /// Maximum number of files to keep
    pub max_files: u32,
    
    /// Rotation frequency
    pub frequency: RotationFrequency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RotationFrequency {
    Hourly,
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureFlags {
    /// Enable advanced analytics
    pub advanced_analytics: bool,
    
    /// Enable machine learning features
    pub machine_learning: bool,
    
    /// Enable sentiment analysis
    pub sentiment_analysis: bool,
    
    /// Enable high-frequency trading
    pub high_frequency_trading: bool,
    
    /// Enable backtesting
    pub backtesting: bool,
    
    /// Enable API access
    pub api_access: bool,
    
    /// Enable web interface
    pub web_interface: bool,
    
    /// Experimental features
    pub experimental: HashMap<String, bool>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            system: SystemConfig {
                name: "Autopoiesis Trading System".to_string(),
                version: "1.0.0".to_string(),
                environment: Environment::Development,
                worker_threads: num_cpus::get(),
                enable_metrics: true,
                metrics_interval_seconds: 60,
            },
            market_data: MarketDataConfig {
                sources: vec![],
                refresh_interval_ms: 1000,
                max_data_age_seconds: 300,
                enable_validation: true,
                storage: DataStorageSettings {
                    backend: StorageBackend::PostgreSQL,
                    connection_string: "postgresql://localhost/autopoiesis".to_string(),
                    max_connections: 10,
                    retention_days: 90,
                    enable_compression: true,
                },
            },
            trading: TradingConfig {
                strategies: vec![],
                default_position_size: 1000.0,
                max_concurrent_orders: 10,
                order_timeout_seconds: 300,
                paper_trading: true,
                exchanges: vec![],
            },
            risk: RiskConfig {
                global_limits: GlobalRiskLimits {
                    max_total_exposure: 100000.0,
                    max_daily_loss: 5000.0,
                    max_drawdown: 0.20,
                    max_correlation: 0.7,
                    blacklisted_symbols: vec![],
                },
                position_sizing: PositionSizingConfig {
                    method: PositionSizingMethod::PercentOfEquity,
                    base_size: 0.02,
                    volatility_adjustment: 0.5,
                    min_size: 100.0,
                    max_size: 10000.0,
                },
                monitoring: RiskMonitoringConfig {
                    frequency_seconds: 30,
                    var_method: VarMethod::Historical,
                    var_confidence: 0.95,
                    var_lookback_days: 252,
                    enable_alerts: true,
                },
                emergency_stops: EmergencyStopConfig {
                    enabled: true,
                    max_drawdown_trigger: 0.15,
                    daily_loss_trigger: 0.05,
                    volatility_trigger: 3.0,
                    auto_recovery: AutoRecoveryConfig {
                        enabled: false,
                        recovery_delay_minutes: 60,
                        gradual_restoration: true,
                        max_attempts: 3,
                    },
                },
            },
            database: DatabaseConfig {
                url: "postgresql://localhost/autopoiesis".to_string(),
                max_connections: 10,
                connection_timeout_seconds: 30,
                enable_ssl: false,
                migrations: MigrationConfig {
                    auto_migrate: true,
                    migrations_dir: "migrations".to_string(),
                    validate_migrations: true,
                },
            },
            logging: LoggingConfig {
                level: LogLevel::Info,
                format: LogFormat::Json,
                outputs: vec![LogOutput::Console],
                rotation: LogRotationConfig {
                    enabled: true,
                    max_size_mb: 100,
                    max_files: 10,
                    frequency: RotationFrequency::Daily,
                },
                structured: true,
            },
            features: FeatureFlags {
                advanced_analytics: true,
                machine_learning: true,
                sentiment_analysis: true,
                high_frequency_trading: false,
                backtesting: true,
                api_access: true,
                web_interface: false,
                experimental: HashMap::new(),
            },
        }
    }
}

impl Config {
    /// Load configuration from file
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::Config(format!("Failed to read config file: {}", e)))?;
        
        let config: Config = toml::from_str(&content)
            .map_err(|e| Error::Config(format!("Failed to parse config file: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }

    /// Save configuration to file
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| Error::Config(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| Error::Config(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }

    /// Load configuration from environment variables
    pub fn load_from_env() -> Result<Self> {
        let mut config = Config::default();
        
        // Override with environment variables
        if let Ok(env) = std::env::var("AUTOPOIESIS_ENV") {
            config.system.environment = match env.to_lowercase().as_str() {
                "development" => Environment::Development,
                "staging" => Environment::Staging,
                "production" => Environment::Production,
                _ => Environment::Development,
            };
        }

        if let Ok(db_url) = std::env::var("DATABASE_URL") {
            config.database.url = db_url;
            config.market_data.storage.connection_string = config.database.url.clone();
        }

        if let Ok(paper_trading) = std::env::var("PAPER_TRADING") {
            config.trading.paper_trading = paper_trading.parse().unwrap_or(true);
        }

        if let Ok(log_level) = std::env::var("LOG_LEVEL") {
            config.logging.level = match log_level.to_lowercase().as_str() {
                "error" => LogLevel::Error,
                "warn" => LogLevel::Warn,
                "info" => LogLevel::Info,
                "debug" => LogLevel::Debug,
                "trace" => LogLevel::Trace,
                _ => LogLevel::Info,
            };
        }

        config.validate()?;
        Ok(config)
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        // Validate system config
        if self.system.worker_threads == 0 {
            return Err(Error::Config("Worker threads must be greater than 0".to_string()));
        }

        // Validate market data config
        if self.market_data.refresh_interval_ms == 0 {
            return Err(Error::Config("Market data refresh interval must be greater than 0".to_string()));
        }

        // Validate trading config
        if self.trading.default_position_size <= 0.0 {
            return Err(Error::Config("Default position size must be greater than 0".to_string()));
        }

        // Validate risk config
        if self.risk.global_limits.max_total_exposure <= 0.0 {
            return Err(Error::Config("Maximum total exposure must be greater than 0".to_string()));
        }

        if self.risk.global_limits.max_drawdown <= 0.0 || self.risk.global_limits.max_drawdown >= 1.0 {
            return Err(Error::Config("Maximum drawdown must be between 0 and 1".to_string()));
        }

        // Validate database config
        if self.database.url.is_empty() {
            return Err(Error::Config("Database URL cannot be empty".to_string()));
        }

        Ok(())
    }

    /// Get configuration for a specific strategy
    pub fn get_strategy_config(&self, strategy_name: &str) -> Option<&StrategyConfig> {
        self.trading.strategies.iter().find(|s| s.name == strategy_name)
    }

    /// Get configuration for a specific exchange
    pub fn get_exchange_config(&self, exchange_name: &str) -> Option<&ExchangeConfigData> {
        self.trading.exchanges.iter().find(|e| e.name == exchange_name)
    }

    /// Check if a feature is enabled
    pub fn is_feature_enabled(&self, feature_name: &str) -> bool {
        match feature_name {
            "advanced_analytics" => self.features.advanced_analytics,
            "machine_learning" => self.features.machine_learning,
            "sentiment_analysis" => self.features.sentiment_analysis,
            "high_frequency_trading" => self.features.high_frequency_trading,
            "backtesting" => self.features.backtesting,
            "api_access" => self.features.api_access,
            "web_interface" => self.features.web_interface,
            _ => self.features.experimental.get(feature_name).copied().unwrap_or(false),
        }
    }

    /// Enable/disable a feature
    pub fn set_feature_enabled(&mut self, feature_name: &str, enabled: bool) {
        match feature_name {
            "advanced_analytics" => self.features.advanced_analytics = enabled,
            "machine_learning" => self.features.machine_learning = enabled,
            "sentiment_analysis" => self.features.sentiment_analysis = enabled,
            "high_frequency_trading" => self.features.high_frequency_trading = enabled,
            "backtesting" => self.features.backtesting = enabled,
            "api_access" => self.features.api_access = enabled,
            "web_interface" => self.features.web_interface = enabled,
            _ => {
                self.features.experimental.insert(feature_name.to_string(), enabled);
            }
        }
    }

    /// Get environment-specific settings
    pub fn is_production(&self) -> bool {
        matches!(self.system.environment, Environment::Production)
    }

    /// Get environment-specific settings
    pub fn is_development(&self) -> bool {
        matches!(self.system.environment, Environment::Development)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let serialized = toml::to_string(&config).unwrap();
        let deserialized: Config = toml::from_str(&serialized).unwrap();
        
        assert_eq!(config.system.name, deserialized.system.name);
        assert_eq!(config.system.version, deserialized.system.version);
    }

    #[test]
    fn test_config_file_operations() {
        let config = Config::default();
        let temp_file = NamedTempFile::new().unwrap();
        
        // Save config
        config.save_to_file(temp_file.path()).unwrap();
        
        // Load config
        let loaded_config = Config::load_from_file(temp_file.path()).unwrap();
        
        assert_eq!(config.system.name, loaded_config.system.name);
    }

    #[test]
    fn test_feature_flags() {
        let mut config = Config::default();
        
        assert!(config.is_feature_enabled("advanced_analytics"));
        
        config.set_feature_enabled("advanced_analytics", false);
        assert!(!config.is_feature_enabled("advanced_analytics"));
        
        config.set_feature_enabled("custom_feature", true);
        assert!(config.is_feature_enabled("custom_feature"));
    }

    #[test]
    fn test_validation() {
        let mut config = Config::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid config should fail
        config.system.worker_threads = 0;
        assert!(config.validate().is_err());
    }
}