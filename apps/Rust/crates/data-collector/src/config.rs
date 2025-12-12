//! Configuration for data collector

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Main configuration for data collector
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectorConfig {
    /// Exchange configurations
    pub exchanges: ExchangeConfigs,
    /// Storage configuration
    pub storage: StorageConfig,
    /// Data collection settings
    pub collection: CollectionConfig,
    /// Performance settings
    pub performance: PerformanceConfig,
}

/// Exchange configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangeConfigs {
    pub binance: BinanceConfig,
    pub coinbase: CoinbaseConfig,
    pub kraken: KrakenConfig,
    pub okx: OkxConfig,
    pub bybit: BybitConfig,
}

/// Binance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinanceConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub rate_limit_requests_per_minute: u32,
    pub include_futures: bool,
    pub include_options: bool,
    pub testnet: bool,
}

/// Coinbase configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoinbaseConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub passphrase: Option<String>,
    pub rate_limit_requests_per_minute: u32,
    pub sandbox: bool,
}

/// Kraken configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KrakenConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub rate_limit_requests_per_minute: u32,
    pub include_futures: bool,
}

/// OKX configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OkxConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub passphrase: Option<String>,
    pub rate_limit_requests_per_minute: u32,
    pub include_futures: bool,
}

/// Bybit configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BybitConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub rate_limit_requests_per_minute: u32,
    pub include_derivatives: bool,
    pub testnet: bool,
}

/// Storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageConfig {
    /// Primary storage backend
    pub backend: StorageBackend,
    /// Base directory for file storage
    pub base_directory: PathBuf,
    /// Database connection string (for database backends)
    pub database_url: Option<String>,
    /// Enable compression
    pub enable_compression: bool,
    /// Compression algorithm
    pub compression_algorithm: CompressionAlgorithm,
    /// Enable data validation
    pub enable_validation: bool,
    /// Backup configuration
    pub backup: BackupConfig,
}

/// Storage backend options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageBackend {
    /// Parquet files (recommended for analytics)
    Parquet,
    /// CSV files (human readable)
    Csv,
    /// SQLite database (embedded)
    Sqlite,
    /// PostgreSQL database (scalable)
    Postgresql,
    /// Combined storage (multiple backends)
    Combined(Vec<StorageBackend>),
}

/// Compression algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    None,
    Gzip,
    Zstd,
    Lz4,
}

/// Backup configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackupConfig {
    pub enabled: bool,
    pub backup_directory: PathBuf,
    pub backup_interval_hours: u64,
    pub max_backup_files: u32,
    pub backup_compression: bool,
}

/// Data collection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Default symbols to collect
    pub default_symbols: Vec<String>,
    /// Default time intervals
    pub default_intervals: Vec<String>,
    /// Default data types
    pub default_data_types: Vec<String>,
    /// Maximum concurrent downloads
    pub max_concurrent_downloads: u32,
    /// Retry attempts for failed requests
    pub retry_attempts: u32,
    /// Delay between retries (seconds)
    pub retry_delay_seconds: u64,
    /// Enable data quality checks
    pub enable_quality_checks: bool,
    /// Quality check thresholds
    pub quality_thresholds: QualityThresholds,
}

/// Data quality thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    /// Minimum success rate for data collection
    pub min_success_rate: f64,
    /// Maximum allowed gaps in time series (percentage)
    pub max_gap_percentage: f64,
    /// Maximum price deviation (standard deviations)
    pub max_price_deviation: f64,
    /// Minimum trade count per period
    pub min_trade_count: u64,
}

/// Performance configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable parallel processing
    pub enable_parallel_processing: bool,
    /// Number of worker threads
    pub worker_threads: Option<u32>,
    /// Memory limit for data processing (MB)
    pub memory_limit_mb: u64,
    /// Enable progress reporting
    pub enable_progress_reporting: bool,
    /// Progress report interval (seconds)
    pub progress_report_interval_seconds: u64,
    /// Enable metrics collection
    pub enable_metrics: bool,
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self {
            exchanges: ExchangeConfigs::default(),
            storage: StorageConfig::default(),
            collection: CollectionConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
}

impl Default for ExchangeConfigs {
    fn default() -> Self {
        Self {
            binance: BinanceConfig::default(),
            coinbase: CoinbaseConfig::default(),
            kraken: KrakenConfig::default(),
            okx: OkxConfig::default(),
            bybit: BybitConfig::default(),
        }
    }
}

impl Default for BinanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            api_key: None,
            secret_key: None,
            rate_limit_requests_per_minute: 1200,
            include_futures: true,
            include_options: false,
            testnet: false,
        }
    }
}

impl Default for CoinbaseConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            api_key: None,
            secret_key: None,
            passphrase: None,
            rate_limit_requests_per_minute: 300,
            sandbox: false,
        }
    }
}

impl Default for KrakenConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            api_key: None,
            secret_key: None,
            rate_limit_requests_per_minute: 180,
            include_futures: false,
        }
    }
}

impl Default for OkxConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default
            api_key: None,
            secret_key: None,
            passphrase: None,
            rate_limit_requests_per_minute: 600,
            include_futures: true,
        }
    }
}

impl Default for BybitConfig {
    fn default() -> Self {
        Self {
            enabled: false, // Disabled by default
            api_key: None,
            secret_key: None,
            rate_limit_requests_per_minute: 600,
            include_derivatives: true,
            testnet: false,
        }
    }
}

impl Default for StorageConfig {
    fn default() -> Self {
        Self {
            backend: StorageBackend::Parquet,
            base_directory: PathBuf::from("./data"),
            database_url: None,
            enable_compression: true,
            compression_algorithm: CompressionAlgorithm::Zstd,
            enable_validation: true,
            backup: BackupConfig::default(),
        }
    }
}

impl Default for BackupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            backup_directory: PathBuf::from("./backups"),
            backup_interval_hours: 24,
            max_backup_files: 30,
            backup_compression: true,
        }
    }
}

impl Default for CollectionConfig {
    fn default() -> Self {
        Self {
            default_symbols: vec![
                "BTCUSDT".to_string(),
                "ETHUSDT".to_string(),
                "ADAUSDT".to_string(),
                "SOLUSDT".to_string(),
                "DOTUSDT".to_string(),
                "MATICUSDT".to_string(),
                "LINKUSDT".to_string(),
                "AVAXUSDT".to_string(),
                "ATOMUSDT".to_string(),
                "NEARUSDT".to_string(),
            ],
            default_intervals: vec![
                "1m".to_string(),
                "5m".to_string(),
                "15m".to_string(),
                "1h".to_string(),
                "4h".to_string(),
                "1d".to_string(),
            ],
            default_data_types: vec![
                "klines".to_string(),
                "trades".to_string(),
                "ticker_24hr".to_string(),
            ],
            max_concurrent_downloads: 10,
            retry_attempts: 3,
            retry_delay_seconds: 5,
            enable_quality_checks: true,
            quality_thresholds: QualityThresholds::default(),
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_success_rate: 0.95,
            max_gap_percentage: 1.0,
            max_price_deviation: 5.0,
            min_trade_count: 10,
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enable_parallel_processing: true,
            worker_threads: None, // Use system default
            memory_limit_mb: 4096, // 4GB default
            enable_progress_reporting: true,
            progress_report_interval_seconds: 30,
            enable_metrics: true,
        }
    }
}

impl CollectorConfig {
    /// Load configuration from file
    pub fn load_from_file(path: &str) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| crate::DataCollectorError::Config(format!("Failed to read config file: {}", e)))?;
            
        let config: Self = toml::from_str(&content)
            .map_err(|e| crate::DataCollectorError::Config(format!("Failed to parse config: {}", e)))?;
            
        Ok(config)
    }
    
    /// Save configuration to file
    pub fn save_to_file(&self, path: &str) -> crate::Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| crate::DataCollectorError::Config(format!("Failed to serialize config: {}", e)))?;
            
        std::fs::write(path, content)
            .map_err(|e| crate::DataCollectorError::Config(format!("Failed to write config file: {}", e)))?;
            
        Ok(())
    }
    
    /// Validate configuration
    pub fn validate(&self) -> crate::Result<Vec<String>> {
        let mut warnings = Vec::new();
        
        // Check if at least one exchange is enabled
        if !self.exchanges.binance.enabled && 
           !self.exchanges.coinbase.enabled && 
           !self.exchanges.kraken.enabled &&
           !self.exchanges.okx.enabled &&
           !self.exchanges.bybit.enabled {
            return Err(crate::DataCollectorError::Config("No exchanges enabled".to_string()));
        }
        
        // Check storage directory exists or can be created
        if !self.storage.base_directory.exists() {
            std::fs::create_dir_all(&self.storage.base_directory)
                .map_err(|e| crate::DataCollectorError::Config(
                    format!("Cannot create storage directory: {}", e)
                ))?;
        }
        
        // Check concurrent downloads limit
        if self.collection.max_concurrent_downloads > 50 {
            warnings.push("High concurrent downloads may trigger rate limits".to_string());
        }
        
        // Check memory limit
        if self.performance.memory_limit_mb < 1024 {
            warnings.push("Low memory limit may impact performance".to_string());
        }
        
        Ok(warnings)
    }
    
    /// Get enabled exchanges
    pub fn get_enabled_exchanges(&self) -> Vec<String> {
        let mut exchanges = Vec::new();
        
        if self.exchanges.binance.enabled {
            exchanges.push("binance".to_string());
        }
        if self.exchanges.coinbase.enabled {
            exchanges.push("coinbase".to_string());
        }
        if self.exchanges.kraken.enabled {
            exchanges.push("kraken".to_string());
        }
        if self.exchanges.okx.enabled {
            exchanges.push("okx".to_string());
        }
        if self.exchanges.bybit.enabled {
            exchanges.push("bybit".to_string());
        }
        
        exchanges
    }
}