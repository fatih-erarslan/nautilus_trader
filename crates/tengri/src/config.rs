//! Configuration management for Tengri trading strategy
//! 
//! Provides comprehensive configuration management for all strategy components
//! including data sources, trading parameters, risk management, and execution settings.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use crate::{Result, TengriError};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TengriConfig {
    /// Strategy configuration
    pub strategy: StrategyConfig,
    
    /// Data source configurations
    pub data_sources: DataSourcesConfig,
    
    /// Exchange configurations
    pub exchanges: ExchangesConfig,
    
    /// Risk management configuration
    pub risk: RiskConfig,
    
    /// Execution configuration
    pub execution: ExecutionConfig,
    
    /// Monitoring and alerting configuration
    pub monitoring: MonitoringConfig,
    
    /// Performance optimization settings
    pub performance: PerformanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy name
    pub name: String,
    
    /// Strategy version
    pub version: String,
    
    /// Base currency for the strategy
    pub base_currency: String,
    
    /// Quote currency for the strategy
    pub quote_currency: String,
    
    /// Trading instruments to monitor
    pub instruments: Vec<String>,
    
    /// Strategy parameters
    pub parameters: StrategyParameters,
    
    /// Strategy mode (live, paper, backtest)
    pub mode: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyParameters {
    /// Timeframes for analysis (in seconds)
    pub timeframes: Vec<u64>,
    
    /// Look-back period for analysis
    pub lookback_period: u64,
    
    /// Minimum price movement to trigger action
    pub min_price_movement: f64,
    
    /// Maximum position size as percentage of capital
    pub max_position_size: f64,
    
    /// Signal confidence threshold
    pub signal_threshold: f64,
    
    /// Correlation threshold for cross-asset analysis
    pub correlation_threshold: f64,
    
    /// Volatility adjustment factor
    pub volatility_factor: f64,
    
    /// Trend following parameters
    pub trend_following: TrendFollowingParams,
    
    /// Mean reversion parameters
    pub mean_reversion: MeanReversionParams,
    
    /// Momentum parameters
    pub momentum: MomentumParams,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendFollowingParams {
    pub enabled: bool,
    pub ema_fast: u64,
    pub ema_slow: u64,
    pub atr_period: u64,
    pub breakout_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeanReversionParams {
    pub enabled: bool,
    pub bollinger_period: u64,
    pub bollinger_std: f64,
    pub rsi_period: u64,
    pub rsi_oversold: f64,
    pub rsi_overbought: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumParams {
    pub enabled: bool,
    pub macd_fast: u64,
    pub macd_slow: u64,
    pub macd_signal: u64,
    pub momentum_period: u64,
    pub momentum_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSourcesConfig {
    /// Databento configuration
    pub databento: DatabentoConfig,
    
    /// Tardis configuration
    pub tardis: TardisConfig,
    
    /// Polymarket configuration
    pub polymarket: PolymarketConfig,
    
    /// Data aggregation settings
    pub aggregation: DataAggregationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabentoConfig {
    pub enabled: bool,
    pub api_key: String,
    pub dataset: String,
    pub symbols: Vec<String>,
    pub schema: String,
    pub stype_in: String,
    pub start: Option<String>,
    pub end: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TardisConfig {
    pub enabled: bool,
    pub api_key: String,
    pub exchanges: Vec<String>,
    pub symbols: Vec<String>,
    pub data_types: Vec<String>,
    pub from_date: String,
    pub to_date: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolymarketConfig {
    pub enabled: bool,
    pub api_key: Option<String>,
    pub markets: Vec<String>,
    pub categories: Vec<String>,
    pub min_liquidity: f64,
    pub update_interval: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAggregationConfig {
    /// Data refresh interval in seconds
    pub refresh_interval: u64,
    
    /// Maximum data age before refresh
    pub max_data_age: u64,
    
    /// Data quality thresholds
    pub quality_thresholds: QualityThresholds,
    
    /// Cross-source data validation
    pub validation: ValidationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityThresholds {
    pub min_data_points: u64,
    pub max_missing_ratio: f64,
    pub max_latency_ms: u64,
    pub min_confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enabled: bool,
    pub cross_validate_sources: bool,
    pub max_price_deviation: f64,
    pub max_volume_deviation: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExchangesConfig {
    /// Binance Spot configuration
    pub binance_spot: BinanceConfig,
    
    /// Binance Futures configuration
    pub binance_futures: BinanceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinanceConfig {
    pub enabled: bool,
    pub api_key: String,
    pub api_secret: String,
    pub testnet: bool,
    pub base_url: Option<String>,
    pub rate_limits: RateLimitConfig,
    pub order_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub orders_per_second: u32,
    pub weight_per_minute: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskConfig {
    /// Maximum portfolio loss percentage
    pub max_portfolio_loss: f64,
    
    /// Maximum daily loss percentage
    pub max_daily_loss: f64,
    
    /// Maximum position size per instrument
    pub max_position_size: f64,
    
    /// Position sizing model
    pub position_sizing: PositionSizingConfig,
    
    /// Stop loss configuration
    pub stop_loss: StopLossConfig,
    
    /// Take profit configuration
    pub take_profit: TakeProfitConfig,
    
    /// Correlation limits
    pub correlation_limits: CorrelationLimits,
    
    /// Volatility controls
    pub volatility_controls: VolatilityControls,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PositionSizingConfig {
    pub method: String, // "fixed", "kelly", "volatility_adjusted", "risk_parity"
    pub base_size: f64,
    pub volatility_lookback: u64,
    pub kelly_fraction: f64,
    pub max_leverage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StopLossConfig {
    pub enabled: bool,
    pub method: String, // "fixed", "atr", "volatility"
    pub percentage: f64,
    pub atr_multiplier: f64,
    pub trailing: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TakeProfitConfig {
    pub enabled: bool,
    pub method: String, // "fixed", "risk_reward", "dynamic"
    pub percentage: f64,
    pub risk_reward_ratio: f64,
    pub partial_profits: Vec<PartialProfitLevel>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartialProfitLevel {
    pub price_level: f64,
    pub quantity_percentage: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationLimits {
    pub max_correlation: f64,
    pub correlation_window: u64,
    pub max_correlated_positions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolatilityControls {
    pub max_volatility: f64,
    pub volatility_window: u64,
    pub volatility_adjustment: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionConfig {
    /// Order execution parameters
    pub execution_style: String, // "aggressive", "passive", "adaptive"
    
    /// Order types to use
    pub order_types: OrderTypeConfig,
    
    /// Slippage controls
    pub slippage: SlippageConfig,
    
    /// Fill management
    pub fill_management: FillManagementConfig,
    
    /// Latency optimization
    pub latency_optimization: LatencyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderTypeConfig {
    pub market_orders: bool,
    pub limit_orders: bool,
    pub stop_orders: bool,
    pub iceberg_orders: bool,
    pub twap_orders: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlippageConfig {
    pub max_slippage: f64,
    pub slippage_model: String, // "linear", "sqrt", "impact"
    pub market_impact_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillManagementConfig {
    pub partial_fill_handling: String, // "accumulate", "cancel", "replace"
    pub fill_timeout_seconds: u64,
    pub price_improvement_threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyConfig {
    pub co_location: bool,
    pub connection_pooling: bool,
    pub message_batching: bool,
    pub priority_queue: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Performance monitoring
    pub performance: PerformanceMonitoringConfig,
    
    /// Alerting configuration
    pub alerting: AlertingConfig,
    
    /// Logging configuration
    pub logging: LoggingConfig,
    
    /// Metrics collection
    pub metrics: MetricsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitoringConfig {
    pub enabled: bool,
    pub update_interval: u64,
    pub metrics_retention: u64,
    pub benchmark_comparison: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    pub enabled: bool,
    pub channels: Vec<AlertChannel>,
    pub thresholds: AlertThresholds,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertChannel {
    pub channel_type: String, // "email", "slack", "telegram", "webhook"
    pub config: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub max_drawdown: f64,
    pub min_sharpe_ratio: f64,
    pub max_correlation: f64,
    pub latency_threshold_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String, // "debug", "info", "warn", "error"
    pub output: String, // "console", "file", "both"
    pub file_path: Option<String>,
    pub rotation: LogRotationConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogRotationConfig {
    pub enabled: bool,
    pub max_size_mb: u64,
    pub max_files: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub enabled: bool,
    pub export_interval: u64,
    pub exporters: Vec<MetricsExporter>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsExporter {
    pub exporter_type: String, // "prometheus", "influxdb", "datadog"
    pub config: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// CPU optimization settings
    pub cpu: CpuConfig,
    
    /// GPU acceleration settings
    pub gpu: GpuConfig,
    
    /// Memory management settings
    pub memory: MemoryConfig,
    
    /// Network optimization settings
    pub network: NetworkConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuConfig {
    pub thread_pool_size: Option<usize>,
    pub enable_simd: bool,
    pub enable_vectorization: bool,
    pub cpu_affinity: Option<Vec<usize>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub enabled: bool,
    pub device_id: Option<u32>,
    pub memory_pool_size_mb: Option<u64>,
    pub operations: GpuOperationsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuOperationsConfig {
    pub statistical_calculations: bool,
    pub technical_indicators: bool,
    pub risk_calculations: bool,
    pub portfolio_optimization: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub enable_memory_pool: bool,
    pub pool_size_mb: Option<u64>,
    pub enable_compression: bool,
    pub cache_size_mb: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkConfig {
    pub connection_pooling: bool,
    pub keep_alive: bool,
    pub timeout_seconds: u64,
    pub retry_attempts: u32,
    pub enable_compression: bool,
}

impl TengriConfig {
    /// Load configuration from a TOML file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| TengriError::Config(format!("Failed to read config file: {}", e)))?;
        
        let config: TengriConfig = toml::from_str(&content)
            .map_err(|e| TengriError::Config(format!("Failed to parse config: {}", e)))?;
        
        config.validate()?;
        Ok(config)
    }
    
    /// Save configuration to a TOML file
    pub fn to_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .map_err(|e| TengriError::Config(format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| TengriError::Config(format!("Failed to write config file: {}", e)))?;
        
        Ok(())
    }
    
    /// Create a default configuration
    pub fn default() -> Self {
        Self {
            strategy: StrategyConfig::default(),
            data_sources: DataSourcesConfig::default(),
            exchanges: ExchangesConfig::default(),
            risk: RiskConfig::default(),
            execution: ExecutionConfig::default(),
            monitoring: MonitoringConfig::default(),
            performance: PerformanceConfig::default(),
        }
    }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<()> {
        // Validate strategy parameters
        if self.strategy.parameters.max_position_size <= 0.0 || self.strategy.parameters.max_position_size > 1.0 {
            return Err(TengriError::Config(
                "max_position_size must be between 0 and 1".to_string()
            ));
        }
        
        // Validate risk parameters
        if self.risk.max_portfolio_loss <= 0.0 || self.risk.max_portfolio_loss > 1.0 {
            return Err(TengriError::Config(
                "max_portfolio_loss must be between 0 and 1".to_string()
            ));
        }
        
        // Validate data source configurations
        if !self.data_sources.databento.enabled && 
           !self.data_sources.tardis.enabled && 
           !self.data_sources.polymarket.enabled {
            return Err(TengriError::Config(
                "At least one data source must be enabled".to_string()
            ));
        }
        
        // Validate exchange configurations
        if !self.exchanges.binance_spot.enabled && !self.exchanges.binance_futures.enabled {
            return Err(TengriError::Config(
                "At least one exchange must be enabled".to_string()
            ));
        }
        
        Ok(())
    }
    
    /// Get strategy instruments
    pub fn get_instruments(&self) -> &Vec<String> {
        &self.strategy.instruments
    }
    
    /// Check if GPU acceleration is enabled
    pub fn is_gpu_enabled(&self) -> bool {
        self.performance.gpu.enabled
    }
    
    /// Get active data sources
    pub fn get_active_data_sources(&self) -> Vec<String> {
        let mut sources = Vec::new();
        
        if self.data_sources.databento.enabled {
            sources.push("databento".to_string());
        }
        
        if self.data_sources.tardis.enabled {
            sources.push("tardis".to_string());
        }
        
        if self.data_sources.polymarket.enabled {
            sources.push("polymarket".to_string());
        }
        
        sources
    }
}

// Default implementations for all config structs
impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            name: "TengriStrategy".to_string(),
            version: "1.0.0".to_string(),
            base_currency: "BTC".to_string(),
            quote_currency: "USDT".to_string(),
            instruments: vec!["BTCUSDT".to_string(), "ETHUSDT".to_string()],
            parameters: StrategyParameters::default(),
            mode: "paper".to_string(),
        }
    }
}

impl Default for StrategyParameters {
    fn default() -> Self {
        Self {
            timeframes: vec![60, 300, 900, 3600], // 1m, 5m, 15m, 1h
            lookback_period: 100,
            min_price_movement: 0.001,
            max_position_size: 0.1,
            signal_threshold: 0.7,
            correlation_threshold: 0.8,
            volatility_factor: 1.0,
            trend_following: TrendFollowingParams::default(),
            mean_reversion: MeanReversionParams::default(),
            momentum: MomentumParams::default(),
        }
    }
}

impl Default for TrendFollowingParams {
    fn default() -> Self {
        Self {
            enabled: true,
            ema_fast: 12,
            ema_slow: 26,
            atr_period: 14,
            breakout_threshold: 0.02,
        }
    }
}

impl Default for MeanReversionParams {
    fn default() -> Self {
        Self {
            enabled: true,
            bollinger_period: 20,
            bollinger_std: 2.0,
            rsi_period: 14,
            rsi_oversold: 30.0,
            rsi_overbought: 70.0,
        }
    }
}

impl Default for MomentumParams {
    fn default() -> Self {
        Self {
            enabled: true,
            macd_fast: 12,
            macd_slow: 26,
            macd_signal: 9,
            momentum_period: 10,
            momentum_threshold: 0.05,
        }
    }
}

impl Default for DataSourcesConfig {
    fn default() -> Self {
        Self {
            databento: DatabentoConfig::default(),
            tardis: TardisConfig::default(),
            polymarket: PolymarketConfig::default(),
            aggregation: DataAggregationConfig::default(),
        }
    }
}

impl Default for DatabentoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: "your_databento_api_key".to_string(),
            dataset: "GLBX.MDP3".to_string(),
            symbols: vec!["ESM4".to_string()],
            schema: "mbo".to_string(),
            stype_in: "raw_symbol".to_string(),
            start: None,
            end: None,
        }
    }
}

impl Default for TardisConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: "your_tardis_api_key".to_string(),
            exchanges: vec!["binance".to_string()],
            symbols: vec!["BTCUSDT".to_string()],
            data_types: vec!["trade".to_string(), "book_change".to_string()],
            from_date: "2024-01-01".to_string(),
            to_date: None,
        }
    }
}

impl Default for PolymarketConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            api_key: None,
            markets: vec!["crypto".to_string()],
            categories: vec!["politics".to_string(), "economics".to_string()],
            min_liquidity: 1000.0,
            update_interval: 300,
        }
    }
}

impl Default for DataAggregationConfig {
    fn default() -> Self {
        Self {
            refresh_interval: 60,
            max_data_age: 300,
            quality_thresholds: QualityThresholds::default(),
            validation: ValidationConfig::default(),
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_data_points: 100,
            max_missing_ratio: 0.05,
            max_latency_ms: 1000,
            min_confidence: 0.8,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cross_validate_sources: true,
            max_price_deviation: 0.01,
            max_volume_deviation: 0.1,
        }
    }
}

impl Default for ExchangesConfig {
    fn default() -> Self {
        Self {
            binance_spot: BinanceConfig::default_spot(),
            binance_futures: BinanceConfig::default_futures(),
        }
    }
}

impl BinanceConfig {
    fn default_spot() -> Self {
        Self {
            enabled: true,
            api_key: "your_binance_api_key".to_string(),
            api_secret: "your_binance_api_secret".to_string(),
            testnet: true,
            base_url: None,
            rate_limits: RateLimitConfig::default(),
            order_types: vec!["MARKET".to_string(), "LIMIT".to_string(), "STOP_LOSS".to_string()],
        }
    }
    
    fn default_futures() -> Self {
        Self {
            enabled: false,
            api_key: "your_binance_futures_api_key".to_string(),
            api_secret: "your_binance_futures_api_secret".to_string(),
            testnet: true,
            base_url: Some("https://testnet.binancefuture.com".to_string()),
            rate_limits: RateLimitConfig::default(),
            order_types: vec!["MARKET".to_string(), "LIMIT".to_string(), "STOP_MARKET".to_string()],
        }
    }
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 1200,
            orders_per_second: 10,
            weight_per_minute: 6000,
        }
    }
}

impl Default for RiskConfig {
    fn default() -> Self {
        Self {
            max_portfolio_loss: 0.05,
            max_daily_loss: 0.02,
            max_position_size: 0.1,
            position_sizing: PositionSizingConfig::default(),
            stop_loss: StopLossConfig::default(),
            take_profit: TakeProfitConfig::default(),
            correlation_limits: CorrelationLimits::default(),
            volatility_controls: VolatilityControls::default(),
        }
    }
}

impl Default for PositionSizingConfig {
    fn default() -> Self {
        Self {
            method: "volatility_adjusted".to_string(),
            base_size: 0.05,
            volatility_lookback: 20,
            kelly_fraction: 0.25,
            max_leverage: 1.0,
        }
    }
}

impl Default for StopLossConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: "atr".to_string(),
            percentage: 0.02,
            atr_multiplier: 2.0,
            trailing: true,
        }
    }
}

impl Default for TakeProfitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            method: "risk_reward".to_string(),
            percentage: 0.04,
            risk_reward_ratio: 2.0,
            partial_profits: vec![
                PartialProfitLevel { price_level: 0.02, quantity_percentage: 0.5 },
                PartialProfitLevel { price_level: 0.04, quantity_percentage: 0.5 },
            ],
        }
    }
}

impl Default for CorrelationLimits {
    fn default() -> Self {
        Self {
            max_correlation: 0.8,
            correlation_window: 50,
            max_correlated_positions: 3,
        }
    }
}

impl Default for VolatilityControls {
    fn default() -> Self {
        Self {
            max_volatility: 0.1,
            volatility_window: 20,
            volatility_adjustment: true,
        }
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            execution_style: "adaptive".to_string(),
            order_types: OrderTypeConfig::default(),
            slippage: SlippageConfig::default(),
            fill_management: FillManagementConfig::default(),
            latency_optimization: LatencyConfig::default(),
        }
    }
}

impl Default for OrderTypeConfig {
    fn default() -> Self {
        Self {
            market_orders: true,
            limit_orders: true,
            stop_orders: true,
            iceberg_orders: false,
            twap_orders: false,
        }
    }
}

impl Default for SlippageConfig {
    fn default() -> Self {
        Self {
            max_slippage: 0.001,
            slippage_model: "sqrt".to_string(),
            market_impact_factor: 0.5,
        }
    }
}

impl Default for FillManagementConfig {
    fn default() -> Self {
        Self {
            partial_fill_handling: "accumulate".to_string(),
            fill_timeout_seconds: 30,
            price_improvement_threshold: 0.0001,
        }
    }
}

impl Default for LatencyConfig {
    fn default() -> Self {
        Self {
            co_location: false,
            connection_pooling: true,
            message_batching: true,
            priority_queue: true,
        }
    }
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            performance: PerformanceMonitoringConfig::default(),
            alerting: AlertingConfig::default(),
            logging: LoggingConfig::default(),
            metrics: MetricsConfig::default(),
        }
    }
}

impl Default for PerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            update_interval: 60,
            metrics_retention: 86400, // 24 hours
            benchmark_comparison: true,
        }
    }
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![],
            thresholds: AlertThresholds::default(),
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            max_drawdown: 0.05,
            min_sharpe_ratio: 0.5,
            max_correlation: 0.9,
            latency_threshold_ms: 100,
        }
    }
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            output: "both".to_string(),
            file_path: Some("logs/tengri.log".to_string()),
            rotation: LogRotationConfig::default(),
        }
    }
}

impl Default for LogRotationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_size_mb: 100,
            max_files: 10,
        }
    }
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            export_interval: 60,
            exporters: vec![],
        }
    }
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            cpu: CpuConfig::default(),
            gpu: GpuConfig::default(),
            memory: MemoryConfig::default(),
            network: NetworkConfig::default(),
        }
    }
}

impl Default for CpuConfig {
    fn default() -> Self {
        Self {
            thread_pool_size: None,
            enable_simd: true,
            enable_vectorization: true,
            cpu_affinity: None,
        }
    }
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_id: Some(0),
            memory_pool_size_mb: Some(1024),
            operations: GpuOperationsConfig::default(),
        }
    }
}

impl Default for GpuOperationsConfig {
    fn default() -> Self {
        Self {
            statistical_calculations: true,
            technical_indicators: true,
            risk_calculations: true,
            portfolio_optimization: true,
        }
    }
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enable_memory_pool: true,
            pool_size_mb: Some(512),
            enable_compression: true,
            cache_size_mb: Some(256),
        }
    }
}

impl Default for NetworkConfig {
    fn default() -> Self {
        Self {
            connection_pooling: true,
            keep_alive: true,
            timeout_seconds: 30,
            retry_attempts: 3,
            enable_compression: true,
        }
    }
}