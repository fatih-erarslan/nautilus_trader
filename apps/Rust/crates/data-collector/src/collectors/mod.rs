//! Data collectors for various cryptocurrency exchanges

pub mod binance;
pub mod coinbase;
// pub mod kraken;
// pub mod okx;
// pub mod bybit;

use crate::Result;
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market data types that can be collected
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum DataType {
    /// OHLCV candlestick data
    Klines,
    /// Individual trades
    Trades,
    /// Order book snapshots
    OrderBook,
    /// Funding rates (for futures)
    FundingRates,
    /// 24hr ticker statistics
    Ticker24hr,
    /// Mark price (for derivatives)
    MarkPrice,
}

/// Time intervals for candlestick data
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Interval {
    #[serde(rename = "1m")]
    OneMinute,
    #[serde(rename = "3m")]
    ThreeMinutes,
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "15m")]
    FifteenMinutes,
    #[serde(rename = "30m")]
    ThirtyMinutes,
    #[serde(rename = "1h")]
    OneHour,
    #[serde(rename = "2h")]
    TwoHours,
    #[serde(rename = "4h")]
    FourHours,
    #[serde(rename = "6h")]
    SixHours,
    #[serde(rename = "8h")]
    EightHours,
    #[serde(rename = "12h")]
    TwelveHours,
    #[serde(rename = "1d")]
    OneDay,
    #[serde(rename = "3d")]
    ThreeDays,
    #[serde(rename = "1w")]
    OneWeek,
    #[serde(rename = "1M")]
    OneMonth,
}

/// OHLCV candlestick data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Kline {
    pub symbol: String,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub trades_count: u64,
    pub taker_buy_base_volume: f64,
    pub taker_buy_quote_volume: f64,
    pub interval: Interval,
    pub exchange: String,
}

/// Individual trade data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Trade {
    pub symbol: String,
    pub trade_id: u64,
    pub price: f64,
    pub quantity: f64,
    pub quote_quantity: f64,
    pub timestamp: DateTime<Utc>,
    pub is_buyer_maker: bool,
    pub exchange: String,
}

/// Order book data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBook {
    pub symbol: String,
    pub timestamp: DateTime<Utc>,
    pub bids: Vec<OrderBookLevel>,
    pub asks: Vec<OrderBookLevel>,
    pub exchange: String,
}

/// Order book price level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderBookLevel {
    pub price: f64,
    pub quantity: f64,
}

/// Funding rate data (for futures)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub funding_rate: f64,
    pub funding_time: DateTime<Utc>,
    pub mark_price: f64,
    pub exchange: String,
}

/// 24hr ticker statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ticker24hr {
    pub symbol: String,
    pub price_change: f64,
    pub price_change_percent: f64,
    pub weighted_avg_price: f64,
    pub prev_close_price: f64,
    pub last_price: f64,
    pub last_qty: f64,
    pub bid_price: f64,
    pub bid_qty: f64,
    pub ask_price: f64,
    pub ask_qty: f64,
    pub open_price: f64,
    pub high_price: f64,
    pub low_price: f64,
    pub volume: f64,
    pub quote_volume: f64,
    pub open_time: DateTime<Utc>,
    pub close_time: DateTime<Utc>,
    pub first_id: u64,
    pub last_id: u64,
    pub count: u64,
    pub exchange: String,
}

/// Collection parameters for data requests
#[derive(Debug, Clone)]
pub struct CollectionParams {
    pub symbols: Vec<String>,
    pub data_types: Vec<DataType>,
    pub interval: Option<Interval>,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub limit: Option<u32>,
}

/// Collection statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionStats {
    pub total_records: u64,
    pub successful_requests: u64,
    pub failed_requests: u64,
    pub data_quality_score: f64,
    pub collection_duration_ms: u64,
    pub average_latency_ms: f64,
    pub rate_limit_hits: u64,
}

/// Trait for cryptocurrency exchange data collectors
#[async_trait]
pub trait ExchangeCollector: Send + Sync {
    /// Get the exchange name
    fn exchange_name(&self) -> &str;
    
    /// Get available trading symbols
    async fn get_symbols(&self) -> Result<Vec<String>>;
    
    /// Get server time for synchronization
    async fn get_server_time(&self) -> Result<DateTime<Utc>>;
    
    /// Collect OHLCV kline data
    async fn collect_klines(
        &self, 
        symbol: &str, 
        interval: Interval,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        limit: Option<u32>
    ) -> Result<Vec<Kline>>;
    
    /// Collect trade data
    async fn collect_trades(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>, 
        end_time: DateTime<Utc>,
        limit: Option<u32>
    ) -> Result<Vec<Trade>>;
    
    /// Collect order book snapshot
    async fn collect_order_book(&self, symbol: &str, limit: Option<u32>) -> Result<OrderBook>;
    
    /// Collect 24hr ticker data
    async fn collect_ticker_24hr(&self, symbol: Option<&str>) -> Result<Vec<Ticker24hr>>;
    
    /// Collect funding rates (for futures exchanges)
    async fn collect_funding_rates(&self, symbol: Option<&str>) -> Result<Vec<FundingRate>> {
        // Default implementation for spot exchanges
        Ok(vec![])
    }
    
    /// Validate API connectivity
    async fn validate_connection(&self) -> Result<bool>;
    
    /// Get rate limit information
    fn get_rate_limits(&self) -> HashMap<String, u32>;
}

/// Main data collector that coordinates multiple exchanges
pub struct DataCollector {
    collectors: HashMap<String, Box<dyn ExchangeCollector>>,
    storage: Option<crate::storage::StorageManager>,
    config: crate::config::CollectorConfig,
}

impl DataCollector {
    /// Create a new data collector with default configuration
    pub async fn new() -> Result<Self> {
        let config = crate::config::CollectorConfig::default();
        Self::with_config(config).await
    }
    
    /// Create a new data collector with custom configuration
    pub async fn with_config(config: crate::config::CollectorConfig) -> Result<Self> {
        let mut collectors: HashMap<String, Box<dyn ExchangeCollector>> = HashMap::new();
        
        // Initialize enabled exchange collectors
        if config.exchanges.binance.enabled {
            collectors.insert(
                "binance".to_string(),
                Box::new(binance::BinanceCollector::new(&config.exchanges.binance).await?)
            );
        }
        
        if config.exchanges.coinbase.enabled {
            collectors.insert(
                "coinbase".to_string(),
                Box::new(coinbase::CoinbaseCollector::new(&config.exchanges.coinbase).await?)
            );
        }
        
        // if config.exchanges.kraken.enabled {
        //     collectors.insert(
        //         "kraken".to_string(),
        //         Box::new(kraken::KrakenCollector::new(&config.exchanges.kraken).await?)
        //     );
        // }
        
        // Initialize storage manager
        let storage = Some(crate::storage::StorageManager::new(config.storage.clone()).await?);
        
        Ok(Self { collectors, storage, config })
    }
    
    /// Download historical data for specific parameters
    pub async fn download_historical_data(
        &self,
        exchange: &str,
        symbol: &str,
        interval: &str,
        start_date: &str,
        end_date: &str,
    ) -> Result<CollectionStats> {
        let collector = self.collectors.get(exchange)
            .ok_or_else(|| crate::DataCollectorError::Config(format!("Exchange {} not supported", exchange)))?;
        
        let interval = self.parse_interval(interval)?;
        let start_time = chrono::DateTime::parse_from_str(&format!("{} 00:00:00 +0000", start_date), "%Y-%m-%d %H:%M:%S %z")?
            .with_timezone(&Utc);
        let end_time = chrono::DateTime::parse_from_str(&format!("{} 23:59:59 +0000", end_date), "%Y-%m-%d %H:%M:%S %z")?
            .with_timezone(&Utc);
        
        tracing::info!("Starting historical data download for {}/{} from {} to {}", 
                      exchange, symbol, start_date, end_date);
        
        let start_collection = std::time::Instant::now();
        let mut total_records = 0u64;
        let mut successful_requests = 0u64;
        let mut failed_requests = 0u64;
        
        // Download data in chunks to respect rate limits
        let chunk_size = self.calculate_optimal_chunk_size(&interval);
        let mut current_start = start_time;
        
        while current_start < end_time {
            let chunk_end = std::cmp::min(current_start + chunk_size, end_time);
            
            match collector.collect_klines(symbol, interval.clone(), current_start, chunk_end, None).await {
                Ok(klines) => {
                    total_records += klines.len() as u64;
                    successful_requests += 1;
                    
                    // Store the data
                    self.store_klines(&klines).await?;
                    
                    tracing::debug!("Downloaded {} records for chunk {} to {}", 
                                   klines.len(), current_start, chunk_end);
                },
                Err(e) => {
                    failed_requests += 1;
                    tracing::warn!("Failed to download chunk {} to {}: {}", current_start, chunk_end, e);
                }
            }
            
            current_start = chunk_end;
            
            // Respect rate limits
            if successful_requests % 10 == 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
        }
        
        let collection_duration = start_collection.elapsed();
        
        Ok(CollectionStats {
            total_records,
            successful_requests,
            failed_requests,
            data_quality_score: self.calculate_data_quality_score(successful_requests, failed_requests),
            collection_duration_ms: collection_duration.as_millis() as u64,
            average_latency_ms: if successful_requests > 0 { 
                collection_duration.as_millis() as f64 / successful_requests as f64 
            } else { 0.0 },
            rate_limit_hits: 0, // TODO: Track actual rate limit hits
        })
    }
    
    /// Get list of available exchanges
    pub fn get_available_exchanges(&self) -> Vec<String> {
        self.collectors.keys().cloned().collect()
    }
    
    /// Validate all configured exchanges
    pub async fn validate_all_exchanges(&self) -> Result<HashMap<String, bool>> {
        let mut results = HashMap::new();
        
        for (exchange, collector) in &self.collectors {
            let is_valid = collector.validate_connection().await.unwrap_or(false);
            results.insert(exchange.clone(), is_valid);
        }
        
        Ok(results)
    }
    
    // Private helper methods
    
    fn parse_interval(&self, interval: &str) -> Result<Interval> {
        match interval {
            "1m" => Ok(Interval::OneMinute),
            "3m" => Ok(Interval::ThreeMinutes),
            "5m" => Ok(Interval::FiveMinutes),
            "15m" => Ok(Interval::FifteenMinutes),
            "30m" => Ok(Interval::ThirtyMinutes),
            "1h" => Ok(Interval::OneHour),
            "2h" => Ok(Interval::TwoHours),
            "4h" => Ok(Interval::FourHours),
            "6h" => Ok(Interval::SixHours),
            "8h" => Ok(Interval::EightHours),
            "12h" => Ok(Interval::TwelveHours),
            "1d" => Ok(Interval::OneDay),
            "3d" => Ok(Interval::ThreeDays),
            "1w" => Ok(Interval::OneWeek),
            "1M" => Ok(Interval::OneMonth),
            _ => Err(crate::DataCollectorError::Config(format!("Invalid interval: {}", interval))),
        }
    }
    
    fn calculate_optimal_chunk_size(&self, interval: &Interval) -> chrono::Duration {
        match interval {
            Interval::OneMinute => chrono::Duration::hours(12),
            Interval::ThreeMinutes => chrono::Duration::hours(24),
            Interval::FiveMinutes => chrono::Duration::days(2),
            Interval::FifteenMinutes => chrono::Duration::days(7),
            Interval::ThirtyMinutes => chrono::Duration::days(15),
            Interval::OneHour => chrono::Duration::days(30),
            Interval::TwoHours => chrono::Duration::days(60),
            Interval::FourHours => chrono::Duration::days(120),
            Interval::SixHours => chrono::Duration::days(180),
            Interval::EightHours => chrono::Duration::days(240),
            Interval::TwelveHours => chrono::Duration::days(365),
            Interval::OneDay => chrono::Duration::days(730),
            Interval::ThreeDays => chrono::Duration::days(1095),
            Interval::OneWeek => chrono::Duration::days(1825),
            Interval::OneMonth => chrono::Duration::days(1825),
        }
    }
    
    async fn store_klines(&self, klines: &[Kline]) -> Result<()> {
        if let Some(storage) = &self.storage {
            storage.store_klines(klines).await?;
        } else {
            tracing::warn!("No storage backend configured - data not persisted");
        }
        Ok(())
    }
    
    fn calculate_data_quality_score(&self, successful: u64, failed: u64) -> f64 {
        if successful + failed == 0 {
            return 0.0;
        }
        successful as f64 / (successful + failed) as f64
    }
}

impl std::fmt::Display for Interval {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Interval::OneMinute => "1m",
            Interval::ThreeMinutes => "3m",
            Interval::FiveMinutes => "5m",
            Interval::FifteenMinutes => "15m",
            Interval::ThirtyMinutes => "30m",
            Interval::OneHour => "1h",
            Interval::TwoHours => "2h",
            Interval::FourHours => "4h",
            Interval::SixHours => "6h",
            Interval::EightHours => "8h",
            Interval::TwelveHours => "12h",
            Interval::OneDay => "1d",
            Interval::ThreeDays => "3d",
            Interval::OneWeek => "1w",
            Interval::OneMonth => "1M",
        };
        write!(f, "{}", s)
    }
}