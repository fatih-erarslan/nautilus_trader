//! Coinbase exchange data collector

use super::*;
use crate::rate_limiter::RateLimiter;
use crate::{Result, DataCollectorError};
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};
use async_trait::async_trait;
use chrono::{DateTime, Utc};

/// Coinbase API endpoints
const COINBASE_BASE_URL: &str = "https://api.exchange.coinbase.com";

/// Coinbase data collector
pub struct CoinbaseCollector {
    client: Client,
    rate_limiter: RateLimiter,
    config: crate::config::CoinbaseConfig,
}

impl CoinbaseCollector {
    pub async fn new(config: &crate::config::CoinbaseConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .build()?;
            
        let rate_limiter = RateLimiter::new(
            config.rate_limit_requests_per_minute,
            std::time::Duration::from_secs(60)
        );
        
        info!("Initializing Coinbase collector with rate limit: {} req/min", 
              config.rate_limit_requests_per_minute);
        
        Ok(Self {
            client,
            rate_limiter,
            config: config.clone(),
        })
    }
    
    /// Make a rate-limited request to Coinbase API
    async fn make_request(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Value> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}{}", COINBASE_BASE_URL, endpoint);
        debug!("Making request to: {}", url);
        
        let mut request = self.client.get(&url);
        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }
        
        let response = request.send().await?;
        let json = response.json().await?;
        Ok(json)
    }
}

#[async_trait]
impl ExchangeCollector for CoinbaseCollector {
    fn exchange_name(&self) -> &str {
        "coinbase"
    }
    
    async fn get_symbols(&self) -> Result<Vec<String>> {
        let response = self.make_request("/products", &[]).await?;
        
        let symbols = response.as_array()
            .ok_or_else(|| DataCollectorError::ParseError("Invalid products response".to_string()))?
            .iter()
            .filter_map(|product| {
                product.get("id")?.as_str().map(|s| s.to_string())
            })
            .collect();
            
        Ok(symbols)
    }
    
    async fn get_server_time(&self) -> Result<DateTime<Utc>> {
        let response = self.make_request("/time", &[]).await?;
        let epoch = response.get("epoch")
            .and_then(|e| e.as_f64())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid time response".to_string()))?;
        
        let secs = epoch as i64;
        let nanos = ((epoch.fract() * 1_000_000_000.0) as u32);
        Ok(DateTime::<Utc>::from_timestamp(secs, nanos).unwrap_or_else(Utc::now))
    }
    
    async fn collect_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>> {
        // Coinbase uses different interval format
        let granularity = match interval {
            Interval::OneMinute => "60",
            Interval::FiveMinutes => "300",
            Interval::FifteenMinutes => "900",
            Interval::OneHour => "3600",
            Interval::OneDay => "86400",
            _ => return Err(DataCollectorError::Validation(format!("Unsupported interval: {:?}", interval))),
        };
        
        let params = [
            ("granularity", granularity),
            ("start", &start_time.timestamp().to_string()),
            ("end", &end_time.timestamp().to_string()),
        ];
        
        let endpoint = format!("/products/{}/candles", symbol);
        let response = self.make_request(&endpoint, &params).await?;
        
        let klines = response.as_array()
            .ok_or_else(|| DataCollectorError::ParseError("Invalid candles response".to_string()))?
            .iter()
            .filter_map(|candle| {
                let candle_array = candle.as_array()?;
                if candle_array.len() >= 6 {
                    let timestamp = candle_array[0].as_i64()?;
                    let open_time = DateTime::<Utc>::from_timestamp(timestamp, 0).unwrap_or_else(Utc::now);
                    let close_time = DateTime::<Utc>::from_timestamp(timestamp + granularity.parse::<i64>().ok()? - 1, 0).unwrap_or_else(Utc::now);
                    Some(Kline {
                        symbol: symbol.to_string(),
                        open_time,
                        close_time,
                        open: candle_array[3].as_str()?.parse().ok()?,
                        high: candle_array[2].as_str()?.parse().ok()?,
                        low: candle_array[1].as_str()?.parse().ok()?,
                        close: candle_array[4].as_str()?.parse().ok()?,
                        volume: candle_array[5].as_str()?.parse().ok()?,
                        quote_volume: 0.0,
                        trades_count: 0,
                        taker_buy_base_volume: 0.0,
                        taker_buy_quote_volume: 0.0,
                        interval,
                        exchange: "coinbase".to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();
            
        Ok(klines)
    }
    
    async fn collect_trades(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        limit: Option<u32>,
    ) -> Result<Vec<Trade>> {
        // Coinbase trade fetching would be implemented here
        Ok(vec![])
    }
    
    async fn collect_order_book(
        &self,
        symbol: &str,
        depth: Option<u32>,
    ) -> Result<OrderBook> {
        let params = [("level", "2")];
        let endpoint = format!("/products/{}/book", symbol);
        let response = self.make_request(&endpoint, &params).await?;
        
        let bids = response.get("bids")
            .and_then(|b| b.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid bids".to_string()))?
            .iter()
            .take(depth.unwrap_or(20) as usize)
            .filter_map(|bid| {
                let bid_array = bid.as_array()?;
                if bid_array.len() >= 2 {
                    Some(OrderBookLevel {
                        price: bid_array[0].as_str()?.parse().ok()?,
                        quantity: bid_array[1].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        let asks = response.get("asks")
            .and_then(|a| a.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid asks".to_string()))?
            .iter()
            .take(depth.unwrap_or(20) as usize)
            .filter_map(|ask| {
                let ask_array = ask.as_array()?;
                if ask_array.len() >= 2 {
                    Some(OrderBookLevel {
                        price: ask_array[0].as_str()?.parse().ok()?,
                        quantity: ask_array[1].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            bids,
            asks,
            exchange: "coinbase".to_string(),
        })
    }
    
    async fn collect_ticker_24hr(&self, symbol: Option<&str>) -> Result<Vec<Ticker24hr>> {
        // Coinbase ticker implementation would go here
        Ok(vec![])
    }
    
    async fn validate_connection(&self) -> Result<bool> {
        // Try to get server time as a connectivity check
        match self.get_server_time().await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }
    
    fn get_rate_limits(&self) -> HashMap<String, u32> {
        let mut limits = HashMap::new();
        limits.insert("requests_per_minute".to_string(), self.config.rate_limit_requests_per_minute);
        limits
    }
}