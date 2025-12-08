//! Kraken exchange data collector

use super::*;
use crate::rate_limiter::RateLimiter;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Kraken API endpoints
const KRAKEN_BASE_URL: &str = "https://api.kraken.com";

/// Kraken data collector
pub struct KrakenCollector {
    client: Client,
    rate_limiter: RateLimiter,
    config: crate::config::KrakenConfig,
}

impl KrakenCollector {
    pub async fn new(config: &crate::config::KrakenConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .build()?;
            
        let rate_limiter = RateLimiter::new(
            config.rate_limit_requests_per_minute,
            std::time::Duration::from_secs(60)
        );
        
        info!("Initializing Kraken collector with rate limit: {} req/min", 
              config.rate_limit_requests_per_minute);
        
        Ok(Self {
            client,
            rate_limiter,
            config: config.clone(),
        })
    }
    
    /// Make a rate-limited request to Kraken API
    async fn make_request(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Value> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}{}", KRAKEN_BASE_URL, endpoint);
        debug!("Making request to: {}", url);
        
        let mut request = self.client.get(&url);
        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }
        
        let response = request.send().await?;
        let json = response.json::<Value>().await?;
        
        // Check for Kraken API errors
        if let Some(error_array) = json.get("error").and_then(|e| e.as_array()) {
            if !error_array.is_empty() {
                let error_msg = error_array.iter()
                    .filter_map(|e| e.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(DataCollectorError::ApiError(error_msg));
            }
        }
        
        Ok(json)
    }
}

#[async_trait]
impl DataCollector for KrakenCollector {
    fn exchange_name(&self) -> &str {
        "kraken"
    }
    
    async fn fetch_symbols(&self) -> Result<Vec<String>> {
        let response = self.make_request("/0/public/AssetPairs", &[]).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let symbols = result.as_object()
            .ok_or_else(|| DataCollectorError::ParseError("Invalid asset pairs response".to_string()))?
            .keys()
            .map(|s| s.to_string())
            .collect();
            
        Ok(symbols)
    }
    
    async fn fetch_klines(
        &self,
        symbol: &str,
        interval: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<crate::types::KlineData>> {
        // Kraken uses minutes for interval
        let interval_minutes = match interval {
            "1m" => "1",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" => "60",
            "4h" => "240",
            "1d" => "1440",
            _ => return Err(DataCollectorError::InvalidInterval(interval.to_string())),
        };
        
        let params = [
            ("pair", symbol),
            ("interval", interval_minutes),
            ("since", &(start_time / 1000).to_string()),
        ];
        
        let response = self.make_request("/0/public/OHLC", &params).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let klines = result.get(symbol)
            .and_then(|data| data.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid OHLC response".to_string()))?
            .iter()
            .filter_map(|candle| {
                let candle_array = candle.as_array()?;
                if candle_array.len() >= 8 {
                    Some(crate::types::KlineData {
                        open_time: candle_array[0].as_i64()? * 1000,
                        open: candle_array[1].as_str()?.parse().ok()?,
                        high: candle_array[2].as_str()?.parse().ok()?,
                        low: candle_array[3].as_str()?.parse().ok()?,
                        close: candle_array[4].as_str()?.parse().ok()?,
                        volume: candle_array[6].as_str()?.parse().ok()?,
                        close_time: candle_array[0].as_i64()? * 1000 + (interval_minutes.parse::<i64>().ok()? * 60 * 1000 - 1),
                        quote_volume: 0.0,
                        trades_count: candle_array[7].as_i64()? as u64,
                        taker_buy_volume: 0.0,
                        taker_buy_quote_volume: 0.0,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        Ok(klines)
    }
    
    async fn fetch_trades(
        &self,
        symbol: &str,
        start_time: i64,
        end_time: i64,
    ) -> Result<Vec<crate::types::TradeData>> {
        let params = [
            ("pair", symbol),
            ("since", &(start_time / 1000).to_string()),
        ];
        
        let response = self.make_request("/0/public/Trades", &params).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let trades = result.get(symbol)
            .and_then(|data| data.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid trades response".to_string()))?
            .iter()
            .filter_map(|trade| {
                let trade_array = trade.as_array()?;
                if trade_array.len() >= 6 {
                    Some(crate::types::TradeData {
                        symbol: symbol.to_string(),
                        trade_id: 0,
                        price: trade_array[0].as_str()?.parse().ok()?,
                        quantity: trade_array[1].as_str()?.parse().ok()?,
                        timestamp: (trade_array[2].as_f64()? * 1000.0) as i64,
                        is_buyer_maker: trade_array[3].as_str()? == "b",
                        is_best_match: true,
                    })
                } else {
                    None
                }
            })
            .filter(|trade| trade.timestamp >= start_time && trade.timestamp <= end_time)
            .collect();
            
        Ok(trades)
    }
    
    async fn fetch_order_book(
        &self,
        symbol: &str,
        depth: usize,
    ) -> Result<crate::types::OrderBookSnapshot> {
        let params = [
            ("pair", symbol),
            ("count", &depth.to_string()),
        ];
        
        let response = self.make_request("/0/public/Depth", &params).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let book_data = result.get(symbol)
            .ok_or_else(|| DataCollectorError::ParseError("Invalid depth response".to_string()))?;
            
        let bids = book_data.get("bids")
            .and_then(|b| b.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid bids".to_string()))?
            .iter()
            .take(depth)
            .filter_map(|bid| {
                let bid_array = bid.as_array()?;
                if bid_array.len() >= 2 {
                    Some(crate::types::OrderBookLevel {
                        price: bid_array[0].as_str()?.parse().ok()?,
                        quantity: bid_array[1].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        let asks = book_data.get("asks")
            .and_then(|a| a.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid asks".to_string()))?
            .iter()
            .take(depth)
            .filter_map(|ask| {
                let ask_array = ask.as_array()?;
                if ask_array.len() >= 2 {
                    Some(crate::types::OrderBookLevel {
                        price: ask_array[0].as_str()?.parse().ok()?,
                        quantity: ask_array[1].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        Ok(crate::types::OrderBookSnapshot {
            symbol: symbol.to_string(),
            timestamp: chrono::Utc::now().timestamp_millis(),
            bids,
            asks,
            last_update_id: 0,
        })
    }
}