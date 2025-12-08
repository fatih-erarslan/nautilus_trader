//! OKX exchange data collector

use super::*;
use crate::rate_limiter::RateLimiter;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// OKX API endpoints
const OKX_BASE_URL: &str = "https://www.okx.com";

/// OKX data collector
pub struct OkxCollector {
    client: Client,
    rate_limiter: RateLimiter,
    config: crate::config::OkxConfig,
}

impl OkxCollector {
    pub async fn new(config: &crate::config::OkxConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .build()?;
            
        let rate_limiter = RateLimiter::new(
            config.rate_limit_requests_per_minute,
            std::time::Duration::from_secs(60)
        );
        
        info!("Initializing OKX collector with rate limit: {} req/min", 
              config.rate_limit_requests_per_minute);
        
        Ok(Self {
            client,
            rate_limiter,
            config: config.clone(),
        })
    }
    
    /// Make a rate-limited request to OKX API
    async fn make_request(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Value> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}{}", OKX_BASE_URL, endpoint);
        debug!("Making request to: {}", url);
        
        let mut request = self.client.get(&url);
        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }
        
        let response = request.send().await?;
        let json = response.json::<Value>().await?;
        
        // Check for OKX API errors
        if let Some(code) = json.get("code").and_then(|c| c.as_str()) {
            if code != "0" {
                let msg = json.get("msg").and_then(|m| m.as_str()).unwrap_or("Unknown error");
                return Err(DataCollectorError::ApiError(format!("OKX API error {}: {}", code, msg)));
            }
        }
        
        Ok(json)
    }
}

#[async_trait]
impl DataCollector for OkxCollector {
    fn exchange_name(&self) -> &str {
        "okx"
    }
    
    async fn fetch_symbols(&self) -> Result<Vec<String>> {
        let response = self.make_request("/api/v5/public/instruments", &[("instType", "SPOT")]).await?;
        
        let data = response.get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid instruments response".to_string()))?;
            
        let symbols = data.iter()
            .filter_map(|inst| {
                inst.get("instId")?.as_str().map(|s| s.to_string())
            })
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
        // OKX interval format
        let bar = match interval {
            "1m" => "1m",
            "5m" => "5m",
            "15m" => "15m",
            "30m" => "30m",
            "1h" => "1H",
            "4h" => "4H",
            "1d" => "1D",
            _ => return Err(DataCollectorError::InvalidInterval(interval.to_string())),
        };
        
        let params = [
            ("instId", symbol),
            ("bar", bar),
            ("before", &start_time.to_string()),
            ("after", &end_time.to_string()),
            ("limit", "300"),
        ];
        
        let response = self.make_request("/api/v5/market/candles", &params).await?;
        
        let data = response.get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid candles response".to_string()))?;
            
        let klines = data.iter()
            .filter_map(|candle| {
                let candle_array = candle.as_array()?;
                if candle_array.len() >= 9 {
                    Some(crate::types::KlineData {
                        open_time: candle_array[0].as_str()?.parse().ok()?,
                        open: candle_array[1].as_str()?.parse().ok()?,
                        high: candle_array[2].as_str()?.parse().ok()?,
                        low: candle_array[3].as_str()?.parse().ok()?,
                        close: candle_array[4].as_str()?.parse().ok()?,
                        volume: candle_array[5].as_str()?.parse().ok()?,
                        close_time: candle_array[0].as_str()?.parse::<i64>().ok()? + 
                                   match interval {
                                       "1m" => 60_000,
                                       "5m" => 300_000,
                                       "15m" => 900_000,
                                       "30m" => 1_800_000,
                                       "1h" => 3_600_000,
                                       "4h" => 14_400_000,
                                       "1d" => 86_400_000,
                                       _ => 60_000,
                                   } - 1,
                        quote_volume: candle_array[7].as_str()?.parse().ok()?,
                        trades_count: 0,
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
            ("instId", symbol),
            ("limit", "500"),
        ];
        
        let response = self.make_request("/api/v5/market/trades", &params).await?;
        
        let data = response.get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid trades response".to_string()))?;
            
        let trades = data.iter()
            .filter_map(|trade| {
                let trade_id = trade.get("tradeId")?.as_str()?.parse().ok()?;
                let timestamp = trade.get("ts")?.as_str()?.parse::<i64>().ok()?;
                
                if timestamp >= start_time && timestamp <= end_time {
                    Some(crate::types::TradeData {
                        symbol: symbol.to_string(),
                        trade_id,
                        price: trade.get("px")?.as_str()?.parse().ok()?,
                        quantity: trade.get("sz")?.as_str()?.parse().ok()?,
                        timestamp,
                        is_buyer_maker: trade.get("side")?.as_str()? == "sell",
                        is_best_match: true,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        Ok(trades)
    }
    
    async fn fetch_order_book(
        &self,
        symbol: &str,
        depth: usize,
    ) -> Result<crate::types::OrderBookSnapshot> {
        let params = [
            ("instId", symbol),
            ("sz", &depth.to_string()),
        ];
        
        let response = self.make_request("/api/v5/market/books", &params).await?;
        
        let data = response.get("data")
            .and_then(|d| d.as_array())
            .and_then(|a| a.first())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid order book response".to_string()))?;
            
        let bids = data.get("bids")
            .and_then(|b| b.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid bids".to_string()))?
            .iter()
            .filter_map(|bid| {
                let bid_array = bid.as_array()?;
                if bid_array.len() >= 4 {
                    Some(crate::types::OrderBookLevel {
                        price: bid_array[0].as_str()?.parse().ok()?,
                        quantity: bid_array[1].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        let asks = data.get("asks")
            .and_then(|a| a.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid asks".to_string()))?
            .iter()
            .filter_map(|ask| {
                let ask_array = ask.as_array()?;
                if ask_array.len() >= 4 {
                    Some(crate::types::OrderBookLevel {
                        price: ask_array[0].as_str()?.parse().ok()?,
                        quantity: ask_array[1].as_str()?.parse().ok()?,
                    })
                } else {
                    None
                }
            })
            .collect();
            
        let timestamp = data.get("ts")
            .and_then(|t| t.as_str())
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
            
        Ok(crate::types::OrderBookSnapshot {
            symbol: symbol.to_string(),
            timestamp,
            bids,
            asks,
            last_update_id: 0,
        })
    }
}