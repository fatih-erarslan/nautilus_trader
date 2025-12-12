//! Bybit exchange data collector

use super::*;
use crate::rate_limiter::RateLimiter;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Bybit API endpoints
const BYBIT_BASE_URL: &str = "https://api.bybit.com";

/// Bybit data collector
pub struct BybitCollector {
    client: Client,
    rate_limiter: RateLimiter,
    config: crate::config::BybitConfig,
}

impl BybitCollector {
    pub async fn new(config: &crate::config::BybitConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .build()?;
            
        let rate_limiter = RateLimiter::new(
            config.rate_limit_requests_per_minute,
            std::time::Duration::from_secs(60)
        );
        
        info!("Initializing Bybit collector with rate limit: {} req/min", 
              config.rate_limit_requests_per_minute);
        
        Ok(Self {
            client,
            rate_limiter,
            config: config.clone(),
        })
    }
    
    /// Make a rate-limited request to Bybit API
    async fn make_request(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Value> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}{}", BYBIT_BASE_URL, endpoint);
        debug!("Making request to: {}", url);
        
        let mut request = self.client.get(&url);
        for (key, value) in params {
            request = request.query(&[(key, value)]);
        }
        
        let response = request.send().await?;
        let json = response.json::<Value>().await?;
        
        // Check for Bybit API errors
        if let Some(ret_code) = json.get("retCode").and_then(|c| c.as_i64()) {
            if ret_code != 0 {
                let msg = json.get("retMsg").and_then(|m| m.as_str()).unwrap_or("Unknown error");
                return Err(DataCollectorError::ApiError(format!("Bybit API error {}: {}", ret_code, msg)));
            }
        }
        
        Ok(json)
    }
}

#[async_trait]
impl DataCollector for BybitCollector {
    fn exchange_name(&self) -> &str {
        "bybit"
    }
    
    async fn fetch_symbols(&self) -> Result<Vec<String>> {
        let response = self.make_request("/v5/market/instruments-info", &[("category", "spot")]).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let list = result.get("list")
            .and_then(|l| l.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid instruments response".to_string()))?;
            
        let symbols = list.iter()
            .filter_map(|inst| {
                inst.get("symbol")?.as_str().map(|s| s.to_string())
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
        // Bybit interval format
        let interval_str = match interval {
            "1m" => "1",
            "5m" => "5",
            "15m" => "15",
            "30m" => "30",
            "1h" => "60",
            "4h" => "240",
            "1d" => "D",
            _ => return Err(DataCollectorError::InvalidInterval(interval.to_string())),
        };
        
        let params = [
            ("category", "spot"),
            ("symbol", symbol),
            ("interval", interval_str),
            ("start", &start_time.to_string()),
            ("end", &end_time.to_string()),
            ("limit", "200"),
        ];
        
        let response = self.make_request("/v5/market/kline", &params).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let list = result.get("list")
            .and_then(|l| l.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid kline response".to_string()))?;
            
        let klines = list.iter()
            .filter_map(|candle| {
                let candle_array = candle.as_array()?;
                if candle_array.len() >= 7 {
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
                        quote_volume: candle_array[6].as_str()?.parse().ok()?,
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
            ("category", "spot"),
            ("symbol", symbol),
            ("limit", "1000"),
        ];
        
        let response = self.make_request("/v5/market/recent-trade", &params).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let list = result.get("list")
            .and_then(|l| l.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid trades response".to_string()))?;
            
        let trades = list.iter()
            .filter_map(|trade| {
                let timestamp = trade.get("time")?.as_str()?.parse::<i64>().ok()?;
                
                if timestamp >= start_time && timestamp <= end_time {
                    Some(crate::types::TradeData {
                        symbol: symbol.to_string(),
                        trade_id: 0,
                        price: trade.get("price")?.as_str()?.parse().ok()?,
                        quantity: trade.get("size")?.as_str()?.parse().ok()?,
                        timestamp,
                        is_buyer_maker: trade.get("side")?.as_str()? == "Sell",
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
            ("category", "spot"),
            ("symbol", symbol),
            ("limit", &depth.to_string()),
        ];
        
        let response = self.make_request("/v5/market/orderbook", &params).await?;
        
        let result = response.get("result")
            .ok_or_else(|| DataCollectorError::ParseError("No result field".to_string()))?;
            
        let bids = result.get("b")
            .and_then(|b| b.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid bids".to_string()))?
            .iter()
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
            
        let asks = result.get("a")
            .and_then(|a| a.as_array())
            .ok_or_else(|| DataCollectorError::ParseError("Invalid asks".to_string()))?
            .iter()
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
            
        let timestamp = result.get("ts")
            .and_then(|t| t.as_str())
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(|| chrono::Utc::now().timestamp_millis());
            
        Ok(crate::types::OrderBookSnapshot {
            symbol: symbol.to_string(),
            timestamp,
            bids,
            asks,
            last_update_id: result.get("u")
                .and_then(|u| u.as_u64())
                .unwrap_or(0),
        })
    }
}