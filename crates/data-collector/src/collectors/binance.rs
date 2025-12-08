//! Binance exchange data collector

use super::*;
use crate::rate_limiter::RateLimiter;
use reqwest::Client;
use serde_json::Value;
use std::collections::HashMap;
use tracing::{debug, info, warn};

/// Binance API endpoints
const BINANCE_BASE_URL: &str = "https://api.binance.com";
const BINANCE_FUTURES_BASE_URL: &str = "https://fapi.binance.com";

/// Binance data collector
pub struct BinanceCollector {
    client: Client,
    rate_limiter: RateLimiter,
    config: crate::config::BinanceConfig,
    futures_enabled: bool,
}

impl BinanceCollector {
    pub async fn new(config: &crate::config::BinanceConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .gzip(true)
            .build()?;
            
        let rate_limiter = RateLimiter::new(
            config.rate_limit_requests_per_minute,
            std::time::Duration::from_secs(60)
        );
        
        info!("Initializing Binance collector with rate limit: {} req/min", 
              config.rate_limit_requests_per_minute);
        
        Ok(Self {
            client,
            rate_limiter,
            config: config.clone(),
            futures_enabled: config.include_futures,
        })
    }
    
    /// Make a rate-limited request to Binance API
    async fn make_request(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Value> {
        self.rate_limiter.wait().await;
        
        let url = format!("{}{}", BINANCE_BASE_URL, endpoint);
        debug!("Making request to: {}", url);
        
        let response = self.client
            .get(&url)
            .query(params)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(crate::DataCollectorError::ExchangeApi(
                format!("Binance API error {}: {}", status, text)
            ));
        }
        
        let json: Value = response.json().await?;
        Ok(json)
    }
    
    /// Make a request to Binance Futures API
    async fn make_futures_request(&self, endpoint: &str, params: &[(&str, &str)]) -> Result<Value> {
        if !self.futures_enabled {
            return Err(crate::DataCollectorError::Config("Futures not enabled".to_string()));
        }
        
        self.rate_limiter.wait().await;
        
        let url = format!("{}{}", BINANCE_FUTURES_BASE_URL, endpoint);
        debug!("Making futures request to: {}", url);
        
        let response = self.client
            .get(&url)
            .query(params)
            .send()
            .await?;
            
        if !response.status().is_success() {
            let status = response.status();
            let text = response.text().await.unwrap_or_default();
            return Err(crate::DataCollectorError::ExchangeApi(
                format!("Binance Futures API error {}: {}", status, text)
            ));
        }
        
        let json: Value = response.json().await?;
        Ok(json)
    }
    
    /// Convert Binance interval to our Interval enum
    fn interval_to_binance_string(&self, interval: &Interval) -> &'static str {
        match interval {
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
        }
    }
    
    /// Parse Binance kline data
    fn parse_kline(&self, symbol: &str, interval: Interval, data: &Value) -> Result<Kline> {
        let array = data.as_array()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid kline format".to_string()))?;
            
        if array.len() < 12 {
            return Err(crate::DataCollectorError::Validation("Incomplete kline data".to_string()));
        }
        
        let open_time = array[0].as_i64()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid open time".to_string()))?;
        let close_time = array[6].as_i64()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid close time".to_string()))?;
            
        Ok(Kline {
            symbol: symbol.to_string(),
            open_time: chrono::DateTime::from_timestamp_millis(open_time)
                .ok_or_else(|| crate::DataCollectorError::Validation("Invalid timestamp".to_string()))?
                .with_timezone(&Utc),
            close_time: chrono::DateTime::from_timestamp_millis(close_time)
                .ok_or_else(|| crate::DataCollectorError::Validation("Invalid timestamp".to_string()))?
                .with_timezone(&Utc),
            open: array[1].as_str().unwrap_or("0").parse()?,
            high: array[2].as_str().unwrap_or("0").parse()?,
            low: array[3].as_str().unwrap_or("0").parse()?,
            close: array[4].as_str().unwrap_or("0").parse()?,
            volume: array[5].as_str().unwrap_or("0").parse()?,
            quote_volume: array[7].as_str().unwrap_or("0").parse()?,
            trades_count: array[8].as_u64().unwrap_or(0),
            taker_buy_base_volume: array[9].as_str().unwrap_or("0").parse()?,
            taker_buy_quote_volume: array[10].as_str().unwrap_or("0").parse()?,
            interval,
            exchange: "binance".to_string(),
        })
    }
    
    /// Parse Binance trade data
    fn parse_trade(&self, symbol: &str, data: &Value) -> Result<Trade> {
        Ok(Trade {
            symbol: symbol.to_string(),
            trade_id: data["id"].as_u64().unwrap_or(0),
            price: data["price"].as_str().unwrap_or("0").parse()?,
            quantity: data["qty"].as_str().unwrap_or("0").parse()?,
            quote_quantity: data["quoteQty"].as_str().unwrap_or("0").parse()?,
            timestamp: {
                let time_ms = data["time"].as_i64().unwrap_or(0);
                chrono::DateTime::from_timestamp_millis(time_ms)
                    .ok_or_else(|| crate::DataCollectorError::Validation("Invalid timestamp".to_string()))?
                    .with_timezone(&Utc)
            },
            is_buyer_maker: data["isBuyerMaker"].as_bool().unwrap_or(false),
            exchange: "binance".to_string(),
        })
    }
}

#[async_trait]
impl ExchangeCollector for BinanceCollector {
    fn exchange_name(&self) -> &str {
        "binance"
    }
    
    async fn get_symbols(&self) -> Result<Vec<String>> {
        let response = self.make_request("/api/v3/exchangeInfo", &[]).await?;
        
        let symbols = response["symbols"]
            .as_array()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid symbols response".to_string()))?
            .iter()
            .filter_map(|s| {
                if s["status"].as_str() == Some("TRADING") {
                    s["symbol"].as_str().map(|sym| sym.to_string())
                } else {
                    None
                }
            })
            .collect();
            
        info!("Retrieved {} trading symbols from Binance", symbols.len());
        Ok(symbols)
    }
    
    async fn get_server_time(&self) -> Result<DateTime<Utc>> {
        let response = self.make_request("/api/v3/time", &[]).await?;
        
        let server_time = response["serverTime"].as_i64()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid server time".to_string()))?;
            
        Ok(chrono::DateTime::from_timestamp_millis(server_time)
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid timestamp".to_string()))?
            .with_timezone(&Utc))
    }
    
    async fn collect_klines(
        &self,
        symbol: &str,
        interval: Interval,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        limit: Option<u32>,
    ) -> Result<Vec<Kline>> {
        let interval_str = self.interval_to_binance_string(&interval);
        let start_time_ms = start_time.timestamp_millis().to_string();
        let end_time_ms = end_time.timestamp_millis().to_string();
        let limit_str = limit.unwrap_or(1000).to_string();
        
        let params = [
            ("symbol", symbol),
            ("interval", interval_str),
            ("startTime", &start_time_ms),
            ("endTime", &end_time_ms),
            ("limit", &limit_str),
        ];
        
        debug!("Collecting klines for {} from {} to {}", symbol, start_time, end_time);
        
        let response = self.make_request("/api/v3/klines", &params).await?;
        
        let klines_data = response.as_array()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid klines response".to_string()))?;
            
        let mut klines = Vec::with_capacity(klines_data.len());
        
        for kline_data in klines_data {
            match self.parse_kline(symbol, interval.clone(), kline_data) {
                Ok(kline) => klines.push(kline),
                Err(e) => {
                    warn!("Failed to parse kline data: {}", e);
                    continue;
                }
            }
        }
        
        info!("Collected {} klines for {} ({})", klines.len(), symbol, interval);
        Ok(klines)
    }
    
    async fn collect_trades(
        &self,
        symbol: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
        limit: Option<u32>,
    ) -> Result<Vec<Trade>> {
        let start_time_ms = start_time.timestamp_millis().to_string();
        let end_time_ms = end_time.timestamp_millis().to_string();
        let limit_str = limit.unwrap_or(1000).to_string();
        
        let params = [
            ("symbol", symbol),
            ("startTime", &start_time_ms),
            ("endTime", &end_time_ms),
            ("limit", &limit_str),
        ];
        
        let response = self.make_request("/api/v3/historicalTrades", &params).await?;
        
        let trades_data = response.as_array()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid trades response".to_string()))?;
            
        let mut trades = Vec::with_capacity(trades_data.len());
        
        for trade_data in trades_data {
            match self.parse_trade(symbol, trade_data) {
                Ok(trade) => trades.push(trade),
                Err(e) => {
                    warn!("Failed to parse trade data: {}", e);
                    continue;
                }
            }
        }
        
        info!("Collected {} trades for {}", trades.len(), symbol);
        Ok(trades)
    }
    
    async fn collect_order_book(&self, symbol: &str, limit: Option<u32>) -> Result<OrderBook> {
        let limit_str = limit.unwrap_or(100).to_string();
        let params = [("symbol", symbol), ("limit", &limit_str)];
        
        let response = self.make_request("/api/v3/depth", &params).await?;
        
        let bids = response["bids"]
            .as_array()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid bids".to_string()))?
            .iter()
            .filter_map(|level| {
                let array = level.as_array()?;
                Some(OrderBookLevel {
                    price: array[0].as_str()?.parse().ok()?,
                    quantity: array[1].as_str()?.parse().ok()?,
                })
            })
            .collect();
            
        let asks = response["asks"]
            .as_array()
            .ok_or_else(|| crate::DataCollectorError::Validation("Invalid asks".to_string()))?
            .iter()
            .filter_map(|level| {
                let array = level.as_array()?;
                Some(OrderBookLevel {
                    price: array[0].as_str()?.parse().ok()?,
                    quantity: array[1].as_str()?.parse().ok()?,
                })
            })
            .collect();
            
        Ok(OrderBook {
            symbol: symbol.to_string(),
            timestamp: Utc::now(),
            bids,
            asks,
            exchange: "binance".to_string(),
        })
    }
    
    async fn collect_ticker_24hr(&self, symbol: Option<&str>) -> Result<Vec<Ticker24hr>> {
        let params = if let Some(s) = symbol {
            vec![("symbol", s)]
        } else {
            vec![]
        };
        
        let response = self.make_request("/api/v3/ticker/24hr", &params).await?;
        
        let tickers_data = if symbol.is_some() {
            vec![response]
        } else {
            response.as_array()
                .ok_or_else(|| crate::DataCollectorError::Validation("Invalid ticker response".to_string()))?
                .clone()
        };
        
        let mut tickers = Vec::new();
        
        for ticker_data in tickers_data {
            let open_time = ticker_data["openTime"].as_i64().unwrap_or(0);
            let close_time = ticker_data["closeTime"].as_i64().unwrap_or(0);
            
            let ticker = Ticker24hr {
                symbol: ticker_data["symbol"].as_str().unwrap_or("").to_string(),
                price_change: ticker_data["priceChange"].as_str().unwrap_or("0").parse()?,
                price_change_percent: ticker_data["priceChangePercent"].as_str().unwrap_or("0").parse()?,
                weighted_avg_price: ticker_data["weightedAvgPrice"].as_str().unwrap_or("0").parse()?,
                prev_close_price: ticker_data["prevClosePrice"].as_str().unwrap_or("0").parse()?,
                last_price: ticker_data["lastPrice"].as_str().unwrap_or("0").parse()?,
                last_qty: ticker_data["lastQty"].as_str().unwrap_or("0").parse()?,
                bid_price: ticker_data["bidPrice"].as_str().unwrap_or("0").parse()?,
                bid_qty: ticker_data["bidQty"].as_str().unwrap_or("0").parse()?,
                ask_price: ticker_data["askPrice"].as_str().unwrap_or("0").parse()?,
                ask_qty: ticker_data["askQty"].as_str().unwrap_or("0").parse()?,
                open_price: ticker_data["openPrice"].as_str().unwrap_or("0").parse()?,
                high_price: ticker_data["highPrice"].as_str().unwrap_or("0").parse()?,
                low_price: ticker_data["lowPrice"].as_str().unwrap_or("0").parse()?,
                volume: ticker_data["volume"].as_str().unwrap_or("0").parse()?,
                quote_volume: ticker_data["quoteVolume"].as_str().unwrap_or("0").parse()?,
                open_time: chrono::DateTime::from_timestamp_millis(open_time)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                close_time: chrono::DateTime::from_timestamp_millis(close_time)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                first_id: ticker_data["firstId"].as_u64().unwrap_or(0),
                last_id: ticker_data["lastId"].as_u64().unwrap_or(0),
                count: ticker_data["count"].as_u64().unwrap_or(0),
                exchange: "binance".to_string(),
            };
            
            tickers.push(ticker);
        }
        
        Ok(tickers)
    }
    
    async fn collect_funding_rates(&self, symbol: Option<&str>) -> Result<Vec<FundingRate>> {
        if !self.futures_enabled {
            return Ok(vec![]);
        }
        
        let params = if let Some(s) = symbol {
            vec![("symbol", s)]
        } else {
            vec![]
        };
        
        let response = self.make_futures_request("/fapi/v1/fundingRate", &params).await?;
        
        let funding_data = if symbol.is_some() {
            vec![response]
        } else {
            response.as_array()
                .ok_or_else(|| crate::DataCollectorError::Validation("Invalid funding rate response".to_string()))?
                .clone()
        };
        
        let mut funding_rates = Vec::new();
        
        for data in funding_data {
            let funding_time = data["fundingTime"].as_i64().unwrap_or(0);
            
            let funding_rate = FundingRate {
                symbol: data["symbol"].as_str().unwrap_or("").to_string(),
                funding_rate: data["fundingRate"].as_str().unwrap_or("0").parse()?,
                funding_time: chrono::DateTime::from_timestamp_millis(funding_time)
                    .unwrap_or_default()
                    .with_timezone(&Utc),
                mark_price: data["markPrice"].as_str().unwrap_or("0").parse()?,
                exchange: "binance".to_string(),
            };
            
            funding_rates.push(funding_rate);
        }
        
        Ok(funding_rates)
    }
    
    async fn validate_connection(&self) -> Result<bool> {
        match self.make_request("/api/v3/ping", &[]).await {
            Ok(_) => {
                info!("Binance API connection validated successfully");
                Ok(true)
            },
            Err(e) => {
                warn!("Binance API connection validation failed: {}", e);
                Ok(false)
            }
        }
    }
    
    fn get_rate_limits(&self) -> HashMap<String, u32> {
        let mut limits = HashMap::new();
        limits.insert("requests_per_minute".to_string(), self.config.rate_limit_requests_per_minute);
        limits.insert("weight_per_minute".to_string(), 1200);
        limits.insert("orders_per_second".to_string(), 10);
        limits.insert("orders_per_day".to_string(), 200000);
        limits
    }
}

impl std::str::FromStr for crate::DataCollectorError {
    type Err = std::num::ParseFloatError;
    
    fn from_str(_s: &str) -> std::result::Result<Self, Self::Err> {
        // This is a placeholder implementation
        Err("0".parse::<f64>().unwrap_err())
    }
}