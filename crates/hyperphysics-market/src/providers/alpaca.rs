//! Alpaca Markets API provider implementation
//!
//! Provides access to stock market data via Alpaca Markets API.
//! Supports both paper and live trading data.
//!
//! # API Documentation
//!
//! - Market Data API: <https://alpaca.markets/docs/api-references/market-data-api/>
//! - Rate Limits: 200 requests/minute (free tier)
//! - Max bars per request: 10,000
//!
//! # Example
//!
//! ```no_run
//! use hyperphysics_market::providers::{AlpacaProvider, MarketDataProvider};
//! use hyperphysics_market::data::Timeframe;
//! use chrono::{Utc, Duration};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = AlpacaProvider::new(
//!         std::env::var("ALPACA_API_KEY")?,
//!         std::env::var("ALPACA_API_SECRET")?,
//!         true, // paper trading
//!     );
//!
//!     let end = Utc::now();
//!     let start = end - Duration::days(7);
//!
//!     let bars = provider.fetch_bars(
//!         "AAPL",
//!         Timeframe::Day1,
//!         start,
//!         end
//!     ).await?;
//!
//!     println!("Fetched {} bars", bars.len());
//!     Ok(())
//! }
//! ```

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use reqwest::{Client, header};
use serde::Deserialize;
use std::time::Duration;
use tracing::{debug, warn};
use reqwest::{Client, header, StatusCode};
use serde::Deserialize;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;
use tracing::{debug, warn, error};

use crate::data::{Bar, Timeframe};
use crate::error::{MarketError, MarketResult};
use super::{MarketDataProvider, ProviderConfig};

/// Rate limiter using token bucket algorithm
struct RateLimiter {
    tokens: f64,
    capacity: f64,
    refill_rate: f64, // tokens per second
    last_refill: Instant,
}

impl RateLimiter {
    fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    fn try_acquire(&mut self) -> bool {
        self.refill();
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
    }

    async fn wait_for_token(&mut self) {
        while !self.try_acquire() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// Alpaca Markets data provider
pub struct AlpacaProvider {
    config: ProviderConfig,
    client: Client,
    rate_limiter: Arc<Mutex<RateLimiter>>,
}

impl AlpacaProvider {
    /// Create a new Alpaca provider instance
    ///
    /// # Arguments
    ///
    /// * `api_key` - Alpaca API key
    /// * `api_secret` - Alpaca API secret
    /// * `paper` - Whether to use paper trading endpoints
    pub fn new(api_key: String, api_secret: String, paper: bool) -> Self {
        let data_url = "https://data.alpaca.markets/v2".to_string();

        let mut headers = header::HeaderMap::new();
        headers.insert("APCA-API-KEY-ID", header::HeaderValue::from_str(&api_key).unwrap());
        headers.insert("APCA-API-SECRET-KEY", header::HeaderValue::from_str(&api_secret).unwrap());

        let client = Client::builder()
            .default_headers(headers)
            .timeout(Duration::from_secs(30))
            .build()
            .expect("Failed to build HTTP client");

        let config = ProviderConfig {
            api_key,
            api_secret,
            base_url: data_url,
            paper_trading: paper,
            timeout_secs: 30,
            max_retries: 3,
        };

        // Free tier: 200 requests/minute = 3.33 requests/second
        let rate_limiter = Arc::new(Mutex::new(RateLimiter::new(200.0, 3.33)));

        Self {
            config,
            client,
            rate_limiter,
        }
    }

    /// Make an HTTP request with retry logic and exponential backoff
    async fn request_with_retry<T: serde::de::DeserializeOwned>(
        &self,
        url: String,
    ) -> MarketResult<T> {
        let mut retries = 0;
        let max_retries = self.config.max_retries;

        loop {
            // Wait for rate limiter token
            {
                let mut limiter = self.rate_limiter.lock().await;
                limiter.wait_for_token().await;
            }

            match self.client.get(&url).send().await {
                Ok(response) => {
                    let status = response.status();

                    match status {
                        StatusCode::OK => {
                            let text = response.text().await
                                .map_err(|e| MarketError::NetworkError(e.to_string()))?;
                            return serde_json::from_str(&text)
                                .map_err(|e| MarketError::ParseError(e.to_string()));
                        }
                        StatusCode::TOO_MANY_REQUESTS => {
                            if retries >= max_retries {
                                return Err(MarketError::RateLimitError(
                                    "Rate limit exceeded after retries".to_string()
                                ));
                            }
                            let backoff = Duration::from_secs(2_u64.pow(retries));
                            warn!("Rate limited, backing off for {:?}", backoff);
                            tokio::time::sleep(backoff).await;
                            retries += 1;
                        }
                        StatusCode::NOT_FOUND => {
                            let body = response.text().await.unwrap_or_default();
                            return Err(MarketError::InvalidSymbol(format!(
                                "Symbol not found: {}",
                                body
                            )));
                        }
                        StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                            return Err(MarketError::AuthenticationError(
                                "Invalid API credentials".to_string()
                            ));
                        }
                        _ => {
                            let body = response.text().await.unwrap_or_default();
                            return Err(MarketError::ApiError(format!(
                                "API request failed with status {}: {}",
                                status, body
                            )));
                        }
                    }
                }
                Err(e) => {
                    if retries >= max_retries {
                        return Err(MarketError::NetworkError(e.to_string()));
                    }
                    let backoff = Duration::from_secs(2_u64.pow(retries));
                    warn!("Network error, retrying in {:?}: {}", backoff, e);
                    tokio::time::sleep(backoff).await;
                    retries += 1;
                }
            }
        }
    }

    /// Validate bar data for anomalies
    fn validate_bar(bar: &AlpacaBar, symbol: &str) -> Result<(), String> {
        // Check for zero or negative prices
        if bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0 {
            return Err(format!(
                "Invalid price data for {}: o={}, h={}, l={}, c={}",
                symbol, bar.open, bar.high, bar.low, bar.close
            ));
        }

        // Check OHLC consistency
        if bar.high < bar.low {
            return Err(format!(
                "High {} is less than low {} for {}",
                bar.high, bar.low, symbol
            ));
        }

        if bar.high < bar.open || bar.high < bar.close {
            return Err(format!(
                "High {} is less than open {} or close {} for {}",
                bar.high, bar.open, bar.close, symbol
            ));
        }

        if bar.low > bar.open || bar.low > bar.close {
            return Err(format!(
                "Low {} is greater than open {} or close {} for {}",
                bar.low, bar.open, bar.close, symbol
            ));
        }

        // Detect extreme price movements (>50% in one bar)
        let max_change = ((bar.high - bar.low) / bar.low).abs();
        if max_change > 0.5 {
            warn!(
                "Extreme price movement detected for {}: {:.2}% change",
                symbol,
                max_change * 100.0
            );
        }

        Ok(())
    }

    /// Convert timeframe to Alpaca API format
    fn timeframe_to_alpaca(timeframe: Timeframe) -> &'static str {
        match timeframe {
            Timeframe::Minute1 => "1Min",
            Timeframe::Minute5 => "5Min",
            Timeframe::Minute15 => "15Min",
            Timeframe::Minute30 => "30Min",
            Timeframe::Hour1 => "1Hour",
            Timeframe::Hour4 => "4Hour",
            Timeframe::Day1 => "1Day",
            Timeframe::Week1 => "1Week",
            Timeframe::Month1 => "1Month",
        }
    }
}

#[async_trait]
impl MarketDataProvider for AlpacaProvider {
    async fn fetch_bars(
        &self,
        symbol: &str,
        timeframe: Timeframe,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> MarketResult<Vec<Bar>> {
        debug!(
            "Fetching bars for {} from {} to {} with timeframe {:?}",
            symbol, start, end, timeframe
        );

        let mut all_bars = Vec::new();
        let mut page_token: Option<String> = None;
        let timeframe_str = Self::timeframe_to_alpaca(timeframe);

        // Alpaca API paginates at 10,000 bars per request
        loop {
            let mut url = format!(
                "{}/stocks/{}/bars?timeframe={}&start={}&end={}&limit=10000&adjustment=raw&feed=sip",
                self.config.base_url,
                symbol,
                timeframe_str,
                start.to_rfc3339(),
                end.to_rfc3339()
            );

            if let Some(ref token) = page_token {
                url.push_str(&format!("&page_token={}", token));
            }

            debug!("Requesting URL: {}", url);

            let response: AlpacaBarResponse = self.request_with_retry(url).await?;

            // Validate and convert bars
            for alpaca_bar in response.bars {
                match Self::validate_bar(&alpaca_bar, symbol) {
                    Ok(_) => match alpaca_bar.to_bar(symbol.to_string()) {
                        Ok(bar) => all_bars.push(bar),
                        Err(e) => {
                            error!("Failed to convert bar for {}: {}", symbol, e);
                        }
                    },
                    Err(validation_error) => {
                        warn!("Bar validation failed: {}", validation_error);
                    }
                }
            }

            // Check for pagination
            if let Some(next_token) = response.next_page_token {
                debug!("Fetching next page with token: {}", next_token);
                page_token = Some(next_token);
            } else {
                break;
            }
        }

        debug!("Fetched {} bars for {}", all_bars.len(), symbol);
        Ok(all_bars)
    }

    async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar> {
        debug!("Fetching latest bar for {}", symbol);

        let url = format!(
            "{}/stocks/{}/bars/latest?feed=sip",
            self.config.base_url, symbol
        );

        debug!("Requesting URL: {}", url);

        let response: AlpacaLatestBarResponse = self.request_with_retry(url).await?;

        match Self::validate_bar(&response.bar, symbol) {
            Ok(_) => response.bar.to_bar(symbol.to_string()),
            Err(validation_error) => {
                Err(MarketError::ApiError(format!(
                    "Bar validation failed: {}",
                    validation_error
                )))
            }
        }
    }

    fn provider_name(&self) -> &str {
        "Alpaca Markets"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        debug!("Checking symbol support for {}", symbol);

        let url = format!(
            "https://api.alpaca.markets/v2/assets/{}",
            symbol
        );

        match self.request_with_retry::<AlpacaAsset>(url).await {
            Ok(asset) => {
                debug!(
                    "Symbol {} is valid: tradable={}, status={:?}",
                    symbol, asset.tradable, asset.status
                );
                Ok(asset.tradable && asset.status == "active")
            }
            Err(MarketError::InvalidSymbol(_)) => Ok(false),
            Err(e) => {
                warn!("Error checking symbol {}: {}", symbol, e);
                Err(e)
            }
        }
    }
}

/// Alpaca API response structure for historical bars
#[derive(Debug, Deserialize)]
struct AlpacaBarResponse {
    bars: Vec<AlpacaBar>,
    #[allow(dead_code)]
    symbol: String,
    next_page_token: Option<String>,
}

/// Alpaca API response structure for latest bar
#[derive(Debug, Deserialize)]
struct AlpacaLatestBarResponse {
    bar: AlpacaBar,
    #[allow(dead_code)]
    symbol: String,
}

/// Alpaca API response structure for asset information
#[derive(Debug, Deserialize)]
struct AlpacaAsset {
    #[serde(rename = "id")]
    _id: String,

    #[serde(rename = "class")]
    _class: String,

    #[serde(rename = "exchange")]
    _exchange: String,

    #[allow(dead_code)]
    symbol: String,

    status: String,

    tradable: bool,
}

/// Alpaca bar structure from API
#[derive(Debug, Deserialize)]
struct AlpacaBar {
    #[serde(rename = "t")]
    timestamp: String,

    #[serde(rename = "o")]
    open: f64,

    #[serde(rename = "h")]
    high: f64,

    #[serde(rename = "l")]
    low: f64,

    #[serde(rename = "c")]
    close: f64,

    #[serde(rename = "v")]
    volume: u64,

    #[serde(rename = "n", default)]
    trade_count: Option<u64>,

    #[serde(rename = "vw", default)]
    vwap: Option<f64>,
}

impl AlpacaBar {
    /// Convert Alpaca bar to unified Bar structure
    fn to_bar(&self, symbol: String) -> MarketResult<Bar> {
        let timestamp = DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| MarketError::DateTimeParseError(e.to_string()))?
            .map_err(|e| MarketError::ApiError(format!("Invalid timestamp: {}", e)))?
            .with_timezone(&Utc);

        Ok(Bar {
            symbol,
            timestamp,
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume,
            vwap: self.vwap,
            trade_count: self.trade_count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpaca_provider_creation() {
        let provider = AlpacaProvider::new(
            "test_key".to_string(),
            "test_secret".to_string(),
            true,
        );

        assert_eq!(provider.provider_name(), "Alpaca Markets");
        assert!(provider.supports_realtime());
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(AlpacaProvider::timeframe_to_alpaca(Timeframe::Minute1), "1Min");
        assert_eq!(AlpacaProvider::timeframe_to_alpaca(Timeframe::Day1), "1Day");
    }
}
