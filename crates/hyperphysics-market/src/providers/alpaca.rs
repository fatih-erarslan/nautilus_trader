//! Alpaca Markets API provider implementation
//!
//! Provides access to stock market data via Alpaca Markets API.
//! Supports both paper and live trading data.
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
use tracing::debug;

use crate::data::{Bar, Timeframe};
use crate::error::{MarketError, MarketResult};
use super::{MarketDataProvider, ProviderConfig};

/// Alpaca Markets data provider
#[allow(dead_code)]
pub struct AlpacaProvider {
    config: ProviderConfig,
    client: Client,
}

impl AlpacaProvider {
    /// Maximum retries for API requests with transient failures
    const MAX_RETRIES: u32 = 3;

    /// Initial retry delay in milliseconds
    const INITIAL_RETRY_DELAY_MS: u64 = 100;

    /// Create a new Alpaca provider instance
    ///
    /// # Arguments
    ///
    /// * `api_key` - Alpaca API key
    /// * `api_secret` - Alpaca API secret
    /// * `paper` - Whether to use paper trading endpoints
    pub fn new(api_key: String, api_secret: String, paper: bool) -> Self {
        let _base_url = if paper {
            "https://paper-api.alpaca.markets/v2"
        } else {
            "https://api.alpaca.markets/v2"
        }.to_string();

        let data_url = if paper {
            "https://data.alpaca.markets/v2"
        } else {
            "https://data.alpaca.markets/v2"
        }.to_string();

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

        Self { config, client }
    }

    /// Convert timeframe to Alpaca API format
    #[allow(dead_code)]
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

    /// Retry request with exponential backoff
    async fn retry_with_backoff<F, Fut, T>(
        &self,
        mut operation: F,
    ) -> MarketResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = MarketResult<T>>,
    {
        let mut retry_count = 0;
        let mut delay_ms = Self::INITIAL_RETRY_DELAY_MS;

        loop {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    // Only retry on transient errors
                    let should_retry = matches!(
                        e,
                        MarketError::NetworkError(_)
                            | MarketError::TimeoutError(_)
                            | MarketError::ConnectionError(_)
                    );

                    if !should_retry || retry_count >= Self::MAX_RETRIES {
                        return Err(e);
                    }

                    retry_count += 1;
                    debug!(
                        "Retrying request (attempt {}/{}), waiting {}ms",
                        retry_count, Self::MAX_RETRIES, delay_ms
                    );

                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    delay_ms *= 2; // Exponential backoff
                }
            }
        }
    }

    /// Validate OHLC data integrity
    fn validate_bar(bar: &AlpacaBar) -> MarketResult<()> {
        // Check for non-positive prices
        if bar.open <= 0.0 || bar.high <= 0.0 || bar.low <= 0.0 || bar.close <= 0.0 {
            return Err(MarketError::DataIntegrityError(
                "Bar contains non-positive prices".to_string(),
            ));
        }

        // Validate OHLC relationships: high >= max(open, close) and low <= min(open, close)
        let max_price = bar.open.max(bar.close);
        let min_price = bar.open.min(bar.close);

        if bar.high < max_price {
            return Err(MarketError::DataIntegrityError(format!(
                "High price {} is less than max(open, close) {}",
                bar.high, max_price
            )));
        }

        if bar.low > min_price {
            return Err(MarketError::DataIntegrityError(format!(
                "Low price {} is greater than min(open, close) {}",
                bar.low, min_price
            )));
        }

        // Additional sanity check: high must be >= low
        if bar.high < bar.low {
            return Err(MarketError::DataIntegrityError(format!(
                "High price {} is less than low price {}",
                bar.high, bar.low
            )));
        }

        // Validate volume is non-negative
        if bar.volume == 0 {
            return Err(MarketError::DataIntegrityError(
                "Bar has zero volume".to_string(),
            ));
        }

        Ok(())
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

        // Implementation based on Alpaca Market Data API v2
        // Reference: https://docs.alpaca.markets/reference/stockbars
        let url = format!("{}/stocks/{}/bars", self.config.base_url, symbol);
        let timeframe_str = Self::timeframe_to_alpaca(timeframe);

        // RFC3339 format required by Alpaca API
        let start_str = start.to_rfc3339();
        let end_str = end.to_rfc3339();

        let response = self.client
            .get(&url)
            .query(&[
                ("timeframe", timeframe_str),
                ("start", &start_str),
                ("end", &end_str),
                ("limit", "10000"), // Max limit per request
                ("adjustment", "raw"), // No corporate action adjustments
                ("feed", "iex"), // IEX feed for free tier
            ])
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!(
                "Alpaca API error: {} - {}",
                status,
                error_text
            )));
        }

        let api_response: AlpacaBarResponse = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        // Convert Alpaca bars to unified Bar format with validation
        let bars: MarketResult<Vec<Bar>> = api_response
            .bars
            .into_iter()
            .filter_map(|alpaca_bar| {
                // Validate bar data integrity
                match Self::validate_bar(&alpaca_bar) {
                    Ok(()) => Some(alpaca_bar.to_bar(symbol.to_string())),
                    Err(e) => {
                        debug!("Skipping invalid bar: {}", e);
                        None
                    }
                }
            })
            .collect();

        // Handle pagination if next_page_token exists
        // TODO: Implement recursive pagination for large datasets
        if api_response.next_page_token.is_some() {
            debug!("More data available via pagination token (not implemented)");
        }

        bars
    }

    async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar> {
        debug!("Fetching latest bar for {}", symbol);

        // Implementation based on Alpaca Market Data API v2
        // Reference: https://docs.alpaca.markets/reference/stocklatestbar
        let url = format!("{}/stocks/{}/bars/latest", self.config.base_url, symbol);

        let response = self.client
            .get(&url)
            .query(&[("feed", "iex")]) // IEX feed for free tier
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        let status = response.status();
        if !status.is_success() {
            let error_text = response.text().await.unwrap_or_default();
            return Err(MarketError::ApiError(format!(
                "Alpaca API error: {} - {}",
                status,
                error_text
            )));
        }

        // Parse response which contains a single bar under "bar" key
        let json: serde_json::Value = response
            .json()
            .await
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        let bar_value = json.get("bar").ok_or_else(|| {
            MarketError::ParseError("Missing 'bar' field in latest bar response".to_string())
        })?;

        let alpaca_bar: AlpacaBar = serde_json::from_value(bar_value.clone())
            .map_err(|e| MarketError::ParseError(e.to_string()))?;

        // Validate bar data integrity
        Self::validate_bar(&alpaca_bar)?;

        alpaca_bar.to_bar(symbol.to_string())
    }

    fn provider_name(&self) -> &str {
        "Alpaca Markets"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        // Implementation based on Alpaca Assets API
        // Reference: https://docs.alpaca.markets/reference/get-v2-assets-symbol
        debug!("Checking symbol support for {}", symbol);

        let url = format!(
            "https://api.alpaca.markets/v2/assets/{}",
            symbol.to_uppercase()
        );

        let response = self.client
            .get(&url)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        // 200 = asset exists and is valid
        // 404 = asset not found
        match response.status().as_u16() {
            200 => {
                // Verify asset is tradable
                let asset: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| MarketError::ParseError(e.to_string()))?;

                let tradable = asset
                    .get("tradable")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                let active = asset
                    .get("status")
                    .and_then(|v| v.as_str())
                    .map(|s| s == "active")
                    .unwrap_or(false);

                Ok(tradable && active)
            }
            404 => Ok(false),
            _ => {
                let error_text = response.text().await.unwrap_or_default();
                Err(MarketError::ApiError(format!(
                    "Alpaca asset check error: {}",
                    error_text
                )))
            }
        }
    }
}

/// Alpaca API response structure for bars
#[allow(dead_code)]
#[derive(Debug, Deserialize)]
struct AlpacaBarResponse {
    bars: Vec<AlpacaBar>,
    symbol: String,
    next_page_token: Option<String>,
}

/// Alpaca bar structure from API
#[allow(dead_code)]
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
    #[allow(dead_code)]
    fn to_bar(&self, symbol: String) -> MarketResult<Bar> {
        let timestamp = DateTime::parse_from_rfc3339(&self.timestamp)
            .map_err(|e| MarketError::DateTimeParseError(e.to_string()))?
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
