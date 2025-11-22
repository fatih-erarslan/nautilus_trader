//! Interactive Brokers Client Portal Gateway REST API Provider
//!
//! Production-grade integration with Interactive Brokers using Client Portal Gateway REST API.
//! Provides historical market data, real-time snapshots, and contract resolution.
//!
//! # Architecture
//!
//! - **Connection**: HTTPS REST API to Client Portal Gateway (default: https://localhost:5000)
//! - **Authentication**: Session-based with SSO portal
//! - **Market Data**: Historical OHLCV bars and real-time Level 1 snapshots
//! - **Rate Limiting**: 60 requests per minute with token bucket algorithm
//! - **Contract Resolution**: Symbol search with caching
//!
//! # API References
//!
//! - Client Portal Gateway: <https://www.interactivebrokers.com/api/doc.html>
//! - Historical Data: `/v1/api/hmds/history`
//! - Market Snapshots: `/v1/api/iserver/marketdata/snapshot`
//! - Contract Search: `/v1/api/iserver/secdef/search`
//!
//! # Setup
//!
//! 1. Download and run Client Portal Gateway from IBKR
//! 2. Authenticate via web interface (https://localhost:5000)
//! 3. Use this provider with active session
//!
//! # Example
//!
//! ```no_run
//! use hyperphysics_market::providers::{InteractiveBrokersProvider, MarketDataProvider};
//! use hyperphysics_market::data::Timeframe;
//! use chrono::{Utc, Duration};
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let provider = InteractiveBrokersProvider::new(
//!         "https://localhost:5000".to_string(),
//!     ).await?;
//!
//!     // Authenticate with existing session
//!     provider.authenticate().await?;
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
use chrono::{DateTime, TimeZone, Utc};
use reqwest::{Client, StatusCode};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

use crate::data::{Bar, Tick, Timeframe};
use crate::data::tick::Quote;
use crate::error::{MarketError, MarketResult};
use super::{MarketDataProvider, ProviderConfig};

/// Historical bar data from IBKR Client Portal Gateway
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IBKRBar {
    /// Unix timestamp (milliseconds)
    #[serde(rename = "t")]
    pub timestamp: i64,

    /// Open price
    #[serde(rename = "o")]
    pub open: f64,

    /// High price
    #[serde(rename = "h")]
    pub high: f64,

    /// Low price
    #[serde(rename = "l")]
    pub low: f64,

    /// Close price
    #[serde(rename = "c")]
    pub close: f64,

    /// Volume
    #[serde(rename = "v")]
    pub volume: f64,
}

impl IBKRBar {
    /// Validate OHLC consistency and data integrity
    pub fn validate(&self) -> Result<(), String> {
        // Volume must be non-negative
        if self.volume < 0.0 {
            return Err(format!("Negative volume: {}", self.volume));
        }

        // High >= Low
        if self.high < self.low {
            return Err(format!("High ({}) < Low ({})", self.high, self.low));
        }

        // Open, Close within [Low, High]
        if self.open < self.low || self.open > self.high {
            return Err(format!(
                "Open ({}) outside [Low ({}), High ({})]",
                self.open, self.low, self.high
            ));
        }

        if self.close < self.low || self.close > self.high {
            return Err(format!(
                "Close ({}) outside [Low ({}), High ({})]",
                self.close, self.low, self.high
            ));
        }

        // Timestamp must be positive
        if self.timestamp <= 0 {
            return Err(format!("Invalid timestamp: {}", self.timestamp));
        }

        Ok(())
    }

    /// Convert to internal Bar representation
    pub fn to_bar(&self, symbol: String) -> Bar {
        Bar {
            symbol,
            timestamp: Utc.timestamp_millis_opt(self.timestamp).unwrap(),
            open: self.open,
            high: self.high,
            low: self.low,
            close: self.close,
            volume: self.volume as u64,
            vwap: None, // Not provided by Client Portal Gateway
            trade_count: None,
        }
    }
}

/// Contract search result from IBKR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IBKRContract {
    /// Contract ID (conid)
    #[serde(rename = "conid")]
    pub contract_id: i64,

    /// Company name
    #[serde(rename = "companyName")]
    pub company_name: Option<String>,

    /// Symbol
    #[serde(rename = "symbol")]
    pub symbol: String,

    /// Exchanges where contract trades
    #[serde(rename = "exchanges", default)]
    pub exchanges: Vec<String>,

    /// Security type (STK, FUT, OPT, etc.)
    #[serde(rename = "secType")]
    pub sec_type: String,
}

/// Real-time market snapshot from IBKR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IBKRSnapshot {
    /// Contract ID
    #[serde(rename = "conid")]
    pub contract_id: i64,

    /// Last traded price (field 31)
    #[serde(rename = "31")]
    pub last_price: Option<String>,

    /// Bid price (field 84)
    #[serde(rename = "84")]
    pub bid_price: Option<String>,

    /// Ask price (field 86)
    #[serde(rename = "86")]
    pub ask_price: Option<String>,

    /// Volume (field 87)
    #[serde(rename = "87")]
    pub volume: Option<String>,

    /// Bid size (field 88)
    #[serde(rename = "88")]
    pub bid_size: Option<String>,

    /// Ask size (field 85)
    #[serde(rename = "85")]
    pub ask_size: Option<String>,
}

impl IBKRSnapshot {
    /// Convert IBKR snapshot to Quote
    pub fn to_quote(&self, symbol: String) -> Result<Quote, String> {
        let bid_price = self.bid_price
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| "Missing or invalid bid price".to_string())?;

        let ask_price = self.ask_price
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| "Missing or invalid ask price".to_string())?;

        let bid_size = self.bid_size
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        let ask_size = self.ask_size
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        Ok(Quote {
            symbol,
            timestamp: Utc::now(),
            bid_price,
            bid_size,
            ask_price,
            ask_size,
            exchange: Some("IBKR".to_string()),
        })
    }

    /// Convert IBKR snapshot to Tick (last trade)
    pub fn to_tick(&self, symbol: String) -> Result<Tick, String> {
        let price = self.last_price
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .ok_or_else(|| "Missing or invalid last price".to_string())?;

        let size = self.volume
            .as_ref()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);

        Ok(Tick {
            symbol,
            timestamp: Utc::now(),
            price,
            size,
            exchange: Some("IBKR".to_string()),
            conditions: None,
        })
    }
}

/// Rate limiter using token bucket algorithm
#[derive(Debug)]
struct RateLimiter {
    /// Request timestamps
    requests: Vec<Instant>,

    /// Maximum requests per window
    max_requests: usize,

    /// Time window duration
    window: Duration,
}

impl RateLimiter {
    fn new(max_requests: usize, window: Duration) -> Self {
        Self {
            requests: Vec::new(),
            max_requests,
            window,
        }
    }

    /// Check if request is allowed, return wait time if rate limited
    fn check_rate_limit(&mut self) -> Result<(), Duration> {
        let now = Instant::now();

        // Remove expired requests
        self.requests.retain(|&req| now.duration_since(req) < self.window);

        if self.requests.len() >= self.max_requests {
            let oldest = self.requests[0];
            let wait_time = self
                .window
                .checked_sub(now.duration_since(oldest))
                .unwrap_or(Duration::from_secs(0));
            return Err(wait_time);
        }

        self.requests.push(now);
        Ok(())
    }

    /// Wait for rate limit to allow request
    async fn wait_for_slot(&mut self) {
        loop {
            match self.check_rate_limit() {
                Ok(()) => break,
                Err(wait_time) => {
                    debug!("Rate limited, waiting {:?}", wait_time);
                    tokio::time::sleep(wait_time + Duration::from_millis(100)).await;
                }
            }
        }
    }
}

/// Interactive Brokers Client Portal Gateway provider
pub struct InteractiveBrokersProvider {
    /// HTTP client with cookie store for session management
    client: Client,

    /// Base URL for Client Portal Gateway
    base_url: String,

    /// Configuration
    #[allow(dead_code)]
    config: ProviderConfig,

    /// Session authentication status
    authenticated: Arc<RwLock<bool>>,

    /// Rate limiter (60 req/min)
    rate_limiter: Arc<RwLock<RateLimiter>>,

    /// Contract ID cache (symbol -> conid)
    contract_cache: Arc<RwLock<HashMap<String, i64>>>,

    /// Request timeout
    timeout: Duration,
}

impl InteractiveBrokersProvider {
    /// Create new IBKR Client Portal Gateway provider
    ///
    /// # Arguments
    /// * `base_url` - Client Portal Gateway URL (e.g., "https://localhost:5000")
    pub async fn new(base_url: String) -> MarketResult<Self> {
        info!("Initializing IBKR Client Portal Gateway provider at {}", base_url);

        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .danger_accept_invalid_certs(true) // For localhost SSL
            .build()
            .map_err(|e| MarketError::ConfigError(format!("Failed to build HTTP client: {}", e)))?;

        Ok(Self {
            client,
            base_url: base_url.clone(),
            config: ProviderConfig {
                api_key: String::new(),
                api_secret: String::new(),
                base_url,
                paper_trading: false,
                timeout_secs: 30,
                max_retries: 3,
            },
            authenticated: Arc::new(RwLock::new(false)),
            rate_limiter: Arc::new(RwLock::new(RateLimiter::new(
                60,
                Duration::from_secs(60),
            ))),
            contract_cache: Arc::new(RwLock::new(HashMap::new())),
            timeout: Duration::from_secs(30),
        })
    }

    /// Authenticate with IBKR Client Portal Gateway
    ///
    /// Checks if session is valid. If not authenticated, user must authenticate
    /// via the Client Portal Gateway web interface.
    pub async fn authenticate(&self) -> MarketResult<()> {
        info!("Checking IBKR Client Portal Gateway authentication status");

        let url = format!("{}/v1/api/iserver/auth/status", self.base_url);

        self.rate_limiter.write().await.wait_for_slot().await;

        let response = self
            .client
            .post(&url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        match response.status() {
            StatusCode::OK => {
                let status: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| MarketError::ApiError(format!("Failed to parse auth status: {}", e)))?;

                let authenticated = status["authenticated"].as_bool().unwrap_or(false);

                if authenticated {
                    info!("IBKR session is authenticated");
                    *self.authenticated.write().await = true;
                    Ok(())
                } else {
                    error!("IBKR session not authenticated");
                    Err(MarketError::AuthenticationError(
                        "Session not authenticated. Please authenticate via Client Portal Gateway UI.".to_string()
                    ))
                }
            }
            StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                Err(MarketError::AuthenticationError("Session expired".to_string()))
            }
            code => {
                let body = response.text().await.unwrap_or_default();
                Err(MarketError::ApiError(format!(
                    "Authentication check failed with status {}: {}",
                    code, body
                )))
            }
        }
    }

    /// Ensure authenticated before API calls
    async fn ensure_authenticated(&self) -> MarketResult<()> {
        if !*self.authenticated.read().await {
            return Err(MarketError::AuthenticationError(
                "Not authenticated. Call authenticate() first.".to_string(),
            ));
        }
        Ok(())
    }

    /// Search for contract by symbol
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol (e.g., "AAPL")
    pub async fn search_contract(&self, symbol: &str) -> MarketResult<Vec<IBKRContract>> {
        self.ensure_authenticated().await?;

        info!("Searching contract for symbol: {}", symbol);

        let url = format!("{}/v1/api/iserver/secdef/search", self.base_url);

        self.rate_limiter.write().await.wait_for_slot().await;

        let response = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "symbol": symbol,
                "name": false,
                "secType": "STK"
            }))
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        match response.status() {
            StatusCode::OK => {
                let contracts: Vec<IBKRContract> = response
                    .json()
                    .await
                    .map_err(|e| MarketError::ApiError(format!("Failed to parse contracts: {}", e)))?;

                if contracts.is_empty() {
                    return Err(MarketError::DataUnavailable(format!(
                        "No contracts found for symbol {}",
                        symbol
                    )));
                }

                // Cache first contract ID
                if let Some(contract) = contracts.first() {
                    self.contract_cache
                        .write()
                        .await
                        .insert(symbol.to_string(), contract.contract_id);
                    info!(
                        "Found contract {} for symbol {} (conid: {})",
                        contract.symbol, symbol, contract.contract_id
                    );
                }

                Ok(contracts)
            }
            StatusCode::NOT_FOUND => Err(MarketError::DataUnavailable(format!(
                "Contract not found: {}",
                symbol
            ))),
            StatusCode::TOO_MANY_REQUESTS => Err(MarketError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            )),
            code => {
                let body = response.text().await.unwrap_or_default();
                Err(MarketError::ApiError(format!(
                    "Contract search failed with status {}: {}",
                    code, body
                )))
            }
        }
    }

    /// Get or fetch contract ID for symbol
    async fn get_contract_id(&self, symbol: &str) -> MarketResult<i64> {
        // Check cache first
        if let Some(&contract_id) = self.contract_cache.read().await.get(symbol) {
            return Ok(contract_id);
        }

        // Search and cache
        let contracts = self.search_contract(symbol).await?;

        contracts
            .first()
            .map(|c| c.contract_id)
            .ok_or_else(|| {
                MarketError::DataUnavailable(format!("No contract found for {}", symbol))
            })
    }

    /// Fetch historical bars from Client Portal Gateway
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    /// * `start_date` - Start date (YYYYMMDD format)
    /// * `end_date` - End date (YYYYMMDD format)
    /// * `bar_size` - Bar size (1min, 5min, 1h, 1d)
    pub async fn fetch_historical_bars(
        &self,
        symbol: &str,
        start_date: &str,
        end_date: &str,
        bar_size: &str,
    ) -> MarketResult<Vec<IBKRBar>> {
        self.ensure_authenticated().await?;

        info!(
            "Fetching historical bars for {} from {} to {} ({})",
            symbol, start_date, end_date, bar_size
        );

        let contract_id = self.get_contract_id(symbol).await?;

        // Convert dates to IBKR format (YYYYMMDD-HH:mm:ss)
        let start = format!("{}-00:00:00", start_date.replace("-", ""));
        let end = format!("{}-23:59:59", end_date.replace("-", ""));

        let url = format!(
            "{}/v1/api/hmds/history?conid={}&period=1m&bar={}&startTime={}&endTime={}&outsideRth=false",
            self.base_url, contract_id, bar_size, start, end
        );

        self.rate_limiter.write().await.wait_for_slot().await;

        let response = self
            .client
            .get(&url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        match response.status() {
            StatusCode::OK => {
                let body: serde_json::Value = response
                    .json()
                    .await
                    .map_err(|e| MarketError::ApiError(format!("Failed to parse response: {}", e)))?;

                // IBKR returns bars in "data" array
                let bars_json = body["data"].as_array().ok_or_else(|| {
                    MarketError::ApiError("Missing 'data' array in response".to_string())
                })?;

                let mut bars = Vec::new();

                for bar_json in bars_json {
                    let bar: IBKRBar = serde_json::from_value(bar_json.clone())
                        .map_err(|e| MarketError::ApiError(format!("Failed to parse bar: {}", e)))?;

                    // Validate bar data
                    if let Err(validation_error) = bar.validate() {
                        warn!("Bar validation failed: {}", validation_error);
                        continue;
                    }

                    bars.push(bar);
                }

                // Validate chronological order
                self.validate_chronological_order(&bars)?;

                info!("Fetched {} bars for {}", bars.len(), symbol);
                Ok(bars)
            }
            StatusCode::NOT_FOUND => Err(MarketError::DataUnavailable(format!(
                "No data found for {}",
                symbol
            ))),
            StatusCode::TOO_MANY_REQUESTS => Err(MarketError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            )),
            code => {
                let body = response.text().await.unwrap_or_default();

                // Check for market closed message
                if body.contains("market is closed") || body.contains("no data") {
                    return Err(MarketError::DataUnavailable(format!(
                        "Market closed or no data for {}",
                        symbol
                    )));
                }

                Err(MarketError::ApiError(format!(
                    "Historical data request failed with status {}: {}",
                    code, body
                )))
            }
        }
    }

    /// Fetch real-time market snapshot
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    pub async fn fetch_snapshot(&self, symbol: &str) -> MarketResult<IBKRSnapshot> {
        self.ensure_authenticated().await?;

        info!("Fetching real-time snapshot for {}", symbol);

        let contract_id = self.get_contract_id(symbol).await?;

        let url = format!(
            "{}/v1/api/iserver/marketdata/snapshot?conids={}&fields=31,84,86,87",
            self.base_url, contract_id
        );

        self.rate_limiter.write().await.wait_for_slot().await;

        let response = self
            .client
            .get(&url)
            .timeout(self.timeout)
            .send()
            .await
            .map_err(|e| MarketError::NetworkError(e.to_string()))?;

        match response.status() {
            StatusCode::OK => {
                let snapshots: Vec<IBKRSnapshot> = response
                    .json()
                    .await
                    .map_err(|e| MarketError::ApiError(format!("Failed to parse snapshot: {}", e)))?;

                snapshots.into_iter().next().ok_or_else(|| {
                    MarketError::DataUnavailable("Empty snapshot response".to_string())
                })
            }
            StatusCode::NOT_FOUND => Err(MarketError::DataUnavailable(format!(
                "No snapshot data for {}",
                symbol
            ))),
            StatusCode::TOO_MANY_REQUESTS => Err(MarketError::RateLimitExceeded(
                "Rate limit exceeded".to_string(),
            )),
            code => {
                let body = response.text().await.unwrap_or_default();
                Err(MarketError::ApiError(format!(
                    "Snapshot request failed with status {}: {}",
                    code, body
                )))
            }
        }
    }

    /// Validate bars are in chronological order
    fn validate_chronological_order(&self, bars: &[IBKRBar]) -> MarketResult<()> {
        for window in bars.windows(2) {
            if window[0].timestamp >= window[1].timestamp {
                return Err(MarketError::DataIntegrityError(format!(
                    "Bars not in chronological order: {} >= {}",
                    window[0].timestamp, window[1].timestamp
                )));
            }
        }
        Ok(())
    }

    /// Convert timeframe to IBKR bar size string
    fn timeframe_to_bar_size(timeframe: Timeframe) -> &'static str {
        match timeframe {
            Timeframe::Minute1 => "1min",
            Timeframe::Minute5 => "5min",
            Timeframe::Minute15 => "15min",
            Timeframe::Minute30 => "30min",
            Timeframe::Hour1 => "1h",
            Timeframe::Hour4 => "4h",
            Timeframe::Day1 => "1d",
            Timeframe::Week1 => "1w",
            Timeframe::Month1 => "1M",
        }
    }

    /// Check if gateway is accessible
    pub async fn health_check(&self) -> MarketResult<()> {
        let url = format!("{}/v1/api/iserver/auth/status", self.base_url);

        let response = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
            .map_err(|e| MarketError::ConnectionError(format!("Health check failed: {}", e)))?;

        if response.status().is_success() {
            Ok(())
        } else {
            Err(MarketError::ApiError(format!(
                "Gateway health check failed with status {}",
                response.status()
            )))
        }
    }

    /// Fetch real-time quote (bid/ask) for symbol
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    pub async fn fetch_quote(&self, symbol: &str) -> MarketResult<Quote> {
        info!("Fetching quote for {}", symbol);

        let snapshot = self.fetch_snapshot(symbol).await?;

        snapshot.to_quote(symbol.to_string())
            .map_err(|e| MarketError::DataUnavailable(format!("Failed to convert snapshot to quote: {}", e)))
    }

    /// Fetch latest tick (trade) for symbol
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    pub async fn fetch_tick(&self, symbol: &str) -> MarketResult<Tick> {
        info!("Fetching tick for {}", symbol);

        let snapshot = self.fetch_snapshot(symbol).await?;

        snapshot.to_tick(symbol.to_string())
            .map_err(|e| MarketError::DataUnavailable(format!("Failed to convert snapshot to tick: {}", e)))
    }

    /// Subscribe to real-time market data for symbol
    ///
    /// This initiates market data subscription. Actual streaming requires
    /// WebSocket connection which is not implemented in REST API.
    /// Use this for snapshot-based polling.
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    pub async fn subscribe_market_data(&self, symbol: &str) -> MarketResult<()> {
        self.ensure_authenticated().await?;

        info!("Subscribing to market data for {}", symbol);

        let contract_id = self.get_contract_id(symbol).await?;

        let url = format!(
            "{}/v1/api/iserver/marketdata/snapshot?conids={}",
            self.base_url, contract_id
        );

        self.rate_limiter.write().await.wait_for_slot().await;

        let response = self
            .client
            .get(&url)
            .timeout(self.timeout)
            .send()
            .await?;

        if response.status().is_success() {
            info!("Successfully subscribed to market data for {}", symbol);
            Ok(())
        } else {
            let body = response.text().await.unwrap_or_default();
            Err(MarketError::ApiError(format!(
                "Market data subscription failed: {}",
                body
            )))
        }
    }

    /// Unsubscribe from market data for symbol
    ///
    /// # Arguments
    /// * `symbol` - Stock symbol
    pub async fn unsubscribe_market_data(&self, symbol: &str) -> MarketResult<()> {
        self.ensure_authenticated().await?;

        info!("Unsubscribing from market data for {}", symbol);

        let contract_id = self.get_contract_id(symbol).await?;

        let url = format!(
            "{}/v1/api/iserver/marketdata/unsubscribe?conid={}",
            self.base_url, contract_id
        );

        self.rate_limiter.write().await.wait_for_slot().await;

        let response = self
            .client
            .get(&url)
            .timeout(self.timeout)
            .send()
            .await?;

        if response.status().is_success() {
            info!("Successfully unsubscribed from market data for {}", symbol);
            Ok(())
        } else {
            let body = response.text().await.unwrap_or_default();
            warn!("Market data unsubscription warning: {}", body);
            Ok(()) // Don't fail on unsubscribe errors
        }
    }

    /// Validate session is still active
    ///
    /// Should be called periodically to ensure connection remains valid
    pub async fn validate_session(&self) -> MarketResult<bool> {
        let url = format!("{}/v1/api/iserver/auth/status", self.base_url);

        self.rate_limiter.write().await.wait_for_slot().await;

        match self
            .client
            .post(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await
        {
            Ok(response) => {
                if let Ok(status) = response.json::<serde_json::Value>().await {
                    let authenticated = status["authenticated"].as_bool().unwrap_or(false);
                    *self.authenticated.write().await = authenticated;
                    Ok(authenticated)
                } else {
                    Ok(false)
                }
            }
            Err(_) => Ok(false),
        }
    }

    /// Reconnect to gateway if session expired
    pub async fn reconnect(&self) -> MarketResult<()> {
        info!("Attempting to reconnect to IBKR gateway");

        *self.authenticated.write().await = false;

        self.authenticate().await?;

        info!("Successfully reconnected to IBKR gateway");
        Ok(())
    }

    /// Execute request with automatic retry logic
    ///
    /// # Arguments
    /// * `max_retries` - Maximum number of retry attempts
    /// * `f` - Async function to execute
    pub async fn with_retry<F, Fut, T>(&self, max_retries: u32, mut f: F) -> MarketResult<T>
    where
        F: FnMut() -> Fut,
        Fut: std::future::Future<Output = MarketResult<T>>,
    {
        let mut attempts = 0;
        let mut last_error = None;

        while attempts <= max_retries {
            match f().await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    attempts += 1;
                    last_error = Some(e);

                    if attempts <= max_retries {
                        // Exponential backoff
                        let delay = Duration::from_millis(100 * 2_u64.pow(attempts - 1));
                        debug!("Retry attempt {} after {:?}", attempts, delay);
                        tokio::time::sleep(delay).await;

                        // Check if we need to reauthenticate
                        if !self.validate_session().await.unwrap_or(false) {
                            if let Err(auth_err) = self.reconnect().await {
                                warn!("Failed to reconnect: {}", auth_err);
                            }
                        }
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            MarketError::ApiError("Max retries exceeded".to_string())
        }))
    }
}

#[async_trait]
impl MarketDataProvider for InteractiveBrokersProvider {
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

        let start_date = start.format("%Y%m%d").to_string();
        let end_date = end.format("%Y%m%d").to_string();
        let bar_size = Self::timeframe_to_bar_size(timeframe);

        let ibkr_bars = self
            .fetch_historical_bars(symbol, &start_date, &end_date, bar_size)
            .await?;

        // Convert IBKR bars to internal Bar representation
        let bars: Vec<Bar> = ibkr_bars
            .into_iter()
            .map(|b| b.to_bar(symbol.to_string()))
            .collect();

        debug!("Fetched {} bars for {}", bars.len(), symbol);
        Ok(bars)
    }

    async fn fetch_latest_bar(&self, symbol: &str) -> MarketResult<Bar> {
        debug!("Fetching latest bar for {} from IBKR", symbol);

        // Try to get snapshot first (real-time)
        match self.fetch_snapshot(symbol).await {
            Ok(snapshot) => {
                if let Some(last_price_str) = snapshot.last_price {
                    if let Ok(last_price) = last_price_str.parse::<f64>() {
                        return Ok(Bar {
                            symbol: symbol.to_string(),
                            timestamp: Utc::now(),
                            open: last_price,
                            high: last_price,
                            low: last_price,
                            close: last_price,
                            volume: 0,
                            vwap: None,
                            trade_count: None,
                        });
                    }
                }
            }
            Err(e) => {
                warn!("Failed to fetch snapshot, falling back to historical: {}", e);
            }
        }

        // Fallback to latest historical bar
        let end = Utc::now();
        let start = end - chrono::Duration::days(1);

        let mut bars = self
            .fetch_bars(symbol, Timeframe::Minute1, start, end)
            .await?;

        bars.pop().ok_or_else(|| {
            MarketError::DataUnavailable(format!("No recent data for {}", symbol))
        })
    }

    fn provider_name(&self) -> &str {
        "Interactive Brokers (Client Portal Gateway)"
    }

    fn supports_realtime(&self) -> bool {
        true
    }

    async fn supports_symbol(&self, symbol: &str) -> MarketResult<bool> {
        // Try to search for contract
        match self.search_contract(symbol).await {
            Ok(contracts) => Ok(!contracts.is_empty()),
            Err(_) => Ok(false),
        }
    }
}

impl Drop for InteractiveBrokersProvider {
    fn drop(&mut self) {
        debug!("Closing IBKR Client Portal Gateway connection");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bar_validation_valid() {
        let bar = IBKRBar {
            timestamp: 1640995200000,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000000.0,
        };

        assert!(bar.validate().is_ok());
    }

    #[test]
    fn test_bar_validation_high_less_than_low() {
        let bar = IBKRBar {
            timestamp: 1640995200000,
            open: 100.0,
            high: 99.0, // Invalid: high < low
            low: 100.0,
            close: 100.0,
            volume: 1000.0,
        };

        assert!(bar.validate().is_err());
    }

    #[test]
    fn test_bar_validation_negative_volume() {
        let bar = IBKRBar {
            timestamp: 1640995200000,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: -1000.0, // Invalid
        };

        assert!(bar.validate().is_err());
    }

    #[test]
    fn test_bar_validation_open_out_of_range() {
        let bar = IBKRBar {
            timestamp: 1640995200000,
            open: 110.0, // Invalid: open > high
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000.0,
        };

        assert!(bar.validate().is_err());
    }

    #[test]
    fn test_bar_validation_close_out_of_range() {
        let bar = IBKRBar {
            timestamp: 1640995200000,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 98.0, // Invalid: close < low
            volume: 1000.0,
        };

        assert!(bar.validate().is_err());
    }

    #[test]
    fn test_bar_validation_invalid_timestamp() {
        let bar = IBKRBar {
            timestamp: -1, // Invalid
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000.0,
        };

        assert!(bar.validate().is_err());
    }

    #[test]
    fn test_rate_limiter() {
        let mut limiter = RateLimiter::new(3, Duration::from_secs(1));

        // First 3 requests should succeed
        assert!(limiter.check_rate_limit().is_ok());
        assert!(limiter.check_rate_limit().is_ok());
        assert!(limiter.check_rate_limit().is_ok());

        // 4th request should fail
        assert!(limiter.check_rate_limit().is_err());
    }

    #[tokio::test]
    async fn test_rate_limiter_async() {
        let mut limiter = RateLimiter::new(2, Duration::from_millis(200));

        limiter.wait_for_slot().await;
        limiter.wait_for_slot().await;

        // Third request should wait
        let start = Instant::now();
        limiter.wait_for_slot().await;
        let elapsed = start.elapsed();

        // Should have waited at least 100ms (with some margin)
        assert!(elapsed >= Duration::from_millis(50));
    }

    #[tokio::test]
    async fn test_provider_creation() {
        let provider = InteractiveBrokersProvider::new("https://localhost:5000".to_string()).await;
        assert!(provider.is_ok());

        let provider = provider.unwrap();
        assert_eq!(provider.provider_name(), "Interactive Brokers (Client Portal Gateway)");
        assert!(provider.supports_realtime());
    }

    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Minute1),
            "1min"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Minute5),
            "5min"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Minute15),
            "15min"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Minute30),
            "30min"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Hour1),
            "1h"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Hour4),
            "4h"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Day1),
            "1d"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Week1),
            "1w"
        );
        assert_eq!(
            InteractiveBrokersProvider::timeframe_to_bar_size(Timeframe::Month1),
            "1M"
        );
    }

    #[tokio::test]
    async fn test_chronological_order_validation() {
        let provider = InteractiveBrokersProvider::new("https://localhost:5000".to_string())
            .await
            .unwrap();

        let bars = vec![
            IBKRBar {
                timestamp: 1640995200000,
                open: 100.0,
                high: 105.0,
                low: 99.0,
                close: 103.0,
                volume: 1000.0,
            },
            IBKRBar {
                timestamp: 1640995260000,
                open: 103.0,
                high: 107.0,
                low: 102.0,
                close: 106.0,
                volume: 1500.0,
            },
        ];

        assert!(provider.validate_chronological_order(&bars).is_ok());

        // Test invalid order
        let invalid_bars = vec![
            IBKRBar {
                timestamp: 1640995260000,
                open: 103.0,
                high: 107.0,
                low: 102.0,
                close: 106.0,
                volume: 1500.0,
            },
            IBKRBar {
                timestamp: 1640995200000, // Earlier timestamp
                open: 100.0,
                high: 105.0,
                low: 99.0,
                close: 103.0,
                volume: 1000.0,
            },
        ];

        assert!(provider.validate_chronological_order(&invalid_bars).is_err());
    }

    #[test]
    fn test_bar_conversion() {
        let ibkr_bar = IBKRBar {
            timestamp: 1640995200000,
            open: 100.0,
            high: 105.0,
            low: 99.0,
            close: 103.0,
            volume: 1000000.0,
        };

        let bar = ibkr_bar.to_bar("AAPL".to_string());

        assert_eq!(bar.symbol, "AAPL");
        assert_eq!(bar.open, 100.0);
        assert_eq!(bar.high, 105.0);
        assert_eq!(bar.low, 99.0);
        assert_eq!(bar.close, 103.0);
        assert_eq!(bar.volume, 1000000);
        assert!(bar.vwap.is_none());
        assert!(bar.trade_count.is_none());
    }

    #[test]
    fn test_snapshot_to_quote() {
        let snapshot = IBKRSnapshot {
            contract_id: 265598,
            last_price: Some("150.25".to_string()),
            bid_price: Some("150.20".to_string()),
            ask_price: Some("150.30".to_string()),
            volume: Some("1000000".to_string()),
            bid_size: Some("100".to_string()),
            ask_size: Some("200".to_string()),
        };

        let quote = snapshot.to_quote("AAPL".to_string()).unwrap();

        assert_eq!(quote.symbol, "AAPL");
        assert_eq!(quote.bid_price, 150.20);
        assert_eq!(quote.ask_price, 150.30);
        assert_eq!(quote.bid_size, 100.0);
        assert_eq!(quote.ask_size, 200.0);
        // Use approximate comparisons for floating-point calculations
        assert!((quote.spread() - 0.10).abs() < 1e-10);
        assert!((quote.mid_price() - 150.25).abs() < 1e-10);
        assert_eq!(quote.exchange, Some("IBKR".to_string()));
    }

    #[test]
    fn test_snapshot_to_tick() {
        let snapshot = IBKRSnapshot {
            contract_id: 265598,
            last_price: Some("150.25".to_string()),
            bid_price: Some("150.20".to_string()),
            ask_price: Some("150.30".to_string()),
            volume: Some("500".to_string()),
            bid_size: None,
            ask_size: None,
        };

        let tick = snapshot.to_tick("AAPL".to_string()).unwrap();

        assert_eq!(tick.symbol, "AAPL");
        assert_eq!(tick.price, 150.25);
        assert_eq!(tick.size, 500.0);
        assert_eq!(tick.exchange, Some("IBKR".to_string()));
        assert!(tick.conditions.is_none());
    }

    #[test]
    fn test_snapshot_to_quote_missing_data() {
        let snapshot = IBKRSnapshot {
            contract_id: 265598,
            last_price: Some("150.25".to_string()),
            bid_price: None, // Missing bid
            ask_price: Some("150.30".to_string()),
            volume: None,
            bid_size: None,
            ask_size: None,
        };

        let result = snapshot.to_quote("AAPL".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_to_tick_missing_price() {
        let snapshot = IBKRSnapshot {
            contract_id: 265598,
            last_price: None, // Missing last price
            bid_price: Some("150.20".to_string()),
            ask_price: Some("150.30".to_string()),
            volume: Some("500".to_string()),
            bid_size: None,
            ask_size: None,
        };

        let result = snapshot.to_tick("AAPL".to_string());
        assert!(result.is_err());
    }

    #[test]
    fn test_contract_search_response() {
        let contract = IBKRContract {
            contract_id: 265598,
            company_name: Some("Apple Inc.".to_string()),
            symbol: "AAPL".to_string(),
            exchanges: vec!["NASDAQ".to_string(), "NYSE".to_string()],
            sec_type: "STK".to_string(),
        };

        assert_eq!(contract.symbol, "AAPL");
        assert_eq!(contract.sec_type, "STK");
        assert!(contract.exchanges.contains(&"NASDAQ".to_string()));
    }
}
