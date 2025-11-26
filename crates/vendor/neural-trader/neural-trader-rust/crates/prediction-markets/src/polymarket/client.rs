//! Polymarket HTTP client implementation

use crate::error::{PredictionMarketError, Result};
use crate::models::*;
use reqwest::{header, Client, StatusCode};
use serde::Deserialize;
use std::time::Duration;
use tracing::{debug, error, info};

const DEFAULT_BASE_URL: &str = "https://clob.polymarket.com";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Polymarket CLOB client configuration
#[derive(Debug, Clone)]
pub struct ClientConfig {
    pub base_url: String,
    pub api_key: String,
    pub timeout: Duration,
    pub max_retries: u32,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            base_url: DEFAULT_BASE_URL.to_string(),
            api_key: String::new(),
            timeout: DEFAULT_TIMEOUT,
            max_retries: 3,
        }
    }
}

impl ClientConfig {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            ..Default::default()
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn with_max_retries(mut self, retries: u32) -> Self {
        self.max_retries = retries;
        self
    }
}

/// Polymarket CLOB HTTP client
#[derive(Clone)]
pub struct PolymarketClient {
    config: ClientConfig,
    http_client: Client,
}

impl PolymarketClient {
    /// Create a new Polymarket client
    pub fn new(config: ClientConfig) -> Result<Self> {
        let mut headers = header::HeaderMap::new();
        headers.insert(
            header::AUTHORIZATION,
            header::HeaderValue::from_str(&format!("Bearer {}", config.api_key))
                .map_err(|e| PredictionMarketError::AuthError(e.to_string()))?,
        );
        headers.insert(
            header::CONTENT_TYPE,
            header::HeaderValue::from_static("application/json"),
        );

        let http_client = Client::builder()
            .timeout(config.timeout)
            .default_headers(headers)
            .build()?;

        Ok(Self {
            config,
            http_client,
        })
    }

    /// Get all active markets
    pub async fn get_markets(&self) -> Result<Vec<Market>> {
        info!("Fetching all markets");
        let url = format!("{}/markets", self.config.base_url);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get specific market by ID
    pub async fn get_market(&self, market_id: &str) -> Result<Market> {
        info!("Fetching market: {}", market_id);
        let url = format!("{}/markets/{}", self.config.base_url, market_id);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get orderbook for a market outcome
    pub async fn get_orderbook(&self, market_id: &str, outcome_id: &str) -> Result<OrderBook> {
        debug!("Fetching orderbook for market {} outcome {}", market_id, outcome_id);
        let url = format!(
            "{}/markets/{}/outcomes/{}/orderbook",
            self.config.base_url, market_id, outcome_id
        );

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Place an order
    pub async fn place_order(&self, order: OrderRequest) -> Result<OrderResponse> {
        info!("Placing order: {:?}", order);

        // Validate order
        order.validate().map_err(PredictionMarketError::InvalidOrder)?;

        let url = format!("{}/orders", self.config.base_url);

        let response = self.http_client
            .post(&url)
            .json(&order)
            .send()
            .await?;

        self.handle_response(response).await
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        info!("Cancelling order: {}", order_id);
        let url = format!("{}/orders/{}", self.config.base_url, order_id);

        let response = self.http_client.delete(&url).send().await?;

        if response.status().is_success() {
            Ok(())
        } else {
            let status = response.status();
            let text = response.text().await?;
            Err(PredictionMarketError::from_status(status.as_u16(), text))
        }
    }

    /// Get all orders for a market
    pub async fn get_orders(&self, market_id: &str) -> Result<Vec<Order>> {
        debug!("Fetching orders for market: {}", market_id);
        let url = format!("{}/markets/{}/orders", self.config.base_url, market_id);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get specific order by ID
    pub async fn get_order(&self, order_id: &str) -> Result<Order> {
        debug!("Fetching order: {}", order_id);
        let url = format!("{}/orders/{}", self.config.base_url, order_id);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get all positions
    pub async fn get_positions(&self) -> Result<Vec<Position>> {
        info!("Fetching positions");
        let url = format!("{}/positions", self.config.base_url);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get positions for a specific market
    pub async fn get_market_positions(&self, market_id: &str) -> Result<Vec<Position>> {
        debug!("Fetching positions for market: {}", market_id);
        let url = format!("{}/markets/{}/positions", self.config.base_url, market_id);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get market statistics
    pub async fn get_market_stats(&self, market_id: &str) -> Result<MarketStats> {
        debug!("Fetching stats for market: {}", market_id);
        let url = format!("{}/markets/{}/stats", self.config.base_url, market_id);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Search markets by query
    pub async fn search_markets(&self, query: &str, limit: Option<usize>) -> Result<Vec<Market>> {
        info!("Searching markets: {}", query);
        let mut url = format!("{}/markets/search?q={}", self.config.base_url, query);

        if let Some(limit) = limit {
            url.push_str(&format!("&limit={}", limit));
        }

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Get markets by category
    pub async fn get_markets_by_category(&self, category: &str) -> Result<Vec<Market>> {
        info!("Fetching markets in category: {}", category);
        let url = format!("{}/markets?category={}", self.config.base_url, category);

        let response = self.http_client.get(&url).send().await?;
        self.handle_response(response).await
    }

    /// Handle HTTP response and deserialize
    async fn handle_response<T: for<'de> Deserialize<'de>>(
        &self,
        response: reqwest::Response,
    ) -> Result<T> {
        let status = response.status();

        if status.is_success() {
            response.json().await.map_err(Into::into)
        } else {
            let text = response.text().await?;
            error!("API error {}: {}", status, text);

            match status {
                StatusCode::TOO_MANY_REQUESTS => {
                    Err(PredictionMarketError::RateLimitExceeded(60))
                }
                StatusCode::NOT_FOUND => {
                    Err(PredictionMarketError::MarketNotFound(text))
                }
                StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN => {
                    Err(PredictionMarketError::AuthError(text))
                }
                _ => Err(PredictionMarketError::from_status(status.as_u16(), text)),
            }
        }
    }

    /// Execute request with retry logic
    #[allow(dead_code)]
    async fn execute_with_retry<F, T>(&self, mut operation: F) -> Result<T>
    where
        F: FnMut() -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<T>> + Send>>,
    {
        let mut attempts = 0;
        let mut last_error_msg = None;

        while attempts < self.config.max_retries {
            match operation().await {
                Ok(result) => return Ok(result),
                Err(e) if e.is_retryable() => {
                    attempts += 1;
                    let delay = e.retry_delay();
                    last_error_msg = Some(e.to_string());

                    if let Some(delay) = delay {
                        tokio::time::sleep(Duration::from_secs(delay)).await;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        Err(PredictionMarketError::InternalError(
            last_error_msg.unwrap_or_else(|| "Max retries exceeded".to_string())
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[test]
    fn test_config_builder() {
        let config = ClientConfig::new("test_key")
            .with_base_url("https://test.com")
            .with_timeout(Duration::from_secs(10))
            .with_max_retries(5);

        assert_eq!(config.api_key, "test_key");
        assert_eq!(config.base_url, "https://test.com");
        assert_eq!(config.timeout, Duration::from_secs(10));
        assert_eq!(config.max_retries, 5);
    }

    #[test]
    fn test_order_request_validation() {
        // Valid limit order
        let valid_order = OrderRequest {
            market_id: "market1".to_string(),
            outcome_id: "yes".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Limit,
            size: dec!(100.0),
            price: Some(dec!(0.6)),
            time_in_force: None,
            client_order_id: None,
        };
        assert!(valid_order.validate().is_ok());

        // Invalid: negative size
        let invalid_order = OrderRequest {
            size: dec!(-10.0),
            ..valid_order.clone()
        };
        assert!(invalid_order.validate().is_err());

        // Invalid: limit order without price
        let invalid_order = OrderRequest {
            price: None,
            ..valid_order.clone()
        };
        assert!(invalid_order.validate().is_err());

        // Invalid: price > 1
        let invalid_order = OrderRequest {
            price: Some(dec!(1.5)),
            ..valid_order
        };
        assert!(invalid_order.validate().is_err());
    }
}
