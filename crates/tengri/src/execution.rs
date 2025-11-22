//! Order execution engine for Tengri trading strategy
//! 
//! Provides intelligent order routing, execution algorithms, and
//! integration with Binance Spot and Futures exchanges.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{broadcast, mpsc, RwLock};
use tokio::time::{interval, sleep, Instant};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use reqwest::Client;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use uuid::Uuid;

use crate::{Result, TengriError};
use crate::config::{ExchangesConfig, BinanceConfig};
use crate::types::{Order, OrderType, OrderSide, OrderStatus, TimeInForce};

/// Order execution engine for multi-exchange trading
pub struct ExecutionEngine {
    config: ExchangesConfig,
    binance_spot: Option<BinanceExchange>,
    binance_futures: Option<BinanceExchange>,
    order_queue: Arc<RwLock<Vec<PendingOrder>>>,
    execution_metrics: Arc<RwLock<ExecutionMetrics>>,
    fill_events: broadcast::Sender<FillEvent>,
}

/// Binance exchange client for order execution
pub struct BinanceExchange {
    config: BinanceConfig,
    client: Client,
    rate_limiter: RateLimiter,
    order_tracker: Arc<RwLock<HashMap<String, Order>>>,
}

/// Pending order in execution queue
#[derive(Debug, Clone)]
pub struct PendingOrder {
    pub order: Order,
    pub target_exchange: String,
    pub priority: OrderPriority,
    pub created_at: Instant,
    pub retry_count: u32,
}

/// Order priority for execution queue
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum OrderPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Emergency = 3,
}

/// Fill event notification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FillEvent {
    pub order_id: String,
    pub symbol: String,
    pub side: OrderSide,
    pub filled_quantity: f64,
    pub average_price: f64,
    pub timestamp: DateTime<Utc>,
    pub exchange: String,
    pub commission: f64,
    pub commission_asset: String,
}

/// Execution metrics for monitoring
#[derive(Debug, Clone, Default)]
pub struct ExecutionMetrics {
    pub total_orders: u64,
    pub filled_orders: u64,
    pub canceled_orders: u64,
    pub rejected_orders: u64,
    pub average_fill_time_ms: f64,
    pub average_slippage_bps: f64,
    pub total_commission: f64,
    pub last_updated: DateTime<Utc>,
}

/// Rate limiter for exchange API calls
pub struct RateLimiter {
    requests_per_minute: u32,
    orders_per_second: u32,
    weight_per_minute: u32,
    request_count: Arc<RwLock<u32>>,
    order_count: Arc<RwLock<u32>>,
    weight_count: Arc<RwLock<u32>>,
    last_reset: Arc<RwLock<Instant>>,
}

/// Binance API response structures
#[derive(Debug, Deserialize)]
struct BinanceOrderResponse {
    #[serde(rename = "orderId")]
    order_id: u64,
    symbol: String,
    status: String,
    #[serde(rename = "type")]
    order_type: String,
    side: String,
    #[serde(rename = "origQty")]
    orig_qty: String,
    price: String,
    #[serde(rename = "executedQty")]
    executed_qty: String,
    #[serde(rename = "cummulativeQuoteQty")]
    cumulative_quote_qty: String,
    #[serde(rename = "transactTime")]
    transact_time: u64,
}

#[derive(Debug, Deserialize)]
struct BinanceAccountInfo {
    #[serde(rename = "makerCommission")]
    maker_commission: u32,
    #[serde(rename = "takerCommission")]
    taker_commission: u32,
    #[serde(rename = "buyerCommission")]
    buyer_commission: u32,
    #[serde(rename = "sellerCommission")]
    seller_commission: u32,
    #[serde(rename = "canTrade")]
    can_trade: bool,
    #[serde(rename = "canWithdraw")]
    can_withdraw: bool,
    #[serde(rename = "canDeposit")]
    can_deposit: bool,
    balances: Vec<BinanceBalance>,
}

#[derive(Debug, Deserialize)]
struct BinanceBalance {
    asset: String,
    free: String,
    locked: String,
}

impl ExecutionEngine {
    /// Create new execution engine
    pub async fn new(config: ExchangesConfig) -> Result<Self> {
        let (fill_sender, _) = broadcast::channel(1000);

        let binance_spot = if config.binance_spot.enabled {
            Some(BinanceExchange::new(config.binance_spot.clone(), false).await?)
        } else {
            None
        };

        let binance_futures = if config.binance_futures.enabled {
            Some(BinanceExchange::new(config.binance_futures.clone(), true).await?)
        } else {
            None
        };

        Ok(Self {
            config,
            binance_spot,
            binance_futures,
            order_queue: Arc::new(RwLock::new(Vec::new())),
            execution_metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
            fill_events: fill_sender,
        })
    }

    /// Create a market order
    pub async fn create_market_order(&self, symbol: &str, side: OrderSide, quantity: f64) -> Result<Order> {
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            order_type: OrderType::Market,
            side,
            quantity,
            price: None,
            status: OrderStatus::New,
            filled_quantity: 0.0,
            average_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            time_in_force: TimeInForce::IOC,
        };

        self.submit_order(order, OrderPriority::Normal).await
    }

    /// Create a limit order
    pub async fn create_limit_order(&self, symbol: &str, side: OrderSide, quantity: f64, price: f64) -> Result<Order> {
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            order_type: OrderType::Limit,
            side,
            quantity,
            price: Some(price),
            status: OrderStatus::New,
            filled_quantity: 0.0,
            average_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            time_in_force: TimeInForce::GTC,
        };

        self.submit_order(order, OrderPriority::Normal).await
    }

    /// Create a stop loss order
    pub async fn create_stop_loss_order(&self, symbol: &str, side: OrderSide, quantity: f64, stop_price: f64) -> Result<Order> {
        let order = Order {
            id: Uuid::new_v4().to_string(),
            symbol: symbol.to_string(),
            order_type: OrderType::StopLoss,
            side,
            quantity,
            price: Some(stop_price),
            status: OrderStatus::New,
            filled_quantity: 0.0,
            average_price: None,
            created_at: Utc::now(),
            updated_at: Utc::now(),
            time_in_force: TimeInForce::GTC,
        };

        self.submit_order(order, OrderPriority::High).await
    }

    /// Submit order to execution queue
    async fn submit_order(&self, order: Order, priority: OrderPriority) -> Result<Order> {
        let exchange = self.determine_best_exchange(&order.symbol).await?;
        
        let pending_order = PendingOrder {
            order: order.clone(),
            target_exchange: exchange,
            priority,
            created_at: Instant::now(),
            retry_count: 0,
        };

        {
            let mut queue = self.order_queue.write().await;
            queue.push(pending_order);
            
            // Sort by priority (highest first)
            queue.sort_by(|a, b| b.priority.cmp(&a.priority));
        }

        // Start order processing if not already running
        self.process_order_queue().await?;

        Ok(order)
    }

    /// Determine the best exchange for order execution
    async fn determine_best_exchange(&self, symbol: &str) -> Result<String> {
        // For now, simple logic - prefer spot for most pairs, futures for leveraged trading
        if symbol.ends_with("USDT") || symbol.ends_with("BUSD") {
            if self.binance_spot.is_some() {
                return Ok("binance_spot".to_string());
            }
        }

        if self.binance_futures.is_some() {
            return Ok("binance_futures".to_string());
        }

        Err(TengriError::Strategy("No suitable exchange available".to_string()))
    }

    /// Process order queue
    async fn process_order_queue(&self) -> Result<()> {
        let queue = {
            let mut queue_guard = self.order_queue.write().await;
            std::mem::take(&mut *queue_guard)
        };

        for pending_order in queue {
            if let Err(e) = self.execute_order(pending_order).await {
                tracing::error!("Failed to execute order: {}", e);
            }
        }

        Ok(())
    }

    /// Execute a single order
    async fn execute_order(&self, pending_order: PendingOrder) -> Result<()> {
        let exchange_name = &pending_order.target_exchange;
        let order = &pending_order.order;

        tracing::info!("Executing order {} on {}", order.id, exchange_name);

        let result = match exchange_name.as_str() {
            "binance_spot" => {
                if let Some(ref exchange) = self.binance_spot {
                    exchange.place_order(order).await
                } else {
                    Err(TengriError::Strategy("Binance Spot not available".to_string()))
                }
            }
            "binance_futures" => {
                if let Some(ref exchange) = self.binance_futures {
                    exchange.place_order(order).await
                } else {
                    Err(TengriError::Strategy("Binance Futures not available".to_string()))
                }
            }
            _ => Err(TengriError::Strategy(format!("Unknown exchange: {}", exchange_name)))
        };

        match result {
            Ok(exchange_order_id) => {
                tracing::info!("Order {} placed successfully with exchange ID: {}", 
                    order.id, exchange_order_id);
                
                // Update metrics
                {
                    let mut metrics = self.execution_metrics.write().await;
                    metrics.total_orders += 1;
                    metrics.last_updated = Utc::now();
                }
            }
            Err(e) => {
                tracing::error!("Failed to place order {}: {}", order.id, e);
                
                // Handle retry logic here
                if pending_order.retry_count < 3 {
                    let mut retry_order = pending_order.clone();
                    retry_order.retry_count += 1;
                    
                    // Add back to queue with delay
                    sleep(Duration::from_secs(1)).await;
                    
                    let mut queue = self.order_queue.write().await;
                    queue.push(retry_order);
                }
            }
        }

        Ok(())
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        tracing::info!("Canceling order {}", order_id);

        // Try to cancel on all possible exchanges
        let mut cancel_success = false;

        if let Some(ref exchange) = self.binance_spot {
            if let Ok(_) = exchange.cancel_order(order_id).await {
                cancel_success = true;
            }
        }

        if let Some(ref exchange) = self.binance_futures {
            if let Ok(_) = exchange.cancel_order(order_id).await {
                cancel_success = true;
            }
        }

        if cancel_success {
            // Update metrics
            let mut metrics = self.execution_metrics.write().await;
            metrics.canceled_orders += 1;
            metrics.last_updated = Utc::now();
            
            Ok(())
        } else {
            Err(TengriError::Strategy(format!("Failed to cancel order {}", order_id)))
        }
    }

    /// Get order status
    pub async fn get_order_status(&self, order_id: &str, exchange: &str) -> Result<Order> {
        match exchange {
            "binance_spot" => {
                if let Some(ref exchange) = self.binance_spot {
                    exchange.get_order_status(order_id).await
                } else {
                    Err(TengriError::Strategy("Binance Spot not available".to_string()))
                }
            }
            "binance_futures" => {
                if let Some(ref exchange) = self.binance_futures {
                    exchange.get_order_status(order_id).await
                } else {
                    Err(TengriError::Strategy("Binance Futures not available".to_string()))
                }
            }
            _ => Err(TengriError::Strategy(format!("Unknown exchange: {}", exchange)))
        }
    }

    /// Subscribe to fill events
    pub fn subscribe_fills(&self) -> broadcast::Receiver<FillEvent> {
        self.fill_events.subscribe()
    }

    /// Get execution metrics
    pub async fn get_execution_metrics(&self) -> ExecutionMetrics {
        self.execution_metrics.read().await.clone()
    }
}

impl BinanceExchange {
    /// Create new Binance exchange client
    pub async fn new(config: BinanceConfig, is_futures: bool) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| TengriError::Network(e))?;

        let rate_limiter = RateLimiter::new(
            config.rate_limits.requests_per_minute,
            config.rate_limits.orders_per_second,
            config.rate_limits.weight_per_minute,
        );

        let exchange = Self {
            config,
            client,
            rate_limiter,
            order_tracker: Arc::new(RwLock::new(HashMap::new())),
        };

        // Test connection
        exchange.test_connection().await?;

        Ok(exchange)
    }

    /// Test connection to Binance API
    async fn test_connection(&self) -> Result<()> {
        let base_url = self.get_base_url();
        let url = format!("{}/api/v3/ping", base_url);
        
        let response = self.client.get(&url).send().await
            .map_err(|e| TengriError::Network(e))?;

        if response.status().is_success() {
            tracing::info!("Successfully connected to Binance API");
            Ok(())
        } else {
            Err(TengriError::Strategy(format!("Failed to connect to Binance API: {}", response.status())))
        }
    }

    /// Place an order on Binance
    pub async fn place_order(&self, order: &Order) -> Result<String> {
        // Check rate limits
        self.rate_limiter.check_limits().await?;

        let base_url = self.get_base_url();
        let endpoint = format!("{}/api/v3/order", base_url);

        let mut params = HashMap::new();
        params.insert("symbol", order.symbol.clone());
        params.insert("side", format!("{:?}", order.side).to_uppercase());
        params.insert("type", self.map_order_type(&order.order_type));
        params.insert("quantity", order.quantity.to_string());
        params.insert("timeInForce", self.map_time_in_force(&order.time_in_force));

        if let Some(price) = order.price {
            params.insert("price", price.to_string());
        }

        // Add timestamp and signature
        let timestamp = chrono::Utc::now().timestamp_millis();
        params.insert("timestamp", timestamp.to_string());

        let query_string = self.build_query_string(&params);
        let signature = self.sign_request(&query_string)?;
        params.insert("signature", signature);

        let response = self.client
            .post(&endpoint)
            .header("X-MBX-APIKEY", &self.config.api_key)
            .form(&params)
            .send()
            .await
            .map_err(|e| TengriError::Network(e))?;

        if response.status().is_success() {
            let order_response: BinanceOrderResponse = response.json().await
                .map_err(|e| TengriError::Network(e))?;
            
            tracing::info!("Order placed successfully: {}", order_response.order_id);
            Ok(order_response.order_id.to_string())
        } else {
            let error_text = response.text().await.unwrap_or_default();
            Err(TengriError::Strategy(format!("Order placement failed: {}", error_text)))
        }
    }

    /// Cancel an order on Binance
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        self.rate_limiter.check_limits().await?;

        let base_url = self.get_base_url();
        let endpoint = format!("{}/api/v3/order", base_url);

        let mut params = HashMap::new();
        params.insert("orderId", order_id.to_string());
        
        let timestamp = chrono::Utc::now().timestamp_millis();
        params.insert("timestamp", timestamp.to_string());

        let query_string = self.build_query_string(&params);
        let signature = self.sign_request(&query_string)?;
        params.insert("signature", signature);

        let response = self.client
            .delete(&endpoint)
            .header("X-MBX-APIKEY", &self.config.api_key)
            .form(&params)
            .send()
            .await
            .map_err(|e| TengriError::Network(e))?;

        if response.status().is_success() {
            tracing::info!("Order canceled successfully: {}", order_id);
            Ok(())
        } else {
            let error_text = response.text().await.unwrap_or_default();
            Err(TengriError::Strategy(format!("Order cancellation failed: {}", error_text)))
        }
    }

    /// Get order status from Binance
    pub async fn get_order_status(&self, order_id: &str) -> Result<Order> {
        self.rate_limiter.check_limits().await?;

        let base_url = self.get_base_url();
        let endpoint = format!("{}/api/v3/order", base_url);

        let mut params = HashMap::new();
        params.insert("orderId", order_id.to_string());
        
        let timestamp = chrono::Utc::now().timestamp_millis();
        params.insert("timestamp", timestamp.to_string());

        let query_string = self.build_query_string(&params);
        let signature = self.sign_request(&query_string)?;

        let url = format!("{}?{}&signature={}", endpoint, query_string, signature);

        let response = self.client
            .get(&url)
            .header("X-MBX-APIKEY", &self.config.api_key)
            .send()
            .await
            .map_err(|e| TengriError::Network(e))?;

        if response.status().is_success() {
            let order_response: BinanceOrderResponse = response.json().await
                .map_err(|e| TengriError::Network(e))?;
            
            // Convert Binance order to our Order type
            let order = self.convert_binance_order(order_response)?;
            Ok(order)
        } else {
            let error_text = response.text().await.unwrap_or_default();
            Err(TengriError::Strategy(format!("Failed to get order status: {}", error_text)))
        }
    }

    /// Get base URL for API calls
    fn get_base_url(&self) -> &str {
        if self.config.testnet {
            "https://testnet.binance.vision"
        } else {
            self.config.base_url.as_deref().unwrap_or("https://api.binance.com")
        }
    }

    /// Sign request with API secret
    fn sign_request(&self, query_string: &str) -> Result<String> {
        let mut mac = Hmac::<Sha256>::new_from_slice(self.config.api_secret.as_bytes())
            .map_err(|e| TengriError::Strategy(format!("Failed to create HMAC: {}", e)))?;
        
        mac.update(query_string.as_bytes());
        let result = mac.finalize();
        Ok(hex::encode(result.into_bytes()))
    }

    /// Build query string from parameters
    fn build_query_string(&self, params: &HashMap<&str, String>) -> String {
        let mut sorted_params: Vec<_> = params.iter().collect();
        sorted_params.sort_by_key(|&(k, _)| k);
        
        sorted_params
            .iter()
            .map(|(k, v)| format!("{}={}", k, v))
            .collect::<Vec<_>>()
            .join("&")
    }

    /// Map internal order type to Binance order type
    fn map_order_type(&self, order_type: &OrderType) -> String {
        match order_type {
            OrderType::Market => "MARKET".to_string(),
            OrderType::Limit => "LIMIT".to_string(),
            OrderType::StopLoss => "STOP_LOSS".to_string(),
            OrderType::StopLossLimit => "STOP_LOSS_LIMIT".to_string(),
            OrderType::TakeProfit => "TAKE_PROFIT".to_string(),
            OrderType::TakeProfitLimit => "TAKE_PROFIT_LIMIT".to_string(),
        }
    }

    /// Map internal time in force to Binance time in force
    fn map_time_in_force(&self, tif: &TimeInForce) -> String {
        match tif {
            TimeInForce::GTC => "GTC".to_string(),
            TimeInForce::IOC => "IOC".to_string(),
            TimeInForce::FOK => "FOK".to_string(),
            TimeInForce::GTD => "GTD".to_string(),
        }
    }

    /// Convert Binance order response to internal Order type
    fn convert_binance_order(&self, binance_order: BinanceOrderResponse) -> Result<Order> {
        let side = match binance_order.side.as_str() {
            "BUY" => OrderSide::Buy,
            "SELL" => OrderSide::Sell,
            _ => return Err(TengriError::Strategy(format!("Unknown order side: {}", binance_order.side))),
        };

        let status = match binance_order.status.as_str() {
            "NEW" => OrderStatus::New,
            "PARTIALLY_FILLED" => OrderStatus::PartiallyFilled,
            "FILLED" => OrderStatus::Filled,
            "CANCELED" => OrderStatus::Canceled,
            "REJECTED" => OrderStatus::Rejected,
            "EXPIRED" => OrderStatus::Expired,
            _ => return Err(TengriError::Strategy(format!("Unknown order status: {}", binance_order.status))),
        };

        let order_type = match binance_order.order_type.as_str() {
            "MARKET" => OrderType::Market,
            "LIMIT" => OrderType::Limit,
            "STOP_LOSS" => OrderType::StopLoss,
            "STOP_LOSS_LIMIT" => OrderType::StopLossLimit,
            "TAKE_PROFIT" => OrderType::TakeProfit,
            "TAKE_PROFIT_LIMIT" => OrderType::TakeProfitLimit,
            _ => return Err(TengriError::Strategy(format!("Unknown order type: {}", binance_order.order_type))),
        };

        Ok(Order {
            id: binance_order.order_id.to_string(),
            symbol: binance_order.symbol,
            order_type,
            side,
            quantity: binance_order.orig_qty.parse().unwrap_or(0.0),
            price: if binance_order.price.is_empty() { None } else { Some(binance_order.price.parse().unwrap_or(0.0)) },
            status,
            filled_quantity: binance_order.executed_qty.parse().unwrap_or(0.0),
            average_price: None, // Would need to calculate from fills
            created_at: DateTime::from_timestamp_millis(binance_order.transact_time as i64).unwrap_or(Utc::now()),
            updated_at: Utc::now(),
            time_in_force: TimeInForce::GTC, // Default
        })
    }
}

impl RateLimiter {
    /// Create new rate limiter
    pub fn new(requests_per_minute: u32, orders_per_second: u32, weight_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            orders_per_second,
            weight_per_minute,
            request_count: Arc::new(RwLock::new(0)),
            order_count: Arc::new(RwLock::new(0)),
            weight_count: Arc::new(RwLock::new(0)),
            last_reset: Arc::new(RwLock::new(Instant::now())),
        }
    }

    /// Check if API limits allow the request
    pub async fn check_limits(&self) -> Result<()> {
        let now = Instant::now();
        let mut last_reset = self.last_reset.write().await;
        
        // Reset counters if a minute has passed
        if now.duration_since(*last_reset) >= Duration::from_secs(60) {
            *self.request_count.write().await = 0;
            *self.weight_count.write().await = 0;
            *last_reset = now;
        }

        // Reset order counter every second
        if now.duration_since(*last_reset) >= Duration::from_secs(1) {
            *self.order_count.write().await = 0;
        }

        // Check limits
        let request_count = *self.request_count.read().await;
        let order_count = *self.order_count.read().await;
        let weight_count = *self.weight_count.read().await;

        if request_count >= self.requests_per_minute {
            return Err(TengriError::Strategy("Request rate limit exceeded".to_string()));
        }

        if order_count >= self.orders_per_second {
            return Err(TengriError::Strategy("Order rate limit exceeded".to_string()));
        }

        if weight_count >= self.weight_per_minute {
            return Err(TengriError::Strategy("Weight rate limit exceeded".to_string()));
        }

        // Increment counters
        *self.request_count.write().await += 1;
        *self.order_count.write().await += 1;
        *self.weight_count.write().await += 1; // Simplified weight calculation

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ExchangesConfig;

    #[tokio::test]
    async fn test_execution_engine_creation() {
        let config = ExchangesConfig::default();
        let engine = ExecutionEngine::new(config).await;
        assert!(engine.is_ok());
    }

    #[tokio::test]
    async fn test_order_creation() {
        let config = ExchangesConfig::default();
        let engine = ExecutionEngine::new(config).await.unwrap();
        
        let order = engine.create_market_order("BTCUSDT", OrderSide::Buy, 0.001).await;
        assert!(order.is_ok());
        
        let order = order.unwrap();
        assert_eq!(order.symbol, "BTCUSDT");
        assert_eq!(order.side, OrderSide::Buy);
        assert_eq!(order.quantity, 0.001);
        assert_eq!(order.order_type, OrderType::Market);
    }

    #[test]
    fn test_rate_limiter() {
        let rate_limiter = RateLimiter::new(100, 10, 1000);
        // Rate limiter tests would be more complex in practice
        assert_eq!(rate_limiter.requests_per_minute, 100);
        assert_eq!(rate_limiter.orders_per_second, 10);
        assert_eq!(rate_limiter.weight_per_minute, 1000);
    }
}