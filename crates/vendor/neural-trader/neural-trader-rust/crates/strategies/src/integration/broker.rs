//! Broker Integration for Trading Strategies
//!
//! Connects all 7 strategies to Agent 3's broker clients for real-time execution.
//! Handles order routing, position management, retries, and error recovery.

use crate::{Result, Signal, StrategyError, Direction};
use rust_decimal::Decimal;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};
use std::time::Duration;

// Re-export broker types for convenience
pub use nt_execution::broker::{
    BrokerClient, BrokerError, OrderFilter, HealthStatus, PositionSide, Position, Account
};
pub use nt_execution::{OrderRequest, OrderResponse, OrderType, TimeInForce, OrderSide};

/// Strategy executor that connects signals to broker
pub struct StrategyExecutor {
    broker: Arc<dyn BrokerClient>,
    /// Current positions cache
    positions: Arc<RwLock<Vec<Position>>>,
    /// Retry configuration
    max_retries: u32,
    retry_delay: Duration,
    /// Dry run mode (no actual orders)
    dry_run: bool,
}

impl StrategyExecutor {
    /// Create new strategy executor
    pub fn new(broker: Arc<dyn BrokerClient>) -> Self {
        Self {
            broker,
            positions: Arc::new(RwLock::new(Vec::new())),
            max_retries: 3,
            retry_delay: Duration::from_millis(500),
            dry_run: false,
        }
    }

    /// Enable dry run mode (simulation only)
    pub fn with_dry_run(mut self, dry_run: bool) -> Self {
        self.dry_run = dry_run;
        self
    }

    /// Set retry configuration
    pub fn with_retry(mut self, max_retries: u32, retry_delay: Duration) -> Self {
        self.max_retries = max_retries;
        self.retry_delay = retry_delay;
        self
    }

    /// Execute a trading signal
    pub async fn execute_signal(&self, signal: &Signal) -> Result<ExecutionResult> {
        info!(
            "Executing signal: {} {} {} @ ${:.2}",
            signal.strategy_id,
            signal.direction,
            signal.symbol,
            signal.entry_price.unwrap_or_default()
        );

        if self.dry_run {
            return Ok(ExecutionResult {
                order_id: format!("DRY_RUN_{}", uuid::Uuid::new_v4()),
                executed: false,
                filled_qty: 0,
                avg_fill_price: Decimal::ZERO,
                fees: Decimal::ZERO,
                message: "Dry run mode - no actual execution".to_string(),
            });
        }

        // Refresh positions
        self.refresh_positions().await?;

        // Check if we should close existing position
        if signal.direction == Direction::Close {
            return self.close_position(&signal.symbol).await;
        }

        // Get account info for position sizing
        let account = self.get_account_with_retry().await?;

        // Build order request
        let order_request = self.build_order_request(signal, &account).await?;

        // Execute with retry logic
        let response = self.place_order_with_retry(order_request).await?;

        Ok(ExecutionResult {
            order_id: response.order_id.clone(),
            executed: true,
            filled_qty: response.filled_qty,
            avg_fill_price: response.filled_avg_price.unwrap_or(Decimal::ZERO),
            fees: Decimal::ZERO, // Commission tracking is broker-specific
            message: format!("Order {} placed successfully", response.order_id),
        })
    }

    /// Build order request from signal
    async fn build_order_request(
        &self,
        signal: &Signal,
        account: &Account,
    ) -> Result<OrderRequest> {
        let side = match signal.direction {
            Direction::Long => OrderSide::Buy,
            Direction::Short => OrderSide::Sell,
            Direction::Close => {
                // Determine side based on current position
                let positions = self.positions.read().await;
                if let Some(pos) = positions.iter().find(|p| p.symbol.as_str() == signal.symbol) {
                    match pos.side {
                        PositionSide::Long => OrderSide::Sell,
                        PositionSide::Short => OrderSide::Buy,
                    }
                } else {
                    return Err(StrategyError::ExecutionError(
                        "No position to close".to_string(),
                    ));
                }
            }
        };

        // Calculate quantity (will be determined by risk management)
        let qty = signal.quantity.unwrap_or(1);

        // Convert string symbol to Symbol type
        let symbol = nt_core::types::Symbol::new(&signal.symbol)
            .map_err(|e| StrategyError::ExecutionError(format!("Invalid symbol: {}", e)))?;

        let order = OrderRequest {
            symbol,
            side,
            order_type: OrderType::Market,
            quantity: qty,
            limit_price: signal.entry_price,
            stop_price: signal.stop_loss,
            time_in_force: TimeInForce::Day,
        };

        debug!("Built order request: {:?}", order);
        Ok(order)
    }

    /// Place order with retry logic
    async fn place_order_with_retry(&self, order: OrderRequest) -> Result<OrderResponse> {
        let mut attempt = 0;

        loop {
            attempt += 1;

            match self.broker.place_order(order.clone()).await {
                Ok(response) => {
                    info!("Order placed successfully: {}", response.order_id);
                    return Ok(response);
                }
                Err(e) => {
                    if attempt >= self.max_retries {
                        error!("Order placement failed after {} attempts: {}", attempt, e);
                        return Err(StrategyError::ExecutionError(e.to_string()));
                    }

                    // Check if error is retryable
                    if !is_retryable_error(&e) {
                        return Err(StrategyError::ExecutionError(e.to_string()));
                    }

                    warn!(
                        "Order placement failed (attempt {}/{}), retrying: {}",
                        attempt, self.max_retries, e
                    );

                    tokio::time::sleep(self.retry_delay).await;
                }
            }
        }
    }

    /// Close position for symbol
    async fn close_position(&self, symbol: &str) -> Result<ExecutionResult> {
        let positions = self.positions.read().await;

        let position = positions
            .iter()
            .find(|p| p.symbol.as_str() == symbol)
            .ok_or_else(|| {
                StrategyError::ExecutionError(format!("No position found for {}", symbol))
            })?;

        let side = match position.side {
            PositionSide::Long => OrderSide::Sell,
            PositionSide::Short => OrderSide::Buy,
        };

        // Convert string to Symbol type
        let sym = nt_core::types::Symbol::new(symbol)
            .map_err(|e| StrategyError::ExecutionError(format!("Invalid symbol: {}", e)))?;

        let order = OrderRequest {
            symbol: sym,
            side,
            order_type: OrderType::Market,
            quantity: position.qty.unsigned_abs() as u32,
            limit_price: None,
            stop_price: None,
            time_in_force: TimeInForce::Day,
        };

        let response = self.place_order_with_retry(order).await?;

        Ok(ExecutionResult {
            order_id: response.order_id.clone(),
            executed: true,
            filled_qty: response.filled_qty,
            avg_fill_price: response.filled_avg_price.unwrap_or(Decimal::ZERO),
            fees: Decimal::ZERO, // Commission tracking is broker-specific
            message: format!("Position closed: {}", response.order_id),
        })
    }

    /// Get account with retry
    async fn get_account_with_retry(&self) -> Result<Account> {
        let mut attempt = 0;

        loop {
            attempt += 1;

            match self.broker.get_account().await {
                Ok(account) => return Ok(account),
                Err(e) => {
                    if attempt >= self.max_retries || !is_retryable_error(&e) {
                        return Err(StrategyError::ExecutionError(e.to_string()));
                    }

                    warn!("Get account failed (attempt {}/{}), retrying", attempt, self.max_retries);
                    tokio::time::sleep(self.retry_delay).await;
                }
            }
        }
    }

    /// Refresh positions cache
    pub async fn refresh_positions(&self) -> Result<()> {
        let positions = self.broker.get_positions().await.map_err(|e| {
            StrategyError::ExecutionError(format!("Failed to get positions: {}", e))
        })?;

        let mut cache = self.positions.write().await;
        *cache = positions;

        Ok(())
    }

    /// Get current position for symbol
    pub async fn get_position(&self, symbol: &str) -> Option<Position> {
        let positions = self.positions.read().await;
        positions.iter().find(|p| p.symbol.as_str() == symbol).cloned()
    }

    /// Check broker health
    pub async fn health_check(&self) -> Result<HealthStatus> {
        self.broker.health_check().await.map_err(|e| {
            StrategyError::ExecutionError(format!("Health check failed: {}", e))
        })
    }

    /// Cancel order by ID
    pub async fn cancel_order(&self, order_id: &str) -> Result<()> {
        self.broker.cancel_order(order_id).await.map_err(|e| {
            StrategyError::ExecutionError(format!("Cancel order failed: {}", e))
        })
    }
}

/// Check if broker error is retryable
fn is_retryable_error(error: &BrokerError) -> bool {
    matches!(
        error,
        BrokerError::Network(_) | BrokerError::RateLimit | BrokerError::Unavailable(_)
    )
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub order_id: String,
    pub executed: bool,
    pub filled_qty: u32,
    pub avg_fill_price: Decimal,
    pub fees: Decimal,
    pub message: String,
}

#[cfg(test)]
mod tests {
    use super::*;
    use nt_core::types::Symbol;

    // Mock broker for testing
    struct MockBroker {
        should_fail: bool,
    }

    #[async_trait]
    impl BrokerClient for MockBroker {
        async fn get_account(&self) -> std::result::Result<Account, BrokerError> {
            if self.should_fail {
                return Err(BrokerError::Network("Mock error".to_string()));
            }

            Ok(Account {
                account_id: "TEST123".to_string(),
                cash: Decimal::from(100000),
                portfolio_value: Decimal::from(100000),
                buying_power: Decimal::from(100000),
                equity: Decimal::from(100000),
                last_equity: Decimal::from(100000),
                multiplier: "1".to_string(),
                currency: "USD".to_string(),
                shorting_enabled: false,
                long_market_value: Decimal::ZERO,
                short_market_value: Decimal::ZERO,
                initial_margin: Decimal::ZERO,
                maintenance_margin: Decimal::ZERO,
                day_trading_buying_power: Decimal::from(100000),
                daytrade_count: 0,
            })
        }

        async fn get_positions(&self) -> std::result::Result<Vec<Position>, BrokerError> {
            Ok(vec![])
        }

        async fn place_order(&self, _order: OrderRequest) -> std::result::Result<OrderResponse, BrokerError> {
            if self.should_fail {
                return Err(BrokerError::Network("Mock error".to_string()));
            }

            Ok(OrderResponse {
                order_id: "ORDER123".to_string(),
                client_order_id: "CLIENT123".to_string(),
                symbol: Symbol("TEST".to_string()),
                side: OrderSide::Buy,
                qty: 10,
                filled_qty: Some(10),
                order_type: OrderType::Market,
                time_in_force: TimeInForce::Day,
                status: nt_execution::OrderStatus::Filled,
                filled_avg_price: Some(Decimal::from(100)),
                commission: Some(Decimal::from(1)),
                created_at: chrono::Utc::now(),
                updated_at: chrono::Utc::now(),
            })
        }

        async fn cancel_order(&self, _order_id: &str) -> std::result::Result<(), BrokerError> {
            Ok(())
        }

        async fn get_order(&self, _order_id: &str) -> std::result::Result<OrderResponse, BrokerError> {
            unimplemented!()
        }

        async fn list_orders(&self, _filter: OrderFilter) -> std::result::Result<Vec<OrderResponse>, BrokerError> {
            Ok(vec![])
        }

        async fn health_check(&self) -> std::result::Result<HealthStatus, BrokerError> {
            Ok(HealthStatus::Healthy)
        }
    }

    #[tokio::test]
    async fn test_execute_signal_dry_run() {
        let broker = Arc::new(MockBroker { should_fail: false });
        let executor = StrategyExecutor::new(broker).with_dry_run(true);

        let signal = Signal::new("test".to_string(), "AAPL".to_string(), Direction::Long);

        let result = executor.execute_signal(&signal).await.unwrap();
        assert!(!result.executed);
        assert!(result.order_id.starts_with("DRY_RUN_"));
    }

    #[tokio::test]
    async fn test_execute_signal_success() {
        let broker = Arc::new(MockBroker { should_fail: false });
        let executor = StrategyExecutor::new(broker);

        let signal = Signal::new("test".to_string(), "AAPL".to_string(), Direction::Long)
            .with_quantity(10);

        let result = executor.execute_signal(&signal).await.unwrap();
        assert!(result.executed);
        assert_eq!(result.order_id, "ORDER123");
        assert_eq!(result.filled_qty, 10);
    }

    #[tokio::test]
    async fn test_health_check() {
        let broker = Arc::new(MockBroker { should_fail: false });
        let executor = StrategyExecutor::new(broker);

        let status = executor.health_check().await.unwrap();
        assert_eq!(status, HealthStatus::Healthy);
    }
}
