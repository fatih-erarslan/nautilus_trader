//! Order execution implementation

use crate::prelude::*;
use crate::models::{MarketData, Order, OrderSide, OrderStatus, OrderType, Trade};
use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use num_traits::ToPrimitive;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

/// Order execution engine that handles trade execution
#[derive(Debug)]
pub struct OrderExecutor {
    /// Executor configuration
    config: OrderExecutorConfig,
    
    /// Pending orders queue
    pending_orders: Arc<Mutex<VecDeque<Order>>>,
    
    /// Active orders tracking
    active_orders: Arc<RwLock<HashMap<Uuid, Order>>>,
    
    /// Execution history
    execution_history: Arc<RwLock<VecDeque<ExecutionRecord>>>,
    
    /// Exchange connections
    exchange_connections: Arc<RwLock<HashMap<String, ExchangeConnection>>>,
    
    /// Execution metrics
    metrics: Arc<RwLock<ExecutionMetrics>>,
    
    /// Risk checks
    risk_manager: Arc<RwLock<ExecutionRiskManager>>,
}

#[derive(Debug, Clone)]
pub struct OrderExecutorConfig {
    /// Maximum orders per second
    pub max_orders_per_second: u32,
    
    /// Order timeout in seconds
    pub order_timeout_seconds: u32,
    
    /// Retry configuration
    pub retry_config: RetryConfig,
    
    /// Exchange configurations
    pub exchange_configs: HashMap<String, ExchangeConfig>,
    
    /// Execution algorithms
    pub execution_algorithms: Vec<ExecutionAlgorithm>,
    
    /// Risk limits
    pub risk_limits: RiskLimits,
    
    /// Latency optimization settings
    pub latency_config: LatencyConfig,
}

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub max_retries: u32,
    pub initial_delay_ms: u64,
    pub max_delay_ms: u64,
    pub backoff_multiplier: f64,
}

#[derive(Debug, Clone)]
pub struct ExchangeConfig {
    pub name: String,
    pub api_endpoint: String,
    pub api_key: Option<String>,
    pub secret_key: Option<String>,
    pub rate_limits: RateLimits,
    pub supported_order_types: Vec<OrderType>,
    pub min_order_size: Decimal,
    pub max_order_size: Decimal,
    pub fee_structure: FeeStructure,
}

#[derive(Debug, Clone)]
pub struct RateLimits {
    pub orders_per_second: u32,
    pub requests_per_minute: u32,
    pub weight_per_order: u32,
    pub max_weight_per_minute: u32,
}

#[derive(Debug, Clone)]
pub struct FeeStructure {
    pub maker_fee_bps: u32,  // Basis points (0.1% = 10 bps)
    pub taker_fee_bps: u32,
    pub withdrawal_fees: HashMap<String, Decimal>,
}

#[derive(Debug, Clone)]
pub enum ExecutionAlgorithm {
    Market,           // Immediate execution at market price
    TWAP,            // Time-Weighted Average Price
    VWAP,            // Volume-Weighted Average Price
    Implementation,  // Implementation Shortfall
    POV,            // Percentage of Volume
    Iceberg,        // Large order splitting
    Sniper,         // Aggressive execution
    Stealth,        // Low-impact execution
}

#[derive(Debug, Clone)]
pub struct RiskLimits {
    pub max_order_value: Decimal,
    pub max_daily_volume: Decimal,
    pub max_position_size: Decimal,
    pub max_drawdown_pct: f64,
    pub max_orders_per_symbol: u32,
    pub forbidden_symbols: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct LatencyConfig {
    pub enable_direct_market_access: bool,
    pub use_colocation: bool,
    pub prefer_fast_exchanges: bool,
    pub max_acceptable_latency_ms: u32,
    pub latency_monitoring: bool,
}

#[derive(Debug)]
struct ExchangeConnection {
    config: ExchangeConfig,
    last_request_time: DateTime<Utc>,
    rate_limit_tokens: u32,
    connection_status: ConnectionStatus,
    latency_stats: LatencyStats,
}

#[derive(Debug, Clone)]
enum ConnectionStatus {
    Connected,
    Connecting,
    Disconnected,
    Error(String),
}

#[derive(Debug, Clone)]
struct LatencyStats {
    average_latency_ms: f64,
    min_latency_ms: u32,
    max_latency_ms: u32,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
struct ExecutionRecord {
    order_id: Uuid,
    execution_time: DateTime<Utc>,
    algorithm_used: ExecutionAlgorithm,
    exchange: String,
    execution_price: Decimal,
    execution_quantity: Decimal,
    slippage: f64,
    latency_ms: u32,
    fees_paid: Decimal,
    execution_status: ExecutionStatus,
}

#[derive(Debug, Clone)]
enum ExecutionStatus {
    Filled,
    PartiallyFilled,
    Rejected,
    Canceled,
    Expired,
}

#[derive(Debug, Clone, Default)]
struct ExecutionMetrics {
    total_orders_executed: u64,
    total_volume_executed: Decimal,
    average_execution_time_ms: f64,
    average_slippage_bps: f64,
    fill_rate: f64,
    rejection_rate: f64,
    total_fees_paid: Decimal,
    successful_executions: u64,
    failed_executions: u64,
}

#[derive(Debug)]
struct ExecutionRiskManager {
    daily_volume: Decimal,
    daily_order_count: u32,
    position_sizes: HashMap<String, Decimal>,
    last_reset: DateTime<Utc>,
}

impl Default for OrderExecutorConfig {
    fn default() -> Self {
        Self {
            max_orders_per_second: 10,
            order_timeout_seconds: 30,
            retry_config: RetryConfig {
                max_retries: 3,
                initial_delay_ms: 100,
                max_delay_ms: 5000,
                backoff_multiplier: 2.0,
            },
            exchange_configs: HashMap::new(),
            execution_algorithms: vec![
                ExecutionAlgorithm::Market,
                ExecutionAlgorithm::TWAP,
                ExecutionAlgorithm::VWAP,
            ],
            risk_limits: RiskLimits {
                max_order_value: Decimal::from(1000000), // $1M
                max_daily_volume: Decimal::from(10000000), // $10M
                max_position_size: Decimal::from(5000000), // $5M
                max_drawdown_pct: 0.05, // 5%
                max_orders_per_symbol: 100,
                forbidden_symbols: vec![],
            },
            latency_config: LatencyConfig {
                enable_direct_market_access: true,
                use_colocation: false,
                prefer_fast_exchanges: true,
                max_acceptable_latency_ms: 50,
                latency_monitoring: true,
            },
        }
    }
}

impl OrderExecutor {
    /// Create a new order executor
    pub async fn new(config: OrderExecutorConfig) -> Result<Self> {
        let risk_manager = ExecutionRiskManager {
            daily_volume: Decimal::ZERO,
            daily_order_count: 0,
            position_sizes: HashMap::new(),
            last_reset: Utc::now(),
        };

        let executor = Self {
            config,
            pending_orders: Arc::new(Mutex::new(VecDeque::new())),
            active_orders: Arc::new(RwLock::new(HashMap::new())),
            execution_history: Arc::new(RwLock::new(VecDeque::new())),
            exchange_connections: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(ExecutionMetrics::default())),
            risk_manager: Arc::new(RwLock::new(risk_manager)),
        };

        // Initialize exchange connections
        executor.initialize_exchange_connections().await?;

        // Start background processing
        executor.start_order_processing().await?;

        Ok(executor)
    }

    /// Submit an order for execution
    pub async fn submit_order(&self, mut order: Order) -> Result<Uuid> {
        // Perform pre-execution risk checks
        self.perform_risk_checks(&order).await?;

        // Set order to submitted status
        order.status = OrderStatus::Submitted;
        order.updated_at = Utc::now();

        let order_id = order.id;

        // Add to active orders
        {
            let mut active_orders = self.active_orders.write().await;
            active_orders.insert(order_id, order.clone());
        }

        // Add to pending queue
        {
            let mut pending_orders = self.pending_orders.lock().await;
            pending_orders.push_back(order);
        }

        info!("Order {} submitted for execution", order_id);
        Ok(order_id)
    }

    /// Cancel an order
    pub async fn cancel_order(&self, order_id: Uuid) -> Result<()> {
        let mut active_orders = self.active_orders.write().await;
        
        if let Some(mut order) = active_orders.get_mut(&order_id) {
            if matches!(order.status, OrderStatus::Submitted | OrderStatus::PartiallyFilled) {
                order.status = OrderStatus::Canceled;
                order.updated_at = Utc::now();
                
                // Send cancel request to exchange
                self.send_cancel_request(&order).await?;
                
                info!("Order {} canceled", order_id);
                Ok(())
            } else {
                Err(Error::Execution(format!("Cannot cancel order {} in status {:?}", order_id, order.status)))
            }
        } else {
            Err(Error::Execution(format!("Order {} not found", order_id)))
        }
    }

    /// Get order status
    pub async fn get_order_status(&self, order_id: Uuid) -> Result<Option<Order>> {
        let active_orders = self.active_orders.read().await;
        Ok(active_orders.get(&order_id).cloned())
    }

    /// Get execution metrics
    pub async fn get_metrics(&self) -> ExecutionMetrics {
        self.metrics.read().await.clone()
    }

    /// Get execution history
    pub async fn get_execution_history(&self, limit: Option<usize>) -> Result<Vec<ExecutionRecord>> {
        let history = self.execution_history.read().await;
        let limit = limit.unwrap_or(100);
        
        Ok(history.iter()
            .rev()
            .take(limit)
            .cloned()
            .collect())
    }

    /// Execute order with specific algorithm
    pub async fn execute_with_algorithm(
        &self,
        order: &Order,
        algorithm: &ExecutionAlgorithm,
        market_data: &MarketData,
    ) -> Result<Vec<Trade>> {
        let start_time = std::time::Instant::now();
        
        match algorithm {
            ExecutionAlgorithm::Market => self.execute_market_order(order, market_data).await,
            ExecutionAlgorithm::TWAP => self.execute_twap_order(order, market_data).await,
            ExecutionAlgorithm::VWAP => self.execute_vwap_order(order, market_data).await,
            ExecutionAlgorithm::Iceberg => self.execute_iceberg_order(order, market_data).await,
            ExecutionAlgorithm::POV => self.execute_pov_order(order, market_data).await,
            _ => {
                warn!("Algorithm {:?} not implemented, using market execution", algorithm);
                self.execute_market_order(order, market_data).await
            }
        }
    }

    /// Get best execution venue for an order
    pub async fn get_best_venue(&self, order: &Order) -> Result<String> {
        let connections = self.exchange_connections.read().await;
        
        let mut best_venue = None;
        let mut best_score = f64::NEG_INFINITY;

        for (exchange_name, connection) in connections.iter() {
            if matches!(connection.connection_status, ConnectionStatus::Connected) {
                // Score based on latency, fees, and liquidity
                let latency_score = 100.0 - connection.latency_stats.average_latency_ms;
                let fee_score = 100.0 - connection.config.fee_structure.taker_fee_bps as f64;
                let liquidity_score = 80.0; // Would be calculated from real data
                
                let total_score = (latency_score + fee_score + liquidity_score) / 3.0;
                
                if total_score > best_score {
                    best_score = total_score;
                    best_venue = Some(exchange_name.clone());
                }
            }
        }

        best_venue.ok_or_else(|| Error::Execution("No available venues".to_string()))
    }

    async fn initialize_exchange_connections(&self) -> Result<()> {
        let mut connections = self.exchange_connections.write().await;

        for (exchange_name, exchange_config) in &self.config.exchange_configs {
            let connection = ExchangeConnection {
                config: exchange_config.clone(),
                last_request_time: Utc::now(),
                rate_limit_tokens: exchange_config.rate_limits.orders_per_second,
                connection_status: ConnectionStatus::Connecting,
                latency_stats: LatencyStats {
                    average_latency_ms: 10.0,
                    min_latency_ms: 5,
                    max_latency_ms: 50,
                    last_updated: Utc::now(),
                },
            };

            connections.insert(exchange_name.clone(), connection);
        }

        // Simulate connection establishment
        for connection in connections.values_mut() {
            connection.connection_status = ConnectionStatus::Connected;
        }

        Ok(())
    }

    async fn start_order_processing(&self) -> Result<()> {
        let pending_orders = Arc::clone(&self.pending_orders);
        let active_orders = Arc::clone(&self.active_orders);
        let execution_history = Arc::clone(&self.execution_history);
        let metrics = Arc::clone(&self.metrics);
        let config = self.config.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_millis(100));
            
            loop {
                interval.tick().await;
                
                // Process pending orders
                let mut pending = pending_orders.lock().await;
                if let Some(order) = pending.pop_front() {
                    drop(pending); // Release lock
                    
                    // Process the order (simplified)
                    match Self::process_order_static(order, &active_orders, &execution_history, &metrics, &config).await {
                        Ok(_) => {
                            trace!("Order processed successfully");
                        }
                        Err(e) => {
                            error!("Failed to process order: {}", e);
                        }
                    }
                }
            }
        });

        Ok(())
    }

    async fn process_order_static(
        mut order: Order,
        active_orders: &Arc<RwLock<HashMap<Uuid, Order>>>,
        execution_history: &Arc<RwLock<VecDeque<ExecutionRecord>>>,
        metrics: &Arc<RwLock<ExecutionMetrics>>,
        config: &OrderExecutorConfig,
    ) -> Result<()> {
        // Simulate order execution
        let execution_start = std::time::Instant::now();
        
        // Simulate execution delay
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        
        // Update order status
        order.status = OrderStatus::Filled;
        order.updated_at = Utc::now();

        // Create execution record
        let execution_record = ExecutionRecord {
            order_id: order.id,
            execution_time: Utc::now(),
            algorithm_used: ExecutionAlgorithm::Market,
            exchange: "simulated".to_string(),
            execution_price: order.price.unwrap_or(Decimal::from(50000)),
            execution_quantity: order.quantity,
            slippage: 0.05, // 5 bps
            latency_ms: execution_start.elapsed().as_millis() as u32,
            fees_paid: order.quantity * Decimal::from_f64_retain(0.001).unwrap_or_default(), // 0.1% fee
            execution_status: ExecutionStatus::Filled,
        };

        // Update active orders
        {
            let mut active = active_orders.write().await;
            active.insert(order.id, order);
        }

        // Add to execution history
        {
            let mut history = execution_history.write().await;
            history.push_back(execution_record);
            
            // Keep history size manageable
            while history.len() > 10000 {
                history.pop_front();
            }
        }

        // Update metrics
        {
            let mut m = metrics.write().await;
            m.total_orders_executed += 1;
            m.successful_executions += 1;
            m.fill_rate = m.successful_executions as f64 / m.total_orders_executed as f64;
        }

        Ok(())
    }

    async fn perform_risk_checks(&self, order: &Order) -> Result<()> {
        let mut risk_manager = self.risk_manager.write().await;
        
        // Reset daily counters if needed
        let now = Utc::now();
        if (now - risk_manager.last_reset).num_hours() >= 24 {
            risk_manager.daily_volume = Decimal::ZERO;
            risk_manager.daily_order_count = 0;
            risk_manager.last_reset = now;
        }

        // Check order value limit
        let order_value = order.quantity * order.price.unwrap_or(Decimal::from(50000));
        if order_value > self.config.risk_limits.max_order_value {
            return Err(Error::Risk(format!("Order value {} exceeds limit {}", 
                order_value, self.config.risk_limits.max_order_value)));
        }

        // Check daily volume limit
        if risk_manager.daily_volume + order_value > self.config.risk_limits.max_daily_volume {
            return Err(Error::Risk("Daily volume limit exceeded".to_string()));
        }

        // Check forbidden symbols
        if self.config.risk_limits.forbidden_symbols.contains(&order.symbol) {
            return Err(Error::Risk(format!("Symbol {} is forbidden", order.symbol)));
        }

        // Update risk tracking
        risk_manager.daily_volume += order_value;
        risk_manager.daily_order_count += 1;

        Ok(())
    }

    async fn send_cancel_request(&self, order: &Order) -> Result<()> {
        // Simulate sending cancel request to exchange
        info!("Sending cancel request for order {} to exchange", order.id);
        
        // In real implementation, this would send actual API request
        tokio::time::sleep(tokio::time::Duration::from_millis(5)).await;
        
        Ok(())
    }

    async fn execute_market_order(&self, order: &Order, market_data: &MarketData) -> Result<Vec<Trade>> {
        let execution_price = match order.side {
            OrderSide::Buy => market_data.ask,
            OrderSide::Sell => market_data.bid,
        };

        let trade = Trade {
            id: Uuid::new_v4(),
            order_id: order.id,
            symbol: order.symbol.clone(),
            side: order.side.clone(),
            price: execution_price,
            quantity: order.quantity,
            fee: order.quantity * execution_price * Decimal::from_f64_retain(0.001).unwrap_or_default(),
            fee_currency: "USD".to_string(),
            timestamp: Utc::now(),
        };

        Ok(vec![trade])
    }

    async fn execute_twap_order(&self, order: &Order, market_data: &MarketData) -> Result<Vec<Trade>> {
        // Time-Weighted Average Price execution
        // Split order into smaller pieces over time
        
        let slice_count = 5; // Split into 5 pieces
        let slice_size = order.quantity / Decimal::from(slice_count);
        let mut trades = Vec::new();

        for i in 0..slice_count {
            // Simulate time delay between slices
            if i > 0 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }

            let execution_price = match order.side {
                OrderSide::Buy => market_data.ask + Decimal::from_f64_retain(i as f64 * 0.01).unwrap_or_default(),
                OrderSide::Sell => market_data.bid - Decimal::from_f64_retain(i as f64 * 0.01).unwrap_or_default(),
            };

            let trade = Trade {
                id: Uuid::new_v4(),
                order_id: order.id,
                symbol: order.symbol.clone(),
                side: order.side.clone(),
                price: execution_price,
                quantity: slice_size,
                fee: slice_size * execution_price * Decimal::from_f64_retain(0.001).unwrap_or_default(),
                fee_currency: "USD".to_string(),
                timestamp: Utc::now(),
            };

            trades.push(trade);
        }

        Ok(trades)
    }

    async fn execute_vwap_order(&self, order: &Order, market_data: &MarketData) -> Result<Vec<Trade>> {
        // Volume-Weighted Average Price execution
        // Execute based on historical volume patterns
        
        let volume_profile = vec![0.3, 0.4, 0.2, 0.1]; // Simulate volume distribution
        let mut trades = Vec::new();

        for (i, &volume_pct) in volume_profile.iter().enumerate() {
            let slice_quantity = order.quantity * Decimal::from_f64_retain(volume_pct).unwrap_or_default();
            
            if slice_quantity > Decimal::ZERO {
                let price_adjustment = Decimal::from_f64_retain((i as f64 - 1.5) * 0.005).unwrap_or_default();
                let execution_price = match order.side {
                    OrderSide::Buy => market_data.ask + price_adjustment,
                    OrderSide::Sell => market_data.bid + price_adjustment,
                };

                let trade = Trade {
                    id: Uuid::new_v4(),
                    order_id: order.id,
                    symbol: order.symbol.clone(),
                    side: order.side.clone(),
                    price: execution_price,
                    quantity: slice_quantity,
                    fee: slice_quantity * execution_price * Decimal::from_f64_retain(0.001).unwrap_or_default(),
                    fee_currency: "USD".to_string(),
                    timestamp: Utc::now(),
                };

                trades.push(trade);
            }

            // Simulate time delay
            tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        }

        Ok(trades)
    }

    async fn execute_iceberg_order(&self, order: &Order, market_data: &MarketData) -> Result<Vec<Trade>> {
        // Iceberg execution - show only small amounts at a time
        let visible_quantity = order.quantity / Decimal::from(10); // Show 10% at a time
        let mut remaining_quantity = order.quantity;
        let mut trades = Vec::new();

        while remaining_quantity > Decimal::ZERO {
            let current_slice = visible_quantity.min(remaining_quantity);
            
            let execution_price = match order.side {
                OrderSide::Buy => market_data.ask,
                OrderSide::Sell => market_data.bid,
            };

            let trade = Trade {
                id: Uuid::new_v4(),
                order_id: order.id,
                symbol: order.symbol.clone(),
                side: order.side.clone(),
                price: execution_price,
                quantity: current_slice,
                fee: current_slice * execution_price * Decimal::from_f64_retain(0.001).unwrap_or_default(),
                fee_currency: "USD".to_string(),
                timestamp: Utc::now(),
            };

            trades.push(trade);
            remaining_quantity -= current_slice;

            // Wait between iceberg slices
            if remaining_quantity > Decimal::ZERO {
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }
        }

        Ok(trades)
    }

    async fn execute_pov_order(&self, order: &Order, market_data: &MarketData) -> Result<Vec<Trade>> {
        // Percentage of Volume execution
        let target_pov = 0.1; // Target 10% of market volume
        let estimated_market_volume = market_data.volume_24h / Decimal::from(24 * 60); // Per minute estimate
        let target_quantity_per_minute = estimated_market_volume * Decimal::from_f64_retain(target_pov).unwrap_or_default();

        let execution_minutes = (order.quantity / target_quantity_per_minute).to_f64().unwrap_or(1.0).ceil() as usize;
        let quantity_per_slice = order.quantity / Decimal::from(execution_minutes);
        
        let mut trades = Vec::new();

        for i in 0..execution_minutes {
            let execution_price = match order.side {
                OrderSide::Buy => market_data.ask + Decimal::from_f64_retain(i as f64 * 0.002).unwrap_or_default(),
                OrderSide::Sell => market_data.bid - Decimal::from_f64_retain(i as f64 * 0.002).unwrap_or_default(),
            };

            let trade = Trade {
                id: Uuid::new_v4(),
                order_id: order.id,
                symbol: order.symbol.clone(),
                side: order.side.clone(),
                price: execution_price,
                quantity: quantity_per_slice,
                fee: quantity_per_slice * execution_price * Decimal::from_f64_retain(0.001).unwrap_or_default(),
                fee_currency: "USD".to_string(),
                timestamp: Utc::now(),
            };

            trades.push(trade);

            // Simulate minute-level execution
            if i < execution_minutes - 1 {
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await; // Shortened for testing
            }
        }

        Ok(trades)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal_macros::dec;

    #[tokio::test]
    async fn test_order_executor_creation() {
        let config = OrderExecutorConfig::default();
        let executor = OrderExecutor::new(config).await;
        
        assert!(executor.is_ok());
    }

    #[tokio::test]
    async fn test_order_submission() {
        let config = OrderExecutorConfig::default();
        let executor = OrderExecutor::new(config).await.unwrap();
        
        let order = Order {
            id: Uuid::new_v4(),
            symbol: "BTC/USD".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: dec!(1.0),
            price: Some(dec!(50000)),
            time_in_force: crate::models::TimeInForce::IOC,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let result = executor.submit_order(order).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_market_execution() {
        let config = OrderExecutorConfig::default();
        let executor = OrderExecutor::new(config).await.unwrap();
        
        let order = Order {
            id: Uuid::new_v4(),
            symbol: "BTC/USD".to_string(),
            side: OrderSide::Buy,
            order_type: OrderType::Market,
            quantity: dec!(1.0),
            price: Some(dec!(50000)),
            time_in_force: crate::models::TimeInForce::IOC,
            status: OrderStatus::Pending,
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };

        let market_data = MarketData {
            symbol: "BTC/USD".to_string(),
            timestamp: Utc::now(),
            bid: dec!(49999),
            ask: dec!(50001),
            mid: dec!(50000),
            last: dec!(50000),
            volume_24h: dec!(1000),
            bid_size: dec!(10),
            ask_size: dec!(10),
        };

        let result = executor.execute_market_order(&order, &market_data).await;
        assert!(result.is_ok());
        
        let trades = result.unwrap();
        assert_eq!(trades.len(), 1);
        assert_eq!(trades[0].price, dec!(50001)); // Should execute at ask price for buy order
    }
}