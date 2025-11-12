# Live Trading Coordinator Architecture

## Executive Summary

The `hyperphysics-trading` crate orchestrates live trading execution with real-time risk monitoring, smart order routing, and emergency stop-loss mechanisms based on thermodynamic entropy monitoring.

## 1. Module Structure

```
hyperphysics-trading/
├── src/
│   ├── coordinator/
│   │   ├── mod.rs              # Main orchestrator
│   │   ├── queen.rs            # Queen agent coordination
│   │   └── lifecycle.rs        # Trading session lifecycle
│   ├── execution/
│   │   ├── mod.rs
│   │   ├── router.rs           # Smart order routing
│   │   ├── vwap.rs             # VWAP execution
│   │   ├── twap.rs             # TWAP execution
│   │   └── iceberg.rs          # Iceberg orders
│   ├── risk/
│   │   ├── mod.rs
│   │   ├── pre_trade.rs        # Pre-trade risk checks
│   │   ├── real_time.rs        # Real-time monitoring
│   │   └── circuit_breaker.rs  # Emergency stops
│   ├── feeds/
│   │   ├── mod.rs
│   │   ├── aggregator.rs       # Multi-source aggregation
│   │   └── validator.rs        # Data quality checks
│   ├── reporting/
│   │   ├── mod.rs
│   │   ├── fills.rs            # Fill reporting
│   │   └── pnl.rs              # P&L tracking
│   └── lib.rs
```

## 2. Core Architecture

### 2.1 Trading Orchestrator

```rust
use std::sync::Arc;
use tokio::sync::{mpsc, RwLock, Mutex};
use async_trait::async_trait;

/// Main trading coordinator
pub struct TradingOrchestrator {
    /// Market data feed
    market_feed: Arc<dyn MarketDataProvider>,

    /// Risk monitor
    risk_monitor: Arc<RwLock<ThermodynamicRiskMonitor>>,

    /// Order router
    order_router: Arc<Mutex<OrderRouter>>,

    /// Portfolio state
    portfolio: Arc<RwLock<Portfolio>>,

    /// Strategy
    strategy: Arc<dyn Strategy>,

    /// Emergency stop flag
    emergency_stop: Arc<AtomicBool>,

    /// Event channels
    signal_tx: mpsc::Sender<SignalEvent>,
    signal_rx: mpsc::Receiver<SignalEvent>,

    fill_tx: mpsc::Sender<FillEvent>,
    fill_rx: mpsc::Receiver<FillEvent>,
}

impl TradingOrchestrator {
    pub async fn new(
        market_feed: Arc<dyn MarketDataProvider>,
        strategy: Arc<dyn Strategy>,
        initial_capital: f64,
    ) -> Result<Self, TradingError> {
        let portfolio = Arc::new(RwLock::new(Portfolio::new(initial_capital)));
        let risk_monitor = Arc::new(RwLock::new(ThermodynamicRiskMonitor::new()));
        let order_router = Arc::new(Mutex::new(OrderRouter::new()));
        let emergency_stop = Arc::new(AtomicBool::new(false));

        let (signal_tx, signal_rx) = mpsc::channel(1000);
        let (fill_tx, fill_rx) = mpsc::channel(1000);

        Ok(Self {
            market_feed,
            risk_monitor,
            order_router,
            portfolio,
            strategy,
            emergency_stop,
            signal_tx,
            signal_rx,
            fill_tx,
            fill_rx,
        })
    }

    /// Start trading session
    pub async fn start(&mut self) -> Result<(), TradingError> {
        println!("Starting trading session...");

        // Spawn market data handler
        self.spawn_market_data_handler().await?;

        // Spawn signal processor
        self.spawn_signal_processor().await?;

        // Spawn fill processor
        self.spawn_fill_processor().await?;

        // Spawn risk monitor
        self.spawn_risk_monitor().await?;

        println!("Trading session started successfully");

        Ok(())
    }

    /// Stop trading session
    pub async fn stop(&mut self) -> Result<(), TradingError> {
        println!("Stopping trading session...");

        // Set emergency stop flag
        self.emergency_stop.store(true, Ordering::SeqCst);

        // Cancel all open orders
        self.cancel_all_orders().await?;

        // Flatten all positions (optional)
        // self.flatten_positions().await?;

        println!("Trading session stopped");

        Ok(())
    }

    async fn spawn_market_data_handler(&self) -> Result<(), TradingError> {
        let market_feed = self.market_feed.clone();
        let strategy = self.strategy.clone();
        let portfolio = self.portfolio.clone();
        let signal_tx = self.signal_tx.clone();
        let emergency_stop = self.emergency_stop.clone();

        tokio::spawn(async move {
            let mut tick_stream = market_feed.subscribe_ticks("*").await.unwrap();

            while let Some(tick) = tick_stream.recv().await {
                if emergency_stop.load(Ordering::SeqCst) {
                    break;
                }

                // Update portfolio prices
                {
                    let mut port = portfolio.write().await;
                    port.update_price(&tick.symbol, tick.price);
                }

                // Generate signals
                let port = portfolio.read().await;
                if let Ok(Some(signal)) = strategy.on_tick(&tick, &port).await {
                    let _ = signal_tx.send(signal).await;
                }
            }
        });

        Ok(())
    }

    async fn spawn_signal_processor(&self) -> Result<(), TradingError> {
        let mut signal_rx = self.signal_rx;
        let risk_monitor = self.risk_monitor.clone();
        let order_router = self.order_router.clone();
        let portfolio = self.portfolio.clone();
        let strategy = self.strategy.clone();
        let emergency_stop = self.emergency_stop.clone();

        tokio::spawn(async move {
            while let Some(signal) = signal_rx.recv().await {
                if emergency_stop.load(Ordering::SeqCst) {
                    break;
                }

                // Convert signal to order
                let port = portfolio.read().await;
                let order = match strategy.signal_to_order(signal, &port) {
                    Ok(order) => order,
                    Err(e) => {
                        eprintln!("Failed to convert signal to order: {}", e);
                        continue;
                    }
                };

                // Pre-trade risk check
                let risk = risk_monitor.read().await;
                if !risk.check_pre_trade(&order, &port) {
                    eprintln!("Order rejected by pre-trade risk check");
                    continue;
                }

                // Submit order
                let mut router = order_router.lock().await;
                if let Err(e) = router.submit_order(order).await {
                    eprintln!("Failed to submit order: {}", e);
                }
            }
        });

        Ok(())
    }

    async fn spawn_fill_processor(&self) -> Result<(), TradingError> {
        let mut fill_rx = self.fill_rx;
        let portfolio = self.portfolio.clone();

        tokio::spawn(async move {
            while let Some(fill) = fill_rx.recv().await {
                // Update portfolio
                let mut port = portfolio.write().await;
                port.apply_fill(fill.clone());

                println!("Fill: {} @ {} (qty: {})",
                    fill.symbol, fill.fill_price, fill.quantity);
            }
        });

        Ok(())
    }

    async fn spawn_risk_monitor(&self) -> Result<(), TradingError> {
        let risk_monitor = self.risk_monitor.clone();
        let portfolio = self.portfolio.clone();
        let emergency_stop = self.emergency_stop.clone();

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(1));

            loop {
                interval.tick().await;

                if emergency_stop.load(Ordering::SeqCst) {
                    break;
                }

                // Check real-time risk
                let port = portfolio.read().await;
                let mut risk = risk_monitor.write().await;

                if let Err(violation) = risk.check_real_time(&port) {
                    eprintln!("RISK VIOLATION: {:?}", violation);

                    // Trigger emergency stop if critical
                    if violation.is_critical() {
                        emergency_stop.store(true, Ordering::SeqCst);
                        eprintln!("EMERGENCY STOP TRIGGERED");
                    }
                }
            }
        });

        Ok(())
    }

    async fn cancel_all_orders(&self) -> Result<(), TradingError> {
        let mut router = self.order_router.lock().await;
        router.cancel_all().await
    }
}
```

### 2.2 Smart Order Routing

```rust
use std::collections::HashMap;

pub struct OrderRouter {
    /// Broker connections
    brokers: HashMap<String, Arc<dyn BrokerAdapter>>,

    /// Routing rules
    routing_rules: RoutingRules,

    /// Active orders
    active_orders: HashMap<String, Order>,
}

impl OrderRouter {
    pub async fn submit_order(&mut self, order: OrderEvent) -> Result<String, TradingError> {
        // Select best broker
        let broker_id = self.routing_rules.select_broker(&order)?;
        let broker = self.brokers.get(&broker_id)
            .ok_or(TradingError::BrokerNotFound(broker_id.clone()))?;

        // Submit to broker
        let order_id = broker.submit_order(order.clone()).await?;

        // Track active order
        self.active_orders.insert(order_id.clone(), Order {
            id: order_id.clone(),
            event: order,
            status: OrderStatus::Pending,
            filled_quantity: 0.0,
        });

        Ok(order_id)
    }

    pub async fn cancel_order(&mut self, order_id: &str) -> Result<(), TradingError> {
        let order = self.active_orders.get(order_id)
            .ok_or(TradingError::OrderNotFound(order_id.to_string()))?;

        // Find broker
        let broker_id = self.routing_rules.select_broker(&order.event)?;
        let broker = self.brokers.get(&broker_id).unwrap();

        broker.cancel_order(order_id).await?;

        // Update status
        if let Some(order) = self.active_orders.get_mut(order_id) {
            order.status = OrderStatus::Cancelled;
        }

        Ok(())
    }

    pub async fn cancel_all(&mut self) -> Result<(), TradingError> {
        let order_ids: Vec<String> = self.active_orders
            .iter()
            .filter(|(_, order)| order.status == OrderStatus::Pending)
            .map(|(id, _)| id.clone())
            .collect();

        for order_id in order_ids {
            let _ = self.cancel_order(&order_id).await;
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
struct Order {
    id: String,
    event: OrderEvent,
    status: OrderStatus,
    filled_quantity: f64,
}

#[derive(Debug, Clone, PartialEq)]
enum OrderStatus {
    Pending,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}

/// Broker adapter trait
#[async_trait]
pub trait BrokerAdapter: Send + Sync {
    async fn submit_order(&self, order: OrderEvent) -> Result<String, TradingError>;
    async fn cancel_order(&self, order_id: &str) -> Result<(), TradingError>;
    async fn get_order_status(&self, order_id: &str) -> Result<OrderStatus, TradingError>;
}
```

### 2.3 VWAP Execution

```rust
use chrono::{DateTime, Utc};

/// Volume-Weighted Average Price execution algorithm
pub struct VWAPExecutor {
    /// Total quantity to execute
    total_quantity: f64,

    /// Start time
    start_time: DateTime<Utc>,

    /// End time
    end_time: DateTime<Utc>,

    /// Remaining quantity
    remaining_quantity: f64,

    /// Executed slices
    executed_slices: Vec<OrderSlice>,
}

#[derive(Debug, Clone)]
struct OrderSlice {
    timestamp: DateTime<Utc>,
    quantity: f64,
    price: f64,
}

impl VWAPExecutor {
    pub fn new(
        total_quantity: f64,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Self {
        Self {
            total_quantity,
            start_time,
            end_time,
            remaining_quantity: total_quantity,
            executed_slices: Vec::new(),
        }
    }

    /// Calculate next slice based on historical volume profile
    pub fn calculate_slice(
        &self,
        current_time: DateTime<Utc>,
        historical_volume: &[f64],
    ) -> Option<f64> {
        if current_time >= self.end_time || self.remaining_quantity <= 0.0 {
            return None;
        }

        // Calculate time progress
        let total_duration = (self.end_time - self.start_time).num_seconds() as f64;
        let elapsed = (current_time - self.start_time).num_seconds() as f64;
        let progress = elapsed / total_duration;

        // Expected volume share based on historical profile
        let current_hour = current_time.hour() as usize;
        let hourly_volume = historical_volume[current_hour];
        let total_volume: f64 = historical_volume.iter().sum();
        let volume_share = hourly_volume / total_volume;

        // Slice size = remaining * volume_share / remaining_time_share
        let remaining_time_share = 1.0 - progress;
        let slice_size = self.remaining_quantity * volume_share / remaining_time_share;

        Some(slice_size.min(self.remaining_quantity))
    }

    pub fn record_execution(&mut self, quantity: f64, price: f64) {
        self.remaining_quantity -= quantity;
        self.executed_slices.push(OrderSlice {
            timestamp: Utc::now(),
            quantity,
            price,
        });
    }

    pub fn average_price(&self) -> f64 {
        let total_value: f64 = self.executed_slices.iter()
            .map(|slice| slice.quantity * slice.price)
            .sum();

        let total_quantity: f64 = self.executed_slices.iter()
            .map(|slice| slice.quantity)
            .sum();

        if total_quantity > 0.0 {
            total_value / total_quantity
        } else {
            0.0
        }
    }
}
```

## 3. Thermodynamic Risk Monitoring

### 3.1 Real-Time Entropy Monitoring

```rust
use std::sync::atomic::{AtomicBool, Ordering};

pub struct ThermodynamicRiskMonitor {
    /// Maximum allowed entropy
    max_entropy: f64,

    /// Entropy threshold for warning
    warning_threshold: f64,

    /// Entropy threshold for critical
    critical_threshold: f64,

    /// Maximum drawdown allowed
    max_drawdown: f64,

    /// Position limits per symbol
    position_limits: HashMap<String, f64>,

    /// Portfolio-level limits
    max_leverage: f64,
    max_concentration: f64,
}

impl ThermodynamicRiskMonitor {
    pub fn new() -> Self {
        Self {
            max_entropy: 2.0,
            warning_threshold: 1.5,
            critical_threshold: 1.8,
            max_drawdown: 0.20,  // 20%
            position_limits: HashMap::new(),
            max_leverage: 2.0,
            max_concentration: 0.25,  // 25% per position
        }
    }

    /// Pre-trade risk check
    pub fn check_pre_trade(
        &self,
        order: &OrderEvent,
        portfolio: &Portfolio,
    ) -> bool {
        // Check position limits
        if let Some(&limit) = self.position_limits.get(&order.symbol) {
            let current_position = portfolio.positions
                .get(&order.symbol)
                .map(|p| p.quantity)
                .unwrap_or(0.0);

            let new_position = match order.direction {
                Direction::Long => current_position + order.quantity,
                Direction::Short => current_position - order.quantity,
            };

            if new_position.abs() > limit {
                eprintln!("Position limit exceeded for {}", order.symbol);
                return false;
            }
        }

        // Check concentration
        let order_value = order.quantity * portfolio.get_price(&order.symbol).unwrap_or(0.0);
        let concentration = order_value / portfolio.value;

        if concentration > self.max_concentration {
            eprintln!("Concentration limit exceeded: {:.2}%", concentration * 100.0);
            return false;
        }

        true
    }

    /// Real-time risk monitoring
    pub fn check_real_time(
        &self,
        portfolio: &Portfolio,
    ) -> Result<(), RiskViolation> {
        // Check entropy
        if portfolio.entropy > self.critical_threshold {
            return Err(RiskViolation::CriticalEntropy(portfolio.entropy));
        } else if portfolio.entropy > self.warning_threshold {
            eprintln!("WARNING: High entropy {:.3}", portfolio.entropy);
        }

        // Check drawdown
        let drawdown = portfolio.calculate_drawdown();
        if drawdown > self.max_drawdown {
            return Err(RiskViolation::MaxDrawdown(drawdown));
        }

        // Check leverage
        let leverage = portfolio.calculate_leverage();
        if leverage > self.max_leverage {
            return Err(RiskViolation::ExcessiveLeverage(leverage));
        }

        Ok(())
    }
}

#[derive(Debug)]
pub enum RiskViolation {
    CriticalEntropy(f64),
    MaxDrawdown(f64),
    ExcessiveLeverage(f64),
    PositionLimitExceeded(String, f64),
}

impl RiskViolation {
    pub fn is_critical(&self) -> bool {
        matches!(self,
            RiskViolation::CriticalEntropy(_) |
            RiskViolation::MaxDrawdown(_)
        )
    }
}
```

## 4. Error Handling

```rust
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TradingError {
    #[error("Broker not found: {0}")]
    BrokerNotFound(String),

    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Risk check failed: {0}")]
    RiskCheckFailed(String),

    #[error("Market data error: {0}")]
    MarketDataError(String),

    #[error("Execution error: {0}")]
    ExecutionError(String),

    #[error("Emergency stop active")]
    EmergencyStop,
}
```

## 5. Queen Orchestrator Integration

```rust
/// Queen agent coordinates multiple trading instances
pub struct QueenOrchestrator {
    /// Trading coordinators for different strategies/markets
    coordinators: HashMap<String, TradingOrchestrator>,

    /// Global risk monitor
    global_risk: Arc<RwLock<ThermodynamicRiskMonitor>>,

    /// Inter-strategy communication
    message_bus: MessageBus,
}

impl QueenOrchestrator {
    pub async fn add_coordinator(
        &mut self,
        name: String,
        coordinator: TradingOrchestrator,
    ) {
        self.coordinators.insert(name, coordinator);
    }

    pub async fn start_all(&mut self) -> Result<(), TradingError> {
        for (name, coordinator) in &mut self.coordinators {
            println!("Starting coordinator: {}", name);
            coordinator.start().await?;
        }

        Ok(())
    }

    pub async fn stop_all(&mut self) -> Result<(), TradingError> {
        for (name, coordinator) in &mut self.coordinators {
            println!("Stopping coordinator: {}", name);
            coordinator.stop().await?;
        }

        Ok(())
    }
}
```

## 6. Academic References

1. Kissell, R., & Glantz, M. (2013). *Optimal Trading Strategies*. AMACOM.

2. Almgren, R., & Chriss, N. (2001). *Optimal execution of portfolio transactions*. Journal of Risk, 3, 5-40.

3. Bertsimas, D., & Lo, A. W. (1998). *Optimal control of execution costs*. Journal of Financial Markets, 1(1), 1-50.

4. Cartea, Á., et al. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press.

5. Lehalle, C. A., & Laruelle, S. (2013). *Market Microstructure in Practice*. World Scientific.
