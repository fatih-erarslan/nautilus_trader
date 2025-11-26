# nt-execution: Neural Trader Execution Engine

## Overview

The `nt-execution` crate provides comprehensive order execution and broker integration for the Neural Trader platform. It supports 11+ brokers and market data providers with high-performance async execution (<10ms latency target).

## Architecture

### Core Components

```
nt-execution/
├── broker.rs              # Broker trait and core interfaces
├── order_manager.rs       # Order lifecycle management with actor pattern
├── fill_reconciliation.rs # Fill tracking and reconciliation
├── router.rs              # Smart order routing
├── alpaca_broker.rs       # Alpaca Markets integration
├── ibkr_broker.rs         # Interactive Brokers integration
├── polygon_broker.rs      # Polygon market data
├── ccxt_broker.rs         # CCXT crypto exchange integration
├── questrade_broker.rs    # Questrade (Canadian) integration
├── oanda_broker.rs        # OANDA forex integration
├── lime_broker.rs         # Lime Trading integration
├── alpha_vantage.rs       # Alpha Vantage market data
├── news_api.rs            # News API integration
├── yahoo_finance.rs       # Yahoo Finance data provider
└── odds_api.rs            # The Odds API for sports betting
```

## Key Features

### 1. Order Manager (Actor Pattern)

The `OrderManager` uses an actor-based design for high-throughput order processing:

```rust
pub struct OrderManager {
    message_tx: mpsc::Sender<OrderMessage>,
    orders: Arc<DashMap<String, TrackedOrder>>,
}

impl OrderManager {
    pub async fn place_order(&self, request: OrderRequest) -> Result<OrderResponse>;
    pub async fn cancel_order(&self, order_id: String) -> Result<()>;
    pub async fn get_order_status(&self, order_id: &str) -> Result<OrderStatus>;
    pub async fn handle_order_update(&self, update: OrderUpdate) -> Result<()>;
}
```

**Performance targets:**
- Order placement: <10ms end-to-end
- Status lookup: <1ms (cached), <5ms (broker query)
- Concurrent processing: 1000+ orders/sec

**Features:**
- Asynchronous message-based architecture
- Automatic retry with exponential backoff (max 3 attempts)
- Fill tracking and partial fill handling
- Real-time order update processing
- DashMap for lock-free concurrent access

### 2. Order Types and Status

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum OrderStatus {
    Pending,          // Submitted but not acknowledged
    Accepted,         // Acknowledged by broker
    PartiallyFilled,  // Partially executed
    Filled,           // Completely executed
    Cancelled,        // User cancelled
    Rejected,         // Broker rejected
    Expired,          // Time expired
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderRequest {
    pub symbol: Symbol,
    pub side: OrderSide,
    pub order_type: OrderType,
    pub quantity: u32,
    pub limit_price: Option<Decimal>,
    pub stop_price: Option<Decimal>,
    pub time_in_force: TimeInForce,
}
```

### 3. Broker Client Trait

All brokers implement the `BrokerClient` trait:

```rust
#[async_trait]
pub trait BrokerClient: Send + Sync {
    async fn place_order(&self, request: OrderRequest) -> Result<OrderResponse>;
    async fn cancel_order(&self, order_id: &str) -> Result<()>;
    async fn get_order(&self, order_id: &str) -> Result<OrderResponse>;
    async fn get_orders(&self, filter: OrderFilter) -> Result<Vec<OrderResponse>>;
    async fn get_account(&self) -> Result<Account>;
    async fn get_positions(&self) -> Result<Vec<Position>>;
    async fn health_check(&self) -> Result<HealthStatus>;
}
```

### 4. Supported Brokers

#### Alpaca Markets
- Paper and live trading
- WebSocket real-time updates
- Options, stocks, crypto
- Rate limiting: 200 requests/min
- Endpoint: `https://api.alpaca.markets`

#### Interactive Brokers (IBKR)
- TWS/Gateway integration
- Global markets support
- Advanced order types
- Real-time quotes

#### Polygon.io
- Market data provider
- Real-time and historical data
- WebSocket streaming
- News and reference data

#### CCXT
- 100+ cryptocurrency exchanges
- Unified API interface
- Order book depth
- WebSocket support

#### Questrade (Canadian)
- Canadian equities and options
- Real-time quotes
- Tax reporting (T5008)
- TSX/TSXV support

#### OANDA
- Forex trading
- CFD support
- Real-time pricing
- Multiple currency pairs

#### Lime Trading
- Direct market access
- Low latency execution
- Institutional-grade
- Smart order routing

## Order Validation Logic

The execution engine implements comprehensive order validation:

```rust
// Symbol validation
- Non-empty
- Alphanumeric + dots only
- Max 10 characters

// Quantity validation
- Must be positive integer
- No fractional shares (unless crypto)

// Price validation
- Limit orders: require limit_price > 0
- Stop orders: require stop_price > 0
- Stop-limit: require both prices
- Market orders: no price required

// Time-in-force validation
- Day, GTC, IOC, FOK
- Market hours validation
- Extended hours support
```

## Safety Features

### Environment-Based Configuration

```bash
# Required for live trading
ENABLE_LIVE_TRADING=true    # Must be explicitly set
PAPER_TRADING=false         # Default: true

# Broker configuration
BROKER=alpaca
BROKER_API_KEY=your_key
BROKER_SECRET_KEY=your_secret
```

### Dual-Check System

1. **Paper Trading Default**: All orders execute in paper mode unless explicitly disabled
2. **Live Trading Gate**: `ENABLE_LIVE_TRADING` must be set to `true`
3. **API Key Validation**: Live trading requires valid broker credentials

### Order Flow

```
┌─────────────────┐
│  Order Request  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Validate Order │ ← Symbol, quantity, price checks
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Check Config   │ ← ENABLE_LIVE_TRADING check
└────────┬────────┘
         │
         ▼
    ╔═══════╗
    ║ Paper ║───► Paper Trading Engine (Default)
    ╚═══╤═══╝
        │
        ▼
    ╔═══════╗
    ║  Live ║───► Broker API (Requires explicit enable)
    ╚═══════╝
```

## Performance Optimization

### 1. Retry with Exponential Backoff

```rust
async fn retry_with_backoff<F, T, E>(
    mut f: F,
    max_attempts: u32,
    initial_delay: Duration,
) -> Result<T>
```

- Initial delay: 100ms
- Backoff factor: 2x
- Max attempts: 3
- Total max wait: ~700ms

### 2. Lock-Free Concurrent Access

Uses `DashMap` for O(1) order lookups without global locks:

```rust
orders: Arc<DashMap<String, TrackedOrder>>
```

### 3. Actor-Based Message Passing

Prevents race conditions and enables high concurrency:

```rust
enum OrderMessage {
    PlaceOrder { request, response_tx },
    CancelOrder { order_id, response_tx },
    GetOrderStatus { order_id, response_tx },
    UpdateOrder { update },
}
```

### 4. Fast-Path Optimization

```rust
// Fast path: check cache first (no async call)
if let Some(order) = self.orders.get(order_id) {
    return Ok(order.status);
}

// Slow path: query broker only if not cached
let order = broker.get_order(order_id).await?;
```

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    #[error("Order validation failed: {0}")]
    ValidationFailed(String),

    #[error("Broker API error: {0}")]
    BrokerApiError(String),

    #[error("Timeout waiting for response")]
    Timeout,

    #[error("Order not found: {0}")]
    OrderNotFound(String),

    #[error("Live trading is disabled")]
    LiveTradingDisabled,
}
```

## Usage Examples

### Basic Order Placement

```rust
use nt_execution::{OrderManager, OrderRequest, OrderType, OrderSide};

// Create order manager from environment
let manager = OrderManager::from_env()?;

// Create order request
let request = OrderRequest {
    symbol: Symbol::new("AAPL")?,
    side: OrderSide::Buy,
    order_type: OrderType::Market,
    quantity: 100,
    limit_price: None,
    stop_price: None,
    time_in_force: TimeInForce::Day,
};

// Place order (async)
let response = manager.place_order(request).await?;
println!("Order placed: {}", response.order_id);
```

### Order Status Tracking

```rust
// Get order status
let status = manager.get_order_status(&order_id).await?;

match status {
    OrderStatus::Filled => println!("Order complete!"),
    OrderStatus::PartiallyFilled => println!("Partial fill..."),
    OrderStatus::Rejected => println!("Order rejected"),
    _ => println!("Status: {:?}", status),
}
```

### Order Cancellation

```rust
// Cancel an order
manager.cancel_order(order_id).await?;
println!("Order cancelled");
```

### Real-Time Updates

```rust
// Handle order update from WebSocket
let update = OrderUpdate {
    order_id: "abc123".to_string(),
    status: OrderStatus::Filled,
    filled_qty: 100,
    filled_avg_price: Some(Decimal::from(175.50)),
    timestamp: Utc::now(),
};

manager.handle_order_update(update).await?;
```

## Testing

The crate includes comprehensive tests:

```bash
# Run all tests
cargo test --package nt-execution

# Run with output
cargo test --package nt-execution -- --nocapture

# Test specific module
cargo test --package nt-execution order_manager::tests
```

## Dependencies

```toml
[dependencies]
nt-core = { version = "2.0.0", path = "../core" }
tokio = { workspace = true }
async-trait = { workspace = true }
serde = { workspace = true }
serde_json = { workspace = true }
chrono = { workspace = true }
rust_decimal = { workspace = true }
reqwest = { workspace = true }
tokio-tungstenite = { workspace = true }
dashmap = "5.5"
uuid = { workspace = true }
tracing = { workspace = true }
```

## Integration with MCP Server

The execution engine is exposed via NAPI bindings for Node.js integration:

```typescript
// Available in mcp_tools.rs
export async function execute_trade(params: {
  strategy: string;
  symbol: string;
  action: string;
  quantity: number;
  order_type?: string;
  limit_price?: number;
}): Promise<string>;
```

## Future Enhancements

1. **Smart Order Routing**: Multi-venue execution optimization
2. **Fill Reconciliation**: Advanced fill tracking and reporting
3. **Order Slicing**: TWAP/VWAP algorithmic execution
4. **Pre-trade Compliance**: Regulatory checks before execution
5. **Post-trade Analytics**: Execution quality measurement

## Compilation Status

✅ **VERIFIED**: The nt-execution crate compiles successfully
✅ **DEPENDENCIES**: All dependencies resolved correctly
✅ **INTEGRATION**: Successfully integrated in napi-bindings
⚠️ **WARNINGS**: 59 warnings (mostly unused imports and dead code)

To apply suggested fixes:
```bash
cargo fix --lib -p nt-execution
```

## Version

Current version: **2.0.0**
