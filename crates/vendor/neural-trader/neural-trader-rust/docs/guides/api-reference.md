# API Reference

Complete API documentation for neural-trader Rust implementation.

## 📚 Table of Contents

- [Market Data](#market-data)
- [Strategies](#strategies)
- [Execution](#execution)
- [Portfolio](#portfolio)
- [Risk Management](#risk-management)
- [Neural Networks](#neural-networks)
- [Backtesting](#backtesting)
- [CLI Commands](#cli-commands)
- [MCP Tools](#mcp-tools)

## 🔌 Installation

```bash
# Add to Cargo.toml
nt-core = "0.1"
nt-strategies = "0.1"
nt-execution = "0.1"
```

## Market Data

### AlpacaProvider

Fetch real-time and historical market data from Alpaca.

```rust
use nt_market_data::{AlpacaProvider, MarketDataProvider};
use chrono::{Utc, Duration};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize provider
    let provider = AlpacaProvider::new().await?;

    // Fetch historical bars
    let start = Utc::now() - Duration::days(365);
    let end = Utc::now();
    let bars = provider.fetch_bars("AAPL", start, end).await?;

    println!("Fetched {} bars", bars.len());
    Ok(())
}
```

#### Methods

##### `new() -> Result<Self>`

Create new Alpaca provider using environment variables.

**Environment Variables**:
- `ALPACA_API_KEY` - API key
- `ALPACA_SECRET_KEY` - Secret key
- `ALPACA_BASE_URL` - Base URL (paper or live)

##### `fetch_bars(symbol: &str, start: DateTime<Utc>, end: DateTime<Utc>) -> Result<Vec<Bar>>`

Fetch OHLCV bars for a symbol.

**Parameters**:
- `symbol` - Ticker symbol (e.g., "AAPL")
- `start` - Start timestamp
- `end` - End timestamp

**Returns**: Vector of `Bar` structs

##### `stream_bars(symbol: &str) -> Result<impl Stream<Item = Bar>>`

Stream real-time market data.

**Parameters**:
- `symbol` - Ticker symbol

**Returns**: Async stream of `Bar` updates

#### Bar Struct

```rust
pub struct Bar {
    pub timestamp: DateTime<Utc>,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: u64,
}
```

## Strategies

### PairsStrategy

Statistical arbitrage strategy for correlated pairs.

```rust
use nt_strategies::{PairsStrategy, Strategy};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let strategy = PairsStrategy::new(
        vec!["AAPL", "MSFT"],  // symbols
        20,                    // lookback period
        2.0                    // entry threshold (z-score)
    );

    let signals = strategy.generate_signals(&market_data).await?;

    for signal in signals {
        println!("{:?}", signal);
    }

    Ok(())
}
```

#### Constructor

```rust
pub fn new(symbols: Vec<&str>, lookback: usize, entry_threshold: f64) -> Self
```

**Parameters**:
- `symbols` - Pair of correlated assets
- `lookback` - Window size for spread calculation
- `entry_threshold` - Z-score threshold for entry

#### Methods

##### `generate_signals(&self, data: &DataFrame) -> Result<Vec<Signal>>`

Generate trading signals based on spread deviation.

**Returns**: Vector of `Signal` enum (Buy, Sell, Hold)

##### `calculate_spread(&self, data: &DataFrame) -> Result<Series>`

Calculate normalized spread between pairs.

##### `get_parameters(&self) -> StrategyParameters`

Get current strategy parameters.

### Strategy Trait

Implement custom strategies:

```rust
use async_trait::async_trait;
use nt_core::{Signal, DataFrame};

#[async_trait]
pub trait Strategy: Send + Sync {
    async fn generate_signals(&self, data: &DataFrame) -> Result<Vec<Signal>>;
    async fn update_parameters(&mut self, params: StrategyParameters) -> Result<()>;
    fn name(&self) -> &str;
}
```

## Execution

### OrderManager

Manage order lifecycle and execution.

```rust
use nt_execution::{OrderManager, Order, Side, OrderType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut manager = OrderManager::new().await?;

    let order = Order::new(
        "AAPL",
        100,
        Side::Buy,
        OrderType::Limit(180.50)
    );

    let fill = manager.submit_order(order).await?;
    println!("Order filled at ${}", fill.price);

    Ok(())
}
```

#### Order Struct

```rust
pub struct Order {
    pub id: Uuid,
    pub symbol: String,
    pub quantity: u32,
    pub side: Side,
    pub order_type: OrderType,
    pub status: OrderStatus,
    pub created_at: DateTime<Utc>,
}
```

#### Enums

```rust
pub enum Side {
    Buy,
    Sell,
}

pub enum OrderType {
    Market,
    Limit(f64),
    Stop(f64),
    StopLimit { stop: f64, limit: f64 },
}

pub enum OrderStatus {
    Pending,
    Submitted,
    PartiallyFilled,
    Filled,
    Cancelled,
    Rejected,
}
```

#### Methods

##### `submit_order(&mut self, order: Order) -> Result<Fill>`

Submit order to broker for execution.

##### `cancel_order(&mut self, order_id: Uuid) -> Result<()>`

Cancel pending order.

##### `get_order_status(&self, order_id: Uuid) -> Result<OrderStatus>`

Query current order status.

##### `get_fills(&self) -> Vec<Fill>`

Get all executed fills.

## Portfolio

### Portfolio Tracker

Track positions and calculate P&L.

```rust
use nt_portfolio::{Portfolio, Position};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut portfolio = Portfolio::new(100_000.0);

    // Add positions
    portfolio.add_position("AAPL", 50, 180.50)?;
    portfolio.add_position("MSFT", 30, 380.25)?;

    // Calculate metrics
    let metrics = portfolio.calculate_metrics()?;
    println!("Total Value: ${:.2}", metrics.total_value);
    println!("P&L: ${:.2}", metrics.pnl);
    println!("Return: {:.2}%", metrics.return_pct);

    Ok(())
}
```

#### Methods

##### `new(initial_capital: f64) -> Self`

Create new portfolio with starting capital.

##### `add_position(&mut self, symbol: &str, quantity: u32, price: f64) -> Result<()>`

Add or update position.

##### `remove_position(&mut self, symbol: &str) -> Result<Position>`

Close position and realize P&L.

##### `calculate_metrics(&self) -> Result<PortfolioMetrics>`

Calculate comprehensive portfolio metrics.

##### `calculate_pnl(&self) -> Result<PnL>`

Calculate realized and unrealized P&L.

##### `get_positions(&self) -> Vec<&Position>`

Get all current positions.

#### PortfolioMetrics Struct

```rust
pub struct PortfolioMetrics {
    pub total_value: f64,
    pub cash: f64,
    pub equity: f64,
    pub pnl: f64,
    pub return_pct: f64,
    pub sharpe_ratio: f64,
    pub max_drawdown: f64,
}
```

## Risk Management

### VaR Calculation

Calculate Value at Risk using various methods.

```rust
use nt_risk::{calculate_var, VarMethod};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let portfolio = load_portfolio()?;

    // Historical VaR
    let var_hist = calculate_var(&portfolio, 0.95, VarMethod::Historical)?;
    println!("VaR (Historical): ${:.2}", var_hist);

    // Monte Carlo VaR
    let var_mc = calculate_var(&portfolio, 0.95, VarMethod::MonteCarlo)?;
    println!("VaR (Monte Carlo): ${:.2}", var_mc);

    Ok(())
}
```

#### Methods

##### `calculate_var(portfolio: &Portfolio, confidence: f64, method: VarMethod) -> Result<f64>`

Calculate Value at Risk.

**Parameters**:
- `portfolio` - Portfolio to analyze
- `confidence` - Confidence level (e.g., 0.95 for 95%)
- `method` - Calculation method

**Methods**:
- `Historical` - Historical simulation
- `Parametric` - Normal distribution assumption
- `MonteCarlo` - Monte Carlo simulation

##### `calculate_position_size(account_value: f64, risk_percent: f64, stop_loss: f64) -> f64`

Calculate optimal position size based on risk tolerance.

##### `check_risk_limits(portfolio: &Portfolio, limits: &RiskLimits) -> Result<Vec<Violation>>`

Validate portfolio against risk limits.

## Neural Networks

### N-HiTS Model

Train and use neural forecasting models.

```rust
use nt_neural::{NHiTS, ForecastModel};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Train model
    let mut model = NHiTS::new(
        input_size: 100,
        horizon: 5,
        hidden_size: 512,
    );

    model.train(&training_data, epochs: 100).await?;

    // Save model
    model.save("models/nhits.bin")?;

    // Load and predict
    let model = NHiTS::load("models/nhits.bin")?;
    let forecast = model.predict(&input_sequence).await?;

    println!("5-day forecast: {:?}", forecast);

    Ok(())
}
```

#### Methods

##### `train(&mut self, data: &DataFrame, epochs: usize) -> Result<TrainingMetrics>`

Train model on historical data.

##### `predict(&self, input: &[f64]) -> Result<Vec<f64>>`

Generate forecast for given input sequence.

##### `evaluate(&self, test_data: &DataFrame) -> Result<EvaluationMetrics>`

Evaluate model performance on test set.

##### `save(&self, path: &str) -> Result<()>`

Save model to disk.

##### `load(path: &str) -> Result<Self>`

Load model from disk.

## Backtesting

### Backtester

Simulate strategy performance on historical data.

```rust
use nt_backtesting::Backtester;
use nt_strategies::PairsStrategy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);
    let backtester = Backtester::new(strategy);

    let results = backtester
        .set_initial_capital(100_000.0)
        .set_commission(0.001)
        .run("2024-01-01", "2024-12-31")
        .await?;

    println!("Total Return: {:.2}%", results.total_return);
    println!("Sharpe Ratio: {:.2}", results.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", results.max_drawdown);

    Ok(())
}
```

#### Methods

##### `new(strategy: impl Strategy) -> Self`

Create backtester with strategy.

##### `set_initial_capital(mut self, capital: f64) -> Self`

Set starting capital (builder pattern).

##### `set_commission(mut self, rate: f64) -> Self`

Set commission rate per trade.

##### `run(&self, start: &str, end: &str) -> Result<BacktestResults>`

Run backtest over date range.

#### BacktestResults Struct

```rust
pub struct BacktestResults {
    pub total_return: f64,
    pub annual_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub win_rate: f64,
    pub total_trades: usize,
    pub profitable_trades: usize,
    pub avg_win: f64,
    pub avg_loss: f64,
    pub profit_factor: f64,
}
```

## CLI Commands

### Global Flags

```bash
--help, -h         Show help
--version, -v      Show version
--config <FILE>    Config file path
--verbose          Verbose logging
```

### Commands

#### `market-data`

Fetch market data.

```bash
neural-trader market-data \
  --symbol AAPL \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --output data/aapl.csv
```

#### `backtest`

Run strategy backtest.

```bash
neural-trader backtest \
  --strategy pairs \
  --symbols AAPL MSFT \
  --start 2024-01-01 \
  --end 2024-12-31 \
  --capital 100000 \
  --output results/backtest.json
```

#### `neural train`

Train neural model.

```bash
neural-trader neural train \
  --model nhits \
  --data data/prices.csv \
  --horizon 5 \
  --epochs 100 \
  --output models/nhits.bin
```

#### `risk`

Calculate risk metrics.

```bash
neural-trader risk var \
  --portfolio portfolio.json \
  --confidence 0.95 \
  --method monte-carlo
```

#### `mcp start`

Start MCP server.

```bash
neural-trader mcp start \
  --port 3000 \
  --log-level info
```

## MCP Tools

### Available Tools

Neural Trader provides 50+ MCP tools for Claude integration.

See full list: [MCP Integration Guide](./mcp-integration.md)

#### Example Tool: `mcp__neural-trader__backtest`

```json
{
  "tool": "mcp__neural-trader__backtest",
  "parameters": {
    "strategy": "pairs",
    "symbols": ["AAPL", "MSFT"],
    "start_date": "2024-01-01",
    "end_date": "2024-12-31",
    "initial_capital": 100000
  }
}
```

---

For more examples, see the [examples/](../examples/) directory.
