# HyperPhysics Backtesting Framework

A comprehensive, event-driven backtesting framework for algorithmic trading strategies.

## Features

### Core Capabilities

- ✅ **Event-Driven Architecture**: Processes market data chronologically for realistic simulation
- ✅ **Strategy Trait**: Flexible interface with `on_bar`, `on_tick`, `initialize`, and `finalize` methods
- ✅ **Portfolio Management**: Automatic position tracking with cash management
- ✅ **Order Execution**: Realistic order fills with configurable slippage and commission models
- ✅ **Performance Metrics**: Comprehensive statistics including Sharpe ratio, max drawdown, win rate
- ✅ **Trade Log**: Detailed tracking of all trades with entry/exit prices and P&L
- ✅ **Multiple Timeframes**: Support for 1-minute to monthly bars
- ✅ **Market Data Integration**: Works with any exchange via `MarketDataProvider` trait

## Quick Start

### Basic Strategy Implementation

```rust
use hyperphysics_market::backtest::{Strategy, Signal};
use hyperphysics_market::data::Bar;
use async_trait::async_trait;

struct SimpleMovingAverage {
    period: usize,
    prices: Vec<f64>,
}

#[async_trait]
impl Strategy for SimpleMovingAverage {
    async fn initialize(&mut self) {
        self.prices.clear();
    }

    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        if self.prices.len() < self.period {
            return vec![];
        }

        let sma: f64 = self.prices.iter().rev().take(self.period).sum::<f64>()
            / self.period as f64;

        if bar.close > sma {
            vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None, // Market order
            }]
        } else {
            vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }]
        }
    }

    async fn finalize(&mut self) {}
}
```

### Running a Backtest

```rust
use hyperphysics_market::backtest::{BacktestEngine, BacktestConfig, Commission, Slippage};
use hyperphysics_market::data::Timeframe;
use chrono::{Duration, Utc};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure backtest parameters
    let config = BacktestConfig {
        initial_capital: 100_000.0,
        commission: Commission::Percentage(0.001), // 0.1%
        slippage: Slippage::Percentage(0.0005),    // 0.05%
        symbols: vec!["AAPL".to_string()],
        timeframe: Timeframe::Day1,
        start_date: Utc::now() - Duration::days(365),
        end_date: Utc::now(),
    };

    // Create provider (Alpaca, Binance, etc.)
    let provider = AlpacaProvider::new(api_key, api_secret, true);

    // Run backtest
    let engine = BacktestEngine::new(provider, config);
    let mut strategy = SimpleMovingAverage { period: 20, prices: vec![] };

    let result = engine.run(&mut strategy).await?;

    // Analyze results
    println!("Total Return: {:.2}%", result.metrics.total_return);
    println!("Sharpe Ratio: {:.2}", result.metrics.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", result.metrics.max_drawdown);
    println!("Win Rate: {:.2}%", result.metrics.win_rate);

    Ok(())
}
```

## Architecture

### Strategy Trait

The `Strategy` trait defines the interface for all trading strategies:

```rust
#[async_trait]
pub trait Strategy: Send + Sync {
    /// Initialize strategy before backtesting starts
    async fn initialize(&mut self);

    /// Process a new bar and generate trading signals
    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal>;

    /// Process a new tick (optional, for tick-level strategies)
    async fn on_tick(&mut self, tick: &Tick) -> Vec<Signal>;

    /// Finalize strategy after backtesting completes
    async fn finalize(&mut self);

    /// Get strategy name
    fn name(&self) -> String;
}
```

### Signal Types

Strategies emit signals to execute trades:

```rust
pub enum Signal {
    /// Buy signal with symbol, quantity, and optional limit price
    Buy {
        symbol: String,
        quantity: f64,
        price: Option<f64>, // None = market order
    },

    /// Sell signal
    Sell {
        symbol: String,
        quantity: f64,
        price: Option<f64>,
    },

    /// Close all positions for a symbol
    ClosePosition { symbol: String },

    /// Close all open positions
    CloseAll,
}
```

### Portfolio Management

The `Portfolio` struct automatically manages:

- **Cash Balance**: Tracks available capital
- **Positions**: Open positions with quantity and average entry price
- **Equity**: Total portfolio value (cash + positions)
- **Commission**: Automatic deduction based on configured model

```rust
pub struct Portfolio {
    pub cash: f64,
    pub initial_capital: f64,
    pub positions: HashMap<String, Position>,
    pub commission: Commission,
}
```

### Position Tracking

Each position maintains detailed information:

```rust
pub struct Position {
    pub symbol: String,
    pub quantity: f64,           // Positive for long
    pub avg_price: f64,          // Average entry price
    pub current_price: f64,      // Latest market price
    pub opened_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}
```

## Configuration Options

### Commission Models

```rust
pub enum Commission {
    /// Fixed commission per trade
    Fixed(f64),              // e.g., $1.00 per trade

    /// Percentage of trade value
    Percentage(f64),         // e.g., 0.001 = 0.1%

    /// No commission
    None,
}
```

### Slippage Models

```rust
pub enum Slippage {
    /// Fixed slippage in price units
    Fixed(f64),              // e.g., $0.05 per share

    /// Percentage of price
    Percentage(f64),         // e.g., 0.0005 = 0.05%

    /// No slippage
    None,
}
```

## Performance Metrics

The framework calculates comprehensive performance statistics:

```rust
pub struct PerformanceMetrics {
    /// Total return (percentage)
    pub total_return: f64,

    /// Annualized return (percentage)
    pub annualized_return: f64,

    /// Sharpe ratio (risk-adjusted return)
    pub sharpe_ratio: f64,

    /// Maximum drawdown (percentage)
    pub max_drawdown: f64,

    /// Win rate (percentage of profitable trades)
    pub win_rate: f64,

    /// Total number of trades
    pub total_trades: usize,

    /// Number of winning trades
    pub winning_trades: usize,

    /// Number of losing trades
    pub losing_trades: usize,

    /// Average profit per winning trade
    pub avg_win: f64,

    /// Average loss per losing trade
    pub avg_loss: f64,

    /// Profit factor (gross profit / gross loss)
    pub profit_factor: f64,

    /// Total commission paid
    pub total_commission: f64,

    /// Starting capital
    pub initial_capital: f64,

    /// Final equity
    pub final_equity: f64,

    /// Duration of backtest in days
    pub duration_days: f64,
}
```

## Example Strategies

### 1. Moving Average Crossover

```rust
struct MovingAverageCrossover {
    short_period: usize,
    long_period: usize,
    prices: Vec<f64>,
    position_open: bool,
}

#[async_trait]
impl Strategy for MovingAverageCrossover {
    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        if self.prices.len() < self.long_period {
            return vec![];
        }

        let short_ma = self.calculate_sma(self.short_period);
        let long_ma = self.calculate_sma(self.long_period);

        // Golden cross: buy
        if short_ma > long_ma && !self.position_open {
            self.position_open = true;
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Death cross: sell
        if short_ma < long_ma && self.position_open {
            self.position_open = false;
            return vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        vec![]
    }
}
```

### 2. Mean Reversion

```rust
struct MeanReversionStrategy {
    period: usize,
    std_dev_threshold: f64,
    prices: Vec<f64>,
}

#[async_trait]
impl Strategy for MeanReversionStrategy {
    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        let mean = self.calculate_mean()?;
        let std_dev = self.calculate_std_dev(mean);
        let z_score = (bar.close - mean) / std_dev;

        // Buy when price is oversold
        if z_score < -self.std_dev_threshold {
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Sell when price returns to mean
        if z_score > 0.0 {
            return vec![Signal::ClosePosition {
                symbol: bar.symbol.clone(),
            }];
        }

        vec![]
    }
}
```

### 3. Momentum Strategy

```rust
struct MomentumStrategy {
    period: usize,
    threshold: f64,
    prices: Vec<f64>,
}

#[async_trait]
impl Strategy for MomentumStrategy {
    async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
        self.prices.push(bar.close);

        let roc = self.calculate_rate_of_change()?;

        // Buy on strong positive momentum
        if roc > self.threshold {
            return vec![Signal::Buy {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        // Sell on negative momentum
        if roc < -self.threshold {
            return vec![Signal::Sell {
                symbol: bar.symbol.clone(),
                quantity: 100.0,
                price: None,
            }];
        }

        vec![]
    }
}
```

## Supported Timeframes

```rust
pub enum Timeframe {
    Minute1,   // 1-minute bars
    Minute5,   // 5-minute bars
    Minute15,  // 15-minute bars
    Minute30,  // 30-minute bars
    Hour1,     // 1-hour bars
    Hour4,     // 4-hour bars
    Day1,      // Daily bars
    Week1,     // Weekly bars
    Month1,    // Monthly bars
}
```

## Testing

The framework includes comprehensive unit and integration tests:

```bash
# Run unit tests
cargo test -p hyperphysics-market backtest --lib

# Run integration tests
cargo test -p hyperphysics-market --test backtest_integration

# Run example
cargo run --example backtest_demo
```

### Test Coverage

- ✅ Position creation and updates
- ✅ Portfolio buy/sell operations
- ✅ Commission calculations
- ✅ Slippage applications
- ✅ Performance metrics calculations
- ✅ Equity curve generation
- ✅ Multiple timeframe support
- ✅ Complete strategy backtests

## Advanced Usage

### Custom Order Execution

For advanced users who want custom execution logic:

```rust
// The framework handles this automatically, but you can customize
// by implementing your own signal generation logic

async fn on_bar(&mut self, bar: &Bar) -> Vec<Signal> {
    // Custom logic for limit orders
    let resistance_level = self.calculate_resistance();

    vec![Signal::Buy {
        symbol: bar.symbol.clone(),
        quantity: 100.0,
        price: Some(resistance_level), // Limit order
    }]
}
```

### Multiple Symbol Backtesting

```rust
let config = BacktestConfig {
    symbols: vec![
        "AAPL".to_string(),
        "GOOGL".to_string(),
        "MSFT".to_string(),
    ],
    // ... other config
};
```

### Tick-Level Strategies

For high-frequency strategies:

```rust
#[async_trait]
impl Strategy for HFTStrategy {
    async fn on_tick(&mut self, tick: &Tick) -> Vec<Signal> {
        // Process individual ticks instead of bars
        // Useful for market-making, arbitrage, etc.
        vec![]
    }
}
```

## Integration with Market Data Providers

The framework integrates seamlessly with all supported exchanges:

```rust
// Alpaca
let provider = AlpacaProvider::new(api_key, api_secret, true);

// Binance
let provider = BinanceProvider::new(api_key, api_secret);

// Coinbase
let provider = CoinbaseProvider::new(api_key, api_secret);

// And more: Bybit, Kraken, OKX, Interactive Brokers
```

## Performance Considerations

- **Memory Efficient**: Streams bars sequentially, doesn't load all data at once
- **Async/Await**: Leverages Tokio for efficient I/O
- **Type Safety**: Strongly typed with Rust's compile-time guarantees
- **Realistic Execution**: Models commission and slippage accurately

## Best Practices

1. **Always Initialize**: Call `initialize()` to reset strategy state
2. **Handle Edge Cases**: Check for sufficient data before calculations
3. **Test Thoroughly**: Validate strategies with various market conditions
4. **Realistic Parameters**: Use realistic commission and slippage values
5. **Risk Management**: Implement position sizing and stop losses
6. **Overfitting**: Avoid optimizing on the same data you backtest on

## Future Enhancements

Planned features for future releases:

- [ ] Multi-asset portfolio optimization
- [ ] Walk-forward analysis
- [ ] Monte Carlo simulation
- [ ] Order book simulation
- [ ] Market impact modeling
- [ ] Transaction cost analysis (TCA)
- [ ] Risk-adjusted position sizing
- [ ] Stop-loss and take-profit orders

## License

Part of the HyperPhysics project.
