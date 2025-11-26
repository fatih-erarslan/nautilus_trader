# Simulation and Backtesting Engine

## Document Purpose

This document defines the **deterministic backtesting engine** for the Neural Rust port. It provides event-sourced architecture, slippage/fee models, order fill simulation, portfolio tracking, statistical validation, and performance metrics calculation.

## Table of Contents

1. [Backtesting Architecture](#backtesting-architecture)
2. [Deterministic Design](#deterministic-design)
3. [Event-Sourced Market Tape](#event-sourced-market-tape)
4. [Slippage and Fee Models](#slippage-and-fee-models)
5. [Order Fill Simulation](#order-fill-simulation)
6. [Portfolio Tracking](#portfolio-tracking)
7. [Statistical Validation](#statistical-validation)
8. [Performance Metrics](#performance-metrics)
9. [Report Generation](#report-generation)
10. [Comparison with Live](#comparison-with-live)
11. [Parameter Optimization](#parameter-optimization)
12. [Walk-Forward Analysis](#walk-forward-analysis)
13. [Monte Carlo Simulation](#monte-carlo-simulation)
14. [Troubleshooting](#troubleshooting)

---

## Backtesting Architecture

### Design Goals

1. **Determinism:** Same inputs → same outputs (seeded RNG)
2. **Accuracy:** Realistic slippage, fees, market impact
3. **Performance:** Process years of data in minutes
4. **Validation:** Compare to Python results (parity)
5. **Insights:** Rich metrics and visualizations

### Module Structure

```
crates/
└── backtesting/
    ├── src/
    │   ├── lib.rs
    │   ├── engine.rs           # Main backtest engine
    │   ├── clock.rs            # Simulated time
    │   ├── market_tape.rs      # Event-sourced data
    │   ├── order_simulator.rs  # Fill simulation
    │   ├── slippage.rs         # Slippage models
    │   ├── fees.rs             # Fee calculation
    │   ├── portfolio.rs        # Portfolio state tracking
    │   ├── metrics.rs          # Performance metrics
    │   ├── report.rs           # HTML/JSON reports
    │   ├── optimization.rs     # Parameter optimization
    │   ├── walk_forward.rs     # Walk-forward analysis
    │   └── monte_carlo.rs      # Monte Carlo simulation
    └── tests/
        ├── engine_test.rs
        ├── parity_test.rs      # vs Python
        └── fixtures/
            └── historical_data.csv
```

---

## Deterministic Design

### Seeded Random Number Generator

```rust
// crates/backtesting/src/engine.rs
use rand::{SeedableRng, rngs::StdRng};
use rand_distr::{Distribution, Normal};

pub struct BacktestEngine {
    config: BacktestConfig,
    rng: StdRng,
    clock: SimulatedClock,
    portfolio: Portfolio,
    market_tape: MarketTape,
}

impl BacktestEngine {
    pub fn new(config: BacktestConfig) -> Self {
        // Seeded RNG for reproducibility
        let rng = StdRng::seed_from_u64(config.random_seed);

        Self {
            rng,
            clock: SimulatedClock::new(config.start_date),
            portfolio: Portfolio::new(config.initial_capital),
            market_tape: MarketTape::load(&config.data_path).unwrap(),
            config,
        }
    }

    pub async fn run(&mut self) -> Result<BacktestResults, BacktestError> {
        tracing::info!("Starting backtest from {} to {}",
            self.config.start_date, self.config.end_date);

        let mut strategy_engine = self.create_strategy_engine();
        let mut risk_manager = RiskManager::new(self.config.risk_params.clone());
        let mut order_simulator = OrderSimulator::new(
            self.config.slippage_model.clone(),
            self.config.fee_model.clone(),
            &mut self.rng,
        );

        // Event loop
        while let Some(event) = self.market_tape.next_event() {
            // Advance simulated time
            self.clock.set_time(event.timestamp);

            match event.event_type {
                EventType::MarketTick(tick) => {
                    // Update market data
                    let signals = strategy_engine.process_tick(&tick, &self.portfolio).await?;

                    // Validate signals with risk manager
                    for signal in signals {
                        if let Ok(approved_signal) = risk_manager.validate_signal(&signal, &self.portfolio) {
                            // Simulate order execution
                            if let Ok(fill) = order_simulator.simulate_fill(
                                &approved_signal,
                                &tick,
                                &mut self.rng
                            ) {
                                // Update portfolio
                                self.portfolio.apply_fill(fill);
                            }
                        }
                    }
                }
                EventType::EndOfDay => {
                    // Mark-to-market portfolio
                    self.portfolio.mark_to_market(&self.market_tape.latest_prices());
                }
            }

            // Check if reached end date
            if self.clock.now() >= self.config.end_date {
                break;
            }
        }

        // Calculate final metrics
        Ok(self.calculate_results())
    }

    fn calculate_results(&self) -> BacktestResults {
        BacktestResults {
            total_return: self.portfolio.total_return(),
            sharpe_ratio: calculate_sharpe_ratio(&self.portfolio.equity_curve()),
            sortino_ratio: calculate_sortino_ratio(&self.portfolio.equity_curve()),
            max_drawdown: calculate_max_drawdown(&self.portfolio.equity_curve()),
            win_rate: self.portfolio.win_rate(),
            profit_factor: self.portfolio.profit_factor(),
            trades: self.portfolio.trade_history().clone(),
            equity_curve: self.portfolio.equity_curve().clone(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BacktestConfig {
    pub start_date: DateTime<Utc>,
    pub end_date: DateTime<Utc>,
    pub initial_capital: Decimal,
    pub strategies: Vec<String>,
    pub slippage_model: SlippageModel,
    pub fee_model: FeeModel,
    pub risk_params: RiskParams,
    pub random_seed: u64,
    pub data_path: PathBuf,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            start_date: Utc.ymd(2024, 1, 1).and_hms(9, 30, 0),
            end_date: Utc.ymd(2024, 12, 31).and_hms(16, 0, 0),
            initial_capital: Decimal::new(100_000, 0),
            strategies: vec!["momentum".to_string()],
            slippage_model: SlippageModel::default(),
            fee_model: FeeModel::default(),
            risk_params: RiskParams::default(),
            random_seed: 42,  // Deterministic by default
            data_path: PathBuf::from("data/historical.csv"),
        }
    }
}
```

---

## Event-Sourced Market Tape

### Market Tape Format

```rust
// crates/backtesting/src/market_tape.rs
use polars::prelude::*;

pub struct MarketTape {
    events: Vec<MarketEvent>,
    current_index: usize,
    prices_cache: HashMap<String, Decimal>,
}

#[derive(Debug, Clone)]
pub enum MarketEvent {
    MarketTick {
        timestamp: DateTime<Utc>,
        symbol: String,
        price: Decimal,
        volume: u64,
        bid: Decimal,
        ask: Decimal,
    },
    EndOfDay {
        timestamp: DateTime<Utc>,
    },
    MarketOpen {
        timestamp: DateTime<Utc>,
    },
    MarketClose {
        timestamp: DateTime<Utc>,
    },
}

impl MarketTape {
    /// Load from CSV file
    pub fn load(path: &Path) -> Result<Self, BacktestError> {
        let df = CsvReader::from_path(path)?
            .has_header(true)
            .finish()?;

        let mut events = Vec::new();

        // Convert DataFrame to events
        let timestamps = df.column("timestamp")?.datetime()?.as_datetime_iter();
        let symbols = df.column("symbol")?.utf8()?.into_iter();
        let prices = df.column("close")?.f64()?.into_iter();
        let volumes = df.column("volume")?.u64()?.into_iter();
        let bids = df.column("bid")?.f64()?.into_iter();
        let asks = df.column("ask")?.f64()?.into_iter();

        for (timestamp, symbol, price, volume, bid, ask) in izip!(
            timestamps, symbols, prices, volumes, bids, asks
        ) {
            if let (Some(ts), Some(sym), Some(p), Some(v), Some(b), Some(a)) = (
                timestamp, symbol, price, volume, bid, ask
            ) {
                events.push(MarketEvent::MarketTick {
                    timestamp: DateTime::from_timestamp(ts / 1_000_000, 0).unwrap(),
                    symbol: sym.to_string(),
                    price: Decimal::from_f64_retain(p).unwrap(),
                    volume: v,
                    bid: Decimal::from_f64_retain(b).unwrap(),
                    ask: Decimal::from_f64_retain(a).unwrap(),
                });
            }
        }

        // Sort by timestamp (critical for determinism)
        events.sort_by_key(|e| match e {
            MarketEvent::MarketTick { timestamp, .. } => *timestamp,
            MarketEvent::EndOfDay { timestamp } => *timestamp,
            MarketEvent::MarketOpen { timestamp } => *timestamp,
            MarketEvent::MarketClose { timestamp } => *timestamp,
        });

        Ok(Self {
            events,
            current_index: 0,
            prices_cache: HashMap::new(),
        })
    }

    /// Load from Parquet (faster for large datasets)
    pub fn load_parquet(path: &Path) -> Result<Self, BacktestError> {
        let df = ParquetReader::new(File::open(path)?)
            .finish()?;

        // Similar to CSV loading
        todo!()
    }

    pub fn next_event(&mut self) -> Option<&MarketEvent> {
        if self.current_index < self.events.len() {
            let event = &self.events[self.current_index];
            self.current_index += 1;

            // Update price cache
            if let MarketEvent::MarketTick { symbol, price, .. } = event {
                self.prices_cache.insert(symbol.clone(), *price);
            }

            Some(event)
        } else {
            None
        }
    }

    pub fn latest_prices(&self) -> &HashMap<String, Decimal> {
        &self.prices_cache
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
        self.prices_cache.clear();
    }
}

/// Save market tape for replay
impl MarketTape {
    pub fn save_to_csv(&self, path: &Path) -> Result<(), BacktestError> {
        let mut writer = csv::Writer::from_path(path)?;

        writer.write_record(&["timestamp", "symbol", "price", "volume", "bid", "ask"])?;

        for event in &self.events {
            if let MarketEvent::MarketTick { timestamp, symbol, price, volume, bid, ask } = event {
                writer.write_record(&[
                    timestamp.to_rfc3339(),
                    symbol.clone(),
                    price.to_string(),
                    volume.to_string(),
                    bid.to_string(),
                    ask.to_string(),
                ])?;
            }
        }

        writer.flush()?;
        Ok(())
    }
}
```

---

## Slippage and Fee Models

### Slippage Models

```rust
// crates/backtesting/src/slippage.rs
#[derive(Debug, Clone)]
pub enum SlippageModel {
    /// No slippage (unrealistic)
    None,

    /// Fixed basis points (e.g., 5 bps)
    FixedBps(Decimal),

    /// Volume-weighted (higher volume = less slippage)
    VolumeWeighted {
        base_bps: Decimal,
        volume_scale: f64,
    },

    /// Spread-based (slippage = fraction of bid-ask spread)
    SpreadBased {
        spread_fraction: f64,  // 0.5 = half spread
    },

    /// Market impact model (price moves against you)
    MarketImpact {
        impact_coefficient: f64,
    },
}

impl Default for SlippageModel {
    fn default() -> Self {
        // Conservative: 5 bps slippage
        Self::FixedBps(Decimal::new(5, 4))
    }
}

impl SlippageModel {
    pub fn calculate_slippage(
        &self,
        side: OrderSide,
        price: Decimal,
        quantity: u64,
        bid: Decimal,
        ask: Decimal,
        volume: u64,
        rng: &mut StdRng,
    ) -> Decimal {
        match self {
            Self::None => Decimal::ZERO,

            Self::FixedBps(bps) => {
                // Slippage = price * bps
                price * *bps
            }

            Self::VolumeWeighted { base_bps, volume_scale } => {
                // Less slippage for high volume
                let volume_factor = 1.0 / (1.0 + (volume as f64 / volume_scale).sqrt());
                let adjusted_bps = base_bps * Decimal::from_f64_retain(volume_factor).unwrap();
                price * adjusted_bps
            }

            Self::SpreadBased { spread_fraction } => {
                let spread = ask - bid;
                let slippage = spread * Decimal::from_f64_retain(*spread_fraction).unwrap();

                match side {
                    OrderSide::Buy => slippage,   // Pay more
                    OrderSide::Sell => -slippage, // Receive less
                }
            }

            Self::MarketImpact { impact_coefficient } => {
                // Slippage increases with order size relative to volume
                let size_ratio = quantity as f64 / volume.max(1) as f64;
                let impact = size_ratio.sqrt() * impact_coefficient;

                // Add noise (uniform ±20% of impact)
                let noise_factor = rng.gen_range(0.8..1.2);
                let total_impact = impact * noise_factor;

                price * Decimal::from_f64_retain(total_impact).unwrap()
            }
        }
    }
}
```

### Fee Models

```rust
// crates/backtesting/src/fees.rs
#[derive(Debug, Clone)]
pub enum FeeModel {
    /// No fees (unrealistic)
    None,

    /// Fixed per-share (e.g., $0.005 per share)
    PerShare(Decimal),

    /// Percentage of trade value (e.g., 0.1%)
    Percentage(Decimal),

    /// Tiered based on monthly volume
    TieredAlpaca {
        tier: AlpacaTier,
    },

    /// SEC fees + exchange fees
    Realistic {
        sec_fee: Decimal,        // SEC: $0.0000278 per dollar
        finra_taf: Decimal,      // FINRA TAF: $0.000166 per share (sell only)
        exchange_fee: Decimal,   // Exchange: varies
    },
}

#[derive(Debug, Clone, Copy)]
pub enum AlpacaTier {
    Free,      // 0 commission
    Unlimited, // 0 commission
}

impl Default for FeeModel {
    fn default() -> Self {
        // Alpaca-like: zero commission but regulatory fees
        Self::Realistic {
            sec_fee: Decimal::new(278, 10),       // $0.0000278 per dollar
            finra_taf: Decimal::new(166, 9),      // $0.000166 per share
            exchange_fee: Decimal::new(3, 3),     // $0.003 per share
        }
    }
}

impl FeeModel {
    pub fn calculate_fee(
        &self,
        side: OrderSide,
        price: Decimal,
        quantity: u64,
    ) -> Decimal {
        match self {
            Self::None => Decimal::ZERO,

            Self::PerShare(fee_per_share) => {
                *fee_per_share * Decimal::from(quantity)
            }

            Self::Percentage(pct) => {
                let trade_value = price * Decimal::from(quantity);
                trade_value * *pct
            }

            Self::TieredAlpaca { tier } => {
                match tier {
                    AlpacaTier::Free | AlpacaTier::Unlimited => {
                        // Still have regulatory fees
                        self.calculate_regulatory_fees(side, price, quantity)
                    }
                }
            }

            Self::Realistic { sec_fee, finra_taf, exchange_fee } => {
                let trade_value = price * Decimal::from(quantity);

                let mut total_fee = Decimal::ZERO;

                // SEC fee (both buy and sell)
                total_fee += trade_value * *sec_fee;

                // FINRA TAF (sell only)
                if side == OrderSide::Sell {
                    total_fee += *finra_taf * Decimal::from(quantity);
                }

                // Exchange fee
                total_fee += *exchange_fee * Decimal::from(quantity);

                total_fee
            }
        }
    }

    fn calculate_regulatory_fees(&self, side: OrderSide, price: Decimal, quantity: u64) -> Decimal {
        let trade_value = price * Decimal::from(quantity);

        let mut fees = Decimal::ZERO;

        // SEC fee
        fees += trade_value * Decimal::new(278, 10);

        // FINRA TAF (sell only)
        if side == OrderSide::Sell {
            fees += Decimal::new(166, 9) * Decimal::from(quantity);
        }

        fees
    }
}
```

---

## Order Fill Simulation

### Fill Simulator

```rust
// crates/backtesting/src/order_simulator.rs
pub struct OrderSimulator {
    slippage_model: SlippageModel,
    fee_model: FeeModel,
}

impl OrderSimulator {
    pub fn simulate_fill(
        &self,
        signal: &Signal,
        tick: &MarketTick,
        rng: &mut StdRng,
    ) -> Result<Fill, SimulationError> {
        // Check if order would fill at this tick
        if !self.would_fill(signal, tick) {
            return Err(SimulationError::OrderNotFilled);
        }

        // Calculate execution price with slippage
        let base_price = match signal.direction {
            Direction::Long => tick.ask,   // Buy at ask
            Direction::Short => tick.bid,  // Sell at bid
            Direction::Neutral => return Err(SimulationError::InvalidDirection),
        };

        let slippage = self.slippage_model.calculate_slippage(
            match signal.direction {
                Direction::Long => OrderSide::Buy,
                Direction::Short => OrderSide::Sell,
                _ => unreachable!(),
            },
            base_price,
            signal.position_size.to_u64().unwrap(),
            tick.bid,
            tick.ask,
            tick.volume,
            rng,
        );

        let execution_price = match signal.direction {
            Direction::Long => base_price + slippage,   // Pay more
            Direction::Short => base_price - slippage,  // Receive less
            _ => unreachable!(),
        };

        // Calculate fees
        let fees = self.fee_model.calculate_fee(
            match signal.direction {
                Direction::Long => OrderSide::Buy,
                Direction::Short => OrderSide::Sell,
                _ => unreachable!(),
            },
            execution_price,
            signal.position_size.to_u64().unwrap(),
        );

        Ok(Fill {
            timestamp: tick.timestamp,
            symbol: signal.symbol.clone(),
            side: match signal.direction {
                Direction::Long => OrderSide::Buy,
                Direction::Short => OrderSide::Sell,
                _ => unreachable!(),
            },
            quantity: signal.position_size.to_u64().unwrap(),
            price: execution_price,
            fees,
            slippage,
        })
    }

    fn would_fill(&self, signal: &Signal, tick: &MarketTick) -> bool {
        // In backtest, assume market orders fill immediately
        // TODO: Simulate partial fills for large orders
        tick.symbol == signal.symbol
    }
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: u64,
    pub price: Decimal,
    pub fees: Decimal,
    pub slippage: Decimal,
}

impl Fill {
    pub fn total_cost(&self) -> Decimal {
        let notional = self.price * Decimal::from(self.quantity);

        match self.side {
            OrderSide::Buy => notional + self.fees,
            OrderSide::Sell => notional - self.fees,
        }
    }
}
```

---

## Portfolio Tracking

### Portfolio State

```rust
// crates/backtesting/src/portfolio.rs
#[derive(Debug, Clone)]
pub struct Portfolio {
    pub cash: Decimal,
    pub positions: HashMap<String, Position>,
    pub equity_curve: Vec<(DateTime<Utc>, Decimal)>,
    pub trade_history: Vec<Trade>,
    initial_capital: Decimal,
}

#[derive(Debug, Clone)]
pub struct Position {
    pub symbol: String,
    pub quantity: i64,  // Positive = long, negative = short
    pub avg_entry_price: Decimal,
    pub market_value: Decimal,
    pub unrealized_pl: Decimal,
}

#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_time: DateTime<Utc>,
    pub exit_time: DateTime<Utc>,
    pub symbol: String,
    pub side: OrderSide,
    pub quantity: u64,
    pub entry_price: Decimal,
    pub exit_price: Decimal,
    pub pnl: Decimal,
    pub return_pct: f64,
}

impl Portfolio {
    pub fn new(initial_capital: Decimal) -> Self {
        Self {
            cash: initial_capital,
            positions: HashMap::new(),
            equity_curve: vec![(Utc::now(), initial_capital)],
            trade_history: Vec::new(),
            initial_capital,
        }
    }

    pub fn apply_fill(&mut self, fill: Fill) {
        match fill.side {
            OrderSide::Buy => {
                // Deduct cost from cash
                self.cash -= fill.total_cost();

                // Update position
                self.positions.entry(fill.symbol.clone())
                    .and_modify(|pos| {
                        let new_quantity = pos.quantity + fill.quantity as i64;
                        let new_total_cost = pos.avg_entry_price * Decimal::from(pos.quantity.abs())
                            + fill.price * Decimal::from(fill.quantity);

                        pos.quantity = new_quantity;
                        pos.avg_entry_price = new_total_cost / Decimal::from(new_quantity.abs());
                    })
                    .or_insert(Position {
                        symbol: fill.symbol.clone(),
                        quantity: fill.quantity as i64,
                        avg_entry_price: fill.price,
                        market_value: fill.price * Decimal::from(fill.quantity),
                        unrealized_pl: Decimal::ZERO,
                    });
            }

            OrderSide::Sell => {
                // Add proceeds to cash
                self.cash += fill.total_cost();

                // Update position
                if let Some(pos) = self.positions.get_mut(&fill.symbol) {
                    // Record trade if closing position
                    if pos.quantity > 0 {
                        let closed_quantity = fill.quantity.min(pos.quantity as u64);

                        if closed_quantity > 0 {
                            let pnl = (fill.price - pos.avg_entry_price) * Decimal::from(closed_quantity);

                            self.trade_history.push(Trade {
                                entry_time: Utc::now(), // TODO: Track actual entry time
                                exit_time: fill.timestamp,
                                symbol: fill.symbol.clone(),
                                side: OrderSide::Buy,
                                quantity: closed_quantity,
                                entry_price: pos.avg_entry_price,
                                exit_price: fill.price,
                                pnl,
                                return_pct: ((fill.price - pos.avg_entry_price) / pos.avg_entry_price).to_f64().unwrap(),
                            });
                        }
                    }

                    pos.quantity -= fill.quantity as i64;

                    // Remove position if fully closed
                    if pos.quantity == 0 {
                        self.positions.remove(&fill.symbol);
                    }
                }
            }
        }
    }

    pub fn mark_to_market(&mut self, prices: &HashMap<String, Decimal>) {
        let mut total_equity = self.cash;

        for (symbol, position) in &mut self.positions {
            if let Some(market_price) = prices.get(symbol) {
                position.market_value = *market_price * Decimal::from(position.quantity.abs());
                position.unrealized_pl = (market_price - position.avg_entry_price) * Decimal::from(position.quantity);

                total_equity += position.market_value * Decimal::from(position.quantity.signum());
            }
        }

        self.equity_curve.push((Utc::now(), total_equity));
    }

    pub fn total_value(&self) -> Decimal {
        let position_value: Decimal = self.positions.values()
            .map(|pos| pos.market_value * Decimal::from(pos.quantity.signum()))
            .sum();

        self.cash + position_value
    }

    pub fn total_return(&self) -> Decimal {
        (self.total_value() - self.initial_capital) / self.initial_capital
    }

    pub fn win_rate(&self) -> f64 {
        let winning_trades = self.trade_history.iter()
            .filter(|t| t.pnl > Decimal::ZERO)
            .count();

        winning_trades as f64 / self.trade_history.len().max(1) as f64
    }

    pub fn profit_factor(&self) -> f64 {
        let gross_profit: Decimal = self.trade_history.iter()
            .filter(|t| t.pnl > Decimal::ZERO)
            .map(|t| t.pnl)
            .sum();

        let gross_loss: Decimal = self.trade_history.iter()
            .filter(|t| t.pnl < Decimal::ZERO)
            .map(|t| t.pnl.abs())
            .sum();

        if gross_loss == Decimal::ZERO {
            return f64::INFINITY;
        }

        (gross_profit / gross_loss).to_f64().unwrap()
    }

    pub fn equity_curve(&self) -> &[(DateTime<Utc>, Decimal)] {
        &self.equity_curve
    }

    pub fn trade_history(&self) -> &[Trade] {
        &self.trade_history
    }
}
```

---

## Statistical Validation

### Validation Methods

```rust
// crates/backtesting/src/validation.rs
pub struct StatisticalValidator;

impl StatisticalValidator {
    /// Validate backtest results are statistically significant
    pub fn validate(results: &BacktestResults) -> ValidationReport {
        let mut report = ValidationReport::default();

        // 1. Minimum trades threshold
        if results.trades.len() < 30 {
            report.warnings.push(format!(
                "Low sample size: {} trades (recommended: ≥30)",
                results.trades.len()
            ));
        }

        // 2. Sharpe ratio significance test
        let sharpe = results.sharpe_ratio;
        let num_periods = results.equity_curve.len();
        let sharpe_std_error = 1.0 / (num_periods as f64).sqrt();
        let sharpe_z_score = sharpe / sharpe_std_error;

        if sharpe_z_score < 1.96 {  // 95% confidence
            report.warnings.push(format!(
                "Sharpe ratio ({:.2}) not statistically significant at 95% confidence",
                sharpe
            ));
        }

        // 3. Check for overfitting (too-perfect results)
        if results.win_rate > 0.9 && results.sharpe_ratio > 3.0 {
            report.warnings.push(
                "Results may indicate overfitting (win rate >90%, Sharpe >3)".to_string()
            );
        }

        // 4. Drawdown validation
        if results.max_drawdown.abs() > 0.5 {
            report.warnings.push(format!(
                "Large maximum drawdown: {:.1}%",
                results.max_drawdown.to_f64().unwrap() * 100.0
            ));
        }

        // 5. Consistency check (returns should be normally distributed)
        let returns: Vec<f64> = results.trades.iter()
            .map(|t| t.return_pct)
            .collect();

        let (mean, std_dev) = Self::mean_and_std(&returns);
        let skewness = Self::skewness(&returns, mean, std_dev);
        let kurtosis = Self::kurtosis(&returns, mean, std_dev);

        if skewness.abs() > 1.0 {
            report.warnings.push(format!(
                "Returns are skewed (skewness: {:.2}), may not be normally distributed",
                skewness
            ));
        }

        if kurtosis.abs() > 3.0 {
            report.warnings.push(format!(
                "Returns have fat tails (kurtosis: {:.2}), higher risk than normal distribution",
                kurtosis
            ));
        }

        report
    }

    fn mean_and_std(data: &[f64]) -> (f64, f64) {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data.len() as f64;
        (mean, variance.sqrt())
    }

    fn skewness(data: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = data.len() as f64;
        let skew = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n;
        skew
    }

    fn kurtosis(data: &[f64], mean: f64, std_dev: f64) -> f64 {
        let n = data.len() as f64;
        let kurt = data.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n;
        kurt - 3.0  // Excess kurtosis
    }
}

#[derive(Debug, Default)]
pub struct ValidationReport {
    pub warnings: Vec<String>,
    pub is_valid: bool,
}
```

---

## Performance Metrics

### Metrics Calculator

```rust
// crates/backtesting/src/metrics.rs
pub fn calculate_sharpe_ratio(equity_curve: &[(DateTime<Utc>, Decimal)]) -> f64 {
    let returns = calculate_returns(equity_curve);

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let std_dev = {
        let variance = returns.iter()
            .map(|r| (r - mean_return).powi(2))
            .sum::<f64>() / returns.len() as f64;
        variance.sqrt()
    };

    // Annualized Sharpe (assuming daily returns)
    let risk_free_rate = 0.04 / 252.0;  // 4% annual / 252 trading days
    (mean_return - risk_free_rate) / std_dev * (252.0_f64).sqrt()
}

pub fn calculate_sortino_ratio(equity_curve: &[(DateTime<Utc>, Decimal)]) -> f64 {
    let returns = calculate_returns(equity_curve);

    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;

    // Downside deviation (only negative returns)
    let downside_variance = returns.iter()
        .filter(|r| **r < 0.0)
        .map(|r| r.powi(2))
        .sum::<f64>() / returns.len() as f64;

    let downside_std = downside_variance.sqrt();

    // Annualized Sortino
    let risk_free_rate = 0.04 / 252.0;
    (mean_return - risk_free_rate) / downside_std * (252.0_f64).sqrt()
}

pub fn calculate_max_drawdown(equity_curve: &[(DateTime<Utc>, Decimal)]) -> Decimal {
    let mut max_drawdown = Decimal::ZERO;
    let mut peak = equity_curve[0].1;

    for (_timestamp, equity) in equity_curve.iter().skip(1) {
        if *equity > peak {
            peak = *equity;
        }

        let drawdown = (*equity - peak) / peak;
        if drawdown < max_drawdown {
            max_drawdown = drawdown;
        }
    }

    max_drawdown
}

pub fn calculate_calmar_ratio(
    total_return: Decimal,
    max_drawdown: Decimal,
    years: f64,
) -> f64 {
    let annualized_return = (total_return.to_f64().unwrap() / years);
    annualized_return / max_drawdown.abs().to_f64().unwrap()
}

fn calculate_returns(equity_curve: &[(DateTime<Utc>, Decimal)]) -> Vec<f64> {
    equity_curve.windows(2)
        .map(|window| {
            let prev = window[0].1;
            let curr = window[1].1;
            ((curr - prev) / prev).to_f64().unwrap()
        })
        .collect()
}

pub fn calculate_alpha_beta(
    strategy_returns: &[f64],
    benchmark_returns: &[f64],
) -> (f64, f64) {
    // Linear regression: strategy = alpha + beta * benchmark
    let n = strategy_returns.len().min(benchmark_returns.len()) as f64;

    let mean_strategy = strategy_returns.iter().sum::<f64>() / n;
    let mean_benchmark = benchmark_returns.iter().sum::<f64>() / n;

    let covariance = strategy_returns.iter().zip(benchmark_returns.iter())
        .map(|(s, b)| (s - mean_strategy) * (b - mean_benchmark))
        .sum::<f64>() / n;

    let benchmark_variance = benchmark_returns.iter()
        .map(|b| (b - mean_benchmark).powi(2))
        .sum::<f64>() / n;

    let beta = covariance / benchmark_variance;
    let alpha = mean_strategy - beta * mean_benchmark;

    (alpha, beta)
}
```

---

## Report Generation

### HTML Report

```rust
// crates/backtesting/src/report.rs
pub struct ReportGenerator;

impl ReportGenerator {
    pub fn generate_html(results: &BacktestResults, output_path: &Path) -> Result<(), std::io::Error> {
        let template = r#"
<!DOCTYPE html>
<html>
<head>
    <title>Backtest Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .metric { display: inline-block; margin: 10px; padding: 15px; border: 1px solid #ccc; }
        .metric-label { font-weight: bold; }
        .metric-value { font-size: 24px; color: #0066cc; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <h1>Backtest Report</h1>

    <h2>Performance Metrics</h2>
    <div class="metric">
        <div class="metric-label">Total Return</div>
        <div class="metric-value">{total_return:.2}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Sharpe Ratio</div>
        <div class="metric-value">{sharpe_ratio:.2}</div>
    </div>
    <div class="metric">
        <div class="metric-label">Max Drawdown</div>
        <div class="metric-value">{max_drawdown:.2}%</div>
    </div>
    <div class="metric">
        <div class="metric-label">Win Rate</div>
        <div class="metric-value">{win_rate:.1}%</div>
    </div>

    <h2>Equity Curve</h2>
    <canvas id="equityCurve" width="800" height="400"></canvas>

    <h2>Trade History</h2>
    <table>
        <tr>
            <th>Entry Time</th>
            <th>Exit Time</th>
            <th>Symbol</th>
            <th>Side</th>
            <th>Quantity</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>P&L</th>
            <th>Return %</th>
        </tr>
        {trade_rows}
    </table>

    <script>
        // Draw equity curve using Canvas API
        const canvas = document.getElementById('equityCurve');
        const ctx = canvas.getContext('2d');

        const equityCurve = {equity_curve_json};

        // ... chart drawing code ...
    </script>
</body>
</html>
        "#;

        let trade_rows: String = results.trades.iter()
            .map(|trade| format!(
                "<tr><td>{}</td><td>{}</td><td>{}</td><td>{:?}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{:.2}%</td></tr>",
                trade.entry_time.format("%Y-%m-%d %H:%M"),
                trade.exit_time.format("%Y-%m-%d %H:%M"),
                trade.symbol,
                trade.side,
                trade.quantity,
                trade.entry_price,
                trade.exit_price,
                trade.pnl,
                trade.return_pct * 100.0
            ))
            .collect();

        let equity_curve_json = serde_json::to_string(&results.equity_curve)?;

        let html = template
            .replace("{total_return}", &format!("{:.2}", results.total_return.to_f64().unwrap() * 100.0))
            .replace("{sharpe_ratio}", &format!("{:.2}", results.sharpe_ratio))
            .replace("{max_drawdown}", &format!("{:.2}", results.max_drawdown.to_f64().unwrap() * 100.0))
            .replace("{win_rate}", &format!("{:.1}", results.win_rate * 100.0))
            .replace("{trade_rows}", &trade_rows)
            .replace("{equity_curve_json}", &equity_curve_json);

        std::fs::write(output_path, html)?;

        Ok(())
    }

    pub fn generate_json(results: &BacktestResults, output_path: &Path) -> Result<(), std::io::Error> {
        let json = serde_json::to_string_pretty(results)?;
        std::fs::write(output_path, json)?;
        Ok(())
    }
}
```

---

## Comparison with Live

### Parity Validation

```rust
// tests/parity_test.rs
#[tokio::test]
async fn test_backtest_parity_with_python() {
    // Load Python results
    let python_results: PythonBacktestResults =
        serde_json::from_str(&std::fs::read_to_string("tests/fixtures/python_backtest_results.json").unwrap())
            .unwrap();

    // Run Rust backtest with same config
    let config = BacktestConfig {
        start_date: python_results.config.start_date,
        end_date: python_results.config.end_date,
        initial_capital: python_results.config.initial_capital,
        random_seed: python_results.config.random_seed,  // Same seed!
        ..Default::default()
    };

    let mut engine = BacktestEngine::new(config);
    let rust_results = engine.run().await.unwrap();

    // Assert results match within tolerance
    assert_approx_eq!(
        rust_results.total_return.to_f64().unwrap(),
        python_results.total_return,
        1e-6
    );

    assert_approx_eq!(
        rust_results.sharpe_ratio,
        python_results.sharpe_ratio,
        1e-4
    );

    assert_eq!(
        rust_results.trades.len(),
        python_results.num_trades
    );
}
```

---

## Parameter Optimization

### Grid Search

```rust
// crates/backtesting/src/optimization.rs
pub struct ParameterOptimizer {
    base_config: BacktestConfig,
}

impl ParameterOptimizer {
    pub async fn grid_search(
        &self,
        param_grid: HashMap<String, Vec<f64>>,
        metric: OptimizationMetric,
    ) -> Vec<OptimizationResult> {
        let mut results = Vec::new();

        // Generate all combinations
        let combinations = self.generate_combinations(&param_grid);

        for params in combinations {
            let mut config = self.base_config.clone();

            // Apply parameters
            // TODO: Map params to config fields

            let mut engine = BacktestEngine::new(config);
            let backtest_results = engine.run().await.unwrap();

            let score = match metric {
                OptimizationMetric::Sharpe => backtest_results.sharpe_ratio,
                OptimizationMetric::Sortino => backtest_results.sortino_ratio,
                OptimizationMetric::Calmar => backtest_results.total_return.to_f64().unwrap()
                    / backtest_results.max_drawdown.abs().to_f64().unwrap(),
            };

            results.push(OptimizationResult {
                params: params.clone(),
                score,
                results: backtest_results,
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        results
    }

    fn generate_combinations(&self, param_grid: &HashMap<String, Vec<f64>>) -> Vec<HashMap<String, f64>> {
        // Cartesian product of all parameter values
        todo!()
    }
}

pub enum OptimizationMetric {
    Sharpe,
    Sortino,
    Calmar,
}

pub struct OptimizationResult {
    pub params: HashMap<String, f64>,
    pub score: f64,
    pub results: BacktestResults,
}
```

---

## Walk-Forward Analysis

### Walk-Forward Test

```rust
// crates/backtesting/src/walk_forward.rs
pub struct WalkForwardAnalysis {
    in_sample_period: Duration,
    out_sample_period: Duration,
}

impl WalkForwardAnalysis {
    pub async fn run(&self, config: BacktestConfig) -> WalkForwardResults {
        let mut results = WalkForwardResults::default();

        let mut current_start = config.start_date;

        while current_start < config.end_date {
            let in_sample_end = current_start + self.in_sample_period;
            let out_sample_end = in_sample_end + self.out_sample_period;

            // Optimize on in-sample period
            let optimizer = ParameterOptimizer {
                base_config: BacktestConfig {
                    start_date: current_start,
                    end_date: in_sample_end,
                    ..config.clone()
                },
            };

            let best_params = optimizer.grid_search(
                /* param grid */,
                OptimizationMetric::Sharpe,
            ).await[0].params.clone();

            // Test on out-of-sample period
            let out_sample_config = BacktestConfig {
                start_date: in_sample_end,
                end_date: out_sample_end,
                // Apply best params
                ..config.clone()
            };

            let mut engine = BacktestEngine::new(out_sample_config);
            let out_sample_results = engine.run().await.unwrap();

            results.periods.push(WalkForwardPeriod {
                in_sample_start: current_start,
                in_sample_end,
                out_sample_end,
                best_params,
                out_sample_sharpe: out_sample_results.sharpe_ratio,
            });

            current_start = out_sample_end;
        }

        results
    }
}
```

---

## Monte Carlo Simulation

### Monte Carlo Bootstrap

```rust
// crates/backtesting/src/monte_carlo.rs
pub struct MonteCarloSimulator {
    num_simulations: usize,
}

impl MonteCarloSimulator {
    pub fn run(&self, trade_history: &[Trade]) -> MonteCarloResults {
        let mut rng = rand::thread_rng();
        let mut simulation_results = Vec::new();

        for _ in 0..self.num_simulations {
            // Randomly sample trades with replacement
            let sampled_trades: Vec<_> = (0..trade_history.len())
                .map(|_| {
                    let idx = rng.gen_range(0..trade_history.len());
                    trade_history[idx].clone()
                })
                .collect();

            // Calculate metrics for this simulation
            let total_return = sampled_trades.iter()
                .map(|t| t.return_pct)
                .sum::<f64>();

            simulation_results.push(total_return);
        }

        // Calculate confidence intervals
        simulation_results.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let percentile_5 = simulation_results[self.num_simulations * 5 / 100];
        let percentile_95 = simulation_results[self.num_simulations * 95 / 100];
        let median = simulation_results[self.num_simulations / 2];

        MonteCarloResults {
            simulations: simulation_results,
            median,
            percentile_5,
            percentile_95,
        }
    }
}
```

---

## Troubleshooting

### Common Issues

#### 1. Non-deterministic Results

**Symptom:** Different results on each run

**Solution:**
- Ensure `random_seed` is set
- Verify event ordering in market tape
- Check for race conditions in async code

#### 2. Unrealistic Returns

**Symptom:** Sharpe ratio >5, win rate >95%

**Diagnosis:** Likely overfitting or look-ahead bias

**Solution:**
- Add slippage and fees
- Check for data leakage
- Use walk-forward validation

#### 3. Performance Slowdown

**Symptom:** Backtest takes hours for 1 year of data

**Solution:**
```rust
// Use Polars for vectorized operations
let df = df.lazy()
    .with_column(col("returns").pct_change())
    .collect()?;

// Parallel strategy execution
let signals = strategies.par_iter()
    .map(|s| s.process(&data))
    .collect();
```

---

## Acceptance Criteria

- [ ] Deterministic backtests (seeded RNG)
- [ ] Event-sourced market tape format
- [ ] Realistic slippage and fee models
- [ ] Portfolio tracking with mark-to-market
- [ ] Statistical validation (Sharpe, Sortino, drawdown)
- [ ] Performance metrics (10+ metrics)
- [ ] HTML and JSON reports
- [ ] Parity with Python (within 1e-6)
- [ ] Parameter optimization (grid search)
- [ ] Walk-forward analysis
- [ ] Monte Carlo simulation

---

## Cross-References

- **Exchange Adapters:** [17_Exchange_Adapters_and_Data_Pipeline.md](./17_Exchange_Adapters_and_Data_Pipeline.md) - Data replay
- **Testing:** [13_Tests_Benchmarks_CI.md](./13_Tests_Benchmarks_CI.md) - Parity tests
- **Strategies:** [06_Strategy_and_Sublinear_Solvers.md](./06_Strategy_and_Sublinear_Solvers.md) - Algorithm design
- **Architecture:** [03_Architecture.md](./03_Architecture.md) - Module design

---

**Document Version:** 1.0.0
**Last Updated:** 2025-11-12
**Owner:** Quantitative Developer
**Status:** Complete
**Next Review:** 2025-11-19
