# Backtesting Framework Architecture

## Executive Summary

The `hyperphysics-backtest` crate provides an event-driven backtesting engine that simulates trading strategies under realistic market conditions. Integration with consciousness metrics (Φ) and causal influence (CI) enables regime-aware strategy adaptation.

## 1. Module Structure

```
hyperphysics-backtest/
├── src/
│   ├── engine/
│   │   ├── mod.rs              # Event-driven simulation engine
│   │   ├── event_queue.rs      # Priority queue for market events
│   │   ├── executor.rs         # Order execution simulator
│   │   └── clock.rs            # Time management
│   ├── data/
│   │   ├── mod.rs
│   │   ├── replay.rs           # Historical data replay
│   │   └── provider.rs         # Data source interface
│   ├── strategy/
│   │   ├── mod.rs
│   │   ├── base.rs             # Strategy trait
│   │   ├── regime_aware.rs     # Consciousness-based adaptation
│   │   └── examples/           # Example strategies
│   ├── execution/
│   │   ├── mod.rs
│   │   ├── slippage.rs         # Slippage models
│   │   ├── costs.rs            # Transaction cost models
│   │   └── fills.rs            # Fill probability models
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── performance.rs      # Sharpe, Sortino, etc.
│   │   ├── drawdown.rs         # Drawdown analysis
│   │   └── thermodynamic.rs    # Entropy-based metrics
│   └── lib.rs
```

## 2. Core Type Definitions

### 2.1 Event-Driven Architecture

```rust
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use chrono::{DateTime, Utc};

/// Market event types
#[derive(Debug, Clone)]
pub enum Event {
    Tick(TickEvent),
    Bar(BarEvent),
    Signal(SignalEvent),
    Order(OrderEvent),
    Fill(FillEvent),
}

impl Event {
    pub fn timestamp(&self) -> DateTime<Utc> {
        match self {
            Event::Tick(e) => e.timestamp,
            Event::Bar(e) => e.timestamp,
            Event::Signal(e) => e.timestamp,
            Event::Order(e) => e.timestamp,
            Event::Fill(e) => e.timestamp,
        }
    }
}

/// Prioritized event wrapper for event queue
#[derive(Debug, Clone)]
struct PrioritizedEvent {
    event: Event,
    priority: u64,  // Microseconds since epoch
}

impl PartialEq for PrioritizedEvent {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for PrioritizedEvent {}

impl PartialOrd for PrioritizedEvent {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PrioritizedEvent {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap
        other.priority.cmp(&self.priority)
    }
}

/// Event queue with chronological ordering
pub struct EventQueue {
    heap: BinaryHeap<PrioritizedEvent>,
}

impl EventQueue {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn push(&mut self, event: Event) {
        let timestamp = event.timestamp();
        let priority = timestamp.timestamp_micros() as u64;

        self.heap.push(PrioritizedEvent { event, priority });
    }

    pub fn pop(&mut self) -> Option<Event> {
        self.heap.pop().map(|pe| pe.event)
    }

    pub fn peek(&self) -> Option<&Event> {
        self.heap.peek().map(|pe| &pe.event)
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

#[derive(Debug, Clone)]
pub struct TickEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct BarEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

#[derive(Debug, Clone)]
pub struct SignalEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub signal_type: SignalType,
    pub strength: f64,  // [-1, 1]
    pub metadata: HashMap<String, f64>,
}

#[derive(Debug, Clone)]
pub enum SignalType {
    Long,
    Short,
    Exit,
    Reduce,
}

#[derive(Debug, Clone)]
pub struct OrderEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub order_type: OrderType,
    pub quantity: f64,
    pub direction: Direction,
    pub limit_price: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum OrderType {
    Market,
    Limit,
    Stop,
    StopLimit,
}

#[derive(Debug, Clone)]
pub enum Direction {
    Long,
    Short,
}

#[derive(Debug, Clone)]
pub struct FillEvent {
    pub timestamp: DateTime<Utc>,
    pub symbol: String,
    pub quantity: f64,
    pub fill_price: f64,
    pub commission: f64,
    pub slippage: f64,
}
```

### 2.2 Backtesting Engine

```rust
use std::sync::Arc;
use async_trait::async_trait;

/// Main backtesting engine
pub struct BacktestEngine {
    /// Event queue
    events: EventQueue,

    /// Data provider
    data_provider: Arc<dyn DataProvider>,

    /// Strategy implementation
    strategy: Arc<dyn Strategy>,

    /// Order executor
    executor: OrderExecutor,

    /// Portfolio tracker
    portfolio: Portfolio,

    /// Performance metrics
    metrics: PerformanceMetrics,

    /// Current simulation time
    current_time: DateTime<Utc>,

    /// Simulation end time
    end_time: DateTime<Utc>,
}

impl BacktestEngine {
    pub fn new(
        data_provider: Arc<dyn DataProvider>,
        strategy: Arc<dyn Strategy>,
        initial_capital: f64,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Self {
        Self {
            events: EventQueue::new(),
            data_provider,
            strategy,
            executor: OrderExecutor::new(),
            portfolio: Portfolio::new(initial_capital),
            metrics: PerformanceMetrics::new(),
            current_time: start_time,
            end_time,
        }
    }

    /// Run backtest simulation
    pub async fn run(&mut self) -> Result<BacktestResults, BacktestError> {
        // Initialize data stream
        let mut data_stream = self.data_provider
            .stream_data(self.current_time, self.end_time)
            .await?;

        // Main event loop
        while let Some(market_event) = data_stream.next().await {
            // Update current time
            self.current_time = market_event.timestamp();

            // Add market event to queue
            self.events.push(market_event);

            // Process all events at current timestamp
            while let Some(event) = self.events.peek() {
                if event.timestamp() > self.current_time {
                    break;
                }

                let event = self.events.pop().unwrap();

                match event {
                    Event::Tick(tick) => self.handle_tick(tick).await?,
                    Event::Bar(bar) => self.handle_bar(bar).await?,
                    Event::Signal(signal) => self.handle_signal(signal).await?,
                    Event::Order(order) => self.handle_order(order).await?,
                    Event::Fill(fill) => self.handle_fill(fill).await?,
                }
            }

            // Update portfolio value
            self.portfolio.update_value(self.current_time);

            // Record metrics
            self.metrics.record(
                self.current_time,
                self.portfolio.total_value(),
                &self.portfolio,
            );
        }

        // Calculate final results
        Ok(self.calculate_results())
    }

    async fn handle_tick(&mut self, tick: TickEvent) -> Result<(), BacktestError> {
        // Update portfolio prices
        self.portfolio.update_price(&tick.symbol, tick.price);

        // Generate signals from strategy
        if let Some(signal) = self.strategy.on_tick(&tick, &self.portfolio).await? {
            self.events.push(Event::Signal(signal));
        }

        Ok(())
    }

    async fn handle_bar(&mut self, bar: BarEvent) -> Result<(), BacktestError> {
        // Update portfolio with bar data
        self.portfolio.update_price(&bar.symbol, bar.close);

        // Generate signals
        if let Some(signal) = self.strategy.on_bar(&bar, &self.portfolio).await? {
            self.events.push(Event::Signal(signal));
        }

        Ok(())
    }

    async fn handle_signal(&mut self, signal: SignalEvent) -> Result<(), BacktestError> {
        // Convert signal to order
        let order = self.strategy.signal_to_order(signal, &self.portfolio)?;
        self.events.push(Event::Order(order));

        Ok(())
    }

    async fn handle_order(&mut self, order: OrderEvent) -> Result<(), BacktestError> {
        // Execute order with slippage and costs
        if let Some(fill) = self.executor.execute(
            order,
            self.current_time,
            &self.portfolio,
        )? {
            self.events.push(Event::Fill(fill));
        }

        Ok(())
    }

    async fn handle_fill(&mut self, fill: FillEvent) -> Result<(), BacktestError> {
        // Update portfolio with fill
        self.portfolio.apply_fill(fill);

        Ok(())
    }

    fn calculate_results(&self) -> BacktestResults {
        BacktestResults {
            total_return: self.metrics.total_return(),
            sharpe_ratio: self.metrics.sharpe_ratio(),
            sortino_ratio: self.metrics.sortino_ratio(),
            max_drawdown: self.metrics.max_drawdown(),
            calmar_ratio: self.metrics.calmar_ratio(),
            trades: self.metrics.total_trades(),
            win_rate: self.metrics.win_rate(),
            profit_factor: self.metrics.profit_factor(),
            entropy: self.portfolio.entropy,
            free_energy: self.portfolio.free_energy,
        }
    }
}

#[derive(Debug, Clone)]
pub struct BacktestResults {
    pub total_return: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown: f64,
    pub calmar_ratio: f64,
    pub trades: usize,
    pub win_rate: f64,
    pub profit_factor: f64,
    pub entropy: f64,
    pub free_energy: f64,
}
```

## 3. Consciousness-Based Strategy

### 3.1 Regime-Aware Trading

```rust
use hyperphysics_metrics::{PhiCalculator, CIAnalyzer, MarketRegime};

/// Strategy that adapts based on market consciousness metrics
pub struct RegimeAwareStrategy {
    /// Φ (Phi) consciousness calculator
    phi_calculator: PhiCalculator,

    /// Causal Influence analyzer
    ci_analyzer: CIAnalyzer,

    /// Current detected regime
    regime: MarketRegime,

    /// Regime-specific parameters
    regime_params: HashMap<MarketRegime, StrategyParams>,

    /// Historical window for calculations
    window_size: usize,
}

#[derive(Debug, Clone)]
pub struct StrategyParams {
    pub position_size: f64,
    pub stop_loss: f64,
    pub take_profit: f64,
    pub rebalance_threshold: f64,
}

#[async_trait]
impl Strategy for RegimeAwareStrategy {
    async fn on_bar(
        &self,
        bar: &BarEvent,
        portfolio: &Portfolio,
    ) -> Result<Option<SignalEvent>, BacktestError> {
        // Calculate consciousness metrics
        let phi = self.phi_calculator.calculate(&bar)?;
        let ci = self.ci_analyzer.analyze(&bar)?;

        // Detect regime
        let new_regime = self.detect_regime(phi, ci);

        if new_regime != self.regime {
            // Regime change detected - adjust strategy
            println!(
                "Regime change: {:?} -> {:?} (Φ={:.3}, CI={:.3})",
                self.regime, new_regime, phi, ci
            );

            self.regime = new_regime;
        }

        // Get regime-specific parameters
        let params = &self.regime_params[&self.regime];

        // Generate signal based on regime
        match self.regime {
            MarketRegime::Bull => self.bull_signal(bar, portfolio, params),
            MarketRegime::Bear => self.bear_signal(bar, portfolio, params),
            MarketRegime::Bubble => self.bubble_signal(bar, portfolio, params),
            MarketRegime::Correction => self.correction_signal(bar, portfolio, params),
        }
    }
}

impl RegimeAwareStrategy {
    /// Detect market regime from consciousness metrics
    fn detect_regime(&self, phi: f64, ci: f64) -> MarketRegime {
        // High Φ + High CI = Bubble (unstable coherence)
        if phi > 0.7 && ci > 0.6 {
            return MarketRegime::Bubble;
        }

        // Low Φ + High CI = Correction (breakdown of order)
        if phi < 0.3 && ci > 0.5 {
            return MarketRegime::Correction;
        }

        // High Φ + Moderate CI = Bull (stable growth)
        if phi > 0.5 && ci > 0.3 && ci < 0.6 {
            return MarketRegime::Bull;
        }

        // Low Φ + Low CI = Bear (declining integration)
        MarketRegime::Bear
    }

    fn bull_signal(
        &self,
        bar: &BarEvent,
        portfolio: &Portfolio,
        params: &StrategyParams,
    ) -> Result<Option<SignalEvent>, BacktestError> {
        // Aggressive long positions in bull regime
        if !portfolio.has_position(&bar.symbol) {
            Ok(Some(SignalEvent {
                timestamp: bar.timestamp,
                symbol: bar.symbol.clone(),
                signal_type: SignalType::Long,
                strength: params.position_size,
                metadata: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    fn bubble_signal(
        &self,
        bar: &BarEvent,
        portfolio: &Portfolio,
        params: &StrategyParams,
    ) -> Result<Option<SignalEvent>, BacktestError> {
        // Reduce exposure in bubble regime
        if portfolio.has_position(&bar.symbol) {
            Ok(Some(SignalEvent {
                timestamp: bar.timestamp,
                symbol: bar.symbol.clone(),
                signal_type: SignalType::Reduce,
                strength: 0.5,  // Cut position by 50%
                metadata: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    fn correction_signal(
        &self,
        bar: &BarEvent,
        portfolio: &Portfolio,
        params: &StrategyParams,
    ) -> Result<Option<SignalEvent>, BacktestError> {
        // Exit all positions during correction
        if portfolio.has_position(&bar.symbol) {
            Ok(Some(SignalEvent {
                timestamp: bar.timestamp,
                symbol: bar.symbol.clone(),
                signal_type: SignalType::Exit,
                strength: 1.0,
                metadata: HashMap::new(),
            }))
        } else {
            Ok(None)
        }
    }

    fn bear_signal(
        &self,
        bar: &BarEvent,
        portfolio: &Portfolio,
        params: &StrategyParams,
    ) -> Result<Option<SignalEvent>, BacktestError> {
        // Conservative short positions or cash
        Ok(None)
    }
}
```

## 4. Execution Simulation

### 4.1 Slippage Models

```rust
/// Slippage estimation based on volume and volatility
pub struct SlippageModel {
    /// Base slippage (bps)
    base_slippage: f64,

    /// Volume impact factor
    volume_impact: f64,

    /// Volatility impact factor
    volatility_impact: f64,
}

impl SlippageModel {
    pub fn calculate(
        &self,
        order: &OrderEvent,
        bar: &BarEvent,
        portfolio: &Portfolio,
    ) -> f64 {
        let order_value = order.quantity * bar.close;
        let volume_ratio = order_value / (bar.volume * bar.close);

        // Volatility (daily range / close)
        let volatility = (bar.high - bar.low) / bar.close;

        // Total slippage (bps)
        let slippage_bps = self.base_slippage
            + self.volume_impact * volume_ratio.sqrt() * 10000.0
            + self.volatility_impact * volatility * 10000.0;

        // Convert to absolute price impact
        let slippage = (slippage_bps / 10000.0) * bar.close;

        match order.direction {
            Direction::Long => slippage,   // Pay more when buying
            Direction::Short => -slippage, // Receive less when selling
        }
    }
}
```

### 4.2 Transaction Cost Models

```rust
/// Commission and fee structure
pub struct TransactionCostModel {
    /// Fixed commission per trade
    fixed_commission: f64,

    /// Variable commission (bps)
    variable_commission_bps: f64,

    /// Exchange fees (bps)
    exchange_fee_bps: f64,

    /// Minimum commission
    minimum_commission: f64,
}

impl TransactionCostModel {
    pub fn calculate(&self, order: &OrderEvent, fill_price: f64) -> f64 {
        let trade_value = order.quantity * fill_price;

        let variable_cost = trade_value * self.variable_commission_bps / 10000.0;
        let exchange_fees = trade_value * self.exchange_fee_bps / 10000.0;

        let total_commission = self.fixed_commission + variable_cost + exchange_fees;

        total_commission.max(self.minimum_commission)
    }
}
```

### 4.3 Order Executor

```rust
pub struct OrderExecutor {
    slippage_model: SlippageModel,
    cost_model: TransactionCostModel,
}

impl OrderExecutor {
    pub fn execute(
        &self,
        order: OrderEvent,
        current_time: DateTime<Utc>,
        portfolio: &Portfolio,
    ) -> Result<Option<FillEvent>, BacktestError> {
        // Get current market data
        let bar = portfolio.get_latest_bar(&order.symbol)?;

        // Check if order can be filled
        if !self.can_fill(&order, &bar) {
            return Ok(None);
        }

        // Calculate fill price
        let base_price = match order.order_type {
            OrderType::Market => bar.close,
            OrderType::Limit => order.limit_price.unwrap(),
            _ => bar.close,
        };

        // Apply slippage
        let slippage = self.slippage_model.calculate(&order, &bar, portfolio);
        let fill_price = base_price + slippage;

        // Calculate commission
        let commission = self.cost_model.calculate(&order, fill_price);

        Ok(Some(FillEvent {
            timestamp: current_time,
            symbol: order.symbol,
            quantity: order.quantity,
            fill_price,
            commission,
            slippage: slippage.abs(),
        }))
    }

    fn can_fill(&self, order: &OrderEvent, bar: &BarEvent) -> bool {
        match order.order_type {
            OrderType::Market => true,
            OrderType::Limit => {
                let limit = order.limit_price.unwrap();
                match order.direction {
                    Direction::Long => bar.low <= limit,
                    Direction::Short => bar.high >= limit,
                }
            },
            _ => true,
        }
    }
}
```

## 5. Performance Metrics

```rust
pub struct PerformanceMetrics {
    /// Equity curve
    equity_curve: Vec<(DateTime<Utc>, f64)>,

    /// Daily returns
    returns: Vec<f64>,

    /// Trade history
    trades: Vec<Trade>,

    /// Initial capital
    initial_capital: f64,
}

impl PerformanceMetrics {
    pub fn total_return(&self) -> f64 {
        let final_value = self.equity_curve.last().unwrap().1;
        (final_value - self.initial_capital) / self.initial_capital
    }

    pub fn sharpe_ratio(&self) -> f64 {
        let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;
        let std_dev = self.calculate_std_dev(&self.returns);

        // Annualized Sharpe (assuming daily returns)
        (mean_return / std_dev) * (252.0_f64).sqrt()
    }

    pub fn sortino_ratio(&self) -> f64 {
        let mean_return = self.returns.iter().sum::<f64>() / self.returns.len() as f64;

        // Downside deviation (only negative returns)
        let downside_returns: Vec<f64> = self.returns.iter()
            .filter(|&&r| r < 0.0)
            .copied()
            .collect();

        let downside_std = self.calculate_std_dev(&downside_returns);

        (mean_return / downside_std) * (252.0_f64).sqrt()
    }

    pub fn max_drawdown(&self) -> f64 {
        let mut peak = self.initial_capital;
        let mut max_dd = 0.0;

        for (_, value) in &self.equity_curve {
            if *value > peak {
                peak = *value;
            }

            let drawdown = (peak - value) / peak;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }

        max_dd
    }

    pub fn calmar_ratio(&self) -> f64 {
        let total_return = self.total_return();
        let max_dd = self.max_drawdown();

        if max_dd > 0.0 {
            total_return / max_dd
        } else {
            f64::INFINITY
        }
    }

    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|&v| (v - mean).powi(2))
            .sum::<f64>() / values.len() as f64;

        variance.sqrt()
    }
}
```

## 6. Academic References

1. Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.

2. Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.

3. Kissell, R. (2013). *The Science of Algorithmic Trading and Portfolio Management*. Academic Press.

4. Bailey, D. H., et al. (2014). *Pseudo-Mathematics and Financial Charlatanism*. The Journal of Portfolio Management, 40(3), 1-9.

5. Harvey, C. R., & Liu, Y. (2015). *Backtesting*. The Journal of Portfolio Management, 42(1), 13-28.
