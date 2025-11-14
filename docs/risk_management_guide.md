# Risk Management Module Guide

## Overview

The HyperPhysics risk management module provides comprehensive risk analysis and position sizing capabilities for trading systems. It integrates seamlessly with the backtesting framework and live trading systems.

## Features

### 1. Position Sizing Strategies

#### Fixed Amount
```rust
use hyperphysics_market::risk::{RiskManager, RiskConfig, PositionSizingStrategy};

let strategy = PositionSizingStrategy::Fixed { amount: 10000.0 };
let shares = risk_manager.calculate_position_size(strategy, 150.0);
```

#### Percentage of Capital
```rust
let strategy = PositionSizingStrategy::Percentage { percentage: 0.1 }; // 10%
let shares = risk_manager.calculate_position_size(strategy, 150.0);
```

#### Kelly Criterion
Optimal position sizing based on win rate and win/loss ratio:
```rust
let strategy = PositionSizingStrategy::Kelly {
    win_rate: 0.6,        // 60% win rate
    win_loss_ratio: 2.0   // 2:1 reward/risk ratio
};
let shares = risk_manager.calculate_position_size(strategy, 150.0);
```

Formula: `f = (p * b - q) / b`
- p = win rate
- q = loss rate (1 - p)
- b = win/loss ratio

#### Volatility-Based (ATR)
Position sizing based on Average True Range:
```rust
let strategy = PositionSizingStrategy::Volatility {
    atr: 3.0,               // $3 ATR
    risk_per_trade: 1000.0  // Risk $1000 per trade
};
let shares = risk_manager.calculate_position_size(strategy, 150.0);
```

### 2. Stop Loss Management

#### Fixed Percentage Stop
```rust
use hyperphysics_market::risk::StopLossType;

let stop_type = StopLossType::Fixed { percentage: 0.02 }; // 2% stop
let stop_price = risk_manager.calculate_stop_loss(150.0, stop_type, true);
```

#### Trailing Stop
```rust
let stop_type = StopLossType::Trailing {
    percentage: 0.05,           // Trail at 5%
    activation_profit: 0.1      // Activate after 10% profit
};
let stop_price = risk_manager.calculate_stop_loss(150.0, stop_type, true);

// Update trailing stop as price moves
let new_stop = risk_manager.update_trailing_stop(&position, stop_type, true);
```

#### ATR-Based Stop
```rust
let stop_type = StopLossType::AtrBased {
    atr: 3.0,
    multiplier: 2.0  // 2x ATR
};
let stop_price = risk_manager.calculate_stop_loss(150.0, stop_type, true);
```

### 3. Risk Metrics

#### Value at Risk (VaR)
Maximum expected loss at a given confidence level:
```rust
let var_95 = risk_manager.calculate_var(0.95); // 95% confidence
```

#### Conditional VaR (CVaR / Expected Shortfall)
Expected loss beyond the VaR threshold:
```rust
let cvar_95 = risk_manager.calculate_cvar(0.95);
```

#### Sharpe Ratio
Risk-adjusted return metric:
```rust
let sharpe = risk_manager.calculate_sharpe_ratio(252); // Annual
```

Formula: `(Return - RiskFreeRate) / StandardDeviation`

#### Sortino Ratio
Similar to Sharpe but only penalizes downside volatility:
```rust
let sortino = risk_manager.calculate_sortino_ratio(252);
```

Formula: `(Return - RiskFreeRate) / DownsideDeviation`

### 4. Risk Limits

Configure maximum risk parameters:
```rust
let config = RiskConfig::default()
    .with_max_position_size(0.15)    // 15% max per position
    .with_max_drawdown(0.25)         // 25% max drawdown
    .with_max_daily_loss(0.05)       // 5% max daily loss
    .with_max_leverage(2.0)          // 2x max leverage
    .with_risk_free_rate(0.03);      // 3% annual risk-free rate

let mut risk_manager = RiskManager::new(100000.0, config);
```

Check for violations:
```rust
let violations = risk_manager.check_risk_limits();
for violation in violations {
    match violation {
        RiskViolation::MaxDrawdown { current, limit } => {
            println!("Drawdown {:.2}% exceeds limit {:.2}%", current * 100.0, limit * 100.0);
        }
        RiskViolation::MaxDailyLoss { current, limit } => {
            println!("Daily loss {:.2}% exceeds limit {:.2}%", current * 100.0, limit * 100.0);
        }
        _ => {}
    }
}
```

### 5. Portfolio Risk Analysis

#### Diversification Score
Measures portfolio concentration (1.0 = perfectly diversified, 0.0 = concentrated):
```rust
let div_score = risk_manager.calculate_diversification_score();
```

Uses Herfindahl index: `1 - Σ(weight_i²)`

#### Leverage Calculation
```rust
let leverage = risk_manager.calculate_leverage();
let margin_req = risk_manager.calculate_margin_requirement(2.0); // For 2x leverage
```

#### Portfolio Metrics
```rust
let metrics = risk_manager.get_metrics();
println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
println!("Sortino Ratio: {:.2}", metrics.sortino_ratio);
println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
println!("VaR (95%): ${:.2}", metrics.var_95);
println!("CVaR (95%): ${:.2}", metrics.cvar_95);
println!("Diversification: {:.2}", metrics.diversification_score);
```

### 6. Position Management

#### Adding Positions
```rust
use hyperphysics_market::risk::Position;
use chrono::Utc;

let position = Position {
    symbol: "AAPL".to_string(),
    entry_price: 150.0,
    current_price: 160.0,
    quantity: 100.0,
    stop_loss: Some(147.0),
    take_profit: Some(165.0),
    entry_time: Utc::now(),
};

risk_manager.add_position(position);
```

#### Updating Positions
```rust
risk_manager.update_position_price("AAPL", 162.0);
```

#### Removing Positions
```rust
let position = risk_manager.remove_position("AAPL");
```

#### Position Analysis
```rust
let pnl = position.unrealized_pnl();
let pnl_pct = position.unrealized_pnl_pct();
let market_value = position.market_value();
```

## Integration with Backtesting

```rust
use hyperphysics_market::{RiskManager, RiskConfig, PositionSizingStrategy};
use hyperphysics_market::backtest::{BacktestEngine, Strategy};

struct TrendFollowingStrategy {
    risk_manager: RiskManager,
}

impl Strategy for TrendFollowingStrategy {
    fn on_bar(&mut self, bar: &Bar) -> Option<Signal> {
        // Calculate position size based on strategy
        let strategy = PositionSizingStrategy::Kelly {
            win_rate: 0.6,
            win_loss_ratio: 2.0
        };

        let shares = self.risk_manager.calculate_position_size(
            strategy,
            bar.close
        );

        // Check risk limits before trading
        let violations = self.risk_manager.check_risk_limits();
        if !violations.is_empty() {
            return None; // Don't trade if limits violated
        }

        // Calculate stop loss
        let stop_type = StopLossType::AtrBased {
            atr: calculate_atr(&bars),
            multiplier: 2.0
        };
        let stop_price = self.risk_manager.calculate_stop_loss(
            bar.close,
            stop_type,
            true
        );

        Some(Signal::Buy {
            quantity: shares,
            stop_loss: Some(stop_price),
            take_profit: None,
        })
    }
}
```

## Best Practices

### 1. Position Sizing
- Never risk more than 1-2% of capital per trade
- Use Kelly Criterion conservatively (half or quarter Kelly)
- Account for correlation between positions
- Adjust size based on volatility

### 2. Stop Losses
- Always use stop losses
- Place stops at technically significant levels
- Use ATR-based stops for volatility adjustment
- Implement trailing stops to protect profits

### 3. Risk Limits
- Set portfolio-level risk limits
- Monitor drawdown in real-time
- Implement daily loss limits
- Cap leverage appropriately

### 4. Diversification
- Maintain diversification score above 0.5
- Limit correlation between positions
- Don't exceed 10-15% per position
- Spread across uncorrelated assets

### 5. Monitoring
- Track Sharpe and Sortino ratios
- Monitor VaR and CVaR regularly
- Update risk parameters as markets change
- Review violations and adjust

## Example: Complete Risk Management Workflow

```rust
use hyperphysics_market::risk::{
    RiskManager, RiskConfig, PositionSizingStrategy, StopLossType, Position
};
use chrono::Utc;

fn main() {
    // 1. Initialize risk manager
    let config = RiskConfig::default()
        .with_max_position_size(0.1)
        .with_max_drawdown(0.2)
        .with_max_daily_loss(0.05);

    let mut risk_manager = RiskManager::new(100000.0, config);

    // 2. Calculate position size
    let atr = 3.0; // Calculate from price data
    let strategy = PositionSizingStrategy::Volatility {
        atr,
        risk_per_trade: 1000.0
    };

    let price = 150.0;
    let shares = risk_manager.calculate_position_size(strategy, price);

    // 3. Calculate stop loss
    let stop_type = StopLossType::AtrBased {
        atr,
        multiplier: 2.0
    };
    let stop_price = risk_manager.calculate_stop_loss(price, stop_type, true);

    // 4. Check risk limits before trading
    let violations = risk_manager.check_risk_limits();
    if !violations.is_empty() {
        println!("Risk limits violated - not trading");
        return;
    }

    // 5. Add position
    let position = Position {
        symbol: "AAPL".to_string(),
        entry_price: price,
        current_price: price,
        quantity: shares,
        stop_loss: Some(stop_price),
        take_profit: Some(price * 1.1),
        entry_time: Utc::now(),
    };

    risk_manager.add_position(position);

    // 6. Monitor position
    risk_manager.update_position_price("AAPL", 155.0);

    // 7. Update trailing stop if profitable
    if let Some(pos) = risk_manager.get_positions().get("AAPL") {
        let trailing = StopLossType::Trailing {
            percentage: 0.05,
            activation_profit: 0.05
        };
        if let Some(new_stop) = risk_manager.update_trailing_stop(pos, trailing, true) {
            println!("Updated trailing stop to ${:.2}", new_stop);
        }
    }

    // 8. Get portfolio metrics
    let metrics = risk_manager.get_metrics();
    println!("Portfolio Value: ${:.2}", metrics.total_value);
    println!("Sharpe Ratio: {:.2}", metrics.sharpe_ratio);
    println!("Max Drawdown: {:.2}%", metrics.max_drawdown * 100.0);

    // 9. Update capital (daily)
    risk_manager.update_capital(105000.0);

    // 10. Reset daily tracking (start of new day)
    risk_manager.reset_daily();
}
```

## Performance Considerations

- Risk calculations are O(n) where n is number of returns
- VaR/CVaR require sorting, O(n log n)
- Position tracking is O(1) with HashMap
- All metrics can be calculated in real-time

## Thread Safety

The `RiskManager` is not thread-safe by default. For concurrent access:

```rust
use std::sync::{Arc, Mutex};

let risk_manager = Arc::new(Mutex::new(RiskManager::new(100000.0, config)));

// In another thread
let shares = {
    let rm = risk_manager.lock().unwrap();
    rm.calculate_position_size(strategy, price)
};
```

## See Also

- [Backtesting Guide](./backtesting_guide.md)
- [Market Data Providers](./market_data_guide.md)
- [Example: risk_management_example.rs](/examples/risk_management_example.rs)
