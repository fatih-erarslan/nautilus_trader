//! Example demonstrating the risk management module
//!
//! This example shows how to:
//! - Configure risk limits
//! - Calculate position sizes using different strategies
//! - Manage stop losses
//! - Monitor portfolio risk metrics
//! - Check for risk violations

use hyperphysics_market::risk::{
    RiskManager, RiskConfig, PositionSizingStrategy, StopLossType, Position,
};
use chrono::Utc;

fn main() {
    println!("=== HyperPhysics Risk Management Example ===\n");

    // 1. Configure Risk Manager
    println!("1. Configuring Risk Manager...");
    let config = RiskConfig::default()
        .with_max_position_size(0.15)    // 15% max per position
        .with_max_drawdown(0.25)         // 25% max drawdown
        .with_max_daily_loss(0.05)       // 5% max daily loss
        .with_max_leverage(2.0)          // 2x max leverage
        .with_risk_free_rate(0.03);      // 3% risk-free rate

    let mut risk_manager = RiskManager::new(100000.0, config);
    println!("Initial capital: $100,000");
    println!("Max position size: 15%");
    println!("Max drawdown: 25%");
    println!("Max daily loss: 5%\n");

    // 2. Position Sizing Examples
    println!("2. Position Sizing Examples:");

    // Fixed amount
    let fixed_strategy = PositionSizingStrategy::Fixed { amount: 10000.0 };
    let shares = risk_manager.calculate_position_size(fixed_strategy, 150.0);
    println!("   Fixed $10,000: {:.0} shares @ $150 = ${:.2}",
        shares, shares * 150.0);

    // Percentage of capital
    let pct_strategy = PositionSizingStrategy::Percentage { percentage: 0.1 };
    let shares = risk_manager.calculate_position_size(pct_strategy, 150.0);
    println!("   10% of capital: {:.0} shares @ $150 = ${:.2}",
        shares, shares * 150.0);

    // Kelly Criterion (60% win rate, 2:1 reward/risk)
    let kelly_strategy = PositionSizingStrategy::Kelly {
        win_rate: 0.6,
        win_loss_ratio: 2.0
    };
    let shares = risk_manager.calculate_position_size(kelly_strategy, 150.0);
    println!("   Kelly Criterion (60% WR, 2:1 R/R): {:.0} shares @ $150 = ${:.2}",
        shares, shares * 150.0);

    // Volatility-based (ATR method)
    let vol_strategy = PositionSizingStrategy::Volatility {
        atr: 3.0,           // $3 ATR
        risk_per_trade: 1000.0  // Risk $1000 per trade
    };
    let shares = risk_manager.calculate_position_size(vol_strategy, 150.0);
    println!("   Volatility-based (ATR=$3): {:.0} shares @ $150 = ${:.2}\n",
        shares, shares * 150.0);

    // 3. Stop Loss Examples
    println!("3. Stop Loss Examples:");

    let entry_price = 150.0;

    // Fixed percentage stop
    let fixed_stop = StopLossType::Fixed { percentage: 0.02 };
    let stop_price = risk_manager.calculate_stop_loss(entry_price, fixed_stop, true);
    println!("   Fixed 2% stop: ${:.2} (${:.2} risk per share)",
        stop_price, entry_price - stop_price);

    // Trailing stop
    let trailing_stop = StopLossType::Trailing {
        percentage: 0.05,
        activation_profit: 0.1  // Activate after 10% profit
    };
    let stop_price = risk_manager.calculate_stop_loss(entry_price, trailing_stop, true);
    println!("   Trailing 5% stop (activates at +10%): ${:.2}", stop_price);

    // ATR-based stop
    let atr_stop = StopLossType::AtrBased {
        atr: 3.0,
        multiplier: 2.0
    };
    let stop_price = risk_manager.calculate_stop_loss(entry_price, atr_stop, true);
    println!("   ATR-based stop (2x ATR): ${:.2} (${:.2} risk per share)\n",
        stop_price, entry_price - stop_price);

    // 4. Add Some Positions
    println!("4. Adding Positions:");

    let position1 = Position {
        symbol: "AAPL".to_string(),
        entry_price: 150.0,
        current_price: 160.0,
        quantity: 100.0,
        stop_loss: Some(147.0),
        take_profit: Some(165.0),
        entry_time: Utc::now(),
    };
    println!("   AAPL: 100 shares @ $150, now @ $160 (P&L: ${:.2})",
        position1.unrealized_pnl());
    risk_manager.add_position(position1);

    let position2 = Position {
        symbol: "GOOGL".to_string(),
        entry_price: 2800.0,
        current_price: 2850.0,
        quantity: 5.0,
        stop_loss: Some(2750.0),
        take_profit: Some(2950.0),
        entry_time: Utc::now(),
    };
    println!("   GOOGL: 5 shares @ $2800, now @ $2850 (P&L: ${:.2})",
        position2.unrealized_pnl());
    risk_manager.add_position(position2);

    let position3 = Position {
        symbol: "TSLA".to_string(),
        entry_price: 240.0,
        current_price: 235.0,
        quantity: 50.0,
        stop_loss: Some(228.0),
        take_profit: Some(260.0),
        entry_time: Utc::now(),
    };
    println!("   TSLA: 50 shares @ $240, now @ $235 (P&L: ${:.2})\n",
        position3.unrealized_pnl());
    risk_manager.add_position(position3);

    // 5. Portfolio Metrics
    println!("5. Portfolio Risk Metrics:");

    // Simulate some trading history for metrics
    let returns = vec![
        0.02, 0.015, -0.01, 0.025, 0.01, -0.005, 0.018, 0.012,
        -0.008, 0.022, 0.015, -0.012, 0.01, 0.02, -0.015, 0.025
    ];
    for (i, ret) in returns.iter().enumerate() {
        let new_capital = risk_manager.get_metrics().total_value * (1.0 + ret);
        risk_manager.update_capital(new_capital);
    }

    let metrics = risk_manager.get_metrics();

    println!("   Total Portfolio Value: ${:.2}", metrics.total_value);
    println!("   Total P&L: ${:.2} ({:.2}%)",
        metrics.total_pnl,
        (metrics.total_pnl / 100000.0) * 100.0);
    println!("   Current Drawdown: {:.2}%", metrics.max_drawdown * 100.0);
    println!("   Sharpe Ratio: {:.3}", metrics.sharpe_ratio);
    println!("   Sortino Ratio: {:.3}", metrics.sortino_ratio);
    println!("   Value at Risk (95%): ${:.2}", metrics.var_95);
    println!("   Conditional VaR (95%): ${:.2}", metrics.cvar_95);
    println!("   Diversification Score: {:.3}", metrics.diversification_score);
    println!("   Current Leverage: {:.2}x\n", metrics.leverage);

    // 6. Risk Limit Violations
    println!("6. Checking Risk Limits:");
    let violations = risk_manager.check_risk_limits();

    if violations.is_empty() {
        println!("   ✓ All risk limits are within acceptable ranges");
    } else {
        println!("   ⚠ Risk violations detected:");
        for violation in violations {
            println!("      - {}", violation);
        }
    }
    println!();

    // 7. Individual Position Risk
    println!("7. Individual Position Risk:");
    for (symbol, position) in risk_manager.get_positions() {
        let risk_amount = position.entry_price - position.stop_loss.unwrap_or(0.0);
        let risk_pct = (risk_amount / position.entry_price) * 100.0;
        let position_size_pct = (position.market_value() / metrics.total_value) * 100.0;

        println!("   {}:", symbol);
        println!("      Entry: ${:.2}, Current: ${:.2}, Stop: ${:.2}",
            position.entry_price,
            position.current_price,
            position.stop_loss.unwrap_or(0.0));
        println!("      P&L: ${:.2} ({:.2}%)",
            position.unrealized_pnl(),
            position.unrealized_pnl_pct());
        println!("      Risk per share: ${:.2} ({:.2}%)", risk_amount, risk_pct);
        println!("      Position size: {:.2}% of portfolio", position_size_pct);
    }
    println!();

    // 8. Margin and Leverage
    println!("8. Margin and Leverage Calculations:");
    let total_exposure = risk_manager.get_positions()
        .values()
        .map(|p| p.market_value())
        .sum::<f64>();

    println!("   Total Exposure: ${:.2}", total_exposure);
    println!("   Current Capital: ${:.2}", metrics.total_value);
    println!("   Current Leverage: {:.2}x", metrics.leverage);

    let margin_req_1x = risk_manager.calculate_margin_requirement(1.0);
    let margin_req_2x = risk_manager.calculate_margin_requirement(2.0);
    let margin_req_3x = risk_manager.calculate_margin_requirement(3.0);

    println!("   Margin Required (1x): ${:.2}", margin_req_1x);
    println!("   Margin Required (2x): ${:.2}", margin_req_2x);
    println!("   Margin Required (3x): ${:.2}", margin_req_3x);

    println!("\n=== Example Complete ===");
}
