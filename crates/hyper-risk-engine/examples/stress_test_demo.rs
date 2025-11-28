//! Stress Test Sentinel Demo
//!
//! Demonstrates the usage of StressTestSentinel with real historical scenarios.

use hyper_risk_engine::{
    Portfolio, Position, Symbol, Price, Quantity, Timestamp, PositionId,
    StressTestSentinel, StressConfig, Scenario, Sentinel,
};

fn main() {
    println!("=== Stress Test Sentinel Demo ===\n");

    // Create a sample portfolio
    let mut portfolio = create_sample_portfolio();
    println!("Portfolio Value: ${:.2}", portfolio.total_value);
    println!("Positions:");
    for pos in &portfolio.positions {
        println!(
            "  {} - Qty: {}, Value: ${:.2}",
            pos.symbol.as_str(),
            pos.quantity.as_f64(),
            pos.market_value()
        );
    }
    println!();

    // Configure stress test sentinel with default betas
    let config = StressConfig {
        max_loss_threshold_pct: 15.0, // 15% max loss threshold
        scenarios_to_run: Vec::new(),  // Run all historical scenarios
        ..Default::default()
    }
    .with_default_betas();

    let sentinel = StressTestSentinel::new(config);

    println!("Running stress tests with {} scenarios...\n", Scenario::all_historical().len());

    // Run all stress scenarios
    let results = sentinel.run_all_scenarios(&portfolio);

    // Display results
    println!("=== Stress Test Results ===\n");
    for result in &results {
        let breach_marker = if result.breach { "⚠️  BREACH" } else { "✓" };
        println!(
            "{} {} - Impact: {:.2}% (${:.2})",
            breach_marker,
            result.scenario_name,
            result.portfolio_impact_pct,
            result.portfolio_impact_abs
        );

        if result.breach {
            println!("  Breach Severity: {:.2}x", result.breach_severity);
        }
    }

    // Show worst case
    println!("\n=== Worst Case Scenario ===");
    if let Some(worst) = sentinel.worst_case_scenario() {
        println!("Scenario: {}", worst.scenario_name);
        println!("Portfolio Impact: {:.2}% (${:.2})", worst.portfolio_impact_pct, worst.portfolio_impact_abs);
        println!("Breach: {}", if worst.breach { "YES" } else { "NO" });

        println!("\nAsset-Level Impacts:");
        for (symbol, impact) in &worst.asset_impacts {
            println!("  {}: ${:.2}", symbol, impact);
        }
    }

    // Demonstrate reverse stress testing
    println!("\n=== Reverse Stress Test (Breaking Scenarios) ===");
    let breaking = sentinel.find_breaking_scenarios(&portfolio);
    println!("Found {} scenarios that breach the {:.2}% limit:", breaking.len(), 15.0);
    for result in &breaking {
        println!(
            "  {} - {:.2}% loss",
            result.scenario_name,
            result.portfolio_impact_pct.abs()
        );
    }

    // Performance metrics
    println!("\n=== Performance Metrics ===");
    println!("Checks performed: {}", sentinel.check_count());
    println!("Average latency: {} ns ({:.2} μs)", sentinel.avg_latency_ns(), sentinel.avg_latency_ns() as f64 / 1000.0);
}

fn create_sample_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::new(1_000_000.0);

    // Add equity position (SPY - S&P 500 ETF)
    portfolio.positions.push(Position {
        id: PositionId::new(),
        symbol: Symbol::new("SPY"),
        quantity: Quantity::from_f64(2000.0),
        avg_entry_price: Price::from_f64(450.0),
        current_price: Price::from_f64(450.0),
        unrealized_pnl: 0.0,
        realized_pnl: 0.0,
        opened_at: Timestamp::now(),
        updated_at: Timestamp::now(),
    });

    // Add bond position (TLT - 20+ Year Treasury ETF)
    portfolio.positions.push(Position {
        id: PositionId::new(),
        symbol: Symbol::new("TLT"),
        quantity: Quantity::from_f64(1000.0),
        avg_entry_price: Price::from_f64(95.0),
        current_price: Price::from_f64(95.0),
        unrealized_pnl: 0.0,
        realized_pnl: 0.0,
        opened_at: Timestamp::now(),
        updated_at: Timestamp::now(),
    });

    // Add crypto position (BTC)
    portfolio.positions.push(Position {
        id: PositionId::new(),
        symbol: Symbol::new("BTC"),
        quantity: Quantity::from_f64(2.0),
        avg_entry_price: Price::from_f64(42000.0),
        current_price: Price::from_f64(42000.0),
        unrealized_pnl: 0.0,
        realized_pnl: 0.0,
        opened_at: Timestamp::now(),
        updated_at: Timestamp::now(),
    });

    portfolio.recalculate();
    portfolio
}
