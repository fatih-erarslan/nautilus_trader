//! Chief Risk Officer Sentinel demonstration.
//!
//! This example shows the CRO sentinel in action managing firm-wide risk.

use hyper_risk_engine::core::types::{Order, OrderSide, Portfolio, Price, Quantity, Symbol, Timestamp};
use hyper_risk_engine::sentinels::{
    ChiefRiskOfficerSentinel, CROConfig, Sentinel, VetoDecision,
};

fn main() {
    println!("=== Chief Risk Officer Sentinel Demo ===\n");

    // Create CRO sentinel with default configuration
    let cro = ChiefRiskOfficerSentinel::new(CROConfig::default());

    println!("1. Initializing strategies with risk metrics...");
    cro.update_strategy_risk(1, 0.015, 0.020, 250_000.0);
    cro.update_strategy_risk(2, 0.018, 0.025, 350_000.0);
    cro.update_strategy_risk(3, 0.012, 0.016, 180_000.0);
    println!("   ✓ 3 strategies registered\n");

    // Aggregate firm-wide risk
    println!("2. Aggregating firm-wide risk metrics...");
    let metrics = cro.aggregate_portfolio_risk();
    println!("   Firm VaR:           ${:.2}", metrics.firm_var);
    println!("   Firm CVaR:          ${:.2}", metrics.firm_cvar);
    println!("   Total Exposure:     ${:.2}", metrics.total_exposure);
    println!("   Concentration Risk: {:.4}", metrics.concentration_risk);
    println!("   Active Strategies:  {}\n", metrics.active_strategies);

    // Test order veto
    println!("3. Testing order veto authority...");
    let portfolio = Portfolio::new(1_000_000.0);

    let small_order = Order {
        symbol: Symbol::new("AAPL"),
        side: OrderSide::Buy,
        quantity: Quantity::from_f64(100.0),
        limit_price: Some(Price::from_f64(150.0)),
        strategy_id: 1,
        timestamp: Timestamp::now(),
    };

    match cro.veto_order(&small_order, &portfolio) {
        VetoDecision::Approve => println!("   ✓ Small order APPROVED"),
        VetoDecision::Reject { reason } => println!("   ✗ Order REJECTED: {}", reason),
        VetoDecision::RequireApproval { reason } => {
            println!("   ⚠ Order requires MANUAL APPROVAL: {}", reason)
        }
    }

    // Large order that would breach concentration
    let large_order = Order {
        symbol: Symbol::new("TSLA"),
        side: OrderSide::Buy,
        quantity: Quantity::from_f64(1000.0),
        limit_price: Some(Price::from_f64(200.0)), // $200k order on $1M portfolio = 20%
        strategy_id: 2,
        timestamp: Timestamp::now(),
    };

    match cro.veto_order(&large_order, &portfolio) {
        VetoDecision::Approve => println!("   ✓ Large order APPROVED"),
        VetoDecision::Reject { reason } => println!("   ✗ Large order REJECTED: {}", reason),
        VetoDecision::RequireApproval { reason } => {
            println!("   ⚠ Large order requires MANUAL APPROVAL: {}", reason)
        }
    }
    println!();

    // Test liquidity crisis detection
    println!("4. Testing liquidity crisis detection...");
    cro.update_bid_ask_spread(1001, 0.01); // Normal spread
    cro.update_bid_ask_spread(1002, 0.015);

    // Simulate market stress
    cro.update_bid_ask_spread(1001, 0.05); // 5x widening
    cro.update_bid_ask_spread(1002, 0.07); // 4.7x widening

    if let Some(crisis) = cro.check_liquidity_crisis() {
        println!("   ⚠ LIQUIDITY CRISIS DETECTED!");
        println!("      Severity:           {}/100", crisis.severity);
        println!("      Affected Assets:    {}", crisis.affected_assets.len());
        println!("      Spread Widening:    {:.2}x", crisis.spread_widening_factor);
        println!(
            "      Est. Liquidation:   {:.1} days",
            crisis.estimated_liquidation_days
        );
    } else {
        println!("   ✓ No liquidity crisis detected");
    }
    println!();

    // Test correlation breakdown
    println!("5. Testing correlation breakdown detection...");
    let historical_corr = vec![
        1.0, 0.8, 0.7,
        0.8, 1.0, 0.6,
        0.7, 0.6, 1.0,
    ];
    cro.update_correlation_matrix(historical_corr);

    let current_corr = vec![
        1.0, 0.3, 0.2, // Major correlation breakdown
        0.3, 1.0, 0.2,
        0.2, 0.2, 1.0,
    ];

    if cro.detect_correlation_breakdown(&current_corr) {
        println!("   ⚠ CORRELATION BREAKDOWN DETECTED!");
        println!("      Risk models may be unreliable");
    } else {
        println!("   ✓ Correlations stable");
    }
    println!();

    // Test VaR breach tracking
    println!("6. Testing VaR breach counter...");
    println!("   Initial breaches: {}", cro.var_breach_count());
    cro.record_var_breach();
    cro.record_var_breach();
    println!("   After 2 breaches: {}", cro.var_breach_count());
    println!("   ⚠ Approaching threshold ({})", CROConfig::default().var_breach_threshold);
    println!();

    // Test counterparty exposure
    println!("7. Testing counterparty exposure tracking...");
    cro.update_counterparty_exposure(101, 180_000.0);
    cro.update_counterparty_exposure(102, 120_000.0);
    cro.update_counterparty_exposure(103, 95_000.0);

    let report = cro.evaluate_counterparty_exposure(1_000_000.0);
    println!("   Total Exposure:        ${:.2}", report.total_exposure);
    println!("   Max Single Exposure:   ${:.2}", report.max_single_exposure);
    println!("   Limit Breaches:        {}", report.limit_breaches.len());

    for (cp_id, exposure, limit) in &report.limit_breaches {
        println!(
            "      CP {}: ${:.2} exceeds limit ${:.2}",
            cp_id, exposure, limit
        );
    }
    println!();

    // Test global halt
    println!("8. Testing global halt authority...");
    println!("   Current halt status: {}", cro.is_global_halt());

    println!("   ⚠ CRO triggering GLOBAL HALT...");
    cro.trigger_global_halt(hyper_risk_engine::sentinels::HaltReason::FirmVaRBreach);
    println!("   Global halt status: {}", cro.is_global_halt());
    println!("   All trading STOPPED");

    println!("   Releasing halt...");
    cro.release_global_halt();
    println!("   Global halt status: {}", cro.is_global_halt());
    println!();

    // Performance check
    println!("9. Checking sentinel performance...");
    let test_order = Order {
        symbol: Symbol::new("SPY"),
        side: OrderSide::Buy,
        quantity: Quantity::from_f64(50.0),
        limit_price: Some(Price::from_f64(450.0)),
        strategy_id: 1,
        timestamp: Timestamp::now(),
    };

    // Warmup
    for _ in 0..100 {
        let _ = cro.check(&test_order, &portfolio);
    }

    // Measure
    let start = std::time::Instant::now();
    for _ in 0..1000 {
        let _ = cro.check(&test_order, &portfolio);
    }
    let elapsed = start.elapsed();
    let avg_ns = elapsed.as_nanos() / 1000;

    println!("   Average check latency: {}ns", avg_ns);
    println!("   Target latency:        50,000ns (50μs)");
    if avg_ns < 50_000 {
        println!("   ✓ Performance target MET");
    } else {
        println!("   ✗ Performance target MISSED");
    }
    println!();

    println!("=== Demo Complete ===");
    println!("\nThe ChiefRiskOfficerSentinel provides:");
    println!("  ✓ Firm-wide VaR aggregation");
    println!("  ✓ Correlation breakdown detection");
    println!("  ✓ Liquidity crisis monitoring");
    println!("  ✓ Counterparty exposure tracking");
    println!("  ✓ Order veto authority");
    println!("  ✓ Global halt capability");
    println!("  ✓ Position reduction mandates");
    println!("  ✓ <50μs latency for real-time protection");
}
