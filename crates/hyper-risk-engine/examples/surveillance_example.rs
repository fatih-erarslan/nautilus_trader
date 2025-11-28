//! Trade Surveillance Sentinel - Usage Example
//!
//! Demonstrates market manipulation detection in action.

#[allow(unused_imports)]
use hyper_risk_engine::{
    TradeSurveillanceSentinel, SurveillanceConfig,
    Order, Portfolio, Sentinel, OrderSide, Quantity, Symbol, Price, Timestamp,
};

fn main() {
    println!("=== Trade Surveillance Sentinel Example ===\n");

    // Initialize sentinel with default regulatory thresholds
    let sentinel = TradeSurveillanceSentinel::default();
    println!("✓ Surveillance sentinel initialized with regulatory defaults");
    println!("  - Spoofing: order/trade > 10:1, cancel rate > 90%");
    println!("  - Layering: 3+ price levels, cancel rate > 85%");
    println!("  - Momentum: 2% price move, 3x volume spike");
    println!("  - Quote Stuffing: 1000+ msg/sec, cancel rate > 95%\n");

    // Simulate normal trading
    println!("=== Scenario 1: Normal Trading ===");
    simulate_normal_trading(&sentinel);
    println!();

    // Reset for next scenario
    sentinel.reset_window();

    // Simulate spoofing pattern
    println!("=== Scenario 2: Spoofing Detection ===");
    simulate_spoofing(&sentinel);
    println!();

    // Reset for next scenario
    sentinel.reset_window();

    // Simulate layering pattern
    println!("=== Scenario 3: Layering Detection ===");
    simulate_layering(&sentinel);
    println!();

    // Reset for next scenario
    sentinel.reset_window();

    // Simulate momentum ignition
    println!("=== Scenario 4: Momentum Ignition Detection ===");
    simulate_momentum_ignition(&sentinel);
    println!();

    // Reset for next scenario
    sentinel.reset_window();

    // Simulate quote stuffing
    println!("=== Scenario 5: Quote Stuffing Detection ===");
    simulate_quote_stuffing(&sentinel);
    println!();

    // Demonstrate configuration presets
    println!("=== Configuration Presets ===");
    demonstrate_configs();
}

fn simulate_normal_trading(sentinel: &TradeSurveillanceSentinel) {
    let portfolio = Portfolio::new(1_000_000.0);

    // Normal trading: orders mostly execute
    for i in 0..10 {
        sentinel.record_order(100.0, 150.0 + i as f64 * 0.1);
    }
    for _ in 0..8 {
        sentinel.record_trade(100.0);
    }
    for _ in 0..2 {
        sentinel.record_cancel();
    }

    let stats = sentinel.get_flow_stats();
    println!("Order/Trade Ratio: {:.1} (threshold: 10.0)", stats.order_to_trade_ratio());
    println!("Cancel Rate: {:.1}% (threshold: 90%)", stats.cancel_rate() * 100.0);

    let order = create_test_order(100.0, 150.0);
    match sentinel.check(&order, &portfolio) {
        Ok(()) => println!("✓ APPROVED - Normal trading pattern"),
        Err(e) => println!("✗ REJECTED - {}", e),
    }
}

fn simulate_spoofing(sentinel: &TradeSurveillanceSentinel) {
    let portfolio = Portfolio::new(1_000_000.0);

    println!("Simulating: 100 orders placed, 95 cancelled, 5 executed...");

    // Create spoofing pattern
    for _ in 0..100 {
        sentinel.record_order(10000.0, 150.0);
    }
    for _ in 0..95 {
        sentinel.record_cancel();
    }
    for _ in 0..5 {
        sentinel.record_trade(10000.0);
    }

    let stats = sentinel.get_flow_stats();
    println!("Order/Trade Ratio: {:.1} (threshold: 10.0)", stats.order_to_trade_ratio());
    println!("Cancel Rate: {:.1}% (threshold: 90%)", stats.cancel_rate() * 100.0);

    let order = create_test_order(10000.0, 150.0);
    match sentinel.check(&order, &portfolio) {
        Ok(()) => println!("✓ APPROVED"),
        Err(e) => {
            println!("✗ REJECTED - SPOOFING DETECTED");
            println!("Evidence: {}", e);
        }
    }
}

fn simulate_layering(sentinel: &TradeSurveillanceSentinel) {
    let portfolio = Portfolio::new(1_000_000.0);

    println!("Simulating: 5 price levels, 90% cancel rate...");

    // Set up layering pattern
    sentinel.update_price_levels(5);
    for _ in 0..100 {
        sentinel.record_order(1000.0, 150.0);
    }
    for _ in 0..90 {
        sentinel.record_cancel();
    }

    let stats = sentinel.get_flow_stats();
    println!("Price Levels: 5 (threshold: 3)");
    println!("Cancel Rate: {:.1}% (threshold: 85%)", stats.cancel_rate() * 100.0);

    let order = create_test_order(1000.0, 150.0);
    match sentinel.check(&order, &portfolio) {
        Ok(()) => println!("✓ APPROVED"),
        Err(e) => {
            println!("✗ REJECTED - LAYERING DETECTED");
            println!("Evidence: {}", e);
        }
    }
}

fn simulate_momentum_ignition(sentinel: &TradeSurveillanceSentinel) {
    let portfolio = Portfolio::new(1_000_000.0);

    println!("Simulating: 2% price jump + 3.5x volume spike...");

    // Set baseline
    sentinel.update_avg_volume(10000.0);

    // Create momentum pattern: rapid price move + volume spike
    sentinel.record_order(5000.0, 150.0);   // Previous price
    sentinel.record_order(35000.0, 153.0);  // 2% jump + 3.5x volume

    println!("Price Move: 2.00% (threshold: 2.00%)");
    println!("Volume Ratio: 3.5x (threshold: 3.0x)");

    let order = create_test_order(1000.0, 153.0);
    match sentinel.check(&order, &portfolio) {
        Ok(()) => println!("✓ APPROVED"),
        Err(e) => {
            println!("✗ REJECTED - MOMENTUM IGNITION DETECTED");
            println!("Evidence: {}", e);
        }
    }
}

fn simulate_quote_stuffing(sentinel: &TradeSurveillanceSentinel) {
    let portfolio = Portfolio::new(1_000_000.0);

    println!("Simulating: 1200 messages/sec, 98% cancel rate...");

    // Create quote stuffing pattern
    for _ in 0..1200 {
        sentinel.record_order(100.0, 150.0);
    }
    for _ in 0..1176 {
        sentinel.record_cancel();
    }

    let stats = sentinel.get_flow_stats();
    println!("Messages/Sec: {:.0} (threshold: 1000)", stats.messages_per_sec);
    println!("Cancel Rate: {:.1}% (threshold: 95%)", stats.cancel_rate() * 100.0);

    let order = create_test_order(100.0, 150.0);
    match sentinel.check(&order, &portfolio) {
        Ok(()) => println!("✓ APPROVED"),
        Err(e) => {
            println!("✗ REJECTED - QUOTE STUFFING DETECTED");
            println!("Evidence: {}", e);
        }
    }
}

fn demonstrate_configs() {
    println!("\n1. Default Configuration (Regulatory Standards):");
    let default = SurveillanceConfig::default();
    println!("   Spoofing order/trade: {}", default.spoofing_order_trade_ratio);
    println!("   Spoofing cancel rate: {:.0}%", default.spoofing_cancel_rate * 100.0);

    println!("\n2. Conservative Configuration (HFT Environments):");
    let conservative = SurveillanceConfig::conservative();
    println!("   Spoofing order/trade: {} (stricter)", conservative.spoofing_order_trade_ratio);
    println!("   Spoofing cancel rate: {:.0}% (stricter)", conservative.spoofing_cancel_rate * 100.0);

    println!("\n3. Permissive Configuration (Normal Trading):");
    let permissive = SurveillanceConfig::permissive();
    println!("   Spoofing order/trade: {} (lenient)", permissive.spoofing_order_trade_ratio);
    println!("   Spoofing cancel rate: {:.0}% (lenient)", permissive.spoofing_cancel_rate * 100.0);
}

fn create_test_order(quantity: f64, price: f64) -> Order {
    Order {
        symbol: Symbol::new("AAPL"),
        side: OrderSide::Buy,
        quantity: Quantity::from_f64(quantity),
        limit_price: Some(Price::from_f64(price)),
        strategy_id: 1,
        timestamp: Timestamp::now(),
    }
}
