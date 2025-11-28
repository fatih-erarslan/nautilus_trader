//! Integration tests for RegulatoryComplianceSentinel

use hyper_risk_engine::sentinels::compliance::*;
use hyper_risk_engine::sentinels::{Sentinel, SentinelStatus};
use hyper_risk_engine::core::types::*;

fn create_test_order(symbol: &str, side: OrderSide, qty: f64, price: f64) -> Order {
    Order {
        symbol: Symbol::new(symbol),
        side,
        quantity: Quantity::from_f64(qty),
        limit_price: Some(Price::from_f64(price)),
        strategy_id: 1,
        timestamp: Timestamp::now(),
    }
}

fn create_test_portfolio() -> Portfolio {
    Portfolio::new(1_000_000.0)
}

#[test]
fn test_compliance_sentinel_creation() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();
    assert_eq!(sentinel.status(), SentinelStatus::Active);
    assert_eq!(sentinel.check_count(), 0);
}

#[test]
fn test_restricted_list_violation() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();
    let restricted_symbol = Symbol::new("RESTRICTED");

    sentinel.add_restricted_symbol(restricted_symbol);

    let order = create_test_order("RESTRICTED", OrderSide::Buy, 100.0, 150.0);
    let portfolio = create_test_portfolio();

    let result = sentinel.check(&order, &portfolio);
    assert!(result.is_err(), "Should reject restricted symbol");

    let violations = sentinel.get_violations();
    assert!(!violations.is_empty(), "Should have logged violation");
}

#[test]
fn test_position_concentration_limit() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();

    // Order for 15% of portfolio (exceeds 10% default limit)
    let order = create_test_order("AAPL", OrderSide::Buy, 1000.0, 150.0);
    let portfolio = create_test_portfolio(); // $1M

    let result = sentinel.check(&order, &portfolio);
    assert!(result.is_err(), "Should reject concentration limit violation");
}

#[test]
fn test_short_sale_restricted() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();
    let symbol = Symbol::new("XYZ");

    sentinel.set_short_sale_status(symbol, ShortSaleStatus::Restricted);

    let order = create_test_order("XYZ", OrderSide::Sell, 100.0, 50.0);
    let portfolio = create_test_portfolio();

    let result = sentinel.check(&order, &portfolio);
    assert!(result.is_err(), "Should reject short sale when circuit breaker active");
}

#[test]
fn test_normal_order_passes() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();

    // Small order well within limits: 100 shares * $150 = $15k (1.5% of $1M)
    let order = create_test_order("GOOGL", OrderSide::Buy, 100.0, 150.0);
    let portfolio = create_test_portfolio();

    let result = sentinel.check(&order, &portfolio);
    assert!(result.is_ok(), "Normal order should pass compliance checks");
    assert_eq!(sentinel.check_count(), 1);
}

#[test]
fn test_sentinel_disable() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();
    let restricted_symbol = Symbol::new("BLOCKED");

    sentinel.add_restricted_symbol(restricted_symbol);
    sentinel.disable();

    assert_eq!(sentinel.status(), SentinelStatus::Disabled);

    // Disabled sentinel should allow restricted symbol
    let order = create_test_order("BLOCKED", OrderSide::Buy, 100.0, 50.0);
    let portfolio = create_test_portfolio();

    let result = sentinel.check(&order, &portfolio);
    assert!(result.is_ok(), "Disabled sentinel should allow all orders");
}

#[test]
fn test_latency_under_100us() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();

    let order = create_test_order("MSFT", OrderSide::Buy, 100.0, 350.0);
    let portfolio = create_test_portfolio();

    for _ in 0..100 {
        let _ = sentinel.check(&order, &portfolio);
    }

    let avg_latency = sentinel.avg_latency_ns();
    assert!(avg_latency < 100_000,
        "Average latency {}ns should be under 100μs", avg_latency);
}

#[test]
fn test_large_trader_threshold_informational() {
    let mut config = ComplianceConfig::default();
    config.large_trader_notional_threshold = 50_000.0; // $50k for test
    config.max_float_percentage = 100.0; // Disable concentration limit for this test

    let sentinel = RegulatoryComplianceSentinel::new(config);

    // Order exceeding threshold: 1000 * $60 = $60k
    let order = create_test_order("TSLA", OrderSide::Buy, 1_000.0, 60.0);
    let portfolio = create_test_portfolio();

    // Should pass (informational only)
    let result = sentinel.check(&order, &portfolio);
    assert!(result.is_ok(), "Large trader threshold should not block orders");

    // But should log violation
    let violations = sentinel.get_violations();
    assert!(violations.iter().any(|v|
        v.check_type == ComplianceCheckType::LargeTrader
    ), "Should log large trader violation");
}

#[test]
fn test_reset_clears_state() {
    let sentinel = RegulatoryComplianceSentinel::with_defaults();

    let order = create_test_order("AAPL", OrderSide::Buy, 100.0, 150.0);
    let portfolio = create_test_portfolio();

    let _ = sentinel.check(&order, &portfolio);
    assert!(sentinel.check_count() > 0);

    sentinel.reset();

    assert_eq!(sentinel.check_count(), 0);
    assert_eq!(sentinel.get_violations().len(), 0);
}
