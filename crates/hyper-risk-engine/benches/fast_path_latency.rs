//! Benchmark: Fast path latency (<100μs target).

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyper_risk_engine::fast_path::pre_trade::{PreTradeChecker, PreTradeConfig};
use hyper_risk_engine::fast_path::limit_checker::{LimitChecker, LimitConfig};
use hyper_risk_engine::fast_path::anomaly_detector::{FastAnomalyDetector, AnomalyConfig};
use hyper_risk_engine::core::types::{Portfolio, Price, Quantity, Order, OrderSide, Symbol, Timestamp};

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

fn bench_pre_trade_check(c: &mut Criterion) {
    let config = PreTradeConfig::default();
    let checker = PreTradeChecker::new(config);
    let portfolio = Portfolio::new(1_000_000.0);
    let order = create_test_order(100.0, 150.0);

    c.bench_function("pre_trade_check", |b| {
        b.iter(|| {
            checker.check(
                black_box(&order),
                black_box(&portfolio),
            )
        })
    });
}

fn bench_limit_checker(c: &mut Criterion) {
    let config = LimitConfig::default();
    let checker = LimitChecker::new(config);
    let portfolio = Portfolio::new(1_000_000.0);
    let order = create_test_order(100.0, 150.0);

    c.bench_function("limit_check", |b| {
        b.iter(|| {
            checker.check(
                black_box(&order),
                black_box(&portfolio),
            )
        })
    });
}

fn bench_anomaly_detector(c: &mut Criterion) {
    let config = AnomalyConfig::default();
    let mut detector = FastAnomalyDetector::new(config);
    let portfolio = Portfolio::new(100_000.0);

    // Prime with data
    for i in 0..100 {
        let order = create_test_order(100.0 + (i as f64) * 0.1, 100.0);
        detector.update(&order, &portfolio);
    }

    let test_order = create_test_order(100.0, 100.5);

    c.bench_function("anomaly_score", |b| {
        b.iter(|| detector.score(black_box(&test_order), black_box(&portfolio)))
    });
}

fn bench_anomaly_update(c: &mut Criterion) {
    let config = AnomalyConfig::default();
    let mut detector = FastAnomalyDetector::new(config);
    let portfolio = Portfolio::new(100_000.0);
    let order = create_test_order(100.0, 100.0);

    c.bench_function("anomaly_update", |b| {
        b.iter(|| detector.update(black_box(&order), black_box(&portfolio)))
    });
}

fn bench_portfolio_sizes(c: &mut Criterion) {
    let config = PreTradeConfig::default();
    let checker = PreTradeChecker::new(config);
    let order = create_test_order(100.0, 150.0);

    let mut group = c.benchmark_group("pre_trade_portfolio_size");

    for size in [100_000.0, 1_000_000.0, 10_000_000.0, 100_000_000.0] {
        let portfolio = Portfolio::new(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("${:.0}", size)),
            &portfolio,
            |b, portfolio| {
                b.iter(|| {
                    checker.check(
                        black_box(&order),
                        black_box(portfolio),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_order_sizes(c: &mut Criterion) {
    let config = PreTradeConfig::default();
    let checker = PreTradeChecker::new(config);
    let portfolio = Portfolio::new(1_000_000.0);

    let mut group = c.benchmark_group("pre_trade_order_size");

    for qty in [10.0, 100.0, 1000.0, 10000.0] {
        let order = create_test_order(qty, 150.0);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("qty_{:.0}", qty)),
            &order,
            |b, order| {
                b.iter(|| {
                    checker.check(
                        black_box(order),
                        black_box(&portfolio),
                    )
                })
            },
        );
    }
    group.finish();
}

fn bench_limit_config_variations(c: &mut Criterion) {
    let portfolio = Portfolio::new(1_000_000.0);
    let order = create_test_order(100.0, 150.0);

    let mut group = c.benchmark_group("limit_config_variations");

    // Strict config
    let strict_config = LimitConfig {
        max_position_value: 100_000.0,
        max_order_value: 10_000.0,
        max_leverage: 1.0,
        max_concentration: 0.10,
        max_positions: 20,
        daily_loss_limit: 0.02,
    };
    let strict_checker = LimitChecker::new(strict_config);

    group.bench_function("strict_limits", |b| {
        b.iter(|| strict_checker.check(black_box(&order), black_box(&portfolio)))
    });

    // Relaxed config
    let relaxed_config = LimitConfig {
        max_position_value: 10_000_000.0,
        max_order_value: 1_000_000.0,
        max_leverage: 10.0,
        max_concentration: 0.50,
        max_positions: 200,
        daily_loss_limit: 0.20,
    };
    let relaxed_checker = LimitChecker::new(relaxed_config);

    group.bench_function("relaxed_limits", |b| {
        b.iter(|| relaxed_checker.check(black_box(&order), black_box(&portfolio)))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_pre_trade_check,
    bench_limit_checker,
    bench_anomaly_detector,
    bench_anomaly_update,
    bench_portfolio_sizes,
    bench_order_sizes,
    bench_limit_config_variations,
);

criterion_main!(benches);
