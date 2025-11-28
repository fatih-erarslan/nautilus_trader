//! Benchmark: Fast path latency (<100μs target).

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use hyper_risk_engine::fast_path::{PreTradeChecker, PreTradeConfig, LimitChecker, LimitConfig, FastAnomalyDetector};
use hyper_risk_engine::core::types::{Portfolio, Price, Quantity};

fn bench_pre_trade_check(c: &mut Criterion) {
    let config = PreTradeConfig::default();
    let checker = PreTradeChecker::new(config);
    let portfolio = Portfolio::new(1_000_000.0);
    let price = Price::from_f64(150.0);
    let quantity = Quantity::from_f64(100.0);

    c.bench_function("pre_trade_check", |b| {
        b.iter(|| {
            checker.check(
                black_box(&portfolio),
                black_box("AAPL"),
                black_box(price),
                black_box(quantity),
            )
        })
    });
}

fn bench_limit_checker(c: &mut Criterion) {
    let config = LimitConfig::default();
    let checker = LimitChecker::new(config);
    let portfolio = Portfolio::new(1_000_000.0);
    let price = Price::from_f64(150.0);
    let quantity = Quantity::from_f64(100.0);

    c.bench_function("limit_check", |b| {
        b.iter(|| {
            checker.check_order(
                black_box(&portfolio),
                black_box("AAPL"),
                black_box(price),
                black_box(quantity),
            )
        })
    });
}

fn bench_anomaly_detector(c: &mut Criterion) {
    let mut detector = FastAnomalyDetector::new(100, 3.0);

    // Prime with data
    for i in 0..100 {
        detector.update(100.0 + (i as f64) * 0.01);
    }

    c.bench_function("anomaly_check", |b| {
        b.iter(|| detector.check(black_box(100.5)))
    });
}

fn bench_portfolio_sizes(c: &mut Criterion) {
    let config = PreTradeConfig::default();
    let checker = PreTradeChecker::new(config);
    let price = Price::from_f64(150.0);
    let quantity = Quantity::from_f64(100.0);

    let mut group = c.benchmark_group("pre_trade_portfolio_size");

    for size in [100_000.0, 1_000_000.0, 10_000_000.0, 100_000_000.0] {
        let portfolio = Portfolio::new(size);
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("${:.0}", size)),
            &portfolio,
            |b, portfolio| {
                b.iter(|| {
                    checker.check(
                        black_box(portfolio),
                        black_box("AAPL"),
                        black_box(price),
                        black_box(quantity),
                    )
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pre_trade_check,
    bench_limit_checker,
    bench_anomaly_detector,
    bench_portfolio_sizes,
);

criterion_main!(benches);
