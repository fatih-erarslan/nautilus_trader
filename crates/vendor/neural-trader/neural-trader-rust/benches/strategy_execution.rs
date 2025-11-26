// Strategy Execution Benchmark
//
// Performance targets:
// - Signal generation: <5ms per strategy
// - Strategy processing: <10ms for 100 bars
// - Throughput: >2,000 signals/sec

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_core::types::{Direction, Symbol};
use nt_market_data::types::{Bar, Timeframe};
use nt_strategies::{
    momentum::MomentumStrategy, mean_reversion::MeanReversionStrategy, Strategy, StrategyConfig,
};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// ============================================================================
// Helper Functions - Generate Test Data
// ============================================================================

fn generate_bars(count: usize, trend: &str) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(count);
    let base_time = chrono::Utc::now();
    let base_price = dec!(100);

    for i in 0..count {
        let price = match trend {
            "uptrend" => base_price + Decimal::from(i) * dec!(0.5),
            "downtrend" => base_price - Decimal::from(i) * dec!(0.5),
            "sideways" => base_price + Decimal::from((i * 7) % 10 - 5) * dec!(0.1),
            _ => base_price,
        };

        bars.push(Bar {
            symbol: Symbol::new("AAPL").unwrap(),
            timestamp: base_time + chrono::Duration::minutes(i as i64),
            open: price - dec!(0.2),
            high: price + dec!(0.5),
            low: price - dec!(0.5),
            close: price,
            volume: dec!(1000000),
            trade_count: 1000,
            vwap: Some(price),
            timeframe: Timeframe::Minute,
        });
    }

    bars
}

// ============================================================================
// Benchmarks - Momentum Strategy
// ============================================================================

fn bench_momentum_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("momentum_strategy");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for data_size in [20, 100, 500].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        // Uptrend scenario
        group.bench_with_input(
            BenchmarkId::new("uptrend", data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size, "uptrend");
                let config = StrategyConfig {
                    lookback_period: 14,
                    threshold: dec!(2.0),
                    ..Default::default()
                };
                let mut strategy = MomentumStrategy::new(config);

                b.to_async(&rt).iter(|| async {
                    strategy.process(black_box(&bars)).await
                });
            },
        );

        // Downtrend scenario
        group.bench_with_input(
            BenchmarkId::new("downtrend", data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size, "downtrend");
                let config = StrategyConfig {
                    lookback_period: 14,
                    threshold: dec!(2.0),
                    ..Default::default()
                };
                let mut strategy = MomentumStrategy::new(config);

                b.to_async(&rt).iter(|| async {
                    strategy.process(black_box(&bars)).await
                });
            },
        );

        // Sideways scenario
        group.bench_with_input(
            BenchmarkId::new("sideways", data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size, "sideways");
                let config = StrategyConfig {
                    lookback_period: 14,
                    threshold: dec!(2.0),
                    ..Default::default()
                };
                let mut strategy = MomentumStrategy::new(config);

                b.to_async(&rt).iter(|| async {
                    strategy.process(black_box(&bars)).await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Mean Reversion Strategy
// ============================================================================

fn bench_mean_reversion_strategy(c: &mut Criterion) {
    let mut group = c.benchmark_group("mean_reversion_strategy");

    let rt = tokio::runtime::Runtime::new().unwrap();

    for data_size in [20, 100, 500].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size, "sideways");
                let config = StrategyConfig {
                    lookback_period: 20,
                    threshold: dec!(2.0),
                    ..Default::default()
                };
                let mut strategy = MeanReversionStrategy::new(config);

                b.to_async(&rt).iter(|| async {
                    strategy.process(black_box(&bars)).await
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Multi-Strategy Execution
// ============================================================================

fn bench_multi_strategy_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_strategy_execution");
    group.sample_size(50);

    let rt = tokio::runtime::Runtime::new().unwrap();

    for strategy_count in [2, 5, 10].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(strategy_count),
            strategy_count,
            |b, &strategy_count| {
                let bars = generate_bars(100, "uptrend");

                b.to_async(&rt).iter(|| async {
                    // Execute multiple strategies in parallel
                    let mut handles = vec![];

                    for i in 0..strategy_count {
                        let bars_clone = bars.clone();
                        let config = StrategyConfig {
                            lookback_period: 14 + (i % 5) * 2,
                            threshold: dec!(1.5) + Decimal::from(i) * dec!(0.1),
                            ..Default::default()
                        };

                        let handle = tokio::spawn(async move {
                            let mut strategy = MomentumStrategy::new(config);
                            strategy.process(&bars_clone).await
                        });

                        handles.push(handle);
                    }

                    // Wait for all strategies to complete
                    for handle in handles {
                        let _ = handle.await;
                    }
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Signal Generation
// ============================================================================

fn bench_signal_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_generation");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("generate_long_signal", |b| {
        let bars = generate_bars(100, "uptrend");
        let config = StrategyConfig {
            lookback_period: 14,
            threshold: dec!(2.0),
            ..Default::default()
        };
        let mut strategy = MomentumStrategy::new(config);

        b.to_async(&rt).iter(|| async {
            strategy.process(black_box(&bars)).await
        });
    });

    group.bench_function("generate_short_signal", |b| {
        let bars = generate_bars(100, "downtrend");
        let config = StrategyConfig {
            lookback_period: 14,
            threshold: dec!(2.0),
            ..Default::default()
        };
        let mut strategy = MomentumStrategy::new(config);

        b.to_async(&rt).iter(|| async {
            strategy.process(black_box(&bars)).await
        });
    });

    group.bench_function("no_signal_sideways", |b| {
        let bars = generate_bars(100, "sideways");
        let config = StrategyConfig {
            lookback_period: 14,
            threshold: dec!(2.0),
            ..Default::default()
        };
        let mut strategy = MomentumStrategy::new(config);

        b.to_async(&rt).iter(|| async {
            strategy.process(black_box(&bars)).await
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Strategy State Updates
// ============================================================================

fn bench_strategy_state_updates(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_state_updates");

    let rt = tokio::runtime::Runtime::new().unwrap();

    group.bench_function("update_with_single_bar", |b| {
        let mut bars = generate_bars(100, "uptrend");
        let new_bar = generate_bars(1, "uptrend")[0].clone();
        let config = StrategyConfig {
            lookback_period: 14,
            threshold: dec!(2.0),
            ..Default::default()
        };
        let mut strategy = MomentumStrategy::new(config);

        b.to_async(&rt).iter(|| async {
            bars.push(black_box(new_bar.clone()));
            strategy.process(black_box(&bars)).await
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Strategy Initialization
// ============================================================================

fn bench_strategy_initialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_initialization");

    group.bench_function("init_momentum_strategy", |b| {
        b.iter(|| {
            let config = StrategyConfig {
                lookback_period: 14,
                threshold: dec!(2.0),
                ..Default::default()
            };
            black_box(MomentumStrategy::new(config))
        });
    });

    group.bench_function("init_mean_reversion_strategy", |b| {
        b.iter(|| {
            let config = StrategyConfig {
                lookback_period: 20,
                threshold: dec!(2.0),
                ..Default::default()
            };
            black_box(MeanReversionStrategy::new(config))
        });
    });

    group.finish();
}

// ============================================================================
// Benchmark Configuration
// ============================================================================

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2))
        .configure_from_args()
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets =
        bench_momentum_strategy,
        bench_mean_reversion_strategy,
        bench_multi_strategy_execution,
        bench_signal_generation,
        bench_strategy_state_updates,
        bench_strategy_initialization
}

criterion_main!(benches);
