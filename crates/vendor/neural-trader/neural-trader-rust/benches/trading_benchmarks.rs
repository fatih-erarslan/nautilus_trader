use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nt_strategies::{PairsStrategy, Strategy};
use nt_portfolio::{Portfolio, PositionTracker};
use nt_risk::{calculate_var, VarMethod};
use nt_execution::OrderManager;
use chrono::Utc;

/// Benchmark market data processing
fn benchmark_market_data(c: &mut Criterion) {
    let mut group = c.benchmark_group("market_data");

    for size in [100, 1000, 10000].iter() {
        group.benchmark_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                b.iter(|| {
                    let data = generate_mock_bars(size);
                    black_box(data);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark strategy signal generation
fn benchmark_strategy_signals(c: &mut Criterion) {
    let mut group = c.benchmark_group("strategy_signals");

    let data = generate_mock_bars(252); // 1 year of daily data

    group.bench_function("pairs_strategy", |b| {
        let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);
        b.iter(|| {
            let signals = strategy.generate_signals(black_box(&data));
            black_box(signals);
        });
    });

    group.finish();
}

/// Benchmark portfolio calculations
fn benchmark_portfolio(c: &mut Criterion) {
    let mut group = c.benchmark_group("portfolio");

    let mut portfolio = Portfolio::new(100_000.0);
    portfolio.add_position("AAPL", 50, 180.50);
    portfolio.add_position("MSFT", 30, 380.25);
    portfolio.add_position("GOOGL", 20, 140.75);

    group.bench_function("calculate_metrics", |b| {
        b.iter(|| {
            let metrics = portfolio.calculate_metrics();
            black_box(metrics);
        });
    });

    group.bench_function("calculate_pnl", |b| {
        b.iter(|| {
            let pnl = portfolio.calculate_pnl();
            black_box(pnl);
        });
    });

    group.finish();
}

/// Benchmark risk calculations
fn benchmark_risk(c: &mut Criterion) {
    let mut group = c.benchmark_group("risk_calculation");

    let portfolio = create_test_portfolio();

    for &confidence in &[0.90, 0.95, 0.99] {
        group.bench_with_input(
            BenchmarkId::new("historical_var", confidence),
            &confidence,
            |b, &conf| {
                b.iter(|| {
                    let var = calculate_var(black_box(&portfolio), conf, VarMethod::Historical);
                    black_box(var);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("monte_carlo_var", confidence),
            &confidence,
            |b, &conf| {
                b.iter(|| {
                    let var = calculate_var(black_box(&portfolio), conf, VarMethod::MonteCarlo);
                    black_box(var);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark order execution
fn benchmark_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("order_execution");

    group.bench_function("order_validation", |b| {
        let order = create_test_order();
        b.iter(|| {
            let validated = validate_order(black_box(&order));
            black_box(validated);
        });
    });

    group.finish();
}

/// Benchmark backtesting performance
fn benchmark_backtesting(c: &mut Criterion) {
    let mut group = c.benchmark_group("backtesting");

    let data = generate_mock_bars(252); // 1 year
    let strategy = PairsStrategy::new(vec!["AAPL", "MSFT"], 20, 2.0);

    group.bench_function("1_year_backtest", |b| {
        b.iter(|| {
            let results = run_backtest(black_box(&strategy), black_box(&data));
            black_box(results);
        });
    });

    group.finish();
}

/// Benchmark neural model inference
fn benchmark_neural_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_inference");

    // Benchmark N-HiTS model inference
    group.bench_function("nhits_forecast", |b| {
        let model = load_nhits_model();
        let input = generate_input_sequence(100);

        b.iter(|| {
            let forecast = model.predict(black_box(&input));
            black_box(forecast);
        });
    });

    group.finish();
}

/// Benchmark data serialization
fn benchmark_serialization(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    let portfolio = create_test_portfolio();

    group.bench_function("json_serialize", |b| {
        b.iter(|| {
            let json = serde_json::to_string(black_box(&portfolio)).unwrap();
            black_box(json);
        });
    });

    group.bench_function("json_deserialize", |b| {
        let json = serde_json::to_string(&portfolio).unwrap();
        b.iter(|| {
            let deserialized: Portfolio = serde_json::from_str(black_box(&json)).unwrap();
            black_box(deserialized);
        });
    });

    group.finish();
}

// Helper functions
fn generate_mock_bars(count: usize) -> Vec<Bar> {
    (0..count)
        .map(|i| Bar {
            timestamp: Utc::now(),
            open: 100.0 + i as f64 * 0.1,
            high: 105.0 + i as f64 * 0.1,
            low: 95.0 + i as f64 * 0.1,
            close: 100.0 + i as f64 * 0.1,
            volume: 1_000_000,
        })
        .collect()
}

fn create_test_portfolio() -> Portfolio {
    let mut portfolio = Portfolio::new(100_000.0);
    portfolio.add_position("AAPL", 50, 180.50);
    portfolio.add_position("MSFT", 30, 380.25);
    portfolio.add_position("GOOGL", 20, 140.75);
    portfolio
}

criterion_group!(
    benches,
    benchmark_market_data,
    benchmark_strategy_signals,
    benchmark_portfolio,
    benchmark_risk,
    benchmark_execution,
    benchmark_backtesting,
    benchmark_neural_inference,
    benchmark_serialization
);

criterion_main!(benches);
