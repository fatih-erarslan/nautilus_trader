//! Pattern Matcher Performance Benchmarks
//!
//! Validates:
//! - DTW computation time (<1ms target)
//! - Pattern extraction speed
//! - Signal generation latency (<10ms target)
//! - AgentDB integration performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nt_strategies::{
    pattern_matcher::{PatternBasedStrategy, PatternMatcherConfig},
    MarketData, Portfolio,
};
use nt_core::types::{Bar, Symbol};
use rust_decimal::Decimal;
use std::str::FromStr;
use chrono::Utc;

/// Benchmark DTW computation (pure Rust vs WASM)
fn bench_dtw_computation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_computation");

    let pattern_sizes = vec![10, 20, 50, 100];

    for size in pattern_sizes {
        let pattern_a: Vec<f64> = (0..size).map(|i| (i as f64) / (size as f64)).collect();
        let pattern_b: Vec<f64> = (0..size).map(|i| ((i + 5) as f64) / (size as f64)).collect();

        group.bench_with_input(
            BenchmarkId::new("rust_dtw", size),
            &(pattern_a.clone(), pattern_b.clone()),
            |b, (a, b_pat)| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = PatternMatcherConfig {
                    use_wasm: false,
                    window_size: size,
                    ..Default::default()
                };
                let strategy = rt.block_on(async {
                    PatternBasedStrategy::new(
                        "bench".to_string(),
                        config,
                        "http://localhost:8765".to_string(),
                    )
                    .await
                    .unwrap()
                });

                b.iter(|| {
                    rt.block_on(async {
                        strategy.calculate_dtw_distance(
                            black_box(a),
                            black_box(b_pat)
                        ).await.unwrap()
                    })
                });
            },
        );
    }

    group.finish();
}

/// Benchmark pattern extraction
fn bench_pattern_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_extraction");

    let bar_counts = vec![50, 100, 200, 500];

    for count in bar_counts {
        let bars = generate_test_bars(count);

        group.bench_with_input(
            BenchmarkId::new("extract_pattern", count),
            &bars,
            |b, bars| {
                let rt = tokio::runtime::Runtime::new().unwrap();
                let config = PatternMatcherConfig::default();
                let strategy = rt.block_on(async {
                    PatternBasedStrategy::new(
                        "bench".to_string(),
                        config,
                        "http://localhost:8765".to_string(),
                    )
                    .await
                    .unwrap()
                });

                let market_data = MarketData::new("TEST".to_string(), bars.clone());

                b.iter(|| {
                    strategy.extract_pattern(black_box(&market_data))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full signal generation pipeline
fn bench_signal_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_generation");
    group.sample_size(10); // Reduce sample size for slower operations

    let bars = generate_test_bars(100);
    let market_data = MarketData::new("AAPL".to_string(), bars);
    let portfolio = Portfolio::new(Decimal::from(100000));

    group.bench_function("full_pipeline", |b| {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let config = PatternMatcherConfig {
            use_wasm: false,
            top_k: 10, // Reduce for faster benchmarking
            ..Default::default()
        };
        let strategy = rt.block_on(async {
            PatternBasedStrategy::new(
                "bench".to_string(),
                config,
                "http://localhost:8765".to_string(),
            )
            .await
            .unwrap()
        });

        b.iter(|| {
            rt.block_on(async {
                strategy.process(
                    black_box(&market_data),
                    black_box(&portfolio)
                ).await.unwrap()
            })
        });
    });

    group.finish();
}

/// Benchmark pattern normalization
fn bench_pattern_normalization(c: &mut Criterion) {
    let mut group = c.benchmark_group("pattern_normalization");

    let sizes = vec![10, 20, 50, 100];

    for size in sizes {
        let prices: Vec<f64> = (0..size)
            .map(|i| 100.0 + (i as f64 * 0.5))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("normalize", size),
            &prices,
            |b, prices| {
                b.iter(|| {
                    let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
                    let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;

                    prices
                        .iter()
                        .map(|p| (p - min) / range)
                        .collect::<Vec<f64>>()
                });
            },
        );
    }

    group.finish();
}

/// Benchmark embedding conversion
fn bench_embedding_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("embedding_conversion");

    let sizes = vec![10, 20, 50, 100];

    for size in sizes {
        let pattern: Vec<f64> = (0..size).map(|i| i as f64).collect();

        group.bench_with_input(
            BenchmarkId::new("f64_to_f32", size),
            &pattern,
            |b, pattern| {
                b.iter(|| {
                    pattern.iter().map(|&x| x as f32).collect::<Vec<f32>>()
                });
            },
        );
    }

    group.finish();
}

// Helper functions

fn generate_test_bars(count: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(count);
    let mut price = 100.0;

    for i in 0..count {
        let change = (i as f64 * 0.1).sin() * 0.5;
        price += change;

        let bar = Bar {
            symbol: Symbol::from_str("TEST").unwrap(),
            timestamp: Utc::now().timestamp() - ((count - i) as i64 * 300),
            open: Decimal::from_f64_retain(price - 0.1).unwrap(),
            high: Decimal::from_f64_retain(price + 0.2).unwrap(),
            low: Decimal::from_f64_retain(price - 0.2).unwrap(),
            close: Decimal::from_f64_retain(price).unwrap(),
            volume: Decimal::from(1000000),
        };

        bars.push(bar);
    }

    bars
}

criterion_group!(
    benches,
    bench_dtw_computation,
    bench_pattern_extraction,
    bench_signal_generation,
    bench_pattern_normalization,
    bench_embedding_conversion,
);

criterion_main!(benches);
