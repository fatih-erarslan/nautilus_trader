// Feature Extraction Latency Benchmark
//
// Performance targets:
// - Feature extraction: <1ms for 100 bars
// - Throughput: >5,000 feature vectors/sec
// - p50/p95/p99 latency tracking

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_core::types::Symbol;
use nt_features::technical::TechnicalIndicators;
use nt_market_data::types::{Bar, Timeframe};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// ============================================================================
// Helper Functions - Generate Test Data
// ============================================================================

fn generate_bars(count: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(count);
    let base_time = chrono::Utc::now();
    let base_price = dec!(100);

    for i in 0..count {
        // Create realistic price movement with trend and noise
        let trend = Decimal::from(i) * dec!(0.1);
        let noise = Decimal::from((i * 7) % 10) - dec!(5); // Simple pseudo-random
        let price = base_price + trend + noise * dec!(0.1);

        bars.push(Bar {
            symbol: Symbol::new("AAPL").unwrap(),
            timestamp: base_time + chrono::Duration::minutes(i as i64),
            open: price - dec!(0.5),
            high: price + dec!(1.0),
            low: price - dec!(1.0),
            close: price,
            volume: dec!(1000000) + Decimal::from(i * 1000),
            trade_count: 1000,
            vwap: Some(price),
            timeframe: Timeframe::Minute,
        });
    }

    bars
}

// ============================================================================
// Benchmarks - Simple Moving Average (SMA)
// ============================================================================

fn bench_sma_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sma_calculation");

    for data_size in [20, 100, 500, 1000].iter() {
        for window in [10, 20, 50].iter() {
            if window > data_size {
                continue;
            }

            group.bench_with_input(
                BenchmarkId::new(format!("data_{}", data_size), window),
                &(*data_size, *window),
                |b, &(data_size, window)| {
                    let bars = generate_bars(data_size);
                    let indicators = TechnicalIndicators::new();

                    b.iter(|| {
                        indicators.sma(black_box(&bars), window)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Exponential Moving Average (EMA)
// ============================================================================

fn bench_ema_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("ema_calculation");

    for data_size in [20, 100, 500, 1000].iter() {
        for window in [10, 20, 50].iter() {
            if window > data_size {
                continue;
            }

            group.bench_with_input(
                BenchmarkId::new(format!("data_{}", data_size), window),
                &(*data_size, *window),
                |b, &(data_size, window)| {
                    let bars = generate_bars(data_size);
                    let indicators = TechnicalIndicators::new();

                    b.iter(|| {
                        indicators.ema(black_box(&bars), window)
                    });
                },
            );
        }
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Relative Strength Index (RSI)
// ============================================================================

fn bench_rsi_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("rsi_calculation");

    for data_size in [20, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size);
                let indicators = TechnicalIndicators::new();

                b.iter(|| {
                    indicators.rsi(black_box(&bars), 14)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - MACD
// ============================================================================

fn bench_macd_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("macd_calculation");

    for data_size in [50, 100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size);
                let indicators = TechnicalIndicators::new();

                b.iter(|| {
                    indicators.macd(black_box(&bars), 12, 26, 9)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Bollinger Bands
// ============================================================================

fn bench_bollinger_bands(c: &mut Criterion) {
    let mut group = c.benchmark_group("bollinger_bands");

    for data_size in [20, 100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size);
                let indicators = TechnicalIndicators::new();

                b.iter(|| {
                    indicators.bollinger_bands(black_box(&bars), 20, dec!(2.0))
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Multiple Indicators (Realistic Scenario)
// ============================================================================

fn bench_full_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_feature_extraction");
    group.sample_size(50);  // Reduce sample size for expensive operation

    for data_size in [100, 500, 1000].iter() {
        group.throughput(Throughput::Elements(*data_size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(data_size),
            data_size,
            |b, &data_size| {
                let bars = generate_bars(data_size);
                let indicators = TechnicalIndicators::new();

                b.iter(|| {
                    // Extract all common indicators at once
                    let _sma_20 = indicators.sma(black_box(&bars), 20);
                    let _sma_50 = indicators.sma(black_box(&bars), 50);
                    let _ema_12 = indicators.ema(black_box(&bars), 12);
                    let _ema_26 = indicators.ema(black_box(&bars), 26);
                    let _rsi = indicators.rsi(black_box(&bars), 14);
                    let _macd = indicators.macd(black_box(&bars), 12, 26, 9);
                    let _bb = indicators.bollinger_bands(black_box(&bars), 20, dec!(2.0));
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Streaming Update (Single Bar Addition)
// ============================================================================

fn bench_streaming_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_update");

    let bars = generate_bars(100);
    let new_bar = generate_bars(1)[0].clone();

    group.bench_function("update_with_new_bar", |b| {
        let indicators = TechnicalIndicators::new();

        b.iter(|| {
            let mut bars_copy = bars.clone();
            bars_copy.push(black_box(new_bar.clone()));

            // Recalculate indicators
            let _sma = indicators.sma(&bars_copy, 20);
            let _rsi = indicators.rsi(&bars_copy, 14);
        });
    });

    group.finish();
}

// ============================================================================
// Benchmarks - Memory Allocation Overhead
// ============================================================================

fn bench_feature_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_allocation");

    for size in [100, 500, 1000].iter() {
        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let bars = generate_bars(size);

                b.iter(|| {
                    // Allocate vectors for results
                    let _sma: Vec<Option<Decimal>> = Vec::with_capacity(bars.len());
                    let _ema: Vec<Option<Decimal>> = Vec::with_capacity(bars.len());
                    let _rsi: Vec<Option<Decimal>> = Vec::with_capacity(bars.len());
                });
            },
        );
    }

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
        bench_sma_calculation,
        bench_ema_calculation,
        bench_rsi_calculation,
        bench_macd_calculation,
        bench_bollinger_bands,
        bench_full_feature_extraction,
        bench_streaming_update,
        bench_feature_allocation
}

criterion_main!(benches);
