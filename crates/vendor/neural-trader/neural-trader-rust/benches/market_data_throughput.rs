// Market Data Throughput Benchmark
//
// Performance targets:
// - Tick ingestion: <100μs/tick
// - Parse throughput: >10,000 ticks/sec
// - Memory per connection: <10MB

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use nt_core::types::Symbol;
use nt_market_data::types::{Bar, Quote, Tick, Timeframe, Trade};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use std::time::Duration;

// ============================================================================
// Helper Functions - Generate Test Data
// ============================================================================

fn generate_tick(symbol: &str, price: Decimal) -> Tick {
    Tick {
        symbol: Symbol::new(symbol).unwrap(),
        price,
        size: dec!(100),
        timestamp: chrono::Utc::now(),
        exchange: "NASDAQ".to_string(),
        conditions: vec![],
    }
}

fn generate_ticks(count: usize) -> Vec<Tick> {
    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];
    let mut ticks = Vec::with_capacity(count);

    for i in 0..count {
        let symbol = symbols[i % symbols.len()];
        let price = dec!(100) + Decimal::from(i % 100);
        ticks.push(generate_tick(symbol, price));
    }

    ticks
}

fn generate_quote(symbol: &str) -> Quote {
    Quote {
        symbol: Symbol::new(symbol).unwrap(),
        bid_price: dec!(100.50),
        ask_price: dec!(100.52),
        bid_size: dec!(100),
        ask_size: dec!(150),
        timestamp: chrono::Utc::now(),
        exchange: "NASDAQ".to_string(),
    }
}

fn generate_bars(count: usize) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(count);
    let base_time = chrono::Utc::now();

    for i in 0..count {
        let base_price = dec!(100) + Decimal::from(i);
        bars.push(Bar {
            symbol: Symbol::new("AAPL").unwrap(),
            timestamp: base_time + chrono::Duration::minutes(i as i64),
            open: base_price,
            high: base_price + dec!(2),
            low: base_price - dec!(1),
            close: base_price + dec!(0.5),
            volume: dec!(1000000),
            trade_count: 1000,
            vwap: Some(base_price + dec!(0.25)),
            timeframe: Timeframe::Minute,
        });
    }

    bars
}

// ============================================================================
// Benchmarks - Tick Ingestion
// ============================================================================

fn bench_tick_ingestion(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_ingestion");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let ticks = generate_ticks(size);

                b.iter(|| {
                    let mut processed = 0;
                    for tick in black_box(&ticks) {
                        // Simulate tick processing
                        let _price = tick.price;
                        let _symbol = &tick.symbol;
                        processed += 1;
                    }
                    processed
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Tick Parsing (JSON)
// ============================================================================

fn bench_tick_parsing(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_parsing");

    // Sample JSON message from Alpaca
    let json_data = r#"{
        "T": "t",
        "S": "AAPL",
        "p": 150.25,
        "s": 100,
        "t": "2025-11-12T20:00:00Z",
        "x": "NASDAQ",
        "c": []
    }"#;

    group.bench_function("parse_single_tick", |b| {
        b.iter(|| {
            let _parsed: Result<serde_json::Value, _> =
                serde_json::from_str(black_box(json_data));
        });
    });

    // Batch parsing
    for batch_size in [10, 100, 1000].iter() {
        let batch_json: Vec<_> = (0..*batch_size)
            .map(|_| json_data)
            .collect();

        group.bench_with_input(
            BenchmarkId::new("parse_batch", batch_size),
            &batch_json,
            |b, batch| {
                b.iter(|| {
                    let mut parsed = 0;
                    for json in black_box(batch) {
                        if serde_json::from_str::<serde_json::Value>(json).is_ok() {
                            parsed += 1;
                        }
                    }
                    parsed
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Quote Processing
// ============================================================================

fn bench_quote_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("quote_processing");

    for size in [100, 1_000, 10_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, &size| {
                let quotes: Vec<_> = (0..size)
                    .map(|i| {
                        let symbol = format!("SYM{}", i % 100);
                        generate_quote(&symbol)
                    })
                    .collect();

                b.iter(|| {
                    let mut total_spread = Decimal::ZERO;
                    for quote in black_box(&quotes) {
                        let spread = quote.ask_price - quote.bid_price;
                        total_spread += spread;
                    }
                    total_spread
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Bar Aggregation
// ============================================================================

fn bench_bar_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("bar_aggregation");

    for size in [60, 300, 1440].iter() {  // 1 hour, 5 hours, 1 day
        group.bench_with_input(
            BenchmarkId::new("aggregate_bars", size),
            size,
            |b, &size| {
                let bars = generate_bars(size);

                b.iter(|| {
                    // Calculate OHLCV aggregates
                    let mut high = Decimal::MIN;
                    let mut low = Decimal::MAX;
                    let mut volume = Decimal::ZERO;

                    for bar in black_box(&bars) {
                        if bar.high > high {
                            high = bar.high;
                        }
                        if bar.low < low {
                            low = bar.low;
                        }
                        volume += bar.volume;
                    }

                    (high, low, volume)
                });
            },
        );
    }

    group.finish();
}

// ============================================================================
// Benchmarks - Memory Allocation
// ============================================================================

fn bench_tick_allocation(c: &mut Criterion) {
    let mut group = c.benchmark_group("tick_allocation");

    group.bench_function("allocate_single_tick", |b| {
        b.iter(|| {
            let tick = black_box(generate_tick("AAPL", dec!(150)));
            drop(tick);
        });
    });

    for size in [100, 1_000, 10_000].iter() {
        group.bench_with_input(
            BenchmarkId::new("allocate_tick_vec", size),
            size,
            |b, &size| {
                b.iter(|| {
                    let ticks = black_box(generate_ticks(size));
                    drop(ticks);
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
        bench_tick_ingestion,
        bench_tick_parsing,
        bench_quote_processing,
        bench_bar_aggregation,
        bench_tick_allocation
}

criterion_main!(benches);
