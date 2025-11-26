// Example Benchmark - Market Data Processing
// Location: benches/example_benchmark.rs
//
// This file demonstrates best practices for benchmarking in the Neural Trading system.
// Run with: cargo bench --bench example_benchmark

use criterion::{
    black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput,
};
use std::time::Duration;

// ============================================================================
// MOCK TYPES (Replace with actual types from your crate)
// ============================================================================

#[derive(Debug, Clone)]
struct Tick {
    symbol: String,
    price: f64,
    volume: f64,
    timestamp: i64,
}

impl Tick {
    fn new(symbol: &str, price: f64, volume: f64, timestamp: i64) -> Self {
        Self {
            symbol: symbol.to_string(),
            price,
            volume,
            timestamp,
        }
    }
}

// Simulate parsing from JSON
fn parse_tick_json(json: &str) -> Result<Tick, Box<dyn std::error::Error>> {
    // Simplified JSON parsing simulation
    Ok(Tick::new("BTC/USD", 50000.0, 1.5, 1234567890))
}

// Simulate parsing from MessagePack (binary)
fn parse_tick_msgpack(data: &[u8]) -> Result<Tick, Box<dyn std::error::Error>> {
    // Simplified MessagePack parsing simulation
    Ok(Tick::new("BTC/USD", 50000.0, 1.5, 1234567890))
}

// Simulate feature extraction
fn extract_features(ticks: &[Tick]) -> Vec<f64> {
    // Simplified feature extraction
    ticks
        .iter()
        .map(|t| t.price / t.volume)
        .collect()
}

// ============================================================================
// BENCHMARK 1: Comparing Serialization Formats
// ============================================================================

fn bench_serialization_formats(c: &mut Criterion) {
    let mut group = c.benchmark_group("serialization");

    // Test data
    let json_data = r#"{"symbol":"BTC/USD","price":50000.0,"volume":1.5,"timestamp":1234567890}"#;
    let msgpack_data: Vec<u8> = vec![0x84, 0xa6, 0x73, 0x79, 0x6d, 0x62, 0x6f, 0x6c]; // Simplified

    // Benchmark JSON parsing
    group.bench_function("json_parsing", |b| {
        b.iter(|| {
            // black_box prevents compiler optimizations from removing the code
            let tick = black_box(parse_tick_json(json_data).unwrap());
            black_box(tick)
        })
    });

    // Benchmark MessagePack parsing
    group.bench_function("msgpack_parsing", |b| {
        b.iter(|| {
            let tick = black_box(parse_tick_msgpack(&msgpack_data).unwrap());
            black_box(tick)
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 2: Throughput Testing
// ============================================================================

fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    // Configure throughput measurement
    group.throughput(Throughput::Elements(1000));

    // Generate test data
    let ticks: Vec<Tick> = (0..1000)
        .map(|i| Tick::new("BTC/USD", 50000.0 + i as f64, 1.5, 1234567890 + i))
        .collect();

    group.bench_function("process_1000_ticks", |b| {
        b.iter(|| {
            let features = black_box(extract_features(&ticks));
            black_box(features)
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 3: Parameterized Benchmarks
// ============================================================================

fn bench_parameterized(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_extraction_sizes");

    // Test with different window sizes
    for window_size in [10, 50, 100, 500, 1000].iter() {
        let ticks: Vec<Tick> = (0..*window_size)
            .map(|i| Tick::new("BTC/USD", 50000.0 + i as f64, 1.5, 1234567890 + i as i64))
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(window_size),
            window_size,
            |b, _size| {
                b.iter(|| {
                    let features = extract_features(&ticks);
                    black_box(features)
                })
            },
        );
    }

    group.finish();
}

// ============================================================================
// BENCHMARK 4: Latency Targets (Performance Gates)
// ============================================================================

fn bench_latency_targets(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_targets");

    // Configure for precise latency measurement
    group.sample_size(1000); // More samples for better precision
    group.measurement_time(Duration::from_secs(10)); // Longer measurement time

    // Target: Market data ingestion < 100μs
    group.bench_function("market_data_ingestion", |b| {
        let json = r#"{"symbol":"BTC/USD","price":50000.0,"volume":1.5,"timestamp":1234567890}"#;
        b.iter(|| {
            let tick = parse_tick_json(json).unwrap();
            black_box(tick)
        })
    });

    // Check if we met the target
    // Note: Actual threshold checking would be done in CI scripts
    // comparing against criterion's output JSON

    group.finish();
}

// ============================================================================
// BENCHMARK 5: Memory Allocation Tracking
// ============================================================================

fn bench_allocations(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocations");

    // Benchmark with minimal allocations (stack-only)
    group.bench_function("stack_only", |b| {
        b.iter(|| {
            let tick = Tick::new("BTC/USD", 50000.0, 1.5, 1234567890);
            black_box(tick)
        })
    });

    // Benchmark with heap allocations
    group.bench_function("heap_allocation", |b| {
        b.iter(|| {
            let ticks: Vec<Tick> = (0..100)
                .map(|i| Tick::new("BTC/USD", 50000.0 + i as f64, 1.5, 1234567890))
                .collect();
            black_box(ticks)
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 6: Async Operations
// ============================================================================

fn bench_async(c: &mut Criterion) {
    let mut group = c.benchmark_group("async_operations");

    // Setup async runtime
    let rt = tokio::runtime::Runtime::new().unwrap();

    // Benchmark async function
    group.bench_function("async_fetch", |b| {
        b.to_async(&rt).iter(|| async {
            // Simulate async operation
            tokio::time::sleep(Duration::from_micros(10)).await;
            black_box(42)
        })
    });

    group.finish();
}

// ============================================================================
// BENCHMARK 7: Comparison Baseline (Rust vs Python)
// ============================================================================

#[cfg(feature = "python-comparison")]
fn bench_python_comparison(c: &mut Criterion) {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    let mut group = c.benchmark_group("rust_vs_python");

    Python::with_gil(|py| {
        // Load Python implementation
        let sys = py.import("sys").unwrap();
        sys.getattr("path")
            .unwrap()
            .call_method1("append", ("../python_impl",))
            .unwrap();

        let python_module = py.import("neural_trader.indicators").unwrap();

        let prices = vec![
            44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89,
        ];

        // Benchmark Python implementation
        group.bench_function("python_sma", |b| {
            let py_prices = PyList::new(py, &prices);
            b.iter(|| {
                python_module
                    .call_method1("calculate_sma", (py_prices, 5))
                    .unwrap()
            })
        });

        // Benchmark Rust implementation
        group.bench_function("rust_sma", |b| {
            b.iter(|| {
                // Simplified SMA calculation
                let sum: f64 = prices.iter().sum();
                let avg = sum / prices.len() as f64;
                black_box(avg)
            })
        });
    });

    group.finish();
}

// ============================================================================
// CRITERION CONFIGURATION
// ============================================================================

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(100)                              // Number of samples
        .measurement_time(Duration::from_secs(10))     // How long to measure
        .warm_up_time(Duration::from_secs(3))          // Warm-up period
        .significance_level(0.05)                      // Statistical significance
        .noise_threshold(0.02);                        // 2% noise tolerance
    targets =
        bench_serialization_formats,
        bench_throughput,
        bench_parameterized,
        bench_latency_targets,
        bench_allocations,
        bench_async,
}

criterion_main!(benches);

// ============================================================================
// RUNNING BENCHMARKS
// ============================================================================
//
// Basic usage:
//   cargo bench --bench example_benchmark
//
// Run specific benchmark:
//   cargo bench --bench example_benchmark -- serialization
//
// Save baseline:
//   cargo bench --bench example_benchmark -- --save-baseline my_baseline
//
// Compare with baseline:
//   cargo bench --bench example_benchmark -- --baseline my_baseline
//
// Generate detailed output:
//   cargo bench --bench example_benchmark -- --verbose
//
// Profile with flamegraph:
//   cargo flamegraph --bench example_benchmark -- --bench
//
// ============================================================================
// PERFORMANCE TARGETS
// ============================================================================
//
// | Operation                | Target Latency | Target Throughput |
// |--------------------------|----------------|-------------------|
// | Market data ingestion    | < 100μs        | 10,000/sec        |
// | Feature extraction       | < 1ms          | 1,000/sec         |
// | Signal generation        | < 5ms          | 500/sec           |
// | Order placement          | < 10ms         | 100/sec           |
// | AgentDB query            | < 1ms          | 10,000/sec        |
//
// ============================================================================
// BENCHMARK BEST PRACTICES
// ============================================================================
//
// 1. Use black_box() to prevent compiler optimizations
// 2. Run on idle system for consistent results
// 3. Use adequate sample sizes (100+ for stable results)
// 4. Warm up before measuring (3+ seconds)
// 5. Set throughput for rate-based benchmarks
// 6. Compare against baselines regularly
// 7. Check statistical significance (p < 0.05)
// 8. Document performance targets
// 9. Use parameterized tests for different input sizes
// 10. Profile with flamegraph to identify bottlenecks
