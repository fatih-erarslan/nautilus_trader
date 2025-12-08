//! Benchmarks for regime detection performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use regime_detector::{RegimeDetector, types::RegimeConfig};
use rand::prelude::*;

fn generate_test_data(size: usize, trend: f32, volatility: f32) -> (Vec<f32>, Vec<f32>) {
    let mut rng = thread_rng();
    let mut prices = Vec::with_capacity(size);
    let mut volumes = Vec::with_capacity(size);
    
    let mut price = 100.0;
    
    for i in 0..size {
        // Add trend and random walk
        let noise = rng.gen::<f32>() - 0.5;
        price += trend + volatility * noise;
        prices.push(price);
        
        // Generate volume with some correlation to price movement
        let volume = 1000.0 + rng.gen::<f32>() * 500.0;
        volumes.push(volume);
    }
    
    (prices, volumes)
}

fn bench_regime_detection(c: &mut Criterion) {
    let detector = RegimeDetector::new();
    
    let sizes = [10, 50, 100, 500, 1000];
    
    let mut group = c.benchmark_group("regime_detection_by_size");
    
    for size in sizes {
        let (prices, volumes) = generate_test_data(size, 0.01, 0.02);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(
            BenchmarkId::new("detect_regime", size),
            &size,
            |b, _| {
                b.iter(|| {
                    black_box(detector.detect_regime(
                        black_box(&prices),
                        black_box(&volumes)
                    ))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_simd_operations(c: &mut Criterion) {
    use regime_detector::simd_ops::*;
    
    let sizes = [100, 500, 1000, 5000];
    let mut group = c.benchmark_group("simd_operations");
    
    for size in sizes {
        let data: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(
            BenchmarkId::new("simd_mean", size),
            &size,
            |b, _| {
                b.iter(|| black_box(simd_mean(black_box(&data))))
            },
        );
        
        let mean = simd_mean(&data);
        group.bench_with_input(
            BenchmarkId::new("simd_variance", size),
            &size,
            |b, _| {
                b.iter(|| black_box(simd_variance(black_box(&data), black_box(mean))))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("simd_linear_slope", size),
            &size,
            |b, _| {
                b.iter(|| black_box(simd_linear_slope(black_box(&data))))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("simd_autocorrelation", size),
            &size,
            |b, _| {
                b.iter(|| black_box(simd_autocorrelation(black_box(&data), 1)))
            },
        );
    }
    
    group.finish();
}

fn bench_cache_performance(c: &mut Criterion) {
    let detector_with_cache = RegimeDetector::with_config(RegimeConfig {
        enable_cache: true,
        cache_size: 1000,
        ..Default::default()
    });
    
    let detector_no_cache = RegimeDetector::with_config(RegimeConfig {
        enable_cache: false,
        ..Default::default()
    });
    
    let (prices, volumes) = generate_test_data(100, 0.01, 0.02);
    
    let mut group = c.benchmark_group("cache_comparison");
    
    group.bench_function("with_cache_first_call", |b| {
        b.iter(|| {
            black_box(detector_with_cache.detect_regime(
                black_box(&prices),
                black_box(&volumes)
            ))
        })
    });
    
    // Prime the cache
    detector_with_cache.detect_regime(&prices, &volumes);
    
    group.bench_function("with_cache_cached_call", |b| {
        b.iter(|| {
            black_box(detector_with_cache.detect_regime(
                black_box(&prices),
                black_box(&volumes)
            ))
        })
    });
    
    group.bench_function("without_cache", |b| {
        b.iter(|| {
            black_box(detector_no_cache.detect_regime(
                black_box(&prices),
                black_box(&volumes)
            ))
        })
    });
    
    group.finish();
}

fn bench_streaming_detection(c: &mut Criterion) {
    let detector = RegimeDetector::new();
    let (mut prices, mut volumes) = generate_test_data(100, 0.01, 0.02);
    
    // Remove last element for streaming
    let new_price = prices.pop().unwrap();
    let new_volume = volumes.pop().unwrap();
    
    c.bench_function("streaming_detection", |b| {
        b.iter(|| {
            black_box(detector.detect_regime_streaming(
                black_box(&prices),
                black_box(&volumes),
                black_box(new_price),
                black_box(new_volume),
            ))
        })
    });
}

fn bench_batch_detection(c: &mut Criterion) {
    let detector = RegimeDetector::new();
    
    let batch_sizes = [1, 5, 10, 20, 50];
    let mut group = c.benchmark_group("batch_detection");
    
    for batch_size in batch_sizes {
        let windows: Vec<(Vec<f32>, Vec<f32>)> = (0..batch_size)
            .map(|_| generate_test_data(100, 0.01, 0.02))
            .collect();
        
        let window_refs: Vec<(&[f32], &[f32])> = windows.iter()
            .map(|(p, v)| (p.as_slice(), v.as_slice()))
            .collect();
        
        group.throughput(Throughput::Elements(batch_size as u64));
        group.bench_with_input(
            BenchmarkId::new("batch_detection", batch_size),
            &batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(detector.detect_regime_batch(black_box(&window_refs)))
                })
            },
        );
    }
    
    group.finish();
}

fn bench_latency_distribution(c: &mut Criterion) {
    let detector = RegimeDetector::new();
    let (prices, volumes) = generate_test_data(100, 0.01, 0.02);
    
    // Measure latency distribution
    let mut group = c.benchmark_group("latency_analysis");
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(10000);
    
    group.bench_function("single_detection_latency", |b| {
        b.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                black_box(detector.detect_regime(
                    black_box(&prices),
                    black_box(&volumes)
                ));
            }
            start.elapsed()
        })
    });
    
    group.finish();
}

fn bench_different_market_conditions(c: &mut Criterion) {
    let detector = RegimeDetector::new();
    
    let conditions = [
        ("trending_bull", 0.05, 0.01),
        ("trending_bear", -0.05, 0.01),
        ("ranging_low_vol", 0.0, 0.005),
        ("ranging_high_vol", 0.0, 0.05),
        ("volatile_trending", 0.02, 0.08),
    ];
    
    let mut group = c.benchmark_group("market_conditions");
    
    for (name, trend, volatility) in conditions {
        let (prices, volumes) = generate_test_data(100, trend, volatility);
        
        group.bench_function(name, |b| {
            b.iter(|| {
                black_box(detector.detect_regime(
                    black_box(&prices),
                    black_box(&volumes)
                ))
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_regime_detection,
    bench_simd_operations,
    bench_cache_performance,
    bench_streaming_detection,
    bench_batch_detection,
    bench_latency_distribution,
    bench_different_market_conditions
);

criterion_main!(benches);