//! Comprehensive SIMD performance benchmarks
//!
//! Validates that all SIMD implementations meet performance targets

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_simd::{unified, benchmarks, detect_cpu_features, best_implementation};
use std::time::Duration;

/// Generate test data for benchmarks
fn generate_test_data(size: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..size).map(|i| (i as f64).sin()).collect();
    let y: Vec<f64> = (0..size).map(|i| (i as f64).cos()).collect();
    let probabilities: Vec<f64> = (0..size).map(|_| 1.0 / size as f64).collect();
    
    (x, y, probabilities)
}

/// Benchmark correlation calculation across different vector sizes
fn bench_correlation(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024, 2048].iter() {
        let (x, y, _) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(unified::correlation(black_box(&x), black_box(&y)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark wavelet transform across different vector sizes
fn bench_dwt_haar(c: &mut Criterion) {
    let mut group = c.benchmark_group("dwt_haar");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024, 2048].iter() {
        let (x, _, _) = generate_test_data(*size);
        let mut approx = vec![0.0; size / 2];
        let mut detail = vec![0.0; size / 2];
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    unified::dwt_haar(
                        black_box(&x),
                        black_box(&mut approx),
                        black_box(&mut detail),
                    )
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Euclidean distance calculation
fn bench_euclidean_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("euclidean_distance");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024, 2048].iter() {
        let (x, y, _) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(unified::euclidean_distance(black_box(&x), black_box(&y)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark signal fusion
fn bench_signal_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("signal_fusion");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let (x, y, _) = generate_test_data(*size);
        let signals = vec![x.as_slice(), y.as_slice()];
        let weights = vec![0.6, 0.4];
        let mut output = vec![0.0; *size];
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    unified::signal_fusion(
                        black_box(&signals),
                        black_box(&weights),
                        black_box(&mut output),
                    )
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark Shannon entropy calculation
fn bench_shannon_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("shannon_entropy");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024].iter() {
        let (_, _, probabilities) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(unified::shannon_entropy(black_box(&probabilities)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark moving average calculation
fn bench_moving_average(c: &mut Criterion) {
    let mut group = c.benchmark_group("moving_average");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [128, 256, 512, 1024].iter() {
        let (x, _, _) = generate_test_data(*size);
        let window = 20;
        let mut output = vec![0.0; size - window + 1];
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    unified::moving_average(
                        black_box(&x),
                        black_box(window),
                        black_box(&mut output),
                    )
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark variance calculation
fn bench_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("variance");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [64, 128, 256, 512, 1024, 2048].iter() {
        let (x, _, _) = generate_test_data(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("unified", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(unified::variance(black_box(&x)))
                })
            },
        );
    }
    
    group.finish();
}

/// Performance validation benchmark
fn bench_performance_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_validation");
    group.measurement_time(Duration::from_secs(30));
    
    // Print system information
    let features = detect_cpu_features();
    let implementation = best_implementation();
    
    println!("\n=== SIMD Performance Benchmark ===");
    println!("CPU Features: {:?}", features);
    println!("Best Implementation: {:?}", implementation);
    println!("Cache Line Size: {} bytes", features.cache_line_size);
    println!("L1 Data Cache: {} KB", features.l1_data_cache_size / 1024);
    println!("L2 Cache: {} KB", features.l2_cache_size / 1024);
    
    group.bench_function("full_benchmark_suite", |b| {
        b.iter(|| {
            let results = benchmarks::run_benchmarks();
            black_box(results)
        })
    });
    
    group.bench_function("performance_target_validation", |b| {
        b.iter(|| {
            let meets_targets = benchmarks::validate_performance_targets();
            black_box(meets_targets)
        })
    });
    
    group.finish();
    
    // Run validation and print results
    let results = benchmarks::run_benchmarks();
    let meets_targets = benchmarks::validate_performance_targets();
    
    println!("\n=== Performance Results ===");
    println!("Correlation: {} ns", results.correlation_ns);
    println!("DWT Haar: {} ns", results.dwt_haar_ns);
    println!("Euclidean Distance: {} ns", results.euclidean_distance_ns);
    println!("Signal Fusion: {} ns", results.signal_fusion_ns);
    println!("Shannon Entropy: {} ns", results.shannon_entropy_ns);
    println!("Moving Average: {} ns", results.moving_average_ns);
    println!("Variance: {} ns", results.variance_ns);
    println!("Performance Targets Met: {}", meets_targets);
    
    // Performance targets by implementation
    let targets = match implementation {
        #[cfg(target_arch = "x86_64")]
        cdfa_simd::SimdImplementation::Avx512 => (50, 50, 25, 200, 100, 100, 100),
        #[cfg(target_arch = "x86_64")]
        cdfa_simd::SimdImplementation::Avx2 => (100, 100, 50, 200, 150, 100, 100),
        #[cfg(target_arch = "aarch64")]
        cdfa_simd::SimdImplementation::Neon => (150, 150, 100, 300, 200, 150, 150),
        #[cfg(target_arch = "wasm32")]
        cdfa_simd::SimdImplementation::WasmSimd => (200, 200, 150, 400, 300, 200, 200),
        _ => (1000, 1000, 500, 1000, 1000, 500, 500),
    };
    
    println!("\n=== Performance Target Analysis ===");
    println!("Correlation: {} ns (target: {} ns) - {}", 
        results.correlation_ns, targets.0, 
        if results.correlation_ns <= targets.0 { "✓ PASS" } else { "✗ FAIL" });
    println!("DWT Haar: {} ns (target: {} ns) - {}", 
        results.dwt_haar_ns, targets.1,
        if results.dwt_haar_ns <= targets.1 { "✓ PASS" } else { "✗ FAIL" });
    println!("Euclidean Distance: {} ns (target: {} ns) - {}", 
        results.euclidean_distance_ns, targets.2,
        if results.euclidean_distance_ns <= targets.2 { "✓ PASS" } else { "✗ FAIL" });
    println!("Signal Fusion: {} ns (target: {} ns) - {}", 
        results.signal_fusion_ns, targets.3,
        if results.signal_fusion_ns <= targets.3 { "✓ PASS" } else { "✗ FAIL" });
    println!("Shannon Entropy: {} ns (target: {} ns) - {}", 
        results.shannon_entropy_ns, targets.4,
        if results.shannon_entropy_ns <= targets.4 { "✓ PASS" } else { "✗ FAIL" });
    println!("Moving Average: {} ns (target: {} ns) - {}", 
        results.moving_average_ns, targets.5,
        if results.moving_average_ns <= targets.5 { "✓ PASS" } else { "✗ FAIL" });
    println!("Variance: {} ns (target: {} ns) - {}", 
        results.variance_ns, targets.6,
        if results.variance_ns <= targets.6 { "✓ PASS" } else { "✗ FAIL" });
        
    if meets_targets {
        println!("\n🎉 ALL PERFORMANCE TARGETS MET! 🎉");
        println!("SIMD optimizations are delivering expected performance gains.");
    } else {
        println!("\n⚠️  Some performance targets not met.");
        println!("This may be due to debug build or CPU limitations.");
    }
}

criterion_group!(
    benches,
    bench_correlation,
    bench_dwt_haar,
    bench_euclidean_distance,
    bench_signal_fusion,
    bench_shannon_entropy,
    bench_moving_average,
    bench_variance,
    bench_performance_validation,
);

criterion_main!(benches);