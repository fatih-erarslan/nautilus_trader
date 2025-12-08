//! Signal processing algorithm benchmarks
//!
//! Validates advanced CDFA algorithm performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_algorithms::prelude::*;
use std::time::Duration;

/// Generate test signals for benchmarks
fn generate_test_signal(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| (i as f64 * 0.1).sin() + (i as f64 * 0.05).cos() + (i as f64 * 0.01).tan())
        .collect()
}

/// Benchmark wavelet transforms
fn bench_wavelet_transforms(c: &mut Criterion) {
    let mut group = c.benchmark_group("wavelet_transforms");
    group.measurement_time(Duration::from_secs(15));
    
    for size in [256, 512, 1024, 2048].iter() {
        let signal = generate_test_signal(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("dwt_daubechies", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut output = vec![0.0; signal.len()];
                    dwt_daubechies(black_box(&signal), black_box(&mut output));
                    black_box(output)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("cwt_morlet", size),
            size,
            |b, _| {
                b.iter(|| {
                    let scales = vec![1.0, 2.0, 4.0, 8.0];
                    black_box(cwt_morlet(black_box(&signal), black_box(&scales)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark entropy calculations
fn bench_entropy_calculations(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy_calculations");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [128, 256, 512, 1024].iter() {
        let signal = generate_test_signal(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("sample_entropy", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(sample_entropy(black_box(&signal), 2, 0.2))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("approximate_entropy", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(approximate_entropy(black_box(&signal), 2, 0.2))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("permutation_entropy", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(permutation_entropy(black_box(&signal), 3))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark clustering algorithms
fn bench_clustering_algorithms(c: &mut Criterion) {
    let mut group = c.benchmark_group("clustering_algorithms");
    group.measurement_time(Duration::from_secs(20));
    
    for num_points in [100, 200, 500, 1000].iter() {
        let points: Vec<Vec<f64>> = (0..*num_points)
            .map(|i| vec![(i as f64).sin(), (i as f64).cos()])
            .collect();
        
        group.throughput(Throughput::Elements(*num_points as u64));
        group.bench_with_input(
            BenchmarkId::new("hierarchical_clustering", num_points),
            &points,
            |b, points| {
                b.iter(|| {
                    black_box(hierarchical_clustering(black_box(points), 5))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("volatility_clustering", num_points),
            &points,
            |b, points| {
                b.iter(|| {
                    let signal: Vec<f64> = points.iter().map(|p| p[0]).collect();
                    black_box(volatility_clustering(black_box(&signal), 20))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark P² quantile estimation
fn bench_p2_quantile(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2_quantile");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [1000, 5000, 10000, 20000].iter() {
        let data = generate_test_signal(*size);
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("p2_quantile_estimation", size),
            size,
            |b, _| {
                b.iter(|| {
                    let mut estimator = P2QuantileEstimator::new(0.5);
                    for &value in &data {
                        estimator.update(value);
                    }
                    black_box(estimator.quantile())
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark calibration methods
fn bench_calibration_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("calibration_methods");
    group.measurement_time(Duration::from_secs(15));
    
    for size in [200, 500, 1000, 2000].iter() {
        let predictions: Vec<f64> = (0..*size).map(|i| (i as f64 / *size as f64)).collect();
        let labels: Vec<bool> = (0..*size).map(|i| i % 2 == 0).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("isotonic_regression", size),
            &(predictions.clone(), labels.clone()),
            |b, (predictions, labels)| {
                b.iter(|| {
                    black_box(isotonic_regression(black_box(predictions), black_box(labels)))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("platt_scaling", size),
            &(predictions, labels),
            |b, (predictions, labels)| {
                b.iter(|| {
                    black_box(platt_scaling(black_box(predictions), black_box(labels)))
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_wavelet_transforms,
    bench_entropy_calculations,
    bench_clustering_algorithms,
    bench_p2_quantile,
    bench_calibration_methods,
);

criterion_main!(benches);