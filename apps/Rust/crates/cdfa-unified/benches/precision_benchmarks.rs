//! Benchmarks for high-precision numerical algorithms
//!
//! This benchmark suite measures the performance of Kahan summation and
//! related algorithms while validating their precision improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_unified::precision::kahan::{KahanAccumulator, NeumaierAccumulator};

#[cfg(feature = "simd")]
use cdfa_unified::precision::kahan::simd;

/// Generate test data for benchmarking
fn generate_test_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64).sin() * 1000.0).collect()
}

/// Generate pathological test data that exposes precision issues
fn generate_pathological_data(n: usize) -> Vec<f64> {
    let mut data = Vec::with_capacity(n * 2);
    for _ in 0..n {
        data.push(1e16);
        data.push(1.0);
    }
    for _ in 0..n {
        data.push(-1e16);
    }
    data
}

/// Benchmark naive summation
fn bench_naive_sum(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("naive_summation");
    
    for size in sizes {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| {
                let sum: f64 = black_box(data).iter().sum();
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark Kahan summation
fn bench_kahan_sum(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("kahan_summation");
    
    for size in sizes {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("kahan", size), &data, |b, data| {
            b.iter(|| {
                let sum = KahanAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark Neumaier summation
fn bench_neumaier_sum(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("neumaier_summation");
    
    for size in sizes {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("neumaier", size), &data, |b, data| {
            b.iter(|| {
                let sum = NeumaierAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark SIMD Kahan summation if available
#[cfg(feature = "simd")]
fn bench_simd_kahan_sum(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("simd_kahan_summation");
    
    for size in sizes {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("simd_kahan", size), &data, |b, data| {
            b.iter(|| {
                let sum = simd::kahan_sum_simd(black_box(data));
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark parallel SIMD Kahan summation if available
#[cfg(all(feature = "simd", feature = "parallel"))]
fn bench_parallel_simd_kahan_sum(c: &mut Criterion) {
    let sizes = vec![10000, 100000, 1000000];
    
    let mut group = c.benchmark_group("parallel_simd_kahan_summation");
    
    for size in sizes {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Elements(size as u64));
        group.bench_with_input(BenchmarkId::new("parallel_simd_kahan", size), &data, |b, data| {
            b.iter(|| {
                let sum = simd::kahan_sum_parallel(black_box(data));
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark accumulator operations
fn bench_accumulator_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("accumulator_operations");
    
    let data = generate_test_data(10000);
    
    group.bench_function("kahan_incremental", |b| {
        b.iter(|| {
            let mut acc = KahanAccumulator::new();
            for &value in black_box(&data) {
                acc.add(value);
            }
            black_box(acc.sum())
        })
    });
    
    group.bench_function("neumaier_incremental", |b| {
        b.iter(|| {
            let mut acc = NeumaierAccumulator::new();
            for &value in black_box(&data) {
                acc.add(value);
            }
            black_box(acc.sum())
        })
    });
    
    group.bench_function("kahan_from_iter", |b| {
        b.iter(|| {
            let acc = KahanAccumulator::from_iter(black_box(&data).iter().copied());
            black_box(acc.sum())
        })
    });
    
    group.finish();
}

/// Benchmark precision vs performance trade-offs
fn bench_precision_pathological_cases(c: &mut Criterion) {
    let sizes = vec![100, 1000, 10000];
    
    let mut group = c.benchmark_group("pathological_precision");
    
    for size in sizes {
        let data = generate_pathological_data(size);
        
        group.throughput(Throughput::Elements(data.len() as u64));
        
        group.bench_with_input(BenchmarkId::new("naive_pathological", size), &data, |b, data| {
            b.iter(|| {
                let sum: f64 = black_box(data).iter().sum();
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("kahan_pathological", size), &data, |b, data| {
            b.iter(|| {
                let sum = KahanAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("neumaier_pathological", size), &data, |b, data| {
            b.iter(|| {
                let sum = NeumaierAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark financial calculations
fn bench_financial_calculations(c: &mut Criterion) {
    use cdfa_unified::algorithms::math_utils::financial::*;
    use cdfa_unified::precision::stable::*;
    
    let mut group = c.benchmark_group("financial_calculations");
    
    // Portfolio calculations
    let weights = vec![0.1, 0.2, 0.3, 0.25, 0.15];
    let returns = vec![0.05, 0.08, -0.02, 0.12, 0.03];
    
    group.bench_function("portfolio_returns", |b| {
        b.iter(|| {
            let result = portfolio_returns(black_box(&weights), black_box(&returns)).unwrap();
            black_box(result)
        })
    });
    
    // Risk calculations
    let large_returns: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 0.01).collect();
    
    group.bench_function("sharpe_ratio", |b| {
        b.iter(|| {
            let result = sharpe_ratio(black_box(&large_returns), 0.01).unwrap();
            black_box(result)
        })
    });
    
    group.bench_function("value_at_risk", |b| {
        b.iter(|| {
            let result = value_at_risk(black_box(&large_returns), 0.95).unwrap();
            black_box(result)
        })
    });
    
    // Variance calculations
    group.bench_function("welford_variance", |b| {
        b.iter(|| {
            let (mean, var) = welford_variance(black_box(&large_returns)).unwrap();
            black_box((mean, var))
        })
    });
    
    group.finish();
}

/// Benchmark stable mathematical operations
fn bench_stable_operations(c: &mut Criterion) {
    use cdfa_unified::precision::stable::*;
    
    let mut group = c.benchmark_group("stable_operations");
    
    let large_values = vec![1000.0, 1001.0, 1002.0, 999.0, 1003.0];
    
    group.bench_function("logsumexp", |b| {
        b.iter(|| {
            let result = logsumexp_stable(black_box(&large_values)).unwrap();
            black_box(result)
        })
    });
    
    group.bench_function("softmax", |b| {
        b.iter(|| {
            let result = softmax_stable(black_box(&large_values)).unwrap();
            black_box(result)
        })
    });
    
    let x_values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.1).collect();
    let y_values: Vec<f64> = (0..1000).map(|i| (i as f64) * 0.15 + 1.0).collect();
    
    group.bench_function("stable_correlation", |b| {
        b.iter(|| {
            let result = stable_correlation(black_box(&x_values), black_box(&y_values)).unwrap();
            black_box(result)
        })
    });
    
    let positive_values: Vec<f64> = (1..1000).map(|i| i as f64).collect();
    
    group.bench_function("stable_geometric_mean", |b| {
        b.iter(|| {
            let result = stable_geometric_mean(black_box(&positive_values)).unwrap();
            black_box(result)
        })
    });
    
    group.finish();
}

// Group all benchmarks
criterion_group!(
    precision_benches,
    bench_naive_sum,
    bench_kahan_sum,
    bench_neumaier_sum,
    bench_accumulator_operations,
    bench_precision_pathological_cases,
    bench_financial_calculations,
    bench_stable_operations
);

// Add SIMD benchmarks if feature is enabled
#[cfg(feature = "simd")]
criterion_group!(
    simd_benches,
    bench_simd_kahan_sum
);

// Add parallel SIMD benchmarks if features are enabled
#[cfg(all(feature = "simd", feature = "parallel"))]
criterion_group!(
    parallel_simd_benches,
    bench_parallel_simd_kahan_sum
);

// Main benchmark runner
#[cfg(all(feature = "simd", feature = "parallel"))]
criterion_main!(precision_benches, simd_benches, parallel_simd_benches);

#[cfg(all(feature = "simd", not(feature = "parallel")))]
criterion_main!(precision_benches, simd_benches);

#[cfg(not(feature = "simd"))]
criterion_main!(precision_benches);