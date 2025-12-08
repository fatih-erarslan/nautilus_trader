//! Simplified benchmarks for Kahan summation precision and performance

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_unified::precision::kahan_simple::{KahanAccumulator, NeumaierAccumulator, financial};

/// Generate test data
fn generate_test_data(n: usize) -> Vec<f64> {
    (0..n).map(|i| (i as f64).sin() * 1000.0).collect()
}

/// Generate pathological data that exposes precision issues
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

/// Benchmark naive summation vs Kahan summation
fn bench_summation_comparison(c: &mut Criterion) {
    let sizes = vec![1000, 10000, 100000];
    
    let mut group = c.benchmark_group("summation_comparison");
    
    for size in sizes {
        let data = generate_test_data(size);
        
        group.throughput(Throughput::Elements(size as u64));
        
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| {
                let sum: f64 = black_box(data).iter().sum();
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("kahan", size), &data, |b, data| {
            b.iter(|| {
                let sum = KahanAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("neumaier", size), &data, |b, data| {
            b.iter(|| {
                let sum = NeumaierAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
    }
    
    group.finish();
}

/// Benchmark pathological cases
fn bench_pathological_cases(c: &mut Criterion) {
    let sizes = vec![100, 1000, 10000];
    
    let mut group = c.benchmark_group("pathological_precision");
    
    for size in sizes {
        let data = generate_pathological_data(size);
        
        group.throughput(Throughput::Elements(data.len() as u64));
        
        group.bench_with_input(BenchmarkId::new("naive", size), &data, |b, data| {
            b.iter(|| {
                let sum: f64 = black_box(data).iter().sum();
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("kahan", size), &data, |b, data| {
            b.iter(|| {
                let sum = KahanAccumulator::sum_slice(black_box(data));
                black_box(sum)
            })
        });
        
        group.bench_with_input(BenchmarkId::new("neumaier", size), &data, |b, data| {
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
    let mut group = c.benchmark_group("financial_calculations");
    
    // Portfolio calculations
    let weights = vec![0.1, 0.2, 0.3, 0.25, 0.15];
    let returns = vec![0.05, 0.08, -0.02, 0.12, 0.03];
    
    group.bench_function("portfolio_return", |b| {
        b.iter(|| {
            let result = financial::portfolio_return(black_box(&weights), black_box(&returns)).unwrap();
            black_box(result)
        })
    });
    
    // Mean and variance calculations
    let large_returns: Vec<f64> = (0..1000).map(|i| (i as f64).sin() * 0.01).collect();
    
    group.bench_function("mean", |b| {
        b.iter(|| {
            let result = financial::mean(black_box(&large_returns)).unwrap();
            black_box(result)
        })
    });
    
    group.bench_function("variance", |b| {
        b.iter(|| {
            let result = financial::variance(black_box(&large_returns)).unwrap();
            black_box(result)
        })
    });
    
    group.finish();
}

/// Benchmark precision test case
fn bench_precision_test(c: &mut Criterion) {
    let mut group = c.benchmark_group("precision_test");
    
    let scales = vec![1e10, 1e12, 1e15, 1e16];
    
    for scale in scales {
        group.bench_with_input(BenchmarkId::new("precision_test", scale as u64), &scale, |b, &scale| {
            b.iter(|| {
                let result = financial::precision_test(black_box(scale));
                black_box(result)
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    precision_benches,
    bench_summation_comparison,
    bench_pathological_cases,
    bench_financial_calculations,
    bench_precision_test
);

criterion_main!(precision_benches);