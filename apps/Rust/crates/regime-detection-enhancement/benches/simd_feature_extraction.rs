use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use regime_detection_enhancement::*;
use std::time::Duration;

fn benchmark_simd_feature_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD Feature Extraction");
    group.measurement_time(Duration::from_secs(10));
    
    for size in [32, 128, 512, 1024, 4096].iter() {
        let market_data: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.001).sin()).collect();
        let config = simd_optimizer::SIMDOptimizerConfig::default();
        let mut optimizer = simd_optimizer::SIMDOptimizer::new(config).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("vectorized_momentum", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimizer.compute_momentum_vectorized(black_box(&market_data)).unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("vectorized_volatility", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimizer.compute_volatility_vectorized(black_box(&market_data)).unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("vectorized_correlation", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimizer.compute_correlation_vectorized(black_box(&market_data)).unwrap()
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("full_feature_extraction", size),
            size,
            |b, _| {
                b.iter(|| {
                    optimizer.extract_features_vectorized(black_box(&market_data)).unwrap()
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_simd_vs_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("SIMD vs Scalar");
    group.measurement_time(Duration::from_secs(8));
    
    let size = 1024;
    let market_data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).sin()).collect();
    let config = simd_optimizer::SIMDOptimizerConfig::default();
    let mut optimizer = simd_optimizer::SIMDOptimizer::new(config).unwrap();
    
    group.bench_function("simd_momentum", |b| {
        b.iter(|| {
            optimizer.compute_momentum_vectorized(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("scalar_momentum", |b| {
        b.iter(|| {
            optimizer.compute_momentum_scalar(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("simd_volatility", |b| {
        b.iter(|| {
            optimizer.compute_volatility_vectorized(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("scalar_volatility", |b| {
        b.iter(|| {
            optimizer.compute_volatility_scalar(black_box(&market_data)).unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_memory_layouts(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory Layout Optimization");
    group.measurement_time(Duration::from_secs(8));
    
    let size = 2048;
    let market_data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).sin()).collect();
    let config = simd_optimizer::SIMDOptimizerConfig::default();
    let mut optimizer = simd_optimizer::SIMDOptimizer::new(config).unwrap();
    
    group.bench_function("aligned_memory", |b| {
        b.iter(|| {
            optimizer.process_aligned_data(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("unaligned_memory", |b| {
        b.iter(|| {
            optimizer.process_unaligned_data(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("chunked_processing", |b| {
        b.iter(|| {
            optimizer.process_chunked_data(black_box(&market_data)).unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_feature_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("Feature Types");
    group.measurement_time(Duration::from_secs(10));
    
    let size = 1024;
    let market_data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).sin()).collect();
    let config = simd_optimizer::SIMDOptimizerConfig::default();
    let mut optimizer = simd_optimizer::SIMDOptimizer::new(config).unwrap();
    
    group.bench_function("technical_indicators", |b| {
        b.iter(|| {
            optimizer.compute_technical_indicators(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("statistical_features", |b| {
        b.iter(|| {
            optimizer.compute_statistical_features(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("spectral_features", |b| {
        b.iter(|| {
            optimizer.compute_spectral_features(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("time_series_features", |b| {
        b.iter(|| {
            optimizer.compute_time_series_features(black_box(&market_data)).unwrap()
        })
    });
    
    group.finish();
}

fn benchmark_parallel_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("Parallel Processing");
    group.measurement_time(Duration::from_secs(12));
    
    let size = 4096;
    let market_data: Vec<f64> = (0..size).map(|i| (i as f64 * 0.001).sin()).collect();
    let config = simd_optimizer::SIMDOptimizerConfig::default();
    let mut optimizer = simd_optimizer::SIMDOptimizer::new(config).unwrap();
    
    group.bench_function("single_threaded", |b| {
        b.iter(|| {
            optimizer.process_single_threaded(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("multi_threaded", |b| {
        b.iter(|| {
            optimizer.process_multi_threaded(black_box(&market_data)).unwrap()
        })
    });
    
    group.bench_function("simd_parallel", |b| {
        b.iter(|| {
            optimizer.process_simd_parallel(black_box(&market_data)).unwrap()
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_simd_feature_extraction,
    benchmark_simd_vs_scalar,
    benchmark_memory_layouts,
    benchmark_feature_types,
    benchmark_parallel_processing
);
criterion_main!(benches);