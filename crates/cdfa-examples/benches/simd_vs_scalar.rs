//! SIMD vs Scalar performance comparison
//!
//! Demonstrates SIMD performance benefits

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_simd::unified;
use std::time::Duration;

/// Benchmark SIMD speedup over scalar implementations
fn bench_simd_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_speedup");
    group.measurement_time(Duration::from_secs(20));
    
    for size in [256, 512, 1024, 2048, 4096].iter() {
        let x: Vec<f64> = (0..*size).map(|i| (i as f64).sin()).collect();
        let y: Vec<f64> = (0..*size).map(|i| (i as f64).cos()).collect();
        
        group.throughput(Throughput::Elements(*size as u64));
        
        // SIMD correlation
        group.bench_with_input(
            BenchmarkId::new("simd_correlation", size),
            size,
            |b, _| {
                b.iter(|| {
                    black_box(unified::correlation(black_box(&x), black_box(&y)))
                })
            },
        );
        
        // Scalar correlation for comparison
        group.bench_with_input(
            BenchmarkId::new("scalar_correlation", size),
            size,
            |b, _| {
                b.iter(|| {
                    let n = x.len() as f64;
                    let sum_x: f64 = x.iter().sum();
                    let sum_y: f64 = y.iter().sum();
                    let sum_xx: f64 = x.iter().map(|&v| v * v).sum();
                    let sum_yy: f64 = y.iter().map(|&v| v * v).sum();
                    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
                    
                    let numerator = n * sum_xy - sum_x * sum_y;
                    let denominator = ((n * sum_xx - sum_x * sum_x) * (n * sum_yy - sum_y * sum_y)).sqrt();
                    
                    black_box(if denominator > 0.0 { numerator / denominator } else { 0.0 })
                })
            },
        );
    }
    
    group.finish();
}

criterion_group!(benches, bench_simd_speedup);
criterion_main!(benches);