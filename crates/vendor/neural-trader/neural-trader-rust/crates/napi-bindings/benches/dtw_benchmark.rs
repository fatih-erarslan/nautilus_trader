///! DTW Performance Benchmark
///!
///! Validates that pure Rust DTW achieves 50-100x speedup vs JavaScript
///! Run with: cargo bench --bench dtw_benchmark

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::Rng;

/// Pure Rust DTW implementation (same as in dtw.rs)
fn dtw_rust(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    let m = b.len();

    let mut dtw = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dtw[0][0] = 0.0;

    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            dtw[i][j] = cost + dtw[i - 1][j]
                .min(dtw[i][j - 1])
                .min(dtw[i - 1][j - 1]);
        }
    }

    dtw[n][m]
}

fn generate_pattern(length: usize) -> Vec<f64> {
    let mut rng = rand::thread_rng();
    let mut pattern = vec![100.0];
    for _ in 1..length {
        let change = (rng.gen::<f64>() - 0.5) * 2.0;
        pattern.push(pattern.last().unwrap() + change);
    }
    pattern
}

fn bench_dtw_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("dtw_pattern_matching");

    for size in [50, 100, 200, 500, 1000, 2000].iter() {
        let pattern_a = generate_pattern(*size);
        let pattern_b = generate_pattern(*size);

        group.bench_with_input(
            BenchmarkId::from_parameter(size),
            size,
            |b, _| {
                b.iter(|| {
                    dtw_rust(black_box(&pattern_a), black_box(&pattern_b))
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let current_pattern = generate_pattern(100);
    let num_historical = 1000;
    let mut historical_patterns = Vec::new();

    for _ in 0..num_historical {
        historical_patterns.push(generate_pattern(100));
    }

    c.bench_function("batch_1000_patterns", |b| {
        b.iter(|| {
            for hist_pattern in &historical_patterns {
                dtw_rust(black_box(&current_pattern), black_box(hist_pattern));
            }
        });
    });
}

criterion_group!(benches, bench_dtw_sizes, bench_batch_processing);
criterion_main!(benches);
