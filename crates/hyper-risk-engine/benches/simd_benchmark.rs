//! SIMD operations benchmark
//!
//! Compares SIMD-optimized implementations against scalar fallbacks
//! to verify performance improvements.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyper_risk_engine::simd::{
    simd_cvar_historical, simd_drawdown_series, simd_portfolio_variance,
    simd_rolling_volatility, simd_var_historical, simd_covariance_matrix,
    simd_correlation_matrix, simd_matrix_multiply, simd_cholesky_decomposition,
};

fn benchmark_var_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("var_calculation");

    for size in [100, 1000, 10000].iter() {
        let returns: Vec<f64> = (0..*size)
            .map(|i| (i as f64 / *size as f64 - 0.5) * 0.1)
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &returns, |b, r| {
            b.iter(|| simd_var_historical(black_box(r), black_box(0.95)));
        });
    }

    group.finish();
}

fn benchmark_cvar_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cvar_calculation");

    for size in [100, 1000, 10000].iter() {
        let returns: Vec<f64> = (0..*size)
            .map(|i| (i as f64 / *size as f64 - 0.5) * 0.1)
            .collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &returns, |b, r| {
            b.iter(|| simd_cvar_historical(black_box(r), black_box(0.95)));
        });
    }

    group.finish();
}

fn benchmark_portfolio_variance(c: &mut Criterion) {
    let mut group = c.benchmark_group("portfolio_variance");

    for n_assets in [5, 10, 50].iter() {
        let weights: Vec<f64> = (0..*n_assets)
            .map(|_| 1.0 / *n_assets as f64)
            .collect();

        let mut covariance = vec![0.0; n_assets * n_assets];
        for i in 0..*n_assets {
            for j in 0..*n_assets {
                if i == j {
                    covariance[i * n_assets + j] = 0.04;
                } else {
                    covariance[i * n_assets + j] = 0.01;
                }
            }
        }

        group.throughput(Throughput::Elements((n_assets * n_assets) as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", n_assets),
            &(&weights, &covariance),
            |b, (w, c)| {
                b.iter(|| simd_portfolio_variance(black_box(w), black_box(c)));
            },
        );
    }

    group.finish();
}

fn benchmark_drawdown_series(c: &mut Criterion) {
    let mut group = c.benchmark_group("drawdown_series");

    for size in [100, 1000, 10000].iter() {
        let mut equity = Vec::with_capacity(*size);
        let mut value = 100.0;

        for i in 0..*size {
            value *= 1.0 + (i as f64 * 0.01).sin() * 0.02;
            equity.push(value);
        }

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::new("simd", size), &equity, |b, e| {
            b.iter(|| simd_drawdown_series(black_box(e)));
        });
    }

    group.finish();
}

fn benchmark_rolling_volatility(c: &mut Criterion) {
    let mut group = c.benchmark_group("rolling_volatility");

    for size in [100, 1000, 10000].iter() {
        let returns: Vec<f64> = (0..*size)
            .map(|i| (i as f64 * 0.01).sin() * 0.02)
            .collect();

        let window = 20;

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            &(&returns, window),
            |b, (r, w)| {
                b.iter(|| simd_rolling_volatility(black_box(r), black_box(*w)));
            },
        );
    }

    group.finish();
}

fn benchmark_covariance_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("covariance_matrix");

    for n_assets in [5, 10, 20].iter() {
        let n_periods = 252; // One year of daily data

        let returns_data: Vec<Vec<f64>> = (0..*n_assets)
            .map(|asset| {
                (0..n_periods)
                    .map(|day| ((day + asset * 7) as f64 * 0.01).sin() * 0.02)
                    .collect()
            })
            .collect();

        let returns_refs: Vec<&[f64]> = returns_data.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements((n_assets * n_assets * n_periods) as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", n_assets),
            &returns_refs,
            |b, r| {
                b.iter(|| simd_covariance_matrix(black_box(r)));
            },
        );
    }

    group.finish();
}

fn benchmark_correlation_matrix(c: &mut Criterion) {
    let mut group = c.benchmark_group("correlation_matrix");

    for n_assets in [5, 10, 20].iter() {
        let n_periods = 252;

        let returns_data: Vec<Vec<f64>> = (0..*n_assets)
            .map(|asset| {
                (0..n_periods)
                    .map(|day| ((day + asset * 7) as f64 * 0.01).sin() * 0.02)
                    .collect()
            })
            .collect();

        let returns_refs: Vec<&[f64]> = returns_data.iter().map(|v| v.as_slice()).collect();

        group.throughput(Throughput::Elements((n_assets * n_assets * n_periods) as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", n_assets),
            &returns_refs,
            |b, r| {
                b.iter(|| simd_correlation_matrix(black_box(r)));
            },
        );
    }

    group.finish();
}

fn benchmark_matrix_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("matrix_multiply");

    for n in [10, 50, 100].iter() {
        let a: Vec<f64> = (0..n * n).map(|i| i as f64).collect();
        let b: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.5).collect();

        group.throughput(Throughput::Elements((n * n * n) as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", n),
            &(&a, &b, *n),
            |bench, (a, b, n)| {
                bench.iter(|| simd_matrix_multiply(black_box(a), black_box(b), black_box(*n)));
            },
        );
    }

    group.finish();
}

fn benchmark_cholesky_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("cholesky_decomposition");

    for n in [5, 10, 20].iter() {
        // Create positive definite matrix
        let mut matrix = vec![0.0; n * n];

        // Fill with correlation-like structure
        for i in 0..*n {
            for j in 0..*n {
                if i == j {
                    matrix[i * n + j] = 1.0;
                } else {
                    let corr = 0.3 * (-(i as f64 - j as f64).abs() / 2.0).exp();
                    matrix[i * n + j] = corr;
                }
            }
        }

        group.throughput(Throughput::Elements((n * n * n / 2) as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", n),
            &(&matrix, *n),
            |b, (m, n)| {
                b.iter(|| simd_cholesky_decomposition(black_box(m), black_box(*n)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_var_calculation,
    benchmark_cvar_calculation,
    benchmark_portfolio_variance,
    benchmark_drawdown_series,
    benchmark_rolling_volatility,
    benchmark_covariance_matrix,
    benchmark_correlation_matrix,
    benchmark_matrix_multiply,
    benchmark_cholesky_decomposition,
);

criterion_main!(benches);
