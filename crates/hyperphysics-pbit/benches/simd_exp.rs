use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyperphysics_pbit::simd::{portable, SimdOps};

/// Benchmark scalar baseline (libm exp)
fn bench_scalar_libm(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_scalar_libm");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01 - 5.0).collect();
            let mut result = vec![0.0; size];

            b.iter(|| {
                for i in 0..size {
                    result[i] = black_box(x[i]).exp();
                }
                black_box(&result);
            });
        });
    }
    group.finish();
}

/// Benchmark scalar Remez implementation
fn bench_scalar_remez(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_scalar_remez");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01 - 5.0).collect();
            let mut result = vec![0.0; size];

            b.iter(|| {
                portable::exp_vec(black_box(&x), black_box(&mut result));
            });
        });
    }
    group.finish();
}

/// Benchmark SIMD vectorized implementation (auto-selects best available)
fn bench_simd_vectorized(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_simd_vectorized");

    for size in [64, 256, 1024, 4096, 16384].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01 - 5.0).collect();
            let mut result = vec![0.0; size];

            b.iter(|| {
                SimdOps::exp(black_box(&x), black_box(&mut result));
            });
        });
    }
    group.finish();
}

/// Benchmark with different input ranges to test edge case performance
fn bench_simd_ranges(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_simd_ranges");
    let size = 1024;
    group.throughput(Throughput::Elements(size as u64));

    // Small values near zero
    group.bench_function("near_zero", |b| {
        let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001 - 0.5).collect();
        let mut result = vec![0.0; size];
        b.iter(|| SimdOps::exp(black_box(&x), black_box(&mut result)));
    });

    // Moderate values
    group.bench_function("moderate", |b| {
        let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.02 - 10.0).collect();
        let mut result = vec![0.0; size];
        b.iter(|| SimdOps::exp(black_box(&x), black_box(&mut result)));
    });

    // Large negative values (underflow region)
    group.bench_function("large_negative", |b| {
        let x: Vec<f64> = (0..size).map(|i| -700.0 + (i as f64) * 0.1).collect();
        let mut result = vec![0.0; size];
        b.iter(|| SimdOps::exp(black_box(&x), black_box(&mut result)));
    });

    // Large positive values (overflow region)
    group.bench_function("large_positive", |b| {
        let x: Vec<f64> = (0..size).map(|i| 690.0 + (i as f64) * 0.01).collect();
        let mut result = vec![0.0; size];
        b.iter(|| SimdOps::exp(black_box(&x), black_box(&mut result)));
    });

    group.finish();
}

/// Benchmark cache effects with aligned vs unaligned data
fn bench_simd_alignment(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_simd_alignment");
    let size = 1024;
    group.throughput(Throughput::Elements(size as u64));

    // Aligned data (naturally aligned from Vec)
    group.bench_function("aligned", |b| {
        let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01 - 5.0).collect();
        let mut result = vec![0.0; size];
        b.iter(|| SimdOps::exp(black_box(&x), black_box(&mut result)));
    });

    // Unaligned data (offset by 1 element)
    group.bench_function("unaligned_offset1", |b| {
        let x_full: Vec<f64> = (0..size + 1).map(|i| (i as f64) * 0.01 - 5.0).collect();
        let mut result_full = vec![0.0; size + 1];
        b.iter(|| {
            let x = &x_full[1..];
            let result = &mut result_full[1..];
            SimdOps::exp(black_box(x), black_box(result));
        });
    });

    group.finish();
}

/// Benchmark throughput comparison (speedup calculation)
fn bench_simd_speedup(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_speedup_comparison");
    let size = 4096;
    group.throughput(Throughput::Elements(size as u64));

    let x: Vec<f64> = (0..size).map(|i| (i as f64) * 0.01 - 5.0).collect();

    // Scalar baseline
    group.bench_function("scalar_baseline", |b| {
        let mut result = vec![0.0; size];
        b.iter(|| {
            for i in 0..size {
                result[i] = black_box(x[i]).exp();
            }
            black_box(&result);
        });
    });

    // Scalar Remez
    group.bench_function("scalar_remez", |b| {
        let mut result = vec![0.0; size];
        b.iter(|| {
            portable::exp_vec(black_box(&x), black_box(&mut result));
        });
    });

    // SIMD vectorized
    group.bench_function("simd_vectorized", |b| {
        let mut result = vec![0.0; size];
        b.iter(|| {
            SimdOps::exp(black_box(&x), black_box(&mut result));
        });
    });

    group.finish();
}

/// Benchmark typical pBit workload: Boltzmann factors
fn bench_boltzmann_factors(c: &mut Criterion) {
    let mut group = c.benchmark_group("boltzmann_factors");

    for size in [256, 1024, 4096].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            // Typical energy differences in Boltzmann factors: exp(-ΔE/kT)
            // For room temperature, kT ≈ 0.026 eV, so ΔE/kT ranges from -20 to 20
            let energy_diff: Vec<f64> = (0..size)
                .map(|i| -10.0 + 20.0 * (i as f64 / size as f64))
                .collect();
            let mut probabilities = vec![0.0; size];

            b.iter(|| {
                SimdOps::exp(black_box(&energy_diff), black_box(&mut probabilities));
                // Normalize probabilities (typical in Metropolis algorithm)
                let sum: f64 = probabilities.iter().sum();
                for p in probabilities.iter_mut() {
                    *p /= sum;
                }
                black_box(&probabilities);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_scalar_libm,
    bench_scalar_remez,
    bench_simd_vectorized,
    bench_simd_ranges,
    bench_simd_alignment,
    bench_simd_speedup,
    bench_boltzmann_factors,
);
criterion_main!(benches);
