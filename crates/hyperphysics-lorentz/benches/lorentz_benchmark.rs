//! Performance benchmarks for hyperphysics-lorentz
//!
//! Validates sub-100μs latency targets for hyperbolic operations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use hyperphysics_lorentz::{
    LorentzModel, SimdMinkowski,
    poincare_to_lorentz, lorentz_to_poincare,
    batch_poincare_to_lorentz, batch_lorentz_to_poincare,
};
use std::time::Duration;

const CURVATURE: f64 = -1.0;

fn bench_minkowski_dot(c: &mut Criterion) {
    let mut group = c.benchmark_group("minkowski_dot");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    for size in [4, 8, 16, 32, 64, 128, 256, 512, 1024].iter() {
        let x: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = (0..*size).map(|i| ((size - i) as f64) * 0.01).collect();

        group.throughput(Throughput::Elements(*size as u64));
        group.bench_with_input(
            BenchmarkId::new("simd", size),
            size,
            |b, _| {
                b.iter(|| {
                    SimdMinkowski::dot(black_box(&x), black_box(&y)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("lorentz_distance");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let model = LorentzModel::default_curvature();

    for dim in [2, 4, 8, 16, 32, 64].iter() {
        let spatial_a: Vec<f64> = (0..*dim).map(|i| (i as f64) * 0.05).collect();
        let spatial_b: Vec<f64> = (0..*dim).map(|i| ((dim - i) as f64) * 0.05).collect();

        let a = model.from_spatial(&spatial_a).unwrap();
        let b = model.from_spatial(&spatial_b).unwrap();

        group.bench_with_input(
            BenchmarkId::new("distance", dim),
            dim,
            |bench, _| {
                bench.iter(|| {
                    a.distance(black_box(&b)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("poincare_lorentz_conversion");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    for dim in [2, 4, 8, 16, 32, 64, 128].iter() {
        let poincare: Vec<f64> = (0..*dim).map(|i| (i as f64) * 0.01).collect();

        group.bench_with_input(
            BenchmarkId::new("poincare_to_lorentz", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    poincare_to_lorentz(black_box(&poincare), CURVATURE).unwrap()
                })
            },
        );

        let lorentz = poincare_to_lorentz(&poincare, CURVATURE).unwrap();

        group.bench_with_input(
            BenchmarkId::new("lorentz_to_poincare", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    lorentz_to_poincare(black_box(&lorentz), CURVATURE).unwrap()
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("roundtrip", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    let l = poincare_to_lorentz(black_box(&poincare), CURVATURE).unwrap();
                    lorentz_to_poincare(black_box(&l), CURVATURE).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_batch_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_conversion");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(500);

    for batch_size in [10, 50, 100, 500, 1000].iter() {
        let dim = 64;
        let points: Vec<Vec<f64>> = (0..*batch_size)
            .map(|j| (0..dim).map(|i| ((i + j) as f64) * 0.005).collect())
            .collect();

        group.throughput(Throughput::Elements(*batch_size as u64));

        group.bench_with_input(
            BenchmarkId::new("batch_poincare_to_lorentz", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    batch_poincare_to_lorentz(black_box(&points), CURVATURE).unwrap()
                })
            },
        );

        let lorentz_points: Vec<Vec<f64>> = points
            .iter()
            .map(|p| poincare_to_lorentz(p, CURVATURE).unwrap())
            .collect();

        group.bench_with_input(
            BenchmarkId::new("batch_lorentz_to_poincare", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    batch_lorentz_to_poincare(black_box(&lorentz_points), CURVATURE).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_exp_log_maps(c: &mut Criterion) {
    let mut group = c.benchmark_group("exp_log_maps");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let model = LorentzModel::default_curvature();

    for dim in [2, 4, 8, 16, 32].iter() {
        let spatial: Vec<f64> = (0..*dim).map(|i| (i as f64) * 0.05).collect();
        let base = model.from_spatial(&spatial).unwrap();

        // Create tangent vector (perpendicular to position in Minkowski sense)
        let mut tangent = vec![0.0; dim + 1];
        for i in 1..=*dim {
            tangent[i] = 0.1 / (*dim as f64).sqrt();
        }

        group.bench_with_input(
            BenchmarkId::new("exp_map", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    base.exp(black_box(&tangent)).unwrap()
                })
            },
        );

        let target_spatial: Vec<f64> = (0..*dim).map(|i| (i as f64) * 0.08).collect();
        let target = model.from_spatial(&target_spatial).unwrap();

        group.bench_with_input(
            BenchmarkId::new("log_map", dim),
            dim,
            |b, _| {
                b.iter(|| {
                    base.log(black_box(&target)).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_geodesic(c: &mut Criterion) {
    let mut group = c.benchmark_group("geodesic");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(1000);

    let model = LorentzModel::default_curvature();
    let a = model.from_spatial(&[0.2, 0.3]).unwrap();
    let b = model.from_spatial(&[0.5, 0.1]).unwrap();

    group.bench_function("geodesic_single", |bench| {
        bench.iter(|| {
            model.geodesic(black_box(&a), black_box(&b), 0.5).unwrap()
        })
    });

    group.bench_function("geodesic_10_points", |bench| {
        bench.iter(|| {
            for i in 0..10 {
                let t = (i as f64) / 9.0;
                let _ = model.geodesic(black_box(&a), black_box(&b), t).unwrap();
            }
        })
    });

    group.finish();
}

fn bench_frechet_mean(c: &mut Criterion) {
    let mut group = c.benchmark_group("frechet_mean");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(200);

    let model = LorentzModel::default_curvature();

    for n_points in [5, 10, 20, 50, 100].iter() {
        let points: Vec<_> = (0..*n_points)
            .map(|i| {
                let x = (i as f64) * 0.05 - 0.25;
                let y = ((i * 7) as f64 % 10.0) * 0.05 - 0.25;
                model.from_spatial(&[x, y]).unwrap()
            })
            .collect();

        group.throughput(Throughput::Elements(*n_points as u64));

        group.bench_with_input(
            BenchmarkId::new("frechet_mean", n_points),
            n_points,
            |b, _| {
                b.iter(|| {
                    model.frechet_mean(black_box(&points), None, 50, 1e-8).unwrap()
                })
            },
        );
    }

    group.finish();
}

fn bench_latency_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("latency_validation");
    group.measurement_time(Duration::from_secs(30));
    group.sample_size(5000);

    let model = LorentzModel::default_curvature();
    let a = model.from_spatial(&[0.3, 0.4]).unwrap();
    let b = model.from_spatial(&[0.5, 0.1]).unwrap();
    let poincare = vec![0.3, 0.4];

    // Target: <5μs for distance
    group.bench_function("distance_5us_target", |bench| {
        bench.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = a.distance(black_box(&b)).unwrap();
            }
            start.elapsed()
        })
    });

    // Target: <2μs for conversion
    group.bench_function("conversion_2us_target", |bench| {
        bench.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = poincare_to_lorentz(black_box(&poincare), CURVATURE).unwrap();
            }
            start.elapsed()
        })
    });

    // Target: <10μs for geodesic
    group.bench_function("geodesic_10us_target", |bench| {
        bench.iter_custom(|iters| {
            let start = std::time::Instant::now();
            for _ in 0..iters {
                let _ = model.geodesic(black_box(&a), black_box(&b), 0.5).unwrap();
            }
            start.elapsed()
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_minkowski_dot,
    bench_distance,
    bench_conversion,
    bench_batch_conversion,
    bench_exp_log_maps,
    bench_geodesic,
    bench_frechet_mean,
    bench_latency_validation,
);

criterion_main!(benches);
