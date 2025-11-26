//! Criterion Benchmarks for Neuro-Divergent Models
//!
//! Compares performance against Python NeuralForecast baseline

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuro_divergent::*;

fn generate_data(samples: usize) -> Vec<f64> {
    (0..samples)
        .map(|i| {
            let t = i as f64;
            10.0 * (t / 100.0).sin() + 0.5 * t
        })
        .collect()
}

fn bench_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("training");

    for &samples in &[100, 500, 1000, 5000] {
        let data = generate_data(samples);

        group.bench_with_input(
            BenchmarkId::new("nhits", samples),
            &data,
            |b, data| {
                b.iter(|| {
                    // TODO: Implement NHITS training
                    // let mut model = NHITSModel::new(config);
                    // model.fit(black_box(data)).unwrap()
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn bench_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("inference");

    let data = generate_data(1000);

    // TODO: Train model once
    // let mut model = NHITSModel::new(config);
    // model.fit(&data).unwrap();

    for &horizon in &[6, 12, 24, 48] {
        group.bench_with_input(
            BenchmarkId::new("nhits_predict", horizon),
            &horizon,
            |b, &horizon| {
                b.iter(|| {
                    // TODO: Implement prediction
                    // model.predict(black_box(horizon)).unwrap()
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn bench_batch_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inference");

    let data = generate_data(1000);

    // TODO: Train model
    // let mut model = NHITSModel::new(config);
    // model.fit(&data).unwrap();

    for &batch_size in &[1, 8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("batch_predict", batch_size),
            &batch_size,
            |b, &batch_size| {
                b.iter(|| {
                    // TODO: Implement batch prediction
                    // let horizons = vec![24; batch_size];
                    // model.predict_batch(black_box(&horizons)).unwrap()
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

fn bench_preprocessing(c: &mut Criterion) {
    let mut group = c.benchmark_group("preprocessing");

    for &samples in &[1000, 10000, 100000] {
        let data = generate_data(samples);

        group.bench_with_input(
            BenchmarkId::new("normalize", samples),
            &data,
            |b, data| {
                b.iter(|| {
                    normalize_data(black_box(data))
                });
            },
        );
    }

    group.finish();
}

fn normalize_data(data: &[f64]) -> Vec<f64> {
    let mean = data.iter().sum::<f64>() / data.len() as f64;
    let std = (data.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / data.len() as f64)
        .sqrt();

    data.iter()
        .map(|&x| (x - mean) / std)
        .collect()
}

fn bench_cross_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_validation");

    let data = generate_data(1000);

    for &n_splits in &[3, 5, 10] {
        group.bench_with_input(
            BenchmarkId::new("time_series_cv", n_splits),
            &n_splits,
            |b, &n_splits| {
                b.iter(|| {
                    // TODO: Implement cross-validation
                    // cross_validate(black_box(&model), black_box(&data), black_box(n_splits))
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_training,
    bench_inference,
    bench_batch_inference,
    bench_preprocessing,
    bench_cross_validation
);
criterion_main!(benches);
