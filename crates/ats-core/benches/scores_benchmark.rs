//! Performance benchmarks for nonconformity scores
//!
//! Target: RAPS scoring <3μs per sample on production hardware

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ats_core::scores::*;
use rand::Rng;

fn generate_softmax(n_classes: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let mut probs: Vec<f32> = (0..n_classes).map(|_| rng.gen::<f32>()).collect();
    let sum: f32 = probs.iter().sum();
    probs.iter_mut().for_each(|p| *p /= sum);
    probs
}

fn bench_raps_single(c: &mut Criterion) {
    let scorer = RapsScorer::default();
    let softmax = generate_softmax(100);

    c.bench_function("raps_single_100classes", |b| {
        b.iter(|| {
            scorer.score(black_box(&softmax), black_box(42), black_box(0.5))
        });
    });
}

fn bench_raps_varying_classes(c: &mut Criterion) {
    let mut group = c.benchmark_group("raps_varying_classes");
    let scorer = RapsScorer::default();

    for n_classes in [10, 50, 100, 500, 1000].iter() {
        let softmax = generate_softmax(*n_classes);

        group.bench_with_input(
            BenchmarkId::from_parameter(n_classes),
            &softmax,
            |b, softmax| {
                b.iter(|| {
                    scorer.score(black_box(softmax), black_box(5), black_box(0.5))
                });
            },
        );
    }

    group.finish();
}

fn bench_all_scorers(c: &mut Criterion) {
    let mut group = c.benchmark_group("all_scorers_100classes");
    let softmax = generate_softmax(100);

    let raps = RapsScorer::default();
    group.bench_function("RAPS", |b| {
        b.iter(|| raps.score(black_box(&softmax), black_box(42), black_box(0.5)));
    });

    let aps = ApsScorer::default();
    group.bench_function("APS", |b| {
        b.iter(|| aps.score(black_box(&softmax), black_box(42), black_box(0.5)));
    });

    let saps = SapsScorer::default();
    group.bench_function("SAPS", |b| {
        b.iter(|| saps.score(black_box(&softmax), black_box(42), black_box(0.5)));
    });

    let thr = ThresholdScorer::default();
    group.bench_function("THR", |b| {
        b.iter(|| thr.score(black_box(&softmax), black_box(42), black_box(0.5)));
    });

    let lac = LacScorer::default();
    group.bench_function("LAC", |b| {
        b.iter(|| lac.score(black_box(&softmax), black_box(42), black_box(0.5)));
    });

    group.finish();
}

fn bench_batch_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_processing");
    let scorer = RapsScorer::default();

    for batch_size in [100, 1000, 10000].iter() {
        let batch: Vec<Vec<f32>> = (0..*batch_size)
            .map(|_| generate_softmax(100))
            .collect();

        let labels: Vec<usize> = (0..*batch_size)
            .map(|i| i % 100)
            .collect();

        let u_values: Vec<f32> = (0..*batch_size)
            .map(|_| rand::thread_rng().gen::<f32>())
            .collect();

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    scorer.score_batch(
                        black_box(&batch),
                        black_box(&labels),
                        black_box(&u_values),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_raps_single,
    bench_raps_varying_classes,
    bench_all_scorers,
    bench_batch_processing
);

criterion_main!(benches);
