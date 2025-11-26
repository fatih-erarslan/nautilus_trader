//! Benchmarks for mixed precision training
//!
//! Compares FP32 vs FP16 performance:
//! - Training speed
//! - Memory usage
//! - Convergence quality
//! - Stability

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use neuro_divergent::optimizations::mixed_precision::*;
use ndarray::{Array1, Array2, Array};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::Uniform;

/// Benchmark data generation
fn generate_training_data(n_samples: usize, n_features: usize) -> (Array2<f64>, Array1<f64>) {
    let x = Array::random((n_samples, n_features), Uniform::new(-1.0, 1.0));
    let y = Array::random(n_samples, Uniform::new(-1.0, 1.0));
    (x, y)
}

/// Simulate FP32 training step
fn fp32_training_step(input: &Array2<f64>, targets: &Array1<f64>) -> f32 {
    // Simple forward pass
    let predictions: Array1<f32> = input.iter()
        .map(|&x| x as f32)
        .take(targets.len())
        .collect();

    // MSE loss
    predictions.iter().zip(targets.iter())
        .map(|(&pred, &target)| {
            let diff = pred - target as f32;
            diff * diff
        })
        .sum::<f32>() / predictions.len() as f32
}

/// Benchmark FP16 conversion
fn bench_fp16_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("fp16_conversion");

    for size in [100, 1000, 10000].iter() {
        let array = Array2::<f64>::random((*size, 10), Uniform::new(-1.0, 1.0));

        group.bench_with_input(
            BenchmarkId::new("to_fp16", size),
            &array,
            |b, arr| {
                b.iter(|| {
                    conversion::to_fp16(black_box(arr))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark gradient scaling
fn bench_gradient_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("gradient_scaling");

    let config = MixedPrecisionConfig::default();
    let scaler = GradScaler::new(&config);

    for n_params in [10, 100, 1000].iter() {
        let mut gradients: Vec<Array2<f64>> = (0..*n_params)
            .map(|_| Array2::random((10, 10), Uniform::new(-1.0, 1.0)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("unscale", n_params),
            &gradients,
            |b, grads| {
                b.iter(|| {
                    let mut g = grads.clone();
                    scaler.unscale(black_box(&mut g))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark overflow detection
fn bench_overflow_detection(c: &mut Criterion) {
    let mut group = c.benchmark_group("overflow_detection");

    let config = MixedPrecisionConfig::default();
    let scaler = GradScaler::new(&config);

    for n_params in [10, 100, 1000].iter() {
        let gradients: Vec<Array2<f64>> = (0..*n_params)
            .map(|_| Array2::random((10, 10), Uniform::new(-1.0, 1.0)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("check_finite", n_params),
            &gradients,
            |b, grads| {
                b.iter(|| {
                    scaler.check_finite_gradients(black_box(grads))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full training step comparison
fn bench_training_step_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("training_step");

    for batch_size in [32, 64, 128].iter() {
        let (x, y) = generate_training_data(*batch_size, 100);

        // FP32 baseline
        group.bench_with_input(
            BenchmarkId::new("fp32", batch_size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    fp32_training_step(black_box(x), black_box(y))
                });
            },
        );

        // FP16 mixed precision
        let config = MixedPrecisionConfig::default();
        let mut trainer = MixedPrecisionTrainer::new(config);

        group.bench_with_input(
            BenchmarkId::new("fp16_mixed", batch_size),
            &(&x, &y),
            |b, (x, y)| {
                b.iter(|| {
                    let input_fp16 = trainer.forward_fp16(black_box(x));
                    let predictions = Array1::from_vec(
                        input_fp16.iter().take(y.len()).map(|&v| v).collect()
                    );
                    trainer.compute_scaled_loss(black_box(&predictions), black_box(y))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");

    for n_weights in [100, 1000, 10000].iter() {
        let weights: Vec<Array2<f64>> = (0..*n_weights)
            .map(|_| Array2::random((10, 10), Uniform::new(-1.0, 1.0)))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("weight_manager_init", n_weights),
            &weights,
            |b, w| {
                b.iter(|| {
                    WeightManager::new(black_box(w.clone()), true)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark loss scale updates
fn bench_loss_scale_updates(c: &mut Criterion) {
    let config = MixedPrecisionConfig::default();
    let mut scaler = GradScaler::new(&config);

    c.bench_function("scale_update_stable", |b| {
        b.iter(|| {
            scaler.update(black_box(false))
        });
    });

    c.bench_function("scale_update_overflow", |b| {
        b.iter(|| {
            scaler.update(black_box(true))
        });
    });
}

/// Benchmark safety checks
fn bench_safety_checks(c: &mut Criterion) {
    let mut group = c.benchmark_group("safety_checks");

    for size in [100, 1000, 10000].iter() {
        let array = Array2::<f64>::random((*size, 10), Uniform::new(-1.0, 1.0));

        group.bench_with_input(
            BenchmarkId::new("fp16_safety", size),
            &array,
            |b, arr| {
                b.iter(|| {
                    conversion::check_fp16_safe(black_box(arr))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fp16_conversion,
    bench_gradient_scaling,
    bench_overflow_detection,
    bench_training_step_comparison,
    bench_memory_efficiency,
    bench_loss_scale_updates,
    bench_safety_checks,
);
criterion_main!(benches);
