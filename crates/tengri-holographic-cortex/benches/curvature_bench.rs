//! Benchmarks for Phase 5: Curvature-Adaptive Attention
//!
//! Wolfram-verified performance targets:
//! - Curvature softmax: <10μs for 1000 elements
//! - Hyperbolic attention: <100μs for 64 keys
//! - Multi-head attention: <1ms for 4 heads × 64 keys

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use tengri_holographic_cortex::{
    CurvatureController, HyperbolicAttention, CurvatureAdaptiveAttentionLayer,
    AdaptiveAttentionConfig, FisherRaoMetric, curvature_softmax,
    CURVATURE_DEFAULT, HYPERBOLIC_DIM,
};

fn bench_curvature_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("curvature_softmax");

    for size in [16, 64, 256, 1024].iter() {
        let logits: Vec<f64> = (0..*size).map(|i| (i as f64) * 0.01).collect();

        group.bench_with_input(BenchmarkId::new("n_elements", size), size, |b, _| {
            b.iter(|| {
                curvature_softmax(black_box(&logits), black_box(-0.5))
            })
        });
    }

    group.finish();
}

fn bench_curvature_controller(c: &mut Criterion) {
    let mut group = c.benchmark_group("curvature_controller");

    group.bench_function("update", |b| {
        let mut controller = CurvatureController::new(-0.5);
        b.iter(|| {
            controller.update(black_box(0.7));
            black_box(controller.kappa())
        })
    });

    group.bench_function("conformal_factor", |b| {
        let controller = CurvatureController::new(-0.5);
        b.iter(|| {
            controller.conformal_factor(black_box(0.5))
        })
    });

    group.finish();
}

fn bench_hyperbolic_attention(c: &mut Criterion) {
    let mut group = c.benchmark_group("hyperbolic_attention");

    let attention = HyperbolicAttention::new(HYPERBOLIC_DIM, CURVATURE_DEFAULT);
    let query: Vec<f64> = (0..HYPERBOLIC_DIM).map(|i| (i as f64) * 0.01).collect();

    for num_keys in [8, 32, 64, 128].iter() {
        let keys: Vec<Vec<f64>> = (0..*num_keys)
            .map(|k| (0..HYPERBOLIC_DIM).map(|i| ((k * HYPERBOLIC_DIM + i) as f64) * 0.005).collect())
            .collect();

        group.bench_with_input(BenchmarkId::new("attention_weights", num_keys), num_keys, |b, _| {
            b.iter(|| {
                attention.attention_weights(black_box(&query), black_box(&keys))
            })
        });
    }

    group.finish();
}

fn bench_attention_layer(c: &mut Criterion) {
    let mut group = c.benchmark_group("attention_layer");

    for num_heads in [1, 2, 4, 8].iter() {
        let config = AdaptiveAttentionConfig {
            dim: 8 * num_heads, // Ensure divisible by num_heads
            num_heads: *num_heads,
            initial_kappa: CURVATURE_DEFAULT,
            temperature: 1.0,
            adaptive: true,
            adaptation_rate: 0.1,
        };

        let layer = CurvatureAdaptiveAttentionLayer::new(config.clone());
        let dim = config.dim;

        let query: Vec<f64> = (0..dim).map(|i| (i as f64) * 0.01).collect();
        let keys: Vec<Vec<f64>> = (0..32)
            .map(|k| (0..dim).map(|i| ((k * dim + i) as f64) * 0.005).collect())
            .collect();
        let values = keys.clone();

        group.bench_with_input(BenchmarkId::new("forward_pass", num_heads), num_heads, |b, _| {
            b.iter(|| {
                layer.forward(black_box(&query), black_box(&keys), black_box(&values))
            })
        });
    }

    group.finish();
}

fn bench_fisher_rao(c: &mut Criterion) {
    let mut group = c.benchmark_group("fisher_rao");

    let metric = FisherRaoMetric::new();

    for size in [4, 16, 64, 256].iter() {
        let p: Vec<f64> = (0..*size).map(|_| 1.0 / *size as f64).collect();
        let q: Vec<f64> = (0..*size).map(|i| if i == 0 { 0.5 } else { 0.5 / (*size - 1) as f64 }).collect();

        group.bench_with_input(BenchmarkId::new("distance", size), size, |b, _| {
            b.iter(|| {
                metric.distance(black_box(&p), black_box(&q))
            })
        });
    }

    group.finish();
}

fn bench_info_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("info_density");

    let config = AdaptiveAttentionConfig::default();
    let layer = CurvatureAdaptiveAttentionLayer::new(config);

    for num_distributions in [4, 16, 64].iter() {
        let weights: Vec<Vec<f64>> = (0..*num_distributions)
            .map(|_| {
                let w = vec![0.1, 0.2, 0.3, 0.4];
                let sum: f64 = w.iter().sum();
                w.iter().map(|x| x / sum).collect()
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("compute", num_distributions), num_distributions, |b, _| {
            b.iter(|| {
                layer.compute_info_density(black_box(&weights))
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_curvature_softmax,
    bench_curvature_controller,
    bench_hyperbolic_attention,
    bench_attention_layer,
    bench_fisher_rao,
    bench_info_density,
);

criterion_main!(benches);
