use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use nn_models::isolation_forest::IsolationForest;
use rand::Rng;

fn generate_data(n_samples: usize, n_features: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::thread_rng();
    (0..n_samples)
        .map(|_| {
            (0..n_features)
                .map(|_| rng.gen_range(-1.0..1.0))
                .collect()
        })
        .collect()
}

fn bench_isolation_forest_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolation_forest_training");
    
    for n_samples in [100, 500, 1000, 5000].iter() {
        let data = generate_data(*n_samples, 10);
        
        group.bench_with_input(
            BenchmarkId::from_parameter(n_samples),
            n_samples,
            |b, _| {
                b.iter(|| {
                    let mut forest = IsolationForest::builder()
                        .n_estimators(200)
                        .max_samples(256)
                        .build();
                    forest.fit(&data);
                });
            },
        );
    }
    
    group.finish();
}

fn bench_isolation_forest_inference(c: &mut Criterion) {
    let mut group = c.benchmark_group("isolation_forest_inference");
    
    // Train a model
    let training_data = generate_data(1000, 10);
    let mut forest = IsolationForest::builder()
        .n_estimators(200)
        .max_samples(256)
        .build();
    forest.fit(&training_data);
    
    // Test single sample inference
    let test_sample = vec![0.5; 10];
    
    group.bench_function("single_sample", |b| {
        b.iter(|| {
            black_box(forest.anomaly_score(&test_sample))
        });
    });
    
    // Test batch inference
    for batch_size in [10, 100, 1000].iter() {
        let test_batch = generate_data(*batch_size, 10);
        
        group.bench_with_input(
            BenchmarkId::new("batch", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    black_box(forest.predict(&test_batch))
                });
            },
        );
    }
    
    group.finish();
}

fn bench_isolation_forest_feature_importance(c: &mut Criterion) {
    let training_data = generate_data(1000, 20);
    let mut forest = IsolationForest::builder()
        .n_estimators(200)
        .build();
    forest.fit(&training_data);
    
    c.bench_function("feature_importance", |b| {
        b.iter(|| {
            black_box(forest.feature_importances())
        });
    });
}

criterion_group!(
    benches,
    bench_isolation_forest_training,
    bench_isolation_forest_inference,
    bench_isolation_forest_feature_importance
);
criterion_main!(benches);