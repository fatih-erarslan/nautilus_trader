//! Machine Learning benchmarks for CDFA
//!
//! Validates ML integration performance targets

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use cdfa_ml::prelude::*;
use std::time::Duration;

/// Generate ML training dataset
fn generate_ml_dataset(num_samples: usize, num_features: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let features: Vec<Vec<f64>> = (0..num_samples)
        .map(|i| {
            (0..num_features)
                .map(|j| ((i + j) as f64 * 0.1).sin() + ((i * j) as f64 * 0.01).cos())
                .collect()
        })
        .collect();
    
    let targets: Vec<f64> = (0..num_samples)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    
    (features, targets)
}

/// Benchmark neural network training
fn bench_neural_training(c: &mut Criterion) {
    let mut group = c.benchmark_group("neural_training");
    group.measurement_time(Duration::from_secs(20));
    
    for num_samples in [100, 500, 1000].iter() {
        let (features, targets) = generate_ml_dataset(*num_samples, 10);
        
        group.throughput(Throughput::Elements(*num_samples as u64));
        group.bench_with_input(
            BenchmarkId::new("basic_mlp", num_samples),
            &(features.clone(), targets.clone()),
            |b, (features, targets)| {
                b.iter(|| {
                    let mut network = BasicMLP::new(10, vec![16, 8], 1);
                    black_box(network.train(black_box(features), black_box(targets), 10))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark feature selection
fn bench_feature_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("feature_selection");
    group.measurement_time(Duration::from_secs(15));
    
    for num_features in [20, 50, 100, 200].iter() {
        let (features, targets) = generate_ml_dataset(500, *num_features);
        
        group.throughput(Throughput::Elements(*num_features as u64));
        group.bench_with_input(
            BenchmarkId::new("mutual_info", num_features),
            &(features.clone(), targets.clone()),
            |b, (features, targets)| {
                b.iter(|| {
                    let selector = MutualInfoSelector::new(10);
                    black_box(selector.select_features(black_box(features), black_box(targets)))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("variance_threshold", num_features),
            &(features, targets),
            |b, (features, _targets)| {
                b.iter(|| {
                    let selector = VarianceThreshold::new(0.1);
                    black_box(selector.select_features(black_box(features)))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark online learning
fn bench_online_learning(c: &mut Criterion) {
    let mut group = c.benchmark_group("online_learning");
    group.measurement_time(Duration::from_secs(10));
    
    for stream_size in [1000, 5000, 10000].iter() {
        let (features, targets) = generate_ml_dataset(*stream_size, 5);
        
        group.throughput(Throughput::Elements(*stream_size as u64));
        group.bench_with_input(
            BenchmarkId::new("online_sgd", stream_size),
            &(features.clone(), targets.clone()),
            |b, (features, targets)| {
                b.iter(|| {
                    let mut learner = OnlineSGD::new(5, 0.01);
                    for (feature, &target) in features.iter().zip(targets.iter()) {
                        learner.update(black_box(feature), black_box(target));
                    }
                    black_box(learner.get_weights())
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("adaptive_learning", stream_size),
            &(features, targets),
            |b, (features, targets)| {
                b.iter(|| {
                    let mut learner = AdaptiveLearner::new(5);
                    for (feature, &target) in features.iter().zip(targets.iter()) {
                        learner.update(black_box(feature), black_box(target));
                    }
                    black_box(learner.predict(&features[0]))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark ensemble methods
fn bench_ensemble_methods(c: &mut Criterion) {
    let mut group = c.benchmark_group("ensemble_methods");
    group.measurement_time(Duration::from_secs(25));
    
    for num_estimators in [5, 10, 25, 50].iter() {
        let (features, targets) = generate_ml_dataset(200, 8);
        
        group.throughput(Throughput::Elements(*num_estimators as u64));
        group.bench_with_input(
            BenchmarkId::new("random_forest", num_estimators),
            &(features.clone(), targets.clone()),
            |b, (features, targets)| {
                b.iter(|| {
                    let mut forest = RandomForest::new(*num_estimators, 3);
                    forest.train(black_box(features), black_box(targets));
                    black_box(forest.predict(&features[0]))
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("gradient_boosting", num_estimators),
            &(features, targets),
            |b, (features, targets)| {
                b.iter(|| {
                    let mut boosting = GradientBoosting::new(*num_estimators, 0.1, 3);
                    boosting.train(black_box(features), black_box(targets));
                    black_box(boosting.predict(&features[0]))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark cross-validation
fn bench_cross_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_validation");
    group.measurement_time(Duration::from_secs(30));
    
    for k_folds in [3, 5, 10].iter() {
        let (features, targets) = generate_ml_dataset(500, 10);
        
        group.bench_with_input(
            BenchmarkId::new("k_fold_cv", k_folds),
            k_folds,
            |b, &k_folds| {
                b.iter(|| {
                    let cv = KFoldCV::new(k_folds);
                    let model = || BasicMLP::new(10, vec![8], 1);
                    black_box(cv.validate(model, black_box(&features), black_box(&targets)))
                })
            },
        );
    }
    
    group.finish();
}

/// Performance validation benchmark
fn bench_performance_validation(c: &mut Criterion) {
    let mut group = c.benchmark_group("performance_validation");
    group.measurement_time(Duration::from_secs(30));
    
    group.bench_function("full_ml_benchmark_suite", |b| {
        b.iter(|| {
            let results = run_ml_benchmarks();
            black_box(results)
        })
    });
    
    group.bench_function("ml_performance_target_validation", |b| {
        b.iter(|| {
            let meets_targets = validate_ml_performance_targets();
            black_box(meets_targets)
        })
    });
    
    group.finish();
    
    // Run validation and print results
    let results = run_ml_benchmarks();
    let meets_targets = validate_ml_performance_targets();
    
    println!("\n=== ML CDFA Performance Results ===");
    println!("Neural Training: {} ns/sample", results.neural_training_ns);
    println!("Feature Selection: {} ns/feature", results.feature_selection_ns);
    println!("Online Learning: {} ns/update", results.online_learning_ns);
    println!("Ensemble Methods: {} ns/estimator", results.ensemble_ns);
    println!("Cross Validation: {} ns/fold", results.cross_validation_ns);
    println!("Performance Targets Met: {}", meets_targets);
    
    if meets_targets {
        println!("\n🎉 ALL ML PERFORMANCE TARGETS MET! 🎉");
        println!("ML CDFA implementations are delivering expected performance.");
    } else {
        println!("\n⚠️  Some ML performance targets not met.");
        println!("This may be due to debug build or system limitations.");
    }
}

criterion_group!(
    benches,
    bench_neural_training,
    bench_feature_selection,
    bench_online_learning,
    bench_ensemble_methods,
    bench_cross_validation,
    bench_performance_validation,
);

criterion_main!(benches);