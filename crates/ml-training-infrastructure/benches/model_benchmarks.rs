use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ml_training_infrastructure::trainer::ModelTrainer;
use ml_training_infrastructure::optimizer::TrainingOptimizer;
use ml_training_infrastructure::config::TrainingConfig;

fn model_training_benchmark(c: &mut Criterion) {
    let config = TrainingConfig::default();
    let trainer = ModelTrainer::new(config);
    let training_data = vec![
        (vec![1.0, 2.0, 3.0], 1.0),
        (vec![2.0, 3.0, 4.0], 2.0),
        (vec![3.0, 4.0, 5.0], 3.0),
    ];
    
    c.bench_function("model_training", |b| {
        b.iter(|| {
            trainer.train_model(black_box(&training_data))
        })
    });
}

fn training_optimization_benchmark(c: &mut Criterion) {
    let optimizer = TrainingOptimizer::new();
    let hyperparameters = vec![0.01, 0.001, 0.0001];
    let validation_data = vec![
        (vec![1.0, 2.0], 1.0),
        (vec![2.0, 3.0], 2.0),
    ];
    
    c.bench_function("training_optimization", |b| {
        b.iter(|| {
            optimizer.optimize_hyperparameters(
                black_box(&hyperparameters),
                black_box(&validation_data)
            )
        })
    });
}

criterion_group!(benches, model_training_benchmark, training_optimization_benchmark);
criterion_main!(benches);