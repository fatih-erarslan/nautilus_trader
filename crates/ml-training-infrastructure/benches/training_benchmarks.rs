//! Benchmarks for ML training infrastructure

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use ml_training_infrastructure::{
    models::{create_model, ModelType},
    data::{TrainingData, DataLoader},
    TrainingConfig,
};
use ndarray::Array3;
use std::sync::Arc;

fn benchmark_model_creation(c: &mut Criterion) {
    let config = TrainingConfig::default();
    
    let mut group = c.benchmark_group("model_creation");
    
    for model_type in &[
        ModelType::Transformer,
        ModelType::LSTM,
        ModelType::XGBoost,
        ModelType::LightGBM,
        ModelType::NeuralNetwork,
    ] {
        group.bench_with_input(
            BenchmarkId::new("create", format!("{:?}", model_type)),
            model_type,
            |b, &model_type| {
                b.iter(|| {
                    let _model = create_model(black_box(model_type), black_box(&config));
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_data_loading(c: &mut Criterion) {
    let config = Arc::new(TrainingConfig::default().data);
    let loader = DataLoader::new(config);
    
    let mut group = c.benchmark_group("data_loading");
    
    // Create synthetic data
    let n_samples = 10000;
    let seq_len = 100;
    let n_features = 10;
    let data = Array3::<f32>::ones((n_samples, seq_len, n_features));
    
    group.bench_function("normalization", |b| {
        b.iter(|| {
            let _normalized = loader.calculate_normalization_params(black_box(&data));
        });
    });
    
    group.bench_function("sequence_creation", |b| {
        let data_2d = data.mean_axis(ndarray::Axis(0)).unwrap();
        b.iter(|| {
            let _sequences = loader.create_sequences(black_box(&data_2d));
        });
    });
    
    group.finish();
}

fn benchmark_model_prediction(c: &mut Criterion) {
    let config = TrainingConfig::default();
    let runtime = tokio::runtime::Runtime::new().unwrap();
    
    let mut group = c.benchmark_group("model_prediction");
    
    // Create test input
    let input = Array3::<f32>::ones((32, 100, 10)); // batch_size=32, seq_len=100, features=10
    
    for model_type in &[ModelType::XGBoost, ModelType::LightGBM] {
        let model = create_model(*model_type, &config).unwrap();
        
        group.bench_with_input(
            BenchmarkId::new("predict", format!("{:?}", model_type)),
            &input,
            |b, input| {
                b.to_async(&runtime).iter(|| async {
                    let _predictions = model.predict(black_box(input)).await;
                });
            },
        );
    }
    
    group.finish();
}

fn benchmark_cross_validation(c: &mut Criterion) {
    use ml_training_infrastructure::validation::{CrossValidator, CVStrategy};
    use ml_training_infrastructure::config::ValidationConfig;
    
    let mut group = c.benchmark_group("cross_validation");
    
    let config = ValidationConfig {
        cv_strategy: CVStrategy::TimeSeriesSplit,
        n_folds: 5,
        gap: 10,
        metrics: vec![],
        walk_forward: false,
        purged: true,
    };
    
    let validator = CrossValidator::new(config);
    
    // Create synthetic training data
    let data = TrainingData {
        x_train: Array3::<f32>::ones((1000, 100, 10)),
        y_train: Array3::<f32>::ones((1000, 10, 10)),
        x_val: Array3::<f32>::ones((200, 100, 10)),
        y_val: Array3::<f32>::ones((200, 10, 10)),
        x_test: Array3::<f32>::ones((200, 100, 10)),
        y_test: Array3::<f32>::ones((200, 10, 10)),
        feature_names: vec!["f1".to_string(); 10],
        timestamps: vec![],
        assets: vec![],
        normalization: Default::default(),
    };
    
    group.bench_function("combine_arrays", |b| {
        let arrays = vec![&data.x_train, &data.x_val, &data.x_test];
        b.iter(|| {
            let _combined = validator.combine_arrays(black_box(&arrays));
        });
    });
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_model_creation,
    benchmark_data_loading,
    benchmark_model_prediction,
    benchmark_cross_validation
);
criterion_main!(benches);