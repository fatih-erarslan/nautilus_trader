//! Integration tests for complete training pipeline

use neuro_divergent::{
    models::{basic::MLP, recurrent::LSTM, advanced::NHITS, transformers::TFT},
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
};

#[path = "../helpers/mod.rs"]
mod helpers;
use helpers::{synthetic, model_testing, performance};

#[test]
fn test_end_to_end_mlp_pipeline() {
    let values = synthetic::complex_series(500, 0.1, 24, 1.5);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12)
        .with_learning_rate(0.01)
        .with_epochs(50);

    let mut model = MLP::new(config);

    // Train
    let (_, train_duration) = performance::time_execution(|| {
        model.fit(&data).unwrap()
    });

    println!("MLP training took: {:?}", train_duration);
    assert!(performance::within_time_budget(train_duration, 30000)); // 30 seconds

    // Predict
    let predictions = model.predict(12).unwrap();
    assert_eq!(predictions.len(), 12);
    assert!(model_testing::predictions_finite(&predictions));

    // Verify training improved
    let history = model.training_history();
    assert!(history.last().unwrap() < &history[0]);
}

#[test]
fn test_end_to_end_lstm_pipeline() {
    let values = synthetic::complex_series(800, 0.1, 24, 2.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let config = ModelConfig::default()
        .with_input_size(48)
        .with_horizon(24)
        .with_hidden_size(64)
        .with_learning_rate(0.005)
        .with_epochs(100);

    let mut model = LSTM::new(config);

    let (_, train_duration) = performance::time_execution(|| {
        model.fit(&data).unwrap()
    });

    println!("LSTM training took: {:?}", train_duration);

    let predictions = model.predict(24).unwrap();
    assert!(model_testing::predictions_finite(&predictions));
}

#[test]
fn test_end_to_end_nhits_pipeline() {
    let values = synthetic::complex_series(1000, 0.1, 24, 1.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let config = ModelConfig::default()
        .with_input_size(96)
        .with_horizon(24)
        .with_n_blocks(3)
        .with_learning_rate(0.001)
        .with_epochs(100);

    let mut model = NHITS::new(config);

    let (_, train_duration) = performance::time_execution(|| {
        model.fit(&data).unwrap()
    });

    println!("NHITS training took: {:?}", train_duration);

    let predictions = model.predict(24).unwrap();
    assert!(model_testing::predictions_finite(&predictions));

    let history = model.training_history();
    let improvement = history[0] - history.last().unwrap();
    assert!(improvement > 0.0, "NHITS should improve during training");
}

#[test]
fn test_cross_validation_workflow() {
    let values = synthetic::complex_series(1000, 0.1, 24, 1.5);

    let config = ModelConfig::default()
        .with_input_size(48)
        .with_horizon(24)
        .with_learning_rate(0.01);

    // Simulate 5-fold time series cross-validation
    let fold_size = 800;
    let test_size = 24;
    let n_folds = 3;

    let mut errors = Vec::new();

    for fold in 0..n_folds {
        let train_end = fold_size + fold * test_size;
        let test_end = train_end + test_size;

        if test_end > values.len() {
            break;
        }

        let train_data = TimeSeriesDataFrame::from_values(
            values[..train_end].to_vec(),
            None,
        ).unwrap();

        let test_actual = &values[train_end..test_end];

        let mut model = MLP::new(config.clone());
        model.fit(&train_data).unwrap();

        let predictions = model.predict(test_size).unwrap();
        let mape = model_testing::mape(&predictions, test_actual);

        errors.push(mape);
    }

    let mean_error = errors.iter().sum::<f64>() / errors.len() as f64;
    println!("Cross-validation MAPE: {:.4}", mean_error);

    assert!(
        mean_error < 1.0,
        "Cross-validation error too high: {}",
        mean_error
    );
}

#[test]
fn test_model_ensemble_pipeline() {
    let values = synthetic::complex_series(800, 0.1, 24, 1.5);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let config = ModelConfig::default()
        .with_input_size(48)
        .with_horizon(12)
        .with_learning_rate(0.01)
        .with_epochs(50);

    // Train multiple models
    let mut mlp = MLP::new(config.clone());
    mlp.fit(&data).unwrap();

    let mut lstm = LSTM::new(config.clone().with_hidden_size(32));
    lstm.fit(&data).unwrap();

    // Get predictions
    let mlp_pred = mlp.predict(12).unwrap();
    let lstm_pred = lstm.predict(12).unwrap();

    // Simple ensemble average
    let ensemble_pred: Vec<f64> = mlp_pred
        .iter()
        .zip(lstm_pred.iter())
        .map(|(&p1, &p2)| (p1 + p2) / 2.0)
        .collect();

    assert_eq!(ensemble_pred.len(), 12);
    assert!(model_testing::predictions_finite(&ensemble_pred));

    println!("Ensemble prediction created successfully");
}

#[test]
fn test_incremental_training_workflow() {
    let initial_data = synthetic::complex_series(500, 0.1, 24, 1.0);
    let new_data = synthetic::complex_series(100, 0.1, 24, 1.0);

    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12)
        .with_learning_rate(0.01)
        .with_epochs(30);

    let mut model = MLP::new(config);

    // Initial training
    let data1 = TimeSeriesDataFrame::from_values(initial_data.clone(), None).unwrap();
    model.fit(&data1).unwrap();
    let pred1 = model.predict(12).unwrap();

    // Incremental update with new data
    let mut combined = initial_data;
    combined.extend(new_data);
    let data2 = TimeSeriesDataFrame::from_values(combined, None).unwrap();

    model.fit(&data2).unwrap();
    let pred2 = model.predict(12).unwrap();

    // Predictions should change after update
    let diff: f64 = pred1
        .iter()
        .zip(pred2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    assert!(
        diff > 0.0,
        "Incremental training should update predictions"
    );
}

#[test]
fn test_hyperparameter_tuning_workflow() {
    let values = synthetic::complex_series(500, 0.1, 24, 1.5);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let learning_rates = vec![0.001, 0.01, 0.05];
    let hidden_sizes = vec![16, 32, 64];

    let mut best_loss = f64::INFINITY;
    let mut best_config = None;

    for &lr in &learning_rates {
        for &hs in &hidden_sizes {
            let config = ModelConfig::default()
                .with_input_size(24)
                .with_horizon(12)
                .with_learning_rate(lr)
                .with_hidden_size(hs)
                .with_epochs(20);

            let mut model = MLP::new(config.clone());
            model.fit(&data).unwrap();

            let history = model.training_history();
            let final_loss = *history.last().unwrap();

            if final_loss < best_loss {
                best_loss = final_loss;
                best_config = Some((lr, hs));
            }
        }
    }

    println!("Best config: {:?} with loss: {}", best_config, best_loss);
    assert!(best_config.is_some());
}

#[test]
fn test_multi_horizon_forecasting() {
    let values = synthetic::complex_series(800, 0.1, 24, 1.5);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let config = ModelConfig::default()
        .with_input_size(48)
        .with_horizon(12)
        .with_learning_rate(0.01)
        .with_epochs(50);

    let mut model = LSTM::new(config);
    model.fit(&data).unwrap();

    // Test multiple horizons
    for horizon in [6, 12, 24, 48] {
        let predictions = model.predict(horizon).unwrap();
        assert_eq!(predictions.len(), horizon);
        assert!(model_testing::predictions_finite(&predictions));

        // Longer horizons should have higher uncertainty (in practice)
        println!("Horizon {}: predictions generated", horizon);
    }
}

#[test]
fn test_data_preprocessing_pipeline() {
    use neuro_divergent::data::DataPreprocessor;

    // Raw data with outliers and missing values (simulated)
    let mut values = synthetic::complex_series(500, 0.1, 24, 1.5);

    // Add some outliers
    values[100] = values[100] * 10.0;
    values[200] = values[200] * -5.0;

    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    // Apply preprocessing
    let preprocessor = DataPreprocessor::new()
        .with_normalization(true)
        .with_outlier_detection(true);

    let processed = preprocessor.transform(&data).unwrap();

    // Train model on processed data
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let mut model = MLP::new(config);
    let result = model.fit(&processed);

    assert!(result.is_ok(), "Model should train on preprocessed data");
}
