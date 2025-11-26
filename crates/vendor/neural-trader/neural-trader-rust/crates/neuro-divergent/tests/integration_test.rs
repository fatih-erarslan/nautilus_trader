//! Integration tests for neuro-divergent

use neuro_divergent::{
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
    models::{basic::MLP, register_all_models},
    ModelFactory,
};

#[test]
fn test_create_time_series() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let data = TimeSeriesDataFrame::from_values(values.clone(), None).unwrap();

    assert_eq!(data.len(), 5);
    assert_eq!(data.n_features(), 1);
}

#[test]
fn test_mlp_basic_workflow() {
    let config = ModelConfig::default()
        .with_input_size(10)
        .with_horizon(5)
        .with_hidden_size(64);

    let mut model = MLP::new(config);
    assert_eq!(model.name(), "MLP");

    let values: Vec<f64> = (0..100).map(|i| (i as f64).sin()).collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    // Fit model
    assert!(model.fit(&data).is_ok());

    // Predict
    let predictions = model.predict(5).unwrap();
    assert_eq!(predictions.len(), 5);

    // Prediction intervals
    let intervals = model.predict_intervals(5, &[0.95]).unwrap();
    assert_eq!(intervals.point_forecast.len(), 5);
    assert_eq!(intervals.levels.len(), 1);
}

#[test]
fn test_model_registry() {
    register_all_models().unwrap();

    let models = ModelFactory::list_models().unwrap();
    assert!(!models.is_empty());

    // Should contain all basic models
    assert!(models.contains(&"mlp".to_string()));
    assert!(models.contains(&"dlinear".to_string()));
    assert!(models.contains(&"nlinear".to_string()));

    // Should contain recurrent models
    assert!(models.contains(&"lstm".to_string()));
    assert!(models.contains(&"gru".to_string()));

    // Should contain advanced models
    assert!(models.contains(&"nhits".to_string()));
    assert!(models.contains(&"nbeats".to_string()));
}

#[test]
fn test_create_model_from_registry() {
    register_all_models().unwrap();

    let config = ModelConfig::default();
    let model = ModelFactory::create("mlp", &config).unwrap();

    assert_eq!(model.name(), "MLP");
}

#[test]
fn test_train_test_split() {
    let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let (train, test) = data.train_test_split(0.8).unwrap();

    assert_eq!(train.len(), 80);
    assert_eq!(test.len(), 20);
}

#[test]
fn test_data_preprocessing() {
    use neuro_divergent::DataPreprocessor;

    let values = vec![1.0, f64::NAN, 3.0, 4.0, f64::NAN, 6.0];
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let preprocessor = DataPreprocessor::new();
    let processed = preprocessor.transform(&data).unwrap();

    // All NaN should be filled
    assert!(processed.values.iter().all(|&x| !x.is_nan()));
}

#[test]
fn test_scaler() {
    use neuro_divergent::data::scaler::{StandardScaler, Scaler};
    use ndarray::arr2;

    let data = arr2(&[[1.0], [2.0], [3.0], [4.0], [5.0]]);
    let mut scaler = StandardScaler::new();

    let scaled = scaler.fit_transform(&data).unwrap();
    let unscaled = scaler.inverse_transform(&scaled).unwrap();

    // Should recover original data
    for i in 0..data.nrows() {
        approx::assert_abs_diff_eq!(
            data[[i, 0]],
            unscaled[[i, 0]],
            epsilon = 1e-10
        );
    }
}

#[test]
fn test_model_serialization() {
    use tempfile::NamedTempFile;

    let config = ModelConfig::default();
    let mut model = MLP::new(config);

    let values: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();
    model.fit(&data).unwrap();

    // Save model
    let temp_file = NamedTempFile::new().unwrap();
    model.save(temp_file.path()).unwrap();

    // Load model
    let loaded_model = MLP::load(temp_file.path()).unwrap();
    assert_eq!(loaded_model.name(), model.name());
}

#[test]
fn test_all_models_basic() {
    register_all_models().unwrap();

    let config = ModelConfig::default()
        .with_input_size(10)
        .with_horizon(5);

    let values: Vec<f64> = (0..50).map(|i| (i as f64).sin()).collect();
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let models_to_test = vec![
        "mlp", "dlinear", "nlinear", "lstm", "gru", "rnn",
        "nbeats", "nhits", "tide", "tft", "tcn", "deepar",
    ];

    for model_name in models_to_test {
        let mut model = ModelFactory::create(model_name, &config).unwrap();
        assert!(model.fit(&data).is_ok(), "Failed to fit {}", model_name);

        let predictions = model.predict(5).unwrap();
        assert_eq!(predictions.len(), 5, "Wrong prediction length for {}", model_name);
    }
}

#[test]
fn test_config_validation() {
    let valid_config = ModelConfig::default();
    assert!(valid_config.validate().is_ok());

    let invalid_config = ModelConfig::default().with_input_size(0);
    assert!(invalid_config.validate().is_err());
}

#[test]
fn test_early_stopping() {
    use neuro_divergent::training::engine::EarlyStopping;

    let mut es = EarlyStopping::new(3, 0.01);

    assert!(!es.should_stop(1.0));
    assert!(!es.should_stop(0.9));
    assert!(!es.should_stop(0.89));
    assert!(!es.should_stop(0.891));
    assert!(!es.should_stop(0.892));
    assert!(es.should_stop(0.893));
}

#[test]
fn test_training_metrics() {
    use neuro_divergent::training::metrics::*;
    use ndarray::arr1;

    let pred = arr1(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    let target = arr1(&[1.1, 2.1, 3.1, 4.1, 5.1]);

    let mae_val = mae(&pred, &target);
    approx::assert_abs_diff_eq!(mae_val, 0.1, epsilon = 1e-10);

    let mse_val = mse(&pred, &target);
    approx::assert_abs_diff_eq!(mse_val, 0.01, epsilon = 1e-10);

    let rmse_val = rmse(&pred, &target);
    approx::assert_abs_diff_eq!(rmse_val, 0.1, epsilon = 1e-10);
}

#[test]
fn test_prediction_intervals() {
    use neuro_divergent::inference::PredictionIntervals;

    let forecast = vec![1.0, 2.0, 3.0];
    let std_dev = vec![0.1, 0.1, 0.1];
    let levels = vec![0.95, 0.80];

    let intervals = PredictionIntervals::from_std(forecast.clone(), std_dev, levels);

    assert_eq!(intervals.point_forecast, forecast);
    assert_eq!(intervals.lower_bounds.len(), 2);
    assert_eq!(intervals.upper_bounds.len(), 2);
    assert_eq!(intervals.levels, vec![0.95, 0.80]);
}
