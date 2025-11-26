//! Comprehensive tests for advanced forecasting models
//!
//! Tests NBEATS, NBEATSx, NHITS, and TiDE implementations

use neuro_divergent::{
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
    models::advanced::{NBEATS, NBEATSx, NHITS, TiDE},
};
use neuro_divergent::models::advanced::nbeats::StackType;
use neuro_divergent::models::advanced::nbeatsx::{ExogVariable, ExogType};
use neuro_divergent::models::advanced::nhits::InterpolationMethod;

#[test]
fn test_nbeats_training_and_prediction() {
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let mut model = NBEATS::new(config)
        .with_stacks(vec![
            StackType::Trend { degree: 2 },
            StackType::Seasonal { harmonics: 2 },
        ]);

    // Create synthetic data
    let data = create_synthetic_data(100, 1);

    // Train model
    assert!(model.fit(&data).is_ok());
    assert_eq!(model.name(), "NBEATS");

    // Make predictions
    let predictions = model.predict(12).unwrap();
    assert_eq!(predictions.len(), 12);

    // Test decomposition
    let decomp = model.decompose().unwrap();
    assert_eq!(decomp.trend.len(), 12);
    assert_eq!(decomp.seasonal.len(), 12);
}

#[test]
fn test_nbeatsx_with_exogenous_variables() {
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let mut model = NBEATSx::new(config)
        .with_exog_vars(vec![
            ExogVariable::new("volume", ExogType::Historical),
            ExogVariable::new("market_index", ExogType::Future),
        ])
        .with_static_vars(2);

    let data = create_synthetic_data(100, 1);

    assert!(model.fit(&data).is_ok());
    assert_eq!(model.name(), "NBEATSx");

    let predictions = model.predict(12).unwrap();
    assert_eq!(predictions.len(), 12);

    // Test feature importance
    let importance = model.feature_importance().unwrap();
    assert_eq!(importance.len(), 2);
}

#[test]
fn test_nhits_long_horizon_forecasting() {
    let config = ModelConfig::default()
        .with_input_size(168)  // 1 week hourly
        .with_horizon(720);     // 1 month hourly

    let mut model = NHITS::new(config)
        .with_pooling_sizes(vec![1, 2, 4, 8, 16])
        .with_mlp_units(vec![512, 512])
        .with_interpolation(InterpolationMethod::Linear);

    let data = create_synthetic_data(500, 1);

    assert!(model.fit(&data).is_ok());
    assert_eq!(model.name(), "NHITS");

    // Test long-horizon prediction
    let predictions = model.predict(720).unwrap();
    assert_eq!(predictions.len(), 720);

    // Verify predictions are reasonable
    assert!(predictions.iter().all(|&x| x.is_finite()));
}

#[test]
fn test_tide_residual_connections() {
    let config = ModelConfig::default()
        .with_input_size(48)
        .with_horizon(24);

    let mut model = TiDE::new(config)
        .with_residual_weight(0.5)
        .with_encoder_sizes(vec![256, 128])
        .with_decoder_sizes(vec![64]);

    let data = create_synthetic_data(150, 1);

    assert!(model.fit(&data).is_ok());
    assert_eq!(model.name(), "TiDE");

    let predictions = model.predict(24).unwrap();
    assert_eq!(predictions.len(), 24);
}

#[test]
fn test_prediction_intervals_all_models() {
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);
    let data = create_synthetic_data(100, 1);

    // Test NBEATS
    let mut nbeats = NBEATS::new(config.clone());
    nbeats.fit(&data).unwrap();
    let intervals = nbeats.predict_intervals(12, &[0.8, 0.95]).unwrap();
    assert!(intervals.median.len() == 12);

    // Test NBEATSx
    let mut nbeatsx = NBEATSx::new(config.clone());
    nbeatsx.fit(&data).unwrap();
    let intervals = nbeatsx.predict_intervals(12, &[0.8, 0.95]).unwrap();
    assert!(intervals.median.len() == 12);

    // Test NHITS
    let mut nhits = NHITS::new(config.clone());
    nhits.fit(&data).unwrap();
    let intervals = nhits.predict_intervals(12, &[0.8, 0.95]).unwrap();
    assert!(intervals.median.len() == 12);

    // Test TiDE
    let mut tide = TiDE::new(config);
    tide.fit(&data).unwrap();
    let intervals = tide.predict_intervals(12, &[0.8, 0.95]).unwrap();
    assert!(intervals.median.len() == 12);
}

#[test]
fn test_model_persistence() {
    use std::path::PathBuf;
    use tempfile::tempdir;

    let config = ModelConfig::default();
    let data = create_synthetic_data(100, 1);

    let dir = tempdir().unwrap();

    // Test NBEATS save/load
    let mut nbeats = NBEATS::new(config.clone());
    nbeats.fit(&data).unwrap();
    let nbeats_path = dir.path().join("nbeats.bin");
    nbeats.save(&nbeats_path).unwrap();
    let loaded_nbeats = NBEATS::load(&nbeats_path).unwrap();
    assert_eq!(loaded_nbeats.name(), "NBEATS");

    // Test NHITS save/load
    let mut nhits = NHITS::new(config.clone());
    nhits.fit(&data).unwrap();
    let nhits_path = dir.path().join("nhits.bin");
    nhits.save(&nhits_path).unwrap();
    let loaded_nhits = NHITS::load(&nhits_path).unwrap();
    assert_eq!(loaded_nhits.name(), "NHITS");
}

#[test]
fn test_horizon_degradation() {
    let config = ModelConfig::default()
        .with_input_size(100)
        .with_horizon(360);

    let data = create_synthetic_data(300, 1);

    let mut nhits = NHITS::new(config)
        .with_pooling_sizes(vec![1, 4, 16]);
    nhits.fit(&data).unwrap();

    // Test multiple horizons
    for horizon in [24, 96, 180, 360] {
        let predictions = nhits.predict(horizon).unwrap();
        assert_eq!(predictions.len(), horizon);

        // Verify all predictions are finite
        assert!(predictions.iter().all(|&x| x.is_finite()));
    }
}

// Helper function to create synthetic time series data
fn create_synthetic_data(length: usize, num_features: usize) -> TimeSeriesDataFrame {
    use ndarray::Array2;

    let mut data = Vec::new();
    for i in 0..length {
        let t = i as f64;
        // Create trend + seasonality + noise
        let trend = 0.1 * t;
        let seasonality = 10.0 * (2.0 * std::f64::consts::PI * t / 24.0).sin();
        let noise = (rand::random::<f64>() - 0.5) * 2.0;
        data.push(trend + seasonality + noise);
    }

    let values = Array2::from_shape_vec((length, num_features), data.clone()).unwrap();
    TimeSeriesDataFrame::new(values, None, vec!["value".to_string()]).unwrap()
}
