//! Integration tests for hyperphysics-neural-trader bridge.

use hyperphysics_neural_trader::prelude::*;
use ndarray::Array2;

/// Create test features with synthetic price data
fn make_test_features(prices: Vec<f64>) -> hyperphysics_neural_trader::adapter::NeuralFeatures {
    let n = prices.len();
    let returns: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0].max(1e-8))
        .collect();

    hyperphysics_neural_trader::adapter::NeuralFeatures {
        prices: prices.clone(),
        returns,
        volatility: vec![0.02; n],
        spreads: vec![0.001; n],
        vwaps: prices.iter().map(|p| p * 0.999).collect(),
        feature_matrix: Array2::zeros((n, 5)),
        targets: None,
        timestamps: (0..n).map(|i| i as f64 * 1000.0).collect(),
    }
}

/// Generate a realistic price series with trend and noise
fn generate_price_series(start: f64, trend: f64, volatility: f64, length: usize) -> Vec<f64> {
    let mut prices = Vec::with_capacity(length);
    let mut price = start;

    for i in 0..length {
        // Add trend component
        price *= 1.0 + trend;
        // Add deterministic "noise" based on position (no random)
        let noise_factor = ((i as f64 * 0.5).sin() * volatility);
        price *= 1.0 + noise_factor;
        prices.push(price);
    }

    prices
}

#[tokio::test]
async fn test_end_to_end_forecast_pipeline() {
    // 1. Create configuration
    let config = NeuralBridgeConfig {
        min_sequence_length: 12,
        forecast_horizon: 6,
        enable_ensemble: true,
        ensemble_models: vec![
            hyperphysics_neural_trader::config::NeuralModelType::NHITS,
            hyperphysics_neural_trader::config::NeuralModelType::LSTMAttention,
        ],
        ensemble_method: hyperphysics_neural_trader::config::EnsembleMethod::WeightedAverage,
        confidence_level: 0.95,
        ..Default::default()
    };

    // 2. Create data adapter
    let adapter = NeuralDataAdapter::new(config.clone());

    // 3. Feed market data
    for i in 0..24 {
        let feed = hyperphysics_neural_trader::MarketFeed {
            price: 100.0 + i as f64 * 0.5,
            returns: vec![0.005],
            volatility: 0.02,
            vwap: 100.0 + i as f64 * 0.5 - 0.1,
            spread: 0.001,
            timestamp: i as f64 * 1000.0,
        };
        adapter.process_feed(&feed).await.ok();
    }

    // 4. Extract features
    assert!(adapter.is_ready().await);
    let features = adapter.extract_features().await.unwrap();
    assert_eq!(features.prices.len(), 24);

    // 5. Create ensemble predictor
    let ensemble = EnsemblePredictor::new(config);

    // 6. Generate forecast
    let forecast = ensemble.predict(&features).await.unwrap();

    // 7. Validate forecast structure
    assert_eq!(forecast.predictions.len(), 6);
    assert_eq!(forecast.lower_bound.len(), 6);
    assert_eq!(forecast.upper_bound.len(), 6);
    assert_eq!(forecast.variance.len(), 6);
    assert_eq!(forecast.confidence_level, 0.95);

    // 8. Validate confidence intervals
    for i in 0..6 {
        assert!(forecast.lower_bound[i] < forecast.predictions[i]);
        assert!(forecast.upper_bound[i] > forecast.predictions[i]);
    }
}

#[tokio::test]
async fn test_hft_optimized_config() {
    let config = NeuralBridgeConfig::hft_optimized();

    // HFT config should have minimal latency settings
    assert_eq!(config.min_sequence_length, 12);
    assert_eq!(config.forecast_horizon, 6);
    assert_eq!(config.max_batch_size, 1);
    assert!(!config.enable_ensemble);
    assert!(!config.enable_conformal);
}

#[tokio::test]
async fn test_high_accuracy_config() {
    let config = NeuralBridgeConfig::high_accuracy();

    // High accuracy config should have larger models
    assert_eq!(config.min_sequence_length, 48);
    assert_eq!(config.hidden_size, 512);
    assert_eq!(config.num_attention_heads, 8);
    assert!(config.enable_ensemble);
    assert!(config.enable_conformal);
    assert!(config.ensemble_models.len() >= 4);
}

#[tokio::test]
async fn test_forecast_with_upward_trend() {
    let config = NeuralBridgeConfig {
        min_sequence_length: 10,
        forecast_horizon: 4,
        ..Default::default()
    };

    let engine = NeuralForecastEngine::new(config);

    // Generate upward trending prices
    let prices = generate_price_series(100.0, 0.002, 0.001, 24);
    let features = make_test_features(prices.clone());

    let forecast = engine.forecast(&features).await.unwrap();

    // With upward trend, predictions should generally be above last price
    let last_price = *prices.last().unwrap();
    assert!(
        forecast.predictions[0] > last_price * 0.95,
        "First prediction {} should be close to or above last price {}",
        forecast.predictions[0],
        last_price
    );
}

#[tokio::test]
async fn test_forecast_with_downward_trend() {
    let config = NeuralBridgeConfig {
        min_sequence_length: 10,
        forecast_horizon: 4,
        ..Default::default()
    };

    let engine = NeuralForecastEngine::new(config);

    // Generate downward trending prices
    let prices = generate_price_series(100.0, -0.002, 0.001, 24);
    let features = make_test_features(prices.clone());

    let forecast = engine.forecast(&features).await.unwrap();

    // With downward trend, predictions should continue the trend
    let last_price = *prices.last().unwrap();
    assert!(
        forecast.predictions[0] < last_price * 1.05,
        "First prediction {} should be close to or below last price {}",
        forecast.predictions[0],
        last_price
    );
}

#[tokio::test]
async fn test_ensemble_weight_adaptation() {
    let config = NeuralBridgeConfig {
        ensemble_models: vec![
            hyperphysics_neural_trader::config::NeuralModelType::NHITS,
            hyperphysics_neural_trader::config::NeuralModelType::LSTMAttention,
            hyperphysics_neural_trader::config::NeuralModelType::Transformer,
        ],
        ensemble_method: hyperphysics_neural_trader::config::EnsembleMethod::WeightedAverage,
        ..Default::default()
    };

    let ensemble = EnsemblePredictor::new(config);

    // Initial weights should be equal
    let initial_weights = ensemble.get_weights().await;
    let initial_nhits = initial_weights
        .get(&hyperphysics_neural_trader::config::NeuralModelType::NHITS)
        .unwrap();
    let initial_lstm = initial_weights
        .get(&hyperphysics_neural_trader::config::NeuralModelType::LSTMAttention)
        .unwrap();

    assert!(
        (initial_nhits - initial_lstm).abs() < 0.01,
        "Initial weights should be approximately equal"
    );

    // Simulate NHITS performing better
    for _ in 0..20 {
        ensemble
            .update_weights(
                hyperphysics_neural_trader::config::NeuralModelType::NHITS,
                0.01,
            )
            .await;
        ensemble
            .update_weights(
                hyperphysics_neural_trader::config::NeuralModelType::LSTMAttention,
                0.05,
            )
            .await;
        ensemble
            .update_weights(
                hyperphysics_neural_trader::config::NeuralModelType::Transformer,
                0.03,
            )
            .await;
    }

    // NHITS should now have higher weight
    let updated_weights = ensemble.get_weights().await;
    let nhits_weight = updated_weights
        .get(&hyperphysics_neural_trader::config::NeuralModelType::NHITS)
        .unwrap();
    let lstm_weight = updated_weights
        .get(&hyperphysics_neural_trader::config::NeuralModelType::LSTMAttention)
        .unwrap();

    assert!(
        nhits_weight > lstm_weight,
        "NHITS weight {} should be higher than LSTM weight {} after better performance",
        nhits_weight,
        lstm_weight
    );
}

#[tokio::test]
async fn test_adapter_buffer_management() {
    let config = NeuralBridgeConfig {
        min_sequence_length: 5,
        feature_window: 10,
        ..Default::default()
    };

    let adapter = NeuralDataAdapter::new(config);

    // Add 15 data points
    for i in 0..15 {
        let feed = hyperphysics_neural_trader::MarketFeed {
            price: 100.0 + i as f64,
            returns: vec![0.01],
            volatility: 0.02,
            vwap: 99.9 + i as f64,
            spread: 0.001,
            timestamp: i as f64 * 1000.0,
        };
        adapter.process_feed(&feed).await.ok();
    }

    // Buffer should be capped at feature_window (10)
    assert_eq!(adapter.buffer_length().await, 10);
}

#[tokio::test]
async fn test_normalization_stats() {
    let config = NeuralBridgeConfig {
        min_sequence_length: 5,
        feature_window: 20,
        ..Default::default()
    };

    let adapter = NeuralDataAdapter::new(config);

    // Add data with known mean
    for i in 0..10 {
        let feed = hyperphysics_neural_trader::MarketFeed {
            price: 100.0 + i as f64, // Mean should be ~104.5
            returns: vec![0.01],
            volatility: 0.02,
            vwap: 99.9,
            spread: 0.001,
            timestamp: i as f64 * 1000.0,
        };
        adapter.process_feed(&feed).await.ok();
    }

    let stats = adapter.get_norm_stats().await;

    // Check that stats were calculated
    assert!(stats.price_mean > 0.0);
    assert!(stats.price_std > 0.0);
    assert_eq!(stats.sample_count, 10);

    // Mean should be approximately 104.5 (100 + 0..9 average)
    assert!(
        (stats.price_mean - 104.5).abs() < 0.1,
        "Price mean {} should be close to 104.5",
        stats.price_mean
    );
}

#[tokio::test]
async fn test_forecast_result_quality_score() {
    let config = NeuralBridgeConfig {
        min_sequence_length: 10,
        forecast_horizon: 4,
        ..Default::default()
    };

    let engine = NeuralForecastEngine::new(config);
    let prices = generate_price_series(100.0, 0.001, 0.001, 24);
    let features = make_test_features(prices);

    let forecast = engine.forecast(&features).await.unwrap();
    let quality = forecast.quality_score();

    // Quality score should be between 0 and 1
    assert!(quality > 0.0 && quality <= 1.0);
}

#[tokio::test]
async fn test_insufficient_data_error() {
    let config = NeuralBridgeConfig {
        min_sequence_length: 50,
        ..Default::default()
    };

    let engine = NeuralForecastEngine::new(config);
    let features = make_test_features(vec![100.0, 101.0, 102.0]);

    let result = engine.forecast(&features).await;

    assert!(matches!(
        result,
        Err(NeuralBridgeError::InsufficientData { required: 50, .. })
    ));
}

#[tokio::test]
async fn test_all_model_types() {
    use hyperphysics_neural_trader::config::NeuralModelType;

    let model_types = vec![
        NeuralModelType::NHITS,
        NeuralModelType::LSTMAttention,
        NeuralModelType::Transformer,
        NeuralModelType::GRU,
        NeuralModelType::TCN,
        NeuralModelType::DeepAR,
        NeuralModelType::NBeats,
        NeuralModelType::Prophet,
    ];

    let prices = generate_price_series(100.0, 0.001, 0.002, 24);
    let features = make_test_features(prices);

    for model_type in model_types {
        let config = NeuralBridgeConfig {
            min_sequence_length: 10,
            forecast_horizon: 4,
            ensemble_models: vec![model_type],
            ensemble_method: hyperphysics_neural_trader::config::EnsembleMethod::Single,
            ..Default::default()
        };

        let engine = NeuralForecastEngine::new(config);
        let result = engine.forecast(&features).await;

        assert!(
            result.is_ok(),
            "Model {:?} should produce valid forecast",
            model_type
        );

        let forecast = result.unwrap();
        assert_eq!(
            forecast.predictions.len(),
            4,
            "Model {:?} should produce 4 predictions",
            model_type
        );
    }
}

#[tokio::test]
async fn test_median_ensemble_robustness() {
    use hyperphysics_neural_trader::config::{EnsembleMethod, NeuralModelType};

    let config = NeuralBridgeConfig {
        min_sequence_length: 10,
        forecast_horizon: 4,
        ensemble_models: vec![
            NeuralModelType::NHITS,
            NeuralModelType::LSTMAttention,
            NeuralModelType::Transformer,
            NeuralModelType::GRU,
            NeuralModelType::TCN,
        ],
        ensemble_method: EnsembleMethod::Median,
        ..Default::default()
    };

    let ensemble = EnsemblePredictor::new(config);
    let prices = generate_price_series(100.0, 0.001, 0.001, 24);
    let features = make_test_features(prices);

    let forecast = ensemble.predict(&features).await.unwrap();

    // Median ensemble should produce predictions within reasonable bounds
    for pred in &forecast.predictions {
        assert!(
            *pred > 50.0 && *pred < 200.0,
            "Prediction {} out of reasonable bounds",
            pred
        );
    }
}
