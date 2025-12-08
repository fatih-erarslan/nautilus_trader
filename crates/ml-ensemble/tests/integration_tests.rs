use ml_ensemble::{
    create_default_ensemble,
    EnsembleConfig,
    EnsemblePrediction,
    MarketCondition,
    ModelType,
    FeatureConfig,
    ModelWeightsConfig,
    CalibrationConfig,
};
use ats_core::types::MarketData;
use anyhow::Result;

#[tokio::test]
async fn test_ensemble_initialization() -> Result<()> {
    let ensemble = create_default_ensemble().await?;
    let metrics = ensemble.get_metrics();
    
    // Should have at least one model
    assert!(!metrics.is_empty());
    
    Ok(())
}

#[tokio::test]
async fn test_prediction_latency() -> Result<()> {
    let ensemble = create_default_ensemble().await?;
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    // Make multiple predictions to warm up
    for _ in 0..10 {
        let _ = ensemble.predict(&market_data).await?;
    }
    
    // Test actual latency
    let prediction = ensemble.predict(&market_data).await?;
    
    // Should meet latency requirement
    assert!(prediction.latency_us < 1000.0); // 1ms for test environment
    
    Ok(())
}

#[tokio::test]
async fn test_confidence_bounds() -> Result<()> {
    let ensemble = create_default_ensemble().await?;
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    let prediction = ensemble.predict(&market_data).await?;
    
    // Confidence should be between 0 and 1
    assert!(prediction.confidence >= 0.0);
    assert!(prediction.confidence <= 1.0);
    
    // Individual model confidences should also be valid
    for model_pred in &prediction.model_predictions {
        assert!(model_pred.confidence >= 0.0);
        assert!(model_pred.confidence <= 1.0);
        assert!(model_pred.weight >= 0.0);
        assert!(model_pred.weight <= 1.0);
    }
    
    Ok(())
}

#[tokio::test]
async fn test_market_conditions() -> Result<()> {
    use ml_ensemble::market_detector::MarketConditionDetector;
    
    let detector = MarketConditionDetector::new();
    
    // Test different market scenarios
    let scenarios = vec![
        // Normal market
        MarketData {
            timestamp: 0,
            bid: 100.0,
            ask: 100.01,
            bid_size: 1000.0,
            ask_size: 1000.0,
        },
        // High spread (potential volatility)
        MarketData {
            timestamp: 1,
            bid: 100.0,
            ask: 100.5,
            bid_size: 1000.0,
            ask_size: 1000.0,
        },
        // Volume imbalance
        MarketData {
            timestamp: 2,
            bid: 100.0,
            ask: 100.01,
            bid_size: 5000.0,
            ask_size: 100.0,
        },
    ];
    
    for data in scenarios {
        let condition = detector.detect_condition(&data)?;
        // Should always return a valid condition
        match condition {
            MarketCondition::Trending |
            MarketCondition::Ranging |
            MarketCondition::HighVolatility |
            MarketCondition::LowVolatility |
            MarketCondition::Breakout |
            MarketCondition::Reversal |
            MarketCondition::Anomalous => {},
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_feature_engineering() -> Result<()> {
    use ml_ensemble::features::FeatureEngineering;
    
    let config = FeatureConfig {
        price_lags: 10,
        technical_indicators: vec!["RSI".to_string(), "MACD".to_string()],
        microstructure_features: true,
        order_flow_features: true,
        sentiment_features: false,
    };
    
    let feature_engine = FeatureEngineering::new(config);
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    let features = feature_engine.extract_features(&market_data)?;
    
    // Should have the expected number of features
    assert!(features.len() > 0);
    
    // Features should be finite
    for &feature in features.iter() {
        assert!(feature.is_finite());
    }
    
    Ok(())
}

#[tokio::test]
async fn test_weight_updates() -> Result<()> {
    use ml_ensemble::weights::WeightManager;
    
    let config = ModelWeightsConfig {
        dynamic_weights: true,
        update_frequency: 1, // Update every second for testing
        lookback_period: 10,
        initial_weights: vec![
            (ModelType::XGBoost, 0.25),
            (ModelType::LightGBM, 0.25),
            (ModelType::LSTM, 0.25),
            (ModelType::Transformer, 0.25),
        ],
    };
    
    let mut weight_manager = WeightManager::new(config);
    
    // Add some performance records
    for i in 0..20 {
        weight_manager.add_prediction_record(
            ModelType::XGBoost,
            0.001,
            0.0015,
            0.25,
        );
        weight_manager.add_prediction_record(
            ModelType::LSTM,
            0.002,
            0.0015,
            0.25,
        );
    }
    
    // Update weights
    weight_manager.update_performance(0.001)?;
    
    let weights = weight_manager.get_all_weights();
    
    // Weights should sum to approximately 1
    let total_weight: f64 = weights.values().sum();
    assert!((total_weight - 1.0).abs() < 0.01);
    
    Ok(())
}

#[tokio::test]
async fn test_calibration() -> Result<()> {
    use ml_ensemble::calibration::ConfidenceCalibrator;
    
    let config = CalibrationConfig {
        isotonic_regression: true,
        platt_scaling: false,
        window_size: 100,
        min_samples: 10,
    };
    
    let mut calibrator = ConfidenceCalibrator::new(config);
    
    // Add calibration samples
    for i in 0..50 {
        let confidence = 0.5 + (i as f64 / 100.0);
        let outcome = if i % 2 == 0 { 1.0 } else { 0.0 };
        calibrator.add_sample(confidence, outcome, MarketCondition::Trending);
    }
    
    // Test calibration
    let raw_confidence = 0.7;
    let calibrated = calibrator.calibrate_confidence(raw_confidence, MarketCondition::Trending)?;
    
    // Calibrated confidence should be valid
    assert!(calibrated >= 0.0 && calibrated <= 1.0);
    
    Ok(())
}

#[tokio::test]
async fn test_model_selection() -> Result<()> {
    use ml_ensemble::model_selector::ModelSelector;
    use ml_ensemble::ensemble::ModelPredictor;
    use std::collections::HashMap;
    use std::sync::Arc;
    
    let selector = ModelSelector::new();
    
    // Create dummy models
    let mut models: HashMap<ModelType, Arc<dyn ModelPredictor>> = HashMap::new();
    
    #[cfg(feature = "tree-models")]
    {
        use ml_ensemble::tree_models::{XGBoostModel, LightGBMModel};
        
        models.insert(
            ModelType::XGBoost,
            Arc::new(XGBoostModel::new(Default::default())?),
        );
        models.insert(
            ModelType::LightGBM,
            Arc::new(LightGBMModel::new(Default::default())?),
        );
    }
    
    // Test selection for different market conditions
    for condition in vec![
        MarketCondition::Trending,
        MarketCondition::Ranging,
        MarketCondition::HighVolatility,
        MarketCondition::Anomalous,
    ] {
        let selected = selector.select_models(condition, &models)?;
        
        // Should select at least one model
        assert!(!selected.is_empty());
        
        // Selected models should be from available models
        for model in selected {
            assert!(models.contains_key(&model.model_type()));
        }
    }
    
    Ok(())
}

#[tokio::test]
async fn test_ensemble_interpretability() -> Result<()> {
    let ensemble = create_default_ensemble().await?;
    
    let market_data = MarketData {
        timestamp: 0,
        bid: 100.0,
        ask: 100.1,
        bid_size: 1000.0,
        ask_size: 1000.0,
    };
    
    let prediction = ensemble.predict(&market_data).await?;
    
    // Should have feature importance
    assert!(!prediction.feature_importance.is_empty());
    
    // Feature importance should be valid
    for (name, importance) in &prediction.feature_importance {
        assert!(!name.is_empty());
        assert!(*importance >= 0.0);
    }
    
    Ok(())
}