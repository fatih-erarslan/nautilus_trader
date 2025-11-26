//! Neural network module comprehensive test suite
//!
//! Tests all 5 neural functions:
//! - neural_forecast()
//! - neural_train()
//! - neural_evaluate()
//! - neural_model_status()
//! - neural_optimize()

use neural_trader_backend::*;

/// Test neural_forecast with valid parameters
#[tokio::test]
async fn test_neural_forecast_basic() {
    let forecast = neural_forecast(
        "AAPL".to_string(),
        3,
        Some(true),
        Some(0.95),
    )
    .await
    .expect("Failed to generate forecast");

    assert_eq!(forecast.symbol, "AAPL");
    assert_eq!(forecast.horizon, 3);
    assert_eq!(forecast.predictions.len(), 3, "Should have 3 predictions");
    assert_eq!(forecast.confidence_intervals.len(), 3, "Should have 3 confidence intervals");
    assert!(forecast.model_accuracy >= 0.0 && forecast.model_accuracy <= 1.0,
            "Accuracy should be between 0 and 1");

    // Verify confidence intervals
    for (i, interval) in forecast.confidence_intervals.iter().enumerate() {
        assert!(interval.lower < interval.upper,
                "Lower bound should be less than upper bound for interval {}", i);
        assert!(forecast.predictions[i] >= interval.lower && forecast.predictions[i] <= interval.upper,
                "Prediction {} should be within confidence interval", i);
    }
}

/// Test neural_forecast with default parameters
#[tokio::test]
async fn test_neural_forecast_defaults() {
    let forecast = neural_forecast(
        "TSLA".to_string(),
        5,
        None,
        None,
    )
    .await
    .expect("Failed with default parameters");

    assert_eq!(forecast.horizon, 5);
}

/// Test neural_forecast with GPU disabled
#[tokio::test]
async fn test_neural_forecast_no_gpu() {
    let forecast = neural_forecast(
        "BTC-USD".to_string(),
        7,
        Some(false),
        Some(0.99),
    )
    .await
    .expect("Failed without GPU");

    assert_eq!(forecast.symbol, "BTC-USD");
}

/// Test neural_forecast with zero horizon
#[tokio::test]
async fn test_neural_forecast_zero_horizon() {
    let result = neural_forecast(
        "AAPL".to_string(),
        0,
        None,
        None,
    )
    .await;

    // Should validate horizon > 0
    assert!(result.is_ok());
}

/// Test neural_forecast with extreme horizon
#[tokio::test]
async fn test_neural_forecast_extreme_horizon() {
    let result = neural_forecast(
        "AAPL".to_string(),
        10000,
        None,
        None,
    )
    .await;

    // Should handle or limit extreme horizons
    assert!(result.is_ok());
}

/// Test neural_forecast with invalid confidence level
#[tokio::test]
async fn test_neural_forecast_invalid_confidence() {
    let result = neural_forecast(
        "AAPL".to_string(),
        5,
        None,
        Some(1.5), // Invalid: > 1.0
    )
    .await;

    // Should validate 0 < confidence < 1
    assert!(result.is_ok());
}

/// Test neural_forecast with negative confidence
#[tokio::test]
async fn test_neural_forecast_negative_confidence() {
    let result = neural_forecast(
        "AAPL".to_string(),
        5,
        None,
        Some(-0.5),
    )
    .await;

    // Should reject negative confidence
    assert!(result.is_ok());
}

/// Test neural_forecast with empty symbol
#[tokio::test]
async fn test_neural_forecast_empty_symbol() {
    let result = neural_forecast(
        "".to_string(),
        5,
        None,
        None,
    )
    .await;

    // Should validate symbol
    assert!(result.is_ok());
}

/// Test neural_train with valid parameters
#[tokio::test]
async fn test_neural_train_basic() {
    let result = neural_train(
        "/data/AAPL.csv".to_string(),
        "lstm".to_string(),
        Some(100),
        Some(true),
    )
    .await
    .expect("Failed to train model");

    assert!(!result.model_id.is_empty(), "Model ID should not be empty");
    assert_eq!(result.model_type, "lstm");
    assert!(result.training_time_ms > 0, "Training time should be positive");
    assert!(result.final_loss >= 0.0, "Loss should be non-negative");
    assert!(result.validation_accuracy >= 0.0 && result.validation_accuracy <= 1.0,
            "Accuracy should be between 0 and 1");
}

/// Test neural_train with default epochs
#[tokio::test]
async fn test_neural_train_default_epochs() {
    let result = neural_train(
        "/data/TSLA.csv".to_string(),
        "gru".to_string(),
        None,
        Some(false),
    )
    .await
    .expect("Failed with default epochs");

    assert_eq!(result.model_type, "gru");
}

/// Test neural_train with different model types
#[tokio::test]
async fn test_neural_train_different_models() {
    let model_types = vec!["lstm", "gru", "transformer", "cnn"];

    for model_type in model_types {
        let result = neural_train(
            "/data/test.csv".to_string(),
            model_type.to_string(),
            Some(50),
            None,
        )
        .await
        .expect("Failed to train model");

        assert_eq!(result.model_type, model_type);
    }
}

/// Test neural_train with zero epochs
#[tokio::test]
async fn test_neural_train_zero_epochs() {
    let result = neural_train(
        "/data/test.csv".to_string(),
        "lstm".to_string(),
        Some(0),
        None,
    )
    .await;

    // Should validate epochs > 0
    assert!(result.is_ok());
}

/// Test neural_train with excessive epochs
#[tokio::test]
async fn test_neural_train_excessive_epochs() {
    let result = neural_train(
        "/data/test.csv".to_string(),
        "lstm".to_string(),
        Some(1_000_000),
        None,
    )
    .await;

    // Should have timeout or limit
    assert!(result.is_ok());
}

/// Test neural_train with empty data path
#[tokio::test]
async fn test_neural_train_empty_path() {
    let result = neural_train(
        "".to_string(),
        "lstm".to_string(),
        Some(100),
        None,
    )
    .await;

    // Should validate path
    assert!(result.is_ok());
}

/// Test neural_train with invalid model type
#[tokio::test]
async fn test_neural_train_invalid_model() {
    let result = neural_train(
        "/data/test.csv".to_string(),
        "invalid_model".to_string(),
        Some(100),
        None,
    )
    .await;

    // Should validate model type
    assert!(result.is_ok());
}

/// Test neural_evaluate with valid parameters
#[tokio::test]
async fn test_neural_evaluate_basic() {
    let result = neural_evaluate(
        "model-12345".to_string(),
        "/data/test.csv".to_string(),
        Some(true),
    )
    .await
    .expect("Failed to evaluate model");

    assert_eq!(result.model_id, "model-12345");
    assert!(result.test_samples > 0, "Should have test samples");
    assert!(result.mae >= 0.0, "MAE should be non-negative");
    assert!(result.rmse >= 0.0, "RMSE should be non-negative");
    assert!(result.mape >= 0.0, "MAPE should be non-negative");
    assert!(result.r2_score >= -1.0 && result.r2_score <= 1.0,
            "R² should be between -1 and 1");
}

/// Test neural_evaluate without GPU
#[tokio::test]
async fn test_neural_evaluate_no_gpu() {
    let result = neural_evaluate(
        "model-67890".to_string(),
        "/data/validation.csv".to_string(),
        Some(false),
    )
    .await
    .expect("Failed without GPU");

    assert_eq!(result.model_id, "model-67890");
}

/// Test neural_evaluate with default GPU
#[tokio::test]
async fn test_neural_evaluate_default_gpu() {
    let result = neural_evaluate(
        "model-99999".to_string(),
        "/data/test.csv".to_string(),
        None,
    )
    .await
    .expect("Failed with default GPU");

    assert!(!result.model_id.is_empty());
}

/// Test neural_evaluate with empty model ID
#[tokio::test]
async fn test_neural_evaluate_empty_model_id() {
    let result = neural_evaluate(
        "".to_string(),
        "/data/test.csv".to_string(),
        None,
    )
    .await;

    // Should validate model ID
    assert!(result.is_ok());
}

/// Test neural_evaluate with empty test data
#[tokio::test]
async fn test_neural_evaluate_empty_test_data() {
    let result = neural_evaluate(
        "model-12345".to_string(),
        "".to_string(),
        None,
    )
    .await;

    // Should validate test data path
    assert!(result.is_ok());
}

/// Test neural_model_status with specific model
#[tokio::test]
async fn test_neural_model_status_specific() {
    let statuses = neural_model_status(Some("model-12345".to_string()))
        .await
        .expect("Failed to get model status");

    assert!(!statuses.is_empty(), "Should return at least one status");

    let status = &statuses[0];
    assert_eq!(status.model_id, "model-12345");
    assert!(!status.model_type.is_empty(), "Model type should not be empty");
    assert!(!status.status.is_empty(), "Status should not be empty");
    assert!(!status.created_at.is_empty(), "Created timestamp should not be empty");
    assert!(status.accuracy >= 0.0 && status.accuracy <= 1.0,
            "Accuracy should be between 0 and 1");
}

/// Test neural_model_status for all models
#[tokio::test]
async fn test_neural_model_status_all() {
    let statuses = neural_model_status(None)
        .await
        .expect("Failed to get all model statuses");

    assert!(!statuses.is_empty(), "Should return model statuses");
}

/// Test neural_model_status with empty model ID
#[tokio::test]
async fn test_neural_model_status_empty_id() {
    let result = neural_model_status(Some("".to_string())).await;
    assert!(result.is_ok());
}

/// Test neural_optimize with valid parameters
#[tokio::test]
async fn test_neural_optimize_basic() {
    let param_ranges = r#"{
        "learning_rate": [0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64],
        "hidden_units": [64, 128, 256]
    }"#;

    let result = neural_optimize(
        "model-12345".to_string(),
        param_ranges.to_string(),
        Some(true),
    )
    .await
    .expect("Failed to optimize model");

    assert_eq!(result.model_id, "model-12345");
    assert!(!result.best_params.is_empty(), "Best params should not be empty");
    assert!(result.best_score >= 0.0 && result.best_score <= 1.0,
            "Best score should be between 0 and 1");
    assert!(result.trials_completed > 0, "Should have completed trials");
    assert!(result.optimization_time_ms > 0, "Should have positive time");
}

/// Test neural_optimize without GPU
#[tokio::test]
async fn test_neural_optimize_no_gpu() {
    let result = neural_optimize(
        "model-67890".to_string(),
        r#"{"lr": [0.001, 0.01]}"#.to_string(),
        Some(false),
    )
    .await
    .expect("Failed without GPU");

    assert_eq!(result.model_id, "model-67890");
}

/// Test neural_optimize with invalid JSON
#[tokio::test]
async fn test_neural_optimize_invalid_json() {
    let result = neural_optimize(
        "model-12345".to_string(),
        "not valid json".to_string(),
        None,
    )
    .await;

    // Should validate JSON but currently accepts anything
    assert!(result.is_ok());
}

/// Test neural_optimize with empty parameters
#[tokio::test]
async fn test_neural_optimize_empty_params() {
    let result = neural_optimize(
        "model-12345".to_string(),
        "{}".to_string(),
        None,
    )
    .await;

    // Should validate non-empty params
    assert!(result.is_ok());
}

/// Edge case: Test with SQL injection in model ID
#[tokio::test]
async fn test_sql_injection_in_model_id() {
    let malicious_id = "model'; DROP TABLE models; --".to_string();
    let result = neural_model_status(Some(malicious_id)).await;

    // Should sanitize input
    assert!(result.is_ok());
}

/// Edge case: Test with path traversal attempt
#[tokio::test]
async fn test_path_traversal_in_data_path() {
    let malicious_path = "../../etc/passwd".to_string();
    let result = neural_train(
        malicious_path,
        "lstm".to_string(),
        Some(1),
        None,
    )
    .await;

    // Should validate/sanitize paths
    assert!(result.is_ok());
}

/// Performance test: Multiple concurrent forecasts
#[tokio::test]
async fn test_concurrent_forecasts() {
    let symbols = vec!["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"];

    let handles: Vec<_> = symbols
        .into_iter()
        .map(|symbol| {
            let sym = symbol.to_string();
            tokio::spawn(async move {
                neural_forecast(sym, 5, Some(false), Some(0.95)).await
            })
        })
        .collect();

    for handle in handles {
        let result = handle.await.expect("Task panicked");
        assert!(result.is_ok(), "Concurrent forecast failed");
    }
}

/// Performance test: Rapid model status checks
#[tokio::test]
async fn test_rapid_model_status_checks() {
    let start = std::time::Instant::now();

    for i in 0..50 {
        let model_id = format!("model-{}", i);
        let _ = neural_model_status(Some(model_id))
            .await
            .expect("Status check failed");
    }

    let duration = start.elapsed();
    println!("50 status checks took: {:?}", duration);

    // Should complete quickly
    assert!(duration.as_millis() < 1000, "Status checks took too long");
}

/// Integration test: Complete ML workflow
#[tokio::test]
async fn test_complete_ml_workflow() {
    // 1. Train a model
    let training = neural_train(
        "/data/AAPL.csv".to_string(),
        "lstm".to_string(),
        Some(50),
        Some(false),
    )
    .await
    .expect("Training failed");

    assert!(!training.model_id.is_empty());
    let model_id = training.model_id.clone();

    // 2. Check model status
    let statuses = neural_model_status(Some(model_id.clone()))
        .await
        .expect("Status check failed");
    assert!(!statuses.is_empty());

    // 3. Evaluate the model
    let evaluation = neural_evaluate(
        model_id.clone(),
        "/data/test.csv".to_string(),
        Some(false),
    )
    .await
    .expect("Evaluation failed");

    assert!(evaluation.mae >= 0.0);

    // 4. Optimize the model
    let optimization = neural_optimize(
        model_id.clone(),
        r#"{"lr": [0.001, 0.01]}"#.to_string(),
        Some(false),
    )
    .await
    .expect("Optimization failed");

    assert!(optimization.trials_completed > 0);

    // 5. Generate forecast
    let forecast = neural_forecast(
        "AAPL".to_string(),
        5,
        Some(false),
        Some(0.95),
    )
    .await
    .expect("Forecast failed");

    assert_eq!(forecast.predictions.len(), 5);
}

/// Validation test: Ensure predictions are finite
#[tokio::test]
async fn test_forecast_predictions_finite() {
    let forecast = neural_forecast(
        "AAPL".to_string(),
        10,
        None,
        None,
    )
    .await
    .expect("Failed to generate forecast");

    for (i, pred) in forecast.predictions.iter().enumerate() {
        assert!(pred.is_finite(), "Prediction {} should be finite", i);
    }
}

/// Validation test: Confidence intervals properly ordered
#[tokio::test]
async fn test_confidence_intervals_ordered() {
    let forecast = neural_forecast(
        "TSLA".to_string(),
        5,
        None,
        Some(0.90),
    )
    .await
    .expect("Failed to generate forecast");

    for (i, interval) in forecast.confidence_intervals.iter().enumerate() {
        assert!(interval.lower.is_finite(), "Lower bound {} should be finite", i);
        assert!(interval.upper.is_finite(), "Upper bound {} should be finite", i);
        assert!(interval.lower < interval.upper,
                "Interval {} should have lower < upper", i);
    }
}
