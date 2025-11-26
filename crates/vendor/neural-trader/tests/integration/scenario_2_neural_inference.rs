// Integration Test Scenario 2: Neural Model Training & Inference
// Tests complete neural network workflow

#[cfg(feature = "neural")]
use neural_trader_neural::models::{NeuralModel, NHiTSModel};
#[cfg(feature = "neural")]
use neural_trader_neural::training::Trainer;
use neural_trader_core::types::*;
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};

#[tokio::test]
#[ignore] // Requires GPU/large compute
#[cfg(feature = "neural")]
async fn test_neural_training_workflow() -> anyhow::Result<()> {
    use polars::prelude::*;

    // Generate synthetic training data
    let dates: Vec<_> = (0..1000)
        .map(|i| Utc::now() - Duration::days(1000 - i))
        .collect();

    let prices: Vec<f64> = (0..1000)
        .map(|i| 100.0 + (i as f64 * 0.1) + ((i as f64 / 10.0).sin() * 5.0))
        .collect();

    let df = DataFrame::new(vec![
        Series::new("timestamp", dates),
        Series::new("close", prices),
    ])?;

    // Create and train model
    let mut model = NHiTSModel::new(
        vec![512, 512, 512], // stack sizes
        20,                   // input_chunk_length
        5,                    // output_chunk_length
        2,                    // num_blocks
        1,                    // num_layers
    )?;

    let trainer = Trainer::new(
        0.001,  // learning_rate
        100,    // epochs
        32,     // batch_size
    );

    trainer.train(&mut model, &df).await?;

    // Test inference
    let test_input = &prices[980..1000]; // Last 20 values
    let predictions = model.predict(test_input)?;

    assert_eq!(predictions.len(), 5, "Should predict 5 future values");
    assert!(predictions.iter().all(|&p| p > 0.0), "Predictions should be positive");

    // Save model
    let model_path = "/tmp/test_model.safetensors";
    model.save(model_path)?;

    // Load and verify
    let loaded_model = NHiTSModel::load(model_path)?;
    let loaded_predictions = loaded_model.predict(test_input)?;

    assert_eq!(
        predictions, loaded_predictions,
        "Loaded model should produce same predictions"
    );

    println!("✅ Neural training and inference completed");
    Ok(())
}

#[tokio::test]
#[cfg(feature = "neural")]
async fn test_neural_model_architecture() -> anyhow::Result<()> {
    // Test model creation without training
    let model = NHiTSModel::new(
        vec![256, 256],
        10,
        3,
        2,
        1,
    )?;

    assert_eq!(model.input_size(), 10);
    assert_eq!(model.output_size(), 3);

    Ok(())
}

#[tokio::test]
async fn test_feature_engineering_for_neural() -> anyhow::Result<()> {
    use neural_trader_features::calculator::FeatureCalculator;
    use polars::prelude::*;

    // Create sample price data
    let prices: Vec<f64> = vec![100.0, 101.0, 102.5, 101.8, 103.0];
    let df = DataFrame::new(vec![
        Series::new("close", prices.clone()),
        Series::new("open", prices.clone()),
        Series::new("high", prices.iter().map(|p| p * 1.02).collect::<Vec<_>>()),
        Series::new("low", prices.iter().map(|p| p * 0.98).collect::<Vec<_>>()),
    ])?;

    let calculator = FeatureCalculator::new();

    // Calculate technical indicators for neural input
    let features = calculator.calculate_all(&df)?;

    // Verify features were calculated
    assert!(features.width() > 4, "Should have additional feature columns");

    Ok(())
}

#[tokio::test]
#[cfg(not(feature = "neural"))]
async fn test_neural_feature_disabled() {
    // When neural feature is disabled, ensure graceful degradation
    println!("ℹ️  Neural features not enabled - skipping neural tests");
}
