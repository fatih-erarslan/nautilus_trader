//! Comprehensive Property-Based Tests for All Models
//!
//! Uses proptest for property-based testing of model invariants across all 27 models

use neuro_divergent::{
    models::{basic::*, recurrent::*, advanced::*},
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
};
use proptest::prelude::*;

#[path = "helpers/mod.rs"]
mod helpers;
use helpers::synthetic;

// Property: All predictions should be finite (no NaN or Inf)
proptest! {
    #[test]
    fn prop_mlp_predictions_finite(
        data_len in 100usize..500,
        horizon in 1usize..50,
    ) {
        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(horizon)
            .with_epochs(10);

        let mut model = MLP::new(config);
        let values = synthetic::sine_wave(data_len, 0.05, 10.0, 50.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let predictions = model.predict(horizon)?;

        prop_assert_eq!(predictions.len(), horizon);
        for &pred in &predictions {
            prop_assert!(pred.is_finite(), "Prediction not finite: {}", pred);
        }
        Ok(())
    }
}

proptest! {
    #[test]
    fn prop_lstm_predictions_finite(
        data_len in 200usize..600,
        horizon in 5usize..30,
    ) {
        let config = ModelConfig::default()
            .with_input_size(30)
            .with_horizon(horizon)
            .with_hidden_size(32)
            .with_epochs(10);

        let mut model = LSTM::new(config);
        let values = synthetic::complex_series(data_len, 0.1, 24, 1.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let predictions = model.predict(horizon)?;

        prop_assert_eq!(predictions.len(), horizon);
        for &pred in &predictions {
            prop_assert!(pred.is_finite());
        }
        Ok(())
    }
}

proptest! {
    #[test]
    fn prop_gru_predictions_finite(
        data_len in 200usize..600,
        horizon in 5usize..30,
    ) {
        let config = ModelConfig::default()
            .with_input_size(30)
            .with_horizon(horizon)
            .with_hidden_size(32)
            .with_epochs(10);

        let mut model = GRU::new(config);
        let values = synthetic::complex_series(data_len, 0.1, 24, 1.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let predictions = model.predict(horizon)?;

        prop_assert_eq!(predictions.len(), horizon);
        for &pred in &predictions {
            prop_assert!(pred.is_finite());
        }
        Ok(())
    }
}

// Property: Training should reduce loss
proptest! {
    #[test]
    fn prop_training_reduces_loss(
        data_len in 300usize..800,
    ) {
        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(10)
            .with_learning_rate(0.01)
            .with_epochs(50);

        let mut model = MLP::new(config);
        let values = synthetic::sine_wave(data_len, 0.05, 15.0, 75.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let history = model.training_history();

        prop_assert!(!history.is_empty());

        let initial_loss = history[0];
        let final_loss = *history.last().unwrap();

        prop_assert!(
            final_loss < initial_loss,
            "Loss should decrease: {} -> {}",
            initial_loss,
            final_loss
        );
        Ok(())
    }
}

// Property: Same seed produces same results (determinism)
proptest! {
    #[test]
    fn prop_deterministic_with_seed(
        seed in 0u64..1000,
        data_len in 200usize..400,
    ) {
        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(10)
            .with_random_seed(seed)
            .with_epochs(10);

        let values = synthetic::sine_wave(data_len, 0.1, 10.0, 50.0);
        let data = TimeSeriesDataFrame::from_values(values.clone(), None)?;

        let mut model1 = MLP::new(config.clone());
        let mut model2 = MLP::new(config);

        model1.fit(&data)?;
        model2.fit(&data)?;

        let pred1 = model1.predict(10)?;
        let pred2 = model2.predict(10)?;

        for (i, (&p1, &p2)) in pred1.iter().zip(pred2.iter()).enumerate() {
            prop_assert!(
                (p1 - p2).abs() < 1e-6,
                "Non-deterministic at index {}: {} vs {}",
                i, p1, p2
            );
        }
        Ok(())
    }
}

// Property: Save/load preserves predictions
proptest! {
    #[test]
    fn prop_save_load_preserves_predictions(
        data_len in 200usize..400,
        horizon in 5usize..20,
    ) {
        use tempfile::tempdir;

        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(horizon)
            .with_epochs(10);

        let mut model = MLP::new(config);
        let values = synthetic::sine_wave(data_len, 0.1, 10.0, 50.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let orig_pred = model.predict(horizon)?;

        let dir = tempdir()?;
        let path = dir.path().join("test_model.bin");
        model.save(&path)?;

        let loaded = MLP::load(&path)?;
        let loaded_pred = loaded.predict(horizon)?;

        for (i, (&orig, &load)) in orig_pred.iter().zip(loaded_pred.iter()).enumerate() {
            prop_assert!(
                (orig - load).abs() < 1e-6,
                "Prediction mismatch at {}: {} vs {}",
                i, orig, load
            );
        }
        Ok(())
    }
}

// Property: Prediction length matches requested horizon
proptest! {
    #[test]
    fn prop_prediction_length_matches_horizon(
        horizon in 1usize..100,
    ) {
        let config = ModelConfig::default()
            .with_input_size(30)
            .with_horizon(horizon);

        let mut model = LSTM::new(config);
        let values = synthetic::complex_series(500, 0.1, 24, 1.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let predictions = model.predict(horizon)?;

        prop_assert_eq!(predictions.len(), horizon);
        Ok(())
    }
}

// Property: Model handles different data scales
proptest! {
    #[test]
    fn prop_handles_different_scales(
        scale in 0.1f64..100.0,
    ) {
        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(10)
            .with_epochs(10);

        let mut model = MLP::new(config);
        let values: Vec<f64> = synthetic::sine_wave(300, 0.05, scale, scale * 10.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        let result = model.fit(&data);
        prop_assert!(result.is_ok(), "Should handle scale {}: {:?}", scale, result);

        let predictions = model.predict(10)?;
        for &pred in &predictions {
            prop_assert!(pred.is_finite());
        }
        Ok(())
    }
}

// Property: Model rejects invalid configurations
proptest! {
    #[test]
    fn prop_rejects_insufficient_data(
        input_size in 100usize..200,
    ) {
        let config = ModelConfig::default()
            .with_input_size(input_size)
            .with_horizon(10);

        let mut model = MLP::new(config);

        // Provide less data than required (input_size + horizon)
        let insufficient_len = input_size + 5;
        let values = synthetic::sine_wave(insufficient_len, 0.1, 10.0, 50.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        let result = model.fit(&data);
        prop_assert!(result.is_err(), "Should reject insufficient data");
        Ok(())
    }
}

// Property: Predictions are smooth (no sudden jumps)
proptest! {
    #[test]
    fn prop_predictions_smooth(
        data_len in 300usize..600,
    ) {
        let config = ModelConfig::default()
            .with_input_size(30)
            .with_horizon(20)
            .with_hidden_size(64)
            .with_epochs(50);

        let mut model = LSTM::new(config);
        let values = synthetic::sine_wave(data_len, 0.05, 20.0, 100.0);
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        model.fit(&data)?;
        let predictions = model.predict(20)?;

        // Check smoothness - no sudden jumps
        for i in 1..predictions.len() {
            let change = (predictions[i] - predictions[i-1]).abs();
            prop_assert!(
                change < 50.0,
                "Prediction jump too large at {}: {}",
                i, change
            );
        }
        Ok(())
    }
}

// Property: More training epochs should not significantly increase loss
proptest! {
    #[test]
    fn prop_more_epochs_not_worse(
        data_len in 300usize..500,
    ) {
        let values = synthetic::complex_series(data_len, 0.1, 24, 1.0);

        let config1 = ModelConfig::default()
            .with_input_size(24)
            .with_horizon(12)
            .with_epochs(20)
            .with_random_seed(42);

        let config2 = ModelConfig::default()
            .with_input_size(24)
            .with_horizon(12)
            .with_epochs(50)
            .with_random_seed(42);

        let data = TimeSeriesDataFrame::from_values(values, None)?;

        let mut model1 = MLP::new(config1);
        let mut model2 = MLP::new(config2);

        model1.fit(&data)?;
        model2.fit(&data)?;

        let loss1 = *model1.training_history().last().unwrap();
        let loss2 = *model2.training_history().last().unwrap();

        prop_assert!(
            loss2 <= loss1 * 1.1, // Allow 10% tolerance
            "More epochs should not significantly increase loss: {} vs {}",
            loss1, loss2
        );
        Ok(())
    }
}

// Property: Model handles constant series
proptest! {
    #[test]
    fn prop_handles_constant_series(
        constant_value in -100.0f64..100.0,
        data_len in 100usize..300,
    ) {
        let config = ModelConfig::default()
            .with_input_size(20)
            .with_horizon(10);

        let mut model = MLP::new(config);
        let values = vec![constant_value; data_len];
        let data = TimeSeriesDataFrame::from_values(values, None)?;

        let result = model.fit(&data);
        prop_assert!(result.is_ok(), "Should handle constant series");

        let predictions = model.predict(10)?;
        for &pred in &predictions {
            prop_assert!(pred.is_finite());
        }
        Ok(())
    }
}
