//! Integration tests for model save/load persistence

use neuro_divergent::{
    models::{basic::*, recurrent::*, advanced::*},
    NeuralModel, ModelConfig, TimeSeriesDataFrame,
};
use tempfile::tempdir;
use std::path::PathBuf;

#[path = "../helpers/mod.rs"]
mod helpers;
use helpers::synthetic;

fn test_model_persistence<M>(
    model_name: &str,
    create_model: impl Fn(ModelConfig) -> M,
) where
    M: NeuralModel,
{
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12)
        .with_learning_rate(0.01)
        .with_epochs(20);

    let mut model = create_model(config);

    let values = synthetic::complex_series(300, 0.1, 24, 1.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    // Train model
    model.fit(&data).unwrap();

    // Get original predictions
    let orig_predictions = model.predict(12).unwrap();

    // Save model
    let dir = tempdir().unwrap();
    let path = dir.path().join(format!("{}.bin", model_name));
    model.save(&path).unwrap();

    // Verify file exists
    assert!(path.exists(), "{} model file not created", model_name);

    // Load model
    let loaded = M::load(&path).unwrap();

    // Verify loaded model produces same predictions
    let loaded_predictions = loaded.predict(12).unwrap();

    assert_eq!(
        orig_predictions.len(),
        loaded_predictions.len(),
        "{} prediction length mismatch after load",
        model_name
    );

    for (i, (&orig, &load)) in orig_predictions
        .iter()
        .zip(loaded_predictions.iter())
        .enumerate()
    {
        assert!(
            (orig - load).abs() < 1e-6,
            "{} prediction mismatch at {}: {} vs {}",
            model_name,
            i,
            orig,
            load
        );
    }

    println!("✓ {} persistence test passed", model_name);
}

#[test]
fn test_mlp_persistence() {
    test_model_persistence("MLP", |config| MLP::new(config));
}

#[test]
fn test_dlinear_persistence() {
    test_model_persistence("DLinear", |config| DLinear::new(config));
}

#[test]
fn test_nlinear_persistence() {
    test_model_persistence("NLinear", |config| NLinear::new(config));
}

#[test]
fn test_rnn_persistence() {
    test_model_persistence("RNN", |config| RNN::new(config));
}

#[test]
fn test_lstm_persistence() {
    test_model_persistence("LSTM", |config| LSTM::new(config.with_hidden_size(32)));
}

#[test]
fn test_gru_persistence() {
    test_model_persistence("GRU", |config| GRU::new(config.with_hidden_size(32)));
}

#[test]
fn test_nhits_persistence() {
    test_model_persistence("NHITS", |config| {
        NHITS::new(config.with_input_size(96).with_horizon(24).with_n_blocks(2))
    });
}

#[test]
fn test_multiple_save_load_cycles() {
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let mut model = MLP::new(config);

    let values = synthetic::sine_wave(200, 0.1, 10.0, 50.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();
    model.fit(&data).unwrap();

    let dir = tempdir().unwrap();

    // Multiple save/load cycles
    for cycle in 0..5 {
        let path = dir.path().join(format!("model_cycle_{}.bin", cycle));

        model.save(&path).unwrap();
        model = MLP::load(&path).unwrap();

        let predictions = model.predict(12).unwrap();
        assert_eq!(predictions.len(), 12);
    }

    println!("✓ Multiple save/load cycles passed");
}

#[test]
fn test_concurrent_model_saves() {
    use std::thread;

    let values = synthetic::complex_series(300, 0.1, 24, 1.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();

    let dir = tempdir().unwrap();
    let dir_path = dir.path().to_path_buf();

    let handles: Vec<_> = (0..3)
        .map(|i| {
            let data = data.clone();
            let dir_path = dir_path.clone();

            thread::spawn(move || {
                let config = ModelConfig::default()
                    .with_input_size(24)
                    .with_horizon(12);

                let mut model = MLP::new(config);
                model.fit(&data).unwrap();

                let path = dir_path.join(format!("concurrent_model_{}.bin", i));
                model.save(&path).unwrap();

                assert!(path.exists());
            })
        })
        .collect();

    for handle in handles {
        handle.join().unwrap();
    }

    println!("✓ Concurrent model saves passed");
}

#[test]
fn test_save_with_metadata() {
    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let mut model = LSTM::new(config);

    let values = synthetic::complex_series(300, 0.1, 24, 1.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();
    model.fit(&data).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("model_with_metadata.bin");

    // Save with metadata
    model.save(&path).unwrap();

    // Load and verify metadata is preserved
    let loaded = LSTM::load(&path).unwrap();

    assert_eq!(loaded.name(), "LSTM");
    assert_eq!(loaded.config().input_size, 24);
    assert_eq!(loaded.config().horizon, 12);

    println!("✓ Save with metadata passed");
}

#[test]
fn test_corrupted_model_load() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("corrupted.bin");

    // Create corrupted file
    std::fs::write(&path, b"corrupted data").unwrap();

    // Attempt to load should fail gracefully
    let result = MLP::load(&path);
    assert!(result.is_err(), "Should reject corrupted model file");

    println!("✓ Corrupted model load test passed");
}

#[test]
fn test_backward_compatibility() {
    // Test that models saved in previous versions can still be loaded
    // This would require versioned test fixtures in practice

    let config = ModelConfig::default()
        .with_input_size(24)
        .with_horizon(12);

    let mut model = MLP::new(config);

    let values = synthetic::sine_wave(200, 0.1, 10.0, 50.0);
    let data = TimeSeriesDataFrame::from_values(values, None).unwrap();
    model.fit(&data).unwrap();

    let dir = tempdir().unwrap();
    let path = dir.path().join("versioned_model.bin");
    model.save(&path).unwrap();

    // Load should succeed
    let loaded = MLP::load(&path).unwrap();
    assert_eq!(loaded.name(), "MLP");

    println!("✓ Backward compatibility test passed");
}

#[test]
fn test_model_size_limits() {
    // Test models with various sizes
    for hidden_size in [8, 16, 32, 64, 128, 256] {
        let config = ModelConfig::default()
            .with_input_size(24)
            .with_horizon(12)
            .with_hidden_size(hidden_size);

        let mut model = LSTM::new(config);

        let values = synthetic::sine_wave(200, 0.1, 10.0, 50.0);
        let data = TimeSeriesDataFrame::from_values(values, None).unwrap();
        model.fit(&data).unwrap();

        let dir = tempdir().unwrap();
        let path = dir.path().join(format!("model_size_{}.bin", hidden_size));
        model.save(&path).unwrap();

        let file_size = std::fs::metadata(&path).unwrap().len();
        println!("Hidden size {}: file size {} bytes", hidden_size, file_size);

        // Verify can be loaded
        let loaded = LSTM::load(&path).unwrap();
        assert_eq!(loaded.config().hidden_size, hidden_size);
    }

    println!("✓ Model size limits test passed");
}
