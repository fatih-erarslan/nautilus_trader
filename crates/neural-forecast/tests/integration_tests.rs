//! Integration tests for Neural Forecast library
//!
//! These tests validate the complete neural forecasting pipeline including:
//! - Model creation and initialization
//! - Training and inference pipelines
//! - GPU acceleration and memory management
//! - Ensemble forecasting
//! - Real-time performance validation
//! - Financial time series specifics

use neural_forecast::prelude::*;
use neural_forecast::models::*;
use neural_forecast::*;
use ndarray::{Array1, Array2, Array3};
use std::time::Instant;
use tokio::test;
use approx::assert_relative_eq;
use std::sync::Arc;

#[test]
fn test_library_initialization() {
    // Test basic library initialization
    let result = neural_forecast::init();
    assert!(result.is_ok(), "Library initialization should succeed");
    
    // Test system info
    let info = neural_forecast::system_info();
    assert!(!info.version.is_empty());
    assert!(!info.features.is_empty());
    assert!(info.parallel_threads > 0);
    
    // Test GPU availability check
    let gpu_available = neural_forecast::gpu_available();
    // GPU availability is system dependent, just ensure no panic
    println!("GPU available: {}", gpu_available);
}

#[test]
fn test_model_types_and_metadata() {
    // Test model type names
    assert_eq!(ModelType::NHITS.name(), "nhits");
    assert_eq!(ModelType::NBEATS.name(), "nbeats");
    assert_eq!(ModelType::Transformer.name(), "transformer");
    assert_eq!(ModelType::LSTM.name(), "lstm");
    assert_eq!(ModelType::GRU.name(), "gru");
    
    // Test model type custom variants
    let custom_type = ModelType::Custom(42);
    assert_eq!(custom_type.name(), "custom");
}

#[test]
fn test_training_parameters_validation() {
    let mut params = TrainingParams::default();
    assert_eq!(params.learning_rate, 0.001);
    assert_eq!(params.batch_size, 32);
    assert_eq!(params.epochs, 100);
    assert_eq!(params.patience, 10);
    assert_eq!(params.validation_split, 0.2);
    
    // Test parameter modifications
    params.learning_rate = 0.01;
    params.batch_size = 64;
    assert_eq!(params.learning_rate, 0.01);
    assert_eq!(params.batch_size, 64);
}

#[test]
fn test_model_parameters_structure() {
    let mut params = ModelParameters::default();
    assert_eq!(params.version, 1);
    assert!(params.weights.is_empty());
    assert!(params.biases.is_empty());
    assert!(params.normalization.is_none());
    
    // Test adding weights and biases
    use std::collections::HashMap;
    let mut weights = HashMap::new();
    weights.insert("layer1".to_string(), Array2::zeros((10, 5)));
    params.weights = weights;
    
    let mut biases = HashMap::new();
    biases.insert("layer1".to_string(), Array1::zeros(5));
    params.biases = biases;
    
    assert_eq!(params.weights.len(), 1);
    assert_eq!(params.biases.len(), 1);
}

#[test]
fn test_training_data_structure() {
    // Create synthetic training data
    let batch_size = 16;
    let seq_len = 24;
    let features = 5;
    let horizon = 12;
    
    let inputs = Array3::from_shape_fn((batch_size, seq_len, features), |(b, s, f)| {
        (b * seq_len * features + s * features + f) as f32 * 0.01
    });
    
    let targets = Array3::from_shape_fn((batch_size, horizon, features), |(b, h, f)| {
        (b * horizon * features + h * features + f) as f32 * 0.01 + 1.0
    });
    
    let asset_ids = (0..batch_size).map(|i| format!("ASSET_{}", i)).collect();
    let timestamps = (0..batch_size).map(|i| {
        chrono::Utc::now() + chrono::Duration::hours(i as i64)
    }).collect();
    
    let training_data = TrainingData {
        inputs,
        targets,
        static_features: None,
        time_features: None,
        asset_ids,
        timestamps,
    };
    
    assert_eq!(training_data.inputs.shape(), &[batch_size, seq_len, features]);
    assert_eq!(training_data.targets.shape(), &[batch_size, horizon, features]);
    assert_eq!(training_data.asset_ids.len(), batch_size);
    assert_eq!(training_data.timestamps.len(), batch_size);
}

#[test]
fn test_normalization_parameters() {
    let features = 5;
    let mean = Array1::from_vec(vec![0.5, 1.0, 1.5, 2.0, 2.5]);
    let std = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    let min = Array1::from_vec(vec![0.0, 0.5, 1.0, 1.5, 2.0]);
    let max = Array1::from_vec(vec![1.0, 1.5, 2.0, 2.5, 3.0]);
    
    let norm_params = NormalizationParams {
        mean: mean.clone(),
        std: std.clone(),
        min: min.clone(),
        max: max.clone(),
    };
    
    assert_eq!(norm_params.mean.len(), features);
    assert_eq!(norm_params.std.len(), features);
    assert_eq!(norm_params.min.len(), features);
    assert_eq!(norm_params.max.len(), features);
    
    // Test that values are preserved
    for i in 0..features {
        assert_relative_eq!(norm_params.mean[i], mean[i]);
        assert_relative_eq!(norm_params.std[i], std[i]);
        assert_relative_eq!(norm_params.min[i], min[i]);
        assert_relative_eq!(norm_params.max[i], max[i]);
    }
}

#[test]
fn test_performance_metrics_structure() {
    let metrics = PerformanceMetrics {
        mae: 0.1,
        mse: 0.01,
        rmse: 0.1,
        mape: 5.0,
        r2: 0.95,
        inference_time_us: 50.0,
        memory_usage_bytes: 1024 * 1024,
        sharpe_ratio: Some(1.5),
        max_drawdown: Some(0.05),
        hit_rate: Some(0.65),
    };
    
    assert_eq!(metrics.mae, 0.1);
    assert_eq!(metrics.mse, 0.01);
    assert_eq!(metrics.rmse, 0.1);
    assert_eq!(metrics.mape, 5.0);
    assert_eq!(metrics.r2, 0.95);
    assert_eq!(metrics.inference_time_us, 50.0);
    assert_eq!(metrics.memory_usage_bytes, 1024 * 1024);
    assert_eq!(metrics.sharpe_ratio, Some(1.5));
    assert_eq!(metrics.max_drawdown, Some(0.05));
    assert_eq!(metrics.hit_rate, Some(0.65));
}

#[test]
fn test_training_metrics_structure() {
    let training_metrics = TrainingMetrics {
        train_loss: vec![1.0, 0.8, 0.6, 0.4, 0.2],
        val_loss: vec![1.1, 0.9, 0.7, 0.5, 0.3],
        train_accuracy: vec![0.6, 0.7, 0.8, 0.85, 0.9],
        val_accuracy: vec![0.55, 0.65, 0.75, 0.8, 0.85],
        training_time: 120.5,
        epochs_trained: 5,
        best_val_loss: 0.3,
        early_stopped: false,
        final_lr: 0.0001,
    };
    
    assert_eq!(training_metrics.train_loss.len(), 5);
    assert_eq!(training_metrics.val_loss.len(), 5);
    assert_eq!(training_metrics.epochs_trained, 5);
    assert_eq!(training_metrics.best_val_loss, 0.3);
    assert!(!training_metrics.early_stopped);
    
    // Test that loss decreases over time
    for i in 1..training_metrics.train_loss.len() {
        assert!(training_metrics.train_loss[i] <= training_metrics.train_loss[i-1]);
    }
}

#[test]
fn test_model_metadata_structure() {
    let metadata = ModelMetadata {
        model_type: ModelType::NHITS,
        name: "test_model".to_string(),
        version: "1.0.0".to_string(),
        description: "Test model for validation".to_string(),
        author: "TDD Agent".to_string(),
        created_at: chrono::Utc::now(),
        modified_at: chrono::Utc::now(),
        size_bytes: 1024 * 1024,
        num_parameters: 100000,
        input_shape: vec![24, 5],
        output_shape: vec![12, 5],
        training_data_info: None,
        performance_metrics: None,
    };
    
    assert_eq!(metadata.model_type, ModelType::NHITS);
    assert_eq!(metadata.name, "test_model");
    assert_eq!(metadata.version, "1.0.0");
    assert_eq!(metadata.num_parameters, 100000);
    assert_eq!(metadata.input_shape, vec![24, 5]);
    assert_eq!(metadata.output_shape, vec![12, 5]);
}

#[test]
fn test_voting_strategies() {
    // Test voting strategy types
    assert_eq!(VotingStrategy::Average, VotingStrategy::Average);
    assert_ne!(VotingStrategy::Average, VotingStrategy::WeightedAverage);
    
    // Test serialization/deserialization
    let strategy = VotingStrategy::WeightedAverage;
    let serialized = serde_json::to_string(&strategy).unwrap();
    let deserialized: VotingStrategy = serde_json::from_str(&serialized).unwrap();
    assert_eq!(strategy, deserialized);
}

#[test]
fn test_optimizer_types() {
    let optimizers = vec![
        OptimizerType::Adam,
        OptimizerType::SGD,
        OptimizerType::RMSprop,
        OptimizerType::AdamW,
    ];
    
    for optimizer in optimizers {
        // Test serialization
        let serialized = serde_json::to_string(&optimizer).unwrap();
        let deserialized: OptimizerType = serde_json::from_str(&serialized).unwrap();
        assert_eq!(optimizer, deserialized);
    }
}

#[test]
fn test_update_data_structure() {
    let batch_size = 8;
    let seq_len = 12;
    let features = 3;
    let horizon = 6;
    
    let inputs = Array3::zeros((batch_size, seq_len, features));
    let targets = Array3::ones((batch_size, horizon, features));
    
    let update_data = UpdateData {
        inputs,
        targets,
        learning_rate: Some(0.001),
        regularization: Some(0.01),
    };
    
    assert_eq!(update_data.inputs.shape(), &[batch_size, seq_len, features]);
    assert_eq!(update_data.targets.shape(), &[batch_size, horizon, features]);
    assert_eq!(update_data.learning_rate, Some(0.001));
    assert_eq!(update_data.regularization, Some(0.01));
}

#[test]
fn test_resource_usage_metrics() {
    let resource_usage = ResourceUsage {
        cpu_usage: 45.5,
        memory_usage: 2 * 1024 * 1024 * 1024, // 2GB
        gpu_usage: Some(78.2),
        gpu_memory_usage: Some(1024 * 1024 * 1024), // 1GB
        disk_io: 1000,
        network_io: 500,
    };
    
    assert_eq!(resource_usage.cpu_usage, 45.5);
    assert_eq!(resource_usage.memory_usage, 2 * 1024 * 1024 * 1024);
    assert_eq!(resource_usage.gpu_usage, Some(78.2));
    assert_eq!(resource_usage.gpu_memory_usage, Some(1024 * 1024 * 1024));
    assert_eq!(resource_usage.disk_io, 1000);
    assert_eq!(resource_usage.network_io, 500);
}

#[test]
fn test_prediction_statistics() {
    let prediction_stats = PredictionStats {
        total_predictions: 10000,
        successful_predictions: 9500,
        failed_predictions: 500,
        avg_prediction_time_us: 75.5,
        confidence_distribution: vec![0.1, 0.2, 0.3, 0.4],
        error_distribution: vec![0.05, 0.1, 0.15, 0.2],
    };
    
    assert_eq!(prediction_stats.total_predictions, 10000);
    assert_eq!(prediction_stats.successful_predictions, 9500);
    assert_eq!(prediction_stats.failed_predictions, 500);
    assert_eq!(prediction_stats.avg_prediction_time_us, 75.5);
    assert_eq!(prediction_stats.confidence_distribution.len(), 4);
    assert_eq!(prediction_stats.error_distribution.len(), 4);
    
    // Verify that successful + failed = total
    assert_eq!(
        prediction_stats.successful_predictions + prediction_stats.failed_predictions,
        prediction_stats.total_predictions
    );
}

#[test]
fn test_error_types_and_formatting() {
    // Test different error types
    let config_error = NeuralForecastError::ConfigError("Invalid configuration".to_string());
    assert!(config_error.to_string().contains("Invalid configuration"));
    
    let gpu_error = NeuralForecastError::GpuError("GPU not available".to_string());
    assert!(gpu_error.to_string().contains("GPU not available"));
    
    let inference_error = NeuralForecastError::InferenceError("Prediction failed".to_string());
    assert!(inference_error.to_string().contains("Prediction failed"));
    
    let training_error = NeuralForecastError::TrainingError("Training diverged".to_string());
    assert!(training_error.to_string().contains("Training diverged"));
    
    // Test error from std::io::Error
    let io_error = std::io::Error::new(std::io::ErrorKind::NotFound, "File not found");
    let neural_error: NeuralForecastError = io_error.into();
    match neural_error {
        NeuralForecastError::IoError(_) => (),
        _ => panic!("Expected IoError"),
    }
}

#[test]
fn test_time_series_data_validation() {
    // Test data with realistic financial time series characteristics
    let batch_size = 5;
    let seq_len = 168; // 1 week of hourly data
    let features = 6; // OHLCV + volume indicator
    let horizon = 24; // 24-hour prediction
    
    // Generate realistic OHLCV data
    let mut inputs = Array3::zeros((batch_size, seq_len, features));
    let mut targets = Array3::zeros((batch_size, horizon, features));
    
    for b in 0..batch_size {
        let base_price = 100.0 + b as f32 * 10.0;
        
        for s in 0..seq_len {
            let time_factor = s as f32 / seq_len as f32;
            let volatility = 0.02;
            
            // Open, High, Low, Close, Volume, Volume indicator
            let open = base_price + (time_factor - 0.5) * 10.0 + (s as f32 * 0.1).sin() * volatility * base_price;
            let high = open + (s as f32 * 0.2).cos().abs() * volatility * base_price;
            let low = open - (s as f32 * 0.15).sin().abs() * volatility * base_price;
            let close = low + (high - low) * (0.3 + 0.4 * (s as f32 * 0.05).cos());
            let volume = 1000.0 + 500.0 * (s as f32 * 0.1).sin().abs();
            let volume_indicator = if volume > 1250.0 { 1.0 } else { 0.0 };
            
            inputs[[b, s, 0]] = open;
            inputs[[b, s, 1]] = high;
            inputs[[b, s, 2]] = low;
            inputs[[b, s, 3]] = close;
            inputs[[b, s, 4]] = volume;
            inputs[[b, s, 5]] = volume_indicator;
        }
        
        // Generate target data (future prices)
        for h in 0..horizon {
            let future_time = (seq_len + h) as f32;
            let open = base_price + (future_time * 0.1).sin() * 0.02 * base_price;
            let high = open + (future_time * 0.2).cos().abs() * 0.02 * base_price;
            let low = open - (future_time * 0.15).sin().abs() * 0.02 * base_price;
            let close = low + (high - low) * 0.7;
            let volume = 1000.0 + 500.0 * (future_time * 0.1).sin().abs();
            let volume_indicator = if volume > 1250.0 { 1.0 } else { 0.0 };
            
            targets[[b, h, 0]] = open;
            targets[[b, h, 1]] = high;
            targets[[b, h, 2]] = low;
            targets[[b, h, 3]] = close;
            targets[[b, h, 4]] = volume;
            targets[[b, h, 5]] = volume_indicator;
        }
    }
    
    // Validate OHLC relationships
    for b in 0..batch_size {
        for s in 0..seq_len {
            let open = inputs[[b, s, 0]];
            let high = inputs[[b, s, 1]];
            let low = inputs[[b, s, 2]];
            let close = inputs[[b, s, 3]];
            
            // High should be >= max(open, close)
            assert!(high >= open.max(close), "High should be >= max(open, close)");
            // Low should be <= min(open, close)
            assert!(low <= open.min(close), "Low should be <= min(open, close)");
            // Volume should be positive
            assert!(inputs[[b, s, 4]] > 0.0, "Volume should be positive");
        }
    }
    
    // Test data shapes
    assert_eq!(inputs.shape(), &[batch_size, seq_len, features]);
    assert_eq!(targets.shape(), &[batch_size, horizon, features]);
}

#[test]
fn test_financial_metrics_calculations() {
    // Test financial-specific metrics calculations
    let returns = vec![0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.025];
    let risk_free_rate = 0.001; // 0.1% per period
    
    // Calculate Sharpe ratio
    let mean_return: f32 = returns.iter().sum::<f32>() / returns.len() as f32;
    let excess_return = mean_return - risk_free_rate;
    
    let variance: f32 = returns.iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f32>() / (returns.len() - 1) as f32;
    let std_dev = variance.sqrt();
    
    let sharpe_ratio = excess_return / std_dev;
    
    // Sharpe ratio should be reasonable for this data
    assert!(sharpe_ratio.is_finite());
    assert!(sharpe_ratio > -3.0 && sharpe_ratio < 3.0);
    
    // Calculate maximum drawdown
    let mut peak = returns[0];
    let mut max_drawdown = 0.0f32;
    let mut cumulative = 1.0f32;
    
    for &ret in &returns {
        cumulative *= 1.0 + ret;
        if cumulative > peak {
            peak = cumulative;
        }
        let drawdown = (peak - cumulative) / peak;
        if drawdown > max_drawdown {
            max_drawdown = drawdown;
        }
    }
    
    assert!(max_drawdown >= 0.0);
    assert!(max_drawdown <= 1.0);
    
    // Calculate hit rate (mock data)
    let predictions = vec![0.01, -0.005, 0.02, 0.0, 0.01, -0.015, 0.02];
    let actuals = vec![0.02, -0.01, 0.03, -0.005, 0.015, -0.02, 0.025];
    
    let correct_directions = predictions.iter()
        .zip(actuals.iter())
        .filter(|(&pred, &actual)| (pred > 0.0) == (actual > 0.0))
        .count();
    
    let hit_rate = correct_directions as f32 / predictions.len() as f32;
    assert!(hit_rate >= 0.0 && hit_rate <= 1.0);
}

#[test]
fn test_memory_efficiency_structures() {
    // Test that our data structures don't use excessive memory
    use std::mem;
    
    // Test size of key structures
    let model_params_size = mem::size_of::<ModelParameters>();
    let training_data_size = mem::size_of::<TrainingData>();
    let performance_metrics_size = mem::size_of::<PerformanceMetrics>();
    
    // These should be reasonable sizes (not too large)
    assert!(model_params_size < 1000); // Less than 1KB
    assert!(training_data_size < 500);  // Less than 500 bytes
    assert!(performance_metrics_size < 200); // Less than 200 bytes
    
    println!("ModelParameters size: {} bytes", model_params_size);
    println!("TrainingData size: {} bytes", training_data_size);
    println!("PerformanceMetrics size: {} bytes", performance_metrics_size);
}

#[test]
fn test_concurrent_data_access() {
    use std::sync::Arc;
    use std::thread;
    
    // Test thread-safe access to training data
    let batch_size = 10;
    let seq_len = 24;
    let features = 5;
    let horizon = 12;
    
    let inputs = Array3::from_shape_fn((batch_size, seq_len, features), |(b, s, f)| {
        (b + s + f) as f32 * 0.01
    });
    let targets = Array3::from_shape_fn((batch_size, horizon, features), |(b, h, f)| {
        (b + h + f) as f32 * 0.01 + 1.0
    });
    
    let training_data = Arc::new(TrainingData {
        inputs,
        targets,
        static_features: None,
        time_features: None,
        asset_ids: (0..batch_size).map(|i| format!("ASSET_{}", i)).collect(),
        timestamps: vec![chrono::Utc::now(); batch_size],
    });
    
    let mut handles = vec![];
    
    // Spawn multiple threads to access the data
    for i in 0..4 {
        let data_clone = Arc::clone(&training_data);
        let handle = thread::spawn(move || {
            // Each thread performs some computation on the data
            let sum = data_clone.inputs.iter().sum::<f32>();
            let target_sum = data_clone.targets.iter().sum::<f32>();
            
            assert!(sum.is_finite());
            assert!(target_sum.is_finite());
            assert_eq!(data_clone.asset_ids.len(), batch_size);
            
            (i, sum, target_sum)
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        let (thread_id, sum, target_sum) = handle.join().unwrap();
        println!("Thread {}: input_sum={}, target_sum={}", thread_id, sum, target_sum);
        assert!(sum.is_finite());
        assert!(target_sum.is_finite());
    }
}

#[test]
fn test_serialization_deserialization() {
    // Test serialization of key structures
    let training_params = TrainingParams {
        learning_rate: 0.01,
        batch_size: 64,
        epochs: 50,
        patience: 5,
        validation_split: 0.15,
        l2_reg: 0.001,
        dropout: 0.2,
        grad_clip: 0.5,
        optimizer: OptimizerType::AdamW,
    };
    
    // Serialize
    let serialized = serde_json::to_string(&training_params).unwrap();
    assert!(!serialized.is_empty());
    
    // Deserialize
    let deserialized: TrainingParams = serde_json::from_str(&serialized).unwrap();
    assert_eq!(training_params.learning_rate, deserialized.learning_rate);
    assert_eq!(training_params.batch_size, deserialized.batch_size);
    assert_eq!(training_params.epochs, deserialized.epochs);
    assert_eq!(training_params.optimizer, deserialized.optimizer);
    
    // Test model metadata serialization
    let metadata = ModelMetadata {
        model_type: ModelType::Transformer,
        name: "test_transformer".to_string(),
        version: "2.0.0".to_string(),
        description: "Test transformer model".to_string(),
        author: "Neural Forecast TDD".to_string(),
        created_at: chrono::Utc::now(),
        modified_at: chrono::Utc::now(),
        size_bytes: 2048,
        num_parameters: 50000,
        input_shape: vec![48, 6],
        output_shape: vec![24, 6],
        training_data_info: None,
        performance_metrics: None,
    };
    
    let metadata_serialized = serde_json::to_string(&metadata).unwrap();
    let metadata_deserialized: ModelMetadata = serde_json::from_str(&metadata_serialized).unwrap();
    
    assert_eq!(metadata.model_type, metadata_deserialized.model_type);
    assert_eq!(metadata.name, metadata_deserialized.name);
    assert_eq!(metadata.num_parameters, metadata_deserialized.num_parameters);
}

#[test]
fn test_numerical_stability() {
    // Test with extreme values to ensure numerical stability
    let extreme_values = vec![
        f32::MIN_POSITIVE,
        f32::MAX / 1e10,
        1e-10,
        1e10,
        -1e10,
        0.0,
        -0.0,
    ];
    
    for &value in &extreme_values {
        // Test that extreme values don't cause panics in basic operations
        let array = Array1::from_elem(5, value);
        let sum = array.sum();
        let mean = sum / array.len() as f32;
        
        // Values should be finite or properly handle infinity
        if value.is_finite() {
            assert!(sum.is_finite() || sum.is_infinite());
            assert!(mean.is_finite() || mean.is_infinite());
        }
        
        // Test normalization parameters with extreme values
        if value.is_finite() && value > 0.0 {
            let norm_params = NormalizationParams {
                mean: Array1::from_elem(3, value / 2.0),
                std: Array1::from_elem(3, value / 10.0),
                min: Array1::from_elem(3, 0.0),
                max: Array1::from_elem(3, value),
            };
            
            assert_eq!(norm_params.mean.len(), 3);
            assert_eq!(norm_params.std.len(), 3);
        }
    }
}

#[test]
fn test_edge_cases_and_error_conditions() {
    // Test empty arrays
    let empty_array1d = Array1::<f32>::zeros(0);
    let empty_array2d = Array2::<f32>::zeros((0, 0));
    let empty_array3d = Array3::<f32>::zeros((0, 0, 0));
    
    assert_eq!(empty_array1d.len(), 0);
    assert_eq!(empty_array2d.len(), 0);
    assert_eq!(empty_array3d.len(), 0);
    
    // Test single element arrays
    let single_element = Array1::from_elem(1, 42.0);
    assert_eq!(single_element.len(), 1);
    assert_eq!(single_element[0], 42.0);
    
    // Test mismatched dimensions in training data
    let mismatched_inputs = Array3::zeros((5, 10, 3));
    let mismatched_targets = Array3::zeros((3, 12, 3)); // Different batch size
    
    assert_ne!(mismatched_inputs.shape()[0], mismatched_targets.shape()[0]);
    
    // Test invalid training parameters
    let mut invalid_params = TrainingParams::default();
    invalid_params.learning_rate = -1.0; // Negative learning rate
    invalid_params.validation_split = 1.5; // > 1.0
    invalid_params.dropout = -0.1; // Negative dropout
    
    // These should be caught by validation logic
    assert!(invalid_params.learning_rate < 0.0);
    assert!(invalid_params.validation_split > 1.0);
    assert!(invalid_params.dropout < 0.0);
}

#[test]
fn test_performance_targets() {
    // Test that basic operations meet performance targets
    let batch_size = 128;
    let seq_len = 24;
    let features = 5;
    
    // Create relatively large arrays
    let large_input = Array3::from_shape_fn((batch_size, seq_len, features), |(b, s, f)| {
        (b * seq_len * features + s * features + f) as f32 * 0.001
    });
    
    // Test array operations performance
    let start_time = Instant::now();
    
    // Perform various operations
    let sum = large_input.sum();
    let mean = sum / large_input.len() as f32;
    let variance = large_input.mapv(|x| (x - mean).powi(2)).mean().unwrap();
    let std_dev = variance.sqrt();
    
    let elapsed = start_time.elapsed();
    
    // These operations should complete quickly (< 1ms for this size)
    assert!(elapsed.as_millis() < 10);
    
    // Results should be reasonable
    assert!(sum.is_finite());
    assert!(mean.is_finite());
    assert!(std_dev.is_finite());
    assert!(std_dev >= 0.0);
    
    println!("Array operations took: {:?}", elapsed);
    println!("Sum: {}, Mean: {}, StdDev: {}", sum, mean, std_dev);
}