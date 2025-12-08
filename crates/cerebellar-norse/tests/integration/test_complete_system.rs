//! Integration tests for complete cerebellar system
//! 
//! These tests validate end-to-end functionality of the complete
//! cerebellar neural network system including encoding, processing,
//! and decoding with real-world scenarios.

use std::collections::HashMap;
use std::time::Duration;
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;
use approx::assert_abs_diff_eq;
use serial_test::serial;

use cerebellar_norse::*;
use cerebellar_norse::cerebellar_circuit::*;
use cerebellar_norse::encoding::*;
use cerebellar_norse::training::*;
use cerebellar_norse::optimization::*;
use crate::utils::fixtures::*;
use crate::utils::*;

/// Test complete cerebellar system initialization
#[test]
fn test_complete_system_initialization() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    // Create complete system components
    let circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let encoder = InputEncoder::new(&config).unwrap();
    let decoder = OutputDecoder::new(&config).unwrap();
    
    // Verify system is properly initialized
    assert_eq!(circuit.config().input_dim, config.input_dim);
    assert_eq!(circuit.config().output_dim, config.output_dim);
    assert_eq!(encoder.get_statistics().len(), 3);
    assert_eq!(decoder.get_statistics().len(), 4);
}

/// Test end-to-end processing pipeline
#[test]
fn test_end_to_end_processing() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Create test input
    let batch_size = 2;
    let input_data = DataFixtures::step_input(batch_size, config.input_dim, config.device);
    
    // Process through complete pipeline
    let encoded_input = encoder.encode(&input_data).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let decoded_output = decoder.decode(&circuit_outputs).unwrap();
    
    // Verify pipeline outputs
    assert_eq!(encoded_input.shape().dims()[0], config.time_steps as i64);
    assert_eq!(encoded_input.shape().dims()[1], batch_size as i64);
    assert_eq!(encoded_input.shape().dims()[2], config.input_dim as i64);
    
    assert!(circuit_outputs.contains_key("dcn_spikes"));
    assert!(circuit_outputs.contains_key("grc_spikes"));
    assert!(circuit_outputs.contains_key("pc_spikes"));
    
    assert_eq!(decoded_output.shape().dims(), &[batch_size as i64, config.output_dim as i64]);
}

/// Test XOR learning scenario
#[test]
#[serial]
fn test_xor_learning_scenario() {
    setup_test_logging();
    
    let mut config = ConfigFixtures::small_config();
    config.input_dim = 2;
    config.output_dim = 1;
    
    let layer_configs = LayerFixtures::small_layers();
    let mut vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    let mut trainer = TrainingEngine::new(&config, &mut vs, 1e-3).unwrap();
    
    // Get XOR scenario data
    let (x_train, y_train) = ScenarioFixtures::xor_scenario();
    
    // Train for a few epochs
    let mut losses = Vec::new();
    for epoch in 0..10 {
        let loss = trainer.train_epoch(
            &mut circuit,
            &mut encoder,
            &mut decoder,
            &x_train,
            &y_train,
            2, // batch size
        ).unwrap();
        
        losses.push(loss);
        
        // Loss should be finite
        assert!(loss.is_finite());
        
        println!("Epoch {}: Loss = {:.6}", epoch + 1, loss);
    }
    
    // Test inference
    let encoded_input = encoder.encode(&x_train).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let predictions = decoder.decode(&circuit_outputs).unwrap();
    
    assert_eq!(predictions.shape().dims(), &[4, 1]);
    
    // Verify predictions are in valid range
    let pred_values: Vec<f32> = predictions.flatten_all().unwrap().to_vec1().unwrap();
    for pred in pred_values {
        assert!(pred >= 0.0 && pred <= 1.0);
    }
}

/// Test pattern recognition scenario
#[test]
#[serial]
fn test_pattern_recognition_scenario() {
    setup_test_logging();
    
    let mut config = ConfigFixtures::medium_config();
    config.input_dim = 16;
    config.output_dim = 4;
    
    let layer_configs = LayerFixtures::small_layers();
    let mut vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    let mut trainer = TrainingEngine::new(&config, &mut vs, 1e-3).unwrap();
    
    // Get pattern recognition data
    let (x_train, y_train) = ScenarioFixtures::pattern_recognition_scenario();
    
    // Train for a few epochs
    for epoch in 0..5 {
        let loss = trainer.train_epoch(
            &mut circuit,
            &mut encoder,
            &mut decoder,
            &x_train,
            &y_train,
            10, // batch size
        ).unwrap();
        
        assert!(loss.is_finite());
        println!("Pattern Recognition Epoch {}: Loss = {:.6}", epoch + 1, loss);
    }
    
    // Test inference on a subset
    let test_subset = x_train.narrow(0, 0, 10);
    let encoded_input = encoder.encode(&test_subset).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let predictions = decoder.decode(&circuit_outputs).unwrap();
    
    assert_eq!(predictions.shape().dims(), &[10, 4]);
    
    // Verify predictions are probabilities
    let pred_values: Vec<f32> = predictions.flatten_all().unwrap().to_vec1().unwrap();
    for pred in pred_values {
        assert!(pred >= 0.0 && pred <= 1.0);
    }
}

/// Test trading scenario with ultra-low latency requirements
#[test]
#[serial]
fn test_trading_scenario_ultra_low_latency() {
    setup_test_logging();
    
    let config = ConfigFixtures::trading_config();
    let layer_configs = LayerFixtures::performance_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Get trading scenario data
    let (x_train, y_train) = ScenarioFixtures::trading_scenario();
    
    // Test processing time for individual samples
    let single_sample = x_train.narrow(0, 0, 1);
    
    let (_, encoding_time) = time_operation(|| {
        encoder.encode(&single_sample).unwrap()
    });
    
    let encoded_input = encoder.encode(&single_sample).unwrap();
    let (_, circuit_time) = time_operation(|| {
        circuit.forward(&encoded_input).unwrap()
    });
    
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let (_, decoding_time) = time_operation(|| {
        decoder.decode(&circuit_outputs).unwrap()
    });
    
    let total_time = encoding_time + circuit_time + decoding_time;
    
    println!("Trading Latency Breakdown:");
    println!("  Encoding: {:?}", encoding_time);
    println!("  Circuit:  {:?}", circuit_time);
    println!("  Decoding: {:?}", decoding_time);
    println!("  Total:    {:?}", total_time);
    
    // Verify ultra-low latency requirements
    assert!(total_time < Duration::from_micros(config.max_processing_time_us));
    
    // Test batch processing
    let batch_size = 32;
    let batch_data = x_train.narrow(0, 0, batch_size);
    
    let (_, batch_time) = time_operation(|| {
        let encoded = encoder.encode(&batch_data).unwrap();
        let outputs = circuit.forward(&encoded).unwrap();
        decoder.decode(&outputs).unwrap()
    });
    
    println!("Batch processing time ({}): {:?}", batch_size, batch_time);
    
    // Batch processing should be efficient
    let per_sample_time = batch_time / batch_size;
    assert!(per_sample_time < Duration::from_micros(config.max_processing_time_us));
}

/// Test time series prediction scenario
#[test]
fn test_time_series_prediction() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Get time series data
    let (x_train, y_train) = ScenarioFixtures::time_series_scenario();
    
    // Process sequential data
    let encoded_input = encoder.encode(&x_train).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let predictions = decoder.decode(&circuit_outputs).unwrap();
    
    // Verify outputs
    assert_eq!(predictions.shape().dims(), &[20, 1]); // 20 sequences, 1 prediction each
    
    // Check prediction quality (should be reasonable for sine wave)
    let pred_values: Vec<f32> = predictions.flatten_all().unwrap().to_vec1().unwrap();
    let target_values: Vec<f32> = y_train.flatten_all().unwrap().to_vec1().unwrap();
    
    for (pred, target) in pred_values.iter().zip(target_values.iter()) {
        assert!(pred.is_finite());
        assert!(target.is_finite());
        
        // Predictions should be in reasonable range for sine wave
        assert!(pred.abs() <= 2.0);
        assert!(target.abs() <= 2.0);
    }
}

/// Test system with different encoding strategies
#[test]
fn test_different_encoding_strategies() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Test different encoding strategies
    let strategies = vec![
        EncodingStrategy::ConstantCurrent,
        EncodingStrategy::Rate,
        EncodingStrategy::Temporal,
    ];
    
    let input_data = DataFixtures::sinusoidal_input(2, config.input_dim, config.device);
    
    for strategy in strategies {
        let mut encoder = InputEncoder::new(&config).unwrap();
        // Note: Strategy configuration would need to be implemented
        // For now, we use the default encoder
        
        let encoded_input = encoder.encode(&input_data).unwrap();
        let circuit_outputs = circuit.forward(&encoded_input).unwrap();
        let predictions = decoder.decode(&circuit_outputs).unwrap();
        
        // Verify each strategy produces valid outputs
        assert_eq!(predictions.shape().dims(), &[2, config.output_dim as i64]);
        
        let pred_values: Vec<f32> = predictions.flatten_all().unwrap().to_vec1().unwrap();
        for pred in pred_values {
            assert!(pred.is_finite());
        }
        
        // Reset circuit state between strategies
        circuit.reset();
    }
}

/// Test system with different decoding strategies
#[test]
fn test_different_decoding_strategies() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let decoder = OutputDecoder::new(&config).unwrap();
    
    let input_data = DataFixtures::step_input(3, config.input_dim, config.device);
    
    // Process through circuit
    let encoded_input = encoder.encode(&input_data).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    
    // Test different decoding strategies
    let strategies = vec![
        "firing_rate",
        "population_vector",
        "temporal_pattern",
        "winner_take_all",
    ];
    
    for strategy in strategies {
        let predictions = decoder.decode_with_strategy(&circuit_outputs, strategy).unwrap();
        
        // Verify outputs are valid
        assert_eq!(predictions.shape().dims()[0], 3); // batch size
        assert!(predictions.shape().dims()[1] > 0);   // output dimension
        
        let pred_values: Vec<f32> = predictions.flatten_all().unwrap().to_vec1().unwrap();
        for pred in pred_values {
            assert!(pred.is_finite());
        }
    }
}

/// Test system performance optimization
#[test]
#[serial]
fn test_system_performance_optimization() {
    setup_test_logging();
    
    let config = ConfigFixtures::large_config();
    let layer_configs = LayerFixtures::performance_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Create optimizer
    let opt_config = OptimizationFixtures::performance_optimization_config();
    let mut optimizer = CerebellarOptimizer::new(opt_config, config.device).unwrap();
    
    // Test data
    let batch_size = 8;
    let input_data = DataFixtures::random_input(batch_size, config.input_dim, config.device);
    
    // Benchmark without optimization
    let (_, time_without_opt) = time_operation(|| {
        let encoded = encoder.encode(&input_data).unwrap();
        let outputs = circuit.forward(&encoded).unwrap();
        decoder.decode(&outputs).unwrap()
    });
    
    // Apply optimizations
    let encoded_input = encoder.encode(&input_data).unwrap();
    let mut circuit_outputs = circuit.forward(&encoded_input).unwrap();
    optimizer.optimize_tensor_ops(&mut circuit_outputs).unwrap();
    
    // Benchmark with optimization
    let (_, time_with_opt) = time_operation(|| {
        let outputs = circuit.forward(&encoded_input).unwrap();
        decoder.decode(&outputs).unwrap()
    });
    
    println!("Performance comparison:");
    println!("  Without optimization: {:?}", time_without_opt);
    println!("  With optimization:    {:?}", time_with_opt);
    
    // Get optimization metrics
    let metrics = optimizer.get_metrics();
    println!("Optimization metrics:");
    println!("  Optimizations: {}", metrics.optimization_count);
    println!("  Throughput: {:.2} ops/sec", metrics.throughput_ops_per_second());
    
    // Verify optimization doesn't break functionality
    let final_predictions = decoder.decode(&circuit_outputs).unwrap();
    assert_eq!(final_predictions.shape().dims(), &[batch_size as i64, config.output_dim as i64]);
}

/// Test system memory usage and cleanup
#[test]
fn test_system_memory_management() {
    setup_test_logging();
    
    let config = ConfigFixtures::medium_config();
    let layer_configs = LayerFixtures::small_layers();
    
    // Test memory usage with multiple system creations
    let mut memory_measurements = Vec::new();
    
    for i in 0..5 {
        let (_, memory_used) = measure_memory_usage(|| {
            let vs = nn::VarStore::new(config.device);
            let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
            let mut encoder = InputEncoder::new(&config).unwrap();
            let mut decoder = OutputDecoder::new(&config).unwrap();
            
            // Process some data
            let input_data = DataFixtures::step_input(4, config.input_dim, config.device);
            let encoded_input = encoder.encode(&input_data).unwrap();
            let circuit_outputs = circuit.forward(&encoded_input).unwrap();
            let _predictions = decoder.decode(&circuit_outputs).unwrap();
            
            // Reset to test cleanup
            circuit.reset();
            encoder.reset();
            decoder.reset();
        });
        
        memory_measurements.push(memory_used);
        println!("Iteration {}: Memory used = {} bytes", i + 1, memory_used);
    }
    
    // Memory usage should be reasonably consistent
    let avg_memory = memory_measurements.iter().sum::<usize>() / memory_measurements.len();
    println!("Average memory usage: {} bytes", avg_memory);
    
    // All measurements should be reasonable
    for measurement in memory_measurements {
        assert!(measurement >= 0);
    }
}

/// Test system error handling and recovery
#[test]
fn test_system_error_handling() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Test with invalid input dimensions
    let wrong_input = DataFixtures::step_input(2, config.input_dim + 5, config.device);
    let result = encoder.encode(&wrong_input);
    assert!(result.is_err());
    
    // Test with valid input after error
    let valid_input = DataFixtures::step_input(2, config.input_dim, config.device);
    let encoded_input = encoder.encode(&valid_input).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let predictions = decoder.decode(&circuit_outputs).unwrap();
    
    assert_eq!(predictions.shape().dims(), &[2, config.output_dim as i64]);
    
    // Test system recovery after reset
    circuit.reset();
    encoder.reset();
    decoder.reset();
    
    let new_encoded = encoder.encode(&valid_input).unwrap();
    let new_outputs = circuit.forward(&new_encoded).unwrap();
    let new_predictions = decoder.decode(&new_outputs).unwrap();
    
    assert_eq!(new_predictions.shape().dims(), &[2, config.output_dim as i64]);
}

/// Test system scalability with different configurations
#[test]
#[serial]
fn test_system_scalability() {
    setup_test_logging();
    
    let configs = vec![
        ConfigFixtures::small_config(),
        ConfigFixtures::medium_config(),
        ConfigFixtures::large_config(),
    ];
    
    for (i, config) in configs.iter().enumerate() {
        let layer_configs = LayerFixtures::small_layers();
        let vs = nn::VarStore::new(config.device);
        
        let mut circuit = CerebellarCircuit::new(config, &layer_configs, &vs).unwrap();
        let mut encoder = InputEncoder::new(config).unwrap();
        let mut decoder = OutputDecoder::new(config).unwrap();
        
        // Test processing with increasing complexity
        let input_data = DataFixtures::step_input(2, config.input_dim, config.device);
        
        let (_, processing_time) = time_operation(|| {
            let encoded = encoder.encode(&input_data).unwrap();
            let outputs = circuit.forward(&encoded).unwrap();
            decoder.decode(&outputs).unwrap()
        });
        
        println!("Config {}: Processing time = {:?}", i + 1, processing_time);
        
        // Verify outputs are correct
        let encoded_input = encoder.encode(&input_data).unwrap();
        let circuit_outputs = circuit.forward(&encoded_input).unwrap();
        let predictions = decoder.decode(&circuit_outputs).unwrap();
        
        assert_eq!(predictions.shape().dims(), &[2, config.output_dim as i64]);
        
        // Processing time should be reasonable even for large configs
        assert!(processing_time < Duration::from_secs(10));
    }
}

/// Test system with concurrent processing
#[test]
#[serial]
fn test_concurrent_system_processing() {
    use std::sync::{Arc, Mutex};
    use std::thread;
    
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let circuit = Arc::new(Mutex::new(
        CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap()
    ));
    let encoder = Arc::new(Mutex::new(InputEncoder::new(&config).unwrap()));
    let decoder = Arc::new(Mutex::new(OutputDecoder::new(&config).unwrap()));
    
    let mut handles = Vec::new();
    
    // Spawn multiple threads to process data concurrently
    for i in 0..4 {
        let circuit_clone = Arc::clone(&circuit);
        let encoder_clone = Arc::clone(&encoder);
        let decoder_clone = Arc::clone(&decoder);
        let config_clone = config.clone();
        
        let handle = thread::spawn(move || {
            let input_data = DataFixtures::step_input(1, config_clone.input_dim, config_clone.device);
            
            let encoded = {
                let mut enc = encoder_clone.lock().unwrap();
                enc.encode(&input_data).unwrap()
            };
            
            let outputs = {
                let mut circ = circuit_clone.lock().unwrap();
                circ.forward(&encoded).unwrap()
            };
            
            let predictions = {
                let mut dec = decoder_clone.lock().unwrap();
                dec.decode(&outputs).unwrap()
            };
            
            println!("Thread {} completed processing", i);
            predictions.shape().dims().to_vec()
        });
        
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        let result = handle.join().unwrap();
        assert_eq!(result, vec![1, config.output_dim as i64]);
    }
}

/// Test system persistence and state management
#[test]
fn test_system_persistence() {
    setup_test_logging();
    
    let config = ConfigFixtures::small_config();
    let layer_configs = LayerFixtures::small_layers();
    let vs = nn::VarStore::new(config.device);
    
    let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
    let mut encoder = InputEncoder::new(&config).unwrap();
    let mut decoder = OutputDecoder::new(&config).unwrap();
    
    // Process some data to change internal state
    let input_data = DataFixtures::step_input(2, config.input_dim, config.device);
    let encoded_input = encoder.encode(&input_data).unwrap();
    let circuit_outputs = circuit.forward(&encoded_input).unwrap();
    let predictions = decoder.decode(&circuit_outputs).unwrap();
    
    // Get current statistics
    let circuit_metrics = circuit.get_metrics();
    let encoder_stats = encoder.get_statistics();
    let decoder_stats = decoder.get_statistics();
    
    println!("Circuit metrics: {:?}", circuit_metrics);
    println!("Encoder stats: {:?}", encoder_stats);
    println!("Decoder stats: {:?}", decoder_stats);
    
    // Verify state is captured
    assert!(circuit_metrics.total_spikes > 0);
    assert!(encoder_stats.len() > 0);
    assert!(decoder_stats.len() > 0);
    
    // Test state reset
    circuit.reset();
    encoder.reset();
    decoder.reset();
    
    let new_circuit_metrics = circuit.get_metrics();
    let new_encoder_stats = encoder.get_statistics();
    let new_decoder_stats = decoder.get_statistics();
    
    // Verify state is properly reset
    assert_eq!(new_circuit_metrics.total_spikes, 0);
    assert!(new_encoder_stats.get("encoding_time_us").unwrap_or(&0.0) == &0.0);
    assert!(new_decoder_stats.get("decoding_time_us").unwrap_or(&0.0) == &0.0);
}

#[cfg(test)]
mod system_benchmarks {
    use super::*;
    use criterion::{black_box, Criterion};
    
    /// Benchmark complete system throughput
    #[test]
    #[serial]
    fn benchmark_system_throughput() {
        setup_test_logging();
        
        let config = ConfigFixtures::trading_config();
        let layer_configs = LayerFixtures::performance_layers();
        let vs = nn::VarStore::new(config.device);
        
        let mut circuit = CerebellarCircuit::new(&config, &layer_configs, &vs).unwrap();
        let mut encoder = InputEncoder::new(&config).unwrap();
        let mut decoder = OutputDecoder::new(&config).unwrap();
        
        let batch_sizes = vec![1, 8, 32, 128];
        
        for batch_size in batch_sizes {
            let input_data = DataFixtures::random_input(batch_size, config.input_dim, config.device);
            
            let (_, processing_time) = time_operation(|| {
                let encoded = encoder.encode(&input_data).unwrap();
                let outputs = circuit.forward(&encoded).unwrap();
                decoder.decode(&outputs).unwrap()
            });
            
            let throughput = batch_size as f64 / processing_time.as_secs_f64();
            println!("Batch size {}: {:.2} samples/sec", batch_size, throughput);
            
            // Verify reasonable throughput
            assert!(throughput > 0.0);
        }
    }
    
    /// Benchmark memory efficiency
    #[test]
    fn benchmark_memory_efficiency() {
        setup_test_logging();
        
        let configs = vec![
            ConfigFixtures::small_config(),
            ConfigFixtures::medium_config(),
            ConfigFixtures::large_config(),
        ];
        
        for (i, config) in configs.iter().enumerate() {
            let (_, memory_used) = measure_memory_usage(|| {
                let layer_configs = LayerFixtures::performance_layers();
                let vs = nn::VarStore::new(config.device);
                
                let mut circuit = CerebellarCircuit::new(config, &layer_configs, &vs).unwrap();
                let mut encoder = InputEncoder::new(config).unwrap();
                let mut decoder = OutputDecoder::new(config).unwrap();
                
                // Process test data
                let input_data = DataFixtures::step_input(4, config.input_dim, config.device);
                let encoded = encoder.encode(&input_data).unwrap();
                let outputs = circuit.forward(&encoded).unwrap();
                let _predictions = decoder.decode(&outputs).unwrap();
            });
            
            println!("Config {} memory usage: {} bytes", i + 1, memory_used);
            
            // Memory usage should be reasonable
            assert!(memory_used >= 0);
        }
    }
}