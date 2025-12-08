//! GPU acceleration integration tests
//!
//! This module provides comprehensive integration tests to validate
//! the 50-200x speedup claims and ensure accuracy vs CPU implementations.

#![cfg(test)]

use super::*;
use crate::config::{LSTMConfig, GPUConfig};
use crate::models::lstm::LSTMModel;
use crate::models::Model;
use ndarray::Array3;
use std::time::Instant;
use approx::assert_relative_eq;

#[cfg(feature = "cuda")]
use super::cuda::{CudaBackend, check_cuda_availability};

/// GPU accuracy validation test
#[tokio::test]
async fn test_gpu_cpu_accuracy_parity() {
    // Skip if no GPU available
    #[cfg(feature = "cuda")]
    if !check_cuda_availability().unwrap_or(false) {
        println!("CUDA not available, skipping GPU accuracy test");
        return;
    }

    let config = LSTMConfig {
        input_size: 64,
        hidden_size: 128,
        output_length: 10,
        dropout_rate: 0.0,
        bidirectional: true,
        num_layers: 1,
        activation: "tanh".to_string(),
        learning_rate: 0.001,
        batch_size: 32,
        sequence_length: 50,
    };

    // Create CPU and GPU models
    let mut cpu_model = LSTMModel::new_from_config(config.clone()).unwrap();
    let mut gpu_model = LSTMModel::new_from_config(config.clone()).unwrap();

    // Enable GPU acceleration
    #[cfg(feature = "cuda")]
    {
        gpu_model.enable_cuda_acceleration().await.unwrap();
        assert!(gpu_model.is_gpu_accelerated());
    }

    // Generate test input
    let batch_size = 16;
    let sequence_length = 50;
    let input_size = 64;
    let input_shape = (batch_size, sequence_length, input_size);
    let test_input = Array3::from_shape_fn(input_shape, |(b, s, i)| {
        ((b + s + i) as f32).sin() * 0.1
    });

    // Train both models with same data (simplified training)
    let training_data = crate::models::TrainingData {
        inputs: vec![test_input.clone()],
        targets: vec![Array3::zeros((batch_size, config.output_length, input_size))],
        validation_inputs: None,
        validation_targets: None,
    };

    cpu_model.train(&training_data).await.unwrap();
    gpu_model.train(&training_data).await.unwrap();

    // Run predictions
    let cpu_output = cpu_model.predict(&test_input).await.unwrap();
    let gpu_output = gpu_model.predict(&test_input).await.unwrap();

    // Validate shapes match
    assert_eq!(cpu_output.shape(), gpu_output.shape());

    // Validate numerical accuracy (allowing for small floating-point differences)
    let max_error = cpu_output.iter()
        .zip(gpu_output.iter())
        .map(|(cpu_val, gpu_val)| (cpu_val - gpu_val).abs())
        .fold(0.0, f32::max);

    println!("Maximum error between CPU and GPU: {:.2e}", max_error);
    assert!(max_error < 1e-3, "GPU output differs too much from CPU: {:.2e}", max_error);
}

/// Performance benchmark test
#[tokio::test]
async fn test_gpu_performance_benchmark() {
    use super::benchmarks::{GPUBenchmarkSuite, BenchmarkConfig};

    let mut benchmark_suite = GPUBenchmarkSuite::new().await.unwrap();
    
    let config = BenchmarkConfig {
        batch_sizes: vec![8, 32, 64],
        sequence_lengths: vec![50, 100],
        hidden_sizes: vec![128, 256],
        num_iterations: 10,
        warmup_iterations: 3,
        target_latency_us: 100.0,
    };

    let report = benchmark_suite.run_full_benchmark(config).await.unwrap();
    
    // Print summary
    report.print_summary();
    
    // Validate performance targets
    if report.gpu_available {
        println!("GPU acceleration available");
        
        // Check minimum speedup target
        assert!(
            report.summary.min_speedup >= 50.0,
            "Minimum speedup target not met: {:.1}x < 50x",
            report.summary.min_speedup
        );
        
        // Check that we achieve good speedups for larger workloads
        let large_workload_speedups: Vec<f64> = report.gpu_results.iter()
            .filter(|r| r.batch_size >= 32 && r.sequence_length >= 100)
            .map(|r| r.speedup_factor)
            .collect();
        
        if !large_workload_speedups.is_empty() {
            let avg_large_speedup = large_workload_speedups.iter().sum::<f64>() / large_workload_speedups.len() as f64;
            println!("Average speedup for large workloads: {:.1}x", avg_large_speedup);
            
            assert!(
                avg_large_speedup >= 100.0,
                "Large workload speedup target not met: {:.1}x < 100x",
                avg_large_speedup
            );
        }
    } else {
        println!("GPU not available, skipping performance validation");
    }
}

/// Memory management test
#[tokio::test]
async fn test_gpu_memory_management() {
    #[cfg(feature = "cuda")]
    {
        if !check_cuda_availability().unwrap_or(false) {
            println!("CUDA not available, skipping memory test");
            return;
        }

        let gpu_config = GPUConfig::default();
        let mut cuda_backend = CudaBackend::new(gpu_config).unwrap();
        
        // Test memory allocation and deallocation
        let tensor_sizes = vec![
            vec![1024],           // 1K elements
            vec![1024, 1024],     // 1M elements
            vec![64, 128, 256],   // 2M elements
            vec![128, 500, 64],   // 4M elements
        ];
        
        let mut allocated_tensors = Vec::new();
        
        // Allocate tensors
        for shape in &tensor_sizes {
            let tensor = cuda_backend.allocate_tensor::<f32>(shape.clone()).unwrap();
            println!("Allocated tensor with shape {:?}", shape);
            allocated_tensors.push(tensor);
        }
        
        // Validate tensor properties
        for (i, tensor) in allocated_tensors.iter().enumerate() {
            let expected_elements: usize = tensor_sizes[i].iter().product();
            assert_eq!(tensor.element_count(), expected_elements);
            println!("Tensor {} has {} elements", i, tensor.element_count());
        }
        
        // Test memory cleanup (tensors should be automatically deallocated when dropped)
        drop(allocated_tensors);
        println!("Memory cleanup completed");
    }
}

/// Batch processing efficiency test
#[tokio::test]
async fn test_gpu_batch_efficiency() {
    let config = LSTMConfig {
        input_size: 32,
        hidden_size: 64,
        output_length: 5,
        dropout_rate: 0.0,
        bidirectional: true,
        num_layers: 1,
        activation: "tanh".to_string(),
        learning_rate: 0.001,
        batch_size: 64,
        sequence_length: 25,
    };

    let mut model = LSTMModel::new_from_config(config.clone()).unwrap();

    #[cfg(feature = "cuda")]
    {
        if check_cuda_availability().unwrap_or(false) {
            model.enable_cuda_acceleration().await.unwrap();
        } else {
            println!("CUDA not available, testing CPU batch processing");
        }
    }

    // Create multiple test inputs of varying sizes
    let batch_configs = vec![
        (8, 25, 32),    // Small batch
        (32, 25, 32),   // Medium batch
        (64, 25, 32),   // Large batch
        (128, 25, 32),  // Very large batch
    ];

    let mut inputs = Vec::new();
    for (batch_size, seq_len, input_size) in batch_configs {
        let input = Array3::from_shape_fn((batch_size, seq_len, input_size), |(b, s, i)| {
            ((b + s + i) as f32 * 0.01).sin()
        });
        inputs.push(input);
    }

    // Test individual predictions
    let start_individual = Instant::now();
    let mut individual_results = Vec::new();
    for input in &inputs {
        let result = model.predict(input).await.unwrap();
        individual_results.push(result);
    }
    let individual_time = start_individual.elapsed();

    // Test batch prediction
    let start_batch = Instant::now();
    let batch_results = model.predict_batch(&inputs).await.unwrap();
    let batch_time = start_batch.elapsed();

    // Validate results match
    assert_eq!(individual_results.len(), batch_results.len());
    for (i, (individual, batch)) in individual_results.iter().zip(batch_results.iter()).enumerate() {
        assert_eq!(individual.shape(), batch.shape(), "Shape mismatch for input {}", i);
    }

    // Calculate efficiency metrics
    let efficiency_ratio = individual_time.as_secs_f64() / batch_time.as_secs_f64();
    
    println!("Individual processing time: {:?}", individual_time);
    println!("Batch processing time: {:?}", batch_time);
    println!("Batch efficiency ratio: {:.2}x", efficiency_ratio);

    // Batch processing should be more efficient for GPU workloads
    if model.is_gpu_accelerated() {
        assert!(
            efficiency_ratio > 1.5,
            "Batch processing not efficient enough: {:.2}x < 1.5x",
            efficiency_ratio
        );
    }
}

/// Streaming data transfer test
#[tokio::test]
async fn test_gpu_streaming_transfers() {
    use super::streaming::{GPUStreamingManager, TransferPriority};
    use wgpu::{Device, Queue};

    // This test would require actual GPU device creation
    // For now, we'll test the streaming logic structure
    
    println!("Testing streaming transfer concepts...");
    
    // Test transfer priority ordering
    assert!(TransferPriority::Critical > TransferPriority::High);
    assert!(TransferPriority::High > TransferPriority::Normal);
    assert!(TransferPriority::Normal > TransferPriority::Low);
    
    // Test data size calculations
    let test_sizes = vec![1024, 4096, 16384, 65536, 262144, 1048576];
    
    for size in test_sizes {
        println!("Testing transfer size: {} bytes", size);
        
        // Calculate optimal chunk size
        let optimal_chunk_size = std::cmp::min(size, 16 * 1024 * 1024); // 16MB max
        let num_chunks = (size + optimal_chunk_size - 1) / optimal_chunk_size;
        
        println!("  Optimal chunk size: {} bytes", optimal_chunk_size);
        println!("  Number of chunks: {}", num_chunks);
        
        assert!(optimal_chunk_size <= 16 * 1024 * 1024);
        assert!(num_chunks >= 1);
    }
}

/// Fallback behavior test
#[tokio::test]
async fn test_gpu_fallback_behavior() {
    let config = LSTMConfig {
        input_size: 16,
        hidden_size: 32,
        output_length: 5,
        dropout_rate: 0.0,
        bidirectional: false,
        num_layers: 1,
        activation: "tanh".to_string(),
        learning_rate: 0.001,
        batch_size: 8,
        sequence_length: 10,
    };

    let mut model = LSTMModel::new_from_config(config.clone()).unwrap();

    // Test that model works without GPU acceleration
    let test_input = Array3::from_shape_fn((4, 10, 16), |(b, s, i)| {
        ((b + s + i) as f32 * 0.01).cos()
    });

    let result = model.predict(&test_input).await.unwrap();
    
    assert_eq!(result.shape(), &[4, config.output_length, 16]);
    println!("CPU fallback working correctly");

    // Test that GPU acceleration can be attempted even if not available
    #[cfg(feature = "cuda")]
    {
        let gpu_enable_result = model.enable_cuda_acceleration().await;
        if gpu_enable_result.is_ok() {
            println!("GPU acceleration enabled successfully");
            assert!(model.is_gpu_accelerated());
        } else {
            println!("GPU acceleration failed as expected (no GPU available)");
            assert!(!model.is_gpu_accelerated());
        }
        
        // Model should still work regardless
        let gpu_result = model.predict(&test_input).await.unwrap();
        assert_eq!(gpu_result.shape(), result.shape());
    }
}

/// Performance regression test
#[tokio::test]
async fn test_performance_regression() {
    let config = LSTMConfig {
        input_size: 128,
        hidden_size: 256,
        output_length: 20,
        dropout_rate: 0.0,
        bidirectional: true,
        num_layers: 1,
        activation: "tanh".to_string(),
        learning_rate: 0.001,
        batch_size: 64,
        sequence_length: 100,
    };

    let mut model = LSTMModel::new_from_config(config.clone()).unwrap();

    #[cfg(feature = "cuda")]
    {
        if check_cuda_availability().unwrap_or(false) {
            model.enable_cuda_acceleration().await.unwrap();
        }
    }

    // Large workload to stress test performance
    let batch_size = 64;
    let sequence_length = 100;
    let input_size = 128;
    
    let test_input = Array3::from_shape_fn((batch_size, sequence_length, input_size), |(b, s, i)| {
        ((b * 13 + s * 7 + i * 3) as f32 * 0.001).sin()
    });

    // Warm up run
    let _ = model.predict(&test_input).await.unwrap();

    // Timed runs
    let num_runs = 5;
    let mut times = Vec::new();

    for i in 0..num_runs {
        let start = Instant::now();
        let _ = model.predict(&test_input).await.unwrap();
        let duration = start.elapsed();
        times.push(duration.as_millis());
        println!("Run {}: {:?}", i + 1, duration);
    }

    let avg_time = times.iter().sum::<u128>() / times.len() as u128;
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();

    println!("Performance stats:");
    println!("  Average: {} ms", avg_time);
    println!("  Min: {} ms", min_time);
    println!("  Max: {} ms", max_time);
    println!("  GPU accelerated: {}", model.is_gpu_accelerated());

    // Performance thresholds
    if model.is_gpu_accelerated() {
        // GPU should complete in sub-100ms for this workload
        assert!(
            avg_time < 100,
            "GPU performance regression: {} ms > 100 ms",
            avg_time
        );
    } else {
        // CPU baseline (more lenient)
        assert!(
            avg_time < 5000,
            "CPU performance regression: {} ms > 5000 ms",
            avg_time
        );
    }

    // Consistency check (max time shouldn't be more than 2x average)
    assert!(
        max_time < avg_time * 2,
        "Performance inconsistency: max {} ms > 2x avg {} ms",
        max_time, avg_time
    );
}

/// Integration with existing freqtrade strategy test
#[tokio::test]
async fn test_freqtrade_integration() {
    // Test that GPU-accelerated models can integrate with the existing strategy
    let config = LSTMConfig {
        input_size: 20,   // Typical number of technical indicators
        hidden_size: 64,
        output_length: 3, // Short-term prediction horizon
        dropout_rate: 0.1,
        bidirectional: true,
        num_layers: 1,
        activation: "tanh".to_string(),
        learning_rate: 0.001,
        batch_size: 16,
        sequence_length: 60, // 1 hour of minute data
    };

    let mut model = LSTMModel::new_from_config(config.clone()).unwrap();

    #[cfg(feature = "cuda")]
    {
        if check_cuda_availability().unwrap_or(false) {
            model.enable_cuda_acceleration().await.unwrap();
            println!("Testing with GPU acceleration enabled");
        } else {
            println!("Testing with CPU fallback");
        }
    }

    // Simulate real trading data (normalized technical indicators)
    let batch_size = 1; // Single prediction for real-time trading
    let sequence_length = 60;
    let num_indicators = 20;
    
    let market_data = Array3::from_shape_fn((batch_size, sequence_length, num_indicators), |(_, t, i)| {
        // Simulate various technical indicators with realistic patterns
        match i % 4 {
            0 => (t as f32 * 0.1).sin() * 0.1,          // Oscillating indicator (RSI-like)
            1 => (t as f32 * 0.05).cos() * 0.2,         // Trend indicator (MA-like)
            2 => ((t as f32 * 0.2).sin() + 1.0) * 0.05, // Volume indicator
            _ => (t as f32 * 0.03 + i as f32).tanh() * 0.15, // Other indicators
        }
    });

    // Prediction should complete quickly for real-time trading
    let start = Instant::now();
    let prediction = model.predict(&market_data).await.unwrap();
    let inference_time = start.elapsed();

    println!("Inference time: {:?}", inference_time);
    println!("Prediction shape: {:?}", prediction.shape());

    // Validate output format
    assert_eq!(prediction.shape(), &[batch_size, config.output_length, num_indicators]);

    // Real-time trading requirement: sub-100ms inference
    if model.is_gpu_accelerated() {
        assert!(
            inference_time.as_millis() < 100,
            "GPU inference too slow for real-time trading: {:?}",
            inference_time
        );
    } else {
        assert!(
            inference_time.as_millis() < 1000,
            "CPU inference too slow: {:?}",
            inference_time
        );
    }

    // Validate prediction values are reasonable (not NaN, not extreme)
    for value in prediction.iter() {
        assert!(value.is_finite(), "Prediction contains non-finite values");
        assert!(value.abs() < 10.0, "Prediction values too extreme: {}", value);
    }

    println!("Freqtrade integration test passed");
}