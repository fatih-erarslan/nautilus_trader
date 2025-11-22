//! WebAssembly Neural Network Demo
//! 
//! Demonstrates the WASM-optimized neural network implementation with:
//! - SIMD128 acceleration when available
//! - Quantization for memory efficiency  
//! - Multiple activation functions
//! - Streaming inference capability
//! - Browser and Node.js compatibility

use cwts_ultra::neural::wasm_nn::*;

fn main() {
    println!("🧠 CWTS Ultra - WebAssembly Neural Network Demo");
    println!("===============================================\n");

    // Demo 1: Basic Neural Layer
    demo_basic_layer();
    
    // Demo 2: Multi-Layer Network
    demo_multi_layer_network();
    
    // Demo 3: Quantization
    demo_quantization();
    
    // Demo 4: Streaming Processing
    demo_streaming();
    
    // Demo 5: Performance Comparison
    demo_performance();
    
    // Demo 6: Classification Network
    demo_classification();

    println!("\n🎉 Demo completed successfully!");
}

fn demo_basic_layer() {
    println!("📋 Demo 1: Basic Neural Layer");
    println!("------------------------------");
    
    let layer = WasmNeuralLayer::new(4, 3);
    println!("{}", layer.get_info());
    println!("Memory usage: {} bytes", layer.memory_usage());
    
    let input = vec![0.5, -0.3, 0.8, -0.2];
    println!("Input: {:?}", input);
    
    // Test different activation functions
    let activations = [
        ("ReLU", 0),
        ("Sigmoid", 1), 
        ("Tanh", 2),
        ("LeakyReLU", 3),
        ("Swish", 4),
        ("GELU", 5),
    ];
    
    for (name, activation) in &activations {
        let output = layer.forward(&input, *activation);
        println!("  {}: {:?}", name, output);
    }
    
    println!();
}

fn demo_multi_layer_network() {
    println!("📋 Demo 2: Multi-Layer Network");
    println!("------------------------------");
    
    let mut network = WasmNeuralNetwork::new();
    network.add_layer(8, 12); // Expand
    network.add_layer(12, 8); // Same size
    network.add_layer(8, 4);  // Contract
    network.add_layer(4, 2);  // Output
    
    println!("{}", network.get_stats());
    
    let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let activations = vec![0, 0, 0, 1]; // ReLU hidden, Sigmoid output
    
    let output = network.predict(&input, &activations);
    println!("Input (8): {:?}", input);
    println!("Output (2): {:?}", output);
    println!("Output range: [{:.4}, {:.4}]", 
             output.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
             output.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)));
    
    println!();
}

fn demo_quantization() {
    println!("📋 Demo 3: Quantization");
    println!("------------------------");
    
    let mut layer = WasmNeuralLayer::new(16, 8);
    let input: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
    
    // Original inference
    let original_output = layer.forward(&input, 0);
    let original_memory = layer.memory_usage();
    
    println!("Original model:");
    println!("  Memory: {} bytes", original_memory);
    println!("  Output: {:?}", &original_output[..4]); // Show first 4 values
    
    // Quantize to INT8
    layer.quantize(1, true); // INT8 with dynamic range
    let quantized_output = layer.forward(&input, 0);
    let quantized_memory = layer.memory_usage();
    
    println!("Quantized model (INT8):");
    println!("  Memory: {} bytes ({:.1}% of original)", 
             quantized_memory, 
             quantized_memory as f64 / original_memory as f64 * 100.0);
    println!("  Output: {:?}", &quantized_output[..4]);
    
    // Calculate accuracy loss
    let mse: f32 = original_output.iter()
        .zip(quantized_output.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>() / original_output.len() as f32;
    
    println!("  MSE vs original: {:.6}", mse);
    println!("  RMSE: {:.6}", mse.sqrt());
    
    println!();
}

fn demo_streaming() {
    println!("📋 Demo 4: Streaming Processing");
    println!("--------------------------------");
    
    let mut streaming = WasmStreamingNN::new(8); // Buffer size: 8
    streaming.add_layer(8, 6, 0);  // ReLU
    streaming.add_layer(6, 4, 0);  // ReLU
    streaming.add_layer(4, 2, 1);  // Sigmoid output
    
    println!("Streaming network initialized with buffer size 8");
    
    // Simulate streaming data
    let data_chunks = vec![
        vec![0.1, 0.2, 0.3],           // 3 samples
        vec![0.4, 0.5],                // 2 samples  
        vec![0.6, 0.7, 0.8, 0.9],      // 4 samples (triggers processing)
        vec![1.0, 1.1],                // 2 more samples
        vec![1.2, 1.3, 1.4, 1.5, 1.6, 1.7], // 6 samples (triggers again)
    ];
    
    for (i, chunk) in data_chunks.iter().enumerate() {
        println!("Chunk {}: {} samples", i + 1, chunk.len());
        let results = streaming.process_stream(chunk);
        
        println!("  {}", streaming.buffer_status());
        
        if !results.is_empty() {
            println!("  Results: {} outputs", results.len());
            println!("  Sample: {:?}", &results[..2.min(results.len())]);
        } else {
            println!("  No output (buffer not full)");
        }
    }
    
    println!();
}

fn demo_performance() {
    println!("📋 Demo 5: Performance Comparison");
    println!("----------------------------------");
    
    let sizes = vec![
        (32, 16),
        (64, 32), 
        (128, 64),
        (256, 128),
    ];
    
    for (input_size, output_size) in sizes {
        let layer = WasmNeuralLayer::new(input_size, output_size);
        let input: Vec<f32> = (0..input_size).map(|i| i as f32 * 0.01).collect();
        
        // Simple timing (not as accurate as proper benchmarking)
        let iterations = 1000;
        let start = std::time::Instant::now();
        
        for _ in 0..iterations {
            let _ = layer.forward(&input, 0);
        }
        
        let duration = start.elapsed();
        let avg_time = duration.as_nanos() as f64 / iterations as f64;
        
        println!("Layer {}→{}: {:.2}μs per inference", 
                 input_size, output_size, avg_time / 1000.0);
        println!("  Memory: {}KB", layer.memory_usage() / 1024);
        println!("  Throughput: ~{:.0} inferences/sec", 1e9 / avg_time);
    }
    
    println!();
}

fn demo_classification() {
    println!("📋 Demo 6: Classification Network");
    println!("----------------------------------");
    
    // Create a typical image classification network architecture
    let hidden_sizes = vec![512, 256, 128, 64];
    let num_classes = 10;
    let input_features = 784; // 28x28 image
    
    let classifier = WasmNeuralNetwork::create_classifier(
        input_features, 
        &hidden_sizes, 
        num_classes
    );
    
    println!("{}", classifier.get_stats());
    
    // Simulate a flattened 28x28 grayscale image
    let image_data: Vec<f32> = (0..input_features)
        .map(|i| ((i % 255) as f32 / 255.0) * if i % 7 == 0 { 0.8 } else { 0.1 })
        .collect();
    
    let activations = vec![0, 0, 0, 0, 1]; // ReLU hidden, Sigmoid output
    
    println!("Processing simulated 28x28 image...");
    
    let start = std::time::Instant::now();
    let predictions = classifier.predict(&image_data, &activations);
    let inference_time = start.elapsed();
    
    println!("Inference time: {:.3}ms", inference_time.as_secs_f64() * 1000.0);
    println!("Class predictions:");
    
    for (class, prob) in predictions.iter().enumerate() {
        println!("  Class {}: {:.4} ({:.1}%)", class, prob, prob * 100.0);
    }
    
    let predicted_class = predictions.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    
    let confidence = predictions[predicted_class];
    println!("Predicted class: {} (confidence: {:.2}%)", predicted_class, confidence * 100.0);
    
    // Benchmark batch processing
    println!("\nBenchmarking batch processing...");
    let batch_size = 32;
    let batch_data: Vec<f32> = (0..batch_size * input_features)
        .map(|i| (i as f32 * 0.001) % 1.0)
        .collect();
    
    let start = std::time::Instant::now();
    
    // Process batch (simulate by doing individual predictions)
    let mut batch_predictions = Vec::new();
    for batch_idx in 0..batch_size {
        let start_idx = batch_idx * input_features;
        let end_idx = start_idx + input_features;
        let single_prediction = classifier.predict(&batch_data[start_idx..end_idx], &activations);
        batch_predictions.extend(single_prediction);
    }
    
    let batch_time = start.elapsed();
    let time_per_sample = batch_time.as_secs_f64() / batch_size as f64;
    
    println!("Batch of {}: {:.3}ms total", batch_size, batch_time.as_secs_f64() * 1000.0);
    println!("Time per sample: {:.3}ms", time_per_sample * 1000.0);
    println!("Throughput: {:.0} samples/sec", 1.0 / time_per_sample);
    
    println!();
}

#[cfg(test)]
mod demo_tests {
    use super::*;
    
    #[test]
    fn test_demo_functions() {
        // Test that demo functions don't panic
        demo_basic_layer();
        demo_multi_layer_network();
        demo_quantization();
        demo_streaming();
        // Note: demo_performance and demo_classification are skipped in tests
        // as they take longer and are primarily for demonstration
    }
}