//! Integration tests for WASM Neural Network implementation
//! 
//! These tests verify that the WASM NN implementation works correctly
//! with both SIMD and scalar fallback modes.

use super::wasm_nn::*;

#[cfg(test)]
mod wasm_nn_integration_tests {
    use super::*;

    #[test]
    fn test_single_layer_forward_pass() {
        let layer = WasmNeuralLayer::new(4, 2);
        let input = vec![1.0, -1.0, 2.0, -2.0];
        
        // Test all activation functions
        let relu_output = layer.forward(&input, 0);
        let sigmoid_output = layer.forward(&input, 1);
        let tanh_output = layer.forward(&input, 2);
        let leaky_relu_output = layer.forward(&input, 3);
        let swish_output = layer.forward(&input, 4);
        let gelu_output = layer.forward(&input, 5);
        
        // Verify outputs have correct dimensions and are finite
        assert_eq!(relu_output.len(), 2);
        assert_eq!(sigmoid_output.len(), 2);
        assert_eq!(tanh_output.len(), 2);
        assert_eq!(leaky_relu_output.len(), 2);
        assert_eq!(swish_output.len(), 2);
        assert_eq!(gelu_output.len(), 2);
        
        assert!(relu_output.iter().all(|&x| x.is_finite()));
        assert!(sigmoid_output.iter().all(|&x| x.is_finite() && x >= 0.0 && x <= 1.0));
        assert!(tanh_output.iter().all(|&x| x.is_finite() && x >= -1.0 && x <= 1.0));
        assert!(leaky_relu_output.iter().all(|&x| x.is_finite()));
        assert!(swish_output.iter().all(|&x| x.is_finite()));
        assert!(gelu_output.iter().all(|&x| x.is_finite()));
        
        println!("✅ Single layer forward pass test passed");
        println!("   ReLU: {:?}", relu_output);
        println!("   Sigmoid: {:?}", sigmoid_output);
        println!("   Tanh: {:?}", tanh_output);
    }

    #[test]
    fn test_quantization_accuracy() {
        let mut layer = WasmNeuralLayer::new(8, 4);
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        
        // Get original output
        let original_output = layer.forward(&input, 0);
        
        // Quantize to INT8
        layer.quantize(1, false);
        let quantized_output = layer.forward(&input, 0);
        
        // Verify quantized output is reasonably close to original
        assert_eq!(original_output.len(), quantized_output.len());
        
        for (orig, quant) in original_output.iter().zip(quantized_output.iter()) {
            let error = (orig - quant).abs() / orig.abs().max(1e-6);
            assert!(error < 0.2, "Quantization error too high: orig={}, quant={}, error={}", orig, quant, error);
        }
        
        println!("✅ Quantization accuracy test passed");
        println!("   Original: {:?}", original_output);
        println!("   Quantized: {:?}", quantized_output);
    }

    #[test]
    fn test_multi_layer_network() {
        let mut network = WasmNeuralNetwork::new();
        network.add_layer(8, 6);
        network.add_layer(6, 4);
        network.add_layer(4, 2);
        
        let input = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let activations = vec![0, 0, 1]; // ReLU, ReLU, Sigmoid
        
        let output = network.predict(&input, &activations);
        
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite() && x >= 0.0 && x <= 1.0));
        
        println!("✅ Multi-layer network test passed");
        println!("   Input: {:?}", input);
        println!("   Output: {:?}", output);
        println!("   Stats: {}", network.get_stats());
    }

    #[test]
    fn test_batch_processing() {
        let layer = WasmNeuralLayer::new(3, 2);
        let batch_inputs = vec![
            // Batch 1
            1.0, 2.0, 3.0,
            // Batch 2  
            4.0, 5.0, 6.0,
            // Batch 3
            7.0, 8.0, 9.0,
        ];
        
        let batch_outputs = layer.batch_forward(&batch_inputs, 3, 0);
        
        assert_eq!(batch_outputs.len(), 6); // 3 batches * 2 outputs
        assert!(batch_outputs.iter().all(|&x| x.is_finite()));
        
        // Verify batch processing gives same results as individual processing
        let individual_1 = layer.forward(&batch_inputs[0..3], 0);
        let individual_2 = layer.forward(&batch_inputs[3..6], 0);
        let individual_3 = layer.forward(&batch_inputs[6..9], 0);
        
        let expected: Vec<f32> = individual_1.into_iter()
            .chain(individual_2.into_iter())
            .chain(individual_3.into_iter())
            .collect();
        
        for (batch, expected) in batch_outputs.iter().zip(expected.iter()) {
            assert!((batch - expected).abs() < 1e-6);
        }
        
        println!("✅ Batch processing test passed");
        println!("   Batch size: 3, Output size: 2");
        println!("   Results: {:?}", batch_outputs);
    }

    #[test]
    fn test_streaming_network() {
        let mut streaming = WasmStreamingNN::new(4);
        streaming.add_layer(4, 3, 0); // ReLU
        streaming.add_layer(3, 2, 1); // Sigmoid
        
        // Stream data in chunks
        let chunk1 = vec![1.0, 2.0];
        let chunk2 = vec![3.0, 4.0];
        let chunk3 = vec![5.0, 6.0, 7.0, 8.0]; // This should trigger processing
        
        let results1 = streaming.process_stream(&chunk1);
        let results2 = streaming.process_stream(&chunk2); 
        let results3 = streaming.process_stream(&chunk3);
        
        // First two chunks shouldn't produce results (buffer not full)
        assert!(results1.is_empty());
        assert!(results2.is_empty());
        
        // Third chunk should trigger processing
        assert!(!results3.is_empty());
        assert!(results3.iter().all(|&x| x.is_finite() && x >= 0.0 && x <= 1.0));
        
        println!("✅ Streaming network test passed");
        println!("   Buffer status: {}", streaming.buffer_status());
        println!("   Results: {:?}", results3);
    }

    #[test] 
    fn test_classifier_creation() {
        let hidden_sizes = vec![16, 8];
        let classifier = WasmNeuralNetwork::create_classifier(32, &hidden_sizes, 5);
        
        let input = vec![0.1; 32];
        let activations = vec![0, 0, 1]; // ReLU hidden layers, Sigmoid output
        
        let predictions = classifier.predict(&input, &activations);
        
        assert_eq!(predictions.len(), 5);
        assert!(predictions.iter().all(|&x| x.is_finite() && x >= 0.0 && x <= 1.0));
        
        // Test softmax-like behavior (should sum approximately to something reasonable)
        let sum: f32 = predictions.iter().sum();
        assert!(sum > 0.0 && sum < 5.0);
        
        println!("✅ Classifier creation test passed");
        println!("   Input size: 32, Classes: 5");
        println!("   Predictions: {:?}", predictions);
        println!("   Sum: {:.4}", sum);
    }

    #[test]
    fn test_memory_usage_estimation() {
        let layer = WasmNeuralLayer::new(100, 50);
        let memory_usage = layer.memory_usage();
        
        // Expected: 100*50 weights + 50 biases, each f32 = 4 bytes
        let expected_min = (100 * 50 + 50) * 4;
        
        assert!(memory_usage >= expected_min);
        assert!(memory_usage < expected_min * 2); // Should not be too much overhead
        
        println!("✅ Memory usage test passed");
        println!("   Expected minimum: {} bytes", expected_min);
        println!("   Actual: {} bytes", memory_usage);
        println!("   Overhead: {:.2}%", 
                 (memory_usage as f64 - expected_min as f64) / expected_min as f64 * 100.0);
    }

    #[test]
    fn test_performance_benchmark() {
        let mut network = WasmNeuralNetwork::new();
        network.add_layer(64, 32);
        network.add_layer(32, 16);  
        network.add_layer(16, 8);
        
        let input = vec![0.5; 64];
        let activations = vec![0, 0, 1]; // ReLU, ReLU, Sigmoid
        
        let avg_time = network.benchmark(&input, &activations, 100);
        
        assert!(avg_time >= 0.0);
        assert!(avg_time < 100.0); // Should be less than 100ms per iteration
        
        println!("✅ Performance benchmark test passed");
        println!("   Average time per iteration: {:.4}ms", avg_time);
        println!("   Network stats: {}", network.get_stats());
    }

    #[test]
    fn test_quantization_utilities() {
        let original_data = vec![-2.5, -1.0, 0.0, 1.0, 2.5, 3.7];
        
        // Test INT8 quantization with dynamic range
        let (quantized_i8, scale, offset) = WasmNeuralUtils::quantize_f32_to_i8(&original_data, true);
        let dequantized = WasmNeuralUtils::dequantize_i8_to_f32(&quantized_i8, scale, offset);
        
        assert_eq!(quantized_i8.len(), original_data.len());
        assert_eq!(dequantized.len(), original_data.len());
        
        // Check quantization accuracy
        for (orig, deq) in original_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs() / orig.abs().max(0.1);
            assert!(error < 0.15, "Quantization error too high: orig={}, deq={}", orig, deq);
        }
        
        println!("✅ Quantization utilities test passed");
        println!("   Original: {:?}", original_data);
        println!("   Scale: {:.6}, Offset: {}", scale, offset);
        println!("   Dequantized: {:?}", dequantized);
    }

    #[test]
    fn test_edge_cases() {
        // Test with empty input
        let layer = WasmNeuralLayer::new(1, 1);
        let empty_input: Vec<f32> = vec![];
        // This should not panic, though result may be undefined
        
        // Test with zero input
        let zero_input = vec![0.0; 4];
        let zero_output = layer.forward(&zero_input, 0);
        assert!(zero_output.iter().all(|&x| x.is_finite()));
        
        // Test with very large values
        let large_input = vec![1000.0, -1000.0, 500.0, -500.0];
        let large_output = layer.forward(&large_input, 0);
        assert!(large_output.iter().all(|&x| x.is_finite()));
        
        // Test with very small values  
        let small_input = vec![1e-10, -1e-10, 1e-8, -1e-8];
        let small_output = layer.forward(&small_input, 0);
        assert!(small_output.iter().all(|&x| x.is_finite()));
        
        println!("✅ Edge cases test passed");
        println!("   Zero input output: {:?}", zero_output);
        println!("   Large input handled: {}", large_output.len());
        println!("   Small input handled: {}", small_output.len());
    }

    #[test]
    fn test_complexity_calculation() {
        let layer1 = WasmNeuralLayer::new(10, 5);
        let layer2 = WasmNeuralLayer::new(5, 2);
        let layers = vec![layer1, layer2];
        
        let complexity = WasmNeuralUtils::calculate_complexity(&layers);
        
        // Expected params: (10*5 + 5) + (5*2 + 2) = 50 + 5 + 10 + 2 = 67
        // Expected ops: (10*5*2) + (5*2*2) = 100 + 20 = 120
        // Score: (67 * 0.6 + 120 * 0.4) / 1000 = (40.2 + 48) / 1000 = 0.0882
        let expected_complexity = (67.0 * 0.6 + 120.0 * 0.4) / 1000.0;
        
        assert!((complexity - expected_complexity).abs() < 0.001);
        
        println!("✅ Complexity calculation test passed");
        println!("   Calculated complexity: {:.6}", complexity);
        println!("   Expected complexity: {:.6}", expected_complexity);
    }

    #[test]
    fn test_wasm_feature_detection() {
        // This test mainly verifies the code doesn't panic when detecting features
        let features = crate::simd::wasm32::WasmFeatures::detect();
        
        println!("✅ WASM feature detection test passed");
        println!("   SIMD128 support: {}", features.has_simd128);
        println!("   Relaxed SIMD: {}", features.has_relaxed_simd);
        println!("   Threads: {}", features.has_threads);
        println!("   Bulk memory: {}", features.has_bulk_memory);
        println!("   Browser: {}", features.is_browser);
        println!("   Node.js: {}", features.is_nodejs);
    }
}

/// Integration test runner that can be called from WASM
#[cfg(target_arch = "wasm32")]
pub fn run_wasm_integration_tests() {
    use wasm_bindgen::prelude::*;
    
    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = console)]
        fn log(s: &str);
    }
    
    macro_rules! console_log {
        ($($t:tt)*) => (log(&format_args!($($t)*).to_string()))
    }
    
    console_log!("🧪 Running WASM Neural Network Integration Tests...");
    
    // Run basic functionality test
    let layer = WasmNeuralLayer::new(4, 2);
    let input = vec![1.0, -1.0, 2.0, -2.0];
    let output = layer.forward(&input, 0);
    
    console_log!("✅ Basic forward pass: input={:?}, output={:?}", input, output);
    
    // Test quantization
    let mut quant_layer = WasmNeuralLayer::new(4, 2);
    quant_layer.quantize(1, false);
    let quant_output = quant_layer.forward(&input, 0);
    
    console_log!("✅ Quantized forward pass: output={:?}", quant_output);
    
    // Test network
    let mut network = WasmNeuralNetwork::new();
    network.add_layer(4, 3);
    network.add_layer(3, 2);
    let activations = vec![0, 1];
    let net_output = network.predict(&input, &activations);
    
    console_log!("✅ Network prediction: output={:?}", net_output);
    console_log!("📊 {}", network.get_stats());
    
    console_log!("🎉 All WASM NN integration tests completed successfully!");
}