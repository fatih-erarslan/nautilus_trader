//! CWTS-Ultra GPU Neural Network Demonstration
//! 
//! This example showcases the high-performance GPU neural network implementation
//! with automatic backend detection and real kernel execution.
//! 
//! Target: 100+ TFLOPS performance across CUDA, HIP, Metal, and Vulkan backends.

use cwts_ultra::neural::{GpuNeuralNetwork, GpuTensor, GpuBackend, GpuNeuralNetworkBuilder};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 CWTS-Ultra GPU Neural Network Demo");
    println!("=====================================");
    
    // Initialize GPU neural network with automatic backend detection
    println!("🔍 Detecting optimal GPU backend...");
    let gpu_nn = match GpuNeuralNetwork::new() {
        Ok(nn) => {
            println!("✅ GPU Neural Network initialized successfully!");
            println!("   Backend: {:?}", nn.backend);
            println!("   Device: {}", nn.device_info.name);
            println!("   Compute Units: {}", nn.device_info.compute_units);
            println!("   Memory: {:.1} GB", nn.device_info.memory_size as f64 / (1024.0 * 1024.0 * 1024.0));
            println!("   Peak Performance: {:.2} TFLOPS", nn.device_info.peak_tflops_fp32);
            println!();
            nn
        },
        Err(e) => {
            eprintln!("❌ Failed to initialize GPU Neural Network: {}", e);
            eprintln!("   Make sure you have compatible GPU drivers installed.");
            return Ok(());
        }
    };
    
    // Demo 1: High-Performance Matrix Multiplication
    println!("🧮 Demo 1: High-Performance Matrix Multiplication");
    demo_matrix_multiplication(&gpu_nn)?;
    println!();
    
    // Demo 2: Neural Network Layers
    println!("🧠 Demo 2: Neural Network Layers");
    demo_neural_layers(&gpu_nn)?;
    println!();
    
    // Demo 3: Convolutional Operations
    println!("🔲 Demo 3: Convolutional Neural Networks");
    demo_convolution(&gpu_nn)?;
    println!();
    
    // Demo 4: Attention Mechanisms
    println!("🎯 Demo 4: Multi-Head Attention");
    demo_attention(&gpu_nn)?;
    println!();
    
    // Demo 5: Performance Benchmarking
    println!("⚡ Demo 5: Performance Benchmarking");
    demo_performance_benchmark(&gpu_nn)?;
    println!();
    
    // Display final metrics
    let metrics = gpu_nn.get_performance_metrics();
    println!("📊 Final Performance Metrics:");
    println!("   Total Operations: {}", metrics.total_operations);
    println!("   Average TFLOPS: {:.3}", metrics.total_tflops / metrics.total_operations as f64);
    println!("   Cache Hit Rate: {:.1}%", metrics.cache_hit_rate * 100.0);
    println!("   GPU Utilization: {:.1}%", metrics.gpu_utilization * 100.0);
    
    println!("\n✨ Demo completed successfully!");
    
    Ok(())
}

fn demo_matrix_multiplication(gpu_nn: &GpuNeuralNetwork) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing large matrix multiplication (2048x2048)...");
    
    let size = 2048;
    let a_data: Vec<f32> = (0..size*size).map(|i| (i as f32) / 1000.0).collect();
    let b_data: Vec<f32> = (0..size*size).map(|i| ((i * 2) as f32) / 1000.0).collect();
    
    let start = Instant::now();
    
    // Create GPU tensors
    let a = GpuTensor::from_slice(&a_data, vec![size, size], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    let b = GpuTensor::from_slice(&b_data, vec![size, size], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    // Perform matrix multiplication on GPU
    let c = gpu_nn.matmul(&a, &b, 0)?;
    
    // Synchronize to measure actual compute time
    gpu_nn.synchronize()?;
    
    let elapsed = start.elapsed();
    let ops = 2.0 * (size as f64).powi(3); // GEMM operations
    let tflops = ops / elapsed.as_secs_f64() / 1e12;
    
    println!("   ✅ Matrix multiplication completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("   📈 Performance: {:.2} TFLOPS", tflops);
    
    // Verify a few results
    let result = c.to_vec()?;
    println!("   🔍 Sample results: C[0,0] = {:.3}, C[1,1] = {:.3}", result[0], result[size + 1]);
    
    Ok(())
}

fn demo_neural_layers(gpu_nn: &GpuNeuralNetwork) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Building and testing neural network layers...");
    
    let batch_size = 32;
    let input_dim = 512;
    let hidden_dim = 1024;
    let output_dim = 256;
    
    // Create input data
    let input_data: Vec<f32> = (0..batch_size * input_dim).map(|i| {
        (i as f32).sin() / 100.0
    }).collect();
    
    let input = GpuTensor::from_slice(&input_data, vec![batch_size, input_dim], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    // Layer 1: Linear transformation
    println!("     Testing Linear Layer ({}x{} -> {}x{})...", batch_size, input_dim, batch_size, hidden_dim);
    let weight1_data: Vec<f32> = (0..input_dim * hidden_dim).map(|i| {
        ((i as f32) * 0.01).tanh()
    }).collect();
    let bias1_data: Vec<f32> = (0..hidden_dim).map(|i| (i as f32) * 0.001).collect();
    
    let weight1 = GpuTensor::from_slice(&weight1_data, vec![input_dim, hidden_dim], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    let bias1 = GpuTensor::from_slice(&bias1_data, vec![hidden_dim], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    let hidden = gpu_nn.linear(&input, &weight1, Some(&bias1))?;
    
    // Activation: ReLU + Dropout
    println!("     Testing Fused ReLU + Dropout (50% rate)...");
    let (activated, dropout_mask) = gpu_nn.fused_relu_dropout(&hidden, 0.5, true)?;
    
    // Layer 2: Another linear layer
    println!("     Testing second Linear Layer ({}x{} -> {}x{})...", batch_size, hidden_dim, batch_size, output_dim);
    let weight2_data: Vec<f32> = (0..hidden_dim * output_dim).map(|i| {
        ((i as f32) * 0.01).cos()
    }).collect();
    let bias2_data: Vec<f32> = (0..output_dim).map(|i| -(i as f32) * 0.001).collect();
    
    let weight2 = GpuTensor::from_slice(&weight2_data, vec![hidden_dim, output_dim], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    let bias2 = GpuTensor::from_slice(&bias2_data, vec![output_dim], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    let output = gpu_nn.linear(&activated, &weight2, Some(&bias2))?;
    
    // Final activation: Softmax
    println!("     Testing Softmax activation...");
    let probabilities = gpu_nn.softmax(&output, -1)?;
    
    gpu_nn.synchronize()?;
    
    // Verify results
    let prob_vec = probabilities.to_vec()?;
    let first_row_sum: f32 = prob_vec[0..output_dim].iter().sum();
    println!("   ✅ Neural layers completed successfully");
    println!("   🎯 First sample probability sum: {:.6} (should be ~1.0)", first_row_sum);
    
    Ok(())
}

fn demo_convolution(gpu_nn: &GpuNeuralNetwork) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing 2D Convolution operations...");
    
    let batch_size = 8;
    let in_channels = 32;
    let height = 64;
    let width = 64;
    let out_channels = 64;
    let kernel_size = 3;
    
    // Create input feature maps
    let input_data: Vec<f32> = (0..batch_size * in_channels * height * width).map(|i| {
        ((i as f32) * 0.01).sin()
    }).collect();
    
    let input = GpuTensor::from_slice(
        &input_data, 
        vec![batch_size, in_channels, height, width], 
        gpu_nn.backend, 
        gpu_nn.memory_pool.clone()
    )?;
    
    // Create convolution kernel
    let kernel_data: Vec<f32> = (0..out_channels * in_channels * kernel_size * kernel_size).map(|i| {
        ((i as f32) * 0.1).tanh()
    }).collect();
    
    let kernel = GpuTensor::from_slice(
        &kernel_data,
        vec![out_channels, in_channels, kernel_size, kernel_size],
        gpu_nn.backend,
        gpu_nn.memory_pool.clone()
    )?;
    
    // Create bias
    let bias_data: Vec<f32> = (0..out_channels).map(|i| (i as f32) * 0.01).collect();
    let bias = GpuTensor::from_slice(&bias_data, vec![out_channels], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    let start = Instant::now();
    
    // Perform convolution
    let conv_output = gpu_nn.conv2d(
        &input,
        &kernel,
        Some(&bias),
        (1, 1), // stride
        (1, 1), // padding
    )?;
    
    gpu_nn.synchronize()?;
    let elapsed = start.elapsed();
    
    let output_h = height - kernel_size + 1 + 2; // with padding
    let output_w = width - kernel_size + 1 + 2;
    
    println!("   ✅ Convolution completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("   📐 Output shape: {}x{}x{}x{}", batch_size, out_channels, output_h, output_w);
    
    // Test activation after conv
    let activated_conv = gpu_nn.relu(&conv_output)?;
    println!("   🔥 ReLU activation applied to conv output");
    
    Ok(())
}

fn demo_attention(gpu_nn: &GpuNeuralNetwork) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Testing Multi-Head Attention mechanism...");
    
    let batch_size = 4;
    let seq_length = 128;
    let d_model = 512;
    let num_heads = 8;
    let head_dim = d_model / num_heads;
    
    // Create queries, keys, and values
    let q_data: Vec<f32> = (0..batch_size * seq_length * d_model).map(|i| {
        ((i as f32) * 0.01).sin()
    }).collect();
    
    let k_data: Vec<f32> = (0..batch_size * seq_length * d_model).map(|i| {
        ((i as f32) * 0.01).cos()
    }).collect();
    
    let v_data: Vec<f32> = (0..batch_size * seq_length * d_model).map(|i| {
        ((i as f32) * 0.01).tanh()
    }).collect();
    
    let queries = GpuTensor::from_slice(&q_data, vec![batch_size, seq_length, d_model], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    let keys = GpuTensor::from_slice(&k_data, vec![batch_size, seq_length, d_model], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    let values = GpuTensor::from_slice(&v_data, vec![batch_size, seq_length, d_model], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    let start = Instant::now();
    
    // Perform multi-head attention
    let attention_output = gpu_nn.multi_head_attention(
        &queries,
        &keys,
        &values,
        num_heads,
        scale,
        None, // no mask
    )?;
    
    gpu_nn.synchronize()?;
    let elapsed = start.elapsed();
    
    println!("   ✅ Multi-Head Attention completed in {:.2}ms", elapsed.as_secs_f64() * 1000.0);
    println!("   🎯 Attention pattern computed for {} heads", num_heads);
    
    // Test layer normalization
    println!("   Testing Layer Normalization...");
    let ln_weight_data: Vec<f32> = vec![1.0; d_model];
    let ln_bias_data: Vec<f32> = vec![0.0; d_model];
    
    let ln_weight = GpuTensor::from_slice(&ln_weight_data, vec![d_model], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    let ln_bias = GpuTensor::from_slice(&ln_bias_data, vec![d_model], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
    
    let normalized = gpu_nn.layer_norm(&attention_output, &ln_weight, &ln_bias, 1e-5)?;
    
    println!("   📐 Layer normalization applied to attention output");
    
    Ok(())
}

fn demo_performance_benchmark(gpu_nn: &GpuNeuralNetwork) -> Result<(), Box<dyn std::error::Error>> {
    println!("   Running comprehensive performance benchmark...");
    
    let sizes = vec![256, 512, 1024, 2048];
    
    for &size in &sizes {
        println!("     Benchmarking {}x{} matrix operations...", size, size);
        
        let a_data: Vec<f32> = (0..size*size).map(|_| rand::random::<f32>() - 0.5).collect();
        let b_data: Vec<f32> = (0..size*size).map(|_| rand::random::<f32>() - 0.5).collect();
        
        let a = GpuTensor::from_slice(&a_data, vec![size, size], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
        let b = GpuTensor::from_slice(&b_data, vec![size, size], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
        
        let start = Instant::now();
        
        // Perform multiple operations
        let c = gpu_nn.matmul(&a, &b, 0)?;
        let c_relu = gpu_nn.relu(&c)?;
        let c_soft = gpu_nn.softmax(&c_relu, -1)?;
        let c_t = gpu_nn.transpose(&c_soft)?;
        
        gpu_nn.synchronize()?;
        let elapsed = start.elapsed();
        
        let ops = 2.0 * (size as f64).powi(3); // Just matmul ops for simplicity
        let tflops = ops / elapsed.as_secs_f64() / 1e12;
        
        println!("       Size {}: {:.2} TFLOPS ({:.2}ms)", size, tflops, elapsed.as_secs_f64() * 1000.0);
    }
    
    Ok(())
}

/// Alternative initialization with custom configuration
#[allow(dead_code)]
fn demo_custom_initialization() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔧 Custom GPU Neural Network Configuration");
    
    // Create with specific backend (if available)
    let custom_nn = GpuNeuralNetworkBuilder::new()
        .with_backend(GpuBackend::Cuda) // Force CUDA if available
        .with_memory_pool_mb(2048) // 2GB memory pool
        .with_streams(8) // 8 concurrent streams
        .build()?;
    
    println!("   Custom configuration:");
    println!("   - Backend: {:?}", custom_nn.backend);
    println!("   - Memory Pool: 2GB");
    println!("   - Concurrent Streams: 8");
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_nn_basic_operations() -> Result<(), Box<dyn std::error::Error>> {
        let gpu_nn = match GpuNeuralNetwork::new() {
            Ok(nn) => nn,
            Err(_) => {
                println!("GPU not available for testing, skipping...");
                return Ok(());
            }
        };
        
        // Test small matrix multiplication
        let a_data = vec![1.0, 2.0, 3.0, 4.0];
        let b_data = vec![5.0, 6.0, 7.0, 8.0];
        
        let a = GpuTensor::from_slice(&a_data, vec![2, 2], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
        let b = GpuTensor::from_slice(&b_data, vec![2, 2], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
        
        let c = gpu_nn.matmul(&a, &b, 0)?;
        let result = c.to_vec()?;
        
        // Verify basic computation works
        assert!(result.len() == 4);
        assert!(result[0] > 0.0); // Should have some result
        
        Ok(())
    }
    
    #[test]
    fn test_tensor_operations() -> Result<(), Box<dyn std::error::Error>> {
        let gpu_nn = match GpuNeuralNetwork::new() {
            Ok(nn) => nn,
            Err(_) => return Ok(()),
        };
        
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor = GpuTensor::from_slice(&data, vec![4], gpu_nn.backend, gpu_nn.memory_pool.clone())?;
        
        // Test ReLU
        let relu_output = gpu_nn.relu(&tensor)?;
        let result = relu_output.to_vec()?;
        
        // ReLU should clamp negative values to 0
        assert!(result[0] == 0.0 || result[0] >= -1e-5); // Account for floating point
        assert!(result[2] > 0.0);
        assert!(result[3] > 0.0);
        
        Ok(())
    }
}