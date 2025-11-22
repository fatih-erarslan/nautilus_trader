#[cfg(test)]
mod metal_tests {
    use super::super::metal::*;
    use std::sync::Arc;
    
    // Test data generators for Metal testing
    fn generate_test_matrix_2x2() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0]
    }
    
    fn generate_test_matrix_4x4() -> Vec<f32> {
        (1..=16).map(|i| i as f32).collect()
    }
    
    fn generate_random_matrix(size: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        (0..size).map(|i| {
            let mut hasher = DefaultHasher::new();
            (i * 17).hash(&mut hasher);
            (hasher.finish() % 1000) as f32 / 1000.0
        }).collect()
    }
    
    fn generate_neural_weights() -> Vec<Vec<f32>> {
        vec![
            vec![0.5, 0.3, 0.2, 0.8, 0.1, 0.7],  // 3->2 layer
            vec![0.9, 0.4],                       // 2->1 layer
        ]
    }
    
    fn generate_neural_biases() -> Vec<Vec<f32>> {
        vec![
            vec![0.1, 0.2],  // 2 neurons
            vec![0.05],      // 1 neuron
        ]
    }
    
    #[test]
    fn test_metal_gpu_creation() {
        match MetalGpu::new() {
            Ok(gpu) => {
                println!("✅ Metal GPU created successfully");
                let info = gpu.get_device_info();
                println!("Metal Device: {}", info.name);
                println!("Max threads: {}", info.max_threads_per_threadgroup);
                println!("Memory: {} MB", info.memory_size / (1024 * 1024));
                
                assert!(!info.name.is_empty());
                assert!(info.max_threads_per_threadgroup > 0);
                assert!(info.memory_size > 0);
                assert!(info.supports_unified_memory);
            }
            Err(e) => {
                println!("⚠️ Metal not available (expected on non-macOS): {}", e);
                assert!(e.contains("Metal") || e.contains("macOS") || e.contains("device"));
            }
        }
    }
    
    #[test]
    fn test_metal_device_info_validation() {
        match MetalGpu::new() {
            Ok(gpu) => {
                let info = gpu.get_device_info();
                
                // Validate DeviceInfo structure
                assert!(!info.name.is_empty());
                assert!(info.memory_size > 1024 * 1024); // At least 1MB
                assert!(info.max_threads_per_threadgroup >= 32);
                assert!(info.max_threads_per_threadgroup <= 1024);
                assert!(info.supports_unified_memory); // macOS should support unified memory
                
                println!("Full device info: {:?}", info);
            }
            Err(_) => {
                println!("Metal not available, skipping device info validation");
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored on macOS
    fn test_metal_matmul_2x2() {
        if let Ok(gpu) = MetalGpu::new() {
            let a = generate_test_matrix_2x2();
            let b = vec![5.0, 6.0, 7.0, 8.0];
            
            let result = gpu.matmul(&a, &b, 2, 2, 2);
            
            // Expected: [[19, 22], [43, 50]]
            assert_eq!(result.len(), 4);
            assert!((result[0] - 19.0).abs() < 0.001);
            assert!((result[1] - 22.0).abs() < 0.001);
            assert!((result[2] - 43.0).abs() < 0.001);
            assert!((result[3] - 50.0).abs() < 0.001);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_matmul_performance() {
        if let Ok(gpu) = MetalGpu::new() {
            // Test different matrix sizes to characterize Metal performance
            let sizes = vec![16, 32, 64, 128];
            
            for size in sizes {
                let a = generate_random_matrix(size * size);
                let b = generate_random_matrix(size * size);
                
                let start = std::time::Instant::now();
                let result = gpu.matmul(&a, &b, size, size, size);
                let duration = start.elapsed();
                
                assert_eq!(result.len(), size * size);
                assert!(result.iter().all(|&x| x.is_finite()));
                
                let gflops = (2.0 * size.pow(3)) as f64 / duration.as_nanos() as f64 * 1e9 / 1e9;
                println!("Metal {}x{} matmul: {:?} ({:.2} GFLOPS)", size, size, duration, gflops);
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored  
    fn test_metal_nn_forward() {
        if let Ok(gpu) = MetalGpu::new() {
            let input = vec![1.0, 0.5, 0.8];
            let weights = generate_neural_weights();
            let biases = generate_neural_biases();
            
            let weights_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
            let biases_refs: Vec<&[f32]> = biases.iter().map(|b| b.as_slice()).collect();
            
            let result = gpu.nn_forward(&input, &weights_refs, &biases_refs);
            
            assert_eq!(result.len(), 1);
            assert!(result[0] >= 0.0); // ReLU should ensure non-negative
            assert!(result[0].is_finite());
            
            println!("NN forward result: {:?}", result);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_conv2d_sobel() {
        if let Ok(gpu) = MetalGpu::new() {
            // 4x4 test image
            let input = vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ];
            
            // Sobel X kernel (edge detection)
            let kernel = vec![
                -1.0, 0.0, 1.0,
                -2.0, 0.0, 2.0,
                -1.0, 0.0, 1.0,
            ];
            
            let result = gpu.conv2d(&input, &kernel, 1, 1, 4, 4, 3);
            
            // Output should be 2x2 (4x4 input with 3x3 kernel)
            assert_eq!(result.len(), 4);
            assert!(result.iter().all(|&x| x.is_finite()));
            
            // Sobel should detect horizontal edges
            println!("Sobel conv2d result: {:?}", result);
            assert!(result.iter().any(|&x| x.abs() > 1.0));
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_reduce_operations() {
        if let Ok(gpu) = MetalGpu::new() {
            // Test sum reduction
            let data_sum = vec![1.0; 1000];
            let sum = gpu.reduce_sum(&data_sum);
            assert!((sum - 1000.0).abs() < 0.001);
            
            // Test max reduction
            let data_max = vec![1.0, 5.0, 3.0, 9.0, 2.0, 7.0, 1.0];
            let max_val = gpu.reduce_max(&data_max);
            assert!((max_val - 9.0).abs() < 0.001);
            
            // Test min reduction
            let min_val = gpu.reduce_min(&data_max);
            assert!((min_val - 1.0).abs() < 0.001);
            
            println!("Reductions: sum={}, max={}, min={}", sum, max_val, min_val);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_simd_operations() {
        if let Ok(gpu) = MetalGpu::new() {
            // Test vector operations that can benefit from SIMD
            let a = (0..1024).map(|i| i as f32).collect::<Vec<_>>();
            let b = (0..1024).map(|i| (1024 - i) as f32).collect::<Vec<_>>();
            
            let start = std::time::Instant::now();
            let dot_product = gpu.dot_product(&a, &b);
            let duration = start.elapsed();
            
            // Verify dot product correctness
            let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            assert!((dot_product - expected).abs() < 1.0);
            
            println!("SIMD dot product (1024 elements): {:?}", duration);
            
            // Test element-wise operations
            let result_add = gpu.vector_add(&a, &b);
            assert_eq!(result_add.len(), 1024);
            assert!((result_add[0] - 1024.0).abs() < 0.001); // 0 + 1024
            assert!((result_add[512] - 1024.0).abs() < 0.001); // 512 + 512
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_threadgroup_optimization() {
        if let Ok(gpu) = MetalGpu::new() {
            let info = gpu.get_device_info();
            let max_threads = info.max_threads_per_threadgroup;
            
            // Test with threadgroup-aligned sizes
            let aligned_size = max_threads * 4;
            let data = vec![2.0; aligned_size];
            
            let start = std::time::Instant::now();
            let sum = gpu.reduce_sum(&data);
            let aligned_duration = start.elapsed();
            
            assert!((sum - (aligned_size * 2) as f32).abs() < 0.001);
            
            // Test with non-aligned size
            let unaligned_size = max_threads * 4 + max_threads / 3;
            let data_unaligned = vec![2.0; unaligned_size];
            
            let start = std::time::Instant::now();
            let sum_unaligned = gpu.reduce_sum(&data_unaligned);
            let unaligned_duration = start.elapsed();
            
            assert!((sum_unaligned - (unaligned_size * 2) as f32).abs() < 0.001);
            
            println!("Aligned vs unaligned performance: {:?} vs {:?}", 
                    aligned_duration, unaligned_duration);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_unified_memory_advantage() {
        if let Ok(gpu) = MetalGpu::new() {
            let info = gpu.get_device_info();
            
            if info.supports_unified_memory {
                // Test large data transfer that benefits from unified memory
                let large_data = generate_random_matrix(100_000);
                
                let start = std::time::Instant::now();
                let sum = gpu.reduce_sum(&large_data);
                let duration = start.elapsed();
                
                assert!(sum.is_finite());
                assert!(sum > 0.0);
                
                println!("Large unified memory operation (400KB): {:?}", duration);
                
                // Should be faster than discrete GPU memory transfers
                assert!(duration.as_millis() < 100);
            }
        }
    }
    
    #[test]
    fn test_metal_shader_constants() {
        // Verify Metal shader source code
        let matmul_shader = MATMUL_SHADER_METAL;
        assert!(matmul_shader.contains("#include <metal_stdlib>"));
        assert!(matmul_shader.contains("using namespace metal;"));
        assert!(matmul_shader.contains("kernel void matmul_kernel"));
        assert!(matmul_shader.contains("threadgroup_position_in_grid"));
        assert!(matmul_shader.contains("simdgroup_matrix"));
        
        let conv2d_shader = CONV2D_SHADER_METAL;
        assert!(conv2d_shader.contains("conv2d_kernel"));
        assert!(conv2d_shader.contains("texture2d"));
        
        let reduce_shader = REDUCE_SHADER_METAL;
        assert!(reduce_shader.contains("reduce_sum_kernel"));
        assert!(reduce_shader.contains("simd_sum"));
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_precision_modes() {
        if let Ok(gpu) = MetalGpu::new() {
            // Test different precision modes if supported
            let a = vec![1.0/3.0, 2.0/3.0, 1.0, 4.0/3.0];
            let b = vec![3.0, 6.0, 9.0, 12.0];
            
            let result = gpu.matmul(&a, &b, 2, 2, 2);
            
            // Check precision with fractional inputs
            assert!((result[0] - 7.0).abs() < 0.0001);
            assert!((result[1] - 10.0).abs() < 0.0001);
            assert!((result[2] - 15.0).abs() < 0.0001);
            assert!((result[3] - 22.0).abs() < 0.0001);
            
            // Test half-precision if available
            if gpu.supports_half_precision() {
                println!("Testing half-precision mode");
                let result_half = gpu.matmul_half_precision(&a, &b, 2, 2, 2);
                
                // Half precision should be less accurate but still reasonable
                for (i, (&full, &half)) in result.iter().zip(result_half.iter()).enumerate() {
                    let diff = (full - half).abs();
                    println!("Element {}: full={}, half={}, diff={}", i, full, half, diff);
                    assert!(diff < 0.01); // Allow larger tolerance for half precision
                }
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_concurrent_operations() {
        if let Ok(gpu) = MetalGpu::new() {
            let gpu = Arc::new(gpu);
            let mut handles = vec![];
            
            // Test concurrent matrix multiplications
            for i in 0..4 {
                let gpu_clone = Arc::clone(&gpu);
                let handle = std::thread::spawn(move || {
                    let size = 16;
                    let a = vec![i as f32 + 1.0; size * size];
                    let b = vec![(i + 1) as f32; size * size];
                    
                    let result = gpu_clone.matmul(&a, &b, size, size, size);
                    assert_eq!(result.len(), size * size);
                    
                    // Verify result consistency
                    let expected_elem = (i as f32 + 1.0) * (i + 1) as f32 * size as f32;
                    assert!((result[0] - expected_elem).abs() < 1.0);
                    
                    println!("Thread {} completed matmul", i);
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().expect("Thread panicked");
            }
            
            println!("✅ Concurrent Metal operations completed");
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_edge_cases() {
        if let Ok(gpu) = MetalGpu::new() {
            // Test 1x1 matrix
            let result_1x1 = gpu.matmul(&[5.0], &[3.0], 1, 1, 1);
            assert_eq!(result_1x1.len(), 1);
            assert!((result_1x1[0] - 15.0).abs() < 0.001);
            
            // Test empty input handling
            let empty_result = gpu.reduce_sum(&[]);
            assert_eq!(empty_result, 0.0);
            
            // Test single element
            let single_result = gpu.reduce_sum(&[42.0]);
            assert!((single_result - 42.0).abs() < 0.001);
            
            // Test all zeros
            let zeros_result = gpu.reduce_sum(&vec![0.0; 1000]);
            assert!((zeros_result - 0.0).abs() < 0.001);
            
            // Test infinity and NaN handling
            let inf_data = vec![f32::INFINITY, 1.0, 2.0];
            let inf_result = gpu.reduce_sum(&inf_data);
            assert!(inf_result.is_infinite());
            
            let nan_data = vec![f32::NAN, 1.0, 2.0];
            let nan_result = gpu.reduce_sum(&nan_data);
            assert!(nan_result.is_nan());
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_memory_pressure() {
        if let Ok(gpu) = MetalGpu::new() {
            let info = gpu.get_device_info();
            let available_memory = info.memory_size;
            
            // Test with large allocation (but safe)
            let safe_size = (available_memory / 8 / 4) as usize; // Use 1/8 of memory for floats
            let large_data = vec![1.0; safe_size];
            
            let start = std::time::Instant::now();
            let sum = gpu.reduce_sum(&large_data);
            let duration = start.elapsed();
            
            assert!((sum - safe_size as f32).abs() < 1.0);
            println!("Large memory test ({}MB): {:?}", 
                    safe_size * 4 / 1024 / 1024, duration);
            
            // Should complete without error
            assert!(duration.as_secs() < 5);
        }
    }
    
    #[test]
    fn test_metal_error_handling() {
        match MetalGpu::new() {
            Ok(_) => println!("Metal GPU available"),
            Err(e) => {
                let error_msg = e.to_lowercase();
                assert!(
                    error_msg.contains("metal") || 
                    error_msg.contains("macos") ||
                    error_msg.contains("device") ||
                    error_msg.contains("failed"),
                    "Error message should be descriptive: {}", e
                );
            }
        }
    }
    
    #[test]
    fn test_metal_drop_cleanup() {
        if let Ok(gpu) = MetalGpu::new() {
            drop(gpu);
            println!("✅ Metal GPU cleanup successful");
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_apple_silicon_optimizations() {
        if let Ok(gpu) = MetalGpu::new() {
            let info = gpu.get_device_info();
            
            if info.name.contains("Apple") || info.name.contains("M1") || info.name.contains("M2") {
                println!("Testing Apple Silicon specific optimizations");
                
                // Test matrix operations that benefit from Apple Silicon
                let size = 256;
                let a = generate_random_matrix(size * size);
                let b = generate_random_matrix(size * size);
                
                let start = std::time::Instant::now();
                let result = gpu.matmul(&a, &b, size, size, size);
                let duration = start.elapsed();
                
                assert_eq!(result.len(), size * size);
                
                // Apple Silicon should be quite fast for this
                let gflops = (2.0 * size.pow(3)) as f64 / duration.as_nanos() as f64 * 1e9 / 1e9;
                println!("Apple Silicon performance: {:.2} GFLOPS", gflops);
                
                // Should achieve reasonable performance
                assert!(gflops > 1.0);
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_metal_neural_network_pipeline() {
        if let Ok(gpu) = MetalGpu::new() {
            // Test a complete neural network pipeline
            let input = vec![1.0, 0.5, -0.5, 2.0, -1.0];
            
            // Multi-layer network: 5 -> 3 -> 2 -> 1
            let weights = vec![
                (0..15).map(|i| (i as f32 * 0.1) - 0.7).collect(), // 5->3 (15 weights)
                vec![0.3, 0.1, -0.2, 0.8, 0.5, -0.1],             // 3->2 (6 weights)
                vec![0.7, -0.4],                                     // 2->1 (2 weights)
            ];
            let biases = vec![
                vec![0.1, -0.05, 0.15],  // 3 neurons
                vec![0.2, -0.1],         // 2 neurons  
                vec![0.05],              // 1 neuron
            ];
            
            let weights_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
            let biases_refs: Vec<&[f32]> = biases.iter().map(|b| b.as_slice()).collect();
            
            let result = gpu.nn_forward(&input, &weights_refs, &biases_refs);
            
            assert_eq!(result.len(), 1);
            assert!(result[0].is_finite());
            println!("NN pipeline result: {:?}", result);
            
            // Test multiple inferences
            for i in 0..10 {
                let varied_input: Vec<f32> = input.iter().map(|&x| x * (i as f32 * 0.1 + 1.0)).collect();
                let inference_result = gpu.nn_forward(&varied_input, &weights_refs, &biases_refs);
                assert_eq!(inference_result.len(), 1);
                assert!(inference_result[0].is_finite());
            }
            
            println!("✅ Neural network pipeline test completed");
        }
    }
}