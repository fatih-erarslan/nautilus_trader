#[cfg(test)]
mod hip_tests {
    use super::super::hip::*;
    use std::sync::Arc;
    
    // Test data generators for HIP testing
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
            (i * 31).hash(&mut hasher);
            (hasher.finish() % 1000) as f32 / 1000.0
        }).collect()
    }
    
    fn generate_conv2d_input() -> Vec<f32> {
        // 1x1x4x4 image (batch=1, channels=1, height=4, width=4)
        (1..=16).map(|i| i as f32).collect()
    }
    
    fn generate_conv2d_kernel() -> Vec<f32> {
        // 3x3 kernel
        vec![1.0, 0.0, -1.0,
             2.0, 0.0, -2.0, 
             1.0, 0.0, -1.0]
    }
    
    #[test]
    fn test_hip_gpu_creation() {
        match HipGpu::new() {
            Ok(gpu) => {
                println!("✅ HIP GPU created successfully");
                let info = gpu.get_device_info();
                println!("AMD GPU: {}", info.name);
                println!("Compute units: {}", info.compute_units);
                println!("Wavefront size: {}", info.wavefront_size);
                assert!(!info.name.is_empty());
                assert!(info.compute_units > 0);
                assert!(info.wavefront_size > 0);
            }
            Err(e) => {
                println!("⚠️ HIP not available (expected on non-AMD systems): {}", e);
                assert!(e.contains("device") || e.contains("HIP"));
            }
        }
    }
    
    #[test]
    fn test_hip_device_info_structure() {
        match HipGpu::new() {
            Ok(gpu) => {
                let info = gpu.get_device_info();
                
                // Validate DeviceInfo structure
                assert!(info.name.len() > 0);
                assert!(info.memory_size > 0);
                assert!(info.shared_memory > 0);
                assert!(info.clock_rate_mhz > 0);
                assert!(info.wavefront_size == 32 || info.wavefront_size == 64);
                
                println!("Device info: {:?}", info);
            }
            Err(_) => {
                println!("HIP not available, skipping device info test");
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored on AMD systems
    fn test_hip_matmul_2x2() {
        if let Ok(gpu) = HipGpu::new() {
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
    fn test_hip_matmul_large() {
        if let Ok(gpu) = HipGpu::new() {
            let size = 32;
            let a = generate_random_matrix(size * size);
            let b = generate_random_matrix(size * size);
            
            let start = std::time::Instant::now();
            let result = gpu.matmul(&a, &b, size, size, size);
            let duration = start.elapsed();
            
            assert_eq!(result.len(), size * size);
            println!("HIP {}x{} matmul took: {:?}", size, size, duration);
            
            // Verify computation correctness with a simple check
            assert!(result.iter().all(|&x| x.is_finite()));
            assert!(result.iter().any(|&x| x != 0.0));
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_nn_forward() {
        if let Ok(gpu) = HipGpu::new() {
            let input = vec![1.0, 0.5, 0.3, 0.8];
            let weights = vec![
                vec![0.5, 0.3, 0.2, 0.8, 0.1, 0.4],  // 4->3
                vec![0.7, 0.4, 0.6],                   // 3->1
            ];
            let biases = vec![
                vec![0.1, 0.2, 0.3],  // 3 neurons
                vec![0.05],            // 1 neuron
            ];
            
            let weights_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
            let biases_refs: Vec<&[f32]> = biases.iter().map(|b| b.as_slice()).collect();
            
            let result = gpu.nn_forward(&input, &weights_refs, &biases_refs);
            
            assert_eq!(result.len(), 1);
            assert!(result[0] >= 0.0); // ReLU activation
            assert!(result[0].is_finite());
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_conv2d() {
        if let Ok(gpu) = HipGpu::new() {
            let input = generate_conv2d_input();
            let kernel = generate_conv2d_kernel();
            
            let result = gpu.conv2d(&input, &kernel, 1, 1, 4, 4, 3);
            
            // Output should be 1x1x2x2 (after 3x3 convolution on 4x4 input)
            assert_eq!(result.len(), 4);
            assert!(result.iter().all(|&x| x.is_finite()));
            
            println!("Conv2d result: {:?}", result);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_reduce_sum() {
        if let Ok(gpu) = HipGpu::new() {
            let data = vec![1.0; 1024];
            let sum = gpu.reduce_sum(&data);
            
            assert!((sum - 1024.0).abs() < 0.001);
            
            // Test with different data
            let data2: Vec<f32> = (1..=100).map(|i| i as f32).collect();
            let sum2 = gpu.reduce_sum(&data2);
            let expected_sum = 100.0 * 101.0 / 2.0; // Sum of 1 to 100
            assert!((sum2 - expected_sum).abs() < 0.1);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_wavefront_optimization() {
        if let Ok(gpu) = HipGpu::new() {
            let info = gpu.get_device_info();
            let wavefront_size = info.wavefront_size as usize;
            
            // Test data size aligned to wavefront
            let aligned_size = wavefront_size * 4;
            let data = vec![1.0; aligned_size];
            
            let start = std::time::Instant::now();
            let sum = gpu.reduce_sum(&data);
            let duration = start.elapsed();
            
            assert!((sum - aligned_size as f32).abs() < 0.001);
            println!("Wavefront-aligned reduction took: {:?}", duration);
            
            // Test unaligned data
            let unaligned_size = wavefront_size * 4 + wavefront_size / 2;
            let data_unaligned = vec![1.0; unaligned_size];
            let sum_unaligned = gpu.reduce_sum(&data_unaligned);
            assert!((sum_unaligned - unaligned_size as f32).abs() < 0.001);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_memory_allocation() {
        if let Ok(gpu) = HipGpu::new() {
            // Test large memory allocation
            let large_size = 1_000_000; // 1M floats = 4MB
            let data = generate_random_matrix(large_size);
            
            // This should test internal memory allocation
            let start = std::time::Instant::now();
            let sum = gpu.reduce_sum(&data);
            let duration = start.elapsed();
            
            assert!(sum.is_finite());
            assert!(sum > 0.0);
            println!("Large reduction ({}MB) took: {:?}", large_size * 4 / 1_000_000, duration);
        }
    }
    
    #[test]
    fn test_hip_kernel_constants() {
        // Verify HIP kernel source code
        let matmul_kernel = MATMUL_KERNEL_HIP;
        assert!(matmul_kernel.contains("extern \"C\" __global__"));
        assert!(matmul_kernel.contains("matmul_kernel_hip"));
        assert!(matmul_kernel.contains("__shared__ float As"));
        assert!(matmul_kernel.contains("__fmaf_rn"));
        
        let conv2d_kernel = CONV2D_KERNEL_HIP;
        assert!(conv2d_kernel.contains("conv2d_kernel_hip"));
        assert!(conv2d_kernel.contains("#pragma unroll"));
        
        let reduction_kernel = REDUCTION_KERNEL_HIP;
        assert!(reduction_kernel.contains("reduce_sum_kernel_hip"));
        assert!(reduction_kernel.contains("atomicAdd"));
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_concurrent_operations() {
        if let Ok(gpu) = HipGpu::new() {
            let gpu = Arc::new(gpu);
            let mut handles = vec![];
            
            for i in 0..3 {
                let gpu_clone = Arc::clone(&gpu);
                let handle = std::thread::spawn(move || {
                    let data = vec![i as f32 + 1.0; 1000];
                    let sum = gpu_clone.reduce_sum(&data);
                    assert!((sum - 1000.0 * (i as f32 + 1.0)).abs() < 0.001);
                    println!("Thread {} completed with sum: {}", i, sum);
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().expect("Thread panicked");
            }
            
            println!("✅ Concurrent HIP operations completed");
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_edge_cases() {
        if let Ok(gpu) = HipGpu::new() {
            // Test empty data (should handle gracefully)
            let empty_data: Vec<f32> = vec![];
            let sum_empty = gpu.reduce_sum(&empty_data);
            assert_eq!(sum_empty, 0.0);
            
            // Test single element
            let single = vec![42.0];
            let sum_single = gpu.reduce_sum(&single);
            assert!((sum_single - 42.0).abs() < 0.001);
            
            // Test all zeros
            let zeros = vec![0.0; 1000];
            let sum_zeros = gpu.reduce_sum(&zeros);
            assert!((sum_zeros - 0.0).abs() < 0.001);
            
            // Test negative values
            let negatives = vec![-1.0; 500];
            let sum_neg = gpu.reduce_sum(&negatives);
            assert!((sum_neg - (-500.0)).abs() < 0.001);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_precision_limits() {
        if let Ok(gpu) = HipGpu::new() {
            // Test with very small numbers
            let small_numbers = vec![1e-6; 1000];
            let sum_small = gpu.reduce_sum(&small_numbers);
            assert!((sum_small - 1e-3).abs() < 1e-4);
            
            // Test with large numbers
            let large_numbers = vec![1e6; 100];
            let sum_large = gpu.reduce_sum(&large_numbers);
            assert!((sum_large - 1e8).abs() < 1e5);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_matmul_precision() {
        if let Ok(gpu) = HipGpu::new() {
            // Test matrix multiplication with known precise result
            let a = vec![
                0.1, 0.2, 0.3,
                0.4, 0.5, 0.6,
                0.7, 0.8, 0.9
            ]; // 3x3
            let b = vec![
                1.0, 2.0, 3.0,
                4.0, 5.0, 6.0,
                7.0, 8.0, 9.0
            ]; // 3x3
            
            let result = gpu.matmul(&a, &b, 3, 3, 3);
            assert_eq!(result.len(), 9);
            
            // First element: 0.1*1 + 0.2*4 + 0.3*7 = 0.1 + 0.8 + 2.1 = 3.0
            assert!((result[0] - 3.0).abs() < 0.0001);
            
            // Check all results are finite and reasonable
            for &val in &result {
                assert!(val.is_finite());
                assert!(val > 0.0); // All positive inputs should give positive results
            }
        }
    }
    
    #[test]
    fn test_hip_error_handling() {
        // Test error cases
        match HipGpu::new() {
            Ok(_) => println!("HIP GPU available"),
            Err(e) => {
                let error_msg = e.to_lowercase();
                assert!(
                    error_msg.contains("hip") || 
                    error_msg.contains("amd") ||
                    error_msg.contains("device") ||
                    error_msg.contains("failed"),
                    "Error message should be descriptive: {}", e
                );
            }
        }
    }
    
    #[test]
    fn test_hip_drop_cleanup() {
        if let Ok(gpu) = HipGpu::new() {
            drop(gpu);
            println!("✅ HIP GPU cleanup successful");
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_performance_characteristics() {
        if let Ok(gpu) = HipGpu::new() {
            let info = gpu.get_device_info();
            
            // Test different workload sizes to characterize performance
            let sizes = vec![8, 16, 32, 64];
            for size in sizes {
                let data = vec![1.0; size * size];
                
                let start = std::time::Instant::now();
                let sum = gpu.reduce_sum(&data);
                let duration = start.elapsed();
                
                assert!((sum - (size * size) as f32).abs() < 0.001);
                println!("Size {}: {:?}, Throughput: {:.2} GFLOPS", 
                        size, duration, 
                        (size * size) as f64 / duration.as_nanos() as f64 * 1e9 / 1e9);
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_hip_memory_patterns() {
        if let Ok(gpu) = HipGpu::new() {
            // Test different memory access patterns
            
            // Sequential pattern
            let sequential: Vec<f32> = (0..10000).map(|i| i as f32).collect();
            let sum_seq = gpu.reduce_sum(&sequential);
            let expected_seq = 10000.0 * 9999.0 / 2.0;
            assert!((sum_seq - expected_seq).abs() < 1.0);
            
            // Alternating pattern
            let alternating: Vec<f32> = (0..10000).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
            let sum_alt = gpu.reduce_sum(&alternating);
            assert!(sum_alt.abs() < 1.0); // Should be close to 0
            
            // Random-like pattern (using simple PRNG)
            let mut random_like = Vec::new();
            let mut seed = 12345u32;
            for _ in 0..10000 {
                seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
                random_like.push((seed % 1000) as f32 / 1000.0);
            }
            let sum_rand = gpu.reduce_sum(&random_like);
            assert!(sum_rand > 0.0 && sum_rand < 10000.0);
        }
    }
}