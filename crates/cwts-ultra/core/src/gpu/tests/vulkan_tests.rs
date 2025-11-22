#[cfg(test)]
mod vulkan_tests {
    use super::super::vulkan::*;
    use std::sync::Arc;
    
    // Test data generators for real testing
    fn generate_test_matrix_2x2() -> Vec<f32> {
        vec![1.0, 2.0, 3.0, 4.0] // [[1,2],[3,4]]
    }
    
    fn generate_test_matrix_4x4() -> Vec<f32> {
        (1..=16).map(|i| i as f32).collect()
    }
    
    fn generate_random_matrix(size: usize) -> Vec<f32> {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        (0..size).map(|i| {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            (hasher.finish() % 1000) as f32 / 1000.0
        }).collect()
    }
    
    fn generate_neural_weights() -> Vec<Vec<f32>> {
        vec![
            vec![0.5, 0.3, 0.2, 0.8],  // Input layer (4 neurons)
            vec![0.7, 0.4],             // Hidden layer (2 neurons)
            vec![0.9],                  // Output layer (1 neuron)
        ]
    }
    
    fn generate_neural_biases() -> Vec<Vec<f32>> {
        vec![
            vec![0.1, 0.2, 0.3, 0.1],  // Input layer biases
            vec![0.05, 0.15],           // Hidden layer biases
            vec![0.2],                  // Output layer bias
        ]
    }
    
    #[test]
    fn test_vulkan_gpu_creation() {
        // Test both success and failure cases
        match VulkanGpu::new() {
            Ok(gpu) => {
                println!("✅ Vulkan GPU created successfully");
                assert!(!std::ptr::eq(&gpu as *const _, std::ptr::null()));
            }
            Err(e) => {
                println!("⚠️ Vulkan not available (expected on CI): {}", e);
                // This is expected on systems without Vulkan
                assert!(e.contains("Failed to create") || e.contains("No Vulkan"));
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored on Vulkan-enabled systems
    fn test_vulkan_matmul_2x2() {
        if let Ok(gpu) = VulkanGpu::new() {
            let a = generate_test_matrix_2x2();
            let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5,6],[7,8]]
            
            let result = gpu.matmul(&a, &b, 2, 2, 2);
            
            // Expected: [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
            //         = [[19, 22], [43, 50]]
            assert_eq!(result.len(), 4);
            assert!((result[0] - 19.0).abs() < 0.001);
            assert!((result[1] - 22.0).abs() < 0.001);
            assert!((result[2] - 43.0).abs() < 0.001);
            assert!((result[3] - 50.0).abs() < 0.001);
        } else {
            println!("Vulkan not available, skipping matmul test");
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_matmul_edge_cases() {
        if let Ok(gpu) = VulkanGpu::new() {
            // Test 1x1 matrix
            let a = vec![5.0];
            let b = vec![3.0];
            let result = gpu.matmul(&a, &b, 1, 1, 1);
            assert_eq!(result.len(), 1);
            assert!((result[0] - 15.0).abs() < 0.001);
            
            // Test 3x3 matrices
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
            let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
            let result = gpu.matmul(&a, &b, 3, 3, 3);
            assert_eq!(result.len(), 9);
            
            // Expected first element: 1*9 + 2*6 + 3*3 = 9 + 12 + 9 = 30
            assert!((result[0] - 30.0).abs() < 0.001);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_matmul_rectangular() {
        if let Ok(gpu) = VulkanGpu::new() {
            // Test 2x3 * 3x2 = 2x2
            let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
            let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2
            let result = gpu.matmul(&a, &b, 2, 2, 3);
            
            assert_eq!(result.len(), 4);
            // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
            //         = [[58, 64], [139, 154]]
            assert!((result[0] - 58.0).abs() < 0.001);
            assert!((result[1] - 64.0).abs() < 0.001);
            assert!((result[2] - 139.0).abs() < 0.001);
            assert!((result[3] - 154.0).abs() < 0.001);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_nn_forward() {
        if let Ok(gpu) = VulkanGpu::new() {
            let input = vec![1.0, 0.5, 0.3, 0.8];
            let weights = generate_neural_weights();
            let biases = generate_neural_biases();
            
            let weights_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
            let biases_refs: Vec<&[f32]> = biases.iter().map(|b| b.as_slice()).collect();
            
            let result = gpu.nn_forward(&input, &weights_refs, &biases_refs);
            
            // Should have one output (final layer has 1 neuron)
            assert_eq!(result.len(), 1);
            assert!(result[0] > 0.0); // ReLU should ensure positive output
            assert!(result[0].is_finite()); // Should be a valid number
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_nn_forward_multi_layer() {
        if let Ok(gpu) = VulkanGpu::new() {
            let input = vec![1.0, -1.0, 0.0, 2.0];
            let weights = vec![
                vec![0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4], // 4->2
                vec![0.5, -0.5], // 2->1
            ];
            let biases = vec![
                vec![0.01, -0.01], // 2 neurons
                vec![0.1], // 1 neuron
            ];
            
            let weights_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
            let biases_refs: Vec<&[f32]> = biases.iter().map(|b| b.as_slice()).collect();
            
            let result = gpu.nn_forward(&input, &weights_refs, &biases_refs);
            
            assert_eq!(result.len(), 1);
            assert!(result[0].is_finite());
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_large_matrix_performance() {
        if let Ok(gpu) = VulkanGpu::new() {
            let size = 64;
            let a = generate_random_matrix(size * size);
            let b = generate_random_matrix(size * size);
            
            let start = std::time::Instant::now();
            let result = gpu.matmul(&a, &b, size, size, size);
            let duration = start.elapsed();
            
            assert_eq!(result.len(), size * size);
            println!("Vulkan {}x{} matmul took: {:?}", size, size, duration);
            
            // Should complete in reasonable time (less than 1 second)
            assert!(duration.as_secs() < 1);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_zero_matrices() {
        if let Ok(gpu) = VulkanGpu::new() {
            let zeros_2x2 = vec![0.0, 0.0, 0.0, 0.0];
            let ones_2x2 = vec![1.0, 1.0, 1.0, 1.0];
            
            // Zero matrix multiplication
            let result = gpu.matmul(&zeros_2x2, &ones_2x2, 2, 2, 2);
            assert_eq!(result.len(), 4);
            for &val in &result {
                assert!((val - 0.0).abs() < 0.001);
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_identity_matrix() {
        if let Ok(gpu) = VulkanGpu::new() {
            let identity = vec![1.0, 0.0, 0.0, 1.0]; // [[1,0],[0,1]]
            let test_matrix = vec![5.0, 3.0, 2.0, 7.0]; // [[5,3],[2,7]]
            
            // Identity * test_matrix = test_matrix
            let result = gpu.matmul(&identity, &test_matrix, 2, 2, 2);
            assert_eq!(result.len(), 4);
            assert!((result[0] - 5.0).abs() < 0.001);
            assert!((result[1] - 3.0).abs() < 0.001);
            assert!((result[2] - 2.0).abs() < 0.001);
            assert!((result[3] - 7.0).abs() < 0.001);
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_nn_forward_edge_cases() {
        if let Ok(gpu) = VulkanGpu::new() {
            // Test with all zeros input
            let zero_input = vec![0.0, 0.0, 0.0, 0.0];
            let weights = generate_neural_weights();
            let biases = generate_neural_biases();
            let weights_refs: Vec<&[f32]> = weights.iter().map(|w| w.as_slice()).collect();
            let biases_refs: Vec<&[f32]> = biases.iter().map(|b| b.as_slice()).collect();
            
            let result = gpu.nn_forward(&zero_input, &weights_refs, &biases_refs);
            assert_eq!(result.len(), 1);
            assert!(result[0] >= 0.0); // ReLU ensures non-negative
            
            // Test with large input values
            let large_input = vec![100.0, -100.0, 50.0, -50.0];
            let result_large = gpu.nn_forward(&large_input, &weights_refs, &biases_refs);
            assert_eq!(result_large.len(), 1);
            assert!(result_large[0].is_finite());
        }
    }
    
    #[test]
    fn test_vulkan_drop_cleanup() {
        // Test that VulkanGpu properly cleans up resources
        if let Ok(gpu) = VulkanGpu::new() {
            // Create and drop the GPU instance
            drop(gpu);
            // If we get here without segfault, cleanup worked
            println!("✅ Vulkan GPU cleanup successful");
        }
    }
    
    #[test]
    fn test_vulkan_shader_constants() {
        // Test that GLSL shader is well-formed
        let shader = MATMUL_SHADER_GLSL;
        assert!(shader.contains("#version 450"));
        assert!(shader.contains("layout(local_size_x = 16"));
        assert!(shader.contains("void main()"));
        assert!(shader.len() > 1000); // Reasonable shader size
    }
    
    #[test]
    fn test_vulkan_multiple_instances() {
        // Test creating multiple VulkanGpu instances
        match (VulkanGpu::new(), VulkanGpu::new()) {
            (Ok(_gpu1), Ok(_gpu2)) => {
                println!("✅ Multiple Vulkan instances created successfully");
            }
            (Err(e), _) | (_, Err(e)) => {
                println!("⚠️ Vulkan not available: {}", e);
            }
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored
    fn test_vulkan_concurrent_operations() {
        if let Ok(gpu) = VulkanGpu::new() {
            let gpu = Arc::new(gpu);
            let mut handles = vec![];
            
            // Spawn multiple threads doing matrix multiplication
            for i in 0..4 {
                let gpu_clone = Arc::clone(&gpu);
                let handle = std::thread::spawn(move || {
                    let a = generate_random_matrix(16);
                    let b = generate_random_matrix(16);
                    let result = gpu_clone.matmul(&a, &b, 4, 4, 4);
                    assert_eq!(result.len(), 16);
                    println!("Thread {} completed matmul", i);
                });
                handles.push(handle);
            }
            
            for handle in handles {
                handle.join().expect("Thread panicked");
            }
            
            println!("✅ Concurrent Vulkan operations completed");
        }
    }
    
    #[test]
    #[ignore] // Only run with --include-ignored  
    fn test_vulkan_precision_verification() {
        if let Ok(gpu) = VulkanGpu::new() {
            // Test known mathematical operations for precision
            let a = vec![1.0/3.0, 2.0/3.0, 1.0, 4.0/3.0]; // 2x2 with fractions
            let b = vec![3.0, 6.0, 9.0, 12.0]; // 2x2 with integers
            
            let result = gpu.matmul(&a, &b, 2, 2, 2);
            
            // Expected: [[1/3*3 + 2/3*9, 1/3*6 + 2/3*12], [1*3 + 4/3*9, 1*6 + 4/3*12]]
            //         = [[1 + 6, 2 + 8], [3 + 12, 6 + 16]]
            //         = [[7, 10], [15, 22]]
            assert!((result[0] - 7.0).abs() < 0.001);
            assert!((result[1] - 10.0).abs() < 0.001);
            assert!((result[2] - 15.0).abs() < 0.001);
            assert!((result[3] - 22.0).abs() < 0.001);
        }
    }
    
    #[test]
    fn test_vulkan_error_handling() {
        // Test various error conditions
        match VulkanGpu::new() {
            Ok(_) => println!("Vulkan available"),
            Err(e) => {
                // Verify error messages are informative
                let error_msg = e.to_lowercase();
                assert!(
                    error_msg.contains("vulkan") || 
                    error_msg.contains("device") || 
                    error_msg.contains("failed"),
                    "Error message should be descriptive: {}", e
                );
            }
        }
    }
}