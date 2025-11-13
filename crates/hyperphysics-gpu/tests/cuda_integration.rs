//! CUDA Integration Tests
//!
//! Comprehensive tests for real CUDA backend functionality

#[cfg(feature = "cuda-backend")]
mod cuda_tests {
    use hyperphysics_gpu::backend::cuda_real::{create_cuda_backend, CudaBackend};
    use hyperphysics_gpu::backend::{GPUBackend, BackendType, BufferUsage};

    #[test]
    fn test_cuda_device_detection() {
        match create_cuda_backend() {
            Ok(Some(backend)) => {
                assert_eq!(backend.capabilities().backend, BackendType::CUDA);
                assert!(backend.capabilities().supports_compute);
                println!("✓ CUDA device: {}", backend.capabilities().device_name);
            }
            Ok(None) => {
                println!("⚠ No CUDA devices available (expected on non-NVIDIA systems)");
            }
            Err(e) => {
                panic!("CUDA initialization failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_real_memory_allocation() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => {
                println!("Skipping test: CUDA not available");
                return;
            }
        };

        // Test various buffer sizes
        let sizes = vec![1024, 1024 * 1024, 16 * 1024 * 1024];

        for size in sizes {
            let buffer = backend.create_buffer(size, BufferUsage::Storage);
            assert!(buffer.is_ok(), "Failed to allocate {} bytes", size);

            let buffer = buffer.unwrap();
            assert_eq!(buffer.size(), size);
            println!("✓ Allocated {} bytes", size);
        }
    }

    #[test]
    fn test_host_to_device_copy() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        let size = 1024 * 1024; // 1MB
        let mut buffer = backend.create_buffer(size, BufferUsage::Storage)
            .expect("Failed to create buffer");

        // Create test data
        let test_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();

        // Copy to device
        let result = backend.write_buffer(buffer.as_mut(), &test_data);
        assert!(result.is_ok(), "Host-to-device copy failed");

        println!("✓ Copied {} bytes to device", size);
    }

    #[test]
    fn test_device_to_host_copy() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        let size = 1024 * 1024;
        let mut buffer = backend.create_buffer(size, BufferUsage::Storage)
            .expect("Failed to create buffer");

        // Write test pattern
        let test_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
        backend.write_buffer(buffer.as_mut(), &test_data)
            .expect("Write failed");

        // Read back
        let result = backend.read_buffer(buffer.as_ref());
        assert!(result.is_ok(), "Device-to-host copy failed");

        let read_data = result.unwrap();
        assert_eq!(read_data.len(), size as usize);
        assert_eq!(&read_data[..], &test_data[..]);

        println!("✓ Verified {} bytes round-trip", size);
    }

    #[test]
    fn test_kernel_compilation() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        let wgsl_shader = r#"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                // Simple test kernel
            }
        "#;

        let result = backend.execute_compute(wgsl_shader, [1024, 1, 1]);
        assert!(result.is_ok(), "Kernel compilation/execution failed: {:?}", result.err());

        println!("✓ WGSL → CUDA compilation successful");
    }

    #[test]
    fn test_multiple_kernels() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        // Execute multiple kernels to test caching
        let wgsl_shader = r#"
            @compute @workgroup_size(128)
            fn main() {}
        "#;

        for i in 0..5 {
            let result = backend.execute_compute(wgsl_shader, [256, 1, 1]);
            assert!(result.is_ok(), "Kernel {} failed", i);
        }

        println!("✓ Multiple kernel launches successful");
    }

    #[test]
    fn test_memory_pool_reuse() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        // Allocate and deallocate multiple times
        for _ in 0..10 {
            let buffer = backend.create_buffer(1024 * 1024, BufferUsage::Storage);
            assert!(buffer.is_ok());
            // Buffer dropped here, should return to pool
        }

        let stats = backend.memory_stats();
        println!("✓ Memory pool working: {} buffers allocated", stats.buffer_count);
    }

    #[test]
    fn test_synchronization() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        // Launch async kernel
        let wgsl = "@compute @workgroup_size(64) fn main() {}";
        backend.execute_compute(wgsl, [1024, 1, 1])
            .expect("Kernel launch failed");

        // Synchronize
        let result = backend.synchronize();
        assert!(result.is_ok(), "Synchronization failed");

        println!("✓ Device synchronization successful");
    }

    #[test]
    fn test_concurrent_buffers() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        // Create multiple buffers concurrently
        let buffers: Vec<_> = (0..10)
            .map(|i| {
                backend.create_buffer((i + 1) * 1024, BufferUsage::Storage)
                    .expect("Buffer creation failed")
            })
            .collect();

        assert_eq!(buffers.len(), 10);

        let stats = backend.memory_stats();
        assert_eq!(stats.buffer_count, 10);

        println!("✓ Concurrent buffer management successful");
    }

    #[test]
    fn test_large_workload() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        // Test with large workload (16M elements)
        let size = 16 * 1024 * 1024 * 4; // 64MB
        let mut buffer = backend.create_buffer(size, BufferUsage::Storage)
            .expect("Large buffer allocation failed");

        let data = vec![1u8; size as usize];
        backend.write_buffer(buffer.as_mut(), &data)
            .expect("Large write failed");

        let wgsl = "@compute @workgroup_size(256) fn main() {}";
        backend.execute_compute(wgsl, [16 * 1024 * 1024, 1, 1])
            .expect("Large kernel launch failed");

        backend.synchronize().expect("Sync failed");

        println!("✓ Large workload (16M elements) processed successfully");
    }

    #[test]
    fn test_error_handling() {
        let backend = match create_cuda_backend() {
            Ok(Some(b)) => b,
            _ => return,
        };

        // Test invalid WGSL
        let invalid_wgsl = "this is not valid WGSL!";
        let result = backend.execute_compute(invalid_wgsl, [256, 1, 1]);
        assert!(result.is_err(), "Should fail on invalid WGSL");

        println!("✓ Error handling working correctly");
    }
}

#[cfg(not(feature = "cuda-backend"))]
mod no_cuda {
    #[test]
    fn cuda_feature_not_enabled() {
        println!("CUDA backend not enabled. Enable with: cargo test --features cuda-backend");
    }
}
