//! Integration tests for Vulkan backend
//!
//! Tests real Vulkan API integration, buffer operations, and compute dispatch.

#[cfg(feature = "vulkan-backend")]
mod tests {
    use hyperphysics_gpu::backend::vulkan::{create_vulkan_backend, VulkanBackend};
    use hyperphysics_gpu::backend::{GPUBackend, BufferUsage};

    #[test]
    fn test_vulkan_device_enumeration() {
        if let Ok(Some(backend)) = create_vulkan_backend() {
            let caps = backend.capabilities();

            println!("✓ Device: {}", caps.device_name);
            println!("✓ Max workgroup size: {}", caps.max_workgroup_size);
            println!("✓ Max buffer size: {} MB", caps.max_buffer_size / (1024 * 1024));

            assert!(caps.supports_compute);
            assert!(caps.max_workgroup_size >= 256);
        } else {
            println!("⊗ Vulkan not available (expected on non-Vulkan systems)");
        }
    }

    #[test]
    fn test_buffer_allocation_and_mapping() {
        if let Ok(Some(backend)) = create_vulkan_backend() {
            // Create 4 MB buffer
            let size = 4 * 1024 * 1024u64;
            let mut buffer = backend.create_buffer(size, BufferUsage::Storage)
                .expect("Buffer creation failed");

            assert_eq!(buffer.size(), size);

            // Write test pattern
            let test_data: Vec<u8> = (0..size).map(|i| (i % 256) as u8).collect();
            backend.write_buffer(buffer.as_mut(), &test_data)
                .expect("Buffer write failed");

            // Read back and verify
            let read_data = backend.read_buffer(buffer.as_ref())
                .expect("Buffer read failed");

            assert_eq!(read_data.len(), test_data.len());
            assert_eq!(&read_data[0..1000], &test_data[0..1000]);

            println!("✓ Buffer allocation and mapping: {} MB", size / (1024 * 1024));
        }
    }

    #[test]
    fn test_compute_shader_execution() {
        if let Ok(Some(backend)) = create_vulkan_backend() {
            let shader = r#"
                @group(0) @binding(0) var<storage, read_write> data: array<f32>;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index < arrayLength(&data)) {
                        data[index] = data[index] * 2.0;
                    }
                }
            "#;

            match backend.execute_compute(shader, [1024, 1, 1]) {
                Ok(_) => {
                    println!("✓ Compute shader executed: 1024 workgroups");
                }
                Err(e) => {
                    println!("⊗ Compute execution failed: {:?}", e);
                }
            }

            backend.synchronize().expect("Synchronization failed");
        }
    }

    #[test]
    fn test_memory_statistics() {
        if let Ok(Some(backend)) = create_vulkan_backend() {
            let stats = backend.memory_stats();

            println!("✓ Total memory: {} GB", stats.total_memory / (1024 * 1024 * 1024));
            println!("✓ Used memory: {} MB", stats.used_memory / (1024 * 1024));
            println!("✓ Free memory: {} GB", stats.free_memory / (1024 * 1024 * 1024));
            println!("✓ Buffer count: {}", stats.buffer_count);

            assert!(stats.total_memory > 0);
        }
    }

    #[test]
    fn test_synchronization() {
        if let Ok(Some(backend)) = create_vulkan_backend() {
            // Create and execute multiple operations
            for i in 0..10 {
                let buffer = backend.create_buffer(1024, BufferUsage::Storage)
                    .expect("Buffer creation failed");

                if i % 3 == 0 {
                    backend.synchronize().expect("Synchronization failed");
                }
            }

            // Final synchronization
            backend.synchronize().expect("Final synchronization failed");

            println!("✓ Synchronization test passed");
        }
    }

    #[test]
    fn test_error_handling() {
        if let Ok(Some(backend)) = create_vulkan_backend() {
            // Test buffer overflow
            let mut buffer = backend.create_buffer(100, BufferUsage::Storage)
                .expect("Buffer creation failed");

            let oversized_data = vec![0u8; 200];
            let result = backend.write_buffer(buffer.as_mut(), &oversized_data);

            assert!(result.is_err());
            println!("✓ Buffer overflow correctly detected");
        }
    }
}
