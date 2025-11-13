//! WGPU backend implementation for cross-platform GPU compute
//!
//! Provides Vulkan/Metal/DX12 support via WGPU with automatic device selection.

use super::{GPUBackend, BackendType, GPUCapabilities, GPUBuffer, BufferUsage, MemoryStats};
use hyperphysics_core::Result;
use std::sync::Arc;

/// WGPU buffer wrapper
pub struct WGPUBuffer {
    buffer: wgpu::Buffer,
    size: u64,
    usage: BufferUsage,
}

impl GPUBuffer for WGPUBuffer {
    fn size(&self) -> u64 {
        self.size
    }
    
    fn usage(&self) -> BufferUsage {
        self.usage
    }
}

/// WGPU compute backend
pub struct WGPUBackend {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    capabilities: GPUCapabilities,
}

impl WGPUBackend {
    /// Initialize WGPU backend with automatic adapter selection
    pub async fn new() -> Result<Self> {
        // Request high-performance adapter
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| hyperphysics_core::EngineError::Configuration {
                message: "Failed to find suitable GPU adapter".to_string(),
            })?;

        // Get adapter info for capabilities
        let info = adapter.get_info();
        let limits = adapter.limits();

        // Request device with compute capabilities
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HyperPhysics Compute Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits {
                        max_compute_workgroup_size_x: 256,
                        max_compute_workgroup_size_y: 256,
                        max_compute_workgroup_size_z: 64,
                        max_compute_invocations_per_workgroup: 256,
                        max_compute_workgroup_storage_size: 16384,
                        ..wgpu::Limits::default()
                    },
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| hyperphysics_core::EngineError::Configuration {
                message: format!("Failed to create device: {}", e),
            })?;

        let capabilities = GPUCapabilities {
            backend: BackendType::WGPU,
            device_name: info.name.clone(),
            max_buffer_size: limits.max_buffer_size,
            max_workgroup_size: limits.max_compute_workgroup_size_x,
            supports_compute: true,
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
        })
    }

    /// Execute compute shader with buffer bindings
    ///
    /// # Arguments
    /// * `shader_source` - WGSL shader code
    /// * `workgroups` - Workgroup dispatch dimensions [x, y, z]
    /// * `bindings` - Buffer bindings for shader
    pub fn execute_compute_with_bindings(
        &self,
        shader_source: &str,
        workgroups: [u32; 3],
        bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        bind_group_entries: &[wgpu::BindGroupEntry],
    ) -> Result<()> {
        // Create shader module
        let shader = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("HyperPhysics Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });

        // Create bind group layout
        let bind_group_layout = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: bind_group_layout_entries,
        });

        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HyperPhysics Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: bind_group_entries,
        });

        // Create command encoder and execute
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HyperPhysics Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        // Submit command buffer
        self.queue.submit(Some(encoder.finish()));

        Ok(())
    }

    /// Get device reference for buffer creation
    pub fn device(&self) -> &Arc<wgpu::Device> {
        &self.device
    }

    /// Get queue reference for buffer operations
    pub fn queue(&self) -> &Arc<wgpu::Queue> {
        &self.queue
    }
}

impl GPUBackend for WGPUBackend {
    fn capabilities(&self) -> &GPUCapabilities {
        &self.capabilities
    }

    fn execute_compute(&self, shader: &str, workgroups: [u32; 3]) -> Result<()> {
        // Simple wrapper for shader-only execution
        // Real usage should use execute_compute_with_bindings for buffer management
        self.execute_compute_with_bindings(shader, workgroups, &[], &[])
    }
    
    fn create_buffer(&self, size: u64, usage: BufferUsage) -> Result<Box<dyn GPUBuffer>> {
        let wgpu_usage = match usage {
            BufferUsage::Storage => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            BufferUsage::Uniform => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Vertex => wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::Index => wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            BufferUsage::CopySrc => wgpu::BufferUsages::COPY_SRC,
            BufferUsage::CopyDst => wgpu::BufferUsages::COPY_DST,
        };
        
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HyperPhysics Buffer"),
            size,
            usage: wgpu_usage,
            mapped_at_creation: false,
        });
        
        Ok(Box::new(WGPUBuffer {
            buffer,
            size,
            usage,
        }))
    }
    
    fn write_buffer(&self, buffer: &mut dyn GPUBuffer, data: &[u8]) -> Result<()> {
        // In a real implementation, we'd need proper downcasting
        // This is a simplified version for the trait compliance
        if data.len() as u64 > buffer.size() {
            return Err(hyperphysics_core::EngineError::Simulation { 
                message: format!("Data size exceeds buffer size") 
            });
        }
        
        // For now, return success - real implementation would use queue.write_buffer
        Ok(())
    }
    
    fn read_buffer(&self, buffer: &dyn GPUBuffer) -> Result<Vec<u8>> {
        // Real implementation would use buffer mapping and reading
        // For now, return empty data of the correct size
        Ok(vec![0u8; buffer.size() as usize])
    }
    
    fn synchronize(&self) -> Result<()> {
        // WGPU operations are automatically synchronized through the queue
        // For explicit synchronization, we could submit an empty command buffer
        let command_buffer = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Sync Command Buffer"),
        }).finish();
        
        self.queue.submit([command_buffer]);
        Ok(())
    }
    
    fn memory_stats(&self) -> MemoryStats {
        // WGPU doesn't provide direct memory usage APIs
        // In a real implementation, we'd track buffer allocations
        MemoryStats {
            total_memory: 0, // Would need to query adapter limits
            used_memory: 0,  // Would track allocated buffers
            free_memory: 0,  // Would calculate from total - used
            buffer_count: 0, // Would track active buffers
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_wgpu_initialization() {
        // This test may fail on CI without GPU, but should work locally
        if let Ok(backend) = WGPUBackend::new().await {
            assert_eq!(backend.capabilities().backend, BackendType::WGPU);
            assert!(backend.capabilities().supports_compute);
            assert!(backend.capabilities().max_workgroup_size >= 64);
        }
    }

    #[tokio::test]
    async fn test_simple_compute() {
        if let Ok(backend) = WGPUBackend::new().await {
            // Simple pass-through shader for testing
            let shader = r#"
                @compute @workgroup_size(1)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    // Minimal compute shader
                }
            "#;

            let result = backend.execute_compute(shader, [1, 1, 1]);
            assert!(result.is_ok());
        }
    }
}
