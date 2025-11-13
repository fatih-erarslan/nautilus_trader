//! GPU Random Number Generation
//!
//! High-performance parallel RNG using Xorshift128+ algorithm.
//! Provides statistically sound random numbers for Monte Carlo simulations.

use crate::backend::wgpu::WGPUBackend;
use crate::kernels::RNG_SHADER;
use hyperphysics_core::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// RNG state for Xorshift128+ (64 bits split into two u32)
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RNGState {
    pub s0: u32,
    pub s1: u32,
}

/// RNG parameters for compute shaders
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct RNGParams {
    pub n_values: u32,
    pub iteration: u32,
}

/// GPU Random Number Generator
pub struct GPURng {
    backend: Arc<WGPUBackend>,
    state_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    n_generators: usize,
    iteration: u32,
}

impl GPURng {
    /// Create new GPU RNG with specified number of parallel generators
    pub fn new(backend: Arc<WGPUBackend>, n_generators: usize) -> Result<Self> {
        // Create state buffer (one RNGState per generator)
        let state_buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("RNG State Buffer"),
            size: (n_generators * std::mem::size_of::<RNGState>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create output buffer for random values
        let output_buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("RNG Output Buffer"),
            size: (n_generators * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create params buffer
        let params_buffer = backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("RNG Params Buffer"),
            size: std::mem::size_of::<RNGParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            backend,
            state_buffer,
            output_buffer,
            params_buffer,
            n_generators,
            iteration: 0,
        })
    }

    /// Initialize RNG with seed
    pub fn seed(&mut self, seed: u32) -> Result<()> {
        // Update params with seed
        let params = RNGParams {
            n_values: self.n_generators as u32,
            iteration: seed,
        };

        self.backend.queue().write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        // Create shader module
        let shader_module = self.backend.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RNG Seed Shader"),
            source: wgpu::ShaderSource::Wgsl(RNG_SHADER.into()),
        });

        // Create bind group layout
        let bind_group_layout = self.backend.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RNG Seed Bind Group Layout"),
            entries: &[
                // @binding(0) rng_states
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1) random_output
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2) params
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = self.backend.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RNG Seed Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.backend.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RNG Seed Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "seed_rng",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = self.backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RNG Seed Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute seeding kernel
        let workgroups = ((self.n_generators + 255) / 256) as u32;

        let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RNG Seed Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RNG Seed Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.backend.queue().submit(Some(encoder.finish()));

        Ok(())
    }

    /// Generate uniform random values in [0, 1)
    pub fn generate_uniform(&mut self) -> Result<()> {
        self.iteration += 1;

        // Update params
        let params = RNGParams {
            n_values: self.n_generators as u32,
            iteration: self.iteration,
        };

        self.backend.queue().write_buffer(
            &self.params_buffer,
            0,
            bytemuck::cast_slice(&[params]),
        );

        // Create shader module
        let shader_module = self.backend.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RNG Uniform Shader"),
            source: wgpu::ShaderSource::Wgsl(RNG_SHADER.into()),
        });

        // Create bind group layout (same as seed)
        let bind_group_layout = self.backend.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RNG Uniform Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout = self.backend.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RNG Uniform Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create compute pipeline
        let pipeline = self.backend.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RNG Uniform Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: "generate_uniform",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create bind group
        let bind_group = self.backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RNG Uniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute generation kernel
        let workgroups = ((self.n_generators + 255) / 256) as u32;

        let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RNG Uniform Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RNG Uniform Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.backend.queue().submit(Some(encoder.finish()));

        Ok(())
    }

    /// Get reference to output buffer (for use in other kernels)
    pub fn output_buffer(&self) -> &wgpu::Buffer {
        &self.output_buffer
    }

    /// Get reference to state buffer
    pub fn state_buffer(&self) -> &wgpu::Buffer {
        &self.state_buffer
    }

    /// Get number of parallel generators
    pub fn n_generators(&self) -> usize {
        self.n_generators
    }

    /// Read back random values to CPU (for testing)
    pub async fn read_output(&self) -> Result<Vec<f32>> {
        let staging_buffer = self.backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("RNG Output Staging Buffer"),
            size: (self.n_generators * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RNG Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.output_buffer,
            0,
            &staging_buffer,
            0,
            (self.n_generators * std::mem::size_of::<f32>()) as u64,
        );

        self.backend.queue().submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver): (
            futures::channel::oneshot::Sender<std::result::Result<(), wgpu::BufferAsyncError>>,
            _
        ) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        self.backend.device().poll(wgpu::Maintain::Wait);
        receiver.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice::<u8, f32>(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::wgpu::WGPUBackend;

    #[tokio::test]
    async fn test_rng_initialization() {
        if let Ok(backend) = WGPUBackend::new().await {
            let backend = Arc::new(backend);
            let mut rng = GPURng::new(backend, 1000).expect("Failed to create RNG");

            // Test seeding
            rng.seed(12345).expect("Failed to seed RNG");

            assert_eq!(rng.n_generators(), 1000);
        }
    }

    #[tokio::test]
    async fn test_uniform_generation() {
        if let Ok(backend) = WGPUBackend::new().await {
            let backend = Arc::new(backend);
            let mut rng = GPURng::new(backend, 1000).expect("Failed to create RNG");

            rng.seed(12345).expect("Failed to seed RNG");
            rng.generate_uniform().expect("Failed to generate random values");

            let values = rng.read_output().await.expect("Failed to read output");

            // Check all values are in [0, 1)
            for &v in &values {
                assert!(v >= 0.0 && v < 1.0, "Random value {} out of range", v);
            }

            // Check mean is roughly 0.5
            let mean: f32 = values.iter().sum::<f32>() / values.len() as f32;
            assert!((mean - 0.5).abs() < 0.05, "Mean {} too far from 0.5", mean);
        }
    }

    #[tokio::test]
    async fn test_statistical_quality() {
        if let Ok(backend) = WGPUBackend::new().await {
            let backend = Arc::new(backend);
            let mut rng = GPURng::new(backend, 10000).expect("Failed to create RNG");

            rng.seed(42).expect("Failed to seed RNG");
            rng.generate_uniform().expect("Failed to generate random values");

            let values = rng.read_output().await.expect("Failed to read output");

            // Compute statistics
            let n = values.len() as f32;
            let mean: f32 = values.iter().sum::<f32>() / n;
            let variance: f32 = values.iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>() / n;

            // For uniform [0,1): mean = 0.5, variance = 1/12 ≈ 0.0833
            assert!((mean - 0.5).abs() < 0.01, "Mean {} not close to 0.5", mean);
            assert!((variance - 1.0/12.0).abs() < 0.01, "Variance {} not close to 0.0833", variance);
        }
    }
}
