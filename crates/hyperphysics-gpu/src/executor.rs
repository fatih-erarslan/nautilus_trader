//! High-level GPU compute executor for HyperPhysics
//!
//! Orchestrates WGPU backend, compute shaders, and buffer management
//! for production-grade pBit lattice simulation.

use super::backend::wgpu::WGPUBackend;
use super::kernels::{PBIT_UPDATE_SHADER, ENERGY_SHADER, ENTROPY_SHADER};
use super::scheduler::GPUScheduler;
use super::monitoring::PerformanceMonitor;
use super::rng::GPURng;
use hyperphysics_core::Result;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use std::time::Instant;
use std::sync::Arc;

/// GPU representation of pBit state
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GPUPBitState {
    pub state: u32,
    pub bias: f32,
    pub coupling_offset: u32,
    pub coupling_count: u32,
}

/// GPU representation of coupling
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct GPUCoupling {
    pub target_idx: u32,
    pub strength: f32,
}

/// Parameters for pBit update kernel
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct SimParams {
    pub temperature: f32,
    pub dt: f32,
    pub n_pbits: u32,
    pub seed: u32,
}

/// Parameters for energy calculation kernel
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EnergyParams {
    pub n_pbits: u32,
    pub n_workgroups: u32,
}

/// Parameters for entropy calculation kernel
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct EntropyParams {
    pub n_pbits: u32,
    pub temperature: f32,
    pub n_workgroups: u32,
}

/// High-level GPU compute executor
pub struct GPUExecutor {
    backend: Arc<WGPUBackend>,
    scheduler: GPUScheduler,

    // Double buffering for pBit states
    state_buffer_a: wgpu::Buffer,
    state_buffer_b: wgpu::Buffer,
    current_buffer: usize,

    // Coupling topology (read-only)
    coupling_buffer: wgpu::Buffer,

    // GPU Random Number Generator
    rng: GPURng,

    // Reduction buffers for energy/entropy
    energy_partial: wgpu::Buffer,
    entropy_partial: wgpu::Buffer,

    // Uniform parameter buffers
    sim_params_buffer: wgpu::Buffer,
    energy_params_buffer: wgpu::Buffer,
    entropy_params_buffer: wgpu::Buffer,

    lattice_size: usize,

    // Performance monitoring
    monitor: PerformanceMonitor,
}

impl GPUExecutor {
    /// Initialize GPU executor with lattice configuration
    ///
    /// # Arguments
    /// * `lattice_size` - Number of pBits in the lattice
    /// * `couplings` - Coupling topology as (source, target, strength) triples
    pub async fn new(
        lattice_size: usize,
        couplings: &[(usize, usize, f64)],
    ) -> Result<Self> {
        let backend: WGPUBackend = WGPUBackend::new().await?;
        let backend_arc = Arc::new(backend);
        let device = backend_arc.device().clone();
        let scheduler = GPUScheduler::new(256); // 256-thread workgroups

        // Calculate workgroup dispatch
        let n_workgroups = scheduler.compute_dispatch(lattice_size)[0] as usize;

        // Build coupling buffer and initial states with correct offset/count
        let (gpu_couplings, initial_states) = Self::build_coupling_buffer(lattice_size, couplings);

        let state_buffer_a = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pBit State Buffer A"),
            contents: bytemuck::cast_slice(&initial_states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let state_buffer_b = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("pBit State Buffer B"),
            contents: bytemuck::cast_slice(&initial_states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let coupling_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Coupling Buffer"),
            contents: bytemuck::cast_slice(&gpu_couplings),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // Initialize GPU Random Number Generator
        let mut rng = GPURng::new(backend_arc.clone(), lattice_size)?;

        // Seed RNG with time-based seed
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as u32;
        rng.seed(seed)?;

        // Reduction buffers
        let energy_partial = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Energy Partial Buffer"),
            contents: bytemuck::cast_slice(&vec![0.0f32; n_workgroups]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        let entropy_partial = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Entropy Partial Buffer"),
            contents: bytemuck::cast_slice(&vec![0.0f32; n_workgroups]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        // Create uniform buffers for parameters
        let sim_params = SimParams {
            temperature: 1.0,
            dt: 0.01,
            n_pbits: lattice_size as u32,
            seed: 0,
        };

        let sim_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Sim Params Buffer"),
            contents: bytemuck::cast_slice(&[sim_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let energy_params = EnergyParams {
            n_pbits: lattice_size as u32,
            n_workgroups: n_workgroups as u32,
        };

        let energy_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Energy Params Buffer"),
            contents: bytemuck::cast_slice(&[energy_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let entropy_params = EntropyParams {
            n_pbits: lattice_size as u32,
            temperature: 1.0,
            n_workgroups: n_workgroups as u32,
        };

        let entropy_params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Entropy Params Buffer"),
            contents: bytemuck::cast_slice(&[entropy_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            backend: backend_arc,
            scheduler,
            state_buffer_a,
            state_buffer_b,
            current_buffer: 0,
            coupling_buffer,
            rng,
            energy_partial,
            entropy_partial,
            sim_params_buffer,
            energy_params_buffer,
            entropy_params_buffer,
            lattice_size,
            monitor: PerformanceMonitor::new(1000), // Keep last 1000 operations
        })
    }

    /// Build coupling buffer with indirection structure
    ///
    /// Returns (coupling_buffer, updated_states) where each pBit has correct offset/count
    fn build_coupling_buffer(
        lattice_size: usize,
        couplings: &[(usize, usize, f64)],
    ) -> (Vec<GPUCoupling>, Vec<GPUPBitState>) {
        // Sort couplings by source index
        let mut sorted_couplings = couplings.to_vec();
        sorted_couplings.sort_by_key(|(source, _, _)| *source);

        // Convert to GPU format
        let gpu_couplings: Vec<GPUCoupling> = sorted_couplings
            .iter()
            .map(|(_, target, strength)| GPUCoupling {
                target_idx: *target as u32,
                strength: *strength as f32,
            })
            .collect();

        // Build pBit states with correct coupling offsets and counts
        let mut states = vec![GPUPBitState {
            state: 0,
            bias: 0.0,
            coupling_offset: 0,
            coupling_count: 0,
        }; lattice_size];

        // Count couplings per source pBit
        let mut current_source = 0;
        let mut current_offset = 0;
        let mut current_count = 0;

        for (source, _, _) in sorted_couplings.iter() {
            if *source != current_source {
                // Finalize previous source
                if current_source < lattice_size {
                    states[current_source].coupling_offset = current_offset;
                    states[current_source].coupling_count = current_count;
                }

                // Move to new source
                current_offset += current_count;
                current_count = 0;
                current_source = *source;
            }
            current_count += 1;
        }

        // Finalize last source
        if current_source < lattice_size {
            states[current_source].coupling_offset = current_offset;
            states[current_source].coupling_count = current_count;
        }

        (gpu_couplings, states)
    }

    /// Update pBit states for one time step
    ///
    /// Uses Gillespie stochastic algorithm on GPU
    pub async fn step(&mut self, temperature: f32, dt: f32) -> Result<()> {
        let start = Instant::now();

        // Update simulation parameters
        let sim_params = SimParams {
            temperature,
            dt,
            n_pbits: self.lattice_size as u32,
            seed: rand::random(),
        };

        self.backend.queue().write_buffer(
            &self.sim_params_buffer,
            0,
            bytemuck::cast_slice(&[sim_params]),
        );

        // Generate random values on GPU using xorshift128+
        self.rng.generate_uniform()?;

        // Compute workgroup dispatch
        let workgroups = self.scheduler.compute_dispatch(self.lattice_size);

        // Create bind group layout
        let bind_group_layout_entries = vec![
            // @binding(0) pbits_in
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(1) pbits_out
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
            // @binding(2) couplings
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(3) random_values
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(4) params
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let (current, next) = if self.current_buffer == 0 {
            (&self.state_buffer_a, &self.state_buffer_b)
        } else {
            (&self.state_buffer_b, &self.state_buffer_a)
        };

        let bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: current.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: next.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.coupling_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.rng.output_buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: self.sim_params_buffer.as_entire_binding(),
            },
        ];

        // Execute pBit update kernel
        self.backend.execute_compute_with_bindings(
            PBIT_UPDATE_SHADER,
            workgroups,
            &bind_group_layout_entries,
            &bind_group_entries,
        )?;

        // Swap buffers
        self.current_buffer = 1 - self.current_buffer;

        // Record performance
        let duration = start.elapsed();
        self.monitor.record_timed("pbit_update", duration, self.lattice_size);

        Ok(())
    }

    /// Compute total system energy (Ising Hamiltonian)
    pub async fn compute_energy(&mut self) -> Result<f64> {
        let start = Instant::now();
        let workgroups = self.scheduler.compute_dispatch(self.lattice_size);
        let n_workgroups = workgroups[0];

        let current = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        // First pass: compute partial energies
        let bind_group_layout_entries = vec![
            // @binding(0) pbits
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(1) couplings
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(2) energy_partial
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // @binding(3) params
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ];

        let bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: current.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.coupling_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.energy_partial.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: self.energy_params_buffer.as_entire_binding(),
            },
        ];

        // Execute energy kernel (main entry point)
        self.backend.execute_compute_with_bindings(
            ENERGY_SHADER,
            workgroups,
            &bind_group_layout_entries,
            &bind_group_entries,
        )?;

        // Second pass: reduce partial sums if multiple workgroups
        if n_workgroups > 1 {
            // Create shader module with reduce_final entry point
            let shader_module = self.backend.device().create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Energy Reduce Final Shader"),
                source: wgpu::ShaderSource::Wgsl(ENERGY_SHADER.into()),
            });

            // Create bind group layout for reduce_final
            let reduce_layout = self.backend.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Reduce Final Bind Group Layout"),
                entries: &bind_group_layout_entries,
            });

            // Create pipeline layout
            let pipeline_layout = self.backend.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Reduce Final Pipeline Layout"),
                bind_group_layouts: &[&reduce_layout],
                push_constant_ranges: &[],
            });

            // Create compute pipeline with reduce_final entry point
            let reduce_pipeline = self.backend.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Energy Reduce Final Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "reduce_final",
                compilation_options: Default::default(),
                cache: None,
            });

            // Create bind group
            let reduce_bind_group = self.backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Reduce Final Bind Group"),
                layout: &reduce_layout,
                entries: &bind_group_entries,
            });

            // Compute dispatch size for reduce_final (ceil(n_workgroups / 256))
            let reduce_workgroups = [(n_workgroups + 255) / 256, 1, 1];

            // Create command encoder and execute reduce_final
            let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Reduce Final Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce Final Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&reduce_pipeline);
                compute_pass.set_bind_group(0, &reduce_bind_group, &[]);
                compute_pass.dispatch_workgroups(reduce_workgroups[0], reduce_workgroups[1], reduce_workgroups[2]);
            }

            self.backend.queue().submit(Some(encoder.finish()));
        }

        // Read back final energy
        let staging_buffer = self.backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Energy Staging Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Energy Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.energy_partial,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<f32>() as u64,
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
        let energy = bytemuck::cast_slice::<u8, f32>(&data)[0] as f64;
        drop(data);
        staging_buffer.unmap();

        // Record performance
        let duration = start.elapsed();
        self.monitor.record_timed("energy", duration, self.lattice_size);

        Ok(energy)
    }

    /// Compute Shannon entropy
    pub async fn compute_entropy(&mut self) -> Result<f64> {
        let start = Instant::now();
        let workgroups = self.scheduler.compute_dispatch(self.lattice_size);
        let n_workgroups = workgroups[0];

        let current = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        // Similar to energy calculation
        let bind_group_layout_entries = vec![
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
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
        ];

        let bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: current.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: self.entropy_partial.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.entropy_params_buffer.as_entire_binding(),
            },
        ];

        self.backend.execute_compute_with_bindings(
            ENTROPY_SHADER,
            workgroups,
            &bind_group_layout_entries,
            &bind_group_entries,
        )?;

        // Second pass: reduce partial sums if multiple workgroups
        if n_workgroups > 1 {
            // Create shader module with reduce_final entry point
            let shader_module = self.backend.device().create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Entropy Reduce Final Shader"),
                source: wgpu::ShaderSource::Wgsl(ENTROPY_SHADER.into()),
            });

            // Create bind group layout for reduce_final
            let reduce_layout = self.backend.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Entropy Reduce Final Bind Group Layout"),
                entries: &bind_group_layout_entries,
            });

            // Create pipeline layout
            let pipeline_layout = self.backend.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Entropy Reduce Final Pipeline Layout"),
                bind_group_layouts: &[&reduce_layout],
                push_constant_ranges: &[],
            });

            // Create compute pipeline with reduce_final entry point
            let reduce_pipeline = self.backend.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Entropy Reduce Final Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "reduce_final",
                compilation_options: Default::default(),
                cache: None,
            });

            // Create bind group
            let reduce_bind_group = self.backend.device().create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Entropy Reduce Final Bind Group"),
                layout: &reduce_layout,
                entries: &bind_group_entries,
            });

            // Compute dispatch size for reduce_final (ceil(n_workgroups / 256))
            let reduce_workgroups = [(n_workgroups + 255) / 256, 1, 1];

            // Create command encoder and execute reduce_final
            let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Entropy Reduce Final Encoder"),
            });

            {
                let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Entropy Reduce Final Pass"),
                    timestamp_writes: None,
                });

                compute_pass.set_pipeline(&reduce_pipeline);
                compute_pass.set_bind_group(0, &reduce_bind_group, &[]);
                compute_pass.dispatch_workgroups(reduce_workgroups[0], reduce_workgroups[1], reduce_workgroups[2]);
            }

            self.backend.queue().submit(Some(encoder.finish()));
        }

        // Read back final entropy
        let staging_buffer = self.backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Entropy Staging Buffer"),
            size: std::mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Entropy Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.entropy_partial,
            0,
            &staging_buffer,
            0,
            std::mem::size_of::<f32>() as u64,
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
        let entropy = bytemuck::cast_slice::<u8, f32>(&data)[0] as f64;
        drop(data);
        staging_buffer.unmap();

        // Record performance
        let duration = start.elapsed();
        self.monitor.record_timed("entropy", duration, self.lattice_size);

        Ok(entropy)
    }

    /// Get current pBit states (async GPU→CPU transfer)
    pub async fn read_states(&mut self) -> Result<Vec<GPUPBitState>> {
        let start = Instant::now();
        let current = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        // Create staging buffer
        let staging_buffer = self.backend.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.lattice_size * std::mem::size_of::<GPUPBitState>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy from GPU
        let mut encoder = self.backend.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            current,
            0,
            &staging_buffer,
            0,
            (self.lattice_size * std::mem::size_of::<GPUPBitState>()) as u64,
        );

        self.backend.queue().submit(Some(encoder.finish()));

        // Read back
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
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        // Record performance
        let duration = start.elapsed();
        self.monitor.record_timed("read_states", duration, self.lattice_size);

        Ok(result)
    }

    /// Get performance statistics
    pub fn performance_stats(&self) -> String {
        self.monitor.report()
    }

    /// Get reference to performance monitor for advanced analysis
    pub fn monitor(&self) -> &PerformanceMonitor {
        &self.monitor
    }

    /// Get mutable reference to performance monitor
    pub fn monitor_mut(&mut self) -> &mut PerformanceMonitor {
        &mut self.monitor
    }

    /// Update pBit biases
    pub async fn update_biases(&mut self, biases: &[f32]) -> Result<()> {
        if biases.len() != self.lattice_size {
            return Err(hyperphysics_core::EngineError::Configuration {
                message: format!(
                    "Bias array size {} does not match lattice size {}",
                    biases.len(),
                    self.lattice_size
                ),
            });
        }

        // Read current states
        let mut states = self.read_states().await?;

        // Update biases
        for (state, &bias) in states.iter_mut().zip(biases.iter()) {
            state.bias = bias;
        }

        // Write back
        let current = if self.current_buffer == 0 {
            &self.state_buffer_a
        } else {
            &self.state_buffer_b
        };

        self.backend.queue().write_buffer(
            current,
            0,
            bytemuck::cast_slice(&states),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_executor_initialization() {
        // Simple 3-node chain: 0 -- 1 -- 2
        let couplings = vec![
            (0, 1, 1.0),
            (1, 0, 1.0),
            (1, 2, 1.0),
            (2, 1, 1.0),
        ];

        let executor = GPUExecutor::new(3, &couplings).await;

        if executor.is_ok() {
            let exec = executor.unwrap();
            assert_eq!(exec.lattice_size, 3);
        }
    }

    #[tokio::test]
    async fn test_state_update() {
        let couplings = vec![(0, 1, 1.0), (1, 0, 1.0)];

        if let Ok(mut executor) = GPUExecutor::new(2, &couplings).await {
            // Run one step
            let result = executor.step(1.0, 0.01).await;
            assert!(result.is_ok());

            // Verify states can be read back
            let states = executor.read_states().await;
            assert!(states.is_ok());
            assert_eq!(states.unwrap().len(), 2);
        }
    }
}
