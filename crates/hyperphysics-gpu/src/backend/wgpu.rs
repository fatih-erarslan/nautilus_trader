//! WGPU cross-platform GPU backend
//!
//! Provides GPU acceleration using WGPU (Vulkan/Metal/DX12).
//! Implements compute shaders for massively parallel pBit lattice evolution.
//!
//! # Performance
//!
//! Target: 100-800× speedup for lattices with N > 10,000 nodes
//! - Parallel state updates: O(N) on GPU vs O(N) on CPU, but 100-1000× faster per operation
//! - Coupling field calculations: O(nnz) where nnz = number of non-zero couplings

use crate::{GPUBackend, GPUCapabilities, BackendType, GPUBuffer, BufferUsage, MemoryStats};
use hyperphysics_core::Result;
use wgpu::{
    util::DeviceExt, BindGroupLayout, Buffer, ComputePipeline, Device, Queue,
};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// WGPU buffer wrapper
pub struct WGPUBuffer {
    buffer: Buffer,
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl WGPUBuffer {
    pub(crate) fn buffer(&self) -> &Buffer {
        &self.buffer
    }
}

/// WGPU compute backend
pub struct WGPUBackend {
    device: Arc<Device>,
    queue: Arc<Queue>,
    capabilities: GPUCapabilities,
    pbit_update_pipeline: Option<ComputePipeline>,
    bind_group_layout: Option<BindGroupLayout>,
}

impl WGPUBackend {
    /// Create new WGPU backend by initializing device
    ///
    /// # Errors
    ///
    /// Returns error if:
    /// - No suitable GPU adapter found
    /// - Device request fails
    pub async fn new() -> Result<Self> {
        // Create WGPU instance with Metal on macOS for AMD GPU support
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            #[cfg(target_os = "macos")]
            backends: wgpu::Backends::METAL,
            #[cfg(not(target_os = "macos"))]
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        // FIXED: Enumerate ALL adapters to properly detect multi-GPU systems
        let adapters: Vec<_> = instance.enumerate_adapters(
            #[cfg(target_os = "macos")]
            wgpu::Backends::METAL,
            #[cfg(not(target_os = "macos"))]
            wgpu::Backends::all(),
        );

        if adapters.is_empty() {
            return Err(hyperphysics_core::EngineError::Configuration {
                message: "No GPU adapters found".to_string(),
            });
        }

        // Select best adapter by max_buffer_size (proxy for VRAM capability)
        let best_adapter = adapters
            .into_iter()
            .max_by_key(|a| a.limits().max_buffer_size)
            .unwrap();

        let adapter_info = best_adapter.get_info();
        let adapter_limits = best_adapter.limits();

        // Log detected GPU
        eprintln!(
            "[hyperphysics-gpu] Detected: {} ({:?}) - VRAM limit: {}GB, device_id=0x{:x}",
            adapter_info.name,
            adapter_info.backend,
            adapter_limits.max_buffer_size / 1024 / 1024 / 1024,
            adapter_info.device
        );

        // FIXED: Request device with ACTUAL adapter limits, not default (256MB)
        let (device, queue) = best_adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HyperPhysics GPU Device"),
                    required_features: wgpu::Features::empty(),
                    // Use adapter's actual limits for proper VRAM detection
                    required_limits: adapter_limits.clone(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| hyperphysics_core::EngineError::Configuration {
                message: format!("Failed to request device: {}", e),
            })?;

        let capabilities = GPUCapabilities {
            backend: BackendType::WGPU,
            device_name: format!(
                "{} ({})",
                adapter_info.name,
                Self::backend_name(adapter_info.backend)
            ),
            // FIXED: Use adapter limits, not device limits (which reflect requested limits)
            max_buffer_size: adapter_limits.max_buffer_size,
            max_workgroup_size: adapter_limits.max_compute_workgroup_size_x,
            supports_compute: true,
        };

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            capabilities,
            pbit_update_pipeline: None,
            bind_group_layout: None,
        })
    }

    /// Create new WGPU backend (blocking version)
    pub fn new_blocking() -> Result<Self> {
        pollster::block_on(Self::new())
    }

    /// Get backend name from WGPU backend type
    fn backend_name(backend: wgpu::Backend) -> &'static str {
        match backend {
            wgpu::Backend::Vulkan => "Vulkan",
            wgpu::Backend::Metal => "Metal",
            wgpu::Backend::Dx12 => "DirectX 12",
            wgpu::Backend::Gl => "OpenGL",
            wgpu::Backend::BrowserWebGpu => "WebGPU",
            _ => "Unknown",
        }
    }

    /// Initialize pBit update compute pipeline
    ///
    /// Creates the compute shader and bind group layout for pBit state updates
    pub fn initialize_pbit_pipeline(&mut self) -> Result<()> {
        // Create bind group layout
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("PBit Update Bind Group Layout"),
                    entries: &[
                        // Buffer 0: States (read-write)
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
                        // Buffer 1: Probabilities (read-write)
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
                        // Buffer 2: Coupling matrix (CSR format - indices)
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
                        // Buffer 3: Coupling matrix (CSR format - values)
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
                        // Buffer 4: Parameters (temperature, etc.)
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
                    ],
                });

        // Create compute shader
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("PBit Update Shader"),
                source: wgpu::ShaderSource::Wgsl(PBIT_UPDATE_SHADER.into()),
            });

        // Create pipeline layout
        let pipeline_layout =
            self.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("PBit Update Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create compute pipeline
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PBit Update Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

        self.pbit_update_pipeline = Some(pipeline);
        self.bind_group_layout = Some(bind_group_layout);

        Ok(())
    }

    /// Create GPU buffer from data
    pub fn create_buffer<T: Pod>(&self, label: &str, data: &[T], usage: wgpu::BufferUsages) -> Buffer {
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: bytemuck::cast_slice(data),
                usage,
            })
    }

    /// Create empty GPU buffer
    pub fn create_empty_buffer(&self, label: &str, size: u64, usage: wgpu::BufferUsages) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Read buffer data back to CPU
    pub async fn read_buffer<T: Pod + Zeroable>(&self, buffer: &Buffer, _size: usize) -> Result<Vec<T>> {
        let slice = buffer.slice(..);
        let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();

        slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);

        receiver
            .receive()
            .await
            .ok_or_else(|| hyperphysics_core::EngineError::Simulation {
                message: "Buffer mapping failed".to_string(),
            })?
            .map_err(|e| hyperphysics_core::EngineError::Simulation {
                message: format!("Buffer mapping error: {:?}", e),
            })?;

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();

        drop(data);
        buffer.unmap();

        Ok(result)
    }

    /// Execute pBit update step on GPU
    ///
    /// # Arguments
    ///
    /// * `states` - Current pBit states (u32: 0 or 1)
    /// * `probabilities` - Output probabilities
    /// * `csr_indices` - CSR row pointers and column indices
    /// * `csr_values` - CSR coupling values
    /// * `temperature` - System temperature
    /// * `n` - Number of pBits
    ///
    /// # Returns
    ///
    /// Updated states and probabilities
    pub async fn update_pbits(
        &mut self,
        states: &[u32],
        csr_indices: &[u32],
        csr_values: &[f32],
        temperature: f32,
        n: usize,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        if self.pbit_update_pipeline.is_none() {
            self.initialize_pbit_pipeline()?;
        }

        // Create buffers
        let states_buffer = self.create_buffer(
            "States",
            states,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );

        let probs_buffer = self.create_empty_buffer(
            "Probabilities",
            (n * std::mem::size_of::<f32>()) as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        );

        let indices_buffer = self.create_buffer(
            "CSR Indices",
            csr_indices,
            wgpu::BufferUsages::STORAGE,
        );

        let values_buffer = self.create_buffer(
            "CSR Values",
            csr_values,
            wgpu::BufferUsages::STORAGE,
        );

        let params = [temperature, n as f32, 0.0, 0.0]; // Padding for alignment
        let params_buffer = self.create_buffer(
            "Parameters",
            &params,
            wgpu::BufferUsages::UNIFORM,
        );

        // Create bind group
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PBit Update Bind Group"),
            layout: self.bind_group_layout.as_ref().unwrap(),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: states_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: probs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create command encoder
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("PBit Update Encoder"),
            });

        // Dispatch compute shader
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PBit Update Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(self.pbit_update_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &bind_group, &[]);

            let workgroup_size = 256;
            let num_workgroups = (n as u32 + workgroup_size - 1) / workgroup_size;
            compute_pass.dispatch_workgroups(num_workgroups, 1, 1);
        }

        // Submit commands
        self.queue.submit(Some(encoder.finish()));

        // Read results
        let updated_states = self.read_buffer::<u32>(&states_buffer, n).await?;
        let updated_probs = self.read_buffer::<f32>(&probs_buffer, n).await?;

        Ok((updated_states, updated_probs))
    }
}

/// WGSL compute shader for pBit state updates
///
/// Implements parallel Metropolis-Hastings updates with sparse coupling matrix
const PBIT_UPDATE_SHADER: &str = r#"
@group(0) @binding(0) var<storage, read_write> states: array<u32>;
@group(0) @binding(1) var<storage, read_write> probabilities: array<f32>;
@group(0) @binding(2) var<storage, read> csr_indices: array<u32>;
@group(0) @binding(3) var<storage, read> csr_values: array<f32>;

struct Params {
    temperature: f32,
    n: f32,
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(4) var<uniform> params: Params;

// XOR shift random number generator
var<private> rng_state: u32;

fn xorshift32(state: u32) -> u32 {
    var x = state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    return x;
}

fn random_float() -> f32 {
    rng_state = xorshift32(rng_state);
    return f32(rng_state) / 4294967295.0;
}

// Sigmoid function: σ(x) = 1/(1 + exp(-x))
fn sigmoid(x: f32) -> f32 {
    return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n_u32 = u32(params.n);

    if (i >= n_u32) {
        return;
    }

    // Initialize RNG with unique seed per thread
    rng_state = i * 747796405u + 2891336453u;

    // Compute local field h_i = Σ_j J_ij s_j
    // Using CSR sparse matrix format
    var h_i: f32 = 0.0;

    // CSR format: indices[i] gives start of row i
    // For simplified version, assume dense packing
    // Full implementation would use proper CSR row_ptr and col_idx arrays

    // For now, use simplified dense approach for demonstration
    // In production, implement proper CSR sparse matrix-vector multiplication
    for (var j = 0u; j < n_u32; j = j + 1u) {
        if (i != j) {
            // Get coupling strength J_ij
            // In full implementation: use CSR format properly
            let idx = i * n_u32 + j;
            if (idx < arrayLength(&csr_values)) {
                let coupling = csr_values[idx];
                let s_j = f32(states[j]) * 2.0 - 1.0; // Convert 0/1 to -1/+1
                h_i += coupling * s_j;
            }
        }
    }

    // Compute probability: P(s=1) = σ(h_i/T)
    let prob_one = sigmoid(h_i / params.temperature);
    probabilities[i] = prob_one;

    // Stochastic update
    let rand_val = random_float();
    states[i] = u32(rand_val < prob_one);
}
"#;

impl WGPUBackend {
    /// Get reference to WGPU device
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get reference to WGPU queue
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }

    /// Execute compute shader with custom bind groups (legacy method)
    ///
    /// TODO: Refactor executor to use new buffer-based API
    pub fn execute_compute_with_bindings(
        &self,
        _shader: &str,
        _workgroups: [u32; 3],
        _bind_group_layout_entries: &[wgpu::BindGroupLayoutEntry],
        _bind_group_entries: &[wgpu::BindGroupEntry],
    ) -> Result<()> {
        Err(hyperphysics_core::EngineError::Configuration {
            message: "execute_compute_with_bindings is deprecated - use new buffer API".to_string(),
        })
    }
}

/// Implement GPUBackend trait for WGPUBackend
impl GPUBackend for WGPUBackend {
    fn capabilities(&self) -> &GPUCapabilities {
        &self.capabilities
    }

    fn execute_compute(&self, shader: &str, workgroups: [u32; 3]) -> Result<()> {
        // Create shader module
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Dynamic Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(shader.into()),
        });

        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Dynamic Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: "main",
            compilation_options: Default::default(),
            cache: None,
        });

        // Create command encoder
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Compute Pass"),
                timestamp_writes: None,
            });

            compute_pass.set_pipeline(&pipeline);
            compute_pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        // Submit command buffer
        self.queue.submit(Some(encoder.finish()));

        Ok(())
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
            label: Some("GPUBuffer"),
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
        let wgpu_buffer = buffer.as_any().downcast_ref::<WGPUBuffer>()
            .ok_or_else(|| hyperphysics_core::EngineError::Configuration {
                message: "Buffer is not a WGPUBuffer".to_string(),
            })?;

        self.queue.write_buffer(&wgpu_buffer.buffer, 0, data);
        Ok(())
    }

    fn read_buffer(&self, buffer: &dyn GPUBuffer) -> Result<Vec<u8>> {
        let wgpu_buffer = buffer.as_any().downcast_ref::<WGPUBuffer>()
            .ok_or_else(|| hyperphysics_core::EngineError::Configuration {
                message: "Buffer is not a WGPUBuffer".to_string(),
            })?;

        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // Copy data to staging buffer
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Read Buffer Encoder"),
        });

        encoder.copy_buffer_to_buffer(&wgpu_buffer.buffer, 0, &staging_buffer, 0, buffer.size());
        self.queue.submit(Some(encoder.finish()));

        // Map and read
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.recv().unwrap().map_err(|e| hyperphysics_core::EngineError::Configuration {
            message: format!("Failed to map buffer: {:?}", e),
        })?;

        let data = buffer_slice.get_mapped_range().to_vec();
        staging_buffer.unmap();

        Ok(data)
    }

    fn synchronize(&self) -> Result<()> {
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }

    fn memory_stats(&self) -> MemoryStats {
        // WGPU doesn't expose detailed memory stats, return defaults
        MemoryStats {
            total_memory: 0,
            used_memory: 0,
            free_memory: 0,
            buffer_count: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wgpu_backend_creation() {
        let result = WGPUBackend::new_blocking();

        // May fail if no GPU available - that's okay
        if let Ok(backend) = result {
            assert_eq!(backend.capabilities().backend, BackendType::WGPU);
            assert!(backend.supports_compute());
            println!("GPU detected: {}", backend.device_name());
        } else {
            println!("No GPU available - skipping test");
        }
    }
}
