//! # GPU Runtime
//!
//! wgpu-based runtime for executing hyperbolic message passing kernels
//! on AMD GPUs (Radeon 6800XT / 5500XT).

#[cfg(feature = "gpu")]
use wgpu::{
    self, Adapter, Buffer, BufferDescriptor, BufferUsages, CommandEncoder,
    ComputePass, ComputePipeline, Device, DeviceDescriptor, Instance, Queue,
    ShaderModule, ShaderModuleDescriptor, ShaderSource,
};

use super::{GpuConfig, GpuNodeData, GpuEdgeData, CsrMatrix, DispatchParams, WGSL_SOURCE};
use crate::{CortexError, Result};

/// GPU device info
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    /// Device name
    pub name: String,
    /// Vendor ID
    pub vendor_id: u32,
    /// Device ID
    pub device_id: u32,
    /// Device type (Discrete, Integrated, etc.)
    pub device_type: String,
    /// Estimated VRAM in bytes
    pub vram_bytes: u64,
}

/// GPU Runtime for hyperbolic cortex compute
#[cfg(feature = "gpu")]
pub struct GpuRuntime {
    /// wgpu instance
    instance: Instance,
    /// Adapter (physical device)
    adapter: Adapter,
    /// Logical device
    device: Device,
    /// Command queue
    queue: Queue,
    /// Configuration
    config: GpuConfig,
    /// Compute pipelines
    pipelines: Pipelines,
    /// GPU buffers
    buffers: Buffers,
}

#[cfg(feature = "gpu")]
struct Pipelines {
    edge_messages: ComputePipeline,
    aggregate_update: ComputePipeline,
    mobius_aggregate: ComputePipeline,
    stdp: ComputePipeline,
}

#[cfg(feature = "gpu")]
struct Buffers {
    nodes_in: Buffer,
    nodes_out: Buffer,
    edges: Buffer,
    messages: Buffer,
    config_uniform: Buffer,
    spike_times: Option<Buffer>,
    weight_updates: Option<Buffer>,
}

#[cfg(feature = "gpu")]
impl GpuRuntime {
    /// Create new GPU runtime
    pub async fn new(config: GpuConfig) -> Result<Self> {
        // Create wgpu instance
        let instance = Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN | wgpu::Backends::METAL,
            ..Default::default()
        });
        
        // Request adapter (prefer discrete GPU)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| CortexError::EngineError("No suitable GPU adapter found".into()))?;
        
        // Request device
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("Tengri Holographic Cortex"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| CortexError::EngineError(format!("Failed to create device: {}", e)))?;
        
        // Compile shader
        let shader = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("Hyperbolic Message Passing"),
            source: ShaderSource::Wgsl(WGSL_SOURCE.into()),
        });
        
        // Create pipelines
        let pipelines = Self::create_pipelines(&device, &shader);
        
        // Create buffers
        let buffers = Self::create_buffers(&device, &config);
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            config,
            pipelines,
            buffers,
        })
    }
    
    fn create_pipelines(device: &Device, shader: &ShaderModule) -> Pipelines {
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cortex Pipeline Layout"),
            bind_group_layouts: &[],
            push_constant_ranges: &[],
        });
        
        let create_pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&layout),
                module: shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        
        Pipelines {
            edge_messages: create_pipeline("compute_edge_messages"),
            aggregate_update: create_pipeline("aggregate_messages"),
            mobius_aggregate: create_pipeline("mobius_aggregate"),
            stdp: create_pipeline("compute_stdp"),
        }
    }
    
    fn create_buffers(device: &Device, config: &GpuConfig) -> Buffers {
        let node_size = std::mem::size_of::<GpuNodeData>() as u64;
        let edge_size = std::mem::size_of::<GpuEdgeData>() as u64;
        
        let create_buffer = |label: &str, size: u64, usage: BufferUsages| {
            device.create_buffer(&BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        
        let storage_rw = BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC;
        let storage_ro = BufferUsages::STORAGE | BufferUsages::COPY_DST;
        
        Buffers {
            nodes_in: create_buffer("nodes_in", config.num_nodes as u64 * node_size, storage_ro),
            nodes_out: create_buffer("nodes_out", config.num_nodes as u64 * node_size, storage_rw),
            edges: create_buffer("edges", config.num_edges as u64 * edge_size, storage_ro),
            messages: create_buffer("messages", config.num_edges as u64 * 4, storage_rw),
            config_uniform: create_buffer("config", 16, BufferUsages::UNIFORM | BufferUsages::COPY_DST),
            spike_times: None,
            weight_updates: None,
        }
    }
    
    /// Get device info
    pub fn device_info(&self) -> DeviceInfo {
        let info = self.adapter.get_info();
        DeviceInfo {
            name: info.name.clone(),
            vendor_id: info.vendor,
            device_id: info.device,
            device_type: format!("{:?}", info.device_type),
            vram_bytes: 0, // wgpu doesn't expose this directly
        }
    }
    
    /// Upload nodes to GPU
    pub fn upload_nodes(&self, nodes: &[GpuNodeData]) {
        self.queue.write_buffer(
            &self.buffers.nodes_in,
            0,
            bytemuck::cast_slice(nodes),
        );
    }
    
    /// Upload edges to GPU
    pub fn upload_edges(&self, edges: &[GpuEdgeData]) {
        self.queue.write_buffer(
            &self.buffers.edges,
            0,
            bytemuck::cast_slice(edges),
        );
    }
    
    /// Download nodes from GPU
    pub async fn download_nodes(&self) -> Vec<GpuNodeData> {
        let size = self.config.num_nodes as u64 * std::mem::size_of::<GpuNodeData>() as u64;
        
        // Create staging buffer
        let staging = self.device.create_buffer(&BufferDescriptor {
            label: Some("staging"),
            size,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy from GPU buffer to staging
        let mut encoder = self.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&self.buffers.nodes_out, 0, &staging, 0, size);
        self.queue.submit(Some(encoder.finish()));
        
        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        
        let data = slice.get_mapped_range();
        let nodes: Vec<GpuNodeData> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        
        nodes
    }
    
    /// Run edge message computation kernel
    pub fn dispatch_edge_messages(&self) {
        let dispatch = DispatchParams::from_config(&self.config);
        
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.edge_messages);
            pass.dispatch_workgroups(dispatch.edge_dispatch, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
    
    /// Run aggregation and update kernel
    pub fn dispatch_aggregate_update(&self) {
        let dispatch = DispatchParams::from_config(&self.config);
        
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.aggregate_update);
            pass.dispatch_workgroups(dispatch.node_dispatch, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
    
    /// Run Möbius aggregation kernel
    pub fn dispatch_mobius_aggregate(&self) {
        let dispatch = DispatchParams::from_config(&self.config);
        
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines.mobius_aggregate);
            pass.dispatch_workgroups(dispatch.node_dispatch, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
    
    /// Run full message passing step
    pub fn step(&self) {
        self.dispatch_edge_messages();
        self.dispatch_aggregate_update();
    }
    
    /// Run full hyperbolic step with Möbius aggregation
    pub fn step_hyperbolic(&self) {
        self.dispatch_edge_messages();
        self.dispatch_mobius_aggregate();
    }
    
    /// Synchronize GPU
    pub fn sync(&self) {
        self.device.poll(wgpu::Maintain::Wait);
    }
}

/// Stub runtime for when GPU feature is disabled
#[cfg(not(feature = "gpu"))]
pub struct GpuRuntime {
    config: GpuConfig,
}

#[cfg(not(feature = "gpu"))]
impl GpuRuntime {
    /// Create stub runtime
    pub fn new(_config: GpuConfig) -> Result<Self> {
        Err(CortexError::EngineError(
            "GPU feature not enabled. Compile with --features gpu".into()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.num_nodes, 65536);
        assert_eq!(config.workgroup_size, 256);
    }
}
