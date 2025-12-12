//! WebGPU backend implementation for cross-platform GPU acceleration
//!
//! This module provides WebGPU-based GPU acceleration for CDFA operations,
//! supporting multiple platforms including web browsers, desktop, and mobile.

#[cfg(feature = "webgpu")]
use wgpu::*;
#[cfg(feature = "webgpu")]
use pollster;

use crate::error::{CdfaError, CdfaResult};
use crate::types::{Float as CdfaFloat, FloatArray2 as CdfaMatrix};
use super::{GpuContext, GpuBuffer, GpuKernel, GpuDeviceInfo, GpuBackend, GpuConfig, MemoryStats};
use std::sync::Arc;
use std::collections::HashMap;

#[cfg(feature = "webgpu")]
/// WebGPU implementation of GPU context
pub struct WebGpuContext {
    device: Device,
    queue: Queue,
    adapter_info: AdapterInfo,
    device_info: GpuDeviceInfo,
    config: GpuConfig,
    shader_modules: HashMap<String, ShaderModule>,
}

#[cfg(feature = "webgpu")]
impl WebGpuContext {
    /// Create new WebGPU context
    pub fn new(device_id: u32, config: &GpuConfig) -> CdfaResult<Self> {
        pollster::block_on(async {
            let instance = Instance::new(InstanceDescriptor {
                backends: Backends::all(),
                ..Default::default()
            });
            
            let adapter = instance
                .request_adapter(&RequestAdapterOptions {
                    power_preference: PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                })
                .await
                .ok_or_else(|| CdfaError::GpuError("No WebGPU adapter found".to_string()))?;
            
            let adapter_info = adapter.get_info();
            
            let (device, queue) = adapter
                .request_device(
                    &DeviceDescriptor {
                        label: Some("CDFA WebGPU Device"),
                        required_features: Features::empty(),
                        required_limits: Limits::default(),
                        memory_hints: MemoryHints::default(),
                    },
                    None,
                )
                .await
                .map_err(|e| CdfaError::GpuError(format!("Failed to create WebGPU device: {}", e)))?;
            
            let device_info = Self::create_device_info(&adapter_info, device_id);
            
            Ok(Self {
                device,
                queue,
                adapter_info,
                device_info,
                config: config.clone(),
                shader_modules: HashMap::new(),
            })
        })
    }
    
    /// Create device information from adapter info
    fn create_device_info(adapter_info: &AdapterInfo, device_id: u32) -> GpuDeviceInfo {
        let backend_name = match adapter_info.backend {
            Backend::Vulkan => "Vulkan",
            Backend::Metal => "Metal",
            Backend::Dx12 => "DirectX 12",
            Backend::Dx11 => "DirectX 11",
            Backend::Gl => "OpenGL",
            Backend::BrowserWebGpu => "Browser WebGPU",
        };
        
        GpuDeviceInfo {
            id: device_id,
            name: adapter_info.name.clone(),
            backend: GpuBackend::WebGpu,
            memory_size: 0, // WebGPU doesn't expose memory information directly
            compute_capability: backend_name.to_string(),
            max_work_group_size: 256, // Conservative default
            supports_double_precision: false, // WebGPU typically doesn't support f64
            supports_half_precision: true,    // WebGPU supports f16
        }
    }
    
    /// Load or create shader module
    fn get_or_create_shader(&mut self, source: &str, label: &str) -> CdfaResult<&ShaderModule> {
        if !self.shader_modules.contains_key(label) {
            let module = self.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label),
                source: ShaderSource::Wgsl(source.into()),
            });
            self.shader_modules.insert(label.to_string(), module);
        }
        
        Ok(self.shader_modules.get(label).unwrap())
    }
}

#[cfg(feature = "webgpu")]
impl GpuContext for WebGpuContext {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }
    
    fn allocate_buffer(&self, size: usize) -> CdfaResult<Box<dyn GpuBuffer>> {
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("CDFA Buffer"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        Ok(Box::new(WebGpuBuffer {
            device: self.device.clone(),
            queue: self.queue.clone(),
            buffer,
            size,
        }))
    }
    
    fn create_kernel(&self, source: &str, entry_point: &str) -> CdfaResult<Box<dyn GpuKernel>> {
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("CDFA Compute Shader"),
            source: ShaderSource::Wgsl(source.into()),
        });
        
        let compute_pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("CDFA Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        
        Ok(Box::new(WebGpuKernel {
            device: self.device.clone(),
            queue: self.queue.clone(),
            pipeline: compute_pipeline,
            bind_groups: Vec::new(),
        }))
    }
    
    fn synchronize(&self) -> CdfaResult<()> {
        // WebGPU synchronization is handled through command submission
        Ok(())
    }
    
    fn memory_stats(&self) -> CdfaResult<MemoryStats> {
        // WebGPU doesn't provide direct memory statistics
        Ok(MemoryStats {
            total_memory: 0,
            used_memory: 0,
            free_memory: 0,
            allocated_buffers: 0,
        })
    }
}

#[cfg(feature = "webgpu")]
/// WebGPU buffer implementation
pub struct WebGpuBuffer {
    device: Device,
    queue: Queue,
    buffer: Buffer,
    size: usize,
}

#[cfg(feature = "webgpu")]
impl GpuBuffer for WebGpuBuffer {
    fn size(&self) -> usize {
        self.size
    }
    
    fn copy_from_host(&mut self, data: &[u8]) -> CdfaResult<()> {
        if data.len() > self.size {
            return Err(CdfaError::InvalidParameter(
                "Data size exceeds buffer capacity".to_string()
            ));
        }
        
        self.queue.write_buffer(&self.buffer, 0, data);
        Ok(())
    }
    
    fn copy_to_host(&self, data: &mut [u8]) -> CdfaResult<()> {
        if data.len() > self.size {
            return Err(CdfaError::InvalidParameter(
                "Output buffer too small".to_string()
            ));
        }
        
        // Create staging buffer for readback
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: self.size as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });
        
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size as u64);
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Map staging buffer and copy data
        pollster::block_on(async {
            let buffer_slice = staging_buffer.slice(..);
            buffer_slice.map_async(MapMode::Read, |_| {});
            self.device.poll(Maintain::Wait).panic_on_timeout();
            
            let mapped_data = buffer_slice.get_mapped_range();
            data[..mapped_data.len().min(data.len())].copy_from_slice(&mapped_data);
            
            drop(mapped_data);
            staging_buffer.unmap();
        });
        
        Ok(())
    }
    
    fn map(&self) -> CdfaResult<*mut u8> {
        Err(CdfaError::UnsupportedOperation(
            "Direct buffer mapping not supported in WebGPU backend".to_string()
        ))
    }
    
    fn unmap(&self) -> CdfaResult<()> {
        Ok(())
    }
}

#[cfg(feature = "webgpu")]
/// WebGPU kernel implementation
pub struct WebGpuKernel {
    device: Device,
    queue: Queue,
    pipeline: ComputePipeline,
    bind_groups: Vec<BindGroup>,
}

#[cfg(feature = "webgpu")]
impl GpuKernel for WebGpuKernel {
    fn set_arg(&mut self, index: u32, buffer: &dyn GpuBuffer) -> CdfaResult<()> {
        // WebGPU uses bind groups for resource binding
        // This is a simplified implementation
        Err(CdfaError::UnsupportedOperation(
            "WebGPU kernel argument setting requires bind group management".to_string()
        ))
    }
    
    fn set_scalar_arg<T: Copy>(&mut self, _index: u32, _value: T) -> CdfaResult<()> {
        // WebGPU handles scalar arguments through uniform buffers
        Ok(())
    }
    
    fn launch(&self, global_size: &[u32], local_size: Option<&[u32]>) -> CdfaResult<()> {
        let mut encoder = self.device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("CDFA Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.pipeline);
            
            // Set bind groups
            for (index, bind_group) in self.bind_groups.iter().enumerate() {
                compute_pass.set_bind_group(index as u32, bind_group, &[]);
            }
            
            // Calculate dispatch size
            let workgroup_size = local_size.unwrap_or(&[64]); // Default workgroup size
            let workgroup_count_x = (global_size[0] + workgroup_size[0] - 1) / workgroup_size[0];
            let workgroup_count_y = global_size.get(1)
                .map(|&y| (y + workgroup_size.get(1).unwrap_or(&1) - 1) / workgroup_size.get(1).unwrap_or(&1))
                .unwrap_or(1);
            let workgroup_count_z = global_size.get(2)
                .map(|&z| (z + workgroup_size.get(2).unwrap_or(&1) - 1) / workgroup_size.get(2).unwrap_or(&1))
                .unwrap_or(1);
            
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, workgroup_count_z);
        }
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        Ok(())
    }
}

/// WebGPU compute shader sources
pub mod shaders {
    /// Matrix multiplication shader
    pub const MATRIX_MULTIPLY_WGSL: &str = r#"
        @group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
        @group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
        @group(0) @binding(3) var<uniform> dimensions: vec3<u32>; // m, n, k
        
        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let row = global_id.y;
            let col = global_id.x;
            let m = dimensions.x;
            let n = dimensions.y;
            let k = dimensions.z;
            
            if (row >= m || col >= n) {
                return;
            }
            
            var sum = 0.0;
            for (var i = 0u; i < k; i++) {
                sum += matrix_a[row * k + i] * matrix_b[i * n + col];
            }
            matrix_c[row * n + col] = sum;
        }
    "#;
    
    /// Element-wise operations shader
    pub const ELEMENT_WISE_WGSL: &str = r#"
        @group(0) @binding(0) var<storage, read> input_a: array<f32>;
        @group(0) @binding(1) var<storage, read> input_b: array<f32>;
        @group(0) @binding(2) var<storage, read_write> output: array<f32>;
        @group(0) @binding(3) var<uniform> params: vec2<u32>; // size, operation_type
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            let size = params.x;
            let op_type = params.y;
            
            if (idx >= size) {
                return;
            }
            
            let a = input_a[idx];
            let b = input_b[idx];
            
            switch (op_type) {
                case 0u: { // Addition
                    output[idx] = a + b;
                }
                case 1u: { // Multiplication
                    output[idx] = a * b;
                }
                case 2u: { // Subtraction
                    output[idx] = a - b;
                }
                case 3u: { // Division
                    output[idx] = a / b;
                }
                default: {
                    output[idx] = a;
                }
            }
        }
    "#;
    
    /// Reduction sum shader
    pub const REDUCE_SUM_WGSL: &str = r#"
        @group(0) @binding(0) var<storage, read> input: array<f32>;
        @group(0) @binding(1) var<storage, read_write> output: array<f32>;
        @group(0) @binding(2) var<uniform> size: u32;
        
        var<workgroup> shared_data: array<f32, 256>;
        
        @compute @workgroup_size(256)
        fn main(
            @builtin(global_invocation_id) global_id: vec3<u32>,
            @builtin(local_invocation_id) local_id: vec3<u32>,
            @builtin(workgroup_id) workgroup_id: vec3<u32>
        ) {
            let tid = local_id.x;
            let idx = global_id.x;
            
            // Load data into shared memory
            if (idx < size) {
                shared_data[tid] = input[idx];
            } else {
                shared_data[tid] = 0.0;
            }
            
            workgroupBarrier();
            
            // Reduction in shared memory
            var s = 128u;
            while (s > 0u) {
                if (tid < s) {
                    shared_data[tid] += shared_data[tid + s];
                }
                workgroupBarrier();
                s = s >> 1u;
            }
            
            // Write result for this workgroup
            if (tid == 0u) {
                output[workgroup_id.x] = shared_data[0];
            }
        }
    "#;
    
    /// Pearson diversity calculation shader
    pub const PEARSON_DIVERSITY_WGSL: &str = r#"
        @group(0) @binding(0) var<storage, read> correlation_matrix: array<f32>;
        @group(0) @binding(1) var<storage, read_write> diversity_scores: array<f32>;
        @group(0) @binding(2) var<uniform> n: u32;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let idx = global_id.x;
            
            if (idx >= n) {
                return;
            }
            
            var sum = 0.0;
            var sum_sq = 0.0;
            var count = 0u;
            
            for (var j = 0u; j < n; j++) {
                if (j != idx) {
                    let corr = correlation_matrix[idx * n + j];
                    sum += corr;
                    sum_sq += corr * corr;
                    count++;
                }
            }
            
            if (count > 0u) {
                let mean = sum / f32(count);
                let variance = (sum_sq / f32(count)) - (mean * mean);
                diversity_scores[idx] = sqrt(variance);
            } else {
                diversity_scores[idx] = 0.0;
            }
        }
    "#;
}

// Provide stub implementations when WebGPU is not available
#[cfg(not(feature = "webgpu"))]
pub struct WebGpuContext;

#[cfg(not(feature = "webgpu"))]
impl WebGpuContext {
    pub fn new(_device_id: u32, _config: &GpuConfig) -> CdfaResult<Self> {
        Err(CdfaError::UnsupportedOperation(
            "WebGPU support not compiled in".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_webgpu_availability() {
        #[cfg(feature = "webgpu")]
        {
            let config = GpuConfig::default();
            match WebGpuContext::new(0, &config) {
                Ok(_) => println!("WebGPU device available"),
                Err(_) => println!("No WebGPU device available"),
            }
        }
        
        #[cfg(not(feature = "webgpu"))]
        {
            let config = GpuConfig::default();
            assert!(WebGpuContext::new(0, &config).is_err());
        }
    }
    
    #[test]
    fn test_shader_sources() {
        assert!(!shaders::MATRIX_MULTIPLY_WGSL.is_empty());
        assert!(!shaders::ELEMENT_WISE_WGSL.is_empty());
        assert!(!shaders::REDUCE_SUM_WGSL.is_empty());
        assert!(!shaders::PEARSON_DIVERSITY_WGSL.is_empty());
    }
}