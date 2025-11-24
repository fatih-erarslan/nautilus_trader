//! Adapter for CWTS-Ultra GPU trait compatibility
//!
//! This module provides adapters that implement the existing CWTS GPU traits
//! using the new unified GPU orchestrator, enabling seamless integration.

use std::sync::Arc;
use parking_lot::RwLock;
use wgpu::{Buffer, BufferUsages, Device, Queue, CommandEncoder};

use crate::orchestrator::{GpuOrchestrator, PipelineScheduler};
use crate::{GpuResult, WorkloadType};

/// Unified GPU accelerator implementing CWTS trait interface
pub struct UnifiedGpuAccelerator {
    /// Reference to the orchestrator
    device: Arc<Device>,
    queue: Arc<Queue>,
    scheduler: Arc<PipelineScheduler>,
}

impl UnifiedGpuAccelerator {
    /// Create from GpuOrchestrator
    pub fn from_orchestrator(orchestrator: &GpuOrchestrator) -> Self {
        let (device, queue) = orchestrator.primary();
        Self {
            device: device.clone(),
            queue: queue.clone(),
            scheduler: orchestrator.scheduler().clone(),
        }
    }

    /// Create for a specific workload type
    pub fn for_workload(orchestrator: &GpuOrchestrator, workload: &WorkloadType) -> Self {
        let (device, queue) = orchestrator.get_device_for_workload(workload);
        Self {
            device: device.clone(),
            queue: queue.clone(),
            scheduler: orchestrator.scheduler().clone(),
        }
    }

    /// Allocate a GPU buffer
    pub fn allocate_buffer(&self, size: usize) -> GpuResult<Arc<UnifiedGpuBuffer>> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("unified-buffer"),
            size: size as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Arc::new(UnifiedGpuBuffer {
            buffer: Arc::new(buffer),
            device: self.device.clone(),
            queue: self.queue.clone(),
            size,
        }))
    }

    /// Create a compute kernel from WGSL source
    pub fn create_kernel(&self, name: &str, source: &str) -> GpuResult<Arc<UnifiedGpuKernel>> {
        let pipeline = self.scheduler.get_or_create_pipeline(
            &self.device,
            source,
            "main",
            Some(name),
        );

        Ok(Arc::new(UnifiedGpuKernel {
            pipeline,
            device: self.device.clone(),
            queue: self.queue.clone(),
            name: name.to_string(),
        }))
    }

    /// Get device reference
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Get queue reference
    pub fn queue(&self) -> &Arc<Queue> {
        &self.queue
    }
}

/// GPU buffer with read/write operations
pub struct UnifiedGpuBuffer {
    buffer: Arc<Buffer>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    size: usize,
}

impl UnifiedGpuBuffer {
    /// Write data to buffer
    pub fn write(&self, data: &[u8]) -> GpuResult<()> {
        self.queue.write_buffer(&self.buffer, 0, data);
        Ok(())
    }

    /// Write data at offset
    pub fn write_at_offset(&self, data: &[u8], offset: usize) -> GpuResult<()> {
        self.queue.write_buffer(&self.buffer, offset as u64, data);
        Ok(())
    }

    /// Read buffer contents (requires mapping)
    pub fn read(&self) -> GpuResult<Vec<u8>> {
        // Create staging buffer for readback
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging-read"),
            size: self.size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Copy to staging
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("read-encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging, 0, self.size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));

        // Map and read
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().map_err(|e| crate::GpuError::BufferMappingFailed(format!("{:?}", e)))?;

        let data = slice.get_mapped_range().to_vec();
        staging.unmap();

        Ok(data)
    }

    /// Get buffer size
    pub fn size(&self) -> usize {
        self.size
    }

    /// Get underlying wgpu buffer
    pub fn inner(&self) -> &Arc<Buffer> {
        &self.buffer
    }
}

/// Compute kernel wrapper
pub struct UnifiedGpuKernel {
    pipeline: Arc<wgpu::ComputePipeline>,
    device: Arc<Device>,
    queue: Arc<Queue>,
    name: String,
}

impl UnifiedGpuKernel {
    /// Execute kernel with buffers
    pub fn execute(
        &self,
        buffers: &[&UnifiedGpuBuffer],
        work_groups: (u32, u32, u32),
    ) -> GpuResult<()> {
        // Create bind group with all buffers
        let bind_group_layout = self.pipeline.get_bind_group_layout(0);

        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.inner().as_entire_binding(),
            })
            .collect();

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{}-bind-group", self.name)),
            layout: &bind_group_layout,
            entries: &entries,
        });

        // Encode and submit
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some(&format!("{}-encoder", self.name)),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("{}-pass", self.name)),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(work_groups.0, work_groups.1, work_groups.2);
        }

        self.queue.submit(std::iter::once(encoder.finish()));

        Ok(())
    }

    /// Execute and wait for completion
    pub fn execute_sync(
        &self,
        buffers: &[&UnifiedGpuBuffer],
        work_groups: (u32, u32, u32),
    ) -> GpuResult<()> {
        self.execute(buffers, work_groups)?;
        self.device.poll(wgpu::Maintain::Wait);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unified_accelerator_creation() {
        let orchestrator = GpuOrchestrator::new().unwrap();
        let accelerator = UnifiedGpuAccelerator::from_orchestrator(&orchestrator);

        let buffer = accelerator.allocate_buffer(1024).unwrap();
        assert_eq!(buffer.size(), 1024);
    }

    #[test]
    fn test_buffer_write_read() {
        let orchestrator = GpuOrchestrator::new().unwrap();
        let accelerator = UnifiedGpuAccelerator::from_orchestrator(&orchestrator);

        let buffer = accelerator.allocate_buffer(256).unwrap();

        // Write test data
        let data: Vec<u8> = (0..256).map(|i| i as u8).collect();
        buffer.write(&data).unwrap();

        // Read back and verify
        let read_data = buffer.read().unwrap();
        assert_eq!(read_data.len(), 256);
        assert_eq!(read_data, data);
    }

    #[test]
    fn test_gpu_compute_kernel() {
        let orchestrator = GpuOrchestrator::new().unwrap();
        let accelerator = UnifiedGpuAccelerator::from_orchestrator(&orchestrator);

        // Simple doubling shader
        let shader = r#"
            @group(0) @binding(0) var<storage, read_write> data: array<f32>;

            @compute @workgroup_size(64)
            fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
                let idx = gid.x;
                if idx < arrayLength(&data) {
                    data[idx] = data[idx] * 2.0;
                }
            }
        "#;

        let kernel = accelerator.create_kernel("double", shader).unwrap();

        // Create buffer with test data (256 f32 values = 1024 bytes)
        let buffer = accelerator.allocate_buffer(1024).unwrap();
        let input: Vec<f32> = (0..256).map(|i| i as f32).collect();
        let input_bytes: Vec<u8> = input.iter().flat_map(|f| f.to_le_bytes()).collect();
        buffer.write(&input_bytes).unwrap();

        // Execute kernel (256 elements / 64 workgroup = 4 workgroups)
        kernel.execute_sync(&[&buffer], (4, 1, 1)).unwrap();

        // Read back and verify doubling
        let output_bytes = buffer.read().unwrap();
        let output: Vec<f32> = output_bytes
            .chunks(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();

        // Verify first few values
        assert_eq!(output[0], 0.0);   // 0 * 2 = 0
        assert_eq!(output[1], 2.0);   // 1 * 2 = 2
        assert_eq!(output[10], 20.0); // 10 * 2 = 20
        assert_eq!(output[100], 200.0); // 100 * 2 = 200

        println!("GPU compute test passed: successfully doubled {} f32 values", output.len());
    }
}
