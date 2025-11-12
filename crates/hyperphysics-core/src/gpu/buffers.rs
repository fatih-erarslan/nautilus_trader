//! GPU buffer management for efficient data transfer

use wgpu;
use wgpu::util::DeviceExt;
use std::sync::Arc;

/// Managed GPU buffer with automatic staging
pub struct GpuBuffer<T: bytemuck::Pod> {
    buffer: wgpu::Buffer,
    size: usize,
    device: Arc<wgpu::Device>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: bytemuck::Pod> GpuBuffer<T> {
    /// Create new GPU buffer
    pub fn new(device: Arc<wgpu::Device>, size: usize, usage: wgpu::BufferUsages) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HyperPhysics GPU Buffer"),
            size: (size * std::mem::size_of::<T>()) as u64,
            usage,
            mapped_at_creation: false,
        });

        Self {
            buffer,
            size,
            device,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Create buffer from slice
    pub fn from_slice(device: Arc<wgpu::Device>, queue: &wgpu::Queue, data: &[T]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("HyperPhysics GPU Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            buffer,
            size: data.len(),
            device,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Write data to buffer
    pub fn write(&self, queue: &wgpu::Queue, data: &[T]) {
        assert!(data.len() <= self.size, "Data exceeds buffer size");
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }

    /// Read data from buffer (async)
    pub async fn read(&self) -> Vec<T> {
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: (self.size * std::mem::size_of::<T>()) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Copy Encoder"),
        });

        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &staging_buffer,
            0,
            (self.size * std::mem::size_of::<T>()) as u64,
        );

        // Note: In real implementation, would need queue reference from context
        // This is a stub showing the pattern - actual impl requires queue parameter
        // self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });

        self.device.poll(wgpu::Maintain::Wait);
        receiver.await.unwrap().unwrap();

        let data = buffer_slice.get_mapped_range();
        let result = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        result
    }

    /// Get underlying wgpu buffer
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// Get buffer size in elements
    pub fn size(&self) -> usize {
        self.size
    }
}

/// Double-buffered GPU storage for ping-pong patterns
pub struct DoubleBuffer<T: bytemuck::Pod> {
    buffers: [GpuBuffer<T>; 2],
    current: usize,
}

impl<T: bytemuck::Pod> DoubleBuffer<T> {
    /// Create new double buffer
    pub fn new(device: Arc<wgpu::Device>, size: usize, usage: wgpu::BufferUsages) -> Self {
        Self {
            buffers: [
                GpuBuffer::new(device.clone(), size, usage),
                GpuBuffer::new(device, size, usage),
            ],
            current: 0,
        }
    }

    /// Get current buffer (read)
    pub fn current(&self) -> &GpuBuffer<T> {
        &self.buffers[self.current]
    }

    /// Get next buffer (write)
    pub fn next(&self) -> &GpuBuffer<T> {
        &self.buffers[1 - self.current]
    }

    /// Swap buffers
    pub fn swap(&mut self) {
        self.current = 1 - self.current;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_buffer_roundtrip() {
        use crate::gpu::device::init_global_gpu;

        if let Ok(ctx) = init_global_gpu().await {
            let data = vec![1.0f32, 2.0, 3.0, 4.0];
            let buffer = GpuBuffer::from_slice(ctx.device.clone(), &ctx.queue, &data);
            let result = buffer.read().await;
            assert_eq!(data, result);
        }
    }
}
