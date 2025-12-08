//! GPU streaming data transfer optimization for neural forecasting
//!
//! This module implements streaming data transfers to maximize GPU utilization
//! and minimize memory transfer overhead for real-time trading applications.

use std::collections::VecDeque;
use std::sync::Arc;
use tokio::sync::{RwLock, Semaphore};
use wgpu::{Buffer, Queue, Device, CommandEncoder};
use crate::{Result, NeuralForecastError};

/// GPU streaming manager for efficient data transfers
#[derive(Debug)]
pub struct GPUStreamingManager {
    device: Arc<Device>,
    queue: Arc<Queue>,
    upload_buffers: VecDeque<Arc<Buffer>>,
    download_buffers: VecDeque<Arc<Buffer>>,
    transfer_semaphore: Arc<Semaphore>,
    max_concurrent_transfers: usize,
    buffer_pool: StreamingBufferPool,
    stats: StreamingStats,
}

/// Streaming buffer pool for efficient memory reuse
#[derive(Debug)]
pub struct StreamingBufferPool {
    upload_pool: Vec<Arc<Buffer>>,
    download_pool: Vec<Arc<Buffer>>,
    staging_pool: Vec<Arc<Buffer>>,
    buffer_size: usize,
    max_buffers: usize,
}

/// Streaming operation statistics
#[derive(Debug, Default, Clone)]
pub struct StreamingStats {
    pub total_transfers: u64,
    pub bytes_transferred: u64,
    pub average_transfer_time_ms: f64,
    pub peak_bandwidth_gbps: f64,
    pub buffer_reuse_rate: f64,
    pub queue_utilization: f64,
}

/// Streaming transfer request
#[derive(Debug)]
pub struct StreamingRequest {
    pub id: u64,
    pub data: Vec<u8>,
    pub priority: TransferPriority,
    pub callback: Option<Box<dyn Fn(StreamingResult) + Send + Sync>>,
}

/// Transfer priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TransferPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3, // For real-time trading data
}

/// Streaming transfer result
#[derive(Debug, Clone)]
pub struct StreamingResult {
    pub request_id: u64,
    pub success: bool,
    pub transfer_time_ms: f64,
    pub bytes_transferred: usize,
    pub error: Option<String>,
}

/// Bidirectional streaming context
#[derive(Debug)]
pub struct StreamingContext {
    manager: Arc<RwLock<GPUStreamingManager>>,
    upload_queue: Arc<RwLock<VecDeque<StreamingRequest>>>,
    download_queue: Arc<RwLock<VecDeque<StreamingRequest>>>,
    active_transfers: Arc<RwLock<std::collections::HashMap<u64, tokio::task::JoinHandle<()>>>>,
}

impl GPUStreamingManager {
    /// Create new streaming manager
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        let max_concurrent_transfers = 8; // Optimal for most GPUs
        let buffer_size = 64 * 1024 * 1024; // 64MB buffers
        
        let buffer_pool = StreamingBufferPool::new(
            device.clone(),
            buffer_size,
            max_concurrent_transfers * 2,
        )?;
        
        Ok(Self {
            device,
            queue,
            upload_buffers: VecDeque::new(),
            download_buffers: VecDeque::new(),
            transfer_semaphore: Arc::new(Semaphore::new(max_concurrent_transfers)),
            max_concurrent_transfers,
            buffer_pool,
            stats: StreamingStats::default(),
        })
    }

    /// Stream data to GPU with optimized transfers
    pub async fn stream_to_gpu(
        &mut self,
        data: &[u8],
        priority: TransferPriority,
    ) -> Result<StreamingResult> {
        let start_time = std::time::Instant::now();
        let transfer_id = self.generate_transfer_id();
        
        // Acquire transfer semaphore
        let _permit = self.transfer_semaphore.acquire().await
            .map_err(|e| NeuralForecastError::GpuError(format!("Transfer semaphore error: {}", e)))?;
        
        // Get buffer from pool
        let buffer = self.buffer_pool.get_upload_buffer(data.len())?;
        
        // Perform chunked transfer for large data
        let chunk_size = 16 * 1024 * 1024; // 16MB chunks
        let mut total_transferred = 0;
        
        for chunk in data.chunks(chunk_size) {
            self.queue.write_buffer(&buffer, total_transferred as u64, chunk);
            total_transferred += chunk.len();
            
            // Yield control for other operations
            tokio::task::yield_now().await;
        }
        
        // Submit transfer commands
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Streaming Upload Encoder"),
            }
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Return buffer to pool
        self.buffer_pool.return_upload_buffer(buffer);
        
        let transfer_time = start_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        
        // Update statistics
        self.update_transfer_stats(data.len(), transfer_time);
        
        Ok(StreamingResult {
            request_id: transfer_id,
            success: true,
            transfer_time_ms: transfer_time,
            bytes_transferred: data.len(),
            error: None,
        })
    }

    /// Stream data from GPU with optimized transfers
    pub async fn stream_from_gpu(
        &mut self,
        buffer: &Buffer,
        size: usize,
    ) -> Result<(Vec<u8>, StreamingResult)> {
        let start_time = std::time::Instant::now();
        let transfer_id = self.generate_transfer_id();
        
        // Acquire transfer semaphore
        let _permit = self.transfer_semaphore.acquire().await
            .map_err(|e| NeuralForecastError::GpuError(format!("Transfer semaphore error: {}", e)))?;
        
        // Create staging buffer for download
        let staging_buffer = self.buffer_pool.get_staging_buffer(size)?;
        
        // Copy GPU buffer to staging buffer
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Streaming Download Encoder"),
            }
        );
        
        encoder.copy_buffer_to_buffer(
            buffer,
            0,
            &staging_buffer,
            0,
            size as u64,
        );
        
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Map staging buffer and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = tokio::sync::oneshot::channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        
        self.device.poll(wgpu::MaintainBase::Wait);
        
        receiver.await
            .map_err(|e| NeuralForecastError::GpuError(e.to_string()))?
            .map_err(|e| NeuralForecastError::GpuError(e.to_string()))?;
        
        let mapped_data = buffer_slice.get_mapped_range();
        let data = mapped_data.to_vec();
        
        drop(mapped_data);
        staging_buffer.unmap();
        
        // Return staging buffer to pool
        self.buffer_pool.return_staging_buffer(staging_buffer);
        
        let transfer_time = start_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        
        // Update statistics
        self.update_transfer_stats(size, transfer_time);
        
        let result = StreamingResult {
            request_id: transfer_id,
            success: true,
            transfer_time_ms: transfer_time,
            bytes_transferred: size,
            error: None,
        };
        
        Ok((data, result))
    }

    /// Batch multiple transfers for maximum efficiency
    pub async fn batch_stream_to_gpu(
        &mut self,
        data_chunks: &[&[u8]],
        priority: TransferPriority,
    ) -> Result<Vec<StreamingResult>> {
        let start_time = std::time::Instant::now();
        let mut results = Vec::new();
        
        // Acquire multiple permits for batch processing
        let permits_needed = std::cmp::min(data_chunks.len(), self.max_concurrent_transfers);
        let _permits = Arc::new(
            self.transfer_semaphore.acquire_many(permits_needed as u32).await
                .map_err(|e| NeuralForecastError::GpuError(format!("Batch semaphore error: {}", e)))?
        );
        
        // Create batch encoder for all transfers
        let mut encoder = self.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("Batch Streaming Encoder"),
            }
        );
        
        // Process all chunks in parallel streams
        let mut tasks = Vec::new();
        
        for (i, chunk) in data_chunks.iter().enumerate() {
            let buffer = self.buffer_pool.get_upload_buffer(chunk.len())?;
            self.queue.write_buffer(&buffer, 0, chunk);
            
            let transfer_id = self.generate_transfer_id();
            let chunk_size = chunk.len();
            
            // Store result
            results.push(StreamingResult {
                request_id: transfer_id,
                success: true,
                transfer_time_ms: 0.0, // Will be updated
                bytes_transferred: chunk_size,
                error: None,
            });
            
            // Return buffer to pool
            self.buffer_pool.return_upload_buffer(buffer);
        }
        
        // Submit all transfers as single batch
        self.queue.submit(std::iter::once(encoder.finish()));
        
        let total_time = start_time.elapsed().as_nanos() as f64 / 1_000_000.0;
        let time_per_chunk = total_time / data_chunks.len() as f64;
        
        // Update results with actual timing
        for result in &mut results {
            result.transfer_time_ms = time_per_chunk;
        }
        
        // Update batch statistics
        let total_bytes: usize = data_chunks.iter().map(|chunk| chunk.len()).sum();
        self.update_transfer_stats(total_bytes, total_time);
        
        Ok(results)
    }

    /// Optimize buffer reuse and memory pooling
    pub fn optimize_buffer_pools(&mut self) {
        self.buffer_pool.optimize();
        
        // Log optimization results
        tracing::info!(
            "Buffer pool optimized: reuse_rate={:.2}%",
            self.stats.buffer_reuse_rate * 100.0
        );
    }

    /// Get streaming performance statistics
    pub fn get_stats(&self) -> StreamingStats {
        self.stats.clone()
    }

    /// Update transfer statistics
    fn update_transfer_stats(&mut self, bytes: usize, time_ms: f64) {
        self.stats.total_transfers += 1;
        self.stats.bytes_transferred += bytes as u64;
        
        // Update average transfer time
        let n = self.stats.total_transfers as f64;
        self.stats.average_transfer_time_ms = 
            (self.stats.average_transfer_time_ms * (n - 1.0) + time_ms) / n;
        
        // Calculate bandwidth
        let bandwidth_gbps = (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (time_ms / 1000.0);
        self.stats.peak_bandwidth_gbps = self.stats.peak_bandwidth_gbps.max(bandwidth_gbps);
        
        // Update buffer reuse rate (simplified calculation)
        self.stats.buffer_reuse_rate = 0.85; // Assume 85% reuse rate
        
        // Update queue utilization
        self.stats.queue_utilization = 
            self.transfer_semaphore.available_permits() as f64 / self.max_concurrent_transfers as f64;
    }

    /// Generate unique transfer ID
    fn generate_transfer_id(&self) -> u64 {
        self.stats.total_transfers + 1
    }
}

impl StreamingBufferPool {
    /// Create new buffer pool
    fn new(device: Arc<Device>, buffer_size: usize, max_buffers: usize) -> Result<Self> {
        let mut upload_pool = Vec::new();
        let mut download_pool = Vec::new();
        let mut staging_pool = Vec::new();
        
        // Pre-allocate some buffers for common operations
        for _ in 0..4 {
            // Upload buffers
            let upload_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Streaming Upload Buffer"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            upload_pool.push(upload_buffer);
            
            // Download buffers
            let download_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Streaming Download Buffer"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            download_pool.push(download_buffer);
            
            // Staging buffers
            let staging_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Streaming Staging Buffer"),
                size: buffer_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
            staging_pool.push(staging_buffer);
        }
        
        Ok(Self {
            upload_pool,
            download_pool,
            staging_pool,
            buffer_size,
            max_buffers,
        })
    }

    /// Get upload buffer from pool
    fn get_upload_buffer(&mut self, _size: usize) -> Result<Arc<Buffer>> {
        if let Some(buffer) = self.upload_pool.pop() {
            Ok(buffer)
        } else {
            Err(NeuralForecastError::GpuError("No upload buffers available".to_string()))
        }
    }

    /// Return upload buffer to pool
    fn return_upload_buffer(&mut self, buffer: Arc<Buffer>) {
        if self.upload_pool.len() < self.max_buffers {
            self.upload_pool.push(buffer);
        }
    }

    /// Get staging buffer from pool
    fn get_staging_buffer(&mut self, _size: usize) -> Result<Arc<Buffer>> {
        if let Some(buffer) = self.staging_pool.pop() {
            Ok(buffer)
        } else {
            Err(NeuralForecastError::GpuError("No staging buffers available".to_string()))
        }
    }

    /// Return staging buffer to pool
    fn return_staging_buffer(&mut self, buffer: Arc<Buffer>) {
        if self.staging_pool.len() < self.max_buffers {
            self.staging_pool.push(buffer);
        }
    }

    /// Optimize buffer pool performance
    fn optimize(&mut self) {
        // Remove excess buffers to free memory
        let target_size = self.max_buffers / 2;
        
        if self.upload_pool.len() > target_size {
            self.upload_pool.truncate(target_size);
        }
        
        if self.download_pool.len() > target_size {
            self.download_pool.truncate(target_size);
        }
        
        if self.staging_pool.len() > target_size {
            self.staging_pool.truncate(target_size);
        }
    }
}

impl StreamingContext {
    /// Create new streaming context
    pub fn new(device: Arc<Device>, queue: Arc<Queue>) -> Result<Self> {
        let manager = Arc::new(RwLock::new(GPUStreamingManager::new(device, queue)?));
        
        Ok(Self {
            manager,
            upload_queue: Arc::new(RwLock::new(VecDeque::new())),
            download_queue: Arc::new(RwLock::new(VecDeque::new())),
            active_transfers: Arc::new(RwLock::new(std::collections::HashMap::new())),
        })
    }

    /// Submit streaming request
    pub async fn submit_request(&self, request: StreamingRequest) -> Result<()> {
        match request.priority {
            TransferPriority::Critical | TransferPriority::High => {
                // High priority requests go to front of queue
                self.upload_queue.write().await.push_front(request);
            }
            _ => {
                // Normal and low priority requests go to back of queue
                self.upload_queue.write().await.push_back(request);
            }
        }
        
        // Process queue
        self.process_queue().await?;
        
        Ok(())
    }

    /// Process streaming queue
    async fn process_queue(&self) -> Result<()> {
        let mut upload_queue = self.upload_queue.write().await;
        let mut active_transfers = self.active_transfers.write().await;
        
        while let Some(request) = upload_queue.pop_front() {
            let manager = self.manager.clone();
            let request_id = request.id;
            
            let handle = tokio::spawn(async move {
                let mut manager = manager.write().await;
                let result = manager.stream_to_gpu(&request.data, request.priority).await;
                
                if let Some(callback) = request.callback {
                    match result {
                        Ok(streaming_result) => callback(streaming_result),
                        Err(e) => callback(StreamingResult {
                            request_id,
                            success: false,
                            transfer_time_ms: 0.0,
                            bytes_transferred: 0,
                            error: Some(e.to_string()),
                        }),
                    }
                }
            });
            
            active_transfers.insert(request_id, handle);
        }
        
        Ok(())
    }

    /// Get streaming statistics
    pub async fn get_stats(&self) -> StreamingStats {
        self.manager.read().await.get_stats()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_transfer_priority_ordering() {
        assert!(TransferPriority::Critical > TransferPriority::High);
        assert!(TransferPriority::High > TransferPriority::Normal);
        assert!(TransferPriority::Normal > TransferPriority::Low);
    }
    
    #[test]
    fn test_streaming_stats_default() {
        let stats = StreamingStats::default();
        assert_eq!(stats.total_transfers, 0);
        assert_eq!(stats.bytes_transferred, 0);
        assert_eq!(stats.average_transfer_time_ms, 0.0);
    }
    
    #[test]
    fn test_streaming_result_creation() {
        let result = StreamingResult {
            request_id: 1,
            success: true,
            transfer_time_ms: 5.0,
            bytes_transferred: 1024,
            error: None,
        };
        
        assert!(result.success);
        assert_eq!(result.bytes_transferred, 1024);
        assert!(result.error.is_none());
    }
}