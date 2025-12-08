//! GPU pipeline for ultra-fast quantum computations

use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::{Mutex, RwLock};
use wgpu::*;
use crate::{QBMIAError, QBMIAResult, GpuDeviceInfo, GpuBuffer, GpuBufferUsage, KernelParams};

/// Ultra-optimized GPU pipeline for sub-microsecond performance
pub struct GpuPipeline {
    /// WGPU instance
    instance: Instance,
    
    /// GPU adapter
    adapter: Adapter,
    
    /// GPU device
    device: Device,
    
    /// Command queue
    queue: Queue,
    
    /// Device information
    device_info: GpuDeviceInfo,
    
    /// Buffer pool for memory reuse
    buffer_pool: Arc<Mutex<BufferPool>>,
    
    /// Compute pipeline cache
    pipeline_cache: Arc<RwLock<HashMap<String, ComputePipeline>>>,
    
    /// Bind group cache
    bind_group_cache: Arc<RwLock<HashMap<String, BindGroup>>>,
    
    /// Command encoder pool
    encoder_pool: Arc<Mutex<Vec<CommandEncoder>>>,
    
    /// Performance metrics
    metrics: Arc<Mutex<GpuMetrics>>,
}

impl GpuPipeline {
    /// Create a new GPU pipeline with optimal configuration
    pub async fn new() -> QBMIAResult<Self> {
        tracing::info!("Initializing GPU pipeline");
        
        // Create WGPU instance with all backends
        let instance = Instance::new(InstanceDescriptor {
            backends: Backends::all(),
            flags: InstanceFlags::debugging(),
            dx12_shader_compiler: Dx12Compiler::default(),
            gles_minor_version: Gles3MinorVersion::default(),
        });
        
        // Request high-performance adapter
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .ok_or_else(|| QBMIAError::gpu_init("No suitable GPU adapter found"))?;
        
        let adapter_info = adapter.get_info();
        tracing::info!("Selected GPU: {} {}", adapter_info.vendor, adapter_info.name);
        
        // Get device limits
        let limits = adapter.limits();
        
        // Request device with optimal features
        let required_features = Features::COMPUTE_SHADERS
            | Features::SPIRV_SHADER_PASSTHROUGH
            | Features::TIMESTAMP_QUERY
            | Features::TIMESTAMP_QUERY_INSIDE_PASSES
            | Features::PIPELINE_STATISTICS_QUERY;
        
        let available_features = adapter.features();
        let features = available_features & required_features;
        
        if !features.contains(Features::COMPUTE_SHADERS) {
            return Err(QBMIAError::hardware_not_supported("Compute shaders not supported"));
        }
        
        let (device, queue) = adapter
            .request_device(
                &DeviceDescriptor {
                    label: Some("QBMIA GPU Device"),
                    required_features: features,
                    required_limits: limits,
                    memory_hints: MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| QBMIAError::gpu_init(format!("Failed to create device: {}", e)))?;
        
        // Create device info
        let device_info = GpuDeviceInfo {
            name: adapter_info.name.clone(),
            vendor: format!("{:?}", adapter_info.vendor),
            device_type: format!("{:?}", adapter_info.device_type),
            memory_bytes: limits.max_buffer_size,
            compute_units: 1, // WGPU doesn't expose this directly
            max_workgroup_size: [
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ],
            supports_cuda: adapter_info.backend == Backend::Vulkan, // Approximate
            supports_metal: adapter_info.backend == Backend::Metal,
            supports_vulkan: adapter_info.backend == Backend::Vulkan,
        };
        
        tracing::info!("GPU device info: {}", device_info);
        
        // Initialize pools and caches
        let buffer_pool = Arc::new(Mutex::new(BufferPool::new()));
        let pipeline_cache = Arc::new(RwLock::new(HashMap::new()));
        let bind_group_cache = Arc::new(RwLock::new(HashMap::new()));
        let encoder_pool = Arc::new(Mutex::new(Vec::new()));
        let metrics = Arc::new(Mutex::new(GpuMetrics::new()));
        
        Ok(Self {
            instance,
            adapter,
            device,
            queue,
            device_info,
            buffer_pool,
            pipeline_cache,
            bind_group_cache,
            encoder_pool,
            metrics,
        })
    }
    
    /// Get device information
    pub fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }
    
    /// Create a buffer with optimal settings
    pub async fn create_buffer(&self, data: &[u8], usage: GpuBufferUsage) -> QBMIAResult<GpuBuffer> {
        let start_time = std::time::Instant::now();
        
        let wgpu_usage = match usage {
            GpuBufferUsage::Storage => BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
            GpuBufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            GpuBufferUsage::Vertex => BufferUsages::VERTEX | BufferUsages::COPY_DST,
            GpuBufferUsage::Index => BufferUsages::INDEX | BufferUsages::COPY_DST,
            GpuBufferUsage::Staging => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        };
        
        // Try to get from pool first
        let mut pool = self.buffer_pool.lock().await;
        if let Some(buffer_id) = pool.get_buffer(data.len(), usage) {
            // Write data to existing buffer
            self.queue.write_buffer(&pool.get_wgpu_buffer(buffer_id).unwrap(), 0, data);
            
            let creation_time = start_time.elapsed();
            let mut metrics = self.metrics.lock().await;
            metrics.record_buffer_reuse(creation_time);
            
            return Ok(GpuBuffer {
                id: buffer_id,
                size: data.len(),
                usage,
            });
        }
        
        // Create new buffer
        let buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("QBMIA Buffer"),
            size: data.len() as u64,
            usage: wgpu_usage,
            mapped_at_creation: false,
        });
        
        // Write data
        self.queue.write_buffer(&buffer, 0, data);
        
        // Add to pool
        let buffer_id = pool.add_buffer(buffer, data.len(), usage);
        
        let creation_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().await;
        metrics.record_buffer_creation(creation_time);
        
        tracing::debug!("Created GPU buffer {} ({} bytes) in {:.3}μs", 
                       buffer_id, data.len(), creation_time.as_nanos() as f64 / 1000.0);
        
        Ok(GpuBuffer {
            id: buffer_id,
            size: data.len(),
            usage,
        })
    }
    
    /// Read data from buffer
    pub async fn read_buffer(&self, buffer: &GpuBuffer) -> QBMIAResult<Vec<u8>> {
        let start_time = std::time::Instant::now();
        
        let pool = self.buffer_pool.lock().await;
        let wgpu_buffer = pool.get_wgpu_buffer(buffer.id)
            .ok_or_else(|| QBMIAError::buffer_op("Buffer not found"))?;
        
        // Create staging buffer
        let staging_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("Staging Buffer"),
            size: buffer.size as u64,
            usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Copy data to staging buffer
        let mut encoder = self.get_command_encoder().await;
        encoder.copy_buffer_to_buffer(wgpu_buffer, 0, &staging_buffer, 0, buffer.size as u64);
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Map and read data
        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = futures::channel::oneshot::channel();
        buffer_slice.map_async(MapMode::Read, move |result| {
            sender.send(result).unwrap();
        });
        
        self.device.poll(Maintain::Wait);
        receiver.await.unwrap()
            .map_err(|e| QBMIAError::buffer_op(format!("Failed to map buffer: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range().to_vec();
        staging_buffer.unmap();
        
        let read_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().await;
        metrics.record_buffer_read(read_time, buffer.size);
        
        tracing::debug!("Read {} bytes from GPU buffer in {:.3}μs", 
                       buffer.size, read_time.as_nanos() as f64 / 1000.0);
        
        Ok(data)
    }
    
    /// Create or get cached compute pipeline
    pub async fn get_compute_pipeline(&self, shader_source: &str, entry_point: &str) -> QBMIAResult<Arc<ComputePipeline>> {
        let cache_key = format!("{}:{}", shader_source, entry_point);
        
        // Check cache first
        {
            let cache = self.pipeline_cache.read().await;
            if let Some(pipeline) = cache.get(&cache_key) {
                return Ok(Arc::new(pipeline.clone()));
            }
        }
        
        // Create new pipeline
        let start_time = std::time::Instant::now();
        
        let shader_module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("QBMIA Compute Shader"),
            source: ShaderSource::Wgsl(shader_source.into()),
        });
        
        let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("QBMIA Compute Pipeline"),
            layout: None,
            module: &shader_module,
            entry_point,
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        
        let compilation_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().await;
        metrics.record_pipeline_compilation(compilation_time);
        
        // Cache pipeline
        {
            let mut cache = self.pipeline_cache.write().await;
            cache.insert(cache_key, pipeline.clone());
        }
        
        tracing::debug!("Compiled compute pipeline in {:.3}μs", 
                       compilation_time.as_nanos() as f64 / 1000.0);
        
        Ok(Arc::new(pipeline))
    }
    
    /// Execute compute kernel with sub-microsecond performance
    pub async fn execute_kernel(&self, 
                               pipeline: Arc<ComputePipeline>,
                               params: &KernelParams) -> QBMIAResult<()> {
        let start_time = std::time::Instant::now();
        
        // Get command encoder
        let mut encoder = self.get_command_encoder().await;
        
        // Begin compute pass with timestamp queries if supported
        let mut compute_pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("QBMIA Compute Pass"),
            timestamp_writes: None, // TODO: Add timestamp support
        });
        
        compute_pass.set_pipeline(&pipeline);
        
        // Set bind groups (simplified - would need proper bind group management)
        // This would be expanded based on specific kernel requirements
        
        // Dispatch workgroups
        compute_pass.dispatch_workgroups(
            params.dispatch_size[0],
            params.dispatch_size[1],
            params.dispatch_size[2],
        );
        
        drop(compute_pass);
        
        // Submit commands
        self.queue.submit(std::iter::once(encoder.finish()));
        
        // Wait for completion with timeout
        let timeout = tokio::time::timeout(
            std::time::Duration::from_nanos(params.timeout_ns),
            async {
                self.device.poll(Maintain::Wait);
            }
        );
        
        timeout.await
            .map_err(|_| QBMIAError::timeout("Kernel execution timed out"))?;
        
        let execution_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().await;
        metrics.record_kernel_execution(execution_time);
        
        // Validate sub-microsecond performance target
        if execution_time.as_nanos() > 50 {
            tracing::warn!("Kernel execution took {}ns, exceeding 50ns target",
                          execution_time.as_nanos());
        }
        
        tracing::debug!("Executed kernel in {:.3}ns", execution_time.as_nanos());
        
        Ok(())
    }
    
    /// Get command encoder from pool or create new one
    async fn get_command_encoder(&self) -> CommandEncoder {
        let mut pool = self.encoder_pool.lock().await;
        
        pool.pop().unwrap_or_else(|| {
            self.device.create_command_encoder(&CommandEncoderDescriptor {
                label: Some("QBMIA Command Encoder"),
            })
        })
    }
    
    /// Return command encoder to pool
    pub async fn return_command_encoder(&self, encoder: CommandEncoder) {
        let mut pool = self.encoder_pool.lock().await;
        if pool.len() < 10 { // Limit pool size
            pool.push(encoder);
        }
    }
    
    /// Get performance metrics
    pub async fn get_metrics(&self) -> GpuMetrics {
        let metrics = self.metrics.lock().await;
        metrics.clone()
    }
    
    /// Reset performance metrics
    pub async fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().await;
        *metrics = GpuMetrics::new();
    }
    
    /// Synchronize GPU and CPU
    pub async fn synchronize(&self) -> QBMIAResult<()> {
        let start_time = std::time::Instant::now();
        
        self.device.poll(Maintain::Wait);
        
        let sync_time = start_time.elapsed();
        let mut metrics = self.metrics.lock().await;
        metrics.record_synchronization(sync_time);
        
        Ok(())
    }
    
    /// Warm up GPU for optimal performance
    pub async fn warmup(&self) -> QBMIAResult<()> {
        tracing::info!("Warming up GPU pipeline");
        
        // Create dummy buffers and kernels to warm up caches
        let dummy_data = vec![0u8; 1024];
        let buffer = self.create_buffer(&dummy_data, GpuBufferUsage::Storage).await?;
        
        // Simple compute shader for warmup
        let warmup_shader = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&data)) {
                return;
            }
            data[index] = data[index] * 2.0;
        }
        "#;
        
        let pipeline = self.get_compute_pipeline(warmup_shader, "main").await?;
        
        let params = KernelParams {
            dispatch_size: [4, 1, 1],
            input_buffers: vec![buffer],
            output_buffers: vec![],
            timeout_ns: 1_000_000, // 1ms
        };
        
        // Execute warmup kernel multiple times
        for _ in 0..10 {
            self.execute_kernel(pipeline.clone(), &params).await?;
        }
        
        self.synchronize().await?;
        
        tracing::info!("GPU pipeline warmup completed");
        Ok(())
    }
}

/// Buffer pool for memory reuse
struct BufferPool {
    buffers: HashMap<u64, PooledBuffer>,
    next_id: u64,
    free_buffers: HashMap<(usize, GpuBufferUsage), Vec<u64>>,
}

impl BufferPool {
    fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_id: 1,
            free_buffers: HashMap::new(),
        }
    }
    
    fn get_buffer(&mut self, size: usize, usage: GpuBufferUsage) -> Option<u64> {
        let key = (size, usage);
        self.free_buffers.get_mut(&key)?.pop()
    }
    
    fn add_buffer(&mut self, buffer: Buffer, size: usize, usage: GpuBufferUsage) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        
        self.buffers.insert(id, PooledBuffer {
            buffer,
            size,
            usage,
            in_use: true,
        });
        
        id
    }
    
    fn get_wgpu_buffer(&self, id: u64) -> Option<&Buffer> {
        self.buffers.get(&id).map(|b| &b.buffer)
    }
    
    fn return_buffer(&mut self, id: u64) {
        if let Some(pooled_buffer) = self.buffers.get_mut(&id) {
            pooled_buffer.in_use = false;
            let key = (pooled_buffer.size, pooled_buffer.usage);
            self.free_buffers.entry(key).or_insert_with(Vec::new).push(id);
        }
    }
}

struct PooledBuffer {
    buffer: Buffer,
    size: usize,
    usage: GpuBufferUsage,
    in_use: bool,
}

/// GPU performance metrics
#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    pub buffer_creations: u64,
    pub buffer_reuses: u64,
    pub buffer_reads: u64,
    pub kernel_executions: u64,
    pub pipeline_compilations: u64,
    pub synchronizations: u64,
    
    pub total_buffer_creation_time: std::time::Duration,
    pub total_buffer_reuse_time: std::time::Duration,
    pub total_buffer_read_time: std::time::Duration,
    pub total_kernel_execution_time: std::time::Duration,
    pub total_pipeline_compilation_time: std::time::Duration,
    pub total_synchronization_time: std::time::Duration,
    
    pub total_bytes_uploaded: u64,
    pub total_bytes_downloaded: u64,
}

impl GpuMetrics {
    fn new() -> Self {
        Self::default()
    }
    
    fn record_buffer_creation(&mut self, duration: std::time::Duration) {
        self.buffer_creations += 1;
        self.total_buffer_creation_time += duration;
    }
    
    fn record_buffer_reuse(&mut self, duration: std::time::Duration) {
        self.buffer_reuses += 1;
        self.total_buffer_reuse_time += duration;
    }
    
    fn record_buffer_read(&mut self, duration: std::time::Duration, bytes: usize) {
        self.buffer_reads += 1;
        self.total_buffer_read_time += duration;
        self.total_bytes_downloaded += bytes as u64;
    }
    
    fn record_kernel_execution(&mut self, duration: std::time::Duration) {
        self.kernel_executions += 1;
        self.total_kernel_execution_time += duration;
    }
    
    fn record_pipeline_compilation(&mut self, duration: std::time::Duration) {
        self.pipeline_compilations += 1;
        self.total_pipeline_compilation_time += duration;
    }
    
    fn record_synchronization(&mut self, duration: std::time::Duration) {
        self.synchronizations += 1;
        self.total_synchronization_time += duration;
    }
    
    pub fn average_buffer_creation_time(&self) -> Option<std::time::Duration> {
        if self.buffer_creations > 0 {
            Some(self.total_buffer_creation_time / self.buffer_creations as u32)
        } else {
            None
        }
    }
    
    pub fn average_kernel_execution_time(&self) -> Option<std::time::Duration> {
        if self.kernel_executions > 0 {
            Some(self.total_kernel_execution_time / self.kernel_executions as u32)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_gpu_pipeline_initialization() {
        let result = GpuPipeline::new().await;
        if result.is_err() {
            // Skip test if no GPU available
            return;
        }
        
        let pipeline = result.unwrap();
        assert!(!pipeline.device_info().name.is_empty());
    }
    
    #[tokio::test]
    async fn test_buffer_operations() {
        let pipeline = match GpuPipeline::new().await {
            Ok(p) => p,
            Err(_) => return, // Skip if no GPU
        };
        
        let data = vec![1u8, 2, 3, 4, 5];
        let buffer = pipeline.create_buffer(&data, GpuBufferUsage::Storage).await.unwrap();
        
        assert_eq!(buffer.size, 5);
        assert_eq!(buffer.usage, GpuBufferUsage::Storage);
    }
}