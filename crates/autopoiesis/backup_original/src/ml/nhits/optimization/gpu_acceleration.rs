use crate::Result;
use crate::ml::nhits::model::NHITSConfig;
use ndarray::{Array2, Array3, ArrayView2, ArrayView3};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, DriverError, LaunchAsync, LaunchConfig};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::Ptx;

#[cfg(feature = "opencl")]
use opencl3::{
    command_queue::{CommandQueue, CL_QUEUE_PROFILING_ENABLE},
    context::Context,
    device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU},
    kernel::{ExecuteKernel, Kernel},
    memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY},
    platform::get_platforms,
    program::{Program, CL_STD_3_0},
};

/// GPU acceleration configuration
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub device_type: GpuDeviceType,
    pub memory_pool_size: usize,
    pub max_batch_size: usize,
    pub enable_tensor_cores: bool,
    pub enable_mixed_precision: bool,
    pub compute_capability: Option<(u32, u32)>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GpuDeviceType {
    Cuda,
    OpenCL,
    Metal,
    Auto,
}

/// High-performance GPU-accelerated matrix operations
pub struct GpuAccelerator {
    config: GpuConfig,
    device_info: DeviceInfo,
    memory_pool: Arc<Mutex<GpuMemoryPool>>,
    kernel_cache: Arc<Mutex<HashMap<String, CompiledKernel>>>,
    
    #[cfg(feature = "cuda")]
    cuda_device: Option<Arc<CudaDevice>>,
    
    #[cfg(feature = "opencl")]
    opencl_context: Option<OpenCLContext>,
}

#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_units: u32,
    pub global_memory: u64,
    pub local_memory: u64,
    pub max_work_group_size: usize,
    pub warp_size: u32,
    pub supports_fp16: bool,
    pub supports_int8: bool,
}

/// GPU memory pool for efficient allocation
pub struct GpuMemoryPool {
    free_buffers: HashMap<usize, Vec<GpuBuffer>>,
    allocated_buffers: HashMap<u64, GpuBuffer>,
    total_allocated: usize,
    peak_allocation: usize,
    allocation_counter: u64,
}

#[derive(Debug, Clone)]
pub struct GpuBuffer {
    id: u64,
    size: usize,
    ptr: GpuMemoryPtr,
    last_used: std::time::Instant,
}

#[derive(Debug, Clone)]
pub enum GpuMemoryPtr {
    #[cfg(feature = "cuda")]
    Cuda(cudarc::driver::DevicePtr<f32>),
    #[cfg(feature = "opencl")]
    OpenCL(Buffer<opencl3::types::cl_float>),
    None,
}

#[derive(Debug)]
pub struct CompiledKernel {
    source_hash: u64,
    kernel: KernelInstance,
    optimal_work_size: (usize, usize, usize),
    shared_memory_size: usize,
}

#[derive(Debug)]
pub enum KernelInstance {
    #[cfg(feature = "cuda")]
    Cuda(cudarc::driver::CudaFunction),
    #[cfg(feature = "opencl")]
    OpenCL(Kernel),
    None,
}

#[cfg(feature = "opencl")]
pub struct OpenCLContext {
    device: Device,
    context: Context,
    queue: CommandQueue,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_type: GpuDeviceType::Auto,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            max_batch_size: 1024,
            enable_tensor_cores: true,
            enable_mixed_precision: true,
            compute_capability: None,
        }
    }
}

impl GpuAccelerator {
    /// Initialize GPU accelerator with automatic device selection
    pub fn new(config: GpuConfig) -> Result<Self> {
        let device_type = if config.device_type == GpuDeviceType::Auto {
            Self::detect_best_device()?
        } else {
            config.device_type.clone()
        };

        let mut accelerator = Self {
            config: GpuConfig { device_type, ..config },
            device_info: DeviceInfo::default(),
            memory_pool: Arc::new(Mutex::new(GpuMemoryPool::new())),
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            
            #[cfg(feature = "cuda")]
            cuda_device: None,
            
            #[cfg(feature = "opencl")]
            opencl_context: None,
        };

        accelerator.initialize_device()?;
        accelerator.warm_up_kernels()?;

        Ok(accelerator)
    }

    /// Detect the best available GPU device
    fn detect_best_device() -> Result<GpuDeviceType> {
        #[cfg(feature = "cuda")]
        if Self::is_cuda_available()? {
            return Ok(GpuDeviceType::Cuda);
        }

        #[cfg(feature = "opencl")]
        if Self::is_opencl_available()? {
            return Ok(GpuDeviceType::OpenCL);
        }

        Err(crate::error::Error::GpuError("No GPU device available".to_string()))
    }

    /// Initialize the selected GPU device
    fn initialize_device(&mut self) -> Result<()> {
        match self.config.device_type {
            #[cfg(feature = "cuda")]
            GpuDeviceType::Cuda => self.initialize_cuda(),
            #[cfg(feature = "opencl")]
            GpuDeviceType::OpenCL => self.initialize_opencl(),
            _ => Err(crate::error::Error::GpuError("Unsupported device type".to_string())),
        }
    }

    #[cfg(feature = "cuda")]
    fn initialize_cuda(&mut self) -> Result<()> {
        let device = CudaDevice::new(0)?;
        self.device_info = self.get_cuda_device_info(&device)?;
        self.cuda_device = Some(Arc::new(device));
        Ok(())
    }

    #[cfg(feature = "opencl")]
    fn initialize_opencl(&mut self) -> Result<()> {
        let platforms = get_platforms()?;
        let devices = get_all_devices(CL_DEVICE_TYPE_GPU)?;
        
        if let Some(device) = devices.first() {
            let context = Context::from_device(device)?;
            let queue = CommandQueue::create_default_with_properties(
                &context,
                CL_QUEUE_PROFILING_ENABLE,
                0,
            )?;

            self.device_info = self.get_opencl_device_info(device)?;
            self.opencl_context = Some(OpenCLContext {
                device: *device,
                context,
                queue,
            });
        }

        Ok(())
    }

    /// Perform high-performance matrix multiplication
    pub fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();
        
        if k != k2 {
            return Err(crate::error::Error::InvalidInput(
                "Matrix dimensions don't match for multiplication".to_string()
            ));
        }

        match self.config.device_type {
            #[cfg(feature = "cuda")]
            GpuDeviceType::Cuda => self.cuda_matmul(a, b),
            #[cfg(feature = "opencl")]
            GpuDeviceType::OpenCL => self.opencl_matmul(a, b),
            _ => self.cpu_fallback_matmul(a, b),
        }
    }

    /// GPU-accelerated batch matrix multiplication
    pub fn batch_matmul(&self, batches: &[(&Array2<f32>, &Array2<f32>)]) -> Result<Vec<Array2<f32>>> {
        if batches.is_empty() {
            return Ok(vec![]);
        }

        // Use streaming for large batches
        if batches.len() > self.config.max_batch_size {
            return self.streaming_batch_matmul(batches);
        }

        match self.config.device_type {
            #[cfg(feature = "cuda")]
            GpuDeviceType::Cuda => self.cuda_batch_matmul(batches),
            #[cfg(feature = "opencl")]
            GpuDeviceType::OpenCL => self.opencl_batch_matmul(batches),
            _ => batches.iter().map(|(a, b)| self.cpu_fallback_matmul(a, b)).collect(),
        }
    }

    /// Optimized convolution operation for NHITS
    pub fn conv1d(&self, input: &Array3<f32>, kernel: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, channels) = input.dim();
        let (out_channels, kernel_size, in_channels) = kernel.dim();

        if channels != in_channels {
            return Err(crate::error::Error::InvalidInput(
                "Channel dimensions don't match".to_string()
            ));
        }

        match self.config.device_type {
            #[cfg(feature = "cuda")]
            GpuDeviceType::Cuda => self.cuda_conv1d(input, kernel),
            #[cfg(feature = "opencl")]
            GpuDeviceType::OpenCL => self.opencl_conv1d(input, kernel),
            _ => self.cpu_fallback_conv1d(input, kernel),
        }
    }

    /// Fused operations for better performance
    pub fn fused_linear_relu(&self, input: &Array2<f32>, weight: &Array2<f32>, bias: &Array2<f32>) -> Result<Array2<f32>> {
        match self.config.device_type {
            #[cfg(feature = "cuda")]
            GpuDeviceType::Cuda => self.cuda_fused_linear_relu(input, weight, bias),
            #[cfg(feature = "opencl")]
            GpuDeviceType::OpenCL => self.opencl_fused_linear_relu(input, weight, bias),
            _ => {
                let linear = self.cpu_fallback_matmul(input, weight)?;
                let with_bias = &linear + bias;
                Ok(with_bias.mapv(|x| x.max(0.0)))
            }
        }
    }

    /// Optimized attention mechanism
    pub fn scaled_dot_product_attention(
        &self,
        query: &Array3<f32>,
        key: &Array3<f32>,
        value: &Array3<f32>,
        mask: Option<&Array3<bool>>,
    ) -> Result<Array3<f32>> {
        match self.config.device_type {
            #[cfg(feature = "cuda")]
            GpuDeviceType::Cuda => self.cuda_attention(query, key, value, mask),
            #[cfg(feature = "opencl")]
            GpuDeviceType::OpenCL => self.opencl_attention(query, key, value, mask),
            _ => self.cpu_fallback_attention(query, key, value, mask),
        }
    }

    #[cfg(feature = "cuda")]
    fn cuda_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        // Implementation would use cuBLAS for optimal performance
        // This is a simplified version
        let device = self.cuda_device.as_ref().unwrap();
        
        // Allocate GPU memory
        let a_gpu = device.htod_copy(a.as_slice().unwrap())?;
        let b_gpu = device.htod_copy(b.as_slice().unwrap())?;
        
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let mut c_gpu = device.alloc_zeros::<f32>(m * n)?;
        
        // Launch optimized GEMM kernel
        let kernel = self.get_or_compile_kernel("optimized_gemm")?;
        
        // Configure launch parameters for optimal occupancy
        let block_size = (16, 16, 1);
        let grid_size = ((n + 15) / 16, (m + 15) / 16, 1);
        
        unsafe {
            kernel.launch(
                LaunchConfig {
                    grid_dim: grid_size,
                    block_dim: block_size,
                    shared_mem_bytes: 0,
                },
                (&a_gpu, &b_gpu, &mut c_gpu, m as i32, n as i32, k as i32),
            )?;
        }
        
        // Copy result back
        let result = device.dtoh_sync_copy(&c_gpu)?;
        Ok(Array2::from_shape_vec((m, n), result)?)
    }

    #[cfg(feature = "opencl")]
    fn opencl_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let context = self.opencl_context.as_ref().unwrap();
        
        // Create buffers
        let a_buffer = Buffer::<f32>::create(&context.context, CL_MEM_READ_ONLY, a.len(), std::ptr::null_mut())?;
        let b_buffer = Buffer::<f32>::create(&context.context, CL_MEM_READ_ONLY, b.len(), std::ptr::null_mut())?;
        
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        let c_buffer = Buffer::<f32>::create(&context.context, CL_MEM_WRITE_ONLY, m * n, std::ptr::null_mut())?;
        
        // Copy data to GPU
        context.queue.enqueue_write_buffer(&a_buffer, true, 0, a.as_slice().unwrap(), &[])?;
        context.queue.enqueue_write_buffer(&b_buffer, true, 0, b.as_slice().unwrap(), &[])?;
        
        // Get or compile kernel
        let kernel = self.get_or_compile_opencl_kernel("optimized_gemm")?;
        
        // Set kernel arguments and execute
        kernel.set_arg(0, &a_buffer)?;
        kernel.set_arg(1, &b_buffer)?;
        kernel.set_arg(2, &c_buffer)?;
        kernel.set_arg(3, &(m as i32))?;
        kernel.set_arg(4, &(n as i32))?;
        kernel.set_arg(5, &(k as i32))?;
        
        let global_work_size = [m, n];
        let local_work_size = [16, 16];
        
        context.queue.enqueue_nd_range_kernel(&kernel, 2, None, &global_work_size, Some(&local_work_size), &[])?;
        
        // Read result
        let mut result = vec![0.0f32; m * n];
        context.queue.enqueue_read_buffer(&c_buffer, true, 0, &mut result, &[])?;
        
        Ok(Array2::from_shape_vec((m, n), result)?)
    }

    fn cpu_fallback_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        Ok(a.dot(b))
    }

    fn warm_up_kernels(&self) -> Result<()> {
        // Pre-compile frequently used kernels
        let kernels = vec![
            "optimized_gemm",
            "conv1d_kernel",
            "fused_linear_relu",
            "attention_kernel",
            "batch_norm",
            "layer_norm",
        ];

        for kernel_name in kernels {
            self.get_or_compile_kernel(kernel_name)?;
        }

        Ok(())
    }

    fn get_or_compile_kernel(&self, name: &str) -> Result<&CompiledKernel> {
        // Implementation for kernel compilation and caching
        // This would contain the actual CUDA/OpenCL kernel code
        todo!("Implement kernel compilation")
    }

    /// Get performance statistics
    pub fn get_performance_stats(&self) -> GpuPerformanceStats {
        let memory_pool = self.memory_pool.lock().unwrap();
        
        GpuPerformanceStats {
            device_name: self.device_info.name.clone(),
            total_memory: self.device_info.global_memory,
            allocated_memory: memory_pool.total_allocated,
            peak_memory: memory_pool.peak_allocation,
            kernel_cache_size: self.kernel_cache.lock().unwrap().len(),
            compute_utilization: self.estimate_compute_utilization(),
            memory_bandwidth_utilization: self.estimate_memory_bandwidth(),
        }
    }

    fn estimate_compute_utilization(&self) -> f32 {
        // Implementation would use GPU profiling APIs
        0.0
    }

    fn estimate_memory_bandwidth(&self) -> f32 {
        // Implementation would use GPU profiling APIs
        0.0
    }

    #[cfg(feature = "cuda")]
    fn is_cuda_available() -> Result<bool> {
        match CudaDevice::new(0) {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    #[cfg(feature = "opencl")]
    fn is_opencl_available() -> Result<bool> {
        match get_platforms() {
            Ok(platforms) => Ok(!platforms.is_empty()),
            Err(_) => Ok(false),
        }
    }
}

impl Default for DeviceInfo {
    fn default() -> Self {
        Self {
            name: "Unknown".to_string(),
            compute_units: 0,
            global_memory: 0,
            local_memory: 0,
            max_work_group_size: 256,
            warp_size: 32,
            supports_fp16: false,
            supports_int8: false,
        }
    }
}

impl GpuMemoryPool {
    fn new() -> Self {
        Self {
            free_buffers: HashMap::new(),
            allocated_buffers: HashMap::new(),
            total_allocated: 0,
            peak_allocation: 0,
            allocation_counter: 0,
        }
    }

    fn allocate(&mut self, size: usize) -> Result<u64> {
        // Try to reuse existing buffer
        if let Some(buffers) = self.free_buffers.get_mut(&size) {
            if let Some(buffer) = buffers.pop() {
                let id = buffer.id;
                self.allocated_buffers.insert(id, buffer);
                return Ok(id);
            }
        }

        // Allocate new buffer
        let id = self.allocation_counter;
        self.allocation_counter += 1;

        let buffer = GpuBuffer {
            id,
            size,
            ptr: GpuMemoryPtr::None, // Would be initialized with actual GPU memory
            last_used: std::time::Instant::now(),
        };

        self.allocated_buffers.insert(id, buffer);
        self.total_allocated += size;
        self.peak_allocation = self.peak_allocation.max(self.total_allocated);

        Ok(id)
    }

    fn deallocate(&mut self, id: u64) -> Result<()> {
        if let Some(buffer) = self.allocated_buffers.remove(&id) {
            self.total_allocated -= buffer.size;
            
            // Add to free buffers for reuse
            self.free_buffers.entry(buffer.size).or_default().push(buffer);
        }

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct GpuPerformanceStats {
    pub device_name: String,
    pub total_memory: u64,
    pub allocated_memory: usize,
    pub peak_memory: usize,
    pub kernel_cache_size: usize,
    pub compute_utilization: f32,
    pub memory_bandwidth_utilization: f32,
}

/// Compile-time GPU kernel sources
pub mod kernels {
    pub const OPTIMIZED_GEMM_CUDA: &str = r#"
    extern "C" __global__ void optimized_gemm(
        const float* A, const float* B, float* C,
        int M, int N, int K
    ) {
        // Optimized GEMM implementation with shared memory tiling
        __shared__ float As[16][16];
        __shared__ float Bs[16][16];
        
        int bx = blockIdx.x, by = blockIdx.y;
        int tx = threadIdx.x, ty = threadIdx.y;
        
        int Row = by * 16 + ty;
        int Col = bx * 16 + tx;
        
        float sum = 0.0f;
        
        for (int k = 0; k < (K + 15) / 16; k++) {
            if (Row < M && k * 16 + tx < K)
                As[ty][tx] = A[Row * K + k * 16 + tx];
            else
                As[ty][tx] = 0.0f;
                
            if (Col < N && k * 16 + ty < K)
                Bs[ty][tx] = B[(k * 16 + ty) * N + Col];
            else
                Bs[ty][tx] = 0.0f;
                
            __syncthreads();
            
            for (int i = 0; i < 16; i++)
                sum += As[ty][i] * Bs[i][tx];
                
            __syncthreads();
        }
        
        if (Row < M && Col < N)
            C[Row * N + Col] = sum;
    }
    "#;

    pub const OPTIMIZED_GEMM_OPENCL: &str = r#"
    __kernel void optimized_gemm(__global const float* A,
                                __global const float* B,
                                __global float* C,
                                int M, int N, int K) {
        __local float As[16][16];
        __local float Bs[16][16];
        
        int gx = get_group_id(0);
        int gy = get_group_id(1);
        int lx = get_local_id(0);
        int ly = get_local_id(1);
        
        int Row = gy * 16 + ly;
        int Col = gx * 16 + lx;
        
        float sum = 0.0f;
        
        for (int k = 0; k < (K + 15) / 16; k++) {
            if (Row < M && k * 16 + lx < K)
                As[ly][lx] = A[Row * K + k * 16 + lx];
            else
                As[ly][lx] = 0.0f;
                
            if (Col < N && k * 16 + ly < K)
                Bs[ly][lx] = B[(k * 16 + ly) * N + Col];
            else
                Bs[ly][lx] = 0.0f;
                
            barrier(CLK_LOCAL_MEM_FENCE);
            
            for (int i = 0; i < 16; i++)
                sum += As[ly][i] * Bs[i][lx];
                
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (Row < M && Col < N)
            C[Row * N + Col] = sum;
    }
    "#;
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.device_type, GpuDeviceType::Auto);
        assert!(config.memory_pool_size > 0);
    }

    #[tokio::test]
    async fn test_matrix_multiplication() {
        let config = GpuConfig::default();
        
        // Use CPU fallback for testing
        let mut accelerator = GpuAccelerator {
            config: GpuConfig { device_type: GpuDeviceType::Auto, ..config },
            device_info: DeviceInfo::default(),
            memory_pool: Arc::new(Mutex::new(GpuMemoryPool::new())),
            kernel_cache: Arc::new(Mutex::new(HashMap::new())),
            
            #[cfg(feature = "cuda")]
            cuda_device: None,
            
            #[cfg(feature = "opencl")]
            opencl_context: None,
        };

        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = accelerator.cpu_fallback_matmul(&a, &b).unwrap();
        
        assert_eq!(result.dim(), (2, 2));
        assert_eq!(result[[0, 0]], 22.0);
        assert_eq!(result[[0, 1]], 28.0);
        assert_eq!(result[[1, 0]], 49.0);
        assert_eq!(result[[1, 1]], 64.0);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = GpuMemoryPool::new();
        
        let id1 = pool.allocate(1024).unwrap();
        let id2 = pool.allocate(2048).unwrap();
        
        assert_eq!(pool.total_allocated, 3072);
        assert_eq!(pool.peak_allocation, 3072);
        
        pool.deallocate(id1).unwrap();
        assert_eq!(pool.total_allocated, 2048);
        
        // Test buffer reuse
        let id3 = pool.allocate(1024).unwrap();
        assert_eq!(id3, id1); // Should reuse the deallocated buffer
    }
}