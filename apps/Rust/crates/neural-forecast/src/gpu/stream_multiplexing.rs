//! GPU stream multiplexing for concurrent execution
//!
//! This module provides advanced stream management for overlapping
//! computation and memory transfers to maximize GPU utilization.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::time::Instant;

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaStream, CudaEvent};

use crate::{Result, NeuralForecastError};

/// GPU stream multiplexer for concurrent execution
pub struct StreamMultiplexer {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    compute_streams: Vec<Arc<Mutex<ComputeStream>>>,
    transfer_streams: Vec<Arc<Mutex<TransferStream>>>,
    scheduler: Arc<Mutex<StreamScheduler>>,
    config: StreamConfig,
}

/// Configuration for stream multiplexing
#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub num_compute_streams: usize,
    pub num_transfer_streams: usize,
    pub enable_priority_scheduling: bool,
    pub max_concurrent_ops: usize,
    pub memory_pool_size: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            num_compute_streams: 4,
            num_transfer_streams: 2,
            enable_priority_scheduling: true,
            max_concurrent_ops: 16,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
        }
    }
}

/// Compute stream for GPU kernels
pub struct ComputeStream {
    #[cfg(feature = "cuda")]
    cuda_stream: CudaStream,
    id: usize,
    is_busy: bool,
    current_task: Option<StreamTask>,
    priority_queue: VecDeque<StreamTask>,
    metrics: StreamMetrics,
}

/// Transfer stream for memory operations
pub struct TransferStream {
    #[cfg(feature = "cuda")]
    cuda_stream: CudaStream,
    id: usize,
    is_busy: bool,
    current_transfer: Option<TransferTask>,
    transfer_queue: VecDeque<TransferTask>,
    metrics: StreamMetrics,
}

/// Stream scheduler for task distribution
pub struct StreamScheduler {
    compute_queue: VecDeque<StreamTask>,
    transfer_queue: VecDeque<TransferTask>,
    active_tasks: HashMap<usize, ActiveTask>,
    task_dependencies: HashMap<usize, Vec<usize>>,
    next_task_id: usize,
    scheduling_policy: SchedulingPolicy,
}

/// Task for stream execution
#[derive(Debug, Clone)]
pub struct StreamTask {
    pub id: usize,
    pub operation: StreamOperation,
    pub priority: TaskPriority,
    pub dependencies: Vec<usize>,
    pub estimated_duration: std::time::Duration,
    pub callback: Option<Box<dyn Fn(Result<()>) + Send + Sync>>,
}

/// Transfer task for memory operations
#[derive(Debug, Clone)]
pub struct TransferTask {
    pub id: usize,
    pub operation: TransferOperation,
    pub priority: TaskPriority,
    pub size_bytes: usize,
    pub callback: Option<Box<dyn Fn(Result<()>) + Send + Sync>>,
}

/// Stream operation types
#[derive(Debug, Clone)]
pub enum StreamOperation {
    MatrixMultiply {
        m: usize,
        n: usize,
        k: usize,
        use_tensor_cores: bool,
    },
    Convolution {
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        kernel_size: usize,
    },
    Attention {
        batch_size: usize,
        num_heads: usize,
        seq_length: usize,
        head_dim: usize,
        use_flash_attention: bool,
    },
    Activation {
        size: usize,
        activation_type: ActivationType,
    },
    Normalization {
        size: usize,
        norm_type: NormalizationType,
    },
    Custom {
        name: String,
        params: Vec<u8>,
    },
}

/// Transfer operation types
#[derive(Debug, Clone)]
pub enum TransferOperation {
    HostToDevice {
        src_ptr: *const u8,
        dst_offset: usize,
        size: usize,
    },
    DeviceToHost {
        src_offset: usize,
        dst_ptr: *mut u8,
        size: usize,
    },
    DeviceToDevice {
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    },
}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TaskPriority {
    Low = 0,
    Medium = 1,
    High = 2,
    Critical = 3,
}

/// Activation function types
#[derive(Debug, Clone, Copy)]
pub enum ActivationType {
    ReLU,
    GELU,
    SiLU,
    Tanh,
    Sigmoid,
}

/// Normalization types
#[derive(Debug, Clone, Copy)]
pub enum NormalizationType {
    BatchNorm,
    LayerNorm,
    GroupNorm,
}

/// Scheduling policies
#[derive(Debug, Clone, Copy)]
pub enum SchedulingPolicy {
    RoundRobin,
    PriorityBased,
    ShortestJobFirst,
    LoadBalanced,
}

/// Active task tracking
#[derive(Debug)]
pub struct ActiveTask {
    pub task_id: usize,
    pub stream_id: usize,
    pub start_time: Instant,
    pub estimated_completion: Instant,
}

/// Stream performance metrics
#[derive(Debug, Clone)]
pub struct StreamMetrics {
    pub tasks_completed: usize,
    pub total_execution_time: std::time::Duration,
    pub average_latency: std::time::Duration,
    pub utilization: f32,
    pub throughput: f32,
}

impl StreamMultiplexer {
    /// Create new stream multiplexer
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>, config: StreamConfig) -> Result<Self> {
        let mut compute_streams = Vec::new();
        let mut transfer_streams = Vec::new();
        
        // Create compute streams
        for i in 0..config.num_compute_streams {
            let cuda_stream = device.fork_default_stream()
                .map_err(|e| NeuralForecastError::GpuError(format!("Failed to create compute stream {}: {}", i, e)))?;
            
            let stream = ComputeStream {
                cuda_stream,
                id: i,
                is_busy: false,
                current_task: None,
                priority_queue: VecDeque::new(),
                metrics: StreamMetrics::default(),
            };
            
            compute_streams.push(Arc::new(Mutex::new(stream)));
        }
        
        // Create transfer streams
        for i in 0..config.num_transfer_streams {
            let cuda_stream = device.fork_default_stream()
                .map_err(|e| NeuralForecastError::GpuError(format!("Failed to create transfer stream {}: {}", i, e)))?;
            
            let stream = TransferStream {
                cuda_stream,
                id: i,
                is_busy: false,
                current_transfer: None,
                transfer_queue: VecDeque::new(),
                metrics: StreamMetrics::default(),
            };
            
            transfer_streams.push(Arc::new(Mutex::new(stream)));
        }
        
        let scheduler = StreamScheduler {
            compute_queue: VecDeque::new(),
            transfer_queue: VecDeque::new(),
            active_tasks: HashMap::new(),
            task_dependencies: HashMap::new(),
            next_task_id: 0,
            scheduling_policy: SchedulingPolicy::PriorityBased,
        };
        
        Ok(Self {
            device,
            compute_streams,
            transfer_streams,
            scheduler: Arc::new(Mutex::new(scheduler)),
            config,
        })
    }
    
    /// CPU-only version
    #[cfg(not(feature = "cuda"))]
    pub fn new(_config: StreamConfig) -> Result<Self> {
        Ok(Self {
            compute_streams: Vec::new(),
            transfer_streams: Vec::new(),
            scheduler: Arc::new(Mutex::new(StreamScheduler::default())),
            config: _config,
        })
    }
    
    /// Submit compute task for execution
    pub fn submit_compute_task(&self, operation: StreamOperation, priority: TaskPriority) -> Result<usize> {
        let mut scheduler = self.scheduler.lock().unwrap();
        let task_id = scheduler.next_task_id;
        scheduler.next_task_id += 1;
        
        let estimated_duration = self.estimate_task_duration(&operation);
        
        let task = StreamTask {
            id: task_id,
            operation,
            priority,
            dependencies: Vec::new(),
            estimated_duration,
            callback: None,
        };
        
        scheduler.compute_queue.push_back(task);
        
        // Try to schedule immediately
        drop(scheduler);
        self.schedule_tasks()?;
        
        Ok(task_id)
    }
    
    /// Submit transfer task
    pub fn submit_transfer_task(&self, operation: TransferOperation, priority: TaskPriority) -> Result<usize> {
        let mut scheduler = self.scheduler.lock().unwrap();
        let task_id = scheduler.next_task_id;
        scheduler.next_task_id += 1;
        
        let size_bytes = self.get_transfer_size(&operation);
        
        let task = TransferTask {
            id: task_id,
            operation,
            priority,
            size_bytes,
            callback: None,
        };
        
        scheduler.transfer_queue.push_back(task);
        
        // Try to schedule immediately
        drop(scheduler);
        self.schedule_tasks()?;
        
        Ok(task_id)
    }
    
    /// Schedule tasks to available streams
    fn schedule_tasks(&self) -> Result<()> {
        let mut scheduler = self.scheduler.lock().unwrap();
        
        // Schedule compute tasks
        self.schedule_compute_tasks(&mut scheduler)?;
        
        // Schedule transfer tasks
        self.schedule_transfer_tasks(&mut scheduler)?;
        
        Ok(())
    }
    
    /// Schedule compute tasks
    fn schedule_compute_tasks(&self, scheduler: &mut StreamScheduler) -> Result<()> {
        // Sort tasks by priority if enabled
        if self.config.enable_priority_scheduling {
            let mut tasks: Vec<_> = scheduler.compute_queue.drain(..).collect();
            tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
            scheduler.compute_queue.extend(tasks);
        }
        
        // Try to assign tasks to available streams
        for stream_arc in &self.compute_streams {
            let mut stream = stream_arc.lock().unwrap();
            
            if !stream.is_busy && !scheduler.compute_queue.is_empty() {
                if let Some(task) = scheduler.compute_queue.pop_front() {
                    // Check dependencies
                    if self.dependencies_satisfied(&task, scheduler) {
                        stream.current_task = Some(task.clone());
                        stream.is_busy = true;
                        
                        let active_task = ActiveTask {
                            task_id: task.id,
                            stream_id: stream.id,
                            start_time: Instant::now(),
                            estimated_completion: Instant::now() + task.estimated_duration,
                        };
                        
                        scheduler.active_tasks.insert(task.id, active_task);
                        
                        // Launch task asynchronously
                        let stream_clone = stream_arc.clone();
                        let task_clone = task.clone();
                        
                        tokio::spawn(async move {
                            let _ = Self::execute_compute_task(stream_clone, task_clone).await;
                        });
                    } else {
                        // Put task back in queue
                        scheduler.compute_queue.push_front(task);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Schedule transfer tasks
    fn schedule_transfer_tasks(&self, scheduler: &mut StreamScheduler) -> Result<()> {
        // Sort by priority
        if self.config.enable_priority_scheduling {
            let mut tasks: Vec<_> = scheduler.transfer_queue.drain(..).collect();
            tasks.sort_by(|a, b| b.priority.cmp(&a.priority));
            scheduler.transfer_queue.extend(tasks);
        }
        
        // Assign to available transfer streams
        for stream_arc in &self.transfer_streams {
            let mut stream = stream_arc.lock().unwrap();
            
            if !stream.is_busy && !scheduler.transfer_queue.is_empty() {
                if let Some(task) = scheduler.transfer_queue.pop_front() {
                    stream.current_transfer = Some(task.clone());
                    stream.is_busy = true;
                    
                    // Launch transfer asynchronously
                    let stream_clone = stream_arc.clone();
                    let task_clone = task.clone();
                    
                    tokio::spawn(async move {
                        let _ = Self::execute_transfer_task(stream_clone, task_clone).await;
                    });
                }
            }
        }
        
        Ok(())
    }
    
    /// Execute compute task
    async fn execute_compute_task(
        stream: Arc<Mutex<ComputeStream>>,
        task: StreamTask,
    ) -> Result<()> {
        let start_time = Instant::now();
        
        // Simulate task execution
        let duration = match &task.operation {
            StreamOperation::MatrixMultiply { m, n, k, use_tensor_cores } => {
                let ops = 2 * m * n * k; // FLOPS
                let throughput = if *use_tensor_cores { 125e12 } else { 19.5e12 };
                std::time::Duration::from_nanos((ops as f64 / throughput * 1e9) as u64)
            }
            StreamOperation::Attention { batch_size, num_heads, seq_length, head_dim, use_flash_attention } => {
                let ops = 4 * batch_size * num_heads * seq_length * seq_length * head_dim;
                let throughput = if *use_flash_attention { 100e12 } else { 50e12 };
                std::time::Duration::from_nanos((ops as f64 / throughput * 1e9) as u64)
            }
            _ => std::time::Duration::from_micros(100), // Default
        };
        
        tokio::time::sleep(duration).await;
        
        // Update stream metrics
        {
            let mut stream_guard = stream.lock().unwrap();
            stream_guard.is_busy = false;
            stream_guard.current_task = None;
            stream_guard.metrics.tasks_completed += 1;
            
            let execution_time = start_time.elapsed();
            stream_guard.metrics.total_execution_time += execution_time;
            stream_guard.metrics.average_latency = 
                stream_guard.metrics.total_execution_time / stream_guard.metrics.tasks_completed as u32;
        }
        
        Ok(())
    }
    
    /// Execute transfer task
    async fn execute_transfer_task(
        stream: Arc<Mutex<TransferStream>>,
        task: TransferTask,
    ) -> Result<()> {
        let start_time = Instant::now();
        
        // Simulate transfer
        let bandwidth = 1555e9; // GPU memory bandwidth (GB/s)
        let transfer_time = task.size_bytes as f64 / bandwidth;
        let duration = std::time::Duration::from_nanos((transfer_time * 1e9) as u64);
        
        tokio::time::sleep(duration).await;
        
        // Update stream metrics
        {
            let mut stream_guard = stream.lock().unwrap();
            stream_guard.is_busy = false;
            stream_guard.current_transfer = None;
            stream_guard.metrics.tasks_completed += 1;
            
            let execution_time = start_time.elapsed();
            stream_guard.metrics.total_execution_time += execution_time;
        }
        
        Ok(())
    }
    
    /// Check if task dependencies are satisfied
    fn dependencies_satisfied(&self, task: &StreamTask, scheduler: &StreamScheduler) -> bool {
        for &dep_id in &task.dependencies {
            if scheduler.active_tasks.contains_key(&dep_id) {
                return false;
            }
        }
        true
    }
    
    /// Estimate task duration
    fn estimate_task_duration(&self, operation: &StreamOperation) -> std::time::Duration {
        match operation {
            StreamOperation::MatrixMultiply { m, n, k, use_tensor_cores } => {
                let ops = 2 * m * n * k;
                let throughput = if *use_tensor_cores { 125e12 } else { 19.5e12 };
                std::time::Duration::from_nanos((ops as f64 / throughput * 1e9) as u64)
            }
            StreamOperation::Attention { batch_size, num_heads, seq_length, head_dim, use_flash_attention } => {
                let ops = 4 * batch_size * num_heads * seq_length * seq_length * head_dim;
                let throughput = if *use_flash_attention { 100e12 } else { 50e12 };
                std::time::Duration::from_nanos((ops as f64 / throughput * 1e9) as u64)
            }
            _ => std::time::Duration::from_micros(100),
        }
    }
    
    /// Get transfer size
    fn get_transfer_size(&self, operation: &TransferOperation) -> usize {
        match operation {
            TransferOperation::HostToDevice { size, .. } => *size,
            TransferOperation::DeviceToHost { size, .. } => *size,
            TransferOperation::DeviceToDevice { size, .. } => *size,
        }
    }
    
    /// Get stream utilization metrics
    pub fn get_metrics(&self) -> StreamMultiplexerMetrics {
        let mut compute_metrics = Vec::new();
        let mut transfer_metrics = Vec::new();
        
        for stream_arc in &self.compute_streams {
            let stream = stream_arc.lock().unwrap();
            compute_metrics.push(stream.metrics.clone());
        }
        
        for stream_arc in &self.transfer_streams {
            let stream = stream_arc.lock().unwrap();
            transfer_metrics.push(stream.metrics.clone());
        }
        
        let scheduler = self.scheduler.lock().unwrap();
        
        StreamMultiplexerMetrics {
            compute_streams: compute_metrics,
            transfer_streams: transfer_metrics,
            pending_compute_tasks: scheduler.compute_queue.len(),
            pending_transfer_tasks: scheduler.transfer_queue.len(),
            active_tasks: scheduler.active_tasks.len(),
        }
    }
    
    /// Wait for all tasks to complete
    pub async fn wait_for_completion(&self) -> Result<()> {
        loop {
            let metrics = self.get_metrics();
            
            if metrics.pending_compute_tasks == 0 && 
               metrics.pending_transfer_tasks == 0 && 
               metrics.active_tasks == 0 {
                break;
            }
            
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
        
        Ok(())
    }
}

impl Default for StreamScheduler {
    fn default() -> Self {
        Self {
            compute_queue: VecDeque::new(),
            transfer_queue: VecDeque::new(),
            active_tasks: HashMap::new(),
            task_dependencies: HashMap::new(),
            next_task_id: 0,
            scheduling_policy: SchedulingPolicy::PriorityBased,
        }
    }
}

impl Default for StreamMetrics {
    fn default() -> Self {
        Self {
            tasks_completed: 0,
            total_execution_time: std::time::Duration::from_secs(0),
            average_latency: std::time::Duration::from_secs(0),
            utilization: 0.0,
            throughput: 0.0,
        }
    }
}

/// Overall multiplexer metrics
#[derive(Debug, Clone)]
pub struct StreamMultiplexerMetrics {
    pub compute_streams: Vec<StreamMetrics>,
    pub transfer_streams: Vec<StreamMetrics>,
    pub pending_compute_tasks: usize,
    pub pending_transfer_tasks: usize,
    pub active_tasks: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_stream_config() {
        let config = StreamConfig::default();
        assert_eq!(config.num_compute_streams, 4);
        assert_eq!(config.num_transfer_streams, 2);
        assert!(config.enable_priority_scheduling);
    }
    
    #[test]
    fn test_task_priority_ordering() {
        let high = TaskPriority::High;
        let medium = TaskPriority::Medium;
        let low = TaskPriority::Low;
        
        assert!(high > medium);
        assert!(medium > low);
        assert!(high > low);
    }
    
    #[tokio::test]
    async fn test_stream_multiplexer_creation() {
        let config = StreamConfig::default();
        let result = StreamMultiplexer::new(config);
        
        #[cfg(feature = "cuda")]
        {
            // This would fail without GPU, so we just check the error type
            match result {
                Ok(_) => {
                    // GPU available
                }
                Err(e) => {
                    // Expected if no GPU
                    assert!(e.to_string().contains("CUDA") || e.to_string().contains("GPU"));
                }
            }
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            assert!(result.is_ok());
        }
    }
}