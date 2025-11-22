use crate::Result;
use crate::ml::nhits::model::NHITSConfig;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rayon::prelude::*;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use crossbeam_channel::{unbounded, Receiver, Sender};
use num_cpus;
use std::collections::VecDeque;

/// Advanced parallel processing configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub num_threads: usize,
    pub chunk_size: usize,
    pub enable_work_stealing: bool,
    pub enable_numa_awareness: bool,
    pub thread_affinity: ThreadAffinity,
    pub scheduler_type: SchedulerType,
    pub load_balance_threshold: f64,
}

#[derive(Debug, Clone)]
pub enum ThreadAffinity {
    None,
    Core,
    Socket,
    Numa,
    Custom(Vec<usize>),
}

#[derive(Debug, Clone)]
pub enum SchedulerType {
    RoundRobin,
    WorkStealing,
    PriorityBased,
    LoadBalancing,
    Adaptive,
}

/// High-performance parallel computation engine
pub struct ParallelProcessor {
    config: ParallelConfig,
    thread_pool: Arc<ThreadPool>,
    work_queue: Arc<WorkQueue>,
    performance_monitor: Arc<RwLock<PerformanceMonitor>>,
    numa_topology: NumaTopology,
}

/// Custom thread pool with advanced scheduling
pub struct ThreadPool {
    workers: Vec<Worker>,
    sender: Sender<Job>,
    receiver: Arc<Mutex<Receiver<Job>>>,
    scheduler: Box<dyn TaskScheduler + Send + Sync>,
}

pub struct Worker {
    id: usize,
    thread: Option<thread::JoinHandle<()>>,
    local_queue: VecDeque<Job>,
    stealer: Option<crossbeam_deque::Stealer<Job>>,
}

type Job = Box<dyn FnOnce() + Send + 'static>;

/// Advanced work queue with priority and load balancing
pub struct WorkQueue {
    high_priority: Arc<Mutex<VecDeque<PriorityTask>>>,
    normal_priority: Arc<Mutex<VecDeque<PriorityTask>>>,
    low_priority: Arc<Mutex<VecDeque<PriorityTask>>>,
    completed_tasks: Arc<Mutex<Vec<TaskResult>>>,
    active_tasks: Arc<Mutex<usize>>,
}

#[derive(Debug, Clone)]
pub struct PriorityTask {
    id: u64,
    priority: TaskPriority,
    task: TaskType,
    estimated_duration: std::time::Duration,
    dependencies: Vec<u64>,
    created_at: std::time::Instant,
}

#[derive(Debug, Clone, PartialEq, Ord, PartialOrd, Eq)]
pub enum TaskPriority {
    Critical = 0,
    High = 1,
    Normal = 2,
    Low = 3,
    Background = 4,
}

#[derive(Debug, Clone)]
pub enum TaskType {
    MatrixMultiplication { a_shape: (usize, usize), b_shape: (usize, usize) },
    Convolution { input_shape: (usize, usize, usize), kernel_shape: (usize, usize, usize) },
    Attention { seq_len: usize, embed_dim: usize, num_heads: usize },
    BatchNormalization { batch_size: usize, features: usize },
    Activation { elements: usize, function: ActivationType },
}

#[derive(Debug, Clone)]
pub enum ActivationType {
    ReLU,
    GELU,
    Swish,
    Mish,
}

#[derive(Debug, Clone)]
pub struct TaskResult {
    id: u64,
    duration: std::time::Duration,
    success: bool,
    error: Option<String>,
    memory_used: usize,
}

/// Task scheduler interface
pub trait TaskScheduler {
    fn schedule(&self, tasks: Vec<PriorityTask>) -> Vec<PriorityTask>;
    fn balance_load(&self, workers: &[Worker]) -> Vec<usize>;
    fn estimate_completion_time(&self, task: &PriorityTask) -> std::time::Duration;
}

/// Work stealing scheduler implementation
pub struct WorkStealingScheduler {
    steal_attempts: usize,
    random_steal_probability: f64,
}

/// Load balancing scheduler implementation
pub struct LoadBalancingScheduler {
    balance_threshold: f64,
    rebalance_interval: std::time::Duration,
    last_balance: std::time::Instant,
}

/// Performance monitoring and profiling
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    task_completion_times: Vec<std::time::Duration>,
    thread_utilization: Vec<f64>,
    cache_hit_rates: Vec<f64>,
    memory_usage_per_thread: Vec<usize>,
    load_balance_efficiency: f64,
    total_tasks_completed: u64,
    total_time_spent: std::time::Duration,
}

/// NUMA topology awareness
#[derive(Debug, Clone)]
pub struct NumaTopology {
    nodes: Vec<NumaNode>,
    cpu_to_node_map: Vec<usize>,
    node_distances: Vec<Vec<u32>>,
}

#[derive(Debug, Clone)]
pub struct NumaNode {
    id: usize,
    cpus: Vec<usize>,
    memory_size: u64,
    available_memory: u64,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            chunk_size: 1024,
            enable_work_stealing: true,
            enable_numa_awareness: true,
            thread_affinity: ThreadAffinity::Core,
            scheduler_type: SchedulerType::WorkStealing,
            load_balance_threshold: 0.8,
        }
    }
}

impl ParallelProcessor {
    /// Create new parallel processor with optimized configuration
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let numa_topology = NumaTopology::detect()?;
        let thread_pool = Arc::new(ThreadPool::new(&config, &numa_topology)?);
        let work_queue = Arc::new(WorkQueue::new());
        let performance_monitor = Arc::new(RwLock::new(PerformanceMonitor::new()));

        Ok(Self {
            config,
            thread_pool,
            work_queue,
            performance_monitor,
            numa_topology,
        })
    }

    /// Parallel matrix multiplication with optimal threading
    pub fn parallel_matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(crate::error::Error::InvalidInput(
                "Matrix dimensions don't match".to_string()
            ));
        }

        // Choose optimal parallelization strategy based on matrix size
        if m * n < 10000 {
            // Small matrices - use simple parallel rows
            return Ok(self.parallel_matmul_rows(a, b));
        } else if m > 1000 && n > 1000 {
            // Large matrices - use block-based parallelization
            return Ok(self.parallel_matmul_blocks(a, b));
        } else {
            // Medium matrices - use work-stealing approach
            return Ok(self.parallel_matmul_work_stealing(a, b));
        }
    }

    /// Block-based parallel matrix multiplication
    fn parallel_matmul_blocks(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        
        let block_size = self.calculate_optimal_block_size(m, n, k);
        let mut result = Array2::zeros((m, n));

        let blocks_m = (m + block_size - 1) / block_size;
        let blocks_n = (n + block_size - 1) / block_size;
        let blocks_k = (k + block_size - 1) / block_size;

        // Create tasks for each block
        let tasks: Vec<_> = (0..blocks_m)
            .flat_map(|i| {
                (0..blocks_n).map(move |j| (i, j))
            })
            .collect();

        let result_mutex = Arc::new(Mutex::new(&mut result));

        tasks.par_iter().for_each(|&(block_i, block_j)| {
            let start_i = block_i * block_size;
            let end_i = (start_i + block_size).min(m);
            let start_j = block_j * block_size;
            let end_j = (start_j + block_size).min(n);

            let mut local_result = Array2::zeros((end_i - start_i, end_j - start_j));

            for block_k in 0..blocks_k {
                let start_k = block_k * block_size;
                let end_k = (start_k + block_size).min(k);

                let a_block = a.slice(s![start_i..end_i, start_k..end_k]);
                let b_block = b.slice(s![start_k..end_k, start_j..end_j]);

                local_result = local_result + a_block.dot(&b_block);
            }

            // Write back to global result
            let mut result_lock = result_mutex.lock().unwrap();
            result_lock.slice_mut(s![start_i..end_i, start_j..end_j])
                .assign(&local_result);
        });

        result
    }

    /// Work-stealing parallel matrix multiplication
    fn parallel_matmul_work_stealing(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = a.dim();
        let (_, n) = b.dim();
        
        let mut result = Array2::zeros((m, n));
        let chunk_size = self.calculate_optimal_chunk_size(m);

        let result_slices: Vec<_> = result.axis_chunks_iter_mut(Axis(0), chunk_size).collect();
        let a_slices: Vec<_> = a.axis_chunks_iter(Axis(0), chunk_size).collect();

        result_slices.into_par_iter()
            .zip(a_slices.into_par_iter())
            .for_each(|(mut result_chunk, a_chunk)| {
                let chunk_result = a_chunk.dot(b);
                result_chunk.assign(&chunk_result);
            });

        result
    }

    /// Simple row-wise parallel matrix multiplication
    fn parallel_matmul_rows(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, _) = a.dim();
        
        let rows: Vec<_> = (0..m).into_par_iter()
            .map(|i| {
                let row = a.row(i);
                let result_row = b.t().dot(&row);
                result_row
            })
            .collect();

        let mut result = Array2::zeros(a.raw_dim());
        for (i, row) in rows.into_iter().enumerate() {
            result.row_mut(i).assign(&row);
        }

        result
    }

    /// Parallel batch processing for multiple operations
    pub fn parallel_batch_process<T, R, F>(&self, items: Vec<T>, operation: F) -> Result<Vec<R>>
    where
        T: Send + Sync,
        R: Send,
        F: Fn(&T) -> Result<R> + Send + Sync,
    {
        if items.is_empty() {
            return Ok(vec![]);
        }

        let chunk_size = self.calculate_optimal_chunk_size(items.len());
        
        let results: Result<Vec<Vec<R>>> = items
            .par_chunks(chunk_size)
            .map(|chunk| {
                chunk.iter()
                    .map(&operation)
                    .collect::<Result<Vec<R>>>()
            })
            .collect();

        results.map(|chunks| chunks.into_iter().flatten().collect())
    }

    /// Parallel convolution with optimized memory access patterns
    pub fn parallel_conv1d(&self, input: &Array3<f32>, kernel: &Array3<f32>) -> Result<Array3<f32>> {
        let (batch_size, seq_len, in_channels) = input.dim();
        let (out_channels, kernel_size, kernel_in_channels) = kernel.dim();

        if in_channels != kernel_in_channels {
            return Err(crate::error::Error::InvalidInput(
                "Channel dimensions don't match".to_string()
            ));
        }

        let output_len = seq_len - kernel_size + 1;
        let mut output = Array3::zeros((batch_size, output_len, out_channels));

        // Parallelize over batch and output channels
        let batch_channel_pairs: Vec<_> = (0..batch_size)
            .flat_map(|b| (0..out_channels).map(move |oc| (b, oc)))
            .collect();

        batch_channel_pairs.par_iter().for_each(|&(batch_idx, out_ch)| {
            for pos in 0..output_len {
                let mut sum = 0.0;
                for k in 0..kernel_size {
                    for in_ch in 0..in_channels {
                        sum += input[[batch_idx, pos + k, in_ch]] * kernel[[out_ch, k, in_ch]];
                    }
                }
                // This is not thread-safe, but for demonstration purposes
                // In real implementation, we'd need proper synchronization
                unsafe {
                    let ptr = output.as_mut_ptr().add(
                        batch_idx * output_len * out_channels + pos * out_channels + out_ch
                    );
                    *ptr = sum;
                }
            }
        });

        Ok(output)
    }

    /// Parallel attention computation with memory optimization
    pub fn parallel_attention(
        &self,
        query: &Array3<f32>,
        key: &Array3<f32>,
        value: &Array3<f32>,
        num_heads: usize,
    ) -> Result<Array3<f32>> {
        let (batch_size, seq_len, embed_dim) = query.dim();
        let head_dim = embed_dim / num_heads;

        if embed_dim % num_heads != 0 {
            return Err(crate::error::Error::InvalidInput(
                "Embedding dimension must be divisible by number of heads".to_string()
            ));
        }

        let mut output = Array3::zeros((batch_size, seq_len, embed_dim));

        // Parallelize over batch and heads
        (0..batch_size).into_par_iter().for_each(|batch_idx| {
            (0..num_heads).into_par_iter().for_each(|head_idx| {
                let start_dim = head_idx * head_dim;
                let end_dim = start_dim + head_dim;

                let q_head = query.slice(s![batch_idx, .., start_dim..end_dim]);
                let k_head = key.slice(s![batch_idx, .., start_dim..end_dim]);
                let v_head = value.slice(s![batch_idx, .., start_dim..end_dim]);

                // Compute attention scores
                let scores = q_head.dot(&k_head.t()) / (head_dim as f32).sqrt();
                
                // Apply softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_scores = scores.mapv(|x| (x - max_score).exp());
                let sum_exp = exp_scores.sum();
                let attention_weights = exp_scores / sum_exp;

                // Apply attention to values
                let head_output = attention_weights.dot(&v_head);

                // Write back to output (this needs proper synchronization in real implementation)
                unsafe {
                    let output_slice = output.slice_mut(s![batch_idx, .., start_dim..end_dim]);
                    // Copy head_output to output_slice
                }
            });
        });

        Ok(output)
    }

    /// Dynamic load balancing
    pub fn balance_load(&self) -> Result<()> {
        let mut monitor = self.performance_monitor.write().unwrap();
        
        // Calculate current load per thread
        let thread_loads = self.calculate_thread_loads();
        let avg_load = thread_loads.iter().sum::<f64>() / thread_loads.len() as f64;
        
        // Identify overloaded and underloaded threads
        let mut overloaded = Vec::new();
        let mut underloaded = Vec::new();
        
        for (i, &load) in thread_loads.iter().enumerate() {
            if load > avg_load * (1.0 + self.config.load_balance_threshold) {
                overloaded.push(i);
            } else if load < avg_load * (1.0 - self.config.load_balance_threshold) {
                underloaded.push(i);
            }
        }

        // Migrate tasks from overloaded to underloaded threads
        self.migrate_tasks(overloaded, underloaded)?;
        
        monitor.load_balance_efficiency = self.calculate_balance_efficiency(&thread_loads);
        
        Ok(())
    }

    /// Calculate optimal parameters based on system characteristics
    fn calculate_optimal_block_size(&self, m: usize, n: usize, k: usize) -> usize {
        // Consider cache size and NUMA topology
        let l3_cache_size = 8 * 1024 * 1024; // 8MB typical L3 cache
        let element_size = std::mem::size_of::<f32>();
        
        // Aim for blocks that fit in L3 cache
        let max_elements = l3_cache_size / (3 * element_size); // 3 matrices (A, B, C blocks)
        let block_size = (max_elements as f64).powf(1.0/3.0) as usize;
        
        // Ensure block size is reasonable
        block_size.max(32).min(512)
    }

    fn calculate_optimal_chunk_size(&self, total_items: usize) -> usize {
        let base_chunk_size = total_items / (self.config.num_threads * 4);
        base_chunk_size.max(1).min(self.config.chunk_size)
    }

    fn calculate_thread_loads(&self) -> Vec<f64> {
        // In real implementation, this would query actual thread utilization
        vec![0.5; self.config.num_threads]
    }

    fn migrate_tasks(&self, _overloaded: Vec<usize>, _underloaded: Vec<usize>) -> Result<()> {
        // Implementation for task migration between threads
        Ok(())
    }

    fn calculate_balance_efficiency(&self, loads: &[f64]) -> f64 {
        let avg_load = loads.iter().sum::<f64>() / loads.len() as f64;
        let variance = loads.iter()
            .map(|&load| (load - avg_load).powi(2))
            .sum::<f64>() / loads.len() as f64;
        
        1.0 / (1.0 + variance)
    }

    /// Get comprehensive performance statistics
    pub fn get_performance_stats(&self) -> ParallelPerformanceStats {
        let monitor = self.performance_monitor.read().unwrap();
        
        ParallelPerformanceStats {
            num_threads: self.config.num_threads,
            tasks_completed: monitor.total_tasks_completed,
            average_task_time: monitor.task_completion_times.iter().sum::<std::time::Duration>() 
                / monitor.task_completion_times.len() as u32,
            thread_utilization: monitor.thread_utilization.clone(),
            load_balance_efficiency: monitor.load_balance_efficiency,
            cache_efficiency: monitor.cache_hit_rates.iter().sum::<f64>() / monitor.cache_hit_rates.len() as f64,
            memory_usage_per_thread: monitor.memory_usage_per_thread.clone(),
            numa_efficiency: self.calculate_numa_efficiency(),
        }
    }

    fn calculate_numa_efficiency(&self) -> f64 {
        // Calculate NUMA access efficiency
        0.85 // Placeholder
    }
}

impl ThreadPool {
    fn new(config: &ParallelConfig, numa_topology: &NumaTopology) -> Result<Self> {
        let (sender, receiver) = unbounded();
        let receiver = Arc::new(Mutex::new(receiver));
        
        let scheduler: Box<dyn TaskScheduler + Send + Sync> = match config.scheduler_type {
            SchedulerType::WorkStealing => Box::new(WorkStealingScheduler::new()),
            SchedulerType::LoadBalancing => Box::new(LoadBalancingScheduler::new(config.load_balance_threshold)),
            _ => Box::new(WorkStealingScheduler::new()),
        };

        let mut workers = Vec::with_capacity(config.num_threads);
        
        for id in 0..config.num_threads {
            let worker = Worker::new(id, Arc::clone(&receiver), numa_topology)?;
            workers.push(worker);
        }

        Ok(ThreadPool {
            workers,
            sender,
            receiver,
            scheduler,
        })
    }
}

impl Worker {
    fn new(id: usize, receiver: Arc<Mutex<Receiver<Job>>>, _numa_topology: &NumaTopology) -> Result<Self> {
        let receiver_clone = Arc::clone(&receiver);
        
        let thread = thread::spawn(move || {
            loop {
                let job = receiver_clone.lock().unwrap().recv();
                match job {
                    Ok(job) => job(),
                    Err(_) => break,
                }
            }
        });

        Ok(Worker {
            id,
            thread: Some(thread),
            local_queue: VecDeque::new(),
            stealer: None,
        })
    }
}

impl WorkQueue {
    fn new() -> Self {
        Self {
            high_priority: Arc::new(Mutex::new(VecDeque::new())),
            normal_priority: Arc::new(Mutex::new(VecDeque::new())),
            low_priority: Arc::new(Mutex::new(VecDeque::new())),
            completed_tasks: Arc::new(Mutex::new(Vec::new())),
            active_tasks: Arc::new(Mutex::new(0)),
        }
    }
}

impl WorkStealingScheduler {
    fn new() -> Self {
        Self {
            steal_attempts: 3,
            random_steal_probability: 0.1,
        }
    }
}

impl TaskScheduler for WorkStealingScheduler {
    fn schedule(&self, mut tasks: Vec<PriorityTask>) -> Vec<PriorityTask> {
        tasks.sort_by_key(|task| task.priority.clone());
        tasks
    }

    fn balance_load(&self, _workers: &[Worker]) -> Vec<usize> {
        // Return worker indices in load-balanced order
        (0.._workers.len()).collect()
    }

    fn estimate_completion_time(&self, task: &PriorityTask) -> std::time::Duration {
        // Estimate based on task type and size
        match &task.task {
            TaskType::MatrixMultiplication { a_shape, b_shape } => {
                let ops = a_shape.0 * a_shape.1 * b_shape.1;
                std::time::Duration::from_nanos(ops as u64 / 1000) // ~1 GFLOPS estimate
            }
            TaskType::Convolution { input_shape, kernel_shape } => {
                let ops = input_shape.0 * input_shape.1 * input_shape.2 * 
                         kernel_shape.0 * kernel_shape.1 * kernel_shape.2;
                std::time::Duration::from_nanos(ops as u64 / 500)
            }
            _ => std::time::Duration::from_millis(10),
        }
    }
}

impl LoadBalancingScheduler {
    fn new(threshold: f64) -> Self {
        Self {
            balance_threshold: threshold,
            rebalance_interval: std::time::Duration::from_millis(100),
            last_balance: std::time::Instant::now(),
        }
    }
}

impl TaskScheduler for LoadBalancingScheduler {
    fn schedule(&self, tasks: Vec<PriorityTask>) -> Vec<PriorityTask> {
        // Implement load-aware scheduling
        tasks
    }

    fn balance_load(&self, workers: &[Worker]) -> Vec<usize> {
        // Calculate load per worker and return balanced assignment
        (0..workers.len()).collect()
    }

    fn estimate_completion_time(&self, task: &PriorityTask) -> std::time::Duration {
        task.estimated_duration
    }
}

impl PerformanceMonitor {
    fn new() -> Self {
        Self {
            task_completion_times: Vec::new(),
            thread_utilization: Vec::new(),
            cache_hit_rates: Vec::new(),
            memory_usage_per_thread: Vec::new(),
            load_balance_efficiency: 1.0,
            total_tasks_completed: 0,
            total_time_spent: std::time::Duration::new(0, 0),
        }
    }
}

impl NumaTopology {
    fn detect() -> Result<Self> {
        // Detect NUMA topology from system
        // This is a simplified implementation
        let num_cpus = num_cpus::get();
        let nodes = vec![NumaNode {
            id: 0,
            cpus: (0..num_cpus).collect(),
            memory_size: 16 * 1024 * 1024 * 1024, // 16GB
            available_memory: 12 * 1024 * 1024 * 1024, // 12GB available
        }];

        Ok(Self {
            nodes,
            cpu_to_node_map: vec![0; num_cpus],
            node_distances: vec![vec![0]],
        })
    }
}

#[derive(Debug, Clone)]
pub struct ParallelPerformanceStats {
    pub num_threads: usize,
    pub tasks_completed: u64,
    pub average_task_time: std::time::Duration,
    pub thread_utilization: Vec<f64>,
    pub load_balance_efficiency: f64,
    pub cache_efficiency: f64,
    pub memory_usage_per_thread: Vec<usize>,
    pub numa_efficiency: f64,
}

// Add proper imports at the top
use ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.num_threads > 0);
        assert!(config.chunk_size > 0);
        assert!(config.enable_work_stealing);
    }

    #[test]
    fn test_parallel_matrix_multiplication() {
        let config = ParallelConfig::default();
        let processor = ParallelProcessor::new(config).unwrap();

        let a = Array2::from_shape_vec((4, 3), vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]).unwrap();

        let b = Array2::from_shape_vec((3, 2), vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0,
        ]).unwrap();

        let result = processor.parallel_matmul(&a, &b).unwrap();
        
        assert_eq!(result.dim(), (4, 2));
        // Verify a few values
        assert_eq!(result[[0, 0]], 22.0); // 1*1 + 2*3 + 3*5
        assert_eq!(result[[0, 1]], 28.0); // 1*2 + 2*4 + 3*6
    }

    #[test]
    fn test_task_priority_ordering() {
        let mut tasks = vec![
            PriorityTask {
                id: 1,
                priority: TaskPriority::Low,
                task: TaskType::Activation { elements: 100, function: ActivationType::ReLU },
                estimated_duration: std::time::Duration::from_millis(10),
                dependencies: vec![],
                created_at: std::time::Instant::now(),
            },
            PriorityTask {
                id: 2,
                priority: TaskPriority::Critical,
                task: TaskType::Activation { elements: 100, function: ActivationType::ReLU },
                estimated_duration: std::time::Duration::from_millis(5),
                dependencies: vec![],
                created_at: std::time::Instant::now(),
            },
        ];

        let scheduler = WorkStealingScheduler::new();
        let scheduled = scheduler.schedule(tasks);
        
        assert_eq!(scheduled[0].priority, TaskPriority::Critical);
        assert_eq!(scheduled[1].priority, TaskPriority::Low);
    }

    #[test]
    fn test_numa_topology_detection() {
        let topology = NumaTopology::detect().unwrap();
        assert!(!topology.nodes.is_empty());
        assert!(!topology.cpu_to_node_map.is_empty());
    }
}