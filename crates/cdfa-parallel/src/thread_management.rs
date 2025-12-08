//! Thread pool management with NUMA awareness and CPU affinity
//!
//! Provides custom thread pools optimized for CDFA workloads with
//! support for NUMA architectures and CPU pinning.

use core_affinity::{self, CoreId};
use libc::{self, cpu_set_t, CPU_SET, CPU_ZERO, sched_setaffinity};
use parking_lot::{Mutex, RwLock};
use rayon::{ThreadPool, ThreadPoolBuilder};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cdfa_core::error::Result;

/// NUMA-aware thread pool for CDFA workloads
pub struct NumaAwareThreadPool {
    /// Per-NUMA node thread pools
    node_pools: HashMap<usize, Arc<ThreadPool>>,
    
    /// CPU topology information
    topology: CpuTopology,
    
    /// Thread assignment strategy
    assignment_strategy: ThreadAssignmentStrategy,
    
    /// Pool statistics
    stats: Arc<RwLock<PoolStatistics>>,
    
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
}

/// CPU topology information
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Number of NUMA nodes
    pub numa_nodes: usize,
    
    /// Cores per NUMA node
    pub cores_per_node: Vec<Vec<CoreId>>,
    
    /// Total number of cores
    pub total_cores: usize,
    
    /// L3 cache sharing groups
    pub cache_groups: Vec<Vec<CoreId>>,
    
    /// SMT siblings (hyperthreading)
    pub smt_siblings: HashMap<CoreId, Vec<CoreId>>,
}

impl CpuTopology {
    /// Detects the system's CPU topology
    pub fn detect() -> Self {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        let total_cores = core_ids.len();
        
        // Simplified detection - in production, use hwloc or similar
        let numa_nodes = Self::detect_numa_nodes();
        let cores_per_node = Self::distribute_cores(&core_ids, numa_nodes);
        let cache_groups = Self::detect_cache_groups(&core_ids);
        let smt_siblings = Self::detect_smt_siblings(&core_ids);
        
        Self {
            numa_nodes,
            cores_per_node,
            total_cores,
            cache_groups,
            smt_siblings,
        }
    }
    
    fn detect_numa_nodes() -> usize {
        // Check /sys/devices/system/node/ in production
        // For now, assume single node for simplicity
        1
    }
    
    fn distribute_cores(cores: &[CoreId], numa_nodes: usize) -> Vec<Vec<CoreId>> {
        let mut nodes = vec![Vec::new(); numa_nodes];
        for (i, &core) in cores.iter().enumerate() {
            nodes[i % numa_nodes].push(core);
        }
        nodes
    }
    
    fn detect_cache_groups(cores: &[CoreId]) -> Vec<Vec<CoreId>> {
        // Group cores by L3 cache sharing
        // Simplified: assume groups of 4 cores share L3
        cores.chunks(4)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
    
    fn detect_smt_siblings(cores: &[CoreId]) -> HashMap<CoreId, Vec<CoreId>> {
        // In production, check /sys/devices/system/cpu/cpu*/topology/thread_siblings
        HashMap::new()
    }
}

/// Thread assignment strategy
#[derive(Debug, Clone, Copy)]
pub enum ThreadAssignmentStrategy {
    /// Spread threads across NUMA nodes
    Spread,
    
    /// Pack threads on same NUMA node
    Pack,
    
    /// Assign based on workload characteristics
    Adaptive,
    
    /// Manual assignment
    Manual,
}

/// Pool statistics
#[derive(Debug, Default)]
pub struct PoolStatistics {
    /// Tasks executed per thread
    pub tasks_per_thread: HashMap<usize, u64>,
    
    /// Average task latency per node
    pub avg_latency_per_node: HashMap<usize, Duration>,
    
    /// Cache misses per node (if available)
    pub cache_misses_per_node: HashMap<usize, u64>,
    
    /// Memory bandwidth usage per node
    pub bandwidth_per_node: HashMap<usize, f64>,
}

impl NumaAwareThreadPool {
    /// Creates a new NUMA-aware thread pool
    pub fn new(
        threads_per_node: usize,
        assignment_strategy: ThreadAssignmentStrategy,
    ) -> Result<Self> {
        let topology = CpuTopology::detect();
        let mut node_pools = HashMap::new();
        
        // Create thread pool for each NUMA node
        for (node_id, cores) in topology.cores_per_node.iter().enumerate() {
            let pool = Self::create_node_pool(
                node_id,
                cores,
                threads_per_node.min(cores.len()),
            )?;
            node_pools.insert(node_id, Arc::new(pool));
        }
        
        Ok(Self {
            node_pools,
            topology,
            assignment_strategy,
            stats: Arc::new(RwLock::new(PoolStatistics::default())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }
    
    /// Creates a thread pool for a specific NUMA node
    fn create_node_pool(
        node_id: usize,
        cores: &[CoreId],
        num_threads: usize,
    ) -> Result<ThreadPool> {
        let pool = ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .thread_name(move |i| format!("cdfa-numa{}-{}", node_id, i))
            .start_handler(move |i| {
                // Pin thread to specific core
                if i < cores.len() {
                    let _ = core_affinity::set_for_current(cores[i]);
                }
            })
            .build()
            .map_err(|e| cdfa_core::error::Error::Config(e.to_string()))?;
        
        Ok(pool)
    }
    
    /// Executes a task on the most appropriate NUMA node
    pub fn execute<F, R>(&self, task: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        let node_id = self.select_node();
        let pool = &self.node_pools[&node_id];
        
        let start = Instant::now();
        let result = pool.install(task);
        let latency = start.elapsed();
        
        // Update statistics
        self.update_stats(node_id, latency);
        
        result
    }
    
    /// Selects the best NUMA node for task execution
    fn select_node(&self) -> usize {
        match self.assignment_strategy {
            ThreadAssignmentStrategy::Spread => {
                // Round-robin across nodes
                static COUNTER: AtomicUsize = AtomicUsize::new(0);
                COUNTER.fetch_add(1, Ordering::Relaxed) % self.topology.numa_nodes
            }
            ThreadAssignmentStrategy::Pack => {
                // Use first node until saturated
                0
            }
            ThreadAssignmentStrategy::Adaptive => {
                // Choose node with lowest average latency
                let stats = self.stats.read();
                stats.avg_latency_per_node
                    .iter()
                    .min_by_key(|(_, &latency)| latency)
                    .map(|(&node, _)| node)
                    .unwrap_or(0)
            }
            ThreadAssignmentStrategy::Manual => 0,
        }
    }
    
    /// Updates pool statistics
    fn update_stats(&self, node_id: usize, latency: Duration) {
        let mut stats = self.stats.write();
        
        // Update task count
        *stats.tasks_per_thread.entry(node_id).or_insert(0) += 1;
        
        // Update average latency
        let count = stats.tasks_per_thread[&node_id];
        let current_avg = stats.avg_latency_per_node.entry(node_id)
            .or_insert(Duration::ZERO);
        
        // Exponential moving average
        let alpha = 0.1;
        *current_avg = Duration::from_secs_f64(
            (1.0 - alpha) * current_avg.as_secs_f64() + alpha * latency.as_secs_f64()
        );
    }
    
    /// Gets current statistics
    pub fn statistics(&self) -> PoolStatistics {
        self.stats.read().clone()
    }
    
    /// Shuts down all thread pools
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
    }
}

/// CPU affinity manager for precise thread control
pub struct CpuAffinityManager {
    /// Core assignments
    assignments: Arc<RwLock<HashMap<thread::ThreadId, CoreId>>>,
    
    /// Available cores
    available_cores: Arc<RwLock<VecDeque<CoreId>>>,
    
    /// Pinned threads
    pinned_threads: Arc<RwLock<HashMap<thread::ThreadId, JoinHandle<()>>>>,
}

impl CpuAffinityManager {
    /// Creates a new CPU affinity manager
    pub fn new() -> Self {
        let core_ids = core_affinity::get_core_ids().unwrap_or_default();
        
        Self {
            assignments: Arc::new(RwLock::new(HashMap::new())),
            available_cores: Arc::new(RwLock::new(core_ids.into_iter().collect())),
            pinned_threads: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Pins the current thread to a specific core
    pub fn pin_to_core(&self, core_id: CoreId) -> Result<()> {
        core_affinity::set_for_current(core_id)
            .then(|| {
                let thread_id = thread::current().id();
                self.assignments.write().insert(thread_id, core_id);
            })
            .ok_or_else(|| cdfa_core::error::Error::Config("Failed to set CPU affinity".into()))
    }
    
    /// Pins the current thread to the next available core
    pub fn pin_to_available(&self) -> Result<CoreId> {
        let mut available = self.available_cores.write();
        let core_id = available.pop_front()
            .ok_or_else(|| cdfa_core::error::Error::Config("No available cores".into()))?;
        
        self.pin_to_core(core_id)?;
        Ok(core_id)
    }
    
    /// Releases a core back to the available pool
    pub fn release_core(&self, core_id: CoreId) {
        self.available_cores.write().push_back(core_id);
    }
    
    /// Gets the current thread's core assignment
    pub fn current_core(&self) -> Option<CoreId> {
        let thread_id = thread::current().id();
        self.assignments.read().get(&thread_id).copied()
    }
    
    /// Sets CPU affinity using raw Linux syscall (more control)
    #[cfg(target_os = "linux")]
    pub fn set_affinity_mask(&self, cpu_mask: &[usize]) -> Result<()> {
        unsafe {
            let mut cpu_set: cpu_set_t = std::mem::zeroed();
            CPU_ZERO(&mut cpu_set);
            
            for &cpu in cpu_mask {
                CPU_SET(cpu, &mut cpu_set);
            }
            
            let tid = 0; // 0 means current thread
            if sched_setaffinity(tid, std::mem::size_of::<cpu_set_t>(), &cpu_set) != 0 {
                return Err(cdfa_core::error::Error::Config("Failed to set affinity mask".into()));
            }
        }
        
        Ok(())
    }
}

/// Custom thread pool optimized for CDFA workloads
pub struct CdfaThreadPool {
    /// Worker threads
    workers: Vec<Worker>,
    
    /// Task queue
    task_queue: Arc<Mutex<VecDeque<Task>>>,
    
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    
    /// Pool configuration
    config: ThreadPoolConfig,
    
    /// Performance metrics
    metrics: Arc<PoolMetrics>,
}

/// Worker thread
struct Worker {
    id: usize,
    thread: Option<JoinHandle<()>>,
}

/// Task to be executed
type Task = Box<dyn FnOnce() + Send + 'static>;

/// Thread pool configuration
#[derive(Clone)]
pub struct ThreadPoolConfig {
    /// Number of worker threads
    pub num_threads: usize,
    
    /// Task queue capacity
    pub queue_capacity: usize,
    
    /// Worker thread stack size
    pub stack_size: usize,
    
    /// Enable CPU pinning
    pub pin_threads: bool,
    
    /// Thread name prefix
    pub thread_prefix: String,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: num_cpus::get(),
            queue_capacity: 10_000,
            stack_size: 2 * 1024 * 1024, // 2MB
            pin_threads: true,
            thread_prefix: "cdfa-worker".to_string(),
        }
    }
}

/// Pool performance metrics
#[derive(Default)]
struct PoolMetrics {
    tasks_executed: AtomicU64,
    tasks_queued: AtomicU64,
    avg_queue_time_ns: AtomicU64,
    avg_execution_time_ns: AtomicU64,
}

impl CdfaThreadPool {
    /// Creates a new CDFA thread pool
    pub fn new(config: ThreadPoolConfig) -> Result<Self> {
        let task_queue = Arc::new(Mutex::new(VecDeque::with_capacity(config.queue_capacity)));
        let shutdown = Arc::new(AtomicBool::new(false));
        let metrics = Arc::new(PoolMetrics::default());
        let affinity_manager = Arc::new(CpuAffinityManager::new());
        
        let mut workers = Vec::with_capacity(config.num_threads);
        
        for id in 0..config.num_threads {
            let worker = Worker::new(
                id,
                Arc::clone(&task_queue),
                Arc::clone(&shutdown),
                Arc::clone(&metrics),
                Arc::clone(&affinity_manager),
                &config,
            )?;
            workers.push(worker);
        }
        
        Ok(Self {
            workers,
            task_queue,
            shutdown,
            config,
            metrics,
        })
    }
    
    /// Executes a task in the thread pool
    pub fn execute<F>(&self, task: F) -> Result<()>
    where
        F: FnOnce() + Send + 'static,
    {
        if self.shutdown.load(Ordering::Acquire) {
            return Err(cdfa_core::error::Error::Shutdown);
        }
        
        let queued_at = Instant::now();
        
        // Wrap task with metrics collection
        let metrics = Arc::clone(&self.metrics);
        let wrapped_task = Box::new(move || {
            let queue_time = queued_at.elapsed();
            metrics.avg_queue_time_ns.fetch_add(queue_time.as_nanos() as u64, Ordering::Relaxed);
            
            let exec_start = Instant::now();
            task();
            let exec_time = exec_start.elapsed();
            
            metrics.avg_execution_time_ns.fetch_add(exec_time.as_nanos() as u64, Ordering::Relaxed);
            metrics.tasks_executed.fetch_add(1, Ordering::Relaxed);
        });
        
        // Queue task
        {
            let mut queue = self.task_queue.lock();
            if queue.len() >= self.config.queue_capacity {
                return Err(cdfa_core::error::Error::QueueFull);
            }
            queue.push_back(wrapped_task);
        }
        
        self.metrics.tasks_queued.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
    
    /// Gets current metrics
    pub fn metrics(&self) -> PoolMetricsSnapshot {
        let tasks_executed = self.metrics.tasks_executed.load(Ordering::Relaxed);
        let tasks_queued = self.metrics.tasks_queued.load(Ordering::Relaxed);
        
        PoolMetricsSnapshot {
            tasks_executed,
            tasks_queued,
            avg_queue_time_ns: if tasks_executed > 0 {
                self.metrics.avg_queue_time_ns.load(Ordering::Relaxed) / tasks_executed
            } else {
                0
            },
            avg_execution_time_ns: if tasks_executed > 0 {
                self.metrics.avg_execution_time_ns.load(Ordering::Relaxed) / tasks_executed
            } else {
                0
            },
            queue_length: self.task_queue.lock().len(),
        }
    }
    
    /// Shuts down the thread pool
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Release);
        
        for worker in &mut self.workers {
            if let Some(thread) = worker.thread.take() {
                let _ = thread.join();
            }
        }
    }
}

/// Snapshot of pool metrics
#[derive(Debug, Clone)]
pub struct PoolMetricsSnapshot {
    pub tasks_executed: u64,
    pub tasks_queued: u64,
    pub avg_queue_time_ns: u64,
    pub avg_execution_time_ns: u64,
    pub queue_length: usize,
}

impl Worker {
    fn new(
        id: usize,
        task_queue: Arc<Mutex<VecDeque<Task>>>,
        shutdown: Arc<AtomicBool>,
        metrics: Arc<PoolMetrics>,
        affinity_manager: Arc<CpuAffinityManager>,
        config: &ThreadPoolConfig,
    ) -> Result<Self> {
        let thread_name = format!("{}-{}", config.thread_prefix, id);
        let stack_size = config.stack_size;
        let pin_threads = config.pin_threads;
        
        let thread = thread::Builder::new()
            .name(thread_name)
            .stack_size(stack_size)
            .spawn(move || {
                // Pin to CPU if requested
                if pin_threads {
                    let _ = affinity_manager.pin_to_available();
                }
                
                // Worker loop
                loop {
                    if shutdown.load(Ordering::Acquire) {
                        break;
                    }
                    
                    // Get next task
                    let task = {
                        let mut queue = task_queue.lock();
                        queue.pop_front()
                    };
                    
                    if let Some(task) = task {
                        task();
                    } else {
                        // No tasks, sleep briefly
                        thread::sleep(Duration::from_micros(100));
                    }
                }
                
                // Release CPU affinity
                if pin_threads {
                    if let Some(core_id) = affinity_manager.current_core() {
                        affinity_manager.release_core(core_id);
                    }
                }
            })
            .map_err(|e| cdfa_core::error::Error::ThreadSpawn(e.to_string()))?;
        
        Ok(Self {
            id,
            thread: Some(thread),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU32;
    
    #[test]
    fn test_cpu_topology_detection() {
        let topology = CpuTopology::detect();
        assert!(topology.total_cores > 0);
        assert!(topology.numa_nodes > 0);
        assert_eq!(topology.cores_per_node.len(), topology.numa_nodes);
    }
    
    #[test]
    fn test_numa_aware_thread_pool() {
        let pool = NumaAwareThreadPool::new(2, ThreadAssignmentStrategy::Spread).unwrap();
        
        let counter = Arc::new(AtomicU32::new(0));
        let counter_clone = Arc::clone(&counter);
        
        let result = pool.execute(move || {
            counter_clone.fetch_add(1, Ordering::Relaxed);
            42
        });
        
        assert_eq!(result, 42);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
    }
    
    #[test]
    fn test_cdfa_thread_pool() {
        let config = ThreadPoolConfig {
            num_threads: 4,
            pin_threads: false, // Disable for testing
            ..Default::default()
        };
        
        let pool = CdfaThreadPool::new(config).unwrap();
        let counter = Arc::new(AtomicU32::new(0));
        
        // Execute multiple tasks
        for _ in 0..10 {
            let counter_clone = Arc::clone(&counter);
            pool.execute(move || {
                counter_clone.fetch_add(1, Ordering::Relaxed);
            }).unwrap();
        }
        
        // Wait for completion
        thread::sleep(Duration::from_millis(100));
        
        assert_eq!(counter.load(Ordering::Relaxed), 10);
        
        let metrics = pool.metrics();
        assert_eq!(metrics.tasks_executed, 10);
    }
}