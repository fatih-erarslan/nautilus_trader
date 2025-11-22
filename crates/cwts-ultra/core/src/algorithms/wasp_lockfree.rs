//! WASP (Wait-free Atomic Swarm Processor) Lock-free Algorithm
//!
//! This module implements a high-performance lock-free swarm execution system
//! using wait-free atomic operations and advanced synchronization primitives.

use crossbeam::queue::{ArrayQueue, SegQueue};
use crossbeam::utils::CachePadded;
use std::alloc::{alloc, dealloc, Layout};
use std::mem;
use std::ptr;
use std::sync::atomic::{fence, AtomicBool, AtomicPtr, AtomicU64, Ordering};
use std::sync::Arc;
// Removed unused rayon::prelude import

/// Maximum number of concurrent worker threads
const MAX_WORKERS: usize = 64;
/// Maximum tasks per worker queue
const TASKS_PER_WORKER: usize = 1024;
/// Hazard pointer domain size
const HAZARD_DOMAIN_SIZE: usize = 128;
/// Work stealing attempts before backoff
const STEAL_ATTEMPTS: u32 = 8;
/// Nanosecond precision timestamps
const NS_PER_SECOND: u64 = 1_000_000_000_u64;
/// RCU grace period multiplier (2x epochs for safety)
const RCU_GRACE_PERIOD_MULTIPLIER: u64 = 2;
/// Epoch duration in nanoseconds (10ms)
const EPOCH_DURATION_NS: u64 = 10_000_000;

/// WASP algorithm errors
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WaspError {
    AllocationFailed,
    TaskQueueFull,
    InvalidWorker,
    PoolExhausted,
}

impl std::fmt::Display for WaspError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaspError::AllocationFailed => write!(f, "Memory allocation failed"),
            WaspError::TaskQueueFull => write!(f, "Task queue is full"),
            WaspError::InvalidWorker => write!(f, "Invalid worker ID"),
            WaspError::PoolExhausted => write!(f, "Task pool exhausted"),
        }
    }
}

impl std::error::Error for WaspError {}

/// Task priority levels
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum TaskPriority {
    Critical = 0, // Immediate execution
    High = 1,     // High priority
    Normal = 2,   // Normal priority
    Low = 3,      // Background tasks
}

/// Task execution status
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TaskStatus {
    Pending = 0,
    Running = 1,
    Completed = 2,
    Failed = 3,
    Cancelled = 4,
}

// Const assertions to verify enum layout
const _: () = assert!(std::mem::size_of::<TaskStatus>() == 1);

impl TaskStatus {
    /// Convert u8 to TaskStatus safely with validation
    #[inline]
    fn from_u8(value: u8) -> Self {
        match value {
            0 => TaskStatus::Pending,
            1 => TaskStatus::Running,
            2 => TaskStatus::Completed,
            3 => TaskStatus::Failed,
            4 => TaskStatus::Cancelled,
            _ => TaskStatus::Failed, // Invalid status defaults to Failed
        }
    }
}

/// Swarm task with wait-free execution
#[repr(C, align(64))] // Cache line aligned
pub struct SwarmTask {
    pub task_id: u64,
    pub priority: TaskPriority,
    pub status: AtomicU64,          // TaskStatus as u64 for atomic ops
    pub creation_time: u64,         // Nanosecond timestamp
    pub start_time: AtomicU64,      // Execution start time
    pub completion_time: AtomicU64, // Execution completion time
    pub worker_id: AtomicU64,       // Worker that executed this task
    pub execution_data: AtomicPtr<ExecutionData>, // Task-specific data
    pub next: AtomicPtr<SwarmTask>, // Next task in queue
    pub hazard_pointer: AtomicPtr<SwarmTask>, // For memory management
    pub retirement_epoch: AtomicU64, // Epoch when task was retired
}

/// Task execution data
#[repr(C, align(32))]
pub struct ExecutionData {
    pub symbol: String,
    pub operation_type: OperationType,
    pub parameters: Vec<f64>,
    pub result: AtomicPtr<ExecutionResult>,
    pub error_count: AtomicU64,
    pub retry_count: AtomicU64,
}

/// Operation types for tasks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OperationType {
    PriceAnalysis,
    VolumeDetection,
    OrderMatching,
    RiskCalculation,
    MarketDataUpdate,
    PositionManagement,
}

/// Execution result
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    pub success: bool,
    pub execution_time_ns: u64,
    pub output_data: Vec<f64>,
    pub error_message: Option<String>,
    pub performance_metrics: PerformanceMetrics,
}

/// Performance metrics for task execution
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub cpu_cycles: u64,
    pub memory_allocations: u64,
    pub cache_misses: u64,
    pub throughput: f64,
}

/// Worker thread state
#[repr(C, align(64))]
pub struct WorkerState {
    pub worker_id: u64,
    pub is_active: AtomicBool,
    pub tasks_executed: AtomicU64,
    pub total_execution_time: AtomicU64,
    pub last_activity: AtomicU64,
    pub current_task: AtomicPtr<SwarmTask>,
    pub local_queue: ArrayQueue<*mut SwarmTask>,
    pub steal_attempts: AtomicU64,
    pub successful_steals: AtomicU64,
    pub active_hazard_slot: AtomicU64, // Which hazard pointer slot this worker uses
}

/// Hazard pointer for safe memory reclamation
#[repr(C, align(32))]
pub struct HazardPointer {
    pub pointer: AtomicPtr<SwarmTask>,
    pub worker_id: AtomicU64,
    pub is_active: AtomicBool,
}

/// Lock-free swarm executor using WASP algorithm
pub struct LockFreeSwarmExecutor {
    // Global task queues by priority
    critical_queue: SegQueue<*mut SwarmTask>,
    high_priority_queue: SegQueue<*mut SwarmTask>,
    normal_priority_queue: SegQueue<*mut SwarmTask>,
    low_priority_queue: SegQueue<*mut SwarmTask>,

    // Worker management
    workers: Vec<CachePadded<WorkerState>>,
    #[allow(dead_code)]
    active_workers: AtomicU64,
    total_workers: usize,

    // Hazard pointer domain for memory management
    hazard_pointers: Vec<CachePadded<HazardPointer>>,
    retired_tasks: SegQueue<RetiredTask>,

    // Epoch-based reclamation
    global_epoch: AtomicU64,
    last_epoch_update: AtomicU64,

    // Global statistics
    execution_count: AtomicU64,
    total_execution_time: AtomicU64,
    #[allow(dead_code)]
    task_completion_rate: AtomicU64,
    average_latency_ns: AtomicU64,

    // Configuration
    max_concurrent_tasks: usize,
    #[allow(dead_code)]
    task_timeout_ns: u64,
    enable_work_stealing: AtomicBool,

    // Memory pool for task allocation
    task_pool: TaskPool,

    // Performance monitoring
    throughput_counter: AtomicU64,
    last_throughput_measurement: AtomicU64,
    peak_throughput: AtomicU64,
}

/// Retired task with epoch information
#[repr(C, align(32))]
struct RetiredTask {
    task_ptr: *mut SwarmTask,
    retirement_epoch: u64,
}

/// Memory pool for efficient task allocation
struct TaskPool {
    free_tasks: SegQueue<*mut SwarmTask>,
    allocated_tasks: AtomicU64,
    pool_size: usize,
}

impl TaskPool {
    fn new(pool_size: usize) -> Result<Self, WaspError> {
        let free_tasks = SegQueue::new();

        // Pre-allocate tasks with null pointer checks
        for _ in 0..pool_size {
            let layout = Layout::new::<SwarmTask>();
            unsafe {
                let task_ptr = alloc(layout) as *mut SwarmTask;

                // CRITICAL SECURITY FIX: Check for null pointer after allocation
                if task_ptr.is_null() {
                    // Clean up already allocated tasks before returning error
                    while let Some(ptr) = free_tasks.pop() {
                        ptr::drop_in_place(ptr);
                        dealloc(ptr as *mut u8, layout);
                    }
                    return Err(WaspError::AllocationFailed);
                }

                ptr::write(task_ptr, SwarmTask::new(0, TaskPriority::Normal));
                free_tasks.push(task_ptr);
            }
        }

        Ok(Self {
            free_tasks,
            allocated_tasks: AtomicU64::new(0),
            pool_size,
        })
    }

    /// Allocate task from pool or create new one
    fn allocate(&self) -> Result<*mut SwarmTask, WaspError> {
        if let Some(task_ptr) = self.free_tasks.pop() {
            self.allocated_tasks.fetch_add(1, Ordering::Relaxed);
            Ok(task_ptr)
        } else {
            // Pool exhausted, allocate new task
            let layout = Layout::new::<SwarmTask>();
            unsafe {
                let task_ptr = alloc(layout) as *mut SwarmTask;

                // CRITICAL SECURITY FIX: Check for null pointer after allocation
                if task_ptr.is_null() {
                    return Err(WaspError::AllocationFailed);
                }

                ptr::write(task_ptr, SwarmTask::new(0, TaskPriority::Normal));
                self.allocated_tasks.fetch_add(1, Ordering::Relaxed);
                Ok(task_ptr)
            }
        }
    }

    /// Return task to pool
    fn deallocate(&self, task_ptr: *mut SwarmTask) {
        unsafe {
            // Reset task state
            (*task_ptr).reset();
        }

        if self.free_tasks.len() < self.pool_size {
            self.free_tasks.push(task_ptr);
        } else {
            // Pool full, actually deallocate
            let layout = Layout::new::<SwarmTask>();
            unsafe {
                ptr::drop_in_place(task_ptr);
                dealloc(task_ptr as *mut u8, layout);
            }
        }

        self.allocated_tasks.fetch_sub(1, Ordering::Relaxed);
    }
}

impl SwarmTask {
    /// Create new swarm task
    pub fn new(task_id: u64, priority: TaskPriority) -> Self {
        Self {
            task_id,
            priority,
            status: AtomicU64::new(TaskStatus::Pending as u64),
            creation_time: Self::get_timestamp_ns(),
            start_time: AtomicU64::new(0),
            completion_time: AtomicU64::new(0),
            worker_id: AtomicU64::new(u64::MAX),
            execution_data: AtomicPtr::new(ptr::null_mut()),
            next: AtomicPtr::new(ptr::null_mut()),
            hazard_pointer: AtomicPtr::new(ptr::null_mut()),
            retirement_epoch: AtomicU64::new(0),
        }
    }

    /// Reset task for reuse
    pub fn reset(&mut self) {
        self.task_id = 0;
        self.priority = TaskPriority::Normal;
        self.status
            .store(TaskStatus::Pending as u64, Ordering::Release);
        self.creation_time = Self::get_timestamp_ns();
        self.start_time.store(0, Ordering::Release);
        self.completion_time.store(0, Ordering::Release);
        self.worker_id.store(u64::MAX, Ordering::Release);
        self.execution_data
            .store(ptr::null_mut(), Ordering::Release);
        self.next.store(ptr::null_mut(), Ordering::Release);
        self.hazard_pointer
            .store(ptr::null_mut(), Ordering::Release);
        self.retirement_epoch.store(0, Ordering::Release);
    }

    /// Get current task status
    pub fn get_status(&self) -> TaskStatus {
        let status_val = self.status.load(Ordering::Acquire) as u8;
        TaskStatus::from_u8(status_val)
    }

    /// Set task status atomically
    pub fn set_status(&self, status: TaskStatus) -> TaskStatus {
        let old_status = self.status.swap(status as u64, Ordering::AcqRel) as u8;
        TaskStatus::from_u8(old_status)
    }

    /// Compare and swap task status
    pub fn compare_and_set_status(
        &self,
        expected: TaskStatus,
        new: TaskStatus,
    ) -> Result<TaskStatus, TaskStatus> {
        match self.status.compare_exchange(
            expected as u64,
            new as u64,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(old) => Ok(TaskStatus::from_u8(old as u8)),
            Err(actual) => Err(TaskStatus::from_u8(actual as u8)),
        }
    }

    /// Mark task as started
    pub fn mark_started(&self, worker_id: u64) -> bool {
        if self
            .compare_and_set_status(TaskStatus::Pending, TaskStatus::Running)
            .is_ok()
        {
            self.start_time
                .store(Self::get_timestamp_ns(), Ordering::Release);
            self.worker_id.store(worker_id, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Mark task as completed
    pub fn mark_completed(&self) -> bool {
        if self
            .compare_and_set_status(TaskStatus::Running, TaskStatus::Completed)
            .is_ok()
        {
            self.completion_time
                .store(Self::get_timestamp_ns(), Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Get task execution duration
    pub fn get_execution_duration(&self) -> Option<u64> {
        let start = self.start_time.load(Ordering::Acquire);
        let end = self.completion_time.load(Ordering::Acquire);

        if start > 0 && end > start {
            Some(end - start)
        } else {
            None
        }
    }

    fn get_timestamp_ns() -> u64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    }
}

impl WorkerState {
    pub fn new(worker_id: u64) -> Self {
        Self {
            worker_id,
            is_active: AtomicBool::new(false),
            tasks_executed: AtomicU64::new(0),
            total_execution_time: AtomicU64::new(0),
            last_activity: AtomicU64::new(SwarmTask::get_timestamp_ns()),
            current_task: AtomicPtr::new(ptr::null_mut()),
            local_queue: ArrayQueue::new(TASKS_PER_WORKER),
            steal_attempts: AtomicU64::new(0),
            successful_steals: AtomicU64::new(0),
            active_hazard_slot: AtomicU64::new(u64::MAX),
        }
    }

    /// Get worker utilization (0.0 to 1.0)
    pub fn get_utilization(&self) -> f64 {
        let total_time = self.total_execution_time.load(Ordering::Acquire);
        let current_time = SwarmTask::get_timestamp_ns();
        let uptime = current_time - self.last_activity.load(Ordering::Acquire);

        if uptime > 0_u64 {
            (total_time as f64) / (uptime as f64)
        } else {
            0.0_f64
        }
    }
}

impl LockFreeSwarmExecutor {
    /// Create new lock-free swarm executor
    pub fn new(worker_count: usize, max_tasks: usize) -> Self {
        let actual_workers = worker_count.min(MAX_WORKERS);
        let mut workers = Vec::with_capacity(actual_workers);
        let mut hazard_pointers = Vec::with_capacity(HAZARD_DOMAIN_SIZE);

        // Initialize workers
        for i in 0..actual_workers {
            workers.push(CachePadded::new(WorkerState::new(i as u64)));
        }

        // Initialize hazard pointers
        for i in 0..HAZARD_DOMAIN_SIZE {
            hazard_pointers.push(CachePadded::new(HazardPointer {
                pointer: AtomicPtr::new(ptr::null_mut()),
                worker_id: AtomicU64::new(i as u64),
                is_active: AtomicBool::new(false),
            }));
        }

        Self {
            critical_queue: SegQueue::new(),
            high_priority_queue: SegQueue::new(),
            normal_priority_queue: SegQueue::new(),
            low_priority_queue: SegQueue::new(),

            workers,
            active_workers: AtomicU64::new(0),
            total_workers: actual_workers,

            hazard_pointers,
            retired_tasks: SegQueue::new(),

            global_epoch: AtomicU64::new(0),
            last_epoch_update: AtomicU64::new(SwarmTask::get_timestamp_ns()),

            execution_count: AtomicU64::new(0),
            total_execution_time: AtomicU64::new(0),
            task_completion_rate: AtomicU64::new(0),
            average_latency_ns: AtomicU64::new(0),

            max_concurrent_tasks: max_tasks,
            task_timeout_ns: 60_u64 * NS_PER_SECOND, // 60 second timeout
            enable_work_stealing: AtomicBool::new(true),

            task_pool: TaskPool::new(max_tasks * 2_usize)
                .expect("Failed to initialize task pool"),

            throughput_counter: AtomicU64::new(0),
            last_throughput_measurement: AtomicU64::new(SwarmTask::get_timestamp_ns()),
            peak_throughput: AtomicU64::new(0),
        }
    }

    /// Submit task for execution
    pub fn submit_task(
        &self,
        task_id: u64,
        priority: TaskPriority,
        operation: OperationType,
        parameters: Vec<f64>,
    ) -> Result<u64, &'static str> {
        // Check if we can accept more tasks
        let current_tasks = self.execution_count.load(Ordering::Acquire);
        if current_tasks >= self.max_concurrent_tasks as u64 {
            return Err("Task queue full");
        }

        // Allocate task from pool with null pointer check
        let task_ptr = self.task_pool.allocate()
            .map_err(|_| "Failed to allocate task from pool")?;

        unsafe {
            // Initialize task
            (*task_ptr).task_id = task_id;
            (*task_ptr).priority = priority;
            (*task_ptr)
                .status
                .store(TaskStatus::Pending as u64, Ordering::Release);
            (*task_ptr).creation_time = SwarmTask::get_timestamp_ns();

            // Create execution data
            let exec_data = Box::into_raw(Box::new(ExecutionData {
                symbol: "".to_string(), // Will be set by caller
                operation_type: operation,
                parameters,
                result: AtomicPtr::new(ptr::null_mut()),
                error_count: AtomicU64::new(0),
                retry_count: AtomicU64::new(0),
            }));

            (*task_ptr)
                .execution_data
                .store(exec_data, Ordering::Release);
        }

        // Submit to appropriate queue based on priority
        match priority {
            TaskPriority::Critical => self.critical_queue.push(task_ptr),
            TaskPriority::High => self.high_priority_queue.push(task_ptr),
            TaskPriority::Normal => self.normal_priority_queue.push(task_ptr),
            TaskPriority::Low => self.low_priority_queue.push(task_ptr),
        }

        self.execution_count.fetch_add(1, Ordering::AcqRel);
        Ok(task_id)
    }

    /// Execute next available task (called by worker threads)
    pub fn execute_next_task(&self, worker_id: u64) -> Option<ExecutionResult> {
        if worker_id >= self.total_workers as u64 {
            return None;
        }

        let worker = &self.workers[worker_id as usize];
        worker.is_active.store(true, Ordering::Release);
        worker
            .last_activity
            .store(SwarmTask::get_timestamp_ns(), Ordering::Release);

        // Try to get task from local queue first
        if let Some(task_ptr) = worker.local_queue.pop() {
            return self.execute_task(worker_id, task_ptr);
        }

        // Try global queues in priority order
        let task_ptr = self.pop_from_global_queues().or_else(|| {
            if self.enable_work_stealing.load(Ordering::Acquire) {
                self.steal_task(worker_id)
            } else {
                None
            }
        });

        if let Some(task_ptr) = task_ptr {
            self.execute_task(worker_id, task_ptr)
        } else {
            worker.is_active.store(false, Ordering::Release);
            None
        }
    }

    /// Pop task from global queues in priority order
    fn pop_from_global_queues(&self) -> Option<*mut SwarmTask> {
        self.critical_queue
            .pop()
            .or_else(|| self.high_priority_queue.pop())
            .or_else(|| self.normal_priority_queue.pop())
            .or_else(|| self.low_priority_queue.pop())
    }

    /// Steal task from another worker's local queue
    fn steal_task(&self, worker_id: u64) -> Option<*mut SwarmTask> {
        let worker = &self.workers[worker_id as usize];
        worker.steal_attempts.fetch_add(1, Ordering::Relaxed);

        // Try to steal from other workers' local queues
        for attempt in 0..STEAL_ATTEMPTS {
            let target_worker_id = (worker_id + 1 + attempt as u64) % self.total_workers as u64;
            let target_worker = &self.workers[target_worker_id as usize];

            if let Some(stolen_task) = target_worker.local_queue.pop() {
                worker.successful_steals.fetch_add(1, Ordering::Relaxed);
                return Some(stolen_task);
            }
        }

        None
    }

    /// Acquire hazard pointer for safe task access
    fn acquire_hazard_pointer(&self, worker_id: u64, task_ptr: *mut SwarmTask) -> Option<usize> {
        // Find or allocate a hazard pointer slot
        for (slot_idx, hazard) in self.hazard_pointers.iter().enumerate() {
            // Try to acquire this slot
            if !hazard.is_active.load(Ordering::Acquire) {
                if hazard
                    .is_active
                    .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    // Successfully acquired slot
                    hazard.worker_id.store(worker_id, Ordering::Release);
                    hazard.pointer.store(task_ptr, Ordering::Release);

                    // Critical: Full memory fence after setting hazard pointer
                    fence(Ordering::SeqCst);

                    // Verify task pointer is still valid (not retired yet)
                    // This prevents TOCTOU race
                    let task = unsafe { &*task_ptr };
                    let retirement_epoch = task.retirement_epoch.load(Ordering::Acquire);
                    if retirement_epoch > 0 {
                        // Task was already retired, release hazard pointer
                        self.release_hazard_pointer(slot_idx);
                        return None;
                    }

                    return Some(slot_idx);
                }
            }
        }

        None // All hazard pointer slots are in use
    }

    /// Release hazard pointer after task access complete
    fn release_hazard_pointer(&self, slot_idx: usize) {
        if slot_idx < self.hazard_pointers.len() {
            let hazard = &self.hazard_pointers[slot_idx];

            // Clear the pointer first
            hazard.pointer.store(ptr::null_mut(), Ordering::Release);

            // Full memory fence before releasing slot
            fence(Ordering::SeqCst);

            // Mark slot as inactive
            hazard.is_active.store(false, Ordering::Release);
        }
    }

    /// Execute a specific task
    fn execute_task(&self, worker_id: u64, task_ptr: *mut SwarmTask) -> Option<ExecutionResult> {
        // CRITICAL FIX: Acquire hazard pointer BEFORE accessing task
        let hazard_slot = self.acquire_hazard_pointer(worker_id, task_ptr)?;

        let worker = &self.workers[worker_id as usize];
        worker
            .active_hazard_slot
            .store(hazard_slot as u64, Ordering::Release);

        let task = unsafe { &*task_ptr };

        // Mark task as started
        if !task.mark_started(worker_id) {
            // Task was already started by another worker or cancelled
            self.release_hazard_pointer(hazard_slot);
            worker
                .active_hazard_slot
                .store(u64::MAX, Ordering::Release);
            return None;
        }

        worker.current_task.store(task_ptr, Ordering::Release);

        let execution_start = SwarmTask::get_timestamp_ns();

        // Execute the task based on operation type
        let result = unsafe {
            let exec_data = &*task.execution_data.load(Ordering::Acquire);
            self.perform_operation(exec_data.operation_type, &exec_data.parameters)
        };

        let execution_end = SwarmTask::get_timestamp_ns();
        let execution_time = execution_end - execution_start;

        // Mark task as completed
        task.mark_completed();

        // Update worker statistics
        worker.tasks_executed.fetch_add(1, Ordering::AcqRel);
        worker
            .total_execution_time
            .fetch_add(execution_time, Ordering::AcqRel);
        worker
            .current_task
            .store(ptr::null_mut(), Ordering::Release);

        // Update global statistics
        self.total_execution_time
            .fetch_add(execution_time, Ordering::AcqRel);
        self.throughput_counter.fetch_add(1, Ordering::AcqRel);

        // Update average latency
        let total_executions = self.execution_count.load(Ordering::Acquire);
        if total_executions > 0 {
            let total_time = self.total_execution_time.load(Ordering::Acquire);
            self.average_latency_ns
                .store(total_time / total_executions, Ordering::Release);
        }

        // Create execution result
        let exec_result = ExecutionResult {
            success: result.is_ok(),
            execution_time_ns: execution_time,
            output_data: result.unwrap_or_else(|_| Vec::new()),
            error_message: None, // Could be enhanced to capture error details
            performance_metrics: PerformanceMetrics {
                cpu_cycles: execution_time, // Simplified - would use actual CPU counters in production
                memory_allocations: 1,      // Simplified
                cache_misses: 0,            // Would need hardware counters
                throughput: self.calculate_current_throughput(),
            },
        };

        // CRITICAL FIX: Release hazard pointer BEFORE retiring task
        self.release_hazard_pointer(hazard_slot);
        worker
            .active_hazard_slot
            .store(u64::MAX, Ordering::Release);

        // Retire task for safe memory reclamation
        self.retire_task(task_ptr);

        Some(exec_result)
    }

    /// Perform the actual operation based on type
    fn perform_operation(
        &self,
        operation_type: OperationType,
        parameters: &[f64],
    ) -> Result<Vec<f64>, &'static str> {
        match operation_type {
            OperationType::PriceAnalysis => {
                // Simplified price analysis - would be more complex in reality
                if parameters.is_empty() {
                    return Err("No price data provided");
                }

                let avg_price = parameters.iter().sum::<f64>() / (parameters.len() as f64);
                let volatility = {
                    let variance = parameters
                        .iter()
                        .map(|p| (p - avg_price).powi(2))
                        .sum::<f64>()
                        / (parameters.len() as f64);
                    variance.sqrt()
                };

                Ok(vec![avg_price, volatility])
            }

            OperationType::VolumeDetection => {
                // Simplified volume analysis
                if parameters.len() < 2 {
                    return Err("Insufficient volume data");
                }

                let total_volume = parameters.iter().sum::<f64>();
                let avg_volume = total_volume / (parameters.len() as f64);
                let max_volume = parameters.iter().fold(0.0_f64, |a, &b| a.max(b));

                Ok(vec![total_volume, avg_volume, max_volume])
            }

            OperationType::OrderMatching => {
                // Simplified order matching logic
                if parameters.len() < 4 {
                    return Err("Insufficient order parameters");
                }

                let bid_price = parameters[0];
                let ask_price = parameters[1];
                let bid_qty = parameters[2];
                let ask_qty = parameters[3];

                if bid_price >= ask_price {
                    let match_price = (bid_price + ask_price) / 2.0_f64;
                    let match_qty = bid_qty.min(ask_qty);
                    Ok(vec![match_price, match_qty])
                } else {
                    Ok(vec![0.0_f64, 0.0_f64]) // No match
                }
            }

            OperationType::RiskCalculation => {
                // Simplified risk calculation
                if parameters.is_empty() {
                    return Err("No position data provided");
                }

                let position_value = parameters.iter().sum::<f64>();
                let risk_factor = 0.02_f64; // 2% risk
                let var = position_value.abs() * risk_factor;

                Ok(vec![position_value, var])
            }

            OperationType::MarketDataUpdate => {
                // Market data processing
                Ok(parameters.to_vec())
            }

            OperationType::PositionManagement => {
                // Position management logic
                if parameters.len() < 2 {
                    return Err("Insufficient position data");
                }

                let current_position = parameters[0];
                let target_position = parameters[1];
                let delta = target_position - current_position;

                Ok(vec![current_position, target_position, delta])
            }
        }
    }

    /// Update global epoch counter
    fn update_epoch(&self) {
        let current_time = SwarmTask::get_timestamp_ns();
        let last_update = self.last_epoch_update.load(Ordering::Acquire);

        if current_time - last_update >= EPOCH_DURATION_NS {
            self.global_epoch.fetch_add(1, Ordering::AcqRel);
            self.last_epoch_update.store(current_time, Ordering::Release);

            // Full memory fence after epoch update
            fence(Ordering::SeqCst);
        }
    }

    /// Retire task for safe memory reclamation using epoch-based hazard pointers
    fn retire_task(&self, task_ptr: *mut SwarmTask) {
        // Update epoch before retiring
        self.update_epoch();

        let task = unsafe { &*task_ptr };
        let current_epoch = self.global_epoch.load(Ordering::Acquire);

        // Mark task with retirement epoch
        task.retirement_epoch
            .store(current_epoch, Ordering::Release);

        // CRITICAL FIX: Full memory fence after marking retirement
        fence(Ordering::SeqCst);

        // Add to retired tasks list
        self.retired_tasks.push(RetiredTask {
            task_ptr,
            retirement_epoch: current_epoch,
        });

        // Periodically clean up retired tasks
        if self.retired_tasks.len() > 100_usize {
            self.reclaim_retired_tasks();
        }
    }

    /// Reclaim retired tasks that are no longer referenced (FIXED VERSION)
    fn reclaim_retired_tasks(&self) {
        let current_epoch = self.global_epoch.load(Ordering::Acquire);

        // CRITICAL FIX: Grace period = 2x RCU epochs for safety
        let grace_period_epochs = RCU_GRACE_PERIOD_MULTIPLIER;

        let mut tasks_to_check = Vec::new();
        let mut tasks_to_retry = Vec::new();

        // Collect retired tasks to check
        while let Some(retired) = self.retired_tasks.pop() {
            tasks_to_check.push(retired);

            // Limit batch size to prevent starvation
            if tasks_to_check.len() >= 100 {
                break;
            }
        }

        // Full memory fence before scanning hazard pointers
        fence(Ordering::SeqCst);

        // CRITICAL FIX: Scan ALL hazard pointers (don't break early!)
        let mut protected_pointers = Vec::new();
        for hazard in &self.hazard_pointers {
            if hazard.is_active.load(Ordering::Acquire) {
                let protected_ptr = hazard.pointer.load(Ordering::Acquire);
                if !protected_ptr.is_null() {
                    protected_pointers.push(protected_ptr);
                }
            }
        }

        // Check each retired task
        for retired in tasks_to_check {
            // CRITICAL FIX: Check grace period (2x RCU epochs)
            let epochs_since_retirement = current_epoch.saturating_sub(retired.retirement_epoch);

            if epochs_since_retirement < grace_period_epochs {
                // Grace period not elapsed, put it back
                tasks_to_retry.push(retired);
                continue;
            }

            // CRITICAL FIX: Check if task is protected by ANY hazard pointer
            let is_protected = protected_pointers
                .iter()
                .any(|&ptr| ptr == retired.task_ptr);

            if is_protected {
                // Still protected, put it back
                tasks_to_retry.push(retired);
            } else {
                // Safe to reclaim - no hazard pointers and grace period elapsed
                self.task_pool.deallocate(retired.task_ptr);
            }
        }

        // Put back tasks that couldn't be reclaimed
        for retired in tasks_to_retry {
            self.retired_tasks.push(retired);
        }
    }

    /// Check if task is protected by any hazard pointer (helper method)
    fn is_hazard_pointer_protected(&self, task_ptr: *mut SwarmTask) -> bool {
        // Full memory fence before checking hazard pointers
        fence(Ordering::Acquire);

        for hazard in &self.hazard_pointers {
            if hazard.is_active.load(Ordering::Acquire) {
                let protected_ptr = hazard.pointer.load(Ordering::Acquire);
                if protected_ptr == task_ptr {
                    return true;
                }
            }
        }
        false
    }

    /// Calculate current throughput (tasks per second)
    fn calculate_current_throughput(&self) -> f64 {
        let current_time = SwarmTask::get_timestamp_ns();
        let last_measurement = self.last_throughput_measurement.load(Ordering::Acquire);
        let time_diff = current_time - last_measurement;

        if time_diff >= NS_PER_SECOND {
            // Update every second
            let task_count = self.throughput_counter.swap(0_u64, Ordering::AcqRel);
            self.last_throughput_measurement
                .store(current_time, Ordering::Release);

            let throughput = (task_count as f64) / (time_diff as f64 / NS_PER_SECOND as f64);

            // Update peak throughput
            let current_peak = self.peak_throughput.load(Ordering::Acquire);
            if throughput as u64 > current_peak {
                self.peak_throughput
                    .store(throughput as u64, Ordering::Release);
            }

            throughput
        } else {
            0.0_f64 // Not enough time elapsed for meaningful measurement
        }
    }

    /// Get comprehensive executor statistics
    pub fn get_statistics(&self) -> SwarmExecutorStats {
        let total_tasks = self.execution_count.load(Ordering::Acquire);
        let total_time = self.total_execution_time.load(Ordering::Acquire);

        let mut worker_stats = Vec::new();
        let mut active_workers = 0_usize;

        for (i, worker) in self.workers.iter().enumerate() {
            if worker.is_active.load(Ordering::Acquire) {
                active_workers += 1_usize;
            }

            worker_stats.push(WorkerStats {
                worker_id: i as u64,
                is_active: worker.is_active.load(Ordering::Acquire),
                tasks_executed: worker.tasks_executed.load(Ordering::Acquire),
                execution_time: worker.total_execution_time.load(Ordering::Acquire),
                utilization: worker.get_utilization(),
                steal_attempts: worker.steal_attempts.load(Ordering::Acquire),
                successful_steals: worker.successful_steals.load(Ordering::Acquire),
                local_queue_size: worker.local_queue.len(),
            });
        }

        SwarmExecutorStats {
            total_tasks_executed: total_tasks,
            total_execution_time_ns: total_time,
            average_latency_ns: self.average_latency_ns.load(Ordering::Acquire),
            current_throughput: self.calculate_current_throughput(),
            peak_throughput: self.peak_throughput.load(Ordering::Acquire) as f64,
            active_workers,
            total_workers: self.total_workers,
            pending_tasks: self.get_pending_task_count(),
            retired_tasks: self.retired_tasks.len(),
            pool_utilization: self.get_pool_utilization(),
            worker_stats,
        }
    }

    /// Get total pending tasks across all queues
    fn get_pending_task_count(&self) -> usize {
        self.critical_queue.len()
            + self.high_priority_queue.len()
            + self.normal_priority_queue.len()
            + self.low_priority_queue.len()
    }

    /// Get task pool utilization
    fn get_pool_utilization(&self) -> f64 {
        let allocated = self.task_pool.allocated_tasks.load(Ordering::Acquire);
        (allocated as f64) / (self.task_pool.pool_size as f64)
    }

    /// Start worker threads
    pub fn start_workers(self: Arc<Self>) {
        for i in 0..self.total_workers {
            let executor = Arc::clone(&self);
            std::thread::spawn(move || {
                executor.worker_loop(i as u64);
            });
        }
    }

    /// Main worker thread loop
    fn worker_loop(&self, worker_id: u64) {
        loop {
            match self.execute_next_task(worker_id) {
                Some(_result) => {
                    // Task executed successfully
                }
                None => {
                    // No tasks available, brief backoff
                    std::thread::sleep(std::time::Duration::from_micros(100_u64));
                }
            }
        }
    }

    /// Execute all pending tasks (blocking until complete)
    pub fn execute(&self) {
        while self.get_pending_task_count() > 0_usize {
            // Try to execute tasks with available workers
            for worker_id in 0..self.total_workers as u64 {
                if !self.workers[worker_id as usize]
                    .is_active
                    .load(Ordering::Acquire)
                {
                    self.execute_next_task(worker_id);
                }
            }

            // Brief yield to prevent busy waiting
            std::thread::yield_now();
        }
    }
}

/// Statistics for individual worker
#[derive(Debug, Clone)]
pub struct WorkerStats {
    pub worker_id: u64,
    pub is_active: bool,
    pub tasks_executed: u64,
    pub execution_time: u64,
    pub utilization: f64,
    pub steal_attempts: u64,
    pub successful_steals: u64,
    pub local_queue_size: usize,
}

/// Comprehensive executor statistics
#[derive(Debug, Clone)]
pub struct SwarmExecutorStats {
    pub total_tasks_executed: u64,
    pub total_execution_time_ns: u64,
    pub average_latency_ns: u64,
    pub current_throughput: f64,
    pub peak_throughput: f64,
    pub active_workers: usize,
    pub total_workers: usize,
    pub pending_tasks: usize,
    pub retired_tasks: usize,
    pub pool_utilization: f64,
    pub worker_stats: Vec<WorkerStats>,
}

// Thread safety implementations
unsafe impl Send for LockFreeSwarmExecutor {}
unsafe impl Sync for LockFreeSwarmExecutor {}
unsafe impl Send for SwarmTask {}
unsafe impl Sync for SwarmTask {}
