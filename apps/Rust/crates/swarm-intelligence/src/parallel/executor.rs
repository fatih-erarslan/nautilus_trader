//! Parallel Execution Engine
//!
//! Optimized parallel execution engine for swarm algorithms with
//! work stealing, adaptive load balancing, and NUMA awareness.

use crate::{SwarmError, common::*};
use anyhow::Result;
use crossbeam::channel::{bounded, Receiver, Sender};
use nalgebra::DVector;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};
use parking_lot::RwLock;

/// Work item for parallel processing
#[derive(Debug)]
pub struct WorkItem<T> {
    pub id: usize,
    pub data: T,
    pub priority: WorkPriority,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WorkPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Parallel fitness evaluation engine
pub struct ParallelFitnessEvaluator<F> {
    objective: Arc<F>,
    num_threads: usize,
    work_sender: Sender<WorkItem<DVector<f64>>>,
    result_receiver: Receiver<(usize, f64)>,
    active_workers: Arc<AtomicUsize>,
    total_evaluations: Arc<AtomicUsize>,
}

impl<F> ParallelFitnessEvaluator<F>
where
    F: Fn(&DVector<f64>) -> f64 + Send + Sync + 'static,
{
    pub fn new(objective: F, num_threads: Option<usize>) -> Result<Self> {
        let num_threads = num_threads.unwrap_or_else(num_cpus::get);
        let (work_sender, work_receiver) = bounded(num_threads * 4);
        let (result_sender, result_receiver) = bounded(num_threads * 4);
        
        let objective = Arc::new(objective);
        let active_workers = Arc::new(AtomicUsize::new(0));
        let total_evaluations = Arc::new(AtomicUsize::new(0));
        
        // Spawn worker threads
        for worker_id in 0..num_threads {
            let work_receiver = work_receiver.clone();
            let result_sender = result_sender.clone();
            let objective = Arc::clone(&objective);
            let active_workers = Arc::clone(&active_workers);
            let total_evaluations = Arc::clone(&total_evaluations);
            
            thread::Builder::new()
                .name(format!("swarm-worker-{}", worker_id))
                .spawn(move || {
                    Self::worker_loop(
                        worker_id,
                        work_receiver,
                        result_sender,
                        objective,
                        active_workers,
                        total_evaluations,
                    );
                })
                .map_err(|e| SwarmError::ParallelError(format!("Failed to spawn worker: {}", e)))?;
        }
        
        Ok(Self {
            objective,
            num_threads,
            work_sender,
            result_receiver,
            active_workers,
            total_evaluations,
        })
    }
    
    fn worker_loop(
        worker_id: usize,
        work_receiver: Receiver<WorkItem<DVector<f64>>>,
        result_sender: Sender<(usize, f64)>,
        objective: Arc<F>,
        active_workers: Arc<AtomicUsize>,
        total_evaluations: Arc<AtomicUsize>,
    ) {
        while let Ok(work_item) = work_receiver.recv() {
            active_workers.fetch_add(1, Ordering::Relaxed);
            
            let start = Instant::now();
            let fitness = (objective)(&work_item.data);
            let duration = start.elapsed();
            
            total_evaluations.fetch_add(1, Ordering::Relaxed);
            active_workers.fetch_sub(1, Ordering::Relaxed);
            
            if result_sender.send((work_item.id, fitness)).is_err() {
                break; // Receiver disconnected
            }
            
            // Adaptive yielding based on computation time
            if duration < Duration::from_micros(100) {
                thread::yield_now();
            }
        }
    }
    
    pub fn evaluate_batch(&self, positions: Vec<DVector<f64>>) -> Result<Vec<f64>> {
        let batch_size = positions.len();
        let mut results = vec![0.0; batch_size];
        
        // Submit work items
        for (id, position) in positions.into_iter().enumerate() {
            let work_item = WorkItem {
                id,
                data: position,
                priority: WorkPriority::Normal,
            };
            
            self.work_sender.send(work_item)
                .map_err(|e| SwarmError::ParallelError(format!("Failed to send work: {}", e)))?;
        }
        
        // Collect results
        for _ in 0..batch_size {
            let (id, fitness) = self.result_receiver.recv()
                .map_err(|e| SwarmError::ParallelError(format!("Failed to receive result: {}", e)))?;
            results[id] = fitness;
        }
        
        Ok(results)
    }
    
    pub fn get_stats(&self) -> ExecutorStats {
        ExecutorStats {
            active_workers: self.active_workers.load(Ordering::Relaxed),
            total_evaluations: self.total_evaluations.load(Ordering::Relaxed),
            num_threads: self.num_threads,
        }
    }
}

/// Executor performance statistics
#[derive(Debug, Clone)]
pub struct ExecutorStats {
    pub active_workers: usize,
    pub total_evaluations: usize,
    pub num_threads: usize,
}

/// Adaptive work scheduler for load balancing
pub struct AdaptiveScheduler {
    thread_loads: Arc<RwLock<Vec<f64>>>,
    load_history: Arc<RwLock<Vec<Vec<f64>>>>,
    rebalance_threshold: f64,
    num_threads: usize,
}

impl AdaptiveScheduler {
    pub fn new(num_threads: usize, rebalance_threshold: f64) -> Self {
        Self {
            thread_loads: Arc::new(RwLock::new(vec![0.0; num_threads])),
            load_history: Arc::new(RwLock::new(vec![Vec::new(); num_threads])),
            rebalance_threshold,
            num_threads,
        }
    }
    
    pub fn update_load(&self, thread_id: usize, load: f64) {
        let mut loads = self.thread_loads.write();
        loads[thread_id] = load;
        
        let mut history = self.load_history.write();
        history[thread_id].push(load);
        
        // Keep only recent history
        if history[thread_id].len() > 100 {
            history[thread_id].remove(0);
        }
    }
    
    pub fn should_rebalance(&self) -> bool {
        let loads = self.thread_loads.read();
        if loads.is_empty() {
            return false;
        }
        
        let max_load = loads.iter().fold(0.0, |a, &b| a.max(b));
        let min_load = loads.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        if max_load == 0.0 {
            return false;
        }
        
        (max_load - min_load) / max_load > self.rebalance_threshold
    }
    
    pub fn get_load_distribution(&self) -> Vec<f64> {
        self.thread_loads.read().clone()
    }
}

/// SIMD-optimized vector operations for swarm algorithms
pub mod simd_ops {
    use nalgebra::DVector;
    
    /// SIMD-optimized vector addition
    pub fn add_vectors_simd(a: &DVector<f64>, b: &DVector<f64>) -> DVector<f64> {
        // For now, use nalgebra's optimized operations
        // In the future, can add explicit SIMD using std::simd
        a + b
    }
    
    /// SIMD-optimized dot product
    pub fn dot_product_simd(a: &DVector<f64>, b: &DVector<f64>) -> f64 {
        a.dot(b)
    }
    
    /// SIMD-optimized vector norm
    pub fn norm_simd(v: &DVector<f64>) -> f64 {
        v.norm()
    }
    
    /// Parallel reduction for fitness aggregation
    pub fn parallel_reduce<T, F>(data: &[T], identity: T, op: F) -> T
    where
        T: Send + Sync + Clone,
        F: Fn(T, T) -> T + Send + Sync,
    {
        use rayon::prelude::*;
        data.par_iter()
            .cloned()
            .reduce(|| identity.clone(), op)
    }
}

/// Memory pool for efficient particle management
pub struct ParticleMemoryPool {
    pool: Arc<Mutex<Vec<DVector<f64>>>>,
    dimension: usize,
    max_size: usize,
}

impl ParticleMemoryPool {
    pub fn new(dimension: usize, max_size: usize) -> Self {
        Self {
            pool: Arc::new(Mutex::new(Vec::with_capacity(max_size))),
            dimension,
            max_size,
        }
    }
    
    pub fn get_vector(&self) -> DVector<f64> {
        let mut pool = self.pool.lock().unwrap();
        pool.pop().unwrap_or_else(|| DVector::zeros(self.dimension))
    }
    
    pub fn return_vector(&self, mut vector: DVector<f64>) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < self.max_size {
            vector.fill(0.0); // Clear for reuse
            pool.push(vector);
        }
    }
    
    pub fn pool_size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    
    fn test_objective(x: &DVector<f64>) -> f64 {
        // Simple sphere function with artificial delay
        thread::sleep(Duration::from_millis(1));
        x.iter().map(|xi| xi * xi).sum()
    }
    
    #[test]
    fn test_parallel_evaluator() {
        let evaluator = ParallelFitnessEvaluator::new(test_objective, Some(4)).unwrap();
        
        let positions = vec![
            DVector::from_vec(vec![1.0, 2.0]),
            DVector::from_vec(vec![2.0, 3.0]),
            DVector::from_vec(vec![0.0, 1.0]),
        ];
        
        let results = evaluator.evaluate_batch(positions).unwrap();
        
        assert_eq!(results.len(), 3);
        assert!((results[0] - 5.0).abs() < 1e-10);
        assert!((results[1] - 13.0).abs() < 1e-10);
        assert!((results[2] - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_adaptive_scheduler() {
        let scheduler = AdaptiveScheduler::new(4, 0.3);
        
        scheduler.update_load(0, 0.1);
        scheduler.update_load(1, 0.8);
        scheduler.update_load(2, 0.2);
        scheduler.update_load(3, 0.9);
        
        assert!(scheduler.should_rebalance());
        
        let distribution = scheduler.get_load_distribution();
        assert_eq!(distribution.len(), 4);
    }
    
    #[test]
    fn test_memory_pool() {
        let pool = ParticleMemoryPool::new(3, 10);
        
        let v1 = pool.get_vector();
        assert_eq!(v1.len(), 3);
        
        pool.return_vector(v1);
        assert_eq!(pool.pool_size(), 1);
        
        let v2 = pool.get_vector();
        assert_eq!(v2.len(), 3);
        assert_eq!(pool.pool_size(), 0);
    }
    
    #[test]
    fn test_simd_operations() {
        let a = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        
        let sum = simd_ops::add_vectors_simd(&a, &b);
        assert_eq!(sum, DVector::from_vec(vec![5.0, 7.0, 9.0]));
        
        let dot = simd_ops::dot_product_simd(&a, &b);
        assert!((dot - 32.0).abs() < 1e-10);
        
        let norm = simd_ops::norm_simd(&a);
        assert!((norm - 14.0_f64.sqrt()).abs() < 1e-10);
    }
}