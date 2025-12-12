//! Parallel Execution Framework
//!
//! High-performance parallel execution infrastructure for swarm algorithms
//! with work stealing, load balancing, and CPU core optimization.

pub mod executor;
pub mod scheduler;
pub mod work_stealing;

pub use executor::*;
pub use scheduler::*;
pub use work_stealing::*;

use anyhow::Result;
use std::sync::Arc;
use rayon::ThreadPool;

/// Parallel execution configuration
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    pub num_threads: Option<usize>,
    pub work_stealing: bool,
    pub chunk_size: Option<usize>,
    pub load_balancing: bool,
    pub cpu_affinity: bool,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_threads: None, // Use all available cores
            work_stealing: true,
            chunk_size: None, // Adaptive chunk sizing
            load_balancing: true,
            cpu_affinity: false,
        }
    }
}

/// Thread pool manager for swarm algorithms
pub struct SwarmThreadPool {
    pool: Arc<ThreadPool>,
    config: ParallelConfig,
}

impl SwarmThreadPool {
    pub fn new(config: ParallelConfig) -> Result<Self> {
        let mut builder = rayon::ThreadPoolBuilder::new();
        
        if let Some(threads) = config.num_threads {
            builder = builder.num_threads(threads);
        }
        
        let pool = Arc::new(
            builder.build()
                .map_err(|e| crate::SwarmError::ParallelError(e.to_string()))?
        );
        
        Ok(Self { pool, config })
    }
    
    pub fn execute<F, R>(&self, work: F) -> R
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        self.pool.install(work)
    }
    
    pub fn current_num_threads(&self) -> usize {
        self.pool.current_num_threads()
    }
}