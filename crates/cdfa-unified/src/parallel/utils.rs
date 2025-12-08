//! Parallel processing utilities

use crate::error::Result;

/// Get optimal number of threads
pub fn optimal_thread_count() -> usize {
    #[cfg(feature = "num_cpus")]
    {
        num_cpus::get()
    }
    #[cfg(not(feature = "num_cpus"))]
    {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
    }
}

/// Parallel backend for accelerated operations
#[derive(Debug)]
pub struct ParallelBackend {
    num_threads: usize,
}

impl ParallelBackend {
    /// Create new parallel backend
    pub fn new(num_threads: usize) -> Result<Self> {
        let threads = if num_threads == 0 {
            optimal_thread_count()
        } else {
            num_threads
        };
        
        Ok(Self {
            num_threads: threads,
        })
    }
    
    /// Get number of threads
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }
}