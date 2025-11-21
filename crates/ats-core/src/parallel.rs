//! High-Performance Parallel Processing for ATS-Core

use crate::{
    config::AtsCpConfig,
    error::{AtsCoreError, Result},
};
use rayon::prelude::*;

/// Parallel processing engine for multi-core optimization
pub struct ParallelProcessor {
    /// Configuration
    #[allow(dead_code)]
    config: AtsCpConfig,
    
    /// Thread pool
    thread_pool: Option<rayon::ThreadPool>,
}

impl ParallelProcessor {
    /// Creates a new parallel processor
    pub fn new(config: &AtsCpConfig) -> Result<Self> {
        let thread_pool = if config.parallel.num_threads > 0 {
            Some(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(config.parallel.num_threads)
                    .build()
                    .map_err(|e| AtsCoreError::parallel(format!("failed to create thread pool: {}", e)))?,
            )
        } else {
            None
        };
        
        Ok(Self {
            config: config.clone(),
            thread_pool,
        })
    }

    /// Processes data in parallel
    pub fn process<T, F>(&self, data: &[T], func: F) -> Result<Vec<T>>
    where
        T: Send + Sync + Clone,
        F: Fn(&T) -> T + Send + Sync,
    {
        let result = if let Some(pool) = &self.thread_pool {
            pool.install(|| data.par_iter().map(func).collect())
        } else {
            data.par_iter().map(func).collect()
        };
        
        Ok(result)
    }
}