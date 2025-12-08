//! Thread pool management

use crate::error::CdfaResult;

/// Thread pool configuration
#[derive(Debug, Clone)]
pub struct ThreadPoolConfig {
    pub num_threads: Option<usize>,
    pub stack_size: Option<usize>,
}

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            num_threads: None,
            stack_size: None,
        }
    }
}

/// Initialize thread pool with configuration
pub fn init_thread_pool(config: ThreadPoolConfig) -> CdfaResult<()> {
    #[cfg(feature = "rayon")]
    {
        let mut builder = rayon::ThreadPoolBuilder::new();
        
        if let Some(threads) = config.num_threads {
            builder = builder.num_threads(threads);
        }
        
        if let Some(stack_size) = config.stack_size {
            builder = builder.stack_size(stack_size);
        }
        
        builder.build_global()?;
    }
    
    Ok(())
}