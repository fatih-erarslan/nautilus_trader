//! Parallel processing utilities

#[cfg(feature = "rayon")]
pub use rayon::prelude::*;

pub mod thread_pool;
pub mod utils;

pub use thread_pool::*;
pub use utils::*;

// Re-export the backend for easier access
pub use utils::ParallelBackend;