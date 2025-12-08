//! Basic parallel operations

use crate::types::Float;

/// Check if parallel processing is available
pub fn parallel_available() -> bool {
    #[cfg(feature = "parallel")]
    {
        true
    }
    #[cfg(not(feature = "parallel"))]
    {
        false
    }
}

/// Parallel map operation
pub fn parallel_map<T, F, R>(data: &[T], f: F) -> Vec<R>
where
    T: Sync,
    F: Fn(&T) -> R + Sync,
    R: Send,
{
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        data.par_iter().map(f).collect()
    }
    #[cfg(not(feature = "parallel"))]
    {
        data.iter().map(f).collect()
    }
}