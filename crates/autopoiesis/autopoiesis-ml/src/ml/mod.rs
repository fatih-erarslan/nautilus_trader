//! Machine Learning Module
//! Advanced neural architectures for autopoietic systems

pub mod nhits;

// Re-export main types
pub use nhits::{NHITS, NHITSConfig, NHITSError};

/// Prelude for convenient imports
pub mod prelude {
    pub use super::nhits::prelude::*;
}