//! SIMD acceleration module for high-performance computing

pub mod avx;
pub mod neon;
pub mod utils;

pub use utils::*;

// Re-export the backend for easier access
pub use utils::SimdBackend;

#[cfg(target_arch = "x86_64")]
pub use avx::*;

#[cfg(target_arch = "aarch64")]
pub use neon::*;