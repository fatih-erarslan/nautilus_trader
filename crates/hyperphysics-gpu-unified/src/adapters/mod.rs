//! Adapters for integrating with existing GPU trait systems

mod cwts_adapter;

pub use cwts_adapter::{UnifiedGpuAccelerator, UnifiedGpuBuffer, UnifiedGpuKernel};
