pub mod cuda;
pub mod hip;
pub mod metal;
pub mod probabilistic_kernels;
pub mod vulkan;

#[cfg(feature = "cuda")]
pub use cuda::*;

#[cfg(feature = "hip")]
pub use hip::*;

#[cfg(target_os = "macos")]
pub use metal::*;

#[cfg(feature = "vulkan")]
pub use vulkan::*;

pub use probabilistic_kernels::*;

// GPU Acceleration Trait Definitions for pBit Engine

/// GPU accelerator abstraction
pub trait GpuAccelerator: Send + Sync {
    fn allocate_buffer(&self, size: usize) -> Result<std::sync::Arc<dyn GpuMemoryBuffer>, String>;
    fn create_kernel(&self, name: &str) -> Result<std::sync::Arc<dyn GpuKernel>, String>;
    fn compile_kernel(
        &self,
        name: &str,
        source: &str,
    ) -> Result<std::sync::Arc<dyn GpuKernel>, String>;
}

/// GPU memory buffer abstraction
pub trait GpuMemoryBuffer: Send + Sync {
    fn write(&self, data: &[u8]) -> Result<(), String>;
    fn read(&self) -> Result<Vec<u8>, String>;
    fn write_at_offset(&self, data: &[u8], offset: usize) -> Result<(), String>;
    fn size(&self) -> usize;
}

/// GPU kernel execution abstraction
pub trait GpuKernel: Send + Sync {
    fn execute(
        &self,
        buffers: &[&dyn GpuMemoryBuffer],
        work_groups: (u32, u32, u32),
    ) -> Result<(), String>;
}
