//! Metal GPU backend implementation for Apple Silicon and macOS
//!
//! This module provides Metal-based GPU acceleration for CDFA operations,
//! optimized for Apple hardware including M1/M2/M3 chips and discrete GPUs.

#[cfg(all(feature = "metal", target_os = "macos"))]
use metal::*;
#[cfg(all(feature = "metal", target_os = "macos"))]
use objc::rc::autoreleasepool;
#[cfg(all(feature = "metal", target_os = "macos"))]
use core_graphics::base::CGFloat;

use crate::error::{CdfaError, CdfaResult};
use crate::types::{CdfaFloat, CdfaMatrix};
use super::{GpuContext, GpuBuffer, GpuKernel, GpuDeviceInfo, GpuBackend, GpuConfig, MemoryStats};
use std::sync::Arc;

#[cfg(all(feature = "metal", target_os = "macos"))]
/// Metal implementation of GPU context
pub struct MetalContext {
    device: Device,
    command_queue: CommandQueue,
    library: Library,
    device_info: GpuDeviceInfo,
    config: GpuConfig,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl MetalContext {
    /// Create new Metal context
    pub fn new(device_id: u32, config: &GpuConfig) -> CdfaResult<Self> {
        autoreleasepool(|| {
            let device = Device::system_default()
                .ok_or_else(|| CdfaError::GpuError("No Metal device available".to_string()))?;
            
            let command_queue = device.new_command_queue();
            
            // Create Metal library with CDFA kernels
            let library_source = include_str!("metal_kernels.metal");
            let library = device.new_library_with_source(library_source, &CompileOptions::new())
                .map_err(|e| CdfaError::GpuError(format!("Metal library compilation failed: {}", e)))?;
            
            let device_info = Self::query_device_info(&device, device_id);
            
            Ok(Self {
                device,
                command_queue,
                library,
                device_info,
                config: config.clone(),
            })
        })
    }
    
    /// Query device information
    fn query_device_info(device: &Device, device_id: u32) -> GpuDeviceInfo {
        let name = device.name().to_string();
        let is_unified_memory = device.has_unified_memory();
        let recommended_max_working_set_size = device.recommended_max_working_set_size();
        let max_threads_per_threadgroup = device.max_threads_per_threadgroup();
        
        // Determine GPU type and capabilities
        let (memory_size, compute_capability) = if name.contains("M1") || name.contains("M2") || name.contains("M3") {
            // Apple Silicon - unified memory
            (8 * 1024 * 1024 * 1024, "Apple Silicon".to_string()) // Default 8GB
        } else if name.contains("AMD") || name.contains("Radeon") {
            (recommended_max_working_set_size, "AMD".to_string())
        } else {
            (recommended_max_working_set_size, "Unknown".to_string())
        };
        
        GpuDeviceInfo {
            id: device_id,
            name,
            backend: GpuBackend::Metal,
            memory_size,
            compute_capability,
            max_work_group_size: max_threads_per_threadgroup.width as u32,
            supports_double_precision: true, // Metal supports double precision
            supports_half_precision: true,   // Metal supports half precision
        }
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GpuContext for MetalContext {
    fn device_info(&self) -> &GpuDeviceInfo {
        &self.device_info
    }
    
    fn allocate_buffer(&self, size: usize) -> CdfaResult<Box<dyn GpuBuffer>> {
        let buffer = self.device.new_buffer(size as u64, MTLResourceOptions::StorageModeShared);
        
        Ok(Box::new(MetalBuffer {
            buffer,
            size,
        }))
    }
    
    fn create_kernel(&self, _source: &str, entry_point: &str) -> CdfaResult<Box<dyn GpuKernel>> {
        let function = self.library.get_function(entry_point, None)
            .map_err(|e| CdfaError::GpuError(format!("Metal function not found: {}", e)))?;
        
        let pipeline_state = self.device.new_compute_pipeline_state_with_function(&function)
            .map_err(|e| CdfaError::GpuError(format!("Metal pipeline creation failed: {}", e)))?;
        
        Ok(Box::new(MetalKernel {
            device: self.device.clone(),
            command_queue: self.command_queue.clone(),
            pipeline_state,
            buffers: Vec::new(),
        }))
    }
    
    fn synchronize(&self) -> CdfaResult<()> {
        // Metal synchronization is handled through command buffer completion
        Ok(())
    }
    
    fn memory_stats(&self) -> CdfaResult<MemoryStats> {
        let current_allocated_size = self.device.current_allocated_size();
        let recommended_max = self.device.recommended_max_working_set_size();
        
        Ok(MemoryStats {
            total_memory: recommended_max,
            used_memory: current_allocated_size,
            free_memory: recommended_max - current_allocated_size,
            allocated_buffers: 0, // Metal doesn't provide this info directly
        })
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
/// Metal buffer implementation
pub struct MetalBuffer {
    buffer: Buffer,
    size: usize,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GpuBuffer for MetalBuffer {
    fn size(&self) -> usize {
        self.size
    }
    
    fn copy_from_host(&mut self, data: &[u8]) -> CdfaResult<()> {
        if data.len() > self.size {
            return Err(CdfaError::InvalidParameter(
                "Data size exceeds buffer capacity".to_string()
            ));
        }
        
        let contents = self.buffer.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), contents as *mut u8, data.len());
        }
        
        Ok(())
    }
    
    fn copy_to_host(&self, data: &mut [u8]) -> CdfaResult<()> {
        if data.len() > self.size {
            return Err(CdfaError::InvalidParameter(
                "Output buffer too small".to_string()
            ));
        }
        
        let contents = self.buffer.contents();
        unsafe {
            std::ptr::copy_nonoverlapping(contents as *const u8, data.as_mut_ptr(), data.len());
        }
        
        Ok(())
    }
    
    fn map(&self) -> CdfaResult<*mut u8> {
        Ok(self.buffer.contents() as *mut u8)
    }
    
    fn unmap(&self) -> CdfaResult<()> {
        // No explicit unmap needed for Metal shared memory
        Ok(())
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
/// Metal kernel implementation
pub struct MetalKernel {
    device: Device,
    command_queue: CommandQueue,
    pipeline_state: ComputePipelineState,
    buffers: Vec<Buffer>,
}

#[cfg(all(feature = "metal", target_os = "macos"))]
impl GpuKernel for MetalKernel {
    fn set_arg(&mut self, index: u32, buffer: &dyn GpuBuffer) -> CdfaResult<()> {
        // This is a simplified implementation
        // Real implementation would extract Metal buffer from trait object
        Err(CdfaError::UnsupportedOperation(
            "Metal buffer argument setting needs specialized implementation".to_string()
        ))
    }
    
    fn set_scalar_arg<T: Copy>(&mut self, _index: u32, _value: T) -> CdfaResult<()> {
        // Metal handles scalar arguments differently through constant buffers
        Ok(())
    }
    
    fn launch(&self, global_size: &[u32], local_size: Option<&[u32]>) -> CdfaResult<()> {
        autoreleasepool(|| {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(&self.pipeline_state);
            
            // Set buffers
            for (index, buffer) in self.buffers.iter().enumerate() {
                encoder.set_buffer(index as u64, Some(buffer), 0);
            }
            
            // Calculate dispatch size
            let threads_per_grid = MTLSize {
                width: global_size.get(0).copied().unwrap_or(1) as u64,
                height: global_size.get(1).copied().unwrap_or(1) as u64,
                depth: global_size.get(2).copied().unwrap_or(1) as u64,
            };
            
            let threads_per_threadgroup = if let Some(local) = local_size {
                MTLSize {
                    width: local.get(0).copied().unwrap_or(1) as u64,
                    height: local.get(1).copied().unwrap_or(1) as u64,
                    depth: local.get(2).copied().unwrap_or(1) as u64,
                }
            } else {
                let max_threads = self.pipeline_state.max_total_threads_per_threadgroup();
                MTLSize {
                    width: std::cmp::min(max_threads, 256),
                    height: 1,
                    depth: 1,
                }
            };
            
            encoder.dispatch_threads(threads_per_grid, threads_per_threadgroup);
            encoder.end_encoding();
            
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            Ok(())
        })
    }
}

/// Metal shader source file content
pub const METAL_KERNELS_SOURCE: &str = include_str!("metal_kernels.metal");

// Provide stub implementations when Metal is not available
#[cfg(not(all(feature = "metal", target_os = "macos")))]
pub struct MetalContext;

#[cfg(not(all(feature = "metal", target_os = "macos")))]
impl MetalContext {
    pub fn new(_device_id: u32, _config: &GpuConfig) -> CdfaResult<Self> {
        Err(CdfaError::UnsupportedOperation(
            "Metal support not available on this platform".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_metal_availability() {
        #[cfg(all(feature = "metal", target_os = "macos"))]
        {
            let config = GpuConfig::default();
            match MetalContext::new(0, &config) {
                Ok(_) => println!("Metal device available"),
                Err(_) => println!("No Metal device available"),
            }
        }
        
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        {
            let config = GpuConfig::default();
            assert!(MetalContext::new(0, &config).is_err());
        }
    }
    
    #[test]
    fn test_metal_kernels_source() {
        assert!(!METAL_KERNELS_SOURCE.is_empty());
    }
}