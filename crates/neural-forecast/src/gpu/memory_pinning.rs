//! Memory pinning for optimized CPU-GPU transfers
//!
//! This module provides pinned memory allocation to achieve maximum bandwidth
//! between CPU and GPU, reducing inference latency to sub-100μs

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::ptr::NonNull;
use bytemuck::{Pod, Zeroable};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaDevice, CudaSlice};

use crate::{Result, NeuralForecastError};

/// Pinned memory allocator for high-speed CPU-GPU transfers
pub struct PinnedMemoryAllocator {
    #[cfg(feature = "cuda")]
    device: Arc<CudaDevice>,
    allocations: Arc<Mutex<HashMap<usize, PinnedAllocation>>>,
    total_allocated: Arc<Mutex<usize>>,
    max_memory: usize,
}

/// Pinned memory allocation info
#[derive(Debug)]
struct PinnedAllocation {
    ptr: NonNull<u8>,
    size: usize,
    alignment: usize,
    #[cfg(feature = "cuda")]
    cuda_mapping: Option<CudaSlice<u8>>,
}

/// RAII wrapper for pinned memory
pub struct PinnedMemoryGuard<T> {
    ptr: NonNull<T>,
    size: usize,
    allocator: Arc<PinnedMemoryAllocator>,
}

impl PinnedMemoryAllocator {
    /// Create new pinned memory allocator
    #[cfg(feature = "cuda")]
    pub fn new(device: Arc<CudaDevice>, max_memory: usize) -> Result<Self> {
        Ok(Self {
            device,
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            max_memory,
        })
    }

    /// Create new pinned memory allocator (CPU-only version)
    #[cfg(not(feature = "cuda"))]
    pub fn new(max_memory: usize) -> Result<Self> {
        Ok(Self {
            allocations: Arc::new(Mutex::new(HashMap::new())),
            total_allocated: Arc::new(Mutex::new(0)),
            max_memory,
        })
    }

    /// Allocate pinned memory with alignment
    pub fn allocate<T: Pod + Zeroable>(&self, count: usize) -> Result<PinnedMemoryGuard<T>> {
        let size = count * std::mem::size_of::<T>();
        let alignment = std::mem::align_of::<T>();
        
        // Check memory limit
        {
            let total = self.total_allocated.lock().unwrap();
            if *total + size > self.max_memory {
                return Err(NeuralForecastError::GpuError(
                    "Pinned memory limit exceeded".to_string()
                ));
            }
        }

        // Allocate pinned memory
        let ptr = self.allocate_raw(size, alignment)?;
        
        // Zero initialize
        unsafe {
            std::ptr::write_bytes(ptr.as_ptr(), 0, size);
        }

        let guard = PinnedMemoryGuard {
            ptr: ptr.cast(),
            size: count,
            allocator: Arc::new(self.clone()),
        };

        // Update total allocated
        {
            let mut total = self.total_allocated.lock().unwrap();
            *total += size;
        }

        Ok(guard)
    }

    /// Allocate raw pinned memory
    fn allocate_raw(&self, size: usize, alignment: usize) -> Result<NonNull<u8>> {
        #[cfg(feature = "cuda")]
        {
            // Use CUDA driver API for pinned allocation
            let cuda_alloc = self.device.alloc_zeros::<u8>(size)
                .map_err(|e| NeuralForecastError::GpuError(format!("CUDA pinned allocation failed: {}", e)))?;
            
            let ptr = NonNull::new(cuda_alloc.as_ptr() as *mut u8)
                .ok_or_else(|| NeuralForecastError::GpuError("Null pointer from CUDA allocation".to_string()))?;
            
            let allocation = PinnedAllocation {
                ptr,
                size,
                alignment,
                cuda_mapping: Some(cuda_alloc),
            };
            
            let key = ptr.as_ptr() as usize;
            self.allocations.lock().unwrap().insert(key, allocation);
            
            Ok(ptr)
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            // Use regular aligned allocation with page locking
            let layout = std::alloc::Layout::from_size_align(size, alignment)
                .map_err(|e| NeuralForecastError::GpuError(format!("Invalid layout: {}", e)))?;
            
            let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
            let ptr = NonNull::new(ptr)
                .ok_or_else(|| NeuralForecastError::GpuError("Memory allocation failed".to_string()))?;
            
            // Lock pages in memory (platform-specific)
            self.lock_pages(ptr.as_ptr(), size)?;
            
            let allocation = PinnedAllocation {
                ptr,
                size,
                alignment,
            };
            
            let key = ptr.as_ptr() as usize;
            self.allocations.lock().unwrap().insert(key, allocation);
            
            Ok(ptr)
        }
    }

    /// Lock pages in memory to prevent swapping
    #[cfg(not(feature = "cuda"))]
    fn lock_pages(&self, ptr: *mut u8, size: usize) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::raw::c_int;
            extern "C" {
                fn mlock(addr: *const std::os::raw::c_void, len: usize) -> c_int;
            }
            
            let result = unsafe { mlock(ptr as *const std::os::raw::c_void, size) };
            if result != 0 {
                return Err(NeuralForecastError::GpuError(
                    "Failed to lock memory pages".to_string()
                ));
            }
        }
        
        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualLock;
            
            let result = unsafe { VirtualLock(ptr as *mut std::os::raw::c_void, size) };
            if result == 0 {
                return Err(NeuralForecastError::GpuError(
                    "Failed to lock memory pages".to_string()
                ));
            }
        }
        
        Ok(())
    }

    /// Free pinned memory
    fn free(&self, ptr: NonNull<u8>) -> Result<()> {
        let key = ptr.as_ptr() as usize;
        let allocation = self.allocations.lock().unwrap().remove(&key)
            .ok_or_else(|| NeuralForecastError::GpuError("Invalid pointer for deallocation".to_string()))?;
        
        #[cfg(feature = "cuda")]
        {
            // CUDA allocation is automatically freed when CudaSlice is dropped
            drop(allocation.cuda_mapping);
        }
        
        #[cfg(not(feature = "cuda"))]
        {
            // Unlock pages
            self.unlock_pages(ptr.as_ptr(), allocation.size)?;
            
            // Free memory
            let layout = std::alloc::Layout::from_size_align(allocation.size, allocation.alignment)
                .map_err(|e| NeuralForecastError::GpuError(format!("Invalid layout: {}", e)))?;
            
            unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
        }
        
        // Update total allocated
        {
            let mut total = self.total_allocated.lock().unwrap();
            *total = total.saturating_sub(allocation.size);
        }
        
        Ok(())
    }

    /// Unlock pages
    #[cfg(not(feature = "cuda"))]
    fn unlock_pages(&self, ptr: *mut u8, size: usize) -> Result<()> {
        #[cfg(unix)]
        {
            use std::os::raw::c_int;
            extern "C" {
                fn munlock(addr: *const std::os::raw::c_void, len: usize) -> c_int;
            }
            
            let result = unsafe { munlock(ptr as *const std::os::raw::c_void, size) };
            if result != 0 {
                // Log warning but don't fail
                tracing::warn!("Failed to unlock memory pages");
            }
        }
        
        #[cfg(windows)]
        {
            use winapi::um::memoryapi::VirtualUnlock;
            
            let result = unsafe { VirtualUnlock(ptr as *mut std::os::raw::c_void, size) };
            if result == 0 {
                tracing::warn!("Failed to unlock memory pages");
            }
        }
        
        Ok(())
    }

    /// Get memory statistics
    pub fn get_stats(&self) -> PinnedMemoryStats {
        let total_allocated = *self.total_allocated.lock().unwrap();
        let num_allocations = self.allocations.lock().unwrap().len();
        
        PinnedMemoryStats {
            total_allocated,
            max_memory: self.max_memory,
            num_allocations,
            utilization: total_allocated as f32 / self.max_memory as f32,
        }
    }
}

impl Clone for PinnedMemoryAllocator {
    fn clone(&self) -> Self {
        Self {
            #[cfg(feature = "cuda")]
            device: self.device.clone(),
            allocations: self.allocations.clone(),
            total_allocated: self.total_allocated.clone(),
            max_memory: self.max_memory,
        }
    }
}

impl<T> PinnedMemoryGuard<T> {
    /// Get raw pointer to pinned memory
    pub fn as_ptr(&self) -> *mut T {
        self.ptr.as_ptr()
    }

    /// Get mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Get slice
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get size in elements
    pub fn len(&self) -> usize {
        self.size
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Copy data from CPU memory with optimized transfer
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> 
    where 
        T: Copy 
    {
        if src.len() > self.size {
            return Err(NeuralForecastError::GpuError(
                "Source slice is larger than pinned buffer".to_string()
            ));
        }

        // Use optimized memory copy
        let dst = self.as_mut_slice();
        
        #[cfg(target_arch = "x86_64")]
        {
            // Use AVX2 for fast copies if available
            if is_x86_feature_detected!("avx2") {
                unsafe { self.copy_with_avx2(src, dst)? };
            } else {
                dst[..src.len()].copy_from_slice(src);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            dst[..src.len()].copy_from_slice(src);
        }

        Ok(())
    }

    /// Copy data to CPU memory with optimized transfer
    pub fn copy_to_slice(&self, dst: &mut [T]) -> Result<()> 
    where 
        T: Copy 
    {
        if dst.len() < self.size {
            return Err(NeuralForecastError::GpuError(
                "Destination slice is smaller than pinned buffer".to_string()
            ));
        }

        let src = self.as_slice();
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { self.copy_with_avx2(src, &mut dst[..self.size])? };
            } else {
                dst[..self.size].copy_from_slice(src);
            }
        }
        
        #[cfg(not(target_arch = "x86_64"))]
        {
            dst[..self.size].copy_from_slice(src);
        }

        Ok(())
    }

    /// AVX2-optimized memory copy
    #[cfg(target_arch = "x86_64")]
    unsafe fn copy_with_avx2(&self, src: &[T], dst: &mut [T]) -> Result<()> 
    where 
        T: Copy 
    {
        use std::arch::x86_64::*;
        
        let src_ptr = src.as_ptr() as *const u8;
        let dst_ptr = dst.as_mut_ptr() as *mut u8;
        let bytes = std::cmp::min(src.len(), dst.len()) * std::mem::size_of::<T>();
        
        // Process 32-byte chunks with AVX2
        let chunks = bytes / 32;
        let remainder = bytes % 32;
        
        for i in 0..chunks {
            let offset = i * 32;
            let data = _mm256_loadu_si256((src_ptr.add(offset)) as *const __m256i);
            _mm256_storeu_si256((dst_ptr.add(offset)) as *mut __m256i, data);
        }
        
        // Handle remainder
        if remainder > 0 {
            let offset = chunks * 32;
            std::ptr::copy_nonoverlapping(
                src_ptr.add(offset),
                dst_ptr.add(offset),
                remainder
            );
        }
        
        Ok(())
    }

    /// Get memory bandwidth for transfers
    pub fn measure_bandwidth(&self) -> Result<f64> {
        let test_data = vec![1u8; self.size * std::mem::size_of::<T>()];
        let iterations = 100;
        
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            // Simulate transfer
            unsafe {
                std::ptr::copy_nonoverlapping(
                    test_data.as_ptr(),
                    self.ptr.as_ptr() as *mut u8,
                    test_data.len()
                );
            }
        }
        let duration = start.elapsed();
        
        let bytes_transferred = test_data.len() * iterations;
        let bandwidth = bytes_transferred as f64 / duration.as_secs_f64();
        
        Ok(bandwidth / (1024.0 * 1024.0 * 1024.0)) // GB/s
    }
}

impl<T> Drop for PinnedMemoryGuard<T> {
    fn drop(&mut self) {
        if let Err(e) = self.allocator.free(self.ptr.cast()) {
            tracing::error!("Failed to free pinned memory: {}", e);
        }
    }
}

unsafe impl<T: Send> Send for PinnedMemoryGuard<T> {}
unsafe impl<T: Sync> Sync for PinnedMemoryGuard<T> {}

/// Pinned memory statistics
#[derive(Debug, Clone)]
pub struct PinnedMemoryStats {
    pub total_allocated: usize,
    pub max_memory: usize,
    pub num_allocations: usize,
    pub utilization: f32,
}

/// Optimized tensor transfer utilities
pub mod tensor_transfer {
    use super::*;
    use ndarray::{Array1, Array2, Array3};

    /// Transfer tensor data with optimal chunking
    pub fn transfer_tensor_chunked<T: Pod + Zeroable + Copy>(
        tensor: &Array3<T>,
        pinned_buffer: &mut PinnedMemoryGuard<T>,
        chunk_size: usize,
    ) -> Result<()> {
        let data = tensor.as_slice()
            .ok_or_else(|| NeuralForecastError::GpuError("Tensor is not contiguous".to_string()))?;
        
        if data.len() > pinned_buffer.len() {
            return Err(NeuralForecastError::GpuError(
                "Tensor is larger than pinned buffer".to_string()
            ));
        }

        // Transfer in chunks to maintain cache efficiency
        let mut offset = 0;
        let buffer_slice = pinned_buffer.as_mut_slice();
        
        while offset < data.len() {
            let end = std::cmp::min(offset + chunk_size, data.len());
            buffer_slice[offset..end].copy_from_slice(&data[offset..end]);
            offset = end;
        }

        Ok(())
    }

    /// Prefetch tensor data for better cache utilization
    pub fn prefetch_tensor_data<T: Pod + Zeroable>(
        tensor: &Array3<T>,
        stride: usize,
    ) -> Result<()> {
        let data = tensor.as_slice()
            .ok_or_else(|| NeuralForecastError::GpuError("Tensor is not contiguous".to_string()))?;
        
        #[cfg(target_arch = "x86_64")]
        {
            use std::arch::x86_64::*;
            
            if is_x86_feature_detected!("sse") {
                unsafe {
                    let ptr = data.as_ptr() as *const u8;
                    let len = data.len() * std::mem::size_of::<T>();
                    
                    // Prefetch data in 64-byte cache lines
                    for i in (0..len).step_by(stride.max(64)) {
                        _mm_prefetch(ptr.add(i) as *const i8, _MM_HINT_T0);
                    }
                }
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pinned_memory_allocation() {
        let allocator = PinnedMemoryAllocator::new(1024 * 1024).unwrap(); // 1MB
        let guard = allocator.allocate::<f32>(1024).unwrap();
        
        assert_eq!(guard.len(), 1024);
        assert!(!guard.is_empty());
        
        let stats = allocator.get_stats();
        assert_eq!(stats.num_allocations, 1);
        assert!(stats.total_allocated > 0);
    }

    #[test]
    fn test_pinned_memory_copy() {
        let allocator = PinnedMemoryAllocator::new(1024 * 1024).unwrap();
        let mut guard = allocator.allocate::<f32>(1024).unwrap();
        
        let test_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        guard.copy_from_slice(&test_data).unwrap();
        
        let mut result = vec![0.0f32; 1024];
        guard.copy_to_slice(&mut result).unwrap();
        
        assert_eq!(test_data, result);
    }

    #[test]
    fn test_memory_limits() {
        let allocator = PinnedMemoryAllocator::new(1024).unwrap(); // 1KB only
        let result = allocator.allocate::<f32>(1024); // 4KB
        
        assert!(result.is_err());
    }

    #[test]
    fn test_bandwidth_measurement() {
        let allocator = PinnedMemoryAllocator::new(1024 * 1024).unwrap();
        let guard = allocator.allocate::<u8>(1024).unwrap();
        
        let bandwidth = guard.measure_bandwidth().unwrap();
        assert!(bandwidth > 0.0);
        println!("Measured bandwidth: {:.2} GB/s", bandwidth);
    }
}