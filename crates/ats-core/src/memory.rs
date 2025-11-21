//! High-Performance Memory Management for ATS-Core
//!
//! This module provides zero-copy array operations and memory-aligned data structures
//! optimized for SIMD operations and cache efficiency.

use crate::{
    config::AtsCpConfig,
    error::{AtsCoreError, Result},
    types::AlignedVec,
};
// use aligned_vec::AlignedVec as ExternalAlignedVec;  // Commented out - using internal implementation
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

/// High-performance memory manager for ATS-CP operations
pub struct MemoryManager {
    /// Configuration parameters
    config: AtsCpConfig,
    
    /// Memory pool for frequent allocations
    #[allow(dead_code)]
    memory_pool: MemoryPool,
    
    /// Total allocated bytes
    total_allocated: u64,
    
    /// Peak memory usage
    peak_usage: u64,
    
    /// Number of allocations
    allocation_count: u64,
}

impl MemoryManager {
    /// Creates a new memory manager with the specified configuration
    pub fn new(config: &AtsCpConfig) -> Result<Self> {
        let memory_pool = MemoryPool::new(
            config.memory.pool_size_mb * 1024 * 1024,
            config.memory.default_alignment,
        )?;
        
        Ok(Self {
            config: config.clone(),
            memory_pool,
            total_allocated: 0,
            peak_usage: 0,
            allocation_count: 0,
        })
    }

    /// Allocates an aligned array
    pub fn allocate_aligned<T>(&mut self, capacity: usize) -> Result<AlignedVec<T>>
    where
        T: Clone + Default + bytemuck::Pod + bytemuck::Zeroable,
    {
        let aligned_vec = AlignedVec::new(capacity, self.config.memory.default_alignment);
        
        // Update statistics
        let bytes = capacity * std::mem::size_of::<T>();
        self.total_allocated += bytes as u64;
        self.allocation_count += 1;
        self.peak_usage = self.peak_usage.max(self.total_allocated);
        
        Ok(aligned_vec)
    }

    /// Returns memory usage statistics
    pub fn get_memory_stats(&self) -> (u64, u64, u64) {
        (self.total_allocated, self.peak_usage, self.allocation_count)
    }
}

/// Memory pool for frequent allocations
struct MemoryPool {
    /// Pool memory
    memory: NonNull<u8>,
    
    /// Pool size in bytes
    size: usize,
    
    /// Current offset
    #[allow(dead_code)]
    offset: usize,
    
    /// Alignment requirement
    alignment: usize,
}

impl MemoryPool {
    /// Creates a new memory pool
    fn new(size: usize, alignment: usize) -> Result<Self> {
        let layout = Layout::from_size_align(size, alignment)
            .map_err(|_| AtsCoreError::memory("invalid memory layout"))?;
        
        let memory = unsafe {
            let ptr = alloc(layout);
            // SECURITY FIX: Use safe NonNull creation to prevent memory corruption (CVSS 5.4)
            NonNull::new(ptr).ok_or_else(|| AtsCoreError::memory("failed to allocate memory pool"))?
        };
        
        Ok(Self {
            memory,
            size,
            offset: 0,
            alignment,
        })
    }
}

impl Drop for MemoryPool {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::from_size_align(self.size, self.alignment).unwrap();
            dealloc(self.memory.as_ptr(), layout);
        }
    }
}

/// Zero-copy array view for high-performance operations
pub struct ArrayView<'a, T> {
    data: &'a [T],
    alignment: usize,
}

impl<'a, T> ArrayView<'a, T> {
    /// Creates a new array view
    pub fn new(data: &'a [T], alignment: usize) -> Self {
        Self { data, alignment }
    }

    /// Returns the data slice
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns the length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}

/// Mutable zero-copy array view
pub struct ArrayViewMut<'a, T> {
    data: &'a mut [T],
    alignment: usize,
}

impl<'a, T> ArrayViewMut<'a, T> {
    /// Creates a new mutable array view
    pub fn new(data: &'a mut [T], alignment: usize) -> Self {
        Self { data, alignment }
    }

    /// Returns the data slice
    pub fn as_slice(&self) -> &[T] {
        self.data
    }

    /// Returns the mutable data slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.data
    }

    /// Returns the length
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns true if empty
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the alignment
    pub fn alignment(&self) -> usize {
        self.alignment
    }
}