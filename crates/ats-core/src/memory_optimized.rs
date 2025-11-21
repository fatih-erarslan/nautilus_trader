//! Memory-Optimized Data Structures for ATS-Core
//!
//! This module implements cache-aligned memory buffers and optimized access patterns
//! to achieve maximum performance in conformal prediction operations.

use crate::{
    error::{AtsCoreError, Result},
    types::Precision,
};
use std::{
    alloc::{alloc_zeroed, dealloc, Layout},
    ptr::{self, NonNull},
    mem,
};

/// Cache line size for modern x86_64 processors (64 bytes)
pub const CACHE_LINE_SIZE: usize = 64;

/// Memory page size (4KB on most systems)
pub const PAGE_SIZE: usize = 4096;

/// NUMA-aware memory allocator for high-performance computing
#[derive(Debug)]
pub struct NumaAlignedAllocator {
    /// NUMA node preference (-1 for no preference)
    numa_node: i32,
    /// Cache line alignment
    alignment: usize,
}

impl NumaAlignedAllocator {
    /// Creates a new NUMA-aware allocator
    pub fn new(numa_node: i32) -> Self {
        Self {
            numa_node,
            alignment: CACHE_LINE_SIZE,
        }
    }
    
    /// Allocates cache-aligned memory
    pub fn allocate<T>(&self, count: usize) -> Result<NonNull<T>> {
        let size = count * mem::size_of::<T>();
        let layout = Layout::from_size_align(size, self.alignment)
            .map_err(|_| AtsCoreError::memory("invalid layout for allocation"))?;
            
        unsafe {
            let ptr = alloc_zeroed(layout);
            if ptr.is_null() {
                return Err(AtsCoreError::memory("allocation failed"));
            }
            
            // Prefault pages for better performance
            self.prefault_pages(ptr, size);
            
            Ok(NonNull::new_unchecked(ptr as *mut T))
        }
    }
    
    /// Deallocates memory
    pub unsafe fn deallocate<T>(&self, ptr: NonNull<T>, count: usize) {
        let size = count * mem::size_of::<T>();
        let layout = Layout::from_size_align(size, self.alignment).unwrap();
        dealloc(ptr.as_ptr() as *mut u8, layout);
    }
    
    /// Prefaults memory pages to reduce TLB misses
    unsafe fn prefault_pages(&self, ptr: *mut u8, size: usize) {
        let pages = (size + PAGE_SIZE - 1) / PAGE_SIZE;
        for i in 0..pages {
            let page_ptr = ptr.add(i * PAGE_SIZE);
            // Touch first byte of each page
            ptr::write_volatile(page_ptr, ptr::read_volatile(page_ptr));
        }
    }
}

/// High-performance cache-aligned vector optimized for SIMD operations
#[derive(Debug)]
pub struct CacheAlignedVec<T> {
    ptr: NonNull<T>,
    len: usize,
    capacity: usize,
    allocator: NumaAlignedAllocator,
}

unsafe impl<T: Send> Send for CacheAlignedVec<T> {}
unsafe impl<T: Sync> Sync for CacheAlignedVec<T> {}

impl<T: Clone + Default> CacheAlignedVec<T> {
    /// Creates a new cache-aligned vector
    pub fn new(capacity: usize) -> Result<Self> {
        Self::with_numa_node(capacity, -1)
    }
    
    /// Creates a new cache-aligned vector on specific NUMA node
    pub fn with_numa_node(capacity: usize, numa_node: i32) -> Result<Self> {
        let allocator = NumaAlignedAllocator::new(numa_node);
        let ptr = allocator.allocate::<T>(capacity)?;
        
        // Initialize with default values
        unsafe {
            for i in 0..capacity {
                ptr::write(ptr.as_ptr().add(i), T::default());
            }
        }
        
        Ok(Self {
            ptr,
            len: capacity,
            capacity,
            allocator,
        })
    }
    
    /// Creates a vector from existing data with optimal memory layout
    pub fn from_slice(data: &[T]) -> Result<Self> {
        let mut vec = Self::new(data.len())?;
        unsafe {
            for (i, item) in data.iter().enumerate() {
                ptr::write(vec.ptr.as_ptr().add(i), item.clone());
            }
        }
        vec.len = data.len();
        Ok(vec)
    }
    
    /// Returns the length of the vector
    pub fn len(&self) -> usize {
        self.len
    }
    
    /// Returns true if the vector is empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    
    /// Returns the capacity of the vector
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Returns a slice view of the data
    pub fn as_slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
    
    /// Returns a mutable slice view of the data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
    
    /// Returns the raw pointer for SIMD operations
    pub fn as_ptr(&self) -> *const T {
        self.ptr.as_ptr()
    }
    
    /// Returns the mutable raw pointer for SIMD operations
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr.as_ptr()
    }
    
    /// Resizes the vector (within capacity)
    pub fn resize(&mut self, new_len: usize) -> Result<()> {
        if new_len > self.capacity {
            return Err(AtsCoreError::validation("new_len", "exceeds capacity"));
        }
        
        if new_len > self.len {
            // Initialize new elements
            unsafe {
                for i in self.len..new_len {
                    ptr::write(self.ptr.as_ptr().add(i), T::default());
                }
            }
        }
        
        self.len = new_len;
        Ok(())
    }
    
    /// Copies data from slice with prefetching for better performance
    pub fn copy_from_slice(&mut self, src: &[T]) -> Result<()> {
        if src.len() > self.capacity {
            return Err(AtsCoreError::validation("src", "exceeds vector capacity"));
        }
        
        unsafe {
            // Prefetch source data
            self.prefetch_data(src.as_ptr(), src.len());
            
            // Copy data with optimal access pattern
            for (i, item) in src.iter().enumerate() {
                ptr::write(self.ptr.as_ptr().add(i), item.clone());
            }
        }
        
        self.len = src.len();
        Ok(())
    }
    
    /// Prefetches data into cache for better performance
    unsafe fn prefetch_data(&self, ptr: *const T, count: usize) {
        let size = count * mem::size_of::<T>();
        let cache_lines = (size + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE;
        
        for i in 0..cache_lines {
            let prefetch_ptr = (ptr as *const u8).add(i * CACHE_LINE_SIZE);
            #[cfg(target_arch = "x86_64")]
            {
                std::arch::x86_64::_mm_prefetch::<{std::arch::x86_64::_MM_HINT_T0}>(
                    prefetch_ptr as *const i8
                );
            }
        }
    }
    
    /// Clears the vector efficiently
    pub fn clear(&mut self) {
        unsafe {
            // Use memset for better performance on large arrays
            ptr::write_bytes(self.ptr.as_ptr(), 0, self.len);
        }
        self.len = 0;
    }
    
    /// Returns memory usage statistics
    pub fn memory_stats(&self) -> MemoryStats {
        MemoryStats {
            allocated_bytes: self.capacity * mem::size_of::<T>(),
            used_bytes: self.len * mem::size_of::<T>(),
            alignment: CACHE_LINE_SIZE,
            cache_lines: (self.capacity * mem::size_of::<T>() + CACHE_LINE_SIZE - 1) / CACHE_LINE_SIZE,
        }
    }
}

impl<T> Drop for CacheAlignedVec<T> {
    fn drop(&mut self) {
        unsafe {
            // Drop elements first
            for i in 0..self.len {
                ptr::drop_in_place(self.ptr.as_ptr().add(i));
            }
            
            // Deallocate memory
            self.allocator.deallocate(self.ptr, self.capacity);
        }
    }
}

impl<T> std::ops::Index<usize> for CacheAlignedVec<T> {
    type Output = T;
    
    fn index(&self, index: usize) -> &Self::Output {
        assert!(index < self.len, "index out of bounds");
        unsafe { &*self.ptr.as_ptr().add(index) }
    }
}

impl<T> std::ops::IndexMut<usize> for CacheAlignedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        assert!(index < self.len, "index out of bounds");
        unsafe { &mut *self.ptr.as_ptr().add(index) }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub allocated_bytes: usize,
    pub used_bytes: usize,
    pub alignment: usize,
    pub cache_lines: usize,
}

/// Ring buffer for streaming data processing with optimal cache behavior
#[derive(Debug)]
pub struct CacheOptimizedRingBuffer<T> {
    data: CacheAlignedVec<T>,
    head: usize,
    tail: usize,
    count: usize,
}

impl<T: Clone + Default> CacheOptimizedRingBuffer<T> {
    /// Creates a new ring buffer with specified capacity
    pub fn new(capacity: usize) -> Result<Self> {
        // Round up to next power of 2 for efficient modulo operations
        let capacity = capacity.next_power_of_two();
        let data = CacheAlignedVec::new(capacity)?;
        
        Ok(Self {
            data,
            head: 0,
            tail: 0,
            count: 0,
        })
    }
    
    /// Pushes an element to the ring buffer
    pub fn push(&mut self, item: T) -> Result<()> {
        if self.count == self.capacity() {
            return Err(AtsCoreError::validation("ring_buffer", "buffer is full"));
        }
        
        self.data[self.tail] = item;
        self.tail = (self.tail + 1) & (self.capacity() - 1); // Fast modulo for powers of 2
        self.count += 1;
        
        Ok(())
    }
    
    /// Pops an element from the ring buffer
    pub fn pop(&mut self) -> Option<T> {
        if self.count == 0 {
            return None;
        }
        
        let item = self.data[self.head].clone();
        self.head = (self.head + 1) & (self.capacity() - 1);
        self.count -= 1;
        
        Some(item)
    }
    
    /// Returns the number of elements in the buffer
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Returns true if the buffer is empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Returns true if the buffer is full
    pub fn is_full(&self) -> bool {
        self.count == self.capacity()
    }
    
    /// Returns the capacity of the buffer
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }
    
    /// Clears all elements from the buffer
    pub fn clear(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.count = 0;
    }
    
    /// Returns a contiguous view of the data (may require copying)
    pub fn as_contiguous_slice(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.count);
        
        for i in 0..self.count {
            let idx = (self.head + i) & (self.capacity() - 1);
            result.push(self.data[idx].clone());
        }
        
        result
    }
}

/// Memory pool for frequent allocations/deallocations
pub struct MemoryPool<T> {
    available_blocks: Vec<CacheAlignedVec<T>>,
    block_size: usize,
    max_blocks: usize,
    allocator: NumaAlignedAllocator,
}

impl<T: Clone + Default> MemoryPool<T> {
    /// Creates a new memory pool
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            available_blocks: Vec::new(),
            block_size,
            max_blocks,
            allocator: NumaAlignedAllocator::new(-1),
        }
    }
    
    /// Acquires a block from the pool
    pub fn acquire(&mut self) -> Result<CacheAlignedVec<T>> {
        if let Some(block) = self.available_blocks.pop() {
            Ok(block)
        } else {
            CacheAlignedVec::new(self.block_size)
        }
    }
    
    /// Returns a block to the pool
    pub fn release(&mut self, mut block: CacheAlignedVec<T>) -> Result<()> {
        if self.available_blocks.len() < self.max_blocks {
            block.clear();
            self.available_blocks.push(block);
        }
        Ok(())
    }
    
    /// Returns pool statistics
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            available_blocks: self.available_blocks.len(),
            block_size: self.block_size,
            max_blocks: self.max_blocks,
            total_memory_bytes: self.available_blocks.len() * self.block_size * mem::size_of::<T>(),
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub available_blocks: usize,
    pub block_size: usize,
    pub max_blocks: usize,
    pub total_memory_bytes: usize,
}

/// Specialized memory layout for conformal prediction data
#[derive(Debug)]
pub struct ConformalDataLayout {
    /// Predictions buffer
    pub predictions: CacheAlignedVec<Precision>,
    /// Calibration scores buffer
    pub calibration_scores: CacheAlignedVec<Precision>,
    /// Working buffer for computations
    pub work_buffer: CacheAlignedVec<Precision>,
    /// Results buffer
    pub results_buffer: CacheAlignedVec<Precision>,
}

impl ConformalDataLayout {
    /// Creates a new conformal data layout with optimal memory arrangement
    pub fn new(
        max_predictions: usize,
        max_calibration_samples: usize,
    ) -> Result<Self> {
        Ok(Self {
            predictions: CacheAlignedVec::new(max_predictions)?,
            calibration_scores: CacheAlignedVec::new(max_calibration_samples)?,
            work_buffer: CacheAlignedVec::new(max_predictions.max(max_calibration_samples))?,
            results_buffer: CacheAlignedVec::new(max_predictions * 2)?, // For intervals (lower, upper)
        })
    }
    
    /// Returns total memory usage
    pub fn total_memory_bytes(&self) -> usize {
        self.predictions.memory_stats().allocated_bytes +
        self.calibration_scores.memory_stats().allocated_bytes +
        self.work_buffer.memory_stats().allocated_bytes +
        self.results_buffer.memory_stats().allocated_bytes
    }
    
    /// Validates memory layout for optimal cache utilization
    pub fn validate_cache_efficiency(&self) -> CacheEfficiencyReport {
        let mut report = CacheEfficiencyReport::default();
        
        // Check alignment
        report.cache_aligned = (self.predictions.as_ptr() as usize) % CACHE_LINE_SIZE == 0;
        
        // Calculate cache line utilization
        let pred_stats = self.predictions.memory_stats();
        report.cache_utilization = pred_stats.used_bytes as f64 / 
                                  (pred_stats.cache_lines * CACHE_LINE_SIZE) as f64;
        
        // Check for false sharing potential
        report.false_sharing_risk = self.check_false_sharing();
        
        report
    }
    
    /// Checks for potential false sharing between buffers
    fn check_false_sharing(&self) -> bool {
        let pred_end = (self.predictions.as_ptr() as usize) + self.predictions.memory_stats().used_bytes;
        let calib_start = self.calibration_scores.as_ptr() as usize;
        
        // If buffers are within the same cache line, there's risk of false sharing
        (pred_end / CACHE_LINE_SIZE) == (calib_start / CACHE_LINE_SIZE)
    }
}

/// Cache efficiency analysis report
#[derive(Debug, Clone, Default)]
pub struct CacheEfficiencyReport {
    pub cache_aligned: bool,
    pub cache_utilization: f64,
    pub false_sharing_risk: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_aligned_vec_creation() {
        let vec: CacheAlignedVec<f64> = CacheAlignedVec::new(100).unwrap();
        assert_eq!(vec.len(), 100);
        assert_eq!(vec.capacity(), 100);
        
        // Check cache alignment
        let ptr = vec.as_ptr() as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }
    
    #[test]
    fn test_cache_aligned_vec_operations() {
        let mut vec: CacheAlignedVec<f64> = CacheAlignedVec::new(10).unwrap();
        
        // Test indexing
        vec[0] = 1.0;
        vec[9] = 2.0;
        assert_eq!(vec[0], 1.0);
        assert_eq!(vec[9], 2.0);
        
        // Test resize
        vec.resize(5).unwrap();
        assert_eq!(vec.len(), 5);
    }
    
    #[test]
    fn test_ring_buffer() {
        let mut ring: CacheOptimizedRingBuffer<i32> = CacheOptimizedRingBuffer::new(4).unwrap();
        
        // Test push and pop
        ring.push(1).unwrap();
        ring.push(2).unwrap();
        ring.push(3).unwrap();
        
        assert_eq!(ring.len(), 3);
        assert_eq!(ring.pop(), Some(1));
        assert_eq!(ring.pop(), Some(2));
        assert_eq!(ring.len(), 1);
        
        // Test circular behavior
        ring.push(4).unwrap();
        ring.push(5).unwrap();
        ring.push(6).unwrap();
        assert!(ring.is_full());
        
        let items = ring.as_contiguous_slice();
        assert_eq!(items, vec![3, 4, 5, 6]);
    }
    
    #[test]
    fn test_memory_pool() {
        let mut pool: MemoryPool<f64> = MemoryPool::new(100, 5);
        
        // Acquire and release blocks
        let block1 = pool.acquire().unwrap();
        let block2 = pool.acquire().unwrap();
        
        pool.release(block1).unwrap();
        pool.release(block2).unwrap();
        
        let stats = pool.stats();
        assert_eq!(stats.available_blocks, 2);
        assert_eq!(stats.block_size, 100);
    }
    
    #[test]
    fn test_conformal_data_layout() {
        let layout = ConformalDataLayout::new(1000, 5000).unwrap();
        
        assert_eq!(layout.predictions.len(), 1000);
        assert_eq!(layout.calibration_scores.len(), 5000);
        assert_eq!(layout.work_buffer.len(), 5000); // max of the two
        assert_eq!(layout.results_buffer.len(), 2000); // predictions * 2
        
        let efficiency = layout.validate_cache_efficiency();
        assert!(efficiency.cache_aligned);
    }
    
    #[test]
    fn test_memory_stats() {
        let vec: CacheAlignedVec<f64> = CacheAlignedVec::new(100).unwrap();
        let stats = vec.memory_stats();
        
        assert_eq!(stats.allocated_bytes, 100 * std::mem::size_of::<f64>());
        assert_eq!(stats.used_bytes, 100 * std::mem::size_of::<f64>());
        assert_eq!(stats.alignment, CACHE_LINE_SIZE);
        assert!(stats.cache_lines > 0);
    }
}