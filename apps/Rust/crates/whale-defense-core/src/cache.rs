//! Cache optimization utilities for whale defense
//! 
//! High-performance cache management and optimization routines.

use crate::{
    error::{WhaleDefenseError, Result},
    config::*,
};
use core::{
    mem::{align_of, size_of},
    ptr::{NonNull, write_bytes},
    slice,
    sync::atomic::{compiler_fence, Ordering},
};

/// Cache line size for optimal alignment
pub const CACHE_LINE_SIZE: usize = 64;

/// Cache warm-up buffer size
const WARMUP_BUFFER_SIZE: usize = 1024 * 1024; // 1MB

/// Warm up CPU caches for optimal performance
/// 
/// This function performs strategic memory access patterns to:
/// - Warm up L1, L2, and L3 caches
/// - Optimize TLB entries
/// - Ensure optimal cache line placement
pub fn warm_up_caches() {
    unsafe {
        // Allocate aligned buffer for cache warm-up
        let layout = core::alloc::Layout::from_size_align(
            WARMUP_BUFFER_SIZE,
            CACHE_LINE_SIZE,
        ).unwrap();
        
        #[cfg(feature = "std")]
        let buffer = {
            let ptr = std::alloc::alloc(layout) as *mut u8;
            if ptr.is_null() {
                return; // Skip warm-up if allocation fails
            }
            slice::from_raw_parts_mut(ptr, WARMUP_BUFFER_SIZE)
        };
        
        #[cfg(not(feature = "std"))]
        let buffer = {
            // In no-std environment, use static buffer
            static mut WARMUP_BUFFER: [u8; WARMUP_BUFFER_SIZE] = [0; WARMUP_BUFFER_SIZE];
            &mut WARMUP_BUFFER
        };
        
        // Sequential access pattern (good for prefetcher)
        warm_up_sequential(buffer);
        
        // Random access pattern (exercises cache hierarchy)
        warm_up_random(buffer);
        
        // Specific data structure patterns
        warm_up_market_order_pattern();
        warm_up_whale_activity_pattern();
        
        // Clean up allocated memory
        #[cfg(feature = "std")]
        {
            std::alloc::dealloc(buffer.as_mut_ptr(), layout);
        }
    }
}

/// Sequential cache warm-up pattern
unsafe fn warm_up_sequential(buffer: &mut [u8]) {
    // Write sequential pattern
    for i in (0..buffer.len()).step_by(CACHE_LINE_SIZE) {
        let cache_line = &mut buffer[i..i.saturating_add(CACHE_LINE_SIZE).min(buffer.len())];
        write_bytes(cache_line.as_mut_ptr(), 0xAA, cache_line.len());
    }
    
    // Read sequential pattern
    let mut sum = 0u64;
    for i in (0..buffer.len()).step_by(8) {
        if i + 8 <= buffer.len() {
            let value = (buffer.as_ptr().add(i) as *const u64).read();
            sum = sum.wrapping_add(value);
        }
    }
    
    // Prevent optimization
    core::hint::black_box(sum);
}

/// Random cache access pattern
unsafe fn warm_up_random(buffer: &mut [u8]) {
    // Simple linear congruential generator for deterministic randomness
    let mut seed = 0x12345678u32;
    
    for _ in 0..1000 {
        // Generate pseudo-random index
        seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let index = (seed as usize) % (buffer.len() / CACHE_LINE_SIZE);
        let offset = index * CACHE_LINE_SIZE;
        
        // Touch cache line
        if offset + CACHE_LINE_SIZE <= buffer.len() {
            let cache_line = &mut buffer[offset..offset + CACHE_LINE_SIZE];
            let value = (cache_line.as_ptr() as *const u64).read();
            (cache_line.as_mut_ptr() as *mut u64).write(value.wrapping_add(1));
        }
    }
}

/// Warm up cache for MarketOrder access patterns
unsafe fn warm_up_market_order_pattern() {
    use crate::core::MarketOrder;
    
    // Simulate typical MarketOrder operations
    let orders = [
        MarketOrder::new(100.0, 1000.0, 1, 1, 0),
        MarketOrder::new(101.0, 2000.0, 1, 2, 1),
        MarketOrder::new(99.0, 1500.0, 2, 1, 0),
        MarketOrder::new(102.0, 3000.0, 1, 3, 2),
    ];
    
    // Access patterns that mirror actual usage
    for order in &orders {
        let _ = order.total_size();
        let _ = order.impact_score();
        let _ = order.timestamp;
    }
    
    // Prevent optimization
    core::hint::black_box(&orders);
}

/// Warm up cache for WhaleActivity access patterns
unsafe fn warm_up_whale_activity_pattern() {
    use crate::core::{WhaleActivity, WhaleType, ThreatLevel};
    use crate::timing::Timestamp;
    
    // Simulate typical WhaleActivity operations
    let activities = [
        WhaleActivity {
            timestamp: Timestamp::now(),
            whale_type: WhaleType::Accumulation,
            volume: 10000.0,
            price_impact: 0.5,
            momentum: 0.3,
            confidence: 0.8,
            threat_level: ThreatLevel::High,
        },
        WhaleActivity {
            timestamp: Timestamp::now(),
            whale_type: WhaleType::Distribution,
            volume: 15000.0,
            price_impact: 0.7,
            momentum: 0.4,
            confidence: 0.9,
            threat_level: ThreatLevel::Critical,
        },
    ];
    
    // Access patterns that mirror actual usage
    for activity in &activities {
        let _ = activity.confidence;
        let _ = activity.threat_level;
        let _ = activity.volume;
    }
    
    // Prevent optimization
    core::hint::black_box(&activities);
}

/// Cache-aligned allocator for critical data structures
pub struct CacheAlignedAllocator;

impl CacheAlignedAllocator {
    /// Allocate cache-aligned memory
    /// 
    /// # Safety
    /// Caller must ensure proper deallocation and size alignment.
    pub unsafe fn allocate<T>(count: usize) -> Result<NonNull<T>> {
        let size = count * size_of::<T>();
        let align = CACHE_LINE_SIZE.max(align_of::<T>());
        
        let layout = core::alloc::Layout::from_size_align(size, align)
            .map_err(|_| WhaleDefenseError::OutOfMemory)?;
        
        #[cfg(feature = "std")]
        {
            let ptr = std::alloc::alloc(layout) as *mut T;
            NonNull::new(ptr).ok_or(WhaleDefenseError::OutOfMemory)
        }
        
        #[cfg(not(feature = "std"))]
        {
            Err(WhaleDefenseError::OutOfMemory) // No allocator in no-std
        }
    }
    
    /// Deallocate cache-aligned memory
    /// 
    /// # Safety
    /// Pointer must have been allocated with `allocate`.
    pub unsafe fn deallocate<T>(ptr: NonNull<T>, count: usize) {
        let size = count * size_of::<T>();
        let align = CACHE_LINE_SIZE.max(align_of::<T>());
        
        let layout = core::alloc::Layout::from_size_align(size, align).unwrap();
        
        #[cfg(feature = "std")]
        {
            std::alloc::dealloc(ptr.as_ptr() as *mut u8, layout);
        }
    }
}

/// Prefetch data into cache for faster access
/// 
/// Uses CPU prefetch instructions to bring data into cache hierarchy.
#[inline(always)]
pub fn prefetch_data<T>(ptr: *const T, locality: Locality) {
    unsafe {
        match locality {
            Locality::None => {
                // Prefetch for all cache levels
                core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
            Locality::Low => {
                // Prefetch to L3 cache
                core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T2);
            }
            Locality::Medium => {
                // Prefetch to L2 cache
                core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T1);
            }
            Locality::High => {
                // Prefetch to L1 cache
                core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
        }
    }
}

/// Cache locality hint for prefetching
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Locality {
    /// No temporal locality (prefetch and evict)
    None,
    /// Low temporal locality (L3 cache)
    Low,
    /// Medium temporal locality (L2 cache)
    Medium,
    /// High temporal locality (L1 cache)
    High,
}

/// Flush cache lines to ensure memory coherency
/// 
/// Forces write-back of modified cache lines to memory.
#[inline(always)]
pub fn flush_cache_line<T>(ptr: *const T) {
    unsafe {
        core::arch::x86_64::_mm_clflush(ptr as *const u8);
    }
}

/// Memory fence for cache coherency
/// 
/// Ensures all memory operations are complete before proceeding.
#[inline(always)]
pub fn memory_fence() {
    unsafe {
        core::arch::x86_64::_mm_mfence();
    }
}

/// Cache-optimized memcpy for small buffers
/// 
/// Optimized for typical whale defense data sizes (<1KB).
#[inline(always)]
pub unsafe fn cache_optimized_memcpy(dst: *mut u8, src: *const u8, len: usize) {
    if len == 0 {
        return;
    }
    
    // For small sizes, use regular copy
    if len <= 64 {
        core::ptr::copy_nonoverlapping(src, dst, len);
        return;
    }
    
    // For larger sizes, use cache-optimized approach
    let mut copied = 0;
    
    // Copy cache line aligned chunks
    while copied + CACHE_LINE_SIZE <= len {
        // Prefetch next cache line
        if copied + CACHE_LINE_SIZE * 2 <= len {
            prefetch_data(src.add(copied + CACHE_LINE_SIZE), Locality::High);
        }
        
        // Copy current cache line
        core::ptr::copy_nonoverlapping(
            src.add(copied),
            dst.add(copied),
            CACHE_LINE_SIZE,
        );
        
        copied += CACHE_LINE_SIZE;
    }
    
    // Copy remaining bytes
    if copied < len {
        core::ptr::copy_nonoverlapping(
            src.add(copied),
            dst.add(copied),
            len - copied,
        );
    }
}

/// Cache-aligned buffer for high-performance operations
#[repr(C, align(64))]
pub struct CacheAlignedBuffer<T, const N: usize> {
    data: [T; N],
}

impl<T, const N: usize> CacheAlignedBuffer<T, N> {
    /// Create new cache-aligned buffer
    pub const fn new(data: [T; N]) -> Self {
        Self { data }
    }
    
    /// Get reference to data
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }
    
    /// Get mutable reference to data
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
    
    /// Get raw pointer to data
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }
    
    /// Get mutable raw pointer to data
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }
}

/// Cache performance counters (if available)
pub struct CacheCounters {
    l1_misses: u64,
    l2_misses: u64,
    l3_misses: u64,
    tlb_misses: u64,
}

impl CacheCounters {
    /// Create new cache counters
    pub fn new() -> Self {
        Self {
            l1_misses: 0,
            l2_misses: 0,
            l3_misses: 0,
            tlb_misses: 0,
        }
    }
    
    /// Read cache performance counters (platform-specific)
    pub fn read_counters(&mut self) -> Result<()> {
        // This would interface with platform-specific performance counters
        // For now, return success without actual counter reading
        Ok(())
    }
    
    /// Get L1 cache miss rate
    pub fn l1_miss_rate(&self) -> f64 {
        // Simplified calculation
        if self.l1_misses > 0 {
            self.l1_misses as f64 / (self.l1_misses + 1000) as f64
        } else {
            0.0
        }
    }
    
    /// Get L2 cache miss rate
    pub fn l2_miss_rate(&self) -> f64 {
        if self.l2_misses > 0 {
            self.l2_misses as f64 / (self.l2_misses + 100) as f64
        } else {
            0.0
        }
    }
    
    /// Get L3 cache miss rate
    pub fn l3_miss_rate(&self) -> f64 {
        if self.l3_misses > 0 {
            self.l3_misses as f64 / (self.l3_misses + 10) as f64
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cache_warmup() {
        // Should not panic
        warm_up_caches();
    }
    
    #[test]
    fn test_cache_aligned_buffer() {
        let buffer = CacheAlignedBuffer::new([1u64, 2, 3, 4]);
        assert_eq!(buffer.as_slice(), &[1, 2, 3, 4]);
        
        // Check alignment
        let ptr = buffer.as_ptr() as usize;
        assert_eq!(ptr % CACHE_LINE_SIZE, 0);
    }
    
    #[test]
    fn test_prefetch() {
        let data = [1u64, 2, 3, 4];
        prefetch_data(data.as_ptr(), Locality::High);
        prefetch_data(data.as_ptr(), Locality::Medium);
        prefetch_data(data.as_ptr(), Locality::Low);
        prefetch_data(data.as_ptr(), Locality::None);
    }
    
    #[test]
    fn test_cache_optimized_memcpy() {
        let src = [1u8; 128];
        let mut dst = [0u8; 128];
        
        unsafe {
            cache_optimized_memcpy(dst.as_mut_ptr(), src.as_ptr(), 128);
        }
        
        assert_eq!(src, dst);
    }
    
    #[test]
    fn test_cache_counters() {
        let mut counters = CacheCounters::new();
        assert!(counters.read_counters().is_ok());
        
        // Initial miss rates should be 0
        assert_eq!(counters.l1_miss_rate(), 0.0);
        assert_eq!(counters.l2_miss_rate(), 0.0);
        assert_eq!(counters.l3_miss_rate(), 0.0);
    }
}