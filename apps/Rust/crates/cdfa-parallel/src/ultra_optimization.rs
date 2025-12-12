//! Ultra-performance optimizations for sub-microsecond latency
//! 
//! This module contains the most aggressive optimizations for critical performance paths.
//! Designed by the Hive-Mind Parallel Processing Expert.

use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use std::ptr;

/// Ultra-fast correlation calculation using manual SIMD
/// 
/// Performance target: <10ns for 64-element vectors on AVX-512
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn correlation_avx2_manual(x: &[f64], y: &[f64]) -> f64 {
    debug_assert_eq!(x.len(), y.len());
    debug_assert!(x.len() >= 4);
    
    let n = x.len();
    let n_f64 = n as f64;
    
    // Initialize SIMD accumulators
    let mut sum_x = _mm256_setzero_pd();
    let mut sum_y = _mm256_setzero_pd();
    let mut sum_xx = _mm256_setzero_pd();
    let mut sum_yy = _mm256_setzero_pd();
    let mut sum_xy = _mm256_setzero_pd();
    
    // Process 4 elements at a time (AVX2)
    let chunks = n / 4;
    for i in 0..chunks {
        let offset = i * 4;
        
        // Load 4 doubles
        let x_vec = _mm256_loadu_pd(x.as_ptr().add(offset));
        let y_vec = _mm256_loadu_pd(y.as_ptr().add(offset));
        
        // Accumulate sums
        sum_x = _mm256_add_pd(sum_x, x_vec);
        sum_y = _mm256_add_pd(sum_y, y_vec);
        
        // Accumulate squares using FMA
        sum_xx = _mm256_fmadd_pd(x_vec, x_vec, sum_xx);
        sum_yy = _mm256_fmadd_pd(y_vec, y_vec, sum_yy);
        
        // Accumulate cross products using FMA
        sum_xy = _mm256_fmadd_pd(x_vec, y_vec, sum_xy);
    }
    
    // Horizontal sum of SIMD registers
    let sum_x_scalar = horizontal_sum_pd(sum_x);
    let sum_y_scalar = horizontal_sum_pd(sum_y);
    let sum_xx_scalar = horizontal_sum_pd(sum_xx);
    let sum_yy_scalar = horizontal_sum_pd(sum_yy);
    let sum_xy_scalar = horizontal_sum_pd(sum_xy);
    
    // Handle remaining elements
    let mut sum_x_remainder = 0.0;
    let mut sum_y_remainder = 0.0;
    let mut sum_xx_remainder = 0.0;
    let mut sum_yy_remainder = 0.0;
    let mut sum_xy_remainder = 0.0;
    
    for i in (chunks * 4)..n {
        let x_val = x[i];
        let y_val = y[i];
        sum_x_remainder += x_val;
        sum_y_remainder += y_val;
        sum_xx_remainder += x_val * x_val;
        sum_yy_remainder += y_val * y_val;
        sum_xy_remainder += x_val * y_val;
    }
    
    // Combine SIMD and scalar results
    let final_sum_x = sum_x_scalar + sum_x_remainder;
    let final_sum_y = sum_y_scalar + sum_y_remainder;
    let final_sum_xx = sum_xx_scalar + sum_xx_remainder;
    let final_sum_yy = sum_yy_scalar + sum_yy_remainder;
    let final_sum_xy = sum_xy_scalar + sum_xy_remainder;
    
    // Compute correlation coefficient
    let numerator = n_f64 * final_sum_xy - final_sum_x * final_sum_y;
    let denominator_x = n_f64 * final_sum_xx - final_sum_x * final_sum_x;
    let denominator_y = n_f64 * final_sum_yy - final_sum_y * final_sum_y;
    let denominator = (denominator_x * denominator_y).sqrt();
    
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

/// Horizontal sum of 4 packed doubles
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_pd(v: __m256d) -> f64 {
    // Sum high and low 128-bit lanes
    let high = _mm256_extractf128_pd(v, 1);
    let low = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(high, low);
    
    // Sum the two doubles in the 128-bit register
    let high64 = _mm_unpackhi_pd(sum128, sum128);
    let sum64 = _mm_add_sd(sum128, high64);
    
    _mm_cvtsd_f64(sum64)
}

/// Lock-free ring buffer optimized for single-producer, single-consumer
/// 
/// Performance target: <1ns push/pop operations
#[repr(align(64))] // Align to cache line
pub struct UltraFastRingBuffer<T> {
    buffer: *mut MaybeUninit<T>,
    capacity: usize,
    capacity_mask: usize, // capacity - 1 (capacity must be power of 2)
    
    // Separate cache lines for producer and consumer
    #[repr(align(64))]
    producer: ProducerData,
    
    #[repr(align(64))]
    consumer: ConsumerData,
}

#[repr(align(64))]
struct ProducerData {
    head: std::sync::atomic::AtomicUsize,
}

#[repr(align(64))]
struct ConsumerData {
    tail: std::sync::atomic::AtomicUsize,
}

impl<T> UltraFastRingBuffer<T> {
    /// Creates a new ultra-fast ring buffer
    /// 
    /// Capacity must be a power of 2 for optimal performance
    pub fn new(capacity: usize) -> Self {
        assert!(capacity.is_power_of_two(), "Capacity must be power of 2");
        assert!(capacity > 0);
        
        let layout = std::alloc::Layout::array::<MaybeUninit<T>>(capacity).unwrap();
        let buffer = unsafe {
            std::alloc::alloc(layout) as *mut MaybeUninit<T>
        };
        
        if buffer.is_null() {
            std::alloc::handle_alloc_error(layout);
        }
        
        Self {
            buffer,
            capacity,
            capacity_mask: capacity - 1,
            producer: ProducerData {
                head: std::sync::atomic::AtomicUsize::new(0),
            },
            consumer: ConsumerData {
                tail: std::sync::atomic::AtomicUsize::new(0),
            },
        }
    }
    
    /// Push an element (wait-free for single producer)
    #[inline(always)]
    pub fn push(&self, item: T) -> Result<(), T> {
        let head = self.producer.head.load(std::sync::atomic::Ordering::Relaxed);
        let next_head = head.wrapping_add(1);
        let tail = self.consumer.tail.load(std::sync::atomic::Ordering::Acquire);
        
        // Check if buffer is full
        if next_head.wrapping_sub(tail) > self.capacity {
            return Err(item);
        }
        
        // Write data
        unsafe {
            ptr::write(
                self.buffer.add(head & self.capacity_mask),
                MaybeUninit::new(item)
            );
        }
        
        // Update head with release ordering
        self.producer.head.store(next_head, std::sync::atomic::Ordering::Release);
        
        Ok(())
    }
    
    /// Pop an element (wait-free for single consumer)
    #[inline(always)]
    pub fn pop(&self) -> Option<T> {
        let tail = self.consumer.tail.load(std::sync::atomic::Ordering::Relaxed);
        let head = self.producer.head.load(std::sync::atomic::Ordering::Acquire);
        
        // Check if buffer is empty
        if tail == head {
            return None;
        }
        
        // Read data
        let item = unsafe {
            ptr::read(self.buffer.add(tail & self.capacity_mask))
                .assume_init()
        };
        
        // Update tail with release ordering
        let next_tail = tail.wrapping_add(1);
        self.consumer.tail.store(next_tail, std::sync::atomic::Ordering::Release);
        
        Some(item)
    }
    
    /// Get current length (approximate)
    #[inline(always)]
    pub fn len(&self) -> usize {
        let head = self.producer.head.load(std::sync::atomic::Ordering::Relaxed);
        let tail = self.consumer.tail.load(std::sync::atomic::Ordering::Relaxed);
        head.wrapping_sub(tail)
    }
    
    /// Check if empty (approximate)
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Drop for UltraFastRingBuffer<T> {
    fn drop(&mut self) {
        // Drop any remaining items
        while self.pop().is_some() {}
        
        // Deallocate buffer
        let layout = std::alloc::Layout::array::<MaybeUninit<T>>(self.capacity).unwrap();
        unsafe {
            std::alloc::dealloc(self.buffer as *mut u8, layout);
        }
    }
}

unsafe impl<T: Send> Send for UltraFastRingBuffer<T> {}
unsafe impl<T: Send> Sync for UltraFastRingBuffer<T> {}

/// Memory prefetching utilities for cache optimization
pub mod prefetch {
    use std::arch::x86_64::*;
    
    /// Prefetch data into L1 cache for reading
    #[inline(always)]
    pub fn prefetch_read_l1(ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }
    
    /// Prefetch data into L2 cache for reading
    #[inline(always)]
    pub fn prefetch_read_l2(ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
        }
    }
    
    /// Prefetch data into L3 cache for reading
    #[inline(always)]
    pub fn prefetch_read_l3(ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T2);
        }
    }
    
    /// Non-temporal prefetch (bypass cache)
    #[inline(always)]
    pub fn prefetch_non_temporal(ptr: *const u8) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_NTA);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ultra_fast_ring_buffer() {
        let buffer = UltraFastRingBuffer::new(16);
        
        // Test push and pop
        assert!(buffer.push(42).is_ok());
        assert!(buffer.push(43).is_ok());
        assert_eq!(buffer.len(), 2);
        
        assert_eq!(buffer.pop(), Some(42));
        assert_eq!(buffer.pop(), Some(43));
        assert_eq!(buffer.pop(), None);
        
        assert!(buffer.is_empty());
    }
    
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_correlation_avx2_manual() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];
        
        let correlation = unsafe { correlation_avx2_manual(&x, &y) };
        
        // Should be perfect correlation (1.0)
        assert!((correlation - 1.0).abs() < 1e-10);
    }
}